//! AAC-LC decode (symphonia) from demuxed packets, and playback through cpal. The audio clock is the master clock:
//! a decode thread fills a lock-free single-producer / single-consumer ring, the device callback drains it and
//! counts the frames it consumed. Without sound (no AAC track, no device) [`Player::silent`] runs the same clock on
//! wall time. The stream may run at another rate and channel count than the source ([`Conv`]): the clock still counts
//! source frames.

use crate::{
    mp4::{Codec, Mp4, asc_info},
    resample::Resampler,
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    io,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::*},
    },
    thread,
    time::{Duration, Instant},
};
use symphonia::{
    core::{
        audio::SampleBuffer,
        codecs::{CODEC_TYPE_AAC, CodecParameters, Decoder, DecoderOptions},
        formats::Packet,
    },
    default::codecs::AacDecoder,
};

fn err(e: impl std::error::Error + Send + Sync + 'static) -> io::Error {
    io::Error::other(e)
}

pub struct Aac {
    dec: AacDecoder,
    out: Option<SampleBuffer<f32>>,
    pub rate: u32,
    pub ch: u16,
}

impl Aac {
    pub fn new(asc: &[u8]) -> io::Result<Aac> {
        let (rate, ch) = asc_info(asc)?;
        let mut p = CodecParameters::new();
        p.for_codec(CODEC_TYPE_AAC).with_extra_data(asc.into());
        Ok(Aac {
            dec: AacDecoder::try_new(&p, &DecoderOptions::default()).map_err(err)?,
            out: None,
            rate,
            ch,
        })
    }

    /// Decodes one raw AAC frame to interleaved f32 (1024 frames for AAC-LC). An error concerns this packet only.
    pub fn decode(&mut self, pkt: &[u8]) -> io::Result<&[f32]> {
        // symphonia 0.5 takes packets by boxed slice: one small allocation per frame
        let d = self
            .dec
            .decode(&Packet::new_from_slice(0, 0, 0, pkt))
            .map_err(err)?;
        let o = self
            .out
            .get_or_insert_with(|| SampleBuffer::new(d.capacity() as u64, *d.spec()));
        o.copy_interleaved_ref(d);
        Ok(o.samples())
    }

    /// Forgets the overlap state, before decoding from a new position.
    pub fn reset(&mut self) {
        self.dec.reset()
    }
}

/// Interleaved audio of a source's rate and channels to the format a stream runs at ([`Conv::on`]): resampled
/// (stream frame n is at source time n / `rate` s exactly), then mono on both sides or the two sides mixed to mono.
pub struct Conv {
    rs: Resampler,
    from: u32,
    /// The stream's rate and channels.
    pub rate: u32,
    pub ch: u16,
    tmp: Vec<f32>,
}

impl Conv {
    /// `ch` to `och` channels: the same, or one to two and back.
    pub fn new(rate: u32, ch: u16, out: u32, och: u16) -> io::Result<Conv> {
        if ch != och && (ch.min(och), ch.max(och)) != (1, 2) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("{ch} channels to {och}"),
            ));
        }
        Ok(Conv {
            rs: Resampler::new(rate, out, ch as usize)?,
            from: rate,
            rate: out,
            ch: och,
            tmp: vec![],
        })
    }

    /// The format a stream on `dev` runs at for audio of `rate` Hz and `ch` channels. macOS: the device's own rate
    /// and up to two of its channels (cpal's CoreAudio backend sets the device's hardware rate to the one asked for,
    /// for every app, and leaves it there; it fails on a rate the device lacks), else the source's. Elsewhere the
    /// source's: ALSA's plugins and PipeWire convert.
    pub fn on(dev: &cpal::Device, rate: u32, ch: u16) -> io::Result<Conv> {
        #[cfg(target_os = "macos")]
        if let Some(c) = (dev.default_output_config().ok())
            .and_then(|c| Conv::new(rate, ch, c.sample_rate(), c.channels().clamp(1, 2)).ok())
        {
            return Ok(c);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = dev;
        Conv::new(rate, ch, rate, ch)
    }

    /// The most samples [`Conv::run`] appends for `n` source frames.
    pub fn most(&self, n: usize) -> usize {
        (n as u64 * self.rate as u64).div_ceil(self.from as u64) as usize * self.ch as usize
    }

    /// Appends the stream's samples for source samples `x`.
    pub fn run(&mut self, x: &[f32], out: &mut Vec<f32>) {
        if self.rs.ch == self.ch as usize {
            return self.rs.run(x, out);
        }
        self.tmp.clear();
        self.rs.run(x, &mut self.tmp);
        self.map(out)
    }

    /// Ends the stream: appends the resampler's last frames.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        if self.rs.ch == self.ch as usize {
            return self.rs.finish(out);
        }
        self.tmp.clear();
        self.rs.finish(&mut self.tmp);
        self.map(out)
    }

    /// Forgets the stream (seek): the next sample is time 0 again.
    pub fn reset(&mut self) {
        self.rs.reset()
    }

    fn map(&self, out: &mut Vec<f32>) {
        match self.ch {
            1 => out.extend(self.tmp.chunks_exact(2).map(|f| 0.5 * (f[0] + f[1]))),
            _ => out.extend(self.tmp.iter().flat_map(|&v| [v, v])),
        }
    }
}

const RING: usize = 1 << 16; // f32 samples: 0.68 s of 48 kHz stereo

struct Shared {
    ring: Box<[AtomicU32]>,
    /// Samples written / read since start; only the producer moves `w`, only the callback `r`.
    w: AtomicUsize,
    r: AtomicUsize,
    /// The source's rate (the clock's and seeks' frames); the stream's rate and channels (the ring's frames).
    rate: u32,
    out: u32,
    ch: usize,
    /// Source frame being heard now.
    clock: AtomicU64,
    /// Seek request (UI): epoch and target frame.
    req: AtomicU32,
    target: AtomicU64,
    /// Producer: the epoch it stopped for, its `w` at that moment (all older data is below it), its restart frame.
    seen: AtomicU32,
    fence: AtomicUsize,
    base: AtomicU64,
    /// Callback: the epoch whose old data it dropped. The producer writes nothing between `seen` and `ack`.
    ack: AtomicU32,
    eof: AtomicBool,
    paused: AtomicBool,
    quit: AtomicBool,
    vol: AtomicU32,
}

impl Shared {
    /// Source frames at `rate`, a stream at `out` Hz with `ch` channels.
    fn new(rate: u32, out: u32, ch: u16) -> Arc<Shared> {
        // 0.68 s at the stream's format, at least
        let k = (out as usize * ch as usize).div_ceil(96000);
        let n = RING * k.next_power_of_two();
        Arc::new(Shared {
            ring: (0..n).map(|_| AtomicU32::new(0)).collect(),
            w: AtomicUsize::new(0),
            r: AtomicUsize::new(0),
            rate,
            out,
            ch: ch as usize,
            clock: AtomicU64::new(0),
            req: AtomicU32::new(0),
            target: AtomicU64::new(0),
            seen: AtomicU32::new(0),
            fence: AtomicUsize::new(0),
            base: AtomicU64::new(0),
            ack: AtomicU32::new(0),
            eof: AtomicBool::new(false),
            paused: AtomicBool::new(true),
            quit: AtomicBool::new(false),
            vol: AtomicU32::new(1f32.to_bits()),
        })
    }
}

const HIST: usize = 32;

/// Device-callback state, in stream frames from the last seek target `base` (a source frame; the clock never reads
/// earlier): `d` = the next sample handed to the device, `ack` = last flushed epoch, `sent` = frames handed to the
/// device. `hist` = the last device buffers: (`sent` before it, `d` of its first sample, frames in it). The sample
/// heard now is looked up there by its place in the hand-over stream, so queued silence (pause, underrun) holds the
/// clock only while it plays.
#[derive(Default)]
struct Play {
    base: u64,
    d: u64,
    ack: u32,
    sent: u64,
    hist: [(u64, u64, u64); HIST],
    n: usize,
}

impl Play {
    /// Source frame of stream frame `d`: the resampler's output n is at source time n / `out` s.
    fn src(&self, s: &Shared, d: u64) -> u64 {
        self.base + d * s.rate as u64 / s.out as u64
    }

    /// Fills one device buffer; `queued` = seconds until its first sample is heard.
    fn fill(&mut self, s: &Shared, out: &mut [f32], queued: f64) {
        let e = s.seen.load(Acquire);
        if e != self.ack {
            s.r.store(s.fence.load(Relaxed), Relaxed);
            (self.base, self.d, self.ack) = (s.base.load(Relaxed), 0, e);
            // audio from before the seek still in the device reads as the seek target
            self.hist.iter_mut().for_each(|b| (b.1, b.2) = (0, 0));
        }
        let start = self.d;
        if s.paused.load(Relaxed) {
            out.fill(0.0);
        } else {
            let (eof, n) = (s.eof.load(Acquire), s.ring.len());
            let (r, w) = (s.r.load(Relaxed), s.w.load(Acquire));
            let k = (w - r).min(out.len()) / s.ch * s.ch;
            let v = f32::from_bits(s.vol.load(Relaxed));
            for (i, o) in out[..k].iter_mut().enumerate() {
                *o = f32::from_bits(s.ring[(r + i) % n].load(Relaxed)) * v;
            }
            out[k..].fill(0.0);
            s.r.store(r + k, Release);
            // past the end the clock runs on silence; an underrun mid-stream holds it
            self.d += (if eof && r + k == w { out.len() } else { k } / s.ch) as u64;
        }
        self.hist[self.n % HIST] = (self.sent, start, self.d - start);
        self.n += 1;
        // the sample heard now was handed over `queued` before this buffer's first one
        let h = self.sent.saturating_sub((queued * s.out as f64) as u64);
        self.sent += (out.len() / s.ch) as u64;
        let mut c = 0;
        for i in 1..=self.n.min(HIST) {
            let b = self.hist[(self.n - i) % HIST];
            c = b.1 + h.saturating_sub(b.0).min(b.2);
            if b.0 <= h {
                break; // else: the device holds more than the history, read its oldest buffer
            }
        }
        s.clock.store(self.src(s, c), Relaxed);
        // after the clock: a reader that sees the ack sees the clock of the seek it acknowledges (`Clock::get`)
        s.ack.store(e, Release);
    }
}

/// Seconds of source audio heard so far: frames the device consumed, minus what is still queued in it.
#[derive(Clone)]
pub struct Clock(Arc<Shared>);

impl Clock {
    pub fn secs(&self) -> f64 {
        self.0.clock.load(Relaxed) as f64 / self.0.rate as f64
    }

    /// `secs`, or None while the last `Player::seek` has not reached the clock yet (a few ms): until then it still
    /// reads the old position. Call it on the thread that seeks.
    pub fn get(&self) -> Option<f64> {
        (self.0.ack.load(Acquire) == self.0.req.load(Relaxed)).then(|| self.secs())
    }
}

/// The output device with this id (`cpal::DeviceId` as text, as the app's settings keep it from
/// `output_devices`), else the system default. An id lists the devices (ALSA: every hint and card): resolve it when
/// the choice changes and keep the device, never per open. macOS: the device kept names it while it is connected
/// (CoreAudio's object id); the app's settings look a gone one up again.
pub fn device(id: Option<&str>) -> Option<cpal::Device> {
    let h = cpal::default_host();
    // an exact match: cpal 0.18's device_by_id rewrites ALSA's `sysdefault:CARD=x` to `...,DEV=0`, which no listed
    // device has
    id.and_then(|s| {
        let id = Some(s.parse().ok()?);
        h.output_devices().ok()?.find(|d| d.id().ok() == id)
    })
    .or_else(|| h.default_output_device())
}

/// Plays the first AAC track of an MP4 on an output device ([`Player::open`]: the default). Starts paused.
pub struct Player {
    sh: Arc<Shared>,
    th: Option<thread::JoinHandle<()>>,
    _stream: Option<cpal::Stream>,
    pub duration: f64,
}

impl Player {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Player> {
        Player::open_on(path, None)
    }

    /// [`Player::open`] on output device `dev` (from [`device`]; None: the default), in the format [`Conv::on`] picks.
    /// A stream that fails to build or start comes back as an `io::Error` holding the `cpal::Error` (`get_ref`);
    /// file, track, decoder and missing device errors do not (the app's `player::lost` tells them apart by it).
    pub fn open_on(path: impl AsRef<Path>, dev: Option<cpal::Device>) -> io::Result<Player> {
        let mp4 = Mp4::open(path)?;
        let t = mp4
            .tracks
            .iter()
            .position(|t| matches!(t.codec, Codec::Aac { .. }))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no AAC track"))?;
        let Codec::Aac { asc, .. } = &mp4.tracks[t].codec else {
            unreachable!()
        };
        let aac = Aac::new(asc)?;
        let duration = mp4.tracks[t].secs();
        let dev = (dev.or_else(|| cpal::default_host().default_output_device()))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no audio output device"))?;
        let cv = Conv::on(&dev, aac.rate, aac.ch)?;
        let sh = Shared::new(aac.rate, cv.rate, cv.ch);
        let cfg = cpal::StreamConfig {
            channels: cv.ch,
            sample_rate: cv.rate,
            buffer_size: cpal::BufferSize::Default,
        };
        let (s, mut p) = (sh.clone(), Play::default());
        let cb = move |out: &mut [f32], info: &cpal::OutputCallbackInfo| {
            let ts = info.timestamp();
            p.fill(
                &s,
                out,
                ts.playback.duration_since(ts.callback).as_secs_f64(),
            );
        };
        let stream = dev
            .build_output_stream(cfg, cb, |e| eprintln!("audio output: {e}"), None)
            .map_err(err)?;
        stream.play().map_err(err)?;
        let s = sh.clone();
        let th = thread::Builder::new()
            .name("audio decode".into())
            .spawn(move || feed(&s, &mp4, t, aac, cv))?;
        Ok(Player {
            sh,
            th: Some(th),
            _stream: Some(stream),
            duration,
        })
    }

    /// The same controls and clock without sound, for files without audio or without an output device: a thread
    /// stands in for the device and feeds it silence at wall-clock pace.
    pub fn silent(duration: f64) -> Player {
        let sh = Shared::new(48000, 48000, 1);
        sh.eof.store(true, Relaxed); // an empty ring at the end: the clock runs on silence
        let s = sh.clone();
        let th = thread::Builder::new()
            .name("silent clock".into())
            .spawn(move || tick(&s))
            .ok();
        Player {
            sh,
            th,
            _stream: None,
            duration,
        }
    }

    pub fn play(&self) {
        self.sh.paused.store(false, Relaxed)
    }

    pub fn pause(&self) {
        self.sh.paused.store(true, Relaxed)
    }

    pub fn seek(&self, secs: f64) {
        self.sh.target.store(
            (secs.max(0.0) * self.sh.rate as f64).round() as u64,
            Relaxed,
        );
        self.sh.req.fetch_add(1, Release);
    }

    pub fn volume(&self, v: f32) {
        self.sh.vol.store(v.to_bits(), Relaxed)
    }

    pub fn clock(&self) -> Clock {
        Clock(self.sh.clone())
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.sh.quit.store(true, Relaxed);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

/// Decode thread: keeps the ring full, restarts on seek requests. A packet goes through `cv` and waits for room.
fn feed(s: &Shared, mp4: &Mp4, t: usize, mut aac: Aac, mut cv: Conv) {
    let (tr, len) = (&mp4.tracks[t], s.ring.len());
    let (ts, rate, ch) = (tr.timescale as u64, s.rate as u64, aac.ch as usize);
    let (mut i, mut skip, mut cur, mut buf) = (0, 0u64, 0u32, vec![]);
    // the packet for the ring; whether the resampler's last frames followed the last one
    let (mut o, mut tail) = (vec![], false);
    let zero = vec![0f32; 2048 * ch]; // the most one packet holds
    let nap = |ms| thread::sleep(Duration::from_millis(ms));
    while !s.quit.load(Relaxed) {
        let e = s.req.load(Acquire);
        if e != cur {
            let f = s.target.load(Relaxed);
            // restart one packet before the one holding frame f: the decoder needs the previous MDCT half
            i = tr.at((f * ts / rate) as i64).saturating_sub(1);
            skip = f.saturating_sub(
                tr.samples
                    .get(i)
                    .map_or(0, |x| x.dts.max(0) as u64 * rate / ts),
            );
            s.eof.store(false, Relaxed);
            s.base.store(f, Relaxed);
            s.fence.store(s.w.load(Relaxed), Relaxed);
            s.seen.store(e, Release);
            while s.ack.load(Acquire) != e && !s.quit.load(Relaxed) {
                nap(1);
            }
            aac.reset();
            cv.reset();
            (cur, tail) = (e, false);
            o.clear();
            continue;
        }
        if o.is_empty() {
            if i >= tr.samples.len() {
                if !tail {
                    cv.finish(&mut o);
                    tail = true;
                    continue;
                }
                s.eof.store(true, Release);
                nap(10);
                continue;
            }
            i += 1;
            let x = tr.samples[i - 1];
            // a packet that fails to read or decode becomes silence of its duration: the clock keeps source time
            let n = ((x.dur as u64 * rate / ts) as usize * ch).min(zero.len());
            let pcm = match mp4.read(&x, &mut buf) {
                Ok(()) => aac.decode(&buf).unwrap_or(&zero[..n]),
                Err(_) => &zero[..n],
            };
            let d = (skip as usize * ch).min(pcm.len());
            skip -= (d / ch) as u64;
            cv.run(&pcm[d..], &mut o);
        }
        let w = s.w.load(Relaxed);
        if len - (w - s.r.load(Acquire)) < o.len() {
            nap(5);
            continue;
        }
        for (k, x) in o.iter().enumerate() {
            s.ring[(w + k) % len].store(x.to_bits(), Relaxed);
        }
        s.w.store(w + o.len(), Release);
        o.clear();
    }
}

/// The silent player's device: takes seek requests (the decode thread's part) and fills silence for the wall time
/// that passed (the callback's part).
fn tick(s: &Shared) {
    let (mut p, mut out, mut cur) = (Play::default(), [0f32; 960], 0);
    let (t0, mut done) = (Instant::now(), 0u64);
    while !s.quit.load(Relaxed) {
        let e = s.req.load(Acquire);
        if e != cur {
            s.base.store(s.target.load(Relaxed), Relaxed);
            s.seen.store(e, Release);
            cur = e;
        }
        let due = (t0.elapsed().as_secs_f64() * s.out as f64) as u64;
        while done < due {
            let n = ((due - done) as usize).min(out.len());
            p.fill(s, &mut out[..n], 0.0);
            done += n as u64;
        }
        thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{fixtures, samples};
    use std::time::Instant;

    fn snr(r: &[f32], x: &[f32]) -> f64 {
        let (s, n) = r.iter().zip(x).fold((0f64, 0f64), |(s, n), (&a, &b)| {
            (s + (a as f64).powi(2), n + (a as f64 - b as f64).powi(2))
        });
        10.0 * (s / n.max(1e-300)).log10()
    }

    /// Whole AAC track through our demuxer and decoder, and FFmpeg's decode of the same file.
    fn decode_both(name: &str) -> (Vec<f32>, Vec<f32>) {
        let p = samples().unwrap().join(name);
        let m = Mp4::open(&p).unwrap();
        let t = m
            .tracks
            .iter()
            .find(|t| matches!(t.codec, Codec::Aac { .. }))
            .unwrap();
        let Codec::Aac { asc, .. } = &t.codec else {
            unreachable!()
        };
        let mut a = Aac::new(asc).unwrap();
        let (mut ours, mut b) = (vec![], vec![]);
        let t0 = Instant::now();
        for s in &t.samples {
            m.read(s, &mut b).unwrap();
            ours.extend_from_slice(a.decode(&b).unwrap());
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        // through a pipe: tests decode the same file at the same time, a shared output file raced
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i", p.to_str().unwrap()])
            .args(["-map", "0:a:0", "-f", "f32le", "-"])
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let r: Vec<f32> = o
            .stdout
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        println!(
            "{name}: {} samples ({} ch, {} Hz) in {ms:.1} ms, ffmpeg {} samples",
            ours.len(),
            a.ch,
            a.rate,
            r.len()
        );
        (ours, r)
    }

    #[test]
    fn aac_matches_ffmpeg() {
        if samples().is_none() {
            return;
        }
        for name in ["v635.insv", "v635.lrv", "v627.insv"] {
            let (ours, r) = decode_both(name);
            assert_eq!(ours.len(), r.len(), "{name}");
            let s = snr(&r, &ours);
            println!("{name}: SNR {s:.1} dB");
            assert!(s > 120.0, "{name}");
        }
    }

    /// The decode thread and the device-callback logic, without a device: seek, sample-accurate restart, clock,
    /// running past the end.
    #[test]
    fn ring_seek_and_clock() {
        let Some(d) = samples() else { return };
        let (full, _) = decode_both("v635.insv");
        let m = Mp4::open(d.join("v635.insv")).unwrap();
        let Codec::Aac { asc, .. } = &m.tracks[2].codec else {
            panic!()
        };
        let sh = Shared::new(48000, 48000, 2);
        sh.paused.store(false, Relaxed);
        let (s, a) = (sh.clone(), Aac::new(asc).unwrap());
        let cv = Conv::new(48000, 2, 48000, 2).unwrap();
        let th = thread::spawn(move || feed(&s, &m, 2, a, cv));
        let mut p = Play::default();
        let mut out = vec![0f32; 960];
        let mut epoch = 0;
        for (secs, want) in [(2.0, 48000), (0.5, 24000), (3.1234, 30000), (5.7, 4000)] {
            let f = (secs * 48000f64).round() as u64;
            sh.target.store(f, Relaxed);
            sh.req.fetch_add(1, Release);
            epoch += 1;
            while p.ack != epoch {
                p.fill(&sh, &mut out, 0.0);
            }
            let mut got = vec![];
            let end = full.len() / 2;
            let want = want.min(end - f as usize);
            while got.len() < want * 2 {
                let t0 = Instant::now();
                while sh.w.load(Acquire) - sh.r.load(Acquire) < out.len() && !sh.eof.load(Acquire) {
                    assert!(t0.elapsed().as_secs() < 5, "decoder stalled");
                    thread::yield_now();
                }
                let n = (sh.w.load(Acquire) - sh.r.load(Acquire))
                    .min(out.len())
                    .min(want * 2 - got.len());
                p.fill(&sh, &mut out[..n], 0.0);
                got.extend_from_slice(&out[..n]);
            }
            let r = &full[f as usize * 2..][..got.len()];
            // one packet of pre-roll makes the restart bit-identical to a continuous decode
            let exact = got.iter().zip(r).filter(|(a, b)| a == b).count();
            let clock = sh.clock.load(Relaxed);
            println!(
                "seek {secs} s: {want} frames, {exact}/{} samples identical, clock {clock}",
                got.len()
            );
            assert_eq!(exact, got.len(), "seek {secs}");
            assert_eq!(p.src(&sh, p.d), f + want as u64);
            assert!(clock >= f && clock < p.src(&sh, p.d));
        }
        // past the end the clock keeps running on silence
        while !sh.eof.load(Acquire) {
            thread::yield_now();
        }
        let pos = p.src(&sh, p.d);
        for _ in 0..10 {
            p.fill(&sh, &mut out, 0.0);
        }
        assert_eq!(p.src(&sh, p.d), pos + 10 * 480);
        assert!(out.iter().all(|&x| x == 0.0));
        sh.quit.store(true, Relaxed);
        th.join().unwrap();
    }

    /// Queued silence (pause, the gap before resumed audio) holds the clock only while it plays: no lag behind the
    /// last sample while paused, no step back on resume.
    #[test]
    fn clock_counts_queued_silence() {
        let sh = Shared::new(48000, 48000, 2);
        sh.w.store(RING, Release); // a ring full of (silent) audio
        sh.paused.store(false, Relaxed);
        let (mut p, mut out) = (Play::default(), vec![0f32; 960]);
        // 1440 frames = 3 device buffers queued ahead of each new one
        let mut tick = |p: &mut Play| {
            p.fill(&sh, &mut out, 0.03);
            sh.clock.load(Relaxed)
        };
        let mut c = vec![];
        for _ in 0..20 {
            c.push(tick(&mut p));
        }
        assert_eq!(
            *c.last().unwrap(),
            p.src(&sh, p.d) - 480 - 1440,
            "heard = handed over − queued"
        );
        let at = p.src(&sh, p.d);
        sh.paused.store(true, Relaxed);
        let pause: Vec<_> = (0..6).map(|_| tick(&mut p)).collect();
        sh.paused.store(false, Relaxed);
        let play: Vec<_> = (0..6).map(|_| tick(&mut p)).collect();
        println!("pause at {at}: {pause:?}, resume: {play:?}");
        assert_eq!(pause, [at - 1440, at - 960, at - 480, at, at, at]);
        assert_eq!(play, [at, at, at, at, at + 480, at + 960]);
        c.extend(pause.iter().chain(&play));
        assert!(c.windows(2).all(|w| w[0] <= w[1]));
    }

    /// On a stream at another rate and channel count (48 kHz audio on a 44.1 kHz mono device) the clock still counts
    /// source frames: it lands on a seek target and reads what a 48 kHz stereo stream reads after the same time, the
    /// device's queue included.
    #[test]
    fn clock_resampled() {
        let (a, b) = (Shared::new(48000, 48000, 2), Shared::new(48000, 44100, 1));
        let (mut pa, mut pb) = (Play::default(), Play::default());
        for s in [&a, &b] {
            s.w.store(s.ring.len(), Release); // a ring full of (silent) audio
            s.paused.store(false, Relaxed);
            // a seek to 2 s, as the decode thread takes it
            s.base.store(96000, Relaxed);
            s.req.store(1, Relaxed);
            s.seen.store(1, Release);
        }
        // 10 ms buffers, 1/32 s queued in the device
        let (mut oa, mut ob, mut c) = (vec![0f32; 2 * 480], vec![0f32; 441], vec![]);
        for _ in 0..50 {
            pa.fill(&a, &mut oa, 0.03125);
            pb.fill(&b, &mut ob, 0.03125);
            c.push((a.clock.load(Relaxed), b.clock.load(Relaxed)));
        }
        println!("48 kHz stereo, 44.1 kHz mono: {:?} … {:?}", &c[..5], c[49]);
        assert_eq!(c[0], (96000, 96000), "the seek target");
        assert!(c.iter().all(|(x, y)| x.abs_diff(*y) <= 1), "{c:?}");
        assert_eq!(pb.src(&b, pb.d), 96000 + 50 * 480);
        assert_eq!(Clock(b.clone()).get(), Some(c[49].1 as f64 / 48000.0));
    }

    /// The decode thread for a stream at another rate and channel count (48 kHz stereo AAC on a 44.1 kHz mono device):
    /// from a seek it hands over the track's decode from the target resampled and mixed down, bit-exact to its last
    /// frames; the clock starts on the target and ends on the track's last frame. The track: two tones made with the
    /// ffmpeg CLI (skipped without it).
    #[test]
    fn feed_resampled() {
        let f = fixtures().join(format!("tones-{}.m4a", std::process::id()));
        let made = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg("aevalsrc=sin(2*PI*1000*t)/2|sin(2*PI*1500*t)/2:s=48000:d=3")
            // no PNS: its noise after a decoder restart is not a continuous decode's
            .args(["-c:a", "aac", "-aac_pns", "0"])
            .arg(&f)
            .status()
            .is_ok_and(|s| s.success());
        if !made {
            eprintln!("ffmpeg could not make the tones: skipped");
            return;
        }
        let m = Mp4::open(&f).unwrap();
        let Codec::Aac { asc, .. } = &m.tracks[0].codec else {
            panic!()
        };
        let (mut a, mut full, mut b) = (Aac::new(asc).unwrap(), vec![], vec![]);
        for s in &m.tracks[0].samples {
            m.read(s, &mut b).unwrap();
            full.extend_from_slice(a.decode(&b).unwrap());
        }
        let (f0, mut want) = (30000, vec![]);
        let mut cv = Conv::new(48000, 2, 44100, 1).unwrap();
        cv.run(&full[2 * f0..], &mut want);
        cv.finish(&mut want);
        let sh = Shared::new(48000, 44100, 1);
        sh.paused.store(false, Relaxed);
        let (s, a) = (sh.clone(), Aac::new(asc).unwrap());
        let cv = Conv::new(48000, 2, 44100, 1).unwrap();
        let th = thread::spawn(move || feed(&s, &m, 0, a, cv));
        sh.target.store(f0 as u64, Relaxed);
        sh.req.fetch_add(1, Release);
        let (mut p, mut out, mut got) = (Play::default(), vec![0f32; 441], vec![]);
        while p.ack != 1 {
            p.fill(&sh, &mut out, 0.0);
        }
        let c0 = sh.clock.load(Relaxed);
        loop {
            let t0 = Instant::now();
            while sh.w.load(Acquire) - sh.r.load(Acquire) < out.len() && !sh.eof.load(Acquire) {
                assert!(t0.elapsed().as_secs() < 5, "decoder stalled");
                thread::yield_now();
            }
            let n = (sh.w.load(Acquire) - sh.r.load(Acquire)).min(out.len());
            if n == 0 {
                break; // the end, all of it heard
            }
            p.fill(&sh, &mut out[..n], 0.0);
            got.extend_from_slice(&out[..n]);
        }
        sh.quit.store(true, Relaxed);
        th.join().unwrap();
        std::fs::remove_file(&f).unwrap();
        let (end, last) = (p.src(&sh, p.d), full.len() as u64 / 2);
        println!(
            "from frame {f0}: {} of {} frames as resampled alone, clock {c0} → {end} (track {last})",
            got.iter().zip(&want).filter(|(a, b)| a == b).count(),
            want.len()
        );
        assert_eq!(c0, f0 as u64, "the seek target");
        assert!(got == want, "not the decode resampled");
        // the stream's last frame is at most one source frame after the track's
        assert!((last..=last + 1).contains(&end), "ends at {end}");
    }

    /// A packet that fails to decode becomes silence of its length: everything after it plays at its source time.
    #[test]
    fn bad_packet_keeps_time() {
        use std::io::{Seek, Write};
        let Some(d) = samples() else { return };
        let (full, _) = decode_both("v635.lrv");
        let bad = fixtures().join("bad_packet.lrv");
        std::fs::copy(d.join("v635.lrv"), &bad).unwrap();
        let m = Mp4::open(&bad).unwrap();
        let (k, s) = (100, m.tracks[1].samples[100]);
        // all zero: symphonia rejects it (all 0xff would decode as an empty frame)
        let mut f = std::fs::OpenOptions::new().write(true).open(&bad).unwrap();
        f.seek(io::SeekFrom::Start(s.off)).unwrap();
        f.write_all(&vec![0; s.size as usize]).unwrap();
        let Codec::Aac { asc, .. } = &m.tracks[1].codec else {
            panic!()
        };
        let mut b = vec![];
        m.read(&s, &mut b).unwrap();
        assert!(
            Aac::new(asc).unwrap().decode(&b).is_err(),
            "corrupt packet decodes"
        );
        let sh = Shared::new(48000, 48000, 2);
        sh.paused.store(false, Relaxed);
        let (s2, a) = (sh.clone(), Aac::new(asc).unwrap());
        let cv = Conv::new(48000, 2, 48000, 2).unwrap();
        let th = thread::spawn(move || feed(&s2, &m, 1, a, cv));
        let (mut p, mut out, mut got) = (Play::default(), vec![0f32; 960], vec![]);
        loop {
            let eof = sh.eof.load(Acquire);
            let n = (sh.w.load(Acquire) - sh.r.load(Acquire)).min(out.len());
            if n == 0 && eof {
                break;
            }
            p.fill(&sh, &mut out[..n], 0.0);
            got.extend_from_slice(&out[..n]);
        }
        sh.quit.store(true, Relaxed);
        th.join().unwrap();
        // the bad packet is silence; the next one decodes without its overlap half, after that it is bit-exact
        let (a, z) = (k * 2048, (k + 2) * 2048);
        let same = |r: std::ops::Range<usize>| got[r.clone()] == full[r];
        println!(
            "{} / {} samples; packet {k} silent: {}; exact before {a} and from {z}",
            got.len(),
            full.len(),
            got[a..a + 2048].iter().all(|&x| x == 0.0)
        );
        assert_eq!(got.len(), full.len());
        assert!(got[a..a + 2048].iter().all(|&x| x == 0.0));
        assert!(same(0..a) && same(z..full.len()));
    }

    /// The silent player's clock: holds at open and while paused, follows wall time while playing, lands exactly on
    /// a seek target.
    #[test]
    fn silent_clock() {
        let p = Player::silent(10.0);
        let c = p.clock();
        thread::sleep(Duration::from_millis(50));
        assert_eq!(c.secs(), 0.0, "paused at open");
        let t0 = Instant::now();
        p.play();
        thread::sleep(Duration::from_millis(300));
        let (a, wall) = (c.secs(), t0.elapsed().as_secs_f64());
        p.pause();
        thread::sleep(Duration::from_millis(20));
        let b = c.secs();
        thread::sleep(Duration::from_millis(100));
        let (b2, s0) = (c.secs(), Instant::now());
        p.seek(3.0);
        // `get` stays None until the seek reaches the clock, then reads the target, never the old position
        let s = loop {
            if let Some(x) = c.get() {
                break x;
            }
            assert!(
                s0.elapsed().as_millis() < 500,
                "seek never reached the clock"
            );
            thread::yield_now();
        };
        thread::sleep(Duration::from_millis(20));
        // wall time from `play`: sleeps overshoot under load, so the paused gap is not 20 ms
        let t1 = Instant::now();
        p.play();
        thread::sleep(Duration::from_millis(200));
        let (s2, sw) = (c.secs(), t1.elapsed().as_secs_f64());
        println!(
            "played {a:.4} s in {wall:.4} s wall; paused {b:.4} → {b2:.4}; seek 3 → {s:.4}, +{sw:.4} s → {s2:.4}"
        );
        assert!((a - wall).abs() < 0.02, "clock {a} vs wall {wall}");
        assert_eq!(b, b2, "paused clock holds");
        assert_eq!(s, 3.0, "seek target");
        assert!((s2 - 3.0 - sw).abs() < 0.02, "after seek {s2}");
    }

    /// Ids resolve as the app's settings keep them; a player opened on a device runs its clock there (paused, at
    /// volume 0: nothing is heard).
    #[test]
    fn plays_on_device() {
        let id = |d: &cpal::Device| d.id().unwrap().to_string();
        let def = id(&device(None).unwrap());
        assert_eq!(
            id(&device(Some("x5desk")).unwrap()),
            def,
            "an id that does not parse"
        );
        assert_eq!(
            id(&device(Some("alsa:x5desk-none")).unwrap()),
            def,
            "no such device"
        );
        let outs: Vec<String> = cpal::default_host()
            .output_devices()
            .unwrap()
            .filter_map(|d| d.id().ok().map(|i| i.to_string()))
            .collect();
        println!("default {def}; outputs {outs:?}");
        for o in &outs {
            assert_eq!(&id(&device(Some(o)).unwrap()), o);
        }
        let Some(d) = samples() else { return };
        let f = d.join("v635.insv");
        let p = Player::open_on(&f, device(Some(&def))).unwrap();
        p.volume(0.0);
        let (c, t0) = (p.clock(), Instant::now());
        p.seek(2.0);
        let s = loop {
            if let Some(x) = c.get() {
                break x;
            }
            assert!(t0.elapsed().as_millis() < 2000, "the device never ran");
            thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(s, 2.0);
        // the player is on the device it was given: ALSA's OSS plugin without /dev/dsp does not open
        if outs.iter().any(|o| o == "alsa:oss") && !Path::new("/dev/dsp").exists() {
            assert!(Player::open_on(&f, device(Some("alsa:oss"))).is_err());
        }
    }

    #[test]
    #[ignore = "plays 2 s of audio on the default output device (volume 0.1)"]
    fn plays_on_default_device() {
        let Some(d) = samples() else { return };
        let p = Player::open(d.join("v635.insv")).unwrap();
        p.volume(0.1);
        let c = p.clock();
        thread::sleep(Duration::from_millis(100));
        assert_eq!(c.secs(), 0.0, "paused at open");
        let t0 = Instant::now();
        p.play();
        // sample the clock every ms to see its step (≈ the device period) and its rate
        let mut steps = vec![];
        let (mut last, mut first_move) = (0.0, None);
        while t0.elapsed() < Duration::from_millis(1500) {
            let x = c.secs();
            if x != last {
                first_move.get_or_insert(t0.elapsed().as_secs_f64());
                steps.push(x - last);
                last = x;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let (a, wall) = (c.secs(), t0.elapsed().as_secs_f64());
        steps.sort_by(f64::total_cmp);
        let step = steps[steps.len() / 2];
        p.seek(4.0);
        thread::sleep(Duration::from_millis(500));
        let b = c.secs();
        p.pause();
        // the audio still queued in the device plays out first
        thread::sleep(Duration::from_millis(100));
        let c1 = c.secs();
        thread::sleep(Duration::from_millis(300));
        let c2 = c.secs();
        println!(
            "clock {a:.3} s after {wall:.3} s wall (first move at {:.3} s, median step {:.1} ms); seek 4.0 + 0.5 s → {b:.3}; \
             paused {c1:.3} → {c2:.3}",
            first_move.unwrap_or(f64::NAN),
            step * 1e3
        );
        // rate: from the first move on the clock tracks wall time
        let fm = first_move.unwrap();
        assert!(
            (a - (wall - fm)).abs() < 0.05 + step,
            "clock {a} vs {}",
            wall - fm
        );
        assert!((b - 4.5).abs() < 0.15, "after seek {b}");
        assert_eq!(c1, c2, "paused clock holds");
    }
}
