//! The editor's preview sound: the timeline mixed as the exporter mixes it (crate media's mixer: clip audio
//! time-stretched to its speed under its volume envelope, music with ducking, the limiter; RNNoise on the clips when
//! denoise is on), played on the output picked in Settings from a playhead position. The device callback drains a
//! single-producer / single-consumer ring and counts what it played: that count is the editor's clock while it plays.
//! An underrun holds the clock. Without a device the clock runs on wall time and nothing is mixed. A mixer that fails
//! reports why and plays silence on, so the clock (and the picture) keep going. The device is looked up, opened and
//! closed on the sound's own thread, one stream at a time: on the UI thread that took 7–8 ms at every Play, seek during
//! playback and denoise change (and listed every ALSA device for a picked one). The stream runs in the format
//! `audio::Conv::on` picks for the mix (macOS: the device's own rate), the mix converted to it.

use cpal::traits::{DeviceTrait, StreamTrait};
use media::{
    audio::{Aac, Conv},
    denoise::{self, Stereo},
    mix::{self, Kind, Mixer, Track},
    mp4::{self, Mp4},
    music::Music,
    stretch::{self, Stretch},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering::*},
    },
    thread,
    time::{Duration, Instant},
};

const R: f64 = mix::RATE;
const BLOCK: usize = 1024;
/// Ring of interleaved samples: 0.34 s of 48 kHz stereo.
const RING: usize = 1 << 15;

/// The open stream's ring of interleaved samples: [`RING`], more for a stream of more samples a second (macOS: a
/// device at its own rate). Made on the sound's thread with the stream.
type Ring = Arc<[AtomicU32]>;

fn ring(cv: &Conv) -> Ring {
    let k = (cv.rate as usize * cv.ch as usize).div_ceil(2 * R as usize);
    (0..RING * k.next_power_of_two())
        .map(|_| AtomicU32::new(0))
        .collect()
}

struct Sh {
    /// samples written to the ring, read from it
    w: AtomicUsize,
    r: AtomicUsize,
    /// frames the device took from the ring; the device's queue, frames; the stream's rate
    played: AtomicU64,
    lat: AtomicU64,
    rate: AtomicU32,
    vol: AtomicU32,
    quit: AtomicBool,
    /// the mixer's first error
    err: Mutex<Option<String>>,
    /// the device: 0 being opened (the clock holds), 1 playing, 2 none (the wall clock runs from `wall`)
    on: AtomicU8,
    wall: OnceLock<Instant>,
}

/// Held by the thread whose stream is open: the next opens once the last closed (a device without mixing refuses a
/// second stream).
static DEV: Mutex<()> = Mutex::new(());

/// Preview playback from one position; dropping it stops (its thread closes the stream).
pub struct Sound {
    sh: Arc<Sh>,
    t0: f64,
}

/// What the mixer needs: the timeline and every file's path (the music: its first).
pub struct Mix {
    pub tl: edit::Timeline,
    pub files: Vec<PathBuf>,
    pub denoise: Option<f32>,
}

impl Sound {
    /// Plays `m` from output time `t0` at volume `vol`.
    pub fn start(m: Mix, t0: f64, vol: f32) -> Sound {
        let sh = Arc::new(Sh {
            w: AtomicUsize::new(0),
            r: AtomicUsize::new(0),
            played: AtomicU64::new(0),
            lat: AtomicU64::new(0),
            rate: AtomicU32::new(R as u32),
            vol: AtomicU32::new(vol.to_bits()),
            quit: AtomicBool::new(false),
            err: Default::default(),
            on: AtomicU8::new(0),
            wall: OnceLock::new(),
        });
        let s = sh.clone();
        thread::Builder::new()
            .name("editor sound".into())
            .spawn(move || {
                let _dev = DEV.lock().unwrap_or_else(|e| e.into_inner());
                // a Sound dropped while waiting (playback restarted again) opens nothing
                let o = (!s.quit.load(Relaxed)).then(|| open(&s)).flatten();
                let Some((_st, mut cv, rg)) = o else {
                    let _ = s.wall.set(Instant::now());
                    s.on.store(2, Release);
                    return;
                };
                s.rate.store(cv.rate, Relaxed);
                s.on.store(1, Release);
                if let Err(e) = run(&s, &rg, &m, t0, &mut cv) {
                    eprintln!("editor mix: {e}");
                    *s.err.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                    let n = rg.len();
                    while !s.quit.load(Relaxed) {
                        let (r, w) = (s.r.load(Acquire), s.w.load(Relaxed));
                        for i in w..r + n {
                            rg[i % n].store(0, Relaxed);
                        }
                        s.w.store(r + n, Release);
                        thread::sleep(Duration::from_millis(5));
                    }
                }
            })
            .expect("sound thread");
        Sound { sh, t0 }
    }

    /// Output time heard now.
    pub fn now(&self) -> f64 {
        let sh = &self.sh;
        match sh.on.load(Acquire) {
            1 => {
                let n = sh.played.load(Relaxed).saturating_sub(sh.lat.load(Relaxed));
                self.t0 + n as f64 / sh.rate.load(Relaxed) as f64
            }
            2 => self.t0 + sh.wall.get().map_or(0.0, |w| w.elapsed().as_secs_f64()),
            _ => self.t0,
        }
    }

    pub fn vol(&self, v: f32) {
        self.sh.vol.store(v.to_bits(), Relaxed);
    }

    pub fn err(&self) -> Option<String> {
        self.sh.err.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Whether a device plays it.
    pub fn live(&self) -> bool {
        self.sh.on.load(Acquire) == 1
    }
}

impl Drop for Sound {
    fn drop(&mut self) {
        self.sh.quit.store(true, Relaxed);
    }
}

/// The output stream draining its ring, playing, the mix's conversion to its format and the ring.
fn open(sh: &Arc<Sh>) -> Option<(cpal::Stream, Conv, Ring)> {
    let dev = crate::settings::output()?;
    stream(sh, &dev).ok()
}

/// [`open`] on `dev`, in the format `Conv::on` picks for the mix.
fn stream(sh: &Arc<Sh>, dev: &cpal::Device) -> std::io::Result<(cpal::Stream, Conv, Ring)> {
    let cv = Conv::on(dev, R as u32, 2)?;
    let cfg = cpal::StreamConfig {
        channels: cv.ch,
        sample_rate: cv.rate,
        buffer_size: cpal::BufferSize::Default,
    };
    let (s, rg, rate, ch) = (sh.clone(), ring(&cv), cv.rate as f64, cv.ch as usize);
    let b = rg.clone();
    let cb = move |o: &mut [f32], info: &cpal::OutputCallbackInfo| {
        // stopped: silence until the stream closes
        if s.quit.load(Relaxed) {
            o.fill(0.0);
            return;
        }
        let ts = info.timestamp();
        let q = ts.playback.duration_since(ts.callback).as_secs_f64();
        s.lat.store((q * rate) as u64, Relaxed);
        let (r, w) = (s.r.load(Relaxed), s.w.load(Acquire));
        let k = (w - r).min(o.len()) / ch * ch;
        let v = f32::from_bits(s.vol.load(Relaxed));
        for (i, x) in o[..k].iter_mut().enumerate() {
            *x = f32::from_bits(b[(r + i) % b.len()].load(Relaxed)) * v;
        }
        o[k..].fill(0.0);
        s.r.store(r + k, Release);
        s.played.fetch_add((k / ch) as u64, Relaxed);
    };
    let st = dev
        .build_output_stream(cfg, cb, |e| eprintln!("editor audio: {e}"), None)
        .map_err(std::io::Error::other)?;
    st.play().map_err(std::io::Error::other)?;
    Ok((st, cv, rg))
}

/// A clip's audio from output frame `j` on: its AAC decoded in order, fed to the time-stretch one output frame at a
/// time with the speed there (export's `Voice`), denoised when asked.
struct Voice {
    k: usize,
    n: std::ops::Range<u64>,
    j: u64,
    tr: usize,
    sh: i64,
    dec: Aac,
    i: usize,
    pkt: Vec<u8>,
    src: Vec<f32>,
    s0: i64,
    st: Stretch,
    chunk: Vec<f32>,
    /// stretched (and denoised up to `dn.1`) output from mixer frame `y0`
    y: Vec<f32>,
    y0: u64,
    done: bool,
    dn: Option<Box<Dn>>,
}

/// A voice's denoiser: frames of `y` done, the delay still to drop, a frame per side.
struct Dn {
    st: Stereo,
    done: usize,
    drop: usize,
    l: [f32; denoise::FRAME],
    r: [f32; denoise::FRAME],
}

impl Voice {
    fn new(
        tl: &edit::Timeline,
        m: &Mp4,
        tr: usize,
        k: usize,
        j: u64,
        dn: Option<f32>,
    ) -> Result<Voice, String> {
        let t = &m.tracks[tr];
        let mp4::Codec::Aac { asc, .. } = &t.codec else {
            unreachable!("an AAC track")
        };
        let dec = Aac::new(asc).map_err(|e| format!("clip {k}: AAC: {e}"))?;
        if dec.rate as f64 != R {
            return Err(format!("clip {k}: audio at {} Hz", dec.rate));
        }
        let (sh, ts, n) = (t.shift(m.timescale), t.timescale as i64, tl.frames(k));
        let s = (tl.content(k, tl.out.fps.t(j - n.start)) * R).round() as i64;
        // the packet holding s, and one before it for the decoder's overlap
        let i = (t.samples)
            .partition_point(|x| (x.pts + sh) * 48000 / ts <= s)
            .saturating_sub(2);
        Ok(Voice {
            k,
            j,
            y0: (tl.out.fps.t(j) * R).round() as u64,
            n,
            tr,
            sh,
            dec,
            i,
            pkt: vec![],
            src: vec![],
            s0: t.samples.get(i).map_or(s, |x| (x.pts + sh) * 48000 / ts),
            st: Stretch::new(2),
            chunk: vec![],
            y: vec![],
            done: false,
            dn: dn.map(|s| {
                let mut st = Stereo::new();
                st.strength = s.clamp(0.0, 1.0);
                Box::new(Dn {
                    st,
                    done: 0,
                    drop: denoise::DELAY,
                    l: [0.0; denoise::FRAME],
                    r: [0.0; denoise::FRAME],
                })
            }),
        })
    }

    /// Frames of `y` ready.
    fn ready(&self) -> usize {
        match &self.dn {
            // nothing until the denoiser's delay is dropped
            Some(d) if d.drop > 0 => 0,
            Some(d) => d.done,
            None => self.y.len() / 2,
        }
    }

    /// Stretches (and denoises) on until the output reaches mixer frame `to` or the clip's end.
    fn fill(&mut self, tl: &edit::Timeline, m: &Mp4, to: u64) -> Result<(), String> {
        let (fps, t) = (tl.out.fps, &m.tracks[self.tr]);
        let ts = t.timescale as i64;
        while !self.done && self.y0 + self.ready() as u64 <= to {
            if self.j == self.n.end {
                self.st.finish(&mut self.y);
                if self.dn.is_some() {
                    // flush the denoiser's delay
                    self.y
                        .resize(self.y.len() + 2 * (denoise::DELAY + denoise::FRAME), 0.0);
                }
                self.done = true;
            } else {
                let (ta, tb) = (
                    fps.t(self.j - self.n.start),
                    fps.t(self.j + 1 - self.n.start),
                );
                let (sa, sb) = (tl.content(self.k, ta), tl.content(self.k, tb));
                let (a, b) = ((sa * R).round() as i64, (sb * R).round() as i64);
                while self.s0 + (self.src.len() / 2) as i64 <= b && self.i < t.samples.len() {
                    let x = t.samples[self.i];
                    self.i += 1;
                    let gap =
                        (x.pts + self.sh) * 48000 / ts - self.s0 - (self.src.len() / 2) as i64;
                    self.src
                        .resize(self.src.len() + 2 * gap.max(0) as usize, 0.0);
                    m.read(&x, &mut self.pkt)
                        .map_err(|e| format!("clip {}: audio: {e}", self.k))?;
                    let ch = self.dec.ch.max(1) as usize;
                    match self.dec.decode(&self.pkt) {
                        Ok(pcm) => (pcm.chunks_exact(ch))
                            .for_each(|f| self.src.extend_from_slice(&[f[0], f[ch.min(2) - 1]])),
                        Err(_) => self.src.resize(self.src.len() + 2 * x.dur as usize, 0.0),
                    }
                }
                self.chunk.clear();
                for f in a..b {
                    let i = (f - self.s0) as usize;
                    self.chunk
                        .extend_from_slice(self.src.get(2 * i..2 * i + 2).unwrap_or(&[0.0; 2]));
                }
                let d = ((b - self.s0).max(0) as usize).min(self.src.len() / 2);
                self.src.drain(..2 * d);
                self.s0 += d as i64;
                let mut v = (sb - sa) / (tb - ta);
                for x in [stretch::MIN, stretch::MAX] {
                    if (v - x).abs() < 1e-9 * x {
                        v = x;
                    }
                }
                self.st.run(&self.chunk, v, &mut self.y);
                self.j += 1;
            }
            if let Some(d) = &mut self.dn {
                const F: usize = denoise::FRAME;
                while d.done + F <= self.y.len() / 2 {
                    let y = &mut self.y[2 * d.done..2 * (d.done + F)];
                    for (i, f) in y.as_chunks::<2>().0.iter().enumerate() {
                        (d.l[i], d.r[i]) = (f[0], f[1]);
                    }
                    d.st.frame(&mut d.l, &mut d.r);
                    for (i, f) in y.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        (f[0], f[1]) = (d.l[i], d.r[i]);
                    }
                    d.done += F;
                }
                // the denoiser's output is DELAY frames late: drop its start once
                if d.drop > 0 && d.done >= d.drop {
                    self.y.drain(..2 * d.drop);
                    d.done -= d.drop;
                    d.drop = 0;
                }
            }
        }
        Ok(())
    }
}

/// The mixer thread: mixes block after block from `t0` while ring `rg` has room, converted by `cv`.
fn run(sh: &Sh, rg: &[AtomicU32], m: &Mix, t0: f64, cv: &mut Conv) -> Result<(), String> {
    let (tl, fps) = (&m.tl, m.tl.out.fps);
    let (mut mx, mut vs) = (Mixer::new(), vec![]);
    let mut mp4s: Vec<Option<(Mp4, usize)>> = (0..tl.files.len()).map(|_| None).collect();
    let start = (t0 * R).round() as u64;
    for (k, c) in tl.clips.iter().enumerate() {
        // dropped meanwhile (a seek while playing): the next Sound waits for this thread (`DEV`), opening no more files
        if sh.quit.load(Relaxed) {
            return Ok(());
        }
        let n = tl.frames(k);
        let span = [fps.t(n.start), fps.t(n.end)];
        if c.hold > 0.0 || c.mute || span[1] <= t0 {
            continue;
        }
        let f = c.file as usize;
        if mp4s[f].is_none() {
            let Some(p) = m.files.get(f) else { continue };
            let Ok(x) = Mp4::open(p) else { continue };
            let Some(t) = x
                .tracks
                .iter()
                .position(|t| matches!(t.codec, mp4::Codec::Aac { .. }))
            else {
                continue;
            };
            mp4s[f] = Some((x, t));
        }
        let mut t = Track::new(Kind::Clip, span);
        t.fade = [0.005; 2];
        if !c.vol.is_empty() {
            t.env = (n.clone())
                .filter_map(|j| Some((fps.t(j), tl.frame(j)?.gain)))
                .collect();
        }
        mx.tracks.push(t);
        vs.push((k, span.map(|s| (s * R).round() as u64), None::<Voice>));
    }
    let mut music = None;
    if let Some(mu) = tl.out.music.as_ref().filter(|_| !sh.quit.load(Relaxed)) {
        let p = m.files.get(mu.file as usize).ok_or("music file")?;
        let err = |x: std::io::Error| format!("music {}: {x}", p.display());
        let mut x = Music::open(p).map_err(err)?;
        let len = if x.duration > 0.0 {
            x.duration
        } else {
            tl.files[mu.file as usize].dur
        };
        let span = [(-mu.offset).max(0.0), tl.dur().min(len - mu.offset)];
        if span[1] > span[0].max(t0) {
            let from = span[0].max(t0);
            x.seek(from + mu.offset).map_err(err)?;
            let mut t = Track::new(Kind::Music, span);
            t.gain = mu.gain;
            mx.tracks.push(t);
            mx.duck.depth = if mu.duck { edit::DUCK } else { 1.0 };
            music = Some((x, (from * R).round() as u64));
        }
    }
    // past the timeline's end the mix is silence, which keeps the clock running to the end
    let (lat, mut at, mut o) = (mix::LATENCY as u64, start, vec![0f32; 2 * BLOCK]);
    let (room, mut c) = (cv.most(BLOCK), vec![]);
    while !sh.quit.load(Relaxed) {
        let (r, w) = (sh.r.load(Acquire), sh.w.load(Relaxed));
        if rg.len() - (w - r) < room {
            thread::sleep(Duration::from_millis(5));
            continue;
        }
        let (p, n) = (at, BLOCK as u64);
        let q = p + n;
        for (j, (k, span, v)) in vs.iter_mut().enumerate() {
            let b = &mut mx.tracks[j].buf;
            b.clear();
            if q <= span[0] || p >= span[1] {
                *v = None;
                continue;
            }
            let (x, tr) = mp4s[tl.clips[*k].file as usize].as_ref().unwrap();
            let v = match v {
                Some(v) => v,
                None => {
                    // from the frame playing at the block's start
                    let f = tl.frames(*k);
                    let j = (fps.n(p as f64 / R).floor() as u64).clamp(f.start, f.end - 1);
                    v.insert(Voice::new(tl, x, *tr, *k, j, m.denoise)?)
                }
            };
            v.fill(tl, x, q)?;
            b.resize(2 * n as usize, 0.0);
            let s = p.max(v.y0);
            let (i, len) = ((s - v.y0) as usize, q.saturating_sub(s) as usize);
            let got = v.ready().saturating_sub(i).min(len);
            if s < q {
                b[2 * (s - p) as usize..][..2 * got].copy_from_slice(&v.y[2 * i..2 * (i + got)]);
            }
            let d = (q.saturating_sub(v.y0) as usize).min(v.ready());
            v.y.drain(..2 * d);
            if let Some(dn) = &mut v.dn {
                dn.done -= d;
            }
            v.y0 += d as u64;
        }
        if let Some((mu, mt)) = &mut music {
            let b = &mut mx.tracks[vs.len()].buf;
            b.clear();
            b.resize(2 * n as usize, 0.0);
            if q > *mt {
                let s = mt.saturating_sub(p) as usize;
                let got = mu
                    .read(&mut b[2 * s..])
                    .map_err(|x| format!("music: {x}"))?;
                b.truncate(2 * (s + got));
                *mt = q;
            }
        }
        mx.run(p as f64 / R, &mut o);
        // output frame i of a block is its input frame i − LATENCY: the first LATENCY frames are before t0
        let s = 2 * (start + lat).saturating_sub(p).min(n) as usize;
        c.clear();
        cv.run(&o[s..], &mut c);
        let w = sh.w.load(Relaxed);
        for (i, x) in c.iter().enumerate() {
            rg[(w + i) % rg.len()].store(x.to_bits(), Relaxed);
        }
        sh.w.store(w + c.len(), Release);
        at = q;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sound's state as on a 48 kHz device, playing.
    fn sh() -> Sh {
        Sh {
            w: AtomicUsize::new(0),
            r: AtomicUsize::new(0),
            played: AtomicU64::new(0),
            lat: AtomicU64::new(0),
            rate: AtomicU32::new(R as u32),
            vol: AtomicU32::new(1f32.to_bits()),
            quit: AtomicBool::new(false),
            err: Default::default(),
            on: AtomicU8::new(1),
            wall: OnceLock::new(),
        }
    }

    /// `secs` of `m` mixed from `t0` by the mixer thread, the test draining the ring as the device would.
    fn mixdown(m: &Mix, t0: f64, secs: f64) -> Vec<f32> {
        mixdown_on(m, t0, secs, Conv::new(R as u32, 2, R as u32, 2).unwrap())
    }

    /// [`mixdown`] for a stream in `cv`'s format.
    fn mixdown_on(m: &Mix, t0: f64, secs: f64, mut cv: Conv) -> Vec<f32> {
        let (sh, rg) = (sh(), ring(&cv));
        let want = cv.ch as usize * (secs * cv.rate as f64) as usize;
        thread::scope(|sc| {
            let h = sc.spawn(|| run(&sh, &rg, m, t0, &mut cv));
            let mut o = Vec::with_capacity(want + rg.len());
            while o.len() < want && !h.is_finished() {
                let (r, w) = (sh.r.load(Relaxed), sh.w.load(Acquire));
                o.extend((r..w).map(|i| f32::from_bits(rg[i % rg.len()].load(Relaxed))));
                sh.r.store(w, Release);
                thread::sleep(Duration::from_millis(1));
            }
            sh.quit.store(true, Relaxed);
            h.join().unwrap().unwrap();
            o.truncate(want);
            o
        })
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    /// Amplitude of `f` Hz in the left channel (Goertzel).
    fn tone(x: &[f32], f: f64) -> f64 {
        let w = std::f64::consts::TAU * f / R;
        let (mut a, mut b) = (0.0, 0.0);
        let n = x.len() / 2;
        for i in 0..n {
            let s = x[2 * i] as f64 + 2.0 * w.cos() * a - b;
            (b, a) = (a, s);
        }
        (a * a + b * b - 2.0 * w.cos() * a * b).sqrt() * 2.0 / n as f64
    }

    /// The preview mix of v606 2–6 s (`X5_SAMPLES`) with a chord as music: the clip's sound comes through; the music
    /// alone has the chord's 220 Hz note at its level; under the clip audio the ducking takes it to about edit's DUCK
    /// (0.3) of that, without ducking it stays; a 2× clip lasts half as long (then silence); a start at 1 s plays what
    /// the start at 0 plays from 1 s; denoise changes the sound and stays finite.
    #[test]
    fn mix() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES unset: skipped");
            return;
        };
        let v = d.join("v606.insv");
        let w = std::env::temp_dir().join(format!("x5desk-mix-{}.wav", std::process::id()));
        crate::editor::tests::chord(&w, 10.0);
        let mk = |speed: f64, mute: bool, music: Option<bool>, dn: Option<f32>| {
            let mut tl = edit::Timeline::new(edit::Rate {
                num: 30000,
                den: 1001,
            });
            tl.files.push(edit::File {
                path: v.to_string_lossy().into(),
                dur: 31.0,
            });
            tl.files.push(edit::File {
                path: w.to_string_lossy().into(),
                dur: 10.0,
            });
            let mut c = edit::Clip::new(0, 2.0, 6.0);
            (c.speed, c.mute) = (
                if speed == 1.0 {
                    vec![]
                } else {
                    vec![(2.0, speed)]
                },
                mute,
            );
            tl.clips.push(c);
            tl.out.music = music.map(|duck| edit::Music {
                file: 1,
                offset: 0.0,
                gain: 1.0,
                duck,
            });
            tl.fix();
            tl.check().unwrap();
            Mix {
                files: vec![v.clone(), w.clone()],
                tl,
                denoise: dn,
            }
        };
        let s = |x: &[f32], a: f64, b: f64| x[2 * (a * R) as usize..2 * (b * R) as usize].to_vec();
        let clip = mixdown(&mk(1.0, false, None, None), 0.0, 4.0);
        let alone = mixdown(&mk(1.0, true, Some(true), None), 0.0, 4.0);
        let duck = mixdown(&mk(1.0, false, Some(true), None), 0.0, 4.0);
        let flat = mixdown(&mk(1.0, false, Some(false), None), 0.0, 4.0);
        let (c, m0) = (rms(&s(&clip, 0.5, 3.5)), tone(&s(&alone, 1.0, 3.0), 220.0));
        let (md, mf) = (
            tone(&s(&duck, 1.0, 3.0), 220.0),
            tone(&s(&flat, 1.0, 3.0), 220.0),
        );
        let (cd, cf) = (tone(&s(&clip, 1.0, 3.0), 220.0), md / mf);
        eprintln!(
            "clip rms {c:.4} ({:.1} dBFS), its 220 Hz {cd:.4}; music 220 Hz alone {m0:.4}, under the clip ducked {md:.4} / not ducked {mf:.4} (ratio {cf:.3})",
            20.0 * c.log10()
        );
        assert!(c > 0.01, "the clip's sound");
        // the chord's notes at −12 dBFS / 3 each: 0.25 / 3 = 0.083
        assert!((m0 - 0.0833).abs() < 0.01, "the music's level {m0}");
        assert!((mf - m0).abs() < 0.2 * m0 + cd, "no ducking: {mf} vs {m0}");
        assert!(md < 0.5 * mf, "ducking: {md} vs {mf}");
        let fast = mixdown(&mk(2.0, false, None, None), 0.0, 3.0);
        let (a, b) = (rms(&s(&fast, 0.2, 1.8)), rms(&s(&fast, 2.1, 2.9)));
        eprintln!("2×: rms while it plays {a:.4}, after its 2 s {b:.6}");
        assert!(a > 0.01 && b < 1e-4, "a 2× clip of 4 s lasts 2 s");
        let late = mixdown(&mk(1.0, false, None, None), 1.0, 2.0);
        let (x, y) = (s(&late, 0.2, 1.8), s(&clip, 1.2, 2.8));
        let r = x.iter().zip(&y).map(|(a, b)| (a * b) as f64).sum::<f64>()
            / (rms(&x) * rms(&y) * x.len() as f64);
        eprintln!("start at 1 s vs the start at 0 from 1 s: correlation {r:.4}");
        assert!(r > 0.95, "correlation {r}");
        let dn = mixdown(&mk(1.0, false, None, Some(1.0)), 0.0, 4.0);
        let (dr, diff) = (
            rms(&s(&dn, 0.5, 3.5)),
            rms(&s(&dn, 0.5, 3.5)
                .iter()
                .zip(s(&clip, 0.5, 3.5))
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>()),
        );
        eprintln!("denoise 1.0: rms {dr:.4} (dry {c:.4}), difference {diff:.4}");
        assert!(dn.iter().all(|v| v.is_finite()) && diff > 0.001 && dr > 0.0);
        let _ = std::fs::remove_file(&w);
    }

    /// On a device at another rate and channel count (44.1 kHz mono: macOS opens a device at its own rate) the preview
    /// plays the 48 kHz stereo mix resampled and mixed down, sample for sample, and its clock reads the device's frames
    /// at that rate. The mix: a chord as music, under a clip whose file is not there.
    #[test]
    fn resampled() {
        let d = std::env::var_os("X5_FIXTURES").map_or_else(std::env::temp_dir, PathBuf::from);
        let w = d.join(format!("x5desk-chord-{}.wav", std::process::id()));
        crate::editor::tests::chord(&w, 3.0);
        let mut tl = edit::Timeline::new(edit::Rate {
            num: 30000,
            den: 1001,
        });
        for p in [d.join("x5desk-nowhere.mp4"), w.clone()] {
            tl.files.push(edit::File {
                path: p.to_string_lossy().into(),
                dur: 3.0,
            });
        }
        tl.clips.push(edit::Clip::new(0, 0.0, 3.0));
        tl.out.music = Some(edit::Music {
            file: 1,
            offset: 0.0,
            gain: 1.0,
            duck: false,
        });
        tl.fix();
        let m = Mix {
            files: tl.files.iter().map(|f| PathBuf::from(&f.path)).collect(),
            tl,
            denoise: None,
        };
        let a = mixdown(&m, 0.5, 2.0);
        let b = mixdown_on(&m, 0.5, 2.0, Conv::new(R as u32, 2, 44100, 1).unwrap());
        let _ = std::fs::remove_file(&w);
        let mut want = vec![];
        Conv::new(R as u32, 2, 44100, 1).unwrap().run(&a, &mut want);
        // the resampler holds back its last input frames: compare what both have
        let n = b.len().min(want.len());
        eprintln!(
            "{} frames at 44.1 kHz mono, rms {:.4} (48 kHz stereo {:.4}); {n} as the mix resampled",
            b.len(),
            rms(&b[..n]),
            rms(&a)
        );
        assert!(n > 88000 && b[..n] == want[..n] && rms(&b) > 0.05);
        let s = Sound {
            sh: Arc::new(Sh {
                played: AtomicU64::new(66150),
                lat: AtomicU64::new(22050),
                rate: AtomicU32::new(44100),
                ..sh()
            }),
            t0: 0.5,
        };
        assert_eq!(s.now(), 1.5);
    }

    /// A mixer told to quit before it opened the clips' files (its Sound dropped by a seek during playback) opens
    /// none: the next Sound waits for its thread (`DEV`). The files are FIFOs, whose open waits for a writer.
    #[test]
    fn quit_setup() {
        use std::os::unix::{ffi::OsStrExt, fs::OpenOptionsExt};
        struct Rm(PathBuf);
        impl Drop for Rm {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-fifo-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let _rm = Rm(d.clone());
        let mut tl = edit::Timeline::new(edit::Rate {
            num: 30000,
            den: 1001,
        });
        let files: Vec<PathBuf> = (0..3).map(|i| d.join(format!("{i}.mp4"))).collect();
        for (i, f) in files.iter().enumerate() {
            let c = std::ffi::CString::new(f.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0, "mkfifo");
            tl.files.push(edit::File {
                path: f.to_string_lossy().into(),
                dur: 10.0,
            });
            tl.clips.push(edit::Clip::new(i as u32, 0.0, 4.0));
        }
        tl.fix();
        let m = Mix {
            tl,
            files: files.clone(),
            denoise: None,
        };
        let sh = Sh {
            vol: AtomicU32::new(0),
            quit: AtomicBool::new(true),
            on: AtomicU8::new(0),
            ..sh()
        };
        let mut cv = Conv::new(R as u32, 2, R as u32, 2).unwrap();
        let rg = ring(&cv);
        thread::scope(|sc| {
            let t = Instant::now();
            let h = sc.spawn(|| run(&sh, &rg, &m, 0.0, &mut cv));
            while !h.is_finished() && t.elapsed() < Duration::from_secs(2) {
                thread::sleep(Duration::from_millis(5));
            }
            let s = t.elapsed().as_secs_f64();
            let quick = h.is_finished();
            // one that opens them anyway goes on past each FIFO a writer opens
            while !h.is_finished() {
                for f in &files {
                    let mut w = std::fs::OpenOptions::new();
                    drop(w.write(true).custom_flags(libc::O_NONBLOCK).open(f));
                }
                thread::sleep(Duration::from_millis(1));
            }
            h.join().unwrap().unwrap();
            eprintln!("mixer told to quit first: back after {:.1} ms", s * 1e3);
            assert!(quick, "it opened the clips' files");
        });
    }

    /// A mixer that fails (the music file gone) says why and plays silence on, so the clock keeps going (on the
    /// default output device at volume 0; without one the wall clock runs anyway).
    #[test]
    fn fail() {
        let mut tl = edit::Timeline::new(edit::Rate {
            num: 30000,
            den: 1001,
        });
        let gone = PathBuf::from("/nowhere/music.wav");
        tl.files.push(edit::File {
            path: gone.to_string_lossy().into(),
            dur: 10.0,
        });
        tl.out.music = Some(edit::Music {
            file: 0,
            offset: 0.0,
            gain: 1.0,
            duck: true,
        });
        let m = Mix {
            tl,
            files: vec![gone],
            denoise: None,
        };
        let s = Sound::start(m, 1.0, 0.0);
        let t = Instant::now();
        while s.sh.on.load(Acquire) == 0 {
            assert!(t.elapsed().as_secs() < 5, "the device never opened");
            thread::sleep(Duration::from_millis(1));
        }
        if !s.live() {
            eprintln!("no output device: skipped");
            return;
        }
        let e = loop {
            if let Some(e) = s.err() {
                break e;
            }
            assert!(t.elapsed().as_secs() < 5, "no error");
            thread::sleep(Duration::from_millis(10));
        };
        let a = s.now();
        thread::sleep(Duration::from_millis(400));
        let b = s.now();
        eprintln!("failed mixer: {e}; the clock went {a:.3} → {b:.3} s in 0.4 s");
        assert!(b - a > 0.25, "the clock holds at {a}");
    }

    /// Play, a seek while playing and a denoise change start a Sound and drop the last one: neither waits for the
    /// device, which the sound's thread opens and closes (on the UI thread a start took 7–8 ms). Of ten started in a
    /// row (a slider dragged) the last plays (at volume 0) and its clock runs from its start.
    #[test]
    fn quick() {
        let tl = edit::Timeline::new(edit::Rate {
            num: 30000,
            den: 1001,
        });
        let (mut ms, mut s) = (vec![], None);
        for _ in 0..10 {
            let m = Mix {
                tl: tl.clone(),
                files: vec![],
                denoise: None,
            };
            let t = Instant::now();
            // the last one dropped here too
            s = Some(Sound::start(m, 1.0, 0.0));
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let s = s.unwrap();
        let t = Instant::now();
        while s.sh.on.load(Acquire) == 0 {
            assert!(t.elapsed().as_secs() < 5, "the device never opened");
            thread::sleep(Duration::from_millis(1));
        }
        let open = t.elapsed().as_secs_f64() * 1e3;
        ms.sort_by(f64::total_cmp);
        let a = s.now();
        thread::sleep(Duration::from_millis(400));
        let b = s.now();
        eprintln!(
            "start and drop ms: median {:.3}, max {:.3}; the last on the device {} {open:.1} ms later, its clock {a:.3} → {b:.3} s in 0.4 s",
            ms[ms.len() / 2],
            ms[ms.len() - 1],
            s.live()
        );
        assert!(ms[ms.len() / 2] < 2.0, "{ms:?}");
        assert!(b - a > 0.25, "the clock holds at {a}");
    }
}
