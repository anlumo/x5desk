//! Music import: MP3, FLAC, WAV, Ogg Vorbis and AAC (ADTS) through symphonia, M4A / MP4 AAC through our demuxer
//! and symphonia's AAC decoder (symphonia 0.5 ignores MP4 edit lists, which carry the encoder delay); decoded to
//! interleaved f32 stereo at 48 kHz (mono plays on both sides, more channels mix down: centres and the other
//! sides at −3 dB as in ITU-R BS.775, no LFE, so peaks may pass 1.0). Gapless: the MP3 and Vorbis decoders trim the
//! delay and padding the container gives, the edit list trims M4A (fragmented too). HE-AAC plays its AAC-LC core
//! (symphonia 0.5 has no SBR): in time, but band-limited to a quarter of the output rate.

use crate::{
    audio::Aac,
    mp4::{Codec, Mp4},
    resample::Resampler,
};
use std::{fs::File, io, path::Path};
use symphonia::core::{
    audio::{Channels, SampleBuffer},
    codecs::{CODEC_TYPE_NULL, Decoder, DecoderOptions},
    errors::Error,
    formats::{FormatOptions, FormatReader, SeekMode, SeekTo},
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

pub const RATE: u32 = 48000;

fn err(e: Error) -> io::Error {
    match e {
        Error::IoError(e) => e,
        e => io::Error::new(io::ErrorKind::InvalidData, e),
    }
}

enum Src {
    /// `tick`: source frames per time-base tick; output starts at source frame `from` (a seek target).
    Sym {
        fmt: Box<dyn FormatReader>,
        dec: Box<dyn Decoder>,
        id: u32,
        tick: f64,
        sb: Option<SampleBuffer<f32>>,
        from: u64,
    },
    /// Decoded frames (track ticks scaled by [`scale`]): next sample `i` starts at `pos`; output runs from `from`
    /// (presentation start or seek target) to `end`; `first` = presentation start.
    Mp4 {
        m: Mp4,
        t: usize,
        aac: Aac,
        i: usize,
        pkt: Vec<u8>,
        pos: u64,
        from: u64,
        first: u64,
        end: u64,
    },
}

pub struct Music {
    src: Src,
    /// Source rate and channels.
    pub rate: u32,
    pub ch: usize,
    rs: Resampler,
    /// One packet as stereo at the source rate; resampled output not handed out yet (from `qi`).
    tmp: Vec<f32>,
    q: Vec<f32>,
    qi: usize,
    eof: bool,
    /// Seconds (0 when the container does not say).
    pub duration: f64,
}

/// `x` in units of `to` per second, from units of `from` per second. Track ticks to decoded frames and back: the
/// timescale need not be the decoder's rate (HE-AAC counts at the SBR output rate while symphonia decodes the LC core
/// at half of it; a remux from a transport stream keeps 90 kHz).
fn scale(x: u64, to: u32, from: u32) -> u64 {
    (x as u128 * to as u128 / from as u128) as u64
}

/// An AAC track of an MP4 with the presentation range its edit list gives (priming, padding).
fn mp4(path: &Path) -> Option<Src> {
    let m = Mp4::open(path).ok()?;
    let t = m
        .tracks
        .iter()
        .position(|t| matches!(t.codec, Codec::Aac { .. }))?;
    let k = &m.tracks[t];
    let Codec::Aac { asc, .. } = &k.codec else {
        unreachable!()
    };
    // media time of the first sample: 0 but in a fragment that starts later (`tfdt`)
    let d0 = k.samples.first().map_or(0, |s| s.dts.max(0) as u64);
    let total = d0 + k.samples.iter().map(|s| s.dur as u64).sum::<u64>();
    let (first, end) = match k.edits.iter().find(|e| e.start >= 0) {
        // a zero duration (fragmented files, whose length the `moov` does not know) runs to the end
        Some(e) if e.dur == 0 => (e.start as u64, total),
        Some(e) => {
            let len = (e.dur as u128 * k.timescale as u128 / m.timescale.max(1) as u128) as u64;
            (e.start as u64, (e.start as u64 + len).min(total))
        }
        None => (d0, total),
    };
    let aac = Aac::new(asc).ok()?;
    let [pos, first, end] = [d0, first, end].map(|x| scale(x, aac.rate, k.timescale));
    Some(Src::Mp4 {
        aac,
        i: 0,
        pkt: vec![],
        pos,
        from: first,
        first,
        end,
        m,
        t,
    })
}

impl Music {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Music> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if matches!(ext.as_str(), "m4a" | "mp4" | "m4b" | "mov" | "3gp")
            && let Some(src) = mp4(path)
        {
            let Src::Mp4 {
                aac, first, end, ..
            } = &src
            else {
                unreachable!()
            };
            let (rate, ch) = (aac.rate, aac.ch as usize);
            let duration = (end - first) as f64 / rate as f64;
            return Music::with(src, rate, ch, duration);
        }
        let mut hint = Hint::new();
        hint.with_extension(&ext);
        let mss = MediaSourceStream::new(Box::new(File::open(path)?), Default::default());
        let opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let fmt = symphonia::default::get_probe()
            .format(&hint, mss, &opts, &MetadataOptions::default())
            .map_err(err)?
            .format;
        let t = fmt
            .tracks()
            .iter()
            .find(|t| {
                t.codec_params.codec != CODEC_TYPE_NULL && t.codec_params.sample_rate.is_some()
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no audio track"))?;
        let p = &t.codec_params;
        let rate = p.sample_rate.unwrap();
        let tick = p
            .time_base
            .map_or(1.0, |b| b.numer as f64 / b.denom as f64 * rate as f64);
        let (id, ch) = (t.id, p.channels.map_or(0, |c| c.count()));
        let duration = p.n_frames.map_or(0.0, |n| n as f64 * tick / rate as f64);
        let dec = symphonia::default::get_codecs()
            .make(p, &DecoderOptions::default())
            .map_err(err)?;
        Music::with(
            Src::Sym {
                fmt,
                dec,
                id,
                tick,
                sb: None,
                from: 0,
            },
            rate,
            ch,
            duration,
        )
    }

    fn with(src: Src, rate: u32, ch: usize, duration: f64) -> io::Result<Music> {
        Ok(Music {
            src,
            rate,
            ch,
            rs: Resampler::new(rate, RATE, 2)?,
            tmp: vec![],
            q: vec![],
            qi: 0,
            eof: false,
            duration,
        })
    }

    /// Fills `out` (interleaved stereo, 48 kHz) and returns the frames written: fewer only at the end.
    pub fn read(&mut self, out: &mut [f32]) -> io::Result<usize> {
        let want = out.len() / 2 * 2;
        let mut n = 0;
        while n < want {
            if self.qi < self.q.len() {
                let k = (self.q.len() - self.qi).min(want - n);
                out[n..n + k].copy_from_slice(&self.q[self.qi..self.qi + k]);
                (n, self.qi) = (n + k, self.qi + k);
                continue;
            }
            if self.eof {
                break;
            }
            self.qi = 0;
            self.q.clear();
            self.tmp.clear();
            // one packet into `tmp` as stereo; false at the end
            let more = match &mut self.src {
                Src::Sym {
                    fmt,
                    dec,
                    id,
                    tick,
                    sb,
                    from,
                } => loop {
                    let pkt = match fmt.next_packet() {
                        Ok(p) => p,
                        // the end, or a chained stream with new parameters: stop there
                        Err(Error::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                            break false;
                        }
                        Err(Error::ResetRequired) => break false,
                        Err(e) => return Err(err(e)),
                    };
                    if pkt.track_id() != *id {
                        continue;
                    }
                    let d = match dec.decode(&pkt) {
                        Ok(d) => d,
                        Err(Error::DecodeError(_)) => continue, // a damaged packet: drop it, keep going
                        Err(e) => return Err(err(e)),
                    };
                    let c = d.spec().channels.count();
                    let b =
                        sb.get_or_insert_with(|| SampleBuffer::new(d.capacity() as u64, *d.spec()));
                    if b.capacity() < d.capacity() * c {
                        *b = SampleBuffer::new(d.capacity() as u64, *d.spec());
                    }
                    // stereo weights per channel (in mask order, as the buffer interleaves them)
                    let (mut w, cs) = ([[0f32; 2]; 32], d.spec().channels);
                    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
                    let l = Channels::REAR_LEFT
                        | Channels::FRONT_LEFT_CENTRE
                        | Channels::SIDE_LEFT
                        | Channels::TOP_FRONT_LEFT
                        | Channels::TOP_REAR_LEFT
                        | Channels::REAR_LEFT_CENTRE
                        | Channels::FRONT_LEFT_WIDE
                        | Channels::FRONT_LEFT_HIGH;
                    let r = Channels::REAR_RIGHT
                        | Channels::FRONT_RIGHT_CENTRE
                        | Channels::SIDE_RIGHT
                        | Channels::TOP_FRONT_RIGHT
                        | Channels::TOP_REAR_RIGHT
                        | Channels::REAR_RIGHT_CENTRE
                        | Channels::FRONT_RIGHT_WIDE
                        | Channels::FRONT_RIGHT_HIGH;
                    for (w, p) in w.iter_mut().zip(cs.iter()) {
                        *w = match p {
                            _ if c == 1 => [1.0, 1.0],
                            Channels::FRONT_LEFT => [1.0, 0.0],
                            Channels::FRONT_RIGHT => [0.0, 1.0],
                            Channels::LFE1 | Channels::LFE2 => [0.0, 0.0],
                            _ if l.contains(p) => [H, 0.0],
                            _ if r.contains(p) => [0.0, H],
                            _ => [H, H],
                        };
                    }
                    b.copy_interleaved_ref(d);
                    // the packet's time places its output (a Vorbis decoder fresh from a seek outputs nothing for
                    // its first packet, so counting decoded frames from the seek point would run early)
                    let s = b.samples();
                    let a = from
                        .saturating_sub((pkt.ts as f64 * *tick).round() as u64)
                        .min((s.len() / c) as u64) as usize;
                    self.ch = c;
                    // 0 + x·1 + y·0 = x: mono and stereo pass bit-exact
                    self.tmp.extend(s[a * c..].chunks_exact(c).flat_map(|f| {
                        let mut o = [0f32; 2];
                        for (x, w) in f.iter().zip(&w) {
                            o = [o[0] + x * w[0], o[1] + x * w[1]];
                        }
                        o
                    }));
                    break true;
                },
                Src::Mp4 {
                    m,
                    t,
                    aac,
                    i,
                    pkt,
                    pos,
                    from,
                    end,
                    ..
                } => {
                    let k = &m.tracks[*t];
                    match k.samples.get(*i) {
                        Some(sm) if *pos < *end => {
                            m.read(sm, pkt)?;
                            let (c, p0) = (aac.ch as usize, *pos);
                            *i += 1;
                            let s = match aac.decode(pkt) {
                                Ok(s) => s,
                                // a damaged frame: silence of its length keeps the timing
                                Err(_) => {
                                    *pos += scale(sm.dur as u64, aac.rate, k.timescale);
                                    let (a, b) = (p0.max(*from), (*pos).min(*end));
                                    self.tmp.resize(2 * b.saturating_sub(a) as usize, 0.0);
                                    self.rs.run(&self.tmp, &mut self.q);
                                    continue;
                                }
                            };
                            // the decoded length places the next frame: scaled ticks may round apart
                            *pos += (s.len() / c) as u64;
                            let (a, b) =
                                ((*from).max(p0) - p0, (*end).min(*pos).saturating_sub(p0));
                            if a < b {
                                self.tmp.extend(
                                    s[a as usize * c..b as usize * c]
                                        .chunks_exact(c)
                                        .flat_map(|f| [f[0], f[(c > 1) as usize]]),
                                );
                            }
                            true
                        }
                        _ => false,
                    }
                }
            };
            if more {
                self.rs.run(&self.tmp, &mut self.q);
            } else {
                self.eof = true;
                self.rs.finish(&mut self.q);
            }
        }
        Ok(n / 2)
    }

    /// Continues from `secs` (to the nearest source frame).
    pub fn seek(&mut self, secs: f64) -> io::Result<()> {
        self.rs.reset();
        (self.qi, self.eof) = (0, false);
        self.q.clear();
        let f = (secs.max(0.0) * self.rate as f64).round() as u64;
        match &mut self.src {
            Src::Sym {
                fmt,
                dec,
                id,
                tick,
                from,
                ..
            } => {
                // half a second early, then decode up to the target: symphonia 0.5 lands wrong when it seeks into
                // the last quarter second of an MP3 (measured: 0.45 off at 0.1 s before the end)
                let to = SeekTo::TimeStamp {
                    ts: (f.saturating_sub(self.rate as u64 / 2) as f64 / *tick).round() as u64,
                    track_id: *id,
                };
                *from = f;
                match fmt.seek(SeekMode::Accurate, to) {
                    Ok(_) => dec.reset(),
                    // past the end
                    Err(Error::SeekError(_)) => self.eof = true,
                    Err(Error::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                        self.eof = true
                    }
                    Err(e) => return Err(err(e)),
                }
            }
            Src::Mp4 {
                m,
                t,
                aac,
                i,
                pos,
                from,
                first,
                ..
            } => {
                // one frame early: the frame before overlaps into the target's (its own output is dropped)
                let k = &m.tracks[*t];
                *from = *first + f;
                *i = k
                    .at(scale(*from, k.timescale, aac.rate) as i64)
                    .saturating_sub(1);
                *pos = k
                    .samples
                    .get(*i)
                    .map_or(0, |s| scale(s.dts.max(0) as u64, aac.rate, k.timescale));
                aac.reset();
            }
        }
        Ok(())
    }
}

/// A whole file at 48 kHz stereo.
pub fn load(path: impl AsRef<Path>) -> io::Result<Vec<f32>> {
    let mut m = Music::open(path)?;
    let mut v = Vec::with_capacity((m.duration * RATE as f64) as usize * 2 + 8192);
    let mut b = vec![0f32; 2 * 8192];
    loop {
        let n = m.read(&mut b)?;
        v.extend_from_slice(&b[..2 * n]);
        if n < 8192 {
            return Ok(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        aacenc::Enc,
        mux::Mux,
        tests::{out, tool},
    };
    use std::f64::consts::TAU;

    /// The test signal at time t: left a 440 Hz tone and a slow chirp, right a 660 Hz tone.
    fn sig(t: f64) -> [f64; 2] {
        [
            (TAU * 440.0 * t).sin() * 0.3 + (TAU * (200.0 * t + 300.0 * t * t)).sin() * 0.2,
            (TAU * 660.0 * t).sin() * 0.4,
        ]
    }

    /// Each format, made by ffmpeg from the same 3 s at 44.1 kHz: length, offset and SNR against the exact signal
    /// at 48 kHz.
    #[test]
    fn formats() {
        let d = out().join("music");
        std::fs::create_dir_all(&d).unwrap();
        let n = 44100 * 3;
        let pcm: Vec<u8> = (0..n)
            .flat_map(|i| sig(i as f64 / 44100.0))
            .flat_map(|v| ((v * 32767.0).round() as i16).to_le_bytes())
            .collect();
        let raw = d.join("sig.s16");
        std::fs::write(&raw, pcm).unwrap();
        let r = raw.to_str().unwrap();
        let mono = d.join("mono.wav");
        tool(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-i",
                r,
                "-ac",
                "1",
                mono.to_str().unwrap(),
            ],
        );
        for (name, args, min) in [
            ("sig.wav", &[][..], 85.0),
            ("sig.flac", &[][..], 85.0),
            ("sig.mp3", &["-c:a", "libmp3lame", "-b:a", "192k"][..], 25.0),
            ("sig.ogg", &["-c:a", "libvorbis", "-q:a", "5"][..], 25.0),
            ("sig.m4a", &["-c:a", "aac", "-b:a", "192k"][..], 25.0),
            ("sig.aac", &["-c:a", "aac", "-b:a", "192k"][..], 25.0),
            // fragmented, without an edit list (FFmpeg writes none there)
            (
                "frag.m4a",
                &[
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "frag_keyframe+empty_moov",
                ][..],
                25.0,
            ),
            // fragmented DASH: its edit list has a zero duration (to the end)
            (
                "dash.m4a",
                &[
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-f",
                    "dash",
                    "-seg_duration",
                    "1",
                    "-single_file",
                    "1",
                    "-single_file_name",
                    "dash.m4a",
                ][..],
                25.0,
            ),
        ] {
            let f = d.join(name);
            let o = if name == "dash.m4a" {
                d.join("dash.mpd")
            } else {
                f.clone()
            };
            let mut a = vec![
                "-v", "error", "-y", "-f", "s16le", "-ar", "44100", "-ac", "2", "-i", r,
            ];
            a.extend_from_slice(args);
            a.push(o.to_str().unwrap());
            tool("ffmpeg", &a);
            let y = load(&f).unwrap();
            let len = y.len() / 2;
            // offset against the exact signal (search ±3000 frames), then SNR over the middle
            let ex: Vec<[f64; 2]> = (0..len + 6000)
                .map(|i| sig((i as f64 - 3000.0) / 48000.0))
                .collect();
            let at = |o: usize| {
                (24000..len - 24000)
                    .step_by(7)
                    .map(|i| y[2 * i] as f64 * ex[i + o][0])
                    .sum::<f64>()
            };
            let o = (0..6000).max_by(|&a, &b| at(a).total_cmp(&at(b))).unwrap();
            let (mut s, mut e) = (0f64, 0f64);
            for i in 24000..len - 24000 {
                for c in 0..2 {
                    s += ex[i + o][c].powi(2);
                    e += (y[2 * i + c] as f64 - ex[i + o][c]).powi(2);
                }
            }
            let snr = 10.0 * (s / e).log10();
            println!(
                "{name}: {len} frames (exact {}), offset {} frames, SNR {snr:.1} dB",
                n * 48000 / 44100,
                o as i64 - 3000
            );
            assert!(snr > min, "{name}");
            // ADTS has no delay field and the fragmented file no edit list: their priming frame stays; ADTS plays
            // all 131 frames, the last sample's duration in the fragment drops the padding (FFmpeg decodes all 131
            // frames there too, while ffprobe gives the duration as ours)
            if let Some(src) = match name {
                "sig.aac" => Some(131 * 1024),
                "frag.m4a" => Some(n + 1024),
                _ => None,
            } {
                assert!(
                    o.abs_diff(3000 - 1115) <= 1 && len == (src * 48000usize).div_ceil(44100),
                    "{name}: not gapless"
                );
            } else {
                // the DASH file's edit cuts the priming and runs to its samples' end, as FFmpeg and mpv play it: its
                // remainder FFmpeg 9.1 states only in `sidx` and the MPD (earlier versions shortened the last sample)
                let end = match name {
                    "dash.m4a" => {
                        let t = &Mp4::open(&f).unwrap().tracks[0];
                        let total: u64 = t.samples.iter().map(|s| s.dur as u64).sum();
                        (total - t.edits[0].start as u64) as usize
                    }
                    _ => n,
                };
                assert!(
                    o == 3000 && len.abs_diff(end * 48000 / 44100) <= 1,
                    "{name}: gapless"
                );
            }
        }
        let y = load(&mono).unwrap();
        assert!(y.chunks(2).all(|f| f[0] == f[1]) && y.len() / 2 == n * 48000 / 44100);
    }

    /// More channels mix down: each 5.1 channel carries its own tone; left takes front left as it is, centre and the
    /// left surround (back or side) at −3 dB, right the mirror image, the LFE neither.
    #[test]
    fn downmix() {
        let d = out().join("music");
        std::fs::create_dir_all(&d).unwrap();
        // FL FR FC LFE (B|S)L (B|S)R
        let fq = [300.0, 500.0, 700.0, 60.0, 900.0, 1100.0];
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let want = [
            [1.0, 0.0],
            [0.0, 1.0],
            [h, h],
            [0.0, 0.0],
            [h, 0.0],
            [0.0, h],
        ];
        let srcs: Vec<String> = fq
            .iter()
            .map(|f| format!("sine=f={f}:d=2:r=48000"))
            .collect();
        for (name, layout, s, args, tol) in [
            ("s51.wav", "5.1(side)", "S", &[][..], 1e-3),
            ("s51.flac", "5.1", "B", &[][..], 1e-3),
            (
                "s51.ogg",
                "5.1",
                "B",
                &["-c:a", "libvorbis", "-q:a", "6"][..],
                0.02,
            ),
        ] {
            let f = d.join(name);
            // by name: join would match the mono inputs (FC) to the centre first
            let fc = format!(
                "[0][1][2][3][4][5]join=inputs=6:channel_layout={layout}:\
                 map=0.0-FL|1.0-FR|2.0-FC|3.0-LFE|4.0-{s}L|5.0-{s}R[a]"
            );
            let mut a = vec!["-v", "error", "-y"];
            for s in &srcs {
                a.extend_from_slice(&["-f", "lavfi", "-i", s]);
            }
            a.extend_from_slice(&["-filter_complex", &fc, "-map", "[a]"]);
            a.extend_from_slice(args);
            a.push(f.to_str().unwrap());
            tool("ffmpeg", &a);
            let y = load(&f).unwrap();
            // each tone's amplitude per side over 1 s (whole periods), against its own 1/8
            let mut worst = 0f64;
            for (k, &fr) in fq.iter().enumerate() {
                for c in 0..2 {
                    let (mut re, mut im) = (0f64, 0f64);
                    for i in 24000..72000 {
                        let p = TAU * fr * i as f64 / 48000.0;
                        (re, im) = (
                            re + y[2 * i + c] as f64 * p.cos(),
                            im + y[2 * i + c] as f64 * p.sin(),
                        );
                    }
                    worst = worst.max((re.hypot(im) / 24000.0 * 8.0 - want[k][c]).abs());
                }
            }
            println!("{name}: downmix gains within {worst:.1e} of BS.775");
            assert!(worst < tol, "{name}");
        }
    }

    /// Seeking gives the same samples as decoding through (FLAC, WAV, MP3 and Vorbis through symphonia, M4A
    /// through our demuxer, also fragmented), also close to the end.
    #[test]
    fn seek() {
        let d = out().join("music");
        std::fs::create_dir_all(&d).unwrap();
        let pcm: Vec<u8> = (0..44100 * 4)
            .flat_map(|i| sig(i as f64 / 44100.0))
            .flat_map(|v| ((v * 32767.0).round() as i16).to_le_bytes())
            .collect();
        let raw = d.join("seek.s16");
        std::fs::write(&raw, pcm).unwrap();
        // no noise substitution in AAC: its random fill differs with the decode history
        let aac = ["-c:a", "aac", "-b:a", "160k", "-aac_pns", "0"];
        let frag = [
            &aac[..],
            &["-movflags", "frag_keyframe", "-frag_duration", "500000"],
        ]
        .concat();
        // the fragmented file has no edit list: 174 frames of 1024 play
        for (name, args, dur) in [
            ("seek.flac", &[][..], 4.0),
            ("seek.wav", &[][..], 4.0),
            ("seek.m4a", &aac[..], 4.0),
            ("seek_frag.m4a", &frag[..], 174.0 * 1024.0 / 44100.0),
            ("seek.mp3", &["-c:a", "libmp3lame", "-b:a", "160k"][..], 4.0),
            ("seek.ogg", &["-c:a", "libvorbis", "-q:a", "5"][..], 4.0),
        ] {
            let f = d.join(name);
            let mut a = vec![
                "-v",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-i",
                raw.to_str().unwrap(),
            ];
            a.extend_from_slice(args);
            a.push(f.to_str().unwrap());
            tool("ffmpeg", &a);
            let all = load(&f).unwrap();
            let mut m = Music::open(&f).unwrap();
            assert!((m.duration - dur).abs() < 0.03, "{name}: {}", m.duration);
            let mut worst = 0f32;
            for at in [2.5, 0.75, 3.8, 3.9, 0.0] {
                m.seek(at).unwrap();
                let mut b = vec![0f32; 2 * 4800];
                assert_eq!(m.read(&mut b).unwrap(), 4800);
                let i = (at * 48000.0).round() as usize;
                // after the resampler's start (half its taps) the samples match the straight decode
                let k = if at > 0.0 { 200 } else { 0 };
                let e = b[2 * k..]
                    .iter()
                    .zip(&all[2 * (i + k)..])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                worst = worst.max(e);
            }
            println!("{name}: seeks match the straight decode within {worst:.1e}");
            assert!(worst == 0.0, "{name}");
            m.seek(10.0).unwrap();
            assert_eq!(m.read(&mut [0f32; 64]).unwrap(), 0);
        }
    }

    /// A track's timescale need not be the decoder's rate: HE-AAC counts at the SBR output rate, twice the LC core
    /// that symphonia decodes; a remux from a transport stream keeps 90 kHz. The same frames muxed with ticks at
    /// 48, 96 and 90 kHz play the same: length, duration, priming and padding trims, seeks.
    #[test]
    fn timescales() {
        let d = out().join("music");
        std::fs::create_dir_all(&d).unwrap();
        let pcm: Vec<f32> = (0..48000 * 3)
            .flat_map(|i| sig(i as f64 / 48000.0))
            .map(|v| v as f32)
            .collect();
        let mut e = Enc::new(48000, 2, 160000).unwrap();
        let mut aus = vec![];
        let mut put = |a: &[u8], pts: i64, dur: u32| aus.push((a.to_vec(), pts, dur));
        e.push(&pcm, &mut put);
        e.finish(&mut put);
        let mut base = vec![];
        for ts in [48000u32, 96000, 90000] {
            let f = d.join(format!("ts{ts}.m4a"));
            let mut x = Mux::create(&f).unwrap();
            let codec = Codec::Aac {
                asc: e.asc().to_vec(),
                rate: 48000,
                ch: 2,
            };
            let t = x.track(codec, ts, 0, 0, false).unwrap();
            for (a, pts, dur) in &aus {
                let p = pts * ts as i64 / 48000;
                x.write(t, a, p, p, (*dur as u64 * ts as u64 / 48000) as u32, true)
                    .unwrap();
            }
            x.finish(false).unwrap();
            let y = load(&f).unwrap();
            let mut m = Music::open(&f).unwrap();
            if ts == 48000 {
                base = y.clone();
            }
            let mut worst = 0f32;
            for at in [2.5, 0.75, 1.0, 2.9] {
                m.seek(at).unwrap();
                let mut b = vec![0f32; 2 * 2400];
                assert_eq!(m.read(&mut b).unwrap(), 2400);
                let i = (at * 48000.0) as usize;
                worst = b[400..]
                    .iter()
                    .zip(&base[2 * i + 400..])
                    .map(|(a, b)| (a - b).abs())
                    .fold(worst, f32::max);
            }
            println!(
                "timescale {ts}: {} frames, duration {:.4} s, same as at 48 kHz: {}, seeks within {worst:.1e}",
                y.len() / 2,
                m.duration,
                y == base
            );
            assert!(y.len() == 2 * 144000 && y == base && m.duration == 3.0 && worst == 0.0);
        }
    }
}
