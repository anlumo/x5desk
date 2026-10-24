//! Editor mixer, 48 kHz stereo: tracks of clip audio and music with gain envelopes, fades and mute; music ducks
//! under clip audio (sidechain on the clip tracks' level, with hold); a look-ahead limiter on the master. For each
//! block the caller fills every track's `buf` with that track's audio, already at 48 kHz stereo with its speed
//! applied ([`crate::stretch`], [`crate::resample`], [`crate::music`]); the mixer does the rest.

use std::collections::VecDeque;

pub const RATE: f64 = 48000.0;
/// Limiter look-ahead (5 ms): output frame i of a block is mixer time `t` + (i − LATENCY) / RATE.
pub const LATENCY: usize = 240;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Drives the ducking.
    Clip,
    /// Ducked under clip audio.
    Music,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub kind: Kind,
    pub gain: f32,
    /// Gain envelope on mixer time (s, linear gain), linear between points, constant beyond; empty: 1.
    pub env: Vec<(f64, f32)>,
    /// Where the track plays on mixer time (s); silent outside. Fade-in and fade-out lengths (s) at its ends
    /// (S-curves).
    pub span: [f64; 2],
    pub fade: [f64; 2],
    pub mute: bool,
    /// This block's input (interleaved stereo); missing samples are silence.
    pub buf: Vec<f32>,
}

impl Track {
    pub fn new(kind: Kind, span: [f64; 2]) -> Track {
        Track {
            kind,
            gain: 1.0,
            env: vec![],
            span,
            fade: [0.0; 2],
            mute: false,
            buf: vec![],
        }
    }
}

/// Ducking: the music gain `depth` while the clip mix's mean square (over ~20 ms) is above `thr`; it falls in
/// `attack` s, holds `hold` s after the level drops and comes back in `release` s (time constants). `depth` 1 turns it
/// off.
#[derive(Clone, Copy, Debug)]
pub struct Duck {
    pub depth: f32,
    pub thr: f32,
    pub attack: f32,
    pub hold: f32,
    pub release: f32,
}

pub struct Mixer {
    pub tracks: Vec<Track>,
    pub master: f32,
    pub duck: Duck,
    /// Limiter ceiling (linear peak); release time (s).
    pub ceil: f32,
    pub release: f32,
    /// Sidechain mean square, duck gain, hold frames left.
    lev: f32,
    dg: f32,
    held: u32,
    /// Limiter: delay line (LATENCY frames), sliding minimum of the wanted gains over LATENCY + 1 frames (index,
    /// gain), released gain, moving average over LATENCY frames (ring and sum), frames seen.
    dl: Vec<f32>,
    mins: VecDeque<(u64, f32)>,
    rel: f32,
    avg: Vec<f32>,
    sum: f64,
    n: u64,
    /// Gains of the current block per frame (clip / music track being mixed).
    g: Vec<f32>,
}

impl Default for Mixer {
    fn default() -> Mixer {
        Mixer::new()
    }
}

impl Mixer {
    pub fn new() -> Mixer {
        let mut m = Mixer {
            tracks: vec![],
            master: 1.0,
            // −10.5 dB under speech louder than −45 dBFS RMS
            duck: Duck {
                depth: 0.3,
                thr: 10f32.powf(-45.0 / 10.0),
                attack: 0.1,
                hold: 0.3,
                release: 0.5,
            },
            ceil: 10f32.powf(-1.0 / 20.0),
            release: 0.1,
            lev: 0.0,
            dg: 1.0,
            held: 0,
            dl: vec![0.0; 2 * LATENCY],
            mins: VecDeque::with_capacity(LATENCY + 2),
            rel: 1.0,
            avg: vec![1.0; LATENCY],
            sum: LATENCY as f64,
            n: 0,
            g: Vec::with_capacity(8192),
        };
        m.reset();
        m
    }

    /// Clears the ducking and limiter state (seek).
    pub fn reset(&mut self) {
        (self.lev, self.dg, self.held, self.rel, self.n) = (0.0, 1.0, 0, 1.0, 0);
        self.dl.fill(0.0);
        self.mins.clear();
        self.avg.fill(1.0);
        self.sum = LATENCY as f64;
    }

    /// Mixes the block of input that starts at mixer time `t` (s) into `out` (interleaved stereo, the tracks' `buf`
    /// cover the same frames); the output lags by [`LATENCY`] frames.
    pub fn run(&mut self, t: f64, out: &mut [f32]) {
        let nf = out.len() / 2;
        out.fill(0.0);
        // clip tracks first (they make the sidechain), music after the ducking pass
        for pass in [Kind::Clip, Kind::Music] {
            if pass == Kind::Music {
                let d = self.duck;
                let (ca, cr, cl) = (coef(d.attack), coef(d.release), coef(0.02));
                let hold = (d.hold as f64 * RATE) as u32;
                // sidechain: the clip mix so far; the duck gain per frame goes to `g`
                self.g.clear();
                for f in out.chunks_exact(2) {
                    self.lev += ((f[0] * f[0] + f[1] * f[1]) * 0.5 - self.lev) * cl;
                    self.held = if self.lev > d.thr {
                        hold
                    } else {
                        self.held.saturating_sub(1)
                    };
                    let want = if self.held > 0 { d.depth } else { 1.0 };
                    self.dg += (want - self.dg) * if want < self.dg { ca } else { cr };
                    self.g.push(self.dg);
                }
            }
            for k in &self.tracks {
                if k.mute || k.kind != pass {
                    continue;
                }
                let [s0, s1] = k.span;
                let (t0, t1) = (t, t + nf as f64 / RATE);
                if t1 <= s0 || t0 >= s1 {
                    continue;
                }
                let f = |x: f64, l: f64| {
                    if l > 0.0 && x < l {
                        0.5 - 0.5 * (std::f64::consts::PI * x / l).cos()
                    } else {
                        1.0
                    }
                };
                let mut e = k.env.partition_point(|p| p.0 <= t0);
                for (i, (o, x)) in out
                    .chunks_exact_mut(2)
                    .zip(k.buf.chunks_exact(2))
                    .enumerate()
                {
                    let tt = t0 + i as f64 / RATE;
                    if tt < s0 || tt >= s1 {
                        continue;
                    }
                    while e < k.env.len() && k.env[e].0 <= tt {
                        e += 1;
                    }
                    let env = match (e, k.env.len()) {
                        (_, 0) => 1.0,
                        (0, _) => k.env[0].1,
                        (e, l) if e == l => k.env[l - 1].1,
                        (e, _) => {
                            let ((a, ga), (b, gb)) = (k.env[e - 1], k.env[e]);
                            ga + (gb - ga) * ((tt - a) / (b - a)) as f32
                        }
                    };
                    let fd = (f(tt - s0, k.fade[0]) * f(s1 - tt, k.fade[1])) as f32;
                    let g = k.gain * env * fd * if pass == Kind::Music { self.g[i] } else { 1.0 };
                    o[0] += g * x[0];
                    o[1] += g * x[1];
                }
            }
        }
        // master and limiter: the gain at each output frame is the moving average over LATENCY frames of the
        // released sliding minimum (over LATENCY + 1 frames) of the gains the input wants, so it never exceeds the
        // wanted gain of the frame it scales, which left the delay line LATENCY frames after it came in
        let rc = coef(self.release);
        for f in out.chunks_exact_mut(2) {
            let (l, r) = (f[0] * self.master, f[1] * self.master);
            let p = l.abs().max(r.abs());
            let want = if p > self.ceil { self.ceil / p } else { 1.0 };
            while self.mins.back().is_some_and(|b| b.1 >= want) {
                self.mins.pop_back();
            }
            self.mins.push_back((self.n, want));
            while self.mins[0].0 + (LATENCY as u64) < self.n {
                self.mins.pop_front();
            }
            self.rel = self.mins[0].1.min(self.rel + (1.0 - self.rel) * rc);
            let j = (self.n % LATENCY as u64) as usize;
            self.sum += (self.rel - self.avg[j]) as f64;
            self.avg[j] = self.rel;
            let g = (self.sum / LATENCY as f64) as f32;
            let (dl, dr) = (self.dl[2 * j], self.dl[2 * j + 1]);
            (self.dl[2 * j], self.dl[2 * j + 1]) = (l, r);
            (f[0], f[1]) = (dl * g, dr * g);
            self.n += 1;
        }
    }
}

/// One-pole coefficient for time constant `s` seconds at 48 kHz.
fn coef(s: f32) -> f32 {
    1.0 - (-1.0 / (s.max(1e-4) as f64 * RATE)).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        aacenc::Enc,
        audio::Aac,
        denoise::{self, Denoise},
        mp4::{Codec, Mp4},
        music::Music,
        mux::Mux,
        stretch::Stretch,
        tests::{out, samples, tool},
    };
    use std::f64::consts::TAU;

    /// Runs `m` over `secs` in blocks of 1000 frames, `fill(track, time, frame)` giving each track's input.
    fn render(m: &mut Mixer, secs: f64, fill: impl Fn(usize, f64) -> [f32; 2]) -> Vec<f32> {
        let n = (secs * RATE) as usize;
        let mut y = vec![0f32; 2 * (n + LATENCY)];
        for (b, o) in y.chunks_mut(2000).enumerate() {
            let t = b as f64 * 1000.0 / RATE;
            for (k, tr) in m.tracks.iter_mut().enumerate() {
                tr.buf.clear();
                for i in 0..o.len() / 2 {
                    tr.buf.extend_from_slice(&fill(k, t + i as f64 / RATE));
                }
            }
            m.run(t, o);
        }
        y.drain(..2 * LATENCY);
        y
    }

    #[test]
    fn envelope_fades_mute() {
        let mut m = Mixer::new();
        let mut a = Track::new(Kind::Clip, [0.5, 2.5]);
        a.env = vec![(1.0, 1.0), (2.0, 0.25)];
        a.fade = [0.2, 0.4];
        a.gain = 0.8;
        let mut b = Track::new(Kind::Clip, [0.0, 3.0]);
        b.mute = true;
        m.tracks = vec![a, b];
        let y = render(
            &mut m,
            3.0,
            |k, _| if k == 0 { [0.5, -0.25] } else { [1.0, 1.0] },
        );
        let at = |t: f64| y[2 * (t * RATE).round() as usize];
        let want = |t: f64| {
            let env = if t < 1.0 {
                1.0
            } else if t > 2.0 {
                0.25
            } else {
                1.0 - 0.75 * (t - 1.0)
            };
            let s = |x: f64, l: f64| {
                if x < l {
                    0.5 - 0.5 * (std::f64::consts::PI * x / l).cos()
                } else {
                    1.0
                }
            };
            if !(0.5..2.5).contains(&t) {
                0.0
            } else {
                0.5 * 0.8 * env * s(t - 0.5, 0.2) * s(2.5 - t, 0.4)
            }
        };
        let mut err = 0f64;
        for i in 0..2900 {
            let t = i as f64 / 1000.0;
            err = err.max((at(t) as f64 - want(t)).abs());
        }
        println!("envelope, fades, mute: max error {err:.1e}");
        assert!(err < 1e-5);
        assert!((y[2 * 72000 + 1] / y[2 * 72000] + 0.5).abs() < 1e-6);
    }

    /// Music under bursts of clip audio: down by the depth after the attack, back after hold + release.
    #[test]
    fn ducking() {
        let mut m = Mixer::new();
        m.tracks = vec![
            Track::new(Kind::Clip, [0.0, 10.0]),
            Track::new(Kind::Music, [0.0, 10.0]),
        ];
        // speech-like bursts 1–2 s and 5–5.5 s (−20 dBFS), music a 300 Hz tone at 0.2
        let y = render(&mut m, 8.0, |k, t| {
            if k == 1 {
                let v = (TAU * 300.0 * t).sin() as f32 * 0.2;
                [v, v]
            } else if (1.0..2.0).contains(&t) || (5.0..5.5).contains(&t) {
                let v = ((TAU * 150.0 * t).sin() * (TAU * 2200.0 * t).sin()) as f32 * 0.14;
                [v, -v]
            } else {
                [0.0; 2]
            }
        });
        // the music level from the sum of channels (the burst is antiphase), as RMS over 20 ms windows
        let lvl = |t: f64| {
            let i = (t * RATE) as usize;
            let s: f64 = (i..i + 960)
                .map(|j| ((y[2 * j] + y[2 * j + 1]) as f64 / 2.0).powi(2))
                .sum();
            (s / 960.0).sqrt() / (0.2 / 2f64.sqrt())
        };
        let pts = [0.5, 1.05, 1.4, 1.9, 2.25, 2.6, 4.5, 5.4, 6.5];
        let v: Vec<f64> = pts.iter().map(|&t| lvl(t)).collect();
        println!(
            "music gain at {pts:?} s: {:?}",
            v.iter().map(|g| format!("{g:.3}")).collect::<Vec<_>>()
        );
        assert!(
            (v[0] - 1.0).abs() < 0.01 && (v[2] - 0.3).abs() < 0.02 && (v[3] - 0.3).abs() < 0.02
        );
        assert!(v[1] > 0.4 && v[1] < 0.95, "attack takes time");
        assert!((v[4] - 0.3).abs() < 0.05, "hold");
        assert!(
            v[5] > 0.35 && v[5] < 0.95 && (v[6] - 1.0).abs() < 0.02 && (v[7] - 0.3).abs() < 0.05
        );
        // off: depth 1
        m.reset();
        m.duck.depth = 1.0;
        let tone = |t: f64| (TAU * 300.0 * t).sin() as f32 * 0.2;
        let y = render(
            &mut m,
            2.0,
            |k, t| if k == 1 { [tone(t); 2] } else { [0.3; 2] },
        );
        let e = (0..96000)
            .map(|i| (y[2 * i] - 0.3 - tone(i as f64 / RATE)).abs())
            .fold(0f32, f32::max);
        assert!(e < 1e-5, "{e}");
    }

    /// The limiter keeps every sample under the ceiling, leaves quiet passages untouched and recovers.
    #[test]
    fn limiter() {
        let mut m = Mixer::new();
        m.tracks = vec![
            Track::new(Kind::Clip, [0.0, 10.0]),
            Track::new(Kind::Clip, [0.0, 10.0]),
        ];
        // two tones summing to peaks near 1.9 between 1 and 1.5 s, a 0.5 tone elsewhere; clicks at 2 s
        let x = |k: usize, t: f64| {
            let a = if (1.0..1.5).contains(&t) {
                0.95
            } else if k == 0 {
                0.5
            } else {
                0.0
            };
            let c = if k == 1 && (2.0..2.0005).contains(&t) {
                3.0
            } else {
                0.0
            };
            let v = (TAU * (300.0 + 37.0 * k as f64) * t).sin() * a + c;
            [v as f32, -v as f32 * 0.5]
        };
        let y = render(&mut m, 3.0, x);
        let peak = y.iter().fold(0f32, |p, v| p.max(v.abs()));
        let mut quiet = 0f32;
        for i in 0..40000 {
            let t = i as f64 / RATE;
            quiet = quiet.max((y[2 * i] - (x(0, t)[0] + x(1, t)[0])).abs());
        }
        let back = (2.6 * RATE) as usize;
        let rec = (back..back + 4800)
            .map(|i| (y[2 * i] - (x(0, i as f64 / RATE)[0] + x(1, i as f64 / RATE)[0])).abs())
            .fold(0f32, f32::max);
        println!(
            "limiter: peak {:.4} (ceiling {:.4}), untouched before: max diff {quiet:.1e}, 0.6 s after the clicks: {rec:.1e}",
            peak, m.ceil
        );
        assert!(peak <= m.ceil * 1.000_01 && quiet < 1e-6 && rec < 1e-3);
    }

    /// The export path end to end: v635's audio denoised and played at 1.5×, over music (a 44.1 kHz file) that ducks
    /// under it and fades out, limited, AAC-encoded and muxed; FFmpeg decodes the file without a message and it lasts
    /// what the timeline says.
    #[test]
    fn export_chain() {
        let Some(dir) = samples() else { return };
        let m = Mp4::open(dir.join("v635.insv")).unwrap();
        let t = m
            .tracks
            .iter()
            .find(|t| matches!(t.codec, Codec::Aac { .. }))
            .unwrap();
        let Codec::Aac { asc, .. } = &t.codec else {
            unreachable!()
        };
        let (mut dec, mut b, mut pcm) = (Aac::new(asc).unwrap(), vec![], vec![]);
        // debug builds run the denoiser at ~70 ms a frame: 2 s there
        let secs = if cfg!(debug_assertions) { 2.0 } else { 5.76 };
        for s in &t.samples[..(secs * RATE / 1024.0) as usize] {
            m.read(s, &mut b).unwrap();
            pcm.extend_from_slice(dec.decode(&b).unwrap());
        }
        // denoise per channel, then drop the denoiser's delay (it is flushed with silence)
        let (mut dn, len) = ([Denoise::new(), Denoise::new()], pcm.len());
        pcm.resize(pcm.len() + 2 * denoise::DELAY, 0.0);
        pcm.resize(
            pcm.len().div_ceil(2 * denoise::FRAME) * 2 * denoise::FRAME,
            0.0,
        );
        for f in pcm.chunks_exact_mut(2 * denoise::FRAME) {
            for (k, d) in dn.iter_mut().enumerate() {
                let mut c: [f32; denoise::FRAME] = std::array::from_fn(|i| f[2 * i + k]);
                d.frame(&mut c);
                for (i, v) in c.into_iter().enumerate() {
                    f[2 * i + k] = v;
                }
            }
        }
        let pcm = &pcm[2 * denoise::DELAY..2 * denoise::DELAY + len];
        let mut st = Stretch::new(2);
        let mut clip = vec![];
        st.run(pcm, 1.5, &mut clip);
        st.finish(&mut clip);
        let clip_s = clip.len() as f64 / 2.0 / RATE;
        let src_s = (secs * RATE / 1024.0) as usize as f64 * 1024.0 / RATE;
        // music: 8 s of a chord at 44.1 kHz
        let d = out().join("mix");
        std::fs::create_dir_all(&d).unwrap();
        let mf = d.join("chord.wav");
        tool(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "aevalsrc=0.2*sin(2*PI*262*t)+0.15*sin(2*PI*330*t)|0.2*sin(2*PI*392*t):s=44100:d=8",
                mf.to_str().unwrap(),
            ],
        );
        let mut music = Music::open(&mf).unwrap();
        let total = clip_s + 2.5;
        let mut mx = Mixer::new();
        mx.tracks = vec![
            Track::new(Kind::Clip, [0.0, clip_s]),
            Track::new(Kind::Music, [0.0, total]),
        ];
        mx.tracks[1].fade = [0.5, 1.0];
        mx.tracks[1].gain = 0.8;
        let mut enc = Enc::new(48000, 2, 192000).unwrap();
        let mut aus = vec![];
        let n = (total * RATE) as usize;
        let (mut o, mut t0) = (vec![0f32; 2048], 0usize);
        // mix LATENCY frames past the end so the limiter hands out the last ones
        while t0 < n + LATENCY {
            let k = 1024.min(n + LATENCY - t0);
            let (a, b) = ((2 * t0).min(clip.len()), (2 * (t0 + k)).min(clip.len()));
            mx.tracks[0].buf.clear();
            mx.tracks[0].buf.extend_from_slice(&clip[a..b]);
            mx.tracks[1].buf.resize(2 * k, 0.0);
            let got = music.read(&mut mx.tracks[1].buf).unwrap();
            mx.tracks[1].buf.truncate(2 * got);
            mx.run(t0 as f64 / RATE, &mut o[..2 * k]);
            let s = 2 * LATENCY.saturating_sub(t0).min(k);
            enc.push(&o[s..2 * k], |au, p, du| aus.push((au.to_vec(), p, du)));
            t0 += k;
        }
        enc.finish(|au, p, du| aus.push((au.to_vec(), p, du)));
        let f = d.join("chain.m4a");
        let mut x = Mux::create(&f).unwrap();
        let tr = x
            .track(
                Codec::Aac {
                    asc: enc.asc().to_vec(),
                    rate: 48000,
                    ch: 2,
                },
                48000,
                0,
                0,
                false,
            )
            .unwrap();
        for (a, p, du) in &aus {
            x.write(tr, a, *p, *p, *du, true).unwrap();
        }
        x.finish(true).unwrap();
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i", f.to_str().unwrap(), "-f", "f32le", "-"])
            .output()
            .unwrap();
        assert!(
            o.status.success() && o.stderr.is_empty(),
            "{}",
            String::from_utf8_lossy(&o.stderr)
        );
        let y: Vec<f32> = o
            .stdout
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let peak = y.iter().fold(0f32, |p, v| p.max(v.abs()));
        let lvl = |a: f64, b: f64| {
            20.0 * rms(&y[2 * (a * RATE) as usize..2 * (b * RATE) as usize]).log10()
        };
        println!(
            "chain: clip {clip_s:.3} s ({src_s:.3} s at 1.5x), file {:.3} s for {total:.3} s, peak {peak:.3}; level {:.1} dBFS with the clip, {:.1} dBFS music after it, {:.1} dBFS in the last 0.2 s (fade)",
            y.len() as f64 / 2.0 / RATE,
            lvl(0.5, clip_s - 0.5),
            lvl(clip_s + 1.0, clip_s + 1.4),
            lvl(total - 0.2, total)
        );
        assert_eq!(y.len() / 2, n);
        assert!((clip_s - src_s / 1.5).abs() < 1e-3 && peak < 1.0);
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }
}
