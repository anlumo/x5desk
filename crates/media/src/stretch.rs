//! Time-stretch with the pitch kept (WSOLA, Verhelst & Roelands 1993): speed s plays the input s times as fast. Hann
//! frames of 21 ms overlap by half on the output; each frame is taken near its ideal input position, where it continues
//! the previous frame best (normalised cross-correlation on the channel sum, ±5.3 ms around a centre that cancels the
//! last frame's offset: single frames stray up to 10.7 ms, their offsets average out). A frame plays at 1×, so it is
//! placed by its middle: the middle's ideal input time is that of its output instant (exact through speed changes). On
//! average output sample j plays input time j · s, so pictures stay in sync. Speeds outside [`MIN`, `MAX`] play as
//! silence of the same duration; speed changes take effect at the input position where they were given, so the output
//! lasts Σ input / speed exactly. Frame sizes assume 48 kHz.

use std::collections::VecDeque;

const N: usize = 1024;
const HS: usize = N / 2;
const DEL: usize = 256;
pub const MIN: f64 = 0.5;
pub const MAX: f64 = 2.0;

pub struct Stretch {
    pub ch: usize,
    win: Vec<f32>,
    /// Input history (interleaved) and its channel sum; `x[0]` is input frame `base`.
    x: Vec<f32>,
    mono: Vec<f32>,
    base: u64,
    nin: u64,
    /// Ideal input position of the next frame's first output sample; where the last frame was taken (None at the
    /// start and after silence).
    a: f64,
    prev: Option<u64>,
    /// Where the last pick fell in its search window (−DEL..=DEL). The search alone leans towards the previous
    /// frame's continuation, which lies behind the ideal position above 1× and ahead below (speech-like input at 2×:
    /// frames taken 2.3 ms early on average, so the sound trails); centring the next window that far before the ideal
    /// position cancels each offset with the next one, so over k frames the offsets average within 2 · DEL / k (single
    /// frames stray up to 2 · DEL).
    lean: i64,
    /// Overlap-add: the pending second half of the last frame, then room for the next.
    ola: Vec<f32>,
    /// Runs of input given with one speed: (end, speed), absolute input frames.
    segs: VecDeque<(u64, f64)>,
    /// Output frames the input asks for (Σ frames / speed) and output frames sent.
    want: f64,
    nout: u64,
}

impl Stretch {
    pub fn new(ch: usize) -> Stretch {
        let win = (0..N)
            .map(|j| (0.5 - 0.5 * (std::f64::consts::TAU * j as f64 / N as f64).cos()) as f32)
            .collect();
        Stretch {
            ch,
            win,
            x: Vec::with_capacity(8 * N * ch),
            mono: Vec::with_capacity(8 * N),
            base: 0,
            nin: 0,
            a: 0.0,
            prev: None,
            lean: 0,
            ola: vec![0.0; N * ch],
            segs: VecDeque::new(),
            want: 0.0,
            nout: 0,
        }
    }

    /// Forgets the stream (seek): the next input is time 0 again.
    pub fn reset(&mut self) {
        self.x.clear();
        self.mono.clear();
        self.ola.fill(0.0);
        self.segs.clear();
        (self.base, self.nin, self.a, self.prev, self.lean) = (0, 0, 0.0, None, 0);
        (self.want, self.nout) = (0.0, 0);
    }

    /// Takes interleaved input `x` played at `speed` (> 0) and appends the output that is ready to `out`.
    pub fn run(&mut self, x: &[f32], speed: f64, out: &mut Vec<f32>) {
        let n = x.len() / self.ch;
        if n == 0 || !(speed > 0.0 && speed.is_finite()) {
            return;
        }
        self.x.extend_from_slice(&x[..n * self.ch]);
        self.mono
            .extend(x.chunks_exact(self.ch).map(|f| f.iter().sum::<f32>()));
        self.nin += n as u64;
        self.want += n as f64 / speed;
        match self.segs.back_mut() {
            Some(b) if b.1 == speed => b.0 = self.nin,
            _ => self.segs.push_back((self.nin, speed)),
        }
        self.pull(None, out);
    }

    /// Ends the stream: appends the rest, so the output has round(Σ frames / speed) frames in all.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        let (want, start) = (self.want.round() as u64, out.len());
        while self.nout < want {
            // silence after the end, at the last speed, until the frames reach the total
            let pad = N + DEL;
            self.x.resize(self.x.len() + pad * self.ch, 0.0);
            self.mono.resize(self.mono.len() + pad, 0.0);
            self.pull(Some(want), out);
        }
        let extra = (self.nout - want) as usize * self.ch;
        out.truncate(out.len().max(start + extra) - extra);
        self.nout = want;
    }

    /// Makes frames while the input reaches; when finishing (`end` = the total), past the input up to the total.
    fn pull(&mut self, end: Option<u64>, out: &mut Vec<f32>) {
        let ch = self.ch;
        loop {
            // runs that end by the frame's start are done
            let at = self.a as u64;
            while self.segs.len() > 1 && self.segs[0].0 <= at {
                self.segs.pop_front();
            }
            if self.segs.is_empty() {
                return;
            }
            // where the next frame starts (this one's middle): HS output frames on, through the runs ahead (exact at
            // speed changes; past the last run only when finishing)
            let (mut next, mut rem) = (self.a, HS as f64);
            for (i, &(e, v)) in self.segs.iter().enumerate() {
                let room = (e as f64 - next) / v;
                if room >= rem || i + 1 == self.segs.len() {
                    next += rem * v;
                    break;
                }
                (next, rem) = (e as f64, rem - room);
            }
            let have = self.base + self.mono.len() as u64;
            // (the run holding `next` must be known)
            if end.map_or(next >= self.nin as f64, |n| self.nout >= n) {
                return;
            }
            // the frame stands for its middle: it plays `next` there, with the speed of the run holding `next`
            // (placed by its start, its sound would trail by (s − 1) · HS, and silence at speeds out of range by HS)
            let s = (self.segs.iter().find(|r| r.0 as f64 > next))
                .unwrap_or(&self.segs[self.segs.len() - 1])
                .1;
            let c = (next - HS as f64).max(0.0).round() as u64;
            // the search centre: before `c` by where the last pick fell in its window
            let m = (c as i64 - self.lean).max(self.base as i64) as u64;
            if (MIN..=MAX).contains(&s) {
                if m + (DEL + N) as u64 > have {
                    return;
                }
                let p = match self.prev {
                    // continue the previous frame's second half: search around `m`
                    Some(q) => {
                        let lo = m.saturating_sub(DEL as u64).max(self.base);
                        let t = &self.mono[(q + HS as u64 - self.base) as usize..][..HS];
                        let y = &self.mono[(lo - self.base) as usize..];
                        let score = |o: usize, step: usize| {
                            let (mut xy, mut yy) = (0f32, 1e-20f32);
                            for j in (0..HS).step_by(step) {
                                (xy, yy) = (xy + t[j] * y[o + j], yy + y[o + j] * y[o + j]);
                            }
                            xy / yy.sqrt()
                        };
                        // coarse on even offsets and samples, nearest to the centre first; then ±1
                        let (k0, k1) = ((m - lo) as usize, (m + DEL as u64 - lo) as usize);
                        let (mut best, mut bv) = (k0, score(k0, 2));
                        for d in (2..=DEL).step_by(2) {
                            for k in [k0.wrapping_sub(d), k0 + d] {
                                // strictly better by a margin: ties (a periodic signal) stay nearest
                                if k <= k1 {
                                    let v = score(k, 2);
                                    if v > bv + 1e-6 * bv.abs() {
                                        (best, bv) = (k, v);
                                    }
                                }
                            }
                        }
                        let (mut fine, mut fv) = (best, score(best, 1));
                        for k in [best.wrapping_sub(1), best + 1] {
                            if k <= k1 {
                                let v = score(k, 1);
                                if v > fv {
                                    (fine, fv) = (k, v);
                                }
                            }
                        }
                        let p = lo + fine as u64;
                        self.lean = p as i64 - m as i64;
                        p
                    }
                    None => c.max(self.base),
                };
                let x = &self.x[(p - self.base) as usize * ch..][..N * ch];
                for ((o, v), w) in self
                    .ola
                    .chunks_exact_mut(ch)
                    .zip(x.chunks_exact(ch))
                    .zip(&self.win)
                {
                    for (o, v) in o.iter_mut().zip(v) {
                        *o += w * v;
                    }
                }
                self.prev = Some(p);
            } else {
                self.prev = None;
            }
            out.extend_from_slice(&self.ola[..HS * ch]);
            self.ola.copy_within(HS * ch.., 0);
            self.ola[HS * ch..].fill(0.0);
            self.nout += HS as u64;
            self.a = next;
            // drop what neither the next search (from its `c` ≥ this `c`, less 2 · DEL) nor the next template reaches
            let keep = (c.saturating_sub(2 * DEL as u64))
                .min(self.prev.map_or(u64::MAX, |q| q + HS as u64));
            let d = keep.saturating_sub(self.base).min(self.mono.len() as u64) as usize;
            if d > 4 * N {
                self.x.drain(..d * ch);
                self.mono.drain(..d);
                self.base += d as u64;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn run(x: &[f32], ch: usize, speeds: &[(usize, f64)]) -> Vec<f32> {
        let mut s = Stretch::new(ch);
        let mut y = vec![];
        let mut i = 0;
        for &(n, v) in speeds {
            // uneven blocks
            for c in x[i * ch..(i + n) * ch].chunks(ch * 700) {
                s.run(c, v, &mut y);
            }
            i += n;
        }
        s.finish(&mut y);
        y
    }

    /// Peak frequency of `x` (mono) around `f0` (Hann-windowed DFT, 0.25 Hz steps) and the residual after
    /// removing the best-fitting sinusoid at that frequency from each 100 ms, relative to the signal (dB).
    fn peak(x: &[f32], f0: f64) -> (f64, f64) {
        let n = x.len();
        let fit = |f: f64| {
            let (mut re, mut im) = (0f64, 0f64);
            for (i, &v) in x.iter().enumerate() {
                let w = 0.5 - 0.5 * (TAU * i as f64 / n as f64).cos();
                let p = TAU * f * i as f64 / 48000.0;
                (re, im) = (re + w * v as f64 * p.sin(), im + w * v as f64 * p.cos());
            }
            (re, im)
        };
        let (mut bf, mut bv) = (f0, 0.0);
        let mut f = f0 - 30.0;
        while f <= f0 + 30.0 {
            let (a, b) = fit(f);
            if a * a + b * b > bv {
                (bf, bv) = (f, a * a + b * b);
            }
            f += 0.25;
        }
        // per 100 ms (slow phase drift from the splices is inaudible; distortion within a window is not)
        let (mut s, mut e) = (0f64, 0f64);
        for (k, w) in x.chunks_exact(4800).enumerate() {
            let (mut a, mut b) = (0f64, 0f64);
            for (i, &v) in w.iter().enumerate() {
                let p = TAU * bf * (k * 4800 + i) as f64 / 48000.0;
                (a, b) = (a + v as f64 * p.sin(), b + v as f64 * p.cos());
            }
            (a, b) = (a / 2400.0, b / 2400.0);
            for (i, &v) in w.iter().enumerate() {
                let p = TAU * bf * (k * 4800 + i) as f64 / 48000.0;
                s += (v as f64).powi(2);
                e += (v as f64 - a * p.sin() - b * p.cos()).powi(2);
            }
        }
        (bf, 10.0 * (e / s).log10())
    }

    #[test]
    fn unity_speed_is_transparent() {
        let x: Vec<f32> = (0..48000 * 2)
            .map(|i| {
                let t = (i / 2) as f64 / 48000.0;
                ((TAU * 220.0 * t).sin() * 0.3 + (TAU * 3170.0 * t * (1.0 + t)).sin() * 0.2) as f32
                    * if i % 2 == 1 { -0.5 } else { 1.0 }
            })
            .collect();
        let y = run(&x, 2, &[(48000, 1.0)]);
        assert_eq!(y.len(), x.len());
        let e = x
            .iter()
            .zip(&y)
            .skip(2 * HS)
            .take(x.len() - 2 * HS - 2 * N - 2 * DEL)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("speed 1: max difference {e:.2e} after the first {HS} frames (fade-in)");
        assert!(e < 1e-5);
    }

    /// A tone keeps its pitch at every speed, lasts input / speed, and stays clean.
    #[test]
    fn tone_keeps_pitch() {
        let x: Vec<f32> = (0..48000 * 2)
            .map(|i| (TAU * 441.0 * i as f64 / 48000.0).sin() as f32 * 0.5)
            .collect();
        for s in [0.5, 0.7, 1.3, 2.0] {
            let y = run(&x, 1, &[(x.len(), s)]);
            assert_eq!(y.len(), (x.len() as f64 / s).round() as usize);
            let (f, db) = peak(&y[4800..y.len() - 4800], 441.0);
            println!(
                "speed {s}: {} frames, peak {f:.2} Hz, residual after the sinusoid {db:.1} dB",
                y.len()
            );
            assert!((f - 441.0).abs() < 0.5 && db < -30.0, "{s}");
        }
    }

    /// Outside [MIN, MAX] the input plays as silence of the right length; the change to and from it does not click.
    #[test]
    fn silence_outside_range() {
        let x: Vec<f32> = (0..48000 * 3)
            .map(|i| (TAU * 300.0 * i as f64 / 48000.0).sin() as f32 * 0.5)
            .collect();
        let y = run(&x, 1, &[(48000, 1.0), (48000, 4.0), (48000, 0.8)]);
        assert_eq!(y.len(), 48000 + 12000 + 60000);
        // silent from the last sounding frame's end to the first one's start (frames sound by their middles)
        let q = y[48000 + HS..60000 - HS]
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()));
        let jump = y
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0f32, f32::max);
        println!(
            "4x: silent part peak {q:.1e}, largest step between samples {jump:.4} (a 300 Hz tone at 0.5 steps 0.0196)"
        );
        assert!(q == 0.0 && jump < 0.03);
    }

    /// The output lasts Σ input / speed: through a ramp of 50 steps, and when the stream ends at a silent speed.
    #[test]
    fn lengths() {
        let x = vec![0.25f32; 2 * 120000];
        let y = run(&x, 2, &[(20000, 1.0), (10000, 4.0)]);
        assert_eq!(y.len(), 2 * (20000 + 2500));
        let ramp: Vec<(usize, f64)> = (0..50)
            .map(|i| (2399, 0.6 + 1.2 * i as f64 / 49.0))
            .collect();
        let want: f64 = ramp.iter().map(|r| r.0 as f64 / r.1).sum();
        let y = run(&x, 2, &ramp);
        assert_eq!(y.len(), 2 * want.round() as usize);
    }

    /// Speech-like input at 0.6× and 1.5× stays in sync: around each second of output, the output matches the input
    /// near output time × speed.
    #[test]
    fn stays_in_sync() {
        let mut seed = 7u32;
        let x: Vec<f32> = (0..48000 * 6)
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let t = i as f64 / 48000.0;
                // vowels with a gliding pitch, gated in syllables, plus a little noise
                let f = 120.0 + 30.0 * (TAU * 0.7 * t).sin();
                let gate = ((TAU * 3.0 * t).sin() * 0.5 + 0.5).powi(2);
                let v: f64 = (1..12)
                    .map(|h| (TAU * f * h as f64 * t).sin() / h as f64)
                    .sum();
                (v * gate * 0.2 + ((seed >> 9) as f64 / (1 << 23) as f64 - 0.5) * 0.01) as f32
            })
            .collect();
        for s in [0.6, 1.5] {
            let y = run(&x, 1, &[(x.len(), s)]);
            let mut worst = 0f64;
            for k in 1..(y.len() / 48000) {
                // the input offset that matches the 5.3 ms of output around k s best (a longer window spans frames that
                // each play at 1× while the ideal position moves at s: its best lag tells where its loudest frame sits)
                let o = k * 48000;
                let c = (o as f64 * s) as i64 - 128;
                let (mut bd, mut bv) = (0i64, f64::MIN);
                for d in -960i64..=960 {
                    let (mut xy, mut yy) = (0f64, 1e-20f64);
                    for j in 0..256 {
                        let v = x[(c + d) as usize + j] as f64;
                        (xy, yy) = (xy + v * y[o - 128 + j] as f64, yy + v * v);
                    }
                    if xy / yy.sqrt() > bv {
                        (bd, bv) = (d, xy / yy.sqrt());
                    }
                }
                worst = worst.max((bd as f64).abs() / 48.0);
            }
            // a frame is taken within 2 · DEL of its ideal position (DEL around a centre up to DEL off) and reads the
            // input at speed 1, so its ends stray (1 − s) · HS further
            let bound = (2.0 * DEL as f64 + (1.0 - s).abs() * HS as f64) / 48.0;
            println!(
                "speed {s}: {} frames, input within {worst:.1} ms of output × speed (bound {bound:.1} ms)",
                y.len()
            );
            assert!(worst <= bound + 0.5, "{s}");
        }
    }

    /// No lag behind or lead on the pictures: at each speed, 256-sample windows of the output at 300 instants spread
    /// over it find their best lag against the input around output time × speed; the mean of those lags is the sync
    /// error (within 1 ms), and so is their median at the frames' middles (how crate export's timeline test measures).
    /// Speech-like input and noise bursts with sharp onsets; with `X5_SAMPLES`, also v606's and v635's sound (≤ 10 s,
    /// windows of digital silence left out).
    #[test]
    fn no_lag() {
        let mut seed = 7u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f64 / (1 << 23) as f64 - 0.5
        };
        let speech: Vec<f32> = (0..48000 * 6)
            .map(|i| {
                let t = i as f64 / 48000.0;
                let f = 120.0 + 30.0 * (TAU * 0.7 * t).sin();
                let gate = ((TAU * 3.0 * t).sin() * 0.5 + 0.5).powi(2);
                let v: f64 = (1..12)
                    .map(|h| (TAU * f * h as f64 * t).sin() / h as f64)
                    .sum();
                (v * gate * 0.2 + rnd() * 0.01) as f32
            })
            .collect();
        // lowpassed noise, a burst every 50–300 ms decaying over 70 ms
        let (mut lp, mut env, mut at) = (0f64, 0f64, 0);
        let bursts: Vec<f32> = (0..48000 * 6)
            .map(|i| {
                if i == at {
                    (env, at) = (1.0, i + 2400 + ((rnd() + 0.5) * 12000.0) as usize);
                }
                env *= 0.9997;
                lp += 0.05 * (rnd() - lp);
                (lp * env * 3.0 + rnd() * 0.002) as f32
            })
            .collect();
        let mut sigs = vec![("speech", speech), ("bursts", bursts)];
        if let Some(d) = crate::tests::samples() {
            use crate::{
                audio::Aac,
                mp4::{Codec, Mp4},
            };
            for f in ["v606.insv", "v635.insv"] {
                let m = Mp4::open(d.join(f)).unwrap();
                let t = (m.tracks.iter())
                    .find(|t| matches!(t.codec, Codec::Aac { .. }))
                    .unwrap();
                let Codec::Aac { asc, .. } = &t.codec else {
                    unreachable!()
                };
                let (mut dec, mut b, mut pcm) = (Aac::new(asc).unwrap(), vec![], vec![]);
                for s in &t.samples[..t.samples.len().min(470)] {
                    m.read(s, &mut b).unwrap();
                    let ch = dec.ch.max(1) as usize;
                    pcm.extend(dec.decode(&b).unwrap().chunks_exact(ch).map(|f| f[0]));
                }
                sigs.push((f, pcm));
            }
        }
        for (name, x) in &sigs {
            for s in [0.5, 0.75, 1.0, 1.5, 2.0] {
                let y = run(x, 1, &[(x.len(), s)]);
                // the best lag of the output around `o` against the input around o · s (output later: positive)
                let lag = |o: usize| {
                    let q = (o as f64 * s).round() as i64 - 128;
                    let w = &y[o - 128..][..256];
                    if w.iter().all(|v| *v == 0.0) {
                        return None;
                    }
                    let (mut bl, mut bv) = (0, f64::MIN);
                    for l in -1100i64..=1100 {
                        let (mut xy, mut xx) = (0f64, 1e-20);
                        for (j, v) in w.iter().enumerate() {
                            let u = x[(q + l) as usize + j] as f64;
                            (xy, xx) = (xy + u * *v as f64, xx + u * u);
                        }
                        if xy / xx.sqrt() > bv {
                            (bl, bv) = (-l, xy / xx.sqrt());
                        }
                    }
                    Some(bl)
                };
                let (lo, hi) = (12000, y.len() - 12000);
                let all: Vec<i64> = (0..300)
                    .filter_map(|k| lag(lo + (hi - lo) * k / 300))
                    .collect();
                let mean = all.iter().sum::<i64>() as f64 / all.len() as f64;
                let (m0, m1) = (lo / HS + 1, hi / HS);
                let mut g: Vec<i64> = (0..300)
                    .filter_map(|k| lag((m0 + k * (m1 - m0) / 300) * HS))
                    .collect();
                g.sort();
                let med = g[g.len() / 2];
                println!(
                    "{name} at {s}: mean lag {mean:+.1} samples ({:+.2} ms) over {} windows, median at frame middles \
                     {med:+} ({:+.2} ms)",
                    mean / 48.0,
                    all.len(),
                    med as f64 / 48.0
                );
                assert!(mean.abs() <= 48.0 && med.abs() <= 48, "{name} at {s}");
            }
        }
    }
}
