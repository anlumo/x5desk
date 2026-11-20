//! Beat tracking without a model, after Ellis, "Beat tracking by dynamic programming" (J. New Music Research 36,
//! 2007): an onset strength envelope (spectral flux of a log mel spectrum, rectified, averaged over the bands,
//! high-passed), the tempo from the envelope's autocorrelation under a log-Gaussian prior around 120 BPM, and the
//! beats as the path through the envelope that best trades onset strength against keeping that period, found by
//! dynamic programming.

use crate::{
    fft::{C, Fft},
    music::{Music, RATE},
};
use std::{io, ops::Range, path::Path};

/// Window and hop at 48 kHz: 32 and 4 ms, Ellis's 256 and 32 samples at 8 kHz.
const N: usize = 1536;
const HOP: usize = 192;
/// Envelope frames per second.
const FPS: f64 = RATE as f64 / HOP as f64;
/// Mel bands up to `TOP` Hz, the first `LOW` the bass (centres up to 200 Hz: kick and bass, whose hits mark the strong
/// beats where a backbeat's snare would outscore them in all bands); the log spectrum's floor under its peak, dB.
const BANDS: usize = 40;
const LOW: usize = 4;
const TOP: f64 = 8000.0;
const RANGE: f32 = 80.0;
/// Tempo prior (Ellis): centre period, s, and spread, octaves; the periods searched, s.
const T0: f64 = 0.5;
const OCT: f64 = 1.4;
const PER: [f64; 2] = [0.25, 1.5];
/// How hard the beats keep the period (Ellis's α, for an envelope of unit deviation).
const TIGHT: f64 = 100.0;
/// A flux frame's time behind its window's start, s: a log spectrum rises as soon as an onset enters the window, so
/// the flux peaks before the window centres on it (`tests::onsets` measures it).
const LAT: f64 = 0.0265;

/// The beats of a piece of music.
#[derive(Clone, Debug, Default)]
pub struct Beats {
    /// Beat times, s, each with its strength: the bass bands' envelope there (in its standard deviations).
    pub t: Vec<(f64, f32)>,
    pub bpm: f64,
    /// Length of the music, s.
    pub dur: f64,
}

/// The onset strength envelope's input side: stereo 48 kHz in, mel band magnitudes per frame out.
struct Onset {
    fft: Fft,
    win: Vec<f32>,
    /// per band: first FFT bin, triangle weights
    bank: Vec<(usize, Vec<f32>)>,
    /// mono samples not yet past the window, samples seen
    buf: Vec<f32>,
    n: u64,
    x: Vec<C>,
    y: Vec<C>,
    mag: Vec<f32>,
    /// `BANDS` magnitudes per frame
    mel: Vec<f32>,
}

impl Onset {
    fn new() -> Onset {
        let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
        let df = RATE as f64 / N as f64;
        let e: Vec<f64> = (0..BANDS + 2)
            .map(|i| hz(mel(TOP) * i as f64 / (BANDS + 1) as f64))
            .collect();
        let bank = (0..BANDS)
            .map(|b| {
                let (lo, c, hi) = (e[b], e[b + 1], e[b + 2]);
                let k0 = (lo / df).ceil() as usize;
                let w = (k0..=(hi / df) as usize)
                    .map(|k| {
                        let f = k as f64 * df;
                        (if f < c {
                            (f - lo) / (c - lo)
                        } else {
                            (hi - f) / (hi - c)
                        }) as f32
                    })
                    .collect();
                (k0, w)
            })
            .collect();
        Onset {
            fft: Fft::new(N),
            win: (0..N)
                .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos()) as f32)
                .collect(),
            bank,
            buf: Vec::with_capacity(N + 16384),
            n: 0,
            x: vec![C::default(); N],
            y: vec![C::default(); N],
            mag: vec![0.0; N / 2 + 1],
            mel: vec![],
        }
    }

    /// Interleaved stereo at 48 kHz.
    fn push(&mut self, s: &[f32]) {
        self.buf
            .extend(s.as_chunks::<2>().0.iter().map(|f| 0.5 * (f[0] + f[1])));
        self.n += s.len() as u64 / 2;
        let mut i = 0;
        while i + N <= self.buf.len() {
            for ((x, v), w) in self.x.iter_mut().zip(&self.buf[i..]).zip(&self.win) {
                *x = C::new(v * w, 0.0);
            }
            self.fft.fwd(&self.x, &mut self.y);
            for (m, y) in self.mag.iter_mut().zip(&self.y) {
                *m = (y.r * y.r + y.i * y.i).sqrt();
            }
            let mag = &self.mag;
            self.mel.extend(
                (self.bank.iter())
                    .map(|(k, w)| w.iter().zip(&mag[*k..]).map(|(a, b)| a * b).sum::<f32>()),
            );
            i += HOP;
        }
        self.buf.drain(..i);
    }

    /// The envelope of mel bands `bs` at `FPS`: per frame the mean rise of their log magnitudes (floored `RANGE` under
    /// the loudest of all), high-passed at 0.4 Hz (Ellis) and scaled to unit deviation.
    fn env(&self, bs: Range<usize>) -> Vec<f32> {
        let db: Vec<f32> = (self.mel.iter())
            .map(|&e| 20.0 * e.max(1e-10).log10())
            .collect();
        let fl = db.iter().fold(f32::MIN, |a, &b| a.max(b)) - RANGE;
        let mut o: Vec<f32> = (0..db.len() / BANDS)
            .map(|f| match f {
                0 => 0.0,
                _ => {
                    (bs.clone())
                        .map(|b| {
                            (db[f * BANDS + b].max(fl) - db[(f - 1) * BANDS + b].max(fl)).max(0.0)
                        })
                        .sum::<f32>()
                        / bs.len() as f32
                }
            })
            .collect();
        let (mut px, mut py) = (0.0, 0.0);
        for v in &mut o {
            (px, py) = (*v, *v - px + 0.99 * py);
            *v = py;
        }
        let sd = (o.iter().map(|v| v * v).sum::<f32>() / o.len().max(1) as f32).sqrt();
        o.iter_mut().for_each(|v| *v /= sd.max(1e-9));
        o
    }

    /// The beats of what was pushed, each with the strength of the bass bands' envelope there.
    fn finish(&self) -> Beats {
        let o = self.env(0..BANDS);
        let dur = self.n as f64 / RATE as f64;
        // the autocorrelation looks two periods and a frame back
        if o.len() < (2.0 * PER[1] * FPS) as usize + 4 {
            return Beats {
                dur,
                ..Beats::default()
            };
        }
        let p = tempo(&o);
        let lo = smooth(&self.env(0..LOW), p / 32.0);
        Beats {
            t: (track(&o, p).into_iter())
                .map(|f| ((f * HOP) as f64 / RATE as f64 + LAT, lo[f]))
                .collect(),
            bpm: 60.0 * FPS / p,
            dur,
        }
    }
}

/// The beat period, frames: the lag whose autocorrelation, weighted by the prior and with half of its double's
/// added (Ellis's TPS2, which picks the beat level in duple metre over its subdivisions), is the largest; refined by
/// a parabola through its neighbours.
fn tempo(o: &[f32]) -> f64 {
    let [lo, hi] = PER.map(|p| (p * FPS).round() as usize);
    let tps: Vec<f64> = (0..=2 * hi + 3)
        .map(|l| {
            let r: f64 = o.iter().zip(&o[l..]).map(|(a, b)| (a * b) as f64).sum();
            let w = (-0.5 * ((l as f64 / FPS / T0).log2() / OCT).powi(2)).exp();
            if l == 0 { 0.0 } else { w * r }
        })
        .collect();
    let t2 = |l: usize| tps[l] + 0.5 * tps[2 * l] + 0.25 * (tps[2 * l - 1] + tps[2 * l + 1]);
    let l = (lo..=hi).max_by(|&a, &b| t2(a).total_cmp(&t2(b))).unwrap();
    let (a, b, c) = (t2(l - 1), t2(l), t2(l + 1));
    let d = a - 2.0 * b + c;
    l as f64
        + if d < 0.0 {
            (0.5 * (a - c) / d).clamp(-0.5, 0.5)
        } else {
            0.0
        }
}

/// `o` convolved with a Gaussian of σ `sg` frames.
fn smooth(o: &[f32], sg: f64) -> Vec<f32> {
    let r = (3.0 * sg).ceil() as usize;
    let g: Vec<f32> = (0..=2 * r)
        .map(|i| (-0.5 * ((i as f64 - r as f64) / sg).powi(2)).exp() as f32)
        .collect();
    (0..o.len())
        .map(|t| {
            (t.saturating_sub(r)..(t + r + 1).min(o.len()))
                .map(|j| o[j] * g[j + r - t])
                .sum()
        })
        .collect()
}

/// Ellis's dynamic programme for period `p` (frames): the beat frames. The local score is the envelope smoothed over
/// p/32; a beat's score adds the best predecessor's, p/2 to 2p back, less `TIGHT` · ln(gap / p)²; the path back from
/// the best frame of the last period, without the weak beats (under half the beats' rms) at its ends, which a steady
/// path runs through silence.
fn track(o: &[f32], p: f64) -> Vec<usize> {
    let (n, l) = (o.len(), smooth(o, p / 32.0));
    let (a, b) = ((p / 2.0).round() as usize, (2.0 * p).round() as usize);
    let pen: Vec<f32> = (a..=b)
        .map(|d| (-TIGHT * (d as f64 / p).ln().powi(2)) as f32)
        .collect();
    let (mut c, mut back) = (vec![0f32; n], vec![usize::MAX; n]);
    for t in 0..n {
        c[t] = l[t];
        if let Some((v, j)) = (a..=b.min(t))
            .map(|d| (c[t - d] + pen[d - a], t - d))
            .max_by(|x, y| x.0.total_cmp(&y.0))
        {
            (c[t], back[t]) = (l[t] + v, j);
        }
    }
    let mut t = (n.saturating_sub(p.round() as usize)..n)
        .max_by(|&x, &y| c[x].total_cmp(&c[y]))
        .unwrap();
    let mut f = vec![t];
    while back[t] != usize::MAX {
        t = back[t];
        f.push(t);
    }
    f.reverse();
    let rms = (f.iter().map(|&t| l[t] * l[t]).sum::<f32>() / f.len() as f32).sqrt();
    let i = f.iter().position(|&t| l[t] >= 0.5 * rms).unwrap_or(0);
    let j = f
        .iter()
        .rposition(|&t| l[t] >= 0.5 * rms)
        .map_or(f.len(), |j| j + 1);
    f[i..j].to_vec()
}

/// The beats of a music file (crate media's `music` reader: every format it takes).
pub fn of(path: impl AsRef<Path>) -> io::Result<Beats> {
    let mut m = Music::open(path)?;
    let (mut o, mut b) = (Onset::new(), vec![0f32; 2 * 8192]);
    loop {
        let n = m.read(&mut b)?;
        o.push(&b[..2 * n]);
        if n < 8192 {
            return Ok(o.finish());
        }
    }
}

#[cfg(test)]
mod tests;
