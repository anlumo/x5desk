//! Polyphase windowed-sinc resampler for rational rate ratios (44.1 → 48 kHz = 160 / 147). Kaiser window (β 10,
//! about 100 dB stopband); passband to 0.907 × the lower Nyquist (20 kHz of 22.05 kHz), stopband from that Nyquist,
//! so nothing aliases or images. Output frame n sits exactly at input time n · fin / fout (no delay to compensate).

use std::io;

/// Stopband attenuation the taps are sized for (dB) and the passband edge as a fraction of the lower Nyquist.
const ATT: f64 = 100.0;
const PASS: f64 = 0.907;
const MAXCH: usize = 8;

pub struct Resampler {
    pub ch: usize,
    /// Up / down factors: output n is at input time n · m / l.
    l: u64,
    m: u64,
    /// Taps per phase; phase-major coefficients (`l` × `t`).
    t: usize,
    h: Vec<f32>,
    /// Input history (interleaved), `t / 2 - 1` zero frames in front of the stream; position of the next output in
    /// 1 / l input frames from `x[0]`; input frames taken and output frames sent.
    x: Vec<f32>,
    pos: u64,
    nin: u64,
    nout: u64,
}

fn i0(x: f64) -> f64 {
    let (mut s, mut t) = (1.0, 1.0);
    for k in 1..60 {
        t *= (x / 2.0 / k as f64).powi(2);
        s += t;
    }
    s
}

impl Resampler {
    /// `ch` ≤ 8 interleaved channels. Fails on ratios that reduce to more than 1024 phases (no common audio rate does:
    /// 11.025 kHz → 48 kHz has 640).
    pub fn new(fin: u32, fout: u32, ch: usize) -> io::Result<Resampler> {
        let g = {
            let (mut a, mut b) = (fin as u64, fout as u64);
            while b > 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let (l, m) = (fout as u64 / g.max(1), fin as u64 / g.max(1));
        if fin == 0 || fout == 0 || l > 1024 || !(1..=MAXCH).contains(&ch) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("resample {fin} → {fout} Hz, {ch} ch"),
            ));
        }
        // Kaiser design in input samples: transition from PASS to 1 × the lower Nyquist
        let ny = fin.min(fout) as f64 / 2.0;
        let dw = std::f64::consts::TAU * (1.0 - PASS) * ny / fin as f64;
        let t = if l == m {
            0
        } else {
            (((ATT - 7.95) / (2.285 * dw)).ceil() as usize + 1) & !1
        };
        let fc = (1.0 + PASS) * ny / fin as f64; // −6 dB point, in cycles per input sample × 2
        let beta = 0.1102 * (ATT - 8.7);
        let (mut h, mut v) = (vec![0f32; l as usize * t], vec![0f64; t]);
        for (p, ph) in
            h.chunks_exact_mut(t.max(1))
                .enumerate()
                .take(if t > 0 { l as usize } else { 0 })
        {
            for (k, c) in v.iter_mut().enumerate() {
                let tau = p as f64 / l as f64 + (t / 2) as f64 - 1.0 - k as f64;
                let a = std::f64::consts::PI * fc * tau;
                let r = tau / (t / 2) as f64;
                *c = fc
                    * if a == 0.0 { 1.0 } else { a.sin() / a }
                    * i0(beta * (1.0 - r * r).max(0.0).sqrt());
            }
            // unit DC gain per phase: a constant stays exactly constant
            let s: f64 = v.iter().sum();
            for (d, c) in ph.iter_mut().zip(&v) {
                *d = (c / s) as f32;
            }
        }
        let mut r = Resampler {
            ch,
            l,
            m,
            t,
            h,
            x: Vec::with_capacity(4 * t * ch + 8192),
            pos: 0,
            nin: 0,
            nout: 0,
        };
        r.reset();
        Ok(r)
    }

    /// Forgets the stream (seek): the next input is time 0 again.
    pub fn reset(&mut self) {
        self.x.clear();
        let z = (self.t / 2).saturating_sub(1);
        self.x.resize(z * self.ch, 0.0);
        (self.pos, self.nin, self.nout) = (z as u64 * self.l, 0, 0);
    }

    /// Taps per output sample (0 when the rates match).
    pub fn taps(&self) -> usize {
        self.t
    }

    /// Appends the output for interleaved input `x` to `out`.
    pub fn run(&mut self, x: &[f32], out: &mut Vec<f32>) {
        let x = &x[..x.len() / self.ch * self.ch];
        self.nin += (x.len() / self.ch) as u64;
        if self.t == 0 {
            out.extend_from_slice(x);
            self.nout = self.nin;
            return;
        }
        self.x.extend_from_slice(x);
        self.pull(u64::MAX, out);
    }

    /// Ends the stream: appends the rest, so the output has ⌈input frames · fout / fin⌉ frames in all.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        if self.t == 0 {
            return;
        }
        self.x.resize(self.x.len() + self.t / 2 * self.ch, 0.0);
        self.pull((self.nin * self.l).div_ceil(self.m), out);
    }

    fn pull(&mut self, end: u64, out: &mut Vec<f32>) {
        let (ch, t, l) = (self.ch, self.t, self.l);
        let frames = self.x.len() / ch;
        while self.nout < end && (self.pos / l) as usize + t / 2 < frames {
            let (p, s) = ((self.pos % l) as usize, (self.pos / l) as usize + 1 - t / 2);
            let h = &self.h[p * t..(p + 1) * t];
            let x = &self.x[s * ch..(s + t) * ch];
            if ch == 2 {
                let (mut a, mut b) = (0f32, 0f32);
                for (c, v) in h.iter().zip(x.chunks_exact(2)) {
                    (a, b) = (a + c * v[0], b + c * v[1]);
                }
                out.extend_from_slice(&[a, b]);
            } else {
                let mut a = [0f32; MAXCH];
                for (c, v) in h.iter().zip(x.chunks_exact(ch)) {
                    for (a, v) in a.iter_mut().zip(v) {
                        *a += c * v;
                    }
                }
                out.extend_from_slice(&a[..ch]);
            }
            (self.pos, self.nout) = (self.pos + self.m, self.nout + 1);
        }
        // drop the history no later output reaches
        let s = ((self.pos / l) as usize + 1)
            .saturating_sub(t / 2)
            .min(frames);
        self.x.drain(..s * ch);
        self.pos -= s as u64 * l;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Frequency response of the prototype (all phases interleaved = the filter at l × fin), gain at `f` Hz.
    fn gain(r: &Resampler, fin: f64, f: f64) -> f64 {
        let (t, l) = (r.t, r.l as usize);
        let (mut re, mut im) = (0f64, 0f64);
        for p in 0..l {
            for k in 0..t {
                // tap (p, k) sits at τ = p / l + t/2 − 1 − k input samples
                let tau = p as f64 / l as f64 + (t / 2) as f64 - 1.0 - k as f64;
                let w = 2.0 * PI * f / fin * tau;
                re += r.h[p * t + k] as f64 * w.cos();
                im += r.h[p * t + k] as f64 * w.sin();
            }
        }
        (re * re + im * im).sqrt() / l as f64
    }

    /// Passband ripple and stopband attenuation of the designed filters, and a tone through the whole path.
    #[test]
    fn response() {
        for (fin, fout) in [
            (44100u32, 48000u32),
            (48000, 44100),
            (96000, 48000),
            (22050, 48000),
            (8000, 48000),
        ] {
            let r = Resampler::new(fin, fout, 1).unwrap();
            let ny = fin.min(fout) as f64 / 2.0;
            let (mut lo, mut hi, mut stop) = (f64::MAX, 0f64, 0f64);
            for i in 0..=400 {
                let g = gain(&r, fin as f64, PASS * ny * i as f64 / 400.0);
                (lo, hi) = (lo.min(g), hi.max(g));
            }
            // stopband: from the lower Nyquist up to the prototype's own Nyquist (l × fin / 2)
            let top = r.l as f64 * fin as f64 / 2.0;
            let n = 4000;
            for i in 0..=n {
                stop = stop.max(gain(&r, fin as f64, ny + (top - ny) * i as f64 / n as f64));
            }
            let ripple = 20.0 * (hi / lo).log10();
            let att = -20.0 * stop.log10();
            println!(
                "{fin} → {fout}: {} phases × {} taps, passband 0–{:.0} Hz ripple {ripple:.4} dB, stopband from {ny:.0} Hz: {att:.1} dB",
                r.l,
                r.t,
                PASS * ny
            );
            assert!(ripple < 0.01 && att > 95.0, "{fin} → {fout}");
        }
    }

    /// Tones resampled in uneven blocks against the exact tone at the output rate: in the passband the error stays
    /// near −100 dB; an input tone above the output Nyquist vanishes.
    #[test]
    fn tones() {
        for (fin, fout, f) in [
            (44100u32, 48000u32, 1000.0),
            (44100, 48000, 19000.0),
            (48000, 44100, 15000.0),
            (96000, 48000, 7000.0),
        ] {
            let mut r = Resampler::new(fin, fout, 2).unwrap();
            let n = fin as usize;
            let x: Vec<f32> = (0..n)
                .flat_map(|i| {
                    let v = (2.0 * PI * f * i as f64 / fin as f64).sin() as f32 * 0.5;
                    [v, -v]
                })
                .collect();
            let mut y = vec![];
            for c in x.chunks(2 * 999) {
                r.run(c, &mut y);
            }
            r.finish(&mut y);
            assert_eq!(
                y.len() / 2,
                (n as u64 * fout as u64).div_ceil(fin as u64) as usize
            );
            let (mut s, mut e) = (0f64, 0f64);
            // skip the edges (the stream starts and ends with silence around it)
            for i in fout as usize / 10..y.len() / 2 - fout as usize / 10 {
                let v = (2.0 * PI * f * i as f64 / fout as f64).sin() * 0.5;
                s += v * v;
                e += (y[2 * i] as f64 - v).powi(2) + (y[2 * i + 1] as f64 + v).powi(2);
            }
            let snr = 10.0 * (2.0 * s / e).log10();
            println!("{fin} → {fout}, {f} Hz tone: SNR {snr:.1} dB");
            assert!(snr > 90.0, "{fin} → {fout} {f}");
        }
        // 30 kHz at 96 kHz into 48 kHz: above the output Nyquist
        let mut r = Resampler::new(96000, 48000, 1).unwrap();
        let x: Vec<f32> = (0..96000)
            .map(|i| (2.0 * PI * 30000.0 * i as f64 / 96000.0).sin() as f32)
            .collect();
        let mut y = vec![];
        r.run(&x, &mut y);
        r.finish(&mut y);
        let rms = (y[4800..43200].iter().map(|v| (v * v) as f64).sum::<f64>() / 38400.0).sqrt();
        println!(
            "96 → 48 kHz, 30 kHz tone: {:.1} dB",
            20.0 * (rms * 2f64.sqrt()).log10()
        );
        assert!(rms < 1e-5);
        // equal rates pass through
        let mut r = Resampler::new(48000, 48000, 2).unwrap();
        let mut y = vec![];
        r.run(&x[..1000], &mut y);
        assert_eq!(y, x[..1000]);
    }
}
