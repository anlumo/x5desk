//! Mixed-radix complex FFT (radix 2, 3, 4, 5): a port of the KISS FFT variant in Opus / RNNoise (BSD-3-Clause,
//! `models/LICENSE-rnnoise`) with the same factorisation, twiddles and butterfly order, so the denoiser follows the C
//! reference to the last float rounding. Plain f32 arithmetic in C's evaluation order (no fused multiply-add).

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C {
    pub r: f32,
    pub i: f32,
}

impl C {
    pub fn new(r: f32, i: f32) -> C {
        C { r, i }
    }
    pub fn mul(self, b: C) -> C {
        C::new(self.r * b.r - self.i * b.i, self.r * b.i + self.i * b.r)
    }
    pub fn add(self, b: C) -> C {
        C::new(self.r + b.r, self.i + b.i)
    }
    pub fn sub(self, b: C) -> C {
        C::new(self.r - b.r, self.i - b.i)
    }
}

pub struct Fft {
    pub n: usize,
    tw: Vec<C>,
    /// (radix, remaining length) per stage, as `kf_factor` orders them.
    f: Vec<(usize, usize)>,
    rev: Vec<usize>,
}

impl Fft {
    /// Panics on lengths with prime factors above 5.
    pub fn new(n: usize) -> Fft {
        let tw = (0..n)
            .map(|i| {
                let p = -2.0 * PI / n as f64 * i as f64;
                C::new(p.cos() as f32, p.sin() as f32)
            })
            .collect();
        // powers of 4, then 2, then 3, 5; a lone 2 moves behind the first 4; then the order reverses
        let (mut p, mut k, mut f) = (4, n, vec![]);
        while k > 1 {
            while k % p != 0 {
                p = match p {
                    4 => 2,
                    2 => 3,
                    _ => p + 2,
                };
                if p * p > k {
                    p = k;
                }
            }
            k /= p;
            assert!(p <= 5, "FFT length {n}");
            f.push(p);
            if p == 2 && f.len() > 2 {
                let l = f.len();
                (f[l - 1], f[1]) = (4, 2);
            }
        }
        f.reverse();
        let mut m = n;
        let f: Vec<_> = f
            .into_iter()
            .map(|p| {
                m /= p;
                (p, m)
            })
            .collect();
        let mut rev = vec![0; n];
        bitrev(&mut rev, 0, 0, 1, &f);
        Fft { n, tw, f, rev }
    }

    /// Forward transform scaled by 1/n (Opus convention).
    pub fn fwd(&self, x: &[C], y: &mut [C]) {
        let s = 1.0 / self.n as f32;
        for (i, &v) in x[..self.n].iter().enumerate() {
            y[self.rev[i]] = C::new(v.r * s, v.i * s);
        }
        self.run(y);
    }

    fn run(&self, y: &mut [C]) {
        let l = self.f.len();
        let mut fs = [1usize; 16];
        for i in 0..l {
            fs[i + 1] = fs[i] * self.f[i].0;
        }
        let mut m = 1;
        for i in (0..l).rev() {
            let m2 = if i > 0 { self.f[i - 1].1 } else { 1 };
            let (s, n) = (fs[i], fs[i]);
            match self.f[i].0 {
                2 => bfly2(y, m, n),
                4 => bfly4(y, &self.tw, s, m, n, m2),
                3 => bfly3(y, &self.tw, s, m, n, m2),
                _ => bfly5(y, &self.tw, s, m, n, m2),
            }
            m = m2;
        }
    }
}

fn bitrev(rev: &mut [usize], mut at: usize, out: usize, stride: usize, f: &[(usize, usize)]) {
    let (p, m) = f[0];
    for j in 0..p {
        if m == 1 {
            rev[at] = out + j;
        } else {
            bitrev(rev, at, out + j * m, stride * p, &f[1..]);
        }
        at += stride;
    }
}

fn bfly2(y: &mut [C], m: usize, n: usize) {
    if m == 1 {
        for f in y[..2 * n].chunks_exact_mut(2) {
            let t = f[1];
            f[1] = f[0].sub(t);
            f[0] = f[0].add(t);
        }
        return;
    }
    // m == 4: a radix 2 only ever follows a radix 4
    let tw = 0.707_106_77_f32;
    for f in y[..8 * n].chunks_exact_mut(8) {
        let (a, b) = f.split_at_mut(4);
        let t = b[0];
        b[0] = a[0].sub(t);
        a[0] = a[0].add(t);
        let t = C::new((b[1].r + b[1].i) * tw, (b[1].i - b[1].r) * tw);
        b[1] = a[1].sub(t);
        a[1] = a[1].add(t);
        let t = C::new(b[2].i, -b[2].r);
        b[2] = a[2].sub(t);
        a[2] = a[2].add(t);
        let t = C::new((b[3].i - b[3].r) * tw, -(b[3].i + b[3].r) * tw);
        b[3] = a[3].sub(t);
        a[3] = a[3].add(t);
    }
}

fn bfly4(y: &mut [C], tw: &[C], s: usize, m: usize, n: usize, mm: usize) {
    if m == 1 {
        for f in y[..4 * n].chunks_exact_mut(4) {
            let s0 = f[0].sub(f[2]);
            f[0] = f[0].add(f[2]);
            let s1 = f[1].add(f[3]);
            f[2] = f[0].sub(s1);
            f[0] = f[0].add(s1);
            let s1 = f[1].sub(f[3]);
            f[1] = C::new(s0.r + s1.i, s0.i - s1.r);
            f[3] = C::new(s0.r - s1.i, s0.i + s1.r);
        }
        return;
    }
    for i in 0..n {
        for j in 0..m {
            let k = i * mm + j;
            let a = y[k + m].mul(tw[j * s]);
            let b = y[k + 2 * m].mul(tw[2 * j * s]);
            let c = y[k + 3 * m].mul(tw[3 * j * s]);
            let s5 = y[k].sub(b);
            y[k] = y[k].add(b);
            let (s3, s4) = (a.add(c), a.sub(c));
            y[k + 2 * m] = y[k].sub(s3);
            y[k] = y[k].add(s3);
            y[k + m] = C::new(s5.r + s4.i, s5.i - s4.r);
            y[k + 3 * m] = C::new(s5.r - s4.i, s5.i + s4.r);
        }
    }
}

fn bfly3(y: &mut [C], tw: &[C], s: usize, m: usize, n: usize, mm: usize) {
    let e = tw[s * m].i;
    for i in 0..n {
        for j in 0..m {
            let k = i * mm + j;
            let a = y[k + m].mul(tw[j * s]);
            let b = y[k + 2 * m].mul(tw[2 * j * s]);
            let (s3, s0) = (a.add(b), a.sub(b));
            let h = C::new(y[k].r - s3.r * 0.5, y[k].i - s3.i * 0.5);
            let s0 = C::new(s0.r * e, s0.i * e);
            y[k] = y[k].add(s3);
            y[k + 2 * m] = C::new(h.r + s0.i, h.i - s0.r);
            y[k + m] = C::new(h.r - s0.i, h.i + s0.r);
        }
    }
}

fn bfly5(y: &mut [C], tw: &[C], s: usize, m: usize, n: usize, mm: usize) {
    let (ya, yb) = (tw[s * m], tw[s * 2 * m]);
    for i in 0..n {
        for u in 0..m {
            let k = i * mm + u;
            let s0 = y[k];
            let s1 = y[k + m].mul(tw[u * s]);
            let s2 = y[k + 2 * m].mul(tw[2 * u * s]);
            let s3 = y[k + 3 * m].mul(tw[3 * u * s]);
            let s4 = y[k + 4 * m].mul(tw[4 * u * s]);
            let (s7, s10, s8, s9) = (s1.add(s4), s1.sub(s4), s2.add(s3), s2.sub(s3));
            y[k] = C::new(y[k].r + (s7.r + s8.r), y[k].i + (s7.i + s8.i));
            let s5 = C::new(
                s0.r + (s7.r * ya.r + s8.r * yb.r),
                s0.i + (s7.i * ya.r + s8.i * yb.r),
            );
            let s6 = C::new(s10.i * ya.i + s9.i * yb.i, -(s10.r * ya.i + s9.r * yb.i));
            y[k + m] = s5.sub(s6);
            y[k + 4 * m] = s5.add(s6);
            let s11 = C::new(
                s0.r + (s7.r * yb.r + s8.r * ya.r),
                s0.i + (s7.i * yb.r + s8.i * ya.r),
            );
            let s12 = C::new(s9.i * ya.i - s10.i * yb.i, s10.r * yb.i - s9.r * ya.i);
            y[k + 2 * m] = s11.add(s12);
            y[k + 3 * m] = s11.sub(s12);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_dft() {
        for n in [4, 8, 60, 64, 120, 480, 512, 960, 1024, 2048] {
            let f = Fft::new(n);
            let x: Vec<C> = (0..n)
                .map(|i| C::new((i as f32 * 0.37).sin(), (i as f32 * 1.3).cos() * 0.5))
                .collect();
            let mut y = vec![C::default(); n];
            f.fwd(&x, &mut y);
            let mut err = 0f64;
            for k in 0..n {
                let (mut r, mut i) = (0f64, 0f64);
                for (j, v) in x.iter().enumerate() {
                    let p = -2.0 * PI * (j * k % n) as f64 / n as f64;
                    r += v.r as f64 * p.cos() - v.i as f64 * p.sin();
                    i += v.r as f64 * p.sin() + v.i as f64 * p.cos();
                }
                err = err.max(
                    (y[k].r as f64 * n as f64 - r)
                        .abs()
                        .max((y[k].i as f64 * n as f64 - i).abs()),
                );
            }
            println!("n {n}: factors {:?}, max err {err:.2e}", f.f);
            assert!(err < 1e-3, "n {n}");
        }
    }
}
