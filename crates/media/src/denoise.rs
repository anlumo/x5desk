//! Speech denoise (wind, hiss, hum, rumble) for clip audio: a port of RNNoise 0.2 (J.-M. Valin, "A Hybrid DSP / Deep
//! Learning Approach to Real-Time Full-Band Speech Enhancement", 2018; xiph/rnnoise at 70f1d25, BSD-3-Clause,
//! `models/LICENSE-rnnoise`) with its trained network (`models/rnnoise.bin`, provenance in `models/README`).
//! 48 kHz mono in 10 ms frames ([`Stereo`]: both sides with one set of gains), [`DELAY`] samples late; one frame
//! takes ~0.4 ms per channel (release). Bit-exact with the reference's portable C path (int8 weights, exact division
//! in tanh / sigmoid), its float / double promotions included; the reference's own x86 build differs from that path
//! (approximate reciprocal, see the tests).

use crate::fft::{C, Fft};
use std::sync::OnceLock;

pub const FRAME: usize = 480;
/// Output sample n is input sample n − DELAY (analysis window plus the network's one frame of look-ahead).
pub const DELAY: usize = 2 * FRAME;
const W: usize = 2 * FRAME;
const FREQ: usize = FRAME + 1;
const NB: usize = 32;
const NF: usize = 2 * NB + 1;
const PMIN: usize = 60;
const PMAX: usize = 768;
const PFRAME: usize = 960;
const PBUF: usize = PMAX + PFRAME;
const H: usize = 384;
/// Band edges in 50 Hz bins (ERB-like, 0–20 kHz).
const EB: [usize; NB + 2] = [
    0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60, 68, 77, 87, 98, 110, 124,
    140, 157, 176, 198, 223, 251, 282, 317, 356, 400,
];

/// One layer as the reference computes it: int8 weights (row-major here; the blob's 8×4 blocks and sparse index
/// expanded) times the input quantised to int8, summed exactly, times a scale per row; or float weights
/// (column-major, summed in column order); then bias, then the GRU recurrent diagonal.
struct Lin {
    rows: usize,
    cols: usize,
    q: Vec<i8>,
    s: Vec<f32>,
    f: Vec<f32>,
    b: Vec<f32>,
    d: Vec<f32>,
}

struct Net {
    c1: Lin,
    c2: Lin,
    gi: [Lin; 3],
    gr: [Lin; 3],
    out: Lin,
    vad: Lin,
}

static NET: OnceLock<Net> = OnceLock::new();

/// The network from the weight blob (RNNoise's own format: 64-byte headers "DNNw", version, type, size, padded
/// size, name).
fn net() -> &'static Net {
    NET.get_or_init(|| {
        let blob: &[u8] = include_bytes!("../models/rnnoise.bin");
        let mut arrays = vec![];
        let mut p = 0;
        while p + 64 <= blob.len() {
            let i =
                |o: usize| u32::from_le_bytes(blob[p + o..p + o + 4].try_into().unwrap()) as usize;
            assert_eq!(&blob[p..p + 4], b"DNNw", "rnnoise.bin");
            let name = &blob[p + 20..p + 64];
            let name = &name[..name.iter().position(|&c| c == 0).unwrap_or(44)];
            arrays.push((name, &blob[p + 64..p + 64 + i(12)]));
            p += 64 + i(16);
        }
        let get = |n: &str| arrays.iter().find(|a| a.0 == n.as_bytes()).map(|a| a.1);
        let f32s = |n: &str| {
            get(n).map_or(vec![], |b| {
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            })
        };
        let lin = |n: &str, rows: usize, cols: usize| {
            let mut l = Lin {
                rows,
                cols,
                q: vec![],
                s: f32s(&format!("{n}_scale")),
                f: f32s(&format!("{n}_weights_float")),
                b: f32s(&format!("{n}_bias")),
                d: f32s(&format!("{n}_weights_diag")),
            };
            if let Some(w) = get(&format!("{n}_weights_int8")) {
                // blocks of 8 rows × 4 columns (row-major inside), all column blocks or those the index lists
                let idx: Vec<usize> = get(&format!("{n}_weights_idx")).map_or_else(
                    || {
                        (0..rows / 8)
                            .flat_map(|_| std::iter::once(cols / 4).chain((0..cols).step_by(4)))
                            .collect()
                    },
                    |b| {
                        b.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|c| i32::from_le_bytes(*c) as usize)
                            .collect()
                    },
                );
                l.q = vec![0; rows * cols];
                let (mut k, mut ix) = (0, idx.iter());
                for i in (0..rows).step_by(8) {
                    for _ in 0..*ix.next().unwrap() {
                        let c = *ix.next().unwrap();
                        for r in 0..8 {
                            for j in 0..4 {
                                l.q[(i + r) * cols + c + j] = w[k + 4 * r + j] as i8;
                            }
                        }
                        k += 32;
                    }
                }
            }
            assert!(
                l.b.len() == rows
                    && (l.q.len() == rows * cols && l.s.len() == rows || l.f.len() == rows * cols),
                "rnnoise.bin: layer {n}"
            );
            l
        };
        let g = |k: usize| {
            (
                lin(&format!("gru{k}_input"), 3 * H, H),
                lin(&format!("gru{k}_recurrent"), 3 * H, H),
            )
        };
        let (g1, g2, g3) = (g(1), g(2), g(3));
        Net {
            c1: lin("conv1", 128, 3 * NF),
            c2: lin("conv2", H, 3 * 128),
            gi: [g1.0, g2.0, g3.0],
            gr: [g1.1, g2.1, g3.1],
            out: lin("dense_out", NB, 4 * H),
            vad: lin("vad_dense", 1, 4 * H),
        }
    })
}

impl Lin {
    fn run(&self, x: &[f32], out: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        let out = &mut out[..rows];
        if self.q.is_empty() {
            out.fill(0.0);
            for (j, &v) in x[..cols].iter().enumerate() {
                for (o, w) in out.iter_mut().zip(&self.f[j * rows..(j + 1) * rows]) {
                    *o += w * v;
                }
            }
        } else {
            // the reference sums in float, but every partial sum is an integer below 2^24: exact either way
            let mut xq = [0i8; 4 * H];
            for (q, &v) in xq.iter_mut().zip(&x[..cols]) {
                *q = (0.5 + (127.0 * v) as f64).floor() as i32 as i8;
            }
            for ((o, w), s) in out.iter_mut().zip(self.q.chunks_exact(cols)).zip(&self.s) {
                *o = w
                    .iter()
                    .zip(&xq)
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum::<i32>() as f32
                    * s;
            }
        }
        for (o, b) in out.iter_mut().zip(&self.b) {
            *o += b;
        }
        if !self.d.is_empty() {
            let m = cols;
            for i in 0..m {
                out[i] += self.d[i] * x[i];
                out[i + m] += self.d[i + m] * x[i];
                out[i + 2 * m] += self.d[i + 2 * m] * x[i];
            }
        }
    }
}

fn tanh(x: f32) -> f32 {
    let x2 = x * x;
    let num = (0.60863042 * x2 + 96.39235687) * x2 + 952.52801514;
    let den = (11.88600922 * x2 + 413.36801147) * x2 + 952.72399902;
    let v = num * x / den;
    let v = if 1.0 < v { 1.0 } else { v };
    if -1.0 > v { -1.0 } else { v }
}

fn sigmoid(x: f32) -> f32 {
    0.5 + 0.5 * tanh(0.5 * x)
}

/// conv1d over the last three frames: `mem` holds the inputs of the two before.
fn conv(l: &Lin, mem: &mut [f32], x: &[f32], out: &mut [f32]) {
    let mut t = [0f32; 3 * 128];
    let (m, n) = (mem.len(), x.len());
    t[..m].copy_from_slice(mem);
    t[m..m + n].copy_from_slice(x);
    l.run(&t, out);
    out[..l.rows].iter_mut().for_each(|v| *v = tanh(*v));
    mem.copy_from_slice(&t[n..n + m]);
}

fn gru(gi: &Lin, gr: &Lin, s: &mut [f32; H], x: &[f32]) {
    let (mut zrh, mut rec) = ([0f32; 3 * H], [0f32; 3 * H]);
    gi.run(x, &mut zrh);
    gr.run(s, &mut rec);
    for i in 0..2 * H {
        zrh[i] = sigmoid(zrh[i] + rec[i]);
    }
    for i in 0..H {
        let h = tanh(zrh[2 * H + i] + rec[2 * H + i] * zrh[H + i]);
        let z = zrh[i];
        s[i] = z * s[i] + (1.0 - z) * h;
    }
}

fn band_energy(x: &[C], p: &[C], out: &mut [f32; NB]) {
    let mut sum = [0f32; NB + 2];
    for i in 0..NB + 1 {
        let n = EB[i + 1] - EB[i];
        for j in 0..n {
            let frac = j as f32 / n as f32;
            let (a, b) = (x[EB[i] + j], p[EB[i] + j]);
            let t = a.r * b.r + a.i * b.i;
            sum[i] += (1.0 - frac) * t;
            sum[i + 1] += frac * t;
        }
    }
    sum[1] = (sum[0] + sum[1]) * 2.0 / 3.0;
    sum[NB] = (sum[NB] + sum[NB + 1]) * 2.0 / 3.0;
    out.copy_from_slice(&sum[1..NB + 1]);
}

/// Per-bin gains from per-band gains (linear between band centres); bins above 20 kHz get 0.
/// The gain falls at most to 0.6 of the last frame's (RT60 135 ms); that floor follows the band energies from the
/// frame before (`dex`) to this one (`ex`).
fn floor(g: &mut [f32; NB], last: &mut [f32; NB], dex: &[f32; NB], ex: &[f32; NB]) {
    for i in 0..NB {
        let f = 0.6 * last[i];
        g[i] = if g[i] > f { g[i] } else { f };
        let v = g[i] as f64 * (dex[i] as f64 + 1e-3) / (ex[i] as f64 + 1e-3);
        last[i] = (if 1.0 < v { 1.0 } else { v }) as f32;
    }
}

fn interp(g: &mut [f32; FREQ], b: &[f32; NB]) {
    g.fill(0.0);
    for i in 1..NB {
        let n = EB[i + 1] - EB[i];
        for j in 0..n {
            let frac = j as f32 / n as f32;
            g[EB[i] + j] = (1.0 - frac) * b[i - 1] + frac * b[i];
        }
    }
    g[..EB[1]].fill(b[0]);
    g[EB[NB]..EB[NB + 1]].fill(b[NB - 1]);
}

/// Σ x·y in order (the reference's float accumulation).
fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).fold(0.0, |s, (a, b)| s + a * b)
}

fn best_pitch(xc: &[f32], y: &[f32], len: usize, best: &mut [usize; 2]) {
    let mut syy = 1f32;
    let (mut num, mut den) = ([-1f32; 2], [0f32; 2]);
    *best = [0, 1];
    for &v in &y[..len] {
        syy += v * v;
    }
    for (i, &c) in xc.iter().enumerate() {
        if c > 0.0 {
            let c16 = c * 1e-12;
            let n = c16 * c16;
            if n * den[1] > num[1] * syy {
                if n * den[0] > num[0] * syy {
                    (num[1], den[1], best[1]) = (num[0], den[0], best[0]);
                    (num[0], den[0], best[0]) = (n, syy, i);
                } else {
                    (num[1], den[1], best[1]) = (n, syy, i);
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = if 1.0 > syy { 1.0 } else { syy };
    }
}

/// One frame's spectrum, pitch spectrum, energies and correlation, and the input as it came.
struct Ana {
    x: [C; FREQ],
    p: [C; FREQ],
    ex: [f32; NB],
    ep: [f32; NB],
    exp: [f32; NB],
    raw: [f32; FRAME],
    silence: bool,
}

pub struct Denoise {
    /// 0 = the input (delayed), 1 = fully denoised (the reference's output); linear in between.
    pub strength: f32,
    /// Voice probability of the last frame.
    pub vad: f32,
    fft: Fft,
    win: [f32; FRAME],
    dct: [f32; NB * NB],
    za: Vec<C>,
    zb: Vec<C>,
    analysis: [f32; FRAME],
    synthesis: [f32; FRAME],
    pbuf: Vec<f32>,
    last_gain: f32,
    last_period: usize,
    hp: [f32; 2],
    lastg: [f32; NB],
    c1: [f32; 2 * NF],
    c2: [f32; 2 * 128],
    g: [[f32; H]; 3],
    /// The frame before: spectrum, pitch spectrum, energies, correlation.
    dx: [C; FREQ],
    dp: [C; FREQ],
    dex: [f32; NB],
    dep: [f32; NB],
    dexp: [f32; NB],
    /// The input of the last two frames (the dry signal at the output's time).
    dry: [f32; DELAY],
}

impl Default for Denoise {
    fn default() -> Denoise {
        Denoise::new()
    }
}

impl Denoise {
    pub fn new() -> Denoise {
        net();
        let mut dct = [0f32; NB * NB];
        for i in 0..NB {
            for j in 0..NB {
                let v =
                    ((i as f64 + 0.5) * j as f64 * std::f64::consts::PI / NB as f64).cos() as f32;
                dct[i * NB + j] = if j == 0 {
                    (v as f64 * 0.5f64.sqrt()) as f32
                } else {
                    v
                };
            }
        }
        let s = |i: usize| (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / FRAME as f64).sin();
        Denoise {
            strength: 1.0,
            vad: 0.0,
            fft: Fft::new(W),
            win: std::array::from_fn(|i| (0.5 * std::f64::consts::PI * s(i) * s(i)).sin() as f32),
            dct,
            za: vec![C::default(); W],
            zb: vec![C::default(); W],
            analysis: [0.0; FRAME],
            synthesis: [0.0; FRAME],
            pbuf: vec![0.0; PBUF],
            last_gain: 0.0,
            last_period: 0,
            hp: [0.0; 2],
            lastg: [0.0; NB],
            c1: [0.0; 2 * NF],
            c2: [0.0; 2 * 128],
            g: [[0.0; H]; 3],
            dx: [C::default(); FREQ],
            dp: [C::default(); FREQ],
            dex: [0.0; NB],
            dep: [0.0; NB],
            dexp: [0.0; NB],
            dry: [0.0; DELAY],
        }
    }

    /// Denoises one 10 ms frame in place (48 kHz mono, ±1 full scale); the output is [`DELAY`] samples late.
    /// Returns the voice probability (0 in silence).
    pub fn frame(&mut self, x: &mut [f32]) -> f32 {
        let mut feat = [0f32; NF];
        let a = self.analyse(x, &mut feat);
        let (mut g, mut gf) = ([0f32; NB], [0f32; NB]);
        self.vad = 0.0;
        if !a.silence {
            self.vad = self.rnn(&feat, &mut g);
            gf = g;
            floor(&mut gf, &mut self.lastg, &self.dex, &a.ex);
        }
        self.out(x, &g, &gf, &a);
        self.vad
    }

    /// DC block and features of frame `x`; its spectra and energies for [`Denoise::out`] a frame later.
    fn analyse(&mut self, x: &[f32], feat: &mut [f32; NF]) -> Ana {
        let mut a = Ana {
            x: [C::default(); FREQ],
            p: [C::default(); FREQ],
            ex: [0.0; NB],
            ep: [0.0; NB],
            exp: [0.0; NB],
            raw: x[..FRAME].try_into().unwrap(),
            silence: false,
        };
        // the network was trained on 16-bit sample values; the scale is a power of two, so exact both ways
        let mut xin = [0f32; FRAME];
        let (b0, b1, a0, a1) = (-2f32, 1f32, -1.99599f64 as f32, 0.99600f64 as f32);
        for (o, &v) in xin.iter_mut().zip(x.iter()) {
            // DC block (biquad); the reference rounds through double here
            let xi = v * 32768.0;
            let yi = xi + self.hp[0];
            self.hp[0] =
                (self.hp[1] as f64 + (b0 as f64 * xi as f64 - a0 as f64 * yi as f64)) as f32;
            self.hp[1] = (b1 as f64 * xi as f64 - a1 as f64 * yi as f64) as f32;
            *o = yi;
        }
        a.silence = self.features(&xin, &mut a, feat);
        a
    }

    /// Unless `a` is silent: the pitch filter by the network's gains `g`, then the gains `gf` (`g` with its floor)
    /// on the frame before `a`; writes that frame to `x`, `a` becomes the frame before.
    fn out(&mut self, x: &mut [f32], g: &[f32; NB], gf: &[f32; NB], a: &Ana) {
        let x = &mut x[..FRAME];
        if !a.silence {
            self.pitch_filter(g);
            let mut f = [0f32; FREQ];
            interp(&mut f, gf);
            for (c, g) in self.dx.iter_mut().zip(&f) {
                (c.r, c.i) = (c.r * g, c.i * g);
            }
        }
        // synthesis of the frame before: inverse FFT by conjugate symmetry, window, overlap-add
        for i in 0..W {
            self.za[i] = if i < FREQ {
                self.dx[i]
            } else {
                C::new(self.dx[W - i].r, -self.dx[W - i].i)
            };
        }
        self.fft.fwd(&self.za, &mut self.zb);
        let mut y = [0f32; W];
        y[0] = W as f32 * self.zb[0].r;
        for i in 1..W {
            y[i] = W as f32 * self.zb[W - i].r;
        }
        self.window(&mut y);
        let s = self.strength;
        for i in 0..FRAME {
            let wet = (y[i] + self.synthesis[i]) * (1.0 / 32768.0);
            let dry = self.dry[i];
            x[i] = if s == 1.0 { wet } else { dry + s * (wet - dry) };
        }
        // the dry path is the input itself (no DC block): strength 0 gives it back unchanged
        self.dry.copy_within(FRAME.., 0);
        self.dry[FRAME..].copy_from_slice(&a.raw);
        self.synthesis.copy_from_slice(&y[FRAME..]);
        (self.dx, self.dp, self.dex, self.dep, self.dexp) = (a.x, a.p, a.ex, a.ep, a.exp);
    }

    fn window(&self, x: &mut [f32; W]) {
        for i in 0..FRAME {
            x[i] *= self.win[i];
            x[W - 1 - i] *= self.win[i];
        }
    }

    /// Windowed FFT of `x` into `out` (bins 0..=FRAME).
    fn spectrum(&mut self, x: &[f32; W], out: &mut [C; FREQ]) {
        for (z, &v) in self.za.iter_mut().zip(x) {
            *z = C::new(v, 0.0);
        }
        self.fft.fwd(&self.za, &mut self.zb);
        out.copy_from_slice(&self.zb[..FREQ]);
    }

    fn dct(&self, x: &[f32; NB], out: &mut [f32]) {
        for i in 0..NB {
            let s = (0..NB).fold(0f32, |s, j| s + x[j] * self.dct[j * NB + i]);
            out[i] = (s as f64 * (2.0f64 / 22.0).sqrt()) as f32;
        }
    }

    fn features(&mut self, xin: &[f32; FRAME], a: &mut Ana, feat: &mut [f32; NF]) -> bool {
        let Ana {
            x: xs,
            p: ps,
            ex,
            ep,
            exp,
            ..
        } = a;
        let mut x = [0f32; W];
        x[..FRAME].copy_from_slice(&self.analysis);
        x[FRAME..].copy_from_slice(xin);
        self.analysis = *xin;
        self.window(&mut x);
        self.spectrum(&x, xs);
        band_energy(xs, xs, ex);
        self.pbuf.copy_within(FRAME.., 0);
        self.pbuf[PBUF - FRAME..].copy_from_slice(xin);
        // pitch: 2× downsampled, whitened by a 4th-order LPC
        let mut lp = [0f32; PBUF / 2];
        let pb = &self.pbuf;
        for i in 1..PBUF / 2 {
            lp[i] = 0.5 * (0.5 * (pb[2 * i - 1] + pb[2 * i + 1]) + pb[2 * i]);
        }
        lp[0] = 0.5 * (0.5 * pb[1] + pb[0]);
        let mut ac = [0f32; 5];
        let n = PBUF / 2;
        for (k, a) in ac.iter_mut().enumerate() {
            *a = dot(&lp[..n - 4], &lp[k..]);
            *a += (k + n - 4..n).fold(0f32, |d, i| d + lp[i] * lp[i - k]);
        }
        ac[0] *= 1.0001;
        for (i, a) in ac.iter_mut().enumerate().skip(1) {
            let c = 0.008 * i as f32;
            *a -= *a * c * c;
        }
        let mut lpc = [0f32; 4];
        let mut err = ac[0];
        if ac[0] != 0.0 {
            for i in 0..4 {
                let mut rr = 0f32;
                for j in 0..i {
                    rr += lpc[j] * ac[i - j];
                }
                rr += ac[i + 1];
                let r = -rr / err;
                lpc[i] = r;
                for j in 0..(i + 1) >> 1 {
                    let (t1, t2) = (lpc[j], lpc[i - 1 - j]);
                    lpc[j] = t1 + r * t2;
                    lpc[i - 1 - j] = t2 + r * t1;
                }
                err = err - r * r * err;
                if err < 0.001 * ac[0] {
                    break;
                }
            }
        }
        let mut t = 1f32;
        for l in &mut lpc {
            t *= 0.9;
            *l *= t;
        }
        let c1 = 0.8f32;
        let num = [
            lpc[0] + 0.8,
            lpc[1] + c1 * lpc[0],
            lpc[2] + c1 * lpc[1],
            lpc[3] + c1 * lpc[2],
            c1 * lpc[3],
        ];
        let mut mem = [0f32; 5];
        for v in lp.iter_mut() {
            let s = *v
                + num[0] * mem[0]
                + num[1] * mem[1]
                + num[2] * mem[2]
                + num[3] * mem[3]
                + num[4] * mem[4];
            mem = [*v, mem[0], mem[1], mem[2], mem[3]];
            *v = s;
        }
        // pitch search: 4× downsampled coarse pass, 2× refinement around the two best
        let (xl, len, maxp) = (&lp[PMAX / 2..], PFRAME, PMAX - 3 * PMIN);
        let x4: [f32; PFRAME / 4] = std::array::from_fn(|j| xl[2 * j]);
        let y4: [f32; (PFRAME + PMAX) / 4] = std::array::from_fn(|j| lp[2 * j]);
        let mut xc = [0f32; PMAX / 2];
        for (i, c) in xc[..maxp / 4].iter_mut().enumerate() {
            *c = dot(&x4[..len / 4], &y4[i..]);
        }
        let mut best = [0usize; 2];
        best_pitch(&xc[..maxp / 4], &y4, len / 4, &mut best);
        for (i, c) in xc[..maxp / 2].iter_mut().enumerate() {
            *c = 0.0;
            if i.abs_diff(2 * best[0]) > 2 && i.abs_diff(2 * best[1]) > 2 {
                continue;
            }
            let s = dot(&xl[..len / 2], &lp[i..]);
            *c = if -1.0 > s { -1.0 } else { s };
        }
        best_pitch(&xc[..maxp / 2], &lp, len / 2, &mut best);
        let off = if best[0] > 0 && best[0] < maxp / 2 - 1 {
            let (a, b, c) = (xc[best[0] - 1], xc[best[0]], xc[best[0] + 1]);
            if c - a > 0.7 * (b - a) {
                1
            } else if a - c > 0.7 * (b - c) {
                -1
            } else {
                0
            }
        } else {
            0
        };
        let mut period = PMAX - (2 * best[0] as isize - off) as usize;
        let gain = self.remove_doubling(&lp, &mut period);
        (self.last_period, self.last_gain) = (period, gain);
        let mut p = [0f32; W];
        p.copy_from_slice(&self.pbuf[PBUF - W - period..PBUF - period]);
        self.window(&mut p);
        self.spectrum(&p, ps);
        band_energy(ps, ps, ep);
        band_energy(xs, ps, exp);
        for i in 0..NB {
            exp[i] = (exp[i] as f64 / (0.001 + (ex[i] * ep[i]) as f64).sqrt()) as f32;
        }
        self.dct(exp, &mut feat[NB..2 * NB]);
        feat[2 * NB] = (0.01 * (period as f64 - 300.0)) as f32;
        let (mut lmax, mut follow, mut e) = (-2f32, -2f32, 0f32);
        let mut ly = [0f32; NB];
        for i in 0..NB {
            let l = (1e-2 + ex[i] as f64).log10() as f32 as f64;
            let a = follow as f64 - 1.5;
            let a = if a > l { a } else { l };
            let b = (lmax - 7.0) as f64;
            ly[i] = (if b > a { b } else { a }) as f32;
            lmax = if lmax > ly[i] { lmax } else { ly[i] };
            let f = follow as f64 - 1.5;
            follow = (if f > ly[i] as f64 { f } else { ly[i] as f64 }) as f32;
            e += ex[i];
        }
        if (e as f64) < 0.04 {
            // no audio: leave the network's state alone
            feat.fill(0.0);
            return true;
        }
        self.dct(&ly, &mut feat[..NB]);
        feat[0] -= 12.0;
        feat[1] -= 4.0;
        false
    }

    /// Period halving check on the 2× downsampled signal; returns the pitch gain, updates `period`.
    fn remove_doubling(&self, lp: &[f32], period: &mut usize) -> f32 {
        const SECOND: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];
        let (maxp, minp, n) = (PMAX / 2, PMIN / 2, PFRAME / 2);
        let x = |i: isize| lp[(maxp as isize + i) as usize];
        let xs = &lp[maxp..];
        let at = |t: usize| &lp[maxp - t..];
        let t0 = (*period / 2).min(maxp - 1);
        let prev = self.last_period / 2;
        let gain = |xy: f32, xx: f32, yy: f32| (xy as f64 / ((1.0 + xx * yy) as f64).sqrt()) as f32;
        let (xx, mut xy) = (dot(&xs[..n], &xs[..n]), dot(&xs[..n], at(t0)));
        let mut yyl = [0f32; PMAX / 2 + 1];
        yyl[0] = xx;
        let mut yy = xx;
        for i in 1..=maxp {
            yy = yy + x(-(i as isize)) * x(-(i as isize))
                - x((n - i) as isize) * x((n - i) as isize);
            yyl[i] = if 0.0 > yy { 0.0 } else { yy };
        }
        yy = yyl[t0];
        let (mut bxy, mut byy) = (xy, yy);
        let g0 = gain(xy, xx, yy);
        let (mut t, mut g) = (t0, g0);
        for k in 2..=15 {
            let t1 = (2 * t0 + k) / (2 * k);
            if t1 < minp {
                break;
            }
            let t1b = if k == 2 {
                if t1 + t0 > maxp { t0 } else { t0 + t1 }
            } else {
                (2 * SECOND[k] * t0 + k) / (2 * k)
            };
            let (a, b) = (dot(&xs[..n], at(t1)), dot(&xs[..n], at(t1b)));
            xy = 0.5 * (a + b);
            yy = 0.5 * (yyl[t1] + yyl[t1b]);
            let g1 = gain(xy, xx, yy);
            let cont = if t1.abs_diff(prev) <= 1 {
                self.last_gain
            } else if t1.abs_diff(prev) <= 2 && 5 * k * k < t0 {
                0.5 * self.last_gain
            } else {
                0.0
            };
            let mx = |a: f32, b: f32| if a > b { a } else { b };
            // a bias against very short periods (short-term correlation)
            let thr = if t1 < 3 * minp {
                mx(0.4, 0.85 * g0 - cont)
            } else {
                mx(0.3, 0.7 * g0 - cont)
            };
            if g1 > thr {
                (bxy, byy, t, g) = (xy, yy, t1, g1);
            }
        }
        bxy = if 0.0 > bxy { 0.0 } else { bxy };
        let mut pg = if byy <= bxy { 1.0 } else { bxy / (byy + 1.0) };
        let c: [f32; 3] = std::array::from_fn(|k| dot(&xs[..n], at(t + k - 1)));
        let off = if c[2] - c[0] > 0.7 * (c[1] - c[0]) {
            1
        } else if c[0] - c[2] > 0.7 * (c[1] - c[2]) {
            -1
        } else {
            0
        };
        if pg > g {
            pg = g;
        }
        *period = ((2 * t) as isize + off).max(PMIN as isize) as usize;
        pg
    }

    /// The network: per-band gains in `g`, returns the voice probability.
    fn rnn(&mut self, feat: &[f32; NF], g: &mut [f32; NB]) -> f32 {
        let n = net();
        let (mut t, mut cat) = ([0f32; 128], [0f32; 4 * H]);
        conv(&n.c1, &mut self.c1, feat, &mut t);
        conv(&n.c2, &mut self.c2, &t, &mut cat[..H]);
        let x = cat;
        gru(&n.gi[0], &n.gr[0], &mut self.g[0], &x[..H]);
        let s = self.g[0];
        gru(&n.gi[1], &n.gr[1], &mut self.g[1], &s);
        let s = self.g[1];
        gru(&n.gi[2], &n.gr[2], &mut self.g[2], &s);
        for k in 0..3 {
            cat[H * (k + 1)..H * (k + 2)].copy_from_slice(&self.g[k]);
        }
        n.out.run(&cat, g);
        g.iter_mut().for_each(|v| *v = sigmoid(*v));
        let mut v = [0f32];
        n.vad.run(&cat, &mut v);
        sigmoid(v[0])
    }

    /// Pitch filter on the frame before: adds the pitch-delayed spectrum where the network's gain says the
    /// harmonics are buried, then restores each band's energy.
    fn pitch_filter(&mut self, g: &[f32; NB]) {
        let mut r = [0f32; NB];
        for i in 0..NB {
            let (e, gi) = (self.dexp[i], g[i]);
            let v = if e > gi {
                1.0
            } else {
                ((e * e) * (1.0 - gi * gi)) as f64 / (0.001 + ((gi * gi) * (1.0 - e * e)) as f64)
            } as f32;
            let v = if 0.0 > v { 0.0 } else { v };
            r[i] = (if 1.0 < v { 1.0 } else { v } as f64).sqrt() as f32;
            r[i] = (r[i] as f64 * (self.dex[i] as f64 / (1e-8 + self.dep[i] as f64)).sqrt()) as f32;
        }
        let mut rf = [0f32; FREQ];
        interp(&mut rf, &r);
        for ((x, p), f) in self.dx.iter_mut().zip(&self.dp).zip(&rf) {
            (x.r, x.i) = (x.r + f * p.r, x.i + f * p.i);
        }
        let mut ne = [0f32; NB];
        band_energy(&self.dx, &self.dx, &mut ne);
        let norm: [f32; NB] =
            std::array::from_fn(|i| (self.dex[i] as f64 / (1e-8 + ne[i] as f64)).sqrt() as f32);
        interp(&mut rf, &norm);
        for (x, f) in self.dx.iter_mut().zip(&rf) {
            (x.r, x.i) = (x.r * f, x.i * f);
        }
    }
}

/// Stereo with one set of gains for both sides, so the image holds still: each side's network hears its own side,
/// both apply the mean of their gains (a silent side does not count) with one floor on the mean band energies; each
/// side keeps its own pitch filter (which keeps band energies). Two sides alike give exactly what one [`Denoise`]
/// gives.
pub struct Stereo {
    /// As [`Denoise::strength`].
    pub strength: f32,
    /// Voice probability of the last frame (the larger of the two).
    pub vad: f32,
    c: [Denoise; 2],
    lastg: [f32; NB],
}

impl Default for Stereo {
    fn default() -> Stereo {
        Stereo::new()
    }
}

impl Stereo {
    pub fn new() -> Stereo {
        Stereo {
            strength: 1.0,
            vad: 0.0,
            c: [Denoise::new(), Denoise::new()],
            lastg: [0.0; NB],
        }
    }

    /// As [`Denoise::frame`], for the left and right frames.
    pub fn frame(&mut self, l: &mut [f32], r: &mut [f32]) -> f32 {
        let mut feat = [[0f32; NF]; 2];
        let [d0, d1] = &mut self.c;
        let mut a = [d0.analyse(l, &mut feat[0]), d1.analyse(r, &mut feat[1])];
        let (mut gm, mut g, mut n) = ([0f32; NB], [0f32; NB], 0);
        self.vad = 0.0;
        for (k, d) in self.c.iter_mut().enumerate() {
            if !a[k].silence {
                self.vad = self.vad.max(d.rnn(&feat[k], &mut g));
                gm.iter_mut().zip(&g).for_each(|(m, g)| *m += g);
                n += 1;
            }
        }
        gm.iter_mut().for_each(|m| *m /= n.max(1) as f32);
        let mut gf = gm;
        if n > 0 {
            // means: x + x = 2x and 2x · 0.5 = x exactly
            let [d0, d1] = &self.c;
            let dex = std::array::from_fn(|i| (d0.dex[i] + d1.dex[i]) * 0.5);
            let ex = std::array::from_fn(|i| (a[0].ex[i] + a[1].ex[i]) * 0.5);
            floor(&mut gf, &mut self.lastg, &dex, &ex);
        }
        for ((d, x), a) in self.c.iter_mut().zip([l, r]).zip(&mut a) {
            a.silence = n == 0;
            d.strength = self.strength;
            d.out(x, &gm, &gf, a);
        }
        self.vad
    }
}

#[cfg(test)]
mod tests;
