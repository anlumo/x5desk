//! Person re-identification without a learned model: no weights, no training data.
//!
//! The feature is LOMO, the local maximal occurrence feature of Liao et al. ("Person Re-identification by Local
//! Maximal Occurrence Representation and Metric Learning", CVPR 2015, arXiv:1406.4216), written from the paper, of
//! the person's box (48 × 128, stretched to the box) with its brightness evened out, at three scales (halved twice):
//! over windows of 10 × 10 px (stride 5) a joint HSV histogram (`L` levels each) and two SILTP texture histograms
//! (scale-invariant local ternary patterns of the 4 neighbours at 3 and 5 px, τ 0.3); per row of windows the largest
//! count of each bin, so what a stripe of the body holds counts wherever it is in the box (a turn, a step aside); log,
//! each of the three parts unit length, then the whole. Each pixel counts by how much its colour is the person's and
//! not the background's, from the colour histogram of the box against that of the strips beside it.
//!
//! Features are compared by cosine similarity, and the tracker's [`Gallery`] (the subject's features and those of the
//! people seen next to it) weighs each dimension by how much more the subject holds it than the others, so that what
//! the subject shares with everyone (dark clothes at night, the ground) counts little. It picks the candidate that
//! looks like the one and not the others when it finds the subject again. Until it has seen enough of the others it
//! compares by plain cosine, against a threshold of its own.

use crate::{Ai, Cam, Job, Pano, RGB, read, tan};
use glam::{Vec2, Vec3};
use std::{ops::Range, time::Instant};

/// Crop width, height: the box is the middle half of the width (the paper's 48 × 128), a quarter of background on
/// each side; crops per batch.
pub(crate) const W: u32 = 96;
pub(crate) const H: u32 = 128;
pub(crate) const MAX: usize = 8;
/// HSV levels per channel; SILTP threshold and neighbour distances.
const L: usize = 6;
const TAU: f32 = 0.3;
const RAD: [isize; 2] = [3, 5];
/// Colour bins; bins per row of windows: colour, then the two textures (3⁴ patterns each); the parts' ranges.
const C: usize = L * L * L;
const BINS: usize = C + 2 * 81;
const PARTS: [Range<usize>; 3] = [0..C, C..C + 81, C + 81..BINS];
/// Rows of windows over the three scales: (128 / 2ˢ − 10) / 5 + 1.
const ROWS: usize = 24 + 11 + 5;
pub const DIM: usize = ROWS * BINS;

/// An appearance feature, unit length.
pub type Emb = [f32; DIM];

/// Cosine similarity of two features.
pub fn sim(a: &Emb, b: &Emb) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Ai {
    /// Appearance features of the people in boxes `b` (centre, angular width and height in degrees) of `p`, `MAX`
    /// crops per pass.
    pub fn embed(&mut self, p: &Pano, b: &[(Vec3, Vec2)]) -> Vec<Emb> {
        let n = (3 * W * H) as usize;
        let mut out = Vec::with_capacity(b.len());
        for ch in b.chunks(MAX) {
            let jobs: Vec<Job> = ch
                .iter()
                .enumerate()
                .map(|(i, &(d, s))| {
                    let c = Cam::new(d, Vec3::Z, tan(s) * Vec2::new(1.0, 0.5));
                    Job::new(&c, [W, H], i * n, &RGB, p.w)
                })
                .collect();
            let d = self.crop(&self.rc, p, &jobs, &self.ri);
            self.spent(0, d);
            let t = Instant::now();
            // all `MAX` crops: a read of the whole input needs no slicing kernel per batch size
            let v = read(self.ri.t.clone());
            for c in v.chunks_exact(n).take(ch.len()) {
                out.push([0.0; DIM]);
                lomo(c, out.last_mut().unwrap());
            }
            self.spent(3, t.elapsed());
        }
        out
    }
}

/// The joint HSV bin of colour `c` (0..1).
fn hsv([r, g, b]: [f32; 3]) -> usize {
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    let c = mx - mn;
    let h = if c <= 1e-6 {
        0.0
    } else if mx == r {
        ((g - b) / c).rem_euclid(6.0)
    } else if mx == g {
        (b - r) / c + 2.0
    } else {
        (r - g) / c + 4.0
    } / 6.0;
    let s = if mx > 1e-6 { c / mx } else { 0.0 };
    let q = |v: f32| ((v * L as f32) as usize).min(L - 1);
    (q(h) * L + q(s)) * L + q(mx)
}

/// The LOMO feature of crop `px` (planar RGB, 0..1, `W` × `H`, the box in the middle half) into `e` (zeroed).
pub(crate) fn lomo(px: &[f32], e: &mut Emb) {
    let (cw, mut w, mut h) = (W as usize, W as usize / 2, H as usize);
    let (n, x0) = (cw * h, w / 2);
    let inside = |i: usize| (x0..x0 + w).contains(&(i % cw));
    // the box's mean to 0.4: a stand-in for the paper's Retinex, so that light and shade change little
    let m = (0..n)
        .filter(|&i| inside(i))
        .map(|i| px[i] + px[n + i] + px[2 * n + i]);
    let k = 0.4 / (m.sum::<f32>() / (3 * w * h) as f32).max(0.02);
    let col: Vec<[f32; 3]> = (0..n)
        .map(|i| [0, 1, 2].map(|c| (px[c * n + i] * k).min(1.0)))
        .collect();
    // a pixel weighs what its colour's count in the box has over that in the strips beside it (as many pixels), as a
    // share: 1 − beside / box; a colour as common beside the box as in it is background
    let mut f = [[0f32; 2]; C];
    for (i, c) in col.iter().enumerate() {
        f[hsv(*c)][inside(i) as usize] += 1.0;
    }
    let (mut im, mut wt): (Vec<[f32; 3]>, Vec<f32>) = (0..n)
        .filter(|&i| inside(i))
        .map(|i| {
            let f = f[hsv(col[i])];
            (col[i], (1.0 - f[0] / f[1]).max(0.0))
        })
        .unzip();
    let mut row = 0;
    for _ in 0..3 {
        let g: Vec<f32> = im.iter().map(|p| (p[0] + p[1] + p[2]) / 3.0).collect();
        let at = |y: isize, x: isize| {
            g[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize]
        };
        // weighted histograms of 5 × 5 blocks; a window is 2 × 2 of them
        let (bh, bw) = (h / 5, w / 5);
        let mut blk = vec![0f32; bh * bw * BINS];
        for y in 0..bh * 5 {
            for x in 0..bw * 5 {
                let b = &mut blk[(y / 5 * bw + x / 5) * BINS..][..BINS];
                let v = wt[y * w + x];
                b[hsv(im[y * w + x])] += v;
                // SILTP: per neighbour 1 if brighter than the pixel by more than τ (relative), 2 if darker so
                let c = g[y * w + x];
                let d =
                    |v: f32| (v > (1.0 + TAU) * c) as usize + 2 * (v < (1.0 - TAU) * c) as usize;
                let (y, x) = (y as isize, x as isize);
                for (r, o) in RAD.into_iter().zip([C, C + 81]) {
                    let t = [(0, r), (-r, 0), (0, -r), (r, 0)]
                        .into_iter()
                        .fold(0, |a, (dy, dx)| 3 * a + d(at(y + dy, x + dx)));
                    b[o + t] += v;
                }
            }
        }
        for r in 0..bh - 1 {
            let f = &mut e[row * BINS..][..BINS];
            for (i, f) in f.iter_mut().enumerate() {
                let s = |y: usize, x: usize| blk[(y * bw + x) * BINS + i];
                let m = (0..bw - 1)
                    .map(|c| s(r, c) + s(r, c + 1) + s(r + 1, c) + s(r + 1, c + 1))
                    .fold(0f32, f32::max);
                *f = m.ln_1p();
            }
            row += 1;
        }
        // the next scale: 2 × 2 means of colours and weights
        (w, h) = (w / 2, h / 2);
        let q = |i: usize| [0, 1, 2 * w, 2 * w + 1].map(|d| 2 * (i / w) * 2 * w + 2 * (i % w) + d);
        wt = (0..w * h)
            .map(|i| q(i).iter().map(|&j| wt[j]).sum::<f32>() / 4.0)
            .collect();
        im = (0..w * h)
            .map(|i| [0, 1, 2].map(|c| q(i).iter().map(|&j| im[j][c]).sum::<f32>() / 4.0))
            .collect();
    }
    let mut l = [0f32; 3];
    for r in e.chunks(BINS) {
        for (l, p) in l.iter_mut().zip(PARTS) {
            *l += r[p].iter().map(|x| x * x).sum::<f32>();
        }
    }
    let l = l.map(|l| 1.0 / (3.0 * l).sqrt().max(1e-12));
    for r in e.chunks_mut(BINS) {
        for (l, p) in l.iter().zip(PARTS) {
            r[p].iter_mut().for_each(|x| *x *= l);
        }
    }
}

/// Re-identification thresholds: a candidate is the subject when it is at least `SAME` similar to one of the
/// subject's features and `MARGIN` more similar to them than to anyone seen next to it, by the gallery's weights. In
/// `tests/footage.rs` `reid` (the gallery from the first 4 s, features 4 to 17 s older): on v627 (night, a crowd) the
/// subject 0.53 (5th percentile) to 0.70, all 37 pass, 899 others 0.43 median, 6 pass; on v606 (day; weights from its
/// 8 features of others, with `FIT` at 1) the subject 0.56 to 0.83, 64 of 66 pass, 768 others 0.31 median, 6 pass.
/// At 0.55, 7 of v627's subject samples fail for 5 fewer others on v606.
const SAME: f32 = 0.5;
const MARGIN: f32 = 0.05;
/// Until the gallery holds `FIT` features of people seen next to the subject it compares by plain cosine, a scale of
/// its own where two different people are 0.48 alike in the median (`sep`): there the subject must be `ALONE` similar.
/// In `reid` with the subject's features alone, `SAME` would pass 561 of v627's 899 others and 308 of v606's 768, 0.6
/// still 11 and 1; 0.63 passes 1 and 0 (OSNet's rule passed 1 of v627's), and 4 of v627's 37 subject samples and 52
/// of v606's 66: plain cosine tells the subject from a crowd at night poorly (the most similar person on 18 of 37 key
/// frames).
const ALONE: f32 = 0.63;
/// Features of others the weights need: from fewer, everyone looks more alike by them. In `reid` on v627, `SAME` by
/// weights from the first 1, 3, 6, 8 and 12 features of others passed 802, 254, 51, 20 and 10 of the 899 others (6 by
/// those from all 24), and 36 or 37 of the subject's 37 samples.
const FIT: usize = 12;

/// Who the subject is: its features (the pick's first, kept; then the newest of those taken while it was tracked
/// with certainty), and features of the people seen next to it (who it is not), with the weights of the dimensions.
#[derive(Default)]
pub struct Gallery {
    pub(crate) me: Vec<Emb>,
    pub(crate) not: Vec<Emb>,
    /// √ of each dimension's weight; empty: all alike (plain cosine).
    w: Vec<f32>,
}

/// Features kept per side.
const KEEP: usize = 24;

impl Gallery {
    /// Adds `e` to the subject's features (`me`) or the others', and refits the weights.
    pub fn add(&mut self, e: Emb, me: bool) {
        let v = if me { &mut self.me } else { &mut self.not };
        if v.len() >= KEEP {
            // the pick's feature stays: it alone is surely the subject
            v.remove(me as usize);
        }
        v.push(e);
        self.fit();
    }

    /// The dimensions' weights: the Fisher ratio of those the subject's features hold more of than the others',
    /// (μ₁ − μ₂)² / (σ₁² + σ₂² + 1e-4), plus 0.05 for all (1e-4 is about 1.5 / `DIM`, a dimension's mean square: what
    /// barely varies on both sides does not blow up); all alike until there are `FIT` of the others.
    fn fit(&mut self) {
        self.w.clear();
        if self.me.is_empty() || self.not.len() < FIT {
            return;
        }
        // per dimension: mean, variance
        let st = |v: &[Emb]| {
            let mut s = vec![[0f32; 2]; DIM];
            for x in v {
                for (s, x) in s.iter_mut().zip(x) {
                    *s = [s[0] + x, s[1] + x * x];
                }
            }
            let n = v.len() as f32;
            s.into_iter()
                .map(move |[a, b]| (a / n, b / n - (a / n).powi(2)))
        };
        self.w = st(&self.me)
            .zip(st(&self.not))
            .map(|(a, b)| ((a.0 - b.0).max(0.0).powi(2) / (a.1 + b.1 + 1e-4) + 0.05).sqrt())
            .collect();
    }

    /// The least similarity to the subject (`score`'s first) that is the subject: `SAME` by the weights, `ALONE` by
    /// plain cosine.
    pub fn same(&self) -> f32 {
        if self.w.is_empty() { ALONE } else { SAME }
    }

    /// Whether a feature `score`d `s` is the subject's: like it by `same`, and `MARGIN` more than like anyone seen
    /// next to it.
    pub fn is(&self, s: (f32, f32)) -> bool {
        s.0 >= self.same() && s.0 >= s.1 + MARGIN
    }

    /// Whether a feature `score`d `s` is someone seen next to the subject: not like the subject by `same`, and
    /// `MARGIN` more like one of them.
    pub fn other(&self, s: (f32, f32)) -> bool {
        s.0 < self.same() && s.1 > s.0 + MARGIN
    }

    /// Best similarity of `e` to the subject and to the others (−1 for none), by the weights, by plain cosine while
    /// there are none.
    pub fn score(&self, e: &Emb) -> (f32, f32) {
        let s = |x: &Emb| {
            if self.w.is_empty() {
                return sim(x, e);
            }
            let (mut d, mut a, mut b) = (0f32, 0f32, 0f32);
            for ((w, x), y) in self.w.iter().zip(x).zip(e) {
                let (x, y) = (w * x, w * y);
                (d, a, b) = (d + x * y, a + x * x, b + y * y);
            }
            d / (a * b).sqrt().max(1e-12)
        };
        let m = |v: &[Emb]| v.iter().map(s).fold(-1f32, f32::max);
        (m(&self.me), m(&self.not))
    }
}
