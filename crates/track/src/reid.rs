//! Person re-identification with OSNet-AIN x1.0 (Zhou et al., torchreid, MIT; trained on MSMT17): a 128 × 256 crop
//! of a person box (RGB, ImageNet mean / std, stretched to the box as torchreid does) -> a 512-d appearance feature,
//! compared by cosine similarity. The tracker keeps the subject's features and those of people seen next to it, and
//! picks the candidate that looks like the one and not the others when it finds the subject again.

use crate::{Ai, Cam, Job, Norm, Pano, read, tan};
use glam::{Vec2, Vec3};
use std::time::Instant;

#[allow(clippy::all, unused)]
pub(crate) mod model {
    include!(concat!(env!("OUT_DIR"), "/model/osnet_ain_x1_0_msmt17.rs"));
}

/// Crop width, height; crops per network run.
pub(crate) const W: u32 = 128;
pub(crate) const H: u32 = 256;
pub(crate) const MAX: usize = 8;
pub const DIM: usize = 512;
pub(crate) const NORM: Norm = Norm {
    a: [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
    b: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    bgr: false,
};

/// An appearance feature, unit length.
pub type Emb = [f32; DIM];

/// Cosine similarity of two features.
pub fn sim(a: &Emb, b: &Emb) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Ai {
    /// Appearance features of the people in boxes `b` (centre, angular width and height in degrees) of `p`, `MAX`
    /// per network run.
    pub fn embed(&mut self, p: &Pano, b: &[(Vec3, Vec2)]) -> Vec<Emb> {
        let n = (3 * W * H) as usize;
        let mut out = Vec::with_capacity(b.len());
        for ch in b.chunks(MAX) {
            let jobs: Vec<Job> = ch
                .iter()
                .enumerate()
                .map(|(i, &(d, s))| {
                    let c = Cam::new(d, Vec3::Z, tan(s) / 2.0);
                    Job::new(&c, [W, H], i * n, &NORM, p.w)
                })
                .collect();
            let d = self.crop(&self.rc, p, &jobs, &self.ri);
            self.spent(0, d);
            let t = Instant::now();
            let x = self.ri.t.clone().narrow(0, 0, ch.len());
            let v = read(self.reid.forward(x));
            for f in v.as_chunks::<DIM>().0 {
                let l = f.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                out.push(std::array::from_fn(|i| f[i] / l));
            }
            self.spent(3, t.elapsed());
        }
        out
    }
}

/// Re-identification thresholds: a candidate is the subject when it is at least `SAME` similar to one of the
/// subject's features and `MARGIN` more similar to them than to anyone seen next to it. On v627 (night, a crowd;
/// `tests/footage.rs` `reid`), against features 4 to 17 s older: the subject 0.71 (5th percentile) to 0.84, 900 others
/// 0.48 median, 1 of them ≥ 0.65.
pub(crate) const SAME: f32 = 0.65;
pub(crate) const MARGIN: f32 = 0.05;

/// Who the subject is: its features (the pick's first, kept; then the newest of those taken while it was tracked
/// with certainty), and features of the people seen next to it (who it is not).
#[derive(Default)]
pub(crate) struct Gallery {
    pub me: Vec<Emb>,
    pub not: Vec<Emb>,
}

/// Features kept per side.
const KEEP: usize = 24;

impl Gallery {
    pub fn add(&mut self, e: Emb, me: bool) {
        let v = if me { &mut self.me } else { &mut self.not };
        if v.len() >= KEEP {
            // the pick's feature stays: it alone is surely the subject
            v.remove(me as usize);
        }
        v.push(e);
    }

    /// Best similarity to the subject and to the others.
    pub fn score(&self, e: &Emb) -> (f32, f32) {
        let m = |v: &[Emb]| v.iter().map(|x| sim(x, e)).fold(-1f32, f32::max);
        (m(&self.me), m(&self.not))
    }
}
