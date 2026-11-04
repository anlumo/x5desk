//! Licence plates: YOLOv9-t plate detector (ankandrew/open-image-models, MIT; graph cut before its NMS) on perspective
//! crops around the vehicles YOLOX finds (COCO car, motorcycle, bus, truck). In 360 footage a plate is a few pixels
//! wide: a crop centred on its vehicle, sampled at the network's side, magnifies it to where the detector finds it,
//! and keeps the cost to a crop per vehicle. Plates come out as [`Region`]s for [`crate::Hide`] and [`crate::Blur`],
//! as faces do. Input: RGB 0..1 (open-image-models' preprocessing; its letterbox pads, a square crop needs none);
//! output per anchor: centre, width, height (input pixels), score (sigmoid applied); anchors of strides 8, 16, 32.

use crate::{
    Ai, Cam, Crops, Det, In, Job, Norm, Pano, Region, iou,
    privacy::{merge, nms, place},
    read, sphere,
};
use glam::{Vec2, Vec3};
use std::time::Instant;

#[allow(clippy::all, unused)]
pub(crate) mod net {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/yolo-v9-t-256-license-plates.rs"
    ));
}

/// Network input side (a multiple of 32), crops per run.
pub(crate) const PS: u32 = 256;
const MAX: usize = 8;
pub(crate) const NORM: Norm = Norm {
    a: [1.0; 3],
    b: [0.0; 3],
    bgr: false,
};
/// COCO car, motorcycle, bus, truck.
pub const VEHICLES: [u32; 4] = [2, 3, 5, 7];
/// YOLOX score of a vehicle. Of the plates labelled on the samples a third (26 of 57 by day, 8 of 43 at night) lie only
/// in crops of vehicles scored 0.1 to 0.25: at 0.25, 23 more of 100 stay uncovered (plate score 0.2,
/// tests/footage.rs `plates`), while the regions holding no labelled plate shrink from 24 / 31 / 9 to 7 / 8 / 0 deg² a
/// labelled frame (v606 / v627 / v635). Privacy first: the low end.
const VMIN: f32 = 0.1;
/// A crop's side: its vehicle's larger side times `GROW`, `FMIN`° at least (`PS` over 8° is 1.5 times the pixels of a
/// 7680 wide equirect: finer adds no detail), `FMAX`° at most; `CMAX` crops a frame, the surest vehicles first.
pub(crate) const GROW: f32 = 1.3;
pub(crate) const FMIN: f32 = 8.0;
pub(crate) const FMAX: f32 = 90.0;
pub(crate) const CMAX: usize = 32;

/// The crops around vehicles `v` (YOLOX detections; other classes left out): square, upright, centred on each, one per
/// object (a van found as car and truck is one).
pub fn plate_crops(v: &[Det]) -> Vec<Cam> {
    let mut v: Vec<&Det> = v.iter().filter(|d| VEHICLES.contains(&d.cls)).collect();
    v.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<&Det> = vec![];
    for d in v {
        if keep.len() < CMAX && keep.iter().all(|k| iou(k.on(), d.on(), false) <= 0.6) {
            keep.push(d);
        }
    }
    keep.iter()
        .map(|d| {
            let f = (d.size.max_element() * GROW).clamp(FMIN, FMAX);
            Cam::new(d.dir, Vec3::Z, Vec2::splat((f / 2.0).to_radians().tan()))
        })
        .collect()
}

/// Boxes (x0, y0, x1, y1 in input pixels) with score ≥ `min` of one crop's output: rows centre x, y, width, height,
/// score, `n` anchors each.
pub(crate) fn boxes(v: &[f32], n: usize, min: f32) -> Vec<([f32; 4], f32)> {
    (0..n)
        .filter(|&a| v[4 * n + a] >= min)
        .map(|a| {
            let [x, y, w, h] = [0, 1, 2, 3].map(|r| v[r * n + a]);
            (
                [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0],
                v[4 * n + a],
            )
        })
        .collect()
}

/// The plate detector and its input, made on first use.
pub(crate) struct Nets {
    pub net: net::Model,
    pub pi: In,
    pub pc: Crops,
}

impl Ai {
    /// The plate detector, taken out of `self` while it runs (put back after).
    pub(crate) fn plate_nets(&mut self) -> Result<Box<Nets>, String> {
        if let Some(n) = self.pl.take() {
            return Ok(n);
        }
        let s = PS as usize;
        Ok(Box::new(Nets {
            net: net::Model::from_embedded(&self.dev),
            pi: In::new(&self.dev, [MAX, 3, s, s])?,
            pc: Crops::new(&self.wd, &self.g[1], MAX, MAX * 3 * s * s),
        }))
    }

    /// Plates with score ≥ `min` on `crops` (`plate_crops`; square), `MAX` crops per run: per crop non-maximum
    /// suppression (IoU 0.45, as the exported graph's), then across crops as `faces`.
    pub fn plates_on(&mut self, p: &Pano, crops: &[Cam], min: f32) -> Result<Vec<Region>, String> {
        let n = self.plate_nets()?;
        let s = PS as usize;
        let a = (s / 8).pow(2) + (s / 16).pow(2) + (s / 32).pow(2);
        let mut all = vec![];
        for ch in crops.chunks(MAX) {
            let jobs: Vec<Job> = ch
                .iter()
                .enumerate()
                .map(|(i, c)| Job::new(c, [PS; 2], i * 3 * s * s, &NORM, p.w))
                .collect();
            let d = self.crop(&n.pc, p, &jobs, &n.pi);
            self.spent(0, d);
            let t = Instant::now();
            let v = read(n.net.forward(n.pi.t.clone().narrow(0, 0, ch.len())));
            for (i, cam) in ch.iter().enumerate() {
                all.extend(
                    nms(boxes(&v[i * 5 * a..][..5 * a], a, min), 0.45)
                        .into_iter()
                        .map(|(b, score)| place(cam, b, s as f32, score)),
                );
            }
            self.spent(2, t.elapsed());
        }
        self.pl = Some(n);
        Ok(merge(all))
    }

    /// Plates with score ≥ `min` on the vehicles (YOLOX score ≥ `VMIN`) over `sphere()`.
    pub fn plates(&mut self, p: &Pano, min: f32) -> Result<Vec<Region>, String> {
        let v = self.detect(p, &sphere(), VMIN);
        self.plates_on(p, &plate_crops(&v), min)
    }
}
