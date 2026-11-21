//! YOLOX-nano (COCO, 80 classes) on square perspective tiles of the sphere. Input: BGR 0..255 without
//! normalisation (YOLOX ≥ 0.1.1rc0); output per anchor: x, y offsets and log w, h in grid cells, objectness and class
//! probabilities (sigmoid applied); anchors of strides 8, 16, 32, row-major.

use crate::{Ai, Cam, Job, Norm, Pano, iou, read};
use burn::tensor::Tensor;
use glam::{Vec2, Vec3};
use std::time::Instant;

#[allow(clippy::all, unused)]
pub(crate) mod model {
    include!(concat!(env!("OUT_DIR"), "/model/yolox_nano.rs"));
}

/// Detector input side (pixels).
pub const SIDE: u32 = 416;
/// Tiles per network run: `sphere` in one.
pub(crate) const MAX: usize = 9;
/// Anchors per tile: 52² + 26² + 13².
const N: usize = 3549;
pub(crate) const NORM: Norm = Norm {
    a: [255.0; 3],
    b: [0.0; 3],
    bgr: true,
};

pub const CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// A detection on the sphere: box centre, angular width and height (degrees), COCO class (`CLASSES`), score
/// (objectness × class probability).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Det {
    pub dir: Vec3,
    pub size: Vec2,
    pub cls: u32,
    pub score: f32,
}

impl Det {
    pub fn on(&self) -> (Vec3, Vec2) {
        (self.dir, self.size)
    }
}

/// `n` square tiles of `fov`° side around the horizon at `pitch`°, the first at yaw 0 (`stitch::View` angles).
pub fn ring(n: usize, fov: f32, pitch: f32) -> Vec<Cam> {
    (0..n)
        .map(|i| Cam::view(360.0 * i as f32 / n as f32, pitch, 0.0, fov, 1.0))
        .collect()
}

/// The tiles auto framing and the tracker's wide search detect over, in one network run: `ring(8, 100, −15)` (pitch
/// −63 to +32 at least) and a 120° tile on the zenith (down to +22 at least): everything above the nadir zone the views
/// keep out of (`frame::LOW`).
pub fn sphere() -> Vec<Cam> {
    let mut t = ring(8, 100.0, -15.0);
    t.push(Cam::view(0.0, 90.0, 0.0, 120.0, 1.0));
    t
}

/// Whether COCO class `c` is a person or an animal (bird to giraffe): what a tap picks.
pub fn alive(c: u32) -> bool {
    matches!(c, 0 | 14..=23)
}

/// The person or animal (`alive`) of detections `c` the user tapped at direction `d`: the smallest box that holds `d`
/// (a child before a grown-up is the child; never the bag or the bike there); None if no box holds it. Boxes of one
/// object under two classes keep the higher score.
pub fn at(mut c: Vec<Det>, d: Vec3) -> Option<Det> {
    c.retain(|x| {
        let h = (x.size / 2.0).to_array().map(|a| a.to_radians().tan());
        alive(x.cls)
            && Cam::new(x.dir, Vec3::Z, Vec2::from(h))
                .px(d)
                .is_some_and(|q| q.abs().max_element() <= 1.0)
    });
    c.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Det> = vec![];
    for x in c {
        if keep.iter().all(|k| iou(k.on(), x.on(), false) <= 0.6) {
            keep.push(x);
        }
    }
    keep.into_iter().min_by(|a, b| {
        a.size
            .element_product()
            .total_cmp(&b.size.element_product())
    })
}

impl Ai {
    /// The person or animal the user tapped at direction `d` (`at`) of those detected (score ≥ 0.3) in crops of 90° and
    /// 30° around it. The 90° crop of one low down reaches into the nadir zone `sphere` keeps out of, where a box over
    /// the subject and the camera's carrier can win (v627, frame 300).
    pub fn pick(&mut self, p: &Pano, d: Vec3) -> Option<Det> {
        let tiles = [1.0, 0.268].map(|t| Cam::new(d, Vec3::Z, Vec2::splat(t)));
        at(self.detect(p, &tiles, 0.3), d)
    }

    /// Detections with score ≥ `min` over `tiles` (square crops at `SIDE`², `MAX` per network run): per tile and
    /// class non-maximum suppression (IoU 0.45), then across tiles, where a box touching its tile's edge yields to
    /// one that doesn't (the object may be cut there).
    pub fn detect(&mut self, p: &Pano, tiles: &[Cam], min: f32) -> Vec<Det> {
        let s = SIDE as usize;
        let mut all = vec![];
        for ch in tiles.chunks(MAX) {
            let jobs: Vec<Job> = ch
                .iter()
                .enumerate()
                .map(|(i, c)| Job::new(c, [SIDE; 2], i * 3 * s * s, &NORM, p.w))
                .collect();
            let d = self.crop(&self.dc, p, &jobs, &self.di);
            self.spent(0, d);
            let t = Instant::now();
            let o = self.det.forward(self.di.t.clone().narrow(0, 0, ch.len()));
            let (best, cls) =
                (o.clone().narrow(2, 4, 1) * o.clone().narrow(2, 5, 80)).max_dim_with_indices(2);
            let v = read(Tensor::cat(vec![o.narrow(2, 0, 4), best, cls.float()], 2));
            for (ti, cam) in ch.iter().enumerate() {
                let v = &v[ti * N * 6..][..N * 6];
                let mut b: Vec<([f32; 4], f32, u32)> = vec![];
                let mut a = 0;
                for (st, g) in [(8.0, 52), (16.0, 26), (32.0, 13)] {
                    for k in 0..g * g {
                        let r = &v[a * 6..][..6];
                        a += 1;
                        if r[4] < min {
                            continue;
                        }
                        let c = Vec2::new(r[0] + (k % g) as f32, r[1] + (k / g) as f32) * st;
                        let h = Vec2::new(r[2].exp(), r[3].exp()) * st / 2.0;
                        b.push((
                            [c.x - h.x, c.y - h.y, c.x + h.x, c.y + h.y],
                            r[4],
                            r[5] as u32,
                        ));
                    }
                }
                b.sort_by(|x, y| y.1.total_cmp(&x.1));
                let mut keep: Vec<([f32; 4], f32, u32)> = vec![];
                for x in b {
                    if keep
                        .iter()
                        .all(|k| k.2 != x.2 || px_iou(&k.0, &x.0) <= 0.45)
                    {
                        keep.push(x);
                    }
                }
                let e = 2.0 / s as f32;
                all.extend(keep.into_iter().map(|(b, score, cls)| {
                    let b = b.map(|v| v * e - 1.0);
                    let (dir, size) = cam.boxed(b);
                    let edge = b.iter().any(|v| v.abs() > 1.0 - 2.0 * e);
                    (
                        edge,
                        Det {
                            dir,
                            size,
                            cls,
                            score,
                        },
                    )
                }));
            }
            self.spent(2, t.elapsed());
        }
        all.sort_by(|x, y| x.0.cmp(&y.0).then(y.1.score.total_cmp(&x.1.score)));
        let mut out: Vec<Det> = vec![];
        for (_, d) in all {
            if out
                .iter()
                .all(|k| k.cls != d.cls || iou(k.on(), d.on(), true) <= 0.6)
            {
                out.push(d);
            }
        }
        out
    }
}

pub(crate) fn px_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let ar = |b: &[f32; 4]| (b[2] - b[0]) * (b[3] - b[1]);
    w * h / (ar(a) + ar(b) - w * h).max(1e-6)
}
