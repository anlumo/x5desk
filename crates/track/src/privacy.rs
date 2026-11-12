//! Privacy: faces (YuNet, OpenCV zoo, MIT) found on perspective tiles of the sphere, then linked over frames into
//! tracks ([`Hide`]) that are interpolated between detections and carried on a little past their ends, so that a
//! face the detector misses for a few frames stays hidden; [`crate::Blur`] blurs the regions in the equirect. Licence
//! plates take the same path (`plates`).

use crate::{Ai, Cam, Crops, In, Job, Norm, Pano, det::px_iou, iou, plates::sure, read};
use burn::tensor::Tensor;
use glam::{Quat, Vec2, Vec3};
use std::time::Instant;

#[allow(clippy::all, unused)]
pub(crate) mod yunet {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/face_detection_yunet_2023mar.rs"
    ));
}

/// Tile side (a multiple of 32), tiles per run.
pub(crate) const FS: u32 = 640;
const MAX: usize = 8;
/// BGR 0..255 (OpenCV's `blobFromImage` without scaling).
pub(crate) const NORM: Norm = Norm {
    a: [255.0; 3],
    b: [0.0; 3],
    bgr: true,
};

/// A face or plate on the sphere: centre, angular width and height (degrees; upright as `Cam::new(dir, Z)`), score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub dir: Vec3,
    pub size: Vec2,
    pub score: f32,
}

impl Region {
    pub fn on(&self) -> (Vec3, Vec2) {
        (self.dir, self.size)
    }
}

/// Face tiles, 20 in three runs: 8 of 75° around the horizon (7.3 px/° at their centre, near the equirect's 8.5 at
/// 3072 wide), 6 of 90° at pitch −60 and 6 at +60 (5.6 px/° or more, past the poles); a face is about 10 px at the
/// least YuNet finds. Upright faces stand within 27° of upright in one of them anywhere, which YuNet needs: it is not
/// rotation-invariant, and on one tile over a pole a face is turned by its azimuth.
pub fn face_tiles() -> Vec<Cam> {
    let mut t = crate::ring(8, 75.0, 0.0);
    t.extend(crate::ring(6, 90.0, -60.0));
    t.extend(crate::ring(6, 90.0, 60.0));
    t
}

/// YuNet and its input, made on first use (80 MB of inputs); macOS: its Core ML form, made on its first use (None
/// inside when it won't load).
pub(crate) struct Nets {
    pub face: yunet::Model,
    pub fi: In,
    pub fc: Crops,
    #[cfg(target_os = "macos")]
    pub ml: Option<Option<crate::coreml::Yunet>>,
}

impl Nets {
    /// burn's YuNet on the first `n` crops of `fi`: per tile, strides 8, 16, 32, per anchor √(cls · obj) (each clamped to
    /// 0..1) and the box, read back.
    pub(crate) fn burn(&self, n: usize) -> Vec<f32> {
        let o = self.face.forward(self.fi.t.clone().narrow(0, 0, n));
        // per stride: score √(cls · obj), box; strides 8, 16, 32 in one row block per tile
        read(Tensor::cat(
            vec![
                Tensor::cat(
                    vec![(o.0.clamp(0.0, 1.0) * o.3.clamp(0.0, 1.0)).sqrt(), o.6],
                    2,
                ),
                Tensor::cat(
                    vec![(o.1.clamp(0.0, 1.0) * o.4.clamp(0.0, 1.0)).sqrt(), o.7],
                    2,
                ),
                Tensor::cat(
                    vec![(o.2.clamp(0.0, 1.0) * o.5.clamp(0.0, 1.0)).sqrt(), o.8],
                    2,
                ),
            ],
            1,
        ))
    }
}

impl Ai {
    /// YuNet, taken out of `self` while it runs (put back after).
    pub(crate) fn nets(&mut self) -> Result<Box<Nets>, String> {
        if let Some(n) = self.pv.take() {
            return Ok(n);
        }
        let f = FS as usize;
        Ok(Box::new(Nets {
            face: yunet::Model::from_embedded(&self.dev),
            fi: In::new(&self.dev, [MAX, 3, f, f])?,
            fc: Crops::new(&self.wd, &self.g[1], MAX, MAX * 3 * f * f),
            #[cfg(target_os = "macos")]
            ml: None,
        }))
    }

    /// Faces with score ≥ `min` over `tiles` (`face_tiles()`; square), `MAX` tiles per run: per tile non-maximum
    /// suppression (IoU 0.3, as OpenCV's `FaceDetectorYN`), then across tiles as `detect`. YuNet's own threshold is
    /// 0.9; privacy wants the faces it is less sure of too (0.5 to 0.6).
    pub fn faces(&mut self, p: &Pano, tiles: &[Cam], min: f32) -> Result<Vec<Region>, String> {
        #[allow(unused_mut)]
        let mut n = self.nets()?;
        #[cfg(target_os = "macos")]
        if self.coreml && n.ml.is_none() {
            const _: () = assert!(MAX == crate::coreml::B);
            n.ml = Some(
                crate::coreml::Yunet::new(&self.wd)
                    .inspect_err(|e| eprintln!("track: YuNet on Core ML: {e}; on burn instead"))
                    .ok(),
            );
        }
        let s = FS as usize;
        let mut all = vec![];
        for ch in tiles.chunks(MAX) {
            let jobs: Vec<Job> = ch
                .iter()
                .enumerate()
                .map(|(i, c)| Job::new(c, [FS; 2], i * 3 * s * s, &NORM, p.w))
                .collect();
            let d = self.crop(&n.fc, p, &jobs, &n.fi);
            self.spent(0, d);
            let t = Instant::now();
            #[cfg(target_os = "macos")]
            let v = {
                let ml =
                    n.ml.as_ref()
                        .and_then(Option::as_ref)
                        .filter(|_| self.coreml);
                match ml.map(|y| y.run(&self.wd, &self.q, &n.fc.dst, ch.len())) {
                    Some(Ok(v)) => v,
                    // a run Core ML fails (the Neural Engine after sleep, memory pressure): burn's from here on, the
                    // crops are in its input already
                    Some(Err(e)) => {
                        eprintln!("track: YuNet on Core ML: {e}; on burn instead");
                        n.ml = Some(None);
                        n.burn(ch.len())
                    }
                    None => n.burn(ch.len()),
                }
            };
            #[cfg(not(target_os = "macos"))]
            let v = n.burn(ch.len());
            let m = v.len() / ch.len() / 5;
            for (ti, cam) in ch.iter().enumerate() {
                let mut a = ti * m;
                let mut b: Vec<([f32; 4], f32)> = vec![];
                for st in [8, 16, 32] {
                    let g = s / st;
                    for k in 0..g * g {
                        let r = &v[a * 5..][..5];
                        a += 1;
                        if r[0] < min {
                            continue;
                        }
                        let st = st as f32;
                        let c = Vec2::new(r[1] + (k % g) as f32, r[2] + (k / g) as f32) * st;
                        let h = Vec2::new(r[3].exp(), r[4].exp()) * st / 2.0;
                        b.push(([c.x - h.x, c.y - h.y, c.x + h.x, c.y + h.y], r[0]));
                    }
                }
                all.extend(
                    nms(b, 0.3)
                        .into_iter()
                        .map(|(b, score)| place(cam, b, s as f32, score)),
                );
            }
            self.spent(2, t.elapsed());
        }
        self.pv = Some(n);
        Ok(merge(all))
    }
}

/// Per tile non-maximum suppression on pixel boxes, best first.
pub(crate) fn nms(mut b: Vec<([f32; 4], f32)>, t: f32) -> Vec<([f32; 4], f32)> {
    b.sort_by(|x, y| y.1.total_cmp(&x.1));
    let mut keep: Vec<([f32; 4], f32)> = vec![];
    for x in b {
        if keep.iter().all(|k| px_iou(&k.0, &x.0) <= t) {
            keep.push(x);
        }
    }
    keep
}

/// A pixel box of a tile `s` pixels wide of `cam` on the sphere, and whether it touches the tile's edge.
pub(crate) fn place(cam: &Cam, b: [f32; 4], s: f32, score: f32) -> (bool, Region) {
    let b = b.map(|v| 2.0 * v / s - 1.0);
    let (dir, size) = cam.boxed(b);
    (
        b.iter().any(|v| v.abs() > 1.0 - 4.0 / s),
        Region { dir, size, score },
    )
}

/// Across tiles: a box touching its tile's edge yields to one that doesn't, then the higher score wins (overlap over
/// the smaller box > 0.5).
pub(crate) fn merge(mut all: Vec<(bool, Region)>) -> Vec<Region> {
    all.sort_by(|x, y| x.0.cmp(&y.0).then(y.1.score.total_cmp(&x.1.score)));
    let mut out: Vec<Region> = vec![];
    for (_, d) in all {
        if out.iter().all(|k| iou(k.on(), d.on(), true) <= 0.5) {
            out.push(d);
        }
    }
    out
}

/// Seconds: detections of one track at most this far apart are bridged; a track goes on this long before its first
/// and after its last detection, at the speed it came and went (the frames between detections the face or plate was
/// already or still there: a key frame's detection is too weak to start or continue a track, it is not gone).
const GAP: f32 = 1.0;
const PAD: f32 = 0.25;
/// Regions grow by this over the detected box (hair, ears, a chin turned away: the same both ways, since a box's
/// axes are its tile's, turned up to 27° against the region's upright frame on the tiles at ±60°), plus degrees (the
/// box's jitter, about the same in pixels for any face: ±4 px for the smallest ones). Tuned on faces; plates take the
/// same, checked on the samples' parked and slow cars only (tests/footage.rs `plates`), not on passing traffic near the
/// camera.
const GROW: f32 = 1.5;
const MARGIN: f32 = 1.0;
/// Plates: a detection reaches this many times its size for its track (a plate is small against its car's motion: one
/// 2.5° wide passing at 4° a searched frame fell into a track a detection, each padded where the plate was not), and a
/// track is hidden with this many `sure` detections only (a plate is found again and again; what is found once, or
/// weakly throughout, is ground or a fence, or a plate too small to read).
const REACH: f32 = 2.0;
const TWICE: usize = 2;
/// Plates, over time: a track of one detection has no motion to predict, so its second may lie this many degrees a
/// second farther off if within `SOON` s (a plate passed within metres sweeps 4° a searched frame, beyond `REACH` of a
/// 1.2° plate: each detection was a track of its own, held still over `PAD` on both sides, a row of blurs beside the
/// plate; over longer gaps the reach joined a stray detection to a plate 0.3 to 0.8 s away, blurring the path between
/// them); a track is carried `CARRY` s past its ends and bridged across gaps of `BRIDGE` s at most (beyond, the
/// straight path between two detections leaves a passing plate). Against the first two galleries' flags
/// (scripts/rule_eval.py), at the frames shown: blurs on nothing 23 to 14, on plates too small to read 26 to 19, plates
/// covered 239 to 241, marked missed plates uncovered 8 to 7, every plate of the second gallery still covered (without
/// the 30 px gate: 60 to 36, 81 to 67); the third gallery's (the views `SWEEP` without `SOON` changed): of its 14 blurs
/// flagged, 10 left, none lost of the 74 left unflagged.
const SWEEP: f32 = 30.0;
const SOON: f32 = 0.25;
const CARRY: f32 = 0.15;
const BRIDGE: f32 = 0.5;

/// Privacy regions of a clip: detections per frame (every frame, or every few: the tracks bridge them), linked over
/// frames; faces and plates each in a `Hide` of their own. Directions are in the frame of the equirect they were found
/// on: blur frames stitched with the same poses (another pose, say another stabilisation, moves them away from their
/// regions).
#[derive(Default)]
pub struct Hide {
    keys: Vec<(usize, Vec<Region>)>,
    /// Plates: the width of the pano they were found on (faces: 0).
    w: u32,
}

impl Hide {
    pub fn new() -> Self {
        Self::default()
    }

    /// For plates found on a pano `w` pixels wide (`REACH`, `TWICE`).
    pub fn plates(w: u32) -> Self {
        Self { keys: vec![], w }
    }

    /// The faces (or plates) of frame `k`, in frame order.
    pub fn push(&mut self, k: usize, d: Vec<Region>) {
        self.keys.push((k, d));
    }

    /// Tracks: each detection joins the nearest track whose last detection is at most `GAP` old and
    /// within reach of where its last motion puts it (its size, `REACH` times for plates, at least 2°, more by the time
    /// since; plates: `SWEEP` more from a track of one detection, within `SOON` s), the best scores first; of plates,
    /// the tracks with `TWICE` sure detections.
    pub fn tracks(&self, fps: f32) -> Vec<Vec<(usize, Region)>> {
        let (reach, sweep) = if self.w > 0 {
            (REACH, SWEEP)
        } else {
            (1.0, 0.0)
        };
        let mut tl: Vec<Vec<(usize, Region)>> = vec![];
        for (k, ds) in &self.keys {
            let mut ds = ds.clone();
            ds.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut used = vec![false; tl.len()];
            for d in ds {
                let best = tl
                    .iter()
                    .enumerate()
                    .filter(|(j, t)| {
                        let l = t[t.len() - 1].0;
                        !used[*j] && (*k - l) as f32 / fps <= GAP
                    })
                    .filter_map(|(j, t)| {
                        let g = (*k - t[t.len() - 1].0) as f32 / fps;
                        let a = at(t[t.len() - 1], t.iter().nth_back(1), *k)
                            .angle_between(d.dir)
                            .to_degrees()
                            / (1.0 + g);
                        let r = (d.size.max_element() * reach).max(2.0)
                            + if t.len() == 1 && g <= SOON {
                                sweep * g
                            } else {
                                0.0
                            };
                        (a <= r).then_some((j, a))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                match best {
                    Some((j, _)) => {
                        used[j] = true;
                        tl[j].push((*k, d));
                    }
                    None => {
                        used.push(true);
                        tl.push(vec![(*k, d)]);
                    }
                }
            }
        }
        tl.retain(|t| self.w == 0 || t.iter().filter(|d| sure(&d.1, self.w)).count() >= TWICE);
        tl
    }

    /// The regions to hide in each of `n` frames: every track, interpolated between its detections (direction along
    /// the great circle, size linearly), carried on `PAD` s before its first and after its last at the speed over its
    /// first and its last `PAD` s or more, grown by `GROW` and `MARGIN` (plates: `CARRY` s, and a gap over `BRIDGE` s
    /// ends one stretch and starts the next).
    pub fn regions(&self, n: usize, fps: f32) -> Vec<Vec<Region>> {
        let mut out = vec![vec![]; n];
        let (pad, bridge) = if self.w > 0 {
            (CARRY, BRIDGE)
        } else {
            (PAD, GAP)
        };
        let pad = (pad * fps).ceil() as usize;
        let r = |a: usize, b: usize| a.min(n)..b.min(n);
        let g = |dir, r: Region| Region {
            dir,
            size: r.size * GROW + MARGIN,
            score: r.score,
        };
        let tl = self.tracks(fps);
        for t in tl
            .iter()
            .flat_map(|t| t.chunk_by(|a, b| (b.0 - a.0) as f32 / fps <= bridge))
        {
            let (f0, f1, e) = (t[0].0, t[t.len() - 1].0, t.len() - 1);
            // the speed over at least the pad where the track is that long: between neighbouring detections, one
            // box's jitter (a head turning away grows it to one side) would swing it
            let y0 = t[1..].iter().find(|d| d.0 - f0 >= pad).or(t[1..].last());
            let y1 = t[..e]
                .iter()
                .rev()
                .find(|d| f1 - d.0 >= pad)
                .or(t[..e].first());
            let a0 = f0.saturating_sub(pad);
            for (f, o) in (a0..).zip(&mut out[r(a0, f0)]) {
                o.push(g(at(t[0], y0, f), t[0].1));
            }
            for w in t.windows(2) {
                let ((a, x), (b, y)) = (w[0], w[1]);
                let q = Quat::from_rotation_arc(x.dir, y.dir);
                for (f, o) in out[r(a, b)].iter_mut().enumerate() {
                    let s = f as f32 / (b - a) as f32;
                    let m = Region {
                        size: x.size.lerp(y.size, s),
                        score: x.score.min(y.score),
                        ..x
                    };
                    o.push(g(Quat::IDENTITY.slerp(q, s) * x.dir, m));
                }
            }
            for (f, o) in (f1..).zip(&mut out[r(f1, f1 + pad + 1)]) {
                o.push(g(at(t[e], y1, f), t[e].1));
            }
        }
        out
    }
}

/// Where a track at detection `x` is at frame `k`, moving at the speed from its neighbouring detection `y` to `x`
/// (on after a last detection, back before a first; still without `y`).
fn at((l, x): (usize, Region), y: Option<&(usize, Region)>, k: usize) -> Vec3 {
    let Some(&(p, y)) = y else {
        return x.dir;
    };
    let (a, v) = Quat::from_rotation_arc(y.dir, x.dir).to_axis_angle();
    Quat::from_axis_angle(a, v * (k as f32 - l as f32) / (l as f32 - p as f32)) * x.dir
}
