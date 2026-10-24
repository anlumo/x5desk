//! Auto framing: detections every few frames over the whole horizon are linked into tracklets, each scored per key
//! frame by class, confidence, size, motion, height above or below the horizon and persistence; the main subject
//! changes only when another one clearly wins for a while. The view follows it and fits its height, then the path is
//! smoothed over time in yaw and pitch, so that it stays level when it pans from one subject to another.

use crate::Det;
use glam::Vec3;

/// The view of one frame: direction (equirect frame, `yaw_pitch` for `stitch::View`), horizontal field of view in
/// degrees for the output aspect, and the tracklet it frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    pub dir: Vec3,
    pub fov: f32,
    pub subject: Option<usize>,
}

/// Seconds: a challenger has to win this long; a subject gone this long is given up; a tracklet unseen this long
/// ends; time constant of the path smoothing.
const SWITCH: f32 = 1.5;
const GONE: f32 = 1.0;
const END: f32 = 2.0;
const SIGMA: f32 = 0.6;
/// The subject fills this share of the frame height; the view's horizontal field of view stays within `FOV`.
const FILL: f32 = 0.45;
const FOV: [f32; 2] = [60.0, 110.0];
/// Lowest pitch (degrees) a view's lower edge reaches: below it is the nadir zone, the operator and the stick (on
/// v627 the operator's head is at about −78°). A view onto a close subject far below the horizon is raised until its
/// edge is clear, the subject lower in the frame.
pub const LOW: f32 = -65.0;

pub struct Framer {
    keys: Vec<(usize, Vec<Det>)>,
    /// Output width / height.
    pub aspect: f32,
}

/// Tracklet: detections linked over key frames (key index, detection).
struct Tl {
    hits: Vec<(usize, Det)>,
}

impl Framer {
    pub fn new(aspect: f32) -> Self {
        Self {
            keys: vec![],
            aspect,
        }
    }

    /// The detections of frame `k` (e.g. `Ai::detect` over `sphere()`), in frame order.
    pub fn push(&mut self, k: usize, d: Vec<Det>) {
        self.keys.push((k, d));
    }

    /// Class weight: people first, then animals, sports gear, vehicles, the rest.
    fn class(c: u32) -> f32 {
        match c {
            0 => 1.0,
            14..=23 => 0.8,
            29..=38 => 0.6,
            1..=8 => 0.4,
            _ => 0.15,
        }
    }

    /// One view per frame for `n` frames at `fps`.
    pub fn path(&self, n: usize, fps: f32) -> Vec<Shot> {
        let ks = &self.keys;
        // link: each detection to the nearest live tracklet of its class, predicted by its last motion
        let mut tl: Vec<Tl> = vec![];
        for (i, (k, ds)) in ks.iter().enumerate() {
            let mut ds = ds.clone();
            ds.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut used = vec![false; tl.len()];
            for d in ds {
                let best = tl
                    .iter()
                    .enumerate()
                    .filter(|(j, t)| {
                        let (l, x) = t.hits[t.hits.len() - 1];
                        !used[*j] && x.cls == d.cls && (*k - ks[l].0) as f32 / fps <= END
                    })
                    .map(|(j, t)| (j, Self::at(t, ks, *k).angle_between(d.dir).to_degrees()))
                    .filter(|&(_, a)| a <= (0.75 * d.size.max_element()).max(6.0))
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                match best {
                    Some((j, _)) => {
                        used[j] = true;
                        tl[j].hits.push((i, d));
                    }
                    None => {
                        used.push(true);
                        tl.push(Tl { hits: vec![(i, d)] });
                    }
                }
            }
        }
        // score per key, pick with hysteresis; per key: frame, view (yaw unwrapped against the last key, pitch, log
        // field of view), subject
        let mut keys: Vec<(usize, Vec3, Option<usize>)> = vec![];
        let (mut cur, mut since, mut lost) = (None::<usize>, 0.0f32, 0.0f32);
        let (mut dir, mut h) = (crate::dir(0.0, 0.0), 0.0);
        for (i, (k, _)) in ks.iter().enumerate() {
            let dt = if i > 0 {
                (k - ks[i - 1].0) as f32 / fps
            } else {
                0.0
            };
            let sc: Vec<(usize, f32, Det)> = tl
                .iter()
                .enumerate()
                .filter_map(|(j, t)| {
                    let p = t.hits.iter().position(|h| h.0 == i)?;
                    let d = t.hits[p].1;
                    // motion over the last second, in subject heights per second
                    let q = t.hits[..p]
                        .iter()
                        .rfind(|h| (k - ks[h.0].0) as f32 / fps >= 1.0)
                        .or(t.hits[..p].first());
                    let m = q.map_or(0.0, |q| {
                        q.1.dir.angle_between(d.dir).to_degrees()
                            / ((k - ks[q.0].0) as f32 / fps)
                            / d.size.y.max(5.0)
                    });
                    let age = t.hits[..=p]
                        .iter()
                        .filter(|h| (k - ks[h.0].0) as f32 / fps <= 2.0)
                        .count() as f32;
                    let pitch = crate::yaw_pitch(d.dir)[1].abs();
                    let s = Self::class(d.cls)
                        * d.score
                        * (d.size.y / 20.0).min(1.5)
                        * (1.0 + 0.5 * m.min(2.0))
                        * ((70.0 - pitch) / 30.0).clamp(0.1, 1.0)
                        * (0.5 + 0.5 * (age * Self::spacing(ks, fps) / 2.0).min(1.0));
                    Some((j, s, d))
                })
                .collect();
            let best = sc.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1));
            let now = cur.and_then(|c| sc.iter().find(|s| s.0 == c).copied());
            match (now, best) {
                (Some(c), Some(b)) => {
                    since = if b.0 != c.0 && b.1 > 1.5 * c.1 {
                        since + dt
                    } else {
                        0.0
                    };
                    lost = 0.0;
                    if since >= SWITCH {
                        (cur, since) = (Some(b.0), 0.0);
                    }
                }
                (None, Some(b)) => {
                    lost += dt;
                    if cur.is_none() || lost >= GONE {
                        (cur, lost, since) = (Some(b.0), 0.0, 0.0);
                    }
                }
                _ => lost += dt,
            }
            if let Some(d) = cur.and_then(|c| sc.iter().find(|s| s.0 == c)).map(|s| s.2) {
                (dir, h) = (d.dir, d.size.y);
            }
            let v = (h / FILL).clamp(1.0, 170.0).to_radians() / 2.0;
            let fov = (2.0 * (v.tan() * self.aspect).atan())
                .to_degrees()
                .clamp(FOV[0], FOV[1]);
            // half the vertical field of view
            let v = ((fov / 2.0).to_radians().tan() / self.aspect)
                .atan()
                .to_degrees();
            let [y, p] = crate::yaw_pitch(dir);
            let y = keys
                .last()
                .map_or(y, |l| y + 360.0 * ((l.1.x - y) / 360.0).round());
            keys.push((*k, Vec3::new(y, p.max(LOW + v), fov.ln()), cur));
        }
        // per frame: hold before the first key and after the last, interpolate between, then smooth, all in yaw,
        // pitch and log field of view: the view pans level from one subject to the next, where the great-circle arc
        // between two far apart below the horizon dips through the nadir (the ground, the operator). Both keep every
        // view clear of `LOW`: half the vertical field of view is convex in log field of view, so a mean of views
        // with pitch ≥ LOW + half vertical fov is one too
        let mut raw = Vec::with_capacity(n);
        let mut j = 0;
        for f in 0..n {
            while j + 1 < keys.len() && keys[j + 1].0 <= f {
                j += 1;
            }
            let Some(&(k0, v0, s0)) = keys.get(j) else {
                raw.push((Vec3::new(0.0, 0.0, FOV[1].ln()), None));
                continue;
            };
            let (k1, v1, _) = keys.get(j + 1).copied().unwrap_or(keys[j]);
            let t = if k1 > k0 && f > k0 {
                ((f - k0) as f32 / (k1 - k0) as f32).min(1.0)
            } else {
                0.0
            };
            raw.push((v0.lerp(v1, t), s0));
        }
        let s = SIGMA * fps;
        let r = (3.0 * s).ceil() as isize;
        (0..n as isize)
            .map(|k| {
                let (mut v, mut ws) = (Vec3::ZERO, 0.0);
                for j in (k - r).max(0)..(k + r + 1).min(n as isize) {
                    let w = (-((j - k) as f32).powi(2) / (2.0 * s * s)).exp();
                    v += w * raw[j as usize].0;
                    ws += w;
                }
                let v = v / ws;
                Shot {
                    dir: crate::dir(v.x, v.y),
                    fov: v.z.exp(),
                    subject: raw[k as usize].1,
                }
            })
            .collect()
    }

    /// Where tracklet `t` should be at frame `k`: its last direction moved on by its last motion.
    fn at(t: &Tl, ks: &[(usize, Vec<Det>)], k: usize) -> Vec3 {
        let (l, x) = t.hits[t.hits.len() - 1];
        let Some(&(p, y)) = t.hits.len().checked_sub(2).map(|i| &t.hits[i]) else {
            return x.dir;
        };
        let q = glam::Quat::from_rotation_arc(y.dir, x.dir);
        let s = (k - ks[l].0) as f32 / (ks[l].0 - ks[p].0).max(1) as f32;
        glam::Quat::IDENTITY.slerp(q, s.min(2.0)) * x.dir
    }

    /// Mean seconds between key frames.
    fn spacing(ks: &[(usize, Vec<Det>)], fps: f32) -> f32 {
        let n = ks.len().max(2) - 1;
        ks.last()
            .map_or(1.0, |l| (l.0 - ks[0].0) as f32 / fps / n as f32)
            .max(1.0 / fps)
    }
}
