//! Auto framing: detections every few frames over the whole horizon are linked into tracklets, each scored per key
//! frame by class, confidence, size, motion, height above or below the horizon and persistence; the main subject
//! changes only when another one clearly wins for a while. The camera's carrier (the rider, the hand on the stick, the
//! vehicle), told by the camera body's frame (`Framer::body`), is never a subject, and while the camera travels
//! (`Framer::moving`) a subject has to earn the view over the direction of travel. The view follows the subject and
//! fits its height, pans to the next at most `TURN` °/s, then the path is smoothed over time in yaw and pitch, so that
//! it stays level when it pans from one subject to another. Where no subject wins (none yet, or the subject gone), the
//! view looks along `Framer::ahead` (the forward view) if given.

use crate::Det;
use glam::{Quat, Vec2, Vec3};

/// The view of one frame: direction (equirect frame, `yaw_pitch` for `stitch::View`), horizontal field of view in
/// degrees for the output aspect, and the tracklet it frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    pub dir: Vec3,
    pub fov: f32,
    pub subject: Option<usize>,
}

/// A vehicle overtaking us close by (`Framer::passes`).
#[derive(Clone, Debug, PartialEq)]
pub struct Pass {
    /// Frames of its last detection in the rear quarter and its first in the front one, and where it came abeam.
    pub a: usize,
    pub b: usize,
    pub at: usize,
    /// On the right of the travel, else the left.
    pub right: bool,
    /// Its lateral distance between the quarters, camera heights.
    pub near: f32,
    /// Its detections from the first to the last: frame, direction (equirect frame), angular size (degrees).
    pub path: Vec<(usize, Vec3, Vec2)>,
}

/// Seconds: a challenger has to win this long; a subject gone this long is given up; a tracklet unseen this long
/// ends; time constant of the path smoothing.
const SWITCH: f32 = 1.5;
const GONE: f32 = 1.0;
const END: f32 = 2.0;
const SIGMA: f32 = 0.6;
/// Fastest turn of the view, °/s: a still scene crosses a 100° view in 2.5 s; a hand-over of 90° takes 2.3 s, of 180°
/// 4.5 s (the hand-overs of auto edit's first framer peaked at 84–110 °/s).
const TURN: f32 = 40.0;
/// While the camera travels a challenger has to earn the view over the direction of travel: a person or an animal
/// (class weight 0.8 or more) `PROM` degrees tall (about 5 m away) at every detection for `EARN` s; the subject stays
/// while `PROM / 2` tall, and once the forward view is back the next one earns it afresh.
const PROM: f32 = 20.0;
const EARN: f32 = 1.5;
/// The subject fills this share of the frame height; the view's horizontal field of view stays within `FOV`.
const FILL: f32 = 0.45;
const FOV: [f32; 2] = [60.0, 110.0];
/// The camera's carrier (`carriers`), degrees unless said: tall or low (medians over a tracklet's detections), fixed to
/// the body (median over its detections of the rms angle from their mean of its body-frame directions within a second),
/// seconds a seed is seen at least, the reach of a seed in the body frame, the least height of a carrier not close,
/// the radius and share of travelling keys of its presence. On 3 sample clips and 4 rides: the rider's tracklets
/// 46–97° tall (medians; the head 26–38°), at 1–22°; the hand on the stick and the scooter's parts at −42° to −58°,
/// at 1–9°; people filmed on purpose 15–33° tall at −14° to −32°, the cyclist at a crossing 37°; people seen where the
/// rider's head is in 26–60 % of a ride's travelling keys, where the cyclist and a walking companion are in 10–15 %.
const NEAR: f32 = 40.0;
const UNDER: f32 = -40.0;
const FIXED: f32 = 25.0;
const SEED: f32 = 2.0;
const SAME: f32 = 30.0;
const TALL: f32 = 20.0;
const ROUND: f32 = 15.0;
const PRESENT: f32 = 0.2;
/// Passes (`passes`): the vehicles (COCO bicycle, car, motorcycle, bus, truck); a quarter's half width (degrees); key
/// frames a track goes unseen at most; the most a track's lateral distances differ (ratio); the farthest lateral
/// distance (camera heights); the fastest a vehicle passes (camera heights a second) and the longest it takes from
/// quarter to quarter (s); seconds between two tracks of one vehicle coming abeam. On the 36 scooter rides of five
/// days, the seven vehicles overtaking within `CLOSE` passed 1.8–2.8 away at 0.6–6.8 a second in 1.2–5.0 s, those in
/// the next lane 3.1–6.0 away at up to 8.9; a track hopping along a parked row passed at 39 a second, a moped waiting
/// beside us at the lights took 30 s; with a ratio of 1.75 two of the seven were lost, while 1.5 keeps a car passing
/// 2.2 away apart from the parked one 5.5 away it crosses.
const VEHICLE: [u32; 5] = [1, 2, 3, 5, 7];
const QUARTER: f32 = 45.0;
const GAP: usize = 2;
const RATIO: f32 = 1.5;
const CLOSE: f32 = 3.0;
const FAST: f32 = 12.0;
const LONG: f32 = 8.0;
const TWICE: f32 = 0.4;
/// Lowest pitch (degrees) a view's lower edge reaches: below it is the nadir zone, the operator and the stick (on
/// v627 the operator's head is at about −78°). A view onto a close subject far below the horizon is raised until its
/// edge is clear, the subject lower in the frame.
pub const LOW: f32 = -65.0;

pub struct Framer {
    keys: Vec<(usize, Vec<Det>)>,
    /// Output width / height.
    pub aspect: f32,
    /// The direction of travel per frame (equirect frame), the view where no subject wins; empty: the last
    /// subject's direction holds (the front before the first).
    pub ahead: Vec<Vec3>,
    /// Per frame, the rotation from the equirect frame into the camera body's (up to a fixed one); empty: no
    /// tracklet is taken for the camera's carrier.
    pub body: Vec<Quat>,
    /// Per frame, whether the camera travels (the forward view's analysis); empty: it stands.
    pub moving: Vec<bool>,
    /// Per frame, the rotation from the equirect frame into the travel's: level, the direction of travel at yaw 0
    /// (`ahead`); empty: no passes are found.
    pub fwd: Vec<Quat>,
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
            ahead: vec![],
            body: vec![],
            moving: vec![],
            fwd: vec![],
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

    /// The tracklets: each detection linked to the nearest live tracklet of its class, predicted by its last motion.
    fn link(&self, fps: f32) -> Vec<Tl> {
        let ks = &self.keys;
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
        tl
    }

    /// One view per frame for `n` frames at `fps`.
    pub fn path(&self, n: usize, fps: f32) -> Vec<Shot> {
        let ks = &self.keys;
        let tl = self.link(fps);
        let (carrier, seeds) = self.carriers(&tl, fps);
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
            let tr = self.moving.get(*k) == Some(&true);
            let sc: Vec<(usize, f32, Det)> = tl
                .iter()
                .enumerate()
                .filter(|(j, _)| !carrier[*j])
                .filter_map(|(j, t)| {
                    let p = t.hits.iter().position(|h| h.0 == i)?;
                    let d = t.hits[p].1;
                    // nor is a detection, close, where a person's seed sits: a tracklet that hopped onto the rider
                    let at = |q: &Quat| {
                        seeds
                            .iter()
                            .any(|s| s.angle_between(*q * d.dir).to_degrees() <= SAME)
                    };
                    let close = d.size.y >= NEAR || crate::yaw_pitch(d.dir)[1] <= UNDER;
                    if close && self.body.get(*k).is_some_and(at) {
                        return None;
                    }
                    // travelling: a challenger has earned it at every detection for `EARN` s (no gap of `GONE`), the
                    // subject stays while `PROM / 2` tall
                    let earns = |q: usize| {
                        Self::class(t.hits[q].1.cls) >= 0.8 && t.hits[q].1.size.y >= PROM
                    };
                    let mut q = p;
                    while q > 0
                        && earns(q - 1)
                        && (ks[t.hits[q].0].0 - ks[t.hits[q - 1].0].0) as f32 / fps <= GONE
                    {
                        q -= 1;
                    }
                    let earned = earns(p) && (k - ks[t.hits[q].0].0) as f32 / fps >= EARN;
                    if tr && (cur == Some(j) && d.size.y < PROM / 2.0 || cur != Some(j) && !earned)
                    {
                        return None;
                    }
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
            let won = cur.and_then(|c| sc.iter().find(|s| s.0 == c)).map(|s| s.2);
            if let Some(d) = won {
                (dir, h) = (d.dir, d.size.y);
            }
            let ahead = (cur.is_none() || won.is_none() && lost >= GONE)
                .then(|| self.ahead.get(*k))
                .flatten();
            if let Some(&a) = ahead {
                dir = a;
                // travelling, the next subject has to earn the view from the direction of travel
                if tr {
                    cur = None;
                }
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
            keys.push((
                *k,
                Vec3::new(y, p.max(LOW + v), fov.ln()),
                cur.filter(|_| ahead.is_none()),
            ));
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
        // hand-overs pan: each view moved from the last by at most TURN / fps in yaw and pitch (never before the switch,
        // which comes `GONE` after a subject was lost or `SWITCH` after a challenger began to win), before the
        // smoothing, which never turns faster; every view a convex combination of the raw ones, so clear of `LOW` too
        for i in 1..n {
            let d = raw[i].0 - raw[i - 1].0;
            raw[i].0 = raw[i - 1].0 + d * (TURN / fps / d.truncate().length()).min(1.0);
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

    /// The vehicles overtaking us close by, in the order they come abeam (none without `fwd`). Their detections in the
    /// travel's frame are linked apart from the framer's tracklets: a box whose centre lies inside a wider one's on the
    /// same side is part of that vehicle (cut at a tile's edge), and a vehicle track takes, on its side, the detection
    /// nearest where its last motion puts it (as `link` does, any vehicle class) whose lateral distance is within
    /// `RATIO` of its own (the median of its last three): its box's lower edge on the ground, in camera heights. A
    /// track unseen for more than `GAP` key frames ends. One that goes from the rear quarter (within `QUARTER`° of
    /// straight behind) to the front one overtakes us, `CLOSE` away or nearer (the median between the quarters), at
    /// `FAST` or slower (the median rate of the cotangent of its azimuth off the travel, on the ground along / lateral
    /// distance, by that distance), within `LONG` s: a parked car only ever goes the other way, one we pass is not told
    /// from it, a track that hops along a parked row jumps, and one that waits beside us at the lights takes longer. A
    /// vehicle seen twice passes once (the longer track).
    pub fn passes(&self, fps: f32) -> Vec<Pass> {
        // key, frame, azimuth off the travel (degrees, right +), pitch, lateral distance, detection
        type V = (usize, usize, f32, f32, f32, Det);
        let (mut tr, mut live): (Vec<Vec<V>>, Vec<usize>) = (vec![], vec![]);
        for (i, (k, ds)) in self.keys.iter().enumerate() {
            let Some(q) = self.fwd.get(*k) else { continue };
            let mut v: Vec<V> = (ds.iter())
                .filter(|d| VEHICLE.contains(&d.cls))
                .filter_map(|d| {
                    let [a, p] = crate::yaw_pitch(*q * d.dir);
                    let e = (d.size.y / 2.0 - p).to_radians();
                    (e > 0.02).then(|| (i, *k, a, p, a.to_radians().sin().abs() / e.tan(), *d))
                })
                .collect();
            v.sort_by(|a, b| b.5.size.x.total_cmp(&a.5.size.x));
            let mut m: Vec<V> = vec![];
            for x in v {
                let part = |w: &V| {
                    w.2.signum() == x.2.signum()
                        && (w.2 - x.2).abs() <= w.5.size.x / 2.0
                        && (w.3 - x.3).abs() <= w.5.size.y / 2.0
                };
                if !m.iter().any(part) {
                    m.push(x);
                }
            }
            m.sort_by(|a, b| b.5.score.total_cmp(&a.5.score));
            live.retain(|&t| i - tr[t][tr[t].len() - 1].0 <= GAP);
            let mut used = vec![];
            for x in m {
                let best = (live.iter().filter(|t| !used.contains(*t)))
                    .filter_map(|&t| {
                        let s = &tr[t];
                        let l = s[s.len() - 1];
                        let y = med(s[s.len().saturating_sub(3)..].iter().map(|w| w.4).collect());
                        let a = match s.len() {
                            1 => l.2,
                            n => {
                                l.2 + (l.2 - s[n - 2].2) / (l.0 - s[n - 2].0) as f32
                                    * (i - l.0).min(2) as f32
                            }
                        };
                        let da = (x.2 - a).abs();
                        (l.2.signum() == x.2.signum()
                            && y.max(x.4) <= RATIO * y.min(x.4)
                            && da <= (0.75 * x.5.size.max_element()).max(6.0))
                        .then_some((t, da))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1));
                let t = best.map_or_else(
                    || {
                        tr.push(vec![]);
                        live.push(tr.len() - 1);
                        tr.len() - 1
                    },
                    |b| b.0,
                );
                used.push(t);
                tr[t].push(x);
            }
        }
        // 0 the front quarter, 2 the rear one, 1 between; the cotangent of the azimuth, through 0 abeam
        let q = |x: &V| match x.2.abs() {
            a if a <= QUARTER => 0,
            a if a >= 180.0 - QUARTER => 2,
            _ => 1,
        };
        let c = |x: &V| 1.0 / x.2.abs().to_radians().tan();
        let mut all: Vec<(usize, f32, Pass)> = vec![];
        for s in &tr {
            // from the last detection in the rear quarter before the first in the front one
            let Some(b) = (s.iter().position(|x| q(x) == 2))
                .and_then(|a| s[a..].iter().position(|x| q(x) == 0).map(|b| a + b))
            else {
                continue;
            };
            let a = (0..b).rev().find(|&x| q(&s[x]) == 2).unwrap();
            let m = &s[a..=b];
            let mut y: Vec<f32> = m.iter().filter(|x| q(x) == 1).map(|x| x.4).collect();
            if y.len() < 2 {
                y = m.iter().map(|x| x.4).collect();
            }
            let near = med(y);
            // how fast it passes, camera heights a second: the cotangent's median rate over lateral distance
            let fast = near
                * med((m.windows(2))
                    .map(|w| (c(&w[1]) - c(&w[0])) / (w[1].1 - w[0].1) as f32 * fps)
                    .collect());
            let w = m.windows(2).find(|w| c(&w[0]) * c(&w[1]) <= 0.0).unwrap();
            let at = w[0].1 as f32 + (w[1].1 - w[0].1) as f32 * c(&w[0]) / (c(&w[0]) - c(&w[1]));
            if near <= CLOSE && fast <= FAST && (m[m.len() - 1].1 - m[0].1) as f32 / fps <= LONG {
                all.push((
                    s.len(),
                    at,
                    Pass {
                        a: m[0].1,
                        b: m[m.len() - 1].1,
                        at: at.round() as usize,
                        right: m[0].2 > 0.0,
                        near,
                        path: s.iter().map(|x| (x.1, x.5.dir, x.5.size)).collect(),
                    },
                ));
            }
        }
        all.sort_by_key(|x| std::cmp::Reverse(x.0));
        let mut out: Vec<(f32, Pass)> = vec![];
        for (_, at, p) in all {
            if !(out.iter()).any(|x| (x.0 - at).abs() / fps < TWICE && x.1.right == p.right) {
                out.push((at, p));
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out.into_iter().map(|x| x.1).collect()
    }

    /// Which tracklets are the camera's carrier. A seed is fixed to the camera body (`FIXED`) for `SEED` s at least and
    /// `UNDER` the horizon (median: the hand on the stick, the vehicle) or, mostly seen travelling, `NEAR` tall (median:
    /// the rider) or `TALL` where people are seen in the body frame (within `ROUND`) in `PRESENT` of the clip's
    /// travelling keys (the rider's head is there all ride, people passing are not); standing, someone close is not
    /// told from the carrier. The carrier too is a tracklet close (`UNDER` or `NEAR`), or travelling, fixed and `TALL`,
    /// where a person's seed sits in the body frame (within `SAME`): the rider's detections hop between legs, hands,
    /// torso and head, and the rider stands at the lights where it rode. None without `body`. Also returns the person
    /// seeds' mean body-frame directions.
    fn carriers(&self, tl: &[Tl], fps: f32) -> (Vec<bool>, Vec<Vec3>) {
        let ks = &self.keys;
        let mv = |k: usize| self.moving.get(k) == Some(&true);
        let med = |mut v: Vec<f32>| {
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        };
        let tk: Vec<&(usize, Vec<Det>)> = ks
            .iter()
            .filter(|x| mv(x.0) && x.0 < self.body.len())
            .collect();
        // share of the travelling keys with a person within `ROUND` of body-frame direction `m`
        let present = |m: Vec3| {
            let on = |x: &&&(usize, Vec<Det>)| {
                (x.1.iter()).any(|d| {
                    d.cls == 0 && (self.body[x.0] * d.dir).angle_between(m).to_degrees() <= ROUND
                })
            };
            tk.iter().filter(on).count() as f32 / tk.len().max(1) as f32
        };
        // per tracklet: close, fixed and tall, a seed, its mean body-frame direction
        let st: Vec<_> = (tl.iter())
            .map(|t| {
                let k = |i: usize| ks[t.hits[i].0].0;
                let b = (0..t.hits.len())
                    .map(|i| Some(*self.body.get(k(i))? * t.hits[i].1.dir))
                    .collect::<Option<Vec<Vec3>>>()?;
                let h = med(t.hits.iter().map(|h| h.1.size.y).collect());
                let low = med(t
                    .hits
                    .iter()
                    .map(|h| crate::yaw_pitch(h.1.dir)[1])
                    .collect())
                    <= UNDER;
                // rms angle from their mean of the body-frame directions within a second
                let spread = |i: usize| {
                    let w: Vec<Vec3> = (0..b.len())
                        .filter(|&j| k(j).abs_diff(k(i)) as f32 <= fps)
                        .map(|j| b[j])
                        .collect();
                    let m = w.iter().sum::<Vec3>().normalize();
                    let s = w.iter().map(|d| d.angle_between(m).powi(2)).sum::<f32>();
                    (s / w.len() as f32).sqrt().to_degrees()
                };
                let fixed = med((0..b.len()).map(spread).collect()) <= FIXED;
                let m = b.iter().sum::<Vec3>().normalize();
                let long = (k(b.len() - 1) - k(0)) as f32 / fps >= SEED;
                let travels = 2 * (0..b.len()).filter(|&i| mv(k(i))).count() >= b.len();
                let seed = fixed
                    && long
                    && (low || travels && (h >= NEAR || h >= TALL && present(m) >= PRESENT));
                // a person's: other classes (a near arm taken for a refrigerator) don't hop
                Some((
                    low || h >= NEAR,
                    travels && fixed && h >= TALL,
                    seed,
                    seed && t.hits[0].1.cls == 0,
                    m,
                ))
            })
            .collect();
        let seeds: Vec<Vec3> = st.iter().flatten().filter(|x| x.3).map(|x| x.4).collect();
        let c = (st.iter())
            .map(|x| {
                x.is_some_and(|(near, tall, seed, _, m)| {
                    seed || (near || tall)
                        && seeds
                            .iter()
                            .any(|s| s.angle_between(m).to_degrees() <= SAME)
                })
            })
            .collect();
        (c, seeds)
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

/// The median of `v` (the upper one of an even count), 0 for none.
fn med(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}
