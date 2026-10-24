//! Reframing: views, keyframes and their interpolation, camera-movement presets.
//!
//! Interpolation: every segment between two keys is a cubic Hermite curve: on the rotation group squad with its
//! control points placed from the end tangents (exactly slerp when both tangents are the chord), on fov and the
//! projection axis a scalar cubic. The key's [`Ease`] picks the tangents of the segment that follows it on screen.
//! Auto tangents (`Smooth`) are the time-weighted mean of the neighbouring chord velocities, zero where the motion
//! reverses and shortened to within one chord velocity of each chord (a scalar: at most twice the slower one): no
//! overshoot on fov, no drift between two equal keys, constant speed along evenly spaced keys, no flips in squad.
//!
//! Projections lie on one axis so they blend: Panini(s) at −s, perspective at 0, Planet(d) at +d, so a Panini →
//! tiny-planet segment passes through rectilinear. Equirect has no neighbours there; it switches halfway.

use crate::{Error, GAP, MAX_T, Result, bad};
use glam::{DQuat, DVec3, EulerRot, Quat};
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Proj {
    /// Rectilinear.
    Persp,
    /// Cylindrical Panini, distance 0 (rectilinear) ..= 1 (`stitch::View::panini`).
    Panini(f32),
    /// Spherical general perspective from eye distance d: ρ = (d + 1)·sin θ / (d + cos θ); 0 rectilinear,
    /// 1 stereographic (`stitch::Mode::Planet`; a tiny planet when looking down).
    Planet(f32),
    Equirect,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// View camera (x right, y up, z forward) -> sphere. Identity looks at the panorama centre, level;
    /// = Ry(yaw)·Rx(−pitch)·Rz(roll) of `stitch::View` ([`View::ypr`]).
    pub q: Quat,
    /// Horizontal field of view, degrees.
    pub fov: f32,
    pub proj: Proj,
}

impl Default for View {
    fn default() -> Self {
        View {
            q: Quat::IDENTITY,
            fov: 100.0,
            proj: Proj::Persp,
        }
    }
}

impl View {
    /// From `stitch::View` angles in degrees: yaw turns right, pitch looks up, roll turns the image.
    pub fn look(yaw: f32, pitch: f32, roll: f32, fov: f32, proj: Proj) -> View {
        let r = f32::to_radians;
        View {
            q: Quat::from_euler(EulerRot::YXZ, r(yaw), -r(pitch), r(roll)),
            fov,
            proj,
        }
    }

    /// Yaw, pitch, roll in degrees for `stitch::View` (at pitch ±90° yaw and roll share an axis; the rotation is
    /// the same).
    pub fn ypr(&self) -> [f32; 3] {
        let (y, x, z) = self.q.to_euler(EulerRot::YXZ);
        [y.to_degrees(), -x.to_degrees(), z.to_degrees()]
    }
}

/// How the segment after a key moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ease {
    /// Constant speed to the next key (slerp / lerp).
    Linear,
    /// Auto tangents at both ends: C¹ through keys whose neighbouring segments are smooth too.
    Smooth,
    /// Starts at rest, auto tangent at the end.
    In,
    /// Auto tangent at the start, arrives at rest.
    Out,
    /// Starts and arrives at rest.
    InOut,
    /// Stays at this key's view until the next key.
    Hold,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    /// Track time, s (content time on a clip, output time on the output track).
    pub t: f64,
    pub v: View,
    pub e: Ease,
}

/// Log on the unit quaternions (S³, not the rotations): length in [0, π], continuous everywhere but at −1.
fn ln(q: DQuat) -> DVec3 {
    let v = DVec3::new(q.x, q.y, q.z);
    let s = v.length();
    if s < 1e-12 { v } else { v * (s.atan2(q.w) / s) }
}

/// Rotation log: half the rotation vector, shortest way round.
fn log(q: DQuat) -> DVec3 {
    ln(if q.w < 0.0 { -q } else { q })
}

fn exp(v: DVec3) -> DQuat {
    let a = v.length();
    let s = if a < 1e-12 { 1.0 } else { a.sin() / a };
    DQuat::from_xyzw(v.x * s, v.y * s, v.z * s, a.cos())
}

/// Rotation angle between two orientations, radians.
pub fn angle(a: Quat, b: Quat) -> f64 {
    2.0 * log(a.as_dquat().normalize().inverse() * b.as_dquat().normalize()).length()
}

/// Projection axis: Panini(s) −s, perspective 0, Planet(d) +d; None for equirect.
fn axis(p: Proj) -> Option<f64> {
    match p {
        Proj::Persp => Some(0.0),
        Proj::Panini(s) => Some(-s as f64),
        Proj::Planet(d) => Some(d as f64),
        Proj::Equirect => None,
    }
}

/// Tangent at a key from the chord velocities into (`a`, over `ha` s) and out of it (`b`, over `hb` s): their
/// time-weighted mean m, shortened to λm with |λm − v| ≤ |v| for both chords v (λ ≤ 2 m·v / |m|²). On a scalar that
/// is at most twice the slower chord, so cubics stay monotone; on rotations it keeps squad regular (see [`at`]).
fn tangent(a: Option<DVec3>, b: Option<DVec3>, ha: f64, hb: f64) -> DVec3 {
    match (a, b) {
        (Some(a), Some(b)) if a.dot(b) > 0.0 => {
            let m = (hb * a + ha * b) / (ha + hb);
            m * (2.0 * m.dot(a).min(m.dot(b)) / m.length_squared()).min(1.0)
        }
        (Some(v), None) | (None, Some(v)) => v,
        _ => DVec3::ZERO,
    }
}

/// The view at track time `t` on keys sorted by time; None without keys. Constant before the first and after the
/// last key. `rev`: the track plays backwards (a reversed clip's source time), so each key's ease shapes the
/// segment before it in track time, the one that follows it on screen.
pub fn at(keys: &[Key], t: f64, rev: bool) -> Option<View> {
    let n = keys.len();
    if n == 0 {
        return None;
    }
    // key m in playing order, on a time axis that grows while playing
    let k = |m: usize| match rev {
        false => keys[m],
        true => Key {
            t: -keys[n - 1 - m].t,
            ..keys[n - 1 - m]
        },
    };
    let (t, i) = match rev {
        false => (t, keys.partition_point(|k| k.t <= t)),
        true => (-t, n - keys.partition_point(|k| k.t < t)),
    };
    if i == 0 || i == n || k(i - 1).e == Ease::Hold {
        return Some(k(i.max(1) - 1).v);
    }
    let j = i - 1;
    let (k0, k1) = (k(j), k(i));
    let h = k1.t - k0.t;
    let u = (t - k0.t) / h;
    let hs = |m: usize| k(m + 1).t - k(m).t;
    let dq = |m: usize| k(m).v.q.as_dquat().normalize();
    // segment m, if it exists and moves
    let live = |m: Option<usize>| m.filter(|&m| m + 1 < n && k(m).e != Ease::Hold);
    // auto tangent at key m for a channel whose chord velocity on segment m is `f(m)`
    let auto = |m: usize, f: &dyn Fn(usize) -> Option<DVec3>| {
        let p = live(m.checked_sub(1));
        let q = live(Some(m));
        tangent(
            p.and_then(f),
            q.and_then(f),
            p.map_or(0.0, hs),
            q.map_or(0.0, hs),
        )
    };
    // start and end tangents of segment j, per s
    let ends = |f: &dyn Fn(usize) -> Option<DVec3>| {
        let c = f(j).unwrap_or_default();
        match k0.e {
            Ease::Linear => (c, c),
            Ease::In => (DVec3::ZERO, auto(i, f)),
            Ease::Out => (auto(j, f), DVec3::ZERO),
            Ease::InOut | Ease::Hold => (DVec3::ZERO, DVec3::ZERO),
            Ease::Smooth => (auto(j, f), auto(i, f)),
        }
    };
    let (u2, u3) = (u * u, u * u * u);
    let herm = |y0: f64, y1: f64, (a, b): (DVec3, DVec3)| {
        (2.0 * u3 - 3.0 * u2 + 1.0) * y0
            + (u3 - 2.0 * u2 + u) * h * a.x
            + (3.0 * u2 - 2.0 * u3) * y1
            + (u3 - u2) * h * b.x
    };

    // squad(p, q, s0, s1, u) = slerp(slerp(p, q, u), slerp(s0, s1, u), 2u(1 − u)) leaves p at rate L + 2·log(p⁻¹s0)
    // and reaches q at L − 2·log(q⁻¹s1) (per segment, body frames, L = log(p⁻¹q)), so the control points come from
    // the Hermite tangents A, B: s0 = p·exp(x), s1 = q·exp(y), x = (A − L)/2, y = (L − B)/2. A = B = L is slerp.
    // Only the key step takes the short way (q on p's side of S³); the other slerps are S³ geodesics between those
    // lifts: a rotation slerp flips its way round where its ends pass 180° apart, which for the third happens
    // mid-segment. On S³ a slerp is singular only between antipodal ends. Every end tangent has |A − L| ≤ |L|
    // (`tangent`; 0 and L too), so |x|, |y| ≤ |L|/2 and by the triangle inequality the third slerp's ends stay at most
    // 2|L| apart on S³: under π for any key step under 180°. On collinear keys squad is the Hermite cubic exactly.
    let p = dq(j);
    let (a, b) = ends(&|m| Some(log(dq(m).inverse() * dq(m + 1)) / hs(m)));
    let l = log(p.inverse() * dq(i));
    let q = p * exp(l);
    let (x, y) = ((a * h - l) / 2.0, (l - b * h) / 2.0);
    let (s0, s1) = (p * exp(x), q * exp(y));
    let pq = p * exp(l * u);
    let s = s0 * exp(ln(s0.inverse() * s1) * u);
    let r = pq * exp(ln(pq.inverse() * s) * (2.0 * u * (1.0 - u)));

    let fv = |m: usize| k(m).v.fov as f64;
    let fov = herm(
        fv(j),
        fv(i),
        ends(&|m| Some(DVec3::X * (fv(m + 1) - fv(m)) / hs(m))),
    );
    let ax = |m: usize| Some(DVec3::X * (axis(k(m + 1).v.proj)? - axis(k(m).v.proj)?) / hs(m));
    let proj = match (axis(k0.v.proj), axis(k1.v.proj)) {
        (Some(x0), Some(x1)) => match herm(x0, x1, ends(&ax)) as f32 {
            x if x > 1e-6 => Proj::Planet(x.min(1.0)),
            x if x < -1e-6 => Proj::Panini((-x).min(1.0)),
            _ => Proj::Persp,
        },
        _ if u < 0.5 => k0.v.proj,
        _ => k1.v.proj,
    };
    Some(View {
        q: r.normalize().as_quat(),
        fov: fov as f32,
        proj,
    })
}

/// Checks a key track: times within ±[`MAX_T`], at least [`GAP`] apart; unit quaternions; fov in (0, 360];
/// projection parameters in 0..=1.
pub fn check(keys: &[Key]) -> Result<()> {
    for (i, k) in keys.iter().enumerate() {
        let v = &k.v;
        let d = match v.proj {
            Proj::Panini(d) | Proj::Planet(d) => d,
            _ => 0.0,
        };
        if !(k.t.abs() <= MAX_T)
            || i > 0 && !(k.t - keys[i - 1].t >= GAP)
            || !v.q.is_finite()
            || (v.q.length() - 1.0).abs() > 1e-3
            || !(v.fov > 0.0 && v.fov <= 360.0)
            || !(0.0..=1.0).contains(&d)
        {
            return Err(Error(format!("view key {i}")));
        }
    }
    Ok(())
}

/// Inserts `k` in time order, replacing the keys closer than [`GAP`]; returns its index.
pub fn put(keys: &mut Vec<Key>, k: Key) -> usize {
    keys.retain(|x| (x.t - k.t).abs() >= GAP);
    let i = keys.partition_point(|x| x.t < k.t);
    keys.insert(i, k);
    i
}

/// Field of view of the presets' tiny planet (the player's, too).
pub const PLANET_FOV: f32 = 280.0;

/// Camera-movement presets: keys from t = 0 (output s), to be placed with `Project::preset`. Our own content.
pub mod preset {
    use super::*;

    /// `turns` full turns about the vertical from `v` over `dur` s (negative: to the left) at constant speed.
    pub fn orbit(v: View, dur: f64, turns: f32) -> Result<Vec<Key>> {
        if !(dur > 0.0) || !turns.is_finite() || turns == 0.0 || turns.abs() > 64.0 {
            return bad("orbit");
        }
        // quarter turns: slerp takes the short way, so keys stay well under 180° apart
        let n = (turns.abs() * 4.0).ceil() as usize;
        Ok((0..=n)
            .map(|i| {
                let f = i as f32 / n as f32;
                Key {
                    t: dur * i as f64 / n as f64,
                    v: View {
                        q: (Quat::from_rotation_y(TAU * turns * f) * v.q).normalize(),
                        ..v
                    },
                    e: Ease::Linear,
                }
            })
            .collect())
    }

    /// Whip pan from `a` to `b` in `dur` s: speeds up out of `a`, fastest halfway, slows into `b`; the field of view
    /// widened by `punch` degrees around the middle, where the projection changes from a's to b's.
    pub fn whip(a: View, b: View, dur: f64, punch: f32) -> Result<Vec<Key>> {
        // Keys at 0, h, h + w, 1 (× dur), at 0, f, 1 − f, 1 of the turn. The inner keys' auto tangent is the chord
        // mean (w·f/h + h·(1 − 2f)/w) / (h + w); this f makes it 1.5× the outer chord f/h, so the speed rises to
        // there with zero slope, peaks halfway and falls mirror-wise. One middle key gets the mean chord speed: the
        // speed peaks at 1/3 and 2/3 and dips halfway.
        let (h, w) = (0.45, 0.1);
        let f = 1.0 / (2.0 + w / h * (1.5 + 0.5 * w / h));
        let m = |s: f64, proj| View {
            q: a.q.slerp(b.q, s as f32).normalize(),
            fov: a.fov + (b.fov - a.fov) * s as f32 + punch,
            proj,
        };
        let ks = [
            (0.0, a, Ease::In),
            (h, m(f, a.proj), Ease::Smooth),
            (h + w, m(1.0 - f, b.proj), Ease::Out),
            (1.0, b, Ease::Smooth),
        ];
        if !(dur > 0.0) || !ks.iter().all(|k| k.1.fov > 0.0 && k.1.fov <= 360.0) {
            return bad("whip");
        }
        Ok(ks.map(|(u, v, e)| Key { t: dur * u, v, e }).to_vec())
    }

    /// Zoom from `v` to field of view `fov` in `dur` s, easing in and out.
    pub fn zoom(v: View, fov: f32, dur: f64) -> Result<Vec<Key>> {
        if !(dur > 0.0) || !(fov > 0.0 && fov <= 360.0) {
            return bad("zoom");
        }
        Ok(vec![
            Key {
                t: 0.0,
                v,
                e: Ease::InOut,
            },
            Key {
                t: dur,
                v: View { fov, ..v },
                e: Ease::Smooth,
            },
        ])
    }

    /// Slow push-in: the field of view narrows by the factor `k` (0 < k < 1) over `dur` s, starting gently,
    /// arriving at speed.
    pub fn push(v: View, dur: f64, k: f32) -> Result<Vec<Key>> {
        if !(dur > 0.0) || !(k > 0.0 && k < 1.0) {
            return bad("push-in");
        }
        Ok(vec![
            Key {
                t: 0.0,
                v,
                e: Ease::In,
            },
            Key {
                t: dur,
                v: View {
                    fov: v.fov * k,
                    ..v
                },
                e: Ease::Smooth,
            },
        ])
    }

    /// Tiny planet (looking straight down, stereographic, [`PLANET_FOV`], facing `v`'s heading) unfolding into
    /// `v` over `dur` s; `v` folding into the planet when `fold`.
    pub fn planet(v: View, dur: f64, fold: bool) -> Result<Vec<Key>> {
        if !(dur > 0.0) {
            return bad("planet");
        }
        let p = View {
            q: Quat::from_rotation_y(v.ypr()[0].to_radians()) * Quat::from_rotation_x(FRAC_PI_2),
            fov: PLANET_FOV,
            proj: Proj::Planet(1.0),
        };
        let (a, b) = if fold { (v, p) } else { (p, v) };
        Ok(vec![
            Key {
                t: 0.0,
                v: a,
                e: Ease::InOut,
            },
            Key {
                t: dur,
                v: b,
                e: Ease::Smooth,
            },
        ])
    }
}
