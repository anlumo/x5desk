//! Direction of travel for the forward view (the vendor's AI Forward). Between two frames, the image motion the gyro's
//! rotation leaves is translational flow: a static point's ray at the first frame `a`, at the second `b` and the
//! translation `t` are coplanar, `t · (a × b) = 0`, so every tracked feature is a great-circle constraint on `t`, and
//! the features move away from `t` (the focus of expansion) and toward `−t`. [`fit`] finds `t` robustly from the
//! features of one pair of frames; [`yaws`] turns the estimates into a view heading per frame.
//!
//! Features that must not vote: those with no parallax (sky, far scenery: they constrain nothing), those that keep
//! their place relative to the camera body (the rider, the vehicle, a stick, the camera's mount), and those on moving
//! objects (outliers to the fit).

use glam::{DMat3, DQuat, DVec3};
use std::f64::consts::{PI, TAU};

/// Least parallax that votes, rad (0.09°; a quarter pixel of the 1024 px wide panorama `export::flow` tracks on).
pub const FMIN: f64 = TAU / 4096.0;
/// Least motion relative to the camera body, rad (as `FMIN`): what moves with the camera (rider, vehicle, mount).
pub const BMIN: f64 = TAU / 4096.0;
/// Inlier distance from the great circle through the feature and the focus of expansion, rad (0.18°).
pub const THR: f64 = TAU / 2048.0;
/// Fewest voting features and inliers for an estimate.
const MIN: usize = 30;

/// The features tracked between two analysed frames.
#[derive(Clone, Debug, Default)]
pub struct Pair {
    /// Frame times, s (as [`crate::poses`] times them).
    pub t: [f64; 2],
    /// Camera orientation (body -> world) at the two frames.
    pub r: [DQuat; 2],
    /// World rays of each feature at the two frames.
    pub m: Vec<[DVec3; 2]>,
}

/// The direction of travel between two frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Est {
    /// Mid time, s.
    pub t: f64,
    /// Unit, world.
    pub dir: DVec3,
    /// Features that voted, and those consistent with `dir`.
    pub n: u32,
    pub inl: u32,
    /// Speed over distance, the inliers' median, 1/s: how fast the scene's parallax runs.
    pub rate: f64,
    /// Speed over the camera's height above the ground, 1/s: the inliers more than 10° below the horizon taken as
    /// ground (0 without 10 of them).
    pub ground: f64,
}

/// Unit eigenvector of the smallest eigenvalue of the symmetric positive semi-definite `m`, by inverse iteration
/// from `x` (a close guess converges in a few steps).
fn least(m: DMat3, x: DVec3) -> DVec3 {
    let s = (m.x_axis.x + m.y_axis.y + m.z_axis.z).max(1e-300);
    let i = (m + DMat3::IDENTITY * (1e-12 * s)).inverse();
    let mut x = x.normalize();
    for _ in 0..20 {
        x = (i * x).normalize();
    }
    x
}

/// The direction of travel from pair `p`; `dir` zero without enough voting features or a consensus among them.
pub fn fit(p: &Pair) -> Est {
    let mut e = Est {
        t: 0.5 * (p.t[0] + p.t[1]),
        dir: DVec3::ZERO,
        n: 0,
        inl: 0,
        rate: 0.0,
        ground: 0.0,
    };
    let dt = p.t[1] - p.t[0];
    let rw = p.r[1] * p.r[0].inverse();
    // (a, b, a × b) of the features that vote
    let v: Vec<[DVec3; 3]> = (p.m.iter())
        .map(|&[a, b]| [a, b, a.cross(b)])
        .filter(|[a, b, c]| c.length() > FMIN && (*b - rw * *a).length() > BMIN)
        .collect();
    e.n = v.len() as u32;
    if v.len() < MIN || dt <= 0.0 {
        return e;
    }
    // distance of b from the great circle through a and t; infinite where it moved toward t
    let res = |t: DVec3, [a, b, _]: &[DVec3; 3]| {
        let n = a.cross(t);
        let l = n.length();
        match l > 1e-9 && a.dot(t) - b.dot(t) > -THR {
            true => (n.dot(*b) / l).abs(),
            false => f64::INFINITY,
        }
    };
    let score = |t: DVec3| v.iter().map(|x| res(t, x).min(THR)).sum::<f64>();
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as usize % v.len()
    };
    let mut best = (f64::MAX, DVec3::ZERO);
    for _ in 0..200 {
        let (i, j) = (rnd(), rnd());
        let t = v[i][2].cross(v[j][2]);
        if t.length() < 1e-12 {
            continue;
        }
        // the sign the two samples expand from
        let t = t.normalize() * (v[i][0] - v[i][1] + v[j][0] - v[j][1]).dot(t).signum();
        let x = score(t);
        if x < best.0 {
            best = (x, t);
        }
    }
    let mut t = best.1;
    if t == DVec3::ZERO {
        return e;
    }
    // least squares of t · (a × b) over the inliers: the residual's noise is about the same for every feature
    for _ in 0..3 {
        let m = (v.iter())
            .filter(|x| res(t, x) < THR)
            .fold(DMat3::ZERO, |m, [_, _, c]| {
                m + DMat3::from_cols(*c * c.x, *c * c.y, *c * c.z)
            });
        let x = least(m, t);
        t = if x.dot(t) < 0.0 { -x } else { x };
    }
    let inl: Vec<&[DVec3; 3]> = v.iter().filter(|x| res(t, x) < THR).collect();
    e.inl = inl.len() as u32;
    if inl.len() < MIN {
        return e;
    }
    let med = |mut x: Vec<f64>| {
        x.sort_by(f64::total_cmp);
        x.get(x.len() / 2).copied().unwrap_or(0.0)
    };
    // speed / distance = parallax / sin(angle from the focus of expansion)
    let rho = |[a, _, c]: &[DVec3; 3]| {
        let s = a.cross(t).length();
        (s > 0.25).then(|| c.length() / s / dt)
    };
    e.rate = med(inl.iter().filter_map(|x| rho(x)).collect());
    // on the ground at height h below: distance h / cos(angle from the nadir), so speed / h = rho / −a.z
    let g: Vec<f64> = (inl.iter())
        .filter(|x| x[0].z < -(10f64.to_radians().sin()))
        .filter_map(|x| Some(rho(x)? / -x[0].z))
        .collect();
    e.ground = if g.len() >= 10 { med(g) } else { 0.0 };
    e.dir = t;
    e
}

/// Speed over the camera's height (`Est::ground`, 1/s) below which the travel counts as standing: below walking pace
/// (v606, walking with the camera held up: 0.4–0.6; standing with it in hand, v606 and v635: 0.03–0.2).
pub const PACE: f64 = 0.3;
/// Least smoothing σ of the travel, s. On eight rides (scooter-stem mount, where the camera's heading is the
/// travel's up to steering) σ 0.25 / 0.5 / 0.75 / 1 s left 3.2 / 2.0 / 1.6 / 1.4 °/s of heading jitter on straight
/// stretches (the horizon lock at σ 0.5: 2.2) and rounded turns by 1.6 / 3.1 / 6.6 / 12.0° (p95 against the raw
/// estimates); the player's default σ, 0.5 s, sits at the knee.
pub const SMIN: f64 = 0.25;
/// Share of a σ window's pairs that must be travelling for the direction of travel to be shown there.
const COVER: f64 = 0.3;
/// The travel's elevation stays within this, as sine: a steeper estimate is noise.
const STEEP: f64 = 0.5;
/// Angle (rad) beyond which an estimate's vote in the window's mean drops to 0 (Tukey's biweight).
const OUT: f64 = 0.35;
/// Standing: σ (s) of the travel's usual offset from the camera's heading at either side, and how long (s) the
/// view takes from the offset at the moment of stopping or starting to it.
const HOLD: f64 = 2.0;
const EDGE: f64 = 1.0;

/// Whether `e` says the camera travels.
pub fn moving(e: &Est) -> bool {
    e.dir != DVec3::ZERO && e.ground >= PACE && e.dir.z.abs() < STEEP
}

/// `a` unwrapped to within π of `b`.
fn near(a: f64, b: f64) -> f64 {
    a + TAU * ((b - a) / TAU).round()
}

/// The view's heading (rad about world Z) at frame times `t`: the direction of travel of the pairs `es` (ascending,
/// one per pair, `dir` zero where it has none) smoothed over a Gaussian window of σ `sigma` (at least `SMIN`) on the
/// circle, robustly. Where too few of a window's pairs travel (standing, below walking pace), the camera's heading
/// `cam` (per frame, e.g. the horizon lock's) turned by the offset the travel usually had from it before and after
/// (`HOLD`), linearly in between, held at the ends, faded in from the offset at the gap's edges over `EDGE`; without
/// travel, `cam` itself.
pub fn yaws(t: &[f64], cam: &[f64], es: &[Est], sigma: f64) -> Vec<f64> {
    let s = sigma.max(SMIN);
    let (mut lo, mut hi, mut mv) = (0, 0, Vec::<(f64, f64)>::new());
    let off: Vec<Option<f64>> = (t.iter().zip(cam))
        .map(|(&tk, &ck)| {
            while lo < es.len() && es[lo].t < tk - 3.0 * s {
                lo += 1;
            }
            while hi < es.len() && es[hi].t <= tk + 3.0 * s {
                hi += 1;
            }
            let w = &es[lo..hi];
            let g = |e: &Est| {
                let x = (e.t - tk) / s;
                (-0.5 * x * x).exp() - crate::E3
            };
            let all: f64 = w.iter().map(g).sum();
            mv.clear();
            mv.extend(
                (w.iter())
                    .filter(|e| moving(e))
                    .map(|e| (g(e), e.dir.y.atan2(e.dir.x))),
            );
            if mv.iter().map(|x| x.0).sum::<f64>() < COVER * all || all <= 0.0 {
                return None;
            }
            // the circular mean, then twice reweighted by each vote's distance from it
            let mut m = None::<f64>;
            for _ in 0..3 {
                let (mut x, mut y) = (0.0, 0.0);
                for &(g, a) in &mv {
                    let b = m.map_or(1.0, |m| {
                        let d = (near(a, m) - m) / OUT;
                        (1.0 - d * d).max(0.0).powi(2)
                    });
                    (x, y) = (x + g * b * a.cos(), y + g * b * a.sin());
                }
                if x == 0.0 && y == 0.0 {
                    break;
                }
                m = Some(y.atan2(x));
            }
            m.map(|m| near(m - ck, 0.0))
        })
        .collect();
    // the gaps: from the offset at either edge to the travel's usual offset from the camera there (a head turned as
    // the travel stops isn't kept), linearly between the two sides' usual offsets
    let usual = |i: usize| {
        let (a, b) = (
            t.partition_point(|&x| x < t[i] - 3.0 * HOLD),
            t.partition_point(|&x| x <= t[i] + 3.0 * HOLD),
        );
        let (x, y) =
            (a..b)
                .filter_map(|j| Some((j, off[j]?)))
                .fold((0.0, 0.0), |(x, y), (j, d)| {
                    let g = (-0.5 * ((t[j] - t[i]) / HOLD).powi(2)).exp();
                    (x + g * d.cos(), y + g * d.sin())
                });
        y.atan2(x)
    };
    let fade = |x: f64| 0.5 + 0.5 * (PI * x.clamp(0.0, 1.0)).cos();
    let mut next = vec![None::<usize>; t.len() + 1];
    for k in (0..t.len()).rev() {
        next[k] = off[k].map(|_| k).or(next[k + 1]);
    }
    let (mut out, mut prev) = (Vec::with_capacity(t.len()), None::<usize>);
    // a gap's sides: the frame travelling last before and first after, with its usual offset
    type Side = Option<(usize, f64)>;
    let mut gap = None::<(Side, Side)>;
    for k in 0..t.len() {
        let d = match off[k] {
            Some(d) => {
                (prev, gap) = (Some(k), None);
                d
            }
            None => {
                let (a, b) = *gap.get_or_insert_with(|| {
                    (prev.map(|i| (i, usual(i))), next[k].map(|j| (j, usual(j))))
                });
                let side = |(e, u): (usize, f64), dt: f64| {
                    (near(off[e].unwrap_or(u), u) - u) * fade(dt / EDGE)
                };
                match (a, b) {
                    (None, None) => 0.0,
                    (Some(a), None) => a.1 + side(a, t[k] - t[a.0]),
                    (None, Some(b)) => b.1 + side(b, t[b.0] - t[k]),
                    (Some(a), Some((j, u))) => {
                        let b = (j, near(u, a.1));
                        a.1 + (b.1 - a.1) * (t[k] - t[a.0]) / (t[j] - t[a.0])
                            + side(a, t[k] - t[a.0])
                            + side(b, t[j] - t[k])
                    }
                }
            }
        };
        out.push(cam[k] + d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// A street: walls 4 m to either side, the ground 1.2 m below, far points at 500 m; the camera moves 0.3 m along
    /// `t` (body yawed and tilted, turning 2° between the frames); features on the rider (fixed to the body), on a
    /// car passing the other way, with 0.15 px of tracking noise on a 1024 px wide panorama.
    fn street(t: DVec3, s: &mut u64) -> Pair {
        let r0 = DQuat::from_rotation_z(0.7) * DQuat::from_rotation_x(1.1);
        let r1 = DQuat::from_rotation_z(0.7 + 2f64.to_radians()) * DQuat::from_rotation_x(1.12);
        let (c0, c1) = (DVec3::ZERO, t * 0.3);
        let px = TAU / 1024.0;
        let noise = |v: DVec3, s: &mut u64| {
            (v + DVec3::new(rng(s), rng(s), rng(s)) * 0.15 * px).normalize()
        };
        let mut m = vec![];
        let f = DVec3::new(t.x, t.y, 0.0).normalize();
        let l = DVec3::Z.cross(f);
        for i in 0..1500 {
            let (u, w) = (rng(s), rng(s));
            let p = match i % 4 {
                0 => f * u * 30.0 + l * 4.0 + DVec3::Z * (w * 3.0 + 1.8),
                1 => f * u * 30.0 - l * 4.0 + DVec3::Z * (w * 3.0 + 1.8),
                2 => f * u * 20.0 + l * w * 4.0 - DVec3::Z * 1.2,
                _ => DVec3::new(u, w, rng(s) * 0.3).normalize() * 500.0,
            };
            m.push([
                noise((p - c0).normalize(), s),
                noise((p - c1).normalize(), s),
            ]);
        }
        // the rider: fixed in the body
        for _ in 0..300 {
            let b = DVec3::new(rng(s), rng(s), -1.0).normalize();
            m.push([noise(r0 * b, s), noise(r1 * b, s)]);
        }
        // a car coming the other way at 0.6 m per frame
        for _ in 0..200 {
            let p = f * (8.0 + rng(s)) - l * 2.0 + DVec3::new(0.0, 0.0, rng(s) * 0.5);
            let q = p - f * 0.6;
            m.push([
                noise((p - c0).normalize(), s),
                noise((q - c1).normalize(), s),
            ]);
        }
        Pair {
            t: [1.0, 1.0 + 1.0 / 30.0],
            r: [r0, r1],
            m,
        }
    }

    #[test]
    fn fit_street() {
        let mut s = 5;
        for (i, t) in [
            DVec3::X,
            DVec3::new(-0.3, 0.9, 0.05),
            DVec3::new(0.0, -1.0, -0.08),
        ]
        .into_iter()
        .enumerate()
        {
            let t = t.normalize();
            let p = street(t, &mut s);
            let e = fit(&p);
            let err = e.dir.angle_between(t).to_degrees();
            eprintln!(
                "{i}: {} voted, {} inliers, error {err:.3}°, rate {:.2}/s, ground {:.2}/s (true {:.2})",
                e.n,
                e.inl,
                e.rate,
                e.ground,
                0.3 * 30.0 / 1.2
            );
            assert!(err < 0.5, "{i}: {err}°");
            assert!((e.ground / (0.3 * 30.0 / 1.2) - 1.0).abs() < 0.1);
        }
        // standing still: nothing but noise and the rider
        let mut p = street(DVec3::X, &mut s);
        for x in &mut p.m {
            x[1] = x[0];
        }
        assert_eq!(fit(&p).dir, DVec3::ZERO);
    }

    /// 60 s at 15 pairs a second: travel turning from azimuth 0 at 3°/s for 20 s (60°), a stop from 20 to 30 s, then
    /// on at 60° with the camera's heading 40° off the travel, swaying ±9°, a tenth of the pairs pointing anywhere, and
    /// 2° of noise. The view follows the travel while moving, the camera turned by the offset while standing.
    #[test]
    fn view() {
        let mut s = 9;
        let travel = |t: f64| (3.0 * t.min(20.0)).to_radians();
        let cam = |t: f64| travel(t) - 40f64.to_radians() + 0.15 * (1.3 * t).sin();
        let es: Vec<Est> = (0..900)
            .map(|i| {
                let t = i as f64 / 15.0;
                let a = match i % 10 {
                    0 => rng(&mut s) * PI,
                    _ => travel(t) + rng(&mut s) * 2f64.to_radians() * 3f64.sqrt(),
                };
                let stand = (20.0..30.0).contains(&t);
                Est {
                    t,
                    dir: DVec3::new(a.cos(), a.sin(), 0.05),
                    n: 500,
                    inl: 400,
                    rate: 1.0,
                    ground: if stand { 0.1 } else { 3.0 },
                }
            })
            .collect();
        let t: Vec<f64> = (0..1800).map(|k| k as f64 / 30.0).collect();
        let c: Vec<f64> = t.iter().map(|&t| cam(t)).collect();
        let y = yaws(&t, &c, &es, 0.5);
        let err = |k: usize, want: f64| (near(y[k], want) - want).abs().to_degrees();
        // moving, away from the stop's edges (the smoothing rounds the turn's corner at 20 s)
        let m = (0..1800)
            .filter(|&k| !(17.0..33.0).contains(&t[k]) && t[k] > 2.0 && t[k] < 58.0)
            .map(|k| err(k, travel(t[k])))
            .fold(0.0, f64::max);
        // standing: the camera's heading 40° off, as it was when the travel stopped and started
        let st = (0..1800)
            .filter(|&k| (21.0..29.0).contains(&t[k]))
            .map(|k| err(k, c[k] + 40f64.to_radians()))
            .fold(0.0, f64::max);
        eprintln!(
            "moving: worst {m:.2}° off the travel; standing: worst {st:.2}° off the camera + 40°"
        );
        assert!(m < 1.0 && st < 3.0);
        // no travel at all: the camera's heading
        let still: Vec<Est> = es.iter().map(|e| Est { ground: 0.0, ..*e }).collect();
        assert_eq!(yaws(&t, &c, &still, 0.5), c);
    }
}
