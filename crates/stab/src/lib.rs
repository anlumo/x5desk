//! Stabilisation maths for the X5: trailer IMU ([`media::insv`]) -> per-frame orientation, a smoothed camera path
//! (FlowState-like), horizon / direction lock and the rolling-shutter term of `stitch::Pose`.
//!
//! Frames: body = the camera's calibration world (`stitch`: Z up when the camera stands upright, lens 0 axis −X);
//! world = Z up (against gravity), yaw arbitrary (no magnetometer); orientations are body -> world.
//!
//! Use: `Track::new(&trailer.imu()?, KP, KI)`, then `poses(&track, &trailer.frames(EXPOSURE)?, &Opts::new(..))`,
//! one [`Pose`] per video frame.
//!
//! # Measurements
//!
//! X5 firmware 1.7.43; clips v606 (31 s walking in daylight, up to 3.6 rad/s), v627 (22 s PureVideo at night) and
//! v635 (6 s, nearly still). Method (`tests/footage.rs`): both lenses stitched to 2048 × 1024 by crate stitch,
//! corners tracked between consecutive frames ([`vis`]), the inter-frame rotation fitted on the sphere (RANSAC +
//! Horn) and compared with the gyro integrated over the same interval.
//! - **Axes** ([`AXES`]): body (x, y, z) = IMU (y, −z, −x) on all three clips; correlation of the visual and gyro
//!   rotations per body axis 0.982–0.999. The free fit lies 0.5° (v627) and 0.8° (v606) from that permutation,
//!   mostly about the vertical (2.2° on v635, which barely moves); gyro scale 0.990 (v627) / 1.013 (v606). At rest
//!   the accelerometer reads +1 g along body +Z.
//! - **Time** ([`OFFSET`]): a frame's centre row is exposed at the exposure record's time − exposure / 2 + 1.0 ms
//!   on the IMU clock, i.e. the record stamps the end of the exposure. The ½·exposure shows on v627, whose exposure
//!   switches between 10 and 20 ms: gyro-vs-visual residual 0.0214°/frame at the record time, 0.0146 at
//!   mid-exposure. With it the offset is +1.14 ms (v627) and +0.84 ms (v606) (v635: +2.5, poorly defined); without
//!   it they disagree (−6.4 / −1.2 ms).
//! - **Rolling shutter** ([`RS`], [`Opts::new`]): `rolling_shutter_time` (#25) is the readout in ms, along texture
//!   rows, row 0 first, the same on both lenses. The inter-frame motion is most rigid, after moving every tracked
//!   ray to its row's time, at a readout of 21.6 ms on v606 (trailer 21.24) and 12.1 ms on v627 (PureVideo, trailer
//!   11.84). At the trailer's readout (v606) the rigid-fit residual is 0.956 px, 0.978 without the term; either
//!   lens reversed or readout along columns is worse than none (0.987–1.034); v627 ranks the same. Through the
//!   stitch shader (v606, 8 s): 0.596 px without the term, 0.569 / 0.556 / 0.564 px at 0.5 / 1 / 1.5 × the
//!   trailer's readout and 0.660 px with the sign flipped.
//! - **Gravity** ([`KP`], [`KI`]): Mahony gains chosen by the horizon tilt measured in the image (near-vertical edges
//!   are meridians when level), horizon lock, v606 / v627 mean: Kp 0.1 / Ki 0.005 0.78° / 1.12°; 0.5 / 0.05
//!   1.31° / 1.02°; 0.05 / 0.001 1.73° / 1.52°; gyro only 7.0° / 7.9° (mostly its start attitude). The image measure
//!   reads 1.23× a known added tilt on these frames. Correction is gated to |a| within 0.1 g of 1 g.
//! - **Yaw** (not corrected: no magnetometer): gyro minus visual rotation over the clip leaves a bias about body Z
//!   of 0.075°/s on v627; direction lock turns −1.9° in 22 s there (visual). On v606 (walking past near columns) the
//!   same measures give 0.63°/s and −22.3° in 31 s, but parallax biases the visual rotation there (that fit also
//!   puts 1.2°/s on the pitch axis), so the true drift is between. The trailer's `gyro_calib` (#31, seven f64,
//!   the same in all three clips) matches none of these biases; it is not used.
//! - IMU stamps come in batches of 16, 1 ms apart, at a true 1002 Hz: up to ~30 µs off, ignored.

pub mod vis;

use glam::{DMat3, DQuat, DVec3, Quat, Vec3};
use media::insv::{Exposure, Imu, Meta};
use std::f64::consts::TAU;

/// IMU axes in the body frame: `body = AXES · imu`, i.e. body (x, y, z) = imu (y, −z, −x).
pub const AXES: DMat3 = DMat3::from_cols(DVec3::NEG_Z, DVec3::X, DVec3::NEG_Y);
/// Frame time (mid-exposure of the centre row) = exposure-record time − exposure / 2 + `OFFSET`, s.
pub const OFFSET: f64 = 1.0e-3;
/// Readout direction: +1, texture row 0 is read first.
pub const RS: f64 = 1.0;
/// Mahony gains: proportional (rad/s per unit error) and integral (rad/s² per unit error).
pub const KP: f64 = 0.1;
pub const KI: f64 = 0.005;

/// Rotation vector of `q` (shortest way round).
pub fn log(q: DQuat) -> DVec3 {
    let q = if q.w < 0.0 { -q } else { q };
    let v = DVec3::new(q.x, q.y, q.z);
    let n = v.length();
    if n < 1e-9 {
        2.0 * v / q.w
    } else {
        2.0 * n.atan2(q.w) / n * v
    }
}

/// Rotation of the rotation vector `v`.
pub fn exp(v: DVec3) -> DQuat {
    DQuat::from_scaled_axis(v)
}

/// The rotation `q` minimising Σ w·|b − q·a|² over the triples (a, b, w) (Horn's quaternion method).
pub fn horn(p: impl IntoIterator<Item = (DVec3, DVec3, f64)>) -> DQuat {
    let mut s = [[0f64; 3]; 3];
    for (a, b, w) in p {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += w * a[i] * b[j];
            }
        }
    }
    let [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = s;
    let mut a = [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, yy - xx - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, zz - xx - yy],
    ];
    // Jacobi sweeps; the eigenvector of the largest eigenvalue is (w, x, y, z)
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for _ in 0..32 {
        let off: f64 = (0..4)
            .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let all: f64 = a.iter().flatten().map(|x| x * x).sum();
        if off <= 1e-30 * all {
            break;
        }
        for p in 0..3 {
            for q in p + 1..4 {
                if a[p][q] == 0.0 {
                    continue;
                }
                let th = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..4 {
                    let (x, y) = (a[k][p], a[k][q]);
                    (a[k][p], a[k][q]) = (c * x - s * y, s * x + c * y);
                    let (x, y) = (v[k][p], v[k][q]);
                    (v[k][p], v[k][q]) = (c * x - s * y, s * x + c * y);
                }
                let (x, y) = (a[p], a[q]);
                a[p] = std::array::from_fn(|k| c * x[k] - s * y[k]);
                a[q] = std::array::from_fn(|k| s * x[k] + c * y[k]);
            }
        }
    }
    let i = (0..4).max_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap();
    DQuat::from_xyzw(v[1][i], v[2][i], v[3][i], v[0][i]).normalize()
}

/// Orientation track at the IMU rate: gyro integration (midpoint rule) with a Mahony gravity correction.
pub struct Track {
    /// Sample times, s (camera clock).
    pub t: Vec<f64>,
    /// Body -> world at `t`.
    pub q: Vec<DQuat>,
}

impl Track {
    /// From the trailer's IMU samples (at least two). The correction pulls the up estimate toward the accelerometer
    /// while |a| is within 0.1 g of 1 g (walking and swinging push it out of that band). The same filter runs
    /// backward over the clip first, from the last sample's gravity, and its attitude at the first sample starts the
    /// forward pass: clips often start while the camera still swings, where averaged gravity is degrees off.
    pub fn new(imu: &[Imu], kp: f64, ki: f64) -> Track {
        let v = |a: [f32; 3]| AXES * DVec3::new(a[0] as f64, a[1] as f64, a[2] as f64);
        let step = |q: DQuat, bi: &mut DVec3, w: DVec3, a: DVec3, dt: f64| {
            let n = a.length();
            let e = if (n - 1.0).abs() < 0.1 {
                (a / n).cross(q.inverse() * DVec3::Z)
            } else {
                DVec3::ZERO
            };
            *bi += ki * dt * e;
            (q * exp((w + kp * e + *bi) * dt)).normalize()
        };
        let dt = |p: &[Imu]| (p[1].t - p[0].t) as f64 * 1e-6;
        let (mut q, mut bi) = (
            DQuat::from_rotation_arc(v(imu[imu.len() - 1].acc).normalize(), DVec3::Z),
            DVec3::ZERO,
        );
        for p in imu.windows(2).rev() {
            q = step(
                q,
                &mut bi,
                -0.5 * (v(p[0].gyro) + v(p[1].gyro)),
                v(p[0].acc),
                dt(p),
            );
        }
        // backward, the integral term learnt the negated bias
        bi = -bi;
        let (mut t, mut qs) = (Vec::with_capacity(imu.len()), Vec::with_capacity(imu.len()));
        t.push(imu[0].t as f64 * 1e-6);
        qs.push(q);
        for p in imu.windows(2) {
            q = step(
                q,
                &mut bi,
                0.5 * (v(p[0].gyro) + v(p[1].gyro)),
                v(p[1].acc),
                dt(p),
            );
            t.push(p[1].t as f64 * 1e-6);
            qs.push(q);
        }
        Track { t, q: qs }
    }

    /// Orientation at `t` (slerp between samples, clamped to the ends).
    pub fn at(&self, t: f64) -> DQuat {
        let i = self
            .t
            .partition_point(|&x| x <= t)
            .clamp(1, self.t.len() - 1);
        let (a, b) = (self.t[i - 1], self.t[i]);
        self.q[i - 1].slerp(self.q[i], ((t - a) / (b - a)).clamp(0.0, 1.0))
    }

    /// Body rotation vector from `a` to `b`.
    pub fn turn(&self, a: f64, b: f64) -> DVec3 {
        log(self.at(a).inverse() * self.at(b))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lock {
    /// Smooth all three axes.
    Free,
    /// Level (gravity down); the heading follows the smoothed path.
    Horizon,
    /// Level, and the heading stays at the start's.
    Direction,
}

#[derive(Clone, Copy, Debug)]
pub struct Opts {
    pub lock: Lock,
    /// Gaussian σ of the path smoothing, s (0: follow the camera).
    pub smooth: f64,
    /// Sensor readout time, s (0: no rolling-shutter term).
    pub readout: f64,
    /// Frame time = exposure-record time − exposure / 2 + `offset`, s.
    pub offset: f64,
}

impl Opts {
    /// The measured timing for a file with metadata `m`.
    pub fn new(m: &Meta, lock: Lock, smooth: f64) -> Self {
        Self {
            lock,
            smooth,
            readout: m.rolling_shutter * 1e-3,
            offset: OFFSET,
        }
    }
}

/// One frame's stitch inputs: `stitch::Pose { q: p.q * framing.q, rs: [p.rs; 2] }`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Output (virtual camera) -> body.
    pub q: Quat,
    /// Body rotation vector over the readout, rad.
    pub rs: Vec3,
}

/// exp(−4.5): the smoothing kernel at 3σ.
const E3: f64 = 0.011108996538242306;

/// The virtual camera's path (virtual camera -> world) for orientations `r` at times `t` (ascending): a Gaussian
/// mean on the rotation manifold over ±3σ (two Gauss-Newton steps from the frame's own orientation), then the lock.
/// Zero-phase, so turns are rounded but never late. The kernel is lowered to reach 0 at ±3σ: a cut-off step there
/// makes the path jump by ~0.3° whenever a frame of an 11 rad/s spin enters the window.
pub fn path(t: &[f64], r: &[DQuat], lock: Lock, sigma: f64) -> Vec<DQuat> {
    let mut s = r.to_vec();
    if sigma > 0.0 {
        let (mut lo, mut hi, w) = (0, 0, 3.0 * sigma);
        for k in 0..r.len() {
            while t[k] - t[lo] > w {
                lo += 1;
            }
            while hi < r.len() && t[hi] - t[k] <= w {
                hi += 1;
            }
            let mut m = r[k];
            for _ in 0..2 {
                let (mut d, mut ws, mi) = (DVec3::ZERO, 0.0, m.inverse());
                // a spin turns past π inside the window: unwrap each rotation vector (v + 2πn·v̂) next to its
                // neighbour's, walking out from k
                let mut add = |j: usize, p: &mut DVec3| {
                    let (v, x) = (log(mi * r[j]), (t[j] - t[k]) / sigma);
                    let n = v.length();
                    *p = if n < 1e-9 {
                        v
                    } else {
                        v + (TAU * ((p.dot(v) / n - n) / TAU).round() / n) * v
                    };
                    let g = (-0.5 * x * x).exp() - E3;
                    (d, ws) = (d + g * *p, ws + g);
                };
                let mut p = DVec3::ZERO;
                (k..hi).for_each(|j| add(j, &mut p));
                p = DVec3::ZERO;
                (lo..k).rev().for_each(|j| add(j, &mut p));
                m = (m * exp(d / ws)).normalize();
            }
            s[k] = m;
        }
    }
    // heading = azimuth of the body X axis (the lens axes): defined upright and upside down
    let yaw = |q: DQuat| {
        let x = q * DVec3::X;
        DQuat::from_rotation_z(x.y.atan2(x.x))
    };
    match lock {
        Lock::Free => {}
        Lock::Horizon => s.iter_mut().for_each(|q| *q = yaw(*q)),
        Lock::Direction => {
            if let Some(&q) = s.first() {
                s.fill(yaw(q));
            }
        }
    }
    s
}

/// Per-frame poses for the frames timed by `ex` (the exposure record from video frame 0 on).
pub fn poses(tr: &Track, ex: &[Exposure], o: &Opts) -> Vec<Pose> {
    let t: Vec<f64> = ex
        .iter()
        .map(|e| e.t as f64 * 1e-6 - 0.5 * e.time + o.offset)
        .collect();
    let r: Vec<DQuat> = t.iter().map(|&t| tr.at(t)).collect();
    let h = o.readout / 2.0;
    path(&t, &r, o.lock, o.smooth)
        .iter()
        .zip(&r)
        .zip(&t)
        .map(|((s, r), &t)| Pose {
            q: (r.inverse() * *s).as_quat(),
            rs: if h > 0.0 {
                (RS * tr.turn(t - h, t + h)).as_vec3()
            } else {
                Vec3::ZERO
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic uniform numbers in −1..1.
    fn rng(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn deg(q: DQuat) -> f64 {
        log(q).length().to_degrees()
    }

    #[test]
    fn log_exp() {
        let mut s = 1;
        for _ in 0..1000 {
            let v = DVec3::new(rng(&mut s), rng(&mut s), rng(&mut s)) * 3.0;
            let v = if v.length() >= std::f64::consts::PI {
                v * 0.5
            } else {
                v
            };
            assert!((log(exp(v)) - v).length() < 1e-12, "{v}");
            // the other cover gives the same rotation vector
            assert!((log(-exp(v)) - v).length() < 1e-12, "{v}");
        }
        let v = DVec3::new(1e-12, -2e-12, 3e-13);
        assert!((log(exp(v)) - v).length() < 1e-24);
        assert_eq!(log(DQuat::IDENTITY), DVec3::ZERO);
        // past π: the shortest way round
        let v = DVec3::Z * 3.5;
        assert!((log(exp(v)) - DVec3::Z * (3.5 - std::f64::consts::TAU)).length() < 1e-12);
    }

    #[test]
    fn horn_fit() {
        let mut s = 7;
        for k in 0..50 {
            let q = exp(DVec3::new(rng(&mut s), rng(&mut s), rng(&mut s)) * 3.0 / 3f64.sqrt())
                .normalize();
            let p: Vec<_> = (0..20)
                .map(|_| {
                    let a = DVec3::new(rng(&mut s), rng(&mut s), rng(&mut s)).normalize();
                    let n =
                        DVec3::new(rng(&mut s), rng(&mut s), rng(&mut s)) * 1e-4 * (k % 2) as f64;
                    (a, q * a + n, 1.0)
                })
                .collect();
            let e = deg(horn(p.iter().copied()) * q.inverse());
            assert!(e < if k % 2 == 0 { 1e-9 } else { 0.01 }, "{k}: {e}°");
        }
        // two rays are enough; 180° rotations too
        let q = DQuat::from_rotation_y(std::f64::consts::PI);
        let p = [(DVec3::X, q * DVec3::X, 1.0), (DVec3::Z, q * DVec3::Z, 1.0)];
        assert!(deg(horn(p) * q.inverse()) < 1e-9);
    }

    /// Body orientation of the synthetic motion: yaw pan with roll / pitch wobble, tilted 10°.
    fn motion(t: f64) -> DQuat {
        DQuat::from_rotation_z(0.4 * t)
            * DQuat::from_rotation_y(10f64.to_radians() + 0.1 * (2.1 * t).sin())
            * DQuat::from_rotation_x(0.15 * (3.3 * t).sin())
    }

    /// IMU samples at 1 kHz in IMU axes for the motion `m`, with a gyro bias.
    fn imu(secs: f64, bias: DVec3, m: fn(f64) -> DQuat) -> Vec<Imu> {
        let f = |v: DVec3| (AXES.transpose() * v).as_vec3().to_array();
        (0..(secs * 1e3) as u64)
            .map(|i| {
                let t = i as f64 * 1e-3;
                let w = log(m(t - 5e-5).inverse() * m(t + 5e-5)) / 1e-4;
                Imu {
                    t: 1_000_000 + i * 1000,
                    acc: f(m(t).inverse() * DVec3::Z),
                    gyro: f(w + bias),
                }
            })
            .collect()
    }

    #[test]
    fn track() {
        // gyro only: relative rotations exact to f32 input precision, tilt from the start attitude
        let s = imu(20.0, DVec3::ZERO, motion);
        let tr = Track::new(&s, 0.0, 0.0);
        let up = |q: DQuat| q.inverse() * DVec3::Z;
        for t in [0.5, 7.25, 19.9] {
            let t = 1.0 + t;
            let tilt = up(tr.at(t)).angle_between(up(motion(t - 1.0))).to_degrees();
            let rel = deg((tr.at(1.0).inverse() * tr.at(t))
                * (motion(0.0).inverse() * motion(t - 1.0)).inverse());
            assert!(
                tilt < 0.01 && rel < 0.01,
                "t {t}: tilt {tilt}°, relative {rel}°"
            );
            let w = tr.turn(t - 0.01, t + 0.01) / 0.02;
            let want = log(motion(t - 1.01).inverse() * motion(t - 0.99)) / 0.02;
            assert!((w - want).length() < 1e-4, "{w} vs {want}");
        }
        // a gyro bias without a pan to average it out: the gyro alone walks away, the correction stays put
        let still = |t: f64| DQuat::from_rotation_z(-0.4 * t) * motion(t);
        let s = imu(60.0, DVec3::new(0.002, -0.0015, 0.0), still);
        let tilt = |tr: &Track| {
            (0..600)
                .map(|i| 1.0 + 0.1 * i as f64)
                .map(|t| up(tr.at(t)).angle_between(up(still(t - 1.0))).to_degrees())
                .fold(0.0, f64::max)
        };
        let (raw, fused) = (
            tilt(&Track::new(&s, 0.0, 0.0)),
            tilt(&Track::new(&s, KP, KI)),
        );
        eprintln!(
            "worst tilt over 60 s with a 0.14°/s bias: gyro only {raw:.2}°, Mahony {fused:.3}°"
        );
        assert!(raw > 3.0 && fused < raw / 5.0);
    }

    #[test]
    fn smoothing() {
        let mut s = 3;
        let t: Vec<f64> = (0..600).map(|k| k as f64 / 30.0).collect();
        // a slow pan, tilted, plus white jitter of ±1.1° per axis
        let slow = |t: f64| DQuat::from_rotation_z(0.4 * t) * DQuat::from_rotation_y(0.17);
        let r: Vec<DQuat> = t
            .iter()
            .map(|&t| slow(t) * exp(DVec3::new(rng(&mut s), rng(&mut s), rng(&mut s)) * 0.02))
            .collect();
        // inter-frame rotation left after removing the pan, and the distance from the pan
        let rough = |p: &[DQuat]| {
            (30..569)
                .map(|k| {
                    deg((slow(t[k]).inverse() * p[k]).inverse()
                        * slow(t[k + 1]).inverse()
                        * p[k + 1])
                })
                .sum::<f64>()
                / 539.0
        };
        let p = path(&t, &r, Lock::Free, 0.3);
        let (a, b) = (rough(&r), rough(&p));
        let off = (30..570)
            .map(|k| deg(p[k] * slow(t[k]).inverse()))
            .fold(0.0, f64::max);
        eprintln!(
            "free σ 0.3 s: jitter {a:.3}°/frame -> {b:.4}°/frame, max distance from the pan {off:.3}°"
        );
        assert!(b < a / 20.0 && off < 0.5);
        assert_eq!(path(&t, &r, Lock::Free, 0.0), r);

        let p = path(&t, &r, Lock::Horizon, 0.3);
        for (k, q) in p.iter().enumerate().skip(30).take(540) {
            let x = slow(t[k]) * DVec3::X;
            let yaw = (q * DVec3::X)
                .angle_between(DVec3::new(x.x, x.y, 0.0))
                .to_degrees();
            assert!(
                (q * DVec3::Z - DVec3::Z).length() < 1e-12 && yaw < 0.5,
                "{k}: heading off by {yaw}°"
            );
        }
        let p = path(&t, &r, Lock::Direction, 0.3);
        assert!(p.iter().all(|q| *q == p[0]) && (p[0] * DVec3::Z - DVec3::Z).length() < 1e-12);

        // a spin speeding up from 3 to 11 rad/s turns more than π inside the ±3σ window; about one axis the manifold
        // mean is the Gaussian mean of the unwrapped angle
        let a = |t: f64| 3.0 * t + 0.2 * t * t;
        let r: Vec<DQuat> = t.iter().map(|&t| DQuat::from_rotation_z(a(t))).collect();
        let p = path(&t, &r, Lock::Free, 0.5);
        let off = (45..555)
            .map(|k| {
                let (mut s, mut ws) = (0.0, 0.0);
                for j in k - 45..=k + 45 {
                    let g = ((-0.5 * ((t[j] - t[k]) / 0.5).powi(2)).exp() - E3).max(0.0);
                    (s, ws) = (s + g * a(t[j]), ws + g);
                }
                deg(p[k] * DQuat::from_rotation_z(s / ws).inverse())
            })
            .fold(0.0, f64::max);
        assert!(off < 1e-6, "spin, σ 0.5 s: {off}° off");
    }

    /// The poses undo the motion: output directions stay put in the world, up to the lock's remaining rotation.
    #[test]
    fn poses_world() {
        let s = imu(12.0, DVec3::ZERO, motion);
        let tr = Track::new(&s, KP, KI);
        let ex: Vec<Exposure> = (0..300)
            .map(|k| Exposure {
                t: 2_000_000 + k * 33_367,
                time: 0.01,
            })
            .collect();
        let tk = |k: usize| ex[k].t as f64 * 1e-6 - 0.005;
        for lock in [Lock::Free, Lock::Horizon, Lock::Direction] {
            let o = Opts {
                lock,
                smooth: 0.5,
                readout: 0.02,
                offset: 0.0,
            };
            let p = poses(&tr, &ex, &o);
            // world direction of the output's centre and up
            let world = |k: usize, v: DVec3| tr.at(tk(k)) * (p[k].q.as_dquat() * v);
            let (mut turn, mut tilt) = (0f64, 0f64);
            for k in 1..p.len() {
                turn = turn.max(
                    world(k, DVec3::NEG_X)
                        .angle_between(world(k - 1, DVec3::NEG_X))
                        .to_degrees(),
                );
                tilt = tilt.max(world(k, DVec3::Z).angle_between(DVec3::Z).to_degrees());
            }
            eprintln!("{lock:?}: max centre turn {turn:.3}°/frame, max tilt {tilt:.2}°");
            match lock {
                Lock::Free => assert!(turn < 1.0),
                Lock::Horizon => assert!(tilt < 0.05 && turn < 1.0),
                Lock::Direction => assert!(tilt < 0.05 && turn < 0.01),
            }
            let w = log(motion(tk(100) - 1.01).inverse() * motion(tk(100) - 0.99));
            let e = (p[100].rs.as_dvec3() - RS * w).length();
            assert!(e < 1e-5, "rolling shutter term off by {e} rad");
        }
    }
}
