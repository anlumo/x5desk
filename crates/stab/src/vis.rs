//! Visual rotation between two equirect frames in `stitch`'s layout (row 0 = zenith; pixel (x, y) at azimuth 2πx/W,
//! polar angle πy/(H − 1); direction `(sin φ cos θ, −sin φ sin θ, cos φ)`): corners tracked with pyramidal
//! Lucas–Kanade on luma (forward-backward checked), turned into rays, and a rotation fitted with RANSAC + Horn.
//! The measuring tool behind the crate's constants (`tests/footage.rs`).

use crate::horn;
use glam::{DQuat, DVec3};
use std::f64::consts::{PI, TAU};

/// Pyramid levels: 2048 px wide frames move up to ~40 px per frame at 3.6 rad/s, 2.5 px on the top level.
const LV: usize = 5;
/// LK window radius.
const R: i32 = 7;
const N: usize = ((2 * R + 1) * (2 * R + 1)) as usize;
/// Corner search cell, px.
const CELL: usize = 16;
/// Minimum corner strength: smaller eigenvalue of the 5 × 5 structure tensor of 0..1 luma.
const CORNER: f32 = 2e-3;

/// Luma pyramid of an equirect frame; level l is (w >> l) × (h >> l) (w, h divisible by 16).
pub struct Pano {
    pub w: usize,
    pub h: usize,
    lv: Vec<Vec<f32>>,
}

impl Pano {
    /// From RGBA8 (`stitch::Stitcher::out` read back), BT.709 luma.
    pub fn rgba(px: &[u8], w: usize, h: usize) -> Pano {
        Self::luma(
            px.as_chunks::<4>()
                .0
                .iter()
                .map(|p| {
                    (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0
                })
                .collect(),
            w,
            h,
        )
    }

    pub fn luma(l0: Vec<f32>, w: usize, h: usize) -> Pano {
        assert!(
            w.is_multiple_of(1 << (LV - 1)) && h.is_multiple_of(1 << (LV - 1)) && l0.len() == w * h
        );
        let mut lv = vec![l0];
        // binomial [1 3 3 1]/8 around 2x + ½: level-l pixel x sits at level-0 position 2^l·(x + ½) − ½
        for l in 1..LV {
            let (pw, ph, p) = (w >> (l - 1), h >> (l - 1), &lv[l - 1]);
            let (nw, nh) = (pw / 2, ph / 2);
            let mut t = vec![0f32; nw * ph];
            for y in 0..ph {
                let r = &p[y * pw..][..pw];
                for x in 0..nw {
                    let s = |i: isize| r[(2 * x as isize + i).rem_euclid(pw as isize) as usize];
                    t[y * nw + x] = (s(-1) + 3.0 * (s(0) + s(1)) + s(2)) / 8.0;
                }
            }
            let mut o = vec![0f32; nw * nh];
            for y in 0..nh {
                let s = |i: isize| {
                    &t[(2 * y as isize + i).clamp(0, ph as isize - 1) as usize * nw..][..nw]
                };
                let (a, b, c, d) = (s(-1), s(0), s(1), s(2));
                for x in 0..nw {
                    o[y * nw + x] = (a[x] + 3.0 * (b[x] + c[x]) + d[x]) / 8.0;
                }
            }
            lv.push(o);
        }
        Pano { w, h, lv }
    }
}

/// Output-world direction of pixel position `p` in a `w` × `h` equirect (`stitch.wgsl` `main`).
pub fn ray(w: usize, h: usize, p: [f64; 2]) -> DVec3 {
    let (th, ph) = (TAU * p[0] / w as f64, PI * p[1] / (h - 1) as f64);
    DVec3::new(ph.sin() * th.cos(), -ph.sin() * th.sin(), ph.cos())
}

/// Bilinear, wrapped in x, clamped in y.
fn at(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let xa = (x0 as isize).rem_euclid(w as isize) as usize;
    let xb = if xa + 1 == w { 0 } else { xa + 1 };
    let ya = (y0 as isize).clamp(0, h as isize - 1) as usize * w;
    let yb = (y0 as isize + 1).clamp(0, h as isize - 1) as usize * w;
    let t = p[ya + xa] + fx * (p[ya + xb] - p[ya + xa]);
    let b = p[yb + xa] + fx * (p[yb + xb] - p[yb + xa]);
    t + fy * (b - t)
}

/// Level-0 displacement of point `p` from `a` to `b`, coarse to fine from the guess `g`.
fn lk(a: &Pano, b: &Pano, p: [f32; 2], g: [f32; 2]) -> Option<[f32; 2]> {
    let (mut t, mut ix, mut iy) = ([0f32; N], [0f32; N], [0f32; N]);
    let top = (1 << (LV - 1)) as f32;
    let mut d = [g[0] / top, g[1] / top];
    for l in (0..LV).rev() {
        let s = (1 << l) as f32;
        let (w, h, pa, pb) = (a.w >> l, a.h >> l, &a.lv[l], &b.lv[l]);
        let c = [(p[0] + 0.5) / s - 0.5, (p[1] + 0.5) / s - 0.5];
        let (mut gxx, mut gxy, mut gyy, mut k) = (0f32, 0f32, 0f32, 0);
        for j in -R..=R {
            for i in -R..=R {
                let (x, y) = (c[0] + i as f32, c[1] + j as f32);
                t[k] = at(pa, w, h, x, y);
                ix[k] = 0.5 * (at(pa, w, h, x + 1.0, y) - at(pa, w, h, x - 1.0, y));
                iy[k] = 0.5 * (at(pa, w, h, x, y + 1.0) - at(pa, w, h, x, y - 1.0));
                (gxx, gxy, gyy) = (
                    gxx + ix[k] * ix[k],
                    gxy + ix[k] * iy[k],
                    gyy + iy[k] * iy[k],
                );
                k += 1;
            }
        }
        let det = gxx * gyy - gxy * gxy;
        if det < 1e-12 {
            return None;
        }
        for _ in 0..20 {
            let (mut ex, mut ey, mut k) = (0f32, 0f32, 0);
            for j in -R..=R {
                for i in -R..=R {
                    let e = at(pb, w, h, c[0] + d[0] + i as f32, c[1] + d[1] + j as f32) - t[k];
                    (ex, ey) = (ex + e * ix[k], ey + e * iy[k]);
                    k += 1;
                }
            }
            let (dx, dy) = ((gyy * ex - gxy * ey) / det, (gxx * ey - gxy * ex) / det);
            d = [d[0] - dx, d[1] - dy];
            if dx * dx + dy * dy < 1e-4 {
                break;
            }
        }
        if !(d[0].is_finite() && d[1].is_finite()) {
            return None;
        }
        if l > 0 {
            d = [2.0 * d[0], 2.0 * d[1]];
        }
    }
    Some(d)
}

/// The strongest corner of each 16 × 16 cell between rows `y0` and `y1`.
fn corners(a: &Pano, y0: usize, y1: usize, out: &mut Vec<[f32; 2]>) {
    let (w, p) = (a.w, &a.lv[0]);
    let g = |x: usize, y: usize| {
        let r = &p[y * w..];
        (
            0.5 * (r[(x + 1) % w] - r[(x + w - 1) % w]),
            0.5 * (p[(y + 1) * w + x] - p[(y - 1) * w + x]),
        )
    };
    for cy in (y0.max(3)..y1.min(a.h - 3).saturating_sub(CELL)).step_by(CELL) {
        for cx in (0..w).step_by(CELL) {
            let mut best = (CORNER, [0f32; 2]);
            for y in (cy..cy + CELL).step_by(2) {
                for x in (cx..cx + CELL).step_by(2) {
                    let (mut sxx, mut sxy, mut syy) = (0f32, 0f32, 0f32);
                    for j in 0..5 {
                        for i in 0..5 {
                            let (gx, gy) = g((x + w + i - 2) % w, y + j - 2);
                            (sxx, sxy, syy) = (sxx + gx * gx, sxy + gx * gy, syy + gy * gy);
                        }
                    }
                    let m =
                        0.5 * (sxx + syy - ((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy).sqrt());
                    if m > best.0 {
                        best = (m, [x as f32, y as f32]);
                    }
                }
            }
            if best.0 > CORNER {
                out.push(best.1);
            }
        }
    }
}

/// Corners of `a` between polar angles `band` (rad) tracked into `b`: (position in a, position in b), kept when
/// tracking back lands within 0.5 px.
pub fn track(a: &Pano, b: &Pano, band: [f64; 2], out: &mut Vec<[[f32; 2]; 2]>) {
    let row = |ph: f64| (ph / PI * (a.h - 1) as f64) as usize;
    let mut c = vec![];
    corners(a, row(band[0]), row(band[1]), &mut c);
    let n = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(16);
    out.clear();
    std::thread::scope(|s| {
        let hs: Vec<_> = c
            .chunks(c.len().div_ceil(n).max(1))
            .map(|ch| {
                s.spawn(move || {
                    ch.iter()
                        .filter_map(|&p| {
                            let d = lk(a, b, p, [0.0; 2])?;
                            let q = [p[0] + d[0], p[1] + d[1]];
                            let e = lk(b, a, q, [-d[0], -d[1]])?;
                            ((d[0] + e[0]).powi(2) + (d[1] + e[1]).powi(2) < 0.25).then_some([p, q])
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in hs {
            out.extend(h.join().unwrap());
        }
    });
}

pub struct Fit {
    /// `b ≈ q·a` for the inliers.
    pub q: DQuat,
    pub inl: Vec<bool>,
    pub n: usize,
    /// RMS chord (≈ angle, rad) of the inliers.
    pub rms: f64,
}

/// Rotation taking the rays `a` to `b`: RANSAC over pairs (256 draws), then Horn on the inliers (chord < `thr`).
pub fn fit(p: &[(DVec3, DVec3)], thr: f64) -> Option<Fit> {
    if p.len() < 10 {
        return None;
    }
    let mut s = 0x853c49e6748fea9bu64;
    let mut rnd = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as usize % p.len()
    };
    let ok = |q: DQuat, (a, b): &(DVec3, DVec3)| (q * *a - *b).length() < thr;
    let mut best = (0, DQuat::IDENTITY);
    for _ in 0..256 {
        let (i, j) = (rnd(), rnd());
        if p[i].0.dot(p[j].0) > 0.995 {
            continue;
        }
        let q = horn([(p[i].0, p[i].1, 1.0), (p[j].0, p[j].1, 1.0)]);
        let n = p.iter().filter(|x| ok(q, x)).count();
        if n > best.0 {
            best = (n, q);
        }
    }
    let mut q = best.1;
    for _ in 0..3 {
        q = horn(p.iter().filter(|x| ok(q, x)).map(|&(a, b)| (a, b, 1.0)));
    }
    let inl: Vec<bool> = p.iter().map(|x| ok(q, x)).collect();
    let n = inl.iter().filter(|&&x| x).count();
    let e: f64 = p
        .iter()
        .zip(&inl)
        .filter(|x| *x.1)
        .map(|((a, b), _)| (q * *a - *b).length_squared())
        .sum();
    (n >= 8).then(|| Fit {
        q,
        inl,
        n,
        rms: (e / n as f64).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixel position of direction `d` (inverse of `ray`).
    fn pixel(w: usize, h: usize, d: DVec3) -> [f64; 2] {
        let th = (-d.y).atan2(d.x).rem_euclid(TAU);
        let ph = (d.z / d.length()).clamp(-1.0, 1.0).acos();
        [th * w as f64 / TAU, ph * (h - 1) as f64 / PI]
    }

    /// Three octaves of value noise on the sphere.
    fn noise(d: DVec3) -> f32 {
        let h = |x: i64, y: i64, z: i64| {
            let v = (x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791))
                as u64;
            (v.wrapping_mul(0x9E3779B97F4A7C15) >> 40) as f64 / (1u64 << 24) as f64
        };
        let mut s = 0.0;
        for (f, a) in [(16.0, 0.5), (32.0, 0.3), (64.0, 0.2)] {
            let p = d * f;
            let i = p.floor();
            let u = p - i;
            let u = u * u * (3.0 - 2.0 * u);
            let (x, y, z) = (i.x as i64, i.y as i64, i.z as i64);
            let l = |a: f64, b: f64, t: f64| a + t * (b - a);
            let e = |dy: i64, dz: i64| l(h(x, y + dy, z + dz), h(x + 1, y + dy, z + dz), u.x);
            s += a * l(l(e(0, 0), e(1, 0), u.y), l(e(0, 1), e(1, 1), u.y), u.z);
        }
        s as f32
    }

    fn render(w: usize, h: usize, k: DQuat) -> Pano {
        let ki = k.inverse();
        let l = (0..w * h)
            .map(|i| noise(ki * ray(w, h, [(i % w) as f64, (i / w) as f64])))
            .collect();
        Pano::luma(l, w, h)
    }

    #[test]
    fn pixel_ray() {
        for p in [[0.0, 10.0], [100.5, 255.0], [511.9, 400.25]] {
            let q = pixel(512, 512, ray(512, 512, p));
            assert!(
                (q[0] - p[0]).abs() < 1e-9 && (q[1] - p[1]).abs() < 1e-9,
                "{p:?} {q:?}"
            );
        }
        // stitch: the centre column of the equator looks along −X, row 0 at the zenith
        assert!((ray(512, 257, [256.0, 128.0]) - DVec3::NEG_X).length() < 1e-12);
        assert!((ray(512, 257, [17.0, 0.0]) - DVec3::Z).length() < 1e-12);
    }

    /// A known rotation of a textured sphere (4°, ~6 px per frame at 512 px) is recovered to < 0.02°.
    #[test]
    fn rotation() {
        let (w, h) = (512, 256);
        let k = DQuat::from_scaled_axis(DVec3::new(0.3, -0.5, 0.8).normalize() * 4f64.to_radians());
        let (a, b) = (render(w, h, DQuat::IDENTITY), render(w, h, k));
        let mut m = vec![];
        track(&a, &b, [0.2 * PI, 0.8 * PI], &mut m);
        let r = |p: [f32; 2]| ray(w, h, p.map(f64::from));
        let rays: Vec<_> = m.iter().map(|p| (r(p[0]), r(p[1]))).collect();
        let f = fit(&rays, 0.01).unwrap();
        let e = crate::log(f.q * k.inverse()).length().to_degrees();
        eprintln!(
            "{} tracks, {} inliers, rms {:.4}°, error {e:.4}°",
            m.len(),
            f.n,
            f.rms.to_degrees()
        );
        assert!(m.len() > 100 && f.n > m.len() * 9 / 10 && e < 0.02);
    }
}
