//! Insta360 lens calibration ("offset") strings, as found in `ExtraMetadata.offset{,_v2,_v3,_v6}` of each file's
//! trailer.
//!
//! Layout and models from `Insta360Lens::OffsetParser::Parse` / `ins::internal::*Projection::Project`
//! (`../insta360/re/analysis/stitchmath/offset.md`). The version lives in the last token: `v >> 16` (0 = v1, where the
//! token is `lens_ver << 8 | lens_type`).

use crate::{Flow, Frame};

/// Lens projection, camera frame: +Z optical axis, +X image right, +Y image down.
#[derive(Clone, Debug)]
pub enum Model {
    /// v1/v2: `r = θ·(a0 + a1θ + a2θ² + a3θ³)`, pixel = c + f·r·(cos φ, sin φ).
    Poly { f: f64, a: [f64; 4] },
    /// v3/v6: Mei unified model `m = (X, Y)/(Z + ξ|P|)` plus radial-tangential distortion;
    /// 5 coefficients (k1 k2 k3 p1 p2) for v3, 13 (`RadtanDistortPro`) for v6.
    Omni {
        xi: f64,
        fx: f64,
        fy: f64,
        d: Vec<f64>,
    },
}

#[derive(Clone, Debug)]
pub struct Lens {
    /// Principal point, lens-local calibration pixels (lens `i` has `i·side` subtracted from x).
    pub cx: f64,
    pub cy: f64,
    /// World -> camera rotation, row-major, including lens 1's fixed 180° about Z.
    pub rot: [[f64; 3]; 3],
    pub model: Model,
}

#[derive(Clone, Debug)]
pub struct Calib {
    pub version: u32,
    /// Per-lens square side of the calibration frame (X5: 5376).
    pub side: f64,
    /// Field of view in degrees (lens type 113 = X5: 200).
    pub fov: f64,
    pub lens: [Lens; 2],
}

/// Lens type 113 (X5, "A3"): FOV and the v1 polynomial normalised to radians (`ins::Lens::getCoeff`).
const X5_TYPE: u32 = 113;
const X5_FOV: f64 = 200.0;
const X5_POLY: [f64; 4] = [1.0, -0.15262551, 0.22872608, -0.076451022];

/// Parameter block size in f32s (`stitch.wgsl` `P`).
pub(crate) const PLEN: usize = 96;

impl Calib {
    pub fn parse(s: &str) -> Result<Self, String> {
        let t: Vec<&str> = s.trim().split('_').collect();
        let num = |i: usize| -> Result<f64, String> {
            t.get(i)
                .ok_or("offset too short")?
                .parse::<f64>()
                .map_err(|e| format!("token {i}: {e}"))
        };
        if num(0)? != 2.0 {
            return Err("only 2-lens offsets are supported".into());
        }
        let tail = num(t.len() - 1)? as u32;
        let version = (tail >> 16).max(1);
        let per = match version {
            1 => 6,
            2 => 16,
            3 => 19,
            6 => 27,
            v => return Err(format!("unknown offset version {v}")),
        };
        let (w, h, ty) = if version == 1 {
            (num(13)?, num(14)?, tail & 0xff)
        } else {
            (num(per - 2)?, num(per - 1)?, num(per)? as u32)
        };
        if ty != X5_TYPE {
            return Err(format!("lens type {ty} unsupported (X5 = {X5_TYPE})"));
        }
        let side = w / 2.0;
        if side != h {
            return Err(format!("unexpected calibration frame {w}x{h}"));
        }
        let tm = (X5_FOV / 2.0).to_radians();
        let poly_f = |r: f64, a: &[f64; 4]| {
            r / (tm * (a[0] + a[1] * tm + a[2] * tm * tm + a[3] * tm * tm * tm))
        };
        let lens = |i: usize| -> Result<Lens, String> {
            let b = 1 + per * i;
            let v = |k: usize| num(b + k);
            let (model, c, ang) = match version {
                1 => (
                    Model::Poly {
                        f: poly_f(v(0)?, &X5_POLY),
                        a: X5_POLY,
                    },
                    1,
                    3,
                ),
                2 => {
                    let a = [v(9)?, v(10)?, v(11)?, v(12)?];
                    let f = poly_f(v(0)?, &a);
                    (Model::Poly { f, a }, 1, 3)
                }
                _ => {
                    let d = (11..per - 3).map(&v).collect::<Result<_, _>>()?;
                    let (xi, fx, fy) = (v(0)?, v(1)?, v(2)?);
                    (Model::Omni { xi, fx, fy, d }, 3, 5)
                }
            };
            let mut rot = rotation(v(ang)?, v(ang + 1)?, v(ang + 2)?);
            if i == 1 {
                // BuildDoubleFisheyeSphere: lens 1 = R1·Rz(π); both lenses store ~(0, 0, 90°)
                for r in &mut rot {
                    r[0] = -r[0];
                    r[1] = -r[1];
                }
            }
            Ok(Lens {
                cx: v(c)? - i as f64 * side,
                cy: v(c + 1)?,
                rot,
                model,
            })
        };
        Ok(Self {
            version,
            side,
            fov: X5_FOV,
            lens: [lens(0)?, lens(1)?],
        })
    }

    /// Project a camera-frame ray to lens-local calibration pixels.
    pub fn project(&self, i: usize, p: [f64; 3]) -> [f64; 2] {
        let l = &self.lens[i];
        let [x, y, z] = p;
        match &l.model {
            Model::Poly { f, a } => {
                let th = x.hypot(y).atan2(z);
                let r = th * (a[0] + a[1] * th + a[2] * th * th + a[3] * th * th * th);
                let ph = y.atan2(x);
                [l.cx + f * r * ph.cos(), l.cy + f * r * ph.sin()]
            }
            Model::Omni { xi, fx, fy, d } => {
                let den = z + xi * (x * x + y * y + z * z).sqrt();
                let [xd, yd] = distort(x / den, y / den, d);
                [l.cx + fx * xd, l.cy + fy * yd]
            }
        }
    }

    /// Static shader parameters (`stitch.wgsl` `P`); output size, colour and NV12 flags are the caller's slots.
    pub(crate) fn pack(&self, fr: Frame, blend: f64, flow: Option<&Flow>) -> [f32; PLEN] {
        let mut p = [0f32; PLEN];
        p[2] = (blend / 2.0) as f32;
        p[3] = (blend.max(15.0) / 2.0) as f32; // X5 nadir widening (TemplateBlenderImpl::getAlpha)
        if let Some(b) = flow {
            // SeamlessBlenderImpl belt: rows span A0..A1 = -200..200° along the seam, columns gnomonic across it
            let d = 400.0 / (b.rows - 1) as f64;
            p[4] = 1.0;
            p[5] = b.cols as f32;
            p[6] = b.rows as f32;
            p[7] = -200.0;
            p[8] = d as f32;
            p[9] = ((b.rows - 1) as f64 / 400.0).to_degrees() as f32; // k = (W−1)/(A1−A0)·180/π·kf, kf = 1
            p[10] = (b.belt_angle / 2.0) as f32;
            p[11] = b.scale as f32; // flow region: oversampling, first column, width
            p[12] = b.roi[0] as f32;
            p[13] = b.roi[1] as f32;
        }
        let crop = if fr.crop > 0.0 {
            fr.crop as f64
        } else {
            self.side
        };
        // ins::OffsetConvert::convertOffset: c -= (src - dst)/2 + crop_offset, then rescale dst -> texture
        let s = fr.w as f64 / crop;
        let c0 = [0, 1].map(|k| (self.side - crop) / 2.0 + fr.shift[k] as f64);
        let lim = (self.fov / 2.0).to_radians();
        if flow.is_none() {
            // seam low band (`stitch.wgsl` `low`): even blocks about 3° across at the seam, 90° off the lens axis,
            // at least 2 per side
            let x = |deg: f64| {
                self.project(0, [deg.to_radians().sin(), 0.0, deg.to_radians().cos()])[0]
            };
            let most = (fr.w.min(fr.h) / 4).max(1) as f64;
            let f = ((x(91.5) - x(88.5)) * s / 2.0).round().clamp(1.0, most) * 2.0;
            p[14] = f as f32;
            p[15] = (fr.w as f64 / f).ceil() as f32;
            p[16] = (fr.h as f64 / f).ceil() as f32;
        }
        for (i, l) in self.lens.iter().enumerate() {
            let q = &mut p[32 + 32 * i..32 + 32 * (i + 1)];
            for (k, v) in l.rot.iter().flatten().enumerate() {
                q[k] = *v as f32;
            }
            q[10] = ((l.cx - c0[0]) * s) as f32;
            q[11] = ((l.cy - c0[1]) * s) as f32;
            q[15] = (lim.cos() - 0.01) as f32; // OmniProjection validity: Z/|P| >= cos(FOV/2) - 0.01
            let [ex, _] = self.project(i, [lim.sin(), 0.0, lim.cos()]);
            q[16] = ((ex - l.cx) * s) as f32; // image-circle radius at FOV/2, texture px
            q[17] = 0.2439; // calcFisheyeMaskv feather: alpha = min(1, distance_px · 0.2439)
            match &l.model {
                Model::Poly { f, a } => {
                    q[9] = 0.0;
                    q[12] = (f * s) as f32;
                    for (k, v) in a.iter().enumerate() {
                        q[18 + k] = *v as f32;
                    }
                }
                Model::Omni { xi, fx, fy, d } => {
                    q[9] = if d.len() == 5 { 1.0 } else { 2.0 };
                    q[12] = (fx * s) as f32;
                    q[13] = (fy * s) as f32;
                    q[14] = *xi as f32;
                    for (k, v) in d.iter().enumerate() {
                        q[18 + k] = *v as f32;
                    }
                }
            }
        }
        p
    }
}

/// `R = Ry(a2 + 90°)·Rz(a1)·Rx(a3)`, degrees; world -> camera.
fn rotation(a1: f64, a2: f64, a3: f64) -> [[f64; 3]; 3] {
    let (sy, cy) = (a2 + 90.0).to_radians().sin_cos();
    let (sz, cz) = a1.to_radians().sin_cos();
    let (sx, cx) = a3.to_radians().sin_cos();
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    mul(mul(ry, rz), rx)
}

fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// v3: k1 k2 k3 p1 p2 (OpenCV radtan); v6: `RadtanDistortPro`, d5/d6 swap the p roles.
fn distort(mx: f64, my: f64, d: &[f64]) -> [f64; 2] {
    let r2 = mx * mx + my * my;
    if d.len() == 5 {
        let rad = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * d[2]));
        return [
            mx * rad + 2.0 * d[3] * mx * my + d[4] * (r2 + 2.0 * mx * mx),
            my * rad + d[3] * (r2 + 2.0 * my * my) + 2.0 * d[4] * mx * my,
        ];
    }
    let rad = 1.0 + r2 * (d[0] + r2 * (d[1] + r2 * (d[2] + r2 * (d[3] + r2 * d[4]))));
    let (ta, tb) = (d[5] + d[7] * r2, d[6] + d[8] * r2);
    [
        mx * rad + (2.0 * mx * mx + r2) * ta + 2.0 * mx * my * tb + d[9] * r2 + d[11] * r2 * r2,
        my * rad + (2.0 * my * my + r2) * tb + 2.0 * mx * my * ta + d[10] * r2 + d[12] * r2 * r2,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // synthetic X5-like strings; real calibration comes from each file's trailer
    const V1: &str = "2_2650.000_2690.000_2685.000_-0.500_0.100_90.000_2652.000_8070.000_2680.000_0.500_-0.100_90.500_10752_5376_1137";
    const V3: &str = "2_2.000000_4270.000_4271.000_2690.000_2685.000_-0.500_0.100_90.000_0.000000_0.000000_0.000000_0.20000000_2.00000000_-3.00000000_-0.00100000_-0.00200000_10752_5376_113_2.000000_4272.000_4273.000_8070.000_2680.000_0.500_-0.100_90.500_-0.001500_0.000100_-0.032000_0.19000000_1.90000000_-2.90000000_0.00100000_-0.00150000_10752_5376_113_197632";
    const V6: &str = "2_2.000000_4270.000_4271.000_2690.000_2685.000_-0.500_0.100_90.000_0.000000_0.000000_0.000000_0.20000000_1.50000000_-0.80000000_-3.50000000_0.00000000_-0.00250000_-0.00150000_0.00500000_0.01000000_-0.00200000_-0.00250000_0.00020000_-0.02500000_10752_5376_113_2.000000_4272.000_4273.000_8070.000_2680.000_0.500_-0.100_90.500_0.000500_0.000050_-0.031500_0.24000000_1.45000000_-0.90000000_-3.20000000_0.00000000_-0.00100000_-0.00010000_0.01300000_-0.00300000_-0.00700000_0.00200000_-0.01200000_0.01700000_10752_5376_113_394240";

    fn close(a: [f64; 2], b: [f64; 2]) {
        assert!(
            (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3,
            "{a:?} vs {b:?}"
        );
    }

    // reference values: re/analysis/stitchmath/offset_model.py (matches the emulated library)
    #[test]
    fn versions() {
        for (s, v) in [(V1, 1), (V3, 3), (V6, 6)] {
            assert_eq!(Calib::parse(s).unwrap().version, v);
        }
    }

    #[test]
    fn v3_lens0() {
        let c = Calib::parse(V3).unwrap();
        close(c.project(0, [0.0, 0.0, 1.0]), [2690.000, 2685.000]);
        close(c.project(0, [1.0, 0.0, 0.0]), [5092.142, 2683.932]);
        close(c.project(0, [-1.0, 0.0, 0.0]), [275.048, 2683.932]);
        close(c.project(0, [0.0, 1.0, 0.0]), [2687.865, 5090.908]);
    }
}
