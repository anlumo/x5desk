//! Overlays (our own content): stickers and watermarks from PNGs, text in egui's default font, composited flat or on
//! the sphere.

use crate::{
    Ctx, bind,
    cube::{enc, lin},
    tex,
};
use glam::Vec3;
use wgpu::util::DeviceExt as _;

/// RGBA8 picture, straight alpha, sRGB-encoded, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

impl Image {
    /// Decodes a PNG (any colour type and depth; palettes and 16-bit reduced to RGBA8; gamma / ICC chunks ignored).
    pub fn png(b: &[u8]) -> Result<Self, String> {
        let mut d = png::Decoder::new(std::io::Cursor::new(b));
        d.set_transformations(png::Transformations::normalize_to_color8());
        let mut r = d.read_info().map_err(|e| format!("PNG: {e}"))?;
        let mut buf = vec![0; r.output_buffer_size().ok_or("PNG: too large")?];
        let i = r.next_frame(&mut buf).map_err(|e| format!("PNG: {e}"))?;
        let n = (i.width * i.height) as usize;
        let px = match i.color_type {
            png::ColorType::Rgba => buf[..4 * n].to_vec(),
            png::ColorType::Rgb => buf[..3 * n]
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|c| [c[0], c[1], c[2], 255])
                .collect(),
            png::ColorType::GrayscaleAlpha => buf[..2 * n]
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|c| [c[0], c[0], c[0], c[1]])
                .collect(),
            png::ColorType::Grayscale => buf[..n].iter().flat_map(|&v| [v, v, v, 255]).collect(),
            png::ColorType::Indexed => return Err("PNG: palette not expanded".into()),
        };
        Ok(Self {
            w: i.width,
            h: i.height,
            px,
        })
    }

    /// `s` (lines split at `\n`) in egui's default proportional font (Ubuntu Light, `epaint_default_fonts`) at `px`
    /// pixels per em, coloured `rgba`, on a transparent background with a 1-pixel margin. Glyphs the font lacks (it
    /// covers Latin, Greek and Cyrillic) come out as its missing-glyph box.
    pub fn text(s: &str, px: f32, rgba: [u8; 4]) -> Self {
        use ab_glyph::{Font, FontRef, ScaleFont, point};
        let f = FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT).expect("bundled font");
        let f = f.as_scaled(px);
        let lh = f.height() + f.line_gap();
        let mut gs = vec![];
        let mut wm = 0f32;
        let lines = s.split('\n').count();
        for (k, line) in s.split('\n').enumerate() {
            let mut x = 0.0;
            let mut prev = None;
            for ch in line.chars() {
                let id = f.glyph_id(ch);
                if let Some(p) = prev {
                    x += f.kern(p, id);
                }
                gs.push(id.with_scale_and_position(px, point(x, f.ascent() + k as f32 * lh)));
                x += f.h_advance(id);
                prev = Some(id);
            }
            wm = wm.max(x);
        }
        let (w, h) = (wm.ceil() as u32 + 2, (lines as f32 * lh).ceil() as u32 + 2);
        let mut a = vec![0f32; (w * h) as usize];
        for g in gs {
            if let Some(o) = f.outline_glyph(g) {
                let b = o.px_bounds();
                o.draw(|x, y, c| {
                    let (x, y) = (b.min.x as i32 + x as i32 + 1, b.min.y as i32 + y as i32 + 1);
                    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                        let v = &mut a[(y as u32 * w + x as u32) as usize];
                        *v = (*v + c).min(1.0);
                    }
                });
            }
        }
        Self {
            w,
            h,
            px: a
                .iter()
                .flat_map(|&c| {
                    [
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        (rgba[3] as f32 * c).round() as u8,
                    ]
                })
                .collect(),
        }
    }
}

/// Where an overlay goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Place {
    /// On a flat frame: centre in frame fractions (0…1, y down), width as a fraction of the frame width (height keeps
    /// the image's aspect), rotation in degrees (clockwise).
    Flat { at: [f32; 2], w: f32, rot: f32 },
    /// On an equirect frame: a picture on the plane tangent to the sphere at direction (yaw right of the panorama
    /// centre, pitch up; −90 = nadir, e.g. to cover a tripod), `fov` degrees across its width (up to 170), rotated
    /// `roll` degrees clockwise as seen from the centre. Its top faces the zenith, or the yaw direction at the poles.
    Sphere {
        yaw: f32,
        pitch: f32,
        fov: f32,
        roll: f32,
    },
}

/// An image on the GPU (premultiplied in linear light, mipmapped) with its placement; drawn by `Fx::run`, blended in
/// linear light.
pub struct Overlay {
    pub place: Place,
    /// 0…1.
    pub opacity: f32,
    w: u32,
    h: u32,
    pub(crate) ub: wgpu::Buffer,
    pub(crate) bg: wgpu::BindGroup,
}

impl Overlay {
    pub fn new(cx: &Ctx, img: &Image, place: Place, opacity: f32) -> Self {
        use wgpu::BindingResource as R;
        assert!(
            img.w > 0 && img.h > 0 && img.px.len() == (4 * img.w * img.h) as usize,
            "overlay image"
        );
        // mip chain premultiplied in linear light, 2×2 box per level, sizes max(1, size >> level) as wgpu expects;
        // stored sRGB-encoded (alpha linear), so sampling decodes before it filters
        let mut lv = vec![(
            img.w,
            img.h,
            img.px
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| {
                    let a = p[3] as f32 / 255.0;
                    [0, 1, 2, 3].map(|c| {
                        if c < 3 {
                            lin(p[c] as f32 / 255.0) * a
                        } else {
                            a
                        }
                    })
                })
                .collect::<Vec<[f32; 4]>>(),
        )];
        while let Some(&(w, h, ref p)) = lv.last()
            && (w > 1 || h > 1)
        {
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let n = (0..nw * nh)
                .map(|i| {
                    let (x, y) = (i % nw, i / nw);
                    let mut s = [0.0; 4];
                    for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let q = p[((2 * y + j).min(h - 1) * w + (2 * x + i).min(w - 1)) as usize];
                        for c in 0..4 {
                            s[c] += q[c] / 4.0;
                        }
                    }
                    s
                })
                .collect();
            lv.push((nw, nh, n));
        }
        let t = tex(
            &cx.dev,
            [img.w, img.h, 1],
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            lv.len() as u32,
        );
        for (k, (w, h, p)) in lv.iter().enumerate() {
            let p: Vec<u8> = p
                .iter()
                .flat_map(|q| {
                    [0, 1, 2, 3]
                        .map(|c| (if c < 3 { enc(q[c]) } else { q[c] } * 255.0).round() as u8)
                })
                .collect();
            cx.q.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &t,
                    mip_level: k as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &p,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(*h),
                },
                wgpu::Extent3d {
                    width: *w,
                    height: *h,
                    depth_or_array_layers: 1,
                },
            );
        }
        let ub = cx
            .dev
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fx overlay"),
                contents: &[0; 64],
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bg = bind(
            &cx.dev,
            &cx.ov,
            &[
                (0, R::TextureView(&crate::view(&t))),
                (1, R::Sampler(&cx.smp)),
                (2, ub.as_entire_binding()),
            ],
        );
        Self {
            place,
            opacity,
            w: img.w,
            h: img.h,
            ub,
            bg,
        }
    }

    /// `over.wgsl` `O` for a `fw` × `fh` frame, and the scissor rect (x, y, w, h) that holds the overlay.
    pub(crate) fn uniform(&self, fw: u32, fh: u32) -> ([f32; 16], [u32; 4]) {
        let (w, h) = (fw as f32, fh as f32);
        let ar = self.h as f32 / self.w as f32;
        let op = self.opacity.clamp(0.0, 1.0);
        let clip = |x0: f32, y0: f32, x1: f32, y1: f32| {
            let (x0, y0) = (
                x0.floor().clamp(0.0, w) as u32,
                y0.floor().clamp(0.0, h) as u32,
            );
            let (x1, y1) = (
                x1.ceil().clamp(0.0, w) as u32,
                y1.ceil().clamp(0.0, h) as u32,
            );
            [x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)]
        };
        match self.place {
            Place::Flat { at, w: sw, rot } => {
                let (s, c) = rot.to_radians().sin_cos();
                let (hw, hh) = (sw * w / 2.0, sw * w / 2.0 * ar);
                let (cx, cy) = (at[0] * w, at[1] * h);
                // half extents of the rotated rect
                let (ex, ey) = (
                    hw * c.abs() + hh * s.abs() + 1.0,
                    hw * s.abs() + hh * c.abs() + 1.0,
                );
                (
                    [
                        cx, cy, c, s, hw, hh, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, w, h, op, 0.0,
                    ],
                    clip(cx - ex, cy - ey, cx + ex, cy + ey),
                )
            }
            Place::Sphere {
                yaw,
                pitch,
                fov,
                roll,
            } => {
                let (y, p) = (yaw.to_radians(), pitch.clamp(-90.0, 90.0).to_radians());
                let f = Vec3::new(p.cos() * y.sin(), p.sin(), p.cos() * y.cos());
                let r = Vec3::new(y.cos(), 0.0, -y.sin());
                let u = f.cross(r);
                let (s, c) = roll.to_radians().sin_cos();
                let (r, u) = (r * c - u * s, r * s + u * c);
                let tw = (fov.clamp(0.1, 170.0).to_radians() / 2.0).tan();
                let th = tw * ar;
                let (r, u) = (r / (2.0 * tw), u / (2.0 * th));
                // rows within the overlay's angular radius of its centre, all columns
                let rho = tw.hypot(th).atan().to_degrees();
                let row = |lat: f32| (90.0 - lat) / 180.0 * (h - 1.0) + 0.5;
                (
                    [
                        f.x, f.y, f.z, 0.0, r.x, r.y, r.z, 0.0, u.x, u.y, u.z, 0.0, w, h, op, 1.0,
                    ],
                    clip(0.0, row(pitch + rho) - 1.0, w, row(pitch - rho) + 1.0),
                )
            }
        }
    }
}
