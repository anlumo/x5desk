//! Star Lapse: a photo or frame series aligned on the sky's rotation and stacked; Parallel Planet.

use crate::{Cache, Ctx, cached, fi, run, tv};
use glam::{DQuat, Mat3, Quat};
use stab::vis;
use std::f64::consts::TAU;

/// The sky's rotation from frame `a` to frame `b` (`b ≈ q·a` on rays of stitch's equirect frame) from luma corners
/// between polar angles `band` (rad; the sky, e.g. [0.05π, 0.45π] on a level shot) tracked with `stab::vis` (pyramidal
/// Lucas–Kanade, forward-backward checked), RANSAC (3 px) + Horn. A frame pair a few pixels apart; chain longer
/// series. None below 10 tracks or 8 inliers.
pub fn align(a: &vis::Pano, b: &vis::Pano, band: [f64; 2]) -> Option<vis::Fit> {
    let mut m = vec![];
    vis::track(a, b, band, &mut m);
    let r = |p: [f32; 2]| vis::ray(a.w, a.h, p.map(f64::from));
    let rays: Vec<_> = m.iter().map(|p| (r(p[0]), r(p[1]))).collect();
    vis::fit(&rays, 3.0 * TAU / a.w as f64)
}

/// Rotations registering a series on its frame `r` by the sky (`align`, `band`): frame i's q (reference ray a shows
/// at q·a in frame i) and its fit's rms (rad). Each frame is aligned to a keyframe, at first `r` itself, so LK's small
/// bias (≈ 0.002° per fit on 2048-wide frames) does not add up over the series; when a fit fails or keeps fewer than a
/// third of the inliers of the keyframe's first fit, the last registered frame becomes the keyframe (rotations
/// chained through it). None where no fit was found.
pub fn series(ps: &[vis::Pano], r: usize, band: [f64; 2]) -> Vec<Option<(DQuat, f64)>> {
    let mut out = vec![None; ps.len()];
    out[r] = Some((DQuat::IDENTITY, 0.0));
    for ix in [
        (r + 1..ps.len()).collect::<Vec<_>>(),
        (0..r).rev().collect(),
    ] {
        // keyframe, its rotation, inliers of its first fit; the last frame registered
        let (mut key, mut qk, mut n0, mut prev) = (r, DQuat::IDENTITY, 0, r);
        for i in ix {
            let mut f = align(&ps[key], &ps[i], band).filter(|f| 3 * f.n >= n0);
            if f.is_none() && prev != key {
                (key, qk, n0) = (prev, out[prev].unwrap().0, 0);
                f = align(&ps[key], &ps[i], band);
            }
            if let Some(f) = f {
                n0 = n0.max(f.n);
                out[i] = Some((f.q * qk, f.rms));
                prev = i;
            }
        }
    }
    out
}

/// An RGBA8 picture (e.g. a photo decoded with `media::mjpeg::Mjpeg::rgba`) as a frame: `Rgba8Unorm`,
/// `TEXTURE_BINDING | COPY_SRC | COPY_DST`.
pub fn image(cx: &Ctx, px: &[u8], w: u32, h: u32) -> wgpu::Texture {
    use wgpu::TextureUsages as U;
    let t = cx.tex(
        w,
        h,
        1,
        wgpu::TextureFormat::Rgba8Unorm,
        U::TEXTURE_BINDING | U::COPY_SRC | U::COPY_DST,
    );
    cx.q.write_texture(
        t.as_image_copy(),
        px,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: None,
        },
        crate::whole(&t),
    );
    t
}

/// Star stacking on the sphere: each frame resampled along the sky's rotation from the reference (`series`, `align`)
/// and averaged in linear light (less noise, the stars sharp), or with `max` the per-channel maximum (star trails: the
/// frames as they are, `q` = identity, draw the sky's turn as arcs); below a horizon the frames are taken as they are
/// (the ground does not turn with the sky). Memory: 16 bytes per output pixel.
pub struct Stack {
    cx: Ctx,
    sum: wgpu::Buffer,
    w: u32,
    h: u32,
    ub: wgpu::Buffer,
    pipe: wgpu::ComputePipeline,
    fin: [wgpu::ComputePipeline; 2],
    /// Frames stacked.
    pub n: u32,
    pub max: bool,
    /// Elevations (degrees) where the ground ends and the sky starts, blended between; None: all sky.
    pub horizon: Option<[f32; 2]>,
    sb: Cache,
    fb: Cache,
}

impl Stack {
    /// A `w` × `h` equirect stack (the reference frame's directions).
    pub fn new(cx: &Ctx, w: u32, h: u32, max: bool) -> Self {
        use wgpu::BufferUsages as B;
        Self {
            sum: cx.buf(16 * w as u64 * h as u64, B::STORAGE | B::COPY_DST),
            w,
            h,
            ub: cx.buf(80, B::UNIFORM | B::COPY_DST),
            pipe: cx.pipe("stack", 0),
            fin: [0, 1].map(|f| cx.pipe("fin", f)),
            n: 0,
            max,
            horizon: None,
            sb: None,
            fb: None,
            cx: cx.clone(),
        }
    }

    pub fn clear(&mut self, enc: &mut wgpu::CommandEncoder) {
        enc.clear_buffer(&self.sum, 0, None);
        self.n = 0;
    }

    /// Records frame `f` (an equirect of any size, `TEXTURE_BINDING`) into the stack: output direction d shows the
    /// frame's `q·d` (`q`: reference -> this frame, the `align` fit), below the horizon its d.
    pub fn push(&mut self, enc: &mut wgpu::CommandEncoder, f: &wgpu::Texture, q: Quat) {
        let m = Mat3::from_quat(q).to_cols_array();
        let mut u = [0u32; 20];
        for c in 0..3 {
            for r in 0..3 {
                u[4 * c + r] = m[3 * c + r].to_bits();
            }
        }
        if let Some([a, b]) = self.horizon {
            u[12..15].copy_from_slice(&[
                a.to_radians().sin().to_bits(),
                b.to_radians().sin().to_bits(),
                1f32.to_bits(),
            ]);
        }
        u[16..19].copy_from_slice(&[self.w, self.h, self.max as u32]);
        self.cx.q.write_buffer(&self.ub, 0, bytemuck::bytes_of(&u));
        let (cx, p, ub, sum) = (&self.cx, &self.pipe, &self.ub, &self.sum);
        let bg = cached(&mut self.sb, f, || {
            cx.bind(
                p,
                &[
                    (10, wgpu::BindingResource::TextureView(&tv(f))),
                    (11, wgpu::BindingResource::Sampler(&cx.smp)),
                    (12, ub.as_entire_binding()),
                    (13, sum.as_entire_binding()),
                ],
            )
        });
        run(enc, p, bg, self.w, self.h);
        self.n += 1;
    }

    /// Records the stack (mean or maximum, sRGB-encoded) into `out` (this size, a `FORMATS` format,
    /// `STORAGE_BINDING`).
    pub fn run(&mut self, enc: &mut wgpu::CommandEncoder, out: &wgpu::Texture) {
        let p = &self.fin[fi(out.format())];
        let (cx, sum) = (&self.cx, &self.sum);
        let bg = cached(&mut self.fb, out, || {
            cx.bind(
                p,
                &[
                    (14, sum.as_entire_binding()),
                    (15, wgpu::BindingResource::TextureView(&tv(out))),
                ],
            )
        });
        run(enc, p, bg, out.width(), out.height());
    }

    /// For the exporter: each frame enters the stack with rotation `q(k)` (output frame k; e.g. the chained `align`
    /// fits from a first pass) and becomes the stack so far (star trails growing, or the noise settling).
    pub fn hook(mut self, q: impl Fn(u64) -> Quat + Send + 'static) -> export::Hook {
        Box::new(move |x: export::Fx| {
            self.push(x.enc, x.tex, q(x.k));
            self.run(x.enc, x.tex);
        })
    }
}

/// Parallel Planet: two views of one equirect in one frame, side by side (or stacked with `tall`); by default the tiny
/// planet (looking down) and its sky world (looking up), each stereographic over 280°.
pub struct Planets {
    pv: [stitch::Preview; 2],
    pub v: [stitch::View; 2],
    pub tall: bool,
    cx: stitch::Ctx,
    /// For the hook: the frame bound for sampling, the target the pair is drawn into.
    pb: Option<(
        wgpu::Texture,
        stitch::Pano,
        wgpu::Texture,
        wgpu::TextureView,
    )>,
}

impl Planets {
    /// Drawing into `format` targets (`Rgba8Unorm`, `Rgba16Float`).
    pub fn new(cx: &stitch::Ctx, format: wgpu::TextureFormat) -> Self {
        let p = |pitch: f32, roll: f32| stitch::View {
            mode: stitch::Mode::Planet,
            pitch,
            roll,
            fov: edit::view::PLANET_FOV,
            planet: 1.0,
            ..Default::default()
        };
        Self {
            pv: [0, 1].map(|_| stitch::Preview::pano(cx, format)),
            v: [p(-90.0, 0.0), p(90.0, 180.0)],
            tall: false,
            cx: cx.clone(),
            pb: None,
        }
    }

    /// Records the pair from `pano` into the whole of `target` (`size` pixels), each view in its half.
    pub fn draw(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pano: &stitch::Pano,
        target: &wgpu::TextureView,
        size: [u32; 2],
    ) {
        let (w, h) = (size[0] as f32, size[1] as f32);
        let r = |i: usize| match self.tall {
            false => [i as f32 * w / 2.0, 0.0, w / 2.0, h],
            true => [0.0, i as f32 * h / 2.0, w, h / 2.0],
        };
        for i in 0..2 {
            self.pv[i].set(&self.v[i], r(i));
        }
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("planets"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        for i in 0..2 {
            let [x, y, w, h] = r(i);
            p.set_viewport(x, y, w, h, 0.0, 1.0);
            self.pv[i].draw(&mut p, pano);
        }
    }

    /// For the exporter on a 360 output (`Out::Pano`, 2:1): the frame becomes the pair, two squares side by side.
    pub fn hook(mut self) -> export::Hook {
        Box::new(move |x: export::Fx| {
            use wgpu::TextureUsages as U;
            let t = x.tex;
            if self.pb.as_ref().is_none_or(|p| p.0 != *t) {
                let o = self.cx.dev.create_texture(&wgpu::TextureDescriptor {
                    label: Some("planets"),
                    size: crate::whole(t),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: t.format(),
                    usage: U::RENDER_ATTACHMENT | U::COPY_SRC,
                    view_formats: &[],
                });
                self.pb = Some((t.clone(), self.cx.pano(&tv(t)), o.clone(), tv(&o)));
            }
            let (_, pano, o, ov) = self.pb.as_ref().unwrap();
            self.draw(x.enc, pano, ov, [t.width(), t.height()]);
            x.enc
                .copy_texture_to_texture(o.as_image_copy(), t.as_image_copy(), crate::whole(t));
        })
    }
}
