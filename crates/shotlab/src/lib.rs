//! Effects (PLAN.md group 14): creative effects on stitched X5 footage, our own. The Insta360 app's templates and
//! models come from its server and are not used (`../insta360/re/analysis/app_features.md` §10, §11.6).
//!
//! Two kinds of output:
//! - Passes on equirect frames in stitch's layout (`Rgba8Unorm` or `Rgba16Float`, sRGB-encoded values), recorded into
//!   the caller's command encoder like stitch and fx: [`Median`] (Empty Scene), [`Trails`] (Flash Neon, Night Road),
//!   [`Stack`] (Frame Stack), [`Delay`] (Frame Delay, time slice), [`Cuts`] (Freeze Trail, Echo Trail: the subject
//!   cut out with a person segmentation network), [`Planets`] (Planet Pair), [`Sky`] (New Sky: the sky found by a
//!   segmentation network, a panorama in its place). Each turns into an `export::Hook` (`hook`), which the
//!   exporter calls on every frame before it encodes it. They are meant for 360 output (`export::Out::Pano`);
//!   `Median`, `Trails` and `Delay` with `Map::Cols` / `Map::Rows` work on flat frames too.
//! - Editor timelines the exporter renders (`export::Cut::Edit`): [`cut`] (Stop Motion, Orbit Lapse, Roll Lapse,
//!   Subject Spin, Pull Out, Subject Framing, Dolly Zoom, Passing Vehicle).
//!
//! [`Clip`] decodes and stitches the frames of a clip for the effects that look at a whole clip first (a clean plate,
//! a star series, a subject's track) and checks with the gyro that a shot is locked off.
//!
//! Parameters a pass writes with `queue.write_buffer` take effect at the next submit: one `push` / `run` of a pass per
//! submit (as the exporter's hook and `Clip::frames` call them). Buffers of a full frame (the trails, a stack) exceed
//! wgpu's default 128 MB binding limit at 5760 × 2880: the device needs the adapter's limits, as `video::Gpu` opens it.

mod clip;
mod clone;
pub mod cut;
mod seg;
mod sky;
mod stars;

pub use clip::Clip;
pub use clone::{Clones, Cuts};
pub use sky::{MW, Sky, gradient};
pub use stars::{Planets, Stack, align, image, series};

use glam::Vec3;

/// Frame formats: `Rgba8Unorm`, `Rgba16Float` (10-bit export).
pub const FORMATS: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba16Float,
];

fn fi(f: wgpu::TextureFormat) -> usize {
    FORMATS
        .iter()
        .position(|&x| x == f)
        .unwrap_or_else(|| panic!("frame format {f:?}: Rgba8Unorm or Rgba16Float"))
}

/// Shared GPU objects: device handles, the shader per destination format, a sampler. Cheap to clone.
#[derive(Clone)]
pub struct Ctx {
    pub dev: wgpu::Device,
    pub q: wgpu::Queue,
    m: [wgpu::ShaderModule; 2],
    /// Linear, repeating across longitude, clamped at the poles.
    smp: wgpu::Sampler,
}

impl Ctx {
    pub fn new(dev: &wgpu::Device, q: &wgpu::Queue) -> Self {
        let src = include_str!("shot.wgsl");
        Self {
            m: ["rgba8unorm", "rgba16float"].map(|f| {
                dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("shot.wgsl"),
                    source: wgpu::ShaderSource::Wgsl(src.replace("OUTFMT", f).into()),
                })
            }),
            smp: dev.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            dev: dev.clone(),
            q: q.clone(),
        }
    }

    /// Entry `e` writing `FORMATS[f]`, its layout derived from the shader.
    fn pipe(&self, e: &str, f: usize) -> wgpu::ComputePipeline {
        self.dev
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: None,
                module: &self.m[f],
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    fn bind(
        &self,
        p: &wgpu::ComputePipeline,
        e: &[(u32, wgpu::BindingResource)],
    ) -> wgpu::BindGroup {
        let e: Vec<_> = e
            .iter()
            .map(|(binding, r)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: r.clone(),
            })
            .collect();
        self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &p.get_bind_group_layout(0),
            entries: &e,
        })
    }

    fn buf(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shotlab"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    fn tex(
        &self,
        w: u32,
        h: u32,
        n: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        self.dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("shotlab"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: n,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }
}

/// A bind group kept while its texture stays the same (the exporter hands a new frame texture per decode pass).
type Cache = Option<(wgpu::Texture, wgpu::BindGroup)>;

fn cached<'a>(
    c: &'a mut Cache,
    t: &wgpu::Texture,
    mk: impl FnOnce() -> wgpu::BindGroup,
) -> &'a wgpu::BindGroup {
    if c.as_ref().is_none_or(|x| x.0 != *t) {
        *c = Some((t.clone(), mk()));
    }
    &c.as_ref().unwrap().1
}

fn run(
    enc: &mut wgpu::CommandEncoder,
    p: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    w: u32,
    h: u32,
) {
    let mut c = enc.begin_compute_pass(&Default::default());
    c.set_pipeline(p);
    c.set_bind_group(0, bg, &[]);
    c.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
}

fn layer(t: &wgpu::Texture, k: u32) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture: t,
        mip_level: 0,
        origin: wgpu::Origin3d { x: 0, y: 0, z: k },
        aspect: wgpu::TextureAspect::All,
    }
}

fn whole(t: &wgpu::Texture) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: t.width(),
        height: t.height(),
        depth_or_array_layers: 1,
    }
}

fn array(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn tv(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&Default::default())
}

/// Empty Scene: the per-pixel, per-channel median of up to `n` frames of a locked-off shot (the lower median for
/// an even count; exact on 8-bit frames, to 1/4095 on 16-bit float ones, clamped to 0..1). Whatever covers a pixel in
/// fewer than half of the frames is gone. Memory: `n` frames (66 MB each at 5760 × 2880, 8-bit).
pub struct Median {
    cx: Ctx,
    ring: wgpu::Texture,
    rv: wgpu::TextureView,
    ub: wgpu::Buffer,
    pipe: [wgpu::ComputePipeline; 2],
    /// Frames pushed.
    pub k: u32,
    bg: Cache,
}

impl Median {
    /// For `w` × `h` frames in `format` (`FORMATS`), `n` of them (2 to the device's array limit, 256 at least).
    pub fn new(cx: &Ctx, w: u32, h: u32, n: u32, format: wgpu::TextureFormat) -> Self {
        use wgpu::TextureUsages as U;
        fi(format);
        let ring = cx.tex(w, h, n.max(2), format, U::COPY_DST | U::TEXTURE_BINDING);
        Self {
            rv: array(&ring),
            ring,
            ub: cx.buf(
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            pipe: [0, 1].map(|f| cx.pipe("med", f)),
            k: 0,
            bg: None,
            cx: cx.clone(),
        }
    }

    /// Frames held.
    pub fn len(&self) -> u32 {
        self.k.min(self.ring.depth_or_array_layers())
    }

    pub fn is_empty(&self) -> bool {
        self.k == 0
    }

    /// Records the copy of frame `f` (this size and format, `COPY_SRC`); once the ring is full, over the oldest.
    pub fn push(&mut self, enc: &mut wgpu::CommandEncoder, f: &wgpu::Texture) {
        let n = self.ring.depth_or_array_layers();
        enc.copy_texture_to_texture(f.as_image_copy(), layer(&self.ring, self.k % n), whole(f));
        self.k += 1;
    }

    /// Records the median of the frames held into `out` (this size, a `FORMATS` format, `STORAGE_BINDING`; it may be
    /// the frame just pushed).
    pub fn run(&mut self, enc: &mut wgpu::CommandEncoder, out: &wgpu::Texture) {
        let deep = self.ring.format() == FORMATS[1];
        let u: [u32; 4] = [
            self.len().max(1),
            if deep { 4095f32 } else { 255.0 }.to_bits(),
            if deep { 12 } else { 8 },
            0,
        ];
        self.cx.q.write_buffer(&self.ub, 0, bytemuck::bytes_of(&u));
        let p = &self.pipe[fi(out.format())];
        let (cx, rv, ub) = (&self.cx, &self.rv, &self.ub);
        let bg = cached(&mut self.bg, out, || {
            cx.bind(
                p,
                &[
                    (0, wgpu::BindingResource::TextureView(rv)),
                    (1, ub.as_entire_binding()),
                    (2, wgpu::BindingResource::TextureView(&tv(out))),
                ],
            )
        });
        run(enc, p, bg, out.width(), out.height());
    }

    /// For the exporter: every `every`-th frame enters the ring and each frame becomes the median of the ring, a running
    /// clean plate (whoever stays on a pixel for less than half of the ring's span, `n · every` frames, vanishes).
    pub fn hook(mut self, every: u64) -> export::Hook {
        Box::new(move |x: export::Fx| {
            if x.k.is_multiple_of(every.max(1)) {
                self.push(x.enc, x.tex);
            }
            self.run(x.enc, x.tex);
        })
    }
}

/// Light trail parameters. Light is linear (sRGB decoded), luminance BT.709.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// A pixel's light enters the trails from luminance `thr` on, fully from `thr + soft`.
    pub thr: f32,
    pub soft: f32,
    /// The trails' factor per frame: 1 keeps them, 0.5^(1 / (half-life · fps)) fades them.
    pub decay: f32,
    /// Light into the trails is scaled by this.
    pub gain: f32,
    /// The trails add to the frame instead of lightening it (per-channel max).
    pub add: bool,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            thr: 0.5,
            soft: 0.25,
            decay: 1.0,
            gain: 1.0,
            add: false,
        }
    }
}

/// Flash Neon / Night Road: the bright parts of every frame kept as a decaying per-channel maximum in linear light and
/// laid over the frame. Memory: 8 bytes per pixel (half floats) plus a frame copy.
pub struct Trails {
    cx: Ctx,
    acc: wgpu::Buffer,
    tmp: wgpu::Texture,
    tv: wgpu::TextureView,
    ub: wgpu::Buffer,
    pipe: wgpu::ComputePipeline,
    pub p: Light,
    bg: Cache,
}

impl Trails {
    /// For `w` × `h` frames in `format` (`FORMATS`).
    pub fn new(cx: &Ctx, w: u32, h: u32, format: wgpu::TextureFormat, p: Light) -> Self {
        use wgpu::{BufferUsages as B, TextureUsages as U};
        let tmp = cx.tex(w, h, 1, format, U::COPY_DST | U::TEXTURE_BINDING);
        Self {
            acc: cx.buf(8 * w as u64 * h as u64, B::STORAGE | B::COPY_DST),
            tv: tv(&tmp),
            tmp,
            ub: cx.buf(32, B::UNIFORM | B::COPY_DST),
            pipe: cx.pipe("trail", fi(format)),
            p,
            bg: None,
            cx: cx.clone(),
        }
    }

    /// Forgets the trails.
    pub fn clear(&self, enc: &mut wgpu::CommandEncoder) {
        enc.clear_buffer(&self.acc, 0, None);
    }

    /// Records: the light of frame `f` (this size and format; `COPY_SRC`, `STORAGE_BINDING`) into the trails when
    /// `on` (else they only decay), then the trails over the frame, back into `f`.
    pub fn push(&mut self, enc: &mut wgpu::CommandEncoder, f: &wgpu::Texture, on: bool) {
        let p = self.p;
        let u: [u32; 8] = [
            p.thr.to_bits(),
            p.soft.max(1e-4).to_bits(),
            p.decay.clamp(0.0, 1.0).to_bits(),
            p.gain.max(0.0).to_bits(),
            p.add as u32,
            on as u32,
            f.width(),
            0,
        ];
        self.cx.q.write_buffer(&self.ub, 0, bytemuck::bytes_of(&u));
        enc.copy_texture_to_texture(f.as_image_copy(), self.tmp.as_image_copy(), whole(f));
        let (cx, pipe, t, acc, ub) = (&self.cx, &self.pipe, &self.tv, &self.acc, &self.ub);
        let bg = cached(&mut self.bg, f, || {
            cx.bind(
                pipe,
                &[
                    (3, wgpu::BindingResource::TextureView(t)),
                    (4, acc.as_entire_binding()),
                    (5, ub.as_entire_binding()),
                    (6, wgpu::BindingResource::TextureView(&tv(f))),
                ],
            )
        });
        run(enc, pipe, bg, f.width(), f.height());
    }

    /// For the exporter: light enters the trails while `win[0] ≤ t < win[1]` (output s); the trails show throughout.
    pub fn hook(mut self, win: [f64; 2]) -> export::Hook {
        Box::new(move |x: export::Fx| {
            let on = x.t >= win[0] && x.t < win[1];
            self.push(x.enc, x.tex, on)
        })
    }
}

/// Where a frame delay lags how much: 0 (now) to 1 (`span` frames back).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Map {
    /// By longitude on a 360 frame: 0 at the direction's azimuth, 1 on the opposite side, both ways round (columns
    /// without a seam).
    Lon(Vec3),
    /// By the angle from the direction on a 360 frame: 0 there, 1 opposite (rings around it; rows with +Z).
    Dir(Vec3),
    /// Across the columns, left to right (a flat frame; a 360 frame has its seam at the edge).
    Cols,
    /// Down the rows.
    Rows,
}

/// Frame Delay / time slice: each pixel shows the frame s·`span` frames back, s ∈ 0..1 from where it is (`Map`,
/// reversed with `rev`), blended linearly between the two nearest frames of a ring on the GPU. Until the ring has
/// filled, the oldest frame stands in for older ones. Memory: `n` frames (66 MB each at 5760 × 2880, 8-bit).
pub struct Delay {
    cx: Ctx,
    ring: wgpu::Texture,
    rv: wgpu::TextureView,
    ub: wgpu::Buffer,
    pipe: wgpu::ComputePipeline,
    /// Frames pushed.
    pub k: u64,
    pub map: Map,
    /// Largest delay, frames (≤ n − 1).
    pub span: f32,
    pub rev: bool,
    bg: Cache,
}

impl Delay {
    /// For `w` × `h` frames in `format`, a ring of `n` (2 to the device's array limit); `span` = n − 1.
    pub fn new(cx: &Ctx, w: u32, h: u32, n: u32, format: wgpu::TextureFormat, map: Map) -> Self {
        use wgpu::TextureUsages as U;
        let n = n.max(2);
        let ring = cx.tex(w, h, n, format, U::COPY_DST | U::TEXTURE_BINDING);
        Self {
            rv: array(&ring),
            ring,
            ub: cx.buf(
                48,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            pipe: cx.pipe("delay", fi(format)),
            k: 0,
            map,
            span: (n - 1) as f32,
            rev: false,
            bg: None,
            cx: cx.clone(),
        }
    }

    /// Records: frame `f` (this size and format; `COPY_SRC`, `STORAGE_BINDING`) into the ring, then the delayed
    /// frame back into `f`.
    pub fn push(&mut self, enc: &mut wgpu::CommandEncoder, f: &wgpu::Texture) {
        let n = self.ring.depth_or_array_layers();
        let h = (self.k % n as u64) as u32;
        enc.copy_texture_to_texture(f.as_image_copy(), layer(&self.ring, h), whole(f));
        self.k += 1;
        let (d, m) = match self.map {
            Map::Lon(d) => (d, 0f32),
            Map::Dir(d) => (d, 1.0),
            Map::Cols => (Vec3::X, 2.0),
            Map::Rows => (Vec3::X, 3.0),
        };
        let d = d.normalize_or(Vec3::NEG_X);
        let u: [u32; 12] = [
            d.x.to_bits(),
            d.y.to_bits(),
            d.z.to_bits(),
            m.to_bits(),
            self.k.min(n as u64) as u32,
            n,
            h,
            self.rev as u32,
            self.span.clamp(0.0, (n - 1) as f32).to_bits(),
            0,
            0,
            0,
        ];
        self.cx.q.write_buffer(&self.ub, 0, bytemuck::bytes_of(&u));
        let (cx, pipe, rv, ub) = (&self.cx, &self.pipe, &self.rv, &self.ub);
        let bg = cached(&mut self.bg, f, || {
            cx.bind(
                pipe,
                &[
                    (7, wgpu::BindingResource::TextureView(rv)),
                    (8, ub.as_entire_binding()),
                    (9, wgpu::BindingResource::TextureView(&tv(f))),
                ],
            )
        });
        run(enc, pipe, bg, f.width(), f.height());
    }

    /// For the exporter: every frame.
    pub fn hook(mut self) -> export::Hook {
        Box::new(move |x: export::Fx| self.push(x.enc, x.tex))
    }
}
