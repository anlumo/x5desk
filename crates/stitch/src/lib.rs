//! Dual-fisheye -> equirect / perspective for the Insta360 X5 on the app's shared wgpu device.
//!
//! The map-based stitch of the Insta360 app (`bcv::stitcher::MapMakerRender` + `DynamicStitchDoubleFisheyeStitchProgram`)
//! evaluated per output pixel, with optional RAFT seam flow along the seam belt of its DISFLOW / AI modes
//! (`ins::stitcher::SeamlessBlenderImpl`), colour matching between the lenses and a per-frame orientation with a
//! rolling-shutter term. Inputs are the decoder's NV12 or P010 textures, one per lens or both lenses side by side in
//! one (`Ctx::rect`); the preview also draws from an equirect (`Ctx::pano`). Nothing passes through the CPU per frame.
//! Maths: `../insta360/re/analysis/stitchmath/*.md`, `PROTOCOL.md` §9.9.
//!
//! Every pass that samples the lenses is recorded into the caller's command encoder, so the decoder's semaphores go
//! on the caller's submit. RAFT (`Stitcher::flow`) submits on its own.

mod offset;
mod preview;
mod raft;

pub use offset::{Calib, Lens, Model};
pub use preview::{Input, Mode, Preview, View};
pub use raft::{Raft, attach, cache};

use glam::{Mat3, Quat, Vec3};
use offset::PLEN;
use wgpu::util::DeviceExt as _;

/// Lens frame geometry: per-lens picture size (its `Src` rect) and the window crop applied by the camera before
/// scaling (X5: calibration 5376², crop 5312², video 3840², `.lrv` 832², live preview 1328²; `window_crop_info` in the
/// trailer, the `.lrv`'s own trailer and the live stream's notification 8234 alike).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub w: u32,
    pub h: u32,
    /// Square side of the per-lens window cut from the calibration frame (0 = no crop).
    pub crop: f32,
    /// Window offset from the centred position (`window_crop_info.crop_offset_x/y`). The `.lrv` needs
    /// −½(crop/832 − crop/3840) ≈ −2.5 more on both axes: its picture sits 0.4 px right of and below the `.insv`'s
    /// (best fit against the `.insv`, +2.2 dB; as if each stream's scaler sampled output pixel i at source i·crop/size).
    pub shift: [f32; 2],
}

/// Seam flow belt: a strip along the seam great circle (rows along it over 400°, columns across it), flow estimated
/// between the two lenses' belts, both lenses warped toward each other.
#[derive(Clone, Copy, Debug)]
pub struct Flow {
    /// Belt columns (across the seam) and rows (along it).
    pub cols: u32,
    pub rows: u32,
    /// beltAngle in degrees: the warp ramps over ±beltAngle/2 around the seam.
    pub belt_angle: f64,
    /// Flow region of the belt (first column, width), rendered `scale`× oversampled and box-averaged.
    pub roi: [u32; 2],
    pub scale: u32,
    /// Each new flow = smooth·new + (1 − smooth)·previous (1 = off).
    pub smooth: f32,
}

impl Flow {
    /// RAFT-small on the central 128 columns (±11.7°, about the lens overlap) of a 192 × 2176 belt (5.4 px/°,
    /// ±17.4° ⊇ the ±16° warp zone of beltAngle 32), rendered 2× and box-averaged, smoothed over time (0.3); tuned
    /// against the app's AI flow (PROTOCOL.md §9.9).
    pub const RAFT: Self = Self {
        cols: 192,
        rows: 2176,
        belt_angle: 32.0,
        roi: [32, 128],
        scale: 2,
        smooth: 0.3,
    };
}

#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// Seam blend width in degrees (`ExtraMetadata` `blend_Angle`, clamped to 0..16; app default 6).
    pub blend: f64,
    pub flow: Option<Flow>,
    /// Colour matching between the lenses: weight of each new fit (1 = no smoothing); None = off.
    pub colour: Option<f32>,
    /// `out` is `Rgba16Float` instead of `Rgba8Unorm`: the precision 10-bit YUV needs.
    pub wide: bool,
    /// A YUV 4:2:0 converter of `out` for an encoder (`Stitcher::yuv`); for P010 set `wide` too.
    pub yuv: Option<Fmt>,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            blend: 6.0,
            flow: None,
            colour: Some(0.1),
            wide: false,
            yuv: None,
        }
    }
}

/// YUV 4:2:0 as `video::Input::copy_buffer` takes it: BT.709, limited range, `h` rows of Y, then `h / 2` rows of
/// interleaved Cb Cr, `pitch` bytes per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fmt {
    /// P010 (16-bit samples, the 10 bits at the top) for H.265 Main 10; else NV12.
    pub deep: bool,
    /// Chroma siting, the encoder's `Cfg::chroma`: 0 left (co-sited with the even columns, between the rows),
    /// 1 centre (2 × 2 mean), 2 top-left.
    pub chroma: u8,
}

/// Per-frame orientation. Body = calibration world of the camera: Z up, lens 0's axis −X; the output's centre
/// looks along the output world's −X.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    /// Output world -> body. Identity: lens 0 at the panorama centre (X5 videos frame on lens 1: yaw 180°).
    pub q: Quat,
    /// Rolling shutter per lens: the body's rotation vector (rad, body frame) over the lens's whole readout,
    /// i.e. angular velocity × readout time. Texture row v ∈ [0, 1] was exposed (v − ½)·readout after the frame
    /// time. Zero = off.
    pub rs: [Vec3; 2],
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            q: Quat::IDENTITY,
            rs: [Vec3::ZERO; 2],
        }
    }
}

impl Pose {
    /// From Euler angles in degrees: `Rz(yaw)·Ry(pitch)·Rx(roll)` (yaw about up, pitch about the side axis, roll
    /// about lens 0's axis).
    pub fn euler(roll: f32, pitch: f32, yaw: f32) -> Self {
        let r = f32::to_radians;
        Self {
            q: Quat::from_rotation_z(r(yaw))
                * Quat::from_rotation_y(r(pitch))
                * Quat::from_rotation_x(r(roll)),
            ..Self::default()
        }
    }

    /// `stitch.wgsl` `Pose`: mat3x3 columns padded to vec4, rs0, rs1 (w = on).
    fn bytes(&self) -> [f32; 20] {
        let m = Mat3::from_quat(self.q).to_cols_array();
        let mut b = [0f32; 20];
        for c in 0..3 {
            b[4 * c..4 * c + 3].copy_from_slice(&m[3 * c..3 * c + 3]);
        }
        for (k, v) in self.rs.iter().enumerate() {
            b[12 + 4 * k..15 + 4 * k].copy_from_slice(&v.to_array());
            b[15 + 4 * k] = (*v != Vec3::ZERO) as u32 as f32;
        }
        b
    }
}

/// One NV12 / P010 lens picture bound for sampling (luma, chroma, its rect and sample scale, sampler).
pub struct Src(wgpu::BindGroup);

/// Full-range Y'CbCr from the plane views' unorm values: a scale for every sample, then Cb Cr less their zero
/// (`stitch.wgsl` `Lens::k`). P010's R16Unorm / Rg16Unorm views read a 10-bit code `d` as 64·d / 65535 (the sample in
/// the high bits), so d / 1023 takes 65535 / 65472 and the chroma zero is 512 / 1023; NV12's R8 / RG8 read d / 255.
fn scale(deep: bool) -> [f32; 4] {
    match deep {
        true => [65535.0 / 65472.0, 512.0 / 1023.0, 0.0, 0.0],
        false => [1.0, 128.0 / 255.0, 0.0, 0.0],
    }
}

/// An RGBA equirect bound for `Preview` (`Input::Pano`).
pub struct Pano(wgpu::BindGroup);

/// Shared GPU objects: device handles, layouts, the compiled shaders and pipelines. Cheap to clone.
#[derive(Clone)]
pub struct Ctx {
    pub dev: wgpu::Device,
    pub q: wgpu::Queue,
    lens: wgpu::BindGroupLayout,
    pano: wgpu::BindGroupLayout,
    /// Linear, repeating across the equirect's longitude (lens gathers stay inside their rects).
    smp: wgpu::Sampler,
    state: wgpu::BindGroupLayout,
    /// Group 3 per pass: stitch into RGBA8, belt, stats, view, stitch into RGBA16F, view with the flows.
    g3: [wgpu::BindGroupLayout; 6],
    /// post.wgsl: fold, fin, yuv.
    post: [wgpu::BindGroupLayout; 3],
    module: wgpu::ShaderModule,
    /// Stitch into RGBA8, RGBA16F.
    main: [wgpu::ComputePipeline; 2],
    belt: wgpu::ComputePipeline,
    stats: wgpu::ComputePipeline,
    fold: wgpu::ComputePipeline,
    fin: wgpu::ComputePipeline,
    yuv: wgpu::ComputePipeline,
}

fn bgl(dev: &wgpu::Device, e: &[(u32, wgpu::BindingType)]) -> wgpu::BindGroupLayout {
    let e: Vec<_> = e
        .iter()
        .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        })
        .collect();
    dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &e,
    })
}

fn storage(read_only: bool) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

const UNIFORM: wgpu::BindingType = wgpu::BindingType::Buffer {
    ty: wgpu::BufferBindingType::Uniform,
    has_dynamic_offset: false,
    min_binding_size: None,
};

fn bind(
    dev: &wgpu::Device,
    l: &wgpu::BindGroupLayout,
    e: &[(u32, wgpu::BindingResource)],
) -> wgpu::BindGroup {
    let e: Vec<_> = e
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: l,
        entries: &e,
    })
}

fn buf(dev: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn init(dev: &wgpu::Device, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage,
    })
}

impl Ctx {
    /// On the app's device (`video::Gpu`); `src` needs `TEXTURE_FORMAT_NV12`, or for P010 `TEXTURE_FORMAT_P010` and
    /// `TEXTURE_FORMAT_16BIT_NORM`.
    pub fn new(dev: &wgpu::Device, q: &wgpu::Queue) -> Self {
        use wgpu::BindingType as T;
        let tex = T::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let smp = T::Sampler(wgpu::SamplerBindingType::Filtering);
        let lens = bgl(dev, &[(0, tex), (1, tex), (2, UNIFORM), (3, smp)]);
        let pano = bgl(dev, &[(3, smp), (4, tex)]);
        let state = bgl(dev, &[(0, storage(true)), (1, UNIFORM), (2, storage(true))]);
        let out = |format| T::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let main = |f| bgl(dev, &[(0, out(f)), (2, storage(true))]);
        let g3 = [
            main(wgpu::TextureFormat::Rgba8Unorm),
            bgl(dev, &[(3, storage(false))]),
            bgl(dev, &[(4, storage(false))]),
            bgl(dev, &[(5, UNIFORM)]),
            main(wgpu::TextureFormat::Rgba16Float),
            bgl(dev, &[(2, storage(true)), (5, UNIFORM)]),
        ];
        let post = [
            bgl(
                dev,
                &[(0, storage(true)), (1, storage(false)), (2, UNIFORM)],
            ),
            bgl(
                dev,
                &[(3, storage(true)), (4, storage(false)), (5, UNIFORM)],
            ),
            bgl(dev, &[(6, tex), (7, storage(false)), (8, UNIFORM)]),
        ];
        let wgsl = |label, src: String| {
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        // the storage format of `dst` is part of the shader
        let src = include_str!("stitch.wgsl");
        let module = wgsl("stitch.wgsl", src.into());
        let wide = wgsl(
            "stitch.wgsl rgba16f",
            src.replace("rgba8unorm", "rgba16float"),
        );
        let pm = dev.create_shader_module(wgpu::include_wgsl!("post.wgsl"));
        let pipe = |m: &wgpu::ShaderModule, entry, groups: &[&wgpu::BindGroupLayout]| {
            let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &groups.iter().map(|g| Some(*g)).collect::<Vec<_>>(),
                immediate_size: 0,
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let g = |i: usize| [&lens, &lens, &state, &g3[i]];
        Self {
            main: [pipe(&module, "main", &g(0)), pipe(&wide, "main", &g(4))],
            belt: pipe(&module, "belt", &g(1)),
            stats: pipe(&module, "stats", &g(2)),
            fold: pipe(&pm, "fold", &[&post[0]]),
            fin: pipe(&pm, "fin", &[&post[1]]),
            yuv: pipe(&pm, "yuv", &[&post[2]]),
            smp: dev.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            dev: dev.clone(),
            q: q.clone(),
            lens,
            pano,
            state,
            g3,
            post,
            module,
        }
    }

    /// A whole NV12 or P010 texture, through its plane views (`R8Unorm`, `Rg8Unorm`; P010: `R16Unorm`, `Rg16Unorm`).
    pub fn src(&self, nv12: &wgpu::Texture) -> Src {
        self.rect(nv12, [0, 0, nv12.width(), nv12.height()])
    }

    /// The lens picture at `r` = (x, y, w, h) luma pixels (all even) of an NV12 or P010 texture (full range, BT.709,
    /// as the X5 records), e.g. a decoder output slot:
    /// its display window (`Decoder::view`; the texture may be larger than the picture), or half of it for both
    /// lenses side by side. It samples like a texture of its own: the other half never bleeds in, and the stitch is
    /// identical to one from two separate textures. The `.lrv` proxy (checked against the `.insv`) and the live
    /// preview (plausibility check on a capture) put lens 0 on the left: lens 0 = (x, y, w/2, h), lens 1 =
    /// (x + w/2, y, w/2, h) of the window (x, y, w, h); both H.264 streams code their exact size (no cropping).
    pub fn rect(&self, nv12: &wgpu::Texture, r: [u32; 4]) -> Src {
        assert!(
            r.iter().all(|v| v % 2 == 0)
                && r[0] + r[2] <= nv12.width()
                && r[1] + r[3] <= nv12.height(),
            "lens rect {r:?}: even and inside the {}x{} texture",
            nv12.width(),
            nv12.height()
        );
        use wgpu::{TextureAspect as A, TextureFormat as F};
        let v = |aspect| {
            nv12.create_view(&wgpu::TextureViewDescriptor {
                aspect,
                ..Default::default()
            })
        };
        // each plane's view format is the one wgpu derives from the aspect: R8 / RG8, or R16 / RG16 for P010
        let deep = nv12.format() == F::P010;
        self.sub(&v(A::Plane0), &v(A::Plane1), r, deep)
    }

    /// Luma (R8) and chroma (RG8, half size) planes from any two whole views; R16 / RG16 hold P010 samples (10 bits
    /// at the top): the plane views of a P010 texture, or textures of their own.
    pub fn planes(&self, y: &wgpu::TextureView, uv: &wgpu::TextureView) -> Src {
        let t = y.texture();
        let deep = matches!(
            t.format(),
            wgpu::TextureFormat::P010 | wgpu::TextureFormat::R16Unorm
        );
        self.sub(y, uv, [0, 0, t.width(), t.height()], deep)
    }

    fn sub(&self, y: &wgpu::TextureView, uv: &wgpu::TextureView, r: [u32; 4], deep: bool) -> Src {
        use wgpu::BindingResource as R;
        let mut u = [0u8; 32];
        u[..16].copy_from_slice(bytemuck::bytes_of(&r));
        u[16..].copy_from_slice(bytemuck::bytes_of(&scale(deep)));
        let b = init(&self.dev, &u, wgpu::BufferUsages::UNIFORM);
        Src(bind(
            &self.dev,
            &self.lens,
            &[
                (0, R::TextureView(y)),
                (1, R::TextureView(uv)),
                (2, b.as_entire_binding()),
                (3, R::Sampler(&self.smp)),
            ],
        ))
    }

    /// An RGBA equirect (`Rgba8Unorm` holding sRGB values) for `Preview`: a `Stitcher`'s `out`, e.g. the RAFT stitch
    /// of a paused frame, or the X5 Webcam Mode's in-camera stitch (2880 × 1440 MJPEG). Layout as the stitcher writes
    /// it: column x at longitude 2πx/W (centre = −X), row y at polar angle πy/(H − 1); a picture with row centres at
    /// π(y + ½)/H is off by at most half a pixel. Taken as body frame: stitch it with `Pose::default()` (plus the
    /// rolling shutter) and give the preview the orientation.
    pub fn pano(&self, rgba: &wgpu::TextureView) -> Pano {
        use wgpu::BindingResource as R;
        Pano(bind(
            &self.dev,
            &self.pano,
            &[(3, R::Sampler(&self.smp)), (4, R::TextureView(rgba))],
        ))
    }
}

/// What every renderer of the two lenses keeps: static parameters, pose, colour matching.
#[derive(Clone)]
struct State {
    pose: wgpu::Buffer,
    col: wgpu::Buffer,
    bg: wgpu::BindGroup,
    /// Colour matching on: statistics (group 3) and fold bind groups.
    colour: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
}

/// Colour statistics workgroups (`stitch.wgsl` `stats`).
const GROUPS: u32 = 64;
/// Identity colour state: gains 1, offsets 0, no fits yet.
const COL0: [f32; 20] = [
    1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
];

impl State {
    fn new(cx: &Ctx, p: &[f32; PLEN], colour: Option<f32>) -> Self {
        use wgpu::BufferUsages as U;
        let d = &cx.dev;
        let pb = init(d, bytemuck::bytes_of(p), U::STORAGE);
        let pose = init(
            d,
            bytemuck::bytes_of(&Pose::default().bytes()),
            U::UNIFORM | U::COPY_DST,
        );
        let col = init(
            d,
            bytemuck::bytes_of(&COL0),
            U::STORAGE | U::COPY_DST | U::COPY_SRC,
        );
        let bg = bind(
            d,
            &cx.state,
            &[
                (0, pb.as_entire_binding()),
                (1, pose.as_entire_binding()),
                (2, col.as_entire_binding()),
            ],
        );
        let colour = colour.map(|s| {
            let part = buf(d, GROUPS as u64 * 64, U::STORAGE);
            let u = init(
                d,
                bytemuck::bytes_of(&[GROUPS as f32, s, 0.0, 0.0]),
                U::UNIFORM,
            );
            (
                bind(d, &cx.g3[2], &[(4, part.as_entire_binding())]),
                bind(
                    d,
                    &cx.post[0],
                    &[
                        (0, part.as_entire_binding()),
                        (1, col.as_entire_binding()),
                        (2, u.as_entire_binding()),
                    ],
                ),
            )
        });
        Self {
            pose,
            col,
            bg,
            colour,
        }
    }

    /// Records the colour fit for this frame (no-op when off); leaves groups 0..3 to be re-bound.
    fn colour(&self, cx: &Ctx, c: &mut wgpu::ComputePass, src: [&Src; 2]) {
        let Some((stats, fold)) = &self.colour else {
            return;
        };
        c.set_pipeline(&cx.stats);
        self.groups(c, src, stats);
        c.dispatch_workgroups(GROUPS, 1, 1);
        c.set_pipeline(&cx.fold);
        c.set_bind_group(0, fold, &[]);
        c.dispatch_workgroups(1, 1, 1);
    }

    fn groups(&self, c: &mut wgpu::ComputePass, src: [&Src; 2], g3: &wgpu::BindGroup) {
        c.set_bind_group(0, &src[0].0, &[]);
        c.set_bind_group(1, &src[1].0, &[]);
        c.set_bind_group(2, &self.bg, &[]);
        c.set_bind_group(3, g3, &[]);
    }
}

/// RAFT state: the belt pass's output, the region flows copied from RAFT, the post pass.
struct Belt {
    cfg: Flow,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    raw: wgpu::Buffer,
    fin: wgpu::Buffer,
    fbg: wgpu::BindGroup,
    /// No flow yet: the next one is taken without smoothing.
    fresh: bool,
}

/// Equirect stitcher at a fixed output size.
pub struct Stitcher {
    cx: Ctx,
    st: State,
    /// The panorama (R'G'B', as the lenses' Y'CbCr converts): row 0 = zenith, centre column = the pose's −X.
    /// `Rgba8Unorm`, or `Rgba16Float` (`Opts::wide`); it can be sampled, copied, rendered into and bound as storage
    /// (an effect pass between `encode` and the conversion).
    pub out: wgpu::Texture,
    /// `out` -> YUV 4:2:0 (`Opts::yuv`): record `yuv.encode` after `encode` (and after any effect pass on `out`),
    /// then copy `yuv.buf` into the encoder's input.
    pub yuv: Option<Yuv>,
    w: u32,
    h: u32,
    g3: wgpu::BindGroup,
    /// the seam flows `encode` warps with (also `Preview::seam`'s)
    flows: wgpu::Buffer,
    wide: bool,
    belt: Option<Belt>,
}

impl Stitcher {
    /// Output `w` × `h` (2:1, w a multiple of 8).
    pub fn new(cx: &Ctx, cal: &Calib, fr: Frame, w: u32, h: u32, o: Opts) -> Result<Self, String> {
        use wgpu::{BindingResource as R, BufferUsages as U};
        if !w.is_multiple_of(8) || !h.is_multiple_of(2) {
            return Err(format!(
                "output {w}x{h}: width must be a multiple of 8, height even"
            ));
        }
        let d = &cx.dev;
        let mut p = cal.pack(fr, o.blend.clamp(0.0, 16.0), o.flow.as_ref());
        p[0] = w as f32;
        p[1] = h as f32;
        let st = State::new(cx, &p, o.colour);
        let wide = o.wide;
        let out = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("pano"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if wide {
                wgpu::TextureFormat::Rgba16Float
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = out.create_view(&Default::default());
        let yuv = match o.yuv {
            Some(f) => Some(Yuv::new(cx, &view, f, true)?),
            None => None,
        };
        // the shader binds the flows also without flow: a minimal buffer then
        let n = o.flow.map_or(1, |f| (f.cols * f.rows) as u64);
        let flows = buf(d, 2 * n * 8, U::STORAGE | U::COPY_DST);
        let g3 = bind(
            d,
            &cx.g3[if wide { 4 } else { 0 }],
            &[(0, R::TextureView(&view)), (2, flows.as_entire_binding())],
        );
        let belt = o.flow.map(|cfg| {
            let n = (cfg.roi[1] * cfg.rows) as u64;
            let b = buf(d, 2 * 3 * n * 4, U::STORAGE | U::COPY_SRC);
            let raw = buf(d, 2 * 2 * n * 4, U::STORAGE | U::COPY_DST);
            let fin = buf(d, 48, U::UNIFORM | U::COPY_DST);
            Belt {
                cfg,
                bg: bind(d, &cx.g3[1], &[(3, b.as_entire_binding())]),
                fbg: bind(
                    d,
                    &cx.post[1],
                    &[
                        (3, raw.as_entire_binding()),
                        (4, flows.as_entire_binding()),
                        (5, fin.as_entire_binding()),
                    ],
                ),
                buf: b,
                raw,
                fin,
                fresh: true,
            }
        });
        Ok(Self {
            cx: cx.clone(),
            st,
            out,
            yuv,
            w,
            h,
            g3,
            flows,
            wide,
            belt,
        })
    }

    /// This frame's orientation and rolling shutter.
    pub fn pose(&self, p: &Pose) {
        self.cx
            .q
            .write_buffer(&self.st.pose, 0, bytemuck::bytes_of(&p.bytes()));
    }

    /// Forget the colour fit and the flow (after a seek or a cut).
    pub fn reset(&mut self) {
        self.cx
            .q
            .write_buffer(&self.st.col, 0, bytemuck::bytes_of(&COL0));
        if let Some(b) = &mut self.belt {
            b.fresh = true;
        }
    }

    /// The current colour fit, read back (waits for the GPU): gain and offset of lens 0, then of lens 1, applied as
    /// `g·rgb + o` on 0..1 values; None when colour matching is off.
    pub fn fit(&self) -> Option<[[f32; 3]; 4]> {
        self.st.colour.as_ref()?;
        let b = read_buffer(&self.cx.dev, &self.cx.q, &self.st.col);
        let f: &[f32] = bytemuck::cast_slice(&b);
        Some(std::array::from_fn(|k| {
            [f[4 * k], f[4 * k + 1], f[4 * k + 2]]
        }))
    }

    /// Records colour matching (if on) and the stitch of `src` (lens 0, lens 1) into `out`.
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder, src: [&Src; 2]) {
        let mut c = enc.begin_compute_pass(&Default::default());
        self.st.colour(&self.cx, &mut c, src);
        c.set_pipeline(&self.cx.main[self.wide as usize]);
        self.st.groups(&mut c, src, &self.g3);
        c.dispatch_workgroups(self.w.div_ceil(8), self.h.div_ceil(8), 1);
    }

    /// Records the belt pass for `src` into RAFT's inputs (needs `Opts::flow`). Submit it, then call `flow`; the
    /// inputs belong to the `Raft`, so one belt → flow at a time per `Raft`.
    pub fn belt(&self, enc: &mut wgpu::CommandEncoder, src: [&Src; 2], raft: &Raft) {
        let b = self.belt.as_ref().expect("Stitcher::belt needs Opts::flow");
        assert_eq!(
            (b.cfg.roi[1], b.cfg.rows),
            raft::SIZE,
            "flow region vs RAFT model"
        );
        {
            let mut c = enc.begin_compute_pass(&Default::default());
            c.set_pipeline(&self.cx.belt);
            self.st.groups(&mut c, src, &b.bg);
            c.dispatch_workgroups(b.cfg.roi[1].div_ceil(8), b.cfg.rows.div_ceil(8), 1);
        }
        let n = b.buf.size() / 2;
        for (k, r) in raft.io.iter().enumerate() {
            let r = r.resource();
            enc.copy_buffer_to_buffer(&b.buf, k as u64 * n, &r.buffer, r.offset, n);
        }
    }

    /// Runs RAFT on the last submitted belt and folds the result into the flows the next `encode` warps with.
    pub fn flow(&mut self, raft: &Raft) -> Result<(), String> {
        let b = self.belt.as_mut().expect("Stitcher::flow needs Opts::flow");
        let o = raft.run()?;
        let (w, h) = (b.cfg.roi[1], b.cfg.rows);
        let size = 4 * (1 + o.st[0] + o.st[1] + (h - 1) * o.st[2] + (w - 1) * o.st[3]) as u64;
        if size > b.raw.size() {
            return Err(format!(
                "RAFT output strides {:?} exceed the flow region",
                o.st
            ));
        }
        let r = o.r.resource();
        let mut u = [0u32; 12];
        u[..4].copy_from_slice(&o.st);
        u[4..10].copy_from_slice(&[
            b.cfg.cols,
            h,
            b.cfg.roi[0],
            w,
            b.cfg.smooth.to_bits(),
            b.fresh as u32,
        ]);
        self.cx.q.write_buffer(&b.fin, 0, bytemuck::bytes_of(&u));
        let mut enc = self.cx.dev.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&r.buffer, r.offset, &b.raw, 0, size);
        {
            let mut c = enc.begin_compute_pass(&Default::default());
            c.set_pipeline(&self.cx.fin);
            c.set_bind_group(0, &b.fbg, &[]);
            c.dispatch_workgroups(b.cfg.cols.div_ceil(8), h.div_ceil(8), 2);
        }
        self.cx.q.submit([enc.finish()]);
        b.fresh = false;
        // RAFT's output storage stays pinned until the copy above is queued
        drop(o);
        Ok(())
    }
}

/// R'G'B' picture -> YUV 4:2:0 (`Fmt`) for an encoder: the stitcher's own (`Stitcher::yuv`), or one for any
/// picture, e.g. a reframed view rendered by `Preview`, after an effect pass.
pub struct Yuv {
    /// The YUV picture, `pitch` bytes per row (`video::Input::copy_buffer(g, &buf, 0, pitch, f)`): the width rounded
    /// up to 4 pixels, times 2 for P010.
    pub buf: wgpu::Buffer,
    pub pitch: u32,
    bg: wgpu::BindGroup,
    n: [u32; 2],
    pipe: wgpu::ComputePipeline,
}

impl Yuv {
    /// For the whole of `src`, any float format (`Rgba8Unorm`, `Rgba16Float`, values clamped to 0..1); width and
    /// height even. `wrap`: column −1 is the last column (an equirect), else the edges repeat.
    pub fn new(cx: &Ctx, src: &wgpu::TextureView, f: Fmt, wrap: bool) -> Result<Self, String> {
        use wgpu::BufferUsages as U;
        let t = src.texture();
        let (w, h) = (t.width(), t.height());
        if (w | h) % 2 != 0 || f.chroma > 2 {
            return Err(format!(
                "YUV of {w}x{h} with chroma siting {}: width and height even, siting 0..=2",
                f.chroma
            ));
        }
        // the shader writes 4 pixels per invocation
        let pitch = w.next_multiple_of(4) * (1 + f.deep as u32);
        let buf = buf(
            &cx.dev,
            pitch as u64 * h as u64 * 3 / 2,
            U::STORAGE | U::COPY_SRC,
        );
        let u = init(
            &cx.dev,
            bytemuck::bytes_of(&[w, h, f.deep as u32, f.chroma as u32, wrap as u32, 0, 0, 0]),
            U::UNIFORM,
        );
        let bg = bind(
            &cx.dev,
            &cx.post[2],
            &[
                (6, wgpu::BindingResource::TextureView(src)),
                (7, buf.as_entire_binding()),
                (8, u.as_entire_binding()),
            ],
        );
        Ok(Self {
            buf,
            pitch,
            bg,
            n: [w.div_ceil(4).div_ceil(8), (h / 2).div_ceil(8)],
            pipe: cx.yuv.clone(),
        })
    }

    /// Records the conversion of the picture as it is at this point of the command stream.
    pub fn encode(&self, enc: &mut wgpu::CommandEncoder) {
        let mut c = enc.begin_compute_pass(&Default::default());
        c.set_pipeline(&self.pipe);
        c.set_bind_group(0, &self.bg, &[]);
        c.dispatch_workgroups(self.n[0], self.n[1], 1);
    }
}

/// Copies a texture's first mip (tightly packed rows) back to the CPU and waits; for tests and frame grabs.
pub fn read_texture(dev: &wgpu::Device, q: &wgpu::Queue, t: &wgpu::Texture) -> Vec<u8> {
    let bpp = t.format().block_copy_size(None).expect("color format");
    let (w, h) = (t.width(), t.height());
    let row = (w * bpp).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let rb = buf(
        dev,
        (row * h) as u64,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut enc = dev.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &rb,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        t.size(),
    );
    q.submit([enc.finish()]);
    let n = (w * bpp) as usize;
    map(dev, &rb, |m| {
        m.chunks(row as usize)
            .flat_map(|r| &r[..n])
            .copied()
            .collect()
    })
}

/// Copies a buffer (needs `COPY_SRC`) back to the CPU and waits.
pub fn read_buffer(dev: &wgpu::Device, q: &wgpu::Queue, b: &wgpu::Buffer) -> Vec<u8> {
    let rb = buf(
        dev,
        b.size(),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut enc = dev.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(b, 0, &rb, 0, b.size());
    q.submit([enc.finish()]);
    map(dev, &rb, <[u8]>::to_vec)
}

fn map<T>(dev: &wgpu::Device, b: &wgpu::Buffer, f: impl FnOnce(&[u8]) -> T) -> T {
    b.map_async(wgpu::MapMode::Read, .., |r| r.expect("map readback"));
    dev.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let v = f(&b.get_mapped_range(..).expect("mapped range"));
    b.unmap();
    v
}
