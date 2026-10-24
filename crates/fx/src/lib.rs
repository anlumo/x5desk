//! Image effects on the app's shared wgpu device (PLAN.md groups 7, 12, 14): colour adjustments, automatic colour and
//! underwater correction, `.cube` LUTs and our own looks, a log → Rec.709 input transform, purple / green / blue
//! defringe, clarity and sharpening, overlays (stickers, watermarks, text) on flat or equirect frames.
//!
//! Frames are `Rgba8Unorm` or `Rgba16Float` textures holding display-encoded values (sRGB transfer, as the stitcher
//! writes them; 16-bit float carries 10-bit export precision, values above 1 pass through). Steps that need linear
//! light convert with the sRGB curve. `Fx::run` records into the caller's command encoder (like `stitch`), so decoder
//! and encoder semaphores go on the caller's submit:
//!
//! 1. In place (`Fx::io` with source = destination): a copy of the frame for the passes to read.
//! 2. Overlays, only with some: one render pass draws them into a layer (linear light, premultiplied, `Overlay`).
//! 3. Luma pyramid, only with clarity: `down0` (2×2 box) → `down` ([1 3 3 1]/8 per axis) per level → `up` (bilinear
//!    2×) back to half size. round(log2(width / 180)) levels, so the radius follows the frame (5760 wide: 5).
//! 4. Automatic corrections, only with `Params::water` or `Params::auto`: `stat` (luma-binned statistics of a sample
//!    grid) → `solve` (the estimate, smoothed over time, and the coefficients), all on the GPU, no readback.
//! 5. One fused pass per pixel, in this order: input transform (1D LUT, e.g. log → Rec.709) → defringe → sharpen and
//!    clarity → underwater and auto corrections → exposure and white balance → highlights / shadows → contrast /
//!    whites / blacks → brightness → saturation / vibrance → look (1D and / or 3D LUT, tetrahedral) mixed by
//!    intensity → the overlay layer over it in linear light. Formulas: `Params`.

mod cube;
mod over;
#[cfg(test)]
mod tests;

pub use cube::{Cube, Grade, Log, Look, Lut};
pub use over::{Image, Overlay, Place};

use glam::{DMat3, DVec3};
use wgpu::util::DeviceExt as _;

/// Effect parameters. Every default is neutral; values are clamped to the ranges given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Stops, −5…5: linear RGB × 2^exposure.
    pub exposure: f32,
    /// −1…1: white balance by von Kries adaptation (CAT02) from an assumed scene white to D65. The scene white lies on
    /// the CIE daylight locus at 10⁶/6504 − 75·temperature mired (4372 K at −1, 12698 K at +1), moved by 0.02·tint
    /// along the locus normal in CIE 1960 uv (Duv, toward green). Positive = warmer / more magenta. Scaled so white
    /// keeps its luminance.
    pub temperature: f32,
    pub tint: f32,
    /// −1…1: exposure gains by tone band in linear light, from the luminance Y as ev = log2(Y / 0.18):
    /// stops = highlights·exp(−½((ev − 1.5) / 1.25)²) + 1.25·shadows·exp(−½((ev + 2.25) / 1.5)²). The gain moves by at
    /// most 0.88 stop per stop (shadows +1 with highlights −1), so the mapping stays monotone for every setting.
    pub highlights: f32,
    pub shadows: f32,
    /// −1…1: per-channel tone curve on encoded values x, in this order, t = x clamped to 0…1 before each step (each
    /// step is monotone): x −= contrast·t(1 − t)(1 − 2t) (pivot ½, ends fixed); x += 0.25·whites·t⁴;
    /// x += 0.15·blacks·(1 − t)⁴.
    pub contrast: f32,
    pub whites: f32,
    pub blacks: f32,
    /// −1…1: midtones with black and white fixed, after the tone curve: x ← max(x, 0)^γ per channel on encoded values,
    /// γ = 2^(−0.6·brightness) (mid grey 0.46 → 0.60 at +1, 0.31 at −1).
    pub brightness: f32,
    /// −1…1, in linear light around the luminance Y: c = Y + (c − Y)·(1 + saturation)·(1 + vibrance·(1 − s)), s =
    /// (max − min) / max of c, so vibrance moves dull colours most.
    pub saturation: f32,
    pub vibrance: f32,
    /// −1…1: local contrast on the encoded luma L, added to every channel (chroma unchanged):
    /// L += 0.8·clarity·(L − B)·4t(1 − t), B = the pyramid base, t = L clamped to 0…1 (midtones most).
    pub clarity: f32,
    /// −1…1: lifts dark areas, pulls in highlights and puts back the local contrast that costs, as shadows +
    /// 0.5·brilliance, highlights − 0.5·brilliance, clarity + 0.3·brilliance (each sum clamped to −1…1).
    pub brilliance: f32,
    /// 0…1: unsharp mask on the encoded luma, L += 2·sharpen·(L − its 3×3 binomial blur), every channel.
    pub sharpen: f32,
    /// 0…1: purple / green / blue fringe reduction. Desaturates toward the pixel's luma (luma unchanged) by
    /// a = min(1, e·(smoothstep(0.6, 0.9, max)·(purple·band(h, 285°, ±30°, ±50°) + green·band(h, 120°, ±25°, ±45°)) +
    /// smoothstep(0.85, 0.97, max)·blue·band(h, 220°, ±15°, ±30°))), h = HSV hue, band = 1 within the inner and 0
    /// beyond the outer distance (smoothstep), e = smoothstep(0.25, 0.6, max − min), max and min over the luma of the
    /// (2·reach + 1)² neighbourhood: near high-contrast edges with a bright side, for blue a near-clipped one (blue sky
    /// borders sunlit stone and statues too; it still loses colour right next to clipped clouds or sun). Neutral pixels
    /// have no chroma and stay as they are.
    pub purple: f32,
    pub green: f32,
    pub blue: f32,
    /// Defringe neighbourhood radius in pixels, 1…12 (rounded; 4 = 9 × 9). Fringes are wider in pixels on bigger
    /// frames and on lenses with more lateral chromatic aberration.
    pub reach: f32,
    /// 0…1: underwater colour correction (our AquaVision substitute), estimated from the frame's statistics (`Fx`).
    /// With M the mean colour over all bins:
    /// 1. red compensation (Ancuti et al., "Color Balance and Fusion for Underwater Image Enhancement", IEEE TIP
    ///    27(1), 2018, eq. 4, on linear values, α = 0.5): r ← r + k·(1 − r)·g, k = 0.5·max(0, M_g − M_r), green
    ///    standing in for the red the water took (the bin means follow exactly through their mean r·g);
    /// 2. veil (backscatter): v = the mean compensated colour of the darkest 1 % of the weight;
    /// 3. attenuation: gains g = 1 / (A^¼·H^¾) per channel, scaled so the smallest is 1 and limited to 8, from gray
    ///    world A = mean − v and white patch H = mean of the brightest 1 % (below the clipped top bin) − v.
    ///
    /// At strength s, in linear light after defringe / sharpen / clarity: r ← r + s·k·(1 − r)·g, c ← g^s·(c − s·v),
    /// clamped at 0 after the auto step. α, the veil and the ¼ / ¾ split scored best on synthetic underwater frames
    /// (tests `underwater`); Ancuti et al. use α = 1 and gray world alone.
    pub water: f32,
    /// 0…1: automatic colour (our Color Plus substitute) from the frame's statistics, after the underwater correction
    /// (at full strength) when that is on:
    /// - white balance: half of a von Kries correction of E = √(gray world · white patch (brightest 2 %)) toward
    ///   neutral, g_a = clamp(√(Y(E) / E), 0.8, 1.25) per channel, rescaled so E keeps its luminance Y;
    /// - black point β = half the luminance at 0.5 % of the weight;
    /// - exposure ev = clamp(½·log2(0.18 / the log-average of luminance − β), −1, 1), at most c + 0.74 where c ≥ 0
    ///   keeps the luminance at 99.5 % from clipping more than it did, and highlights −clamp((ev − c) / 0.74, 0, 1)
    ///   (the highlights band gives those 0.74 EV back at luminance 1);
    /// - contrast + clamp(2·(0.25 − σ), 0, 0.3), σ = the standard deviation of the encoded luminance after these;
    /// - vibrance + clamp(2·(0.35 − the mean HSV saturation), 0, 0.5).
    ///
    /// At strength s: c ← max(g_a^s·2^(s·ev)·(c − s·β), 0) in linear light; highlights, contrast and vibrance +
    /// s·their auto values (sums clamped to −1…1).
    pub auto: f32,
    /// 0…1: look intensity, out = mix(c, look(c), mix).
    pub mix: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            temperature: 0.0,
            tint: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            contrast: 0.0,
            whites: 0.0,
            blacks: 0.0,
            brightness: 0.0,
            saturation: 0.0,
            vibrance: 0.0,
            clarity: 0.0,
            brilliance: 0.0,
            sharpen: 0.0,
            purple: 0.0,
            green: 0.0,
            blue: 0.0,
            reach: 4.0,
            water: 0.0,
            auto: 0.0,
            mix: 1.0,
        }
    }
}

/// Rec.709 luma / luminance weights.
const W: DVec3 = DVec3::new(0.2126, 0.7152, 0.0722);

/// Time constant of the automatic corrections' smoothing (s).
const TAU: f64 = 0.5;

/// White-balance matrix on linear Rec.709 RGB (`Params::temperature`).
fn wb(temp: f32, tint: f32) -> DMat3 {
    if temp == 0.0 && tint == 0.0 {
        return DMat3::IDENTITY;
    }
    // CIE daylight locus (CIE 15), 4000–25000 K
    let day = |t: f64| {
        let s = 1e3 / t;
        let x = if t <= 7e3 {
            ((-4.607 * s + 2.9678) * s + 0.09911) * s + 0.244063
        } else {
            ((-2.0064 * s + 1.9018) * s + 0.24748) * s + 0.23704
        };
        (x, (-3.0 * x + 2.87) * x - 0.275)
    };
    let xyz = |(x, y): (f64, f64)| DVec3::new(x / y, 1.0, (1.0 - x - y) / y);
    // CIE 1960 uv
    let uv = |t: f64| {
        let (x, y) = day(t);
        let d = -2.0 * x + 12.0 * y + 3.0;
        glam::DVec2::new(4.0 * x / d, 6.0 * y / d)
    };
    let k = 1e6 / (1e6 / 6504.0 - 75.0 * temp as f64);
    let t = (uv(k * 1.001) - uv(k / 1.001)).normalize();
    // the locus normal on its green (+v) side
    let n = t.perp() * t.perp().y.signum();
    let p = uv(k) + 0.02 * tint as f64 * n;
    let (u, v) = (p.x, p.y);
    let e = 2.0 * u - 8.0 * v + 4.0;
    // sRGB → XYZ (D65) and CAT02, column-major
    let to = DMat3::from_cols_array(&[
        0.4124564, 0.2126729, 0.0193339, 0.3575761, 0.7151522, 0.1191920, 0.1804375, 0.0721750,
        0.9503041,
    ]);
    let cat = DMat3::from_cols_array(&[
        0.7328, -0.7036, 0.0030, 0.4296, 1.6975, 0.0136, -0.1624, 0.0061, 0.9834,
    ]);
    let g = (cat * xyz(day(6504.0))) / (cat * xyz((3.0 * u / e, 2.0 * v / e)));
    let m = to.inverse() * cat.inverse() * DMat3::from_diagonal(g) * cat * to;
    m / W.dot(m * DVec3::ONE)
}

/// `fx.wgsl` `Par`: 16 vec4.
type Par = [[f32; 4]; 16];

/// Output formats: `Rgba8Unorm`, `Rgba16Float`.
pub const FORMATS: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba16Float,
];

/// Overlay layer format: blending into an sRGB target happens in linear light.
const LAYER: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// `fx.wgsl` statistics: values per workgroup (64 luma bins × 6).
const NS: u64 = 384;

fn bgl(
    dev: &wgpu::Device,
    vis: wgpu::ShaderStages,
    e: &[(u32, wgpu::BindingType)],
) -> wgpu::BindGroupLayout {
    let e: Vec<_> = e
        .iter()
        .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty,
            count: None,
        })
        .collect();
    dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &e,
    })
}

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

fn tex(
    dev: &wgpu::Device,
    size: [u32; 3],
    dim: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    mips: u32,
) -> wgpu::Texture {
    dev.create_texture(&wgpu::TextureDescriptor {
        label: Some("fx"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: size[2],
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: dim,
        format,
        usage,
        view_formats: &[],
    })
}

fn view(t: &wgpu::Texture) -> wgpu::TextureView {
    t.create_view(&Default::default())
}

/// Shared GPU objects: device handles, layouts, pipelines. Cheap to clone.
#[derive(Clone)]
pub struct Ctx {
    pub dev: wgpu::Device,
    pub q: wgpu::Queue,
    src: wgpu::BindGroupLayout,
    st: wgpu::BindGroupLayout,
    lv: wgpu::BindGroupLayout,
    /// Main pass group 2 per `FORMATS` entry: pyramid base, output, automatic corrections.
    out: [wgpu::BindGroupLayout; 2],
    /// Statistics group: per-workgroup sums, state.
    an: wgpu::BindGroupLayout,
    ov: wgpu::BindGroupLayout,
    /// Per `FORMATS` entry: without, with the overlay layer.
    main: [[wgpu::ComputePipeline; 2]; 2],
    down0: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    up: wgpu::ComputePipeline,
    stat: wgpu::ComputePipeline,
    solve: wgpu::ComputePipeline,
    over: wgpu::RenderPipeline,
    /// Overlays: trilinear, anisotropic, clamped.
    smp: wgpu::Sampler,
    /// 1-texel placeholders: r32float (`down0`'s pyramid input), 3D look, overlay layer.
    nil: [wgpu::TextureView; 3],
}

impl Ctx {
    /// On the app's device (`video::Gpu`); needs no optional features.
    pub fn new(dev: &wgpu::Device, q: &wgpu::Queue) -> Self {
        use wgpu::{BindingType as T, ShaderStages as V, TextureFormat as F};
        let t = |d| T::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: d,
            multisampled: false,
        };
        let w = |format| T::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let buf = |ty| T::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let sb = |read_only| buf(wgpu::BufferBindingType::Storage { read_only });
        let d2 = wgpu::TextureViewDimension::D2;
        let c = V::COMPUTE;
        let src = bgl(dev, c, &[(0, t(d2))]);
        let st = bgl(
            dev,
            c,
            &[
                (0, buf(wgpu::BufferBindingType::Uniform)),
                (1, t(wgpu::TextureViewDimension::D3)),
                (2, sb(true)),
                (3, t(d2)),
            ],
        );
        let lv = bgl(dev, c, &[(0, t(d2)), (1, w(F::R32Float))]);
        let out = FORMATS.map(|f| bgl(dev, c, &[(0, t(d2)), (2, w(f)), (3, sb(true))]));
        let an = bgl(dev, c, &[(4, sb(false)), (5, sb(false))]);
        let ov = bgl(
            dev,
            V::FRAGMENT,
            &[
                (
                    0,
                    T::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: d2,
                        multisampled: false,
                    },
                ),
                (1, T::Sampler(wgpu::SamplerBindingType::Filtering)),
                (2, buf(wgpu::BufferBindingType::Uniform)),
            ],
        );
        let code = include_str!("fx.wgsl");
        let module = FORMATS.map(|f| {
            dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fx"),
                source: wgpu::ShaderSource::Wgsl(
                    code.replace(
                        "OUTFMT",
                        if f == F::Rgba8Unorm {
                            "rgba8unorm"
                        } else {
                            "rgba16float"
                        },
                    )
                    .into(),
                ),
            })
        });
        let layout = |g: &[Option<&wgpu::BindGroupLayout>]| {
            dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: g,
                immediate_size: 0,
            })
        };
        // every shader writes its workgroup memory before reading it
        let pipe = |m: &wgpu::ShaderModule, entry, l: &wgpu::PipelineLayout, ov: f64| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(l),
                module: m,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("OV", ov)],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        };
        let pl = layout(&[Some(&src), Some(&st), Some(&lv)]);
        let al = layout(&[Some(&src), Some(&st), Some(&an)]);
        let om = dev.create_shader_module(wgpu::include_wgsl!("over.wgsl"));
        let ol = layout(&[Some(&ov)]);
        let over = dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fx overlay"),
            layout: Some(&ol),
            vertex: wgpu::VertexState {
                module: &om,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &om,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: LAYER,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let u = wgpu::TextureUsages::TEXTURE_BINDING;
        let one = [1, 1, 1];
        let d2t = wgpu::TextureDimension::D2;
        Self {
            main: [0, 1].map(|i| {
                let l = layout(&[Some(&src), Some(&st), Some(&out[i])]);
                [0.0, 1.0].map(|o| pipe(&module[i], "main", &l, o))
            }),
            down0: pipe(&module[0], "down0", &pl, 0.0),
            down: pipe(&module[0], "down", &pl, 0.0),
            up: pipe(&module[0], "up", &pl, 0.0),
            stat: pipe(&module[0], "stat", &al, 0.0),
            solve: pipe(&module[0], "solve", &al, 0.0),
            over,
            smp: dev.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 16,
                ..Default::default()
            }),
            nil: [
                view(&tex(dev, one, d2t, F::R32Float, u, 1)),
                view(&tex(
                    dev,
                    one,
                    wgpu::TextureDimension::D3,
                    F::Rgba32Float,
                    u,
                    1,
                )),
                view(&tex(dev, one, d2t, LAYER, u, 1)),
            ],
            dev: dev.clone(),
            q: q.clone(),
            src,
            st,
            lv,
            out,
            an,
            ov,
        }
    }
}

/// A source and destination bound for one `Fx`. `src`: same size, any float-sampled colour format with
/// `TEXTURE_BINDING` (not an sRGB format: values are read as stored). `dst`: one of `FORMATS` with `STORAGE_BINDING`.
/// They may be the same texture (in place, e.g. `export::Fx::tex`), which then also needs `COPY_SRC`: the `Io` keeps a
/// copy of the frame for the passes to read, refreshed by each run (one frame copy, what a caller's own second texture
/// and copy back would cost).
pub struct Io {
    g0: wgpu::BindGroup,
    g2: wgpu::BindGroup,
    /// In place: the frame and its copy.
    cp: Option<(wgpu::Texture, wgpu::Texture)>,
    /// `FORMATS` index of the destination.
    f: usize,
}

/// Effect state for frames of one size: parameters, LUTs, the clarity pyramid (about 2.7 bytes per pixel, e.g. 44 MB
/// at 5760 × 2880), the automatic corrections' statistics and state (under 1 MB), and once overlays are drawn their
/// layer (4 bytes per pixel). Parameters and LUTs take effect at the next submit, so frames graded differently in one
/// submit need one `Fx` each.
///
/// The automatic corrections (`Params::water`, `Params::auto`) estimate from statistics of each run's frame: a grid of
/// samples about 512 across (step = ⌈max(width, height) / 512⌉ pixels, at the cell centres) of the source after the
/// input transform, weighted by solid angle on equirects (cos latitude), in 64 bins of encoded luma holding each bin's
/// weight, mean linear colour (channels clamped to 0…1), mean r·g and mean HSV saturation. The estimate is smoothed
/// over time (`Fx::time`).
pub struct Fx {
    cx: Ctx,
    w: u32,
    h: u32,
    pano: bool,
    par: Par,
    ub: wgpu::Buffer,
    /// 3D look LUT texture (None: the placeholder).
    l3: Option<wgpu::Texture>,
    /// 1D LUT entries: input transform, look.
    t1: [Vec<[f32; 4]>; 2],
    tb: wgpu::Buffer,
    g1: wgpu::BindGroup,
    /// Pyramid passes in order (`down0`, `down` per level, `up` per level): bind group, output size.
    lv: Vec<(wgpu::BindGroup, [u32; 2])>,
    /// Pyramid levels.
    n: usize,
    base: wgpu::TextureView,
    /// Automatic corrections' state, f32 (`fx.wgsl` `solve`): 0…14 the smoothed estimate (k, v, g of the underwater
    /// step; g_a, β, ev, contrast, vibrance, highlights of the auto step), 16…25 what `main` applies (k, gains,
    /// offsets, contrast, vibrance, highlights), 32…415 the last frame's statistics summed (64 bins × weight, Σr, Σg,
    /// Σb, Σr·g, Σs; weights × 64, values × 64·262143).
    kb: wgpu::Buffer,
    /// Statistics group (per-workgroup sums, state) and `stat` workgroups.
    an: wgpu::BindGroup,
    sg: [u32; 2],
    /// Overlay layer, made at the first run with overlays.
    lay: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Frame time of the last `time` call.
    t: Option<f64>,
    /// A correction was turned on: the next `time` call keeps the run on this frame's own estimate.
    snap: bool,
}

/// Group 1: parameters, 3D look (or the placeholder), 1D tables, overlay layer (or the placeholder).
fn state(
    cx: &Ctx,
    ub: &wgpu::Buffer,
    l3: Option<&wgpu::Texture>,
    tb: &wgpu::Buffer,
    lay: Option<&wgpu::TextureView>,
) -> wgpu::BindGroup {
    use wgpu::BindingResource as R;
    let v = l3.map(view);
    bind(
        &cx.dev,
        &cx.st,
        &[
            (0, ub.as_entire_binding()),
            (1, R::TextureView(v.as_ref().unwrap_or(&cx.nil[1]))),
            (2, tb.as_entire_binding()),
            (3, R::TextureView(lay.unwrap_or(&cx.nil[2]))),
        ],
    )
}

/// The `Par` slot of each 1D table's domain (`imin`, `lmin`; max follows).
const T1: [usize; 2] = [6, 8];

/// 1D LUT entries as `fx.wgsl` `T` holds them.
fn entries(l: &Lut) -> Vec<[f32; 4]> {
    l.v.iter().map(|c| [c[0], c[1], c[2], 0.0]).collect()
}

impl Fx {
    /// For `w` × `h` frames; `pano`: equirect, so neighbourhoods wrap across longitude and statistics weigh by solid
    /// angle.
    pub fn new(cx: &Ctx, w: u32, h: u32, pano: bool) -> Self {
        let d = &cx.dev;
        let n = ((w as f32 / 180.0).log2().round() as i32).clamp(1, 8) as usize;
        let u = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let mut s = vec![[w, h]];
        for k in 0..n {
            s.push(s[k].map(|v| v.div_ceil(2)));
        }
        let mk = |s: [u32; 2]| {
            view(&tex(
                d,
                [s[0], s[1], 1],
                wgpu::TextureDimension::D2,
                wgpu::TextureFormat::R32Float,
                u,
                1,
            ))
        };
        // p[k]: down level k + 1; q[k]: its upsampled base (q[n - 1] = p[n - 1], the coarsest)
        let p: Vec<_> = (1..=n).map(|k| mk(s[k])).collect();
        let q: Vec<_> = (1..n).map(|k| mk(s[k])).collect();
        let g = |a: &wgpu::TextureView, b: &wgpu::TextureView| {
            use wgpu::BindingResource as R;
            bind(d, &cx.lv, &[(0, R::TextureView(a)), (1, R::TextureView(b))])
        };
        let mut lv = vec![(g(&cx.nil[0], &p[0]), s[1])];
        for k in 1..n {
            lv.push((g(&p[k - 1], &p[k]), s[k + 1]));
        }
        for k in (0..n - 1).rev() {
            lv.push((
                g(if k == n - 2 { &p[n - 1] } else { &q[k + 1] }, &q[k]),
                s[k + 1],
            ));
        }
        let ub = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx"),
            size: size_of::<Par>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tb = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx"),
            contents: &[0; 16],
            usage: wgpu::BufferUsages::STORAGE,
        });
        // statistics grid: step, columns, rows; its workgroups
        let st = w.max(h).div_ceil(512).max(1);
        let (gx, gy) = (w.div_ceil(st), h.div_ceil(st));
        let sg = [gx.div_ceil(16), gy.div_ceil(16)];
        let buf = |n: u64| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fx stats"),
                size: 4 * n,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let pb = buf(NS * (sg[0] * sg[1]) as u64);
        let kb = buf(32 + NS);
        let an = bind(
            d,
            &cx.an,
            &[(4, pb.as_entire_binding()), (5, kb.as_entire_binding())],
        );
        let mut par = [[0.0; 4]; 16];
        par[13][0] = 1.0;
        par[15] = [st, gx, gy, sg[0]].map(f32::from_bits);
        let mut fx = Self {
            g1: state(cx, &ub, None, &tb, None),
            cx: cx.clone(),
            w,
            h,
            pano,
            par,
            ub,
            l3: None,
            t1: [vec![], vec![]],
            tb,
            lv,
            base: if n > 1 { q[0].clone() } else { p[0].clone() },
            n,
            kb,
            an,
            sg,
            lay: None,
            t: None,
            snap: false,
        };
        fx.set(&Params::default());
        fx
    }

    /// Sets the parameters (written to the GPU now, used by the next submit).
    pub fn set(&mut self, p: &Params) {
        let c = |v: f32| v.clamp(-1.0, 1.0);
        let u = |v: f32| v.clamp(0.0, 1.0);
        let br = c(p.brilliance);
        let (hl, sd, cl) = (
            c(p.highlights - 0.5 * br),
            c(p.shadows + 0.5 * br),
            c(p.clarity + 0.3 * br),
        );
        let m = wb(c(p.temperature), c(p.tint)) * 2f64.powf(p.exposure.clamp(-5.0, 5.0) as f64);
        for r in 0..3 {
            let v = m.row(r).as_vec3();
            self.par[r] = [v.x, v.y, v.z, 0.0];
        }
        self.par[3] = [c(p.contrast), 0.25 * c(p.whites), 0.15 * c(p.blacks), hl];
        self.par[4] = [1.25 * sd, c(p.saturation), c(p.vibrance), u(p.mix)];
        self.par[5] = [0.8 * cl, 2.0 * u(p.sharpen), u(p.purple), u(p.green)];
        let (wa, au) = (u(p.water), u(p.auto));
        self.par[12] = [u(p.blue), (-0.6 * c(p.brightness)).exp2(), wa, au];
        let fl = self.pano as u32
            | ((cl != 0.0) as u32) << 1
            | ((u(p.purple) > 0.0 || u(p.green) > 0.0 || u(p.blue) > 0.0) as u32) << 2
            | ((u(p.sharpen) > 0.0) as u32) << 3
            | ((wa > 0.0) as u32) << 4
            | ((au > 0.0) as u32) << 5;
        // a correction turned on, or the underwater one under auto changed: start from this frame's estimate
        if (fl ^ self.par[14][2].to_bits()) & 48 != 0 && fl & 48 != 0 {
            self.par[13][0] = 1.0;
            self.snap = true;
        }
        let r = (p.reach.round() as u32).clamp(1, 12);
        self.par[14] = [self.w, self.h, fl, r].map(f32::from_bits);
        self.cx
            .q
            .write_buffer(&self.ub, 0, bytemuck::cast_slice(&self.par));
    }

    /// The frame time (s) of the next runs, for the automatic corrections: each run moves their estimate toward its
    /// frame's own by 1 − e^(−Δt / 0.5 s), Δt since the previous call (0, the same frame again as in a paused redraw,
    /// keeps it). The first call, a step back or ahead by more than 1 s (a seek), and turning a correction on take the
    /// frame's own estimate. Without calls (stills) every run takes its frame's own. Every run moves the estimate, so
    /// frames drawn twice (two views) adapt twice as fast.
    pub fn time(&mut self, t: f64) {
        let d = self.t.map_or(-1.0, |p| t - p);
        self.t = Some(t);
        self.par[13][0] = if (0.0..=1.0).contains(&d) && !self.snap {
            1.0 - (-d / TAU).exp()
        } else {
            1.0
        } as f32;
        self.snap = false;
        self.cx
            .q
            .write_buffer(&self.ub, 0, bytemuck::cast_slice(&self.par));
    }

    /// The look: a parsed `.cube` (1D and / or 3D table) or one of ours (`Look::cube`, `Grade::cube`); None = off.
    /// Its intensity is `Params::mix`.
    pub fn look(&mut self, c: Option<&Cube>) {
        let d1 = c.and_then(|c| c.d1.as_ref());
        self.table(1, d1);
        self.par[10] = [0.0; 4];
        self.l3 = None;
        if let Some(l) = c.and_then(|c| c.d3.as_ref()) {
            self.par[10] = [l.min[0], l.min[1], l.min[2], l.n as f32];
            self.par[11] = [l.max[0], l.max[1], l.max[2], 0.0];
            let t = tex(
                &self.cx.dev,
                [l.n; 3],
                wgpu::TextureDimension::D3,
                wgpu::TextureFormat::Rgba32Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                1,
            );
            self.cx.q.write_texture(
                t.as_image_copy(),
                bytemuck::cast_slice(&entries(l)),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16 * l.n),
                    rows_per_image: Some(l.n),
                },
                t.size(),
            );
            self.l3 = Some(t);
        }
        self.upload();
    }

    /// The input transform: a 1D LUT applied to the source before everything else, e.g. `Log::rec709`; None = off.
    pub fn input(&mut self, t: Option<&Lut>) {
        self.table(0, t);
        self.upload();
    }

    fn table(&mut self, k: usize, t: Option<&Lut>) {
        assert!(
            t.is_none_or(|t| t.v.len() == t.n as usize),
            "a 1D LUT (n entries)"
        );
        self.t1[k] = t.map(entries).unwrap_or_default();
        self.par[T1[k]] = t.map_or([0.0; 4], |t| [t.min[0], t.min[1], t.min[2], t.n as f32]);
        self.par[T1[k] + 1] = t.map_or([0.0; 4], |t| [t.max[0], t.max[1], t.max[2], 0.0]);
    }

    /// Re-packs the 1D tables (input first) and rebinds the state.
    fn upload(&mut self) {
        let mut v = self.t1.concat();
        self.par[T1[1] + 1][3] = self.t1[0].len() as f32;
        v.push([0.0; 4]);
        self.tb = self
            .cx
            .dev
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fx"),
                contents: bytemuck::cast_slice(&v),
                usage: wgpu::BufferUsages::STORAGE,
            });
        self.cx
            .q
            .write_buffer(&self.ub, 0, bytemuck::cast_slice(&self.par));
        self.g1 = state(
            &self.cx,
            &self.ub,
            self.l3.as_ref(),
            &self.tb,
            self.lay.as_ref().map(|l| &l.1),
        );
    }

    /// Binds a source and a destination (`Io`'s rules); bind once per texture pair, not per frame.
    pub fn io(&self, src: &wgpu::Texture, dst: &wgpu::Texture) -> Io {
        use wgpu::BindingResource as R;
        assert!(
            [src.width(), src.height(), dst.width(), dst.height()]
                == [self.w, self.h, self.w, self.h],
            "fx frames are {}x{}",
            self.w,
            self.h
        );
        let f = FORMATS
            .iter()
            .position(|&f| f == dst.format())
            .expect("fx output: Rgba8Unorm or Rgba16Float");
        let d = &self.cx.dev;
        let cp = (src == dst).then(|| {
            let t = tex(
                d,
                [self.w, self.h, 1],
                wgpu::TextureDimension::D2,
                src.format(),
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                1,
            );
            (src.clone(), t)
        });
        Io {
            g0: bind(
                d,
                &self.cx.src,
                &[(0, R::TextureView(&view(cp.as_ref().map_or(src, |c| &c.1))))],
            ),
            g2: bind(
                d,
                &self.cx.out[f],
                &[
                    (0, R::TextureView(&self.base)),
                    (2, R::TextureView(&view(dst))),
                    (3, self.kb.as_entire_binding()),
                ],
            ),
            cp,
            f,
        }
    }

    /// Records the effects from `io`'s source into its destination with `ov` over it in order. Overlay placements
    /// are written now (an overlay drawn into frames of different sizes in one submit needs one `Overlay` each).
    pub fn run(&mut self, enc: &mut wgpu::CommandEncoder, io: &Io, ov: &[&Overlay]) {
        if let Some((s, t)) = &io.cp {
            enc.copy_texture_to_texture(s.as_image_copy(), t.as_image_copy(), s.size());
        }
        if !ov.is_empty() {
            if self.lay.is_none() {
                let t = tex(
                    &self.cx.dev,
                    [self.w, self.h, 1],
                    wgpu::TextureDimension::D2,
                    LAYER,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    1,
                );
                self.lay = Some((t.clone(), view(&t)));
                self.upload();
            }
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fx overlays"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.lay.as_ref().unwrap().1,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            p.set_pipeline(&self.cx.over);
            for o in ov {
                let (u, r) = o.uniform(self.w, self.h);
                self.cx.q.write_buffer(&o.ub, 0, bytemuck::cast_slice(&u));
                if r[2] > 0 && r[3] > 0 {
                    p.set_scissor_rect(r[0], r[1], r[2], r[3]);
                    p.set_bind_group(0, &o.bg, &[]);
                    p.draw(0..3, 0..1);
                }
            }
        }
        let cx = &self.cx;
        let fl = self.par[14][2].to_bits();
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx"),
            timestamp_writes: None,
        });
        p.set_bind_group(0, &io.g0, &[]);
        p.set_bind_group(1, &self.g1, &[]);
        if fl & 2 != 0 {
            for (k, (g, s)) in self.lv.iter().enumerate() {
                p.set_pipeline(match k {
                    0 => &cx.down0,
                    k if k < self.n => &cx.down,
                    _ => &cx.up,
                });
                p.set_bind_group(2, g, &[]);
                p.dispatch_workgroups(s[0].div_ceil(16), s[1].div_ceil(16), 1);
            }
        }
        if fl & 48 != 0 {
            p.set_bind_group(2, &self.an, &[]);
            p.set_pipeline(&cx.stat);
            p.dispatch_workgroups(self.sg[0], self.sg[1], 1);
            p.set_pipeline(&cx.solve);
            p.dispatch_workgroups(1, 1, 1);
        }
        p.set_pipeline(&cx.main[io.f][!ov.is_empty() as usize]);
        p.set_bind_group(2, &io.g2, &[]);
        p.dispatch_workgroups(self.w.div_ceil(16), self.h.div_ceil(16), 1);
    }
}
