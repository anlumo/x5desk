//! AI tracking and framing for 360 video with open models on the app's shared wgpu device (PLAN.md group 11):
//! VitTrack (OpenCV zoo) follows one subject, YOLOX-nano (Megvii) detects people and objects; for the privacy blur
//! ([`Hide`], [`Blur`]) YuNet (OpenCV zoo) finds faces and a YOLOv9-t plate detector (open-image-models) finds licence
//! plates on crops around YOLOX's vehicles; people are told apart by LOMO (Liao et al.), a hand-made appearance
//! feature with no model, so the tracker finds its subject again; [`Scorer`] picks highlights and [`timeline`] cuts
//! them into an editor timeline. The networks are Apache-2.0 or MIT (`models/README`), but for the plate detector: its
//! repository is MIT, its weights have no licence of their own and were trained with GPL-3.0 code
//! (`models/plate-detector-licensing.md`). They are compiled in by burn-onnx (`build.rs`) and run by burn.
//!
//! Everything happens on the sphere: a network only ever sees a perspective crop of the stitched equirect, centred
//! where it should look and written straight into its input tensor by `crop.wgsl`, so seams and poles never reach it.
//! Directions are unit vectors in the equirect's frame (stitch layout: centre −X, Z up); [`yaw_pitch`] turns one into
//! `stitch::View` angles. Feed the equirect stitched with the pose the player shows, and those angles are the view
//! onto the subject.

mod blur;
mod det;
mod frame;
mod high;
mod plates;
mod privacy;
mod reid;
#[cfg(test)]
mod tests;
mod vit;

pub use blur::Blur;
pub use det::{CLASSES, Det, SIDE, ring, sphere};
pub use frame::{Framer, LOW, Shot};
pub use high::{BIN, Highlight, Scorer, Sig, timeline};
pub use plates::{VEHICLES, plate_crops};
pub use privacy::{Hide, Region, face_tiles};
pub use reid::{DIM, Emb, Gallery, sim};
pub use vit::{Sample, Seen, State, Subject, Tracker, smooth};

use burn::backend::Cube;
use burn::cubecl::wgpu::{AutoCompiler, WgpuResource, WgpuServer};
use burn::tensor::{DType, Device, Tensor};
use burn_cubecl::CubeBackend;
use glam::{Mat3, Vec2, Vec3};
use std::time::{Duration, Instant};

/// A perspective crop: forward, right, up (orthonormal, equirect frame) and the tangent-plane half extents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cam {
    pub f: Vec3,
    pub r: Vec3,
    pub u: Vec3,
    pub ext: Vec2,
}

impl Cam {
    /// Looking along `f`, `up` as close to up as it can be.
    pub fn new(f: Vec3, up: Vec3, ext: Vec2) -> Self {
        let f = f.normalize();
        let r = f
            .cross(up)
            .try_normalize()
            .unwrap_or_else(|| f.any_orthonormal_vector());
        Self {
            f,
            r,
            u: r.cross(f),
            ext,
        }
    }

    /// What a `stitch::Preview` shows for `View { yaw, pitch, roll, fov }` (perspective, Panini 0) on a target of
    /// `aspect` = width / height.
    pub fn view(yaw: f32, pitch: f32, roll: f32, fov: f32, aspect: f32) -> Self {
        let r = f32::to_radians;
        let m = Mat3::from_cols(Vec3::Y, Vec3::Z, Vec3::NEG_X)
            * Mat3::from_rotation_y(r(yaw))
            * Mat3::from_rotation_x(-r(pitch))
            * Mat3::from_rotation_z(r(roll));
        let t = r(fov / 2.0).tan();
        Self {
            f: m.z_axis,
            r: m.x_axis,
            u: m.y_axis,
            ext: Vec2::new(t, t / aspect),
        }
    }

    /// The ray through image point `p` (−1..1, y down).
    pub fn ray(&self, p: Vec2) -> Vec3 {
        (self.f + p.x * self.ext.x * self.r - p.y * self.ext.y * self.u).normalize()
    }

    /// The image point (−1..1, y down) of ray `w`; None behind the camera.
    pub fn px(&self, w: Vec3) -> Option<Vec2> {
        let z = w.dot(self.f);
        (z > 1e-6).then(|| {
            Vec2::new(
                w.dot(self.r) / (z * self.ext.x),
                -w.dot(self.u) / (z * self.ext.y),
            )
        })
    }

    /// Centre ray and angular width, height (degrees, through the centre) of box `b` = (x0, y0, x1, y1) in −1..1.
    pub fn boxed(&self, b: [f32; 4]) -> (Vec3, Vec2) {
        let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
        let a = |x0, y0, x1, y1| {
            self.ray(Vec2::new(x0, y0))
                .angle_between(self.ray(Vec2::new(x1, y1)))
                .to_degrees()
        };
        (
            self.ray(Vec2::new(cx, cy)),
            Vec2::new(a(b[0], cy, b[2], cy), a(cx, b[1], cx, b[3])),
        )
    }
}

/// `stitch::View` yaw and pitch (degrees) that look along `d`.
pub fn yaw_pitch(d: Vec3) -> [f32; 2] {
    [
        d.y.atan2(-d.x).to_degrees(),
        d.z.clamp(-1.0, 1.0).asin().to_degrees(),
    ]
}

/// The direction of `stitch::View` yaw and pitch (degrees).
pub fn dir(yaw: f32, pitch: f32) -> Vec3 {
    let (y, p) = (yaw.to_radians(), pitch.to_radians());
    Vec3::new(-p.cos() * y.cos(), p.cos() * y.sin(), p.sin())
}

/// Angular size (degrees) -> width in the tangent plane at the centre, and back.
fn tan(deg: Vec2) -> Vec2 {
    2.0 * Vec2::new(
        (deg.x / 2.0).to_radians().tan(),
        (deg.y / 2.0).to_radians().tan(),
    )
}

fn deg(t: Vec2) -> Vec2 {
    2.0 * Vec2::new(
        (t.x / 2.0).atan().to_degrees(),
        (t.y / 2.0).atan().to_degrees(),
    )
}

/// Intersection over union of two boxes on the sphere (centre, angular size), in the tangent plane at `a`'s centre;
/// with `min` the intersection over the smaller box instead.
pub fn iou(a: (Vec3, Vec2), b: (Vec3, Vec2), min: bool) -> f32 {
    let (sa, sb) = (tan(a.1), tan(b.1));
    let Some(c) = Cam::new(a.0, Vec3::Z, Vec2::ONE).px(b.0) else {
        return 0.0;
    };
    let lo = (-sa / 2.0).max(c - sb / 2.0);
    let hi = (sa / 2.0).min(c + sb / 2.0);
    let i = (hi - lo).max(Vec2::ZERO).element_product();
    let (aa, ab) = (sa.element_product(), sb.element_product());
    i / if min { aa.min(ab) } else { aa + ab - i }.max(1e-12)
}

type Res = cubecl_runtime::storage::ManagedResource<WgpuResource>;

/// A network input whose storage the crop pass writes; holding `r` keeps the allocation in place.
struct In {
    t: Tensor<4>,
    r: Res,
}

impl In {
    fn new(dev: &Device, shape: [usize; 4]) -> Result<Self, String> {
        let t = Tensor::<4>::zeros(shape, dev);
        let f = t
            .clone()
            .try_into_primitive::<Cube>()
            .map_err(|e| format!("input not on the cubecl backend: {e:?}"))?;
        let c = f.client.clone().resolve_tensor_float::<CubeBackend>(f);
        let st = c.meta.strides();
        let [_, a, b, w] = shape;
        let want = [a * b * w, b * w, w, 1];
        if c.dtype != DType::F32 || (0..4).any(|i| st[i] != want[i]) {
            return Err(format!(
                "input {:?} strides {st:?}: crops write contiguous f32 NCHW",
                c.dtype
            ));
        }
        let r = c
            .client
            .get_resource::<WgpuServer<AutoCompiler>>(c.handle.clone())
            .map_err(|e| format!("input storage: {e:?}"))?;
        c.client.flush().map_err(|e| format!("flush: {e:?}"))?;
        Ok(Self { t, r })
    }
}

/// Network input normalisation: channel c = colour[c] · a[c] + b[c] on 0..1 colours, in BGR order if `bgr`.
struct Norm {
    a: [f32; 3],
    b: [f32; 3],
    bgr: bool,
}

/// Colours as they are (thumbnails, re-identification).
const RGB: Norm = Norm {
    a: [1.0; 3],
    b: [0.0; 3],
    bgr: false,
};

/// `crop.wgsl` `Job`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Job {
    f: [f32; 4],
    r: [f32; 4],
    u: [f32; 4],
    e: [f32; 4],
    a: [f32; 4],
    b: [f32; 4],
    o: [u32; 4],
}

impl Job {
    /// `c` as an `s[0]` × `s[1]` crop at element `off` of the crop buffer, from an equirect `pw` pixels wide.
    fn new(c: &Cam, s: [u32; 2], off: usize, n: &Norm, pw: u32) -> Self {
        // equirect pixels per output pixel at the centre (the most): that many taps per axis, at most 4
        let k = ((2.0 * c.ext / Vec2::new(s[0] as f32, s[1] as f32)).max_element() * pw as f32
            / std::f32::consts::TAU)
            .ceil()
            .clamp(1.0, 4.0) as u32;
        let v = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let w = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        Self {
            f: v(c.f),
            r: v(c.r),
            u: v(c.u),
            e: [c.ext.x, c.ext.y, s[0] as f32, s[1] as f32],
            a: w(n.a),
            b: w(n.b),
            o: [off as u32, k, n.bgr as u32, 0],
        }
    }
}

/// A crop pass's buffers: jobs, and the planar output copied into network inputs.
struct Crops {
    jobs: wgpu::Buffer,
    dst: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

impl Crops {
    fn new(dev: &wgpu::Device, l: &wgpu::BindGroupLayout, jobs: usize, floats: usize) -> Self {
        use wgpu::BufferUsages as U;
        let b = |size, usage| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let jobs = b((jobs * size_of::<Job>()) as u64, U::STORAGE | U::COPY_DST);
        let dst = b(4 * floats as u64, U::STORAGE | U::COPY_SRC);
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: l,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: jobs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
            ],
        });
        Self { jobs, dst, bg }
    }
}

/// An equirect bound for crops: `Rgba8Unorm` (or any float format) holding sRGB values in the stitcher's layout, e.g.
/// a `stitch::Stitcher`'s `out`.
pub struct Pano {
    bg: wgpu::BindGroup,
    w: u32,
}

/// Wall time summed while `Ai::timing` is set: crop passes (each waited for), tracker and detector runs (network,
/// read-back, decoding) and re-ID passes (read-back, features), and how many of each.
#[derive(Clone, Copy, Debug, Default)]
pub struct Times {
    pub crop: Duration,
    pub vit: Duration,
    pub det: Duration,
    pub reid: Duration,
    pub n: [u32; 4],
}

/// The networks on the shared device, the crop pipeline and the detector's input.
pub struct Ai {
    pub dev: Device,
    wd: wgpu::Device,
    q: wgpu::Queue,
    pipe: wgpu::ComputePipeline,
    g: [wgpu::BindGroupLayout; 2],
    smp: wgpu::Sampler,
    vit: vit::model::Model,
    det: det::model::Model,
    di: In,
    dc: Crops,
    ri: In,
    rc: Crops,
    /// Face detection, made on first use.
    pv: Option<Box<privacy::Nets>>,
    /// Plate detection, made on first use.
    pl: Option<Box<plates::Nets>>,
    /// Thumbnail crops, made on first use.
    th: Option<(Crops, In)>,
    /// Wait for each crop pass and sum the times into `times` (a GPU sync per crop).
    pub timing: bool,
    pub times: Times,
}

impl Ai {
    /// On the shared device, through `stitch::attach`: every `Ai` (and `Raft`) on `dev` shares its one burn runtime,
    /// so kernels compiled once serve them all, and the device's uncaptured-error handler is replaced only by the
    /// first attach (call `stitch::attach` before installing yours, ARCHITECTURE.md). `ad`, `inst`: what `dev` was
    /// opened from; burn shares the queue, so its submits interleave with the caller's (safe around a decoder's
    /// `acquire` / `release`). The first runs compile and autotune the kernels (seconds, `warm`).
    pub fn new(
        inst: &wgpu::Instance,
        ad: &wgpu::Adapter,
        dev: &wgpu::Device,
        q: &wgpu::Queue,
    ) -> Result<Self, String> {
        let bd = stitch::attach(inst, ad, dev, q)?;
        let e = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let sto = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bgl = |e: &[wgpu::BindGroupLayoutEntry]| {
            dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: e,
            })
        };
        let tex = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let smp = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let g = [
            bgl(&[e(0, tex), e(1, smp)]),
            bgl(&[e(0, sto(true)), e(1, sto(false))]),
        ];
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&g[0]), Some(&g[1])],
            immediate_size: 0,
        });
        let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("crop"),
            layout: Some(&layout),
            module: &dev.create_shader_module(wgpu::include_wgsl!("crop.wgsl")),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let s = SIDE as usize;
        let (rw, rh) = (reid::W as usize, reid::H as usize);
        Ok(Self {
            vit: vit::model::Model::from_embedded(&bd),
            det: det::model::Model::from_embedded(&bd),
            di: In::new(&bd, [det::MAX, 3, s, s])?,
            dc: Crops::new(dev, &g[1], det::MAX, det::MAX * 3 * s * s),
            ri: In::new(&bd, [reid::MAX, 3, rh, rw])?,
            rc: Crops::new(dev, &g[1], reid::MAX, reid::MAX * 3 * rw * rh),
            pv: None,
            pl: None,
            th: None,
            dev: bd,
            wd: dev.clone(),
            q: q.clone(),
            pipe,
            g,
            smp: dev.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            timing: false,
            times: Times::default(),
        })
    }

    /// Binds an equirect for crops; keep it while the texture stays the same.
    pub fn pano(&self, v: &wgpu::TextureView) -> Pano {
        Pano {
            bg: self.wd.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.g[0],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(v),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.smp),
                    },
                ],
            }),
            w: v.texture().width(),
        }
    }

    /// Runs the tracker and the detector at every batch size (1 to 9 tiles) once on a black panorama, so that the
    /// kernels of each shape are compiled and autotuned before the first frame that needs them instead of stalling it
    /// (seconds on the first use of each; 16 to 39 s in all here). Call once after `new`, off the UI thread. Faces
    /// (`faces`) and plates (`plates`) compile on their first call (11 to 17 s here for faces); re-identification and
    /// thumbnails run no network.
    pub fn warm(&mut self) -> Result<(), String> {
        let t = self.wd.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let p = self.pano(&t.create_view(&Default::default()));
        let s = Subject {
            dir: Vec3::NEG_X,
            size: Vec2::splat(10.0),
            cls: None,
        };
        Tracker::new(self, &p, s)?.step(self, &p);
        let c = ring(det::MAX, 90.0, 0.0);
        for n in 1..=det::MAX {
            self.detect(&p, &c[..n], 1.0);
        }
        Ok(())
    }

    /// Renders `jobs` (consecutive in `c`'s buffer) and copies them to the start of `to`'s storage; submitted. The
    /// time it took when `timing` (for `spent`).
    fn crop(&self, c: &Crops, p: &Pano, jobs: &[Job], to: &In) -> Duration {
        let t = Instant::now();
        self.q.write_buffer(&c.jobs, 0, bytemuck::cast_slice(jobs));
        let s = |i: usize| jobs.iter().map(|j| j.e[i] as u32).max().unwrap_or(0);
        let n: u64 = jobs.iter().map(|j| 3 * (j.e[2] * j.e[3]) as u64).sum();
        let mut enc = self.wd.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(&self.pipe);
            cp.set_bind_group(0, &p.bg, &[]);
            cp.set_bind_group(1, &c.bg, &[]);
            cp.dispatch_workgroups(s(2).div_ceil(8), s(3).div_ceil(8), jobs.len() as u32);
        }
        let r = to.r.resource();
        enc.copy_buffer_to_buffer(&c.dst, 0, &r.buffer, r.offset, 4 * n);
        self.q.submit([enc.finish()]);
        if self.timing {
            let _ = self.wd.poll(wgpu::PollType::wait_indefinitely());
        }
        t.elapsed()
    }

    /// Adds to `times`: 0 crop, 1 tracker, 2 detector, 3 re-ID.
    fn spent(&mut self, i: usize, d: Duration) {
        if self.timing {
            *[
                &mut self.times.crop,
                &mut self.times.vit,
                &mut self.times.det,
                &mut self.times.reid,
            ][i] += d;
            self.times.n[i] += 1;
        }
    }
}

/// A tensor's values on the CPU (waits for the GPU).
fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap_or_default()
}
