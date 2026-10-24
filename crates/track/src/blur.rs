//! Blurs privacy regions ([`crate::Region`]) in an equirect before encoding: a render pass from `src` into `dst`, the
//! same size and in the stitcher's layout; `Rgba8Unorm` or `Rgba16Float` (any filterable colour format: the pipeline
//! is made for `dst`'s). Each region is an ellipse on the sphere (its box's angular size, upright), blurred over a
//! disc of a third of its smaller half extent and feathered over its last fifth.

use crate::Region;
use glam::Vec3;
use std::f32::consts::{PI, TAU};

/// `blur.wgsl` `Reg`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Reg {
    f: [f32; 4],
    r: [f32; 4],
    u: [f32; 4],
    e: [f32; 4],
    q: [f32; 4],
}

pub struct Blur {
    pipe: [wgpu::RenderPipeline; 2],
    bgl: wgpu::BindGroupLayout,
    smp: wgpu::Sampler,
    buf: wgpu::Buffer,
    regs: Vec<Reg>,
}

impl Blur {
    pub fn new(dev: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let e = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty,
            count: None,
        };
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                e(
                    0,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                e(
                    1,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                e(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let m = dev.create_shader_module(wgpu::include_wgsl!("blur.wgsl"));
        let pipe = |vs, fs, topology, blend| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blur"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &m,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &m,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        use wgpu::{BlendComponent as C, BlendFactor as F, BlendOperation as O};
        // premultiplied over, the destination's alpha kept
        let over = wgpu::BlendState {
            color: C {
                src_factor: F::One,
                dst_factor: F::OneMinusSrcAlpha,
                operation: O::Add,
            },
            alpha: C {
                src_factor: F::Zero,
                dst_factor: F::One,
                operation: O::Add,
            },
        };
        Self {
            pipe: [
                pipe("full", "copy", wgpu::PrimitiveTopology::TriangleList, None),
                pipe(
                    "quad",
                    "blur",
                    wgpu::PrimitiveTopology::TriangleStrip,
                    Some(over),
                ),
            ],
            bgl,
            smp: dev.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            buf: Self::buffer(dev, 64),
            regs: vec![],
        }
    }

    fn buffer(dev: &wgpu::Device, n: usize) -> wgpu::Buffer {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * size_of::<Reg>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Records `dst` = `src` with regions `r` blurred into `enc`; `r` in `src`'s frame (`Hide`: found on this frame
    /// stitched with the same pose). The regions go to the GPU by `q.write_buffer`, which lands before the next submit:
    /// one `run` per submit.
    pub fn run(
        &mut self,
        dev: &wgpu::Device,
        q: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        r: &[Region],
    ) {
        let s = src.texture().size();
        let (w, h) = (s.width as f32, s.height as f32);
        self.regs.clear();
        for x in r {
            let c = crate::Cam::new(x.dir, Vec3::Z, crate::tan(x.size) / 2.0);
            let v = |v: Vec3, k: f32| [v.x, v.y, v.z, k];
            // the bound of the feathered ellipse: a cap of radius `a` around the centre, in columns and rows
            let a = (1.25 * c.ext.length()).atan();
            let (t, p) = (
                (-c.f.y).atan2(c.f.x).rem_euclid(TAU),
                c.f.z.clamp(-1.0, 1.0).acos(),
            );
            let row = |p: f32| (p / PI * (h - 1.0) + 0.5).clamp(0.0, h);
            let (y0, y1) = (row(p - a - 0.01), row(p + a + 0.01) + 1.0);
            let dt = if p - a <= 0.0 || p + a >= PI {
                PI
            } else {
                ((a.sin() / p.sin()).min(1.0)).asin() + 0.01
            };
            let col = |t: f32| t / TAU * w;
            let reg = |x0: f32, x1: f32| Reg {
                f: v(c.f, 0.0),
                r: v(c.r, 0.0),
                u: v(c.u, 0.0),
                e: [c.ext.x, c.ext.y, c.ext.min_element() / 3.0, 0.0],
                q: [x0, y0.min(h), x1, y1.min(h)],
            };
            if dt >= PI {
                self.regs.push(reg(0.0, w));
                continue;
            }
            let (x0, x1) = (col(t - dt) - 1.0, col(t + dt) + 1.0);
            self.regs.push(reg(x0, x1));
            // across the seam: the same region again a turn over
            if x0 < 0.0 {
                self.regs.push(reg(x0 + w, x1 + w));
            } else if x1 > w {
                self.regs.push(reg(x0 - w, x1 - w));
            }
        }
        if self.regs.len() * size_of::<Reg>() > self.buf.size() as usize {
            self.buf = Self::buffer(dev, self.regs.len().next_power_of_two());
        }
        q.write_buffer(&self.buf, 0, bytemuck::cast_slice(&self.regs));
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.smp),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf.as_entire_binding(),
                },
            ],
        });
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blur"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        rp.set_bind_group(0, &bg, &[]);
        rp.set_pipeline(&self.pipe[0]);
        rp.draw(0..3, 0..1);
        if !self.regs.is_empty() {
            rp.set_pipeline(&self.pipe[1]);
            rp.draw(0..4, 0..self.regs.len() as u32);
        }
    }
}
