//! Non-local means for stills from video (frame grabs): the vendor denoises them, video frames pass untouched (the
//! camera denoises its recording). In place on an `Rgba8Unorm` frame (grabs are 8-bit; the tiles hold RGB packed in
//! 8 bits), recorded into the caller's encoder.
//!
//! Each pixel becomes the mean of the pixels within ±7 px (15 × 15), weighted by exp(−D / h²), D the mean squared
//! luma difference of their 7 × 7 patches; RGB is averaged with those weights (a separate, stronger chroma filter
//! measured within 0.2 dB of it), the pixel itself weighted as its best neighbour. h follows the frame's noise: g² =
//! the 10th percentile, over a grid every 4 px of mid-tone patches (luma 0.15…0.85), of the mean over a 5 × 5 patch of
//! the squared luma differences 4 px away, h² = k²·max(g² − g0², 0), k = 4·strength, g0 = 1.5/255. Day and indoor
//! footage measures g = 0.8…1.4/255 and stays bit-identical at any strength; night footage 2.7…4.9/255 (5760 and 7680
//! wide; the measurements: TODO.md "Frame grab denoise").

/// h's scale per strength and the noise level below which nothing is filtered (the module's docs).
const K: f32 = 4.0;
const G0: f32 = 1.5 / 255.0;

pub struct Nlm {
    /// the frame's copy the passes read
    cp: wgpu::Texture,
    tex: wgpu::Texture,
    hg: wgpu::Buffer,
    bg: wgpu::BindGroup,
    /// stat, solve, main
    p: [wgpu::ComputePipeline; 3],
}

impl Nlm {
    /// For the frame `tex` (`Rgba8Unorm`; TEXTURE_BINDING, STORAGE_BINDING, COPY_SRC) at `strength` 0…1; `pano`: an
    /// equirect (x wraps, the statistic skips the poles).
    pub fn new(dev: &wgpu::Device, tex: &wgpu::Texture, pano: bool, strength: f32) -> Self {
        use wgpu::{BindingType as T, BufferBindingType as B};
        assert_eq!(
            tex.format(),
            wgpu::TextureFormat::Rgba8Unorm,
            "Nlm frames are Rgba8Unorm"
        );
        let (w, h) = (tex.width(), tex.height());
        let cp = crate::tex(
            dev,
            [w, h, 1],
            wgpu::TextureDimension::D2,
            tex.format(),
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            1,
        );
        let buf = |ty| T::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let l = crate::bgl(
            dev,
            wgpu::ShaderStages::COMPUTE,
            &[
                (
                    0,
                    T::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                (
                    1,
                    T::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
                (2, buf(B::Storage { read_only: false })),
                (3, buf(B::Uniform)),
            ],
        );
        let hg = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nlm"),
            size: 4 * 257,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k = K * strength.clamp(0.0, 1.0);
        let mut u = [0u8; 32];
        u[..16].copy_from_slice(bytemuck::cast_slice(&[w, h, pano as u32, 0]));
        u[16..].copy_from_slice(bytemuck::cast_slice(&[k * k, G0 * G0, 0.0, 0.0]));
        let ub = wgpu::util::DeviceExt::create_buffer_init(
            dev,
            &wgpu::util::BufferInitDescriptor {
                label: Some("nlm"),
                contents: &u,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        use wgpu::BindingResource as R;
        let bg = crate::bind(
            dev,
            &l,
            &[
                (0, R::TextureView(&crate::view(&cp))),
                (1, R::TextureView(&crate::view(tex))),
                (2, hg.as_entire_binding()),
                (3, ub.as_entire_binding()),
            ],
        );
        let m = dev.create_shader_module(wgpu::include_wgsl!("nlm.wgsl"));
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&l)],
            immediate_size: 0,
        });
        let p = ["stat", "solve", "main"].map(|e| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pl),
                module: &m,
                entry_point: Some(e),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    // every shader writes its workgroup memory before reading it
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            })
        });
        Nlm {
            cp,
            tex: tex.clone(),
            hg,
            bg,
            p,
        }
    }

    /// Records the filter of the frame as it is when the encoder runs.
    pub fn run(&self, enc: &mut wgpu::CommandEncoder) {
        let (w, h) = (self.tex.width(), self.tex.height());
        enc.copy_texture_to_texture(
            self.tex.as_image_copy(),
            self.cp.as_image_copy(),
            self.tex.size(),
        );
        enc.clear_buffer(&self.hg, 0, None);
        let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nlm"),
            timestamp_writes: None,
        });
        p.set_bind_group(0, &self.bg, &[]);
        for (k, n) in [
            [w.div_ceil(64), h.div_ceil(64)],
            [1, 1],
            [w.div_ceil(16), h.div_ceil(16)],
        ]
        .into_iter()
        .enumerate()
        {
            p.set_pipeline(&self.p[k]);
            p.dispatch_workgroups(n[0], n[1], 1);
        }
    }
}
