//! GPU tests: DMA-BUF export, the OBS protocol with real DMA-BUFs, the PipeWire node with a consumer. They render a
//! known pattern on the GPU and check it where a consumer reads it (LINEAR mappings, memfds, or a re-import of a
//! tiled DMA-BUF).
use crate::dmabuf::*;
use ash::vk;
use std::{
    iter,
    os::fd::{AsRawFd, RawFd},
    ptr,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};
use video::Gpu;
use wgpu_hal::api::Vulkan as Vk;

pub(crate) fn gpu() -> &'static Arc<Gpu> {
    static G: OnceLock<Arc<Gpu>> = OnceLock::new();
    G.get_or_init(|| Arc::new(Gpu::new().expect("GPU")))
}

pub(crate) const U: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC);

const WGSL: &str = r#"
struct P { frame: u32, spin: u32, magic: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
// `spin` multiplications per pixel make the frame slow; h stays odd, magic is 0, so alpha stays 1
fn alpha(x: u32, y: u32) -> f32 {
    var h = (x * 65599u + y) | 1u;
    for (var k = 0u; k < p.spin; k++) { h *= 747796405u; }
    return select(1.0, 0.0, h == p.magic);
}
@fragment fn rgba(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let x = u32(q.x); let y = u32(q.y);
    return vec4f(f32(x & 255u) / 255.0, f32(y & 255u) / 255.0, f32(p.frame & 255u) / 255.0, alpha(x, y));
}
@fragment fn luma(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let x = u32(q.x); let y = u32(q.y);
    return vec4f(f32((x + 3u * y + p.frame) & 255u) / 255.0, 0.0, 0.0, 1.0);
}
@fragment fn chroma(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let x = u32(q.x); let y = u32(q.y);
    return vec4f(f32((5u * x + p.frame) & 255u) / 255.0, f32((7u * y + 128u) & 255u) / 255.0, 0.0, 1.0);
}
"#;

/// The expected bytes of pixel (x, y) of plane `k` (NV12: plane 1 is the half-size chroma plane).
pub(crate) fn px(fmt: Fmt, k: usize, x: u32, y: u32, f: u32) -> [u8; 4] {
    let (x8, y8, f8) = (x as u8, y as u8, f as u8);
    match (fmt, k) {
        (Fmt::Rgba, _) => [x8, y8, f8, 255],
        (Fmt::Bgra, _) => [f8, y8, x8, 255],
        (Fmt::Nv12, 0) => [(x + 3 * y + f) as u8, 0, 0, 0],
        (Fmt::Nv12, _) => [(5 * x + f) as u8, (7 * y + 128) as u8, 0, 0],
    }
}

/// Checks frame `f` of the pattern on a grid (and the last row / column); `plane(k)` = (first byte, row pitch).
/// Returns the mismatches.
pub(crate) fn check(
    fmt: Fmt,
    w: u32,
    h: u32,
    f: u32,
    plane: impl Fn(usize) -> (*const u8, usize),
) -> usize {
    let mut bad = 0;
    let planes: &[(u32, u32, usize)] = match fmt {
        Fmt::Nv12 => &[(1, 1, 1), (2, 2, 2)],
        _ => &[(1, 1, 4)],
    };
    for (k, &(sx, sy, bpp)) in planes.iter().enumerate() {
        let (pw, ph) = (w / sx, h / sy);
        let (b, pitch) = plane(k);
        let ys = (0..ph).step_by(7).chain(iter::once(ph - 1));
        for y in ys {
            for x in (0..pw).step_by(5).chain(iter::once(pw - 1)) {
                let e = px(fmt, k, x, y, f);
                let got = unsafe {
                    std::slice::from_raw_parts(b.add(y as usize * pitch + x as usize * bpp), bpp)
                };
                if got != &e[..bpp] {
                    if bad < 4 {
                        eprintln!(
                            "{fmt:?} plane {k} ({x}, {y}) frame {f}: {got:?} != {:?}",
                            &e[..bpp]
                        );
                    }
                    bad += 1;
                }
            }
        }
    }
    bad
}

pub(crate) struct Pat {
    pipe: [wgpu::RenderPipeline; 4],
    bg: wgpu::BindGroup,
    u: wgpu::Buffer,
}

impl Pat {
    pub(crate) fn new(g: &Gpu) -> Pat {
        let d = &g.device;
        let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let u = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u.as_entire_binding(),
            }],
        });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |fs, f: wgpu::TextureFormat| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fs),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &m,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &m,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(f.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        use wgpu::TextureFormat as T;
        Pat {
            pipe: [
                pipe("rgba", T::Rgba8Unorm),
                pipe("rgba", T::Bgra8Unorm),
                pipe("luma", T::R8Unorm),
                pipe("chroma", T::Rg8Unorm),
            ],
            bg,
            u,
        }
    }

    /// Renders frame `f` of the pattern into `t` and submits.
    pub(crate) fn draw(&self, g: &Gpu, t: &wgpu::Texture, f: u32, spin: u32) {
        g.queue
            .write_buffer(&self.u, 0, &[f, spin, 0, 0].map(u32::to_ne_bytes).concat());
        let mut e = g.device.create_command_encoder(&Default::default());
        let passes: &[(usize, wgpu::TextureAspect)] = match t.format() {
            wgpu::TextureFormat::NV12 => &[
                (2, wgpu::TextureAspect::Plane0),
                (3, wgpu::TextureAspect::Plane1),
            ],
            wgpu::TextureFormat::Bgra8Unorm => &[(1, wgpu::TextureAspect::All)],
            _ => &[(0, wgpu::TextureAspect::All)],
        };
        for &(k, aspect) in passes {
            let v = t.create_view(&wgpu::TextureViewDescriptor {
                aspect,
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                ..Default::default()
            });
            let mut p = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &v,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            p.set_pipeline(&self.pipe[k]);
            p.set_bind_group(0, &self.bg, &[]);
            p.draw(0..3, 0..1);
        }
        g.queue.submit([e.finish()]);
    }
}

/// A read-only shared mapping, bracketed by DMA_BUF_IOCTL_SYNC (which waits for the DMA-BUF's write fences).
pub(crate) struct Map {
    pub p: *const u8,
    n: usize,
    fd: RawFd,
}

const DMA_BUF_SYNC: u32 = 0x4008_6200; // _IOW('b', 0, struct dma_buf_sync)

impl Map {
    pub(crate) fn new(fd: RawFd, n: usize, dmabuf: bool) -> Map {
        let p = unsafe { libc::mmap(ptr::null_mut(), n, libc::PROT_READ, libc::MAP_SHARED, fd, 0) };
        assert_ne!(
            p,
            libc::MAP_FAILED,
            "mmap: {}",
            std::io::Error::last_os_error()
        );
        if dmabuf {
            let s = 1u64; // READ | START
            assert_eq!(unsafe { libc::ioctl(fd, DMA_BUF_SYNC as _, &s) }, 0);
        }
        Map {
            p: p.cast(),
            n,
            fd: if dmabuf { fd } else { -1 },
        }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        unsafe {
            if self.fd >= 0 {
                let s = 1u64 | 4; // READ | END
                libc::ioctl(self.fd, DMA_BUF_SYNC as _, &s);
            }
            libc::munmap(self.p as *mut _, self.n);
        }
    }
}

pub(crate) fn readable(fd: RawFd, ms: i32) -> bool {
    let mut p = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe { libc::poll(&mut p, 1, ms) == 1 }
}

pub(crate) fn ino(fd: RawFd) -> u64 {
    let mut s: libc::stat = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::fstat(fd, &mut s) }, 0);
    s.st_ino
}

#[test]
fn probe() {
    let g = gpu();
    for f in [Fmt::Rgba, Fmt::Bgra, Fmt::Nv12] {
        let m = modifiers(g, f, crate::pw::USAGE, 3840, 1920);
        let n = modifiers(g, f, U, 3840, 1920).len();
        println!("{f:?}: {m:x?} ({n} without COPY_DST and STORAGE_BINDING)");
        assert!(m.iter().any(|m| m.0 == LINEAR));
    }
}

/// Re-imports a released image as a consumer would and counts the mismatches against frame `f`.
fn reread(g: &Gpu, img: &Image, f: u32) -> usize {
    let (t, vi) = unsafe { import(g, &shared(img)) };
    let rb = readback(g, &t, vi, img.fam);
    check(img.fmt, img.w, img.h, f, |k| (rb[k].0.as_ptr(), rb[k].1))
}

const CS: &str = r#"
@group(0) @binding(0) var<uniform> f: vec4u;
@group(0) @binding(1) var o: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8) fn cs(@builtin(global_invocation_id) i: vec3u) {
    if (any(i.xy >= textureDimensions(o))) { return; }
    textureStore(o, i.xy, vec4f(vec3f(vec3u(i.x, i.y, f.x) & vec3u(255u)) / 255.0, 1.0));
}
"#;

/// The writers the app has besides render passes, into the camera's usage and the driver's pick (tiled) and
/// LINEAR at the equirect size: a compute shader's storage writes, a copy of an RGBA texture (the stitcher's
/// `out`) and a copy of an NV12 buffer in the stitcher's layout (pitch = width, Y rows then CbCr rows).
#[test]
fn writers() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let (w, h, u) = (3840, 1920, crate::pw::USAGE);
    let d = &g.device;
    let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(CS.into()),
    });
    let cp = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("store"),
        layout: None,
        module: &m,
        entry_point: Some("cs"),
        compilation_options: Default::default(),
        cache: None,
    });
    let ub = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let src = d.create_texture(&wgpu::TextureDescriptor {
        label: Some("pano"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut nv = vec![0u8; (w * h * 3 / 2) as usize];
    let nb = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nv12"),
        size: nv.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for (fmt, how) in [
        (Fmt::Rgba, "storage"),
        (Fmt::Rgba, "texture copy"),
        (Fmt::Nv12, "buffer copy"),
    ] {
        let all: Vec<u64> = modifiers(g, fmt, u, w, h).iter().map(|m| m.0).collect();
        for mods in [&all[..], &[LINEAR]] {
            let img = Image::new(g, fmt, w, h, u, mods).unwrap();
            let f = img.modifier as u32 & 0xff ^ how.len() as u32;
            img.acquire().unwrap();
            let mut e = d.create_command_encoder(&Default::default());
            match how {
                "storage" => {
                    g.queue
                        .write_buffer(&ub, 0, &[f, 0, 0, 0].map(u32::to_ne_bytes).concat());
                    let v = img.tex.create_view(&Default::default());
                    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &cp.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: ub.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&v),
                            },
                        ],
                    });
                    let mut c = e.begin_compute_pass(&Default::default());
                    c.set_pipeline(&cp);
                    c.set_bind_group(0, &bg, &[]);
                    c.dispatch_workgroups(w / 8, h / 8, 1);
                }
                "texture copy" => {
                    pat.draw(g, &src, f, 0);
                    e.copy_texture_to_texture(
                        src.as_image_copy(),
                        img.tex.as_image_copy(),
                        src.size(),
                    );
                }
                _ => {
                    let (y, c) = nv.split_at_mut((w * h) as usize);
                    for (i, b) in y.iter_mut().enumerate() {
                        *b = px(fmt, 0, i as u32 % w, i as u32 / w, f)[0];
                    }
                    for (i, b) in c.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                        let i = i as u32 * 2;
                        *b = px(fmt, 1, i % w / 2, i / w, f)[..2].try_into().unwrap();
                    }
                    g.queue.write_buffer(&nb, 0, &nv);
                    for (aspect, off, s) in [
                        (wgpu::TextureAspect::Plane0, 0, 1),
                        (wgpu::TextureAspect::Plane1, w * h, 2),
                    ] {
                        e.copy_buffer_to_texture(
                            wgpu::TexelCopyBufferInfo {
                                buffer: &nb,
                                layout: wgpu::TexelCopyBufferLayout {
                                    offset: off as u64,
                                    bytes_per_row: Some(w),
                                    rows_per_image: None,
                                },
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture: &img.tex,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect,
                            },
                            wgpu::Extent3d {
                                width: w / s,
                                height: h / s,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                }
            }
            g.queue.submit([e.finish()]);
            let sf = img.release().unwrap();
            assert!(readable(sf.as_raw_fd(), 2000));
            let bad = reread(g, &img, f);
            println!("{fmt:?} {:#x} by {how}: {bad} bad", img.modifier);
            assert_eq!(bad, 0, "{fmt:?} {:#x} by {how}", img.modifier);
        }
    }
}

/// acquire after what wgpu last did without a release in between, in the driver's pick: a copy out (the MemFd
/// path leaves the image in COPY_SRC) and a frame dropped unqueued (COLOR_TARGET); then a frame goes out.
#[test]
fn retrack() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let (w, h, u) = (1920, 1080, crate::pw::USAGE);
    let out = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (w * h * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for fmt in [Fmt::Rgba, Fmt::Nv12] {
        let all: Vec<u64> = modifiers(g, fmt, u, w, h).iter().map(|m| m.0).collect();
        let img = Image::new(g, fmt, w, h, u, &all).unwrap();
        for f in 0..3 {
            img.acquire().unwrap();
            pat.draw(g, &img.tex, f, 0);
            let mut e = g.device.create_command_encoder(&Default::default());
            let aspect = match fmt {
                Fmt::Nv12 => wgpu::TextureAspect::Plane0,
                _ => wgpu::TextureAspect::All,
            };
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    aspect,
                    ..img.tex.as_image_copy()
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &out,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(w * 4),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            g.queue.submit([e.finish()]);
            img.acquire().unwrap();
            pat.draw(g, &img.tex, f + 1, 0);
            img.acquire().unwrap();
            pat.draw(g, &img.tex, f + 2, 0);
            let sf = img.release().unwrap();
            assert!(readable(sf.as_raw_fd(), 2000));
            assert_eq!(reread(g, &img, f + 2), 0, "{fmt:?} {:#x}", img.modifier);
        }
        println!("{fmt:?} {:#x}: 3 rounds right", img.modifier);
    }
}

/// LINEAR export of every format: plane layout, the pattern through a mapping, the sync_file, and the implicit
/// fence on the DMA-BUF while a slow frame renders.
#[test]
fn linear() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let (w, h) = (1920, 1080);
    for fmt in [Fmt::Rgba, Fmt::Bgra, Fmt::Nv12] {
        let img = Image::new(g, fmt, w, h, U, &[LINEAR]).unwrap();
        assert_eq!(img.modifier, LINEAR);
        let bpp = if fmt == Fmt::Nv12 { 1 } else { 4 };
        assert_eq!(img.planes.len(), if fmt == Fmt::Nv12 { 2 } else { 1 });
        assert!(img.planes.iter().all(|p| p.stride >= w * bpp));
        for f in 0..3 {
            img.acquire().unwrap();
            pat.draw(g, &img.tex, f, 0);
            let sf = img.release().unwrap();
            assert!(readable(sf.as_raw_fd(), 2000), "sync_file never signalled");
            let m = Map::new(img.fd.as_raw_fd(), img.size as usize, true);
            let pl = &img.planes;
            assert_eq!(
                check(fmt, w, h, f, |k| unsafe {
                    (m.p.add(pl[k].off as usize), pl[k].stride as usize)
                }),
                0
            );
        }
        println!("{fmt:?} LINEAR planes {:?} size {}", img.planes, img.size);
    }
    // implicit sync with a slow frame: right after release the DMA-BUF is busy and a mapping waits for the frame.
    // On RADV that alone proves little: every submit fences every exported BO (global BO list), so the import is
    // checked on its own on a udmabuf (a memfd DMA-BUF amdgpu never touches): busy exactly until the frame is done.
    let img = Image::new(g, Fmt::Rgba, w, h, U, &[LINEAR]).unwrap();
    let ud = udmabuf();
    for (f, spin) in [(7, 0), (8, 0), (9, 20000)] {
        img.acquire().unwrap();
        let t = Instant::now();
        pat.draw(g, &img.tex, f, spin);
        let sf = img.release().unwrap();
        let busy = !readable(img.fd.as_raw_fd(), 0);
        let pending = !readable(sf.as_raw_fd(), 0);
        let fenced = ud.as_ref().map(|u| {
            assert!(readable(u.as_raw_fd(), 0));
            attach(u.as_raw_fd(), sf.as_raw_fd()).unwrap();
            !readable(u.as_raw_fd(), 0)
        });
        let m = Map::new(img.fd.as_raw_fd(), img.size as usize, true);
        let dt = t.elapsed();
        let p = img.planes[0];
        assert_eq!(
            check(Fmt::Rgba, w, h, f, |_| unsafe {
                (m.p.add(p.off as usize), p.stride as usize)
            }),
            0
        );
        let freed =
            readable(sf.as_raw_fd(), 0) && ud.as_ref().is_none_or(|u| readable(u.as_raw_fd(), 0));
        println!(
            "spin {spin}: after release DMA-BUF busy {busy}, sync_file pending {pending}, udmabuf with it fenced \
             {fenced:?}; mapped after {dt:?}, then sync_file and udmabuf signalled {freed}"
        );
        if spin > 0 {
            assert!(
                busy && pending && fenced != Some(false),
                "the slow frame finished before the check; raise spin"
            );
        }
        assert!(freed);
    }
}

/// A DMA-BUF from a sealed memfd (/dev/udmabuf), or None without access.
fn udmabuf() -> Option<std::os::fd::OwnedFd> {
    use std::os::fd::FromRawFd;
    unsafe {
        let m = libc::memfd_create(
            c"udmabuf".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        let m = std::os::fd::OwnedFd::from_raw_fd(m);
        assert_eq!(libc::ftruncate(m.as_raw_fd(), 4096), 0);
        assert_eq!(
            libc::fcntl(m.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK),
            0
        );
        let d = libc::open(c"/dev/udmabuf".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
        if d < 0 {
            println!("no /dev/udmabuf: {}", std::io::Error::last_os_error());
            return None;
        }
        let d = std::os::fd::OwnedFd::from_raw_fd(d);
        // struct udmabuf_create { u32 memfd, flags; u64 offset, size }, UDMABUF_CREATE = _IOW('u', 0x42, 24 bytes)
        let c: [u32; 6] = [m.as_raw_fd() as u32, 1, 0, 0, 4096, 0];
        let fd = libc::ioctl(d.as_raw_fd(), 0x4018_7542u32 as _, c.as_ptr());
        assert!(
            fd >= 0,
            "UDMABUF_CREATE: {}",
            std::io::Error::last_os_error()
        );
        Some(std::os::fd::OwnedFd::from_raw_fd(fd))
    }
}

/// What a consumer gets for a DMA-BUF frame.
pub(crate) struct Shared {
    fmt: Fmt,
    w: u32,
    h: u32,
    modifier: u64,
    fd: RawFd,
    planes: Vec<Plane>,
}

fn shared(i: &Image) -> Shared {
    Shared {
        fmt: i.fmt,
        w: i.w,
        h: i.h,
        modifier: i.modifier,
        fd: i.fd.as_raw_fd(),
        planes: i.planes.clone(),
    }
}

/// Imports `img`'s DMA-BUF as a new Vulkan image with its modifier and plane layout, as a consumer on another
/// device would, and wraps it for wgpu.
unsafe fn import(g: &Gpu, img: &Shared) -> (wgpu::Texture, vk::Image) {
    unsafe {
        let pl: Vec<_> = img
            .planes
            .iter()
            .map(|p| vk::SubresourceLayout {
                offset: p.off as u64,
                size: 0,
                row_pitch: p.stride as u64,
                array_pitch: 0,
                depth_pitch: 0,
            })
            .collect();
        let dm = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        let mut ex = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(img.modifier)
            .plane_layouts(&pl);
        let mut em = vk::ExternalMemoryImageCreateInfo::default().handle_types(dm);
        let mut fl = vk::ImageFormatListCreateInfo::default().view_formats(img.fmt.views());
        let mut info = vk::ImageCreateInfo::default()
            .flags(img.fmt.flags())
            .image_type(vk::ImageType::TYPE_2D)
            .format(img.fmt.vk())
            .extent(vk::Extent3D {
                width: img.w,
                height: img.h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut ex)
            .push_next(&mut em);
        if img.fmt == Fmt::Nv12 {
            info = info.push_next(&mut fl);
        }
        let vi = g.dev.create_image(&info, None).unwrap();
        let fd = libc::dup(img.fd);
        let mfd = g.mem_fd.as_ref().unwrap();
        let mut fp = vk::MemoryFdPropertiesKHR::default();
        mfd.get_memory_fd_properties(dm, fd, &mut fp).unwrap();
        let req = g.dev.get_image_memory_requirements(vi);
        let t = (0..32)
            .find(|i| req.memory_type_bits & fp.memory_type_bits >> i & 1 == 1)
            .unwrap();
        let mut im = vk::ImportMemoryFdInfoKHR::default().handle_type(dm).fd(fd);
        let mut de = vk::MemoryDedicatedAllocateInfo::default().image(vi);
        let mem = g
            .dev
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(t)
                    .push_next(&mut im)
                    .push_next(&mut de),
                None,
            )
            .unwrap();
        g.dev.bind_image_memory(vi, mem, 0).unwrap();
        let size = wgpu::Extent3d {
            width: img.w,
            height: img.h,
            depth_or_array_layers: 1,
        };
        let hd = g.device.as_hal::<Vk>().unwrap();
        let hal = hd.texture_from_raw(
            vi,
            &wgpu_hal::TextureDescriptor {
                label: Some("import"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: img.fmt.wgpu(),
                usage: wgpu::TextureUses::COPY_SRC,
                memory_flags: wgpu_hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            None,
            wgpu_hal::vulkan::TextureMemory::Dedicated(mem),
        );
        let t = g.device.create_texture_from_hal::<Vk>(
            hal,
            &wgpu::TextureDescriptor {
                label: Some("import"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: img.fmt.wgpu(),
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::TextureUses::RESOURCE,
        );
        (t, vi)
    }
}

/// Copies every plane of `t` to the CPU (rows padded to 256 bytes), after acquiring the imported image `vi`
/// from the foreign queue family in the layout `release` left it in.
fn readback(g: &Gpu, t: &wgpu::Texture, vi: vk::Image, fam: u32) -> Vec<(Vec<u8>, usize)> {
    let mut a = g.device.create_command_encoder(&Default::default());
    let b = vk::ImageMemoryBarrier::default()
        .dst_access_mask(vk::AccessFlags::MEMORY_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(fam)
        .dst_queue_family_index(g.gfx)
        .image(vi)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    unsafe {
        a.as_hal_mut::<Vk, _, _>(|e| {
            g.dev.cmd_pipeline_barrier(
                e.unwrap().raw_handle(),
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[b],
            )
        })
    };
    let mut e = g.device.create_command_encoder(&Default::default());
    let planes: &[(wgpu::TextureAspect, u32, u32)] = match t.format() {
        wgpu::TextureFormat::NV12 => &[
            (wgpu::TextureAspect::Plane0, 1, 1),
            (wgpu::TextureAspect::Plane1, 2, 2),
        ],
        _ => &[(wgpu::TextureAspect::All, 1, 4)],
    };
    let bufs: Vec<_> = planes
        .iter()
        .map(|&(aspect, s, bpp)| {
            let (w, h) = (t.width() / s, t.height() / s);
            let pitch = (w * bpp).next_multiple_of(256);
            let b = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (pitch * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: t,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &b,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(pitch),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            (b, pitch as usize)
        })
        .collect();
    g.queue.submit([a.finish(), e.finish()]);
    bufs.iter()
        .map(|(b, pitch)| {
            b.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
            g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let v = b.get_mapped_range(..).unwrap().to_vec();
            b.unmap();
            (v, *pitch)
        })
        .collect()
}

/// Every modifier the device exports: render, release, re-import the DMA-BUF with the reported modifier and
/// plane layout, read it back, compare.
#[test]
fn tiled() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let (w, h) = (1920, 1080);
    for fmt in [Fmt::Rgba, Fmt::Bgra, Fmt::Nv12] {
        let all = modifiers(g, fmt, U, w, h);
        let pick = Image::new(
            g,
            fmt,
            w,
            h,
            U,
            &all.iter().map(|m| m.0).collect::<Vec<_>>(),
        )
        .unwrap();
        println!(
            "{fmt:?}: driver picks {:#x} of {}, planes {:?}",
            pick.modifier,
            all.len(),
            pick.planes
        );
        assert_eq!(pick.modifier, all[0].0);
        for &(m, n) in &all {
            let img = Image::new(g, fmt, w, h, U, &[m]).unwrap();
            assert_eq!((img.modifier, img.planes.len()), (m, n as usize));
            img.acquire().unwrap();
            pat.draw(g, &img.tex, 5, 0);
            let sf = img.release().unwrap();
            assert!(readable(sf.as_raw_fd(), 2000));
            let (t, vi) = unsafe { import(g, &shared(&img)) };
            let rb = readback(g, &t, vi, img.fam);
            assert_eq!(
                check(fmt, w, h, 5, |k| (rb[k].0.as_ptr(), rb[k].1)),
                0,
                "{fmt:?} {m:#x}"
            );
        }
    }
}

/// The OBS frame protocol carrying real exported DMA-BUFs and sync_files: the client waits for the sync_file,
/// maps the planes and finds the pattern; `done` frees the image for the ring.
#[test]
fn obs_dmabuf() {
    use crate::obs::*;
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let dir = std::env::var_os("XDG_RUNTIME_DIR").unwrap_or("/tmp".into());
    let p =
        std::path::Path::new(&dir).join(format!("x5desk-dmabuf-{}/obs.sock", std::process::id()));
    let mut s = Server::bind(&p).unwrap();
    s.add(1, "360");
    for fmt in [Fmt::Nv12, Fmt::Rgba] {
        let (w, h) = (1920, 960);
        let mut ring = Ring::new(g, fmt, w, h, U, &[LINEAR], 3).unwrap();
        let m = ring.imgs[0].modifier;
        s.format(
            1,
            Format {
                modifier: m,
                w,
                h,
                fourcc: fmt.drm(),
                fps: [30, 1],
                color: BT709,
                ..Default::default()
            },
        );
        let mut c = Client::connect(&p, 1, &[]).unwrap();
        s.poll();
        assert!(matches!(
            c.recv(Some(Duration::from_secs(1))).unwrap(),
            Msg::Hello(_)
        ));
        assert!(
            matches!(c.recv(Some(Duration::from_secs(1))).unwrap(), Msg::Format(1, f) if f.fourcc == fmt.drm())
        );
        let mut held = vec![];
        for f in 0..8u32 {
            s.poll();
            // the client holds every frame until the next one arrived: at most 2 of 3 images busy
            let i = ring.next(|i| s.busy(1, i as u32)).expect("a free image");
            let img = &ring.imgs[i];
            img.acquire().unwrap();
            pat.draw(g, &img.tex, f, 0);
            let sf = img.release().unwrap();
            let mut fr = Frame {
                ts: f as u64,
                modifier: img.modifier,
                w,
                h,
                fourcc: fmt.drm(),
                buf: i as u32,
                ..Default::default()
            };
            for (k, pl) in img.planes.iter().enumerate() {
                (fr.off[k], fr.stride[k]) = (pl.off, pl.stride);
            }
            let fds = [std::os::fd::AsFd::as_fd(&img.fd); 4];
            assert_eq!(
                s.frame(
                    1,
                    &fr,
                    &fds[..img.planes.len()],
                    Some(std::os::fd::AsFd::as_fd(&sf))
                ),
                1
            );
            let Msg::Frame {
                f: got,
                planes,
                sync: Some(sy),
                ..
            } = c.recv(Some(Duration::from_secs(1))).unwrap()
            else {
                panic!("no frame")
            };
            assert!(readable(sy.as_raw_fd(), 2000));
            let pl0 = planes[0].as_ref().unwrap();
            // the same DMA-BUF, not a copy
            assert_eq!(ino(pl0.as_raw_fd()), ino(img.fd.as_raw_fd()));
            let mp = Map::new(pl0.as_raw_fd(), img.size as usize, true);
            assert_eq!(
                check(fmt, w, h, f, |k| unsafe {
                    (mp.p.add(got.off[k] as usize), got.stride[k] as usize)
                }),
                0
            );
            drop(mp);
            held.push(got.buf);
            if held.len() == 2 {
                c.done(1, held.remove(0)).unwrap();
            }
        }
        s.poll();
        assert!(s.busy(1, held[0]));
        c.done(1, held[0]).unwrap();
        s.poll();
        assert!((0..3).all(|i| !s.busy(1, i)));
    }
}

/// Modifier negotiation over the OBS protocol: the ring follows what every client of the source imports. A client
/// listing every modifier gets the driver's pick (tiled), one listing all but that narrows it, one without a list
/// forces LINEAR, and its leaving widens it again. Each client re-imports every frame with the modifier and plane
/// layout the frame names, as the plugin will, and finds the pattern.
#[test]
fn obs_mods() {
    use crate::obs::*;
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let dir = std::env::var_os("XDG_RUNTIME_DIR").unwrap_or("/tmp".into());
    let p = std::path::Path::new(&dir).join(format!("x5desk-mods-{}/obs.sock", std::process::id()));
    let mut s = Server::bind(&p).unwrap();
    s.add(1, "360");
    let (fmt, w, h) = (Fmt::Rgba, 1920, 960);
    let all: Vec<u64> = modifiers(g, fmt, U, w, h).iter().map(|m| m.0).collect();
    let fewer: Vec<u64> = all[1..].iter().copied().filter(|&m| m != LINEAR).collect();
    // each client and the modifier of the last Format it got
    let mut cl: Vec<(Client, u64)> = vec![];
    let (mut ring, mut mods, mut n) = (None::<Ring>, None, 0u32);
    let steps: [(&str, Option<&[u64]>); 4] = [
        ("all", Some(&all)),
        ("all but the pick", Some(&fewer)),
        ("no list", Some(&[])),
        ("no-list client gone", None),
    ];
    for (what, list) in steps {
        match list {
            Some(l) => cl.push((Client::connect(&p, 1, &[(fmt.drm(), l)]).unwrap(), u64::MAX)),
            None => drop(cl.pop()),
        }
        std::thread::sleep(Duration::from_millis(20));
        assert!(s.poll(), "{what}: no change reported");
        let m = s.modifiers(1, fmt.drm());
        if mods.as_ref() != Some(&m) {
            let r = Ring::new(g, fmt, w, h, U, &m, 3).unwrap();
            assert!(!s.takes(1, fmt.drm(), all[0]) || r.imgs[0].modifier == all[0]);
            s.format(
                1,
                Format {
                    modifier: r.imgs[0].modifier,
                    w,
                    h,
                    fourcc: fmt.drm(),
                    fps: [30, 1],
                    ..Default::default()
                },
            );
            (ring, mods) = (Some(r), Some(m));
        }
        let r = ring.as_mut().unwrap();
        let md = r.imgs[0].modifier;
        assert!(s.takes(1, fmt.drm(), md));
        let want = match what {
            "all" => all[0],
            "no list" => LINEAR,
            _ => *fewer.first().unwrap_or(&LINEAR),
        };
        println!(
            "{what}: {} clients, common list {:x?}, ring in {md:#x}",
            cl.len(),
            mods.as_ref().unwrap()
        );
        // the driver's pick among the list: its first in device order
        assert_eq!(md, want, "{what}");
        for _ in 0..3 {
            s.poll();
            let i = r.next(|i| s.busy(1, i as u32)).expect("a free image");
            let img = &r.imgs[i];
            img.acquire().unwrap();
            pat.draw(g, &img.tex, n, 0);
            let sf = img.release().unwrap();
            let mut fr = Frame {
                ts: n as u64,
                modifier: md,
                w,
                h,
                fourcc: fmt.drm(),
                buf: i as u32,
                ..Default::default()
            };
            for (k, pl) in img.planes.iter().enumerate() {
                (fr.off[k], fr.stride[k]) = (pl.off, pl.stride);
            }
            let fds = [std::os::fd::AsFd::as_fd(&img.fd); 4];
            let sync = Some(std::os::fd::AsFd::as_fd(&sf));
            assert_eq!(s.frame(1, &fr, &fds[..img.planes.len()], sync), cl.len());
            for (c, last) in &mut cl {
                let (f, pl) = loop {
                    match c.recv(Some(Duration::from_secs(1))).unwrap() {
                        Msg::Frame {
                            f,
                            mut planes,
                            sync,
                            ..
                        } => {
                            assert!(readable(sync.unwrap().as_raw_fd(), 2000));
                            break (f, planes[0].take().unwrap());
                        }
                        // a client that joins gets the source's format, then the one re-exported for it
                        Msg::Format(_, f) => *last = f.modifier,
                        _ => {}
                    }
                };
                assert_eq!((*last, f.modifier), (md, md), "{what}");
                let planes = (0..f.planes as usize)
                    .map(|k| Plane {
                        off: f.off[k],
                        stride: f.stride[k],
                        len: 0,
                    })
                    .collect();
                let sh = Shared {
                    fmt,
                    w: f.w,
                    h: f.h,
                    modifier: f.modifier,
                    fd: pl.as_raw_fd(),
                    planes,
                };
                let (t, vi) = unsafe { import(g, &sh) };
                let rb = readback(g, &t, vi, img.fam);
                assert_eq!(
                    check(fmt, w, h, n, |k| (rb[k].0.as_ptr(), rb[k].1)),
                    0,
                    "{what} frame {n}"
                );
                c.done(1, f.buf).unwrap();
            }
            n += 1;
        }
    }
}

// --- PipeWire: a consumer stream through the same FFI ---

use crate::pw::{self, Cam, spa};
use std::ffi::{CString, c_void};

struct Got {
    pts: u64,
    ty: u32,
    ino: u64,
    /// every data block shares the first one's DMA-BUF
    one: bool,
    /// pattern mismatches when the consumer could map the frame
    bad: Option<usize>,
    planes: Vec<Plane>,
    fd: Option<std::os::fd::OwnedFd>,
}

struct SinkSt {
    s: *mut pw::Stream,
    mode: Option<(Fmt, u32, u32, Option<u64>)>,
    got: Vec<Got>,
    /// keep a dup of each DMA-BUF (tiled frames are checked by re-import afterwards)
    keep: bool,
}

struct Sink {
    tl: *mut pw::ThreadLoop,
    st: *mut SinkSt,
}

unsafe extern "C" fn sink_param(d: *mut c_void, id: u32, p: *const pw::Pod) {
    let st = unsafe { &mut *d.cast::<SinkSt>() };
    if id != spa::FORMAT {
        return;
    }
    st.mode = unsafe { pw::parse(p) }.map(|(f, w, h, m)| (f, w, h, m.map(|m| m.1[0])));
    let Some((.., m)) = st.mode else { return };
    let mut b = pw::B::default();
    b.obj(spa::OBJ_BUFFERS, spa::BUFFERS, |b| {
        b.prop(spa::B_BUFFERS, 0)
            .choice(spa::RANGE, spa::INT, 4, pw::raw(&[4u32, 1, 16]));
        let t = if m.is_some() { spa::DMABUF } else { spa::MEMFD };
        b.prop(spa::B_DATATYPE, 0)
            .choice(spa::FLAGS, spa::INT, 4, pw::raw(&[1u32 << t]));
    });
    unsafe { pw::update(st.s, &[b.done(), pw::meta()]) };
}

unsafe extern "C" fn sink_process(d: *mut c_void) {
    let st = unsafe { &mut *d.cast::<SinkSt>() };
    let Some((fmt, w, h, m)) = st.mode else {
        return;
    };
    loop {
        let b = unsafe { pw::pw_stream_dequeue_buffer(st.s) };
        if b.is_null() {
            return;
        }
        unsafe {
            let sb = &*(*b).buffer;
            let datas = std::slice::from_raw_parts(sb.datas, sb.n_datas as usize);
            let metas = std::slice::from_raw_parts(sb.metas, sb.n_metas as usize);
            let pts = metas
                .iter()
                .find(|x| x.ty == spa::META_HEADER)
                .map_or(u64::MAX, |x| (*x.data.cast::<pw::Header>()).pts as u64);
            let planes: Vec<Plane> = datas
                .iter()
                .map(|d| Plane {
                    off: (*d.chunk).offset,
                    stride: (*d.chunk).stride as u32,
                    len: (*d.chunk).size,
                })
                .collect();
            let (d0, fd) = (&datas[0], datas[0].fd as RawFd);
            let bad = (d0.ty == spa::MEMFD || m == Some(LINEAR)).then(|| {
                let mp = Map::new(
                    fd,
                    (d0.mapoffset + d0.maxsize) as usize,
                    d0.ty == spa::DMABUF,
                );
                let p0 = planes[0];
                check(fmt, w, h, pts as u32, |k| match d0.ty {
                    spa::DMABUF => (mp.p.add(planes[k].off as usize), planes[k].stride as usize),
                    _ => (
                        mp.p.add((p0.off + k as u32 * p0.stride * h) as usize),
                        p0.stride as usize,
                    ),
                })
            });
            st.got.push(Got {
                pts,
                ty: d0.ty,
                ino: ino(fd),
                one: datas.iter().all(|d| ino(d.fd as RawFd) == ino(fd)),
                bad,
                planes,
                fd: st
                    .keep
                    .then(|| std::os::fd::FromRawFd::from_raw_fd(libc::dup(fd))),
            });
            pw::pw_stream_queue_buffer(st.s, b);
        }
    }
}

static SINK_EV: pw::Events = pw::Events {
    version: 2,
    destroy: None,
    state_changed: None,
    control_info: None,
    io_changed: None,
    param_changed: Some(sink_param),
    add_buffer: None,
    remove_buffer: None,
    process: Some(sink_process),
    drained: None,
    command: None,
    trigger_done: None,
};

impl Sink {
    /// A consumer of node `target` (its node.name) for `fmt` at `w` × `h`; `mods`: DMA-BUF with these
    /// modifiers, None: MemFd.
    fn new(target: &str, fmt: Fmt, w: u32, h: u32, mods: Option<&[u64]>, keep: bool) -> Sink {
        unsafe {
            pw::pw_init(ptr::null_mut(), ptr::null_mut());
            let tl = pw::pw_thread_loop_new(c"x5desk-sink".as_ptr(), ptr::null());
            let st = Box::into_raw(Box::new(SinkSt {
                s: ptr::null_mut(),
                mode: None,
                got: vec![],
                keep,
            }));
            pw::pw_thread_loop_lock(tl);
            let p = pw::pw_properties_new(ptr::null::<std::ffi::c_char>());
            let t = CString::new(target).unwrap();
            for (k, v) in [
                (c"media.type", c"Video"),
                (c"media.category", c"Capture"),
                (c"media.role", c"Camera"),
                (c"node.name", c"x5desk-test-sink"),
                (c"target.object", t.as_c_str()),
            ] {
                pw::pw_properties_set(p, k.as_ptr(), v.as_ptr());
            }
            let s = pw::pw_stream_new_simple(
                pw::pw_thread_loop_get_loop(tl),
                c"sink".as_ptr(),
                p,
                &SINK_EV,
                st.cast(),
            );
            (*st).s = s;
            let o = pw::offer(fmt, &[[w, h]], 30, mods);
            let mut v = [o.as_ptr().cast::<pw::Pod>()];
            assert!(
                pw::pw_stream_connect(s, spa::INPUT, spa::ANY, spa::AUTOCONNECT, v.as_mut_ptr(), 1)
                    >= 0
            );
            assert!(pw::pw_thread_loop_start(tl) >= 0);
            pw::pw_thread_loop_unlock(tl);
            Sink { tl, st }
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut SinkSt) -> R) -> R {
        unsafe {
            pw::pw_thread_loop_lock(self.tl);
            let r = f(&mut *self.st);
            pw::pw_thread_loop_unlock(self.tl);
            r
        }
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        unsafe {
            pw::pw_thread_loop_lock(self.tl);
            pw::pw_stream_destroy((*self.st).s);
            pw::pw_thread_loop_unlock(self.tl);
            pw::pw_thread_loop_stop(self.tl);
            pw::pw_thread_loop_destroy(self.tl);
            drop(Box::from_raw(self.st));
        }
    }
}

/// Renders frame n with pts n into every buffer the camera hands out until `done()` or 10 s; returns the frames
/// sent, the DMA-BUF inodes of the images, and the mean CPU time of `dequeue` + `queue` per frame.
fn produce(cam: &Cam, pat: &Pat, mut done: impl FnMut() -> bool) -> (u32, Vec<u64>, Duration) {
    let (g, t) = (gpu(), Instant::now());
    let (mut n, mut inos, mut cpu) = (0, vec![], Duration::ZERO);
    while !done() && t.elapsed() < Duration::from_secs(10) {
        let t0 = Instant::now();
        if let Some(f) = cam.dequeue().unwrap() {
            let t1 = Instant::now();
            let i = ino(f.img.fd.as_raw_fd());
            if !inos.contains(&i) {
                inos.push(i);
            }
            pat.draw(g, &f.img.tex, n, 0);
            let t2 = Instant::now();
            f.queue(n as u64).unwrap();
            cpu += t1 - t0 + t2.elapsed();
            n += 1;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    (n, inos, cpu / n.max(1))
}

fn node(cam: &Cam) -> u32 {
    let t = Instant::now();
    loop {
        if let Some(id) = cam.node() {
            return id;
        }
        assert!(
            t.elapsed() < Duration::from_secs(3),
            "node never registered: {:?}",
            cam.error()
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// `pw-dump <id>` without whitespace, or None without pw-dump.
fn dump(id: u32) -> Option<String> {
    let o = std::process::Command::new("pw-dump")
        .arg(id.to_string())
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&o.stdout).into_owned();
    Some(s.chars().filter(|c| !c.is_whitespace()).collect())
}

/// DMA-BUF, LINEAR, NV12 1080p: the consumer maps every frame and finds the pattern, in the camera's own DMA-BUFs;
/// pw-dump shows the node's properties and the negotiated format.
#[test]
fn pw_linear() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let cam = Cam::new(g.clone(), "X5 test linear", &pw::SIZES, 30).unwrap();
    let id = node(&cam);
    let sink = Sink::new(
        "x5desk.x5-test-linear",
        Fmt::Nv12,
        1920,
        1080,
        Some(&[LINEAR]),
        false,
    );
    let (n, inos, cpu) = produce(&cam, &pat, || sink.with(|s| s.got.len() >= 30));
    let mode = cam.mode();
    assert_eq!(
        mode,
        Some(pw::Mode {
            fmt: Fmt::Nv12,
            w: 1920,
            h: 1080,
            modifier: Some(LINEAR)
        })
    );
    if let Some(d) = dump(id) {
        for s in [
            r#""media.class":"Video/Source""#,
            r#""media.role":"Camera""#,
            r#""node.description":"X5testlinear""#,
            r#""Format":[{"mediaType":"video","mediaSubtype":"raw","format":"NV12","modifier":0,"size":{"width":1920,"height":1080},"framerate":{"num":30,"denom":1}"#,
            r#""colorRange":"16-235","colorMatrix":"bt709""#,
        ] {
            assert!(d.contains(s), "pw-dump lacks {s}");
        }
    }
    sink.with(|s| {
        println!(
            "sent {n}, received {} in {} DMA-BUFs; dequeue + queue {cpu:?} CPU per frame",
            s.got.len(),
            inos.len()
        );
        assert!(s.got.len() >= 30);
        for x in &s.got {
            assert_eq!(
                (x.ty, x.bad, x.one),
                (spa::DMABUF, Some(0), true),
                "frame {}",
                x.pts
            );
            assert!(
                inos.contains(&x.ino),
                "frame {} not in one of the camera's DMA-BUFs",
                x.pts
            );
            assert_eq!(x.planes.len(), 2);
        }
        let mut p: Vec<u64> = s.got.iter().map(|x| x.pts).collect();
        p.dedup();
        assert_eq!(p.len(), s.got.len(), "a frame arrived twice");
    });
}

/// Two consumers of one node at once, both NV12 1080p LINEAR DMA-BUF: each gets frames in the camera's DMA-BUFs,
/// every one with the pattern of the frame its pts names.
#[test]
fn pw_two() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let cam = Cam::new(g.clone(), "X5 test two", &pw::SIZES, 30).unwrap();
    node(&cam);
    let mk = || {
        Sink::new(
            "x5desk.x5-test-two",
            Fmt::Nv12,
            1920,
            1080,
            Some(&[LINEAR]),
            false,
        )
    };
    let (a, b) = (mk(), mk());
    let (n, inos, _) = produce(&cam, &pat, || {
        a.with(|s| s.got.len()) >= 20 && b.with(|s| s.got.len()) >= 20
    });
    std::thread::sleep(Duration::from_millis(100));
    let (ga, gb) = (a.with(|s| s.got.len()), b.with(|s| s.got.len()));
    println!(
        "sent {n} in {} DMA-BUFs; received {ga} and {gb}; mode {:x?}",
        inos.len(),
        cam.mode()
    );
    for k in [&a, &b] {
        k.with(|s| {
            assert!(s.got.len() >= 20);
            for x in &s.got {
                assert_eq!((x.ty, x.bad), (spa::DMABUF, Some(0)), "frame {}", x.pts);
                assert!(inos.contains(&x.ino));
            }
        });
    }
}

/// DMA-BUF with every modifier the device has, RGBx 3840 × 1920: we fixate the driver's pick (tiled, DCC), and the
/// last frame's DMA-BUF, re-imported in Vulkan with the negotiated modifier and the chunk offsets / strides, holds
/// the pattern.
#[test]
fn pw_tiled() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let (w, h) = (3840, 1920);
    let cam = Cam::new(g.clone(), "X5 test tiled", &pw::SIZES, 30).unwrap();
    node(&cam);
    let all: Vec<u64> = modifiers(g, Fmt::Rgba, U, w, h)
        .iter()
        .map(|m| m.0)
        .collect();
    let sink = Sink::new("x5desk.x5-test-tiled", Fmt::Rgba, w, h, Some(&all), true);
    let (n, _, cpu) = produce(&cam, &pat, || sink.with(|s| s.got.len() >= 10));
    let m = cam.mode().unwrap();
    println!(
        "sent {n}; mode {m:x?} (device order {all:x?}); dequeue + queue {cpu:?} CPU per frame"
    );
    assert_eq!(m.modifier, Some(all[0]));
    std::thread::sleep(Duration::from_millis(100));
    let fam = match g.ext.contains(&ash::ext::queue_family_foreign::NAME) {
        true => vk::QUEUE_FAMILY_FOREIGN_EXT,
        false => vk::QUEUE_FAMILY_EXTERNAL,
    };
    sink.with(|s| {
        let x = s.got.last().unwrap();
        assert_eq!(x.ty, spa::DMABUF);
        let img = Shared {
            fmt: Fmt::Rgba,
            w,
            h,
            modifier: all[0],
            fd: x.fd.as_ref().unwrap().as_raw_fd(),
            planes: x.planes.clone(),
        };
        let (t, vi) = unsafe { import(g, &img) };
        let rb = readback(g, &t, vi, fam);
        assert_eq!(
            check(Fmt::Rgba, w, h, x.pts as u32, |k| (
                rb[k].0.as_ptr(),
                rb[k].1
            )),
            0
        );
        println!(
            "frame {} of {} checked through a Vulkan import, planes {:?}",
            x.pts,
            s.got.len(),
            x.planes
        );
    });
}

/// A consumer without DMA-BUF: MemFd buffers with the frame copied through the CPU, BGRx and NV12.
#[test]
fn pw_memfd() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    for (fmt, name) in [
        (Fmt::Bgra, "X5 test memfd bgrx"),
        (Fmt::Nv12, "X5 test memfd nv12"),
    ] {
        let cam = Cam::new(g.clone(), name, &pw::SIZES, 30).unwrap();
        node(&cam);
        let target = format!("x5desk.{}", name.to_lowercase().replace(' ', "-"));
        let sink = Sink::new(&target, fmt, 1920, 1080, None, false);
        let t = Instant::now();
        let (n, _, cpu) = produce(&cam, &pat, || sink.with(|s| s.got.len() >= 20));
        let dt = t.elapsed();
        assert_eq!(cam.mode().unwrap().modifier, None);
        sink.with(|s| {
            println!(
                "{fmt:?}: sent {n}, received {} in {dt:?} (with negotiation); dequeue + queue {cpu:?} CPU per frame",
                s.got.len()
            );
            assert!(s.got.len() >= 20);
            for x in &s.got {
                assert_eq!((x.ty, x.bad), (spa::MEMFD, Some(0)), "frame {}", x.pts);
            }
        });
    }
}

/// GStreamer's pipewiresrc as a third-party consumer: NV12 1080p over MemFd into a file (every frame checked), and
/// RGBx 3840 × 1920 as DMA-BUF in the device's preferred (DCC) modifier into fakesink.
#[test]
fn pw_gst() {
    use std::process::Command;
    if Command::new("gst-launch-1.0")
        .arg("--version")
        .output()
        .is_err()
    {
        println!("no gst-launch-1.0: skipped");
        return;
    }
    let (g, pat) = (gpu(), Pat::new(gpu()));
    // the rate it announces is the one asked for
    let cam = Cam::new(g.clone(), "X5 test gst", &pw::SIZES, 60).unwrap();
    node(&cam);
    let dir = std::env::var_os("XDG_RUNTIME_DIR").unwrap_or("/tmp".into());
    let out = std::path::Path::new(&dir).join(format!("x5desk-gst-{}.nv12", std::process::id()));
    let dcc = modifiers(g, Fmt::Rgba, U, 3840, 1920)[0].0;
    let runs = [
        (
            format!(
                "video/x-raw,format=NV12,width=1920,height=1080 ! filesink location={}",
                out.display()
            ),
            None,
        ),
        (
            format!(
                "video/x-raw(memory:DMABuf),format=DMA_DRM,drm-format=XB24:{dcc:#018x},width=3840,height=1920 ! fakesink"
            ),
            Some(dcc),
        ),
    ];
    for (tail, m) in runs {
        let mut c = Command::new("gst-launch-1.0")
            .args([
                "-v",
                "pipewiresrc",
                "target-object=x5desk.x5-test-gst",
                "num-buffers=8",
                "!",
            ])
            .args(tail.split(' '))
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut seen = None;
        let (n, _, cpu) = produce(&cam, &pat, || {
            seen = cam.mode().or(seen);
            c.try_wait().unwrap().is_some()
        });
        if c.try_wait().unwrap().is_none() {
            c.kill().unwrap();
        }
        let o = c.wait_with_output().unwrap();
        assert!(o.status.success(), "gst-launch failed: {tail}");
        let caps = String::from_utf8_lossy(&o.stdout);
        assert!(caps.contains("framerate=(fraction)60/1"), "{caps}");
        println!(
            "{tail}: sent {n}, mode {seen:x?}, 60/1 negotiated, dequeue + queue {cpu:?} CPU per frame"
        );
        assert_eq!(seen.unwrap().modifier, m);
    }
    let d = std::fs::read(&out).unwrap();
    std::fs::remove_file(&out).unwrap();
    let fs = 1920 * 1080 * 3 / 2;
    assert_eq!(d.len(), 8 * fs);
    for fr in d.chunks_exact(fs) {
        // the frame number is Y(0, 0)
        assert_eq!(
            check(Fmt::Nv12, 1920, 1080, fr[0] as u32, |k| (
                fr[k * 1920 * 1080..].as_ptr(),
                1920
            )),
            0
        );
    }
}

/// Keeps a camera node up for `X5_HOLD` seconds (default 20), streaming the pattern to whoever connects, for
/// checks from the shell (pw-cli, pw-dump, gst-launch, the camera portal).
#[test]
#[ignore]
fn pw_hold() {
    let (g, pat) = (gpu(), Pat::new(gpu()));
    let s: u64 = std::env::var("X5_HOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let cam = Cam::new(g.clone(), "X5 Camera", &pw::SIZES, 30).unwrap();
    println!("node {}", node(&cam));
    let t = Instant::now();
    let (mut n, mut last) = (0, None);
    while t.elapsed() < Duration::from_secs(s) {
        let m = cam.mode();
        if m != last {
            println!("{:?}: mode {m:x?}", t.elapsed());
            last = m;
        }
        if let Some(f) = cam.dequeue().unwrap() {
            pat.draw(g, &f.img.tex, n, 0);
            f.queue(n as u64).unwrap();
            n += 1;
        }
        std::thread::sleep(Duration::from_millis(33));
    }
    println!("sent {n}");
}
