//! DMA-BUF export of render targets (PLAN.md §3.1, §3.2): Vulkan images with a DRM format modifier on the shared
//! device, wrapped as wgpu textures, each exported as one DMA-BUF that holds all its memory planes.
//!
//! Frame contract, with the same staging rules as the decoder's acquire / release (ARCHITECTURE.md):
//! - [`Image::acquire`] before the first submit that writes the image: the GPU waits for the fences that readers
//!   left on the DMA-BUF (implicit sync), and the image goes to its home state `RESOURCE` with its old content
//!   discarded, so render every pixel. Whatever wgpu last did with it (render, copy out) needs no release first.
//! - [`Image::release`] after the last submit that writes it: the image goes to `GENERAL` and is released to the
//!   foreign queue family (the modifier's canonical layout), a binary semaphore signals, and its sync_file goes to
//!   the caller (explicit sync) and also into the DMA-BUF's write fences (`DMA_BUF_IOCTL_IMPORT_SYNC_FILE`) for
//!   consumers that only sync implicitly (PipeWire clients, OBS's PipeWire source).
//! - Between the two a consumer may read it; [`Ring::next`] hands out only images no consumer holds.
//!
//! Staging: a wait staged here may go out with another thread's earlier submit (wgpu-hal makes every submit wait
//! for the one before through its relay semaphores, so the wait still orders every later submit), but a signal must
//! not: `release` stages it only after its own barrier is submitted.
use ash::{ext, vk};
use std::{
    iter,
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    sync::Mutex,
};
use video::{
    Error::{self, Bad},
    Gpu, Result,
};
use wgpu_hal::api::Vulkan as Vk;

pub const LINEAR: u64 = 0;
const DMA_BUF: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
const SYNC_FD: vk::ExternalSemaphoreHandleTypeFlags = vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD;
const RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};
// <linux/dma-buf.h>
const SYNC_WRITE: u32 = 2;
const EXPORT_SYNC_FILE: u32 = 0xc008_6202; // _IOWR('b', 2, struct dma_buf_export_sync_file)
const IMPORT_SYNC_FILE: u32 = 0x4008_6203; // _IOW('b', 3, struct dma_buf_import_sync_file)

#[repr(C)]
struct SyncFile {
    flags: u32,
    fd: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fmt {
    /// R, G, B, A bytes: DRM ABGR8888, PipeWire RGBx / RGBA
    Rgba,
    /// B, G, R, A bytes: DRM ARGB8888, PipeWire BGRx / BGRA
    Bgra,
    /// render through the plane views `R8Unorm` / `Rg8Unorm`
    Nv12,
}

impl Fmt {
    pub fn vk(self) -> vk::Format {
        match self {
            Fmt::Rgba => vk::Format::R8G8B8A8_UNORM,
            Fmt::Bgra => vk::Format::B8G8R8A8_UNORM,
            Fmt::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        }
    }

    pub fn wgpu(self) -> wgpu::TextureFormat {
        match self {
            Fmt::Rgba => wgpu::TextureFormat::Rgba8Unorm,
            Fmt::Bgra => wgpu::TextureFormat::Bgra8Unorm,
            Fmt::Nv12 => wgpu::TextureFormat::NV12,
        }
    }

    /// DRM fourcc
    pub fn drm(self) -> u32 {
        u32::from_le_bytes(*match self {
            Fmt::Rgba => b"AB24",
            Fmt::Bgra => b"AR24",
            Fmt::Nv12 => b"NV12",
        })
    }

    // NV12 renders through per-plane views of other formats
    pub(crate) fn views(self) -> &'static [vk::Format] {
        match self {
            Fmt::Nv12 => &[
                vk::Format::G8_B8R8_2PLANE_420_UNORM,
                vk::Format::R8_UNORM,
                vk::Format::R8G8_UNORM,
            ],
            _ => &[],
        }
    }

    pub(crate) fn flags(self) -> vk::ImageCreateFlags {
        match self {
            Fmt::Nv12 => {
                vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE
            }
            _ => vk::ImageCreateFlags::empty(),
        }
    }
}

/// A memory plane in the DMA-BUF: byte offset, row pitch and size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Plane {
    pub off: u32,
    pub stride: u32,
    pub len: u32,
}

fn usages(u: wgpu::TextureUsages) -> (vk::ImageUsageFlags, wgpu::TextureUses) {
    use vk::ImageUsageFlags as V;
    use wgpu::{TextureUsages as W, TextureUses as H};
    [
        (W::COPY_SRC, V::TRANSFER_SRC, H::COPY_SRC),
        (W::COPY_DST, V::TRANSFER_DST, H::COPY_DST),
        (W::TEXTURE_BINDING, V::SAMPLED, H::RESOURCE),
        (W::STORAGE_BINDING, V::STORAGE, H::STORAGE_READ_WRITE),
        (W::RENDER_ATTACHMENT, V::COLOR_ATTACHMENT, H::COLOR_TARGET),
    ]
    .into_iter()
    .filter(|t| u.contains(t.0))
    .fold((V::empty(), H::empty()), |a, t| (a.0 | t.1, a.1 | t.2))
}

/// The DRM modifiers this device exports `fmt` with for `usage` at `w` × `h`, with their memory plane counts, in
/// the driver's order.
pub fn modifiers(g: &Gpu, fmt: Fmt, usage: wgpu::TextureUsages, w: u32, h: u32) -> Vec<(u64, u32)> {
    let props = |l: &mut vk::DrmFormatModifierPropertiesListEXT| unsafe {
        let mut p = vk::FormatProperties2::default().push_next(l);
        g.inst
            .get_physical_device_format_properties2(g.pd, fmt.vk(), &mut p);
    };
    let mut l = vk::DrmFormatModifierPropertiesListEXT::default();
    props(&mut l);
    let mut v = vec![Default::default(); l.drm_format_modifier_count as usize];
    props(
        &mut vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut v),
    );
    let u = usages(usage).0;
    v.iter()
        .filter(|m| unsafe {
            let mut mi = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
                .drm_format_modifier(m.drm_format_modifier)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let mut ei = vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(DMA_BUF);
            let mut fl = vk::ImageFormatListCreateInfo::default().view_formats(fmt.views());
            let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(fmt.vk())
                .ty(vk::ImageType::TYPE_2D)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(u)
                .flags(fmt.flags())
                .push_next(&mut mi)
                .push_next(&mut ei);
            if !fmt.views().is_empty() {
                info = info.push_next(&mut fl);
            }
            let mut ep = vk::ExternalImageFormatProperties::default();
            let mut p = vk::ImageFormatProperties2::default().push_next(&mut ep);
            let ok = g
                .inst
                .get_physical_device_image_format_properties2(g.pd, &info, &mut p)
                .is_ok();
            let e = p.image_format_properties.max_extent;
            ok && e.width >= w
                && e.height >= h
                && ep
                    .external_memory_properties
                    .external_memory_features
                    .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        })
        .map(|m| (m.drm_format_modifier, m.drm_format_modifier_plane_count))
        .collect()
}

fn ready(fd: RawFd, ev: i16) -> bool {
    let mut p = libc::pollfd {
        fd,
        events: ev,
        revents: 0,
    };
    unsafe { libc::poll(&mut p, 1, 0) == 1 }
}

/// An exportable image and its wgpu texture. wgpu owns the Vulkan image and memory; the DMA-BUF fd keeps the
/// memory alive for consumers beyond that.
pub struct Image {
    pub tex: wgpu::Texture,
    pub fmt: Fmt,
    pub w: u32,
    pub h: u32,
    pub modifier: u64,
    /// the DMA-BUF; every plane lives in it (the image is not disjoint)
    pub fd: OwnedFd,
    pub planes: Vec<Plane>,
    /// bytes of the DMA-BUF
    pub size: u64,
    img: vk::Image,
    /// signalled by `release`'s submit, exported as the frame's sync_file
    done: vk::Semaphore,
    /// takes the readers' fences in `acquire`
    wait: vk::Semaphore,
    /// the submit that carries (or follows) `wait`'s last import: it must have run before the next import
    imported: Mutex<Option<wgpu::SubmissionIndex>>,
    /// the queue family a consumer reads from: foreign, or external without VK_EXT_queue_family_foreign
    pub(crate) fam: u32,
    gfx: u32,
    dev: ash::Device,
    sem: ash::khr::external_semaphore_fd::Device,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Image {
    /// An exportable `fmt` image of `w` × `h` with `usage`, in the modifier the driver prefers among those in
    /// `mods` it supports, or LINEAR when it supports none of them (the fallback every consumer reads).
    pub fn new(
        g: &Gpu,
        fmt: Fmt,
        w: u32,
        h: u32,
        usage: wgpu::TextureUsages,
        mods: &[u64],
    ) -> Result<Image> {
        let (Some(mfd), Some(sem)) = (&g.mem_fd, &g.sem_fd) else {
            return Err(Bad("no external memory / semaphore fd support"));
        };
        if !g.ext.contains(&ext::image_drm_format_modifier::NAME) {
            return Err(Bad("no VK_EXT_image_drm_format_modifier"));
        }
        let have = modifiers(g, fmt, usage, w, h);
        let mut m: Vec<u64> = have
            .iter()
            .map(|x| x.0)
            .filter(|x| mods.contains(x))
            .collect();
        if m.is_empty() {
            if !have.iter().any(|x| x.0 == LINEAR) {
                return Err(Bad(
                    "LINEAR is not exportable for this format, usage and size",
                ));
            }
            m.push(LINEAR);
        }
        let (vu, hu) = usages(usage);
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let hd = unsafe { g.device.as_hal::<Vk>() }.ok_or(Bad("not a Vulkan device"))?;
        unsafe {
            let mut ml =
                vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&m);
            let mut em = vk::ExternalMemoryImageCreateInfo::default().handle_types(DMA_BUF);
            let mut fl = vk::ImageFormatListCreateInfo::default().view_formats(fmt.views());
            let mut info = vk::ImageCreateInfo::default()
                .flags(fmt.flags())
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt.vk())
                .extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vu)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ml)
                .push_next(&mut em);
            if !fmt.views().is_empty() {
                info = info.push_next(&mut fl);
            }
            let img = g.dev.create_image(&info, None)?;
            let mut mp = vk::ImageDrmFormatModifierPropertiesEXT::default();
            let mp = ext::image_drm_format_modifier::Device::new(&g.inst, &g.dev)
                .get_image_drm_format_modifier_properties(img, &mut mp)
                .map(|_| mp.drm_format_modifier);
            let req = g.dev.get_image_memory_requirements(img);
            let mt = &g.mem;
            let find = |want| {
                (0..mt.memory_type_count).find(|&i| {
                    req.memory_type_bits >> i & 1 == 1
                        && mt.memory_types[i as usize].property_flags.contains(want)
                })
            };
            use vk::MemoryPropertyFlags as M;
            let mem = mp
                .map_err(Error::from)
                .and_then(|m| {
                    // CPU-visible when LINEAR, so consumers can mmap the DMA-BUF (amdgpu refuses that for
                    // memory without CPU access)
                    (m == LINEAR)
                        .then(|| find(M::DEVICE_LOCAL | M::HOST_VISIBLE))
                        .flatten()
                        .or_else(|| find(M::DEVICE_LOCAL))
                        .ok_or(Bad("no device-local memory for the image"))
                })
                .and_then(|t| {
                    let mut ex = vk::ExportMemoryAllocateInfo::default().handle_types(DMA_BUF);
                    let mut de = vk::MemoryDedicatedAllocateInfo::default().image(img);
                    let info = vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(t)
                        .push_next(&mut ex)
                        .push_next(&mut de);
                    let mem = g.dev.allocate_memory(&info, None)?;
                    match g.dev.bind_image_memory(img, mem, 0) {
                        Ok(()) => Ok(mem),
                        Err(e) => {
                            g.dev.free_memory(mem, None);
                            Err(e.into())
                        }
                    }
                });
            let mem = match mem {
                Ok(m) => m,
                Err(e) => {
                    g.dev.destroy_image(img, None);
                    return Err(e);
                }
            };
            // wgpu owns image and memory from here and frees them once the texture is unused
            let desc = wgpu_hal::TextureDescriptor {
                label: Some("dmabuf"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt.wgpu(),
                usage: hu,
                memory_flags: wgpu_hal::MemoryFlags::empty(),
                view_formats: vec![],
            };
            let hal = hd.texture_from_raw(
                img,
                &desc,
                None,
                wgpu_hal::vulkan::TextureMemory::Dedicated(mem),
            );
            let tex = g.device.create_texture_from_hal::<Vk>(
                hal,
                &wgpu::TextureDescriptor {
                    label: Some("dmabuf"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: fmt.wgpu(),
                    usage,
                    view_formats: &[],
                },
                wgpu::TextureUses::RESOURCE,
            );
            let fd = mfd.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(mem)
                    .handle_type(DMA_BUF),
            )?;
            let fd = OwnedFd::from_raw_fd(fd);
            let modifier = mp?;
            let n = have.iter().find(|x| x.0 == modifier).map_or(1, |x| x.1);
            const ASP: [vk::ImageAspectFlags; 4] = [
                vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
                vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
                vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
            ];
            let planes = ASP[..n as usize]
                .iter()
                .map(|&a| {
                    let l = g.dev.get_image_subresource_layout(
                        img,
                        vk::ImageSubresource {
                            aspect_mask: a,
                            mip_level: 0,
                            array_layer: 0,
                        },
                    );
                    Plane {
                        off: l.offset as u32,
                        stride: l.row_pitch as u32,
                        len: l.size as u32,
                    }
                })
                .collect();
            let mut es = vk::ExportSemaphoreCreateInfo::default().handle_types(SYNC_FD);
            let done = g
                .dev
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut es), None)?;
            let wait = match g.dev.create_semaphore(&Default::default(), None) {
                Ok(s) => s,
                Err(e) => {
                    g.dev.destroy_semaphore(done, None);
                    return Err(e.into());
                }
            };
            Ok(Image {
                tex,
                fmt,
                w,
                h,
                modifier,
                fd,
                planes,
                size: req.size,
                img,
                done,
                wait,
                imported: Mutex::new(None),
                fam: match g.ext.contains(&ext::queue_family_foreign::NAME) {
                    true => vk::QUEUE_FAMILY_FOREIGN_EXT,
                    false => vk::QUEUE_FAMILY_EXTERNAL,
                },
                gfx: g.gfx,
                dev: g.dev.clone(),
                sem: sem.clone(),
                device: g.device.clone(),
                queue: g.queue.clone(),
            })
        }
    }

    /// A wgpu encoder that moves wgpu's view of the image to `RESOURCE` (from whatever it last did), then a raw one
    /// with the barrier `from` → `to`, `src` → `dst` family: PLAN.md §2.1 rule 5.
    fn barrier(
        &self,
        from: vk::ImageLayout,
        to: vk::ImageLayout,
        src: u32,
        dst: u32,
    ) -> [wgpu::CommandBuffer; 2] {
        let mut t = self.device.create_command_encoder(&Default::default());
        t.transition_resources(
            iter::empty(),
            iter::once(wgpu::TextureTransition {
                texture: &self.tex,
                selector: None,
                state: wgpu::TextureUses::RESOURCE,
            }),
        );
        let mut e = self.device.create_command_encoder(&Default::default());
        let b = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(from)
            .new_layout(to)
            .src_queue_family_index(src)
            .dst_queue_family_index(dst)
            .image(self.img)
            .subresource_range(RANGE);
        unsafe {
            e.as_hal_mut::<Vk, _, _>(|e| {
                if let Some(e) = e {
                    self.dev.cmd_pipeline_barrier(
                        e.raw_handle(),
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[b],
                    );
                }
            })
        };
        [t.finish(), e.finish()]
    }

    /// Call before the first submit that writes the image (on the thread that makes it): stages a wait for the
    /// readers' fences on the DMA-BUF, if any are pending, and submits the move to `RESOURCE`. The content is
    /// undefined afterwards.
    pub fn acquire(&self) -> Result<()> {
        let q = unsafe { self.queue.as_hal::<Vk>() }.ok_or(Bad("not a Vulkan queue"))?;
        let mut imp = self.imported.lock().unwrap();
        let busy = !ready(self.fd.as_raw_fd(), libc::POLLOUT);
        if busy {
            // a semaphore takes a new payload only once the wait on its last one has run (VUID-01142); that wait
            // went out with the submit recorded below or an earlier one. A frame or more ago, so rarely a wait
            if let Some(i) = imp.take() {
                let w = wgpu::PollType::Wait {
                    submission_index: Some(i),
                    timeout: None,
                };
                self.device.poll(w).map_err(|e| Error::Gpu(e.to_string()))?;
            }
            // a writer waits for every fence, readers' and writers'
            let mut s = SyncFile {
                flags: SYNC_WRITE,
                fd: -1,
            };
            if unsafe { libc::ioctl(self.fd.as_raw_fd(), EXPORT_SYNC_FILE as _, &mut s) } < 0 {
                return Err(sys("DMA_BUF_IOCTL_EXPORT_SYNC_FILE"));
            }
            let info = vk::ImportSemaphoreFdInfoKHR::default()
                .semaphore(self.wait)
                .flags(vk::SemaphoreImportFlags::TEMPORARY)
                .handle_type(SYNC_FD)
                .fd(s.fd);
            // a successful import owns the fd
            if let Err(e) = unsafe { self.sem.import_semaphore_fd(&info) } {
                unsafe { libc::close(s.fd) };
                return Err(e.into());
            }
            q.add_wait_semaphore(self.wait, None, vk::PipelineStageFlags::ALL_COMMANDS);
        }
        drop(q);
        let b = self.barrier(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
        );
        let i = self.queue.submit(b);
        if busy {
            *imp = Some(i);
        }
        Ok(())
    }

    /// Call after the last submit that writes the image: hands it over and returns the sync_file that signals
    /// once it is written; the same fence is on the DMA-BUF as a write fence.
    pub fn release(&self) -> Result<OwnedFd> {
        let b = self.barrier(
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::GENERAL,
            self.gfx,
            self.fam,
        );
        self.queue.submit(b);
        // staged only now: whichever submit takes it, ours below or another thread's, follows the barrier. wgpu
        // holds one lock across each hal submit, so once ours returns, the signal is submitted
        unsafe { self.queue.as_hal::<Vk>() }
            .ok_or(Bad("not a Vulkan queue"))?
            .add_signal_semaphore(self.done, None);
        self.queue.submit(iter::empty());
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.done)
            .handle_type(SYNC_FD);
        let sf = unsafe { OwnedFd::from_raw_fd(self.sem.get_semaphore_fd(&info)?) };
        attach(self.fd.as_raw_fd(), sf.as_raw_fd())?;
        Ok(sf)
    }
}

/// Adds sync_file `sf` to DMA-BUF `fd` as a write fence: implicit-sync readers wait for it (kernel >= 6.0).
pub fn attach(fd: RawFd, sf: RawFd) -> Result<()> {
    let s = SyncFile {
        flags: SYNC_WRITE,
        fd: sf,
    };
    if unsafe { libc::ioctl(fd, IMPORT_SYNC_FILE as _, &s) } < 0 {
        return Err(sys("DMA_BUF_IOCTL_IMPORT_SYNC_FILE"));
    }
    Ok(())
}

fn sys(what: &str) -> Error {
    Error::Gpu(format!("{what}: {}", std::io::Error::last_os_error()))
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            // operations staged but never submitted must not reach a destroyed semaphore
            if let Some(q) = self.queue.as_hal::<Vk>() {
                while q.remove_wait_semaphore(self.wait) {}
                while q.remove_signal_semaphore(self.done) {}
            }
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.dev.destroy_semaphore(self.done, None);
            self.dev.destroy_semaphore(self.wait, None);
        }
    }
}

/// N images of one format, size and modifier, for a consumer that hands images back by index (the OBS protocol's
/// `done`).
pub struct Ring {
    pub imgs: Vec<Image>,
    next: usize,
}

impl Ring {
    /// `n` images; the first picks the modifier (as `Image::new`), the rest use it.
    pub fn new(
        g: &Gpu,
        fmt: Fmt,
        w: u32,
        h: u32,
        usage: wgpu::TextureUsages,
        mods: &[u64],
        n: usize,
    ) -> Result<Ring> {
        let a = Image::new(g, fmt, w, h, usage, mods)?;
        let m = [a.modifier];
        let mut imgs = vec![a];
        for _ in 1..n {
            imgs.push(Image::new(g, fmt, w, h, usage, &m)?);
        }
        Ok(Ring { imgs, next: 0 })
    }

    /// The next image, round robin, that no consumer holds (`busy(i)`: a consumer still reads image `i`); None
    /// when every image is held. `acquire` it before rendering.
    pub fn next(&mut self, busy: impl Fn(usize) -> bool) -> Option<usize> {
        let n = self.imgs.len();
        let i = (0..n).map(|k| (self.next + k) % n).find(|&i| !busy(i))?;
        self.next = i + 1;
        Some(i)
    }
}
