//! Vulkan Video decode (H.264 / H.265 Main and Main 10) into a ring of NV12 / P010 images that wgpu samples in place.
//! Where the driver keeps DPB and output images distinct (RADV), a picture decodes into its ring image, no copy; where
//! they coincide (NVIDIA), into its DPB slot, and the decode queue copies it into the ring image: a DPB image never
//! leaves the decode queue, so later pictures reference it while wgpu samples the copy. Rules from PLAN.md §2.1:
//! - An output image's home state is RESOURCE (SHADER_READ_ONLY_OPTIMAL), wgpu's initial state for it. The decode
//!   queue moves it to VIDEO_DECODE_DST (coinciding: TRANSFER_DST), decodes (copies) and moves it back; wgpu only
//!   samples it.
//! - Output images are CONCURRENT across wgpu's family and the decode family; DPB images are private.
//! - Timeline `done`: the decode of frame f signals f + 1. Timeline `free`: each wgpu submit that samples f waits for
//!   `done ≥ f + 1` (`Ring::acquire`), the next submit after the last of them signals f + 1 (`Ring::release`); the
//!   decode into a slot waits for `free` and `done` from the frame that used the slot before.
//! - A dropped decoder's memory is kept for the next decoder of its shape (`Bin`) instead of freed.
pub use crate::pic::{Codec, MAXS, OUT};
pub(crate) use crate::pic::{Pic, Ref};
use crate::{
    Error::{self, Bad},
    Gpu, Result, h264, h265,
    pic::St,
};
use ash::vk::{self, native::*};
use std::{
    cell::Cell,
    ffi::c_void,
    mem::take,
    ptr,
    sync::{Arc, Mutex, Weak},
};
use wgpu_hal::api::Vulkan as Vk;

const BS: u64 = 8 << 20; // bitstream bytes per access unit
const WAIT: u64 = 50_000_000; // ns decode() waits for its slot before returning Full
const PARK: usize = 2; // parked bodies and rings per shape: a deck's two lenses, while another deck opens
const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
const P010: vk::Format = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
const RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// What decoders that can share memory have in common: codec, profile, 10-bit, texture size, DPB slots, ring size.
type Shape = (Codec, u32, bool, u32, u32, usize, usize);

/// A decoder's session (with its memory), DPB, bitstream buffer, command buffers and result status queries.
struct Body {
    dpb: Vec<(vk::Image, vk::ImageView)>,
    mem: Vec<vk::DeviceMemory>,
    session: vk::VideoSessionKHR,
    bs: vk::Buffer,
    map: *mut u8,
    pool: vk::CommandPool,
    cbs: Vec<vk::CommandBuffer>,
    qp: vk::QueryPool,
}

// `map` points into host-coherent memory that only its decoder touches
unsafe impl Send for Body {}

impl Body {
    /// Destroys it; nothing of it is in flight.
    unsafe fn free(self, d: &ash::Device, vq: &ash::khr::video_queue::Device) {
        unsafe {
            (vq.fp().destroy_video_session_khr)(d.handle(), self.session, ptr::null());
            for &(img, v) in &self.dpb {
                d.destroy_image_view(v, None);
                d.destroy_image(img, None);
            }
            d.destroy_buffer(self.bs, None);
            d.destroy_command_pool(self.pool, None);
            d.destroy_query_pool(self.qp, None);
            for &m in &self.mem {
                d.free_memory(m, None);
            }
        }
    }
}

/// A ring's textures and their decode views. The next ring gets new timelines and numbers its frames from 0: a slot's
/// first decode then leaves `UNDEFINED`, which any layout may.
struct Out {
    tex: Vec<wgpu::Texture>,
    imgs: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
}

impl Out {
    /// Destroys it; nothing samples or decodes into it any more.
    unsafe fn free(self, d: &ash::Device) {
        for &v in &self.views {
            unsafe { d.destroy_image_view(v, None) };
        }
        // the textures go after their views
    }
}

/// Decoder memory parked for reuse (`Gpu::bin`), at most `PARK` bodies and rings per shape. amdgpu makes the
/// page-table update that unmaps freed memory wait for every submission queued in the process's VM, decodes included,
/// and graphics submissions after it wait for that update. During playback the decode queue never drains (two 3840²
/// lenses keep a VCN busy), so memory freed then held the UI's frames back by the backlog: 40–190 ms at the editor's
/// cuts, where a deck is dropped while the next one decodes from its keyframe. Another process's decodes are not in
/// our VM. So a dropped decoder parks its body here, a dropped ring its textures, and the next decoder of the shape
/// takes them (timelines are no memory: each ring makes its own): once a deck's worth is parked, a cut frees and
/// allocates nothing.
#[derive(Default)]
pub struct Bin {
    body: Vec<(Shape, Body)>,
    out: Vec<(Shape, Out)>,
}

impl Bin {
    /// Parks `t` unless `PARK` of its shape are parked; else hands it back.
    fn park<T>(v: &mut Vec<(Shape, T)>, s: Shape, t: T) -> Option<T> {
        match v.iter().filter(|x| x.0 == s).count() < PARK {
            true => {
                v.push((s, t));
                None
            }
            false => Some(t),
        }
    }

    fn take<T>(v: &mut Vec<(Shape, T)>, s: Shape) -> Option<T> {
        let i = v.iter().position(|x| x.0 == s)?;
        Some(v.swap_remove(i).1)
    }

    /// Destroys everything parked.
    pub(crate) unsafe fn free(self, d: &ash::Device, vq: &ash::khr::video_queue::Device) {
        for (_, b) in self.body {
            unsafe { b.free(d, vq) };
        }
        for (_, o) in self.out {
            unsafe { o.free(d) };
        }
    }
}

/// A decoder's output: the NV12 (8-bit) or P010 (10-bit) textures and the two timelines. Share it (`Arc`) with the
/// thread that samples.
pub struct Ring {
    tex: Vec<wgpu::Texture>,
    imgs: Vec<vk::Image>,      // their images (wgpu owns them)
    views: Vec<vk::ImageView>, // and decode views
    pub done: vk::Semaphore,
    pub free: vk::Semaphore,
    // `free` values: the highest staged, the highest a submit has taken. Staging under the lock keeps at most one,
    // rising, value in wgpu's list.
    rel: Mutex<[u64; 2]>,
    // its shape once whole, and where it parks
    shape: Option<Shape>,
    bin: Weak<Mutex<Bin>>,
    dev: ash::Device,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Ring {
    /// The texture that holds frame `f`. Its format is the stream's, the same for every frame: NV12, sampled through
    /// R8Unorm / Rg8Unorm plane views, or P010 (H.265 Main 10), sampled through R16Unorm / Rg16Unorm plane views that
    /// read the 10-bit sample `d` as `64·d / 65535` (P010 keeps it in the high bits, the low 6 are zero).
    pub fn texture(&self, f: u64) -> &wgpu::Texture {
        &self.tex[(f % self.tex.len() as u64) as usize]
    }

    /// P010 (H.265 Main 10) rather than NV12.
    pub fn deep(&self) -> bool {
        self.tex[0].format() == wgpu::TextureFormat::P010
    }

    /// Frame `f`'s luma and chroma (half size) plane views: R8Unorm / Rg8Unorm, or R16Unorm / Rg16Unorm for P010.
    /// What both backends have: macOS keeps the planes in textures of their own.
    pub fn planes(&self, f: u64) -> [wgpu::TextureView; 2] {
        let t = self.texture(f);
        [wgpu::TextureAspect::Plane0, wgpu::TextureAspect::Plane1].map(|aspect| {
            t.create_view(&wgpu::TextureViewDescriptor {
                aspect,
                ..Default::default()
            })
        })
    }

    /// Plane `i` of frame `f` (0 luma, 1 chroma) as a copy source or destination at the plane's origin; chroma
    /// coordinates are half the picture's.
    pub fn plane(&self, f: u64, i: usize) -> wgpu::TexelCopyTextureInfo<'_> {
        wgpu::TexelCopyTextureInfo {
            texture: self.texture(f),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: [wgpu::TextureAspect::Plane0, wgpu::TextureAspect::Plane1][i],
        }
    }

    /// Frame `f`'s textures (here its one NV12 / P010 texture), e.g. to transition them back home.
    pub fn textures(&self, f: u64) -> impl Iterator<Item = &wgpu::Texture> {
        std::iter::once(self.texture(f))
    }

    /// Its frames at once: frame `f` is in slot `f % slots()` ([`OUT`] unless [`Decoder::with_out`] asked for more).
    pub fn slots(&self) -> usize {
        self.tex.len()
    }

    /// Frames fully decoded: frames below this are ready to sample without stalling wgpu's queue.
    pub fn ready(&self) -> u64 {
        unsafe { self.dev.get_semaphore_counter_value(self.done) }.unwrap_or(0)
    }

    /// Call before each wgpu submit that samples frame `f` (a frame `decode` returned); a paused view acquires its
    /// frame again for every redraw. Stages the wait `done ≥ f + 1`. wgpu attaches staged semaphores to the next
    /// submit on its queue from any thread: when another thread's submit takes the wait first, it still orders this
    /// one, which comes later in submission order.
    pub fn acquire(&self, g: &Gpu, f: u64) {
        if let Some(q) = unsafe { g.queue.as_hal::<Vk>() } {
            q.add_wait_semaphore(self.done, Some(f + 1), vk::PipelineStageFlags::ALL_COMMANDS);
        }
    }

    /// Call after the last submit that samples frame `f`. Stages the signal `free = f + 1`: the decoder may then
    /// overwrite `f` and every earlier frame, so a consumer holding several frames releases them in order. The next
    /// submit from any thread carries the signal; with none (an idle UI), `decode` sends it with an empty submit once
    /// it needs the slot.
    pub fn release(&self, g: &Gpu, f: u64) {
        let Some(q) = (unsafe { g.queue.as_hal::<Vk>() }) else {
            return;
        };
        let mut r = self.rel.lock().unwrap_or_else(|e| e.into_inner());
        if r[0] <= f {
            // one signal per submit: two in one batch execute in either order, and the values must rise
            if !q.remove_signal_semaphore(self.free) {
                r[1] = r[0];
            }
            q.add_signal_semaphore(self.free, Some(f + 1));
            r[0] = f + 1;
        }
    }

    /// Sends a release still staged that reaches `free ≥ v` with an empty submit, unless one a submit has taken
    /// reaches it: wgpu 30 hands the staged semaphores to `vkQueueSubmit` also without command buffers.
    fn flush(&self, g: &Gpu, v: u64) {
        let Some(q) = (unsafe { g.queue.as_hal::<Vk>() }) else {
            return;
        };
        let mut r = self.rel.lock().unwrap_or_else(|e| e.into_inner());
        if r[0] < v || r[1] >= v {
            return;
        }
        // taking it out is the only way to see whether a submit has taken it since
        if !q.remove_signal_semaphore(self.free) {
            r[1] = r[0];
            return;
        }
        q.add_signal_semaphore(self.free, Some(r[0]));
        drop((q, r));
        g.queue.submit([]);
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        unsafe {
            // operations staged but never submitted must not reach a destroyed semaphore
            if let Some(q) = self.queue.as_hal::<Vk>() {
                while q.remove_wait_semaphore(self.done) {}
                while q.remove_signal_semaphore(self.free) {}
            }
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.dev.destroy_semaphore(self.done, None);
            self.dev.destroy_semaphore(self.free, None);
            let o = Out {
                tex: take(&mut self.tex),
                imgs: take(&mut self.imgs),
                views: take(&mut self.views),
            };
            let left = match (self.shape, self.bin.upgrade()) {
                (Some(s), Some(b)) => {
                    Bin::park(&mut b.lock().unwrap_or_else(|e| e.into_inner()).out, s, o)
                }
                _ => Some(o),
            };
            if let Some(o) = left {
                o.free(&self.dev);
            }
        }
    }
}

/// One stream's decoder. Lives on a media thread; `ring` (or the `Deref` to it) gives the textures, `acquire` and
/// `release`.
pub struct Decoder {
    pub ring: Arc<Ring>,
    /// texture size: the session's maximum coded extent
    pub w: u32,
    pub h: u32,
    /// display window (x, y, w, h) inside the texture: the SPS's cropping of the last picture decoded (or of the
    /// first SPS before that); it changes only with a new SPS
    pub view: [u32; 4],
    /// index of the next frame
    pub frame: u64,
    deep: bool,
    co: bool, // DPB and output coincide: a picture decodes into its DPB slot, then is copied into the ring
    st: St,
    refs: Vec<Ref>,
    dpb: Vec<(vk::Image, vk::ImageView)>,
    mem: Vec<vk::DeviceMemory>, // session, DPB and bitstream memory
    session: vk::VideoSessionKHR,
    params: vk::VideoSessionParametersKHR,
    bs: vk::Buffer,
    map: *mut u8,
    slot: usize,  // bitstream bytes per ring slot
    align: usize, // bitstream size alignment
    pool: vk::CommandPool,
    cbs: Vec<vk::CommandBuffer>,
    offs: Vec<u32>,
    reset: bool,
    qp: vk::QueryPool, // a result status query per ring slot, null where the decode family has none
    // frames whose status is checked, those of them not decoded (`failed`)
    chk: Cell<[u64; 2]>,
    // its shape once whole, and where its body parks
    shape: Option<Shape>,
    bin: Weak<Mutex<Bin>>,
    dev: ash::Device,
    vq: ash::khr::video_queue::Device,
}

// `map` points into host-coherent memory that only this decoder touches
unsafe impl Send for Decoder {}

impl std::ops::Deref for Decoder {
    type Target = Ring;
    fn deref(&self) -> &Ring {
        &self.ring
    }
}

/// The driver's decode capabilities for profile `pr` (H.265 when `h265`, else H.264), without a session: what
/// `Decoder::new` and `check` ask. The generic ones and the decode flags; the codec's own aren't used.
pub(crate) unsafe fn caps(
    g: &Gpu,
    pr: &vk::VideoProfileInfoKHR,
    h265: bool,
) -> Result<(
    vk::VideoCapabilitiesKHR<'static>,
    vk::VideoDecodeCapabilityFlagsKHR,
)> {
    let (mut c5, mut c4) = (
        vk::VideoDecodeH265CapabilitiesKHR::default(),
        vk::VideoDecodeH264CapabilitiesKHR::default(),
    );
    let mut dc = vk::VideoDecodeCapabilitiesKHR::default();
    let mut caps = vk::VideoCapabilitiesKHR::default();
    dc.p_next = match h265 {
        true => &mut c5 as *mut _ as *mut c_void,
        false => &mut c4 as *mut _ as *mut c_void,
    };
    caps.p_next = &mut dc as *mut _ as *mut c_void;
    unsafe { (g.vqi.fp().get_physical_device_video_capabilities_khr)(g.pd, pr, &mut caps) }
        .result()?;
    caps.p_next = ptr::null_mut();
    Ok((caps, dc.flags))
}

/// The DPB mode a decoder here takes from them, reference images apart either way: Some(false) DPB and output images
/// distinct, preferred; Some(true) only coinciding (NVIDIA, Intel), which needs transfer on the decode queue for the
/// copy into the ring; None neither.
pub(crate) fn coincide(
    g: &Gpu,
    caps: &vk::VideoCapabilitiesKHR,
    dc: vk::VideoDecodeCapabilityFlagsKHR,
) -> Option<bool> {
    use vk::VideoDecodeCapabilityFlagsKHR as D;
    let copy = || {
        let q = unsafe { g.inst.get_physical_device_queue_family_properties(g.pd) };
        q[g.dec.family as usize]
            .queue_flags
            .contains(vk::QueueFlags::TRANSFER)
    };
    match () {
        _ if !(caps.flags).contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES) => None,
        _ if dc.contains(D::DPB_AND_OUTPUT_DISTINCT) => Some(false),
        _ if dc.contains(D::DPB_AND_OUTPUT_COINCIDE) && copy() => Some(true),
        _ => None,
    }
}

/// The decode family reports a decode's result status (NVIDIA's does; RADV's doesn't, and a query there hangs).
fn status(g: &Gpu) -> bool {
    let n = unsafe {
        g.inst
            .get_physical_device_queue_family_properties2_len(g.pd)
    };
    let mut rs = vec![vk::QueueFamilyQueryResultStatusPropertiesKHR::default(); n];
    let mut qp: Vec<_> = (rs.iter_mut())
        .map(|r| vk::QueueFamilyProperties2::default().push_next(r))
        .collect();
    unsafe {
        g.inst
            .get_physical_device_queue_family_properties2(g.pd, &mut qp)
    };
    drop(qp);
    rs[g.dec.family as usize].query_result_status_support == vk::TRUE
}

fn barrier(
    img: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) -> vk::ImageMemoryBarrier<'static> {
    let rw = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
    vk::ImageMemoryBarrier::default()
        .src_access_mask(rw)
        .dst_access_mask(rw)
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(img)
        .subresource_range(RANGE)
}

impl Decoder {
    /// A decoder for `codec` with the parameter sets in `params` (VPS / SPS / PPS NAL units from `hvcC` / `avcC`
    /// or in band) and textures of `w` × `h`, grown to the SPS's coded size if that is larger. The first SPS's bit
    /// depth picks the textures' format (`Ring::texture`): 8 NV12, 10 P010; a picture of the other depth is an error.
    pub fn new<P: AsRef<[u8]>>(
        g: &Gpu,
        codec: Codec,
        params: &[P],
        w: u32,
        h: u32,
    ) -> Result<Decoder> {
        Self::with_out(g, codec, params, w, h, OUT)
    }

    /// [`Decoder::new`] with a ring of `out` frames (at least [`OUT`]): a consumer that shows B-frame video in
    /// presentation order keeps frames that decode before ones it shows first.
    pub fn with_out<P: AsRef<[u8]>>(
        g: &Gpu,
        codec: Codec,
        params: &[P],
        w: u32,
        h: u32,
        out: usize,
    ) -> Result<Decoder> {
        let out = out.max(OUT);
        let mut st = match codec {
            Codec::H264 => St::H264(Default::default()),
            Codec::H265 => St::H265(Default::default()),
        };
        for p in params.iter().map(AsRef::as_ref).filter(|p| p.len() > 2) {
            st.put(p)?;
        }
        // profile, coded size, display window, DPB slots (references, the current picture, one spare) and bit depth
        // from the first SPS
        let (idc, cw, ch, view, slots, deep) = match &st {
            St::H264(s) => {
                let s = s.sps.iter().flatten().next().ok_or(Bad("no SPS"))?;
                (
                    s.profile_idc as u32,
                    s.width_mbs * 16,
                    s.height_mbs * 16,
                    s.view(),
                    s.max_num_ref_frames as usize + 2,
                    false,
                )
            }
            St::H265(s) => {
                let s = s.sps.iter().flatten().next().ok_or(Bad("no SPS"))?;
                (
                    s.ptl.profile_idc as u32,
                    s.width,
                    s.height,
                    s.view(),
                    s.dpbm.max_dec_minus1[s.max_sub as usize] as usize + 2,
                    s.depth > 8,
                )
            }
        };
        let (w, h, h265) = (w.max(cw), h.max(ch), codec == Codec::H265);
        let (fmt, tf, bits) = match deep {
            true => (
                P010,
                wgpu::TextureFormat::P010,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
            ),
            false => (
                NV12,
                wgpu::TextureFormat::NV12,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            ),
        };
        if !g.device.features().contains(tf.required_features()) {
            return Err(Bad("wgpu lacks the output texture format"));
        }
        let ext = vk::Extent2D {
            width: w,
            height: h,
        };
        unsafe {
            let mut p5 = vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(idc);
            let mut p4 = vk::VideoDecodeH264ProfileInfoKHR::default()
                .std_profile_idc(idc)
                .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
            let pr = vk::VideoProfileInfoKHR::default()
                .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
                .luma_bit_depth(bits)
                .chroma_bit_depth(bits);
            let pr = [match h265 {
                true => pr
                    .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
                    .push_next(&mut p5),
                false => pr
                    .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
                    .push_next(&mut p4),
            }];
            let list = vk::VideoProfileListInfoKHR::default().profiles(&pr);
            let list = &list as *const _ as *const c_void;

            let (caps, dc) = caps(g, &pr[0], h265)?;
            let co =
                coincide(g, &caps, dc).ok_or(Bad("the driver lacks a DPB mode we decode in"))?;
            if w > caps.max_coded_extent.width || h > caps.max_coded_extent.height {
                return Err(Bad("picture larger than the driver decodes"));
            }
            let slots = slots.min(caps.max_dpb_slots as usize).min(MAXS);
            let hd = g.device.as_hal::<Vk>().ok_or(Bad("not a Vulkan device"))?;
            let shape = (codec, idc, deep, w, h, slots, out);
            let bin = || g.bin.lock().unwrap_or_else(|e| e.into_inner());

            // from here every object belongs to `ring`, `d` or wgpu as soon as it exists: an error drops it
            let image = |usage, flags, fams: &[u32]| -> Result<(vk::Image, vk::DeviceMemory)> {
                let mut info = vk::ImageCreateInfo::default()
                    .flags(flags)
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(fmt)
                    .extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                if fams.len() > 1 {
                    info = info
                        .sharing_mode(vk::SharingMode::CONCURRENT)
                        .queue_family_indices(fams);
                }
                info.p_next = list;
                let img = g.dev.create_image(&info, None)?;
                let m = g
                    .alloc(
                        g.dev.get_image_memory_requirements(img),
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                    .and_then(|m| match g.dev.bind_image_memory(img, m, 0) {
                        Ok(()) => Ok(m),
                        Err(e) => {
                            g.dev.free_memory(m, None);
                            Err(e.into())
                        }
                    });
                if m.is_err() {
                    g.dev.destroy_image(img, None);
                }
                Ok((img, m?))
            };
            let iv = |img, usage| {
                let mut u = vk::ImageViewUsageCreateInfo::default().usage(usage);
                let info = vk::ImageViewCreateInfo::default()
                    .image(img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(fmt)
                    .subresource_range(RANGE)
                    .push_next(&mut u);
                g.dev.create_image_view(&info, None)
            };
            let sem = || {
                let mut t = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE);
                g.dev
                    .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut t), None)
            };
            let o = Bin::take(&mut bin().out, shape).unwrap_or(Out {
                tex: vec![],
                imgs: vec![],
                views: vec![],
            });
            let mut ring = Ring {
                tex: o.tex,
                imgs: o.imgs,
                views: o.views,
                done: vk::Semaphore::null(),
                free: vk::Semaphore::null(),
                rel: Mutex::new([0; 2]),
                shape: None,
                bin: Arc::downgrade(&g.bin),
                dev: g.dev.clone(),
                device: g.device.clone(),
                queue: g.queue.clone(),
            };
            ring.done = sem()?;
            ring.free = sem()?;
            let size = wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            };
            let dst = match co {
                true => vk::ImageUsageFlags::TRANSFER_DST,
                false => vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
            };
            while ring.tex.len() < out {
                let (img, m) = image(
                    dst | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE,
                    &[g.gfx, g.dec.family],
                )?;
                let desc = wgpu_hal::TextureDescriptor {
                    label: Some("decoded"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: tf,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
                    memory_flags: wgpu_hal::MemoryFlags::empty(),
                    view_formats: vec![],
                };
                // wgpu owns image and memory from here and frees them once the texture is unused
                let hal = hd.texture_from_raw(
                    img,
                    &desc,
                    None,
                    wgpu_hal::vulkan::TextureMemory::Dedicated(m),
                );
                let desc = wgpu::TextureDescriptor {
                    label: Some("decoded"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: tf,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                };
                ring.tex.push(g.device.create_texture_from_hal::<Vk>(
                    hal,
                    &desc,
                    wgpu::TextureUses::RESOURCE,
                ));
                ring.imgs.push(img);
                if !co {
                    ring.views.push(iv(img, dst)?);
                }
            }
            ring.shape = Some(shape);

            let mut d = Decoder {
                ring: Arc::new(ring),
                w,
                h,
                view,
                frame: 0,
                deep,
                co,
                st,
                refs: vec![Ref::default(); slots],
                dpb: vec![],
                mem: vec![],
                session: vk::VideoSessionKHR::null(),
                params: vk::VideoSessionParametersKHR::null(),
                bs: vk::Buffer::null(),
                map: ptr::null_mut(),
                slot: BS.next_multiple_of(caps.min_bitstream_buffer_offset_alignment.max(1))
                    as usize,
                align: caps.min_bitstream_buffer_size_alignment.max(1) as usize,
                pool: vk::CommandPool::null(),
                cbs: vec![],
                offs: vec![],
                reset: true,
                qp: vk::QueryPool::null(),
                chk: Cell::new([0; 2]),
                shape: None,
                bin: Arc::downgrade(&g.bin),
                dev: g.dev.clone(),
                vq: g.vq.clone(),
            };
            if let Some(b) = Bin::take(&mut bin().body, shape) {
                (d.dpb, d.mem, d.session, d.bs, d.map, d.pool, d.cbs, d.qp) =
                    (b.dpb, b.mem, b.session, b.bs, b.map, b.pool, b.cbs, b.qp);
                d.params(g)?;
                d.shape = Some(shape);
                return Ok(d);
            }
            let info = vk::VideoSessionCreateInfoKHR::default()
                .queue_family_index(g.dec.family)
                .video_profile(&pr[0])
                .picture_format(fmt)
                .max_coded_extent(ext)
                .reference_picture_format(fmt)
                .max_dpb_slots(slots as u32)
                .max_active_reference_pictures(
                    (slots as u32 - 1).min(caps.max_active_reference_pictures),
                )
                .std_header_version(&caps.std_header_version);
            let (vq, dh) = (g.vq.fp(), g.dev.handle());
            (vq.create_video_session_khr)(dh, &info, ptr::null(), &mut d.session).result()?;
            let mut n = 0;
            (vq.get_video_session_memory_requirements_khr)(dh, d.session, &mut n, ptr::null_mut())
                .result()?;
            let mut reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); n as usize];
            (vq.get_video_session_memory_requirements_khr)(
                dh,
                d.session,
                &mut n,
                reqs.as_mut_ptr(),
            )
            .result()?;
            let mut binds = vec![];
            for r in &reqs {
                // a bind may allow no device-local type at all; then any type it allows
                let q = r.memory_requirements;
                d.mem.push(
                    g.alloc(q, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                        .or_else(|_| g.alloc(q, vk::MemoryPropertyFlags::empty()))?,
                );
                binds.push(
                    vk::BindVideoSessionMemoryInfoKHR::default()
                        .memory_bind_index(r.memory_bind_index)
                        .memory(d.mem[d.mem.len() - 1])
                        .memory_size(r.memory_requirements.size),
                );
            }
            (vq.bind_video_session_memory_khr)(dh, d.session, binds.len() as u32, binds.as_ptr())
                .result()?;
            // coinciding: a slot is also the decode's output and the copy's source
            let (u, x) = match co {
                true => (
                    vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                        | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
                    vk::ImageUsageFlags::TRANSFER_SRC,
                ),
                false => (
                    vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
                    vk::ImageUsageFlags::empty(),
                ),
            };
            for i in 0..slots {
                let (img, m) = image(u | x, vk::ImageCreateFlags::empty(), &[])?;
                d.mem.push(m);
                d.dpb.push((img, vk::ImageView::null()));
                d.dpb[i].1 = iv(img, u)?;
            }

            let mut info = vk::BufferCreateInfo::default()
                .size((d.slot * out) as u64)
                .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR);
            info.p_next = list;
            d.bs = g.dev.create_buffer(&info, None)?;
            d.mem.push(g.alloc(
                g.dev.get_buffer_memory_requirements(d.bs),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
            let m = d.mem[d.mem.len() - 1];
            g.dev.bind_buffer_memory(d.bs, m, 0)?;
            d.map = g
                .dev
                .map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                as *mut u8;
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(g.dec.family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            d.pool = g.dev.create_command_pool(&info, None)?;
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(d.pool)
                .command_buffer_count(out as u32);
            d.cbs = g.dev.allocate_command_buffers(&info)?;
            if status(g) {
                let mut info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::RESULT_STATUS_ONLY_KHR)
                    .query_count(out as u32);
                info.p_next = &pr[0] as *const _ as *const c_void;
                d.qp = g.dev.create_query_pool(&info, None)?;
            }
            d.params(g)?;
            d.warm(g)?;
            d.shape = Some(shape);
            Ok(d)
        }
    }

    /// Grows the command buffers' upload memory before any decode: RADV starts it at 16 KB on a buffer's first use
    /// and doubles it whenever a command does not fit (a coding control and a decode take `embedded_size` there each,
    /// over 16 KB on VCN), freeing the outgrown ones at the buffer's next reset, i.e. during the first `2 · OUT`
    /// decodes while the decoder runs ahead (`Bin`). Recorded, never submitted: as many coding controls as a first
    /// decode makes uploads (reset, decode) and one more; the reset here frees the outgrown ones.
    unsafe fn warm(&self, g: &Gpu) -> Result<()> {
        let (vq, d) = (g.vq.fp(), &self.dev);
        let begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.params);
        let reset =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        unsafe {
            for &cb in &self.cbs {
                d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
                (vq.cmd_begin_video_coding_khr)(cb, &begin);
                for _ in 0..3 {
                    (vq.cmd_control_video_coding_khr)(cb, &reset);
                }
                (vq.cmd_end_video_coding_khr)(cb, &vk::VideoEndCodingInfoKHR::default());
                d.end_command_buffer(cb)?;
            }
            d.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
        }
        Ok(())
    }

    /// Waits until every semaphore in `s` reaches `v`.
    unsafe fn wait(&self, s: &[vk::Semaphore], v: u64, ns: u64) -> Result<bool> {
        match unsafe {
            self.dev.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(s)
                    .values(&[v; 2][..s.len()]),
                ns,
            )
        } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// (Re)creates the session parameters from every stored parameter set.
    unsafe fn params(&mut self, g: &Gpu) -> Result<()> {
        let (vq, dh) = (g.vq.fp(), g.dev.handle());
        let mut p = vk::VideoSessionParametersKHR::null();
        let mut info =
            vk::VideoSessionParametersCreateInfoKHR::default().video_session(self.session);
        let r = match &self.st {
            St::H265(s) => {
                let vps: Vec<_> = s.vps.iter().flatten().map(h265::Vps::std).collect();
                let sps: Vec<_> = s.sps.iter().flatten().map(h265::Sps::std).collect();
                let vps_id = |p: &h265::Pps| Some(s.sps.get(p.sps_id as usize)?.as_ref()?.vps_id);
                let pps: Vec<_> = s
                    .pps
                    .iter()
                    .flatten()
                    .filter_map(|p| Some(p.std(vps_id(p)?)))
                    .collect();
                let (v, q): (Vec<_>, Vec<_>) = (
                    vps.iter().map(|x| x.raw).collect(),
                    sps.iter().map(|x| x.raw).collect(),
                );
                let add = vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
                    .std_vp_ss(&v)
                    .std_sp_ss(&q)
                    .std_pp_ss(&pps);
                let c = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
                    .max_std_vps_count(v.len() as u32)
                    .max_std_sps_count(q.len() as u32)
                    .max_std_pps_count(pps.len() as u32)
                    .parameters_add_info(&add);
                info.p_next = &c as *const _ as *const c_void;
                unsafe { (vq.create_video_session_parameters_khr)(dh, &info, ptr::null(), &mut p) }
            }
            St::H264(s) => {
                let sps: Vec<_> = s.sps.iter().flatten().map(h264::Sps::std).collect();
                let pps: Vec<_> = s.pps.iter().flatten().map(h264::Pps::std).collect();
                let add = vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
                    .std_sp_ss(&sps)
                    .std_pp_ss(&pps);
                let c = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
                    .max_std_sps_count(sps.len() as u32)
                    .max_std_pps_count(pps.len() as u32)
                    .parameters_add_info(&add);
                info.p_next = &c as *const _ as *const c_void;
                unsafe { (vq.create_video_session_parameters_khr)(dh, &info, ptr::null(), &mut p) }
            }
        };
        r.result()?;
        if self.params != vk::VideoSessionParametersKHR::null() {
            // decodes in flight still use the old object
            unsafe {
                self.wait(&[self.ring.done], self.frame, u64::MAX)?;
                (vq.destroy_video_session_parameters_khr)(dh, self.params, ptr::null());
            }
        }
        self.params = p;
        Ok(())
    }

    /// Pictures that did not decode, counted once their decode is done: here the decodes the driver signalled done
    /// unfinished, whose result status isn't complete (NVIDIA's RTX 4090 on 620.73 now and then: cut off at some CTU,
    /// the rest zero), where the decode family reports one. The pictures after one reference it up to the next IDR.
    /// A picture the driver refuses up front is an error (`decode`). macOS conceals instead and counts them.
    pub fn failed(&self) -> u64 {
        self.verify();
        self.chk.get()[1]
    }

    /// Checks the result status of every decode done since the last check: before its query slot's next reset.
    fn verify(&self) {
        let [mut c, mut n] = self.chk.get();
        let (r, done) = (self.ring.slots() as u64, self.ring.ready().min(self.frame));
        while self.qp != vk::QueryPool::null() && c < done {
            let (k, mut v) = ((c % r) as u32, [0i32]);
            let f = vk::QueryResultFlags::WITH_STATUS_KHR;
            // not ready after its decode is done: the driver never finished it
            let ok = unsafe { self.dev.get_query_pool_results(self.qp, k, &mut v, f) };
            n += (ok.is_err() || v[0] != vk::QueryResultStatusKHR::COMPLETE.as_raw()) as u64;
            c += 1;
        }
        self.chk.set([c, n]);
    }

    /// Decoding restarts at the next access unit, a random access point (a seek), as in a new decoder: the session is
    /// reset and the DPB emptied. An H.265 CRA there is handled as a BLA (HandleCraAsBlaFlag): its RASL pictures,
    /// which reference pictures before it, are `Error::Skip`. So are the pictures shown before an H.264 non-IDR I
    /// picture there.
    pub fn restart(&mut self) {
        // the session holds state of the device's own (VCN): with only the DPB emptied, H.264 pictures some frames
        // after a restart still came out wrong, without an error
        self.reset = true;
        self.refs.fill(Ref::default());
        match &mut self.st {
            St::H264(s) => s.restart(),
            St::H265(s) => s.restart(),
        }
    }

    /// Decodes one access unit (the NAL units of one sample: its slices plus any in-band parameter sets) into
    /// texture `frame % slots()` and returns the frame index. Waits up to 50 ms for the consumer to release that slot
    /// (`free`; a release no submit has carried yet goes out with an empty one); then returns `Error::Full` with
    /// nothing changed. A leading picture that may reference pictures before where decoding started or restarted
    /// (`restart`) is `Error::Skip`, no frame.
    pub fn decode(&mut self, g: &Gpu, au: &[&[u8]]) -> Result<u64> {
        let (f, r) = (self.frame, self.ring.slots() as u64);
        let k = (f % r) as usize;
        // the consumer released frame f - r and its decode is done (a frame released without being sampled may
        // still be decoding): its texture, bitstream slot and command buffer are free
        let (s, v) = ([self.ring.free, self.ring.done], (f + 1).saturating_sub(r));
        if f >= r && !unsafe { self.wait(&s, v, 0)? } {
            self.ring.flush(g, v);
            if !unsafe { self.wait(&s, v, WAIT)? } {
                return Err(Error::Full);
            }
        }
        self.verify();
        let h265 = matches!(self.st, St::H265(_));
        let mut dirty = false;
        for n in au {
            if n.len() > 2 && self.st.kind(n) == 2 {
                dirty |= self.st.put(n)?;
            }
        }
        if dirty {
            unsafe { self.params(g)? };
        }

        // bitstream: start code + NAL unit for every slice
        self.offs.clear();
        let (base, mut o) = (unsafe { self.map.add(k * self.slot) }, 0);
        for n in au.iter().filter(|n| n.len() > 2 && self.st.kind(n) == 1) {
            if o + 3 + n.len() + self.align > self.slot {
                return Err(Bad("access unit too large"));
            }
            self.offs.push(o as u32);
            unsafe {
                ptr::copy_nonoverlapping([0u8, 0, 1].as_ptr(), base.add(o), 3);
                ptr::copy_nonoverlapping(n.as_ptr(), base.add(o + 3), n.len());
            }
            o += 3 + n.len();
        }
        if o == 0 {
            return Err(Bad("no slices"));
        }
        let range = o.next_multiple_of(self.align);
        unsafe { ptr::write_bytes(base.add(o), 0, range - o) };

        let mut act = [0u8; MAXS];
        let (mut p5, mut p4): (StdVideoDecodeH265PictureInfo, StdVideoDecodeH264PictureInfo) =
            unsafe { std::mem::zeroed() };
        let pic = match &mut self.st {
            St::H265(s) => s.picture(au, &mut self.refs, &mut p5, &mut act)?,
            St::H264(s) => s.picture(au, &mut self.refs, &mut p4, &mut act)?,
        };
        if pic.w > self.w || pic.h > self.h {
            return Err(Bad("picture larger than the decoder"));
        }
        if pic.deep != self.deep {
            return Err(Bad("picture bit depth other than the decoder's"));
        }
        self.view = pic.view;

        // reference slots: the active ones, then the setup slot (slot index -1 in begin, its index in decode)
        let ext = vk::Extent2D {
            width: pic.w,
            height: pic.h,
        };
        let (mut r5, mut r4): (
            [StdVideoDecodeH265ReferenceInfo; MAXS],
            [StdVideoDecodeH264ReferenceInfo; MAXS],
        ) = unsafe { std::mem::zeroed() };
        let mut d5 = [vk::VideoDecodeH265DpbSlotInfoKHR::default(); MAXS];
        let mut d4 = [vk::VideoDecodeH264DpbSlotInfoKHR::default(); MAXS];
        let mut res = [vk::VideoPictureResourceInfoKHR::default(); MAXS];
        let mut sl = [vk::VideoReferenceSlotInfoKHR::default(); MAXS];
        for (j, i) in act[..pic.n]
            .iter()
            .map(|&i| i as usize)
            .chain([pic.setup])
            .enumerate()
        {
            let r = self.refs[i];
            r5[j].PicOrderCntVal = r.poc[0];
            (r4[j].FrameNum, r4[j].PicOrderCnt) = (r.num as u16, r.poc);
            r4[j].flags.set_used_for_long_term_reference(r.long as u32);
            d5[j].p_std_reference_info = &r5[j];
            d4[j].p_std_reference_info = &r4[j];
            res[j] = vk::VideoPictureResourceInfoKHR::default()
                .coded_extent(ext)
                .image_view_binding(self.dpb[i].1);
            sl[j] = vk::VideoReferenceSlotInfoKHR {
                p_next: if h265 {
                    &d5[j] as *const _ as *const c_void
                } else {
                    &d4[j] as *const _ as *const c_void
                },
                slot_index: i as i32,
                p_picture_resource: &res[j],
                ..Default::default()
            };
        }
        let n = pic.n;
        let setup = sl[n];
        sl[n].slot_index = -1;
        let reset = std::mem::replace(&mut self.reset, false);
        let i5 = vk::VideoDecodeH265PictureInfoKHR::default()
            .std_picture_info(&p5)
            .slice_segment_offsets(&self.offs);
        let i4 = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&p4)
            .slice_offsets(&self.offs);
        let mut dec = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bs)
            .src_buffer_offset((k * self.slot) as u64)
            .src_buffer_range(range as u64)
            .dst_picture_resource(match self.co {
                true => res[n],
                false => vk::VideoPictureResourceInfoKHR::default()
                    .coded_extent(ext)
                    .image_view_binding(self.ring.views[k]),
            })
            .setup_reference_slot(&setup)
            .reference_slots(&sl[..n]);
        dec.p_next = if h265 {
            &i5 as *const _ as *const c_void
        } else {
            &i4 as *const _ as *const c_void
        };

        let (cb, all, d) = (self.cbs[k], vk::PipelineStageFlags::ALL_COMMANDS, &self.dev);
        let (vq, none) = (g.vq.fp(), vk::DependencyFlags::empty());
        unsafe {
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            // the output image leaves home (still UNDEFINED before its first decode); the setup slot's old
            // picture is dead. The global barrier makes the earlier decodes' DPB writes visible to this one's
            // reference reads: an image barrier's access scopes cover only its own image.
            let home = match f < r {
                true => vk::ImageLayout::UNDEFINED,
                false => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let (out, s) = (self.ring.imgs[k], self.dpb[pic.setup].0);
            let (dpb, src) = (
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            let to = match self.co {
                true => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                false => vk::ImageLayout::VIDEO_DECODE_DST_KHR,
            };
            let bars = [
                barrier(out, home, to),
                barrier(s, vk::ImageLayout::UNDEFINED, dpb),
            ];
            let mem = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
            d.cmd_pipeline_barrier(cb, all, all, none, &mem, &[], &bars);
            let begin = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.params)
                .reference_slots(&sl[..=n]);
            let (qp, q) = (self.qp, self.qp != vk::QueryPool::null());
            if q {
                d.cmd_reset_query_pool(cb, qp, k as u32, 1);
            }
            (vq.cmd_begin_video_coding_khr)(cb, &begin);
            if reset {
                (vq.cmd_control_video_coding_khr)(
                    cb,
                    &vk::VideoCodingControlInfoKHR::default()
                        .flags(vk::VideoCodingControlFlagsKHR::RESET),
                );
            }
            if q {
                d.cmd_begin_query(cb, qp, k as u32, vk::QueryControlFlags::empty());
            }
            (g.vdq.fp().cmd_decode_video_khr)(cb, &dec);
            if q {
                d.cmd_end_query(cb, qp, k as u32);
            }
            (vq.cmd_end_video_coding_khr)(cb, &vk::VideoEndCodingInfoKHR::default());
            if self.co {
                // the slot goes back to the DPB layout after: layout transitions keep its picture
                d.cmd_pipeline_barrier(cb, all, all, none, &[], &[], &[barrier(s, dpb, src)]);
                let cp = [
                    (vk::ImageAspectFlags::PLANE_0, 1),
                    (vk::ImageAspectFlags::PLANE_1, 2),
                ]
                .map(|(aspect_mask, q)| {
                    let l = vk::ImageSubresourceLayers {
                        aspect_mask,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    vk::ImageCopy::default()
                        .src_subresource(l)
                        .dst_subresource(l)
                        .extent(vk::Extent3D {
                            width: pic.w / q,
                            height: pic.h / q,
                            depth: 1,
                        })
                });
                d.cmd_copy_image(cb, s, src, out, to, &cp);
            }
            let back = [
                barrier(out, to, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                barrier(s, src, dpb),
            ];
            let nb = 1 + self.co as usize;
            d.cmd_pipeline_barrier(cb, all, all, none, &[], &[], &back[..nb]);
            d.end_command_buffer(cb)?;

            let nw = (f >= r) as usize;
            let (ws, wv, wst) = ([self.ring.free], [v], [all]);
            let (ss, sv, cbs) = ([self.ring.done], [f + 1], [cb]);
            let mut tl = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wv[..nw])
                .signal_semaphore_values(&sv);
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&ws[..nw])
                .wait_dst_stage_mask(&wst[..nw])
                .command_buffers(&cbs)
                .signal_semaphores(&ss)
                .push_next(&mut tl);
            let q = g.dec.queue.lock().unwrap_or_else(|e| e.into_inner());
            d.queue_submit(*q, &[submit], vk::Fence::null())?;
        }
        self.frame += 1;
        Ok(f)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            // every submitted decode can finish: decode() submits only once its slot is released
            let idle = self.wait(&[self.ring.done], self.frame, u64::MAX);
            (self.vq.fp().destroy_video_session_parameters_khr)(
                self.dev.handle(),
                self.params,
                ptr::null(),
            );
            let b = Body {
                dpb: take(&mut self.dpb),
                mem: take(&mut self.mem),
                session: self.session,
                bs: self.bs,
                map: self.map,
                pool: self.pool,
                cbs: take(&mut self.cbs),
                qp: self.qp,
            };
            let left = match (self.shape, self.bin.upgrade()) {
                (Some(s), Some(bin)) if idle.is_ok_and(|x| x) => Bin::park(
                    &mut bin.lock().unwrap_or_else(|e| e.into_inner()).body,
                    s,
                    b,
                ),
                _ => Some(b),
            };
            if let Some(b) = left {
                b.free(&self.dev, &self.vq);
            }
        }
    }
}
