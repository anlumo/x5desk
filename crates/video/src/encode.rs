//! Vulkan Video encode (H.265 Main / Main 10, H.264 High) from a ring of NV12 / P010 images that wgpu fills; the
//! decoder's rules (PLAN.md §2.1) mirrored:
//! - An input image's home state is COPY_DST (TRANSFER_DST_OPTIMAL), wgpu's initial state for it. The encode queue
//!   moves it to VIDEO_ENCODE_SRC, encodes and moves it back; wgpu only copies into it.
//! - Input images are CONCURRENT across wgpu's family and the encode family. The DPB is one layered image (RADV has
//!   no separate reference images for encode), private to the encoder.
//! - Timeline `filled`: the last wgpu submit that writes frame f signals f + 1 (`Input::release`). Timeline `done`:
//!   the encode of frame f signals f + 1. The first submit that writes f waits for `done ≥ f + 1 − SLOTS`
//!   (`Input::acquire`), the encode of f for `filled ≥ f + 1`; the host reads f's bitstream after `done ≥ f + 1`.
//! - GOP: an IDR every `gop` frames or on request (`Input::key`), P-frames in between with one reference, the
//!   previous picture. No B-frames (RADV 26.2 has them for H.264 only), so no reordering and dts = pts.
pub use crate::stream::{Cfg, Packet, Rate};
use crate::{
    Error::{self, Bad},
    Gpu, Result, bitw,
    decode::Codec,
};
use ash::vk::{self, native::*};
use std::{
    ffi::c_void,
    ops::Range,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
};
use wgpu_hal::api::Vulkan as Vk;

/// Input ring size: how far the producer runs ahead of the encoder. Also the bitstream, query and command buffer
/// ring, so at most this many frames are in flight before `Encoder::packet` must collect one.
pub const SLOTS: usize = 4;
const WAIT: u64 = 50_000_000; // ns acquire() waits for its slot before returning Full
const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
const P010: vk::Format = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
const ALL: vk::PipelineStageFlags = vk::PipelineStageFlags::ALL_COMMANDS;
const DPB: u32 = 2; // the reference and the picture being encoded
// NVIDIA's (620.73) default quality levels, H.264 and H.265, where `Cfg::quality` 0 lands: its own 0, the fastest,
// already matches RADV's 0, and these add 0.05 to 0.25 dB PSNR(Y) on the app's exports at 0.55 to 0.7 of the encoder's
// speed (H.265 3: 136 fps at 3840 × 1920, H.264 2: 149), more than the 4090's exports or a 4K60 stream need; H.265's
// 0 to 2 write the same stream here (they differ in the B-frames we don't use)
const NV: [u32; 2] = [2, 3];

/// NAL units of an Annex B buffer, without start codes and trailing zero bytes (a NAL unit never ends in 0).
fn split(b: &[u8], out: &mut Vec<Range<usize>>) {
    out.clear();
    let (mut i, mut s) = (0, None);
    let end = |s: Option<usize>, mut e: usize, out: &mut Vec<Range<usize>>| {
        if let Some(s) = s {
            while e > s && b[e - 1] == 0 {
                e -= 1;
            }
            if e > s {
                out.push(s..e);
            }
        }
    };
    while i + 2 < b.len() {
        if b[i + 2] > 1 {
            i += 3;
        } else if b[i..i + 3] == [0, 0, 1] {
            end(s, i, out);
            (i, s) = (i + 3, Some(i + 3));
        } else {
            i += 1;
        }
    }
    end(s, b.len(), out);
}

fn wait(dev: &ash::Device, s: vk::Semaphore, v: u64, ns: u64) -> Result<bool> {
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(std::slice::from_ref(&s))
        .values(std::slice::from_ref(&v));
    match unsafe { dev.wait_semaphores(&info, ns) } {
        Ok(()) => Ok(true),
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

fn barrier(
    img: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
    layers: u32,
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
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        })
}

fn mem_barrier(src: vk::AccessFlags, dst: vk::AccessFlags) -> [vk::MemoryBarrier<'static>; 1] {
    [vk::MemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst)]
}

/// The encoder's input: NV12 / P010 textures and the two timelines. Share it (`Arc`) with the thread that submits
/// the wgpu work that fills it.
pub struct Input {
    tex: Vec<wgpu::Texture>,
    imgs: Vec<vk::Image>,      // their images (wgpu owns them)
    views: Vec<vk::ImageView>, // and encode views
    pub filled: vk::Semaphore,
    pub done: vk::Semaphore,
    /// picture size; the textures may be larger (the encoder's input granularity)
    pub w: u32,
    pub h: u32,
    // `filled` values: the highest staged, the highest a submit has taken (as `Ring::rel`)
    rel: Mutex<[u64; 2]>,
    key: AtomicBool,
    dev: ash::Device,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Input {
    /// The NV12 / P010 texture for frame `f`: COPY_DST, written through its plane aspects (`copy_buffer_to_texture`
    /// with a 256-byte row pitch) or with `copy_buffer` / `copy_texture`.
    pub fn texture(&self, f: u64) -> &wgpu::Texture {
        &self.tex[(f % SLOTS as u64) as usize]
    }

    /// Call before the first wgpu submit that writes frame `f`. Waits up to 50 ms for the encode of frame
    /// `f − SLOTS` (the slot's last use) to finish, then returns `Error::Full` with nothing staged; otherwise stages
    /// the wait `done ≥ f + 1 − SLOTS` (already reached: it orders the copies after the encode's reads and layout
    /// changes, and never stalls wgpu's queue).
    pub fn acquire(&self, g: &Gpu, f: u64) -> Result<()> {
        let Some(v) = (f + 1).checked_sub(SLOTS as u64).filter(|&v| v > 0) else {
            return Ok(());
        };
        if !wait(&self.dev, self.done, v, WAIT)? {
            return Err(Error::Full);
        }
        if let Some(q) = unsafe { g.queue.as_hal::<Vk>() } {
            q.add_wait_semaphore(self.done, Some(v), ALL);
        }
        Ok(())
    }

    /// Call after the last wgpu submit that writes frame `f`, then hand `f` to the encoder thread. Stages the signal
    /// `filled = f + 1` for the next submit from any thread; `Encoder::encode` sends it with an empty submit if none
    /// has carried it by then. Frames are released in order.
    pub fn release(&self, g: &Gpu, f: u64) {
        let Some(q) = (unsafe { g.queue.as_hal::<Vk>() }) else {
            return;
        };
        let mut r = self.rel.lock().unwrap_or_else(|e| e.into_inner());
        if r[0] <= f {
            // one signal per submit: two in one batch execute in either order, and the values must rise
            if !q.remove_signal_semaphore(self.filled) {
                r[1] = r[0];
            }
            q.add_signal_semaphore(self.filled, Some(f + 1));
            r[0] = f + 1;
        }
    }

    /// Makes the next encoded frame an IDR, from any thread (an RTMP reconnect). The GOP restarts there; a request
    /// that meets a scheduled IDR (or frame 0) adds none.
    pub fn key(&self) {
        self.key.store(true, Relaxed);
    }

    /// Whether `filled ≥ v` has been staged; sends it with an empty submit unless a submit has taken it.
    fn flush(&self, g: &Gpu, v: u64) -> bool {
        let Some(q) = (unsafe { g.queue.as_hal::<Vk>() }) else {
            return false;
        };
        let mut r = self.rel.lock().unwrap_or_else(|e| e.into_inner());
        if r[0] < v || r[1] >= v {
            return r[0] >= v;
        }
        // taking it out is the only way to see whether a submit has taken it since
        if !q.remove_signal_semaphore(self.filled) {
            r[1] = r[0];
            return true;
        }
        q.add_signal_semaphore(self.filled, Some(r[0]));
        drop((q, r));
        g.queue.submit([]);
        true
    }

    /// A raw command buffer for wgpu's queue around `rec`: earlier writes (the producer's shaders) are visible to
    /// the transfer, and the transfer's reads and writes finish before anything later on the queue.
    fn record(
        &self,
        g: &Gpu,
        rec: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<wgpu::CommandBuffer> {
        let mut e = g.device.create_command_encoder(&Default::default());
        let ok = unsafe {
            e.as_hal_mut::<Vk, _, _>(|e| {
                let Some(e) = e else { return false };
                let (cb, d, none) = (e.raw_handle(), &self.dev, vk::DependencyFlags::empty());
                let t = vk::PipelineStageFlags::TRANSFER;
                let (r, w) = (vk::AccessFlags::MEMORY_READ, vk::AccessFlags::MEMORY_WRITE);
                d.cmd_pipeline_barrier(cb, ALL, t, none, &mem_barrier(w, r | w), &[], &[]);
                rec(d, cb);
                d.cmd_pipeline_barrier(cb, t, ALL, none, &mem_barrier(w, r | w), &[], &[]);
                true
            })
        };
        if !ok {
            return Err(Bad("not a Vulkan device"));
        }
        Ok(e.finish())
    }

    /// A command buffer for wgpu's queue that copies an NV12 / P010 picture from `buf` into frame `f`'s texture:
    /// `h` rows of Y at `off`, then `h / 2` rows of interleaved CbCr, rows `pitch` bytes apart (crate stitch's
    /// `Yuv` gives `buf` and `pitch`: the width rounded up to 4 px, times 2 for P010). Any pitch the texel size divides: the copy is raw Vulkan, without wgpu's
    /// 256-byte row rule. Submit it after the work that writes `buf`, between `acquire` and `release`; wgpu does not
    /// see this use of `buf`, so keep it alive until the submit is done.
    pub fn copy_buffer(
        &self,
        g: &Gpu,
        buf: &wgpu::Buffer,
        off: u64,
        pitch: u32,
        f: u64,
    ) -> Result<wgpu::CommandBuffer> {
        let (w, h) = (self.w, self.h);
        let b = 1 + (self.tex[0].format() == wgpu::TextureFormat::P010) as u32; // bytes per sample
        let need = off + pitch as u64 * (h + h / 2) as u64;
        if pitch < w * b
            || !pitch.is_multiple_of(2 * b)
            || !off.is_multiple_of(2 * b as u64)
            || need > buf.size()
        {
            return Err(Bad("buffer layout does not fit the picture"));
        }
        let src = unsafe { buf.as_hal::<Vk>() }
            .map(|b| unsafe { b.raw_handle() })
            .ok_or(Bad("not a Vulkan buffer"))?;
        let img = self.imgs[(f % SLOTS as u64) as usize];
        let reg = |aspect, o, s| vk::BufferImageCopy {
            buffer_offset: off + o,
            buffer_row_length: pitch / b / s,
            buffer_image_height: h / s,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: w / s,
                height: h / s,
                depth: 1,
            },
        };
        let regs = [
            reg(vk::ImageAspectFlags::PLANE_0, 0, 1),
            reg(vk::ImageAspectFlags::PLANE_1, (pitch * h) as u64, 2),
        ];
        self.record(g, |d, cb| unsafe {
            d.cmd_copy_buffer_to_image(cb, src, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regs)
        })
    }

    /// A command buffer for wgpu's queue that copies the rectangle `[x, y, w, h]` (even) of an NV12 / P010 texture
    /// in the RESOURCE state (a decoder's output: acquire and release it around the submit as usual) to `[dx, dy]`
    /// in frame `f`'s texture. Submit it between `acquire` and `release`.
    pub fn copy_texture(
        &self,
        g: &Gpu,
        src: &wgpu::Texture,
        [x, y, w, h]: [u32; 4],
        [dx, dy]: [u32; 2],
        f: u64,
    ) -> Result<wgpu::CommandBuffer> {
        let dst = &self.tex[0];
        if src.format() != dst.format()
            || (x | y | w | h | dx | dy) & 1 != 0
            || x + w > src.width()
            || y + h > src.height()
            || dx + w > dst.width()
            || dy + h > dst.height()
        {
            return Err(Bad("copy outside the textures or between formats"));
        }
        let si = unsafe { src.as_hal::<Vk>() }
            .map(|t| unsafe { t.raw_handle() })
            .ok_or(Bad("not a Vulkan texture"))?;
        let img = self.imgs[(f % SLOTS as u64) as usize];
        let reg = |aspect, s| {
            let l = vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let o = |x: u32, y: u32| vk::Offset3D {
                x: (x / s) as i32,
                y: (y / s) as i32,
                z: 0,
            };
            vk::ImageCopy {
                src_subresource: l,
                src_offset: o(x, y),
                dst_subresource: l,
                dst_offset: o(dx, dy),
                extent: vk::Extent3D {
                    width: w / s,
                    height: h / s,
                    depth: 1,
                },
            }
        };
        let regs = [
            reg(vk::ImageAspectFlags::PLANE_0, 1),
            reg(vk::ImageAspectFlags::PLANE_1, 2),
        ];
        use vk::ImageLayout as L;
        self.record(g, |d, cb| unsafe {
            let none = vk::DependencyFlags::empty();
            let to = [barrier(
                si,
                L::SHADER_READ_ONLY_OPTIMAL,
                L::TRANSFER_SRC_OPTIMAL,
                1,
            )];
            d.cmd_pipeline_barrier(cb, ALL, ALL, none, &[], &[], &to);
            d.cmd_copy_image(
                cb,
                si,
                L::TRANSFER_SRC_OPTIMAL,
                img,
                L::TRANSFER_DST_OPTIMAL,
                &regs,
            );
            let back = [barrier(
                si,
                L::TRANSFER_SRC_OPTIMAL,
                L::SHADER_READ_ONLY_OPTIMAL,
                1,
            )];
            d.cmd_pipeline_barrier(cb, ALL, ALL, none, &[], &[], &back);
        })
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        unsafe {
            // operations staged but never submitted must not reach a destroyed semaphore
            if let Some(q) = self.queue.as_hal::<Vk>() {
                while q.remove_wait_semaphore(self.done) {}
                while q.remove_signal_semaphore(self.filled) {}
            }
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            self.dev.destroy_semaphore(self.filled, None);
            self.dev.destroy_semaphore(self.done, None);
            // the textures (fields) go after their views
            for &v in &self.views {
                self.dev.destroy_image_view(v, None);
            }
        }
    }
}

// Table A.8 from level 3: MaxLumaPs, MaxLumaSr, MaxBR main / high tier (kbit/s). Index + 3 = StdVideoH265LevelIdc.
const L5: [(u64, u64, u64, u64); 10] = [
    (552960, 16588800, 6000, 6000),
    (983040, 33177600, 10000, 10000),
    (2228224, 66846720, 12000, 30000),
    (2228224, 133693440, 20000, 50000),
    (8912896, 267386880, 25000, 100000),
    (8912896, 534773760, 40000, 160000),
    (8912896, 1069547520, 60000, 240000),
    (35651584, 1069547520, 60000, 240000),
    (35651584, 2139095040, 120000, 480000),
    (35651584, 4278190080, 240000, 800000),
];
// Table A-1 from level 3: MaxMBPS, MaxFS, MaxBR (kbit/s, × 1.25 for High), level_idc. Index + 7 =
// StdVideoH264LevelIdc.
const L4: [(u64, u64, u64, u8); 12] = [
    (40500, 1620, 10000, 30),
    (108000, 3600, 14000, 31),
    (216000, 5120, 20000, 32),
    (245760, 8192, 20000, 40),
    (245760, 8192, 50000, 41),
    (522240, 8704, 50000, 42),
    (589824, 22080, 135000, 50),
    (983040, 36864, 240000, 51),
    (2073600, 36864, 240000, 52),
    (4177920, 139264, 240000, 60),
    (8355840, 139264, 480000, 61),
    (16711680, 139264, 800000, 62),
];

/// One stream's encoder. Lives on its own thread; the producer fills `input` on whichever thread submits its wgpu
/// work and tells this thread each frame it released.
pub struct Encoder {
    pub input: Arc<Input>,
    /// Parameter sets (H.265 VPS, SPS, PPS; H.264 SPS, PPS), NAL units without start codes: `hvcC` / `avcC`, or in
    /// band before a key frame. As the driver writes them, except an H.264 SPS the driver left as given: that one is
    /// written here (see `new`).
    pub params: Vec<Vec<u8>>,
    /// Quality levels usable: the driver's, but at most 3 on RADV for H.265 under rate control: level 3 (the
    /// firmware's high quality preset) writes corrupt slice data there (Mesa 26.2.5; FFmpeg's `hevc_vulkan` as well;
    /// constant QP and H.264 decode clean), and levels above it are untested.
    pub levels: u32,
    /// Index of the next frame to encode.
    pub frame: u64,
    level: u32, // the quality level in use
    c: Cfg,
    ext: vk::Extent2D, // coded size
    idr: u64,          // the last IDR's frame
    idr_id: u16,       // H.264 idr_pic_id
    next: u64,         // the next packet
    meta: [(i64, bool); SLOTS],
    session: vk::VideoSessionKHR,
    sp: vk::VideoSessionParametersKHR,
    dpb: vk::Image,
    dpbv: vk::ImageView,
    bs: vk::Buffer,
    map: *const u8,
    slot: usize, // bitstream bytes per ring slot
    query: vk::QueryPool,
    pool: vk::CommandPool,
    cbs: Vec<vk::CommandBuffer>,
    mem: Vec<vk::DeviceMemory>, // session, DPB and bitstream memory
    dev: ash::Device,
    vq: ash::khr::video_queue::Device,
    veq: ash::khr::video_encode_queue::Device,
}

// `map` points into host-coherent memory that only this encoder touches
unsafe impl Send for Encoder {}

impl Encoder {
    /// An encoder on the encode queue of `g` for `c`. Fails when the driver can't: no encode queue or codec, no NV12 /
    /// P010 encode input or DPB images, a size outside its coded extents, a rate control mode, bitrate, QP or level
    /// beyond its capabilities.
    pub fn new(g: &Gpu, c: &Cfg) -> Result<Encoder> {
        let q = g.enc.as_ref().ok_or(Bad("no video encode queue"))?;
        let h265 = c.codec == Codec::H265;
        let op = match h265 {
            true => vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            false => vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
        };
        if !q.ops.contains(op) {
            return Err(Bad("the encode queue lacks the codec"));
        }
        if c.deep && !h265 {
            return Err(Bad("10-bit encode is H.265 only"));
        }
        if (c.w | c.h) & 1 != 0
            || c.w == 0
            || c.h == 0
            || c.fps.contains(&0)
            || c.gop == 0
            || c.vbv == 0 && !matches!(c.rate, Rate::Qp(_))
            || c.chroma > 5
        {
            return Err(Bad(
                "odd or zero size, zero frame rate, GOP or rate control buffer, chroma location above 5",
            ));
        }
        let (fmt, depth, tf) = match c.deep {
            true => (
                P010,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
                wgpu::TextureFormat::P010,
            ),
            false => (
                NV12,
                vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
                wgpu::TextureFormat::NV12,
            ),
        };
        if !g.device.features().contains(tf.required_features()) {
            return Err(Bad("wgpu lacks the input texture format"));
        }
        // coded size: H.265 in 8 × 8 minimum coding blocks, H.264 in macroblocks; cropped back to w × h
        let a = if h265 { 8 } else { 16 };
        let ext = vk::Extent2D {
            width: c.w.next_multiple_of(a),
            height: c.h.next_multiple_of(a),
        };
        let peak = match c.rate {
            Rate::Cbr(b) => b,
            Rate::Vbr(a, p) => p.max(a),
            Rate::Qp(_) => 0,
        };
        let mode = match c.rate {
            Rate::Cbr(_) => vk::VideoEncodeRateControlModeFlagsKHR::CBR,
            Rate::Vbr(..) => vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            Rate::Qp(_) => vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
        };
        unsafe {
            let mut p5 =
                vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(1 + c.deep as u32);
            let mut p4 = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(100);
            let pr = vk::VideoProfileInfoKHR::default()
                .video_codec_operation(op)
                .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
                .luma_bit_depth(depth)
                .chroma_bit_depth(depth);
            let pr = [match h265 {
                true => pr.push_next(&mut p5),
                false => pr.push_next(&mut p4),
            }];
            let list = vk::VideoProfileListInfoKHR::default().profiles(&pr);
            let list = &list as *const _ as *const c_void;

            let (mut c5, mut c4) = (
                vk::VideoEncodeH265CapabilitiesKHR::default(),
                vk::VideoEncodeH264CapabilitiesKHR::default(),
            );
            let mut ec = vk::VideoEncodeCapabilitiesKHR::default();
            let mut caps = vk::VideoCapabilitiesKHR::default();
            ec.p_next = match h265 {
                true => &mut c5 as *mut _ as *mut c_void,
                false => &mut c4 as *mut _ as *mut c_void,
            };
            caps.p_next = &mut ec as *mut _ as *mut c_void;
            (g.vqi.fp().get_physical_device_video_capabilities_khr)(g.pd, &pr[0], &mut caps)
                .result()?;
            let (mn, mx) = (caps.min_coded_extent, caps.max_coded_extent);
            if ext.width > mx.width
                || ext.height > mx.height
                || ext.width < mn.width
                || ext.height < mn.height
            {
                return Err(Bad("size outside the driver's encode range"));
            }
            if caps.max_dpb_slots < DPB || caps.max_active_reference_pictures < 1 {
                return Err(Bad("the driver encodes no P-frames"));
            }
            if !ec.rate_control_modes.contains(mode) {
                return Err(Bad("rate control mode unsupported"));
            }
            if peak > ec.max_bitrate {
                return Err(Bad("bitrate above the driver's maximum"));
            }
            let fb = vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN;
            if !ec.supported_encode_feedback_flags.contains(fb) {
                return Err(Bad("no encode feedback"));
            }
            let qps = if h265 {
                c5.min_qp..=c5.max_qp
            } else {
                c4.min_qp..=c4.max_qp
            };
            if let Rate::Qp(qp) = c.rate
                && !qps.contains(&(qp as i32))
            {
                return Err(Bad("QP outside the driver's range"));
            }
            let fp = g.vqi.fp().get_physical_device_video_format_properties_khr;
            for u in [
                vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
                vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
            ] {
                let mut fi = vk::PhysicalDeviceVideoFormatInfoKHR::default().image_usage(u);
                fi.p_next = list;
                let (mut n, mut v) = (0, vec![]);
                if fp(g.pd, &fi, &mut n, ptr::null_mut()) == vk::Result::SUCCESS {
                    v.resize(n as usize, vk::VideoFormatPropertiesKHR::default());
                    fp(g.pd, &fi, &mut n, v.as_mut_ptr()).result()?;
                    v.truncate(n as usize);
                }
                if !v.iter().any(|p| p.format == fmt) {
                    return Err(Bad("the driver lacks NV12 / P010 as encode input or DPB"));
                }
            }
            let mut dp = vk::PhysicalDeviceDriverProperties::default();
            g.inst.get_physical_device_properties2(
                g.pd,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut dp),
            );
            let levels = match dp.driver_id == vk::DriverId::MESA_RADV
                && h265
                && !matches!(c.rate, Rate::Qp(_))
            {
                true => ec.max_quality_levels.min(3),
                false => ec.max_quality_levels,
            };
            let quality = (c.quality)
                .saturating_add(match dp.driver_id {
                    vk::DriverId::NVIDIA_PROPRIETARY => NV[h265 as usize],
                    _ => 0,
                })
                .min(levels.saturating_sub(1));
            let info = vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR::default()
                .video_profile(&pr[0])
                .quality_level(quality);
            let mut qp = vk::VideoEncodeQualityLevelPropertiesKHR::default();
            (g.veqi
                .fp()
                .get_physical_device_video_encode_quality_level_properties_khr)(
                g.pd, &info, &mut qp,
            )
            .result()?;

            // the lowest level (and tier) whose limits hold the coded picture, its rate and the peak bitrate
            let (cw, ch) = (ext.width as u64, ext.height as u64);
            let rate = |n: u64| n * c.fps[0] as u64 / c.fps[1] as u64;
            let (level, tier) = if h265 {
                let ps = cw * ch;
                L5.iter()
                    .enumerate()
                    .find_map(|(i, &(p, s, m, hi))| {
                        let fits = ps <= p && cw.max(ch).pow(2) <= 8 * p && rate(ps) <= s;
                        let t = [m, hi].iter().position(|&m| peak <= m * 1000);
                        t.filter(|_| fits).map(|t| (i as u32 + 3, t as u32))
                    })
                    .filter(|&(l, _)| l <= c5.max_level_idc)
            } else {
                let (mw, mh) = (cw / 16, ch / 16);
                L4.iter()
                    .position(|&(mbps, fs, br, _)| {
                        mw * mh <= fs
                            && mw.max(mh).pow(2) <= 8 * fs
                            && rate(mw * mh) <= mbps
                            && peak <= br * 1250
                    })
                    .map(|i| (i as u32 + 7, 0))
                    .filter(|&(l, _)| l <= c4.max_level_idc)
            }
            .ok_or(Bad("needs a level above the driver's maximum"))?;
            let hd = g.device.as_hal::<Vk>().ok_or(Bad("not a Vulkan device"))?;

            // from here every object belongs to `input`, `e` or wgpu as soon as it exists: an error drops it
            let gr = (
                caps.picture_access_granularity
                    .width
                    .max(ec.encode_input_picture_granularity.width),
                caps.picture_access_granularity
                    .height
                    .max(ec.encode_input_picture_granularity.height),
            );
            let (iw, ih) = (
                ext.width.next_multiple_of(gr.0.max(1)),
                ext.height.next_multiple_of(gr.1.max(1)),
            );
            let image = |usage, layers, fams: &[u32]| -> Result<(vk::Image, vk::DeviceMemory)> {
                let mut info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(fmt)
                    .extent(vk::Extent3D {
                        width: iw,
                        height: ih,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(layers)
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
            let iv = |img, ty, layers| {
                let info = vk::ImageViewCreateInfo::default()
                    .image(img)
                    .view_type(ty)
                    .format(fmt)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: layers,
                    });
                g.dev.create_image_view(&info, None)
            };
            let sem = || {
                let mut t = vk::SemaphoreTypeCreateInfo::default()
                    .semaphore_type(vk::SemaphoreType::TIMELINE);
                g.dev
                    .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut t), None)
            };
            let mut input = Input {
                tex: vec![],
                imgs: vec![],
                views: vec![],
                filled: vk::Semaphore::null(),
                done: vk::Semaphore::null(),
                w: c.w,
                h: c.h,
                rel: Mutex::new([0; 2]),
                key: AtomicBool::new(false),
                dev: g.dev.clone(),
                device: g.device.clone(),
                queue: g.queue.clone(),
            };
            input.filled = sem()?;
            input.done = sem()?;
            let size = wgpu::Extent3d {
                width: iw,
                height: ih,
                depth_or_array_layers: 1,
            };
            for _ in 0..SLOTS {
                let (img, m) = image(
                    vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST,
                    1,
                    &[g.gfx, q.family],
                )?;
                let desc = wgpu_hal::TextureDescriptor {
                    label: Some("encode input"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: tf,
                    usage: wgpu::TextureUses::COPY_DST,
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
                    label: Some("encode input"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: tf,
                    usage: wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                };
                input.tex.push(g.device.create_texture_from_hal::<Vk>(
                    hal,
                    &desc,
                    wgpu::TextureUses::COPY_DST,
                ));
                input.imgs.push(img);
                input.views.push(iv(img, vk::ImageViewType::TYPE_2D, 1)?);
            }

            // the uncompressed 4:2:0 picture (12 or 15 bits a pixel): RADV puts an incompressible key picture at
            // 0.85 bytes a pixel and reports a picture past the slot as complete
            let slot = (cw * ch * [12, 15][c.deep as usize] / 8)
                .max(1 << 20)
                .next_multiple_of(
                    caps.min_bitstream_buffer_offset_alignment
                        .max(caps.min_bitstream_buffer_size_alignment)
                        .max(4096),
                ) as usize;
            let mut e = Encoder {
                input: Arc::new(input),
                params: vec![],
                levels,
                level: quality,
                frame: 0,
                c: *c,
                ext,
                idr: 0,
                idr_id: 0,
                next: 0,
                meta: [(0, false); SLOTS],
                session: vk::VideoSessionKHR::null(),
                sp: vk::VideoSessionParametersKHR::null(),
                dpb: vk::Image::null(),
                dpbv: vk::ImageView::null(),
                bs: vk::Buffer::null(),
                map: ptr::null(),
                slot,
                query: vk::QueryPool::null(),
                pool: vk::CommandPool::null(),
                cbs: vec![],
                mem: vec![],
                dev: g.dev.clone(),
                vq: g.vq.clone(),
                veq: g.veq.clone(),
            };
            let mut sc5 = vk::VideoEncodeH265SessionCreateInfoKHR::default();
            let mut info = vk::VideoSessionCreateInfoKHR::default()
                .queue_family_index(q.family)
                .flags(vk::VideoSessionCreateFlagsKHR::ALLOW_ENCODE_PARAMETER_OPTIMIZATIONS)
                .video_profile(&pr[0])
                .picture_format(fmt)
                .max_coded_extent(ext)
                .reference_picture_format(fmt)
                .max_dpb_slots(DPB)
                .max_active_reference_pictures(1)
                .std_header_version(&caps.std_header_version);
            if h265 {
                info = info.push_next(&mut sc5);
            }
            let (vq, dh) = (g.vq.fp(), g.dev.handle());
            (vq.create_video_session_khr)(dh, &info, ptr::null(), &mut e.session).result()?;
            let mut n = 0;
            (vq.get_video_session_memory_requirements_khr)(dh, e.session, &mut n, ptr::null_mut())
                .result()?;
            let mut reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); n as usize];
            (vq.get_video_session_memory_requirements_khr)(
                dh,
                e.session,
                &mut n,
                reqs.as_mut_ptr(),
            )
            .result()?;
            let mut binds = vec![];
            for r in &reqs {
                // a bind may allow no device-local type at all; then any type it allows
                let q = r.memory_requirements;
                e.mem.push(
                    g.alloc(q, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                        .or_else(|_| g.alloc(q, vk::MemoryPropertyFlags::empty()))?,
                );
                binds.push(
                    vk::BindVideoSessionMemoryInfoKHR::default()
                        .memory_bind_index(r.memory_bind_index)
                        .memory(e.mem[e.mem.len() - 1])
                        .memory_size(r.memory_requirements.size),
                );
            }
            (vq.bind_video_session_memory_khr)(dh, e.session, binds.len() as u32, binds.as_ptr())
                .result()?;
            let (img, m) = image(vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR, DPB, &[])?;
            (e.dpb, e.dpbv) = (img, vk::ImageView::null());
            e.mem.push(m);
            e.dpbv = iv(img, vk::ImageViewType::TYPE_2D_ARRAY, DPB)?;

            // parameter sets: BT.709 colour, the chroma siting asked for, timing, one reference, no reordering
            let crop = [(ext.width - c.w) / 2, (ext.height - c.h) / 2]; // right, bottom: 4:2:0 units of 2
            let mut ptl: StdVideoH265ProfileTierLevel = std::mem::zeroed();
            ptl.flags.set_general_tier_flag(tier);
            ptl.flags.set_general_progressive_source_flag(1);
            ptl.flags.set_general_frame_only_constraint_flag(1);
            (ptl.general_profile_idc, ptl.general_level_idc) = (1 + c.deep as u32, level);
            let mut dpbm: StdVideoH265DecPicBufMgr = std::mem::zeroed();
            dpbm.max_dec_pic_buffering_minus1[0] = 1;
            let mut vps: StdVideoH265VideoParameterSet = std::mem::zeroed();
            vps.flags.set_vps_temporal_id_nesting_flag(1);
            vps.flags.set_vps_sub_layer_ordering_info_present_flag(1);
            vps.flags.set_vps_timing_info_present_flag(1);
            (vps.vps_num_units_in_tick, vps.vps_time_scale) = (c.fps[1], c.fps[0]);
            (vps.pDecPicBufMgr, vps.pProfileTierLevel) = (&dpbm, &ptl);
            let mut vui5: StdVideoH265SequenceParameterSetVui = std::mem::zeroed();
            let f = &mut vui5.flags;
            f.set_video_signal_type_present_flag(1);
            f.set_video_full_range_flag(c.full as u32);
            f.set_colour_description_present_flag(1);
            f.set_chroma_loc_info_present_flag(1);
            f.set_vui_timing_info_present_flag(1);
            vui5.video_format = 5; // unspecified
            (
                vui5.colour_primaries,
                vui5.transfer_characteristics,
                vui5.matrix_coeffs,
            ) = (1, 1, 1);
            (
                vui5.chroma_sample_loc_type_top_field,
                vui5.chroma_sample_loc_type_bottom_field,
            ) = (c.chroma, c.chroma);
            (vui5.vui_num_units_in_tick, vui5.vui_time_scale) = (c.fps[1], c.fps[0]);
            // CTB: the largest the driver has; transform blocks: its whole range
            let log2 = |b: u32| 31 - b.leading_zeros();
            let ctb = 4 + log2(c5.ctb_sizes.as_raw().max(1)) as u8;
            let tb = c5.transform_block_sizes.as_raw().max(1);
            let (tmin, tmax) = (2 + tb.trailing_zeros() as u8, 2 + log2(tb) as u8);
            let mut sps5: StdVideoH265SequenceParameterSet = std::mem::zeroed();
            let f = &mut sps5.flags;
            f.set_sps_temporal_id_nesting_flag(1);
            f.set_sps_sub_layer_ordering_info_present_flag(1);
            f.set_amp_enabled_flag(1);
            f.set_sample_adaptive_offset_enabled_flag(1);
            f.set_strong_intra_smoothing_enabled_flag(1);
            f.set_vui_parameters_present_flag(1);
            f.set_conformance_window_flag((crop != [0, 0]) as u32);
            sps5.chroma_format_idc =
                StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420;
            (
                sps5.pic_width_in_luma_samples,
                sps5.pic_height_in_luma_samples,
            ) = (ext.width, ext.height);
            sps5.bit_depth_luma_minus8 = 2 * c.deep as u8;
            sps5.bit_depth_chroma_minus8 = 2 * c.deep as u8;
            sps5.log2_max_pic_order_cnt_lsb_minus4 = 4;
            sps5.log2_diff_max_min_luma_coding_block_size = ctb - 3;
            sps5.log2_min_luma_transform_block_size_minus2 = tmin - 2;
            sps5.log2_diff_max_min_luma_transform_block_size = tmax - tmin;
            sps5.max_transform_hierarchy_depth_inter = 3.min(ctb - tmin);
            sps5.max_transform_hierarchy_depth_intra = ctb - tmin;
            (sps5.conf_win_right_offset, sps5.conf_win_bottom_offset) = (crop[0], crop[1]);
            (sps5.pProfileTierLevel, sps5.pDecPicBufMgr) = (&ptl, &dpbm);
            sps5.pSequenceParameterSetVui = &vui5;
            let mut pps5: StdVideoH265PictureParameterSet = std::mem::zeroed();
            pps5.flags.set_cu_qp_delta_enabled_flag(1);
            pps5.flags.set_transform_skip_enabled_flag(1);
            pps5.flags.set_pps_loop_filter_across_slices_enabled_flag(1);

            let mut vui4: StdVideoH264SequenceParameterSetVui = std::mem::zeroed();
            let f = &mut vui4.flags;
            f.set_video_signal_type_present_flag(1);
            f.set_video_full_range_flag(c.full as u32);
            f.set_color_description_present_flag(1);
            f.set_chroma_loc_info_present_flag(1);
            f.set_timing_info_present_flag(1);
            f.set_fixed_frame_rate_flag(1);
            // no reordering: players output each picture as soon as it is decoded
            f.set_bitstream_restriction_flag(1);
            vui4.video_format = 5;
            (
                vui4.colour_primaries,
                vui4.transfer_characteristics,
                vui4.matrix_coefficients,
            ) = (1, 1, 1);
            (vui4.num_units_in_tick, vui4.time_scale) = (c.fps[1], 2 * c.fps[0]);
            (vui4.max_num_reorder_frames, vui4.max_dec_frame_buffering) = (0, 1);
            (
                vui4.chroma_sample_loc_type_top_field,
                vui4.chroma_sample_loc_type_bottom_field,
            ) = (c.chroma, c.chroma);
            let mut sps4: StdVideoH264SequenceParameterSet = std::mem::zeroed();
            let f = &mut sps4.flags;
            f.set_direct_8x8_inference_flag(1);
            f.set_frame_mbs_only_flag(1);
            f.set_frame_cropping_flag((crop != [0, 0]) as u32);
            f.set_vui_parameters_present_flag(1);
            (sps4.profile_idc, sps4.level_idc) = (100, level);
            sps4.chroma_format_idc =
                StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
            (
                sps4.log2_max_frame_num_minus4,
                sps4.log2_max_pic_order_cnt_lsb_minus4,
            ) = (4, 4);
            sps4.pic_order_cnt_type = StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0;
            sps4.max_num_ref_frames = 1;
            sps4.pic_width_in_mbs_minus1 = ext.width / 16 - 1;
            sps4.pic_height_in_map_units_minus1 = ext.height / 16 - 1;
            (sps4.frame_crop_right_offset, sps4.frame_crop_bottom_offset) = (crop[0], crop[1]);
            sps4.pSequenceParameterSetVui = &vui4;
            let mut pps4: StdVideoH264PictureParameterSet = std::mem::zeroed();
            pps4.flags.set_entropy_coding_mode_flag(1);
            pps4.flags.set_deblocking_filter_control_present_flag(1);

            let mut ql = vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(quality);
            let a5 = vk::VideoEncodeH265SessionParametersAddInfoKHR::default()
                .std_vp_ss(std::slice::from_ref(&vps))
                .std_sp_ss(std::slice::from_ref(&sps5))
                .std_pp_ss(std::slice::from_ref(&pps5));
            let mut s5 = vk::VideoEncodeH265SessionParametersCreateInfoKHR::default()
                .max_std_vps_count(1)
                .max_std_sps_count(1)
                .max_std_pps_count(1)
                .parameters_add_info(&a5);
            let a4 = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
                .std_sp_ss(std::slice::from_ref(&sps4))
                .std_pp_ss(std::slice::from_ref(&pps4));
            let mut s4 = vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
                .max_std_sps_count(1)
                .max_std_pps_count(1)
                .parameters_add_info(&a4);
            let info = vk::VideoSessionParametersCreateInfoKHR::default()
                .video_session(e.session)
                .push_next(&mut ql);
            let info = match h265 {
                true => info.push_next(&mut s5),
                false => info.push_next(&mut s4),
            };
            (vq.create_video_session_parameters_khr)(dh, &info, ptr::null(), &mut e.sp).result()?;
            let g5 = vk::VideoEncodeH265SessionParametersGetInfoKHR::default()
                .write_std_vps(true)
                .write_std_sps(true)
                .write_std_pps(true);
            let g4 = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
                .write_std_sps(true)
                .write_std_pps(true);
            let mut gi = vk::VideoEncodeSessionParametersGetInfoKHR::default()
                .video_session_parameters(e.sp);
            gi.p_next = match h265 {
                true => &g5 as *const _ as *const c_void,
                false => &g4 as *const _ as *const c_void,
            };
            let get = g.veq.fp().get_encoded_video_session_parameters_khr;
            let mut fb4 = vk::VideoEncodeH264SessionParametersFeedbackInfoKHR::default();
            let mut fbi = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default();
            if !h265 {
                fbi.p_next = &mut fb4 as *mut _ as *mut c_void;
            }
            let mut n = 0;
            get(dh, &gi, &mut fbi, &mut n, ptr::null_mut()).result()?;
            let mut b = vec![0u8; n];
            get(
                dh,
                &gi,
                ptr::null_mut(),
                &mut n,
                b.as_mut_ptr() as *mut c_void,
            )
            .result()?;
            let mut r = vec![];
            split(&b[..n], &mut r);
            e.params = r.into_iter().map(|r| b[r].to_vec()).collect();
            // RADV fills the H.264 VUI fields the std structs lack with false limits (no motion vector across the
            // picture boundary, none longer than a quarter sample): unless the driver changed the SPS, it is written
            // here, the same but with those unlimited
            if !h265 && fb4.has_std_sps_overrides == 0 {
                let (s, v, mut w) = (&sps4, &vui4, bitw::W::default());
                w.u(8, 0x67)
                    .u(8, 100)
                    .u(8, 0)
                    .u(8, L4[level as usize - 7].3 as u32);
                // id, 4:2:0, 8 bits, no transform bypass, no scaling lists, frame_num and POC (type 0) lsb lengths
                w.ue(0).ue(1).ue(0).ue(0).f(false).f(false);
                w.ue(s.log2_max_frame_num_minus4 as u32).ue(0);
                w.ue(s.log2_max_pic_order_cnt_lsb_minus4 as u32);
                w.ue(s.max_num_ref_frames as u32).f(false);
                w.ue(s.pic_width_in_mbs_minus1)
                    .ue(s.pic_height_in_map_units_minus1);
                w.f(true).f(true).f(crop != [0, 0]); // frames only, direct 8x8 inference
                if crop != [0, 0] {
                    w.ue(0).ue(crop[0]).ue(0).ue(crop[1]);
                }
                // VUI: no aspect ratio or overscan; colour; chroma siting; timing; no HRD or pic_struct
                w.f(true).f(false).f(false);
                w.f(true).u(3, 5).f(c.full).f(true).u(8, 1).u(8, 1).u(8, 1);
                w.f(true).ue(c.chroma as u32).ue(c.chroma as u32);
                w.f(true)
                    .u(32, v.num_units_in_tick)
                    .u(32, v.time_scale)
                    .f(true);
                w.f(false).f(false).f(false);
                // bitstream restriction: MVs over the boundary, no size limits, log2 MV lengths 15, no reordering
                w.f(true).f(true).ue(0).ue(0).ue(15).ue(15).ue(0).ue(1);
                if let Some(p) = e.params.iter_mut().find(|p| p[0] & 31 == 7) {
                    *p = w.nal();
                }
            }

            let mut qf =
                vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default().encode_feedback_flags(fb);
            qf.p_next = &pr[0] as *const _ as *const c_void;
            let mut info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
                .query_count(SLOTS as u32);
            info.p_next = &qf as *const _ as *const c_void;
            e.query = g.dev.create_query_pool(&info, None)?;

            let mut info = vk::BufferCreateInfo::default()
                .size((e.slot * SLOTS) as u64)
                .usage(vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR);
            info.p_next = list;
            e.bs = g.dev.create_buffer(&info, None)?;
            // cached: the host reads every byte
            let (req, hv) = (
                g.dev.get_buffer_memory_requirements(e.bs),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            e.mem.push(
                g.alloc(req, hv | vk::MemoryPropertyFlags::HOST_CACHED)
                    .or_else(|_| g.alloc(req, hv))?,
            );
            let m = e.mem[e.mem.len() - 1];
            g.dev.bind_buffer_memory(e.bs, m, 0)?;
            e.map = g
                .dev
                .map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                as *const u8;
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(q.family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            e.pool = g.dev.create_command_pool(&info, None)?;
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(e.pool)
                .command_buffer_count(SLOTS as u32);
            e.cbs = g.dev.allocate_command_buffers(&info)?;

            // home states: inputs COPY_DST (wgpu writes them first), the DPB layers VIDEO_ENCODE_DPB for good
            use vk::ImageLayout as L;
            let cb = e.cbs[0];
            let mut bars: Vec<_> = e
                .input
                .imgs
                .iter()
                .map(|&i| barrier(i, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, 1))
                .collect();
            bars.push(barrier(e.dpb, L::UNDEFINED, L::VIDEO_ENCODE_DPB_KHR, DPB));
            g.dev
                .begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default())?;
            g.dev
                .cmd_pipeline_barrier(cb, ALL, ALL, vk::DependencyFlags::empty(), &[], &[], &bars);
            g.dev.end_command_buffer(cb)?;
            let fence = g.dev.create_fence(&Default::default(), None)?;
            let cbs = [cb];
            let r = {
                let q = q.queue.lock().unwrap_or_else(|e| e.into_inner());
                g.dev.queue_submit(
                    *q,
                    &[vk::SubmitInfo::default().command_buffers(&cbs)],
                    fence,
                )
            }
            .and_then(|_| g.dev.wait_for_fences(&[fence], true, u64::MAX));
            g.dev.destroy_fence(fence, None);
            r?;
            Ok(e)
        }
    }

    /// Encodes frame `self.frame`, which the producer has released (`Input::release`), and returns its index;
    /// `pts` comes back with its packet. Returns `Error::Full` with nothing submitted while `SLOTS` frames are in
    /// flight: collect one with `packet` first.
    pub fn encode(&mut self, g: &Gpu, pts: i64) -> Result<u64> {
        let (f, r) = (self.frame, SLOTS as u64);
        if f >= self.next + r {
            return Err(Error::Full);
        }
        // the producer's release reaches the GPU before the encode queue waits for it
        if !self.input.flush(g, f + 1) {
            return Err(Bad("encode of a frame not released"));
        }
        let (k, h265, c) = ((f % r) as usize, self.c.codec == Codec::H265, self.c);
        let prev_key = f > 0 && self.idr == f - 1;
        // the request is taken even when this frame is an IDR anyway, or it would make the next one an IDR too
        let req = self.input.key.swap(false, Relaxed);
        let key = req || f == 0 || f - self.idr >= c.gop as u64;
        if key {
            self.idr = f;
            self.idr_id = self.idr_id.wrapping_add(1);
        }
        // pictures since the IDR: H.265 PicOrderCntVal, H.264 frame_num (every picture is a reference)
        let poc = (f - self.idr) as i32;
        let (setup, refs) = ((f % 2) as u32, !key as usize);
        let qp = match c.rate {
            Rate::Qp(q) => q as i32,
            _ => 0,
        };

        let (mut pi5, mut sh5, mut rl5, mut rps): (
            StdVideoEncodeH265PictureInfo,
            StdVideoEncodeH265SliceSegmentHeader,
            StdVideoEncodeH265ReferenceListsInfo,
            StdVideoH265ShortTermRefPicSet,
        ) = unsafe { std::mem::zeroed() };
        let (mut pi4, mut sh4, mut rl4): (
            StdVideoEncodeH264PictureInfo,
            StdVideoEncodeH264SliceHeader,
            StdVideoEncodeH264ReferenceListsInfo,
        ) = unsafe { std::mem::zeroed() };
        let (mut ri5, mut ri4): (
            [StdVideoEncodeH265ReferenceInfo; 2],
            [StdVideoEncodeH264ReferenceInfo; 2],
        ) = unsafe { std::mem::zeroed() };
        // [0]: the picture being encoded (setup slot), [1]: its reference
        for (i, (p, k)) in [(poc, key), (poc - 1, prev_key)].into_iter().enumerate() {
            (ri5[i].pic_type, ri5[i].PicOrderCntVal) = (if k { 3 } else { 0 }, p); // IDR / P
            (ri4[i].primary_pic_type, ri4[i].FrameNum) = (if k { 5 } else { 0 }, p as u32 & 255);
            ri4[i].PicOrderCnt = 2 * p;
        }
        pi5.flags.set_is_reference(1);
        pi5.flags.set_IrapPicFlag(key as u32);
        pi5.flags.set_pic_output_flag(1);
        (pi5.pic_type, pi5.PicOrderCntVal) = (ri5[0].pic_type, poc);
        rl5.RefPicList0 = [0xff; 15];
        rl5.RefPicList1 = [0xff; 15];
        rl5.RefPicList0[0] = 1 - setup as u8;
        if !key {
            // short-term RPS in the slice header: the previous picture, used
            (rps.num_negative_pics, rps.used_by_curr_pic_s0_flag) = (1, 1);
            pi5.pRefLists = &rl5;
        }
        pi5.pShortTermRefPicSet = &rps;
        sh5.flags.set_first_slice_segment_in_pic_flag(1);
        sh5.flags.set_slice_sao_luma_flag(1);
        sh5.flags.set_slice_sao_chroma_flag(1);
        sh5.slice_type = if key { 2 } else { 1 }; // I / P
        sh5.MaxNumMergeCand = 5;
        pi4.flags.set_IdrPicFlag(key as u32);
        pi4.flags.set_is_reference(1);
        (pi4.idr_pic_id, pi4.primary_pic_type) = (self.idr_id, ri4[0].primary_pic_type);
        (pi4.frame_num, pi4.PicOrderCnt) = (ri4[0].FrameNum, ri4[0].PicOrderCnt);
        rl4.RefPicList0 = [0xff; 32];
        rl4.RefPicList1 = [0xff; 32];
        rl4.RefPicList0[0] = 1 - setup as u8;
        if !key {
            pi4.pRefLists = &rl4;
        }
        sh4.slice_type = if key { 2 } else { 0 }; // I / P
        let ns5 = [vk::VideoEncodeH265NaluSliceSegmentInfoKHR::default()
            .constant_qp(qp)
            .std_slice_segment_header(&sh5)];
        let ns4 = [vk::VideoEncodeH264NaluSliceInfoKHR::default()
            .constant_qp(qp)
            .std_slice_header(&sh4)];
        let e5 = vk::VideoEncodeH265PictureInfoKHR::default()
            .nalu_slice_segment_entries(&ns5)
            .std_picture_info(&pi5);
        let e4 = vk::VideoEncodeH264PictureInfoKHR::default()
            .nalu_slice_entries(&ns4)
            .std_picture_info(&pi4);
        let d5 = [0, 1]
            .map(|i| vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(&ri5[i]));
        let d4 = [0, 1]
            .map(|i| vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&ri4[i]));
        let res = [setup, 1 - setup].map(|l| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_extent(self.ext)
                .base_array_layer(l)
                .image_view_binding(self.dpbv)
        });
        let sl = [0, 1].map(|i| vk::VideoReferenceSlotInfoKHR {
            p_next: match h265 {
                true => &d5[i] as *const _ as *const c_void,
                false => &d4[i] as *const _ as *const c_void,
            },
            slot_index: [setup, 1 - setup][i] as i32,
            p_picture_resource: &res[i],
            ..Default::default()
        });
        let mut enc = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.bs)
            .dst_buffer_offset((k * self.slot) as u64)
            .dst_buffer_range(self.slot as u64)
            .src_picture_resource(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_extent(self.ext)
                    .image_view_binding(self.input.views[k]),
            )
            .setup_reference_slot(&sl[0])
            .reference_slots(&sl[1..1 + refs]);
        enc.p_next = match h265 {
            true => &e5 as *const _ as *const c_void,
            false => &e4 as *const _ as *const c_void,
        };
        // begin binds the reference, then the setup slot's picture without a slot (it is not active yet)
        let mut bsl = [sl[1], sl[0]];
        bsl[1].slot_index = -1;

        // rate control: set with the session reset, then restated by every begin
        let avg = match c.rate {
            Rate::Cbr(b) | Rate::Vbr(b, _) => b,
            Rate::Qp(_) => 0,
        };
        // no maxFrameSize: RADV (26.2.5, main of 2026-10) hands the spec's bytes to the firmware's max_au_size, which
        // counts bits (radeonsi passes VA-API's bits there), so every frame was clamped to an eighth and CBR padded
        // the rest with filler; scaled to bits it changed no frame (H.264 1080p at 100 and 250 ms, H.265 4K at 1000 ms)
        let layer = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(avg)
            .max_bitrate(match c.rate {
                Rate::Cbr(b) => b,
                Rate::Vbr(a, p) => p.max(a),
                Rate::Qp(_) => 0,
            })
            .frame_rate_numerator(c.fps[0])
            .frame_rate_denominator(c.fps[1]);
        // no QP limits; the spec leaves the codec's layer struct out, but NVIDIA (620.73) then rejects the control
        // command that sets the layer: vkEndCommandBuffer returns ERROR_INVALID_VIDEO_STD_PARAMETERS
        let (mut l5, mut l4) = (
            vk::VideoEncodeH265RateControlLayerInfoKHR::default(),
            vk::VideoEncodeH264RateControlLayerInfoKHR::default(),
        );
        let layer = [match h265 {
            true => layer.push_next(&mut l5),
            false => layer.push_next(&mut l4),
        }];
        let mut r5 = vk::VideoEncodeH265RateControlInfoKHR::default()
            .flags(
                vk::VideoEncodeH265RateControlFlagsKHR::REGULAR_GOP
                    | vk::VideoEncodeH265RateControlFlagsKHR::REFERENCE_PATTERN_FLAT,
            )
            .gop_frame_count(c.gop)
            .idr_period(c.gop)
            .sub_layer_count(1);
        let mut r4 = vk::VideoEncodeH264RateControlInfoKHR::default()
            .flags(
                vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP
                    | vk::VideoEncodeH264RateControlFlagsKHR::REFERENCE_PATTERN_FLAT,
            )
            .gop_frame_count(c.gop)
            .idr_period(c.gop)
            .temporal_layer_count(1);
        let mut rc = vk::VideoEncodeRateControlInfoKHR::default();
        match c.rate {
            Rate::Qp(_) => rc.rate_control_mode = vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
            _ => {
                rc = rc
                    .rate_control_mode(match c.rate {
                        Rate::Cbr(_) => vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                        _ => vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                    })
                    .layers(&layer)
                    .virtual_buffer_size_in_ms(c.vbv)
                    .initial_virtual_buffer_size_in_ms(c.vbv * 9 / 10);
                rc.p_next = match h265 {
                    true => &mut r5 as *mut _ as *mut c_void,
                    false => &mut r4 as *mut _ as *mut c_void,
                };
            }
        }
        let mut ql = vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(self.level);
        ql.p_next = &mut rc as *mut _ as *mut c_void;
        // the GOP's frames left, this one included: required where requiresGopRemainingFrames (not RADV, which ignores
        // them), a rate control hint elsewhere
        let (gi, gp) = (key as u32, c.gop - poc as u32 - key as u32);
        let mut g5 = vk::VideoEncodeH265GopRemainingFrameInfoKHR::default()
            .use_gop_remaining_frames(true)
            .gop_remaining_i(gi)
            .gop_remaining_p(gp);
        let mut g4 = vk::VideoEncodeH264GopRemainingFrameInfoKHR::default()
            .use_gop_remaining_frames(true)
            .gop_remaining_i(gi)
            .gop_remaining_p(gp);
        if f > 0 {
            (g5.p_next, g4.p_next) = (
                &rc as *const _ as *const c_void,
                &rc as *const _ as *const c_void,
            );
        }
        let mut begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.sp)
            .reference_slots(&bsl[1 - refs..]);
        begin.p_next = match h265 {
            true => &g5 as *const _ as *const c_void,
            false => &g4 as *const _ as *const c_void,
        };
        let mut ctl = vk::VideoCodingControlInfoKHR::default().flags(
            vk::VideoCodingControlFlagsKHR::RESET
                | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
                | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL,
        );
        ctl.p_next = &ql as *const _ as *const c_void;

        let (cb, d, none) = (self.cbs[k], &self.dev, vk::DependencyFlags::empty());
        let (vq, img) = (self.vq.fp(), self.input.imgs[k]);
        use vk::{AccessFlags as A, ImageLayout as L};
        unsafe {
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            // the global barrier makes the last encode's DPB write visible to this one's reference read: an image
            // barrier's access scopes cover only its own image
            let to = [barrier(
                img,
                L::TRANSFER_DST_OPTIMAL,
                L::VIDEO_ENCODE_SRC_KHR,
                1,
            )];
            let rw = A::MEMORY_READ | A::MEMORY_WRITE;
            d.cmd_pipeline_barrier(
                cb,
                ALL,
                ALL,
                none,
                &mem_barrier(A::MEMORY_WRITE, rw),
                &[],
                &to,
            );
            d.cmd_reset_query_pool(cb, self.query, k as u32, 1);
            (vq.cmd_begin_video_coding_khr)(cb, &begin);
            if f == 0 {
                (vq.cmd_control_video_coding_khr)(cb, &ctl);
            }
            d.cmd_begin_query(cb, self.query, k as u32, vk::QueryControlFlags::empty());
            (self.veq.fp().cmd_encode_video_khr)(cb, &enc);
            d.cmd_end_query(cb, self.query, k as u32);
            (vq.cmd_end_video_coding_khr)(cb, &vk::VideoEndCodingInfoKHR::default());
            let back = [barrier(
                img,
                L::VIDEO_ENCODE_SRC_KHR,
                L::TRANSFER_DST_OPTIMAL,
                1,
            )];
            d.cmd_pipeline_barrier(cb, ALL, ALL, none, &[], &[], &back);
            let host = vk::PipelineStageFlags::HOST;
            d.cmd_pipeline_barrier(
                cb,
                ALL,
                host,
                none,
                &mem_barrier(A::MEMORY_WRITE, A::HOST_READ),
                &[],
                &[],
            );
            d.end_command_buffer(cb)?;

            let (ws, wv, wst) = ([self.input.filled], [f + 1], [ALL]);
            let (ss, sv, cbs) = ([self.input.done], [f + 1], [cb]);
            let mut tl = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wv)
                .signal_semaphore_values(&sv);
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&ws)
                .wait_dst_stage_mask(&wst)
                .command_buffers(&cbs)
                .signal_semaphores(&ss)
                .push_next(&mut tl);
            let q = g.enc.as_ref().ok_or(Bad("no video encode queue"))?;
            let q = q.queue.lock().unwrap_or_else(|e| e.into_inner());
            d.queue_submit(*q, &[submit], vk::Fence::null())?;
        }
        self.meta[k] = (pts, key);
        self.frame += 1;
        Ok(f)
    }

    /// The oldest frame in flight into `p`, once encoded: waits up to `ns`, returns false when nothing is in flight
    /// or it timed out. An error is final (the frame stays the oldest, `encode` returns `Full` once the ring is
    /// full): the stream is broken from there, make a new encoder.
    pub fn packet(&mut self, p: &mut Packet, ns: u64) -> Result<bool> {
        let f = self.next;
        if f >= self.frame || !wait(&self.dev, self.input.done, f + 1, ns)? {
            return Ok(false);
        }
        let k = (f % SLOTS as u64) as usize;
        // 32-bit results: NVIDIA (620.73) writes offset and size as 32 bits even when asked for 64, the status at the
        // third 64-bit slot
        let mut r = [[0u32; 3]];
        unsafe {
            self.dev.get_query_pool_results(
                self.query,
                k as u32,
                &mut r,
                vk::QueryResultFlags::WITH_STATUS_KHR,
            )?;
        }
        let [off, n, st] = r[0].map(|v| v as u64);
        match st as i32 {
            1 => {}
            -1000299000 => return Err(Bad("picture larger than the bitstream buffer")),
            _ => return Err(Bad("encode failed")),
        }
        if off + n > self.slot as u64 {
            return Err(Bad("encode feedback outside the bitstream slot"));
        }
        let b = unsafe {
            std::slice::from_raw_parts(self.map.add(k * self.slot + off as usize), n as usize)
        };
        p.data.clear();
        p.data.extend_from_slice(b);
        split(&p.data, &mut p.nals);
        (p.frame, (p.pts, p.key)) = (f, self.meta[k]);
        p.dts = p.pts;
        self.next += 1;
        Ok(true)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            // every submitted encode can finish: encode() submits only after the producer's release went out
            let _ = wait(&self.dev, self.input.done, self.frame, u64::MAX);
            let (d, vq, dh) = (&self.dev, self.vq.fp(), self.dev.handle());
            (vq.destroy_video_session_parameters_khr)(dh, self.sp, ptr::null());
            (vq.destroy_video_session_khr)(dh, self.session, ptr::null());
            d.destroy_image_view(self.dpbv, None);
            d.destroy_image(self.dpb, None);
            d.destroy_query_pool(self.query, None);
            d.destroy_buffer(self.bs, None);
            d.destroy_command_pool(self.pool, None);
            for &m in &self.mem {
                d.free_memory(m, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb() {
        // 3- and 4-byte start codes, trailing zeros, an emulation prevention byte
        let b = [
            0, 0, 0, 1, 0x40, 1, 0, 0, 1, 0x26, 1, 0, 0, 3, 1, 9, 0, 0, 0, 0, 1, 2, 0,
        ];
        let mut p = Packet {
            data: b.to_vec(),
            ..Default::default()
        };
        split(&p.data, &mut p.nals);
        assert_eq!(p.nals, [4..6, 9..16, 21..22]);
        let mut o = vec![];
        p.prefixed(&mut o);
        assert_eq!(
            o,
            [
                0, 0, 0, 2, 0x40, 1, 0, 0, 0, 7, 0x26, 1, 0, 0, 3, 1, 9, 0, 0, 0, 1, 2
            ]
        );
        split(&[0, 0, 1], &mut p.nals);
        assert!(p.nals.is_empty());
    }
}
