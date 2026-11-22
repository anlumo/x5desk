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
//! - Where the decode family reports a decode's result status (NVIDIA), a decode submit signals the decoder's own
//!   timeline instead, and its checker thread signals `done` for a frame once its status is complete. A decode the
//!   driver signalled unfinished (620.73 on the RTX 4090 now and then: cut off at some CTU) is recorded and submitted
//!   again with every decode after it, before any of them reaches `done`, reading its bitstream from device memory;
//!   what they are made of is kept (`Rec`), and a DPB slot that an unchecked decode reads is not set up again until
//!   then (spare slots, `Decoder::phys`).
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
    collections::VecDeque,
    ffi::c_void,
    mem::take,
    ptr,
    sync::{Arc, Condvar, Mutex, MutexGuard, Weak},
    thread::JoinHandle,
};
use wgpu_hal::api::Vulkan as Vk;

const BS: u64 = 8 << 20; // bitstream bytes per access unit
const WAIT: u64 = 50_000_000; // ns decode() waits for its slot before returning Full
const PARK: usize = 2; // parked bodies and rings per shape: a deck's two lenses, while another deck opens
const TRIES: u8 = 3; // redos of a decode the driver leaves unfinished before it counts as failed
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
    rbs: (vk::Buffer, *mut u8, bool),
    pool: vk::CommandPool,
    cbs: Vec<vk::CommandBuffer>,
    qp: vk::QueryPool,
}

// `map` points into host-coherent memory that only its decoder touches
unsafe impl Send for Body {}

/// A decode's commands as `record` makes them: frame, ring slot, references (`n`) and setup slot with their DPB
/// images and Std infos, picture info, slice offsets, bitstream range, coded size, session reset, parameters.
struct Rec {
    f: u64,
    k: usize,
    n: usize,
    slot: [u8; MAXS],
    p5: StdVideoDecodeH265PictureInfo,
    p4: StdVideoDecodeH264PictureInfo,
    r5: [StdVideoDecodeH265ReferenceInfo; MAXS],
    r4: [StdVideoDecodeH264ReferenceInfo; MAXS],
    offs: Vec<u32>,
    range: u64,
    ext: vk::Extent2D,
    reset: bool,
    params: vk::VideoSessionParametersKHR,
}

/// What a decode's commands take of its decoder, shared with the checker.
struct Kit {
    dev: ash::Device,
    vq: ash::khr::video_queue::Device,
    vdq: ash::khr::video_decode_queue::Device,
    session: vk::VideoSessionKHR,
    bs: vk::Buffer,
    map: *mut u8,
    rbs: (vk::Buffer, *mut u8, bool),
    slot: usize,
    dpb: Vec<(vk::Image, vk::ImageView)>,
    imgs: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    co: bool,
    h265: bool,
    qp: vk::QueryPool,
}

// the decode thread writes a bitstream slot only once its last decode is checked, the checker the redo buffer only
unsafe impl Send for Kit {}
unsafe impl Sync for Kit {}

/// Records decode `e` into `cb`: the output image leaves home (still UNDEFINED before its first decode), the setup
/// slot's old picture is dead; the decode (in the result status query where there is one), coinciding the copy of the
/// setup slot into the output image; both back. `once`: never submitted again. `redo`: from the redo bitstream buffer.
unsafe fn record(x: &Kit, cb: vk::CommandBuffer, e: &Rec, once: bool, redo: bool) -> Result<()> {
    let (n, k, ext) = (e.n, e.k, e.ext);
    let mut d5 = [vk::VideoDecodeH265DpbSlotInfoKHR::default(); MAXS];
    let mut d4 = [vk::VideoDecodeH264DpbSlotInfoKHR::default(); MAXS];
    let mut res = [vk::VideoPictureResourceInfoKHR::default(); MAXS];
    let mut sl = [vk::VideoReferenceSlotInfoKHR::default(); MAXS];
    for j in 0..=n {
        let p = e.slot[j] as usize;
        d5[j].p_std_reference_info = &e.r5[j];
        d4[j].p_std_reference_info = &e.r4[j];
        res[j] = vk::VideoPictureResourceInfoKHR::default()
            .coded_extent(ext)
            .image_view_binding(x.dpb[p].1);
        sl[j] = vk::VideoReferenceSlotInfoKHR {
            p_next: if x.h265 {
                &d5[j] as *const _ as *const c_void
            } else {
                &d4[j] as *const _ as *const c_void
            },
            slot_index: p as i32,
            p_picture_resource: &res[j],
            ..Default::default()
        };
    }
    // the setup slot: index -1 in begin, its index in decode
    let setup = sl[n];
    sl[n].slot_index = -1;
    let i5 = vk::VideoDecodeH265PictureInfoKHR::default()
        .std_picture_info(&e.p5)
        .slice_segment_offsets(&e.offs);
    let i4 = vk::VideoDecodeH264PictureInfoKHR::default()
        .std_picture_info(&e.p4)
        .slice_offsets(&e.offs);
    let mut dec = vk::VideoDecodeInfoKHR::default()
        .src_buffer(if redo { x.rbs.0 } else { x.bs })
        .src_buffer_offset((k * x.slot) as u64)
        .src_buffer_range(e.range)
        .dst_picture_resource(match x.co {
            true => res[n],
            false => vk::VideoPictureResourceInfoKHR::default()
                .coded_extent(ext)
                .image_view_binding(x.views[k]),
        })
        .setup_reference_slot(&setup)
        .reference_slots(&sl[..n]);
    dec.p_next = if x.h265 {
        &i5 as *const _ as *const c_void
    } else {
        &i4 as *const _ as *const c_void
    };
    let (all, d, vq) = (vk::PipelineStageFlags::ALL_COMMANDS, &x.dev, x.vq.fp());
    let none = vk::DependencyFlags::empty();
    let once = match once {
        true => vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        false => vk::CommandBufferUsageFlags::empty(),
    };
    unsafe {
        d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(once))?;
        // the global barrier makes the earlier decodes' DPB writes visible to this one's reference reads: an image
        // barrier's access scopes cover only its own image
        let home = match e.f < x.imgs.len() as u64 {
            true => vk::ImageLayout::UNDEFINED,
            false => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let (out, s) = (x.imgs[k], x.dpb[e.slot[n] as usize].0);
        let (dpb, src) = (
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        let to = match x.co {
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
            .video_session(x.session)
            .video_session_parameters(e.params)
            .reference_slots(&sl[..=n]);
        let (qp, q) = (x.qp, x.qp != vk::QueryPool::null());
        if q {
            d.cmd_reset_query_pool(cb, qp, k as u32, 1);
        }
        (vq.cmd_begin_video_coding_khr)(cb, &begin);
        if e.reset {
            (vq.cmd_control_video_coding_khr)(
                cb,
                &vk::VideoCodingControlInfoKHR::default()
                    .flags(vk::VideoCodingControlFlagsKHR::RESET),
            );
        }
        if q {
            d.cmd_begin_query(cb, qp, k as u32, vk::QueryControlFlags::empty());
        }
        (x.vdq.fp().cmd_decode_video_khr)(cb, &dec);
        if q {
            d.cmd_end_query(cb, qp, k as u32);
        }
        (vq.cmd_end_video_coding_khr)(cb, &vk::VideoEndCodingInfoKHR::default());
        if x.co {
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
                        width: ext.width / q,
                        height: ext.height / q,
                        depth: 1,
                    })
            });
            d.cmd_copy_image(cb, s, src, out, to, &cp);
        }
        let back = [
            barrier(out, to, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            barrier(s, src, dpb),
        ];
        let nb = 1 + x.co as usize;
        d.cmd_pipeline_barrier(cb, all, all, none, &[], &[], &back[..nb]);
        d.end_command_buffer(cb)?;
    }
    Ok(())
}

/// A decode not yet checked: its commands, the DPB images it reads (bits), its last submit's value on `Chk::dec`,
/// its redos.
struct Fly {
    e: Rec,
    refs: u32,
    seq: u64,
    n: u8,
}

/// The unchecked decodes in submit order, the counts (`Decoder::redone`, `failed`), test marks (`Decoder::fake`), a
/// checker's error, under `Chk::st`.
#[derive(Default)]
struct Flight {
    q: VecDeque<Fly>,
    seq: u64,
    redone: u64,
    failed: u64,
    fake: Vec<(u64, bool)>,
    err: Option<vk::Result>,
    said: bool,
    stop: bool,
}

/// What a decoder shares with its checker thread (`check`).
struct Chk {
    st: Mutex<Flight>,
    cv: Condvar,
    dec: vk::Semaphore,
}

impl Chk {
    fn lock(&self) -> MutexGuard<'_, Flight> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The checker thread: takes the unchecked decodes in order; once a decode's last submit is done, a complete status
/// signals `done` for its frame, anything else (the driver signalled it unfinished) records it and every decode after
/// it again into `cbs` and submits them, reading what they read the first time (`Decoder::phys`). Recorded anew:
/// NVIDIA's driver (620.73) breaks for good on a command buffer submitted twice, every P picture after it wrong. After
/// `TRIES` redos a decode counts as failed and goes out as it is. An error ends it: kept for `decode` to return, and
/// every unchecked frame goes out as failed (`done` reaches the last, `failed` counts them), so that nothing waits
/// for a checker that is gone. Its lock is held for host work only: the decode thread waits on it under export's
/// `gate`.
fn check(
    c: Arc<Chk>,
    x: Arc<Kit>,
    queue: Arc<Mutex<vk::Queue>>,
    cbs: Vec<vk::CommandBuffer>,
    done: vk::Semaphore,
) {
    let dev = &x.dev;
    // Ok(false): stopped
    let till = |v| loop {
        let (s, v) = ([c.dec], [v]);
        let i = vk::SemaphoreWaitInfo::default().semaphores(&s).values(&v);
        match unsafe { dev.wait_semaphores(&i, 100_000_000) } {
            Ok(()) => return Ok(true),
            Err(vk::Result::TIMEOUT) if !c.lock().stop => {}
            Err(vk::Result::TIMEOUT) => return Ok(false),
            Err(e) => return Err(e),
        }
    };
    let signal = |f: u64| unsafe {
        let i = vk::SemaphoreSignalInfo::default()
            .semaphore(done)
            .value(f + 1);
        dev.signal_semaphore(&i)
    };
    let r = (|| loop {
        let (f, k, seq, n) = {
            let mut s = c.lock();
            loop {
                match (s.q.front(), s.stop) {
                    (Some(u), _) => break (u.e.f, u.e.k, u.seq, u.n),
                    (None, true) => return Ok(()),
                    _ => s = c.cv.wait(s).unwrap_or_else(|e| e.into_inner()),
                }
            }
        };
        if !till(seq)? {
            return Ok(());
        }
        let (mut v, fl) = ([0i32], vk::QueryResultFlags::WITH_STATUS_KHR);
        // not ready once its decode is done: the driver never finished it
        let ok = unsafe { dev.get_query_pool_results(x.qp, k as u32, &mut v, fl) }.is_ok()
            && v[0] == vk::QueryResultStatusKHR::COMPLETE.as_raw();
        let last = {
            let mut s = c.lock();
            let fake = (s.fake.iter().position(|y| y.0 == f)).map(|i| s.fake.swap_remove(i).1);
            if fake == Some(true) {
                return Err(vk::Result::ERROR_UNKNOWN);
            }
            if ok && fake.is_none() || n >= TRIES {
                (s.failed, s.redone) = (s.failed + !ok as u64, s.redone + (ok && n > 0) as u64);
                s.q.pop_front();
                signal(f)?;
                continue;
            }
            s.seq
        };
        // everything submitted so far done first (`cbs` are free again), the decode thread free meanwhile
        if !till(last)? {
            return Ok(());
        }
        let mut s = c.lock();
        if !std::mem::replace(&mut s.said, true) {
            let m = if x.rbs.2 { "device" } else { "host" };
            eprintln!(
                "video: frame {f}'s decode came back unfinished (the driver); redone from {m} memory"
            );
        }
        s.seq += 1;
        s.q[0].n += 1;
        let (sv, mut cb) = ([s.seq], vec![]);
        for (e, &b) in s.q.iter_mut().zip(&cbs) {
            e.seq = sv[0];
            // the redo reads its bitstream from device memory: a decode that reads it over a busy PCIe link fails more
            let o = e.e.k * x.slot;
            unsafe { ptr::copy_nonoverlapping(x.map.add(o), x.rbs.1.add(o), e.e.range as usize) };
            unsafe { record(&x, b, &e.e, true, true) }.map_err(|e| match e {
                Error::Vk(r) => r,
                _ => vk::Result::ERROR_UNKNOWN,
            })?;
            cb.push(b);
        }
        let mut tl = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&sv);
        let ss = [c.dec];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&cb)
            .signal_semaphores(&ss)
            .push_next(&mut tl);
        let q = queue.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { dev.queue_submit(*q, &[submit], vk::Fence::null()) }?;
    })();
    if let Err(e) = r {
        let mut s = c.lock();
        s.err = Some(e);
        s.failed += s.q.len() as u64;
        if let Some(u) = s.q.back() {
            let _ = signal(u.e.f);
        }
        s.q.clear();
    }
}

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
            d.destroy_buffer(self.rbs.0, None);
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
    // DPB image of each of the parsers' slots (`refs`): the same index unless checked, where the spares in `dpb` stand
    // in for a slot whose image an unchecked decode still reads
    phys: Vec<u8>,
    dpb: Vec<(vk::Image, vk::ImageView)>,
    mem: Vec<vk::DeviceMemory>, // session, DPB and bitstream memory
    session: vk::VideoSessionKHR,
    params: vk::VideoSessionParametersKHR,
    bs: vk::Buffer,
    map: *mut u8,
    rbs: (vk::Buffer, *mut u8, bool), // checked: the redos' bitstream buffer, in device memory (a type allowed it)
    slot: usize,                      // bitstream bytes per ring slot
    align: usize,                     // bitstream size alignment
    pool: vk::CommandPool,
    cbs: Vec<vk::CommandBuffer>,
    offs: Vec<u32>,
    reset: bool,
    qp: vk::QueryPool, // a result status query per ring slot, null where the decode family has none
    kit: Option<Arc<Kit>>,
    // with `qp` but no checker (decode and encode share the queue): frames whose status is counted, those not complete
    seen: Cell<[u64; 2]>,
    chk: Option<(Arc<Chk>, JoinHandle<()>, vk::CommandPool)>, // with `qp`; the pool of its redos
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

/// Decodes are checked, and those left unfinished redone (`check`), where the decode family reports a decode's result
/// status (NVIDIA's does; RADV's doesn't, and a query there hangs), unless decode and encode share the queue: there a
/// redo, and the checker's wait for the decodes before it, would queue behind an encode batch that waits for frames
/// the renderer fills only once their decodes are checked. There a failure is only counted (`Decoder::failed`).
fn checked(g: &Gpu) -> bool {
    status(g)
        && !g
            .enc
            .as_ref()
            .is_some_and(|e| Arc::ptr_eq(&e.queue, &g.dec.queue))
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
            // checked: a spare image per decode in flight, for the slots unchecked decodes read
            let (stat, chkd) = (status(g), checked(g));
            let np = match chkd {
                true => (slots + out).min(caps.max_dpb_slots as usize).min(MAXS),
                false => slots,
            };
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
                phys: (0..slots as u8).collect(),
                dpb: vec![],
                mem: vec![],
                session: vk::VideoSessionKHR::null(),
                params: vk::VideoSessionParametersKHR::null(),
                bs: vk::Buffer::null(),
                map: ptr::null_mut(),
                rbs: (vk::Buffer::null(), ptr::null_mut(), false),
                slot: BS.next_multiple_of(caps.min_bitstream_buffer_offset_alignment.max(1))
                    as usize,
                align: caps.min_bitstream_buffer_size_alignment.max(1) as usize,
                pool: vk::CommandPool::null(),
                cbs: vec![],
                offs: vec![],
                reset: true,
                qp: vk::QueryPool::null(),
                kit: None,
                seen: Cell::new([0; 2]),
                chk: None,
                shape: None,
                bin: Arc::downgrade(&g.bin),
                dev: g.dev.clone(),
                vq: g.vq.clone(),
            };
            if let Some(b) = Bin::take(&mut bin().body, shape) {
                (d.dpb, d.mem, d.session, d.bs, d.map, d.rbs) =
                    (b.dpb, b.mem, b.session, b.bs, b.map, b.rbs);
                (d.pool, d.cbs, d.qp) = (b.pool, b.cbs, b.qp);
                d.params(g)?;
                d.arm(g)?;
                d.shape = Some(shape);
                return Ok(d);
            }
            let info = vk::VideoSessionCreateInfoKHR::default()
                .queue_family_index(g.dec.family)
                .video_profile(&pr[0])
                .picture_format(fmt)
                .max_coded_extent(ext)
                .reference_picture_format(fmt)
                .max_dpb_slots(np as u32)
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
            for i in 0..np {
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
            if chkd {
                d.rbs.0 = g.dev.create_buffer(&info, None)?;
                let q = g.dev.get_buffer_memory_requirements(d.rbs.0);
                let hv =
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                // without Resizable BAR the host-visible device heap is 256 MiB: later decoders get host memory
                let dl = g.alloc(q, vk::MemoryPropertyFlags::DEVICE_LOCAL | hv);
                d.rbs.2 = dl.is_ok();
                d.mem.push(dl.or_else(|_| g.alloc(q, hv))?);
                let m = d.mem[d.mem.len() - 1];
                g.dev.bind_buffer_memory(d.rbs.0, m, 0)?;
                d.rbs.1 = g
                    .dev
                    .map_memory(m, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *mut u8;
            }
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(g.dec.family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            d.pool = g.dev.create_command_pool(&info, None)?;
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(d.pool)
                .command_buffer_count(out as u32);
            d.cbs = g.dev.allocate_command_buffers(&info)?;
            if stat {
                let mut info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::RESULT_STATUS_ONLY_KHR)
                    .query_count(out as u32);
                info.p_next = &pr[0] as *const _ as *const c_void;
                d.qp = g.dev.create_query_pool(&info, None)?;
            }
            d.params(g)?;
            d.warm(g)?;
            d.arm(g)?;
            d.shape = Some(shape);
            Ok(d)
        }
    }

    /// What its decodes' commands take (`Kit`), and the checker where the decode family reports result status
    /// (`check`), with command buffers of its own for redos.
    unsafe fn arm(&mut self, g: &Gpu) -> Result<()> {
        let x = Arc::new(Kit {
            dev: self.dev.clone(),
            vq: g.vq.clone(),
            vdq: g.vdq.clone(),
            session: self.session,
            bs: self.bs,
            map: self.map,
            rbs: self.rbs,
            slot: self.slot,
            dpb: self.dpb.clone(),
            imgs: self.ring.imgs.clone(),
            views: self.ring.views.clone(),
            co: self.co,
            h265: matches!(self.st, St::H265(_)),
            qp: self.qp,
        });
        self.kit = Some(x.clone());
        if self.qp == vk::QueryPool::null() || !checked(g) {
            return Ok(());
        }
        let d = &self.dev;
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(g.dec.family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe { d.create_command_pool(&info, None) }?;
        let mut t =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(self.cbs.len() as u32);
        let r = unsafe { d.allocate_command_buffers(&info) }.and_then(|cbs| {
            let s = vk::SemaphoreCreateInfo::default().push_next(&mut t);
            Ok((cbs, unsafe { d.create_semaphore(&s, None) }?))
        });
        let (cbs, dec) = match r {
            Ok(x) => x,
            Err(e) => {
                unsafe { d.destroy_command_pool(pool, None) };
                return Err(e.into());
            }
        };
        let c = Arc::new(Chk {
            st: Default::default(),
            cv: Condvar::new(),
            dec,
        });
        let a = (c.clone(), g.dec.queue.clone(), self.ring.done);
        let h = std::thread::Builder::new()
            .name("decode check".into())
            .spawn(move || check(a.0, x, a.1, cbs, a.2));
        match h {
            Ok(h) => self.chk = Some((c, h, pool)),
            Err(_) => {
                unsafe {
                    d.destroy_semaphore(c.dec, None);
                    d.destroy_command_pool(pool, None);
                }
                return Err(Bad("no thread for the decode checker"));
            }
        }
        Ok(())
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
                self.drain(self.frame)?;
                (vq.destroy_video_session_parameters_khr)(dh, self.params, ptr::null());
            }
        }
        self.params = p;
        Ok(())
    }

    /// Pictures that did not decode, counted once their frame is ready, where the decode family reports a decode's
    /// result status: decodes the driver still left unfinished after `TRIES` redos, those a checker that ended in an
    /// error (`decode` returns it) let go unchecked, or, where decodes are only counted (`checked`), every unfinished
    /// one. The pictures after one reference it up to the next IDR. A picture the driver refuses up front is an error
    /// (`decode`). macOS conceals instead and counts them.
    pub fn failed(&self) -> u64 {
        match &self.chk {
            Some(c) => c.0.lock().failed,
            None => {
                self.verify();
                self.seen.get()[1]
            }
        }
    }

    /// Where decodes are only counted: checks the result status of every decode done since the last check, before
    /// its query slot's next reset.
    fn verify(&self) {
        let [mut c, mut n] = self.seen.get();
        let (r, done) = (self.ring.slots() as u64, self.ring.ready().min(self.frame));
        while self.qp != vk::QueryPool::null() && self.chk.is_none() && c < done {
            let (k, mut v) = ((c % r) as u32, [0i32]);
            let f = vk::QueryResultFlags::WITH_STATUS_KHR;
            // not ready after its decode is done: the driver never finished it
            let ok = unsafe { self.dev.get_query_pool_results(self.qp, k, &mut v, f) };
            n += (ok.is_err() || v[0] != vk::QueryResultStatusKHR::COMPLETE.as_raw()) as u64;
            c += 1;
        }
        self.seen.set([c, n]);
    }

    /// Pictures whose decode the driver left unfinished and a redo then completed, before their frames were ready
    /// (NVIDIA's RTX 4090 on 620.73 now and then); each counts once however many redos it took.
    pub fn redone(&self) -> u64 {
        self.chk.as_ref().map_or(0, |c| c.0.lock().redone)
    }

    /// Tests: frame `f`'s first decode counts as unfinished, so it is redone; or, `err`, the checker ends in an error
    /// there. False where decodes aren't checked.
    #[doc(hidden)]
    pub fn fake(&self, f: u64, err: bool) -> bool {
        self.chk
            .as_ref()
            .map(|c| c.0.lock().fake.push((f, err)))
            .is_some()
    }

    /// Waits until `done` reaches `v`; where a checker signals it, only while the checker lives: its error otherwise.
    unsafe fn drain(&self, v: u64) -> Result<()> {
        let s = [self.ring.done];
        let Some((c, ..)) = &self.chk else {
            return unsafe { self.wait(&s, v, u64::MAX) }.map(drop);
        };
        while !unsafe { self.wait(&s, v, 100_000_000) }? {
            if let Some(e) = c.lock().err {
                return Err(e.into());
            }
        }
        Ok(())
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
        if let Some(e) = self.chk.as_ref().and_then(|c| c.0.lock().err) {
            return Err(e.into());
        }
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
        // checked: a redo of an unchecked decode reads what it read the first time, so the setup slot's image must
        // not be one of those; a spare stands in, or, with none left, the decodes in flight are checked first. So
        // before a session reset, which deactivates the slots they read.
        if let Some((c, ..)) = &self.chk {
            if self.reset {
                unsafe { self.drain(f)? };
            }
            let busy = c.lock().q.iter().fold(0, |m, u| m | u.refs);
            if busy >> self.phys[pic.setup] & 1 == 1 {
                let used = self.phys.iter().fold(busy, |m, &p| m | 1 << p);
                match (0..self.dpb.len()).find(|&p| used >> p & 1 == 0) {
                    Some(p) => self.phys[pic.setup] = p as u8,
                    None => unsafe { self.drain(f)? },
                }
            }
        }

        // H.265's RPS lists slots too
        for v in (p5.RefPicSetStCurrBefore.iter_mut())
            .chain(&mut p5.RefPicSetStCurrAfter)
            .chain(&mut p5.RefPicSetLtCurr)
            .filter(|v| **v != 0xff)
        {
            *v = self.phys[*v as usize];
        }
        // reference slots: the active ones, then the setup slot
        let mut e = Rec {
            f,
            k,
            n: pic.n,
            slot: [0; MAXS],
            p5,
            p4,
            r5: unsafe { std::mem::zeroed() },
            r4: unsafe { std::mem::zeroed() },
            offs: take(&mut self.offs),
            range: range as u64,
            ext: vk::Extent2D {
                width: pic.w,
                height: pic.h,
            },
            reset: std::mem::replace(&mut self.reset, false),
            params: self.params,
        };
        let mut reads = 0u32; // the DPB images it reads
        for (j, i) in act[..pic.n]
            .iter()
            .map(|&i| i as usize)
            .chain([pic.setup])
            .enumerate()
        {
            let (r, p) = (self.refs[i], self.phys[i]);
            reads |= ((j < pic.n) as u32) << p;
            e.slot[j] = p;
            e.r5[j].PicOrderCntVal = r.poc[0];
            (e.r4[j].FrameNum, e.r4[j].PicOrderCnt) = (r.num as u16, r.poc);
            e.r4[j]
                .flags
                .set_used_for_long_term_reference(r.long as u32);
        }
        let Some(x) = self.kit.clone() else {
            return Err(Bad("decoder not set up"));
        };
        let (cb, d) = (self.cbs[k], &self.dev);
        unsafe {
            record(&x, cb, &e, self.chk.is_none(), false)?;
            // checked: it signals the checker's timeline, and the checker `done`; in flight under the checker's lock,
            // so a redo and this submit don't interleave
            let ck = self.chk.as_ref().map(|c| &c.0);
            let mut fl = ck.map(|c| c.lock());
            let (sem, sv) = match (ck, &mut fl) {
                (Some(c), Some(s)) => {
                    // a checker that ended since: nothing would signal this one
                    if let Some(e) = s.err {
                        return Err(e.into());
                    }
                    s.seq += 1;
                    let (refs, seq) = (reads, s.seq);
                    s.q.push_back(Fly {
                        e: Rec {
                            offs: take(&mut e.offs),
                            ..e
                        },
                        refs,
                        seq,
                        n: 0,
                    });
                    (c.dec, seq)
                }
                _ => (self.ring.done, f + 1),
            };
            let (nw, all) = ((f >= r) as usize, vk::PipelineStageFlags::ALL_COMMANDS);
            let (ws, wv, wst) = ([self.ring.free], [v], [all]);
            let (ss, sv, cbs) = ([sem], [sv], [cb]);
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
            let r = d.queue_submit(*q, &[submit], vk::Fence::null());
            if let (Some(s), Err(_)) = (&mut fl, &r) {
                s.q.pop_back();
                s.seq -= 1;
            }
            drop((q, fl));
            // unchecked, its slice offsets come back for the next decode
            self.offs = take(&mut e.offs);
            r?;
            if let Some(c) = ck {
                c.cv.notify_one();
            }
        }
        self.frame += 1;
        Ok(f)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            // every submitted decode can finish: decode() submits only once its slot is released
            let idle = self.drain(self.frame);
            if let Some((c, h, pool)) = self.chk.take() {
                c.lock().stop = true;
                c.cv.notify_one();
                let _ = h.join();
                self.dev.destroy_semaphore(c.dec, None);
                self.dev.destroy_command_pool(pool, None);
            }
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
                rbs: self.rbs,
                pool: self.pool,
                cbs: take(&mut self.cbs),
                qp: self.qp,
            };
            let left = match (self.shape, self.bin.upgrade()) {
                (Some(s), Some(bin)) if idle.is_ok() => Bin::park(
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
