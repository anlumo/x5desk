//! VideoToolbox encode (H.265 Main / Main 10, H.264 High) from a ring of IOSurface pictures that wgpu fills, the
//! Linux backend's rules mirrored:
//! - Each input slot is a CVPixelBuffer (bi-planar 4:2:0, 8-bit or P010-like 10-bit) whose planes wgpu writes
//!   through Metal textures over the IOSurface. `copy_buffer` writes them with a render pass (wgpu's Metal backend has
//!   no NV12 / P010 textures, and its buffer copies want 256-byte rows).
//! - Event `filled`: the last wgpu submit that writes frame f signals f + 1 (`Input::release`, an MTLSharedEvent
//!   staged on wgpu's queue). Counter `done`: VideoToolbox has given back frame f's packet, so it is done with f's
//!   picture. `acquire` waits for `done ≥ f + 1 − SLOTS`, `encode` for `filled ≥ f + 1`.
//! - GOP: an IDR every `gop` frames or on request (`Input::key`), no B-frames (frame reordering off), so no reordering
//!   and dts = pts.
use super::sys::{self, CFTypeRef, CMTime, Cf, check};
pub use crate::stream::{Cfg, Packet, Rate};
use crate::{
    Error::{self, Bad},
    Gpu, Result,
    decode::Codec,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLDevice, MTLSharedEvent};
use std::{
    collections::VecDeque,
    ffi::c_void,
    ptr,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};
use wgpu_hal::api::Metal as Mtl;

/// Input ring size: how far the producer runs ahead of the encoder, so at most this many frames are in flight
/// before `Encoder::packet` must collect one.
pub const SLOTS: usize = 4;
const WAIT: Duration = Duration::from_millis(50); // acquire() waits for its slot before returning Full

/// The encoder's input: per slot a picture and its two plane textures, and the two timelines. Share it (`Arc`) with
/// the thread that submits the wgpu work that fills it.
pub struct Input {
    tex: Vec<[wgpu::Texture; 2]>,
    pics: Vec<Cf>,
    deep: bool,
    /// picture size
    pub w: u32,
    pub h: u32,
    filled: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    // `filled` values: the highest staged, the highest a submit has taken
    rel: Mutex<[u64; 2]>,
    /// frames VideoToolbox is done with
    done: Mutex<u64>,
    freed: Condvar,
    key: AtomicBool,
    copy: Copy,
}

// the pictures are IOSurfaces only wgpu (through `tex`) and VideoToolbox touch
unsafe impl Send for Input {}
unsafe impl Sync for Input {}

impl Input {
    /// Frame `f`'s luma and chroma (half size) planes: R8Unorm / Rg8Unorm, or R16Unorm / Rg16Unorm (10-bit,
    /// the sample in the high bits), render targets and copy destinations.
    pub fn planes(&self, f: u64) -> &[wgpu::Texture; 2] {
        &self.tex[(f % SLOTS as u64) as usize]
    }

    /// Call before the first wgpu submit that writes frame `f`. Waits up to 50 ms for VideoToolbox to finish
    /// frame `f − SLOTS` (the slot's last use), then returns `Error::Full`.
    pub fn acquire(&self, _: &Gpu, f: u64) -> Result<()> {
        let Some(need) = (f + 1).checked_sub(SLOTS as u64) else {
            return Ok(());
        };
        let d = self.done.lock().unwrap_or_else(|e| e.into_inner());
        let (d, _) = self
            .freed
            .wait_timeout_while(d, WAIT, |d| *d < need)
            .unwrap_or_else(|e| e.into_inner());
        match *d >= need {
            true => Ok(()),
            false => Err(Error::Full),
        }
    }

    /// Call after the last wgpu submit that writes frame `f`, then hand `f` to the encoder thread. Stages the signal
    /// `filled = f + 1` for the next submit from any thread; `Encoder::encode` sends it with an empty submit if none
    /// has carried it by then. Frames are released in order.
    pub fn release(&self, g: &Gpu, f: u64) {
        let Some(q) = (unsafe { g.queue.as_hal::<Mtl>() }) else {
            return;
        };
        let mut r = self.rel.lock().unwrap_or_else(|e| e.into_inner());
        if r[0] <= f {
            // one signal per submit, rising
            if !q.remove_signal_event(&self.filled) {
                r[1] = r[0];
            }
            q.add_signal_event(self.filled.clone(), f + 1);
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
        let Some(q) = (unsafe { g.queue.as_hal::<Mtl>() }) else {
            return false;
        };
        let mut r = self.rel.lock().unwrap_or_else(|e| e.into_inner());
        if r[0] < v || r[1] >= v {
            return r[0] >= v;
        }
        if !q.remove_signal_event(&self.filled) {
            r[1] = r[0];
            return true;
        }
        q.add_signal_event(self.filled.clone(), r[0]);
        drop((q, r));
        g.queue.submit([]);
        true
    }

    /// A command buffer for wgpu's queue that copies an NV12 / P010 picture from `buf` into frame `f`'s planes:
    /// `h` rows of Y at `off`, then `h / 2` rows of interleaved CbCr, rows `pitch` bytes apart (crate stitch's
    /// `Yuv` gives `buf` and `pitch`: the width rounded up to 4 px, times 2 for P010). Any even pitch (a render pass
    /// reads `buf` as storage, so it needs STORAGE usage). Submit it after the work that writes `buf`, between
    /// `acquire` and `release`.
    pub fn copy_buffer(
        &self,
        g: &Gpu,
        buf: &wgpu::Buffer,
        off: u64,
        pitch: u32,
        f: u64,
    ) -> Result<wgpu::CommandBuffer> {
        let (w, h) = (self.w, self.h);
        let b = 1 + self.deep as u32; // bytes per sample
        let need = off + pitch as u64 * (h + h / 2) as u64;
        if pitch < w * b
            || !pitch.is_multiple_of(2 * b)
            || !off.is_multiple_of(2 * b as u64)
            || need > buf.size()
            || !buf.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(Bad("buffer layout does not fit the picture"));
        }
        let mut e = g.device.create_command_encoder(&Default::default());
        let [y, uv] = self.planes(f);
        let planes = [
            (y, 0, off as u32),
            (uv, 1, (off + pitch as u64 * h as u64) as u32),
        ];
        for (t, chroma, at) in planes {
            self.copy
                .run(g, &mut e, buf, t, [at, pitch, self.deep as u32, chroma]);
        }
        Ok(e.finish())
    }
}

/// The render pass of `copy_buffer`: one fragment per texel reads its samples out of the buffer.
struct Copy {
    pipe: [wgpu::RenderPipeline; 4],
}

const COPY: &str = r#"
struct P { at: u32, pitch: u32, deep: u32, chroma: u32 }
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<uniform> p: P;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0);
}

fn sample(o: u32) -> f32 {
    let word = src[o >> 2u];
    if p.deep == 1u {
        return f32((word >> ((o & 2u) * 8u)) & 0xffffu) / 65535.0;
    }
    return f32((word >> ((o & 3u) * 8u)) & 0xffu) / 255.0;
}

@fragment fn fs(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let b = 1u + p.deep;
    let n = 1u + p.chroma;
    let o = p.at + u32(q.y) * p.pitch + u32(q.x) * n * b;
    var v = vec4f(sample(o), 0.0, 0.0, 1.0);
    if p.chroma == 1u {
        v.y = sample(o + b);
    }
    return v;
}
"#;

impl Copy {
    fn new(dev: &wgpu::Device) -> Copy {
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("encoder input copy"),
            source: wgpu::ShaderSource::Wgsl(COPY.into()),
        });
        use wgpu::TextureFormat as F;
        let pipe = [F::R8Unorm, F::Rg8Unorm, F::R16Unorm, F::Rg16Unorm].map(|format| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("encoder input copy"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &m,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &m,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        Copy { pipe }
    }

    fn run(
        &self,
        g: &Gpu,
        e: &mut wgpu::CommandEncoder,
        buf: &wgpu::Buffer,
        t: &wgpu::Texture,
        u: [u32; 4],
    ) {
        let pipe = &self.pipe[(u[2] * 2 + u[3]) as usize];
        let ub = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        ub.slice(..)
            .get_mapped_range_mut()
            .expect("mapped at creation")
            .copy_from_slice(bytemuck_u32(&u));
        ub.unmap();
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ub.as_entire_binding(),
                },
            ],
        });
        let view = t.create_view(&Default::default());
        let mut p = e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("encoder input copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        p.set_pipeline(pipe);
        p.set_bind_group(0, &bg, &[]);
        p.draw(0..3, 0..1);
    }
}

fn bytemuck_u32(u: &[u32; 4]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, 16) }
}

/// What the output callback hands back, in encode order: frame, status, the sample (None: dropped).
struct Out {
    q: Mutex<VecDeque<(u64, i32, Option<Cf>)>>,
    got: Condvar,
    /// the encoder's input (None: the parameter set session's)
    input: Option<Arc<Input>>,
}

unsafe extern "C" fn output(
    refcon: *mut c_void,
    frame: *mut c_void,
    status: i32,
    flags: u32,
    sample: CFTypeRef,
) {
    let out = unsafe { &*(refcon as *const Out) };
    let sample = match flags & sys::kVTEncodeInfo_FrameDropped {
        0 => Cf::retain(sample),
        _ => None,
    };
    let f = frame as u64;
    out.q
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((f, status, sample));
    out.got.notify_all();
    // VideoToolbox is done with the picture once its packet is out
    if let Some(i) = &out.input {
        let mut d = i.done.lock().unwrap_or_else(|e| e.into_inner());
        *d = (*d).max(f + 1);
        i.freed.notify_all();
    }
}

/// One stream's encoder. Lives on its own thread; the producer fills `input` on whichever thread submits its wgpu
/// work and tells this thread each frame it released.
pub struct Encoder {
    pub input: Arc<Input>,
    /// Parameter sets (H.265 VPS, SPS, PPS; H.264 SPS, PPS), NAL units without start codes: `hvcC` / `avcC`, or in
    /// band before a key frame. As VideoToolbox writes them for this configuration (from a picture encoded by a
    /// session of its own in `new`).
    pub params: Vec<Vec<u8>>,
    /// Quality levels usable: VideoToolbox has its own rate-quality trade-off, so only level 0.
    pub levels: u32,
    /// Index of the next frame to encode.
    pub frame: u64,
    c: Cfg,
    idr: u64,  // the last IDR's frame
    next: u64, // the next packet
    meta: [(i64, bool); SLOTS],
    session: Cf,
    out: Box<Out>,
}

// the session's callback only touches `out`; the session is used by the encoder's thread only
unsafe impl Send for Encoder {}

impl Encoder {
    /// An encoder on the media engine for `c`. Fails when VideoToolbox can't: no hardware encoder for the codec, a
    /// size or rate control setting it refuses.
    pub fn new(g: &Gpu, c: &Cfg) -> Result<Encoder> {
        if c.w % 2 != 0 || c.h % 2 != 0 || c.w == 0 || c.h == 0 {
            return Err(Bad("picture size not even"));
        }
        if c.deep && c.codec != Codec::H265 {
            return Err(Bad("10-bit encode is H.265 only"));
        }
        let fmt = match (c.deep, c.full) {
            (false, false) => sys::NV12_VIDEO,
            (false, true) => sys::NV12_FULL,
            (true, false) => sys::P010_VIDEO,
            (true, true) => sys::P010_FULL,
        };
        let (fy, fc, my, mc) = match c.deep {
            true => (
                wgpu::TextureFormat::R16Unorm,
                wgpu::TextureFormat::Rg16Unorm,
                sys::MTLPixelFormatR16Unorm,
                sys::MTLPixelFormatRG16Unorm,
            ),
            false => (
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::Rg8Unorm,
                sys::MTLPixelFormatR8Unorm,
                sys::MTLPixelFormatRG8Unorm,
            ),
        };
        let siting = unsafe {
            match c.chroma {
                1 => sys::kCVImageBufferChromaLocation_Center,
                2 => sys::kCVImageBufferChromaLocation_TopLeft,
                _ => sys::kCVImageBufferChromaLocation_Left,
            }
        };
        let (mut pics, mut tex) = (vec![], vec![]);
        for _ in 0..SLOTS {
            let p = picture(c.w, c.h, fmt)?;
            unsafe {
                for k in [
                    sys::kCVImageBufferChromaLocationTopFieldKey,
                    sys::kCVImageBufferChromaLocationBottomFieldKey,
                ] {
                    sys::CVBufferSetAttachment(
                        p.0,
                        k,
                        siting,
                        sys::kCVAttachmentMode_ShouldPropagate,
                    );
                }
                tex.push([
                    super::decode::plane(g, &p, 0, my, c.w, c.h, fy)?,
                    super::decode::plane(g, &p, 1, mc, c.w / 2, c.h / 2, fc)?,
                ]);
            }
            pics.push(p);
        }
        let filled = g.mtl.newSharedEvent().ok_or(Bad("no shared event"))?;
        let input = Arc::new(Input {
            tex,
            pics,
            deep: c.deep,
            w: c.w,
            h: c.h,
            filled,
            rel: Mutex::new([0; 2]),
            done: Mutex::new(0),
            freed: Condvar::new(),
            key: AtomicBool::new(false),
            copy: Copy::new(&g.device),
        });

        // the parameter sets: a session of its own encodes one black picture
        let params = {
            let out = Box::new(Out {
                q: Mutex::new(VecDeque::new()),
                got: Condvar::new(),
                input: None,
            });
            let s = session(c, &out)?;
            let p = picture(c.w, c.h, fmt)?;
            let r = unsafe {
                sys::VTCompressionSessionEncodeFrame(
                    s.0,
                    p.0,
                    CMTime::new(0, c.fps[0] as i32),
                    CMTime::invalid(),
                    ptr::null(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            };
            check("VTCompressionSessionEncodeFrame", r)?;
            unsafe {
                sys::VTCompressionSessionCompleteFrames(s.0, CMTime::invalid());
                sys::VTCompressionSessionInvalidate(s.0);
            }
            let q = out.q.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
            match q {
                Some((_, 0, Some(sample))) => sets(c.codec, &sample)?,
                Some((_, st, _)) => return Err(Error::Os("encode of the first picture", st)),
                None => return Err(Bad("no picture encoded")),
            }
        };

        let out = Box::new(Out {
            q: Mutex::new(VecDeque::new()),
            got: Condvar::new(),
            input: Some(input.clone()),
        });
        let session = session(c, &out)?;
        Ok(Encoder {
            input,
            params,
            levels: 1,
            frame: 0,
            c: *c,
            idr: 0,
            next: 0,
            meta: [(0, false); SLOTS],
            session,
            out,
        })
    }

    /// Encodes frame `self.frame`, which the producer has released (`Input::release`), and returns its index;
    /// `pts` comes back with its packet. Returns `Error::Full` with nothing encoded while `SLOTS` frames are in
    /// flight: collect one with `packet` first.
    pub fn encode(&mut self, g: &Gpu, pts: i64) -> Result<u64> {
        let (f, r) = (self.frame, SLOTS as u64);
        if f >= self.next + r {
            return Err(Error::Full);
        }
        // the producer's release reaches the GPU, and its copies are done before VideoToolbox reads the picture
        if !self.input.flush(g, f + 1) {
            return Err(Bad("encode of a frame not released"));
        }
        while !self
            .input
            .filled
            .waitUntilSignaledValue_timeoutMS(f + 1, 1000)
        {
            if self.input.filled.signaledValue() >= f + 1 {
                break;
            }
        }
        let k = (f % r) as usize;
        // the request is taken even when this frame is an IDR anyway, or it would make the next one an IDR too
        let req = self.input.key.swap(false, Relaxed);
        let key = req || f == 0 || f - self.idr >= self.c.gop as u64;
        if key {
            self.idr = f;
        }
        let props = key.then(|| {
            let d = Cf::dict();
            unsafe {
                d.set(
                    sys::kVTEncodeFrameOptionKey_ForceKeyFrame,
                    sys::kCFBooleanTrue,
                )
            };
            d
        });
        let [n, d] = self.c.fps;
        let t = |f: u64| CMTime::new(f as i64 * d as i64, n as i32);
        self.meta[k] = (pts, key);
        let s = unsafe {
            sys::VTCompressionSessionEncodeFrame(
                self.session.0,
                self.input.pics[k].0,
                t(f),
                t(1),
                props.as_ref().map_or(ptr::null(), |p| p.0),
                f as *mut c_void,
                ptr::null_mut(),
            )
        };
        check("VTCompressionSessionEncodeFrame", s)?;
        self.frame += 1;
        Ok(f)
    }

    /// The oldest frame in flight into `p`, once encoded: waits up to `ns`, returns false when nothing is in flight
    /// or it timed out. An error is final (the frame stays the oldest, `encode` returns `Full` once the ring is
    /// full): the stream is broken from there, make a new encoder.
    pub fn packet(&mut self, p: &mut Packet, ns: u64) -> Result<bool> {
        let f = self.next;
        if f >= self.frame {
            return Ok(false);
        }
        let end = Instant::now() + Duration::from_nanos(ns);
        let mut q = self.out.q.lock().unwrap_or_else(|e| e.into_inner());
        let mut pushed = false;
        loop {
            if let Some(i) = q.iter().position(|x| x.0 == f) {
                let (_, st, sample) = q.remove(i).unwrap();
                drop(q);
                check("encode", st)?;
                let k = (f % SLOTS as u64) as usize;
                let (pts, key) = self.meta[k];
                (p.frame, p.pts, p.dts, p.key) = (f, pts, pts, key);
                p.data.clear();
                p.nals.clear();
                if let Some(s) = sample {
                    p.key |= unsafe { sync(&s) };
                    unsafe { annex_b(&s, p) }?;
                }
                self.next += 1;
                return Ok(true);
            }
            let now = Instant::now();
            if now >= end {
                return Ok(false);
            }
            // VideoToolbox may hold frames back for its own look-ahead: past half the wait it is made to emit them
            if !pushed && end - now < Duration::from_nanos(ns / 2) {
                pushed = true;
                drop(q);
                unsafe {
                    sys::VTCompressionSessionCompleteFrames(self.session.0, CMTime::invalid())
                };
                q = self.out.q.lock().unwrap_or_else(|e| e.into_inner());
                continue;
            }
            q = self
                .out
                .got
                .wait_timeout(q, end - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe {
            sys::VTCompressionSessionCompleteFrames(self.session.0, CMTime::invalid());
            sys::VTCompressionSessionInvalidate(self.session.0);
        }
    }
}

/// An IOSurface picture VideoToolbox reads and Metal writes.
fn picture(w: u32, h: u32, fmt: u32) -> Result<Cf> {
    let attrs = Cf::dict();
    let surf = Cf::dict();
    unsafe {
        attrs
            .set(sys::kCVPixelBufferIOSurfacePropertiesKey, surf.0)
            .set(
                sys::kCVPixelBufferMetalCompatibilityKey,
                sys::kCFBooleanTrue,
            );
    }
    let mut p = ptr::null();
    let r = unsafe {
        sys::CVPixelBufferCreate(ptr::null(), w as usize, h as usize, fmt, attrs.0, &mut p)
    };
    Cf::own(p)
        .filter(|_| r == 0)
        .ok_or(Error::Os("CVPixelBufferCreate", r))
}

/// A compression session for `c` that reports to `out`.
fn session(c: &Cfg, out: &Out) -> Result<Cf> {
    let spec = Cf::dict();
    unsafe {
        spec.set(
            sys::kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,
            sys::kCFBooleanTrue,
        );
    }
    let codec = match c.codec {
        Codec::H264 => sys::kCMVideoCodecType_H264,
        Codec::H265 => sys::kCMVideoCodecType_HEVC,
    };
    let mut s = ptr::null();
    check("VTCompressionSessionCreate", unsafe {
        sys::VTCompressionSessionCreate(
            ptr::null(),
            c.w as i32,
            c.h as i32,
            codec,
            spec.0,
            ptr::null(),
            ptr::null(),
            output,
            out as *const Out as *mut c_void,
            &mut s,
        )
    })?;
    let s = Cf::own(s).ok_or(Bad("no compression session"))?;
    let set = |k: CFTypeRef, v: CFTypeRef| unsafe { sys::VTSessionSetProperty(s.0, k, v) };
    unsafe {
        let profile = match (c.codec, c.deep) {
            (Codec::H264, _) => sys::kVTProfileLevel_H264_High_AutoLevel,
            (Codec::H265, false) => sys::kVTProfileLevel_HEVC_Main_AutoLevel,
            (Codec::H265, true) => sys::kVTProfileLevel_HEVC_Main10_AutoLevel,
        };
        check(
            "profile",
            set(sys::kVTCompressionPropertyKey_ProfileLevel, profile),
        )?;
        set(sys::kVTCompressionPropertyKey_RealTime, sys::kCFBooleanTrue);
        check(
            "frame reordering",
            set(
                sys::kVTCompressionPropertyKey_AllowFrameReordering,
                sys::kCFBooleanFalse,
            ),
        )?;
        set(
            sys::kVTCompressionPropertyKey_MaxFrameDelayCount,
            Cf::i32(0).0,
        );
        // IDRs are forced where the GOP wants them; VideoToolbox inserts none of its own before
        set(
            sys::kVTCompressionPropertyKey_MaxKeyFrameInterval,
            Cf::i32(c.gop.max(1) as i32 + 1).0,
        );
        let fps = c.fps[0] as f64 / c.fps[1].max(1) as f64;
        set(
            sys::kVTCompressionPropertyKey_ExpectedFrameRate,
            Cf::f64(fps).0,
        );
        set(
            sys::kVTCompressionPropertyKey_ColorPrimaries,
            sys::kCVImageBufferColorPrimaries_ITU_R_709_2,
        );
        set(
            sys::kVTCompressionPropertyKey_TransferFunction,
            sys::kCVImageBufferTransferFunction_ITU_R_709_2,
        );
        set(
            sys::kVTCompressionPropertyKey_YCbCrMatrix,
            sys::kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        );
        let window = |bps: u64| {
            // the rate control buffer: bytes per `vbv` ms
            let secs = c.vbv.max(1) as f64 / 1000.0;
            let bytes = Cf::i64((bps as f64 / 8.0 * secs) as i64);
            let secs = Cf::f64(secs);
            Cf::array(&[&bytes, &secs])
        };
        match c.rate {
            Rate::Cbr(bps) => {
                if set(
                    sys::kVTCompressionPropertyKey_ConstantBitRate,
                    Cf::i64(bps as i64).0,
                ) != 0
                {
                    check(
                        "bitrate",
                        set(
                            sys::kVTCompressionPropertyKey_AverageBitRate,
                            Cf::i64(bps as i64).0,
                        ),
                    )?;
                    set(sys::kVTCompressionPropertyKey_DataRateLimits, window(bps).0);
                }
            }
            Rate::Vbr(avg, peak) => {
                check(
                    "bitrate",
                    set(
                        sys::kVTCompressionPropertyKey_AverageBitRate,
                        Cf::i64(avg as i64).0,
                    ),
                )?;
                set(
                    sys::kVTCompressionPropertyKey_DataRateLimits,
                    window(peak).0,
                );
            }
            Rate::Qp(q) => {
                let q = Cf::i32(q as i32);
                check(
                    "QP",
                    set(sys::kVTCompressionPropertyKey_MinAllowedFrameQP, q.0),
                )?;
                check(
                    "QP",
                    set(sys::kVTCompressionPropertyKey_MaxAllowedFrameQP, q.0),
                )?;
            }
        }
        check(
            "VTCompressionSessionPrepareToEncodeFrames",
            sys::VTCompressionSessionPrepareToEncodeFrames(s.0),
        )?;
    }
    Ok(s)
}

/// Whether a sample is a sync sample (no `NotSync` attachment).
unsafe fn sync(s: &Cf) -> bool {
    unsafe {
        let a = sys::CMSampleBufferGetSampleAttachmentsArray(s.0, 0);
        if a.is_null() || sys::CFArrayGetCount(a) == 0 {
            return true;
        }
        let d = sys::CFArrayGetValueAtIndex(a, 0);
        let v = sys::CFDictionaryGetValue(d, sys::kCMSampleAttachmentKey_NotSync);
        v.is_null() || sys::CFBooleanGetValue(v) == 0
    }
}

/// A sample's length-prefixed NAL units as Annex B into `p`.
unsafe fn annex_b(s: &Cf, p: &mut Packet) -> Result<()> {
    let mut avcc = vec![];
    unsafe {
        let b = sys::CMSampleBufferGetDataBuffer(s.0);
        if b.is_null() {
            return Err(Bad("no sample data"));
        }
        avcc.resize(sys::CMBlockBufferGetDataLength(b), 0);
        check(
            "CMBlockBufferCopyDataBytes",
            sys::CMBlockBufferCopyDataBytes(b, 0, avcc.len(), avcc.as_mut_ptr() as *mut c_void),
        )?;
    }
    let mut i = 0;
    while i + 4 <= avcc.len() {
        let n = u32::from_be_bytes(avcc[i..i + 4].try_into().unwrap()) as usize;
        let nal = avcc.get(i + 4..i + 4 + n).ok_or(Bad("sample NAL length"))?;
        p.data.extend_from_slice(&[0, 0, 0, 1]);
        let at = p.data.len();
        p.data.extend_from_slice(nal);
        p.nals.push(at..p.data.len());
        i += 4 + n;
    }
    Ok(())
}

/// The parameter sets of an encoded sample's format description.
fn sets(codec: Codec, s: &Cf) -> Result<Vec<Vec<u8>>> {
    let d = unsafe { sys::CMSampleBufferGetFormatDescription(s.0) };
    if d.is_null() {
        return Err(Bad("no format description"));
    }
    let get = |i: usize, set: &mut *const u8, size: &mut usize, count: &mut usize| unsafe {
        let mut nal = 0;
        match codec {
            Codec::H264 => sys::CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                d, i, set, size, count, &mut nal,
            ),
            Codec::H265 => sys::CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                d, i, set, size, count, &mut nal,
            ),
        }
    };
    let (mut set, mut size, mut count) = (ptr::null(), 0, 0);
    check("parameter sets", get(0, &mut set, &mut size, &mut count))?;
    (0..count)
        .map(|i| {
            check("parameter set", get(i, &mut set, &mut size, &mut count))?;
            Ok(unsafe { std::slice::from_raw_parts(set, size) }.to_vec())
        })
        .collect()
}
