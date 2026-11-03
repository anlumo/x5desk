//! VideoToolbox decode (H.264 / H.265 Main and Main 10) into a ring of plane textures that wgpu samples: luma R8 and
//! chroma RG8 (half size), or R16 / RG16 for 10-bit (P010's layout: the sample in the high bits). wgpu's Metal
//! backend has no NV12 / P010 textures, so each slot is two textures and `Ring::planes` hands out their views, as the
//! Linux backend hands out the plane views of one NV12 / P010 texture.
//! - VideoToolbox decodes into an IOSurface (CVPixelBuffer). A copy on wgpu's queue moves its planes into the slot;
//!   everything that samples a slot is later on the same queue, so `acquire` stages nothing.
//! - `release(f)` marks frame f and every earlier one free (a counter); the decode into a slot waits for the frame
//!   that used it before, then submits its copy after the consumer's submits that sampled it, which Metal orders.
//! - The parameter set parsers still do picture management, so leading pictures are `Error::Skip` exactly as on
//!   Linux; VideoToolbox gets every other picture.
//! - A picture VideoToolbox fails on shows the one before it (a copy of that slot): loss shows as concealment, as on
//!   Linux.
use super::sys::{self, CFTypeRef, CMSampleTimingInfo, CMTime, Cf, check};
pub use crate::pic::{Codec, MAXS, OUT};
pub(crate) use crate::pic::{Pic, Ref};
use crate::{
    Error::{self, Bad},
    Gpu, Result,
    pic::St,
};
use std::{
    ffi::c_void,
    ptr,
    sync::{
        Arc, Condvar, Mutex, RwLock, TryLockError,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    thread,
    time::{Duration, Instant},
};
use wgpu_hal::api::Metal as Mtl;

const WAIT: Duration = Duration::from_millis(50); // decode() waits for its slot before returning Full
const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

/// A decoder's output: per slot a luma and a chroma texture, and the two counters. Share it (`Arc`) with the thread
/// that samples.
pub struct Ring {
    tex: Vec<[wgpu::Texture; 2]>,
    deep: bool,
    /// frames whose copy is submitted
    done: AtomicU64,
    /// frames released: `free` reaches f + 1 once frame f and every earlier one may be overwritten
    free: Mutex<u64>,
    freed: Condvar,
}

impl Ring {
    /// P010-like 10-bit planes (H.265 Main 10) rather than 8-bit.
    pub fn deep(&self) -> bool {
        self.deep
    }

    /// Frame `f`'s luma and chroma (half size) plane views: R8Unorm / Rg8Unorm, or R16Unorm / Rg16Unorm for 10-bit
    /// streams, which read the 10-bit sample `d` as `64·d / 65535` as the plane views of a P010 texture do.
    pub fn planes(&self, f: u64) -> [wgpu::TextureView; 2] {
        self.slot(f)
            .each_ref()
            .map(|t| t.create_view(&Default::default()))
    }

    /// Plane `i` of frame `f` (0 luma, 1 chroma) as a copy source or destination at the plane's origin; chroma
    /// coordinates are half the picture's.
    pub fn plane(&self, f: u64, i: usize) -> wgpu::TexelCopyTextureInfo<'_> {
        self.slot(f)[i].as_image_copy()
    }

    /// Frame `f`'s textures (its two planes), e.g. to transition them back home.
    pub fn textures(&self, f: u64) -> impl Iterator<Item = &wgpu::Texture> {
        self.slot(f).iter()
    }

    fn slot(&self, f: u64) -> &[wgpu::Texture; 2] {
        &self.tex[(f % self.tex.len() as u64) as usize]
    }

    /// Its frames at once: frame `f` is in slot `f % slots()` ([`OUT`] unless [`Decoder::with_out`] asked for more).
    pub fn slots(&self) -> usize {
        self.tex.len()
    }

    /// Frames decoded and copied (on wgpu's queue): frames below this can be sampled.
    pub fn ready(&self) -> u64 {
        self.done.load(Relaxed)
    }

    /// Call before each wgpu submit that samples frame `f`. Nothing to stage: the frame's copy went to the same
    /// queue before `decode` returned it.
    pub fn acquire(&self, _: &Gpu, _: u64) {}

    /// Call after the last submit that samples frame `f`: the decoder may then overwrite `f` and every earlier
    /// frame, so a consumer holding several frames releases them in order.
    pub fn release(&self, _: &Gpu, f: u64) {
        let mut r = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if *r <= f {
            *r = f + 1;
            self.freed.notify_all();
        }
    }
}

/// What the output callback hands back: the status and the picture.
type Out = Mutex<Option<(i32, Option<Cf>)>>;

unsafe extern "C" fn output(
    refcon: *mut c_void,
    _: *mut c_void,
    status: i32,
    _: u32,
    image: CFTypeRef,
    _: CMTime,
    _: CMTime,
) {
    let out = unsafe { &*(refcon as *const Out) };
    *out.lock().unwrap_or_else(|e| e.into_inner()) = Some((status, Cf::retain(image)));
}

/// One stream's decoder. Lives on a media thread; `ring` (or the `Deref` to it) gives the textures, `acquire` and
/// `release`.
pub struct Decoder {
    pub ring: Arc<Ring>,
    /// texture size: the first SPS's coded size, or larger
    pub w: u32,
    pub h: u32,
    /// display window (x, y, w, h) inside the texture: the SPS's cropping of the last picture decoded (or of the
    /// first SPS before that); it changes only with a new SPS
    pub view: [u32; 4],
    /// index of the next frame
    pub frame: u64,
    /// frames that show the frame before them (concealment): VideoToolbox failed on their picture, or had nothing to
    /// decode it against
    pub failed: u64,
    /// the time the last frame's `decode` took once it had its slot: parsing, VideoToolbox's synchronous decode and
    /// the copy's submit (work Linux's decode queue does after `decode` returns)
    pub took: Duration,
    deep: bool,
    codec: Codec,
    st: St,
    refs: Vec<Ref>,
    desc: Cf,
    session: Option<Cf>,
    out: Box<Out>,
    sample: Vec<u8>,
    reset: bool,
    /// the host's lock around the submits (`gated`)
    gate: Option<&'static RwLock<()>>,
    /// a picture decoded since the start or the last `restart`: frame − 1's slot holds it, or what stands in for it
    good: bool,
    /// a picture failed: the next random access point gets a new session
    sick: bool,
    /// VideoToolbox has nothing to decode against until the next random access point (its session went away, a
    /// random access point failed): the pictures before it are concealed without decoding
    lost: bool,
}

impl std::ops::Deref for Decoder {
    type Target = Ring;
    fn deref(&self) -> &Ring {
        &self.ring
    }
}

impl Decoder {
    /// A decoder for `codec` with the parameter sets in `params` (VPS / SPS / PPS NAL units from `hvcC` / `avcC`
    /// or in band) and textures of `w` × `h`, grown to the SPS's coded size if that is larger. The first SPS's bit
    /// depth picks the planes' formats (`Ring::planes`): 8 R8 / RG8, 10 R16 / RG16; a picture of the other depth is
    /// an error.
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
        // coded size, display window, DPB slots (references, the current picture, one spare) and bit depth from the
        // first SPS
        let (cw, ch, view, slots, deep) = match &st {
            St::H264(s) => {
                let s = s.sps.iter().flatten().next().ok_or(Bad("no SPS"))?;
                (
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
                    s.width,
                    s.height,
                    s.view(),
                    s.dpbm.max_dec_minus1[s.max_sub as usize] as usize + 2,
                    s.depth > 8,
                )
            }
        };
        let (w, h) = (w.max(cw), h.max(ch));
        let (fy, fc) = match deep {
            true => (
                wgpu::TextureFormat::R16Unorm,
                wgpu::TextureFormat::Rg16Unorm,
            ),
            false => (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm),
        };
        let tex = (0..out)
            .map(|_| {
                [(fy, 1), (fc, 2)].map(|(format, s)| {
                    g.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("decoded plane"),
                        size: wgpu::Extent3d {
                            width: w / s,
                            height: h / s,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: USAGE,
                        view_formats: &[],
                    })
                })
            })
            .collect();
        let desc = describe(codec, &st)?;
        Ok(Decoder {
            ring: Arc::new(Ring {
                tex,
                deep,
                done: AtomicU64::new(0),
                free: Mutex::new(0),
                freed: Condvar::new(),
            }),
            w,
            h,
            view,
            frame: 0,
            failed: 0,
            took: Duration::ZERO,
            deep,
            codec,
            st,
            refs: vec![Ref::default(); slots.min(MAXS)],
            desc,
            session: None,
            out: Box::new(Mutex::new(None)),
            sample: vec![],
            reset: false,
            gate: None,
            good: false,
            sick: false,
            lost: false,
        })
    }

    /// Takes `gate` shared around each of its submits on wgpu's queue (the app's GATE: a surface configure fails while
    /// another thread submits, and the app retries it holding GATE exclusively); VideoToolbox's decode stays outside
    /// it. For a consumer that decodes outside that lock: one holding it around `decode` must not, as std's RwLock
    /// queues a reader behind a waiting writer. A submit waits for it up to 50 ms, then goes without it: the thread
    /// holding it may be waiting for this one (dropping a consumer joins its thread).
    pub fn gated(mut self, gate: &'static RwLock<()>) -> Decoder {
        self.gate = Some(gate);
        self
    }

    /// A session for the current format description, decoding into IOSurfaces of the stream's own sample format
    /// (range and bit depth as coded: VideoToolbox converts nothing).
    fn session(&mut self) -> Result<CFTypeRef> {
        if let Some(s) = &self.session {
            return Ok(s.0);
        }
        let full = unsafe {
            let b = sys::CMFormatDescriptionGetExtension(
                self.desc.0,
                sys::kCMFormatDescriptionExtension_FullRangeVideo,
            );
            !b.is_null() && sys::CFBooleanGetValue(b) != 0
        };
        let fmt = match (self.deep, full) {
            (false, false) => sys::NV12_VIDEO,
            (false, true) => sys::NV12_FULL,
            (true, false) => sys::P010_VIDEO,
            (true, true) => sys::P010_FULL,
        };
        let attrs = Cf::dict();
        let (fmt, surf) = (Cf::i64(fmt as i64), Cf::dict());
        unsafe {
            attrs
                .set(sys::kCVPixelBufferPixelFormatTypeKey, fmt.0)
                .set(sys::kCVPixelBufferIOSurfacePropertiesKey, surf.0)
                .set(
                    sys::kCVPixelBufferMetalCompatibilityKey,
                    sys::kCFBooleanTrue,
                );
        }
        let cb = sys::VTDecompressionOutputCallbackRecord {
            callback: output,
            refcon: &*self.out as *const Out as *mut c_void,
        };
        let mut s = ptr::null();
        check("VTDecompressionSessionCreate", unsafe {
            sys::VTDecompressionSessionCreate(
                ptr::null(),
                self.desc.0,
                ptr::null(),
                attrs.0,
                &cb,
                &mut s,
            )
        })?;
        let s = Cf::own(s).ok_or(Bad("no decompression session"))?;
        let r = s.0;
        self.session = Some(s);
        Ok(r)
    }

    fn close(&mut self) {
        if let Some(s) = self.session.take() {
            unsafe {
                sys::VTDecompressionSessionWaitForAsynchronousFrames(s.0);
                sys::VTDecompressionSessionInvalidate(s.0);
            }
        }
    }

    /// Decoding restarts at the next access unit, a random access point (a seek), as in a new decoder: the session is
    /// made anew and the DPB emptied. An H.265 CRA there is handled as a BLA (HandleCraAsBlaFlag): its RASL pictures,
    /// which reference pictures before it, are `Error::Skip`. So are the pictures shown before an H.264 non-IDR I
    /// picture there.
    pub fn restart(&mut self) {
        // nothing before it stands in for a picture that fails there
        (self.reset, self.good, self.sick, self.lost) = (true, false, false, false);
        self.refs.fill(Ref::default());
        match &mut self.st {
            St::H264(s) => s.restart(),
            St::H265(s) => s.restart(),
        }
    }

    /// Decodes one access unit (the NAL units of one sample: its slices plus any in-band parameter sets) into
    /// slot `frame % slots()` and returns the frame index. Waits up to 50 ms for the consumer to release that slot;
    /// then returns `Error::Full` with nothing changed. A leading picture that may reference pictures before where
    /// decoding started or restarted (`restart`) is `Error::Skip`, no frame. A picture VideoToolbox fails on (the
    /// live preview's lost packets) still takes its frame, which shows the frame before it (`failed` counts them);
    /// with no picture decoded before it since the start or the restart it is an error. The next random access point
    /// after a failure gets a new session; when VideoToolbox has nothing to decode against (its session went away, a
    /// random access point failed), the pictures before the next one are concealed without being decoded.
    pub fn decode(&mut self, g: &Gpu, au: &[&[u8]]) -> Result<u64> {
        let (f, r) = (self.frame, self.ring.slots() as u64);
        let k = (f % r) as usize;
        if f >= r {
            let need = f + 1 - r;
            let free = self.ring.free.lock().unwrap_or_else(|e| e.into_inner());
            let (free, _) = self
                .ring
                .freed
                .wait_timeout_while(free, WAIT, |v| *v < need)
                .unwrap_or_else(|e| e.into_inner());
            if *free < need {
                return Err(Error::Full);
            }
        }
        let t0 = Instant::now();
        let mut dirty = false;
        for n in au {
            if n.len() > 2 && self.st.kind(n) == 2 {
                dirty |= self.st.put(n)?;
            }
        }
        if dirty {
            let desc = describe(self.codec, &self.st)?;
            let keep = self.session.as_ref().is_some_and(|s| unsafe {
                sys::VTDecompressionSessionCanAcceptFormatDescription(s.0, desc.0) != 0
            });
            if !keep {
                self.close();
            }
            self.desc = desc;
        }

        // picture management as the Vulkan decoder does it: POC, references, leading pictures to skip
        let mut act = [0u8; MAXS];
        let (mut p5, mut p4): (
            ash::vk::native::StdVideoDecodeH265PictureInfo,
            ash::vk::native::StdVideoDecodeH264PictureInfo,
        ) = unsafe { std::mem::zeroed() };
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

        // one sample of 4-byte length-prefixed slices
        self.sample.clear();
        for n in au.iter().filter(|n| n.len() > 2 && self.st.kind(n) == 1) {
            self.sample
                .extend_from_slice(&(n.len() as u32).to_be_bytes());
            self.sample.extend_from_slice(n);
        }
        if self.sample.is_empty() {
            return Err(Bad("no slices"));
        }
        if std::mem::replace(&mut self.reset, false) {
            self.close();
        }
        let key = self.key(au);
        if self.sick && key {
            self.close();
            (self.sick, self.lost) = (false, false);
        }
        let image = match self.lost {
            true => Err(Bad("no reference: waiting for a random access point")),
            false => {
                let session = self.session()?;
                unsafe { self.run(session) }?
            }
        };
        match image {
            Ok(image) => {
                self.copy(g, &image, k)?;
                self.good = true;
            }
            Err(e) => {
                // an invalidated session (the media server restarted, the machine slept): a new one at the next
                // random access point, the first picture it can decode
                let gone = matches!(e, Error::Os(_, -12903));
                if gone {
                    self.close();
                }
                (self.sick, self.lost) = (true, self.lost || gone || key || !self.good);
                if !self.good {
                    return Err(e);
                }
                self.conceal(g, k);
                self.failed += 1;
            }
        }
        self.ring.done.store(f + 1, Relaxed);
        self.frame += 1;
        self.took = t0.elapsed();
        Ok(f)
    }

    /// A picture that references no other: an H.264 IDR or I picture, an H.265 IRAP picture.
    fn key(&self, au: &[&[u8]]) -> bool {
        let Some(n) = au.iter().find(|n| n.len() > 2 && self.st.kind(n) == 1) else {
            return false;
        };
        match &self.st {
            St::H264(s) => s.slice(n).is_ok_and(|s| s.idr || s.intra),
            St::H265(_) => (16..=21).contains(&crate::h265::nal_type(n)),
        }
    }

    /// Decodes `self.sample` synchronously: the picture, or the error VideoToolbox failed on it with.
    unsafe fn run(&mut self, session: CFTypeRef) -> Result<Result<Cf>> {
        let mut block = ptr::null();
        let len = self.sample.len();
        check("CMBlockBufferCreateWithMemoryBlock", unsafe {
            sys::CMBlockBufferCreateWithMemoryBlock(
                ptr::null(),
                self.sample.as_mut_ptr() as *mut c_void,
                len,
                sys::kCFAllocatorNull,
                ptr::null(),
                0,
                len,
                0,
                &mut block,
            )
        })?;
        let block = Cf::own(block).ok_or(Bad("no block buffer"))?;
        let timing = CMSampleTimingInfo {
            duration: CMTime::invalid(),
            pts: CMTime::new(self.frame as i64, 1),
            dts: CMTime::invalid(),
        };
        let mut sample = ptr::null();
        check("CMSampleBufferCreateReady", unsafe {
            sys::CMSampleBufferCreateReady(
                ptr::null(),
                block.0,
                self.desc.0,
                1,
                1,
                &timing,
                1,
                &len,
                &mut sample,
            )
        })?;
        let sample = Cf::own(sample).ok_or(Bad("no sample buffer"))?;
        *self.out.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let mut info = 0;
        let s = unsafe {
            sys::VTDecompressionSessionDecodeFrame(session, sample.0, 0, ptr::null_mut(), &mut info)
        };
        unsafe { sys::VTDecompressionSessionWaitForAsynchronousFrames(session) };
        let got = self.out.lock().unwrap_or_else(|e| e.into_inner()).take();
        Ok(match (s, got) {
            (0, Some((0, Some(image)))) => Ok(image),
            (0, Some((0, None))) | (0, None) => Err(Bad("VideoToolbox gave no picture")),
            (0, Some((s, _))) | (s, _) => Err(Error::Os("VTDecompressionSessionDecodeFrame", s)),
        })
    }

    /// Copies `image`'s planes into slot `k` on wgpu's queue. A picture of the display window's size (VideoToolbox
    /// crops) lands at the window's place.
    fn copy(&self, g: &Gpu, image: &Cf, k: usize) -> Result<()> {
        let (pw, ph) = unsafe {
            (
                sys::CVPixelBufferGetWidth(image.0) as u32,
                sys::CVPixelBufferGetHeight(image.0) as u32,
            )
        };
        let [vx, vy, vw, vh] = self.view;
        let (ox, oy) = match (pw, ph) == (vw, vh) {
            true => (vx, vy),
            false => (0, 0),
        };
        let (w, h) = (pw.min(self.w - ox) & !1, ph.min(self.h - oy) & !1);
        let dst = &self.ring.tex[k];
        let (fy, fc) = match self.deep {
            true => (sys::MTLPixelFormatR16Unorm, sys::MTLPixelFormatRG16Unorm),
            false => (sys::MTLPixelFormatR8Unorm, sys::MTLPixelFormatRG8Unorm),
        };
        let mut e = g.device.create_command_encoder(&Default::default());
        for (i, mf, s) in [(0, fy, 1), (1, fc, 2)] {
            let src = unsafe { plane(g, image, i, mf, pw / s, ph / s, dst[i].format()) }?;
            e.copy_texture_to_texture(
                src.as_image_copy(),
                wgpu::TexelCopyTextureInfo {
                    origin: wgpu::Origin3d {
                        x: ox / s,
                        y: oy / s,
                        z: 0,
                    },
                    ..dst[i].as_image_copy()
                },
                wgpu::Extent3d {
                    width: w / s,
                    height: h / s,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.submit(g, e.finish());
        g.flush();
        Ok(())
    }

    /// Copies frame f − 1's planes into slot `k` (frame f's) on wgpu's queue: the last picture decoded, or what
    /// stands in for it. Only frame f − 1 + `slots()` writes that slot again.
    fn conceal(&self, g: &Gpu, k: usize) {
        let t = &self.ring.tex;
        let (src, dst) = (&t[(k + t.len() - 1) % t.len()], &t[k]);
        let mut e = g.device.create_command_encoder(&Default::default());
        for (s, d) in src.iter().zip(dst) {
            e.copy_texture_to_texture(s.as_image_copy(), d.as_image_copy(), d.size());
        }
        self.submit(g, e.finish());
    }

    /// Submits `cb` holding the gate shared (`gated`): waits for it up to 50 ms, then submits without it.
    fn submit(&self, g: &Gpu, cb: wgpu::CommandBuffer) {
        let t = Instant::now();
        let _g = self.gate.and_then(|l| {
            loop {
                match l.try_read() {
                    Ok(x) => break Some(x),
                    Err(TryLockError::Poisoned(x)) => break Some(x.into_inner()),
                    Err(TryLockError::WouldBlock) if t.elapsed() < WAIT => {
                        thread::sleep(Duration::from_micros(200))
                    }
                    Err(TryLockError::WouldBlock) => break None,
                }
            }
        });
        g.queue.submit([cb]);
    }
}

/// A CF reference a wgpu texture's drop callback releases.
struct Keep(#[allow(dead_code)] Cf, #[allow(dead_code)] Cf);

/// Plane `i` of an IOSurface picture as a wgpu texture (`w` × `h` texels of `format`), which keeps the picture until
/// wgpu is done with it.
pub(crate) unsafe fn plane(
    g: &Gpu,
    image: &Cf,
    i: usize,
    mf: usize,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> Result<wgpu::Texture> {
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_metal::MTLTexture;
    // MTLTextureUsage ShaderRead | ShaderWrite | RenderTarget: the encoder's input is rendered to
    let attrs = Cf::dict();
    let usage = Cf::i64(7);
    unsafe { attrs.set(sys::kCVMetalTextureUsage, usage.0) };
    let mut t = ptr::null();
    let r = unsafe {
        sys::CVMetalTextureCacheCreateTextureFromImage(
            ptr::null(),
            g.cache.0,
            image.0,
            attrs.0,
            mf,
            w as usize,
            h as usize,
            i,
            &mut t,
        )
    };
    let t = Cf::own(t)
        .filter(|_| r == 0)
        .ok_or(Error::Os("CVMetalTextureCacheCreateTextureFromImage", r))?;
    let raw = unsafe { sys::CVMetalTextureGetTexture(t.0) } as *mut ProtocolObject<dyn MTLTexture>;
    let raw = unsafe { Retained::retain(raw) }.ok_or(Bad("no Metal texture"))?;
    let keep = Keep(t, Cf::retain(image.0).ok_or(Bad("no picture"))?);
    let hal = unsafe {
        wgpu_hal::metal::Device::texture_from_raw(
            raw,
            format,
            objc2_metal::MTLTextureType::Type2D,
            1,
            1,
            wgpu_hal::CopyExtent {
                width: w,
                height: h,
                depth: 1,
            },
            Some(Box::new(move || drop(keep))),
        )
    };
    Ok(unsafe {
        g.device.create_texture_from_hal::<Mtl>(
            hal,
            &wgpu::TextureDescriptor {
                label: Some("IOSurface plane"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::TextureUses::RESOURCE,
        )
    })
}

// a session's callback only writes `out`; the session is only used by the thread that owns the decoder
unsafe impl Send for Decoder {}

/// The format description of every stored parameter set.
fn describe(codec: Codec, st: &St) -> Result<Cf> {
    let sets: Vec<&[u8]> = match st {
        St::H264(s) => s.raw().collect(),
        St::H265(s) => s.raw().collect(),
    };
    let (p, n): (Vec<_>, Vec<_>) = sets.iter().map(|s| (s.as_ptr(), s.len())).unzip();
    let mut d = ptr::null();
    let r = unsafe {
        match codec {
            Codec::H264 => sys::CMVideoFormatDescriptionCreateFromH264ParameterSets(
                ptr::null(),
                p.len(),
                p.as_ptr(),
                n.as_ptr(),
                4,
                &mut d,
            ),
            Codec::H265 => sys::CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                ptr::null(),
                p.len(),
                p.as_ptr(),
                n.as_ptr(),
                4,
                ptr::null(),
                &mut d,
            ),
        }
    };
    check("CMVideoFormatDescriptionCreate", r)?;
    Cf::own(d).ok_or(Bad("no format description"))
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // the callback's target goes with the decoder
        self.close();
    }
}
