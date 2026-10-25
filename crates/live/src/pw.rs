//! PipeWire camera node (PLAN.md §3.2): a `Video/Source` node (`media.role` Camera) that the app feeds with frames
//! it renders on the GPU. Hand-written FFI to libpipewire-0.3; SPA pods are built and parsed here, since spa's pod
//! builder is header-only inline C.
//!
//! Negotiation: the node offers NV12, BGRx and RGBx at the given sizes and frame rate, first as DMA-BUF with every
//! modifier the device exports (mandatory, and fixated here: the driver's pick among those the consumer takes),
//! then the same formats without a modifier as MemFd. Each PipeWire buffer owns one exported [`Image`]. On the
//! MemFd path the app still renders into that image; [`Frame::queue`] submits its copy into a readback buffer and
//! hands the frame to the node's copy thread, which waits for that copy, copies it into the buffer's memfd and
//! queues the buffer: the thread rendering frames never waits for the GPU or copies through the CPU.
//!
//! Threads: PipeWire's callbacks run on its thread loop with the loop lock held; `Cam` takes the same lock. The
//! node drives the graph: [`Frame::queue`] queues the buffer and triggers a cycle, so frames go out at the rate
//! the app renders them.
use crate::dmabuf::{Fmt, Image, LINEAR, modifiers};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    io,
    mem::size_of,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    ptr, slice,
    sync::{Arc, Mutex, Once, mpsc},
    thread,
};
use video::{Error, Gpu};

pub(crate) enum ThreadLoop {}
pub(crate) enum Loop {}
pub(crate) enum Stream {}
pub(crate) enum Props {}

#[repr(C)]
pub(crate) struct Pod {
    pub size: u32,
    pub ty: u32,
}

#[repr(C)]
pub(crate) struct Chunk {
    pub offset: u32,
    pub size: u32,
    pub stride: i32,
    pub flags: i32,
}

#[repr(C)]
pub(crate) struct Data {
    pub ty: u32,
    pub flags: u32,
    pub fd: i64,
    pub mapoffset: u32,
    pub maxsize: u32,
    pub data: *mut c_void,
    pub chunk: *mut Chunk,
}

#[repr(C)]
pub(crate) struct Meta {
    pub ty: u32,
    pub size: u32,
    pub data: *mut c_void,
}

#[repr(C)]
pub(crate) struct Header {
    pub flags: u32,
    pub offset: u32,
    pub pts: i64,
    pub dts_offset: i64,
    pub seq: u64,
}

#[repr(C)]
pub(crate) struct SpaBuffer {
    pub n_metas: u32,
    pub n_datas: u32,
    pub metas: *mut Meta,
    pub datas: *mut Data,
}

/// `struct pw_buffer`
#[repr(C)]
pub(crate) struct Buffer {
    pub buffer: *mut SpaBuffer,
    pub user_data: *mut c_void,
    pub size: u64,
    pub requested: u64,
    pub time: u64,
}

type Cb = Option<unsafe extern "C" fn(*mut c_void)>;

/// `struct pw_stream_events`, version 2
#[repr(C)]
pub(crate) struct Events {
    pub version: u32,
    pub destroy: Cb,
    pub state_changed: Option<unsafe extern "C" fn(*mut c_void, c_int, c_int, *const c_char)>,
    pub control_info: Option<unsafe extern "C" fn(*mut c_void, u32, *const c_void)>,
    pub io_changed: Option<unsafe extern "C" fn(*mut c_void, u32, *mut c_void, u32)>,
    pub param_changed: Option<unsafe extern "C" fn(*mut c_void, u32, *const Pod)>,
    pub add_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut Buffer)>,
    pub remove_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut Buffer)>,
    pub process: Cb,
    pub drained: Cb,
    pub command: Option<unsafe extern "C" fn(*mut c_void, *const c_void)>,
    pub trigger_done: Cb,
}

#[link(name = "pipewire-0.3")]
unsafe extern "C" {
    pub(crate) fn pw_init(argc: *mut c_int, argv: *mut *mut *mut c_char);
    pub(crate) fn pw_thread_loop_new(name: *const c_char, props: *const c_void) -> *mut ThreadLoop;
    pub(crate) fn pw_thread_loop_get_loop(l: *mut ThreadLoop) -> *mut Loop;
    pub(crate) fn pw_thread_loop_start(l: *mut ThreadLoop) -> c_int;
    pub(crate) fn pw_thread_loop_stop(l: *mut ThreadLoop);
    pub(crate) fn pw_thread_loop_destroy(l: *mut ThreadLoop);
    pub(crate) fn pw_thread_loop_lock(l: *mut ThreadLoop);
    pub(crate) fn pw_thread_loop_unlock(l: *mut ThreadLoop);
    pub(crate) fn pw_properties_new(key: *const c_char, ...) -> *mut Props;
    pub(crate) fn pw_properties_set(
        p: *mut Props,
        key: *const c_char,
        value: *const c_char,
    ) -> c_int;
    pub(crate) fn pw_stream_new_simple(
        l: *mut Loop,
        name: *const c_char,
        props: *mut Props,
        ev: *const Events,
        data: *mut c_void,
    ) -> *mut Stream;
    pub(crate) fn pw_stream_connect(
        s: *mut Stream,
        dir: u32,
        target: u32,
        flags: u32,
        params: *mut *const Pod,
        n: u32,
    ) -> c_int;
    pub(crate) fn pw_stream_update_params(s: *mut Stream, params: *mut *const Pod, n: u32)
    -> c_int;
    pub(crate) fn pw_stream_dequeue_buffer(s: *mut Stream) -> *mut Buffer;
    pub(crate) fn pw_stream_queue_buffer(s: *mut Stream, b: *mut Buffer) -> c_int;
    fn pw_stream_return_buffer(s: *mut Stream, b: *mut Buffer) -> c_int;
    fn pw_stream_trigger_process(s: *mut Stream) -> c_int;
    pub(crate) fn pw_stream_get_node_id(s: *mut Stream) -> u32;
    fn pw_stream_get_state(s: *mut Stream, err: *mut *const c_char) -> c_int;
    fn pw_stream_set_error(s: *mut Stream, res: c_int, err: *const c_char, ...) -> c_int;
    pub(crate) fn pw_stream_destroy(s: *mut Stream);
}

/// SPA and PipeWire constants (spa/utils/type.h, spa/param/*.h, pipewire/stream.h).
pub(crate) mod spa {
    pub const ID: u32 = 3;
    pub const INT: u32 = 4;
    pub const LONG: u32 = 5;
    pub const RECT: u32 = 10;
    pub const FRAC: u32 = 11;
    pub const OBJECT: u32 = 15;
    pub const CHOICE: u32 = 19;
    // object types
    pub const OBJ_FORMAT: u32 = 0x40003;
    pub const OBJ_BUFFERS: u32 = 0x40004;
    pub const OBJ_META: u32 = 0x40005;
    // param ids
    pub const ENUM_FORMAT: u32 = 3;
    pub const FORMAT: u32 = 4;
    pub const BUFFERS: u32 = 5;
    pub const META: u32 = 6;
    // format keys and values
    pub const MEDIA_TYPE: u32 = 1;
    pub const MEDIA_SUBTYPE: u32 = 2;
    pub const VIDEO_FORMAT: u32 = 0x20001;
    pub const MODIFIER: u32 = 0x20002;
    pub const SIZE: u32 = 0x20003;
    pub const FRAMERATE: u32 = 0x20004;
    pub const COLOR_RANGE: u32 = 0x2000c;
    pub const COLOR_MATRIX: u32 = 0x2000d;
    pub const VIDEO: u32 = 2;
    pub const RAW: u32 = 1;
    pub const RGBX: u32 = 7;
    pub const BGRX: u32 = 8;
    pub const RGBA: u32 = 11;
    pub const BGRA: u32 = 12;
    pub const NV12: u32 = 23;
    pub const RANGE_16_235: u32 = 2;
    pub const MATRIX_BT709: u32 = 3;
    // buffers and meta keys
    pub const B_BUFFERS: u32 = 1;
    pub const B_BLOCKS: u32 = 2;
    pub const B_SIZE: u32 = 3;
    pub const B_STRIDE: u32 = 4;
    pub const B_DATATYPE: u32 = 6;
    pub const M_TYPE: u32 = 1;
    pub const M_SIZE: u32 = 2;
    pub const META_HEADER: u32 = 1;
    // choices and property flags
    pub const RANGE: u32 = 1;
    pub const ENUM: u32 = 3;
    pub const FLAGS: u32 = 4;
    pub const MANDATORY: u32 = 1 << 3;
    pub const DONT_FIXATE: u32 = 1 << 4;
    // data types and flags
    pub const MEMFD: u32 = 2;
    pub const DMABUF: u32 = 3;
    pub const READABLE: u32 = 1;
    pub const MAPPABLE: u32 = 8;
    // streams
    #[cfg(test)]
    pub const INPUT: u32 = 0;
    pub const OUTPUT: u32 = 1;
    #[cfg(test)]
    pub const AUTOCONNECT: u32 = 1;
    pub const DRIVER: u32 = 1 << 3;
    pub const ALLOC_BUFFERS: u32 = 1 << 8;
    pub const STREAMING: i32 = 3;
    pub const ERROR: i32 = -1;
    pub const ANY: u32 = u32::MAX;
}

pub(crate) fn raw<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_ne_bytes(b[i..i + 4].try_into().unwrap())
}

/// One SPA pod, built in place: values append, `obj` patches its size when its closure returns.
#[derive(Default)]
pub(crate) struct B(Vec<u8>);

impl B {
    fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }

    /// A value of SPA type `ty`; bodies pad to 8 bytes.
    pub fn val(&mut self, ty: u32, body: &[u8]) -> &mut Self {
        self.u32(body.len() as u32)
            .u32(ty)
            .0
            .extend_from_slice(body);
        self.0.resize(self.0.len().next_multiple_of(8), 0);
        self
    }

    pub fn id(&mut self, v: u32) -> &mut Self {
        self.val(spa::ID, &v.to_ne_bytes())
    }

    pub fn int(&mut self, v: u32) -> &mut Self {
        self.val(spa::INT, &v.to_ne_bytes())
    }

    pub fn long(&mut self, v: u64) -> &mut Self {
        self.val(spa::LONG, &v.to_ne_bytes())
    }

    /// `kind` over elements of type `ty`, `size` bytes each; `vals` holds them all, the default first.
    pub fn choice(&mut self, kind: u32, ty: u32, size: u32, vals: &[u8]) -> &mut Self {
        let mut c = raw(&[kind, 0, size, ty]).to_vec();
        c.extend_from_slice(vals);
        self.val(spa::CHOICE, &c)
    }

    pub fn prop(&mut self, key: u32, flags: u32) -> &mut Self {
        self.u32(key).u32(flags)
    }

    pub fn obj(&mut self, ty: u32, id: u32, f: impl FnOnce(&mut B)) -> &mut Self {
        let at = self.0.len();
        self.u32(0).u32(spa::OBJECT).u32(ty).u32(id);
        f(self);
        let n = (self.0.len() - at - 8) as u32;
        self.0[at..at + 4].copy_from_slice(&n.to_ne_bytes());
        self
    }

    /// The pod, 8-byte aligned as PipeWire reads it.
    pub fn done(&self) -> Vec<u64> {
        self.0
            .as_chunks()
            .0
            .iter()
            .map(|c| u64::from_ne_bytes(*c))
            .collect()
    }
}

/// The properties of an object pod: (key, flags, value type, value body).
pub(crate) unsafe fn props<'a>(p: *const Pod) -> Vec<(u32, u32, u32, &'a [u8])> {
    let mut v = vec![];
    if p.is_null() || unsafe { (*p).ty } != spa::OBJECT {
        return v;
    }
    let b = unsafe { slice::from_raw_parts(p.cast::<u8>().add(8), (*p).size as usize) };
    let mut o = 8;
    while o + 16 <= b.len() {
        let (size, ty) = (u32_at(b, o + 8) as usize, u32_at(b, o + 12));
        let Some(body) = b.get(o + 16..o + 16 + size) else {
            break;
        };
        v.push((u32_at(b, o), u32_at(b, o + 4), ty, body));
        o += 16 + size.next_multiple_of(8);
    }
    v
}

/// A value's elements: (element type, element size, elements); a choice's default comes first.
pub(crate) fn alts(ty: u32, body: &[u8]) -> (u32, usize, &[u8]) {
    match ty == spa::CHOICE && body.len() >= 16 {
        true => (u32_at(body, 12), u32_at(body, 8) as usize, &body[16..]),
        false => (ty, body.len(), body),
    }
}

/// A video format pod: format, size and the modifier property (flags, values without repeats) if it has one.
pub(crate) unsafe fn parse(p: *const Pod) -> Option<(Fmt, u32, u32, Option<(u32, Vec<u64>)>)> {
    let (mut fmt, mut size, mut m) = (None, None, None);
    for (key, flags, ty, body) in unsafe { props(p) } {
        let (_, n, v) = alts(ty, body);
        match key {
            spa::VIDEO_FORMAT if v.len() >= 4 => {
                fmt = match u32_at(v, 0) {
                    spa::NV12 => Some(Fmt::Nv12),
                    spa::RGBX | spa::RGBA => Some(Fmt::Rgba),
                    spa::BGRX | spa::BGRA => Some(Fmt::Bgra),
                    _ => None,
                }
            }
            spa::SIZE if v.len() >= 8 => size = Some((u32_at(v, 0), u32_at(v, 4))),
            spa::MODIFIER if n == 8 => {
                let mut l: Vec<u64> = v
                    .as_chunks()
                    .0
                    .iter()
                    .map(|c| u64::from_ne_bytes(*c))
                    .collect();
                let mut seen = vec![];
                l.retain(|x| {
                    !seen.contains(x) && {
                        seen.push(*x);
                        true
                    }
                });
                m = Some((flags, l));
            }
            _ => {}
        }
    }
    let (w, h) = size?;
    Some((fmt?, w, h, m))
}

/// An EnumFormat of `fmt` at `sizes` (the first is the default) and `fps`; with modifiers for DMA-BUF (one: fixated).
/// NV12 is BT.709 limited range, as the stitcher writes it.
pub(crate) fn offer(fmt: Fmt, sizes: &[[u32; 2]], fps: u32, mods: Option<&[u64]>) -> Vec<u64> {
    let mut b = B::default();
    b.obj(spa::OBJ_FORMAT, spa::ENUM_FORMAT, |b| {
        b.prop(spa::MEDIA_TYPE, 0).id(spa::VIDEO);
        b.prop(spa::MEDIA_SUBTYPE, 0).id(spa::RAW);
        b.prop(spa::VIDEO_FORMAT, 0).id(match fmt {
            Fmt::Nv12 => spa::NV12,
            Fmt::Rgba => spa::RGBX,
            Fmt::Bgra => spa::BGRX,
        });
        match mods {
            Some([m]) => b.prop(spa::MODIFIER, spa::MANDATORY).long(*m),
            Some(m) => b
                .prop(spa::MODIFIER, spa::MANDATORY | spa::DONT_FIXATE)
                .choice(spa::ENUM, spa::LONG, 8, raw(&[&m[..1], m].concat())),
            None => b,
        };
        b.prop(spa::SIZE, 0);
        match sizes {
            [s] => b.val(spa::RECT, raw(s)),
            _ => b.choice(spa::ENUM, spa::RECT, 8, raw(&[&sizes[..1], sizes].concat())),
        };
        b.prop(spa::FRAMERATE, 0).val(spa::FRAC, raw(&[fps, 1]));
        if fmt == Fmt::Nv12 {
            b.prop(spa::COLOR_RANGE, 0).id(spa::RANGE_16_235);
            b.prop(spa::COLOR_MATRIX, 0).id(spa::MATRIX_BT709);
        }
    });
    b.done()
}

pub(crate) fn meta() -> Vec<u64> {
    let mut b = B::default();
    b.obj(spa::OBJ_META, spa::META, |b| {
        b.prop(spa::M_TYPE, 0).id(spa::META_HEADER);
        b.prop(spa::M_SIZE, 0).int(size_of::<Header>() as u32);
    });
    b.done()
}

pub(crate) unsafe fn update(s: *mut Stream, p: &[Vec<u64>]) -> c_int {
    let mut v: Vec<*const Pod> = p.iter().map(|p| p.as_ptr().cast()).collect();
    unsafe { pw_stream_update_params(s, v.as_mut_ptr(), v.len() as u32) }
}

/// What every camera image allows: render, sample, copy out (MemFd), copy in (the stitcher's RGBA texture or NV12
/// buffer) and storage writes.
pub const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::STORAGE_BINDING);
/// Buffers we ask for (the consumer may settle on 2 to 8).
const NBUF: u32 = 4;
/// Flat 1080p and the 2:1 equirect.
pub const SIZES: [[u32; 2]; 2] = [[1920, 1080], [3840, 1920]];

/// The negotiated stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    pub fmt: Fmt,
    pub w: u32,
    pub h: u32,
    /// the DMA-BUFs' modifier; None for MemFd
    pub modifier: Option<u64>,
}

/// MemFd layout, GStreamer's default for these sizes: (stride, bytes); NV12 has its chroma plane after `h` rows.
fn shm(fmt: Fmt, w: u32, h: u32) -> (u32, u32) {
    match fmt {
        Fmt::Nv12 => (w, w * h * 3 / 2),
        _ => (w * 4, w * 4 * h),
    }
}

struct Buf {
    img: Arc<Image>,
    /// MemFd path: the memfd and its mapping
    shm: Option<(OwnedFd, *mut u8, usize)>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if let Some((_, p, n)) = self.shm {
            unsafe { libc::munmap(p.cast(), n) };
        }
    }
}

/// The stream's state; touched only with the loop lock held.
struct St {
    g: Arc<Gpu>,
    s: *mut Stream,
    offers: Vec<Vec<u64>>,
    fps: u32,
    mode: Option<Mode>,
    /// bumps whenever PipeWire drops buffers: frames dequeued before are stale
    epoch: u64,
    on: bool,
    /// by `pw_buffer.user_data`
    bufs: Vec<Option<Buf>>,
    /// the image made while settling the format; the first buffer takes it
    spare: Option<Image>,
    seq: u64,
    /// the copy thread's last failure
    err: Option<String>,
}

unsafe fn fail(s: *mut Stream, e: &str) {
    let e = CString::new(e).unwrap_or_default();
    unsafe { pw_stream_set_error(s, -libc::EIO, c"%s".as_ptr(), e.as_ptr()) };
}

unsafe extern "C" fn on_state(d: *mut c_void, _old: c_int, new: c_int, _err: *const c_char) {
    unsafe { (*d.cast::<St>()).on = new == spa::STREAMING };
}

unsafe extern "C" fn on_param(d: *mut c_void, id: u32, p: *const Pod) {
    let st = unsafe { &mut *d.cast::<St>() };
    if id != spa::FORMAT {
        return;
    }
    (st.epoch, st.mode, st.spare) = (st.epoch + 1, None, None);
    let Some((fmt, w, h, m)) = (unsafe { parse(p) }) else {
        return;
    };
    let mods = m.as_ref().map_or(&[][..], |m| &m.1);
    let img = match Image::new(&st.g, fmt, w, h, USAGE, mods) {
        Ok(i) => i,
        Err(e) => return unsafe { fail(st.s, &format!("camera image: {e}")) },
    };
    // Image::new falls back to LINEAR, which the consumer may not have listed
    if m.is_some() && !mods.contains(&img.modifier) {
        return unsafe { fail(st.s, "no DRM modifier both sides support") };
    }
    if let Some((fl, l)) = &m
        && fl & spa::DONT_FIXATE != 0
        && l.len() > 1
    {
        // fixate to the driver's pick among the modifiers both sides take; PipeWire then sets the format again
        let mut p = vec![offer(fmt, &[[w, h]], st.fps, Some(&[img.modifier]))];
        p.extend(st.offers.iter().cloned());
        unsafe { update(st.s, &p) };
        return;
    }
    let modifier = m.map(|_| img.modifier);
    let (stride, size) = shm(fmt, w, h);
    let (blocks, size, stride, ty) = match modifier {
        Some(_) => (
            img.planes.len() as u32,
            img.size as u32,
            img.planes[0].stride,
            spa::DMABUF,
        ),
        None => (1, size, stride, spa::MEMFD),
    };
    let mut b = B::default();
    b.obj(spa::OBJ_BUFFERS, spa::BUFFERS, |b| {
        b.prop(spa::B_BUFFERS, 0)
            .choice(spa::RANGE, spa::INT, 4, raw(&[NBUF, 2, 8]));
        b.prop(spa::B_BLOCKS, 0).int(blocks);
        b.prop(spa::B_SIZE, 0).int(size);
        b.prop(spa::B_STRIDE, 0).int(stride);
        b.prop(spa::B_DATATYPE, 0)
            .choice(spa::FLAGS, spa::INT, 4, raw(&[1u32 << ty]));
    });
    (st.mode, st.spare) = (
        Some(Mode {
            fmt,
            w,
            h,
            modifier,
        }),
        Some(img),
    );
    unsafe { update(st.s, &[b.done(), meta()]) };
}

unsafe extern "C" fn on_add(d: *mut c_void, b: *mut Buffer) {
    let st = unsafe { &mut *d.cast::<St>() };
    let Some(m) = st.mode else { return };
    let sb = unsafe { &mut *(*b).buffer };
    let datas = unsafe { slice::from_raw_parts_mut(sb.datas, sb.n_datas as usize) };
    let img = match st.spare.take().map_or_else(
        || Image::new(&st.g, m.fmt, m.w, m.h, USAGE, m.modifier.as_slice()),
        Ok,
    ) {
        Ok(i) => i,
        Err(e) => return unsafe { fail(st.s, &format!("camera image: {e}")) },
    };
    let want = if m.modifier.is_some() {
        spa::DMABUF
    } else {
        spa::MEMFD
    };
    if datas.is_empty() || datas[0].ty & 1 << want == 0 {
        return unsafe { fail(st.s, "buffer memory type not negotiated") };
    }
    let shm = match m.modifier {
        Some(md) => {
            for (d, p) in datas.iter_mut().zip(&img.planes) {
                d.ty = spa::DMABUF;
                d.flags = spa::READABLE | if md == LINEAR { spa::MAPPABLE } else { 0 };
                (d.fd, d.mapoffset, d.maxsize, d.data) = (
                    img.fd.as_raw_fd() as i64,
                    0,
                    img.size as u32,
                    ptr::null_mut(),
                );
                unsafe {
                    *d.chunk = Chunk {
                        offset: p.off,
                        size: p.len,
                        stride: p.stride as i32,
                        flags: 0,
                    }
                };
            }
            None
        }
        None => {
            let (stride, n) = shm(m.fmt, m.w, m.h);
            let fd = unsafe { libc::memfd_create(c"x5desk-camera".as_ptr(), libc::MFD_CLOEXEC) };
            if fd < 0 {
                return unsafe { fail(st.s, "memfd_create failed") };
            }
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            let p = unsafe {
                if libc::ftruncate(fd.as_raw_fd(), n as i64) < 0 {
                    libc::MAP_FAILED
                } else {
                    let prot = libc::PROT_READ | libc::PROT_WRITE;
                    libc::mmap(
                        ptr::null_mut(),
                        n as usize,
                        prot,
                        libc::MAP_SHARED,
                        fd.as_raw_fd(),
                        0,
                    )
                }
            };
            if p == libc::MAP_FAILED {
                return unsafe { fail(st.s, "memfd mapping failed") };
            }
            let d = &mut datas[0];
            d.ty = spa::MEMFD;
            d.flags = spa::READABLE | spa::MAPPABLE;
            (d.fd, d.mapoffset, d.maxsize, d.data) = (fd.as_raw_fd() as i64, 0, n, p);
            unsafe {
                *d.chunk = Chunk {
                    offset: 0,
                    size: n,
                    stride: stride as i32,
                    flags: 0,
                }
            };
            Some((fd, p.cast(), n as usize))
        }
    };
    unsafe { (*b).user_data = st.bufs.len() as *mut c_void };
    st.bufs.push(Some(Buf {
        img: Arc::new(img),
        shm,
    }));
}

unsafe extern "C" fn on_remove(d: *mut c_void, b: *mut Buffer) {
    let st = unsafe { &mut *d.cast::<St>() };
    st.epoch += 1;
    if let Some(x) = st.bufs.get_mut(unsafe { (*b).user_data } as usize) {
        *x = None;
    }
    if st.bufs.iter().all(Option::is_none) {
        st.bufs.clear();
    }
}

static EV: Events = Events {
    version: 2,
    destroy: None,
    state_changed: Some(on_state),
    control_info: None,
    io_changed: None,
    param_changed: Some(on_param),
    add_buffer: Some(on_add),
    remove_buffer: Some(on_remove),
    process: None,
    drained: None,
    command: None,
    trigger_done: None,
};

/// The camera node. Frames: [`Cam::dequeue`] an image no consumer holds, render into `Frame::img.tex` (NV12 through
/// its plane views), submit, then [`Frame::queue`].
pub struct Cam {
    tl: *mut ThreadLoop,
    st: *mut St,
    g: Arc<Gpu>,
    /// MemFd path: readback buffers not in use, the copy thread
    rb: Arc<Mutex<Vec<wgpu::Buffer>>>,
    cp: Option<(mpsc::Sender<Job>, thread::JoinHandle<()>)>,
}

// every access to `st` and the stream holds the thread loop's lock
unsafe impl Send for Cam {}
unsafe impl Sync for Cam {}

struct Lock(*mut ThreadLoop);

impl Lock {
    fn new(tl: *mut ThreadLoop) -> Lock {
        unsafe { pw_thread_loop_lock(tl) };
        Lock(tl)
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        unsafe { pw_thread_loop_unlock(self.0) };
    }
}

impl Cam {
    /// Publishes the node, `name` as its description (e.g. "X5 Camera"), offering `sizes` (e.g. [`SIZES`]) at `fps`,
    /// the rate the app queues frames at.
    pub fn new(g: Arc<Gpu>, name: &str, sizes: &[[u32; 2]], fps: u32) -> io::Result<Cam> {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe { pw_init(ptr::null_mut(), ptr::null_mut()) });
        let cs = |s: &str| {
            CString::new(s).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL in name"))
        };
        let node: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        let kv = [
            (c"media.type", cs("Video")?),
            (c"media.category", cs("Capture")?),
            (c"media.role", cs("Camera")?),
            (c"media.class", cs("Video/Source")?),
            (c"node.name", cs(&format!("x5desk.{node}"))?),
            (c"node.description", cs(name)?),
        ];
        let (w, h) = sizes
            .iter()
            .fold((0, 0), |a, s| (a.0.max(s[0]), a.1.max(s[1])));
        let fmts = [Fmt::Nv12, Fmt::Bgra, Fmt::Rgba];
        let mut offers: Vec<_> = fmts
            .iter()
            .filter_map(|&f| {
                let m: Vec<u64> = modifiers(&g, f, USAGE, w, h).iter().map(|m| m.0).collect();
                (!m.is_empty()).then(|| offer(f, sizes, fps, Some(&m)))
            })
            .collect();
        offers.extend(fmts.iter().map(|&f| offer(f, sizes, fps, None)));
        unsafe {
            let tl = pw_thread_loop_new(c"x5desk-camera".as_ptr(), ptr::null());
            if tl.is_null() {
                return Err(io::Error::last_os_error());
            }
            let st = Box::into_raw(Box::new(St {
                g: g.clone(),
                s: ptr::null_mut(),
                offers,
                fps,
                mode: None,
                epoch: 0,
                on: false,
                bufs: vec![],
                spare: None,
                seq: 0,
                err: None,
            }));
            // Drop cleans up from here
            let mut cam = Cam {
                tl,
                st,
                g,
                rb: Default::default(),
                cp: None,
            };
            let (tx, rx) = mpsc::channel();
            let (p, g, rb) = (Ptr(tl, st), cam.g.clone(), cam.rb.clone());
            let th = thread::Builder::new()
                .name("camera copy".into())
                .spawn(move || copier(p, g, rx, rb))?;
            cam.cp = Some((tx, th));
            let _l = cam.lock();
            let p = pw_properties_new(ptr::null::<c_char>());
            for (k, v) in &kv {
                pw_properties_set(p, k.as_ptr(), v.as_ptr());
            }
            let s = pw_stream_new_simple(
                pw_thread_loop_get_loop(tl),
                kv[5].1.as_ptr(),
                p,
                &EV,
                st.cast(),
            );
            if s.is_null() {
                return Err(io::Error::last_os_error());
            }
            (*st).s = s;
            let mut v: Vec<*const Pod> = (*st).offers.iter().map(|o| o.as_ptr().cast()).collect();
            let fl = spa::DRIVER | spa::ALLOC_BUFFERS;
            let r = pw_stream_connect(s, spa::OUTPUT, spa::ANY, fl, v.as_mut_ptr(), v.len() as u32);
            let r = if r < 0 { r } else { pw_thread_loop_start(tl) };
            if r < 0 {
                return Err(io::Error::from_raw_os_error(-r));
            }
            drop(_l);
            Ok(cam)
        }
    }

    fn lock(&self) -> Lock {
        Lock::new(self.tl)
    }

    /// The PipeWire node id, once the server registered the node.
    pub fn node(&self) -> Option<u32> {
        let _l = self.lock();
        let id = unsafe { pw_stream_get_node_id((*self.st).s) };
        (id != spa::ANY).then_some(id)
    }

    /// The negotiated stream while a consumer runs it.
    pub fn mode(&self) -> Option<Mode> {
        let _l = self.lock();
        let st = unsafe { &*self.st };
        st.mode.filter(|_| st.on)
    }

    /// The stream's error, if it failed, else the copy thread's last one.
    pub fn error(&self) -> Option<String> {
        let _l = self.lock();
        let mut e = ptr::null();
        let s = unsafe { pw_stream_get_state((*self.st).s, &mut e) };
        (s == spa::ERROR && !e.is_null())
            .then(|| unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned())
            .or_else(|| unsafe { (*self.st).err.clone() })
    }

    /// The next buffer's image, acquired for rendering (`dmabuf::Image::acquire`), while a consumer streams and a
    /// buffer is free. Dropping the frame without `queue` hands the buffer back unused.
    pub fn dequeue(&self) -> video::Result<Option<Frame<'_>>> {
        let f = {
            let _l = self.lock();
            let st = unsafe { &mut *self.st };
            let Some(mode) = st.mode.filter(|_| st.on) else {
                return Ok(None);
            };
            let b = unsafe { pw_stream_dequeue_buffer(st.s) };
            if b.is_null() {
                return Ok(None);
            }
            let i = unsafe { (*b).user_data } as usize;
            let Some(Some(x)) = st.bufs.get(i) else {
                unsafe { pw_stream_return_buffer(st.s, b) };
                return Ok(None);
            };
            Frame {
                img: x.img.clone(),
                mode,
                cam: self,
                b,
                epoch: st.epoch,
                i,
            }
        };
        f.img.acquire()?;
        Ok(Some(f))
    }
}

impl Drop for Cam {
    fn drop(&mut self) {
        // the copy thread queues what it has first: it uses the stream
        if let Some((tx, th)) = self.cp.take() {
            drop(tx);
            let _ = th.join();
        }
        unsafe {
            pw_thread_loop_lock(self.tl);
            if !(*self.st).s.is_null() {
                pw_stream_destroy((*self.st).s);
            }
            pw_thread_loop_unlock(self.tl);
            pw_thread_loop_stop(self.tl);
            pw_thread_loop_destroy(self.tl);
            drop(Box::from_raw(self.st));
        }
    }
}

/// A buffer being rendered.
pub struct Frame<'a> {
    pub img: Arc<Image>,
    pub mode: Mode,
    cam: &'a Cam,
    b: *mut Buffer,
    epoch: u64,
    i: usize,
}

/// A buffer going out: PipeWire's, the epoch it was dequeued in, our index, the stream's mode, its pts (ns).
struct Out {
    b: *mut Buffer,
    epoch: u64,
    i: usize,
    mode: Mode,
    pts: u64,
}

/// A MemFd frame for the copy thread: the readback buffer and its mapping's result, the buffer it goes into, the
/// planes in the readback buffer (row bytes, rows, pitch, offset).
struct Job {
    rb: wgpu::Buffer,
    map: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    o: Out,
    lay: [(u32, u32, u32, u32); 2],
    n: usize,
}

/// The stream's thread loop and state, for the copy thread.
struct Ptr(*mut ThreadLoop, *mut St);

// used with the loop lock held; Cam joins the copy thread before it frees them
unsafe impl Send for Ptr {}
unsafe impl Send for Job {}

/// The copy thread: each job's readback once the GPU wrote it, into its buffer's memfd, which then goes out.
///
/// It never polls the device with a wait: wgpu holds the device's lock for that whole wait, and presenting a frame
/// (the UI) needs that lock exclusively. It waits for the mapping's callback instead, which another thread's poll
/// or a short one of its own every millisecond runs.
fn copier(p: Ptr, g: Arc<Gpu>, rx: mpsc::Receiver<Job>, pool: Arc<Mutex<Vec<wgpu::Buffer>>>) {
    use mpsc::RecvTimeoutError::Timeout;
    let Ptr(tl, st) = p;
    for j in rx {
        let m = loop {
            match j.map.recv_timeout(std::time::Duration::from_millis(1)) {
                Ok(r) => break r.map_err(|e| e.to_string()),
                Err(Timeout) => {
                    if let Err(e) = g.device.poll(wgpu::PollType::Poll) {
                        break Err(e.to_string());
                    }
                }
                // the callback was dropped uncalled: wgpu-core 30 drops the callbacks a `Surface::configure` collected
                // when it fails because another thread submitted meanwhile (a window resize). The map itself ran,
                // so the mapping says whether it worked
                Err(_) => break Ok(()),
            }
        };
        let v = m.and_then(|_| j.rb.get_mapped_range(..).map_err(|e| e.to_string()));
        match &v {
            Ok(v) => unsafe { put(tl, st, &j.o, Some((v, &j.lay[..j.n]))) },
            Err(e) => unsafe {
                back(tl, st, &j.o);
                let _l = Lock::new(tl);
                (*st).err = Some(format!("camera readback: {e}"));
            },
        }
        drop(v);
        j.rb.unmap();
        pool.lock().unwrap().push(j.rb);
    }
}

/// Queues buffer `o` and triggers a cycle, MemFd after copying the readback `(bytes, layout)` into it; does nothing
/// when PipeWire dropped the buffer meanwhile.
unsafe fn put(
    tl: *mut ThreadLoop,
    st: *mut St,
    o: &Out,
    src: Option<(&[u8], &[(u32, u32, u32, u32)])>,
) {
    let _l = Lock::new(tl);
    let st = unsafe { &mut *st };
    if st.epoch != o.epoch {
        return;
    }
    unsafe {
        if let (
            Some((v, lay)),
            Some(Some(Buf {
                shm: Some((_, p, _)),
                ..
            })),
        ) = (src, st.bufs.get(o.i))
        {
            let (stride, _) = shm(o.mode.fmt, o.mode.w, o.mode.h);
            let mut dst = *p;
            for &(row, rows, pitch, at) in lay {
                for y in 0..rows as usize {
                    let s = v.as_ptr().add(at as usize + y * pitch as usize);
                    ptr::copy_nonoverlapping(s, dst.add(y * stride as usize), row as usize);
                }
                dst = dst.add((rows * stride) as usize);
            }
        }
        let sb = &*(*o.b).buffer;
        for x in slice::from_raw_parts(sb.metas, sb.n_metas as usize) {
            if x.ty == spa::META_HEADER && x.size as usize >= size_of::<Header>() {
                *x.data.cast::<Header>() = Header {
                    flags: 0,
                    offset: 0,
                    pts: o.pts as i64,
                    dts_offset: 0,
                    seq: st.seq,
                };
            }
        }
        st.seq += 1;
        pw_stream_queue_buffer(st.s, o.b);
        pw_stream_trigger_process(st.s);
    }
}

/// Hands buffer `o` back to PipeWire unused, unless PipeWire dropped it meanwhile.
unsafe fn back(tl: *mut ThreadLoop, st: *mut St, o: &Out) {
    let _l = Lock::new(tl);
    let st = unsafe { &*st };
    if st.epoch == o.epoch {
        unsafe { pw_stream_return_buffer(st.s, o.b) };
    }
}

impl Frame<'_> {
    /// Hands the frame to the consumer, after the submits that rendered it; `pts` in ns. DMA-BUF: releases the
    /// image (its fence goes onto the DMA-BUF). MemFd: submits its copy into a readback buffer, and the copy thread
    /// does the rest.
    pub fn queue(mut self, pts: u64) -> video::Result<()> {
        let (cam, m) = (self.cam, self.mode);
        let o = Out {
            b: std::mem::replace(&mut self.b, ptr::null_mut()),
            epoch: self.epoch,
            i: self.i,
            mode: m,
            pts,
        };
        if m.modifier.is_some() {
            let r = self.img.release();
            match r {
                Ok(_) => unsafe { put(cam.tl, cam.st, &o, None) },
                Err(_) => unsafe { back(cam.tl, cam.st, &o) },
            }
            return r.map(drop);
        }
        let planes: &[(wgpu::TextureAspect, u32, u32)] = match m.fmt {
            Fmt::Nv12 => &[
                (wgpu::TextureAspect::Plane0, 1, 1),
                (wgpu::TextureAspect::Plane1, 2, 2),
            ],
            _ => &[(wgpu::TextureAspect::All, 1, 4)],
        };
        // (row bytes, rows, pitch, offset): plane rows padded to 256 bytes in the readback buffer, planes in a row
        let mut lay = [(0, 0, 0, 0); 2];
        let mut n = 0;
        for (l, &(_, s, bpp)) in lay.iter_mut().zip(planes) {
            let (row, rows) = (m.w / s * bpp, m.h / s);
            let pitch = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            *l = (row, rows, pitch, n);
            n += pitch * rows;
        }
        let g = &cam.g;
        let rb = {
            let mut p = cam.rb.lock().unwrap();
            p.retain(|b| b.size() == n as u64);
            p.pop()
        };
        let rb = rb.unwrap_or_else(|| {
            g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("camera readback"),
                size: n as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        let mut e = g.device.create_command_encoder(&Default::default());
        for (&(aspect, s, _), &(_, _, pitch, at)) in planes.iter().zip(&lay) {
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.img.tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &rb,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: at as u64,
                        bytes_per_row: Some(pitch),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: m.w / s,
                    height: m.h / s,
                    depth_or_array_layers: 1,
                },
            );
        }
        g.queue.submit([e.finish()]);
        let (mt, map) = mpsc::sync_channel(1);
        rb.map_async(wgpu::MapMode::Read, .., move |r| drop(mt.send(r)));
        let n = planes.len();
        let tx = &cam.cp.as_ref().unwrap().0;
        tx.send(Job { rb, map, o, lay, n }).map_err(|x| {
            unsafe { back(cam.tl, cam.st, &x.0.o) };
            Error::Gpu("camera copy thread gone".into())
        })
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        if self.b.is_null() {
            return;
        }
        let o = Out {
            b: self.b,
            epoch: self.epoch,
            i: self.i,
            mode: self.mode,
            pts: 0,
        };
        unsafe { back(self.cam.tl, self.cam.st, &o) };
    }
}
