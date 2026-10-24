//! OBS source plugin (PLAN.md §3.1): "X5 Camera (x5desk)" shows one source of the running app (360, reframed,
//! lens, …), imported without a copy from the DMA-BUFs the app exports, plus its audio. The protocol is crate
//! live's `obs.rs`, compiled into this library.
//!
//! Per OBS source a receive thread runs the protocol client. It waits for each frame's sync_file on the CPU (so
//! OBS's render thread never stalls on the app's GPU work and keeps drawing the previous frame meanwhile), queues
//! the frame (at most [`QUEUE`]; a full queue drops its oldest), and passes audio with its timestamps to
//! `obs_source_output_audio`. Each OBS tick (`video_tick`) lets `video_render` (graphics thread) take one queued
//! frame: with the app's frame clock in phase with OBS's, frames often arrive two per tick and one per tick
//! later, and showing only the newest dropped about 15 %. `video_render` imports each buffer once (cached per buffer
//! index and DMA-BUF inode). When it moves on to a newer frame it fences its draws of the old one: `gs_sync_create`
//! exported into a point of a DRM timeline syncobj. The receive thread sends `done` once that point signalled, so
//! the app only rewrites a buffer after OBS's GPU finished reading it. Without OBS sync support it flushes GL
//! instead and answers right away; the app's `acquire` then waits for OBS's read fences on the DMA-BUF.
//! A hidden source (`hide`) selects no source, so the app stops rendering it; `show` selects it again. Socket sends
//! never block (a stalled app must not hang OBS's source destroy, which joins the thread): what doesn't go through
//! is retried.
#![cfg(target_os = "linux")]

mod drm;
mod ffi;
#[allow(dead_code)]
#[path = "../../live/src/obs.rs"]
mod proto;
#[cfg(test)]
mod tests;

use ffi::*;
use proto::{Client, Format, Frame, Msg};
use std::{
    collections::VecDeque,
    ffi::{CStr, CString, c_char, c_int, c_void},
    mem,
    os::fd::{AsRawFd, OwnedFd},
    path::PathBuf,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering::Relaxed},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const AB24: u32 = proto::fourcc(*b"AB24");
const AR24: u32 = proto::fourcc(*b"AR24");
const NV12: u32 = proto::fourcc(*b"NV12");
const R8: u32 = proto::fourcc(*b"R8  ");
const GR88: u32 = proto::fourcc(*b"GR88");
const MOD_INVALID: u64 = 0x00ff_ffff_ffff_ffff;
/// Buffer indices the import cache takes.
const BUFS: u32 = 64;
/// Frames received and not yet shown.
const QUEUE: usize = 2;

/// NV12 from two textures: luma `image` (R8), chroma `chroma` (GR88, half size); rows of the YUV → RGB matrix as
/// `dot(row.xyz, yuv) + row.w`.
const EFFECT: &CStr = c"
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d chroma;
uniform float4 yuv_r;
uniform float4 yuv_g;
uniform float4 yuv_b;

sampler_state smp {
	Filter   = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

struct V {
	float4 pos : POSITION;
	float2 uv  : TEXCOORD0;
};

V vs(V v)
{
	V o;
	o.pos = mul(float4(v.pos.xyz, 1.0), ViewProj);
	o.uv = v.uv;
	return o;
}

float4 ps(V v) : TARGET
{
	float3 c = float3(image.Sample(smp, v.uv).r, chroma.Sample(smp, v.uv).rg);
	return float4(saturate(float3(dot(yuv_r.xyz, c) + yuv_r.w, dot(yuv_g.xyz, c) + yuv_g.w,
		dot(yuv_b.xyz, c) + yuv_b.w)), 1.0);
}

technique Draw
{
	pass
	{
		vertex_shader = vs(v);
		pixel_shader  = ps(v);
	}
}
";

fn log(level: c_int, m: &str) {
    if let Ok(m) = CString::new(format!("[x5desk] {m}")) {
        unsafe { blog(level, c"%s".as_ptr(), m.as_ptr()) };
    }
}

/// Rows of the YUV → RGB matrix for a `Format::color` ([`proto::BT601`] or else BT.709, | [`proto::FULL`]) on
/// normalised texels.
fn yuv(color: u32) -> [Vec4; 3] {
    let (kr, kb) = match color & 0xff {
        proto::BT601 => (0.299, 0.114),
        _ => (0.2126, 0.0722),
    };
    let kg = 1. - kr - kb;
    let (ys, yo, cs) = match color & proto::FULL {
        0 => (255. / 219., 16. / 255., 255. / 224.),
        _ => (1., 0., 1.),
    };
    let co = 128. / 255.;
    let row = |u: f32, v: f32| Vec4([ys, u * cs, v * cs, -yo * ys - (u + v) * cs * co]);
    [
        row(0., 2. * (1. - kr)),
        row(-2. * kb * (1. - kb) / kg, -2. * kr * (1. - kr) / kg),
        row(2. * (1. - kb), 0.),
    ]
}

/// A frame from the receive thread, not yet drawn.
struct New {
    f: Frame,
    fds: [Option<OwnedFd>; 4],
    ino: u64,
    epoch: u64,
}

#[derive(Default)]
struct St {
    /// settings: socket path ("" = the default) and source name ("" = the first)
    socket: String,
    want: String,
    rev: u64,
    /// from the receive thread
    up: bool,
    err: String,
    list: Vec<(u32, String)>,
    id: u32,
    fmt: Option<Format>,
    /// shown in some view: only then does it select a source
    shown: bool,
    /// bumps when the selection or the connection changes: frames of an older epoch are gone, without `done`
    epoch: u64,
    q: VecDeque<New>,
    /// (buf, epoch, point): `done` once the point signalled; point 0 right away
    rel: VecDeque<(u32, u64, u64)>,
    /// stats: frames dropped from a full queue, frames whose sync_file was pending on arrival, `done`s held back
    /// by a pending fence point
    superseded: u32,
    waited: u32,
    deferred: u32,
}

impl St {
    fn reset(&mut self) {
        self.epoch += 1;
        self.q.clear();
        self.rel.clear();
    }
}

/// Shared by an OBS source and its receive thread.
struct Sh {
    st: Mutex<St>,
    quit: AtomicBool,
    /// w << 32 | h
    size: AtomicU64,
    /// OBS ticks so far
    tick: AtomicU64,
    src: *mut obs_source_t,
    /// what OBS imports: (fourcc, modifiers)
    fmts: Vec<(u32, Vec<u64>)>,
    obj: Option<drm::Obj>,
}

// `src` outlives the thread (destroy joins it first) and obs_source_output_audio is thread-safe
unsafe impl Send for Sh {}
unsafe impl Sync for Sh {}

struct Tex {
    /// DMA-BUF inode and the frame's layout (ts, buf and sync zeroed)
    key: (u64, Frame),
    /// NV12: luma, chroma; else the RGBA texture and null
    t: [*mut gs_texture_t; 2],
}

// dropped in the graphics context only
impl Drop for Tex {
    fn drop(&mut self) {
        self.t
            .into_iter()
            .filter(|t| !t.is_null())
            .for_each(|t| unsafe { gs_texture_destroy(t) });
    }
}

struct Cur {
    buf: u32,
    ts: u64,
    drawn: bool,
}

/// Graphics-thread state.
struct Gfx {
    /// imports of the buffers of epoch `ep`
    cache: Vec<Option<Tex>>,
    ep: u64,
    cur: Option<Cur>,
    eff: *mut gs_effect_t,
    sync: bool,
    point: u64,
    failed: bool,
    /// the tick whose frame it took
    tick: u64,
    /// stats since `t0`: frames shown, latency sum and max (ns), ticks that found the queue empty
    t0: u64,
    shown: u32,
    lat: (u64, u64),
    starved: u32,
}

struct Src {
    sh: Arc<Sh>,
    th: Option<JoinHandle<()>>,
    gfx: Mutex<Gfx>,
}

fn pick(list: &[(u32, String)], want: &str) -> u32 {
    list.iter()
        .find(|s| want.is_empty() || s.1 == want)
        .map_or(0, |s| s.0)
}

fn readable(fd: &OwnedFd, ms: i32) -> bool {
    let mut p = libc::pollfd {
        fd: fd.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe { libc::poll(&mut p, 1, ms) == 1 }
}

fn ino(fd: &OwnedFd) -> Option<u64> {
    let mut s: libc::stat = unsafe { mem::zeroed() };
    (unsafe { libc::fstat(fd.as_raw_fd(), &mut s) } == 0).then_some(s.st_ino)
}

/// The receive thread.
fn run(sh: &Sh) {
    let fmts: Vec<(u32, &[u64])> = sh.fmts.iter().map(|(f, m)| (*f, &m[..])).collect();
    let (mut cl, mut rev, mut id, mut retry) = (None::<Client>, u64::MAX, 0, Instant::now());
    let (mut path, mut want, mut on) = (PathBuf::new(), String::new(), false);
    // (src, buf) to hand back, in order; what a full socket refused stays for the next round
    let mut done: Vec<(u32, u32)> = Vec::with_capacity(8);
    let mut held = 0;
    while !sh.quit.load(Relaxed) {
        let (busy, sel) = {
            let mut st = sh.st.lock().unwrap();
            if st.rev != rev {
                rev = st.rev;
                want.clone_from(&st.want);
                on = st.shown;
                let p = match st.socket.as_str() {
                    "" => proto::path().unwrap_or_default(),
                    s => s.into(),
                };
                if p != path {
                    (path, retry) = (p, Instant::now());
                    if cl.take().is_some() {
                        (st.up, st.id, id) = (false, 0, 0);
                        st.reset();
                        done.clear();
                    }
                }
            }
            let w = if on { pick(&st.list, &want) } else { 0 };
            (!st.rel.is_empty() || !done.is_empty(), w)
        };
        let Some(c) = cl.as_mut() else {
            if Instant::now() < retry {
                thread::sleep(Duration::from_millis(50));
                continue;
            }
            let r = Client::connect(&path, 0, &fmts);
            let mut st = sh.st.lock().unwrap();
            match r {
                Ok(c) => {
                    log(LOG_INFO, &format!("connected to {}", path.display()));
                    (cl, id, st.up, st.err) = (Some(c), 0, true, String::new());
                }
                Err(e) => {
                    let e = format!("{}: {e}", path.display());
                    if st.err != e {
                        log(LOG_INFO, &format!("waiting for x5desk at {e}"));
                        st.err = e;
                    }
                    retry = Instant::now() + Duration::from_millis(500);
                }
            }
            continue;
        };
        let mut lost = false;
        // socket calls wait until the message (which borrows the client) is gone
        match c.recv(Some(Duration::from_millis(if busy { 2 } else { 100 }))) {
            Ok(Msg::Hello(v)) => sh.st.lock().unwrap().list = v,
            Ok(Msg::Format(s, f)) if s == id => {
                sh.size.store((f.w as u64) << 32 | f.h as u64, Relaxed);
                sh.st.lock().unwrap().fmt = Some(f);
            }
            Ok(Msg::Frame {
                src,
                f,
                planes,
                sync,
            }) if src == id => {
                // the app's writes, before OBS may sample it
                let w = sync.as_ref().is_some_and(|s| !readable(s, 0));
                let ok = !w || sync.as_ref().is_some_and(|s| readable(s, 1000));
                match planes[0].as_ref().and_then(ino).filter(|_| ok) {
                    Some(ino) => {
                        sh.size.store((f.w as u64) << 32 | f.h as u64, Relaxed);
                        let mut st = sh.st.lock().unwrap();
                        st.waited += w as u32;
                        let epoch = st.epoch;
                        st.q.push_back(New {
                            f,
                            fds: planes,
                            ino,
                            epoch,
                        });
                        if st.q.len() > QUEUE {
                            let o = st.q.pop_front().unwrap();
                            done.push((id, o.f.buf));
                            st.superseded += 1;
                        }
                    }
                    None => {
                        log(
                            LOG_WARNING,
                            "dropped a frame: no plane or its sync_file timed out",
                        );
                        done.push((id, f.buf));
                    }
                }
            }
            Ok(Msg::Audio {
                src,
                ts,
                rate,
                ch,
                pcm,
            }) if src == id && matches!(ch, 1..=6 | 8) => {
                let mut data = [ptr::null(); 8];
                data[0] = pcm.as_ptr().cast();
                let a = Audio {
                    data,
                    frames: (pcm.len() / ch as usize) as u32,
                    speakers: ch as c_int,
                    format: AUDIO_FORMAT_FLOAT,
                    samples_per_sec: rate,
                    timestamp: ts,
                };
                unsafe { obs_source_output_audio(sh.src, &a) };
            }
            Ok(Msg::Bye(0)) => lost = true,
            Ok(Msg::Bye(s)) if s == id => {
                // it selects again once the source comes back in a hello
                id = 0;
                let mut st = sh.st.lock().unwrap();
                (st.id, st.fmt) = (0, None);
                st.list.retain(|x| x.0 != s);
                st.reset();
                done.clear();
            }
            Ok(_) => {}
            Err(e) => lost = e.kind() != std::io::ErrorKind::TimedOut,
        }
        {
            let mut st = sh.st.lock().unwrap();
            let ep = st.epoch;
            while let Some(&(b, e, p)) = st.rel.front() {
                if e == ep && p != 0 && sh.obj.as_ref().is_some_and(|o| !o.signaled(p)) {
                    st.deferred += (p != held) as u32;
                    held = p;
                    break;
                }
                st.rel.pop_front();
                if e == ep {
                    done.push((id, b));
                }
            }
        }
        let mut k = 0;
        for &(s, b) in &done {
            match c.done(s, b) {
                Ok(()) => k += 1,
                Err(e) => {
                    lost |= e.kind() != std::io::ErrorKind::WouldBlock;
                    break;
                }
            }
        }
        done.drain(..k);
        if sel != id && !lost {
            match c.select(sel) {
                Ok(()) => {
                    id = sel;
                    done.clear();
                    let mut st = sh.st.lock().unwrap();
                    (st.id, st.fmt) = (sel, None);
                    st.reset();
                }
                Err(e) => lost = e.kind() != std::io::ErrorKind::WouldBlock,
            }
        }
        if lost {
            log(LOG_INFO, "x5desk closed the connection");
            (cl, id, retry) = (None, 0, Instant::now() + Duration::from_millis(500));
            done.clear();
            let mut st = sh.st.lock().unwrap();
            (st.up, st.id, st.fmt) = (false, 0, None);
            st.reset();
        }
    }
}

impl Gfx {
    /// Fences every draw so far; returns its syncobj point, or 0 after a flush (implicit sync).
    unsafe fn fence(&mut self, sh: &Sh) -> u64 {
        if let Some(o) = sh.obj.as_ref().filter(|_| self.sync) {
            self.point += 1;
            unsafe {
                let s = gs_sync_create();
                let ok = !s.is_null()
                    && gs_sync_export_syncobj_timeline_point(s, o.fd.as_raw_fd(), self.point);
                if !s.is_null() {
                    gs_sync_destroy(s);
                }
                if ok {
                    return self.point;
                }
            }
            self.sync = false;
            log(
                LOG_WARNING,
                "fence export failed, falling back to implicit sync",
            );
        }
        unsafe { gs_flush() };
        0
    }

    /// Imports `n` into the cache unless its buffer is there already.
    unsafe fn import(&mut self, n: &New) -> bool {
        let (f, i) = (&n.f, n.f.buf as usize);
        if n.f.buf >= BUFS {
            return false;
        }
        let key = (
            n.ino,
            Frame {
                ts: 0,
                buf: 0,
                sync: 0,
                ..*f
            },
        );
        if self.cache.len() <= i {
            self.cache.resize_with(i + 1, || None);
        }
        if self.cache[i].as_ref().is_some_and(|t| t.key == key) {
            return true;
        }
        self.cache[i] = None;
        let fds = n
            .fds
            .each_ref()
            .map(|f| f.as_ref().map_or(-1, |f| f.as_raw_fd()));
        let m = [f.modifier; 4];
        let t = unsafe {
            match (f.fourcc, f.planes) {
                (NV12, 2) => [
                    gs_texture_create_from_dmabuf(
                        f.w,
                        f.h,
                        R8,
                        GS_R8,
                        1,
                        &fds[0],
                        &f.stride[0],
                        &f.off[0],
                        &m[0],
                    ),
                    gs_texture_create_from_dmabuf(
                        f.w.div_ceil(2),
                        f.h.div_ceil(2),
                        GR88,
                        GS_R8G8,
                        1,
                        &fds[1],
                        &f.stride[1],
                        &f.off[1],
                        &m[1],
                    ),
                ],
                (AB24 | AR24, 1..=4) => [
                    gs_texture_create_from_dmabuf(
                        f.w,
                        f.h,
                        f.fourcc,
                        if f.fourcc == AB24 { GS_RGBA } else { GS_BGRA },
                        f.planes,
                        fds.as_ptr(),
                        f.stride.as_ptr(),
                        f.off.as_ptr(),
                        m.as_ptr(),
                    ),
                    ptr::null_mut(),
                ],
                _ => [ptr::null_mut(); 2],
            }
        };
        if t[0].is_null() || f.fourcc == NV12 && t[1].is_null() {
            drop(Tex { key, t });
            if !mem::replace(&mut self.failed, true) {
                let c = f.fourcc.to_le_bytes();
                log(
                    LOG_WARNING,
                    &format!(
                        "import failed: {} {}x{} modifier {:#x}, {} planes",
                        String::from_utf8_lossy(&c),
                        f.w,
                        f.h,
                        f.modifier,
                        f.planes
                    ),
                );
            }
            return false;
        }
        self.failed = false;
        self.cache[i] = Some(Tex { key, t });
        true
    }
}

unsafe extern "C" fn get_name(_: *mut c_void) -> *const c_char {
    c"X5 Camera (x5desk)".as_ptr()
}

unsafe extern "C" fn get_defaults(d: *mut obs_data_t) {
    unsafe {
        obs_data_set_default_string(d, c"source".as_ptr(), c"".as_ptr());
        obs_data_set_default_string(d, c"socket".as_ptr(), c"".as_ptr());
    }
}

unsafe extern "C" fn update(d: *mut c_void, set: *mut obs_data_t) {
    let s = unsafe { &*(d as *const Src) };
    let get = |k: &CStr| unsafe {
        let v = obs_data_get_string(set, k.as_ptr());
        if v.is_null() {
            "".into()
        } else {
            CStr::from_ptr(v).to_string_lossy()
        }
    };
    let mut st = s.sh.st.lock().unwrap();
    st.want = get(c"source").into_owned();
    st.socket = get(c"socket").into_owned();
    st.rev += 1;
}

unsafe extern "C" fn refresh(
    _: *mut obs_properties_t,
    _: *mut obs_property_t,
    _: *mut c_void,
) -> bool {
    true
}

unsafe extern "C" fn get_properties(d: *mut c_void) -> *mut obs_properties_t {
    let cs = |s: &str| CString::new(s.replace('\0', "")).unwrap_or_default();
    unsafe {
        let p = obs_properties_create();
        let l = obs_properties_add_list(
            p,
            c"source".as_ptr(),
            c"x5desk source".as_ptr(),
            OBS_COMBO_TYPE_LIST,
            OBS_COMBO_FORMAT_STRING,
        );
        obs_property_list_add_string(l, c"First available".as_ptr(), c"".as_ptr());
        let state = match (d as *const Src).as_ref() {
            None => String::new(),
            Some(s) => {
                let st = s.sh.st.lock().unwrap();
                for (_, n) in &st.list {
                    let n = cs(n);
                    obs_property_list_add_string(l, n.as_ptr(), n.as_ptr());
                }
                if !st.want.is_empty() && !st.list.iter().any(|x| x.1 == st.want) {
                    let w = cs(&st.want);
                    let n = cs(&format!("{} (not available)", st.want));
                    obs_property_list_add_string(l, n.as_ptr(), w.as_ptr());
                }
                let name = st.list.iter().find(|x| x.0 == st.id).map_or("", |x| &x.1);
                match (st.up, st.id, st.fmt) {
                    (false, ..) => format!("Not connected: {}", st.err),
                    (true, 0, _) => format!("Connected, {} sources, none selected", st.list.len()),
                    (true, _, None) => format!("Showing {name}, waiting for its format"),
                    (true, _, Some(f)) => format!(
                        "Showing {name}: {}x{} {} at {}/{} fps, modifier {:#x}; audio {} Hz × {}",
                        f.w,
                        f.h,
                        String::from_utf8_lossy(&f.fourcc.to_le_bytes()),
                        f.fps[0],
                        f.fps[1].max(1),
                        f.modifier,
                        f.rate,
                        f.ch
                    ),
                }
            }
        };
        let st = cs(&state);
        obs_properties_add_text(p, c"state".as_ptr(), st.as_ptr(), OBS_TEXT_INFO);
        obs_properties_add_button2(
            p,
            c"refresh".as_ptr(),
            c"Refresh".as_ptr(),
            refresh,
            ptr::null_mut(),
        );
        obs_properties_add_text(
            p,
            c"socket".as_ptr(),
            c"Socket (empty: $XDG_RUNTIME_DIR/x5desk/obs.sock)".as_ptr(),
            OBS_TEXT_DEFAULT,
        );
        p
    }
}

/// The modifiers OBS imports `fourcc` with, without INVALID.
unsafe fn mods(fourcc: u32) -> Vec<u64> {
    let (mut p, mut n) = (ptr::null_mut(), 0);
    unsafe {
        if !gs_query_dmabuf_modifiers_for_format(fourcc, &mut p, &mut n) || p.is_null() {
            return vec![];
        }
        let v = std::slice::from_raw_parts(p, n)
            .iter()
            .copied()
            .filter(|&m| m != MOD_INVALID)
            .collect();
        bfree(p.cast());
        v
    }
}

unsafe extern "C" fn create(set: *mut obs_data_t, src: *mut obs_source_t) -> *mut c_void {
    let (fmts, sync, eff) = unsafe {
        obs_enter_graphics();
        let (r8, gr88) = (mods(R8), mods(GR88));
        // NV12 is imported plane by plane
        let nv12 = r8.into_iter().filter(|m| gr88.contains(m)).collect();
        let fmts = vec![(AB24, mods(AB24)), (AR24, mods(AR24)), (NV12, nv12)];
        let sync = gs_query_sync_capabilities();
        let mut err = ptr::null_mut();
        let eff = gs_effect_create(EFFECT.as_ptr(), c"x5desk-nv12.effect".as_ptr(), &mut err);
        obs_leave_graphics();
        if !err.is_null() {
            log(
                LOG_WARNING,
                &format!("NV12 effect: {}", CStr::from_ptr(err).to_string_lossy()),
            );
            bfree(err.cast());
        }
        (fmts, sync, eff)
    };
    let obj = sync.then(drm::Obj::new).flatten();
    log(
        LOG_INFO,
        &format!(
            "imports {}; explicit release fences: {}",
            fmts.iter()
                .map(|(f, m)| format!("{} {m:x?}", String::from_utf8_lossy(&f.to_le_bytes())))
                .collect::<Vec<_>>()
                .join(", "),
            match (sync, obj.is_some()) {
                (true, true) => "yes",
                (true, false) => "no (no DRM syncobj), implicit sync",
                _ => "no (OBS lacks sync support), implicit sync",
            }
        ),
    );
    let sh = Arc::new(Sh {
        st: Default::default(),
        quit: AtomicBool::new(false),
        size: AtomicU64::new(0),
        tick: AtomicU64::new(0),
        src,
        fmts,
        obj,
    });
    let mut s = Box::new(Src {
        sh: sh.clone(),
        th: None,
        gfx: Mutex::new(Gfx {
            cache: vec![],
            ep: 0,
            cur: None,
            eff,
            sync: sync && sh.obj.is_some(),
            point: 0,
            failed: false,
            tick: 0,
            t0: unsafe { os_gettime_ns() },
            shown: 0,
            lat: (0, 0),
            starved: 0,
        }),
    });
    unsafe { update(&mut *s as *mut Src as *mut c_void, set) };
    match thread::Builder::new()
        .name("x5desk-obs".into())
        .spawn(move || run(&sh))
    {
        Ok(t) => s.th = Some(t),
        Err(e) => log(LOG_WARNING, &format!("no receive thread: {e}")),
    }
    Box::into_raw(s).cast()
}

unsafe extern "C" fn destroy(d: *mut c_void) {
    let mut s = unsafe { Box::from_raw(d as *mut Src) };
    s.sh.quit.store(true, Relaxed);
    if let Some(t) = s.th.take() {
        let _ = t.join();
    }
    let g = s.gfx.get_mut().unwrap();
    unsafe {
        obs_enter_graphics();
        g.cache.clear();
        if !g.eff.is_null() {
            gs_effect_destroy(g.eff);
        }
        obs_leave_graphics();
    }
}

unsafe extern "C" fn get_width(d: *mut c_void) -> u32 {
    (unsafe { &*(d as *const Src) }.sh.size.load(Relaxed) >> 32) as u32
}

unsafe extern "C" fn get_height(d: *mut c_void) -> u32 {
    unsafe { &*(d as *const Src) }.sh.size.load(Relaxed) as u32
}

unsafe extern "C" fn show(d: *mut c_void) {
    let mut st = unsafe { &*(d as *const Src) }.sh.st.lock().unwrap();
    (st.shown, st.rev) = (true, st.rev + 1);
}

unsafe extern "C" fn hide(d: *mut c_void) {
    let mut st = unsafe { &*(d as *const Src) }.sh.st.lock().unwrap();
    (st.shown, st.rev) = (false, st.rev + 1);
}

unsafe extern "C" fn video_tick(d: *mut c_void, _: f32) {
    unsafe { &*(d as *const Src) }.sh.tick.fetch_add(1, Relaxed);
}

unsafe extern "C" fn video_render(d: *mut c_void, _: *mut gs_effect_t) {
    let s = unsafe { &*(d as *const Src) };
    let mut g = s.gfx.lock().unwrap();
    let g = &mut *g;
    let tick = s.sh.tick.load(Relaxed);
    let (new, ep, color) = {
        let mut st = s.sh.st.lock().unwrap();
        // one frame per OBS tick, however often OBS renders the source in it
        let next = mem::replace(&mut g.tick, tick) != tick;
        let n = next.then(|| st.q.pop_front()).flatten();
        g.starved += (next && n.is_none() && g.cur.is_some()) as u32;
        (n, st.epoch, st.fmt.map_or(0, |f| f.color))
    };
    // another source or connection: the frames and buffers of the old one are gone
    if g.ep != ep {
        (g.ep, g.cur) = (ep, None);
        g.cache.clear();
    }
    if let Some(n) = new.filter(|n| n.epoch == ep) {
        // every draw of the old frame is behind this fence
        if let Some(c) = g.cur.take() {
            let p = unsafe { g.fence(&s.sh) };
            s.sh.st.lock().unwrap().rel.push_back((c.buf, ep, p));
        }
        if unsafe { g.import(&n) } {
            g.cur = Some(Cur {
                buf: n.f.buf,
                ts: n.f.ts,
                drawn: false,
            });
        } else {
            s.sh.st.lock().unwrap().rel.push_back((n.f.buf, ep, 0));
        }
    }
    let Some(c) = g.cur.as_mut() else {
        return;
    };
    let Some(Some(t)) = g.cache.get(c.buf as usize) else {
        return;
    };
    let t = t.t;
    unsafe {
        let e = if t[1].is_null() {
            let e = obs_get_base_effect(OBS_EFFECT_DEFAULT);
            gs_effect_set_texture(gs_effect_get_param_by_name(e, c"image".as_ptr()), t[0]);
            e
        } else {
            let e = g.eff;
            if e.is_null() {
                return;
            }
            let p = |n: &CStr| gs_effect_get_param_by_name(e, n.as_ptr());
            gs_effect_set_texture(p(c"image"), t[0]);
            gs_effect_set_texture(p(c"chroma"), t[1]);
            let [r, gr, b] = yuv(color);
            gs_effect_set_vec4(p(c"yuv_r"), &r);
            gs_effect_set_vec4(p(c"yuv_g"), &gr);
            gs_effect_set_vec4(p(c"yuv_b"), &b);
            e
        };
        while gs_effect_loop(e, c"Draw".as_ptr()) {
            gs_draw_sprite(t[0], 0, 0, 0);
        }
    }
    let now = unsafe { os_gettime_ns() };
    if !mem::replace(&mut c.drawn, true) {
        let l = now.saturating_sub(c.ts);
        g.shown += 1;
        g.lat = (g.lat.0 + l, g.lat.1.max(l));
    }
    if now - g.t0 >= 5_000_000_000 {
        let (sup, wait, def) = {
            let st = &mut *s.sh.st.lock().unwrap();
            (
                mem::take(&mut st.superseded),
                mem::take(&mut st.waited),
                mem::take(&mut st.deferred),
            )
        };
        let n = g.shown.max(1) as f64;
        log(
            LOG_INFO,
            &format!(
                "{:.1} fps shown, {sup} dropped from a full queue, {} ticks without a new frame, latency to \
                 first draw avg {:.1} max {:.1} ms; {wait} sync_files pending on arrival, {def} releases held for \
                 OBS's fence",
                g.shown as f64 * 1e9 / (now - g.t0) as f64,
                g.starved,
                g.lat.0 as f64 / n / 1e6,
                g.lat.1 as f64 / 1e6
            ),
        );
        (g.t0, g.shown, g.lat, g.starved) = (now, 0, (0, 0), 0);
    }
}

static MODULE: AtomicPtr<obs_module_t> = AtomicPtr::new(ptr::null_mut());

#[unsafe(no_mangle)]
pub extern "C" fn obs_module_set_pointer(m: *mut obs_module_t) {
    MODULE.store(m, Relaxed);
}

#[unsafe(no_mangle)]
pub extern "C" fn obs_current_module() -> *mut obs_module_t {
    MODULE.load(Relaxed)
}

#[unsafe(no_mangle)]
pub extern "C" fn obs_module_ver() -> u32 {
    LIBOBS_API_VER
}

#[unsafe(no_mangle)]
pub extern "C" fn obs_module_name() -> *const c_char {
    c"x5desk".as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn obs_module_description() -> *const c_char {
    c"Insta360 X5 sources from the x5desk app, zero-copy over DMA-BUF".as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn obs_module_load() -> bool {
    let i = SourceInfo {
        id: c"x5desk_source".as_ptr(),
        ty: OBS_SOURCE_TYPE_INPUT,
        output_flags: OBS_SOURCE_VIDEO | OBS_SOURCE_AUDIO | OBS_SOURCE_CUSTOM_DRAW,
        get_name: Some(get_name),
        create: Some(create),
        destroy: Some(destroy),
        get_width: Some(get_width),
        get_height: Some(get_height),
        get_defaults: Some(get_defaults),
        get_properties: Some(get_properties),
        update: Some(update),
        activate: None,
        deactivate: None,
        show: Some(show),
        hide: Some(hide),
        video_tick: Some(video_tick),
        video_render: Some(video_render),
        _b: [None; 20],
        icon_type: OBS_ICON_TYPE_CAMERA,
    };
    unsafe { obs_register_source_s(&i, mem::size_of::<SourceInfo>()) };
    true
}
