//! OBS frame protocol between the app (server: several sources, several clients) and our OBS
//! plugin (client). Self-contained, std and libc only, so the plugin can share this file.
//!
//! A `SOCK_SEQPACKET` Unix socket, by default `$XDG_RUNTIME_DIR/x5desk/obs.sock`; one message per
//! packet, native byte order (same host). Header [`Hdr`], then `len` bytes of payload; file
//! descriptors travel with `SCM_RIGHTS`.
//!
//! | kind   | from   | payload | fds |
//! |--------|--------|---------|-----|
//! | hello  | client | the formats it imports, each `fourcc u32, n u32, modifier[n] u64` (none: LINEAR only); `src` = the source to receive, 0 for none; switching drops the frames it holds | |
//! | hello  | server | its sources as `id u32, n u32, name[n]`; the reply to hello, and again on changes | |
//! | format | server | [`Format`] | |
//! | frame  | server | [`Frame`] | one DMA-BUF per plane, then the sync_file if `sync` |
//! | done   | client | `buf u32`: the client no longer reads that buffer of source `src` | |
//! | audio  | server | `ts u64, rate u32, ch u32`, then interleaved f32 | |
//! | bye    | either | none; from the server `src` = the source that ended, 0 = shutting down | |
//!
//! Timestamps are `CLOCK_MONOTONIC` ns, the clock of OBS's `os_gettime_ns`. A client holds each
//! frame it gets until it answers `done`; one that holds [`HOLD`] frames misses the next ones, so
//! a stalled client pins neither buffers nor file descriptors beyond that, and resumes with the
//! newest frame. A full socket also just misses the message (the server never blocks); only a
//! broken connection or a bad message drops a client. A client's sends don't block either: with
//! the app stalled they return `WouldBlock`, and the client retries.
//!
//! Modifiers: every client takes LINEAR. The app exports a source in a modifier all its clients
//! list ([`Server::modifiers`]) and re-exports when [`Server::poll`] reports a change that alters
//! that list; each [`Frame`] names its modifier.

use std::{
    ffi::OsStr,
    fs::DirBuilder,
    io,
    mem::{self, size_of, size_of_val},
    os::{
        fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd},
        unix::{ffi::OsStrExt, fs::DirBuilderExt},
    },
    path::{Path, PathBuf},
    ptr, slice,
    time::Duration,
};

pub const MAGIC: u32 = u32::from_le_bytes(*b"X5OB");
pub const VERSION: u16 = 1;
pub const HELLO: u8 = 1;
pub const FORMAT: u8 = 2;
pub const FRAME: u8 = 3;
pub const AUDIO: u8 = 4;
pub const BYE: u8 = 5;
pub const DONE: u8 = 6;
/// Frames a client may hold, sent and not yet `done`: our plugin's queue of 2, the frame it shows and the one before
/// until OBS's GPU is done with it.
pub const HOLD: usize = 4;
/// Largest message, header included. Longer audio is split.
pub const MAX: usize = 65536;
/// Largest client message, header included (a hello with its formats).
pub const MAX_HELLO: usize = 4096;
const HDR: usize = size_of::<Hdr>();

pub const BT601: u32 = 1;
pub const BT709: u32 = 2;
pub const FULL: u32 = 0x100;
pub const MOD_LINEAR: u64 = 0;

pub const fn fourcc(c: [u8; 4]) -> u32 {
    u32::from_le_bytes(c)
}

// The wire structs are repr(C) without padding, and every bit pattern is a valid value.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Hdr {
    pub magic: u32,
    pub ver: u16,
    pub kind: u8,
    /// File descriptors attached.
    pub fds: u8,
    pub src: u32,
    /// Payload bytes after the header.
    pub len: u32,
}

/// A source's stream; `w` 0 = no video, `rate` 0 = no audio.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Format {
    /// DRM format modifier of the frames.
    pub modifier: u64,
    pub w: u32,
    pub h: u32,
    /// DRM fourcc of the frames.
    pub fourcc: u32,
    pub fps: [u32; 2],
    /// YUV matrix ([`BT601`], [`BT709`]) | [`FULL`] range.
    pub color: u32,
    pub rate: u32,
    pub ch: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Frame {
    pub ts: u64,
    pub modifier: u64,
    pub w: u32,
    pub h: u32,
    pub fourcc: u32,
    /// The exporter's buffer index, so a client can cache its import of a recycled buffer.
    pub buf: u32,
    /// Plane fds attached (set by [`Server::frame`]).
    pub planes: u32,
    /// 1 if a sync_file follows the plane fds (set by [`Server::frame`]).
    pub sync: u32,
    pub off: [u32; 4],
    pub stride: [u32; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Pcm {
    ts: u64,
    rate: u32,
    ch: u32,
}

fn bytes<T: ?Sized>(t: &T) -> &[u8] {
    unsafe { slice::from_raw_parts((t as *const T).cast(), size_of_val(t)) }
}

fn read<T: Copy>(b: &[u8]) -> io::Result<T> {
    if b.len() < size_of::<T>() {
        return Err(bad("short message"));
    }
    Ok(unsafe { b.as_ptr().cast::<T>().read_unaligned() })
}

fn bad(m: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m)
}

fn cvt(r: isize) -> io::Result<usize> {
    if r < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(r as usize)
    }
}

fn iov(b: &[u8]) -> libc::iovec {
    libc::iovec {
        iov_base: b.as_ptr() as *mut _,
        iov_len: b.len(),
    }
}

fn send(
    fd: RawFd,
    kind: u8,
    src: u32,
    p: &[u8],
    tail: &[u8],
    fds: &[RawFd],
    flags: i32,
) -> io::Result<()> {
    let h = Hdr {
        magic: MAGIC,
        ver: VERSION,
        kind,
        fds: fds.len() as u8,
        src,
        len: (p.len() + tail.len()) as u32,
    };
    let mut v = [iov(bytes(&h)), iov(p), iov(tail)];
    let mut c = [0u64; 8];
    let mut m: libc::msghdr = unsafe { mem::zeroed() };
    m.msg_iov = v.as_mut_ptr();
    m.msg_iovlen = v.len() as _;
    if !fds.is_empty() {
        let n = size_of_val(fds) as u32;
        m.msg_control = c.as_mut_ptr().cast();
        m.msg_controllen = unsafe { libc::CMSG_SPACE(n) } as _;
        unsafe {
            let h = libc::CMSG_FIRSTHDR(&m);
            (*h).cmsg_level = libc::SOL_SOCKET;
            (*h).cmsg_type = libc::SCM_RIGHTS;
            (*h).cmsg_len = libc::CMSG_LEN(n) as _;
            ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(h).cast(), fds.len());
        }
    }
    cvt(unsafe { libc::sendmsg(fd, &m, flags | libc::MSG_NOSIGNAL) }).map(drop)
}

/// One message into `b`, its fds into `fds`; returns the checked header.
fn recv(fd: RawFd, b: &mut [u8], fds: &mut [Option<OwnedFd>; 5], flags: i32) -> io::Result<Hdr> {
    fds.iter_mut().for_each(|f| *f = None);
    let mut v = iov(b);
    let mut c = [0u64; 8];
    let mut m: libc::msghdr = unsafe { mem::zeroed() };
    m.msg_iov = &mut v;
    m.msg_iovlen = 1;
    m.msg_control = c.as_mut_ptr().cast();
    m.msg_controllen = size_of_val(&c) as _;
    let n = cvt(unsafe { libc::recvmsg(fd, &mut m, flags | libc::MSG_CMSG_CLOEXEC) })?;
    // own every fd first, so none leaks whatever else is wrong
    let mut k = 0;
    unsafe {
        let mut h = libc::CMSG_FIRSTHDR(&m);
        while !h.is_null() {
            if (*h).cmsg_level == libc::SOL_SOCKET && (*h).cmsg_type == libc::SCM_RIGHTS {
                let d = libc::CMSG_DATA(h).cast::<RawFd>();
                for i in
                    0..((*h).cmsg_len as usize - libc::CMSG_LEN(0) as usize) / size_of::<RawFd>()
                {
                    let f = OwnedFd::from_raw_fd(d.add(i).read_unaligned());
                    if let Some(s) = fds.get_mut(k) {
                        *s = Some(f);
                    }
                    k += 1;
                }
            }
            h = libc::CMSG_NXTHDR(&m, h);
        }
    }
    if n == 0 {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    if m.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0 {
        return Err(bad("truncated message"));
    }
    let h: Hdr = read(&b[..n])?;
    if h.magic != MAGIC || h.ver != VERSION {
        return Err(bad("not x5desk OBS protocol version 1"));
    }
    if h.len as usize != n - HDR || h.fds as usize != k {
        return Err(bad("message length or fd count mismatch"));
    }
    Ok(h)
}

fn sock(flags: i32) -> io::Result<OwnedFd> {
    let fd = cvt(unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC | flags,
            0,
        )
    } as isize)?;
    Ok(unsafe { OwnedFd::from_raw_fd(fd as RawFd) })
}

fn addr(p: &Path) -> io::Result<libc::sockaddr_un> {
    let mut a: libc::sockaddr_un = unsafe { mem::zeroed() };
    a.sun_family = libc::AF_UNIX as _;
    let b = p.as_os_str().as_bytes();
    if b.len() >= a.sun_path.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "socket path too long",
        ));
    }
    a.sun_path.iter_mut().zip(b).for_each(|(d, s)| *d = *s as _);
    Ok(a)
}

/// Never waits: a server whose listen backlog is full gives `WouldBlock`.
fn connect(p: &Path) -> io::Result<OwnedFd> {
    let (a, s) = (addr(p)?, sock(libc::SOCK_NONBLOCK)?);
    let n = size_of_val(&a) as u32;
    cvt(
        unsafe { libc::connect(s.as_raw_fd(), (&a as *const libc::sockaddr_un).cast(), n) }
            as isize,
    )?;
    Ok(s)
}

/// `$XDG_RUNTIME_DIR/x5desk/obs.sock`
pub fn path() -> io::Result<PathBuf> {
    let d = std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty());
    let d =
        d.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "XDG_RUNTIME_DIR is not set"))?;
    Ok(Path::new(&d).join("x5desk/obs.sock"))
}

/// A broken connection, unlike a full socket or a transient limit such as `ETOOMANYREFS` (this
/// user's fds in flight on all Unix sockets exceed `RLIMIT_NOFILE`).
fn gone(e: &io::Error) -> bool {
    use io::ErrorKind::*;
    matches!(e.kind(), BrokenPipe | ConnectionReset | NotConnected)
}

struct Cl {
    fd: OwnedFd,
    src: u32,
    /// Buffers of `src` it holds: `held[..n]`.
    held: [u32; HOLD],
    n: usize,
    /// (fourcc, modifiers) it imports besides LINEAR
    mods: Vec<(u32, Vec<u64>)>,
}

struct Src {
    id: u32,
    name: String,
    fmt: Option<Format>,
}

/// The app side. Driven by the caller: [`Server::poll`] accepts clients and reads their messages.
pub struct Server {
    l: OwnedFd,
    path: PathBuf,
    cl: Vec<Cl>,
    src: Vec<Src>,
    /// a client came, left, or changed its source or formats since the last poll
    changed: bool,
}

impl Server {
    /// Listens at `p`, replacing a stale socket; fails with `AddrInUse` if another server answers.
    pub fn bind(p: &Path) -> io::Result<Self> {
        if let Some(d) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            DirBuilder::new().recursive(true).mode(0o700).create(d)?;
        }
        match connect(p) {
            Err(e) if e.kind() != io::ErrorKind::WouldBlock => {}
            _ => return Err(io::ErrorKind::AddrInUse.into()),
        }
        match std::fs::remove_file(p) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
        let (a, l) = (addr(p)?, sock(libc::SOCK_NONBLOCK)?);
        let n = size_of_val(&a) as u32;
        cvt(
            unsafe { libc::bind(l.as_raw_fd(), (&a as *const libc::sockaddr_un).cast(), n) }
                as isize,
        )?;
        cvt(unsafe { libc::listen(l.as_raw_fd(), 16) } as isize)?;
        Ok(Self {
            l,
            path: p.into(),
            cl: vec![],
            src: vec![],
            changed: false,
        })
    }

    /// Accepts clients and answers their messages; never blocks. True when a client came, left, or
    /// changed its source or formats since the last call: check [`Server::modifiers`] then.
    pub fn poll(&mut self) -> bool {
        let fl = libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK;
        while let Ok(fd) =
            cvt(
                unsafe { libc::accept4(self.l.as_raw_fd(), ptr::null_mut(), ptr::null_mut(), fl) }
                    as isize,
            )
        {
            self.cl.push(Cl {
                fd: unsafe { OwnedFd::from_raw_fd(fd as RawFd) },
                src: 0,
                held: [0; HOLD],
                n: 0,
                mods: vec![],
            });
            self.changed = true;
        }
        let (mut b, mut fds) = ([0u8; MAX_HELLO], Default::default());
        let mut i = 0;
        while i < self.cl.len() {
            let ok = loop {
                let c = &mut self.cl[i];
                let fd = c.fd.as_raw_fd();
                match recv(fd, &mut b, &mut fds, libc::MSG_DONTWAIT) {
                    Ok(h) if h.kind == HELLO => {
                        let (mut p, mut m) = (&b[HDR..HDR + h.len as usize], vec![]);
                        while let Ok([f, n]) = read::<[u32; 2]>(p) {
                            let Some(v) = p.get(8..8 + 8 * n as usize) else {
                                break;
                            };
                            let v = v.as_chunks().0.iter().map(|c| u64::from_ne_bytes(*c));
                            m.push((f, v.collect()));
                            p = &p[8 + 8 * n as usize..];
                        }
                        if !p.is_empty() {
                            break false;
                        }
                        // frames of the source it already has may still be queued for it
                        if h.src != c.src {
                            (c.src, c.n) = (h.src, 0);
                        }
                        c.mods = m;
                        self.changed = true;
                        let f = self.src.iter().find(|s| s.id == h.src).and_then(|s| s.fmt);
                        let r = send(fd, HELLO, 0, &self.list(), &[], &[], libc::MSG_DONTWAIT);
                        if r.and_then(|_| {
                            f.map_or(Ok(()), |f| {
                                send(fd, FORMAT, h.src, bytes(&f), &[], &[], libc::MSG_DONTWAIT)
                            })
                        })
                        .is_err_and(|e| gone(&e))
                        {
                            break false;
                        }
                    }
                    Ok(h) if h.kind == DONE => {
                        let Ok(buf) = read::<u32>(&b[HDR..HDR + h.len as usize]) else {
                            break false;
                        };
                        // a done for a source it switched away from no longer counts
                        if h.src == c.src
                            && let Some(k) = c.held[..c.n].iter().position(|&x| x == buf)
                        {
                            c.n -= 1;
                            c.held[k] = c.held[c.n];
                        }
                    }
                    Ok(h) if h.kind == BYE => break false,
                    Ok(_) => {}
                    Err(e) => break e.kind() == io::ErrorKind::WouldBlock,
                }
            };
            if ok {
                i += 1;
            } else {
                self.cl.swap_remove(i);
                self.changed = true;
            }
        }
        mem::take(&mut self.changed)
    }

    fn list(&self) -> Vec<u8> {
        let mut p = vec![];
        for s in &self.src {
            p.extend_from_slice(&s.id.to_ne_bytes());
            p.extend_from_slice(&(s.name.len() as u32).to_ne_bytes());
            p.extend_from_slice(s.name.as_bytes());
        }
        p
    }

    /// Sends to the clients of `src` (all with `src` 0); returns how many got it. A frame
    /// (`buf` set) skips clients that hold [`HOLD`] frames and counts as held by the others.
    /// Drops clients whose connection broke.
    fn cast(
        &mut self,
        kind: u8,
        src: u32,
        p: &[u8],
        tail: &[u8],
        fds: &[RawFd],
        buf: Option<u32>,
    ) -> usize {
        let mut n = 0;
        self.cl.retain_mut(|c| {
            if src != 0 && c.src != src || buf.is_some() && c.n == HOLD {
                return true;
            }
            match send(
                c.fd.as_raw_fd(),
                kind,
                src,
                p,
                tail,
                fds,
                libc::MSG_DONTWAIT,
            ) {
                Ok(()) => {
                    if let Some(b) = buf {
                        c.held[c.n] = b;
                        c.n += 1;
                    }
                    n += 1;
                    true
                }
                Err(e) => {
                    self.changed |= gone(&e);
                    !gone(&e)
                }
            }
        });
        n
    }

    /// Adds or renames source `id` (not 0) and tells every client.
    pub fn add(&mut self, id: u32, name: &str) {
        match self.src.iter_mut().find(|s| s.id == id) {
            Some(s) => s.name = name.into(),
            None => self.src.push(Src {
                id,
                name: name.into(),
                fmt: None,
            }),
        }
        self.cast(HELLO, 0, &self.list(), &[], &[], None);
    }

    /// Ends source `id`: bye to its clients, which then receive nothing until they select again,
    /// and the new list to every client.
    pub fn remove(&mut self, id: u32) {
        self.src.retain(|s| s.id != id);
        self.cast(BYE, id, &[], &[], &[], None);
        self.cl
            .iter_mut()
            .filter(|c| c.src == id)
            .for_each(|c| (c.src, c.n) = (0, 0));
        self.cast(HELLO, 0, &self.list(), &[], &[], None);
    }

    pub fn format(&mut self, id: u32, f: Format) {
        if let Some(s) = self.src.iter_mut().find(|s| s.id == id) {
            s.fmt = Some(f);
        }
        self.cast(FORMAT, id, bytes(&f), &[], &[], None);
    }

    /// Whether any client receives source `id`; without one the app can skip the export.
    pub fn watched(&self, id: u32) -> bool {
        self.cl.iter().any(|c| c.src == id)
    }

    /// How many clients receive source `id`.
    pub fn clients(&self, id: u32) -> usize {
        self.cl.iter().filter(|c| c.src == id).count()
    }

    /// Whether every client of source `id` imports `fourcc` with `modifier`; LINEAR they all do.
    pub fn takes(&self, id: u32, fourcc: u32, modifier: u64) -> bool {
        modifier == MOD_LINEAR
            || self.cl.iter().filter(|c| c.src == id).all(|c| {
                c.mods
                    .iter()
                    .any(|(f, m)| *f == fourcc && m.contains(&modifier))
            })
    }

    /// The modifiers besides LINEAR that every client of source `id` imports `fourcc` with, in the
    /// first client's order: the list for `dmabuf::Ring::new`, which falls back to LINEAR.
    pub fn modifiers(&self, id: u32, fourcc: u32) -> Vec<u64> {
        let Some(c) = self.cl.iter().find(|c| c.src == id) else {
            return vec![];
        };
        let all = c.mods.iter().filter(|m| m.0 == fourcc).flat_map(|m| &m.1);
        all.copied()
            .filter(|&m| m != MOD_LINEAR && self.takes(id, fourcc, m))
            .collect()
    }

    /// Whether a client still reads buffer `buf` of source `id`; the app must not write it then.
    pub fn busy(&self, id: u32, buf: u32) -> bool {
        self.cl
            .iter()
            .any(|c| c.src == id && c.held[..c.n].contains(&buf))
    }

    /// Sends a frame of source `id` with its plane DMA-BUFs (at most 4) and the sync_file that
    /// signals when they are written; returns how many clients got it. Those hold `f.buf` until
    /// done.
    pub fn frame(
        &mut self,
        id: u32,
        f: &Frame,
        planes: &[BorrowedFd],
        sync: Option<BorrowedFd>,
    ) -> usize {
        assert!(planes.len() <= 4, "{} planes", planes.len());
        let (n, s) = (planes.len(), sync.is_some() as usize);
        let mut fds = [-1; 5];
        fds.iter_mut()
            .zip(&planes[..n])
            .for_each(|(d, p)| *d = p.as_raw_fd());
        fds[n] = sync.map_or(-1, |s| s.as_raw_fd());
        let f = Frame {
            planes: n as u32,
            sync: s as u32,
            ..*f
        };
        self.cast(FRAME, id, bytes(&f), &[], &fds[..n + s], Some(f.buf))
    }

    /// Sends interleaved PCM of source `id`, split to fit [`MAX`]; returns how many clients got
    /// the last part. Nothing goes out if one sample per channel doesn't fit (`ch` > 16376).
    pub fn audio(&mut self, id: u32, ts: u64, rate: u32, ch: u32, pcm: &[f32]) -> usize {
        let ch = ch.max(1);
        let k = (MAX - HDR - size_of::<Pcm>()) / 4 / ch as usize * ch as usize;
        if k == 0 {
            return 0;
        }
        let mut n = 0;
        for (i, c) in pcm.chunks(k).enumerate() {
            let t = ts + (i * k / ch as usize) as u64 * 1_000_000_000 / rate.max(1) as u64;
            n = self.cast(
                AUDIO,
                id,
                bytes(&Pcm { ts: t, rate, ch }),
                bytes(c),
                &[],
                None,
            );
        }
        n
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.cast(BYE, 0, &[], &[], &[], None);
        let _ = std::fs::remove_file(&self.path);
    }
}

pub enum Msg<'a> {
    /// The server's sources, (id, name).
    Hello(Vec<(u32, String)>),
    Format(u32, Format),
    /// Wait for `sync` (readable = signalled) before reading the planes, and call
    /// [`Client::done`] once no longer reading them.
    Frame {
        src: u32,
        f: Frame,
        planes: [Option<OwnedFd>; 4],
        sync: Option<OwnedFd>,
    },
    Audio {
        src: u32,
        ts: u64,
        rate: u32,
        ch: u32,
        pcm: &'a [f32],
    },
    /// Source `0`: the server shut down.
    Bye(u32),
}

/// The plugin side: `recv` waits, sending never does.
pub struct Client {
    fd: OwnedFd,
    /// the hello payload: the formats it imports
    fmts: Vec<u8>,
    /// u64 so that the PCM after the 32 bytes of headers is f32-aligned.
    b: Box<[u64]>,
    fds: [Option<OwnedFd>; 5],
}

impl Client {
    /// Connects for source `src`, importing the (fourcc, modifiers) in `fmts` besides LINEAR
    /// (e.g. what `gs_query_dmabuf_modifiers_for_format` lists). `WouldBlock` if the server
    /// doesn't accept connections (a full listen backlog).
    pub fn connect(p: &Path, src: u32, fmts: &[(u32, &[u64])]) -> io::Result<Self> {
        let mut f = vec![];
        for (fourcc, m) in fmts {
            f.extend_from_slice(&fourcc.to_ne_bytes());
            f.extend_from_slice(&(m.len() as u32).to_ne_bytes());
            m.iter().for_each(|m| f.extend_from_slice(&m.to_ne_bytes()));
        }
        if HDR + f.len() > MAX_HELLO {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "formats exceed MAX_HELLO",
            ));
        }
        let fd = connect(p)?;
        cvt(unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, 0) } as isize)?;
        let c = Self {
            fd,
            fmts: f,
            b: vec![0; MAX / 8].into(),
            fds: Default::default(),
        };
        c.select(src)?;
        Ok(c)
    }

    /// Asks for source `src` (0: none); the server answers with its sources and that format.
    /// Switching sources drops the frames held so far: stop reading them, and send no `done`.
    /// Never blocks: `WouldBlock` while the server reads nothing (a stalled app); send it again
    /// later.
    pub fn select(&self, src: u32) -> io::Result<()> {
        send(
            self.fd.as_raw_fd(),
            HELLO,
            src,
            &self.fmts,
            &[],
            &[],
            libc::MSG_DONTWAIT,
        )
    }

    /// Hands buffer `buf` of source `src` back. Every frame received needs this, or after
    /// [`HOLD`] of them no more arrive. Never blocks: `WouldBlock` as for [`Client::select`],
    /// and the buffer counts as held until a later call gets through.
    pub fn done(&self, src: u32, buf: u32) -> io::Result<()> {
        send(
            self.fd.as_raw_fd(),
            DONE,
            src,
            &buf.to_ne_bytes(),
            &[],
            &[],
            libc::MSG_DONTWAIT,
        )
    }

    /// The next message: `TimedOut` after `timeout`, `UnexpectedEof` once the server is gone.
    pub fn recv(&mut self, timeout: Option<Duration>) -> io::Result<Msg<'_>> {
        let fd = self.fd.as_raw_fd();
        if let Some(t) = timeout {
            let mut p = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            if cvt(unsafe {
                let ms = t.as_nanos().div_ceil(1_000_000).min(i32::MAX as u128);
                libc::poll(&mut p, 1, ms as i32)
            } as isize)?
                == 0
            {
                return Err(io::ErrorKind::TimedOut.into());
            }
        }
        let b = unsafe {
            slice::from_raw_parts_mut(self.b.as_mut_ptr().cast::<u8>(), self.b.len() * 8)
        };
        let h = recv(fd, b, &mut self.fds, 0)?;
        let p = &b[HDR..HDR + h.len as usize];
        Ok(match h.kind {
            HELLO => {
                let (mut v, mut p) = (vec![], p);
                while p.len() >= 8 {
                    let (id, n): (u32, u32) = (read(p)?, read(&p[4..])?);
                    let s = p
                        .get(8..8 + n as usize)
                        .ok_or_else(|| bad("bad source list"))?;
                    v.push((id, OsStr::from_bytes(s).to_string_lossy().into_owned()));
                    p = &p[8 + n as usize..];
                }
                Msg::Hello(v)
            }
            FORMAT => Msg::Format(h.src, read(p)?),
            FRAME => {
                let f: Frame = read(p)?;
                let n = f.planes as usize;
                if n > 4 || n + (f.sync != 0) as usize != h.fds as usize {
                    return Err(bad("frame fd count mismatch"));
                }
                let mut planes: [Option<OwnedFd>; 4] = Default::default();
                planes
                    .iter_mut()
                    .zip(&mut self.fds[..n])
                    .for_each(|(d, s)| *d = s.take());
                let sync = if f.sync != 0 {
                    self.fds[n].take()
                } else {
                    None
                };
                Msg::Frame {
                    src: h.src,
                    f,
                    planes,
                    sync,
                }
            }
            AUDIO => {
                let a: Pcm = read(p)?;
                let s = &p[size_of::<Pcm>()..];
                let pcm = unsafe { slice::from_raw_parts(s.as_ptr().cast::<f32>(), s.len() / 4) };
                Msg::Audio {
                    src: h.src,
                    ts: a.ts,
                    rate: a.rate,
                    ch: a.ch,
                    pcm,
                }
            }
            BYE => Msg::Bye(h.src),
            _ => return Err(bad("unknown message kind")),
        })
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = send(
            self.fd.as_raw_fd(),
            BYE,
            0,
            &[],
            &[],
            &[],
            libc::MSG_DONTWAIT,
        );
    }
}
