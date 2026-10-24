//! UVC capture of the X5's Webcam Mode through V4L2 (mmap streaming, MJPEG). Requesting 2880×1440 selects the
//! in-camera stitched equirect, 1920×1080 the flat view (PLAN.md §3.2).
//!
//! The camera streams over a bulk endpoint and keeps producing while nobody reads, so a new stream first delivers
//! stale data at bus speed: up to ~130 buffers, part of them not even frame-aligned. Frames are dropped until one
//! arrives at the camera's own pace (≥ 20 ms after the previous buffer), and so are frames of another size (right
//! after a mode switch the backlog holds 1–3 frames of the previous mode).

use crate::mjpeg;
use bytemuck::Zeroable;
use std::{
    fs::{self, OpenOptions},
    io,
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::fs::OpenOptionsExt,
    },
    path::{Path, PathBuf},
    time::Duration,
};

#[repr(C)]
#[derive(Zeroable)]
struct Cap {
    driver: [u8; 16],
    card: [u8; 32],
    bus: [u8; 32],
    version: u32,
    caps: u32,
    dev_caps: u32,
    res: [u32; 3],
}

#[repr(C)]
#[derive(Zeroable)]
struct Pix {
    width: u32,
    height: u32,
    format: u32,
    field: u32,
    stride: u32,
    size: u32,
    colorspace: u32,
    private: u32,
    flags: u32,
    enc: u32,
    quant: u32,
    xfer: u32,
}

/// `v4l2_format`: the union after `type` is 8-aligned on 64-bit (it holds pointers in `v4l2_window`).
#[repr(C)]
#[derive(Zeroable)]
struct Format {
    kind: u32,
    _pad: u32,
    pix: Pix,
    _raw: [u64; 19], // rest of the 200-byte union
}

#[repr(C)]
#[derive(Zeroable)]
struct Reqbufs {
    count: u32,
    kind: u32,
    memory: u32,
    caps: u32,
    flags: [u8; 4],
}

#[repr(C)]
#[derive(Zeroable)]
struct Buf {
    index: u32,
    kind: u32,
    used: u32,
    flags: u32,
    field: u32,
    ts: [i64; 2], // timeval, CLOCK_MONOTONIC
    timecode: [u32; 4],
    seq: u32,
    memory: u32,
    offset: u32, // union { offset, userptr, planes, fd }: 8 bytes, offset first (little endian)
    _m: u32,
    length: u32,
    _res2: u32,
    _req_fd: u32,
}

const _: () = assert!(size_of::<Cap>() == 104 && size_of::<Format>() == 208);
const _: () = assert!(size_of::<Reqbufs>() == 20 && size_of::<Buf>() == 88);

const fn ioc(dir: u64, nr: u64, size: usize) -> u64 {
    dir << 30 | (size as u64) << 16 | (b'V' as u64) << 8 | nr
}
const QUERYCAP: u64 = ioc(2, 0, size_of::<Cap>());
const S_FMT: u64 = ioc(3, 5, size_of::<Format>());
const REQBUFS: u64 = ioc(3, 8, size_of::<Reqbufs>());
const QUERYBUF: u64 = ioc(3, 9, size_of::<Buf>());
const QBUF: u64 = ioc(3, 15, size_of::<Buf>());
const DQBUF: u64 = ioc(3, 17, size_of::<Buf>());
const STREAMON: u64 = ioc(1, 18, 4);
const STREAMOFF: u64 = ioc(1, 19, 4);

const CAPTURE: u32 = 1; // V4L2_BUF_TYPE_VIDEO_CAPTURE / V4L2_CAP_VIDEO_CAPTURE
const MMAP: u32 = 1;
const CAP_STREAMING: u32 = 0x0400_0000;
const CAP_DEVICE_CAPS: u32 = 0x8000_0000;
const NBUF: u32 = 4;

fn ioctl<T>(fd: &OwnedFd, req: u64, arg: &mut T) -> io::Result<()> {
    loop {
        // SAFETY: `arg` is the repr(C) struct this request number encodes (sizes asserted above).
        if unsafe { libc::ioctl(fd.as_raw_fd(), req as _, arg as *mut T) } >= 0 {
            return Ok(());
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

fn now() -> f64 {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid out pointer
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as f64 + t.tv_nsec as f64 * 1e-9
}

fn open(p: &Path) -> io::Result<OwnedFd> {
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(p)?
        .into())
}

pub struct Uvc {
    fd: OwnedFd,
    maps: Vec<(*mut libc::c_void, usize)>,
    pub w: u32,
    pub h: u32,
    /// Timestamp of the previous buffer, s.
    prev: f64,
    live: bool,
}

// SAFETY: the mappings belong to this handle alone.
unsafe impl Send for Uvc {}

impl Uvc {
    /// The capture node of an X5 in Webcam Mode (card "Insta360 X5"; UVC also adds a metadata node).
    pub fn find() -> io::Result<PathBuf> {
        let mut nodes: Vec<_> = fs::read_dir("/dev")?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("video"))
            })
            .collect();
        nodes.sort();
        for p in nodes {
            let Ok(fd) = open(&p) else { continue };
            let mut c = Cap::zeroed();
            if ioctl(&fd, QUERYCAP, &mut c).is_err() {
                continue;
            }
            let caps = if c.caps & CAP_DEVICE_CAPS != 0 {
                c.dev_caps
            } else {
                c.caps
            };
            let card = c.card.split(|&b| b == 0).next().unwrap_or(&[]);
            if card == b"Insta360 X5" && caps & CAPTURE != 0 && caps & CAP_STREAMING != 0 {
                return Ok(p);
            }
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no Insta360 X5 webcam",
        ))
    }

    /// Starts MJPEG streaming at `w`×`h` (2880×1440 or 1920×1080).
    pub fn open(path: impl AsRef<Path>, w: u32, h: u32) -> io::Result<Uvc> {
        let fd = open(path.as_ref())?;
        let mut f = Format::zeroed();
        f.kind = CAPTURE;
        (f.pix.width, f.pix.height, f.pix.format, f.pix.field) =
            (w, h, u32::from_le_bytes(*b"MJPG"), 1);
        ioctl(&fd, S_FMT, &mut f)?;
        if (f.pix.width, f.pix.height, f.pix.format) != (w, h, u32::from_le_bytes(*b"MJPG")) {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("camera offers {}×{}", f.pix.width, f.pix.height),
            ));
        }
        let mut r = Reqbufs {
            count: NBUF,
            kind: CAPTURE,
            memory: MMAP,
            ..Zeroable::zeroed()
        };
        ioctl(&fd, REQBUFS, &mut r)?;
        let mut u = Uvc {
            fd,
            maps: vec![],
            w,
            h,
            // the wait from STREAMON to the first buffer says nothing about its age: only gaps between buffers count
            prev: f64::INFINITY,
            live: false,
        };
        for i in 0..r.count {
            let mut b = Buf {
                index: i,
                kind: CAPTURE,
                memory: MMAP,
                ..Zeroable::zeroed()
            };
            ioctl(&u.fd, QUERYBUF, &mut b)?;
            let len = b.length as usize;
            // SAFETY: maps the driver buffer `i` at the offset QUERYBUF returned; unmapped in Drop.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    u.fd.as_raw_fd(),
                    b.offset as libc::off_t,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            u.maps.push((p, len));
            ioctl(&u.fd, QBUF, &mut b)?;
        }
        ioctl(&u.fd, STREAMON, &mut (CAPTURE as i32))?;
        Ok(u)
    }

    /// Waits up to `timeout` for the next fresh, complete frame and copies the JPEG into `out`. Ok(false) on timeout.
    pub fn next(&mut self, out: &mut Vec<u8>, timeout: Duration) -> io::Result<bool> {
        let until = now() + timeout.as_secs_f64();
        loop {
            let left = until - now();
            if left <= 0.0 {
                return Ok(false);
            }
            let mut pfd = libc::pollfd {
                fd: self.fd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one valid pollfd
            match unsafe { libc::poll(&mut pfd, 1, (left * 1e3).ceil() as i32) } {
                0 => return Ok(false),
                n if n < 0 => match io::Error::last_os_error() {
                    e if e.kind() == io::ErrorKind::Interrupted => continue,
                    e => return Err(e),
                },
                _ => {}
            }
            let mut b = Buf {
                kind: CAPTURE,
                memory: MMAP,
                ..Zeroable::zeroed()
            };
            match ioctl(&self.fd, DQBUF, &mut b) {
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => continue,
                r => r?,
            }
            let t = b.ts[0] as f64 + b.ts[1] as f64 * 1e-6;
            self.live |= t - self.prev >= 0.02;
            self.prev = t;
            let (p, len) = self.maps[b.index as usize];
            // SAFETY: the driver filled `used` bytes of this dequeued buffer; we own it until QBUF.
            let d =
                unsafe { std::slice::from_raw_parts(p as *const u8, (b.used as usize).min(len)) };
            // a buffer can hold two frames (seen in captures) or a partial one: take its last complete frame
            let jpg = if self.live {
                mjpeg::frames(d)
                    .filter(|j| mjpeg::size(j) == Some((self.w, self.h)))
                    .last()
            } else {
                None
            };
            let ok = jpg.is_some();
            if let Some(j) = jpg {
                out.clear();
                out.extend_from_slice(j);
            }
            ioctl(&self.fd, QBUF, &mut b)?;
            if ok {
                return Ok(true);
            }
        }
    }
}

impl Drop for Uvc {
    fn drop(&mut self) {
        let _ = ioctl(&self.fd, STREAMOFF, &mut (CAPTURE as i32));
        for &(p, len) in &self.maps {
            // SAFETY: mapped in `open`, not referenced after this
            unsafe { libc::munmap(p, len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only QUERYCAP runs here; streaming needs the camera in Webcam Mode.
    #[test]
    fn find() {
        match Uvc::find() {
            Ok(p) => println!("X5 webcam at {}", p.display()),
            Err(e) => {
                assert_eq!(e.kind(), io::ErrorKind::NotFound);
                println!("no X5 webcam: {e}");
            }
        }
        assert_eq!(QUERYCAP, 0x8068_5600);
        assert_eq!(
            (S_FMT, REQBUFS, QUERYBUF, QBUF, DQBUF),
            (
                0xc0d0_5605,
                0xc014_5608,
                0xc058_5609,
                0xc058_560f,
                0xc058_5611
            )
        );
        assert_eq!((STREAMON, STREAMOFF), (0x4004_5612, 0x4004_5613));
    }
}
