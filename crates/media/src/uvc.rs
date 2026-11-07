//! UVC capture of the X5's Webcam Mode through V4L2 (mmap streaming, MJPEG). Requesting 2880×1440 selects the
//! in-camera stitched equirect, 1920×1080 the flat view (PLAN.md §3.2).
//!
//! The camera streams over a bulk endpoint and keeps producing while nobody reads, so a new stream first delivers
//! stale data at bus speed: up to ~130 buffers, part of them not even frame-aligned; a mode switch's backlog holds the
//! previous mode's frames (1–3 right after its stream, up to ~130 after the camera sat idle). After a switch (~1.7 s
//! without data) the camera may send the new mode's first frame ~20 times over at bus speed. A frame counts only when
//! its buffer comes [`FRESH`] or more a frame after the previous one (the driver numbers every frame, also those it
//! drops), and only at the size asked for.
//!
//! The camera's microphone (UAC, the same USB device) is [`mic`].

pub mod mic;

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
/// Bus-speed buffers (the stale backlog, a switch's copies) came ≤ 5.0 ms a frame apart on an X5, the camera's own
/// frames ≥ 10.1 ms (1920 × 1080; ≥ 19.5 ms at 2880 × 1440): the driver's start-of-frame timestamps of two buffers
/// over the frames it counted from one to the other (`v4l2_buffer.sequence`). uvcvideo counts every frame, also those
/// it drops while no buffer is queued, so a reader that stalls in the backlog sees one long gap over many frames
/// (14.6 ms over 33), never a fresh-looking one. The first new frame after a switch's copies came 7.9–13.6 ms after
/// the last copy (4 switches logged), so it can be dropped with them (1 of the 4).
const FRESH: f64 = 0.008;

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
    /// Timestamp of the previous buffer, s; the driver's number of its frame.
    prev: f64,
    seq: u32,
    /// Buffers that gave no frame: the stale backlog, frames cut off or of another size.
    pub skip: u64,
}

// SAFETY: the mappings belong to this handle alone.
unsafe impl Send for Uvc {}

impl Uvc {
    /// The capture node of an X5 in Webcam Mode (card "Insta360 X5[: …]"; UVC also adds a metadata node).
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
            // uvcvideo names the card after the USB product and appends ": <iFunction or iInterface>" when the device
            // has one: the X5's is "Insta360 X5: Insta360 X5"
            let card = c.card.split(|&b| b == 0 || b == b':').next();
            if card == Some(&b"Insta360 X5"[..]) && caps & CAPTURE != 0 && caps & CAP_STREAMING != 0
            {
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
            seq: 0,
            skip: 0,
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
            let (p, len) = self.maps[b.index as usize];
            // SAFETY: the driver filled `used` bytes of this dequeued buffer; we own it until QBUF.
            let d =
                unsafe { std::slice::from_raw_parts(p as *const u8, (b.used as usize).min(len)) };
            let jpg = pick(
                d,
                t - std::mem::replace(&mut self.prev, t),
                b.seq.wrapping_sub(std::mem::replace(&mut self.seq, b.seq)),
                (self.w, self.h),
            );
            let ok = jpg.is_some();
            if let Some(j) = jpg {
                out.clear();
                out.extend_from_slice(j);
            }
            ioctl(&self.fd, QBUF, &mut b)?;
            if ok {
                return Ok(true);
            }
            self.skip += 1;
        }
    }
}

/// The frame of buffer `d`, `dt` s and `n` frames (the driver's count) after the previous buffer: its last complete
/// JPEG of size `wh` (a buffer can hold two frames, seen in captures, or a partial one), none at bus speed.
fn pick(d: &[u8], dt: f64, n: u32, wh: (u32, u32)) -> Option<&[u8]> {
    match dt / n.max(1) as f64 >= FRESH {
        true => mjpeg::frames(d)
            .filter(|j| mjpeg::size(j) == Some(wh))
            .last(),
        false => None,
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

    /// An X5 in Webcam Mode on USB (2e1a:0005), from sysfs.
    pub(super) fn x5() -> bool {
        fs::read_dir("/sys/bus/usb/devices")
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| {
                let r = |f| fs::read_to_string(e.path().join(f)).unwrap_or_default();
                r("idVendor").trim() == "2e1a" && r("idProduct").trim() == "0005"
            })
    }

    /// Only QUERYCAP runs here: an X5 in Webcam Mode on USB is found, else nothing.
    #[test]
    fn find() {
        let r = Uvc::find();
        match &r {
            Ok(p) => println!("X5 webcam at {}", p.display()),
            Err(e) => {
                assert_eq!(e.kind(), io::ErrorKind::NotFound);
                println!("no X5 webcam: {e}");
            }
        }
        assert_eq!(r.is_ok(), x5());
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

    /// A minimal JPEG of size `w`×`h` as `mjpeg::frames` / `size` read it, `n` its scan.
    fn jpg(w: u16, h: u16, n: u8) -> Vec<u8> {
        let ([w0, w1], [h0, h1]) = (w.to_be_bytes(), h.to_be_bytes());
        let sof = [0xff, 0xc0, 0, 11, 8, h0, h1, w0, w1, 1, 1, 0x11, 0];
        let sos = [0xff, 0xda, 0, 8, 1, 1, 0, 0, 0x3f, 0, n, n, 0xff, 0xd9];
        [&[0xff, 0xd8][..], &sof, &sos].concat()
    }

    /// A switch to 2880 × 1440 after 1920 × 1080 streamed and 3 s without a reader, as an X5 sent it (12–19 copies in
    /// 6 of 10 such switches, 17–18 in 2 of 2 without the wait): the old mode's backlog at bus speed, 1.7 s of
    /// nothing, the new mode's first frame 19 times over at bus speed, a new frame 7.9 ms later, then the camera's
    /// pace. Only the first copy and the frames at that pace count: the new frame, under [`FRESH`] after the last copy,
    /// is dropped with the copies.
    #[test]
    fn burst() {
        let (old, copy) = (jpg(1920, 1080, 0), jpg(2880, 1440, 1));
        let new: Vec<_> = (2..6).map(|n| jpg(2880, 1440, n)).collect();
        let mut s = vec![(f64::NEG_INFINITY, &old)];
        s.extend([(0.4, &old); 4]);
        s.push((1733.5, &copy));
        s.extend([(0.7, &copy); 18]);
        s.extend([7.9, 38.8, 26.1, 31.3].into_iter().zip(&new));
        let got: Vec<_> = s
            .iter()
            .filter_map(|&(g, d)| pick(d, g * 1e-3, 1, (2880, 1440)))
            .map(|j| j[25])
            .collect();
        assert_eq!(got, [1, 3, 4, 5]);
    }

    /// A same-size stream after the camera sat idle, as an X5 sent it to a reader that stalled 15 ms in the backlog and
    /// then paused after each frame it took (a decode): the backlog at bus speed (0.44 ms a frame), the stall's buffer
    /// 14.6 ms but 33 frames on, 3 frames at 1.6 ms, each pause's buffer 12–15 ms but 28–33 frames on, the 3 buffers
    /// queued meanwhile, then the camera's pace. None of the backlog counts.
    #[test]
    fn stall() {
        let (old, new) = (jpg(2880, 1440, 0), jpg(2880, 1440, 1));
        let mut s = vec![(f64::NEG_INFINITY, 1, &old)];
        s.extend([(0.44, 1, &old); 30]);
        s.extend([(14.6, 33, &old), (1.6, 3, &old)]);
        for (g, n) in [(12.4, 28), (14.5, 33), (13.1, 30)] {
            s.push((g, n, &old));
            s.extend([(0.44, 1, &old); 3]);
        }
        s.extend([(21.0, 1, &new), (33.4, 1, &new)]);
        let got: Vec<_> = s
            .iter()
            .filter_map(|&(g, n, d)| pick(d, g * 1e-3, n, (2880, 1440)))
            .map(|j| j[25])
            .collect();
        assert_eq!(got, [1, 1]);
    }

    /// The X5 in Webcam Mode (skipped without it; it shows whoever sits there: nothing of it is written anywhere):
    /// each size, switched each way and restarted, streams after its stale backlog only fresh frames at the camera's
    /// 30 fps, and every one decodes, also after the reader stalled in a same-size backlog. `X5_UVC_SECS` of each
    /// stream (default 5).
    #[test]
    fn real() {
        if !x5() {
            eprintln!("no X5 in Webcam Mode (2e1a:0005): skipping");
            return;
        }
        let p = Uvc::find().unwrap();
        let secs = std::env::var("X5_UVC_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5.0);
        let (mut m, mut j, mut px) = (mjpeg::Mjpeg::default(), vec![], vec![]);
        // 3 s without a reader before a switch: its backlog holds the old mode's frames, and the new mode's first frame
        // may come ~20 times over. 8 s at the same size: a backlog of ~130 frames, where the reader stalls 15 ms after
        // 30 (the driver drops what comes meanwhile), then takes frames with the decode's pause after each
        for (w, h, idle, stall) in [
            (2880, 1440, 0, false),
            (1920, 1080, 0, false),
            (2880, 1440, 3, false),
            (2880, 1440, 0, false),
            (2880, 1440, 8, true),
            (1920, 1080, 0, false),
            (1920, 1080, 8, true),
        ] {
            std::thread::sleep(Duration::from_secs(idle));
            let t0 = now();
            let mut u = Uvc::open(&p, w, h).unwrap();
            while stall && u.skip < 30 {
                let f = u.next(&mut j, Duration::from_millis(1)).unwrap();
                assert!(
                    !f && now() - t0 < 5.0,
                    "{w}×{h}: a backlog under 30 buffers"
                );
            }
            if stall {
                std::thread::sleep(Duration::from_millis(15));
            }
            let (mut ts, mut kb, mut bad) = (vec![], 0, 0);
            let mut first = (0.0, 0);
            while ts.len() < 2 || now() - t0 - first.0 < secs {
                assert!(
                    u.next(&mut j, Duration::from_secs(5)).unwrap(),
                    "{w}×{h}: no frame"
                );
                if ts.is_empty() {
                    first = (now() - t0, u.skip);
                }
                ts.push(u.prev);
                kb += j.len() / 1000;
                match m.rgba(&j, &mut px) {
                    Ok(s) => assert_eq!(s, (w as usize, h as usize)),
                    Err(e) => (bad, _) = (bad + 1, eprintln!("{w}×{h} #{}: {e}", ts.len())),
                }
            }
            let mut d: Vec<f64> = ts.windows(2).map(|t| (t[1] - t[0]) * 1e3).collect();
            let n = d.len() as f64;
            let fps = n / (ts[ts.len() - 1] - ts[0]);
            let mean = d.iter().sum::<f64>() / n;
            let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
            let late = d.iter().filter(|&&x| x > 50.0).count();
            d.sort_by(f64::total_cmp);
            let q = |f: f64| d[((n - 1.0) * f).round() as usize];
            println!(
                "{w}×{h}: first fresh frame {:.2} s after open, {} buffers dropped before it, {} after; {} frames, \
                 {fps:.2} fps; interval ms min {:.1} p50 {:.1} p99 {:.1} max {:.1} sd {sd:.1}, {late} over 50; \
                 {} KB a frame; {bad} decode errors",
                first.0,
                first.1,
                u.skip - first.1,
                ts.len(),
                q(0.0),
                q(0.5),
                q(0.99),
                q(1.0),
                kb / ts.len()
            );
            // the camera's pace (28.8–29.7 fps over 30 s, at most 30.05 over 5 s measured): no stale frames at bus
            // speed, few of its own dropped; none after a same-size backlog (each stale frame taken is followed by
            // more of the backlog)
            assert!(
                bad == 0 && fps > 27.0 && fps < 30.0 + 1.0 / secs && first.0 < 5.0,
                "{fps} fps"
            );
            assert!(!stall || u.skip == first.1, "{w}×{h}: stale frames taken");
        }
    }
}
