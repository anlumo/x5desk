//! HTTP/1.1 GETs from the camera's server over N connections, each writing its byte range at its offset
//! (PROTOCOL.md §7.7, §8.10). The camera honours a range's start but streams to EOF and never closes a tunnel
//! connection, so every connection is cut locally at its range end. A file download keeps its data in
//! `<dst>.part` and the missing ranges in `<dst>.part.todo`; a later download of the same file resumes from them.

use crate::{Done, Error, Result};
use std::{
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

/// A connection keeps its range only while the rest is at least this long; below it, splitting costs more than
/// it gains.
const MIN: u64 = 1 << 20;
const UNKNOWN: u64 = u64::MAX;

/// Progress of a download, shared with the app. `total` is 0 until the first response header.
#[derive(Default, Debug)]
pub struct Prog {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
}

pub(crate) enum Dst {
    Mem(Vec<u8>),
    File(File, PathBuf),
}

pub(crate) struct Conn {
    pub c: u32,
    pub got: u64,
    pos: u64,
    end: u64,
    head: Vec<u8>,
    body: bool,
}

pub(crate) struct Xfer {
    uri: String,
    dst: Dst,
    etag: String,
    total: u64,
    todo: Vec<(u64, u64)>,
    pub cs: Vec<Conn>,
    n: usize,
    pub prog: Arc<Prog>,
    pub done: Option<Done>,
    pub last: Instant,
    saved: Instant,
    pub lost: u8,
}

fn ext(p: &Path, e: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(e);
    s.into()
}

/// Status, `Content-Length`, `Content-Range` (start, total) and `ETag` of a response head (without the blank line).
pub(crate) fn head(b: &[u8]) -> Option<(u16, Option<u64>, Option<(u64, u64)>, &str)> {
    let s = std::str::from_utf8(b).ok()?;
    let mut ls = s.split("\r\n");
    let st = ls.next()?.split(' ').nth(1)?.parse().ok()?;
    let (mut len, mut rng, mut etag) = (None, None, "");
    for l in ls {
        let Some((k, v)) = l.split_once(':') else {
            continue;
        };
        let v = v.trim();
        match &*k.trim().to_ascii_lowercase() {
            "content-length" => len = v.parse().ok(),
            "content-range" => {
                let (a, t) = v.strip_prefix("bytes ")?.split_once('/')?;
                rng = Some((a.split_once('-')?.0.parse().ok()?, t.parse().ok()?))
            }
            "etag" => etag = v,
            _ => {}
        }
    }
    Some((st, len, rng, etag))
}

impl Xfer {
    pub fn new(uri: String, path: Option<&Path>, n: usize, prog: Arc<Prog>) -> io::Result<Xfer> {
        let now = Instant::now();
        let mut x = Xfer {
            uri,
            dst: Dst::Mem(vec![]),
            etag: String::new(),
            total: UNKNOWN,
            todo: vec![(0, UNKNOWN)],
            cs: vec![],
            n: n.max(1),
            prog,
            done: None,
            last: now,
            saved: now,
            lost: 0,
        };
        let Some(p) = path else { return Ok(x) };
        let (part, todo) = (ext(p, ".part"), ext(p, ".part.todo"));
        // Sidecar: etag, total, then one missing range "a b" per line. Without its part file it describes nothing.
        let st = fs::read_to_string(&todo).ok();
        let st = st.filter(|_| part.exists()).and_then(|s| {
            let mut ls = s.lines();
            let (e, t): (_, u64) = (ls.next()?.to_owned(), ls.next()?.parse().ok()?);
            let r = ls.map(|l| {
                l.split_once(' ')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
            });
            Some((e, t, r.collect::<Option<Vec<(u64, u64)>>>()?))
        });
        if st.is_none() {
            // Starting over: a stale sidecar next to the fresh part file would be resumed from after a crash.
            let _ = fs::remove_file(&todo);
        }
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(st.is_none())
            .open(part)?;
        if let Some((e, t, r)) = st {
            x.prog.total.store(t, Relaxed);
            x.prog.done.store(
                t.saturating_sub(r.iter().map(|r| r.1.saturating_sub(r.0)).sum()),
                Relaxed,
            );
            (x.etag, x.total, x.todo) = (e, t, r);
            // lowest first (`fill` pops from the end): in order, no range starts past what is written
            x.todo.sort_unstable_by_key(|r| std::cmp::Reverse(r.0));
        }
        // Ranges are written out of order: without holes a write past the file's valid data length first zero-fills
        // the gap, on the link's I/O thread, so there one connection writes in order.
        if !holes(&f) {
            x.n = 1
        }
        x.dst = Dst::File(f, p.into());
        Ok(x)
    }

    /// Opens connections up to `n`: missing ranges first, then the upper half of the longest running range.
    /// While the total is unknown, one connection runs alone. `open` returns whether the link is busy after it.
    pub fn fill(
        &mut self,
        nc: &mut u32,
        host: &str,
        open: &mut dyn FnMut(u32, &[u8]) -> io::Result<bool>,
    ) -> io::Result<()> {
        while self.cs.len() < self.n && !(self.total == UNKNOWN && !self.cs.is_empty()) {
            let (a, b) = if let Some(r) = self.todo.pop() {
                r
            } else if let Some(k) = self
                .cs
                .iter_mut()
                .max_by_key(|k| k.end - k.pos)
                .filter(|k| k.end - k.pos >= 2 * MIN)
            {
                let e = k.end;
                k.end = k.pos + (e - k.pos) / 2;
                (k.end, e)
            } else {
                break;
            };
            let c = *nc;
            *nc = c.wrapping_add(1);
            let q = format!(
                "GET {} HTTP/1.1\r\nHost: {host}\r\nRange: bytes={a}-\r\nConnection: close\r\n\r\n",
                self.uri
            );
            // A failed open hands the range back, or the sidecar would lose it and a resume would leave a hole.
            let busy = open(c, q.as_bytes()).inspect_err(|_| self.todo.push((a, b)))?;
            self.cs.push(Conn {
                c,
                got: 0,
                pos: a,
                end: b,
                head: vec![],
                body: false,
            });
            if busy {
                break;
            }
        }
        Ok(())
    }

    /// Bytes for connection `i`; true when its range is complete.
    pub fn data(&mut self, i: usize, d: &[u8]) -> Result<bool> {
        let Xfer {
            cs,
            dst,
            prog,
            todo,
            ..
        } = self;
        let k = &mut cs[i];
        let h;
        let mut d = d;
        let mut save = false;
        if !k.body {
            k.head.extend_from_slice(d);
            let Some(e) = k.head.windows(4).position(|w| w == b"\r\n\r\n") else {
                return if k.head.len() > 1 << 16 {
                    Err(Error::Http("header too long".into()))
                } else {
                    Ok(false)
                };
            };
            h = std::mem::take(&mut k.head);
            d = &h[e + 4..];
            k.body = true;
            let (st, len, rng, etag) =
                head(&h[..e]).ok_or_else(|| Error::Http("bad response header".into()))?;
            let t = match (st, rng) {
                (206, Some((a, t))) if a == k.pos => t,
                (200, _) if k.pos == 0 => len.unwrap_or(UNKNOWN),
                _ => {
                    return Err(Error::Http(
                        String::from_utf8_lossy(&h[..e])
                            .lines()
                            .next()
                            .unwrap_or("")
                            .into(),
                    ));
                }
            };
            if !self.etag.is_empty() && (self.etag != etag || self.total != t) {
                return Err(Error::Changed);
            }
            self.etag = etag.into();
            if self.total == UNKNOWN && t != UNKNOWN {
                self.total = t;
                prog.total.store(t, Relaxed);
                for e in cs
                    .iter_mut()
                    .map(|k| &mut k.end)
                    .chain(todo.iter_mut().map(|r| &mut r.1))
                {
                    *e = (*e).min(t);
                }
                if let Dst::Mem(v) = dst {
                    v.resize(t as usize, 0);
                }
                save = true;
            }
        }
        let k = &mut cs[i];
        let n = d.len().min((k.end - k.pos) as usize);
        match dst {
            Dst::Mem(v) => {
                let (a, b) = (k.pos as usize, k.pos as usize + n);
                if v.len() < b {
                    v.resize(b, 0)
                }
                v[a..b].copy_from_slice(&d[..n])
            }
            Dst::File(f, _) => write_at(f, &d[..n], k.pos)?,
        }
        k.pos += n as u64;
        prog.done.fetch_add(n as u64, Relaxed);
        self.last = Instant::now();
        if save || self.saved.elapsed() >= Duration::from_secs(1) {
            self.save();
        }
        Ok(self.cs[i].pos >= self.cs[i].end)
    }

    /// Connection `i` ended before its range did: its rest goes back to the missing ranges.
    pub fn drop_conn(&mut self, i: usize) {
        let k = self.cs.remove(i);
        if k.pos < k.end {
            self.todo.push((k.pos, k.end));
            self.lost += 1;
        }
    }

    pub fn complete(&self) -> bool {
        self.cs.is_empty() && self.todo.is_empty()
    }

    /// Writes the sidecar of a file download, once the total is known.
    pub fn save(&mut self) {
        self.saved = Instant::now();
        let Dst::File(_, p) = &self.dst else { return };
        if self.total == UNKNOWN {
            return;
        }
        let mut s = format!("{}\n{}\n", self.etag, self.total);
        for (a, b) in self
            .todo
            .iter()
            .copied()
            .chain(self.cs.iter().map(|k| (k.pos, k.end)))
            .filter(|r| r.0 < r.1)
        {
            s += &format!("{a} {b}\n");
        }
        // Replaced whole: a sidecar cut short by a crash still parses, with ranges missing.
        let t = ext(p, ".part.todo.new");
        let _ = fs::write(&t, s).and_then(|_| fs::rename(&t, ext(p, ".part.todo")));
    }

    /// Final result: the body for a memory GET; for a file, the part file renamed to its name.
    pub fn finish(self) -> Result<Vec<u8>> {
        match self.dst {
            Dst::Mem(mut v) => {
                v.truncate(self.total.min(v.len() as u64) as usize);
                Ok(v)
            }
            Dst::File(f, p) => {
                drop(f);
                fs::rename(ext(&p, ".part"), &p)?;
                let _ = fs::remove_file(ext(&p, ".part.todo"));
                Ok(vec![])
            }
        }
    }

    /// Removes the partial file (it no longer matches the camera's file).
    pub fn discard(&self) {
        if let Dst::File(_, p) = &self.dst {
            let _ = fs::remove_file(ext(p, ".part"));
            let _ = fs::remove_file(ext(p, ".part.todo"));
        }
    }
}

/// `b` at `pos`, whole.
#[cfg(unix)]
fn write_at(f: &File, b: &[u8], pos: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(f, b, pos)
}

/// Whether `f` keeps ranges never written as holes, so ranges can be written in any order without the file system
/// first zero-filling up to each write: not on FAT or exFAT, and on Windows only once marked sparse
/// (FSCTL_SET_SPARSE, which NTFS and ReFS take). Unknown: yes, a FUSE mount too (exfat-fuse among them).
#[cfg(target_os = "linux")]
fn holes(f: &File) -> bool {
    use std::os::fd::AsRawFd;
    // SAFETY: fstatfs only writes the struct it is given.
    let s = unsafe {
        let mut s: libc::statfs = std::mem::zeroed();
        (libc::fstatfs(f.as_raw_fd(), &mut s) == 0).then_some(s)
    };
    // MSDOS_SUPER_MAGIC, EXFAT_SUPER_MAGIC
    !s.is_some_and(|s| matches!(s.f_type, 0x4d44 | 0x2011_bab0))
}

/// See the Linux [`holes`]; the file system by its name.
#[cfg(target_os = "macos")]
fn holes(f: &File) -> bool {
    use std::os::fd::AsRawFd;
    // SAFETY: fstatfs only writes the struct it is given.
    let s = unsafe {
        let mut s: libc::statfs = std::mem::zeroed();
        (libc::fstatfs(f.as_raw_fd(), &mut s) == 0).then_some(s)
    };
    !s.is_some_and(|s| {
        let n = s.f_fstypename.map(|c| c as u8);
        let n = std::ffi::CStr::from_bytes_until_nul(&n).map_or(&[][..], |n| n.to_bytes());
        matches!(n, b"msdos" | b"exfat")
    })
}

/// See the Linux [`holes`].
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn holes(_: &File) -> bool {
    true
}

/// See the Linux [`holes`]: marks `f` sparse, so ranges never written take no space and are never zero-filled.
#[cfg(windows)]
fn holes(f: &File) -> bool {
    use std::{ffi::c_void, os::windows::io::AsRawHandle, ptr::null_mut};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn DeviceIoControl(
            h: *mut c_void,
            code: u32,
            i: *mut c_void,
            ni: u32,
            o: *mut c_void,
            no: u32,
            ret: *mut u32,
            ov: *mut c_void,
        ) -> i32;
    }
    const FSCTL_SET_SPARSE: u32 = 0x0009_00c4;
    let mut ret = 0;
    // SAFETY: an open handle with write access, no buffers, and no OVERLAPPED: std opens files for synchronous I/O.
    unsafe {
        DeviceIoControl(
            f.as_raw_handle(),
            FSCTL_SET_SPARSE,
            null_mut(),
            0,
            null_mut(),
            0,
            &mut ret,
            null_mut(),
        ) != 0
    }
}

/// `b` at `pos`, whole: `seek_write` may write less (and moves the cursor, which nothing here reads).
#[cfg(windows)]
fn write_at(f: &File, mut b: &[u8], mut pos: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !b.is_empty() {
        match f.seek_write(b, pos) {
            Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
            Ok(n) => (b, pos) = (&b[n..], pos + n as u64),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_head() {
        let h = b"HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes 100-999/1000\r\n\
                  Content-Length: 900\r\nEtag: \"6a1b2c3d.1000\"\r\nConnection: keep-alive";
        assert_eq!(
            head(h),
            Some((206, Some(900), Some((100, 1000)), "\"6a1b2c3d.1000\""))
        );
        assert_eq!(head(b"HTTP/1.1 404 Not Found"), Some((404, None, None, "")));
    }

    /// Serves `f` like the camera: the range start is honoured, the body runs to EOF in `chunk`-sized pieces.
    fn serve(f: &[u8], q: &[u8], chunk: usize) -> Vec<Vec<u8>> {
        let q = std::str::from_utf8(q).unwrap();
        let a: usize = q
            .split("bytes=")
            .nth(1)
            .unwrap()
            .split('-')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let mut v = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{}/{}\r\nContent-Length: {}\r\nEtag: \"e.{}\"\r\n\r\n",
            f.len() - 1,
            f.len(),
            f.len() - a,
            f.len()
        )
        .into_bytes();
        v.extend(&f[a..a + 10]);
        let mut o = vec![v];
        o.extend(f[a + 10..].chunks(chunk).map(|c| c.to_vec()));
        o
    }

    /// Runs a transfer to completion against `serve`, interleaving the connections round-robin; `stop` aborts
    /// after that many chunks (simulating a lost link).
    fn run(x: &mut Xfer, f: &[u8], stop: usize) -> bool {
        let (mut nc, mut qs, mut n) = (0, vec![], 0);
        x.fill(&mut nc, "h", &mut |c, q| {
            Ok(qs.push((c, serve(f, q, 300_000), 0))).map(|_| false)
        })
        .unwrap();
        while !x.complete() {
            let mut opened = vec![];
            for (c, ch, j) in qs.iter_mut() {
                let Some(i) = x.cs.iter().position(|k| k.c == *c) else {
                    continue;
                };
                if *j == ch.len() {
                    x.drop_conn(i);
                    continue;
                }
                *j += 1;
                n += 1;
                if n == stop {
                    x.save();
                    return false;
                }
                if x.data(i, &ch[*j - 1]).unwrap() {
                    x.cs.remove(i);
                }
                x.fill(&mut nc, "h", &mut |c, q| {
                    Ok(opened.push((c, serve(f, q, 300_000), 0))).map(|_| false)
                })
                .unwrap();
            }
            qs.extend(opened);
        }
        true
    }

    /// A download keeps its connections where the file system holds holes (the temp dir's) and runs one on FAT or
    /// exFAT (a dir on one in `X5_NOHOLES`, when set).
    #[test]
    fn holes_fs() {
        let n = |d: &Path| {
            let p = d.join(format!("x5link-h{}", std::process::id()));
            let n = Xfer::new("/f".into(), Some(&p), 4, Arc::new(Prog::default()))
                .unwrap()
                .n;
            fs::remove_file(ext(&p, ".part")).unwrap();
            n
        };
        assert_eq!(n(&std::env::temp_dir()), 4);
        let Some(d) = std::env::var_os("X5_NOHOLES") else {
            return eprintln!("X5_NOHOLES not set: no FAT or exFAT dir checked");
        };
        assert_eq!(n(Path::new(&d)), 1);
    }

    /// A resume opens its missing ranges lowest first, so one connection never starts past what is written.
    #[test]
    fn resume_in_order() {
        let dir = std::env::temp_dir().join(format!("x5link-r{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        fs::write(ext(&p, ".part"), b"0123456789").unwrap();
        fs::write(
            ext(&p, ".part.todo"),
            "e\n400\n10 100\n200 300\n300 400\n100 200\n",
        )
        .unwrap();
        let mut x = Xfer::new("/f".into(), Some(&p), 1, Arc::new(Prog::default())).unwrap();
        let mut at = vec![];
        while let Some(r) = x.todo.last().copied() {
            x.fill(&mut 0, "h", &mut |_, _| Ok(false)).unwrap();
            at.push(x.cs.pop().map(|k| (k.pos, k.end)));
            assert_eq!(at.last(), Some(&Some(r)));
        }
        assert_eq!(
            at,
            [10, 100, 200, 300].map(|a| Some((a, a + 100 - a % 100)))
        );
        drop(x);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn busy_link_holds_back() {
        let mut x = Xfer::new("/f".into(), None, 4, Arc::new(Prog::default())).unwrap();
        (x.total, x.todo) = (
            30 << 20,
            vec![(0, 10 << 20), (10 << 20, 20 << 20), (20 << 20, 30 << 20)],
        );
        let mut nc = 0;
        x.fill(&mut nc, "h", &mut |_, _| Ok(true)).unwrap();
        assert_eq!((x.cs.len(), x.todo.len()), (1, 2));
        x.fill(&mut nc, "h", &mut |_, _| Ok(false)).unwrap();
        // Then the remaining ranges, and the longest one split for the fourth connection.
        assert_eq!((x.cs.len(), x.todo.len(), nc), (4, 0, 4));
    }

    #[test]
    fn split_and_resume() {
        let f: Vec<u8> = (0..9_000_003u32).map(|i| (i * 7 + i / 251) as u8).collect();
        let prog = Arc::new(Prog::default());
        let mut x = Xfer::new("/f".into(), None, 4, prog.clone()).unwrap();
        assert!(run(&mut x, &f, 0));
        assert_eq!(prog.done.load(Relaxed), f.len() as u64);
        assert!(x.finish().unwrap() == f);

        let dir = std::env::temp_dir().join(format!("x5link-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        let mut x = Xfer::new("/f".into(), Some(&p), 4, Arc::new(Prog::default())).unwrap();
        assert!(!run(&mut x, &f, 25));
        drop(x);
        let prog = Arc::new(Prog::default());
        let mut x = Xfer::new("/f".into(), Some(&p), 3, prog.clone()).unwrap();
        let before = prog.done.load(Relaxed);
        assert!(before > 0 && before < f.len() as u64);
        assert!(run(&mut x, &f, 0));
        x.finish().unwrap();
        assert!(fs::read(&p).unwrap() == f);
        assert!(!ext(&p, ".part.todo").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// An open that fails after taking its range (a refused connect on Wi-Fi), then the save `Io::fail` does:
    /// first on a split of the running range, then on a missing range of the resumed transfer. The sidecar keeps
    /// every missing byte, and the last resume is byte-identical.
    #[test]
    fn open_failure_keeps_range() {
        let f: Vec<u8> = (0..9_000_003u32)
            .map(|i| (i * 13 + i / 509) as u8)
            .collect();
        let dir = std::env::temp_dir().join(format!("x5link-o{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        // A sidecar whose part file is gone describes nothing: start over, and drop it.
        fs::write(ext(&p, ".part.todo"), "\"e\"\n100\n0 50\n").unwrap();
        let mut x = Xfer::new("/f".into(), Some(&p), 4, Arc::new(Prog::default())).unwrap();
        assert_eq!((x.total, x.todo.len()), (UNKNOWN, 1));
        assert!(!ext(&p, ".part.todo").exists());

        let refuse = |x: &mut Xfer, nc: &mut u32| {
            let mut k = 0;
            let r = x.fill(nc, "h", &mut |_, _| {
                k += 1;
                if k == 2 {
                    Err(io::ErrorKind::ConnectionRefused.into())
                } else {
                    Ok(false)
                }
            });
            assert!(r.is_err());
            x.save();
        };
        let (mut nc, mut q) = (0, vec![]);
        x.fill(&mut nc, "h", &mut |_, r| Ok(q = r.to_vec()).map(|_| false))
            .unwrap();
        // The first head makes the total known (10 body bytes come with it); the next opens split.
        x.data(0, &serve(&f, &q, 300_000)[0]).unwrap();
        refuse(&mut x, &mut nc);
        drop(x);
        let resume = || {
            let prog = Arc::new(Prog::default());
            let x = Xfer::new("/f".into(), Some(&p), 3, prog.clone()).unwrap();
            assert_eq!((prog.done.load(Relaxed), x.todo.len()), (10, 3));
            x
        };
        refuse(&mut resume(), &mut nc);
        let mut x = resume();
        assert!(run(&mut x, &f, 0));
        x.finish().unwrap();
        assert!(fs::read(&p).unwrap() == f);
        fs::remove_dir_all(&dir).unwrap();
    }
}
