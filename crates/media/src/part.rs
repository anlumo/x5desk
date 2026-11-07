//! Output files written beside their name: `<p>.part` is made new (never over a file or through a link) and known by
//! its identity from then on; it is published without replacing anything; every cleanup goes by that identity, so a
//! file that took either name meanwhile (a download's, the GUI's, the user's) is never removed. macOS FAT and exFAT
//! number a file by its first cluster, an empty one by a placeholder: there the identity changes as the file is
//! written, and [`finish`] takes it from the open file.

use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering::Relaxed},
};

/// What [`finish`] answers when its stop flag was set (crate export's word for a cancel).
pub const CANCELLED: &str = "cancelled";

/// A file's identity. unix: `st_dev`, `st_ino` and the birth time (statx on Linux, `st_birthtime` on macOS; None when
/// the file system keeps none); Windows: the volume serial and the file index, birth None.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id {
    pub dev: u64,
    pub ino: u64,
    /// ns since 1970
    pub birth: Option<i128>,
}

/// A reserved `<p>.part`: open read + write, and who it is (macOS FAT and exFAT: until it is written).
#[derive(Debug)]
pub struct Part {
    pub f: File,
    pub id: Id,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settled {
    /// `dst` is the file, this many bytes long
    Done(u64),
    /// neither name holds the file any more (its `.part` was removed if it still did)
    Gone,
}

/// Free space: what an unprivileged writer may still use, and the volume (`Id::dev`'s numbering).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Space {
    pub free: u64,
    pub vol: u64,
}

/// `<p>.part`
pub fn of(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".part");
    s.into()
}

/// `<p>.part` new, read + write (the faststart shift reads): `O_CREAT|O_EXCL` (a link at the name is not followed:
/// EEXIST), `O_CLOEXEC`, mode 0o666 (umask applies, as `Mux::create`). From now on every writer that checks
/// ([`free`]: the GUI's, hosts', downloads') numbers past it.
pub fn create(p: &Path) -> io::Result<Part> {
    let mut o = File::options();
    o.read(true).write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o666);
    let f = o.open(of(p))?;
    let id = sys::id_of(&f)?;
    Ok(Part { f, id })
}

/// The identity of the file `f` is open on, as it is now.
pub fn id_of(f: &File) -> io::Result<Id> {
    sys::id_of(f)
}

/// `path` itself (a link not followed) is that file: dev + ino (+ birth when both are known); when dev differs (a USB
/// disk, a btrfs subvolume after a reboot), ino + birth, both known.
pub fn is(path: &Path, id: &Id) -> bool {
    sys::lid(path).is_some_and(|(x, _)| same(&x, id))
}

fn same(a: &Id, b: &Id) -> bool {
    a.ino == b.ino
        && match a.birth.zip(b.birth) {
            Some((x, y)) => x == y,
            None => a.dev == b.dev,
        }
}

/// The length of `path` (a link not followed) when it is that file.
pub fn len_if(path: &Path, id: &Id) -> Option<u64> {
    sys::lid(path).filter(|(x, _)| same(x, id)).map(|(_, n)| n)
}

/// Unlinks `path` only if it is that file (a µs between the check and the unlink, accepted).
pub fn remove_if(path: &Path, id: &Id) -> bool {
    is(path, id) && fs::remove_file(path).is_ok()
}

/// Renames `part` to `p` unless something is at `p`, a dangling link included (`AlreadyExists`: "appeared while it
/// was rendered; nothing replaced"). Linux `renameat2(RENAME_NOREPLACE)`, macOS `renamex_np(RENAME_EXCL)`, Windows
/// `MoveFileExW` without replacing; where the file system has no such rename, a hard link and an unlink, and without
/// hard links (FAT, exFAT) a look and a rename.
pub fn publish(part: &Path, p: &Path) -> io::Result<()> {
    sys::publish(part, p).map_err(|e| match e.kind() {
        io::ErrorKind::AlreadyExists => {
            io::Error::new(e.kind(), "appeared while it was rendered; nothing replaced")
        }
        _ => e,
    })
}

/// The one cleanup of a job that may or may not have published: (1) `dst` is `id` → Done, and `part` removed if it
/// is `id` too (a link fallback's second name); (2) else `part` removed if it is `id`, then `dst` looked at again (a
/// publish that landed meanwhile) → Done; (3) else Gone. `id` is `create`'s: not on FAT or exFAT ([`fat`]).
pub fn settle(part: &Path, dst: &Path, id: &Id) -> Settled {
    let done = || {
        sys::lid(dst)
            .filter(|(x, _)| same(x, id))
            .map(|(_, n)| Settled::Done(n))
    };
    if let Some(d) = done() {
        remove_if(part, id);
        return d;
    }
    remove_if(part, id);
    done().unwrap_or(Settled::Gone)
}

/// A job's end: the final stop check ([`CANCELLED`]), whether `part` is still the file `f` (the job's, open), the
/// publish, and on every error the cleanup by identity: `f`'s now, which the writes may have changed (macOS FAT and
/// exFAT). A `part` that is someone else's by now is neither published nor removed.
pub fn finish<T>(
    part: &Path,
    dst: &Path,
    f: &File,
    stop: &AtomicBool,
    r: Result<T, String>,
) -> Result<T, String> {
    let id = sys::id_of(f);
    let r = r.and_then(|v| match (stop.load(Relaxed), &id) {
        (true, _) => Err(CANCELLED.into()),
        (_, Err(e)) => Err(format!("{}: {e}", part.display())),
        _ => Ok(v),
    });
    let id = id.ok();
    let ours = id.is_some_and(|id| is(part, &id));
    let rm = || id.is_some_and(|id| remove_if(part, &id));
    match r {
        Ok(_) if !ours => Err(format!("{}: removed while it was written", part.display())),
        Ok(v) => publish(part, dst).map(|_| v).map_err(|e| {
            rm();
            format!("{}: {e}", dst.display())
        }),
        Err(e) => {
            rm();
            Err(e)
        }
    }
}

/// Nothing is at `p` or `<p>.part`, not even a dangling link (lstat; an unreadable name counts as free, as
/// `Path::exists` has it, so a numbering loop always ends).
pub fn free(p: &Path) -> bool {
    fs::symlink_metadata(p).is_err() && fs::symlink_metadata(of(p)).is_err()
}

/// The free space of `dir`'s volume: Linux statvfs (`f_bavail × f_frsize`), macOS statfs (64-bit counts), Windows
/// `GetDiskFreeSpaceExW`; the volume as [`Id::dev`] numbers it.
pub fn avail(dir: &Path) -> io::Result<Space> {
    sys::avail(dir)
}

/// `dir` is on FAT or exFAT (Linux statfs `f_type`, macOS `f_fstypename`, Windows the volume's file system), whose
/// inode numbers may change across mounts (macOS: as a file is written): identities there are unsafe for cleanups
/// across processes.
pub fn fat(dir: &Path) -> bool {
    sys::fat(dir)
}

#[cfg(unix)]
mod sys {
    use super::{Id, Space};
    use std::{
        ffi::CString,
        fs::{self, File, Metadata},
        io,
        os::unix::{ffi::OsStrExt, fs::MetadataExt},
        path::Path,
        sync::Mutex,
        time::{SystemTime, UNIX_EPOCH},
    };

    /// A birth time in ns; zero (a file system that keeps none but answers anyway) is none.
    fn ns(t: SystemTime) -> Option<i128> {
        let n = match t.duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_nanos() as i128,
            Err(e) => -(e.duration().as_nanos() as i128),
        };
        (n != 0).then_some(n)
    }

    fn ident(m: &Metadata) -> Id {
        Id {
            dev: m.dev(),
            ino: m.ino(),
            birth: m.created().ok().and_then(ns),
        }
    }

    pub(super) fn id_of(f: &File) -> io::Result<Id> {
        f.metadata().map(|m| ident(&m))
    }

    /// lstat: the identity and the length
    pub(super) fn lid(p: &Path) -> Option<(Id, u64)> {
        fs::symlink_metadata(p).ok().map(|m| (ident(&m), m.len()))
    }

    fn c(p: &Path) -> io::Result<CString> {
        CString::new(p.as_os_str().as_bytes()).map_err(|_| io::ErrorKind::InvalidInput.into())
    }

    fn ok(r: libc::c_long) -> io::Result<()> {
        match r {
            0 => Ok(()),
            _ => Err(io::Error::last_os_error()),
        }
    }

    #[cfg(target_os = "linux")]
    fn excl(a: &Path, b: &Path) -> io::Result<()> {
        let (a, b) = (c(a)?, c(b)?);
        // the raw call: glibc's wrapper needs 2.28
        ok(unsafe {
            libc::syscall(
                libc::SYS_renameat2,
                libc::AT_FDCWD,
                a.as_ptr(),
                libc::AT_FDCWD,
                b.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        })
    }

    /// The flag unknown to this file system (some FUSE, NFS)
    #[cfg(target_os = "linux")]
    fn unsupported(e: &io::Error) -> bool {
        matches!(e.raw_os_error(), Some(libc::EINVAL | libc::ENOSYS))
    }

    #[cfg(target_os = "macos")]
    fn excl(a: &Path, b: &Path) -> io::Result<()> {
        let (a, b) = (c(a)?, c(b)?);
        ok(unsafe { libc::renamex_np(a.as_ptr(), b.as_ptr(), libc::RENAME_EXCL) }.into())
    }

    /// Not APFS or HFS+ (exFAT, SMB: which of the two they answer is unverified)
    #[cfg(target_os = "macos")]
    fn unsupported(e: &io::Error) -> bool {
        matches!(e.raw_os_error(), Some(libc::ENOTSUP | libc::EPERM))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn excl(_: &Path, _: &Path) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn unsupported(e: &io::Error) -> bool {
        e.kind() == io::ErrorKind::Unsupported
    }

    pub(super) fn publish(part: &Path, p: &Path) -> io::Result<()> {
        match excl(part, p) {
            Err(e) if unsupported(&e) => linked(part, p),
            r => r,
        }
    }

    /// No no-replace rename here: a second name (`link` never replaces), then the first one dropped. A crash between
    /// the two leaves both names, which `settle` resolves. A link refused for any other reason than a name there (no
    /// hard links: FAT and exFAT's EPERM or ENOTSUP, FUSE's ENOSYS on older kernels) is `looked`: the rename may still
    /// work, and where it fails too the publish fails as the link would have.
    pub(super) fn linked(part: &Path, p: &Path) -> io::Result<()> {
        match fs::hard_link(part, p) {
            Ok(()) => {
                let _ = fs::remove_file(part);
                Ok(())
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Err(e),
            Err(_) => looked(part, p),
        }
    }

    /// Neither: a look, then a plain rename, with the window the other two close (logged once per volume).
    pub(super) fn looked(part: &Path, p: &Path) -> io::Result<()> {
        static SEEN: Mutex<Vec<u64>> = Mutex::new(vec![]);
        let vol = p
            .parent()
            .and_then(|d| fs::metadata(d).ok())
            .map_or(0, |m| m.dev());
        let mut s = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !s.contains(&vol) {
            s.push(vol);
            eprintln!(
                "{}: no no-replace rename or hard links on this volume: published by a look and a rename",
                p.display()
            );
        }
        drop(s);
        match fs::symlink_metadata(p) {
            Ok(_) => Err(io::ErrorKind::AlreadyExists.into()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => fs::rename(part, p),
            Err(e) => Err(e),
        }
    }

    #[cfg(target_os = "linux")]
    pub(super) fn avail(dir: &Path) -> io::Result<Space> {
        let d = c(dir)?;
        let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
        ok(unsafe { libc::statvfs(d.as_ptr(), &mut s) }.into())?;
        Ok(Space {
            free: s.f_bavail as u64 * s.f_frsize as u64,
            vol: fs::metadata(dir)?.dev(),
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn avail(dir: &Path) -> io::Result<Space> {
        let s = statfs(dir)?;
        Ok(Space {
            free: s.f_bavail * u64::from(s.f_bsize),
            vol: fs::metadata(dir)?.dev(),
        })
    }

    fn statfs(dir: &Path) -> io::Result<libc::statfs> {
        let d = c(dir)?;
        let mut s: libc::statfs = unsafe { std::mem::zeroed() };
        ok(unsafe { libc::statfs(d.as_ptr(), &mut s) }.into())?;
        Ok(s)
    }

    #[cfg(target_os = "linux")]
    pub(super) fn fat(dir: &Path) -> bool {
        // MSDOS_SUPER_MAGIC (vfat too), EXFAT_SUPER_MAGIC (the kernel's; a FUSE exFAT reads as fuseblk)
        statfs(dir).is_ok_and(|s| matches!(s.f_type as u32, 0x4d44 | 0x2011_bab0))
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn fat(dir: &Path) -> bool {
        statfs(dir).is_ok_and(|s| {
            let n = s.f_fstypename.map(|c| c as u8);
            std::ffi::CStr::from_bytes_until_nul(&n)
                .is_ok_and(|n| matches!(n.to_bytes(), b"msdos" | b"exfat"))
        })
    }
}

#[cfg(windows)]
mod sys {
    use super::{Id, Space};
    use std::{
        fs::File,
        io,
        os::windows::{ffi::OsStrExt, fs::OpenOptionsExt, io::AsRawHandle},
        path::Path,
        ptr::null_mut,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        GetDiskFreeSpaceExW, GetFileInformationByHandle, GetVolumeInformationByHandleW,
        MoveFileExW,
    };

    fn info(f: &File) -> io::Result<BY_HANDLE_FILE_INFORMATION> {
        let mut i: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        match unsafe { GetFileInformationByHandle(f.as_raw_handle(), &mut i) } {
            0 => Err(io::Error::last_os_error()),
            _ => Ok(i),
        }
    }

    fn ident(i: &BY_HANDLE_FILE_INFORMATION) -> Id {
        Id {
            dev: u64::from(i.dwVolumeSerialNumber),
            ino: u64::from(i.nFileIndexHigh) << 32 | u64::from(i.nFileIndexLow),
            birth: None,
        }
    }

    /// `p` itself, a link not followed; a directory too; no access asked (attributes only)
    fn open(p: &Path) -> io::Result<File> {
        File::options()
            .access_mode(0)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
            .open(p)
    }

    pub(super) fn id_of(f: &File) -> io::Result<Id> {
        info(f).map(|i| ident(&i))
    }

    pub(super) fn lid(p: &Path) -> Option<(Id, u64)> {
        let i = info(&open(p).ok()?).ok()?;
        Some((
            ident(&i),
            u64::from(i.nFileSizeHigh) << 32 | u64::from(i.nFileSizeLow),
        ))
    }

    fn wide(p: &Path) -> Vec<u16> {
        p.as_os_str().encode_wide().chain([0]).collect()
    }

    /// No `MOVEFILE_REPLACE_EXISTING`: an existing `p` fails it (`ERROR_ALREADY_EXISTS`) on every Windows file system,
    /// FAT and SMB included. std opens files with `FILE_SHARE_DELETE`, so an open `.part` moves.
    pub(super) fn publish(part: &Path, p: &Path) -> io::Result<()> {
        match unsafe { MoveFileExW(wide(part).as_ptr(), wide(p).as_ptr(), 0) } {
            0 => Err(io::Error::last_os_error()),
            _ => Ok(()),
        }
    }

    pub(super) fn avail(dir: &Path) -> io::Result<Space> {
        let mut free = 0u64;
        if unsafe { GetDiskFreeSpaceExW(wide(dir).as_ptr(), &mut free, null_mut(), null_mut()) }
            == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(Space {
            free,
            vol: id_of(&open(dir)?)?.dev,
        })
    }

    pub(super) fn fat(dir: &Path) -> bool {
        let Ok(f) = open(dir) else { return false };
        let mut n = [0u16; 32];
        let ok = unsafe {
            GetVolumeInformationByHandleW(
                f.as_raw_handle(),
                null_mut(),
                0,
                null_mut(),
                null_mut(),
                null_mut(),
                n.as_mut_ptr(),
                n.len() as u32,
            )
        } != 0;
        let n = String::from_utf16_lossy(&n[..n.iter().position(|&c| c == 0).unwrap_or(n.len())]);
        ok && (n.to_ascii_uppercase().starts_with("FAT") || n.eq_ignore_ascii_case("exFAT"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

    struct Tmp(PathBuf);

    impl Tmp {
        fn new(n: &str) -> Tmp {
            Tmp::under(&std::env::temp_dir(), n)
        }

        fn under(b: &Path, n: &str) -> Tmp {
            let d = b.join(format!("x5desk-part-{n}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&d);
            fs::create_dir_all(&d).unwrap();
            Tmp(d)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn there(p: &Path) -> bool {
        fs::symlink_metadata(p).is_ok()
    }

    /// `create` makes `<p>.part` read + write and refuses any name already there, a dangling link without writing
    /// through it; its mode is what std gives a new file (0o666 & ~umask).
    #[test]
    fn create_new_only() {
        let t = Tmp::new("create");
        let p = t.0.join("a.mp4");
        let Part { mut f, id } = create(&p).unwrap();
        assert!(is(&of(&p), &id) && !there(&p));
        f.write_all(b"faststart").unwrap();
        f.seek(SeekFrom::Start(4)).unwrap();
        let mut s = String::new();
        f.read_to_string(&mut s).unwrap();
        assert_eq!(s, "start");
        assert_eq!(create(&p).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
            fs::write(t.0.join("ref"), b"").unwrap();
            assert_eq!(mode(&of(&p)), mode(&t.0.join("ref")));
            let q = t.0.join("b.mp4");
            std::os::unix::fs::symlink(t.0.join("elsewhere"), of(&q)).unwrap();
            assert_eq!(create(&q).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
            assert!(!there(&t.0.join("elsewhere")), "written through the link");
        }
    }

    /// `is` and `remove_if` know the file, not the name; with another dev (a disk after a reboot) the inode and the
    /// birth time decide, when the file system keeps one.
    #[test]
    fn identity() {
        let t = Tmp::new("is");
        let (p, held) = (t.0.join("a.mp4"), t.0.join("held"));
        let a = of(&p);
        let Part { id, .. } = create(&p).unwrap();
        println!("{id:?}");
        // the inode stays taken under another name, so the next file can't reuse it
        fs::rename(&a, &held).unwrap();
        fs::write(&a, b"someone else's").unwrap();
        assert!(!is(&a, &id) && !remove_if(&a, &id) && there(&a));
        assert!(is(&held, &id));
        let moved = Id {
            dev: id.dev ^ 1,
            ..id
        };
        assert_eq!(is(&held, &moved), id.birth.is_some());
        let reborn = Id {
            birth: id.birth.map(|b| b + 1),
            ..id
        };
        assert_eq!(is(&held, &reborn), id.birth.is_none());
        assert!(!is(
            &held,
            &Id {
                ino: id.ino ^ 1,
                ..id
            }
        ));
        assert!(remove_if(&held, &id) && !there(&held));
        assert!(!is(&held, &id) && !remove_if(&held, &id));
    }

    /// `publish` never replaces: a file or (unix) a dangling link at `p` stays as it was, and so does the `.part`.
    #[test]
    fn publish_never_replaces() {
        let t = Tmp::new("publish");
        let p = t.0.join("a.mp4");
        let Part { mut f, id } = create(&p).unwrap();
        f.write_all(b"new").unwrap();
        fs::write(&p, b"old").unwrap();
        let e = publish(&of(&p), &p).unwrap_err();
        assert!(
            e.kind() == io::ErrorKind::AlreadyExists && e.to_string().contains("appeared"),
            "{e}"
        );
        assert_eq!(fs::read(&p).unwrap(), b"old");
        assert!(is(&of(&p), &id));
        fs::remove_file(&p).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(t.0.join("elsewhere"), &p).unwrap();
            assert_eq!(
                publish(&of(&p), &p).unwrap_err().kind(),
                io::ErrorKind::AlreadyExists
            );
            assert!(
                fs::symlink_metadata(&p).unwrap().is_symlink() && !there(&t.0.join("elsewhere"))
            );
            fs::remove_file(&p).unwrap();
        }
        publish(&of(&p), &p).unwrap();
        assert!(is(&p, &id) && !there(&of(&p)));
        assert_eq!(fs::read(&p).unwrap(), b"new");
    }

    /// The fallbacks, run here on a file system that has the no-replace rename: they refuse an existing name too, and
    /// publish the same inode.
    #[cfg(unix)]
    #[test]
    fn publish_fallbacks() {
        let t = Tmp::new("fallbacks");
        for (i, f) in [sys::linked, sys::looked].into_iter().enumerate() {
            let p = t.0.join(format!("{i}.mp4"));
            let Part { id, .. } = create(&p).unwrap();
            fs::write(&p, b"old").unwrap();
            assert_eq!(
                f(&of(&p), &p).unwrap_err().kind(),
                io::ErrorKind::AlreadyExists
            );
            assert!(is(&of(&p), &id) && fs::read(&p).unwrap() == b"old");
            fs::remove_file(&p).unwrap();
            f(&of(&p), &p).unwrap();
            assert!(is(&p, &id) && !there(&of(&p)));
        }
    }

    /// `settle`: published → Done; not published → Gone, the `.part` removed only when it is the file; both names
    /// (a crash inside the link fallback) → Done, the `.part` name gone.
    #[test]
    fn settle_cases() {
        let t = Tmp::new("settle");
        let p = t.0.join("gone.mp4");
        let Part { id, .. } = create(&p).unwrap();
        assert_eq!(settle(&of(&p), &p, &id), Settled::Gone);
        assert!(!there(&of(&p)));

        let p = t.0.join("other.mp4");
        let Part { id, .. } = create(&p).unwrap();
        fs::rename(of(&p), t.0.join("held")).unwrap();
        fs::write(of(&p), b"a download's").unwrap();
        assert_eq!(settle(&of(&p), &p, &id), Settled::Gone);
        assert_eq!(fs::read(of(&p)).unwrap(), b"a download's");
        fs::write(&p, b"the user's").unwrap();
        assert_eq!(settle(&of(&p), &p, &id), Settled::Gone);
        assert!(there(&p) && there(&of(&p)));

        let p = t.0.join("done.mp4");
        let Part { mut f, id } = create(&p).unwrap();
        f.write_all(b"12345").unwrap();
        publish(&of(&p), &p).unwrap();
        assert_eq!(settle(&of(&p), &p, &id), Settled::Done(5));
        assert!(there(&p));

        let p = t.0.join("both.mp4");
        let Part { mut f, id } = create(&p).unwrap();
        f.write_all(b"123").unwrap();
        drop(f);
        fs::hard_link(of(&p), &p).unwrap();
        assert_eq!(settle(&of(&p), &p, &id), Settled::Done(3));
        assert!(is(&p, &id) && !there(&of(&p)));
    }

    /// `finish`: Ok publishes; a stop or an error leaves nothing, also before the first write; a `.part` that is
    /// someone else's by now is neither published nor removed; a file at `dst` stays and the `.part` goes. Each
    /// `.part` is written as `Mux` writes it (emptied, then data): on macOS FAT and exFAT (`X5_FAT_DIR`, a folder on
    /// one, as well as the temp dir) its identity is no longer the one `create` saw.
    #[test]
    fn finish_cases() {
        let fd = std::env::var_os("X5_FAT_DIR").map(|d| Tmp::under(Path::new(&d), "finish"));
        if fd.is_none() {
            eprintln!("X5_FAT_DIR unset: the temp dir only");
        }
        let t = Tmp::new("finish");
        let (go, stop) = (AtomicBool::new(false), AtomicBool::new(true));
        let write = |mut f: &File, b: &[u8]| {
            f.set_len(0).unwrap();
            f.write_all(b).unwrap();
        };
        for d in [Some(&t.0), fd.as_ref().map(|t| &t.0)]
            .into_iter()
            .flatten()
        {
            println!("{}: FAT {}", d.display(), fat(d));
            let p = d.join("ok.mp4");
            let Part { f, .. } = create(&p).unwrap();
            write(&f, b"12345");
            assert_eq!(finish(&of(&p), &p, &f, &go, Ok(7)), Ok(7));
            assert!(is(&p, &id_of(&f).unwrap()) && !there(&of(&p)));
            assert_eq!(fs::read(&p).unwrap(), b"12345");

            let p = d.join("stop.mp4");
            let Part { f, .. } = create(&p).unwrap();
            write(&f, b"12345");
            assert_eq!(
                finish(&of(&p), &p, &f, &stop, Ok(())),
                Err(CANCELLED.into())
            );
            assert!(!there(&p) && !there(&of(&p)));

            for (n, b) in [("err.mp4", &b"12345"[..]), ("empty.mp4", b"")] {
                let p = d.join(n);
                let Part { f, .. } = create(&p).unwrap();
                if !b.is_empty() {
                    write(&f, b);
                }
                assert_eq!(
                    finish::<()>(&of(&p), &p, &f, &go, Err("boom".into())),
                    Err("boom".into())
                );
                assert!(!there(&p) && !there(&of(&p)), "{n}");
            }

            let p = d.join("taken.mp4");
            let Part { f, .. } = create(&p).unwrap();
            write(&f, b"12345");
            fs::rename(of(&p), d.join("held")).unwrap();
            fs::write(of(&p), b"another writer's").unwrap();
            let e = finish(&of(&p), &p, &f, &go, Ok(())).unwrap_err();
            assert!(e.contains("removed while it was written"), "{e}");
            assert!(!there(&p) && fs::read(of(&p)).unwrap() == b"another writer's");

            let p = d.join("appeared.mp4");
            let Part { f, .. } = create(&p).unwrap();
            write(&f, b"12345");
            fs::write(&p, b"the user's").unwrap();
            let e = finish(&of(&p), &p, &f, &go, Ok(())).unwrap_err();
            assert!(
                e.contains("appeared while it was rendered; nothing replaced"),
                "{e}"
            );
            assert!(fs::read(&p).unwrap() == b"the user's" && !there(&of(&p)));
        }
    }

    /// `free` sees the file, the `.part`, and (unix) a dangling link at either name.
    #[test]
    fn free_names() {
        let t = Tmp::new("free");
        let p = t.0.join("a.mp4");
        assert!(free(&p));
        fs::write(of(&p), b"").unwrap();
        assert!(!free(&p), "being written");
        fs::remove_file(of(&p)).unwrap();
        fs::write(&p, b"").unwrap();
        assert!(!free(&p));
        #[cfg(unix)]
        {
            let (q, r) = (t.0.join("b.mp4"), t.0.join("c.mp4"));
            std::os::unix::fs::symlink(t.0.join("nowhere"), &q).unwrap();
            std::os::unix::fs::symlink(t.0.join("nowhere"), of(&r)).unwrap();
            assert!(!free(&q) && !free(&r));
        }
    }

    /// `avail` and `fat` on the temp dir: some space, its volume, not FAT; a missing dir is an error.
    #[test]
    fn space() {
        let t = Tmp::new("avail");
        let s = avail(&t.0).unwrap();
        println!("{s:?}");
        assert!(s.free > 0);
        let Part { id, .. } = create(&t.0.join("a")).unwrap();
        assert_eq!(s.vol, id.dev);
        assert!(!fat(&t.0));
        assert!(avail(&t.0.join("missing")).is_err());
    }
}
