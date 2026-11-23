//! USB mass storage: the X5 in its U-disk mode, or its card in a reader, is a block device that udisks2 mounts (under
//! `/run/media/$USER/`). Mounted volumes come from `/proc/self/mountinfo`: one is the camera when its block device's
//! USB ancestor calls itself `Insta360 …` (the app's test is the product string, PROTOCOL.md §8.1), and one under a
//! removable-media mount point counts as a card when it holds a `DCIM` tree. Its files are listed as the app lists
//! them (§8.7) and grouped like the camera's album, with each main file's trailer metadata and thumbnail and the
//! favourites of `DCIM/.editinfo_list.list`. Imports go through the download queue (`dl`) as [`copy`]: resumable from
//! `<dst>.part`. On macOS the mount table is the kernel's (`getfsstat` without waiting, so a share whose server went
//! away holds nothing up), the volumes are block devices under `/Volumes`, and the USB ancestor is looked up in the
//! IOKit registry, up from the disk's IOMedia. Elsewhere no card is found (the test hook's directories still count),
//! and a card's files list and import as on Linux.

use crate::album::{self, Item};
use link::{Error, Prog, pb};
use media::insv::{self, Trailer};
use prost::Message;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::Ordering::Relaxed,
};

/// A mounted volume with a `DCIM` tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vol {
    pub root: PathBuf,
    /// the block device, e.g. `/dev/sdb1`
    pub dev: String,
    /// the USB product string when the camera itself is the disk
    pub cam: Option<String>,
}

impl Vol {
    pub fn name(&self) -> String {
        let l = self
            .root
            .file_name()
            .map_or("".into(), |n| n.to_string_lossy());
        match &self.cam {
            // udisks2 and macOS name the mount point after the volume label, the X5 its card after its product
            Some(c) if *c == l => c.clone(),
            Some(c) => format!("{c} ({l})"),
            None => format!("Card {l}"),
        }
    }
}

/// Where to look: the mount table, sysfs, where removable media get mounted, and directories that count as mounted
/// camera disks (test hook).
#[derive(Clone, Debug)]
pub struct Watch {
    // the mount table and sysfs are Linux's
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub info: PathBuf,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub sys: PathBuf,
    pub media: Vec<PathBuf>,
    pub fake: Vec<PathBuf>,
}

impl Watch {
    pub fn new(fake: Vec<PathBuf>) -> Watch {
        Watch {
            info: "/proc/self/mountinfo".into(),
            sys: "/sys".into(),
            // udisks2's places (`/media` on some distributions) and the usual manual one; macOS's one
            media: match cfg!(target_os = "macos") {
                true => vec!["/Volumes".into()],
                false => ["/run/media", "/media", "/mnt"].map(PathBuf::from).into(),
            },
            fake,
        }
    }

    /// The test hook's: the directories `fake` only, nothing of the mount table or of what is mounted.
    pub fn hook(fake: Vec<PathBuf>) -> Watch {
        Watch {
            info: PathBuf::new(),
            media: vec![],
            ..Watch::new(fake)
        }
    }

    pub fn vols(&self) -> Vec<Vol> {
        #[cfg(target_os = "macos")]
        let mut v = volumes(&table(), &self.media, usb);
        #[cfg(target_os = "linux")]
        let mut v = vols(
            &fs::read_to_string(&self.info).unwrap_or_default(),
            &self.sys,
            &self.media,
        );
        // no mount table looked at here
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let mut v = vec![];
        v.extend(
            self.fake
                .iter()
                .filter(|r| r.join("DCIM").is_dir())
                .map(|r| Vol {
                    root: r.clone(),
                    dev: "fake".into(),
                    cam: Some("Insta360 X5, simulated".into()),
                }),
        );
        v
    }
}

/// `\040` and the other octal escapes of the mount table.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // the mount table and sysfs are Linux's
fn unesc(s: &str) -> String {
    let b = s.as_bytes();
    let mut o = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b.get(i..i + 4) {
            Some([b'\\', d @ ..]) if d.iter().all(|c| (b'0'..=b'7').contains(c)) => {
                o.push(d.iter().fold(0u8, |a, c| a.wrapping_mul(8) + c - b'0'));
                i += 4
            }
            _ => {
                o.push(b[i]);
                i += 1
            }
        }
    }
    String::from_utf8_lossy(&o).into_owned()
}

/// The USB product string of the device a block device sits on, if it is an Insta360 camera.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // the mount table and sysfs are Linux's
fn cam(sys: &Path, dev: &str) -> Option<String> {
    let n = Path::new(dev).file_name()?;
    let sys = fs::canonicalize(sys).ok()?;
    let p = fs::canonicalize(sys.join("class/block").join(n)).ok()?;
    // the USB device is the first ancestor with a product string and a vendor id
    let d = p
        .ancestors()
        .take_while(|a| a.starts_with(&sys))
        .find(|a| a.join("idVendor").exists())?;
    let s = fs::read_to_string(d.join("product")).ok()?;
    let s = s.trim();
    s.starts_with("Insta360").then(|| s.to_owned())
}

/// Volumes of the mount table `info` with a `DCIM` directory: block devices on an Insta360 camera, and ones mounted
/// under a `media` directory.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // the mount table and sysfs are Linux's
pub fn vols(info: &str, sys: &Path, media: &[PathBuf]) -> Vec<Vol> {
    let mut v = vec![];
    for l in info.lines() {
        // id parent dev root mountpoint options [optional…] - fstype source super
        let (a, b) = l.split_once(" - ").unwrap_or((l, ""));
        let (Some(mp), Some(src)) = (a.split(' ').nth(4), b.split(' ').nth(1)) else {
            continue;
        };
        let (root, src) = (PathBuf::from(unesc(mp)), unesc(src));
        if !src.starts_with("/dev/") || v.iter().any(|x: &Vol| x.root == root) {
            continue;
        }
        let c = cam(sys, &src);
        if (c.is_some() || media.iter().any(|m| root.starts_with(m))) && root.join("DCIM").is_dir()
        {
            v.push(Vol {
                root,
                dev: src,
                cam: c,
            })
        }
    }
    v
}

/// Volumes of the mount table `t` (macOS's: source, mount point, shown by Finder) with a `DCIM` directory: block
/// devices mounted under a `media` directory; `cam` names the camera's by its source. Only those paths are looked at:
/// a network share's can block, and the hidden mounts (`nobrowse`) are the system's.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))] // macOS's mount table
fn volumes(
    t: &[(String, PathBuf, bool)],
    media: &[PathBuf],
    cam: impl Fn(&str) -> Option<String>,
) -> Vec<Vol> {
    let mut v: Vec<Vol> = vec![];
    for (src, root, shown) in t {
        if !shown
            || !src.starts_with("/dev/")
            || !media.iter().any(|m| root.starts_with(m))
            || v.iter().any(|x| x.root == *root)
        {
            continue;
        }
        if root.join("DCIM").is_dir() {
            v.push(Vol {
                root: root.clone(),
                dev: src.clone(),
                cam: cam(src),
            })
        }
    }
    v
}

/// macOS: the mount table as the kernel keeps it (`MNT_NOWAIT`: no file system is asked): source, mount point,
/// whether Finder shows it.
#[cfg(target_os = "macos")]
fn table() -> Vec<(String, PathBuf, bool)> {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    // SAFETY: no buffer: the count
    let n = unsafe { libc::getfsstat(std::ptr::null_mut(), 0, libc::MNT_NOWAIT) };
    // room for mounts made in between
    let mut v = vec![libc::statfs::default(); n.max(0) as usize + 8];
    let len = (v.len() * size_of::<libc::statfs>()) as libc::c_int;
    // SAFETY: `v` holds `len` bytes of statfs records to fill
    let n = unsafe { libc::getfsstat(v.as_mut_ptr(), len, libc::MNT_NOWAIT) };
    v.truncate(n.max(0) as usize);
    let s = |b: &[libc::c_char]| -> Vec<u8> {
        b.iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8)
            .collect()
    };
    v.iter()
        .map(|m| {
            (
                String::from_utf8_lossy(&s(&m.f_mntfromname)).into_owned(),
                PathBuf::from(OsString::from_vec(s(&m.f_mntonname))),
                m.f_flags & libc::MNT_DONTBROWSE as u32 == 0,
            )
        })
        .collect()
}

/// macOS: the USB product string of the device the BSD disk `dev` (`/dev/disk4s1`) sits on, if it is an Insta360
/// camera, from the IOKit registry (no I/O on the disk).
#[cfg(target_os = "macos")]
fn usb(dev: &str) -> Option<String> {
    use objc2_core_foundation::{CFDictionary, CFRetained, CFString};
    use objc2_io_kit::{
        IOBSDNameMatching, IOObjectRelease, IORegistryEntryCreateCFProperty,
        IORegistryEntryGetParentEntry, IOServiceGetMatchingService, io_registry_entry_t,
        kIOServicePlane,
    };
    let n = std::ffi::CString::new(dev.strip_prefix("/dev/")?).ok()?;
    let prop = |e: io_registry_entry_t, k: &str| {
        // SAFETY: a live entry, a key; the property comes retained
        unsafe { IORegistryEntryCreateCFProperty(e, Some(&CFString::from_str(k)), None, 0) }
    };
    // SAFETY: the default main port (0: `kIOMainPortDefault` needs macOS 12), a C string; a matching dictionary is a
    // CFDictionary, which the lookup consumes
    let mut e = unsafe {
        let m = IOBSDNameMatching(0, 0, n.as_ptr());
        IOServiceGetMatchingService(0, m.map(|m| CFRetained::cast_unchecked::<CFDictionary>(m)))
    };
    // the USB device is the first entry up from the IOMedia with a vendor id and a product string (newer macOS also
    // names it kUSBProductString, which nusb reads first)
    let mut s = None;
    while e != 0 {
        if prop(e, "idVendor").is_some() {
            s = ["kUSBProductString", "USB Product Name"]
                .into_iter()
                .find_map(|k| prop(e, k)?.downcast::<CFString>().ok());
        }
        let mut p = 0;
        let plane = kIOServicePlane.as_ptr().cast_mut().cast();
        // SAFETY: a live entry, the plane's name (an io_name_t to IOKit, which only reads it), a handle to fill
        let up = s.is_none() && unsafe { IORegistryEntryGetParentEntry(e, plane, &mut p) } == 0;
        IOObjectRelease(e);
        e = if up { p } else { 0 };
    }
    let s = s?.to_string();
    s.starts_with("Insta360").then_some(s)
}

/// A file the app collects from a card (`DocumentFileUtils`): its prefixes and extensions, no `__`, 4–6 parts.
fn media(n: &str) -> bool {
    let e = n.rsplit_once('.').map_or("", |e| e.1).to_ascii_lowercase();
    ["VID_", "PRO_VID_", "LRV_", "PRO_LRV_", "IMG_", "LIV_"]
        .iter()
        .any(|p| n.starts_with(p))
        && ["insv", "insp", "mp4", "jpg", "dng", "lrv", "insdata"].contains(&e.as_str())
        && !n.contains("__")
        && (4..=6).contains(&n.split('_').count())
}

/// The card's album: its media files as camera URIs (`/DCIM/…`, relative to `root`) grouped into items, with each
/// item's metadata from the trailer of its first file and the card's favourites. Empty files are left out: an
/// interrupted recording leaves a 0-byte main file beside a whole proxy, whose trailer then speaks for the item.
pub fn list(root: &Path) -> io::Result<Vec<Item>> {
    let mut uris = vec![];
    let mut st = vec![root.join("DCIM")];
    fs::read_dir(&st[0])?;
    while let Some(d) = st.pop() {
        for e in fs::read_dir(&d).into_iter().flatten().flatten() {
            let (p, n) = (e.path(), e.file_name());
            let n = n.to_string_lossy();
            if n.starts_with('.') {
                continue;
            }
            if p.is_dir() {
                st.push(p)
            } else if media(&n) && p.metadata().is_ok_and(|m| m.is_file() && m.len() > 0) {
                // a camera URI is '/'-separated, also where paths are not (Windows)
                let r: Vec<_> = p
                    .strip_prefix(root)
                    .unwrap_or(&p)
                    .iter()
                    .map(|c| c.to_string_lossy())
                    .collect();
                uris.push(format!("/{}", r.join("/")))
            }
        }
    }
    uris.sort();
    let ed = fs::read(root.join("DCIM/.editinfo_list.list"))
        .ok()
        .and_then(|b| pb::FileMarks::decode(&b[..]).ok())
        .unwrap_or_default();
    // one trailer per item: grouped first to know each item's first file, then again with their metadata
    let infos: Vec<(String, Option<pb::Metadata>)> = album::group(&uris, &[], &ed)
        .into_iter()
        .map(|it| {
            let u = &it.uris[0];
            (u.clone(), meta(&root.join(&u[1..])))
        })
        .collect();
    Ok(album::group(&uris, &infos, &ed))
}

/// Date, length and size from a file's trailer (lenient: the camera's own fields), the size from the file system.
fn meta(p: &Path) -> Option<pb::Metadata> {
    let n = fs::metadata(p).ok()?.len();
    let m = Trailer::open(p).map(|t| t.meta).unwrap_or_default();
    Some(pb::Metadata {
        created: m.created,
        ms: m.duration_ms,
        dims: (m.dimension[0] > 0).then(|| pb::metadata::Pair {
            x: m.dimension[0] as i32,
            y: m.dimension[1] as i32,
        }),
        size: n,
        ..Default::default()
    })
}

/// The trailer's thumbnail (the NV12 record the camera serves as `GET_FILE_EXTRA` 2) scaled to the album's box.
pub fn thumb(p: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let mut b = vec![];
    Trailer::open(p).ok()?.get(insv::THUMBNAIL, &mut b).ok()?;
    album::thumb(&b)
}

/// What [`copy`] compares before resuming: the end of the partial file must match the source there.
const CHECK: u64 = 1 << 16;

/// Fills `b` from `f` at `at`.
#[cfg(unix)]
fn read_at(f: &File, b: &mut [u8], at: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, b, at)
}

/// Fills `b` from `f` at `at`: `seek_read` until it is full (it moves the file's position, which [`copy`] sets
/// afterwards).
#[cfg(windows)]
fn read_at(f: &File, mut b: &mut [u8], mut at: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !b.is_empty() {
        match f.seek_read(b, at) {
            Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => (b, at) = (&mut b[n..], at + n as u64),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Copies `src` to `dst` through `<dst>.part`, resuming from it: the partial data counts if its last [`CHECK`]
/// bytes match the source (a source replaced since, or a torn write after a crash, starts over). Progress and the
/// cancel flag in `prog`; the finished file takes the source's modification time.
pub fn copy(src: &Path, dst: &Path, prog: &Prog) -> link::Result<u64> {
    let mut part = dst.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let mut i = File::open(src)?;
    let (n, mt) = (i.metadata()?.len(), i.metadata()?.modified()?);
    prog.total.store(n, Relaxed);
    let mut o = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&part)?;
    let mut at = o.metadata()?.len();
    if at > n {
        at = 0
    }
    if at > 0 {
        let k = at.min(CHECK);
        let (mut a, mut b) = (vec![0; k as usize], vec![0; k as usize]);
        read_at(&i, &mut a, at - k)?;
        read_at(&o, &mut b, at - k)?;
        if a != b {
            at = 0
        }
    }
    o.set_len(at)?;
    i.seek(SeekFrom::Start(at))?;
    o.seek(SeekFrom::Start(at))?;
    prog.done.store(at, Relaxed);
    let mut buf = vec![0; 4 << 20];
    while at < n {
        if prog.cancel.load(Relaxed) {
            o.sync_data()?;
            return Err(Error::Cancelled);
        }
        let k = i.read(&mut buf[..(n - at).min(4 << 20) as usize])?;
        if k == 0 {
            return Err(Error::Changed);
        }
        o.write_all(&buf[..k])?;
        at += k as u64;
        prog.done.store(at, Relaxed);
    }
    o.sync_all()?;
    o.set_modified(mt)?;
    drop(o);
    fs::rename(&part, dst)?;
    Ok(n)
}

#[cfg(test)]
mod tests;
