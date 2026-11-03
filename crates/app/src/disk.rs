//! USB mass storage: the X5 in its U-disk mode, or its card in a reader, is a block device that udisks2 mounts (under
//! `/run/media/$USER/`). Mounted volumes come from `/proc/self/mountinfo`: one is the camera when its block device's
//! USB ancestor calls itself `Insta360 …` (the app's test is the product string, PROTOCOL.md §8.1), and one under a
//! removable-media mount point counts as a card when it holds a `DCIM` tree. Its files are listed as the app lists
//! them (§8.7) and grouped like the camera's album, with each main file's trailer metadata and thumbnail and the
//! favourites of `DCIM/.editinfo_list.list`. Imports go through the download queue (`dl`) as [`copy`]: resumable from
//! `<dst>.part`. On macOS the volumes are the ones under `/Volumes`, and the camera's is the one labelled with its
//! product string (the X5 labels its card so): there is no sysfs to ask the USB device.

use crate::album::{self, Item};
use link::{Error, Prog, pb};
use media::insv::{self, Trailer};
use prost::Message;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    os::unix::fs::FileExt,
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
            // udisks2 names the mount point after the volume label, and the X5 labels its card with its product string
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
    pub info: PathBuf,
    #[cfg_attr(target_os = "macos", allow(dead_code))] // the mount table and sysfs are Linux's
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

    pub fn vols(&self) -> Vec<Vol> {
        #[cfg(target_os = "macos")]
        let mut v = volumes(&self.media);
        #[cfg(not(target_os = "macos"))]
        let mut v = vols(
            &fs::read_to_string(&self.info).unwrap_or_default(),
            &self.sys,
            &self.media,
        );
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
#[cfg_attr(target_os = "macos", allow(dead_code))] // the mount table and sysfs are Linux's
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
#[cfg_attr(target_os = "macos", allow(dead_code))] // the mount table and sysfs are Linux's
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
#[cfg_attr(target_os = "macos", allow(dead_code))] // the mount table and sysfs are Linux's
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

/// macOS: the volumes mounted at a directory of `media` (`/Volumes`) with a `DCIM` directory; one labelled
/// `Insta360 …` is the camera.
#[cfg(target_os = "macos")]
fn volumes(media: &[PathBuf]) -> Vec<Vol> {
    let mut v = vec![];
    for e in media.iter().flat_map(fs::read_dir).flatten().flatten() {
        let root = e.path();
        let Some(dev) = mounted(&root).filter(|_| root.join("DCIM").is_dir()) else {
            continue;
        };
        let l = e.file_name().to_string_lossy().into_owned();
        v.push(Vol {
            cam: l.starts_with("Insta360").then(|| l.clone()),
            root,
            dev,
        })
    }
    v
}

/// The block device mounted exactly at `p` (not a symlink to another mount, as `/Volumes/Macintosh HD` is).
#[cfg(target_os = "macos")]
fn mounted(p: &Path) -> Option<String> {
    use std::{ffi::CStr, os::unix::ffi::OsStrExt};
    let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a C string, `s` a statfs to fill
    if unsafe { libc::statfs(c.as_ptr(), &mut s) } != 0 {
        return None;
    }
    let on = unsafe { CStr::from_ptr(s.f_mntonname.as_ptr()) };
    let from = unsafe { CStr::from_ptr(s.f_mntfromname.as_ptr()) }.to_string_lossy();
    (on.to_bytes() == p.as_os_str().as_bytes() && from.starts_with("/dev/"))
        .then(|| from.into_owned())
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
                let r = p.strip_prefix(root).unwrap_or(&p);
                uris.push(format!("/{}", r.to_string_lossy()))
            }
        }
    }
    uris.sort();
    let ed = fs::read(root.join("DCIM/.editinfo_list.list"))
        .ok()
        .and_then(|b| pb::EditFileInfoList::decode(&b[..]).ok())
        .unwrap_or_default();
    // one trailer per item: grouped first to know each item's first file, then again with their metadata
    let infos: Vec<(String, Option<pb::ExtraMetadata>)> = album::group(&uris, &[], &ed)
        .into_iter()
        .map(|it| {
            let u = &it.uris[0];
            (u.clone(), meta(&root.join(&u[1..])))
        })
        .collect();
    Ok(album::group(&uris, &infos, &ed))
}

/// Date, length and size from a file's trailer (lenient: the camera's own fields), the size from the file system.
fn meta(p: &Path) -> Option<pb::ExtraMetadata> {
    let n = fs::metadata(p).ok()?.len();
    let m = Trailer::open(p).map(|t| t.meta).unwrap_or_default();
    Some(pb::ExtraMetadata {
        creation_time: m.created,
        total_time_ms: m.duration_ms,
        dimension: (m.dimension[0] > 0).then(|| pb::extra_metadata::Vector2 {
            x: m.dimension[0] as i32,
            y: m.dimension[1] as i32,
        }),
        file_size: n,
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
        i.read_exact_at(&mut a, at - k)?;
        o.read_exact_at(&mut b, at - k)?;
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
