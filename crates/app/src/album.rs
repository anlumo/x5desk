//! The camera's album: listed files grouped into items the way the official app groups them (a recording's lens
//! files and proxy, a photo and its raw; PROTOCOL.md §9.2 fileKey), their metadata and favourite flags, and
//! thumbnails: the trailer's NV12 picture scaled down and cached on disk.

use crate::library::key;
use link::pb;
use std::{
    collections::HashMap,
    io::Read,
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    #[default]
    Video,
    Photo,
    Other,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Item {
    /// fileKey: directory and name key
    pub key: String,
    pub kind: Kind,
    /// every file in download order: main files in lens order (or the photo), proxy, raw, the rest; the first one
    /// carries the metadata and the thumbnail
    pub uris: Vec<String>,
    /// camera local time as decimal YYYYMMDDhhmmss (metadata, else the name)
    pub date: u64,
    pub ms: u64,
    pub dim: [u32; 2],
    /// of the first file per its metadata, 0 if unknown: exact for a video's media data, approximate for a photo
    pub size: u64,
    pub fav: bool,
}

impl Item {
    /// The file the camera keeps the item's favourite flag on, picked as the app does (`WorkEditInfoUtils.a`): a
    /// photo's first file, a video's last main file (`_10_` of a recording in two files).
    pub fn flag(&self) -> &str {
        let u = &self.uris;
        match self.kind {
            Kind::Video => u.iter().rfind(|u| rank(u).0 == 0).unwrap_or(&u[0]),
            _ => &u[0],
        }
    }
}

pub fn name(u: &str) -> &str {
    u.rsplit('/').next().unwrap_or(u)
}

/// Directory and name: what a path keeps under another mount prefix.
fn tail(u: &str) -> &str {
    u.rmatch_indices('/').nth(1).map_or(u, |(i, _)| &u[i + 1..])
}

fn ext(u: &str) -> String {
    let n = name(u);
    n.rsplit_once('.').map_or("", |e| e.1).to_ascii_lowercase()
}

/// Download order within an item: main file or photo, proxy, raw, other; then the stream part (00, 10, 01, 11).
fn rank(u: &str) -> (u8, u8) {
    let n = name(u);
    let stem = n.rsplit_once('.').map_or(n, |s| s.0);
    let c = match ext(u).as_str() {
        "insv" | "mp4" | "jpg" | "jpeg" | "insp" => 0,
        "lrv" => 1,
        "dng" | "raw" => 2,
        _ => 3,
    };
    let p = stem
        .split('_')
        .find_map(|x| ["00", "10", "01", "11"].iter().position(|s| *s == x));
    (c, p.map_or(4, |p| p as u8))
}

/// YYYYMMDDhhmmss from the date and time parts of a name (8 and 6 digits).
fn name_date(n: &str) -> u64 {
    let d = |k: usize| {
        n.split(['_', '.'])
            .find(|p| p.len() == k && p.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|p| p.parse::<u64>().ok())
    };
    d(8).map_or(0, |a| a * 1_000_000 + d(6).unwrap_or(0))
}

/// Items of the listed `uris` in listing order, with the metadata of `infos` and the favourite flags of `ed`
/// (matched by directory and name, the edit list's paths need not carry the listing's prefix; else by name alone).
pub fn group(
    uris: &[String],
    infos: &[(String, Option<pb::ExtraMetadata>)],
    ed: &pb::EditFileInfoList,
) -> Vec<Item> {
    let mut v: Vec<Item> = vec![];
    let mut at = HashMap::new();
    for u in uris {
        let d = u.rsplit_once('/').map_or("", |d| d.0);
        let k = format!("{d}/{}", key(name(u)));
        let i = *at.entry(k.clone()).or_insert_with(|| {
            v.push(Item {
                key: k,
                ..Default::default()
            });
            v.len() - 1
        });
        if !v[i].uris.contains(u) {
            v[i].uris.push(u.clone())
        }
    }
    let info: HashMap<&str, &pb::ExtraMetadata> = infos
        .iter()
        .filter_map(|(p, m)| Some((p.as_str(), m.as_ref()?)))
        .collect();
    let fav: HashMap<&str, bool> = ed
        .edit_file_info_list
        .iter()
        .map(|e| {
            let f = e.favorite_info.as_ref();
            (
                tail(&e.file_path),
                f.is_some_and(|f| f.playback_is_favorite),
            )
        })
        .collect();
    for it in &mut v {
        it.uris
            .sort_by(|a, b| (rank(a), name(a)).cmp(&(rank(b), name(b))));
        let u = &it.uris[0];
        it.kind = match it.uris.iter().map(|u| ext(u)).find(|e| {
            matches!(
                e.as_str(),
                "insv" | "mp4" | "lrv" | "jpg" | "jpeg" | "insp" | "dng" | "raw"
            )
        }) {
            Some(e) if matches!(e.as_str(), "insv" | "mp4" | "lrv") => Kind::Video,
            Some(_) => Kind::Photo,
            None => Kind::Other,
        };
        if let Some(m) = info.get(u.as_str()) {
            it.date = m.creation_time;
            it.ms = match m.total_time_ms {
                0 => m.total_time as u64 * 1000,
                t => t as u64,
            };
            it.dim = m.dimension.map_or([0; 2], |d| [d.x as u32, d.y as u32]);
            it.size = m.file_size;
        }
        if it.date == 0 {
            it.date = name_date(name(u));
        }
        it.fav = it
            .uris
            .iter()
            .any(|u| fav.get(tail(u)).or_else(|| fav.get(name(u))) == Some(&true));
    }
    v
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sort {
    Date,
    Name,
    Size,
    Length,
}

impl Sort {
    pub const ALL: [(Sort, &str); 4] = [
        (Sort::Date, "Date"),
        (Sort::Name, "Name"),
        (Sort::Size, "Size"),
        (Sort::Length, "Length"),
    ];
}

/// What the album shows: sort order and filters.
#[derive(Clone, Debug)]
pub struct View {
    pub sort: Sort,
    pub desc: bool,
    pub kind: Option<Kind>,
    pub fav: bool,
    pub find: String,
}

impl Default for View {
    fn default() -> View {
        View {
            sort: Sort::Date,
            desc: true,
            kind: None,
            fav: false,
            find: String::new(),
        }
    }
}

impl View {
    /// Indices into `items` of what is shown, in order.
    pub fn apply(&self, items: &[Item], out: &mut Vec<usize>) {
        out.clear();
        let f = self.find.to_ascii_lowercase();
        out.extend((0..items.len()).filter(|&i| {
            let t = &items[i];
            self.kind.is_none_or(|k| t.kind == k)
                && (!self.fav || t.fav)
                && (f.is_empty()
                    || t.uris
                        .iter()
                        .any(|u| name(u).to_ascii_lowercase().contains(&f)))
        }));
        out.sort_by(|&a, &b| {
            let (x, y) = (&items[a], &items[b]);
            let o = match self.sort {
                Sort::Date => x.date.cmp(&y.date),
                Sort::Name => name(&x.uris[0]).cmp(name(&y.uris[0])),
                Sort::Size => x.size.cmp(&y.size),
                Sort::Length => x.ms.cmp(&y.ms),
            }
            .then_with(|| x.key.cmp(&y.key));
            if self.desc { o.reverse() } else { o }
        });
    }
}

/// Thumbnail box [`TW`] × [`TW`] / 2: the X5's trailer pictures (1280 × 640 videos, 2560 × 1280 photos) scale down
/// to it by whole factors.
pub const TW: u32 = 320;

/// RGBA of a trailer thumbnail ([`link::nv12`]) scaled down by the whole factor that fits the [`TW`] box (any result
/// is under twice the box in each direction): Y box-filtered, chroma from the block centre, BT.601 full range (the
/// X5's Y spans 0..255). `None` if nothing is left.
pub fn thumb(b: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let (w, h, p) = link::nv12(b)?;
    let k = (w / TW).max(h / (TW / 2)).max(1);
    let (ow, oh) = (w / k, h / k);
    if ow == 0 || oh == 0 {
        return None;
    }
    let (y, uv) = p.split_at((w * h) as usize);
    let n = (k * k) as f32;
    let mut o = Vec::with_capacity((ow * oh * 4) as usize);
    for j in 0..oh {
        for i in 0..ow {
            let mut s = 0u32;
            for r in j * k..(j + 1) * k {
                let a = (r * w + i * k) as usize;
                s += y[a..a + k as usize].iter().map(|&x| x as u32).sum::<u32>();
            }
            let c = ((j * k + k / 2) / 2 * w + (i * k + k / 2) / 2 * 2) as usize;
            let (l, u, v) = (s as f32 / n, uv[c] as f32 - 128.0, uv[c + 1] as f32 - 128.0);
            o.extend(
                [
                    l + 1.402 * v,
                    l - 0.344136 * u - 0.714136 * v,
                    l + 1.772 * u,
                ]
                .map(|x| (x + 0.5).clamp(0.0, 255.0) as u8),
            );
            o.push(255);
        }
    }
    Some((ow, oh, o))
}

/// Thumbnails on disk, one file per item and camera: `u32 w, h` (LE), then RGBA. Least recently used first out
/// (the modification time is the last use) once they take more than `cap` bytes.
#[derive(Clone)]
pub struct Cache {
    dir: Option<PathBuf>,
    pub cap: u64,
}

/// Default disk cache size: about 5000 thumbnails.
pub const CAP: u64 = 1_000_000_000;

impl Cache {
    /// `<cache dir>/x5desk/thumbs`.
    pub fn new() -> Cache {
        Cache::at(crate::xdg("XDG_CACHE_HOME", ".cache").map(|d| d.join("x5desk").join("thumbs")))
    }

    pub fn at(dir: Option<PathBuf>) -> Cache {
        Cache { dir, cap: CAP }
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Entries and their bytes.
    pub fn usage(&self) -> (usize, u64) {
        let Some(d) = &self.dir else { return (0, 0) };
        std::fs::read_dir(d)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok()?.metadata().ok())
            .filter(|m| m.is_file())
            .fold((0, 0), |(n, b), m| (n + 1, b + m.len()))
    }

    /// FNV-1a of serial, URI, size and date: a file replaced under the same name gets a new entry.
    pub fn path(&self, sn: &str, it: &Item) -> Option<PathBuf> {
        let mut x = 0xcbf2_9ce4_8422_2325u64;
        let s = format!("{sn}\0{}\0{}\0{}", it.uris[0], it.size, it.date);
        for b in s.bytes() {
            x = (x ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
        Some(self.dir.as_ref()?.join(format!("{x:016x}")))
    }

    /// `None` unless the file is a whole entry of a size [`thumb`] makes: a short or damaged one (a crash between
    /// write and rename) is fetched again and rewritten.
    pub fn get(p: &Path) -> Option<(u32, u32, Vec<u8>)> {
        let mut f = std::fs::File::open(p).ok()?;
        let mut b = vec![];
        f.read_to_end(&mut b).ok()?;
        let u = |o: usize| Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?));
        let (w, h) = (u(0)?, u(4)?);
        let ok = (1..2 * TW).contains(&w) && (1..TW).contains(&h);
        if !ok || b.len() != 8 + (w * h * 4) as usize {
            return None;
        }
        let _ = f.set_modified(SystemTime::now());
        Some((w, h, b.split_off(8)))
    }

    pub fn put(p: &Path, w: u32, h: u32, px: &[u8]) {
        let mut b = Vec::with_capacity(8 + px.len());
        b.extend(w.to_le_bytes());
        b.extend(h.to_le_bytes());
        b.extend(px);
        // written whole, then renamed: a reader never sees half a file
        let t = p.with_extension("new");
        let r = p
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|_| std::fs::write(&t, b))
            .and_then(|_| std::fs::rename(&t, p));
        if let Err(e) = r {
            eprintln!("{}: {e}", p.display());
        }
    }

    /// Drops the least recently used entries until the rest fit in `cap` bytes.
    pub fn trim(&self, cap: u64) {
        let Some(d) = &self.dir else { return };
        let mut v: Vec<(SystemTime, u64, PathBuf)> = std::fs::read_dir(d)
            .into_iter()
            .flatten()
            .filter_map(|e| {
                let e = e.ok()?;
                let m = e.metadata().ok()?;
                Some((m.modified().ok()?, m.len(), e.path()))
            })
            .collect();
        let mut n: u64 = v.iter().map(|e| e.1).sum();
        v.sort();
        for (_, l, p) in v {
            if n <= cap {
                break;
            }
            if std::fs::remove_file(&p).is_ok() {
                n -= l
            }
        }
    }
}

/// `20260116152935` → `2026-01-16 15:29:35`.
pub fn date(d: u64) -> String {
    let s = format!("{d:014}");
    format!(
        "{}-{}-{} {}:{}:{}",
        &s[..4],
        &s[4..6],
        &s[6..8],
        &s[8..10],
        &s[10..12],
        &s[12..14]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(t: u64, size: u64, ms: u32, d: [i32; 2]) -> Option<pb::ExtraMetadata> {
        Some(pb::ExtraMetadata {
            creation_time: t,
            file_size: size,
            total_time_ms: ms,
            dimension: Some(pb::extra_metadata::Vector2 { x: d[0], y: d[1] }),
            ..Default::default()
        })
    }

    #[test]
    fn groups() {
        let c = "/DCIM/Camera01/";
        let u: Vec<String> = [
            "IMG_20260112_204234_00_597.jpg",
            "LRV_20260116_152935_01_603.lrv",
            "VID_20260116_152935_00_603.insv",
            "VID_20260116_152935_10_603.insv",
            "IMG_20260118_124432_10_659.jpg",
            "IMG_20260118_124432_10_659.dng",
            "LRV_20260503_190639_01_710.lrv",
            "VID_20260307_135339_00_670.insv",
        ]
        .iter()
        .map(|n| format!("{c}{n}"))
        .chain(["/DCIM/Camera02/VID_20260116_152935_00_603.insv".into()])
        .collect();
        let infos = vec![
            (
                u[0].clone(),
                meta(20260112204234, 11568287, 0, [11904, 5952]),
            ),
            (
                u[2].clone(),
                meta(20260116152936, 121819479, 42_500, [7680, 3840]),
            ),
            (u[7].clone(), None),
        ];
        let ed = pb::EditFileInfoList {
            edit_file_info_list: vec![pb::EditFileInfo {
                file_path: "/mnt/sd/DCIM/Camera01/VID_20260116_152935_00_603.insv".into(),
                favorite_info: Some(pb::FavoriteInfo {
                    playback_is_favorite: true,
                    modify_timestamp: 1,
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let v = group(&u, &infos, &ed);
        assert_eq!(v.len(), 6);
        let f = |n: &str| v.iter().find(|i| i.uris[0].ends_with(n)).unwrap();
        let rec = f("VID_20260116_152935_00_603.insv");
        assert_eq!(rec.uris.len(), 3);
        assert!(rec.uris[1].ends_with("_10_603.insv") && rec.uris[2].ends_with(".lrv"));
        assert_eq!(rec.flag(), rec.uris[1]);
        assert_eq!(
            (rec.kind, rec.date, rec.ms, rec.dim, rec.size, rec.fav),
            (
                Kind::Video,
                20260116152936,
                42_500,
                [7680, 3840],
                121819479,
                true
            )
        );
        // same name in another directory: its own item, not a favourite
        let other = v
            .iter()
            .find(|i| i.key.starts_with("/DCIM/Camera02/"))
            .unwrap();
        assert_eq!((other.uris.len(), other.fav), (1, false));
        let p = f("IMG_20260112_204234_00_597.jpg");
        assert_eq!(
            (p.kind, p.dim, p.uris.len()),
            (Kind::Photo, [11904, 5952], 1)
        );
        let raw = f("IMG_20260118_124432_10_659.jpg");
        assert_eq!((raw.kind, raw.uris.len()), (Kind::Photo, 2));
        assert!(raw.uris[1].ends_with(".dng"));
        assert_eq!(raw.flag(), raw.uris[0]);
        // no metadata: the date comes from the name; a proxy without its main file is still a video
        assert_eq!(f("VID_20260307_135339_00_670.insv").date, 20260307135339);
        let lone = f("LRV_20260503_190639_01_710.lrv");
        assert_eq!(
            (lone.kind, lone.date, lone.size),
            (Kind::Video, 20260503190639, 0)
        );
        assert_eq!(lone.flag(), lone.uris[0]);

        let mut o = vec![];
        let mut w = View::default();
        w.apply(&v, &mut o);
        assert!(o.windows(2).all(|p| v[p[0]].date >= v[p[1]].date));
        assert_eq!(v[o[0]].date, 20260503190639);
        w.kind = Some(Kind::Photo);
        w.apply(&v, &mut o);
        assert_eq!(o.len(), 2);
        w = View {
            fav: true,
            ..Default::default()
        };
        w.apply(&v, &mut o);
        assert_eq!(o.len(), 1);
        w = View {
            find: "_670".into(),
            sort: Sort::Size,
            desc: false,
            ..Default::default()
        };
        w.apply(&v, &mut o);
        assert_eq!(o.len(), 1);
        assert_eq!(date(20260116152936), "2026-01-16 15:29:36");
    }

    /// A 1280 × 640 picture of flat colours scales to 320 × 160 with the colours kept; a cached copy reads back
    /// the same.
    #[test]
    fn thumbs() {
        let (w, h) = (1280u32, 640u32);
        let mut b = vec![0u8; 40 + (w * h * 3 / 2) as usize];
        let n = b.len() as u32;
        for (o, v) in [(0, 1), (4, n), (16, w), (20, h)] {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes())
        }
        // left half grey (Y 128, neutral), right half red (Y 76, U 85, V 255 in full-range BT.601)
        for r in 0..h {
            for c in 0..w {
                b[40 + (r * w + c) as usize] = if c < w / 2 { 128 } else { 76 };
            }
        }
        for r in 0..h / 2 {
            for c in 0..w / 2 {
                let o = 40 + (w * h + r * w + c * 2) as usize;
                let red = c >= w / 4;
                b[o] = if red { 85 } else { 128 };
                b[o + 1] = if red { 255 } else { 128 };
            }
        }
        let (tw, th, px) = thumb(&b).unwrap();
        assert_eq!((tw, th, px.len()), (320, 160, 320 * 160 * 4));
        let at = |x: u32, y: u32| &px[((y * tw + x) * 4) as usize..][..4];
        assert_eq!(at(10, 10), [128, 128, 128, 255]);
        let r = at(300, 100);
        assert!(r[0] > 250 && r[1] < 5 && r[2] < 5, "{r:?}");
        assert!(thumb(&b[..1000]).is_none());
        // a tall picture fits the box too: 64 × 1280 → 8 × 160
        let mut t = vec![128u8; 40 + 64 * 1280 * 3 / 2];
        let n = t.len() as u32;
        for (o, v) in [(0, 1), (4, n), (16, 64), (20, 1280)] {
            t[o..o + 4].copy_from_slice(&v.to_le_bytes())
        }
        assert_eq!(thumb(&t).map(|t| (t.0, t.1)), Some((8, 160)));
        t[16..20].copy_from_slice(&2u32.to_le_bytes());
        t[20..24].copy_from_slice(&40960u32.to_le_bytes());
        assert!(thumb(&t).is_none(), "2 × 40960 leaves nothing");

        // a name of its own: settings::tests::cache_thread runs alongside in `x5desk-thumbs-<pid>`
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-cache-{}", std::process::id()));
        let c = Cache::at(Some(d.clone()));
        let it = Item {
            uris: vec!["/DCIM/Camera01/a.insv".into()],
            size: 5,
            ..Default::default()
        };
        let p = c.path("sn", &it).unwrap();
        assert_ne!(Some(&p), c.path("sn2", &it).as_ref());
        Cache::put(&p, tw, th, &px);
        assert_eq!(Cache::get(&p).as_ref(), Some(&(tw, th, px.clone())));
        // what a crash or a stray file leaves: empty, short, a header without its pixels, sizes no thumbnail has
        let mut whole = std::fs::read(&p).unwrap();
        for b in [
            &whole[..0],
            &whole[..5],
            &whole[..8],
            &whole[..whole.len() - 1],
        ] {
            std::fs::write(&p, b).unwrap();
            assert_eq!(Cache::get(&p), None, "{} bytes", b.len());
        }
        for (w, h) in [(0, 0), (u32::MAX, u32::MAX), (1 << 15, 1 << 15), (8, 4096)] {
            whole[..8].copy_from_slice(&[w.to_le_bytes(), h.to_le_bytes()].concat());
            std::fs::write(&p, &whole).unwrap();
            assert_eq!(Cache::get(&p), None, "{w} × {h}");
        }

        // least recently used out: a hit counts as a use
        let at =
            |s: u64| SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000 + s);
        let ps: Vec<PathBuf> = (0..4).map(|i| d.join(format!("t{i}"))).collect();
        for (i, q) in ps.iter().enumerate() {
            Cache::put(q, tw, th, &px);
            let f = std::fs::File::options().write(true).open(q).unwrap();
            f.set_modified(at(i as u64)).unwrap();
        }
        assert!(Cache::get(&ps[0]).is_some());
        std::fs::remove_file(&p).unwrap();
        let n = std::fs::metadata(&ps[0]).unwrap().len();
        c.trim(2 * n);
        let left: Vec<bool> = ps.iter().map(|q| q.exists()).collect();
        assert_eq!(left, [true, false, false, true]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The real camera's thumbnails (wave-1 captures in `X5_LINK`): a photo (2560 × 1280) and a video (1280 × 640)
    /// scale to 320 × 160.
    #[test]
    fn real_thumbs() {
        let Some(d) = std::env::var_os("X5_LINK").map(PathBuf::from) else {
            eprintln!("X5_LINK not set: skipping");
            return;
        };
        for f in ["thumb.bin", "vthumb.bin"] {
            let b = std::fs::read(d.join(f)).unwrap();
            let t = std::time::Instant::now();
            let (w, h, px) = thumb(&b).unwrap();
            println!(
                "{f}: {w} × {h} in {:.1} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            assert_eq!((w, h, px.len()), (320, 160, 320 * 160 * 4));
        }
    }
}
