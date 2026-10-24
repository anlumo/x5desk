//! MP4 / MOV demux: sample tables, codec configs, edit lists and `udta` passthrough. Works on X5 `.insv` (two HEVC
//! tracks, one per lens, plus AAC; the trailer sits in a top-level `inst` box after `moov`) and `.lrv` (one
//! side-by-side H.264 track plus AAC). Fragmented MP4: the samples of every `moof` (all of them read at open)
//! follow those of the `moov`.

use crate::{Bits, Rd, bad, read_at};
use std::{fs::File, io, path::Path};

/// One access unit. Times are in the track's timescale, on the media timeline (before the edit list, see
/// [`Track::shift`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    pub off: u64,
    pub dts: i64,
    pub pts: i64,
    pub size: u32,
    pub dur: u32,
    pub sync: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Codec {
    /// `hvc1` / `hev1`: the `hvcC` NAL arrays in box order (VPS, SPS, PPS, SEI), `nal` = NAL length size in bytes.
    Hevc {
        ps: Vec<Vec<u8>>,
        nal: u8,
    },
    /// `avc1` / `avc3`: SPS then PPS NALs from `avcC`.
    Avc {
        ps: Vec<Vec<u8>>,
        nal: u8,
    },
    /// `mp4a` with an AAC AudioSpecificConfig (from `esds`); `rate` and `ch` are decoded from it.
    Aac {
        asc: Vec<u8>,
        rate: u32,
        ch: u16,
    },
    Other([u8; 4]),
}

/// `elst` entry. `dur` is in the movie timescale, `start` in the track's (−1 = empty edit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edit {
    pub dur: u64,
    pub start: i64,
    pub rate: f32,
}

#[derive(Clone, Debug)]
pub struct Track {
    pub id: u32,
    /// `hdlr` type: `vide`, `soun`, …
    pub handler: [u8; 4],
    /// Sample entry type (`hvc1`, `avc1`, `mp4a`, …).
    pub fourcc: [u8; 4],
    pub codec: Codec,
    pub timescale: u32,
    /// `mdhd` duration, track timescale.
    pub duration: u64,
    /// Sample entry size (video).
    pub width: u32,
    pub height: u32,
    pub edits: Vec<Edit>,
    pub samples: Vec<Sample>,
}

impl Track {
    pub fn secs(&self) -> f64 {
        self.duration as f64 / self.timescale as f64
    }
    /// Media → presentation offset in track ticks (add it to `pts` / `dts`) for the usual edit lists: leading empty
    /// edits (a start delay), then one media edit (B-frame delay, AAC priming). Later edits (cuts, repeats) and
    /// rates other than 1 are not applied. `movie_ts` = [`Mp4::timescale`].
    pub fn shift(&self, movie_ts: u32) -> i64 {
        let mut d = 0u128;
        for e in &self.edits {
            if e.start >= 0 {
                return (d * self.timescale as u128 / movie_ts.max(1) as u128) as i64 - e.start;
            }
            d += e.dur as u128;
        }
        0
    }
    /// Index of the last sample with `dts <= t` (0 if none).
    pub fn at(&self, t: i64) -> usize {
        self.samples
            .partition_point(|s| s.dts <= t)
            .saturating_sub(1)
    }
}

/// A top-level or `udta` child box: kind and its whole extent (header included) in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Atom {
    pub kind: [u8; 4],
    pub off: u64,
    pub len: u64,
}

pub struct Mp4 {
    pub file: File,
    pub brand: [u8; 4],
    pub timescale: u32,
    /// `mvhd` duration, movie timescale.
    pub duration: u64,
    pub tracks: Vec<Track>,
    /// Children of `moov/udta`, for passthrough with [`Mp4::copy_udta`].
    pub udta: Vec<Atom>,
}

impl Mp4 {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Mp4> {
        let file = File::open(path)?;
        let end = file.metadata()?.len();
        let mut m = Mp4 {
            file,
            brand: [0; 4],
            timescale: 0,
            duration: 0,
            tracks: vec![],
            udta: vec![],
        };
        let (mut moov, mut moofs) = (None, vec![]);
        for a in atoms(&m.file, 0, end)? {
            match &a.kind {
                b"ftyp" => {
                    m.brand = m
                        .load(a)?
                        .get(8..12)
                        .ok_or_else(|| bad("short ftyp"))?
                        .try_into()
                        .unwrap()
                }
                b"moov" => moov = Some(a),
                b"moof" => moofs.push(a),
                _ => {}
            }
        }
        let moov = moov.ok_or_else(|| bad("no moov"))?;
        // `trex` per track: id, default duration, size, flags
        let mut trex = vec![];
        let h = hdr(&m.file, moov.off)?;
        for a in atoms(&m.file, moov.off + h, moov.off + moov.len)? {
            match &a.kind {
                b"mvhd" => {
                    let b = m.load(a)?;
                    let mut r = Rd(&b[hdr_len(&b)..]);
                    let v1 = r.u8()? == 1;
                    r.skip(3)?;
                    r.uv(v1)?;
                    r.uv(v1)?;
                    m.timescale = r.u32()?;
                    m.duration = r.uv(v1)?;
                }
                b"trak" => {
                    let b = m.load(a)?;
                    m.tracks.push(trak(&b[hdr_len(&b)..], end)?);
                }
                b"udta" => {
                    let h = hdr(&m.file, a.off)?;
                    m.udta = atoms(&m.file, a.off + h, a.off + a.len)?;
                }
                b"mvex" => {
                    let b = m.load(a)?;
                    for (_, p) in kids(&b[hdr_len(&b)..]).filter(|(k, _)| k == b"trex") {
                        let mut r = Rd(p);
                        r.skip(4)?;
                        let id = r.u32()?;
                        r.skip(4)?;
                        trex.push((id, r.u32()?, r.u32()?, r.u32()?));
                    }
                }
                _ => {}
            }
        }
        for a in moofs {
            let b = m.load(a)?;
            // a `traf` without an explicit base starts where the one before ended (the first: at the `moof`)
            let mut prev = a.off;
            for (_, tf) in kids(&b[hdr_len(&b)..]).filter(|(k, _)| k == b"traf") {
                let mut r = Rd(kid(tf, b"tfhd").ok_or_else(|| bad("no tfhd"))?);
                let fl = r.u32()?;
                let id = r.u32()?;
                let Some(t) = m.tracks.iter_mut().find(|t| t.id == id) else {
                    continue;
                };
                let (_, mut dd, mut ds, mut df) =
                    trex.iter().copied().find(|x| x.0 == id).unwrap_or_default();
                let base = if fl & 1 != 0 {
                    r.u64()?
                } else if fl & 0x20000 != 0 {
                    a.off
                } else {
                    prev
                };
                if fl & 2 != 0 {
                    r.skip(4)?;
                }
                for (bit, v) in [(8, &mut dd), (0x10, &mut ds), (0x20, &mut df)] {
                    if fl & bit != 0 {
                        *v = r.u32()?;
                    }
                }
                let mut dts = match kid(tf, b"tfdt") {
                    Some(p) => {
                        let mut r = Rd(p);
                        let v1 = r.u8()? == 1;
                        r.skip(3)?;
                        r.uv(v1)? as i64
                    }
                    None => t.samples.last().map_or(0, |s| s.dts + s.dur as i64),
                };
                let mut off = base;
                for (_, p) in kids(tf).filter(|(k, _)| k == b"trun") {
                    let mut r = Rd(p);
                    let (fl, n) = (r.u32()? & 0xffffff, r.u32()? as usize);
                    if fl & 1 != 0 {
                        off = base.wrapping_add_signed(r.u32()? as i32 as i64);
                    }
                    let first = if fl & 4 != 0 { r.u32()? } else { df };
                    // per-sample fields bound the count; without them, a track never has more samples than the
                    // file has bytes, and samples of the default size must fit in it (as for a constant `stsz`)
                    let per = (fl >> 8 & 15).count_ones() as usize * 4;
                    if per > 0 && n > r.0.len() / per
                        || (t.samples.len() + n) as u64 > end
                        || fl & 0x200 == 0 && n as u64 * ds as u64 > end
                    {
                        return Err(bad("bad trun"));
                    }
                    t.samples.reserve(n);
                    for j in 0..n {
                        let mut f = |bit, d| if fl & bit != 0 { r.u32() } else { Ok(d) };
                        let (dur, size) = (f(0x100, dd)?, f(0x200, ds)?);
                        let sf = f(0x400, if j == 0 { first } else { df })?;
                        // signed in both versions, as in `ctts`
                        let cto = f(0x800, 0)? as i32 as i64;
                        t.samples.push(Sample {
                            off,
                            dts,
                            pts: dts + cto,
                            size,
                            dur,
                            // not a non-sync sample, depends on no other (as FFmpeg reads it)
                            sync: sf & 0x0101_0000 == 0,
                        });
                        (off, dts) = (off + size as u64, dts + dur as i64);
                    }
                }
                prev = off;
                // the `mdhd` / `mvhd` of a fragmented file count only the `moov`'s samples (often none)
                let d0 = t.samples.first().map_or(dts, |s| s.dts);
                t.duration = t.duration.max((dts - d0).max(0) as u64);
                let d = (t.duration as u128 * m.timescale as u128 / t.timescale as u128) as u64;
                m.duration = m.duration.max(d);
            }
        }
        Ok(m)
    }

    /// Reads `len` bytes at `off` into `buf`, reusing its allocation. Positional, so `&self` suffices.
    pub fn read_at(&self, off: u64, len: usize, buf: &mut Vec<u8>) -> io::Result<()> {
        buf.resize(len, 0);
        read_at(&self.file, buf, off)
    }

    pub fn read(&self, s: &Sample, buf: &mut Vec<u8>) -> io::Result<()> {
        self.read_at(s.off, s.size as usize, buf)
    }

    /// The raw `udta` child boxes that pass `keep`, concatenated (for [`crate::mux::Mux::udta`]).
    pub fn copy_udta(&self, keep: impl Fn(&Atom) -> bool) -> io::Result<Vec<u8>> {
        let (mut out, mut b) = (vec![], vec![]);
        for a in self.udta.iter().filter(|a| keep(a)) {
            self.read_at(a.off, a.len as usize, &mut b)?;
            out.extend_from_slice(&b);
        }
        Ok(out)
    }

    fn load(&self, a: Atom) -> io::Result<Vec<u8>> {
        if a.len > 1 << 30 {
            return Err(bad("box too large"));
        }
        let mut b = vec![];
        self.read_at(a.off, a.len as usize, &mut b)?;
        Ok(b)
    }
}

/// Header length of the box at `off` (8, or 16 with a 64-bit size).
fn hdr(f: &File, off: u64) -> io::Result<u64> {
    let mut h = [0; 4];
    read_at(f, &mut h, off)?;
    Ok(if u32::from_be_bytes(h) == 1 { 16 } else { 8 })
}

fn hdr_len(b: &[u8]) -> usize {
    if b[..4] == [0, 0, 0, 1] { 16 } else { 8 }
}

/// Box headers in `[start, end)` of the file, read without loading payloads.
pub(crate) fn atoms(f: &File, start: u64, end: u64) -> io::Result<Vec<Atom>> {
    let mut v = vec![];
    let mut p = start;
    while p + 8 <= end {
        let mut h = [0; 16];
        read_at(f, &mut h[..8], p)?;
        let mut len = u32::from_be_bytes(h[..4].try_into().unwrap()) as u64;
        if len == 1 {
            read_at(f, &mut h[8..], p + 8)?;
            len = u64::from_be_bytes(h[8..].try_into().unwrap());
        } else if len == 0 {
            len = end - p;
        }
        if len < 8 || p + len > end {
            break; // truncated tail (e.g. a file still being written): keep what parsed
        }
        v.push(Atom {
            kind: h[4..8].try_into().unwrap(),
            off: p,
            len,
        });
        p += len;
    }
    Ok(v)
}

/// In-memory child boxes: (kind, payload). Stops at the first malformed header.
fn kids(mut b: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    std::iter::from_fn(move || {
        if b.len() < 8 {
            return None;
        }
        let (mut n, mut h) = (u32::from_be_bytes(b[..4].try_into().unwrap()) as usize, 8);
        if n == 1 {
            n = u64::from_be_bytes(b.get(8..16)?.try_into().unwrap())
                .try_into()
                .ok()?;
            h = 16;
        } else if n == 0 {
            n = b.len();
        }
        if n < h || n > b.len() {
            return None;
        }
        let (x, r) = b.split_at(n);
        b = r;
        Some((x[4..8].try_into().unwrap(), &x[h..]))
    })
}

fn kid<'a>(b: &'a [u8], k: &[u8; 4]) -> Option<&'a [u8]> {
    kids(b).find(|(t, _)| t == k).map(|(_, p)| p)
}

/// `flen` = file size, which bounds the sample count of a constant-size `stsz`.
fn trak(b: &[u8], flen: u64) -> io::Result<Track> {
    let mut t = Track {
        id: 0,
        handler: [0; 4],
        fourcc: [0; 4],
        codec: Codec::Other([0; 4]),
        timescale: 0,
        duration: 0,
        width: 0,
        height: 0,
        edits: vec![],
        samples: vec![],
    };
    let mut r = Rd(kid(b, b"tkhd").ok_or_else(|| bad("no tkhd"))?);
    let v1 = r.u8()? == 1;
    r.skip(3)?;
    r.uv(v1)?;
    r.uv(v1)?;
    t.id = r.u32()?;
    if let Some(e) = kid(b, b"edts").and_then(|e| kid(e, b"elst")) {
        let mut r = Rd(e);
        let v1 = r.u8()? == 1;
        r.skip(3)?;
        for _ in 0..r.u32()? {
            let dur = r.uv(v1)?;
            let start = if v1 {
                r.u64()? as i64
            } else {
                r.u32()? as i32 as i64
            };
            let rate = r.u32()? as i32 as f32 / 65536.0;
            t.edits.push(Edit { dur, start, rate });
        }
    }
    let mdia = kid(b, b"mdia").ok_or_else(|| bad("no mdia"))?;
    let mut r = Rd(kid(mdia, b"mdhd").ok_or_else(|| bad("no mdhd"))?);
    let v1 = r.u8()? == 1;
    r.skip(3)?;
    r.uv(v1)?;
    r.uv(v1)?;
    t.timescale = r.u32()?;
    t.duration = r.uv(v1)?;
    if t.timescale == 0 {
        return Err(bad("zero timescale"));
    }
    if let Some(h) = kid(mdia, b"hdlr") {
        t.handler = Rd(h).take(12)?[8..].try_into().unwrap();
    }
    let stbl = kid(mdia, b"minf")
        .and_then(|m| kid(m, b"stbl"))
        .ok_or_else(|| bad("no stbl"))?;
    if let Some(d) = kid(stbl, b"stsd") {
        stsd(&mut t, d)?;
    }
    t.samples = table(stbl, flen)?;
    Ok(t)
}

fn stsd(t: &mut Track, b: &[u8]) -> io::Result<()> {
    let Some((k, e)) = kids(b.get(8..).unwrap_or(&[])).next() else {
        return Ok(());
    };
    t.fourcc = k;
    t.codec = Codec::Other(k);
    let mut r = Rd(e);
    r.skip(8)?; // reserved, data_reference_index
    match &k {
        b"hvc1" | b"hev1" | b"avc1" | b"avc3" => {
            r.skip(16)?;
            t.width = r.u16()? as u32;
            t.height = r.u16()? as u32;
            r.skip(50)?;
            let hevc = k[0] == b'h';
            let Some(c) = kid(r.0, if hevc { b"hvcC" } else { b"avcC" }) else {
                return Ok(());
            };
            let mut r = Rd(c);
            let mut ps = vec![];
            t.codec = if hevc {
                r.skip(21)?;
                let nal = (r.u8()? & 3) + 1;
                for _ in 0..r.u8()? {
                    r.u8()?;
                    for _ in 0..r.u16()? {
                        let n = r.u16()? as usize;
                        ps.push(r.take(n)?.to_vec());
                    }
                }
                Codec::Hevc { ps, nal }
            } else {
                r.skip(4)?;
                let nal = (r.u8()? & 3) + 1;
                for i in 0..2 {
                    for _ in 0..r.u8()? & if i == 0 { 0x1f } else { 0xff } {
                        let n = r.u16()? as usize;
                        ps.push(r.take(n)?.to_vec());
                    }
                }
                Codec::Avc { ps, nal }
            };
        }
        b"mp4a" => {
            let v = r.u16()?;
            r.skip(18 + [0, 16, 36].get(v as usize).copied().unwrap_or(0))?;
            // QuickTime nests esds in `wave`
            let es = kid(r.0, b"esds").or_else(|| kid(r.0, b"wave").and_then(|w| kid(w, b"esds")));
            if let Some(asc) = es.and_then(|e| esds(e).ok()) {
                let (rate, ch) = asc_info(&asc)?;
                t.codec = Codec::Aac { asc, rate, ch };
            }
        }
        _ => {}
    }
    Ok(())
}

/// DecoderSpecificInfo of an AAC `esds` (object type 0x40, 0x66–0x68).
fn esds(b: &[u8]) -> io::Result<Vec<u8>> {
    let mut r = Rd(b);
    r.skip(4)?;
    let desc = |r: &mut Rd| -> io::Result<(u8, usize)> {
        let t = r.u8()?;
        let mut n = 0;
        for _ in 0..4 {
            let c = r.u8()?;
            n = n << 7 | (c & 0x7f) as usize;
            if c < 0x80 {
                break;
            }
        }
        Ok((t, n))
    };
    if desc(&mut r)?.0 != 3 {
        return Err(bad("no ES_Descriptor"));
    }
    r.skip(2)?;
    let fl = r.u8()?;
    if fl & 0x80 != 0 {
        r.skip(2)?;
    }
    if fl & 0x40 != 0 {
        let n = r.u8()? as usize;
        r.skip(n)?;
    }
    if fl & 0x20 != 0 {
        r.skip(2)?;
    }
    if desc(&mut r)?.0 != 4 || !matches!(r.u8()?, 0x40 | 0x66..=0x68) {
        return Err(bad("not AAC"));
    }
    r.skip(12)?;
    let (t, n) = desc(&mut r)?;
    if t != 5 {
        return Err(bad("no DecoderSpecificInfo"));
    }
    Ok(r.take(n)?.to_vec())
}

/// Sample rate and channel count from an AudioSpecificConfig.
pub fn asc_info(asc: &[u8]) -> io::Result<(u32, u16)> {
    const RATES: [u32; 13] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
    ];
    let mut b = Bits { b: asc, p: 0 };
    if b.u(5)? == 31 {
        b.u(6)?;
    }
    let rate = match b.u(4)? {
        15 => b.u(24)?,
        i => *RATES
            .get(i as usize)
            .ok_or_else(|| bad("bad AAC rate index"))?,
    };
    Ok((rate, b.u(4)? as u16))
}

/// Expands stts/ctts/stss/stsc/stsz|stz2/stco|co64 into one entry per sample.
fn table(stbl: &[u8], flen: u64) -> io::Result<Vec<Sample>> {
    let full = |k: &[u8; 4]| kid(stbl, k).map(|b| Rd(b.get(4..).unwrap_or(&[])));
    let mut s: Vec<Sample>;
    if let Some(mut r) = full(b"stsz") {
        let (one, n) = (r.u32()?, r.u32()? as usize);
        // a constant size has no table to bound the count: its samples must fit in the file
        if one == 0 && n > r.0.len() / 4 || one > 0 && n as u64 > flen / one as u64 {
            return Err(bad("bad stsz"));
        }
        s = vec![
            Sample {
                size: one,
                sync: true,
                ..Default::default()
            };
            n
        ];
        if one == 0 {
            for x in &mut s {
                x.size = r.u32()?;
            }
        }
    } else if let Some(mut r) = full(b"stz2") {
        r.skip(3)?;
        let (bits, n) = (r.u8()?, r.u32()? as usize);
        if !matches!(bits, 4 | 8 | 16) || n > r.0.len() * 8 / bits as usize {
            return Err(bad("bad stz2"));
        }
        s = vec![
            Sample {
                sync: true,
                ..Default::default()
            };
            n
        ];
        let mut b = Bits { b: r.0, p: 0 };
        for x in &mut s {
            x.size = b.u(bits as u32)?;
        }
    } else {
        return Ok(vec![]);
    }
    let n = s.len();

    let (mut i, mut t) = (0, 0i64);
    let mut r = full(b"stts").ok_or_else(|| bad("no stts"))?;
    for _ in 0..r.u32()? {
        let (c, d) = (r.u32()? as usize, r.u32()?);
        for x in s.iter_mut().skip(i).take(c) {
            (x.dts, x.pts, x.dur) = (t, t, d);
            t += d as i64;
        }
        i += c;
    }
    if let Some(mut r) = full(b"ctts") {
        let mut i = 0;
        for _ in 0..r.u32()? {
            // version 0 is unsigned on paper, but writers put negative offsets there too
            let (c, o) = (r.u32()? as usize, r.u32()? as i32 as i64);
            for x in s.iter_mut().skip(i).take(c) {
                x.pts = x.dts + o;
            }
            i += c;
        }
    }
    if let Some(mut r) = full(b"stss") {
        s.iter_mut().for_each(|x| x.sync = false);
        for _ in 0..r.u32()? {
            if let Some(x) = (r.u32()? as usize)
                .checked_sub(1)
                .and_then(|k| s.get_mut(k))
            {
                x.sync = true;
            }
        }
    }

    let (mut r, wide) = match (full(b"stco"), full(b"co64")) {
        (Some(r), _) => (r, false),
        (_, Some(r)) => (r, true),
        _ => return Err(bad("no stco")),
    };
    let nc = r.u32()? as usize;
    let mut sc = full(b"stsc").ok_or_else(|| bad("no stsc"))?;
    let ne = sc.u32()? as usize;
    let mut e = Rd(sc.take(ne * 12)?);
    let mut per = 0;
    let mut next = if ne > 0 {
        e.u32()?.max(1) as usize - 1
    } else {
        usize::MAX
    };
    let mut k = 0;
    for c in 0..nc {
        while c >= next {
            let first = next;
            per = e.u32()? as usize;
            e.u32()?;
            next = if e.0.is_empty() {
                usize::MAX
            } else {
                e.u32()?.max(1) as usize - 1
            };
            if next <= first {
                return Err(bad("bad stsc"));
            }
        }
        let mut off = r.uv(wide)?;
        for x in s.iter_mut().skip(k).take(per) {
            x.off = off;
            off += x.size as u64;
        }
        k += per;
    }
    if k < n {
        s.truncate(k); // chunk table covers fewer samples than stsz: trust the chunks
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{probe, samples, tool};

    #[test]
    fn demux_matches_ffprobe() {
        let Some(d) = samples() else { return };
        for (name, kinds) in [
            ("v635.insv", &[b"hvc1", b"hvc1", b"mp4a"][..]),
            ("v635.lrv", &[b"avc1", b"mp4a"]),
            ("v606.insv", &[b"hvc1", b"hvc1", b"mp4a"]),
            ("v627.insv", &[b"hvc1", b"hvc1", b"mp4a"]),
        ] {
            let p = d.join(name);
            let m = Mp4::open(&p).unwrap();
            let pk = probe(p.to_str().unwrap());
            assert_eq!(m.tracks.len(), kinds.len(), "{name}");
            assert_eq!(pk.len(), kinds.len(), "{name}");
            for ((t, k), ff) in m.tracks.iter().zip(kinds).zip(&pk) {
                assert_eq!(&&t.fourcc, k, "{name}");
                let ours: Vec<_> = t
                    .samples
                    .iter()
                    .map(|s| (s.pts, s.dts, s.dur, s.size, s.off, s.sync))
                    .collect();
                assert_eq!(&ours, ff, "{name} track {}", t.id);
                assert_eq!(
                    t.duration,
                    t.samples.iter().map(|s| s.dur as u64).sum::<u64>(),
                    "{name}"
                );
                match &t.codec {
                    Codec::Hevc { ps, nal } => {
                        assert_eq!(
                            ps.iter().map(|n| n[0] >> 1 & 0x3f).collect::<Vec<_>>(),
                            [32, 33, 34]
                        );
                        assert_eq!(*nal, 4);
                        assert_eq!((t.width, t.height), (3840, 3840));
                    }
                    Codec::Avc { ps, nal } => {
                        assert_eq!(ps.iter().map(|n| n[0] & 0x1f).collect::<Vec<_>>(), [7, 8]);
                        assert_eq!(*nal, 4);
                        assert_eq!((t.width, t.height), (1664, 832));
                    }
                    Codec::Aac { rate, ch, .. } => assert_eq!((*rate, *ch), (48000, 2)),
                    c => panic!("{name}: {c:?}"),
                }
            }
            // stream durations as ffprobe reports them
            let st = tool(
                "ffprobe",
                &[
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=duration_ts",
                    "-of",
                    "csv=p=0",
                    p.to_str().unwrap(),
                ],
            );
            let ffd: Vec<u64> = st
                .lines()
                .map(|l| l.trim_end_matches(',').parse().unwrap())
                .collect();
            assert_eq!(
                m.tracks.iter().map(|t| t.duration).collect::<Vec<_>>(),
                ffd,
                "{name}"
            );
            let u: Vec<_> = m
                .udta
                .iter()
                .map(|a| std::str::from_utf8(&a.kind).unwrap().to_string())
                .collect();
            println!(
                "{name}: brand {} {}, {} / {} / {} samples, udta {u:?}",
                std::str::from_utf8(&m.brand).unwrap(),
                m.duration as f64 / m.timescale as f64,
                m.tracks[0].samples.len(),
                m.tracks.get(1).map_or(0, |t| t.samples.len()),
                m.tracks.get(2).map_or(0, |t| t.samples.len()),
            );
        }
    }

    /// Fragmented files as FFmpeg writes them: samples only in `moof`s, in the `moov` and then in `moof`s, two
    /// tracks per `moof` with B-frames (composition offsets), one `moof` per track; data bases explicit, at the
    /// `moof`, and implied (the second `traf` continues after the first's data).
    #[test]
    fn fragments_match_ffprobe() {
        let d = crate::tests::out().join("frag");
        std::fs::create_dir_all(&d).unwrap();
        let a = ["-f", "lavfi", "-i", "sine=f=440:d=3:r=44100"];
        let av = [
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=160x120:r=25:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=3:r=48000",
            "-c:v",
            "libx264",
            "-bf",
            "2",
            "-g",
            "25",
            "-c:a",
            "aac",
        ];
        for (name, src, fl) in [
            ("a_empty.m4a", &a[..], "frag_keyframe+empty_moov"),
            ("a_moov.m4a", &a, "frag_keyframe"),
            (
                "av_moof.mp4",
                &av,
                "frag_keyframe+empty_moov+default_base_moof",
            ),
            ("av_base.mp4", &av, "frag_keyframe+empty_moov"),
            (
                "av_sep.mp4",
                &av,
                "frag_keyframe+empty_moov+separate_moof+omit_tfhd_offset",
            ),
            (
                "av_impl.mp4",
                &av,
                "frag_keyframe+empty_moov+omit_tfhd_offset",
            ),
        ] {
            let p = d.join(name);
            let mut args = vec!["-v", "error", "-y"];
            args.extend_from_slice(src);
            args.extend_from_slice(&["-movflags", fl, "-frag_duration", "500000"]);
            args.push(p.to_str().unwrap());
            tool("ffmpeg", &args);
            let m = Mp4::open(&p).unwrap();
            let pk = probe(p.to_str().unwrap());
            assert_eq!(m.tracks.len(), pk.len(), "{name}");
            for (t, ff) in m.tracks.iter().zip(&pk) {
                let ours: Vec<_> = t
                    .samples
                    .iter()
                    .map(|s| (s.pts, s.dts, s.dur, s.size, s.off, s.sync))
                    .collect();
                assert_eq!(&ours, ff, "{name} track {}", t.id);
                assert_eq!(
                    t.duration,
                    t.samples.iter().map(|s| s.dur as u64).sum::<u64>(),
                    "{name}"
                );
            }
            println!(
                "{name}: {:?} samples, {:.3} s",
                m.tracks.iter().map(|t| t.samples.len()).collect::<Vec<_>>(),
                m.duration as f64 / m.timescale as f64
            );
        }
    }

    #[test]
    fn read_sample_reuses_buffer() {
        let Some(d) = samples() else { return };
        let m = Mp4::open(d.join("v635.lrv")).unwrap();
        let t = &m.tracks[0];
        let mut b = Vec::with_capacity(t.samples.iter().map(|s| s.size as usize).max().unwrap());
        let cap = b.capacity();
        for s in &t.samples {
            m.read(s, &mut b).unwrap();
            assert_eq!(b.len(), s.size as usize);
            // AVC samples are length-prefixed NALs that exactly fill the sample
            let mut p = 0;
            while p < b.len() {
                p += 4 + u32::from_be_bytes(b[p..p + 4].try_into().unwrap()) as usize;
            }
            assert_eq!(p, b.len());
        }
        assert_eq!(b.capacity(), cap);
    }
}
