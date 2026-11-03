//! Progressive MP4 writer: samples stream into one `mdat`, `moov` goes at the end, or in front with `faststart`
//! (the media data is shifted in place). Video is `hvc1` / `avc1` with `hvcC` / `avcC` built from the parameter-set
//! NALs; audio is `mp4a` / `esds` passthrough. 360 tracks get Spherical Video V2 metadata (`st3d`, `sv3d`).

use crate::{
    Bits, bad,
    mp4::{Codec, asc_info},
};
use std::{
    fs::File,
    io::{self, Read as _, Seek as _, SeekFrom, Write as _},
    path::Path,
};

struct Trk {
    codec: Codec,
    ts: u32,
    w: u32,
    h: u32,
    sph: bool,
    size: Vec<u32>,
    dur: Vec<u32>,
    cts: Vec<i32>,
    /// 1-based sync sample numbers
    sync: Vec<u32>,
    /// (file offset, index of its first sample)
    chunks: Vec<(u64, u32)>,
    /// first dts, min pts, max pts + dur, the end set by `Mux::end`: caller's timeline, track ticks
    dts0: i64,
    pmin: i64,
    pend: i64,
    end: i64,
}

pub struct Mux {
    f: File,
    /// Offset of the 8-byte `free` placeholder in front of the 32-bit `mdat` header.
    mdat: u64,
    pos: u64,
    last: usize,
    tracks: Vec<Trk>,
    udta: Vec<u8>,
    /// Write `co64` even when offsets fit 32 bits (tests).
    pub(crate) co64: bool,
}

impl Mux {
    pub fn create(path: impl AsRef<Path>) -> io::Result<Mux> {
        // read access for the faststart shift
        let mut f = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut h = vec![];
        bx(&mut h, b"ftyp", |b| {
            b.extend_from_slice(b"isom\0\0\x02\0isomiso2mp41")
        });
        let mdat = h.len() as u64;
        // if mdat outgrows 32 bits, `free` + header become one 16-byte 64-bit header at finish
        h.extend_from_slice(b"\0\0\0\x08free\0\0\0\0mdat");
        f.write_all(&h)?;
        let pos = h.len() as u64;
        Ok(Mux {
            f,
            mdat,
            pos,
            last: usize::MAX,
            tracks: vec![],
            udta: vec![],
            co64: false,
        })
    }

    /// Adds a track: `Codec::Hevc` / `Codec::Avc` video of `w`×`h` (`sph` = mono equirectangular 360), or
    /// `Codec::Aac` audio (timescale = sample rate, `w`, `h` = 0). Fails on what would make an unreadable file: other
    /// codecs, parameter sets without a parsable SPS, a bad AudioSpecificConfig, timescale 0.
    pub fn track(
        &mut self,
        codec: Codec,
        timescale: u32,
        w: u32,
        h: u32,
        sph: bool,
    ) -> io::Result<usize> {
        match &codec {
            Codec::Hevc { ps, .. } => hvcc(&mut vec![], ps, 3)?,
            Codec::Avc { ps, .. } => avcc(&mut vec![], ps, 3)?,
            Codec::Aac { asc, .. } => _ = asc_info(asc)?,
            Codec::Other(f) => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("cannot mux {}", String::from_utf8_lossy(f)),
                ));
            }
        }
        if timescale == 0 {
            return Err(bad("zero timescale"));
        }
        let (size, dur, cts, sync, chunks) = Default::default();
        self.tracks.push(Trk {
            codec,
            ts: timescale,
            w,
            h,
            sph,
            size,
            dur,
            cts,
            sync,
            chunks,
            dts0: 0,
            pmin: i64::MAX,
            pend: i64::MIN,
            end: i64::MAX,
        });
        Ok(self.tracks.len() - 1)
    }

    /// Raw child boxes for `moov/udta` (see [`crate::mp4::Mp4::copy_udta`]).
    pub fn udta(&mut self, boxes: &[u8]) {
        self.udta.extend_from_slice(boxes);
    }

    /// Appends one sample (length-prefixed NALs for video, a raw AAC frame for audio). A track's samples come in
    /// decode order; `pts - dts` must fit i32. A track switch starts a new chunk. An empty one (a frame the encoder
    /// dropped) is not written: the previous sample lasts `dur` longer, or the track starts later.
    ///
    /// `dts` / `pts` are in the track's timescale on one timeline shared by all tracks, with any origin: the file
    /// starts at the earliest pts of all tracks, or at 0 when that is negative, since negative pts are not presented
    /// (AAC priming, B-frame stream starts). Edit lists carry the rest: a track that starts later gets an empty edit,
    /// and the media time of its first presented sample (B-frame delay, priming) is skipped.
    pub fn write(
        &mut self,
        t: usize,
        data: &[u8],
        dts: i64,
        pts: i64,
        dur: u32,
        sync: bool,
    ) -> io::Result<()> {
        let k = &mut self.tracks[t];
        if data.is_empty() {
            if let Some(d) = k.dur.last_mut() {
                *d += dur;
                k.pend = k.pend.max(pts + dur as i64);
            }
            return Ok(());
        }
        if self.last != t {
            k.chunks.push((self.pos, k.size.len() as u32));
            self.last = t;
        }
        if sync {
            k.sync.push(k.size.len() as u32 + 1);
        }
        if k.size.is_empty() {
            k.dts0 = dts;
        }
        k.size.push(data.len() as u32);
        k.dur.push(dur);
        k.cts.push((pts - dts) as i32);
        (k.pmin, k.pend) = (k.pmin.min(pts), k.pend.max(pts + dur as i64));
        self.f.write_all(data)?;
        self.pos += data.len() as u64;
        Ok(())
    }

    /// Ends track `t`'s presentation at `pts` (track ticks, `write`'s timeline): its edit and `tkhd` duration stop
    /// there, the samples stay (`mdhd` is their sum), e.g. the AAC frame that runs past the last video frame. An end
    /// after the last sample's changes nothing.
    pub fn end(&mut self, t: usize, pts: i64) {
        self.tracks[t].end = pts;
    }

    pub fn finish(mut self, faststart: bool) -> io::Result<()> {
        let data = self.pos - self.mdat - 16;
        if 8 + data <= u32::MAX as u64 {
            self.at(self.mdat + 8, &(8 + data as u32).to_be_bytes())?;
        } else {
            let mut h = *b"\0\0\0\x01mdat\0\0\0\0\0\0\0\0";
            h[8..].copy_from_slice(&(16 + data).to_be_bytes());
            self.at(self.mdat, &h)?;
        }
        if !faststart {
            let m = self.moov(0);
            return self.at(self.pos, &m);
        }
        // moov's size depends on its chunk offsets (stco vs co64), which depend on its size
        let mut m = self.moov(0);
        while m.len() != self.moov(m.len() as u64).len() {
            m = self.moov(m.len() as u64);
        }
        let m = self.moov(m.len() as u64);
        let n = m.len() as u64;
        let mut buf = vec![0; 1 << 22];
        let mut end = self.pos;
        while end > self.mdat {
            let a = end.saturating_sub(buf.len() as u64).max(self.mdat);
            let b = &mut buf[..(end - a) as usize];
            self.f.seek(SeekFrom::Start(a))?;
            self.f.read_exact(b)?;
            self.at(a + n, b)?;
            end = a;
        }
        self.at(self.mdat, &m)
    }

    fn at(&mut self, off: u64, b: &[u8]) -> io::Result<()> {
        self.f.seek(SeekFrom::Start(off))?;
        self.f.write_all(b)
    }

    fn moov(&self, shift: u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(4096);
        // movie timescale: a multiple of 1000 and every track's, so edits end exactly on a track's tick (an audio
        // track on its sample, not the next ms); their largest if the multiple outgrows 32 bits
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let ts = (self.tracks.iter())
            .try_fold(1000u64, |a, k| {
                Some(a / gcd(a, k.ts as u64) * k.ts as u64).filter(|&l| l <= u32::MAX as u64)
            })
            .unwrap_or_else(|| {
                self.tracks
                    .iter()
                    .map(|k| k.ts as u64)
                    .max()
                    .unwrap_or(1000)
            });
        // first presented time: negative pts are cut, and the media timeline starts at the first dts
        let p0 = |k: &Trk| k.pmin.max(k.dts0).max(0);
        let mv =
            |x: i64, k: &Trk| ((x as u128 * ts as u128 + k.ts as u128 / 2) / k.ts as u128) as u64;
        let t0 = (self.tracks.iter())
            .filter(|k| !k.size.is_empty())
            .map(|k| mv(p0(k), k))
            .min()
            .unwrap_or(0);
        // per track: empty edit (movie ticks), media time of the first presented sample, presented span (movie ticks)
        let ed: Vec<(u64, i64, u64)> = self
            .tracks
            .iter()
            .map(|k| {
                if k.size.is_empty() {
                    return (0, 0, 0);
                }
                let p = p0(k);
                let span = (k.pend.min(k.end) - p).max(0) as u128 * ts as u128;
                (
                    mv(p, k) - t0,
                    p - k.dts0,
                    span.div_ceil(k.ts as u128) as u64,
                )
            })
            .collect();
        let mdur = ed.iter().map(|e| e.0 + e.2).max().unwrap_or(0);
        bx(&mut v, b"moov", |v| {
            fbx(v, b"mvhd", 1, 0, |v| {
                put(v, &[0u64, 0]);
                put(v, &[ts as u32]);
                put(v, &[mdur]);
                put(v, &[0x10000u32]);
                put(v, &[0x100u16, 0]);
                put(v, &[0u32, 0]);
                matrix(v);
                put(v, &[0u32; 6]);
                put(v, &[self.tracks.len() as u32 + 1]);
            });
            for (i, k) in self.tracks.iter().enumerate() {
                let vid = !matches!(k.codec, Codec::Aac { .. });
                let (delay, mt, span) = ed[i];
                bx(v, b"trak", |v| {
                    fbx(v, b"tkhd", 1, 3, |v| {
                        put(v, &[0u64, 0]);
                        put(v, &[i as u32 + 1, 0]);
                        put(v, &[delay + span, 0]);
                        put(v, &[0u16, !vid as u16, if vid { 0 } else { 0x100 }, 0]); // layer, group, volume
                        matrix(v);
                        put(v, &[k.w << 16, k.h << 16]);
                    });
                    // also when pts before the first presented one or after the end must be cut
                    if delay > 0 || mt != 0 || k.pmin < k.dts0 + mt || k.end < k.pend {
                        bx(v, b"edts", |v| {
                            fbx(v, b"elst", 1, 0, |v| {
                                put(v, &[1 + (delay > 0) as u32]);
                                if delay > 0 {
                                    put(v, &[delay, u64::MAX]); // media time −1: empty
                                    put(v, &[0x10000u32]);
                                }
                                put(v, &[span, mt as u64]);
                                put(v, &[0x10000u32]);
                            })
                        });
                    }
                    bx(v, b"mdia", |v| {
                        fbx(v, b"mdhd", 1, 0, |v| {
                            put(v, &[0u64, 0]);
                            put(v, &[k.ts]);
                            put(v, &[k.dur.iter().map(|&d| d as u64).sum::<u64>()]);
                            put(v, &[0x55c4u16, 0]); // "und"
                        });
                        fbx(v, b"hdlr", 0, 0, |v| {
                            put(v, &[0u32]);
                            v.extend_from_slice(if vid { b"vide" } else { b"soun" });
                            put(v, &[0u32; 3]);
                            v.extend_from_slice(if vid {
                                b"VideoHandler\0"
                            } else {
                                b"SoundHandler\0"
                            });
                        });
                        bx(v, b"minf", |v| {
                            if vid {
                                fbx(v, b"vmhd", 0, 1, |v| put(v, &[0u16; 4]));
                            } else {
                                fbx(v, b"smhd", 0, 0, |v| put(v, &[0u32]));
                            }
                            bx(v, b"dinf", |v| {
                                fbx(v, b"dref", 0, 0, |v| {
                                    put(v, &[1u32]);
                                    fbx(v, b"url ", 0, 1, |_| {});
                                })
                            });
                            bx(v, b"stbl", |v| self.stbl(v, k, shift));
                        });
                    });
                });
            }
            if !self.udta.is_empty() {
                bx(v, b"udta", |v| v.extend_from_slice(&self.udta));
            }
        });
        v
    }

    fn stbl(&self, v: &mut Vec<u8>, k: &Trk, shift: u64) {
        fbx(v, b"stsd", 0, 0, |v| {
            put(v, &[1u32]);
            entry(v, k);
        });
        fbx(v, b"stts", 0, 0, |v| runs(v, k.dur.iter().copied()));
        if k.cts.iter().any(|&c| c != 0) {
            fbx(v, b"ctts", k.cts.iter().any(|&c| c < 0) as u8, 0, |v| {
                runs(v, k.cts.iter().map(|&c| c as u32))
            });
        }
        if k.sync.len() < k.size.len() {
            fbx(v, b"stss", 0, 0, |v| {
                put(v, &[k.sync.len() as u32]);
                put(v, &k.sync);
            });
        }
        fbx(v, b"stsc", 0, 0, |v| {
            let at = v.len();
            put(v, &[0u32]);
            let mut prev = 0;
            for (c, w) in k.chunks.iter().enumerate() {
                let n = k.chunks.get(c + 1).map_or(k.size.len() as u32, |x| x.1) - w.1;
                if n != prev {
                    put(v, &[c as u32 + 1, n, 1]);
                    prev = n;
                }
            }
            let n = ((v.len() - at - 4) / 12) as u32;
            v[at..at + 4].copy_from_slice(&n.to_be_bytes());
        });
        fbx(v, b"stsz", 0, 0, |v| {
            let one = k
                .size
                .first()
                .filter(|&&s| k.size.iter().all(|&x| x == s))
                .copied()
                .unwrap_or(0);
            put(v, &[one, k.size.len() as u32]);
            if one == 0 {
                put(v, &k.size);
            }
        });
        let wide = self.co64
            || k.chunks
                .last()
                .is_some_and(|c| c.0 + shift > u32::MAX as u64);
        fbx(v, if wide { b"co64" } else { b"stco" }, 0, 0, |v| {
            put(v, &[k.chunks.len() as u32]);
            for c in &k.chunks {
                if wide {
                    put(v, &[c.0 + shift])
                } else {
                    put(v, &[(c.0 + shift) as u32])
                }
            }
        });
    }
}

/// Run-length (count, value) table with its entry count; extends the last entry in place.
fn runs(v: &mut Vec<u8>, it: impl Iterator<Item = u32>) {
    let at = v.len();
    put(v, &[0u32]);
    for x in it {
        let l = v.len();
        if l > at + 4 && v[l - 4..] == x.to_be_bytes() {
            let c = u32::from_be_bytes(v[l - 8..l - 4].try_into().unwrap()) + 1;
            v[l - 8..l - 4].copy_from_slice(&c.to_be_bytes());
        } else {
            put(v, &[1, x]);
        }
    }
    let n = ((v.len() - at - 4) / 8) as u32;
    v[at..at + 4].copy_from_slice(&n.to_be_bytes());
}

fn entry(v: &mut Vec<u8>, k: &Trk) {
    match &k.codec {
        Codec::Aac { asc, rate, ch } => bx(v, b"mp4a", |v| {
            put(v, &[0u32, 1]);
            put(v, &[0u32, 0]);
            put(v, &[*ch, 16, 0, 0]);
            put(v, &[if *rate < 65536 { rate << 16 } else { 0 }]);
            fbx(v, b"esds", 0, 0, |v| esds(v, k, asc));
        }),
        Codec::Hevc { ps, nal } | Codec::Avc { ps, nal } => {
            let hevc = matches!(k.codec, Codec::Hevc { .. });
            bx(v, if hevc { b"hvc1" } else { b"avc1" }, |v| {
                put(v, &[0u32, 1]);
                put(v, &[0u32; 4]);
                put(v, &[k.w as u16, k.h as u16]);
                put(v, &[0x480000u32, 0x480000, 0]);
                put(v, &[1u16]);
                v.extend_from_slice(&[0; 32]);
                put(v, &[0x18u16, 0xffff]);
                // parameter sets were checked in `track`
                let l = nal.clamp(&1, &4) - 1;
                if hevc {
                    bx(v, b"hvcC", |v| _ = hvcc(v, ps, l));
                } else {
                    bx(v, b"avcC", |v| _ = avcc(v, ps, l));
                }
                if k.sph {
                    fbx(v, b"st3d", 0, 0, |v| v.push(0)); // monoscopic
                    bx(v, b"sv3d", |v| {
                        fbx(v, b"svhd", 0, 0, |v| v.extend_from_slice(b"x5desk\0"));
                        bx(v, b"proj", |v| {
                            fbx(v, b"prhd", 0, 0, |v| put(v, &[0u32; 3])); // yaw, pitch, roll
                            fbx(v, b"equi", 0, 0, |v| put(v, &[0u32; 4])); // no cropping bounds
                        });
                    });
                }
            });
        }
        Codec::Other(_) => {} // rejected in `track`
    }
}

fn esds(v: &mut Vec<u8>, k: &Trk, asc: &[u8]) {
    let (bytes, ticks) = (
        k.size.iter().map(|&s| s as u64).sum::<u64>(),
        k.dur.iter().map(|&d| d as u64).sum::<u64>(),
    );
    let avg = (bytes * 8 * k.ts as u64).checked_div(ticks).unwrap_or(0) as u32;
    // max bits in any one-second window
    let (mut max, mut win, mut t, mut a) = (0u64, 0u64, 0u64, 0);
    for (&s, &d) in k.size.iter().zip(&k.dur) {
        (win, t) = (win + s as u64, t + d as u64);
        while t > k.ts as u64 {
            (win, t, a) = (win - k.size[a] as u64, t - k.dur[a] as u64, a + 1);
        }
        max = max.max(win * 8);
    }
    let desc = |v: &mut Vec<u8>, tag: u8, body: &dyn Fn(&mut Vec<u8>)| {
        v.extend_from_slice(&[tag, 0, 0, 0, 0]);
        let at = v.len();
        body(v);
        // 4-byte expandable size: 7 bits per byte, high bit = more follows
        let n = v.len() - at;
        for (i, b) in v[at - 4..at].iter_mut().enumerate() {
            *b = (n >> (21 - 7 * i)) as u8 & 0x7f | if i < 3 { 0x80 } else { 0 };
        }
    };
    desc(v, 3, &|v| {
        v.extend_from_slice(&[0, 0, 0]); // ES_ID, flags
        desc(v, 4, &|v| {
            v.extend_from_slice(&[0x40, 0x15]); // AAC, audio stream
            v.extend_from_slice(&k.size.iter().max().copied().unwrap_or(0).to_be_bytes()[1..]);
            put(v, &[max.max(avg as u64) as u32, avg]);
            desc(v, 5, &|v| v.extend_from_slice(asc));
        });
        desc(v, 6, &|v| v.push(2));
    });
}

/// RBSP of a NAL payload: drops emulation-prevention bytes (00 00 03).
fn rbsp(b: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(b.len());
    let mut z = 0;
    for &c in b {
        if z >= 2 && c == 3 {
            z = 0;
            continue;
        }
        z = if c == 0 { z + 1 } else { 0 };
        v.push(c);
    }
    v
}

fn hvcc(v: &mut Vec<u8>, ps: &[Vec<u8>], l: u8) -> io::Result<()> {
    let ty = |n: &Vec<u8>| n.first().map_or(0xff, |c| c >> 1 & 0x3f);
    let sps = ps
        .iter()
        .find(|n| n.len() > 2 && ty(n) == 33)
        .ok_or_else(|| bad("no SPS"))?;
    let r = rbsp(&sps[2..]);
    let mut b = Bits { b: &r, p: 0 };
    b.u(4)?;
    let (sub, nested) = (b.u(3)? as usize, b.u(1)?);
    let ptl = r.get(1..13).ok_or_else(|| bad("short SPS"))?;
    b.p += 96;
    let mut pres = [0; 8];
    for p in &mut pres[..sub] {
        *p = b.u(2)?;
    }
    if sub > 0 {
        b.p += 2 * (8 - sub);
    }
    for p in &pres[..sub] {
        b.p += if p & 2 != 0 { 88 } else { 0 } + if p & 1 != 0 { 8 } else { 0 };
    }
    b.ue()?;
    let chroma = b.ue()?;
    if chroma == 3 {
        b.u(1)?;
    }
    b.ue()?;
    b.ue()?;
    if b.u(1)? == 1 {
        for _ in 0..4 {
            b.ue()?;
        }
    }
    let (luma, cdep) = (b.ue()?, b.ue()?);
    v.push(1);
    v.extend_from_slice(ptl);
    v.extend_from_slice(&[
        0xf0,
        0,
        0xfc,
        0xfc | chroma as u8,
        0xf8 | luma as u8,
        0xf8 | cdep as u8,
        0,
        0,
    ]);
    v.push(((sub as u32 + 1) << 3 | nested << 2 | l as u32) as u8);
    let mut types = vec![];
    for n in ps.iter().filter(|n| !n.is_empty()) {
        if !types.contains(&ty(n)) {
            types.push(ty(n));
        }
    }
    v.push(types.len() as u8);
    for t in types {
        v.push(0x80 | t);
        put(v, &[ps.iter().filter(|n| ty(n) == t).count() as u16]);
        for n in ps.iter().filter(|n| ty(n) == t) {
            put(v, &[n.len() as u16]);
            v.extend_from_slice(n);
        }
    }
    Ok(())
}

fn avcc(v: &mut Vec<u8>, ps: &[Vec<u8>], l: u8) -> io::Result<()> {
    let of = |t| {
        ps.iter()
            .filter(move |n| n.first().is_some_and(|c| c & 0x1f == t))
    };
    let s = of(7)
        .next()
        .filter(|s| s.len() > 4)
        .ok_or_else(|| bad("no SPS"))?;
    v.extend_from_slice(&[1, s[1], s[2], s[3], 0xfc | l, 0xe0 | of(7).count() as u8]);
    for n in of(7) {
        put(v, &[n.len() as u16]);
        v.extend_from_slice(n);
    }
    v.push(of(8).count() as u8);
    for n in of(8) {
        put(v, &[n.len() as u16]);
        v.extend_from_slice(n);
    }
    if matches!(s[1], 100 | 110 | 122 | 144) {
        let r = rbsp(&s[4..]);
        let mut b = Bits { b: &r, p: 0 };
        b.ue()?;
        let chroma = b.ue()?;
        if chroma == 3 {
            b.u(1)?;
        }
        let (luma, cdep) = (b.ue()?, b.ue()?);
        v.extend_from_slice(&[0xfc | chroma as u8, 0xf8 | luma as u8, 0xf8 | cdep as u8, 0]);
    }
    Ok(())
}

fn bx(v: &mut Vec<u8>, k: &[u8; 4], f: impl FnOnce(&mut Vec<u8>)) {
    let at = v.len();
    v.extend_from_slice(&[0; 4]);
    v.extend_from_slice(k);
    f(v);
    let n = (v.len() - at) as u32;
    v[at..at + 4].copy_from_slice(&n.to_be_bytes());
}

fn fbx(v: &mut Vec<u8>, k: &[u8; 4], ver: u8, flags: u32, f: impl FnOnce(&mut Vec<u8>)) {
    bx(v, k, |v| {
        put(v, &[(ver as u32) << 24 | flags]);
        f(v)
    })
}

fn matrix(v: &mut Vec<u8>) {
    put(v, &[0x10000u32, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000]);
}

trait Be: Copy {
    fn be(self, v: &mut Vec<u8>);
}

macro_rules! be {
    ($($t:ty),*) => {$(impl Be for $t {
        fn be(self, v: &mut Vec<u8>) {
            v.extend_from_slice(&self.to_be_bytes())
        }
    })*};
}
be!(u16, u32, u64);

fn put<T: Be>(v: &mut Vec<u8>, xs: &[T]) {
    xs.iter().for_each(|x| x.be(v));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mp4::Mp4,
        tests::{fixtures, probe, samples, tool},
    };
    use std::path::Path;

    /// Copies `tracks` of `src` into `dst`, interleaved by decode time, with the source's edit lists applied to the
    /// timestamps; `sph` marks video as 360.
    fn remux(
        src: &Path,
        tracks: &[usize],
        dst: &Path,
        faststart: bool,
        sph: bool,
        co64: bool,
    ) -> Mp4 {
        let m = Mp4::open(src).unwrap();
        let mut x = Mux::create(dst).unwrap();
        x.co64 = co64;
        for &t in tracks {
            let k = &m.tracks[t];
            x.track(
                k.codec.clone(),
                k.timescale,
                k.width,
                k.height,
                sph && k.width > 0,
            )
            .unwrap();
        }
        let sh: Vec<_> = tracks
            .iter()
            .map(|&t| m.tracks[t].shift(m.timescale))
            .collect();
        x.udta(&m.copy_udta(|a| &a.kind == b"AMBA").unwrap());
        let mut next = vec![0; tracks.len()];
        let mut buf = vec![];
        loop {
            let secs = |(j, &t): (usize, &usize)| {
                let k = &m.tracks[t];
                k.samples
                    .get(next[j])
                    .map(|s| (s.dts + sh[j]) as f64 / k.timescale as f64)
            };
            let Some(j) = tracks
                .iter()
                .enumerate()
                .filter(|x| secs(*x).is_some())
                .min_by(|a, b| secs(*a).partial_cmp(&secs(*b)).unwrap())
                .map(|x| x.0)
            else {
                break;
            };
            let s = m.tracks[tracks[j]].samples[next[j]];
            m.read(&s, &mut buf).unwrap();
            x.write(j, &buf, s.dts + sh[j], s.pts + sh[j], s.dur, s.sync)
                .unwrap();
            next[j] += 1;
        }
        x.finish(faststart).unwrap();
        m
    }

    fn hashes(p: &Path) -> Vec<Vec<String>> {
        let out = tool(
            "ffprobe",
            &[
                "-v",
                "error",
                "-show_data_hash",
                "MD5",
                "-show_entries",
                "packet=stream_index,pts,dts,duration,size,flags,data_hash",
                "-of",
                "csv=p=0",
                p.to_str().unwrap(),
            ],
        );
        let mut v: Vec<Vec<String>> = vec![];
        for l in out.lines() {
            let i: usize = l.split(',').next().unwrap().parse().unwrap();
            v.resize_with(v.len().max(i + 1), Vec::new);
            v[i].push(l.to_string());
        }
        v
    }

    fn decodes_clean(p: &Path) {
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i", p.to_str().unwrap(), "-f", "null", "-"])
            .output()
            .unwrap();
        assert!(
            o.status.success() && o.stderr.is_empty(),
            "ffmpeg: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }

    fn top(p: &Path) -> Vec<String> {
        let m = std::fs::File::open(p).unwrap();
        crate::mp4::atoms(&m, 0, m.metadata().unwrap().len())
            .unwrap()
            .iter()
            .map(|a| String::from_utf8_lossy(&a.kind).into_owned())
            .collect()
    }

    #[test]
    fn remux_lrv_packets_identical() {
        let Some(d) = samples() else { return };
        let src = d.join("v635.lrv");
        let want = hashes(&src);
        for (name, fs, co64) in [
            ("lrv.mp4", false, false),
            ("lrv_fast.mp4", true, false),
            ("lrv_co64.mp4", true, true),
        ] {
            let dst = fixtures().join(name);
            let m = remux(&src, &[0, 1], &dst, fs, false, co64);
            assert_eq!(hashes(&dst), want, "{name}");
            decodes_clean(&dst);
            assert_eq!(
                top(&dst),
                if fs {
                    ["ftyp", "moov", "free", "mdat"]
                } else {
                    ["ftyp", "free", "mdat", "moov"]
                },
                "{name}"
            );
            // our own demuxer reads it back: same samples apart from file offsets, same codec configs
            let back = Mp4::open(&dst).unwrap();
            for (a, b) in m.tracks.iter().zip(&back.tracks) {
                let strip = |t: &crate::mp4::Track| {
                    t.samples
                        .iter()
                        .map(|s| (s.pts, s.dts, s.dur, s.size, s.sync))
                        .collect::<Vec<_>>()
                };
                assert_eq!(strip(a), strip(b), "{name}");
                assert_eq!(a.codec, b.codec, "{name}");
                assert_eq!(a.duration, b.duration, "{name}");
            }
            assert_eq!(
                back.udta.iter().map(|a| a.kind).collect::<Vec<_>>(),
                [*b"AMBA"]
            );
            let st = tool(
                "ffprobe",
                &[
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=duration_ts,duration:format=duration",
                    "-of",
                    "csv=p=0",
                    dst.to_str().unwrap(),
                ],
            );
            println!("{name}: {}", st.replace('\n', " | "));
        }
    }

    /// ffprobe's stream start times and durations, s.
    fn spans(p: &Path) -> Vec<(f64, f64)> {
        tool(
            "ffprobe",
            &[
                "-v",
                "error",
                "-show_entries",
                "stream=start_time,duration",
                "-of",
                "csv=p=0",
                p.to_str().unwrap(),
            ],
        )
        .lines()
        .map(|l| {
            let f: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            (f[0], f[1])
        })
        .collect()
    }

    /// Audio frames FFmpeg decodes from the first audio stream (stereo f32).
    fn decoded_frames(p: &Path) -> usize {
        let f = p.with_extension("f32");
        tool(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-i",
                p.to_str().unwrap(),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                f.to_str().unwrap(),
            ],
        );
        std::fs::metadata(&f).unwrap().len() as usize / 8
    }

    /// Trimmed copies that keep the source timestamps: the file starts at the earliest copied pts, nothing is lost,
    /// and the tracks keep their offset (an empty edit on the later one).
    #[test]
    fn trim_keeps_timestamps() {
        let Some(d) = samples() else { return };
        let src = d.join("v635.lrv");
        let m = Mp4::open(&src).unwrap();
        let (v, a) = (&m.tracks[0], &m.tracks[1]);
        // audio alone from packet 47 on
        let dst = fixtures().join("trim47.mp4");
        let mut x = Mux::create(&dst).unwrap();
        x.track(a.codec.clone(), a.timescale, 0, 0, false).unwrap();
        let mut buf = vec![];
        for s in &a.samples[47..] {
            m.read(s, &mut buf).unwrap();
            x.write(0, &buf, s.dts, s.pts, s.dur, s.sync).unwrap();
        }
        x.finish(true).unwrap();
        let want: usize = a.samples[47..].iter().map(|s| s.dur as usize).sum();
        let (got, sp) = (decoded_frames(&dst), spans(&dst));
        println!("trim47: {got} / {want} frames, start {:?}", sp[0]);
        assert_eq!(got, want);
        assert_eq!(sp[0].0, 0.0);
        // video from the second keyframe (1.001 s), audio from the packet holding that time (0.981 s)
        let kv = 1 + v.samples[1..].iter().position(|s| s.sync).unwrap();
        let ka = a.at(v.samples[kv].pts * a.timescale as i64 / v.timescale as i64);
        let (tv, ta) = (
            v.samples[kv].pts as f64 / v.timescale as f64,
            a.samples[ka].pts as f64 / a.timescale as f64,
        );
        let dst = fixtures().join("trim_av.mp4");
        let mut x = Mux::create(&dst).unwrap();
        x.track(v.codec.clone(), v.timescale, v.width, v.height, false)
            .unwrap();
        x.track(a.codec.clone(), a.timescale, 0, 0, false).unwrap();
        let (mut i, mut j) = (kv, ka);
        while i < v.samples.len() || j < a.samples.len() {
            let sv = v.samples.get(i).map(|s| s.dts * a.timescale as i64);
            let sa = a.samples.get(j).map(|s| s.dts * v.timescale as i64);
            let (t, s) = if sa.is_none() || sv.is_some_and(|x| x <= sa.unwrap()) {
                i += 1;
                (0, v.samples[i - 1])
            } else {
                j += 1;
                (1, a.samples[j - 1])
            };
            m.read(&s, &mut buf).unwrap();
            x.write(t, &buf, s.dts, s.pts, s.dur, s.sync).unwrap();
        }
        x.finish(false).unwrap();
        let (pv, pa) = (hashes(&src), hashes(&dst));
        assert_eq!(pa[0].len(), v.samples.len() - kv);
        assert_eq!(pa[1].len(), a.samples.len() - ka);
        // same data, flags and durations in the same order
        let body = |l: &String| {
            let f: Vec<_> = l.split(',').collect();
            [f[3], f[4], f[5], f[6]].join(",")
        };
        for (s, (o, k)) in [(&pv[0], (&pa[0], kv)), (&pv[1], (&pa[1], ka))] {
            assert!(o.iter().zip(&s[k..]).all(|(a, b)| body(a) == body(b)));
        }
        let want: usize = a.samples[ka..].iter().map(|s| s.dur as usize).sum();
        let (got, sp) = (decoded_frames(&dst), spans(&dst));
        println!(
            "trim_av: video {tv:.6} s, audio {ta:.6} s → starts {:?}, audio {got} / {want} frames",
            sp
        );
        assert_eq!(sp[1].0, 0.0);
        assert!((sp[0].0 - (tv - ta)).abs() <= 0.001, "{sp:?}");
        assert_eq!(got, want);
        decodes_clean(&dst);
    }

    /// `Mux::end` cuts the audio that runs past the last video frame: v635.lrv's first 121 frames (4.0374 s, no whole
    /// ms) with all of its audio (5.717 s, AAC in whole 1024-sample frames), the audio ended at the video's end. With
    /// the cut the stream durations ffprobe reports agree within an audio sample (the movie timescale is a multiple of
    /// both tracks'), and FFmpeg's decode, which applies edit lists as QuickTime does, ends there too; all packets stay
    /// in the file.
    #[test]
    fn end_cuts_tail() {
        let Some(d) = samples() else { return };
        let m = Mp4::open(d.join("v635.lrv")).unwrap();
        let (v, a) = (&m.tracks[0], &m.tracks[1]);
        let (sv, sa) = (v.shift(m.timescale), a.shift(m.timescale));
        let vs = &v.samples[..121];
        let ve = vs.iter().map(|s| s.pts + s.dur as i64).max().unwrap() + sv;
        let ts = a.timescale as f64;
        let ae = ve * a.timescale as i64 / v.timescale as i64;
        let mut res = vec![];
        for cut in [false, true] {
            let dst = fixtures().join(format!("end_cut{}.mp4", cut as u8));
            let mut x = Mux::create(&dst).unwrap();
            x.track(v.codec.clone(), v.timescale, v.width, v.height, false)
                .unwrap();
            x.track(a.codec.clone(), a.timescale, 0, 0, false).unwrap();
            let (mut i, mut j, mut buf) = (0, 0, vec![]);
            while i < vs.len() || j < a.samples.len() {
                let tv = vs.get(i).map(|s| (s.dts + sv) * a.timescale as i64);
                let ta = a.samples.get(j).map(|s| (s.dts + sa) * v.timescale as i64);
                let (t, s, sh) = if ta.is_none() || tv.is_some_and(|x| x <= ta.unwrap()) {
                    i += 1;
                    (0, vs[i - 1], sv)
                } else {
                    j += 1;
                    (1, a.samples[j - 1], sa)
                };
                m.read(&s, &mut buf).unwrap();
                x.write(t, &buf, s.dts + sh, s.pts + sh, s.dur, s.sync)
                    .unwrap();
            }
            if cut {
                x.end(1, ae);
            }
            x.finish(true).unwrap();
            let (sp, n) = (spans(&dst), decoded_frames(&dst));
            let gap = sp[1].0 + sp[1].1 - sp[0].0 - sp[0].1;
            println!(
                "cut {cut}: video {:?} s, audio {:?} s (start, duration): audio ends {:+.4} ms after the video; FFmpeg \
                 decodes {n} audio frames ({:+} against the video's end)",
                sp[0],
                sp[1],
                gap * 1e3,
                n as i64 - ae
            );
            // the samples past the end stay in the file (FFmpeg's demuxer drops those wholly outside the edit)
            assert_eq!(
                Mp4::open(&dst).unwrap().tracks[1].samples.len(),
                a.samples.len()
            );
            decodes_clean(&dst);
            res.push((gap, n as i64 - ae));
        }
        assert!(res[0].0 > 1.0, "the uncut audio runs on");
        assert!(res[1].0.abs() <= 1.0 / ts && res[1].1.abs() <= 1, "{res:?}");
    }

    /// Edit lists survive a remux through `Track::shift`: B-frame delay (the shifted video starts at dts < 0) and
    /// AAC priming (the first audio pts < 0). The source is made by FFmpeg (test tool); like the other tool-checked
    /// tests it runs only with `X5_SAMPLES` set.
    #[test]
    fn remux_edit_lists() {
        if samples().is_none() {
            return;
        }
        let src = fixtures().join("bframes.mp4");
        tool(
            "ffmpeg",
            &[
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=30:duration=3",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=3",
                "-c:v",
                "libx264",
                "-bf",
                "2",
                "-g",
                "30",
                "-c:a",
                "aac",
                src.to_str().unwrap(),
            ],
        );
        let dst = fixtures().join("bframes_remux.mp4");
        let m = remux(&src, &[0, 1], &dst, true, false, false);
        let sh: Vec<_> = m.tracks.iter().map(|t| t.shift(m.timescale)).collect();
        let first: Vec<_> = m
            .tracks
            .iter()
            .zip(&sh)
            .map(|(t, s)| (t.samples[0].dts + s, t.samples[0].pts + s))
            .collect();
        println!("shift {sh:?}, first (dts, pts) {first:?}");
        assert!(first[0].0 < 0 && first[1].1 < 0, "{first:?}");
        assert_eq!(hashes(&dst), hashes(&src));
        let (a, b) = (spans(&src), spans(&dst));
        println!("source {a:?}, remux {b:?}");
        assert_eq!(a, b);
        decodes_clean(&dst);
    }

    /// Codecs and configs that would make an unreadable file are refused; a long AudioSpecificConfig (multi-byte
    /// descriptor sizes) reads back unchanged.
    #[test]
    fn refuses_bad_tracks_long_asc() {
        let dst = fixtures().join("long_asc.mp4");
        let mut x = Mux::create(&dst).unwrap();
        assert!(x.track(Codec::Other(*b"mp4v"), 90000, 8, 8, false).is_err());
        let junk = Codec::Hevc {
            ps: vec![vec![0x42, 1, 0xff]],
            nal: 4,
        };
        assert!(x.track(junk, 90000, 8, 8, false).is_err());
        assert!(
            x.track(Codec::Avc { ps: vec![], nal: 4 }, 90000, 8, 8, false)
                .is_err()
        );
        let mut asc = vec![0x11, 0x90]; // AAC-LC, 48 kHz, stereo
        asc.resize(300, 0x5a);
        let c = Codec::Aac {
            asc,
            rate: 48000,
            ch: 2,
        };
        assert_eq!(x.track(c.clone(), 48000, 0, 0, false).unwrap(), 0);
        x.write(0, &[0x21, 0x10, 0x04], 0, 0, 1024, true).unwrap();
        x.finish(false).unwrap();
        assert_eq!(Mp4::open(&dst).unwrap().tracks[0].codec, c);
    }

    #[test]
    fn hevc_spherical() {
        let Some(d) = samples() else { return };
        let src = d.join("v635.insv");
        let dst = fixtures().join("lens0_360.mp4");
        let m = remux(&src, &[0, 2], &dst, true, true, false);
        let sd = tool(
            "ffprobe",
            &[
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_tag_string,width,height,duration_ts,nb_frames:stream_side_data",
                "-of",
                "default",
                dst.to_str().unwrap(),
            ],
        );
        println!("{sd}");
        assert!(
            sd.contains("side_data_type=Spherical Mapping")
                && sd.contains("projection=equirectangular"),
            "{sd}"
        );
        assert!(
            sd.contains("side_data_type=Stereo 3D") && sd.contains("type=2D"),
            "{sd}"
        );
        assert!(
            sd.contains("codec_tag_string=hvc1")
                && sd.contains("nb_frames=175")
                && sd.contains("duration_ts=175175"),
            "{sd}"
        );
        let (a, b) = (probe(src.to_str().unwrap()), probe(dst.to_str().unwrap()));
        let strip = |v: &Vec<(i64, i64, u32, u32, u64, bool)>| {
            v.iter()
                .map(|x| (x.0, x.1, x.2, x.3, x.5))
                .collect::<Vec<_>>()
        };
        assert_eq!(strip(&a[0]), strip(&b[0]));
        assert_eq!(strip(&a[2]), strip(&b[1]));
        decodes_clean(&dst);
        let back = Mp4::open(&dst).unwrap();
        assert_eq!(back.tracks[0].codec, m.tracks[0].codec);
    }

    /// Empty samples (frames the encoder dropped) are not written: the one before lasts longer, a track that starts
    /// with one starts at the next.
    #[test]
    fn empty_samples() {
        let dst = fixtures().join("empty_samples.mp4");
        let mut x = Mux::create(&dst).unwrap();
        let aac = Codec::Aac {
            asc: vec![0x11, 0x90],
            rate: 48000,
            ch: 2,
        };
        x.track(aac, 48000, 0, 0, false).unwrap();
        for (i, d) in [&b""[..], b"a", b"", b"", b"bc"].into_iter().enumerate() {
            let t = 1024 * i as i64;
            x.write(0, d, t, t, 1024, true).unwrap();
        }
        x.finish(false).unwrap();
        let s = &Mp4::open(&dst).unwrap().tracks[0].samples;
        let got: Vec<_> = s.iter().map(|s| (s.dts, s.dur, s.size)).collect();
        assert_eq!(got, [(0, 3072, 1), (3072, 1024, 2)]);
    }
}
