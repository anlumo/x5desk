//! Minimal ISO-BMFF reader for the tests only (the app's demuxer is crate `media`): video tracks with their
//! `hvcC` / `avcC` parameter sets and sample locations.
use std::{fs::File, io::Read, io::Seek, io::SeekFrom, path::Path};
use video::Codec;

pub struct Track {
    pub codec: Codec,
    pub width: u32,
    pub height: u32,
    pub nal_len: usize,
    pub params: Vec<Vec<u8>>,
    pub samples: Vec<(u64, u32)>, // file offset, size
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes(b[..4].try_into().unwrap())
}
fn be64(b: &[u8]) -> u64 {
    u64::from_be_bytes(b[..8].try_into().unwrap())
}

/// Child boxes of `b`: (type, body).
fn boxes(mut b: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut v = vec![];
    while b.len() >= 8 {
        let (mut n, t) = (be32(b) as usize, b[4..8].try_into().unwrap());
        let h = if n == 1 {
            n = be64(&b[8..]) as usize;
            16
        } else {
            8
        };
        if n == 0 {
            n = b.len();
        }
        if n < h || n > b.len() {
            break;
        }
        v.push((t, &b[h..n]));
        b = &b[n..];
    }
    v
}

fn find<'a>(b: &'a [u8], t: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(b).into_iter().find(|(x, _)| x == t).map(|(_, y)| y)
}

pub fn read(path: &Path) -> std::io::Result<(File, Vec<Track>)> {
    let mut f = File::open(path)?;
    let len = f.seek(SeekFrom::End(0))?;
    let mut pos = 0u64;
    // top level: walk headers only, up to moov (an .insv trailer follows the boxes)
    let moov = loop {
        if pos + 16 > len {
            return Err(std::io::Error::other("no moov"));
        }
        let mut h = [0u8; 16];
        f.seek(SeekFrom::Start(pos))?;
        f.read_exact(&mut h)?;
        let (mut n, mut hl) = (be32(&h) as u64, 8);
        if n == 1 {
            (n, hl) = (be64(&h[8..]), 16);
        }
        if &h[4..8] == b"moov" {
            let mut m = vec![0u8; (n - hl) as usize];
            f.seek(SeekFrom::Start(pos + hl))?;
            f.read_exact(&mut m)?;
            break m;
        }
        if n < 8 {
            return Err(std::io::Error::other("bad box"));
        }
        pos += n;
    };
    let mut tracks = vec![];
    for (t, trak) in boxes(&moov) {
        if &t != b"trak" {
            continue;
        }
        let mdia = find(trak, b"mdia").unwrap();
        if &find(mdia, b"hdlr").unwrap()[8..12] != b"vide" {
            continue;
        }
        let stbl = find(find(mdia, b"minf").unwrap(), b"stbl").unwrap();
        let stsd = find(stbl, b"stsd").unwrap();
        let (fourcc, entry) = boxes(&stsd[8..])[0];
        let (width, height) = (
            u16::from_be_bytes([entry[24], entry[25]]) as u32,
            u16::from_be_bytes([entry[26], entry[27]]) as u32,
        );
        let (nal_len, mut params);
        let codec = if let Some(c) = find(&entry[78..], b"hvcC") {
            (nal_len, params) = ((c[21] & 3) as usize + 1, vec![]);
            let mut p = 23;
            for _ in 0..c[22] {
                let n = u16::from_be_bytes([c[p + 1], c[p + 2]]);
                p += 3;
                for _ in 0..n {
                    let l = u16::from_be_bytes([c[p], c[p + 1]]) as usize;
                    params.push(c[p + 2..p + 2 + l].to_vec());
                    p += 2 + l;
                }
            }
            Codec::H265
        } else if let Some(c) = find(&entry[78..], b"avcC") {
            (nal_len, params) = ((c[4] & 3) as usize + 1, vec![]);
            let mut p = 5;
            for mask in [0x1f, 0xff] {
                let n = c[p] & mask;
                p += 1;
                for _ in 0..n {
                    let l = u16::from_be_bytes([c[p], c[p + 1]]) as usize;
                    params.push(c[p + 2..p + 2 + l].to_vec());
                    p += 2 + l;
                }
            }
            Codec::H264
        } else {
            panic!("unknown codec {}", String::from_utf8_lossy(&fourcc));
        };
        let stsz = find(stbl, b"stsz").unwrap();
        let (fixed, count) = (be32(&stsz[4..]), be32(&stsz[8..]) as usize);
        let sizes: Vec<u32> = (0..count)
            .map(|i| {
                if fixed != 0 {
                    fixed
                } else {
                    be32(&stsz[12 + 4 * i..])
                }
            })
            .collect();
        let chunks: Vec<u64> = if let Some(c) = find(stbl, b"stco") {
            (0..be32(&c[4..]) as usize)
                .map(|i| be32(&c[8 + 4 * i..]) as u64)
                .collect()
        } else {
            let c = find(stbl, b"co64").unwrap();
            (0..be32(&c[4..]) as usize)
                .map(|i| be64(&c[8 + 8 * i..]))
                .collect()
        };
        let stsc = find(stbl, b"stsc").unwrap();
        let sc: Vec<(usize, usize)> = (0..be32(&stsc[4..]) as usize)
            .map(|i| {
                (
                    be32(&stsc[8 + 12 * i..]) as usize,
                    be32(&stsc[12 + 12 * i..]) as usize,
                )
            })
            .collect();
        let mut samples = Vec::with_capacity(count);
        for (ci, &off) in chunks.iter().enumerate() {
            let per = sc
                .iter()
                .rev()
                .find(|(first, _)| *first <= ci + 1)
                .unwrap()
                .1;
            let mut o = off;
            for _ in 0..per {
                if samples.len() == count {
                    break;
                }
                let s = sizes[samples.len()];
                samples.push((o, s));
                o += s as u64;
            }
        }
        tracks.push(Track {
            codec,
            width,
            height,
            nal_len,
            params,
            samples,
        });
    }
    Ok((f, tracks))
}

/// Reads sample `i` of `t` into `buf`.
pub fn sample(f: &mut File, t: &Track, i: usize, buf: &mut Vec<u8>) {
    let (off, size) = t.samples[i];
    buf.resize(size as usize, 0);
    f.seek(SeekFrom::Start(off)).unwrap();
    f.read_exact(buf).unwrap();
}

/// Splits one length-prefixed sample into NAL units.
pub fn nals(mut b: &[u8], n: usize) -> Vec<&[u8]> {
    let mut v = vec![];
    while b.len() > n {
        let l = b[..n]
            .iter()
            .fold(0usize, |a, &x| a << 8 | x as usize)
            .min(b.len() - n);
        v.push(&b[n..n + l]);
        b = &b[n + l..];
    }
    v
}
