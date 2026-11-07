//! A synthetic X5 recording for tests, CPU only (feature `synth`): both lenses as H.264 High tracks of
//! `video::check`'s I_PCM pictures in one file, an IDR every `GOP` frames with skipped P pictures between, at the
//! X5's 29.97 fps, the `moov` last, then a V3 trailer with a synthetic calibration as the X5 writes it (no IMU, no
//! audio). Every picture decodes exactly on a decoder that takes High; about 1 MB per lens for 300 frames at 256².

use media::{insv, mp4::Codec, mux::Mux};
use std::{fs::OpenOptions, io, io::Write, path::Path};
use video::check;

/// An IDR every this many frames.
pub const GOP: u32 = 30;
/// Timescale, frame duration.
pub const FPS: [u32; 2] = [30000, 1001];

/// X5-like lens calibration (stitch's tests' own; lens type 113), as trailer field #111 holds it.
const CAL: &str = "2_2.000000_4270.000_4271.000_2690.000_2685.000_-0.500_0.100_90.000_0.000000_0.000000_0.000000_0.20000000_1.50000000_-0.80000000_-3.50000000_0.00000000_-0.00250000_-0.00150000_0.00500000_0.01000000_-0.00200000_-0.00250000_0.00020000_-0.02500000_10752_5376_113_2.000000_4272.000_4273.000_8070.000_2680.000_0.500_-0.100_90.500_0.000500_0.000050_-0.031500_0.24000000_1.45000000_-0.90000000_-3.20000000_0.00000000_-0.00100000_-0.00010000_0.01300000_-0.00300000_-0.00700000_0.00200000_-0.01200000_0.01700000_10752_5376_113_394240";

/// `frames` frames of `side` × `side` per lens (a multiple of 16) at `p`. Each GOP is one grey per lens.
pub fn insv(p: &Path, frames: u32, side: u32) -> io::Result<()> {
    write(p, frames, side, None)
}

/// `insv` with frame `k` of both lenses broken: VideoToolbox conceals it and the P pictures after it up to the next
/// IDR (`Decoder::failed`); Vulkan Video fails the decode instead.
pub fn damaged(p: &Path, frames: u32, side: u32, k: u32) -> io::Result<()> {
    write(p, frames, side, Some(k))
}

/// The luma of GOP `g` on lens `l`, within video range.
fn luma(g: u32, l: u32) -> u8 {
    (16 + (g * 40 + l * 100) % 220) as u8
}

fn write(p: &Path, frames: u32, side: u32, bad: Option<u32>) -> io::Result<()> {
    let ps = check::sets(side, true);
    let mut m = Mux::create(p)?;
    let mut t = [0; 2];
    for t in &mut t {
        *t = m.track(
            Codec::Avc {
                ps: ps.clone(),
                nal: 4,
            },
            FPS[0],
            side,
            side,
            false,
        )?;
    }
    let mut b = vec![];
    for k in 0..frames {
        let (g, i) = (k / GOP, k % GOP);
        for (l, &t) in t.iter().enumerate() {
            // frame_num has 4 bits: it wraps within a GOP
            let nal = match () {
                _ if bad == Some(k) => check::broken(g % 2),
                _ if i == 0 => check::flat(side, g % 2, luma(g, l as u32)),
                _ => check::skip(side, i % 16),
            };
            b.clear();
            b.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            b.extend_from_slice(&nal);
            let ts = k as i64 * FPS[1] as i64;
            m.write(t, &b, ts, ts, FPS[1], i == 0)?;
        }
    }
    m.finish(false)?;
    let mut f = OpenOptions::new().append(true).open(p)?;
    f.write_all(&trailer(frames))
}

/// The trailer: `ExtraMetadata` (camera, calibration, dual track, dual fisheye, H.264, rate, frames), its record
/// header, the V3 footer.
fn trailer(frames: u32) -> Vec<u8> {
    fn varint(b: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            b.push(v as u8 | 0x80);
            v >>= 7;
        }
        b.push(v as u8);
    }
    let int = |b: &mut Vec<u8>, tag: u64, v: u64| {
        varint(b, tag << 3);
        varint(b, v);
    };
    let bytes = |b: &mut Vec<u8>, tag: u64, v: &[u8]| {
        varint(b, tag << 3 | 2);
        varint(b, v.len() as u64);
        b.extend_from_slice(v);
    };
    // the camera first: read as a box header, its bytes give a length past the file's end (`Mp4::open` stops there)
    let mut md = vec![];
    bytes(&mut md, 2, b"Insta360 X5");
    bytes(&mut md, 111, CAL.as_bytes());
    let mut r = vec![];
    int(&mut r, 1, FPS[0] as u64);
    int(&mut r, 2, FPS[1] as u64);
    bytes(&mut md, 78, &r);
    int(&mut md, 40, frames as u64);
    int(&mut md, 129, 2);
    int(&mut md, 131, 3);
    int(&mut md, 132, 1);
    let n = md.len() as u32;
    md.extend_from_slice(&[0, insv::METADATA]);
    md.extend_from_slice(&n.to_le_bytes());
    let mut foot = [0u8; 72];
    foot[32..36].copy_from_slice(&(n + 6 + 72).to_le_bytes());
    foot[36..40].copy_from_slice(&3u32.to_le_bytes());
    foot[40..].copy_from_slice(b"8db42d694ccc418790edff439fe026bf");
    md.extend_from_slice(&foot);
    md
}
