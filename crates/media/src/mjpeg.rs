//! MJPEG (the X5's Webcam Mode): splitting a byte stream into complete JPEGs, decode to NV12 or RGBA with
//! zune-jpeg. X5 frames are baseline 4:2:0, full range, BT.601 (JFIF), so the NV12 output is full-range BT.601.

use std::io;
use zune_jpeg::{
    JpegDecoder,
    zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions},
};

/// Complete JPEGs (SOI … EOI) in a byte stream such as a raw capture dump; a frame cut off before its EOI (the
/// next SOI or the end of `b` comes first) is skipped.
pub fn frames(b: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut p = 0;
    std::iter::from_fn(move || {
        loop {
            let s = p + b
                .get(p..)?
                .windows(3)
                .position(|w| w == [0xff, 0xd8, 0xff])?;
            match end(&b[s..]) {
                Ok(n) => {
                    p = s + n;
                    return Some(&b[s..p]);
                }
                Err(q) => p = s + q.max(2),
            }
        }
    })
}

/// Length of the JPEG at the start of `b` through its EOI, or Err(offset where it broke off).
pub fn end(b: &[u8]) -> Result<usize, usize> {
    let (mut p, mut scan) = (2, false);
    while p + 1 < b.len() {
        if b[p] != 0xff {
            if !scan {
                return Err(p);
            }
            // entropy-coded data: skip to the next FF
            p += b[p..]
                .iter()
                .position(|&c| c == 0xff)
                .unwrap_or(b.len() - p);
            continue;
        }
        match b[p + 1] {
            0x00 | 0xd0..=0xd7 if scan => p += 2, // stuffing, restart markers
            0xff => p += 1,
            0xd9 if scan => return Ok(p + 2),
            0xd8 | 0xd9 => return Err(p),
            m => {
                let Some(x) = b.get(p + 2..p + 4) else {
                    return Err(b.len());
                };
                p += 2 + u16::from_be_bytes([x[0], x[1]]) as usize;
                scan = m == 0xda;
            }
        }
    }
    Err(b.len())
}

/// Frame size (w, h) from the JPEG's SOF header, without decoding.
pub fn size(b: &[u8]) -> Option<(u32, u32)> {
    let mut p = 2;
    loop {
        let m = b.get(p..p + 9)?;
        match m {
            [0xff, 0xff, ..] => p += 1,
            [0xff, 0xc0..=0xcf, _, _, _, h0, h1, w0, w1] if !matches!(m[1], 0xc4 | 0xc8 | 0xcc) => {
                return Some((
                    u16::from_be_bytes([*w0, *w1]) as u32,
                    u16::from_be_bytes([*h0, *h1]) as u32,
                ));
            }
            [0xff, 0xd9 | 0xda, ..] => return None,
            [0xff, _, l0, l1, ..] => p += 2 + u16::from_be_bytes([*l0, *l1]) as usize,
            _ => return None,
        }
    }
}

/// Decoder with a reusable full-resolution YCbCr scratch buffer for the NV12 path.
#[derive(Default)]
pub struct Mjpeg {
    ycc: Vec<u8>,
}

impl Mjpeg {
    /// Decodes to RGBA8 in `out` (resized to w·h·4); returns (w, h).
    pub fn rgba(&mut self, jpg: &[u8], out: &mut Vec<u8>) -> io::Result<(usize, usize)> {
        decode(jpg, ColorSpace::RGBA, out)
    }

    /// Decodes to NV12 in `out`: a w×h Y plane, then interleaved CbCr at half resolution (row stride 2·⌈w/2⌉).
    /// zune-jpeg upsamples chroma (triangle filter); it is box-averaged back down, which leaves the chroma within a
    /// few levels of the coded one (measured against FFmpeg in the tests).
    pub fn nv12(&mut self, jpg: &[u8], out: &mut Vec<u8>) -> io::Result<(usize, usize)> {
        let (w, h) = decode(jpg, ColorSpace::YCbCr, &mut self.ycc)?;
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        out.resize(w * h + 2 * cw * ch, 0);
        let (y, uv) = out.split_at_mut(w * h);
        for (d, s) in y.chunks_exact_mut(w).zip(self.ycc.chunks_exact(3 * w)) {
            for (o, p) in d.iter_mut().zip(s.as_chunks::<3>().0) {
                *o = p[0];
            }
        }
        for (j, d) in uv.chunks_exact_mut(2 * cw).enumerate() {
            let a = &self.ycc[2 * j * w * 3..][..w * 3];
            let b = &self.ycc[(2 * j + 1).min(h - 1) * w * 3..][..w * 3];
            for (i, o) in d.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let (x0, x1) = (6 * i, (2 * i + 1).min(w - 1) * 3);
                for c in 1..3 {
                    let s =
                        a[x0 + c] as u16 + a[x1 + c] as u16 + b[x0 + c] as u16 + b[x1 + c] as u16;
                    o[c - 1] = ((s + 2) >> 2) as u8;
                }
            }
        }
        Ok((w, h))
    }
}

fn decode(jpg: &[u8], cs: ColorSpace, out: &mut Vec<u8>) -> io::Result<(usize, usize)> {
    let err = |e| io::Error::new(io::ErrorKind::InvalidData, format!("jpeg: {e:?}"));
    // strict: entropy errors are errors (frames spliced from a broken backlog decode "fine" otherwise)
    let o = DecoderOptions::default()
        .set_strict_mode(true)
        .jpeg_set_out_colorspace(cs);
    let mut d = JpegDecoder::new_with_options(ZCursor::new(jpg), o);
    d.decode_headers().map_err(err)?;
    let (w, h) = d.dimensions().unwrap();
    out.resize(d.output_buffer_size().unwrap(), 0);
    d.decode_into(out).map_err(err)?;
    Ok((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fixtures;
    use std::{path::PathBuf, time::Instant};

    fn webcam() -> Option<PathBuf> {
        let d = std::env::var_os("X5_WEBCAM").map(PathBuf::from);
        if d.is_none() {
            eprintln!("X5_WEBCAM not set: skipping");
        }
        d
    }

    fn dims(j: &[u8]) -> (usize, usize) {
        let mut d = JpegDecoder::new(ZCursor::new(j));
        d.decode_headers().unwrap();
        d.dimensions().unwrap()
    }

    /// Every buffer the capture log calls complete (starts with SOI, ends with EOI) is found at its offset; frames
    /// the splitter assembles across the partial backlog buffers are reported separately.
    #[test]
    fn split_captures() {
        let Some(d) = webcam() else { return };
        for name in ["t_2880x1440", "t_1920x1080", "t_single", "front_long"] {
            let b = std::fs::read(d.join(format!("{name}.mjpg"))).unwrap();
            let log = std::fs::read_to_string(d.join(format!("{name}.txt"))).unwrap();
            let mut off = 0;
            let mut complete = vec![];
            for l in log.lines().filter(|l| l.contains("bytesused:")) {
                let n: usize = l
                    .split("bytesused:")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                let f = &b[off..off + n];
                let t = &f[..n - f.iter().rev().take_while(|&&c| c == 0).count()];
                if t.starts_with(&[0xff, 0xd8]) && t.ends_with(&[0xff, 0xd9]) {
                    complete.push((off, t.len()));
                }
                off += n;
            }
            let ours: Vec<_> = frames(&b)
                .map(|f| (f.as_ptr() as usize - b.as_ptr() as usize, f.len()))
                .collect();
            // a "complete" buffer can hold several frames (one t_2880x1440 buffer has two, 2 stray bytes apart):
            // ours must cover it
            for &(o, n) in &complete {
                let inside: Vec<_> = ours.iter().filter(|x| x.0 >= o && x.0 < o + n).collect();
                assert!(
                    inside[0].0 == o && inside.windows(2).all(|w| w[1].0 - (w[0].0 + w[0].1) < 16),
                    "{name} @{o}"
                );
                assert_eq!(inside.last().map(|x| x.0 + x.1), Some(o + n), "{name} @{o}");
            }
            let extra = ours
                .iter()
                .filter(|x| !complete.iter().any(|&(o, n)| x.0 >= o && x.0 < o + n))
                .count();
            let mut m = Mjpeg::default();
            let mut out = vec![];
            let bad = |sel: &dyn Fn(&(usize, usize)) -> bool, m: &mut Mjpeg, out: &mut Vec<u8>| {
                ours.iter()
                    .filter(|x| sel(x))
                    .filter(|x| m.rgba(&b[x.0..x.0 + x.1], out).is_err())
                    .count()
            };
            let inb = |x: &(usize, usize)| complete.iter().any(|&(o, n)| x.0 >= o && x.0 < o + n);
            println!(
                "{name}: strict decode errors: {} of {} inside complete buffers, {} of {extra} outside",
                bad(&inb, &mut m, &mut out),
                ours.len() - extra,
                bad(&|x| !inb(x), &mut m, &mut out)
            );
            let mut sizes = std::collections::BTreeMap::new();
            for f in frames(&b) {
                let (w, h) = dims(f);
                assert_eq!(size(f), Some((w as u32, h as u32)), "{name}");
                *sizes.entry((w, h)).or_insert(0) += 1;
            }
            println!(
                "{name}: {} complete buffers in the log, {} frames split ({extra} outside them), sizes {sizes:?}",
                complete.len(),
                ours.len()
            );
        }
    }

    fn psnr(a: impl Iterator<Item = (u8, u8)>) -> (f64, u8) {
        let (mut se, mut n, mut mx) = (0f64, 0f64, 0u8);
        for (x, y) in a {
            let d = x.abs_diff(y);
            (se, n, mx) = (se + (d as f64).powi(2), n + 1.0, mx.max(d));
        }
        (10.0 * (255f64.powi(2) / (se / n).max(1e-12)).log10(), mx)
    }

    #[test]
    fn decode_matches_ffmpeg() {
        let Some(d) = webcam() else { return };
        let mut m = Mjpeg::default();
        let (mut nv, mut rgba) = (vec![], vec![]);
        for name in ["t_2880x1440", "t_1920x1080", "t_single", "front_long"] {
            let b = std::fs::read(d.join(format!("{name}.mjpg"))).unwrap();
            let fr: Vec<_> = frames(&b).collect();
            // frame 0 of t_* is a stale frame of the previous mode; the backlog of t_single / front_long holds frames
            // assembled across lost data (corrupt, see split_captures), so those two are sampled from the middle on
            let n = fr.len();
            let first = if name.starts_with("t_") && name != "t_single" {
                0
            } else {
                n / 2
            };
            for k in [first, 3 * n / 4, n - 1] {
                let j = fr[k];
                let f = fixtures().join(format!("{name}_{k}.jpg"));
                std::fs::write(&f, j).unwrap();
                let (w, h) = m.nv12(j, &mut nv).unwrap();
                assert_eq!(m.rgba(j, &mut rgba).unwrap(), (w, h));
                let run = |pix: &str| {
                    let o = std::process::Command::new("ffmpeg")
                        .args([
                            "-v",
                            "error",
                            "-i",
                            f.to_str().unwrap(),
                            "-f",
                            "rawvideo",
                            "-pix_fmt",
                            pix,
                            "-",
                        ])
                        .output()
                        .unwrap();
                    assert!(
                        o.status.success() && o.stderr.is_empty(),
                        "{name} {k}: {}",
                        String::from_utf8_lossy(&o.stderr)
                    );
                    o.stdout
                };
                let (yuv, rgb) = (run("yuvj420p"), run("rgba"));
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                assert_eq!(
                    (yuv.len(), rgb.len(), nv.len(), rgba.len()),
                    (
                        w * h + 2 * cw * ch,
                        w * h * 4,
                        w * h + 2 * cw * ch,
                        w * h * 4
                    )
                );
                let y = psnr(
                    nv[..w * h]
                        .iter()
                        .copied()
                        .zip(yuv[..w * h].iter().copied()),
                );
                let uv = &nv[w * h..];
                let cb = psnr(
                    uv.iter()
                        .step_by(2)
                        .copied()
                        .zip(yuv[w * h..][..cw * ch].iter().copied()),
                );
                let cr = psnr(
                    uv.iter()
                        .skip(1)
                        .step_by(2)
                        .copied()
                        .zip(yuv[w * h + cw * ch..].iter().copied()),
                );
                let px = psnr(
                    rgba.iter()
                        .zip(&rgb)
                        .enumerate()
                        .filter(|(i, _)| i % 4 != 3)
                        .map(|(_, (&a, &b))| (a, b)),
                );
                println!(
                    "{name} #{k} {w}×{h}: Y {:.1} dB (max {}), Cb {:.1} dB (max {}), Cr {:.1} dB (max {}), RGBA {:.1} dB (max {})",
                    y.0, y.1, cb.0, cb.1, cr.0, cr.1, px.0, px.1
                );
                assert!(
                    y.0 > 45.0 && cb.0 > 40.0 && cr.0 > 40.0 && px.0 > 35.0,
                    "{name} #{k}"
                );
                assert!(rgba.iter().skip(3).step_by(4).all(|&a| a == 255));
            }
        }
    }

    #[test]
    fn decode_speed() {
        let Some(d) = webcam() else { return };
        let mut m = Mjpeg::default();
        let mut out = vec![];
        for name in ["t_2880x1440", "t_1920x1080"] {
            let b = std::fs::read(d.join(format!("{name}.mjpg"))).unwrap();
            let fr: Vec<_> = frames(&b).skip(10).take(60).collect();
            for (what, f) in [
                (
                    "nv12",
                    Mjpeg::nv12
                        as fn(&mut Mjpeg, &[u8], &mut Vec<u8>) -> io::Result<(usize, usize)>,
                ),
                ("rgba", Mjpeg::rgba),
            ] {
                let t = Instant::now();
                for j in &fr {
                    f(&mut m, j, &mut out).unwrap();
                }
                println!(
                    "{name} {what}: {:.1} ms/frame",
                    t.elapsed().as_secs_f64() * 1e3 / fr.len() as f64
                );
            }
        }
    }

    #[test]
    fn skips_cut_frames() {
        // two fake frames: header segment, SOS, entropy data with stuffing and a restart marker, EOI
        let jpg = |fill: u8| {
            let mut v = vec![0xff, 0xd8, 0xff, 0xe0, 0, 4, 1, 2, 0xff, 0xda, 0, 2];
            v.extend_from_slice(&[fill, 0xff, 0x00, fill, 0xff, 0xd3, fill, 0xff, 0xd9]);
            v
        };
        let (a, b) = (jpg(1), jpg(2));
        let mut s = vec![0; 5];
        s.extend_from_slice(&a[..14]); // cut in the scan: the next SOI interrupts it
        s.extend_from_slice(&b);
        s.extend_from_slice(&[0; 3]);
        s.extend_from_slice(&a);
        s.extend_from_slice(&b[..7]); // cut in the header
        assert_eq!(frames(&s).collect::<Vec<_>>(), [&b[..], &a[..]]);
    }
}
