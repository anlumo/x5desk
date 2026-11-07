//! Containers and media I/O without FFmpeg: MP4 demux ([`mp4`]) and mux ([`mux`]), the Insta360 trailer
//! ([`insv`]), AAC decode and the audio master clock ([`audio`]), MJPEG decode ([`mjpeg`]), baseline JPEG encode
//! ([`jpeg`]) and UVC capture of the X5's Webcam Mode ([`uvc`], 64-bit Linux). Audio for the editor and export:
//! AAC-LC encoder ([`aacenc`]), music import ([`music`]), resampler ([`resample`]), time-stretch ([`stretch`]), mixer
//! ([`mix`]), speech denoise ([`denoise`]). Output files written beside their name and published without replacing
//! anything ([`part`]); an RGBA area downscaler ([`scale`]).

pub mod aacenc;
pub mod audio;
pub mod denoise;
mod fft;
pub mod insv;
pub mod jpeg;
pub mod mix;
pub mod mjpeg;
pub mod mp4;
pub mod music;
pub mod mux;
pub mod part;
pub mod resample;
pub mod scale;
pub mod stretch;
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub mod uvc;

#[cfg(test)]
mod tests;

use std::{fs::File, io};

fn bad(m: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m.to_string())
}

/// Positional read: no shared cursor, so readers on several threads can share one `File`.
#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match std::os::windows::fs::FileExt::seek_read(f, buf, off)? {
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n => (buf, off) = (&mut buf[n..], off + n as u64),
        }
    }
    Ok(())
}

/// Big-endian cursor over a box payload; every read is bounds-checked (files are untrusted).
#[derive(Clone, Copy)]
struct Rd<'a>(&'a [u8]);

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if n > self.0.len() {
            return Err(bad("truncated box"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn skip(&mut self, n: usize) -> io::Result<()> {
        self.take(n).map(|_| ())
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// 32-bit field in version 0 boxes, 64-bit in version 1.
    fn uv(&mut self, v1: bool) -> io::Result<u64> {
        if v1 {
            self.u64()
        } else {
            self.u32().map(u64::from)
        }
    }
}

/// MSB-first bit reader over an RBSP (emulation prevention removed) or an AudioSpecificConfig.
struct Bits<'a> {
    b: &'a [u8],
    p: usize,
}

impl Bits<'_> {
    fn u(&mut self, n: u32) -> io::Result<u32> {
        let mut v = 0;
        for _ in 0..n {
            let c = *self
                .b
                .get(self.p >> 3)
                .ok_or_else(|| bad("bitstream overrun"))?;
            v = v << 1 | (c >> (7 - (self.p & 7)) & 1) as u32;
            self.p += 1;
        }
        Ok(v)
    }
    fn ue(&mut self) -> io::Result<u32> {
        let mut z = 0;
        while self.u(1)? == 0 {
            z += 1;
            if z > 31 {
                return Err(bad("bad exp-golomb"));
            }
        }
        Ok(((1u64 << z) - 1 + self.u(z)? as u64) as u32)
    }
}
