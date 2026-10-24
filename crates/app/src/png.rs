//! Minimal PNG writer for the screenshot hook: RGBA8, zlib with stored (uncompressed) deflate blocks.

use std::{io, path::Path};

const CRC: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb88320 ^ c >> 1
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

fn crc(b: &[u8]) -> u32 {
    !b.iter().fold(!0u32, |c, &x| {
        CRC[((c ^ x as u32) & 0xff) as usize] ^ c >> 8
    })
}

fn chunk(o: &mut Vec<u8>, kind: &[u8; 4], d: &[u8]) {
    o.extend((d.len() as u32).to_be_bytes());
    let at = o.len();
    o.extend(kind);
    o.extend(d);
    let c = crc(&o[at..]);
    o.extend(c.to_be_bytes());
}

/// `w` × `h` RGBA8 pixels, rows top to bottom.
pub fn encode(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let row = w as usize * 4;
    assert_eq!(rgba.len(), row * h as usize);
    // filter byte 0 (none) before each row
    let mut raw = Vec::with_capacity((row + 1) * h as usize);
    for r in rgba.chunks_exact(row.max(1)).take(h as usize) {
        raw.push(0);
        raw.extend(r);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    let n = raw.len().div_ceil(65535).max(1);
    let mut z = Vec::with_capacity(raw.len() + 5 * n + 6);
    z.extend([0x78, 0x01]);
    for i in 0..n {
        let c = &raw[(i * 65535).min(raw.len())..((i + 1) * 65535).min(raw.len())];
        z.push((i + 1 == n) as u8);
        z.extend((c.len() as u16).to_le_bytes());
        z.extend((!(c.len() as u16)).to_le_bytes());
        z.extend(c);
    }
    z.extend((b << 16 | a).to_be_bytes());
    let mut o = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut hdr = [0u8; 13];
    hdr[..4].copy_from_slice(&w.to_be_bytes());
    hdr[4..8].copy_from_slice(&h.to_be_bytes());
    hdr[8..10].copy_from_slice(&[8, 6]); // 8 bits, RGBA
    chunk(&mut o, b"IHDR", &hdr);
    chunk(&mut o, b"IDAT", &z);
    chunk(&mut o, b"IEND", &[]);
    o
}

pub fn write(path: &Path, w: u32, h: u32, rgba: &[u8]) -> io::Result<()> {
    std::fs::write(path, encode(w, h, rgba))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parses the file back: chunk CRCs, header, stored blocks, Adler-32, pixels.
    #[test]
    fn roundtrip() {
        assert_eq!(crc(b"IEND"), 0xae426082);
        // wide enough for several stored blocks
        let (w, h) = (20000u32, 2u32);
        let px: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 % 251) as u8).collect();
        let f = encode(w, h, &px);
        assert_eq!(&f[..8], b"\x89PNG\r\n\x1a\n");
        let (mut p, mut idat, mut kinds) = (8, vec![], vec![]);
        while p < f.len() {
            let n = u32::from_be_bytes(f[p..p + 4].try_into().unwrap()) as usize;
            let body = &f[p + 4..p + 8 + n];
            let c = u32::from_be_bytes(f[p + 8 + n..p + 12 + n].try_into().unwrap());
            assert_eq!(crc(body), c);
            kinds.push(body[..4].to_vec());
            match &body[..4] {
                b"IHDR" => assert_eq!(&body[4..], &[0, 0, 0x4e, 0x20, 0, 0, 0, 2, 8, 6, 0, 0, 0]),
                b"IDAT" => idat.extend(&body[4..]),
                _ => {}
            }
            p += 12 + n;
        }
        assert_eq!(kinds, [b"IHDR", b"IDAT", b"IEND"]);
        let (mut q, mut raw) = (2, Vec::<u8>::new());
        loop {
            let last = idat[q] & 1;
            let n = u16::from_le_bytes([idat[q + 1], idat[q + 2]]) as usize;
            assert_eq!(n as u16, !u16::from_le_bytes([idat[q + 3], idat[q + 4]]));
            raw.extend(&idat[q + 5..q + 5 + n]);
            q += 5 + n;
            if last == 1 {
                break;
            }
        }
        let (mut a, mut b) = (1u32, 0u32);
        for &x in &raw {
            (a, b) = ((a + x as u32) % 65521, (b + (a + x as u32) % 65521) % 65521);
        }
        assert_eq!(&idat[q..], &(b << 16 | a).to_be_bytes());
        let row = w as usize * 4;
        for (y, r) in raw.chunks(row + 1).enumerate() {
            assert_eq!(r[0], 0);
            assert_eq!(&r[1..], &px[y * row..][..row]);
        }
    }
}
