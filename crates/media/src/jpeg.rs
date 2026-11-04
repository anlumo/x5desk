//! Baseline JPEG (ITU T.81) of an RGBA8 picture: JFIF Y'CbCr (BT.601 matrix, full range), 4:2:0 with centred
//! chroma (2 × 2 mean, as JFIF sites it), the Annex K quantisation tables scaled by quality as libjpeg does, the
//! Annex K Huffman tables, and a restart marker after every MCU row so the rows encode on all cores. An optional XMP
//! packet goes into APP1 (GPano for photo spheres, `gpano`).

/// Natural (row-major) index of each zigzag position.
const ZZ: [usize; 64] = {
    let (mut z, mut i, mut s) = ([0; 64], 0, 0usize);
    while s < 15 {
        let (lo, hi) = (s.saturating_sub(7), if s < 7 { s } else { 7 });
        let mut k = 0;
        while k <= hi - lo {
            // odd diagonals run down-left, even ones up-right
            let r = if s % 2 == 1 { lo + k } else { hi - k };
            z[i] = r * 8 + s - r;
            (i, k) = (i + 1, k + 1);
        }
        s += 1;
    }
    z
};

/// Annex K.1, natural order: luminance, chrominance.
const Q: [[u8; 64]; 2] = [
    [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ],
    [
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99,
        99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ],
];

/// Annex K.3: codes per length 1..=16, then the symbols; DC luminance, DC chrominance, AC luminance, AC chrominance.
const HT: [([u8; 16], &[u8]); 4] = [
    (
        [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    (
        [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    (
        [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
        &[
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1,
            0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18,
            0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
            0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
            0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
        ],
    ),
    (
        [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
        &[
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07,
            0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09,
            0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25,
            0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56,
            0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74,
            0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
            0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6,
            0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2,
            0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
        ],
    ),
];

/// Code and length per symbol (Annex C).
type Codes = [(u16, u8); 256];

fn codes((bits, vals): &([u8; 16], &[u8])) -> Codes {
    let (mut t, mut c, mut i) = ([(0, 0); 256], 0u16, 0);
    for (l, &n) in bits.iter().enumerate() {
        for _ in 0..n {
            t[vals[i] as usize] = (c, l as u8 + 1);
            (c, i) = (c + 1, i + 1);
        }
        c <<= 1;
    }
    t
}

/// Bit writer with byte stuffing (0xFF → 0xFF 0x00).
struct Bits<'a> {
    o: &'a mut Vec<u8>,
    acc: u32,
    n: u32,
}

impl Bits<'_> {
    fn put(&mut self, v: u32, l: u32) {
        self.acc = self.acc << l | v & ((1 << l) - 1);
        self.n += l;
        while self.n >= 8 {
            self.n -= 8;
            let b = (self.acc >> self.n) as u8;
            self.o.push(b);
            if b == 0xff {
                self.o.push(0);
            }
        }
        self.acc &= (1 << self.n) - 1;
    }

    fn code(&mut self, (c, l): (u16, u8)) {
        self.put(c as u32, l as u32);
    }
}

/// Magnitude category: bits of |v|.
fn cat(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// `M[u][x] = ½·C(u)·cos((2x + 1)uπ/16)`: F = M·f·Mᵀ is the 8 × 8 FDCT.
fn dct_matrix() -> [[f32; 8]; 8] {
    std::array::from_fn(|u| {
        std::array::from_fn(|x| {
            let c = if u == 0 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            (0.5 * c * ((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0).cos()) as f32
        })
    })
}

/// One MCU row: 16 picture rows (edges repeated), DC predictors from 0, padded to a byte.
#[allow(clippy::too_many_arguments)]
fn row(
    px: &[u8],
    w: usize,
    h: usize,
    my: usize,
    m: &[[f32; 8]; 8],
    rq: &[[f32; 64]; 2],
    hc: &[Codes; 4],
    o: &mut Vec<u8>,
) {
    let mut b = Bits { o, acc: 0, n: 0 };
    let mut pred = [0i32; 3];
    let (mut blk, mut t, mut c) = ([[0f32; 64]; 6], [0f32; 64], [0i32; 64]);
    for mx in 0..w.div_ceil(16) {
        for v in &mut blk[4..] {
            v.fill(0.0);
        }
        for y in 0..16 {
            let r = &px[(my * 16 + y).min(h - 1) * w * 4..][..w * 4];
            for x in 0..16 {
                let p = &r[(mx * 16 + x).min(w - 1) * 4..][..3];
                let [r, g, bl] = [p[0] as f32, p[1] as f32, p[2] as f32];
                blk[y / 8 * 2 + x / 8][(y % 8) * 8 + x % 8] =
                    0.299 * r + 0.587 * g + 0.114 * bl - 128.0;
                let i = (y / 2) * 8 + x / 2;
                blk[4][i] += 0.25 * (-0.168736 * r - 0.331264 * g + 0.5 * bl);
                blk[5][i] += 0.25 * (0.5 * r - 0.418688 * g - 0.081312 * bl);
            }
        }
        for (k, f) in blk.iter().enumerate() {
            let ci = k.saturating_sub(3);
            let q = (ci > 0) as usize;
            // t = M·f, then F = t·Mᵀ, quantised in zigzag order
            for u in 0..8 {
                for x in 0..8 {
                    t[u * 8 + x] = (0..8).map(|y| m[u][y] * f[y * 8 + x]).sum();
                }
            }
            for (z, &n) in ZZ.iter().enumerate() {
                let (u, v) = (n / 8, n % 8);
                let s: f32 = (0..8).map(|x| t[u * 8 + x] * m[v][x]).sum();
                c[z] = (s * rq[q][n]).round() as i32;
            }
            let d = c[0] - pred[ci];
            pred[ci] = c[0];
            let n = cat(d);
            b.code(hc[q][n as usize]);
            b.put((d - (d < 0) as i32) as u32, n);
            let mut run = 0;
            for &a in &c[1..] {
                if a == 0 {
                    run += 1;
                    continue;
                }
                while run > 15 {
                    b.code(hc[2 + q][0xf0]);
                    run -= 16;
                }
                let n = cat(a);
                b.code(hc[2 + q][(run << 4 | n) as usize]);
                b.put((a - (a < 0) as i32) as u32, n);
                run = 0;
            }
            if run > 0 {
                b.code(hc[2 + q][0]);
            }
        }
    }
    // pad with 1 bits
    if b.n > 0 {
        b.put(0xff, 8 - b.n);
    }
}

fn seg(o: &mut Vec<u8>, m: u8, body: &[u8]) {
    o.extend_from_slice(&[0xff, m]);
    o.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
    o.extend_from_slice(body);
}

/// JPEG of the RGBA8 picture `px` (`w` × `h`, rows packed, alpha ignored) at `quality` 1..=100 (libjpeg's scale),
/// with `xmp` (a whole XMP packet) in APP1. Sizes up to 65535 × 65535.
pub fn encode(
    px: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    xmp: Option<&str>,
) -> Result<Vec<u8>, String> {
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 || w > 65535 || h > 65535 || px.len() < w * h * 4 {
        return Err(format!("JPEG of {w}x{h} from {} bytes", px.len()));
    }
    let q = quality.clamp(1, 100) as u32;
    let s = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let qt = Q.map(|t| t.map(|v| ((v as u32 * s + 50) / 100).clamp(1, 255) as u8));
    let rq = qt.map(|t| t.map(|v| 1.0 / v as f32));
    let hc = HT.each_ref().map(codes);
    let m = dct_matrix();
    let rows = h.div_ceil(16);
    let mut data = vec![vec![]; rows];
    let nt = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(rows);
    std::thread::scope(|sc| {
        for (i, ch) in data.chunks_mut(rows.div_ceil(nt)).enumerate() {
            let (rq, hc, m) = (&rq, &hc, &m);
            let at = i * rows.div_ceil(nt);
            sc.spawn(move || {
                for (k, o) in ch.iter_mut().enumerate() {
                    row(px, w, h, at + k, m, rq, hc, o);
                }
            });
        }
    });
    let mut o = Vec::with_capacity(data.iter().map(|d| d.len() + 2).sum::<usize>() + 1024);
    o.extend_from_slice(&[0xff, 0xd8]);
    seg(&mut o, 0xe0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    if let Some(x) = xmp {
        let b = [&b"http://ns.adobe.com/xap/1.0/\0"[..], x.as_bytes()].concat();
        if b.len() > 65533 {
            return Err("XMP packet over 64 KB".into());
        }
        seg(&mut o, 0xe1, &b);
    }
    let mut b = vec![];
    for (i, t) in qt.iter().enumerate() {
        b.push(i as u8);
        b.extend(ZZ.map(|n| t[n]));
    }
    seg(&mut o, 0xdb, &b);
    b.clear();
    b.push(8);
    b.extend_from_slice(&(h as u16).to_be_bytes());
    b.extend_from_slice(&(w as u16).to_be_bytes());
    b.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    seg(&mut o, 0xc0, &b);
    b.clear();
    // class (DC 0, AC 1) and id (luminance 0, chrominance 1) of each table
    for ((bits, vals), tc) in HT.iter().zip([0x00, 0x01, 0x10, 0x11]) {
        b.push(tc);
        b.extend_from_slice(bits);
        b.extend_from_slice(vals);
    }
    seg(&mut o, 0xc4, &b);
    seg(&mut o, 0xdd, &(w.div_ceil(16) as u16).to_be_bytes());
    seg(&mut o, 0xda, &[3, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0]);
    for (i, d) in data.iter().enumerate() {
        if i > 0 {
            o.extend_from_slice(&[0xff, 0xd0 + (i - 1) as u8 % 8]);
        }
        o.extend_from_slice(d);
    }
    o.extend_from_slice(&[0xff, 0xd9]);
    Ok(o)
}

/// XMP that makes viewers treat a `w` × `h` equirect as a photo sphere (Google's GPano), the whole sphere.
pub fn gpano(w: u32, h: u32) -> String {
    format!(
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\
<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\
<rdf:Description rdf:about=\"\" xmlns:GPano=\"http://ns.google.com/photos/1.0/panorama/\">\
<GPano:ProjectionType>equirectangular</GPano:ProjectionType>\
<GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>\
<GPano:CroppedAreaImageWidthPixels>{w}</GPano:CroppedAreaImageWidthPixels>\
<GPano:CroppedAreaImageHeightPixels>{h}</GPano:CroppedAreaImageHeightPixels>\
<GPano:FullPanoWidthPixels>{w}</GPano:FullPanoWidthPixels>\
<GPano:FullPanoHeightPixels>{h}</GPano:FullPanoHeightPixels>\
<GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>\
<GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>\
<GPano:StitchingSoftware>x5desk</GPano:StitchingSoftware>\
</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables() {
        let mut z = ZZ;
        z.sort();
        assert!(z.iter().enumerate().all(|(i, &n)| i == n));
        assert_eq!(ZZ[..10], [0, 1, 8, 16, 9, 2, 3, 10, 17, 24]);
        assert_eq!(ZZ[58..], [61, 54, 47, 55, 62, 63]);
        for (bits, vals) in HT {
            assert_eq!(bits.iter().map(|&n| n as usize).sum::<usize>(), vals.len());
        }
        // Annex K.3.1: the first DC luminance codes
        let c = codes(&HT[0]);
        assert_eq!(
            [c[0], c[1], c[5], c[11]],
            [(0, 2), (2, 3), (6, 3), (510, 9)]
        );
    }
}
