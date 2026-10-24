//! AAC-LC encoder (ISO/IEC 14496-3), mono or stereo at 44.1 / 48 kHz (the rates the editor and the camera use; other
//! rates need their own scalefactor-band tables). KBD-window MDCT with block switching (long, start, eight short,
//! stop) on transients; a psychoacoustic model (tonality from FFT unpredictability, Bark spreading, absolute
//! threshold, pre-echo control); M/S stereo per band; scalefactors from the allowed noise per band; rate control on a
//! bit reservoir; Huffman sectioning by dynamic programming. Output: raw access units for MP4 ([`Enc::asc`]) or ADTS.

use crate::fft::{C, Fft};
use std::io;

mod huff;

const N: usize = 1024;
/// Coded bands per channel and frame: 49 long, or up to 8 groups × 14 short.
const NB: usize = 120;
const LONG: [usize; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];
const SHORT: [usize; 15] = [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];
/// Required signal-to-mask distance of tonal and noise-like maskers (dB), and of short blocks.
const TMN: f32 = 24.0;
const NMT: f32 = 6.0;
const SNR_SHORT: f32 = 8.0;
/// Level of a full-scale sine for the absolute threshold (dB SPL).
const FS_DB: f32 = 105.0;
/// Below the masking threshold the allowed noise falls this many times faster towards a floor that is nearly flat
/// (white noise costs the fewest bits per dB of SNR once nothing is audible); the floor follows the band energy with
/// this exponent (measured on the samples: 0.3 gains 1-2 dB SNR and NMR over a white floor, 0.8 loses 6 dB SNR).
const FLAT: f32 = 2.5;
const TILT: f32 = 0.3;
/// Rounding offset of the quantiser (minimises the error of the ¾-power companding).
const MAGIC: f32 = 0.4054;
const INF: u32 = 1 << 24;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seq {
    Long,
    Start,
    Short,
    Stop,
}

struct Chan {
    /// Input, scaled to 16-bit range: frames k − 1, k (being coded), k + 1 (lookahead).
    x: Vec<f32>,
    /// MDCT coefficients in coding order (short blocks grouped and interleaved), their |·|^¾, the quantised values.
    spec: Vec<f32>,
    p34: Vec<f32>,
    q: Vec<i16>,
    en: [f32; NB],
    thr: [f32; NB],
    ath: [f32; NB],
    /// log2 of: energy, masking threshold, the band's share of the noise floor (`TILT`) raised until it touches the
    /// masking threshold of one audible band, 6.75 / Σ√|x| (for the scalefactor); the smallest scalefactor keeping
    /// |q| ≤ 8191.
    le: [f32; NB],
    lt: [f32; NB],
    lw: [f32; NB],
    lf: [f32; NB],
    smin: [i32; NB],
    sf: [i32; NB],
    cb: [u8; NB],
    gg: i32,
    /// Last long frame's thresholds (pre-echo control).
    prev: [f32; 49],
    /// FFT magnitudes and unit phasors of the last two frames (tonality by unpredictability).
    r: [Vec<f32>; 2],
    u: [Vec<C>; 2],
    unp: Vec<f32>,
    /// Transient detector: high-pass state, the last 8 block energies.
    hp: f32,
    eh: [f32; 8],
}

impl Chan {
    fn new() -> Chan {
        Chan {
            x: vec![0.0; 3 * N],
            spec: vec![0.0; N],
            p34: vec![0.0; N],
            q: vec![0; N],
            en: [0.0; NB],
            thr: [0.0; NB],
            ath: [0.0; NB],
            le: [0.0; NB],
            lt: [0.0; NB],
            lw: [0.0; NB],
            lf: [0.0; NB],
            smin: [0; NB],
            sf: [0; NB],
            cb: [0; NB],
            gg: 0,
            prev: [f32::INFINITY; 49],
            r: [vec![0.0; N], vec![0.0; N]],
            u: [vec![C::new(1.0, 0.0); N], vec![C::new(1.0, 0.0); N]],
            unp: vec![1.0; N],
            hp: 0.0,
            eh: [0.0; 8],
        }
    }
}

/// Layout shared by both channels of a frame (common window).
struct Frame {
    seq: Seq,
    ng: usize,
    glen: [usize; 8],
    nb: usize,
    /// Band offsets in coding order, group-major.
    off: [usize; NB + 1],
    ms: [bool; NB],
    msp: u32,
}

struct Bw {
    b: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bw {
    fn put(&mut self, v: u32, n: u32) {
        self.acc = self.acc << n | v as u64;
        self.n += n;
        while self.n >= 8 {
            self.n -= 8;
            self.b.push((self.acc >> self.n) as u8);
        }
    }
    fn align(&mut self) {
        if self.n > 0 {
            self.put(0, 8 - self.n);
        }
    }
}

pub struct Enc {
    pub rate: u32,
    pub ch: usize,
    sfi: u8,
    /// Bits per frame (all channels) and the fraction carried over; reservoir fill and size.
    mean: f64,
    acc: f64,
    res: f64,
    rmax: f64,
    /// Rate control: log2(allowed noise / masking threshold) the reservoir settles on.
    lam: f32,
    /// Samples in `x` past frame k − 1; input samples taken; access units sent.
    fill: usize,
    taken: u64,
    k: u64,
    prev: Seq,
    /// Attack window (0–7) of frame k, found one frame ahead.
    att: Option<usize>,
    c: Vec<Chan>,
    fr: Frame,
    nbl: usize,
    nbs: usize,
    fl: Fft,
    fs: Fft,
    fp: Fft,
    twl: Vec<C>,
    tws: Vec<C>,
    wl: Vec<f32>,
    ws: Vec<f32>,
    hann: Vec<f32>,
    athl: [f32; 49],
    aths: [f32; 14],
    /// Spreading to the next higher / lower band (long, short).
    spl: [(f32, f32); 49],
    sps: [(f32, f32); 14],
    za: Vec<C>,
    zb: Vec<C>,
    zw: Vec<f32>,
    out: Bw,
    #[cfg(test)]
    shorts: usize,
    #[cfg(test)]
    long_only: bool,
}

fn mdct(fft: &Fft, tw: &[C], z: &[f32], out: &mut [f32], a: &mut [C], b: &mut [C]) {
    // folded to a DCT-IV of (−c_r − d, a − b_r), which runs as an m/2-point complex FFT between two twiddles
    let m = out.len();
    let h = m / 2;
    let v = |i: usize| {
        if i < h {
            -z[3 * h - 1 - i] - z[3 * h + i]
        } else {
            z[i - h] - z[3 * h - 1 - i]
        }
    };
    for n in 0..h {
        a[n] = C::new(v(2 * n), v(m - 1 - 2 * n)).mul(tw[n]);
    }
    fft.fwd(&a[..h], b);
    let s = m as f32; // the FFT's 1/h and the standard's factor 2
    for k in 0..h {
        let y = b[k].mul(tw[k]);
        out[2 * k] = y.r * s;
        out[m - 1 - 2 * k] = -y.i * s;
    }
}

/// Rising half of a Kaiser-Bessel-derived window of length `n`.
fn kbd(n: usize, a: f64) -> Vec<f32> {
    let i0 = |x: f64| {
        (1..50)
            .fold((1.0, 1.0), |(s, t), k| {
                let t = t * (x / 2.0 / k as f64).powi(2);
                (s + t, t)
            })
            .0
    };
    let h = n / 2;
    let k: Vec<f64> = (0..=h)
        .map(|j| {
            i0(std::f64::consts::PI * a * (1.0 - (2.0 * j as f64 / h as f64 - 1.0).powi(2)).sqrt())
        })
        .collect();
    let tot: f64 = k.iter().sum();
    let mut c = 0.0;
    (0..h)
        .map(|j| {
            c += k[j];
            (c / tot).sqrt() as f32
        })
        .collect()
}

fn bark(f: f32) -> f32 {
    13.0 * (0.00076 * f).atan() + 3.5 * (f / 7500.0).powi(2).atan()
}

/// Absolute threshold of hearing (dB SPL), Terhardt; held at its 15 kHz value above (many listeners still hear
/// there, and at high rates the bits are spare).
fn ath(f: f32) -> f32 {
    let k = f.clamp(20.0, 15000.0) / 1000.0;
    3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4)
}

/// Huffman-codes `q` with spectral codebook `cb`, calling `put(codeword, length)`; counting and writing share it.
fn code(q: &[i16], cb: usize, mut put: impl FnMut(u32, u32)) {
    let t = huff::CB[cb];
    // tuple size, index base, offset of signed books (unsigned books send signs and escapes after the codeword)
    let (dim, b, off) = [
        (0, 0, 0),
        (4, 3, 1),
        (4, 3, 1),
        (4, 3, 0),
        (4, 3, 0),
        (2, 9, 4),
        (2, 9, 4),
        (2, 8, 0),
        (2, 8, 0),
        (2, 13, 0),
        (2, 13, 0),
        (2, 17, 0),
    ][cb];
    for v in q.chunks_exact(dim) {
        let i = v.iter().fold(0, |i, &x| {
            i * b
                + if off > 0 {
                    (x + off) as usize
                } else {
                    (x.unsigned_abs() as usize).min(16)
                }
        });
        put(t[i] & 0xff_ffff, t[i] >> 24);
        if off > 0 {
            continue;
        }
        for &x in v.iter().filter(|&&x| x != 0) {
            put((x < 0) as u32, 1);
        }
        for x in v
            .iter()
            .map(|x| x.unsigned_abs() as u32)
            .filter(|&x| x >= 16)
        {
            // escape (book 11): n − 4 ones, a zero, the n low bits of x, n = ⌊log2 x⌋
            let n = x.ilog2();
            put(((1 << (n - 4)) - 1) << 1, n - 3);
            put(x - (1 << n), n);
        }
    }
}

fn quant(fr: &Frame, c: &mut Chan, lam: f32) -> usize {
    let nbn = fr.ng * fr.nb;
    let mut mx = [0u32; NB];
    let mut last = -1;
    for i in 0..nbn {
        let (a, b) = (fr.off[i], fr.off[i + 1]);
        // allowed noise: the masking threshold scaled by 2^lam; below it (lam < 0) it also flattens, see FLAT
        let ln = if lam >= 0.0 {
            c.lt[i] + lam
        } else {
            (c.lt[i] + lam).min(c.lw[i] + FLAT * lam)
        };
        if ln >= c.le[i] {
            c.q[a..b].fill(0);
            continue; // noise allowed above the band's energy: drop it
        }
        // quantisation noise of a band ≈ 4/27 · Σ√|x| · step^1.5, step = 2^((sf − 100) / 4)
        let mut sf = ((100.0 + 8.0 / 3.0 * (ln + c.lf[i])).round() as i32)
            .max(c.smin[i])
            .clamp(0, 255);
        if last >= 0 {
            sf = sf.clamp(last - 60, last + 60);
        }
        let k = (-0.1875 * (sf - 100) as f32).exp2();
        let mut m = 0;
        for ((q, &p), &x) in c.q[a..b].iter_mut().zip(&c.p34[a..b]).zip(&c.spec[a..b]) {
            let v = ((p * k + MAGIC) as i32).min(8191);
            m = m.max(v);
            *q = if x < 0.0 { -v } else { v } as i16;
        }
        if m > 0 {
            (mx[i], c.sf[i], last) = (m as u32, sf, sf);
        }
    }
    // per band and codebook: bits, or INF where the values exceed the book
    let mut cost = [[INF; 12]; NB];
    for i in 0..nbn {
        let q = &c.q[fr.off[i]..fr.off[i + 1]];
        let first = match mx[i] {
            0 => 0,
            1 => 1,
            2 => 3,
            3..=4 => 5,
            5..=7 => 7,
            8..=12 => 9,
            _ => 11,
        };
        if first == 0 {
            cost[i][0] = 0;
        }
        for cb in first.max(1)..12 {
            let mut n = 0;
            code(q, cb, |_, l| n += l);
            cost[i][cb] = n;
        }
    }
    let (sb, esc) = if fr.seq == Seq::Short {
        (3, 7)
    } else {
        (5, 31)
    };
    let mut bits = 0;
    for g in 0..fr.ng {
        let o = g * fr.nb;
        let mut best = [0u32; 50];
        let mut from = [(0usize, 0u8); 50];
        for j in 1..=fr.nb {
            best[j] = u32::MAX;
            for cb in 0..12 {
                let mut s = 0;
                for i in (0..j).rev() {
                    s += cost[o + i][cb];
                    if s >= INF {
                        break;
                    }
                    let t = best[i] + 4 + sb * (1 + (j - i) as u32 / esc) + s;
                    if t < best[j] {
                        (best[j], from[j]) = (t, (i, cb as u8));
                    }
                }
            }
        }
        bits += best[fr.nb] as usize;
        let mut j = fr.nb;
        while j > 0 {
            let (i, cb) = from[j];
            c.cb[o + i..o + j].fill(cb);
            j = i;
        }
    }
    // scalefactors: differences along the coded bands, from the global gain; all-zero bands in a coded section repeat
    // the previous value
    c.gg = (0..nbn).find(|&i| mx[i] > 0).map_or(100, |i| c.sf[i]);
    let mut last = c.gg;
    for i in 0..nbn {
        if c.cb[i] != 0 {
            if mx[i] == 0 {
                c.sf[i] = last;
            }
            bits += (huff::SCF[(c.sf[i] - last + 60) as usize] >> 24) as usize;
            last = c.sf[i];
        }
    }
    bits + 8 + 3
}

impl Enc {
    /// `bitrate` in bit/s for all channels, 16000 per channel up to the AAC-LC limit of 6144 bits per channel and
    /// frame (288 kbit/s per channel at 48 kHz, 264.6 at 44.1); 128–256 kbit/s is the useful range for stereo.
    pub fn new(rate: u32, ch: u16, bitrate: u32) -> io::Result<Enc> {
        let sfi = match rate {
            48000 => 3,
            44100 => 4,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("AAC encode at {rate} Hz"),
                ));
            }
        };
        if !(1..=2).contains(&ch)
            || bitrate < 16000 * ch as u32
            || bitrate as u64 * N as u64 > 6144 * ch as u64 * rate as u64
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("AAC encode {ch} ch at {bitrate} bit/s"),
            ));
        }
        let ch = ch as usize;
        let mean = bitrate as f64 * N as f64 / rate as f64;
        // audio bandwidth by bit rate per channel
        let bpc = bitrate as f32 / ch as f32;
        let cut = (11000.0 + (bpc - 32000.0) * 9000.0 / 64000.0)
            .clamp(11000.0, 20000.0)
            .min(rate as f32 * 0.45);
        let bin = (cut * 2048.0 / rate as f32) as usize;
        let nbl = (1..50).find(|&b| LONG[b] >= bin).unwrap_or(49);
        let nbs = (1..15).find(|&b| SHORT[b] * 8 >= bin).unwrap_or(14);
        let twl = (0..512)
            .map(|n| {
                let p = -std::f64::consts::PI * (n as f64 + 0.125) / 1024.0;
                C::new(p.cos() as f32, p.sin() as f32)
            })
            .collect();
        let tws = (0..64)
            .map(|n| {
                let p = -std::f64::consts::PI * (n as f64 + 0.125) / 128.0;
                C::new(p.cos() as f32, p.sin() as f32)
            })
            .collect();
        let mut e = Enc {
            rate,
            ch,
            sfi,
            mean,
            acc: 0.0,
            res: (6144 * ch) as f64 / 2.0 - mean / 2.0,
            rmax: (6144 * ch) as f64 - mean,
            lam: 0.0,
            fill: N,
            taken: 0,
            k: 0,
            prev: Seq::Long,
            att: None,
            c: (0..ch).map(|_| Chan::new()).collect(),
            fr: Frame {
                seq: Seq::Long,
                ng: 1,
                glen: [1; 8],
                nb: 0,
                off: [0; NB + 1],
                ms: [false; NB],
                msp: 0,
            },
            nbl,
            nbs,
            fl: Fft::new(512),
            fs: Fft::new(64),
            fp: Fft::new(2048),
            twl,
            tws,
            wl: kbd(2048, 4.0),
            ws: kbd(256, 6.0),
            hann: (0..2048)
                .map(|n| {
                    (0.5 - 0.5 * (std::f64::consts::PI * 2.0 * (n as f64 + 0.5) / 2048.0).cos())
                        as f32
                })
                .collect(),
            athl: [0.0; 49],
            aths: [0.0; 14],
            spl: [(0.0, 0.0); 49],
            sps: [(0.0, 0.0); 14],
            za: vec![C::default(); 2048],
            zb: vec![C::default(); 2048],
            zw: vec![0.0; 2048],
            out: Bw {
                b: Vec::with_capacity(6144 * ch / 8 + 64),
                acc: 0,
                n: 0,
            },
            #[cfg(test)]
            shorts: 0,
            #[cfg(test)]
            long_only: false,
        };
        // calibration: MDCT energy of a full-scale sine at a bin centre, long and short
        let mut s = vec![0f32; N];
        let z: Vec<f32> = (0..2048)
            .map(|n| {
                32768.0
                    * (std::f32::consts::PI * 100.5 * n as f32 / 1024.0).cos()
                    * e.wl[n.min(2047 - n)]
            })
            .collect();
        mdct(&e.fl, &e.twl, &z, &mut s, &mut e.za, &mut e.zb);
        let efl: f32 = s.iter().map(|x| x * x).sum();
        let z: Vec<f32> = (0..256)
            .map(|n| {
                32768.0
                    * (std::f32::consts::PI * 12.5 * n as f32 / 128.0).cos()
                    * e.ws[n.min(255 - n)]
            })
            .collect();
        mdct(&e.fs, &e.tws, &z, &mut s[..128], &mut e.za, &mut e.zb);
        let efs: f32 = s[..128].iter().map(|x| x * x).sum();
        let hz = rate as f32 / 2048.0;
        for (t, ath_, sp, bw, e0) in [
            (&LONG[..], &mut e.athl[..], &mut e.spl[..], hz, efl),
            (&SHORT[..], &mut e.aths[..], &mut e.sps[..], hz * 8.0, efs),
        ] {
            let nb = t.len() - 1;
            for b in 0..nb {
                let a = (t[b]..t[b + 1])
                    .map(|k| ath((k as f32 + 0.5) * bw))
                    .fold(f32::MAX, f32::min);
                ath_[b] = e0 * 10f32.powf((a - FS_DB) / 10.0);
                if b + 1 < nb {
                    let dz = bark((t[b + 2] + t[b + 1]) as f32 * 0.5 * bw)
                        - bark((t[b + 1] + t[b]) as f32 * 0.5 * bw);
                    sp[b] = (10f32.powf(-1.5 * dz), 10f32.powf(-3.0 * dz));
                }
            }
        }
        Ok(e)
    }

    /// AudioSpecificConfig (AAC-LC, 1024-sample frames) for the MP4 `esds`.
    pub fn asc(&self) -> [u8; 2] {
        [
            0x10 | self.sfi >> 1,
            (self.sfi & 1) << 7 | (self.ch as u8) << 3,
        ]
    }

    /// ADTS header (no CRC) for an access unit of `len` bytes.
    pub fn adts(&self, len: usize) -> [u8; 7] {
        let (n, s, c) = (len + 7, self.sfi as usize, self.ch);
        [
            0xff,
            0xf1,
            (1 << 6 | s << 2 | c >> 2) as u8,
            ((c & 3) << 6 | n >> 11) as u8,
            (n >> 3) as u8,
            ((n & 7) << 5 | 0x1f) as u8,
            0xfc,
        ]
    }

    /// Feeds interleaved samples (±1.0 full scale) and calls `f(access_unit, pts, duration)` for each finished
    /// frame; pts in samples of the input, the first frame (encoder priming) at −1024.
    pub fn push(&mut self, pcm: &[f32], mut f: impl FnMut(&[u8], i64, u32)) {
        let ch = self.ch;
        let mut s = &pcm[..pcm.len() / ch * ch];
        while !s.is_empty() {
            let n = (s.len() / ch).min(3 * N - self.fill);
            for (c, st) in self.c.iter_mut().enumerate() {
                for (d, v) in st.x[self.fill..self.fill + n]
                    .iter_mut()
                    .zip(s[c..].iter().step_by(ch))
                {
                    *d = v * 32768.0;
                }
            }
            (self.fill, self.taken, s) = (self.fill + n, self.taken + n as u64, &s[n * ch..]);
            if self.fill == 3 * N {
                self.frame(&mut f, N as u32);
            }
        }
    }

    /// Pads the input with silence and sends the remaining frames; the last one's duration ends at the last input
    /// sample.
    pub fn finish(&mut self, mut f: impl FnMut(&[u8], i64, u32)) {
        // frame k decodes to input [(k − 1)·1024, k·1024)
        let last = self.taken.div_ceil(N as u64);
        while self.taken > 0 && self.k <= last {
            for st in &mut self.c {
                st.x[self.fill..].fill(0.0);
            }
            self.fill = 3 * N;
            let d = if self.k == last {
                self.taken - (last - 1) * N as u64
            } else {
                N as u64
            };
            self.frame(&mut f, d as u32);
        }
    }

    fn frame(&mut self, f: &mut impl FnMut(&[u8], i64, u32), dur: u32) {
        // transient in the lookahead frame: high-passed energy of the 8 blocks under its short windows against the
        // blocks before
        let mut nxt: Option<usize> = None;
        for st in &mut self.c {
            for w in 0..8 {
                let mut e = 0.0;
                for &v in &st.x[1536 + 128 * w..1664 + 128 * w] {
                    let y = v - st.hp;
                    st.hp = v;
                    e += y * y;
                }
                let avg = st.eh.iter().sum::<f32>() / 8.0;
                st.eh.copy_within(1.., 0);
                st.eh[7] = e;
                if e > 10.0 * avg && e > 1e6 {
                    nxt = Some(nxt.map_or(w, |a: usize| a.min(w)));
                }
            }
        }
        #[cfg(test)]
        if self.long_only {
            nxt = None;
        }
        let seq = match (self.att, self.prev, nxt) {
            (Some(_), ..) | (None, Seq::Short, Some(_)) => Seq::Short,
            (None, Seq::Short, None) => Seq::Stop,
            (None, _, Some(_)) => Seq::Start,
            _ => Seq::Long,
        };
        let fr = &mut self.fr;
        fr.seq = seq;
        if seq == Seq::Short {
            // the attack window alone, the windows before and after it in one group each
            let a = self.att.unwrap_or(4);
            fr.ng = 0;
            for l in [a, 1, 7 - a] {
                if l > 0 {
                    fr.glen[fr.ng] = l;
                    fr.ng += 1;
                }
            }
            if self.att.is_none() {
                (fr.ng, fr.glen[0], fr.glen[1]) = (2, 4, 4);
            }
            fr.nb = self.nbs;
            let mut p = 0;
            for g in 0..fr.ng {
                for b in 0..fr.nb {
                    fr.off[g * fr.nb + b] = p;
                    p += (SHORT[b + 1] - SHORT[b]) * fr.glen[g];
                }
            }
            fr.off[fr.ng * fr.nb] = p;
            #[cfg(test)]
            {
                self.shorts += 1;
            }
        } else {
            (fr.ng, fr.nb) = (1, self.nbl);
            fr.off[..=fr.nb].copy_from_slice(&LONG[..=fr.nb]);
        }
        for c in 0..self.ch {
            self.analyse(c);
        }
        self.stereo();
        for st in &mut self.c {
            let nbn = self.fr.ng * self.fr.nb;
            let mut top = f32::NEG_INFINITY;
            for i in 0..nbn {
                let (a, b) = (self.fr.off[i], self.fr.off[i + 1]);
                let (mut ff, mut mx) = (0f32, 0f32);
                for (p, &x) in st.p34[a..b].iter_mut().zip(&st.spec[a..b]) {
                    let s = x.abs().sqrt();
                    (*p, ff, mx) = (s * s.sqrt(), ff + s, mx.max(x.abs()));
                }
                let w = ((b - a) as f32).log2();
                (st.le[i], st.lt[i]) = (st.en[i].log2(), st.thr[i].log2());
                st.lw[i] = w + TILT * (st.le[i] - w);
                st.lf[i] = (6.75 / ff.max(1e-9)).log2();
                st.smin[i] = (30.67 + 4.0 * mx.max(1.0).log2()).ceil() as i32;
                if st.en[i] > st.ath[i] {
                    top = top.max(st.lt[i] - st.lw[i]); // audible bands only: the absolute threshold soars at the top
                }
            }
            // bands under the absolute threshold keep it: flattening them would spend bits on what nobody hears
            for i in 0..nbn {
                st.lw[i] = if st.en[i] > st.ath[i] {
                    st.lw[i] + top
                } else {
                    f32::INFINITY
                };
            }
        }
        // rate control: the noise level the reservoir asks for; more noise only if the frame would not fit
        let mut mb = self.mean + self.acc;
        self.acc = mb - mb.floor();
        mb = mb.floor();
        let avail = (mb + self.res).min((6144 * self.ch) as f64) as usize;
        let mid = self.rmax / 2.0;
        let lam = (self.lam - ((self.res - mid) / mb) as f32).clamp(-40.0, 24.0);
        // the noise level that would spend exactly the mean on this frame, by one secant step; the reference level
        // follows it
        let mut bits = self.count(lam);
        let d = if bits as f64 > mb { 1.0 } else { -1.0 };
        let s = (self.count(lam + d) as f32 - bits as f32) / d;
        let be = if s < -1.0 {
            (lam + (mb as f32 - bits as f32) / s).clamp(lam - 8.0, lam + 8.0)
        } else {
            lam
        };
        self.lam = (self.lam + 0.5 * (be - self.lam)).clamp(-40.0, 24.0);
        bits = self.count(lam);
        if bits > avail {
            let (mut lo, mut hi) = (lam, lam + 2.0);
            while self.count(hi) > avail && hi < 64.0 {
                (lo, hi) = (hi, hi + 4.0);
            }
            for _ in 0..8 {
                let m = (lo + hi) / 2.0;
                if self.count(m) > avail {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            bits = self.count(hi);
        }
        // fill elements grow the AU to what leaves the reservoir full, within the 6144 bits per channel a decoder
        // buffers; bits an AU at that limit cannot carry are lost
        let max = 6144 * self.ch;
        let to = (self.res + mb - self.rmax).ceil().max(0.0) as usize;
        self.write(to.min(max));
        let used = self.out.b.len() * 8;
        debug_assert!(
            used == bits || used > bits && used <= to.min(max),
            "counted {bits} bits, wrote {used} for {to}"
        );
        self.res += mb - used as f64;
        if to > max {
            self.res = self.res.min(self.rmax);
        }
        f(&self.out.b, self.k as i64 * N as i64 - N as i64, dur);
        self.k += 1;
        (self.prev, self.att, self.fill) = (seq, nxt, 2 * N);
        for st in &mut self.c {
            st.x.copy_within(N.., 0);
        }
    }

    /// MDCT and masking thresholds of channel `c` for the current frame layout.
    fn analyse(&mut self, c: usize) {
        let fr = &self.fr;
        let st = &mut self.c[c];
        // tonality: how far each FFT bin is from its extrapolation from the two frames before (0 = predictable)
        for n in 0..2048 {
            self.za[n] = C::new(st.x[n] * self.hann[n], 0.0);
        }
        self.fp.fwd(&self.za, &mut self.zb);
        for k in 0..N {
            let y = self.zb[k];
            let r = (y.r * y.r + y.i * y.i).sqrt();
            let u = if r > 0.0 {
                C::new(y.r / r, y.i / r)
            } else {
                C::new(1.0, 0.0)
            };
            let rp = 2.0 * st.r[0][k] - st.r[1][k];
            let (u1, u2) = (st.u[0][k], st.u[1][k]);
            let up = u1.mul(u1).mul(C::new(u2.r, -u2.i));
            let (dr, di) = (y.r - rp * up.r, y.i - rp * up.i);
            let d = r + rp.abs();
            st.unp[k] = if d > 0.0 {
                (dr * dr + di * di).sqrt() / d
            } else {
                1.0
            };
            (st.r[1][k], st.r[0][k], st.u[1][k], st.u[0][k]) = (st.r[0][k], r, u1, u);
        }
        let x = &st.x;
        if fr.seq != Seq::Short {
            let (wl, ws) = (&self.wl, &self.ws);
            for n in 0..2048 {
                // long halves rise wl[n] / fall wl[2047 − n]; start and stop frames use the short halves
                let w = match (fr.seq, n) {
                    (Seq::Stop, 0..448) | (Seq::Start, 1600..) => 0.0,
                    (Seq::Stop, 448..576) => ws[n - 448],
                    (Seq::Stop, 576..1024) | (Seq::Start, 1024..1472) => 1.0,
                    (Seq::Start, 1472..1600) => ws[1599 - n],
                    (_, 0..1024) => wl[n],
                    _ => wl[2047 - n],
                };
                self.zw[n] = x[n] * w;
            }
            mdct(
                &self.fl,
                &self.twl,
                &self.zw,
                &mut st.spec,
                &mut self.za,
                &mut self.zb,
            );
            let nb = fr.nb;
            for b in 0..nb {
                let (lo, hi) = (LONG[b], LONG[b + 1]);
                let en: f32 = st.spec[lo..hi].iter().map(|v| v * v).sum();
                let (mut cs, mut ws) = (0f32, 1e-20f32);
                for k in lo..hi {
                    let p = st.r[0][k] * st.r[0][k];
                    (cs, ws) = (cs + st.unp[k] * p, ws + p);
                }
                let t = (-0.299 - 0.43 * (cs / ws).max(1e-9).ln()).clamp(0.0, 1.0);
                st.en[b] = en;
                st.thr[b] = en * 10f32.powf(-(t * TMN + (1.0 - t) * NMT) / 10.0);
            }
            spread(&mut st.thr[..nb], &self.spl);
            for b in 0..nb {
                // pre-echo control (3GPP TS 26.403): a threshold may not jump above twice the last long frame's,
                // nor fall below 1/100 of its own; the history keeps the unclamped value, or a silent frame would
                // pin it to zero (short frames reset it)
                let t = st.thr[b];
                st.thr[b] = t.min(2.0 * st.prev[b]).max(0.01 * t).max(self.athl[b]);
                (st.prev[b], st.ath[b]) = (t, self.athl[b]);
            }
            st.spec[LONG[nb]..].fill(0.0);
            return;
        }
        // eight short windows, then grouped: energies add, the group's threshold is its quietest window's
        let nb = fr.nb;
        let mut thr = [[0f32; 14]; 8];
        let mut en = [[0f32; 14]; 8];
        let ws = &self.ws;
        for w in 0..8 {
            let z: [f32; 256] = std::array::from_fn(|n| x[448 + 128 * w + n] * ws[n.min(255 - n)]);
            let o = &mut self.zw[1024 + 128 * w..1152 + 128 * w];
            mdct(&self.fs, &self.tws, &z, o, &mut self.za, &mut self.zb);
            for b in 0..nb {
                let e: f32 = o[SHORT[b]..SHORT[b + 1]].iter().map(|v| v * v).sum();
                (en[w][b], thr[w][b]) = (e, e * 10f32.powf(-SNR_SHORT / 10.0));
            }
            spread(&mut thr[w][..nb], &self.sps);
        }
        st.prev = [f32::INFINITY; 49];
        let (mut p, mut w0) = (0, 0);
        for g in 0..fr.ng {
            let l = fr.glen[g];
            for b in 0..nb {
                let i = g * nb + b;
                st.en[i] = (w0..w0 + l).map(|w| en[w][b]).sum();
                st.ath[i] = self.aths[b] * l as f32;
                st.thr[i] = (w0..w0 + l)
                    .map(|w| thr[w][b].max(self.aths[b]))
                    .fold(f32::MAX, f32::min)
                    * l as f32;
                for w in w0..w0 + l {
                    let s = &self.zw[1024 + 128 * w + SHORT[b]..1024 + 128 * w + SHORT[b + 1]];
                    st.spec[p..p + s.len()].copy_from_slice(s);
                    p += s.len();
                }
            }
            w0 += l;
        }
        st.spec[p..].fill(0.0);
    }

    /// M/S per band where it needs fewer bits by perceptual entropy; the noise of M and S adds up in L and R, so
    /// each gets half the lower threshold.
    fn stereo(&mut self) {
        let fr = &mut self.fr;
        fr.msp = 0;
        if self.ch < 2 {
            return;
        }
        let nbn = fr.ng * fr.nb;
        let (l, r) = self.c.split_at_mut(1);
        let (l, r) = (&mut l[0], &mut r[0]);
        let pe = |e: f32, t: f32| if e > t { (e / t).log2() } else { 0.0 };
        let mut n = 0;
        for i in 0..nbn {
            let (a, b) = (fr.off[i], fr.off[i + 1]);
            let (mut em, mut es) = (0f32, 0f32);
            for k in a..b {
                let (m, s) = ((l.spec[k] + r.spec[k]) * 0.5, (l.spec[k] - r.spec[k]) * 0.5);
                (em, es) = (em + m * m, es + s * s);
            }
            let t = l.thr[i].min(r.thr[i]) * 0.5;
            fr.ms[i] = pe(em, t) + pe(es, t) < pe(l.en[i], l.thr[i]) + pe(r.en[i], r.thr[i]);
            if fr.ms[i] {
                for k in a..b {
                    (l.spec[k], r.spec[k]) =
                        ((l.spec[k] + r.spec[k]) * 0.5, (l.spec[k] - r.spec[k]) * 0.5);
                }
                (l.en[i], r.en[i], l.thr[i], r.thr[i]) = (em, es, t, t);
                n += 1;
            }
        }
        fr.msp = match n {
            0 => 0,
            _ if n == nbn => 2,
            _ => 1,
        };
    }

    /// Quantises every channel at noise level `lam` and returns the frame size in bits (byte-aligned, no fill).
    fn count(&mut self, lam: f32) -> usize {
        let fr = &self.fr;
        let info = if fr.seq == Seq::Short { 15 } else { 11 };
        let mut bits = 3 + 4 + 3;
        for st in &mut self.c {
            bits += quant(fr, st, lam);
        }
        bits += if self.ch == 2 {
            1 + info + 2 + if fr.msp == 1 { fr.ng * fr.nb } else { 0 }
        } else {
            info
        };
        bits.div_ceil(8) * 8
    }

    /// Writes the AU, with fill elements up to `to` bits when it is shorter.
    fn write(&mut self, to: usize) {
        let fr = &self.fr;
        let w = &mut self.out;
        w.b.clear();
        let info = |w: &mut Bw| {
            w.put(0, 1);
            w.put(fr.seq as u32, 2);
            w.put(1, 1); // KBD window
            if fr.seq == Seq::Short {
                w.put(fr.nb as u32, 4);
                let mut g = 0u32;
                for (i, &l) in fr.glen[..fr.ng].iter().enumerate() {
                    // a 1 for each window that continues the group before it; window 0 has no bit
                    let first = fr.glen[..i].iter().sum::<usize>();
                    for k in first.max(1)..first + l {
                        g |= ((k > first) as u32) << (7 - k);
                    }
                }
                w.put(g, 7);
            } else {
                w.put(fr.nb as u32, 6);
                w.put(0, 1); // no prediction
            }
        };
        if self.ch == 2 {
            w.put(1, 3); // CPE
            w.put(0, 4);
            w.put(1, 1); // common window
            info(w);
            w.put(fr.msp, 2);
            if fr.msp == 1 {
                for &m in &fr.ms[..fr.ng * fr.nb] {
                    w.put(m as u32, 1);
                }
            }
        } else {
            w.put(0, 3); // SCE
            w.put(0, 4);
        }
        let (sb, esc) = if fr.seq == Seq::Short {
            (3, 7)
        } else {
            (5, 31)
        };
        for st in &self.c {
            w.put(st.gg as u32, 8);
            if self.ch == 1 {
                info(w);
            }
            for g in 0..fr.ng {
                let o = g * fr.nb;
                let mut i = 0;
                while i < fr.nb {
                    let cb = st.cb[o + i];
                    let mut j = i + 1;
                    while j < fr.nb && st.cb[o + j] == cb {
                        j += 1;
                    }
                    w.put(cb as u32, 4);
                    let mut l = j - i;
                    while l >= esc {
                        w.put(esc as u32, sb);
                        l -= esc;
                    }
                    w.put(l as u32, sb);
                    i = j;
                }
            }
            let mut last = st.gg;
            for i in 0..fr.ng * fr.nb {
                if st.cb[i] != 0 {
                    let e = huff::SCF[(st.sf[i] - last + 60) as usize];
                    w.put(e & 0xff_ffff, e >> 24);
                    last = st.sf[i];
                }
            }
            w.put(0, 3); // no pulse, TNS, gain control
            for i in 0..fr.ng * fr.nb {
                if st.cb[i] != 0 {
                    code(
                        &st.q[fr.off[i]..fr.off[i + 1]],
                        st.cb[i] as usize,
                        |c, l| w.put(c, l),
                    );
                }
            }
        }
        // fill elements of EXT_FILL bytes: n bytes take 7 + 8n bits below 15, 15 + 8n from there; room for the END
        // and the byte alignment stays
        let mut f = (to & !7).saturating_sub(w.b.len() * 8 + w.n as usize + 3);
        while f >= 15 {
            let n = if f >= 135 {
                ((f - 15) / 8).min(269)
            } else {
                ((f - 7) / 8).min(14)
            };
            w.put(6, 3);
            if n < 15 {
                w.put(n as u32, 4);
            } else {
                w.put(15, 4);
                w.put(n as u32 - 14, 8);
            }
            w.put(0, 8);
            for _ in 1..n {
                w.put(0xa5, 8);
            }
            f -= 8 * n + if n < 15 { 7 } else { 15 };
        }
        w.put(7, 3); // END
        w.align();
    }
}

/// Masking spreads into the neighbouring bands: 15 dB/Bark upwards, 30 dB/Bark downwards.
fn spread(t: &mut [f32], sp: &[(f32, f32)]) {
    for b in 1..t.len() {
        t[b] = t[b].max(t[b - 1] * sp[b - 1].0);
    }
    for b in (0..t.len().saturating_sub(1)).rev() {
        t[b] = t[b].max(t[b + 1] * sp[b].1);
    }
}

#[cfg(test)]
mod tests;
