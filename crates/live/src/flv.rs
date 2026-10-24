//! FLV tags for H.264 and AAC. The builders write tag bodies, which are also RTMP message payloads;
//! [`Writer`] adds the file framing. Timestamps are milliseconds.

use crate::amf;
use std::io::{self, Write};

pub const AUDIO: u8 = 8;
pub const VIDEO: u8 = 9;
pub const DATA: u8 = 18;

/// Stream properties for `onMetaData`; zero leaves a field out.
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub w: u32,
    pub h: u32,
    pub fps: f64,
    /// kbit/s
    pub vkbps: f64,
    pub rate: u32,
    pub ch: u32,
    /// kbit/s
    pub akbps: f64,
}

/// `"onMetaData", {…}`. RTMP puts `"@setDataFrame"` in front.
pub fn meta(o: &mut Vec<u8>, m: &Meta) {
    let (v, a) = ((m.w > 0) as u8 as f64, (m.rate > 0) as u8 as f64);
    let f = [
        ("width", m.w as f64),
        ("height", m.h as f64),
        ("framerate", m.fps),
        ("videocodecid", 7.0 * v),
        ("videodatarate", m.vkbps),
        ("audiocodecid", 10.0 * a),
        ("audiosamplerate", m.rate as f64),
        ("audiosamplesize", 16.0 * a),
        ("audiodatarate", m.akbps),
    ];
    amf::str(o, "onMetaData");
    amf::ecma(
        o,
        f.iter().filter(|f| f.1 > 0.0).count() as u32 + a as u32 + 1,
    );
    for (k, x) in f.into_iter().filter(|f| f.1 > 0.0) {
        amf::key(o, k);
        amf::num(o, x);
    }
    if a > 0.0 {
        amf::key(o, "stereo");
        amf::bool(o, m.ch > 1);
    }
    amf::key(o, "encoder");
    amf::str(o, "x5desk");
    amf::end(o);
}

/// AVCDecoderConfigurationRecord from one SPS and one PPS (NAL units without start codes).
/// The High-profile extension (chroma format, bit depths) is left out, as the X5 does; decoders
/// read those from the SPS.
pub fn avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut o = vec![1, sps[1], sps[2], sps[3], 0xFF, 0xE1];
    o.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    o.extend_from_slice(sps);
    o.push(1);
    o.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    o.extend_from_slice(pps);
    o
}

/// The NAL units of an Annex-B stream, without start codes.
pub fn nals(b: &[u8]) -> impl Iterator<Item = &[u8]> {
    let sc = |i: usize| {
        b[i..]
            .windows(3)
            .position(|w| w == [0, 0, 1])
            .map(|p| i + p)
    };
    let mut i = sc(0).map(|p| p + 3);
    std::iter::from_fn(move || {
        let s = i?;
        let n = sc(s);
        i = n.map(|p| p + 3);
        let mut e = n.unwrap_or(b.len());
        // trailing_zero_8bits and the first byte of a 4-byte start code; a NAL never ends in 0
        while e > s && b[e - 1] == 0 {
            e -= 1;
        }
        Some(&b[s..e])
    })
    .filter(|n| !n.is_empty())
}

pub fn avc_seq(o: &mut Vec<u8>, avcc: &[u8]) {
    o.extend_from_slice(&[0x17, 0, 0, 0, 0]);
    o.extend_from_slice(avcc);
}

/// One access unit; `cts` = pts − dts in ms.
pub fn avc<'a>(o: &mut Vec<u8>, key: bool, cts: i32, nals: impl IntoIterator<Item = &'a [u8]>) {
    o.extend_from_slice(&[if key { 0x17 } else { 0x27 }, 1]);
    o.extend_from_slice(&cts.to_be_bytes()[1..]);
    for n in nals {
        o.extend_from_slice(&(n.len() as u32).to_be_bytes());
        o.extend_from_slice(n);
    }
}

// FLV fixes AAC's flags at 44.1 kHz, 16 bit, stereo; the real values are in the AudioSpecificConfig.
pub fn aac_seq(o: &mut Vec<u8>, asc: &[u8]) {
    o.extend_from_slice(&[0xAF, 0]);
    o.extend_from_slice(asc);
}

/// One raw AAC frame (no ADTS header).
pub fn aac(o: &mut Vec<u8>, raw: &[u8]) {
    o.extend_from_slice(&[0xAF, 1]);
    o.extend_from_slice(raw);
}

/// FLV file framing around tag bodies.
pub struct Writer<W: Write> {
    pub w: W,
}

impl<W: Write> Writer<W> {
    pub fn new(mut w: W, video: bool, audio: bool) -> io::Result<Self> {
        let f = (audio as u8) << 2 | video as u8;
        w.write_all(&[b'F', b'L', b'V', 1, f, 0, 0, 0, 9, 0, 0, 0, 0])?;
        Ok(Self { w })
    }

    /// `ty` is [`AUDIO`], [`VIDEO`] or [`DATA`]; `ts` is the dts.
    pub fn tag(&mut self, ty: u8, ts: u32, body: &[u8]) -> io::Result<()> {
        let n = body.len() as u32;
        let mut h = [0; 11];
        h[0] = ty;
        h[1..4].copy_from_slice(&n.to_be_bytes()[1..]);
        h[4..7].copy_from_slice(&ts.to_be_bytes()[1..]);
        h[7] = (ts >> 24) as u8;
        self.w.write_all(&h)?;
        self.w.write_all(body)?;
        self.w.write_all(&(n + 11).to_be_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb() {
        let b = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 0, 1, 0x65, 0, 0, 3, 1, 0,
        ];
        let v: Vec<_> = nals(&b).collect();
        assert_eq!(v, [&[0x67, 1, 2][..], &[0x68, 3], &[0x65, 0, 0, 3, 1]]);
        assert_eq!(nals(&[1, 2, 3]).count(), 0);
    }

    #[test]
    fn bodies() {
        let mut o = vec![];
        avc(&mut o, false, -33, [&[0x41u8, 9][..]]);
        assert_eq!(o, [0x27, 1, 0xFF, 0xFF, 0xDF, 0, 0, 0, 2, 0x41, 9]);
        o.clear();
        meta(
            &mut o,
            &Meta {
                w: 1664,
                h: 832,
                fps: 30.0,
                rate: 48000,
                ch: 2,
                ..Default::default()
            },
        );
        let v = amf::parse(&o);
        assert_eq!(v[0].str(), "onMetaData");
        assert_eq!(v[1].get("videocodecid").and_then(amf::V::num), Some(7.0));
        assert_eq!(v[1].get("stereo"), Some(&amf::V::Bool(true)));
        assert_eq!(v[1].get("videodatarate"), None);
        // the ECMA count matches the properties written
        assert_eq!(u32::from_be_bytes(o[14..18].try_into().unwrap()), 9);
    }
}
