//! UCD v1 framing (PROTOCOL.md §4): `u32 total_len, u8 type, u16 pad`, then the payload and `pad` zero bytes.

pub const SYNC: &[u8] = b"\x11\0\0\0\x06\0\0syNceNdinS";
pub const MARK: &[u8] = b"syNceNdinS";
pub const HB: &[u8] = b"\x07\0\0\0\x05\0\0";
/// Camera's answer to a sync while another client holds the session.
pub const BUSY: &[u8] = b"\x08\0\0\0\xb0\0\0\x01";
/// Largest request body per fragment: legacy budget, keeps every fragment below 2 MiB (§3.9).
pub const FRAG: usize = 2_097_135;

/// Appends one packet. A packet whose unpadded size is a multiple of 512 gets one pad byte, so no USB bulk
/// transfer ends on a max-packet boundary and the camera never needs a ZLP (§4.3, §8.3).
pub fn put(v: &mut Vec<u8>, t: u8, head: &[u8], body: &[u8]) {
    let n = 7 + head.len() + body.len();
    let pad = n.is_multiple_of(512) as usize;
    v.extend(((n + pad) as u32).to_le_bytes());
    v.push(t);
    v.extend((pad as u16).to_le_bytes());
    v.extend(head);
    v.extend(body);
    v.resize(v.len() + pad, 0);
}

/// Message packet (type 4): method, content type 2 (protobuf), `end | dir 0 | id`, reserved.
pub fn msg(v: &mut Vec<u8>, code: u16, id: u32, end: bool, body: &[u8]) {
    let mut h = [0; 9];
    h[..2].copy_from_slice(&code.to_le_bytes());
    h[2] = 2;
    h[3..7].copy_from_slice(&((end as u32) << 31 | id & 0x3FFF_FFFF).to_le_bytes());
    put(v, 4, &h, body)
}

/// SocketTunnel packet (type 7, §8.6).
pub fn tunnel(v: &mut Vec<u8>, c: u32, flags: u16, win: u64, data: &[u8]) {
    let mut h = [0; 14];
    h[..4].copy_from_slice(&c.to_le_bytes());
    h[4..6].copy_from_slice(&flags.to_le_bytes());
    h[6..].copy_from_slice(&win.to_le_bytes());
    put(v, 7, &h, data)
}

#[derive(Debug, PartialEq)]
pub enum Pkt<'a> {
    /// `note`: camera→phone direction bit, i.e. a notification; otherwise a response whose `code` is the status.
    Msg {
        code: u16,
        note: bool,
        id: u32,
        end: bool,
        body: &'a [u8],
    },
    Stream {
        kind: u8,
        ts: u64,
        data: &'a [u8],
    },
    Tunnel {
        c: u32,
        flags: u16,
        data: &'a [u8],
    },
    /// Heartbeat, sync, LinuxCmd, SyncMediaTime, unknown types.
    Other,
}

fn u16_(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn u32_(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}

/// The packet at the start of `b` and its size; `Ok(None)` while incomplete, `Err` on a desync (§3.7).
pub fn next(b: &[u8]) -> Result<Option<(Pkt<'_>, usize)>, ()> {
    if b.len() < 7 {
        return Ok(None);
    }
    let n = u32_(b) as usize;
    if n < 7 {
        return Err(());
    }
    if b.len() < n {
        return Ok(None);
    }
    let e = n
        .checked_sub(u16_(&b[5..]) as usize)
        .filter(|&e| e >= 7)
        .ok_or(())?;
    let p = &b[7..e];
    let k = match b[4] {
        4 if p.len() >= 9 => {
            let w = u32_(&p[3..]);
            Pkt::Msg {
                code: u16_(p),
                note: w >> 30 & 1 == 1,
                id: w & 0x3FFF_FFFF,
                end: w >> 31 == 1,
                body: &p[9..],
            }
        }
        1 if p.len() >= 9 => Pkt::Stream {
            kind: p[0],
            ts: u64::from_le_bytes(p[1..9].try_into().unwrap()),
            data: &p[9..],
        },
        7 if p.len() >= 14 => Pkt::Tunnel {
            c: u32_(p),
            flags: u16_(&p[4..]),
            data: &p[14..],
        },
        _ => Pkt::Other,
    };
    Ok(Some((k, n)))
}

/// Offset just past the sync marker.
pub fn mark(b: &[u8]) -> Option<usize> {
    b.windows(MARK.len())
        .position(|w| w == MARK)
        .map(|i| i + MARK.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        s.split_whitespace()
            .map(|x| u8::from_str_radix(x, 16).unwrap())
            .collect()
    }

    #[test]
    fn encode() {
        let mut v = vec![];
        msg(&mut v, 13, 1, true, &hex("08 02 18 64"));
        assert_eq!(
            v,
            hex("14 00 00 00 04 00 00 0d 00 02 01 00 00 80 00 00 08 02 18 64")
        );
        // Pad case: 496-byte body → n = 512 → 513 bytes on the wire, pad 1.
        v.clear();
        msg(&mut v, 8, 1, true, &[7; 496]);
        assert_eq!(
            (v.len(), &v[..7], v[512]),
            (513, &hex("01 02 00 00 04 01 00")[..], 0)
        );
        v.clear();
        tunnel(&mut v, 0, 8, 0xC0_0000, &[]);
        assert_eq!(
            v,
            hex("15 00 00 00 07 00 00 00 00 00 00 08 00 00 00 c0 00 00 00 00 00")
        );
        v.clear();
        let get = b"GET /DCIM/Camera01/IMG_x.insp HTTP/1.1\r\nHost: 127.0.0.1:9099\r\n\r\n";
        tunnel(&mut v, 0, 0x2d, 0xA0_0000, get);
        assert_eq!(
            &v[..21],
            &hex("55 00 00 00 07 00 00 00 00 00 00 2d 00 00 00 a0 00 00 00 00 00")[..]
        );
    }

    #[test]
    fn decode() {
        let b = hex(
            "16 00 00 00 04 00 00 03 20 02 05 00 00 c0 00 00 0a 04 10 50 18 64 07 00 00 00 05 00 00 12",
        );
        let (p, n) = next(&b).unwrap().unwrap();
        assert_eq!(n, 22);
        assert_eq!(
            p,
            Pkt::Msg {
                code: 8195,
                note: true,
                id: 5,
                end: true,
                body: &hex("0a 04 10 50 18 64")[..]
            }
        );
        assert_eq!(next(&b[22..]).unwrap(), Some((Pkt::Other, 7)));
        assert_eq!(next(&b[29..]).unwrap(), None);
        // Error response: status 500, not a notification.
        let b = hex("12 00 00 00 04 00 00 f4 01 02 01 00 00 80 00 00 08 03");
        assert_eq!(
            next(&b).unwrap().unwrap().0,
            Pkt::Msg {
                code: 500,
                note: false,
                id: 1,
                end: true,
                body: &[8, 3]
            }
        );
        // A padded packet strips its pad.
        let mut v = vec![];
        msg(&mut v, 200, 9, false, &[7; 496]);
        let Pkt::Msg { body, end, .. } = next(&v).unwrap().unwrap().0 else {
            panic!()
        };
        assert_eq!((body.len(), end), (496, false));
        assert!(next(&hex("06 00 00 00 05 00 00")).is_err());
        let s = hex("01 02 11 00 00 00 06 00 00 73 79 4e 63 65 4e 64 69 6e 53 aa");
        assert_eq!(mark(&s), Some(19));
    }
}
