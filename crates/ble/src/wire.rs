//! Bytes on and around the BLE link: UCD v1 packets (PROTOCOL.md §4; the subset of `link::ucd` this crate needs),
//! the UCP "Go2BLE" wrapper (§2.5.3), the phone's wake advertisement (§2.6), and advertised names, iBeacons and the
//! X5's Wi-Fi state record.

/// Message content per UCD packet: the app's BLE budget 206 − 16 (§2.4), so each packet fits one 206-byte write.
pub const SPAN: usize = 190;
/// The same inside UCP frames, which add 7 bytes (`UcpCodec` 183).
pub const SPAN_UCP: usize = 183;
/// Largest GATT write the app makes, whatever the MTU (§2.3).
pub const CHUNK: usize = 206;
/// Company id of the wake advertisement's manufacturer record (Apple, iBeacon layout).
pub const APPLE: u16 = 0x004C;
/// Company id (not assigned to Insta360) of the record in an X5's `X5 <token>` advertisement, `00 tt vv` ([`Proto`]);
/// the app's wake advertisement for GO cameras uses it too (ble.md §3).
pub const INSTA: u16 = 0x1A2B;

/// Appends one packet; one pad byte when the unpadded size is a multiple of 512 (§4.3).
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

/// A request as frames to write: packets of at most [`SPAN`] (UCP: [`SPAN_UCP`]) body bytes, same id, `end` on the
/// last (§4.8); each wrapped in a UCP frame when `ucp`.
pub fn frames(out: &mut Vec<Vec<u8>>, code: u16, id: u32, body: &[u8], ucp: bool) {
    let span = if ucp { SPAN_UCP } else { SPAN };
    let n = body.len().div_ceil(span).max(1);
    for (i, c) in body
        .chunks(span)
        .chain(body.is_empty().then_some(&[][..]))
        .enumerate()
    {
        let mut p = vec![];
        msg(&mut p, code, id, i + 1 == n, c);
        out.push(if ucp {
            let mut f = vec![];
            frame(&mut f, 7, 0x40, &p);
            f
        } else {
            p
        })
    }
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
    /// Heartbeat, sync, stream and every other type.
    Other,
}

fn u16_(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

/// The packet at the start of `b` and its size; `Ok(None)` while incomplete, `Err` on a desync (§3.7). Lengths
/// above 4 MiB count as a desync: nothing the camera sends is that large, and waiting for it would stall the link.
pub fn next(b: &[u8]) -> Result<Option<(Pkt<'_>, usize)>, ()> {
    if b.len() < 7 {
        return Ok(None);
    }
    let n = u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
    if !(7..=1 << 22).contains(&n) {
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
            let w = u32::from_le_bytes(p[3..7].try_into().unwrap());
            Pkt::Msg {
                code: u16_(p),
                note: w >> 30 & 1 == 1,
                id: w & 0x3FFF_FFFF,
                end: w >> 31 == 1,
                body: &p[9..],
            }
        }
        _ => Pkt::Other,
    };
    Ok(Some((k, n)))
}

/// CRC-16/MODBUS (init 0xFFFF, reflected polynomial 0xA001), the UCP frame check.
pub fn crc(b: &[u8]) -> u16 {
    let mut c = 0xFFFFu16;
    for &x in b {
        c ^= x as u16;
        for _ in 0..8 {
            c = if c & 1 == 1 { c >> 1 ^ 0xA001 } else { c >> 1 };
        }
    }
    c
}

/// UCP frame: `ff dir type len:u16 payload crc:u16`, all little-endian, CRC over everything before it.
pub fn frame(v: &mut Vec<u8>, dir: u8, t: u8, p: &[u8]) {
    let s = v.len();
    v.extend([0xFF, dir, t]);
    v.extend((p.len() as u16).to_le_bytes());
    v.extend(p);
    let c = crc(&v[s..]);
    v.extend(c.to_le_bytes());
}

/// The phone's answer to the camera's sync frame (`ff 06 41`): legacy layout, five `ab ba`.
pub fn sync(v: &mut Vec<u8>) {
    frame(v, 7, 0x41, &[0xAB, 0xBA].repeat(5))
}

/// A UCP frame: direction, type, payload. The frame is 7 bytes longer than its payload.
pub type Frame<'a> = (u8, u8, &'a [u8]);

/// The UCP frame at the start of `b`; `Ok(None)` while incomplete, `Err` when `b` does not start with a valid frame.
pub fn unframe(b: &[u8]) -> Result<Option<Frame<'_>>, ()> {
    match b {
        [] => Ok(None),
        [0xFF, ..] if b.len() < 5 => Ok(None),
        [0xFF, d, t, l0, l1, ..] => {
            let n = 7 + u16::from_le_bytes([*l0, *l1]) as usize;
            if b.len() < n {
                return Ok(None);
            }
            (crc(&b[..n - 2]) == u16_(&b[n - 2..]))
                .then_some(Some((*d, *t, &b[5..n - 2])))
                .ok_or(())
        }
        _ => Err(()),
    }
}

/// Manufacturer data (company [`APPLE`]) of the phone's wake advertisement for the camera whose serial ends in `s`
/// (§2.6): an iBeacon whose UUID is `"\tORBIT\t\xff\x0f\0"` plus `s` zero-padded, major and minor 0, tx −28, then
/// `01`. `None` when `s` is longer than 6 bytes (the app fails the same way). Sent through BlueZ only.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn wake(s: &str) -> Option<[u8; 24]> {
    let s = s.as_bytes();
    (s.len() <= 6).then(|| {
        let mut d = [0; 24];
        d[..12].copy_from_slice(b"\x02\x15\tORBIT\t\xff\x0f\0");
        d[12..12 + s.len()].copy_from_slice(s);
        d[22] = 0xE4;
        d[23] = 1;
        d
    })
}

/// An iBeacon in an Apple manufacturer record (`02 15 uuid major minor tx`, major / minor big-endian).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Beacon {
    pub uuid: [u8; 16],
    pub major: u16,
    pub minor: u16,
    pub tx: i8,
}

impl Beacon {
    /// Major of an X5's own iBeacon (`IBEACON_MAJOR_X_PANO_X5`); minor 1 asks for a fast transfer, 3 is a general
    /// wakeup, 6 AI memory (ble.md §2.4, inferred from config: the camera side was never observed).
    pub const X5: u16 = 18;

    pub fn parse(d: &[u8]) -> Option<Beacon> {
        let d = d.strip_prefix(&[2, 0x15])?.get(..21)?;
        Some(Beacon {
            uuid: d[..16].try_into().unwrap(),
            major: u16::from_be_bytes([d[16], d[17]]),
            minor: u16::from_be_bytes([d[18], d[19]]),
            tx: d[20] as i8,
        })
    }
}

/// What the app (FastBle `BleScanProto`) reads from an X5's `1a2b` record `00 tt vv`: format 0, then `tt` the
/// seconds its Wi-Fi stays up and `vv` the Wi-Fi password version; a fourth byte would be the net type (3: Wi-Fi
/// Aware), which the X5 does not send. Format 0 has no power state: the app takes the camera as booted
/// (`bootUpStatus` 1) whatever it does, and the X5 keeps advertising the record in standby. Seen on a real X5:
/// `00 ff 01` and `00 78 01` (on USB), `00 64 01` (awake, AP up), `00 52 01` and `00 06 01` (standby, 82 s and 6 s
/// after an AP session).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proto {
    /// Seconds the camera's access point stays up, counted from this advertisement (`wifiActiveTimeS`): 0 off, 255
    /// not started since boot (`WIFI_UNINITIALIZED`). The app joins a saved AP directly, without BLE, while more than
    /// 30 s are left and [`Proto::pwd`] equals the saved password's version.
    pub wifi: u8,
    /// The camera's version number of the current AP password.
    pub pwd: u8,
}

impl Proto {
    /// From the record's data (after the company id).
    pub fn parse(d: &[u8]) -> Option<Proto> {
        match *d {
            [0, wifi, pwd] | [0, wifi, pwd, _] => Some(Proto { wifi, pwd }),
            _ => None,
        }
    }
}

/// Model and serial token of an advertised camera name (PROTOCOL.md §1.2): `_` reads as a space, the last token is
/// the serial suffix, the rest the model (`"X5"`, `"X4 Air"`, `"X4_Air"`).
pub fn model(name: &str) -> Option<(&str, &str)> {
    name.trim()
        .rsplit_once([' ', '_'])
        .filter(|(m, t)| !m.is_empty() && !t.is_empty())
}

/// Whether an advertised name belongs to the camera with this serial: with every `-` and the digits 1–9 after it
/// removed (`InstaCameraUtils.filterCameraName`) and trimmed, it ends with the serial's last six characters.
pub fn is(name: &str, serial: &str) -> bool {
    let mut s = String::with_capacity(name.len());
    let mut skip = false;
    for c in name.chars() {
        skip = c == '-' || skip && ('1'..='9').contains(&c);
        if !skip {
            s.push(c)
        }
    }
    let k = serial.len().saturating_sub(6);
    serial
        .get(k..)
        .is_some_and(|t| !t.is_empty() && s.trim().ends_with(t))
}

#[cfg(test)]
pub(crate) fn hex(s: &str) -> Vec<u8> {
    s.split_whitespace()
        .map(|x| u8::from_str_radix(x, 16).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucd() {
        // Plain heartbeat (ble.md §10).
        let mut v = vec![];
        put(&mut v, 5, &[], &[]);
        assert_eq!(v, hex("07 00 00 00 05 00 00"));
        // Response header, status 200, id 5 (ble.md §10).
        let b = hex("10 00 00 00 04 00 00 c8 00 02 05 00 00 80 00 00");
        assert_eq!(
            next(&b).unwrap(),
            Some((
                Pkt::Msg {
                    code: 200,
                    note: false,
                    id: 5,
                    end: true,
                    body: &[]
                },
                16
            ))
        );
        // Notification 8215 CAM_WIFI_START, camera id 0x123456 (ble.md §10: `17 20 02 <id | 0x40000000 | end>`).
        let b = hex("10 00 00 00 04 00 00 17 20 02 56 34 12 c0 00 00");
        assert_eq!(
            next(&b).unwrap().unwrap().0,
            Pkt::Msg {
                code: 8215,
                note: true,
                id: 0x123456,
                end: true,
                body: &[]
            }
        );
        assert_eq!(next(&b[..15]).unwrap(), None);
        assert!(next(&hex("06 00 00 00 05 00 00")).is_err());
        assert!(next(&hex("00 00 80 00 04 00 00")).is_err());
        // Sync packet (§4.10): framed past, ignored.
        let s = hex("11 00 00 00 06 00 00 73 79 4e 63 65 4e 64 69 6e 53");
        assert_eq!(next(&s).unwrap(), Some((Pkt::Other, 17)));
    }

    #[test]
    fn fragments() {
        let body: Vec<u8> = (0..400).map(|i| i as u8).collect();
        for (ucp, span) in [(false, SPAN), (true, SPAN_UCP)] {
            let mut out = vec![];
            frames(&mut out, 8, 3, &body, ucp);
            assert_eq!(out.len(), 3);
            let mut got = vec![];
            for (i, f) in out.iter().enumerate() {
                assert!(f.len() <= CHUNK, "{} > {CHUNK}", f.len());
                let p = if ucp {
                    let (d, t, p) = unframe(f).unwrap().unwrap();
                    assert_eq!((d, t, p.len() + 7), (7, 0x40, f.len()));
                    p
                } else {
                    f
                };
                let Pkt::Msg {
                    code,
                    id,
                    end,
                    body,
                    ..
                } = next(p).unwrap().unwrap().0
                else {
                    panic!()
                };
                assert_eq!((code, id, end), (8, 3, i == 2));
                assert!(body.len() <= span);
                got.extend_from_slice(body);
            }
            assert_eq!(got, body);
        }
        // An empty body is still one packet.
        let mut out = vec![];
        frames(&mut out, 150, 7, &[], false);
        assert_eq!(
            out,
            [hex("10 00 00 00 04 00 00 96 00 02 07 00 00 80 00 00")]
        );
        // Pad case (§4.10): 496-byte packet body → 513 bytes, pad 1.
        let mut v = vec![];
        msg(&mut v, 8, 1, true, &[7; 496]);
        assert_eq!(
            (v.len(), &v[..7], v[512]),
            (513, &hex("01 02 00 00 04 01 00")[..], 0)
        );
    }

    #[test]
    fn ucp() {
        let mut v = vec![];
        sync(&mut v);
        assert_eq!(v, hex("ff 07 41 0a 00 ab ba ab ba ab ba ab ba ab ba 4e 23"));
        v.clear();
        frame(&mut v, 7, 0x41, &[0xAB, 0xBA].repeat(8));
        assert_eq!(
            v,
            hex("ff 07 41 10 00 ab ba ab ba ab ba ab ba ab ba ab ba ab ba ab ba d2 74")
        );
        v.clear();
        frame(&mut v, 7, 0x40, &hex("07 00 00 00 05 00 00"));
        assert_eq!(v, hex("ff 07 40 07 00 07 00 00 00 05 00 00 6b 46"));
        assert_eq!(unframe(&v).unwrap().unwrap(), (7, 0x40, &v[5..12]));
        assert_eq!(unframe(&v[..13]).unwrap(), None);
        // The app's literal disconnect frame is one byte short of its length field; the CRC it would need is `54 29`.
        let mut d = hex("ff 0c 01 01 00 00 cc");
        assert_eq!(crc(&d[..6]), 0x2954);
        assert_eq!(unframe(&d).unwrap(), None);
        d.push(0);
        assert!(unframe(&d).is_err());
        assert!(unframe(&hex("07 00")).is_err());
    }

    #[test]
    fn advert() {
        // Wake payload for suffix ABCDEF (ble.md §3) and the raw AD structure (PROTOCOL.md §2.6).
        let d = wake("ABCDEF").unwrap();
        assert_eq!(
            d[..],
            hex("02 15 09 4f 52 42 49 54 09 ff 0f 00 41 42 43 44 45 46 00 00 00 00 e4 01")
        );
        let mut ad = vec![0x1B, 0xFF];
        ad.extend(APPLE.to_le_bytes());
        ad.extend(d);
        assert_eq!(
            ad,
            hex(
                "1b ff 4c 00 02 15 09 4f 52 42 49 54 09 ff 0f 00 41 42 43 44 45 46 00 00 00 00 e4 01"
            )
        );
        assert_eq!(&wake("AB").unwrap()[12..18], b"AB\0\0\0\0");
        assert_eq!(wake("ABCDEFG"), None);
        // Our own wake advert reads as an iBeacon with major / minor 0.
        let b = Beacon::parse(&d).unwrap();
        assert_eq!(
            (b.major, b.minor, b.tx, &b.uuid[..6]),
            (0, 0, -28, &b"\tORBIT"[..])
        );
        // The camera's "fast transfer" beacon (PROTOCOL.md §11 X12: `4c 00 02 15 <uuid> 00 12 00 01`).
        let mut c = hex("02 15");
        c.extend([0x11; 16]);
        c.extend(hex("00 12 00 01 c5"));
        let b = Beacon::parse(&c).unwrap();
        assert_eq!((b.major, b.minor, b.tx), (Beacon::X5, 1, -59));
        assert_eq!(Beacon::parse(&c[..22]), None);
        assert_eq!(Beacon::parse(&hex("02 01 00")), None);
    }

    #[test]
    fn proto() {
        // Records a real X5 sent (`BleScanProto` format 0, five bytes with the company id): AP up for another 100 s,
        // password version 1; 255 = Wi-Fi not started since boot.
        assert_eq!(
            Proto::parse(&hex("00 64 01")),
            Some(Proto { wifi: 100, pwd: 1 })
        );
        assert_eq!(Proto::parse(&hex("00 ff 01")).unwrap().wifi, 255);
        // The six-byte variant's net type is not part of the X5's state.
        assert_eq!(
            Proto::parse(&hex("00 06 01 03")),
            Some(Proto { wifi: 6, pwd: 1 })
        );
        // The BSA advertisement's `000f: 00 00 00` would read as Wi-Fi off; other formats are not the X5's.
        assert_eq!(
            Proto::parse(&hex("00 00 00")),
            Some(Proto { wifi: 0, pwd: 0 })
        );
        for d in ["01 64 01 01", "02 00 01", "00 64", "00 64 01 00 00", ""] {
            assert_eq!(Proto::parse(&hex(d)), None, "{d}");
        }
    }

    #[test]
    fn names() {
        assert_eq!(model("X5 ABCDEF"), Some(("X5", "ABCDEF")));
        assert_eq!(model("X4 Air ABCDEF"), Some(("X4 Air", "ABCDEF")));
        assert_eq!(model("X5_ABCDEF"), Some(("X5", "ABCDEF")));
        assert_eq!(model("X5"), None);
        assert_eq!(model("X5 "), None);
        assert!(is("X5 ABCDEF", "IAHxx00ABCDEF"));
        assert!(is("X5 ABCDEF-2", "IAHxx00ABCDEF"));
        assert!(is("X5 ABCDEF- ", "ABCDEF"));
        assert!(!is("X5 ABCDEF", "IAHxx00ABCDEG"));
        assert!(!is("X5 ABCDEF", ""));
        // `-0` is not a copy marker: the zero stays and the name no longer ends with the suffix.
        assert!(!is("X5 ABCDEF-0", "ABCDEF"));
    }
}
