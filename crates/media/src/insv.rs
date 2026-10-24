//! Insta360 trailer ("extra info" V3) of `.insv` / `.insp` / `.lrv` files (re/analysis/formats.md §3): the
//! `ExtraMetadata` protobuf (record 1), IMU samples (record 3), per-frame exposure (records 4 and 12) and raw access
//! to every other record. The parsers also take record payloads fetched remotely (`GET_FILE_EXTRA`).
//!
//! Verified on X5 footage (firmware 1.7.43): timestamps are µs on the camera clock; the IMU runs at ~1 kHz;
//! `pts_type` 2 (`VIDEO_PTS_EXPOSURE_FILE`): exposure item k with `t == first_frame` is video frame 0.

use crate::{bad, read_at};
use std::{fs::File, io, path::Path};

const MAGIC: &[u8; 32] = b"8db42d694ccc418790edff439fe026bf";

pub const METADATA: u8 = 1;
pub const THUMBNAIL: u8 = 2;
pub const GYRO: u8 = 3;
pub const EXPOSURE: u8 = 4;
pub const EXT_THUMBNAIL: u8 = 5;
pub const FRAME_PTS: u8 = 6;
pub const GPS: u8 = 7;
/// Exposure of the second sensor (`EXPOTIME_VIN1`).
pub const EXPOSURE_VIN1: u8 = 12;

/// Index entry; `off` is absolute in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rec {
    pub kind: u8,
    pub fmt: u8,
    pub off: u64,
    pub size: u32,
}

/// `file_group_info` (tag 26). The X5 sets `identify` to the main file's camera path in the main file and its
/// `.lrv` alike (index 0 and 1): that pairs them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Group {
    /// `SubMediaType` (0 VIDEO_NORMAL, 20 VIDEO_PURE, …)
    pub kind: u32,
    pub index: u32,
    pub identify: String,
    pub total: u32,
}

/// The `ExtraMetadata` fields the app uses. Tag numbers as #N; enums as raw values (formats_extra_info.proto).
#[derive(Clone, Debug, Default)]
pub struct Meta {
    pub serial: String,   // #1
    pub camera: String,   // #2 "Insta360 X5"
    pub firmware: String, // #3
    /// #7 local date and time as the decimal number YYYYMMDDhhmmss.
    pub created: u64,
    /// #117 seconds east of GMT.
    pub tz: i32,
    /// #93, else #10 × 1000.
    pub duration_ms: u32,
    /// #19 sensor-side size, #44 size of the stored stream (differs in the `.lrv`).
    pub dimension: [u32; 2],
    pub resolution: [u32; 2],
    /// #78 num/den, else #20.
    pub fps: f64,
    /// #40, 0 if absent.
    pub frames: u32,
    pub group: Group, // #26
    /// #129 1 plane, 2 dual fisheye, 3 fisheye, 4 wide angle, 5 dual half fisheye, 6 equirect.
    pub category: u32,
    /// #131 1 single, 2 dual file, 3 dual track, 4 dual track reversed (X5: track 0 = lens 1).
    pub stream: u32,
    /// #82 1 original, 2 stitched in camera.
    pub stitch: u32,
    /// #75 thumbnail codec: 1 YUV (NV12 on the X5), 2 JPEG, 3 H.264/H.265.
    pub thumb: u32,
    /// #132 1 H.264, 2 H.265, 3 MJPEG.
    pub codec: u32,
    /// Best calibration string: #111 offset_v6 > #54 v3 > #5 v1 (the app's order).
    pub offset: String,
    /// #136 OffsetVersion the camera captured with (4 = V6).
    pub offset_version: u32,
    /// #27 window_crop_info: square window cut from the calibration frame, and its offset from centre.
    pub crop: f32,
    pub shift: [f32; 2],
    /// #128 blend_Angle in degrees; the camera writes 0 and the app keeps its 6° default unless > 0.
    pub blend: Option<f64>,
    /// #62 gyro record holds raw 20-byte items (else 56-byte processed ones).
    pub raw_gyro: bool,
    /// #65 full-scale ranges for raw items: acc in g, gyro in °/s (defaults 8 / 2000).
    pub acc_range: u32,
    pub gyro_range: u32,
    /// #24 timestamp of video frame 0, µs.
    pub first_frame: u64,
    /// #25 sensor readout time in ms, texture row 0 first (21.24 normal video, 11.84 PureVideo; matches the video's
    /// rolling shutter, crate stab).
    pub rolling_shutter: f64,
    /// #64 1 MP4 timestamps, 2 frame times from the exposure record.
    pub pts_type: u32,
    /// #42 stabilised in camera.
    pub flowstate: bool,
    /// #88 1 not split, 2 split recording.
    pub split: u32,
}

/// One IMU sample: µs, acceleration in g, angular rate in rad/s, sensor axes (`stab::AXES` maps them to the lens
/// calibration frame).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Imu {
    pub t: u64,
    pub acc: [f32; 3],
    pub gyro: [f32; 3],
}

/// Start time (µs) and shutter time (s) of one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Exposure {
    pub t: u64,
    pub time: f64,
}

pub struct Trailer {
    f: File,
    /// Offset of the trailer block = end of the container data.
    pub start: u64,
    pub recs: Vec<Rec>,
    pub meta: Meta,
}

impl Trailer {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Trailer> {
        let f = File::open(path)?;
        let size = f.metadata()?.len();
        if size < 78 {
            return Err(bad("no trailer"));
        }
        let mut tail = [0; 78]; // last record header (6) + footer (72)
        read_at(&f, &mut tail, size - 78)?;
        if &tail[46..] != MAGIC || u32le(&tail[42..]) != 3 {
            return Err(bad("no V3 trailer"));
        }
        let total = u32le(&tail[38..]) as u64;
        if !(78..=size).contains(&total) {
            return Err(bad("bad trailer size"));
        }
        let start = size - total;
        let mut t = Trailer {
            f,
            start,
            recs: vec![],
            meta: Meta::default(),
        };
        if tail[1] == 0 {
            let n = u32le(&tail[2..]) as u64;
            if n > size - 78 - start {
                return Err(bad("bad trailer index"));
            }
            let mut idx = vec![0; n as usize];
            read_at(&t.f, &mut idx, size - 78 - n)?;
            for e in idx.as_chunks::<10>().0.iter().filter(|e| e[0] != 0) {
                let (off, n) = (start + u32le(&e[6..]) as u64, u32le(&e[2..]));
                // records lie inside the trailer; an untrusted size would make `get` allocate up to 4 GB
                if off + n as u64 <= size - 78 - idx.len() as u64 {
                    t.recs.push(Rec {
                        kind: e[0],
                        fmt: e[1],
                        off,
                        size: n,
                    });
                }
            }
        } else {
            // no index: walk the 6-byte headers backwards (they follow their payloads)
            let mut p = size - 72;
            while p >= start + 6 {
                let mut h = [0; 6];
                read_at(&t.f, &mut h, p - 6)?;
                let n = u32le(&h[2..]) as u64;
                if (h[1] == 0 && n == 0) || n > p - 6 - start {
                    break;
                }
                t.recs.push(Rec {
                    kind: h[1],
                    fmt: h[0],
                    off: p - 6 - n,
                    size: n as u32,
                });
                p -= 6 + n;
            }
        }
        let mut b = vec![];
        if t.get(METADATA, &mut b)? {
            t.meta = Meta::parse(&b);
        }
        Ok(t)
    }

    /// Payload of the first record of `kind` into `buf`; false if the file has none.
    pub fn get(&self, kind: u8, buf: &mut Vec<u8>) -> io::Result<bool> {
        let Some(r) = self.recs.iter().find(|r| r.kind == kind) else {
            return Ok(false);
        };
        buf.resize(r.size as usize, 0);
        read_at(&self.f, buf, r.off)?;
        Ok(true)
    }

    pub fn imu(&self) -> io::Result<Vec<Imu>> {
        let mut b = vec![];
        self.get(GYRO, &mut b)?;
        Ok(imu(&b, &self.meta))
    }

    /// Per-frame exposure from video frame 0 on: `kind` is [`EXPOSURE`] (sensor 0) or [`EXPOSURE_VIN1`].
    pub fn frames(&self, kind: u8) -> io::Result<Vec<Exposure>> {
        let mut b = vec![];
        self.get(kind, &mut b)?;
        let mut e = exposure(&b);
        // nearest item: sensor 1's stamps sit a few µs off sensor 0's
        let ff = self.meta.first_frame;
        let k = e.partition_point(|x| x.t < ff);
        let k = if k > 0 && e.get(k).is_none_or(|x| ff - e[k - 1].t < x.t - ff) {
            k - 1
        } else {
            k
        };
        e.drain(..k);
        Ok(e)
    }
}

/// Exposure record (16-byte items: u64 µs, f64 shutter seconds).
pub fn exposure(b: &[u8]) -> Vec<Exposure> {
    b.as_chunks::<16>()
        .0
        .iter()
        .map(|c| Exposure {
            t: u64le(c),
            time: f64::from_bits(u64le(&c[8..])),
        })
        .collect()
}

/// Gyro record. Raw items: u64 µs, then u16 acc x y z, gyro x y z, offset binary scaled by the #65 ranges (the
/// gyro scale matches the video's rotation within ±1.3 %, crate stab). The X5 writes them in batches of 16 stamped
/// 1 ms apart at a true 1002 Hz, so stamps inside a batch run up to ~30 µs fast.
/// Processed items (56 B): i64, then f64 acc x y z, gyro x y z. `arvbmg::ParseGyroData` (libarvbmg
/// @0x2c17f48) copies the six values into the `GyroData` fields its raw path fills with acc (g) and gyro (rad/s), and
/// scales both kinds of stamps by the same 1e-3, so they share the raw items' units; no X5 sample has them.
/// (formats.md's gyro-first order comes from the Java tool's decode of the single metadata sample #14.)
pub fn imu(b: &[u8], m: &Meta) -> Vec<Imu> {
    if m.raw_gyro {
        let (a, g) = (
            m.acc_range as f32 / 32768.0,
            m.gyro_range as f32 / 32768.0 * std::f32::consts::PI / 180.0,
        );
        let v =
            |c: &[u8], i: usize| u16::from_le_bytes([c[8 + 2 * i], c[9 + 2 * i]]) as f32 - 32768.0;
        b.as_chunks::<20>()
            .0
            .iter()
            .map(|c| Imu {
                t: u64le(c),
                acc: [v(c, 0) * a, v(c, 1) * a, v(c, 2) * a],
                gyro: [v(c, 3) * g, v(c, 4) * g, v(c, 5) * g],
            })
            .collect()
    } else {
        let v = |c: &[u8], i: usize| f64::from_bits(u64le(&c[8 + 8 * i..])) as f32;
        b.as_chunks::<56>()
            .0
            .iter()
            .map(|c| Imu {
                t: u64le(c),
                acc: [v(c, 0), v(c, 1), v(c, 2)],
                gyro: [v(c, 3), v(c, 4), v(c, 5)],
            })
            .collect()
    }
}

impl Meta {
    /// Parses the `ExtraMetadata` protobuf; unknown or malformed trailing fields are ignored.
    pub fn parse(md: &[u8]) -> Meta {
        let mut m = Meta {
            acc_range: 8,
            gyro_range: 2000,
            ..Default::default()
        };
        let (mut v1, mut v3, mut v6) = (None, None, None);
        let (mut secs, mut fq, mut fi) = (0, None, 0);
        let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
        let xy = |b: &[u8]| {
            let mut v = [0; 2];
            for (t, f) in fields(b) {
                if let (1..=2, F::Int(x)) = (t, f) {
                    v[t as usize - 1] = x as u32;
                }
            }
            v
        };
        for (tag, f) in fields(md) {
            match (tag, f) {
                (1, F::Bytes(b)) => m.serial = s(b),
                (2, F::Bytes(b)) => m.camera = s(b),
                (3, F::Bytes(b)) => m.firmware = s(b),
                (5, F::Bytes(b)) => v1 = Some(b),
                (54, F::Bytes(b)) => v3 = Some(b),
                (111, F::Bytes(b)) => v6 = Some(b),
                (7, F::Int(x)) => m.created = x,
                (10, F::Int(x)) => secs = x as u32,
                (93, F::Int(x)) => m.duration_ms = x as u32,
                (117, F::Int(x)) => m.tz = x as i32,
                (19, F::Bytes(b)) => m.dimension = xy(b),
                (44, F::Bytes(b)) => m.resolution = xy(b),
                (20, F::Int(x)) => fi = x as i32,
                (78, F::Bytes(b)) => {
                    let r = xy(b);
                    fq = (r[1] > 0).then(|| r[0] as f64 / r[1] as f64);
                }
                (40, F::Int(x)) => m.frames = x as u32,
                (26, F::Bytes(b)) => {
                    for (t, f) in fields(b) {
                        match (t, f) {
                            (1, F::Int(x)) => m.group.kind = x as u32,
                            (2, F::Int(x)) => m.group.index = x as u32,
                            (3, F::Bytes(b)) => m.group.identify = s(b),
                            (4, F::Int(x)) => m.group.total = x as u32,
                            _ => {}
                        }
                    }
                }
                (129, F::Int(x)) => m.category = x as u32,
                (131, F::Int(x)) => m.stream = x as u32,
                (82, F::Int(x)) => m.stitch = x as u32,
                (75, F::Int(x)) => m.thumb = x as u32,
                (132, F::Int(x)) => m.codec = x as u32,
                (136, F::Int(x)) => m.offset_version = x as u32,
                (27, F::Bytes(w)) => {
                    for (t, f) in fields(w) {
                        match (t, f) {
                            (3, F::Int(x)) => m.crop = x as f32,
                            (5, F::Int(x)) => m.shift[0] = x as i32 as f32,
                            (6, F::Int(x)) => m.shift[1] = x as i32 as f32,
                            _ => {}
                        }
                    }
                }
                (128, F::Int(a)) if a as i32 > 0 => m.blend = Some(a as i32 as f64),
                (62, F::Int(x)) => m.raw_gyro = x != 0,
                (65, F::Bytes(b)) => {
                    let r = xy(b);
                    if r[0] > 0 {
                        m.acc_range = r[0];
                    }
                    if r[1] > 0 {
                        m.gyro_range = r[1];
                    }
                }
                (24, F::Int(x)) => m.first_frame = x,
                (25, F::F64(x)) => m.rolling_shutter = x,
                (64, F::Int(x)) => m.pts_type = x as u32,
                (42, F::Int(x)) => m.flowstate = x != 0,
                (88, F::Int(x)) => m.split = x as u32,
                _ => {}
            }
        }
        m.fps = fq.unwrap_or(fi as f64);
        if m.duration_ms == 0 {
            m.duration_ms = secs * 1000;
        }
        m.offset = [v6, v3, v1]
            .into_iter()
            .flatten()
            .find(|b| !b.is_empty())
            .map(s)
            .unwrap_or_default();
        m
    }
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}

fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

enum F<'a> {
    Int(u64),
    F64(f64),
    Bytes(&'a [u8]),
    Other,
}

/// Minimal protobuf wire reader; stops at the first malformed field.
fn fields(mut b: &[u8]) -> impl Iterator<Item = (u64, F<'_>)> {
    fn varint(b: &mut &[u8]) -> Option<u64> {
        let mut v = 0;
        for s in (0..64).step_by(7) {
            let (&c, r) = b.split_first()?;
            *b = r;
            v |= ((c & 0x7f) as u64) << s;
            if c < 0x80 {
                return Some(v);
            }
        }
        None
    }
    std::iter::from_fn(move || {
        let k = varint(&mut b)?;
        let f = match k & 7 {
            0 => F::Int(varint(&mut b)?),
            1 => {
                let (x, r) = b.split_first_chunk::<8>()?;
                b = r;
                F::F64(f64::from_le_bytes(*x))
            }
            5 => {
                b = b.get(4..)?;
                F::Other
            }
            2 => {
                let n = varint(&mut b)? as usize;
                let (v, r) = (b.get(..n)?, b.get(n..)?);
                b = r;
                F::Bytes(v)
            }
            _ => return None,
        };
        Some((k >> 3, f))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mp4::Mp4, tests::samples};

    #[test]
    fn x5_trailer() {
        let Some(d) = samples() else { return };
        let mut ids: Vec<(&str, String, u32, f64)> = vec![];
        for name in ["v635.insv", "v606.insv", "v627.insv", "v635.lrv"] {
            let p = d.join(name);
            let (t, m) = (Trailer::open(&p).unwrap(), Mp4::open(&p).unwrap());
            let (mt, v) = (&t.meta, &m.tracks[0]);
            let lrv = name.ends_with(".lrv");
            // the trailer starts where the MP4 data ends (in a top-level `inst` box)
            let inst = m.file.metadata().unwrap().len() - t.start;
            assert!(
                inst > 0 && t.recs.iter().all(|r| r.off >= t.start),
                "{name}"
            );
            assert_eq!(mt.camera, "Insta360 X5", "{name}");
            assert!(
                mt.firmware.starts_with('v') && !mt.serial.is_empty(),
                "{name}"
            );
            assert!(
                mt.offset.split('_').count() > 10,
                "{name}: calibration string"
            );
            assert_eq!(mt.resolution, [v.width, v.height], "{name}");
            assert_eq!(mt.stream, if lrv { 1 } else { 4 }, "{name}");
            assert_eq!(mt.category, 2, "{name}");
            // total_time_ms is within one frame of the MP4 duration (v606: one frame longer); the .lrv carries its
            // main file's metadata, while its own track is 2 frames shorter
            let (movie, fps) = (
                m.duration as f64 / m.timescale as f64,
                v.timescale as f64 / v.samples[0].dur as f64,
            );
            let ref_s = if lrv { ids[0].3 } else { movie };
            assert!(
                (mt.duration_ms as f64 / 1e3 - ref_s).abs() < 1.0 / fps + 1e-3,
                "{name}: {} vs {ref_s}",
                mt.duration_ms
            );
            assert!((mt.fps - fps).abs() < 0.05, "{name}: {} vs {fps}", mt.fps);
            assert!(mt.raw_gyro && mt.pts_type == 2, "{name}");

            // frame timeline = exposure record from first_frame on, one item per video frame (plus a few spare)
            let f = t.frames(EXPOSURE).unwrap();
            assert_eq!(f[0].t, mt.first_frame, "{name}"); // exact for sensor 0
            assert!(
                f.len() >= v.samples.len() && f.len() <= v.samples.len() + 8,
                "{name}: {} items",
                f.len()
            );
            let n = v.samples.len() - 1;
            let dt = (f[n].t - f[0].t) as f64 / n as f64;
            let want = 1e6 * v.samples[0].dur as f64 / v.timescale as f64;
            assert!(
                (dt / want - 1.0).abs() < 1e-3,
                "{name}: frame interval {dt} µs vs {want}"
            );
            assert!(
                f.iter().all(|e| e.time > 0.0 && e.time < 1.0 / 30.0),
                "{name}: shutter"
            );
            let f1 = t.frames(EXPOSURE_VIN1).unwrap();
            if !f1.is_empty() {
                assert!(
                    f1.len() == f.len() && f1.iter().zip(&f).all(|(a, b)| a.t.abs_diff(b.t) < 100),
                    "{name}: vin1"
                );
            }

            let imu = t.imu().unwrap();
            let span = (imu.last().unwrap().t - imu[0].t) as f64 * 1e-6;
            let rate = (imu.len() - 1) as f64 / span;
            assert!((rate - 1000.0).abs() < 10.0, "{name}: imu {rate} Hz");
            assert!(
                imu[0].t <= f[0].t && imu.last().unwrap().t >= f[n].t,
                "{name}: imu covers the video"
            );
            let g = imu
                .iter()
                .map(|s| s.acc.iter().map(|a| a * a).sum::<f32>().sqrt())
                .sum::<f32>()
                / imu.len() as f32;
            assert!((0.9..1.1).contains(&g), "{name}: mean |acc| {g} g");
            let wmax = imu
                .iter()
                .flat_map(|s| s.gyro)
                .fold(0f32, |a, b| a.max(b.abs()));
            assert!(wmax < (mt.gyro_range as f32).to_radians(), "{name}");
            println!(
                "{name}: {} {} {}×{} {:.3} fps {} ms (MP4 {movie:.3} s), group {:?}, stream {}, offset v{} ({} chars), {} recs, \
                 {} frames ({} video), frame dt {dt:.1} µs, shutter {:.5}–{:.5} s, rolling shutter {} ms, \
                 imu {} @ {rate:.1} Hz, mean |acc| {g:.3} g, max |ω| {wmax:.2} rad/s, acc/gyro range {}/{}",
                mt.camera,
                mt.firmware,
                mt.resolution[0],
                mt.resolution[1],
                mt.fps,
                mt.duration_ms,
                (mt.group.kind, mt.group.index, mt.group.total),
                mt.stream,
                mt.offset_version,
                mt.offset.len(),
                t.recs.len(),
                f.len(),
                v.samples.len(),
                f.iter().map(|e| e.time).fold(f64::MAX, f64::min),
                f.iter().map(|e| e.time).fold(0.0, f64::max),
                mt.rolling_shutter,
                imu.len(),
                mt.acc_range,
                mt.gyro_range,
            );
            ids.push((name, mt.group.identify.clone(), mt.group.index, movie));
        }
        // LRV pairing: the proxy carries its main file's identify, at index 1
        let (main, lrv) = (&ids[0], &ids[3]);
        assert!(!main.1.is_empty() && main.1 == lrv.1 && (main.2, lrv.2) == (0, 1));
        assert!(ids[..3].iter().all(|x| x.1.ends_with(".insv")) && ids[0].1 != ids[1].1);
    }
}
