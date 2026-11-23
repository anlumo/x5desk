//! A simulated X5 for runs without the camera (tests, the `replay=` hook): link's simulated camera, so the real I/O
//! thread, framing, session and the app's workers all run. It answers what the app asks (identity, options with the
//! live calibration and the media clock, capture status, params, `/CONF`) and, on START_LIVE_STREAM,
//! streams a recorded live preview (`x5 live` output: `<p>.h264` with `<p>.timestamps.txt`, IMU in `<p>.30`) at its
//! recorded timestamps in real time, looping, on its own camera clock, in pairs as the X5 sends it over USB. Writes and capture commands are logged and
//! answered 200, nothing more.

use crate::camera::Connect;
use link::{Camera, Sim, code, pb, stream};
use prost::Message;
use std::{
    ops::Range,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    thread,
    time::{Duration, Instant},
};

/// A recorded preview: per video packet its camera ms and bytes in `es`; the IMU samples (56 bytes each).
pub struct Rec {
    pub aus: Vec<(u64, Range<usize>)>,
    pub es: Vec<u8>,
    pub imu: Vec<u8>,
}

impl Rec {
    /// `<p>.h264` and its `.timestamps.txt` ("ts bytes" per packet); `.30` if there.
    pub fn load(p: &Path) -> Result<Rec, String> {
        let r = |e: &str| {
            let f = p.with_extension(e);
            std::fs::read(&f).map_err(|x| format!("{}: {x}", f.display()))
        };
        let (es, ts) = (r("h264")?, r("timestamps.txt")?);
        let mut o = 0;
        let mut aus = vec![];
        for l in String::from_utf8_lossy(&ts).lines() {
            let mut w = l.split_whitespace().map(|x| x.parse::<u64>());
            let (Some(Ok(t)), Some(Ok(n))) = (w.next(), w.next()) else {
                return Err(format!("timestamps: bad line {l:?}"));
            };
            let n = n as usize;
            if o + n > es.len() {
                return Err("timestamps: more bytes than the stream".into());
            }
            aus.push((t, o..o + n));
            o += n;
        }
        let imu = r("30").unwrap_or_default();
        Ok(Rec { aus, es, imu })
    }

    /// Recorded ms from the first stamped packet to the last.
    fn span(&self) -> u64 {
        let t = self.aus.iter().map(|a| a.0).filter(|&t| t > 0);
        t.clone().max().unwrap_or(0) - t.min().unwrap_or(0)
    }
}

/// What the simulated camera serves.
pub struct Replay {
    pub rec: Rec,
    /// the v3 calibration (setting `calib_v3`) and the live window crop (side, offset) it was calibrated with
    pub cal: String,
    pub crop: (u32, [i32; 2]),
    pub conf: Option<Vec<u8>>,
    /// every request: code and body (tests)
    pub log: Mutex<Vec<(u16, Vec<u8>)>>,
    /// the battery level it reports, %
    pub battery: u32,
}

impl Replay {
    /// The recording `h264`, the calibration of `insv` (a file from the same camera: its v3 calibration string, which the
    /// X5 also reports as setting `calib_v3`, and its window crop), `/CONF` from `conf`.
    pub fn open(h264: &Path, insv: &Path, conf: Option<&Path>) -> Result<Replay, String> {
        let t = media::insv::Trailer::open(insv).map_err(|e| format!("{}: {e}", insv.display()))?;
        let mut b = vec![];
        t.get(media::insv::METADATA, &mut b)
            .map_err(|e| e.to_string())?;
        let x = pb::Metadata::decode(&b[..]).map_err(|e| e.to_string())?;
        Ok(Replay {
            rec: Rec::load(h264)?,
            cal: x.calib_v3,
            crop: (
                t.meta.crop as u32,
                [t.meta.shift[0] as i32, t.meta.shift[1] as i32],
            ),
            conf: conf
                .map(|c| std::fs::read(c).map_err(|e| format!("{}: {e}", c.display())))
                .transpose()?,
            log: Mutex::new(vec![]),
            battery: 80,
        })
    }
}

/// The simulated camera's clock: ms since it booted 278 s before the session (a real one's clock is its uptime).
fn clock(e: Instant) -> u64 {
    278_000 + e.elapsed().as_millis() as u64
}

/// Streams the recording until `stop`, each packet stamped on the camera clock and sent at the recorded time of the
/// odd one of its pair: the X5 holds every other frame back for the next (half its packet intervals under 17 ms,
/// measured 2026-10-25). The IMU in packets of 40 samples, each sent once its last sample is due. The packets stamped
/// 0 that lead a recording (right after a boot, its first IDR among them) go out at once and stamped 0 in the first
/// round, and a frame apart before the round's first stamped packet in the later ones: every round starts on that IDR.
fn stream(r: Arc<Replay>, s: Sim, e: Instant, stop: Arc<AtomicBool>) {
    let rec = &r.rec;
    let (c0, t0) = (clock(e), Instant::now());
    let first = rec.aus.iter().map(|a| a.0).find(|&t| t > 0).unwrap_or(0);
    let z = rec.aus.iter().take_while(|a| a.0 == 0).count();
    // the recorded time of packet j, a leading one stamped 0 counted a frame apart before the first stamped one
    let rt = |j: usize| match rec.aus[j].0 {
        0 => first.saturating_sub(33 * z.saturating_sub(j) as u64),
        t => t,
    };
    let lap = rec.span() + 33 * (z as u64 + 1);
    let imu = rec.imu.as_chunks::<{ 56 * 40 }>().0;
    let its = |c: &[u8]| i64::from_le_bytes(c[c.len() - 56..][..8].try_into().unwrap()) as u64;
    let note = pb::LiveNote {
        width: 2656,
        height: 1328,
        fps: 30,
        crop: Some(pb::Crop {
            src_w: 5376,
            src_h: 5376,
            dst_w: r.crop.0,
            dst_h: r.crop.0,
            x: r.crop.1[0],
            y: r.crop.1[1],
        }),
    };
    s.note(8234, &note.encode_to_vec());
    for k in 0u64.. {
        let (mut gi, sh) = (0, k * lap);
        // ms after t0 of packet j (lap ≥ 33 (z + 1) keeps it from going negative); None: at once, stamped 0
        let at = |j: usize| {
            (rec.aus.get(j))
                .filter(|a| k > 0 || a.0 > 0)
                .map(|_| rt(j) + sh - first)
        };
        for (j, (_, b)) in rec.aus.iter().enumerate() {
            let rel = at(j);
            let send = rel.map_or(0, |r| at(j | 1).map_or(r, |o| o.max(r)));
            if let Some(d) =
                (t0 + Duration::from_millis(send)).checked_duration_since(Instant::now())
            {
                thread::sleep(d);
            }
            if stop.load(Relaxed) {
                return;
            }
            // the IMU due by this frame, its stamps moved onto the camera clock as the frame's
            while gi < imu.len() && its(&imu[gi]) <= rel.map_or(first, |_| rt(j)) {
                let mut b = imu[gi];
                for c in b.as_chunks_mut::<56>().0 {
                    let x = i64::from_le_bytes(c[..8].try_into().unwrap()) as u64;
                    c[..8].copy_from_slice(
                        &((x + c0 + sh).saturating_sub(first) as i64).to_le_bytes(),
                    );
                }
                s.frame(stream::GYRO, 0, &b);
                gi += 1;
            }
            let ts = rel.map_or(0, |r| c0 + r);
            s.frame(stream::VIDEO, ts, &rec.es[b.clone()]);
        }
    }
}

/// Answers like the X5 for what the app sends.
fn handler(
    r: Arc<Replay>,
    sim: Arc<Mutex<Option<Sim>>>,
) -> impl FnMut(u16, &[u8]) -> (u16, Vec<u8>) + Send + 'static {
    let (e, mut live) = (Instant::now(), None::<Arc<AtomicBool>>);
    move |c, b| {
        r.log.lock().unwrap().push((c, b.to_vec()));
        let ok = |m: &dyn Fn() -> Vec<u8>| (200, m());
        match c {
            code::GET_OPTIONS => ok(&|| {
                let o = pb::Settings {
                    model: "Insta360 X5".into(),
                    serial: "SIM0001".into(),
                    firmware: "v1.13.21".into(),
                    battery: Some(pb::Battery {
                        level: r.battery,
                        ..Default::default()
                    }),
                    storage: Some(pb::Storage {
                        free: 400_000_000_000,
                        total: 512_000_000_000,
                        ..Default::default()
                    }),
                    calib_v3: r.cal.clone(),
                    media_clock: clock(e) as i64,
                    video_mode: pb::VideoMode::VideoNormal as i32,
                    lens: pb::Lens::Both as i32,
                    stitching: true,
                    video_left: 7200,
                    temp: 41,
                    ..Default::default()
                };
                pb::ReadSettingsReply {
                    keys: vec![],
                    settings: Some(o),
                }
                .encode_to_vec()
            }),
            code::GET_PHOTO_OPTIONS => ok(&|| {
                let x = pb::params::Exposure {
                    program: pb::params::exposure::Program::FullAuto as i32,
                    ..Default::default()
                };
                let o = pb::Params {
                    // RESOLUTION_2880_2880_30: 5.7K30, numbered by the lens
                    resolution: 10,
                    video_exposure: Some(x),
                    photo_exposure: Some(x),
                    time_left: 7200,
                    ..Default::default()
                };
                pb::ReadParamsReply { params: Some(o) }.encode_to_vec()
            }),
            code::GET_FILE if r.conf.is_some() => ok(&|| {
                pb::LocateFileReply {
                    uri: "/CONF".into(),
                    version: 71,
                }
                .encode_to_vec()
            }),
            code::GET_FILE => (500, vec![]),
            code::GET_FILE_LIST => ok(&|| pb::ListFilesReply::default().encode_to_vec()),
            code::START_LIVE => {
                if let Some(s) = sim.lock().unwrap().clone()
                    && live.is_none()
                {
                    let stop = Arc::new(AtomicBool::new(false));
                    let (r, st) = (r.clone(), stop.clone());
                    thread::Builder::new()
                        .name("replay".into())
                        .spawn(move || stream(r, s, e, st))
                        .expect("replay thread");
                    live = Some(stop);
                }
                (200, vec![])
            }
            code::STOP_LIVE => {
                if let Some(s) = live.take() {
                    s.store(true, Relaxed);
                }
                (200, vec![])
            }
            _ => (200, vec![]),
        }
    }
}

/// The camera worker's connection to the simulated camera.
pub fn connect(r: Arc<Replay>) -> Connect {
    Box::new(move || {
        let s = Arc::new(Mutex::new(None));
        let rr = r.clone();
        let files = move |u: &str| {
            (u == "/CONF")
                .then(|| rr.conf.clone().map(Arc::new))
                .flatten()
        };
        match Camera::sim(handler(r.clone(), s.clone()), files, 50_000_000) {
            Ok((c, ev, sim)) => {
                *s.lock().unwrap() = Some(sim);
                Ok((c, ev)).into()
            }
            Err(e) => Err(e).into(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A recording that starts with packets stamped 0 (after a boot: its first IDR and the frames after it) loops onto
    /// that IDR: every round sends every packet in order, the first round those stamped 0 at once and stamped 0, the
    /// later rounds a frame apart before the round's first stamped packet, the stamps rising.
    #[test]
    fn rounds() {
        let ts = [0, 0, 0, 1000, 1033, 1066];
        let rec = Rec {
            aus: (ts.iter().enumerate())
                .map(|(j, &t)| (t, j..j + 1))
                .collect(),
            es: (0..ts.len() as u8).collect(),
            imu: vec![],
        };
        let r = Replay {
            rec,
            cal: String::new(),
            crop: (0, [0; 2]),
            conf: None,
            log: Mutex::new(vec![]),
            battery: 80,
        };
        let (c, ev) = connect(Arc::new(r))().wait().unwrap();
        c.live_start().wait().unwrap();
        let mut got = vec![];
        while got.len() < 3 * ts.len() {
            match ev.recv_timeout(Duration::from_secs(5)) {
                Ok(link::Event::Frame(f)) if f.kind == stream::VIDEO => got.push((f.data[0], f.ts)),
                Ok(_) => {}
                Err(e) => panic!("{e}: {got:?}"),
            }
        }
        let ids: Vec<u8> = got.iter().map(|g| g.0).collect();
        assert_eq!(ids, [[0, 1, 2, 3, 4, 5]; 3].concat(), "{got:?}");
        assert!(got[..3].iter().all(|g| g.1 == 0), "{got:?}");
        assert!(got[3..].windows(2).all(|w| w[1].1 > w[0].1), "{got:?}");
    }
}
