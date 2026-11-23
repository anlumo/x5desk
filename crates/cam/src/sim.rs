//! A simulated X5 for tests of the camera worker and of what sits on it: a fake card (files, bytes, favourites, a
//! clock, requests refused or never answered; capture when asked for: `/CONF`, modes, settings, photos and clips, the
//! notifications that go with them) behind link's simulated camera, its real I/O thread, UCD framing and tunnel HTTP;
//! a worker connected to it, and waits for what the worker reports.

use crate::{
    album::{self, Cache, Item},
    camera::{Conn, Connect, Info, Stats, Taps, Up, Worker, clock},
    conf::{self, Slot},
    dl::{self, St},
};
use link::{Camera, pb};
use prost::Message;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    thread,
    time::{Duration, Instant},
};

/// The fake card: files, their bytes, favourite flags, and every request with the time it came.
#[derive(Default)]
pub struct Card {
    pub uris: Vec<String>,
    pub data: HashMap<String, Arc<Vec<u8>>>,
    pub favs: Vec<(String, bool)>,
    pub log: Vec<(Instant, u16, Vec<u8>)>,
    /// HTTP connections opened per URI
    pub opens: HashMap<String, usize>,
    /// trailer thumbnails of videos and photos instead of a grey ramp
    pub th: Option<(Vec<u8>, Vec<u8>)>,
    /// files DELETE_FILES has trouble with: 1 a request of several with it fails, 2 a request of several answers
    /// but leaves it, 3 every request with it fails
    pub stuck: HashMap<String, u8>,
    /// request codes answered with status 500
    pub refuse: Vec<u16>,
    /// request codes never answered
    pub mute: Vec<u16>,
    /// the camera clock SET_OPTIONS set and when; until then 30 s ahead of this computer's
    pub clk: Option<(u64, Instant)>,
    /// a SET_OPTIONS without the clock (the sync's second, the zone alone) answered with status 500 (the clock's is
    /// taken)
    pub zone: bool,
    /// the capture state CAPTURE_STATUS answers (no notification goes with a change)
    pub cap: i32,
    /// the access point's SSID and password GET_OPTIONS reports; none: `X5 SIM001.OSC`, `88888888`
    pub ap: Option<(String, String)>,
    /// the battery level GET_OPTIONS reports (%); none: 80
    pub battery: Option<u32>,
    /// the newest session's control, for the notifications capture sends ([`spawn`] keeps it; a host of its own too)
    pub link: Option<link::Sim>,
    /// capture as the X5 does it; none: its requests answered empty
    pub capture: Option<Capt>,
}

pub type Fake = Arc<Mutex<Card>>;

/// The fake camera's capture: sub-modes as the camera sends them (raw: VIDEO_NONE for a photo mode, values it may not
/// know), its lens (`lens`), the params per profile (a write sets the fields its keys
/// name), the next file number. A photo adds an `.insp`, a stop a clip and its proxy; a start and a stop send the
/// capture state (8208), a photo 8311.
pub struct Capt {
    pub video: i32,
    pub photo: i32,
    pub focus: i32,
    pub po: HashMap<i32, pb::Params>,
    pub n: u32,
    /// a stop's answer names no file (as an interval's or a starlapse's may)
    pub nameless: bool,
    /// a start answered at once, the capture state (and its 8208) this much later
    pub lag: Option<Duration>,
}

/// A `/CONF` for the simulated X5: 360 modes switched and shot on an X5 (PHOTO_SINGLE, VIDEO_NORMAL) and others
/// (PHOTO_HDR, VIDEO_PURE, VIDEO_TIMELAPSE, a live mode), settings written on an X5 (white balance, resolution,
/// exposure program and ISO in VIDEO_NORMAL) and others (EV, the self-timer), the read-only duration, ISO fixed at
/// Auto by FULL_AUTO; a single-lens block.
pub const CONF: &str = r#"{"Sphere06C": {
    "photo_sub_mode": ["PHOTO_SINGLE", "PHOTO_HDR"],
    "video_sub_mode": ["VIDEO_NORMAL", "VIDEO_PURE", "VIDEO_TIMELAPSE"],
    "live_sub_mode": ["VIDEO_LIVING_MODE_PANO"],
    "PHOTO_SINGLE": {
        "white_balance": {"depend_on": [], "list": {"default": ["0", "4000", "5500"]}},
        "exposure_bias": {"depend_on": [], "list": {"default": ["-0.3", "0", "0.3"]}},
        "photography_self_timer": {"depend_on": [], "list": {"default": ["0", "2"]}}
    },
    "PHOTO_HDR": {
        "white_balance": {"depend_on": [], "list": {"default": ["0", "4000"]}}
    },
    "VIDEO_NORMAL": {
        "record_resolution": {"depend_on": [], "list": {"default": ["3840_1920_30", "5760_2880_30"]}},
        "exposure_program": {"depend_on": [], "list": {"default": ["FULL_AUTO", "MANUAL"]}},
        "exposure_iso": {"depend_on": ["exposure_program"],
            "list": {"FULL_AUTO": ["0"], "MANUAL": ["100", "400", "800"]}},
        "white_balance": {"depend_on": [], "list": {"default": ["0", "4000", "5500"]}},
        "exposure_bias": {"depend_on": [], "list": {"default": ["-0.3", "0", "0.3"]}},
        "record_duration": {"depend_on": [], "list": {"default": ["0", "600"]}}
    },
    "VIDEO_PURE": {
        "white_balance": {"depend_on": [], "list": {"default": ["0", "4000"]}}
    },
    "VIDEO_TIMELAPSE": {},
    "VIDEO_LIVING_MODE_PANO": {}
}, "Fisheyes06C": {
    "photo_sub_mode": ["PHOTO_SINGLE"],
    "video_sub_mode": ["VIDEO_NORMAL"],
    "live_sub_mode": [],
    "PHOTO_SINGLE": {},
    "VIDEO_NORMAL": {
        "white_balance": {"depend_on": [], "list": {"default": ["0", "4000"]}}
    }
}}"#;

impl Capt {
    /// PHOTO_SINGLE, 360, FULL_AUTO, the settings at the first of their values.
    pub fn new() -> Capt {
        let e = pb::params::Exposure {
            program: pb::params::exposure::Program::FullAuto as i32,
            ..Default::default()
        };
        let v = |s, x| conf::val(s, x).unwrap();
        let o = pb::Params {
            video_exposure: Some(e),
            photo_exposure: Some(e),
            resolution: v(Slot::Res, "3840_1920_30") as i32,
            ev: v(Slot::Ev, "0") as i32,
            ..Default::default()
        };
        let po = [pb::Profile::PhotoSingle, pb::Profile::VideoNormal]
            .into_iter()
            .chain(conf::func("PHOTO_HDR"))
            .chain(conf::func("VIDEO_PURE"))
            .chain(conf::func("VIDEO_TIMELAPSE"))
            .map(|f| (f as i32, o))
            .collect();
        Capt {
            video: pb::VideoMode::VideoNone as i32,
            photo: pb::PhotoMode::PhotoSingle as i32,
            focus: pb::Lens::Both as i32,
            po,
            n: 1,
            nameless: false,
            lag: None,
        }
    }

    /// A new file's name: `<pre>_20261105_<hhmmss>_<part>_<nnn>.<ext>`, a second apart per number.
    fn name(&self, pre: &str, part: &str, ext: &str) -> String {
        let t = 120000 + self.n % 60 + self.n / 60 * 100;
        format!(
            "/DCIM/Camera01/{pre}_20261105_{t:06}_{part}_{:03}.{ext}",
            self.n
        )
    }
}

impl Default for Capt {
    fn default() -> Capt {
        Capt::new()
    }
}

/// Capture as the X5 does it on card `c`, `/CONF` served at `/CONF`.
pub fn capture(c: &mut Card) {
    c.capture = Some(Capt::new());
    c.data
        .insert("/CONF".into(), Arc::new(CONF.as_bytes().to_vec()));
}

/// The fields of `from` its write's types `ts` name, into `to` (what a camera keeps of a SET_PHOTO_OPTIONS).
fn merge(to: &mut pb::Params, from: &pb::Params, ts: &[i32]) {
    use pb::Param as T;
    for t in ts.iter().filter_map(|&t| T::try_from(t).ok()) {
        match t {
            T::VideoExposure => to.video_exposure = from.video_exposure,
            T::PhotoExposure => to.photo_exposure = from.photo_exposure,
            T::Ev => to.ev = from.ev,
            T::WhiteBalance => to.white_balance = from.white_balance,
            T::IsoMax => to.iso_max = from.iso_max,
            T::Color => to.color = from.color,
            T::LensExposure => to.lens_exposure = from.lens_exposure,
            T::Log => to.log = from.log,
            T::Hdr => to.hdr = from.hdr,
            T::LowLight => to.low_light = from.low_light,
            T::Resolution => to.resolution = from.resolution,
            T::PhotoResolution => to.photo_resolution = from.photo_resolution,
            T::Format => to.format = from.format,
            T::PhotoHdr => to.photo_hdr = from.photo_hdr,
            T::Brackets => to.brackets = from.brackets,
            T::Timer => to.timer = from.timer,
            T::Speed => to.speed = from.speed,
            T::BurstCount => to.burst_count = from.burst_count,
            T::BurstSecs => to.burst_secs = from.burst_secs,
            T::Fov => to.fov = from.fov,
            T::Horizon => to.horizon = from.horizon,
            T::Aspect => to.aspect = from.aspect,
            T::P3 => to.p3 = from.p3,
            _ => {}
        }
    }
}

pub fn dir(n: &str) -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join(format!("x5desk-{n}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A trailer thumbnail as the X5 sends it, 640 × 320 (scales to 320 × 160 like the real ones).
pub fn nv12() -> Vec<u8> {
    let (w, h) = (640u32, 320u32);
    let mut b = vec![128u8; 40 + (w * h * 3 / 2) as usize];
    let n = b.len() as u32;
    b[..40].fill(0);
    for (o, v) in [(0, 1), (4, n), (8, 1), (16, w), (20, h)] {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes())
    }
    for (i, y) in b[40..40 + (w * h) as usize].iter_mut().enumerate() {
        *y = (i % w as usize * 255 / w as usize) as u8
    }
    b
}

pub fn card(names: &[&str]) -> Fake {
    let c = Card {
        uris: names
            .iter()
            .map(|n| format!("/DCIM/Camera01/{n}"))
            .collect(),
        ..Default::default()
    };
    Arc::new(Mutex::new(c))
}

/// Answers like the X5 for what the worker sends.
pub fn handler(f: Fake) -> impl FnMut(u16, &[u8]) -> (u16, Vec<u8>) + Send + 'static {
    let ramp = nv12();
    move |code, b| {
        let mut c = f.lock().unwrap();
        c.log.push((Instant::now(), code, b.to_vec()));
        if c.refuse.contains(&code) {
            return (500, vec![]);
        }
        if c.mute.contains(&code) {
            return (0, vec![]);
        }
        let ok = |m: &dyn Fn() -> Vec<u8>| (200, m());
        match code {
            link::code::GET_OPTIONS => ok(&|| {
                let o = pb::Settings {
                    model: "Insta360 X5".into(),
                    serial: "SIM0001".into(),
                    firmware: "v1.13.21".into(),
                    battery: Some(pb::Battery {
                        level: c.battery.unwrap_or(80),
                        ..Default::default()
                    }),
                    video_mode: c.capture.as_ref().map_or(0, |k| k.video),
                    photo_mode: c.capture.as_ref().map_or(0, |k| k.photo),
                    lens: c.capture.as_ref().map_or(0, |k| k.focus),
                    storage: Some(pb::Storage {
                        free: 100_000_000_000,
                        total: 500_000_000_000,
                        ..Default::default()
                    }),
                    wifi: Some(match &c.ap {
                        Some((s, p)) => pb::settings::Wifi {
                            ssid: s.clone(),
                            password: p.clone(),
                            ..Default::default()
                        },
                        None => pb::settings::Wifi {
                            ssid: "X5 SIM001.OSC".into(),
                            password: "88888888".into(),
                            ..Default::default()
                        },
                    }),
                    clock: c
                        .clk
                        .map_or(clock().0 + 30, |(t, at)| t + at.elapsed().as_secs()),
                    ..Default::default()
                };
                pb::ReadSettingsReply {
                    keys: vec![],
                    settings: Some(o),
                }
                .encode_to_vec()
            }),
            link::code::GET_FILE_LIST => {
                let q = pb::ListFiles::decode(b).unwrap();
                let dng = q.media == pb::Media::Raw as i32;
                let all: Vec<&String> = c
                    .uris
                    .iter()
                    .filter(|u| u.ends_with(".dng") == dng)
                    .collect();
                let page = all
                    .iter()
                    .skip(q.offset as usize)
                    .take(q.limit as usize)
                    .map(|u| u.to_string())
                    .collect();
                let r = pb::ListFilesReply {
                    uris: page,
                    total: all.len() as u32,
                };
                (200, r.encode_to_vec())
            }
            link::code::GET_FILEINFO_LIST => {
                let fi = c
                    .uris
                    .iter()
                    .filter(|u| !u.ends_with(".lrv") && !u.ends_with(".dng"))
                    .map(|u| {
                        let n = album::name(u);
                        let p: Vec<&str> = n.split('_').collect();
                        let m = pb::Metadata {
                            created: format!("{}{}", p[1], p[2]).parse().unwrap(),
                            size: c.data.get(u).map_or(1000, |d| d.len() as u64),
                            ms: if n.ends_with(".insv") { 61_000 } else { 0 },
                            dims: Some(pb::metadata::Pair { x: 7680, y: 3840 }),
                            ..Default::default()
                        };
                        pb::Entry {
                            path: u.clone(),
                            meta: m.encode_to_vec(),
                        }
                    })
                    .collect();
                let r = pb::Entries {
                    entries: fi,
                    volume: 0,
                };
                (200, r.encode_to_vec())
            }
            link::code::GET_EDITINFO_LIST => {
                let l = c
                    .favs
                    .iter()
                    .map(|(u, f)| pb::FileMark {
                        path: u.clone(),
                        favorite: Some(pb::Favorite {
                            on: *f,
                            modified: 1,
                        }),
                    })
                    .collect();
                let r = pb::FileMarks { marks: l };
                (200, r.encode_to_vec())
            }
            link::code::GET_FILE_EXTRA => {
                let q = pb::ReadExtra::decode(b).unwrap();
                match c.uris.contains(&q.uri) && q.kind == 2 {
                    true => match &c.th {
                        Some((v, p)) => (200, if q.uri.ends_with(".jpg") { p } else { v }.clone()),
                        None => (200, ramp.clone()),
                    },
                    false => (500, vec![]),
                }
            }
            link::code::DELETE_FILES => {
                let q = pb::DeleteFiles::decode(b).unwrap();
                let (c, many) = (&mut *c, q.uris.len() > 1);
                let is = |u: &String, m: u8| c.stuck.get(u) == Some(&m);
                if q.uris.iter().any(|u| is(u, 3) || many && is(u, 1)) {
                    return (500, vec![]);
                }
                c.uris.retain(|u| !q.uris.contains(u) || many && is(u, 2));
                // a camera's deleted file is gone: no bytes, no flag
                if c.capture.is_some() {
                    let left = c.uris.clone();
                    c.data.retain(|u, _| u == "/CONF" || left.contains(u));
                    c.favs.retain(|f| left.contains(&f.0));
                }
                (200, pb::DeleteFilesReply::default().encode_to_vec())
            }
            link::code::SET_OPTIONS => {
                let q = pb::WriteSettings::decode(b).unwrap();
                let has = |t: pb::Setting| q.keys.contains(&(t as i32));
                let v = q.settings.clone().unwrap_or_default();
                match has(pb::Setting::Clock) {
                    true => c.clk = Some((v.clock, Instant::now())),
                    false if c.zone => return (500, vec![]),
                    false => {}
                }
                if let Some(k) = &mut c.capture {
                    if has(pb::Setting::PhotoMode) {
                        (k.photo, k.video) = (v.photo_mode, pb::VideoMode::VideoNone as i32)
                    }
                    if has(pb::Setting::VideoMode) {
                        k.video = v.video_mode
                    }
                }
                (200, pb::WriteSettingsReply::default().encode_to_vec())
            }
            link::code::CAPTURE_STATUS => ok(&|| {
                let s = pb::CaptureStatus {
                    state: c.cap,
                    ..Default::default()
                };
                pb::CaptureStatusReply { status: Some(s) }.encode_to_vec()
            }),
            link::code::SET_FAVORITE => {
                for s in pb::SetFavorites::decode(b).unwrap().flags {
                    c.favs.retain(|f| f.0 != s.path);
                    c.favs.push((s.path, s.on == 1));
                }
                (200, vec![])
            }
            code if c.capture.is_some() => capt(&f, &mut c, code, b),
            _ => (200, vec![]),
        }
    }
}

/// Notification `n` from the card's camera.
fn note(c: &Card, n: u16, b: &[u8]) {
    if let Some(l) = &c.link {
        l.note(n, b)
    }
}

/// The capture state, as the camera sends it when it changes (8208).
fn state(c: &Card) {
    let s = pb::CaptureNote {
        state: c.cap,
        ..Default::default()
    };
    note(c, 8208, &s.encode_to_vec())
}

/// The capture requests of a card with capture ([`Capt`]); `f`: the card's own handle (a late capture state).
fn capt(f: &Fake, c: &mut Card, code: u16, b: &[u8]) -> (u16, Vec<u8>) {
    use link::code::*;
    use pb::CaptureState as C;
    let k = c.capture.as_mut().unwrap();
    match code {
        GET_FILE => (
            200,
            pb::LocateFileReply {
                uri: "/CONF".into(),
                version: 71,
            }
            .encode_to_vec(),
        ),
        GET_PHOTO_OPTIONS => {
            let q = pb::ReadParams::decode(b).unwrap_or_default();
            let r = pb::ReadParamsReply {
                params: k.po.get(&q.profile).cloned(),
            };
            (200, r.encode_to_vec())
        }
        SET_PHOTO_OPTIONS => {
            let q = pb::WriteParams::decode(b).unwrap_or_default();
            let to = k.po.entry(q.profile).or_default();
            merge(to, &q.params.unwrap_or_default(), &q.keys);
            (200, pb::WriteParamsReply::default().encode_to_vec())
        }
        TAKE_PICTURE => {
            let u = k.name("IMG", "00", "insp");
            k.n += 1;
            let p = pb::PhotoFile {
                uri: u.clone(),
                size: 4096,
            };
            let r = pb::TakePhotoReply { photo: Some(p) };
            c.uris.push(u.clone());
            c.data.insert(u, Arc::new(vec![7; 4096]));
            note(c, 8311, &r.encode_to_vec());
            (200, r.encode_to_vec())
        }
        START_CAPTURE | START_TIMELAPSE => {
            let to = match code {
                START_CAPTURE => C::Recording,
                _ => C::TimelapseRecording,
            } as i32;
            match k.lag {
                None => {
                    c.cap = to;
                    state(c)
                }
                Some(d) => {
                    let f = f.clone();
                    thread::spawn(move || {
                        thread::sleep(d);
                        let mut c = f.lock().unwrap();
                        c.cap = to;
                        state(&c)
                    });
                }
            }
            (200, vec![])
        }
        STOP_CAPTURE | STOP_TIMELAPSE => {
            let (v, l) = (k.name("VID", "00", "insv"), k.name("LRV", "01", "lrv"));
            let named = !k.nameless;
            k.n += 1;
            c.cap = C::Idle as i32;
            for u in [&v, &l] {
                c.uris.push(u.clone());
                c.data.insert(u.clone(), Arc::new(vec![9; 8192]));
            }
            let video = named.then_some(pb::VideoFile { uri: v });
            state(c);
            match code {
                STOP_CAPTURE => (200, pb::StopRecordingReply { file: video }.encode_to_vec()),
                _ => (200, pb::StopLapseReply { file: video }.encode_to_vec()),
            }
        }
        _ => (200, vec![]),
    }
}

/// A worker whose connections go to the fake card at `rate` bytes/s per HTTP connection; the newest session's
/// control.
pub fn spawn(
    f: &Fake,
    rate: u64,
    q: dl::Shared,
    cache: Cache,
) -> (Worker, Arc<Mutex<Option<link::Sim>>>) {
    let sim = Arc::new(Mutex::new(None));
    let (f2, s2) = (f.clone(), sim.clone());
    let connect: Connect = Box::new(move || {
        let ff = f2.clone();
        let files = move |u: &str| {
            let mut c = ff.lock().unwrap();
            *c.opens.entry(u.into()).or_default() += 1;
            c.data.get(u).cloned()
        };
        Camera::sim(handler(f2.clone()), files, rate)
            .map(|(c, ev, s)| {
                f2.lock().unwrap().link = Some(s.clone());
                *s2.lock().unwrap() = Some(s);
                (c, ev)
            })
            .into()
    });
    let auto = Arc::new(AtomicBool::new(true));
    let w = Worker::spawn(connect, q, cache, auto, false, Taps::default(), || {});
    (w, sim)
}

pub fn wait<T>(w: &Worker, secs: f64, mut f: impl FnMut(Up) -> Option<T>) -> T {
    let end = Instant::now() + Duration::from_secs_f64(secs);
    loop {
        let u =
            w.rx.recv_timeout(end.saturating_duration_since(Instant::now()))
                .expect("timed out");
        if let Some(t) = f(u) {
            return t;
        }
    }
}

pub fn album(w: &Worker) -> Vec<Item> {
    wait(w, 5.0, |u| match u {
        Up::Album(v) => Some(v),
        _ => None,
    })
}

/// The connect's note: after the listing, and after the clock's answers when the connect sets the clock.
pub fn connected(w: &Worker) -> String {
    wait(w, 5.0, |u| match u {
        Up::Note(s, false) if s.starts_with("connected") => Some(s),
        _ => None,
    })
}

pub fn stats(w: &Worker, thumbs: &mut usize) -> Stats {
    wait(w, 10.0, |u| match u {
        Up::Thumb(_, Some((320, 160, _))) => {
            *thumbs += 1;
            None
        }
        Up::Thumb(k, t) => panic!("thumbnail {k}: {:?}", t.map(|t| (t.0, t.1))),
        Up::Stats(s) => Some(s),
        _ => None,
    })
}

pub fn conn(w: &Worker, want: Conn) -> Info {
    wait(w, 5.0, |u| match u {
        Up::Info(i) if i.conn == want => Some(i),
        _ => None,
    })
}

pub fn count(f: &Fake, code: u16) -> usize {
    f.lock().unwrap().log.iter().filter(|l| l.1 == code).count()
}

/// State, bytes done and total of the first download.
pub fn prog(q: &dl::Shared) -> (St, u64, u64) {
    let g = q.lock().unwrap();
    let x = &g.v[0];
    (x.st, x.prog.done.load(Relaxed), x.prog.total.load(Relaxed))
}

pub fn until(q: &dl::Shared, f: &dyn Fn((St, u64, u64)) -> bool) -> (St, u64, u64) {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let p = prog(q);
        if f(p) {
            return p;
        }
        assert!(Instant::now() < end, "stuck at {p:?}");
        thread::sleep(Duration::from_millis(2));
    }
}

pub const NAMES: [&str; 8] = [
    "VID_20260116_152935_00_603.insv",
    "LRV_20260116_152935_01_603.lrv",
    "IMG_20260116_174103_00_620.jpg",
    "IMG_20260116_175112_00_621.jpg",
    "IMG_20260116_175112_00_621.dng",
    "VID_20260117_141148_00_635.insv",
    "LRV_20260117_141148_01_635.lrv",
    "LRV_20260503_190639_01_710.lrv",
];
