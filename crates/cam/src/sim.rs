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
    /// a SET_OPTIONS without LOCAL_TIME (the sync's second, the zone alone) answered with status 500 (the clock's is
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
/// know), its lens (`focus_sensor`), the photography options per function mode (a write sets the fields its types
/// name), the next file number. A photo adds an `.insp`, a stop a clip and its proxy; a start and a stop send the
/// capture state (8208), a photo 8311.
pub struct Capt {
    pub video: i32,
    pub photo: i32,
    pub focus: i32,
    pub po: HashMap<i32, pb::PhotographyOptions>,
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
        let e = pb::photography_options::ExposureOptions {
            program: pb::photography_options::exposure_options::Program::FullAuto as i32,
            ..Default::default()
        };
        let v = |s, x| conf::val(s, x).unwrap();
        let o = pb::PhotographyOptions {
            video_exposure: Some(e),
            still_exposure: Some(e),
            record_resolution: v(Slot::Res, "3840_1920_30") as i32,
            exposure_bias: v(Slot::Ev, "0") as i32,
            ..Default::default()
        };
        let po = [pb::FunctionMode::NormalImage, pb::FunctionMode::NormalVideo]
            .into_iter()
            .chain(conf::func("PHOTO_HDR"))
            .chain(conf::func("VIDEO_PURE"))
            .chain(conf::func("VIDEO_TIMELAPSE"))
            .map(|f| (f as i32, o.clone()))
            .collect();
        Capt {
            video: pb::VideoSubMode::VideoNone as i32,
            photo: pb::PhotoSubMode::PhotoSingle as i32,
            focus: pb::SensorDevice::All as i32,
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
fn merge(to: &mut pb::PhotographyOptions, from: &pb::PhotographyOptions, ts: &[i32]) {
    use pb::PhotographyOptionType as T;
    for t in ts.iter().filter_map(|&t| T::try_from(t).ok()) {
        match t {
            T::VideoExposureOptions => to.video_exposure = from.video_exposure,
            T::StillExposureOptions => to.still_exposure = from.still_exposure,
            T::ExposureBias => to.exposure_bias = from.exposure_bias,
            T::WhiteBalanceValue => to.white_balance_value = from.white_balance_value,
            T::VideoIsoTopLimit => to.video_iso_top_limit = from.video_iso_top_limit,
            T::VideoGammaMode => to.gamma_mode = from.gamma_mode,
            T::PanoExposureMode => to.pano_exposure_mode = from.pano_exposure_mode,
            T::ILogSwitch => to.i_log_switch = from.i_log_switch,
            T::HdrSwitchStatus => to.hdr_switch_status = from.hdr_switch_status,
            T::PureVideoEnhancType => to.purevideo_enhanc_switch = from.purevideo_enhanc_switch,
            T::RecordResolution => to.record_resolution = from.record_resolution,
            T::PhotoResolution => to.photo_resolution = from.photo_resolution,
            T::RawCaptureType => to.raw_capture_type = from.raw_capture_type,
            T::PhotoHdrType => to.hdr_mode = from.hdr_mode,
            T::AebCaptureNum => to.aeb_capture_num = from.aeb_capture_num,
            T::PhotographySelfTimer => to.photography_self_timer = from.photography_self_timer,
            T::AccelerateFrequency => to.accelerate_frequency = from.accelerate_frequency,
            T::BurstCaptureNum => to.burst_capture_num = from.burst_capture_num,
            T::BurstCaptureTime => to.burst_capture_time = from.burst_capture_time,
            T::FovType => to.fov_type = from.fov_type,
            T::FlowstateHorizonEnable => {
                to.flowstate_horizon_enable = from.flowstate_horizon_enable
            }
            T::PhotoSizeId => to.photo_size_id = from.photo_size_id,
            T::P3Switch => to.p3_switch = from.p3_switch,
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
                let o = pb::Options {
                    camera_type: "Insta360 X5".into(),
                    serial_number: "SIM0001".into(),
                    firmware_revision: "v1.13.21".into(),
                    battery_info: Some(pb::BatteryInfo {
                        battery_level: c.battery.unwrap_or(80),
                        ..Default::default()
                    }),
                    video_sub_mode: c.capture.as_ref().map_or(0, |k| k.video),
                    photo_sub_mode: c.capture.as_ref().map_or(0, |k| k.photo),
                    focus_sensor: c.capture.as_ref().map_or(0, |k| k.focus),
                    storage_state: Some(pb::StorageState {
                        free_space: 100_000_000_000,
                        total_space: 500_000_000_000,
                        ..Default::default()
                    }),
                    wifi_info: Some(match &c.ap {
                        Some((s, p)) => pb::options::WifiInfo {
                            ssid: s.clone(),
                            password: p.clone(),
                            ..Default::default()
                        },
                        None => pb::options::WifiInfo {
                            ssid: "X5 SIM001.OSC".into(),
                            password: "88888888".into(),
                            ..Default::default()
                        },
                    }),
                    local_time: c
                        .clk
                        .map_or(clock().0 + 30, |(t, at)| t + at.elapsed().as_secs()),
                    ..Default::default()
                };
                pb::GetOptionsResp {
                    option_types: vec![],
                    value: Some(o),
                }
                .encode_to_vec()
            }),
            link::code::GET_FILE_LIST => {
                let q = pb::GetFileList::decode(b).unwrap();
                let dng = q.media_type == pb::MediaType::Dng as i32;
                let all: Vec<&String> = c
                    .uris
                    .iter()
                    .filter(|u| u.ends_with(".dng") == dng)
                    .collect();
                let page = all
                    .iter()
                    .skip(q.start as usize)
                    .take(q.limit as usize)
                    .map(|u| u.to_string())
                    .collect();
                let r = pb::GetFileListResp {
                    uri: page,
                    total_count: all.len() as u32,
                    ..Default::default()
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
                        let m = pb::ExtraMetadata {
                            creation_time: format!("{}{}", p[1], p[2]).parse().unwrap(),
                            file_size: c.data.get(u).map_or(1000, |d| d.len() as u64),
                            total_time_ms: if n.ends_with(".insv") { 61_000 } else { 0 },
                            dimension: Some(pb::extra_metadata::Vector2 { x: 7680, y: 3840 }),
                            ..Default::default()
                        };
                        pb::FileInfo {
                            file_path: u.clone(),
                            metadata: m.encode_to_vec(),
                        }
                    })
                    .collect();
                let r = pb::FileInfoList {
                    file_info: fi,
                    card_location: 0,
                };
                (200, r.encode_to_vec())
            }
            link::code::GET_EDITINFO_LIST => {
                let l = c
                    .favs
                    .iter()
                    .map(|(u, f)| pb::EditFileInfo {
                        file_path: u.clone(),
                        favorite_info: Some(pb::FavoriteInfo {
                            playback_is_favorite: *f,
                            modify_timestamp: 1,
                        }),
                        ..Default::default()
                    })
                    .collect();
                let r = pb::EditFileInfoList {
                    edit_file_info_list: l,
                    ..Default::default()
                };
                (200, r.encode_to_vec())
            }
            link::code::GET_FILE_EXTRA => {
                let q = pb::GetFileExtra::decode(b).unwrap();
                match c.uris.contains(&q.uri) && q.extra_type == 2 {
                    true => match &c.th {
                        Some((v, p)) => (200, if q.uri.ends_with(".jpg") { p } else { v }.clone()),
                        None => (200, ramp.clone()),
                    },
                    false => (500, vec![]),
                }
            }
            link::code::DELETE_FILES => {
                let q = pb::DeleteFiles::decode(b).unwrap();
                let (c, many) = (&mut *c, q.uri.len() > 1);
                let is = |u: &String, m: u8| c.stuck.get(u) == Some(&m);
                if q.uri.iter().any(|u| is(u, 3) || many && is(u, 1)) {
                    return (500, vec![]);
                }
                c.uris.retain(|u| !q.uri.contains(u) || many && is(u, 2));
                // a camera's deleted file is gone: no bytes, no flag
                if c.capture.is_some() {
                    let left = c.uris.clone();
                    c.data.retain(|u, _| u == "/CONF" || left.contains(u));
                    c.favs.retain(|f| left.contains(&f.0));
                }
                (200, pb::DeleteFilesResp::default().encode_to_vec())
            }
            link::code::SET_OPTIONS => {
                let q = pb::SetOptions::decode(b).unwrap();
                let has = |t: pb::OptionType| q.option_types.contains(&(t as i32));
                let v = q.value.clone().unwrap_or_default();
                match has(pb::OptionType::LocalTime) {
                    true => c.clk = Some((v.local_time, Instant::now())),
                    false if c.zone => return (500, vec![]),
                    false => {}
                }
                if let Some(k) = &mut c.capture {
                    if has(pb::OptionType::PhotoSubMode) {
                        (k.photo, k.video) = (v.photo_sub_mode, pb::VideoSubMode::VideoNone as i32)
                    }
                    if has(pb::OptionType::VideoSubMode) {
                        k.video = v.video_sub_mode
                    }
                }
                (200, pb::SetOptionsResp::default().encode_to_vec())
            }
            link::code::CAPTURE_STATUS => ok(&|| {
                let s = pb::CameraCaptureStatus {
                    state: c.cap,
                    ..Default::default()
                };
                pb::GetCurrentCaptureStatusResp { status: Some(s) }.encode_to_vec()
            }),
            link::code::SET_FAVORITE => {
                for s in pb::SetFavoriteList::decode(b).unwrap().set_favorite {
                    c.favs.retain(|f| f.0 != s.material_url);
                    c.favs.push((s.material_url, s.favorite_flag == 1));
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
    let s = pb::CaptureStatus {
        state: c.cap,
        ..Default::default()
    };
    note(c, 8208, &s.encode_to_vec())
}

/// The capture requests of a card with capture ([`Capt`]); `f`: the card's own handle (a late capture state).
fn capt(f: &Fake, c: &mut Card, code: u16, b: &[u8]) -> (u16, Vec<u8>) {
    use link::code::*;
    use pb::CameraCaptureState as C;
    let k = c.capture.as_mut().unwrap();
    match code {
        GET_FILE => (
            200,
            pb::GetFileResp {
                uri: "/CONF".into(),
                version: 71,
            }
            .encode_to_vec(),
        ),
        GET_PHOTO_OPTIONS => {
            let q = pb::GetPhotographyOptions::decode(b).unwrap_or_default();
            let r = pb::GetPhotographyOptionsResp {
                value: k.po.get(&q.function_mode).cloned(),
                ..Default::default()
            };
            (200, r.encode_to_vec())
        }
        SET_PHOTO_OPTIONS => {
            let q = pb::SetPhotographyOptions::decode(b).unwrap_or_default();
            let to = k.po.entry(q.function_mode).or_default();
            merge(to, &q.value.unwrap_or_default(), &q.option_types);
            (
                200,
                pb::SetPhotographyOptionsResp::default().encode_to_vec(),
            )
        }
        TAKE_PICTURE => {
            let u = k.name("IMG", "00", "insp");
            k.n += 1;
            let p = pb::Photo {
                uri: u.clone(),
                file_size: 4096,
                ..Default::default()
            };
            let r = pb::TakePictureResponse {
                image: Some(p),
                ..Default::default()
            };
            c.uris.push(u.clone());
            c.data.insert(u, Arc::new(vec![7; 4096]));
            note(c, 8311, &r.encode_to_vec());
            (200, r.encode_to_vec())
        }
        START_CAPTURE | START_TIMELAPSE => {
            let to = match code {
                START_CAPTURE => C::NormalCapture,
                _ => C::TimelapseCapture,
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
            c.cap = C::NotCapture as i32;
            for u in [&v, &l] {
                c.uris.push(u.clone());
                c.data.insert(u.clone(), Arc::new(vec![9; 8192]));
            }
            let video = named.then(|| pb::Video {
                uri: v,
                ..Default::default()
            });
            state(c);
            match code {
                STOP_CAPTURE => (200, pb::StopCaptureResp { video }.encode_to_vec()),
                _ => (200, pb::StopTimelapseResp { video }.encode_to_vec()),
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
