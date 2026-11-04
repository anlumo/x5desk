//! A simulated X5 for tests of the camera worker and of what sits on it: a fake card (files, bytes, favourites, a
//! clock, requests refused or never answered) behind link's simulated camera, its real I/O thread, UCD framing and
//! tunnel HTTP; a worker connected to it, and waits for what the worker reports.

use crate::{
    album::{self, Cache, Item},
    camera::{Conn, Connect, Info, Stats, Taps, Up, Worker, clock},
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
}

pub type Fake = Arc<Mutex<Card>>;

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
                        battery_level: 80,
                        ..Default::default()
                    }),
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
                (200, pb::DeleteFilesResp::default().encode_to_vec())
            }
            link::code::SET_OPTIONS => {
                let q = pb::SetOptions::decode(b).unwrap();
                match q.option_types.contains(&(pb::OptionType::LocalTime as i32)) {
                    true => c.clk = Some((q.value.unwrap().local_time, Instant::now())),
                    false if c.zone => return (500, vec![]),
                    false => {}
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
            _ => (200, vec![]),
        }
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
