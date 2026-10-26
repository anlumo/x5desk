//! The camera worker against link's simulated camera: the real I/O thread, UCD framing and tunnel HTTP, with a fake
//! card behind them. The commands that change the camera are checked here at the message level only.

use super::*;
use prost::Message;

/// The fake card: files, their bytes, favourite flags, and every request with the time it came.
#[derive(Default)]
struct Card {
    uris: Vec<String>,
    data: HashMap<String, Arc<Vec<u8>>>,
    favs: Vec<(String, bool)>,
    log: Vec<(Instant, u16, Vec<u8>)>,
    /// HTTP connections opened per URI
    opens: HashMap<String, usize>,
    /// trailer thumbnails of videos and photos instead of a grey ramp
    th: Option<(Vec<u8>, Vec<u8>)>,
    /// files DELETE_FILES has trouble with: 1 a request of several with it fails, 2 a request of several answers
    /// but leaves it, 3 every request with it fails
    stuck: HashMap<String, u8>,
    /// request codes answered with status 500
    refuse: Vec<u16>,
}

type Fake = Arc<Mutex<Card>>;

fn dir(n: &str) -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join(format!("x5desk-{n}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A trailer thumbnail as the X5 sends it, 640 × 320 (scales to 320 × 160 like the real ones).
fn nv12() -> Vec<u8> {
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

fn card(names: &[&str]) -> Fake {
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
fn handler(f: Fake) -> impl FnMut(u16, &[u8]) -> (u16, Vec<u8>) + Send + 'static {
    let ramp = nv12();
    move |code, b| {
        let mut c = f.lock().unwrap();
        c.log.push((Instant::now(), code, b.to_vec()));
        if c.refuse.contains(&code) {
            return (500, vec![]);
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
                    wifi_info: Some(pb::options::WifiInfo {
                        ssid: "X5 SIM001.OSC".into(),
                        password: "88888888".into(),
                        ..Default::default()
                    }),
                    local_time: clock().0 + 30,
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
fn spawn(
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

fn wait<T>(w: &Worker, secs: f64, mut f: impl FnMut(Up) -> Option<T>) -> T {
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

fn album(w: &Worker) -> Vec<Item> {
    wait(w, 5.0, |u| match u {
        Up::Album(v) => Some(v),
        _ => None,
    })
}

fn stats(w: &Worker, thumbs: &mut usize) -> Stats {
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

fn conn(w: &Worker, want: Conn) -> Info {
    wait(w, 5.0, |u| match u {
        Up::Info(i) if i.conn == want => Some(i),
        _ => None,
    })
}

fn count(f: &Fake, code: u16) -> usize {
    f.lock().unwrap().log.iter().filter(|l| l.1 == code).count()
}

/// State, bytes done and total of the first download.
fn prog(q: &dl::Shared) -> (St, u64, u64) {
    let g = q.lock().unwrap();
    let x = &g.v[0];
    (x.st, x.prog.done.load(Relaxed), x.prog.total.load(Relaxed))
}

fn until(q: &dl::Shared, f: &dyn Fn((St, u64, u64)) -> bool) -> (St, u64, u64) {
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

const NAMES: [&str; 8] = [
    "VID_20260116_152935_00_603.insv",
    "LRV_20260116_152935_01_603.lrv",
    "IMG_20260116_174103_00_620.jpg",
    "IMG_20260116_175112_00_621.jpg",
    "IMG_20260116_175112_00_621.dng",
    "VID_20260117_141148_00_635.insv",
    "LRV_20260117_141148_01_635.lrv",
    "LRV_20260503_190639_01_710.lrv",
];

/// A camera enumerated in app mode that never answers (the hardware run of 2026-10-23): the connection shows as
/// connecting, Disconnect cancels it at once, and the worker (the app closing) ends at once instead of after the
/// 30 s sync deadline.
#[test]
fn silent_camera() {
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let auto = Arc::new(AtomicBool::new(true));
    let w = Worker::spawn(
        Box::new(Camera::mute),
        q,
        Cache::at(None),
        auto,
        false,
        Taps::default(),
        || {},
    );
    w.tx.send(Cmd::Connect).unwrap();
    assert!(conn(&w, Conn::Connecting).since.is_some());
    thread::sleep(Duration::from_millis(300));
    let t = Instant::now();
    w.tx.send(Cmd::Disconnect).unwrap();
    conn(&w, Conn::Off);
    let c = t.elapsed();
    w.tx.send(Cmd::Connect).unwrap();
    conn(&w, Conn::Connecting);
    thread::sleep(Duration::from_millis(300));
    let t = Instant::now();
    drop(w);
    let d = t.elapsed();
    println!(
        "silent camera: cancelled in {:.1} ms, worker gone in {:.1} ms",
        c.as_secs_f64() * 1e3,
        d.as_secs_f64() * 1e3
    );
    assert!(c < Duration::from_secs(1) && d < Duration::from_secs(2));
}

/// The note for a camera that never answered reads as one sentence (it was a literal wrapped without its `\`).
#[test]
fn silent_note() {
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let auto = Arc::new(AtomicBool::new(false));
    let connect: Connect = Box::new(|| Err(link::Error::Silent(0)).into());
    let w = Worker::spawn(
        connect,
        q,
        Cache::at(None),
        auto,
        false,
        Taps::default(),
        || {},
    );
    w.tx.send(Cmd::Connect).unwrap();
    let s = wait(&w, 5.0, |u| match u {
        Up::Note(s, true) => Some(s),
        _ => None,
    });
    assert!(
        s.contains("Wake it with its power button,") && !s.contains("  "),
        "{s}"
    );
}

/// Connect, album with metadata and favourites, thumbnails from the camera and then from the disk cache; battery
/// and file notifications; an unplug.
#[test]
fn album_and_cache() {
    let d = dir("album");
    let f = card(&NAMES);
    f.lock().unwrap().favs = vec![(
        "/DCIM/Camera01/VID_20260117_141148_00_635.insv".into(),
        true,
    )];
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, sim) = spawn(&f, 50_000_000, q, Cache::at(Some(d.join("thumbs"))));
    w.tx.send(Cmd::Connect).unwrap();
    // connected: identity and options come together
    let i = conn(&w, Conn::On);
    assert_eq!(i.id.as_ref().unwrap().model, "Insta360 X5");
    assert_eq!(i.wifi.unwrap().0, "X5 SIM001.OSC");
    assert!(i.skew.is_some_and(|s| (29..=31).contains(&s)));
    assert_eq!(i.storage.unwrap().total_space, 500_000_000_000);
    let v = album(&w);
    assert_eq!(v.len(), 5);
    let rec = v.iter().find(|i| i.uris[0].ends_with("_635.insv")).unwrap();
    assert_eq!(
        (rec.kind, rec.uris.len(), rec.ms, rec.date, rec.fav),
        (Kind::Video, 2, 61_000, 20260117141148, true)
    );
    let raw = v.iter().find(|i| i.uris[0].ends_with("_621.jpg")).unwrap();
    assert_eq!((raw.kind, raw.uris.len()), (Kind::Photo, 2));
    let mut n = 0;
    let s = stats(&w, &mut n);
    // the proxy without its main file also has a trailer thumbnail
    assert_eq!((n, s.cam, s.cached, s.bytes), (5, 5, 0, 5 * nv12().len()));

    w.tx.send(Cmd::Disconnect).unwrap();
    conn(&w, Conn::Off);
    // an entry a crash left empty (rename without fsync) comes from the camera again and is rewritten
    let e = std::fs::read_dir(d.join("thumbs"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::write(&e, b"").unwrap();
    let k = count(&f, link::code::GET_FILE_EXTRA);
    w.tx.send(Cmd::Connect).unwrap();
    assert_eq!(album(&w).len(), 5);
    let mut n = 0;
    let s = stats(&w, &mut n);
    assert_eq!((n, s.cam, s.cached), (5, 1, 4));
    assert_eq!(
        count(&f, link::code::GET_FILE_EXTRA),
        k + 1,
        "the other thumbnails from the warm cache"
    );
    assert_eq!(std::fs::metadata(&e).unwrap().len(), 8 + 320 * 160 * 4);

    let sim = sim.lock().unwrap().clone().unwrap();
    let b = pb::NotificationBatteryUpdate {
        battery_status: Some(pb::BatteryInfo {
            battery_level: 42,
            ..Default::default()
        }),
    };
    sim.note(8195, &b.encode_to_vec());
    wait(&w, 2.0, |u| match u {
        Up::Info(i) if i.battery.as_ref().is_some_and(|b| b.battery_level == 42) => Some(()),
        _ => None,
    });
    // a file appeared (capture stopped): the album is listed again
    f.lock()
        .unwrap()
        .uris
        .push("/DCIM/Camera01/IMG_20260118_120546_00_649.jpg".into());
    sim.note(
        8201,
        &pb::NotificationCaptureStopped::default().encode_to_vec(),
    );
    assert_eq!(album(&w).len(), 6);
    sim.unplug();
    conn(&w, Conn::Off);
    wait(&w, 2.0, |u| match u {
        Up::Note(s, true) if s.contains("link lost") => Some(()),
        _ => None,
    });
    drop(w);
    std::fs::remove_dir_all(&d).unwrap();
}

/// A low battery warns (20 % or less, not charging): the battery-low notification gives an error note and the level
/// the screens show; 21 % and a charging camera do not.
#[test]
fn low_battery() {
    use pb::battery_info::PowerType::{self, *};
    let f = card(&NAMES[..1]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, sim) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect).unwrap();
    assert!(!low(conn(&w, Conn::On).battery.as_ref().unwrap()));
    let sim = sim.lock().unwrap().clone().unwrap();
    let b = |l, p: PowerType| {
        Some(pb::BatteryInfo {
            battery_level: l,
            power_type: p as i32,
            ..Default::default()
        })
    };
    let n = pb::NotificationBatteryLow {
        battery_status: b(9, Battery),
    };
    sim.note(8196, &n.encode_to_vec());
    let s = wait(&w, 2.0, |u| match u {
        Up::Note(s, true) => Some(s),
        _ => None,
    });
    assert!(s.starts_with("9 %, low: charge the camera"), "{s}");
    let i = conn(&w, Conn::On);
    assert!(low(i.battery.as_ref().unwrap()));
    for (l, p, want) in [
        (20, Battery, true),
        (21, Battery, false),
        (9, Adapter, false),
    ] {
        let n = pb::NotificationBatteryUpdate {
            battery_status: b(l, p),
        };
        sim.note(8195, &n.encode_to_vec());
        let i = wait(&w, 2.0, |u| match u {
            Up::Info(i) if i.battery == b(l, p) => Some(i),
            _ => None,
        });
        assert_eq!(low(i.battery.as_ref().unwrap()), want, "{l} % {p:?}");
    }
    drop(w);
}

/// Every connect sets the camera clock once, as the app does (§5.1 step 5, §5.3): its two SET_OPTIONS right after the
/// identity, before anything else; the note says so. A camera that refuses it still connects and lists its album,
/// the refusal in an error note.
#[test]
fn connect_time() {
    let f = card(&NAMES[..2]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    let run = || {
        let n0 = f.lock().unwrap().log.len();
        w.tx.send(Cmd::Connect).unwrap();
        let mut notes = vec![];
        let v = wait(&w, 10.0, |u| match u {
            Up::Note(s, e) => {
                notes.push((s, e));
                None
            }
            Up::Album(v) => Some(v),
            _ => None,
        });
        let l: Vec<(u16, Vec<u8>)> = (f.lock().unwrap().log[n0..].iter())
            .map(|l| (l.1, l.2.clone()))
            .collect();
        (notes, v.len(), l)
    };
    let (t, _, off, tz, _) = clock();
    for i in 0..2 {
        let (notes, n, l) = run();
        let k: Vec<u16> = l.iter().map(|l| l.0).collect();
        let s: Vec<usize> = (0..k.len())
            .filter(|&i| k[i] == link::code::SET_OPTIONS)
            .collect();
        println!("connect {i}: requests {k:?}; notes {notes:?}");
        assert_eq!(n, 1);
        assert!(s.len() == 2 && s[1] == s[0] + 1, "{k:?}");
        assert!(
            s[0] > 0 && k[..s[0]].iter().all(|&c| c == link::code::GET_OPTIONS),
            "only the identity before: {k:?}"
        );
        let set: Vec<pb::SetOptions> = (s.iter())
            .map(|&i| pb::SetOptions::decode(&l[i].1[..]).unwrap())
            .collect();
        assert_eq!(
            (&set[0].option_types[..], &set[1].option_types[..]),
            (&[12, 13, 175][..], &[175][..])
        );
        let o = set[0].value.as_ref().unwrap();
        assert!(o.local_time.abs_diff(t) <= 5);
        assert_eq!(
            (o.time_zone_seconds_from_gmt, &*o.geographic_time_zone),
            (off, &*tz)
        );
        assert_eq!(set[1].value.as_ref().unwrap().geographic_time_zone, tz);
        assert!(
            notes.len() == 1
                && !notes[0].1
                && notes[0].0.starts_with("connected to Insta360 X5 in ")
                && notes[0].0.contains("; camera clock set to ")
                && notes[0].0.ends_with(&format!("({tz})")),
            "{notes:?}"
        );
        w.tx.send(Cmd::Disconnect).unwrap();
        wait(&w, 5.0, |u| {
            matches!(u, Up::Note(s, _) if s == "disconnected").then_some(())
        });
    }
    f.lock().unwrap().refuse = vec![link::code::SET_OPTIONS];
    let (notes, n, l) = run();
    println!("refused: notes {notes:?}");
    assert_eq!(n, 1, "listed all the same");
    assert!(l.iter().any(|l| l.0 == link::code::GET_FILE_LIST));
    assert!(
        notes.len() == 2
            && !notes[0].1
            && notes[0].0.starts_with("connected to Insta360 X5 in ")
            && !notes[0].0.contains("clock")
            && notes[1].1
            && notes[1].0.starts_with("time sync: status 500"),
        "{notes:?}"
    );
    drop(w);
}

/// A download over four tunnel connections, cancelled, resumed after a restart of the worker from the queue file,
/// cut by an unplug, resumed after reconnecting; byte-identical at the end.
#[test]
fn download_resume() {
    let d = dir("dl");
    let name = "VID_20260116_152935_00_603.insv";
    let f = card(&[name]);
    let big: Arc<Vec<u8>> = Arc::new((0..9_000_011u32).map(|i| (i ^ i >> 11) as u8).collect());
    let uri = format!("/DCIM/Camera01/{name}");
    f.lock().unwrap().data.insert(uri.clone(), big.clone());
    let qf = d.join("downloads");
    let dst = d.join("out").join(name);
    let mut q = dl::Queue::load(Some(qf.clone()));
    q.add(&uri, dst.clone());
    let q = Arc::new(Mutex::new(q));
    // 4 × 3 MB/s: about 0.75 s for the whole file
    let rate = 3_000_000;
    let (w, _) = spawn(&f, rate, q.clone(), Cache::at(None));
    w.tx.send(Cmd::Connect).unwrap();
    until(&q, &|p| p.0 == St::Running && p.1 >= 2_000_000);
    q.lock().unwrap().cancel(0);
    let (_, had, total) = until(&q, &|p| p.0 == St::Paused);
    assert!(had < total && total == big.len() as u64, "{had} / {total}");
    assert_eq!(f.lock().unwrap().opens[&uri], 4, "four connections");
    let part = |e: &str| {
        let mut s = dst.as_os_str().to_owned();
        s.push(e);
        PathBuf::from(s)
    };
    assert!(part(".part").exists() && part(".part.todo").exists() && !dst.exists());
    drop(w);

    // the app starts again: the queue comes from its file
    let mut q = dl::Queue::load(Some(qf.clone()));
    assert_eq!(q.v[0].st, St::Paused);
    q.resume(0);
    let q = Arc::new(Mutex::new(q));
    let (w, sim) = spawn(&f, rate, q.clone(), Cache::at(None));
    w.tx.send(Cmd::Connect).unwrap();
    until(&q, &|p| p.0 == St::Running && p.2 > 0);
    // total and done of a resumed transfer are stored one after the other
    thread::sleep(Duration::from_millis(5));
    let (_, again, _) = prog(&q);
    assert!(again >= had, "resumed at {again}, had {had}");
    until(&q, &|p| p.1 >= again + 1_000_000);
    sim.lock().unwrap().as_ref().unwrap().unplug();
    conn(&w, Conn::Off);
    let (st, cut, _) = until(&q, &|p| p.0 != St::Running);
    assert_eq!(st, St::Queued);
    assert!(cut > again);
    w.tx.send(Cmd::Connect).unwrap();
    let p = wait(&w, 10.0, |u| match u {
        Up::Done(p) => Some(p),
        _ => None,
    });
    assert_eq!(p, dst);
    assert!(std::fs::read(&dst).unwrap() == *big);
    assert!(!part(".part").exists() && !part(".part.todo").exists());
    drop(w);
    assert_eq!(dl::Queue::load(Some(qf)).v[0].st, St::Done);
    std::fs::remove_dir_all(&d).unwrap();
}

/// A video capture puts the files off limits (§7.12): the running download goes back to the queue with what it
/// has, nothing starts while the capture runs, the download resumes once it stops; byte-identical at the end.
#[test]
fn capture_hold() {
    let d = dir("hold");
    let name = "VID_20260116_152935_00_603.insv";
    let f = card(&[name]);
    let big: Arc<Vec<u8>> = Arc::new((0..6_000_007u32).map(|i| (i ^ i >> 9) as u8).collect());
    let uri = format!("/DCIM/Camera01/{name}");
    f.lock().unwrap().data.insert(uri.clone(), big.clone());
    let dst = d.join(name);
    let mut q = dl::Queue::load(None);
    q.add(&uri, dst.clone());
    let q = Arc::new(Mutex::new(q));
    // 4 × 1 MB/s: about 1.5 s for the whole file
    let (w, sim) = spawn(&f, 1_000_000, q.clone(), Cache::at(None));
    w.tx.send(Cmd::Connect).unwrap();
    until(&q, &|p| p.0 == St::Running && p.1 >= 1_000_000);
    let sim = sim.lock().unwrap().clone().unwrap();
    let st = |s: i32| {
        pb::CaptureStatus {
            state: s,
            ..Default::default()
        }
        .encode_to_vec()
    };
    sim.note(8208, &st(1));
    wait(&w, 2.0, |u| match u {
        Up::Info(i) if i.busy => Some(()),
        _ => None,
    });
    let (_, had, total) = until(&q, &|p| p.0 == St::Queued);
    let opens = f.lock().unwrap().opens[&uri];
    thread::sleep(Duration::from_millis(500));
    let (st1, now, _) = prog(&q);
    assert_eq!(
        (st1, f.lock().unwrap().opens[&uri]),
        (St::Queued, opens),
        "nothing starts while it records"
    );
    assert!(now < total, "{now} of {total}: stopped");
    sim.note(8208, &st(0));
    until(&q, &|p| p.0 == St::Running && p.2 > 0);
    thread::sleep(Duration::from_millis(5));
    let again = prog(&q).1;
    assert!(again >= had, "resumed at {again}, had {had}");
    let p = wait(&w, 10.0, |u| match u {
        Up::Done(p) => Some(p),
        _ => None,
    });
    assert_eq!(p, dst);
    assert!(std::fs::read(&dst).unwrap() == *big);
    drop(w);
    std::fs::remove_dir_all(&d).unwrap();
}

/// The app's delete passes (§7.8): a batch the camera refuses goes again one URI at a time, what a fresh listing
/// still has gets another pass, a single URI it refuses ends it.
#[test]
fn delete_passes() {
    let names: Vec<String> = (0..9)
        .map(|i| format!("IMG_20260118_1200{i:02}_00_{i:03}.jpg"))
        .collect();
    let f = card(&names.iter().map(String::as_str).collect::<Vec<_>>());
    let u = f.lock().unwrap().uris.clone();
    f.lock().unwrap().stuck =
        HashMap::from([(u[0].clone(), 1), (u[3].clone(), 2), (u[6].clone(), 3)]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect).unwrap();
    assert_eq!(album(&w).len(), 9);
    let run = |v: &[String]| {
        let n0 = f.lock().unwrap().log.len();
        w.tx.send(Cmd::Delete(v.to_vec())).unwrap();
        let s = wait(&w, 10.0, |u| match u {
            Up::Note(s, e) => Some((s, e)),
            _ => None,
        });
        let left = album(&w).len();
        let n: Vec<usize> = f.lock().unwrap().log[n0..]
            .iter()
            .filter(|l| l.1 == link::code::DELETE_FILES)
            .map(|l| pb::DeleteFiles::decode(&l.2[..]).unwrap().uri.len())
            .collect();
        (s, n, left)
    };
    let ok = |s: &str| (s.to_string(), false);
    assert_eq!(
        run(&u[0..3]),
        (ok("deleted 3 of 3 files"), vec![3, 1, 1, 1], 6)
    );
    assert_eq!(run(&u[3..6]), (ok("deleted 3 of 3 files"), vec![3, 1], 3));
    let (s, n, left) = run(&u[6..9]);
    assert!(s.1 && s.0.starts_with("delete: status 500"), "{s:?}");
    assert_eq!((n, left), (vec![3, 1], 3));
    drop(w);
}

/// Delete in batches of 200, 150 ms apart; favourites; format; time sync (two SET_OPTIONS, §5.3). Each is followed
/// by a fresh album.
#[test]
fn commands() {
    let names: Vec<String> = (0..450)
        .map(|i| format!("IMG_20260118_12{:02}{:02}_00_{i:03}.jpg", i / 60, i % 60))
        .collect();
    let f = card(&names.iter().map(String::as_str).collect::<Vec<_>>());
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect).unwrap();
    assert_eq!(album(&w).len(), 450);
    let after = |w: &Worker| {
        let s = wait(w, 10.0, |u| match u {
            Up::Note(s, e) => Some((s, e)),
            _ => None,
        });
        assert!(!s.1, "{}", s.0);
        (s.0, album(w))
    };
    let uris: Vec<String> = f.lock().unwrap().uris.clone();
    let n0 = f.lock().unwrap().log.len();
    w.tx.send(Cmd::Delete(uris.clone())).unwrap();
    let (s, v) = after(&w);
    assert_eq!((s.as_str(), v.len()), ("deleted 450 of 450 files", 0));
    let dels: Vec<(Instant, Vec<String>)> = f.lock().unwrap().log[n0..]
        .iter()
        .filter(|l| l.1 == link::code::DELETE_FILES)
        .map(|l| (l.0, pb::DeleteFiles::decode(&l.2[..]).unwrap().uri))
        .collect();
    assert_eq!(
        dels.iter().map(|d| d.1.len()).collect::<Vec<_>>(),
        [200, 200, 50]
    );
    assert!(
        dels.windows(2)
            .all(|p| p[1].0 - p[0].0 >= Duration::from_millis(150))
    );
    assert_eq!(
        dels.iter().flat_map(|d| d.1.clone()).collect::<Vec<_>>(),
        uris
    );

    let u = "/DCIM/Camera01/VID_20260116_152935_00_603.insv".to_string();
    f.lock().unwrap().uris.push(u.clone());
    w.tx.send(Cmd::Fav(vec![(u.clone(), true)])).unwrap();
    let (_, v) = after(&w);
    assert!(v[0].fav);
    let l = f.lock().unwrap().log.clone();
    let s = l
        .iter()
        .rev()
        .find(|l| l.1 == link::code::SET_FAVORITE)
        .unwrap();
    let s = &pb::SetFavoriteList::decode(&s.2[..]).unwrap().set_favorite[0];
    assert_eq!((&*s.material_url, s.favorite_flag), (&*u, 1));
    assert!((s.modify_timestamp - clock().1).abs() <= 2);

    w.tx.send(Cmd::Format).unwrap();
    after(&w);
    let l = f.lock().unwrap().log.clone();
    assert!(
        l.iter()
            .any(|l| l.1 == link::code::ERASE_SD && l.2.is_empty())
    );

    let n0 = f.lock().unwrap().log.len();
    w.tx.send(Cmd::Time).unwrap();
    after(&w);
    let l = f.lock().unwrap().log[n0..].to_vec();
    let set: Vec<pb::SetOptions> = l
        .iter()
        .filter(|l| l.1 == link::code::SET_OPTIONS)
        .map(|l| pb::SetOptions::decode(&l.2[..]).unwrap())
        .collect();
    let (t, _, off, tz, _) = clock();
    assert_eq!(set.len(), 2);
    assert_eq!(
        (&set[0].option_types[..], &set[1].option_types[..]),
        (&[12, 13, 175][..], &[175][..])
    );
    let o = set[0].value.as_ref().unwrap();
    assert!(o.local_time.abs_diff(t) <= 2);
    assert_eq!(
        (o.time_zone_seconds_from_gmt, &*o.geographic_time_zone),
        (off, &*tz)
    );
    assert_eq!(set[1].value.as_ref().unwrap().geographic_time_zone, tz);
    drop(w);
}

/// The local clock helper agrees with the date tool: the offset now, and the standard one (no DST, the smaller of
/// mid-January's and mid-July's) in the favourite timestamp.
#[test]
fn local_clock() {
    let (t, l, off, tz, s) = clock();
    assert!(!tz.is_empty() && s.len() == 19, "{tz} {s}");
    let z = |a: &[&str]| {
        let o = std::process::Command::new("date").args(a).output().ok()?;
        let z = String::from_utf8_lossy(&o.stdout);
        let z = z.trim();
        let m = z.get(1..3)?.parse::<i64>().ok()? * 3600 + z.get(3..5)?.parse::<i64>().ok()? * 60;
        Some(if z.starts_with('-') { -m } else { m })
    };
    let d = |m: &str| format!("{}-{m}-15", &s[..4]);
    let (Some(n), Some(a), Some(b)) = (
        z(&["+%z"]),
        z(&["-d", &d("01"), "+%z"]),
        z(&["-d", &d("07"), "+%z"]),
    ) else {
        return eprintln!("no date tool: skipping the comparison");
    };
    println!("{tz}: now {n:+} s, January {a:+} s, July {b:+} s");
    assert_eq!(n, off as i64);
    assert_eq!(l - t as i64, a.min(b));
}

/// A Camera screen whose worker is the test: reports go in through the sender, commands come out of the receiver.
fn headless(dest: PathBuf, q: dl::Shared) -> (Cam, Sender<Up>, Receiver<Cmd>) {
    let set = Set {
        dest,
        auto: false,
        file: None,
    };
    let none: Connect = Box::new(|| Err(Error::NoDevice).into());
    let ctx = egui::Context::default();
    let mut c = Cam::with(&ctx, None, set, q, none, Cache::at(None), false);
    let ((tx, crx), (utx, rx)) = (mpsc::channel(), mpsc::channel());
    c.w = Worker { tx, rx, th: None };
    (c, utx, crx)
}

/// The album keeps [`TEX`] thumbnail textures: past that, ones off screen are not kept and come again when they are
/// shown; the least recently drawn go first, never the ones on screen.
#[test]
fn texture_cap() {
    let d = dir("tex");
    let (mut c, up, cmd) = headless(d.clone(), Arc::new(Mutex::new(dl::Queue::load(None))));
    c.items = (0..2000u64)
        .map(|i| Item {
            key: format!("k{i:04}"),
            kind: Kind::Photo,
            uris: vec![format!("/DCIM/Camera01/IMG_{i:04}.jpg")],
            date: i,
            ..Default::default()
        })
        .collect();
    c.dirty = true;
    let th = || Some((2, 1, vec![255u8; 8]));
    for it in &c.items {
        up.send(Up::Thumb(it.key.clone(), th())).unwrap();
    }
    c.tick();
    assert_eq!(c.tex.len(), TEX);
    let frame = |c: &mut Cam| {
        let r = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
        let i = egui::RawInput {
            screen_rect: Some(r),
            ..Default::default()
        };
        let mut o = c.ctx.clone().run_ui(i, |ui| {
            c.album(ui);
        });
        // no renderer takes them
        o.textures_delta.clear();
    };
    // newest first: what is on screen came last, past the cap
    frame(&mut c);
    let Ok(Cmd::Want(ks)) = cmd.try_recv() else {
        panic!("nothing asked for")
    };
    assert!(
        ks.len() >= 12 && ks.iter().all(|k| !c.tex.contains_key(k)),
        "{ks:?}"
    );
    assert_eq!(ks[0], "k1999");
    for k in &ks {
        up.send(Up::Thumb(k.clone(), th())).unwrap();
    }
    c.tick();
    assert_eq!(c.tex.len(), TEX + ks.len());
    frame(&mut c);
    assert_eq!(c.tex.len(), TEX * 3 / 4);
    assert!(ks.iter().all(|k| c.tex.contains_key(k)));
    frame(&mut c);
    assert!(cmd.try_recv().is_err(), "all on screen have their textures");
    drop(c);
    std::fs::remove_dir_all(&d).unwrap();
}

/// The Library's downloaded list: recordings in the destination folder and finished downloads elsewhere, one file
/// per recording, oldest first; photos, partial files and finished entries whose file is gone are left out.
#[test]
fn library_list() {
    let d = dir("lib");
    let (dest, old) = (d.join("dl"), d.join("old"));
    let mut q = dl::Queue::load(None);
    for n in [
        "VID_20260503_190639_00_710.insv",
        "VID_20260601_120000_00_800.insv",
    ] {
        q.add("/DCIM/Camera01/x", old.join(n));
        q.v.last_mut().unwrap().st = St::Done;
    }
    for (p, n, t) in [
        (&dest, "VID_20260116_152935_00_603.insv", 3),
        (&dest, "LRV_20260116_152935_01_603.lrv", 4),
        (&dest, "VID_20260117_141148_00_635.insv", 1),
        (&dest, "IMG_20260118_120546_00_649.jpg", 5),
        (&dest, "VID_20260307_135339_00_670.insv.part", 6),
        (&old, "VID_20260503_190639_00_710.insv", 2),
    ] {
        std::fs::create_dir_all(p).unwrap();
        std::fs::write(p.join(n), b"x").unwrap();
        let f = std::fs::File::options()
            .write(true)
            .open(p.join(n))
            .unwrap();
        f.set_modified(UNIX_EPOCH + Duration::from_secs(1_700_000_000 + t))
            .unwrap();
    }
    let (c, ..) = headless(dest.clone(), Arc::new(Mutex::new(q)));
    assert_eq!(
        c.done(),
        [
            dest.join("VID_20260117_141148_00_635.insv"),
            old.join("VID_20260503_190639_00_710.insv"),
            dest.join("LRV_20260116_152935_01_603.lrv"),
        ]
    );
    drop(c);
    std::fs::remove_dir_all(&d).unwrap();
}

/// The Downloads pane has whether a queued import waits for its card from the volumes the worker found mounted, not
/// from a look at every queued file each frame (a whole card's album queued at once stat'ed each file per frame).
#[test]
fn downloads_card() {
    let d = dir("dlcard");
    let card = d.join("X5");
    let mut q = dl::Queue::load(None);
    for i in 0..3 {
        let u = format!("file://{}/DCIM/Camera01/VID_{i}_00_1.insv", card.display());
        q.add(&u, d.join(format!("VID_{i}.insv")));
    }
    let (mut c, ..) = headless(d.clone(), Arc::new(Mutex::new(q)));
    fn walk(s: &egui::Shape, v: &mut Vec<String>) {
        match s {
            egui::Shape::Text(t) => v.push(t.galley.text().to_string()),
            egui::Shape::Vec(x) => x.iter().for_each(|s| walk(s, v)),
            _ => {}
        }
    }
    let shown = |c: &mut Cam, s: &str| {
        let r = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
        let i = egui::RawInput {
            screen_rect: Some(r),
            ..Default::default()
        };
        let mut o = c.ctx.clone().run_ui(i, |ui| drop(c.downloads(ui)));
        o.textures_delta.clear();
        let mut v = vec![];
        o.shapes.iter().for_each(|x| walk(&x.shape, &mut v));
        v.iter().filter(|t| *t == s).count()
    };
    assert_eq!(shown(&mut c, "queued: waiting for the card"), 3);
    c.info.disk = vec![disk::Vol {
        root: card,
        dev: "/dev/sdb1".into(),
        cam: None,
    }];
    assert_eq!(shown(&mut c, "queued"), 3, "the card is in");
    drop(c);
    std::fs::remove_dir_all(&d).unwrap();
}

/// A mounted card (a fake mount: a DCIM tree, with the samples of `X5_SAMPLES` as a recording when set) through
/// the worker: found and shown with no camera connected, the camera's grouping, thumbnails from the trailers, an
/// import through the queue without a camera, the card's removal.
#[test]
fn card_album() {
    let d = dir("cardalbum");
    let (r, c) = (d.join("X5"), d.join("X5/DCIM/Camera01"));
    std::fs::create_dir_all(&c).unwrap();
    let s = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    for (n, f) in [
        ("VID_20260117_141148_00_635.insv", "v635.insv"),
        ("LRV_20260117_141148_01_635.lrv", "v635.lrv"),
    ] {
        match &s {
            Some(s) => std::os::unix::fs::symlink(s.join(f), c.join(n)).unwrap(),
            None => std::fs::write(c.join(n), vec![7u8; 300_000]).unwrap(),
        }
    }
    std::fs::write(c.join("IMG_20260118_124432_00_659.jpg"), vec![9u8; 200_000]).unwrap();
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(
        &card(&[]),
        50_000_000,
        q.clone(),
        Cache::at(Some(d.join("thumbs"))),
    );
    let watch = disk::Watch {
        info: PathBuf::new(),
        sys: "/sys".into(),
        media: vec![],
        fake: vec![r.clone()],
    };
    w.tx.send(Cmd::Watch(watch)).unwrap();
    let i = wait(&w, 5.0, |u| match u {
        Up::Info(i) if i.src.is_some() => Some(i),
        _ => None,
    });
    assert_eq!(
        (i.src.as_ref(), i.disk.len(), i.conn),
        (Some(&r), 1, Conn::Off)
    );
    let v = album(&w);
    assert_eq!(v.len(), 2);
    let rec = v.iter().find(|i| i.kind == Kind::Video).unwrap();
    assert_eq!(rec.uris.len(), 2);
    let (mut got, mut none) = (0, 0);
    let st = wait(&w, 10.0, |u| match u {
        Up::Thumb(_, Some((320, 160, _))) => {
            got += 1;
            None
        }
        Up::Thumb(_, None) => {
            none += 1;
            None
        }
        Up::Stats(s) => Some(s),
        _ => None,
    });
    println!(
        "card: {} items listed in {:.1} ms, thumbnails {got} from the trailers, {none} without one, in {:.1} ms",
        v.len(),
        st.list_ms,
        st.secs * 1e3
    );
    assert_eq!((got, none), if s.is_some() { (1, 1) } else { (0, 2) });
    // the camera on the bus in a mode other than app mode, its own disk shown: the bar says so
    w.tx.send(Cmd::Plug(Plug::Other)).unwrap();
    let i = wait(&w, 5.0, |u| match u {
        Up::Info(i) if i.plug == Some(Plug::Other) => Some(i),
        _ => None,
    });
    assert_eq!(
        status(&i),
        "Camera in U-disk mode: the Album shows its card; choose Android App on its USB menu for the full features \
         (settings, capture, live preview)"
    );

    // import the proxy and the photo, no camera connected
    let dest = d.join("lib");
    let uris = [
        &rec.uris[1],
        &v.iter().find(|i| i.kind == Kind::Photo).unwrap().uris[0],
    ];
    {
        let mut g = q.lock().unwrap();
        for u in uris {
            g.add(
                &format!("file://{}", r.join(&u[1..]).display()),
                dest.join(album::name(u)),
            );
        }
    }
    let mut done = vec![];
    while done.len() < 2 {
        done.push(wait(&w, 20.0, |u| match u {
            Up::Done(p) => Some(p),
            _ => None,
        }));
    }
    for u in uris {
        let p = dest.join(album::name(u));
        assert!(done.contains(&p));
        assert!(std::fs::read(&p).unwrap() == std::fs::read(r.join(&u[1..])).unwrap());
    }

    // the card goes: the album empties, the source is the camera again
    std::fs::rename(r.join("DCIM"), r.join("gone")).unwrap();
    let i = wait(&w, 5.0, |u| match u {
        Up::Info(i) if i.src.is_none() => Some(i),
        _ => None,
    });
    assert!(i.disk.is_empty());
    assert_eq!(
        status(&i),
        "Insta360 camera in another USB mode: choose Android App on its USB menu"
    );
    assert!(album(&w).is_empty());
    drop(w);
    std::fs::remove_dir_all(&d).unwrap();
}

/// The real X5 over USB, through the worker and queue the UI uses. `X5_CAMERA=<scratch dir>`,
/// `X5_CAMERA_DL=<uri>[,<uri>…]` (full downloads, the first one also cancelled and resumed after a restart),
/// `X5_CAMERA_REF=<dir>` (reference copies by file name). Read-only on the camera apart from the HTTP downloads and
/// the clock, which every connect sets to this computer's (the connect notes say whether the camera took it).
#[test]
fn real_camera() {
    let Some(d) = std::env::var_os("X5_CAMERA").map(PathBuf::from) else {
        eprintln!("X5_CAMERA not set: skipping");
        return;
    };
    let uris: Vec<String> = std::env::var("X5_CAMERA_DL")
        .unwrap()
        .split(',')
        .map(Into::into)
        .collect();
    let refs = PathBuf::from(std::env::var_os("X5_CAMERA_REF").unwrap());
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let qf = d.join("downloads");
    let new = |q: dl::Queue| {
        let q = Arc::new(Mutex::new(q));
        let cache = Cache::at(Some(d.join("thumbs")));
        let auto = Arc::new(AtomicBool::new(false));
        let w = Worker::spawn(
            Box::new(Camera::usb),
            q.clone(),
            cache,
            auto,
            true,
            Taps::default(),
            || {},
        );
        (w, q)
    };
    let (w, q) = new(dl::Queue::load(Some(qf.clone())));
    wait(&w, 5.0, |u| match u {
        Up::Info(i) if i.plug == Some(Plug::App) => Some(()),
        _ => None,
    });
    println!("hotplug watch: X5 in app mode");
    let t = Instant::now();
    w.tx.send(Cmd::Connect).unwrap();
    let mut notes = vec![];
    let i = wait(&w, 5.0, |u| match u {
        Up::Note(s, e) => {
            println!("note: {s}");
            notes.push((s, e));
            None
        }
        Up::Info(i) if i.conn == Conn::On => Some(i),
        _ => None,
    });
    assert!(
        notes.iter().any(|n| n.0.contains("; camera clock set to "))
            && !notes.iter().any(|n| n.0.starts_with("time sync")),
        "the clock not set on connect: {notes:?}"
    );
    let id = i.id.unwrap();
    println!(
        "connected in {:.0} ms: {} {}, battery {}, storage {}, Wi-Fi {}, clock skew {:?} s",
        t.elapsed().as_secs_f64() * 1e3,
        id.model,
        id.firmware,
        i.battery.as_ref().map_or("?".into(), battery),
        i.storage.as_ref().map_or("?".into(), storage),
        i.wifi.as_ref().map_or("none", |_| "reported"),
        i.skew
    );
    let v = album(&w);
    let (vids, photos) = (
        v.iter().filter(|i| i.kind == Kind::Video).count(),
        v.iter().filter(|i| i.kind == Kind::Photo).count(),
    );
    let mut n = 0;
    let s = wait(&w, 120.0, |u| match u {
        Up::Thumb(_, Some(_)) => {
            n += 1;
            None
        }
        Up::Stats(s) => Some(s),
        _ => None,
    });
    println!(
        "album: {} items ({vids} videos, {photos} photos, {} files, {} favourites) listed in {:.0} ms",
        v.len(),
        v.iter().map(|i| i.uris.len()).sum::<usize>(),
        v.iter().filter(|i| i.fav).count(),
        s.list_ms
    );
    println!(
        "thumbnails, cold cache: {n} shown, {} from the camera ({:.1} MB) in {:.2} s = {:.1}/s, {:.1} MB/s",
        s.cam,
        s.bytes as f64 / 1e6,
        s.secs,
        s.cam as f64 / s.secs,
        s.bytes as f64 / 1e6 / s.secs
    );
    assert!(n > 0 && n == s.cam);

    let done = |w: &Worker, dst: &PathBuf| {
        wait(w, 300.0, |u| match u {
            Up::Done(p) if p == *dst => Some(()),
            Up::Note(s, true) => panic!("{s}"),
            _ => None,
        })
    };
    let same = |dst: &PathBuf| {
        let r = refs.join(dst.file_name().unwrap());
        let (a, b) = (std::fs::read(dst).unwrap(), std::fs::read(&r).unwrap());
        assert!(a == b, "{} differs from {}", dst.display(), r.display());
        a.len()
    };
    for u in &uris {
        let dst = d.join("full").join(album::name(u));
        q.lock().unwrap().add(u, dst.clone());
        let t = Instant::now();
        done(&w, &dst);
        let s = t.elapsed().as_secs_f64();
        let n = same(&dst);
        println!(
            "download {} ({} connections): {n} B in {s:.2} s = {:.1} MB/s, identical to the reference",
            album::name(u),
            dl::CONNS,
            n as f64 / s / 1e6
        );
    }

    // cancel at a third, quit, start again from the queue file, resume
    let u = &uris[0];
    let dst = d.join("resume").join(album::name(u));
    q.lock().unwrap().add(u, dst.clone());
    let prog = |q: &dl::Shared| {
        let g = q.lock().unwrap();
        let x = g.v.iter().find(|x| x.dst == dst).unwrap();
        (x.st, x.prog.done.load(Relaxed), x.prog.total.load(Relaxed))
    };
    let until = |q: &dl::Shared, f: &dyn Fn((St, u64, u64)) -> bool| {
        let end = Instant::now() + Duration::from_secs(120);
        loop {
            let p = prog(q);
            if f(p) {
                return p;
            }
            assert!(Instant::now() < end, "stuck at {p:?}");
            thread::sleep(Duration::from_millis(1));
        }
    };
    until(&q, &|p| p.0 == St::Running && p.2 > 0 && p.1 * 3 >= p.2);
    {
        let mut g = q.lock().unwrap();
        let i = g.v.iter().position(|x| x.dst == dst).unwrap();
        g.cancel(i);
    }
    let (_, had, total) = until(&q, &|p| p.0 == St::Paused);
    println!("cancelled at {had} of {total} B");
    assert!(had < total);
    let t = Instant::now();
    drop(w);
    println!(
        "worker quit (session closed) in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    let mut q = dl::Queue::load(Some(qf.clone()));
    let i = q.v.iter().position(|x| x.dst == dst).unwrap();
    assert_eq!(q.v[i].st, St::Paused);
    q.resume(i);
    let (w, q) = new(q);
    w.tx.send(Cmd::Connect).unwrap();
    conn(&w, Conn::On);
    let t = Instant::now();
    until(&q, &|p| p.0 == St::Running && p.2 > 0);
    thread::sleep(Duration::from_millis(5));
    let again = prog(&q).1;
    done(&w, &dst);
    let s = t.elapsed().as_secs_f64();
    same(&dst);
    println!(
        "resumed at {again} B after a restart: {} B more in {s:.2} s = {:.1} MB/s, identical to the reference",
        total - again,
        (total - again) as f64 / s / 1e6
    );
    assert!(again >= had);

    // unplug and replug, as far as software goes: disconnect, connect; the thumbnails come from the cache
    let t = Instant::now();
    w.tx.send(Cmd::Disconnect).unwrap();
    conn(&w, Conn::Off);
    println!("disconnected in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    w.tx.send(Cmd::Connect).unwrap();
    conn(&w, Conn::On);
    let n2 = album(&w).len();
    let s = wait(&w, 60.0, |u| match u {
        Up::Stats(s) => Some(s),
        _ => None,
    });
    println!(
        "reconnected and listed {n2} items in {:.0} ms; thumbnails: {} cached, {} from the camera, {:.2} s",
        t.elapsed().as_secs_f64() * 1e3,
        s.cached,
        s.cam,
        s.secs
    );
    assert_eq!((n2, s.cam), (v.len(), 0));
    w.tx.send(Cmd::Disconnect).unwrap();
    conn(&w, Conn::Off);
    drop(w);
}

/// The Camera screen in a window against the fake card, through the screenshots of `X5DESK_UI_SIM=<dir>` (thumbnails
/// from wave 1's real captures in `X5_LINK` if set): album with a selection, downloads in flight, the camera page, the
/// delete and format confirmations. Under XWayland when the display is off (no frame callbacks on Wayland then).
#[test]
fn ui_sim() {
    use winit::platform::{wayland::EventLoopBuilderExtWayland, x11::EventLoopBuilderExtX11};
    let Some(out) = std::env::var_os("X5DESK_UI_SIM").map(PathBuf::from) else {
        eprintln!("X5DESK_UI_SIM not set: skipping");
        return;
    };
    let d = dir("ui");
    let mut names: Vec<String> = NAMES.iter().map(|s| s.to_string()).collect();
    names.extend((0..14).map(|i| format!("IMG_20260118_12{i:02}46_00_{:03}.jpg", 649 + i)));
    names.extend((0..4).map(|i| format!("VID_20260117_13{i:02}31_00_{:03}.insv", 628 + i)));
    let f = card(&names.iter().map(String::as_str).collect::<Vec<_>>());
    {
        let mut c = f.lock().unwrap();
        c.favs = vec![(c.uris[2].clone(), true), (c.uris[9].clone(), true)];
        let small: Arc<Vec<u8>> = Arc::new((0..2_000_000u32).map(|i| i as u8).collect());
        let big: Arc<Vec<u8>> = Arc::new((0..40_000_000u32).map(|i| (i >> 3) as u8).collect());
        for u in c.uris.clone() {
            let b = if u.ends_with(".insv") { &big } else { &small };
            c.data.insert(u, b.clone());
        }
        if let Some(l) = std::env::var_os("X5_LINK").map(PathBuf::from) {
            let r = |n: &str| std::fs::read(l.join(n)).unwrap();
            c.th = Some((r("vthumb.bin"), r("thumb.bin")));
        }
    }
    let set = Set {
        dest: d.join("dl"),
        auto: false,
        file: None,
    };
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    std::fs::create_dir_all(&set.dest).unwrap();
    // one recording already here
    for u in [
        "VID_20260117_141148_00_635.insv",
        "LRV_20260117_141148_01_635.lrv",
    ] {
        std::fs::write(set.dest.join(u), b"x").unwrap();
    }

    struct Ui {
        cam: Option<(Fake, Set, dl::Shared, PathBuf)>,
        c: Option<Cam>,
        out: PathBuf,
        step: usize,
        frames: u32,
        /// when the step began: a modal fades in over a few frames, which take microseconds here
        t: Instant,
        shot: Arc<AtomicBool>,
        busy: bool,
    }
    impl eframe::App for Ui {
        fn ui(&mut self, ui: &mut egui::Ui, _: &mut eframe::Frame) {
            let ctx = ui.ctx().clone();
            if let Some((f, set, q, d)) = self.cam.take() {
                let ff = f.clone();
                let connect: Connect = Box::new(move || {
                    let f2 = ff.clone();
                    let files = move |u: &str| f2.lock().unwrap().data.get(u).cloned();
                    Camera::sim(handler(ff.clone()), files, 1_500_000)
                        .map(|(c, ev, _)| (c, ev))
                        .into()
                });
                let c = Cam::with(
                    &ctx,
                    None,
                    set,
                    q,
                    connect,
                    Cache::at(Some(d.join("thumbs"))),
                    false,
                );
                let _ = c.w.tx.send(Cmd::Connect);
                self.c = Some(c);
            }
            let c = self.c.as_mut().unwrap();
            c.tick();
            c.ui(ui);
            ctx.request_repaint_after(Duration::from_millis(20));
            if self.busy {
                if self.shot.swap(false, Relaxed) {
                    (self.busy, self.step, self.frames, self.t) =
                        (false, self.step + 1, 0, Instant::now());
                }
                return;
            }
            self.frames += 1;
            if self.frames % 100 == 0 {
                eprintln!("ui_sim: step {} frame {}", self.step, self.frames);
            }
            const STEPS: [&str; 5] = ["album", "downloads", "camera", "delete", "format"];
            let Some(name) = STEPS.get(self.step) else {
                return ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            };
            let ready = match self.step {
                0 if self.frames == 1 => false,
                0 => c.stats.is_some() && self.t.elapsed() > Duration::from_millis(600),
                1 if self.frames == 1 => {
                    // what the Download button does for the first three items shown
                    let picked: Vec<String> = c.shown[..3]
                        .iter()
                        .map(|&i| c.items[i].key.clone())
                        .collect();
                    c.sel = picked.iter().cloned().collect();
                    let mut q = c.q.lock().unwrap();
                    for it in c.items.iter().filter(|it| c.sel.contains(&it.key)) {
                        for u in &it.uris {
                            q.add(u, c.set.dest.join(album::name(u)));
                        }
                    }
                    false
                }
                1 => {
                    c.pane = Pane::Downloads;
                    c.q.lock()
                        .unwrap()
                        .v
                        .iter()
                        .any(|x| x.st == St::Running && x.rate > 0.0)
                        && self.frames > 30
                }
                2 => {
                    c.pane = Pane::Settings;
                    self.t.elapsed() > Duration::from_millis(600)
                }
                3 => {
                    c.pane = Pane::Album;
                    if self.frames == 1 {
                        let u: Vec<String> = c
                            .items
                            .iter()
                            .filter(|it| c.sel.contains(&it.key))
                            .flat_map(|it| it.uris.clone())
                            .collect();
                        c.ask = Some(Ask::Delete(u, c.sel.len(), 2));
                    }
                    self.t.elapsed() > Duration::from_millis(600)
                }
                _ => {
                    if self.frames == 1 {
                        c.ask = Some(Ask::Format);
                        c.typed = "FORM".into();
                    }
                    self.t.elapsed() > Duration::from_millis(600)
                }
            };
            if ready {
                let (p, s) = (self.out.join(format!("ui_{name}.png")), self.shot.clone());
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                    egui::ScreenshotCallback::new(move |img| {
                        let px: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                        let (w, h) = (img.size[0] as u32, img.size[1] as u32);
                        crate::png::write(&p, w, h, &px).unwrap();
                        println!("screenshot {w} × {h} → {}", p.display());
                        s.store(true, Relaxed);
                    }),
                ));
                self.busy = true;
            }
        }
    }

    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("x5desk camera test"),
        // no vsync: with the display off, presenting waits on a vblank that comes about once a second
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            surface: eframe::egui_wgpu::SurfaceConfig {
                present_mode: wgpu::PresentMode::AutoNoVsync,
                desired_maximum_frame_latency: Some(2),
            },
            ..Default::default()
        },
        event_loop_builder: Some(Box::new(|b| {
            EventLoopBuilderExtX11::with_any_thread(b, true);
            EventLoopBuilderExtWayland::with_any_thread(b, true);
        })),
        ..Default::default()
    };
    let ui = Ui {
        cam: Some((f.clone(), set, q, d.clone())),
        c: None,
        out,
        step: 0,
        frames: 0,
        t: Instant::now(),
        shot: Arc::new(AtomicBool::new(false)),
        busy: false,
    };
    eframe::run_native(
        "x5desk camera test",
        opts,
        Box::new(move |_| Ok(Box::new(ui))),
    )
    .unwrap();
    std::fs::remove_dir_all(&d).unwrap();
}
