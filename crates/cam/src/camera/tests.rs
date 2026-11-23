//! The camera worker against link's simulated camera (`sim`): the real I/O thread, UCD framing and tunnel HTTP, with
//! a fake card behind them. The commands that change the camera are checked here at the message level only; so are
//! the worker's guards (no delete while recording, a connect that leaves the clock).

use super::*;
use crate::{album::Kind, dl::St, sim::*};
use prost::Message;

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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    assert!(conn(&w, Conn::Connecting).since.is_some());
    thread::sleep(Duration::from_millis(300));
    let t = Instant::now();
    w.tx.send(Cmd::Disconnect).unwrap();
    conn(&w, Conn::Off);
    let c = t.elapsed();
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    // connected: identity and options come together
    let i = conn(&w, Conn::On);
    assert_eq!(i.id.as_ref().unwrap().model, "Insta360 X5");
    assert_eq!(i.wifi.unwrap().0, "X5 SIM001.OSC");
    assert!(i.skew.is_some_and(|s| (29..=31).contains(&s)));
    assert_eq!(i.storage.unwrap().total, 500_000_000_000);
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    let b = pb::BatteryNote {
        battery: Some(pb::Battery {
            level: 42,
            ..Default::default()
        }),
    };
    sim.note(8195, &b.encode_to_vec());
    wait(&w, 2.0, |u| match u {
        Up::Info(i) if i.battery.as_ref().is_some_and(|b| b.level == 42) => Some(()),
        _ => None,
    });
    // a file appeared (capture stopped): the album is listed again
    f.lock()
        .unwrap()
        .uris
        .push("/DCIM/Camera01/IMG_20260118_120546_00_649.jpg".into());
    sim.note(8201, &pb::StoppedNote::default().encode_to_vec());
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

/// Files the camera lists under names that are no safe file names (out of the download folder on Windows, a device)
/// stay out of the album, said in a note; so does one that appears later.
#[test]
fn unsafe_names() {
    let bad = [
        r"..\..\AppData\Roaming\x.lrv",
        r"C:\Users\Public\x.lrv",
        "CON.lrv",
    ];
    let f = card(&[&NAMES[..2], &bad[..]].concat());
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, sim) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect { clock: false }).unwrap();
    let n = wait(&w, 5.0, |u| match u {
        Up::Note(s, true) if s.starts_with("album:") => Some(s),
        _ => None,
    });
    assert!(
        n.contains("3 files left out") && n.contains("CON.lrv"),
        "{n}"
    );
    let v = album(&w);
    assert_eq!(v.len(), 1, "{v:?}");
    assert_eq!(v[0].uris.len(), 2);
    assert!(
        v.iter()
            .flat_map(|i| &i.uris)
            .all(|u| album::safe(album::name(u)))
    );
    f.lock().unwrap().uris.push("/DCIM/Camera01/nul.lrv".into());
    sim.lock()
        .unwrap()
        .clone()
        .unwrap()
        .note(8201, &pb::StoppedNote::default().encode_to_vec());
    assert_eq!(album(&w).len(), 1);
}

/// A low battery warns (20 % or less, not charging): the battery-low notification gives an error note and the level
/// the screens show; 21 % and a charging camera do not.
#[test]
fn low_battery() {
    use pb::battery::Power::{self, *};
    let f = card(&NAMES[..1]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, sim) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    assert!(!low(conn(&w, Conn::On).battery.as_ref().unwrap()));
    let sim = sim.lock().unwrap().clone().unwrap();
    let b = |l, p: Power| {
        Some(pb::Battery {
            level: l,
            power: p as i32,
            ..Default::default()
        })
    };
    let n = pb::BatteryLowNote {
        battery: b(9, Battery),
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
        let n = pb::BatteryNote { battery: b(l, p) };
        sim.note(8195, &n.encode_to_vec());
        let i = wait(&w, 2.0, |u| match u {
            Up::Info(i) if i.battery == b(l, p) => Some(i),
            _ => None,
        });
        assert_eq!(low(i.battery.as_ref().unwrap()), want, "{l} % {p:?}");
    }
    drop(w);
}

/// Every connect sets the camera clock once, in the app's order (§5.1, §5.3): identity, options and capture status,
/// then its two SET_OPTIONS, then the listing; the note says so once both answered. A camera that refuses it still
/// connects and lists its album, the refusal in an error note after the connect note.
#[test]
#[cfg(unix)] // the clock is set on unix only
fn connect_time() {
    let f = card(&NAMES[..2]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    let run = || {
        let n0 = f.lock().unwrap().log.len();
        w.tx.send(Cmd::Connect { clock: true }).unwrap();
        let (mut notes, mut n) = (vec![], None);
        wait(&w, 10.0, |u| {
            match u {
                Up::Note(s, e) => notes.push((s, e)),
                Up::Album(v) => n = Some(v.len()),
                _ => {}
            }
            let clk = |x: &(String, bool)| x.0.contains("clock") || x.0.starts_with("time sync");
            (n.is_some() && notes.iter().any(clk)).then_some(())
        });
        let l: Vec<(u16, Vec<u8>)> = (f.lock().unwrap().log[n0..].iter())
            .map(|l| (l.1, l.2.clone()))
            .collect();
        (notes, n.unwrap(), l)
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
        use link::code::*;
        assert!(s.len() == 2 && s[1] == s[0] + 1, "{k:?}");
        assert_eq!(
            k[..s[0]],
            [GET_OPTIONS, GET_OPTIONS, CAPTURE_STATUS],
            "{k:?}"
        );
        assert_eq!(k.get(s[1] + 1), Some(&GET_FILE_LIST), "{k:?}");
        let set: Vec<pb::WriteSettings> = (s.iter())
            .map(|&i| pb::WriteSettings::decode(&l[i].1[..]).unwrap())
            .collect();
        assert_eq!(
            (&set[0].keys[..], &set[1].keys[..]),
            (&[12, 13, 175][..], &[175][..])
        );
        let o = set[0].settings.as_ref().unwrap();
        assert!(o.clock.abs_diff(t) <= 5);
        assert_eq!((o.utc_offset, &*o.zone), (off, &*tz));
        assert_eq!(set[1].settings.as_ref().unwrap().zone, tz);
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

/// A camera that never answers the clock's SET_OPTIONS holds nothing up (the connect waited for each answer, up to
/// 10 s, with Disconnect and Quit unhandled): the album is listed and listed again meanwhile, Disconnect and the
/// worker's end are handled at once, and the connect's notes come with the answers' timeout.
#[test]
#[cfg(unix)] // the clock is set on unix only
fn silent_time() {
    let f = card(&NAMES[..2]);
    f.lock().unwrap().mute = vec![link::code::SET_OPTIONS];
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    assert_eq!(album(&w).len(), 1);
    let listed = ms(t);
    assert_eq!(count(&f, link::code::SET_OPTIONS), 2, "both sent");
    let t = Instant::now();
    w.tx.send(Cmd::Refresh).unwrap();
    album(&w);
    let again = ms(t);
    let t = Instant::now();
    w.tx.send(Cmd::Disconnect).unwrap();
    let mut notes = vec![];
    wait(&w, 5.0, |u| match u {
        Up::Note(s, _) if s == "disconnected" => Some(()),
        Up::Note(s, e) => {
            notes.push((s, e));
            None
        }
        _ => None,
    });
    let off = ms(t);
    assert!(notes.is_empty(), "{notes:?}");

    let (t, g) = (Instant::now(), count(&f, link::code::GET_OPTIONS));
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    album(&w);
    let mut notes = vec![];
    wait(&w, 15.0, |u| match u {
        Up::Note(s, e) => {
            notes.push((s, e));
            (notes.len() == 2).then_some(())
        }
        _ => None,
    });
    let late = ms(t);
    assert!(
        notes[0].0.starts_with("connected to Insta360 X5 in ")
            && !notes[0].1
            && notes[1] == ("time sync: timeout".into(), true),
        "{notes:?}"
    );
    // the options not read again after the timeout: the connect's two only
    assert_eq!(count(&f, link::code::GET_OPTIONS), g + 2);

    w.tx.send(Cmd::Disconnect).unwrap();
    conn(&w, Conn::Off);
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    album(&w);
    let t = Instant::now();
    drop(w);
    let gone = ms(t);
    println!(
        "silent clock: listed in {listed:.0} ms and again in {again:.0} ms, disconnected in {off:.0} ms, its \
         timeout noted after {late:.0} ms, worker gone in {gone:.0} ms"
    );
    assert!(listed < 2000.0 && again < 1000.0 && off < 1000.0 && gone < 2000.0);
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    until(&q, &|p| p.0 == St::Running && p.1 >= 1_000_000);
    let sim = sim.lock().unwrap().clone().unwrap();
    let st = |s: i32| {
        pb::CaptureNote {
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

/// At quit nothing new starts: the download the session's close cuts goes back to the queue, and the import queued
/// behind it (its file there) waits for the next run instead of starting while the worker waits for that end.
#[test]
fn quit_import() {
    let d = dir("quit");
    let name = "VID_20260116_152935_00_603.insv";
    let f = card(&[name]);
    let uri = format!("/DCIM/Camera01/{name}");
    f.lock()
        .unwrap()
        .data
        .insert(uri.clone(), Arc::new(vec![5; 8_000_000]));
    let src = d.join("card/DCIM/Camera01/IMG_20260118_124432_00_659.jpg");
    std::fs::create_dir_all(src.parent().unwrap()).unwrap();
    std::fs::write(&src, vec![9u8; 100_000]).unwrap();
    let (qf, dst) = (
        d.join("downloads"),
        d.join("IMG_20260118_124432_00_659.jpg"),
    );
    let mut q = dl::Queue::load(Some(qf.clone()));
    q.add(&uri, d.join(name));
    q.add(&format!("file://{}", src.display()), dst.clone());
    let q = Arc::new(Mutex::new(q));
    // 4 × 1 MB/s: 2 s for the download
    let (w, _) = spawn(&f, 1_000_000, q.clone(), Cache::at(None));
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    until(&q, &|p| p.0 == St::Running && p.1 > 0);
    drop(w);
    let st: Vec<St> = (dl::Queue::load(Some(qf)).v.iter()).map(|x| x.st).collect();
    thread::sleep(Duration::from_millis(200));
    let mut part = dst.clone().into_os_string();
    part.push(".part");
    println!("after quit: {st:?}");
    assert_eq!(st, [St::Queued, St::Queued]);
    assert!(
        !dst.exists() && !Path::new(&part).exists(),
        "the import ran"
    );
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    assert_eq!(album(&w).len(), 9);
    connected(&w);
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
            .map(|l| pb::DeleteFiles::decode(&l.2[..]).unwrap().uris.len())
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

/// Delete in batches of 200, 150 ms apart; favourites; format, each followed by a fresh album; time sync (unix: two
/// SET_OPTIONS, §5.3, then the clock read).
#[test]
fn commands() {
    let names: Vec<String> = (0..450)
        .map(|i| format!("IMG_20260118_12{:02}{:02}_00_{i:03}.jpg", i / 60, i % 60))
        .collect();
    let f = card(&names.iter().map(String::as_str).collect::<Vec<_>>());
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    assert_eq!(album(&w).len(), 450);
    connected(&w);
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
        .map(|l| (l.0, pb::DeleteFiles::decode(&l.2[..]).unwrap().uris))
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
    let s = &pb::SetFavorites::decode(&s.2[..]).unwrap().flags[0];
    assert_eq!((&*s.path, s.on), (&*u, 1));
    assert!((s.modified - clock().1).abs() <= 2);

    w.tx.send(Cmd::Format).unwrap();
    after(&w);
    let l = f.lock().unwrap().log.clone();
    assert!(
        l.iter()
            .any(|l| l.1 == link::code::ERASE_SD && l.2.is_empty())
    );

    // the clock: unix only (`no_zone` elsewhere)
    #[cfg(unix)]
    {
        let n0 = f.lock().unwrap().log.len();
        w.tx.send(Cmd::Time).unwrap();
        let s = wait(&w, 10.0, |u| match u {
            Up::Note(s, e) => Some((s, e)),
            _ => None,
        });
        assert!(!s.1 && s.0.starts_with("camera clock set to "), "{s:?}");
        let l = f.lock().unwrap().log[n0..].to_vec();
        use link::code::*;
        let k: Vec<u16> = (l.iter().map(|l| l.1))
            .filter(|&c| c != GET_FILE_EXTRA)
            .collect();
        assert_eq!(k, [SET_OPTIONS, SET_OPTIONS, GET_OPTIONS]);
        let set: Vec<pb::WriteSettings> = l
            .iter()
            .filter(|l| l.1 == link::code::SET_OPTIONS)
            .map(|l| pb::WriteSettings::decode(&l.2[..]).unwrap())
            .collect();
        let (t, _, off, tz, _) = clock();
        assert_eq!(set.len(), 2);
        assert_eq!(
            (&set[0].keys[..], &set[1].keys[..]),
            (&[12, 13, 175][..], &[175][..])
        );
        let o = set[0].settings.as_ref().unwrap();
        assert!(o.clock.abs_diff(t) <= 2);
        assert_eq!((o.utc_offset, &*o.zone), (off, &*tz));
        assert_eq!(set[1].settings.as_ref().unwrap().zone, tz);
    }
    drop(w);
}

/// The local clock helper agrees with the date tool: the offset now, and the standard one (no DST, the smaller of
/// mid-January's and mid-July's) in the favourite timestamp.
#[test]
#[cfg(unix)] // no IANA zone or local time elsewhere
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

/// No delete while the camera records (§7.12: its files are off limits): the album keeps what it showed, its Delete
/// is refused with a note and nothing goes out; once the recording stops it goes through.
#[test]
fn delete_recording() {
    let f = card(&NAMES[..3]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, sim) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    let v = album(&w);
    connected(&w);
    let sim = sim.lock().unwrap().clone().unwrap();
    let st = |s: i32| {
        pb::CaptureNote {
            state: s,
            ..Default::default()
        }
        .encode_to_vec()
    };
    let note = |w: &Worker| {
        wait(w, 10.0, |u| match u {
            Up::Note(s, e) => Some((s, e)),
            _ => None,
        })
    };
    sim.note(8208, &st(1));
    wait(&w, 2.0, |u| match u {
        Up::Info(i) if i.busy => Some(()),
        _ => None,
    });
    let u = v[0].uris.clone();
    w.tx.send(Cmd::Delete(u.clone())).unwrap();
    let s = note(&w);
    println!("recording: {s:?}");
    assert!(s.1 && s.0.contains("recording"), "{s:?}");
    assert_eq!(count(&f, link::code::DELETE_FILES), 0);
    sim.note(8208, &st(0));
    wait(&w, 2.0, |u| match u {
        Up::Info(i) if !i.busy => Some(()),
        _ => None,
    });
    w.tx.send(Cmd::Delete(u.clone())).unwrap();
    assert_eq!(
        note(&w),
        (format!("deleted {0} of {0} files", u.len()), false)
    );
    assert!(count(&f, link::code::DELETE_FILES) > 0);
    drop(w);
}

/// A connect asked to leave the camera clock (an agent's) sends no SET_OPTIONS, and its notes come all the same; so
/// do the worker's reconnects after a lost link (automatic, the X5 still in app mode): after 2 s, and at once when the
/// camera comes back on the bus (a glitch re-enumerates it). The app's own connect still sets it (unix), so does its
/// auto-connect of a camera plugged in after a disconnect.
#[test]
fn connect_no_clock() {
    let f = card(&NAMES[..2]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, sim) = spawn(&f, 50_000_000, q, Cache::at(None));
    let set = || count(&f, link::code::SET_OPTIONS);
    let unplug = || sim.lock().unwrap().as_ref().unwrap().unplug();
    w.tx.send(Cmd::Connect { clock: false }).unwrap();
    assert_eq!(album(&w).len(), 1);
    let n = connected(&w);
    assert!(!n.contains("clock"), "{n}");
    // on the bus in app mode, auto-connect on: the link lost is connected again after 2 s
    w.tx.send(Cmd::Plug(Plug::App)).unwrap();
    unplug();
    conn(&w, Conn::Off);
    wait(&w, 10.0, |u| match u {
        Up::Info(i) if i.conn == Conn::On => Some(()),
        _ => None,
    });
    let n = connected(&w);
    println!("reconnected: {n}");
    assert!(!n.contains("clock"), "{n}");
    assert_eq!(set(), 0, "no SET_OPTIONS");
    // the camera off the bus and back (no timer: it was not in app mode when the link went)
    w.tx.send(Cmd::Plug(Plug::None)).unwrap();
    unplug();
    conn(&w, Conn::Off);
    w.tx.send(Cmd::Plug(Plug::App)).unwrap();
    let t = Instant::now();
    let n = connected(&w);
    println!(
        "back on the bus: {n} after {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    assert!(
        !n.contains("clock") && t.elapsed() < Duration::from_secs(2),
        "{n}"
    );
    assert_eq!(set(), 0, "no SET_OPTIONS");
    w.tx.send(Cmd::Disconnect).unwrap();
    conn(&w, Conn::Off);
    #[cfg(unix)]
    {
        w.tx.send(Cmd::Connect { clock: true }).unwrap();
        let n = connected(&w);
        assert!(n.contains("; camera clock set to "), "{n}");
        assert_eq!(set(), 2);
        // an agent's session closed: a camera plugged in after it is the app's to connect, with the clock
        w.tx.send(Cmd::Disconnect).unwrap();
        conn(&w, Conn::Off);
        w.tx.send(Cmd::Connect { clock: false }).unwrap();
        connected(&w);
        w.tx.send(Cmd::Disconnect).unwrap();
        conn(&w, Conn::Off);
        w.tx.send(Cmd::Plug(Plug::None)).unwrap();
        w.tx.send(Cmd::Plug(Plug::App)).unwrap();
        let n = connected(&w);
        assert!(n.contains("; camera clock set to "), "{n}");
        assert_eq!(set(), 4);
    }
    drop(w);
}

/// The Delete guard asks the camera too, its notification aside: a capture it reports without one (one just started,
/// its note on the way; one a card's album left unread at the connect) refuses the Delete and, a video one, puts the
/// files off limits; an interval or starlapse refuses it and leaves the album listed; idle, the Delete goes through.
#[test]
fn delete_unnoted() {
    use pb::CaptureState as C;
    let f = card(&NAMES[..3]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, sim) = spawn(&f, 50_000_000, q, Cache::at(None));
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    let u = album(&w)[0].uris.clone();
    connected(&w);
    let sim = sim.lock().unwrap().clone().unwrap();
    use link::code::{CAPTURE_STATUS, DELETE_FILES};
    // the Delete's note, whether an Info before it showed the camera busy, the status reads and deletes it made
    let del = |st: C| {
        f.lock().unwrap().cap = st as i32;
        let (s0, d0) = (count(&f, CAPTURE_STATUS), count(&f, DELETE_FILES));
        w.tx.send(Cmd::Delete(u.clone())).unwrap();
        let mut busy = false;
        let s = wait(&w, 10.0, |x| match x {
            Up::Info(i) => {
                busy |= i.busy;
                None
            }
            Up::Note(s, e) => Some((s, e)),
            _ => None,
        });
        let n = (count(&f, CAPTURE_STATUS) - s0, count(&f, DELETE_FILES) - d0);
        println!("{st:?}: {s:?}, busy {busy}, {n:?}");
        (s, busy, n)
    };
    let (s, busy, n) = del(C::Recording);
    assert!(
        s.1 && s.0.contains("recording") && busy && n == (1, 0),
        "{s:?}"
    );
    // its note that it stopped: the album listed again
    f.lock().unwrap().cap = 0;
    let idle = pb::CaptureNote::default().encode_to_vec();
    sim.note(8208, &idle);
    album(&w);
    for st in [C::Starlapse, C::IntervalPhotos] {
        let (s, busy, n) = del(st);
        assert!(
            s.1 && s.0.contains("recording") && !busy && n == (1, 0),
            "{s:?}"
        );
    }
    let (s, _, n) = del(C::Idle);
    assert_eq!(s, (format!("deleted {0} of {0} files", u.len()), false));
    // its fresh listing may have asked for the status too by now
    assert!(n.0 >= 1 && n.1 > 0, "{n:?}");
    drop(w);
}

/// No IANA zone known (not unix): a connect asked to set the camera clock leaves it, said once for the worker's
/// connects, and Sync time is refused; nothing of the clock goes out.
#[test]
#[cfg(not(unix))]
fn no_zone() {
    let f = card(&NAMES[..2]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    let mut said = vec![];
    for _ in 0..2 {
        w.tx.send(Cmd::Connect { clock: true }).unwrap();
        // the connect's notes come after the listing
        album(&w);
        w.tx.send(Cmd::Disconnect).unwrap();
        wait(&w, 5.0, |u| match u {
            Up::Note(s, _) if s == "disconnected" => Some(()),
            Up::Note(s, e) => {
                said.push((s, e));
                None
            }
            _ => None,
        });
    }
    println!("notes: {said:?}");
    let not = |s: &(String, bool)| s.0.starts_with("camera clock not set");
    assert!(
        said.len() == 3 && said.iter().filter(|s| not(s)).count() == 1 && said.iter().all(|s| !s.1),
        "{said:?}"
    );
    assert!(
        said[0].0.starts_with("connected") && not(&said[1]) && said[2].0.starts_with("connected")
    );
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    connected(&w);
    w.tx.send(Cmd::Time).unwrap();
    let s = wait(&w, 5.0, |u| match u {
        Up::Note(s, e) => Some((s, e)),
        _ => None,
    });
    assert!(
        s.1 && s.0.starts_with("time sync: no IANA time zone"),
        "{s:?}"
    );
    assert_eq!(count(&f, link::code::SET_OPTIONS), 0);
    drop(w);
}
