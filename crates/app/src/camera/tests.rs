//! The Camera screen on the camera worker (`cam::camera`), against cam's simulated camera (`cam::sim`) or with the
//! test standing in for the worker: the album's textures and tiles, the Library's list, the Downloads pane, a card's
//! album and the bar, the Clock row; the real X5; the screen in a window. The worker's own tests are cam's.

use super::*;
use cam::sim::*;
use link::Error;
use std::{
    sync::mpsc::Sender,
    time::{Instant, UNIX_EPOCH},
};

/// A camera that reports its clock (setting `clock`; the X5 leaves it out): the Clock row shows the skew read on connect
/// until the clock is set, then the clock as set, by the time "camera clock set to …" is noted; the same after Sync
/// time; after a refused sync it shows the clock still behind, after one that took the clock and refused the zone the
/// clock as set, next to the refusal.
#[test]
fn clock_row() {
    let f = card(&NAMES[..2]);
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let (w, _) = spawn(&f, 50_000_000, q, Cache::at(None));
    // the skew by the clock's note, and the note
    let row = |w: &Worker, mut s: Option<i64>| {
        let n = wait(w, 10.0, |u| match u {
            Up::Info(i) => {
                s = i.skew;
                None
            }
            Up::Note(n, e) if n.contains("camera clock set to ") || e => Some(n),
            _ => None,
        });
        (s, n)
    };
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
    let i = conn(&w, Conn::On);
    assert!(
        i.skew.is_some_and(|s| (29..=31).contains(&s)),
        "{:?}",
        i.skew
    );
    let ok = Some("in sync with this computer".to_string());
    assert_eq!(row(&w, i.skew).0.map(skew), ok);
    // the camera's clock drifts 100 s behind
    f.lock().unwrap().clk = Some((clock().0 - 100, Instant::now()));
    w.tx.send(Cmd::Refresh).unwrap();
    let i = wait(&w, 5.0, |u| match u {
        Up::Info(i) if i.skew.is_some_and(|s| s < -90) => Some(i),
        _ => None,
    });
    w.tx.send(Cmd::Time).unwrap();
    assert_eq!(row(&w, i.skew).0.map(skew), ok);
    f.lock().unwrap().clk = Some((clock().0 - 100, Instant::now()));
    f.lock().unwrap().refuse = vec![link::code::SET_OPTIONS];
    w.tx.send(Cmd::Refresh).unwrap();
    let i = wait(&w, 5.0, |u| match u {
        Up::Info(i) if i.skew.is_some_and(|s| s < -90) => Some(i),
        _ => None,
    });
    w.tx.send(Cmd::Time).unwrap();
    // still 100 s behind (± the second two reads can straddle)
    let (s, n) = row(&w, i.skew);
    let no = "time sync: status 500";
    assert!(
        s.is_some_and(|s| (-101..=-99).contains(&s)) && n.starts_with(no),
        "{s:?} {n}"
    );
    f.lock().unwrap().refuse.clear();
    f.lock().unwrap().zone = true;
    w.tx.send(Cmd::Time).unwrap();
    let (s, n) = row(&w, i.skew);
    assert!(s.map(skew) == ok && n.starts_with(no), "{s:?} {n}");
    drop(w);
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
    c.w = Worker::fake(tx, rx);
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

/// A card's album tiles: a flat photo's thumbnail (square, or wider than 2:1) keeps its aspect centred in the 2:1 box,
/// a 360 one fills it; a double click opens a photo from the card as it does a video.
#[test]
fn tiles() {
    let d = dir("tiles");
    let card = d.join("X5");
    let (mut c, up, _cmd) = headless(d.join("dl"), Arc::new(Mutex::new(dl::Queue::load(None))));
    c.info.src = Some(card.clone());
    let it = |k: &str, kind, n: &str, date| Item {
        key: k.into(),
        kind,
        uris: vec![format!("/DCIM/Camera01/{n}")],
        date,
        ..Default::default()
    };
    c.items = vec![
        it("sq", Kind::Photo, "IMG_20260118_124432_00_659.jpg", 3),
        it("wide", Kind::Photo, "IMG_20260118_124433_00_660.jpg", 2),
        it("pano", Kind::Video, "VID_20260118_124434_00_661.insv", 1),
    ];
    c.dirty = true;
    for (k, w, h) in [("sq", 160, 160), ("wide", 320, 135), ("pano", 320, 160)] {
        up.send(Up::Thumb(
            k.into(),
            Some((w, h, vec![200u8; (w * h * 4) as usize])),
        ))
        .unwrap();
    }
    c.tick();
    let mut t = 0.0;
    let mut frame = |c: &mut Cam, ev: Vec<egui::Event>| {
        t += 0.05;
        let i = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            time: Some(t),
            events: ev,
            ..Default::default()
        };
        let mut open = None;
        let mut o = c.ctx.clone().run_ui(i, |ui| open = c.album(ui));
        o.textures_delta.clear();
        (o.shapes, open)
    };
    fn meshes(s: &egui::Shape, v: &mut Vec<(egui::TextureId, egui::Rect)>) {
        match s {
            egui::Shape::Mesh(m) => v.push((
                m.texture_id,
                egui::Rect::from_points(&m.vertices.iter().map(|x| x.pos).collect::<Vec<_>>()),
            )),
            egui::Shape::Vec(x) => x.iter().for_each(|s| meshes(s, v)),
            _ => {}
        }
    }
    let (shapes, _) = frame(&mut c, vec![]);
    let mut m = vec![];
    shapes.iter().for_each(|x| meshes(&x.shape, &mut m));
    let at = |k: &str| {
        let id = c.tex[k].0.as_ref().unwrap().id();
        m.iter().find(|x| x.0 == id).expect(k).1
    };
    let (sq, wide, pano) = (at("sq"), at("wide"), at("pano"));
    println!("thumbnails drawn at: square {sq:?}, wide {wide:?}, 360 {pano:?}");
    // the tiles are 200 wide, the box 200 × 100
    let near = |a: f32, b: f32| (a - b).abs() < 0.5;
    assert!(near(pano.width(), 200.0) && near(pano.height(), 100.0));
    assert!(near(sq.width(), 100.0) && near(sq.height(), 100.0));
    assert!(near(wide.width(), 200.0) && near(wide.height(), 200.0 * 135.0 / 320.0));
    // centred in a box of the same top as the 360 one's
    for r in [sq, wide] {
        assert!(near(r.center().y - pano.min.y, 50.0), "{r:?}");
    }
    assert!(
        near(sq.center().x - (pano.min.x - 210.0 * 2.0), 100.0),
        "{sq:?}"
    );

    // double click on the square photo: opened from the card
    let p = sq.center();
    let btn = |pressed| egui::Event::PointerButton {
        pos: p,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Default::default(),
    };
    let mut open = vec![];
    for ev in [
        vec![egui::Event::PointerMoved(p)],
        vec![btn(true)],
        vec![btn(false)],
        vec![btn(true)],
        vec![btn(false)],
        vec![],
    ] {
        open.extend(frame(&mut c, ev).1);
    }
    assert_eq!(
        open,
        [card.join("DCIM/Camera01/IMG_20260118_124432_00_659.jpg")]
    );
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
    // in U-disk mode the X5 is not on the bus under its vendor id: the bar still says what is shown
    let full = "choose Android App on its USB menu for the full features (settings, capture, live preview)";
    assert_eq!(i.plug, None);
    assert_eq!(
        status(&i),
        format!("Camera in U-disk mode: the Album shows its card; {full}")
    );
    let mut j = i.clone();
    j.src = None;
    assert_eq!(
        status(&j),
        format!("Camera in U-disk mode: pick its card in the Album; {full}")
    );
    (j.src, j.disk[0].cam) = (Some(r.clone()), None);
    assert_eq!(
        status(&j),
        "No camera: the Album shows a card; connect the X5 by USB in app mode"
    );
    // the X5 on the bus as a disk (070a:4027) not mounted yet, or the first seconds of app mode
    let k = Info {
        plug: Some(Plug::Disk),
        ..Default::default()
    };
    assert_eq!(
        status(&k),
        format!(
            "Camera in U-disk mode (or starting): its card shows in the Album once mounted; {full}"
        )
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    // the connect's notes come once the clock's answers did, after the listing
    let (mut n, mut st) = (0, None);
    let clk =
        |x: &(String, bool)| x.0.contains("; camera clock set to ") || x.0.starts_with("time sync");
    wait(&w, 120.0, |u| {
        match u {
            Up::Thumb(_, Some(_)) => n += 1,
            Up::Stats(s) => st = Some(s),
            Up::Note(s, e) => {
                println!("note: {s}");
                notes.push((s, e))
            }
            _ => {}
        }
        (st.is_some() && notes.iter().any(clk)).then_some(())
    });
    assert!(
        notes.iter().any(|n| n.0.contains("; camera clock set to "))
            && !notes.iter().any(|n| n.0.starts_with("time sync")),
        "the clock not set on connect: {notes:?}"
    );
    let s = st.unwrap();
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
    w.tx.send(Cmd::Connect { clock: true }).unwrap();
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
/// Linux only: elsewhere winit runs its event loop on the main thread alone.
#[cfg(target_os = "linux")]
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
                let _ = c.w.tx.send(Cmd::Connect { clock: true });
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
