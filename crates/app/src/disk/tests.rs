//! Mount detection against a fake mount table and sysfs; a fake card (a DCIM tree in the fixtures directory, with the
//! samples of `X5_SAMPLES` as its recordings when set); the import copy's resume rules, and imports through the
//! download queue.

use super::*;
use crate::dl::{self, St};
use std::{
    os::unix::fs::symlink,
    sync::{Arc, Mutex},
    time::{Duration, Instant, UNIX_EPOCH},
};

fn dir(n: &str) -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join(format!("x5desk-{n}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The camera's disk by its USB product string (wherever it is mounted, a space in the path), a card under a
/// removable-media place; not: a USB stick elsewhere, a volume without DCIM, a non-device source, a second mount.
#[test]
fn mounts() {
    let d = dir("mounts");
    let sys = d.join("sys");
    let bus = "devices/pci0000:00/0000:00:14.0/usb2";
    for (port, product, h, b) in [
        ("2-1", "Insta360 X5", 5, "sdb"),
        ("2-2", "Ultra Fit", 6, "sdc"),
    ] {
        let u = sys.join(bus).join(port);
        let blk = format!("{port}:1.0/host{h}/target{h}:0:0/{h}:0:0:0/block/{b}/{b}1");
        fs::create_dir_all(u.join(&blk)).unwrap();
        fs::write(u.join("idVendor"), "2e1a\n").unwrap();
        fs::write(u.join("product"), format!("{product}\n")).unwrap();
        fs::create_dir_all(sys.join("class/block")).unwrap();
        symlink(
            format!("../../{bus}/{port}/{blk}"),
            sys.join("class/block").join(format!("{b}1")),
        )
        .unwrap();
    }
    let media = d.join("media");
    let (x5, stick, card, empty) = (
        d.join("mnt/X5 CARD"),
        d.join("mnt/stick"),
        media.join("user/SD"),
        media.join("user/EMPTY"),
    );
    for p in [&x5, &stick, &card] {
        fs::create_dir_all(p.join("DCIM/Camera01")).unwrap()
    }
    fs::create_dir_all(&empty).unwrap();
    let e = |p: &Path| p.display().to_string().replace(' ', "\\040");
    let info = format!(
        "22 1 0:21 / /proc rw,nosuid shared:5 - proc proc rw\n\
         30 1 0:31 /@ / rw,relatime shared:1 - btrfs /dev/mapper/root rw\n\
         90 29 8:17 / {x5} rw,nosuid,nodev shared:300 - exfat /dev/sdb1 rw,uid=1000\n\
         91 29 8:33 / {stick} rw,nosuid - vfat /dev/sdc1 rw\n\
         92 29 179:1 / {card} rw,nosuid - exfat /dev/mmcblk0p1 rw\n\
         93 29 179:2 / {empty} rw - exfat /dev/mmcblk0p2 rw\n\
         94 29 0:50 / {tmp} rw - tmpfs tmpfs rw\n\
         95 29 8:17 / {x5} rw - exfat /dev/sdb1 rw\n",
        x5 = e(&x5),
        stick = e(&stick),
        card = e(&card),
        empty = e(&empty),
        tmp = e(&card),
    );
    let v = vols(&info, &sys, std::slice::from_ref(&media));
    assert_eq!(
        v,
        [
            Vol {
                root: x5,
                dev: "/dev/sdb1".into(),
                cam: Some("Insta360 X5".into()),
            },
            Vol {
                root: card,
                dev: "/dev/mmcblk0p1".into(),
                cam: None,
            },
        ]
    );
    assert_eq!(
        (v[0].name(), v[1].name()),
        ("Insta360 X5 (X5 CARD)".into(), "Card SD".into())
    );
    // mounted under its label, which is the product string: named once
    let x5 = Vol {
        root: media.join("user/Insta360 X5"),
        ..v[0].clone()
    };
    assert_eq!(x5.name(), "Insta360 X5");
    // this machine's table parses, whatever is mounted
    println!(
        "mounted camera disks and cards here: {:?}",
        Watch::new(vec![]).vols()
    );
    fs::remove_dir_all(&d).unwrap();
}

/// A card's DCIM as the album shows it: the camera's grouping, metadata and thumbnails from the trailers (one per
/// item: its first file's), the favourites file; files the app would not collect and empty ones left out (an
/// interrupted recording: a 0-byte main file beside its proxy).
#[test]
fn card() {
    let d = dir("card");
    let (r, c) = (d.join("X5"), d.join("X5/DCIM/Camera01"));
    for p in ["DCIM/Camera01", "DCIM/Camera02", "DCIM/.trash"] {
        fs::create_dir_all(r.join(p)).unwrap();
    }
    let s = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    for (n, f) in [
        ("VID_20260117_141148_00_635.insv", "v635.insv"),
        ("LRV_20260117_141148_01_635.lrv", "v635.lrv"),
        ("VID_20260116_152935_00_606.insv", "v606.insv"),
        ("LRV_20260503_190639_01_708.lrv", "v635.lrv"),
    ] {
        match &s {
            Some(s) => symlink(s.join(f), c.join(n)).unwrap(),
            None => fs::write(c.join(n), b"no trailer").unwrap(),
        }
    }
    for n in [
        "IMG_20260118_124432_00_659.jpg",
        "IMG_20260118_124432_00_659.dng",
        "notes.txt",
        "VID__20260118_124432_00_1.insv",
        "VID_1.insv",
        "IMG_20260118_124432_00_660.jpg.part",
    ] {
        fs::write(c.join(n), b"x").unwrap();
    }
    for n in [
        "VID_20260503_190639_00_708.insv",
        "IMG_20260504_101010_00_709.jpg",
    ] {
        fs::write(c.join(n), b"").unwrap();
    }
    fs::write(
        r.join("DCIM/Camera02/IMG_20260119_101010_00_700.jpg"),
        b"yy",
    )
    .unwrap();
    fs::write(r.join("DCIM/.trash/VID_20260101_000000_00_001.insv"), b"z").unwrap();
    let ed = pb::EditFileInfoList {
        edit_file_info_list: vec![pb::EditFileInfo {
            file_path: "/storage/DCIM/Camera01/VID_20260116_152935_00_606.insv".into(),
            favorite_info: Some(pb::FavoriteInfo {
                playback_is_favorite: true,
                modify_timestamp: 1,
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    fs::write(r.join("DCIM/.editinfo_list.list"), ed.encode_to_vec()).unwrap();
    let t = Instant::now();
    let v = list(&r).unwrap();
    println!(
        "listed {} items in {:.1} ms",
        v.len(),
        t.elapsed().as_secs_f64() * 1e3
    );
    let f = |n: &str| v.iter().find(|i| i.uris[0].ends_with(n)).unwrap();
    assert_eq!(v.len(), 5, "{v:#?}");
    let a = f("_00_635.insv");
    assert_eq!(
        (a.kind, &a.uris[..], &*a.key),
        (
            album::Kind::Video,
            &[
                "/DCIM/Camera01/VID_20260117_141148_00_635.insv".to_string(),
                "/DCIM/Camera01/LRV_20260117_141148_01_635.lrv".into()
            ][..],
            "/DCIM/Camera01/VID_20260117_141148_635_"
        )
    );
    assert!(f("_606.insv").fav && !a.fav);
    let p = f("_659.jpg");
    assert_eq!(
        (p.kind, p.uris.len(), p.size, p.date),
        (album::Kind::Photo, 2, 1, 20260118124432)
    );
    assert_eq!(f("_700.jpg").key, "/DCIM/Camera02/IMG_20260119_101010_700_");
    // the proxy alone, its size (and with the samples its length and size) from its own trailer
    let lone = f("_708.lrv");
    let ln = fs::metadata(c.join("LRV_20260503_190639_01_708.lrv"))
        .unwrap()
        .len();
    assert_eq!(
        (lone.kind, lone.uris.len(), lone.size),
        (album::Kind::Video, 1, ln)
    );
    assert!(list(&d).is_err(), "no DCIM");
    let Some(s) = s else {
        eprintln!("X5_SAMPLES not set: no trailer metadata or thumbnails checked");
        return fs::remove_dir_all(&d).unwrap();
    };
    println!(
        "v635 from its trailer: {} ms, {} × {}, {} B, created {}",
        a.ms, a.dim[0], a.dim[1], a.size, a.date
    );
    assert_eq!(a.size, fs::metadata(s.join("v635.insv")).unwrap().len());
    assert!(a.ms > 1000 && a.dim[0] > 0 && a.dim[1] > 0);
    assert!(
        (20000101000000..21000101000000).contains(&a.date),
        "{}",
        a.date
    );
    println!(
        "lone v635.lrv from its trailer: {} ms, {} × {}, {} B",
        lone.ms, lone.dim[0], lone.dim[1], lone.size
    );
    assert!(lone.ms > 1000 && lone.dim[0] > 0 && lone.dim[1] > 0);
    assert!(lone.ms.abs_diff(a.ms) < 1000, "{} vs {}", lone.ms, a.ms);
    for u in [&a.uris[0], &a.uris[1], &f("_606.insv").uris[0]] {
        let t = Instant::now();
        let th = thumb(&r.join(&u[1..]));
        println!(
            "{}: thumbnail {:?} in {:.1} ms",
            album::name(u),
            th.as_ref().map(|t| (t.0, t.1)),
            t.elapsed().as_secs_f64() * 1e3
        );
        assert_eq!(th.map(|t| (t.0, t.1)), Some((320, 160)));
    }
    assert!(thumb(&c.join("IMG_20260118_124432_00_659.jpg")).is_none());
    fs::remove_dir_all(&d).unwrap();
}

/// The copy resumes from a partial file whose end matches the source, starts over otherwise, and keeps the source's
/// modification time; through the queue an import runs without a camera and waits while its card is gone.
#[test]
fn import() {
    let d = dir("import");
    let (card, lib) = (d.join("card/DCIM/Camera01"), d.join("lib"));
    fs::create_dir_all(&card).unwrap();
    let n = 24_000_000usize;
    let data: Vec<u8> = (0..n as u32).map(|i| (i ^ i >> 11) as u8).collect();
    let src = card.join("VID_20260117_141148_00_635.insv");
    fs::write(&src, &data).unwrap();
    let mt = UNIX_EPOCH + Duration::from_secs(1_768_000_000);
    File::options()
        .write(true)
        .open(&src)
        .unwrap()
        .set_modified(mt)
        .unwrap();
    let dst = d.join("lib/VID_20260117_141148_00_635.insv");
    let part = d.join("lib/VID_20260117_141148_00_635.insv.part");
    fs::create_dir_all(&lib).unwrap();
    // cancelled before it reads anything: where it would go on from
    let at = |p: &[u8]| {
        fs::write(&part, p).unwrap();
        let g = Prog::default();
        g.cancel.store(true, Relaxed);
        assert!(matches!(copy(&src, &dst, &g), Err(Error::Cancelled)));
        assert_eq!(g.total.load(Relaxed), n as u64);
        let k = g.done.load(Relaxed);
        assert_eq!(fs::metadata(&part).unwrap().len(), k);
        k
    };
    assert_eq!(at(&data[..10_000_000]), 10_000_000);
    let mut bad = data[..10_000_000].to_vec();
    bad[9_999_000] ^= 1;
    assert_eq!(at(&bad), 0, "the end differs");
    assert_eq!(
        at(&[&data[..], b"more"].concat()),
        0,
        "longer than the source"
    );
    assert_eq!(
        at(&data[..CHECK as usize / 2]),
        CHECK / 2,
        "shorter than the check"
    );
    fs::write(&part, &data[..5_000_000]).unwrap();
    let g = Prog::default();
    let t = Instant::now();
    assert_eq!(copy(&src, &dst, &g).unwrap(), n as u64);
    println!(
        "copied the last {} MB in {:.1} ms",
        (n - 5_000_000) / 1_000_000,
        t.elapsed().as_secs_f64() * 1e3
    );
    assert!(fs::read(&dst).unwrap() == data && !part.exists());
    assert_eq!(fs::metadata(&dst).unwrap().modified().unwrap(), mt);
    fs::remove_file(&dst).unwrap();

    let q: dl::Shared = Arc::new(Mutex::new(dl::Queue::load(None)));
    let gone = lib.join("IMG_20260118_124432_00_659.jpg");
    {
        let mut g = q.lock().unwrap();
        g.add(
            &format!(
                "file://{}",
                d.join("away/IMG_20260118_124432_00_659.jpg").display()
            ),
            gone.clone(),
        );
        g.add(&format!("file://{}", src.display()), dst.clone());
    }
    let mut cur = None;
    let end = Instant::now() + Duration::from_secs(20);
    let f = loop {
        if let Some(f) = dl::step(&q, None, &mut cur) {
            break f;
        }
        assert!(Instant::now() < end, "stuck");
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(f, dst);
    assert!(fs::read(&dst).unwrap() == data);
    for _ in 0..5 {
        assert!(dl::step(&q, None, &mut cur).is_none() && cur.is_none());
    }
    let st: Vec<St> = q.lock().unwrap().v.iter().map(|x| x.st).collect();
    assert_eq!(st, [St::Queued, St::Done]);
    assert!(!gone.exists());
    fs::remove_dir_all(&d).unwrap();
}

#[test]
fn escapes() {
    assert_eq!(
        unesc(r"/run/media/a/X5\040CARD\011\134"),
        "/run/media/a/X5 CARD\t\\"
    );
    assert_eq!(unesc(r"\04"), r"\04");
    assert!(
        media("VID_20260117_141148_00_635.insv") && media("PRO_LRV_20260117_141148_01_635.lrv")
    );
    assert!(!media("VID_20260117.insv") && !media("VID__1_2_3.insv") && !media("DSC_1_2_3.jpg"));
}
