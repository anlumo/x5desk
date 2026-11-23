//! Read-only checks against a real X5 in USB app mode.
//!
//! x5 info                     identity, options, capture status, storage
//! x5 ls                       all files (MEDIA_VIDEO_PHOTO, MEDIA_RAW) and the file info list
//! x5 conf <out>               the capture-parameter JSON
//! x5 favs                     favourite flags (GET_EDITINFO_LIST): path, flag, modified
//! x5 raw <code> [hex]         a request from `RO`, raw
//! x5 thumb <uri> <out>        trailer thumbnail
//! x5 burst <n>                thumbnails of the first n files, one by one, then all n in flight at once
//! x5 dl <uri> <out> <conns> [cancel_ms]   download with timing; cancel after cancel_ms (resume: run again)
//! x5 par - <uri>...          parallel whole-file GETs into memory, aggregate MB/s
//! x5 live <secs> <prefix>     preview: <prefix>.h264, <prefix>.timestamps.txt ("ts_ms bytes" per frame), .30 / .40 raw
//! x5 notes <secs>             print notifications
use link::{Camera, Event, Note, pb, stream};
use std::{
    fs::File,
    io::Write,
    sync::atomic::Ordering::Relaxed,
    time::{Duration, Instant},
};

/// What `raw` may send: requests that change nothing on the camera.
const RO: [u16; 8] = {
    use link::code::*;
    [
        GET_OPTIONS,
        GET_PHOTO_OPTIONS,
        GET_FILE_EXTRA,
        GET_FILE_LIST,
        CAPTURE_STATUS,
        GET_FILEINFO_LIST,
        GET_FILE,
        GET_EDITINFO_LIST,
    ]
};

fn main() -> link::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().is_some_and(|s| s == "raw") && !RO.contains(&a[1].parse().unwrap()) {
        return Err(link::Error::Http(format!("{} is not in {RO:?}", a[1])));
    }
    let t0 = Instant::now();
    let (c, ev) = Camera::usb().wait()?;
    eprintln!("synced in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    let drain = |ev: &std::sync::mpsc::Receiver<Event>| {
        for e in ev.try_iter() {
            match e {
                Event::Note(n) => eprintln!("note {n:?}"),
                Event::Frame(f) => eprintln!("frame {:#x} ts {} {} B", f.kind, f.ts, f.data.len()),
                Event::Closed(e) => eprintln!("closed: {e}"),
            }
        }
    };
    // Every command ends in a clean close, also on an error.
    let r = (|| -> link::Result<()> {
        match a.first().map(|s| s.as_str()).unwrap_or("info") {
            "info" => {
                let t = Instant::now();
                let id = c.identify().wait()?;
                println!(
                    "identify ({:.0} ms): {id:?}",
                    t.elapsed().as_secs_f64() * 1e3
                );
                use pb::Setting::*;
                let o = c
                    .options(&[Storage, Storages, Stitching, PhotoMode, VideoMode])
                    .wait()?;
                println!(
                    "storage {:?}\nsplicing {} photo_mode {:?} video_mode {:?}",
                    o.storage,
                    o.stitching,
                    o.photo_mode(),
                    o.video_mode()
                );
                println!("capture status {:?}", c.capture_status().wait()?);
            }
            "ls" => {
                let t = Instant::now();
                let f = c.files(pb::Media::VideoPhoto, pb::Volume::Camera).wait()?;
                let d = c.files(pb::Media::Raw, pb::Volume::Camera).wait()?;
                println!(
                    "{} files + {} DNG in {:.0} ms",
                    f.len(),
                    d.len(),
                    t.elapsed().as_secs_f64() * 1e3
                );
                for u in f.iter().chain(&d) {
                    println!("{u}")
                }
                let t = Instant::now();
                let fi = c.file_infos(pb::Volume::Camera).wait()?;
                let n = fi.iter().filter(|f| f.1.is_some()).count();
                println!(
                    "file info: {} entries, {n} with metadata, {:.0} ms",
                    fi.len(),
                    t.elapsed().as_secs_f64() * 1e3
                );
                if let Some((p, Some(m))) = fi.iter().find(|f| f.1.is_some()) {
                    println!(
                        "{p}: size {} creation {} dims {:?} secs {} category {}",
                        m.size, m.created, m.dims, m.secs, m.category
                    );
                }
            }
            "raw" => {
                let k = a[1].parse().unwrap();
                let h = a.get(2).map_or(&b""[..], |h| h.as_bytes());
                let b = (0..h.len() / 2)
                    .map(|i| {
                        u8::from_str_radix(std::str::from_utf8(&h[2 * i..2 * i + 2]).unwrap(), 16)
                            .unwrap()
                    })
                    .collect();
                match c.raw(k, b, Duration::from_secs(10)).wait() {
                    Ok(b) => println!("200, {} B: {:02x?}", b.len(), &b[..b.len().min(64)]),
                    Err(e) => println!("{e}"),
                }
            }
            "favs" => {
                let l = c.favorites().wait()?;
                let f = l
                    .marks
                    .iter()
                    .filter(|e| e.favorite.as_ref().is_some_and(|f| f.on));
                println!(
                    "edit info: {} files, {} favourites",
                    l.marks.len(),
                    f.count()
                );
                for e in &l.marks {
                    let f = e.favorite.unwrap_or_default();
                    println!("{} {} {}", e.path, f.on, f.modified);
                }
            }
            "conf" => {
                let (v, j) = c.conf().wait()?;
                File::create(&a[1])?.write_all(&j)?;
                println!("/CONF version {v}, {} B", j.len());
            }
            "thumb" => {
                let b = c.thumbnail(&a[1]).wait()?;
                File::create(&a[2])?.write_all(&b)?;
                println!(
                    "thumbnail {} B, NV12 {:?}",
                    b.len(),
                    link::nv12(&b).map(|t| (t.0, t.1))
                );
            }
            "burst" => {
                let f = c.files(pb::Media::VideoPhoto, pb::Volume::Camera).wait()?;
                let f = &f[..a[1].parse::<usize>().unwrap().min(f.len())];
                let t = Instant::now();
                let one: Vec<_> = f
                    .iter()
                    .map(|u| c.thumbnail(u).wait())
                    .collect::<link::Result<_>>()?;
                let t1 = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let ps: Vec<_> = f.iter().map(|u| c.thumbnail(u)).collect();
                let all: Vec<_> = ps
                    .into_iter()
                    .map(|p| p.wait())
                    .collect::<link::Result<_>>()?;
                println!(
                    "{} thumbnails, {} B: one by one {t1:.0} ms, all in flight {:.0} ms, identical {}",
                    f.len(),
                    one.iter().map(|b| b.len()).sum::<usize>(),
                    t.elapsed().as_secs_f64() * 1e3,
                    one == all
                );
            }
            "dl" => {
                let t = Instant::now();
                let resume = std::path::Path::new(&format!("{}.part.todo", a[2])).exists();
                let d = c.download(&a[1], a[2].as_ref(), a[3].parse().unwrap());
                // A resumed transfer knows its total from the sidecar once the I/O thread has set it up.
                while resume && d.prog.total.load(Relaxed) == 0 {
                    std::thread::sleep(Duration::from_millis(1))
                }
                let p0 = d.prog.done.load(Relaxed);
                if let Some(ms) = a.get(4) {
                    std::thread::sleep(Duration::from_millis(ms.parse().unwrap()));
                    d.prog.cancel.store(true, Relaxed);
                }
                let r = d.done.wait();
                let s = t.elapsed().as_secs_f64();
                let n = d.prog.done.load(Relaxed) - p0;
                println!(
                    "{r:?}: {n} B new (resumed at {p0}) in {s:.2} s = {:.1} MB/s",
                    n as f64 / s / 1e6
                );
            }
            "par" => {
                // n parallel whole-file GETs into memory (like re/scripts/x5tunbench.py)
                let t = Instant::now();
                let ps: Vec<_> = a[2..].iter().map(|u| c.get(u)).collect();
                let n: usize = ps
                    .into_iter()
                    .map(|p| p.wait().map(|b| b.len()))
                    .sum::<link::Result<usize>>()?;
                let s = t.elapsed().as_secs_f64();
                println!(
                    "{} GETs, {n} B in {s:.2} s = {:.1} MB/s",
                    a.len() - 2,
                    n as f64 / s / 1e6
                );
            }
            "live" => {
                let secs: f64 = a[1].parse().unwrap();
                let mut h = File::create(format!("{}.h264", a[2]))?;
                let mut ts = File::create(format!("{}.timestamps.txt", a[2]))?;
                let mut g = File::create(format!("{}.30", a[2]))?;
                let mut x = File::create(format!("{}.40", a[2]))?;
                c.live_start().wait()?;
                let end = Instant::now() + Duration::from_secs_f64(secs);
                let mut n = [0usize; 3];
                let mut imu = (0, f64::MAX, 0f64, 0f64);
                while let Ok(e) = ev.recv_timeout(end.saturating_duration_since(Instant::now())) {
                    match e {
                        Event::Frame(f) => {
                            let (k, o): (usize, &mut File) = match f.kind {
                                stream::VIDEO => {
                                    writeln!(ts, "{} {}", f.ts, f.data.len())?;
                                    (0, &mut h)
                                }
                                stream::GYRO => {
                                    for s in f.imu() {
                                        let m = s.acc.iter().map(|a| a * a).sum::<f64>().sqrt();
                                        (imu.0, imu.1, imu.2) =
                                            (imu.0 + 1, imu.1.min(m), imu.2.max(m));
                                        imu.3 = imu
                                            .3
                                            .max(s.gyro.iter().fold(0f64, |a, g| a.max(g.abs())));
                                    }
                                    (1, &mut g)
                                }
                                _ => (2, &mut x),
                            };
                            if n[k] == 0 {
                                eprintln!(
                                    "first {:#x} ts {} {} B: {:02x?}",
                                    f.kind,
                                    f.ts,
                                    f.data.len(),
                                    &f.data[..f.data.len().min(48)]
                                );
                            }
                            n[k] += 1;
                            o.write_all(&f.data)?;
                        }
                        Event::Note(n) => eprintln!("note {n:?}"),
                        Event::Closed(e) => return Err(e),
                    }
                }
                c.live_stop().wait()?;
                // Packets come out in wire order, so whatever is queued now came before the answer.
                let fs = |ev: &std::sync::mpsc::Receiver<Event>| {
                    ev.try_iter()
                        .filter(|e| matches!(e, Event::Frame(_)))
                        .count()
                };
                let early = fs(&ev);
                println!(
                    "video {} gyro {} exposure {} packets in {secs} s",
                    n[0], n[1], n[2]
                );
                println!(
                    "imu: {} samples, |acc| {:.3}..{:.3}, max |gyro| {:.4}",
                    imu.0, imu.1, imu.2, imu.3
                );
                std::thread::sleep(Duration::from_millis(300));
                println!(
                    "stream packets: {early} sent before the STOP_LIVE_STREAM answer, not saved; {} after it",
                    fs(&ev)
                );
            }
            "notes" => {
                let end = Instant::now() + Duration::from_secs(a[1].parse().unwrap());
                while let Ok(e) = ev.recv_timeout(end.saturating_duration_since(Instant::now())) {
                    if let Event::Note(n) = e {
                        match n {
                            Note::Other(c, b) => {
                                println!("note {c} {} B {:02x?}", b.len(), &b[..b.len().min(32)])
                            }
                            n => println!("note {n:?}"),
                        }
                    }
                }
            }
            x => eprintln!("unknown command {x}"),
        }
        Ok(())
    })();
    drain(&ev);
    let t = Instant::now();
    c.close().wait()?;
    eprintln!("closed in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    r
}
