//! Request encodings against PROTOCOL.md's hex examples, and the whole session against a simulated camera.

use super::*;
use crate::ucd::Pkt;
use std::{sync::atomic::Ordering::Relaxed, time::Instant};

fn hex(s: &str) -> Vec<u8> {
    s.split_whitespace()
        .map(|x| u8::from_str_radix(x, 16).unwrap())
        .collect()
}

/// A handle whose requests land in `rx` instead of an I/O thread.
fn fake() -> (Camera, Receiver<Msg>) {
    let (tx, rx) = mpsc::channel();
    (Camera(Arc::new(Inner(tx))), rx)
}

fn req(rx: &Receiver<Msg>) -> (u16, Vec<u8>) {
    match rx.recv_timeout(Duration::from_secs(1)).unwrap() {
        Msg::Req(c, b, _, _) => (c, b),
        _ => panic!("not a request"),
    }
}

/// Whole packet as sent for request id `id`.
fn wire(code: u16, id: u32, body: &[u8]) -> Vec<u8> {
    let mut v = vec![];
    ucd::msg(&mut v, code, id, true, body);
    v
}

#[test]
fn encodings() {
    let (c, rx) = fake();
    let _p = c.options(&[
        pb::OptionType::CameraType,
        pb::OptionType::SerialNumber,
        pb::OptionType::BatteryInfo,
        pb::OptionType::ConnectionStatus,
    ]);
    let (k, b) = req(&rx);
    assert_eq!(
        wire(k, 1, &b),
        hex("19 00 00 00 04 00 00 08 00 02 01 00 00 80 00 00 08 30 08 0f 08 0b 08 c6 01")
    );
    let _p = c.files(
        pb::MediaType::VideoAndPhoto,
        pb::CardLocation::StorClCameraSd,
    );
    let (k, b) = req(&rx);
    assert_eq!(
        wire(k, 7, &b),
        hex("16 00 00 00 04 00 00 0d 00 02 07 00 00 80 00 00 08 02 18 64 20 03")
    );
    let _p = c.file_infos(pb::CardLocation::StorClCameraSd);
    let (k, b) = req(&rx);
    assert_eq!(
        wire(k, 10, &b),
        hex("12 00 00 00 04 00 00 26 00 02 0a 00 00 80 00 00 08 03")
    );
    let _p = c.capture_status();
    let (k, b) = req(&rx);
    assert_eq!(
        wire(k, 2, &b),
        hex("10 00 00 00 04 00 00 0f 00 02 02 00 00 80 00 00")
    );
    let uri = "/DCIM/Camera01/VID_20250101_120000_00_001.insv";
    let _p = c.thumbnail(uri);
    let (k, b) = req(&rx);
    let mut w = hex("46 00 00 00 04 00 00 0b 00 02 0b 00 00 80 00 00 0a 2e");
    w.extend(uri.as_bytes());
    w.extend(hex("10 02 1a 02 12 00"));
    assert_eq!(wire(k, 11, &b), w);
    let _p = c.delete(vec![uri.into()]);
    let (k, b) = req(&rx);
    let mut w = hex("40 00 00 00 04 00 00 0c 00 02 0c 00 00 80 00 00 0a 2e");
    w.extend(uri.as_bytes());
    assert_eq!(wire(k, 12, &b), w);
    let _p = c.live_start();
    assert_eq!(req(&rx), (1, hex("10 01 30 28 38 2c 40 01 48 28 50 22")));
    let _p = c.start_capture(pb::CaptureMode::Normal);
    assert_eq!(req(&rx), (4, hex("08 01")));
    let _p = c.stop_capture(pb::CaptureMode::Normal);
    assert_eq!(req(&rx), (5, hex("10 01")));
    let _p = c.start_capture(pb::CaptureMode::PureVideo);
    assert_eq!(req(&rx), (4, hex("08 0b")));
    let _p = c.start_timelapse(pb::TimelapseMode::TimelapseIntervalShooting);
    assert_eq!(req(&rx), (22, hex("08 02")));
    let _p = c.stop_timelapse(pb::TimelapseMode::MobileTimelapseVideo);
    assert_eq!(req(&rx), (23, hex("08 01")));
    let _p = c.set_mode(Mode::Video(pb::VideoSubMode::VideoNormal));
    assert_eq!(req(&rx), (7, hex("08 29 12 00")));
    let _p = c.set_mode(Mode::Photo(pb::PhotoSubMode::PhotoSingle));
    assert_eq!(req(&rx), (7, hex("08 28 12 00")));
    let _p = c.format(pb::CardLocation::StorClCamera);
    assert_eq!(req(&rx), (24, vec![]));
    let _p = c.format(pb::CardLocation::StorClCameraSd);
    assert_eq!(req(&rx), (232, hex("08 03")));
    let _p = c.take_photo(pb::TakePicture::default());
    assert_eq!(req(&rx), (3, vec![]));
    // burst with PureShot: mode 3, raw_capture_type 3
    let _p = c.take_photo(pb::TakePicture {
        mode: pb::take_picture::Mode::Burst as i32,
        raw_capture_type: pb::RawCaptureType::Pureshot as i32,
        ..Default::default()
    });
    assert_eq!(req(&rx), (3, hex("08 03 20 03")));
    let _p = c.highlight(1500);
    let (k, b) = req(&rx);
    assert_eq!((k, &b[..4]), (60, &hex("08 dc 0b 1a")[..]));
    let _p = c.set_favorites(vec![pb::SetFavorite {
        favorite_flag: 1,
        material_url: uri.into(),
        modify_timestamp: 5,
        ..Default::default()
    }]);
    let (k, b) = req(&rx);
    assert_eq!(
        (
            k,
            pb::SetFavoriteList::decode(&b[..]).unwrap().set_favorite[0].favorite_flag
        ),
        (176, 1)
    );
    // Time: 2025-01-01 12:00:00 UTC, +3600 s, Europe/Vienna; two requests (§5.3), both out before any answer (this
    // handle never answers) and ahead of the next call.
    let _p = c.set_time(1735732800, 3600, "Europe/Vienna");
    let _q = c.capture_status();
    let (k, b) = req(&rx);
    let w = hex(
        "32 00 00 00 04 00 00 07 00 02 04 00 00 80 00 00 08 0c 08 0d 08 af 01 12 19 60 c0 dc d4 bb 06 68 90 1c fa 0a 0d \
         45 75 72 6f 70 65 2f 56 69 65 6e 6e 61",
    );
    assert_eq!(wire(k, 4, &b), w);
    assert_eq!(
        req(&rx),
        (
            7,
            hex("08 af 01 12 10 fa 0a 0d 45 75 72 6f 70 65 2f 56 69 65 6e 6e 61")
        )
    );
    assert_eq!(req(&rx).0, code::CAPTURE_STATUS);
    let q = pb::CheckAuthorization {
        id: "01234567-89ab-cdef-0123-456789abcdef".into(),
        initiator_type: 2,
        ..Default::default()
    };
    let w = hex(
        "38 00 00 00 04 00 00 27 00 02 04 00 00 80 00 00 0a 24 30 31 32 33 34 35 36 37 2d 38 39 61 62 2d 63 64 65 66 \
         2d 30 31 32 33 2d 34 35 36 37 38 39 61 62 63 64 65 66 10 02",
    );
    assert_eq!(wire(39, 4, &q.encode_to_vec()), w);
}

#[test]
fn ident_firmware() {
    let o = |a: &str, b: &str| {
        let v = pb::Options {
            ota_pkg_version: a.into(),
            box_ota_pkg_version: b.into(),
            firmware_revision: "r".into(),
            ..Default::default()
        };
        ident(
            pb::GetOptionsResp {
                value: Some(v),
                ..Default::default()
            }
            .encode_to_vec(),
        )
        .unwrap()
        .firmware
    };
    assert_eq!(
        (o("v1.13.21", "v1.9.30"), o("", "v2"), o("", "")),
        ("v1.9.30".into(), "v2".into(), "r".into())
    );
}

/// Simulated camera behind a [`Link`]: answers sync, a few requests and HTTP like the real X5 (range start
/// honoured, body to EOF, tunnel connections never closed by the camera, stale tunnel data of an earlier session).
/// `direct` = Wi-Fi style HTTP sockets.
struct Cam {
    tx: Sender<Msg>,
    big: Arc<Vec<u8>>,
    direct: bool,
    log: Arc<Mutex<Vec<String>>>,
}

fn reply(v: &mut Vec<u8>, status: u16, id: u32, end: bool, body: &[u8]) {
    ucd::msg(v, status, id, end, body)
}

fn note(v: &mut Vec<u8>, code: u16, body: &[u8]) {
    let mut h = [0; 9];
    h[..2].copy_from_slice(&code.to_le_bytes());
    h[2] = 2;
    h[3..7].copy_from_slice(&(3u32 << 30 | 77).to_le_bytes());
    ucd::put(v, 4, &h, body)
}

impl Cam {
    fn body(&self, uri: &str) -> Vec<u8> {
        if uri == "/CONF" {
            b"{\"Sphere06C\":{}}".to_vec()
        } else {
            self.big.to_vec()
        }
    }

    fn http(&self, req: &[u8]) -> Vec<Vec<u8>> {
        let q = std::str::from_utf8(req).unwrap();
        let uri = q.split(' ').nth(1).unwrap();
        let a: usize = q
            .split("bytes=")
            .nth(1)
            .unwrap()
            .split('-')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let f = self.body(uri);
        let h = format!(
            "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {a}-{}/{}\r\nContent-Length: {}\r\n\
             Etag: \"5f00.{}\"\r\nConnection: keep-alive\r\nConnection: close\r\n\r\n",
            f.len() - 1,
            f.len(),
            f.len() - a,
            f.len()
        );
        let mut o = vec![h.into_bytes()];
        // "/stall": one chunk, then nothing; the connection stays open.
        let n = if uri == "/stall" { 1 } else { usize::MAX };
        o.extend(f[a..].chunks(200_000).take(n).map(|c| c.to_vec()));
        o
    }
}

impl Link for Cam {
    fn send(&mut self, p: &[u8]) -> std::io::Result<()> {
        let mut v = vec![];
        if p == ucd::SYNC {
            // Queued heartbeats before the reply, then a LinuxCmd; split across reads.
            for _ in 0..5 {
                v.extend(ucd::HB)
            }
            v.extend(ucd::SYNC);
            v.extend(hex("0a 00 00 00 09 00 00 01 02 01"));
            let (a, b) = v.split_at(20);
            self.tx.send(Msg::Rx(a.to_vec())).unwrap();
            self.tx.send(Msg::Rx(b.to_vec())).unwrap();
            return Ok(());
        }
        let Some((Pkt::Msg { code, id, body, .. }, _)) = ucd::next(p).unwrap() else {
            return Ok(());
        };
        self.log.lock().unwrap().push(format!("req {code}"));
        match code {
            8 => {
                let o = pb::Options {
                    camera_type: "Insta360 X5".into(),
                    serial_number: "SERIAL".into(),
                    firmware_revision: "v1.13.21".into(),
                    ..Default::default()
                };
                reply(
                    &mut v,
                    200,
                    id,
                    true,
                    &pb::GetOptionsResp {
                        option_types: vec![48, 15, 30],
                        value: Some(o),
                    }
                    .encode_to_vec(),
                )
            }
            13 => {
                let q = pb::GetFileList::decode(body).unwrap();
                let uri = (q.start..150.min(q.start + q.limit))
                    .map(|i| format!("/DCIM/Camera01/IMG_{i:03}.jpg"))
                    .collect();
                reply(
                    &mut v,
                    200,
                    id,
                    true,
                    &pb::GetFileListResp {
                        uri,
                        total_count: 150,
                        ..Default::default()
                    }
                    .encode_to_vec(),
                )
            }
            38 => {
                // Fragmented: the method/status of the last fragment counts.
                let fi = (0..300)
                    .map(|i| pb::FileInfo {
                        file_path: format!("/f{i}"),
                        metadata: vec![],
                    })
                    .collect();
                let b = pb::FileInfoList {
                    file_info: fi,
                    card_location: 0,
                }
                .encode_to_vec();
                let (x, y) = b.split_at(b.len() / 2);
                note(
                    &mut v,
                    8195,
                    &pb::NotificationBatteryUpdate {
                        battery_status: Some(pb::BatteryInfo {
                            battery_level: 80,
                            battery_scale: 100,
                            ..Default::default()
                        }),
                    }
                    .encode_to_vec(),
                );
                reply(&mut v, 0, id, false, x);
                reply(&mut v, 200, id, true, y);
            }
            1 => {
                reply(&mut v, 200, id, true, &[]);
                note(
                    &mut v,
                    8234,
                    &pb::LiveStreamParamsUpdate {
                        res_width: 2656,
                        res_height: 1328,
                        res_fps: 30,
                        ..Default::default()
                    }
                    .encode_to_vec(),
                );
                for i in 0..3u64 {
                    let mut h = [0; 9];
                    h[0] = stream::VIDEO;
                    h[1..].copy_from_slice(&(1000 + 33 * i).to_le_bytes());
                    ucd::put(&mut v, 1, &h, &[0, 0, 0, 1, 0x65, i as u8]);
                }
            }
            83 => reply(
                &mut v,
                200,
                id,
                true,
                &pb::GetFileResp {
                    uri: "/CONF".into(),
                    version: 71,
                }
                .encode_to_vec(),
            ),
            99 => reply(&mut v, 500, id, true, &hex("08 03")),
            98 => {} // never answered
            _ => reply(&mut v, 200, id, true, &[]),
        }
        self.tx.send(Msg::Rx(v)).unwrap();
        Ok(())
    }

    fn http_open(&mut self, c: u32, req: &[u8]) -> std::io::Result<()> {
        self.log.lock().unwrap().push(format!("open {c}"));
        // What a crashed predecessor's connections 0..4 still deliver after this session synced (the X5 sent an
        // earlier request's head like this one).
        for k in (0..4).filter(|_| !self.direct) {
            let mut v = vec![];
            let h = b"HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 9-9/10\r\n\r\nx";
            ucd::tunnel(&mut v, k, 0x0c, 0, h);
            self.tx.send(Msg::Rx(v)).unwrap();
        }
        for d in self.http(req) {
            let m = if self.direct {
                Msg::Http(c, d)
            } else {
                let mut v = vec![];
                ucd::tunnel(&mut v, c, 0x04, 0, &d);
                Msg::Rx(v)
            };
            self.tx.send(m).unwrap();
        }
        if self.direct && !req.starts_with(b"GET /stall ") {
            self.tx.send(Msg::Http(c, vec![])).unwrap()
        }
        Ok(())
    }

    fn http_ack(&mut self, _: u32, _: u64) -> std::io::Result<()> {
        Ok(())
    }

    fn http_close(&mut self, c: u32) -> std::io::Result<()> {
        self.log.lock().unwrap().push(format!("close {c}"));
        Ok(())
    }

    fn busy(&self) -> bool {
        false
    }

    fn host(&self) -> &str {
        "127.0.0.1"
    }

    fn stop(&mut self) {
        self.log.lock().unwrap().push("stop".into())
    }
}

fn session(direct: bool) {
    let big: Arc<Vec<u8>> = Arc::new((0..5_000_001u32).map(|i| (i ^ i >> 9) as u8).collect());
    let log = Arc::new(Mutex::new(vec![]));
    let (b, l) = (big.clone(), log.clone());
    let (c, ev) = start(
        move |tx, _| {
            Ok(Cam {
                tx,
                big: b,
                direct,
                log: l,
            })
        },
        None,
    )
    .wait()
    .unwrap();

    let id = c.identify().wait().unwrap();
    assert_eq!(
        (&*id.model, &*id.serial, &*id.firmware),
        ("Insta360 X5", "SERIAL", "v1.13.21")
    );
    let f = c
        .files(pb::MediaType::VideoAndPhoto, pb::CardLocation::StorClCamera)
        .wait()
        .unwrap();
    assert_eq!((f.len(), &*f[149]), (150, "/DCIM/Camera01/IMG_149.jpg"));
    let fi = c.file_infos(pb::CardLocation::StorClCamera).wait().unwrap();
    assert_eq!(
        (fi.len(), &*fi[299].0, fi[0].1.is_none()),
        (300, "/f299", true)
    );
    let Event::Note(Note::Battery(b)) = ev.recv().unwrap() else {
        panic!()
    };
    assert_eq!(b.battery_level, 80);
    let Err(Error::Status(500, Some(e))) = c.raw(99, vec![], T).wait() else {
        panic!()
    };
    assert_eq!(e.code(), pb::error::ErrorCode::ExecuteError);
    assert!(matches!(
        c.raw(98, vec![], Duration::from_millis(300)).wait(),
        Err(Error::Timeout)
    ));

    c.live_start().wait().unwrap();
    let Event::Note(Note::LiveParams(p)) = ev.recv().unwrap() else {
        panic!()
    };
    assert_eq!((p.res_width, p.res_height), (2656, 1328));
    for i in 0..3 {
        let Event::Frame(f) = ev.recv().unwrap() else {
            panic!()
        };
        assert_eq!(
            (f.kind, f.ts, f.data[5]),
            (stream::VIDEO, 1000 + 33 * i, i as u8)
        );
    }

    assert_eq!(
        c.conf().wait().unwrap(),
        (71, b"{\"Sphere06C\":{}}".to_vec())
    );
    assert!(c.get("/x").wait().unwrap() == *big);

    let dir = std::env::temp_dir().join(format!("x5link-s{}-{direct}", std::process::id()));
    fs_create(&dir);
    let p = dir.join("v.insv");
    let d = c.download("/DCIM/Camera01/v.insv", &p, 4);
    assert_eq!(d.done.wait().unwrap(), big.len() as u64);
    assert_eq!(d.prog.done.load(Relaxed), big.len() as u64);
    assert!(std::fs::read(&p).unwrap() == *big);

    // Closing during a download that stalls on 4 connections, with the preview on.
    let s = dir.join("s.insv");
    let d = c.download("/stall", &s, 4);
    while d.prog.done.load(Relaxed) < 4 * 200_000 {
        thread::sleep(Duration::from_millis(1))
    }
    c.close().wait().unwrap();
    assert!(matches!(d.done.wait(), Err(Error::Closed)));
    let prog = Arc::new(Prog::default());
    http::Xfer::new("/stall".into(), Some(&s), 1, prog.clone()).unwrap();
    assert_eq!(prog.done.load(Relaxed), 4 * 200_000);
    std::fs::remove_dir_all(&dir).unwrap();
    let log = log.lock().unwrap();
    let opens = log.iter().filter(|l| l.starts_with("open")).count();
    let closes = log.iter().filter(|l| l.starts_with("close")).count();
    // conf + get: one connection each; each download splits into 4.
    assert_eq!((opens, closes), (10, 10), "{log:?}");
    // The stalled connections close first, then the preview stops, and its answer is read before the stop.
    let ids: Vec<_> = log.iter().filter_map(|l| l.strip_prefix("open ")).collect();
    let want: Vec<String> = ids[6..].iter().map(|c| format!("close {c}")).collect();
    let n = log.len();
    assert_eq!(&log[n - 6..n - 2], want);
    assert_eq!(&log[n - 2..], ["req 2", "stop"]);
    assert!(c.identify().wait().is_err());
}

fn fs_create(d: &Path) {
    std::fs::create_dir_all(d).unwrap()
}

#[test]
fn session_usb_tunnel() {
    session(false)
}

#[test]
fn session_direct_http() {
    session(true)
}

/// Wi-Fi transport on real sockets against a local camera: UCD on one port (sync after queued heartbeats;
/// CHECK_AUTHORIZATION, refused unless CHECK_TYPE came first, answers UNAUTHORIZED, then 8209s: a REJECT for
/// another operation type, PENDING, then empty = user confirmed), HTTP on another (range start honoured, body to
/// EOF, then the server closes, as with `Connection: close`).
#[test]
fn wifi_sockets() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let big: Arc<Vec<u8>> = Arc::new(
        (0..3_000_017u32)
            .map(|i| ((i * 31) ^ (i >> 7)) as u8)
            .collect(),
    );
    let (ctl, web) = (
        TcpListener::bind("127.0.0.1:0").unwrap(),
        TcpListener::bind("127.0.0.1:0").unwrap(),
    );
    let (ca, wa) = (ctl.local_addr().unwrap(), web.local_addr().unwrap());
    let codes = Arc::new(Mutex::new(vec![]));
    let cs = codes.clone();
    thread::spawn(move || {
        let (mut s, _) = ctl.accept().unwrap();
        let (mut b, mut buf) = (vec![0; 1 << 16], vec![]);
        let mut synced = false;
        loop {
            let n = s.read(&mut b).unwrap();
            if n == 0 {
                return;
            }
            buf.extend_from_slice(&b[..n]);
            let mut v = vec![];
            if !synced {
                let Some(i) = ucd::mark(&buf) else { continue };
                buf.drain(..i);
                synced = true;
                v.extend(ucd::HB.repeat(3));
                v.extend(ucd::SYNC);
            }
            while let Ok(Some((p, n))) = ucd::next(&buf) {
                if let Pkt::Msg { code, id, .. } = p {
                    let mut cs = cs.lock().unwrap();
                    cs.push(code);
                    match code {
                        39 => {
                            let r = pb::CheckAuthorizationResp {
                                authorization_status: if cs[0] == 8 { 1 } else { 2 },
                                ..Default::default()
                            };
                            reply(&mut v, 200, id, true, &r.encode_to_vec());
                            for (r, o) in [(1, 1), (4, 0)] {
                                let n = pb::NotificationAuthorizationResult {
                                    authorization_result: r,
                                    authorization_operation_type: o,
                                };
                                note(&mut v, 8209, &n.encode_to_vec());
                            }
                            note(&mut v, 8209, &[]);
                        }
                        8 => {
                            let o = pb::Options {
                                camera_type: "Insta360 X5".into(),
                                ..Default::default()
                            };
                            reply(
                                &mut v,
                                200,
                                id,
                                true,
                                &pb::GetOptionsResp {
                                    option_types: vec![48],
                                    value: Some(o),
                                }
                                .encode_to_vec(),
                            )
                        }
                        _ => reply(&mut v, 200, id, true, &[]),
                    }
                }
                buf.drain(..n);
            }
            s.write_all(&v).unwrap();
        }
    });
    let f = big.clone();
    thread::spawn(move || {
        for s in web.incoming() {
            let (mut s, f) = (s.unwrap(), f.clone());
            thread::spawn(move || {
                let (mut b, mut q) = (vec![0; 4096], vec![]);
                while !q.windows(4).any(|w| w == b"\r\n\r\n") {
                    let n = s.read(&mut b).unwrap();
                    q.extend_from_slice(&b[..n]);
                }
                let q = String::from_utf8(q).unwrap();
                let a: usize = q
                    .split("bytes=")
                    .nth(1)
                    .unwrap()
                    .split('-')
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                let h = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{}/{}\r\nContent-Length: {}\r\nEtag: \"1.{}\"\r\n\r\n",
                    f.len() - 1,
                    f.len(),
                    f.len() - a,
                    f.len()
                );
                let _ = s.write_all(h.as_bytes()).and_then(|_| s.write_all(&f[a..]));
            });
        }
    });
    let (c, ev) = start(
        move |tx, pool| tcp::Tcp::open(ca, wa, tx, pool),
        Some("uuid".into()),
    )
    .wait()
    .unwrap();
    // The 8209s are also delivered to the app.
    for _ in 0..3 {
        assert!(matches!(ev.recv().unwrap(), Event::Note(Note::Auth(_))));
    }
    assert_eq!(c.identify().wait().unwrap().model, "Insta360 X5");
    let dir = std::env::temp_dir().join(format!("x5link-w{}", std::process::id()));
    fs_create(&dir);
    let p = dir.join("w.insv");
    let d = c.download("/DCIM/Camera01/w.insv", &p, 4);
    assert_eq!(d.done.wait().unwrap(), big.len() as u64);
    assert!(std::fs::read(&p).unwrap() == *big);
    std::fs::remove_dir_all(&dir).unwrap();
    c.close().wait().unwrap();
    // CHECK_TYPE, CHECK_AUTHORIZATION, identify, and the capture status read before closing.
    assert_eq!(*codes.lock().unwrap(), [8, 39, 8, 15]);
}

#[test]
fn imu() {
    let mut data = vec![];
    for (t, v) in [(7i64, [-1.0, 0.5, 0.25, 0.1, 0.2, 0.3]), (9, [0.0; 6])] {
        data.extend(t.to_le_bytes());
        v.iter().for_each(|x: &f64| data.extend(x.to_le_bytes()));
    }
    let f = Frame {
        kind: stream::GYRO,
        ts: 7,
        data,
        at: Instant::now(),
    };
    let s: Vec<Imu> = f.imu().collect();
    assert_eq!(
        s[0],
        Imu {
            ts: 7,
            acc: [-1.0, 0.5, 0.25],
            gyro: [0.1, 0.2, 0.3]
        }
    );
    assert_eq!((s.len(), s[1].ts), (2, 9));
}

#[test]
fn thumbnail_nv12() {
    let mut b = vec![0; 40 + 8 * 4 * 3 / 2];
    let n = b.len() as u32;
    for (o, v) in [(0, 1), (4, n), (8, 1), (16, 8), (20, 4)] {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes())
    }
    assert_eq!(nv12(&b).map(|t| (t.0, t.1, t.2.len())), Some((8, 4, 48)));
    assert!(nv12(&b[..60]).is_none());
    let mut none = |w: u32, h: u32| {
        b[16..20].copy_from_slice(&w.to_le_bytes());
        b[20..24].copy_from_slice(&h.to_le_bytes());
        nv12(&b).is_none()
    };
    // each of these has the length's 48 bytes in u32 (268435458 × 16 × 3 / 2 wraps), odd ones no whole chroma blocks
    assert!(none(268435458, 16) && none(1, 32) && none(32, 1) && !none(8, 4));
}

/// The hotplug watch opens (netlink uevents need no privileges) and reports the bus state at once; with
/// `X5_PLUG=none|disk|other|app` (what is plugged in now: `disk` for the X5 in U-disk mode) it is that.
#[test]
fn watch_reports_now() {
    let (tx, rx) = mpsc::channel();
    watch(move |p| {
        let _ = tx.send(p);
        false
    })
    .unwrap();
    let p = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(p, scan());
    println!("bus: {p:?}");
    if let Ok(w) = std::env::var("X5_PLUG") {
        assert_eq!(format!("{p:?}").to_lowercase(), w);
    }
}

/// A transport that never answers: the camera enumerated but asleep. Its `stop` says when the I/O thread let go.
struct Mute(Arc<Mutex<Option<Instant>>>);

impl Link for Mute {
    fn send(&mut self, _: &[u8]) -> std::io::Result<()> {
        Ok(())
    }
    fn http_open(&mut self, _: u32, _: &[u8]) -> std::io::Result<()> {
        Ok(())
    }
    fn http_ack(&mut self, _: u32, _: u64) -> std::io::Result<()> {
        Ok(())
    }
    fn http_close(&mut self, _: u32) -> std::io::Result<()> {
        Ok(())
    }
    fn busy(&self) -> bool {
        false
    }
    fn host(&self) -> &str {
        ""
    }
    fn stop(&mut self) {
        *self.0.lock().unwrap() = Some(Instant::now());
    }
}

/// Dropping an [`Opening`] whose camera stays silent ends the sync at once and releases the transport, instead of
/// after the 30 s sync deadline; a ready one hands out its session once.
#[test]
fn opening_cancel() {
    let stop = Arc::new(Mutex::new(None));
    let s = stop.clone();
    let mut o = start(move |_, _| Ok(Mute(s)), None);
    thread::sleep(Duration::from_millis(300));
    assert!(o.poll().is_none(), "still syncing");
    let t = Instant::now();
    drop(o);
    let at = loop {
        if let Some(at) = *stop.lock().unwrap() {
            break at;
        }
        assert!(
            t.elapsed() < Duration::from_secs(2),
            "the I/O thread kept the transport"
        );
        thread::sleep(Duration::from_millis(1));
    };
    println!(
        "cancelled sync released the transport after {:.1} ms",
        (at - t).as_secs_f64() * 1e3
    );
    let mut o: Opening = Err(Error::Silent(0)).into();
    assert!(matches!(o.poll(), Some(Err(Error::Silent(0)))));
    let (c, _) = fake();
    let mut o: Opening = Ok((c, mpsc::channel().1)).into();
    assert!(matches!(o.poll(), Some(Ok(_))));
    assert!(matches!(o.poll(), Some(Err(Error::Closed))));
}

/// Stream packets a simulated camera sends arrive as `Event::Frame`, in order, with kind, stamp and payload; and
/// notification 8214 decodes as `Note::Temp`.
#[cfg(feature = "sim")]
#[test]
fn sim_frames() {
    let (c, ev, s) = Camera::sim(|_, _| (200, vec![]), |_| None, 1).unwrap();
    let big: Vec<u8> = (0..600_000u32).map(|i| i as u8).collect();
    s.frame(stream::VIDEO, 0, &big);
    s.frame(stream::GYRO, 278164, &[1; 112]);
    let t = pb::TempState {
        temp_state: pb::TemperatureState::TempHigh as i32,
        temp_value: 61,
        ..Default::default()
    };
    s.note(8214, &t.encode_to_vec());
    let mut got = vec![];
    while got.len() < 3 {
        match ev.recv_timeout(Duration::from_secs(2)).unwrap() {
            Event::Frame(f) => got.push((f.kind, f.ts, f.data.len(), f.data == big)),
            Event::Note(Note::Temp(x)) => {
                assert_eq!(x, t);
                got.push((0, 0, 0, false))
            }
            _ => {}
        }
    }
    assert_eq!(
        got,
        [
            (stream::VIDEO, 0, 600_000, true),
            (stream::GYRO, 278164, 112, false),
            (0, 0, 0, false)
        ]
    );
    c.close().wait().unwrap();
}
