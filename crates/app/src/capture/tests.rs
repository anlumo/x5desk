//! The capture path without the camera: link's simulated camera (`replay`) streams the recorded preview (`X5_LIVE`)
//! at its timestamps through the camera worker, the capture worker, the decode thread and the stitch, with the v635
//! calibration (`X5_SAMPLES`) and the camera's `/CONF` (`X5_CONF`, optional); and the capture commands as they reach
//! the (simulated) camera. On the real X5 (`X5_CAMERA*`): the preview, and the state-changing commands the user
//! allowed, each restored.

use super::*;
use crate::{
    album::{Cache, Item},
    camera::{Cmd as Cc, Up, Worker},
    dl,
    replay::{self, Replay},
};
use prost::Message;
use std::{collections::HashSet, path::Path, sync::atomic::AtomicBool};

fn recording() -> Option<Arc<Replay>> {
    let (Some(l), Some(s)) = (std::env::var_os("X5_LIVE"), std::env::var_os("X5_SAMPLES")) else {
        eprintln!("X5_LIVE or X5_SAMPLES not set: skipping");
        return None;
    };
    let c = std::env::var_os("X5_CONF").map(PathBuf::from);
    let r = Replay::open(
        l.as_ref(),
        &PathBuf::from(s).join("v635.insv"),
        c.as_deref(),
    )
    .unwrap();
    Some(Arc::new(r))
}

/// A camera worker on the simulated camera and a Capture screen on its session, without a window.
fn rig(r: &Arc<Replay>, on: bool) -> (Arc<Gpu>, Worker, Capture) {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = Ctx::new(&g.device, &g.queue);
    let tap = Taps::default();
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let auto = Arc::new(AtomicBool::new(false));
    let w = Worker::spawn(
        replay::connect(r.clone()),
        q,
        Cache::at(None),
        auto,
        false,
        tap.clone(),
        || {},
    );
    w.tx.send(Cc::Connect).unwrap();
    let h = Hook {
        on,
        ..Default::default()
    };
    let c = Capture::new(&g, &cx, &egui::Context::default(), tap, Some(h));
    (g, w, c)
}

fn until(c: &mut Capture, secs: f64, f: impl Fn(&Capture) -> bool) {
    let t = Instant::now();
    while !f(c) {
        assert!(t.elapsed().as_secs_f64() < secs, "timed out");
        c.tick();
        thread::sleep(Duration::from_millis(5));
    }
}

/// 10 s of the replayed preview in a 1280 × 720 viewer polled at 60 Hz: every picture decoded, nearly every one
/// shown, the calibration from MEDIA_OFFSET_V3, the clock synced, and a picture that is not black.
#[test]
fn replay_path() {
    let Some(r) = recording() else { return };
    let (g, _w, mut c) = rig(&r, true);
    let t = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1280,
            height: 720,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::player::FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    let t0 = Instant::now();
    let mut first = None;
    while first.is_none_or(|f: Instant| f.elapsed() < Duration::from_secs(10)) {
        c.tick();
        if let Some(f) = &c.feed {
            let mut f = f.lock().unwrap();
            f.show(&v, [1280, 720], false);
            if f.stats.shown > 0 && first.is_none() {
                first = Some(Instant::now());
                println!(
                    "first picture {:.0} ms after the start",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(40), "no picture in 40 s");
        thread::sleep(Duration::from_millis(16));
    }
    let f = c.feed.as_ref().unwrap().lock().unwrap();
    let (k, s) = (f.counts(), &f.stats);
    println!("replay: {}", f.report());
    assert!(f.err().is_none(), "{:?}", f.err());
    assert_eq!(f.from, "MEDIA_OFFSET_V3");
    assert_eq!(k.bad, 0);
    // real time on a shared host and GPU: the numbers above say how it went, these only that it flowed
    assert!(k.dec >= 270, "{} decoded in 10 s", k.dec);
    assert!(s.shown >= 225, "{} shown in 10 s", s.shown);
    assert!(s.g2g.n >= 200, "the camera clock: {:?}", c.st.off);
    assert!(s.stitch.n > 0, "no GPU timestamps");
    assert_eq!(
        c.st.params.as_ref().map(|p| (p.res_width, p.res_height)),
        Some((2656, 1328))
    );
    // the view, and the whole sphere at the same size; into `X5_FIXTURES` for a look if set
    let px = stitch::read_texture(&g.device, &g.queue, &t);
    drop(f);
    {
        let mut f = c.feed.as_ref().unwrap().lock().unwrap();
        f.look = Look::Equirect;
        f.view = Look::Equirect.view();
        f.show(&v, [1280, 720], true);
    }
    let eq = stitch::read_texture(&g.device, &g.queue, &t);
    for (k, px) in [("view", &px), ("360", &eq)] {
        let n = (px.len() / 4) as f64;
        let luma = |p: &[u8]| (p[0] as f64 + p[1] as f64 + p[2] as f64) / 3.0;
        let m = px.chunks(4).map(luma).sum::<f64>() / n;
        let sd = (px.chunks(4).map(|p| (luma(p) - m).powi(2)).sum::<f64>() / n).sqrt();
        println!("{k}: mean {m:.1}, sd {sd:.1}");
        assert!(m > 20.0 && m < 235.0 && sd > 5.0, "{k}: a flat picture");
        if let Some(d) = std::env::var_os("X5_FIXTURES").map(PathBuf::from) {
            crate::png::write(&d.join(format!("replay-{k}.png")), 1280, 720, px).unwrap();
        }
    }
    assert_eq!(c.st.mode.as_deref(), Some("VIDEO_NORMAL"));
    if r.conf.is_some() {
        assert!(c.st.conf.is_some());
    }
}

/// Ready for a screenshot (main.rs) once the worker reported the camera's state, also without `/CONF`: the replayed
/// camera has none here (a `replay=` run without `conf=`; a camera whose `/CONF` read fails), the preview is off.
#[test]
fn idle_without_conf() {
    let Some(r) = recording() else { return };
    let mut x = Arc::into_inner(r).unwrap();
    x.conf = None;
    let (_g, _w, mut c) = rig(&Arc::new(x), false);
    until(&mut c, 20.0, |c| c.st.on);
    assert_eq!(c.st.mode.as_deref(), Some("VIDEO_NORMAL"));
    assert!(
        c.st.conf.is_none() && c.idle(),
        "waits for a /CONF that is not there"
    );
}

/// The replayed preview (sent in pairs, as the X5 does) as a live source: 10 s of a 30 fps output clock taking and
/// drawing it as the Live screen's thread does, a 60 Hz viewer beside it. The clock takes every picture in turn, none
/// dropped and none repeated but around a stall of this shared host; the viewer shows what the clock takes.
#[test]
fn replay_output() {
    let Some(r) = recording() else { return };
    let (g, _w, mut c) = rig(&r, true);
    let tex = |w, h| {
        g.device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::player::FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let (out, view) = (tex(1280, 640), tex(640, 360));
    let t0 = Instant::now();
    loop {
        c.tick();
        if let Some(f) = &c.feed {
            let mut f = f.lock().unwrap();
            f.show(&view, [640, 360], false);
            if f.ready() && f.counts().dec > 30 {
                break;
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(40), "no picture in 40 s");
        thread::sleep(Duration::from_millis(16));
    }
    let f = c.feed.clone().unwrap();
    let (k0, s0) = {
        let f = f.lock().unwrap();
        (f.counts(), f.stats.shown)
    };
    let (per, mut next, mut ui) = (1_000_000_000 / 30, mono(), Instant::now());
    let (mut late, end) = (0u64, mono() + 10_000_000_000);
    while next < end {
        c.tick();
        let now = mono();
        if now >= next {
            late = late.max(now - next);
            let mut f = f.lock().unwrap();
            crate::live::Source::tick(&mut *f, next);
            if crate::live::Source::ready(&*f) {
                crate::live::Source::draw(&mut *f, &out, [1280, 640], true);
            }
            next += per;
        }
        if ui.elapsed() >= Duration::from_micros(16_667) {
            ui = Instant::now();
            f.lock().unwrap().show(&view, [640, 360], false);
        }
        thread::sleep(Duration::from_millis(1));
    }
    let f = f.lock().unwrap();
    let k = f.counts();
    let (ticks, rep, skip) = (k.ticks - k0.ticks, k.rep - k0.rep, k.skip - k0.skip);
    let stalls = k.late - k0.late;
    println!(
        "output clock: {ticks} ticks (latest {:.1} ms late), {rep} repeated, {skip} dropped, {stalls} pictures \
         stalled, {} decoded, {} lost on the link; viewer drew {}; {}",
        late as f64 * 1e-6,
        k.dec - k0.dec,
        f.lost,
        f.stats.shown - s0,
        f.report()
    );
    assert!(f.err().is_none(), "{:?}", f.err());
    // real time on a shared host and GPU: a picture stalled past its due costs a repeat and a drop or two
    // (feed::tests::pacing); none at all but the start without
    let ok = 1 + 2 * stalls;
    assert!(ticks >= 295, "{ticks} ticks");
    assert!(
        rep <= ok && skip <= ok,
        "{rep} repeated, {skip} dropped, {stalls} stalled"
    );
    assert!(
        f.stats.shown - s0 >= ticks - rep - 3,
        "the viewer lost pictures"
    );
}

/// The commands the screen sends, as the camera gets them: a setting (from the camera's values), a mode switch, the
/// shutter in the video mode, a highlight mark.
#[test]
fn commands() {
    let Some(r) = recording() else { return };
    if r.conf.is_none() {
        eprintln!("X5_CONF not set: skipping");
        return;
    }
    let (_g, _w, mut c) = rig(&r, false);
    until(&mut c, 10.0, |c| c.st.conf.is_some() && c.st.mode.is_some());
    let n0 = r.log.lock().unwrap().len();
    let po = c.st.po.clone();
    for x in [
        Cmd::Set("exposure_bias".into(), "-0.3".into()),
        Cmd::Set("exposure_iso".into(), "400".into()),
        Cmd::Mode("VIDEO_PURE".into()),
        Cmd::Shoot,
        Cmd::Mark,
    ] {
        c.tx.send(x).unwrap();
    }
    let want = |k: u16| {
        move |r: &Replay| {
            r.log.lock().unwrap()[n0..]
                .iter()
                .filter(|l| l.0 == k)
                .count()
        }
    };
    let t = Instant::now();
    while want(code::KEY_TIME_POINT)(&r) == 0 {
        assert!(t.elapsed() < Duration::from_secs(10), "commands not sent");
        c.tick();
        thread::sleep(Duration::from_millis(5));
    }
    let log = r.log.lock().unwrap()[n0..].to_vec();
    let sent: Vec<u16> = log
        .iter()
        .map(|l| l.0)
        .filter(|&k| matches!(k, 4 | 7 | 9 | 60))
        .collect();
    println!("sent {sent:?}");
    assert_eq!(sent, [9, 9, 7, 4, 60]);
    let body = |k: u16, i: usize| log.iter().filter(|l| l.0 == k).nth(i).unwrap().1.clone();
    let fm = pb::FunctionMode::NormalVideo;
    let e = |s, v| match conf::write(s, v, &po, fm, false).unwrap() {
        Write::Photo(m) => m.encode_to_vec(),
        Write::Opts(m) => m.encode_to_vec(),
    };
    assert_eq!(body(9, 0), e(Slot::Ev, "-0.3"));
    // the camera's FULL_AUTO and shutter go along with the ISO
    let q = pb::SetPhotographyOptions::decode(&body(9, 1)[..]).unwrap();
    let x = q.value.unwrap().video_exposure.unwrap();
    assert_eq!((x.program, x.iso), (5, 400));
    // SET_OPTIONS {VIDEO_SUB_MODE: VIDEO_PURE (11)}
    assert_eq!(body(7, 0), [0x08, 0x29, 0x12, 0x03, 0xc8, 0x02, 0x0b]);
    // START_CAPTURE {NORMAL}: the simulated camera stays in VIDEO_NORMAL
    assert_eq!(body(4, 0), [0x08, 0x01]);
    let k = pb::SetKeyTimePoint::decode(&body(60, 0)[..]).unwrap();
    assert_eq!(k.type_detail, "com.insta360.keyTime.highlight");
}

/// The real X5 over USB (`X5_CAMERA` set; read-only apart from the preview's start and stop): 30 s of its live
/// preview through the camera worker, the capture worker and the decode thread into a 1600 × 800 viewer at 60 Hz;
/// the calibration it reports against the v635 file's offset_v3 (`X5_SAMPLES`; same camera: the string the stitch
/// crate's live test checked the seam with); its `/CONF` and mode; the notifications; a clean stop and close. The
/// equirect and a view across the seam (not levelled) go to `X5_FIXTURES` if set.
#[test]
fn real_camera() {
    if std::env::var_os("X5_CAMERA").is_none() {
        eprintln!("X5_CAMERA not set: skipping");
        return;
    }
    let secs: f64 = std::env::var("X5_CAMERA_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let g = Arc::new(Gpu::new().unwrap());
    let cx = Ctx::new(&g.device, &g.queue);
    let tap = Taps::default();
    let q = Arc::new(Mutex::new(dl::Queue::load(None)));
    let auto = Arc::new(AtomicBool::new(false));
    let w = Worker::spawn(
        Box::new(Camera::usb),
        q,
        Cache::at(None),
        auto,
        false,
        tap.clone(),
        || {},
    );
    let t0 = Instant::now();
    w.tx.send(Cc::Connect).unwrap();
    let h = Hook {
        on: true,
        ..Default::default()
    };
    let mut c = Capture::new(&g, &cx, &egui::Context::default(), tap, Some(h));
    let (tw, th) = (1600u32, 800u32);
    let t = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: tw,
            height: th,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::player::FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    let mut first = None;
    while first.is_none_or(|f: Instant| f.elapsed().as_secs_f64() < secs) {
        c.tick();
        if let Some(f) = &c.feed {
            let mut f = f.lock().unwrap();
            f.show(&v, [tw, th], false);
            if f.stats.shown > 0 && first.is_none() {
                first = Some(Instant::now());
                println!(
                    "first picture {:.0} ms after Connect",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "no picture in 60 s");
        thread::sleep(Duration::from_millis(16));
    }
    let s = c.st.clone();
    {
        let f = c.feed.as_ref().unwrap().lock().unwrap();
        println!("real camera: {}", f.report());
        assert!(f.err().is_none(), "{:?}", f.err());
        assert!(f.counts().imu > 0, "no IMU in the stream");
        assert!(
            f.stats.shown as f64 >= 25.0 * secs,
            "{} shown",
            f.stats.shown
        );
    }
    let (cal, from) = s.cal.clone().expect("calibration");
    println!(
        "calibration: {from} from the camera, {} bytes, parses: {}",
        cal.len(),
        stitch::Calib::parse(&cal).is_ok()
    );
    if let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) {
        let t = media::insv::Trailer::open(d.join("v635.insv")).unwrap();
        let mut b = vec![];
        t.get(media::insv::METADATA, &mut b).unwrap();
        let v3 = pb::ExtraMetadata::decode(&b[..]).unwrap().offset_v3;
        println!(
            "the camera's {from} is the v635 file's offset_v3: {}",
            v3 == cal
        );
    }
    let p = s.params.expect("8234");
    let wc = p.window_crop_info.unwrap_or_default();
    println!(
        "stream {} × {} at {} fps, window {}² of {}² offset ({}, {}); camera clock round trip {:?} ms",
        p.res_width,
        p.res_height,
        p.res_fps,
        wc.dst_width,
        wc.src_width,
        wc.crop_offset_x,
        wc.crop_offset_y,
        s.off.map(|o| o.1)
    );
    let cf = s.conf.clone().expect("/CONF");
    println!(
        "/CONF version {}, mode {:?} ({}), {} settings in it; battery {:?}, storage {:?}, temperature {:?} / {} °C; \
         {} notifications, last {}",
        cf.version,
        s.mode,
        if s.both { "360" } else { "single lens" },
        s.mode
            .as_deref()
            .and_then(|m| cf.block(s.both)?.mode(m))
            .map_or(0, |m| m.keys.len()),
        s.o.battery_info.as_ref().map(camera::battery),
        s.o.storage_state
            .as_ref()
            .map(|x| (x.free_space, x.total_space)),
        s.temp,
        s.o.temp_value,
        s.notes,
        s.last
    );
    if let Some(d) = std::env::var_os("X5_FIXTURES").map(PathBuf::from) {
        for (k, look, yaw) in [
            ("360", Look::Equirect, 0.0),
            ("seam", Look::Perspective, 90.0),
        ] {
            {
                let mut f = c.feed.as_ref().unwrap().lock().unwrap();
                (f.look, f.level) = (look, false);
                f.view = stitch::View { yaw, ..look.view() };
                f.show(&v, [tw, th], true);
            }
            let px = stitch::read_texture(&g.device, &g.queue, &t);
            crate::png::write(&d.join(format!("real-{k}.png")), tw, th, &px).unwrap();
        }
    }
    // stop: the preview goes, STOP_LIVE_STREAM is answered
    let t1 = Instant::now();
    c.want = false;
    until(&mut c, 5.0, |c| c.feed.is_none() && !c.st.live);
    println!(
        "preview stopped in {:.0} ms",
        t1.elapsed().as_secs_f64() * 1e3
    );
    let t2 = Instant::now();
    drop(c);
    drop(w);
    println!(
        "session closed in {:.0} ms",
        t2.elapsed().as_secs_f64() * 1e3
    );
}

/// A new session takes no report of the one before: the old session's report still in the channel when the session
/// changes (here after a note on it) leaves the new one not ready (it used to be taken and mark it ready early). The
/// new session's camera never answers, so it reports nothing meanwhile.
#[test]
fn stale_report() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = Ctx::new(&g.device, &g.queue);
    let tap = Taps::default();
    let mut c = Capture::new(&g, &cx, &egui::Context::default(), tap.clone(), None);
    let sim = |s: u16| Camera::sim(move |_, _| (s, vec![]), |_| None, 1).unwrap();
    let set = |c: &Camera, s: u32| {
        let mut t = tap.lock().unwrap();
        (t.cam, t.sid) = (Some(c.clone()), s);
    };
    let (a, _ea, _sa) = sim(200);
    set(&a, 1);
    until(&mut c, 10.0, |c| c.st.on);
    c.tx.send(Cmd::Note(Note::StorageFull)).unwrap();
    thread::sleep(Duration::from_millis(300));
    let (b, _eb, _sb) = sim(0);
    set(&b, 2);
    c.tick();
    let (on, note, idle) = (c.st.on, c.st.note.clone(), c.idle());
    // the new session's requests end with its session, so the screen's worker can quit
    b.close().wait().unwrap();
    assert!(!on && note.is_none() && !idle, "{on} {note:?} {idle}");
}

/// The notifications the status shows: capture state and time, battery (low), temperature, a stopped capture.
#[test]
fn notes() {
    use pb::CameraCaptureState as C;
    let (tx, _rx) = mpsc::channel();
    let mut w = W {
        cam: None,
        sid: 0,
        st: St::default(),
        up: tx,
        wake: Box::new(|| {}),
    };
    let cap = |s: C, t| {
        Note::CaptureStatus(pb::CaptureStatus {
            state: s as i32,
            capture_time: t,
            ..Default::default()
        })
    };
    w.on(cap(C::NormalCapture, 12));
    let s = w.st.since.unwrap().elapsed().as_secs_f64();
    assert!(running(w.st.cap) && (12.0..12.5).contains(&s), "{s}");
    w.on(Note::Battery(pb::BatteryInfo {
        battery_level: 9,
        ..Default::default()
    }));
    assert!(camera::low(w.st.o.battery_info.as_ref().unwrap()));
    let t = pb::TempState {
        temp_state: pb::TemperatureState::TempHigh as i32,
        temp_value: 61,
        ..Default::default()
    };
    w.on(Note::Temp(t));
    assert_eq!(w.st.temp, Some(t));
    w.on(cap(C::NotCapture, 0));
    assert!(!running(w.st.cap) && w.st.since.is_none());
    w.on(Note::CaptureStopped(pb::NotificationCaptureStopped {
        err_code: pb::notification_capture_stopped::ErrorCode::LowBattery as i32,
        ..Default::default()
    }));
    let (n, e) = w.st.note.clone().unwrap();
    assert!(e && n.contains("LowBattery"), "{n}");
    assert_eq!(w.st.notes, 5);
    // photos and the self-timer leave the files alone; timelapses run until stopped
    assert!(!running(C::SingleShooting as i32) && running(C::TimelapseCapture as i32));
}

/// A Capture screen and a camera worker on the real X5 (no window), what the worker reported, every notification
/// with when it came, and what to set back.
struct Real {
    // the screen goes first: its worker's requests end before the session closes
    c: Capture,
    w: Worker,
    cam: Camera,
    notes: Arc<Mutex<Vec<(Instant, Note)>>>,
    album: Vec<Item>,
    albums: usize,
    said: Vec<(String, bool)>,
    /// the capture screen's last note, printed once
    cnote: Option<(String, bool)>,
    /// (mode, setting, value before the first change), in the order of the changes
    undo: Vec<(String, String, String)>,
    /// the captures the test made, as the camera reported them
    made: Vec<String>,
    _g: Arc<Gpu>,
}

impl Real {
    fn open(d: &Path) -> Real {
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let tap = Taps::default();
        let q = Arc::new(Mutex::new(dl::Queue::load(None)));
        let w = Worker::spawn(
            Box::new(Camera::usb),
            q,
            Cache::at(Some(d.join("thumbs"))),
            Arc::new(AtomicBool::new(false)),
            false,
            tap.clone(),
            || {},
        );
        w.tx.send(Cc::Connect).unwrap();
        let mut c = Capture::new(&g, &cx, &egui::Context::default(), tap.clone(), None);
        let notes = Arc::new(Mutex::new(vec![]));
        let (t2, n2) = (c.tx.clone(), notes.clone());
        tap.lock().unwrap().notes = Some(Box::new(move |n: &Note| {
            n2.lock().unwrap().push((Instant::now(), n.clone()));
            drop(t2.send(Cmd::Note(n.clone())))
        }));
        let t = Instant::now();
        let cam = loop {
            c.tick();
            if let Some(x) = tap.lock().unwrap().cam.clone() {
                break x;
            }
            assert!(t.elapsed() < Duration::from_secs(40), "not connected");
            thread::sleep(Duration::from_millis(10));
        };
        Real {
            c,
            w,
            cam,
            notes,
            album: vec![],
            albums: 0,
            said: vec![],
            cnote: None,
            undo: vec![],
            made: vec![],
            _g: g,
        }
    }

    /// Runs the screen and takes the worker's reports until `f` holds; false after `secs`.
    fn pump(&mut self, secs: f64, f: impl Fn(&Real) -> bool) -> bool {
        let end = Instant::now() + Duration::from_secs_f64(secs.max(0.0));
        loop {
            self.c.tick();
            while let Ok(u) = self.w.rx.try_recv() {
                match u {
                    Up::Album(v) => (self.album, self.albums) = (v, self.albums + 1),
                    Up::Note(s, e) => {
                        println!("  camera screen: {s}{}", if e { " (error)" } else { "" });
                        self.said.push((s, e))
                    }
                    _ => {}
                }
            }
            if self.c.st.note.is_some() && self.c.st.note != self.cnote {
                self.cnote = self.c.st.note.clone();
                let (s, e) = self.cnote.as_ref().unwrap();
                println!("  capture screen: {s}{}", if *e { " (error)" } else { "" });
            }
            if f(self) {
                return true;
            }
            if Instant::now() >= end {
                return false;
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn files(&self) -> Result<Vec<String>, String> {
        use pb::{CardLocation::StorClCamera as L, MediaType as M};
        let mut v =
            (self.cam.files(M::VideoAndPhoto, L).wait()).map_err(|e| format!("files: {e}"))?;
        // the X5 lists a photo's DNG under both media types
        for u in (self.cam.files(M::Dng, L).wait()).map_err(|e| format!("files: {e}"))? {
            if !v.contains(&u) {
                v.push(u)
            }
        }
        Ok(v)
    }

    /// What the tests change and set back: the sub-modes, in-camera stitching, and the photography options of the
    /// photo and video modes they use (every type the shown settings read).
    fn snap(&self) -> String {
        use pb::OptionType::*;
        let o = self.cam.options(&[
            PhotoSubMode,
            VideoSubMode,
            InternalSplicing,
            InstapanoFocusSensor,
        ]);
        let mut s = format!("{:?}\n", o.wait());
        let mut ts = vec![];
        for (_, sl, _) in conf::KEYS {
            for &t in conf::types(sl, true).iter().chain(conf::types(sl, false)) {
                if !ts.contains(&t) {
                    ts.push(t)
                }
            }
        }
        for fm in [pb::FunctionMode::NormalImage, pb::FunctionMode::NormalVideo] {
            let v = self.cam.photo_options(&ts, fm).wait().map(|r| r.value);
            s += &format!("{fm:?}: {v:?}\n");
        }
        s
    }

    fn key(&self, k: &str) -> Option<conf::Key> {
        let s = &self.c.st;
        let m = s.conf.as_ref()?.block(s.both)?.mode(s.mode.as_ref()?)?;
        m.keys.iter().find(|x| x.name == k).cloned()
    }

    /// The value the screen shows for setting `k`.
    fn cur(&self, k: &str) -> Option<String> {
        let s = &self.c.st;
        self.c
            .cur(s.conf.as_ref()?.block(s.both)?.mode(s.mode.as_ref()?)?, k)
    }

    /// The setting control's change (offered values only), then the value the screen shows and a read of its own.
    fn set(&mut self, k: &str, v: &str) -> Result<(), String> {
        let m = self.c.st.mode.clone().ok_or("no mode")?;
        let al = (self.key(k).map(|x| x.allowed(&|d| self.cur(d)).to_vec()))
            .ok_or(format!("{k}: not a setting of {m}"))?;
        if !al.iter().any(|x| x == v) {
            return Err(format!("{k} = {v}: not offered now ({al:?})"));
        }
        let old = self.cur(k).ok_or(format!("{k}: no value"))?;
        if old == v {
            return Ok(());
        }
        if !self.undo.iter().any(|u| u.0 == m && u.1 == k) {
            self.undo.push((m.clone(), k.into(), old.clone()))
        }
        let t = Instant::now();
        self.c.tx.send(Cmd::Set(k.into(), v.into())).unwrap();
        if !self.pump(10.0, |r| r.cur(k).as_deref() == Some(v)) {
            return Err(format!(
                "{k} {old} → {v}: the screen shows {:?} after 10 s",
                self.cur(k)
            ));
        }
        let sl = conf::slot(k).unwrap();
        let photo = conf::photo(&m);
        let po = self
            .cam
            .photo_options(conf::types(sl, photo), conf::func(&m).unwrap())
            .wait();
        let o = self.cam.options(&[pb::OptionType::InternalSplicing]).wait();
        let (po, o) = (
            po.map_err(|e| e.to_string())?,
            o.map_err(|e| e.to_string())?,
        );
        let x = conf::get(
            sl,
            &po.value.unwrap_or_default(),
            o.enable_internal_splicing,
            photo,
        );
        if !conf::val(sl, v).is_some_and(|y| conf::same(x, y)) {
            return Err(format!("{k} = {v}: the camera reads back {x}"));
        }
        println!(
            "  {m}: {k} {old} → {v} in {:.0} ms, read back",
            t.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    fn mode(&mut self, m: &str) -> Result<(), String> {
        let (t, was) = (Instant::now(), self.c.st.mode.clone());
        self.c.tx.send(Cmd::Mode(m.into())).unwrap();
        if !self.pump(15.0, |r| r.c.st.mode.as_deref() == Some(m)) {
            return Err(format!(
                "mode {m}: the screen shows {:?} after 15 s",
                self.c.st.mode
            ));
        }
        let o = self
            .cam
            .options(&[pb::OptionType::PhotoSubMode, pb::OptionType::VideoSubMode])
            .wait();
        let o = o.map_err(|e| e.to_string())?;
        if conf::current(&o).as_deref() != Some(m) {
            return Err(format!(
                "mode {m}: the camera reads back {:?}",
                conf::current(&o)
            ));
        }
        println!(
            "  mode {was:?} → {m} in {:.0} ms, read back",
            t.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    /// The shutter in a photo mode: the new photo's URI (the screen's note).
    fn photo(&mut self) -> Result<String, String> {
        let (t, prev) = (Instant::now(), self.c.st.note.clone());
        self.c.tx.send(Cmd::Shoot).unwrap();
        let got = |r: &Real| r.c.st.note != prev && r.c.st.note.is_some();
        if !self.pump(40.0, got) || self.c.st.note.as_ref().unwrap().1 {
            return Err(format!("photo: {:?}", self.c.st.note));
        }
        let u = self
            .c
            .st
            .note
            .as_ref()
            .unwrap()
            .0
            .trim_start_matches("photo taken ")
            .to_string();
        println!("  photo {u} in {:.1} s", t.elapsed().as_secs_f64());
        self.made.push(u.clone());
        Ok(u)
    }

    /// The shutter in a video mode: start, a highlight mark `mark` s after it, stop after `secs`. The clip's URI, and
    /// when the screen showed the recording.
    fn record(&mut self, secs: f64, mark: Option<f64>) -> Result<(String, f64), String> {
        let t = Instant::now();
        self.c.tx.send(Cmd::Shoot).unwrap();
        // the screen knows it records from the camera's capture status
        if !self.pump(5.0, |r| running(r.c.st.cap)) {
            return Err(format!(
                "recording: no capture state 5 s after the shutter; {:?}",
                self.c.st.note
            ));
        }
        let shown = t.elapsed().as_secs_f64();
        println!(
            "  recording: the screen shows it {:.0} ms after the shutter",
            shown * 1e3
        );
        if let Some(m) = mark {
            self.pump(m - t.elapsed().as_secs_f64(), |_| false);
            self.c.tx.send(Cmd::Mark).unwrap();
            let m = |r: &Real| {
                r.c.st
                    .note
                    .as_ref()
                    .is_some_and(|n| n.0.starts_with("highlight"))
            };
            if !self.pump(5.0, m) {
                return Err(format!("highlight: {:?}", self.c.st.note));
            }
        }
        self.pump(secs - t.elapsed().as_secs_f64(), |_| false);
        let prev = self.c.st.note.clone();
        self.c.tx.send(Cmd::Shoot).unwrap();
        let got = |r: &Real| r.c.st.note != prev && r.c.st.note.is_some();
        if !self.pump(30.0, got) || !self.c.st.note.as_ref().unwrap().0.starts_with("recorded") {
            return Err(format!("stop: {:?}", self.c.st.note));
        }
        let u = self
            .c
            .st
            .note
            .as_ref()
            .unwrap()
            .0
            .trim_start_matches("recorded ")
            .to_string();
        self.made.push(u.clone());
        println!(
            "  recorded {u}, stopped {:.1} s after the shutter",
            t.elapsed().as_secs_f64()
        );
        Ok((u, shown))
    }

    /// The album's Delete of the items holding exactly `files` (what its button sends), then a fresh listing.
    fn delete(&mut self, files: &[String]) -> Result<(), String> {
        let f: HashSet<&String> = files.iter().collect();
        // the album is listed again shortly after the new files' notifications
        self.pump(15.0, |r| {
            f.iter()
                .all(|u| r.album.iter().any(|it| it.uris.contains(u)))
        });
        let items = (self.album.iter())
            .filter(|it| it.uris.iter().any(|u| f.contains(u)))
            .count();
        let u: Vec<String> = (self.album.iter())
            .filter(|it| it.uris.iter().any(|u| f.contains(u)))
            .flat_map(|it| it.uris.clone())
            .collect();
        if u.iter().collect::<HashSet<_>>() != f {
            return Err(format!(
                "delete: the album's items {u:?} are not the files {files:?}"
            ));
        }
        let (t, k) = (Instant::now(), self.said.len());
        self.w.tx.send(Cc::Delete(u)).unwrap();
        if !self.pump(30.0, |r| r.said[k..].iter().any(|s| s.0.contains("delete"))) {
            return Err("delete: no answer in 30 s".into());
        }
        let s = self.said[k..]
            .iter()
            .find(|s| s.0.contains("delete"))
            .unwrap()
            .clone();
        if s.1 || s.0 != format!("deleted {0} of {0} files", files.len()) {
            return Err(format!("delete: {}", s.0));
        }
        println!(
            "  {items} items deleted through the album in {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        Ok(())
    }

    /// The camera as the test found it, however the test ended: a running capture stopped, files it made deleted
    /// (only those: [`ours`]), the settings it changed set back (latest first, each in its mode), the mode. Returns
    /// what failed.
    fn restore(&mut self, files0: &[String], mode0: &str) -> Vec<String> {
        let mut bad = vec![];
        if let Ok(s) = self.cam.capture_status().wait()
            && running(s.state)
        {
            bad.push(format!("a capture still ran ({:?}): stopped", s.state()));
            if let Ok(v) = self.cam.stop_capture(pb::CaptureMode::Normal).wait() {
                self.made.push(v.uri)
            }
            self.pump(3.0, |_| false);
        }
        match self.files() {
            Ok(v) => {
                let (left, other) = ours(v, files0, &self.made);
                if !other.is_empty() {
                    bad.push(format!(
                        "new files the test did not make, left alone: {other:?}"
                    ));
                }
                if !left.is_empty() {
                    bad.push(format!("files left by the test, deleted: {left:?}"));
                    if let Err(e) = self.cam.delete(left).wait() {
                        bad.push(format!("deleting them: {e}"))
                    }
                }
            }
            Err(e) => bad.push(e),
        }
        while let Some((m, k, v)) = self.undo.pop() {
            if self.c.st.mode.as_deref() != Some(&m)
                && let Err(e) = self.mode(&m)
            {
                bad.push(e)
            }
            let n = self.undo.len();
            if let Err(e) = self.set(&k, &v) {
                bad.push(format!("setting back: {e}"))
            }
            // set pushes nothing for a value it sets back
            self.undo.truncate(n);
        }
        if self.c.st.mode.as_deref() != Some(mode0)
            && let Err(e) = self.mode(mode0)
        {
            bad.push(e)
        }
        bad
    }
}

/// Of the camera's files `now`, the ones a real-camera test may delete: new since `files0` and of a capture it made
/// (`made`, the URIs the camera reported; their recording's other files by the recording key: the proxy, a photo's
/// DNG); and the other new ones, someone else's (a clip shot on the camera meanwhile), which stay.
fn ours(now: Vec<String>, files0: &[String], made: &[String]) -> (Vec<String>, Vec<String>) {
    let key = |u: &str| crate::library::key(crate::album::name(u));
    let ks: HashSet<String> = (made.iter())
        .filter(|u| !u.is_empty())
        .map(|u| key(u))
        .collect();
    (now.into_iter())
        .filter(|u| !files0.contains(u))
        .partition(|u| ks.contains(&key(u)))
}

/// What a real-camera test deletes: the files of the photo and the clip it made, not a clip that appeared meanwhile.
#[test]
fn ours_only() {
    let d = |n: &str| format!("/DCIM/Camera01/{n}");
    let f0 = vec![d("VID_20261025_100000_00_001.insv")];
    let now: Vec<String> = [
        "VID_20261025_100000_00_001.insv",
        "IMG_20261025_101010_00_002.insp",
        "IMG_20261025_101010_00_002.dng",
        "VID_20261025_101100_00_003.insv",
        "LRV_20261025_101100_01_003.lrv",
        "VID_20261025_101200_00_004.insv",
        "LRV_20261025_101200_01_004.lrv",
    ]
    .map(d)
    .into();
    let made = [
        d("IMG_20261025_101010_00_002.insp"),
        d("VID_20261025_101100_00_003.insv"),
        String::new(),
    ];
    let (del, other) = ours(now.clone(), &f0, &made);
    assert_eq!((&del[..], &other[..]), (&now[1..5], &now[5..]));
}

/// The Mode list marks every mode but the two whose switch ran on an X5 (`real_writes`).
#[test]
fn modes_marked() {
    let l = ["PHOTO_SINGLE", "PHOTO_BURST", "VIDEO_NORMAL", "VIDEO_PURE"].map(mlabel);
    assert_eq!(l, ["Photo", "Burst ⚠", "Video", "PureVideo ⚠"]);
}

/// The camera's wall clock in a file name (`IMG_20261025_034512_00_123.jpg`) read in this computer's zone, unix s.
fn named(u: &str) -> Option<i64> {
    let p: Vec<&str> = crate::album::name(u).split('_').collect();
    let n = |s: &str, a: usize, b: usize| s.get(a..b)?.parse::<i32>().ok();
    let (d, t) = (p.get(1)?, p.get(2)?);
    // SAFETY: mktime only reads and normalises the tm it is given
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        (tm.tm_year, tm.tm_mon, tm.tm_mday) = (n(d, 0, 4)? - 1900, n(d, 4, 6)? - 1, n(d, 6, 8)?);
        (tm.tm_hour, tm.tm_min, tm.tm_sec, tm.tm_isdst) =
            (n(t, 0, 2)?, n(t, 2, 4)?, n(t, 4, 6)?, -1);
        Some(libc::mktime(&mut tm) as i64)
    }
}

fn unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

/// Notifications by kind, as `kind ×count`.
fn codes(notes: &[(Instant, Note)]) -> String {
    let mut v: Vec<(String, usize)> = vec![];
    for (_, n) in notes {
        let k = match n {
            Note::Other(c, _) => format!("{c}"),
            n => format!("{n:?}").split('(').next().unwrap_or("").to_string(),
        };
        match v.iter_mut().find(|x| x.0 == k) {
            Some(x) => x.1 += 1,
            None => v.push((k, 1)),
        }
    }
    v.iter()
        .map(|(k, n)| format!("{k} ×{n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The real X5 with the state-changing checks the user allowed (`X5_CAMERA_WRITE=<scratch dir>`), each through the
/// command its control sends, checked with reads of its own on the same session: a setting of the current mode
/// changed and set back; a test photo; a time sync, the clock skew from the photo's and the clip's names before and
/// after; a favourite on an existing clip and off again; a 5 s clip in VIDEO_NORMAL with a highlight mark (the clip's
/// HIGHLIGHT trailer record) and the notifications meanwhile; the photo and the clip in the album, then deleted from
/// it. Mode, settings and file list end as they began, also when a check fails.
#[test]
fn real_writes() {
    let Some(d) = std::env::var_os("X5_CAMERA_WRITE").map(PathBuf::from) else {
        eprintln!("X5_CAMERA_WRITE not set: skipping");
        return;
    };
    std::fs::create_dir_all(&d).unwrap();
    let mut r = Real::open(&d);
    let ready = |r: &Real| r.c.st.conf.is_some() && r.c.st.mode.is_some() && r.albums > 0;
    assert!(r.pump(60.0, ready), "no /CONF, mode or album in 60 s");
    let (files0, snap0, mode0) = (r.files().unwrap(), r.snap(), r.c.st.mode.clone().unwrap());
    println!("start: mode {mode0}, {} files\n{snap0}", files0.len());
    let mut fav0 = None;
    let res = (|| -> Result<(), String> {
        // a setting of the current mode and back: the first of these it offers a choice in
        let (k, was, to) = ["exposure_bias", "white_balance", "photography_self_timer"]
            .into_iter()
            .find_map(|k| {
                let was = r.cur(k)?;
                let al = r.key(k)?.allowed(&|d| r.cur(d)).to_vec();
                Some((k, was.clone(), al.into_iter().find(|v| *v != was)?))
            })
            .ok_or("no setting to change")?;
        let n = r.notes.lock().unwrap().len();
        r.set(k, &to)?;
        r.pump(1.0, |_| false);
        println!(
            "  notifications after the change: {}",
            codes(&r.notes.lock().unwrap()[n..])
        );
        r.set(k, &was)?;
        r.undo.clear();

        // a favourite on an existing clip (the one the test downloads elsewhere), off again
        let it = (r.album.iter().find(|i| i.key.contains("141148")))
            .or(r.album.iter().find(|i| i.kind == crate::album::Kind::Video))
            .ok_or("no clip")?
            .clone();
        let flag = it.flag().to_string();
        fav0 = Some((flag.clone(), it.fav));
        for on in [!it.fav, it.fav] {
            let (n, k, a) = (r.albums, r.notes.lock().unwrap().len(), r.said.len());
            r.w.tx.send(Cc::Fav(vec![(flag.clone(), on)])).unwrap();
            let said = |r: &Real| {
                r.said[a..]
                    .iter()
                    .find(|s| s.0.starts_with("favourite"))
                    .cloned()
            };
            if !r.pump(10.0, |r| said(r).is_some()) {
                return Err(format!("favourite {on}: no answer"));
            }
            let refused = said(&r).unwrap().1;
            let shown =
                |r: &Real| r.albums > n && r.album.iter().any(|x| x.key == it.key && x.fav == on);
            if !refused && !r.pump(10.0, shown) {
                return Err(format!("favourite {on}: the album does not show it"));
            }
            let l = r.cam.favorites().wait().map_err(|e| e.to_string())?;
            let f = l.edit_file_info_list.iter().find(|e| e.file_path == flag);
            let f = f
                .and_then(|e| e.favorite_info)
                .map(|f| f.playback_is_favorite);
            println!(
                "  favourite {on} on {flag}: {}; the camera's list says {f:?} ({} favourites); notifications: {}",
                if refused {
                    "REFUSED by the camera"
                } else {
                    "the album shows it"
                },
                l.edit_file_info_list
                    .iter()
                    .filter(|e| e.favorite_info.is_some_and(|f| f.playback_is_favorite))
                    .count(),
                codes(&r.notes.lock().unwrap()[k..])
            );
            // the X5 (fw v1.13.21) answers SET_FAVORITE with status 400 in every form tried: nothing to set back
            if f.unwrap_or(false) != if refused { it.fav } else { on } {
                return Err(format!("favourite {on}: the camera says {f:?}"));
            }
            if refused {
                break;
            }
        }

        let t = unix();
        let p = r.photo()?;
        let before = named(&p).ok_or("photo name")? - t;
        println!("  clock skew before the sync (photo name): {before:+} s");

        let k = r.said.len();
        r.w.tx.send(Cc::Time).unwrap();
        let synced = |r: &Real| {
            r.said[k..]
                .iter()
                .find(|s| s.0.contains("clock") || s.0.contains("time sync"))
                .cloned()
        };
        match (r.pump(10.0, |r| synced(r).is_some()), synced(&r)) {
            (true, Some((_, false))) => {}
            (_, s) => return Err(format!("time sync: {s:?}")),
        }

        r.mode("VIDEO_NORMAL")?;
        let n = r.notes.lock().unwrap().len();
        let t = unix();
        let (v, shown) = r.record(5.0, Some(3.5))?;
        let after = named(&v).ok_or("clip name")? - t;
        let ns = r.notes.lock().unwrap()[n..].to_vec();
        println!("  clock skew after the sync (clip name): {after:+} s");
        println!("  notifications during the clip: {}", codes(&ns));
        for (at, x) in &ns {
            if let Note::CaptureStatus(_) | Note::RecordStatus(_) | Note::CaptureStopped(_) = x {
                println!(
                    "    +{:.2} s {x:?}",
                    at.duration_since(ns[0].0).as_secs_f64()
                )
            }
        }
        if after.abs() > 2 {
            return Err(format!("clock skew {after:+} s after the sync"));
        }
        let (new, other) = ours(r.files()?, &files0, &r.made);
        println!(
            "  new files: {new:?}, not the test's: {other:?} (the screen showed the clip after {shown:.2} s)"
        );
        let h = r.cam.file_extra(&v, 10).wait();
        println!(
            "  HIGHLIGHT record of the clip: {:?}",
            h.as_ref().map(|b| (
                b.len(),
                b.iter()
                    .take(48)
                    .map(|x| format!("{x:02x}"))
                    .collect::<String>()
            ))
        );
        if h.as_ref().map_or(true, |b| b.is_empty()) {
            return Err(format!("no highlight in the clip: {:?}", h.err()));
        }
        r.delete(&new)?;
        r.mode(&mode0)
    })();
    let mut bad: Vec<String> = res.err().into_iter().collect();
    if let Some((u, f)) = fav0
        && r.cam.favorites().wait().is_ok_and(|l| {
            l.edit_file_info_list.iter().any(|e| {
                e.file_path == u && e.favorite_info.is_some_and(|x| x.playback_is_favorite) != f
            })
        })
    {
        bad.push(format!("favourite of {u} not back at {f}: set back"));
        let _ = r.w.tx.send(Cc::Fav(vec![(u, f)]));
        r.pump(3.0, |_| false);
    }
    bad.extend(r.restore(&files0, &mode0));
    let (files1, snap1) = (r.files(), r.snap());
    if files1.as_ref() != Ok(&files0) {
        bad.push(format!("file list differs: {files1:?}"))
    }
    if snap1 != snap0 {
        bad.push(format!("settings differ:\n{snap1}"))
    }
    println!(
        "end: mode {:?}, {} files, settings {}",
        r.c.st.mode,
        files1.map_or(0, |v| v.len()),
        if snap1 == snap0 {
            "as at the start"
        } else {
            "CHANGED"
        }
    );
    let t = Instant::now();
    drop(r);
    println!(
        "session closed in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    assert!(bad.is_empty(), "{bad:#?}");
}

/// I-Log against the standard colour on the real X5, for the I-Log curve's measurement (`X5_CAMERA_ILOG=<out dir>`):
/// VIDEO_NORMAL at 5.7K30 with Active HDR and AdaptiveTone off (what I-Log needs), manual exposure
/// (`X5_ILOG_EXPO=<iso>,<shutter>`, default 1600,1/30) and a fixed white balance; 3 s in I-Log, then 3 s in the standard
/// colour of the same scene, no highlight marks (their HIGHLIGHT records printed against real_writes' marked clip);
/// both clips' main files downloaded to the out dir and the clips deleted through the album; every setting and the
/// mode set back.
#[test]
fn real_ilog() {
    let Some(d) = std::env::var_os("X5_CAMERA_ILOG").map(PathBuf::from) else {
        eprintln!("X5_CAMERA_ILOG not set: skipping");
        return;
    };
    let ex = std::env::var("X5_ILOG_EXPO").unwrap_or("1600,1/30".into());
    let (iso, sh) = ex.split_once(',').unwrap();
    std::fs::create_dir_all(&d).unwrap();
    let mut r = Real::open(&d);
    let ready = |r: &Real| r.c.st.conf.is_some() && r.c.st.mode.is_some() && r.albums > 0;
    assert!(r.pump(60.0, ready), "no /CONF, mode or album in 60 s");
    let (files0, snap0, mode0) = (r.files().unwrap(), r.snap(), r.c.st.mode.clone().unwrap());
    println!("start: mode {mode0}, {} files\n{snap0}", files0.len());
    let mut clips = vec![];
    let res = (|| -> Result<(), String> {
        r.mode("VIDEO_NORMAL")?;
        let ks = [
            "record_resolution",
            "hdr_switch",
            "p3_switch",
            "i_log_switch",
            "color_mode",
            "exposure_program",
            "exposure_iso",
            "exposure_shutter_speed",
            "white_balance",
            "exposure_bias",
        ];
        println!("  VIDEO_NORMAL: {:?}", ks.map(|k| (k, r.cur(k))));
        for (k, v) in [
            ("record_resolution", "5760_2880_30"),
            ("hdr_switch", "true"),
            ("p3_switch", "false"),
            ("exposure_program", "MANUAL"),
            ("exposure_iso", iso),
            ("exposure_shutter_speed", sh),
            ("white_balance", "4000"),
            ("exposure_bias", "0"),
        ] {
            r.set(k, v)?
        }
        for ilog in ["true", "false"] {
            let f0 = r.files()?;
            r.set("i_log_switch", ilog)?;
            if ilog == "false" {
                r.set("color_mode", "STANDARD")?;
            }
            r.pump(1.0, |_| false);
            let (u, _) = r.record(3.0, None)?;
            let new = ours(r.files()?, &f0, &r.made).0;
            let h = r
                .cam
                .file_extra(&u, 10)
                .wait()
                .map(|b| b.iter().map(|x| format!("{x:02x}")).collect::<String>());
            println!("  I-Log {ilog}: {new:?}; HIGHLIGHT record {h:?}");
            clips.push(new);
        }
        for u in clips.iter().flatten().filter(|u| !u.ends_with(".lrv")) {
            let (t, dst) = (Instant::now(), d.join(crate::album::name(u)));
            let n = r
                .cam
                .download(u, &dst, dl::CONNS)
                .done
                .wait()
                .map_err(|e| format!("{u}: {e}"))?;
            println!(
                "  downloaded {} ({n} B) in {:.1} s",
                dst.display(),
                t.elapsed().as_secs_f64()
            );
        }
        r.delete(&clips.concat())
    })();
    let mut bad: Vec<String> = res.err().into_iter().collect();
    bad.extend(r.restore(&files0, &mode0));
    let (files1, snap1) = (r.files(), r.snap());
    if files1.as_ref() != Ok(&files0) {
        bad.push(format!("file list differs: {files1:?}"))
    }
    if snap1 != snap0 {
        bad.push(format!("settings differ:\n{snap1}"))
    }
    println!(
        "end: mode {:?}, {} files, settings {}",
        r.c.st.mode,
        files1.map_or(0, |v| v.len()),
        if snap1 == snap0 {
            "as at the start"
        } else {
            "CHANGED"
        }
    );
    drop(r);
    assert!(bad.is_empty(), "{bad:#?}");
}
