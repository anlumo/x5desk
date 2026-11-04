//! The capture worker: the notifications its state follows, and its guards against link's simulated camera, a camera
//! with a small `/CONF`, a capture state and a battery of the test's: settings only to values `/CONF` allows, the
//! shutter's explicit start and stop, no start at 5 % battery.

use super::*;
use link::Sim;
use prost::Message;
use std::sync::Mutex;

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

/// A `/CONF` of three modes: ISO depends on the exposure program (Auto only in FULL_AUTO), EV does not.
const CONF: &str = r#"{"Sphere06C": {
    "photo_sub_mode": ["PHOTO_SINGLE"],
    "video_sub_mode": ["VIDEO_NORMAL", "VIDEO_TIMELAPSE"],
    "live_sub_mode": [],
    "PHOTO_SINGLE": {},
    "VIDEO_NORMAL": {
        "exposure_program": {"depend_on": [], "list": {"default": ["FULL_AUTO", "MANUAL"]}},
        "exposure_iso": {"depend_on": ["exposure_program"],
            "list": {"FULL_AUTO": ["0"], "MANUAL": ["100", "400", "800"]}},
        "exposure_bias": {"depend_on": [], "list": {"default": ["-0.3", "0", "0.3"]}}
    },
    "VIDEO_TIMELAPSE": {}
}}"#;

/// The simulated camera's state: its mode (a video sub-mode, `None`: PHOTO_SINGLE), exposure program, capture state
/// (a start or stop changes it), battery, and every request.
struct Fake {
    video: Option<pb::VideoSubMode>,
    prog: pb::photography_options::exposure_options::Program,
    state: pb::CameraCaptureState,
    battery: pb::BatteryInfo,
    log: Vec<(u16, Vec<u8>)>,
}

type Shared = Arc<Mutex<Fake>>;

fn fake() -> Shared {
    Arc::new(Mutex::new(Fake {
        video: Some(pb::VideoSubMode::VideoNormal),
        prog: pb::photography_options::exposure_options::Program::FullAuto,
        state: pb::CameraCaptureState::NotCapture,
        battery: pb::BatteryInfo {
            battery_level: 80,
            ..Default::default()
        },
        log: vec![],
    }))
}

fn ok(m: impl Message) -> (u16, Vec<u8>) {
    (200, m.encode_to_vec())
}

/// The fake camera, and a capture worker on its session (its first report, with `/CONF` and the mode, taken).
fn rig(f: &Shared) -> (Worker, Camera, Sim) {
    let f2 = f.clone();
    let req = move |code: u16, b: &[u8]| {
        let mut f = f2.lock().unwrap();
        f.log.push((code, b.to_vec()));
        use pb::CameraCaptureState as C;
        match code {
            code::GET_OPTIONS => ok(pb::GetOptionsResp {
                option_types: vec![],
                value: Some(pb::Options {
                    video_sub_mode: f.video.unwrap_or(pb::VideoSubMode::VideoNone) as i32,
                    focus_sensor: pb::SensorDevice::All as i32,
                    battery_info: Some(f.battery),
                    ..Default::default()
                }),
            }),
            code::CAPTURE_STATUS => ok(pb::GetCurrentCaptureStatusResp {
                status: Some(pb::CameraCaptureStatus {
                    state: f.state as i32,
                    ..Default::default()
                }),
            }),
            code::GET_PHOTO_OPTIONS => ok(pb::GetPhotographyOptionsResp {
                value: Some(pb::PhotographyOptions {
                    video_exposure: Some(pb::photography_options::ExposureOptions {
                        program: f.prog as i32,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            code::GET_FILE => ok(pb::GetFileResp {
                uri: "/CONF".into(),
                version: 71,
            }),
            code::START_CAPTURE => {
                f.state = C::NormalCapture;
                (200, vec![])
            }
            code::START_TIMELAPSE => {
                f.state = C::TimelapseCapture;
                (200, vec![])
            }
            code::STOP_CAPTURE | code::STOP_TIMELAPSE => {
                f.state = C::NotCapture;
                ok(pb::StopCaptureResp {
                    video: Some(pb::Video {
                        uri: "/DCIM/Camera01/VID_20261104_120000_00_001.insv".into(),
                        ..Default::default()
                    }),
                })
            }
            _ => (200, vec![]),
        }
    };
    let files = |u: &str| (u == "/CONF").then(|| Arc::new(CONF.as_bytes().to_vec()));
    let (c, _ev, sim) = Camera::sim(req, files, 50_000_000).unwrap();
    let w = Worker::spawn(|| {});
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    let st = next(&w);
    assert!(st.conf.is_some() && st.mode.is_some(), "{:?}", st.note);
    (w, c, sim)
}

/// The worker's next report: the one after the command sent last.
fn next(w: &Worker) -> St {
    w.rx.recv_timeout(Duration::from_secs(10))
        .expect("no report")
        .1
}

/// `c` sent, the worker's report after it, and the requests of `codes` it made meanwhile.
fn run(w: &Worker, f: &Shared, c: Cmd, codes: &[u16]) -> (St, Vec<(u16, Vec<u8>)>) {
    let n = f.lock().unwrap().log.len();
    w.tx.send(c).unwrap();
    let st = next(w);
    let l = f.lock().unwrap().log[n..].to_vec();
    (st, l.into_iter().filter(|l| codes.contains(&l.0)).collect())
}

const WRITES: [u16; 2] = [code::SET_PHOTO_OPTIONS, code::SET_OPTIONS];

/// A setting goes out only with a value its `/CONF` list allows given the values it depends on, as the screen offers
/// them: an allowed EV is written; an EV the list lacks, ISO 400 in FULL_AUTO (Auto only) and a mode the camera is
/// not in send nothing and say why; ISO 400 once the camera is in MANUAL is written.
#[test]
fn set_allowed() {
    let f = fake();
    let (w, c, _sim) = rig(&f);
    let (st, l) = run(
        &w,
        &f,
        Cmd::Set("exposure_bias".into(), "-0.3".into()),
        &WRITES,
    );
    let po = pb::PhotographyOptions {
        video_exposure: Some(pb::photography_options::ExposureOptions {
            program: 5,
            ..Default::default()
        }),
        ..Default::default()
    };
    let fm = pb::FunctionMode::NormalVideo;
    let want = |s, v| match conf::write(s, v, &po, fm, false).unwrap() {
        Write::Photo(m) => m.encode_to_vec(),
        Write::Opts(m) => m.encode_to_vec(),
    };
    assert_eq!(l, [(code::SET_PHOTO_OPTIONS, want(conf::Slot::Ev, "-0.3"))]);
    assert!(st.note.is_none(), "{:?}", st.note);
    assert_eq!(st.allowed("exposure_iso"), ["0"]);
    for (k, v) in [("exposure_bias", "-0.7"), ("exposure_iso", "400")] {
        let (st, l) = run(&w, &f, Cmd::Set(k.into(), v.into()), &WRITES);
        let (n, e) = st.note.clone().unwrap();
        println!("{k} {v}: {n}");
        assert!(l.is_empty() && e && n.contains("not allowed"), "{l:?} {n}");
    }
    // the camera switched to MANUAL (another client, its own buttons): the mode's values read again
    f.lock().unwrap().prog = pb::photography_options::exposure_options::Program::Manual;
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    assert_eq!(next(&w).allowed("exposure_iso"), ["100", "400", "800"]);
    let (_, l) = run(
        &w,
        &f,
        Cmd::Set("exposure_iso".into(), "400".into()),
        &WRITES,
    );
    let q = pb::SetPhotographyOptions::decode(&l[0].1[..]).unwrap();
    assert_eq!(q.value.unwrap().video_exposure.unwrap().iso, 400);
    // a mode without that setting (TIMELAPSE has none here)
    f.lock().unwrap().video = Some(pb::VideoSubMode::VideoTimelapse);
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    assert_eq!(next(&w).mode.as_deref(), Some("VIDEO_TIMELAPSE"));
    let (st, l) = run(
        &w,
        &f,
        Cmd::Set("exposure_bias".into(), "0".into()),
        &WRITES,
    );
    assert!(l.is_empty() && st.note.is_some_and(|n| n.1), "{l:?}");
    drop(w);
    c.close().wait().unwrap();
}

const SHUTTER: [u16; 5] = [
    code::TAKE_PICTURE,
    code::START_CAPTURE,
    code::STOP_CAPTURE,
    code::START_TIMELAPSE,
    code::STOP_TIMELAPSE,
];

/// The shutter says what it means and the worker looks at the camera's capture state first: a stop while it does not
/// record and a start while it does send nothing (a toggle would have gone the wrong way), the start and the stop
/// otherwise go out, the stop's clip noted; a photo in a video mode sends nothing; the same for a timelapse.
#[test]
fn start_stop() {
    let f = fake();
    let (w, c, _sim) = rig(&f);
    let codes = |l: Vec<(u16, Vec<u8>)>| l.into_iter().map(|x| x.0).collect::<Vec<_>>();
    // what went out, and the refusal's note (a note stays until the next one)
    let step = |cmd, want: &[u16], no: Option<&str>| {
        let (st, l) = run(&w, &f, cmd, &SHUTTER);
        let n = st.note.clone();
        println!("{:?} → {n:?}", codes(l.clone()));
        assert_eq!(codes(l), want);
        if let Some(s) = no {
            assert!(n.is_some_and(|n| n.1 && n.0.contains(s)), "{s}");
        }
        st
    };
    let (rec, stop) = (Some("captures already"), Some("nothing to stop"));
    step(Cmd::Stop, &[], stop);
    // the state follows the camera's notifications, which this one does not send: the worker asks it first
    step(Cmd::Start, &[code::START_CAPTURE], None);
    step(Cmd::Start, &[], rec);
    step(Cmd::Shoot, &[], Some("nothing taken"));
    let st = step(Cmd::Stop, &[code::STOP_CAPTURE], None);
    assert!(st.note.unwrap().0.starts_with("recorded /DCIM/"));
    // the camera started recording by itself (its button): a Start the screen sent before it knew sends nothing,
    // and the state shows the recording
    f.lock().unwrap().state = pb::CameraCaptureState::NormalCapture;
    let st = step(Cmd::Start, &[], rec);
    assert!(running(st.cap) && st.since.is_some());
    step(Cmd::Stop, &[code::STOP_CAPTURE], None);
    step(Cmd::Stop, &[], stop);
    f.lock().unwrap().video = Some(pb::VideoSubMode::VideoTimelapse);
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    assert_eq!(next(&w).mode.as_deref(), Some("VIDEO_TIMELAPSE"));
    step(Cmd::Start, &[code::START_TIMELAPSE], None);
    step(Cmd::Start, &[], rec);
    step(Cmd::Stop, &[code::STOP_TIMELAPSE], None);
    step(Cmd::Stop, &[], stop);
    // a photo mode: the shutter takes one, start and stop send nothing
    f.lock().unwrap().video = None;
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    assert_eq!(next(&w).mode.as_deref(), Some("PHOTO_SINGLE"));
    step(Cmd::Start, &[], Some("no recording or timelapse mode"));
    step(Cmd::Shoot, &[code::TAKE_PICTURE], None);
    drop(w);
    c.close().wait().unwrap();
}

/// No recording or timelapse starts at 5 % battery or less unless the camera charges (the app's rule, the screen's
/// Start disabled): the level as the battery notification reports it, as the screen has it. A stop still goes out.
#[test]
fn battery_start() {
    use pb::battery_info::PowerType::{self, *};
    let f = fake();
    let (w, c, _sim) = rig(&f);
    let bat = |l: u32, p: PowerType| {
        let b = pb::BatteryInfo {
            battery_level: l,
            power_type: p as i32,
            ..Default::default()
        };
        f.lock().unwrap().battery = b;
        w.tx.send(Cmd::Note(Note::Battery(b))).unwrap();
        assert!(next(&w).o.battery_info == Some(b));
    };
    for (l, p, ok) in [
        (5, Battery, false),
        (0, Battery, false),
        (6, Battery, true),
        (5, Adapter, true),
    ] {
        bat(l, p);
        assert_eq!(camera::empty(&f.lock().unwrap().battery), !ok);
        let (st, s) = run(&w, &f, Cmd::Start, &SHUTTER);
        println!("{l} % {p:?}: {:?}", st.note);
        assert_eq!(s.len(), ok as usize, "{l} % {p:?}");
        if ok {
            run(&w, &f, Cmd::Stop, &SHUTTER);
        } else {
            assert!(st.note.is_some_and(|n| n.1 && n.0.contains("5 %")));
        }
    }
    // a recording that runs at 5 % stops
    f.lock().unwrap().state = pb::CameraCaptureState::NormalCapture;
    bat(5, Battery);
    let (_, s) = run(&w, &f, Cmd::Stop, &SHUTTER);
    assert_eq!(s.len(), 1);
    // a timelapse is held to the same rule
    f.lock().unwrap().video = Some(pb::VideoSubMode::VideoTimelapse);
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    next(&w);
    let (_, s) = run(&w, &f, Cmd::Start, &SHUTTER);
    assert!(s.is_empty());
    drop(w);
    c.close().wait().unwrap();
}
