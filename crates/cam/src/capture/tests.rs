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
    use pb::CaptureState as C;
    let (tx, _rx) = mpsc::channel();
    let mut w = W {
        cam: None,
        sid: 0,
        st: St::default(),
        up: tx,
        wake: Box::new(|| {}),
        bad: false,
        retry: None,
    };
    let cap = |s: C, t| {
        Note::CaptureStatus(pb::CaptureNote {
            state: s as i32,
            elapsed: t,
        })
    };
    w.on(cap(C::Recording, 12));
    let s = w.st.since.unwrap().elapsed().as_secs_f64();
    assert!(running(w.st.cap) && (12.0..12.5).contains(&s), "{s}");
    w.on(Note::Battery(pb::Battery {
        level: 9,
        ..Default::default()
    }));
    assert!(camera::low(w.st.o.battery.as_ref().unwrap()));
    let t = pb::TempNote {
        level: pb::Temp::High as i32,
        celsius: 61,
    };
    w.on(Note::Temp(t));
    assert_eq!(w.st.temp, Some(t));
    w.on(cap(C::Idle, 0));
    assert!(!running(w.st.cap) && w.st.since.is_none());
    w.on(Note::CaptureStopped(pb::StoppedNote {
        cause: pb::stopped_note::Cause::BatteryLow as i32,
    }));
    let (n, e) = w.st.note.clone().unwrap();
    assert!(e && n.contains("BatteryLow"), "{n}");
    assert_eq!(w.st.notes, 5);
    // photos and the self-timer leave the files alone; timelapses run until stopped
    assert!(!running(C::TakingPhoto as i32) && running(C::TimelapseRecording as i32));
}

/// A reload after the session closed (a release racing a notification's reload) notes nothing.
#[test]
fn closed_quietly() {
    let f = fake();
    let (w, c, _sim) = rig(&f);
    let st = run(&w, &f, Cmd::Note(Note::Temp(Default::default())), &[]).0;
    assert!(st.vals.is_some());
    c.close().wait().unwrap();
    let st = run(&w, &f, Cmd::Note(Note::SubMode(Default::default())), &[]).0;
    // the reload ran (its values dropped first) and failed
    assert!(st.vals.is_none() && st.note.is_none(), "{:?}", st.note);
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
    video: Option<pb::VideoMode>,
    prog: pb::params::exposure::Program,
    state: pb::CaptureState,
    battery: pb::Battery,
    log: Vec<(u16, Vec<u8>)>,
    /// requests answered with status 500, or with bytes that do not decode
    refuse: Vec<u16>,
    junk: Vec<u16>,
}

type Shared = Arc<Mutex<Fake>>;

fn fake() -> Shared {
    Arc::new(Mutex::new(Fake {
        video: Some(pb::VideoMode::VideoNormal),
        prog: pb::params::exposure::Program::FullAuto,
        state: pb::CaptureState::Idle,
        battery: pb::Battery {
            level: 80,
            ..Default::default()
        },
        log: vec![],
        refuse: vec![],
        junk: vec![],
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
        use pb::CaptureState as C;
        if f.refuse.contains(&code) {
            return (500, vec![]);
        }
        if f.junk.contains(&code) {
            return (200, vec![0xff; 3]);
        }
        match code {
            code::GET_OPTIONS => ok(pb::ReadSettingsReply {
                keys: vec![],
                settings: Some(pb::Settings {
                    video_mode: f.video.unwrap_or(pb::VideoMode::VideoNone) as i32,
                    lens: pb::Lens::Both as i32,
                    battery: Some(f.battery),
                    ..Default::default()
                }),
            }),
            code::CAPTURE_STATUS => ok(pb::CaptureStatusReply {
                status: Some(pb::CaptureStatus {
                    state: f.state as i32,
                    ..Default::default()
                }),
            }),
            code::GET_PHOTO_OPTIONS => ok(pb::ReadParamsReply {
                params: Some(pb::Params {
                    video_exposure: Some(pb::params::Exposure {
                        program: f.prog as i32,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
            }),
            code::GET_FILE => ok(pb::LocateFileReply {
                uri: "/CONF".into(),
                version: 71,
            }),
            code::START_CAPTURE => {
                f.state = C::Recording;
                (200, vec![])
            }
            code::START_TIMELAPSE => {
                f.state = C::TimelapseRecording;
                (200, vec![])
            }
            code::TAKE_PICTURE => ok(pb::TakePhotoReply {
                photo: Some(pb::PhotoFile {
                    uri: "/DCIM/Camera01/IMG_20261104_120000_00_002.insp".into(),
                    ..Default::default()
                }),
            }),
            code::STOP_CAPTURE | code::STOP_TIMELAPSE => {
                f.state = C::Idle;
                ok(pb::StopRecordingReply {
                    file: Some(pb::VideoFile {
                        uri: "/DCIM/Camera01/VID_20261104_120000_00_001.insv".into(),
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
    let po = pb::Params {
        video_exposure: Some(pb::params::Exposure {
            program: 5,
            ..Default::default()
        }),
        ..Default::default()
    };
    let fm = pb::Profile::VideoNormal;
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
    f.lock().unwrap().prog = pb::params::exposure::Program::Manual;
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    assert_eq!(next(&w).allowed("exposure_iso"), ["100", "400", "800"]);
    let (_, l) = run(
        &w,
        &f,
        Cmd::Set("exposure_iso".into(), "400".into()),
        &WRITES,
    );
    let q = pb::WriteParams::decode(&l[0].1[..]).unwrap();
    assert_eq!(q.params.unwrap().video_exposure.unwrap().iso, 400);
    // a mode without that setting (TIMELAPSE has none here)
    f.lock().unwrap().video = Some(pb::VideoMode::VideoTimelapse);
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
    f.lock().unwrap().state = pb::CaptureState::Recording;
    let st = step(Cmd::Start, &[], rec);
    assert!(running(st.cap) && st.since.is_some());
    step(Cmd::Stop, &[code::STOP_CAPTURE], None);
    step(Cmd::Stop, &[], stop);
    f.lock().unwrap().video = Some(pb::VideoMode::VideoTimelapse);
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
    use pb::battery::Power::{self, *};
    let f = fake();
    let (w, c, _sim) = rig(&f);
    let bat = |l: u32, p: Power| {
        let b = pb::Battery {
            level: l,
            power: p as i32,
            ..Default::default()
        };
        f.lock().unwrap().battery = b;
        w.tx.send(Cmd::Note(Note::Battery(b))).unwrap();
        assert!(next(&w).o.battery == Some(b));
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
    f.lock().unwrap().state = pb::CaptureState::Recording;
    bat(5, Battery);
    let (_, s) = run(&w, &f, Cmd::Stop, &SHUTTER);
    assert_eq!(s.len(), 1);
    // a timelapse is held to the same rule
    f.lock().unwrap().video = Some(pb::VideoMode::VideoTimelapse);
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    next(&w);
    let (_, s) = run(&w, &f, Cmd::Start, &SHUTTER);
    assert!(s.is_empty());
    drop(w);
    c.close().wait().unwrap();
}

/// An asked command's outcome comes in the report after it under its number, kept across sessions: a write sent, a
/// value refused, a mode, lens or sub-mode other than the one checked refused, a shutter refused or its photo or clip
/// named, a start; a mode or setting change while the camera captures sends nothing; a start sends nothing when the
/// capture state cannot be read, and one the camera refuses did not begin; an answer that does not decode a failure.
/// No session: nothing sent. The mode's values loaded name their mode.
#[test]
fn asked() {
    let f = fake();
    let (w, c, _sim) = rig(&f);
    let seen = |m: &str| {
        Some(Seen {
            mode: m.into(),
            both: true,
            known: true,
        })
    };
    let ask = |n, c: Cmd, s: Option<Seen>| {
        let (st, l) = run(
            &w,
            &f,
            Cmd::Ask {
                n,
                c: Box::new(c),
                seen: s,
            },
            &[WRITES.as_slice(), &SHUTTER, &[code::SET_OPTIONS]].concat(),
        );
        let (k, o) = st.out.clone().expect("an outcome");
        assert_eq!(k, n);
        println!("{n}: {o:?} {:?}", l.iter().map(|x| x.0).collect::<Vec<_>>());
        (st, o, l.len())
    };
    let no = |o: &Out, w: &str| matches!(o, Out::No(t) if t.contains(w));
    let (st, ..) = ask(0, Cmd::Mark, seen("PHOTO_SINGLE"));
    assert_eq!(st.vals.as_deref(), Some("VIDEO_NORMAL"));
    let set = || Cmd::Set("exposure_bias".into(), "0.3".into());
    assert_eq!(ask(1, set(), seen("VIDEO_NORMAL")).1, Out::Sent(None));
    let (_, o, n) = ask(2, Cmd::Set("exposure_bias".into(), "9".into()), None);
    assert!(no(&o, "not allowed") && n == 0, "{o:?}");
    let (_, o, n) = ask(3, set(), seen("PHOTO_SINGLE"));
    assert!(no(&o, "in VIDEO_NORMAL now") && n == 0, "{o:?}");
    let lens = Seen {
        both: false,
        ..seen("VIDEO_NORMAL").unwrap()
    };
    let (_, o, n) = ask(4, set(), Some(lens));
    assert!(no(&o, "lens or sub-mode") && n == 0, "{o:?}");
    let (_, o, n) = ask(5, Cmd::Shoot, None);
    assert!(no(&o, "nothing taken") && n == 0, "{o:?}");
    let (_, o, n) = ask(6, Cmd::Mode("VIDEO_NOSUCH".into()), None);
    assert!(matches!(&o, Out::No(_)) && n == 0, "{o:?}");
    // a start: its capture state unreadable, or refused by the camera: nothing began
    f.lock().unwrap().refuse = vec![code::CAPTURE_STATUS];
    let (_, o, n) = ask(7, Cmd::Start, None);
    assert!(no(&o, "nothing sent") && n == 0, "{o:?}");
    f.lock().unwrap().refuse = vec![code::START_CAPTURE];
    let (st, o, _) = ask(8, Cmd::Start, None);
    assert!(no(&o, "refused the start") && !running(st.cap), "{o:?}");
    f.lock().unwrap().refuse.clear();
    assert_eq!(ask(9, Cmd::Start, None).1, Out::Sent(None));
    // capturing: no mode or setting change goes out, the camera's state read first
    for (k, c) in [(10, Cmd::Mode("VIDEO_TIMELAPSE".into())), (11, set())] {
        let (st, o, n) = ask(k, c, None);
        assert!(no(&o, "captures") && n == 0, "{o:?}");
        assert!(running(st.cap));
    }
    let (_, o, _) = ask(12, Cmd::Stop, None);
    assert!(
        matches!(&o, Out::Sent(Some(u)) if u.ends_with("_00_001.insv")),
        "{o:?}"
    );
    // kept across a session
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    assert_eq!(next(&w).out.map(|x| x.0), Some(12));
    // a photo mode: its photo named; an answer that does not decode a failure
    f.lock().unwrap().video = None;
    w.tx.send(Cmd::Cam(1, Some(c.clone()))).unwrap();
    next(&w);
    let (_, o, _) = ask(13, Cmd::Shoot, seen("PHOTO_SINGLE"));
    let u = "/DCIM/Camera01/IMG_20261104_120000_00_002.insp";
    assert_eq!(o, Out::Sent(Some(u.into())));
    f.lock().unwrap().junk = vec![code::TAKE_PICTURE];
    assert!(matches!(ask(14, Cmd::Shoot, None).1, Out::Failed(_)));
    drop(w);
    c.close().wait().unwrap();
    // no session
    let w = Worker::spawn(|| {});
    let c = Cmd::Ask {
        n: 15,
        c: Box::new(Cmd::Start),
        seen: None,
    };
    w.tx.send(c).unwrap();
    assert_eq!(
        next(&w).out,
        Some((15, Out::No("no camera session".into())))
    );
}
