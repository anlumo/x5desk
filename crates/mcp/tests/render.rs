//! The GPU tools over an in-memory pipe, on a fake backend (no GPU, no samples: every OS). [`Fake`] stands in for the
//! app's engine: its catalog, a check of small text recordings in an `--allow-dir`, a start that can fail, take a
//! while or panic, two lanes that write into the handed `.part` and end through `media::part::finish` as
//! `export::run_into` does, runs that can hang ignoring their stop. What is tested is mcp's: the tools and their
//! refusals, the records, cancels, names, the host's end, the next host's crash cleanup, timed recordings against
//! renders, the job tools by kind. The watchdog needs its `cfg(test)` times: `server::render`'s unit tests have it.

use mcp::{
    Host, Opts, Tiers,
    config::Source,
    render::{
        Adjust, Backend, Cap, Catalog, Codec, Found, GpuInfo, GpuSt, Kind, Lane, Later, Named,
        Output, Phase, Plan, Preset, Projection, Range, Render, Req, Run, RunSt, Start, Still,
        Track, Want, no_blur,
    },
    renders::{ENDED, End, GONE, RJob, Renders},
    sim,
};
use media::part::{CANCELLED, Id, Part};
use rmcp::{
    ClientHandler, RoleClient, ServiceExt,
    model::{CallToolRequestParams, CallToolResult, ProgressNotificationParam},
    service::{NotificationContext, RunningService},
};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
        mpsc,
    },
    thread::sleep,
    time::{Duration, Instant, SystemTime},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The recording most tests render: 300 frames at 30 fps, with IMU data.
const REC: &str = "VID_20260117_141148_00_635.insv";

/// The recordings in every test's `--allow-dir`: the fake's text (`frames=`, `imu=`, `trailer=`, `bytes=`: the
/// estimate), one in a file per lens, a proxy alone, an `.mp4` and a photo.
const RECS: [(&str, &str); 8] = [
    (REC, "frames=300 imu=1 trailer=1"),
    ("short.insv", "frames=30 imu=0 trailer=1"),
    ("bare.insv", "frames=300 imu=1 trailer=0"),
    (LENS, "frames=300 imu=1 trailer=1"),
    (
        "VID_20260117_141148_10_641.insv",
        "frames=300 imu=1 trailer=1",
    ),
    ("LRV_20260117_141148_01_636.lrv", "a proxy"),
    ("x.mp4", "an export"),
    ("IMG_20260117_141148_00_637.insp", "a photo"),
];

/// A recording in a file per lens: its `_00_`, beside its `_10_`.
const LENS: &str = "VID_20260117_141148_00_641.insv";

/// The GPU tools.
const GPU: [&str; 3] = ["catalog", "render_still", "export"];

/// What the fake does, set by each test.
#[derive(Default)]
struct Knobs {
    /// start: how long it takes (a halt ends it), its failure and when it may be tried again, a panic
    start_ms: u64,
    start_fails: Option<(String, Option<Duration>)>,
    start_panics: bool,
    /// check: held while set, refusing with what it was asked (`echo`), a panic
    hold: bool,
    echo: bool,
    check_panics: bool,
    catalog_panics: bool,
    gpu_panics: bool,
    /// submit: how long it takes, its refusal, a panic
    submit_ms: u64,
    submit_err: Option<String>,
    submit_panics: bool,
    /// runs: ms a frame (a searched one too); the next run's stuck stage, its stop ignored (after that many frames: 0
    /// in its setup; after that many searched in its blur pass)
    frame_ms: u64,
    stuck: Option<u64>,
    stuck_search: Option<u64>,
    state_panics: bool,
    cancel_panics: bool,
    /// the GPU up without burn: why (no RAFT, no blur)
    no_burn: Option<String>,
}

/// The app's engine as mcp sees it, without a GPU.
struct Fake {
    k: Arc<Mutex<Knobs>>,
    st: Mutex<GpuSt>,
    starts: AtomicUsize,
    checks: AtomicUsize,
    /// what the last check was asked
    asked: Mutex<Option<Want>>,
    runs: Mutex<Vec<Arc<FRun>>>,
    wedged: Mutex<Vec<(Lane, String)>>,
    /// a thread a lane, as `export::Queue`'s
    lanes: [mpsc::Sender<Arc<FRun>>; 2],
}

/// A run of the fake's: queued on its lane, then its blur pass, setup, frames into its `.part` and the finish.
struct FRun {
    kind: Kind,
    lane: Lane,
    total: u64,
    /// the plan's: what its pass blurs, the source frames it searches
    blur: Vec<String>,
    search: u64,
    frame_ms: u64,
    dst: PathBuf,
    part: Mutex<Option<Part>>,
    stop: AtomicBool,
    st: Mutex<RunSt>,
    stuck: Option<u64>,
    stuck_search: Option<u64>,
    /// a stuck run waits while set (in the driver: its stop not looked at)
    held: AtomicBool,
    started: AtomicBool,
    /// when it was submitted, and when its first cancel came
    made: Instant,
    stopped: Mutex<Option<Instant>>,
    k: Arc<Mutex<Knobs>>,
}

impl Fake {
    fn new() -> Arc<Fake> {
        let lane = |n: &str| {
            let (tx, rx) = mpsc::channel::<Arc<FRun>>();
            let t = std::thread::Builder::new().name(n.into());
            t.spawn(move || rx.into_iter().for_each(|r| work(&r)))
                .unwrap();
            tx
        };
        Arc::new(Fake {
            k: Arc::new(Mutex::new(Knobs {
                frame_ms: 5,
                ..Default::default()
            })),
            st: Mutex::new(GpuSt::Off),
            starts: AtomicUsize::new(0),
            checks: AtomicUsize::new(0),
            asked: Mutex::default(),
            runs: Mutex::default(),
            wedged: Mutex::default(),
            lanes: [lane("fake-still"), lane("fake-export")],
        })
    }

    fn set(&self, f: impl FnOnce(&mut Knobs)) {
        f(&mut self.k.lock().unwrap())
    }

    /// The GPU up, as a start leaves it.
    fn up(&self) {
        *self.st.lock().unwrap() = GpuSt::Ready(self.info())
    }

    /// The device as it starts: burn attached unless `no_burn`.
    fn info(&self) -> GpuInfo {
        GpuInfo {
            raft: self.k.lock().unwrap().no_burn.clone(),
            ..info()
        }
    }

    fn gpu_is(&self, st: GpuSt) {
        *self.st.lock().unwrap() = st
    }

    /// Its `i`th run.
    fn run(&self, i: usize) -> Arc<FRun> {
        self.runs.lock().unwrap()[i].clone()
    }

    fn starts(&self) -> usize {
        self.starts.load(Relaxed)
    }

    fn asked(&self) -> Want {
        self.asked.lock().unwrap().clone().unwrap()
    }
}

impl FRun {
    /// Lets a stuck run go on.
    fn go(&self) {
        self.held.store(false, Relaxed)
    }

    fn now(&self) -> RunSt {
        self.st.lock().unwrap().clone()
    }

    fn ended(&self) -> bool {
        matches!(
            self.now(),
            RunSt::Done { .. } | RunSt::Failed(_) | RunSt::Cancelled
        )
    }
}

impl Run for FRun {
    fn state(&self) -> RunSt {
        let p = self.k.lock().unwrap().state_panics;
        if p {
            panic!("the fake's state broke")
        }
        self.now()
    }

    fn cancel(&self) {
        let p = self.k.lock().unwrap().cancel_panics;
        if p {
            panic!("the fake's cancel broke")
        }
        self.stop.store(true, Relaxed);
        self.stopped
            .lock()
            .unwrap()
            .get_or_insert_with(Instant::now);
        let mut st = self.st.lock().unwrap();
        // queued: cancelled at once, never started; its `.part` closed (mcp settles it)
        if *st == RunSt::Queued {
            *st = RunSt::Cancelled;
            drop(self.part.lock().unwrap().take())
        }
    }
}

/// A reading past the blur pass, if any: `searched` its frames searched (they stay counted, as the app's).
fn running(phase: Phase, done: u64, total: u64, secs: f64, searched: u64) -> RunSt {
    let fps = done as f64 / secs.max(1e-3);
    RunSt::Running {
        phase,
        done,
        total,
        fps,
        eta: (total - done) as f64 / fps.max(1e-3),
        concealed: 0,
        searched,
    }
}

/// A 3200 × 1600 JPEG, a still's file.
fn jpeg() -> Arc<[u8]> {
    static J: OnceLock<Arc<[u8]>> = OnceLock::new();
    J.get_or_init(|| {
        let px: Vec<u8> = (0..3200 * 1600)
            .flat_map(|i| [(i % 3200 / 13) as u8, (i / 3200 / 7) as u8, 99, 255])
            .collect();
        media::jpeg::encode(&px, 3200, 1600, 92, None)
            .unwrap()
            .into()
    })
    .clone()
}

/// A blur pass's reading: `k` of the run's frames to search done, the first at `first`; its rate (and the seconds
/// left) once two are, as the app's.
fn searching(r: &FRun, k: u64, first: Option<Instant>) -> RunSt {
    let fps = match first {
        Some(t) if k >= 2 => (k - 1) as f64 / t.elapsed().as_secs_f64().max(1e-3),
        _ => 0.0,
    };
    RunSt::Running {
        phase: Phase::Detecting,
        done: 0,
        total: r.total,
        fps,
        eta: if fps > 0.0 {
            (r.search - k) as f64 / fps
        } else {
            0.0
        },
        concealed: 0,
        searched: k,
    }
}

/// The fake's blur pass (the app's `Job::pre`): a frame searched every `frame_ms`, its stop read before each, held
/// after `stuck_search` of them (`hold`); a face found on each, a plate on every other, one track of each kind asked.
fn pass(r: &FRun, hold: impl Fn()) -> Result<Found, String> {
    let (t0, mut first) = (Instant::now(), None);
    for k in 1..=r.search {
        if r.stop.load(Relaxed) {
            return Err(CANCELLED.into());
        }
        sleep(Duration::from_millis(r.frame_ms));
        let t = *first.get_or_insert_with(Instant::now);
        *r.st.lock().unwrap() = searching(r, k, Some(t));
        if r.stuck_search == Some(k) {
            hold()
        }
    }
    let has = |k: &str| r.blur.iter().any(|b| b == k);
    let faces = if has("faces") { r.search } else { 0 };
    let plates = if has("plates") {
        r.search.div_ceil(2)
    } else {
        0
    };
    // the fake's source rate is its output's
    Ok(Found {
        faces,
        plates,
        face_tracks: (faces > 0) as u64,
        plate_tracks: (plates > 0) as u64,
        searched: r.search,
        blurred: r.total,
        frames: r.total,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// A lane's job: its blur pass when the plan has one, its setup, then a frame every `frame_ms` into the handed
/// `.part` (a still's JPEG at once, an export 1000 bytes a frame), its stop read before each; the end through
/// `media::part::finish`, as `run_into`'s.
fn work(r: &FRun) {
    let part = {
        let mut st = r.st.lock().unwrap();
        if *st != RunSt::Queued {
            return;
        }
        // running before the pass, as the app's lane
        *st = match r.search {
            0 => running(Phase::Setup, 0, r.total, 0.0, 0),
            _ => searching(r, 0, None),
        };
        r.started.store(true, Relaxed);
        r.part.lock().unwrap().take()
    };
    let Some(Part { f, .. }) = part else { return };
    // the run's clock, its pass in it as in the app's
    let t0 = Instant::now();
    let hold = || {
        while r.held.load(Relaxed) {
            sleep(Duration::from_millis(5))
        }
    };
    let go = || -> Result<Option<Found>, String> {
        let found = (r.search > 0).then(|| pass(r, hold)).transpose()?;
        let t1 = Instant::now();
        *r.st.lock().unwrap() = running(Phase::Setup, 0, r.total, 0.0, r.search);
        if r.stuck == Some(0) {
            hold()
        }
        for i in 1..=r.total {
            if r.stop.load(Relaxed) {
                return Err(CANCELLED.into());
            }
            let j = jpeg();
            let b: &[u8] = match r.kind {
                Kind::Still => &j[..],
                Kind::Export => &[b'x'; 1000],
            };
            (&f).write_all(b).map_err(|e| e.to_string())?;
            sleep(Duration::from_millis(r.frame_ms));
            let s = t1.elapsed().as_secs_f64();
            *r.st.lock().unwrap() = running(Phase::Frames, i, r.total, s, r.search);
            if r.stuck == Some(i) {
                hold()
            }
        }
        let s = t1.elapsed().as_secs_f64();
        *r.st.lock().unwrap() = running(Phase::Finishing, r.total, r.total, s, r.search);
        Ok(found)
    };
    let (res, found) = match go() {
        Ok(found) => (Ok(()), found),
        Err(e) => (Err(e), None),
    };
    let end = match media::part::finish(&media::part::of(&r.dst), &r.dst, &f, &r.stop, res) {
        Ok(()) => RunSt::Done {
            frames: r.total,
            secs: t0.elapsed().as_secs_f64(),
            bytes: fs::metadata(&r.dst).map_or(0, |m| m.len()),
            concealed: 0,
            found,
        },
        Err(e) if e == CANCELLED => RunSt::Cancelled,
        Err(e) => RunSt::Failed(e),
    };
    drop(f);
    *r.st.lock().unwrap() = end
}

/// Two presets (`p360` 64 wide, `flat` 32 × 16), two codecs, one look, off and horizon, perspective, the app's blur
/// targets; Shot Lab and LUTs later.
fn cat() -> Catalog {
    let r = |min, max, d| Range {
        min,
        max,
        default: Some(d),
    };
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let preset = |id: &str, output, width, height, codec: &str| Preset {
        id: id.into(),
        name: id.into(),
        output,
        width,
        height,
        codec: codec.into(),
        ten_bit: false,
        mbps: 60.0,
    };
    let later = |id: &str, aliases: &[&str], effects: &[&str]| Later {
        id: id.into(),
        name: None,
        aliases: s(aliases),
        effects: s(effects),
    };
    Catalog {
        presets: vec![
            preset("p360", Output::Pano, 64, 32, "h265"),
            preset("flat", Output::Flat, 32, 16, "h264"),
        ],
        default_preset: "p360".into(),
        pano_widths: vec![64, 128],
        flat_sizes: vec![[32, 16], [64, 32]],
        codecs: vec![
            Codec {
                id: "h265".into(),
                ten_bit: true,
                max_px: 8192,
            },
            Codec {
                id: "h264".into(),
                ten_bit: false,
                max_px: 4096,
            },
        ],
        mbps: r(1.0, 400.0, 60.0),
        qp: r(10.0, 45.0, 24.0),
        fps: s(&["source", "24", "25", "29.97", "30", "60"]),
        raft: true,
        colour_match: true,
        stabilisation: s(&["off", "horizon"]),
        smoothing_s: r(0.0, 2.0, 0.5),
        looks: vec![Named {
            id: "teal_orange".into(),
            name: Some("Teal & Orange".into()),
            aliases: s(&["teal"]),
        }],
        look_strength: r(0.0, 1.0, 1.0),
        adjust: vec![Adjust {
            id: "exposure".into(),
            label: "Exposure (EV)".into(),
            min: -3.0,
            max: 3.0,
            default: 0.0,
        }],
        projections: vec![Projection {
            id: "perspective".into(),
            aliases: s(&["persp"]),
            fov: [20.0, 160.0],
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            default_fov: 100.0,
        }],
        still: Still {
            pano_width: 3200,
            flat: [1920, 1080],
            jpeg_quality: r(50.0, 100.0, 92.0),
        },
        blur: vec![
            Named {
                id: "faces".into(),
                name: Some("Faces".into()),
                aliases: s(&["face"]),
            },
            Named {
                id: "plates".into(),
                name: Some("Licence plates".into()),
                aliases: s(&["plate", "licence_plates", "license_plates", "number_plates"]),
            },
        ],
        blur_every: 3,
        later: vec![
            later("shot_lab", &["effect"], &["trails", "stars"]),
            later("lut", &["luts"], &[]),
        ],
    }
}

/// A GPU that decodes the X5's H.265 and encodes H.265 (10-bit too) and H.264, RAFT there.
fn info() -> GpuInfo {
    let cap = |c: &str, ten_bit, m| Cap {
        codec: c.into(),
        ten_bit,
        min: [16, 16],
        max: [m, m],
        rate: vec!["cbr".into(), "cqp".into()],
        max_mbps: Some(400.0),
        qp: Some([10, 45]),
    };
    GpuInfo {
        adapter: "Fake GPU".into(),
        backend: "fake".into(),
        decode: vec![cap("h265", false, 8192), cap("h265", true, 8192)],
        encode: vec![
            cap("h265", false, 8192),
            cap("h265", true, 8192),
            cap("h264", false, 4096),
        ],
        max_texture: 16384,
        ..Default::default()
    }
}

/// The fake's check of a text recording: its frames at 30 fps, its IMU data and trailer; the still's time and the
/// export's range within the clip; the defaults of the catalog; the estimate (`bytes=`, else 1000 a frame); a blur
/// pass as the app's (a still's one frame, an export's every 3rd; no file per lens), its name marked.
fn plan(r: &Req) -> Result<Plan, String> {
    let f = &r.files[0];
    let s = fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
    let kv = |k: &str| {
        (s.split_whitespace())
            .find_map(|w| w.strip_prefix(k)?.strip_prefix('=')?.parse::<f64>().ok())
    };
    let name = f.file_name().unwrap().to_string_lossy();
    if kv("trailer") == Some(0.0) {
        return Err(format!("{name}: no Insta360 trailer"));
    }
    let (dur, imu) = (kv("frames").unwrap_or(300.0) / 30.0, kv("imu") == Some(1.0));
    let (w, c) = (&r.want, cat());
    let stab =
        (w.stabilisation.clone()).unwrap_or_else(|| if imu { "horizon" } else { "off" }.into());
    if stab != "off" && !imu {
        return Err(format!("{name}: no IMU data, so no {stab} stabilisation"));
    }
    let blur = w.blur.clone();
    if !blur.is_empty() && r.files.len() > 1 {
        return Err(
            "blur: a recording in a file per lens (_00_ + _10_) is not supported yet; leave out blur".into(),
        );
    }
    let pre = if blur.is_empty() { "" } else { "blur_" };
    let stem = f.file_stem().unwrap().to_string_lossy();
    let raft = w.raft.unwrap_or(c.raft);
    let tracks = vec![Track {
        codec: "h265".into(),
        ten_bit: false,
        size: [3840, 3840],
    }];
    if r.kind == Kind::Still {
        let t = w.time_s.unwrap_or(0.0);
        if t >= dur {
            return Err(format!("time_s {t}: outside the clip (0 to {dur:.2} s)"));
        }
        let out = w.output.unwrap_or(Output::Pano);
        let size = match (w.width.zip(w.height), out) {
            (Some((x, y)), _) => [x, y],
            (None, Output::Pano) => [c.still.pano_width, c.still.pano_width / 2],
            (None, Output::Flat) => c.still.flat,
        };
        let tag = match out {
            Output::Pano => format!("360_{}", size[0]),
            Output::Flat => format!("flat_{}x{}", size[0], size[1]),
        };
        return Ok(Plan {
            frames: 1,
            secs: 1.0 / 30.0,
            size,
            codec: None,
            ten_bit: false,
            raft,
            search: (!blur.is_empty()) as u64,
            blur: blur.clone(),
            mbps: None,
            qp: None,
            tracks,
            stem: format!("{stem}_{pre}{t:.2}s_{tag}"),
            ext: "jpg",
            bytes: size[0] as u64 * size[1] as u64 / 2,
            settings: json!({
                "output": out, "width": size[0], "height": size[1], "time_s": t, "raft": raft,
                "stabilisation": stab, "look": w.look, "blur": blur,
            }),
        });
    }
    let p = (c.presets.iter())
        .find(|p| Some(&p.id) == w.preset.as_ref())
        .unwrap_or(&c.presets[0]);
    let (a, b) = (w.start_s.unwrap_or(0.0), w.end_s.unwrap_or(dur));
    if a >= dur || b > dur + 1e-9 {
        return Err(format!("{a}-{b} s: outside the clip (0 to {dur:.2} s)"));
    }
    let fps = match w.fps.as_deref() {
        None | Some("source") => 30.0,
        Some(l) => l.parse().unwrap(),
    };
    let frames = (((b - a) * fps).round() as u64).max(1);
    // the source's frames of the span, every 3rd searched
    let search = match blur.is_empty() {
        true => 0,
        false => (((b - a) * 30.0).round() as u64).max(1).div_ceil(3),
    };
    let out = w.output.unwrap_or(p.output);
    let size = match (w.width.zip(w.height), out) {
        (Some((x, y)), _) => [x, y],
        (None, o) if o == p.output => [p.width, p.height],
        (None, Output::Pano) => [64, 32],
        (None, Output::Flat) => [32, 16],
    };
    let codec = w.codec.clone().unwrap_or(p.codec.clone());
    let ten_bit = w.ten_bit.unwrap_or(p.ten_bit);
    Ok(Plan {
        frames,
        secs: frames as f64 / fps,
        size,
        codec: Some(codec.clone()),
        ten_bit,
        raft,
        blur: blur.clone(),
        search,
        mbps: w.mbps.or(w.qp.is_none().then_some(p.mbps)),
        qp: w.qp,
        tracks,
        stem: format!("{stem}_{pre}{}", p.id),
        ext: "mp4",
        bytes: kv("bytes").map_or(frames * 1000, |b| b as u64),
        settings: json!({
            "preset": p.id, "output": out, "width": size[0], "height": size[1], "codec": codec, "ten_bit": ten_bit,
            "raft": raft, "fps": fps, "start_s": a, "end_s": b, "stabilisation": stab, "look": w.look, "blur": blur,
        }),
    })
}

impl Render for Fake {
    fn catalog(&self) -> Catalog {
        let p = self.k.lock().unwrap().catalog_panics;
        if p {
            panic!("the fake's catalog broke")
        }
        cat()
    }

    fn check(&self, r: &Req) -> Result<Plan, String> {
        self.checks.fetch_add(1, Relaxed);
        while self.k.lock().unwrap().hold {
            sleep(Duration::from_millis(5))
        }
        let (echo, p) = {
            let k = self.k.lock().unwrap();
            (k.echo, k.check_panics)
        };
        if p {
            panic!("the fake's check broke")
        }
        *self.asked.lock().unwrap() = Some(r.want.clone());
        if echo {
            let w = &r.want;
            let f: Vec<String> = r.files.iter().map(|f| f.display().to_string()).collect();
            return Err(format!(
                "the fake refuses {f:?}: look {:?}, adjust {:?}, fps {:?}, codec {:?}",
                w.look,
                w.adjust.keys().collect::<Vec<_>>(),
                w.fps,
                w.codec
            ));
        }
        plan(r)
    }

    fn start(&self, s: &Start) -> Result<GpuInfo, String> {
        self.starts.fetch_add(1, Relaxed);
        let (ms, fails, p) = {
            let k = self.k.lock().unwrap();
            (k.start_ms, k.start_fails.clone(), k.start_panics)
        };
        self.gpu_is(GpuSt::Starting);
        let t = Instant::now();
        while t.elapsed() < Duration::from_millis(ms) {
            if s.halt.load(Relaxed) {
                self.gpu_is(GpuSt::Off);
                return Err("halted".into());
            }
            sleep(Duration::from_millis(10))
        }
        if p {
            self.gpu_is(GpuSt::Off);
            panic!("the fake's start broke")
        }
        (s.log)("the fake GPU started".into());
        match fails {
            Some((why, retry_in)) => {
                self.gpu_is(GpuSt::Failed {
                    why: why.clone(),
                    step: Some(1),
                    retry_in,
                });
                Err(why)
            }
            None => {
                self.up();
                Ok(self.info())
            }
        }
    }

    fn gpu(&self) -> GpuSt {
        let p = self.k.lock().unwrap().gpu_panics;
        if p {
            panic!("the fake's gpu broke")
        }
        self.st.lock().unwrap().clone()
    }

    fn selftest(&self) -> Result<GpuInfo, (u8, String)> {
        Ok(self.info())
    }

    fn probe(&self) -> Result<(), String> {
        Ok(())
    }

    fn submit(
        &self,
        r: Req,
        p: &Plan,
        dst: &Path,
        part: Part,
        lane: Lane,
    ) -> Result<Arc<dyn Run>, String> {
        let (ms, e, panics, frame_ms, stuck, stuck_search) = {
            let mut k = self.k.lock().unwrap();
            let e = k.submit_err.clone();
            let (s, ss) = (k.stuck.take(), k.stuck_search.take());
            (k.submit_ms, e, k.submit_panics, k.frame_ms, s, ss)
        };
        sleep(Duration::from_millis(ms));
        if panics {
            panic!("the fake's submit broke")
        }
        if let Some(e) = e {
            return Err(e);
        }
        if !matches!(*self.st.lock().unwrap(), GpuSt::Ready(_)) {
            return Err("the fake GPU is not up".into());
        }
        if let Some((_, w)) = self.wedged.lock().unwrap().iter().find(|w| w.0 == lane) {
            return Err(w.clone());
        }
        let run = Arc::new(FRun {
            kind: r.kind,
            lane,
            total: p.frames,
            blur: p.blur.clone(),
            search: p.search,
            frame_ms,
            dst: dst.into(),
            part: Mutex::new(Some(part)),
            stop: AtomicBool::new(false),
            st: Mutex::new(RunSt::Queued),
            stuck,
            stuck_search,
            held: AtomicBool::new(stuck.is_some() || stuck_search.is_some()),
            started: AtomicBool::new(false),
            made: Instant::now(),
            stopped: Mutex::default(),
            k: self.k.clone(),
        });
        self.runs.lock().unwrap().push(run.clone());
        let i = (lane == Lane::Export) as usize;
        (self.lanes[i].send(run.clone())).map_err(|_| "the fake's lane is gone".to_string())?;
        Ok(run)
    }

    fn wedge(&self, lane: Lane, why: &str) {
        self.wedged.lock().unwrap().push((lane, why.into()));
        for r in self.runs.lock().unwrap().iter().filter(|r| r.lane == lane) {
            let mut st = r.st.lock().unwrap();
            if *st == RunSt::Queued {
                *st = RunSt::Failed(why.into())
            }
        }
    }

    /// Never called here: the poller trims after 5 idle minutes (`server::render`'s unit tests, with a test time).
    fn trim(&self) {}
}

/// Progress, total and message of each progress notification.
type Progs = Arc<Mutex<Vec<(f64, Option<f64>, Option<String>)>>>;

#[derive(Clone, Default)]
struct H {
    prog: Progs,
}

impl ClientHandler for H {
    async fn on_progress(&self, p: ProgressNotificationParam, _: NotificationContext<RoleClient>) {
        self.prog
            .lock()
            .unwrap()
            .push((p.progress, p.total, p.message))
    }
}

struct Cli {
    s: RunningService<RoleClient, H>,
    h: H,
    /// every byte the server wrote to this client
    got: Arc<Mutex<Vec<u8>>>,
}

/// A test's dir under the temp dir: `rec` (an `--allow-dir`) with [`RECS`] in it.
fn dirs(n: &str) -> (PathBuf, PathBuf) {
    let d = std::env::temp_dir().join(format!("x5desk-mcp-render-{n}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    let rec = d.join("rec");
    fs::create_dir_all(&rec).unwrap();
    for (f, s) in RECS {
        fs::write(rec.join(f), s).unwrap()
    }
    (d, fs::canonicalize(rec).unwrap())
}

/// A `--sim` host on fake `f` with tiers `tiers` and `rec` as an `--allow-dir`, its files kept in `keep` (for the next
/// host) or in a temp dir of its own: never the user's config or folders.
async fn host(f: &Arc<Fake>, rec: &Path, tiers: &str, keep: Option<&Path>) -> Arc<Host> {
    let h = Host::new(Opts {
        sim: Some(sim::Opts {
            rate: 50_000_000,
            size: 200_000,
            dir: keep.map(Path::to_path_buf),
        }),
        tiers: Some(Source::Fixed(Tiers::parse(tiers).unwrap())),
        allow: vec![rec.to_path_buf()],
        backend: Backend {
            render: Some(f.clone()),
        },
        ..Default::default()
    })
    .await
    .unwrap();
    assert!(
        h.agent().starts_with(std::env::temp_dir()),
        "{}",
        h.agent().display()
    );
    h
}

/// A client on `h`, narrowed by `narrow` (its own `--tiers`), the server's bytes tapped on their way.
async fn connect(h: &Arc<Host>, narrow: Option<Tiers>) -> Cli {
    let (up_c, up_s) = tokio::io::duplex(1 << 16);
    let (down_s, mut down_t) = tokio::io::duplex(1 << 16);
    let (mut cli_t, cli_c) = tokio::io::duplex(1 << 16);
    let got = Arc::new(Mutex::new(vec![]));
    let g = got.clone();
    tokio::spawn(async move {
        let mut b = vec![0; 1 << 16];
        while let Ok(n) = down_t.read(&mut b).await
            && n > 0
        {
            g.lock().unwrap().extend_from_slice(&b[..n]);
            if cli_t.write_all(&b[..n]).await.is_err() {
                break;
            }
        }
    });
    let hh = h.clone();
    tokio::spawn(async move { hh.serve(up_s, down_s, "test", narrow).await });
    let c = H::default();
    let s = c.clone().serve((cli_c, up_c)).await.unwrap();
    Cli { s, h: c, got }
}

fn params(n: &str, a: Value) -> CallToolRequestParams {
    CallToolRequestParams::new(n.to_string()).with_arguments(a.as_object().unwrap().clone())
}

async fn call(c: &Cli, n: &str, a: Value) -> CallToolResult {
    (c.s.call_tool(params(n, a)).await).unwrap_or_else(|e| panic!("{n}: {e}"))
}

fn text(r: &CallToolResult) -> String {
    (r.content.iter())
        .filter_map(|c| c.as_text())
        .map(|t| t.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A result's JSON: structured, or a still's first text block.
fn of(r: &CallToolResult) -> Value {
    let t = || r.content.first().and_then(|c| c.as_text()).unwrap();
    (r.structured_content.clone()).unwrap_or_else(|| serde_json::from_str(&t().text).unwrap())
}

async fn ok(c: &Cli, n: &str, a: Value) -> Value {
    let r = call(c, n, a).await;
    assert_eq!(r.is_error, Some(false), "{n}: {}", text(&r));
    of(&r)
}

async fn err(c: &Cli, n: &str, a: Value) -> String {
    let r = call(c, n, a).await;
    assert_eq!(r.is_error, Some(true), "{n} should fail: {}", text(&r));
    text(&r)
}

async fn names(c: &Cli) -> Vec<String> {
    let mut v: Vec<String> = (c.s.list_all_tools().await.unwrap().into_iter())
        .map(|t| t.name.to_string())
        .collect();
    v.sort();
    v
}

async fn until(secs: f64, mut f: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs_f64(secs);
    while Instant::now() < end {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await
    }
    f()
}

/// Job `job`'s state once `f` holds (up to 20 s).
async fn when(c: &Cli, job: &Value, f: impl Fn(&Value) -> bool) -> Value {
    let end = Instant::now() + Duration::from_secs(20);
    loop {
        let v = ok(c, "job_status", json!({ "job": job })).await;
        if f(&v) || Instant::now() > end {
            assert!(f(&v), "{v}");
            return v;
        }
        tokio::time::sleep(Duration::from_millis(20)).await
    }
}

/// Job `job`'s state once it reads `state`.
async fn reads(c: &Cli, job: &Value, state: &str) -> Value {
    when(c, job, |v| v["state"] == state).await
}

/// Job `job` running with at least `n` frames done.
async fn frames(c: &Cli, job: &Value, n: u64) -> Value {
    when(c, job, |v| {
        v["state"] == "running" && v["frames_done"].as_u64() >= Some(n)
    })
    .await
}

/// The files under `d` and its subfolders, relative, sorted; none when it is missing.
fn files(d: &Path) -> Vec<String> {
    let (mut v, mut todo) = (vec![], vec![d.to_path_buf()]);
    while let Some(p) = todo.pop() {
        for e in fs::read_dir(&p).into_iter().flatten().flatten() {
            match e.file_type().is_ok_and(|t| t.is_dir()) {
                true => todo.push(e.path()),
                false => v.push(e.path().strip_prefix(d).unwrap().display().to_string()),
            }
        }
    }
    v.sort();
    v
}

/// The render records kept in state dir `d`.
fn records(d: &Path) -> Vec<RJob> {
    Renders::load(d.join("mcp-renders")).unwrap().0.v
}

/// The newest job of job_list (of `kind`).
async fn newest(c: &Cli, kind: &str) -> Value {
    let v = ok(c, "job_list", json!({ "kind": kind, "limit": 1 })).await;
    v["jobs"][0].clone()
}

/// job_list's handles and kinds.
async fn list(c: &Cli, a: Value) -> Vec<(Value, Value)> {
    let v = ok(c, "job_list", a).await;
    (v["jobs"].as_array().unwrap().iter())
        .map(|j| (j["job"].clone(), j["kind"].clone()))
        .collect()
}

fn unb64(s: &str) -> Vec<u8> {
    let v = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        _ => 63,
    };
    let b: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut o = vec![];
    for ch in b.chunks(4) {
        let n = (ch.iter().enumerate()).fold(0u32, |n, (i, &c)| n | (v(c) as u32) << (18 - 6 * i));
        o.extend_from_slice(&n.to_be_bytes()[1..ch.len()]);
    }
    o
}

/// A done still's answer: its JSON first, then one JPEG preview of the whole picture at a long edge of 1568 under
/// 1 MB, no structured content (a client preferring it would drop the image).
fn still_answer(r: &CallToolResult) -> Value {
    assert!(
        r.is_error == Some(false) && r.structured_content.is_none() && r.content.len() == 2,
        "{r:?}"
    );
    let v = of(r);
    let img = r.content[1].as_image().unwrap();
    assert_eq!(img.mime_type, "image/jpeg");
    let j = unb64(&img.data);
    assert!(
        j.len() < 1_000_000 && j.len() == v["preview"]["bytes"],
        "{v}"
    );
    let mut px = vec![];
    let (w, h) = media::mjpeg::Mjpeg::default().rgba(&j, &mut px).unwrap();
    assert_eq!((w, h), (1568, 784));
    assert_eq!(
        (
            &v["state"],
            &v["width"],
            &v["height"],
            &v["preview"]["width"]
        ),
        (&json!("done"), &json!(3200), &json!(1600), &json!(1568))
    );
    assert!(v["time_s"].is_number() && v["secs"].is_number(), "{v}");
    v
}

/// With a backend the GPU tools are listed: catalog in the read tier (read-only), render_still and export in the local
/// tier; a client narrowed to read gets catalog and is refused the others, a read-only host refuses them by tier; the
/// instructions name them. Nothing of it reaches the backend's check or starts the GPU.
#[tokio::test(flavor = "multi_thread")]
async fn lists() {
    let f = Fake::new();
    let (d, rec) = dirs("lists");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local,camera", None).await;
    let c = connect(&h, None).await;
    let n = names(&c).await;
    assert!(GPU.iter().all(|g| n.iter().any(|x| x == g)), "{n:?}");
    for t in c.s.list_all_tools().await.unwrap() {
        let ro = t.annotations.unwrap().read_only_hint;
        if GPU.contains(&&*t.name) {
            assert_eq!(ro, Some(t.name == "catalog"), "{}", t.name);
        }
    }
    let i = c.s.peer_info().unwrap().instructions.clone().unwrap();
    assert!(
        i.contains("catalog, then render_still or export, then job_wait; photos: later")
            && i.contains("Faces and licence plates: blur (best effort)."),
        "{i}"
    );
    let r = connect(&h, Tiers::parse("read").ok()).await;
    let n = names(&r).await;
    assert!(
        n.iter().any(|x| x == "catalog") && !n.iter().any(|x| x == "render_still" || x == "export"),
        "{n:?}"
    );
    assert_eq!(ok(&r, "catalog", json!({})).await["version"], 1);
    let e = err(&r, "render_still", json!({ "path": p })).await;
    assert!(e.contains("this client's --tiers leaves out"), "{e}");
    let h2 = host(&f, &rec, "read", None).await;
    let c2 = connect(&h2, None).await;
    let n = names(&c2).await;
    assert!(n.iter().any(|x| x == "catalog") && !n.iter().any(|x| x == "export"));
    let e = err(&c2, "export", json!({ "path": p })).await;
    assert!(e.contains("in the local tier, which is not enabled"), "{e}");
    assert_eq!((f.starts(), f.checks.load(Relaxed)), (0, 0));
    h.shutdown().await;
    h2.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// Every refusal before the GPU starts: a later feature by name and by alias, a word agents guess for blur, a blur of
/// the wrong type or id, the other tool's fields, a typo, sizes outside the catalog's lists, a camera file's name, the
/// source (a proxy alone, an `.mp4`, a photo, a missing file, outside the roots, no UTF-8), the backend's check (blur
/// on a file per lens too), a memory-only host. None starts the GPU or leaves a file, and the catalog and status show
/// it off. Then the spellings agents use are accepted: `stabilization`, `color_match`, fps as a number or a label,
/// output 360 as a number, blur ids by alias and name in any case, a look's display name, a projection's alias.
#[tokio::test(flavor = "multi_thread")]
async fn refusals() {
    let f = Fake::new();
    let (d, rec) = dirs("refusals");
    let p = rec.join(REC).display().to_string();
    let with = |a: Value| {
        let mut v = json!({ "path": p });
        (v.as_object_mut().unwrap()).extend(a.as_object().unwrap().clone());
        v
    };
    let h = host(&f, &rec, "read,local,camera", None).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "catalog", json!({})).await;
    assert_eq!(
        (
            &v["default_preset"],
            &v["gpu"]["state"],
            &v["outputs"]["360"]["video_widths"],
            &v["outputs"]["flat"]["video_sizes"]
        ),
        (
            &json!("p360"),
            &json!("off"),
            &json!([64, 128]),
            &json!([[32, 16], [64, 32]])
        )
    );
    assert_eq!(v["presets"][1]["id"], "flat");
    // the blur targets, none by default; available as far as it knows (the GPU is off)
    let b = &v["blur"];
    assert_eq!(
        (
            &b["targets"][0]["id"],
            &b["targets"][1]["aliases"][0],
            &b["default"]
        ),
        (&json!("faces"), &json!("plate"), &json!([]))
    );
    assert!(
        b.get("available").is_none()
            && b["how"].as_str().unwrap().contains("one source frame in 3"),
        "{b}"
    );
    assert!(
        !(v["later"].as_array().unwrap().iter()).any(|l| l["id"] == "face_blur"),
        "{}",
        v["later"]
    );
    assert_eq!(ok(&c, "status", json!({})).await["gpu"]["state"], "off");
    let later =
        |k: &str, l: &str| format!("`{k}` is not available to agents yet (catalog: later: {l})");
    for (t, a, w) in [
        (
            "export",
            json!({ "faces": true }),
            r#"`faces` is no argument: blur: ["faces"] (catalog: blur)"#.into(),
        ),
        (
            "render_still",
            json!({ "plate_blur": true }),
            r#"`plate_blur` is no argument: blur: ["plates"] (catalog: blur)"#.into(),
        ),
        (
            "render_still",
            json!({ "blur": 1 }),
            "blur 1: true, a catalog blur id or a list of them (catalog: blur)".into(),
        ),
        (
            "export",
            json!({ "blur": ["faces", "cats"] }),
            "blur \"cats\": one of faces, plates".into(),
        ),
        (
            "export",
            json!({ "trails": true }),
            later("trails", "shot_lab"),
        ),
        ("export", json!({ "lut": "a.cube" }), later("lut", "lut")),
        (
            "export",
            json!({ "time_s": 1 }),
            "`time_s` is for render_still only".into(),
        ),
        (
            "render_still",
            json!({ "fps": 30 }),
            "`fps` is for export only".into(),
        ),
        (
            "export",
            json!({ "width": 100 }),
            "width 100: a 360 export is 64, 128 wide (catalog outputs)".into(),
        ),
        (
            "export",
            json!({ "output": "flat", "width": 33, "height": 16 }),
            "33 × 16: a flat export is 32 × 16, 64 × 32 (catalog outputs)".into(),
        ),
        (
            "export",
            json!({ "name": "IMG_20260201_111111_00_701" }),
            "name \"IMG_20260201_111111_00_701\": a camera file's name: pick another".into(),
        ),
        (
            "render_still",
            json!({ "name": "VID_20260117_141148_00_635.jpg" }),
            "name \"VID_20260117_141148_00_635.jpg\": a camera file's name: pick another".into(),
        ),
        (
            "export",
            json!({ "fps": 31 }),
            "fps 31: one of source, 24, 25, 29.97, 30, 60".into(),
        ),
        (
            "export",
            json!({ "look": "sepia" }),
            "look \"sepia\": not a catalog look (teal_orange)".into(),
        ),
        (
            "export",
            json!({ "subdir": "../x" }),
            "../x: `..` is refused".into(),
        ),
    ] {
        assert_eq!(err(&c, t, with(a.clone())).await, w, "{t} {a}");
    }
    let e = err(&c, "export", with(json!({ "lok": "teal" }))).await;
    assert!(e.contains("unknown field `lok`"), "{e}");
    let src = |x: &str| json!({ "path": rec.join(x).display().to_string() });
    for (x, w) in [
        (
            "LRV_20260117_141148_01_636.lrv",
            "an .lrv is the camera's proxy",
        ),
        ("x.mp4", "an .mp4 has no Insta360 trailer"),
        (
            "IMG_20260117_141148_00_637.insp",
            "a photo; render_still and export take X5 recordings (.insv): photos later",
        ),
        ("nope.insv", "nope.insv: "),
    ] {
        let e = err(&c, "export", src(x)).await;
        assert!(e.contains(w), "{x}: {e}");
    }
    let out = json!({ "path": d.join("VID_20260117_141148_00_999.insv").display().to_string() });
    fs::write(d.join("VID_20260117_141148_00_999.insv"), "frames=300").unwrap();
    let e = err(&c, "render_still", out).await;
    assert!(e.contains("not under a read root"), "{e}");
    // a source whose canonical path is no UTF-8 (where the file system takes such a name: not APFS)
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let odd = rec.join(std::ffi::OsStr::from_bytes(b"caf\xe9"));
        if fs::create_dir(&odd).is_ok() {
            fs::write(odd.join(REC), "frames=300").unwrap();
            std::os::unix::fs::symlink(odd.join(REC), rec.join("odd.insv")).unwrap();
            let e = err(&c, "export", src("odd.insv")).await;
            assert!(e.ends_with("the path is no UTF-8"), "{e}");
        }
    }
    assert_eq!(f.checks.load(Relaxed), 0, "nothing reached the backend yet");
    // the backend's check: what it says, as it says it
    assert_eq!(
        err(&c, "export", src("bare.insv")).await,
        "bare.insv: no Insta360 trailer"
    );
    let e = err(&c, "render_still", with(json!({ "time_s": 10 }))).await;
    assert_eq!(e, "time_s 10: outside the clip (0 to 10.00 s)");
    let a = json!({ "path": rec.join("short.insv"), "stabilisation": "horizon" });
    assert_eq!(
        err(&c, "export", a).await,
        "short.insv: no IMU data, so no horizon stabilisation"
    );
    // blur on a recording in a file per lens: refused, in the app's words
    let a = json!({ "path": rec.join(LENS), "blur": true });
    assert_eq!(
        err(&c, "render_still", a).await,
        "blur: a recording in a file per lens (_00_ + _10_) is not supported yet; leave out blur"
    );
    // a subfolder that leads out of the agent folder
    #[cfg(unix)]
    {
        fs::create_dir_all(h.agent()).unwrap();
        std::os::unix::fs::symlink(&d, h.agent().join("out")).unwrap();
        let e = err(&c, "render_still", with(json!({ "subdir": "out/x" }))).await;
        assert!(e.contains("leads out of the agent folder"), "{e}");
        assert!(!d.join("x").exists(), "made where the link leads");
        fs::remove_file(h.agent().join("out")).unwrap();
    }
    // a host whose renders file it can't read keeps its jobs in memory: GPU tools refused (a bad argument said first)
    let mem = d.join("mem");
    fs::create_dir_all(mem.join("mcp-renders")).unwrap();
    let hm = host(&f, &rec, "read,local", Some(&mem)).await;
    let cm = connect(&hm, None).await;
    assert_eq!(
        err(&cm, "export", json!({ "path": p })).await,
        "another x5desk host holds the jobs file: GPU tools need it to clean up after a crash"
    );
    let e = err(&cm, "export", with(json!({ "look": "sepia" }))).await;
    assert!(e.starts_with("look \"sepia\""), "{e}");
    hm.shutdown().await;
    // none started the GPU, made a job or left a file
    assert_eq!(f.starts(), 0);
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    assert_eq!(ok(&c, "job_list", json!({})).await["total"], 0);
    assert_eq!(ok(&c, "status", json!({})).await["gpu"]["state"], "off");
    // accepted as agents spell them
    let accepted = [
        json!({ "stabilization": "Horizon", "color_match": false }),
        json!({ "fps": 30, "output": 360 }),
        json!({ "fps": "29.97", "codec": "H264", "qp": 24 }),
        json!({ "blur": ["Licence_Plates", "Faces", "face"] }),
        json!({ "blur": "plate" }),
        json!({ "look": "TEAL & orange", "output": "flat", "view": { "projection": "persp", "yaw": 90 } }),
    ];
    for (i, a) in accepted.into_iter().enumerate() {
        let v = ok(&c, "export", with(a.clone())).await;
        assert_eq!(v["state"], "queued", "{a}: {v}");
        let w = f.asked();
        match i {
            0 => assert_eq!(
                (w.stabilisation.as_deref(), w.colour_match),
                (Some("horizon"), Some(false))
            ),
            1 => assert_eq!(
                (w.fps.as_deref(), w.output),
                (Some("30"), Some(Output::Pano))
            ),
            2 => assert_eq!(
                (w.fps.as_deref(), w.codec.as_deref(), w.qp),
                (Some("29.97"), Some("h264"), Some(24))
            ),
            // the catalog's order, once each
            3 => assert_eq!(w.blur, ["faces", "plates"]),
            4 => assert_eq!(w.blur, ["plates"]),
            _ => {
                let view = w.view.unwrap();
                assert_eq!(
                    (w.look.as_deref(), view.projection.as_deref(), view.yaw),
                    (Some("teal_orange"), Some("perspective"), Some(90.0))
                )
            }
        }
        let x = ok(&c, "job_cancel", json!({ "job": v["job"] })).await;
        assert_eq!(x["state"], "cancelled", "{x}");
    }
    assert_eq!(f.starts(), 1, "the first accepted job started the GPU");
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// At most 16 unfinished render jobs; the free space counts this host's unfinished jobs' estimates, and no longer a
/// cancelled one's.
#[tokio::test(flavor = "multi_thread")]
async fn limits() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 50);
    let (d, rec) = dirs("limits");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let mut v = vec![];
    for i in 0..16 {
        v.push(
            ok(&c, "export", json!({ "path": p, "name": format!("n{i}") })).await["job"].clone(),
        );
    }
    assert_eq!(
        err(&c, "render_still", json!({ "path": p })).await,
        "16 render jobs of this server are unfinished: job_wait or job_cancel one first"
    );
    for j in &v {
        assert_eq!(
            ok(&c, "job_cancel", json!({ "job": j })).await["state"],
            "cancelled"
        );
    }
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    // two jobs of two thirds of the space left beside the spare: the second is refused while the first is unfinished
    let sp = media::part::avail(h.agent()).unwrap();
    let spare = 2u64 << 30;
    if sp.free > spare + (1 << 30) {
        let b = (sp.free - spare) / 3 * 2;
        fs::write(rec.join("big.insv"), format!("frames=300 imu=1 bytes={b}")).unwrap();
        let big = json!({ "path": rec.join("big.insv") });
        let x = ok(&c, "export", big.clone()).await;
        let e = err(&c, "export", big.clone()).await;
        assert!(
            e.starts_with("not enough space in ")
                && e.ends_with(" free after this server's other jobs"),
            "{e}"
        );
        ok(&c, "job_cancel", json!({ "job": x["job"] })).await;
        let y = ok(&c, "export", big).await;
        ok(&c, "job_cancel", json!({ "job": y["job"] })).await;
    } else {
        eprintln!("less than 3 GB free: the free-space case skipped")
    }
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// A timed recording and this host's renders exclude each other, both ways (its stop is this host's timer, which a
/// crash mid-render would lose); an untimed recording and renders coexist.
#[tokio::test(flavor = "multi_thread")]
async fn timed() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 30);
    let (d, rec) = dirs("timed");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local,camera", None).await;
    let c = connect(&h, None).await;
    ok(&c, "camera_connect", json!({})).await;
    ok(&c, "capture_status", json!({})).await;
    ok(&c, "capture_mode", json!({ "mode": "video" })).await;
    let start = |s: Option<u32>| match s {
        Some(s) => json!({ "action": "start", "seconds": s }),
        None => json!({ "action": "start" }),
    };
    let r = ok(&c, "capture_record", start(Some(120))).await;
    assert_eq!(r["capturing"], true, "{r}");
    for t in ["export", "render_still"] {
        let e = err(&c, t, json!({ "path": p })).await;
        assert!(
            e.starts_with("the camera records until ")
                && e.ends_with(" (capture_record, seconds): render after it ends"),
            "{e}"
        );
    }
    assert_eq!(f.checks.load(Relaxed), 0);
    ok(&c, "capture_record", json!({ "action": "stop" })).await;
    let v = ok(&c, "export", json!({ "path": p })).await;
    let e = err(&c, "capture_record", start(Some(5))).await;
    assert!(
        e.starts_with(
            "this host renders (1 unfinished render job, job_list): a recording with seconds waits"
        ),
        "{e}"
    );
    // untimed: it records while the export runs, and a still renders meanwhile
    ok(&c, "capture_record", start(None)).await;
    let s = call(&c, "render_still", json!({ "path": p })).await;
    still_answer(&s);
    assert!(ok(&c, "capture_record", json!({ "action": "stop" })).await["file"].is_string());
    assert_eq!(
        ok(&c, "job_cancel", json!({ "job": v["job"] })).await["state"],
        "cancelled"
    );
    // no render left: a timed recording starts again
    assert_eq!(
        ok(&c, "capture_record", start(Some(120))).await["capturing"],
        true
    );
    ok(&c, "capture_record", json!({ "action": "stop" })).await;
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// A stop that comes before the camera begins: the host stops the recording once it does, and it stays untimed, so
/// renders go on meanwhile (no timed recording rests on this host's timer); one with seconds stays timed, until now.
#[tokio::test(flavor = "multi_thread")]
async fn stopped_early() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 100);
    let (d, rec) = dirs("early");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local,camera", None).await;
    let c = connect(&h, None).await;
    ok(&c, "camera_connect", json!({})).await;
    ok(&c, "capture_status", json!({})).await;
    ok(&c, "capture_mode", json!({ "mode": "video" })).await;
    let v = ok(&c, "export", json!({ "path": p })).await;
    let cam = &h.sim().unwrap().fake;
    cam.lock().unwrap().capture.as_mut().unwrap().lag = Some(Duration::from_secs(7));
    let r = ok(&c, "capture_record", json!({ "action": "start" })).await;
    assert_eq!(r["capturing"], false, "{r}");
    let r = ok(&c, "capture_record", json!({ "action": "stop" })).await;
    assert!(
        r["note"]
            .as_str()
            .unwrap()
            .ends_with("the host stops it as soon as it does"),
        "{r}"
    );
    assert_eq!(ok(&c, "capture_status", json!({})).await["stops_in_s"], 0);
    still_answer(&call(&c, "render_still", json!({ "path": p })).await);
    assert!(until(20.0, || h.with(|s| s.rec.is_none())).await);
    let e = h.with(|s| s.ended.clone());
    assert!(matches!(e, Some((_, Ok(_)))), "{e:?}");
    assert_eq!(
        ok(&c, "job_cancel", json!({ "job": v["job"] })).await["state"],
        "cancelled"
    );
    // one with seconds stays timed until it stops, which is now
    let r = ok(
        &c,
        "capture_record",
        json!({ "action": "start", "seconds": 120 }),
    )
    .await;
    assert_eq!(r["capturing"], false, "{r}");
    let late = err(&c, "render_still", json!({ "path": p })).await;
    ok(&c, "capture_record", json!({ "action": "stop" })).await;
    let now = err(&c, "render_still", json!({ "path": p })).await;
    assert!(
        now.starts_with("the camera records until ") && now < late,
        "{now} / {late}"
    );
    assert!(until(20.0, || h.with(|s| s.rec.is_none())).await);
    cam.lock().unwrap().capture.as_mut().unwrap().lag = None;
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// A name in use is never written over: a file, a `.part`, a dangling link and an unfinished render job (whose
/// `.part` another program removed) at the name are each numbered past; a file made at the name while the job runs
/// fails the job and stays as it was, as do the others.
#[tokio::test(flavor = "multi_thread")]
async fn never_over() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 10);
    let (d, rec) = dirs("never");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let a = h.agent().to_path_buf();
    fs::create_dir_all(&a).unwrap();
    // a still stuck in its setup, its `.part` removed: its name stays taken while it is unfinished
    f.set(|k| k.stuck = Some(0));
    let u = ok(
        &c,
        "render_still",
        json!({ "path": p, "name": "u", "wait_s": 0 }),
    )
    .await;
    assert_eq!(u["path"], a.join("u.jpg").display().to_string());
    when(&c, &u["job"], |v| v["phase"] == "setting_up").await;
    fs::remove_file(a.join("u.jpg.part")).unwrap();
    let u2 = ok(
        &c,
        "render_still",
        json!({ "path": p, "name": "u.jpg", "wait_s": 0 }),
    )
    .await;
    assert_eq!(u2["path"], a.join("u-2.jpg").display().to_string());
    // a file, a download's `.part`, a dangling link
    let stem = "VID_20260117_141148_00_635_p360";
    fs::write(a.join(format!("{stem}.mp4")), b"mine").unwrap();
    fs::write(a.join(format!("{stem}-2.mp4.part")), b"a download's").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(a.join("gone"), a.join(format!("{stem}-3.mp4"))).unwrap();
    let n = if cfg!(unix) { 4 } else { 3 };
    let v = ok(&c, "export", json!({ "path": p })).await;
    let dst = a.join(format!("{stem}-{n}.mp4"));
    assert_eq!(v["path"], dst.display().to_string());
    // the user's file at the name meanwhile: the job fails at its publish, the file stays
    frames(&c, &v["job"], 2).await;
    fs::write(&dst, b"the user's").unwrap();
    let x = reads(&c, &v["job"], "failed").await;
    assert!(
        x["error"]
            .as_str()
            .unwrap()
            .ends_with("appeared while it was rendered; nothing replaced"),
        "{x}"
    );
    assert_eq!(fs::read(&dst).unwrap(), b"the user's");
    assert!(!media::part::of(&dst).exists());
    assert_eq!(fs::read(a.join(format!("{stem}.mp4"))).unwrap(), b"mine");
    assert_eq!(
        fs::read(a.join(format!("{stem}-2.mp4.part"))).unwrap(),
        b"a download's"
    );
    // the stills: the queued one cancelled, the stuck one once let go
    assert_eq!(
        ok(&c, "job_cancel", json!({ "job": u2["job"] })).await["state"],
        "cancelled"
    );
    let x = ok(&c, "job_cancel", json!({ "job": u["job"] })).await;
    assert_eq!(x["state"], "running", "it ignores its stop: {x}");
    f.run(0).go();
    reads(&c, &u["job"], "cancelled").await;
    assert!(!a.join("u.jpg").exists() && !a.join("u-2.jpg").exists());
    assert!(
        !files(&a).iter().any(|x| x.ends_with(".jpg.part")),
        "{:?}",
        files(&a)
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// job_cancel: a queued export is cancelled at once and never starts, a running one stops within 1 s, one cancelled
/// while its submit is under way never runs, a parked one (the GPU starting) ends at once and is never submitted.
/// Each reads cancelled with no file and no `.part`, and its record says so.
#[tokio::test(flavor = "multi_thread")]
async fn cancel() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 20);
    let (d, rec) = dirs("cancel");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = Arc::new(connect(&h, None).await);
    let a = h.agent().to_path_buf();
    let v = ok(&c, "export", json!({ "path": p, "name": "run" })).await;
    assert_eq!(
        (&v["duration_s"], &v["width"], &v["height"]),
        (&json!(10.0), &json!(64), &json!(32)),
        "{v}"
    );
    let q = ok(&c, "export", json!({ "path": p, "name": "queued" })).await;
    frames(&c, &v["job"], 3).await;
    assert_eq!(
        ok(&c, "job_status", json!({ "job": q["job"] })).await["phase"],
        "waiting"
    );
    let t = Instant::now();
    let x = ok(&c, "job_cancel", json!({ "job": q["job"] })).await;
    assert_eq!(
        (&x["state"], &x["cancel_requested"]),
        (&json!("cancelled"), &json!(true))
    );
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    assert!(!f.run(1).started.load(Relaxed), "never started");
    let t = Instant::now();
    let x = ok(&c, "job_cancel", json!({ "job": v["job"] })).await;
    assert_eq!(
        (&x["state"], &x["note"]),
        (&json!("cancelled"), &Value::Null),
        "{x}"
    );
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    assert!(files(&a).is_empty(), "{:?}", files(&a));
    // cancelled while its submit is under way: the export call is still waiting for it
    f.set(|k| k.submit_ms = 1000);
    let cc = c.clone();
    let pp = p.clone();
    let slow =
        tokio::spawn(async move { ok(&cc, "export", json!({ "path": pp, "name": "slow" })).await });
    assert!(until(5.0, || f.checks.load(Relaxed) == 3).await);
    let mut s = Value::Null;
    for _ in 0..100 {
        s = newest(&c, "export").await;
        if s["path"].as_str().is_some_and(|x| x.ends_with("slow.mp4")) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await
    }
    let x = ok(&c, "job_cancel", json!({ "job": s["job"] })).await;
    assert_eq!(x["state"], "cancelled", "{x}");
    slow.await.unwrap();
    // its run was told to stop as its submit returned: the cancel had come meanwhile
    let r = f.run(2);
    let at = r.stopped.lock().unwrap().unwrap();
    assert!(
        at.duration_since(r.made) < Duration::from_millis(100),
        "{:?}",
        at - r.made
    );
    assert!(files(&a).is_empty(), "{:?}", files(&a));
    // parked while the GPU starts: never submitted
    f.gpu_is(GpuSt::Off);
    f.set(|k| (k.submit_ms, k.start_ms) = (0, 1000));
    let w = ok(&c, "export", json!({ "path": p, "name": "parked" })).await;
    assert_eq!(w["phase"], "gpu_starting");
    let t = Instant::now();
    assert_eq!(
        ok(&c, "job_cancel", json!({ "job": w["job"] })).await["state"],
        "cancelled"
    );
    assert!(
        t.elapsed() < Duration::from_millis(500),
        "{:?}",
        t.elapsed()
    );
    assert!(until(5.0, || matches!(*f.st.lock().unwrap(), GpuSt::Ready(_))).await);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(f.runs.lock().unwrap().len(), 3, "nothing submitted for it");
    assert!(files(&a).is_empty(), "{:?}", files(&a));
    h.shutdown().await;
    let r = records(&state);
    assert_eq!(r.len(), 4);
    for j in &r {
        let e = j.end.as_ref().unwrap();
        assert!(
            e.state == "cancelled" && e.err.is_empty() && j.cancelled,
            "{j:?}"
        );
    }
    fs::remove_dir_all(&d).unwrap();
}

/// A submit that fails or panics fails its job: the call says why, the record ends failed, no `.part` is left; the
/// same for a job parked while the GPU started. The host's locks are fine after the panic.
#[tokio::test(flavor = "multi_thread")]
async fn submit_fails() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.submit_err = Some("the fake's lane refused".into()));
    let (d, rec) = dirs("submit");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let e = err(&c, "export", json!({ "path": p, "name": "a" })).await;
    assert!(
        e.starts_with("export ") && e.ends_with(" failed: the fake's lane refused"),
        "{e}"
    );
    let j = newest(&c, "export").await;
    let x = ok(&c, "job_status", json!({ "job": j["job"] })).await;
    assert_eq!(
        (&x["state"], &x["error"]),
        (&json!("failed"), &json!("the fake's lane refused"))
    );
    f.set(|k| (k.submit_err, k.submit_panics) = (None, true));
    let e = err(&c, "render_still", json!({ "path": p, "name": "b" })).await;
    assert!(
        e.ends_with(" failed: the GPU backend panicked: the fake's submit broke"),
        "{e}"
    );
    assert_eq!(newest(&c, "still").await["state"], "failed");
    // parked: its submit at the GPU's start fails it
    f.gpu_is(GpuSt::Off);
    f.set(|k| (k.submit_panics, k.submit_err) = (false, Some("refused".into())));
    let v = ok(&c, "export", json!({ "path": p, "name": "c" })).await;
    assert_eq!(v["phase"], "gpu_starting");
    assert_eq!(reads(&c, &v["job"], "failed").await["error"], "refused");
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    assert_eq!(
        ok(&c, "status", json!({})).await["jobs"],
        json!({ "running": 0, "queued": 0, "finished": 3 })
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// render_still: the JSON in the first text block and the preview after it, the full JPEG at `path`; with wait_s 0
/// the job at once, then job_wait's answer with the image; a call cancelled after its reservation leaves the still to
/// finish (job_list has it, done, and job_wait its image).
#[tokio::test(flavor = "multi_thread")]
async fn still_image() {
    let f = Fake::new();
    f.up();
    let (d, rec) = dirs("still");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let v = still_answer(&call(&c, "render_still", json!({ "path": p })).await);
    let dst = h
        .agent()
        .join("VID_20260117_141148_00_635_0.00s_360_3200.jpg");
    assert_eq!(
        (&v["kind"], &v["path"]),
        (&json!("still"), &json!(dst.display().to_string()))
    );
    assert_eq!(fs::read(&dst).unwrap(), *jpeg());
    assert_eq!(v["bytes"], jpeg().len());
    // answered at once, the image from job_wait
    f.set(|k| k.frame_ms = 400);
    let w = ok(
        &c,
        "render_still",
        json!({ "path": p, "time_s": 2.5, "wait_s": 0 }),
    )
    .await;
    assert_eq!(
        (&w["finished"], &w["note"]),
        (
            &json!(false),
            &json!("still rendering: job_wait answers with the image")
        )
    );
    let r = call(&c, "job_wait", json!({ "job": w["job"], "wait_s": 30 })).await;
    let x = still_answer(&r);
    assert!(
        x["path"].as_str().unwrap().ends_with("_2.50s_360_3200.jpg"),
        "{x}"
    );
    // the call cancelled once its job is made: the still goes on
    let ask = call(
        &c,
        "render_still",
        json!({ "path": p, "time_s": 1, "wait_s": 30 }),
    );
    let made = async {
        loop {
            let v = ok(&c, "job_list", json!({ "kind": "still" })).await;
            if v["total"] == 3 {
                return v["jobs"][0]["job"].clone();
            }
            tokio::time::sleep(Duration::from_millis(10)).await
        }
    };
    let j = tokio::select! {
        r = ask => panic!("answered before the cancel: {r:?}"),
        j = made => j,
    };
    let r = call(&c, "job_wait", json!({ "job": j, "wait_s": 30 })).await;
    assert_eq!(still_answer(&r)["job"], j);
    let log = h.sim().unwrap().log.clone();
    let said = || {
        fs::read_to_string(&log)
            .is_ok_and(|l| l.contains(" render_still ") && l.contains("error: cancelled"))
    };
    assert!(
        until(5.0, said).await,
        "{}",
        fs::read_to_string(&log).unwrap()
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// The host's end with renders running and queued: shutdown cancels them and settles each by identity within 1 s,
/// the records say why, no `.part` is left; the next host shows them cancelled.
#[tokio::test(flavor = "multi_thread")]
async fn end() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 20);
    let (d, rec) = dirs("end");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "export", json!({ "path": p, "name": "x" })).await;
    ok(&c, "export", json!({ "path": p, "name": "y" })).await;
    frames(&c, &v["job"], 3).await;
    let t = Instant::now();
    h.shutdown().await;
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    let r = records(&state);
    let st: Vec<(&str, &str)> = (r.iter())
        .map(|j| j.end.as_ref().unwrap())
        .map(|e| (e.state.as_str(), e.err.as_str()))
        .collect();
    assert_eq!(st, [("cancelled", ENDED), ("cancelled", ENDED)]);
    drop(c);
    drop(h);
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let x = ok(&c, "job_status", json!({ "job": v["job"] })).await;
    assert_eq!(
        (&x["state"], &x["note"]),
        (&json!("cancelled"), &json!(ENDED)),
        "{x}"
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// The host's end while a job's submit is under way: the end waits for it, its run is cancelled as soon as the
/// submit returns and settled cancelled; no `.part` is left and nothing is published.
#[tokio::test(flavor = "multi_thread")]
async fn end_submitting() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.submit_ms = 1500);
    let (d, rec) = dirs("submitting");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = Arc::new(connect(&h, None).await);
    let cc = c.clone();
    tokio::spawn(async move { cc.s.call_tool(params("export", json!({ "path": p }))).await });
    assert!(until(5.0, || files(h.agent()).len() == 1).await, "reserved");
    let t = Instant::now();
    h.shutdown().await;
    let e = t.elapsed();
    assert!(
        e >= Duration::from_millis(800) && e < mcp::host::RENDERS,
        "{e:?}"
    );
    let r = records(&state);
    assert_eq!(
        r[0].end
            .as_ref()
            .map(|e| (e.state.as_str(), e.err.as_str())),
        Some(("cancelled", ENDED))
    );
    let run = f.run(0);
    assert!(run.stop.load(Relaxed) && until(5.0, || run.ended()).await);
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    fs::remove_dir_all(&d).unwrap();
}

/// A run stuck in the driver (its stop ignored) at the host's end: the end waits `RENDERS` for it, then settles it by
/// identity, its `.part` gone and its record cancelled, within 11 s; dropping the host and the backend then takes no
/// time (nothing joins the lane); the writer, let go at last, ends on its stop and publishes nothing.
#[tokio::test(flavor = "multi_thread")]
async fn end_stuck() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.stuck = Some(2));
    let (d, rec) = dirs("stuck");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "export", json!({ "path": p, "name": "x" })).await;
    frames(&c, &v["job"], 2).await;
    let dst = h.agent().join("x.mp4");
    assert!(media::part::of(&dst).exists());
    let t = Instant::now();
    h.shutdown().await;
    let e = t.elapsed();
    assert!(
        e >= mcp::host::RENDERS && e < Duration::from_secs(11),
        "{e:?}"
    );
    assert!(!media::part::of(&dst).exists() && !dst.exists());
    let r = records(&state);
    assert_eq!(
        r[0].end
            .as_ref()
            .map(|e| (e.state.as_str(), e.err.as_str())),
        Some(("cancelled", ENDED))
    );
    let log = fs::read_to_string(&h.sim().unwrap().log).unwrap();
    let j = v["job"].as_str().unwrap();
    assert!(
        log.contains(&format!(
            "export {j} still in frames at the host's end, 8 s after its cancel"
        )),
        "{log}"
    );
    let run = f.run(0);
    let t = Instant::now();
    drop(c);
    drop(h);
    drop(f);
    assert!(
        t.elapsed() < Duration::from_millis(200),
        "{:?}",
        t.elapsed()
    );
    run.go();
    assert!(until(5.0, || run.ended()).await);
    assert_eq!(run.now(), RunSt::Cancelled);
    assert!(!media::part::of(&dst).exists() && !dst.exists());
    fs::remove_dir_all(&d).unwrap();
}

/// A call held in the backend's check while the host ends is refused once it goes on ("the host is shutting down",
/// in the log: its client went with the host) and leaves neither a record nor a `.part`.
#[tokio::test(flavor = "multi_thread")]
async fn end_race() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.hold = true);
    let (d, rec) = dirs("race");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = Arc::new(connect(&h, None).await);
    let cc = c.clone();
    tokio::spawn(async move { cc.s.call_tool(params("export", json!({ "path": p }))).await });
    assert!(until(5.0, || f.checks.load(Relaxed) == 1).await);
    h.shutdown().await;
    f.set(|k| k.hold = false);
    let log = h.sim().unwrap().log.clone();
    let said = || {
        fs::read_to_string(&log).is_ok_and(|l| {
            l.lines()
                .any(|l| l.contains(" export ") && l.contains("error: the host is shutting down"))
        })
    };
    assert!(
        until(5.0, said).await,
        "{}",
        fs::read_to_string(&log).unwrap()
    );
    assert!(records(&state).is_empty() && files(h.agent()).is_empty());
    assert!(f.runs.lock().unwrap().is_empty());
    fs::remove_dir_all(&d).unwrap();
}

/// A record that can't be saved (the state dir read-only) refuses the call: no job, no `.part`.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn save_fails() {
    use std::os::unix::fs::PermissionsExt;
    if unsafe { libc::geteuid() } == 0 {
        return;
    }
    let f = Fake::new();
    f.up();
    let (d, rec) = dirs("save");
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    fs::create_dir_all(h.agent()).unwrap();
    let mode = |m| fs::set_permissions(&state, fs::Permissions::from_mode(m)).unwrap();
    mode(0o500);
    let r = call(&c, "export", json!({ "path": rec.join(REC) })).await;
    mode(0o700);
    assert!(
        text(&r).starts_with("can't keep the job record: "),
        "{}",
        text(&r)
    );
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    assert_eq!(ok(&c, "job_list", json!({})).await["total"], 0);
    assert!(f.runs.lock().unwrap().is_empty());
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// An agent folder on FAT or exFAT (`X5_FAT_DIR`, a folder on one: a disk image; skipped without it), here through a
/// link, refuses renders at their reservation, a subfolder too: no record, no `.part`, nothing submitted.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn fat_refused() {
    let Some(fat) = std::env::var_os("X5_FAT_DIR").map(PathBuf::from) else {
        eprintln!("X5_FAT_DIR unset: skipped");
        return;
    };
    let f = Fake::new();
    f.up();
    let (d, rec) = dirs("fat");
    let (state, on) = (
        d.join("state"),
        fat.join(format!("x5desk-mcp-fat-{}", std::process::id())),
    );
    let _ = fs::remove_dir_all(&on);
    fs::create_dir_all(&on).unwrap();
    fs::create_dir_all(&state).unwrap();
    std::os::unix::fs::symlink(&on, state.join("agent")).unwrap();
    assert!(media::part::fat(&on), "{}: not FAT or exFAT", on.display());
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let p = rec.join(REC);
    let asks = [
        ("render_still", json!({ "path": p })),
        ("export", json!({ "path": p, "subdir": "x" })),
    ];
    for (tool, a) in asks {
        let e = err(&c, tool, a).await;
        assert!(
            e.ends_with(
                ": on FAT or exFAT, whose file numbers can change while a file is written: x5desk can't \
                 keep track of renders there"
            ),
            "{tool}: {e}"
        );
    }
    // macOS keeps a folder's metadata there in `._x`
    let left: Vec<_> = (files(&on).into_iter())
        .filter(|n| !n.rsplit('/').next().unwrap().starts_with("._"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
    assert_eq!(ok(&c, "job_list", json!({})).await["total"], 0);
    assert!(f.runs.lock().unwrap().is_empty());
    h.shutdown().await;
    fs::remove_dir_all(&on).unwrap();
    fs::remove_dir_all(&d).unwrap();
}

/// A record as an earlier host leaves it: unfinished, into `dst`, its `.part` known by `part`.
fn left(dst: &Path, part: Option<Id>, kind: Kind) -> RJob {
    RJob {
        h: String::new(),
        kind,
        at: SystemTime::now() - Duration::from_secs(60),
        src: vec!["/v/VID_20260117_141148_00_635.insv".into()],
        dst: dst.into(),
        part,
        lane: match kind {
            Kind::Export => Lane::Export,
            Kind::Still => Lane::Still,
        },
        settings: json!({ "preset": "p360" }),
        frames: if kind == Kind::Still { 1 } else { 300 },
        search: 0,
        bytes: 300_000,
        cancelled: false,
        end: None,
        live: None,
    }
}

/// The next host settles what a crashed one left, by identity only: a `.part` that is the recorded file is removed,
/// one that is another file stays; a published file that is the recorded one reads done, also when the `.part` name
/// holds it too (the link fallback's crash: that name goes); a record with no identity unlinks nothing and says so.
/// A download's `.part` beside them stays.
#[tokio::test(flavor = "multi_thread")]
async fn crash_cleanup() {
    let (d, rec) = dirs("crashed");
    let state = d.join("state");
    let a = state.join("agent");
    fs::create_dir_all(&a).unwrap();
    let at = |n: &str| a.join(n);
    let mut r = Renders::load(state.join("mcp-renders")).unwrap().0;
    let mut add = |n: &str, j: RJob| r.add(format!("zz.0001-{n}"), j);
    // 1: its `.part`
    add(
        "1",
        left(
            &at("a.mp4"),
            Some(media::part::create(&at("a.mp4")).unwrap().id),
            Kind::Export,
        ),
    );
    // 2: another file at its `.part`'s name
    let mut other = media::part::create(&at("b.mp4")).unwrap().id;
    other.ino ^= 0x5a5a_5a5a;
    add("2", left(&at("b.mp4"), Some(other), Kind::Export));
    // 3: published
    let c3 = media::part::create(&at("c.jpg")).unwrap();
    (&c3.f).write_all(b"a still").unwrap();
    media::part::publish(&media::part::of(&at("c.jpg")), &at("c.jpg")).unwrap();
    add("3", left(&at("c.jpg"), Some(c3.id), Kind::Still));
    // 4: both names (a crash between the link fallback's link and unlink)
    let c4 = media::part::create(&at("e.mp4")).unwrap();
    (&c4.f).write_all(b"0123456789").unwrap();
    fs::hard_link(media::part::of(&at("e.mp4")), at("e.mp4")).unwrap();
    add("4", left(&at("e.mp4"), Some(c4.id), Kind::Export));
    // 5: no identity recorded (a crash in the µs before it was saved)
    fs::write(media::part::of(&at("f.mp4")), b"").unwrap();
    add("5", left(&at("f.mp4"), None, Kind::Export));
    r.save().unwrap();
    drop((c3, c4));
    fs::write(at("VID_20260116_152935_00_603.insv.part"), b"a download's").unwrap();
    let f = Fake::new();
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let st = |n: &str| {
        let c = &c;
        let j = format!("zz.0001-{n}");
        async move { ok(c, "job_status", json!({ "job": j })).await }
    };
    for (n, s, note) in [
        ("1", "cancelled", GONE),
        ("2", "cancelled", GONE),
        ("5", "cancelled", GONE),
    ] {
        let v = st(n).await;
        assert_eq!(
            (&v["state"], &v["note"]),
            (&json!(s), &json!(note)),
            "{n}: {v}"
        );
    }
    let v = st("3").await;
    assert_eq!(
        (&v["state"], &v["kind"], &v["bytes"]),
        (&json!("done"), &json!("still"), &json!(7))
    );
    let v = st("4").await;
    assert_eq!((&v["state"], &v["bytes"]), (&json!("done"), &json!(10)));
    assert_eq!(
        files(&a),
        [
            "VID_20260116_152935_00_603.insv.part",
            "b.mp4.part",
            "c.jpg",
            "e.mp4",
            "f.mp4.part"
        ]
    );
    let log = fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(
        log.contains("render jobs an earlier host left unfinished: 2 done, 3 cancelled"),
        "{log}"
    );
    assert!(
        log.contains("f.mp4.part: left (no identity recorded)"),
        "{log}"
    );
    h.shutdown().await;
    assert!(records(&state).iter().all(|j| j.end.is_some()));
    fs::remove_dir_all(&d).unwrap();
}

/// A host that dies mid-export (a child process of this test, ended with its renders running, a crash to the files):
/// the next host on its dir settles both of its unfinished jobs by identity, cancelled with no `.part` left.
#[tokio::test(flavor = "multi_thread")]
async fn crash() {
    let (d, rec) = dirs("crash");
    let o = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["crash_child", "--exact", "--nocapture", "--test-threads=1"])
        .env("X5_MCP_CRASH", &d)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&o.stdout) + String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(3), "{said}");
    let state = d.join("state");
    let a = state.join("agent");
    assert_eq!(
        files(&a),
        ["a.mp4.part", "b.mp4.part"],
        "the crash left them"
    );
    let f = Fake::new();
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let l = ok(&c, "job_list", json!({})).await;
    assert_eq!(l["total"], 2);
    for j in l["jobs"].as_array().unwrap() {
        let v = ok(&c, "job_status", json!({ "job": j["job"] })).await;
        assert_eq!(
            (&v["state"], &v["note"]),
            (&json!("cancelled"), &json!(GONE)),
            "{v}"
        );
    }
    assert!(files(&a).is_empty(), "{:?}", files(&a));
    let log = fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(
        log.contains("render jobs an earlier host left unfinished: 0 done, 2 cancelled"),
        "{log}"
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// [`crash`]'s host, in its child process only (`X5_MCP_CRASH`): an export running, another queued, then the process
/// ends without the host's end.
#[tokio::test(flavor = "multi_thread")]
async fn crash_child() {
    let Some(d) = std::env::var_os("X5_MCP_CRASH").map(PathBuf::from) else {
        return;
    };
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 50);
    let p = d.join("rec").join(REC).display().to_string();
    let h = host(&f, &d.join("rec"), "read,local", Some(&d.join("state"))).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "export", json!({ "path": p, "name": "a" })).await;
    ok(&c, "export", json!({ "path": p, "name": "b" })).await;
    frames(&c, &v["job"], 3).await;
    std::process::exit(3)
}

/// 150 finished stills an earlier host kept don't push a download out of the jobs: the renders keep the newest 100 of
/// their own, and the download stays verified for camera_delete.
#[tokio::test(flavor = "multi_thread")]
async fn prune_kinds() {
    let f = Fake::new();
    let (d, rec) = dirs("prune");
    let state = d.join("state");
    let h = host(&f, &rec, "read,local,camera,delete", Some(&state)).await;
    let c = connect(&h, None).await;
    ok(&c, "camera_connect", json!({})).await;
    let l = ok(&c, "album_list", json!({})).await;
    let item = (l["items"].as_array().unwrap().iter())
        .find(|i| i["files"][0] == "IMG_20260116_174103_00_620.jpg")
        .unwrap()["key"]
        .clone();
    let j = ok(&c, "album_download", json!({ "keys": [item] })).await;
    assert_eq!(
        ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 30 })).await["state"],
        "done"
    );
    h.shutdown().await;
    let mut r = Renders::load(state.join("mcp-renders")).unwrap().0;
    for i in 0..150 {
        let mut s = left(&h.agent().join(format!("s{i}.jpg")), None, Kind::Still);
        s.at = SystemTime::now() - Duration::from_secs(1000 - i);
        s.end = Some(End::done(1, 0.5, 1000, 0));
        r.add(format!("zz.0002-{}", i + 1), s);
    }
    r.save().unwrap();
    let h = host(&f, &rec, "read,local,camera,delete", Some(&state)).await;
    let c = connect(&h, None).await;
    let n = |k: &str| {
        let c = &c;
        let k = k.to_string();
        async move { ok(c, "job_list", json!({ "kind": k })).await["total"].clone() }
    };
    assert_eq!(
        (n("still").await, n("download").await),
        (json!(100), json!(1))
    );
    assert_eq!(newest(&c, "still").await["job"], "zz.0002-150");
    ok(&c, "camera_connect", json!({})).await;
    let r = ok(
        &c,
        "camera_delete",
        json!({ "keys": [item], "downloaded_only": true }),
    )
    .await;
    assert_eq!(
        r["deleted"],
        json!([{ "key": item, "local": "verified" }]),
        "{r}"
    );
    h.shutdown().await;
    assert_eq!(records(&state).len(), 100);
    fs::remove_dir_all(&d).unwrap();
}

/// The records and the job tools by kind: a download, an export and a still in one host; job_list newest first, by
/// kind and limit; job_status and job_wait of each kind; a cancel after the end leaves it as it ended; status counts
/// both kinds; the next host has them as they ended, a still's image too.
#[tokio::test(flavor = "multi_thread")]
async fn kinds() {
    let f = Fake::new();
    f.up();
    let (d, rec) = dirs("kinds");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local,camera", Some(&state)).await;
    let c = connect(&h, None).await;
    ok(&c, "camera_connect", json!({})).await;
    let l = ok(&c, "album_list", json!({})).await;
    let item = (l["items"].as_array().unwrap().iter())
        .find(|i| i["files"][0] == "VID_20260117_141148_00_635.insv")
        .unwrap()["key"]
        .clone();
    let dl = ok(
        &c,
        "album_download",
        json!({ "keys": [item], "proxy": true }),
    )
    .await;
    let x = ok(&c, "export", json!({ "path": p, "start_s": 1, "end_s": 3 })).await;
    let s = still_answer(&call(&c, "render_still", json!({ "path": p, "output": "flat" })).await);
    let w = ok(&c, "job_wait", json!({ "job": x["job"], "wait_s": 30 })).await;
    assert_eq!(
        (
            &w["kind"],
            &w["state"],
            &w["frames_done"],
            &w["frames_total"],
            &w["bytes"]
        ),
        (
            &json!("export"),
            &json!("done"),
            &json!(60),
            &json!(60),
            &json!(60_000)
        )
    );
    assert_eq!(
        (&w["settings"]["preset"], &w["settings"]["start_s"]),
        (&json!("p360"), &json!(1.0))
    );
    assert_eq!(
        fs::metadata(w["path"].as_str().unwrap()).unwrap().len(),
        60_000
    );
    assert!(
        w["fps"].as_f64().unwrap() > 0.0 && w["concealed_frames"] == 0,
        "{w}"
    );
    let v = ok(&c, "job_wait", json!({ "job": dl["job"], "wait_s": 30 })).await;
    assert_eq!(
        (&v["kind"], &v["state"]),
        (&json!("download"), &json!("done"))
    );
    let all = [
        (s["job"].clone(), json!("still")),
        (x["job"].clone(), json!("export")),
        (dl["job"].clone(), json!("download")),
    ];
    assert_eq!(list(&c, json!({})).await, all);
    assert_eq!(list(&c, json!({ "kind": "export" })).await, all[1..2]);
    assert_eq!(list(&c, json!({ "kind": "download" })).await, all[2..]);
    assert_eq!(list(&c, json!({ "limit": 1 })).await, all[..1]);
    assert_eq!(
        ok(&c, "status", json!({})).await["jobs"],
        json!({ "running": 0, "queued": 0, "finished": 3 })
    );
    let y = ok(&c, "job_cancel", json!({ "job": s["job"] })).await;
    assert_eq!(
        (&y["state"], &y["cancel_requested"]),
        (&json!("done"), &json!(false))
    );
    assert!(
        err(&c, "job_status", json!({ "job": format!("{}-9", h.id()) }))
            .await
            .contains("unknown job")
    );
    h.shutdown().await;
    // in the order they ended
    let r = records(&state);
    let e = (r.iter().find(|j| j.kind == Kind::Export))
        .and_then(|j| j.end.as_ref())
        .unwrap();
    assert_eq!(
        (r.len(), e.state.as_str(), e.frames, e.bytes),
        (2, "done", 60, 60_000)
    );
    let h = host(&f, &rec, "read,local,camera", Some(&state)).await;
    let c = connect(&h, None).await;
    assert_eq!(list(&c, json!({})).await, all);
    let v = ok(&c, "job_status", json!({ "job": x["job"] })).await;
    assert_eq!((&v["state"], &v["bytes"]), (&json!("done"), &json!(60_000)));
    let r = call(&c, "job_wait", json!({ "job": s["job"] })).await;
    assert_eq!(still_answer(&r)["path"], s["path"]);
    let log = fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(
        log.contains("kept from earlier hosts: 1 jobs, ") && log.contains(", 2 render jobs"),
        "{log}"
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// A start that fails: the parked job fails with why, no `.part`; until its retry is due calls are refused at once,
/// the catalog shows it; once due, the next call starts it again. A start that panics fails its jobs alike, and the
/// host serves on.
#[tokio::test(flavor = "multi_thread")]
async fn gpu_failed() {
    let f = Fake::new();
    f.set(|k| k.start_fails = Some(("no Vulkan device".into(), Some(Duration::from_secs(60)))));
    let (d, rec) = dirs("gpu");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "export", json!({ "path": p })).await;
    let x = reads(&c, &v["job"], "failed").await;
    let why = "GPU tools are unavailable: no Vulkan device (self-test step 1); tried again in 60 s; the other \
               tools work.";
    assert_eq!(x["error"], why);
    assert!(files(h.agent()).is_empty());
    assert_eq!(err(&c, "render_still", json!({ "path": p })).await, why);
    let g = ok(&c, "catalog", json!({})).await["gpu"].clone();
    assert_eq!(
        (&g["state"], &g["why"], &g["step"], &g["retry_in_s"]),
        (
            &json!("failed"),
            &json!("no Vulkan device"),
            &json!(1),
            &json!(60)
        )
    );
    // for good
    f.gpu_is(GpuSt::Failed {
        why: "no Vulkan device".into(),
        step: Some(1),
        retry_in: None,
    });
    let e = err(&c, "export", json!({ "path": p })).await;
    assert_eq!(
        e,
        "GPU tools are unavailable: no Vulkan device (self-test step 1); the other tools work."
    );
    // the retry due: started again, and it works
    f.gpu_is(GpuSt::Failed {
        why: "no Vulkan device".into(),
        step: Some(1),
        retry_in: Some(Duration::ZERO),
    });
    f.set(|k| k.start_fails = None);
    let v = ok(&c, "export", json!({ "path": p })).await;
    assert_eq!(
        ok(&c, "job_wait", json!({ "job": v["job"], "wait_s": 30 })).await["state"],
        "done"
    );
    assert_eq!(f.starts(), 2);
    // a start that panics
    let g = Fake::new();
    g.set(|k| k.start_panics = true);
    let h2 = host(&g, &rec, "read,local", None).await;
    let c2 = connect(&h2, None).await;
    let v = ok(&c2, "render_still", json!({ "path": p, "wait_s": 0 })).await;
    let x = reads(&c2, &v["job"], "failed").await;
    assert_eq!(
        x["error"],
        "GPU tools are unavailable: the GPU backend panicked: the fake's start broke; the other tools work."
    );
    assert!(files(h2.agent()).is_empty());
    assert_eq!(ok(&c2, "status", json!({})).await["jobs"]["finished"], 1);
    h.shutdown().await;
    h2.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// The backend panicking in each of its calls: the call (or the job) fails saying so, nothing is left in the agent
/// folder, and the host's locks are not poisoned (every other tool still answers).
#[tokio::test(flavor = "multi_thread")]
async fn backend_panics() {
    let f = Fake::new();
    f.up();
    let (d, rec) = dirs("panics");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let works = || async {
        ok(&c, "status", json!({})).await;
        ok(&c, "job_list", json!({})).await;
    };
    f.set(|k| k.catalog_panics = true);
    let e = err(&c, "catalog", json!({})).await;
    assert_eq!(e, "the GPU backend panicked: the fake's catalog broke");
    assert_eq!(err(&c, "export", json!({ "path": p })).await, e);
    works().await;
    f.set(|k| (k.catalog_panics, k.gpu_panics) = (false, true));
    assert_eq!(ok(&c, "status", json!({})).await["gpu"]["state"], "failed");
    let e = err(&c, "export", json!({ "path": p })).await;
    assert_eq!(
        e,
        "GPU tools are unavailable: the GPU backend panicked: the fake's gpu broke; the other tools work."
    );
    f.set(|k| (k.gpu_panics, k.check_panics) = (false, true));
    let e = err(&c, "export", json!({ "path": p })).await;
    assert_eq!(e, "the GPU backend panicked: the fake's check broke");
    assert_eq!(ok(&c, "job_list", json!({})).await["total"], 0);
    // a run's state: the job fails; its writer, going on, publishes nothing
    f.set(|k| (k.check_panics, k.frame_ms) = (false, 20));
    let v = ok(&c, "export", json!({ "path": p, "name": "s" })).await;
    f.set(|k| k.state_panics = true);
    let x = reads(&c, &v["job"], "failed").await;
    assert_eq!(
        x["error"],
        "the GPU backend panicked: the fake's state broke"
    );
    f.set(|k| k.state_panics = false);
    assert!(until(10.0, || f.run(0).ended()).await);
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    // a run's cancel: it goes on and finishes, the cancel too late
    f.set(|k| (k.frame_ms, k.cancel_panics) = (2, true));
    let v = ok(&c, "export", json!({ "path": p, "name": "c" })).await;
    ok(&c, "job_cancel", json!({ "job": v["job"] })).await;
    f.set(|k| k.cancel_panics = false);
    let x = reads(&c, &v["job"], "done").await;
    assert_eq!(
        (&x["note"], &x["cancel_requested"]),
        (&json!("finished before the cancel"), &json!(true))
    );
    works().await;
    assert!(ok(&c, "catalog", json!({})).await["gpu"]["state"] == "ready");
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// job_wait on an export tells its progress: at most one notification a second, rising, out of the frames + 1.
#[tokio::test(flavor = "multi_thread")]
async fn progress_rises() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 20);
    let (d, rec) = dirs("progress");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "export", json!({ "path": p })).await;
    let w = ok(&c, "job_wait", json!({ "job": v["job"], "wait_s": 50 })).await;
    assert_eq!(w["state"], "done");
    let pr = c.h.prog.lock().unwrap().clone();
    assert!(pr.len() >= 3, "{pr:?}");
    assert!(pr.windows(2).all(|w| w[1].0 > w[0].0), "rising: {pr:?}");
    assert!(
        pr.iter().all(|p| p.1 == Some(301.0) && p.0 < 301.0),
        "{pr:?}"
    );
    // the next export of that size: its first answer guesses its time from this one's
    let v = ok(&c, "export", json!({ "path": p, "end_s": 5 })).await;
    let n = v["note"].as_str().unwrap();
    assert!(
        n.starts_with("5.00 s, 150 frames at 64 × 32; about ")
            && n.ends_with(" frames/s for that size; job_wait follows it"),
        "{n}"
    );
    ok(&c, "job_cancel", json!({ "job": v["job"] })).await;
    // a blurred one: its pass, then its frames, rising out of the frames to search + the frames + 1 throughout; its
    // note has no earlier pass to time the pass by
    c.h.prog.lock().unwrap().clear();
    let v = ok(
        &c,
        "export",
        json!({ "path": p, "blur": "faces", "name": "b" }),
    )
    .await;
    let n = v["note"].as_str().unwrap();
    assert!(
        n.ends_with(
            " frames/s for that size; a blur pass over 100 source frames comes first (no earlier pass on this server \
             to time it by: roughly 1.6 times the span's length on an Apple M4); job_wait follows it"
        ),
        "{n}"
    );
    let w = ok(&c, "job_wait", json!({ "job": v["job"], "wait_s": 50 })).await;
    assert_eq!(
        (&w["state"], &w["blur"]["frames_searched"]),
        (&json!("done"), &json!(100))
    );
    let pr = c.h.prog.lock().unwrap().clone();
    assert!(pr.len() >= 3, "{pr:?}");
    assert!(pr.windows(2).all(|w| w[1].0 > w[0].0), "rising: {pr:?}");
    assert!(
        pr.iter().all(|p| p.1 == Some(401.0) && p.0 < 401.0),
        "{pr:?}"
    );
    // told the pass first, then the frames
    let said = |m: &str, f: fn(f64) -> bool| {
        (pr.iter()).any(|p| f(p.0) && p.2.as_deref().is_some_and(|x| x.contains(m)))
    };
    assert!(
        said("export: blur pass: ", |x| x < 100.0) && said(" of 300 frames", |x| x > 100.0),
        "{pr:?}"
    );
    // the frames' counting the pass's in full: x = 100 searched + the frames done + a creep
    for p in &pr {
        let Some((done, _)) =
            p.2.as_deref()
                .and_then(|m| m.strip_prefix("export: ")?.split_once(" of 300 frames"))
        else {
            continue;
        };
        assert!(p.0 >= 100.0 + done.parse::<f64>().unwrap(), "{pr:?}");
    }
    // the next blurred one's note times its pass from that one's
    let v = ok(
        &c,
        "export",
        json!({ "path": p, "blur": ["face"], "end_s": 5, "name": "b2" }),
    )
    .await;
    let n = v["note"].as_str().unwrap();
    let (a, b) = n
        .split_once("; first a blur pass over 50 source frames, about ")
        .unwrap_or_else(|| panic!("{n}"));
    assert!(
        a.ends_with(" frames/s for that size")
            && b.ends_with(" frames/s; job_wait follows it")
            && b.contains(" s at this server's last "),
        "{n}"
    );
    ok(&c, "job_cancel", json!({ "job": v["job"] })).await;
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// A blurred export's pass as job_status shows it: the phase, the frames searched of its plan's, the pass's rate and
/// seconds left under `blur` and none at the top, no frame done; held there (its stop ignored), it stays so. Done,
/// what it found and the frames' rate without the pass, kept in the record for the next host. A blurred still answers
/// what it found with its image. Their names say blur.
#[tokio::test(flavor = "multi_thread")]
async fn blur_status() {
    let f = Fake::new();
    f.up();
    f.set(|k| (k.frame_ms, k.stuck_search) = (10, Some(5)));
    let (d, rec) = dirs("blurst");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let a = h.agent().to_path_buf();
    let v = ok(
        &c,
        "export",
        json!({ "path": p, "blur": ["plates", "faces"], "end_s": 4 }),
    )
    .await;
    let dst = a.join("VID_20260117_141148_00_635_blur_p360.mp4");
    assert_eq!(v["path"], dst.display().to_string());
    assert_eq!(
        (
            &v["settings"]["blur"],
            &v["blur"]["targets"],
            &v["blur"]["frames_to_search"],
            &v["frames_total"]
        ),
        (
            &json!(["faces", "plates"]),
            &json!(["faces", "plates"]),
            &json!(40),
            &json!(120)
        )
    );
    let x = when(&c, &v["job"], |x| x["blur"]["frames_searched"] == 5).await;
    let at_top = |x: &Value| {
        (
            x["state"].clone(),
            x["phase"].clone(),
            x["frames_done"].clone(),
            x["fps"].clone(),
            x["eta_s"].clone(),
        )
    };
    let pass = (
        json!("running"),
        json!("detecting"),
        json!(0),
        Value::Null,
        Value::Null,
    );
    assert_eq!(at_top(&x), pass, "{x}");
    let b = &x["blur"];
    assert!(
        b["fps"].as_f64().unwrap() > 0.0
            && b["eta_s"].as_f64().unwrap() > 0.0
            && b["found"].is_null(),
        "{b}"
    );
    assert!(
        b["note"].as_str().unwrap().starts_with("best effort: "),
        "{b}"
    );
    // held 2 s in its pass: no frame comes, the pass stays where it was
    tokio::time::sleep(Duration::from_secs(2)).await;
    let y = ok(&c, "job_status", json!({ "job": v["job"] })).await;
    assert_eq!(
        (at_top(&y), &y["blur"]["frames_searched"]),
        (pass, &json!(5)),
        "{y}"
    );
    f.run(0).go();
    let w = ok(&c, "job_wait", json!({ "job": v["job"], "wait_s": 30 })).await;
    let g = &w["blur"]["found"];
    assert_eq!(
        (
            &w["state"],
            &w["frames_done"],
            &w["blur"]["frames_searched"],
            &g["faces"],
            &g["plates"],
            &g["blurred_frames"],
            &g["frames"]
        ),
        (
            &json!("done"),
            &json!(120),
            &json!(40),
            &json!({ "detections": 40, "tracks": 1 }),
            &json!({ "detections": 20, "tracks": 1 }),
            &json!(120),
            &json!(120)
        ),
        "{w}"
    );
    // the frames took about 1.2 s, the pass over 2: the rate is the frames' alone
    let (fps, pass_s) = (w["fps"].as_f64().unwrap(), g["pass_s"].as_f64().unwrap());
    assert!(fps > 55.0 && pass_s >= 2.0, "{w}");
    assert_eq!(fs::metadata(&dst).unwrap().len(), 120_000);
    // a still: its one frame searched
    let s = still_answer(
        &call(
            &c,
            "render_still",
            json!({ "path": p, "blur": true, "time_s": 1 }),
        )
        .await,
    );
    assert!(
        s["path"]
            .as_str()
            .unwrap()
            .ends_with("_635_blur_1.00s_360_3200.jpg"),
        "{s}"
    );
    assert_eq!(
        (
            &s["blur"]["frames_searched"],
            &s["blur"]["frames_to_search"],
            &s["blur"]["found"]["faces"]["detections"],
            &s["blur"]["found"]["plates"]["detections"]
        ),
        (&json!(1), &json!(1), &json!(1), &json!(1))
    );
    h.shutdown().await;
    let r = records(&state);
    let e = (r.iter()).find(|j| j.kind == Kind::Export).unwrap();
    let found = e.end.as_ref().unwrap().found.unwrap();
    assert_eq!(
        (e.search, found.searched, found.faces, found.plates),
        (40, 40, 40, 20)
    );
    // the next host shows them as they ended
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let x = ok(&c, "job_status", json!({ "job": v["job"] })).await;
    assert_eq!(
        (&x["state"], &x["blur"]["found"], &x["fps"]),
        (&json!("done"), g, &w["fps"]),
        "{x}"
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// job_cancel mid-pass: a blurred export searching reads cancelled within 1 s, never made a frame, found nothing it
/// says, but the frames it searched, left neither its file nor its `.part`; its record says so.
#[tokio::test(flavor = "multi_thread")]
async fn blur_cancel() {
    let f = Fake::new();
    f.up();
    f.set(|k| k.frame_ms = 20);
    let (d, rec) = dirs("blurcancel");
    let p = rec.join(REC).display().to_string();
    let state = d.join("state");
    let h = host(&f, &rec, "read,local", Some(&state)).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "export", json!({ "path": p, "blur": "plates" })).await;
    let x = when(&c, &v["job"], |x| {
        x["blur"]["frames_searched"].as_u64() >= Some(3)
    })
    .await;
    assert_eq!(x["phase"], "detecting", "{x}");
    let t = Instant::now();
    let x = ok(&c, "job_cancel", json!({ "job": v["job"] })).await;
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    assert_eq!(
        (
            &x["state"],
            &x["frames_done"],
            &x["blur"]["found"],
            &x["note"]
        ),
        (&json!("cancelled"), &json!(0), &Value::Null, &Value::Null),
        "{x}"
    );
    let n = x["blur"]["frames_searched"].as_u64().unwrap();
    assert!(
        n >= 3 && n < x["blur"]["frames_to_search"].as_u64().unwrap(),
        "{x}"
    );
    assert_eq!(f.run(0).now(), RunSt::Cancelled);
    assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    h.shutdown().await;
    let j = &records(&state)[0];
    let e = j.end.as_ref().unwrap();
    assert!(
        e.state == "cancelled"
            && e.err.is_empty()
            && j.cancelled
            && e.found.is_none()
            && e.searched >= n,
        "{j:?}"
    );
    fs::remove_dir_all(&d).unwrap();
}

/// The host's end mid-pass: a blurred export searching is cancelled and settled by identity within 1 s, as at a
/// frame; one held in its pass (its stop ignored, as in the detectors' first compile) is waited for `RENDERS`, then
/// settled, its log line saying it was still detecting; let go, it ends on its stop. Neither leaves a file or a
/// `.part`, and the records say the host ended and the frames searched.
#[tokio::test(flavor = "multi_thread")]
async fn blur_end() {
    let (d, rec) = dirs("blurend");
    let p = rec.join(REC).display().to_string();
    for (n, stuck) in [("a", None), ("b", Some(2))] {
        let f = Fake::new();
        f.up();
        f.set(|k| (k.frame_ms, k.stuck_search) = (20, stuck));
        let state = d.join(n);
        let h = host(&f, &rec, "read,local", Some(&state)).await;
        let c = connect(&h, None).await;
        let v = ok(&c, "export", json!({ "path": p, "blur": true })).await;
        when(&c, &v["job"], |x| {
            x["blur"]["frames_searched"].as_u64() >= Some(2)
        })
        .await;
        let t = Instant::now();
        h.shutdown().await;
        let e = t.elapsed();
        let j = v["job"].as_str().unwrap();
        let log = fs::read_to_string(&h.sim().unwrap().log).unwrap();
        let still =
            format!("export {j} still in detecting at the host's end, 8 s after its cancel");
        match stuck {
            None => assert!(
                e < Duration::from_secs(1) && !log.contains(&still),
                "{e:?}\n{log}"
            ),
            Some(_) => assert!(
                e >= mcp::host::RENDERS && e < Duration::from_secs(11) && log.contains(&still),
                "{e:?}\n{log}"
            ),
        }
        assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
        let r = records(&state);
        assert_eq!(
            r[0].end
                .as_ref()
                .map(|e| (e.state.as_str(), e.err.as_str(), e.searched >= 2)),
            Some(("cancelled", ENDED, true))
        );
        let run = f.run(0);
        run.go();
        assert!(until(5.0, || run.ended()).await);
        assert_eq!(run.now(), RunSt::Cancelled);
        assert!(files(h.agent()).is_empty(), "{:?}", files(h.agent()));
    }
    fs::remove_dir_all(&d).unwrap();
}

/// A GPU up without burn: blur is refused with why (its detectors run on burn), `raft: false` no way out and no
/// quiet default to fall back to; the catalog and status say so. A blurred job parked while the GPU started fails at
/// its submit with the same why, after the RAFT re-check it gets too. An unblurred still renders.
#[tokio::test(flavor = "multi_thread")]
async fn blur_no_burn() {
    let f = Fake::new();
    let why = "burn's attach failed: no adapter for it";
    f.set(|k| (k.no_burn, k.start_ms) = (Some(why.into()), 300));
    let (d, rec) = dirs("noburn");
    let p = rec.join(REC).display().to_string();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    let v = ok(&c, "export", json!({ "path": p, "blur": "faces" })).await;
    assert_eq!(v["phase"], "gpu_starting", "{v}");
    let x = reads(&c, &v["job"], "failed").await;
    assert_eq!(x["error"], no_blur(why));
    assert_eq!(x["settings"]["raft"], false, "re-checked without RAFT: {x}");
    for (t, a) in [
        ("render_still", json!({ "path": p, "blur": true })),
        (
            "export",
            json!({ "path": p, "blur": ["plates"], "raft": false }),
        ),
    ] {
        assert_eq!(err(&c, t, a.clone()).await, no_blur(why), "{t} {a}");
    }
    let v = ok(&c, "catalog", json!({})).await;
    let off = json!({ "available": false, "why": why });
    assert_eq!(
        (
            &v["blur"]["available"],
            &v["blur"]["why"],
            &v["gpu"]["blur"],
            &v["gpu"]["raft"]
        ),
        (&json!(false), &json!(why), &off, &off)
    );
    assert_eq!(ok(&c, "status", json!({})).await["gpu"]["blur"], off);
    let s = still_answer(&call(&c, "render_still", json!({ "path": p })).await);
    assert!(s.get("blur").is_none(), "{s}");
    assert_eq!(f.runs.lock().unwrap().len(), 1, "only the still ran");
    assert_eq!(
        files(h.agent()),
        ["VID_20260117_141148_00_635_0.00s_360_3200.jpg"]
    );
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}

/// The marker the host knows as a secret, passed through every argument the GPU tools take (the path, name, subdir,
/// look, blur, adjust keys, fps, codec) and echoed by the backend's errors, never reaches a client's byte or the log.
#[tokio::test(flavor = "multi_thread")]
async fn scrub_render() {
    let f = Fake::new();
    f.up();
    let (d, rec) = dirs("scrub");
    let m = sim::MARKER;
    let p = rec.join(REC).display().to_string();
    let mp = rec.join(format!("{m}.insv"));
    fs::write(&mp, "frames=30 imu=1").unwrap();
    let h = host(&f, &rec, "read,local", None).await;
    let c = connect(&h, None).await;
    for (t, a) in [
        ("export", json!({ "path": p, "name": m })),
        ("render_still", json!({ "path": p, "subdir": m })),
        ("render_still", json!({ "path": mp })),
        ("export", json!({ "path": p, "look": m })),
        ("export", json!({ "path": p, "blur": m })),
        ("render_still", json!({ "path": p, "blur": ["faces", m] })),
        ("export", json!({ "path": p, "adjust": { m: 1 } })),
        ("export", json!({ "path": p, "fps": m })),
        ("export", json!({ "path": p, "codec": m })),
        ("export", json!({ "path": p, "name": m })),
        ("export", json!({ "path": format!("{p}{m}") })),
        ("job_status", json!({ "job": m })),
    ] {
        let _ = call(&c, t, a).await;
    }
    f.set(|k| k.echo = true);
    let e = err(&c, "export", json!({ "path": mp })).await;
    assert!(
        e.starts_with("the fake refuses") && e.contains("[redacted]"),
        "{e}"
    );
    f.set(|k| k.echo = false);
    for t in ["job_list", "status"] {
        ok(&c, t, json!({})).await;
    }
    let l = ok(&c, "job_list", json!({})).await["jobs"].clone();
    for j in l.as_array().unwrap() {
        let _ = call(&c, "job_wait", json!({ "job": j["job"], "wait_s": 10 })).await;
    }
    assert!(
        files(h.agent()).iter().any(|x| x.contains(m)),
        "the files have the name"
    );
    let got = c.got.lock().unwrap().clone();
    assert!(
        !got.windows(m.len()).any(|w| w == m.as_bytes()),
        "a client's byte"
    );
    assert!(String::from_utf8_lossy(&got).contains("[redacted]"));
    let log = fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(log.contains("[redacted]") && !log.contains(m), "{log}");
    h.shutdown().await;
    fs::remove_dir_all(&d).unwrap();
}
