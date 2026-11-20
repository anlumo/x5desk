//! The agents' render engine: crate mcp's [`Render`] for `x5desk mcp`'s `catalog`, `render_still` and `export`, over
//! this screen's own tables and jobs. One per process ([`ENG`], never dropped: dropping an `export::Queue` joins its
//! thread, which may be in the driver), its GPU opened at the first job and kept until the process ends. A child
//! (`x5desk mcp gpu-check`, [`Agent::selftest`]) checks the device first, so a driver that crashes or hangs at its
//! first init, or decodes wrongly, takes the child down instead of the host and its camera session. Two lanes, each an
//! `export::Queue` on the one device: stills, and exports, so a still never waits behind an export. A job is the one
//! the Export screen would queue (`set_of` sets what its widgets would, `span` and `clip_job` build it), written into
//! the `.part` mcp reserved and strict: a picture that could not be decoded fails an export unless `allow_frozen`.
//! A blurred job runs the screen's privacy pass first on its lane (`Job::pre`), the models held per searched frame.
//! `X5DESK_TEST=mcp-noattach` keeps burn off the device, as a failed attach would: RAFT and blur are off.

use super::{
    ABR, CANCELLED, FLAT, FPS, H264_MAX, H265_MAX, Hold, PANO, PRESETS, STEP, Set, Spec, clip_job,
    hook, keys, privacy_pass, probe, span, stem_tag,
};
use crate::{editor::doc, gpu, gyro, lab, player};
use ::export::Dest;
use mcp::render::{
    self as r, Catalog, GpuInfo, GpuSt, Kind, Lane, Output, Phase, Plan, Render, Req, Run, RunSt,
    Start, Want,
};
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read},
    os::unix::process::ExitStatusExt,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError,
        atomic::{
            AtomicBool, AtomicU32, AtomicU64,
            Ordering::{Acquire, Relaxed, Release},
        },
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use stitch::View;
use video::{Codec, Gpu, check};

/// The process's one engine: never dropped, so no lane is ever joined.
static ENG: OnceLock<Arc<Agent>> = OnceLock::new();

/// How long the self-test child may take, how often it is looked at, its stderr kept for the log at most.
const CHILD: Duration = Duration::from_secs(20);
const TICK: Duration = Duration::from_millis(50);
const STDERR: usize = 16 << 10;
/// A start that failed before the process had a device is tried again after the first, then after these.
const RETRY: [Duration; 2] = [Duration::from_secs(60), Duration::from_secs(600)];
/// Self-test crashes (a signal, a hang) after which GPU tools are off for the process.
const CRASHES: u32 = 3;
/// The probe's readback within this.
const PROBE: Duration = Duration::from_secs(5);
/// `Agent::drv` once the process ends: no probe or trim starts.
const ENDED: u32 = u32::MAX;
/// The X5's lens side (the H.265 check's), the pixel check's side.
const X5: u32 = 3840;
const PCM: u32 = 256;
/// A still's defaults, preset-free: 360 at this width, a flat size, the JPEG quality.
const STILL_360: u32 = 7680;
const STILL_FLAT: [u32; 2] = [1920, 1080];
const JQ: u8 = 92;
/// Times on a clip are this close to its end, s.
const EPS: f64 = 1e-3;
/// A QP export's bits a pixel at QP 24 at most, 360 and flat: the X5's footage at the smallest sizes ([`PANO`],
/// [`FLAT`]), where a pixel carries most (H.264 at 29.97 fps: 360 up to 0.64, flat up to 1.56, a planet), with room.
/// A bigger 360's pixel carries less (0.38 at 5760, 0.30 at 7680): its bits grow as the square root of its pixels. A
/// flat one's need not (a planet at 3840 × 2160: 0.65). Twice as many every 6 QP lower, half as many every 6 higher.
const QP24: [f64; 2] = [1.0, 1.75];
/// A set rate's overshoot at most (VideoToolbox's at 1 Mbit/s over 3 s: 6 %) and the container.
const OVER: f64 = 1.1;

/// Ids (stable API) of the codecs, with whether they do 10-bit and their long edge at most.
const CODECS: [(Codec, &str, bool, u32); 2] = [
    (Codec::H265, "h265", true, H265_MAX),
    (Codec::H264, "h264", false, H264_MAX),
];
/// Look ids in `fx::Look::ALL`'s order; the first word of the name (the test hook's) is an alias where it differs.
const LOOKS: [&str; 6] = ["vivid", "warm", "cool", "film", "mono", "teal_orange"];
/// Stabilisation ids, `gyro::parse`'s, in `gyro::MODES`' order.
const STAB: [&str; 5] = ["off", "free", "horizon", "direction", "forward"];
/// The flat projections (`player::Look::parse`'s ids, the project files' `persp` too): the 360 output is the equirect.
const VIEWS: [(&str, &[&str], player::Look); 3] = [
    ("perspective", &["persp"], player::Look::Perspective),
    ("panini", &[], player::Look::Panini),
    ("planet", &[], player::Look::Planet),
];
/// Blur targets (stable API): id, name, aliases, in `Set::faces`, `Set::plates` order.
const BLUR: [(&str, &str, &[&str]); 2] = [
    ("faces", "Faces", &["face"]),
    (
        "plates",
        "Licence plates",
        &["plate", "licence_plates", "license_plates", "number_plates"],
    ),
];
/// What agents can't do yet: id, name, the argument names that ask for it (Shot Lab's effects are `lab::FXS`').
const LATER: [(&str, Option<&str>, &[&str]); 6] = [
    ("follow", Some("Deep Track follow"), &["track"]),
    ("photos", None, &[]),
    ("cube_luts", None, &["lut"]),
    ("overlays", None, &["overlay", "watermark"]),
    ("editor_timelines", None, &[]),
    ("shot_lab", None, &["effect"]),
];

/// The engine: its state, the device once open, what the device reported.
pub struct Agent {
    st: Mutex<Eng>,
    /// `Starting` ended
    cv: Condvar,
    /// the device and its lanes once open, held here for good (a `Dead` engine's too)
    dev: OnceLock<Arc<Ready>>,
    /// the device's log, errors and suspicion (the newest start's sink)
    w: Arc<gpu::Watch>,
    /// starts that failed before the process had a device, and the self-test children that crashed or hung
    fails: AtomicU32,
    crashes: AtomicU32,
    /// the lanes' device-poll errors seen (`Queue::lost`): a new one makes the GPU suspect
    lost: Mutex<[Option<String>; 2]>,
    /// driver work outside the lanes running now (probes, trims); [`ENDED`] once [`busy`] found none
    drv: AtomicU32,
    cat: OnceLock<Catalog>,
    /// `X5DESK_TEST=mcp-noattach`: burn never attaches (RAFT off)
    noattach: bool,
}

enum Eng {
    Off,
    Starting,
    Ready(Arc<Ready>),
    Failed {
        why: String,
        step: Option<u8>,
        at: Instant,
        reopen: Reopen,
    },
    Dead(String),
}

enum Reopen {
    After(Duration),
    Never,
}

/// The device, its info at the start, the lanes; whether a RAFT flow ran (burn's kernels compiled).
struct Ready {
    g: Arc<Gpu>,
    info: GpuInfo,
    still: LaneQ,
    export: LaneQ,
    warm: AtomicBool,
}

struct LaneQ {
    q: ::export::Queue,
    /// why it is unusable for the process
    wedged: Mutex<Option<String>>,
}

impl Ready {
    fn lane(&self, l: Lane) -> &LaneQ {
        match l {
            Lane::Still => &self.still,
            Lane::Export => &self.export,
        }
    }

    /// The wedged lane stuck in a detection, if one is: it holds the models (and BURN), so blur and RAFT are refused.
    fn stuck(&self) -> Option<Lane> {
        let h = crate::ai::MODELS.holder()?;
        let at = |l: &LaneQ| l.q.thread() == Some(h) && lock(&l.wedged).is_some();
        [Lane::Still, Lane::Export]
            .into_iter()
            .find(|&l| at(self.lane(l)))
    }
}

/// How a start failed: halted (the host ends), or at a step of the self-test (a crash of its child counts).
enum Fail {
    Halted,
    At {
        step: Option<u8>,
        why: String,
        crash: bool,
    },
}

impl Fail {
    fn at(step: Option<u8>, why: String) -> Fail {
        Fail::At {
            step,
            why,
            crash: false,
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Agent {
    /// The process's engine (made on first use: it only allocates; the GPU starts with the first job).
    pub fn headless() -> Arc<Agent> {
        ENG.get_or_init(|| {
            let test = std::env::var("X5DESK_TEST").unwrap_or_default();
            Arc::new(Agent::new(
                test.split(',').any(|w| w.trim() == "mcp-noattach"),
            ))
        })
        .clone()
    }

    fn new(noattach: bool) -> Agent {
        Agent {
            st: Mutex::new(Eng::Off),
            cv: Condvar::new(),
            dev: OnceLock::new(),
            w: Arc::new(gpu::Watch::default()),
            fails: AtomicU32::new(0),
            crashes: AtomicU32::new(0),
            lost: Mutex::new([None, None]),
            drv: AtomicU32::new(0),
            cat: OnceLock::new(),
            noattach,
        }
    }

    /// Whether driver work may still run: a probe or a trim, the engine starting, a lane with a job or a trim (a
    /// `Dead` engine's lanes too: a probe that failed leaves them in the driver). Once it has looked, no probe or trim
    /// starts (one that ran before sent its lanes' trims first).
    fn busy(&self) -> bool {
        (self.drv.compare_exchange(0, ENDED, Acquire, Relaxed)).is_err()
            || matches!(*lock(&self.st), Eng::Starting)
            || (self.dev.get()).is_some_and(|rd| rd.still.q.busy() || rd.export.q.busy())
    }

    /// `f`, driver work outside the lanes, counted by [`Agent::busy`]; None once the process ends. A panic in it
    /// leaves it counted (`_exit`).
    fn driver<T>(&self, f: impl FnOnce() -> T) -> Option<T> {
        let up = |n| (n != ENDED).then(|| n + 1);
        self.drv.fetch_update(Relaxed, Relaxed, up).ok()?;
        let r = f();
        self.drv.fetch_sub(1, Release);
        Some(r)
    }

    /// `rd`'s info as it is now: uncaptured errors, suspicion (a lane's new device-poll error, or its thread gone,
    /// too), the lanes wedged or gone, blur refused while one of them is stuck in a detection.
    fn info(&self, rd: &Ready) -> GpuInfo {
        let mut i = rd.info.clone();
        let mut seen = lock(&self.lost);
        for (k, l) in [&rd.still, &rd.export].into_iter().enumerate() {
            let x = match l.q.alive() {
                true => l.q.lost(),
                false => Some("the lane's thread ended".into()),
            };
            if x.is_some() && x != seen[k] {
                self.w.suspect.store(true, Relaxed);
                seen[k] = x;
            }
        }
        i.errors = self.w.errors.load(Relaxed);
        i.suspect = self.w.suspect.load(Relaxed);
        i.wedged = [Lane::Still, Lane::Export]
            .into_iter()
            .filter(|&l| lock(&rd.lane(l).wedged).is_some() || !rd.lane(l).q.alive())
            .collect();
        i.blur_stuck = rd.stuck();
        i
    }

    fn ready(&self) -> Option<Arc<Ready>> {
        match &*lock(&self.st) {
            Eng::Ready(rd) => Some(rd.clone()),
            _ => None,
        }
    }

    /// The start: the child's self-test, then the device in process with a host's error policy, the pipelines, the
    /// caps (the child's when it ran on this adapter, else steps 3 and 5 here), burn attached, the lanes. `opened`: the
    /// process has a device from here on (a failure after that is final).
    fn open(&self, s: &Start, opened: &AtomicBool) -> Result<Arc<Ready>, Fail> {
        let t0 = Instant::now();
        let kid = match &s.child {
            Some(exe) => Some(child(exe, &s.halt, &s.log)?),
            None => None,
        };
        if s.halt.load(Relaxed) {
            return Err(Fail::Halted);
        }
        let p = gpu::Policy::Log(self.w.clone());
        let (g, a) = gpu::open(s.cubecl.as_deref(), p, !self.noattach)
            .map_err(|e| Fail::at(Some(1), hint(e)))?;
        opened.store(true, Relaxed);
        let cx = ::export::scoped(&g.device, || Ok(stitch::Ctx::new(&g.device, &g.queue)))
            .map_err(|e| Fail::at(Some(2), e))?;
        let mut info = base(&g);
        let same = |k: &GpuInfo| {
            (k.vendor, k.device, &k.driver) == (info.vendor, info.device, &info.driver)
        };
        (info.decode, info.encode) = match kid.filter(same) {
            Some(k) => (k.decode, k.encode),
            None => {
                let say = |l: String| (s.log)(l);
                let d = decode(&g, &say).map_err(|(n, w)| Fail::at(Some(n), w))?;
                (d, encode(&g, &say))
            }
        };
        info.raft = a.err();
        let lane = |n: &'static str| {
            ::export::Queue::with(g.clone(), cx.clone(), n)
                .map(|q| LaneQ {
                    q,
                    wedged: Mutex::new(None),
                })
                .map_err(|e| Fail::at(None, format!("the GPU lane {n}: {e}")))
        };
        let (still, export) = (lane("x5-still")?, lane("x5-export")?);
        info.ms = t0.elapsed().as_millis() as u64;
        Ok(Arc::new(Ready {
            g,
            info,
            still,
            export,
            warm: AtomicBool::new(false),
        }))
    }

    /// The default seam: RAFT, unless the GPU is up without it (burn did not attach) or a lane stuck in a detection
    /// holds burn.
    fn raft_ok(&self) -> bool {
        (self.ready()).is_none_or(|rd| rd.info.raft.is_none() && rd.stuck().is_none())
    }
}

/// Whether a probe or trim runs, the engine is starting or a lane has a job or a trim ([`Agent::busy`]): the process
/// then ends with `_exit` (no atexit, no driver teardown while a thread is in the driver). Asked once, as it ends.
pub fn busy() -> bool {
    ENG.get().is_some_and(|a| a.busy())
}

/// `e` from `Gpu::new`, with a hint where Linux finds no Vulkan device at all.
fn hint(e: String) -> String {
    match cfg!(target_os = "linux") && e.contains("No suitable graphics adapter") {
        true => hint_in(e, Path::new("/dev/dri")),
        false => e,
    }
}

/// [`hint`]'s text from the render nodes in `dri`: none (no GPU driver), none this user may open, or open ones.
fn hint_in(e: String, dri: &Path) -> String {
    let nodes: Vec<PathBuf> = (std::fs::read_dir(dri).into_iter().flatten().flatten())
        .map(|d| d.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("renderD"))
        })
        .collect();
    let open = |p: &PathBuf| {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(p)
            .is_ok()
    };
    let why = match nodes.iter().any(open) {
        _ if nodes.is_empty() => {
            "no /dev/dri/renderD*: no GPU driver here (a VM or container needs the GPU passed through)"
        }
        false => "is this user in the `render` group (/dev/dri/renderD*)?",
        true => "is a Vulkan driver (Mesa's, NVIDIA's) installed, with the Vulkan loader?",
    };
    format!("{e}; no Vulkan device: {why}")
}

/// What the device is, before its caps: the adapter, the driver, the texture limit, the variables that choose it.
fn base(g: &Gpu) -> GpuInfo {
    let a = g.adapter.get_info();
    GpuInfo {
        adapter: a.name,
        backend: a.backend.to_str().into(),
        driver: format!("{} {}", a.driver, a.driver_info).trim().into(),
        vendor: a.vendor,
        device: a.device,
        max_texture: g.device.limits().max_texture_dimension_2d,
        env: env(),
        ..Default::default()
    }
}

fn codec_id(c: Codec) -> &'static str {
    CODECS.iter().find(|k| k.0 == c).map_or("", |k| k.1)
}

/// A profile's caps as mcp reads them.
fn cap(c: &check::Cap) -> r::Cap {
    r::Cap {
        codec: codec_id(c.codec).into(),
        ten_bit: c.deep,
        min: c.min,
        max: c.max,
        rate: c.rate.iter().map(|s| s.to_string()).collect(),
        max_mbps: c.bps.map(|b| (b as f64 / 1e6) as f32),
        qp: c.qp,
    }
}

fn profile(codec: Codec, deep: bool) -> String {
    format!("{}{}", codec_id(codec), if deep { " 10-bit" } else { "" })
}

/// Self-test step 3: the X5's H.265 at its size; what the device decodes per profile.
fn decode(g: &Gpu, say: &dyn Fn(String)) -> Result<Vec<r::Cap>, (u8, String)> {
    ::export::scoped(&g.device, || check::h265(g, X5, false)).map_err(|e| {
        (
            3,
            format!("this GPU cannot decode the X5's {X5} × {X5} H.265: {e}"),
        )
    })?;
    let d = check::caps(g).0;
    for c in d.iter().filter(|c| c.fmt == Some(false)) {
        say(format!(
            "decode {}: the driver lists no NV12/P010 output (decoding is tried anyway)",
            profile(c.codec, c.deep)
        ))
    }
    Ok(d.iter().map(cap).collect())
}

/// Self-test step 5: the encode profiles that put a picture through (none: stills only).
fn encode(g: &Gpu, say: &dyn Fn(String)) -> Vec<r::Cap> {
    (check::caps(g).1.iter())
        .filter(
            |c| match ::export::scoped(&g.device, || check::encode(g, c.codec, c.deep)) {
                Ok(()) => true,
                Err(e) => {
                    say(format!("encode {}: {e}", profile(c.codec, c.deep)));
                    false
                }
            },
        )
        .map(cap)
        .collect()
}

/// The self-test in a child, `<exe> mcp gpu-check`: its stdin a pipe (it ends at its EOF, so a host that is killed
/// takes it along), its stdout's one line read back, its stderr (16 KiB of it) and its pass into `log`; killed after
/// [`CHILD`] or once `halt` is set.
fn child(
    exe: &Path,
    halt: &AtomicBool,
    log: &Arc<dyn Fn(String) + Send + Sync>,
) -> Result<GpuInfo, Fail> {
    let mut c = Command::new(exe)
        .args(["mcp", "gpu-check"])
        .current_dir(std::env::temp_dir())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Fail::at(None, format!("the GPU check {}: {e}", exe.display())))?;
    let (out, err) = (c.stdout.take(), c.stderr.take());
    let (tx, rx) = mpsc::channel();
    let _ = thread::Builder::new()
        .name("x5-gpu-out".into())
        .spawn(move || {
            let mut s = String::new();
            if let Some(o) = out {
                let _ = o.take(1 << 20).read_to_string(&mut s);
            }
            let _ = tx.send(s);
        });
    let l = log.clone();
    let _ = thread::Builder::new()
        .name("x5-gpu-err".into())
        .spawn(move || {
            let Some(e) = err else { return };
            let (mut left, mut more) = (STDERR, 0);
            for x in BufReader::new(e).split(b'\n').map_while(Result::ok) {
                match x.len() < left {
                    true => {
                        left -= x.len() + 1;
                        l(format!("gpu-check: {}", String::from_utf8_lossy(&x)))
                    }
                    false => more += x.len() + 1,
                }
            }
            if more > 0 {
                l(format!("gpu-check: {more} bytes more of its stderr"))
            }
        });
    let t0 = Instant::now();
    let st = loop {
        match c.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if halt.load(Relaxed) || t0.elapsed() >= CHILD => {
                let _ = c.kill();
                let _ = c.wait();
                match halt.load(Relaxed) {
                    true => return Err(Fail::Halted),
                    false => break None,
                }
            }
            Ok(None) => thread::sleep(TICK),
            Err(e) => {
                let _ = c.kill();
                let _ = c.wait();
                return Err(Fail::at(None, format!("the GPU check: {e}")));
            }
        }
    };
    let line = rx.recv_timeout(Duration::from_secs(2)).unwrap_or_default();
    let crash = |why: String| Fail::At {
        step: None,
        why,
        crash: true,
    };
    match st {
        None => Err(crash(format!(
            "the GPU check hung (no answer in {} s)",
            CHILD.as_secs()
        ))),
        Some(s) if s.signal().is_some() => Err(crash(format!("the GPU check crashed ({s})"))),
        Some(_) => {
            let last = line.lines().rfind(|l| !l.trim().is_empty()).unwrap_or("");
            let i = mcp::cli::gpu_checked(last).map_err(|(step, why)| Fail::at(step, why))?;
            log(format!(
                "gpu-check: passed in {} ms",
                t0.elapsed().as_millis()
            ));
            Ok(i)
        }
    }
}

impl Render for Agent {
    fn catalog(&self) -> Catalog {
        self.cat.get_or_init(catalog).clone()
    }

    fn check(&self, q: &Req) -> Result<Plan, String> {
        let still = q.kind == Kind::Still;
        let src = ::export::source(&q.files, false)?;
        // blur: refused where its pass can't run, before the GPU starts, never dropped; the clip as the pass reads it
        let clip = match q.want.blur.is_empty() {
            true => None,
            false if q.files.len() > 1 => {
                return Err("blur: a recording in a file per lens (_00_ + _10_) is not supported yet; leave out \
                            blur"
                    .into());
            }
            false if self.noattach => return Err(r::no_blur(gpu::NOATTACH)),
            false => match self.dev.get().and_then(|rd| rd.stuck()) {
                Some(l) => return Err(r::stuck_blur(l)),
                None => Some(shotlab::Clip::open(&q.files[0]).map_err(|e| format!("blur: {e}"))?),
            },
        };
        // the pass stabilises with shotlab's reading of the IMU
        let no_imu = clip.as_ref().is_some_and(|c| c.imu.is_none());
        // horizon when the recording can be stabilised, unless asked otherwise
        let given = q.want.stabilisation.as_deref();
        let stab = match given.unwrap_or(if src.imu { "horizon" } else { "off" }) {
            "off" => "off",
            s => match ::export::source(&q.files, true) {
                Ok(_) if no_imu && given.is_none() => "off",
                Ok(_) if no_imu => {
                    return Err(
                        "blur: no IMU record to stabilise with: use stabilisation off".into(),
                    );
                }
                Ok(_) => s,
                Err(_) if given.is_none() => "off",
                Err(e) => return Err(format!("{e}: use stabilisation off")),
            },
        };
        let mut w = q.want.clone();
        w.stabilisation = Some(stab.into());
        let (rate, dur) = probe(&q.files[0])?;
        let (x, v, proj) = set_of(&w, q.kind, self.raft_ok())?;
        let fps = x.rate(rate);
        let asked = range(&w, q.kind, fps, dur)?;
        let spec = Spec {
            files: q.files.clone(),
            photo: false,
            dur,
            t: asked[0],
            view: v,
            look: player::Look::Perspective,
            stab: x.stab,
            smooth: x.smooth,
            follow: None,
            faces: false,
            plates: false,
        };
        if let Some(e) = x.check(&spec, still, asked) {
            return Err(e);
        }
        let ab = span(asked, still, fps, dur);
        // the source frames its pass searches (the pass's own `keys`)
        let search = clip.map_or(0, |c| keys(&c.times, ab).2.len() as u64);
        let at = move |_| v;
        let (_, frames) = clip_job(
            &q.files,
            dur,
            rate,
            &x,
            ab,
            still,
            at,
            PathBuf::new(),
            None,
            false,
        );
        let stem = (q.files[0].file_stem()).map_or("x5".into(), |n| n.to_string_lossy());
        let [pw, ph] = x.size();
        let secs = ab[1] - ab[0];
        // a blurred file is never taken for the sharp one
        let pre = match x.faces || x.plates {
            true => "blur_",
            false => "",
        };
        Ok(Plan {
            frames,
            secs,
            size: [pw, ph],
            codec: (!still).then(|| codec_id(x.codec).into()),
            ten_bit: !still && x.deep && x.codec == Codec::H265,
            raft: x.raft,
            blur: blur_ids(&x),
            search,
            mbps: (!still && x.qp.is_none()).then_some(x.mbps),
            qp: x.qp.filter(|_| !still),
            tracks: (src.tracks.iter())
                .map(|t| r::Track {
                    codec: codec_id(t.codec).into(),
                    ten_bit: t.deep,
                    size: t.size,
                })
                .collect(),
            stem: stem_tag(&stem, &x, still.then_some(asked[0]), pre),
            ext: if still { "jpg" } else { "mp4" },
            bytes: match still {
                true => pw as u64 * ph as u64 / 2,
                false => video_bytes(&x, frames, secs),
            },
            settings: settings(&x, &v, proj, &w, q.kind, ab, asked[0]),
        })
    }

    fn start(&self, s: &Start) -> Result<GpuInfo, String> {
        *lock(&self.w.sink) = s.log.clone();
        let mut st = lock(&self.st);
        loop {
            match &*st {
                Eng::Ready(rd) => return Ok(self.info(rd)),
                Eng::Starting => st = self.cv.wait(st).unwrap_or_else(PoisonError::into_inner),
                Eng::Dead(w) => return Err(w.clone()),
                Eng::Failed {
                    why,
                    reopen: Reopen::Never,
                    ..
                } => return Err(why.clone()),
                Eng::Failed {
                    why,
                    at,
                    reopen: Reopen::After(d),
                    ..
                } if at.elapsed() < *d => return Err(why.clone()),
                _ => break,
            }
        }
        *st = Eng::Starting;
        drop(st);
        let opened = AtomicBool::new(false);
        let got = catch_unwind(AssertUnwindSafe(|| self.open(s, &opened))).unwrap_or_else(|p| {
            let why = format!("the GPU's start panicked: {}", r::panicked(&*p));
            Err(Fail::at(None, why))
        });
        let set = |e: &[String]| match e.is_empty() {
            true => String::new(),
            false => format!("; set: {}", e.join(" ")),
        };
        let (next, out) = match got {
            Ok(rd) => {
                let i = &rd.info;
                let raft = i
                    .raft
                    .as_ref()
                    .map_or("on".into(), |w| format!("off ({w})"));
                let driver = match i.driver.is_empty() {
                    true => String::new(),
                    false => format!(", driver {}", i.driver),
                };
                (s.log)(format!(
                    "the GPU is up in {} ms: {} ({}{driver}){}; RAFT {raft}",
                    i.ms,
                    i.adapter,
                    i.backend,
                    set(&i.env)
                ));
                let _ = self.dev.set(rd.clone());
                (Eng::Ready(rd.clone()), Ok(self.info(&rd)))
            }
            Err(Fail::Halted) => (
                Eng::Off,
                Err("the GPU's start halted: the host ends".into()),
            ),
            Err(Fail::At { step, why, crash }) => {
                let reopen = match opened.load(Relaxed) {
                    // a second device and attach would leak burn's runtime: final for the process
                    true => Reopen::Never,
                    false => {
                        let n = self.fails.fetch_add(1, Relaxed) as usize;
                        let c = self.crashes.fetch_add(crash as u32, Relaxed) + crash as u32;
                        match c >= CRASHES {
                            true => Reopen::Never,
                            false => Reopen::After(RETRY[n.min(1)]),
                        }
                    }
                };
                let n = step.map_or(String::new(), |n| format!(" at self-test step {n}"));
                (s.log)(format!("the GPU did not start{n}: {why}{}", set(&env())));
                let why = match (crash, &reopen) {
                    (true, Reopen::Never) => format!(
                        "the GPU check crashed {CRASHES} times (see ReportCrash or core dumps): GPU tools are off \
                         until this MCP server restarts"
                    ),
                    _ => why,
                };
                let f = Eng::Failed {
                    why: why.clone(),
                    step,
                    at: Instant::now(),
                    reopen,
                };
                (f, Err(why))
            }
        };
        *lock(&self.st) = next;
        self.cv.notify_all();
        out
    }

    fn gpu(&self) -> GpuSt {
        match &*lock(&self.st) {
            Eng::Off => GpuSt::Off,
            Eng::Starting => GpuSt::Starting,
            Eng::Ready(rd) => GpuSt::Ready(self.info(rd)),
            Eng::Failed {
                why,
                step,
                at,
                reopen,
            } => GpuSt::Failed {
                why: why.clone(),
                step: *step,
                retry_in: match reopen {
                    Reopen::After(d) => Some(d.saturating_sub(at.elapsed())),
                    Reopen::Never => None,
                },
            },
            Eng::Dead(w) => GpuSt::Dead(w.clone()),
        }
    }

    fn selftest(&self) -> Result<GpuInfo, (u8, String)> {
        let t0 = Instant::now();
        let say = |l: String| eprintln!("x5desk mcp gpu-check: {l}");
        let g = Gpu::new().map_err(|e| (1, hint(e.to_string())))?;
        ::export::scoped(&g.device, || Ok(stitch::Ctx::new(&g.device, &g.queue)))
            .map_err(|e| (2, e))?;
        let dec = decode(&g, &say)?;
        match ::export::scoped(&g.device, || check::pcm(&g, PCM)) {
            Ok(None) => {}
            Ok(Some(w)) => say(format!("the pixel check skipped: {w}")),
            Err(e) => return Err((4, format!("the decoder's pictures are wrong: {e}"))),
        }
        let mut i = base(&g);
        (i.decode, i.encode) = (dec, encode(&g, &say));
        i.ms = t0.elapsed().as_millis() as u64;
        Ok(i)
    }

    fn probe(&self) -> Result<(), String> {
        let rd = match &*lock(&self.st) {
            Eng::Ready(rd) => rd.clone(),
            Eng::Dead(w) => return Err(w.clone()),
            _ => return Ok(()),
        };
        let Some(r) = self.driver(|| ::export::scoped(&rd.g.device, || sound(&rd.g))) else {
            return Err("not probed: the process ends".into());
        };
        match r {
            Ok(()) => {
                self.w.suspect.store(false, Relaxed);
                Ok(())
            }
            Err(e) => {
                let mut st = lock(&self.st);
                if let Eng::Ready(_) = &*st {
                    *st = Eng::Dead(e.clone());
                }
                Err(e)
            }
        }
    }

    fn submit(
        &self,
        q: Req,
        p: &Plan,
        dst: &Path,
        part: media::part::Part,
        lane: Lane,
    ) -> Result<Arc<dyn Run>, String> {
        let Some(rd) = self.ready() else {
            return Err("the GPU is not up".into());
        };
        let l = rd.lane(lane);
        if let Some(w) = &*lock(&l.wedged) {
            return Err(w.clone());
        }
        if !l.q.alive() {
            return Err(format!("the GPU's {} lane's thread ended", lane.name()));
        }
        let still = q.kind == Kind::Still;
        // as it was checked: the stabilisation its source allowed, the seam
        let mut w = q.want;
        let checked = p.settings["stabilisation"].as_str().map(String::from);
        (w.stabilisation, w.raft) = (checked.or(w.stabilisation), Some(p.raft));
        let (rate, dur) = probe(&q.files[0])?;
        let (x, v, _) = set_of(&w, q.kind, p.raft)?;
        // mcp's device check refuses it first
        if let (true, Some(why)) = (x.faces || x.plates, &rd.info.raft) {
            return Err(r::no_blur(why));
        }
        if let (true, Some(l)) = (x.faces || x.plates, rd.stuck()) {
            return Err(r::stuck_blur(l));
        }
        // its flows would wait for that lane's burn past the stop, and wedge this lane too
        if let (true, Some(l)) = (x.raft, rd.stuck()) {
            return Err(r::stuck_raft(l));
        }
        let fps = x.rate(rate);
        let ab = span(range(&w, q.kind, fps, dur)?, still, fps, dur);
        let at = move |_| v;
        let (mut job, n) = clip_job(
            &q.files,
            dur,
            rate,
            &x,
            ab,
            still,
            at,
            dst.into(),
            None,
            false,
        );
        if n != p.frames {
            return Err(format!(
                "the source changed since it was checked: {n} frames, not {}",
                p.frames
            ));
        }
        // its blur pass, at the source's rate (`rate`: `Hide` links and pads by it, not by the output's `fps`)
        let scan = (x.faces || x.plates).then(|| Arc::new(Scan::default()));
        if let Some(s) = &scan {
            job.pre = Some(pre(&rd.g, &q.files, &x, ab, rate, v, s.clone()));
        }
        let d = Dest {
            part: Some(part),
            strict: true,
            frozen_ok: still || w.allow_frozen,
        };
        // pushed under `wedged`: a wedge since the look above refuses it, a later one finds it queued
        let wedged = lock(&l.wedged);
        if let Some(w) = &*wedged {
            return Err(w.clone());
        }
        // counted as driver work until the lane counts it: none is pushed once the process ends
        let id = (self.driver(|| l.q.push_into(job, d)))
            .ok_or("the process ends")?
            .map_err(|e| format!("the GPU's {} lane: {e}", lane.name()))?;
        drop(wedged);
        Ok(Arc::new(AgentRun {
            r: rd,
            lane,
            id,
            raft: x.raft,
            dst: dst.into(),
            cut: AtomicBool::new(false),
            t: Mutex::new((Instant::now(), None, None)),
            scan,
            search: p.search,
        }))
    }

    fn wedge(&self, lane: Lane, why: &str) {
        let Some(rd) = self.ready().or_else(|| self.dev.get().cloned()) else {
            return;
        };
        let l = rd.lane(lane);
        // after a submit that holds it has pushed: its job is cancelled below
        *lock(&l.wedged) = Some(why.into());
        for (id, s) in l.q.jobs() {
            if s == ::export::State::Queued {
                l.q.cancel(id)
            }
        }
    }

    fn trim(&self) {
        if let Some(rd) = self.ready() {
            self.driver(|| {
                rd.g.trim();
                rd.still.q.trim();
                rd.export.q.trim();
                // the blur's detectors too, unless a pass holds them (a faces and plates pass left about 160 MB of
                // them on the M4 with the earlier plate detector; they load again in about 20 ms, burn's kernels kept)
                crate::ai::MODELS.trim();
            });
        }
    }
}

/// The probe of a device that may be lost: `poll(Poll)` (wgpu panics on a lost one: `scoped` catches it), then a 4 × 4
/// clear read back, waited for [`PROBE`] at most.
fn sound(g: &Gpu) -> Result<(), String> {
    g.device
        .poll(wgpu::PollType::Poll)
        .map_err(|e| format!("poll: {e}"))?;
    let t = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let b = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe"),
        size: 256 * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = g.device.create_command_encoder(&Default::default());
    {
        let v = t.create_view(&Default::default());
        let c = wgpu::Color {
            r: 0.0,
            g: 0.5,
            b: 1.0,
            a: 1.0,
        };
        e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("probe"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &v,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(c),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
    }
    e.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &b,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        t.size(),
    );
    let i = {
        let _g = crate::GATE.read();
        g.queue.submit([e.finish()])
    };
    let s = b.slice(..);
    let (tx, rx) = mpsc::channel();
    s.map_async(wgpu::MapMode::Read, move |r| _ = tx.send(r));
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(i),
            timeout: Some(PROBE),
        })
        .map_err(|e| format!("the GPU's probe: {e}"))?;
    match rx.try_recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(format!("the GPU's probe: {e}")),
        Err(_) => return Err("the GPU's probe: its readback never came".into()),
    }
    let px: Vec<u8> = s
        .get_mapped_range()
        .map_err(|e| format!("the GPU's probe: {e:?}"))?[..4]
        .to_vec();
    match px[0] == 0 && px[1].abs_diff(128) <= 1 && px[2] == 255 && px[3] == 255 {
        true => Ok(()),
        false => Err(format!("the GPU's probe read {px:?}, not its clear colour")),
    }
}

/// What an export of `n` frames over `secs` at `x` writes at most: its video at the set rate (and its overshoot), or at
/// QP its pixels at [`QP24`]'s bits; then the audio, and a key picture's worth (a still's figure) on top.
fn video_bytes(x: &Set, n: u64, secs: f64) -> u64 {
    let [w, h] = x.size();
    let px = w as f64 * h as f64;
    let qpx = match x.pano {
        true => (px * PANO[0] as f64 * (PANO[0] / 2) as f64).sqrt(),
        false => px,
    };
    let at = |q: f64| QP24[!x.pano as usize] * 2f64.powf((24.0 - q) / 6.0) * qpx * n as f64;
    let v = match x.qp {
        Some(q) => at(q as f64),
        // a rate under what QP 51 (H.264's and H.265's coarsest) makes cannot be met
        None => (x.mbps as f64 * 1e6 * secs * OVER).max(at(51.0)),
    };
    ((v + ABR as f64 * secs) / 8.0 + px / 2.0) as u64
}

/// Which of the variables that choose the device are set (names only).
fn env() -> Vec<String> {
    std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .filter(|k| {
            ["DRI_PRIME", "__NV_PRIME_RENDER_OFFLOAD", "VK_ICD_FILENAMES"].contains(&k.as_str())
                || k.starts_with("WGPU_")
        })
        .collect()
}

/// A job on a lane, as mcp's poller reads it; dropped, its entry goes once it is finished.
struct AgentRun {
    r: Arc<Ready>,
    lane: Lane,
    id: u64,
    raft: bool,
    dst: PathBuf,
    /// mcp cancelled it (a lane's wedge cancels too, which reads failed)
    cut: AtomicBool,
    /// submitted, first seen running, the wall time once done (s)
    t: Mutex<(Instant, Option<Instant>, Option<f64>)>,
    /// its blur pass (None: no blur), the source frames it searches (the plan's)
    scan: Option<Arc<Scan>>,
    search: u64,
}

/// A blur pass's progress, shared by its job's `pre` and its `AgentRun`: source frames searched (of the plan's
/// `search`), when the first was, its end (the hook made), what it found.
#[derive(Default)]
struct Scan {
    searched: AtomicU64,
    first: OnceLock<Instant>,
    over: AtomicBool,
    found: Mutex<Option<r::Found>>,
}

/// Job `x`'s blur pass (`Job::pre`): the Export screen's privacy pass over `ab` of `files` with `x`'s stabilisation
/// and targets, the models held per searched frame; then its hook for `x`'s output in view `v`. `src`: the source's
/// rate (`probe`), not the output's `x.rate(src)`: `Hide` links and pads by it.
fn pre(
    g: &Arc<Gpu>,
    files: &[PathBuf],
    x: &Set,
    ab: [f64; 2],
    src: edit::Rate,
    v: View,
    s: Arc<Scan>,
) -> ::export::Pre {
    let (g, files) = (g.clone(), files.to_vec());
    let (stab, kinds, flat) = (x.stab.map(|l| (l, x.smooth)), [x.faces, x.plates], !x.pano);
    Box::new(move |cx: &stitch::Ctx, stop: &AtomicBool| {
        let t0 = Instant::now();
        let mut say = |i: usize, _| {
            if i == 1 {
                let _ = s.first.set(Instant::now());
            }
            s.searched.store(i as u64, Relaxed)
        };
        // its own scope: a wgpu error or panic in the pass fails the job now, named, not once the frames ran
        let (times, regs, c) = ::export::scoped(&g.device, || {
            privacy_pass(
                &g,
                cx,
                &files,
                stab,
                ab,
                src,
                kinds,
                Hold::Frame,
                stop,
                None,
                &mut say,
            )
        })
        .map_err(|e| match e == CANCELLED {
            true => e,
            false => format!("the blur pass: {e}"),
        })?;
        let [ff, fp, n, tf, tp, on, all] = c.map(|n| n as u64);
        *lock(&s.found) = Some(r::Found {
            faces: ff,
            plates: fp,
            face_tracks: tf,
            plate_tracks: tp,
            searched: n,
            blurred: on,
            frames: all,
            secs: t0.elapsed().as_secs_f64(),
        });
        s.over.store(true, Release);
        // as `Keep::video`'s: output time t shows source ab[0] + t, in the job's fixed view
        Ok(Some(hook(&g, flat, vec![(times, regs)], move |_, t| {
            Some((0, ab[0] + t, v))
        })))
    })
}

impl Drop for AgentRun {
    fn drop(&mut self) {
        self.r.lane(self.lane).q.forget(self.id)
    }
}

impl Run for AgentRun {
    fn state(&self) -> RunSt {
        let l = self.r.lane(self.lane);
        let gone = || {
            RunSt::Failed(format!(
                "the GPU's {} lane's thread ended",
                self.lane.name()
            ))
        };
        match l.q.state(self.id) {
            None => RunSt::Failed("the job is gone from its lane".into()),
            Some(::export::State::Queued) if !l.q.alive() => gone(),
            Some(::export::State::Queued) => RunSt::Queued,
            Some(::export::State::Running(_)) if !l.q.alive() => gone(),
            Some(::export::State::Running(p)) => {
                lock(&self.t).1.get_or_insert_with(Instant::now);
                if p.done > 0 && self.raft {
                    self.r.warm.store(true, Relaxed)
                }
                // `Running` is set before the pass starts: its phase from the first look
                let scan = self.scan.as_deref();
                let pass = scan.is_some_and(|s| !s.over.load(Acquire));
                let phase = match p.done {
                    0 if pass => Phase::Detecting,
                    0 if self.raft && !self.r.warm.load(Relaxed) => Phase::Compiling,
                    0 => Phase::Setup,
                    n if n >= p.total => Phase::Finishing,
                    _ => Phase::Frames,
                };
                let searched = scan.map_or(0, |s| s.searched.load(Relaxed));
                // while detecting the rates are the pass's (the frames' are 0): searched frames/s since the first, once
                // two are, and its seconds left
                let (fps, eta) = match (phase, scan.and_then(|s| s.first.get())) {
                    (Phase::Detecting, Some(t)) if searched >= 2 => {
                        let f = (searched - 1) as f64 / t.elapsed().as_secs_f64().max(1e-3);
                        (f, self.search.saturating_sub(searched) as f64 / f)
                    }
                    (Phase::Detecting, _) => (0.0, 0.0),
                    _ => (p.fps, p.eta),
                };
                RunSt::Running {
                    phase,
                    done: p.done,
                    total: p.total,
                    fps,
                    eta,
                    concealed: p.concealed,
                    searched,
                }
            }
            Some(::export::State::Done(p)) => {
                // a still's one flow ran (its done is 1 only just before the end, seldom seen running)
                if self.raft {
                    self.r.warm.store(true, Relaxed)
                }
                let secs = {
                    let mut t = lock(&self.t);
                    let from = t.1.unwrap_or(t.0);
                    *t.2.get_or_insert_with(|| from.elapsed().as_secs_f64())
                };
                RunSt::Done {
                    frames: p.done,
                    secs,
                    bytes: std::fs::symlink_metadata(&self.dst).map_or(0, |m| m.len()),
                    concealed: p.concealed,
                    found: self.scan.as_ref().and_then(|s| *lock(&s.found)),
                }
            }
            Some(::export::State::Failed(e)) => RunSt::Failed(e),
            Some(::export::State::Cancelled) => match &*lock(&l.wedged) {
                Some(w) if !self.cut.load(Relaxed) => RunSt::Failed(w.clone()),
                _ => RunSt::Cancelled,
            },
        }
    }

    fn cancel(&self) {
        self.cut.store(true, Relaxed);
        self.r.lane(self.lane).q.cancel(self.id)
    }
}

/// The catalog from this screen's tables (built once): its presets, sizes (360 videos without the photos' 11904),
/// codecs, rates and ranges; the looks, the editor's adjustments, the stabilisation modes, the player's flat
/// projections; the still's preset-free defaults; the blur targets and the pass's stride; what agents can't do yet.
fn catalog() -> Catalog {
    let range = |min: f64, max: f64, d: Option<f64>| r::Range {
        min,
        max,
        default: d,
    };
    let mut adj = fx::Params::default();
    let d: Vec<f32> = doc::adj(&mut adj).into_iter().map(|v| *v).collect();
    Catalog {
        presets: (PRESETS.iter())
            .map(|p| r::Preset {
                id: p.0.into(),
                name: p.1.into(),
                output: if p.2 { Output::Pano } else { Output::Flat },
                width: p.3[0],
                height: p.3[1],
                codec: codec_id(p.4).into(),
                ten_bit: p.5,
                mbps: p.6,
            })
            .collect(),
        default_preset: PRESETS[0].0.into(),
        pano_widths: PANO.into_iter().filter(|&w| w < 11904).collect(),
        flat_sizes: FLAT.to_vec(),
        codecs: (CODECS.iter())
            .map(|k| r::Codec {
                id: k.1.into(),
                ten_bit: k.2,
                max_px: k.3,
            })
            .collect(),
        mbps: range(1.0, 400.0, None),
        qp: range(10.0, 45.0, Some(24.0)),
        fps: FPS.iter().map(|f| f.1.to_ascii_lowercase()).collect(),
        raft: true,
        colour_match: true,
        stabilisation: STAB.map(String::from).into(),
        smoothing_s: range(0.0, gyro::SMOOTH_MAX, Some(gyro::SMOOTH)),
        looks: (LOOKS.iter().zip(fx::Look::ALL))
            .map(|(&id, l)| {
                let first = l
                    .name()
                    .split(' ')
                    .next()
                    .unwrap_or("")
                    .to_ascii_lowercase();
                r::Named {
                    id: id.into(),
                    name: Some(l.name().into()),
                    aliases: (first != id).then_some(first).into_iter().collect(),
                }
            })
            .collect(),
        look_strength: range(0.0, 1.0, Some(1.0)),
        adjust: (doc::ADJ.iter().zip(d))
            .map(|(a, d)| r::Adjust {
                id: a.0.into(),
                label: a.1.into(),
                min: a.2,
                max: a.3,
                default: d,
            })
            .collect(),
        projections: (VIEWS.iter())
            .map(|(id, al, l)| {
                let (v, f) = (l.view(), l.fov());
                r::Projection {
                    id: (*id).into(),
                    aliases: al.iter().map(|a| a.to_string()).collect(),
                    fov: [f.0, f.1],
                    yaw: v.yaw,
                    pitch: v.pitch,
                    roll: v.roll,
                    default_fov: v.fov,
                }
            })
            .collect(),
        still: r::Still {
            pano_width: STILL_360,
            flat: STILL_FLAT,
            jpeg_quality: range(50.0, 100.0, Some(JQ as f64)),
        },
        blur: (BLUR.iter())
            .map(|(id, name, al)| r::Named {
                id: (*id).into(),
                name: Some((*name).into()),
                aliases: al.iter().map(|a| a.to_string()).collect(),
            })
            .collect(),
        blur_every: STEP as u32,
        later: (LATER.iter())
            .map(|(id, name, al)| r::Later {
                id: (*id).into(),
                name: name.map(String::from),
                aliases: al.iter().map(|a| a.to_string()).collect(),
                effects: match *id {
                    "shot_lab" => lab::FXS.iter().map(|f| f.2.to_string()).collect(),
                    _ => vec![],
                },
            })
            .collect(),
    }
}

/// `w`, every option already checked against the catalog by mcp, as the Export screen's settings and flat view (and
/// its projection's id) hold them, set as its widgets set them: a still from preset-free defaults (360 at 7680, flat
/// 1920 × 1080, quality 92), an export from its preset (a changed size, codec, depth or rate clears it, as on the
/// screen); then the options asked, the blur's boxes too. `raft`: the seam's default. `w.stabilisation` is the
/// resolved one (`check`).
fn set_of(w: &Want, kind: Kind, raft: bool) -> Result<(Set, View, &'static str), String> {
    let bad = |what: &str, v: &str| format!("{what} {v:?}: not in the catalog");
    let flat = w.output == Some(Output::Flat);
    let mut x = Set::default();
    match kind {
        Kind::Still => {
            x.preset = None;
            (x.pano, x.w, x.h) = match flat {
                true => (false, STILL_FLAT[0], STILL_FLAT[1]),
                false => (true, STILL_360, STILL_360 / 2),
            };
            x.jq = w.jpeg_quality.unwrap_or(JQ);
        }
        Kind::Export => {
            let k = w.preset.as_deref().unwrap_or(PRESETS[0].0);
            let i = (PRESETS.iter().position(|p| p.0 == k)).ok_or_else(|| bad("preset", k))?;
            x.apply(i);
            // the screen's output buttons
            if w.output.is_some() && flat == x.pano {
                (x.pano, x.w, x.h) = match flat {
                    true => (false, 1920, 1080),
                    false => (true, 5760, x.h),
                };
            }
            if let Some(c) = &w.codec {
                x.codec = (CODECS.iter().find(|k| k.1 == c))
                    .ok_or_else(|| bad("codec", c))?
                    .0
            }
            if let Some(t) = w.ten_bit {
                x.deep = t
            }
            if let Some(m) = w.mbps {
                (x.mbps, x.qp) = (m, None)
            }
            if w.qp.is_some() {
                x.qp = w.qp
            }
            if let Some(f) = &w.fps {
                x.fps = (FPS.iter().find(|r| r.1.eq_ignore_ascii_case(f)))
                    .ok_or_else(|| bad("fps", f))?
                    .0
            }
        }
    }
    if let Some(a) = w.width {
        x.w = a
    }
    if let (false, Some(b)) = (x.pano, w.height) {
        x.h = b
    }
    x.deep &= x.codec == Codec::H265;
    let out = |s: &Set| (s.pano, s.w, s.h, s.codec, s.deep, s.mbps, s.qp);
    if let Some(i) = x.preset {
        let mut p = Set::default();
        p.apply(i);
        if out(&p) != out(&x) {
            x.preset = None
        }
    }
    x.raft = w.raft.unwrap_or(raft);
    x.colour = w.colour_match.unwrap_or(true);
    let s = w.stabilisation.as_deref().unwrap_or("off");
    x.stab = gyro::parse(s).ok_or_else(|| bad("stabilisation", s))?;
    x.smooth = w.smoothing_s.unwrap_or(gyro::SMOOTH);
    x.look = match &w.look {
        Some(l) => Some(
            (LOOKS.iter().position(|x| x == l))
                .map(|i| fx::Look::ALL[i])
                .ok_or_else(|| bad("look", l))?,
        ),
        None => None,
    };
    x.amt = w.look_strength.unwrap_or(1.0);
    x.adj = fx::Params::default();
    for (k, &v) in &w.adjust {
        let i = (doc::ADJ.iter().position(|a| a.0 == k)).ok_or_else(|| bad("adjust", k))?;
        *doc::adj(&mut x.adj)[i] = v;
    }
    for b in &w.blur {
        if !BLUR.iter().any(|t| t.0 == b) {
            return Err(bad("blur", b));
        }
    }
    let on = |i: usize| w.blur.iter().any(|b| b == BLUR[i].0);
    (x.faces, x.plates) = (on(0), on(1));
    let a = w.view.clone().unwrap_or_default();
    let p = a.projection.as_deref().unwrap_or(VIEWS[0].0);
    let (id, l) = (VIEWS.iter().find(|v| v.0 == p || v.1.contains(&p)))
        .map(|v| (v.0, v.2))
        .ok_or_else(|| bad("projection", p))?;
    let d = l.view();
    let v = View {
        yaw: a.yaw.unwrap_or(d.yaw),
        pitch: a.pitch.unwrap_or(d.pitch),
        roll: a.roll.unwrap_or(d.roll),
        fov: a.fov.unwrap_or(d.fov),
        ..d
    };
    Ok((x, v, id))
}

/// What `w` asks of a clip `dur` s long at output rate `fps`: a still's frame `[t, t]` (`time_s`, default 0, on the
/// clip); an export's `start_s`-`end_s` (default the whole clip), at least a frame and not past the end. No silent
/// clamping.
fn range(w: &Want, kind: Kind, fps: edit::Rate, dur: f64) -> Result<[f64; 2], String> {
    if kind == Kind::Still {
        let t = w.time_s.unwrap_or(0.0);
        return match t < dur {
            true => Ok([t; 2]),
            false => Err(format!("time_s {t}: the clip is {dur:.3} s long")),
        };
    }
    let (a, b) = (w.start_s.unwrap_or(0.0), w.end_s.unwrap_or(dur));
    if b > dur + EPS {
        return Err(format!("end_s {b}: the clip is {dur:.3} s long"));
    }
    let b = b.min(dur);
    if b - a < fps.t(1) - EPS {
        return Err(format!(
            "start_s {a} to end_s {b}: at least a frame ({:.3} s) of the clip, which is {dur:.3} s long",
            fps.t(1)
        ));
    }
    Ok([a, b])
}

/// The blur ids `x` renders with, in the catalog's order (none: no pass).
fn blur_ids(x: &Set) -> Vec<String> {
    (BLUR.iter().zip([x.faces, x.plates]))
        .filter(|(_, on)| *on)
        .map(|(t, _)| t.0.into())
        .collect()
}

/// The settings a job renders with, as results show them (`blur` always, `[]` without); `t`: a still's time asked.
fn settings(x: &Set, v: &View, proj: &str, w: &Want, kind: Kind, ab: [f64; 2], t: f64) -> Value {
    let [pw, ph] = x.size();
    let stab = (gyro::MODES.iter().position(|m| m.0 == x.stab)).map_or("off", |i| STAB[i]);
    let look = (x.look).and_then(|l| fx::Look::ALL.iter().position(|&k| k == l).map(|i| LOOKS[i]));
    let mut s = json!({
        "output": if x.pano { "360" } else { "flat" },
        "width": pw,
        "height": ph,
        "raft": x.raft,
        "colour_match": x.colour,
        "stabilisation": stab,
        "smoothing_s": x.smooth,
        "look": look,
        "look_strength": x.amt,
        "adjust": w.adjust,
        "blur": blur_ids(x),
    });
    if !x.pano {
        s["view"] = json!({ "projection": proj, "yaw": v.yaw, "pitch": v.pitch, "roll": v.roll, "fov": v.fov });
    }
    match kind {
        Kind::Still => {
            s["time_s"] = t.into();
            s["jpeg_quality"] = x.jq.into();
        }
        Kind::Export => {
            let fps = (FPS.iter().find(|f| f.0 == x.fps)).map(|f| f.1.to_ascii_lowercase());
            for (k, v) in [
                ("preset", json!(x.preset.map(|i| PRESETS[i].0))),
                ("codec", json!(codec_id(x.codec))),
                ("ten_bit", json!(x.deep)),
                ("mbps", json!(x.qp.is_none().then_some(x.mbps))),
                ("qp", json!(x.qp)),
                ("fps", json!(fps)),
                ("start_s", json!(ab[0])),
                ("end_s", json!(ab[1])),
                ("allow_frozen", json!(w.allow_frozen)),
            ] {
                s[k] = v
            }
        }
    }
    s
}

#[cfg(test)]
mod tests;
