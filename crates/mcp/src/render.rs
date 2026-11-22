//! The seam to the GPU: what `render_still`, `export` and `catalog` ask of a backend, as plain data. mcp has no GPU
//! dependency on any OS; the app implements [`Render`] over its engine and hands it in through [`crate::main`]
//! ([`Backend`]). Without one (Windows, mcp's tests) the GPU tools are not listed. mcp validates every id and range
//! against the backend's [`Catalog`], reserves the output's `.part` and keeps the records; the backend resolves the
//! source ([`Render::check`]), starts the device once per process ([`Render::start`]) and runs the jobs on its lanes
//! ([`Render::submit`]). `check`, `start`, `submit`, `probe`, `wedge` and `trim` block and run on `spawn_blocking`;
//! `catalog`, `gpu` and [`Run`]'s calls are quick and called inline, inside `catch_unwind`, with no lock of mcp held.
//! A blurred job ([`Plan::blur`]) runs its detection pass first ([`Phase::Detecting`], the frames it searched in
//! [`RunSt::Running`]) and ends done with what it found ([`Found`]).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    any::Any,
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

/// What a GPU tool answers when this x5desk has no backend.
pub const NO_GPU: &str = "this x5desk has no GPU tools (Windows: later)";

/// What a strict export's failure says of a source picture that could not be decoded (crate export's words, which
/// the app's tests hold to): mcp names `allow_frozen` after it.
#[cfg(target_os = "macos")]
pub const CONCEALED: &str =
    "could not be decoded and would repeat the one before (did the computer sleep?)";
#[cfg(not(target_os = "macos"))]
pub const CONCEALED: &str = "could not be decoded and would show damaged up to the next key frame (the GPU driver left a decode unfinished)";

/// What the app hands `mcp::main`. None: catalog, render_still and export are not listed (Windows, mcp's tests).
#[derive(Clone, Default)]
pub struct Backend {
    pub render: Option<Arc<dyn Render>>,
}

impl fmt::Debug for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Backend")
            .field("render", &self.render.is_some())
            .finish()
    }
}

pub trait Render: Send + Sync + 'static {
    /// The options, ids, ranges and defaults from the app's tables, built once. No GPU, no file, no wait.
    fn catalog(&self) -> Catalog;
    /// `r` resolved against its files: trailer, both lenses, calibration, IMU, rate, length, tracks, every option.
    /// Blocking (file reads), never the GPU.
    fn check(&self, r: &Req) -> Result<Plan, String>;
    /// The GPU, once per process. Blocking for seconds; concurrent callers wait for the same start. Each call makes
    /// `s.log` the process's log sink. Never panics: a panic inside ends `Failed`.
    fn start(&self, s: &Start) -> Result<GpuInfo, String>;
    /// A snapshot cloned under a short lock; never waits for a start.
    fn gpu(&self) -> GpuSt;
    /// What `x5desk mcp gpu-check` runs in the child (the self-test's steps 1-5); never attaches burn. Err: the
    /// step that failed and why.
    fn selftest(&self) -> Result<GpuInfo, (u8, String)>;
    /// `poll(Poll)` (errors once lost), then a scoped 4 × 4 clear and readback whose wait is bounded (5 s). Err: the
    /// device is lost, and `gpu` reads `Dead` from then on (mcp's poller stops this host's runs either way).
    fn probe(&self) -> Result<(), String>;
    /// Queues checked `r` on `lane`, writing into `part` (the reserved `<dst>.part`) and publishing `dst`. Quick (a
    /// header read, a channel send). Err: refused (not `Ready`, the lane wedged or gone); `part` dropped.
    fn submit(
        &self,
        r: Req,
        p: &Plan,
        dst: &Path,
        part: media::part::Part,
        lane: Lane,
    ) -> Result<Arc<dyn Run>, String>;
    /// `lane` unusable for this process: its queued runs end `Failed(why)`, later submits are refused.
    fn wedge(&self, lane: Lane, why: &str);
    /// Idle for 5 min: the decoder bin freed, the lanes' RAFT models and the blur's detectors dropped unless in use
    /// (they load again, about 150 ms and 20 ms).
    fn trim(&self);
}

/// A submitted job.
pub trait Run: Send + Sync {
    /// Quick: a short lock on the lane's entry. Only mcp's poller calls it.
    fn state(&self) -> RunSt;
    /// Non-blocking. A queued run reads `Cancelled` at once and never starts.
    fn cancel(&self);
}

#[derive(Clone, Debug, PartialEq)]
pub enum RunSt {
    Queued,
    Running {
        phase: Phase,
        done: u64,
        total: u64,
        /// frames/s and seconds left; in `Detecting` the pass's (searched frames/s, its seconds left)
        fps: f64,
        eta: f64,
        concealed: u64,
        /// the blur pass's source frames searched (0: no pass), its count kept in the phases after it; how many it
        /// searches is the plan's `search`
        searched: u64,
    },
    Done {
        frames: u64,
        secs: f64,
        bytes: u64,
        concealed: u64,
        /// what its blur pass found (None: no pass)
        found: Option<Found>,
    },
    Failed(String),
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// the blur pass: source frames searched for faces and plates (`RunSt::Running::searched`)
    Detecting,
    /// plan, decoders, model
    Setup,
    /// the first RAFT flow
    Compiling,
    Frames,
    Finishing,
}

/// What a blur pass found: detections per searched frame summed, the tracks they link into (about the faces and
/// plates seen; one lost for over a second counts again), the source frames searched, those blurred (a region on
/// them) of the span's, the pass's wall time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Found {
    pub faces: u64,
    pub plates: u64,
    pub face_tracks: u64,
    pub plate_tracks: u64,
    pub searched: u64,
    pub blurred: u64,
    pub frames: u64,
    pub secs: f64,
}

/// Why `blur` is refused on a GPU whose burn did not attach (`why`): mcp's device check and the backend's own.
pub fn no_blur(why: &str) -> String {
    format!(
        "blur is unavailable on this GPU: its detectors run on burn, which did not attach ({why}); leave out blur"
    )
}

/// Why `blur` is refused once `lane`, wedged, is stuck in a detection: it holds the detectors (and burn) until that
/// detection ends, if it ever does.
pub fn stuck_blur(lane: Lane) -> String {
    format!(
        "blur is unavailable while the GPU's {} lane is stuck in a detection, which holds the detectors: if it never \
         ends, this MCP server must restart to blur; leave out blur",
        lane.name()
    )
}

/// Why RAFT is refused then: its job would wait for that lane's burn past its stop, and wedge its own lane.
pub fn stuck_raft(lane: Lane) -> String {
    format!(
        "RAFT (optical-flow seams) is unavailable while the GPU's {} lane is stuck in a detection, which holds burn; \
         pass raft: false",
        lane.name()
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lane {
    Still,
    Export,
}

impl Lane {
    /// As messages name it.
    pub fn name(self) -> &'static str {
        match self {
            Lane::Still => "still",
            Lane::Export => "export",
        }
    }
}

/// How [`Render::start`] starts the GPU.
pub struct Start {
    /// burn's cubecl base; None: memory only
    pub cubecl: Option<PathBuf>,
    /// the self-test child's executable (Linux /proc/self/exe, macOS current_exe); None: in process (tests)
    pub child: Option<PathBuf>,
    pub log: Arc<dyn Fn(String) + Send + Sync>,
    /// set by the host's end: a running child is killed within 50 ms and the start ends `Off` (not a failure)
    pub halt: Arc<AtomicBool>,
}

#[derive(Clone, Debug)]
pub enum GpuSt {
    Off,
    Starting,
    Ready(GpuInfo),
    /// `retry_in` None: final for the process
    Failed {
        why: String,
        step: Option<u8>,
        retry_in: Option<Duration>,
    },
    Dead(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub adapter: String,
    pub backend: String,
    pub driver: String,
    /// the child's adapter is matched against the host's
    pub vendor: u32,
    pub device: u32,
    /// per profile: h265 Main, h265 Main 10, h264 High
    pub decode: Vec<Cap>,
    /// empty: stills only
    pub encode: Vec<Cap>,
    pub max_texture: u32,
    /// None: burn attached (RAFT and blur); Some(why): it is not
    pub raft: Option<String>,
    /// a wedged lane stuck in a detection, which holds the detectors and burn: blur refused ([`stuck_blur`]), RAFT too
    /// ([`stuck_raft`]: off by default then)
    #[serde(default)]
    pub blur_stuck: Option<Lane>,
    /// uncaptured errors since the start
    pub errors: u32,
    /// a probe is due: OOM/Internal outside a scope, cubecl's loss line
    pub suspect: bool,
    pub wedged: Vec<Lane>,
    /// which of DRI_PRIME, __NV_PRIME_RENDER_OFFLOAD, VK_ICD_FILENAMES, WGPU_* are set (names only)
    pub env: Vec<String>,
    /// how long the check took
    pub ms: u64,
    /// what the decode checks found so far; None: the device's decodes aren't checked (boxed: `GpuSt` stays small)
    #[serde(default)]
    pub checks: Option<Box<Checks>>,
}

/// The decode checks' counts since the device opened, where the driver reports how each decode ended and decodes on a
/// queue of its own (NVIDIA): pictures decoded again and recovered (correct ones), pictures that could not be decoded
/// (shown as the driver left them), the memory the redos read their bitstream from ("device", "host", "device and
/// host"; none before the first).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Checks {
    pub recovered: u64,
    pub failed: u64,
    pub redo_memory: Option<String>,
}

/// A codec profile the device decodes or encodes.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Cap {
    pub codec: String,
    pub ten_bit: bool,
    pub min: [u32; 2],
    pub max: [u32; 2],
    /// encode: "cbr", "cqp"
    pub rate: Vec<String>,
    pub max_mbps: Option<f32>,
    pub qp: Option<[u8; 2]>,
}

/// A render job as mcp checked it.
pub struct Req {
    pub kind: Kind,
    /// the source's main files in lens order (`cam::work::pair`), each under a read root
    pub files: Vec<PathBuf>,
    pub want: Want,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Export,
    Still,
}

impl Kind {
    /// As the job tools name it (`kind`).
    pub fn name(self) -> &'static str {
        match self {
            Kind::Export => "export",
            Kind::Still => "still",
        }
    }
}

/// The 360 sphere (equirectangular) or a flat view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Output {
    #[serde(rename = "360")]
    Pano,
    #[serde(rename = "flat")]
    Flat,
}

/// A flat output's projection and direction, degrees; None: the projection's default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct View {
    /// a catalog projection id
    pub projection: Option<String>,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub roll: Option<f32>,
    pub fov: Option<f32>,
}

/// What the agent asked for, every option already checked against the catalog by mcp; None: the default.
#[derive(Clone, Debug, Default)]
pub struct Want {
    pub preset: Option<String>,
    pub output: Option<Output>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub view: Option<View>,
    pub look: Option<String>,
    pub look_strength: Option<f32>,
    pub adjust: BTreeMap<String, f32>,
    pub stabilisation: Option<String>,
    pub smoothing_s: Option<f64>,
    pub raft: Option<bool>,
    pub colour_match: Option<bool>,
    /// catalog blur ids, in the catalog's order; empty: none
    pub blur: Vec<String>,
    // stills
    pub time_s: Option<f64>,
    pub jpeg_quality: Option<u8>,
    /// the picture's noise reduction strength; None: the catalog's default
    pub noise_reduction: Option<f32>,
    // exports
    pub start_s: Option<f64>,
    pub end_s: Option<f64>,
    pub codec: Option<String>,
    pub ten_bit: Option<bool>,
    pub mbps: Option<f32>,
    pub qp: Option<u8>,
    /// a catalog label: mcp maps numbers onto them
    pub fps: Option<String>,
    /// RNNoise strength; None or 0: off
    pub denoise: Option<f32>,
    /// a flat output follows the recording's auto framing
    pub auto_framing: bool,
    pub allow_frozen: bool,
}

/// A checked job, resolved against its source.
#[derive(Clone, Debug)]
pub struct Plan {
    /// output frames: the job's `Progress::total` (1 for a still)
    pub frames: u64,
    pub secs: f64,
    pub size: [u32; 2],
    pub codec: Option<String>,
    pub ten_bit: bool,
    pub raft: bool,
    /// the blur ids it renders with (empty: none)
    pub blur: Vec<String>,
    /// the source frames its blur pass searches (0: none)
    pub search: u64,
    pub mbps: Option<f32>,
    pub qp: Option<u8>,
    /// the source's video tracks, for the decode caps
    pub tracks: Vec<Track>,
    /// the default name: `<source stem>_<tag>`
    pub stem: String,
    /// "mp4" | "jpg"
    pub ext: &'static str,
    /// the estimate for the free-space check
    pub bytes: u64,
    /// resolved, as results show them
    pub settings: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub codec: String,
    pub ten_bit: bool,
    pub size: [u32; 2],
}

/// What render_still and export accept, from the app's tables; `catalog` shows it with mcp's own rules and the
/// GPU's state. Ids are stable API.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Catalog {
    pub presets: Vec<Preset>,
    /// exports' default preset
    pub default_preset: String,
    /// video sizes: 360 widths (the height is half), flat sizes
    pub pano_widths: Vec<u32>,
    pub flat_sizes: Vec<[u32; 2]>,
    pub codecs: Vec<Codec>,
    /// the constant bit rate, Mbit/s, and the constant QP
    pub mbps: Range,
    pub qp: Range,
    /// "source", then the rates as labels ("29.97")
    pub fps: Vec<String>,
    /// the defaults
    pub raft: bool,
    pub colour_match: bool,
    /// stabilisation ids
    pub stabilisation: Vec<String>,
    pub smoothing_s: Range,
    pub looks: Vec<Named>,
    pub look_strength: Range,
    /// an export's RNNoise strength (0: off, the default)
    #[serde(default)]
    pub denoise: Range,
    pub adjust: Vec<Adjust>,
    pub projections: Vec<Projection>,
    pub still: Still,
    /// what `blur` takes (none: this x5desk has no blur)
    #[serde(default)]
    pub blur: Vec<Named>,
    /// a blur pass searches every this many source frames
    #[serde(default)]
    pub blur_every: u32,
    /// what agents can't do yet
    pub later: Vec<Later>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub id: String,
    pub name: String,
    pub output: Output,
    pub width: u32,
    pub height: u32,
    pub codec: String,
    pub ten_bit: bool,
    pub mbps: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Codec {
    pub id: String,
    /// 10-bit allowed
    pub ten_bit: bool,
    /// the long edge at most
    pub max_px: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Range {
    pub min: f64,
    pub max: f64,
    pub default: Option<f64>,
}

/// A choice: its id, display name and the other words accepted for it (matched case-insensitively).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Named {
    pub id: String,
    pub name: Option<String>,
    pub aliases: Vec<String>,
}

/// An `adjust` key: the project files' `adjust`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Adjust {
    pub id: String,
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

/// A flat projection, its field of view's range and its default direction (degrees).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub id: String,
    pub aliases: Vec<String>,
    pub fov: [f32; 2],
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub default_fov: f32,
}

/// The still's defaults, preset-free.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Still {
    /// the 360 width (the height is half) and the flat size
    pub pano_width: u32,
    pub flat: [u32; 2],
    pub jpeg_quality: Range,
    /// the picture's noise reduction (non-local means) strength
    #[serde(default)]
    pub noise_reduction: Range,
}

/// What agents can't do yet, and the argument names that ask for it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Later {
    pub id: String,
    pub name: Option<String>,
    pub aliases: Vec<String>,
    /// Shot Lab's effect ids
    pub effects: Vec<String>,
}

/// A panic's message, for a backend call caught with `catch_unwind`.
pub fn panicked(p: &(dyn Any + Send)) -> String {
    match (p.downcast_ref::<&str>(), p.downcast_ref::<String>()) {
        (Some(s), _) => s.to_string(),
        (_, Some(s)) => s.clone(),
        _ => "a panic".into(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A backend with no GPU work: its self-test passes, fails at step 3 or panics; the rest is never called yet.
    pub(crate) enum Fake {
        Ok,
        Fails,
        Panics,
    }

    pub(crate) fn info() -> GpuInfo {
        GpuInfo {
            adapter: "Fake GPU".into(),
            backend: "metal".into(),
            vendor: 0x106b,
            decode: vec![Cap {
                codec: "h265".into(),
                max: [8192, 8192],
                ..Default::default()
            }],
            max_texture: 16384,
            ms: 7,
            ..Default::default()
        }
    }

    impl Render for Fake {
        fn catalog(&self) -> Catalog {
            Catalog::default()
        }
        fn check(&self, _: &Req) -> Result<Plan, String> {
            Err("the fake checks nothing".into())
        }
        fn start(&self, _: &Start) -> Result<GpuInfo, String> {
            panic!("the GPU started")
        }
        fn gpu(&self) -> GpuSt {
            GpuSt::Off
        }
        fn selftest(&self) -> Result<GpuInfo, (u8, String)> {
            match self {
                Fake::Ok => Ok(info()),
                Fake::Fails => Err((
                    3,
                    "this GPU cannot decode the X5's 3840 × 3840 H.265".into(),
                )),
                Fake::Panics => panic!("the driver went away"),
            }
        }
        fn probe(&self) -> Result<(), String> {
            Ok(())
        }
        fn submit(
            &self,
            _: Req,
            _: &Plan,
            _: &Path,
            _: media::part::Part,
            _: Lane,
        ) -> Result<Arc<dyn Run>, String> {
            Err("the fake runs nothing".into())
        }
        fn wedge(&self, _: Lane, _: &str) {}
        fn trim(&self) {}
    }

    #[test]
    fn debug() {
        let b = Backend {
            render: Some(Arc::new(Fake::Ok)),
        };
        assert_eq!(format!("{b:?}"), "Backend { render: true }");
        assert_eq!(
            format!("{:?}", Backend::default()),
            "Backend { render: false }"
        );
        assert_eq!(
            panicked(&*std::panic::catch_unwind(|| panic!("x")).unwrap_err()),
            "x"
        );
        let n = 3;
        let p = std::panic::catch_unwind(|| panic!("n = {n}")).unwrap_err();
        assert_eq!(panicked(&*p), "n = 3");
    }
}
