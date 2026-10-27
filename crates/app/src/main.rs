//! x5desk: desktop app for the Insta360 X5. One eframe window on the shared Vulkan device (`video::Gpu`); a side
//! panel switches screens. `x5desk [file]` opens a file.
//!
//! Test hooks: `X5DESK_SHOT=<file.png>` (with `X5DESK_SHOT_AFTER=<s>`, default 1) saves a screenshot once that time has
//! passed and the player has nothing pending, then exits. `X5DESK_TEST=<tokens>` (comma separated) sets up the first
//! file opened: `play`, `loop`, `vol=<0..1>` (default 0.1 under the hook), `view=perspective|panini|equirect|planet`,
//! `seek=<frame>`, `proxy`, `stats=<s>` (play, print pacing statistics after s seconds, exit), `novsync` (present
//! without waiting for vblank, repaint on a 60 Hz timer: for a compositor that sends no frame callbacks, e.g. with the
//! display off), `full` (no automatic switch to the proxy), `stab=off|free|horizon|direction` (default horizon),
//! `smooth=<s>` (the stabilisation's σ, default 0.5), `camera[=album|downloads|settings|wireless]` (start on the Camera screen;
//! with `X5DESK_SHOT` the shot waits for the album's thumbnails), `dl=<camera URI>` (queue a download, repeatable),
//! `live` (start on the Live screen), `end` (show the Live screen scrolled to its end), `rtmp=<url>` (stream there once
//! the file shows a picture; the key is the URL's last segment), `res=1080|1440|360`, `fps=<n>`, `kbps=<n>`, `obs`
//! (serve the OBS source), `pwcam` (publish the PipeWire camera), `grab=<file.png>` (once the player is idle, e.g. a
//! paused frame's RAFT stitch done: save the live outputs' `view` and `360` pictures as `<file>-view.png`,
//! `<file>-360.png`), `min=<s>` (minimise the window after s seconds: the compositor stops drawing it, the live outputs
//! run on), `max` (maximise the window: a tiling compositor gives it some size of its own otherwise; the screenshot
//! waits for it, asked up to 12 times 250 ms apart, about 3 s). Under the hook the Live screen neither reads nor
//! writes the user's live settings, and the live output thread prints its state every 5 s. Capture: `capture` (start
//! on the Capture screen with the live preview running),
//! `replay=<rec.h264>` (a simulated camera streaming that `x5 live` recording in real time instead of the X5, with
//! the calibration of `$X5_SAMPLES/v635.insv` and `/CONF` from `conf=<file>` or `$X5_CONF`; `battery=<%>`: the level it
//! reports), `webcam[=1920|<file.mjpg>]`
//! (the Webcam Mode instead, 2880 × 1440, 1920 × 1080 or a capture replayed), `cstats=<s>` (print the preview's
//! statistics after s seconds of pictures, exit), `view=` / `yaw=<deg>` (its view), `level=off`, `pace=off` (the viewer shows the newest
//! picture instead of going at the camera's pace; the live outputs keep it), `src=preview|webcam`
//! (the Live screen's source). Export: `export=<preset>[:<look>][:faces][:noraft][:qp=<n>]` (repeatable; queue an
//! export of the first file once it shows, from the player's view and stabilisation; presets by key, e.g. `pano57`,
//! `flat1080`; looks by their first word: vivid, warm, cool, film, black, teal), `frame=360|flat` (a frame grab of the
//! frame shown, repeatable), `dest=<dir>` (where they go), `aim=<yaw>/<pitch>[/<fov>]` (the first file's view, degrees); the Export screen shows, each job's end is printed, and the run ends
//! (or takes its screenshot) once the queue is done. Editor: `editor` (start on the Editor screen), `project=<file>`
//! (load a project), `clip=<file>` (append a recording, repeatable), `undo=<n>`, `redo=<n>`, `at=<s>` (the playhead),
//! `sel=<clip>`, `etab=clip|reframe|colour|audio|overlays`, `e360` (preview the 360 frame), `epreset=<preset>` (the
//! export preset the viewer frames for), `esave=<file>` (save the project), `tlexport=<preset>[:noraft][:qp=<n>]` (hand
//! the timeline to the Export screen; the run ends with the queue), `eplay` (play once the editor is idle, at `vol=`;
//! prints the frames shown when it stops). Settings and help: `settings` (start on the Settings screen),
//! `help[=<page>]` (the Help screen at `keys`, `about`, or a screen's guide, e.g. `camera`). Other camera links:
//! `camera=wireless` (the Bluetooth and Wi-Fi pane), `mount=<dir>` (a directory with a camera's `DCIM` counts as a
//! mounted X5; repeatable; the real mount table is not read then; starts on the album unless `camera=` says otherwise),
//! `import` (import the mounted card's whole album once listed), `ble=sim` (a simulated Bluetooth adapter with one X5
//! that scans, connects, pairs, reads its access point and joins `HomeNet` in station mode at once; starts on the
//! Wireless pane), `ble=sta` (the same X5 on `HomeNet` from an earlier session: scans, connects, pairs and reads its
//! station connection, no join). Under the hook nothing of the user's config or cache is read, written or trimmed
//! (the app's and the camera's settings, the recent list, the download queue: the run's own; no thumbnail cache), and
//! `dest=` is the download folder too. AI and Shot Lab:
//! `box=<x0>/<y0>/<x1>/<y1>` (Deep Track in the player: that box, 0..1 of the view at 16:9 as a flat 1080p export
//! frames it, whatever the window, drawn once the frame shows, its subject printed; the view follows the subject when
//! the run is over, and an `export=` waits for it and takes the follow view),
//! `restab=off|free|horizon|direction[/<σ>]` (once that run is over, the player's stabilisation changes to this, as its
//! controls would: the follow view turns into the new frame), `blur`
//! (the player's face blur preview), `autoedit[=<preset>]` (an auto edit of the first file onto the editor's
//! timeline; with a preset key exported once made, the run ending with the queue), `ebox=<x0>/<y0>/<x1>/<y1>` (Deep
//! Track in the editor's viewer at the playhead once it is set up: that box, 0..1 of the export preset's flat frame
//! (`epreset=`; 16:9 for a 360 one, as `box=`) whatever the window, its subject printed), `lab=<effect>` (start on
//! Shot Lab with that effect, e.g. `trails`, `shadow`, `center`), `lrange=<a>/<b>` (its range, s), `lpre` (make its
//! preview), `lexp=<preset>[:faces]` (export it with that preset, and the face blur, once the clip, and for the subject
//! effects the `box=` run, is there; the run ends with the queue).

mod ai;
mod album;
mod bt;
mod camera;
mod capture;
mod clip;
mod conf;
mod disk;
mod dl;
mod editor;
mod export;
mod feed;
mod gyro;
mod help;
mod lab;
mod library;
mod live;
mod movie;
mod photo;
mod player;
mod png;
mod replay;
mod settings;
mod webcam;

use eframe::{egui, egui_wgpu};
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError, RwLock, RwLockWriteGuard,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Default)]
pub struct Test {
    pub play: bool,
    pub looping: bool,
    pub vol: Option<f32>,
    pub look: Option<player::Look>,
    pub seek: Option<usize>,
    pub proxy: bool,
    pub stats: Option<f64>,
    pub novsync: bool,
    pub full: bool,
    pub stab: Option<gyro::Mode>,
    pub smooth: Option<f64>,
    pub camera: Option<camera::Pane>,
    pub dl: Vec<String>,
    pub live: bool,
    pub end: bool,
    pub out: live::Hook,
    pub grab: Option<PathBuf>,
    pub min: Option<f64>,
    pub max: bool,
    pub cap: Option<capture::Hook>,
    pub replay: Option<PathBuf>,
    pub conf: Option<PathBuf>,
    pub battery: Option<u32>,
    pub export: Vec<String>,
    pub frame: Vec<bool>,
    /// the player's (or photo's) view: yaw, pitch, field of view (°)
    pub aim: Option<[f32; 3]>,
    pub dest: Option<PathBuf>,
    /// the editor's tokens
    pub ed: editor::Hook,
    pub settings: bool,
    pub help: Option<help::Page>,
    pub mount: Vec<PathBuf>,
    pub import: bool,
    /// the simulated Bluetooth adapter: its camera on HomeNet from an earlier session
    pub ble: Option<bool>,
    /// the player's Deep Track box, face blur preview, an auto edit (and the preset to export it with)
    pub tbox: Option<[f32; 4]>,
    pub restab: Option<(gyro::Mode, f64)>,
    pub blur: bool,
    pub autoedit: Option<String>,
    pub lab: lab::Hook,
}

/// `x0/y0/x1/y1`.
fn quad(v: &str) -> Option<[f32; 4]> {
    let a: Vec<f32> = v.split('/').filter_map(|x| x.parse().ok()).collect();
    a.try_into().ok()
}

impl Test {
    fn parse(s: &str) -> Test {
        let mut t = Test {
            vol: Some(0.1),
            ..Default::default()
        };
        for w in s.split(',').map(str::trim).filter(|w| !w.is_empty()) {
            let (k, v) = w.split_once('=').unwrap_or((w, ""));
            match k {
                "capture" => t.cap.get_or_insert_default().on = true,
                "webcam" => {
                    let c = t.cap.get_or_insert_default();
                    (c.on, c.webcam) = (true, Some(capture::webcam_input(v)))
                }
                "cstats" => t.cap.get_or_insert_default().stats = v.parse().ok(),
                "yaw" => t.cap.get_or_insert_default().yaw = v.parse().ok(),
                "level" => t.cap.get_or_insert_default().level = Some(v != "off"),
                "pace" => t.cap.get_or_insert_default().pace = Some(v != "off"),
                "replay" => t.replay = Some(v.into()),
                "conf" => t.conf = Some(v.into()),
                "battery" => t.battery = v.parse().ok(),
                "src" => {
                    t.out.pick = match v {
                        "preview" => live::Pick::Preview,
                        "webcam" => live::Pick::Webcam,
                        _ => live::Pick::Player,
                    }
                }
                "play" => t.play = true,
                "loop" => t.looping = true,
                "proxy" => t.proxy = true,
                "novsync" => t.novsync = true,
                "full" => t.full = true,
                "vol" => t.vol = v.parse().ok(),
                "seek" => t.seek = v.parse().ok(),
                "stats" => t.stats = v.parse().ok(),
                "view" => {
                    t.look = player::Look::parse(v);
                    if let Some(h) = &mut t.cap {
                        h.look = t.look
                    }
                }
                "stab" => t.stab = gyro::parse(v),
                "smooth" => t.smooth = v.parse().ok(),
                "camera" => t.camera = camera::Pane::parse(v),
                "dl" => t.dl.push(v.into()),
                "live" => t.live = true,
                "end" => t.end = true,
                "rtmp" => t.out.rtmp = Some(v.into()),
                "res" => t.out.res = ["1080", "1440", "360"].iter().position(|r| *r == v),
                "fps" => t.out.fps = v.parse().ok(),
                "kbps" => t.out.kbps = v.parse().ok(),
                "obs" => t.out.obs = true,
                "pwcam" => t.out.cam = true,
                "grab" => t.grab = Some(v.into()),
                "min" => t.min = v.parse().ok(),
                "max" => t.max = true,
                "export" => t.export.push(v.into()),
                "frame" => t.frame.push(v != "flat"),
                "aim" => {
                    let a: Vec<f32> = v.split('/').filter_map(|x| x.parse().ok()).collect();
                    t.aim =
                        (a.len() >= 2).then(|| [a[0], a[1], a.get(2).copied().unwrap_or(f32::NAN)]);
                }
                "dest" => t.dest = Some(v.into()),
                "editor" => t.ed.on = true,
                "project" => (t.ed.on, t.ed.project) = (true, Some(v.into())),
                "clip" => {
                    t.ed.on = true;
                    t.ed.add.push(v.into())
                }
                "at" => t.ed.at = v.parse().ok(),
                "sel" => t.ed.sel = v.parse().ok(),
                "etab" => t.ed.tab = editor::Tab::parse(v),
                "undo" => t.ed.undo = v.parse().unwrap_or(1),
                "redo" => t.ed.redo = v.parse().unwrap_or(1),
                "esave" => t.ed.save = Some(v.into()),
                "tlexport" => t.ed.export = Some(v.into()),
                "eplay" => t.ed.play = true,
                "e360" => t.ed.pano = true,
                "epreset" => t.ed.preset = Some(v.into()),
                "settings" => t.settings = true,
                "help" => t.help = Some(help::Page::parse(v).unwrap_or(help::Page::Keys)),
                "mount" => t.mount.push(v.into()),
                "import" => t.import = true,
                "ble" => {
                    t.ble = [("sim", false), ("sta", true)]
                        .into_iter()
                        .find(|x| x.0 == v)
                        .map(|x| x.1)
                }
                "box" => t.tbox = quad(v),
                "restab" => {
                    let (m, s) = v.split_once('/').unwrap_or((v, ""));
                    t.restab = gyro::parse(m).map(|m| (m, s.parse().unwrap_or(gyro::SMOOTH)));
                }
                "ebox" => t.ed.tbox = quad(v),
                "blur" => t.blur = true,
                "autoedit" => t.autoedit = Some(v.into()),
                "lab" => (t.lab.on, t.lab.fx) = (true, lab::Fx::parse(v)),
                "lrange" => {
                    let a: Vec<f64> = v.split('/').filter_map(|x| x.parse().ok()).collect();
                    t.lab.range = a.try_into().ok()
                }
                "lpre" => t.lab.pre = true,
                "lexp" => t.lab.exp = Some(v.into()),
                _ => eprintln!("X5DESK_TEST: unknown token {w}"),
            }
        }
        t
    }
}

/// `$<var>` if absolute, else `~/<home>` (XDG base directories; the other platforms' come with their ports).
pub fn xdg(var: &str, home: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|d| d.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(home)))
}

/// `d`, made owner-only if missing, if it is a dir of this user's that no other can write to: in a shared tmp another
/// user may have made it first, or put a symlink there.
fn mine(d: PathBuf) -> Option<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    let _ = std::fs::DirBuilder::new().mode(0o700).create(&d);
    let m = d.symlink_metadata().ok()?;
    // SAFETY: geteuid has no preconditions.
    let me = unsafe { libc::geteuid() };
    (m.is_dir() && m.uid() == me && m.mode() & 0o022 == 0).then_some(d)
}

/// Makes the app's config dir `d` before a save there: created if missing, owner-only also when it exists already
/// (it holds the stream key, the client id, the recent files). Its parents keep their modes.
pub fn private(d: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    if let Some(p) = d.parent() {
        std::fs::create_dir_all(p)?
    }
    match std::fs::DirBuilder::new().mode(0o700).create(d) {
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => return Err(e),
        _ => {}
    }
    std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700))
}

/// wgpu-core's `Surface::configure` waits for the queue to go idle and fails (`GpuWaitTimeout`) when another thread
/// submits meanwhile. egui-wgpu then draws the new window size into the old swapchain, and on native Wayland nothing
/// reports that swapchain `Outdated`. So the app retries it: the threads of its own that submit on their own hold
/// this shared around each piece of GPU work (the live outputs per pass, the RTMP encoder per frame, the export's
/// threads and decoders through `::export::locks`, the export screen's face pass and photos), and after a failed
/// configure the UI thread holds it and [`BURN`] exclusively from the end of its next frame's UI until egui has
/// configured the surface again and acquired that frame ([`Retry`]). The player's decoders' empty flush submits stay
/// outside: a decoder waiting for a slot would wait for the UI's next submit (the export's renderer releases its
/// decoders' slots without the lock).
pub static GATE: RwLock<()> = RwLock::new(());
/// Held for a whole job by the threads whose GPU work runs through burn (the RAFT worker): burn submits throughout
/// it. Apart from [`GATE`], because a waiting writer closes GATE to new readers: a retry waits for the job here first
/// (≈ 70 ms; ≈ 9 s for the first, while burn compiles and tunes its kernels), then for GATE, so the outputs wait for
/// the configure only, never for a job.
pub static BURN: Mutex<()> = Mutex::new(());
/// a configure failed that way; the next frame retries it
static RACE: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// the UI thread's hold on [`BURN`] and [`GATE`] during a retry
    static HELD: RefCell<Option<(MutexGuard<'static, ()>, RwLockWriteGuard<'static, ()>)>> =
        const { RefCell::new(None) };
}

/// The device's error policy: a surface configure that ran while another thread submitted is retried (above),
/// anything else is a bug and panics, as wgpu's default handler does. Installed after burn attaches to the device
/// (`stitch::attach`), which installs cubecl's own once and never again.
pub fn errors(dev: &wgpu::Device) {
    dev.on_uncaptured_error(Arc::new(|e| {
        if !race(&e) {
            panic!("wgpu error: {e}");
        }
        RACE.store(true, Relaxed);
        eprintln!("x5desk: surface reconfigure overlapped another thread's submit; retrying it");
    }));
}

/// Paints nothing; ends a retry. egui paints it once the surface is configured and the frame acquired, before the
/// present; a frame it does not paint (no surface texture) drops it after that frame.
struct Retry;

impl egui_wgpu::CallbackTrait for Retry {
    fn paint(
        &self,
        _: egui::PaintCallbackInfo,
        _: &mut wgpu::RenderPass<'static>,
        _: &egui_wgpu::CallbackResources,
    ) {
        HELD.take();
    }
}

impl Drop for Retry {
    fn drop(&mut self) {
        HELD.take();
    }
}

fn race(e: &wgpu::Error) -> bool {
    use wgpu::wgc::{error::ContextError, present::ConfigureSurfaceError};
    match e {
        wgpu::Error::Validation { source, .. } => source
            .downcast_ref::<ContextError>()
            .and_then(|c| c.source.downcast_ref::<ConfigureSurfaceError>())
            .is_some_and(|c| matches!(c, ConfigureSurfaceError::GpuWaitTimeout)),
        _ => false,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Library,
    Editor,
    Lab,
    Camera,
    Capture,
    Live,
    Export,
    Settings,
    Help,
}

/// Screens and their names.
const TABS: [(Tab, &str); 9] = [
    (Tab::Library, "Library"),
    (Tab::Editor, "Editor"),
    (Tab::Lab, "Shot Lab"),
    (Tab::Camera, "Camera"),
    (Tab::Capture, "Capture"),
    (Tab::Live, "Live"),
    (Tab::Export, "Export"),
    (Tab::Settings, "Settings"),
    (Tab::Help, "Help"),
];

struct Shot {
    path: PathBuf,
    after: Duration,
    sent: bool,
    done: Arc<AtomicBool>,
}

struct App {
    tab: Tab,
    // dropped first: the outputs stop on the source's last picture instead of repeating it while the sources shut down
    live: live::Live,
    lib: library::Library,
    cam: camera::Cam,
    cap: capture::Capture,
    exp: export::Exp,
    ed: editor::Editor,
    lab: lab::Lab,
    set: settings::Settings,
    help: help::Help,
    /// test hook: the exports and the frame grab (360?) to queue once the first file shows
    auto: Option<(Vec<String>, Vec<bool>)>,
    shot: Option<Shot>,
    /// the test hook ends the run once the player is idle
    exit: bool,
    /// the test hook saves the live outputs' pictures once the player is idle, and minimises the window after `min` s
    grab: Option<PathBuf>,
    min: Option<f64>,
    /// the test hook's `max`: when to ask for it next and how many more times, until the window is maximised
    max: Option<(Duration, u8)>,
    /// the test hook shows the Live screen scrolled to its end
    end: bool,
    t0: Instant,
}

impl App {
    fn new(
        cc: &eframe::CreationContext,
        g: Arc<video::Gpu>,
        file: Option<PathBuf>,
        test: Option<Test>,
    ) -> App {
        let cx = stitch::Ctx::new(&g.device, &g.queue);
        let ch = test.as_ref().and_then(|t| t.cap.clone());
        let auto = test
            .as_ref()
            .filter(|t| !t.export.is_empty() || !t.frame.is_empty())
            .map(|t| (t.export.clone(), t.frame.clone()));
        let hook = (test.as_ref())
            .map(|t| editor::Hook {
                vol: t.vol,
                ..t.ed.clone()
            })
            .filter(|h| h.on);
        let exit = auto.is_some()
            || hook.as_ref().is_some_and(|h| h.export.is_some())
            || test.as_ref().is_some_and(|t| {
                t.autoedit.as_ref().is_some_and(|a| !a.is_empty()) || t.lab.exp.is_some()
            })
            || test.as_ref().is_some_and(|t| {
                t.stats.is_some() || ch.as_ref().is_some_and(|h| h.stats.is_some())
            });
        let mut exp = export::Exp::new(
            &g,
            &cx,
            &cc.egui_ctx,
            test.as_ref().and_then(|t| t.dest.clone()),
            test.is_some(),
        );
        // a test-hook run leaves the user's settings alone
        let prefs = settings::Prefs::load(match test {
            Some(_) => None,
            None => {
                crate::xdg("XDG_CONFIG_HOME", ".config").map(|d| d.join("x5desk").join("settings"))
            }
        });
        if let Some(i) = prefs.preset() {
            exp.set.apply(i)
        }
        let (mut pane, dls) = test
            .as_ref()
            .map_or((None, vec![]), |t| (t.camera, t.dl.clone()));
        // a test-hook run of another screen leaves the camera alone
        let preview = test.as_ref().is_some_and(|t| {
            t.out.pick == live::Pick::Preview
                || ch.as_ref().is_some_and(|h| h.on && h.webcam.is_none())
        });
        let usb = test.is_none() || pane.is_some() || preview;
        let sim = test.as_ref().and_then(|t| t.replay.as_ref()).and_then(|r| {
            let s = std::env::var_os("X5_SAMPLES").map(PathBuf::from)?;
            let c = test
                .as_ref()
                .and_then(|t| t.conf.clone())
                .or_else(|| std::env::var_os("X5_CONF").map(PathBuf::from));
            match replay::Replay::open(r, &s.join("v635.insv"), c.as_deref()) {
                Ok(mut x) => {
                    x.battery = test.as_ref().and_then(|t| t.battery).unwrap_or(x.battery);
                    Some(replay::connect(Arc::new(x)))
                }
                Err(e) => {
                    eprintln!("x5desk: replay: {e}");
                    None
                }
            }
        });
        let (mount, ble) = test
            .as_ref()
            .map_or((vec![], None), |t| (t.mount.clone(), t.ble));
        let mut disks = disk::Watch::new(mount.clone());
        if test.is_some() {
            // the hook's directories only
            disks.info = PathBuf::new();
        }
        let mut first = vec![
            bt::Cmd::Scan(true),
            bt::Cmd::Connect(::ble::Dev("sim/X5 SIM001".into())),
            bt::Cmd::Pair,
        ];
        match ble {
            Some(false) => first.extend([
                bt::Cmd::Ap,
                bt::Cmd::Join("HomeNet".into(), "correct horse".into()),
            ]),
            Some(true) => first.push(bt::Cmd::Sta),
            None => first.clear(),
        }
        let open: bt::Open = match ble {
            Some(s) => bt::sim(s),
            None => Box::new(::ble::Ble::open),
        };
        pane = pane
            .or(ble.map(|_| camera::Pane::Wireless))
            .or((!mount.is_empty()).then_some(camera::Pane::Album));
        let mut cam = camera::Cam::new(
            &cc.egui_ctx,
            pane,
            &dls,
            usb,
            sim,
            &prefs,
            (usb || !mount.is_empty()).then_some(disks),
            (open, first),
            test.is_none(),
        );
        if let Some(t) = &test {
            cam.import = t.import;
            if let Some(d) = &t.dest {
                cam.set_dest(d.clone(), false)
            }
        }
        let cap = capture::Capture::new(&g, &cx, &cc.egui_ctx, cam.tap.clone(), ch.clone());
        let live = live::Live::new(&g, &cx, &cc.egui_ctx, test.as_ref().map(|t| t.out.clone()));
        // the RAFT worker the player and the editor share
        let still = player::still(&g, &cx);
        let mut ed = editor::Editor::new(&g, &cx, &cc.egui_ctx, still.clone(), hook);
        ed.autox = (test.as_ref())
            .and_then(|t| t.autoedit.clone())
            .filter(|a| !a.is_empty());
        let lab = lab::Lab::new(
            &g,
            &cx,
            &cc.egui_ctx,
            test.as_ref().map(|t| t.lab.clone()).filter(|h| h.on),
        );
        let tab = match &test {
            Some(t) if t.live => Tab::Live,
            Some(t) if t.settings => Tab::Settings,
            Some(t) if t.help.is_some() => Tab::Help,
            _ if ed.first() => Tab::Editor,
            // a Deep Track box is drawn in the player's view: the Library first, Shot Lab once its preview starts
            _ if lab.first() && test.as_ref().is_none_or(|t| t.tbox.is_none()) => Tab::Lab,
            _ if ch.as_ref().is_some_and(|h| h.on) => Tab::Capture,
            _ if pane.is_some() => Tab::Camera,
            _ => Tab::Library,
        };
        let (grab, min, max, end) = test.as_ref().map_or((None, None, None, false), |t| {
            (
                t.grab.clone(),
                t.min,
                t.max.then_some((Duration::ZERO, 12)),
                t.end,
            )
        });
        let help = help::Help::new(&g, test.as_ref().and_then(|t| t.help));
        let set = settings::Settings::new(&g, prefs);
        let mut lib = library::Library::new(&g, &cx, &cc.egui_ctx, still, test);
        for p in cam.done() {
            lib.downloaded(&p);
        }
        if let Some(f) = file {
            lib.open(&f);
        }
        let shot = std::env::var_os("X5DESK_SHOT").map(|p| Shot {
            path: p.into(),
            after: Duration::from_secs_f64(
                std::env::var("X5DESK_SHOT_AFTER")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0),
            ),
            sent: false,
            done: Arc::new(AtomicBool::new(false)),
        });
        App {
            tab,
            lib,
            cam,
            cap,
            exp,
            ed,
            lab,
            set,
            help,
            auto,
            live,
            shot,
            exit,
            grab,
            min,
            max,
            end,
            t0: Instant::now(),
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        if let Some(p) = dropped.first() {
            match self.tab {
                Tab::Editor => {
                    for p in &dropped {
                        if let Err(e) = self.ed.add(p, None) {
                            self.ed.err = Some(e);
                        }
                    }
                }
                _ => {
                    self.tab = Tab::Library;
                    self.lib.open(p);
                }
            }
        }
        self.lib.tick();
        for p in self.cam.tick() {
            self.lib.downloaded(&p);
        }
        self.cap.tick();
        self.exp.tick();
        self.ed.tick(self.tab == Tab::Editor);
        self.lab.tick(self.lib.spec(), self.lib.track());
        if std::mem::take(&mut self.lab.show) {
            self.tab = Tab::Lab;
        }
        if let Some(a) = self.lab.ask.take() {
            match lab::send(a, &mut self.exp, &mut self.ed) {
                Ok(true) => self.tab = Tab::Editor,
                Ok(false) => self.tab = Tab::Export,
                Err(e) => {
                    eprintln!("x5desk lab: {e}");
                    self.lab.err = Some(e);
                }
            }
        }
        if let Some((p, st)) = self.lib.auto.take() {
            self.tab = Tab::Editor;
            self.ed.auto(&p, st);
        }
        if let Some(p) = self.lib.add.take() {
            self.tab = Tab::Editor;
            if let Err(e) = self.ed.add(&p, None) {
                self.ed.err = Some(e);
            }
        }
        if let Some((j, tok)) = self.ed.ask.take() {
            match export::parse(self.exp.set, &tok).and_then(|x| self.exp.timeline(j, x)) {
                Ok(()) => self.tab = Tab::Export,
                Err(e) => {
                    eprintln!("x5desk editor: export: {e}");
                    self.ed.err = Some(e);
                }
            }
        }
        if let Some((s, grab)) = self.lib.ask.take() {
            self.exp.source(s);
            if let Some(Err(e)) = grab.map(|p| self.exp.grab(p)) {
                self.exp.err = Some(e);
            }
            self.tab = Tab::Export;
        }
        if self.auto.is_some()
            && self.lib.idle()
            && let Some(s) = self.lib.spec()
            && let Some((ex, fr)) = self.auto.take()
        {
            self.exp.source(s);
            for e in ex {
                if let Err(x) = self.exp.auto(&e) {
                    eprintln!("x5desk: export={e}: {x}");
                }
            }
            for p in fr {
                if let Err(e) = self.exp.grab(p) {
                    eprintln!("x5desk: frame: {e}");
                }
            }
            self.tab = Tab::Export;
        }
        self.live.tick(match self.live.pick {
            live::Pick::Player => self.lib.source(),
            live::Pick::Preview => self.cap.source(capture::Src::Preview),
            live::Pick::Webcam => self.cap.source(capture::Src::Webcam),
        });
        egui::Panel::left("nav")
            .resizable(false)
            .exact_size(110.0)
            .show(ui, |ui| {
                ui.add_space(8.0);
                ui.heading("x5desk");
                ui.add_space(8.0);
                for (t, name) in TABS {
                    if t == Tab::Settings {
                        ui.separator();
                    }
                    ui.selectable_value(&mut self.tab, t, name);
                }
                // the X5 shut down at 9 % with no word from the app (2026-10-23): a low battery shows on every screen,
                // the capture worker's level first (it polls every 10 s)
                if let Some(b) =
                    (self.cap.st.o.battery_info.as_ref()).or(self.cam.info.battery.as_ref())
                    && camera::low(b)
                {
                    ui.add_space(16.0);
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        format!("⚠ Camera battery {} %", camera::level(b).0),
                    )
                    .on_hover_text("Charge the X5: it shuts down when the battery is empty");
                }
            });
        egui::CentralPanel::default_margins().show(ui, |ui| {
            match TABS.iter().find(|t| t.0 == self.tab) {
                Some((Tab::Library, ..)) => match frame.wgpu_render_state() {
                    Some(rs) => self.lib.ui(ui, rs),
                    None => {
                        ui.label("No wgpu renderer.");
                    }
                },
                Some((Tab::Live, ..)) => match frame.wgpu_render_state() {
                    Some(rs) => {
                        egui::ScrollArea::vertical()
                            .auto_shrink(false)
                            .stick_to_bottom(self.end)
                            .show(ui, |ui| self.live.ui(ui, rs));
                    }
                    None => {
                        ui.label("No wgpu renderer.");
                    }
                },
                Some((Tab::Camera, ..)) => {
                    if let Some(p) = self.cam.ui(ui) {
                        self.tab = Tab::Library;
                        self.lib.open(&p);
                    }
                }
                Some((Tab::Capture, ..)) => match frame.wgpu_render_state() {
                    Some(rs) => self.cap.ui(ui, rs),
                    None => {
                        ui.label("No wgpu renderer.");
                    }
                },
                Some((Tab::Lab, ..)) => self.lab.ui(ui, self.exp.set),
                Some((Tab::Editor, ..)) => match frame.wgpu_render_state() {
                    Some(rs) => {
                        let media = self.lib.videos();
                        self.ed.ui(ui, rs, &media);
                    }
                    None => {
                        ui.label("No wgpu renderer.");
                    }
                },
                Some((Tab::Export, ..)) => {
                    if let Some(p) = self.exp.ui(ui) {
                        self.tab = Tab::Library;
                        self.lib.open(&p);
                    }
                }
                Some((Tab::Settings, ..)) => {
                    let n = ui.ctx().cumulative_pass_nr();
                    self.set.ui(ui, n, &mut self.cam, &mut self.exp)
                }
                Some((Tab::Help, ..)) => self.help.ui(ui),
                None => {}
            }
        });
        // test hooks
        let ready = self.t0.elapsed() >= self.shot.as_ref().map_or(Duration::ZERO, |s| s.after)
            && self.lib.idle()
            && (self.tab != Tab::Camera || self.cam.idle())
            && self.cap.idle()
            && (self.tab != Tab::Editor || self.ed.idle())
            && self.ed.ask.is_none()
            && self.auto.is_none()
            && self.lab.idle()
            && self.lib.auto.is_none()
            && self.exp.idle()
            && self.max.is_none();
        // an X11 window manager ignores a maximise for a window it has not mapped, and eframe maps it after its first
        // frame (winit sends `with_maximized` before): asked every 250 ms until it is, for 3 s (one without EWMH, or
        // bare Xvfb, never maximises)
        if let Some((t, n)) = self.max {
            let now = self.t0.elapsed();
            self.max = match ctx.input(|i| i.viewport().maximized) {
                Some(true) => None,
                _ if now < t => self.max,
                _ if n == 0 => {
                    let s = ctx.input(|i| i.content_rect().size() * i.pixels_per_point);
                    eprintln!(
                        "x5desk: not maximised in 3 s; going on at {:.0} × {:.0}",
                        s.x, s.y
                    );
                    None
                }
                _ => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
                    Some((now + Duration::from_millis(250), n - 1))
                }
            };
            if self.max.is_some() {
                ctx.request_repaint_after(Duration::from_millis(50))
            }
        }
        if self.lib.idle() && self.grab.is_some() {
            self.live.grab = self.grab.take();
        }
        if let Some(s) = self.min {
            match self.t0.elapsed().as_secs_f64() >= s {
                true => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                    self.min = None;
                }
                false => ctx.request_repaint_after(Duration::from_millis(100)),
            }
        }
        // a player that never settles still ends the run (exports take their time)
        let late = self.t0.elapsed() > Duration::from_secs(120)
            && self.auto.is_none()
            && self.ed.ask.is_none()
            && self.exp.idle();
        match &mut self.shot {
            Some(s) if s.done.load(Relaxed) => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            Some(s) if !s.sent && (ready || late) => {
                let (p, d) = (s.path.clone(), s.done.clone());
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                    egui::ScreenshotCallback::new(move |img| {
                        let px: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                        let (w, h) = (img.size[0] as u32, img.size[1] as u32);
                        match png::write(&p, w, h, &px) {
                            Ok(()) => eprintln!("x5desk: screenshot {w} × {h} → {}", p.display()),
                            Err(e) => eprintln!("x5desk: {}: {e}", p.display()),
                        }
                        d.store(true, Relaxed);
                    }),
                ));
                s.sent = true;
                ctx.request_repaint();
            }
            None if self.exit && (ready || late) => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close)
            }
            _ => {}
        }
        if self.shot.is_some() || self.exit {
            ctx.request_repaint_after(Duration::from_millis(20));
        }
        // last, holding no lock the workers take (they finish what they submit, then wait for the retry), and in the
        // pass egui paints: a discarded pass drops its shapes before the paint
        if !ctx.will_discard()
            && RACE.swap(false, Relaxed)
            && let Some(mut c) = frame.wgpu_surface_config()
        {
            // egui configures again when the config changes: Some(2) and None (wgpu's default, 2) are the same
            c.desired_maximum_frame_latency = match c.desired_maximum_frame_latency {
                Some(_) => None,
                None => Some(2),
            };
            frame.set_wgpu_surface_config(c);
            let b = BURN.lock().unwrap_or_else(PoisonError::into_inner);
            let w = GATE.write().unwrap_or_else(PoisonError::into_inner);
            HELD.set(Some((b, w)));
            let r = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1.0, 1.0));
            ui.painter()
                .add(egui_wgpu::Callback::new_paint_callback(r, Retry));
        }
    }
}

fn main() -> eframe::Result {
    let file = std::env::args_os().nth(1).map(PathBuf::from);
    let test = std::env::var("X5DESK_TEST").ok().map(|s| Test::parse(&s));
    let g = Arc::new(video::Gpu::new().unwrap_or_else(|e| {
        eprintln!("x5desk: {e}");
        std::process::exit(1)
    }));
    // burn's device measurements and autotune results; the test hook's runs share a dir of their own (warm too)
    let cc = match test {
        Some(_) => mine(
            std::env::var_os("XDG_RUNTIME_DIR")
                .map_or_else(std::env::temp_dir, PathBuf::from)
                .join("x5desk-cubecl"),
        ),
        None => xdg("XDG_CACHE_HOME", ".cache").map(|d| d.join("x5desk/cubecl")),
    };
    stitch::cache(cc.as_deref());
    // cubecl's handler panics on the surface race: attach first, so ours replaces it for good (RAFT and the
    // trackers reuse the attachment)
    if let Err(e) = stitch::attach(&g.instance, &g.adapter, &g.device, &g.queue) {
        eprintln!("x5desk: {e}");
    }
    errors(&g.device);
    ::export::locks(&GATE, &BURN);
    let setup = egui_wgpu::WgpuSetupExisting {
        instance: g.instance.clone(),
        adapter: g.adapter.clone(),
        device: g.device.clone(),
        queue: g.queue.clone(),
    };
    let opts = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::Existing(setup),
            surface: egui_wgpu::SurfaceConfig {
                present_mode: match test.as_ref().is_some_and(|t| t.novsync) {
                    true => wgpu::PresentMode::AutoNoVsync,
                    false => wgpu::PresentMode::AutoVsync,
                },
                desired_maximum_frame_latency: Some(2),
            },
            ..Default::default()
        },
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title("x5desk")
            .with_app_id("x5desk")
            .with_drag_and_drop(true),
        // video pixels as decoded
        dithering: false,
        ..Default::default()
    };
    eframe::run_native(
        "x5desk",
        opts,
        Box::new(move |cc| Ok(Box::new(App::new(cc, g, file, test)))),
    )
}

#[cfg(test)]
mod tests {
    use wgpu::wgc::{error::ContextError, present::ConfigureSurfaceError};

    #[test]
    fn capture_tokens() {
        let t = super::Test::parse(
            "replay=/r/live.h264,capture,view=equirect,yaw=90,cstats=30,src=preview",
        );
        let h = t.cap.unwrap();
        assert_eq!(
            t.replay.as_deref(),
            Some(std::path::Path::new("/r/live.h264"))
        );
        assert_eq!(
            (h.on, h.look, h.yaw, h.stats),
            (
                true,
                Some(super::player::Look::Equirect),
                Some(90.0),
                Some(30.0)
            )
        );
        assert_eq!(t.out.pick, super::live::Pick::Preview);
    }

    #[test]
    fn test_tokens() {
        let t = super::Test::parse("seek=300,stab=direction,smooth=1.5");
        assert_eq!(
            (t.seek, t.stab, t.smooth),
            (Some(300), Some(Some(stab::Lock::Direction)), Some(1.5))
        );
        assert_eq!(super::Test::parse("stab=off").stab, Some(None));
        assert_eq!(super::Test::parse("play").stab, None);
        let ble = |v| super::Test::parse(v).ble;
        assert_eq!(
            (ble("ble=sim"), ble("ble=sta"), ble("ble=x")),
            (Some(false), Some(true), None)
        );
    }

    /// The hook's cubecl dir: made owner-only; refused when another user owns it (here root), when it is a symlink (to
    /// a dir of ours too) or when others can write to it.
    #[test]
    fn cubecl_dir() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        struct Rm(std::path::PathBuf);
        impl Drop for Rm {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let run = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(std::env::temp_dir, Into::into);
        let b = Rm(run.join(format!("x5desk-mine-{}", std::process::id())));
        std::fs::create_dir(&b.0).unwrap();
        let mode = |d: &std::path::Path| std::fs::metadata(d).unwrap().permissions().mode() & 0o777;
        let d = b.0.join("x5desk-cubecl");
        assert_eq!(super::mine(d.clone()), Some(d.clone()));
        assert_eq!(mode(&d), 0o700);
        assert_eq!(super::mine("/".into()), None);
        let l = b.0.join("link");
        symlink(&d, &l).unwrap();
        assert_eq!(super::mine(l), None);
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert_eq!(super::mine(d.clone()), None);
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(super::mine(d.clone()), Some(d));
    }

    /// The error wgpu 30 reports for `Surface::configure` when another thread submitted during its idle wait, and
    /// another one of the same call.
    #[test]
    fn surface_race() {
        let e = |c: ConfigureSurfaceError| wgpu::Error::Validation {
            source: Box::new(ContextError {
                fn_ident: "Surface::configure",
                source: Box::new(c),
                label: String::new(),
            }),
            description: String::new(),
        };
        assert!(super::race(&e(ConfigureSurfaceError::GpuWaitTimeout)));
        assert!(!super::race(&e(ConfigureSurfaceError::ZeroArea)));
    }
}
