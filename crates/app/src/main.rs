//! x5desk: desktop app for the Insta360 X5. One eframe window on the shared Vulkan device (`video::Gpu`); a side
//! panel switches screens. `x5desk [file]` opens a file.
//!
//! Test hooks: `X5DESK_SHOT=<file.png>` (with `X5DESK_SHOT_AFTER=<s>`, default 1) saves a screenshot once that
//! time has passed and the player has nothing pending, then exits. `X5DESK_TEST=<tokens>` (comma separated) sets up
//! the first file opened: `play`, `loop`, `vol=<0..1>` (default 0.1 under the hook), `view=perspective|panini|
//! equirect|planet`, `seek=<frame>`, `proxy`, `stats=<s>` (play, print pacing statistics after s seconds, exit),
//! `novsync` (present without waiting for vblank, repaint on a 60 Hz timer: for a compositor that sends no frame
//! callbacks, e.g. with the display off), `full` (no automatic switch to the proxy), `stab=off|free|horizon|direction`
//! (default horizon), `smooth=<s>` (the stabilisation's σ, default 0.5), `camera[=album|downloads|settings]` (start on
//! the Camera screen; with `X5DESK_SHOT` the shot waits for the album's thumbnails), `dl=<camera URI>` (queue a
//! download, repeatable), `live` (start on the Live screen), `rtmp=<url>` (stream there once the file shows a
//! picture; the key is the URL's last segment), `res=1080|1440|360`, `fps=<n>`, `kbps=<n>`, `obs` (serve the OBS
//! source), `pwcam` (publish the PipeWire camera), `grab=<file.png>` (once the player is idle, e.g. a paused
//! frame's RAFT stitch done: save the live outputs' `view` and `360` pictures as `<file>-view.png`,
//! `<file>-360.png`), `min=<s>` (minimise the window after s seconds: the compositor stops drawing it, the live outputs
//! run on). Under the hook the Live screen neither reads nor writes the user's live settings, and the live output
//! thread prints its state every 5 s.

mod album;
mod camera;
mod clip;
mod dl;
mod gyro;
mod library;
mod live;
mod player;
mod png;

use eframe::{egui, egui_wgpu};
use std::{
    cell::RefCell,
    path::PathBuf,
    sync::{
        Arc, PoisonError, RwLock, RwLockWriteGuard,
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
    pub out: live::Hook,
    pub grab: Option<PathBuf>,
    pub min: Option<f64>,
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
                "play" => t.play = true,
                "loop" => t.looping = true,
                "proxy" => t.proxy = true,
                "novsync" => t.novsync = true,
                "full" => t.full = true,
                "vol" => t.vol = v.parse().ok(),
                "seek" => t.seek = v.parse().ok(),
                "stats" => t.stats = v.parse().ok(),
                "view" => t.look = player::Look::parse(v),
                "stab" => t.stab = gyro::parse(v),
                "smooth" => t.smooth = v.parse().ok(),
                "camera" => t.camera = camera::Pane::parse(v),
                "dl" => t.dl.push(v.into()),
                "live" => t.live = true,
                "rtmp" => t.out.rtmp = Some(v.into()),
                "res" => t.out.res = ["1080", "1440", "360"].iter().position(|r| *r == v),
                "fps" => t.out.fps = v.parse().ok(),
                "kbps" => t.out.kbps = v.parse().ok(),
                "obs" => t.out.obs = true,
                "pwcam" => t.out.cam = true,
                "grab" => t.grab = Some(v.into()),
                "min" => t.min = v.parse().ok(),
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

/// wgpu-core's `Surface::configure` waits for the queue to go idle and fails (`GpuWaitTimeout`) when another thread
/// submits meanwhile. egui-wgpu then draws the new window size into the old swapchain, and on native Wayland nothing
/// reports that swapchain `Outdated`. So the app retries it: the threads of its own that submit on their own (the
/// live outputs, the RAFT worker) hold this shared for their GPU work, and after a failed configure the UI thread
/// holds it exclusively from the end of its next frame's UI until egui has configured the surface again and acquired
/// that frame ([`Retry`]). The decoder's empty flush submit stays outside: a decoder waiting for a slot would wait
/// for the UI's next submit.
pub static GATE: RwLock<()> = RwLock::new(());
/// a configure failed that way; the next frame retries it
static RACE: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// the UI thread's hold on [`GATE`] during a retry
    static HELD: RefCell<Option<RwLockWriteGuard<'static, ()>>> = const { RefCell::new(None) };
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
    Camera,
    Capture,
    Live,
    Export,
}

/// Screens: name, and for the ones not built yet the milestone that brings them (PLAN.md §4).
const TABS: [(Tab, &str, &str); 5] = [
    (Tab::Library, "Library", ""),
    (Tab::Camera, "Camera", ""),
    (
        Tab::Capture,
        "Capture",
        "Live preview, capture controls and the shutter come with milestone 3.",
    ),
    (Tab::Live, "Live", ""),
    (Tab::Export, "Export", "Export comes with milestone 5."),
];

struct Shot {
    path: PathBuf,
    after: Duration,
    sent: bool,
    done: Arc<AtomicBool>,
}

struct App {
    tab: Tab,
    lib: library::Library,
    cam: camera::Cam,
    live: live::Live,
    shot: Option<Shot>,
    /// the test hook ends the run once the player is idle
    exit: bool,
    /// the test hook saves the live outputs' pictures once the player is idle, and minimises the window after `min` s
    grab: Option<PathBuf>,
    min: Option<f64>,
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
        let exit = test.as_ref().is_some_and(|t| t.stats.is_some());
        let (pane, dls) = test
            .as_ref()
            .map_or((None, vec![]), |t| (t.camera, t.dl.clone()));
        // a test-hook run of another screen leaves the camera alone
        let usb = test.is_none() || pane.is_some();
        let cam = camera::Cam::new(&cc.egui_ctx, pane, &dls, usb);
        let live = live::Live::new(&g, &cx, &cc.egui_ctx, test.as_ref().map(|t| t.out.clone()));
        let tab = match &test {
            Some(t) if t.live => Tab::Live,
            _ if pane.is_some() => Tab::Camera,
            _ => Tab::Library,
        };
        let (grab, min) = test
            .as_ref()
            .map_or((None, None), |t| (t.grab.clone(), t.min));
        let mut lib = library::Library::new(&g, &cx, &cc.egui_ctx, test);
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
            live,
            shot,
            exit,
            grab,
            min,
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
            self.tab = Tab::Library;
            self.lib.open(p);
        }
        self.lib.tick();
        for p in self.cam.tick() {
            self.lib.downloaded(&p);
        }
        self.live.tick(self.lib.source());
        egui::Panel::left("nav")
            .resizable(false)
            .exact_size(110.0)
            .show(ui, |ui| {
                ui.add_space(8.0);
                ui.heading("x5desk");
                ui.add_space(8.0);
                for (t, name, _) in TABS {
                    ui.selectable_value(&mut self.tab, t, name);
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
                    Some(rs) => self.live.ui(ui, rs),
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
                Some((_, name, note)) => {
                    ui.heading(*name);
                    ui.label(*note);
                }
                None => {}
            }
        });
        // test hooks
        let ready = self.t0.elapsed() >= self.shot.as_ref().map_or(Duration::ZERO, |s| s.after)
            && self.lib.idle()
            && (self.tab != Tab::Camera || self.cam.idle());
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
        // a player that never settles still ends the run
        let late = self.t0.elapsed() > Duration::from_secs(120);
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
            HELD.set(Some(GATE.write().unwrap_or_else(PoisonError::into_inner)));
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
    // cubecl's handler panics on the surface race: attach first, so ours replaces it for good (RAFT and the
    // trackers reuse the attachment)
    if let Err(e) = stitch::attach(&g.instance, &g.adapter, &g.device, &g.queue) {
        eprintln!("x5desk: {e}");
    }
    errors(&g.device);
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
    fn test_tokens() {
        let t = super::Test::parse("seek=300,stab=direction,smooth=1.5");
        assert_eq!(
            (t.seek, t.stab, t.smooth),
            (Some(300), Some(Some(stab::Lock::Direction)), Some(1.5))
        );
        assert_eq!(super::Test::parse("stab=off").stab, Some(None));
        assert_eq!(super::Test::parse("play").stab, None);
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
