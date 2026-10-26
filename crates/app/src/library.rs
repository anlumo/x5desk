//! Library: local files opened from the dialog, drag and drop or the command line; the recent list (config dir; an
//! entry whose file is not there, such as one on a card that is not mounted, is greyed out until it is back);
//! pairing of a recording's files by name (lens files `_00_` / `_10_`, proxies `_01_` / `_11_`); the player of the
//! open work, shared with the live output thread, the viewer of the open photo, or of a plain video (an export). Export
//! jobs start here: from the player or the photo (its view), or from a file's menu in the lists.

use crate::{
    Test,
    export::Spec,
    gyro, movie, photo,
    player::{self, Player},
};
use eframe::{egui, egui_wgpu::RenderState};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};
use video::Gpu;

/// How often the lists' files are looked for.
const LOOK: Duration = Duration::from_secs(2);

/// The files of one recording: main files in lens order (one with both lenses on the X5), the proxy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Work {
    pub main: Vec<PathBuf>,
    pub lrv: Option<PathBuf>,
}

fn name(p: &Path) -> &str {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("")
}

fn ext(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Recording key: the name without its extension and its stream parts (`00`, `01`, `10`, `11`), `LRV` read as
/// `VID` (the official app's fileKey, re/analysis/media.md §3.4).
pub fn key(name: &str) -> String {
    let stem = name.rsplit_once('.').map_or(name, |s| s.0);
    let mut k = String::new();
    for p in stem
        .split('_')
        .filter(|p| !matches!(*p, "00" | "01" | "10" | "11"))
    {
        k += if p.eq_ignore_ascii_case("lrv") {
            "VID"
        } else {
            p
        };
        k += "_";
    }
    k
}

/// The stream part (`00` lens 0 or both, `10` lens 1, `01` / `11` their proxies); files without one sort first.
fn part(p: &Path) -> u8 {
    let n = name(p);
    let n = n.rsplit_once('.').map_or(n, |s| s.0);
    n.split('_')
        .find_map(|x| ["00", "01", "10", "11"].iter().position(|s| *s == x))
        .map_or(0, |i| i as u8 + 1)
}

/// The work `p` belongs to among `all` (its directory).
pub fn group(p: &Path, all: &[PathBuf]) -> Work {
    let k = key(name(p));
    let (mut main, mut lrv) = (vec![], vec![]);
    for q in all.iter().map(PathBuf::as_path).chain([p]) {
        let v = match ext(q).as_str() {
            "insv" | "mp4" => &mut main,
            "lrv" => &mut lrv,
            _ => continue,
        };
        if key(name(q)) == k && !v.contains(&q) {
            v.push(q);
        }
    }
    main.sort_by_key(|q| (part(q), name(q).to_string()));
    lrv.sort_by_key(|q| (part(q), name(q).to_string()));
    Work {
        main: main.into_iter().map(Path::to_path_buf).collect(),
        lrv: lrv.first().map(|q| q.to_path_buf()),
    }
}

pub fn pair(p: &Path) -> Work {
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty());
    let all: Vec<PathBuf> = std::fs::read_dir(dir.unwrap_or(Path::new(".")))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .collect();
    group(p, &all)
}

/// Recently opened files, newest first, one path per line in `<config>/x5desk/recent`.
pub struct Recent {
    pub list: Vec<PathBuf>,
    file: Option<PathBuf>,
}

const RECENT: usize = 20;

impl Recent {
    pub fn load() -> Recent {
        Recent::at(
            crate::xdg("XDG_CONFIG_HOME", ".config").map(|d| d.join("x5desk").join("recent")),
        )
    }

    pub fn at(file: Option<PathBuf>) -> Recent {
        let list = file
            .as_ref()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .map(|s| {
                s.lines()
                    .filter(|l| !l.is_empty())
                    .map(PathBuf::from)
                    .collect()
            })
            .unwrap_or_default();
        Recent { list, file }
    }

    pub fn add(&mut self, p: &Path) {
        self.list.retain(|q| q != p);
        self.list.insert(0, p.to_path_buf());
        self.list.truncate(RECENT);
        let Some(f) = &self.file else { return };
        let mut s = String::new();
        for p in &self.list {
            s += &p.to_string_lossy();
            s += "\n";
        }
        if let Err(e) = f
            .parent()
            .map_or(Ok(()), crate::private)
            .and_then(|_| std::fs::write(f, s))
        {
            eprintln!("{}: {e}", f.display());
        }
    }
}

pub struct Library {
    g: Arc<Gpu>,
    cx: stitch::Ctx,
    ctx: egui::Context,
    recent: Recent,
    /// downloaded recordings (from the camera screen), newest first, one file per work
    down: Vec<PathBuf>,
    /// files of the lists that are not there (on a card that is not mounted): greyed out until they are back; looked
    /// for off the UI thread (a stat can hang on a card pulled without an unmount), the next look
    gone: HashSet<PathBuf>,
    look: Option<mpsc::Receiver<HashSet<PathBuf>>>,
    next: Instant,
    player: Option<Arc<Mutex<Player>>>,
    photo: Option<photo::Photo>,
    movie: Option<movie::Movie>,
    /// for the export screen: a source, and a frame grab of it (360 or the view) or None (show the settings)
    pub ask: Option<(Spec, Option<bool>)>,
    /// for the editor: a recording to append to its timeline; one to make an auto edit of, with the player's
    /// stabilisation
    pub add: Option<PathBuf>,
    pub auto: Option<(PathBuf, (crate::gyro::Mode, f64))>,
    still: mpsc::Sender<player::Job>,
    gyro: mpsc::Sender<gyro::Req>,
    dlg: Option<mpsc::Receiver<Option<PathBuf>>>,
    err: Option<String>,
    test: Option<Test>,
    side: bool,
}

impl Library {
    /// `still`: the app's RAFT worker (`player::still`).
    pub fn new(
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        ctx: &egui::Context,
        still: mpsc::Sender<player::Job>,
        test: Option<Test>,
    ) -> Library {
        // a test-hook run neither reads nor writes the user's list
        let recent = match test {
            Some(_) => Recent::at(None),
            None => Recent::load(),
        };
        Library {
            g: g.clone(),
            cx: cx.clone(),
            ctx: ctx.clone(),
            recent,
            down: vec![],
            gone: HashSet::new(),
            look: None,
            next: Instant::now(),
            player: None,
            photo: None,
            movie: None,
            ask: None,
            add: None,
            auto: None,
            still,
            gyro: gyro::worker(),
            dlg: None,
            err: None,
            test,
            side: true,
        }
    }

    pub fn open(&mut self, p: &Path) {
        let p = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        // the old player goes first: its decoders free their memory before the new ones allocate
        (self.player, self.photo, self.movie) = (None, None, None);
        if photo::is(&p) {
            let t = self.test.take();
            let mut ph = photo::Photo::open(
                &self.g,
                &self.cx,
                &p,
                &self.ctx,
                t.as_ref().and_then(|t| t.look),
            );
            if let Some([y, x, f]) = t.and_then(|t| t.aim) {
                (ph.view.yaw, ph.view.pitch) = (y, x);
                ph.view.fov = if f.is_nan() { ph.view.fov } else { f };
            }
            self.photo = Some(ph);
            self.recent.add(&p);
            self.err = None;
            return;
        }
        // no trailer: a plain video, e.g. one of our exports
        if ext(&p) == "mp4" && media::insv::Trailer::open(&p).is_err() {
            let t = self.test.take();
            match movie::Movie::open(
                &self.g,
                &self.cx,
                &p,
                &self.ctx,
                t.as_ref().and_then(|t| t.look),
            ) {
                Ok(mut m) => {
                    m.vol(t.and_then(|t| t.vol).unwrap_or(1.0));
                    self.movie = Some(m);
                    self.recent.add(&p);
                    self.err = None;
                }
                Err(e) => self.err = Some(format!("{}: {e}", p.display())),
            }
            return;
        }
        let w = pair(&p);
        // one entry per recording, whichever of its files was opened
        let r = w.main.first().or(w.lrv.as_ref()).unwrap_or(&p).clone();
        match Player::open(
            &self.g,
            &self.cx,
            w,
            &self.ctx,
            &self.still,
            &self.gyro,
            self.test.take(),
        ) {
            Ok(pl) => {
                self.recent.add(&r);
                self.player = Some(Arc::new(Mutex::new(pl)));
                self.err = None;
            }
            Err(e) => self.err = Some(format!("{}: {e}", p.display())),
        }
    }

    /// A file the camera screen downloaded: its recording or the photo joins the downloaded list.
    pub fn downloaded(&mut self, p: &Path) {
        let w = match ext(p).as_str() {
            "insv" | "lrv" | "mp4" => pair(p),
            _ if photo::is(p) => Work {
                main: vec![p.to_path_buf()],
                lrv: None,
            },
            _ => return,
        };
        let Some(r) = w.main.first().or(w.lrv.as_ref()) else {
            return;
        };
        let k = key(name(r));
        self.down.retain(|q| key(name(q)) != k);
        self.down.insert(0, r.clone());
    }

    /// Every frame, whatever screen is shown: the clock, frame hand-over, finished dialogs, the lists' files.
    pub fn tick(&mut self) {
        if let Some(r) = &self.dlg
            && let Ok(p) = r.try_recv()
        {
            self.dlg = None;
            if let Some(p) = p {
                self.open(&p);
            }
        }
        if let Some(g) = self.look.as_ref().and_then(|r| r.try_recv().ok()) {
            (self.gone, self.look) = (g, None);
        }
        let now = Instant::now();
        if self.look.is_none() && now >= self.next {
            let (tx, rx) = mpsc::channel();
            let l: Vec<PathBuf> = self.recent.list.iter().chain(&self.down).cloned().collect();
            let (old, ctx) = (self.gone.clone(), self.ctx.clone());
            thread::spawn(move || {
                let g: HashSet<PathBuf> = l.into_iter().filter(|p| !p.exists()).collect();
                let new = g != old;
                if tx.send(g).is_ok() && new {
                    ctx.request_repaint()
                }
            });
            (self.look, self.next) = (Some(rx), now + LOOK);
        }
        if let Some(p) = &self.player {
            let mut p = p.lock().unwrap();
            p.tick(true);
            if std::mem::take(&mut p.auto)
                && let Some(f) = p.work.main.first()
            {
                self.auto = Some((f.clone(), p.stab()));
            }
        }
        if let Some(p) = &mut self.photo {
            p.tick();
        }
        if let Some(m) = &mut self.movie {
            m.tick(&self.ctx);
        }
    }

    /// The open work or photo as the export screen takes it (None: nothing open, or a proxy without its main file).
    pub fn spec(&self) -> Option<Spec> {
        match (&self.player, &self.photo) {
            (Some(p), _) => p.lock().unwrap().spec(),
            (_, Some(p)) => Some(p.spec()),
            _ => None,
        }
    }

    /// The videos of the recent and downloaded lists that are there (the editor's media), one per recording.
    pub fn videos(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = vec![];
        for p in self.down.iter().chain(&self.recent.list) {
            if matches!(ext(p).as_str(), "insv" | "lrv" | "mp4")
                && !self.gone.contains(p)
                && !v.iter().any(|q| key(name(q)) == key(name(p)))
            {
                v.push(p.clone());
            }
        }
        v
    }

    /// The open player's Deep Track run.
    pub fn track(&self) -> Option<Arc<crate::ai::Track>> {
        self.player.as_ref()?.lock().unwrap().trk.clone()
    }

    /// The open player, for the live outputs.
    pub fn source(&self) -> Option<crate::live::Src> {
        self.player.clone().map(|p| p as _)
    }

    /// Done with what the test hook asked for (`stats=`), ready for a screenshot.
    pub fn idle(&self) -> bool {
        self.player
            .as_ref()
            .is_none_or(|p| p.lock().unwrap().idle())
            && self.photo.as_ref().is_none_or(|p| p.idle())
            && self.movie.as_ref().is_none_or(|m| m.idle())
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        egui::Panel::top("lib bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked() && self.dlg.is_none() {
                    let (tx, rx) = mpsc::channel();
                    let ctx = self.ctx.clone();
                    // the portal dialog blocks: off the UI thread
                    thread::spawn(move || {
                        let p = rfd::FileDialog::new()
                            .add_filter(
                                "X5 video or photo",
                                &["insv", "lrv", "mp4", "jpg", "jpeg", "insp"],
                            )
                            .pick_file();
                        let _ = tx.send(p);
                        ctx.request_repaint();
                    });
                    self.dlg = Some(rx);
                }
                ui.toggle_value(&mut self.side, "Files");
                let spec = self.spec();
                if (self.player.is_some() || self.photo.is_some())
                    && ui
                        .add_enabled(spec.is_some(), egui::Button::new("Export…"))
                        .on_hover_text("Export settings for this file and view")
                        .on_disabled_hover_text(
                            "The proxy alone does not export: open the main file",
                        )
                        .clicked()
                {
                    self.ask = spec.clone().map(|s| (s, None));
                }
                if let Some(s) = spec.filter(|s| !s.photo) {
                    if ui
                        .button("Grab 360")
                        .on_hover_text("This frame as a 360 JPEG")
                        .clicked()
                    {
                        self.ask = Some((s.clone(), Some(true)));
                    }
                    if ui
                        .button("Grab view")
                        .on_hover_text("This view as a JPEG")
                        .clicked()
                    {
                        self.ask = Some((s, Some(false)));
                    }
                }
                if let Some(p) = &self.player {
                    let p = p.lock().unwrap();
                    let w = &p.work;
                    let mut s: Vec<&str> = w.main.iter().map(|p| name(p)).collect();
                    if let Some(l) = &w.lrv {
                        s.push(name(l));
                    }
                    ui.label(s.join(" + "));
                }
                if let Some(p) = &self.photo {
                    ui.label(name(&p.path));
                }
                if let Some(m) = &self.movie {
                    ui.label(name(&m.path));
                }
            });
            if let Some(e) = &self.err {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
        });
        if self.side {
            let (mut pick, mut exp, mut add) = (None, None, None);
            // the next look for the lists' files, also while nothing else repaints
            ui.ctx().request_repaint_after(LOOK);
            egui::Panel::left("recent")
                .default_size(240.0)
                .show(ui, |ui| {
                    for (h, l) in [("Recent", &self.recent.list), ("Downloaded", &self.down)] {
                        ui.heading(h);
                        egui::ScrollArea::vertical()
                            .id_salt(h)
                            .max_height(ui.available_height() / 2.0)
                            .show(ui, |ui| {
                                for p in l {
                                    let b = egui::Button::selectable(false, name(p));
                                    let r = ui
                                        .add_enabled(!self.gone.contains(p), b)
                                        .on_hover_text(p.to_string_lossy())
                                        .on_disabled_hover_text(format!(
                                            "{}: not there (on a card that is not mounted?)",
                                            p.display()
                                        ));
                                    if r.clicked() {
                                        pick = Some(p.clone());
                                    }
                                    r.context_menu(|ui| {
                                        if ui.button("Export…").clicked() {
                                            exp = Some(p.clone());
                                        }
                                        if !photo::is(p) && ui.button("Add to editor").clicked() {
                                            add = Some(p.clone());
                                        }
                                    });
                                }
                            });
                        ui.add_space(8.0);
                    }
                });
            if let Some(p) = pick {
                self.open(&p);
            }
            if add.is_some() {
                self.add = add;
            }
            if let Some(p) = exp {
                match Spec::of(&p) {
                    Ok(s) => self.ask = Some((s, None)),
                    Err(e) => self.err = Some(e),
                }
            }
        }
        egui::CentralPanel::no_frame().show(ui, |ui| match (&self.player, &mut self.photo, &mut self.movie) {
            (Some(p), ..) => p.lock().unwrap().ui(ui, rs),
            (_, Some(p), _) => p.ui(ui, rs),
            (.., Some(m)) => m.ui(ui, rs),
            _ => {
                ui.centered_and_justified(|ui| {
                    ui.label("Open an X5 video (.insv, .lrv) or photo (.jpg, .insp), or drop one here.")
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys() {
        let k = key("VID_20250101_120000_00_001.insv");
        assert_eq!(k, key("VID_20250101_120000_10_001.insv"));
        assert_eq!(k, key("LRV_20250101_120000_01_001.lrv"));
        assert_eq!(k, key("LRV_20250101_120000_11_001.lrv"));
        assert_ne!(k, key("VID_20250101_120000_00_002.insv"));
        assert_ne!(k, key("PRO_VID_20250101_120000_00_001.insv"));
        assert_eq!(
            key("PRO_VID_20250101_120000_00_001.insv"),
            key("PRO_LRV_20250101_120000_01_001.lrv")
        );
        assert_eq!(key("v635.insv"), key("v635.lrv"));
        assert_ne!(key("v635.insv"), key("v606.insv"));
    }

    #[test]
    fn pairs() {
        let all: Vec<PathBuf> = [
            "VID_20250101_120000_10_001.insv",
            "LRV_20250101_120000_11_001.lrv",
            "LRV_20250101_120000_01_001.lrv",
            "VID_20250101_120000_00_001.insv",
            "VID_20250101_120500_00_002.insv",
            "LRV_20250101_120500_01_002.lrv",
            "IMG_20250101_120000_00_003.insp",
            "v635.insv",
            "v635.lrv",
            "v606.insv",
        ]
        .iter()
        .map(|n| Path::new("/d").join(n))
        .collect();
        let p = |n: &str| Path::new("/d").join(n);
        let dual = Work {
            main: vec![
                p("VID_20250101_120000_00_001.insv"),
                p("VID_20250101_120000_10_001.insv"),
            ],
            lrv: Some(p("LRV_20250101_120000_01_001.lrv")),
        };
        // any file of the recording opens the same work
        for n in [
            "VID_20250101_120000_10_001.insv",
            "LRV_20250101_120000_11_001.lrv",
            "VID_20250101_120000_00_001.insv",
        ] {
            assert_eq!(group(&p(n), &all), dual, "{n}");
        }
        assert_eq!(
            group(&p("LRV_20250101_120500_01_002.lrv"), &all),
            Work {
                main: vec![p("VID_20250101_120500_00_002.insv")],
                lrv: Some(p("LRV_20250101_120500_01_002.lrv")),
            }
        );
        assert_eq!(
            group(&p("v635.lrv"), &all),
            Work {
                main: vec![p("v635.insv")],
                lrv: Some(p("v635.lrv")),
            }
        );
        assert_eq!(
            group(&p("v606.insv"), &all),
            Work {
                main: vec![p("v606.insv")],
                lrv: None,
            }
        );
        // a proxy without its main file, a file missing from the listing
        assert_eq!(
            group(&p("x.lrv"), &[]),
            Work {
                main: vec![],
                lrv: Some(p("x.lrv")),
            }
        );
    }

    #[test]
    fn recent() {
        // a directory of its own: a save makes the file's directory owner-only
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-recent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let f = d.join("recent");
        let mut r = Recent::at(Some(f.clone()));
        assert!(r.list.is_empty());
        for i in 0..25 {
            r.add(Path::new(&format!("/v/{i}.insv")));
        }
        r.add(Path::new("/v/10.insv"));
        let r2 = Recent::at(Some(f.clone()));
        assert_eq!(r2.list.len(), RECENT);
        assert_eq!(r2.list[0], Path::new("/v/10.insv"));
        assert_eq!(r2.list[1], Path::new("/v/24.insv"));
        assert_eq!(r2.list.iter().filter(|p| p.ends_with("10.insv")).count(), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A recent entry whose file went away (its card unmounted) is greyed out within a look or two and does not open
    /// (it opened into an error); it comes back with the file. One that is there opens: here into the error of a file
    /// that is no video, which shows the click got through.
    #[test]
    fn recent_gone() {
        let g = Arc::new(Gpu::new().unwrap());
        let cx = stitch::Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let f = wgpu::TextureFormat::Bgra8Unorm;
        let rs = RenderState {
            adapter: g.adapter.clone(),
            available_adapters: vec![],
            instance: g.instance.clone(),
            device: g.device.clone(),
            queue: g.queue.clone(),
            target_format: f,
            renderer: Arc::new(egui::mutex::RwLock::new(eframe::egui_wgpu::Renderer::new(
                &g.device,
                f,
                Default::default(),
            ))),
        };
        let (still, _jobs) = mpsc::channel();
        let mut l = Library::new(&g, &cx, &ctx, still, Some(Test::default()));
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-gone-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let (card, away) = (d.join("X5"), d.join("away"));
        let n = "VID_20260118_124432_00_659.mp4";
        let p = card.join("DCIM/Camera01").join(n);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"no video").unwrap();
        l.recent.list = vec![p.clone()];
        let mut t = 0.0;
        // one frame with `ev`: the entry's middle and whether it is greyed out
        let mut frame = |l: &mut Library, ev: Vec<egui::Event>| {
            l.tick();
            t += 0.05;
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                time: Some(t),
                events: ev,
                ..Default::default()
            };
            let mut o = ctx.run_ui(raw, |ui| l.ui(ui, &rs));
            o.textures_delta.clear();
            let u = o.platform_output.accesskit_update.expect("AccessKit is on");
            let (_, e) = (u.nodes.iter())
                .find(|(_, x)| x.role() == egui::Role::Button && x.label() == Some(n))
                .expect("the entry");
            let b = e.bounds().unwrap();
            let m = egui::pos2(((b.x0 + b.x1) / 2.0) as f32, ((b.y0 + b.y1) / 2.0) as f32);
            (m, e.is_disabled())
        };
        type Frame<'a> = dyn FnMut(&mut Library, Vec<egui::Event>) -> (egui::Pos2, bool) + 'a;
        // frames until the entry is greyed out or not, the last look in
        let until = |f: &mut Frame, l: &mut Library, off: bool| {
            let t0 = Instant::now();
            loop {
                let (m, o) = f(l, vec![]);
                if o == off && l.look.is_none() {
                    return (m, t0.elapsed());
                }
                assert!(t0.elapsed() < 3 * LOOK, "still {o}");
                thread::sleep(Duration::from_millis(20));
            }
        };
        let click = |f: &mut Frame, l: &mut Library, m: egui::Pos2| {
            let b = |pressed| egui::Event::PointerButton {
                pos: m,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            };
            for ev in [
                vec![egui::Event::PointerMoved(m)],
                vec![b(true)],
                vec![b(false)],
            ] {
                f(l, ev);
            }
        };
        let (m, _) = until(&mut frame, &mut l, false);
        click(&mut frame, &mut l, m);
        assert!(l.err.take().is_some_and(|e| e.contains(n)), "opened");
        std::fs::rename(&card, &away).unwrap();
        let (m, s) = until(&mut frame, &mut l, true);
        println!(
            "greyed out {:.0} ms after the card went",
            s.as_secs_f64() * 1e3
        );
        click(&mut frame, &mut l, m);
        assert!(l.err.is_none() && l.movie.is_none() && l.player.is_none());
        assert!(l.videos().is_empty());
        std::fs::rename(&away, &card).unwrap();
        let (_, s) = until(&mut frame, &mut l, false);
        println!("back {:.0} ms after the card", s.as_secs_f64() * 1e3);
        assert_eq!(l.videos(), [p]);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
