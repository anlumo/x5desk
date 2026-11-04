//! Capture screen (milestone 3): the camera's live preview ([`Feed`]) or its Webcam Mode ([`Webcam`]) in the 360
//! viewer; the capture controls generated from the camera's `/CONF` (`conf`) for its current mode, with the values
//! it reports; the shutter (photo, recording, timelapse, the self-timer the app counts down itself), highlight marks;
//! the status from the notifications. The capture worker (`cam::capture`) runs the requests on the camera worker's
//! session (`camera::Tap`). On an X5 (fw v1.13.21, `tests::real_writes` / `real_ilog`) these ran, each read back:
//! switches to the modes in [`SWITCHED`], the settings in [`CHECKED`], the photo and the normal video shutter,
//! highlight marks. The screen marks the rest ⚠.

use crate::{
    camera::{self, Taps},
    conf::{self, Shot, Slot},
    feed::Feed,
    player::Look,
    webcam::{self, Webcam},
};
pub use cam::capture::{Cmd, St, Worker, running};
use eframe::{egui, egui_wgpu::RenderState};
use link::{Note, pb};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};
use stitch::Ctx;
use video::Gpu;

/// The `/CONF` settings (mode, key) whose writes ran on an X5 and read back.
const CHECKED: [(&str, &str); 8] = [
    ("PHOTO_SINGLE", "white_balance"),
    ("VIDEO_NORMAL", "record_resolution"),
    ("VIDEO_NORMAL", "p3_switch"),
    ("VIDEO_NORMAL", "exposure_program"),
    ("VIDEO_NORMAL", "exposure_iso"),
    ("VIDEO_NORMAL", "exposure_shutter_speed"),
    ("VIDEO_NORMAL", "white_balance"),
    ("VIDEO_NORMAL", "i_log_switch"),
];

/// The modes the app's switch to ran on an X5 and read back.
const SWITCHED: [&str; 2] = ["PHOTO_SINGLE", "VIDEO_NORMAL"];

const UNV: &str = "⚠ Not run on an X5 yet: the app's message, unit-tested";

/// Mode `m` as the camera's app names it, marked ⚠ when the switch to it did not run on an X5.
fn mlabel(m: &str) -> String {
    match SWITCHED.contains(&m) {
        true => conf::label(m).into(),
        false => format!("{} ⚠", conf::label(m)),
    }
}

/// What the test hook asks of this screen (`X5DESK_TEST`).
#[derive(Clone, Debug, Default)]
pub struct Hook {
    /// start the preview at once, the Webcam Mode (capture file) instead
    pub on: bool,
    pub webcam: Option<webcam::Input>,
    /// print the statistics after this many seconds of pictures, then end the run
    pub stats: Option<f64>,
    pub look: Option<Look>,
    pub yaw: Option<f32>,
    pub level: Option<bool>,
    pub pace: Option<bool>,
}

/// Which picture the screen shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    Preview,
    Webcam,
}

pub struct Capture {
    g: Arc<Gpu>,
    cx: Ctx,
    ctx: egui::Context,
    tap: Taps,
    /// the capture worker
    w: Worker,
    pub st: St,
    sid: u32,
    pub feed: Option<Arc<Mutex<Feed>>>,
    pub web: Option<Arc<Mutex<Webcam>>>,
    pub src: Src,
    web_in: webcam::Input,
    /// the user wants the preview; the self-timer's shutter time
    want: bool,
    timer: Option<Instant>,
    stats: bool,
    hook: Option<Hook>,
    done: bool,
}

impl Capture {
    pub fn new(
        g: &Arc<Gpu>,
        cx: &Ctx,
        ctx: &egui::Context,
        tap: Taps,
        hook: Option<Hook>,
    ) -> Capture {
        let c = ctx.clone();
        let w = Worker::spawn(move || c.request_repaint());
        let t2 = w.tx.clone();
        tap.lock().unwrap().notes = Some(Box::new(move |n: &Note| {
            drop(t2.send(Cmd::Note(n.clone())))
        }));
        let h = hook.clone().unwrap_or_default();
        Capture {
            g: g.clone(),
            cx: cx.clone(),
            ctx: ctx.clone(),
            tap,
            w,
            st: St::default(),
            sid: 0,
            feed: None,
            web: None,
            src: if h.webcam.is_some() {
                Src::Webcam
            } else {
                Src::Preview
            },
            web_in: h.webcam.clone().unwrap_or(webcam::Input::Uvc(2880, 1440)),
            want: h.on,
            timer: None,
            stats: h.stats.is_some(),
            hook,
            done: false,
        }
    }

    /// The live source of the Live screen.
    pub fn source(&self, s: Src) -> Option<crate::live::Src> {
        match s {
            Src::Preview => self.feed.clone().map(|f| f as _),
            Src::Webcam => self.web.clone().map(|w| w as _),
        }
    }

    fn view<R>(&self, f: impl FnOnce(&mut Look, &mut stitch::View, &mut bool) -> R) -> Option<R> {
        let mut lv = true;
        match self.src {
            Src::Preview => self.feed.as_ref().map(|x| {
                let mut x = x.lock().unwrap();
                let (mut l, mut v) = (x.look, x.view);
                lv = x.level;
                let r = f(&mut l, &mut v, &mut lv);
                (x.look, x.view, x.level) = (l, v, lv);
                r
            }),
            Src::Webcam => self.web.as_ref().map(|x| {
                let mut x = x.lock().unwrap();
                let (mut l, mut v) = (x.look, x.view);
                let r = f(&mut l, &mut v, &mut lv);
                (x.look, x.view) = (l, v);
                r
            }),
        }
    }

    /// Every UI frame: the session, the worker's reports, the preview's start and stop, the self-timer.
    pub fn tick(&mut self) {
        let (sid, cam, lost) = {
            let t = self.tap.lock().unwrap();
            (t.sid, t.cam.clone(), t.lost)
        };
        if sid != self.sid {
            // a new session: a preview of the old one starts again on it
            (self.sid, self.feed, self.st) = (sid, None, St::default());
            self.tap.lock().unwrap().frames = None;
            let _ = self.w.tx.send(Cmd::Cam(sid, cam.clone()));
        }
        // a report of an earlier session (sent before the worker took the new one) is stale
        while let Ok((n, s)) = self.w.rx.try_recv() {
            if n != self.sid {
                continue;
            }
            if let Some(f) = &self.feed {
                let mut f = f.lock().unwrap();
                if s.cal != self.st.cal
                    && let Some((c, n)) = &s.cal
                    && let Err(e) = f.cal(c, n)
                {
                    eprintln!("x5desk: live calibration from {n}: {e}");
                }
                if let Some(p) = &s.params {
                    f.params(p);
                }
                f.off = s.off.map(|o| o.0);
            }
            self.st = s;
        }
        let on = cam.is_some() && self.st.on;
        let want = self.want && self.src == Src::Preview;
        match (&self.feed, want && on) {
            (None, true) => {
                let (tx, rx) = mpsc::sync_channel(64);
                let c = self.ctx.clone();
                let mut f = Feed::new(&self.g, &self.cx, rx, "Camera live preview", move || {
                    c.request_repaint()
                });
                if let Some(h) = &self.hook {
                    f.look = h.look.unwrap_or(f.look);
                    f.view = stitch::View {
                        yaw: h.yaw.unwrap_or(0.0),
                        ..f.look.view()
                    };
                    f.level = h.level.unwrap_or(true);
                    f.pace(h.pace.unwrap_or(true));
                }
                {
                    let mut t = self.tap.lock().unwrap();
                    (t.frames, t.lost) = (Some(tx), 0);
                }
                self.feed = Some(Arc::new(Mutex::new(f)));
                // a new feed takes the calibration of a new report
                self.st.cal = None;
                let _ = self.w.tx.send(Cmd::Live(true));
            }
            (Some(_), false) => {
                self.tap.lock().unwrap().frames = None;
                self.feed = None;
                if on {
                    let _ = self.w.tx.send(Cmd::Live(false));
                }
            }
            (Some(f), true) => f.lock().unwrap().lost = lost,
            _ => {}
        }
        match (&self.web, self.want && self.src == Src::Webcam) {
            (None, true) => {
                let c = self.ctx.clone();
                let mut w = Webcam::new(&self.g, &self.cx, self.web_in.clone(), move || {
                    c.request_repaint()
                });
                if let Some(h) = &self.hook {
                    w.look = h.look.unwrap_or(w.look);
                    w.view = stitch::View {
                        yaw: h.yaw.unwrap_or(0.0),
                        ..w.look.view()
                    };
                }
                self.web = Some(Arc::new(Mutex::new(w)));
            }
            (Some(_), false) => self.web = None,
            _ => {}
        }
        if self.timer.is_some_and(|t| Instant::now() >= t) {
            self.timer = None;
            // a self-timer runs in a photo mode that did not capture: its photo, or its interval or starlapse started
            if let Some(m) = self.st.mode.as_deref()
                && let Some(c) = press(conf::shot(m), false)
            {
                let _ = self.w.tx.send(c);
            }
        }
        self.measure();
        if self.timer.is_some() || self.st.since.is_some() {
            self.ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    /// The test hook's statistics are printed (and the run may end); ready for a screenshot: a second of pictures
    /// and, with a camera session, the worker's first report on it (it sends one only after reading the mode, its
    /// values and `/CONF`, which may fail or be absent: a replayed camera without `conf=`).
    pub fn idle(&self) -> bool {
        let shown = match self.src {
            Src::Preview => self
                .feed
                .as_ref()
                .map_or(0, |f| f.lock().unwrap().counts().dec),
            Src::Webcam => self.web.as_ref().map_or(0, |w| w.lock().unwrap().taken()),
        };
        let stats = self.hook.as_ref().is_none_or(|h| h.stats.is_none()) || self.done;
        let cam = self.tap.lock().unwrap().cam.is_some();
        (!self.want || shown > 30) && (!cam || self.st.on) && stats
    }

    /// Prints the statistics once `stats=` seconds of pictures went by (test hook).
    fn measure(&mut self) {
        let Some(secs) = self
            .hook
            .as_ref()
            .and_then(|h| h.stats)
            .filter(|_| !self.done)
        else {
            return;
        };
        let (t0, r) = match self.src {
            Src::Preview => match &self.feed {
                Some(f) => {
                    let f = f.lock().unwrap();
                    (f.stats.t0, f.report())
                }
                None => return,
            },
            Src::Webcam => match &self.web {
                Some(w) => {
                    let w = w.lock().unwrap();
                    (w.t0, w.report())
                }
                None => return,
            },
        };
        if t0.is_some_and(|t| t.elapsed().as_secs_f64() >= secs) {
            let c = self.st.off.map_or("no camera clock".into(), |o| {
                format!("camera clock round trip {:.1} ms", o.1)
            });
            println!("x5desk capture stats ({:?}, {c}): {r}", self.src);
            self.done = true;
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        egui::Panel::top("cap bar").show(ui, |ui| self.bar(ui));
        egui::Panel::right("cap side")
            .default_size(330.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side(ui));
            });
        let pv = self.src == Src::Preview;
        egui::Panel::bottom("cap view").show(ui, |ui| {
            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                self.view(|l, v, lv| {
                    for (x, name) in Look::ALL {
                        if ui.selectable_label(*l == x, name).clicked() && *l != x {
                            *l = x;
                            *v = stitch::View {
                                yaw: v.yaw,
                                ..x.view()
                            };
                        }
                    }
                    if pv {
                        ui.separator();
                        ui.checkbox(lv, "Level")
                            .on_hover_text("Level the horizon with the camera's IMU");
                    }
                });
                if let Some(f) = self.feed.as_ref().filter(|_| self.src == Src::Preview) {
                    let f = f.lock().unwrap();
                    let mut p = f.paced();
                    if ui
                        .checkbox(&mut p, "Smooth")
                        .on_hover_text(
                            "Show the pictures at the camera's pace (the X5 sends them in pairs): about a frame of \
                             latency. Off: always the newest picture. The Live screen's outputs always go at the \
                             camera's pace, and while they run this view shows what they send",
                        )
                        .changed()
                    {
                        f.pace(p);
                    }
                }
                ui.separator();
                ui.toggle_value(&mut self.stats, "Stats");
            });
            ui.add_space(2.0);
        });
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewer(ui, rs));
    }

    fn bar(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.strong("Capture");
            ui.separator();
            ui.selectable_value(&mut self.src, Src::Preview, "Live preview")
                .on_hover_text("The camera in app mode (USB): its preview, stitched here");
            ui.selectable_value(&mut self.src, Src::Webcam, "Webcam Mode")
                .on_hover_text(
                    "The camera in Webcam Mode (UVC): its own 360° equirect or flat picture",
                );
            if self.src == Src::Webcam {
                let mut w = self.web_in.clone();
                if matches!(w, webcam::Input::Uvc(..)) {
                    egui::ComboBox::from_id_salt("uvc size")
                        .selected_text(match w {
                            webcam::Input::Uvc(2880, _) => "2880 × 1440 equirect",
                            _ => "1920 × 1080 flat",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut w,
                                webcam::Input::Uvc(2880, 1440),
                                "2880 × 1440 equirect",
                            );
                            ui.selectable_value(
                                &mut w,
                                webcam::Input::Uvc(1920, 1080),
                                "1920 × 1080 flat",
                            );
                        });
                }
                if w != self.web_in {
                    (self.web_in, self.web) = (w, None);
                }
            }
            let on = self.tap.lock().unwrap().cam.is_some();
            let can = self.src == Src::Webcam || on;
            if ui
                .add_enabled(
                    can || self.want,
                    egui::Button::new(if self.want { "Stop" } else { "Start" }),
                )
                .clicked()
            {
                self.want = !self.want;
            }
            ui.separator();
            let s = match self.src {
                Src::Preview if !on => {
                    "Connect the camera (Camera screen) for its live preview.".into()
                }
                Src::Preview => match (&self.st.cal, &self.st.params) {
                    (Some((_, n)), Some(p)) => {
                        format!(
                            "{} × {} at {} fps, calibration {n} from the camera",
                            p.res_width, p.res_height, p.res_fps
                        )
                    }
                    (Some((_, n)), None) => format!("calibration {n} from the camera"),
                    _ if self.want => "starting…".into(),
                    _ => String::new(),
                },
                Src::Webcam => match &self.web_in {
                    webcam::Input::File(p) => format!("replaying {}", p.display()),
                    _ => "The camera in Webcam Mode: on the X5, USB mode → Webcam Mode".into(),
                },
            };
            ui.label(s);
        });
        ui.add_space(2.0);
    }

    fn viewer(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        let (rect, resp) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, egui::Color32::BLACK);
        let ppp = ui.ctx().pixels_per_point();
        let tex = self.view(|l, v, _| {
            let r = l.fit(rect);
            (r, l.steer(ui, &resp, r, v))
        });
        let (mut lines, stats) = (vec![], self.stats);
        if let Some((r, dirty)) = tex {
            let size = [r.width(), r.height()].map(|x| ((x * ppp).round() as u32).max(1));
            let (id, rep, err) = match self.src {
                Src::Preview => {
                    let mut f = self.feed.as_ref().unwrap().lock().unwrap();
                    let id = f.render(rs, size, dirty).filter(|_| f.ready());
                    if !f.ready() {
                        lines.push(match &self.st.cal {
                            None => "Waiting for the camera's calibration…".into(),
                            Some(_) => "Waiting for the camera's picture…".into(),
                        });
                    }
                    (id, stats.then(|| f.report()), f.err())
                }
                Src::Webcam => {
                    let mut w = self.web.as_ref().unwrap().lock().unwrap();
                    let id = w.render(rs, size, dirty).filter(|_| w.pano().is_some());
                    if w.pano().is_none() {
                        lines.push("Waiting for the Webcam Mode's picture…".into());
                    }
                    (id, stats.then(|| w.report()), w.err())
                }
            };
            if let Some(id) = id {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                p.image(id, r, uv, egui::Color32::WHITE);
            }
            lines.extend(err);
            lines.extend(rep);
            self.ctx.request_repaint_after(Duration::from_millis(250));
        } else {
            lines.push(match self.src {
                Src::Preview => "Live preview off.".into(),
                Src::Webcam => "Webcam Mode off.".into(),
            });
        }
        let mut y = rect.top() + 6.0;
        for l in lines {
            let g = p.layout(
                l,
                egui::FontId::monospace(12.0),
                egui::Color32::WHITE,
                rect.width() - 12.0,
            );
            let at = egui::pos2(rect.left() + 6.0, y);
            p.rect_filled(
                egui::Rect::from_min_size(at, g.size()).expand(3.0),
                3.0,
                egui::Color32::from_black_alpha(160),
            );
            y += g.size().y + 8.0;
            p.galley(at, g, egui::Color32::WHITE);
        }
    }

    /// The current value of setting `k` of mode `m` as `/CONF` spells it, if the camera reported it.
    fn cur(&self, m: &conf::Mode, k: &str) -> Option<String> {
        self.st.cur(m, k)
    }

    fn side(&mut self, ui: &mut egui::Ui) {
        let s = self.st.clone();
        if !s.on {
            ui.label("Connect the camera on the Camera screen for its capture controls.");
            return;
        }
        ui.weak("⚠: not run on an X5 yet (the app's messages, unit-tested)");
        ui.add_space(4.0);
        self.status(ui, &s);
        ui.add_space(6.0);
        // the worker reports a session only once it tried to read /CONF
        let Some(cf) = &s.conf else {
            ui.label("The camera's /CONF could not be read: no capture controls.");
            return;
        };
        let Some(b) = cf.block(s.both) else {
            ui.label("/CONF has no block for this lens setting.");
            return;
        };
        ui.horizontal(|ui| {
            ui.heading("Mode");
            ui.weak(format!("/CONF version {}, {}", cf.version, b.name));
        });
        let m = s.mode.clone().unwrap_or_default();
        egui::ComboBox::from_id_salt("cap mode")
            .selected_text(conf::label(&m))
            .width(200.0)
            .show_ui(ui, |ui| {
                for (h, l) in [("Photo", &b.photo), ("Video", &b.video), ("Live", &b.live)] {
                    ui.label(egui::RichText::new(h).weak());
                    for x in l {
                        let mut r = ui.selectable_label(*x == m, mlabel(x));
                        if !SWITCHED.contains(&&**x) {
                            r = r.on_hover_text(UNV);
                        }
                        if r.clicked() && *x != m {
                            let _ = self.w.tx.send(Cmd::Mode(x.clone()));
                        }
                    }
                }
            });
        let Some(md) = b.mode(&m) else {
            ui.label(format!("/CONF has no mode {m}"));
            return;
        };
        self.shutter(ui, &s, &m);
        ui.add_space(6.0);
        ui.heading("Settings");
        let cur = |k: &str| self.cur(md, k);
        egui::Grid::new("cap set")
            .num_columns(2)
            .spacing([10.0, 4.0])
            .show(ui, |ui| {
                for (k, sl, label) in conf::KEYS {
                    let Some(key) = md.keys.iter().find(|x| x.name == k) else {
                        continue;
                    };
                    let al = key.allowed(&cur);
                    if al.is_empty() {
                        continue;
                    }
                    // a single value allowed is the one the camera uses now (e.g. ISO Auto with HDR on), whatever
                    // manual value it keeps for later
                    let c = match al {
                        [v] => Some(v.clone()),
                        _ => cur(k),
                    };
                    let (wired, ok) = (conf::wired(sl), CHECKED.contains(&(&m, k)));
                    ui.label(match wired && !ok {
                        true => format!("{label} ⚠"),
                        false => label.into(),
                    });
                    let tip = match (wired, ok) {
                        (false, _) => "Read only here: its encoding is not known for sure",
                        (_, false) => UNV,
                        _ => "",
                    };
                    let mut pick = None;
                    ui.add_enabled_ui(wired, |ui| {
                        let r = if al.iter().all(|v| v == "true" || v == "false") {
                            let mut b = c.as_deref() == Some("true");
                            // a single value: shown, nothing to choose
                            let r = ui.add_enabled(al.len() == 2, egui::Checkbox::new(&mut b, ""));
                            if r.changed() {
                                pick = Some(b.to_string());
                            }
                            r
                        } else {
                            egui::ComboBox::from_id_salt(k)
                                .selected_text(c.as_deref().map_or("?".into(), |v| show(sl, v)))
                                .show_ui(ui, |ui| {
                                    for v in al {
                                        if ui
                                            .selectable_label(c.as_deref() == Some(v), show(sl, v))
                                            .clicked()
                                        {
                                            pick = Some(v.clone());
                                        }
                                    }
                                })
                                .response
                        };
                        if !tip.is_empty() {
                            r.on_hover_text(tip);
                        }
                    });
                    if let Some(v) = pick.filter(|v| Some(v) != c.as_ref()) {
                        let _ = self.w.tx.send(Cmd::Set(k.into(), v));
                    }
                    ui.end_row();
                }
            });
        let other: Vec<&conf::Key> = md
            .keys
            .iter()
            .filter(|k| conf::slot(&k.name).is_none() && !k.all().is_empty())
            .collect();
        if !other.is_empty() {
            ui.collapsing(
                format!("Other /CONF settings of this mode ({})", other.len()),
                |ui| {
                    for k in other {
                        ui.label(format!("{}: {}", k.name, k.allowed(&cur).join(", ")));
                    }
                },
            );
        }
    }

    fn status(&self, ui: &mut egui::Ui, s: &St) {
        egui::Grid::new("cap st")
            .num_columns(2)
            .spacing([10.0, 3.0])
            .show(ui, |ui| {
                let mut row = |k: &str, v: String, warn: bool| {
                    ui.label(k);
                    match warn {
                        true => ui.colored_label(ui.visuals().warn_fg_color, v),
                        false => ui.label(v),
                    };
                    ui.end_row();
                };
                let st = pb::CameraCaptureState::try_from(s.cap)
                    .unwrap_or(pb::CameraCaptureState::NotCapture);
                let t = s.since.map_or(String::new(), |t| {
                    let x = t.elapsed().as_secs();
                    format!(" {}:{:02}", x / 60, x % 60)
                });
                row(
                    "State",
                    match st {
                        pb::CameraCaptureState::NotCapture => "idle".into(),
                        x => format!("{}{t}", x.as_str_name().to_lowercase().replace('_', " ")),
                    },
                    false,
                );
                let photo = s.mode.as_deref().is_some_and(conf::photo);
                let x = s.o.remaining_capture_time.max(s.po.remaining_time);
                match (photo, s.o.remaining_pictures, x) {
                    (true, n, _) if n > 0 => row("Remaining", format!("{n} photos"), false),
                    (false, _, x) if x > 0 => row(
                        "Remaining",
                        format!("{}:{:02} h of video", x / 3600, x / 60 % 60),
                        false,
                    ),
                    _ => {}
                }
                if let Some(x) = &s.o.storage_state {
                    row(
                        "Storage",
                        format!(
                            "{:.1} GB free of {:.1} GB",
                            x.free_space as f64 / 1e9,
                            x.total_space as f64 / 1e9
                        ),
                        x.free_space < 2_000_000_000,
                    );
                }
                if let Some(b) = &s.o.battery_info {
                    row("Battery", camera::battery(b), camera::low(b));
                }
                let temp = match (&s.temp, s.o.temp_value) {
                    (Some(t), _) => Some((
                        format!(
                            "{} ({} °C)",
                            t.temp_state()
                                .as_str_name()
                                .trim_start_matches("TEMP_")
                                .to_lowercase(),
                            t.temp_value
                        ),
                        t.temp_state >= pb::TemperatureState::TempHigh as i32,
                    )),
                    (None, v) if v != 0 => Some((format!("{v} °C"), false)),
                    _ => None,
                };
                if let Some((t, w)) = temp {
                    row("Temperature", t, w);
                }
                ui.label("Notifications");
                ui.add(egui::Label::new(format!("{} · {}", s.notes, s.last)).truncate());
                ui.end_row();
            });
        if let Some((n, e)) = &s.note {
            match e {
                true => ui.colored_label(ui.visuals().error_fg_color, n),
                false => ui.label(n),
            };
        }
    }

    fn shutter(&mut self, ui: &mut egui::Ui, s: &St, m: &str) {
        let shot = conf::shot(m);
        let run = running(s.cap);
        // what ran on an X5: a photo, a normal recording
        let ok = matches!(
            shot,
            Shot::Photo(pb::take_picture::Mode::Normal)
                | Shot::Rec(pb::CaptureMode::Normal)
                | Shot::None
        );
        let low = s.o.battery_info.as_ref().is_some_and(camera::empty);
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let (txt, can) = match shot {
                Shot::Photo(_) => ("Take photo", true),
                Shot::Rec(_) | Shot::Lapse(_) if run => ("Stop", true),
                // the app's rule: no recording at 5 % unless charging
                Shot::Rec(_) | Shot::Lapse(_) => ("Start recording", !low),
                Shot::None => ("Streamed by this app (Live screen)", false),
            };
            let txt = match (self.timer, ok) {
                (Some(t), _) => format!(
                    "Cancel ({:.0} s)",
                    (t - Instant::now()).as_secs_f64().ceil()
                ),
                (_, true) => txt.into(),
                (_, false) => format!("{txt} ⚠"),
            };
            let mut b = ui.add_enabled(
                can,
                egui::Button::new(egui::RichText::new(txt).size(16.0))
                    .min_size(egui::vec2(150.0, 32.0)),
            );
            if !ok {
                b = b.on_hover_text(UNV);
            }
            if b.clicked() {
                let timer = if conf::photo(m) {
                    s.po.photography_self_timer
                } else {
                    0
                };
                self.timer = match (self.timer, timer, run) {
                    (Some(_), ..) => None,
                    // the X5 leaves the self-timer to the app (app_features.md §5)
                    (None, t, false) if t > 0 => {
                        Some(Instant::now() + Duration::from_secs(t as u64))
                    }
                    _ => {
                        if let Some(c) = press(shot, run) {
                            let _ = self.w.tx.send(c);
                        }
                        None
                    }
                };
            }
            if run && conf::marks(m) && ui.button("Highlight").clicked() {
                let _ = self.w.tx.send(Cmd::Mark);
            }
        });
        if low {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Battery at 5 % or less: charge the camera to record.",
            );
        }
    }
}

/// What the shutter sends in a mode whose shutter does `shot`, the camera capturing (`run`) or not as the screen shows
/// it: a photo, or the start or stop its button says (the worker sends nothing when the camera is not as shown).
fn press(shot: Shot, run: bool) -> Option<Cmd> {
    match shot {
        Shot::Photo(_) => Some(Cmd::Shoot),
        Shot::Rec(_) | Shot::Lapse(_) if run => Some(Cmd::Stop),
        Shot::Rec(_) | Shot::Lapse(_) => Some(Cmd::Start),
        Shot::None => None,
    }
}

/// A `/CONF` value as the UI shows it.
fn show(s: Slot, v: &str) -> String {
    let p: Vec<&str> = v.split('_').collect();
    match (s, v) {
        (Slot::Iso | Slot::Shutter | Slot::Wb | Slot::IsoTop, "0") => "Auto".into(),
        (Slot::Timer | Slot::Accel, "0") => "Off".into(),
        (Slot::Wb, v) => format!("{v} K"),
        (Slot::Timer, v) => format!("{v} s"),
        (Slot::Accel, v) => format!("{v}×"),
        (Slot::Shutter, v) => format!("{v} s"),
        (Slot::Ev, v) if !v.starts_with('-') && v != "0" => format!("+{v}"),
        (Slot::Res, _) if p.len() >= 3 => format!(
            "{} × {}, {} fps{}",
            p[0],
            p[1],
            p[2],
            if p.len() > 3 { " +" } else { "" }
        ),
        (Slot::PhotoRes, _) if p.len() == 2 => format!("{} × {}", p[0], p[1]),
        (Slot::Dur, v) => v.parse::<u32>().map_or(v.into(), |x| match x {
            0 => "Unlimited".into(),
            x if x % 3600 == 0 => format!("{} h", x / 3600),
            x => format!("{} min", x / 60),
        }),
        (Slot::Burst, v) => v
            .split_once('/')
            .map_or(v.into(), |(n, t)| format!("{n} in {t} s")),
        _ => v.replace('_', " ").to_lowercase(),
    }
}

impl Drop for Capture {
    /// The notes stop, the preview goes, then the worker (its field's drop: a quit and a join).
    fn drop(&mut self) {
        self.tap.lock().unwrap().notes = None;
        self.feed = None;
    }
}

/// `X5DESK_TEST`'s `webcam=<file>` / `webcam`.
pub fn webcam_input(v: &str) -> webcam::Input {
    match v {
        "" | "2880" => webcam::Input::Uvc(2880, 1440),
        "1920" => webcam::Input::Uvc(1920, 1080),
        f => webcam::Input::File(PathBuf::from(f)),
    }
}

#[cfg(test)]
mod tests;
