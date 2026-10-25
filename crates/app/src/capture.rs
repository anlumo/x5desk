//! Capture screen (milestone 3): the camera's live preview ([`Feed`]) or its Webcam Mode ([`Webcam`]) in the 360
//! viewer; the capture controls generated from the camera's `/CONF` (`conf`) for its current mode, with the values
//! it reports; the shutter (photo, recording, timelapse, the self-timer the app counts down itself), highlight marks;
//! the status from the notifications. A worker thread runs the requests on the camera worker's session
//! (`camera::Tap`). On an X5 (fw v1.13.21, `tests::real_writes` / `real_ilog`) these ran, each read back: mode
//! switches, the settings in [`CHECKED`], the photo and the normal video shutter, highlight marks. The screen marks
//! the rest ⚠.

use crate::{
    camera::{self, Taps},
    conf::{self, Conf, Shot, Slot, Write},
    feed::{Feed, mono},
    player::Look,
    webcam::{self, Webcam},
};
use eframe::{egui, egui_wgpu::RenderState};
use link::{Camera, Note, code, pb};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant},
};
use stitch::Ctx;
use video::Gpu;

const T: Duration = Duration::from_secs(10);

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

const UNV: &str = "⚠ Not run on an X5 yet: the app's message, unit-tested";

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

enum Cmd {
    Cam(Option<Camera>),
    Live(bool),
    Mode(String),
    Set(String, String),
    Shoot,
    Mark,
    Note(Note),
    Quit,
}

/// What the worker knows: the camera's `/CONF`, lens block, mode and its values, status, and the live preview's
/// calibration, parameters and clock.
#[derive(Clone, Default)]
pub struct St {
    pub on: bool,
    pub conf: Option<Arc<Conf>>,
    /// 360 (`Sphere06C`), else a single lens
    pub both: bool,
    pub mode: Option<String>,
    pub po: pb::PhotographyOptions,
    pub o: pb::Options,
    pub cap: i32,
    /// capturing since (this computer's clock); the camera's capture time then, s
    pub since: Option<Instant>,
    pub temp: Option<pb::TempState>,
    /// calibration string and the option it came from
    pub cal: Option<(String, &'static str)>,
    pub params: Option<pb::LiveStreamParamsUpdate>,
    /// CLOCK_MONOTONIC ns − camera ns, and the round trip it was measured with, ms
    pub off: Option<(i64, f64)>,
    pub live: bool,
    pub note: Option<(String, bool)>,
    /// notifications seen, the last one
    pub notes: u64,
    pub last: String,
}

/// Capture states with a running clip (`camera::rec`'s video states plus the timelapse ones): the shutter stops them.
fn running(s: i32) -> bool {
    use pb::CameraCaptureState as C;
    !matches!(
        C::try_from(s).unwrap_or(C::NotCapture),
        C::NotCapture
            | C::SingleShooting
            | C::HdrShooting
            | C::SelfTimerShooting
            | C::SettingsNewValue
            | C::BurstShooting
            | C::AebNightShooting
            | C::SinglePowerPanoShooting
            | C::HdrPowerPanoShooting
            | C::PreRecord
            | C::PreRecordStop
    )
}

struct W {
    cam: Option<Camera>,
    st: St,
    up: Sender<St>,
    wake: Box<dyn Fn() + Send>,
}

impl W {
    fn send(&self) {
        let _ = self.up.send(self.st.clone());
        (self.wake)()
    }

    fn note(&mut self, s: impl Into<String>, err: bool) {
        let s = s.into();
        if err {
            eprintln!("x5desk: capture: {s}");
        }
        self.st.note = Some((s, err));
    }

    fn run(mut self, rx: Receiver<Cmd>) {
        // the status every T on its own clock: a camera sending notifications all along still gets polled
        let mut next = Instant::now() + T;
        loop {
            match rx.recv_timeout(next.saturating_duration_since(Instant::now())) {
                Ok(Cmd::Quit) | Err(RecvTimeoutError::Disconnected) => break,
                Ok(c) => self.cmd(c),
                Err(RecvTimeoutError::Timeout) => {
                    self.status();
                    next = Instant::now() + T;
                }
            }
            self.send();
        }
        if let (Some(c), true) = (&self.cam, self.st.live) {
            let _ = c.live_stop().wait();
        }
    }

    fn cmd(&mut self, c: Cmd) {
        let Some(cam) = self.cam.clone() else {
            if let Cmd::Cam(Some(c)) = c {
                self.cam = Some(c);
                self.st = St {
                    on: true,
                    ..Default::default()
                };
                self.load();
            }
            return;
        };
        let r: link::Result<()> = match c {
            Cmd::Cam(c) => {
                (self.cam, self.st) = (None, St::default());
                if c.is_some() {
                    return self.cmd(Cmd::Cam(c));
                }
                Ok(())
            }
            Cmd::Live(true) => self.live(&cam),
            Cmd::Live(false) => cam.live_stop().wait().map(|_| self.st.live = false),
            Cmd::Mode(m) => match conf::sub(&m) {
                Some(s) => cam.set_mode(s).wait().map(|_| self.load()),
                None => Ok(()),
            },
            Cmd::Set(k, v) => self.set(&cam, &k, &v),
            Cmd::Shoot => self.shoot(&cam),
            Cmd::Mark => {
                let ms = self.st.since.map_or(0, |t| t.elapsed().as_millis() as u64);
                cam.highlight(ms).wait().map(|_| {
                    self.note(
                        format!("highlight marked at {:.1} s", ms as f64 / 1e3),
                        false,
                    )
                })
            }
            Cmd::Note(n) => {
                self.on(n);
                Ok(())
            }
            Cmd::Quit => Ok(()),
        };
        if let Err(e) = r {
            self.note(format!("{e}"), true);
        }
    }

    /// The mode, its values and the status; `/CONF` once per session.
    fn load(&mut self) {
        let Some(c) = self.cam.clone() else { return };
        let r = (|| {
            use pb::OptionType::*;
            let o = c
                .options(&[
                    PhotoSubMode,
                    VideoSubMode,
                    InstapanoFocusSensor,
                    BatteryInfo,
                    StorageState,
                    RemainingCaptureTime,
                    RemainingPictures,
                    InternalSplicing,
                    TempValue,
                ])
                .wait()?;
            let s = &mut self.st;
            s.both = !matches!(
                o.focus_sensor(),
                pb::SensorDevice::Front | pb::SensorDevice::Rear
            );
            s.mode = conf::current(&o);
            s.o = o;
            // a camera already recording: the elapsed time and highlight marks count from its capture time (s)
            let x = c.capture_status().wait()?;
            s.cap = x.state;
            s.since = running(x.state)
                .then(|| Instant::now() - Duration::from_secs(x.capture_time as u64));
            if s.conf.is_none() {
                let (v, b) = c.conf().wait()?;
                match Conf::parse(v, &b) {
                    Ok(x) => s.conf = Some(Arc::new(x)),
                    Err(e) => self.note(e, true),
                }
            }
            self.values(&c)
        })();
        if let Err(e) = r {
            self.note(format!("camera state: {e}"), true);
        }
    }

    /// The current mode's photography options: every type its settings use, and the remaining time.
    fn values(&mut self, c: &Camera) -> link::Result<()> {
        let s = &mut self.st;
        let (Some(m), Some(cf)) = (&s.mode, &s.conf) else {
            return Ok(());
        };
        let Some(fm) = conf::func(m) else {
            return Ok(());
        };
        let p = conf::photo(m);
        let mut ts = vec![pb::PhotographyOptionType::RemainingTime];
        if let Some(md) = cf.block(s.both).and_then(|b| b.mode(m)) {
            for k in &md.keys {
                for &t in conf::slot(&k.name).map_or(&[][..], |x| conf::types(x, p)) {
                    if !ts.contains(&t) {
                        ts.push(t)
                    }
                }
            }
        }
        s.po = c.photo_options(&ts, fm).wait()?.value.unwrap_or_default();
        Ok(())
    }

    /// Battery, storage, remaining time and temperature (every 10 s without anything else to do); the clock again
    /// while the preview runs.
    fn status(&mut self) {
        let Some(c) = self.cam.clone() else { return };
        match self.st.live.then(|| self.sync(&c)) {
            Some(Err(link::Error::Closed)) | Some(Ok(())) | None => {}
            Some(Err(e)) => self.note(format!("camera clock: {e}"), true),
        }
        use pb::OptionType::*;
        let ts = [
            BatteryInfo,
            StorageState,
            RemainingCaptureTime,
            RemainingPictures,
            TempValue,
        ];
        if let Ok(o) = c.options(&ts).wait() {
            let x = &mut self.st.o;
            x.battery_info = o.battery_info.or(x.battery_info.take());
            x.storage_state = o.storage_state.or(x.storage_state.take());
            (x.remaining_capture_time, x.remaining_pictures, x.temp_value) =
                (o.remaining_capture_time, o.remaining_pictures, o.temp_value);
        }
    }

    /// The live calibration (MEDIA_OFFSET_V6, else _V3, else the v1 string: the app's order), the camera clock against
    /// this computer's (the fastest of 5 MEDIA_TIME round trips), then the preview.
    fn live(&mut self, c: &Camera) -> link::Result<()> {
        use pb::OptionType::*;
        let o = c
            .options(&[MediaOffsetV6, MediaOffsetV3, MediaOffset])
            .wait()?;
        self.st.cal = [
            (o.media_offset_v6, "MEDIA_OFFSET_V6"),
            (o.media_offset_v3, "MEDIA_OFFSET_V3"),
            (o.media_offset, "MEDIA_OFFSET"),
        ]
        .into_iter()
        .find(|x| !x.0.is_empty());
        if self.st.cal.is_none() {
            self.note("the camera reported no lens calibration", true);
        }
        self.sync(c)?;
        // the camera announces the stream's size and crop in 8234 right after
        c.live_start().wait()?;
        self.st.live = true;
        self.send();
        Ok(())
    }

    /// The camera clock against this computer's: 5 MEDIA_TIME round trips (ms since the camera booted, the stream's
    /// clock), the fastest of these and of the earlier ones counts. Busy USB (a download, thumbnails) slows them.
    fn sync(&mut self, c: &Camera) -> link::Result<()> {
        let mut best = self.st.off;
        for _ in 0..5 {
            let t0 = mono();
            let m = c.options(&[pb::OptionType::MediaTime]).wait()?.media_time;
            let t1 = mono();
            let rtt = (t1 - t0) as f64 * 1e-6;
            if m > 0 && best.is_none_or(|b| rtt < b.1) {
                best = Some((((t0 + t1) / 2) as i64 - m * 1_000_000, rtt));
            }
        }
        self.st.off = best;
        Ok(())
    }

    fn set(&mut self, c: &Camera, k: &str, v: &str) -> link::Result<()> {
        let s = &self.st;
        let (Some(m), Some(sl)) = (&s.mode, conf::slot(k)) else {
            return Ok(());
        };
        let Some(fm) = conf::func(m) else {
            return Ok(());
        };
        match conf::write(sl, v, &s.po, fm, conf::photo(m)) {
            Some(Write::Photo(q)) => c
                .call::<pb::SetPhotographyOptionsResp>(code::SET_PHOTO_OPTIONS, &q, T)
                .wait()
                .map(drop)?,
            Some(Write::Opts(q)) => c
                .call::<pb::SetOptionsResp>(code::SET_OPTIONS, &q, T)
                .wait()
                .map(drop)?,
            None => return Ok(()),
        }
        self.load();
        Ok(())
    }

    fn shoot(&mut self, c: &Camera) -> link::Result<()> {
        let Some(m) = self.st.mode.clone() else {
            return Ok(());
        };
        let run = running(self.st.cap);
        match conf::shot(&m) {
            Shot::Photo(p) => {
                let q = pb::TakePicture {
                    mode: p as i32,
                    raw_capture_type: self.st.po.raw_capture_type,
                    ..Default::default()
                };
                let r = c.take_photo(q).wait()?;
                let u = r.image.map(|i| i.uri).unwrap_or_default();
                self.note(format!("photo taken {u}"), false);
            }
            Shot::Rec(x) if run => {
                let v = c.stop_capture(x).wait()?;
                self.note(format!("recorded {}", v.uri), false);
            }
            Shot::Rec(x) => c.start_capture(x).wait()?,
            Shot::Lapse(x) if run => drop(c.stop_timelapse(x).wait()?),
            Shot::Lapse(x) => c.start_timelapse(x).wait()?,
            Shot::None => {}
        }
        Ok(())
    }

    fn on(&mut self, n: Note) {
        let s = &mut self.st;
        s.notes += 1;
        s.last = format!("{n:?}").chars().take(80).collect();
        match n {
            Note::Battery(b) | Note::BatteryLow(b) => s.o.battery_info = Some(b),
            Note::CaptureStatus(x) => {
                let (was, now) = (running(s.cap), running(x.state));
                s.cap = x.state;
                if now && !was {
                    s.since = Some(Instant::now() - Duration::from_secs(x.capture_time as u64));
                } else if !now {
                    s.since = None;
                }
            }
            Note::Temp(t) => s.temp = Some(t),
            Note::LiveParams(p) => s.params = Some(p),
            Note::PictureState(p) => {
                let x = format!("photo: {:?}", p.state());
                self.note(x, false);
            }
            Note::CaptureStopped(x) => {
                let e = x.err_code();
                self.note(format!("capture stopped: {e:?}"), true);
                self.load();
            }
            Note::Shutdown(x) => {
                self.note(format!("the camera shuts down: {:?}", x.error_code()), true)
            }
            Note::StorageFull => self.note("the SD card is full", true),
            Note::Storage(_) | Note::SubMode(_) | Note::PhotoOptions(_) | Note::Photo(_) => {
                self.load()
            }
            _ => {}
        }
    }
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
    tx: Sender<Cmd>,
    rx: Receiver<St>,
    th: Option<thread::JoinHandle<()>>,
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
        let (tx, crx) = mpsc::channel();
        let (utx, rx) = mpsc::channel();
        let c = ctx.clone();
        let w = W {
            cam: None,
            st: St::default(),
            up: utx,
            wake: Box::new(move || c.request_repaint()),
        };
        let th = thread::Builder::new()
            .name("x5-capture".into())
            .spawn(move || w.run(crx))
            .expect("capture thread");
        let t2 = tx.clone();
        tap.lock().unwrap().notes = Some(Box::new(move |n: &Note| {
            drop(t2.send(Cmd::Note(n.clone())))
        }));
        let h = hook.clone().unwrap_or_default();
        Capture {
            g: g.clone(),
            cx: cx.clone(),
            ctx: ctx.clone(),
            tap,
            tx,
            rx,
            th: Some(th),
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
            let _ = self.tx.send(Cmd::Cam(cam.clone()));
        }
        while let Ok(s) = self.rx.try_recv() {
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
                let _ = self.tx.send(Cmd::Live(true));
            }
            (Some(_), false) => {
                self.tap.lock().unwrap().frames = None;
                self.feed = None;
                if on {
                    let _ = self.tx.send(Cmd::Live(false));
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
            let _ = self.tx.send(Cmd::Shoot);
        }
        self.measure();
        if self.timer.is_some() || self.st.since.is_some() {
            self.ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    /// The test hook's statistics are printed (and the run may end); ready for a screenshot: a second of pictures
    /// and, connected, `/CONF` read.
    pub fn idle(&self) -> bool {
        let shown = match self.src {
            Src::Preview => self
                .feed
                .as_ref()
                .map_or(0, |f| f.lock().unwrap().counts().dec),
            Src::Webcam => self.web.as_ref().map_or(0, |w| w.lock().unwrap().taken()),
        };
        let stats = self.hook.as_ref().is_none_or(|h| h.stats.is_none()) || self.done;
        (!self.want || shown > 30) && (!self.st.on || self.st.conf.is_some()) && stats
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
        let s = &self.st;
        let photo = s.mode.as_deref().is_some_and(conf::photo);
        let sl = conf::slot(k)?;
        let x = conf::get(sl, &s.po, s.o.enable_internal_splicing, photo);
        m.keys
            .iter()
            .find(|q| q.name == k)?
            .all()
            .into_iter()
            .find(|v| conf::val(sl, v).is_some_and(|y| conf::same(x, y)))
            .map(Into::into)
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
        let Some(cf) = &s.conf else {
            ui.label("Reading the camera's /CONF…");
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
                        if ui.selectable_label(*x == m, conf::label(x)).clicked() && *x != m {
                            let _ = self.tx.send(Cmd::Mode(x.clone()));
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
                        let _ = self.tx.send(Cmd::Set(k.into(), v));
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
        let low = s.o.battery_info.as_ref().is_some_and(|b| {
            let (l, c) = camera::level(b);
            l <= 5 && !c
        });
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
                        let _ = self.tx.send(Cmd::Shoot);
                        None
                    }
                };
            }
            if run && conf::marks(m) && ui.button("Highlight").clicked() {
                let _ = self.tx.send(Cmd::Mark);
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
    fn drop(&mut self) {
        self.tap.lock().unwrap().notes = None;
        self.feed = None;
        let _ = self.tx.send(Cmd::Quit);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
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
