//! The capture worker: a thread that runs the capture requests on the camera worker's session (`camera::Tap`): the
//! mode and its values, `/CONF` (`conf`) once per session, the status and the notifications, the live preview's
//! calibration, clock and start, setting changes, the shutter (a photo; a recording's or timelapse's explicit start
//! or stop), highlight marks. It checks what a host asks for: a setting only to a value `/CONF` allows the mode now,
//! no start at 5 % battery or less unless charging, a start or stop only when the camera's capture state is the other
//! one, no mode or setting change while the camera captures (each read from the camera first). Its reports carry the
//! session they belong to; a command sent as [`Cmd::Ask`] gets its outcome in the report after it ([`St::out`]).

use crate::{
    camera,
    conf::{self, Conf, Shot, Write},
    mono,
};
use link::{Camera, Note, code, pb};
use std::{
    sync::{
        Arc,
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const T: Duration = Duration::from_secs(10);

pub enum Cmd {
    /// the session (`camera::Tap::sid`) and its handle
    Cam(u32, Option<Camera>),
    Live(bool),
    Mode(String),
    /// a setting of the current mode (`/CONF` key and value): only a value its list allows now
    Set(String, String),
    /// the shutter of a photo mode
    Shoot,
    /// a recording or timelapse of the current mode started, stopped: not while the camera already does, or does
    /// not; no start at 5 % battery or less unless charging
    Start,
    Stop,
    /// a highlight mark in the running recording
    Mark,
    Note(Note),
    /// command `c` (Mode, Set, Shoot, Start, Stop, Mark) with its outcome in the report after it ([`St::out`], under
    /// number `n`); with `seen`, refused unless the camera is as the caller saw it when it checked `c`
    Ask {
        n: u64,
        c: Box<Cmd>,
        seen: Option<Seen>,
    },
    Quit,
}

/// What a caller checked a command against: the camera's mode, its lens block (360), whether its sub-mode is one the
/// protocol knows (`conf::known`).
#[derive(Clone, Debug, PartialEq)]
pub struct Seen {
    pub mode: String,
    pub both: bool,
    pub known: bool,
}

impl Seen {
    /// The camera as `st` shows it; none without a mode.
    pub fn of(st: &St) -> Option<Seen> {
        Some(Seen {
            mode: st.mode.clone()?,
            both: st.both,
            known: conf::known(&st.o),
        })
    }
}

/// What an asked command ([`Cmd::Ask`]) came to.
#[derive(Clone, Debug, PartialEq)]
pub enum Out {
    /// it went to the camera, which took it; what it answered (the photo's or clip's URI), if anything
    Sent(Option<String>),
    /// nothing went to the camera, or nothing more after a look at its state: why
    No(String),
    /// the camera or the link failed
    Failed(String),
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
    pub po: pb::Params,
    pub o: pb::Settings,
    pub cap: i32,
    /// capturing since (this computer's clock); the camera's capture time then, s
    pub since: Option<Instant>,
    pub temp: Option<pb::TempNote>,
    /// calibration string and the option it came from
    pub cal: Option<(String, &'static str)>,
    pub params: Option<pb::LiveNote>,
    /// CLOCK_MONOTONIC ns − camera ns, and the round trip it was measured with, ms
    pub off: Option<(i64, f64)>,
    pub live: bool,
    pub note: Option<(String, bool)>,
    /// notifications seen, the last one
    pub notes: u64,
    pub last: String,
    /// the mode whose params the last load read (`po` is that mode's); none while it failed
    pub vals: Option<String>,
    /// the newest asked command's number and outcome ([`Cmd::Ask`]), kept across sessions
    pub out: Option<(u64, Out)>,
    /// when the capture state was last read from the camera or reported by it
    pub cap_at: Option<Instant>,
}

impl St {
    /// The current value of setting `k` of mode `m` as `/CONF` spells it, if the camera reported it.
    pub fn cur(&self, m: &conf::Mode, k: &str) -> Option<String> {
        let photo = self.mode.as_deref().is_some_and(conf::photo);
        let sl = conf::slot(k)?;
        let x = conf::get(sl, &self.po, self.o.stitching, photo);
        m.keys
            .iter()
            .find(|q| q.name == k)?
            .all()
            .into_iter()
            .find(|v| conf::val(sl, v).is_some_and(|y| conf::same(x, y)))
            .map(Into::into)
    }

    /// The values `/CONF` allows setting `k` of the current mode now, given the current values of those it depends on
    /// (what the screen offers); none without `/CONF`, a mode or that setting.
    pub fn allowed(&self, k: &str) -> Vec<String> {
        let md = (|| {
            self.conf
                .as_ref()?
                .block(self.both)?
                .mode(self.mode.as_ref()?)
        })();
        let Some(md) = md else { return vec![] };
        let key = md.keys.iter().find(|x| x.name == k);
        key.map_or(vec![], |x| x.allowed(&|d| self.cur(md, d)).to_vec())
    }
}

/// Capture states with a running clip (`camera::rec`'s video states plus the timelapse ones): the shutter stops them.
pub fn running(s: i32) -> bool {
    use pb::CaptureState as C;
    !matches!(
        C::try_from(s).unwrap_or(C::Idle),
        C::Idle
            | C::TakingPhoto
            | C::TakingHdrPhoto
            | C::SelfTimer
            | C::ApplyingSettings
            | C::TakingBurst
            | C::TakingNightPhoto
            | C::TakingPano
            | C::TakingHdrPano
            | C::PreRecording
            | C::PreRecordingStopped
    )
}

/// The worker's thread: commands go in through `tx` (a session first, [`Cmd::Cam`]), its state comes out of `rx` after
/// each command and status poll, with the session it belongs to. Dropping it ends the thread (the preview stopped).
pub struct Worker {
    pub tx: Sender<Cmd>,
    pub rx: Receiver<(u32, St)>,
    th: Option<JoinHandle<()>>,
}

impl Worker {
    /// `wake`: a report arrived.
    pub fn spawn(wake: impl Fn() + Send + 'static) -> Worker {
        let (tx, crx) = mpsc::channel();
        let (utx, rx) = mpsc::channel();
        let w = W {
            cam: None,
            sid: 0,
            st: St::default(),
            up: utx,
            wake: Box::new(wake),
            bad: false,
            retry: None,
        };
        let th = thread::Builder::new()
            .name("x5-capture".into())
            .spawn(move || w.run(crx))
            .expect("capture thread");
        Worker {
            tx,
            rx,
            th: Some(th),
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

struct W {
    cam: Option<Camera>,
    /// the session `cam` belongs to: its reports carry it
    sid: u32,
    st: St,
    up: Sender<(u32, St)>,
    wake: Box<dyn Fn() + Send>,
    /// the session's `/CONF` did not parse: not fetched again in it
    bad: bool,
    /// a load failed: the poll's next try not before then
    retry: Option<Instant>,
}

impl W {
    fn send(&self) {
        let _ = self.up.send((self.sid, self.st.clone()));
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
        match c {
            Cmd::Cam(s, c) => {
                (self.bad, self.retry) = (false, None);
                let (on, out) = (c.is_some(), self.st.out.take());
                (self.sid, self.cam, self.st) = (
                    s,
                    c,
                    St {
                        on,
                        out,
                        ..Default::default()
                    },
                );
                self.load()
            }
            Cmd::Ask { n, c, seen } => {
                let o = self.ask(*c, seen);
                self.st.out = Some((n, o))
            }
            c => drop(self.act(c)),
        }
    }

    /// An asked command, refused when the camera is not as the caller saw it when it checked it (another mode, lens
    /// or sub-mode since).
    fn ask(&mut self, c: Cmd, seen: Option<Seen>) -> Out {
        if self.cam.is_none() {
            return Out::No("no camera session".into());
        }
        if let Some(w) = seen
            && Seen::of(&self.st).as_ref() != Some(&w)
        {
            let now = self.st.mode.as_deref().unwrap_or("a mode not known");
            return Out::No(match self.st.mode.as_ref() == Some(&w.mode) {
                true => format!("the camera's lens or sub-mode changed in {now}: check again"),
                false => format!("the camera is in {now} now, not {}", w.mode),
            });
        }
        match c {
            Cmd::Cam(..) | Cmd::Ask { .. } | Cmd::Note(_) | Cmd::Quit => {
                Out::No("not a command to ask for".into())
            }
            c => self.act(c),
        }
    }

    /// A command on the session, and what it came to; a failure noted (the screen shows it).
    fn act(&mut self, c: Cmd) -> Out {
        let Some(cam) = self.cam.clone() else {
            return Out::No("no camera session".into());
        };
        let sent = Out::Sent(None);
        let r: link::Result<Out> = match c {
            Cmd::Live(true) => self.live(&cam).map(|_| sent),
            Cmd::Live(false) => cam.live_stop().wait().map(|_| {
                self.st.live = false;
                sent
            }),
            Cmd::Mode(m) => self.mode(&cam, &m),
            Cmd::Set(k, v) => self.set(&cam, &k, &v),
            Cmd::Shoot => self.shoot(&cam),
            Cmd::Start => self.rec(&cam, true),
            Cmd::Stop => self.rec(&cam, false),
            Cmd::Mark => {
                let ms = self.st.since.map_or(0, |t| t.elapsed().as_millis() as u64);
                cam.highlight(ms).wait().map(|_| {
                    self.note(
                        format!("highlight marked at {:.1} s", ms as f64 / 1e3),
                        false,
                    );
                    sent
                })
            }
            Cmd::Note(n) => {
                self.on(n);
                Ok(sent)
            }
            Cmd::Cam(..) | Cmd::Ask { .. } | Cmd::Quit => Ok(sent),
        };
        r.unwrap_or_else(|e| {
            self.note(format!("{e}"), true);
            Out::Failed(e.to_string())
        })
    }

    /// The camera's capture state read before a command (a stale one would let a change in under a recording): the
    /// state, or the refusal's words when it cannot be read (nothing sent).
    fn status_first(&mut self, c: &Camera) -> Result<pb::CaptureStatus, Out> {
        c.capture_status().wait().map_err(|e| {
            let n = format!("the camera's capture state could not be read ({e}): nothing sent");
            self.note(n.clone(), true);
            Out::No(n)
        })
    }

    /// The refusal's words when the camera captures (its state read first, [`W::status_first`]).
    fn idle(&mut self, c: &Camera, what: &str) -> link::Result<Option<Out>> {
        let x = match self.status_first(c) {
            Ok(x) => x,
            Err(no) => return Ok(Some(no)),
        };
        self.cap(x.state, x.elapsed as u64);
        Ok(running(x.state).then(|| {
            let n = format!("the camera captures: no {what} until it stops");
            self.note(n.clone(), true);
            Out::No(n)
        }))
    }

    /// Switches to mode `m`, not while the camera captures (its stop would then take the new mode's way); the mode's
    /// values then.
    fn mode(&mut self, c: &Camera, m: &str) -> link::Result<Out> {
        let Some(s) = conf::sub(m) else {
            return Ok(Out::No(format!("{m}: no mode the camera has")));
        };
        if let Some(no) = self.idle(c, "mode change")? {
            return Ok(no);
        }
        c.set_mode(s).wait()?;
        self.load();
        Ok(Out::Sent(None))
    }

    /// The mode, its values and the status; `/CONF` once per session.
    fn load(&mut self) {
        let Some(c) = self.cam.clone() else { return };
        self.st.vals = None;
        let r = (|| {
            use pb::Setting::*;
            let o = c
                .settings(&[
                    PhotoMode, VideoMode, Lens, Battery, Storage, VideoLeft, PhotosLeft, Stitching,
                    Temp,
                ])
                .wait()?;
            let s = &mut self.st;
            s.both = !matches!(o.lens(), pb::Lens::Front | pb::Lens::Rear);
            s.mode = conf::current(&o);
            s.o = o;
            // a camera already recording: the elapsed time and highlight marks count from its capture time (s)
            let x = c.capture_status().wait()?;
            (s.cap, s.cap_at) = (x.state, Some(Instant::now()));
            s.since =
                running(x.state).then(|| Instant::now() - Duration::from_secs(x.elapsed as u64));
            if s.conf.is_none() && !self.bad {
                let (v, b) = c.conf().wait()?;
                match Conf::parse(v, &b) {
                    Ok(x) => s.conf = Some(Arc::new(x)),
                    Err(e) => {
                        self.bad = true;
                        self.note(e, true)
                    }
                }
            }
            self.values(&c)
        })();
        // a session closed meanwhile (its release, a lost link the session reports) is no failure of the camera's
        match r {
            Ok(()) | Err(link::Error::Closed) => {}
            Err(e) => self.note(format!("camera state: {e}"), true),
        }
    }

    /// The current mode's params: every key its settings use, and the remaining time.
    fn values(&mut self, c: &Camera) -> link::Result<()> {
        let s = &mut self.st;
        let (Some(m), Some(cf)) = (&s.mode, &s.conf) else {
            return Ok(());
        };
        let Some(fm) = conf::func(m) else {
            return Ok(());
        };
        let p = conf::photo(m);
        let mut ts = vec![pb::Param::TimeLeft];
        if let Some(md) = cf.block(s.both).and_then(|b| b.mode(m)) {
            for k in &md.keys {
                for &t in conf::slot(&k.name).map_or(&[][..], |x| conf::types(x, p)) {
                    if !ts.contains(&t) {
                        ts.push(t)
                    }
                }
            }
        }
        s.po = c.params(&ts, fm).wait()?.params.unwrap_or_default();
        s.vals = Some(m.clone());
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
        use pb::Setting::*;
        let ts = [Battery, Storage, VideoLeft, PhotosLeft, Temp];
        if let Ok(o) = c.settings(&ts).wait() {
            let x = &mut self.st.o;
            x.battery = o.battery.or(x.battery.take());
            x.storage = o.storage.or(x.storage.take());
            (x.video_left, x.photos_left, x.temp) = (o.video_left, o.photos_left, o.temp);
        }
        // the capture state too (a notification of it can be missed), and the mode's values while a load failed
        if let Ok(x) = c.capture_status().wait() {
            self.cap(x.state, x.elapsed as u64)
        }
        // a failed load tried again at most once a minute (a /CONF that did not parse not fetched again)
        let m = self.st.mode.as_deref();
        let due = self.retry.is_none_or(|t| Instant::now() >= t);
        if due
            && (self.st.conf.is_none() && !self.bad
                || m.is_some_and(|m| conf::func(m).is_some()) && self.st.vals != self.st.mode)
        {
            self.load();
            self.retry =
                (self.st.vals != self.st.mode).then(|| Instant::now() + Duration::from_secs(60));
        }
    }

    /// The live calibration (`calib_v6`, else `calib_v3`, else the v1 string `calib`), the camera clock against this
    /// computer's (the fastest of 5 `media_clock` round trips), then the preview.
    fn live(&mut self, c: &Camera) -> link::Result<()> {
        use pb::Setting::*;
        let o = c.settings(&[CalibV6, CalibV3, Calib]).wait()?;
        self.st.cal = [(o.calib_v6, "v6"), (o.calib_v3, "v3"), (o.calib, "v1")]
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

    /// The camera clock against this computer's: 5 `media_clock` round trips (ms since the camera booted, the stream's
    /// clock), the fastest of these and of the earlier ones counts. Busy USB (a download, thumbnails) slows them.
    fn sync(&mut self, c: &Camera) -> link::Result<()> {
        let mut best = self.st.off;
        for _ in 0..5 {
            let t0 = mono();
            let m = c.settings(&[pb::Setting::MediaClock]).wait()?.media_clock;
            let t1 = mono();
            let rtt = (t1 - t0) as f64 * 1e-6;
            if m > 0 && best.is_none_or(|b| rtt < b.1) {
                best = Some((((t0 + t1) / 2) as i64 - m * 1_000_000, rtt));
            }
        }
        self.st.off = best;
        Ok(())
    }

    /// Setting `k` to `v` if `/CONF` allows it now (what the screen offers) and the camera does not capture, then the
    /// mode's values again.
    fn set(&mut self, c: &Camera, k: &str, v: &str) -> link::Result<Out> {
        let s = &self.st;
        let Some(m) = s.mode.clone() else {
            return Ok(Out::No("the camera's mode is not known".into()));
        };
        let Some(sl) = conf::slot(k) else {
            return Ok(Out::No(format!("{k}: no setting this app knows")));
        };
        let Some(fm) = conf::func(&m) else {
            return Ok(Out::No(format!("{m}: no settings this app can write")));
        };
        // another mode's values would go out with it (the exposure options go whole): read once more first
        if s.vals.as_ref() != Some(&m) {
            self.load();
        }
        let s = &self.st;
        if s.vals.as_ref() != Some(&m) || s.mode.as_ref() != Some(&m) {
            let n = format!("{m}: its values were not read: no setting change");
            self.note(n.clone(), true);
            return Ok(Out::No(n));
        }
        let al = s.allowed(k);
        if !al.iter().any(|x| x == v) {
            let al = if al.is_empty() {
                "none".into()
            } else {
                al.join(", ")
            };
            let n = format!("{k} = {v}: not allowed in {m} now (allowed: {al})");
            self.note(n.clone(), true);
            return Ok(Out::No(n));
        }
        let Some(w) = conf::write(sl, v, &s.po, fm, conf::photo(&m)) else {
            let n = format!(
                "{k} = {v}: this app cannot write it (no wire value known, or the camera did not report the \
                 mode's exposure)"
            );
            self.note(n.clone(), true);
            return Ok(Out::No(n));
        };
        if let Some(no) = self.idle(c, "setting change")? {
            return Ok(no);
        }
        match w {
            Write::Photo(q) => c
                .call::<pb::WriteParamsReply>(code::WRITE_PARAMS, &q, T)
                .wait()
                .map(drop)?,
            Write::Opts(q) => c
                .call::<pb::WriteSettingsReply>(code::WRITE_SETTINGS, &q, T)
                .wait()
                .map(drop)?,
        }
        self.load();
        Ok(Out::Sent(None))
    }

    /// The shutter of a photo mode; a recording or timelapse needs [`Cmd::Start`] or [`Cmd::Stop`].
    fn shoot(&mut self, c: &Camera) -> link::Result<Out> {
        let Some(m) = self.st.mode.clone() else {
            return Ok(Out::No("the camera's mode is not known".into()));
        };
        Ok(match conf::shot(&m) {
            // the photo's raw type is the mode's: its values read (once more if a load failed)
            Shot::Photo(_)
                if self.st.vals.as_ref() != Some(&m) && {
                    self.load();
                    self.st.vals.as_ref() != Some(&m) || self.st.mode.as_ref() != Some(&m)
                } =>
            {
                let n = format!("{m}: its values were not read: no photo taken");
                self.note(n.clone(), true);
                Out::No(n)
            }
            Shot::Photo(p) => {
                let q = pb::TakePhoto {
                    kind: p as i32,
                    format: self.st.po.format,
                };
                let r = c.take_photo(q).wait()?;
                let u = r.photo.map(|i| i.uri).unwrap_or_default();
                self.note(format!("photo taken {u}"), false);
                Out::Sent(Some(u).filter(|u| !u.is_empty()))
            }
            Shot::Rec(_) | Shot::Lapse(_) => {
                let n = format!("{m}: captures with start and stop, nothing taken");
                self.note(n.clone(), true);
                Out::No(n)
            }
            Shot::None => Out::No(format!("{m}: a live mode, no shutter")),
        })
    }

    /// Starts (`on`) or stops the current mode's recording or timelapse, after a look at the camera's capture state: a
    /// start while it captures, or a stop while it does not, sends nothing (a toggle would go the wrong way). No start
    /// at 5 % battery or less unless charging.
    fn rec(&mut self, c: &Camera, on: bool) -> link::Result<Out> {
        let Some(m) = self.st.mode.clone() else {
            return Ok(Out::No("the camera's mode is not known".into()));
        };
        let shot = conf::shot(&m);
        if !matches!(shot, Shot::Rec(_) | Shot::Lapse(_)) {
            let n = format!("{m}: no recording or timelapse mode");
            self.note(n.clone(), true);
            return Ok(Out::No(n));
        }
        let x = match self.status_first(c) {
            Ok(x) => x,
            Err(no) => return Ok(no),
        };
        self.cap(x.state, x.elapsed as u64);
        let no = match (running(x.state), on) {
            (true, true) => Some("the camera captures already: nothing started"),
            (false, false) => Some("the camera does not capture: nothing to stop"),
            (false, true) if self.st.o.battery.as_ref().is_some_and(camera::empty) => {
                Some("battery at 5 % or less: charge the camera to record")
            }
            _ => None,
        };
        if let Some(n) = no {
            self.note(n, true);
            return Ok(Out::No(n.into()));
        }
        // a start the camera answered with an error did not begin (a timeout may have)
        let started = |r: link::Result<()>, me: &mut W| match r {
            Ok(()) => Ok(Out::Sent(None)),
            Err(e @ link::Error::Status(..)) => {
                let n = format!("the camera refused the start: {e}");
                me.note(n.clone(), true);
                Ok(Out::No(n))
            }
            Err(e) => Err(e),
        };
        let v = match (shot, on) {
            (Shot::Rec(x), true) => return started(c.start_capture(x).wait(), self),
            (Shot::Lapse(x), true) => return started(c.start_timelapse(x).wait(), self),
            (Shot::Rec(x), false) => c.stop_capture(x).wait()?,
            (Shot::Lapse(x), false) => c.stop_timelapse(x).wait()?,
            _ => return Ok(Out::No(format!("{m}: no recording or timelapse mode"))),
        };
        // an interval's or starlapse's stop may name no file
        match v.uri.is_empty() {
            true => self.note("timelapse stopped", false),
            false => self.note(format!("recorded {}", v.uri), false),
        }
        Ok(Out::Sent(Some(v.uri).filter(|u| !u.is_empty())))
    }

    /// A capture state the camera reported: the capture's start (its capture time ago, s) when one begins.
    fn cap(&mut self, state: i32, secs: u64) {
        let s = &mut self.st;
        let (was, now) = (running(s.cap), running(state));
        (s.cap, s.cap_at) = (state, Some(Instant::now()));
        if now && !was {
            s.since = Some(Instant::now() - Duration::from_secs(secs));
        } else if !now {
            s.since = None;
        }
    }

    fn on(&mut self, n: Note) {
        let s = &mut self.st;
        s.notes += 1;
        s.last = format!("{n:?}").chars().take(80).collect();
        match n {
            Note::Battery(b) | Note::BatteryLow(b) => s.o.battery = Some(b),
            Note::Capture(x) => self.cap(x.state, x.elapsed as u64),
            Note::Temp(t) => s.temp = Some(t),
            Note::Live(p) => s.params = Some(p),
            Note::PhotoStage(p) => {
                let x = format!("photo: {:?}", p.stage());
                self.note(x, false);
            }
            Note::Stopped(x) => {
                let e = x.cause();
                self.note(format!("capture stopped: {e:?}"), true);
                self.load();
            }
            Note::Shutdown(x) => self.note(format!("the camera shuts down: {:?}", x.cause()), true),
            Note::CardFull => self.note("the SD card is full", true),
            Note::Card(_) | Note::Mode(_) | Note::Params(_) | Note::Photo(_) => self.load(),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
