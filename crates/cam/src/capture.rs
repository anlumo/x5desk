//! The capture worker: a thread that runs the capture requests on the camera worker's session (`camera::Tap`): the
//! mode and its values, `/CONF` (`conf`) once per session, the status and the notifications, the live preview's
//! calibration, clock and start, setting changes, the shutter (a photo; a recording's or timelapse's explicit start
//! or stop), highlight marks. It checks what a host asks for: a setting only to a value `/CONF` allows the mode now,
//! no start at 5 % battery or less unless charging, a start or stop only when the camera's capture state is the other
//! one. Its reports carry the session they belong to.

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

impl St {
    /// The current value of setting `k` of mode `m` as `/CONF` spells it, if the camera reported it.
    pub fn cur(&self, m: &conf::Mode, k: &str) -> Option<String> {
        let photo = self.mode.as_deref().is_some_and(conf::photo);
        let sl = conf::slot(k)?;
        let x = conf::get(sl, &self.po, self.o.enable_internal_splicing, photo);
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
        if let Cmd::Cam(s, c) = c {
            let on = c.is_some();
            (self.sid, self.cam, self.st) = (
                s,
                c,
                St {
                    on,
                    ..Default::default()
                },
            );
            return self.load();
        }
        let Some(cam) = self.cam.clone() else { return };
        let r: link::Result<()> = match c {
            Cmd::Live(true) => self.live(&cam),
            Cmd::Live(false) => cam.live_stop().wait().map(|_| self.st.live = false),
            Cmd::Mode(m) => match conf::sub(&m) {
                Some(s) => cam.set_mode(s).wait().map(|_| self.load()),
                None => Ok(()),
            },
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
                    )
                })
            }
            Cmd::Note(n) => {
                self.on(n);
                Ok(())
            }
            Cmd::Cam(..) | Cmd::Quit => Ok(()),
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

    /// Setting `k` to `v` if `/CONF` allows it now (what the screen offers), then the mode's values again.
    fn set(&mut self, c: &Camera, k: &str, v: &str) -> link::Result<()> {
        let s = &self.st;
        let (Some(m), Some(sl)) = (&s.mode, conf::slot(k)) else {
            return Ok(());
        };
        let Some(fm) = conf::func(m) else {
            return Ok(());
        };
        let al = s.allowed(k);
        if !al.iter().any(|x| x == v) {
            let al = if al.is_empty() {
                "none".into()
            } else {
                al.join(", ")
            };
            self.note(
                format!("{k} = {v}: not allowed in {m} now (allowed: {al})"),
                true,
            );
            return Ok(());
        }
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

    /// The shutter of a photo mode; a recording or timelapse needs [`Cmd::Start`] or [`Cmd::Stop`].
    fn shoot(&mut self, c: &Camera) -> link::Result<()> {
        let Some(m) = self.st.mode.clone() else {
            return Ok(());
        };
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
            Shot::Rec(_) | Shot::Lapse(_) => self.note(
                format!("{m}: captures with start and stop, nothing taken"),
                true,
            ),
            Shot::None => {}
        }
        Ok(())
    }

    /// Starts (`on`) or stops the current mode's recording or timelapse, after a look at the camera's capture state: a
    /// start while it captures, or a stop while it does not, sends nothing (a toggle would go the wrong way). No start
    /// at 5 % battery or less unless charging.
    fn rec(&mut self, c: &Camera, on: bool) -> link::Result<()> {
        let Some(m) = self.st.mode.clone() else {
            return Ok(());
        };
        let shot = conf::shot(&m);
        if !matches!(shot, Shot::Rec(_) | Shot::Lapse(_)) {
            self.note(format!("{m}: no recording or timelapse mode"), true);
            return Ok(());
        }
        let x = c.capture_status().wait()?;
        self.cap(x.state, x.capture_time as u64);
        let no = match (running(x.state), on) {
            (true, true) => Some("the camera captures already: nothing started"),
            (false, false) => Some("the camera does not capture: nothing to stop"),
            (false, true) if self.st.o.battery_info.as_ref().is_some_and(camera::empty) => {
                Some("battery at 5 % or less: charge the camera to record")
            }
            _ => None,
        };
        if let Some(n) = no {
            self.note(n, true);
            return Ok(());
        }
        match (shot, on) {
            (Shot::Rec(x), true) => c.start_capture(x).wait()?,
            (Shot::Rec(x), false) => {
                let v = c.stop_capture(x).wait()?;
                self.note(format!("recorded {}", v.uri), false);
            }
            (Shot::Lapse(x), true) => c.start_timelapse(x).wait()?,
            (Shot::Lapse(x), false) => drop(c.stop_timelapse(x).wait()?),
            _ => {}
        }
        Ok(())
    }

    /// A capture state the camera reported: the capture's start (its capture time ago, s) when one begins.
    fn cap(&mut self, state: i32, secs: u64) {
        let s = &mut self.st;
        let (was, now) = (running(s.cap), running(state));
        s.cap = state;
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
            Note::Battery(b) | Note::BatteryLow(b) => s.o.battery_info = Some(b),
            Note::CaptureStatus(x) => self.cap(x.state, x.capture_time as u64),
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

#[cfg(test)]
mod tests;
