//! The camera worker: the X5 over USB or Wi-Fi. A worker thread owns the link: it follows USB hotplug, connects
//! (automatically if wanted; the sync runs on the link's thread and a disconnect or quit cancels it), reads identity,
//! battery, storage, Wi-Fi and clock, sets the camera clock (on every connect asked to, as the app does on its own,
//! its answers not waited for; an agent's connect leaves it), lists the album, fetches thumbnails (disk cache first,
//! the ones on screen first), runs the download queue and the camera commands (no delete while the camera records),
//! and follows the camera's notifications. A host (the app's Camera screen) sends it commands and shows what it
//! reports. The capture worker shares the session through [`Tap`]: the handle, and the live-preview packets and
//! notifications, which the events thread hands it directly.
//!
//! The album can show a mounted card instead (`disk`: the X5 in USB mass-storage mode, or its card in a reader): the
//! worker watches the mount table, lists the card's DCIM with the same grouping, reads thumbnails from the trailers,
//! and imports through the download queue. A camera that joined the user's network (station mode) connects over
//! Wi-Fi.

use crate::{
    album::{self, Cache, Item},
    capture, disk, dl,
};
use link::{Camera, Error, Event, Note, Plug, pb};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    net::IpAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub type Connect = Box<dyn FnMut() -> link::Opening + Send>;

/// What the Capture screen takes from the session: the handle and its session number, and where the events thread
/// hands the live-preview packets (a full channel drops them, counted) and copies of the notifications.
#[derive(Default)]
pub struct Tap {
    pub cam: Option<Camera>,
    pub sid: u32,
    pub frames: Option<mpsc::SyncSender<link::Frame>>,
    pub lost: u64,
    pub notes: Option<NoteFn>,
}

pub type NoteFn = Box<dyn Fn(&Note) + Send>;

pub type Taps = Arc<Mutex<Tap>>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Conn {
    #[default]
    Off,
    Connecting,
    On,
}

/// What the worker knows about the camera.
#[derive(Clone, Debug, Default)]
pub struct Info {
    /// none before the first look at the bus
    pub plug: Option<Plug>,
    pub conn: Conn,
    pub id: Option<link::Ident>,
    pub battery: Option<pb::BatteryInfo>,
    pub storage: Option<pb::StorageState>,
    /// SSID and password of the camera's access point
    pub wifi: Option<(String, String)>,
    /// camera clock minus the computer's, s
    pub skew: Option<i64>,
    /// capturing: the files are off limits (PROTOCOL.md §7.12)
    pub busy: bool,
    /// connection attempts that finished; when the one running began
    pub tries: u32,
    pub since: Option<Instant>,
    /// mounted camera disks and cards
    pub disk: Vec<disk::Vol>,
    /// the album shows this card (its mount point) instead of the camera
    pub src: Option<PathBuf>,
}

/// One album load: listing (files, file info, favourites) and thumbnails (from the camera, from the disk cache).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub list_ms: f64,
    pub cam: usize,
    pub bytes: usize,
    pub cached: usize,
    pub secs: f64,
}

pub enum Cmd {
    Plug(Plug),
    /// Connect over USB; `clock`: set the camera clock once connected (§5.1: the app's own connects, by hand or
    /// automatic; an agent's never). Retries of this connection, and reconnects after a lost link (the 2 s timer, or
    /// the camera back on the bus), do the same.
    Connect {
        clock: bool,
    },
    Disconnect,
    Refresh,
    /// Item keys whose thumbnails go first.
    Want(Vec<String>),
    /// Not while the camera records (§7.12).
    Delete(Vec<String>),
    Fav(Vec<(String, bool)>),
    Format,
    Time,
    Ev(u32, Event),
    /// Watch for mounted cards.
    Watch(disk::Watch),
    /// The album's source: a mounted card, or the camera.
    Source(Option<PathBuf>),
    /// Connect over Wi-Fi to the camera at `ip` with client id `id` (station mode); `clock` as for
    /// [`Cmd::Connect`].
    Wifi {
        ip: IpAddr,
        id: String,
        clock: bool,
    },
    Quit,
}

pub enum Up {
    Info(Info),
    Album(Vec<Item>),
    Thumb(String, Option<(u32, u32, Vec<u8>)>),
    /// A download finished.
    Done(PathBuf),
    Note(String, bool),
    Stats(Stats),
}

pub struct Worker {
    pub tx: Sender<Cmd>,
    pub rx: Receiver<Up>,
    th: Option<JoinHandle<()>>,
}

impl Worker {
    /// `usb`: follow the bus (otherwise only [`Cmd::Connect`] connects); `auto`: connect a camera that appears on it
    /// (setting the clock, as the app does; one back after a lost link or while a retry waits as its session did), and
    /// try again after a busy camera or a lost link; `wake`: something arrived.
    pub fn spawn(
        connect: Connect,
        q: dl::Shared,
        cache: Cache,
        auto: Arc<AtomicBool>,
        usb: bool,
        tap: Taps,
        wake: impl Fn() + Send + 'static,
    ) -> Worker {
        let (tx, crx) = mpsc::channel();
        let (utx, rx) = mpsc::channel();
        let mut scan = None;
        if usb {
            let t = tx.clone();
            if let Err(e) = link::watch(move |p| t.send(Cmd::Plug(p)).is_ok()) {
                let _ = utx.send(Up::Note(
                    format!("no USB hotplug ({e}): looking every 2 s"),
                    true,
                ));
                scan = Some(Instant::now());
            }
        }
        let w = W {
            tx: tx.clone(),
            up: utx,
            wake: Box::new(wake),
            connect,
            auto,
            cam: None,
            opening: None,
            tap,
            sid: 0,
            info: Info::default(),
            sent: HashSet::new(),
            cache,
            keys: HashMap::new(),
            todo: VecDeque::new(),
            fly: vec![],
            stats: Stats::default(),
            t0: None,
            q,
            cur: None,
            relist: None,
            retry: None,
            scan,
            watch: None,
            look: None,
            clk: None,
            sync: true,
            lost: false,
            told: false,
        };
        let th = thread::Builder::new()
            .name("x5-camera".into())
            .spawn(move || w.run(crx))
            .expect("camera thread");
        Worker {
            tx,
            rx,
            th: Some(th),
        }
    }

    /// No thread: the commands go to `tx`, the reports come from `rx` (a test standing in for the worker).
    #[cfg(any(test, feature = "sim"))]
    pub fn fake(tx: Sender<Cmd>, rx: Receiver<Up>) -> Worker {
        Worker { tx, rx, th: None }
    }
}

impl Drop for Worker {
    /// Closes the session cleanly before the app goes.
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
        if let Some(h) = self.th.take() {
            let _ = h.join();
        }
    }
}

/// Thumbnails in flight.
const FLY: usize = 3;

/// How often the bus is looked at when there are no hotplug events.
const SCAN: Duration = Duration::from_secs(2);

/// Capture states that put the files off limits: the video ones (the app's `CaptureType.isVideoType`, §7.12; wire 1
/// `NORMAL_CAPTURE` is a video). Photos, the self-timer and pre-recording leave the files alone.
fn rec(state: i32) -> bool {
    matches!(state, 1 | 2 | 7 | 9 | 11..=13 | 17 | 18 | 20..=24 | 29..=31 | 33)
}

struct W {
    tx: Sender<Cmd>,
    up: Sender<Up>,
    wake: Box<dyn Fn() + Send>,
    connect: Connect,
    auto: Arc<AtomicBool>,
    cam: Option<Camera>,
    /// a connection syncing (dropping it cancels the sync)
    opening: Option<link::Opening>,
    tap: Taps,
    /// session number: events of an earlier session are dropped
    sid: u32,
    info: Info,
    /// thumbnails sent this session
    sent: HashSet<String>,
    cache: Cache,
    /// the album's items: key → thumbnail URI and cache file
    keys: HashMap<String, (String, Option<PathBuf>)>,
    /// thumbnails to get by item key
    todo: VecDeque<String>,
    fly: Vec<(String, Option<PathBuf>, link::Pending<Vec<u8>>)>,
    stats: Stats,
    t0: Option<Instant>,
    q: dl::Shared,
    cur: dl::Cur,
    relist: Option<Instant>,
    retry: Option<Instant>,
    /// next look at the bus, without hotplug events
    scan: Option<Instant>,
    /// where cards get mounted; the next look at the mount table
    watch: Option<disk::Watch>,
    look: Option<Instant>,
    clk: Option<Clk>,
    /// the connection opening, or the session up, sets the camera clock (its retries and reconnects do too)
    sync: bool,
    /// the last session's link was lost, not closed: the next automatic connect is its reconnect
    lost: bool,
    /// said once that the camera clock is left alone (no IANA zone known here)
    told: bool,
}

/// A clock sync in flight (PROTOCOL.md §5.3): the answers to come, the first error, whether one did not come or came
/// garbled (a timeout, the link gone, an undecodable reply), the note for success, and a connect's notes, which wait
/// for it.
struct Clk {
    p: Vec<link::Pending<pb::SetOptionsResp>>,
    err: Option<Error>,
    lost: bool,
    ok: String,
    conn: Vec<(String, bool)>,
}

/// Unix time, the app's favourite timestamp (unix + standard offset: Java's `ZONE_OFFSET`, no DST), offset from UTC
/// in s, IANA zone (TZ, else the /etc/localtime link), local time as text.
#[cfg(unix)]
pub fn clock() -> (u64, i64, i32, String, String) {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let at = |t: u64| {
        // SAFETY: localtime_r only writes the tm it is given.
        unsafe {
            let mut tm: libc::tm = std::mem::zeroed();
            libc::localtime_r(&(t as libc::time_t), &mut tm);
            tm
        }
    };
    let tm = at(t);
    let off = tm.tm_gmtoff as i32;
    // DST adds to the standard offset and lasts under 9 months: one of four samples a quarter apart is outside it
    let std = (0..4)
        .map(|q| at(t + q * 91 * 86400).tm_gmtoff)
        .min()
        .unwrap_or(0);
    let tz = std::env::var("TZ")
        .ok()
        .map(|s| s.trim_start_matches(':').to_owned())
        .filter(|s| s.contains('/'))
        .or_else(|| {
            let p = std::fs::read_link("/etc/localtime").ok()?;
            Some(p.to_str()?.split_once("zoneinfo/")?.1.to_owned())
        })
        .unwrap_or_else(|| "UTC".into());
    let s = format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    );
    (t, t as i64 + std, off, tz, s)
}

/// Unix time, the favourite timestamp (unix + standard offset), offset from UTC in s, both from the system's zone
/// rules (Windows: `GetTimeZoneInformation`; none known: UTC). No IANA zone or local time is made up, so the worker
/// never sets the camera clock here.
#[cfg(not(unix))]
pub fn clock() -> (u64, i64, i32, String, String) {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    #[cfg(windows)]
    let (std, off) = zone();
    #[cfg(not(windows))]
    let (std, off) = (0, 0);
    (t, t as i64 + std as i64, off, String::new(), String::new())
}

/// The standard and the current offset from UTC in s: UTC = local + bias (minutes), the standard or daylight bias
/// added to it; (0, 0) when the call fails.
#[cfg(windows)]
fn zone() -> (i32, i32) {
    /// TIME_ZONE_INFORMATION; the dates are SYSTEMTIMEs, 8 u16 each.
    #[repr(C)]
    struct Tzi {
        bias: i32,
        std_name: [u16; 32],
        std_date: [u16; 8],
        std_bias: i32,
        dst_name: [u16; 32],
        dst_date: [u16; 8],
        dst_bias: i32,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetTimeZoneInformation(tzi: *mut Tzi) -> u32;
    }
    // SAFETY: the call only writes the struct it is given, which is plain data of the documented layout.
    let (id, z) = unsafe {
        let mut z: Tzi = std::mem::zeroed();
        (GetTimeZoneInformation(&mut z), z)
    };
    let s = -(z.bias + z.std_bias) * 60;
    match id {
        0xffff_ffff => (0, 0),
        2 => (s, -(z.bias + z.dst_bias) * 60),
        _ => (s, s),
    }
}

/// Level in % (the X5 reports 0..100 with scale 0) and whether it charges.
pub fn level(b: &pb::BatteryInfo) -> (u32, bool) {
    let l = match b.battery_scale {
        0 => b.battery_level,
        s => b.battery_level * 100 / s,
    };
    (
        l,
        b.power_type == pb::battery_info::PowerType::Adapter as i32,
    )
}

/// Low enough to warn, on every screen: 20 % or less and not charging. The X5 shut down at level 9 (2026-10-23), so
/// its gauge's last tenth is not to be trusted; twice that leaves time to stop or plug it in.
pub fn low(b: &pb::BatteryInfo) -> bool {
    let (l, c) = level(b);
    l <= 20 && !c
}

/// Too low to start a recording or timelapse: 5 % or less and not charging (the app's rule).
pub fn empty(b: &pb::BatteryInfo) -> bool {
    let (l, c) = level(b);
    l <= 5 && !c
}

pub fn battery(b: &pb::BatteryInfo) -> String {
    let (l, c) = level(b);
    format!(
        "{l} %{}",
        match (c, low(b)) {
            (true, _) => ", charging",
            (_, true) => ", low",
            _ => "",
        }
    )
}

impl W {
    fn send(&self, u: Up) {
        let _ = self.up.send(u);
        (self.wake)()
    }

    fn info(&self) {
        self.send(Up::Info(self.info.clone()))
    }

    fn note(&self, s: impl Into<String>, err: bool) {
        self.send(Up::Note(s.into(), err))
    }

    fn run(mut self, rx: Receiver<Cmd>) {
        self.cache.trim(self.cache.cap);
        loop {
            let busy = !self.fly.is_empty()
                || self.opening.is_some()
                || (self.cam.is_some() || self.info.src.is_some()) && !self.todo.is_empty()
                || self.cur.is_some()
                || self.relist.is_some()
                || self.retry.is_some()
                || self.clk.is_some();
            let t = Duration::from_millis(if busy { 15 } else { 250 });
            match rx.recv_timeout(t) {
                Ok(Cmd::Quit) | Err(RecvTimeoutError::Disconnected) => break,
                Ok(c) => self.cmd(c),
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.tick();
        }
        self.close();
        // the end of the running download goes into the queue before it is saved; nothing new starts
        for _ in 0..200 {
            if self.cur.is_none() {
                break;
            }
            dl::end(&mut self.q.lock().unwrap(), &mut self.cur);
            thread::sleep(Duration::from_millis(10));
        }
        self.q.lock().unwrap().save();
    }

    fn tick(&mut self) {
        if let Some(r) = self.opening.as_mut().and_then(link::Opening::poll) {
            self.opening = None;
            self.opened(r);
        }
        if let Some(k) = &mut self.clk {
            k.p.retain(|p| {
                p.poll()
                    .map(|r| {
                        k.lost |= r.as_ref().is_err_and(|e| !matches!(e, Error::Status(..)));
                        k.err = k.err.take().or(r.err())
                    })
                    .is_none()
            });
            if k.p.is_empty() {
                let Clk {
                    err,
                    lost,
                    ok,
                    mut conn,
                    ..
                } = self.clk.take().unwrap();
                // the clock as set, for the Clock row (the connect read the options before the sync), also after a
                // refusal: one of the zone alone leaves the clock set. Not after an answer that never came or came
                // garbled: on a silent camera GET_OPTIONS would hold the worker until its own timeout
                if !lost {
                    self.opts()
                }
                match (err, conn.first_mut()) {
                    (Some(e), _) => conn.push((format!("time sync: {e}"), true)),
                    (None, Some(c)) => c.0 = format!("{}; {ok}", c.0),
                    (None, None) => conn.push((ok, false)),
                }
                for (s, e) in conn {
                    self.note(s, e)
                }
            }
        }
        let now = Instant::now();
        if self.scan.is_some_and(|t| now >= t) {
            self.scan = Some(now + SCAN);
            let p = link::scan();
            if self.info.plug != Some(p) {
                self.cmd(Cmd::Plug(p))
            }
        }
        if let Some(w) = &self.watch
            && self.look.is_none_or(|t| now >= t)
        {
            self.look = Some(now + SCAN);
            let v = w.vols();
            if v != self.info.disk {
                let gone =
                    (self.info.src.as_ref()).is_some_and(|r| !v.iter().any(|x| x.root == *r));
                // a card that appears while no camera is connected and no other card shown: the album shows it
                let new = (v.iter())
                    .find(|x| !self.info.disk.contains(x))
                    .filter(|_| self.cam.is_none() && self.info.src.is_none())
                    .map(|x| x.root.clone());
                self.info.disk = v;
                match (gone, new) {
                    (true, _) => {
                        self.source(None);
                        self.note("the card was removed", false)
                    }
                    (_, Some(r)) => {
                        self.note(format!("card mounted at {}", r.display()), false);
                        self.source(Some(r))
                    }
                    _ => self.info(),
                }
            }
        }
        if self.retry.is_some_and(|t| now >= t) {
            self.retry = None;
            if self.info.plug == Some(Plug::App) && self.auto.load(Relaxed) {
                self.connect(self.sync)
            }
        }
        if self.relist.is_some_and(|t| now >= t) {
            self.album()
        }
        self.thumbs();
        // no new download while the camera captures (§7.12)
        let c = self.cam.as_ref().filter(|_| !self.info.busy);
        if let Some(p) = dl::step(&self.q, c, &self.info.disk, &mut self.cur) {
            self.send(Up::Done(p))
        }
    }

    fn cmd(&mut self, c: Cmd) {
        match c {
            Cmd::Plug(p) => {
                self.info.plug = Some(p);
                self.info();
                if p == Plug::App && self.cam.is_none() && self.auto.load(Relaxed) {
                    // a camera plugged in is the app's to connect, with the clock; one back on the bus after a lost
                    // link (a glitch re-enumerates it), or while a retry waits, is that session's reconnect
                    let re = self.lost || self.retry.is_some();
                    self.connect(!re || self.sync)
                }
            }
            Cmd::Connect { clock } => self.connect(clock),
            Cmd::Disconnect => {
                if self.opening.take().is_some() {
                    self.lost = false;
                    (self.info.conn, self.info.since) = (Conn::Off, None);
                    self.info();
                    self.note("connection cancelled", false)
                } else if self.cam.is_some() {
                    self.close();
                    self.note("disconnected", false)
                }
            }
            Cmd::Refresh => {
                self.opts();
                self.album()
            }
            // first, also ones sent before that the UI let go of
            Cmd::Want(ks) => {
                self.todo.retain(|k| !ks.contains(k));
                for k in ks.into_iter().rev() {
                    if self.keys.contains_key(&k) && !self.fly.iter().any(|f| f.0 == k) {
                        self.todo.push_front(k)
                    }
                }
            }
            Cmd::Ev(s, e) if s == self.sid => self.event(e),
            Cmd::Ev(..) | Cmd::Quit => {}
            Cmd::Watch(w) => (self.watch, self.look) = (Some(w), None),
            Cmd::Source(r) => self.source(r),
            Cmd::Wifi { ip, id, clock } => {
                if self.cam.is_some() || self.opening.is_some() {
                    return self.note("disconnect the camera first", true);
                }
                (self.retry, self.sync) = (None, clock);
                (self.info.conn, self.info.since) = (Conn::Connecting, Some(Instant::now()));
                self.info();
                self.note(format!("connecting over Wi-Fi to {ip}"), false);
                self.opening = Some(Camera::wifi(ip, &id));
            }
            c => self.act(c),
        }
    }

    /// Starts a connection, which sets the camera clock if `clock`; [`W::tick`] takes it once the camera synced.
    fn connect(&mut self, clock: bool) {
        if self.cam.is_some() || self.opening.is_some() {
            return;
        }
        (self.retry, self.sync) = (None, clock);
        (self.info.conn, self.info.since) = (Conn::Connecting, Some(Instant::now()));
        self.info();
        self.opening = Some((self.connect)());
    }

    fn opened(&mut self, r: link::Result<(Camera, Receiver<Event>)>) {
        let t = self.info.since.take().unwrap_or_else(Instant::now);
        match r.and_then(|(c, ev)| Ok((c.identify().wait()?, c, ev))) {
            Ok((id, c, ev)) => {
                self.sid += 1;
                let (tx, s, tap) = (self.tx.clone(), self.sid, self.tap.clone());
                let _ = thread::Builder::new()
                    .name("x5-events".into())
                    .spawn(move || {
                        for e in ev {
                            // the live preview's packets go straight to the Capture screen, a copy of each note too
                            let e = match e {
                                Event::Frame(f) => {
                                    let mut t = tap.lock().unwrap();
                                    if let Some(x) = &t.frames
                                        && x.try_send(f).is_err()
                                    {
                                        t.lost += 1;
                                    }
                                    continue;
                                }
                                Event::Note(n) => {
                                    if let Some(f) = &tap.lock().unwrap().notes {
                                        f(&n)
                                    }
                                    Event::Note(n)
                                }
                                e => e,
                            };
                            if tx.send(Cmd::Ev(s, e)).is_err() {
                                break;
                            }
                        }
                    });
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let mut conn = vec![(format!("connected to {} in {ms:.0} ms", id.model), false)];
                if let Some(b) = id.battery.as_ref().filter(|b| low(b)) {
                    conn.push((format!("{}: charge the camera", battery(b)), true))
                }
                (self.info.battery, self.info.id) = (id.battery, Some(id));
                self.info.conn = Conn::On;
                self.info.tries += 1;
                self.cam = Some(c.clone());
                // the app's order on every USB and Wi-Fi connect (§5.1): options and capture status, then the clock
                // (its answers not waited for, a refusal leaves the session up), then the Capture screen and the
                // listing. A connect that leaves the clock has its notes after the listing
                self.opts();
                let st = c.capture_status().wait();
                let conn = match self.sync {
                    true => self.time(&c, conn),
                    false => conn,
                };
                {
                    let mut tp = self.tap.lock().unwrap();
                    (tp.cam, tp.sid) = (Some(c), self.sid);
                }
                self.list(Some(st));
                for (s, e) in conn {
                    self.note(s, e)
                }
            }
            Err(e) => {
                self.info.conn = Conn::Off;
                self.info.tries += 1;
                self.info();
                // another client holds the session: it may let go
                if matches!(e, Error::Busy) && self.auto.load(Relaxed) {
                    self.retry = Some(Instant::now() + Duration::from_secs(5))
                }
                let m = match e {
                    Error::Silent(_) => format!(
                        "the X5 is in app mode but does not answer ({e}): it may be asleep or hung. Wake it with \
                         its power button, or unplug and replug it, then press Connect"
                    ),
                    e => format!("connect: {e}"),
                };
                self.note(m, true);
            }
        }
    }

    /// Ends the session cleanly (or cancels the connection syncing): a running download keeps its partial data and
    /// goes back to the queue.
    fn close(&mut self) {
        (self.opening, self.lost) = (None, false);
        if let Some(c) = self.cam.take() {
            let _ = c.close().wait();
        }
        self.reset()
    }

    fn reset(&mut self) {
        (self.cam, self.clk) = (None, None);
        {
            let mut t = self.tap.lock().unwrap();
            t.cam = None;
            t.sid = t.sid.max(self.sid) + 1;
        }
        self.sid += 1;
        // a card's album stays
        if self.info.src.is_none() {
            (self.todo, self.fly, self.relist, self.t0) = Default::default();
            self.sent.clear();
            self.keys.clear();
        }
        let i = &mut self.info;
        *i = Info {
            plug: i.plug,
            tries: i.tries,
            disk: std::mem::take(&mut i.disk),
            src: i.src.take(),
            ..Default::default()
        };
        self.info();
    }

    /// Shows a card's album (its mount point) or the camera's.
    fn source(&mut self, r: Option<PathBuf>) {
        (self.todo, self.fly, self.relist, self.t0) = Default::default();
        self.sent.clear();
        self.keys.clear();
        self.info.src = r;
        self.info();
        self.album()
    }

    /// Battery, storage, Wi-Fi and clock.
    fn opts(&mut self) {
        let Some(c) = &self.cam else { return };
        use pb::OptionType::*;
        match c
            .options(&[BatteryInfo, StorageState, WifiInfo, LocalTime])
            .wait()
        {
            Ok(o) => {
                let i = &mut self.info;
                i.battery = o.battery_info.or(i.battery.take());
                i.storage = o.storage_state;
                i.wifi = o.wifi_info.map(|w| (w.ssid, w.password));
                i.skew = (o.local_time > 0)
                    .then(|| (o.local_time as i64).saturating_sub(clock().0 as i64));
            }
            Err(e) => self.note(format!("options: {e}"), true),
        }
        self.info();
    }

    /// Sets the camera clock and zone to this computer's with the app's two SET_OPTIONS (PROTOCOL.md §5.3), its
    /// answers not waited for: [`W::tick`] notes them, after `conn` and the notes of a sync still in flight, and reads
    /// the options again once the camera answered both, a refusal too (not after a garbled answer). Without an IANA
    /// zone (not unix: [`clock`]) the camera keeps its clock, said once: then `conn` comes back for the caller to note.
    fn time(&mut self, c: &Camera, mut conn: Vec<(String, bool)>) -> Vec<(String, bool)> {
        if !cfg!(unix) {
            if !std::mem::replace(&mut self.told, true) {
                let s = "camera clock not set: no IANA time zone known on this system";
                conn.push((s.into(), false))
            }
            return conn;
        }
        let (t, _, off, tz, s) = clock();
        let mut v = self.clk.take().map_or(vec![], |k| k.conn);
        v.extend(conn);
        self.clk = Some(Clk {
            p: c.set_time(t, off, &tz).into(),
            err: None,
            lost: false,
            ok: format!("camera clock set to {s} ({tz})"),
            conn: v,
        });
        vec![]
    }

    /// Lists the card's or the camera's files (not while it records) and queues the thumbnails not sent yet.
    fn album(&mut self) {
        self.list(None)
    }

    /// [`W::album`] with the camera's capture status if it was just read.
    fn list(&mut self, st: Option<link::Result<pb::CameraCaptureStatus>>) {
        self.relist = None;
        let t = Instant::now();
        if let Some(r) = self.info.src.clone() {
            match disk::list(&r) {
                Ok(v) => self.show(v, t),
                Err(e) => self.note(format!("{}: {e}", r.display()), true),
            }
            return;
        }
        let Some(c) = self.cam.clone() else {
            return self.send(Up::Album(vec![]));
        };
        let r = (|| {
            if rec(st.unwrap_or_else(|| c.capture_status().wait())?.state) {
                return Ok(None);
            }
            use pb::{CardLocation::StorClCamera as L, MediaType as M};
            let mut u = c.files(M::VideoAndPhoto, L).wait()?;
            u.extend(c.files(M::Dng, L).wait()?);
            let fi = c.file_infos(L).wait()?;
            // not part of every firmware
            let ed = c.favorites().wait().unwrap_or_default();
            Ok::<_, Error>(Some(album::group(&u, &fi, &ed)))
        })();
        match r {
            Ok(None) => {
                self.info.busy = true;
                self.q.lock().unwrap().hold();
                self.info();
                self.note(
                    "the camera is recording: its files are listed once it stops",
                    false,
                )
            }
            Ok(Some(v)) => {
                if self.info.busy {
                    self.info.busy = false;
                    self.info()
                }
                self.show(v, t)
            }
            Err(e) => self.note(format!("album: {e}"), true),
        }
    }

    /// Hands a listing listed since `t` to the UI and queues the thumbnails it does not have yet: a card's from
    /// its files (keyed by their path), the camera's by URI.
    fn show(&mut self, v: Vec<Item>, t: Instant) {
        self.stats = Stats {
            list_ms: t.elapsed().as_secs_f64() * 1e3,
            ..Default::default()
        };
        let (sn, r) = match &self.info.src {
            // cards have no serial: the cache key's path, size and date tell their files apart
            Some(r) => ("card", Some(r)),
            None => (self.info.id.as_ref().map_or("", |i| &i.serial[..]), None),
        };
        self.keys = v
            .iter()
            .map(|it| {
                let u = match r {
                    Some(r) => r.join(&it.uris[0][1..]).to_string_lossy().into_owned(),
                    None => it.uris[0].clone(),
                };
                (it.key.clone(), (u, self.cache.path(sn, it)))
            })
            .collect();
        self.todo = v
            .iter()
            .filter(|it| !self.sent.contains(&it.key) && !self.fly.iter().any(|f| f.0 == it.key))
            .map(|it| it.key.clone())
            .collect();
        self.t0 = Some(Instant::now());
        self.send(Up::Album(v));
    }

    /// Cache hits (a batch per turn), then the camera with [`FLY`] requests in flight.
    fn thumbs(&mut self) {
        let mut n = 0;
        while self.fly.len() < FLY && n < 16 {
            let Some(k) = self.todo.pop_front() else {
                break;
            };
            let Some((u, p)) = self.keys.get(&k) else {
                continue;
            };
            if let Some(t) = p.as_deref().and_then(Cache::get) {
                self.stats.cached += 1;
                n += 1;
                self.sent.insert(k.clone());
                self.send(Up::Thumb(k, Some(t)));
            } else if self.info.src.is_some() {
                // a card: the trailer's picture, read here (a few ms), four a turn
                let t = disk::thumb(Path::new(u));
                if let (Some((w, h, px)), Some(p)) = (&t, p) {
                    Cache::put(p, *w, *h, px)
                }
                self.stats.cam += 1;
                n += 4;
                self.sent.insert(k.clone());
                self.send(Up::Thumb(k, t));
            } else if let Some(c) = &self.cam {
                self.fly.push((k, p.clone(), c.thumbnail(u)));
            } else {
                self.todo.push_front(k);
                break;
            }
        }
        let mut i = 0;
        while i < self.fly.len() {
            let Some(r) = self.fly[i].2.poll() else {
                i += 1;
                continue;
            };
            let (k, p, _) = self.fly.swap_remove(i);
            let t = r.ok().and_then(|b| {
                self.stats.cam += 1;
                self.stats.bytes += b.len();
                album::thumb(&b)
            });
            if let (Some((w, h, px)), Some(p)) = (&t, &p) {
                Cache::put(p, *w, *h, px)
            }
            self.sent.insert(k.clone());
            self.send(Up::Thumb(k, t));
        }
        if self.todo.is_empty()
            && self.fly.is_empty()
            && let Some(t) = self.t0.take()
        {
            self.stats.secs = t.elapsed().as_secs_f64();
            self.send(Up::Stats(self.stats));
        }
    }

    fn event(&mut self, e: Event) {
        let soon = Some(Instant::now() + Duration::from_millis(800));
        match e {
            Event::Note(Note::Battery(b)) => {
                self.info.battery = Some(b);
                self.info()
            }
            Event::Note(Note::BatteryLow(b)) => {
                self.note(
                    format!(
                        "{}: charge the camera, it shuts down when empty",
                        battery(&b)
                    ),
                    true,
                );
                self.info.battery = Some(b);
                self.info()
            }
            Event::Note(Note::Storage(_) | Note::StorageFull | Note::StorageLocation(_)) => {
                self.opts();
                self.relist = soon
            }
            // new, deleted or re-flagged files (8284: favourite change)
            Event::Note(
                Note::DeleteResult(_)
                | Note::CaptureStopped(_)
                | Note::Photo(_)
                | Note::AutoSplit(_)
                | Note::Other(8284, _),
            ) => self.relist = soon,
            Event::Note(Note::CaptureStatus(s)) => {
                let b = rec(s.state);
                if self.info.busy && !b {
                    self.relist = soon
                }
                if b {
                    self.q.lock().unwrap().hold()
                }
                self.info.busy = b;
                self.info()
            }
            Event::Note(Note::Shutdown(_)) => {
                self.close();
                self.note("the camera shut down", false)
            }
            Event::Closed(e) => {
                self.reset();
                self.lost = true;
                self.note(format!("link lost: {e}"), true);
                if self.info.plug == Some(Plug::App) && self.auto.load(Relaxed) {
                    self.retry = Some(Instant::now() + Duration::from_secs(2))
                }
            }
            _ => {}
        }
    }

    /// The commands that change the camera (the app's screen asks the user first).
    fn act(&mut self, c: Cmd) {
        let Some(cam) = self.cam.clone() else {
            return self.note("not connected", true);
        };
        // noted once the camera answered, the options read again if it took both
        if let Cmd::Time = c {
            if !cfg!(unix) {
                return self.note("time sync: no IANA time zone known on this system", true);
            }
            return drop(self.time(&cam, vec![]));
        }
        // the files are off limits while it records (§7.12), the album shows what it had. The camera is asked too: the
        // note of a capture just started can still be on its way, a card's album left the connect's status unread; a
        // timelapse, interval or starlapse running counts as well
        if let Cmd::Delete(_) = c {
            let st = match self.info.busy {
                true => Ok(None),
                false => cam.capture_status().wait().map(|s| Some(s.state)),
            };
            match st {
                Ok(Some(s)) if !capture::running(s) => {}
                Ok(s) => {
                    if s.is_some_and(rec) {
                        self.info.busy = true;
                        self.q.lock().unwrap().hold();
                        self.info()
                    }
                    return self.note(
                        "delete: the camera is recording; delete once it stops",
                        true,
                    );
                }
                Err(e) => return self.note(format!("delete: capture status: {e}"), true),
            }
        }
        let what = match c {
            Cmd::Delete(_) => "delete",
            Cmd::Fav(_) => "favourites",
            Cmd::Format => "format",
            _ => return,
        };
        let r: link::Result<String> = match c {
            // the app's passes (PROTOCOL.md §7.8): batches of 200 URIs 150 ms apart, one by one from a batch that
            // fails on, a failure of one ends it; then the leftovers of a fresh listing, five passes at most
            Cmd::Delete(u) => (|| {
                use pb::{CardLocation::StorClCamera as L, MediaType as M};
                let mut left = u.clone();
                for _ in 0..5 {
                    let (mut i, mut n) = (0, 200);
                    while i < left.len() {
                        if i > 0 || n == 1 {
                            thread::sleep(Duration::from_millis(150))
                        }
                        let ch = &left[i..(i + n).min(left.len())];
                        match cam.delete(ch.to_vec()).wait() {
                            Ok(_) => i += ch.len(),
                            Err(e) if n == 1 => return Err(e),
                            Err(_) => n = 1,
                        }
                    }
                    // the app trusts the answer for a single file
                    if left.len() == 1 {
                        left.clear();
                        break;
                    }
                    let mut l: HashSet<String> =
                        cam.files(M::VideoAndPhoto, L).wait()?.into_iter().collect();
                    l.extend(cam.files(M::Dng, L).wait()?);
                    left.retain(|x| l.contains(x));
                    if left.is_empty() {
                        break;
                    }
                }
                Ok(format!(
                    "deleted {} of {} files",
                    u.len() - left.len(),
                    u.len()
                ))
            })(),
            // the app sends at most 100 per request; the newer timestamp wins on the camera
            Cmd::Fav(v) => (|| {
                let ts = clock().1;
                for ch in v.chunks(100) {
                    let f = ch.iter().map(|(u, f)| pb::SetFavorite {
                        favorite_flag: *f as u32,
                        material_url: u.clone(),
                        modify_timestamp: ts,
                        ..Default::default()
                    });
                    cam.set_favorites(f.collect()).wait()?;
                }
                Ok(format!("favourite flag set on {} files", v.len()))
            })(),
            Cmd::Format => cam
                .format(pb::CardLocation::StorClCamera)
                .wait()
                .map(|_| "SD card formatted".into()),
            _ => return,
        };
        match r {
            Ok(s) => self.note(s, false),
            Err(e) => self.note(format!("{what}: {e}"), true),
        }
        self.opts();
        self.album();
    }
}

#[cfg(test)]
mod tests;
