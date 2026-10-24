//! Camera screen: the X5 over USB. A worker thread owns the link: it follows USB hotplug, connects (automatically
//! if wanted), reads identity, battery, storage, Wi-Fi and clock, lists the album, fetches thumbnails (disk cache
//! first, the ones on screen first), runs the download queue and the camera commands, and follows the camera's
//! notifications. The UI sends it commands and draws what it reports.

use crate::{
    album::{self, Cache, Item, Kind, Sort, View},
    dl::{self, St},
};
use eframe::egui;
use link::{Camera, Error, Event, Note, Plug, pb};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub type Connect = Box<dyn FnMut() -> link::Result<(Camera, Receiver<Event>)> + Send>;

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
    /// connection attempts that finished
    pub tries: u32,
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
    Connect,
    Disconnect,
    Refresh,
    /// Item keys whose thumbnails go first.
    Want(Vec<String>),
    Delete(Vec<String>),
    Fav(Vec<(String, bool)>),
    Format,
    Time,
    Ev(u32, Event),
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
    /// `usb`: follow the bus (otherwise only [`Cmd::Connect`] connects); `wake`: something arrived.
    pub fn spawn(
        connect: Connect,
        q: dl::Shared,
        cache: Cache,
        auto: Arc<AtomicBool>,
        usb: bool,
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

/// Thumbnail textures the album keeps (320 × 160 RGBA, 200 KB each): the least recently drawn go first.
const TEX: usize = 512;

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
}

/// Unix time, the app's favourite timestamp (unix + standard offset: Java's `ZONE_OFFSET`, no DST), offset from UTC
/// in s, IANA zone (TZ, else the /etc/localtime link), local time as text.
fn clock() -> (u64, i64, i32, String, String) {
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
        self.cache.trim(album::CAP);
        loop {
            let busy = !self.fly.is_empty()
                || self.cam.is_some() && !self.todo.is_empty()
                || self.cur.is_some()
                || self.relist.is_some()
                || self.retry.is_some();
            let t = Duration::from_millis(if busy { 15 } else { 250 });
            match rx.recv_timeout(t) {
                Ok(Cmd::Quit) | Err(RecvTimeoutError::Disconnected) => break,
                Ok(c) => self.cmd(c),
                Err(RecvTimeoutError::Timeout) => {}
            }
            self.tick();
        }
        self.close();
        // the end of the running download goes into the queue before it is saved
        for _ in 0..200 {
            if self.cur.is_none() {
                break;
            }
            dl::step(&self.q, None, &mut self.cur);
            thread::sleep(Duration::from_millis(10));
        }
        self.q.lock().unwrap().save();
    }

    fn tick(&mut self) {
        let now = Instant::now();
        if self.scan.is_some_and(|t| now >= t) {
            self.scan = Some(now + SCAN);
            let p = link::scan();
            if self.info.plug != Some(p) {
                self.cmd(Cmd::Plug(p))
            }
        }
        if self.retry.is_some_and(|t| now >= t) {
            self.retry = None;
            if self.info.plug == Some(Plug::App) && self.auto.load(Relaxed) {
                self.connect()
            }
        }
        if self.relist.is_some_and(|t| now >= t) {
            self.album()
        }
        self.thumbs();
        // no new download while the camera captures (§7.12)
        let c = self.cam.as_ref().filter(|_| !self.info.busy);
        if let Some(p) = dl::step(&self.q, c, &mut self.cur) {
            self.send(Up::Done(p))
        }
    }

    fn cmd(&mut self, c: Cmd) {
        match c {
            Cmd::Plug(p) => {
                self.info.plug = Some(p);
                self.info();
                if p == Plug::App && self.cam.is_none() && self.auto.load(Relaxed) {
                    self.connect()
                }
            }
            Cmd::Connect => self.connect(),
            Cmd::Disconnect => {
                if self.cam.is_some() {
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
            c => self.act(c),
        }
    }

    fn connect(&mut self) {
        if self.cam.is_some() {
            return;
        }
        self.retry = None;
        self.info.conn = Conn::Connecting;
        self.info();
        let t = Instant::now();
        match (self.connect)().and_then(|(c, ev)| Ok((c.identify().wait()?, c, ev))) {
            Ok((id, c, ev)) => {
                self.sid += 1;
                let (tx, s) = (self.tx.clone(), self.sid);
                let _ = thread::Builder::new()
                    .name("x5-events".into())
                    .spawn(move || {
                        for e in ev {
                            if tx.send(Cmd::Ev(s, e)).is_err() {
                                break;
                            }
                        }
                    });
                self.note(
                    format!(
                        "connected to {} in {:.0} ms",
                        id.model,
                        t.elapsed().as_secs_f64() * 1e3
                    ),
                    false,
                );
                (self.info.battery, self.info.id) = (id.battery, Some(id));
                self.info.conn = Conn::On;
                self.info.tries += 1;
                self.cam = Some(c);
                self.opts();
                self.album();
            }
            Err(e) => {
                self.info.conn = Conn::Off;
                self.info.tries += 1;
                self.info();
                // another client holds the session: it may let go
                if matches!(e, Error::Busy) && self.auto.load(Relaxed) {
                    self.retry = Some(Instant::now() + Duration::from_secs(5))
                }
                self.note(format!("connect: {e}"), true);
            }
        }
    }

    /// Ends the session cleanly: a running download keeps its partial data and goes back to the queue.
    fn close(&mut self) {
        if let Some(c) = self.cam.take() {
            let _ = c.close().wait();
        }
        self.reset()
    }

    fn reset(&mut self) {
        self.cam = None;
        self.sid += 1;
        (self.todo, self.fly, self.relist, self.t0) = Default::default();
        self.sent.clear();
        self.keys.clear();
        self.info = Info {
            plug: self.info.plug,
            tries: self.info.tries,
            ..Default::default()
        };
        self.info();
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

    /// Lists the camera's files (not while it records) and queues the thumbnails not sent yet.
    fn album(&mut self) {
        self.relist = None;
        let Some(c) = self.cam.clone() else { return };
        let t = Instant::now();
        let r = (|| {
            if rec(c.capture_status().wait()?.state) {
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
                self.stats = Stats {
                    list_ms: t.elapsed().as_secs_f64() * 1e3,
                    ..Default::default()
                };
                let sn = self.info.id.as_ref().map_or("", |i| &i.serial);
                self.keys = v
                    .iter()
                    .map(|it| {
                        (
                            it.key.clone(),
                            (it.uris[0].clone(), self.cache.path(sn, it)),
                        )
                    })
                    .collect();
                self.todo = v
                    .iter()
                    .filter(|it| {
                        !self.sent.contains(&it.key) && !self.fly.iter().any(|f| f.0 == it.key)
                    })
                    .map(|it| it.key.clone())
                    .collect();
                self.t0 = Some(Instant::now());
                self.send(Up::Album(v));
            }
            Err(e) => self.note(format!("album: {e}"), true),
        }
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
            Event::Note(Note::Battery(b) | Note::BatteryLow(b)) => {
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
                self.note(format!("link lost: {e}"), true);
                if self.info.plug == Some(Plug::App) && self.auto.load(Relaxed) {
                    self.retry = Some(Instant::now() + Duration::from_secs(2))
                }
            }
            _ => {}
        }
    }

    /// The commands that change the camera, each confirmed in the UI first.
    fn act(&mut self, c: Cmd) {
        let Some(cam) = self.cam.clone() else {
            return self.note("not connected", true);
        };
        let what = match c {
            Cmd::Delete(_) => "delete",
            Cmd::Fav(_) => "favourites",
            Cmd::Format => "format",
            Cmd::Time => "time sync",
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
            Cmd::Time => {
                let (t, _, off, tz, s) = clock();
                cam.set_time(t, off, &tz)
                    .wait()
                    .map(|_| format!("camera clock set to {s} ({tz})"))
            }
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

/// Destination folder and auto-connect, as `key=value` lines in `<config>/x5desk/camera`.
struct Set {
    dest: PathBuf,
    auto: bool,
    file: Option<PathBuf>,
}

impl Set {
    fn load(file: Option<PathBuf>) -> Set {
        let mut s = Set {
            dest: std::env::var_os("HOME")
                .map_or_else(|| PathBuf::from("."), PathBuf::from)
                .join("Videos")
                .join("x5desk"),
            auto: true,
            file,
        };
        let t = s
            .file
            .as_ref()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .unwrap_or_default();
        for l in t.lines() {
            match l.split_once('=') {
                Some(("dest", v)) => s.dest = v.into(),
                Some(("auto", v)) => s.auto = v == "1",
                _ => {}
            }
        }
        s
    }

    fn save(&self) {
        let Some(f) = &self.file else { return };
        let s = format!("dest={}\nauto={}\n", self.dest.display(), self.auto as u8);
        if let Err(e) = f
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|_| std::fs::write(f, s))
        {
            eprintln!("{}: {e}", f.display());
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    Album,
    Downloads,
    Settings,
}

impl Pane {
    pub fn parse(s: &str) -> Option<Pane> {
        Some(match s {
            "" | "album" => Pane::Album,
            "downloads" => Pane::Downloads,
            "settings" => Pane::Settings,
            _ => return None,
        })
    }
}

/// A camera change waiting for the user's confirmation.
enum Ask {
    /// URIs, items, items not downloaded
    Delete(Vec<String>, usize, usize),
    Fav(Vec<(String, bool)>),
    Format,
    Time,
}

fn mb(b: u64) -> String {
    match b {
        0..1_000_000_000 => format!("{:.1} MB", b as f64 / 1e6),
        _ => format!("{:.2} GB", b as f64 / 1e9),
    }
}

fn dur(ms: u64) -> String {
    let s = ms / 1000;
    match s {
        0..3600 => format!("{}:{:02}", s / 60, s % 60),
        _ => format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60),
    }
}

fn battery(b: &pb::BatteryInfo) -> String {
    // the X5 reports level 0..100 with scale 0
    let l = match b.battery_scale {
        0 => b.battery_level,
        s => b.battery_level * 100 / s,
    };
    let c = b.power_type == pb::battery_info::PowerType::Adapter as i32;
    format!("{l} %{}", if c { ", charging" } else { "" })
}

fn storage(s: &pb::StorageState) -> String {
    use pb::CardState::*;
    match s.card_state() {
        StorCsPass => format!("{} free of {}", mb(s.free_space), mb(s.total_space)),
        StorCsNocard => "no SD card".into(),
        StorCsNospace => "SD card full".into(),
        StorCsInvalidFormat => "SD card needs formatting".into(),
        StorCsFormatting => "formatting".into(),
        x => format!("{x:?}"),
    }
}

pub struct Cam {
    w: Worker,
    q: dl::Shared,
    auto: Arc<AtomicBool>,
    set: Set,
    info: Info,
    items: Vec<Item>,
    shown: Vec<usize>,
    view: View,
    dirty: bool,
    sel: HashSet<String>,
    anchor: Option<usize>,
    /// thumbnail textures by item key (`None`: the item has none) and the album frame they were last drawn in
    tex: HashMap<String, (Option<egui::TextureHandle>, Cell<u64>)>,
    frame: u64,
    /// items whose files are all in the destination folder
    have: HashSet<String>,
    stats: Option<Stats>,
    pub pane: Pane,
    ask: Option<Ask>,
    typed: String,
    pw: bool,
    note: Option<(String, bool)>,
    dlg: Option<Receiver<Option<PathBuf>>>,
    wanted: Vec<String>,
    ctx: egui::Context,
}

impl Cam {
    /// `usb`: follow the bus and connect (off for test-hook runs of other screens); `dls`: camera URIs to queue
    /// (test hook).
    pub fn new(ctx: &egui::Context, pane: Option<Pane>, dls: &[String], usb: bool) -> Cam {
        let cfg = crate::xdg("XDG_CONFIG_HOME", ".config").map(|d| d.join("x5desk"));
        let set = Set::load(cfg.as_ref().map(|d| d.join("camera")));
        let mut q = dl::Queue::load(cfg.map(|d| d.join("downloads")));
        for u in dls {
            q.add(u, set.dest.join(album::name(u)));
        }
        let q = Arc::new(Mutex::new(q));
        Cam::with(ctx, pane, set, q, Box::new(Camera::usb), Cache::new(), usb)
    }

    fn with(
        ctx: &egui::Context,
        pane: Option<Pane>,
        set: Set,
        q: dl::Shared,
        connect: Connect,
        cache: Cache,
        usb: bool,
    ) -> Cam {
        let auto = Arc::new(AtomicBool::new(set.auto));
        let c = ctx.clone();
        let w = Worker::spawn(connect, q.clone(), cache, auto.clone(), usb, move || {
            c.request_repaint()
        });
        Cam {
            w,
            q,
            auto,
            set,
            info: Info::default(),
            items: vec![],
            shown: vec![],
            view: View::default(),
            dirty: false,
            sel: HashSet::new(),
            anchor: None,
            tex: HashMap::new(),
            frame: 0,
            have: HashSet::new(),
            stats: None,
            pane: pane.unwrap_or(Pane::Album),
            ask: None,
            typed: String::new(),
            pw: false,
            note: None,
            dlg: None,
            wanted: vec![],
            ctx: ctx.clone(),
        }
    }

    /// Recordings in the destination folder and finished downloads elsewhere, one file per recording, oldest
    /// first (for the Library: what is there, also after the queue let go of it).
    pub fn done(&self) -> Vec<PathBuf> {
        let q = self.q.lock().unwrap();
        let mut v: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&self.set.dest)
            .into_iter()
            .flatten()
            .filter_map(|e| Some(e.ok()?.path()))
            .chain(
                q.v.iter()
                    .filter(|d| d.st == St::Done)
                    .map(|d| d.dst.clone()),
            )
            .filter(|p| {
                p.extension().is_some_and(|e| {
                    ["insv", "lrv", "mp4"]
                        .iter()
                        .any(|x| e.eq_ignore_ascii_case(x))
                })
            })
            .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
            .collect();
        v.sort_by(|a, b| b.cmp(a));
        let mut seen = HashSet::new();
        v.retain(|x| {
            seen.insert(crate::library::key(
                &x.1.file_name().unwrap_or_default().to_string_lossy(),
            ))
        });
        v.into_iter().rev().map(|x| x.1).collect()
    }

    /// Nothing pending: the bus was looked at, an automatic connection was tried, an album has its thumbnails
    /// (test hook).
    pub fn idle(&self) -> bool {
        let i = &self.info;
        i.plug.is_some()
            && match i.conn {
                Conn::Connecting => false,
                Conn::On => i.busy || self.stats.is_some(),
                Conn::Off => i.plug != Some(Plug::App) || !self.set.auto || i.tries > 0,
            }
    }

    fn have(&mut self) {
        let d: HashSet<std::ffi::OsString> = std::fs::read_dir(&self.set.dest)
            .into_iter()
            .flatten()
            .filter_map(|e| Some(e.ok()?.file_name()))
            .collect();
        self.have = self
            .items
            .iter()
            .filter(|it| {
                it.uris
                    .iter()
                    .all(|u| d.contains(std::ffi::OsStr::new(album::name(u))))
            })
            .map(|it| it.key.clone())
            .collect();
    }

    /// Every frame: takes what the worker reported. Returns files that finished downloading.
    pub fn tick(&mut self) -> Vec<PathBuf> {
        let mut done = vec![];
        while let Ok(u) = self.w.rx.try_recv() {
            match u {
                Up::Info(i) => {
                    if i.conn != Conn::On && self.info.conn == Conn::On {
                        (self.items, self.stats) = (vec![], None);
                        self.tex.clear();
                        self.wanted.clear();
                        self.dirty = true;
                    }
                    self.info = i
                }
                Up::Album(v) => {
                    self.items = v;
                    self.sel
                        .retain(|k| self.items.iter().any(|it| it.key == *k));
                    self.dirty = true;
                    self.have();
                }
                Up::Thumb(k, t) => {
                    // a big card: the ones off screen wait until they are shown, the worker sends them again then
                    if t.is_some() && self.tex.len() >= TEX && !self.wanted.contains(&k) {
                        continue;
                    }
                    let t = t.map(|(w, h, px)| {
                        let img =
                            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
                        self.ctx.load_texture(
                            format!("thumb {k}"),
                            img,
                            egui::TextureOptions::LINEAR,
                        )
                    });
                    self.tex.insert(k, (t, Cell::new(self.frame)));
                }
                Up::Done(p) => {
                    self.have();
                    done.push(p)
                }
                Up::Note(s, e) => {
                    if e {
                        eprintln!("x5desk: camera: {s}")
                    }
                    self.note = Some((s, e))
                }
                Up::Stats(s) => {
                    eprintln!(
                        "x5desk: album of {} items listed in {:.0} ms; thumbnails: {} from the camera ({}) and {} \
                         from the cache in {:.2} s",
                        self.items.len(),
                        s.list_ms,
                        s.cam,
                        mb(s.bytes as u64),
                        s.cached,
                        s.secs
                    );
                    self.stats = Some(s)
                }
            }
        }
        if let Some(r) = &self.dlg
            && let Ok(p) = r.try_recv()
        {
            self.dlg = None;
            if let Some(p) = p {
                self.set.dest = p;
                self.set.save();
                self.have();
            }
        }
        if self.q.lock().unwrap().v.iter().any(|d| d.st == St::Running) {
            self.ctx.request_repaint_after(Duration::from_millis(250));
        }
        done
    }

    /// Returns a downloaded file to open in the Library.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<PathBuf> {
        egui::Panel::top("cam bar").show(ui, |ui| self.bar(ui));
        self.dialog(ui.ctx());
        let mut open = None;
        egui::CentralPanel::default_margins().show(ui, |ui| match self.pane {
            Pane::Album => open = self.album(ui),
            Pane::Downloads => open = self.downloads(ui),
            Pane::Settings => self.settings(ui),
        });
        open
    }

    fn bar(&mut self, ui: &mut egui::Ui) {
        let i = &self.info;
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let st = match (i.conn, i.plug, &i.id) {
                (Conn::On, _, Some(id)) => format!("{} · firmware {}", id.model, id.firmware),
                (Conn::Connecting, ..) => "Connecting…".into(),
                (_, Some(Plug::App), _) => "X5 found".into(),
                (_, Some(Plug::Other), _) => {
                    "Insta360 camera in another USB mode: choose Android App on its USB menu".into()
                }
                _ => "No camera: connect the X5 by USB in app mode".into(),
            };
            ui.strong(st);
            let on = i.conn == Conn::On;
            if on {
                if ui.button("Disconnect").clicked() {
                    let _ = self.w.tx.send(Cmd::Disconnect);
                }
            } else if ui
                .add_enabled(
                    i.conn == Conn::Off && i.plug == Some(Plug::App),
                    egui::Button::new("Connect"),
                )
                .clicked()
            {
                let _ = self.w.tx.send(Cmd::Connect);
            }
            if ui.checkbox(&mut self.set.auto, "Auto-connect").changed() {
                self.auto.store(self.set.auto, Relaxed);
                self.set.save();
                // a camera already plugged in gets no plug event
                if self.set.auto && i.conn == Conn::Off && i.plug == Some(Plug::App) {
                    let _ = self.w.tx.send(Cmd::Connect);
                }
            }
            if let Some(b) = &i.battery {
                ui.separator();
                ui.label(format!("Battery {}", battery(b)));
            }
            if let Some(s) = &i.storage {
                ui.separator();
                let used = 1.0 - s.free_space as f32 / s.total_space.max(1) as f32;
                ui.add(
                    egui::ProgressBar::new(used)
                        .desired_width(220.0)
                        .text(storage(s)),
                );
            }
        });
        ui.horizontal(|ui| {
            let n = self
                .q
                .lock()
                .unwrap()
                .v
                .iter()
                .filter(|d| d.st != St::Done)
                .count();
            for (p, s) in [
                (Pane::Album, format!("Album ({})", self.items.len())),
                (Pane::Downloads, format!("Downloads ({n})")),
                (Pane::Settings, "Camera".into()),
            ] {
                ui.selectable_value(&mut self.pane, p, s);
            }
            if let Some((s, e)) = &self.note {
                ui.separator();
                match e {
                    true => ui.colored_label(ui.visuals().error_fg_color, s),
                    false => ui.label(s),
                };
            }
        });
        ui.add_space(2.0);
    }

    fn album(&mut self, ui: &mut egui::Ui) -> Option<PathBuf> {
        let on = self.info.conn == Conn::On;
        let v0 = (
            self.view.sort,
            self.view.desc,
            self.view.kind,
            self.view.fav,
            self.view.find.clone(),
        );
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(on, egui::Button::new("Refresh")).clicked() {
                let _ = self.w.tx.send(Cmd::Refresh);
            }
            let k = &mut self.view.kind;
            egui::ComboBox::from_id_salt("kind")
                .selected_text(match k {
                    None => "All",
                    Some(Kind::Video) => "Videos",
                    Some(Kind::Photo) => "Photos",
                    Some(Kind::Other) => "Other",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(k, None, "All");
                    ui.selectable_value(k, Some(Kind::Video), "Videos");
                    ui.selectable_value(k, Some(Kind::Photo), "Photos");
                });
            ui.toggle_value(&mut self.view.fav, "★ Favourites");
            ui.add(
                egui::TextEdit::singleline(&mut self.view.find)
                    .hint_text("Search names")
                    .desired_width(140.0),
            );
            let s = &mut self.view.sort;
            egui::ComboBox::from_id_salt("sort")
                .selected_text(Sort::ALL.iter().find(|x| x.0 == *s).map_or("", |x| x.1))
                .show_ui(ui, |ui| {
                    for (x, n) in Sort::ALL {
                        ui.selectable_value(s, x, n);
                    }
                });
            let d = &mut self.view.desc;
            if ui
                .button(if *d { "Descending" } else { "Ascending" })
                .clicked()
            {
                *d = !*d;
            }
            ui.separator();
            if ui.button("Select all").clicked() {
                self.sel = self
                    .shown
                    .iter()
                    .map(|&i| self.items[i].key.clone())
                    .collect();
            }
            if ui.button("Select none").clicked() {
                self.sel.clear();
            }
            ui.label(format!("{} selected", self.sel.len()));
            let any = !self.sel.is_empty();
            let picked: Vec<&Item> = self
                .items
                .iter()
                .filter(|it| self.sel.contains(&it.key))
                .collect();
            if ui.add_enabled(any, egui::Button::new("Download")).clicked() {
                let mut q = self.q.lock().unwrap();
                for u in picked.iter().flat_map(|it| &it.uris) {
                    q.add(u, self.set.dest.join(album::name(u)));
                }
                q.save();
            }
            let fav = |f: bool| picked.iter().map(|it| (it.flag().to_owned(), f)).collect();
            if ui
                .add_enabled(any && on, egui::Button::new("★ Favourite"))
                .clicked()
            {
                self.ask = Some(Ask::Fav(fav(true)));
            }
            if ui
                .add_enabled(any && on, egui::Button::new("☆ Unfavourite"))
                .clicked()
            {
                self.ask = Some(Ask::Fav(fav(false)));
            }
            if ui
                .add_enabled(any && on, egui::Button::new("Delete…"))
                .clicked()
            {
                let u: Vec<String> = picked.iter().flat_map(|it| it.uris.clone()).collect();
                let not = picked
                    .iter()
                    .filter(|it| !self.have.contains(&it.key))
                    .count();
                self.ask = Some(Ask::Delete(u, picked.len(), not));
            }
        });
        let v = &self.view;
        if self.dirty || v0 != (v.sort, v.desc, v.kind, v.fav, v.find.clone()) {
            self.view.apply(&self.items, &mut self.shown);
            self.dirty = false;
        }
        if self.items.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(match self.info.conn {
                    Conn::On if self.info.busy => "The camera is capturing.",
                    Conn::On => "No files on the camera.",
                    _ => "Connect the camera to see its files.",
                })
            });
            return None;
        }
        ui.add_space(4.0);
        let (tw, gap) = (200.0, 10.0);
        let th = tw / 2.0 + 40.0;
        let cols = ((ui.available_width() + gap) / (tw + gap)).floor().max(1.0) as usize;
        let rows = self.shown.len().div_ceil(cols);
        let mods = ui.input(|i| i.modifiers);
        let (mut open, mut want, mut click) = (None, vec![], None);
        self.frame += 1;
        ui.spacing_mut().item_spacing = egui::vec2(gap, gap);
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show_rows(ui, th, rows, |ui, rr| {
                for r in rr {
                    ui.horizontal(|ui| {
                        for k in r * cols..((r + 1) * cols).min(self.shown.len()) {
                            let it = &self.items[self.shown[k]];
                            let (rect, resp) =
                                ui.allocate_exact_size(egui::vec2(tw, th), egui::Sense::click());
                            self.tile(ui, rect, it);
                            if !self.tex.contains_key(&it.key) {
                                want.push(it.key.clone());
                            }
                            let resp = resp.on_hover_text(
                                it.uris
                                    .iter()
                                    .map(|u| album::name(u))
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            );
                            if resp.double_clicked()
                                && it.kind == Kind::Video
                                && self.have.contains(&it.key)
                            {
                                open = Some(self.set.dest.join(album::name(&it.uris[0])));
                            } else if resp.clicked() {
                                click = Some(k);
                            }
                        }
                    });
                }
            });
        if let Some(k) = click {
            let key = self.items[self.shown[k]].key.clone();
            match (mods.command, mods.shift, self.anchor) {
                (true, ..) => {
                    if !self.sel.remove(&key) {
                        self.sel.insert(key);
                    }
                }
                (_, true, Some(a)) => {
                    let (a, b) = (a.min(k), a.max(k).min(self.shown.len() - 1));
                    self.sel
                        .extend((a..=b).map(|j| self.items[self.shown[j]].key.clone()));
                }
                _ => self.sel = HashSet::from([key]),
            }
            if !mods.shift {
                self.anchor = Some(k);
            }
        }
        if self.tex.len() > TEX {
            // least recently drawn first, never what is on screen
            let f = self.frame;
            let mut s: Vec<(u64, String)> = self
                .tex
                .iter()
                .filter(|t| t.1.1.get() < f)
                .map(|(k, t)| (t.1.get(), k.clone()))
                .collect();
            s.sort_unstable();
            for (_, k) in s.iter().take(self.tex.len() - TEX * 3 / 4) {
                self.tex.remove(k);
            }
        }
        if !want.is_empty() && want != self.wanted {
            let _ = self.w.tx.send(Cmd::Want(want.clone()));
            self.wanted = want;
        }
        open
    }

    fn tile(&self, ui: &egui::Ui, rect: egui::Rect, it: &Item) {
        let p = ui.painter_at(rect.expand(3.0));
        let v = ui.visuals();
        let img = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), rect.width() / 2.0));
        let t = self.tex.get(&it.key).map(|t| {
            t.1.set(self.frame);
            &t.0
        });
        match t {
            Some(Some(t)) => {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                p.image(t.id(), img, uv, egui::Color32::WHITE);
            }
            t => {
                p.rect_filled(img, 4.0, v.faint_bg_color);
                let s = if t.is_some() { "no thumbnail" } else { "…" };
                p.text(
                    img.center(),
                    egui::Align2::CENTER_CENTER,
                    s,
                    egui::FontId::proportional(13.0),
                    v.weak_text_color(),
                );
            }
        }
        let small = egui::FontId::proportional(12.0);
        let badge = |at: egui::Pos2, a: egui::Align2, s: &str, c: egui::Color32| {
            let r = p.text(at, a, s, small.clone(), c);
            p.rect_filled(r.expand(2.0), 3.0, egui::Color32::from_black_alpha(150));
            p.text(at, a, s, small.clone(), c);
        };
        let pad = egui::vec2(5.0, 4.0);
        if it.kind == Kind::Video && it.ms > 0 {
            badge(
                img.left_bottom() + egui::vec2(pad.x, -pad.y),
                egui::Align2::LEFT_BOTTOM,
                &dur(it.ms),
                egui::Color32::WHITE,
            );
        }
        if it.fav {
            badge(
                img.right_top() + egui::vec2(-pad.x, pad.y),
                egui::Align2::RIGHT_TOP,
                "★",
                egui::Color32::GOLD,
            );
        }
        if self.have.contains(&it.key) {
            badge(
                img.left_top() + pad,
                egui::Align2::LEFT_TOP,
                "✔ downloaded",
                egui::Color32::LIGHT_GREEN,
            );
        }
        let mut y = img.bottom() + 4.0;
        for (s, c) in [
            (album::date(it.date), v.strong_text_color()),
            (
                [
                    (it.dim[0] > 0).then(|| format!("{} × {}", it.dim[0], it.dim[1])),
                    (it.size > 0).then(|| mb(it.size)),
                    (it.uris.len() > 1).then(|| format!("{} files", it.uris.len())),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" · "),
                v.text_color(),
            ),
        ] {
            y = p
                .text(
                    egui::pos2(rect.left() + 2.0, y),
                    egui::Align2::LEFT_TOP,
                    s,
                    small.clone(),
                    c,
                )
                .bottom()
                + 2.0;
        }
        if self.sel.contains(&it.key) {
            p.rect_stroke(
                rect.expand(1.0),
                4.0,
                egui::Stroke::new(2.5, v.selection.stroke.color),
                egui::StrokeKind::Outside,
            );
        }
    }

    fn downloads(&mut self, ui: &mut egui::Ui) -> Option<PathBuf> {
        let mut open = None;
        // wrapped: a row wider than the panel widens it for everything below, the bars ran off the window
        ui.horizontal_wrapped(|ui| {
            ui.label("Destination");
            ui.monospace(self.set.dest.display().to_string());
            if ui.button("Change…").clicked() && self.dlg.is_none() {
                let (tx, rx) = mpsc::channel();
                let (ctx, d) = (self.ctx.clone(), self.set.dest.clone());
                // the portal dialog blocks: off the UI thread
                thread::spawn(move || {
                    let _ = tx.send(rfd::FileDialog::new().set_directory(d).pick_folder());
                    ctx.request_repaint();
                });
                self.dlg = Some(rx);
            }
            if ui.button("Clear finished").clicked() {
                let mut q = self.q.lock().unwrap();
                q.clear_done();
                q.save();
            }
        });
        ui.separator();
        let mut q = self.q.lock().unwrap();
        if q.v.is_empty() {
            ui.label("Nothing queued. Select files in the album and press Download.");
            return None;
        }
        let (mut act, on) = (None, self.info.conn == Conn::On);
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                // the bar comes last: only the last grid column gets the width that is left, so a narrow window clips nothing
                egui::Grid::new("dls")
                    .num_columns(3)
                    .striped(true)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        for (i, d) in q.v.iter().enumerate() {
                            ui.label(d.name());
                            let (n, t) = (d.prog.done.load(Relaxed), d.prog.total.load(Relaxed));
                            let f = if t > 0 { n as f32 / t as f32 } else { 0.0 };
                            let mut s = match d.st {
                                St::Running if t > 0 => {
                                    format!("{} of {} · {:.1} MB/s", mb(n), mb(t), d.rate / 1e6)
                                }
                                St::Running => "starting…".into(),
                                St::Queued if on => "queued".into(),
                                St::Queued => "queued: waiting for the camera".into(),
                                St::Paused if t > 0 => format!("stopped at {} of {}", mb(n), mb(t)),
                                St::Paused => "stopped".into(),
                                St::Done => "done".into(),
                                St::Failed => format!("failed: {}", d.err),
                            };
                            if !d.err.is_empty() && d.st != St::Failed {
                                s = format!("{s} · {}", d.err)
                            }
                            ui.horizontal(|ui| {
                                match d.st {
                                    St::Running | St::Queued => {
                                        if ui.button("Cancel").clicked() {
                                            act = Some((i, 0))
                                        }
                                    }
                                    St::Paused | St::Failed => {
                                        if ui.button("Resume").clicked() {
                                            act = Some((i, 1))
                                        }
                                    }
                                    St::Done => {
                                        let v = matches!(
                                            album::name(&d.uri).rsplit('.').next(),
                                            Some("insv" | "lrv" | "mp4")
                                        );
                                        if ui.add_enabled(v, egui::Button::new("Open")).clicked() {
                                            open = Some(d.dst.clone())
                                        }
                                    }
                                }
                                if d.st != St::Running && ui.button("Remove").clicked() {
                                    act = Some((i, 2))
                                }
                            });
                            let f = if d.st == St::Done { 1.0 } else { f };
                            ui.add(egui::ProgressBar::new(f).text(s));
                            ui.end_row();
                        }
                    });
            });
        if let Some((i, a)) = act {
            match a {
                0 => q.cancel(i),
                1 => q.resume(i),
                _ => q.remove(i),
            }
            q.save();
        }
        open
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        let i = &self.info;
        let Some(id) = i.id.as_ref().filter(|_| i.conn == Conn::On) else {
            ui.label("Connect the camera to see its settings.");
            return;
        };
        egui::Grid::new("cam info")
            .num_columns(2)
            .striped(true)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(k);
                    ui.label(v);
                    ui.end_row();
                };
                row("Model", id.model.clone());
                row("Firmware", id.firmware.clone());
                row("Serial number", id.serial.clone());
                row(
                    "Battery",
                    i.battery.as_ref().map_or("unknown".into(), battery),
                );
                row(
                    "Storage",
                    i.storage.as_ref().map_or("unknown".into(), storage),
                );
                row(
                    "Clock",
                    i.skew.map_or("unknown".into(), |s| match s {
                        -2..=2 => "in sync with this computer".into(),
                        s => format!("{s:+} s from this computer"),
                    }),
                );
            });
        ui.add_space(8.0);
        if ui
            .button("Sync time…")
            .on_hover_text("Set the camera clock and time zone to this computer's")
            .clicked()
        {
            self.ask = Some(Ask::Time);
        }
        ui.add_space(12.0);
        ui.heading("Wi-Fi");
        match &i.wifi {
            Some((ssid, pw)) => {
                egui::Grid::new("wifi")
                    .num_columns(3)
                    .spacing([16.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Name");
                        ui.monospace(ssid);
                        ui.end_row();
                        ui.label("Password");
                        ui.monospace(if self.pw {
                            pw.clone()
                        } else {
                            "•".repeat(pw.chars().count())
                        });
                        ui.toggle_value(&mut self.pw, "Show");
                        ui.end_row();
                    });
            }
            None => {
                ui.label("The camera did not report its Wi-Fi.");
            }
        }
        ui.add_space(12.0);
        ui.heading("SD card");
        if ui.button("Format SD card…").clicked() {
            self.typed.clear();
            self.ask = Some(Ask::Format);
        }
    }

    /// The confirmation of a camera change.
    fn dialog(&mut self, ctx: &egui::Context) {
        let Some(a) = &self.ask else { return };
        let mut go = None;
        let r = egui::Modal::new(egui::Id::unique("cam ask")).show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.heading(match a {
                Ask::Delete(..) => "Delete",
                Ask::Fav(..) => "Favourites",
                Ask::Format => "Format SD card",
                Ask::Time => "Sync time",
            });
            let (ok, ready) = match a {
                Ask::Delete(u, n, not) => {
                    ui.label(format!(
                        "Delete {n} item{} ({} files) from the camera? This cannot be undone.",
                        if *n == 1 { "" } else { "s" },
                        u.len()
                    ));
                    if *not > 0 {
                        ui.colored_label(ui.visuals().warn_fg_color, format!("{not} of them are not downloaded."));
                    }
                    ("Delete", true)
                }
                Ask::Fav(v) => {
                    let on = v.first().is_some_and(|f| f.1);
                    ui.label(format!(
                        "{} {} item{} as favourite on the camera?",
                        if on { "Mark" } else { "Unmark" },
                        v.len(),
                        if v.len() == 1 { "" } else { "s" }
                    ));
                    (if on { "Mark" } else { "Unmark" }, true)
                }
                Ask::Format => {
                    ui.label("Format the camera's SD card? Everything on it is erased, including files not downloaded.");
                    ui.label("Type FORMAT to confirm.");
                    ui.add(egui::TextEdit::singleline(&mut self.typed).desired_width(120.0));
                    ("Format", self.typed == "FORMAT")
                }
                Ask::Time => {
                    let (_, _, off, tz, s) = clock();
                    ui.label(format!(
                        "Set the camera clock to this computer's: {s}, {tz} (UTC{:+03}:{:02})?",
                        off / 3600,
                        off.abs() / 60 % 60
                    ));
                    ("Set", true)
                }
            };
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(ready, egui::Button::new(ok)).clicked() {
                    go = Some(true);
                }
                if ui.button("Cancel").clicked() {
                    go = Some(false);
                }
            });
        });
        if r.should_close() && go.is_none() {
            go = Some(false);
        }
        if let Some(g) = go {
            let a = self.ask.take().unwrap();
            if g {
                let c = match a {
                    Ask::Delete(u, ..) => Cmd::Delete(u),
                    Ask::Fav(v) => Cmd::Fav(v),
                    Ask::Format => Cmd::Format,
                    Ask::Time => Cmd::Time,
                };
                let _ = self.w.tx.send(c);
                self.sel.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests;
