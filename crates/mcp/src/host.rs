//! The headless host: the camera worker of `cam` (USB followed, never connected on hotplug, never the camera clock),
//! the download queue and the jobs, mounted cards watched, and the state the tools read. A pump thread owns the worker
//! and turns its reports into [`State`], waking whoever waits on it; tools send the worker commands and wait with
//! timeouts. The camera is released after [`IDLE`] without a tool call (not while a call runs, nor a download that runs
//! or can start), and on [`Host::shutdown`] (stdin's end, a signal: the process side calls it).
//!
//! The queue and the jobs are kept in the config dir (`mcp-downloads`, cam's queue file, and `mcp-jobs`, see [`job`]),
//! the app's own queue apart, under a lock (`mcp-downloads.lock`) a host holds from its start until its worker is gone
//! and both are saved the last time (a worker a camera call holds up past [`ENDING`] is left behind); a host that
//! starts while the one before it ends waits up to [`HANDOVER`] for it. What it takes is checked: downloads into the
//! agent folder, from a camera or a card. So a later host goes on with what an earlier one left: the downloads that
//! waited or ran (`camera_release` and the shutdown leave them so) run once the camera is connected (a card import at
//! once), the jobs keep their handles, and a cancelled download, its `.part` kept, resumes under its name when its item
//! is downloaded again. A host whose lock stays another's (another endpoint's host) keeps its queue in memory, and says
//! so in the log.
//!
//! The enabled tiers are re-read every 2 s; a change tells every client (`notifications/tools/list_changed`, on
//! 2026-07-28 to its `subscriptions/listen` streams). Each connection is one rmcp service over the shared host
//! ([`Host::serve`]), its output scrubbed ([`crate::scrub::Writer`]).

use crate::{
    config::{self, Source, Tiers},
    job::{self, Jobs},
    log::Log,
    paths,
    scrub::{self, Secrets},
    server::Conn,
    sim::{self, Sim},
};
use cam::{
    album::{self, Cache, Item},
    camera::{Cmd, Conn as Link, Info, Taps, Up, Worker},
    disk, dl,
};
use link::Plug;
use rmcp::ServiceExt;
use std::{
    collections::VecDeque,
    fs::File,
    io,
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
        mpsc::{RecvTimeoutError, Sender},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::watch,
};
use tokio_util::sync::CancellationToken;

/// How long the camera stays connected without a tool call.
pub const IDLE: Duration = Duration::from_secs(300);

/// How often the tiers file is looked at.
pub const RELOAD: Duration = Duration::from_secs(2);

/// How long a host waits for the one before it (ending: it saves the queue and the jobs once its worker is gone) to
/// let go of them.
pub const HANDOVER: Duration = Duration::from_secs(10);

/// How long a host's end waits for its worker (a camera call can hold it up for 30 s) before it saves the queue and
/// the jobs and lets go of them without it: well within [`HANDOVER`].
pub const ENDING: Duration = Duration::from_secs(5);

/// Thumbnails kept for `album_thumbnail` (320 × 160 RGBA, 200 kB each).
const THUMBS: usize = 32;

/// Worker notes kept for `camera_status` and the connect's outcome.
const NOTES: usize = 64;

/// How a host is made.
#[derive(Clone, Debug)]
pub struct Opts {
    /// `--sim`: the simulated camera and temp dirs instead of USB and the user's folders
    pub sim: Option<sim::Opts>,
    /// narrows the enabled set for every connection (`x5desk mcp` leaves it unset: its `--tiers` narrow only its own
    /// client, through [`Host::serve`])
    pub narrow: Option<Tiers>,
    /// `--allow-dir`s: more read roots; each must exist
    pub allow: Vec<PathBuf>,
    /// where the tiers come from; none: `<config>/x5desk/mcp`, under `--sim` or `dir` `X5DESK_MCP_TIERS` (a comma
    /// list), else all but `delete`
    pub tiers: Option<Source>,
    /// [`IDLE`]
    pub idle: Duration,
    /// a real camera, but the agent folder (`agent`) and the log under this dir instead of the user's, no thumbnail
    /// cache, nothing of the user's config read, a Wi-Fi client id of its own, the tiers as under `--sim` (tests on
    /// a camera, `X5DESK_TEST`)
    pub dir: Option<PathBuf>,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            sim: None,
            narrow: None,
            allow: vec![],
            tiers: None,
            idle: IDLE,
            dir: None,
        }
    }
}

/// A thumbnail's width, height and RGBA; `None`: the item has none.
pub type Thumb = Option<(u32, u32, Vec<u8>)>;

/// What the worker reported, for the tools.
#[derive(Default)]
pub struct State {
    /// the camera worker's newest `Info`
    pub info: Info,
    /// the album of `info.src` (a card) or of the camera; none until listed
    pub album: Option<Vec<Item>>,
    /// the newest thumbnails by item key
    pub thumbs: VecDeque<(String, Thumb)>,
    /// the worker's notes: number, text, whether an error
    pub notes: VecDeque<(u64, String, bool)>,
    /// the newest note's number
    pub seq: u64,
    /// the idle release was sent for this connection
    released: bool,
}

impl State {
    /// The thumbnail of item `k` if it is among the newest.
    pub fn thumb(&self, k: &str) -> Option<&Thumb> {
        self.thumbs.iter().find(|t| t.0 == k).map(|t| &t.1)
    }

    /// Notes after number `n`.
    pub fn since(&self, n: u64) -> impl Iterator<Item = &(u64, String, bool)> {
        self.notes.iter().filter(move |x| x.0 > n)
    }
}

/// The part of the host its threads and tasks share.
pub(crate) struct Shared {
    pub id: String,
    pub tx: Mutex<Sender<Cmd>>,
    pub st: Mutex<State>,
    /// bumped after every batch of reports
    pub ver: watch::Sender<u64>,
    pub q: dl::Shared,
    /// locked before `q` where both are
    pub jobs: Mutex<Jobs>,
    /// the lock of the kept queue and jobs, let go once they are saved the last time
    held: Mutex<Option<File>>,
    pub secrets: Secrets,
    pub log: Log,
    pub agent: PathBuf,
    /// the `--allow-dir`s, canonical
    pub allow: Vec<PathBuf>,
    /// the app's download folders are read roots ([`paths::downloads`], looked up at each use); not under `--sim` or
    /// `dir`
    pub downloads: bool,
    pub tiers: watch::Sender<Tiers>,
    pub ct: CancellationToken,
    pub stop: AtomicBool,
    /// tool calls running, the end of the last one
    pub calls: AtomicUsize,
    pub last: Mutex<Instant>,
    pub idle: Duration,
    /// the Wi-Fi client id the camera authorizes, looked up at the first Wi-Fi connect
    wifi: OnceLock<String>,
    pub sim: bool,
    pub conns: AtomicUsize,
}

/// A tool call in progress: the camera is not idle meanwhile.
pub(crate) struct Call<'a>(&'a Shared);

impl Drop for Call<'_> {
    fn drop(&mut self) {
        *self.0.last.lock().unwrap() = Instant::now();
        self.0.calls.fetch_sub(1, Relaxed);
    }
}

impl Shared {
    /// Sends the worker a command.
    pub fn cmd(&self, c: Cmd) {
        let _ = self.tx.lock().unwrap().send(c);
    }

    /// The Wi-Fi client id (a secret from then on).
    pub fn wifi(&self) -> String {
        let id = self
            .wifi
            .get_or_init(|| if self.sim { sim::ID.into() } else { wifi_id() });
        self.secrets.add(id);
        id.clone()
    }

    /// Marks a tool call running until the guard drops.
    pub fn call(&self) -> Call<'_> {
        self.calls.fetch_add(1, Relaxed);
        *self.last.lock().unwrap() = Instant::now();
        Call(self)
    }

    /// Writes the queue and the jobs after a change: the queue first, so a kept job has its files in it.
    pub fn save(&self, jobs: &mut Jobs, q: &mut dl::Queue) {
        q.save();
        if let Err(e) = jobs.save() {
            self.log.line(&format!("the jobs file: {e}"))
        }
    }

    /// Whether a download runs or can start ([`busy`]).
    pub fn busy(&self) -> bool {
        busy(&self.q.lock().unwrap())
    }

    /// Waits up to `secs` for `f` to find what it wants in the state, looking again after every report. None on
    /// timeout or shutdown (`f`'s last answer).
    pub async fn until<T>(&self, secs: f64, mut f: impl FnMut(&State) -> Option<T>) -> Option<T> {
        let mut rx = self.ver.subscribe();
        let end = tokio::time::Instant::now() + Duration::from_secs_f64(secs);
        loop {
            if let Some(t) = f(&self.st.lock().unwrap()) {
                return Some(t);
            }
            if self.ct.is_cancelled() {
                return None;
            }
            tokio::select! {
                r = tokio::time::timeout_at(end, rx.changed()) => {
                    if !matches!(r, Ok(Ok(()))) {
                        return f(&self.st.lock().unwrap());
                    }
                }
                _ = self.ct.cancelled() => return None,
            }
        }
    }

    /// All read roots now, canonicalised: the agent folder, the `--allow-dir`s, the app's download folders (where it
    /// downloads now), the mounted cards.
    pub fn read_roots(&self) -> Vec<PathBuf> {
        let cards: Vec<PathBuf> = (self.st.lock().unwrap().info.disk.iter())
            .map(|v| v.root.clone())
            .collect();
        let dl = match self.downloads {
            true => paths::downloads(),
            false => vec![],
        };
        paths::canon(
            [self.agent.clone()]
                .into_iter()
                .chain(self.allow.iter().cloned())
                .chain(dl)
                .chain(cards),
        )
    }

    fn apply(&self, u: Up) {
        let mut s = self.st.lock().unwrap();
        match u {
            Up::Info(i) => {
                if let Some((_, pw)) = &i.wifi {
                    self.secrets.add(pw)
                }
                // a new source, or the camera's session come or gone: the album is listed again
                if i.src != s.info.src || (i.src.is_none() && i.conn != s.info.conn) {
                    s.album = None;
                    s.thumbs.clear();
                }
                if i.conn != s.info.conn {
                    s.released = false
                }
                s.info = i
            }
            Up::Album(v) => s.album = Some(v),
            Up::Thumb(k, t) => {
                s.thumbs.retain(|x| x.0 != k);
                if s.thumbs.len() >= THUMBS {
                    s.thumbs.pop_front();
                }
                s.thumbs.push_back((k, t))
            }
            Up::Note(n, e) => {
                self.log.line(&format!("camera: {n}"));
                s.seq += 1;
                let k = s.seq;
                if s.notes.len() >= NOTES {
                    s.notes.pop_front();
                }
                s.notes.push_back((k, n, e))
            }
            Up::Done(p) => self.log.line(&format!("downloaded {}", p.display())),
            Up::Stats(_) => {}
        }
    }

    fn bump(&self) {
        self.ver.send_modify(|v| *v += 1)
    }

    /// Releases the camera after [`Shared::idle`] without a call, unless a call runs or a download runs or can start
    /// ([`busy`]).
    fn idle_check(&self) {
        let on = {
            let s = self.st.lock().unwrap();
            s.info.conn == Link::On && !s.released
        };
        if !on
            || self.calls.load(Relaxed) > 0
            || self.last.lock().unwrap().elapsed() < self.idle
            || self.busy()
        {
            return;
        }
        self.st.lock().unwrap().released = true;
        self.log.line(&format!(
            "released the camera after {} s without a call",
            self.idle.as_secs()
        ));
        self.cmd(Cmd::Disconnect)
    }
}

/// Whether a download of `q` runs or can start: a queued card import whose file is there, any queued camera
/// download (the idle check runs only while the camera is on). An import whose card went never starts, so it does not
/// keep the camera.
fn busy(q: &dl::Queue) -> bool {
    q.v.iter().any(|d| match d.st {
        dl::St::Running => true,
        dl::St::Queued => dl::local(&d.uri).is_none_or(Path::exists),
        _ => false,
    })
}

/// The pump: the worker's reports into the state until the host stops, then the session closed and the worker gone
/// (its drop quits it cleanly; a running download keeps its `.part`).
fn pump(sh: Arc<Shared>, w: Worker) {
    while !sh.stop.load(Relaxed) {
        match w.rx.recv_timeout(Duration::from_millis(250)) {
            Ok(u) => {
                sh.apply(u);
                while let Ok(u) = w.rx.try_recv() {
                    sh.apply(u)
                }
                sh.bump()
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        sh.idle_check()
    }
    let _ = w.tx.send(Cmd::Disconnect);
    // its end puts the running download back in the queue, stopped, and saves the queue; one held up by a camera call
    // past ENDING is left to the process's end, and the queue handed on without it
    let (tx, rx) = std::sync::mpsc::channel();
    let r = std::thread::Builder::new()
        .name("x5-mcp-end".into())
        .spawn(move || {
            drop(w);
            let _ = tx.send(());
        });
    let stuck = r.is_ok() && rx.recv_timeout(ENDING) == Err(RecvTimeoutError::Timeout);
    {
        let mut j = sh.jobs.lock().unwrap();
        let mut q = sh.q.lock().unwrap();
        if stuck {
            sh.log.line(&format!(
                "the camera worker did not end within {} s: its queue goes to the next host without it",
                ENDING.as_secs()
            ));
            q.quit()
        }
        if let Err(e) = j.close() {
            sh.log.line(&format!("the jobs file: {e}"))
        }
        q.close();
        // a worker that comes back starts nothing more
        if stuck {
            for d in q.v.iter_mut().filter(|d| d.st == dl::St::Queued) {
                d.st = dl::St::Paused
            }
        }
    }
    // the next host takes them from here
    drop(sh.held.lock().unwrap().take());
    sh.log.line("host stopped");
    sh.bump()
}

/// The lock file `p` (owner-only, its dir too; not through a symlink on unix) taken, trying for up to `wait`; none
/// if it stays another's.
async fn lock_wait(p: &Path, wait: Duration) -> io::Result<Option<File>> {
    if let Some(d) = p.parent() {
        cam::dirs::private(d)?
    }
    let f = config::lock_file(p)?;
    let end = tokio::time::Instant::now() + wait;
    loop {
        match f.try_lock() {
            Ok(()) => return Ok(Some(f)),
            Err(std::fs::TryLockError::WouldBlock) if tokio::time::Instant::now() >= end => {
                return Ok(None);
            }
            Err(std::fs::TryLockError::WouldBlock) => {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            Err(std::fs::TryLockError::Error(e)) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(std::fs::TryLockError::Error(e)) => return Err(e),
        }
    }
}

/// The queue and the jobs kept in `dir`, for the agent folder `agent`, with their lock (waited for up to
/// [`HANDOVER`]) and what to log. Kept downloads and jobs a host of that folder cannot have made are left out (a file
/// edited by hand, or forged): a file outside the folder, one whose name is no safe file name, a source that is
/// neither a camera URI nor a card's `DCIM`. A download whose file is there without its `.part` is done (it finished
/// uncollected). Then what earlier hosts left is pruned ([`Jobs::prune`]) and a finished file's size read for the jobs
/// that show it. In memory without a dir, while the lock stays another's, or when a file
/// is there but cannot be read (it is not taken over); a queue file that is no UTF-8 is moved to
/// `mcp-downloads.bad`.
async fn kept(dir: Option<&Path>, agent: &Path) -> (dl::Queue, Jobs, Option<File>, Vec<String>) {
    let mem = |said: Vec<String>| (dl::Queue::load(None), Jobs::new(String::new()), None, said);
    let Some(d) = dir else { return mem(vec![]) };
    let lf = d.join("mcp-downloads.lock");
    let l = match lock_wait(&lf, HANDOVER).await {
        Ok(Some(l)) => l,
        Ok(None) => {
            return mem(vec![format!(
                "{}: another host holds the download queue, so this host's stays in memory",
                lf.display()
            )]);
        }
        Err(e) => {
            return mem(vec![format!(
                "{}: {e}; the download queue stays in memory",
                lf.display()
            )]);
        }
    };
    let (qf, jf) = (d.join("mcp-downloads"), d.join("mcp-jobs"));
    let lost = |f: &Path, e: &dyn std::fmt::Display| {
        vec![format!(
            "{}: {e}; the download queue stays in memory",
            f.display()
        )]
    };
    let mut said = vec![];
    match std::fs::read(&qf) {
        Ok(b) if std::str::from_utf8(&b).is_err() => {
            let bad = d.join("mcp-downloads.bad");
            if let Err(e) = std::fs::rename(&qf, &bad) {
                return mem(lost(&qf, &e));
            }
            said.push(format!(
                "{}: no UTF-8, moved to {}",
                qf.display(),
                bad.display()
            ))
        }
        Err(e) if e.kind() != io::ErrorKind::NotFound => return mem(lost(&qf, &e)),
        _ => {}
    }
    let (mut jobs, bad) = match Jobs::load(String::new(), jf.clone()) {
        Ok(j) => j,
        Err(e) => return mem(lost(&jf, &e)),
    };
    if bad > 0 {
        said.push(format!(
            "{}: {bad} lines that do not parse left out",
            jf.display()
        ))
    }
    let mut q = dl::Queue::load(Some(qf));
    let roots = [
        agent.to_path_buf(),
        std::fs::canonicalize(agent).unwrap_or_default(),
    ];
    let inside = |p: &Path| {
        roots
            .iter()
            .any(|r| !r.as_os_str().is_empty() && p.starts_with(r))
            && !(p.components()).any(|c| matches!(c, Component::ParentDir | Component::CurDir))
            && p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(album::safe)
    };
    let from = |u: &str| match dl::local(u) {
        Some(s) => {
            s.is_absolute()
                && s.components().any(|c| c.as_os_str() == "DCIM")
                && !(s.components()).any(|c| matches!(c, Component::ParentDir | Component::CurDir))
        }
        None => u.starts_with('/'),
    } && album::safe(album::name(u));
    let (nq, nj) = (q.v.len(), jobs.v.len());
    q.v.retain(|x| inside(&x.dst) && from(&x.uri));
    jobs.v
        .retain(|j| j.files.iter().all(|f| inside(&f.dst) && from(&f.uri)));
    if q.v.len() < nq || jobs.v.len() < nj {
        said.push(format!(
            "left out, not from a camera or card into {}: {} downloads, {} jobs",
            agent.display(),
            nq - q.v.len(),
            nj - jobs.v.len()
        ))
    }
    // a download that finished but was not collected (its host's worker held up at its end), or a file there
    // meanwhile: done, never fetched over it
    for x in q.v.iter_mut().filter(|x| x.st != dl::St::Done) {
        if x.dst.is_file() && !job::part_of(&x.dst).exists() {
            x.st = dl::St::Done
        }
    }
    jobs.prune(&mut q);
    for x in q.v.iter().filter(|x| x.st == dl::St::Done) {
        if let Ok(m) = std::fs::metadata(&x.dst) {
            x.prog.total.store(m.len(), Relaxed);
            x.prog.done.store(m.len(), Relaxed)
        }
    }
    q.save();
    if let Err(e) = jobs.save() {
        said.push(format!("the jobs file: {e}"))
    }
    if !jobs.v.is_empty() || !q.v.is_empty() {
        said.push(format!(
            "kept from earlier hosts: {} jobs, {} downloads",
            jobs.v.len(),
            q.v.len()
        ))
    }
    (q, jobs, Some(l), said)
}

/// The `id=` of the app's `<config>/x5desk/settings` (the client id the camera remembers once its user allowed this
/// computer), and its thumbnail cache limit.
fn app_settings() -> (Option<String>, Option<u64>) {
    let f = cam::dirs::xdg("XDG_CONFIG_HOME", ".config").map(|d| d.join("x5desk").join("settings"));
    let t = f
        .and_then(|f| std::fs::read_to_string(f).ok())
        .unwrap_or_default();
    let v = |k: &str| t.lines().find_map(|l| l.strip_prefix(k)).map(str::trim);
    (
        v("id=").filter(|s| !s.is_empty()).map(String::from),
        v("cache=").and_then(|s| s.parse().ok()),
    )
}

/// A random (version 4) UUID.
fn uuid() -> String {
    let (a, b) = (job::random(), job::random());
    let mut x = [0u8; 16];
    x[..8].copy_from_slice(&a.to_le_bytes());
    x[8..].copy_from_slice(&b.to_le_bytes());
    (x[6], x[8]) = (x[6] & 0x0f | 0x40, x[8] & 0x3f | 0x80);
    let h: String = x.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

/// The Wi-Fi client id: the app's (the camera already knows it if its user allowed x5desk), else one of the agents'
/// own, kept as `id` in `<config>/x5desk/mcp` (looked up or made under the file's lock, so hosts agree on one).
fn wifi_id() -> String {
    if let (Some(id), _) = app_settings() {
        return id;
    }
    let Some(f) = config::file() else {
        return uuid();
    };
    config::get_or_set(&f, "id", uuid).unwrap_or_else(|e| {
        eprintln!("x5desk mcp: {}: {e}", f.display());
        uuid()
    })
}

/// A host: the camera worker, the state, the jobs, its connections.
pub struct Host {
    pub(crate) sh: Arc<Shared>,
    pump: Mutex<Option<JoinHandle<()>>>,
    sim: Option<Sim>,
}

impl Host {
    /// Takes the kept queue and jobs (waiting for an ending host to let go of them, [`HANDOVER`]), starts the
    /// worker, its pump and the tiers' watch (a task: call it inside a tokio runtime with time enabled). Fails when an
    /// `--allow-dir` is missing, `X5DESK_MCP_TIERS` does not parse, there is no video folder, or the `--sim` dirs
    /// cannot be made.
    pub async fn new(o: Opts) -> Result<Arc<Host>, String> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| "Host::new needs a tokio runtime".to_string())?;
        let secrets = Secrets::default();
        let mut allow = vec![];
        for d in &o.allow {
            let c = std::fs::canonicalize(d)
                .map_err(|e| format!("--allow-dir {}: {e}", d.display()))?;
            if !c.is_dir() {
                return Err(format!("--allow-dir {}: not a directory", d.display()));
            }
            allow.push(c)
        }
        let sim = match &o.sim {
            Some(so) => Some(Sim::new(so).map_err(|e| format!("--sim: {e}"))?),
            None => None,
        };
        let (connect, cache, watch, agent, logp, downloads) = match &sim {
            Some(s) => (
                s.connect(),
                Cache::at(None),
                disk::Watch::hook(vec![s.card.clone()]),
                s.agent.clone(),
                Some(s.log.clone()),
                false,
            ),
            None if o.dir.is_some() => {
                let d = o.dir.as_ref().unwrap();
                (
                    Box::new(link::Camera::usb) as cam::camera::Connect,
                    Cache::at(None),
                    disk::Watch::new(vec![]),
                    d.join("agent"),
                    Some(d.join("mcp.log")),
                    false,
                )
            }
            None => {
                let mut c = Cache::new();
                c.cap = app_settings().1.unwrap_or(c.cap);
                let agent = paths::agent().ok_or("no video folder (HOME not set)")?;
                (
                    Box::new(link::Camera::usb) as cam::camera::Connect,
                    c,
                    disk::Watch::new(vec![]),
                    agent,
                    crate::log::file(),
                    true,
                )
            }
        };
        // the simulated camera's access-point password is not among them: the host learns it from the camera
        if sim.is_some() {
            secrets.add(sim::MARKER);
            secrets.add(sim::ID)
        }
        let src = match (&o.tiers, &sim) {
            (Some(s), _) => s.clone(),
            (None, _) if sim.is_some() || o.dir.is_some() => {
                match std::env::var("X5DESK_MCP_TIERS") {
                    Ok(v) => Source::Fixed(
                        Tiers::parse(&v).map_err(|e| format!("X5DESK_MCP_TIERS: {e}"))?,
                    ),
                    Err(_) => Source::Fixed(Tiers::SIM),
                }
            }
            (None, _) => config::file().map_or(Source::Fixed(Tiers::NONE), Source::File),
        };
        let mut tw = config::Watch::new(src, o.narrow);
        // the queue and the jobs are kept in the config dir; under `--sim` and `dir` in theirs
        let state = match (&sim, &o.dir) {
            (Some(s), _) => Some(s.dir.clone()),
            (None, Some(d)) => Some(d.clone()),
            (None, None) => config::file().and_then(|f| f.parent().map(Path::to_path_buf)),
        };
        let (q, mut jobs, held, said) = kept(state.as_deref(), &agent).await;
        // a handle names one job: an id none of the kept jobs has
        let mut id = job::host_id();
        while (jobs.v.iter()).any(|j| j.h.rsplit_once('-').is_some_and(|(h, _)| h == id)) {
            id = job::host_id()
        }
        jobs.host = id.clone();
        let q = Arc::new(Mutex::new(q));
        let auto = Arc::new(AtomicBool::new(false));
        // the bus is followed (the camera's plug state) but never connected on its own; the simulated camera is
        // "plugged in" instead
        let w = Worker::spawn(
            connect,
            q.clone(),
            cache,
            auto,
            sim.is_none(),
            Taps::default(),
            || {},
        );
        let _ = w.tx.send(Cmd::Watch(watch));
        if sim.is_some() {
            let _ = w.tx.send(Cmd::Plug(Plug::App));
        }
        let log = Log::new(logp, secrets.clone());
        log.line(&format!(
            "host {id} started{}, tiers {}",
            if sim.is_some() {
                " (simulated camera)"
            } else {
                ""
            },
            tw.tiers()
        ));
        for l in said {
            log.line(&l)
        }
        let sh = Arc::new(Shared {
            jobs: Mutex::new(jobs),
            held: Mutex::new(held),
            id,
            tx: Mutex::new(w.tx.clone()),
            st: Mutex::default(),
            ver: watch::Sender::new(0),
            q,
            secrets,
            log,
            agent,
            allow,
            downloads,
            tiers: watch::Sender::new(tw.tiers()),
            ct: CancellationToken::new(),
            stop: AtomicBool::new(false),
            calls: AtomicUsize::new(0),
            last: Mutex::new(Instant::now()),
            idle: o.idle,
            wifi: match o.dir {
                Some(_) => OnceLock::from(uuid()),
                None => OnceLock::new(),
            },
            sim: sim.is_some(),
            conns: AtomicUsize::new(0),
        });
        let s2 = sh.clone();
        let th = std::thread::Builder::new()
            .name("x5-mcp-pump".into())
            .spawn(move || pump(s2, w))
            .map_err(|e| format!("pump thread: {e}"))?;
        let s3 = sh.clone();
        rt.spawn(async move {
            let mut t = tokio::time::interval(RELOAD);
            loop {
                tokio::select! {
                    _ = t.tick() => {}
                    _ = s3.ct.cancelled() => break,
                }
                if let Some(n) = tw.poll() {
                    s3.log.line(&format!("tiers now {n}"));
                    s3.tiers.send_replace(n);
                }
            }
        });
        Ok(Arc::new(Host {
            sh,
            pump: Mutex::new(Some(th)),
            sim,
        }))
    }

    /// This host's id: the prefix of its job handles, and what `status` reports.
    pub fn id(&self) -> &str {
        &self.sh.id
    }

    /// The secrets it knows, for a writer of the process's own around a transport ([`scrub::Writer`]).
    pub fn secrets(&self) -> Secrets {
        self.sh.secrets.clone()
    }

    /// The call log (`<cache>/x5desk/mcp.log`; under `--sim` the temp dir's).
    pub fn log(&self) -> &Log {
        &self.sh.log
    }

    /// Where agents write: `<videos>/x5desk/agent` (under `--sim` the temp dir's).
    pub fn agent(&self) -> &Path {
        &self.sh.agent
    }

    /// The tiers enabled now.
    pub fn tiers(&self) -> Tiers {
        *self.sh.tiers.borrow()
    }

    /// The simulated camera and its dirs (`--sim`).
    pub fn sim(&self) -> Option<&Sim> {
        self.sim.as_ref()
    }

    /// Reads the worker's state as it is now.
    pub fn with<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        f(&self.sh.st.lock().unwrap())
    }

    /// Resolves once the host is shutting down.
    pub async fn closed(&self) {
        self.sh.ct.cancelled().await
    }

    /// Whether it is shutting down.
    pub fn is_closed(&self) -> bool {
        self.sh.ct.is_cancelled()
    }

    /// Serves one client on `r` / `w` (a stdio pair, a socket's halves, a duplex) with an rmcp service of its own
    /// over this host, until the client goes or the host shuts down. Everything written to `w` is scrubbed. `client`
    /// names it in the log (`stdio`, `endpoint 3`); `narrow`: that client's own `--tiers`, which its `tools/list`,
    /// calls, `status` and instructions follow (the other connections get the host's tiers).
    pub async fn serve<R, W>(
        self: &Arc<Self>,
        r: R,
        w: W,
        client: &str,
        narrow: Option<Tiers>,
    ) -> Result<(), String>
    where
        R: AsyncRead + Send + Unpin + 'static,
        W: AsyncWrite + Send + Unpin + 'static,
    {
        if self.sh.ct.is_cancelled() {
            return Err("the host is shutting down".into());
        }
        self.sh.conns.fetch_add(1, Relaxed);
        let conn = Conn::new(self.clone(), client, narrow);
        let sinks = conn.sinks.clone();
        let w = scrub::Writer::new(w, self.sh.secrets.clone());
        let r = conn.serve_with_ct((r, w), self.sh.ct.child_token()).await;
        let svc = match r {
            Ok(s) => s,
            Err(e) => {
                self.sh.conns.fetch_sub(1, Relaxed);
                return Err(format!("{client}: {e}"));
            }
        };
        self.sh.log.line(&format!("{client} connected"));
        let peer = svc.peer().clone();
        let mut tiers = self.sh.tiers.subscribe();
        let end = svc.waiting();
        tokio::pin!(end);
        loop {
            tokio::select! {
                _ = &mut end => break,
                r = tiers.changed() => {
                    if r.is_err() {
                        continue;
                    }
                    // a session that began with `initialize` (before 2026-07-28) takes it on the session; later ones
                    // on their subscriptions
                    if peer.peer_info().is_some() {
                        let _ = peer.notify_tool_list_changed().await;
                    }
                    let v: Vec<_> = sinks.lock().unwrap().clone();
                    for s in v {
                        let _ = s.notify_tool_list_changed().await;
                    }
                }
            }
        }
        self.sh.conns.fetch_sub(1, Relaxed);
        self.sh.log.line(&format!("{client} gone"));
        Ok(())
    }

    /// Stops serving (every connection ends), releases the camera and ends the worker; a running download goes back
    /// to the queue with its `.part`, for the next host. Returns once the worker is gone and the queue and the jobs
    /// are saved (the next host may take them).
    pub async fn shutdown(&self) {
        self.sh.ct.cancel();
        self.sh.stop.store(true, Relaxed);
        let h = self.pump.lock().unwrap().take();
        if let Some(h) = h {
            let _ = tokio::task::spawn_blocking(move || h.join()).await;
        }
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.sh.ct.cancel();
        self.sh.stop.store(true, Relaxed);
        if let Some(h) = self.pump.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a later host takes: the downloads and jobs into the agent folder from a camera or a card (a finished
    /// file's size read), not those a forged or hand-edited file adds; a queue file that is no UTF-8 is moved aside;
    /// a jobs file there but unreadable is left alone, the queue in memory and the lock let go.
    #[tokio::test]
    async fn taken() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-taken-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let agent = d.join("agent");
        std::fs::create_dir_all(&agent).unwrap();
        let (qf, jf) = (d.join("mcp-downloads"), d.join("mcp-jobs"));
        let (vid, img, lrv) = (
            agent.join("VID_20260116_152935_00_603.insv"),
            agent.join("IMG_20260116_174103_00_620.jpg"),
            agent.join("LRV_20260116_152935_01_603.lrv"),
        );
        std::fs::write(&img, b"12345").unwrap();
        std::fs::write(&lrv, b"123").unwrap();
        let line = |st: &str, u: &str, p: &Path| format!("{st}\t{u}\t\t{}\n", p.display());
        let q = [
            line(
                "queued",
                "/DCIM/Camera01/VID_20260116_152935_00_603.insv",
                &vid,
            ),
            line("queued", "/DCIM/Camera01/IMG_1.jpg", &d.join("zshrc")),
            line("queued", "file:///etc/passwd", &agent.join("passwd")),
            line(
                "queued",
                "/DCIM/Camera01/a.insv",
                &agent.join("..").join("a.insv"),
            ),
            line(
                "done",
                "/DCIM/Camera01/IMG_20260116_174103_00_620.jpg",
                &img,
            ),
            // unsafe names, sources no camera or card has
            line("queued", "/DCIM/Camera01/b.insv", &agent.join("CON")),
            line("queued", "/DCIM/Camera01/c.insv", &agent.join("a:b")),
            line("queued", "/DCIM/Camera01/CON.lrv", &agent.join("d.lrv")),
            line("queued", "DCIM/Camera01/e.insv", &agent.join("e.insv")),
            line(
                "queued",
                "file:///v/X5/DCIM/../../etc/f.insv",
                &agent.join("f.insv"),
            ),
            // finished, but its host did not collect it: done (then pruned, no job has it)
            line(
                "queued",
                "/DCIM/Camera01/LRV_20260116_152935_01_603.lrv",
                &lrv,
            ),
        ];
        std::fs::write(&qf, q.concat()).unwrap();
        let f = |u: &str, p: &Path| job::File::new(u.into(), p.to_path_buf());
        let (mut j, _) = Jobs::load("old".into(), jf.clone()).unwrap();
        j.add(
            vec![],
            vec![
                f("/DCIM/Camera01/VID_20260116_152935_00_603.insv", &vid),
                f("/DCIM/Camera01/IMG_20260116_174103_00_620.jpg", &img),
            ],
        );
        j.add(
            vec![],
            vec![f("/DCIM/Camera01/IMG_1.jpg", &d.join("zshrc"))],
        );
        j.save().unwrap();
        let (q, j, l, said) = kept(Some(&d), &agent).await;
        let v: Vec<(&Path, dl::St)> = q.v.iter().map(|x| (x.dst.as_path(), x.st)).collect();
        assert_eq!(
            v,
            [
                (vid.as_path(), dl::St::Queued),
                (img.as_path(), dl::St::Done)
            ]
        );
        assert_eq!(q.v[1].prog.total.load(Relaxed), 5);
        assert_eq!((j.v.len(), j.v[0].h.as_str()), (1, "old-1"));
        assert!(l.is_some());
        assert!(
            said.iter()
                .any(|s| s.contains("left out") && s.contains("8 downloads, 1 jobs")),
            "{said:?}"
        );
        drop((q, j, l));
        // an unreadable queue file: not taken over, left as it is
        #[cfg(unix)]
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            let t = std::fs::read(&qf).unwrap();
            std::fs::set_permissions(&qf, std::fs::Permissions::from_mode(0o000)).unwrap();
            let (q, _, l, said) = kept(Some(&d), &agent).await;
            assert!(
                q.v.is_empty() && l.is_none() && said[0].contains("stays in memory"),
                "{said:?}"
            );
            std::fs::set_permissions(&qf, std::fs::Permissions::from_mode(0o600)).unwrap();
            assert_eq!(std::fs::read(&qf).unwrap(), t);
        }
        // a queue file that is no UTF-8: moved aside
        std::fs::write(&qf, b"queued\t/DCIM/x\t\t\xff\n").unwrap();
        let (q, _, l, said) = kept(Some(&d), &agent).await;
        assert!(
            q.v.is_empty() && l.is_some() && d.join("mcp-downloads.bad").exists(),
            "{said:?}"
        );
        drop(l);
        // an unreadable jobs file: not taken over
        #[cfg(unix)]
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&jf, std::fs::Permissions::from_mode(0o000)).unwrap();
            let (_, _, l, said) = kept(Some(&d), &agent).await;
            assert!(
                l.is_none() && said[0].contains("stays in memory"),
                "{said:?}"
            );
            let lf = d.join("mcp-downloads.lock");
            assert!(
                lock_wait(&lf, Duration::ZERO).await.unwrap().is_some(),
                "let go"
            );
            std::fs::set_permissions(&jf, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A lock another holds is waited for up to the wait, and taken once let go; owner-only, its dir too.
    #[tokio::test]
    async fn handover() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let p = d.join("state").join("mcp-downloads.lock");
        let a = lock_wait(&p, Duration::ZERO).await.unwrap().unwrap();
        let t = Instant::now();
        assert!(
            lock_wait(&p, Duration::from_millis(300))
                .await
                .unwrap()
                .is_none()
        );
        assert!(t.elapsed() >= Duration::from_millis(300));
        let w = tokio::spawn({
            let p = p.clone();
            async move { lock_wait(&p, Duration::from_secs(5)).await }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!w.is_finished());
        drop(a);
        assert!(w.await.unwrap().unwrap().is_some());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!((m(&p), m(p.parent().unwrap())), (0o600, 0o700));
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A queued import whose card went does not keep the camera; one that can start, a camera download or a running
    /// one does.
    #[test]
    fn busy() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-busy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let src = d.join("card.insv");
        std::fs::write(&src, b"x").unwrap();
        let mut q = dl::Queue::default();
        assert!(!super::busy(&q));
        q.add(
            &format!("file://{}", d.join("gone.insv").display()),
            d.join("a"),
        );
        assert!(!super::busy(&q), "its card went: it never starts");
        for st in [dl::St::Paused, dl::St::Done, dl::St::Failed] {
            q.v[0].st = st;
            assert!(!super::busy(&q), "{st:?}");
        }
        q.v[0].st = dl::St::Running;
        assert!(super::busy(&q), "running");
        q.v[0].st = dl::St::Queued;
        q.add(&format!("file://{}", src.display()), d.join("b"));
        assert!(super::busy(&q), "an import that can start");
        q.v.pop();
        assert!(!super::busy(&q));
        q.add(
            "/DCIM/Camera01/VID_20260116_152935_00_603.insv",
            d.join("c"),
        );
        assert!(super::busy(&q), "a camera download");
        std::fs::remove_dir_all(&d).unwrap();
    }
}
