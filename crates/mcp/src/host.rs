//! The headless host: the camera worker of `cam` (USB followed, never connected on hotplug, never the camera clock)
//! and its capture worker on the same session, the download queue and the jobs, mounted cards watched, and the state
//! the tools read. A pump thread owns the camera worker and turns its reports into [`State`], waking whoever waits on
//! it; a capture thread does the same for the capture worker, which it keeps on the camera worker's session (`Tap`),
//! and records the outcome of every command a tool asked it for by number. Tools send the workers commands and wait
//! with timeouts; camera-writing tools take turns ([`Shared::act`]). The camera is released after [`IDLE`] without a
//! tool call (not while a call runs, a download runs or can start, or the camera captures), and on
//! [`Host::shutdown`] (stdin's end, a signal: the process side calls it), which first stops a recording this host
//! started ([`Rec`], within [`STOPPING`]: on the session directly, not behind the capture worker's queue).
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
    capture::{self, Out},
    conf::{self, Shot},
    disk, dl,
};
use link::{Plug, pb};
use rmcp::ServiceExt;
use std::{
    collections::{HashSet, VecDeque},
    fs::File,
    io,
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed},
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

/// How long a host waits for the one before it (ending: it stops its recording, then saves the queue and the jobs once
/// its worker is gone) to let go of them: more than [`STOPPING`] and [`ENDING`] together.
pub const HANDOVER: Duration = Duration::from_secs(15);

/// How long a host's end waits for its worker (a camera call can hold it up for 30 s) before it saves the queue and
/// the jobs and lets go of them without it: well within [`HANDOVER`].
pub const ENDING: Duration = Duration::from_secs(5);

/// How long a host's end tries to stop a recording it started (its start's outcome first) before the session closes.
pub const STOPPING: Duration = Duration::from_secs(5);

/// Outcomes of asked capture commands kept.
const OUTS: usize = 32;

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
    /// the capture worker's newest report and its session: the camera's now when `csid == tsid`
    pub cap: capture::St,
    pub csid: u32,
    /// the camera worker's session now (`Tap::sid`: it changes with each connect and each end of one), taken by the
    /// capture thread and by a report of the connection
    pub tsid: u32,
    /// asked capture commands' outcomes by number, the newest [`OUTS`]
    pub outs: VecDeque<(u64, Out)>,
    /// the newest asked capture command: its number and tool
    pub asked: (u64, &'static str),
    /// the note number the newest listing came after, and whether it listed the camera (not a card)
    pub album_seq: u64,
    pub album_cam: bool,
    /// a delete runs: the note number before it (the delete's note ends it)
    pub deleting: Option<u64>,
    /// item keys this host deleted
    pub deleted: HashSet<String>,
    /// a recording or timelapse this host started
    pub rec: Option<Rec>,
    /// how the newest one this host started ended: its start's number, the clip a stop of this host's named (none:
    /// it named none), or why it ended without one
    pub ended: Option<(u64, Result<Option<String>, String>)>,
    /// the session whose capture state was read whole once (a report of a load that did not fail)
    judged: u32,
}

/// A recording or timelapse this host started: kept across the sessions of that camera (a link lost while it records
/// leaves it recording), until a stop of this host's, a refused start, or the camera reporting it ended.
#[derive(Clone, Debug)]
pub struct Rec {
    /// the camera's serial
    pub serial: Option<String>,
    /// the start's number; its outcome came and was no refusal; the camera reported capturing since
    pub n: u64,
    pub sent: bool,
    pub seen: bool,
    /// the mode it was started in (the stop's way)
    pub mode: String,
    pub at: Instant,
    /// the host stops it then (`capture_record`'s `seconds`)
    pub until: Option<Instant>,
    /// a stop of this host's asked for it: its number (its outcome ends it)
    pub stop: Option<u64>,
    /// when its start's outcome came (it went out, or may have)
    pub sent_at: Option<Instant>,
}

/// How long after its start went out a recording the camera has not shown capturing yet is still waited for (a stop
/// meanwhile is tried again once it shows; a capture state read later that shows none: not begun).
pub const LATE: Duration = Duration::from_secs(10);

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
    /// the capture worker's commands
    ctx: Mutex<Sender<capture::Cmd>>,
    /// capture commands asked so far
    asks: AtomicU64,
    /// camera-writing tools take turns, so each command's outcome is its own and a check holds until it is sent
    pub act: tokio::sync::Mutex<()>,
    /// the camera worker's session, which the capture worker follows
    pub tap: Taps,
    /// the capture thread ends
    cstop: AtomicBool,
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

    /// Sends the capture worker command `c` as asked by `tool` (`seen`: the camera as it was checked against); its
    /// number, under which its outcome comes ([`State::outs`]).
    pub fn ask(&self, tool: &'static str, c: capture::Cmd, seen: Option<capture::Seen>) -> u64 {
        self.ask_with(tool, c, seen, |_, _| {})
    }

    /// [`Shared::ask`], with `f` given the state and the number before the command goes (its outcome cannot come
    /// before what `f` sets).
    pub fn ask_with(
        &self,
        tool: &'static str,
        c: capture::Cmd,
        seen: Option<capture::Seen>,
        f: impl FnOnce(&mut State, u64),
    ) -> u64 {
        let n = self.asks.fetch_add(1, Relaxed) + 1;
        {
            let mut s = self.st.lock().unwrap();
            s.asked = (n, tool);
            f(&mut s, n)
        }
        let c = capture::Cmd::Ask {
            n,
            c: Box::new(c),
            seen,
        };
        let _ = self.ctx.lock().unwrap().send(c);
        n
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
        if let Up::Done(p) = &u {
            self.log.line(&format!("downloaded {}", p.display()));
            return self.fetched(p);
        }
        // the camera worker's session, which it moves on before reporting a connection gone (the tap is never locked
        // under the state)
        let tsid = matches!(u, Up::Info(_)).then(|| self.tap.lock().unwrap().sid);
        let mut s = self.st.lock().unwrap();
        match u {
            Up::Info(i) => {
                if let Some((_, pw)) = &i.wifi {
                    self.secrets.add(pw)
                }
                // another camera: a recording this host started is that one's
                if let (Some(r), Some(id)) = (&s.rec, &i.id)
                    && r.serial.as_ref().is_some_and(|x| *x != id.serial)
                {
                    s.rec = None
                }
                // a new source, or the camera's session come or gone: the album is listed again
                if i.src != s.info.src || (i.src.is_none() && i.conn != s.info.conn) {
                    s.album = None;
                    s.thumbs.clear();
                }
                if i.conn != s.info.conn {
                    s.released = false;
                    // the capture state of the session before is not the camera's now, also before the capture
                    // thread looks (every 100 ms) and sends the new one
                    if let Some(n) = tsid {
                        s.tsid = s.tsid.max(n)
                    }
                }
                s.info = i
            }
            Up::Album(v) => {
                (s.album_seq, s.album_cam) = (s.seq, s.info.src.is_none());
                s.album = Some(v)
            }
            Up::Thumb(k, t) => {
                s.thumbs.retain(|x| x.0 != k);
                if s.thumbs.len() >= THUMBS {
                    s.thumbs.pop_front();
                }
                s.thumbs.push_back((k, t))
            }
            Up::Note(n, e) => {
                self.log.line(&format!("camera: {n}"));
                // a delete's note: it ran (the worker took no other command meanwhile)
                if s.deleting.is_some()
                    && (n.starts_with("deleted ")
                        || n.starts_with("delete:")
                        || n == "not connected")
                {
                    s.deleting = None
                }
                s.note(n, e)
            }
            Up::Done(_) => {}
            Up::Stats(_) => {}
        }
    }

    fn bump(&self) {
        self.ver.send_modify(|v| *v += 1)
    }

    /// A download into `p` finished: the bytes the camera sent kept with the jobs' files (what `camera_delete` checks
    /// a local copy against, in later hosts too).
    fn fetched(&self, p: &Path) {
        let (mut jobs, mut q) = (self.jobs.lock().unwrap(), self.q.lock().unwrap());
        let Some((u, n)) = (q.v.iter())
            .find(|d| d.dst == p && d.st == dl::St::Done)
            .map(|d| (d.uri.clone(), d.prog.total.load(Relaxed)))
            .filter(|x| x.1 > 0)
        else {
            return;
        };
        let mut hit = false;
        for f in jobs.v.iter_mut().flat_map(|j| j.files.iter_mut()) {
            if f.dst == p && f.uri == u {
                (f.size, hit) = (Some(n), true)
            }
        }
        if hit {
            self.save(&mut jobs, &mut q)
        }
    }

    /// A capture worker's report (`sid`: its session): every asked command's outcome taken, the rest only of the
    /// camera's session now. A recording this host started: its start's outcome, the camera capturing, and its end:
    /// its stop's outcome when this host asked for one, else the camera saying it stopped, the capture state back to
    /// idle in a session that showed it capturing, or a new session (the link was lost meanwhile) that shows it idle
    /// from its start. A report without a camera (the link gone) says nothing of it.
    fn apply_cap(&self, sid: u32, st: capture::St) {
        let mut s = self.st.lock().unwrap();
        if let Some((k, o)) = st.out.clone()
            && !s.outs.iter().any(|x| x.0 == k)
        {
            if let Some(r) = s.rec.clone() {
                if r.n == k {
                    match &o {
                        Out::No(why) => {
                            s.ended = Some((k, Err(why.clone())));
                            s.rec = None
                        }
                        // a failure after the start went out (a timeout): it may have begun
                        Out::Sent(_) | Out::Failed(_) => {
                            let r = s.rec.as_mut().unwrap();
                            (r.sent, r.sent_at) = (true, Some(Instant::now()))
                        }
                    }
                } else if r.stop == Some(k) {
                    match &o {
                        Out::Sent(u) => {
                            s.ended = Some((r.n, Ok(u.clone())));
                            s.rec = None
                        }
                        // a start the camera has not shown capturing yet: stopped once it does (its stop due now)
                        Out::No(t)
                            if t.contains("nothing to stop")
                                && !r.seen
                                && r.sent_at.is_none_or(|t| t.elapsed() < LATE) =>
                        {
                            let r = s.rec.as_mut().unwrap();
                            (r.stop, r.until) = (None, Some(Instant::now()))
                        }
                        Out::No(t) if t.contains("nothing to stop") => {
                            s.ended = Some((r.n, Err("it had ended".into())));
                            s.rec = None
                        }
                        _ => s.rec.as_mut().unwrap().stop = None,
                    }
                }
            }
            if s.outs.len() >= OUTS {
                s.outs.pop_front();
            }
            s.outs.push_back((k, o))
        }
        // a report of the newest one's session also once a connection's report moved on (one on its way then): its
        // stop notes still end this host's recording, and the readers take it as the camera's only while csid == tsid
        if sid != s.tsid && sid != s.csid {
            return;
        }
        // its notes with the camera worker's
        if let Some((t, e)) = &st.note
            && (sid != s.csid || s.cap.note != st.note)
        {
            self.log.line(&format!("capture: {t}"));
            s.note(format!("capture: {t}"), *e);
        }
        let run = capture::running(st.cap);
        // a session's capture state read whole (its load did not fail: no default state taken for the camera's), the
        // first time
        let read = st.on && st.mode.is_some() && st.vals == st.mode;
        let (first, was) = (
            read && s.judged != sid,
            sid == s.csid && s.cap.on && capture::running(s.cap.cap),
        );
        if read {
            s.judged = sid
        }
        let stopped = (st.note.as_ref())
            .filter(|n| s.cap.note.as_ref() != Some(n) || sid != s.csid)
            .map(|n| n.0.clone())
            .filter(|n| {
                [
                    "capture stopped",
                    "the SD card is full",
                    "the camera shuts down",
                ]
                .iter()
                .any(|w| n.starts_with(w))
            });
        if let Some(r) = s.rec.as_mut().filter(|_| st.on && st.mode.is_some()) {
            if run {
                r.seen = true
            }
            // a stop of this host's on its way: its outcome says how it ended
            let late = r.sent_at.is_some_and(|t| t.elapsed() > LATE);
            let why = match (
                stopped,
                !run && (was || first && r.sent) && r.stop.is_none(),
            ) {
                (Some(n), _) if r.stop.is_none() => Some(n),
                (None, true) => Some("the camera stopped capturing".to_string()),
                // gone out, and a capture state read since LATE passed shows none: not begun
                (None, false)
                    if r.sent
                        && !r.seen
                        && !run
                        && read
                        && late
                        && r.stop.is_none()
                        && st.cap_at.zip(r.sent_at).is_some_and(|(a, s)| a > s + LATE) =>
                {
                    Some("the camera did not begin capturing".to_string())
                }
                _ => None,
            };
            if let Some(w) = why {
                let n = r.n;
                if s.ended.as_ref().is_none_or(|e| e.0 != n) {
                    s.ended = Some((n, Err(w)))
                }
                s.rec = None
            }
        }
        (s.cap, s.csid) = (st, sid)
    }

    /// Releases the camera after [`Shared::idle`] without a call, unless a call runs, a download runs or can start
    /// ([`busy`]), or the camera captures (or this host started a recording).
    fn idle_check(&self) {
        let on = {
            let s = self.st.lock().unwrap();
            s.info.conn == Link::On && !s.released && !capturing(&s) && s.rec.is_none()
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

impl State {
    /// A note of the workers: numbered, the newest [`NOTES`] kept.
    fn note(&mut self, n: String, e: bool) {
        self.seq += 1;
        let k = self.seq;
        if self.notes.len() >= NOTES {
            self.notes.pop_front();
        }
        self.notes.push_back((k, n, e))
    }
}

/// Whether the camera captures: a recording (the camera worker's word) or any capture the capture worker reported on
/// the session now (a timelapse, interval or starlapse too).
pub fn capturing(s: &State) -> bool {
    s.info.busy || (s.csid == s.tsid && s.cap.on && capture::running(s.cap.cap))
}

/// The capture thread: the capture worker kept on the camera worker's session (a new one sent as it comes, `None`
/// when it ends), its reports into the state, until the host stops; then the worker dropped (its pending calls fail
/// once the session closed).
fn cap_pump(sh: Arc<Shared>, w: capture::Worker) {
    let mut sid = None;
    while !sh.cstop.load(Relaxed) {
        let (n, c) = {
            let t = sh.tap.lock().unwrap();
            (t.sid, t.cam.clone())
        };
        if sid != Some(n) {
            sid = Some(n);
            sh.st.lock().unwrap().tsid = n;
            let _ = w.tx.send(capture::Cmd::Cam(n, c));
            sh.bump()
        }
        match w.rx.recv_timeout(Duration::from_millis(100)) {
            Ok((n, st)) => {
                sh.apply_cap(n, st);
                while let Ok((n, st)) = w.rx.try_recv() {
                    sh.apply_cap(n, st)
                }
                sh.bump()
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    sh.tap.lock().unwrap().notes = None;
}

/// The stops of recordings this host started with `seconds` (`capture_record`): once one is due and its start went
/// out, its stop, again every 5 s while it fails (the link lost meanwhile: once it is back), until the host ends.
async fn timer(h: std::sync::Weak<Host>) {
    let mut t = tokio::time::interval(Duration::from_millis(250));
    let mut next = None;
    loop {
        t.tick().await;
        let Some(h) = h.upgrade() else { break };
        if h.sh.ct.is_cancelled() {
            break;
        }
        let due = {
            let s = h.sh.st.lock().unwrap();
            (s.rec.as_ref())
                .filter(|r| r.sent && r.until.is_some_and(|u| u <= Instant::now()))
                .map(|r| r.n)
        };
        let Some(n) = due else {
            next = None;
            continue;
        };
        if next.is_some_and(|t| Instant::now() < t) {
            continue;
        }
        let c = Conn::new(h.clone(), "timer", None);
        match c.timed_stop(n).await {
            Ok(()) => next = None,
            Err(e) => {
                if next.is_none() {
                    h.sh.log.line(&format!(
                        "the timed stop of capture {n} failed: {e}; again every 5 s"
                    ))
                }
                next = Some(Instant::now() + Duration::from_secs(5))
            }
        }
    }
}

/// What the end's stop came to.
enum Stopped {
    /// it stopped the camera's capture: the clip
    Clip(String),
    /// the recording had ended, or its start was refused
    Ended,
    /// the camera did not capture until the end: its start went out (it may begin yet), or was not answered
    Never(bool),
}

/// At the host's end: a recording this host started stopped on the camera's session directly (a stop behind the
/// capture worker's queue could come after the session closed). The camera's capture state is read until it shows the
/// capture (then stopped), or the recording is known to have ended or been refused, or [`STOPPING`] passed (a start
/// still on its way, or the camera slow to begin). What came of it goes to the log: a recording that may still run is
/// said so.
fn stop_own(sh: &Arc<Shared>) {
    let end = Instant::now() + STOPPING;
    if sh.st.lock().unwrap().rec.is_none() {
        return;
    }
    let Some(cam) = sh.tap.lock().unwrap().cam.clone() else {
        sh.log
            .line("the camera's session is gone: the recording this host started may still run");
        return;
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let s2 = sh.clone();
    let r = std::thread::Builder::new()
        .name("x5-mcp-stop".into())
        .spawn(move || {
            let r = (|| -> link::Result<Stopped> {
                loop {
                    let x = cam.capture_status().wait()?;
                    let Some(rec) = s2.st.lock().unwrap().rec.clone() else {
                        return Ok(Stopped::Ended);
                    };
                    if capture::running(x.state) {
                        let v = match conf::shot(&rec.mode) {
                            Shot::Lapse(x) => cam.stop_timelapse(x).wait()?,
                            Shot::Rec(x) => cam.stop_capture(x).wait()?,
                            _ => cam.stop_capture(pb::CaptureMode::Normal).wait()?,
                        };
                        return Ok(Stopped::Clip(v.uri));
                    }
                    if rec.seen {
                        return Ok(Stopped::Ended);
                    }
                    if Instant::now() + Duration::from_millis(300) >= end {
                        return Ok(Stopped::Never(rec.sent));
                    }
                    std::thread::sleep(Duration::from_millis(300))
                }
            })();
            let _ = tx.send(r);
        });
    let w = end.saturating_duration_since(Instant::now()) + Duration::from_millis(500);
    let may = "the recording this host started may still run";
    let l = match r.map(|_| rx.recv_timeout(w)) {
        Ok(Ok(Ok(Stopped::Clip(u)))) => format!("stopped the recording this host started: {u}"),
        Ok(Ok(Ok(Stopped::Ended))) => "the recording this host started had ended".into(),
        Ok(Ok(Ok(Stopped::Never(true)))) => format!(
            "{may}: its start went out, but the camera did not capture within {} s",
            STOPPING.as_secs()
        ),
        Ok(Ok(Ok(Stopped::Never(false)))) => format!("{may}: its start was not answered"),
        Ok(Ok(Err(e))) => format!("{may}: its stop failed: {e}"),
        Ok(Err(_)) => format!(
            "{may}: no answer to its stop within {} s",
            STOPPING.as_secs()
        ),
        Err(e) => format!("{may}: {e}"),
    };
    sh.log.line(&l)
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
    stop_own(&sh);
    let _ = w.tx.send(Cmd::Disconnect);
    // the capture worker ends on its own thread: its calls fail once the session closed
    sh.cstop.store(true, Relaxed);
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
                let t = match std::env::var("X5DESK_MCP_TIERS") {
                    Ok(v) => Tiers::parse(&v).map_err(|e| format!("X5DESK_MCP_TIERS: {e}"))?,
                    Err(_) => Tiers::SIM,
                };
                // `dir` is a real camera: the environment gives it no camera, delete or untested tier
                Source::Fixed(match sim {
                    Some(_) => t,
                    None => t.and(Tiers::FILES),
                })
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
        // the capture worker on the camera worker's session, the camera's notifications passed on to it (under the
        // tap's lock: a send only)
        let tap = Taps::default();
        let cw = capture::Worker::spawn(|| {});
        let ctx = cw.tx.clone();
        let t2 = ctx.clone();
        tap.lock().unwrap().notes = Some(Box::new(move |n: &link::Note| {
            drop(t2.send(capture::Cmd::Note(n.clone())))
        }));
        // the bus is followed (the camera's plug state) but never connected on its own; the simulated camera is
        // "plugged in" instead
        let w = Worker::spawn(
            connect,
            q.clone(),
            cache,
            auto,
            sim.is_none(),
            tap.clone(),
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
            ctx: Mutex::new(ctx),
            asks: AtomicU64::new(0),
            act: tokio::sync::Mutex::new(()),
            tap,
            cstop: AtomicBool::new(false),
        });
        let s4 = sh.clone();
        std::thread::Builder::new()
            .name("x5-mcp-cap".into())
            .spawn(move || cap_pump(s4, cw))
            .map_err(|e| format!("capture thread: {e}"))?;
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
        let h = Arc::new(Host {
            sh,
            pump: Mutex::new(Some(th)),
            sim,
        });
        rt.spawn(timer(Arc::downgrade(&h)));
        Ok(h)
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

    /// A report of the session just gone, come after the connection's report moved `tsid` on (a battery's capture
    /// stop, then the shutdown), still ends this host's recording with the camera's reason, and is not taken as the
    /// camera's now; one of an older session is not applied.
    #[tokio::test(flavor = "multi_thread")]
    async fn late_report() {
        let h = Host::new(Opts {
            sim: Some(crate::sim::Opts {
                rate: 50_000_000,
                size: 1000,
                dir: None,
            }),
            tiers: Some(Source::Fixed(Tiers::parse("read").unwrap())),
            ..Default::default()
        })
        .await
        .unwrap();
        // the host's threads gone (the capture thread within 100 ms): only this test touches the state
        h.shutdown().await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        use pb::CameraCaptureState as C;
        let st = |c: C, note: Option<&str>| capture::St {
            on: true,
            mode: Some("VIDEO_NORMAL".into()),
            vals: Some("VIDEO_NORMAL".into()),
            cap: c as i32,
            note: note.map(|n| (n.to_string(), true)),
            ..Default::default()
        };
        let rec = Rec {
            serial: None,
            n: 3,
            sent: true,
            seen: true,
            mode: "VIDEO_NORMAL".into(),
            at: Instant::now(),
            until: None,
            stop: None,
            sent_at: Some(Instant::now()),
        };
        let set = |rec: &Rec| {
            let mut s = h.sh.st.lock().unwrap();
            (s.cap, s.csid, s.judged) = (st(C::NormalCapture, None), 7, 7);
            // the connection's report: the session gone
            (s.tsid, s.rec, s.ended) = (8, Some(rec.clone()), None);
        };
        set(&rec);
        let stop = "capture stopped: LowBattery";
        h.sh.apply_cap(6, st(C::NotCapture, Some(stop)));
        assert!(
            h.with(|s| s.rec.is_some() && s.csid == 7),
            "an older session's"
        );
        h.sh.apply_cap(7, st(C::NotCapture, Some(stop)));
        h.with(|s| {
            assert!(s.rec.is_none() && !capturing(s));
            assert_eq!(s.ended, Some((3, Err(stop.to_string()))));
        });
        // nor taken as the camera's now
        set(&rec);
        h.sh.apply_cap(7, st(C::NormalCapture, None));
        assert!(h.with(|s| s.rec.is_some() && s.csid == 7 && !capturing(s)));
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
