//! Render records: the stills and exports of `render_still` and `export` ([`crate::render`]), kept in `mcp-renders`
//! beside the jobs file and under its lock, one JSON object a line; an older x5desk never reads the file. Their
//! handles come from the jobs' numbering ([`crate::job::Jobs::handle`]), so a handle names one job of either kind. A
//! render never carries over to another host: what a host left unfinished (a crash, a kill) the next one settles by
//! the identity of its `.part`, never by name ([`Renders::settle`]), then keeps every unfinished record and the newest
//! [`KEEP`] finished ones ([`Renders::prune`]); a host prunes its own as they end ([`Renders::end`]). A running job's
//! state is the poller's latest reading, in memory ([`RJob::live`]): readers never ask the backend.

use crate::{
    job,
    render::{Kind, Lane, Phase, RunSt},
};
use media::part::{Id, Settled};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{Instant, SystemTime},
};

/// Finished render jobs a host keeps, of those before it and of its own, beside every unfinished one (downloads keep
/// their own).
pub const KEEP: usize = 100;

/// The renders file's first line.
const HEAD: &str = "# x5desk mcp renders: one JSON object a line";

/// Why a job its host left unfinished ended cancelled.
pub const GONE: &str = "its host ended before it finished";

/// Why a job its host stopped at its end ended cancelled.
pub const ENDED: &str = "its host ended: x5desk stops its renders when its MCP server stops";

/// One render job: what it reads and writes, and how it ended. Never resumed by another host.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RJob {
    /// its handle, `<host id>-<n>`, from the shared numbering
    #[serde(rename = "job")]
    pub h: String,
    pub kind: Kind,
    /// when it was made
    #[serde(rename = "at_ms", with = "crate::job::ms")]
    pub at: SystemTime,
    /// the source's main files, canonical, UTF-8
    pub src: Vec<PathBuf>,
    /// the file it publishes, in the agent folder, UTF-8
    pub dst: PathBuf,
    /// the reserved `<dst>.part`'s identity; none until it exists
    #[serde(default)]
    pub part: Option<Id>,
    pub lane: Lane,
    /// resolved, as results show them
    pub settings: Value,
    /// output frames (1 for a still)
    pub frames: u64,
    /// the estimate the free-space check counted
    pub bytes: u64,
    /// `job_cancel` was called on it
    #[serde(default)]
    pub cancelled: bool,
    /// how it ended, once settled (none while it may still write)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<End>,
    /// the poller's latest reading of its run; memory only
    #[serde(skip)]
    pub live: Option<Snap>,
}

/// How a render job ended: `done`, `failed` or `cancelled`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct End {
    pub state: String,
    /// why it failed, or why it was cancelled when not by a cancel
    #[serde(default)]
    pub err: String,
    pub frames: u64,
    /// how long it ran (0: unknown, its host crashed)
    pub secs: f64,
    /// the file's bytes, done
    #[serde(default)]
    pub bytes: u64,
    /// pictures that could not be decoded and repeat the one before
    #[serde(default)]
    pub concealed: u64,
}

impl End {
    /// Done: `frames` in `secs`, the file `bytes` long.
    pub fn done(frames: u64, secs: f64, bytes: u64, concealed: u64) -> End {
        End {
            state: "done".into(),
            err: String::new(),
            frames,
            secs,
            bytes,
            concealed,
        }
    }

    /// Failed: `why`. No file is left.
    pub fn failed(why: &str) -> End {
        End {
            state: "failed".into(),
            ..End::cancelled(why)
        }
    }

    /// Cancelled; `why` when not by a cancel. No file is left.
    pub fn cancelled(why: &str) -> End {
        End {
            state: "cancelled".into(),
            err: why.into(),
            frames: 0,
            secs: 0.0,
            bytes: 0,
            concealed: 0,
        }
    }
}

/// The poller's reading of a run: its state, when read, when its last frame came (before the first: when the job was
/// made, so job_wait's progress creeps on through the queue and the setup), and when it last moved (it began running,
/// changed phase or made a frame: the watchdog's).
#[derive(Clone, Debug)]
pub struct Snap {
    pub st: RunSt,
    pub at: Instant,
    pub last_frame_at: Instant,
    pub moved: Instant,
}

impl Snap {
    /// Reading `st` at `now` of a job made at `made`, after the reading `prev`.
    pub fn after(prev: Option<&Snap>, st: RunSt, now: Instant, made: SystemTime) -> Snap {
        let stage = |s: &RunSt| match s {
            RunSt::Running { phase, .. } => Some(*phase),
            _ => None,
        };
        let framed = frames(&st) > prev.map_or(0, |p| frames(&p.st));
        let made = (SystemTime::now().duration_since(made).ok())
            .and_then(|d| now.checked_sub(d))
            .unwrap_or(now);
        let last_frame_at = match (framed, prev) {
            (true, _) => now,
            (false, Some(p)) => p.last_frame_at,
            (false, None) => made,
        };
        let moved = match prev {
            Some(p) if !framed && stage(&p.st) == stage(&st) => p.moved,
            _ => now,
        };
        Snap {
            st,
            at: now,
            last_frame_at,
            moved,
        }
    }
}

/// The output frames a reading says were made: none unless running or done.
pub fn frames(s: &RunSt) -> u64 {
    match s {
        RunSt::Running { done, .. } => *done,
        RunSt::Done { frames, .. } => *frames,
        _ => 0,
    }
}

/// A render job's state as its readers show it ([`Renders::prog`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Prog {
    /// `queued`, `running`, `done`, `failed`, `cancelled`, as for downloads
    pub state: &'static str,
    /// unfinished only: `gpu_starting` (no lane yet), `waiting` (queued in its lane), `setting_up`, `compiling` (the
    /// first RAFT flow), `frames`, `finishing`
    pub phase: Option<&'static str>,
    /// output frames there, and in all
    pub done: u64,
    pub total: u64,
    /// frames/s and the seconds left while it runs; frames/s over the run once done
    pub fps: Option<f64>,
    pub eta: Option<f64>,
    /// pictures that could not be decoded and repeat the one before (`allow_frozen`)
    pub concealed: u64,
    /// the file's bytes, done
    pub bytes: Option<u64>,
    /// since it was made while unfinished, the run's once finished (none: its host crashed)
    pub secs: Option<f64>,
    /// why it failed, or why it was cancelled when not by a cancel
    pub err: String,
}

impl Prog {
    /// Nothing of it runs or waits any more.
    pub fn finished(&self) -> bool {
        !matches!(self.state, "queued" | "running")
    }
}

fn phase(p: Phase) -> &'static str {
    match p {
        Phase::Setup => "setting_up",
        Phase::Compiling => "compiling",
        Phase::Frames => "frames",
        Phase::Finishing => "finishing",
    }
}

/// The render jobs, with those kept from earlier hosts.
#[derive(Default)]
pub struct Renders {
    /// every one, as made; one this host ended moved last, so the finished go in the order they ended
    pub v: Vec<RJob>,
    /// where they are kept
    file: Option<PathBuf>,
    /// the file's text as read or last written
    saved: String,
    /// the host ends: no job is reserved any more
    pub closing: bool,
}

impl Renders {
    /// The records kept in `file` (none when it is missing), and how many of its lines did not parse (left out).
    /// Fails when the file is there but cannot be read: it is not taken over.
    pub fn load(file: PathBuf) -> io::Result<(Renders, usize)> {
        let (v, bad, saved) = job::lines(&file)?;
        let r = Renders {
            v,
            file: Some(file),
            saved,
            closing: false,
        };
        Ok((r, bad))
    }

    /// Writes the file unless it holds this already, through `<file>.new` renamed into place. Err when a record is
    /// left out (a path that is no UTF-8) or the file can't be written: a reservation fails then.
    pub fn save(&mut self) -> io::Result<()> {
        let Some(f) = &self.file else { return Ok(()) };
        job::write(
            f,
            HEAD,
            &self.v,
            |r| &r.h,
            &mut self.saved,
            "renders not kept",
        )
    }

    /// Saves, then keeps whatever changes later in memory only (the file is another host's from here on).
    pub fn close(&mut self) -> io::Result<()> {
        let r = self.save();
        self.file = None;
        r
    }

    /// Where they are kept; none: in memory only (another host's lock, an unreadable file, or closed).
    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    /// A new record under handle `h` ([`crate::job::Jobs::handle`]).
    pub fn add(&mut self, h: String, mut r: RJob) {
        r.h = h;
        self.v.push(r)
    }

    /// Record `i`'s state: its end once written; else the poller's snapshot, a run that ended and is not settled yet
    /// (or a still whose preview is to come) still finishing; with neither, this host's job before its lane
    /// (`gpu_starting`: [`Renders::settle`] leaves no earlier host's record without an end).
    pub fn prog(&self, i: usize) -> Prog {
        let r = &self.v[i];
        let age = SystemTime::now().duration_since(r.at);
        let base = Prog {
            state: "queued",
            phase: Some("gpu_starting"),
            done: 0,
            total: r.frames,
            fps: None,
            eta: None,
            concealed: 0,
            bytes: None,
            secs: Some(age.map_or(0.0, |d| d.as_secs_f64())),
            err: String::new(),
        };
        if let Some(e) = &r.end {
            let state = match e.state.as_str() {
                "done" => "done",
                "failed" => "failed",
                _ => "cancelled",
            };
            let ran = e.secs > 0.0;
            return Prog {
                state,
                phase: None,
                done: e.frames,
                fps: (ran && state == "done").then(|| e.frames as f64 / e.secs),
                concealed: e.concealed,
                bytes: (state == "done").then_some(e.bytes),
                secs: ran.then_some(e.secs),
                err: e.err.clone(),
                ..base
            };
        }
        let Some(s) = &r.live else { return base };
        let ending = Prog {
            state: "running",
            phase: Some("finishing"),
            ..base.clone()
        };
        match &s.st {
            RunSt::Queued => Prog {
                phase: Some("waiting"),
                ..base
            },
            RunSt::Running {
                phase: p,
                done,
                total,
                fps,
                eta,
                concealed,
            } => Prog {
                state: "running",
                phase: Some(phase(*p)),
                done: *done,
                total: if *total > 0 { *total } else { r.frames },
                fps: Some(*fps),
                eta: Some(*eta),
                concealed: *concealed,
                ..base
            },
            RunSt::Done {
                frames, concealed, ..
            } => Prog {
                done: *frames,
                concealed: *concealed,
                ..ending
            },
            RunSt::Failed(_) | RunSt::Cancelled => ending,
        }
    }

    /// Ends what an earlier host left unfinished (a crash, a kill), by the identity of each record's `.part` only
    /// ([`media::part::settle`]): the published file reads done (a `.part` name the link fallback left beside it
    /// removed), anything else cancelled, its `.part` removed if it is still that file. A record with no identity (a
    /// crash in the µs before it was saved) or on FAT/exFAT (file numbers can change across mounts) unlinks nothing:
    /// its `.part`, if there, is left and said. Returns what to log. Under the files' lock, before [`Renders::prune`].
    pub fn settle(&mut self) -> Vec<String> {
        let (mut said, mut n) = (vec![], [0; 2]);
        for r in self.v.iter_mut().filter(|r| r.end.is_none()) {
            let part = media::part::of(&r.dst);
            let fat = r.dst.parent().is_some_and(media::part::fat);
            let why = if r.cancelled { "" } else { GONE };
            let end = match (r.part, fat) {
                (Some(id), false) => match media::part::settle(&part, &r.dst, &id) {
                    Settled::Done(b) => End::done(r.frames, 0.0, b, 0),
                    Settled::Gone => End::cancelled(why),
                },
                (id, _) => {
                    let w = match id {
                        None => "no identity recorded",
                        Some(_) => "on FAT or exFAT, whose file numbers can change across mounts",
                    };
                    if fs::symlink_metadata(&part).is_ok() {
                        said.push(format!("{}: left ({w})", part.display()))
                    }
                    End::cancelled(why)
                }
            };
            n[(end.state != "done") as usize] += 1;
            r.end = Some(end);
        }
        if n != [0, 0] {
            said.push(format!(
                "render jobs an earlier host left unfinished: {} done, {} cancelled",
                n[0], n[1]
            ))
        }
        said
    }

    /// Record `h`'s end written, as its host settles it: it moves last, and all but the [`KEEP`] finished last go
    /// ([`Renders::prune`]), so a host's own records stay bounded too.
    pub fn end(&mut self, h: &str, end: End) {
        if let Some(i) = self.v.iter().position(|r| r.h == h) {
            let mut r = self.v.remove(i);
            r.end = Some(end);
            self.v.push(r);
            self.prune()
        }
    }

    /// Drops all but every unfinished record and the newest [`KEEP`] finished ones.
    pub fn prune(&mut self) {
        let fin = self.v.iter().filter(|r| r.end.is_some()).count();
        let mut i = 0;
        self.v.retain(|r| {
            r.end.is_none() || {
                i += 1;
                fin - i < KEEP
            }
        });
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    /// An unfinished export into `dst`, its handle unset.
    pub(crate) fn rjob(dst: &str) -> RJob {
        RJob {
            h: String::new(),
            kind: Kind::Export,
            at: SystemTime::now(),
            src: vec!["/v/VID_20260117_141148_00_635.insv".into()],
            dst: dst.into(),
            part: None,
            lane: Lane::Export,
            settings: json!({ "preset": "pano57", "raft": true }),
            frames: 175,
            bytes: 92_000_000,
            cancelled: false,
            end: None,
            live: None,
        }
    }

    fn tmp(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-r{n}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn snap(st: RunSt) -> Option<Snap> {
        let t = Instant::now();
        Some(Snap {
            st,
            at: t,
            last_frame_at: t,
            moved: t,
        })
    }

    /// A reading carries the last frame's time on until a frame comes (before the first, the job's making), and the
    /// last move's until the run begins, changes phase or makes a frame.
    #[test]
    fn snaps() {
        let t0 = Instant::now();
        let made = SystemTime::now() - Duration::from_secs(10);
        let at = |s: u64| t0 + Duration::from_secs(s);
        let run = |phase, done| RunSt::Running {
            phase,
            done,
            total: 10,
            fps: 1.0,
            eta: 1.0,
            concealed: 0,
        };
        let a = Snap::after(None, RunSt::Queued, at(1), made);
        let age = at(1).duration_since(a.last_frame_at);
        assert!(age >= Duration::from_secs(10) && age < Duration::from_secs(12));
        assert_eq!(a.moved, at(1));
        let b = Snap::after(Some(&a), RunSt::Queued, at(2), made);
        assert_eq!(
            (b.last_frame_at, b.moved, b.at),
            (a.last_frame_at, at(1), at(2))
        );
        let c = Snap::after(Some(&b), run(Phase::Setup, 0), at(3), made);
        assert_eq!((c.last_frame_at, c.moved), (a.last_frame_at, at(3)));
        let d = Snap::after(Some(&c), run(Phase::Setup, 0), at(4), made);
        assert_eq!(d.moved, at(3));
        let e = Snap::after(Some(&d), run(Phase::Compiling, 0), at(5), made);
        assert_eq!((e.last_frame_at, e.moved), (a.last_frame_at, at(5)));
        let f = Snap::after(Some(&e), run(Phase::Frames, 1), at(6), made);
        assert_eq!((f.last_frame_at, f.moved), (at(6), at(6)));
        let g = Snap::after(Some(&f), run(Phase::Frames, 1), at(7), made);
        assert_eq!((g.last_frame_at, g.moved), (at(6), at(6)));
        let h = Snap::after(Some(&g), run(Phase::Frames, 2), at(8), made);
        assert_eq!((h.last_frame_at, h.moved), (at(8), at(8)));
    }

    /// A later host reads the records as saved (identity, end, cancel; the snapshot is memory only), lines that do
    /// not parse left out; a save that changes nothing leaves the file; a closed set saves no more; a record whose
    /// path is no UTF-8 is left out and said.
    #[test]
    fn kept() {
        let d = tmp("kept");
        let f = d.join("mcp-renders");
        let (mut r, bad) = Renders::load(f.clone()).unwrap();
        assert!(r.v.is_empty() && bad == 0 && !f.exists() && r.file() == Some(&*f));
        let id = Id {
            dev: 16_777_232,
            ino: 4_242_424_242,
            birth: Some(1_762_340_000_123_456_789),
        };
        let mut a = rjob("/agent/VID_20260117_141148_00_635_pano57.mp4");
        (a.part, a.cancelled) = (Some(id), true);
        a.live = snap(RunSt::Queued);
        r.add("a.0001-1".into(), a);
        let mut b = rjob("/agent/VID_20260117_141148_00_635_3.00s_360_7680.jpg");
        (b.kind, b.lane, b.frames) = (Kind::Still, Lane::Still, 1);
        b.end = Some(End::done(1, 3.4, 5_123_456, 1));
        r.add("a.0001-3".into(), b);
        r.save().unwrap();
        let t = fs::read_to_string(&f).unwrap();
        assert!(t.starts_with(HEAD) && t.lines().count() == 3, "{t}");
        assert!(
            !t.contains("live") && t.contains(r#""kind":"still""#),
            "{t}"
        );
        fs::write(
            &f,
            [t.as_bytes(), b"{\"job\": 1}\nnot json\n\xff\xfe\n"].concat(),
        )
        .unwrap();
        let (k, bad) = Renders::load(f.clone()).unwrap();
        assert_eq!(bad, 3);
        let ms = |t: SystemTime| {
            t.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        };
        let (x, y) = (&k.v[0], &k.v[1]);
        assert_eq!((x.h.as_str(), y.h.as_str()), ("a.0001-1", "a.0001-3"));
        assert_eq!(
            (x.part, x.cancelled, x.end.as_ref()),
            (Some(id), true, None)
        );
        assert!(x.live.is_none());
        assert_eq!(ms(x.at), ms(r.v[0].at));
        assert_eq!(
            (x.kind, x.lane, &x.src, &x.dst),
            (Kind::Export, Lane::Export, &r.v[0].src, &r.v[0].dst)
        );
        assert_eq!(
            (x.settings["preset"].as_str(), x.frames, x.bytes),
            (Some("pano57"), 175, 92_000_000)
        );
        assert_eq!(
            (y.kind, y.lane, y.part, y.cancelled),
            (Kind::Still, Lane::Still, None, false)
        );
        assert_eq!(y.end, Some(End::done(1, 3.4, 5_123_456, 1)));
        // written as read, the bad lines gone; unchanged, the file stays
        let (mut k, _) = Renders::load(f.clone()).unwrap();
        k.save().unwrap();
        assert_eq!(fs::read_to_string(&f).unwrap(), t);
        let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        let file = || fs::File::options().write(true).open(&f).unwrap();
        let (mut k, _) = Renders::load(f.clone()).unwrap();
        file().set_modified(old).unwrap();
        k.save().unwrap();
        assert_eq!(file().metadata().unwrap().modified().unwrap(), old);
        // closed: saved once more, then not
        k.v[0].end = Some(End::cancelled(GONE));
        k.close().unwrap();
        assert!(k.file().is_none());
        let (j, _) = Renders::load(f.clone()).unwrap();
        assert_eq!(j.v[0].end, Some(End::cancelled(GONE)));
        k.v.clear();
        k.save().unwrap();
        assert_eq!(
            Renders::load(f.clone()).unwrap().0.v.len(),
            2,
            "closed: not saved"
        );
        // a record whose path is no UTF-8 is left out and said; the others are kept
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let (mut k, _) = Renders::load(f.clone()).unwrap();
            let mut odd = rjob("/x");
            odd.dst = d.join(std::ffi::OsStr::from_bytes(b"x\xff.mp4"));
            k.add("a.0001-4".into(), odd);
            let e = k.save().unwrap_err().to_string();
            assert!(
                e.contains("renders not kept") && e.contains("a.0001-4"),
                "{e}"
            );
            assert_eq!(Renders::load(f.clone()).unwrap().0.v.len(), 2);
        }
        // there but unreadable: an error, not an empty set to write over
        #[cfg(unix)]
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            let mode = |m| fs::set_permissions(&f, fs::Permissions::from_mode(m)).unwrap();
            mode(0o000);
            assert!(Renders::load(f.clone()).is_err());
            mode(0o600);
        }
        fs::remove_dir_all(&d).unwrap();
    }

    /// The pruning keeps every unfinished record and the newest finished ones, in their order.
    #[test]
    fn prune() {
        let mut r = Renders::default();
        for i in 0..KEEP + 10 {
            let mut j = rjob("/agent/x.mp4");
            j.end = Some(End::cancelled(""));
            r.add(format!("h-{i}"), j);
            // three unfinished among them: the first, one in the middle, the last
            if [0, 50, KEEP + 9].contains(&i) {
                r.add(format!("u-{i}"), rjob("/agent/y.mp4"))
            }
        }
        r.prune();
        assert_eq!(r.v.len(), KEEP + 3);
        let h: Vec<&str> = r.v.iter().map(|j| j.h.as_str()).collect();
        assert_eq!(&h[..3], ["u-0", "h-10", "h-11"]);
        assert!(h.contains(&"u-50") && h.last() == Some(&"u-109"), "{h:?}");
        assert_eq!(h[h.len() - 2], "h-109");
        // under KEEP: nothing goes
        r.prune();
        assert_eq!(r.v.len(), KEEP + 3);
    }

    /// What readers show: the end once written, over any snapshot; else the snapshot, a run ended but not settled
    /// still finishing; with neither, waiting for the GPU.
    #[test]
    fn progress() {
        let mut r = Renders::default();
        r.add("h-1".into(), rjob("/agent/x.mp4"));
        let p = |r: &Renders| {
            let p = r.prog(0);
            (p.state, p.phase, p.done, p.total, p.finished())
        };
        assert_eq!(p(&r), ("queued", Some("gpu_starting"), 0, 175, false));
        assert!(r.prog(0).secs.is_some_and(|s| s < 5.0));
        r.v[0].live = snap(RunSt::Queued);
        assert_eq!(p(&r), ("queued", Some("waiting"), 0, 175, false));
        r.v[0].live = snap(RunSt::Running {
            phase: Phase::Compiling,
            done: 0,
            total: 0,
            fps: 0.0,
            eta: 0.0,
            concealed: 0,
        });
        assert_eq!(p(&r), ("running", Some("compiling"), 0, 175, false));
        r.v[0].live = snap(RunSt::Running {
            phase: Phase::Frames,
            done: 120,
            total: 175,
            fps: 12.3,
            eta: 4.5,
            concealed: 2,
        });
        let x = r.prog(0);
        assert_eq!(
            (x.state, x.phase, x.done, x.fps, x.eta, x.concealed),
            ("running", Some("frames"), 120, Some(12.3), Some(4.5), 2)
        );
        // the run ended: finishing until the poller has settled it
        for st in [RunSt::Cancelled, RunSt::Failed("no".into())] {
            r.v[0].live = snap(st);
            assert_eq!(p(&r).0, "running");
            assert_eq!(p(&r).1, Some("finishing"));
        }
        r.v[0].live = snap(RunSt::Done {
            frames: 175,
            secs: 14.2,
            bytes: 9,
            concealed: 0,
        });
        assert_eq!(p(&r), ("running", Some("finishing"), 175, 175, false));
        r.v[0].end = Some(End::done(175, 14.0, 9_000, 0));
        let x = r.prog(0);
        assert_eq!(
            (x.state, x.phase, x.done, x.bytes, x.secs, x.finished()),
            ("done", None, 175, Some(9_000), Some(14.0), true)
        );
        assert_eq!(x.fps, Some(12.5));
        r.v[0].live = None;
        r.v[0].end = Some(End::cancelled(GONE));
        let x = r.prog(0);
        assert_eq!(
            (x.state, x.bytes, x.secs, x.err.as_str()),
            ("cancelled", None, None, GONE)
        );
        r.v[0].end.as_mut().unwrap().state = "failed".into();
        assert_eq!(r.prog(0).state, "failed");
    }

    /// A crashed host's unfinished records end by identity: its own `.part` removed (cancelled), one of another
    /// identity at the name left (cancelled), a published file done (and a link fallback's second name gone), no
    /// identity: nothing removed, said. A download's `.part` beside them and a settled record's files stay.
    #[test]
    fn settle() {
        let d = tmp("settle");
        let p = |n: &str| d.join(n).display().to_string();
        let mut r = Renders::default();
        let mut add = |h: &str, n: &str, part: Option<Id>| {
            let mut j = rjob(&p(n));
            j.part = part;
            r.add(h.into(), j);
        };
        // its own `.part`
        let a = media::part::create(&d.join("a.mp4")).unwrap().id;
        add("h-1", "a.mp4", Some(a));
        // another file's identity at the name: someone else's `.part` there now
        let other = media::part::create(&d.join("other.jpg")).unwrap().id;
        fs::write(media::part::of(&d.join("b.mp4")), b"the user's").unwrap();
        add("h-2", "b.mp4", Some(other));
        // published, the record not yet saved
        let c = media::part::create(&d.join("c.mp4")).unwrap().id;
        fs::write(media::part::of(&d.join("c.mp4")), b"12345").unwrap();
        media::part::publish(&media::part::of(&d.join("c.mp4")), &d.join("c.mp4")).unwrap();
        add("h-3", "c.mp4", Some(c));
        // the link fallback's crash: both names
        let e = media::part::create(&d.join("e.mp4")).unwrap().id;
        fs::write(media::part::of(&d.join("e.mp4")), b"1234567").unwrap();
        fs::hard_link(media::part::of(&d.join("e.mp4")), d.join("e.mp4")).unwrap();
        add("h-4", "e.mp4", Some(e));
        // no identity: an empty `.part` left
        fs::write(media::part::of(&d.join("f.mp4")), b"").unwrap();
        add("h-5", "f.mp4", None);
        // cancelled by its client, its `.part` its own
        let g = media::part::create(&d.join("g.mp4")).unwrap().id;
        add("h-6", "g.mp4", Some(g));
        // settled already: its files stay whatever they are
        let k = media::part::create(&d.join("k.mp4")).unwrap().id;
        add("h-7", "k.mp4", Some(k));
        r.v[5].cancelled = true;
        r.v[6].end = Some(End::cancelled(""));
        // a download's `.part` beside them
        let dl = media::part::of(&d.join("VID_20260116_152935_00_603.insv"));
        fs::write(&dl, b"half").unwrap();
        let said = r.settle();
        let st: Vec<(&str, &str)> =
            r.v.iter()
                .map(|j| j.end.as_ref().unwrap())
                .map(|e| (e.state.as_str(), e.err.as_str()))
                .collect();
        assert_eq!(
            st,
            [
                ("cancelled", GONE),
                ("cancelled", GONE),
                ("done", ""),
                ("done", ""),
                ("cancelled", GONE),
                ("cancelled", ""),
                ("cancelled", ""),
            ]
        );
        let there = |n: &str| fs::symlink_metadata(d.join(n)).is_ok();
        assert!(!there("a.mp4.part") && !there("a.mp4"));
        assert_eq!(fs::read(d.join("b.mp4.part")).unwrap(), b"the user's");
        assert!(there("other.jpg.part"));
        assert_eq!(
            (r.v[2].end.as_ref().unwrap().bytes, there("c.mp4")),
            (5, true)
        );
        assert_eq!(
            (
                r.v[3].end.as_ref().unwrap().bytes,
                there("e.mp4"),
                there("e.mp4.part")
            ),
            (7, true, false)
        );
        assert!(there("f.mp4.part") && !there("g.mp4.part") && there("k.mp4.part") && dl.exists());
        assert!(
            said.iter()
                .any(|s| s.contains("f.mp4.part: left (no identity recorded)")),
            "{said:?}"
        );
        assert_eq!(
            said.last().unwrap(),
            "render jobs an earlier host left unfinished: 2 done, 4 cancelled"
        );
        assert_eq!(said.len(), 2, "{said:?}");
        // nothing left to settle
        assert!(r.settle().is_empty());
        assert_eq!(r.prog(2).state, "done");
        fs::remove_dir_all(&d).unwrap();
    }
}
