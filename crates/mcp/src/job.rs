//! Jobs: what one call started and a client follows with `job_status` / `job_wait` / `job_cancel`. Two kinds: a
//! download (an item's files in the host's download queue) and a render (a still or an export, [`crate::renders`],
//! kept in a file of its own). Jobs of one item share a file's download while it waits or runs; one that stopped,
//! failed, or finished and went is the job's alone that starts it again, and the older jobs show it as it was then
//! ([`File::was`]). A handle is `<host id>-<n>`, of the host that made the job, numbered over both kinds
//! ([`Jobs::handle`]); the host id (the pid in base 36 and a random part) differs between hosts. The jobs are kept
//! beside the queue, a JSON object a line ([`Jobs::load`], [`Jobs::save`]), so a later host follows an earlier one's
//! under their handles; it keeps every unfinished job and the newest [`KEEP`] ([`Jobs::prune`]).

use crate::renders::Renders;
use cam::dl::{self, St};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};
use std::{
    hash::{BuildHasher, Hasher},
    io,
    path::{Path, PathBuf},
    sync::atomic::Ordering::Relaxed,
    time::{Duration, SystemTime},
};

/// Jobs a host keeps of those before it: every unfinished one and the newest this many.
pub const KEEP: usize = 100;

/// The jobs file's first line.
const HEAD: &str = "# x5desk mcp jobs: one JSON object a line";

fn b36(mut n: u64) -> String {
    let d = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut s = vec![];
    loop {
        s.push(d[(n % 36) as usize]);
        n /= 36;
        if n == 0 {
            break;
        }
    }
    s.reverse();
    String::from_utf8(s).unwrap()
}

/// 64 random bits: std's hasher keys come from the system's random source.
pub fn random() -> u64 {
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    h.finish()
}

/// This process's host id: the pid in base 36, a dot, 4 random base-36 digits (`1x3k.9fz2`).
pub fn host_id() -> String {
    format!(
        "{}.{:0>4}",
        b36(std::process::id() as u64),
        b36(random() % 36u64.pow(4))
    )
}

/// One file of a job: the queue's URI and where it goes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct File {
    /// a camera URI, or `file://` and a card's file
    pub uri: String,
    /// in the agent folder
    pub dst: PathBuf,
    /// none while the queue's entry for it is this job's (shared with the jobs that joined it while it waited or
    /// ran); its state when a later job started it again (it had stopped, failed, or finished and gone): it shows so
    /// from then on, and this job's cancel leaves it
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub was: Option<Part>,
    /// the bytes the camera sent for it, once its download finished (a local copy checked against it later)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

impl File {
    /// A file of a new job: its entry its own.
    pub fn new(uri: String, dst: PathBuf) -> File {
        File {
            uri,
            dst,
            was: None,
            size: None,
        }
    }

    /// The queue's entry is still this job's: no later job started it again.
    pub fn live(&self) -> bool {
        self.was.is_none()
    }
}

/// [`SystemTime`] as milliseconds since 1970.
pub(crate) mod ms {
    use super::*;

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let d = t.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
        s.serialize_u64(d.as_millis() as u64)
    }

    /// A time the system cannot hold (Windows: past about 30828) is an error: the line does not parse.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let ms = Duration::from_millis(u64::deserialize(d)?);
        (SystemTime::UNIX_EPOCH.checked_add(ms))
            .ok_or_else(|| serde::de::Error::custom("at_ms out of range"))
    }
}

/// One call's downloads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    /// its handle, `<host id>-<n>`: of the host that made it
    #[serde(rename = "job")]
    pub h: String,
    /// the album keys it was made for
    pub keys: Vec<String>,
    /// what it downloads
    pub files: Vec<File>,
    /// when it was made
    #[serde(rename = "at_ms", with = "ms")]
    pub at: SystemTime,
    /// `job_cancel` was called on it: a file of it still queued or running (shared with a job that still wants it)
    /// counts as cancelled
    #[serde(default)]
    pub cancelled: bool,
}

/// A file's state in the queue.
#[derive(Clone, Debug, Serialize)]
pub struct Part {
    /// where it goes
    pub dst: PathBuf,
    /// as for [`Prog::state`]
    pub state: &'static str,
    /// bytes there
    pub done: u64,
    /// bytes of the whole file (0 until known)
    pub total: u64,
    /// why it failed
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub err: String,
}

/// A [`Part`] as kept.
#[derive(Deserialize)]
struct Kept {
    dst: PathBuf,
    state: String,
    done: u64,
    total: u64,
    #[serde(default)]
    err: String,
}

impl<'de> Deserialize<'de> for Part {
    /// A state that waited or ran no longer does for the job it was kept for: it reads as cancelled.
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Part, D::Error> {
        let k = Kept::deserialize(d)?;
        let state = match k.state.as_str() {
            "done" => "done",
            "failed" => "failed",
            _ => "cancelled",
        };
        Ok(Part {
            dst: k.dst,
            state,
            done: k.done,
            total: k.total,
            err: k.err,
        })
    }
}

/// A job's state: `queued`, `running`, `done`, `failed` (some file failed, none still to run), `cancelled`.
#[derive(Clone, Debug)]
pub struct Prog {
    /// `queued`, `running`, `done`, `failed`, `cancelled` (any running: running; else any queued: queued)
    pub state: &'static str,
    /// bytes there, over its files
    pub done: u64,
    /// bytes in all (of the files whose size is known)
    pub total: u64,
    /// per file
    pub parts: Vec<Part>,
}

impl Prog {
    /// Nothing of it runs or waits any more.
    pub fn finished(&self) -> bool {
        !matches!(self.state, "queued" | "running")
    }
}

/// File `f`'s state in the queue `q`, for a job that is `cancelled` or not: one the queue dropped counts as done when
/// it is there; once cancelled, one that runs or waits counts as cancelled.
fn part(f: &File, cancelled: bool, q: &dl::Queue) -> Part {
    let Some(d) = q.v.iter().find(|d| d.dst == f.dst) else {
        return Part {
            dst: f.dst.clone(),
            state: if f.dst.exists() { "done" } else { "failed" },
            done: 0,
            total: 0,
            err: String::new(),
        };
    };
    let (done, total) = (d.prog.done.load(Relaxed), d.prog.total.load(Relaxed));
    let state = match d.st {
        St::Queued | St::Running if cancelled => "cancelled",
        St::Queued => "queued",
        St::Running => "running",
        St::Paused => "cancelled",
        St::Done => "done",
        St::Failed => "failed",
    };
    Part {
        dst: f.dst.clone(),
        state,
        done: if d.st == St::Done {
            total.max(done)
        } else {
            done
        },
        total,
        err: d.err.clone(),
    }
}

impl Job {
    /// Its state from the queue `q` ([`part`]); a file a later job started again shows as it was then (that
    /// transfer is the later job's).
    pub fn prog(&self, q: &dl::Queue) -> Prog {
        let parts: Vec<Part> = (self.files.iter())
            .map(|f| f.was.clone().unwrap_or_else(|| part(f, self.cancelled, q)))
            .collect();
        let any = |s: &str| parts.iter().any(|p| p.state == s);
        let state = if any("running") {
            "running"
        } else if any("queued") {
            "queued"
        } else if parts.iter().all(|p| p.state == "done") {
            "done"
        } else if any("failed") {
            "failed"
        } else {
            "cancelled"
        };
        Prog {
            state,
            done: parts.iter().map(|p| p.done).sum(),
            total: parts.iter().map(|p| p.total).sum(),
            parts,
        }
    }
}

/// The objects of `file`, one JSON object a line (none when it is missing), how many of its lines did not parse (no
/// UTF-8, no object of its kind: left out), and its text as read. Fails when the file is there but cannot be read: it
/// is not taken over.
pub(crate) fn lines<T: DeserializeOwned>(file: &Path) -> io::Result<(Vec<T>, usize, String)> {
    let b = match std::fs::read(file) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => vec![],
        Err(e) => return Err(e),
    };
    let (mut v, mut bad) = (vec![], 0);
    for l in b.split(|&c| c == b'\n') {
        match std::str::from_utf8(l) {
            Ok(l) if l.trim().is_empty() || l.starts_with('#') => {}
            Ok(l) => match serde_json::from_str::<T>(l) {
                Ok(j) => v.push(j),
                Err(_) => bad += 1,
            },
            Err(_) => bad += 1,
        }
    }
    Ok((v, bad, String::from_utf8(b).unwrap_or_default()))
}

/// Writes `v` to `file` under `head` unless it holds that already (`saved`, the text as read or last written), through
/// `<file>.new` renamed into place. One that does not serialize (a path that is no UTF-8) is left out and said by its
/// handle (`h`), after `what`.
pub(crate) fn write<T: Serialize>(
    file: &Path,
    head: &str,
    v: &[T],
    h: impl Fn(&T) -> &str,
    saved: &mut String,
    what: &str,
) -> io::Result<()> {
    let (mut s, mut out) = (format!("{head}\n"), vec![]);
    for j in v {
        match serde_json::to_string(j) {
            Ok(l) => {
                s += &l;
                s.push('\n')
            }
            Err(_) => out.push(h(j)),
        }
    }
    if s != *saved {
        let t = file.with_extension("new");
        std::fs::write(&t, &s).and_then(|_| std::fs::rename(&t, file))?;
        *saved = s;
    }
    match out.is_empty() {
        true => Ok(()),
        false => Err(io::Error::other(format!("{what}: {}", out.join(", ")))),
    }
}

/// Which kind of job a handle names, and where it is: in [`Jobs::v`] or in [`Renders::v`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Dl(usize),
    Render(usize),
}

/// The host's jobs, with those kept from earlier hosts.
pub struct Jobs {
    /// the host id, the prefix of its handles
    pub host: String,
    /// every download job, oldest first
    pub v: Vec<Job>,
    /// the render jobs, in `mcp-renders` beside these, their handles from the same numbering
    pub r: Renders,
    /// this host's jobs so far, of both kinds
    n: u64,
    /// where they are kept
    file: Option<PathBuf>,
    /// the file's text as read or last written
    saved: String,
}

impl Jobs {
    /// No jobs yet, on host `host`, kept nowhere.
    pub fn new(host: String) -> Jobs {
        Jobs {
            host,
            v: vec![],
            r: Renders::default(),
            n: 0,
            file: None,
            saved: String::new(),
        }
    }

    /// The jobs kept in `file` (none when it is missing), for host `host`, and how many of its lines did not parse
    /// (no UTF-8, no job: left out). Fails when the file is there but cannot be read: it is not taken over. No render
    /// jobs: [`Renders::load`] reads their file.
    pub fn load(host: String, file: PathBuf) -> io::Result<(Jobs, usize)> {
        let (v, bad, saved) = lines(&file)?;
        let j = Jobs {
            host,
            v,
            r: Renders::default(),
            n: 0,
            file: Some(file),
            saved,
        };
        Ok((j, bad))
    }

    /// Writes the file unless it holds this already, through `<file>.new` renamed into place. A job that does not
    /// serialize (a path that is no UTF-8) is left out and said. The render jobs are saved apart ([`Renders::save`]).
    pub fn save(&mut self) -> io::Result<()> {
        let Some(f) = &self.file else { return Ok(()) };
        write(f, HEAD, &self.v, |j| &j.h, &mut self.saved, "jobs not kept")
    }

    /// Saves both files, then keeps whatever changes later in memory only (the files are another host's from here
    /// on). The first error, once both are closed.
    pub fn close(&mut self) -> io::Result<()> {
        let r = self.save();
        self.file = None;
        r.and(self.r.close())
    }

    /// The next handle of this host, for a job of either kind.
    pub fn handle(&mut self) -> String {
        self.n += 1;
        format!("{}-{}", self.host, self.n)
    }

    /// A new job for album `keys`; returns its handle.
    pub fn add(&mut self, keys: Vec<String>, files: Vec<File>) -> String {
        let h = self.handle();
        self.v.push(Job {
            h: h.clone(),
            keys,
            files,
            at: SystemTime::now(),
            cancelled: false,
        });
        h
    }

    /// Which job a handle names, a download or a render. One that ended and is not kept (another host's, or one of
    /// this host's pruned) is told apart.
    pub fn which(&self, h: &str) -> Result<Which, String> {
        let h = h.trim();
        if let Some(i) = self.v.iter().position(|j| j.h == h) {
            return Ok(Which::Dl(i));
        }
        if let Some(i) = self.r.v.iter().position(|r| r.h == h) {
            return Ok(Which::Render(i));
        }
        match parse(h)? {
            (id, n) if id == self.host && n > self.n => Err(format!("unknown job {h}")),
            (id, _) if id == self.host => Err(format!(
                "unknown job {h} (it has ended, and is not among those kept: job_list lists them)"
            )),
            _ => Err(format!(
                "unknown job {h} (its host has ended, and the job is not among those kept: job_list lists them)"
            )),
        }
    }

    /// The download job a handle names ([`Jobs::which`]).
    pub fn find(&self, h: &str) -> Result<&Job, String> {
        match self.which(h)? {
            Which::Dl(i) => Ok(&self.v[i]),
            Which::Render(_) => Err(format!("{} is a render job, no download", h.trim())),
        }
    }

    /// A new job starts the file at `dst` again (no download waits or runs for it): every job that had it keeps its
    /// state as `q` shows it now and leaves it to the new one. Call it before the queue changes.
    pub fn take(&mut self, dst: &Path, q: &dl::Queue) {
        for j in &mut self.v {
            let c = j.cancelled;
            for f in (j.files.iter_mut()).filter(|f| f.dst == dst && f.live()) {
                f.was = Some(part(f, c, q))
            }
        }
    }

    /// What cancelling job `i` (of [`Jobs::v`]) stops: its files whose entry is still its own, but none another job
    /// that is not cancelled shares (two calls for one item while it waits or runs).
    pub fn cut(&self, i: usize) -> Vec<PathBuf> {
        let Some(j) = self.v.get(i) else {
            return vec![];
        };
        let wanted = |d: &Path| {
            (self.v.iter().enumerate()).any(|(k, o)| {
                k != i && !o.cancelled && o.files.iter().any(|f| f.live() && f.dst == d)
            })
        };
        (j.files.iter())
            .filter(|f| f.live() && !wanted(&f.dst))
            .map(|f| f.dst.clone())
            .collect()
    }

    /// Drops what earlier hosts left that no longer matters: of the finished jobs all but the newest [`KEEP`] and
    /// those with a stopped file whose `.part` is there (its finished files keep their queue entries, so downloading
    /// the item again resumes it at its number), then the queue's entries no kept job has, unless they wait to run or
    /// are stopped with their `.part` there.
    pub fn prune(&mut self, q: &mut dl::Queue) {
        let stopped = |j: &Job| {
            (j.files.iter()).filter(|f| f.live()).any(|f| {
                (q.v.iter()).any(|d| {
                    d.dst == f.dst
                        && matches!(d.st, St::Paused | St::Failed)
                        && media::part::of(&d.dst).exists()
                })
            })
        };
        let (n, mut i) = (self.v.len(), 0);
        self.v.retain(|j| {
            i += 1;
            n - i < KEEP || !j.prog(q).finished() || stopped(j)
        });
        let has = |d: &Path| (self.v.iter()).any(|j| j.files.iter().any(|f| f.dst == d));
        q.v.retain(|d| match d.st {
            St::Queued | St::Running => true,
            _ if has(&d.dst) => true,
            St::Done => false,
            St::Paused | St::Failed => media::part::of(&d.dst).exists(),
        });
    }
}

/// The host id in handle `h` (`<host id>-<n>`).
fn parse(h: &str) -> Result<(&str, u64), String> {
    let bad = || format!("{h:?} is not a job handle (<host>-<n>)");
    let (id, n) = h.rsplit_once('-').ok_or_else(bad)?;
    match n.parse::<u64>() {
        Ok(n) if !id.is_empty() => Ok((id, n)),
        _ => Err(bad()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(n: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-{n}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn handles() {
        let h = host_id();
        let (pid, r) = h.split_once('.').unwrap();
        assert_eq!(
            u64::from_str_radix(pid, 36).unwrap(),
            std::process::id() as u64
        );
        assert_eq!(r.len(), 4);
        assert!(h.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.'));
        assert_eq!(b36(0), "0");
        assert_eq!(b36(36 * 36 + 35), "10z");
        let mut j = Jobs::new("1x3k.9fz2".into());
        let h1 = j.add(vec!["k".into()], vec![]);
        assert_eq!(h1, "1x3k.9fz2-1");
        assert_eq!(j.add(vec![], vec![]), "1x3k.9fz2-2");
        assert_eq!(j.find("1x3k.9fz2-1").unwrap().h, h1);
        assert_eq!(j.find(" 1x3k.9fz2-1 ").unwrap().h, h1);
        assert_eq!(j.which("1x3k.9fz2-2"), Ok(Which::Dl(1)));
        assert_eq!(
            j.find("1x3k.9fz2-3").unwrap_err(),
            "unknown job 1x3k.9fz2-3"
        );
        // one of this host's that is no more
        let k = j.v.remove(0);
        assert!(
            j.find(&h1)
                .unwrap_err()
                .contains("it has ended, and is not among those kept")
        );
        j.v.insert(0, k);
        assert!(
            j.find("1x3k.0000-1")
                .unwrap_err()
                .contains("its host has ended")
        );
        for h in ["", "1", "1x3k.9fz2", "1x3k.9fz2-x", "1x3k.9fz2-", "-1"] {
            assert!(j.find(h).unwrap_err().contains("not a job handle"), "{h}");
        }
        // an earlier host's job among them is found under its handle
        j.v.push(Job {
            h: "zz.0000-7".into(),
            ..j.v[0].clone()
        });
        assert_eq!(j.which("zz.0000-7"), Ok(Which::Dl(2)));
    }

    /// Downloads and renders share the host's numbering: a handle names one job of either kind, the unknown-job
    /// wording the same for both.
    #[test]
    fn shared() {
        use crate::renders::tests::rjob;
        let mut j = Jobs::new("1x3k.9fz2".into());
        assert_eq!(j.add(vec![], vec![]), "1x3k.9fz2-1");
        let h = j.handle();
        assert_eq!(h, "1x3k.9fz2-2");
        j.r.add(h.clone(), rjob("/a/b.mp4"));
        assert_eq!(j.add(vec![], vec![]), "1x3k.9fz2-3");
        let h4 = j.handle();
        j.r.add(h4, rjob("/a/c.jpg"));
        assert_eq!(j.r.v[0].h, h);
        assert_eq!(j.which(" 1x3k.9fz2-2 "), Ok(Which::Render(0)));
        assert_eq!(j.which("1x3k.9fz2-3"), Ok(Which::Dl(1)));
        assert_eq!(j.which("1x3k.9fz2-4"), Ok(Which::Render(1)));
        assert_eq!(
            j.which("1x3k.9fz2-5").unwrap_err(),
            "unknown job 1x3k.9fz2-5"
        );
        assert!(
            j.which("zz.0000-2")
                .unwrap_err()
                .contains("its host has ended")
        );
        assert!(j.which("x").unwrap_err().contains("not a job handle"));
        assert!(j.find("1x3k.9fz2-2").unwrap_err().contains("a render job"));
        assert_eq!(j.find("1x3k.9fz2-3").unwrap().h, "1x3k.9fz2-3");
        // an earlier host's render among them is found under its handle
        j.r.add("zz.0000-9".into(), rjob("/a/d.mp4"));
        assert_eq!(j.which("zz.0000-9"), Ok(Which::Render(2)));
    }

    #[test]
    fn states() {
        let mut q = dl::Queue::default();
        let d = std::env::temp_dir().join(format!("x5desk-mcp-job-{}", std::process::id()));
        let f = |n: &str| File::new(format!("/DCIM/Camera01/{n}"), d.join(n));
        let job = Job {
            h: "h-1".into(),
            keys: vec![],
            files: vec![f("a"), f("b")],
            at: SystemTime::now(),
            cancelled: false,
        };
        q.add("/DCIM/Camera01/a", d.join("a"));
        q.add("/DCIM/Camera01/b", d.join("b"));
        assert_eq!(job.prog(&q).state, "queued");
        q.v[0].st = St::Running;
        q.v[0].prog.total.store(10, Relaxed);
        q.v[0].prog.done.store(4, Relaxed);
        let p = job.prog(&q);
        assert_eq!(
            (p.state, p.done, p.total, p.finished()),
            ("running", 4, 10, false)
        );
        q.v[0].st = St::Done;
        q.cancel(1);
        assert_eq!(job.prog(&q).state, "cancelled");
        q.v[1].st = St::Failed;
        assert_eq!(job.prog(&q).state, "failed");
        q.v[1].st = St::Done;
        let p = job.prog(&q);
        assert_eq!((p.state, p.done, p.finished()), ("done", 10, true));
        // cancelled while another job's download of its file goes on
        let job = Job {
            cancelled: true,
            ..job
        };
        q.v[1].st = St::Running;
        let p = job.prog(&q);
        assert_eq!(
            (p.state, p.parts[1].state, p.finished()),
            ("cancelled", "cancelled", true)
        );
        q.v[1].st = St::Done;
        assert_eq!(job.prog(&q).state, "done", "its files there all the same");
    }

    /// Two jobs of a waiting download share it: cancelling one leaves it to the other. A job that starts a stopped or
    /// failed file again takes it over: the older ones show it as it was, and their cancels leave it.
    #[test]
    fn owners() {
        let mut q = dl::Queue::default();
        let d = std::env::temp_dir().join(format!("x5desk-mcp-owners-{}", std::process::id()));
        let (u, p) = ("/DCIM/Camera01/a", d.join("a"));
        let f = || vec![File::new(u.into(), p.clone())];
        let mut j = Jobs::new("h".into());
        q.add(u, p.clone());
        let (a, b) = (j.add(vec![], f()), j.add(vec![], f()));
        let at = |j: &Jobs, h: &str| match j.which(h) {
            Ok(Which::Dl(i)) => i,
            w => panic!("{h}: {w:?}"),
        };
        let (a, b) = (at(&j, &a), at(&j, &b));
        assert!(j.cut(b).is_empty(), "a still wants it");
        j.v[b].cancelled = true;
        assert_eq!(j.cut(a), std::slice::from_ref(&p));
        j.v[a].cancelled = true;
        q.cancel(0);
        // a third job resumes it
        j.take(&p, &q);
        q.add(u, p.clone());
        let c = j.add(vec![], f());
        let c = at(&j, &c);
        q.v[0].st = St::Running;
        for n in [a, b] {
            let x = j.v[n].prog(&q);
            assert_eq!((x.state, x.finished()), ("cancelled", true), "job {n}");
            assert!(j.cut(n).is_empty(), "job {n}: no longer its own");
        }
        assert_eq!(j.v[c].prog(&q).state, "running");
        assert_eq!(j.cut(c), std::slice::from_ref(&p));
        // it fails, a fourth job retries it: the third shows it failed
        (q.v[0].st, q.v[0].err) = (St::Failed, "disk full".into());
        j.take(&p, &q);
        q.add(u, p.clone());
        let e = j.add(vec![], f());
        let e = at(&j, &e);
        q.v[0].prog.done.store(5, Relaxed);
        let x = j.v[c].prog(&q);
        assert_eq!(
            (x.state, x.done, x.parts[0].err.as_str()),
            ("failed", 0, "disk full")
        );
        assert!(j.cut(c).is_empty());
        assert_eq!(j.v[e].prog(&q).state, "queued");
        assert_eq!(j.cut(e), [p]);
        assert!(j.cut(99).is_empty());
    }

    /// A later host reads the jobs as they were saved (a kept state that waited or ran reads as cancelled), lines that
    /// do not parse left out; a save that changes nothing leaves the file; a closed set saves no more.
    #[test]
    fn kept() {
        let d = tmp("kept");
        let f = d.join("mcp-jobs");
        let (mut j, bad) = Jobs::load("a.0001".into(), f.clone()).unwrap();
        assert!(j.v.is_empty() && bad == 0 && !f.exists());
        let x = |n: &str| File::new(format!("/DCIM/Camera01/{n}"), d.join(n));
        let h1 = j.add(vec!["k1".into()], vec![x("a")]);
        let mut b = x("b");
        b.was = Some(Part {
            dst: b.dst.clone(),
            state: "running",
            done: 3,
            total: 9,
            err: "no".into(),
        });
        let h2 = j.add(vec!["k2".into(), "k3".into()], vec![b, x("c")]);
        j.v[1].cancelled = true;
        j.save().unwrap();
        let t = std::fs::read_to_string(&f).unwrap();
        assert!(
            t.starts_with("# x5desk mcp jobs") && t.lines().count() == 3,
            "{t}"
        );
        std::fs::write(
            &f,
            [t.as_bytes(), b"{\"job\": 1}\nnot json\n\xff\xfe\n"].concat(),
        )
        .unwrap();
        let (mut k, bad) = Jobs::load("b.0002".into(), f.clone()).unwrap();
        assert_eq!(bad, 3);
        assert_eq!(
            k.v.iter().map(|j| j.h.as_str()).collect::<Vec<_>>(),
            [&h1, &h2]
        );
        let ms = |t: SystemTime| {
            t.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        };
        assert_eq!(ms(k.v[0].at), ms(j.v[0].at));
        assert_eq!(
            (k.v[1].keys.len(), k.v[1].cancelled, k.v[0].cancelled),
            (2, true, false)
        );
        let w = k.v[1].files[0].was.as_ref().unwrap();
        assert_eq!(
            (w.state, w.done, w.total, w.err.as_str()),
            ("cancelled", 3, 9, "no")
        );
        assert!(k.v[1].files[1].live() && k.v[1].files[1].dst == d.join("c"));
        // this host's handles are its own
        assert_eq!(k.add(vec![], vec![]), "b.0002-1");
        k.v.pop();
        // written as read: the bad lines go, the kept state reads as it was read
        k.save().unwrap();
        let t2 = std::fs::read_to_string(&f).unwrap();
        assert!(
            t2.lines().count() == 3 && t2.contains(r#""state":"cancelled""#),
            "{t2}"
        );
        // unchanged: the file stays
        let old = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        let file = || std::fs::File::options().write(true).open(&f).unwrap();
        let (mut k, _) = Jobs::load("b.0002".into(), f.clone()).unwrap();
        file().set_modified(old).unwrap();
        k.save().unwrap();
        assert_eq!(file().metadata().unwrap().modified().unwrap(), old);
        k.v[0].cancelled = true;
        k.close().unwrap();
        assert!(Jobs::load("c".into(), f.clone()).unwrap().0.v[0].cancelled);
        k.v.clear();
        k.save().unwrap();
        assert_eq!(
            Jobs::load("c".into(), f.clone()).unwrap().0.v.len(),
            2,
            "closed: not saved"
        );
        // a time the system cannot hold ends no load (on Windows the line does not parse)
        let at = format!("\"at_ms\":{}", ms(j.v[0].at));
        let huge = t
            .lines()
            .nth(1)
            .unwrap()
            .replace(&at, "\"at_ms\":18446744073709551615");
        assert_ne!(huge, t.lines().nth(1).unwrap());
        std::fs::write(&f, format!("{HEAD}\n{huge}\n")).unwrap();
        let (k, bad) = Jobs::load("c".into(), f.clone()).unwrap();
        assert_eq!(k.v.len() + bad, 1);
        #[cfg(windows)]
        assert_eq!(bad, 1);
        // a job whose path is no UTF-8 is left out and said; the others are kept
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let (mut k, _) = Jobs::load("c".into(), f.clone()).unwrap();
            k.v.clear();
            let odd = d.join(std::ffi::OsStr::from_bytes(b"x\xff"));
            k.add(vec![], vec![x("a")]);
            let h = k.add(vec![], vec![File::new("/DCIM/Camera01/x".into(), odd)]);
            let e = k.save().unwrap_err().to_string();
            assert!(e.contains("jobs not kept") && e.contains(&h), "{e}");
            assert_eq!(Jobs::load("c".into(), f.clone()).unwrap().0.v.len(), 1);
        }
        // there but unreadable: an error, not an empty set to write over
        #[cfg(unix)]
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            let mode =
                |m| std::fs::set_permissions(&f, std::fs::Permissions::from_mode(m)).unwrap();
            mode(0o000);
            assert!(Jobs::load("c".into(), f.clone()).is_err());
            mode(0o600);
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The pruning keeps every unfinished job and the newest finished ones, and of the queue what waits, what a kept
    /// job has, and what stopped with its `.part` there.
    #[test]
    fn prune() {
        let d = tmp("prune");
        let mut q = dl::Queue::default();
        let mut j = Jobs::new("h".into());
        let x = |n: &str| File::new(format!("/DCIM/Camera01/{n}"), d.join(n));
        // the oldest waits; then an item cancelled between its files (one done, one stopped with its `.part`); then
        // KEEP + 20 finished ones (their files gone: failed)
        j.add(vec![], vec![x("wait")]);
        q.add("/DCIM/Camera01/wait", d.join("wait"));
        j.add(vec![], vec![x("vid"), x("lrv")]);
        j.v[1].cancelled = true;
        for n in ["vid", "lrv"] {
            q.add(&format!("/DCIM/Camera01/{n}"), d.join(n));
        }
        (q.v[1].st, q.v[2].st) = (St::Done, St::Paused);
        std::fs::write(d.join("vid"), b"x").unwrap();
        std::fs::write(media::part::of(&d.join("lrv")), b"x").unwrap();
        for i in 0..KEEP + 20 {
            j.add(vec![], vec![x(&format!("f{i}"))]);
        }
        // the newest has a file the queue finished
        j.add(vec![], vec![x("kept")]);
        q.add("/DCIM/Camera01/kept", d.join("kept"));
        q.v[3].st = St::Done;
        // no job has these
        for (n, st, part) in [
            ("done", St::Done, false),
            ("paused", St::Paused, true),
            ("failed", St::Failed, true),
            ("gone", St::Paused, false),
            ("queued", St::Queued, false),
        ] {
            q.add(&format!("/DCIM/Camera01/{n}"), d.join(n));
            q.v.last_mut().unwrap().st = st;
            if part {
                std::fs::write(media::part::of(&d.join(n)), b"x").unwrap();
            }
        }
        j.prune(&mut q);
        assert_eq!(j.v.len(), KEEP + 2);
        assert_eq!(j.v[0].h, "h-1", "unfinished: kept");
        assert_eq!(j.v[1].h, "h-2", "a stopped file: kept");
        assert_eq!(
            j.v[2].h,
            format!("h-{}", 3 + 20 + 1),
            "the oldest finished went"
        );
        let left: Vec<&str> = q.v.iter().map(|e| e.name()).collect();
        assert_eq!(
            left,
            ["wait", "vid", "lrv", "kept", "paused", "failed", "queued"]
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
