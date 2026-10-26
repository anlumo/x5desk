//! Downloads: a persistent queue of camera files, run one at a time over four tunnel connections (one connection
//! is camera-limited to ~18 MB/s, four reach 29–42 MB/s, PLAN.md §1). Data goes to `<dst>.part` with link's sidecar,
//! so a cancelled or interrupted file resumes where it stopped, also after a restart. Imports from a mounted card
//! (`file://` sources) run in the same queue as local copies ([`disk::copy`], also resumable), with or without a
//! camera; one whose card is gone waits for it, one whose card is there without its file fails. The queue is a text
//! file in the config dir: state, URI, failure reason and destination per line.

use crate::disk;
use link::{Camera, Download, Error, Pending, Prog};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, atomic::Ordering::Relaxed, mpsc},
    thread,
    time::{Duration, Instant},
};

pub const CONNS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum St {
    Queued,
    Running,
    Paused,
    Done,
    Failed,
}

const NAMES: [(St, &str); 4] = [
    (St::Queued, "queued"),
    (St::Paused, "paused"),
    (St::Done, "done"),
    (St::Failed, "failed"),
];

pub struct Dl {
    pub uri: String,
    pub dst: PathBuf,
    pub st: St,
    pub prog: Arc<Prog>,
    /// bytes per second over the last half second while running
    pub rate: f64,
    mark: Option<(Instant, u64)>,
    pub err: String,
}

impl Dl {
    pub fn name(&self) -> &str {
        crate::album::name(&self.uri)
    }
}

#[derive(Default)]
pub struct Queue {
    pub v: Vec<Dl>,
    file: Option<PathBuf>,
    /// the file's text as read or last written
    saved: String,
}

pub type Shared = Arc<Mutex<Queue>>;

/// The file of an import from a mounted card (`file://<path>`), None for a camera URI.
pub fn local(uri: &str) -> Option<&Path> {
    uri.strip_prefix("file://").map(Path::new)
}

/// A local copy on a thread of its own, as a download.
fn import(src: &Path, dst: &Path) -> Download {
    let (prog, (tx, rx)) = (Arc::new(Prog::default()), mpsc::sync_channel(1));
    let (p, s, d) = (prog.clone(), src.to_path_buf(), dst.to_path_buf());
    thread::spawn(move || drop(tx.send(disk::copy(&s, &d, &p))));
    Download {
        prog,
        done: rx.into(),
    }
}

fn part(p: &Path, e: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(e);
    s.into()
}

impl Queue {
    /// A running download of the last run counts as queued.
    pub fn load(file: Option<PathBuf>) -> Queue {
        let s = file
            .as_ref()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .unwrap_or_default();
        let v = s
            .lines()
            .filter_map(|l| {
                let mut f = l.splitn(4, '\t');
                let st = f.next()?;
                let st = NAMES.iter().find(|n| n.1 == st)?.0;
                let uri = f.next()?.into();
                // the destination is last: a path may hold tabs; lines without a reason come from before it was kept
                let (err, dst) = match (f.next()?, f.next()) {
                    (e, Some(d)) => (e, d),
                    (d, None) => ("", d),
                };
                Some(Dl {
                    uri,
                    dst: dst.into(),
                    st,
                    prog: Default::default(),
                    rate: 0.0,
                    mark: None,
                    err: err.into(),
                })
            })
            .collect();
        Queue { v, file, saved: s }
    }

    /// Writes the file unless it holds this already (every run's end saves).
    pub fn save(&mut self) {
        let Some(f) = &self.file else { return };
        let mut s = String::new();
        for d in &self.v {
            let st = NAMES.iter().find(|n| n.0 == d.st).map_or("queued", |n| n.1);
            let e = d.err.replace(['\t', '\n', '\r'], " ");
            s += &format!("{st}\t{}\t{e}\t{}\n", d.uri, d.dst.display());
        }
        if s == self.saved {
            return;
        }
        let t = f.with_extension("new");
        let r = f
            .parent()
            .map_or(Ok(()), crate::private)
            .and_then(|_| std::fs::write(&t, &s))
            .and_then(|_| std::fs::rename(&t, f));
        match r {
            Ok(()) => self.saved = s,
            Err(e) => eprintln!("{}: {e}", f.display()),
        }
    }

    /// Queues `uri` to `dst` unless that file is there; an entry for `dst` that stopped is resumed.
    pub fn add(&mut self, uri: &str, dst: PathBuf) {
        if dst.exists() {
            return;
        }
        match self.v.iter().position(|d| d.dst == dst) {
            Some(i) if self.v[i].st != St::Done => self.resume(i),
            i => {
                if let Some(i) = i {
                    self.v.remove(i);
                }
                self.v.push(Dl {
                    uri: uri.into(),
                    dst,
                    st: St::Queued,
                    prog: Default::default(),
                    rate: 0.0,
                    mark: None,
                    err: String::new(),
                })
            }
        }
    }

    /// Stops item `i`, keeping what arrived: a running one ends once the link sees the flag.
    pub fn cancel(&mut self, i: usize) {
        let d = &mut self.v[i];
        match d.st {
            St::Running => d.prog.cancel.store(true, Relaxed),
            St::Queued => d.st = St::Paused,
            _ => {}
        }
    }

    /// Stops the running download until the camera stops capturing (§7.12, as the app does): it goes back to the
    /// queue with what arrived. A card import is no camera file: it goes on.
    pub fn hold(&mut self) {
        for d in (self.v.iter_mut()).filter(|d| d.st == St::Running && local(&d.uri).is_none()) {
            d.prog.cancel.store(true, Relaxed);
            d.st = St::Queued
        }
    }

    pub fn resume(&mut self, i: usize) {
        let d = &mut self.v[i];
        if matches!(d.st, St::Paused | St::Failed) {
            (d.st, d.err) = (St::Queued, String::new());
        }
    }

    /// Drops item `i` (not while it runs) and its partial data; a finished file stays.
    pub fn remove(&mut self, i: usize) {
        if self.v[i].st == St::Running {
            return;
        }
        let d = self.v.remove(i);
        for e in [".part", ".part.todo"] {
            let _ = std::fs::remove_file(part(&d.dst, e));
        }
    }

    pub fn clear_done(&mut self) {
        self.v.retain(|d| d.st != St::Done);
    }
}

/// The running download: its destination and the link's result.
pub type Cur = Option<(PathBuf, Pending<u64>)>;

/// Collects the end of the running download into the queue, which is saved then; starts nothing (also the app's
/// quit). Returns the file that just finished.
pub fn end(g: &mut Queue, cur: &mut Cur) -> Option<PathBuf> {
    let mut fin = None;
    if let Some((dst, p)) = cur.as_ref()
        && let Some(r) = p.poll()
    {
        if let Some(d) = g.v.iter_mut().find(|d| d.dst == *dst) {
            (d.rate, d.mark) = (0.0, None);
            match r {
                Ok(_) => {
                    d.st = St::Done;
                    fin = Some(d.dst.clone())
                }
                // held (queued again) or cancelled since: that stays
                Err(Error::Cancelled) if d.st == St::Running => d.st = St::Paused,
                Err(Error::Cancelled) => {}
                // the link went away: on with the next session
                Err(Error::Closed | Error::NoDevice | Error::Timeout) => d.st = St::Queued,
                // the card went away: on once it is back
                Err(_) if local(&d.uri).is_some_and(|s| !s.exists()) => d.st = St::Queued,
                // the partial data was dropped; a new run starts over
                Err(Error::Changed) => {
                    (d.st, d.err) = (St::Queued, "changed at the source, restarted".into())
                }
                Err(e) => (d.st, d.err) = (St::Failed, e.to_string()),
            }
        }
        *cur = None;
        g.save();
    }
    fin
}

/// One turn of the runner: [`end`], then starts the next queued one when `cam` is up or its file is there, fails an
/// import whose card is among the mounted `vols` without its file, updates rates. Returns the file that just
/// finished.
pub fn step(
    q: &Shared,
    cam: Option<&Camera>,
    vols: &[disk::Vol],
    cur: &mut Cur,
) -> Option<PathBuf> {
    let mut g = q.lock().unwrap();
    let fin = end(&mut g, cur);
    // the next one that can run: an import whose file is there, a download while the camera is. An import whose card
    // is mounted without its file fails; the file is looked at before the card, so one unmounted in between waits.
    let mut failed = false;
    for d in g.v.iter_mut().filter(|d| d.st == St::Queued) {
        if cur.is_some() {
            break;
        }
        let ok = match local(&d.uri).map(|s| (s.try_exists(), s)) {
            None => cam.is_some(),
            Some((Ok(false), s))
                if (vols.iter())
                    .any(|v| s.starts_with(&v.root) && v.root.join("DCIM").is_dir()) =>
            {
                (d.st, d.err, failed) = (St::Failed, "not on the card".into(), true);
                false
            }
            Some((e, _)) => e.unwrap_or(false),
        };
        if !ok {
            continue;
        }
        if let Some(p) = d.dst.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        if let Some(dl) = local(&d.uri)
            .map(|s| import(s, &d.dst))
            .or_else(|| cam.map(|c| c.download(&d.uri, &d.dst, CONNS)))
        {
            (d.prog, d.st, d.rate, d.mark) = (dl.prog, St::Running, 0.0, None);
            *cur = Some((d.dst.clone(), dl.done));
        }
    }
    if failed {
        g.save()
    }
    let now = Instant::now();
    for d in g.v.iter_mut().filter(|d| d.st == St::Running) {
        let (n, t) = (d.prog.done.load(Relaxed), d.prog.total.load(Relaxed));
        // a resumed transfer reports what it already has right after it starts: count from there
        match d.mark {
            None if t > 0 && n > 0 => d.mark = Some((now, n)),
            Some((m, k)) if now - m >= Duration::from_millis(500) => {
                d.rate = (n - k.min(n)) as f64 / (now - m).as_secs_f64();
                d.mark = Some((now, n))
            }
            _ => {}
        }
    }
    fin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persist() {
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-q-{}", std::process::id()));
        let f = d.join("downloads");
        let mut q = Queue::load(Some(f.clone()));
        assert!(q.v.is_empty());
        for n in ["a", "b", "c", "d"] {
            q.add(&format!("/DCIM/Camera01/{n}.insv"), d.join(n));
        }
        q.add("/DCIM/Camera01/a.insv", d.join("a"));
        assert_eq!(q.v.len(), 4);
        (q.v[1].st, q.v[2].st, q.v[3].st) = (St::Running, St::Done, St::Failed);
        q.cancel(0);
        q.save();
        let mut q = Queue::load(Some(f.clone()));
        let st: Vec<St> = q.v.iter().map(|d| d.st).collect();
        assert_eq!(st, [St::Paused, St::Queued, St::Done, St::Failed]);
        assert_eq!(q.v[1].uri, "/DCIM/Camera01/b.insv");
        // adding a stopped one again resumes it; a finished one whose file is gone is queued anew
        q.add("/DCIM/Camera01/d.insv", d.join("d"));
        q.add("/DCIM/Camera01/c.insv", d.join("c"));
        let st: Vec<St> = q.v.iter().map(|d| d.st).collect();
        assert_eq!(st, [St::Paused, St::Queued, St::Queued, St::Queued]);
        assert_eq!(q.v[3].uri, "/DCIM/Camera01/c.insv");
        std::fs::write(part(&q.v[0].dst, ".part"), b"x").unwrap();
        q.remove(0);
        assert!(!part(&d.join("a"), ".part").exists());
        q.v[0].st = St::Done;
        q.clear_done();
        assert_eq!(q.v.len(), 2);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A save with nothing changed leaves the file alone, also right after a load (every run's end saves); a change
    /// is written.
    #[test]
    fn unchanged() {
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-q2-{}", std::process::id()));
        let f = d.join("downloads");
        let mut q = Queue::load(Some(f.clone()));
        q.add("/DCIM/Camera01/a.insv", d.join("a"));
        q.save();
        let old = std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        let file = || std::fs::File::options().write(true).open(&f).unwrap();
        file().set_modified(old).unwrap();
        q.save();
        let mut q = Queue::load(Some(f.clone()));
        q.save();
        assert_eq!(file().metadata().unwrap().modified().unwrap(), old);
        q.cancel(0);
        q.save();
        assert_ne!(file().metadata().unwrap().modified().unwrap(), old);
        assert_eq!(Queue::load(Some(f)).v[0].st, St::Paused);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The camera capturing holds its downloads only: a running card import goes on.
    #[test]
    fn hold() {
        let mut q = Queue::default();
        q.add("/DCIM/Camera01/a.insv", "/nowhere/a".into());
        q.add(
            "file:///run/media/u/X5/DCIM/Camera01/b.insv",
            "/nowhere/b".into(),
        );
        for d in &mut q.v {
            d.st = St::Running
        }
        q.hold();
        let v: Vec<(St, bool)> = (q.v.iter())
            .map(|d| (d.st, d.prog.cancel.load(Relaxed)))
            .collect();
        assert_eq!(v, [(St::Queued, true), (St::Running, false)]);
    }
}
