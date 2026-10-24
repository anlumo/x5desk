//! Stabilisation in the player (crate stab): a worker thread turns a clip's trailer IMU and exposure record into one
//! pose per frame, cached per clip and per setting. [`Stab`] is a player's side of it: the setting, the poses shown,
//! and the mapping of any of the clip's videos (the `.lrv` too) onto the posed one by time.

use eframe::egui;
use media::insv::{EXPOSURE, Exposure, Trailer};
use stab::{KI, KP, Lock, Opts, Pose, Track};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread,
    time::Instant,
};

/// Default smoothing σ, s (crate stab's measurements use it), and the slider's range.
pub const SMOOTH: f64 = 0.5;
pub const SMOOTH_MAX: f64 = 2.0;
/// Cached poses in all, beyond which the cache starts over (28 bytes each).
const BUDGET: usize = 1 << 21;

/// Stabilisation setting: None = off.
pub type Mode = Option<Lock>;

pub const MODES: [(Mode, &str); 4] = [
    (None, "Off"),
    (Some(Lock::Free), "Free"),
    (Some(Lock::Horizon), "Horizon lock"),
    (Some(Lock::Direction), "Direction lock"),
];

pub fn parse(s: &str) -> Option<Mode> {
    Some(match s {
        "off" => None,
        "free" => Some(Lock::Free),
        "horizon" => Some(Lock::Horizon),
        "direction" => Some(Lock::Direction),
        _ => return None,
    })
}

/// Lock and smoothing σ (bits).
type Key = (Lock, u64);
type Poses = Arc<[Pose]>;

pub enum Msg {
    /// What the worker does now, for the progress line.
    Stage(&'static str),
    /// The poses of `Key`, and how long they took, s.
    Done(Key, Result<(Poses, f64), String>),
}

pub struct Req {
    file: PathBuf,
    key: Key,
    tx: mpsc::Sender<Msg>,
    ctx: egui::Context,
}

/// The orientation track, frame record and timing of one clip.
struct Clip {
    tr: Track,
    ex: Vec<Exposure>,
    o: Opts,
}

fn load(f: &Path) -> Result<Clip, String> {
    let t = Trailer::open(f).map_err(|e| e.to_string())?;
    let (imu, ex) = (
        t.imu().map_err(|e| e.to_string())?,
        t.frames(EXPOSURE).map_err(|e| e.to_string())?,
    );
    if imu.len() < 2 || ex.is_empty() {
        return Err("no IMU or exposure record in the trailer".into());
    }
    Ok(Clip {
        tr: Track::new(&imu, KP, KI),
        ex,
        o: Opts::new(&t.meta, Lock::Free, 0.0),
    })
}

/// The stabilisation worker, one per app: keeps the last clip's orientation track and every clip's poses per setting.
pub fn worker() -> mpsc::Sender<Req> {
    let (tx, rx) = mpsc::channel::<Req>();
    let run = move || {
        let mut clip: Option<(PathBuf, Result<Clip, String>)> = None;
        let mut cache: HashMap<(PathBuf, u8, u64), Poses> = HashMap::new();
        while let Ok(mut r) = rx.recv() {
            // a slider drag queues many: only the newest counts (one player at a time asks)
            while let Ok(n) = rx.try_recv() {
                r = n;
            }
            let t0 = Instant::now();
            let ck = (r.file.clone(), r.key.0 as u8, r.key.1);
            let res = match cache.get(&ck) {
                Some(p) => Ok(p.clone()),
                None => {
                    if clip.as_ref().is_none_or(|c| c.0 != r.file) {
                        let _ = r.tx.send(Msg::Stage("reading and integrating the IMU"));
                        r.ctx.request_repaint();
                        clip = Some((r.file.clone(), load(&r.file)));
                    }
                    clip.as_ref()
                        .unwrap()
                        .1
                        .as_ref()
                        .map_err(Clone::clone)
                        .map(|c| {
                            let _ = r.tx.send(Msg::Stage("smoothing the camera path"));
                            r.ctx.request_repaint();
                            let o = Opts {
                                lock: r.key.0,
                                smooth: f64::from_bits(r.key.1),
                                ..c.o
                            };
                            let p: Poses = stab::poses(&c.tr, &c.ex, &o).into();
                            if cache.values().map(|p| p.len()).sum::<usize>() > BUDGET {
                                cache.clear();
                            }
                            cache.insert(ck, p.clone());
                            p
                        })
                }
            };
            let _ = r.tx.send(Msg::Done(
                r.key,
                res.map(|p| (p, t0.elapsed().as_secs_f64())),
            ));
            r.ctx.request_repaint();
        }
    };
    thread::Builder::new()
        .name("stab".into())
        .spawn(run)
        .expect("stab thread");
    tx
}

/// A player's stabilisation.
pub struct Stab {
    pub mode: Mode,
    /// σ, s
    pub smooth: f64,
    /// the file whose trailer is posed; its video's frame times and first frame's camera time (µs)
    file: PathBuf,
    times: Arc<[f64]>,
    ff: u64,
    /// framing: X5 videos frame on lens 1
    fr: stitch::Pose,
    /// the newest poses in (shown until the current setting's arrive) and the setting asked for
    have: Option<(Key, Poses)>,
    want: Option<Key>,
    pub stage: Option<&'static str>,
    pub err: Option<String>,
    /// compute time of the last poses, s
    pub took: f64,
    req: mpsc::Sender<Req>,
    rx: (mpsc::Sender<Msg>, mpsc::Receiver<Msg>),
    ctx: egui::Context,
}

impl Stab {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        req: &mpsc::Sender<Req>,
        file: PathBuf,
        times: Arc<[f64]>,
        ff: u64,
        yaw: f32,
        mode: Mode,
        smooth: f64,
        ctx: &egui::Context,
    ) -> Stab {
        let mut s = Stab {
            mode,
            smooth,
            file,
            times,
            ff,
            fr: stitch::Pose::euler(0.0, 0.0, yaw),
            have: None,
            want: None,
            stage: None,
            err: None,
            took: 0.0,
            req: req.clone(),
            rx: mpsc::channel(),
            ctx: ctx.clone(),
        };
        s.ask();
        s
    }

    fn key(&self) -> Option<Key> {
        self.mode.map(|l| (l, self.smooth.to_bits()))
    }

    /// After a setting change: asks the worker for its poses unless they are in or asked for.
    pub fn ask(&mut self) {
        let Some(k) = self.key() else { return };
        if self.err.is_some()
            || self.want == Some(k)
            || self.have.as_ref().is_some_and(|h| h.0 == k)
        {
            return;
        }
        self.want = Some(k);
        self.stage = Some("queued");
        let _ = self.req.send(Req {
            file: self.file.clone(),
            key: k,
            tx: self.rx.0.clone(),
            ctx: self.ctx.clone(),
        });
    }

    /// Takes the worker's messages; true when the poses changed.
    pub fn tick(&mut self) -> bool {
        let mut new = false;
        while let Ok(m) = self.rx.1.try_recv() {
            match m {
                Msg::Stage(s) => self.stage = Some(s),
                Msg::Done(k, r) => {
                    if self.want == Some(k) {
                        (self.want, self.stage) = (None, None);
                    }
                    match r {
                        Ok((p, s)) => (self.have, self.took, new) = (Some((k, p)), s, true),
                        Err(e) => (self.err, self.want, self.stage) = (Some(e), None, None),
                    }
                }
            }
        }
        new
    }

    /// Off, or the current setting's poses are in (or cannot be made).
    pub fn ready(&self) -> bool {
        self.key()
            .is_none_or(|k| self.err.is_some() || self.have.as_ref().is_some_and(|h| h.0 == k))
    }

    /// The stitch pose of the frame at time `t` of a video of this clip whose first frame is at `t0` (s) and camera
    /// time `ff` (µs): the posed video's frame nearest to the same instant.
    pub fn pose(&self, t: f64, t0: f64, ff: u64) -> stitch::Pose {
        let Some((_, p)) = self.have.as_ref().filter(|_| self.mode.is_some()) else {
            return self.fr;
        };
        let (x, ts) = (
            t - t0 + (ff as f64 - self.ff as f64) * 1e-6 + self.times[0],
            &self.times,
        );
        let i = ts.partition_point(|&s| s < x).min(ts.len() - 1);
        let k = if i > 0 && x - ts[i - 1] < ts[i] - x {
            i - 1
        } else {
            i
        };
        let p = p[k.min(p.len() - 1)];
        stitch::Pose {
            q: p.q * self.fr.q,
            rs: [p.rs; 2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    /// Frames map by time onto the posed video, also from a video that starts later on the camera clock; past the end
    /// the last pose holds; off is the framing alone.
    #[test]
    fn mapping() {
        assert_eq!(parse("horizon"), Some(Some(Lock::Horizon)));
        assert_eq!((parse("off"), parse("level")), (Some(None), None));
        let times: Arc<[f64]> = (0..100).map(|k| k as f64 * 1001.0 / 30000.0).collect();
        let (tx, _rx) = mpsc::channel();
        let ctx = egui::Context::default();
        let ff = 1_000_000;
        let mut s = Stab::new(&tx, "x".into(), times.clone(), ff, 0.0, None, SMOOTH, &ctx);
        s.mode = Some(Lock::Free);
        s.have = Some((
            (Lock::Free, SMOOTH.to_bits()),
            (0..100)
                .map(|k| Pose {
                    q: Quat::from_rotation_z(k as f32 * 0.01),
                    rs: Vec3::X * k as f32,
                })
                .collect(),
        ));
        let k = |p: stitch::Pose| p.rs[1].x as usize;
        for i in 0..100 {
            assert_eq!(k(s.pose(times[i], 0.0, ff)), i);
        }
        // a proxy whose own clock starts at 0.5 s, 10 frames (33 367 µs each, ±6 µs) after the posed video's first
        for j in 0..90 {
            assert_eq!(k(s.pose(0.5 + times[j], 0.5, ff + 333_730)), j + 10, "{j}");
        }
        assert_eq!(k(s.pose(1e3, 0.0, ff)), 99);
        s.mode = None;
        assert_eq!(s.pose(times[5], 0.0, ff).rs, [Vec3::ZERO; 2]);
    }
}
