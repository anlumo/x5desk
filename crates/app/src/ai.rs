//! AI features (PLAN.md milestone 9, group 11) on crate track's open models. [`MODELS`]: the networks, loaded once
//! per app run on first use and warmed up off the UI thread. Deep Track ([`Track`]): a subject the user boxes in a
//! view, followed through the clip on a background thread (the player's follow view, the editor's reframing keys,
//! Shot Lab's subject). Auto edit ([`Auto`]): highlights cut along the auto framer's path into an editor timeline.
//! [`privacy`]: the faces and licence plates of one frame for the viewers' privacy preview.
//!
//! Frames come from `shotlab::Clip` (decoded and stitched on the shared device) with the stabilisation the result is
//! for: directions are in the frame of that stitch, which is the frame the player's and the exporter's views turn in.
//! Locks: the models' mutex first; around each network call `BURN` (burn submits throughout), never inside `GATE`,
//! which the decode and stitch submits hold shared in between (the UI's retry takes BURN, then GATE).

use crate::{BURN, GATE, gyro};
use eframe::egui;
use glam::{Quat, Vec2, Vec3};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, TryLockError, Weak,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    thread,
    time::Duration,
};
use stitch::View;
use track::{Sample, State, Subject};
use video::Gpu;

pub const CANCELLED: &str = "cancelled";
/// Equirect width the networks look at (the face tiles' and the tracker tests' resolution, crate track).
const W: u32 = 3072;
/// Smoothing of a finished track for views and keys, s (`track::smooth`'s σ).
const SIGMA: f64 = 0.25;
/// Auto edit: every this many frames analysed; the share of the clip the highlights fill (at least `MIN_AUTO` s).
const STEP: usize = 10;
const SHARE: f64 = 0.3;
const MIN_AUTO: f64 = 2.0;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The networks of one device, loaded on first use; one user at a time.
pub struct Models {
    /// the device they are on (by its `Gpu`: devices of separate instances compare equal by id; the weak reference
    /// keeps the address from being reused), the networks, warmed up
    ai: Mutex<Option<Loaded>>,
    note: Mutex<String>,
    /// the thread in `with_stop` (an agents' lane: one stuck in a detection keeps them, and BURN, for good)
    holder: Mutex<Option<thread::ThreadId>>,
}

/// What `Models::ai` holds once they are loaded.
type Loaded = (Weak<Gpu>, track::Ai, bool);

/// The app's models: one load per run (and per device: the tests open several).
pub static MODELS: Models = Models {
    ai: Mutex::new(None),
    note: Mutex::new(String::new()),
    holder: Mutex::new(None),
};

impl Models {
    /// What the models are doing ("" when nothing is in the way).
    pub fn note(&self) -> String {
        lock(&self.note).clone()
    }

    fn say(&self, s: &str, ctx: Option<&egui::Context>) {
        *lock(&self.note) = s.into();
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    /// `n` steps with the progress note `<what>: <i> of <n>`, cleared however they end.
    fn steps(
        &self,
        what: &str,
        n: usize,
        ctx: Option<&egui::Context>,
        mut f: impl FnMut(usize) -> Result<(), String>,
    ) -> Result<(), String> {
        let r = (0..n).try_for_each(|i| {
            self.say(&format!("{what}: {} of {n}", i + 1), ctx);
            f(i)
        });
        self.say("", ctx);
        r
    }

    /// Runs `f` with the models on this thread, loaded on the run's first use (through `stitch::attach`: main.rs
    /// attached first) and, with `warm`, warmed up first: the tracker and the detector at every batch size compiled
    /// and tuned on a black panorama, a network run at a time under `BURN`, with a progress note (seconds each on the
    /// first use, 16 to 39 s in all, crate track; re-ID runs no network).
    pub fn with<T>(
        &self,
        g: &Arc<Gpu>,
        warm: bool,
        ctx: Option<&egui::Context>,
        f: impl FnOnce(&mut track::Ai) -> Result<T, String>,
    ) -> Result<T, String> {
        self.held(&mut lock(&self.ai), g, warm, ctx, f)
    }

    /// `with` (no warm-up) where the wait for the models sees `stop` (`acquire`): a set one ends it with `CANCELLED`.
    pub fn with_stop<T>(
        &self,
        g: &Arc<Gpu>,
        stop: &AtomicBool,
        ctx: Option<&egui::Context>,
        f: impl FnOnce(&mut track::Ai) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut m = self.acquire(stop)?;
        // cleared however `f` ends, a panic too
        struct Holds<'a>(&'a Mutex<Option<thread::ThreadId>>);
        impl Drop for Holds<'_> {
            fn drop(&mut self) {
                *lock(self.0) = None
            }
        }
        *lock(&self.holder) = Some(thread::current().id());
        let _h = Holds(&self.holder);
        self.held(&mut m, g, false, ctx, f)
    }

    /// The thread in `with_stop` now, if one is.
    pub fn holder(&self) -> Option<thread::ThreadId> {
        *lock(&self.holder)
    }

    /// The models' mutex taken by polling it every 5 ms (a thread blocked in `lock` can't see its stop): Err(CANCELLED)
    /// once `stop` is set while another holds it; a poisoned one taken over, as `lock` does.
    fn acquire(&self, stop: &AtomicBool) -> Result<MutexGuard<'_, Option<Loaded>>, String> {
        loop {
            match self.ai.try_lock() {
                Ok(m) => return Ok(m),
                Err(TryLockError::Poisoned(e)) => return Ok(e.into_inner()),
                Err(TryLockError::WouldBlock) if stop.load(Relaxed) => return Err(CANCELLED.into()),
                Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(5)),
            }
        }
    }

    /// The models dropped unless they are in use (they load again at the next use), under BURN: none of their burn
    /// calls is in flight then. Whether they were loaded and dropped.
    pub fn trim(&self) -> bool {
        let mut m = match self.ai.try_lock() {
            Ok(m) => m,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return false,
        };
        let _b = BURN.lock();
        m.take().is_some()
    }

    /// Whether they are loaded; None while another holds them.
    #[cfg(test)]
    pub fn loaded(&self) -> Option<bool> {
        match self.ai.try_lock() {
            Ok(m) => Some(m.is_some()),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner().is_some()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// `with`'s work once it holds the models (`m`).
    fn held<T>(
        &self,
        m: &mut Option<Loaded>,
        g: &Arc<Gpu>,
        warm: bool,
        ctx: Option<&egui::Context>,
        f: impl FnOnce(&mut track::Ai) -> Result<T, String>,
    ) -> Result<T, String> {
        if m.as_ref().is_none_or(|x| x.0.as_ptr() != Arc::as_ptr(g)) {
            *m = None;
            self.say("Loading the AI models…", ctx);
            let r = {
                let _b = BURN.lock();
                track::Ai::new(&g.instance, &g.adapter, &g.device, &g.queue)
            };
            self.say("", ctx);
            *m = Some((Arc::downgrade(g), r?, false));
        }
        let (_, ai, warmed) = m.as_mut().unwrap();
        if warm && !*warmed {
            let t = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("warm-up"),
                size: wgpu::Extent3d {
                    width: 2,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let p = ai.pano(&t.create_view(&Default::default()));
            let tiles = track::sphere();
            let n = 1 + tiles.len();
            self.steps("Warming up the AI models (first use)", n, ctx, |i| {
                let _b = BURN.lock();
                match i {
                    0 => {
                        let s = Subject {
                            dir: Vec3::NEG_X,
                            size: Vec2::splat(10.0),
                            cls: None,
                        };
                        track::Tracker::new(ai, &p, s)?.step(ai, &p);
                    }
                    _ => drop(ai.detect(&p, &tiles[..i], 1.0)),
                }
                Ok(())
            })?;
            *warmed = true;
        }
        f(ai)
    }
}

/// Where a background job is.
#[derive(Clone, Debug, PartialEq)]
pub enum St {
    /// waiting for the models (loading, warming up, another job)
    Wait,
    /// frames done of all
    Run(usize, usize),
    Done,
    Failed(String),
    Cancelled,
}

impl St {
    pub fn over(&self) -> bool {
        matches!(self, St::Done | St::Failed(_) | St::Cancelled)
    }

    /// The progress line of a job doing `what`.
    pub fn line(&self, what: &str) -> String {
        match self {
            St::Wait => match MODELS.note() {
                n if n.is_empty() => format!("{what}: waiting for the AI models…"),
                n => format!("{what}: {n}"),
            },
            St::Run(k, n) => format!("{what}: {k} of {n} frames"),
            St::Done => format!("{what}: done"),
            St::Failed(e) => format!("{what} failed: {e}"),
            St::Cancelled => format!("{what}: cancelled"),
        }
    }
}

/// A subject's box (centre, angular size) per frame where it was tracked.
pub type Boxes = Vec<Option<(Vec3, Vec2)>>;

/// A Deep Track run: the subject's samples from frame `k0` on as they come, the smoothed path once done.
#[derive(Debug)]
pub struct Track {
    /// the clip (an `.insv` with both lenses) and the stabilisation it was stitched with: the directions' frame
    pub file: PathBuf,
    pub stab: (gyro::Mode, f64),
    pub k0: usize,
    /// the clip's frame times, s (set as the run starts)
    pub times: OnceLock<Vec<f64>>,
    pub smp: Mutex<Vec<Sample>>,
    path: OnceLock<Vec<Vec3>>,
    /// the stitch rotation of every frame of the clip under `stab` (set as the run starts)
    qs: OnceLock<Vec<Quat>>,
    st: Mutex<St>,
    pub stop: AtomicBool,
}

impl Track {
    /// Follows subject `s` (in the frame of `file` stitched with `stab`, as on frame `k0`) from `k0` through `k1` on
    /// a thread of its own.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        ctx: &egui::Context,
        file: PathBuf,
        stab: (gyro::Mode, f64),
        k0: usize,
        k1: usize,
        s: Subject,
    ) -> Arc<Track> {
        let t = Arc::new(Track {
            file,
            stab,
            k0,
            times: OnceLock::new(),
            smp: Mutex::new(vec![]),
            path: OnceLock::new(),
            qs: OnceLock::new(),
            st: Mutex::new(St::Wait),
            stop: AtomicBool::new(false),
        });
        let (g, cx, ctx, t2) = (g.clone(), cx.clone(), ctx.clone(), t.clone());
        thread::Builder::new()
            .name("deep track".into())
            .spawn(move || {
                let _a = crate::awake::hold();
                let t = &t2;
                let r = crate::export::scoped(&g.device, || t.run(&g, &cx, &ctx, k1, s));
                let ts = t.times.get().map_or(&[][..], |v| &v[..]);
                let fps = match ts.len() {
                    n @ 2.. => (n - 1) as f64 / (ts[n - 1] - ts[0]).max(1e-3),
                    _ => 30.0,
                };
                let smp = lock(&t.smp).clone();
                let _ = t.path.set(track::smooth(&smp, (SIGMA * fps) as f32));
                *lock(&t.st) = match r {
                    Ok(()) => St::Done,
                    Err(_) if t.stop.load(Relaxed) => St::Cancelled,
                    Err(e) => St::Failed(e),
                };
                ctx.request_repaint();
            })
            .expect("deep track thread");
        t
    }

    fn run(
        &self,
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        ctx: &egui::Context,
        k1: usize,
        s: Subject,
    ) -> Result<(), String> {
        let c = shotlab::Clip::open(&self.file)?;
        let _ = self.times.set(c.times.clone());
        let poses = poses(&c, self.stab)?;
        let _ = (self.qs).set(
            (0..c.times.len())
                .map(|k| c.pose(k, poses.as_deref()).q)
                .collect(),
        );
        if self.k0 >= c.times.len() {
            return Err(format!("frame {} of {}", self.k0, c.times.len()));
        }
        let ks: Vec<usize> = (self.k0..=k1.min(c.times.len() - 1)).collect();
        let n = ks.len();
        MODELS.with(g, true, Some(ctx), |ai| {
            let (mut pano, mut tr) = (None, None::<track::Tracker>);
            let mut gate = Some(GATE.read());
            c.frames(
                g,
                cx,
                W,
                false,
                &ks,
                |k| c.pose(k, poses.as_deref()),
                |tex, _| {
                    if self.stop.load(Relaxed) {
                        return Err(CANCELLED.into());
                    }
                    let p =
                        pano.get_or_insert_with(|| ai.pano(&tex.create_view(&Default::default())));
                    gate.take();
                    let r = {
                        let _b = BURN.lock();
                        match &mut tr {
                            Some(t) => Ok(t.step(ai, p)),
                            None => {
                                // the detector's class at the box narrows re-detection to it (the box stays the
                                // user's)
                                let e = ((1.5 * s.size.max_element()).clamp(20.0, 120.0) / 2.0)
                                    .to_radians()
                                    .tan();
                                let cam = track::Cam::new(s.dir, Vec3::Z, Vec2::splat(e));
                                let cls = (ai.detect(p, &[cam], 0.3).into_iter())
                                    .map(|d| (track::iou((s.dir, s.size), d.on(), false), d.cls))
                                    .filter(|x| x.0 >= 0.3)
                                    .max_by(|a, b| a.0.total_cmp(&b.0))
                                    .map(|x| x.1);
                                track::Tracker::new(ai, p, Subject { cls, ..s }).map(|mut t| {
                                    // every frame is stepped: the subject's reach is per second of the clip
                                    t.fps = c.meta.fps as f32;
                                    tr = Some(t);
                                    Sample {
                                        raw: s.dir,
                                        dir: s.dir,
                                        size: s.size,
                                        score: 1.0,
                                        state: State::Tracked,
                                    }
                                })
                            }
                        }
                    };
                    gate = Some(GATE.read());
                    let mut v = lock(&self.smp);
                    v.push(r?);
                    *lock(&self.st) = St::Run(v.len(), n);
                    ctx.request_repaint();
                    Ok(())
                },
            )
        })
    }

    pub fn st(&self) -> St {
        lock(&self.st).clone()
    }

    /// The sample of frame `k`, if tracked (so far).
    pub fn at(&self, k: usize) -> Option<Sample> {
        lock(&self.smp).get(k.checked_sub(self.k0)?).copied()
    }

    /// The smoothed direction at frame `k` once the run is over, the first and last held outside it.
    pub fn dir(&self, k: usize) -> Option<Vec3> {
        let p = self.path.get()?;
        p.get(k.saturating_sub(self.k0).min(p.len().checked_sub(1)?))
            .copied()
    }

    /// The rotation of frame `k` from the track's frame into that of the clip stitched with rotation `q` there:
    /// q⁻¹ · q_track (about the identity under the track's own stabilisation); None before the run started.
    pub fn turn(&self, k: usize, q: Quat) -> Option<Quat> {
        Some(q.inverse() * *self.qs.get()?.get(k)?)
    }

    /// The frame at clip time `t` (0 before the run started).
    pub fn frame(&self, t: f64) -> usize {
        self.times.get().map_or(0, |v| crate::clip::frame_at(v, t))
    }

    /// The samples' boxes per frame from `k0` (centre, size) where the tracker had the subject, re-expressed in the
    /// frame of the clip stitched with `stab` (`turn`).
    pub fn boxes(&self, stab: (gyro::Mode, f64)) -> Result<Boxes, String> {
        let smp = lock(&self.smp).clone();
        let ok = |s: &Sample| matches!(s.state, State::Tracked | State::Checked | State::Found);
        let qs = match stab == self.stab {
            true => None,
            false => Some(rots(&shotlab::Clip::open(&self.file)?, stab)?),
        };
        let q = |k: usize| {
            qs.as_ref()
                .map_or(Some(Quat::IDENTITY), |q| self.turn(k, *q.get(k)?))
        };
        Ok((smp.iter().enumerate())
            .map(|(i, s)| (q(self.k0 + i).filter(|_| ok(s))).map(|q| (q * s.raw, s.size)))
            .collect())
    }
}

/// The poses of clip `c` stitched with `stab` (None: off).
pub fn poses(
    c: &shotlab::Clip,
    stab: (gyro::Mode, f64),
) -> Result<Option<Vec<stab::Pose>>, String> {
    match stab.0 {
        Some(l) => c
            .poses(l, stab.1)
            .ok_or("no IMU record to stabilise with".into())
            .map(Some),
        None => Ok(None),
    }
}

/// The stitch rotation of every frame of clip `c` stitched with `stab` (`shotlab::Clip::pose`).
pub fn rots(c: &shotlab::Clip, stab: (gyro::Mode, f64)) -> Result<Vec<Quat>, String> {
    let ps = poses(c, stab)?;
    Ok((0..c.times.len())
        .map(|k| c.pose(k, ps.as_deref()).q)
        .collect())
}

/// Reframing keys that follow the track over source times `a..b` (clip content time), evenly spaced about `every` s
/// apart: view `v` (projection, roll, field of view) turned onto the smoothed direction.
pub fn keys(t: &Track, a: f64, b: f64, every: f64, v: &edit::View) -> Vec<edit::Key> {
    let (Some(ts), n) = (t.times.get(), lock(&t.smp).len()) else {
        return vec![];
    };
    let (Some(&t0), Some(&t1)) = (ts.get(t.k0), ts.get((t.k0 + n).saturating_sub(1))) else {
        return vec![];
    };
    let (a, b) = (a.max(t0), b.min(t1));
    if b <= a {
        return vec![];
    }
    let [_, _, roll] = v.ypr();
    let m = ((b - a) / every).round().max(1.0) as usize;
    (0..=m)
        .filter_map(|i| {
            let s = a + (b - a) * i as f64 / m as f64;
            let [yaw, pitch] = track::yaw_pitch(t.dir(t.frame(s))?);
            Some(edit::Key {
                t: s,
                v: edit::View::look(yaw, pitch, roll, v.fov, v.proj),
                e: edit::Ease::Smooth,
            })
        })
        .collect()
}

/// A timeline colour per tracker state: tracked green, checked by the detector teal, missing amber, lost red, found
/// again blue.
pub fn colour(s: State) -> egui::Color32 {
    match s {
        State::Tracked => egui::Color32::from_rgb(70, 190, 90),
        State::Checked => egui::Color32::from_rgb(60, 170, 170),
        State::Miss => egui::Color32::from_rgb(235, 170, 40),
        State::Lost => egui::Color32::from_rgb(220, 60, 50),
        State::Found => egui::Color32::from_rgb(70, 130, 240),
    }
}

/// Paints the track's states as runs between `y` rows at `x(frame)` (None: off screen).
pub fn strip(p: &egui::Painter, t: &Track, y: egui::Rangef, x: impl Fn(usize) -> Option<f32>) {
    let smp = lock(&t.smp);
    let mut i = 0;
    while i < smp.len() {
        let s = smp[i].state;
        let j = (i..smp.len())
            .find(|&j| smp[j].state != s)
            .unwrap_or(smp.len());
        if let (Some(a), Some(b)) = (x(t.k0 + i), x(t.k0 + j)) {
            p.rect_filled(
                egui::Rect::from_x_y_ranges(a..=b.max(a + 1.0), y),
                0.0,
                colour(s),
            );
        }
        i = j;
    }
}

/// A box drawn by dragging on a viewer, corner to corner: the box while dragged (to paint over the picture); on
/// release, (x0, y0, x1, y1) in 0..1 of picture rect `r`, when it is at least 6 points each way.
pub fn drag_box(
    ui: &egui::Ui,
    resp: &egui::Response,
    r: egui::Rect,
    from: &mut Option<egui::Pos2>,
) -> (Option<egui::Rect>, Option<[f32; 4]>) {
    if resp.drag_started() {
        *from = ui.input(|i| i.pointer.press_origin());
    }
    let (Some(a), Some(b)) = (*from, resp.interact_pointer_pos()) else {
        return (None, None);
    };
    let bx = egui::Rect::from_two_pos(a, b).intersect(r);
    if !resp.drag_stopped() {
        return (Some(bx), None);
    }
    *from = None;
    let u = |p: egui::Pos2| {
        [(p.x - r.left()) / r.width(), (p.y - r.top()) / r.height()].map(|v| v.clamp(0.0, 1.0))
    };
    let ([x0, y0], [x1, y1]) = (u(bx.min), u(bx.max));
    (
        None,
        (bx.width() >= 6.0 && bx.height() >= 6.0).then_some([x0, y0, x1, y1]),
    )
}

/// Paints a box being drawn.
pub fn paint_box(p: &egui::Painter, b: egui::Rect) {
    p.rect_stroke(
        b,
        0.0,
        egui::Stroke::new(2.0, egui::Color32::YELLOW),
        egui::StrokeKind::Middle,
    );
}

/// The subject in box `b` (0..1 of the picture) of perspective view `v` on a picture `aspect` wide.
pub fn subject(v: &View, aspect: f32, b: [f32; 4]) -> Subject {
    Subject::boxed(&track::Cam::view(v.yaw, v.pitch, v.roll, v.fov, aspect), b)
}

/// The outline of a box (centre, angular size) in perspective view `v` shown in picture rect `r`; None when a
/// corner is behind the view.
pub fn outline(v: &View, r: egui::Rect, dir: Vec3, size: Vec2) -> Option<[egui::Pos2; 4]> {
    let cam = track::Cam::view(v.yaw, v.pitch, v.roll, v.fov, r.aspect_ratio());
    let e = Vec2::new(
        (size.x / 2.0).to_radians().tan(),
        (size.y / 2.0).to_radians().tan(),
    );
    let b = track::Cam::new(dir, Vec3::Z, e);
    let mut o = [egui::Pos2::ZERO; 4];
    for (i, c) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .into_iter()
        .enumerate()
    {
        let p = cam.px(b.ray(Vec2::new(c.0, c.1)))?;
        o[i] = egui::pos2(
            r.left() + (p.x + 1.0) / 2.0 * r.width(),
            r.top() + (p.y + 1.0) / 2.0 * r.height(),
        );
    }
    Some(o)
}

/// The faces and licence plates (`find`: faces, plates) on frame `k` of `file` stitched with `stab` (the viewer's
/// frame), for the privacy preview: the regions, the faces first, and how many of them are faces. With plates the
/// equirect is the export's `PLATES_W` wide, faces then on it too.
pub fn privacy(
    g: &Arc<Gpu>,
    cx: &stitch::Ctx,
    file: &std::path::Path,
    stab: (gyro::Mode, f64),
    k: usize,
    [fa, pl]: [bool; 2],
) -> Result<(Vec<track::Region>, usize), String> {
    let c = shotlab::Clip::open(file)?;
    let poses = poses(&c, stab)?;
    let tiles = track::face_tiles();
    MODELS.with(g, false, None, |ai| {
        let mut out = (vec![], 0);
        let mut gate = Some(GATE.read());
        c.frames(
            g,
            cx,
            if pl { crate::export::PLATES_W } else { W },
            false,
            &[k],
            |k| c.pose(k, poses.as_deref()),
            |tex, _| {
                gate.take();
                let _b = BURN.lock();
                let p = ai.pano(&tex.create_view(&Default::default()));
                let mut r = if fa {
                    ai.faces(&p, &tiles, 0.5)?
                } else {
                    vec![]
                };
                let n = r.len();
                if pl {
                    r.extend(ai.plates_still(&p)?);
                }
                out = (r, n);
                Ok(())
            },
        )?;
        Ok(out)
    })
}

/// An auto edit run: highlights of a clip cut along the auto framer's path, the editor timeline once done.
pub struct Auto {
    pub file: PathBuf,
    pub stab: (gyro::Mode, f64),
    st: Mutex<St>,
    pub out: Mutex<Option<edit::Timeline>>,
    pub stop: AtomicBool,
}

impl Auto {
    /// Analyses `file` stitched with `stab` (every `STEP`th frame: detections over the sphere for the framer and the
    /// people, faces, a thumbnail for visual change; camera motion from the gyro, loudness from the AAC track), picks
    /// highlights filling `SHARE` of it and cuts them into a timeline of views framed for `aspect`, on a thread.
    pub fn start(
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        ctx: &egui::Context,
        file: PathBuf,
        stab: (gyro::Mode, f64),
        aspect: f32,
    ) -> Arc<Auto> {
        let a = Arc::new(Auto {
            file,
            stab,
            st: Mutex::new(St::Wait),
            out: Mutex::new(None),
            stop: AtomicBool::new(false),
        });
        let (g, cx, ctx, a2) = (g.clone(), cx.clone(), ctx.clone(), a.clone());
        thread::Builder::new()
            .name("auto edit".into())
            .spawn(move || {
                let _a = crate::awake::hold();
                let a = &a2;
                let r = crate::export::scoped(&g.device, || a.run(&g, &cx, &ctx, aspect));
                *lock(&a.st) = match r {
                    Ok(tl) => {
                        *lock(&a.out) = Some(tl);
                        St::Done
                    }
                    Err(_) if a.stop.load(Relaxed) => St::Cancelled,
                    Err(e) => St::Failed(e),
                };
                ctx.request_repaint();
            })
            .expect("auto edit thread");
        a
    }

    pub fn st(&self) -> St {
        lock(&self.st).clone()
    }

    fn run(
        &self,
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        ctx: &egui::Context,
        aspect: f32,
    ) -> Result<edit::Timeline, String> {
        let c = shotlab::Clip::open(&self.file)?;
        let poses = poses(&c, self.stab)?;
        let (fps, dur) = crate::export::probe(&self.file)?;
        let n = c.times.len();
        let ks: Vec<usize> = (0..n).step_by(STEP).collect();
        let (mut sc, mut fm) = (track::Scorer::new(dur), track::Framer::new(aspect));
        let (st, ft) = (track::sphere(), track::face_tiles());
        MODELS.with(g, false, Some(ctx), |ai| {
            let mut pano = None;
            let mut gate = Some(GATE.read());
            c.frames(
                g,
                cx,
                W,
                false,
                &ks,
                |k| c.pose(k, poses.as_deref()),
                |tex, k| {
                    if self.stop.load(Relaxed) {
                        return Err(CANCELLED.into());
                    }
                    let p =
                        pano.get_or_insert_with(|| ai.pano(&tex.create_view(&Default::default())));
                    gate.take();
                    let r = {
                        let _b = BURN.lock();
                        let ds = ai.detect(p, &st, 0.3);
                        ai.faces(p, &ft, 0.6)
                            .and_then(|fs| Ok((ds, fs, ai.thumb(p)?)))
                    };
                    gate = Some(GATE.read());
                    let (ds, fs, th) = r?;
                    let t = c.times[k];
                    sc.people(t, &ds, &fs);
                    sc.frame(t, th);
                    fm.push(k, ds);
                    *lock(&self.st) = St::Run(k / STEP + 1, ks.len());
                    ctx.request_repaint();
                    Ok(())
                },
            )
        })?;
        if let Some((tr, ex)) = &c.imu {
            sc.gyro(tr, ex[0].t as f64 * 1e-6 - 0.5 * ex[0].time + stab::OFFSET);
        }
        sc.audio(&self.file)
            .map_err(|e| format!("{}: {e}", self.file.display()))?;
        let hs = sc.pick((SHARE * dur).max(MIN_AUTO));
        let r = fps.num as f64 / fps.den as f64;
        let path = fm.path(n, r as f32);
        track::timeline(&self.file.to_string_lossy(), dur, fps, r, &hs, &path).map_err(|e| e.0)
    }
}

#[cfg(test)]
pub(crate) mod tests;
