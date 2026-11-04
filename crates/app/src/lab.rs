//! Shot Lab screen (PLAN.md milestone 9, group 14) on crate shotlab, for the library's clip: each effect with its
//! parameters; a preview (the effect over the range up to its out point at preview size, on a thread, shown as the
//! frame it makes); "Export" (a job on the Export screen with the effect as its hook, 360) and, for the editing
//! presets, "Add to timeline" (their clips onto the editor's timeline) or "Export" (flat). The effects that need a
//! still camera check the gyro over the range and warn; they stitch with the direction lock (every frame into the
//! same world directions). The subject effects take the player's Deep Track run, re-expressed in the frame the effect
//! stitches with.

use crate::{
    BURN, GATE, ai,
    export::{Exp, Lab as Job, Maker, PRESETS, Spec, Tl},
    gyro,
};
use eframe::egui;
use glam::Quat;
use shotlab::{Clones, Light, Map};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc,
    },
    thread,
};
use stitch::View;
use video::Gpu;

/// Preview width (2:1), the last stretch of the range an accumulating preview runs over (frames), the turn (degrees)
/// past which a still-camera effect warns: a camera on a stick held still sways by about 5° (v627, v635 on a
/// railing), one carried along turns by tens.
const PW: u32 = 2048;
const SPAN: usize = 300;
const TURN: f64 = 10.0;
/// The editing presets' preview: cells of a 2 × 2 grid.
const CW: u32 = 640;
const CH: u32 = 360;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fx {
    Passerby,
    Trails,
    Stars,
    Delay,
    Shadow,
    Loop,
    Planets,
    StopMotion,
    FlyLapse,
    Flip,
    Center,
    Spin,
    FlyOut,
    Dolly,
}

/// Effects: kind, name, key (test hook, file names), what it does.
#[rustfmt::skip]
pub const FXS: [(Fx, &str, &str, &str); 14] = [
    (Fx::Passerby,   "Remove Passerby", "passerby", "The median of frames across the range: whoever passes by is gone (a still camera)"),
    (Fx::Trails,     "Light Trails",    "trails",   "Bright lights leave trails (Night Road, Flash Neon; a still camera)"),
    (Fx::Stars,      "Star Lapse",      "stars",    "Frames stacked: the noise settles, or with Maximum the stars draw trails (a still camera)"),
    (Fx::Delay,      "Frame Delay",     "delay",    "Each part of the picture lags behind by its place: a time slice"),
    (Fx::Shadow,     "Shadow Clone",    "shadow",   "Copies of the tracked subject stay behind as it moves (a still camera)"),
    (Fx::Loop,       "Clone Loop",      "loop",     "The tracked subject's past selves follow it (a still camera)"),
    (Fx::Planets,    "Parallel Planet", "planets",  "The tiny planet and its sky world side by side"),
    (Fx::StopMotion, "Stop Motion",     "stopmotion", "Freeze frames a step of source apart"),
    (Fx::FlyLapse,   "Fly Lapse",       "flylapse", "A hyperlapse orbiting out of a tiny planet"),
    (Fx::Flip,       "Flip My Day",     "flip",     "A hyperlapse rolling the world over"),
    (Fx::Center,     "Center Stage",    "center",   "Framed on the tracked subject"),
    (Fx::Spin,       "Spin Me Around",  "spin",     "Framed on the tracked subject while the picture turns"),
    (Fx::FlyOut,     "Fly Out",         "flyout",   "Framed on the tracked subject, pulling out into a tiny planet"),
    (Fx::Dolly,      "Dolly Zoom",      "dolly",    "The tracked subject keeps its size while the background moves"),
];

impl Fx {
    pub fn parse(s: &str) -> Option<Fx> {
        FXS.iter().find(|f| f.2 == s).map(|f| f.0)
    }

    fn name(self) -> &'static str {
        FXS.iter().find(|f| f.0 == self).unwrap().1
    }

    fn key(self) -> &'static str {
        FXS.iter().find(|f| f.0 == self).unwrap().2
    }

    /// An editing preset (clips for the editor), not a pass on the frames.
    pub fn cut(self) -> bool {
        matches!(
            self,
            Fx::StopMotion
                | Fx::FlyLapse
                | Fx::Flip
                | Fx::Center
                | Fx::Spin
                | Fx::FlyOut
                | Fx::Dolly
        )
    }

    /// Needs a still camera: checks the gyro, stitches with the direction lock.
    fn still(self) -> bool {
        matches!(
            self,
            Fx::Passerby | Fx::Trails | Fx::Stars | Fx::Shadow | Fx::Loop
        )
    }

    /// Needs the Deep Track subject.
    fn subject(self) -> bool {
        matches!(
            self,
            Fx::Shadow | Fx::Loop | Fx::Center | Fx::Spin | Fx::FlyOut | Fx::Dolly
        )
    }
}

/// The effects' parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct P {
    /// Remove Passerby: frames in the median
    pub n: u32,
    /// Light Trails: the light that enters, its half-life (s; 0 keeps it)
    pub light: Light,
    pub half: f32,
    /// Star Lapse: the maximum (trails) instead of the mean
    pub max: bool,
    /// Frame Delay: the largest delay (s), where it lags (0 by longitude from the view, 1 around the view, 2 left to
    /// right, 3 top to bottom), reversed
    pub span: f64,
    pub map: u8,
    pub rev: bool,
    /// Shadow Clone: a copy each `gap` of the subject's widths, `copies` shown; Clone Loop: `copies` selves `lag` s
    /// apart
    pub gap: f32,
    pub copies: u32,
    pub lag: f64,
    /// Parallel Planet: stacked instead of side by side
    pub tall: bool,
    /// editing presets: hyperlapse speed, turns, Stop Motion's step (s), the subject's share of the frame height,
    /// Fly Out's pull (s), Dolly Zoom's first field of view (°)
    pub speed: f64,
    pub turns: f32,
    pub step: f64,
    pub fill: f32,
    pub out: f64,
    pub fov: f32,
}

impl Default for P {
    fn default() -> P {
        P {
            n: 16,
            light: Light {
                thr: 0.45,
                soft: 0.2,
                ..Light::default()
            },
            half: 4.0,
            max: false,
            span: 1.0,
            map: 0,
            rev: false,
            gap: 1.2,
            copies: 6,
            lag: 0.6,
            tall: false,
            speed: 4.0,
            turns: 1.0,
            step: 0.5,
            fill: 0.45,
            out: 2.0,
            fov: 80.0,
        }
    }
}

/// What the screen hands over: an effect job for the Export screen, a timeline for it, or clips for the editor (with
/// the stabilisation their views are in).
pub enum Ask {
    Export(Spec, crate::export::Set, [f64; 2], Job),
    Timeline(Tl, crate::export::Set),
    Editor(edit::Timeline, (gyro::Mode, f64)),
}

/// A preview being made: its state, the picture (RGBA, w, h).
struct Pre {
    st: Mutex<ai::St>,
    px: Mutex<Option<(Vec<u8>, u32, u32)>>,
    stop: AtomicBool,
}

/// The gyro's turn (degrees) over a range (s) of a file.
type Turn = (PathBuf, [f64; 2], Option<f64>);

/// Test hook: the effect, the range, make the preview, export with these tokens (a preset key).
#[derive(Clone, Debug, Default)]
pub struct Hook {
    pub on: bool,
    pub fx: Option<Fx>,
    pub range: Option<[f64; 2]>,
    pub pre: bool,
    pub exp: Option<String>,
}

pub struct Lab {
    g: Arc<Gpu>,
    cx: stitch::Ctx,
    ctx: egui::Context,
    pub fx: Fx,
    pub p: P,
    /// the library's clip, the player's Deep Track run (over), the source span (s)
    src: Option<Spec>,
    trk: Option<Arc<ai::Track>>,
    pub range: [f64; 2],
    /// the gyro's turn over the range (degrees) for (file, range), a check out
    turn: Option<Turn>,
    tq: Option<mpsc::Receiver<Turn>>,
    pre: Option<(Fx, Arc<Pre>)>,
    tex: Option<(Fx, egui::TextureHandle)>,
    /// export preset: 360 for the passes, flat for the editing presets
    pano: usize,
    flat: usize,
    pub ask: Option<Ask>,
    pub err: Option<String>,
    hook: Option<Hook>,
    /// the test hook started the preview: show the screen
    pub show: bool,
}

impl Lab {
    pub fn new(g: &Arc<Gpu>, cx: &stitch::Ctx, ctx: &egui::Context, hook: Option<Hook>) -> Lab {
        Lab {
            g: g.clone(),
            cx: cx.clone(),
            ctx: ctx.clone(),
            fx: hook.as_ref().and_then(|h| h.fx).unwrap_or(Fx::Trails),
            p: P::default(),
            src: None,
            trk: None,
            range: [0.0; 2],
            turn: None,
            tq: None,
            pre: None,
            tex: None,
            pano: 1,
            flat: 4,
            ask: None,
            err: None,
            hook,
            show: false,
        }
    }

    /// Starts on Shot Lab (test hook).
    pub fn first(&self) -> bool {
        self.hook.as_ref().is_some_and(|h| h.on)
    }

    /// The stabilisation an effect stitches with: the direction lock (no smoothing) for a still camera, else the
    /// source's.
    fn stab(&self, s: &Spec) -> (gyro::Mode, f64) {
        match self.fx.still() {
            true => (Some(stab::Lock::Direction), 0.0),
            false => (s.stab, s.smooth),
        }
    }

    /// Every frame: the library's clip and the player's finished Deep Track run; replies; the test hook's steps.
    pub fn tick(&mut self, src: Option<Spec>, trk: Option<Arc<ai::Track>>) {
        let src = src.filter(|s| !s.photo && s.files.len() == 1);
        if src.as_ref().map(|s| &s.files) != self.src.as_ref().map(|s| &s.files) {
            self.range = src.as_ref().map_or([0.0; 2], |s| [0.0, s.dur]);
            (self.tex, self.turn, self.err) = (None, None, None);
            if let Some(r) = self.hook.as_ref().and_then(|h| h.range) {
                self.range = r;
            }
        }
        self.src = src;
        self.trk = trk.filter(|t| t.st() == ai::St::Done);
        if let Some(r) = &self.tq
            && let Ok(x) = r.try_recv()
        {
            (self.turn, self.tq) = (Some(x), None);
        }
        // the gyro's turn over the range, when the range or the clip changed
        if let Some(s) = &self.src
            && self.tq.is_none()
            && self
                .turn
                .as_ref()
                .is_none_or(|t| t.0 != s.files[0] || t.1 != self.range)
        {
            let (tx, rx) = mpsc::channel();
            let (f, r, ctx) = (s.files[0].clone(), self.range, self.ctx.clone());
            thread::spawn(move || {
                let t = shotlab::Clip::open(&f).ok().and_then(|c| {
                    let k0 = crate::clip::frame_at(&c.times, r[0]);
                    let k1 = crate::clip::frame_at(&c.times, r[1]);
                    c.turn(&(k0..=k1).step_by(5).collect::<Vec<_>>())
                });
                let _ = tx.send((f, r, t));
                ctx.request_repaint();
            });
            self.tq = Some(rx);
        }
        if let Some((fx, p)) = &self.pre
            && p.st.lock().unwrap_or_else(|e| e.into_inner()).over()
        {
            match p.st.lock().unwrap_or_else(|e| e.into_inner()).clone() {
                ai::St::Done => {
                    if let Some((px, w, h)) = p.px.lock().unwrap_or_else(|e| e.into_inner()).take()
                    {
                        let img =
                            egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
                        self.tex = Some((
                            *fx,
                            self.ctx.load_texture("shot lab", img, Default::default()),
                        ));
                    }
                }
                ai::St::Cancelled => {}
                st => self.err = Some(st.line("Preview")),
            }
            self.pre = None;
        }
        // test hook: the preview, then the export, once the source (and a subject, if needed) is there
        if let Some(h) = &mut self.hook
            && self.src.is_some()
            && (!self.fx.subject() || self.trk.is_some())
            && self.pre.is_none()
        {
            if h.pre {
                h.pre = false;
                self.show = true;
                self.preview();
            } else if let Some(k) = h.exp.take() {
                let key = k.split(':').next().unwrap_or("");
                match PRESETS.iter().position(|p| p.0 == key) {
                    Some(i) if PRESETS[i].2 => self.pano = i,
                    Some(i) => self.flat = i,
                    None => eprintln!("x5desk lab: no preset {key}"),
                }
                let x = crate::export::parse(Default::default(), &k);
                if let Err(e) = x.and_then(|x| self.export(Some(x))) {
                    eprintln!("x5desk lab: {e}");
                    self.err = Some(e);
                }
            } else {
                self.hook = None;
            }
        }
        if self.pre.is_some() {
            self.ctx
                .request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    /// Nothing pending (test hook): no preview being made, the hook's steps done.
    pub fn idle(&self) -> bool {
        self.pre.is_none() && self.hook.is_none() && self.ask.is_none()
    }

    /// The tracked subject per frame from the track's first frame, in the frame of `s` stitched with `st`, with the
    /// track's first frame; why not.
    fn boxes(&self, st: (gyro::Mode, f64)) -> Result<(usize, ai::Boxes), String> {
        let s = self.src.as_ref().ok_or("no clip")?;
        let t = (self.trk.as_ref())
            .filter(|t| t.file == s.files[0])
            .ok_or("Track a subject in the player first (Deep Track)")?;
        Ok((t.k0, t.boxes(st)?))
    }

    /// The effect's hook maker for a job whose output frame k shows source frame `k0` + k of a clip at `fps`.
    fn maker(&self, fps: f64, k0: usize) -> Result<Maker, String> {
        let (fx, p, cx) = (self.fx, self.p, self.cx.clone());
        let s = self.src.as_ref().ok_or("no clip")?;
        let view = s.view;
        let sub = match fx.subject() {
            true => Some(self.boxes(self.stab(s))?),
            false => None,
        };
        let frames = ((self.range[1] - self.range[0]) * fps).max(1.0) as u64;
        Ok(Box::new(move |g, w, h, f| {
            let sx = shotlab::Ctx::new(&g.device, &g.queue);
            let subj = move |k: u64| {
                let (t0, b) = sub.as_ref()?;
                *b.get((k0 + k as usize).checked_sub(*t0)?)?
            };
            Ok(match fx {
                Fx::Passerby => {
                    // the ring spans the range: every `every`-th frame enters it
                    let n = p.n.clamp(2, 64);
                    shotlab::Median::new(&sx, w, h, n, f).hook(frames.div_ceil(n as u64).max(1))
                }
                Fx::Trails => {
                    let light = Light {
                        decay: match p.half {
                            h if h > 0.0 => 0.5f32.powf(1.0 / (h * fps as f32)),
                            _ => 1.0,
                        },
                        ..p.light
                    };
                    shotlab::Trails::new(&sx, w, h, f, light).hook([0.0, f64::INFINITY])
                }
                Fx::Stars => shotlab::Stack::new(&sx, w, h, p.max).hook(|_| Quat::IDENTITY),
                Fx::Delay => {
                    let n = ((p.span * fps).round() as u32 + 1).clamp(2, 64);
                    let d = track::dir(view.yaw, view.pitch);
                    let map =
                        [Map::Lon(d), Map::Dir(d), Map::Cols, Map::Rows][p.map.min(3) as usize];
                    let mut x = shotlab::Delay::new(&sx, w, h, n, f, map);
                    x.rev = p.rev;
                    x.hook()
                }
                Fx::Shadow | Fx::Loop => {
                    let m = match fx {
                        Fx::Shadow => Clones::Shadow {
                            gap: p.gap,
                            n: p.copies.clamp(1, 16),
                        },
                        _ => Clones::Loop {
                            lag: ((p.lag * fps).round() as u32).max(1),
                            n: p.copies.clamp(1, 8),
                        },
                    };
                    // the person network runs through burn, which submits from this thread: under BURN, as
                    // crate export's RAFT (a job calls its hook outside GATE)
                    let mut h = {
                        let _b = BURN.lock();
                        shotlab::Cuts::new(g, &sx, m.slots(), 512, f)?.hook(m, subj)?
                    };
                    Box::new(move |x: export::Fx| {
                        let _b = BURN.lock();
                        h(x)
                    })
                }
                Fx::Planets => {
                    let mut x = shotlab::Planets::new(&cx, f);
                    x.tall = p.tall;
                    x.hook()
                }
                _ => return Err(format!("{} is an editing preset", fx.name())),
            })
        }))
    }

    /// An editing preset's timeline (one clip of the source) and the stabilisation its views are in.
    fn timeline(&self) -> Result<(edit::Timeline, (gyro::Mode, f64)), String> {
        let s = self
            .src
            .as_ref()
            .ok_or("Open an X5 video in the library first.")?;
        let (fps, dur) = crate::export::probe(&s.files[0])?;
        let [a, b] = self.range;
        let p = self.p;
        let (_, _, _, [w, h], ..) = PRESETS[self.flat];
        let aspect = w as f32 / h as f32;
        let v = crate::editor::to_edit(&s.view);
        let ramp = (0.15 * (b - a)).min(1.0);
        // the subject over time: the smoothed path, the box sizes where the tracker had it
        let subj = || -> Result<(Vec<shotlab::cut::Subj>, (gyro::Mode, f64)), String> {
            let t = (self.trk.as_ref())
                .filter(|t| t.file == s.files[0])
                .ok_or("Track a subject in the player first (Deep Track)")?;
            let (ts, smp) = (
                t.times.get().ok_or("no track")?,
                t.smp.lock().unwrap_or_else(|e| e.into_inner()),
            );
            let v = (smp.iter().enumerate())
                .filter(|(_, x)| {
                    matches!(
                        x.state,
                        track::State::Tracked | track::State::Checked | track::State::Found
                    )
                })
                .filter_map(|(i, x)| {
                    Some(shotlab::cut::Subj {
                        t: *ts.get(t.k0 + i)?,
                        dir: t.dir(t.k0 + i)?,
                        size: x.size,
                    })
                })
                .collect();
            Ok((v, t.stab))
        };
        let own = (s.stab, s.smooth);
        let e = |e: edit::Error| e.0;
        let (clips, st) = match self.fx {
            Fx::StopMotion => (
                shotlab::cut::stop_motion(0, a, b, p.step, (p.step / p.speed).max(fps.t(1)))
                    .map_err(e)?,
                own,
            ),
            Fx::FlyLapse => (
                vec![shotlab::cut::fly_lapse(0, a, b, p.speed, ramp, v, p.turns, 1.0).map_err(e)?],
                own,
            ),
            Fx::Flip => (
                vec![shotlab::cut::flip(0, a, b, p.speed, ramp, v, p.turns).map_err(e)?],
                own,
            ),
            Fx::Center | Fx::Spin | Fx::FlyOut | Fx::Dolly => {
                let (x, st) = subj()?;
                let c = match self.fx {
                    Fx::Center => shotlab::cut::center_stage(0, a, b, &x, p.fill, aspect, 0.5),
                    Fx::Spin => shotlab::cut::spin(0, a, b, &x, p.fill, aspect, p.turns),
                    Fx::FlyOut => shotlab::cut::fly_out(0, a, b, &x, p.fill, aspect, p.out),
                    _ => shotlab::cut::dolly(0, a, b, &x, p.fov, 0.5),
                };
                (vec![c.map_err(e)?], st)
            }
            _ => return Err(format!("{} is a pass on the frames", self.fx.name())),
        };
        let tl =
            shotlab::cut::timeline(&s.files[0].to_string_lossy(), dur, fps, clips).map_err(e)?;
        Ok((tl, st))
    }

    /// Hands the effect over: a pass as a 360 job (or the editing preset as a flat timeline job) with the Export
    /// screen's settings `x` (None: the defaults) and the screen's preset.
    pub fn export(&mut self, x: Option<crate::export::Set>) -> Result<(), String> {
        let s = self
            .src
            .clone()
            .ok_or("Open an X5 video in the library first.")?;
        let mut x = x.unwrap_or_default();
        if self.fx.cut() {
            let (tl, st) = self.timeline()?;
            x.apply(self.flat);
            let name = format!(
                "{}_{}",
                s.files[0]
                    .file_stem()
                    .map_or("x5".into(), |n| n.to_string_lossy()),
                self.fx.key()
            );
            self.ask = Some(Ask::Timeline(
                Tl {
                    name,
                    files: vec![s.files.clone()],
                    tl,
                    looks: Default::default(),
                    over: vec![],
                    denoise: None,
                    stab: st.0,
                    smooth: st.1,
                },
                x,
            ));
            return Ok(());
        }
        let (fps, _) = crate::export::probe(&s.files[0])?;
        let r = fps.num as f64 / fps.den as f64;
        let k0 = crate::clip::frame_at(&shotlab::Clip::open(&s.files[0])?.times, self.range[0]);
        let hook = self.maker(r, k0)?;
        x.apply(self.pano);
        let st = self.stab(&s);
        // output frame k is source frame k0 + k: the source's frame rate
        // (the Export screen's face and plate blur runs before the effect, on the sphere: what the effect keeps is
        // blurred)
        (x.stab, x.smooth, x.fps) = (st.0, st.1, None);
        self.ask = Some(Ask::Export(
            s,
            x,
            self.range,
            Job {
                key: self.fx.key(),
                name: self.fx.name().into(),
                hook: Mutex::new(Some(hook)),
                flat: self.fx == Fx::Planets,
            },
        ));
        Ok(())
    }

    /// Makes the preview on a thread: a pass over the range's last `SPAN` frames (Remove Passerby: `n` frames across
    /// it) at `PW` wide, its last frame; an editing preset's timeline at four times through its views.
    pub fn preview(&mut self) {
        let Some(s) = self.src.clone() else { return };
        if let Some((_, p)) = &self.pre {
            p.stop.store(true, Relaxed);
        }
        let pre = Arc::new(Pre {
            st: Mutex::new(ai::St::Wait),
            px: Mutex::new(None),
            stop: AtomicBool::new(false),
        });
        let st = self.stab(&s);
        let job = match self.fx.cut() {
            true => self.timeline().map(|(tl, st)| (None, Some(tl), st)),
            false => crate::export::probe(&s.files[0]).and_then(|(fps, _)| {
                let r = fps.num as f64 / fps.den as f64;
                let k0 =
                    crate::clip::frame_at(&shotlab::Clip::open(&s.files[0])?.times, self.range[0]);
                Ok((Some((self.maker(r, k0)?, k0)), None, st))
            }),
        };
        let (mk, tl, st) = match job {
            Ok(j) => j,
            Err(e) => {
                self.err = Some(e);
                return;
            }
        };
        let (g, cx, ctx, p, fx, range, nm) = (
            self.g.clone(),
            self.cx.clone(),
            self.ctx.clone(),
            pre.clone(),
            self.fx,
            self.range,
            self.p.n.clamp(2, 64) as usize,
        );
        thread::Builder::new()
            .name("shot lab preview".into())
            .spawn(move || {
                let r = crate::export::scoped(&g.device, || {
                    let c = shotlab::Clip::open(&s.files[0])?;
                    let ps = ai::poses(&c, st)?;
                    let pose = |k| c.pose(k, ps.as_deref());
                    let stop = || match p.stop.load(Relaxed) {
                        true => Err(ai::CANCELLED.to_string()),
                        false => Ok(()),
                    };
                    let mut out = None;
                    match (mk, tl) {
                        (Some((mk, k0)), _) => {
                            let k1 = c
                                .times
                                .partition_point(|&t| t < range[1] - 1e-3)
                                .max(k0 + 1)
                                - 1;
                            let mut ks: Vec<usize> = match fx {
                                Fx::Passerby => {
                                    (0..nm).map(|i| k0 + (k1 - k0) * i / (nm - 1)).collect()
                                }
                                _ => (k1.saturating_sub(SPAN).max(k0)..=k1).collect(),
                            };
                            ks.dedup();
                            // the maker and the hook take BURN where they run burn: never inside GATE
                            let mut h = mk(&g, PW, PW / 2, wgpu::TextureFormat::Rgba8Unorm)?;
                            let (n, mut i) = (ks.len(), 0);
                            let mut gate = Some(GATE.read());
                            c.frames(&g, &cx, PW, false, &ks, pose, |tex, k| {
                                stop()?;
                                gate.take();
                                let mut e = g.device.create_command_encoder(&Default::default());
                                // Remove Passerby's preview: every frame into the median
                                h(export::Fx {
                                    enc: &mut e,
                                    tex,
                                    k: if fx == Fx::Passerby {
                                        0
                                    } else {
                                        (k - k0) as u64
                                    },
                                    t: c.times[k] - c.times[k0],
                                });
                                gate = Some(GATE.read());
                                g.queue.submit([e.finish()]);
                                i += 1;
                                *p.st.lock().unwrap_or_else(|e| e.into_inner()) = ai::St::Run(i, n);
                                ctx.request_repaint();
                                if i == n {
                                    out = Some((
                                        stitch::read_texture(&g.device, &g.queue, tex),
                                        PW,
                                        PW / 2,
                                    ));
                                }
                                Ok(())
                            })?;
                        }
                        (_, Some(tl)) => {
                            // four output times through the timeline's views, each from its source frame
                            let d = tl.dur();
                            let mut want: Vec<(usize, View, usize)> = (0..4)
                                .filter_map(|i| {
                                    let e = tl.at((d * i as f64 / 3.0)
                                        .min(d - tl.out.fps.t(1))
                                        .max(0.0))?;
                                    Some((
                                        crate::clip::frame_at(&c.times, e.src),
                                        export::view(&e.view),
                                        i,
                                    ))
                                })
                                .collect();
                            want.sort_by_key(|w| w.0);
                            let ks: Vec<usize> = want.iter().map(|w| w.0).collect();
                            let grid = g.device.create_texture(&wgpu::TextureDescriptor {
                                label: Some("shot lab preview"),
                                size: wgpu::Extent3d {
                                    width: 2 * CW,
                                    height: 2 * CH,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                    | wgpu::TextureUsages::COPY_SRC,
                                view_formats: &[],
                            });
                            let gv = grid.create_view(&Default::default());
                            let pv = stitch::Preview::pano(&cx, wgpu::TextureFormat::Rgba8Unorm);
                            let (n, mut done) = (want.len(), 0);
                            let mut gate = Some(GATE.read());
                            c.frames(&g, &cx, PW, false, &ks, pose, |tex, k| {
                                stop()?;
                                // a surface configure the UI retries gets its turn between frames
                                gate.take();
                                gate = Some(GATE.read());
                                let pano = cx.pano(&tex.create_view(&Default::default()));
                                for (_, v, i) in want.iter().filter(|w| w.0 == k) {
                                    let r = [(i % 2) as u32 * CW, (i / 2) as u32 * CH]
                                        .map(|x| x as f32);
                                    pv.set(v, [r[0], r[1], CW as f32, CH as f32]);
                                    let mut e =
                                        g.device.create_command_encoder(&Default::default());
                                    {
                                        let mut rp =
                                            e.begin_render_pass(&wgpu::RenderPassDescriptor {
                                                label: Some("shot lab preview"),
                                                color_attachments: &[Some(
                                                    wgpu::RenderPassColorAttachment {
                                                        view: &gv,
                                                        depth_slice: None,
                                                        resolve_target: None,
                                                        ops: wgpu::Operations {
                                                            load: match done {
                                                                0 => wgpu::LoadOp::Clear(
                                                                    wgpu::Color::BLACK,
                                                                ),
                                                                _ => wgpu::LoadOp::Load,
                                                            },
                                                            store: wgpu::StoreOp::Store,
                                                        },
                                                    },
                                                )],
                                                ..Default::default()
                                            });
                                        rp.set_viewport(r[0], r[1], CW as f32, CH as f32, 0.0, 1.0);
                                        pv.draw(&mut rp, &pano);
                                    }
                                    g.queue.submit([e.finish()]);
                                    done += 1;
                                    *p.st.lock().unwrap_or_else(|e| e.into_inner()) =
                                        ai::St::Run(done, n);
                                }
                                ctx.request_repaint();
                                Ok(())
                            })?;
                            out = Some((
                                stitch::read_texture(&g.device, &g.queue, &grid),
                                2 * CW,
                                2 * CH,
                            ));
                        }
                        _ => {}
                    }
                    out.ok_or_else(|| "nothing to preview".to_string())
                });
                *p.st.lock().unwrap_or_else(|e| e.into_inner()) = match r {
                    Ok(x) => {
                        *p.px.lock().unwrap_or_else(|e| e.into_inner()) = Some(x);
                        ai::St::Done
                    }
                    Err(_) if p.stop.load(Relaxed) => ai::St::Cancelled,
                    Err(e) => ai::St::Failed(e),
                };
                ctx.request_repaint();
            })
            .expect("shot lab preview thread");
        self.pre = Some((self.fx, pre));
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, exp: crate::export::Set) {
        egui::Panel::left("lab list")
            .default_size(190.0)
            .show(ui, |ui| {
                ui.heading("Shot Lab");
                for (i, (f, n, _, d)) in FXS.iter().enumerate() {
                    if i == 7 {
                        ui.separator();
                        ui.label("Editing presets");
                    }
                    if ui
                        .selectable_label(self.fx == *f, *n)
                        .on_hover_text(*d)
                        .clicked()
                        && self.fx != *f
                    {
                        (self.fx, self.err) = (*f, None);
                    }
                }
            });
        egui::CentralPanel::default_margins().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| self.panel(ui, exp));
        });
    }

    fn panel(&mut self, ui: &mut egui::Ui, exp: crate::export::Set) {
        let Some(s) = self.src.clone() else {
            ui.label(
                "Open an X5 video (.insv with both lenses) in the library: Shot Lab works on it.",
            );
            return;
        };
        let fx = self.fx;
        ui.heading(fx.name());
        ui.label(FXS.iter().find(|f| f.0 == fx).unwrap().3);
        ui.horizontal_wrapped(|ui| {
            ui.label("Clip");
            ui.monospace(
                s.files[0]
                    .file_name()
                    .map_or(String::new(), |n| n.to_string_lossy().into()),
            );
            ui.label(format!("{:.2} s", s.dur));
        });
        let p = &mut self.p;
        egui::Grid::new("lab set")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                ui.label("Range");
                ui.horizontal(|ui| {
                    let [a, b] = &mut self.range;
                    let dv = |v| {
                        egui::DragValue::new(v)
                            .range(0.0..=s.dur)
                            .speed(0.05)
                            .max_decimals(2)
                            .suffix(" s")
                    };
                    ui.add(dv(a));
                    ui.label("to");
                    ui.add(dv(b));
                    *a = a.min(*b);
                    if ui.button("Whole clip").clicked() {
                        self.range = [0.0, s.dur];
                    }
                });
                ui.end_row();
                let sl = |ui: &mut egui::Ui, n: &str, w: egui::Slider| {
                    ui.label(n);
                    ui.add(w);
                    ui.end_row();
                };
                match fx {
                    Fx::Passerby => sl(ui, "Frames", egui::Slider::new(&mut p.n, 2..=64)),
                    Fx::Trails => {
                        sl(
                            ui,
                            "Threshold",
                            egui::Slider::new(&mut p.light.thr, 0.0..=1.0),
                        );
                        sl(
                            ui,
                            "Softness",
                            egui::Slider::new(&mut p.light.soft, 0.01..=1.0),
                        );
                        sl(
                            ui,
                            "Half-life (0: keep)",
                            egui::Slider::new(&mut p.half, 0.0..=20.0).suffix(" s"),
                        );
                        sl(ui, "Gain", egui::Slider::new(&mut p.light.gain, 0.1..=4.0));
                        ui.label("Blend");
                        ui.checkbox(&mut p.light.add, "Add light (else lighten)");
                        ui.end_row();
                    }
                    Fx::Stars => {
                        ui.label("Stack");
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut p.max, false, "Mean (less noise)");
                            ui.selectable_value(&mut p.max, true, "Maximum (star trails)");
                        });
                        ui.end_row();
                    }
                    Fx::Delay => {
                        sl(
                            ui,
                            "Largest delay",
                            egui::Slider::new(&mut p.span, 0.05..=2.0).suffix(" s"),
                        );
                        ui.label("Lag by");
                        ui.horizontal_wrapped(|ui| {
                            for (i, n) in ["Longitude", "Angle from the view", "Columns", "Rows"]
                                .into_iter()
                                .enumerate()
                            {
                                ui.selectable_value(&mut p.map, i as u8, n);
                            }
                            ui.checkbox(&mut p.rev, "Reverse");
                        });
                        ui.end_row();
                    }
                    Fx::Shadow => {
                        sl(
                            ui,
                            "Spacing (subject widths)",
                            egui::Slider::new(&mut p.gap, 0.3..=4.0),
                        );
                        sl(ui, "Copies", egui::Slider::new(&mut p.copies, 1..=16));
                    }
                    Fx::Loop => {
                        sl(ui, "Selves", egui::Slider::new(&mut p.copies, 1..=8));
                        sl(
                            ui,
                            "Apart",
                            egui::Slider::new(&mut p.lag, 0.1..=2.0).suffix(" s"),
                        );
                    }
                    Fx::Planets => {
                        ui.label("Layout");
                        ui.checkbox(&mut p.tall, "Stacked");
                        ui.end_row();
                    }
                    Fx::StopMotion => {
                        sl(
                            ui,
                            "Step",
                            egui::Slider::new(&mut p.step, 0.05..=5.0).suffix(" s"),
                        );
                        sl(
                            ui,
                            "Speed",
                            egui::Slider::new(&mut p.speed, 1.0..=16.0).suffix("×"),
                        );
                    }
                    Fx::FlyLapse | Fx::Flip => {
                        sl(
                            ui,
                            "Speed",
                            egui::Slider::new(&mut p.speed, 1.0..=16.0).suffix("×"),
                        );
                        sl(ui, "Turns", egui::Slider::new(&mut p.turns, -4.0..=4.0));
                    }
                    Fx::Center | Fx::Spin | Fx::FlyOut => {
                        sl(
                            ui,
                            "Subject height",
                            egui::Slider::new(&mut p.fill, 0.1..=1.0),
                        );
                        if fx == Fx::Spin {
                            sl(ui, "Turns", egui::Slider::new(&mut p.turns, -4.0..=4.0));
                        }
                        if fx == Fx::FlyOut {
                            sl(
                                ui,
                                "Pull out over",
                                egui::Slider::new(&mut p.out, 0.5..=10.0).suffix(" s"),
                            );
                        }
                    }
                    Fx::Dolly => sl(
                        ui,
                        "First field of view",
                        egui::Slider::new(&mut p.fov, 20.0..=150.0).suffix("°"),
                    ),
                }
                ui.label("Output");
                let pick = if fx.cut() {
                    &mut self.flat
                } else {
                    &mut self.pano
                };
                egui::ComboBox::from_id_salt("lab preset")
                    .selected_text(PRESETS[*pick].1)
                    .show_ui(ui, |ui| {
                        for (i, x) in PRESETS.iter().enumerate().filter(|x| x.1.2 != fx.cut()) {
                            ui.selectable_value(pick, i, x.1);
                        }
                    });
                ui.end_row();
            });
        let warn = ui.visuals().warn_fg_color;
        if fx.still() {
            match &self.turn {
                Some((_, _, Some(d))) if *d > TURN => {
                    ui.colored_label(
                        warn,
                        format!(
                            "⚠ The camera turned {d:.1}° over the range (gyro): {} needs a still camera. The direction \
                             lock takes the turning out, not a camera that moves.",
                            fx.name()
                        ),
                    );
                }
                Some((_, _, Some(d))) => {
                    ui.label(format!(
                        "The camera held still: it turned {d:.1}° over the range (gyro)."
                    ));
                }
                Some((_, _, None)) => {
                    ui.colored_label(
                        warn,
                        "⚠ No gyro record: whether the camera held still is unknown.",
                    );
                }
                None => {
                    ui.label("Checking the gyro…");
                }
            }
        }
        if fx.subject() {
            match self.trk.as_ref().filter(|t| t.file == s.files[0]) {
                Some(t) => {
                    let n = t.smp.lock().unwrap_or_else(|e| e.into_inner()).len();
                    ui.label(format!(
                        "Subject: the player's Deep Track run, {n} frames from frame {}.",
                        t.k0
                    ));
                }
                None => {
                    ui.colored_label(warn, "Track a subject in the player first (Deep Track)");
                }
            }
        }
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let busy = self.pre.is_some();
            if ui
                .add_enabled(!busy, egui::Button::new("Preview"))
                .clicked()
            {
                self.err = None;
                self.preview();
            }
            if let Some((_, p)) = &self.pre {
                ui.label(
                    p.st.lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .line("Preview"),
                );
                if ui.button("Cancel").clicked() {
                    p.stop.store(true, Relaxed);
                }
            }
            if ui
                .button("Export")
                .on_hover_text("Queue it on the Export screen")
                .clicked()
                && let Err(e) = self.export(Some(exp))
            {
                self.err = Some(e);
            }
            if fx.cut()
                && ui
                    .button("Add to timeline")
                    .on_hover_text("Its clips onto the end of the editor's timeline")
                    .clicked()
            {
                match self.timeline() {
                    Ok((tl, st)) => self.ask = Some(Ask::Editor(tl, st)),
                    Err(e) => self.err = Some(e),
                }
            }
        });
        if let Some(e) = &self.err {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        if let Some((f, t)) = self.tex.as_ref().filter(|t| t.0 == fx) {
            let w = ui.available_width().min(1400.0);
            let sz = t.size_vec2();
            ui.label(match f.cut() {
                true => {
                    "Preview: the start, a third, two thirds and the end through the preset's views"
                }
                false => "Preview: the out point of the range (360)",
            });
            ui.image((t.id(), egui::vec2(w, w * sz.y / sz.x)));
        }
    }
}

/// Hands a Shot Lab request to the Export screen or the editor; true when it went to the editor.
pub fn send(a: Ask, exp: &mut Exp, ed: &mut crate::editor::Editor) -> Result<bool, String> {
    match a {
        Ask::Export(s, x, r, j) => exp.lab(s, x, r, j).map(|_| false),
        Ask::Timeline(t, x) => exp.timeline(t, x).map(|_| false),
        Ask::Editor(tl, st) => ed.put(tl, st).map(|_| true),
    }
}

#[cfg(test)]
mod tests;
