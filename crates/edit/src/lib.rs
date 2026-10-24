//! Editor model (PLAN.md milestone 7, group 10): a timeline of clips cut from source files with speed (constant,
//! ramps, freeze frames, reverse), reframing keyframes and camera-movement presets, look and audio hooks, undo and
//! project files. Pure data: the player and the exporter ask [`Timeline::at`] / [`Timeline::frame`] what to render
//! and [`Timeline::segs`] what to decode; every change goes through [`Project`], which keeps the undo log.
//!
//! # Time
//!
//! - Output frame n starts at n·den/num s ([`Rate`]) and shows its clip at that instant. Each clip fills a whole
//!   number of frames, max(1, ⌊exact output duration · fps + ½⌋), so cuts land on frames; the sub-frame the
//!   rounding adds plays the clip's last source instant, the one it drops is not shown. A duration within 10⁻⁶
//!   frames below a half rounds up, so both parts of a split keep the frames of the whole (13 source frames at 2×:
//!   6.5 frames, split into 3 + 3.5).
//! - Inside a clip, output time and source time map through the speed curve in closed form both ways
//!   ([`Clip::src`] / [`Clip::out`]): exact inverses, strictly increasing. Speed is linear in source time between
//!   speed keys, so the integral is a logarithm and its inverse an exponential.
//! - A frame shows the source frame with the latest pts ≤ the source time given (+ 1 µs for rounding). A reversed
//!   clip samples at the lower end of each output frame's source interval, so the same rule holds backwards.
//! - Content time: a clip's view keys, volume envelope and speed keys sit on its source time, so trim, split and speed
//!   changes leave them on the same picture; a freeze frame's keys sit on its hold time. The output timeline's view
//!   keys sit on output time and stay put when clips change.

// `!(x > 0.0)` rejects NaN on purpose
#![allow(clippy::neg_cmp_op_on_partial_ord)]

mod file;
mod project;
pub mod view;

pub use project::Project;
pub use view::{Ease, Key, Proj, View};

use std::{fmt, ops::Range};

pub const MIN_SPEED: f64 = 0.1;
pub const MAX_SPEED: f64 = 16.0;
/// Music gain factor while clip audio plays and ducking is on (about −10 dB).
pub const DUCK: f32 = 0.3;
/// Longest freeze frame and longest source file, s: keeps frame counts far from overflow.
pub const MAX_HOLD: f64 = 3600.0;
pub const MAX_FILE: f64 = 1.0e6;
/// A time this many frames before a frame boundary counts as the boundary (absorbs the rounding of n·den/num).
const EPS: f64 = 1e-6;
/// Closest spacing and largest |time| of speed, volume and view keys, s: keep ramp slopes, view tangents and
/// differences of key times finite.
pub const GAP: f64 = 1e-6;
pub const MAX_T: f64 = 1e9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn bad<T>(m: impl Into<String>) -> Result<T> {
    Err(Error(m.into()))
}

/// Output frame rate num / den (30000 / 1001 = 29.97).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rate {
    pub num: u32,
    pub den: u32,
}

impl Rate {
    /// Start of frame n, s (one rounding).
    pub fn t(self, n: u64) -> f64 {
        (n as f64 * self.den as f64) / self.num as f64
    }

    /// `t` s in frames.
    pub fn n(self, t: f64) -> f64 {
        t * self.num as f64 / self.den as f64
    }
}

/// A piece of a source file on the timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    /// Index into [`Timeline::files`].
    pub file: u32,
    /// Source in and out, s: source times a..b play. A frame shows the source frame with the latest pts ≤ its source
    /// time, so an in point between two frames starts on the earlier one. A freeze frame shows `a` (b = a).
    pub a: f64,
    pub b: f64,
    /// Speed keys (source time, speed in [`MIN_SPEED`, `MAX_SPEED`]), source times increasing by at least [`GAP`];
    /// speed is linear in source time between keys and constant beyond them. Empty: 1×; one key: constant.
    pub speed: Vec<(f64, f64)>,
    /// > 0: a freeze frame held this long (output s); `speed` and `rev` do not apply.
    pub hold: f64,
    /// Plays b → a.
    pub rev: bool,
    /// Reframing keys in content time (source s; hold s on a freeze). Empty: the output timeline's keys apply. A
    /// key's ease shapes the motion that follows it on screen, on a reversed clip towards lower source times.
    pub view: Vec<Key>,
    /// Look (crate fx) id, "" = none, and its intensity.
    pub look: String,
    pub amt: f32,
    /// Audio gain envelope (content time, points at least [`GAP`] apart, linear gain 0..=16), linear between points,
    /// constant beyond. Empty: 1.
    pub vol: Vec<(f64, f32)>,
    pub mute: bool,
}

impl Clip {
    /// Source `a..b` of `file` at 1×.
    pub fn new(file: u32, a: f64, b: f64) -> Clip {
        Clip {
            file,
            a,
            b,
            speed: vec![],
            hold: 0.0,
            rev: false,
            view: vec![],
            look: String::new(),
            amt: 1.0,
            vol: vec![],
            mute: false,
        }
    }

    /// A freeze frame: source time `s` of `file` held `hold` s.
    pub fn still(file: u32, s: f64, hold: f64) -> Clip {
        Clip {
            hold,
            ..Clip::new(file, s, s)
        }
    }

    /// The speed piece at source time x, going forwards: speed at x, slope (speed per source s), end of the piece.
    fn piece(&self, x: f64) -> (f64, f64, f64) {
        let s = &self.speed;
        let i = s.partition_point(|k| k.0 <= x);
        match s.len() {
            0 => (1.0, 0.0, f64::INFINITY),
            _ if i == 0 => (s[0].1, 0.0, s[0].0),
            n if i == n => (s[n - 1].1, 0.0, f64::INFINITY),
            _ => {
                let (p, q) = (s[i - 1], s[i]);
                let k = (q.1 - p.1) / (q.0 - p.0);
                (p.1 + k * (x - p.0), k, q.0)
            }
        }
    }

    /// Speed at source time `s`.
    pub fn rate(&self, s: f64) -> f64 {
        self.piece(s).0
    }

    /// Output s from `a` to source time `s` ≥ a, playing forwards: ∫ ds / speed.
    pub fn out(&self, s: f64) -> f64 {
        let (mut x, mut u) = (self.a, 0.0);
        while x < s {
            let (v, k, e) = self.piece(x);
            let y = e.min(s);
            u += int(y - x, v, k);
            x = y;
        }
        u
    }

    /// Source time `u` ≥ 0 output s after `a`, playing forwards: the inverse of [`Clip::out`].
    pub fn src(&self, mut u: f64) -> f64 {
        let mut x = self.a;
        loop {
            let (v, k, e) = self.piece(x);
            let d = if e.is_finite() {
                int(e - x, v, k)
            } else {
                f64::INFINITY
            };
            // NaN stops here too
            if !(u > d) {
                return x + ext(u, v, k);
            }
            (u, x) = (u - d, e);
        }
    }

    /// Exact output duration, s.
    pub fn dur(&self) -> f64 {
        if self.hold > 0.0 {
            self.hold
        } else {
            self.out(self.b)
        }
    }
}

/// ∫₀ᵈ dx / (v + kx): output s over d source s from speed v with slope k.
fn int(d: f64, v: f64, k: f64) -> f64 {
    let z = k * d / v;
    d / v
        * if z.abs() < 1e-9 {
            1.0 - z / 2.0
        } else {
            z.ln_1p() / z
        }
}

/// Source s covered in u output s from speed v with slope k (ds/dt = v + k·s): the inverse of `int`.
fn ext(u: f64, v: f64, k: f64) -> f64 {
    let w = k * u;
    v * u
        * if w.abs() < 1e-9 {
            1.0 + w / 2.0
        } else {
            w.exp_m1() / w
        }
}

/// A source file: path as the app opened it (the primary file of an `.insv` pair) and its duration, s.
#[derive(Clone, Debug, PartialEq)]
pub struct File {
    pub path: String,
    pub dur: f64,
}

/// Background music.
#[derive(Clone, Debug, PartialEq)]
pub struct Music {
    pub file: u32,
    /// Music time = output time + offset (negative: the music starts later).
    pub offset: f64,
    pub gain: f32,
    /// Lower the music by [`DUCK`] while clip audio plays.
    pub duck: bool,
}

/// The timeline-wide settings (one undo unit).
#[derive(Clone, Debug, PartialEq)]
pub struct Out {
    pub fps: Rate,
    /// Reframing keys on output time; a clip with keys of its own uses those instead.
    pub view: Vec<Key>,
    pub music: Option<Music>,
}

/// A keyframe track: the output timeline's or clip k's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum On {
    Out,
    Clip(usize),
}

/// What to render at one output instant.
#[derive(Clone, Copy, Debug)]
pub struct Eval<'a> {
    pub k: usize,
    pub clip: &'a Clip,
    /// Source time to show, s (the source frame with the latest pts ≤ src + 1 µs).
    pub src: f64,
    /// Source s per output s here (0 on a freeze frame); the source runs backwards when `clip.rev`.
    pub speed: f64,
    pub view: View,
    /// Clip audio gain: the envelope, 0 when muted or frozen.
    pub gain: f32,
    /// Music file time, s, and gain, while the music plays.
    pub music: Option<(f64, f32)>,
}

/// A run of output frames from one clip that decodes in one sequential pass, split at the speed keys where the speed
/// curve bends.
#[derive(Clone, Debug, PartialEq)]
pub struct Seg {
    pub k: usize,
    pub file: u32,
    pub frames: Range<u64>,
    /// Source times of the first and last frame (descending when `rev`, equal on a freeze frame).
    pub src: [f64; 2],
    /// Speed at the first and last frame: equal on a run of constant speed, 0 on a freeze frame.
    pub speed: [f64; 2],
    pub rev: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Timeline {
    pub files: Vec<File>,
    pub clips: Vec<Clip>,
    pub out: Out,
    /// Per clip: end frame (exclusive), exact output duration. Rebuilt by `fix`.
    span: Vec<(u64, f64)>,
}

impl Timeline {
    pub fn new(fps: Rate) -> Timeline {
        Timeline {
            files: vec![],
            clips: vec![],
            out: Out {
                fps,
                view: vec![],
                music: None,
            },
            span: vec![],
        }
    }

    /// Recomputes the frame spans; call after changing `clips` or `out.fps` directly ([`Project`] does).
    pub fn fix(&mut self) {
        let (fps, mut e) = (self.out.fps, 0u64);
        self.span = self
            .clips
            .iter()
            .map(|c| {
                let d = c.dur();
                e += ((fps.n(d) + 0.5 + EPS).floor() as u64).max(1);
                (e, d)
            })
            .collect();
    }

    /// Output length, frames.
    pub fn len(&self) -> u64 {
        self.span.last().map_or(0, |s| s.0)
    }

    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Output length, s.
    pub fn dur(&self) -> f64 {
        self.out.fps.t(self.len())
    }

    /// Clip k's output frames.
    pub fn frames(&self, k: usize) -> Range<u64> {
        (if k == 0 { 0 } else { self.span[k - 1].0 })..self.span[k].0
    }

    /// The clip showing frame n.
    pub fn find(&self, n: u64) -> Option<usize> {
        let k = self.span.partition_point(|s| s.0 <= n);
        (k < self.span.len()).then_some(k)
    }

    /// The frame showing at `t` s.
    pub fn frame_at(&self, t: f64) -> Option<u64> {
        let x = (self.out.fps.n(t) + EPS).floor();
        (x >= 0.0 && x < self.len() as f64).then_some(x as u64)
    }

    /// What to show at output time `t` (continuous: the view and source time move within a frame).
    pub fn at(&self, t: f64) -> Option<Eval<'_>> {
        let k = self.find(self.frame_at(t)?)?;
        let t0 = self.out.fps.t(self.frames(k).start);
        Some(self.eval(k, (t - t0).max(0.0), t))
    }

    /// What to show in output frame `n` (exact frame arithmetic).
    pub fn frame(&self, n: u64) -> Option<Eval<'_>> {
        let (k, fps) = (self.find(n)?, self.out.fps);
        Some(self.eval(k, fps.t(n - self.frames(k).start), fps.t(n)))
    }

    /// Source time clip k (moving) shows `tau` s into it.
    fn sample(&self, k: usize, tau: f64) -> f64 {
        let (c, d) = (&self.clips[k], self.span[k].1);
        let u = if c.rev {
            d - tau - self.out.fps.t(1)
        } else {
            tau
        };
        c.src(u.clamp(0.0, d)).clamp(c.a, c.b)
    }

    /// Clip k's content time `tau` s into it: source time, or hold time on a freeze frame.
    pub fn content(&self, k: usize, tau: f64) -> f64 {
        let c = &self.clips[k];
        if c.hold > 0.0 {
            tau.clamp(0.0, c.hold)
        } else {
            self.sample(k, tau)
        }
    }

    /// Inverse of [`Timeline::content`]: s into clip k, clamped to its frames.
    pub fn local(&self, k: usize, x: f64) -> f64 {
        let (c, d, n, dt) = (
            &self.clips[k],
            self.span[k].1,
            self.frames(k),
            self.out.fps.t(1),
        );
        let tau = match (c.hold > 0.0, c.rev) {
            (true, _) => x,
            (_, false) => c.out(x.clamp(c.a, c.b)),
            (_, true) => d - dt - c.out(x.clamp(c.a, c.b)),
        };
        tau.clamp(0.0, self.out.fps.t(n.end - n.start))
    }

    pub fn track(&self, on: On) -> Option<&[Key]> {
        match on {
            On::Out => Some(&self.out.view),
            On::Clip(k) => self.clips.get(k).map(|c| &c.view[..]),
        }
    }

    /// Output time `t` on track `on`'s time axis; None outside the clip.
    pub fn to_track(&self, on: On, t: f64) -> Option<f64> {
        let On::Clip(k) = on else {
            return t.is_finite().then_some(t);
        };
        self.clips.get(k)?;
        let n = self.frames(k);
        let tau = t - self.out.fps.t(n.start);
        let tol = 0.5 * self.out.fps.t(1);
        (tau >= -tol && tau <= self.out.fps.t(n.end - n.start) + tol)
            .then(|| self.content(k, tau.max(0.0)))
    }

    /// Track time `x` on output time; None for a clip key outside the clip's content.
    pub fn from_track(&self, on: On, x: f64) -> Option<f64> {
        let On::Clip(k) = on else { return Some(x) };
        let c = self.clips.get(k)?;
        let (lo, hi) = if c.hold > 0.0 {
            (0.0, c.hold)
        } else {
            (c.a, c.b)
        };
        (x >= lo && x <= hi).then(|| self.out.fps.t(self.frames(k).start) + self.local(k, x))
    }

    /// The key of track `on` within half a frame of output time `t`.
    pub fn near(&self, on: On, t: f64) -> Option<usize> {
        let tol = 0.5 * self.out.fps.t(1);
        (self.track(on)?.iter()).position(|key| {
            self.from_track(on, key.t)
                .is_some_and(|x| (x - t).abs() <= tol)
        })
    }

    fn eval(&self, k: usize, tau: f64, t: f64) -> Eval<'_> {
        let c = &self.clips[k];
        let (x, frozen) = (self.content(k, tau), c.hold > 0.0);
        let gain = if c.mute || frozen {
            0.0
        } else {
            env(&c.vol, x)
        };
        let view = match c.view.is_empty() {
            true => view::at(&self.out.view, t, false),
            false => view::at(&c.view, x, c.rev && !frozen),
        };
        let music = self.out.music.as_ref().and_then(|m| {
            let mt = t + m.offset;
            let duck = if m.duck && gain > 0.0 { DUCK } else { 1.0 };
            (mt >= 0.0 && mt < self.files.get(m.file as usize)?.dur).then_some((mt, m.gain * duck))
        });
        Eval {
            k,
            clip: c,
            src: if frozen { c.a } else { x },
            speed: if frozen { 0.0 } else { c.rate(x) },
            view: view.unwrap_or_default(),
            gain,
            music,
        }
    }

    /// The decode runs in output order: one per clip, moving clips split where a speed key falls inside (not at a key
    /// with the same speed on both sides).
    pub fn segs(&self) -> Vec<Seg> {
        let (fps, mut v) = (self.out.fps, vec![]);
        for (k, c) in self.clips.iter().enumerate() {
            let r = self.frames(k);
            let n = r.end - r.start;
            if c.hold > 0.0 {
                v.push(Seg {
                    k,
                    file: c.file,
                    frames: r,
                    src: [c.a; 2],
                    speed: [0.0; 2],
                    rev: false,
                });
                continue;
            }
            let mut run = |f0: u64, f1: u64| {
                if f1 > f0 {
                    let src = [self.sample(k, fps.t(f0)), self.sample(k, fps.t(f1 - 1))];
                    v.push(Seg {
                        k,
                        file: c.file,
                        frames: r.start + f0..r.start + f1,
                        src,
                        speed: src.map(|s| c.rate(s)),
                        rev: c.rev,
                    });
                }
            };
            let (d, dt, m) = (self.span[k].1, fps.t(1), c.speed.len());
            let mut f0 = 0;
            let sp = &c.speed;
            for i in 0..m {
                let j = if c.rev { m - 1 - i } else { i };
                let (s, v) = sp[j];
                let flat = |o: Option<usize>| o.and_then(|o| sp.get(o)).is_none_or(|q| q.1 == v);
                if s > c.a && s < c.b && !(flat(j.checked_sub(1)) && flat(Some(j + 1))) {
                    let u = c.out(s);
                    let x = fps.n(if c.rev { d - dt - u } else { u });
                    // first frame on the far side of the key
                    let f = (x - EPS).ceil().clamp(f0 as f64, n as f64) as u64;
                    run(f0, f);
                    f0 = f;
                }
            }
            run(f0, n);
        }
        v
    }

    /// Checks every invariant the evaluation relies on (load and [`Project`] call it).
    pub fn check(&self) -> Result<()> {
        let Rate { num, den } = self.out.fps;
        if num == 0 || den == 0 || num as u64 > 1000 * den as u64 || den as u64 > 1000 * num as u64
        {
            return bad(format!("frame rate {num}/{den}"));
        }
        for (i, f) in self.files.iter().enumerate() {
            if !(f.dur > 0.0 && f.dur <= MAX_FILE) {
                return bad(format!("file {i}: duration {}", f.dur));
            }
        }
        for (i, c) in self.clips.iter().enumerate() {
            self.ok(c)
                .map_err(|e| Error(format!("clip {i}: {}", e.0)))?;
        }
        view::check(&self.out.view).map_err(|e| Error(format!("output view: {}", e.0)))?;
        if let Some(m) = &self.out.music
            && (m.file as usize >= self.files.len()
                || !m.offset.is_finite()
                || !(m.gain >= 0.0 && m.gain <= 16.0))
        {
            return bad("music");
        }
        Ok(())
    }

    /// Checks one clip against the file table.
    pub fn ok(&self, c: &Clip) -> Result<()> {
        let Some(f) = self.files.get(c.file as usize) else {
            return bad(format!("no file {}", c.file));
        };
        if c.hold != 0.0 {
            if !(c.hold > 0.0 && c.hold <= MAX_HOLD && c.a >= 0.0 && c.a <= f.dur && c.b == c.a) {
                return bad("freeze frame");
            }
        } else if !(c.a >= 0.0 && c.a < c.b && c.b <= f.dur) {
            return bad(format!("source {}..{} outside 0..{}", c.a, c.b, f.dur));
        }
        if !inc(&c.speed) || !c.speed.iter().all(|k| k.1 >= MIN_SPEED && k.1 <= MAX_SPEED) {
            return bad("speed keys");
        }
        if !inc(&c.vol) || !c.vol.iter().all(|k| k.1 >= 0.0 && k.1 <= 16.0) {
            return bad("volume keys");
        }
        if !c.amt.is_finite() {
            return bad("look intensity");
        }
        view::check(&c.view)
    }
}

/// Times within ±[`MAX_T`], at least [`GAP`] apart.
fn inc<T>(v: &[(f64, T)]) -> bool {
    v.iter().all(|p| p.0.abs() <= MAX_T) && v.windows(2).all(|w| w[1].0 - w[0].0 >= GAP)
}

/// Piecewise-linear envelope, constant beyond its ends; empty: 1.
fn env(v: &[(f64, f32)], x: f64) -> f32 {
    let i = v.partition_point(|p| p.0 <= x);
    match (i.checked_sub(1).map(|j| v[j]), v.get(i).copied()) {
        (Some(a), Some(b)) => a.1 + (b.1 - a.1) * ((x - a.0) / (b.0 - a.0)) as f32,
        (Some(p), None) | (None, Some(p)) => p.1,
        (None, None) => 1.0,
    }
}

#[cfg(test)]
mod tests;
