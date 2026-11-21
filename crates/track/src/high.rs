//! Highlights and auto edit: a clip is scored in `BIN` s bins by five signals, each scaled within the clip between
//! its 10th and 90th percentile (with a floor on that spread, so a flat signal's noise stays small) and weighted:
//! - camera motion: angular speed of the camera from the trailer gyro (`stab::Track`) over 0.5 s, so pans and
//!   turns count and shake mostly averages out;
//! - visual change: mean difference of consecutive thumbnails of the stabilised sphere ([`Ai::thumb`]): things
//!   moving, new places;
//! - people: the persons the detector sees (weighted by size, so near ones count most) and the faces;
//! - audio: loudness (RMS in dB) of the file's AAC track;
//! - overtakes: vehicles overtaking close by ([`crate::Pass`]), 1 over each one from the rear quarter to the front.
//!
//! The score is smoothed (Gaussian, σ 1 s); highlights grow from the best bins while the score stays above the median,
//! 2 to 8 s each, until they fill the duration asked for. [`fit`] moves their cuts onto a music track's beats;
//! [`timeline`] cuts them into an `edit::Timeline` in clip order, each reframed along the auto framer's path
//! ([`crate::Framer`]).

use crate::{Ai, Cam, Crops, Det, In, Job, Pano, RGB, Region, Shot, read, yaw_pitch};
use glam::{Vec2, Vec3};
use std::{io, path::Path};

/// Bin width, s.
pub const BIN: f64 = 0.5;

/// What a value pushed to [`Scorer`] measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sig {
    /// Camera angular speed, rad/s.
    Motion,
    /// Thumbnail difference per second (0..1 colour units).
    Change,
    /// People seen (`Scorer::people`).
    People,
    /// Loudness, dB (RMS, full scale 0).
    Audio,
    /// A vehicle overtaking close by (`Scorer::passes`).
    Pass,
}

/// Weights and the floors of the 10–90 % spread, per `Sig`. The first four keep their ratios of before overtakes
/// counted (0.2, 0.25, 0.35, 0.2), so a clip without one scores in proportion and gets the same highlights.
const W: [f32; 5] = [0.16, 0.2, 0.28, 0.16, 0.2];
const FLOOR: [f32; 5] = [0.3, 0.02, 1.0, 6.0, 1.0];
/// Highlight length, s; the shortest [`fit`] trims one to.
const MIN: f64 = 2.0;
const MAX: f64 = 8.0;
const SHORT: f64 = 1.0;

/// A highlight: source range, s, its mean score, and its scored span `lo..hi` around it (on while the score stays
/// at the median, at most halfway to its neighbours), which [`fit`] may extend it into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Highlight {
    pub a: f64,
    pub b: f64,
    pub score: f32,
    pub lo: f64,
    pub hi: f64,
}

pub struct Scorer {
    /// Clip duration, s.
    pub dur: f64,
    /// Per bin and signal: sum, count.
    acc: Vec<[(f32, u32); 5]>,
    /// The last thumbnail and its time.
    last: Option<(f64, Vec<f32>)>,
}

impl Scorer {
    pub fn new(dur: f64) -> Self {
        Self {
            dur,
            acc: vec![[(0.0, 0); 5]; (dur / BIN).ceil().max(1.0) as usize],
            last: None,
        }
    }

    /// A value of signal `s` at clip time `t`, s.
    pub fn push(&mut self, s: Sig, t: f64, v: f32) {
        let i = ((t / BIN) as usize).min(self.acc.len() - 1);
        let a = &mut self.acc[i][s as usize];
        (a.0, a.1) = (a.0 + v, a.1 + 1);
    }

    /// Camera motion from the gyro: every 0.1 s, the angle the camera turned over the 0.5 s around it. `t0`: the time
    /// of clip time 0 on the IMU clock (the first frame's exposure record).
    pub fn gyro(&mut self, tr: &stab::Track, t0: f64) {
        for i in 0..(self.dur * 10.0) as usize {
            let t = i as f64 / 10.0;
            let w = tr.turn(t0 + t - 0.25, t0 + t + 0.25).length() / 0.5;
            self.push(Sig::Motion, t, w as f32);
        }
    }

    /// Visual change: thumbnail `th` (`Ai::thumb`) of the frame at clip time `t`, against the last one.
    pub fn frame(&mut self, t: f64, th: Vec<f32>) {
        if let Some((l, p)) = &self.last
            && t > *l
            && p.len() == th.len()
        {
            let d = p.iter().zip(&th).map(|(a, b)| (a - b).abs()).sum::<f32>() / th.len() as f32;
            self.push(Sig::Change, t, d / (t - l) as f32);
        }
        self.last = Some((t, th));
    }

    /// People at clip time `t`: persons of `d` (`Ai::detect` over `sphere()`) by their height (a 15° tall one, ~4 m
    /// away, counts 1), each face of `f` 0.5.
    pub fn people(&mut self, t: f64, d: &[Det], f: &[Region]) {
        let v = d
            .iter()
            .filter(|x| x.cls == 0)
            .map(|x| x.score * (x.size.y / 15.0).min(1.5))
            .sum::<f32>()
            + 0.5 * f.len() as f32;
        self.push(Sig::People, t, v);
    }

    /// Loudness of the file's first AAC track in 0.1 s windows (RMS over channels, dB). The edit list's priming
    /// (~20 ms) is ignored.
    pub fn audio(&mut self, path: &Path) -> io::Result<()> {
        use media::mp4::{Codec, Mp4};
        let m = Mp4::open(path)?;
        let Some(k) = m
            .tracks
            .iter()
            .find(|t| matches!(t.codec, Codec::Aac { .. }))
        else {
            return Ok(());
        };
        let Codec::Aac { asc, .. } = &k.codec else {
            unreachable!()
        };
        let mut dec = media::audio::Aac::new(asc)?;
        let (r, c) = (dec.rate as f64, dec.ch.max(1) as usize);
        let (mut buf, mut e, mut n, mut w) = (vec![], 0.0f64, 0usize, 0usize);
        let win = (r / 10.0) as usize;
        for s in &k.samples {
            m.read(s, &mut buf)?;
            // a bad packet costs its 21 ms, not the clip's loudness
            let Ok(x) = dec.decode(&buf) else { continue };
            for f in x.chunks_exact(c) {
                e += f.iter().map(|v| (v * v) as f64).sum::<f64>() / c as f64;
                n += 1;
                if n == win {
                    let db = 10.0 * (e / n as f64).max(1e-10).log10();
                    self.push(Sig::Audio, (w as f64 + 0.5) / 10.0, db as f32);
                    (e, n, w) = (0.0, 0, w + 1);
                }
            }
        }
        Ok(())
    }

    /// Overtakes: 1 in the bins of each pass's span `a..b` (clip s), 0 in every other bin (a bin without a value
    /// would take its nearest's); none without passes.
    pub fn passes(&mut self, p: &[[f64; 2]]) {
        if p.is_empty() {
            return;
        }
        for i in 0..self.acc.len() {
            let (a, b) = (i as f64 * BIN, (i + 1) as f64 * BIN);
            let on = p.iter().any(|x| x[0] < b && x[1] > a);
            self.push(Sig::Pass, a, on as u8 as f32);
        }
    }

    /// Per bin: each signal's mean, filled across bins without samples from the nearest that has one, then scaled
    /// 0..1 between its 10th and 90th percentile.
    pub fn signals(&self) -> Vec<[f32; 5]> {
        let n = self.acc.len();
        let mut out = vec![[0.0; 5]; n];
        for s in 0..5 {
            let have: Vec<usize> = (0..n).filter(|&i| self.acc[i][s].1 > 0).collect();
            if have.is_empty() {
                continue;
            }
            let mut v: Vec<f32> = (0..n)
                .map(|i| {
                    let j = have[have.partition_point(|&j| j < i).min(have.len() - 1)];
                    let k = have[have.partition_point(|&j| j <= i).saturating_sub(1)];
                    let j = if i.abs_diff(k) < j.abs_diff(i) { k } else { j };
                    self.acc[j][s].0 / self.acc[j][s].1 as f32
                })
                .collect();
            let mut q = v.clone();
            q.sort_by(f32::total_cmp);
            let (lo, hi) = (q[n / 10], q[(n * 9 / 10).min(n - 1)]);
            let d = (hi - lo).max(FLOOR[s]);
            v.iter_mut()
                .for_each(|x| *x = ((*x - lo) / d).clamp(0.0, 1.0));
            for (o, x) in out.iter_mut().zip(v) {
                o[s] = x;
            }
        }
        out
    }

    /// The combined score per bin, smoothed (Gaussian, σ 1 s).
    pub fn score(&self) -> Vec<f32> {
        let s: Vec<f32> = self
            .signals()
            .iter()
            .map(|x| (0..5).map(|i| W[i] * x[i]).sum())
            .collect();
        let r = (2.0 / BIN) as isize;
        (0..s.len() as isize)
            .map(|k| {
                let (mut a, mut ws) = (0.0, 0.0);
                for j in (k - r).max(0)..(k + r + 1).min(s.len() as isize) {
                    let w = (-((j - k) as f32 * BIN as f32).powi(2) / 2.0).exp();
                    (a, ws) = (a + w * s[j as usize], ws + w);
                }
                a / ws
            })
            .collect()
    }

    /// Highlights filling about `total` s, in clip order: from the best bin not yet used, grown on the higher side
    /// while the score stays at or above the median, `MIN` to `MAX` s and at most what is left of `total` (but `MIN`),
    /// kept 1 s apart; then each one's scored span.
    pub fn pick(&self, total: f64) -> Vec<Highlight> {
        let s = self.score();
        let n = s.len();
        let mut q = s.clone();
        q.sort_by(f32::total_cmp);
        let med = q[n / 2];
        let (lo, hi) = ((MIN / BIN) as usize, (MAX / BIN) as usize);
        let mut used = vec![false; n];
        let (mut out, mut sum) = (vec![], 0.0);
        while sum < total {
            let Some(i) = (0..n)
                .filter(|&i| !used[i])
                .max_by(|&a, &b| s[a].total_cmp(&s[b]))
            else {
                break;
            };
            let free = |j: usize| j < n && !used[j];
            let cap = (((total - sum) / BIN).ceil() as usize).clamp(lo, hi);
            let (mut a, mut b) = (i, i + 1);
            while b - a < cap {
                let l = a.checked_sub(1).filter(|&j| free(j));
                let r = Some(b).filter(|&j| free(j));
                let pick = match (l, r) {
                    (Some(l), Some(r)) => Some(if s[l] >= s[r] { l } else { r }),
                    (l, r) => l.or(r),
                };
                match pick {
                    Some(j) if s[j] >= med || b - a < lo => {
                        if j < a {
                            a = j
                        } else {
                            b = j + 1
                        }
                    }
                    _ => break,
                }
            }
            // a range squeezed under `MIN` between others is no highlight; its bins are spent either way
            let gap = (1.0 / BIN) as usize;
            used[a.saturating_sub(gap)..(b + gap).min(n)].fill(true);
            if b - a < lo {
                continue;
            }
            let (ta, tb) = (a as f64 * BIN, (b as f64 * BIN).min(self.dur));
            out.push(Highlight {
                a: ta,
                b: tb,
                score: s[a..b].iter().sum::<f32>() / (b - a) as f32,
                lo: ta,
                hi: tb,
            });
            sum += tb - ta;
        }
        out.sort_by(|x, y| x.a.total_cmp(&y.a));
        for i in 0..out.len() {
            let l = if i > 0 {
                (out[i - 1].b + out[i].a) / 2.0
            } else {
                0.0
            };
            let r = out.get(i + 1).map_or(self.dur, |x| (out[i].b + x.a) / 2.0);
            let (mut a, mut b) = (
                (out[i].a / BIN).round() as usize,
                (out[i].b / BIN).ceil() as usize,
            );
            while a > 0 && s[a - 1] >= med && (a - 1) as f64 * BIN >= l {
                a -= 1;
            }
            while b < n && s[b] >= med && ((b + 1) as f64 * BIN).min(self.dur) <= r {
                b += 1;
            }
            (out[i].lo, out[i].hi) = (a as f64 * BIN, (b as f64 * BIN).min(self.dur));
        }
        out
    }
}

/// Moves the cuts of highlights `h` (clip order, played back to back from output time 0 at 1×, as [`timeline`] cuts
/// them at `fps`) onto beats `bt` (output times, s, ascending, with their strength: `media::beat::Beats::t`). In turn,
/// each one's end goes to the beat, on a whole frame, that its length reaches (at least `SHORT`, or its own length if
/// shorter, at most its scored span) and that is nearest its own end in beat periods, less 0.4 per rms of strength (up
/// to 2): a beat of twice the rms wins over a silent one up to 0.8 periods nearer. It ends earlier, or later and, past
/// its span's end, starts earlier. One no beat is in reach of stays as it is.
pub fn fit(h: &mut [Highlight], bt: &[(f64, f32)], fps: edit::Rate) {
    if bt.len() < 2 {
        return;
    }
    let q = (bt.iter().map(|b| b.1 * b.1).sum::<f32>() / bt.len() as f32).sqrt();
    let mut d: Vec<f64> = bt.windows(2).map(|w| fps.n(w[1].0 - w[0].0)).collect();
    d.sort_by(f64::total_cmp);
    let p = d[d.len() / 2];
    let cost = |w: f64, (f, s): (f64, f32)| {
        (f - w).abs() / p - 0.4 * (s / q.max(1e-6)).clamp(0.0, 2.0) as f64
    };
    // the clip's first output frame
    let mut f0 = 0u64;
    for x in h {
        let len = x.b - x.a;
        let at = |t: f64| f0 as f64 + fps.n(t);
        let (w, lo, hi) = (at(len), at(SHORT.min(len)) - 1e-6, at(x.hi - x.lo) + 1e-6);
        let best = (bt.iter())
            .map(|&(t, s)| (fps.n(t).round(), s))
            .filter(|&(f, _)| f >= lo && f <= hi)
            .min_by(|&a, &b| cost(w, a).total_cmp(&cost(w, b)));
        f0 = match best {
            Some((f, _)) => {
                let l = fps.t(f as u64 - f0);
                if x.a + l <= x.hi {
                    x.b = x.a + l;
                } else {
                    (x.a, x.b) = ((x.hi - l).max(x.lo), x.hi);
                }
                f as u64
            }
            // as `edit::Timeline::fix` rounds it
            None => f0 + ((fps.n(len) + 0.5 + 1e-6).floor() as u64).max(1),
        };
    }
}

/// The highlights `h` of `file` (`dur` s long) cut into a timeline at `fps`, in clip order, each reframed by keys
/// every 0.5 s from `path` (`Framer::path`, one shot per source frame at `src_fps`): perspective views, smooth.
pub fn timeline(
    file: &str,
    dur: f64,
    fps: edit::Rate,
    src_fps: f64,
    h: &[Highlight],
    path: &[Shot],
) -> edit::Result<edit::Timeline> {
    let mut tl = edit::Timeline::new(fps);
    tl.files.push(edit::File {
        path: file.into(),
        dur,
    });
    for x in h {
        let mut c = edit::Clip::new(0, x.a, x.b.min(dur));
        let n = ((c.b - c.a) / 0.5).ceil() as usize;
        for i in 0..=n {
            let t = (c.a + i as f64 * 0.5).min(c.b);
            if i > 0 && t - c.view.last().map_or(f64::MIN, |k: &edit::Key| k.t) < 0.25 {
                continue;
            }
            let Some(s) = path.get(((t * src_fps).round() as usize).min(path.len().max(1) - 1))
            else {
                break;
            };
            let [y, p] = yaw_pitch(s.dir);
            c.view.push(edit::Key {
                t,
                v: edit::View::look(y, p, 0.0, s.fov, edit::Proj::Persp),
                e: edit::Ease::Smooth,
            });
        }
        tl.clips.push(c);
    }
    tl.fix();
    tl.check()?;
    Ok(tl)
}

/// Thumbnail side: six cube faces of `TH`² per sphere.
const TH: u32 = 32;

impl Ai {
    /// A thumbnail of the whole sphere for change detection: the six cube faces at `TH`² (RGB, 0..1, planar).
    pub fn thumb(&mut self, p: &Pano) -> Result<Vec<f32>, String> {
        let n = (3 * TH * TH) as usize;
        if self.th.is_none() {
            let t = In::new(&self.dev, [6, 3, TH as usize, TH as usize])?;
            self.th = Some((Crops::new(&self.wd, &self.g[1], 6, 6 * n), t));
        }
        let (c, t) = self.th.as_ref().unwrap();
        let jobs: Vec<Job> = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ]
        .iter()
        .enumerate()
        .map(|(i, &f)| Job::new(&Cam::new(f, Vec3::Z, Vec2::ONE), [TH; 2], i * n, &RGB, p.w))
        .collect();
        let _ = self.crop(c, p, &jobs, t);
        Ok(read(t.t.clone()))
    }
}
