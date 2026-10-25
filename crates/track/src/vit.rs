//! Single-subject tracking on the sphere with VitTrack (OpenCV's `TrackerVit`, an OSTrack-style one-stream ViT):
//! the template is a 128² crop of 2·√(w·h) around the subject at the pick, a second one refreshed as its apparent
//! size changes; each frame's search a 256² crop of 4·√(w·h) centred on the subject's last direction (sizes in the
//! tangent plane), scored against both. Out of a 16² grid come a confidence (peak after a Hann window, as OpenCV),
//! the box centre offset and its size relative to the search crop. When the confidence drops, the detector checks the
//! box; when the subject is lost, it searches around its last direction and then the whole horizon. A person is
//! found again by appearance (`reid`): its features from the pick and from frames where it was tracked with
//! certainty against those of the people seen next to it; anything else by the pick's template.

use crate::reid::{self, Emb, Gallery};
use crate::{Ai, Cam, Crops, Det, In, Job, Norm, Pano, deg, iou, read, sphere, tan};
use burn::tensor::Tensor;
use glam::{Vec2, Vec3};
use std::time::Instant;

#[allow(clippy::all, unused)]
pub(crate) mod model {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/object_tracking_vittrack_2023sep.rs"
    ));
}

const TPL: u32 = 128;
const SEA: u32 = 256;
/// ImageNet mean and std in BGR order, as OpenCV feeds it.
pub(crate) const NORM: Norm = Norm {
    a: [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
    b: [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    bgr: true,
};
/// Confidence: at or above `HI` the box is taken; between `LO` and `HI` only if the detector confirms it.
const HI: f32 = 0.5;
const LO: f32 = 0.25;
/// Frames without a box before the subject counts as lost; while lost, detection every `EVERY` frames, around the
/// last direction first, the whole horizon after `WIDE` frames.
const MISS: u32 = 8;
const EVERY: u32 = 5;
const WIDE: u32 = 45;
/// Detector score for candidates; candidates the tracker checks per search.
const DMIN: f32 = 0.3;
const FIND: usize = 8;
/// Search crop: at most 150° wide; the subject at least 1° (tangent units).
const SMAX: f32 = 3.732;
const SMIN: f32 = 0.0175;
/// Frames between appearance samples while a person is tracked.
const LOOK: u32 = 10;

/// What to track: centre direction and angular width, height (degrees), and the detector class if known (it
/// narrows re-detection; None takes any class).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    pub dir: Vec3,
    pub size: Vec2,
    pub cls: Option<u32>,
}

impl Subject {
    /// The box `b` = (x0, y0, x1, y1), 0..1 of the image of view `cam` (`Cam::view` of the player's view).
    pub fn boxed(cam: &Cam, b: [f32; 4]) -> Self {
        let (dir, size) = cam.boxed(b.map(|v| 2.0 * v - 1.0));
        Self {
            dir,
            size,
            cls: None,
        }
    }
}

impl From<Det> for Subject {
    fn from(d: Det) -> Self {
        Self {
            dir: d.dir,
            size: d.size,
            cls: Some(d.cls),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// The tracker is confident.
    Tracked,
    /// Low confidence, but the detector sees the subject's class at the box.
    Checked,
    /// No box this frame: `raw` holds the last one.
    Miss,
    /// Missing for `MISS` frames or more; the detector is searching.
    Lost,
    /// Found again by detection and confirmed by the tracker.
    Found,
}

/// One frame of a track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// This frame's box centre (the last one while missing).
    pub raw: Vec3,
    /// Smoothed online: jitter well below the subject's size is damped, larger moves pass at once.
    pub dir: Vec3,
    /// Angular width and height, degrees.
    pub size: Vec2,
    /// Tracker confidence (0 when it found no box).
    pub score: f32,
    pub state: State,
}

pub struct Tracker {
    /// Templates: 0 from the pick (who the subject is: re-detection checks candidates against it alone), 1 the latest
    /// refresh (how the subject looks now); tracking takes the better box of the first `nt`. `ts`: √(w·h) of the
    /// subject (tangent plane) when each was taken.
    tpl: [In; 2],
    nt: usize,
    ts: [f32; 2],
    sea: In,
    c: Crops,
    /// The subject: last box centre, size in the tangent plane, class.
    d: Vec3,
    sz: Vec2,
    cls: Option<u32>,
    /// Crop up vector, carried from frame to frame (continuous over the poles).
    up: Vec3,
    sm: Vec3,
    han: [f32; 256],
    miss: u32,
    /// Persons: who the subject is and who it is not; frames since the last sample.
    gal: Option<Gallery>,
    tick: u32,
}

impl Tracker {
    /// Starts on `s` in the equirect `p` (the template comes from this frame).
    pub fn new(ai: &mut Ai, p: &Pano, s: Subject) -> Result<Self, String> {
        let (t, w) = (TPL as usize, SEA as usize);
        // OpenCV's centred Hann window, outer product of the 1-D one
        let h = |i: usize| 0.5 - 0.5 * (std::f32::consts::TAU / 17.0 * (i + 1) as f32).cos();
        let mut tr = Self {
            tpl: [
                In::new(&ai.dev, [1, 3, t, t])?,
                In::new(&ai.dev, [1, 3, t, t])?,
            ],
            nt: 1,
            ts: [0.0; 2],
            sea: In::new(&ai.dev, [1, 3, w, w])?,
            c: Crops::new(&ai.wd, &ai.g[1], 1, 3 * w * w),
            d: s.dir.normalize(),
            sz: tan(s.size).max(Vec2::splat(SMIN)),
            cls: s.cls,
            up: Vec3::Z,
            sm: s.dir.normalize(),
            han: std::array::from_fn(|i| h(i / 16) * h(i % 16)),
            miss: 0,
            gal: (s.cls == Some(0)).then(Gallery::default),
            tick: 0,
        };
        tr.template(ai, p, 0);
        if tr.gal.is_some() {
            let v = tr.seen(ai, p, tr.d, tr.sz);
            match v.first() {
                // the pick is the subject even next to others: its feature first, then theirs
                Some((a, x)) if *a > 0.5 => tr.look(ai, p, x, &v[1..], true),
                _ => {
                    let e = ai.embed(p, &[(tr.d, deg(tr.sz))]);
                    tr.gal.as_mut().unwrap().add(e[0], true);
                    None
                }
            };
        }
        Ok(tr)
    }

    /// Similarity of feature `e` to the subject and to the people seen next to it (`Gallery::score`; persons only).
    pub fn appearance(&self, e: &Emb) -> Option<(f32, f32)> {
        self.gal.as_ref().map(|g| g.score(e))
    }

    /// Samples the appearance of the subject's box `x` and of the others `o` in view, and returns whether `x` looked
    /// like one of the others and not like the subject before (`Gallery::other`). `x` joins the subject's features
    /// only if it looks like it (`Gallery::same`) and more than like the others (or `sure`): a box that drifted onto
    /// someone else must not teach the tracker their look, also one nobody seen next to the subject looks like more;
    /// one that looks like the others and not like the subject joins the others.
    fn look(
        &mut self,
        ai: &mut Ai,
        p: &Pano,
        x: &Det,
        o: &[(f32, Det)],
        sure: bool,
    ) -> Option<bool> {
        let g = self.gal.as_mut()?;
        // others overlapping the box may be the subject detected twice
        let b: Vec<_> = std::iter::once(x.on())
            .chain(
                o.iter()
                    .filter(|o| o.0 < 0.1)
                    .take(reid::MAX - 1)
                    .map(|o| o.1.on()),
            )
            .collect();
        let e = ai.embed(p, &b);
        let s = if g.me.is_empty() {
            (1.0, -1.0)
        } else {
            g.score(&e[0])
        };
        let d = g.other(s);
        if sure || s.0 >= s.1.max(g.same()) {
            g.add(e[0], true);
        } else if s.0 < s.1.min(g.same()) {
            g.add(e[0], false);
        }
        for e in &e[1..] {
            g.add(*e, false);
        }
        Some(d)
    }

    /// Takes template `i` around the subject's box in `p`; tracking uses it from now on.
    fn template(&mut self, ai: &mut Ai, p: &Pano, i: usize) {
        let cam = self.cam(self.d, 1.0);
        let d = ai.crop(
            &self.c,
            p,
            &[Job::new(&cam, [TPL; 2], 0, &NORM, p.w)],
            &self.tpl[i],
        );
        ai.spent(0, d);
        self.ts[i] = self.sz.element_product().sqrt();
        self.nt = i + 1;
    }

    /// The detections of the subject's class around box `d`, `sz` (tangent plane), with their IoU with it, most
    /// first.
    fn seen(&mut self, ai: &mut Ai, p: &Pano, d: Vec3, sz: Vec2) -> Vec<(f32, Det)> {
        let cam = self.cam(d, 2.0);
        let cls = self.cls;
        let mut v: Vec<(f32, Det)> = ai
            .detect(p, &[cam], DMIN)
            .into_iter()
            .filter(|x| cls.is_none_or(|c| c == x.cls))
            .map(|x| (iou((d, deg(sz)), x.on(), false), x))
            .collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        v
    }

    /// A square crop centred on `f` of `k`·√(w·h) half side, its up vector carried over from the last one and pulled
    /// toward the world's up as far as it is defined (fully at the horizon, not at all at the poles).
    fn cam(&mut self, f: Vec3, k: f32) -> Cam {
        let t = (self.up - self.up.dot(f) * f).normalize_or(Vec3::Z);
        let z = Vec3::Z - f.z * f;
        let l = z.length();
        self.up = if l > 1e-4 {
            t.lerp(z / l, l * l).normalize_or(t)
        } else {
            t
        };
        let s = (k * self.sz.element_product().sqrt()).min(SMAX);
        Cam::new(f, self.up, Vec2::splat(s))
    }

    /// The tracker's best box over the first `n` templates in a search crop around `f` sized for the subject:
    /// (confidence, centre, size in the tangent plane).
    fn search(&mut self, ai: &mut Ai, p: &Pano, f: Vec3, n: usize) -> (f32, Vec3, Vec2) {
        let cam = self.cam(f, 2.0);
        let d = ai.crop(
            &self.c,
            p,
            &[Job::new(&cam, [SEA; 2], 0, &NORM, p.w)],
            &self.sea,
        );
        ai.spent(0, d);
        let t = Instant::now();
        let mut r = (f32::MIN, f, self.sz);
        for tp in &self.tpl[..n] {
            let (c, s, o) = ai.vit.forward(tp.t.clone(), self.sea.t.clone());
            let v = read(Tensor::cat(vec![c, s, o], 1));
            let (i, sc) = (0..256)
                .map(|i| (i, v[i] * self.han[i]))
                .fold((0, f32::MIN), |a, b| if b.1 > a.1 { b } else { a });
            let x = (Vec2::new((i % 16) as f32, (i / 16) as f32)
                + Vec2::new(v[768 + i], v[1024 + i]))
                / 16.0;
            if sc > r.0 {
                r = (
                    sc,
                    cam.ray(2.0 * x - 1.0),
                    Vec2::new(v[256 + i], v[512 + i]) * 2.0 * cam.ext.x,
                );
            }
        }
        ai.spent(1, t.elapsed());
        r
    }

    /// Tracks the subject into the next frame `p`.
    pub fn step(&mut self, ai: &mut Ai, p: &Pano) -> Sample {
        let (mut sc, mut d, mut sz) = self.search(ai, p, self.d, self.nt);
        let mut st = State::Tracked;
        let mut take = sc >= HI;
        if !take && sc >= LO {
            // the detector on the search area: the subject's class at the box confirms it
            take = self.seen(ai, p, d, sz).first().is_some_and(|a| a.0 > 0.3);
            st = State::Checked;
        }
        // after a gap the tracker takes whoever stands where the subject was: a person is back only if they look
        // like it
        if take
            && self.miss >= MISS
            && let Some(g) = &self.gal
        {
            let e = ai.embed(p, &[(d, deg(sz))]);
            take = g.is(g.score(&e[0]));
        }
        // a box several times larger or smaller in one frame is the tracker jumping to something else
        sz = sz.clamp(self.sz * 0.7, self.sz * 1.4);
        // a person's look, now and then while tracked: a box that looks like someone seen next to the subject and
        // not like the subject has drifted onto them, so the subject counts as lost and is searched for at once
        let mut seen = None;
        if take && st == State::Tracked && self.gal.is_some() {
            self.tick += 1;
            if self.tick >= LOOK {
                // tried every `LOOK` frames, sampled only when the subject stands clear of others
                self.tick = 0;
                let v = self.seen(ai, p, d, sz);
                if let [(a, x), o @ ..] = &v[..]
                    && *a > 0.5
                    && o.first().is_none_or(|o| o.0 < 0.1)
                    && self.look(ai, p, x, o, false) == Some(true)
                {
                    take = false;
                    self.miss = self.miss.max(MISS - 1);
                }
                seen = Some(v);
            }
        }
        if !take
            && self.miss + 1 >= MISS
            && (self.miss + 1 - MISS).is_multiple_of(EVERY)
            && let Some(x) = self.find(ai, p)
        {
            (take, st, sc, d, sz) = (true, State::Found, x.score, x.dir, tan(x.size));
        }
        if take {
            self.miss = 0;
            self.d = d;
            self.sz = sz.max(Vec2::splat(SMIN));
            let a = self.sm.angle_between(d) / self.sz.element_product().sqrt().atan().max(1e-4);
            self.sm = self.sm.lerp(d, (0.2 + 0.8 * a).min(1.0)).normalize();
            // a new template once the subject looks √2 larger or smaller (it comes closer, walks away): against the
            // pick's alone the confidence sags toward HI as the view of the subject changes. Only when the tracker is
            // sure, the detector sees the subject's class there and nobody else overlapping (which is which is unsure
            // then), and the pick's template finds it too, so that a refresh never changes who is tracked; around
            // the detector's box, which holds the subject alone where the tracker's grows over people close by.
            let r = self.sz.element_product().sqrt() / self.ts[self.nt - 1];
            if st == State::Tracked
                && !(0.7..1.4).contains(&r)
                && let [(a, x), o @ ..] = &seen.unwrap_or_else(|| self.seen(ai, p, d, self.sz))[..]
                && *a > 0.5
                && o.first().is_none_or(|o| o.0 < 0.1)
            {
                let (keep, x) = ((self.d, self.sz), *x);
                (self.d, self.sz) = (x.dir, tan(x.size).max(Vec2::splat(SMIN)));
                let (s0, d0, z0) = self.search(ai, p, x.dir, 1);
                if s0 >= HI && iou((d0, deg(z0)), x.on(), false) > 0.5 {
                    self.template(ai, p, 1);
                } else {
                    (self.d, self.sz) = keep;
                }
            }
        } else {
            self.miss += 1;
            st = if self.miss >= MISS {
                // how the subject looked before the gap is no guide after it
                self.nt = 1;
                State::Lost
            } else {
                State::Miss
            };
        }
        Sample {
            raw: self.d,
            dir: self.sm,
            size: deg(self.sz),
            score: if take { sc } else { 0.0 },
            state: st,
        }
    }

    /// Detects candidates of the subject's class near its last direction (over `sphere()` once it has been gone
    /// `WIDE` frames) and returns, of the `FIND` nearest within reach, the one the pick's template likes best
    /// (confidence ≥ `HI`, its box on the detection) as the tracker's box and confidence. The best, not the first:
    /// VitTrack's confidence passes `HI` on other people too, so the nearest confirmed one is often someone else.
    fn find(&mut self, ai: &mut Ai, p: &Pano) -> Option<Det> {
        let tiles = if self.miss >= WIDE {
            sphere()
        } else {
            let up = self.cam(self.d, 1.0).u;
            [-60f32, 0.0, 60.0]
                .map(|a| {
                    Cam::new(
                        glam::Quat::from_rotation_z(a.to_radians()) * self.d,
                        up,
                        Vec2::ONE,
                    )
                })
                .into()
        };
        // a person covers about 3 of its heights a second at most (0.1 a frame at 30 fps): one farther from where the
        // subject was lost is someone else, however much the tracker likes it
        let (cls, reach) = (
            self.cls,
            (deg(self.sz).y * (1.0 + 0.1 * self.miss as f32)).to_radians(),
        );
        let mut c: Vec<Det> = ai
            .detect(p, &tiles, DMIN)
            .into_iter()
            .filter(|x| cls.is_none_or(|c| c == x.cls) && x.dir.angle_between(self.d) <= reach)
            .collect();
        c.sort_by(|a, b| {
            a.dir
                .angle_between(self.d)
                .total_cmp(&b.dir.angle_between(self.d))
        });
        c.truncate(FIND);
        let keep = self.sz;
        if let Some(g) = &self.gal {
            // the candidate most like the subject and clearly less like anyone seen next to it; its box from the
            // tracker if that agrees, else the detector's
            let e = ai.embed(p, &c.iter().map(|x| x.on()).collect::<Vec<_>>());
            let (m, x) = c
                .iter()
                .zip(&e)
                .map(|(x, e)| (g.score(e), x))
                .filter(|(s, _)| g.is(*s))
                .max_by(|a, b| a.0.0.total_cmp(&b.0.0))
                .map(|((m, _), x)| (m, *x))?;
            self.sz = tan(x.size);
            let (sc, d, sz) = self.search(ai, p, x.dir, 1);
            self.sz = keep;
            return Some(if sc >= LO && iou((d, deg(sz)), x.on(), false) > 0.3 {
                Det {
                    dir: d,
                    size: deg(sz),
                    score: sc,
                    ..x
                }
            } else {
                Det { score: m, ..x }
            });
        }
        let mut best: Option<Det> = None;
        for x in c {
            self.sz = tan(x.size);
            let (sc, d, sz) = self.search(ai, p, x.dir, 1);
            if sc >= HI
                && best.is_none_or(|b| sc > b.score)
                && iou((d, deg(sz)), x.on(), false) > 0.3
            {
                best = Some(Det {
                    dir: d,
                    size: deg(sz),
                    score: sc,
                    ..x
                });
            }
        }
        self.sz = keep;
        best
    }
}

/// Zero-lag smoothing of a finished track (for keyframes): a Gaussian of `sigma` frames over the box centres,
/// weighted by confidence; missing and lost frames carry no weight and are bridged.
pub fn smooth(t: &[Sample], sigma: f32) -> Vec<Vec3> {
    let r = (3.0 * sigma).ceil() as isize;
    let mut last = t.first().map_or(Vec3::NEG_X, |s| s.raw);
    (0..t.len() as isize)
        .map(|k| {
            let mut a = Vec3::ZERO;
            for j in (k - r).max(0)..(k + r + 1).min(t.len() as isize) {
                let s = &t[j as usize];
                let w = (-((j - k) as f32).powi(2) / (2.0 * sigma * sigma)).exp() * s.score;
                a += w * s.raw;
            }
            last = a.try_normalize().unwrap_or(last);
            last
        })
        .collect()
}
