use super::*;
use crate::view::{angle, preset};
use glam::{DVec3, Quat, Vec3};

const NTSC: Rate = Rate {
    num: 30000,
    den: 1001,
};
const DUR: [f64; 2] = [31.197833, 21.855167];

/// Source frame k's pts on 29.97 footage.
fn sf(k: i64) -> f64 {
    k as f64 * 1001.0 / 30000.0
}

/// The source frame shown at `s`: latest pts ≤ s + 1 µs.
fn pick(s: f64) -> i64 {
    ((s + 1e-6) * 30000.0 / 1001.0).floor() as i64
}

struct Rng(u64);

impl Rng {
    fn u(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f64 {
        (self.u() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
    fn n(&mut self, n: usize) -> usize {
        (self.u() % n as u64) as usize
    }
    fn view(&mut self) -> View {
        let proj = match self.n(4) {
            0 => Proj::Panini(self.r(0.0, 1.0) as f32),
            1 => Proj::Planet(self.r(0.0, 1.0) as f32),
            _ => Proj::Persp,
        };
        View::look(
            self.r(-180.0, 180.0) as f32,
            self.r(-80.0, 80.0) as f32,
            self.r(-20.0, 20.0) as f32,
            self.r(40.0, 160.0) as f32,
            proj,
        )
    }
    fn ease(&mut self) -> Ease {
        [
            Ease::Linear,
            Ease::Smooth,
            Ease::Smooth,
            Ease::In,
            Ease::Out,
            Ease::InOut,
            Ease::Hold,
        ][self.n(7)]
    }
    /// A random clip of file f: moving (constant, ramp, reversed) or frozen, with keys.
    fn clip(&mut self, f: u32) -> Clip {
        let dur = DUR[f as usize];
        let mut c = if self.n(5) == 0 {
            Clip::still(f, self.r(0.0, dur), self.r(0.1, 2.0))
        } else {
            let a = self.r(0.0, dur - 2.0);
            let mut c = Clip::new(f, a, self.r(a + 0.5, (a + 6.0).min(dur)));
            match self.n(3) {
                0 => {}
                1 => c.speed = vec![(0.0, self.r(MIN_SPEED, MAX_SPEED))],
                _ => {
                    let mut t = a - 1.0;
                    c.speed = (0..1 + self.n(4))
                        .map(|_| {
                            t += self.r(0.2, 3.0);
                            (t, self.r(MIN_SPEED, 4.0))
                        })
                        .collect();
                }
            }
            c.rev = self.n(3) == 0;
            c
        };
        let (lo, hi) = if c.hold > 0.0 {
            (0.0, c.hold)
        } else {
            (c.a, c.b)
        };
        let mut t = lo - 0.5;
        for _ in 0..self.n(4) {
            t += self.r(0.1, (hi - lo) / 2.0);
            let (v, e) = (self.view(), self.ease());
            c.view.push(Key { t, v, e });
        }
        if self.n(2) == 0 {
            c.vol = vec![(lo, self.r(0.0, 2.0) as f32), (hi, self.r(0.0, 2.0) as f32)];
        }
        c.mute = self.n(5) == 0;
        c.look = ["", "warm", "teal & orange \"v2\""][self.n(3)].into();
        c.amt = self.r(0.0, 1.0) as f32;
        if self.n(2) == 0 {
            (c.adj.exposure, c.adj.reach) = (self.r(-3.0, 3.0) as f32, self.r(1.0, 12.0) as f32);
        }
        c
    }
}

fn base() -> Project {
    let mut p = Project::new(NTSC).unwrap();
    p.file("/media/a.insv", DUR[0]).unwrap();
    p.file("/media/b.insv", DUR[1]).unwrap();
    p
}

fn random(seed: u64, n: usize) -> Project {
    let (mut r, mut p) = (Rng(seed), base());
    for k in 0..n {
        let f = r.n(2) as u32;
        let c = r.clip(f);
        p.insert(k, c).unwrap();
    }
    if r.n(2) == 0 {
        p.edit_out(|o| {
            o.view = vec![
                Key {
                    t: 0.5,
                    v: r.view(),
                    e: Ease::Smooth,
                },
                Key {
                    t: 3.0,
                    v: r.view(),
                    e: Ease::InOut,
                },
            ];
            o.music = Some(Music {
                file: 1,
                offset: -1.0,
                gain: 0.8,
                duck: true,
            });
        })
        .unwrap();
    }
    p
}

/// (file, source time) of every output frame.
fn shots(tl: &Timeline) -> Vec<(u32, f64)> {
    (0..tl.len())
        .map(|n| tl.frame(n).map(|e| (e.clip.file, e.src)).unwrap())
        .collect()
}

// ---- time mapping ----

#[test]
fn ramp_round_trip_and_monotonic() {
    let mut c = Clip::new(0, 0.5, 9.7);
    c.speed = vec![
        (1.0, 0.5),
        (3.0, 4.0),
        (4.0, 4.0),
        (6.0, 0.25),
        (8.0, 16.0),
        (8.1, 16.0),
        (9.0, 0.1),
    ];
    let d = c.dur();
    let (mut pu, mut ps) = (-1.0, -1.0);
    for i in 0..=200_000 {
        let s = c.a + (c.b - c.a) * i as f64 / 200_000.0;
        let u = c.out(s);
        assert!((c.src(u) - s).abs() < 1e-9, "src(out({s})) = {}", c.src(u));
        assert!(u > pu, "out not increasing at {s}");
        pu = u;
        let u = d * i as f64 / 200_000.0;
        let s = c.src(u);
        assert!((c.out(s) - u).abs() < 1e-9);
        assert!(s > ps, "src not increasing at {u}");
        ps = s;
    }
    // the closed form against Simpson's rule on 1 / speed, piece by piece (speed has kinks at keys)
    let mut edges = vec![c.a];
    edges.extend(c.speed.iter().map(|k| k.0).filter(|&x| x > c.a && x < c.b));
    edges.push(c.b);
    let mut num = 0.0;
    for w in edges.windows(2) {
        let (lo, n) = (w[0], 20_000);
        let h = (w[1] - lo) / n as f64;
        let f = |x: f64| 1.0 / c.rate(x);
        num += h / 3.0
            * (0..=n)
                .map(|i| {
                    f(lo + i as f64 * h)
                        * if i == 0 || i == n {
                            1.0
                        } else if i % 2 == 1 {
                            4.0
                        } else {
                            2.0
                        }
                })
                .sum::<f64>();
    }
    assert!((num - d).abs() < 1e-9 * d, "closed form {d}, Simpson {num}");
    // speed is continuous and piecewise linear
    assert!((c.rate(2.0) - 2.25).abs() < 1e-12 && c.rate(0.0) == 0.5 && c.rate(20.0) == 0.1);
}

#[test]
fn constant_speed_durations() {
    for v in [MIN_SPEED, 0.5, 1.0, 2.0, 3.0, MAX_SPEED] {
        let mut c = Clip::new(0, sf(10), sf(100));
        c.speed = vec![(5.0, v)];
        assert!((c.dur() - (c.b - c.a) / v).abs() < 1e-12);
    }
}

/// Every output frame at 29.97 shows the expected source frame of 29.97 footage, forwards and backwards, at
/// constant speeds, across cuts.
#[test]
fn frame_accurate_ntsc() {
    let mut p = base();
    // (speed as p / q, first source frame, source frames)
    let cases: [(i64, i64, i64, i64); 7] = [
        (1, 1, 10, 90),
        (2, 1, 10, 90),
        (1, 2, 333, 47),
        (1, 10, 7, 13),
        (16, 1, 40, 160),
        (3, 1, 5, 299),
        (1, 1, 890, 35),
    ];
    for rev in [false, true] {
        for &(sp, sq, a, n) in &cases {
            let mut c = Clip::new(0, sf(a), sf(a + n));
            c.speed = vec![(0.0, sp as f64 / sq as f64)];
            c.rev = rev;
            let k = p.tl().clips.len();
            p.insert(k, c).unwrap();
        }
    }
    let tl = p.tl();
    // forwards frame j shows a + ⌊j·v⌋; backwards it shows b − ⌈(j + 1)·v⌉ (never before a)
    let mut want = vec![];
    for rev in [false, true] {
        for &(sp, sq, a, n) in &cases {
            let len = ((n as f64 * sq as f64 / sp as f64).round() as i64).max(1);
            want.extend((0..len).map(|j| {
                if rev {
                    (a + n - ((j + 1) * sp + sq - 1) / sq).max(a)
                } else {
                    a + j * sp / sq
                }
            }));
        }
    }
    let got: Vec<i64> = (0..tl.len())
        .map(|n| pick(tl.frame(n).unwrap().src))
        .collect();
    assert_eq!(got, want);
    // continuous time lands in the same frames
    for n in (0..tl.len()).step_by(7) {
        let t = NTSC.t(n);
        assert!(
            (tl.at(t).unwrap().src - tl.frame(n).unwrap().src).abs() < 1e-12,
            "frame {n}"
        );
        assert_eq!(tl.frame_at(t + 0.9 * NTSC.t(1)), Some(n));
    }
    // one hour of 1× cuts: no drift
    let mut p = base();
    p.file("/media/long.insv", 4000.0).unwrap();
    for i in 0..40 {
        p.insert(
            i,
            Clip::new(2, sf(i as i64 * 2700), sf((i as i64 + 1) * 2700)),
        )
        .unwrap();
    }
    let tl = p.tl();
    assert_eq!(tl.len(), 108_000);
    assert!((0..tl.len()).all(|n| pick(tl.frame(n).unwrap().src) == n as i64));
}

#[test]
fn ramp_frames_monotonic() {
    let mut p = base();
    let mut c = Clip::new(0, 1.0, 20.0);
    c.speed = vec![
        (2.0, 0.25),
        (5.0, 8.0),
        (9.0, 1.0),
        (12.0, 1.0),
        (15.0, 0.1),
        (16.0, 3.0),
    ];
    p.insert(0, c.clone()).unwrap();
    c.rev = true;
    p.insert(1, c).unwrap();
    let tl = p.tl();
    let s: Vec<f64> = (0..tl.len()).map(|n| tl.frame(n).unwrap().src).collect();
    let r = tl.frames(1);
    assert!(s[..r.start as usize].windows(2).all(|w| w[1] > w[0]));
    assert!(s[r.start as usize..].windows(2).all(|w| w[1] < w[0]));
    // the reversed clip shows the forward clip's source positions backwards, one output frame later in source
    let n = r.start as usize;
    let dt = NTSC.t(1);
    for j in 0..n {
        let c = &tl.clips[0];
        let back = c.src((c.dur() - (j as f64 + 1.0) * dt).max(0.0));
        assert!((s[n + j] - back.max(c.a)).abs() < 1e-9);
    }
    assert_eq!(s[0], 1.0);
    assert!(s[s.len() - 1] < 1.0 + 0.2 * NTSC.t(1));
}

// ---- operations ----

#[test]
fn split_keeps_every_frame() {
    for seed in 1..40 {
        let mut p = random(seed * 7919, 5);
        let before = shots(p.tl());
        let views: Vec<View> = (0..p.tl().len())
            .map(|n| p.tl().frame(n).unwrap().view)
            .collect();
        let mut r = Rng(seed);
        for _ in 0..6 {
            let n = r.n(p.tl().len() as usize) as u64;
            let k = p.tl().find(n).unwrap();
            let cut = n == p.tl().frames(k).start;
            match p.split(NTSC.t(n) + r.r(0.0, 0.99) * NTSC.t(1)) {
                Ok(k) => assert!(!cut && p.tl().frames(k).start == n, "seed {seed} frame {n}"),
                Err(_) => assert!(cut, "seed {seed} frame {n}"),
            }
        }
        let after = shots(p.tl());
        assert_eq!(before.len(), after.len(), "seed {seed}");
        for (n, (a, b)) in before.iter().zip(&after).enumerate() {
            assert!(
                a.0 == b.0 && (a.1 - b.1).abs() < 1e-9,
                "seed {seed} frame {n}: {a:?} {b:?}"
            );
            let v = p.tl().frame(n as u64).unwrap().view;
            assert!(
                angle(v.q, views[n].q) < 1e-5 && (v.fov - views[n].fov).abs() < 1e-3,
                "seed {seed} frame {n}"
            );
        }
    }
}

#[test]
fn trim_delete_move_dup_speed() {
    let mut p = base();
    p.insert(0, Clip::new(0, 1.0, 3.0)).unwrap();
    p.insert(1, Clip::new(1, 0.0, 1.0)).unwrap();
    p.insert(2, Clip::new(0, 10.0, 11.0)).unwrap();
    assert_eq!(p.tl().len(), 60 + 30 + 30);
    p.trim(0, 2.0, 3.0).unwrap();
    assert_eq!((p.tl().clips[0].a, p.tl().len()), (2.0, 90));
    assert!(
        p.trim(0, 3.0, 2.0).is_err()
            && p.trim(0, -1.0, 2.0).is_err()
            && p.trim(0, 1.0, 40.0).is_err()
    );
    assert!(p.trim(0, f64::NAN, 2.0).is_err() && p.trim(9, 1.0, 2.0).is_err());
    p.mv(0, 2).unwrap();
    assert_eq!(
        p.tl().clips.iter().map(|c| c.a).collect::<Vec<_>>(),
        [0.0, 10.0, 2.0]
    );
    p.mv(2, 0).unwrap();
    assert_eq!(
        p.tl().clips.iter().map(|c| c.a).collect::<Vec<_>>(),
        [2.0, 0.0, 10.0]
    );
    assert!(p.mv(0, 3).is_err());
    p.dup(1).unwrap();
    assert_eq!(
        p.tl().clips.iter().map(|c| c.file).collect::<Vec<_>>(),
        [0, 1, 1, 0]
    );
    p.delete(2).unwrap();
    assert_eq!(p.tl().clips.len(), 3);
    assert!(p.delete(3).is_err());
    p.speed(0, vec![(0.0, 2.0)]).unwrap();
    assert_eq!(p.tl().frames(0), 0..15);
    assert!(p.speed(0, vec![(0.0, 17.0)]).is_err() && p.speed(0, vec![(0.0, 0.05)]).is_err());
    assert!(
        p.speed(0, vec![(2.0, 1.0), (1.0, 2.0)]).is_err()
            && p.speed(0, vec![(1.0, 1.0), (1.0, 2.0)]).is_err()
    );
    assert!(
        p.insert(5, Clip::new(0, 0.0, 1.0)).is_err()
            && p.insert(0, Clip::new(7, 0.0, 1.0)).is_err()
    );
    // speed keys closer than GAP: the ramp slope would overflow
    assert!(
        p.speed(0, vec![(0.0, 0.1), (1e-310, 16.0)]).is_err()
            && p.speed(0, vec![(0.0, 0.1), (0.5 * GAP, 16.0)]).is_err()
            && p.speed(0, vec![(-2e9, 1.0), (1.0, 2.0)]).is_err()
    );
    assert_eq!(p.tl().clips[0].speed, [(0.0, 2.0)]);
    // NaN neither loops nor panics
    let mut c = Clip::new(0, 0.0, 9.0);
    c.speed = vec![(1.0, 0.5), (3.0, 4.0)];
    assert!(c.src(f64::NAN).is_nan() && p.tl().content(0, f64::NAN).is_nan());
    // a file can not shrink under its clips, nor under clips an undo or redo brings back
    assert!(p.file("/media/a.insv", 2.5).is_err());
    assert_eq!(p.file("/media/a.insv", 40.0), Ok(0));
    p.insert(0, Clip::new(0, 30.0, 39.0)).unwrap();
    p.delete(0).unwrap();
    assert!(p.file("/media/a.insv", 35.0).is_err());
    p.undo();
    p.undo();
    assert!(p.file("/media/a.insv", 35.0).is_err());
    assert_eq!(p.file("/media/a.insv", 39.0), Ok(0));
    assert!(p.redo() && p.tl().check().is_ok());
}

/// Clips of a whole number of source frames at 2× often last a half output frame more than a whole number: every
/// split of them must keep the rounded-up frame (13 frames at 2×: 6.5 → 7 = 3 + 4, not 3 + 3).
#[test]
fn half_frame_splits() {
    for rev in [false, true] {
        for (a, n) in [(20, 13), (7, 1), (101, 299), (0, 45)] {
            let mut c = Clip::new(0, sf(a), sf(a + n));
            c.speed = vec![(0.0, 2.0)];
            c.rev = rev;
            for clip in [c, Clip::still(0, sf(a), NTSC.t(n as u64) / 2.0)] {
                let mut p = base();
                p.insert(0, clip).unwrap();
                let (len, before) = (p.tl().len(), shots(p.tl()));
                assert_eq!(len, (n as u64).div_ceil(2));
                for j in 1..len {
                    let mut q = base();
                    q.insert(0, p.tl().clips[0].clone()).unwrap();
                    q.split(NTSC.t(j)).unwrap();
                    assert_eq!(
                        q.tl().frames(1),
                        j..len,
                        "{n} frames, split at {j}, rev {rev}"
                    );
                    assert!(
                        shots(q.tl())
                            .iter()
                            .zip(&before)
                            .all(|(x, y)| (x.1 - y.1).abs() < 1e-9)
                    );
                }
                // split at every frame, last first
                for j in (1..len).rev() {
                    p.split(NTSC.t(j)).unwrap();
                }
                assert_eq!((p.tl().clips.len() as u64, p.tl().len()), (len, len));
                assert!(
                    shots(p.tl())
                        .iter()
                        .zip(&before)
                        .all(|(x, y)| (x.1 - y.1).abs() < 1e-9)
                );
            }
        }
    }
}

#[test]
fn freeze_frame() {
    let mut p = base();
    p.insert(0, Clip::new(0, sf(30), sf(120))).unwrap();
    let v = View {
        fov: 60.0,
        ..View::default()
    };
    p.edit(0, |c| {
        c.view = vec![
            Key {
                t: sf(30),
                v,
                e: Ease::Linear,
            },
            Key {
                t: sf(120),
                v: View::default(),
                e: Ease::Linear,
            },
        ]
    })
    .unwrap();
    let before = shots(p.tl());
    let k = p.freeze(NTSC.t(40) + 0.01, 2.0).unwrap();
    let tl = p.tl();
    assert_eq!((k, tl.clips.len()), (1, 3));
    assert_eq!(tl.len(), 90 + 60);
    let after = shots(tl);
    let same = |a: &[(u32, f64)], b: &[(u32, f64)]| {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() < 1e-9)
    };
    assert!(same(&after[..40], &before[..40]));
    assert!(after[40..100].iter().all(|s| pick(s.1) == 70));
    assert!(same(&after[100..], &before[40..]));
    let e = tl.frame(50).unwrap();
    assert_eq!((e.speed, e.gain), (0.0, 0.0));
    // the freeze starts at the view the clip had there
    let fov = |n| tl.frame(n).unwrap().view.fov;
    assert!((fov(40) - fov(39)).abs() < 0.5 && fov(99) == fov(40));
    assert!(
        p.freeze(NTSC.t(50), 1.0).is_err(),
        "no freeze inside a freeze"
    );
    assert!(p.freeze(NTSC.t(10), 0.0).is_err() && p.freeze(NTSC.t(10), f64::INFINITY).is_err());
    assert!(p.freeze(1e9, 1.0).is_err());
    // at a cut: no split
    let n = p.tl().clips.len();
    p.freeze(0.0, 1.0).unwrap();
    assert_eq!(p.tl().clips.len(), n + 1);
    // splitting a freeze keeps its frames
    let before = shots(p.tl());
    p.split(NTSC.t(10)).unwrap();
    assert!(same(&shots(p.tl()), &before));
}

#[test]
fn segments() {
    for seed in 1..30 {
        let p = random(seed * 104729, 6);
        let tl = p.tl();
        let segs = tl.segs();
        let mut next = 0;
        for s in &segs {
            assert_eq!(s.frames.start, next);
            assert!(s.frames.end > s.frames.start);
            next = s.frames.end;
            let c = &tl.clips[s.k];
            let (f, l) = (
                tl.frame(s.frames.start).unwrap(),
                tl.frame(s.frames.end - 1).unwrap(),
            );
            assert_eq!(
                (f.k, l.k, s.file, s.rev),
                (s.k, s.k, c.file, c.rev && c.hold == 0.0)
            );
            assert_eq!(s.src, [f.src, l.src]);
            assert_eq!(s.speed, [f.speed, l.speed]);
            // no speed key strictly inside a run but where the speed is the same on both sides
            let (lo, hi) = (s.src[0].min(s.src[1]), s.src[0].max(s.src[1]));
            assert!(
                c.speed.iter().all(|k| k.0 <= lo + 1e-6
                    || k.0 >= hi - 1e-6
                    || c.hold > 0.0
                    || c.rate(k.0 - 1e-3) == k.1 && c.rate(k.0 + 1e-3) == k.1),
                "seed {seed} {s:?}"
            );
        }
        assert_eq!(next, tl.len());
    }
    // a ramp with a flat piece: the flat run reports one speed
    let mut p = base();
    let mut c = Clip::new(0, 0.0, 10.0);
    c.speed = vec![(2.0, 1.0), (4.0, 3.0), (7.0, 3.0), (8.0, 0.5)];
    p.insert(0, c).unwrap();
    let segs = p.tl().segs();
    assert_eq!(segs.len(), 5);
    assert_eq!(segs[2].speed, [3.0, 3.0]);
    assert!(segs[1].speed[0] < segs[1].speed[1]);
    // keys with the same speed on both sides split nothing, forwards or reversed; the ramp still splits
    for rev in [false, true] {
        p.edit(0, |c| {
            c.speed = vec![(2.0, 2.0), (5.0, 2.0), (7.0, 1.0), (8.0, 1.0)];
            c.rev = rev;
        })
        .unwrap();
        let segs = p.tl().segs();
        let sp: Vec<_> = segs.iter().map(|g| g.speed).collect();
        let (f, l) = if rev { (1.0, 2.0) } else { (2.0, 1.0) };
        assert_eq!((segs.len(), sp[0], sp[2]), (3, [f; 2], [l; 2]), "{rev}");
        assert!(sp[1][0] != sp[1][1]);
    }
    // the same speed but a bend: (1, 1) → (3, 3) → (5, 1) splits at 3
    p.edit(0, |c| {
        c.speed = vec![(1.0, 1.0), (3.0, 3.0), (5.0, 1.0)];
        c.rev = false;
    })
    .unwrap();
    assert_eq!(p.tl().segs().len(), 4);
}

#[test]
fn audio_gains() {
    let mut p = base();
    p.insert(0, Clip::new(0, 0.0, 4.0)).unwrap();
    p.insert(1, Clip::new(0, 4.0, 6.0)).unwrap();
    p.edit(0, |c| c.vol = vec![(1.0, 1.0), (3.0, 0.0)]).unwrap();
    p.edit(1, |c| c.mute = true).unwrap();
    p.edit_out(|o| {
        o.music = Some(Music {
            file: 1,
            offset: -0.5,
            gain: 0.5,
            duck: true,
        })
    })
    .unwrap();
    let tl = p.tl();
    let at = |t: f64| tl.at(t).unwrap();
    assert_eq!((at(0.2).gain, at(0.2).music), (1.0, None));
    assert!((at(2.0).gain - 0.5).abs() < 1e-6);
    assert_eq!(at(1.0).music, Some((0.5, 0.5 * DUCK)));
    assert_eq!(at(3.5).gain, 0.0);
    assert_eq!(at(3.5).music.unwrap().1, 0.5);
    assert_eq!(at(5.0).gain, 0.0);
    assert!(tl.at(7.0).is_none() && tl.at(-0.1).is_none() && tl.at(f64::NAN).is_none());
}

// ---- keyframes ----

/// Body-frame rotation rate between two orientations h s apart (rad/s).
fn rate(a: Quat, b: Quat, h: f64) -> DVec3 {
    rot(a, b) / h
}

/// Rotation vector from a to b, body frame.
fn rot(a: Quat, b: Quat) -> DVec3 {
    let r = a.as_dquat().inverse() * b.as_dquat();
    let r = if r.w < 0.0 { -r } else { r };
    let v = DVec3::new(r.x, r.y, r.z);
    let s = v.length();
    if s < 1e-12 {
        2.0 * v
    } else {
        v * (2.0 * s.atan2(r.w) / s)
    }
}

#[test]
fn interpolation_continuity() {
    let h = 2e-3;
    let mut r = Rng(99);
    let mut kinks = 0;
    for _ in 0..200 {
        let mut t = 0.0;
        let ks: Vec<Key> = (0..5)
            .map(|_| {
                t += r.r(0.5, 3.0);
                let e = [
                    Ease::Smooth,
                    Ease::Smooth,
                    Ease::Linear,
                    Ease::In,
                    Ease::Out,
                ][r.n(5)];
                Key {
                    t,
                    v: View {
                        proj: Proj::Persp,
                        ..r.view()
                    },
                    e,
                }
            })
            .collect();
        let at = |t: f64| view::at(&ks, t, false).unwrap();
        for i in 1..4 {
            let tk = ks[i].t;
            // C0 everywhere, the key exactly at its time
            assert!(angle(at(tk).q, ks[i].v.q) < 1e-6 && (at(tk).fov - ks[i].v.fov).abs() < 1e-4);
            assert!(angle(at(tk - 1e-7).q, at(tk).q) < 1e-5);
            // C1 where both neighbouring segments use the auto tangent at this key; one-sided second-order
            // differences (f'(0) ≈ (4 f(h) − f(2h)) / 2h for f(0) = 0) on each side
            let q = at(tk).q;
            let wl = (rot(q, at(tk - 2.0 * h).q) - 4.0 * rot(q, at(tk - h).q)) / (2.0 * h);
            let wr = (4.0 * rot(q, at(tk + h).q) - rot(q, at(tk + 2.0 * h).q)) / (2.0 * h);
            let f = |t: f64| (at(t).fov - at(tk).fov) as f64;
            let (fl, fr) = (
                (f(tk - 2.0 * h) - 4.0 * f(tk - h)) / (2.0 * h),
                (4.0 * f(tk + h) - f(tk + 2.0 * h)) / (2.0 * h),
            );
            let auto = matches!(ks[i - 1].e, Ease::Smooth | Ease::In)
                && matches!(ks[i].e, Ease::Smooth | Ease::Out);
            let stop = matches!(ks[i - 1].e, Ease::Out) && matches!(ks[i].e, Ease::In);
            if auto || stop {
                assert!(
                    (wl - wr).length() < 0.05 * wl.length().max(1.0),
                    "rate {wl} vs {wr} at key {i}: eases {:?}, angles {:?}, dts {:?}",
                    ks.iter().map(|k| k.e).collect::<Vec<_>>(),
                    ks.windows(2)
                        .map(|w| angle(w[0].v.q, w[1].v.q).to_degrees())
                        .collect::<Vec<_>>(),
                    ks.windows(2).map(|w| w[1].t - w[0].t).collect::<Vec<_>>()
                );
                assert!(
                    (fl - fr).abs() < 0.05 * fl.abs().max(10.0),
                    "fov rate {fl} vs {fr}"
                );
            }
            if stop {
                assert!(wl.length() < 0.05 && wr.length() < 0.05);
            }
            if ks[i - 1].e == Ease::Linear && ks[i].e == Ease::Linear && (wl - wr).length() > 0.1 {
                kinks += 1;
            }
        }
    }
    // the measure sees velocity jumps where linear keys turn
    assert!(kinks > 0);
}

/// Largest angular step between n + 1 samples of the segment w over its path length / n (1 at constant speed).
fn swing(ks: &[Key], w: &[Key], n: usize, rev: bool) -> f64 {
    let q: Vec<Quat> = (0..=n)
        .map(|i| {
            view::at(ks, w[0].t + (w[1].t - w[0].t) * i as f64 / n as f64, rev)
                .unwrap()
                .q
        })
        .collect();
    let st: Vec<f64> = q.windows(2).map(|q| angle(q[0], q[1])).collect();
    let path: f64 = st.iter().sum();
    st.iter().fold(0f64, |a, &b| a.max(b)) * n as f64 / path.max(1e-12)
}

/// Inside segments: no jumps (a rotation slerp in squad flipped its way round mid-segment) and no fast swings near
/// squad's singularity (tangents too long for its control points), on random orientations, all moving eases and
/// both playing directions.
#[test]
fn interpolation_dense() {
    let mut r = Rng(31);
    let mut worst = 0f64;
    for _ in 0..200 {
        let mut t = 0.0;
        let ks: Vec<Key> = (0..5)
            .map(|_| {
                t += r.r(0.01, 3.0);
                let mut c = || r.r(-1.0, 1.0) as f32;
                let q = Quat::from_xyzw(c(), c(), c(), c()).normalize();
                let e = [Ease::Linear, Ease::Smooth, Ease::In, Ease::Out, Ease::InOut][r.n(5)];
                let v = View {
                    q,
                    ..View::default()
                };
                Key { t, v, e }
            })
            .collect();
        for rev in [false, true] {
            for w in ks.windows(2) {
                worst = worst.max(swing(&ks, w, 500, rev));
            }
        }
    }
    assert!(worst < 4.0, "{worst}");
    // found by hill-climbing towards the singularity with tangents up to 3× the slower chord: a 2.2 rad swing
    // within 1/20000 of the middle segment
    let k = |t, y, p, r, e| Key {
        t,
        v: View::look(y, p, r, 90.0, Proj::Persp),
        e,
    };
    let ks = [
        k(0.9583, 136.62221, 24.918308, 158.3864, Ease::Linear),
        k(1.2071, -59.741543, 3.0002813, 51.11634, Ease::Smooth),
        k(3.0236, 117.635254, 23.65274, 8.074716, Ease::Smooth),
        k(3.1311, 133.09901, 21.788382, -179.69339, Ease::Out),
    ];
    for rev in [false, true] {
        let s = swing(&ks, &ks[1..3], 20_000, rev);
        assert!(s < 4.0, "{s}");
    }
}

#[test]
fn ease_shapes() {
    let a = View::look(0.0, 0.0, 0.0, 100.0, Proj::Persp);
    let b = View::look(90.0, 30.0, 0.0, 50.0, Proj::Persp);
    let two = |e| {
        [
            Key { t: 1.0, v: a, e },
            Key {
                t: 3.0,
                v: b,
                e: Ease::Smooth,
            },
        ]
    };
    // linear = slerp at constant speed
    let ks = two(Ease::Linear);
    for i in 0..=20 {
        let u = i as f32 / 20.0;
        let v = view::at(&ks, 1.0 + 2.0 * u as f64, false).unwrap();
        assert!(angle(v.q, a.q.slerp(b.q, u)) < 1e-5);
        assert!((v.fov - (100.0 - 50.0 * u)).abs() < 1e-3);
    }
    // two smooth keys: also linear (one-sided tangents are the chord)
    let (l, s) = (two(Ease::Linear), two(Ease::Smooth));
    assert!((0..20).all(|i| angle(
        view::at(&l, 1.0 + 0.1 * i as f64, false).unwrap().q,
        view::at(&s, 1.0 + 0.1 * i as f64, false).unwrap().q
    ) < 1e-5));
    // in: starts at rest; out: arrives at rest; in-out: both, symmetric
    let at = |ks: &[Key], t| view::at(ks, t, false).unwrap();
    let h = 1e-3;
    let ks = two(Ease::In);
    assert!(rate(at(&ks, 1.0).q, at(&ks, 1.0 + h).q, h).length() < 0.01);
    assert!(rate(at(&ks, 3.0 - h).q, at(&ks, 3.0).q, h).length() > 0.5);
    let ks = two(Ease::Out);
    assert!(rate(at(&ks, 3.0 - h).q, at(&ks, 3.0).q, h).length() < 0.01);
    let ks = two(Ease::InOut);
    assert!(angle(at(&ks, 2.0).q, a.q.slerp(b.q, 0.5)) < 1e-5);
    // hold: a step at the next key
    let ks = two(Ease::Hold);
    assert_eq!(at(&ks, 2.999), a);
    assert_eq!(at(&ks, 3.0), b);
    assert_eq!((at(&ks, 0.0), at(&ks, 9.0)), (a, b));
    assert!(view::at(&[], 1.0, false).is_none());
}

#[test]
fn no_drift_no_overshoot() {
    let a = View::look(10.0, 0.0, 0.0, 100.0, Proj::Persp);
    let b = View::look(80.0, 20.0, 5.0, 60.0, Proj::Persp);
    let ks = [
        Key {
            t: 0.0,
            v: a,
            e: Ease::Smooth,
        },
        Key {
            t: 2.0,
            v: a,
            e: Ease::Smooth,
        },
        Key {
            t: 4.0,
            v: b,
            e: Ease::Smooth,
        },
        Key {
            t: 6.0,
            v: b,
            e: Ease::Smooth,
        },
        Key {
            t: 7.0,
            v: a,
            e: Ease::Smooth,
        },
    ];
    for i in 0..=700 {
        let t = i as f64 / 100.0;
        let v = view::at(&ks, t, false).unwrap();
        if t <= 2.0 {
            assert!(
                angle(v.q, a.q) < 1e-5 && (v.fov - 100.0).abs() < 1e-4,
                "drift at {t}"
            );
        }
        if (4.0..=6.0).contains(&t) {
            assert!(angle(v.q, b.q) < 1e-5, "drift at {t}");
        }
        assert!(
            v.fov >= 60.0 - 1e-4 && v.fov <= 100.0 + 1e-4,
            "overshoot at {t}: {}",
            v.fov
        );
    }
}

#[test]
fn projections_blend() {
    let k = |t, proj, fov| Key {
        t,
        v: View {
            proj,
            fov,
            ..View::default()
        },
        e: Ease::Linear,
    };
    let ks = [
        k(0.0, Proj::Panini(1.0), 170.0),
        k(2.0, Proj::Planet(1.0), 280.0),
        k(3.0, Proj::Equirect, 360.0),
    ];
    let at = |t| view::at(&ks, t, false).unwrap().proj;
    let x = |p| match p {
        Proj::Panini(s) => -s,
        Proj::Planet(d) => d,
        _ => 0.0,
    };
    assert!(matches!(at(0.5), Proj::Panini(s) if (s - 0.5).abs() < 1e-6));
    assert_eq!(at(1.0), Proj::Persp);
    assert!(matches!(at(1.5), Proj::Planet(d) if (d - 0.5).abs() < 1e-6));
    assert_eq!((at(2.4), at(2.6)), (Proj::Planet(1.0), Proj::Equirect));
    // continuous along the axis
    for i in 0..200 {
        let t = i as f64 / 100.0;
        assert!((x(at(t + 0.01)) - x(at(t))).abs() < 0.0101);
    }
}

#[test]
fn no_gimbal_lock() {
    // tilting down through the nadir: yaw and roll jump by 180° there, the motion does not
    let a = View::look(0.0, -60.0, 0.0, 90.0, Proj::Persp);
    let b = View {
        q: a.q * Quat::from_rotation_x(60f32.to_radians()),
        ..a
    };
    let [y, p, r] = b.ypr();
    assert!(
        (y.abs() - 180.0).abs() < 1e-3 && (p + 60.0).abs() < 1e-3 && (r.abs() - 180.0).abs() < 1e-3,
        "{:?}",
        b.ypr()
    );
    let ks = [
        Key {
            t: 0.0,
            v: a,
            e: Ease::Linear,
        },
        Key {
            t: 1.0,
            v: b,
            e: Ease::Linear,
        },
    ];
    let total = angle(a.q, b.q);
    assert!((total - 60f64.to_radians()).abs() < 1e-5);
    let mut p = view::at(&ks, 0.0, false).unwrap();
    for i in 1..=100 {
        let v = view::at(&ks, i as f64 / 100.0, false).unwrap();
        assert!((angle(p.q, v.q) - total / 100.0).abs() < 1e-5);
        p = v;
    }
    let mid = view::at(&ks, 0.5, false).unwrap();
    assert!(
        (mid.ypr()[1] + 90.0).abs() < 0.01,
        "looks straight down halfway: {:?}",
        mid.ypr()
    );
}

#[test]
fn ypr_matches_stitch() {
    let v = View::look(30.0, 20.0, -10.0, 90.0, Proj::Persp);
    let [y, p, r] = v.ypr();
    assert!((y - 30.0).abs() < 1e-4 && (p - 20.0).abs() < 1e-4 && (r + 10.0).abs() < 1e-4);
    // stitch: forward = Ry(yaw)·Rx(−pitch)·Rz(roll)·z; yaw turns right (+x), pitch looks up (+y)
    let f = View::look(30.0, 0.0, 0.0, 90.0, Proj::Persp).q * Vec3::Z;
    assert!(f.x > 0.4 && f.y.abs() < 1e-6);
    let f = View::look(0.0, 20.0, 0.0, 90.0, Proj::Persp).q * Vec3::Z;
    assert!(f.y > 0.3);
}

/// About one axis every rotation commutes, and squad on Hermite control points is the scalar Hermite cubic of the
/// angle: the fov channel carrying the same angles (same tangent rules) predicts it exactly, also for 170° steps and
/// eased ends, where the inner points of squad lie more than 180° apart.
#[test]
fn squad_collinear_is_hermite() {
    let mut r = Rng(5);
    let ax = Vec3::new(0.3, 1.0, -0.2).normalize();
    let mut worst = 0f64;
    for _ in 0..300 {
        let (mut t, mut th) = (0.0, 0.0f64);
        let ks: Vec<Key> = (0..5)
            .map(|_| {
                t += r.r(0.2, 3.0);
                th += r.r(-170.0, 170.0);
                let q = Quat::from_axis_angle(ax, th.to_radians() as f32);
                Key {
                    t,
                    v: View {
                        q,
                        fov: (1000.0 + th) as f32,
                        proj: Proj::Persp,
                    },
                    e: r.ease(),
                }
            })
            .collect();
        for rev in [false, true] {
            for i in 0..=500 {
                let x = ks[0].t - 0.1 + (t - ks[0].t + 0.2) * i as f64 / 500.0;
                let v = view::at(&ks, x, rev).unwrap();
                let want = Quat::from_axis_angle(ax, (v.fov as f64 - 1000.0).to_radians() as f32);
                worst = worst.max(angle(v.q, want));
            }
        }
    }
    assert!(worst < 1e-4, "{worst} rad");
}

/// A key's ease shapes the motion that follows it on screen, also on a reversed clip, whose keys sit on source
/// time running backwards.
#[test]
fn reversed_clip_eases() {
    let mut p = base();
    let mut c = Clip::new(0, 2.0, 8.0);
    c.rev = true;
    p.insert(0, c).unwrap();
    let (a, b) = (
        View::look(0.0, 0.0, 0.0, 90.0, Proj::Persp),
        View::look(90.0, 0.0, 0.0, 90.0, Proj::Persp),
    );
    let h = 1e-3;
    for (e, rest0, rest1) in [
        (Ease::In, true, false),
        (Ease::Out, false, true),
        (Ease::InOut, true, true),
        (Ease::Linear, false, false),
    ] {
        p.edit(0, |c| c.view.clear()).unwrap();
        p.key(On::Clip(0), 1.0, a, e).unwrap();
        p.key(On::Clip(0), 4.0, b, Ease::Smooth).unwrap();
        let tl = p.tl();
        assert_eq!(
            tl.clips[0].view[0].v, b,
            "the later key on screen sits earlier in source time"
        );
        let w = |t: f64| angle(tl.at(t).unwrap().view.q, tl.at(t + h).unwrap().view.q) / h;
        assert!(
            angle(tl.at(1.0).unwrap().view.q, a.q) < 1e-5
                && angle(tl.at(4.0).unwrap().view.q, b.q) < 1e-5
        );
        assert_eq!(
            (w(1.0) < 0.02, w(4.0 - h) < 0.02),
            (rest0, rest1),
            "{e:?}: {} {}",
            w(1.0),
            w(4.0 - h)
        );
    }
    // hold: a until the next key on screen
    p.edit(0, |c| c.view.clear()).unwrap();
    p.key(On::Clip(0), 1.0, a, Ease::Hold).unwrap();
    p.key(On::Clip(0), 4.0, b, Ease::Smooth).unwrap();
    let tl = p.tl();
    assert!((0..50).all(|i| tl.at(1.0 + 0.06 * i as f64).unwrap().view == a));
    assert_eq!(
        (
            tl.at(0.5).unwrap().view,
            tl.at(4.0).unwrap().view,
            tl.at(5.0).unwrap().view
        ),
        (a, b, b)
    );
}

// ---- presets ----

#[test]
fn presets() {
    let v = View::look(20.0, -10.0, 0.0, 90.0, Proj::Persp);
    let h = 1e-3;
    // orbit: one turn about the vertical at constant speed, the horizon stays
    let ks = preset::orbit(v, 8.0, 1.0).unwrap();
    assert_eq!(ks.len(), 5);
    let at = |ks: &[Key], t| view::at(ks, t, false).unwrap();
    let w0 = angle(at(&ks, 0.0).q, at(&ks, h).q);
    for i in 0..800 {
        let t = i as f64 / 100.0;
        let (a, b) = (at(&ks, t), at(&ks, t + h));
        assert!((angle(a.q, b.q) - w0).abs() < 2e-6, "speed at {t}");
        assert!(
            ((a.q * Vec3::Z).y - (v.q * Vec3::Z).y).abs() < 1e-5,
            "elevation at {t}"
        );
        assert!(
            ((a.q * Vec3::X).y - (v.q * Vec3::X).y).abs() < 1e-5,
            "roll at {t}"
        );
    }
    assert!((w0 / h * 8.0 - std::f64::consts::TAU).abs() < 1e-2);
    assert!(angle(at(&ks, 8.0).q, v.q) < 1e-5);
    assert_eq!(preset::orbit(v, 4.0, -2.5).unwrap().len(), 11);
    assert!(preset::orbit(v, 0.0, 1.0).is_err() && preset::orbit(v, 1.0, 0.0).is_err());
    // whip: starts and stops at rest, widest halfway
    let b = View::look(160.0, 0.0, 0.0, 90.0, Proj::Persp);
    let ks = preset::whip(v, b, 0.6, 30.0).unwrap();
    let hh = 1e-4;
    assert!(angle(at(&ks, 0.0).q, v.q) < 1e-6 && angle(at(&ks, 0.6).q, b.q) < 1e-6);
    assert!(angle(at(&ks, 0.0).q, at(&ks, hh).q) / hh < 0.05);
    assert!(angle(at(&ks, 0.6 - hh).q, at(&ks, 0.6).q) / hh < 0.05);
    assert!(angle(at(&ks, 0.3 - h).q, at(&ks, 0.3).q) / h > 3.0);
    assert!(
        (at(&ks, 0.3).fov - 120.0).abs() < 1e-3
            && at(&ks, 0.2).fov < 120.0
            && at(&ks, 0.4).fov < 120.0
    );
    // one speed peak, halfway: rising before, falling after (f32 noise aside)
    let w: Vec<f64> = (0..600)
        .map(|i| i as f64 * 1e-3)
        .map(|t| angle(at(&ks, t).q, at(&ks, t + 1e-3).q) / 1e-3)
        .collect();
    let (top, mean) = (w.iter().cloned().fold(0.0, f64::max), angle(v.q, b.q) / 0.6);
    for i in 1..600 {
        let d = w[i] - w[i - 1];
        assert!(
            if i < 300 {
                d > -1e-4 * top
            } else {
                d < 1e-4 * top
            },
            "{i}: {} {}",
            w[i - 1],
            w[i]
        );
    }
    assert!((top / mean - 1.57).abs() < 0.02, "{}", top / mean);
    // zoom and push-in: monotonic field of view
    for ks in [
        preset::zoom(v, 40.0, 2.0).unwrap(),
        preset::push(v, 2.0, 0.8).unwrap(),
    ] {
        let f: Vec<f32> = (0..=200).map(|i| at(&ks, i as f64 / 100.0).fov).collect();
        assert!(f.windows(2).all(|w| w[1] <= w[0]));
        assert_eq!((f[0], f[200]), (90.0, ks[1].v.fov));
    }
    let ks = preset::zoom(v, 130.0, 2.0).unwrap();
    let f: Vec<f32> = (0..=200).map(|i| at(&ks, i as f64 / 100.0).fov).collect();
    assert!(
        f.windows(2).all(|w| w[1] >= w[0]) && (f[0], f[200]) == (90.0, 130.0),
        "zoom out"
    );
    let ks = preset::push(v, 2.0, 0.8).unwrap();
    assert!((at(&ks, h).fov - 90.0).abs() < 1e-4, "starts gently");
    assert!(preset::zoom(v, 0.0, 1.0).is_err() && preset::push(v, 1.0, 1.5).is_err());
    // tiny planet → perspective: looks down at first, the projection slides from stereographic to rectilinear
    let ks = preset::planet(v, 3.0, false).unwrap();
    let p0 = at(&ks, 0.0);
    assert_eq!((p0.proj, p0.fov), (Proj::Planet(1.0), view::PLANET_FOV));
    assert!(((p0.q * Vec3::Z).y + 1.0).abs() < 1e-5);
    // the top of the frame faces v's heading
    let r = 20f32.to_radians();
    assert!((p0.q * Vec3::Y - Vec3::new(r.sin(), 0.0, r.cos())).length() < 1e-5);
    let d: Vec<f32> = (0..=300)
        .map(|i| match at(&ks, i as f64 / 100.0).proj {
            Proj::Planet(d) => d,
            Proj::Persp => 0.0,
            p => panic!("{p:?}"),
        })
        .collect();
    assert!(d.windows(2).all(|w| w[1] <= w[0] && w[0] - w[1] < 0.02));
    assert_eq!(at(&ks, 3.0), v);
    let ks = preset::planet(v, 3.0, true).unwrap();
    assert!(
        angle(at(&ks, 0.0).q, v.q) < 1e-6
            && at(&ks, 0.0).proj == Proj::Persp
            && at(&ks, 3.0).proj == Proj::Planet(1.0)
    );
}

// ---- keys on tracks ----

#[test]
fn keys_at_playhead() {
    let mut p = base();
    p.insert(0, Clip::new(0, 2.0, 6.0)).unwrap();
    p.insert(1, Clip::new(1, 1.0, 5.0)).unwrap();
    let (a, b) = (
        View::look(0.0, 0.0, 0.0, 80.0, Proj::Persp),
        View::look(90.0, 0.0, 0.0, 120.0, Proj::Persp),
    );
    // output track
    assert_eq!(p.key(On::Out, 1.0, a, Ease::Smooth), Ok(0));
    assert_eq!(p.key(On::Out, 5.0, b, Ease::Smooth), Ok(1));
    assert_eq!(
        p.key(On::Out, 5.0 + 0.4 * NTSC.t(1), b, Ease::Linear),
        Ok(1),
        "within half a frame: replaced"
    );
    assert_eq!(p.tl().out.view.len(), 2);
    assert_eq!(p.tl().out.view[1].e, Ease::Linear);
    assert_eq!(p.tl().at(5.0).unwrap().view, b);
    // put replaces the keys closer than GAP
    let mut ks = p.tl().out.view.clone();
    let (x, y) = (1.0 + 0.5 * GAP, 1.0 + 1.5 * GAP);
    assert_eq!(
        view::put(
            &mut ks,
            Key {
                t: x,
                v: b,
                e: Ease::Hold
            }
        ),
        0
    );
    assert_eq!(
        view::put(
            &mut ks,
            Key {
                t: y,
                v: b,
                e: Ease::Hold
            }
        ),
        0
    );
    assert_eq!(ks.iter().map(|k| k.t).collect::<Vec<_>>(), [y, 5.0]);
    assert_eq!(
        view::put(
            &mut ks,
            Key {
                t: y + 2.0 * GAP,
                v: a,
                e: Ease::Hold
            }
        ),
        1
    );
    assert_eq!(ks.len(), 3);
    // clip track (source time) wins over the output track on its clip
    assert!(
        p.key(On::Clip(0), 5.0, a, Ease::Smooth).is_err(),
        "playhead not on clip 0"
    );
    let i = p.key(On::Clip(1), 6.0, b, Ease::Smooth).unwrap();
    assert_eq!(
        (i, p.tl().clips[1].view[0].t),
        (0, p.tl().at(6.0).unwrap().src)
    );
    assert_eq!(p.tl().at(4.5).unwrap().view, b);
    assert_ne!(p.tl().at(3.0).unwrap().view, b);
    p.key(On::Clip(1), 7.0, a, Ease::Smooth).unwrap();
    assert_eq!(p.tl().near(On::Clip(1), 7.0 + 0.01), Some(1));
    // move: by output time, replacing a key it lands on
    let j = p.mv_key(On::Clip(1), 1, 4.5).unwrap();
    assert!(j == 0 && angle(p.tl().at(4.5).unwrap().view.q, a.q) < 1e-6);
    let j = p.mv_key(On::Clip(1), 0, 6.0).unwrap();
    assert_eq!((j, p.tl().clips[1].view.len()), (0, 1));
    assert_eq!(p.tl().clips[1].view[0].v, a);
    assert!(p.mv_key(On::Clip(1), 0, 1.0).is_err() && p.mv_key(On::Clip(1), 5, 6.0).is_err());
    p.unkey(On::Clip(1), 0).unwrap();
    assert!(p.unkey(On::Clip(1), 5).is_err() && p.unkey(On::Clip(7), 0).is_err());
    // keys on source time: the view stays on the picture through a speed change and a trim
    p.key(On::Clip(1), 5.0, a, Ease::Smooth).unwrap();
    p.key(On::Clip(1), 7.5, b, Ease::Smooth).unwrap();
    let src = |p: &Project, t| p.tl().at(t).unwrap().src;
    let s = src(&p, 6.5);
    let v = p.tl().at(6.5).unwrap().view;
    p.speed(1, vec![(0.0, 2.0)]).unwrap();
    p.trim(1, 1.5, 5.0).unwrap();
    let t = p.tl().from_track(On::Clip(1), s).unwrap();
    assert!((src(&p, t) - s).abs() < 1e-9 && angle(p.tl().at(t).unwrap().view.q, v.q) < 1e-5);
}

#[test]
fn presets_on_tracks() {
    let mut p = base();
    p.insert(0, Clip::new(0, 0.0, 10.0)).unwrap();
    let mut c = Clip::new(0, 10.0, 20.0);
    c.rev = true;
    p.insert(1, c).unwrap();
    let v = View::default();
    let b = View::look(120.0, 0.0, 0.0, 90.0, Proj::Persp);
    let ks = preset::whip(v, b, 2.0, 20.0).unwrap();
    // output track: replaces the keys inside its range
    p.key(On::Out, 2.0, b, Ease::Smooth).unwrap();
    p.key(On::Out, 8.0, b, Ease::Smooth).unwrap();
    p.preset(On::Out, 1.0, &ks).unwrap();
    assert_eq!(
        p.tl().out.view.iter().map(|k| k.t).collect::<Vec<_>>(),
        [1.0, 1.9, 2.1, 3.0, 8.0]
    );
    // a reversed clip: mirrored into source order, same motion in output time
    p.preset(On::Clip(1), 12.0, &ks).unwrap();
    assert!(
        p.preset(On::Clip(1), 19.0, &ks).is_err(),
        "runs past the clip"
    );
    let tl = p.tl();
    let cv = &tl.clips[1].view;
    assert!(cv.windows(2).all(|w| w[0].t < w[1].t));
    assert_eq!(
        cv.iter().map(|k| k.e).collect::<Vec<_>>(),
        [Ease::Smooth, Ease::Out, Ease::Smooth, Ease::In]
    );
    let h = NTSC.t(1);
    for (t, want) in [(12.0, v), (14.0, b)] {
        let n = tl.frame_at(t + 1e-9).unwrap();
        assert!(angle(tl.frame(n).unwrap().view.q, want.q) < 1e-3, "at {t}");
    }
    // at rest at both ends in output time, fast in the middle
    let w = |t: f64| angle(tl.at(t).unwrap().view.q, tl.at(t + h).unwrap().view.q) / h;
    assert!(
        w(12.0) < 0.15 && w(14.0 - h) < 0.15 && w(13.0) > 0.8,
        "{} {} {}",
        w(12.0),
        w(14.0 - h),
        w(13.0)
    );
}

// ---- undo / redo ----

#[test]
fn undo_redo_random() {
    for seed in 1..25 {
        let mut r = Rng(seed * 31337);
        let mut p = Project::with(random(seed, 3).tl().clone()).unwrap();
        let mut states = vec![p.tl().clone()];
        for _ in 0..60 {
            let len = p.tl().clips.len().max(1);
            let t = r.r(0.0, p.tl().dur());
            let n_undo = states.len();
            let op = r.n(13);
            let res = match op {
                0 => {
                    let (k, f) = (r.n(len + 1), r.n(2) as u32);
                    p.insert(k, r.clip(f))
                }
                1 => {
                    let k = r.n(len);
                    let a = r.r(0.0, 10.0);
                    p.trim(k, a, a + r.r(-1.0, 5.0))
                }
                2 => p.split(t).map(|_| ()),
                3 => p.delete(r.n(len)),
                4 => p.dup(r.n(len)),
                5 => p.mv(r.n(len), r.n(len)),
                6 => p.speed(r.n(len), vec![(5.0, r.r(0.05, 17.0))]),
                7 => p.freeze(t, r.r(0.0, 2.0)).map(|_| ()),
                8 => p.edit(r.n(len), |c| c.rev = !c.rev),
                9 => p.key(On::Out, t, r.view(), r.ease()).map(|_| ()),
                10 => {
                    let on = p
                        .tl()
                        .find(p.tl().frame_at(t).unwrap_or(0))
                        .map_or(On::Out, On::Clip);
                    p.preset(on, t, &preset::orbit(r.view(), r.r(0.1, 3.0), 1.0).unwrap())
                }
                11 => p.edit_out(|o| {
                    o.fps = [NTSC, Rate { num: 25, den: 1 }, Rate { num: 60, den: 1 }][r.n(3)]
                }),
                _ => {
                    // a drag: many steps, one undo step
                    p.begin();
                    let k = r.n(len);
                    let mut res = Ok(());
                    for i in 0..20 {
                        res = res.and(p.edit(k, |c| c.amt = i as f32 / 20.0));
                    }
                    p.end();
                    res
                }
            };
            if res.is_ok() {
                assert_eq!(p.tl().check(), Ok(()));
                // edits that change nothing leave no step
                if p.tl() != &states[n_undo - 1] {
                    states.push(p.tl().clone());
                }
                assert_eq!(p.history().0, states.len() - 1, "seed {seed} op {op}");
            } else {
                assert_eq!(
                    p.tl(),
                    &states[n_undo - 1],
                    "a failed edit changed the timeline"
                );
            }
        }
        let last = states.len() - 1;
        for i in (0..last).rev() {
            assert!(p.undo());
            assert_eq!(p.tl(), &states[i], "seed {seed}: undo to {i}");
        }
        assert!(!p.undo());
        for (i, s) in states.iter().enumerate().skip(1) {
            assert!(p.redo());
            assert_eq!(p.tl(), s, "seed {seed}: redo to {i}");
        }
        assert!(!p.redo());
        // undo some, then edit: the redo branch is gone
        p.undo();
        p.undo();
        assert!(p.can_redo());
        p.edit_out(|o| {
            o.music = Some(Music {
                file: 0,
                offset: 0.125,
                gain: 0.25,
                duck: false,
            })
        })
        .unwrap();
        assert!(!p.can_redo());
    }
}

#[test]
fn drags_coalesce() {
    let mut p = base();
    p.insert(0, Clip::new(0, 1.0, 5.0)).unwrap();
    p.key(On::Clip(0), 1.0, View::default(), Ease::Smooth)
        .unwrap();
    let start = p.tl().clone();
    // trim handle drag
    p.begin();
    for i in 0..100 {
        p.trim(0, 1.0 + i as f64 * 0.01, 5.0).unwrap();
    }
    assert!(p.trim(0, 9.0, 5.0).is_err());
    p.end();
    let dragged = p.tl().clone();
    assert!((dragged.clips[0].a - 1.99).abs() < 1e-12);
    // keyframe drag, then a second drag of the same thing: two steps
    for d in 0..3 {
        p.begin();
        for i in 0..50 {
            p.mv_key(On::Clip(0), 0, 0.1 + d as f64 * 0.5 + i as f64 * 0.02)
                .unwrap();
        }
        // the third drag ends where it started: no step
        if d == 2 {
            p.mv_key(On::Clip(0), 0, 1.58).unwrap();
        }
        p.end();
    }
    // insert, key, trim drag, two key drags
    assert_eq!(p.history(), (5, 0));
    // settings drag (music gain)
    p.begin();
    for i in 0..30 {
        p.edit_out(|o| {
            o.music = Some(Music {
                file: 1,
                offset: 0.0,
                gain: i as f32 / 30.0,
                duck: false,
            })
        })
        .unwrap();
    }
    p.end();
    let fin = p.tl().clone();
    assert!(p.undo() && p.tl().out.music.is_none());
    assert!(p.undo() && p.undo());
    assert_eq!(p.tl(), &dragged);
    assert!(p.undo());
    assert_eq!(p.tl(), &start);
    for _ in 0..4 {
        assert!(p.redo());
    }
    assert_eq!(p.tl(), &fin);
    // without a drag every edit is its own step
    p.trim(0, 2.0, 5.0).unwrap();
    p.trim(0, 2.1, 5.0).unwrap();
    p.undo();
    assert_eq!(p.tl().clips[0].a, 2.0);
    // an edit that changes nothing is no step
    p.trim(0, 2.0, 5.0).unwrap();
    p.undo();
    assert!((p.tl().clips[0].a - 1.99).abs() < 1e-12);
}

#[test]
fn noop_drags_keep_redo() {
    let mut p = base();
    p.insert(0, Clip::new(0, 1.0, 5.0)).unwrap();
    p.key(On::Clip(0), 1.0, View::default(), Ease::Smooth)
        .unwrap();
    p.insert(1, Clip::new(1, 0.0, 20.0)).unwrap();
    let full = p.tl().clone();
    assert!(p.undo());
    let less = p.tl().clone();
    let music = |g: f32| {
        move |o: &mut Out| {
            o.music = (g > 0.0).then_some(Music {
                file: 1,
                offset: 0.0,
                gain: g,
                duck: false,
            })
        }
    };
    // a trim drag, a settings drag and a key drag that end where they started, and an empty one: the redo step stays
    for d in 0..4 {
        p.begin();
        match d {
            0 => {
                p.trim(0, 1.5, 5.0).unwrap();
                p.trim(0, 1.0, 5.0).unwrap();
            }
            1 => {
                p.edit_out(music(0.5)).unwrap();
                p.edit_out(music(0.0)).unwrap();
            }
            2 => {
                p.mv_key(On::Clip(0), 0, 2.0).unwrap();
                p.mv_key(On::Clip(0), 0, 1.0).unwrap();
            }
            _ => {}
        }
        p.end();
        assert_eq!((p.tl(), p.history()), (&less, (2, 1)), "drag {d}");
    }
    // while the drag is held, the redo step's clip keeps its file from shrinking
    p.begin();
    p.trim(0, 1.5, 5.0).unwrap();
    assert!(p.file("/media/b.insv", 10.0).is_err());
    p.trim(0, 1.0, 5.0).unwrap();
    p.end();
    assert!(p.redo());
    assert_eq!(p.tl(), &full);
    // a drag that ends elsewhere drops it
    assert!(p.undo());
    p.begin();
    p.trim(0, 1.5, 5.0).unwrap();
    p.end();
    assert_eq!(p.history(), (3, 0));
    // undone in the middle of a drag: the drag's step is the redo step
    p.begin();
    p.trim(0, 1.25, 5.0).unwrap();
    assert!(p.undo());
    p.end();
    assert_eq!(p.history(), (3, 1));
    assert!(p.redo());
    assert_eq!(p.tl().clips[0].a, 1.25);
}

#[test]
fn key_drags() {
    let mut p = base();
    p.insert(0, Clip::new(0, 0.0, 10.0)).unwrap();
    let (a, b, c) = (
        View::look(0.0, 0.0, 0.0, 80.0, Proj::Persp),
        View::look(90.0, 0.0, 0.0, 90.0, Proj::Persp),
        View::look(-90.0, 0.0, 0.0, 100.0, Proj::Persp),
    );
    // (output time, view) of every key
    let keys = |p: &Project, on| {
        let tl = p.tl();
        (tl.track(on).unwrap().iter())
            .map(|k| ((tl.from_track(on, k.t).unwrap() * 1e6).round() / 1e6, k.v))
            .collect::<Vec<_>>()
    };
    for on in [On::Out, On::Clip(0)] {
        for (t, v) in [(1.0, a), (2.0, b), (4.0, c)] {
            p.key(on, t, v, Ease::Smooth).unwrap();
        }
        let (start, h) = (p.tl().clone(), p.history().0);
        // drag a across b: b is replaced while a is on it and back once a moved on
        p.begin();
        let mut i = 0;
        for s in 0..=40 {
            let t = 1.0 + s as f64 * 0.05;
            i = p.mv_key(on, i, t).unwrap();
            let n = keys(&p, on).len();
            assert_eq!(
                n,
                if (t - 2.0).abs() < 0.01 { 2 } else { 3 },
                "{on:?} at {t}"
            );
            // the drag leaves the clip (or the time is garbage) and comes back
            if s == 20 {
                assert!(p.mv_key(on, i, f64::NAN).is_err());
                assert!(on == On::Out || p.mv_key(on, i, 99.0).is_err());
            }
        }
        p.end();
        assert_eq!(keys(&p, on), [(2.0, b), (3.0, a), (4.0, c)], "{on:?}");
        assert_eq!(p.history().0, h + 1);
        let one = p.tl().clone();
        // a drag that ends on a key replaces it, also after passing it; another key's move in the same drag starts
        // from the track as it is
        p.begin();
        let i = p.mv_key(on, 1, 4.5).unwrap();
        let i = p.mv_key(on, i, 4.0).unwrap();
        assert_eq!(keys(&p, on), [(2.0, b), (4.0, a)]);
        p.mv_key(on, 0, 0.5).unwrap();
        p.end();
        assert_eq!(keys(&p, on), [(0.5, b), (4.0, a)]);
        assert_eq!(p.history().0, h + 2);
        assert!(p.undo());
        assert_eq!(p.tl(), &one);
        assert!(p.undo());
        assert_eq!(p.tl(), &start);
        assert!(p.redo() && p.redo());
        assert_eq!(keys(&p, on), [(0.5, b), (4.0, a)]);
        assert_eq!(i, 1);
        // an edit in between ends the run: the next move starts from the track as it is then
        p.begin();
        let i = p.mv_key(on, 0, 4.0).unwrap();
        p.key(on, 6.0, c, Ease::Hold).unwrap();
        p.mv_key(on, i, 5.0).unwrap();
        p.end();
        assert_eq!(keys(&p, on), [(5.0, b), (6.0, c)]);
        assert_eq!(p.history().0, h + 3);
    }
    // random drags end as one move from where they started would
    let mut r = Rng(4242);
    for _ in 0..300 {
        let mut p = base();
        p.insert(0, Clip::new(0, 0.0, 10.0)).unwrap();
        let on = [On::Out, On::Clip(0)][r.n(2)];
        for _ in 0..1 + r.n(8) {
            // some within half a frame of each other
            let t = (r.n(20) as f64 * 0.5 + r.r(0.0, 0.02)).min(9.9);
            p.key(on, t, r.view(), r.ease()).unwrap();
        }
        let n = p.tl().track(on).unwrap().len();
        let (i, end) = (r.n(n), r.r(0.0, 9.9));
        let mut q = Project::with(p.tl().clone()).unwrap();
        let (want, h) = (q.mv_key(on, i, end).unwrap(), p.history().0);
        p.begin();
        let mut j = i;
        for _ in 0..r.n(60) {
            j = p.mv_key(on, j, r.r(0.0, 9.9)).unwrap();
        }
        assert_eq!(p.mv_key(on, j, end), Ok(want));
        p.end();
        assert_eq!(p.tl(), q.tl());
        assert!(p.history().0 <= h + 1);
    }
}

// ---- project files ----

/// Frames, continuous time, segments and key-track mapping all give finite values.
fn finite(tl: &Timeline) {
    let fin = |e: Eval| {
        let v = e.view;
        [
            e.src,
            e.speed,
            v.q.length() as f64,
            v.fov as f64,
            e.gain as f64,
        ]
        .iter()
        .chain(e.music.map(|m| m.0).iter())
        .all(|x| x.is_finite())
    };
    let (len, fps) = (tl.len(), tl.out.fps);
    for n in (0..len).step_by(len as usize / 200 + 1) {
        assert!(fin(tl.frame(n).unwrap()), "frame {n}");
        assert!(fin(tl.at(fps.t(n) + 0.3 * fps.t(1)).unwrap()));
        let k = tl.find(n).unwrap();
        let x = tl.to_track(On::Clip(k), fps.t(n)).unwrap();
        assert!(x.is_finite() && tl.from_track(On::Clip(k), x).is_some_and(f64::is_finite));
    }
    let mut end = 0;
    for g in tl.segs() {
        assert!(g.src.iter().chain(&g.speed).all(|x| x.is_finite()));
        end = g.frames.end;
    }
    assert_eq!(end, len);
}

/// What passes check() evaluates to finite values: random timelines with fields set to extreme values, neighbouring
/// keys extremely close or far apart.
#[test]
fn check_implies_finite() {
    let base: Vec<Timeline> = (1..40).map(|s| random(s * 17, 3).tl().clone()).collect();
    let x = [
        0.0,
        5e-324,
        1e-310,
        1e-300,
        1e-9,
        1e-6,
        0.1,
        1.0,
        16.0,
        3600.0,
        1e6,
        1e9,
        1e300,
        1e308,
        f64::MAX,
    ];
    let g = [
        5e-324, 1e-310, 1e-300, 1e-12, 5e-7, 1e-6, 2e-6, 1e-3, 1.0, 1e9, 1e300, 1e308,
    ];
    let mut r = Rng(11);
    let mut oks = 0;
    for _ in 0..30_000 {
        let mut tl = base[r.n(base.len())].clone();
        for _ in 0..1 + r.n(3) {
            let v = x[r.n(x.len())] * [1.0, -1.0][r.n(2)];
            // a second key closely after the first, or mirrored (keys spanning more than f64::MAX)
            let w = [v + g[r.n(g.len())], -v][r.n(2)];
            let (sp, q) = ([MIN_SPEED, 1.0, MAX_SPEED][r.n(3)], r.view());
            let key = |t, v| Key {
                t,
                v,
                e: Ease::Smooth,
            };
            let k = r.n(tl.clips.len());
            let c = &mut tl.clips[k];
            match r.n(9) {
                0 => c.a = v,
                1 => c.b = v,
                2 => c.hold = v,
                3 => {
                    // a ramp only plays from its first key on: subnormal gaps exist only near 0
                    c.speed = vec![(v, MAX_SPEED + MIN_SPEED - sp), (w, sp)];
                    c.a = c.a.min(v.abs());
                }
                4 => c.vol = vec![(v, 0.0), (w, 16.0)],
                5 => c.view = vec![key(v, View::default()), key(w, q)],
                6 => tl.out.view = vec![key(v, q), key(w, View::default())],
                7 => tl.files[c.file as usize].dur = v,
                _ => {
                    tl.out.fps = [
                        Rate { num: 1, den: 1 },
                        Rate { num: 1000, den: 1 },
                        Rate {
                            num: u32::MAX,
                            den: u32::MAX / 1000,
                        },
                        Rate {
                            num: u32::MAX / 1000,
                            den: u32::MAX,
                        },
                    ][r.n(4)]
                }
            }
        }
        if tl.check().is_ok() {
            oks += 1;
            tl.fix();
            finite(&tl);
        }
    }
    assert!(oks > 2000, "{oks}");
}

#[test]
fn save_load_round_trip() {
    for seed in 1..30 {
        let mut p = random(seed * 65537, 6);
        p.file("dir with \"quotes\"\\ and\ttabs/ü 😀 \u{1}.insv", 3.0)
            .unwrap();
        let s = p.save();
        let q = Project::load(&s).unwrap_or_else(|e| panic!("seed {seed}: {e}\n{s}"));
        assert_eq!(q.tl(), p.tl(), "seed {seed}");
        assert_eq!(q.save(), s);
    }
    let mut p = base();
    p.insert(0, Clip::new(0, 0.1, 0.30000000000000004)).unwrap();
    p.edit(0, |c| {
        c.speed = vec![(1e-300, 0.1), (5e-5, 16.0)];
        c.view = vec![Key {
            t: -1e-7,
            v: View {
                proj: Proj::Equirect,
                fov: 360.0,
                ..View::default()
            },
            e: Ease::Hold,
        }];
    })
    .unwrap();
    assert_eq!(Project::load(&p.save()).unwrap().tl(), p.tl());
}

/// Per-clip adjustments: a file holds the fields that differ from fx's defaults and reads them back; a clip without
/// any (files from before) has the defaults; split parts and a freeze frame keep the clip's; malformed ones are
/// refused, unknown fields in them ignored.
#[test]
fn adjust() {
    let mut p = base();
    p.insert(0, Clip::new(0, 1.0, 4.0)).unwrap();
    p.edit(0, |c| (c.adj.exposure, c.adj.reach) = (0.5, 6.0))
        .unwrap();
    let s = p.save();
    assert!(
        s.contains("\"adjust\": {\"exposure\": 0.5, \"reach\": 6.0}"),
        "{s}"
    );
    assert_eq!(Project::load(&s).unwrap().tl(), p.tl());
    p.split(1.0).unwrap();
    p.freeze(0.5, 1.0).unwrap();
    let a = p.tl().clips[0].adj;
    assert!(
        p.tl().clips.len() == 4 && p.tl().clips.iter().all(|c| c.adj == a) && a.exposure == 0.5
    );
    let old = "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"a\", \"dur\": 5}], \"clips\": [{\"file\": 0, \
               \"a\": 1, \"b\": 2}], \"view\": []}";
    assert_eq!(
        Project::load(old).unwrap().tl().clips[0].adj,
        fx::Params::default()
    );
    for (a, ok) in [
        ("{}", true),
        ("{\"tint\": -1, \"later\": [1]}", true),
        ("null", true),
        ("[]", false),
        ("{\"tint\": \"x\"}", false),
        // finite as f64, not as f32
        ("{\"tint\": 1e39}", false),
    ] {
        let s = old.replace("\"b\": 2}", &format!("\"b\": 2, \"adjust\": {a}}}"));
        let r = Project::load(&s);
        assert_eq!(r.is_ok(), ok, "{a}: {:?}", r.err());
    }
}

#[test]
fn load_rejects_garbage() {
    let good = random(4242, 5).save();
    // whatever loads checks, round-trips, evaluates to finite values and takes edits
    let check = |s: &str| {
        if let Ok(mut p) = Project::load(s) {
            let tl = p.tl();
            assert_eq!(tl.check(), Ok(()));
            assert_eq!(Project::load(&p.save()).unwrap().tl(), tl);
            finite(tl);
            let t = tl.dur() / 2.0;
            _ = p.key(On::Out, t, View::default(), Ease::Smooth);
            _ = p.split(t);
            _ = p.freeze(t, 1.0);
            assert_eq!(p.tl().check(), Ok(()));
        }
    };
    // every truncation
    for (i, _) in good.trim_end().char_indices() {
        assert!(Project::load(&good[..i]).is_err());
    }
    // byte mutations (ASCII for ASCII keeps it UTF-8)
    let mut r = Rng(7);
    let pool = b"0123456789-+.eE{}[]\":, tfnrul\\x";
    let mut oks = 0;
    for _ in 0..20_000 {
        let mut b = good.clone().into_bytes();
        for _ in 0..1 + r.n(3) {
            let i = r.n(b.len());
            if b[i].is_ascii() {
                match r.n(3) {
                    0 => b[i] = pool[r.n(pool.len())],
                    1 => _ = b.remove(i),
                    _ => b.insert(i, pool[r.n(pool.len())]),
                }
            }
        }
        if let Ok(s) = String::from_utf8(b) {
            oks += Project::load(&s).is_ok() as u32;
            check(&s);
        }
    }
    assert!(oks > 100, "some mutations stay valid ({oks})");
    // random bytes
    for _ in 0..5000 {
        let b: Vec<u8> = (0..r.n(200)).map(|_| r.u() as u8).collect();
        check(&String::from_utf8_lossy(&b));
    }
    let deep = "[".repeat(100_000);
    for s in [
        "",
        "null",
        "[]",
        "{}",
        &deep,
        "{\"x5edit\": 2, \"fps\": [30, 1], \"files\": [], \"clips\": [], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 0], \"files\": [], \"clips\": [], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"a\", \"dur\": 1e999}], \"clips\": [], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"a\", \"dur\": 5}], \"clips\": [{\"file\": 0, \"a\": 3, \"b\": 2}], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"a\", \"dur\": 5}], \"clips\": [{\"file\": 1, \"a\": 1, \"b\": 2}], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"a\", \"dur\": 5}], \"clips\": [{\"file\": 0, \"a\": 1, \"b\": 2, \"speed\": [[0, 20]]}], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"a\", \"dur\": 5}], \"clips\": [{\"file\": 0, \"a\": 1, \"b\": 2, \"hold\": 1e9}], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [], \"clips\": [], \"view\": [{\"t\": 0, \"q\": [0, 0, 0, 2], \"fov\": 90, \"proj\": \"persp\", \"ease\": \"smooth\"}]}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [], \"clips\": [], \"view\": [], \"music\": {\"file\": 0, \"offset\": 0, \"gain\": 1, \"duck\": true}}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"\\ud800\", \"dur\": 5}], \"clips\": [], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [], \"clips\": [], \"view\": []} x",
        "{\"x5edit\": 1.5}",
        "{\"x5edit\": NaN}",
        // keys too close (an infinite ramp slope, view tangent) or too far apart (an infinite key span)
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"a\", \"dur\": 5}], \"clips\": [{\"file\": 0, \"a\": 0, \"b\": 2, \"speed\": [[0, 0.1], [1e-310, 16]]}], \"view\": []}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [], \"clips\": [], \"view\": [{\"t\": 0, \"q\": [0, 0, 0, 1], \"fov\": 90, \"proj\": \"persp\", \"ease\": \"smooth\"}, {\"t\": 1e-310, \"q\": [0, 0.7071, 0, 0.7071], \"fov\": 90, \"proj\": \"persp\", \"ease\": \"smooth\"}]}",
        "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [], \"clips\": [], \"view\": [{\"t\": -1e308, \"q\": [0, 0, 0, 1], \"fov\": 90, \"proj\": \"persp\", \"ease\": \"smooth\"}, {\"t\": 1e308, \"q\": [0, 0.7071, 0, 0.7071], \"fov\": 90, \"proj\": \"persp\", \"ease\": \"smooth\"}]}",
    ] {
        assert!(Project::load(s).is_err(), "{s:.80}");
    }
    // numbers swapped for extreme ones
    let mut spans = vec![];
    let (b, mut i, mut q) = (good.as_bytes(), 0, false);
    while i < b.len() {
        match b[i] {
            b'\\' if q => i += 1,
            b'"' => q = !q,
            b'-' | b'0'..=b'9' if !q => {
                let j = i;
                while matches!(
                    b.get(i),
                    Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                ) {
                    i += 1;
                }
                spans.push(j..i);
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    let pool = [
        "1e-310",
        "5e-324",
        "-0",
        "0",
        "1e308",
        "-1e308",
        "1e300",
        "-1e300",
        "1e-300",
        "1e-7",
        "9.99e-7",
        "1e-6",
        "0.1",
        "0.5",
        "-1",
        "2",
        "16",
        "16.0001",
        "359.9",
        "360",
        "1000",
        "3600",
        "1e6",
        "1e9",
        "-1e9",
        "4294967295",
    ];
    let mut oks = 0;
    for _ in 0..20_000 {
        let mut pick: Vec<usize> = (0..1 + r.n(3)).map(|_| r.n(spans.len())).collect();
        pick.sort_unstable();
        pick.dedup();
        let mut s = good.clone();
        for &k in pick.iter().rev() {
            s.replace_range(spans[k].clone(), pool[r.n(pool.len())]);
        }
        oks += Project::load(&s).is_ok() as u32;
        check(&s);
    }
    assert!(oks > 2000, "some swaps stay valid ({oks})");
    let ok = "{\"x5edit\": 1, \"fps\": [30, 1], \"files\": [{\"path\": \"\\u00fc\\ud83d\\ude00\", \"dur\": 5}], \
              \"clips\": [{\"file\": 0, \"a\": 1, \"b\": 2, \"future\": {\"x\": [1, 2]}}], \"view\": []}";
    let p = Project::load(ok).unwrap();
    assert_eq!((p.tl().files[0].path.as_str(), p.tl().len()), ("ü😀", 30));
}
