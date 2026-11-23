//! Effects presets that are pure editing: clips for an editor timeline (crate edit), which the exporter renders
//! (`export::Cut::Edit`). Our own content. View keys sit on the clip's content time (source s), so trimming and speed
//! changes keep them on their pictures. Directions are in the equirect frame the exporter reframes: the clip stitched
//! with its pose (stabilisation, framing), as `Clip::frames` with `Clip::pose` renders it for the tracker.

// `!(x > 0.0)` rejects NaN on purpose
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use edit::{Clip, Ease, Key, Proj, Rate, Result, Timeline, View, view::PLANET_FOV};
use glam::{Quat, Vec2, Vec3};
use std::f32::consts::{FRAC_PI_2, TAU};

fn bad<T>(m: &str) -> Result<T> {
    Err(edit::Error(m.into()))
}

/// A timeline of `clips` from one source file (`path`, `dur` s) at `fps`: fixed and checked (`Timeline::check`).
pub fn timeline(path: &str, dur: f64, fps: Rate, clips: Vec<Clip>) -> Result<Timeline> {
    let mut tl = Timeline::new(fps);
    tl.files.push(edit::File {
        path: path.into(),
        dur,
    });
    tl.clips = clips;
    tl.fix();
    tl.check()?;
    Ok(tl)
}

/// Stop Motion: source `a..b` of `file` as freeze frames `step` s of source apart, each held `hold` s, so the picture
/// jumps (step 0.5 s, hold 1/6 s: 3× as fast at 6 pictures per second).
pub fn stop_motion(file: u32, a: f64, b: f64, step: f64, hold: f64) -> Result<Vec<Clip>> {
    if !(step > 0.0 && hold > 0.0 && b > a && (b - a) / step < 1e5) {
        return bad("stop motion");
    }
    let n = ((b - a) / step).ceil() as usize;
    Ok((0..n)
        .map(|i| Clip::still(file, a + i as f64 * step, hold))
        .collect())
}

/// Hyperlapse speed: `speed`× over `a..b` (`edit::MAX_SPEED` at most), up from 1× over the first `ramp` source s and
/// back to 1× over the last.
fn lapse(file: u32, a: f64, b: f64, speed: f64, ramp: f64) -> Result<Clip> {
    if !(b > a && (edit::MIN_SPEED..=edit::MAX_SPEED).contains(&speed) && ramp >= 0.0) {
        return bad("hyperlapse");
    }
    let mut c = Clip::new(file, a, b);
    c.speed = if ramp > 0.0 && 2.0 * ramp < b - a {
        vec![(a, 1.0), (a + ramp, speed), (b - ramp, speed), (b, 1.0)]
    } else {
        vec![(a, speed)]
    };
    Ok(c)
}

/// Output-time keys (from the clip's start) onto its content time.
fn on(c: &Clip, k: Vec<Key>) -> Vec<Key> {
    let d = c.dur();
    k.into_iter()
        .map(|k| Key {
            t: c.src(k.t.clamp(0.0, d)).min(c.b),
            ..k
        })
        .collect()
}

/// Orbit Lapse: a hyperlapse of `a..b` (`speed`×, `ramp` as in `lapse`) with view `v` orbiting `orbit` times about the
/// vertical at constant speed, unfolding from a tiny planet over the first `planet` output s (0: none).
#[allow(clippy::too_many_arguments)]
pub fn orbit(
    file: u32,
    a: f64,
    b: f64,
    speed: f64,
    ramp: f64,
    v: View,
    orbit: f32,
    planet: f64,
) -> Result<Clip> {
    let mut c = lapse(file, a, b, speed, ramp)?;
    let d = c.dur();
    let mut k = edit::view::preset::orbit(v, d, orbit)?;
    if planet > 0.0 && planet < d {
        let at = edit::view::at(&k, planet, false).unwrap_or(v);
        k.retain(|x| x.t > planet + edit::GAP);
        k.splice(0..0, edit::view::preset::planet(at, planet, false)?);
    }
    c.view = on(&c, k);
    Ok(c)
}

/// Roll Lapse: a hyperlapse of `a..b` (`speed`×, `ramp` as in `lapse`) while view `v` rolls `turns` full turns about
/// its axis, the world going over and coming back upright, easing in and out.
pub fn roll(file: u32, a: f64, b: f64, speed: f64, ramp: f64, v: View, turns: f32) -> Result<Clip> {
    if !(turns.is_finite() && turns != 0.0 && turns.abs() <= 16.0) {
        return bad("roll lapse");
    }
    let mut c = lapse(file, a, b, speed, ramp)?;
    let (d, n) = (c.dur(), (turns.abs() * 4.0).ceil() as usize);
    // quarter turns at most between keys: the interpolation takes the short way
    let k = (0..=n)
        .map(|i| Key {
            t: d * i as f64 / n as f64,
            v: View {
                q: (v.q * Quat::from_rotation_z(TAU * turns * i as f32 / n as f32)).normalize(),
                ..v
            },
            e: match i {
                0 => Ease::In,
                i if i + 1 == n => Ease::Out,
                _ => Ease::Smooth,
            },
        })
        .collect();
    c.view = on(&c, k);
    Ok(c)
}

/// A subject over time: source time (s), direction (equirect frame, see the module) and angular width, height
/// (degrees); e.g. `track::Tracker` samples, their directions smoothed with `track::smooth`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subj {
    pub t: f64,
    pub dir: Vec3,
    pub size: Vec2,
}

/// Keys on the subject every `every` source s over its track within `a..b`: `f(sample, share of the way 0..1)`. The
/// sample's size is the median over the samples within `every` / 2 of it: a tracker's box jitters by ±15 % from frame to
/// frame, and the field of view must not pump with it.
fn follow(
    s: &[Subj],
    a: f64,
    b: f64,
    every: f64,
    mut f: impl FnMut(&Subj, f32) -> View,
) -> Result<Vec<Key>> {
    let s: Vec<&Subj> = s.iter().filter(|x| x.t >= a && x.t <= b).collect();
    if s.len() < 2 || !(every > 0.0) {
        return bad("subject track: at least two samples inside the clip");
    }
    let (t0, t1) = (s[0].t, s[s.len() - 1].t);
    let mut k: Vec<Key> = vec![];
    let mut sz = vec![];
    for (i, x) in s.iter().enumerate() {
        if i + 1 == s.len() || k.last().is_none_or(|l| x.t - l.t >= every) {
            if i + 1 == s.len() && k.len() > 1 && k.last().is_some_and(|l| x.t - l.t < every / 2.0)
            {
                k.pop();
            }
            sz.clear();
            sz.extend(
                s.iter()
                    .filter(|y| (y.t - x.t).abs() <= every / 2.0)
                    .map(|y| y.size),
            );
            let m = |c: usize| {
                let mut v: Vec<f32> = sz.iter().map(|z| z[c]).collect();
                v.sort_by(f32::total_cmp);
                v[v.len() / 2]
            };
            let x = Subj {
                size: Vec2::new(m(0), m(1)),
                ..**x
            };
            k.push(Key {
                t: x.t,
                v: f(&x, ((x.t - t0) / (t1 - t0).max(1e-9)) as f32),
                e: Ease::Smooth,
            });
        }
    }
    Ok(k)
}

/// Horizontal field of view (degrees) for an `aspect` view in which angular height `h` fills `fill` of the frame.
fn fit(h: f32, fill: f32, aspect: f32) -> f32 {
    let v = (h / fill.clamp(0.05, 1.0)).clamp(5.0, 150.0).to_radians() / 2.0;
    (2.0 * (v.tan() * aspect).atan())
        .to_degrees()
        .clamp(10.0, 170.0)
}

fn look(d: Vec3, roll: f32, fov: f32, proj: Proj) -> View {
    let [y, p] = track::yaw_pitch(d);
    View::look(y, p, roll, fov, proj)
}

/// Subject Framing: source `a..b` of `file` reframed on the subject, its height filling `fill` of an `aspect` frame,
/// keys every `every` s.
pub fn framing(
    file: u32,
    a: f64,
    b: f64,
    s: &[Subj],
    fill: f32,
    aspect: f32,
    every: f64,
) -> Result<Clip> {
    let mut c = Clip::new(file, a, b);
    c.view = follow(s, a, b, every, |x, _| {
        look(x.dir, 0.0, fit(x.size.y, fill, aspect), Proj::Persp)
    })?;
    Ok(c)
}

/// Passing Vehicle: source `a..b` of `file` turned onto a passing vehicle (`track::Pass::path`) at a fixed horizontal
/// field of view `fov` (fitting its height would zoom from a few degrees to over 100 as it comes abeam), keys every
/// `every` s.
pub fn passing(file: u32, a: f64, b: f64, s: &[Subj], fov: f32, every: f64) -> Result<Clip> {
    if !(fov > 0.0 && fov < 170.0) {
        return bad("passing vehicle");
    }
    let mut c = Clip::new(file, a, b);
    c.view = follow(s, a, b, every, |x, _| look(x.dir, 0.0, fov, Proj::Persp))?;
    Ok(c)
}

/// Subject Spin: Subject Framing while the picture turns `turns` full turns about the subject.
pub fn spin(
    file: u32,
    a: f64,
    b: f64,
    s: &[Subj],
    fill: f32,
    aspect: f32,
    turns: f32,
) -> Result<Clip> {
    if !(turns.is_finite() && turns.abs() <= 16.0) {
        return bad("spin");
    }
    // a quarter turn at most between keys, so the interpolation turns the intended way
    let every = ((b - a) / (4.0 * turns.abs() as f64).max(1.0)).min(0.5);
    let mut c = Clip::new(file, a, b);
    c.view = follow(s, a, b, every, |x, f| {
        look(
            x.dir,
            360.0 * turns * f,
            fit(x.size.y, fill, aspect),
            Proj::Persp,
        )
    })?;
    Ok(c)
}

/// Pull Out: Subject Framing that pulls out over the last `out` source s into a tiny planet facing the subject's
/// heading.
pub fn pull_out(
    file: u32,
    a: f64,
    b: f64,
    s: &[Subj],
    fill: f32,
    aspect: f32,
    out: f64,
) -> Result<Clip> {
    if !(out > 0.0 && out < b - a) {
        return bad("pull out");
    }
    let mut c = framing(file, a, b - out, s, fill, aspect, 0.5)?;
    c.b = b;
    let l = c.view.last_mut().unwrap();
    l.e = Ease::InOut;
    let y = l.v.ypr()[0].to_radians();
    c.view.push(Key {
        t: b,
        v: View {
            q: Quat::from_rotation_y(y) * Quat::from_rotation_x(FRAC_PI_2),
            fov: PLANET_FOV,
            proj: Proj::Planet(1.0),
        },
        e: Ease::Smooth,
    });
    Ok(c)
}

/// Dolly Zoom: reframed on the subject with the field of view following its apparent height, so it keeps its size on
/// screen while the camera walks toward it (the background falls away) or away from it (the background closes in).
/// `fov`: the horizontal field of view at the first key; keys every `every` s.
pub fn dolly(file: u32, a: f64, b: f64, s: &[Subj], fov: f32, every: f64) -> Result<Clip> {
    let h0 = s.iter().find(|x| x.t >= a).map(|x| x.size.y).unwrap_or(0.0);
    if !(h0 > 0.0 && fov > 0.0 && fov < 170.0) {
        return bad("dolly zoom");
    }
    let t0 = (fov / 2.0).to_radians().tan() / (h0 / 2.0).to_radians().tan();
    let mut c = Clip::new(file, a, b);
    c.view = follow(s, a, b, every, |x, _| {
        let f = 2.0 * (t0 * (x.size.y.clamp(0.5, 170.0) / 2.0).to_radians().tan()).atan();
        look(x.dir, 0.0, f.to_degrees().clamp(5.0, 170.0), Proj::Persp)
    })?;
    Ok(c)
}
