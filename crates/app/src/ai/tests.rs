use super::*;
use std::time::Instant;

/// One device for the GPU tests of this module and `lab`'s, one test at a time.
pub(crate) fn gpu() -> (MutexGuard<'static, ()>, Arc<Gpu>, stitch::Ctx) {
    static G: OnceLock<(Mutex<()>, Arc<Gpu>, stitch::Ctx)> = OnceLock::new();
    let (m, g, cx) = G.get_or_init(|| {
        let g = Arc::new(Gpu::new().expect("GPU"));
        let cx = stitch::Ctx::new(&g.device, &g.queue);
        (Mutex::new(()), g, cx)
    });
    (lock(m), g.clone(), cx.clone())
}

pub(crate) fn samples() -> Option<PathBuf> {
    let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        eprintln!("X5_SAMPLES unset: skipped");
    }
    d
}

/// Waits for a job to end (at most `secs`).
pub(crate) fn wait(st: impl Fn() -> St, secs: u64) -> St {
    let t = Instant::now();
    loop {
        let s = st();
        if s.over() || t.elapsed().as_secs() > secs {
            return s;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// A finished run made up: 30 fps, frames 10..70 tracked (a pan of 1° per frame in yaw at pitch 10°), 30..40
/// missing.
pub(crate) fn made() -> Track {
    let times: Vec<f64> = (0..100).map(|k| k as f64 / 30.0).collect();
    let smp: Vec<Sample> = (10..70)
        .map(|k| {
            let d = track::dir(k as f32 - 10.0, 10.0);
            let miss = (30..40).contains(&k);
            Sample {
                raw: d,
                dir: d,
                size: Vec2::new(8.0, 20.0),
                score: if miss { 0.0 } else { 0.9 },
                state: if miss { State::Miss } else { State::Tracked },
            }
        })
        .collect();
    Track {
        file: "x.insv".into(),
        stab: (None, 0.0),
        k0: 10,
        times: OnceLock::from(times),
        path: OnceLock::from(track::smooth(&smp, 7.5)),
        qs: OnceLock::from(vec![Quat::IDENTITY; 100]),
        smp: Mutex::new(smp),
        st: Mutex::new(St::Done),
        stop: AtomicBool::new(false),
    }
}

/// Keys follow the smoothed path: evenly spaced over the tracked span inside the range asked for, the view's
/// projection, roll and field of view kept, yaw and pitch on the subject (bridged over the missing frames); the
/// direction held outside the run; samples by frame.
#[test]
fn keys_follow() {
    let t = made();
    let v = edit::View::look(0.0, 0.0, 5.0, 70.0, edit::Proj::Panini(0.5));
    let k = keys(&t, 0.0, 10.0, 0.5, &v);
    let (t0, t1) = (10.0 / 30.0, 69.0 / 30.0);
    assert!((k[0].t - t0).abs() < 1e-9 && (k[k.len() - 1].t - t1).abs() < 1e-9);
    assert_eq!(
        k.len(),
        5,
        "{:?}",
        k.iter().map(|k| k.t).collect::<Vec<_>>()
    );
    for w in k.windows(2) {
        assert!((w[1].t - w[0].t - (t1 - t0) / 4.0).abs() < 1e-9);
    }
    for key in &k {
        let [y, p, r] = key.v.ypr();
        let want = (key.t * 30.0).round() as f32 - 10.0;
        // within σ of the 1°/frame pan: the ends and the keys by the gap see one-sided windows
        assert!(
            (y - want).abs() < 7.5 && (p - 10.0).abs() < 0.5,
            "{} {y} {p}",
            key.t
        );
        assert!(
            (r - 5.0).abs() < 1e-3 && key.v.fov == 70.0 && key.v.proj == edit::Proj::Panini(0.5)
        );
    }
    assert_eq!(keys(&t, 1.0, 1.5, 0.25, &v).len(), 3);
    assert!(keys(&t, 3.0, 4.0, 0.5, &v).is_empty(), "outside the run");
    assert_eq!(t.dir(0), t.dir(10));
    assert_eq!(t.dir(99), t.dir(69));
    assert_eq!(t.at(35).unwrap().state, State::Miss);
    assert!(t.at(9).is_none() && t.at(70).is_none());
    let b = t.boxes((None, 0.0)).unwrap();
    assert_eq!(
        (b.len(), b.iter().filter(|b| b.is_none()).count()),
        (60, 10)
    );
}

/// A box drawn on a perspective view is the subject seen there: its outline in that view comes back on the box's
/// corners (a centred box within a point; the tall one of the follow-view run within 6 % of the height, the box
/// being upright on the sphere, not in the view).
#[test]
fn box_round_trip() {
    let v = View {
        yaw: -71.0,
        pitch: -35.0,
        fov: 110.0,
        ..View::default()
    };
    let r = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(1802.0, 735.0));
    for (b, tol) in [
        ([0.45, 0.4, 0.55, 0.6], 1.0),
        ([0.41, 0.16, 0.58, 0.95], 0.06 * 735.0),
    ] {
        let s = subject(&v, r.aspect_ratio(), b);
        let o = outline(&v, r, s.dir, s.size).unwrap();
        let want = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
            .map(|(x, y)| r.min + egui::vec2(x * r.width(), y * r.height()));
        for (a, w) in o.iter().zip(want) {
            assert!(a.distance(w) < tol, "{b:?}: {a:?} vs {w:?}");
        }
    }
}

/// Deep Track from the UI's path on v606 (`X5_SAMPLES`): the woman in the purple coat boxed in the player's view
/// (yaw −71°, pitch −35°, 110°, the box of the follow-view run) on frame 0 with the horizon lock, followed through
/// 240 frames: the samples come in order, she is had on most of them, the smoothed path stays on her boxes, and the
/// boxes re-expressed under the direction lock turn by no more than the stabilisation difference.
#[test]
fn track_v606() {
    let Some(d) = samples() else { return };
    let (_l, g, cx) = gpu();
    let v = View {
        yaw: -71.0,
        pitch: -35.0,
        fov: 110.0,
        ..View::default()
    };
    let s = subject(&v, 1802.0 / 735.0, [0.41, 0.16, 0.58, 0.95]);
    let st = (Some(stab::Lock::Horizon), gyro::SMOOTH);
    let t0 = Instant::now();
    let ctx = egui::Context::default();
    let t = Track::start(&g, &cx, &ctx, d.join("v606.insv"), st, 0, 239, s);
    let r = wait(|| t.st(), 600);
    let secs = t0.elapsed().as_secs_f64();
    assert_eq!(r, St::Done, "{r:?}");
    let smp = lock(&t.smp).clone();
    assert_eq!(smp.len(), 240);
    let n = |s: State| smp.iter().filter(|x| x.state == s).count();
    eprintln!(
        "v606: 240 frames in {secs:.1} s with loading and warm-up (GPU shared: noisy); tracked {}, checked {}, \
         missing {}, lost {}, found {}; subject {:.1}° × {:.1}° at frame 0, {:.1}° × {:.1}° at 239",
        n(State::Tracked),
        n(State::Checked),
        n(State::Miss),
        n(State::Lost),
        n(State::Found),
        smp[0].size.x,
        smp[0].size.y,
        smp[239].size.x,
        smp[239].size.y
    );
    assert!(n(State::Tracked) + n(State::Checked) + n(State::Found) > 200);
    for (i, x) in smp
        .iter()
        .enumerate()
        .filter(|x| x.1.state == State::Tracked)
    {
        let a = t.dir(i).unwrap().angle_between(x.raw).to_degrees();
        assert!(a < x.size.y / 3.0, "frame {i}: path {a:.1}° off the box");
    }
    eprintln!(
        "she moved {:.1}° in the view over 8 s",
        smp[0].raw.angle_between(smp[239].raw).to_degrees()
    );
    let same = t.boxes(st).unwrap();
    let dl = t.boxes((Some(stab::Lock::Direction), 0.0)).unwrap();
    let c = shotlab::Clip::open(&t.file).unwrap();
    let (ph, pd) = (
        c.poses(stab::Lock::Horizon, gyro::SMOOTH).unwrap(),
        c.poses(stab::Lock::Direction, 0.0).unwrap(),
    );
    let mut worst = 0f32;
    for (k, (a, b)) in same.iter().zip(&dl).enumerate() {
        let (Some(a), Some(b)) = (a, b) else {
            assert!(a.is_none() && b.is_none());
            continue;
        };
        let x = (c.pose(k, Some(&pd)).q.inverse() * c.pose(k, Some(&ph)).q)
            .to_axis_angle()
            .1
            .to_degrees();
        let turn = x.min(360.0 - x);
        assert!(
            a.0.angle_between(b.0).to_degrees() <= turn + 1e-2,
            "frame {k}"
        );
        worst = worst.max(turn);
    }
    eprintln!(
        "under the direction lock her boxes turn by up to {worst:.1}° (the stabilisation difference)"
    );
}

/// Auto edit on v635 (`X5_SAMPLES`, 5.8 s): highlights of about 30 % of it (2 s at least), cut into a timeline crate
/// edit checks, each clip reframed by perspective keys inside its source span.
#[test]
fn auto_v635() {
    let Some(d) = samples() else { return };
    let (_l, g, cx) = gpu();
    let t0 = Instant::now();
    let a = Auto::start(
        &g,
        &cx,
        &egui::Context::default(),
        d.join("v635.insv"),
        (Some(stab::Lock::Horizon), gyro::SMOOTH),
        16.0 / 9.0,
    );
    let r = wait(|| a.st(), 600);
    assert_eq!(r, St::Done, "{r:?}");
    let tl = lock(&a.out).clone().unwrap();
    tl.check().unwrap();
    eprintln!(
        "v635 auto edit in {:.1} s (faces compile on first use; GPU shared: noisy): {} clips, {:.2} s: {:?}",
        t0.elapsed().as_secs_f64(),
        tl.clips.len(),
        tl.dur(),
        tl.clips
            .iter()
            .map(|c| (c.a, c.b, c.view.len()))
            .collect::<Vec<_>>()
    );
    assert!(!tl.clips.is_empty() && tl.dur() >= 1.9 && tl.dur() <= 5.9);
    for c in &tl.clips {
        assert!(c.view.len() >= 2);
        for k in &c.view {
            assert!(k.t >= c.a - 1e-9 && k.t <= c.b + 1e-9 && k.v.proj == edit::Proj::Persp);
        }
    }
}
