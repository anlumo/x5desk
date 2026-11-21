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

/// Skips a test that loads Core ML in this process unless `CFFIXED_USER_HOME` is set: Core ML's ANE cache follows it,
/// not `HOME`, so without it the cache would go to the real home.
pub(crate) fn no_cf_home() -> bool {
    let unset = cfg!(target_os = "macos") && std::env::var_os("CFFIXED_USER_HOME").is_none();
    if unset {
        eprintln!("CFFIXED_USER_HOME unset (Core ML's cache would go to the real home): skipped");
    }
    unset
}

/// A progress note goes once its steps end, also when one fails (the warm-up's note stayed up after a failure).
#[test]
fn steps_note() {
    let m = Models {
        ai: Mutex::new(None),
        note: Mutex::new(String::new()),
        holder: Mutex::new(None),
    };
    let mut seen = vec![];
    let r = m.steps("Warming", 3, None, |i| {
        seen.push(m.note());
        match i {
            1 => Err("out of memory".into()),
            _ => Ok(()),
        }
    });
    assert_eq!(seen, ["Warming: 1 of 3", "Warming: 2 of 3"]);
    assert_eq!((r, m.note()), (Err("out of memory".into()), String::new()));
}

/// `with_stop`'s wait for the models (`acquire`) while another thread holds them: a stop ends it within 50 ms, with
/// `CANCELLED`; with the stop clear it waits and gets them once the other lets go.
#[test]
fn with_stop() {
    let m = Models {
        ai: Mutex::new(None),
        note: Mutex::new(String::new()),
        holder: Mutex::new(None),
    };
    let (stop, ms) = (AtomicBool::new(false), std::time::Duration::from_millis);
    thread::scope(|s| {
        let held = lock(&m.ai);
        let w = s.spawn(|| (m.acquire(&stop).map(drop), Instant::now()));
        thread::sleep(ms(100));
        assert!(!w.is_finished(), "got models another holds");
        let t = Instant::now();
        stop.store(true, Relaxed);
        let (r, at) = w.join().unwrap();
        assert_eq!(r, Err(CANCELLED.into()));
        let d = at.saturating_duration_since(t);
        assert!(d < ms(50), "{d:?} after the stop");
        stop.store(false, Relaxed);
        let w = s.spawn(|| m.acquire(&stop).is_ok());
        thread::sleep(ms(100));
        assert!(!w.is_finished(), "got models another holds");
        drop(held);
        assert!(w.join().unwrap());
    });
}

/// `trim` with none loaded, or while another thread holds the models (it doesn't wait for them): nothing dropped;
/// loaded (on this module's device) and free: dropped, once.
#[test]
fn trim() {
    let m = Models {
        ai: Mutex::new(None),
        note: Mutex::new(String::new()),
        holder: Mutex::new(None),
    };
    assert!(!m.trim());
    let (_l, g, _) = gpu();
    m.with(&g, false, None, |_| Ok(())).unwrap();
    thread::scope(|s| {
        let held = lock(&m.ai);
        assert!(!s.spawn(|| m.trim()).join().unwrap());
        drop(held);
    });
    assert!(m.loaded() == Some(true) && m.trim());
    assert!(m.loaded() == Some(false) && !m.trim());
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
        auto: false,
        times: OnceLock::from(times),
        path: OnceLock::from(track::smooth(&smp, 7.5)),
        passes: OnceLock::new(),
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

/// A box drawn on a perspective view is the subject seen there: its outline in that view, one closed line, comes back
/// on the box's corners (a centred box within a point; the tall one of the follow-view run within 6 % of the height,
/// the box being upright on the sphere, not in the view). In the equirect a box behind the view's direction shows at
/// both edges of the picture, nowhere else.
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
        let o = outline(&v, r, s.dir, s.size);
        assert!(o.len() == 1 && o[0].first() == o[0].last(), "{o:?}");
        let want = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
            .map(|(x, y)| r.min + egui::vec2(x * r.width(), y * r.height()));
        for w in want {
            let a = o[0].iter().map(|a| a.distance(w)).fold(f32::MAX, f32::min);
            assert!(a < tol, "{b:?}: {a} from {w:?}");
        }
    }
    let e = View {
        mode: stitch::Mode::Equirect,
        yaw: 30.0,
        ..View::default()
    };
    let o = outline(&e, r, track::dir(210.0, 0.0), Vec2::splat(10.0));
    let x: Vec<f32> = (o.iter().flatten())
        .filter(|p| r.contains(**p))
        .map(|p| p.x - r.left())
        .collect();
    assert!(x.iter().any(|&x| x < 26.0) && x.iter().any(|&x| x > 1802.0 - 26.0));
    assert!(x.iter().all(|&x| !(26.0..=1802.0 - 26.0).contains(&x)));
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
    if no_cf_home() {
        return;
    }
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
        None,
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
    assert!(!tl.clips.is_empty() && tl.dur() >= 1.9 && tl.dur() <= 5.9 && tl.out.music.is_none());
    for c in &tl.clips {
        assert!(c.view.len() >= 2);
        for k in &c.view {
            assert!(k.t >= c.a - 1e-9 && k.t <= c.b + 1e-9 && k.v.proj == edit::Proj::Persp);
        }
    }
}

/// Auto framing of v635 (`X5_SAMPLES`, 5.8 s, horizon lock): a run from frame 0 without samples whose path covers every
/// frame and turns smoothly: between frames at most what the framer's smoothing leaves of a 180° hand-over (120°/s at
/// σ 0.6 s); keys from it span the whole clip.
#[test]
fn framing_v635() {
    if no_cf_home() {
        return;
    }
    let Some(d) = samples() else { return };
    let (_l, g, cx) = gpu();
    let t0 = Instant::now();
    let st = (Some(stab::Lock::Horizon), gyro::SMOOTH);
    let ctx = egui::Context::default();
    let t = Track::framing(&g, &cx, &ctx, d.join("v635.insv"), st, 16.0 / 9.0);
    let r = wait(|| t.st(), 600);
    assert_eq!(r, St::Done, "{r:?}");
    let (ts, p) = (t.times.get().unwrap(), t.path.get().unwrap());
    let n = ts.len();
    let fps = (n - 1) as f32 / (ts[n - 1] - ts[0]) as f32;
    let fast = (p.windows(2))
        .map(|w| w[0].angle_between(w[1]).to_degrees() * fps)
        .fold(0f32, f32::max);
    let [y, x] = track::yaw_pitch(p[0]);
    eprintln!(
        "v635 auto framing in {:.1} s (GPU shared: noisy): {n} frames, from yaw {y:.1}° pitch {x:.1}°, fastest turn \
         {fast:.1}°/s",
        t0.elapsed().as_secs_f64()
    );
    assert!(t.auto && t.k0 == 0 && lock(&t.smp).is_empty() && p.len() == n);
    assert!(fast < 120.0);
    let v = edit::View::look(0.0, 0.0, 0.0, 100.0, edit::Proj::Persp);
    let k = keys(&t, 0.0, 10.0, 0.5, &v);
    assert!(k.len() >= 11 && k[0].t == ts[0] && (k[k.len() - 1].t - ts[n - 1]).abs() < 1e-9);
}

/// Overtakes on a ride (`X5_RIDES`, the rides' folder: 2026-04-16 16:52, 60 s at 60 fps, horizon lock): auto framing
/// finds the two cars that overtake on the left, abeam at 3.85 and 34.03 s (by eye: a silver saloon as we roll off, a
/// dark hatchback while we wait), each 1–3 camera heights away, and nothing else, past rows of parked cars on both
/// sides.
#[test]
fn overtakes_ride() {
    if no_cf_home() {
        return;
    }
    let Some(f) = std::env::var_os("X5_RIDES")
        .map(|d| PathBuf::from(d).join("VID_20260416_165209_00_738.insv"))
    else {
        eprintln!("X5_RIDES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let t0 = Instant::now();
    let st = (Some(stab::Lock::Horizon), gyro::SMOOTH);
    let t = Track::framing(&g, &cx, &egui::Context::default(), f, st, 16.0 / 9.0);
    let r = wait(|| t.st(), 1800);
    assert_eq!(r, St::Done, "{r:?}");
    let (ts, p) = (t.times.get().unwrap(), t.passes.get().unwrap());
    eprintln!(
        "ride overtakes in {:.1} s: {:?}",
        t0.elapsed().as_secs_f64(),
        (p.iter())
            .map(|x| (ts[x.at], x.right, x.near, ts[x.a], ts[x.b], x.path.len()))
            .collect::<Vec<_>>()
    );
    assert_eq!(p.len(), 2);
    for (x, at) in p.iter().zip([3.85, 34.03]) {
        assert!(!x.right && (ts[x.at] - at).abs() < 0.3 && (1.0..3.0).contains(&x.near));
    }
}

/// A 16-bit stereo 48 kHz WAV of a drum loop at `bpm`, `secs` long, the first beat 0.23 s in: a kick (a falling sine)
/// on every beat, the bar's first louder, a hi-hat (differenced noise) on the off-beats.
fn drums(p: &std::path::Path, secs: f64, bpm: f64) {
    let (n, per) = ((secs * 48000.0) as usize, 60.0 / bpm);
    let mut y = vec![0f32; n];
    let mut r = 1u32;
    for (i, t) in (0..)
        .map(|i| 0.23 + i as f64 * per)
        .take_while(|&t| t < secs)
        .enumerate()
    {
        let a = if i % 4 == 0 { 0.7 } else { 0.45 };
        let i0 = (t * 48000.0) as usize;
        for (j, v) in y[i0..n.min(i0 + 19200)].iter_mut().enumerate() {
            let u = j as f64 / 48000.0;
            let ph = std::f64::consts::TAU * (50.0 * u + 2.1 * (1.0 - (-u / 0.03).exp()));
            *v += (a * ph.sin() * (-u / 0.15).exp()) as f32;
        }
        let i0 = ((t + per / 2.0) * 48000.0) as usize;
        let mut last = 0f32;
        for (j, v) in y[i0.min(n)..n.min(i0 + 2880)].iter_mut().enumerate() {
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            let x = r as f32 / 2147483648.0 - 1.0;
            *v += 0.08 * (x - last) * (-(j as f32) / 720.0).exp();
            last = x;
        }
    }
    let mut b = Vec::with_capacity(44 + 4 * n);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + 4 * n as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    for v in [16u32, 0x0002_0001, 48000, 4 * 48000, 0x0010_0004] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(4 * n as u32).to_le_bytes());
    for v in y {
        let s = ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes();
        b.extend_from_slice(&[s[0], s[1], s[0], s[1]]);
    }
    std::fs::write(p, b).unwrap();
}

/// Auto edit on v635 (`X5_SAMPLES`) with music, a drum loop at 100 BPM made here: the timeline carries it as its
/// music from its start, ducked at gain 1 (the editor's import); every cut (each clip's end, the last too) lies within
/// a frame of one of its beats (`media::beat::of`); each clip at least 1 s long and keyed inside its source span; the
/// whole between 1 s and the clip's length.
#[test]
fn auto_v635_music() {
    if no_cf_home() {
        return;
    }
    let Some(d) = samples() else { return };
    let tmp = crate::clip::tests::Tmp::new("auto-music");
    let m = tmp.0.join("drums100.wav");
    drums(&m, 20.0, 100.0);
    let (_l, g, cx) = gpu();
    let a = Auto::start(
        &g,
        &cx,
        &egui::Context::default(),
        d.join("v635.insv"),
        (Some(stab::Lock::Horizon), gyro::SMOOTH),
        16.0 / 9.0,
        Some(m.clone()),
    );
    let r = wait(|| a.st(), 600);
    assert_eq!(r, St::Done, "{r:?}");
    let tl = lock(&a.out).clone().unwrap();
    tl.check().unwrap();
    let mu = tl.out.music.clone().unwrap();
    assert_eq!((mu.file, mu.offset, mu.gain, mu.duck), (1, 0.0, 1.0, true));
    assert_eq!(PathBuf::from(&tl.files[1].path), m);
    let b = media::beat::of(&m).unwrap();
    let dt = tl.out.fps.t(1);
    let cuts: Vec<(f64, f64)> = (0..tl.clips.len())
        .map(|k| {
            let t = tl.out.fps.t(tl.frames(k).end);
            let e =
                b.t.iter()
                    .map(|x| x.0 - t)
                    .min_by(|x, y| x.abs().total_cmp(&y.abs()));
            (t, e.unwrap())
        })
        .collect();
    eprintln!(
        "v635 auto edit to {:.1} BPM ({} beats): {} clips, {:.2} s, cuts (s, off the nearest beat): {cuts:?}; clips {:?}",
        b.bpm,
        b.t.len(),
        tl.clips.len(),
        tl.dur(),
        tl.clips.iter().map(|c| (c.a, c.b)).collect::<Vec<_>>()
    );
    assert!(cuts.iter().all(|c| c.1.abs() <= dt));
    assert!(tl.dur() >= 1.0 && tl.dur() <= tl.files[0].dur);
    for c in &tl.clips {
        assert!(c.b - c.a >= 1.0 - 1e-9 && c.view.len() >= 2);
        assert!(
            c.view
                .iter()
                .all(|k| k.t >= c.a - 1e-9 && k.t <= c.b + 1e-9)
        );
    }
}
