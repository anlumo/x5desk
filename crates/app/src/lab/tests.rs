use super::*;
use crate::ai::tests::{gpu, made, samples, wait};
use glam::Vec2;
use std::sync::OnceLock;

/// The effects' table: keys unique and parsed back, the passes and the editing presets apart, what needs a still
/// camera or a subject.
#[test]
fn table() {
    for (f, _, k, _) in FXS {
        assert_eq!(Fx::parse(k), Some(f));
        assert_eq!(FXS.iter().filter(|x| x.2 == k).count(), 1);
    }
    assert_eq!(FXS.iter().filter(|x| x.0.cut()).count(), 8);
    assert!(FXS[..7].iter().all(|x| !x.0.cut()) && FXS[7..].iter().all(|x| x.0.cut()));
    assert!(
        Fx::Passerby.still() && Fx::Shadow.still() && !Fx::Planets.still() && !Fx::Delay.still()
    );
    assert!(Fx::Center.subject() && Fx::Loop.subject() && !Fx::Trails.subject());
    assert!(Fx::Overtake.cut() && !Fx::Overtake.subject() && !Fx::Overtake.still());
}

/// A Lab on clip `f` (its whole range), the library's spec of it.
fn lab(g: &Arc<Gpu>, cx: &stitch::Ctx, f: &std::path::Path) -> Lab {
    let mut l = Lab::new(g, cx, &egui::Context::default(), None);
    l.tick(Some(Spec::of(f).unwrap()), None, None);
    l
}

/// The editing presets on v635 (`X5_SAMPLES`): Fly Lapse and Stop Motion over 0.5–4.5 s are timelines crate edit
/// checks, at the source's frame rate, in the source's stabilisation; Center Stage wants the Deep Track subject, and
/// with one (made up, on this file) its keys follow it inside the range; Overtake wants the auto framing run's passes,
/// and with one (made up) its keys follow the vehicle at a fixed field of view; a pass is no editing preset. "Add to
/// timeline": the clips go onto an empty editor timeline as one undo step, which takes their stabilisation.
#[test]
fn presets_v635() {
    let Some(d) = samples() else { return };
    let (_l, g, cx) = gpu();
    let f = d.join("v635.insv");
    let mut l = lab(&g, &cx, &f);
    l.range = [0.5, 4.5];
    l.fx = Fx::FlyLapse;
    let (tl, st) = l.timeline().unwrap();
    tl.check().unwrap();
    assert_eq!(
        (tl.clips.len(), tl.out.fps.num, tl.out.fps.den),
        (1, 30000, 1001)
    );
    assert_eq!(st, (Some(stab::Lock::Horizon), gyro::SMOOTH));
    assert!(tl.dur() < 4.0 / 2.0 && tl.clips[0].view.len() > 2);
    l.fx = Fx::StopMotion;
    let (sm, _) = l.timeline().unwrap();
    assert_eq!(sm.clips.len(), 8);
    assert!(sm.clips.iter().all(|c| c.hold > 0.0));
    l.fx = Fx::Center;
    assert!(l.timeline().unwrap_err().contains("Deep Track"));
    let mut t = made();
    t.file = f.clone();
    l.tick(Some(Spec::of(&f).unwrap()), Some(Arc::new(t)), None);
    l.range = [0.5, 2.0];
    let (cs, _) = l.timeline().unwrap();
    let c = &cs.clips[0];
    assert!(
        c.view.len() >= 3
            && c.view
                .iter()
                .all(|k| k.t >= 0.5 - 1e-9 && k.t <= 2.0 + 1e-9)
    );
    for k in &c.view {
        let [y, p, _] = k.v.ypr();
        let want = (k.t * 30.0).round() as f32 - 10.0;
        assert!(
            (y - want).abs() < 4.0 && (p - 10.0).abs() < 1.0,
            "{} {y} {p}",
            k.t
        );
    }
    // Overtake wants the player's auto framing run; with one (made up: a vehicle on the right from 1 to 2 s, sweeping
    // from 45° to 135° in yaw, seen every 10th frame from 0.67 to 2.33 s) the range is its pass and a second either
    // side, and the keys turn onto it at the field of view chosen
    l.fx = Fx::Overtake;
    assert!(l.timeline().unwrap_err().contains("Auto framing"));
    let mut a = made();
    (a.file, a.auto) = (f.clone(), true);
    let yaw = |k: usize| 45.0 + 3.0 * (k as f32 - 30.0);
    a.passes = OnceLock::from(vec![track::Pass {
        a: 30,
        b: 60,
        at: 45,
        right: true,
        near: 2.0,
        path: (20..=70)
            .step_by(10)
            .map(|k| (k, track::dir(yaw(k), -5.0), Vec2::new(30.0, 15.0)))
            .collect(),
    }]);
    l.tick(Some(Spec::of(&f).unwrap()), None, Some(Arc::new(a)));
    l.around(0);
    l.p.fov = 70.0;
    let (ot, st) = l.timeline().unwrap();
    let c = &ot.clips[0];
    assert_eq!((l.range, st, c.view.len()), ([0.0, 3.0], (None, 0.0), 6));
    for k in &c.view {
        let [y, p, _] = k.v.ypr();
        let want = yaw((k.t * 30.0).round() as usize);
        assert!(
            (y - want).abs() < 0.5 && (p + 5.0).abs() < 0.5 && (k.v.fov - 70.0).abs() < 1e-3,
            "{} {y} {p} {}",
            k.t,
            k.v.fov
        );
    }
    l.fx = Fx::Trails;
    assert!(l.timeline().is_err());
    let mut e = crate::editor::Editor::new(
        &g,
        &cx,
        &egui::Context::default(),
        crate::player::still(&g, &cx),
        None,
    );
    let st = (Some(stab::Lock::Direction), 0.0);
    assert_eq!(e.put(tl.clone(), st).unwrap(), 0);
    assert_eq!(
        (e.pj.tl().clips.len(), e.x.stab, e.x.smooth),
        (1, st.0, st.1)
    );
    e.tick(true);
    assert_eq!(e.history().0, 1, "one undo step");
    assert!(
        e.put(sm, (None, 0.0))
            .unwrap_err()
            .contains("stabilisation")
    );
    assert_eq!(e.put(tl, st).unwrap(), 1);
}

/// The light trails' preview on v627 (`X5_SAMPLES`) over 8–9 s (30 frames, the camera on a stick at night): a 2048 ×
/// 1024 picture of the out point that lightens the frame itself (the trails are a per-channel maximum laid over it)
/// and differs from it where lights moved.
#[test]
fn preview_trails() {
    let Some(d) = samples() else { return };
    let (_l, g, cx) = gpu();
    let f = d.join("v627.insv");
    let mut l = lab(&g, &cx, &f);
    (l.fx, l.range, l.p.half) = (Fx::Trails, [8.0, 9.0], 0.0);
    l.preview();
    let p = l.pre.as_ref().unwrap().1.clone();
    let r = wait(|| p.st.lock().unwrap().clone(), 300);
    assert_eq!(r, ai::St::Done, "{r:?}");
    let (px, w, h) = p.px.lock().unwrap().take().unwrap();
    assert_eq!((w, h, px.len()), (PW, PW / 2, (PW * PW / 2 * 4) as usize));
    // the out point's frame as it is, stitched the same way (the same frames before it: colour matching follows them)
    let c = shotlab::Clip::open(&f).unwrap();
    let ps = c.poses(stab::Lock::Direction, 0.0).unwrap();
    let (k0, k1) = (
        crate::clip::frame_at(&c.times, 8.0),
        c.times.partition_point(|&t| t < 9.0 - 1e-3) - 1,
    );
    let mut plain = vec![];
    let ks: Vec<usize> = (k0..=k1).collect();
    c.frames(
        &g,
        &cx,
        PW,
        false,
        &ks,
        |k| c.pose(k, Some(&ps)),
        |t, k| {
            if k == k1 {
                plain = stitch::read_texture(&g.device, &g.queue, t);
            }
            Ok(())
        },
    )
    .unwrap();
    let (mut darker, mut lit) = (0u8, 0usize);
    for (a, b) in px.chunks(4).zip(plain.chunks(4)) {
        darker = darker.max((0..3).map(|i| b[i].saturating_sub(a[i])).max().unwrap());
        lit += (0..3).any(|i| a[i] > b[i].saturating_add(24)) as usize;
    }
    let n = (w * h) as usize;
    eprintln!(
        "trails preview: no pixel darker than the frame by more than {darker} levels, {:.3} % lit by trails (> 24 levels)",
        100.0 * lit as f64 / n as f64
    );
    assert!(darker <= 2, "the trails darkened the frame");
    assert!(lit > 100, "no trails");
}

/// Shadow Clone's export hook runs its person network under BURN (burn submits from the export thread, which calls
/// the hook outside GATE): with a subject made up on v635 (`X5_SAMPLES`), the hook waits while another thread holds
/// BURN and runs its frames once that lets go.
#[test]
fn clones_burn() {
    let Some(d) = samples() else { return };
    let (_l, g, cx) = gpu();
    let f = d.join("v635.insv");
    let mut l = lab(&g, &cx, &f);
    let mut t = made();
    t.file = f.clone();
    l.tick(Some(Spec::of(&f).unwrap()), Some(Arc::new(t)), None);
    l.fx = Fx::Shadow;
    let (w, h) = (512, 256);
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let mut hk = (l.maker(30.0, 10).unwrap())(&g, w, h, fmt).unwrap();
    use wgpu::TextureUsages as U;
    let tex = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: U::TEXTURE_BINDING | U::STORAGE_BINDING | U::COPY_SRC | U::COPY_DST,
        view_formats: &[],
    });
    let held = BURN.lock().unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let (g2, d2) = (g.clone(), done.clone());
    let th = thread::spawn(move || {
        for k in 0..3 {
            let mut e = g2.device.create_command_encoder(&Default::default());
            hk(export::Fx {
                enc: &mut e,
                tex: &tex,
                k,
                t: k as f64 / 30.0,
            });
            let _g = GATE.read();
            g2.queue.submit([e.finish()]);
        }
        d2.store(true, Relaxed);
    });
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert!(!done.load(Relaxed), "the hook ran while BURN was held");
    drop(held);
    th.join().unwrap();
    assert!(done.load(Relaxed));
}
