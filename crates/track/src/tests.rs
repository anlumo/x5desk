//! The imported networks against onnxruntime (`X5_ORT` = a Python with `onnxruntime` and `numpy`, a test tool) on
//! crops of a real frame (`X5_SAMPLES`): both see the same input tensors, read back from the GPU. Scratch files (a
//! 19 MB frame among them) go to `X5_FIXTURES` (default: the system temp dir).

use crate::{Ai, Cam, Crops, In, Job, SIDE, det, dir, plates, privacy, read, reid, vit};
use glam::{Vec2, Vec3};
use std::{path::PathBuf, process::Command, sync::OnceLock};

const ORT: &str = r#"
import sys, numpy as np, onnxruntime as ort
m, outd, *ins = sys.argv[1:]
s = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
feed = {}
for a in ins:
    name, path, shape = a.split(":")
    feed[name] = np.fromfile(path, np.float32).reshape([int(v) for v in shape.split(",")])
for i, o in enumerate(s.run(None, feed)):
    o.astype(np.float32).tofile(f"{outd}/ort{i}.f32")
"#;

fn ort(
    py: &str,
    model: &str,
    d: &std::path::Path,
    ins: &[(&str, &[f32], &[usize])],
) -> Vec<Vec<f32>> {
    let mut args = vec![];
    for (name, v, sh) in ins {
        let p = d.join(format!("{name}.f32"));
        std::fs::write(&p, bytemuck::cast_slice(v)).unwrap();
        let sh: Vec<String> = sh.iter().map(|v| v.to_string()).collect();
        args.push(format!("{name}:{}:{}", p.display(), sh.join(",")));
    }
    let m = concat!(env!("CARGO_MANIFEST_DIR"), "/models/").to_string() + model;
    for i in 0..16 {
        let _ = std::fs::remove_file(d.join(format!("ort{i}.f32")));
    }
    let ok = Command::new(py)
        .args(["-I", "-c", ORT, &m])
        .arg(d)
        .args(&args)
        .status()
        .unwrap();
    assert!(ok.success(), "onnxruntime");
    (0..16)
        .map_while(|i| std::fs::read(d.join(format!("ort{i}.f32"))).ok())
        .map(|b| bytemuck::cast_slice::<u8, f32>(&b).to_vec())
        .collect()
}

/// Largest absolute difference and the reference's largest magnitude.
fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold((0f32, 0f32), |(d, m), (x, y)| {
        (d.max((x - y).abs()), m.max(y.abs()))
    })
}

/// Skips a test that loads Core ML in this process unless `CFFIXED_USER_HOME` is set: Core ML's ANE cache follows it,
/// not HOME, so it would land in the real `~/Library/Caches`.
#[cfg(target_os = "macos")]
fn no_cf_home() -> bool {
    let unset = cfg!(target_os = "macos") && std::env::var_os("CFFIXED_USER_HOME").is_none();
    if unset {
        eprintln!("CFFIXED_USER_HOME unset (Core ML's cache would go to the real home): skipped");
    }
    unset
}

/// A device opened like `video::Gpu`: every adapter feature (burn needs them), the experimental ones. From the test
/// binary's one instance: instances created on parallel test threads race in NVIDIA's ICD (`video::gpu` `SCAN`).
fn open() -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    static I: OnceLock<(wgpu::Instance, wgpu::Adapter)> = OnceLock::new();
    let (inst, ad) = I.get_or_init(|| {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let ad = pollster::block_on(inst.request_adapter(&Default::default())).unwrap();
        (inst, ad)
    });
    let (dev, q) = pollster::block_on(
        ad.request_device(&wgpu::DeviceDescriptor {
            required_features: ad
                .features()
                .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            required_limits: ad.limits(),
            // SAFETY: cubecl may run passthrough shaders on this device, as on its own
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }),
    )
    .unwrap();
    (inst.clone(), ad.clone(), dev, q)
}

/// The app's start-up order (`stitch::attach`, then its own uncaptured-error handler), then two `Ai`s: both reuse
/// the attachment (one burn runtime, no new cubecl handler), so a provoked validation error (a buffer mapped at
/// creation with an unaligned size) still reaches the app's handler instead of cubecl's, which panics.
#[test]
fn attach() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    };
    let (inst, ad, dev, q) = open();
    let bd = stitch::attach(&inst, &ad, &dev, &q).unwrap();
    let hits = Arc::new(AtomicUsize::new(0));
    let h = hits.clone();
    dev.on_uncaptured_error(Arc::new(move |_: wgpu::Error| {
        h.fetch_add(1, SeqCst);
    }));
    let (a, b) = (
        Ai::new(&inst, &ad, &dev, &q).unwrap(),
        Ai::new(&inst, &ad, &dev, &q).unwrap(),
    );
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 3,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        })
    }));
    assert!(r.is_ok(), "cubecl's handler replaced the app's");
    assert_eq!(hits.load(SeqCst), 1, "the app's handler sees the error");
    assert!(a.dev == bd && b.dev == bd, "Ai on the attached burn device");
}

#[test]
fn vs_onnxruntime() {
    let (Some(smp), Some(py)) = (std::env::var_os("X5_SAMPLES"), std::env::var("X5_ORT").ok())
    else {
        eprintln!("X5_SAMPLES or X5_ORT unset: skipped");
        return;
    };
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join("x5-track-ort");
    std::fs::create_dir_all(&d).unwrap();
    // a lens picture of v627 stretched to 2:1 stands in for an equirect: real content is what matters here
    let (w, h) = (3072u32, 1536u32);
    let raw = d.join("pano.rgba");
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(PathBuf::from(smp).join("v627.insv"))
        .args([
            "-map",
            "0:v:1",
            "-vf",
            &format!("select=eq(n\\,150),scale={w}:{h}"),
            "-frames:v",
            "1",
        ])
        .args(["-f", "rawvideo", "-pix_fmt", "rgba"])
        .arg(&raw)
        .status()
        .unwrap();
    assert!(ok.success());
    let (inst, ad, dev, q) = open();
    let tex = dev.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    q.write_texture(
        tex.as_image_copy(),
        &std::fs::read(&raw).unwrap(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: None,
        },
        tex.size(),
    );
    let ai = Ai::new(&inst, &ad, &dev, &q).unwrap();
    let p = ai.pano(&tex.create_view(&Default::default()));

    // VitTrack: template and search crops around the same point, slightly shifted
    let tc = Cam::new(dir(0.0, -10.0), Vec3::Z, Vec2::splat(0.15));
    let sc = Cam::new(dir(2.0, -9.0), Vec3::Z, Vec2::splat(0.3));
    let (tpl, sea) = (
        In::new(&ai.dev, [1, 3, 128, 128]).unwrap(),
        In::new(&ai.dev, [1, 3, 256, 256]).unwrap(),
    );
    let cr = Crops::new(&dev, &ai.g[1], 1, 3 * 256 * 256);
    let _ = ai.crop(
        &cr,
        &p,
        &[Job::new(&tc, [128; 2], 0, &vit::NORM, p.w)],
        &tpl,
    );
    let _ = ai.crop(
        &cr,
        &p,
        &[Job::new(&sc, [256; 2], 0, &vit::NORM, p.w)],
        &sea,
    );
    let (ti, si) = (read(tpl.t.clone()), read(sea.t.clone()));
    let (a, b, c) = ai.vit.forward(tpl.t.clone(), sea.t.clone());
    let r = ort(
        &py,
        "object_tracking_vittrack_2023sep.onnx",
        &d,
        &[
            ("template", &ti, &[1, 3, 128, 128]),
            ("search", &si, &[1, 3, 256, 256]),
        ],
    );
    for (k, t) in [a, b, c].into_iter().enumerate() {
        let (e, m) = diff(&read(t), &r[k]);
        eprintln!("VitTrack output {k}: max |burn − ort| {e:.2e} (max |ort| {m:.3})");
        assert!(e < 1e-3 * m.max(1.0), "VitTrack output {k}");
    }

    // YOLOX: two tiles in one batch against one run each
    let tiles = crate::ring(2, 100.0, -15.0);
    let s = SIDE as usize;
    let jobs: Vec<Job> = tiles
        .iter()
        .enumerate()
        .map(|(i, c)| Job::new(c, [SIDE; 2], i * 3 * s * s, &det::NORM, p.w))
        .collect();
    let _ = ai.crop(&ai.dc, &p, &jobs, &ai.di);
    let x = ai.di.t.clone().narrow(0, 0, 2);
    let o = read(ai.det.forward(x.clone()));
    let xi = read(x);
    let n = 3549 * 85;
    for t in 0..2 {
        let r = ort(
            &py,
            "yolox_nano.onnx",
            &d,
            &[("images", &xi[t * 3 * s * s..][..3 * s * s], &[1, 3, s, s])],
        );
        let (e, m) = diff(&o[t * n..][..n], &r[0]);
        let hits = r[0].chunks(85).filter(|a| a[4] > 0.3).count();
        eprintln!(
            "YOLOX tile {t}: max |burn − ort| {e:.2e} (max |ort| {m:.1}, {hits} anchors with objectness > 0.3)"
        );
        assert!(e < 1e-3 * m.max(1.0), "YOLOX tile {t}");
    }

    // YuNet: two face tiles in one batch against one run each, all 12 outputs
    let mut ai = ai;
    let nets = ai.nets().unwrap();
    let s = privacy::FS as usize;
    let tiles = crate::ring(2, 75.0, 0.0);
    let jobs: Vec<Job> = tiles
        .iter()
        .enumerate()
        .map(|(i, c)| Job::new(c, [privacy::FS; 2], i * 3 * s * s, &privacy::NORM, p.w))
        .collect();
    let _ = ai.crop(&nets.fc, &p, &jobs, &nets.fi);
    let x = nets.fi.t.clone().narrow(0, 0, 2);
    let o = nets.face.forward(x.clone());
    let o = [o.0, o.1, o.2, o.3, o.4, o.5, o.6, o.7, o.8, o.9, o.10, o.11].map(read);
    let xi = read(x);
    for t in 0..2 {
        let r = ort(
            &py,
            "face_detection_yunet_2023mar.onnx",
            &d,
            &[("input", &xi[t * 3 * s * s..][..3 * s * s], &[1, 3, s, s])],
        );
        assert_eq!(r.len(), 12);
        let (mut e, mut m) = (0f32, 0f32);
        for (a, b) in o.iter().zip(&r) {
            let (x, y) = diff(&a[t * b.len()..][..b.len()], b);
            (e, m) = (e.max(x), m.max(y));
        }
        let faces = r[0]
            .iter()
            .zip(&r[3])
            .filter(|(c, o)| (*c * *o).sqrt() > 0.6)
            .count();
        eprintln!(
            "YuNet tile {t}: max |burn − ort| {e:.2e} over 12 outputs (max |ort| {m:.1}, {faces} stride-8 cells > 0.6)"
        );
        assert!(e < 1e-3 * m.max(1.0), "YuNet tile {t}");
    }

    // plates: a 45° crop and the 16° crop of a grid over the frame the detector is surest of (scores near a decision
    // compared too) in one batch against one run each
    let nets = ai.plate_nets().unwrap();
    let s = plates::PS as usize;
    let a = (s / 8).pow(2) + (s / 16).pow(2) + (s / 32).pow(2);
    let run = |ai: &Ai, c: &[Cam]| {
        let jobs: Vec<Job> = (c.iter().enumerate())
            .map(|(i, c)| Job::new(c, [plates::PS; 2], i * 3 * s * s, &plates::NORM, p.w))
            .collect();
        let _ = ai.crop(&nets.pc, &p, &jobs, &nets.pi);
        nets.pi.t.clone().narrow(0, 0, c.len())
    };
    let grid: Vec<Cam> = (0..420)
        .map(|i| {
            let d = dir((i / 7) as f32 * 6.0, (i % 7) as f32 * 10.0 - 30.0);
            Cam::new(d, Vec3::Z, Vec2::splat(0.14))
        })
        .collect();
    let mut best = (0f32, grid[0]);
    for ch in grid.chunks(8) {
        let v = read(nets.net.forward(run(&ai, ch)));
        for (i, c) in ch.iter().enumerate() {
            let m = v[i * 6 * a..][..6 * a]
                .chunks(6)
                .fold(0f32, |m, r| m.max(r[4] * r[5]));
            if m > best.0 {
                best = (m, *c);
            }
        }
    }
    let x = run(
        &ai,
        &[
            Cam::new(dir(90.0, -5.0), Vec3::Z, Vec2::splat(0.41)),
            best.1,
        ],
    );
    let o = read(nets.net.forward(x.clone()));
    let xi = read(x);
    for t in 0..2 {
        let r = ort(
            &py,
            "plates-yolox-nano-256.onnx",
            &d,
            &[("images", &xi[t * 3 * s * s..][..3 * s * s], &[1, 3, s, s])],
        );
        // box terms (grid cells, log sizes) and probabilities (0..1) apart: the boxes' scale would hide an error
        let o = &o[t * 6 * a..][..6 * a];
        let part = |v: &[f32], c: std::ops::Range<usize>| -> Vec<f32> {
            v.chunks(6).flat_map(|r| r[c.clone()].to_vec()).collect()
        };
        let (e, m) = diff(&part(o, 0..4), &part(&r[0], 0..4));
        let (es, ms) = diff(&part(o, 4..6), &part(&r[0], 4..6));
        eprintln!(
            "Plates crop {t}: max |burn − ort| box terms {e:.2e} (max |ort| {m:.1}), probabilities {es:.2e} (max \
             {ms:.3})"
        );
        assert!(e < 1e-3 * m.max(1.0) && es < 1e-4, "plates crop {t}");
    }
}

/// LOMO on synthetic crops (a figure 24 px wide, red over blue, in the middle of the box, on grey, a noise texture):
/// unit length; the figure in half the light is the same, 5 px aside more alike than one with its colours swapped
/// (whose texture, two thirds of the feature, is the same).
#[test]
fn lomo() {
    let (w, h) = (reid::W as usize, reid::H as usize);
    let mut s = 1u32;
    let noise: Vec<f32> = (0..w * h)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1 << 24) as f32
        })
        .collect();
    let fig = |top: [f32; 3], bot: [f32; 3], x0: usize, k: f32| {
        let mut px = vec![0f32; 3 * w * h];
        for (i, v) in px.iter_mut().enumerate() {
            let (c, y, x) = (i / (w * h), i / w % h, i % w);
            let col = match (x0..x0 + 24).contains(&x) {
                false => [0.5; 3],
                true if y < h / 2 => top,
                _ => bot,
            };
            *v = k * col[c] * (0.8 + 0.2 * noise[i % (w * h)]);
        }
        let mut e = [0.0; reid::DIM];
        reid::lomo(&px, &mut e);
        e
    };
    let (r, b) = ([0.8, 0.1, 0.1], [0.1, 0.2, 0.7]);
    let a = fig(r, b, 36, 1.0);
    let s = |e: reid::Emb| reid::sim(&a, &e);
    let (one, dark, aside, other) = (
        s(a),
        s(fig(r, b, 36, 0.5)),
        s(fig(r, b, 41, 1.0)),
        s(fig(b, r, 36, 1.0)),
    );
    eprintln!("length² {one:.5}, darker {dark:.4}, aside {aside:.3}, colours swapped {other:.3}");
    assert!((one - 1.0).abs() < 1e-4 && dark > 0.999 && aside > 0.95 && other < aside - 0.2);
}

#[test]
fn geometry() {
    use crate::{iou, yaw_pitch};
    for (y, p) in [(0.0, 0.0), (37.0, -20.0), (-170.0, 60.0), (90.0, -89.0)] {
        let (d, c) = (dir(y, p), Cam::view(y, p, 0.0, 90.0, 16.0 / 9.0));
        assert!(c.f.abs_diff_eq(d, 1e-5), "view axis {y} {p}");
        let [yy, pp] = yaw_pitch(d);
        assert!(
            (yy - y).abs() < 1e-3 && (pp - p).abs() < 1e-3,
            "yaw, pitch {y} {p}"
        );
        // no roll: right stays level, up points up
        assert!(c.r.z.abs() < 1e-5 && c.u.z >= 0.0);
        let q = Vec2::new(0.3, -0.7);
        assert!(c.px(c.ray(q)).unwrap().abs_diff_eq(q, 1e-4));
    }
    // a box (half the crop wide, centred) is 2·atan(0.25) wide; overlap of shifted copies
    let c = Cam::new(dir(10.0, 5.0), Vec3::Z, Vec2::splat(0.5));
    let (d, s) = c.boxed([-0.5, -0.5, 0.5, 0.5]);
    assert!((s.x - 2.0 * 0.25f32.atan().to_degrees()).abs() < 1e-3);
    assert!((iou((d, s), (d, s), false) - 1.0).abs() < 1e-5);
    let b = c.boxed([0.0, -0.5, 1.0, 0.5]);
    assert!(
        (iou((d, s), b, false) - 1.0 / 3.0).abs() < 0.02,
        "{}",
        iou((d, s), b, false)
    );
    assert!((iou((d, s), b, true) - 0.5).abs() < 0.02);
}

/// The speed the subject needs to be a box (`vit::Seen::speed`, heights a second at 30 fps) and its prior
/// (`vit::Seen::prior`). The yaw-21 pick of v627 under `X5_BLANK=150:60`: 8 frames after the tracker's last box of the
/// subject (22.7° tall) `find` took someone 27° away and a third as tall, which the reach it had then (the subject's
/// height times 1 + 0.1 a missed frame, 38° there) let through. The v635 walking man (`X5_TRACK=v635:-134:-33`): 9
/// frames after he was last seen for sure (26.7° tall, heading up and right at 8.85° a second) two look-alikes 27° and
/// 37° away, less than half as tall, needed 2.9 and 2.4 heights a second, under a sprint's 3: in a third of a second
/// nobody gets that fast. A long gap, a half-hidden subject, walking pace, the heading, a jog.
#[test]
fn speed() {
    let seen = |y, p, h: f32, w: Vec3, k| vit::Seen {
        d: dir(y, p),
        h: 2.0 * (h / 2.0).to_radians().tan(),
        w,
        k,
    };
    let b = |y, p, h: f32| (dir(y, p), Vec2::new(0.4 * h, h));
    let a = seen(16.0, -14.2, 22.7, Vec3::ZERO, 295);
    let (v, q) = (|x, k| a.speed(x, k, 30.0), |x, k| a.prior(x, k, 30.0));
    let far = b(42.9, -4.9, 6.3);
    eprintln!("far {:.2}, 10 s later {:.3}", v(far, 303), v(far, 595));
    assert!(q(far, 303).is_none(), "someone else");
    assert!(q(far, 595).unwrap() < 0.01, "after a long gap anyone");
    assert_eq!(q(b(16.0, -14.2, 22.7), 296), Some(0.0));
    assert!(v(b(16.0, -18.0, 11.3), 310) < 0.15, "half hidden");
    let (d0, d1) = (dir(-116.7, -27.6), dir(-115.0, -26.4));
    let w = (d1 - d0).normalize() * (8.85f32 / 30.0).to_radians();
    let a = seen(-116.7, -27.6, 26.7, w - w.dot(d0) * d0, 45);
    let x = [
        (dir(-89.47, -12.21), Vec2::new(3.89, 9.32)),
        (dir(-80.83, -11.95), Vec2::new(4.75, 14.16)),
        (dir(-115.86, -24.5), Vec2::new(10.09, 23.57)),
    ];
    let u = x.map(|x| a.speed(x, 54, 30.0));
    eprintln!("v635 frame 54: {u:.3?}");
    assert!(u[0] < 3.0 && u[1] < 3.0 && u[2] == 0.0);
    assert!(x[..2].iter().all(|&x| a.prior(x, 54, 30.0).is_none()));
    assert_eq!(a.prior(x[2], 54, 30.0), Some(0.0));
    // a person 10° tall (5.7 of its heights away) walks a height sideways in a second: about half a height a second
    // for the box centres' slack and jitter
    let a = seen(0.0, 0.0, 10.0, Vec3::ZERO, 0);
    let walk = a.speed(b(10.0, 0.0, 10.0), 30, 30.0);
    // heading towards yaw +10° a second (+Y at yaw 0): there after a second for nothing, the other way twice the walk
    let h = seen(0.0, 0.0, 10.0, Vec3::Y * (10f32 / 30.0).to_radians(), 0);
    let (ahead, back) = (
        h.speed(b(10.0, 0.0, 10.0), 30, 30.0),
        h.speed(b(-10.0, 0.0, 10.0), 30, 30.0),
    );
    eprintln!("walk {walk:.3}, heading: ahead {ahead:.3}, back {back:.3}");
    assert!((0.45..0.75).contains(&walk) && ahead == 0.0 && back > 1.8 * walk);
    // a walk's pace costs little, more the faster: twice the distance in the second, four times as much and more
    let (q1, j) = (
        a.prior(b(10.0, 0.0, 10.0), 30, 30.0).unwrap(),
        b(25.0, 0.0, 10.0),
    );
    let (q2, s2) = (a.prior(j, 30, 30.0).unwrap(), a.speed(j, 30, 30.0));
    eprintln!("walk costs {q1:.3}, {s2:.3} heights a second {q2:.3}");
    assert!(q1 < 0.06 && s2 > 2.0 * walk && (q2 / q1 - (s2 / walk).powi(2)).abs() < 1e-3);
}

/// `find`'s rule for persons (`vit::Tried`, `vit::hit`) on features of P (0.7 like the subject) and Q (0.68 like it,
/// 0.48 like P), both the subject by the gallery. P, taken at VitTrack 0.34 (the v635 man's at frame 115) and lost
/// before VitTrack was sure of them, is tried: one at least as like P as like the subject needs 0.5 then, Q still
/// 0.25, until P is one of more than `TRIED` ago. A candidate the pick's template is unsure of or puts its box beside
/// is not taken at all, never on its detection's box.
#[test]
fn tried() {
    use crate::{Det, Gallery, tan};
    let f = |c: &[(usize, f32)]| {
        let mut e = [0f32; reid::DIM];
        for &(i, v) in c {
            e[i] = v;
        }
        e
    };
    let (s, p, q) = (
        f(&[(0, 1.0)]),
        f(&[(0, 0.7), (1, 0.51f32.sqrt())]),
        f(&[(0, 0.68), (2, (1.0 - 0.68f32.powi(2)).sqrt())]),
    );
    let mut g = Gallery::default();
    g.add(s, true);
    for i in 3..6 {
        g.add(f(&[(i, 1.0)]), false);
    }
    let (sp, sq) = (g.score(&p), g.score(&q));
    assert!(g.is(sp) && g.is(sq));
    // P's detection; the template's box on it, beside it
    let x = Det {
        dir: dir(10.0, 0.0),
        size: Vec2::new(4.0, 10.0),
        cls: 0,
        score: 0.9,
    };
    let (on, off, z) = (dir(10.2, 0.3), dir(16.0, 0.0), Vec2::new(4.2, 10.4));
    let mut t = vit::Tried::default();
    let lo = t.need(&g, &p, sp.0);
    let h = vit::hit(x, (0.34, on, tan(z)), lo).expect("P at 0.34, nobody tried");
    assert!(lo == 0.25 && h.dir == on && h.score == 0.34 && h.size.abs_diff_eq(z, 1e-4));
    // `find` took P; the tracker lost them before VitTrack was sure of them (no `Tracked` frame cleared `took`)
    t.took = Some(Box::new(p));
    t.lost();
    assert!(t.took.is_none() && t.v.len() == 1);
    let lo = t.need(&g, &p, sp.0);
    assert_eq!(lo, 0.5, "P tried");
    assert!(vit::hit(x, (0.34, on, tan(z)), lo).is_none());
    assert!(vit::hit(x, (0.5, on, tan(z)), lo).is_some());
    assert_eq!(t.need(&g, &q, sq.0), 0.25, "Q less like P than the subject");
    // the last `TRIED` (8) are kept
    for i in 0..8 {
        t.took = Some(Box::new(f(&[(10 + i, 1.0)])));
        t.lost();
    }
    assert!(t.v.len() == 8 && t.v[0][10] == 1.0 && t.need(&g, &p, sp.0) == 0.25);
    // the template unsure, or its box beside the detection: no box (the detection's was taken before)
    assert!(vit::hit(x, (0.2, on, tan(z)), 0.25).is_none());
    assert!(vit::hit(x, (0.9, off, tan(z)), 0.25).is_none());
}

/// Synthetic detections, all 15° below the horizon: A (a person, 20° tall) walks from yaw 0 to 60° and leaves after
/// 6 s; B (a person, 10° tall) stands at yaw −132° (168° on from A's end); C (a chair, larger) at yaw 90°. The path
/// follows A without jumping, then turns to B once A has been gone a while, level: the great-circle arc from A to B
/// dips to 69° below the horizon.
#[test]
fn framing() {
    use crate::{Det, Framer, yaw_pitch};
    let (fps, n) = (30.0, 300);
    let det = |y: f32, h: f32, cls| Det {
        dir: dir(y, -15.0),
        size: Vec2::new(h / 3.0, h),
        cls,
        score: 0.8,
    };
    let mut f = Framer::new(16.0 / 9.0);
    for k in (0..n).step_by(10) {
        let t = k as f32 / fps;
        let mut d = vec![det(-132.0, 10.0, 0), det(90.0, 30.0, 56)];
        if t < 6.0 {
            d.push(det(10.0 * t, 20.0, 0));
        }
        f.push(k, d);
    }
    let p = f.path(n, fps);
    for k in [60, 120, 150] {
        let y = yaw_pitch(p[k].dir)[0];
        assert!(
            (y - 10.0 * k as f32 / fps).abs() < 4.0,
            "frame {k}: yaw {y}"
        );
    }
    let y = yaw_pitch(p[n - 1].dir)[0];
    assert!((y + 132.0).abs() < 2.0, "the end frames B: yaw {y}");
    let low = p.iter().map(|s| yaw_pitch(s.dir)[1]).fold(0f32, f32::min);
    assert!(low > -15.5, "the path stays level: lowest pitch {low}");
    let step = p
        .windows(2)
        .map(|w| w[0].dir.angle_between(w[1].dir).to_degrees())
        .fold(0f32, f32::max);
    assert!(step < 6.0, "largest turn between frames {step}°");
    let fov = p[90].fov;
    assert!(
        (60.0..80.0).contains(&fov),
        "A (20°) fills 45 % of the height: fov {fov}"
    );

    // D, a person 32° tall crouching close at pitch −49 (v627's end): the widest view, raised until its lower edge
    // clears the nadir zone, D still in it
    let mut f = Framer::new(16.0 / 9.0);
    let d = Det {
        dir: dir(-90.0, -49.0),
        size: Vec2::new(16.0, 32.0),
        cls: 0,
        score: 0.8,
    };
    for k in (0..n).step_by(10) {
        f.push(k, vec![d]);
    }
    for s in f.path(n, fps) {
        let [y, p] = yaw_pitch(s.dir);
        let v = ((s.fov / 2.0).to_radians().tan() * 9.0 / 16.0)
            .atan()
            .to_degrees();
        assert!(p - v >= crate::LOW - 0.01, "lower edge {}", p - v);
        let c = Cam::view(y, p, 0.0, s.fov, 16.0 / 9.0).px(d.dir).unwrap();
        assert!(c.abs().max_element() < 1.0, "D at {c} in the view");
    }

    // with the direction of travel (turning from yaw −30° at 2°/s): no subject for 3 s, then a person at yaw 90°
    // until 6 s: the view looks ahead, frames the person, and looks ahead again once the person has been gone a while
    let mut f = Framer::new(16.0 / 9.0);
    let ahead = |k: usize| -30.0 + 2.0 * k as f32 / fps;
    f.ahead = (0..n).map(|k| dir(ahead(k), 0.0)).collect();
    for k in (0..n).step_by(10) {
        let t = k as f32 / fps;
        f.push(
            k,
            match (3.0..6.0).contains(&t) {
                true => vec![det(90.0, 20.0, 0)],
                false => vec![],
            },
        );
    }
    let p = f.path(n, fps);
    for (k, want, sub) in [
        (30, ahead(30), None),
        (135, 90.0, Some(0)),
        (270, ahead(270), None),
    ] {
        let y = yaw_pitch(p[k].dir)[0];
        assert!((y - want).abs() < 3.0, "frame {k}: yaw {y}, want {want}");
        assert_eq!(p[k].subject, sub, "frame {k}");
    }
}

/// Every direction up to 2° from the poles lies well inside a face tile (|x|, |y| ≤ 0.85), in one where a face upright
/// to the world stands within 30° of upright.
#[test]
fn face_tiles() {
    let t = crate::face_tiles();
    for p in (-44..=44).map(|i| i as f32 * 2.0) {
        for y in (0..72).map(|i| i as f32 * 5.0) {
            let w = dir(y, p);
            let up = (Vec3::Z - w.z * w).normalize();
            let ok = t.iter().any(|c| {
                c.px(w).zip(c.px(w + 1e-3 * up)).is_some_and(|(a, b)| {
                    let d = b - a;
                    a.abs().max_element() <= 0.85 && d.x.atan2(-d.y).abs() <= 30f32.to_radians()
                })
            });
            assert!(ok, "yaw {y} pitch {p}");
        }
    }
}

/// A face walking 30° in 3 s, detected on every 3rd frame but for a gap of 0.6 s, a second one seen once: the
/// regions cover all of the first (its box's ellipse) where it is on every frame from 0.25 s before its first to 0.25 s
/// after its last detection, walking on before and after; the second gets its own short track.
#[test]
fn hide() {
    use crate::{Hide, Region};
    let (fps, n) = (30.0, 120);
    let face = |k: usize| Region {
        dir: dir(10.0 * k as f32 / fps, -5.0),
        size: Vec2::new(3.0, 4.0),
        score: 0.9,
    };
    let mut h = Hide::new();
    for k in (9..90).step_by(3) {
        let mut d = vec![];
        if !(40..58).contains(&k) {
            d.push(face(k));
        }
        if k == 60 {
            d.push(Region {
                dir: dir(-90.0, 0.0),
                ..face(0)
            });
        }
        h.push(k, d);
    }
    let t = h.tracks(fps);
    assert_eq!(t.len(), 2, "tracks: {}", t.len());
    let r = h.regions(n, fps);
    let pad = (0.25f32 * fps).ceil() as usize;
    for (k, r) in r.iter().enumerate() {
        let on = |w: Vec3| {
            r.iter().any(|x| {
                let e = (x.size / 2.0).to_array().map(|a| a.to_radians().tan());
                Cam::new(x.dir, Vec3::Z, Vec2::new(e[0], e[1]))
                    .px(w)
                    .is_some_and(|p| p.length() <= 1.0)
            })
        };
        let f = face(k);
        let e = (f.size / 2.0).to_array().map(|a| a.to_radians().tan());
        let c = Cam::new(f.dir, Vec3::Z, Vec2::new(e[0], e[1]));
        let all =
            (0..8).all(|i| on(c.ray(Vec2::from_angle(i as f32 * std::f32::consts::FRAC_PI_4))));
        let want = (9 - pad..=87 + pad).contains(&k);
        assert_eq!(on(f.dir), want, "frame {k}");
        assert!(all || !want, "frame {k}: the face's edge uncovered");
        assert_eq!(
            on(dir(-90.0, 0.0)),
            (60 - pad..=60 + pad).contains(&k),
            "frame {k}: the other"
        );
    }
}

/// Plate decoding: per anchor x, y offsets and log width, height in grid cells, objectness and plate probability, the
/// grids of strides 8, 16, 32 in turn; boxes whose product is over the threshold, non-maximum suppression at IoU 0.45
/// (two boxes of one plate are one, a box beside it stays), and a box's place on the sphere through its crop.
#[test]
fn plate_decode() {
    let s = plates::PS as usize;
    let first = [0, (s / 8).pow(2), (s / 8).pow(2) + (s / 16).pow(2)];
    // (stride, cell x, y, offset x, y, w, h in pixels, objectness, probability): a plate at (100, 200), the same plate
    // 4 px right and 1 px down, another one at (200, 120), one at (50, 50) under the threshold
    let rows = [
        (8usize, 12, 25, 0.5, 0.0, 60.0, 14.0, 1.0, 0.9),
        (8, 13, 25, 0.0, 0.125, 60.0, 14.0, 0.75, 0.8),
        (16, 12, 7, 0.5, 0.5, 40.0, 10.0, 0.5, 0.6),
        (32, 1, 1, 0.5625, 0.5625, 30.0, 8.0, 0.5, 0.2),
    ];
    let mut v = vec![0f32; 6 * (first[2] + (s / 32).pow(2))];
    for (st, x, y, dx, dy, w, h, o, p) in rows {
        let a = first[st.trailing_zeros() as usize - 3] + y * (s / st) + x;
        let st = st as f32;
        v[a * 6..][..6].copy_from_slice(&[dx, dy, (w / st).ln(), (h / st).ln(), o, p]);
    }
    let b = plates::boxes(&v, s, 0.25);
    let off = |x: [f32; 4], y: [f32; 4]| (0..4).fold(0f32, |m, i| m.max((x[i] - y[i]).abs()));
    assert_eq!(b.len(), 3, "{b:?}");
    assert!(
        off(b[0].0, [70.0, 193.0, 130.0, 207.0]) < 1e-3 && b[0].1 == 0.9,
        "{b:?}"
    );
    assert!(
        off(b[2].0, [180.0, 115.0, 220.0, 125.0]) < 1e-3 && b[2].1 == 0.3,
        "{b:?}"
    );
    let k = privacy::nms(b, 0.45);
    assert_eq!(
        k.iter().map(|x| x.1).collect::<Vec<_>>(),
        [0.9, 0.3],
        "{k:?}"
    );
    // a box a quarter of a 20° crop wide and a sixteenth high, in its middle: a quarter, a sixteenth of the crop's
    // tangent-plane width, on its axis
    let s = plates::PS as f32;
    let c = Cam::new(
        dir(40.0, -10.0),
        Vec3::Z,
        Vec2::splat(10f32.to_radians().tan()),
    );
    let (edge, r) = privacy::place(
        &c,
        [
            s / 2.0 - s / 8.0,
            s / 2.0 - s / 32.0,
            s / 2.0 + s / 8.0,
            s / 2.0 + s / 32.0,
        ],
        s,
        0.7,
    );
    let w = 2.0 * (0.25 * 10f32.to_radians().tan()).atan().to_degrees();
    let h = 2.0 * (0.0625 * 10f32.to_radians().tan()).atan().to_degrees();
    assert!(!edge && r.dir.angle_between(c.f) < 1e-4, "{r:?}");
    assert!(
        (r.size - Vec2::new(w, h)).abs().max_element() < 1e-3,
        "{r:?} against {w} × {h}"
    );
}

/// Crops around vehicles: other classes left out, one crop per object (car and truck boxes of one van), the surest
/// first and at most `CMAX`; each centred on its vehicle and upright, its side the vehicle's larger one times `GROW`
/// within `FMIN`..`FMAX`, so the whole vehicle is in it.
#[test]
fn plate_crops() {
    use crate::{Det, VEHICLES, plate_crops};
    let d = |cls, y: f32, w: f32, h: f32, score| Det {
        dir: dir(y, -3.0),
        size: Vec2::new(w, h),
        cls,
        score,
    };
    let v = [
        d(2, 30.0, 12.0, 5.0, 0.8),
        d(7, 30.2, 12.5, 5.2, 0.5),
        d(0, 60.0, 3.0, 9.0, 0.9),
        d(5, -90.0, 120.0, 40.0, 0.7),
        d(3, 150.0, 2.0, 2.5, 0.6),
    ];
    let c = plate_crops(&v);
    assert_eq!(c.len(), 3, "{c:?}");
    let side = |c: &Cam| 2.0 * c.ext.x.atan().to_degrees();
    for (c, x, want) in [
        (&c[0], &v[0], 12.0 * plates::GROW),
        (&c[1], &v[3], plates::FMAX),
        (&c[2], &v[4], plates::FMIN),
    ] {
        assert!(c.f.angle_between(x.dir) < 1e-4 && c.u.z > 0.0 && (c.ext.x - c.ext.y).abs() < 1e-6);
        assert!((side(c) - want).abs() < 1e-3, "{} against {want}", side(c));
        if want > x.size.max_element() {
            let t = (x.size / 2.0).to_array().map(|a| a.to_radians().tan());
            let b = Cam::new(x.dir, Vec3::Z, Vec2::from(t));
            for p in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let q = c.px(b.ray(Vec2::new(p.0, p.1))).unwrap();
                assert!(q.abs().max_element() < 1.0, "vehicle corner at {q}");
            }
        }
    }
    let many: Vec<Det> = (0..40)
        .map(|i| d(VEHICLES[i % 4], i as f32 * 9.0, 4.0, 3.0, i as f32 / 40.0))
        .collect();
    let c = plate_crops(&many);
    assert_eq!(c.len(), plates::CMAX);
    assert!(
        c[0].f.angle_between(many[39].dir) < 1e-4,
        "the surest first"
    );
}

/// Synthetic signals over 30 s: people from 10 to 14 s, loud from 20 to 23 s, the camera turning from 4 to 6 s.
fn scorer() -> crate::Scorer {
    use crate::{Det, Scorer, Sig};
    let mut s = Scorer::new(30.0);
    for i in 0..300 {
        let t = i as f64 / 10.0;
        s.push(
            Sig::Motion,
            t,
            if (4.0..6.0).contains(&t) { 1.5 } else { 0.05 },
        );
        s.push(
            Sig::Audio,
            t,
            if (20.0..23.0).contains(&t) {
                -12.0
            } else {
                -40.0
            },
        );
        let p = Det {
            dir: dir(30.0, -10.0),
            size: Vec2::new(6.0, 15.0),
            cls: 0,
            score: 0.9,
        };
        let p = [p];
        s.people(
            t,
            if (10.0..14.0).contains(&t) {
                &p[..]
            } else {
                &[]
            },
            &[],
        );
    }
    s
}

/// The picks of `scorer` (12 s) lie on its events, fill the budget within a bin plus `MIN`, are 2 to 8 s, in order
/// and apart, each inside its scored span, the spans apart; their timeline passes crate edit's checks with view keys
/// from the framing path.
#[test]
fn highlights() {
    use crate::{Shot, timeline};
    let h = scorer().pick(12.0);
    let sum: f64 = h.iter().map(|x| x.b - x.a).sum();
    assert!(
        (12.0..=12.0 + crate::BIN + 2.0).contains(&sum),
        "total {sum}: {h:?}"
    );
    for x in &h {
        assert!(x.b - x.a >= 2.0 - 1e-9 && x.b - x.a <= 8.0 + 1e-9, "{x:?}");
    }
    assert!(
        h.windows(2)
            .all(|w| w[1].a >= w[0].b + 1.0 && w[1].lo >= w[0].hi)
    );
    assert!(
        h.iter()
            .all(|x| x.lo <= x.a && x.hi >= x.b && x.lo >= 0.0 && x.hi <= 30.0)
    );
    for c in [12.0, 21.5, 5.0] {
        assert!(
            h.iter().any(|x| x.a <= c && c <= x.b),
            "{c} s not picked: {h:?}"
        );
    }
    let path: Vec<Shot> = (0..900)
        .map(|k| Shot {
            dir: dir(k as f32 * 0.2, -5.0),
            fov: 80.0,
            subject: None,
        })
        .collect();
    let fps = edit::Rate { num: 30, den: 1 };
    let tl = timeline("a.insv", 30.0, fps, 30.0, &h, &path).unwrap();
    assert_eq!(tl.clips.len(), h.len());
    assert!((tl.dur() - sum).abs() < 0.05);
    let p = edit::Project::with(tl).unwrap();
    // the view at a highlight's middle looks where the path does
    let (k, x) = (1, h[1]);
    let n = p.tl().frames(k);
    let e = p.tl().frame((n.start + n.end) / 2).unwrap();
    let [y, _, _] = e.view.ypr();
    let want = (e.src * 30.0) as f32 * 0.2;
    assert!((y - want).abs() < 2.0, "{x:?}: yaw {y}, path {want}");
}

/// Cuts onto beats: `scorer`'s picks (12 s) fitted to beats at 70, 120 and 180 BPM over 40 s of music (every second
/// beat strong, times jittered ±15 ms) at 30 and 29.97 fps. Every cut of the timeline (each clip's end) lies within a
/// frame of a beat; every clip inside its highlight's scored span and at least 1 s long; the total within a beat period
/// per clip of the picks'; the timeline with the music on passes crate edit's checks.
#[test]
fn beats() {
    use crate::{Shot, fit, timeline};
    let h0 = scorer().pick(12.0);
    let sum: f64 = h0.iter().map(|x| x.b - x.a).sum();
    let path: Vec<Shot> = (0..900)
        .map(|k| Shot {
            dir: dir(k as f32 * 0.2, -5.0),
            fov: 80.0,
            subject: None,
        })
        .collect();
    let mut r = 1u32;
    for fps in [
        edit::Rate { num: 30, den: 1 },
        edit::Rate {
            num: 30000,
            den: 1001,
        },
    ] {
        for bpm in [70.0, 120.0, 180.0] {
            let p = 60.0 / bpm;
            let bt: Vec<(f64, f32)> = (0..(40.0 / p) as usize)
                .map(|i| {
                    r = r.wrapping_mul(1664525).wrapping_add(1013904223);
                    let j = 0.015 * ((r >> 8) as f64 / (1 << 23) as f64 - 1.0);
                    (0.21 + i as f64 * p + j, if i % 2 == 0 { 2.0 } else { 0.5 })
                })
                .collect();
            let mut h = h0.clone();
            fit(&mut h, &bt, fps);
            let mut tl = timeline("a.insv", 30.0, fps, 30.0, &h, &path).unwrap();
            tl.files.push(edit::File {
                path: "m.wav".into(),
                dur: 40.0,
            });
            tl.out.music = Some(edit::Music {
                file: 1,
                offset: 0.0,
                gain: 1.0,
                duck: true,
            });
            tl.check().unwrap();
            let (dt, mut strong) = (fps.t(1), 0);
            for (k, (c, x)) in tl.clips.iter().zip(&h).enumerate() {
                let t = fps.t(tl.frames(k).end);
                let b = (bt.iter())
                    .min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs()))
                    .unwrap();
                assert!(
                    (b.0 - t).abs() <= dt,
                    "{bpm} BPM: cut {k} at {t}, beat {b:?}"
                );
                strong += (b.1 > 1.0) as usize;
                assert!(
                    c.a >= x.lo - 1e-9 && c.b <= x.hi + 1e-9 && c.b - c.a >= 1.0 - 1e-9,
                    "{bpm} BPM: clip {k} {}..{} outside {x:?}",
                    c.a,
                    c.b
                );
            }
            println!(
                "{bpm} BPM at {:.2} fps: {} cuts on beats ({strong} strong), {:.2} s against {sum:.2}: {:?}",
                fps.num as f64 / fps.den as f64,
                tl.clips.len(),
                tl.dur(),
                tl.clips.iter().map(|c| (c.a, c.b)).collect::<Vec<_>>()
            );
            assert!((tl.dur() - sum).abs() <= h0.len() as f64 * p);
        }
    }
}

/// The blur on a checkerboard equirect (1 px squares, Rgba8Unorm) with regions in the middle, across the seam and on
/// the zenith: pixels well inside each region come out flat (the squares averaged), pixels away from all are copied
/// unchanged.
#[test]
fn blur() {
    use crate::{Blur, Region};
    let (_, _, dev, q) = open();
    let (w, h) = (512u32, 256u32);
    let px: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = if (i % w + i / w) % 2 == 0 { 255 } else { 0 };
            [v, v, v, 255]
        })
        .collect();
    let t = |usage| {
        dev.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        })
    };
    use wgpu::TextureUsages as U;
    let (src, dst) = (
        t(U::TEXTURE_BINDING | U::COPY_DST),
        t(U::RENDER_ATTACHMENT | U::COPY_SRC),
    );
    q.write_texture(
        src.as_image_copy(),
        &px,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: None,
        },
        src.size(),
    );
    let r = |y: f32, p: f32, s: f32| Region {
        dir: dir(y, p),
        size: Vec2::splat(s),
        score: 1.0,
    };
    // yaw 180 is the seam (column 0)
    let regs = [r(0.0, 0.0, 20.0), r(180.0, -20.0, 20.0), r(0.0, 90.0, 30.0)];
    let mut b = Blur::new(&dev, wgpu::TextureFormat::Rgba8Unorm);
    let mut enc = dev.create_command_encoder(&Default::default());
    b.run(
        &dev,
        &q,
        &mut enc,
        &src.create_view(&Default::default()),
        &dst.create_view(&Default::default()),
        &regs,
    );
    q.submit([enc.finish()]);
    let o = stitch::read_texture(&dev, &q, &dst);
    // the pixel of a direction (stitch layout)
    let at = |d: Vec3| {
        let x = ((-d.y).atan2(d.x) / std::f32::consts::TAU).rem_euclid(1.0) * w as f32;
        let y = d.z.clamp(-1.0, 1.0).acos() / std::f32::consts::PI * (h - 1) as f32;
        ((x as u32).min(w - 1), (y.round() as u32).min(h - 1))
    };
    let v = |(x, y): (u32, u32)| o[((y * w + x) * 4) as usize] as i32;
    for g in &regs {
        // inside: the centre and points 0.6 of the way to the edge, flat grey
        let c = Cam::new(
            g.dir,
            Vec3::Z,
            Vec2::splat((g.size.x / 2.0).to_radians().tan()),
        );
        for p in [
            Vec2::ZERO,
            Vec2::new(0.6, 0.0),
            Vec2::new(-0.6, 0.0),
            Vec2::new(0.0, 0.6),
        ] {
            let x = v(at(c.ray(p)));
            assert!((x - 128).abs() < 40, "region {:?} at {p}: {x}", g.dir);
        }
    }
    // away from all: unchanged
    let mut same = 0;
    for (y, p) in [(90.0, 0.0), (-90.0, 30.0), (45.0, -60.0), (120.0, 20.0)] {
        let (x, yy) = at(dir(y, p));
        same += (v((x, yy)) == px[((yy * w + x) * 4) as usize] as i32) as u32;
    }
    assert_eq!(same, 4, "pixels outside the regions changed");
}

/// macOS: YuNet through Core ML (FP16 on the Neural Engine) against burn's (f32 on the GPU) on the same crops of a
/// busy panorama (sinusoids and blocky noise, 3072 × 1536), 8 face tiles in one run: scores within 0.02, boxes within
/// 2 % of their range, the tiles past a short run untouched by it. A compiled model in the cache that is cut short is
/// made again.
#[cfg(target_os = "macos")]
#[test]
fn coreml() {
    if no_cf_home() {
        return;
    }
    let (inst, ad, dev, q) = open();
    let (w, h) = (3072u32, 1536u32);
    let px: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let n = ((i / w / 24).wrapping_mul(0x9e37_79b9)
                ^ (i % w / 24).wrapping_mul(0x85eb_ca6b))
                >> 24;
            let v = |a: f32, b: f32| {
                (127.0 + 60.0 * (x * a + y * b).sin() + (n as f32 - 128.0) * 0.5) as u8
            };
            [v(0.013, 0.007), v(0.005, 0.021), v(0.031, 0.002), 255]
        })
        .collect();
    let tex = dev.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    q.write_texture(
        tex.as_image_copy(),
        &px,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: None,
        },
        tex.size(),
    );
    let mut ai = Ai::new(&inst, &ad, &dev, &q).unwrap();
    let p = ai.pano(&tex.create_view(&Default::default()));
    let nets = ai.nets().unwrap();
    let s = privacy::FS as usize;
    let tiles = privacy::face_tiles();
    // a cache of its own: the shared one is the app's and other tests'
    struct Rm(PathBuf);
    impl Drop for Rm {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let root = Rm(std::env::temp_dir().join(format!("x5desk-coreml-test-{}", std::process::id())));
    std::fs::create_dir_all(&root.0).unwrap();
    let ml = crate::coreml::Yunet::at(&dev, &root.0).unwrap();
    // a full run, then a short one of other tiles: the buffer's tiles past it still hold the full run's
    let mut last = vec![];
    for (n, t) in [(8, &tiles[..8]), (3, &tiles[tiles.len() - 3..])] {
        let jobs: Vec<Job> = t
            .iter()
            .enumerate()
            .map(|(i, c)| Job::new(c, [privacy::FS; 2], i * 3 * s * s, &privacy::NORM, p.w))
            .collect();
        let _ = ai.crop(&nets.fc, &p, &jobs, &nets.fi);
        let a = nets.burn(n);
        // the second run's time: the first one sets up the Neural Engine's buffers
        let mut ms = 0.0;
        let b = (0..2)
            .map(|_| {
                let t = std::time::Instant::now();
                let b = ml.run(&dev, &q, &nets.fc.dst, n).unwrap();
                ms = t.elapsed().as_secs_f64() * 1e3;
                b
            })
            .last()
            .unwrap();
        assert_eq!(a.len(), b.len());
        let (mut ds, mut db, mut mb) = (0f32, 0f32, 0f32);
        for (x, y) in a.chunks(5).zip(b.chunks(5)) {
            ds = ds.max((x[0] - y[0]).abs());
            for k in 1..5 {
                (db, mb) = (db.max((x[k] - y[k]).abs()), mb.max(x[k].abs()));
            }
        }
        let hot = a.chunks(5).filter(|r| r[0] > 0.3).count();
        eprintln!(
            "{n} tiles: Core ML {ms:.1} ms; max |Core ML − burn| score {ds:.4}, box {db:.3} of {mb:.1}; {hot} anchors \
             above 0.3"
        );
        assert!(
            ds < 0.02 && db < 0.02 * mb,
            "{n} tiles: score {ds}, box {db} of {mb}"
        );
        last = b;
    }
    // the compiled model cut short: made again, and the same as before
    drop(ml);
    let at = crate::coreml::cached(&root.0);
    let mut cut = vec![];
    for e in std::fs::read_dir(&at).unwrap().flatten() {
        let p = e.path();
        match p.is_dir() {
            true => std::fs::remove_dir_all(p).unwrap(),
            false => {
                std::fs::write(&p, b"cut").unwrap();
                cut.push(p)
            }
        }
    }
    assert!(!cut.is_empty());
    let ml = crate::coreml::Yunet::at(&dev, &root.0).unwrap();
    assert!(
        cut.iter().all(|p| std::fs::read(p).unwrap() != b"cut"),
        "{cut:?}"
    );
    let again = ml.run(&dev, &q, &nets.fc.dst, 3).unwrap();
    let d = again
        .iter()
        .zip(&last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(d <= 1e-6, "rebuilt model: {d}");
    ai.pv = Some(nets);
}

/// Half to single precision: zero, normals, the infinities, subnormals (the smallest, one in the middle, the largest).
#[cfg(target_os = "macos")]
#[test]
fn half() {
    use crate::coreml::half;
    for (h, f) in [
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3c00, 1.0),
        (0xc000, -2.0),
        (0x0400, 2f32.powi(-14)),
        (0x7bff, 65504.0),
        (0x7c00, f32::INFINITY),
        (0x0001, 2f32.powi(-24)),
        (0x0200, 2f32.powi(-15)),
        (0x03ff, 1023.0 * 2f32.powi(-24)),
    ] {
        assert_eq!(half(h).to_bits(), f.to_bits(), "{h:#06x}");
    }
}
