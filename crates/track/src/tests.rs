//! The imported networks against onnxruntime (`X5_ORT` = a Python with `onnxruntime` and `numpy`, a test tool) on
//! crops of a real frame (`X5_SAMPLES`): both see the same input tensors, read back from the GPU. Scratch files (a
//! 19 MB frame among them) go to `X5_FIXTURES` (default: the system temp dir).

use crate::{Ai, Cam, Crops, In, Job, SIDE, det, dir, privacy, read, reid, vit};
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

/// Synthetic signals over 30 s: people from 10 to 14 s, loud from 20 to 23 s, the camera turning from 4 to 6 s. The
/// picks (12 s) lie on those, fill the budget within a bin plus `MIN`, are 2 to 8 s, in order and apart; their
/// timeline passes crate edit's checks with view keys from the framing path.
#[test]
fn highlights() {
    use crate::{Det, Scorer, Shot, Sig, timeline};
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
    let h = s.pick(12.0);
    let sum: f64 = h.iter().map(|x| x.b - x.a).sum();
    assert!(
        (12.0..=12.0 + crate::BIN + 2.0).contains(&sum),
        "total {sum}: {h:?}"
    );
    for x in &h {
        assert!(x.b - x.a >= 2.0 - 1e-9 && x.b - x.a <= 8.0 + 1e-9, "{x:?}");
    }
    assert!(h.windows(2).all(|w| w[1].a >= w[0].b + 1.0));
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
