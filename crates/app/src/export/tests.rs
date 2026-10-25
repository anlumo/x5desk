use super::*;

/// The preview's ray of pixel `p` on an `s` target (`blur.wgsl` `ray`, `stitch.wgsl`), in the view camera frame.
fn ray(p: Vec2, mode: u32, f: f32, d: f32, s: Vec2) -> Vec3 {
    let u = p / s;
    let x = Vec2::new(2.0 * u.x - 1.0, (1.0 - 2.0 * u.y) * s.y / s.x) * f;
    let pi = std::f32::consts::PI;
    let r = match mode {
        1 => {
            let (lon, lat) = ((2.0 * u.x - 1.0) * pi, (0.5 - u.y) * pi);
            Vec3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos())
        }
        2 => {
            let p = x.length();
            let k = p * p / ((d + 1.0) * (d + 1.0));
            let e = (d + (k * (1.0 - d * d) + 1.0).sqrt()) / (k + 1.0);
            (p * e / (d + 1.0) * (x + Vec2::new(1e-12, 0.0)).normalize()).extend(e - d)
        }
        _ => {
            let k = x.x * x.x / ((d + 1.0) * (d + 1.0));
            let cl = (-k * d + (k * k * d * d - (k + 1.0) * (k * d * d - 1.0)).sqrt()) / (k + 1.0);
            let s = (d + 1.0) / (d + cl);
            let (lon, lat) = (x.x.atan2(s * cl), x.y.atan2(s));
            Vec3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos())
        }
    };
    r.normalize()
}

/// `px` inverts the preview's ray in every projection (perspective, Panini, planet, equirect) within 0.05 px, and the
/// view matrix is the preview's: the view's forward direction lands on the centre.
#[test]
fn projection() {
    let s = Vec2::new(1920.0, 1080.0);
    let views = [
        View {
            fov: 100.0,
            yaw: 30.0,
            pitch: -10.0,
            ..View::default()
        },
        View {
            fov: 170.0,
            panini: 1.0,
            ..View::default()
        },
        View {
            mode: Mode::Planet,
            pitch: -90.0,
            fov: 280.0,
            ..View::default()
        },
        View {
            mode: Mode::Equirect,
            ..View::default()
        },
    ];
    for v in views {
        let (m, mode, f, d) = proj(&v);
        let c = px(m.transpose() * track::dir(v.yaw, v.pitch), mode, f, d, s).unwrap();
        assert!((c - s / 2.0).length() < 0.05, "{v:?}: centre at {c}");
        let mut worst = 0f32;
        for y in (10..1080).step_by(97) {
            for x in (10..1920).step_by(131) {
                let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let r = ray(p, mode, f, d, s);
                if let Some(q) = px(r, mode, f, d, s) {
                    worst = worst.max((q - p).length());
                }
            }
        }
        assert!(worst < 0.05, "{v:?}: off by {worst} px");
    }
}

/// Presets apply their output settings and keep the rest; the hook's tokens; H.264 stops at 4096 px.
#[test]
fn presets() {
    let d = Set::default();
    assert_eq!(
        (d.preset, d.pano, d.size(), d.codec, d.deep),
        (Some(0), true, [5760, 2880], Codec::H265, true)
    );
    let x = parse(Set { faces: false, ..d }, "flat1080:film:faces").unwrap();
    assert_eq!(
        (
            x.pano,
            x.size(),
            x.codec,
            x.deep,
            x.mbps,
            x.look,
            x.faces,
            x.raft
        ),
        (
            false,
            [1920, 1080],
            Codec::H264,
            false,
            20.0,
            Some(fx::Look::Film),
            true,
            true
        )
    );
    let x = parse(d, "pano4k:noraft:qp=20:black").unwrap();
    assert_eq!(
        (x.size(), x.raft, x.qp, x.look),
        ([3840, 1920], false, Some(20), Some(fx::Look::Mono))
    );
    assert!(parse(d, "nope").is_err() && parse(d, "pano4k:sepia").is_err());
    let s = Spec {
        files: vec!["v.insv".into()],
        photo: false,
        dur: 5.0,
        t: 1.0,
        view: View::default(),
        look: Look::Perspective,
        stab: None,
        smooth: 0.5,
        follow: None,
        faces: false,
    };
    let h264 = Set {
        codec: Codec::H264,
        ..d
    };
    assert!(h264.check(&s, false, [0.0, 5.0]).is_some());
    assert!(
        h264.check(&s, true, [0.0, 5.0]).is_none(),
        "a grab is a JPEG"
    );
    assert!(d.check(&s, false, [0.0, 5.0]).is_none());
    assert!(d.check(&s, false, [2.0, 2.0]).is_some());
    let big = Set { w: 11904, ..d };
    assert!(
        big.check(&s, false, [0.0, 5.0]).is_some(),
        "past the encoder"
    );
    assert!(
        big.check(&s, true, [0.0, 5.0]).is_none(),
        "a grab is a JPEG"
    );
}

/// One device for the GPU tests, one test at a time.
fn gpu() -> (std::sync::MutexGuard<'static, ()>, Arc<Gpu>, Ctx) {
    static G: std::sync::OnceLock<(Mutex<()>, Arc<Gpu>, Ctx)> = std::sync::OnceLock::new();
    let (m, g, cx) = G.get_or_init(|| {
        let g = Arc::new(Gpu::new().expect("GPU"));
        let cx = Ctx::new(&g.device, &g.queue);
        (Mutex::new(()), g, cx)
    });
    (
        m.lock().unwrap_or_else(|e| e.into_inner()),
        g.clone(),
        cx.clone(),
    )
}

fn fixture(name: &str) -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join(format!("x5desk-export-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn task(spec: Spec, set: Set, path: PathBuf, range: [f64; 2], grab: bool) -> Task {
    Task {
        it: Arc::new(Item {
            name: String::new(),
            path,
            st: Mutex::new(St::Queued),
            note: Mutex::new(String::new()),
            stop: AtomicBool::new(false),
        }),
        spec,
        set,
        range,
        grab,
        ed: None,
        lab: None,
    }
}

fn texture(g: &Gpu, w: u32, h: u32, usage: wgpu::TextureUsages) -> wgpu::Texture {
    g.device.create_texture(&wgpu::TextureDescriptor {
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
}

/// The flat blur on a 1-px checkerboard seen through a turned 90° view: a 12° region at a chosen pixel turns flat grey
/// there (the 0 / 255 pattern averaged), the frame outside its feathered ellipse stays exactly as it was.
#[test]
fn flat_blur() {
    let (_l, g, _) = gpu();
    let (w, h) = (640u32, 360u32);
    let px0: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = if (i % w + i / w) % 2 == 0 { 255 } else { 0 };
            [v, v, v, 255]
        })
        .collect();
    use wgpu::TextureUsages as U;
    let (src, dst) = (
        texture(&g, w, h, U::TEXTURE_BINDING | U::COPY_DST),
        texture(&g, w, h, U::RENDER_ATTACHMENT | U::COPY_SRC),
    );
    g.queue.write_texture(
        src.as_image_copy(),
        &px0,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: None,
        },
        src.size(),
    );
    let v = View {
        yaw: 30.0,
        pitch: 10.0,
        fov: 90.0,
        ..View::default()
    };
    let (m, mode, f, d) = proj(&v);
    let s = Vec2::new(w as f32, h as f32);
    let at = Vec2::new(480.5, 120.5);
    let reg = track::Region {
        dir: m * ray(at, mode, f, d, s),
        size: Vec2::splat(12.0),
        score: 1.0,
    };
    let mut b = Blur::new(&g.device, wgpu::TextureFormat::Rgba8Unorm);
    let mut e = g.device.create_command_encoder(&Default::default());
    b.run(
        &g.device,
        &g.queue,
        &mut e,
        &src.create_view(&Default::default()),
        &dst.create_view(&Default::default()),
        &v,
        &[reg],
    );
    g.queue.submit([e.finish()]);
    let o = stitch::read_texture(&g.device, &g.queue, &dst);
    let p = |x: u32, y: u32| o[((y * w + x) * 4) as usize] as i32;
    // the region's radius in pixels at its centre: 6° off its axis
    let edge = px(
        m.transpose()
            * track::Cam::new(reg.dir, Vec3::Z, Vec2::ONE)
                .ray(Vec2::new(0.6 * 6f32.to_radians().tan(), 0.0)),
        mode,
        f,
        d,
        s,
    )
    .unwrap();
    let r = (edge - at).length() / 0.6;
    eprintln!("region radius {r:.1} px at {at}");
    assert!(r > 20.0 && r < 60.0);
    let (cx, cy) = (at.x as u32, at.y as u32);
    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (5, -3), (-7, 4)] {
        let q = p((cx as i32 + dx) as u32, (cy as i32 + dy) as u32);
        assert!((100..=155).contains(&q), "({dx}, {dy}): {q}, not grey");
    }
    // outside 1.3 radii nothing changed
    let mut changed = 0;
    for y in 0..h {
        for x in 0..w {
            let far = (Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - at).length() > 1.3 * r;
            if far && p(x, y) != px0[((y * w + x) * 4) as usize] as i32 {
                changed += 1;
            }
        }
    }
    assert_eq!(changed, 0, "pixels changed outside the region");
}

/// The blur hook takes each output frame's view from `pick`: a region seen through a view that turns between two
/// frames (the follow view) blurs where that frame's view shows it, and where the other frame's view shows it the
/// checkerboard stays as it was.
#[test]
fn hook_view() {
    let (_l, g, _) = gpu();
    let (w, h) = (640u32, 360u32);
    let px0: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let v = if (i % w + i / w) % 2 == 0 { 255 } else { 0 };
            [v, v, v, 255]
        })
        .collect();
    use wgpu::TextureUsages as U;
    let tex = texture(
        &g,
        w,
        h,
        U::TEXTURE_BINDING | U::RENDER_ATTACHMENT | U::COPY_SRC | U::COPY_DST,
    );
    let vs = [(30.0, 10.0), (50.0, 15.0)].map(|(yaw, pitch)| View {
        yaw,
        pitch,
        fov: 90.0,
        ..View::default()
    });
    let s = Vec2::new(w as f32, h as f32);
    let (m, mode, f, d) = proj(&vs[0]);
    let reg = track::Region {
        dir: m * ray(Vec2::new(480.5, 120.5), mode, f, d, s),
        size: Vec2::splat(12.0),
        score: 1.0,
    };
    let at = vs.map(|v| {
        let (m, mode, f, d) = proj(&v);
        px(m.transpose() * reg.dir, mode, f, d, s).unwrap()
    });
    eprintln!(
        "the region at {} in frame 0's view, {} in frame 1's",
        at[0], at[1]
    );
    assert!(at[1].cmpge(Vec2::ZERO).all() && at[1].cmplt(s).all() && at[0].distance(at[1]) > 100.0);
    let mut hk = hook(&g, true, vec![(vec![0.0], vec![vec![reg]])], move |k, _| {
        Some((0, 0.0, vs[k as usize]))
    });
    for k in 0..2 {
        g.queue.write_texture(
            tex.as_image_copy(),
            &px0,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: None,
            },
            tex.size(),
        );
        let mut e = g.device.create_command_encoder(&Default::default());
        hk(::export::Fx {
            enc: &mut e,
            tex: &tex,
            k,
            t: k as f64 / 30.0,
        });
        g.queue.submit([e.finish()]);
        let o = stitch::read_texture(&g.device, &g.queue, &tex);
        let i = |q: Vec2| ((q.y as u32 * w + q.x as u32) * 4) as usize;
        let (me, other) = (at[k as usize], at[1 - k as usize]);
        assert!(
            (100..=155).contains(&o[i(me)]),
            "frame {k}: {} where its view shows the region, not grey",
            o[i(me)]
        );
        for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-3.0, 2.0)] {
            let q = other + Vec2::new(dx, dy);
            assert_eq!(
                o[i(q)],
                px0[i(q)],
                "frame {k}: blurred where the other view shows the region"
            );
        }
    }
}

/// A synthetic 2048 × 1024 equirect photo (smooth colours, a red square at its centre) exported as 360 (GPano, the
/// same picture within 36 dB PSNR), as the default view (the red square in the middle, no GPano) and in black and
/// white (grey).
#[test]
fn photo() {
    let (_l, g, cx) = gpu();
    let (w, h) = (2048u32, 1024u32);
    let px0: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let sq = x.abs_diff(w / 2) < 40 && y.abs_diff(h / 2) < 40;
            let t = std::f32::consts::TAU * x as f32 / w as f32;
            match sq {
                true => [230, 20, 20, 255],
                false => [
                    (128.0 + 90.0 * t.cos()) as u8,
                    (40 + y * 160 / h) as u8,
                    (128.0 + 90.0 * t.sin()) as u8,
                    255,
                ],
            }
        })
        .collect();
    let src = fixture("photo.jpg");
    std::fs::write(&src, export::jpeg::encode(&px0, w, h, 95, None).unwrap()).unwrap();
    let spec = Spec {
        files: vec![src.clone()],
        photo: true,
        dur: 0.0,
        t: 0.0,
        view: Look::Perspective.view(),
        look: Look::Perspective,
        stab: None,
        smooth: 0.0,
        follow: None,
        faces: false,
    };
    let mut k = Keep::default();
    let run = |k: &mut Keep, set: Set, name: &str| {
        let t = task(spec.clone(), set, fixture(name), [0.0; 2], false);
        k.photo(&g, &cx, &t).unwrap();
        let b = std::fs::read(&t.it.path).unwrap();
        let (px, w, h) = photo::load(&t.it.path).unwrap();
        (px, w, h, b.windows(5).any(|w| w == b"GPano"))
    };
    let base = Set {
        pano: true,
        w,
        preset: None,
        ..Set::default()
    };
    let (px, w1, h1, gp) = run(&mut k, base, "pano.jpg");
    assert_eq!((w1, h1, gp), (w, h, true));
    let (src_px, ..) = photo::load(&src).unwrap();
    let mse = (px.iter().zip(&src_px).enumerate())
        .filter(|(i, _)| i % 4 != 3)
        .map(|(_, (a, b))| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / (w * h * 3) as f64;
    let psnr = 10.0 * (255.0f64.powi(2) / mse).log10();
    eprintln!("360 photo export: PSNR {psnr:.2} dB against the source");
    assert!(psnr > 36.0);
    let flat = Set {
        pano: false,
        w: 640,
        h: 360,
        ..base
    };
    let (px, w1, h1, gp) = run(&mut k, flat, "flat.jpg");
    assert_eq!((w1, h1, gp), (640, 360, false));
    let c = &px[((180 * 640 + 320) * 4) as usize..][..3];
    eprintln!("flat view centre {c:?}");
    assert!(
        c[0] > 180 && c[1] < 70 && c[2] < 70,
        "centre {c:?}: not the red square"
    );
    let mono = Set {
        look: Some(fx::Look::Mono),
        ..flat
    };
    let (px, ..) = run(&mut k, mono, "mono.jpg");
    let worst = px
        .chunks(4)
        .map(|p| p[0].abs_diff(p[1]).max(p[1].abs_diff(p[2])))
        .max()
        .unwrap();
    eprintln!("black & white: largest channel difference {worst}");
    assert!(worst <= 6);
    // the Insta360 trailer's file category decides, also for a 2:1 picture: 6 an equirect, 2 a dual fisheye
    let jpg = std::fs::read(&src).unwrap();
    let insp = |cat: u8, name: &str| {
        let p = fixture(name);
        // ExtraMetadata field 129 (varint), then a record header (format, kind, size) and the V3 footer
        let md = [0x88, 0x08, cat];
        let mut b = [&jpg[..], &md, &[0, media::insv::METADATA, 3, 0, 0, 0]].concat();
        let mut foot = [0u8; 72];
        foot[32..36].copy_from_slice(&(3 + 6 + 72u32).to_le_bytes());
        foot[36..40].copy_from_slice(&3u32.to_le_bytes());
        foot[40..].copy_from_slice(b"8db42d694ccc418790edff439fe026bf");
        b.extend_from_slice(&foot);
        std::fs::write(&p, b).unwrap();
        p
    };
    let (dual, eq) = (insp(2, "dual.insp"), insp(6, "eq.insp"));
    assert_eq!(media::insv::Trailer::open(&dual).unwrap().meta.category, 2);
    assert_eq!(photo::flat(&eq, w, h), None);
    assert_eq!(
        photo::flat(&src, w, h),
        None,
        "no trailer: 2:1 is an equirect"
    );
    assert!(photo::flat(&fixture("flat.jpg"), 640, 360).is_some());
    let why = photo::flat(&dual, w, h).expect("a dual fisheye is no equirect");
    let t = task(
        Spec {
            files: vec![dual.clone()],
            ..spec.clone()
        },
        base,
        fixture("dual.jpg"),
        [0.0; 2],
        false,
    );
    let e = k.photo(&g, &cx, &t).unwrap_err();
    eprintln!("dual-fisheye .insp: {e}");
    assert!(e == why && e.contains("dual-fisheye") && !t.it.path.exists());
    std::fs::remove_dir_all(src.parent().unwrap()).unwrap();
}

/// Frame `k` of a video as RGB24, decoded by FFmpeg (a test tool).
fn ff_frame(p: &Path, k: usize) -> Vec<u8> {
    let o = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(p)
        .args([
            "-vf",
            &format!("select=eq(n\\,{k})"),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()
        .expect("ffmpeg");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    o.stdout
}

/// Privacy blur on v635 (`X5_SAMPLES`): the face pass finds faces in its first 0.3 s; a flat 960 × 540 view aimed at
/// the strongest one on frame 0, exported with and without the blur (H.264 QP 18, template seam, no stabilisation):
/// frame 0 differs inside the face's region and is the same away from it.
#[test]
fn faces() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let f = d.join("v635.insv");
    let (fps, dur) = probe(&f).unwrap();
    let mut spec = Spec {
        files: vec![f],
        photo: false,
        dur,
        t: 0.0,
        view: View::default(),
        look: Look::Perspective,
        stab: None,
        smooth: 0.0,
        follow: None,
        faces: false,
    };
    let set = Set {
        preset: None,
        pano: false,
        w: 960,
        h: 540,
        codec: Codec::H264,
        deep: false,
        qp: Some(18),
        raft: false,
        stab: None,
        faces: true,
        ..Set::default()
    };
    let (mut k, ctx, range) = (Keep::default(), egui::Context::default(), [0.0, 0.3]);
    let t = task(spec.clone(), set, fixture("x.mp4"), range, false);
    let t0 = Instant::now();
    let (_, regs) = (k.faces(
        &g,
        &cx,
        &t,
        &spec.files,
        None,
        range,
        fps,
        &mut [0; 5],
        &ctx,
    ))
    .unwrap();
    eprintln!(
        "face pass {:.1} s (first use compiles YuNet): {}",
        t0.elapsed().as_secs_f64(),
        t.it.note()
    );
    let r = *regs[0]
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("a face on frame 0");
    let [yaw, pitch] = track::yaw_pitch(r.dir);
    spec.view = View {
        yaw,
        pitch,
        fov: 60.0,
        ..View::default()
    };
    eprintln!(
        "aiming at yaw {yaw:.1}° pitch {pitch:.1}°, region {:.1}° × {:.1}° (score {:.2})",
        r.size.x, r.size.y, r.score
    );
    let mut run = |faces: bool| {
        let t = task(
            spec.clone(),
            Set { faces, ..set },
            fixture(&format!("faces_{faces}.mp4")),
            range,
            false,
        );
        let p = k.video(&g, &cx, &t, &ctx).unwrap();
        assert_eq!(p.done, 9);
        ff_frame(&t.it.path, 0)
    };
    let (a, b) = (run(true), run(false));
    let (w, h) = (960usize, 540usize);
    assert_eq!((a.len(), b.len()), (w * h * 3, w * h * 3));
    // every region in the view: centre and larger radius in pixels (twice that at the edge of a 60° view at most)
    let (m, mode, f, d) = proj(&spec.view);
    let s = Vec2::new(w as f32, h as f32);
    let px_of = |q: &track::Region| {
        let c = px(m.transpose() * q.dir, mode, f, d, s)?;
        Some((
            c,
            (q.size.max_element() / 2.0).to_radians().tan() / 30f32.to_radians().tan() * s.x / 2.0,
        ))
    };
    let all: Vec<(Vec2, f32)> = regs[0].iter().filter_map(px_of).collect();
    let rad =
        (r.size.min_element() / 2.0).to_radians().tan() / 30f32.to_radians().tan() * s.x / 2.0;
    let (mut din, mut nin, mut dout, mut nout) = (0.0, 0, 0.0, 0);
    for y in 0..h {
        for x in 0..w {
            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let dd = (0..3)
                .map(|c| a[(y * w + x) * 3 + c].abs_diff(b[(y * w + x) * 3 + c]) as f64)
                .sum::<f64>()
                / 3.0;
            if (p - s / 2.0).length() < 0.5 * rad {
                (din, nin) = (din + dd, nin + 1);
            } else if all
                .iter()
                .all(|(c, r)| (p - *c).length() > 2.0 * 1.25 * r + 32.0)
            {
                (dout, nout) = (dout + dd, nout + 1);
            }
        }
    }
    let (din, dout) = (din / nin as f64, dout / nout.max(1) as f64);
    eprintln!(
        "{} regions in front; aimed one {rad:.0} px: mean |blurred − sharp| inside {din:.2}, away from all regions {dout:.3} \
         ({nout} px)",
        all.len()
    );
    assert!(din > 3.0, "the face is not blurred");
    assert!(
        nout > 10000 && dout < 0.1,
        "the frame changed away from the faces"
    );
    std::fs::remove_dir_all(fixture("x").parent().unwrap()).unwrap();
}

/// Privacy blur on an editor timeline of v635 (`X5_SAMPLES`): clip 0 plays 1.0–1.2 s looking away, clip 1 plays
/// 0.0–0.3 s with a view aimed at the strongest face of frame 0 (H.264 QP 18, template seam, no stabilisation).
/// Exported with and without the blur, clip 1's first output frame differs inside the face's region and is the same
/// away from all regions: each output frame takes its own clip's face pass and view.
#[test]
fn timeline_faces() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let f = d.join("v635.insv");
    let (fps, dur) = probe(&f).unwrap();
    let spec = Spec {
        files: vec![f.clone()],
        photo: false,
        dur,
        t: 0.0,
        view: View::default(),
        look: Look::Perspective,
        stab: None,
        smooth: 0.0,
        follow: None,
        faces: false,
    };
    let set = Set {
        preset: None,
        pano: false,
        w: 960,
        h: 540,
        codec: Codec::H264,
        deep: false,
        qp: Some(18),
        raft: false,
        stab: None,
        faces: true,
        ..Set::default()
    };
    let (mut k, ctx) = (Keep::default(), egui::Context::default());
    let t = task(spec.clone(), set, fixture("y.mp4"), [0.0, 0.3], false);
    let (_, regs) = (k.faces(
        &g,
        &cx,
        &t,
        &spec.files,
        None,
        [0.0, 0.3],
        fps,
        &mut [0; 5],
        &ctx,
    ))
    .unwrap();
    let r = *regs[0]
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("a face on frame 0");
    let [yaw, pitch] = track::yaw_pitch(r.dir);
    let key = |t: f64, yaw: f32| edit::Key {
        t,
        v: edit::View::look(yaw, pitch, 0.0, 60.0, edit::Proj::Persp),
        e: edit::Ease::Smooth,
    };
    let mut tl = edit::Timeline::new(fps);
    tl.files.push(edit::File {
        path: f.to_string_lossy().into(),
        dur,
    });
    tl.clips.push(edit::Clip {
        view: vec![key(1.0, yaw + 180.0)],
        ..edit::Clip::new(0, 1.0, 1.2)
    });
    tl.clips.push(edit::Clip {
        view: vec![key(0.0, yaw)],
        ..edit::Clip::new(0, 0.0, 0.3)
    });
    tl.fix();
    let n1 = tl.frames(1).start as usize;
    let mut run = |faces: bool| {
        let t = task(
            spec.clone(),
            Set { faces, ..set },
            fixture(&format!("tl_faces_{faces}.mp4")),
            [0.0, tl.dur()],
            false,
        );
        let e = Tl {
            name: "t".into(),
            files: vec![vec![f.clone()]],
            tl: tl.clone(),
            looks: Default::default(),
            over: vec![],
            denoise: None,
            stab: None,
            smooth: 0.0,
        };
        let p = k.timeline(&g, &cx, &t, &e, &ctx).unwrap();
        assert_eq!(p.done, tl.len());
        if faces {
            eprintln!("timeline face passes: {}", t.it.note());
        }
        ff_frame(&t.it.path, n1)
    };
    let (a, b) = (run(true), run(false));
    let (w, h) = (960usize, 540usize);
    let v = View {
        yaw,
        pitch,
        fov: 60.0,
        ..View::default()
    };
    let (m, mode, f, d) = proj(&v);
    let s = Vec2::new(w as f32, h as f32);
    let all: Vec<(Vec2, f32)> = (regs[0].iter())
        .filter_map(|q| {
            let c = px(m.transpose() * q.dir, mode, f, d, s)?;
            Some((
                c,
                (q.size.max_element() / 2.0).to_radians().tan() / 30f32.to_radians().tan() * s.x
                    / 2.0,
            ))
        })
        .collect();
    let rad =
        (r.size.min_element() / 2.0).to_radians().tan() / 30f32.to_radians().tan() * s.x / 2.0;
    let (mut din, mut nin, mut dout, mut nout) = (0.0, 0, 0.0, 0);
    for y in 0..h {
        for x in 0..w {
            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let dd = (0..3)
                .map(|c| a[(y * w + x) * 3 + c].abs_diff(b[(y * w + x) * 3 + c]) as f64)
                .sum::<f64>()
                / 3.0;
            if (p - s / 2.0).length() < 0.5 * rad {
                (din, nin) = (din + dd, nin + 1);
            } else if all
                .iter()
                .all(|(c, r)| (p - *c).length() > 2.0 * 1.25 * r + 32.0)
            {
                (dout, nout) = (dout + dd, nout + 1);
            }
        }
    }
    let (din, dout) = (din / nin as f64, dout / nout.max(1) as f64);
    eprintln!(
        "clip 1's first frame ({n1}): mean |blurred − sharp| inside the aimed face {din:.2}, away from all regions \
         {dout:.3} ({nout} px)"
    );
    assert!(din > 3.0, "the face is not blurred");
    assert!(
        nout > 10000 && dout < 0.1,
        "the frame changed away from the faces"
    );
    for x in [true, false] {
        std::fs::remove_file(fixture(&format!("tl_faces_{x}.mp4"))).unwrap();
    }
}

/// The queue on v635 (`X5_SAMPLES`): a job cancelled while it runs ends cancelled and leaves no file, the next one
/// runs to the end with its progress (frames, steady rate, ETA) on the way, and a job whose source is gone fails with
/// the reason.
#[test]
fn queue() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let dir = fixture("q").parent().unwrap().join("q");
    let mut e = Exp::new(&g, &cx, &egui::Context::default(), Some(dir.clone()), false);
    e.source(Spec::of(&d.join("v635.insv")).unwrap());
    let x = Set {
        preset: None,
        pano: false,
        w: 640,
        h: 360,
        codec: Codec::H264,
        deep: false,
        raft: false,
        ..Set::default()
    };
    e.queue(x, false).unwrap();
    e.queue(x, false).unwrap();
    let (a, b) = (e.items[0].clone(), e.items[1].clone());
    assert_ne!(a.path, b.path, "a second job gets its own file");
    let t0 = Instant::now();
    while !matches!(a.st(), St::Run(p) if p.done >= 10) {
        assert!(t0.elapsed().as_secs() < 120, "{:?}", a.st());
        thread::sleep(std::time::Duration::from_millis(5));
    }
    a.stop.store(true, Relaxed);
    let mut seen = None;
    while !b.over() {
        if let St::Run(p) = b.st()
            && p.done > 20
        {
            seen.get_or_insert(p);
        }
        assert!(t0.elapsed().as_secs() < 240, "{:?}", b.st());
        thread::sleep(std::time::Duration::from_millis(5));
    }
    eprintln!(
        "cancelled: {:?}; next: {:?}, on the way {seen:?}",
        a.st(),
        b.st()
    );
    assert_eq!(a.st(), St::Cancelled);
    assert!(
        !a.path.exists()
            && !dir
                .join(format!(
                    "{}.part",
                    a.path.file_name().unwrap().to_string_lossy()
                ))
                .exists()
    );
    assert!(matches!(b.st(), St::Done(p, _) if p.done == 175 && p.total == 175));
    let p = seen.expect("progress while running");
    assert!(p.total == 175 && p.fps > 0.0 && p.eta > 0.0);
    assert!(b.path.exists());
    // a source that went away
    let gone = dir.join("gone.insv");
    std::fs::copy(d.join("v635.lrv"), &gone).unwrap();
    let mut s = e.src.clone().unwrap();
    s.files = vec![gone.clone()];
    e.source(s);
    std::fs::remove_file(&gone).unwrap();
    e.queue(x, false).unwrap();
    let c = e.items[2].clone();
    while !c.over() {
        thread::sleep(std::time::Duration::from_millis(5));
    }
    eprintln!("gone: {:?}", c.st());
    assert!(matches!(c.st(), St::Failed(m) if m.contains("gone.insv")));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The surface-configure contract (main.rs `GATE`, `BURN`) on v635 (`X5_SAMPLES`): while a RAFT export with face
/// blur runs (the face pass, then crate export's renderer, decoders and encoder), holding BURN, then GATE exclusively
/// as the UI's retry does leaves the queue idle with no submit meanwhile (wgpu-core's configure check: `poll` waits
/// for the last submission and finds the queue empty), and the job still finishes. Every other check goes without
/// the locks and must see the job's submits, or the check proves nothing.
#[test]
fn gate() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    ::export::locks(&crate::GATE, &crate::BURN);
    let dir = fixture("g").parent().unwrap().join("g");
    let mut e = Exp::new(&g, &cx, &egui::Context::default(), Some(dir.clone()), false);
    e.source(Spec::of(&d.join("v635.insv")).unwrap());
    e.range = [0.0, 2.0];
    let x = Set {
        preset: None,
        pano: true,
        w: 1920,
        codec: Codec::H264,
        deep: false,
        raft: true,
        faces: true,
        ..Set::default()
    };
    e.queue(x, false).unwrap();
    let it = e.items[0].clone();
    // held checks in the face pass and in the render, checks without the locks, of them the busy ones
    let (mut held, mut free, mut busy) = ([0; 2], 0, 0);
    let t0 = Instant::now();
    while !it.over() {
        assert!(t0.elapsed().as_secs() < 300, "{:?}", it.st());
        let lock = held[0] + held[1] == free;
        let ph = matches!(it.st(), St::Run(_)) as usize;
        let s = {
            let _b = lock.then(|| crate::BURN.lock().unwrap_or_else(|e| e.into_inner()));
            let _w = lock.then(|| crate::GATE.write().unwrap_or_else(|e| e.into_inner()));
            g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap()
        };
        match lock {
            true => {
                assert!(
                    s.is_queue_empty(),
                    "a submit while the locks were held ({:?})",
                    it.st()
                );
                held[ph] += 1;
            }
            false => (free, busy) = (free + 1, busy + !s.is_queue_empty() as usize),
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
    eprintln!(
        "{:?}; held: {} in the face pass, {} in the render, all idle; without: {busy} of {free} saw a submit",
        it.st(),
        held[0],
        held[1]
    );
    assert!(matches!(it.st(), St::Done(p, _) if p.done == p.total && p.total > 0));
    assert!(held[0] >= 3 && held[1] >= 3 && busy > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}
