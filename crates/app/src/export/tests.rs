use super::*;
pub(super) use crate::ai::tests::no_cf_home;

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

/// A name is free only when neither the file nor its `.part` is there: a job in another process (an agent's) may be
/// writing that name. A dangling link at either name takes it too (a writer would write through it).
#[test]
fn free_names() {
    let d = crate::clip::tests::Tmp::new("free");
    let p = |n: &str| d.0.join(n);
    std::fs::write(p("a.mp4"), b"").unwrap();
    std::fs::write(p("b.mp4.part"), b"").unwrap();
    std::os::unix::fs::symlink(p("nowhere"), p("d.mp4.part")).unwrap();
    assert!(!free(&p("a.mp4")));
    assert!(!free(&p("b.mp4")));
    assert!(free(&p("c.mp4")));
    assert!(!free(&p("d.mp4")));
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
    let x = parse(d, "pano4k:plates").unwrap();
    assert!(x.plates && !x.faces && !parse(d, "pano4k:faces").unwrap().plates);
    let on = Set {
        denoise: Some(1.0),
        ..d
    };
    assert_eq!(parse(d, "pano4k:denoise=0.5").unwrap().denoise, Some(0.5));
    assert_eq!(parse(on, "pano4k:denoise=0").unwrap().denoise, None);
    assert_eq!(parse(on, "pano4k").unwrap().denoise, Some(1.0));
    assert!(parse(d, "pano4k:denoise=1.5").is_err() && parse(d, "pano4k:denoise=x").is_err());
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
        plates: false,
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

/// A source brings its face and plate blur: one the player previewed blurred turns it on, the next one turns it off
/// again.
#[test]
fn source_faces() {
    let (_l, g, cx) = gpu();
    let mut e = Exp::new(
        &g,
        &cx,
        &egui::Context::default(),
        Some(fixture("sf")),
        false,
    );
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
        faces: true,
        plates: false,
    };
    e.source(s.clone());
    assert!(e.set.faces);
    e.source(Spec {
        faces: false,
        ..s.clone()
    });
    assert!(!e.set.faces, "the blur stayed on for the next source");
    e.source(Spec {
        plates: true,
        ..s.clone()
    });
    assert!(e.set.plates && e.set.faces);
    e.source(Spec { faces: false, ..s });
    assert!(
        !e.set.plates,
        "the plate blur stayed on for the next source"
    );
}

/// The player's blur previews go into its export spec, faces and plates each on its own (the player on v606,
/// `X5_SAMPLES`).
#[test]
fn player_spec() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let mut p = crate::player::Player::open(
        &g,
        &cx,
        crate::library::pair(&d.join("v606.insv")),
        &egui::Context::default(),
        &mpsc::channel().0,
        &gyro::worker(),
        None,
    )
    .unwrap();
    for (faces, plates) in [(true, false), (false, true), (true, true), (false, false)] {
        (p.blur, p.plates) = (faces, plates);
        let s = p.spec().unwrap();
        assert_eq!((s.faces, s.plates), (faces, plates));
    }
    // a finished auto framing is its follow view while followed
    let mut fr = crate::ai::tests::made();
    fr.auto = true;
    let fr = Arc::new(fr);
    (p.frm, p.framing) = (Some(fr.clone()), true);
    assert!(
        p.spec()
            .unwrap()
            .follow
            .is_some_and(|f| Arc::ptr_eq(&f, &fr))
    );
    p.framing = false;
    assert!(p.spec().unwrap().follow.is_none());
}

/// A flat export following auto framing (v635, `X5_SAMPLES`, 0–1 s, 320 × 180 H.264 QP 18, template seam, horizon lock):
/// the run (`ai::Track::framing`) as the spec's follow view, as the player hands it over. It runs, and its first frame
/// is the one a fixed view along the path's first direction shows, not the unframed view's (turned 90° from it).
#[test]
fn follow_framing() {
    if no_cf_home() {
        return;
    }
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let f = d.join("v635.insv");
    let st = (Some(stab::Lock::Horizon), gyro::SMOOTH);
    let ctx = egui::Context::default();
    let fr = crate::ai::Track::framing(&g, &cx, &ctx, f.clone(), st, 16.0 / 9.0);
    let r = crate::ai::tests::wait(|| fr.st(), 600);
    assert_eq!(r, crate::ai::St::Done, "{r:?}");
    let [yaw, pitch] = track::yaw_pitch(fr.dir(0).unwrap());
    let (_, dur) = probe(&f).unwrap();
    let view = View {
        yaw: yaw + 90.0,
        fov: 90.0,
        ..View::default()
    };
    let spec = Spec {
        files: vec![f],
        photo: false,
        dur,
        t: 0.0,
        view,
        look: Look::Perspective,
        stab: st.0,
        smooth: st.1,
        follow: None,
        faces: false,
        plates: false,
    };
    let set = Set {
        preset: None,
        pano: false,
        w: 320,
        h: 180,
        codec: Codec::H264,
        deep: false,
        qp: Some(18),
        raft: false,
        colour: false,
        stab: st.0,
        smooth: st.1,
        ..Set::default()
    };
    let mut k = Keep::default();
    let mut run = |name: &str, s: Spec| {
        let t = task(s, set, fixture(name), [0.0, 1.0], false);
        let p = k.video(&g, &cx, &t, &ctx).unwrap();
        (t.it.path.clone(), p)
    };
    let follow = Some(fr.clone());
    let (fol, p) = run(
        "autoframe.mp4",
        Spec {
            follow,
            ..spec.clone()
        },
    );
    let (fixed, _) = run("fixed.mp4", spec.clone());
    let view = View { yaw, pitch, ..view };
    let (along, _) = run("along.mp4", Spec { view, ..spec });
    let diff = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.abs_diff(*y) as f64)
            .sum::<f64>()
            / a.len() as f64
    };
    let a = ff_frame(&fol, 0);
    let (dx, da) = (
        diff(&a, &ff_frame(&fixed, 0)),
        diff(&a, &ff_frame(&along, 0)),
    );
    eprintln!(
        "auto framed export: {} of {} frames; frame 0 off the unframed view's by {dx:.2}, off the path's by {da:.2} \
         (mean per channel)",
        p.done, p.total
    );
    assert!(p.done == p.total && p.total >= 29);
    assert!(da < 1.0 && dx > 10.0, "{da} {dx}");
    for p in [fol, fixed, along] {
        std::fs::remove_file(p).unwrap();
    }
}

/// Licence plates on v606 (`X5_SAMPLES`) through the privacy pass as a frame grab runs it (one frame, no
/// stabilisation): its plates are a photo's, the sure ones. The parked Mercedes's front plate at frame 120 (yaw 118.1°,
/// pitch −8.6° unstabilised, measured and checked on a view aimed there) is found at 0.75 but only 20 px wide, under the
/// photo rule's width: no region there with plates, none with faces. The sample clips hold no plate wide enough for a
/// region to be asserted.
#[test]
fn plates() {
    if no_cf_home() {
        return;
    }
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let f = d.join("v606.insv");
    let (fps, dur) = probe(&f).unwrap();
    let a = shotlab::Clip::open(&f).unwrap().times[120];
    let spec = Spec {
        files: vec![f],
        photo: false,
        dur,
        t: a,
        view: View::default(),
        look: Look::Perspective,
        stab: None,
        smooth: 0.0,
        follow: None,
        faces: false,
        plates: false,
    };
    let (mut k, ctx, range) = (Keep::default(), egui::Context::default(), [a, a + 0.02]);
    let mut run = |faces, plates| {
        let set = Set {
            faces,
            plates,
            stab: None,
            ..Set::default()
        };
        let t = task(spec.clone(), set, fixture("plates.mp4"), range, false);
        let (_, r) = k
            .privacy(
                &g,
                &cx,
                &t,
                &spec.files,
                None,
                range,
                fps,
                &mut [0; 7],
                &ctx,
            )
            .unwrap();
        eprintln!("faces {faces}, plates {plates}: {}", t.it.note());
        r.into_iter().nth(120).unwrap()
    };
    let want = track::dir(118.1, -8.6);
    let on = |r: &[track::Region]| {
        r.iter()
            .any(|r| r.dir.angle_between(want).to_degrees() < 1.5 && r.size.x > r.size.y)
    };
    let p = run(false, true);
    for r in &p {
        let [y, x] = track::yaw_pitch(r.dir);
        eprintln!(
            "  plate region yaw {y:.1} pitch {x:.1}, {:.1}° × {:.1}°, score {:.2}",
            r.size.x, r.size.y, r.score
        );
    }
    assert!(!on(&p), "a region on the Mercedes's 20 px plate");
    assert!(!on(&run(true, false)), "a face region on the plate");
}

/// Output names: a preset's key while it is unchanged, else the kind; a grab by its time and kind; a photo by its kind
/// alone; the prefix (Shot Lab, follow); numbered past a taken name; an editor timeline by its project's name.
#[test]
fn names() {
    let (_l, g, cx) = gpu();
    let dir = fixture("names").parent().unwrap().join("names");
    let mut e = Exp::new(&g, &cx, &egui::Context::default(), Some(dir.clone()), false);
    let s = Spec {
        files: vec!["/x/VID_1_00_001.insv".into()],
        photo: false,
        dur: 5.0,
        t: 1.234,
        view: View::default(),
        look: Look::Perspective,
        stab: None,
        smooth: 0.5,
        follow: None,
        faces: false,
        plates: false,
    };
    let ph = Spec {
        files: vec!["/x/IMG_1.insp".into()],
        photo: true,
        ..s.clone()
    };
    let d = Set::default();
    let flat = parse(d, "flat1080").unwrap();
    let free = Set {
        preset: None,
        ..flat
    };
    let pano = Set {
        preset: None,
        w: 7680,
        ..d
    };
    let p = |e: &Exp, s: &Spec, x: &Set, grab: bool, pre: &str| {
        let p = e.path(s, x, grab, pre);
        p.file_name().unwrap().to_string_lossy().into_owned()
    };
    assert_eq!(p(&e, &s, &d, false, ""), "VID_1_00_001_pano57.mp4");
    assert_eq!(
        p(&e, &s, &free, false, ""),
        "VID_1_00_001_flat_1920x1080.mp4"
    );
    assert_eq!(
        p(&e, &s, &flat, true, ""),
        "VID_1_00_001_1.23s_flat_1920x1080.jpg"
    );
    assert_eq!(
        p(&e, &s, &pano, true, ""),
        "VID_1_00_001_1.23s_360_7680.jpg"
    );
    assert_eq!(
        p(&e, &s, &d, false, "mosaic_"),
        "VID_1_00_001_mosaic_pano57.mp4"
    );
    assert_eq!(
        p(&e, &s, &free, false, "follow_"),
        "VID_1_00_001_follow_flat_1920x1080.mp4"
    );
    assert_eq!(p(&e, &ph, &d, false, ""), "IMG_1_360_5760.jpg");
    assert_eq!(p(&e, &ph, &flat, false, ""), "IMG_1_flat_1920x1080.jpg");
    assert_eq!(p(&e, &ph, &free, true, ""), "IMG_1_flat_1920x1080.jpg");
    // unnumbered, without the extension, as agents' jobs take it
    assert_eq!(stem_tag("v", &d, None, ""), "v_pano57");
    assert_eq!(stem_tag("v", &pano, Some(3.0), "x_"), "v_x_3.00s_360_7680");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("VID_1_00_001_pano57.mp4"), b"").unwrap();
    assert_eq!(p(&e, &s, &d, false, ""), "VID_1_00_001_pano57-2.mp4");
    // timelines: the job fails at once (no such file); only its name matters
    let mut tl = edit::Timeline::new(edit::Rate { num: 30, den: 1 });
    tl.files.push(edit::File {
        path: "/x/VID_1_00_001.insv".into(),
        dur: 5.0,
    });
    tl.clips.push(edit::Clip::new(0, 0.0, 2.0));
    tl.fix();
    let t = Tl {
        name: "trip".into(),
        tl,
        files: vec![vec![]],
        looks: Default::default(),
        over: vec![],
        denoise: None,
        stab: None,
        smooth: 0.5,
    };
    e.timeline(t.clone(), d).unwrap();
    e.timeline(t.clone(), free).unwrap();
    e.timeline(t, d).unwrap();
    let n: Vec<_> = (e.items.iter())
        .map(|i| i.path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        n,
        [
            "trip_pano57.mp4",
            "trip_flat_1920x1080.mp4",
            "trip_pano57-2.mp4"
        ]
    );
    drop(e);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `parity`'s jobs: three presets with other settings over a range, a grab at the clip's end (flat) and one in 360.
fn cases(file: &Path) -> Vec<(Spec, Set, [f64; 2], bool)> {
    let s = Spec {
        files: vec![file.to_path_buf()],
        photo: false,
        dur: 5.0,
        t: 4.99,
        view: View {
            yaw: 30.0,
            pitch: -10.0,
            fov: 90.0,
            ..View::default()
        },
        look: Look::Perspective,
        stab: None,
        smooth: 0.5,
        follow: None,
        faces: false,
        plates: false,
    };
    let d = Set::default();
    let flat = Set {
        fps: Some((25, 1)),
        amt: 0.7,
        adj: fx::Params {
            exposure: 0.5,
            ..Default::default()
        },
        stab: None,
        colour: false,
        ..parse(d, "flat1080:film:qp=23").unwrap()
    };
    let vert = Set {
        raft: false,
        stab: Some(stab::Lock::Direction),
        smooth: 1.5,
        ..parse(d, "vert1080").unwrap()
    };
    let grab = Set {
        preset: None,
        jq: 85,
        ..flat
    };
    let pano = Set {
        preset: None,
        w: 7680,
        ..d
    };
    vec![
        (s.clone(), d, [1.0, 3.5], false),
        (s.clone(), flat, [0.5, 2.0], false),
        (s.clone(), vert, [4.0, 5.0], false),
        (s.clone(), grab, [0.0, 5.0], true),
        (Spec { t: 1.0, ..s }, pano, [0.0, 5.0], true),
    ]
}

/// FNV-1a, 64 bits.
fn fnv(s: &str) -> u64 {
    (s.bytes()).fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// What `parity` compares of a job (paths as file names): the clip's range, look and frames, the whole timeline as a
/// hash of its Debug, the files, looks, output (a flat view at 0 and 1 s), encoding, seam, colour, stabilisation.
fn fields(j: &mut ::export::Job, n: u64) -> String {
    let name = |p: &Path| p.file_name().unwrap().to_string_lossy().into_owned();
    let ::export::Cut::Edit(e) = &j.cut else {
        panic!("a job of spans")
    };
    let mut tl = e.tl.clone();
    for f in &mut tl.files {
        f.path = name(Path::new(&f.path));
    }
    let c = &tl.clips[0];
    let files: Vec<Vec<_>> = (e.files.iter())
        .map(|f| f.iter().map(|p| name(p)).collect())
        .collect();
    let mut looks: Vec<_> = (e.looks.iter()).map(|(k, c)| (k, &c.title)).collect();
    looks.sort();
    let cut = format!(
        "{:?} {}..{} {:?} {} len {} tl {:016x} files {files:?} looks {looks:?} over {} abr {} denoise {:?}",
        tl.out.fps,
        c.a,
        c.b,
        c.look,
        c.amt,
        tl.len(),
        fnv(&format!("{tl:?}")),
        e.over.len(),
        e.abr,
        e.denoise
    );
    let v = |v: View| {
        format!(
            "{:?} {:?}",
            v.mode,
            [v.yaw, v.pitch, v.roll, v.fov, v.panini, v.planet]
        )
    };
    let out = match &mut j.out {
        ::export::Out::Pano(w) => format!("pano {w}"),
        ::export::Out::Flat { w, h, view } => {
            let f = view.as_mut().expect("a view");
            format!("flat {w}x{h} {} {}", v(f(0.0)), v(f(1.0)))
        }
    };
    format!(
        "{cut} | {out} | {:?} raft {} colour {:?} stab {:?} flat {} fx {} path {} n {n}",
        j.enc,
        j.raft,
        j.colour,
        j.stab,
        j.flat,
        j.fx.is_some(),
        j.path.display()
    )
}

/// `span` and `clip_job` build the jobs `Keep::video` built before they were factored out of it: `cases` at 29.97 fps,
/// compared with what `fields` read from `Keep::video`'s jobs then (grabs since with the default noise reduction).
#[test]
fn parity() {
    const WANT: [&str; 5] = [
        "Rate { num: 30000, den: 1001 } 1..3.5 \"\" 1 len 75 tl 0a044c2975e68409 files [[\"v.insv\"]] looks [] over 0 \
         abr 192000 denoise None | pano 5760 | Video { codec: H265, deep: true, rate: Cbr(120000000), gop: 30 } \
         raft true colour Some(0.1) stab Some((Horizon, 0.5)) flat false fx false path o.mp4 n 75",
        "Rate { num: 25, den: 1 } 0.5..2 \"Film\" 0.7 len 38 tl 053399d64078cf89 files [[\"v.insv\"]] \
         looks [(\"Film\", \"Film\")] over 0 abr 192000 denoise None | flat 1920x1080 \
         Perspective [30.0, -10.0, 0.0, 90.0, 0.0, 1.0] Perspective [30.0, -10.0, 0.0, 90.0, 0.0, 1.0] | \
         Video { codec: H264, deep: false, rate: Qp(23), gop: 25 } raft true colour None stab None flat false \
         fx false path o.mp4 n 38",
        "Rate { num: 30000, den: 1001 } 4..5 \"\" 1 len 30 tl 9e6c7bde94342b30 files [[\"v.insv\"]] looks [] over 0 \
         abr 192000 denoise None | flat 1080x1920 Perspective [30.0, -10.0, 0.0, 90.0, 0.0, 1.0] \
         Perspective [30.0, -10.0, 0.0, 90.0, 0.0, 1.0] | \
         Video { codec: H264, deep: false, rate: Cbr(20000000), gop: 30 } raft false colour Some(0.1) \
         stab Some((Direction, 1.5)) flat false fx false path o.mp4 n 30",
        "Rate { num: 25, den: 1 } 4.96..5 \"Film\" 0.7 len 1 tl 97f2b9a45f78a9dd files [[\"v.insv\"]] \
         looks [(\"Film\", \"Film\")] over 0 abr 192000 denoise None | flat 1920x1080 \
         Perspective [30.0, -10.0, 0.0, 90.0, 0.0, 1.0] Perspective [30.0, -10.0, 0.0, 90.0, 0.0, 1.0] | \
         Jpeg(85, 0.5) raft true colour None stab None flat false fx false path o.mp4 n 1",
        "Rate { num: 30000, den: 1001 } 1..1.0333666666666668 \"\" 1 len 1 tl 330828e57e6f3050 files [[\"v.insv\"]] \
         looks [] over 0 abr 192000 denoise None | pano 7680 | Jpeg(92, 0.5) raft true colour Some(0.1) \
         stab Some((Horizon, 0.5)) flat false fx false path o.mp4 n 1",
    ];
    let src = edit::Rate {
        num: 30000,
        den: 1001,
    };
    for ((s, x, r, grab), want) in cases(Path::new("/x/v.insv")).into_iter().zip(WANT) {
        let ab = span(if grab { [s.t; 2] } else { r }, grab, x.rate(src), s.dur);
        let v = s.view;
        let (mut j, n) = clip_job(
            &s.files,
            s.dur,
            src,
            &x,
            ab,
            grab,
            move |_| v,
            "o.mp4".into(),
            None,
            false,
        );
        assert_eq!(fields(&mut j, n), want, "{x:?}");
    }
}

/// One device for the GPU tests, one test at a time.
pub(super) fn gpu() -> (std::sync::MutexGuard<'static, ()>, Arc<Gpu>, Ctx) {
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

pub(super) fn task(spec: Spec, set: Set, path: PathBuf, range: [f64; 2], grab: bool) -> Task {
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
        dest: Mutex::new(None),
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
        plates: false,
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
pub(super) fn ff_frame(p: &Path, k: usize) -> Vec<u8> {
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
    if no_cf_home() {
        return;
    }
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
        plates: false,
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
    let (_, regs) = (k.privacy(
        &g,
        &cx,
        &t,
        &spec.files,
        None,
        range,
        fps,
        &mut [0; 7],
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
    if no_cf_home() {
        return;
    }
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
        plates: false,
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
    let (_, regs) = (k.privacy(
        &g,
        &cx,
        &t,
        &spec.files,
        None,
        [0.0, 0.3],
        fps,
        &mut [0; 7],
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
    // before the encoder no pixel away from the regions differs, on either GPU; the encoder's other choices around the
    // blurred face spread its quantisation noise into blocks further off: RADV 0.014, NVIDIA (620.73, its H.264 level 2)
    // 0.290 (level 0: 0.248, 0.184 at QP 10, 0.062 at QP 1); a blur reaching 1 % of these pixels (|Δ| about 30, as
    // inside) would add 0.3
    let max = match g.adapter.get_info().driver == "NVIDIA" {
        true => 0.4,
        false => 0.1,
    };
    assert!(
        nout > 10000 && dout < max,
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

/// Denoise on a library export (`Set::denoise`, the editor's `export::Edit::denoise`) of v606 2–5 s (`X5_SAMPLES`),
/// flat 640 × 360 H.264, template seam, no stabilisation. Off, the audio is the source's over the span; on, it is the
/// source run through `media::denoise::Stereo` from the span's start with its delay dropped: in step with it (best lag
/// 0 within ±1200 samples), closer to it than to the source, quieter than off, and it ends with the video, which is
/// as long as off's.
#[test]
fn denoise() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let f = d.join("v606.insv");
    let (_, dur) = probe(&f).unwrap();
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
        plates: false,
    };
    let set = Set {
        preset: None,
        pano: false,
        w: 640,
        h: 360,
        codec: Codec::H264,
        deep: false,
        raft: false,
        colour: false,
        stab: None,
        ..Set::default()
    };
    let (mut k, ctx) = (Keep::default(), egui::Context::default());
    let mut run = |dn: Option<f32>| {
        let p = fixture(&format!("denoise_{}.mp4", dn.is_some()));
        let t = task(
            spec.clone(),
            Set { denoise: dn, ..set },
            p,
            [2.0, 5.0],
            false,
        );
        k.video(&g, &cx, &t, &ctx).unwrap();
        t.it.path.clone()
    };
    let (off, on) = (run(None), run(Some(1.0)));
    let tool = |cmd: &str, a: &[&str]| {
        let o = std::process::Command::new(cmd).args(a).output().expect(cmd);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        o.stdout
    };
    // video and audio ends, s
    let ends = |p: &Path| -> Vec<f64> {
        let a = [
            "-v",
            "error",
            "-show_entries",
            "stream=start_time,duration",
            "-of",
            "csv=p=0",
        ];
        let o = tool("ffprobe", &[&a[..], &[p.to_str().unwrap()]].concat());
        (String::from_utf8_lossy(&o).lines())
            .map(|l| l.split(',').map(|v| v.parse::<f64>().unwrap()).sum())
            .collect()
    };
    // interleaved stereo, 48 kHz, from `pre` (FFmpeg input options)
    let pcm = |p: &Path, pre: &[&str]| -> Vec<f32> {
        let a = [&["-v", "error"], pre, &["-i", p.to_str().unwrap()]].concat();
        let o = tool(
            "ffmpeg",
            &[
                &a[..],
                &["-map", "0:a:0", "-ar", "48000", "-f", "f32le", "-"],
            ]
            .concat(),
        );
        o.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    };
    let avg = |x: &[f32]| -> Vec<f32> {
        x.as_chunks::<2>()
            .0
            .iter()
            .map(|s| 0.5 * (s[0] + s[1]))
            .collect()
    };
    let raw = pcm(&f, &["-ss", "2.0", "-t", "3.2"]);
    const F: usize = media::denoise::FRAME;
    let (mut st, mut wet) = (media::denoise::Stereo::new(), vec![]);
    for c in raw.as_chunks::<{ 2 * F }>().0 {
        let mut l: [f32; F] = std::array::from_fn(|i| c[2 * i]);
        let mut r: [f32; F] = std::array::from_fn(|i| c[2 * i + 1]);
        st.frame(&mut l, &mut r);
        wet.extend(l.iter().zip(&r).map(|(a, b)| 0.5 * (a + b)));
    }
    let wet = &wet[media::denoise::DELAY..];
    let (dry, a, b) = (avg(&raw), avg(&pcm(&off, &[])), avg(&pcm(&on, &[])));
    // 2 s from 0.5 s in, lags up to ±1200 samples (the denoiser's delay is 960): the best lag, its correlation
    let (o, n, m) = (24000usize, 96000usize, 1200isize);
    let xc = |x: &[f32], y: &[f32]| {
        let c = |l: isize| {
            let (mut s, mut p, mut q) = (0f64, 0f64, 0f64);
            for i in o..o + n {
                let (u, v) = (x[i] as f64, y[(i as isize + l) as usize] as f64);
                (s, p, q) = (s + u * v, p + u * u, q + v * v);
            }
            s / (p * q).sqrt().max(1e-30)
        };
        (-m..=m)
            .map(|l| (l, c(l)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
    };
    let rms = |x: &[f32]| {
        (x[o..o + n].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt()
    };
    let ((lo, co), (lw, cw), (ld, cd)) = (xc(&a, &dry), xc(&b, wet), xc(&b, &dry));
    let (eo, en) = (ends(&off), ends(&on));
    eprintln!(
        "library export, denoise off: video / audio end {eo:?} s, against the source lag {lo} correlation {co:.4}, \
         level {:.4}; on: ends {en:?} s, against the source denoised lag {lw} correlation {cw:.4}, against it dry \
         lag {ld} correlation {cd:.4}, level {:.4}",
        rms(&a),
        rms(&b)
    );
    assert!(lo == 0 && co > 0.99, "off: not the source's audio");
    assert!(
        lw == 0 && cw > 0.95 && cw > cd + 0.05,
        "on: not the denoised source"
    );
    assert!(rms(&b) < 0.8 * rms(&a), "on: not quieter");
    assert!(
        (en[0] - eo[0]).abs() < 1e-6 && (en[1] - en[0]).abs() <= 1.0 / 48000.0,
        "on: length"
    );
    for p in [off, on] {
        std::fs::remove_file(p).unwrap();
    }
}

/// The surface-configure contract (main.rs `GATE`, `BURN`) on v635 (`X5_SAMPLES`): while a RAFT export with face
/// blur runs (the face pass, then crate export's renderer, decoders and encoder), holding BURN, then GATE exclusively
/// as the UI's retry does leaves the queue idle with no submit meanwhile (wgpu-core's configure check: `poll` waits
/// for the last submission and finds the queue empty), and the job still finishes. Every other check goes without
/// the locks and must see the job's submits, or the check proves nothing.
#[test]
fn gate() {
    if no_cf_home() {
        return;
    }
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

/// A pass's regions per frame as bits (direction, size, score), to compare runs exactly.
fn bits(r: &Regs) -> Vec<Vec<[u32; 6]>> {
    let b = |q: &track::Region| {
        let ([x, y, z], [w, h]) = (q.dir.to_array(), q.size.to_array());
        [x, y, z, w, h, q.score].map(f32::to_bits)
    };
    r.iter().map(|r| r.iter().map(b).collect()).collect()
}

/// The privacy pass before it was factored out of `Keep::privacy`: a verbatim copy of crates/app/src/export.rs
/// 1338-1448 at 5660ea9, so `privacy_parity` judges the refactor against the code it replaced.
impl Keep {
    #[allow(clippy::too_many_arguments)]
    fn old_privacy(
        &mut self,
        g: &Arc<Gpu>,
        cx: &Ctx,
        t: &Task,
        f: &[PathBuf],
        stab: Option<(stab::Lock, f64)>,
        [a, b]: [f64; 2],
        fps: edit::Rate,
        sum: &mut [usize; 7],
        ctx: &egui::Context,
    ) -> Result<(Vec<f64>, Regs), String> {
        if f.len() > 1 {
            return Err("privacy blur: recordings in a file per lens are not supported yet".into());
        }
        let c = shotlab::Clip::open(&f[0])?;
        let poses = match stab {
            Some((l, s)) => Some(
                c.poses(l, s)
                    .ok_or("privacy blur: no IMU record to stabilise with")?,
            ),
            None => None,
        };
        let (fa, pl) = (t.set.faces, t.set.plates);
        let what = match (fa, pl) {
            (true, false) => "faces",
            (false, true) => "license plates",
            _ => "faces and license plates",
        };
        let k0 = clip::frame_at(&c.times, a);
        let k1 = c.times.partition_point(|&x| x < b - 1e-3).max(k0 + 1) - 1;
        let ks: Vec<usize> = (k0..=k1).step_by(STEP).collect();
        let (it, n, tiles) = (&t.it, ks.len(), track::face_tiles());
        let (mut hide, mut pano, mut i, mut found) = (
            [track::Hide::new(), track::Hide::plates(PLATES_W)],
            None,
            0,
            [0; 2],
        );
        it.set(St::Pass(0, n, what));
        crate::ai::MODELS.with(g, false, Some(ctx), |ai| {
            // shotlab decodes and submits each stitch on this thread between the calls: a surface configure the UI
            // retries waits for that (no wait there depends on another thread); the detection runs under BURN instead,
            // never inside GATE (the UI takes BURN, then GATE)
            let mut gate = Some(crate::GATE.read());
            c.frames(
                g,
                cx,
                if pl { PLATES_W } else { FACES_W },
                false,
                &ks,
                |k| c.pose(k, poses.as_deref()),
                |tex, k| {
                    if it.stop.load(Relaxed) {
                        return Err(CANCELLED.into());
                    }
                    let p =
                        pano.get_or_insert_with(|| ai.pano(&tex.create_view(&Default::default())));
                    gate.take();
                    // burn submits throughout
                    let r = {
                        let _b = crate::BURN.lock();
                        let f = if fa {
                            ai.faces(p, &tiles, MIN)
                        } else {
                            Ok(vec![])
                        };
                        let q = if pl { ai.plates(p, PMIN) } else { Ok(vec![]) };
                        f.and_then(|f| Ok([f, q?]))
                    };
                    gate = Some(crate::GATE.read());
                    for (j, r) in r?.into_iter().enumerate() {
                        found[j] += r.len();
                        hide[j].push(k, r);
                    }
                    i += 1;
                    it.set(St::Pass(i, n, what));
                    ctx.request_repaint();
                    Ok(())
                },
            )
        })?;
        let fps = fps.num as f32 / fps.den as f32;
        let [rf, rp] = hide.each_ref().map(|h| h.regions(c.times.len(), fps));
        let r: Regs = rf
            .into_iter()
            .zip(rp)
            .map(|(a, b)| [a, b].concat())
            .collect();
        let on = r[k0..=k1].iter().filter(|r| !r.is_empty()).count();
        let tr = hide.each_ref().map(|h| h.tracks(fps).len());
        for (s, x) in sum
            .iter_mut()
            .zip([found[0], found[1], n, tr[0], tr[1], on, k1 + 1 - k0])
        {
            *s += x;
        }
        let [ff, fp, n, tf, tp, on, all] = *sum;
        let (d, tr) = match (fa, pl) {
            (true, false) => (format!("{ff} face"), tf.to_string()),
            (false, true) => (format!("{fp} plate"), tp.to_string()),
            _ => (
                format!("{ff} face and {fp} plate"),
                format!("{tf} and {tp}"),
            ),
        };
        *it.note.lock().unwrap_or_else(|e| e.into_inner()) =
            format!("{d} detections on {n} frames, {tr} tracks, blurred on {on} of {all} frames");
        Ok((c.times, r))
    }
}

/// A privacy pass on the synthetic recording (300 frames of grey per GOP, no IMU) over [0, 2] s with faces and plates:
/// 20 frames searched of 60, all 300 frame times, nothing found, the note and the last state the Export screen shows
/// (recorded from `Keep::privacy` before the pass was factored out of it). A recording in a file per lens, and a
/// stabilisation without IMU records, are refused.
#[test]
fn privacy_synth() {
    if no_cf_home() {
        return;
    }
    let (_l, g, cx) = gpu();
    let f = fixture("privacy_synth.insv");
    ::export::synth::insv(&f, 300, 256).unwrap();
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
        plates: false,
    };
    let set = Set {
        faces: true,
        plates: true,
        stab: None,
        ..Set::default()
    };
    let (mut k, ctx, ab) = (Keep::default(), egui::Context::default(), [0.0, 2.0]);
    let t = task(spec.clone(), set, fixture("privacy_synth.mp4"), ab, false);
    let mut sum = [0; 7];
    let (times, r) = (k.privacy(&g, &cx, &t, &spec.files, None, ab, fps, &mut sum, &ctx)).unwrap();
    assert_eq!((times.len(), r.len()), (300, 300));
    assert!(r.iter().all(|r| r.is_empty()), "a region on grey");
    assert_eq!(sum, [0, 0, 20, 0, 0, 0, 60]);
    assert_eq!(t.it.st(), St::Pass(20, 20, "faces and license plates"));
    assert_eq!(
        t.it.note(),
        "0 face and 0 plate detections on 20 frames, 0 and 0 tracks, blurred on 0 of 60 frames"
    );
    let two = [f.clone(), f.clone()];
    let e = k.privacy(&g, &cx, &t, &two, None, ab, fps, &mut sum, &ctx);
    assert_eq!(
        e.unwrap_err(),
        "privacy blur: recordings in a file per lens are not supported yet"
    );
    let horizon = Some((stab::Lock::Horizon, gyro::SMOOTH));
    let e = k.privacy(&g, &cx, &t, &spec.files, horizon, ab, fps, &mut sum, &ctx);
    assert_eq!(
        e.unwrap_err(),
        "privacy blur: no IMU record to stabilise with"
    );
    // the pass itself with either hold: the same, its progress from 0 of 20 to 20 of 20, a stop seen at the next frame
    for hold in [Hold::Pass, Hold::Frame] {
        let (stop, mut said) = (AtomicBool::new(false), vec![]);
        let pass = |f: &[PathBuf], stab, say: &mut dyn FnMut(usize, usize)| {
            privacy_pass(&g, &cx, f, stab, ab, fps, [true; 2], hold, &stop, None, say)
        };
        let (times, r, c) = pass(&spec.files, None, &mut |i, n| said.push((i, n))).unwrap();
        assert_eq!(
            (times.len(), r.len(), c),
            (300, 300, [0, 0, 20, 0, 0, 0, 60]),
            "{hold:?}"
        );
        assert!(r.iter().all(|r| r.is_empty()), "{hold:?}: a region on grey");
        assert_eq!(
            said,
            (0..=20).map(|i| (i, 20)).collect::<Vec<_>>(),
            "{hold:?}"
        );
        let mut last = 0;
        let e = pass(&spec.files, None, &mut |i, _| {
            last = i;
            stop.store(i == 5, Relaxed);
        });
        assert_eq!((e.map(drop), last), (Err(CANCELLED.into()), 5), "{hold:?}");
        stop.store(false, Relaxed);
        let e = pass(&two, None, &mut |_, _| {});
        assert_eq!(
            e.map(drop).unwrap_err(),
            "privacy blur: recordings in a file per lens are not supported yet"
        );
        let e = pass(&spec.files, horizon, &mut |_, _| {});
        assert_eq!(
            e.map(drop).unwrap_err(),
            "privacy blur: no IMU record to stabilise with"
        );
    }
    std::fs::remove_file(&f).unwrap();
}

/// What a privacy pass gives: frame times, regions per frame (bits), note, last state, counts.
type Pass = (Vec<f64>, Vec<Vec<[u32; 6]>>, String, St, [usize; 7]);

/// Where two passes differ, if they do.
fn differ(a: &Pass, b: &Pass) -> Option<String> {
    if a.0 != b.0 {
        return Some("the frame times".into());
    }
    if let Some(k) = (0..a.1.len().max(b.1.len())).find(|&k| a.1.get(k) != b.1.get(k)) {
        return Some(format!(
            "frame {k}'s regions: {:?} and {:?}",
            a.1.get(k),
            b.1.get(k)
        ));
    }
    (a.2 != b.2 || a.3 != b.3 || a.4 != b.4).then(|| {
        format!(
            "{:?} {:?} {:?} and {:?} {:?} {:?}",
            a.2, a.3, a.4, b.2, b.3, b.4
        )
    })
}

/// The recording the privacy tests take from `X5_SAMPLES` (`d`): v635, else its first `.insv`.
pub(super) fn sample(d: &Path) -> PathBuf {
    match d.join("v635.insv") {
        f if f.exists() => f,
        _ => {
            let mut v: Vec<PathBuf> = (std::fs::read_dir(d).unwrap())
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|x| x == "insv"))
                .collect();
            v.sort();
            v.into_iter().next().expect("an .insv in X5_SAMPLES")
        }
    }
}

/// `Keep::privacy` finds what the pass found before it was factored out of it (`old_privacy`), on a recording
/// (`X5_SAMPLES`: v635, else its first `.insv`): faces over [0, 3] s, faces and plates over [1, 2] s with the horizon
/// lock, plates over the second from `X5_PLATES_AT` (s, 1 by default; one with a plate: a second, as `Hide::plates`
/// hides no plate found on one frame alone). Each kind is found in a case at least; the same frame times, regions (by their bits),
/// note, last state and counts. The old pass runs twice: if those differ, the detectors aren't deterministic in this
/// process and the test says so instead of comparing.
#[test]
fn privacy_parity() {
    if no_cf_home() {
        return;
    }
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let f = sample(&d);
    let at: f64 =
        std::env::var("X5_PLATES_AT").map_or(1.0, |s| s.parse().expect("X5_PLATES_AT: seconds"));
    let (_l, g, cx) = gpu();
    let (fps, dur) = probe(&f).unwrap();
    let spec = Spec {
        files: vec![f],
        photo: false,
        dur,
        t: at,
        view: View::default(),
        look: Look::Perspective,
        stab: None,
        smooth: 0.0,
        follow: None,
        faces: false,
        plates: false,
    };
    let horizon = Some((stab::Lock::Horizon, gyro::SMOOTH));
    let cases = [
        ([true, false], None, [0.0, 3.0]),
        ([true, true], horizon, [1.0, 2.0]),
        ([false, true], None, [at, (at + 1.0).min(dur)]),
    ];
    let (mut k, ctx) = (Keep::default(), egui::Context::default());
    let mut found = [false; 2];
    for ([faces, plates], stab, ab) in cases {
        let mut run = |old: bool| -> Pass {
            let set = Set {
                faces,
                plates,
                stab: stab.map(|s| s.0),
                ..Set::default()
            };
            let t = task(spec.clone(), set, fixture("privacy.mp4"), ab, false);
            let (fs, mut sum) = (&spec.files, [0; 7]);
            let (times, r) = match old {
                true => k.old_privacy(&g, &cx, &t, fs, stab, ab, fps, &mut sum, &ctx),
                false => k.privacy(&g, &cx, &t, fs, stab, ab, fps, &mut sum, &ctx),
            }
            .unwrap();
            (times, bits(&r), t.it.note(), t.it.st(), sum)
        };
        let t0 = Instant::now();
        let old = run(true);
        eprintln!(
            "faces {faces}, plates {plates}, {ab:?} s, stab {stab:?}: {} ({:.1} s)",
            old.2,
            t0.elapsed().as_secs_f64()
        );
        if let Some(e) = differ(&old, &run(true)) {
            panic!(
                "the detectors aren't deterministic in this process: two runs of the old pass differ in {e}"
            );
        }
        if let Some(e) = differ(&old, &run(false)) {
            panic!("Keep::privacy differs from the old pass in {e}");
        }
        // the agents' pass: the models held per searched frame, no note
        let (mut said, stop) = ((0, 0), AtomicBool::new(false));
        let (fs, hold) = (&spec.files, Hold::Frame);
        let (times, r, c) = privacy_pass(
            &g,
            &cx,
            fs,
            stab,
            ab,
            fps,
            [faces, plates],
            hold,
            &stop,
            None,
            &mut |i, n| said = (i, n),
        )
        .unwrap();
        let St::Pass(.., what) = old.3 else {
            panic!("{:?}", old.3)
        };
        let frame = (
            times,
            bits(&r),
            old.2.clone(),
            St::Pass(said.0, said.1, what),
            c,
        );
        if let Some(e) = differ(&old, &frame) {
            panic!("the agents' pass (Hold::Frame) differs from the old pass in {e}");
        }
        found = [found[0] | (old.4[0] > 0), found[1] | (old.4[1] > 0)];
    }
    // each kind on real detections in a case at least, or it shows no parity; not in every case: the plates of the
    // horizon-locked one sit at the detector's edge and come and go with the stitch (0 to 3 detections)
    assert!(
        found == [true; 2],
        "faces, plates found: {found:?} on {}: no parity shown; X5_PLATES_AT: a time with a plate",
        spec.files[0].display()
    );
}
