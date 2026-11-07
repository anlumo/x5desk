use super::*;

fn want(kind: Kind) -> Want {
    Want {
        preset: (kind == Kind::Export).then(|| "pano57".into()),
        stabilisation: Some("horizon".into()),
        ..Want::default()
    }
}

/// The catalog's ids are the app's tables': looks in `fx::Look::ALL`'s order with the test hook's first words and the
/// display names; the player's projections but the equirect; `gyro::parse`'s modes; the editor's adjustments with
/// their defaults; Shot Lab's effects; the presets, the 360 video widths without the photos' 11904.
#[test]
fn catalog_tables() {
    let c = catalog();
    assert_eq!(c.looks.len(), fx::Look::ALL.len());
    for (n, l) in c.looks.iter().zip(fx::Look::ALL) {
        assert_eq!(n.name.as_deref(), Some(l.name()));
        // the test hook's spelling (`parse`) finds the same look
        let first = l.name().split(' ').next().unwrap().to_ascii_lowercase();
        assert!(n.id == first || n.aliases == [first.clone()], "{n:?}");
        let x = super::super::parse(Set::default(), &format!("pano57:{first}")).unwrap();
        assert_eq!(x.look, Some(l));
    }
    let ids: Vec<&str> = c.looks.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids, LOOKS);
    let flat: Vec<player::Look> = (player::Look::ALL.iter())
        .map(|l| l.0)
        .filter(|&l| l != player::Look::Equirect)
        .collect();
    assert_eq!(VIEWS.map(|v| v.2).to_vec(), flat);
    for (p, v) in c.projections.iter().zip(VIEWS) {
        assert_eq!(player::Look::parse(&p.id), Some(v.2));
        assert_eq!(p.fov, [v.2.fov().0, v.2.fov().1]);
        assert_eq!(p.default_fov, v.2.view().fov);
    }
    assert_eq!(c.projections[0].aliases, ["persp"]);
    assert_eq!(c.projections[2].pitch, -90.0);
    for (i, s) in c.stabilisation.iter().enumerate() {
        assert_eq!(gyro::parse(s), Some(gyro::MODES[i].0));
    }
    assert_eq!(c.stabilisation.len(), gyro::MODES.len());
    assert_eq!(c.adjust.len(), doc::ADJ.len());
    let reach = c.adjust.iter().find(|a| a.id == "reach").unwrap();
    assert_eq!((reach.default, reach.min, reach.max), (4.0, 1.0, 12.0));
    assert!(
        c.adjust
            .iter()
            .filter(|a| a.id != "reach")
            .all(|a| a.default == 0.0)
    );
    let lab = c.later.iter().find(|l| l.id == "shot_lab").unwrap();
    assert_eq!(lab.effects.len(), lab::FXS.len());
    assert!(lab.effects.iter().zip(lab::FXS).all(|(e, f)| e == f.2));
    for a in ["faces", "plates"] {
        assert!(c.later.iter().any(|l| l.aliases.contains(&a.to_string())));
    }
    assert_eq!(c.presets.len(), PRESETS.len());
    assert_eq!(
        (&c.presets[0].id, c.presets[0].output, &c.presets[0].codec),
        (&"pano57".to_string(), Output::Pano, &"h265".to_string())
    );
    assert_eq!(c.default_preset, "pano57");
    assert_eq!(c.pano_widths, [3840, 5760, 7680]);
    assert_eq!(c.flat_sizes.len(), FLAT.len());
    assert_eq!(c.fps[0], "source");
    assert!(c.fps.contains(&"29.97".to_string()));
    assert_eq!((c.still.pano_width, c.still.flat), (7680, [1920, 1080]));
}

/// Stills start from preset-free defaults whatever the preset, exports from theirs; a changed size or rate clears the
/// preset as on the screen; looks, adjustments, the view and the rate land where the screen keeps them.
#[test]
fn set_of_wants() {
    let (x, _, p) = set_of(&want(Kind::Still), Kind::Still, true).unwrap();
    assert_eq!(
        (x.preset, x.pano, x.size(), x.jq, x.raft, x.faces, x.plates),
        (None, true, [7680, 3840], 92, true, false, false)
    );
    assert_eq!(p, "perspective");
    let w = Want {
        output: Some(Output::Flat),
        jpeg_quality: Some(70),
        view: Some(r::View {
            projection: Some("persp".into()),
            yaw: Some(30.0),
            ..Default::default()
        }),
        ..want(Kind::Still)
    };
    let (x, v, p) = set_of(&w, Kind::Still, false).unwrap();
    assert_eq!(
        (x.pano, x.size(), x.jq, x.raft),
        (false, [1920, 1080], 70, false)
    );
    assert_eq!(
        (p, v.yaw, v.fov, v.pitch),
        ("perspective", 30.0, 100.0, 0.0)
    );
    let w = Want {
        output: Some(Output::Flat),
        view: Some(r::View {
            projection: Some("planet".into()),
            fov: Some(300.0),
            ..Default::default()
        }),
        ..want(Kind::Still)
    };
    let (_, v, _) = set_of(&w, Kind::Still, true).unwrap();
    assert_eq!(
        (v.mode, v.pitch, v.fov),
        (stitch::Mode::Planet, -90.0, 300.0)
    );

    let (x, _, _) = set_of(&want(Kind::Export), Kind::Export, true).unwrap();
    assert_eq!(
        (x.preset, x.size(), x.codec, x.deep, x.mbps, x.qp),
        (Some(0), [5760, 2880], Codec::H265, true, 120.0, None)
    );
    let w = Want {
        width: Some(5760),
        height: Some(2880),
        look: Some("teal_orange".into()),
        adjust: [("exposure".to_string(), 0.5), ("reach".to_string(), 8.0)].into(),
        ..want(Kind::Export)
    };
    let (x, _, _) = set_of(&w, Kind::Export, true).unwrap();
    // the preset's own size keeps it
    assert_eq!(x.preset, Some(0));
    assert_eq!(x.look, Some(fx::Look::TealOrange));
    assert_eq!(
        (x.adj.exposure, x.adj.reach, x.adj.contrast),
        (0.5, 8.0, 0.0)
    );
    for w in [
        Want {
            width: Some(7680),
            height: Some(3840),
            ..want(Kind::Export)
        },
        Want {
            qp: Some(24),
            ..want(Kind::Export)
        },
        Want {
            codec: Some("h264".into()),
            width: Some(3840),
            height: Some(1920),
            ..want(Kind::Export)
        },
    ] {
        let (x, _, _) = set_of(&w, Kind::Export, true).unwrap();
        assert_eq!(x.preset, None, "{w:?}");
        assert!(stem_tag("v", &x, None, "").starts_with("v_360_"));
    }
    // H.264 has no 10-bit: the screen's checkbox goes off
    let w = Want {
        preset: Some("flat1080".into()),
        codec: Some("h264".into()),
        ten_bit: Some(true),
        fps: Some("29.97".into()),
        stabilisation: Some("off".into()),
        ..Want::default()
    };
    let (x, _, _) = set_of(&w, Kind::Export, true).unwrap();
    assert_eq!(
        (x.preset, x.deep, x.fps, x.stab, x.pano),
        (Some(4), false, Some((30000, 1001)), None, false)
    );
    assert_eq!(stem_tag("v", &x, None, ""), "v_flat1080");
    // a 360 preset turned flat: the screen's 1920 × 1080, no preset
    let w = Want {
        output: Some(Output::Flat),
        ..want(Kind::Export)
    };
    let (x, _, _) = set_of(&w, Kind::Export, true).unwrap();
    assert_eq!((x.preset, x.pano, x.size()), (None, false, [1920, 1080]));
    assert_eq!(stem_tag("v", &x, None, ""), "v_flat_1920x1080");
    let w = Want {
        look: Some("sepia".into()),
        ..want(Kind::Export)
    };
    assert!(
        set_of(&w, Kind::Export, true)
            .unwrap_err()
            .contains("sepia")
    );
}

/// A still's time within the clip, an export's range at least a frame and not past the end: no clamping.
#[test]
fn ranges() {
    let fps = edit::Rate {
        num: 30000,
        den: 1001,
    };
    let still = |t| Want {
        time_s: Some(t),
        ..want(Kind::Still)
    };
    assert_eq!(range(&still(2.5), Kind::Still, fps, 5.0), Ok([2.5; 2]));
    assert!(
        range(&still(5.0), Kind::Still, fps, 5.0)
            .unwrap_err()
            .contains("5.000 s long")
    );
    let all = range(&want(Kind::Export), Kind::Export, fps, 5.0).unwrap();
    assert_eq!(all, [0.0, 5.0]);
    let part = |a, b| Want {
        start_s: a,
        end_s: b,
        ..want(Kind::Export)
    };
    assert_eq!(
        range(&part(Some(1.0), Some(2.0)), Kind::Export, fps, 5.0),
        Ok([1.0, 2.0])
    );
    assert!(
        range(&part(None, Some(5.5)), Kind::Export, fps, 5.0)
            .unwrap_err()
            .contains("end_s")
    );
    assert!(
        range(&part(Some(4.99), None), Kind::Export, fps, 5.0)
            .unwrap_err()
            .contains("at least a frame")
    );
    // a frame's worth is enough
    assert!(range(&part(Some(4.96), None), Kind::Export, fps, 5.0).is_ok());
}

/// Results show the settings as resolved: a still's time and quality, an export's preset, rate and range, the flat
/// view; the keys the export time guess compares.
#[test]
fn shown() {
    let w = Want {
        output: Some(Output::Flat),
        ..want(Kind::Still)
    };
    let (x, v, p) = set_of(&w, Kind::Still, true).unwrap();
    let s = settings(&x, &v, p, &w, Kind::Still, [3.0, 3.03], 3.0);
    assert_eq!(s["output"], "flat");
    assert_eq!(
        (s["width"].as_u64(), s["height"].as_u64()),
        (Some(1920), Some(1080))
    );
    assert_eq!(
        (&s["time_s"], &s["jpeg_quality"]),
        (&json!(3.0), &json!(92))
    );
    assert_eq!(s["view"]["projection"], "perspective");
    assert_eq!(s["stabilisation"], "horizon");
    assert!(s.get("preset").is_none());
    let w = Want {
        qp: Some(24),
        ..want(Kind::Export)
    };
    let (x, v, p) = set_of(&w, Kind::Export, true).unwrap();
    let s = settings(&x, &v, p, &w, Kind::Export, [0.0, 5.0], 0.0);
    assert_eq!(
        (&s["preset"], &s["qp"], &s["mbps"]),
        (&json!(null), &json!(24), &json!(null))
    );
    assert_eq!(
        (&s["fps"], &s["end_s"], &s["raft"]),
        (&json!("source"), &json!(5.0), &json!(true))
    );
    assert_eq!(
        (&s["codec"], &s["ten_bit"], &s["width"]),
        (&json!("h265"), &json!(true), &json!(5760))
    );
    assert!(s.get("view").is_none() && s.get("jpeg_quality").is_none());
}

/// mcp finds a strict export's failure at a picture that could not be decoded by export's own words, and names
/// `allow_frozen` after it.
#[test]
fn concealed_words() {
    assert_eq!(::export::CONCEALED, r::CONCEALED);
}

/// The end `_exit`s while a thread may be in the driver (`busy`): a probe or trim running (and none starts once it
/// has looked), a lane in a job of an engine a failed probe made `Dead`. A RAFT run that finished warms the engine,
/// though the poller never saw it past frame 0 (`compiling` no more). On this machine's GPU; the lane is held in its
/// job by its flat view, the run says RAFT (the job has none, for speed).
#[test]
fn in_driver() {
    let a = Agent::new(true);
    assert_eq!(a.driver(|| a.busy()), Some(true), "a probe");
    assert!(!a.busy());
    assert_eq!(a.driver(|| ()), None, "a probe after the end");
    let d = std::env::temp_dir().join(format!("x5-agent-driver-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let src = d.join("synth.insv");
    ::export::synth::insv(&src, 30, 256).unwrap();
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let lane = |n| LaneQ {
        q: ::export::Queue::with(g.clone(), cx.clone(), n).unwrap(),
        wedged: Mutex::new(None),
    };
    let rd = Arc::new(Ready {
        g: g.clone(),
        info: base(&g),
        still: lane("x5-test-still"),
        export: lane("x5-test-export"),
        warm: AtomicBool::new(false),
    });
    let w = Want {
        output: Some(Output::Flat),
        width: Some(640),
        height: Some(360),
        codec: Some("h264".into()),
        mbps: Some(8.0),
        raft: Some(false),
        stabilisation: Some("off".into()),
        ..want(Kind::Export)
    };
    let (x, v, _) = set_of(&w, Kind::Export, false).unwrap();
    let (rate, dur) = probe(&src).unwrap();
    let fps = x.rate(rate);
    let ab = span(range(&w, Kind::Export, fps, dur).unwrap(), false, fps, dur);
    let go = Arc::new(AtomicBool::new(false));
    let held = go.clone();
    let at = move |_| {
        while !held.load(Relaxed) {
            thread::sleep(Duration::from_millis(10))
        }
        v
    };
    let dst = d.join("held.mp4");
    let (job, _) = clip_job(
        &[src],
        dur,
        rate,
        &x,
        ab,
        false,
        at,
        dst.clone(),
        None,
        false,
    );
    let id = rd.export.q.push(job);
    let t0 = Instant::now();
    // running, not just pushed: busy counts a job from its push
    while !matches!(rd.export.q.state(id), Some(export::State::Running(_))) {
        assert!(t0.elapsed() < Duration::from_secs(20), "never ran");
        thread::sleep(Duration::from_millis(10))
    }
    let b = Agent::new(true);
    let _ = b.dev.set(rd.clone());
    *lock(&b.st) = Eng::Dead("the GPU's probe: timed out".into());
    assert!(b.busy(), "a Dead engine's lane in its job");
    assert_eq!(
        b.driver(|| ()),
        None,
        "a probe after an end that found a lane busy"
    );
    let run = AgentRun {
        r: rd.clone(),
        lane: Lane::Export,
        id,
        raft: true,
        dst: dst.clone(),
        cut: AtomicBool::new(false),
        t: Mutex::new((Instant::now(), None, None)),
    };
    go.store(true, Relaxed);
    while !matches!(rd.export.q.state(id), Some(::export::State::Done(_))) {
        let s = rd.export.q.state(id);
        assert!(t0.elapsed() < Duration::from_secs(60), "{s:?}");
        assert!(!matches!(s, Some(::export::State::Failed(_))), "{s:?}");
        thread::sleep(Duration::from_millis(10))
    }
    assert!(matches!(run.state(), RunSt::Done { .. }));
    assert!(
        rd.warm.load(Relaxed),
        "a RAFT run done: the kernels compiled"
    );
    drop(run);
    std::fs::remove_dir_all(&d).unwrap();
}

/// An export's size estimate is never below a real file's (this Mac's VideoToolbox on the X5's footage, mostly 3 s at
/// 29.97 fps: 90 frames, AAC included; at a set rate too, the preset's 120 Mbit/s on a flat 1280 × 720 among them),
/// nor far above it (a 360's at QP 24 or lower within 3 times, every one of 72 frames or more within 12), and a QP
/// export's grows with the size and the frame rate asked.
#[test]
fn sizes() {
    let flat = |w, h, c: &str| Want {
        output: Some(Output::Flat),
        width: Some(w),
        height: Some(h),
        codec: Some(c.into()),
        ..want(Kind::Export)
    };
    let pre = |p: &str| Want {
        preset: Some(p.into()),
        ..want(Kind::Export)
    };
    let qp = |w: Want, q| Want { qp: Some(q), ..w };
    let mbps = |w: Want, m| Want { mbps: Some(m), ..w };
    let fps = |w: Want, f: &str| Want {
        fps: Some(f.into()),
        ..w
    };
    let est = |w: &Want, n| {
        let f = w.fps.as_deref().map_or(29.97, |f| f.parse().unwrap());
        video_bytes(&set_of(w, Kind::Export, false).unwrap().0, n, n as f64 / f)
    };
    let real = [
        (flat(1280, 720, "h264"), 90, 45_789_958),
        (mbps(flat(1280, 720, "h264"), 1.0), 90, 470_652),
        (mbps(flat(1280, 720, "h264"), 3.0), 90, 1_253_523),
        (mbps(pre("pano4k"), 5.0), 90, 2_013_146),
        (mbps(pre("pano8k"), 400.0), 90, 152_583_058),
        (qp(flat(1280, 720, "h264"), 24), 90, 6_574_993),
        (qp(flat(1280, 720, "h264"), 10), 90, 25_416_851),
        (qp(flat(1280, 720, "h264"), 34), 90, 1_860_058),
        (qp(flat(1280, 720, "h264"), 45), 90, 485_072),
        (qp(flat(1280, 720, "h265"), 10), 90, 22_661_974),
        (qp(fps(flat(1280, 720, "h264"), "60"), 10), 180, 29_547_580),
        (qp(fps(flat(1280, 720, "h264"), "24"), 24), 72, 5_680_772),
        (qp(flat(1920, 1080, "h264"), 24), 90, 11_069_768),
        (qp(flat(1920, 1080, "h264"), 10), 90, 45_540_570),
        (qp(flat(1080, 1920, "h264"), 24), 90, 13_711_086),
        // a view of the ground; one frame
        (qp(flat(1080, 1080, "h264"), 24), 90, 15_071_674),
        (qp(flat(1080, 1080, "h264"), 10), 90, 47_421_199),
        (qp(flat(1080, 1080, "h264"), 10), 1, 857_484),
        (mbps(flat(1280, 720, "h264"), 1.0), 1, 24_587),
        (qp(pre("pano4k264"), 10), 1, 4_147_050),
        (qp(flat(3840, 2160, "h265"), 10), 90, 74_070_052),
        (qp(pre("pano4k264"), 24), 90, 53_147_600),
        (qp(pre("pano4k264"), 10), 90, 211_921_376),
        (qp(pre("pano4k"), 24), 90, 47_637_404),
        (qp(pre("pano4k"), 10), 90, 176_862_319),
        (qp(pre("pano57"), 24), 90, 71_579_366),
        (qp(pre("pano57"), 10), 90, 250_930_437),
        (qp(pre("pano57"), 45), 90, 4_730_489),
        (qp(pre("pano8k"), 24), 90, 98_652_045),
        (qp(pre("pano8k"), 10), 90, 360_809_459),
        // 120 s in: planets (pitch -90), the 360s at their densest
        (qp(flat(1280, 720, "h264"), 24), 90, 16_217_224),
        (qp(flat(1080, 1080, "h264"), 24), 90, 18_619_274),
        (qp(flat(1920, 1080, "h264"), 24), 90, 27_015_983),
        (qp(flat(2560, 1440, "h264"), 24), 90, 37_489_192),
        (qp(flat(3840, 2160, "h264"), 24), 90, 60_480_281),
        (qp(flat(1080, 1080, "h264"), 10), 90, 54_332_396),
        (qp(pre("pano57"), 10), 90, 273_632_402),
        (qp(pre("pano8k"), 10), 90, 414_773_146),
    ];
    for (w, n, b) in &real {
        let e = est(w, *n);
        assert!(e >= *b, "{e} < {b}: {w:?}");
        assert!(*n < 72 || e <= 12 * b, "{e} > 12 × {b}: {w:?}");
        let pano = set_of(w, Kind::Export, false).unwrap().0.pano;
        if pano && *n >= 72 && w.qp.is_some_and(|q| q <= 24) {
            assert!(e <= 3 * b, "{e} > 3 × {b}: {w:?}");
        }
    }
    let q = |w| est(&qp(w, 24), 90);
    assert!(q(flat(1920, 1080, "h264")) > 2 * q(flat(1280, 720, "h264")));
    assert!(q(pre("pano8k")) > 2 * q(pre("pano4k")));
    let at60 = est(&qp(fps(flat(1280, 720, "h264"), "60"), 24), 180);
    assert!(at60 > q(flat(1280, 720, "h264")) * 19 / 10);
}

/// No Vulkan device on Linux: the hint names what the render nodes show (none, none to open, open ones).
#[cfg(unix)]
#[test]
fn hints() {
    use std::os::unix::fs::PermissionsExt;
    let d = std::env::temp_dir().join(format!("x5desk-dri-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir(&d).unwrap();
    let e = || "No suitable graphics adapter found".to_string();
    assert!(
        hint_in(e(), &d)
            .ends_with("no GPU driver here (a VM or container needs the GPU passed through)")
    );
    let n = d.join("renderD128");
    std::fs::write(&n, b"").unwrap();
    assert!(hint_in(e(), &d).ends_with("installed, with the Vulkan loader?"));
    std::fs::set_permissions(&n, std::fs::Permissions::from_mode(0o000)).unwrap();
    // root opens it anyway
    if std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&n)
        .is_err()
    {
        assert!(hint_in(e(), &d).ends_with("in the `render` group (/dev/dri/renderD*)?"));
    }
    assert_eq!(hint("another error".into()), "another error");
    std::fs::remove_dir_all(&d).unwrap();
}
