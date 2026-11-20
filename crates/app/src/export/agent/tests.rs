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
/// their defaults; Shot Lab's effects; the presets, the 360 video widths without the photos' 11904; the blur targets in
/// `Set::faces`, `Set::plates` order, searched every `STEP` frames, and no longer later; denoise 0 to 1, off by
/// default.
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
    assert_eq!(c.later.len(), 6);
    for a in ["faces", "plates", "blur", "face_blur", "plate_blur"] {
        let a = a.to_string();
        assert!(
            !c.later.iter().any(|l| l.id == a || l.aliases.contains(&a)),
            "{a}"
        );
    }
    let ids: Vec<&str> = c.blur.iter().map(|n| n.id.as_str()).collect();
    assert_eq!(ids, ["faces", "plates"]);
    assert_eq!(c.blur[1].name.as_deref(), Some("Licence plates"));
    assert!(c.blur[1].aliases.contains(&"license_plates".to_string()));
    assert_eq!(c.blur_every, 3);
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
    assert_eq!(
        (c.denoise.min, c.denoise.max, c.denoise.default),
        (0.0, 1.0, Some(0.0))
    );
}

/// Stills start from preset-free defaults whatever the preset, exports from theirs; a changed size or rate clears the
/// preset as on the screen; looks, adjustments, the view, the rate and the blur's boxes land where the screen keeps
/// them.
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
    let blur = |b: &[&str]| Want {
        blur: b.iter().map(|b| b.to_string()).collect(),
        ..want(Kind::Export)
    };
    for (b, on) in [
        (&["plates"][..], (false, true)),
        (&["faces"], (true, false)),
        (&["faces", "plates"], (true, true)),
        (&[], (false, false)),
    ] {
        let (x, _, _) = set_of(&blur(b), Kind::Export, true).unwrap();
        assert_eq!((x.faces, x.plates), on, "{b:?}");
    }
    assert_eq!(
        set_of(&blur(&["faces", "cats"]), Kind::Still, true).unwrap_err(),
        "blur \"cats\": not in the catalog"
    );
    // denoise: an export's strength, 0 off; a still has none
    for (d, on) in [(Some(0.5), Some(0.5)), (Some(0.0), None), (None, None)] {
        let w = Want {
            denoise: d,
            ..want(Kind::Export)
        };
        assert_eq!(
            set_of(&w, Kind::Export, true).unwrap().0.denoise,
            on,
            "{d:?}"
        );
    }
    let w = Want {
        denoise: Some(1.0),
        ..want(Kind::Still)
    };
    assert_eq!(set_of(&w, Kind::Still, true).unwrap().0.denoise, None);
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
/// view, the blur's targets (none: `[]`); the keys the export time guess compares.
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
    assert_eq!((&s["blur"], &s["denoise"]), (&json!([]), &json!(0.0)));
    let w = Want {
        blur: vec!["faces".into()],
        denoise: Some(0.5),
        ..want(Kind::Export)
    };
    let (x, v, p) = set_of(&w, Kind::Export, true).unwrap();
    let s = settings(&x, &v, p, &w, Kind::Export, [0.0, 5.0], 0.0);
    assert_eq!(
        (&s["blur"], &s["denoise"]),
        (&json!(["faces"]), &json!(0.5))
    );
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
        scan: None,
        search: 0,
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

fn req(kind: Kind, files: &[PathBuf], want: Want) -> Req {
    Req {
        kind,
        files: files.to_vec(),
        want,
    }
}

/// A blur of these ids.
fn blur(b: &[&str]) -> Vec<String> {
    b.iter().map(|b| b.to_string()).collect()
}

/// Blur's check, before the GPU (`shotlab::Clip::open` only), on the synthetic recording (300 frames at 30 fps, no
/// IMU): the source frames its pass searches (an export's every 3rd of its span whatever its rate, a still's one), the
/// plan's targets and settings, the `_blur_` name; none without blur. Refused on a recording in a file per lens (which
/// the source check takes) and with burn kept off; an asked stabilisation there is no IMU for is refused by the
/// source check first, a defaulted one resolves to off.
#[test]
fn blur_check() {
    let d = std::env::temp_dir().join(format!("x5-agent-blur-check-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let src = d.join("VID_20260117_141148_00_635.insv");
    ::export::synth::insv(&src, 300, 256).unwrap();
    let one = [src.clone()];
    let a = Agent::new(false);
    let ex = |b: &[&str]| Want {
        end_s: Some(2.0),
        stabilisation: None,
        blur: blur(b),
        ..want(Kind::Export)
    };
    let p = a
        .check(&req(Kind::Export, &one, ex(&["faces", "plates"])))
        .unwrap();
    assert_eq!((p.search, p.frames), (20, 60));
    assert_eq!(p.blur, ["faces", "plates"]);
    assert_eq!(
        (&p.settings["blur"], &p.settings["stabilisation"]),
        (&json!(["faces", "plates"]), &json!("off"))
    );
    assert_eq!(p.stem, "VID_20260117_141148_00_635_blur_pano57");
    let w = Want {
        fps: Some("25".into()),
        ..ex(&["plates"])
    };
    let p = a.check(&req(Kind::Export, &one, w)).unwrap();
    assert_eq!(
        (p.search, p.frames, &p.blur[..]),
        (20, 50, &["plates".to_string()][..])
    );
    let w = Want {
        time_s: Some(1.0),
        stabilisation: None,
        blur: blur(&["faces"]),
        ..want(Kind::Still)
    };
    let p = a.check(&req(Kind::Still, &one, w)).unwrap();
    assert_eq!((p.search, p.frames), (1, 1));
    assert_eq!(p.stem, "VID_20260117_141148_00_635_blur_1.00s_360_7680");
    let p = a.check(&req(Kind::Export, &one, ex(&[]))).unwrap();
    assert_eq!(
        (p.search, &p.blur[..], &p.settings["blur"]),
        (0, &[][..], &json!([]))
    );
    assert_eq!(p.stem, "VID_20260117_141148_00_635_pano57");
    // a file per lens: the source check takes the pair, the pass doesn't
    let ten = d.join("VID_20260117_141148_10_635.insv");
    std::fs::copy(&src, &ten).unwrap();
    let pair = [src.clone(), ten];
    assert!(a.check(&req(Kind::Export, &pair, ex(&[]))).is_ok());
    assert_eq!(
        a.check(&req(Kind::Export, &pair, ex(&["faces"])))
            .unwrap_err(),
        "blur: a recording in a file per lens (_00_ + _10_) is not supported yet; leave out blur"
    );
    let e = Agent::new(true).check(&req(Kind::Export, &one, ex(&["faces"])));
    assert_eq!(e.unwrap_err(), r::no_blur(gpu::NOATTACH));
    let w = Want {
        stabilisation: Some("horizon".into()),
        ..ex(&["faces"])
    };
    let e = a.check(&req(Kind::Export, &one, w)).unwrap_err();
    assert!(
        e.ends_with(": use stabilisation off") && !e.starts_with("blur"),
        "{e}"
    );
    std::fs::remove_dir_all(&d).unwrap();
}

/// Denoise in the check's settings: off when left out, on the synthetic recording (no audio) too, where a strength is
/// refused and 0 passes; on v635 (`X5_SAMPLES`, marked AI wind reduction like every X5 file) off when left out, an
/// asked strength as asked; a still has none.
#[test]
fn denoise_check() {
    let d = std::env::temp_dir().join(format!("x5-agent-denoise-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let src = d.join("VID_20260117_141148_00_635.insv");
    ::export::synth::insv(&src, 60, 64).unwrap();
    let a = Agent::new(false);
    let ex = |dn: Option<f32>| Want {
        end_s: Some(1.0),
        stabilisation: None,
        denoise: dn,
        ..want(Kind::Export)
    };
    let shown = |f: &[PathBuf], w: Want| {
        a.check(&req(Kind::Export, f, w))
            .map(|p| p.settings["denoise"].clone())
    };
    let one = [src];
    assert_eq!(shown(&one, ex(None)), Ok(json!(0.0)));
    assert_eq!(shown(&one, ex(Some(0.0))), Ok(json!(0.0)));
    assert_eq!(
        shown(&one, ex(Some(0.5))),
        Err("denoise: the recording has no audio; leave out denoise".into())
    );
    std::fs::remove_dir_all(&d).unwrap();
    let Some(s) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: v635 skipped");
        return;
    };
    let v = [s.join("v635.insv")];
    for (dn, want) in [(None, 0.0), (Some(0.5), 0.5), (Some(0.0), 0.0)] {
        assert_eq!(shown(&v, ex(dn)), Ok(json!(want)), "{dn:?}");
    }
    let w = Want {
        stabilisation: None,
        ..want(Kind::Still)
    };
    let p = a.check(&req(Kind::Still, &v, w)).unwrap();
    assert!(p.settings.get("denoise").is_none());
}

/// An engine up on the tests' one device (`export::tests::gpu`: `ai::MODELS` stays loaded for it), with burn as the
/// tests attach it; `raft`: why the device has none (blur refused).
fn engine(g: &Arc<Gpu>, cx: &stitch::Ctx, raft: Option<&str>) -> Agent {
    let lane = |n| LaneQ {
        q: ::export::Queue::with(g.clone(), cx.clone(), n).unwrap(),
        wedged: Mutex::new(None),
    };
    let rd = Arc::new(Ready {
        g: g.clone(),
        info: GpuInfo {
            raft: raft.map(String::from),
            ..base(g)
        },
        still: lane("x5-test-still"),
        export: lane("x5-test-export"),
        warm: AtomicBool::new(false),
    });
    let a = Agent::new(false);
    let _ = a.dev.set(rd.clone());
    *lock(&a.st) = Eng::Ready(rd);
    a
}

/// How long a blurred run may take at most: ai's tests (their own device) hold the models for whole runs of minutes
/// (Deep Track, auto edit with `X5_SAMPLES`), which a pass waits for.
const BOUND: Duration = Duration::from_secs(1800);

/// `q` checked and run on `lane` of `a` into `<dir>/<plan's stem>.<ext>`: the plan, its readings every 5 ms until it
/// finished (the last one's how), the file.
fn run_on(a: &Agent, q: Req, dir: &Path, lane: Lane) -> (Plan, Vec<RunSt>, PathBuf) {
    let p = a.check(&q).unwrap();
    let dst = dir.join(format!("{}.{}", p.stem, p.ext));
    let part = media::part::create(&dst).unwrap();
    let run = a.submit(q, &p, &dst, part, lane).unwrap();
    let (t0, mut seen) = (Instant::now(), vec![]);
    loop {
        let s = run.state();
        let over = !matches!(s, RunSt::Queued | RunSt::Running { .. });
        seen.push(s);
        if over {
            break;
        }
        assert!(t0.elapsed() < BOUND, "{:?}", seen.last());
        thread::sleep(Duration::from_millis(5))
    }
    (p, seen, dst)
}

/// A blurred export of the synthetic recording on an engine over the tests' one device: `detecting` from its first
/// reading, the frames searched rising to the plan's with the pass's own rate and its seconds left falling to 0, then
/// the frames (the searched ones still counted); done with what it found (nothing on grey: 20 frames searched of 60)
/// and its file. After the idle trim a blurred still loads the detectors again. One cancelled in its pass ends there,
/// with no file. A device without burn refuses blur at submit, in mcp's words.
#[test]
fn blur_run() {
    if super::super::tests::no_cf_home() {
        return;
    }
    let (_l, g, cx) = super::super::tests::gpu();
    let d = std::env::temp_dir().join(format!("x5-agent-blur-run-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let src = d.join("synth.insv");
    ::export::synth::insv(&src, 300, 256).unwrap();
    let one = [src.clone()];
    let w = Want {
        output: Some(Output::Flat),
        width: Some(640),
        height: Some(360),
        codec: Some("h264".into()),
        mbps: Some(8.0),
        raft: Some(false),
        stabilisation: None,
        end_s: Some(2.0),
        blur: blur(&["faces", "plates"]),
        ..want(Kind::Export)
    };
    let a = engine(&g, &cx, None);
    let (p, seen, dst) = run_on(&a, req(Kind::Export, &one, w.clone()), &d, Lane::Export);
    let mut pass = vec![];
    let mut after = false;
    for s in &seen {
        match *s {
            RunSt::Running {
                phase: Phase::Detecting,
                done,
                searched,
                fps,
                eta,
                ..
            } => {
                assert!(!after && done == 0, "{seen:?}");
                pass.push((searched, fps, eta))
            }
            RunSt::Running { done, searched, .. } => {
                assert_eq!(searched, p.search, "{s:?}");
                after |= done > 0;
            }
            _ => {}
        }
    }
    assert!(
        matches!(
            seen[0],
            RunSt::Queued
                | RunSt::Running {
                    phase: Phase::Detecting,
                    ..
                }
        ),
        "{:?}",
        seen[0]
    );
    assert!(pass.len() > 1, "{seen:?}");
    assert!(pass.windows(2).all(|w| w[0].0 <= w[1].0), "{pass:?}");
    for &(k, fps, eta) in &pass {
        match k >= 2 {
            true => assert!(
                fps > 0.0 && eta.is_finite() && (eta > 0.0) == (k < p.search),
                "{pass:?}"
            ),
            false => assert_eq!((fps, eta), (0.0, 0.0)),
        }
    }
    let RunSt::Done {
        frames,
        found: Some(f),
        ..
    } = seen.last().unwrap().clone()
    else {
        panic!("{:?}", seen.last())
    };
    assert_eq!(frames, 60);
    assert!(f.secs > 0.0, "{f:?}");
    let zero = r::Found {
        searched: 20,
        frames: 60,
        secs: f.secs,
        ..Default::default()
    };
    assert_eq!(f, zero);
    assert!(dst.exists() && !media::part::of(&dst).exists());
    // the idle trim drops the detectors (once ai's tests, on their own device, let go of them): a blurred still loads
    // them again
    let (t0, mut kept) = (Instant::now(), 0);
    loop {
        a.trim();
        match crate::ai::MODELS.loaded() {
            Some(false) => break,
            // loaded again by another test in between, at most a few times
            Some(true) => kept += 1,
            None => assert!(t0.elapsed() < BOUND, "the models held"),
        }
        assert!(kept < 3, "the trim kept the detectors");
        thread::sleep(Duration::from_millis(50))
    }
    let w1 = Want {
        time_s: Some(1.0),
        ..w.clone()
    };
    let (_, seen, _) = run_on(&a, req(Kind::Still, &one, w1), &d, Lane::Still);
    let Some(RunSt::Done { found: Some(f), .. }) = seen.last() else {
        panic!("{:?}", seen.last())
    };
    assert_eq!((f.searched, f.frames, f.faces + f.plates), (1, 1, 0));
    // cancelled in its pass (the whole clip: 100 frames to search): at its next searched frame, no file
    let q = req(
        Kind::Export,
        &one,
        Want {
            end_s: None,
            ..w.clone()
        },
    );
    let p = a.check(&q).unwrap();
    let o = d.join("cut.mp4");
    let run = (a.submit(q, &p, &o, media::part::create(&o).unwrap(), Lane::Export)).unwrap();
    let t0 = Instant::now();
    while !matches!(
        run.state(),
        RunSt::Running {
            phase: Phase::Detecting,
            searched: 2..,
            ..
        }
    ) {
        assert!(t0.elapsed() < BOUND, "{:?}", run.state());
        thread::sleep(Duration::from_millis(5))
    }
    let (t1, at) = (Instant::now(), run.state());
    run.cancel();
    while !matches!(run.state(), RunSt::Cancelled) {
        assert!(t1.elapsed() < Duration::from_secs(3), "{:?}", run.state());
        thread::sleep(Duration::from_millis(5))
    }
    assert!(
        !o.exists() && !media::part::of(&o).exists(),
        "cancelled at {at:?}"
    );
    assert_eq!(p.search, 100);
    // the GPU up without burn: refused at submit too, though mcp's device check refuses it first
    let b = engine(&g, &cx, Some("burn's attach failed"));
    let q = req(Kind::Export, &one, w);
    let p = b.check(&q).unwrap();
    let o = d.join("refused.mp4");
    let e = b.submit(q, &p, &o, media::part::create(&o).unwrap(), Lane::Export);
    assert_eq!(e.err(), Some(r::no_blur("burn's attach failed")));
    std::fs::remove_dir_all(&d).unwrap();
}

/// A lane wedged while it is in a detection (`ai::MODELS` held in `with_stop`, as a driver call that never returns
/// would) keeps the detectors and BURN: blur is refused at check, at submit and in the GPU's info (mcp's device check),
/// on the other lane too, while unblurred jobs check (without RAFT by default; asked for, it is refused at submit);
/// once the detection ends, blur and RAFT are back. A lane in a detection that isn't wedged refuses nothing.
#[test]
fn blur_stuck() {
    let (_l, g, cx) = super::super::tests::gpu();
    let d = std::env::temp_dir().join(format!("x5-agent-blur-stuck-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let src = d.join("synth.insv");
    ::export::synth::insv(&src, 30, 256).unwrap();
    let one = [src.clone()];
    let a = engine(&g, &cx, None);
    let rd = a.ready().unwrap();
    // an export whose pass holds the models until let go
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
    let dst = d.join("held.mp4");
    let (mut job, _) = clip_job(&one, dur, rate, &x, ab, false, move |_| v, dst, None, false);
    let (inside, go) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let (gg, i, l) = (g.clone(), inside.clone(), go.clone());
    job.pre = Some(Box::new(move |_, stop| {
        crate::ai::MODELS.with_stop(&gg, stop, None, |_| {
            i.store(true, Relaxed);
            while !l.load(Relaxed) {
                thread::sleep(Duration::from_millis(5))
            }
            Ok(())
        })?;
        Ok(None)
    }));
    let id = rd.export.q.push(job);
    // let go however the test ends: dropping the lane joins its thread
    struct Go(Arc<AtomicBool>);
    impl Drop for Go {
        fn drop(&mut self) {
            self.0.store(true, Relaxed)
        }
    }
    let go = Go(go);
    let t0 = Instant::now();
    while !inside.load(Relaxed) {
        assert!(t0.elapsed() < BOUND, "{:?}", rd.export.q.state(id));
        thread::sleep(Duration::from_millis(5))
    }
    let seam = |b: &[&str], raft| {
        let w = Want {
            time_s: Some(1.0),
            stabilisation: None,
            blur: blur(b),
            raft,
            ..want(Kind::Still)
        };
        req(Kind::Still, &one, w)
    };
    let still = |b: &[&str]| seam(b, None);
    let p = a.check(&still(&["faces"])).unwrap();
    let rp = a.check(&seam(&[], Some(true))).unwrap();
    assert!(a.check(&still(&[])).unwrap().raft);
    let info = |a: &Agent| match a.gpu() {
        GpuSt::Ready(i) => i,
        s => panic!("{s:?}"),
    };
    assert_eq!(info(&a).blur_stuck, None);
    a.wedge(Lane::Export, "stuck in detecting");
    let stuck = r::stuck_blur(Lane::Export);
    assert_eq!(a.check(&still(&["faces"])).unwrap_err(), stuck);
    assert!(
        !a.check(&still(&[])).unwrap().raft,
        "RAFT by default while stuck"
    );
    assert_eq!(info(&a).blur_stuck, Some(Lane::Export));
    let sub = |q, p, n: &str| {
        let o = d.join(n);
        a.submit(q, p, &o, media::part::create(&o).unwrap(), Lane::Still)
    };
    assert_eq!(sub(still(&["faces"]), &p, "blur.jpg").err(), Some(stuck));
    let e = sub(seam(&[], Some(true)), &rp, "raft.jpg").err();
    assert_eq!(e, Some(r::stuck_raft(Lane::Export)));
    drop(go);
    while !matches!(rd.export.q.state(id), Some(::export::State::Done(_))) {
        let s = rd.export.q.state(id);
        assert!(
            t0.elapsed() < BOUND && !matches!(s, Some(::export::State::Failed(_))),
            "{s:?}"
        );
        thread::sleep(Duration::from_millis(5))
    }
    assert!(a.check(&still(&["faces"])).is_ok());
    assert!(a.check(&still(&[])).unwrap().raft);
    assert_eq!(info(&a).blur_stuck, None);
    std::fs::remove_dir_all(&d).unwrap();
}

/// An agent's blurred still is the Export screen's grab, byte for byte, on a recording (`X5_SAMPLES`: v635, else its
/// first `.insv`) at `X5_BLUR_AT` s (0 by default): a flat 1920 × 1080 view with the template seam and the stabilisation
/// the check resolves, aimed at the strongest face or plate the pass finds there; both blurs. The screen's grab runs
/// twice first: if those differ, the detectors aren't deterministic in this process, and the test says so and only
/// asks that the agent's still was blurred.
#[test]
fn agent_grab_parity() {
    use super::super::tests::{gpu, no_cf_home, sample, task};
    if no_cf_home() {
        return;
    }
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let f = sample(&d);
    let at: f64 =
        std::env::var("X5_BLUR_AT").map_or(0.0, |s| s.parse().expect("X5_BLUR_AT: seconds"));
    let (_l, g, cx) = gpu();
    let o = std::env::temp_dir().join(format!("x5-agent-grab-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&o);
    std::fs::create_dir_all(&o).unwrap();
    let a = engine(&g, &cx, None);
    let mut w = Want {
        output: Some(Output::Flat),
        raft: Some(false),
        time_s: Some(at),
        blur: blur(&["faces", "plates"]),
        ..Want::default()
    };
    let files = [f.clone()];
    let p = a.check(&req(Kind::Still, &files, w.clone())).unwrap();
    w.stabilisation = p.settings["stabilisation"].as_str().map(String::from);
    // aimed at what the pass finds on the frame
    let (x, _, _) = set_of(&w, Kind::Still, false).unwrap();
    let (src, dur) = probe(&f).unwrap();
    let ab = span([at; 2], true, x.rate(src), dur);
    let stop = AtomicBool::new(false);
    let stab = x.stab.map(|l| (l, x.smooth));
    let (times, regs, _) = privacy_pass(
        &g,
        &cx,
        &files,
        stab,
        ab,
        src,
        [true; 2],
        Hold::Pass,
        &stop,
        None,
        &mut |_, _| {},
    )
    .unwrap();
    let k0 = keys(&times, ab).0;
    let r = *(regs[k0].iter())
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .unwrap_or_else(|| {
            panic!("nothing found at {at} s: X5_BLUR_AT a time with a face or plate")
        });
    let [yaw, pitch] = track::yaw_pitch(r.dir);
    eprintln!(
        "{} regions at {at} s; aiming at yaw {yaw:.1}° pitch {pitch:.1}° ({:.1}° × {:.1}°, score {:.2})",
        regs[k0].len(),
        r.size.x,
        r.size.y,
        r.score
    );
    w.view = Some(r::View {
        yaw: Some(yaw),
        pitch: Some(pitch),
        ..Default::default()
    });
    let (x, v, _) = set_of(&w, Kind::Still, false).unwrap();
    // the screen's grab of that frame with these settings
    let (mut k, ctx) = (
        super::super::Keep::default(),
        eframe::egui::Context::default(),
    );
    let mut grab = |n: &str, x: Set| {
        let s = Spec {
            files: files.to_vec(),
            photo: false,
            dur,
            t: at,
            view: v,
            look: player::Look::Perspective,
            stab: x.stab,
            smooth: x.smooth,
            follow: None,
            faces: x.faces,
            plates: x.plates,
        };
        let t = task(s, x, o.join(format!("{n}.jpg")), [at; 2], true);
        k.video(&g, &cx, &t, &ctx).unwrap();
        (std::fs::read(&t.it.path).unwrap(), t.it.note())
    };
    let (gui, note) = grab("gui", x);
    let again = grab("gui2", x).0;
    let sharp = grab(
        "sharp",
        Set {
            faces: false,
            plates: false,
            ..x
        },
    )
    .0;
    let (p, seen, dst) = run_on(&a, req(Kind::Still, &files, w), &o, Lane::Still);
    assert!(p.stem.contains("_blur_"), "{}", p.stem);
    let Some(RunSt::Done { found: Some(c), .. }) = seen.last() else {
        panic!("{:?}", seen.last())
    };
    let agent = std::fs::read(&dst).unwrap();
    let said = format!(
        "{} face and {} plate detections on {} frames, {} and {} tracks, blurred on {} of {} frames",
        c.faces, c.plates, c.searched, c.face_tracks, c.plate_tracks, c.blurred, c.frames
    );
    eprintln!("the screen: {note}; the agent: {said}");
    assert!(agent != sharp, "the agent's still is not blurred");
    assert!(c.faces + c.plates > 0 && c.blurred == 1, "{c:?}");
    match gui == again {
        true => {
            assert_eq!(said, note);
            assert!(
                gui != sharp && agent == gui,
                "the agent's still differs from the screen's"
            );
        }
        false => eprintln!(
            "the detectors aren't deterministic in this process: two grabs of the screen differ; only the agent's \
             blur checked"
        ),
    }
    std::fs::remove_dir_all(&o).unwrap();
}
