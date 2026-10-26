use super::*;

/// The extras survive a project file round trip next to edit's timeline (a clip's adjustments in it), and edit's
/// reader ignores them.
#[test]
fn extras() {
    let mut pj = Project::new(FPS).unwrap();
    let f = pj.file("/v/a.insv", 10.0).unwrap();
    pj.insert(0, Clip::new(f, 1.0, 4.0)).unwrap();
    pj.edit(0, |c| (c.adj.exposure, c.adj.purple) = (0.5, 0.25))
        .unwrap();
    let x = Extra {
        denoise: Some(0.7),
        stab: Some(stab::Lock::Direction),
        smooth: 1.25,
        over: vec![
            Over {
                pic: Pic::Text {
                    s: "Hi \"there\"\n\\ é".into(),
                    px: 48.0,
                    rgba: [10, 20, 30, 255],
                },
                place: fx::Place::Flat {
                    at: [0.25, 0.75],
                    w: 0.3,
                    rot: 12.0,
                },
                opacity: 0.5,
            },
            Over {
                pic: Pic::Disc([1, 2, 3, 4]),
                place: fx::Place::Sphere {
                    yaw: 0.0,
                    pitch: -90.0,
                    fov: 40.0,
                    roll: 5.0,
                },
                opacity: 1.0,
            },
            Over {
                pic: Pic::Png("/l/logo.png".into()),
                place: fx::Place::Flat {
                    at: [0.9, 0.1],
                    w: 0.1,
                    rot: 0.0,
                },
                opacity: 0.9,
            },
        ],
    };
    let s = doc::save(&pj, &x);
    assert!(edit::Project::load(&s).is_ok(), "edit reads the file");
    let (p2, x2) = doc::load(&s).unwrap();
    assert_eq!(p2.tl(), pj.tl());
    assert_eq!(x2, x);
    // a plain edit file: the defaults
    let (_, x3) = doc::load(&pj.save()).unwrap();
    assert_eq!(x3, Extra::default());
    assert!(doc::parse("{\"a\": [1, 2,]}").is_err());
    assert!(
        doc::parse("[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]").is_err()
    );
    assert_eq!(
        doc::parse(" [true, null, -1.5e2, \"\\u00e9\\ud83d\\ude00\"] ").unwrap(),
        doc::J::A(vec![
            doc::J::B(true),
            doc::J::Null,
            doc::J::N(-150.0),
            doc::J::S("é😀".into())
        ])
    );
    // the nadir disc: opaque inside, a soft edge, clear corners
    let i = x.over[1].image().unwrap();
    let a = |x: u32, y: u32| i.px[((y * i.w + x) * 4 + 3) as usize];
    assert_eq!(
        (i.w, a(128, 128), a(0, 0), a(128, 8), a(128, 2)),
        (256, 4, 0, 4, 2)
    );
}

/// A 16-bit stereo 48 kHz WAV of a three-note chord at −12 dBFS, `secs` long.
pub fn chord(p: &Path, secs: f64) {
    let n = (secs * 48000.0) as u32;
    let mut b = Vec::with_capacity(44 + 4 * n as usize);
    for (k, v) in [(b"RIFF", 36 + 4 * n), (b"WAVE", 0), (b"fmt ", 16)] {
        b.extend_from_slice(k);
        if k != b"WAVE" {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    for v in [1u16, 2] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [48000u32, 4 * 48000] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [4u16, 16] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(4 * n).to_le_bytes());
    for i in 0..n {
        let t = i as f64 / 48000.0;
        let x: f64 = [220.0, 277.18, 329.63]
            .iter()
            .map(|f| (std::f64::consts::TAU * f * t).sin())
            .sum();
        let s = (x / 3.0 * 0.25 * 32767.0) as i16;
        b.extend_from_slice(&s.to_le_bytes());
        b.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(p, b).unwrap();
}

fn out(name: &str) -> PathBuf {
    let d = std::env::var_os("X5_OUT")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join("x5desk-editor");
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

/// The demo project from the samples (`X5_SAMPLES`), made with the screen's own actions: v635 0.5–4.5 s ramped
/// 1× → 2× → 1×, v606 5–11 s with a 1 s freeze frame in the middle, v627 2–8 s panning 120° on its own keys with the
/// Teal & Orange look at 0.8, a text overlay, a music chord ducked under the clips. Each action's undo step, the whole
/// sequence undone back to the empty timeline and redone, every state as before; a drag's trims as one step; the
/// project file round trip (timeline and extras). `X5_DEMO`: where the project goes (else the temp dir).
#[test]
fn demo() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let still = crate::player::still(&g, &cx);
    let mut e = Editor::new(&g, &cx, &egui::Context::default(), still.clone(), None);
    // every state after an undo step
    let mut st = vec![e.pj.tl().clone()];
    fn snap(st: &mut Vec<edit::Timeline>, e: &Editor) {
        assert!(e.err.is_none(), "{:?}", e.err);
        st.push(e.pj.tl().clone());
        assert_eq!(
            e.pj.history(),
            (st.len() - 1, 0),
            "one undo step per action"
        );
    }
    for f in ["v635.insv", "v606.insv", "v627.insv"] {
        e.add(&d.join(f), None).unwrap();
        snap(&mut st, &e);
    }
    let fps = e.pj.tl().out.fps;
    assert_eq!((fps.num, fps.den), (30000, 1001));
    // trims as drags: many steps, one undo step each
    for (k, a, b) in [(0, 0.5, 4.5), (1, 5.0, 11.0), (2, 2.0, 8.0)] {
        e.begin();
        for i in 1..=10 {
            let f = i as f64 / 10.0;
            let r = e.pj.trim(
                k,
                a * f,
                e.pj.tl().files[e.pj.tl().clips[k].file as usize]
                    .dur
                    .min(b / f),
            );
            e.did(r);
        }
        e.end();
        assert_eq!((e.pj.tl().clips[k].a, e.pj.tl().clips[k].b), (a, b));
        snap(&mut st, &e);
    }
    e.ed(|p| p.speed(0, vec![(0.5, 1.0), (1.5, 2.0), (3.5, 2.0), (4.5, 1.0)]));
    snap(&mut st, &e);
    let d0 = e.pj.tl().clips[0].dur();
    assert!(
        (d0 - (2.0 * 2f64.ln() + 1.0)).abs() < 1e-9,
        "ramped clip {d0} s"
    );
    // a freeze frame in the middle of clip 1
    let t1 = e.pj.tl().out.fps.t(e.pj.tl().frames(1).start);
    e.seek(t1 + 2.0);
    e.hold = 1.0;
    e.freeze();
    snap(&mut st, &e);
    let tl = e.pj.tl();
    assert_eq!(tl.clips.len(), 5);
    assert_eq!((tl.clips[2].hold, e.sel), (1.0, Some(2)));
    assert!(
        (tl.clips[2].a - tl.clips[1].b).abs() < 1e-9,
        "the freeze shows the cut"
    );
    // the pan: a key at the clip's first and last frame from the viewer's view
    let n = tl.frames(4);
    let (s, z) = (tl.out.fps.t(n.start), tl.out.fps.t(n.end - 1));
    e.seek(s);
    e.free = Some(View {
        yaw: 0.0,
        fov: 90.0,
        ..View::default()
    });
    e.key_here();
    snap(&mut st, &e);
    e.seek(z);
    e.free = Some(View {
        yaw: 120.0,
        fov: 90.0,
        ..View::default()
    });
    e.ease = Ease::InOut;
    e.key_here();
    snap(&mut st, &e);
    assert_eq!(e.pj.tl().clips[4].view.len(), 2);
    e.seek((s + z) / 2.0);
    let y = e.shown().yaw;
    assert!(e.free.is_none() && y > 30.0 && y < 90.0, "halfway yaw {y}");
    e.ed(|p| p.edit(4, |c| (c.look, c.amt) = ("Teal & Orange".into(), 0.8)));
    snap(&mut st, &e);
    let m = out("chord.wav");
    chord(&m, 20.0);
    e.music(&m).unwrap();
    snap(&mut st, &e);
    let r = e.pj.edit_out(|o| o.music.as_mut().unwrap().gain = 0.8);
    e.did(r);
    snap(&mut st, &e);
    e.x.over.push(Over {
        pic: Pic::Text {
            s: "x5desk editor".into(),
            px: 56.0,
            rgba: [255, 255, 255, 230],
        },
        place: fx::Place::Flat {
            at: [0.5, 0.9],
            w: 0.3,
            rot: 0.0,
        },
        opacity: 0.9,
    });
    // undo to the start and redo to the end: the overlay first (a step of the extras), then every state as it was (the
    // file table keeps the files added, which edit's log leaves alone)
    let same = |a: &edit::Timeline, b: &edit::Timeline| {
        a.clips == b.clips && a.out == b.out && a.files[..b.files.len()] == b.files[..]
    };
    let n = st.len() - 1;
    e.undo();
    assert!(
        e.x.over.is_empty() && same(e.pj.tl(), &st[n]),
        "undo the overlay"
    );
    for i in (0..n).rev() {
        e.undo();
        assert!(same(e.pj.tl(), &st[i]), "undo to state {i}");
    }
    assert!(!e.pj.can_undo() && e.pj.tl().is_empty());
    assert_eq!(e.history(), (0, n + 1));
    for (i, s) in st.iter().enumerate().skip(1) {
        e.redo();
        assert!(same(e.pj.tl(), s), "redo to state {i}");
    }
    assert!(e.x.over.is_empty());
    e.redo();
    assert_eq!(
        (e.x.over.len(), e.history(), e.pj.history()),
        (1, (n + 1, 0), (n, 0))
    );
    let tl = e.pj.tl();
    eprintln!(
        "demo: {} clips, {} frames = {:.3} s, {n} timeline undo steps and 1 of the extras; clips (s): {:?}",
        tl.clips.len(),
        tl.len(),
        tl.dur(),
        (0..tl.clips.len())
            .map(|k| (tl.frames(k).end - tl.frames(k).start) as f64 * 1001.0 / 30000.0)
            .collect::<Vec<_>>()
    );
    // the job the Export screen gets
    let j = e.job().unwrap();
    assert_eq!(
        (j.files.len(), &j.files[3], j.looks.len()),
        (4, &vec![m.clone()], 1)
    );
    assert!(j.files[0][0].ends_with("v635.insv") && j.over.len() == 1);
    // project file round trip
    let p = std::env::var_os("X5_DEMO").map_or_else(|| out("demo.x5edit"), PathBuf::from);
    e.save(&p).unwrap();
    assert!(!e.modified());
    let mut e2 = Editor::new(&g, &cx, &egui::Context::default(), still, None);
    e2.load(&p).unwrap();
    assert_eq!((e2.pj.tl(), &e2.x), (e.pj.tl(), &e.x));
    assert_eq!(
        doc::save(&e2.pj, &e2.x),
        std::fs::read_to_string(&p).unwrap()
    );
    eprintln!("demo project → {}", p.display());
}

/// The undo history of the extras (no samples needed): settled changes are steps in order with edit's, a slider held or
/// a text being typed is not settled yet, undo and redo restore the extras, a new change drops the steps undone; a
/// `.cube` file imported as a clip's look is parsed, kept by its path and goes with the export job; a broken one is
/// refused.
#[test]
fn history() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let mut e = Editor::new(
        &g,
        &cx,
        &egui::Context::default(),
        player::still(&g, &cx),
        None,
    );
    let f = e.pj.file("/nowhere/a.insv", 20.0).unwrap();
    e.ed(|p| p.insert(0, Clip::new(f, 0.0, 10.0)));
    e.x.denoise = Some(0.5);
    e.sync();
    e.ed(|p| p.split(4.0));
    e.x.over.push(Over {
        pic: Pic::Disc([1, 2, 3, 255]),
        place: fx::Place::Flat {
            at: [0.5, 0.5],
            w: 0.2,
            rot: 0.0,
        },
        opacity: 1.0,
    });
    e.sync();
    let h = (e.x.stab, e.x.smooth);
    e.stab(Some(stab::Lock::Direction), 1.0);
    e.sync();
    assert_eq!(e.history(), (5, 0));
    let x = e.x.clone();
    e.undo();
    assert_eq!(((e.x.stab, e.x.smooth), e.x.over.len()), (h, 1));
    e.undo();
    assert!(e.x.over.is_empty() && e.pj.tl().clips.len() == 2);
    e.undo();
    assert_eq!(e.pj.tl().clips.len(), 1);
    e.redo();
    e.redo();
    e.redo();
    assert_eq!((&e.x, e.history()), (&x, (5, 0)));
    e.undo();
    e.undo();
    e.x.denoise = Some(0.25);
    e.sync();
    assert_eq!(e.history(), (4, 0));
    e.redo();
    assert!(e.x.over.is_empty() && (e.x.stab, e.x.smooth) == h && e.x.denoise == Some(0.25));
    for _ in 0..4 {
        e.undo();
    }
    assert_eq!((&e.x, e.pj.tl().clips.len()), (&Extra::default(), 0));
    // a .cube look on the selected clip
    e.redo();
    e.sel = Some(0);
    let c = out("warm.cube");
    let lut = "TITLE \"warm\"\nLUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 0.8\n1 0 0.8\n0 1 0.8\n1 1 0.8\n";
    std::fs::write(&c, lut).unwrap();
    e.cube(&c).unwrap();
    let id = e.pj.tl().clips[0].look.clone();
    assert_eq!(Path::new(&id), std::fs::canonicalize(&c).unwrap());
    let j = e.job().unwrap();
    assert_eq!(j.looks.get(&id), Some(&fx::Cube::parse(lut).unwrap()));
    let bad = out("bad.cube");
    std::fs::write(&bad, "LUT_3D_SIZE 2\n0 0 0\n").unwrap();
    assert!(e.cube(&bad).is_err() && e.pj.tl().clips[0].look == id);
    // a clip of a recording with no main file (a proxy, say from a project file) fails the job, not the export
    let f = e.pj.file("/nowhere/b.lrv", 20.0).unwrap();
    e.ed(|p| p.insert(0, Clip::new(f, 0.0, 5.0)));
    let r = e.job().map(|_| ());
    assert_eq!(r, Err("b.lrv: no full-resolution file".into()));
}

/// Adjustments are per clip (no samples needed): a project file from before, with the timeline's adjustments in
/// `"x5desk"`, loads with them on every clip and saves them there; a drag on the Colour tab's Exposure slider (egui
/// input, the slider found through AccessKit) changes the selected clip's only, as one undo step and without pausing
/// (no `changed`); the job takes each clip's and the denoise setting.
#[test]
fn colour() {
    // as the app wrote it before
    let old = r#"{"x5edit": 1,
 "fps": [30000, 1001],
 "files": [
  {"path": "/nowhere/a.insv", "dur": 20.0}],
 "clips": [
  {"file": 0, "a": 0.5, "b": 4.5, "hold": 0.0, "rev": false, "mute": false, "amt": 1.0, "look": "", "speed": [], "vol": [], "view": []},
  {"file": 0, "a": 5.0, "b": 9.0, "hold": 0.0, "rev": false, "mute": false, "amt": 1.0, "look": "", "speed": [], "vol": [], "view": []}],
 "view": [],
 "music": null,
 "x5desk": {"adjust": {"exposure": 0.5, "brightness": 0.0, "brilliance": 0.0, "contrast": 0.25, "highlights": 0.0, "shadows": 0.0, "whites": 0.0, "blacks": 0.0, "saturation": 0.0, "vibrance": 0.0, "temperature": 0.0, "tint": 0.0, "clarity": 0.0, "sharpen": 0.0, "auto": 0.0, "water": 0.0, "purple": 0.0, "green": 0.0, "blue": 0.0, "reach": 4.0},
  "over": [],
  "denoise": 0.7,
  "stab": "horizon", "smooth": 0.5}}
"#;
    let (pj, x) = doc::load(old).unwrap();
    let a = fx::Params {
        exposure: 0.5,
        contrast: 0.25,
        ..Default::default()
    };
    assert!(pj.tl().clips.iter().all(|c| c.adj == a));
    assert_eq!((x.denoise, x.smooth), (Some(0.7), 0.5));
    let s = doc::save(&pj, &x);
    assert_eq!(
        s.matches("\"adjust\": {\"exposure\": 0.5, \"contrast\": 0.25}")
            .count(),
        2,
        "{s}"
    );
    assert_eq!(s.matches("\"adjust\"").count(), 2, "{s}");
    let (p2, x2) = doc::load(&s).unwrap();
    assert_eq!((p2.tl(), &x2), (pj.tl(), &x));

    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut e = Editor::new(&g, &cx, &ctx, player::still(&g, &cx), None);
    e.take(pj, x, None);
    e.sel = Some(1);
    let mut t = 0.0;
    let mut step = |e: &mut Editor, ev: Vec<egui::Event>| {
        t += 1.0 / 60.0;
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(500.0, 1400.0),
            )),
            time: Some(t),
            events: ev,
            ..Default::default()
        };
        let mut o = ctx.run_ui(raw, |ui| e.colour_ui(ui));
        o.textures_delta.clear();
        e.sync();
        o.platform_output.accesskit_update
    };
    let u = step(&mut e, vec![]).expect("AccessKit is on");
    // the slider its text labels
    let l = (u.nodes.iter())
        .find(|(_, n)| n.role() == egui::Role::Label && n.value() == Some("Exposure (EV)"))
        .map(|x| x.0);
    let r = (u.nodes.iter())
        .find(|(_, n)| {
            n.role() == egui::Role::Slider && l.is_some_and(|l| n.labelled_by().contains(&l))
        })
        .and_then(|(_, n)| n.bounds())
        .expect("the Exposure slider");
    let (v0, h0) = (e.ver, e.history());
    let y = ((r.y0 + r.y1) / 2.0) as f32;
    let (p, q) = (
        egui::pos2((r.x0 + 0.5 * (r.x1 - r.x0)) as f32, y),
        egui::pos2((r.x0 + 0.75 * (r.x1 - r.x0)) as f32, y),
    );
    let btn = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Default::default(),
    };
    for ev in [
        vec![egui::Event::PointerMoved(p)],
        vec![btn(p, true)],
        vec![egui::Event::PointerMoved(p + (q - p) * 0.5)],
        vec![egui::Event::PointerMoved(q)],
        vec![btn(q, false)],
        vec![],
    ] {
        step(&mut e, ev);
    }
    let ex = |e: &Editor| {
        e.pj.tl()
            .clips
            .iter()
            .map(|c| c.adj.exposure)
            .collect::<Vec<_>>()
    };
    eprintln!(
        "Exposure slider at {r:?}: exposures {:?}, history {:?} (before {h0:?})",
        ex(&e),
        e.history()
    );
    assert!(ex(&e)[0] == 0.5 && ex(&e)[1] > 1.0, "{:?}", ex(&e));
    assert_eq!((e.history(), e.ver), ((h0.0 + 1, 0), v0));
    let j = e.job().unwrap();
    assert_eq!(
        (
            j.tl.clips
                .iter()
                .map(|c| c.adj.exposure)
                .collect::<Vec<_>>(),
            j.denoise
        ),
        (ex(&e), Some(0.7))
    );
    e.undo();
    assert_eq!(ex(&e), [0.5, 0.5]);
}

/// One timeline frame with `ev` as the input, a 60th of a second after the last: the timeline drawn in a known rect,
/// what its widgets asked for applied (the drags' acts).
fn frame(e: &mut Editor, ctx: &egui::Context, ev: Vec<egui::Event>) -> egui::Rect {
    let mut rect = egui::Rect::NOTHING;
    let mut acts = vec![];
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1400.0, 300.0),
        )),
        time: Some(ctx.input(|i| i.time) + 1.0 / 60.0),
        events: ev,
        ..Default::default()
    };
    let mut o = ctx.run_ui(raw, |ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(1400.0, 200.0), egui::Sense::hover());
        rect = r;
        e.paint(ui, r, &mut acts);
    });
    // no renderer takes them (epaint's debug check panics on dropped deltas)
    o.textures_delta.clear();
    for a in acts {
        e.act(a);
    }
    rect
}

fn btn(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: Default::default(),
    }
}

/// A drag from `a` to `b` through egui's input, frame by frame; `mid` runs halfway.
fn drag_with(
    e: &mut Editor,
    ctx: &egui::Context,
    a: egui::Pos2,
    b: egui::Pos2,
    mid: &mut dyn FnMut(&mut Editor),
) {
    frame(e, ctx, vec![egui::Event::PointerMoved(a)]);
    frame(e, ctx, vec![btn(a, true)]);
    frame(e, ctx, vec![egui::Event::PointerMoved(a + (b - a) * 0.5)]);
    mid(e);
    frame(e, ctx, vec![egui::Event::PointerMoved(b)]);
    frame(e, ctx, vec![btn(b, false)]);
    frame(e, ctx, vec![]);
}

fn drag(e: &mut Editor, ctx: &egui::Context, a: egui::Pos2, b: egui::Pos2) {
    drag_with(e, ctx, a, b, &mut |_| {});
}

/// Speed keys and volume points drag on the timeline (egui input, no samples needed): a speed key moves in source time
/// by the pointer's travel on the clip's timing at the start and takes the speed at the pointer's height; a volume
/// point ends under the pointer; each drag is one undo step.
#[test]
fn drags() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let ctx = egui::Context::default();
    let mut e = Editor::new(&g, &cx, &ctx, player::still(&g, &cx), None);
    let f = e.pj.file("/nowhere/a.insv", 20.0).unwrap();
    e.ed(|p| p.insert(0, Clip::new(f, 2.0, 12.0)));
    e.ed(|p| p.speed(0, vec![(2.0, 1.0), (5.0, 2.0), (12.0, 1.0)]));
    e.ed(|p| p.edit(0, |c| c.vol = vec![(4.0, 1.0), (8.0, 0.5)]));
    e.sync();
    let u0 = e.history();
    let rect = frame(&mut e, &ctx, vec![]);
    let (z, x0) = (e.zoom, rect.left() + 8.0);
    let xt = |t: f64| x0 + t as f32 * z;
    // rows: ruler, view keys, video, audio (heights and 3 px apart)
    let top = |i: usize| rect.top() + ROWS[..i].iter().map(|h| h + 3.0).sum::<f32>();
    let (vt, vb) = (top(2), top(2) + ROWS[2]);
    let (at, ab) = (top(3), top(3) + ROWS[3]);
    let (lo, hi) = (edit::MIN_SPEED.ln(), edit::MAX_SPEED.ln());
    let ys = |v: f64| vb - 4.0 - ((v.ln() - lo) / (hi - lo)) as f32 * (vb - vt - 8.0);
    let c0 = e.pj.tl().clips[0].clone();
    let t = e.pj.tl().from_track(On::Clip(0), 5.0).unwrap();
    let (a, b) = (
        egui::pos2(xt(t), ys(2.0)),
        egui::pos2(xt(t) + 90.0, ys(4.0) - 0.0),
    );
    drag(&mut e, &ctx, a, b);
    let k = e.pj.tl().clips[0].speed[1];
    let want = c0.src(c0.out(5.0) + 90.0 / z as f64);
    eprintln!(
        "speed key (5 s, 2×) dragged 90 px right and up to 4×: ({:.4} s, {:.4}×), expected ({want:.4} s, 4×)",
        k.0, k.1
    );
    assert!(
        (k.0 - want).abs() < 1e-3 && (k.1 - 4.0).abs() < 1e-3,
        "{k:?}"
    );
    e.sync();
    assert_eq!(e.history(), (u0.0 + 1, 0), "one step per drag");
    // the volume point at 8 s (0.5) to the pointer
    let tl = e.pj.tl();
    let gy = |g: f32| ab - 2.0 - (g / 2.0).clamp(0.0, 1.0) * (ab - at - 4.0);
    let t = tl.from_track(On::Clip(0), 8.0).unwrap();
    let (a, b) = (
        egui::pos2(xt(t), gy(0.5)),
        egui::pos2(xt(t) - 40.0, gy(1.5)),
    );
    drag(&mut e, &ctx, a, b);
    let tl = e.pj.tl();
    let p = tl.clips[0].vol[1];
    let x = xt(tl.from_track(On::Clip(0), p.0).unwrap());
    eprintln!(
        "volume point (8 s, 0.5) dragged 40 px left and up: ({:.4} s, {:.3}), at x {x:.2} px for the pointer's {:.2}",
        p.0, p.1, b.x
    );
    assert!((x - b.x).abs() < 0.5 && (p.1 - 1.5).abs() < 1e-3, "{p:?}");
    e.sync();
    assert_eq!(e.history(), (u0.0 + 2, 0));
    e.undo();
    assert_eq!(e.pj.tl().clips[0].vol, c0.vol);
    e.undo();
    assert_eq!(e.pj.tl().clips[0].speed, c0.speed);
}

/// A clip's move ends when the timeline changes under it (egui input, no samples needed): the clip dragged deleted
/// mid-drag (Delete), its index another clip's now (the widget's id) or gone, leaves no insertion line, and the release
/// moves nothing.
#[test]
fn move_lost() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let ctx = egui::Context::default();
    let mut e = Editor::new(&g, &cx, &ctx, player::still(&g, &cx), None);
    for (n, k) in [(3, 0), (2, 1)] {
        e.take(Project::new(FPS).unwrap(), Extra::default(), None);
        for i in 0..n {
            let f = e.pj.file(&format!("/nowhere/{i}.insv"), 20.0).unwrap();
            e.ed(|p| p.insert(i, Clip::new(f, 0.0, 4.0)));
        }
        let files = |e: &Editor| e.pj.tl().clips.iter().map(|c| c.file).collect::<Vec<_>>();
        let rect = frame(&mut e, &ctx, vec![]);
        let tl = e.pj.tl();
        let x = |t: f64| rect.left() + 8.0 + t as f32 * e.zoom;
        let y = rect.top() + ROWS[0] + ROWS[1] + 6.0 + ROWS[2] / 2.0;
        let a = egui::pos2(x(tl.out.fps.t(tl.frames(k).start)) + 20.0, y);
        let end = egui::pos2(x(tl.dur()) + 100.0, y);
        let step = |e: &mut Editor, ev: Vec<egui::Event>| {
            frame(e, &ctx, ev);
        };
        step(&mut e, vec![egui::Event::PointerMoved(a)]);
        step(&mut e, vec![btn(a, true)]);
        step(
            &mut e,
            vec![egui::Event::PointerMoved(a + egui::vec2(30.0, 0.0))],
        );
        assert!(
            matches!(e.drag, Some(Drag::Move(j)) if j == k),
            "{n} clips: no move of clip {k}"
        );
        e.delete(k);
        let left = files(&e);
        step(&mut e, vec![egui::Event::PointerMoved(end)]);
        let line = matches!(e.drag, Some(Drag::Move(_)));
        step(&mut e, vec![btn(end, false)]);
        step(&mut e, vec![]);
        eprintln!(
            "{n} clips, clip {k} deleted mid-move: insertion line {line}, files after the release {:?} (left {left:?})",
            files(&e)
        );
        assert!(!line && e.drag.is_none(), "the move went on");
        assert_eq!(files(&e), left, "the release moved a clip");
    }
}

/// Escape cancels a clip's move (egui input, no samples needed): egui ends the drag on it as if it stopped, the button
/// still down, and the move was placed; the release after it moves nothing. A drag released there moves the clip.
#[test]
fn move_escape() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let ctx = egui::Context::default();
    let mut e = Editor::new(&g, &cx, &ctx, player::still(&g, &cx), None);
    for i in 0..3 {
        let f = e.pj.file(&format!("/nowhere/{i}.insv"), 20.0).unwrap();
        e.ed(|p| p.insert(i, Clip::new(f, 0.0, 4.0)));
    }
    let files = |e: &Editor| e.pj.tl().clips.iter().map(|c| c.file).collect::<Vec<_>>();
    let rect = frame(&mut e, &ctx, vec![]);
    let x = |t: f64| rect.left() + 8.0 + t as f32 * e.zoom;
    let y = rect.top() + ROWS[0] + ROWS[1] + 6.0 + ROWS[2] / 2.0;
    let (a, end) = (
        egui::pos2(x(0.0) + 20.0, y),
        egui::pos2(x(e.pj.tl().dur()) + 100.0, y),
    );
    let esc = |pressed| egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers: Default::default(),
    };
    for ev in [
        vec![egui::Event::PointerMoved(a)],
        vec![btn(a, true)],
        vec![egui::Event::PointerMoved(a + egui::vec2(30.0, 0.0))],
        vec![egui::Event::PointerMoved(end)],
    ] {
        frame(&mut e, &ctx, ev);
    }
    assert!(matches!(e.drag, Some(Drag::Move(0))), "no move");
    for ev in [
        vec![esc(true)],
        vec![esc(false)],
        vec![btn(end, false)],
        vec![],
    ] {
        frame(&mut e, &ctx, ev);
    }
    eprintln!(
        "clip 0 moved past the end, Escape, release: files {:?}",
        files(&e)
    );
    assert!(e.drag.is_none(), "the move went on");
    assert_eq!(files(&e), [0, 1, 2], "Escape placed the move");
    drag(&mut e, &ctx, a, end);
    assert_eq!(files(&e), [1, 2, 0], "the drop moved nothing");
}

/// Index-based drags end when the timeline changes under them (egui input, no samples needed): a trim handle, a speed
/// key, a volume point and a view key of clip 1 of 3, each dragged while clip 1 is deleted (Delete) or split at the
/// playhead (S) halfway. The drag ends before the edit as an undo step of its own, the edit is one too, and the rest
/// of the drag changes nothing (a trim went on onto the clip that took the index, with the deleted clip's in and out).
#[test]
fn drag_lost() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let ctx = egui::Context::default();
    let mut e = Editor::new(&g, &cx, &ctx, player::still(&g, &cx), None);
    let v = edit::View::default();
    for kind in ["trim", "speed", "volume", "key"] {
        for split in [false, true] {
            e.take(Project::new(FPS).unwrap(), Extra::default(), None);
            for i in 0..3 {
                let f = e.pj.file(&format!("/nowhere/{i}.insv"), 20.0).unwrap();
                let mut c = Clip::new(f, 2.0, 10.0);
                c.speed = vec![(2.0, 1.0), (6.0, 2.0), (10.0, 1.0)];
                c.vol = vec![(4.0, 1.0), (8.0, 0.5)];
                c.view = [4.0, 7.0]
                    .map(|t| Key {
                        t,
                        v,
                        e: Ease::Smooth,
                    })
                    .into();
                e.ed(|p| p.insert(i, c));
            }
            e.sync();
            let s0 = e.pj.tl().clone();
            let rect = frame(&mut e, &ctx, vec![]);
            let (z, x0) = (e.zoom, rect.left() + 8.0);
            let xt = |t: f64| x0 + t as f32 * z;
            let top = |i: usize| rect.top() + ROWS[..i].iter().map(|h| h + 3.0).sum::<f32>();
            let (vt, vb, at, ab) = (top(2), top(2) + ROWS[2], top(3), top(3) + ROWS[3]);
            let (lo, hi) = (edit::MIN_SPEED.ln(), edit::MAX_SPEED.ln());
            let ys = |v: f64| vb - 4.0 - ((v.ln() - lo) / (hi - lo)) as f32 * (vb - vt - 8.0);
            let gy = |g: f32| ab - 2.0 - (g / 2.0).clamp(0.0, 1.0) * (ab - at - 4.0);
            let x = |s: f64| xt(s0.from_track(On::Clip(1), s).unwrap());
            let n = s0.frames(1);
            let (a, d) = match kind {
                "trim" => (
                    // above the speed keys: one at the next clip's start sits on the edge
                    egui::pos2(xt(s0.out.fps.t(n.end)) - 3.0, vt + 6.0),
                    egui::vec2(-60.0, 0.0),
                ),
                "speed" => (egui::pos2(x(6.0), ys(2.0)), egui::vec2(60.0, -10.0)),
                "volume" => (egui::pos2(x(8.0), gy(0.5)), egui::vec2(-40.0, -8.0)),
                _ => (
                    egui::pos2(x(4.0), top(1) + ROWS[1] / 2.0),
                    egui::vec2(40.0, 0.0),
                ),
            };
            e.seek(s0.out.fps.t((n.start + n.end) / 2));
            let (mut pre, mut post, mut on) = (None, None, "nothing");
            drag_with(&mut e, &ctx, a, a + d, &mut |e| {
                on = match (&e.drag, e.grp) {
                    (Some(Drag::Trim(1, true, ..)), true) => "trim",
                    (Some(Drag::Speed(1, 1, ..)), true) => "speed",
                    (Some(Drag::Vol(1, 1)), true) => "volume",
                    (Some(Drag::Key(On::Clip(1), ..)), true) => "key",
                    _ => "another",
                };
                pre = Some(e.pj.tl().clone());
                if split {
                    e.split();
                } else {
                    e.delete(1);
                }
                post = Some(e.pj.tl().clone());
            });
            let (pre, post, w) = (
                pre.unwrap(),
                post.unwrap(),
                if split { "split" } else { "delete" },
            );
            e.sync();
            eprintln!(
                "{kind} drag, {w} halfway: {on} dragged then, {} clips after, drag over {}, history {:?}",
                post.clips.len(),
                e.drag.is_none() && !e.grp,
                e.history()
            );
            assert!(e.err.is_none(), "{kind}, {w}: {:?}", e.err);
            assert!(on == kind && pre.clips != s0.clips, "{kind}: {on} dragged");
            assert_eq!(
                post.clips.len(),
                if split { 4 } else { 2 },
                "{kind}: no {w}"
            );
            assert!(
                e.pj.tl().clips == post.clips,
                "{kind}, {w}: the drag went on"
            );
            assert!(e.drag.is_none() && !e.grp, "{kind}, {w}: the drag is open");
            e.undo();
            assert!(
                e.pj.tl().clips == pre.clips,
                "{kind}, {w}: not a step of its own"
            );
            e.undo();
            assert!(
                e.pj.tl().clips == s0.clips,
                "{kind}, {w}: the drag not a step"
            );
        }
    }
}

/// The playhead stays on a timeline that gets shorter (no samples needed): paused at 14 s, the last clip deleted
/// leaves it at the new end (it read "0:14.00 / 0:09.41" over the new last frame), and so does undoing the clips.
#[test]
fn clamp() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let mut e = Editor::new(
        &g,
        &cx,
        &egui::Context::default(),
        player::still(&g, &cx),
        None,
    );
    for (i, b) in [9.41, 6.0].into_iter().enumerate() {
        let f = e.pj.file(&format!("/nowhere/{i}.insv"), 20.0).unwrap();
        e.ed(|p| p.insert(i, Clip::new(f, 0.0, b)));
    }
    e.seek(14.0);
    e.delete(1);
    let d = e.pj.tl().dur();
    eprintln!(
        "paused at 14 s, the last clip deleted: {} / {}",
        mmss(e.t),
        mmss(d)
    );
    assert_eq!(e.t, d);
    e.undo();
    assert_eq!(e.t, d, "a longer timeline moved it");
    e.undo();
    e.undo();
    assert_eq!((e.pj.tl().clips.len(), e.t), (0, 0.0));
}

/// The history keeps in step with edit's log (no samples needed): a drag that ends where it started after an undo keeps
/// edit's redo, and so the history's; a Shot Lab replacement that would fail halfway changes nothing (redo kept); no
/// undo mid-drag; a view key dragged past another (egui input) stays in the drag and ends as one step.
#[test]
fn sync() {
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let ctx = egui::Context::default();
    let mut e = Editor::new(&g, &cx, &ctx, player::still(&g, &cx), None);
    let f = e.pj.file("/nowhere/a.insv", 20.0).unwrap();
    let a = |e: &Editor| e.pj.tl().clips.first().map(|c| c.a);
    let state = |e: &Editor, w: &str| {
        let s = format!(
            "{w}: history {:?}, edit's {:?}, a {:?}, denoise {:?}",
            e.history(),
            e.pj.history(),
            a(e),
            e.x.denoise
        );
        eprintln!("{s}");
    };
    e.ed(|p| p.insert(0, Clip::new(f, 0.0, 10.0)));
    e.ed(|p| p.trim(0, 1.0, 10.0));
    e.sync();
    e.x.denoise = Some(0.5);
    e.sync();
    e.ed(|p| p.trim(0, 2.0, 10.0));
    e.undo();
    state(&e, "undone the trim to 2");
    assert_eq!(
        (e.history(), e.pj.history(), a(&e)),
        ((3, 1), (2, 1), Some(1.0))
    );
    // out to 3 and back to 1 in one drag: no step, and the trim to 2 stays to redo
    e.begin();
    e.ed(|p| p.trim(0, 3.0, 10.0));
    e.ed(|p| p.trim(0, 1.0, 10.0));
    e.end();
    e.sync();
    state(&e, "drag out and back");
    assert_eq!((e.history(), e.pj.history()), ((3, 1), (2, 1)));
    e.redo();
    assert_eq!(
        (e.history(), a(&e)),
        ((4, 0), Some(2.0)),
        "the trim to 2 redone"
    );
    e.undo();
    // back in order: the extras step, the trim, the insert
    e.undo();
    state(&e, "undo");
    assert_eq!((a(&e), e.x.denoise), (Some(1.0), None));
    e.undo();
    state(&e, "undo");
    assert_eq!(a(&e), Some(0.0));
    e.undo();
    state(&e, "undo");
    assert_eq!((a(&e), e.history(), e.pj.history()), (None, (0, 4), (0, 3)));
    for _ in 0..3 {
        e.redo();
    }
    state(&e, "redone");
    assert_eq!(
        (a(&e), e.x.denoise, e.history()),
        (Some(1.0), Some(0.5), (3, 1))
    );
    // a replacement whose second clip is bad: nothing changes, the steps undone stay to redo (edit's trim first)
    e.undo();
    e.undo();
    e.shot(
        0,
        Ok(vec![Clip::new(f, 0.0, 5.0), Clip::new(f, 15.0, 25.0)]),
    );
    state(&e, "bad Shot Lab replacement");
    assert!(e.err.is_some());
    assert_eq!(
        (e.pj.tl().clips.len(), a(&e), e.history(), e.pj.history()),
        (1, Some(0.0), (1, 3), (1, 2))
    );
    e.redo();
    assert_eq!((e.pj.tl().clips.len(), a(&e)), (1, Some(1.0)));
    e.redo();
    assert_eq!((e.x.denoise, e.history()), (Some(0.5), (3, 1)));
    e.shot(0, Ok(vec![Clip::new(f, 0.0, 5.0), Clip::new(f, 5.0, 9.0)]));
    e.sync();
    assert_eq!(
        (e.pj.tl().clips.len(), e.history()),
        (2, (4, 0)),
        "one step"
    );
    e.undo();
    assert_eq!((e.pj.tl().clips.len(), a(&e)), (1, Some(1.0)));
    // no undo while a drag is open
    e.begin();
    e.ed(|p| p.trim(0, 4.0, 10.0));
    e.undo();
    assert_eq!(a(&e), Some(4.0));
    e.end();
    e.undo();
    assert_eq!(
        (a(&e), e.history()),
        (Some(1.0), (3, 1)),
        "the drag dropped the shot's redo"
    );
    // a view key dragged past its neighbour keeps the drag and lands under the pointer
    e.whole = true;
    let v = edit::View::default();
    e.ed(|p| p.key(On::Out, 2.0, v, Ease::Smooth));
    e.ed(|p| p.key(On::Out, 4.0, v, Ease::Linear));
    e.sync();
    let u = e.history().0;
    let rect = frame(&mut e, &ctx, vec![]);
    let (z, x0) = (e.zoom, rect.left() + 8.0);
    let y = rect.top() + ROWS[0] + 3.0 + ROWS[1] / 2.0;
    // halfway the pointer is on the key at 2 s, which the dragged one replaces until it moves on: the track is one
    // key short there
    drag(&mut e, &ctx, egui::pos2(x0 + 4.0 * z, y), egui::pos2(x0, y));
    e.sync();
    let ks: Vec<_> = e.pj.tl().out.view.iter().map(|k| (k.t, k.e)).collect();
    eprintln!("view key at 4 s dragged over the one at 2 s to 0 s: {ks:?}");
    assert_eq!(ks.len(), 2, "{ks:?}");
    assert!(ks[0].0.abs() < 1e-3 && ks[0].1 == Ease::Linear && ks[1] == (2.0, Ease::Smooth));
    assert!(e.drag.is_none() && !e.grp, "the drag ended on its widget");
    assert_eq!(e.history().0, u + 1, "one step");
    e.undo();
    assert_eq!(
        e.pj.tl().out.view.iter().map(|k| k.t).collect::<Vec<_>>(),
        [2.0, 4.0]
    );
}

/// The viewer's overlays: a PNG sticker moved and faded keeps its picture on the GPU (its file, deleted after the first
/// build, is not read again) and takes the new place and opacity; a changed picture is made again (and fails without
/// the file); a disc and text before it, removed, leave it in order.
#[test]
fn overs() {
    let g = Arc::new(Gpu::new().unwrap());
    let mut v = Viewer::new(&g);
    let png = out("sticker.png");
    let i = fx::Image::text("hi", 32.0, [255, 0, 0, 255]);
    crate::png::write(&png, i.w, i.h, &i.px).unwrap();
    let flat = |x: f32| fx::Place::Flat {
        at: [x, 0.5],
        w: 0.2,
        rot: 0.0,
    };
    let st = |x: f32, op: f32| Over {
        pic: Pic::Png(png.to_string_lossy().into()),
        place: flat(x),
        opacity: op,
    };
    let disc = Over {
        pic: Pic::Disc([0, 0, 0, 255]),
        place: flat(0.5),
        opacity: 1.0,
    };
    v.overs(&[disc.clone(), st(0.25, 1.0)]);
    assert!(v.err.is_none() && v.ov.iter().all(|o| o.is_some()));
    std::fs::remove_file(&png).unwrap();
    v.overs(&[disc.clone(), st(0.75, 0.5)]);
    let o = v.ov[1].as_ref().expect("kept");
    assert!(v.err.is_none() && o.place == flat(0.75) && o.opacity == 0.5);
    // the disc goes: the sticker moves to index 0 and is made again from its (gone) file
    v.overs(&[st(0.75, 0.5)]);
    assert!(v.ov.len() == 1 && v.ov[0].is_none() && v.err.is_some());
}

/// A render state on the shared device as eframe makes one: the viewer registers its texture there.
fn rs(g: &Gpu) -> RenderState {
    let f = wgpu::TextureFormat::Bgra8Unorm;
    RenderState {
        adapter: g.adapter.clone(),
        available_adapters: vec![],
        instance: g.instance.clone(),
        device: g.device.clone(),
        queue: g.queue.clone(),
        target_format: f,
        renderer: Arc::new(egui::mutex::RwLock::new(eframe::egui_wgpu::Renderer::new(
            &g.device,
            f,
            Default::default(),
        ))),
    }
}

/// The viewer's picture: luma's standard deviation (a cleared frame has none) and the mean RGB difference to `b`.
fn look_at(a: &[u8], b: &[u8]) -> (f64, f64) {
    let y: Vec<f64> = (a.chunks(4))
        .map(|p| (2 * p[0] as u32 + 5 * p[1] as u32 + p[2] as u32) as f64 / 8.0)
        .collect();
    let m = y.iter().sum::<f64>() / y.len() as f64;
    let sd = (y.iter().map(|v| (v - m).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
    let d = (a.chunks(4).zip(b.chunks(4)))
        .map(|(p, q)| (0..3).map(|i| p[i].abs_diff(q[i]) as u32).sum::<u32>())
        .sum::<u32>() as f64
        / (3 * y.len()) as f64;
    (sd, d)
}

/// An edit while paused shows at once on the frame under the playhead and stays (`X5_SAMPLES`, v635 with the Teal &
/// Orange look, the playhead at 2 s, the whole screen each frame, the edits made between the tick and the viewer as the
/// inspector and the keys make them): the look's intensity at 0.3 changes the picture, a mute and a key at the playhead
/// leave it, undo brings the first back. Never black, also frames on (the frame stays the same: no new one comes).
#[test]
fn paused_edit() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let (ctx, rs) = (egui::Context::default(), rs(&g));
    let mut e = Editor::new(&g, &cx, &ctx, player::still(&g, &cx), None);
    e.add(&d.join("v635.insv"), None).unwrap();
    e.ed(|p| p.edit(0, |c| c.look = "Teal & Orange".into()));
    e.seek(2.0);
    let mut n = 0.0;
    let mut frame = |e: &mut Editor, f: &mut dyn FnMut(&mut Editor)| {
        e.tick(true);
        f(e);
        n += 1.0 / 60.0;
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1400.0, 900.0),
            )),
            time: Some(n),
            ..Default::default()
        };
        // egui's own textures: no renderer takes them (epaint's debug check panics on dropped deltas)
        (ctx.run_ui(raw, |ui| e.ui(ui, &rs, &[])).textures_delta).clear();
        thread::sleep(Duration::from_millis(5));
    };
    let pic = |e: &Editor| {
        let t = &e.vw.tgt.as_ref().expect("a picture").tex;
        stitch::read_texture(&g.device, &g.queue, t)
    };
    let t = Instant::now();
    while !e.idle() {
        assert!(t.elapsed().as_secs() < 120, "never idle");
        frame(&mut e, &mut |_| {});
    }
    assert!(e.err.is_none(), "{:?}", e.err);
    let a = pic(&e);
    let raft = e.seam.is_some();
    eprintln!(
        "first picture ({}): luma sd {:.1}",
        if raft { "RAFT stitch" } else { "template seam" },
        look_at(&a, &a).0
    );
    // the edit, then 30 frames on: the picture against `x` (the same, else changed)
    let mut step = |e: &mut Editor,
                    w: &str,
                    f: &mut dyn FnMut(&mut Editor),
                    same: bool,
                    x: &[u8]| {
        frame(e, f);
        let p0 = pic(e);
        for _ in 0..30 {
            frame(e, &mut |_| {});
        }
        let p = pic(e);
        let ((s0, d0), (s, d), (_, da)) = (look_at(&p0, &p), look_at(&p, x), look_at(&p, &a));
        eprintln!(
            "{w}: luma sd {s0:.1} at once, {s:.1} 30 frames on (diff {d0:.2}); diff {d:.2} to the picture expected {}, {da:.2} to the first",
            if same { "the same" } else { "changed" }
        );
        assert!(s0 > 10.0 && s > 10.0, "{w}: black");
        assert!(d0 < 0.5, "{w}: the picture changed frames after the edit");
        assert!(if same { d < 0.5 } else { d > 1.0 }, "{w}: diff {d}");
        p
    };
    let b = step(
        &mut e,
        "intensity 0.3",
        &mut |e| {
            e.ed(|p| p.edit(0, |c| c.amt = 0.3));
        },
        false,
        &a,
    );
    step(
        &mut e,
        "mute",
        &mut |e| {
            e.ed(|p| p.edit(0, |c| c.mute = true));
        },
        true,
        &b,
    );
    step(
        &mut e,
        "key at the playhead",
        &mut |e| e.key_here(),
        true,
        &b,
    );
    step(
        &mut e,
        "undo the key and the mute",
        &mut |e| {
            e.undo();
            e.undo();
        },
        true,
        &b,
    );
    step(&mut e, "undo the intensity", &mut |e| e.undo(), true, &a);
    assert!(e.err.is_none(), "{:?}", e.err);
}

/// Music whose container gives no length (of a 6 s chord encoded by the ffmpeg CLI into a pipe: AAC, MP3, Vorbis, FLAC;
/// those the reader knows the length of anyway are left out) is counted on a thread: the import returns at once
/// and the music goes in, 6 s long, once counted; an import meanwhile supersedes it.
#[test]
fn count() {
    let w = out("count.wav");
    chord(&w, 6.0);
    let mut v = vec![];
    for (f, args) in [
        ("count.aac", &["-c:a", "aac", "-f", "adts"][..]),
        ("count.mp3", &["-c:a", "libmp3lame", "-f", "mp3"][..]),
        ("count.ogg", &["-c:a", "libvorbis", "-f", "ogg"][..]),
        // through a pipe: the STREAMINFO keeps no sample count
        ("count.flac", &["-c:a", "flac", "-f", "flac"][..]),
    ] {
        let o = out(f);
        let ok = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(&w)
            .args(args)
            .arg("pipe:1")
            .stdout(std::fs::File::create(&o).unwrap())
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            eprintln!("ffmpeg could not make {f}: skipped");
            continue;
        }
        let d = media::music::Music::open(&o).unwrap().duration;
        eprintln!("{f}: the container says {d} s");
        if d == 0.0 {
            v.push(o);
        }
    }
    assert!(!v.is_empty(), "no file without a length");
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let mut e = Editor::new(
        &g,
        &cx,
        &egui::Context::default(),
        player::still(&g, &cx),
        None,
    );
    for o in &v {
        let t = Instant::now();
        e.music(o).unwrap();
        let back = t.elapsed().as_secs_f64();
        assert!(e.pj.tl().out.music.is_none() && !e.settled());
        while e.mq.0.is_some() {
            assert!(t.elapsed().as_secs() < 30, "not counted");
            e.tick(true);
            thread::sleep(Duration::from_millis(2));
        }
        let tl = e.pj.tl();
        let d = tl.files[tl.out.music.as_ref().unwrap().file as usize].dur;
        eprintln!(
            "{}: import back after {:.1} ms, counted {d:.3} s after {:.1} ms",
            o.display(),
            back * 1e3,
            t.elapsed().as_secs_f64() * 1e3
        );
        // the encoder's priming and padding: within a frame or two
        assert!((d - 6.0).abs() < 0.1, "{d}");
        e.ed(|p| p.edit_out(|o| o.music = None));
    }
    // a WAV (its length known) imported while a count runs wins
    e.music(&v[0]).unwrap();
    e.music(&w).unwrap();
    let t = Instant::now();
    while t.elapsed().as_secs_f64() < 2.0 {
        e.tick(true);
        thread::sleep(Duration::from_millis(5));
    }
    let tl = e.pj.tl();
    assert_eq!(
        tl.files[tl.out.music.as_ref().unwrap().file as usize].path,
        w.to_string_lossy()
    );
}

/// The stabilisation poses follow the setting (`X5_SAMPLES`, v606; added as one undo step with its frame rate): through
/// changes and back to a setting used before (whose poses another setting's had replaced), and through quick changes,
/// where the replies for the settings left on the way are dropped and one thread at a time works per file.
#[test]
fn poses() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let mut e = Editor::new(
        &g,
        &cx,
        &egui::Context::default(),
        player::still(&g, &cx),
        None,
    );
    // a first clip at another rate than the timeline's: its rate and the clip are one undo step
    e.pj = Project::new(Rate { num: 25, den: 1 }).unwrap();
    e.add(&d.join("v606.insv"), None).unwrap();
    e.sync();
    let fps = |e: &Editor| (e.pj.tl().out.fps.num, e.pj.tl().clips.len());
    assert_eq!((e.history(), fps(&e)), ((1, 0), (30000, 1)));
    e.undo();
    assert_eq!(fps(&e), (25, 0));
    e.redo();
    assert_eq!(fps(&e), (30000, 1));
    let path = e.pj.tl().files[0].path.clone();
    let wait = |e: &mut Editor| {
        let t = Instant::now();
        while !e.posed() {
            assert!(t.elapsed().as_secs() < 60, "no poses for {:?}", e.pkey());
            e.tick(true);
            thread::sleep(Duration::from_millis(5));
        }
        let i = e.info[&path].as_ref().unwrap();
        let (k, p) = i.poses.as_ref().unwrap();
        assert!(*k == e.pkey() && p.is_ok(), "{k:?}");
        // stabilised: the frames turn
        assert!((0..300).any(|n| i.pose(n, *k).q.dot(i.frame.q).abs() < 0.9999));
        t.elapsed().as_secs_f64()
    };
    let (h, fr) = (Some(stab::Lock::Horizon), Some(stab::Lock::Free));
    for (l, s) in [(h, 0.5), (h, 0.55), (h, 0.5), (fr, 0.5), (h, 0.5)] {
        e.stab(l, s);
        let w = wait(&mut e);
        eprintln!("poses for {l:?} {s}: in after {w:.2} s");
    }
    // a reply for a setting left meanwhile is dropped and the current setting asked for next
    let k = |s: f64| (h, s.to_bits());
    e.stab(h, 0.6);
    e.stab(h, 0.65);
    assert_eq!(e.asked.get(&path), Some(&k(0.6)), "one thread per file");
    let t = Instant::now();
    while e.asked.get(&path) == Some(&k(0.6)) {
        assert!(t.elapsed().as_secs() < 60, "no reply");
        e.tick(true);
        thread::sleep(Duration::from_millis(5));
    }
    let p = e.info[&path].as_ref().unwrap().poses.as_ref().unwrap().0;
    assert_eq!((p, e.asked.get(&path)), (k(0.5), Some(&k(0.65))));
    let w = wait(&mut e);
    eprintln!("0.6 → 0.65 while 0.6 was made: 0.6 dropped, 0.65 in after {w:.2} s");
    e.stab(None, 0.6);
    assert!(e.posed(), "off needs no poses");
}

/// Deep Track in the editor (`X5_SAMPLES`, v606 trimmed to 0–4 s): a box around the woman in the purple coat in the
/// viewer's view (yaw −71°, pitch −35°, 110°) at the playhead becomes the clip's reframing keys, one undo step: keys
/// every 0.5 s over the clip, the view's field of view kept, each aimed within a few degrees of her box; the viewer
/// then follows the keys.
#[test]
fn deep_track() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let g = Arc::new(Gpu::new().unwrap());
    let cx = stitch::Ctx::new(&g.device, &g.queue);
    let mut e = Editor::new(
        &g,
        &cx,
        &egui::Context::default(),
        player::still(&g, &cx),
        None,
    );
    e.add(&d.join("v606.insv"), None).unwrap();
    e.ed(|p| p.trim(0, 0.0, 4.0));
    e.free = Some(View {
        yaw: -71.0,
        pitch: -35.0,
        fov: 110.0,
        ..View::default()
    });
    let t = Instant::now();
    while e.cur().is_none() || !e.posed() {
        assert!(t.elapsed().as_secs() < 60, "no frame");
        e.tick(true);
        thread::sleep(Duration::from_millis(5));
    }
    e.sync();
    let h0 = e.history().0;
    // the box of the player's run, for the 16:9 picture the editor frames by default
    e.track([0.41, 0.25, 0.58, 0.83]).unwrap();
    while e.trk.as_ref().is_some_and(|t| !t.keyed) {
        assert!(t.elapsed().as_secs() < 600, "no keys");
        e.tick(true);
        thread::sleep(Duration::from_millis(20));
    }
    assert!(e.err.is_none(), "{:?}", e.err);
    e.sync();
    assert_eq!(e.history().0, h0 + 1, "one undo step");
    let tr = e.trk.as_ref().unwrap().t.clone();
    let c = &e.pj.tl().clips[0];
    eprintln!(
        "{} keys over {:.2}–{:.2} s from {} samples in {:.1} s (GPU shared: noisy)",
        c.view.len(),
        c.view[0].t,
        c.view[c.view.len() - 1].t,
        tr.smp.lock().unwrap().len(),
        t.elapsed().as_secs_f64()
    );
    assert!(c.view.len() >= 8 && c.view[0].t < 0.05 && c.view[c.view.len() - 1].t > 3.9);
    for k in &c.view {
        let s = tr.at(tr.frame(k.t)).unwrap();
        let [y, p, _] = k.v.ypr();
        let a = track::dir(y, p).angle_between(s.raw).to_degrees();
        assert!(
            a < s.size.y / 3.0 && (k.v.fov - 110.0).abs() < 1e-3,
            "{}: {a:.1}° off",
            k.t
        );
    }
    assert!(e.free.is_none(), "the viewer follows the keys");
    e.undo();
    assert!(e.pj.tl().clips[0].view.is_empty());
}
