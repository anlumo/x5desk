//! The forward view's analysis (`export::flow`, `stab::fwd`) on the X5's footage: `X5_SAMPLES` (v606 walking, v627 at
//! night, v635 standing) and the rides listed in `X5_RIDES` (paths separated by `:`); skipped when neither is set.
//! The features of each clip are cached in `X5_FIXTURES` (default: Cargo's test tmp directory; `X5_FWD_W` another
//! panorama width), the estimates and the per-frame headings written there as CSV (`estimates`), contact sheets to
//! `X5_FWD_OUT` (`sheets`); `cost` times the pass, `job` runs a forward export twice. Run with `--release`.

use glam::{DQuat, DVec3};
use stab::fwd::{self, Pair};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, atomic::AtomicBool},
    time::Instant,
};
use stitch::Ctx;
use video::Gpu;

fn gpu() -> (MutexGuard<'static, ()>, Arc<Gpu>, Ctx) {
    static G: OnceLock<(Mutex<()>, Arc<Gpu>, Ctx)> = OnceLock::new();
    static GATE: RwLock<()> = RwLock::new(());
    static BURN: Mutex<()> = Mutex::new(());
    let (m, g, cx) = G.get_or_init(|| {
        let g = Arc::new(Gpu::new().expect("GPU"));
        let cx = Ctx::new(&g.device, &g.queue);
        export::locks(&GATE, &BURN);
        (Mutex::new(()), g, cx)
    });
    (
        m.lock().unwrap_or_else(|e| e.into_inner()),
        g.clone(),
        cx.clone(),
    )
}

fn fixtures() -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The samples, then the rides.
fn clips() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::env::var_os("X5_SAMPLES")
        .map(PathBuf::from)
        .into_iter()
        .flat_map(|d| ["v606", "v627", "v635"].map(|n| d.join(format!("{n}.insv"))))
        .collect();
    if let Some(r) = std::env::var_os("X5_RIDES") {
        v.extend(std::env::split_paths(&r));
    }
    if v.is_empty() {
        eprintln!("X5_SAMPLES and X5_RIDES unset: skipped");
    }
    v
}

fn stem(f: &Path) -> String {
    f.file_stem().unwrap().to_string_lossy().into()
}

/// The analysis' panorama width: `X5_FWD_W`, else the app's.
fn width() -> u32 {
    std::env::var("X5_FWD_W")
        .ok()
        .and_then(|w| w.parse().ok())
        .unwrap_or(export::FWD_W)
}

/// The CSV files' tag: empty at the app's width, else `_<width>`.
fn tag() -> String {
    match width() {
        w if w == export::FWD_W => String::new(),
        w => format!("_{w}"),
    }
}

/// The pairs of clip `f` (cached; the GPU only to make them): per pair t0 t1, r0 r1 (xyzw), the feature count, then
/// the rays (f32).
fn pairs(f: &Path) -> Vec<Pair> {
    let c = fixtures().join(format!("{}_fwd{}.pairs", stem(f), width()));
    if let Ok(b) = std::fs::read(&c) {
        let (mut out, mut b) = (vec![], &b[..]);
        let f64s = |b: &mut &[u8], n: usize| -> Vec<f64> {
            let (x, r) = b.split_at(8 * n);
            *b = r;
            x.as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect()
        };
        while !b.is_empty() {
            let h = f64s(&mut b, 10);
            let q = |i: usize| DQuat::from_xyzw(h[i], h[i + 1], h[i + 2], h[i + 3]);
            let (n, r) = b.split_first_chunk::<4>().unwrap();
            let (m, r) = r.split_at(24 * u32::from_le_bytes(*n) as usize);
            b = r;
            let v = |c: &[u8], i: usize| {
                f32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap()) as f64
            };
            out.push(Pair {
                t: [h[0], h[1]],
                r: [q(2), q(6)],
                m: m.as_chunks::<24>()
                    .0
                    .iter()
                    .map(|c| {
                        [
                            DVec3::new(v(c, 0), v(c, 1), v(c, 2)),
                            DVec3::new(v(c, 3), v(c, 4), v(c, 5)),
                        ]
                    })
                    .collect(),
            });
        }
        return out;
    }
    let (_l, g, cx) = gpu();
    let (t0, mut out, mut n) = (Instant::now(), vec![], 0);
    export::flow(
        &g,
        &cx,
        &[f.to_path_buf()],
        (width(), export::FWD_DT),
        &AtomicBool::new(false),
        &mut |i, _| n = i,
        |p| {
            out.push(p);
            Ok(())
        },
    )
    .unwrap();
    let s = t0.elapsed().as_secs_f64();
    let dur = out.last().map_or(0.0, |p| p.t[1]) - out.first().map_or(0.0, |p| p.t[0]);
    eprintln!(
        "{}: {n} frames analysed over {dur:.1} s in {s:.1} s ({:.2} s per minute of footage)",
        stem(f),
        s / dur * 60.0
    );
    let mut b = vec![];
    for p in &out {
        for x in [p.t[0], p.t[1]]
            .into_iter()
            .chain(p.r[0].to_array())
            .chain(p.r[1].to_array())
        {
            b.extend(x.to_le_bytes());
        }
        b.extend((p.m.len() as u32).to_le_bytes());
        for v in p.m.iter().flatten() {
            for x in v.to_array() {
                b.extend((x as f32).to_le_bytes());
            }
        }
    }
    std::fs::write(&c, b).unwrap();
    out
}

/// Per clip: the estimates as CSV (time, direction, votes, inliers, rates, the camera's heading: the azimuth of the
/// body's X axis, which the X5's framing centres).
#[test]
fn estimates() {
    for f in clips() {
        let ps = pairs(&f);
        let t0 = Instant::now();
        let es: Vec<(fwd::Est, f64)> = ps
            .iter()
            .map(|p| {
                let x = p.r[1] * DVec3::X + p.r[0] * DVec3::X;
                (fwd::fit(p), x.y.atan2(x.x))
            })
            .collect();
        let s = t0.elapsed().as_secs_f64();
        let mut csv = String::from("t,x,y,z,n,inl,rate,ground,cam,feat\n");
        for ((e, cam), p) in es.iter().zip(&ps) {
            csv += &format!(
                "{:.4},{:.5},{:.5},{:.5},{},{},{:.4},{:.4},{cam:.5},{}\n",
                e.t,
                e.dir.x,
                e.dir.y,
                e.dir.z,
                e.n,
                e.inl,
                e.rate,
                e.ground,
                p.m.len()
            );
        }
        std::fs::write(
            fixtures().join(format!("{}_fwd{}.csv", stem(&f), tag())),
            csv,
        )
        .unwrap();
        // per frame: the view's heading under the forward view and the horizon lock (σ 0.5 s), the camera's own
        let tr = media::insv::Trailer::open(&f).unwrap();
        // the course over ground where the camera had GPS (from the phone app): none of ours has
        let gps: u32 = (tr.recs.iter())
            .filter(|r| r.kind == media::insv::GPS)
            .map(|r| r.size)
            .sum();
        eprintln!("{}: GPS record {gps} bytes", stem(&f));
        let (imu, ex) = (tr.imu().unwrap(), tr.frames(media::insv::EXPOSURE).unwrap());
        let tk = stab::Track::new(&imu, stab::KP, stab::KI);
        let ahead: Vec<fwd::Est> = es.iter().map(|e| e.0).collect();
        let pose = |lock| {
            let o = stab::Opts {
                ahead: &ahead,
                ..stab::Opts::new(&tr.meta, lock, 0.5)
            };
            stab::poses(&tk, &ex, &o)
        };
        let (pf, ph) = (pose(stab::Lock::Forward), pose(stab::Lock::Horizon));
        let mut csv = String::from("t,fwd,hor,cam\n");
        for (k, e) in ex.iter().enumerate() {
            let t = e.t as f64 * 1e-6 - 0.5 * e.time + stab::OFFSET;
            let r = tk.at(t);
            let az = |v: DVec3| v.y.atan2(v.x);
            csv += &format!(
                "{t:.4},{:.5},{:.5},{:.5}\n",
                az(r * pf[k].q.as_dquat() * DVec3::X),
                az(r * ph[k].q.as_dquat() * DVec3::X),
                az(r * DVec3::X)
            );
        }
        std::fs::write(
            fixtures().join(format!("{}_fwd{}_frames.csv", stem(&f), tag())),
            csv,
        )
        .unwrap();
        let ok = es.iter().filter(|e| e.0.dir != DVec3::ZERO).count();
        eprintln!(
            "{}: {ok} of {} pairs fitted in {s:.2} s",
            stem(&f),
            es.len()
        );
    }
}

/// Where the contact sheets go: `X5_FWD_OUT`, else the fixtures.
fn sheets_out() -> PathBuf {
    let d = std::env::var_os("X5_FWD_OUT").map_or_else(fixtures, PathBuf::from);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Bilinear RGBA of an equirect (`stab::vis::ray`'s layout) in direction `d`.
fn sample(px: &[u8], w: usize, h: usize, d: DVec3) -> [f64; 3] {
    let d = d.normalize();
    let th = (-d.y).atan2(d.x).rem_euclid(std::f64::consts::TAU);
    let (x, y) = (
        th * w as f64 / std::f64::consts::TAU,
        d.z.clamp(-1.0, 1.0).acos() * (h - 1) as f64 / std::f64::consts::PI,
    );
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |x: f64, y: f64| {
        let (x, y) = (
            (x as isize).rem_euclid(w as isize) as usize,
            (y as usize).min(h - 1),
        );
        let p = &px[4 * (y * w + x)..];
        [p[0], p[1], p[2]].map(f64::from)
    };
    let (a, b, c, e) = (
        at(x0, y0),
        at(x0 + 1.0, y0),
        at(x0, y0 + 1.0),
        at(x0 + 1.0, y0 + 1.0),
    );
    std::array::from_fn(|i| {
        (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + e[i] * fx) * fy
    })
}

/// A perspective tile (`tw` × `th`, horizontal field of view `fov`°) of the equirect looking along `f` with `up`
/// upward, into `out` (RGBA, `ow` wide) at (`x0`, `y0`); marks: (direction, colour) as rings.
#[allow(clippy::too_many_arguments)]
fn tile(
    px: &[u8],
    (w, h): (usize, usize),
    f: DVec3,
    up: DVec3,
    fov: f64,
    (tw, th): (usize, usize),
    out: &mut [u8],
    ow: usize,
    (x0, y0): (usize, usize),
    marks: &[(DVec3, [u8; 3])],
) {
    let f = f.normalize();
    let up = (up - f * up.dot(f)).normalize();
    let r = f.cross(up);
    let s = (fov.to_radians() / 2.0).tan();
    for j in 0..th {
        for i in 0..tw {
            let (u, v) = (
                (2.0 * (i as f64 + 0.5) / tw as f64 - 1.0) * s,
                (2.0 * (j as f64 + 0.5) / th as f64 - 1.0) * s * th as f64 / tw as f64,
            );
            let c = sample(px, w, h, f + r * u - up * v);
            let o = 4 * ((y0 + j) * ow + x0 + i);
            out[o..o + 3].copy_from_slice(&c.map(|x| x as u8));
            out[o + 3] = 255;
        }
    }
    // the centre's cross, white
    let (cx, cy) = (x0 + tw / 2, y0 + th / 2);
    for d in 4..12 {
        for (x, y) in [(cx + d, cy), (cx - d, cy), (cx, cy + d), (cx, cy - d)] {
            out[4 * (y * ow + x)..][..3].copy_from_slice(&[255; 3]);
        }
    }
    for &(m, col) in marks {
        let m = m.normalize();
        let z = m.dot(f);
        if z <= 0.1 {
            continue;
        }
        let (u, v) = (m.dot(r) / z / s, -m.dot(up) / z / s * tw as f64 / th as f64);
        let (px_, py) = ((u + 1.0) / 2.0 * tw as f64, (v + 1.0) / 2.0 * th as f64);
        for a in 0..48 {
            let t = a as f64 / 48.0 * std::f64::consts::TAU;
            for rr in [7.0, 8.0] {
                let (x, y) = (px_ + rr * t.cos(), py + rr * t.sin());
                if x >= 0.0 && y >= 0.0 && (x as usize) < tw && (y as usize) < th {
                    let o = 4 * ((y0 + y as usize) * ow + x0 + x as usize);
                    out[o..o + 3].copy_from_slice(&col);
                }
            }
        }
    }
}

/// Contact sheets of the forward view every 3 s (`X5_FWD_OUT`/`<clip>_forward.jpg`): per moment the forward view
/// (top; 100° wide, level) and the horizon lock's (below: the camera's heading), each with its centre's cross, the
/// moment's unsmoothed direction of travel as a red ring (when travelling), and the forward view's centre as a green
/// ring in the horizon lock's view. Row-major in time. The clip is exported unstabilised as a 1920 × 960 equirect
/// and the views are cut from it on the CPU with the poses `stab::poses` gives for each lock.
#[test]
fn sheets() {
    use export::{Cut, Enc, Fx, Job, Out, Queue, Seg, State};
    use media::insv::{EXPOSURE, Trailer};
    let (ew, eh, every, cols, (tw, th)) = (1920usize, 960usize, 3.0, 6usize, (400usize, 225usize));
    for f in clips() {
        let ps = pairs(&f);
        let (_l, g, cx) = gpu();
        let es: Vec<fwd::Est> = ps.iter().map(fwd::fit).collect();
        let tr = Trailer::open(&f).unwrap();
        let (imu, ex) = (tr.imu().unwrap(), tr.frames(EXPOSURE).unwrap());
        let tk = stab::Track::new(&imu, stab::KP, stab::KI);
        let fps = tr.meta.fps;
        let n = ex.len();
        let step = (every * fps).round() as u64;
        let ks: Vec<u64> = (0..n as u64).step_by(step as usize).collect();
        // the frames captured as the unstabilised export renders them
        let got = Arc::new(Mutex::new(vec![]));
        let (g2, ks2, o) = (g.clone(), ks.clone(), got.clone());
        let hook = move |fx: Fx<'_>| {
            if !ks2.contains(&fx.k) {
                return;
            }
            let t = g2.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: fx.tex.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fx.tex.format(),
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            fx.enc.copy_texture_to_texture(
                fx.tex.as_image_copy(),
                t.as_image_copy(),
                fx.tex.size(),
            );
            o.lock().unwrap().push((fx.k, t));
        };
        let path = fixtures().join(format!("{}_sheet_src.mp4", stem(&f)));
        let _ = std::fs::remove_file(&path);
        let q = Queue::new(g.clone(), cx.clone());
        let id = q.push(Job {
            cut: Cut::Segs(vec![Seg {
                files: vec![f.clone()],
                t: [0.0, f64::INFINITY],
            }]),
            path: path.clone(),
            out: Out::Pano(ew as u32),
            enc: Enc::Video {
                codec: video::Codec::H264,
                deep: false,
                rate: video::Rate::Cbr(4_000_000),
                gop: 30,
            },
            raft: false,
            colour: Some(0.1),
            stab: None,
            fx: Some(Box::new(hook)),
            flat: false,
            pre: None,
        });
        loop {
            match q.state(id).unwrap() {
                State::Done(_) => break,
                State::Failed(e) => panic!("export failed: {e}"),
                State::Cancelled => panic!("cancelled"),
                _ => std::thread::sleep(std::time::Duration::from_millis(50)),
            }
        }
        let _ = std::fs::remove_file(&path);
        let mut frames: Vec<(u64, Vec<u8>)> = std::mem::take(&mut *got.lock().unwrap())
            .into_iter()
            .map(|(k, t)| (k, stitch::read_texture(&g.device, &g.queue, &t)))
            .collect();
        frames.sort_by_key(|x| x.0);
        let pose = |lock| {
            let o = stab::Opts {
                ahead: &es,
                ..stab::Opts::new(&tr.meta, lock, 0.5)
            };
            stab::poses(&tk, &ex, &o)
        };
        let (pf, ph) = (pose(stab::Lock::Forward), pose(stab::Lock::Horizon));
        // the X5 frames on lens 1
        let fr = stitch::Pose::euler(0.0, 0.0, if tr.meta.stream == 4 { 180.0 } else { 0.0 })
            .q
            .as_dquat();
        let rows = frames.len().div_ceil(cols);
        let (sw, sh) = (cols * tw, rows * 2 * th);
        let mut sheet = vec![0u8; sw * sh * 4];
        for (i, (k, px)) in frames.iter().enumerate() {
            let k = *k as usize;
            let e = ex[k.min(n - 1)];
            let t = e.t as f64 * 1e-6 - 0.5 * e.time + stab::OFFSET;
            // world -> the unstabilised output
            let wo = (tk.at(t) * fr).inverse();
            let centre = |p: &[stab::Pose]| tk.at(t) * p[k].q.as_dquat() * fr * DVec3::NEG_X;
            let (cf, ch) = (centre(&pf), centre(&ph));
            let raw = es
                .iter()
                .filter(|e| fwd::moving(e))
                .min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
                .filter(|e| (e.t - t).abs() < 0.2)
                .map(|e| (wo * e.dir, [255, 40, 40]));
            let (x0, y0) = ((i % cols) * tw, (i / cols) * 2 * th);
            let up = wo * DVec3::Z;
            let m: Vec<_> = raw.into_iter().collect();
            tile(
                px,
                (ew, eh),
                wo * cf,
                up,
                100.0,
                (tw, th),
                &mut sheet,
                sw,
                (x0, y0),
                &m,
            );
            let mut m2 = m.clone();
            m2.push((wo * cf, [40, 255, 40]));
            tile(
                px,
                (ew, eh),
                wo * ch,
                up,
                100.0,
                (tw, th),
                &mut sheet,
                sw,
                (x0, y0 + th),
                &m2,
            );
        }
        let jpg = export::jpeg::encode(&sheet, sw as u32, sh as u32, 85, None).unwrap();
        let out = sheets_out().join(format!("{}_forward.jpg", stem(&f)));
        std::fs::write(&out, jpg).unwrap();
        eprintln!(
            "{}: {} moments -> {}",
            stem(&f),
            frames.len(),
            out.display()
        );
    }
}

/// What the analysis costs (`X5_FWD_COST`: a recording; skipped when unset): seconds per minute of footage for a
/// frame a second (about the decoding alone), then panoramas 2048 and 1024 wide every 15th of a second, tracked and
/// fitted.
#[test]
fn cost() {
    let Some(f) = std::env::var_os("X5_FWD_COST").map(PathBuf::from) else {
        eprintln!("X5_FWD_COST unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    for (w, dt) in [(2048, 1.0), (2048, 1.0 / 15.0), (1024, 1.0 / 15.0)] {
        let (t0, mut span, mut n) = (Instant::now(), [f64::MAX, 0.0], 0);
        export::flow(
            &g,
            &cx,
            std::slice::from_ref(&f),
            (w, dt),
            &AtomicBool::new(false),
            &mut |i, _| n = i,
            |p| {
                span = [span[0].min(p.t[0]), span[1].max(p.t[1])];
                fwd::fit(&p);
                Ok(())
            },
        )
        .unwrap();
        let s = t0.elapsed().as_secs_f64();
        eprintln!(
            "{}: {w} wide every {dt:.3} s: {n} frames in {s:.1} s, {:.1} s per minute of footage",
            stem(&f),
            s / (span[1] - span[0]) * 60.0
        );
    }
}

/// A job with the forward view (v606, 2 s as a flat view): its plan runs the clip's analysis and keeps it, so a
/// second job starts at once and `travelled` has it.
#[test]
fn job() {
    use export::{Cut, Enc, Job, Out, Queue, Seg, State};
    let Some(f) = std::env::var_os("X5_SAMPLES").map(|d| PathBuf::from(d).join("v606.insv")) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let (_l, g, cx) = gpu();
    let q = Queue::new(g.clone(), cx.clone());
    let run = |name: &str| {
        let path = fixtures().join(name);
        let _ = std::fs::remove_file(&path);
        let t0 = Instant::now();
        let id = q.push(Job {
            cut: Cut::Segs(vec![Seg {
                files: vec![f.clone()],
                t: [10.0, 12.0],
            }]),
            path: path.clone(),
            out: Out::Flat {
                w: 640,
                h: 360,
                view: None,
            },
            enc: Enc::Video {
                codec: video::Codec::H264,
                deep: false,
                rate: video::Rate::Cbr(2_000_000),
                gop: 30,
            },
            raft: false,
            colour: Some(0.1),
            stab: Some((stab::Lock::Forward, 0.5)),
            fx: None,
            flat: false,
            pre: None,
        });
        let p = loop {
            match q.state(id).unwrap() {
                State::Done(p) => break p,
                State::Failed(e) => panic!("export failed: {e}"),
                State::Cancelled => panic!("cancelled"),
                _ => std::thread::sleep(std::time::Duration::from_millis(20)),
            }
        };
        let _ = std::fs::remove_file(&path);
        (p.done, t0.elapsed().as_secs_f64())
    };
    assert!(export::travelled(std::slice::from_ref(&f)).is_none());
    let (a, b) = (run("v606_forward_a.mp4"), run("v606_forward_b.mp4"));
    eprintln!(
        "v606 forward, 2 s: {} frames in {:.1} s with the analysis, {} in {:.1} s kept",
        a.0, a.1, b.0, b.1
    );
    assert!(a.0 >= 59 && b.0 == a.0 && b.1 < a.1 / 3.0);
    let es = export::travelled(std::slice::from_ref(&f)).unwrap();
    assert!(es.iter().filter(|e| fwd::moving(e)).count() > es.len() / 2);
}
