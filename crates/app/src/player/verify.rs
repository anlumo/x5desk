//! Stabilisation measured on the player's own pictures: equirects rendered by a source's preview with each setting's
//! poses and read back, the inter-frame rotation fitted with `stab::vis` (as crate stab's footage tests do), the
//! horizon tilt, and the proxy against the full-resolution video. `X5_SAMPLES` (skipped when unset); CSVs and the
//! synthetic proxy go to `X5_FIXTURES` (default: the temp dir); ffmpeg cuts the proxy (test tool). Run with
//! `--release`.

use super::*;
use glam::DVec3;
use media::insv::{EXPOSURE, GYRO};
use stab::{Lock, log, vis};
use std::{
    f64::consts::{PI, TAU},
    io::{Read as _, Seek as _},
    process::Command,
};

/// Tracked polar angles: the operator stays below 115°.
const BAND: [f64; 2] = [15.0 * PI / 180.0, 115.0 * PI / 180.0];

fn samples() -> Option<PathBuf> {
    let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        eprintln!("X5_SAMPLES not set: skipping");
    }
    d
}

fn fixtures() -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES").map_or_else(std::env::temp_dir, PathBuf::from);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The player's pieces without a window.
struct Rig {
    g: Arc<Gpu>,
    cx: Ctx,
    ctx: egui::Context,
    gyro: mpsc::Sender<gyro::Req>,
}

impl Rig {
    fn new() -> Rig {
        let g = Arc::new(Gpu::new().unwrap());
        Rig {
            cx: Ctx::new(&g.device, &g.queue),
            g,
            ctx: egui::Context::default(),
            gyro: gyro::worker(),
        }
    }

    /// A source opened as the player opens it, seeking frame `k`.
    fn src(&self, files: &[PathBuf], m: &Meta, iw: f32, k: usize) -> Src {
        let mut s = Src::open(&self.g, &self.cx, files, m, iw, &self.ctx).unwrap();
        s.v.seek(&self.g, k, None);
        s
    }

    /// The stabilisation of `file` (`s` plays it) once its poses are in.
    fn stab(&self, file: &Path, s: &Src, m: &Meta, mode: gyro::Mode) -> Stab {
        let yaw = if m.stream == 4 { 180.0 } else { 0.0 };
        let mut st = Stab::new(
            &self.gyro,
            file.into(),
            s.v.times.clone(),
            s.ff,
            yaw,
            mode,
            gyro::SMOOTH,
            &self.ctx,
        );
        let t0 = Instant::now();
        while !st.ready() {
            st.tick();
            assert!(t0.elapsed().as_secs() < 30, "poses");
            thread::sleep(Duration::from_millis(1));
        }
        assert!(st.err.is_none(), "{:?}", st.err);
        st
    }

    /// Frame `k` of `s` (the one after the last, or a seek target); the frame before is released.
    fn next(&self, s: &Src, k: usize) -> Pic {
        let t0 = Instant::now();
        let p = loop {
            if let (Some(p), _) = s.v.take(s.v.times[k], false) {
                break p;
            }
            assert!(t0.elapsed().as_secs() < 10, "frame {k} stalled");
            thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(p.k, k);
        s.v.release_before(&self.g, &p);
        p
    }

    fn target(&self, w: usize, h: usize) -> wgpu::Texture {
        self.g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("readback"),
            size: wgpu::Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FMT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// `p` of `s` drawn as the player draws it (pose, then the preview; `colour` for a new frame), in the equirect
    /// view filling `t`, read back as RGBA8.
    fn shot(&self, s: &Src, p: &Pic, o: &Pose, t: &wgpu::Texture, colour: bool) -> Vec<u8> {
        let src = s.src(p);
        s.pv.pose(o);
        let mut e = self.g.device.create_command_encoder(&Default::default());
        if colour {
            s.pv.colour(&mut e, src);
        }
        let v = t.create_view(&Default::default());
        let size = [t.width(), t.height()];
        s.pv.render(&mut e, &v, size, &Look::Equirect.view(), src);
        s.v.acquire(&self.g, p);
        self.g.queue.submit([e.finish()]);
        stitch::read_texture(&self.g.device, &self.g.queue, t)
    }
}

/// Rotation vector (rad, output frame) taking equirect `a` to `b` and the RMS of its inliers (px); None below 40
/// inliers. The preview's
/// equirect puts pixel (x, y) at θ = 2π(x + ½)/W, φ = π(y + ½)/H, `vis::ray`'s layout at θ = 2πx/W, φ = πy/(H − 1).
fn turn(a: &vis::Pano, b: &vis::Pano, m: &mut Vec<[[f32; 2]; 2]>) -> Option<(DVec3, f64)> {
    vis::track(a, b, BAND, m);
    let (w, h) = (a.w, a.h);
    let r = |p: [f32; 2]| {
        let y = (p[1] as f64 + 0.5) * (h - 1) as f64 / h as f64;
        vis::ray(w, h, [p[0] as f64 + 0.5, y])
    };
    let rays: Vec<_> = m.iter().map(|p| (r(p[0]), r(p[1]))).collect();
    // RANSAC inlier chord: 3 px
    let f = vis::fit(&rays, 3.0 * TAU / w as f64)?;
    (f.n >= 40).then(|| (log(f.q), f.rms * w as f64 / TAU))
}

/// Mean, 95th percentile, max.
fn stats(v: &[f64]) -> [f64; 3] {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len().max(1);
    [
        s.iter().sum::<f64>() / n as f64,
        s.get(n * 95 / 100).copied().unwrap_or(f64::NAN),
        s.last().copied().unwrap_or(f64::NAN),
    ]
}

/// Jitter: each inter-frame rotation minus the mean of its ±7 neighbours, RMS in degrees.
fn jitter(v: &[DVec3]) -> f64 {
    let s: f64 = (0..v.len())
        .map(|i| {
            let r = i.saturating_sub(7)..(i + 8).min(v.len());
            let n = r.len() as f64;
            (v[i] - v[r].iter().sum::<DVec3>() / n).length_squared()
        })
        .sum();
    (s / v.len() as f64).sqrt().to_degrees()
}

/// Horizon tilt (deg) of an equirect from its near-vertical edges within ±25° of the equator: world verticals are
/// meridians when level; a tilt ε about the horizontal axis at azimuth θa leans the edge at azimuth θ and elevation e
/// by ε·cos(θ − θa)·sec²e. Fitted as lean = A cos θ + B sin θ + C (C: a shear, e.g. rolling shutter in a pan), again
/// without the 3σ outliers. Crate stab's `tests/footage.rs` measure; it reads 1.23× a known added tilt on v606.
fn tilt(px: &[u8], w: usize, h: usize) -> Option<f64> {
    let l = |x: usize, y: usize| {
        let p = &px[4 * (y * w + x)..];
        0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
    };
    let (b, row) = (16, |deg: f64| (deg / 180.0 * (h - 1) as f64) as usize);
    let mut e = vec![];
    for by in (row(65.0)..row(115.0) - b).step_by(b) {
        for bx in (0..w).step_by(b) {
            let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
            for y in by..by + b {
                for x in bx..bx + b {
                    let (gx, gy) = (
                        l((x + 1) % w, y) - l((x + w - 1) % w, y),
                        l(x, y + 1) - l(x, y - 1),
                    );
                    (xx, xy, yy) = (xx + gx * gx, xy + gx * gy, yy + gy * gy);
                }
            }
            let (s, d, beta) = (
                xx + yy,
                ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt(),
                0.5 * (2.0 * xy).atan2(xx - yy),
            );
            // strong, one-directional, near-vertical edges
            if s < 100.0 * (b * b) as f64 || d < 0.8 * s || beta.abs() > 20f64.to_radians() {
                continue;
            }
            let (th, ph) = (
                TAU * (bx + b / 2) as f64 / w as f64,
                PI * (by + b / 2) as f64 / h as f64,
            );
            e.push((th, -beta.tan() * ph.sin().powi(2)));
        }
    }
    let solve = |e: &[(f64, f64)]| {
        let (mut m, mut r) = (glam::DMat3::ZERO, DVec3::ZERO);
        for &(th, a) in e {
            let v = DVec3::new(th.cos(), th.sin(), 1.0);
            m += glam::DMat3::from_cols(v * v.x, v * v.y, v * v.z);
            r += a * v;
        }
        (e.len() >= 20 && m.determinant().abs() > 1e-9).then(|| m.inverse() * r)
    };
    let f = solve(&e)?;
    let res = |&(th, a): &(f64, f64)| (a - f.x * th.cos() - f.y * th.sin() - f.z).abs();
    let mut r: Vec<f64> = e.iter().map(res).collect();
    r.sort_by(f64::total_cmp);
    let lim = 3.0 * 1.4826 * r[r.len() / 2];
    e.retain(|x| res(x) < lim);
    let f = solve(&e)?;
    Some(f.x.hypot(f.y).to_degrees())
}

/// v606 (31 s walking, up to 3.6 rad/s) through the player at 2048 × 1024, every frame, stabilisation off, free,
/// horizon lock and direction lock (σ 0.5 s): residual inter-frame rotation, jitter, heading change, horizon tilt
/// every 5th frame (CSV `v606_player_tilt.csv`).
#[test]
fn stab_v606() {
    let Some(d) = samples() else { return };
    let (r, f) = (Rig::new(), d.join("v606.insv"));
    let m = Trailer::open(&f).unwrap().meta;
    let s = r.src(std::slice::from_ref(&f), &m, 3840.0, 0);
    let modes = [
        ("off", None),
        ("free", Some(Lock::Free)),
        ("horizon", Some(Lock::Horizon)),
        ("direction", Some(Lock::Direction)),
    ];
    let st: Vec<Stab> = modes.iter().map(|x| r.stab(&f, &s, &m, x.1)).collect();
    eprintln!(
        "v606: poses on the worker in {:.1?} ms (free, horizon, direction; the first includes reading and integrating \
         the IMU)",
        st[1..].iter().map(|s| s.took * 1e3).collect::<Vec<_>>()
    );
    let (w, h, n) = (2048, 1024, s.v.times.len());
    let t = r.target(w, h);
    let mut prev: Vec<Option<vis::Pano>> = modes.iter().map(|_| None).collect();
    let (mut rot, mut tl, mut mm) = (vec![vec![]; 4], vec![vec![]; 4], vec![]);
    let (t0, mut last) = (Instant::now(), None);
    for k in 0..n {
        let p = r.next(&s, k);
        for (i, st) in st.iter().enumerate() {
            let px = r.shot(&s, &p, &st.pose(p.t, s.v.times[0], s.ff), &t, i == 0);
            if k % 5 == 0
                && let Some(e) = tilt(&px, w, h)
            {
                tl[i].push((k, e));
            }
            let b = vis::Pano::rgba(&px, w, h);
            if let Some(a) = &prev[i]
                && let Some((v, _)) = turn(a, &b, &mut mm)
            {
                rot[i].push(v);
            }
            prev[i] = Some(b);
        }
        last = Some(p);
        if k % 100 == 99 {
            eprintln!("  {} frames, {:.0} s", k + 1, t0.elapsed().as_secs_f64());
        }
    }
    s.v.release(&r.g, &last.unwrap());
    eprintln!(
        "v606: {n} frames × {} settings rendered, read back and tracked in {:.1} s (GPU shared: noisy)",
        modes.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut csv = String::from("mode,frame,tilt_deg\n");
    let mut res = vec![];
    for (((name, _), rot), tl) in modes.iter().zip(&rot).zip(&tl) {
        let a: Vec<f64> = rot.iter().map(|v| v.length().to_degrees()).collect();
        let ts: Vec<f64> = tl.iter().map(|x| x.1).collect();
        let ([am, a95, amax], [tm, t95, tmax], j) = (stats(&a), stats(&ts), jitter(rot));
        eprintln!(
            "v606 {name}: {} pairs, inter-frame rotation mean {am:.3}° p95 {a95:.3}° max {amax:.3}°, jitter rms \
             {j:.4}°, heading change {:.1}°; horizon tilt (measure, ÷1.23 for degrees) mean {tm:.2}° p95 {t95:.2}° \
             max {tmax:.2}° over {} frames",
            a.len(),
            rot.iter().map(|v| v.z).sum::<f64>().to_degrees(),
            ts.len()
        );
        for (k, e) in tl {
            csv += &format!("{name},{k},{e:.3}\n");
        }
        res.push((am, j, tm));
    }
    // the horizon-lock tilt over time, per 3 s
    let hz = &tl[2];
    eprintln!(
        "v606 horizon lock, tilt per 3 s (mean of every 5th frame): {}",
        hz.chunks(18)
            .map(|c| format!("{:.2}", c.iter().map(|x| x.1).sum::<f64>() / c.len() as f64))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let out = fixtures().join("v606_player_tilt.csv");
    std::fs::write(&out, csv).unwrap();
    eprintln!("tilt over time: {}", out.display());
    let [off, free, hor, dir] = res[..] else {
        unreachable!()
    };
    assert!(free.1 < off.1 / 5.0 && hor.1 < off.1 / 5.0, "jitter");
    assert!(dir.0 < off.0 / 5.0, "direction lock");
    assert!(hor.2 < 1.5 && hor.2 < off.2 / 5.0 && dir.2 < 1.5, "tilt");
}

/// Protobuf varint.
fn varint(mut x: u64) -> Vec<u8> {
    let mut b = vec![];
    while x >= 0x80 {
        b.push(x as u8 | 0x80);
        x >>= 7;
    }
    b.push(x as u8);
    b
}

/// A proxy of v606 the way the camera makes one: both lenses side by side at 832² (lens 0 left), H.264, the main
/// file's trailer appended; it starts at main frame `s0` and its trailer's first frame says so. Made under names of
/// this process and thread and renamed into place: another test run that finds it must never read it half written.
fn proxy(d: &Path, s0: usize, n: usize) -> PathBuf {
    let out = fixtures().join(format!("v606_proxy_{s0}_{n}.lrv"));
    if out.exists() {
        return out;
    }
    let f = d.join("v606.insv");
    let id = format!("{}-{:?}", std::process::id(), std::thread::current().id());
    let (tmp, part) = (
        out.with_extension(format!("{id}.mp4")),
        out.with_extension(format!("{id}.part")),
    );
    // DUAL_STREAM_TRACK_REVERSE: track 1 holds lens 0
    let cut = |t: usize, o: &str| {
        format!("[0:v:{t}]trim=start_frame={s0},setpts=PTS-STARTPTS,scale=832:832{o}")
    };
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&f)
        .args([
            "-filter_complex",
            &format!("{};{};[a][b]hstack", cut(1, "[a]"), cut(0, "[b]")),
        ])
        .args(["-frames:v", &n.to_string(), "-fps_mode", "passthrough"])
        // no B-frames, as in the camera's proxies (pts = dts): the player shows samples in decode order
        .args([
            "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf",
            "16", "-an",
        ])
        .arg(&tmp)
        .status()
        .expect("ffmpeg");
    assert!(st.success());
    let t = Trailer::open(&f).unwrap();
    let mut tr = vec![];
    let mut file = std::fs::File::open(&f).unwrap();
    file.seek(std::io::SeekFrom::Start(t.start)).unwrap();
    file.read_to_end(&mut tr).unwrap();
    // first_frame (#24, varint) := the exposure time of main frame s0
    let ex = t.frames(EXPOSURE).unwrap();
    let (a, b) = (
        [&[0xc0, 0x01][..], &varint(t.meta.first_frame)].concat(),
        [&[0xc0, 0x01][..], &varint(ex[s0].t)].concat(),
    );
    assert_eq!(a.len(), b.len());
    let md = t.recs.iter().find(|r| r.kind == 1).unwrap();
    let o = (md.off - t.start) as usize;
    let i = o + tr[o..o + md.size as usize]
        .windows(a.len())
        .position(|w| w == a)
        .unwrap();
    tr[i..i + a.len()].copy_from_slice(&b);
    let mut mp4 = std::fs::read(&tmp).unwrap();
    mp4.extend(tr);
    std::fs::write(&part, mp4).unwrap();
    std::fs::remove_file(&tmp).unwrap();
    std::fs::rename(&part, &out).unwrap();
    out
}

/// Main frames `k + s0` against proxy frames `k` (k < n), stabilised by `st` at 1024 × 512: the rotation (deg) from
/// the main picture to the proxy's posed by time, and posed by frame number (as if both videos started together);
/// the proxy's inter-frame rotation unstabilised (`off`) and stabilised (rotation vectors).
fn against(
    r: &Rig,
    sm: &Src,
    sl: &Src,
    st: &Stab,
    off: &Stab,
    s0: usize,
    n: usize,
) -> ([Vec<f64>; 2], [Vec<DVec3>; 2]) {
    let (w, h) = (1024, 512);
    let t = r.target(w, h);
    let (mut d, mut v, mut mm) = ([vec![], vec![]], [vec![], vec![]], vec![]);
    let mut prev: [Option<vis::Pano>; 2] = [None, None];
    let mut last = None;
    for k in 0..n {
        let (pm, pl) = (r.next(sm, k + s0), r.next(sl, k));
        let a = vis::Pano::rgba(
            &r.shot(sm, &pm, &st.pose(pm.t, sm.v.times[0], sm.ff), &t, true),
            w,
            h,
        );
        let shots = [
            st.pose(pl.t, sl.v.times[0], sl.ff),
            st.pose(pl.t, sl.v.times[0], sm.ff),
            off.pose(pl.t, sl.v.times[0], sl.ff),
        ]
        .map(|o| vis::Pano::rgba(&r.shot(sl, &pl, &o, &t, true), w, h));
        for i in 0..2 {
            if let Some((x, _)) = turn(&a, &shots[i], &mut mm) {
                d[i].push(x.length().to_degrees());
            }
        }
        // shots[2] unstabilised, shots[0] stabilised
        for (i, b) in [(0, &shots[2]), (1, &shots[0])] {
            if let Some(a) = &prev[i]
                && let Some((x, _)) = turn(a, b, &mut mm)
            {
                v[i].push(x);
            }
        }
        let [s, _, u] = shots;
        prev = [Some(u), Some(s)];
        last = Some((pm, pl));
    }
    let (pm, pl) = last.unwrap();
    sm.v.release(&r.g, &pm);
    sl.v.release(&r.g, &pl);
    (d, v)
}

/// The proxy plays with the main file's poses mapped by time. The real `.lrv` of v635 carries the same IMU and
/// exposure records and first frame as its `.insv`, so each of its frames takes the pose of the main frame at the same
/// time, and its stabilised picture lies on the main one's. A v606 proxy that starts 10 frames late (its trailer
/// says so) checks the mapping on a moving clip, and the stabilisation of proxy playback.
#[test]
fn stab_proxy() {
    let Some(d) = samples() else { return };
    let r = Rig::new();
    let (fm, fl) = (d.join("v635.insv"), d.join("v635.lrv"));
    let (tm, tl) = (Trailer::open(&fm).unwrap(), Trailer::open(&fl).unwrap());
    for (k, name) in [(GYRO, "IMU"), (EXPOSURE, "exposure")] {
        let (mut a, mut b) = (vec![], vec![]);
        assert!(tm.get(k, &mut a).unwrap() && tl.get(k, &mut b).unwrap());
        eprintln!(
            "v635: {name} record {} bytes in the .insv, {} in the .lrv, identical: {}",
            a.len(),
            b.len(),
            a == b
        );
        assert_eq!(a, b);
    }
    let (imu, ex) = (tl.imu().unwrap(), tl.frames(EXPOSURE).unwrap());
    eprintln!(
        "v635.lrv trailer: {} IMU samples over {:.2} s, {} exposure items from frame 0, first frame {} µs (.insv {} µs), \
         readout {:.2} ms",
        imu.len(),
        (imu[imu.len() - 1].t - imu[0].t) as f64 * 1e-6,
        ex.len(),
        tl.meta.first_frame,
        tm.meta.first_frame,
        tl.meta.rolling_shutter
    );
    assert_eq!(tl.meta.first_frame, tm.meta.first_frame);
    let m = tm.meta;
    let sm = r.src(std::slice::from_ref(&fm), &m, 3840.0, 0);
    let sl = r.src(&[fl], &m, sm.fr.w as f32, 0);
    let (st, off) = (
        r.stab(&fm, &sm, &m, Some(Lock::Horizon)),
        r.stab(&fm, &sm, &m, None),
    );
    let n = sl.v.times.len();
    let same = (0..n)
        .filter(|&k| {
            st.pose(sl.v.times[k], sl.v.times[0], sl.ff).q
                == st.pose(sm.v.times[k], sm.v.times[0], sm.ff).q
        })
        .count();
    eprintln!(
        "v635: {same} of {n} proxy frames take the pose of the main frame with the same number ({} main frames)",
        sm.v.times.len()
    );
    assert_eq!(same, n);
    let (dd, _) = against(&r, &sm, &sl, &st, &off, 0, n);
    let [a, ..] = stats(&dd[0]);
    eprintln!(
        "v635 horizon lock: main vs proxy picture {a:.3}° apart on average ({} of {n} frames fitted)",
        dd[0].len()
    );
    assert!(a < 0.1 && dd[0].len() > n * 9 / 10);
    drop((sm, sl));

    let (s0, n) = (10, 300);
    let fm = d.join("v606.insv");
    let fl = proxy(&d, s0, n);
    let m = Trailer::open(&fm).unwrap().meta;
    let mut sm = r.src(std::slice::from_ref(&fm), &m, 3840.0, 0);
    sm.v.seek(&r.g, s0, None);
    let sl = r.src(&[fl], &m, sm.fr.w as f32, 0);
    assert_eq!(sl.v.times.len(), n);
    let (st, off) = (
        r.stab(&fm, &sm, &m, Some(Lock::Horizon)),
        r.stab(&fm, &sm, &m, None),
    );
    eprintln!(
        "v606 proxy from main frame {s0}: first frame {} µs later",
        sl.ff - sm.ff
    );
    let t0 = Instant::now();
    let (dd, vv) = against(&r, &sm, &sl, &st, &off, s0, n);
    let ([a, a95, _], [b, ..]) = (stats(&dd[0]), stats(&dd[1]));
    let [u, s] = [&vv[0], &vv[1]].map(|v| {
        (
            stats(
                &v.iter()
                    .map(|x| x.length().to_degrees())
                    .collect::<Vec<_>>(),
            )[0],
            jitter(v),
        )
    });
    eprintln!(
        "v606 proxy, horizon lock ({n} frames, {:.1} s): main vs proxy picture posed by time {a:.3}° apart (p95 \
         {a95:.3}°, {} fitted), posed by frame number {b:.2}°; proxy inter-frame rotation {:.3}° -> {:.3}° mean, jitter \
         rms {:.4}° -> {:.4}° (off -> horizon lock)",
        t0.elapsed().as_secs_f64(),
        dd[0].len(),
        u.0,
        s.0,
        u.1,
        s.1
    );
    assert!(a < 0.15 && b > 5.0 * a && dd[0].len() > n * 9 / 10);
    assert!(s.1 < u.1 / 5.0);
}

/// The paused frame's RAFT stitch with stabilisation, through the player's RAFT worker: the stitch takes the
/// rolling-shutter term (body frame), the preview turns it. On v606's fastest frame (horizon lock) its picture lies
/// on the template-seam preview of the same pose, closer than the RAFT stitch without the term.
#[test]
fn stab_raft() {
    let Some(d) = samples() else { return };
    let (r, f) = (Rig::new(), d.join("v606.insv"));
    let m = Trailer::open(&f).unwrap().meta;
    let mut s = r.src(std::slice::from_ref(&f), &m, 3840.0, 0);
    let st = r.stab(&f, &s, &m, Some(Lock::Horizon));
    let o = |k: usize| st.pose(s.v.times[k], s.v.times[0], s.ff);
    // the largest rotation over the readout
    let k = (0..s.v.times.len())
        .max_by(|&a, &b| o(a).rs[0].length().total_cmp(&o(b).rs[0].length()))
        .unwrap();
    let o = o(k);
    s.v.seek(&r.g, k, None);
    let p = r.next(&s, k);
    let (w, h) = (2048, 1024);
    let t = r.target(w, h);
    let lens = vis::Pano::rgba(&r.shot(&s, &p, &o, &t, true), w, h);
    let (still, (tx, rx)) = (still(&r.g, &r.cx), mpsc::channel());
    let (mut snap, mut mm) = (None, vec![]);
    let t0 = Instant::now();
    let res = [o.rs, [glam::Vec3::ZERO; 2]].map(|rs| {
        let key = (s.id, k, rs[0] != glam::Vec3::ZERO);
        still
            .send(s.job(&r.g, &p, rs, key, &mut snap, &tx, &r.ctx))
            .unwrap();
        let (got, tex) = rx.recv_timeout(Duration::from_secs(300)).unwrap().unwrap();
        assert_eq!(got, key);
        let pano = r.cx.pano(&tex.create_view(&Default::default()));
        s.pv.pose(&o);
        let mut e = r.g.device.create_command_encoder(&Default::default());
        let v = t.create_view(&Default::default());
        s.pv.render(
            &mut e,
            &v,
            [w as u32, h as u32],
            &Look::Equirect.view(),
            &pano,
        );
        r.g.queue.submit([e.finish()]);
        let px = stitch::read_texture(&r.g.device, &r.g.queue, &t);
        turn(&lens, &vis::Pano::rgba(&px, w, h), &mut mm).unwrap()
    });
    s.v.release(&r.g, &p);
    eprintln!(
        "v606 frame {k} (readout rotation {:.2}°), RAFT stitch vs template preview, horizon lock: with the \
         rolling-shutter term {:.3}° apart, rms {:.3} px; without {:.3}°, rms {:.3} px ({:.1} s with RAFT's first run)",
        o.rs[0].length().to_degrees(),
        res[0].0.length().to_degrees(),
        res[0].1,
        res[1].0.length().to_degrees(),
        res[1].1,
        t0.elapsed().as_secs_f64()
    );
    assert!(res[0].0.length().to_degrees() < 0.1 && res[0].1 < res[1].1);
}
