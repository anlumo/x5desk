//! Footage tests: the IMU axes, gyro time offset and rolling shutter measured against the video itself, the
//! stabilised output checked visually, and the levelling against the camera's own stitched thumbnail (`level`). They
//! read `X5_SAMPLES` (skip when unset) and write fixtures (cached feature tracks, preview videos, horizon tilt over
//! time and the levelling per clip as CSV) to `X5_FIXTURES` (default: the system temp dir). ffmpeg serves as a test tool only: it
//! decodes the lens streams to NV12 and encodes the preview videos. Run with `--release`.
//!
//! Visual inter-frame rotation: both lenses (scaled to 1920²) stitched to a 2048 × 1024 equirect by crate stitch,
//! corners between 15° and 115° polar angle (the operator stays below) tracked with `stab::vis`, a rotation fitted
//! with RANSAC (3 px) + Horn.

use glam::{DQuat, DVec3, Quat};
use media::insv::{self, Exposure, Imu, Meta, Trailer};
use stab::{AXES, KI, KP, Lock, OFFSET, Opts, Track, exp, horn, log, vis};
use std::{
    f64::consts::{PI, TAU},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Mutex, MutexGuard, OnceLock},
};
use stitch::{Calib, Ctx, Frame, Pose, Src, Stitcher};

/// Lens texture side the streams are scaled to.
const S: u32 = 1920;
/// Tracking panorama.
const W: usize = 2048;
const H: usize = 1024;
const BAND: [f64; 2] = [15.0 * PI / 180.0, 115.0 * PI / 180.0];
/// RANSAC inlier chord: 3 px.
const THR: f64 = 3.0 * TAU / W as f64;

struct Gpu {
    dev: wgpu::Device,
    q: wgpu::Queue,
}

/// One device per test binary (NV12 sampling needs `TEXTURE_FORMAT_NV12`, macOS's R8 / RG8 planes nothing); the lock
/// serialises GPU tests.
fn gpu() -> MutexGuard<'static, Gpu> {
    static G: OnceLock<Mutex<Gpu>> = OnceLock::new();
    G.get_or_init(|| {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let ad = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("adapter");
        let (dev, q) = pollster::block_on(ad.request_device(&wgpu::DeviceDescriptor {
            required_features: match cfg!(target_os = "macos") {
                true => wgpu::Features::empty(),
                false => wgpu::Features::TEXTURE_FORMAT_NV12,
            },
            required_limits: ad.limits(),
            ..Default::default()
        }))
        .expect("device");
        eprintln!("GPU: {}", ad.get_info().name);
        Mutex::new(Gpu { dev, q })
    })
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

fn samples() -> Option<PathBuf> {
    let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        eprintln!("X5_SAMPLES unset: skipped");
    }
    d
}

fn fixtures() -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(|| std::env::temp_dir().join("x5-stab"), PathBuf::from);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Clip {
    name: &'static str,
    path: PathBuf,
    m: Meta,
    imu: Vec<Imu>,
    /// Exposure items from video frame 0 on.
    ex: Vec<Exposure>,
    /// Video frames.
    n: usize,
    cal: Calib,
    fr: Frame,
}

fn clip(d: &Path, name: &'static str) -> Clip {
    clip_at(d.join(format!("{name}.insv")), name)
}

fn clip_at(path: PathBuf, name: &'static str) -> Clip {
    let t = Trailer::open(&path).unwrap();
    let cal = Calib::parse(&t.meta.offset).unwrap();
    Clip {
        name,
        n: media::mp4::Mp4::open(&path).unwrap().tracks[0]
            .samples
            .len(),
        imu: t.imu().unwrap(),
        ex: t.frames(insv::EXPOSURE).unwrap(),
        fr: Frame {
            w: S,
            h: S,
            crop: t.meta.crop,
            shift: t.meta.shift,
        },
        m: t.meta,
        cal,
        path,
    }
}

impl Clip {
    /// Exposure-record time of frame k, s.
    fn t(&self, k: usize) -> f64 {
        self.ex[k].t as f64 * 1e-6
    }

    /// Lens and texture position (0..1, row 0 at the top) that sees body ray `b`; None within 8° of the seam.
    fn row(&self, b: DVec3) -> Option<(usize, [f64; 2])> {
        let cam = |i: usize| {
            let r = &self.cal.lens[i].rot;
            DVec3::new(
                r[0][0] * b.x + r[0][1] * b.y + r[0][2] * b.z,
                r[1][0] * b.x + r[1][1] * b.y + r[1][2] * b.z,
                r[2][0] * b.x + r[2][1] * b.y + r[2][2] * b.z,
            )
        };
        let (c0, c1) = (cam(0), cam(1));
        let (i, c) = if c0.z > c1.z { (0, c0) } else { (1, c1) };
        if c.z < 8f64.to_radians().sin() {
            return None;
        }
        let crop = if self.m.crop > 0.0 {
            self.m.crop as f64
        } else {
            self.cal.side
        };
        let p = self.cal.project(i, c.to_array());
        Some((
            i,
            [0, 1].map(|k| (p[k] - (self.cal.side - crop) / 2.0 - self.m.shift[k] as f64) / crop),
        ))
    }
}

/// ffmpeg's hardware decoder on macOS (VideoToolbox); the software one on Linux: Vulkan Video on the GPU that also
/// stitches returned runs of corrupt frames in crate track's footage tests (its `HWACCEL`), which decoded the same way.
const HWACCEL: &str = if cfg!(target_os = "macos") {
    "videotoolbox"
} else {
    "none"
};

/// Both lens streams as NV12 at S × S, frame by frame (two ffmpeg processes).
struct Dec {
    kids: Vec<Child>,
    buf: [Vec<u8>; 2],
}

impl Dec {
    fn new(c: &Clip, frames: usize) -> Dec {
        // DUAL_STREAM_TRACK_REVERSE (4): track 0 holds lens 1
        let tracks = if c.m.stream == 4 { [1, 0] } else { [0, 1] };
        let kids = tracks
            .map(|t| {
                Command::new("ffmpeg")
                    .args(["-v", "error", "-hwaccel", HWACCEL, "-i"])
                    .arg(&c.path)
                    .args([
                        "-map",
                        &format!("0:v:{t}"),
                        "-frames:v",
                        &frames.to_string(),
                    ])
                    .args([
                        "-vf",
                        &format!("scale={S}:{S}"),
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "nv12",
                        "-",
                    ])
                    .stdout(Stdio::piped())
                    .spawn()
                    .expect("ffmpeg")
            })
            .into();
        let n = (S * S * 3 / 2) as usize;
        Dec {
            kids,
            buf: [vec![0; n], vec![0; n]],
        }
    }

    /// The next frame of both lenses into `buf`; false at the end.
    fn next(&mut self) -> bool {
        (0..2).all(|k| {
            self.kids[k]
                .stdout
                .as_mut()
                .unwrap()
                .read_exact(&mut self.buf[k])
                .is_ok()
        })
    }
}

impl Drop for Dec {
    fn drop(&mut self) {
        for k in &mut self.kids {
            let _ = k.kill();
            let _ = k.wait();
        }
    }
}

/// Two lens pictures and a stitcher. A lens is one NV12 texture, written and sampled through its plane aspects as
/// crate video decodes on Linux, or an R8 and an RG8 texture as on macOS (Metal has no NV12 textures).
struct Rig {
    cx: Ctx,
    tex: [[(wgpu::Texture, wgpu::TextureAspect); 2]; 2],
    src: [Src; 2],
    st: Stitcher,
}

impl Rig {
    fn new(g: &Gpu, c: &Clip, w: u32, h: u32) -> Rig {
        let cx = Ctx::new(&g.dev, &g.q);
        use wgpu::{TextureAspect as A, TextureFormat as F};
        let t = |s: u32, format| {
            g.dev.create_texture(&wgpu::TextureDescriptor {
                label: Some("lens"),
                size: wgpu::Extent3d {
                    width: S / s,
                    height: S / s,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let tex = [0, 1].map(|_| match cfg!(target_os = "macos") {
            true => [(t(1, F::R8Unorm), A::All), (t(2, F::Rg8Unorm), A::All)],
            false => {
                let nv = t(1, F::NV12);
                [(nv.clone(), A::Plane0), (nv, A::Plane1)]
            }
        });
        let planes = |l: &[(wgpu::Texture, A); 2]| {
            l.each_ref().map(|(t, aspect)| {
                t.create_view(&wgpu::TextureViewDescriptor {
                    aspect: *aspect,
                    ..Default::default()
                })
            })
        };
        let o = stitch::Opts {
            blend: c.m.blend.unwrap_or(6.0),
            colour: None,
            ..Default::default()
        };
        Rig {
            src: [0, 1].map(|l| cx.rect(&planes(&tex[l]), [0, 0, S, S])),
            st: Stitcher::new(&cx, &c.cal, c.fr, w, h, o).unwrap(),
            tex,
            cx,
        }
    }

    fn upload(&self, g: &Gpu, nv: &[Vec<u8>; 2]) {
        let n = (S * S) as usize;
        for (l, nv) in self.tex.iter().zip(nv) {
            for ((t, aspect), data, w, bpp) in
                [(&l[0], &nv[..n], S, 1), (&l[1], &nv[n..], S / 2, 2)]
            {
                g.q.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: t,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: *aspect,
                    },
                    data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(w * bpp),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: w,
                        height: w,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }

    /// The panorama for pose `p`, read back as RGBA8.
    fn render(&self, g: &Gpu, p: &Pose) -> Vec<u8> {
        self.st.pose(p);
        let mut enc = g.dev.create_command_encoder(&Default::default());
        self.st.encode(&mut enc, [&self.src[0], &self.src[1]]);
        g.q.submit([enc.finish()]);
        stitch::read_texture(&g.dev, &g.q, &self.st.out)
    }
}

type Tracks = Vec<Vec<[[f32; 2]; 2]>>;

/// Decodes frames 0..n and calls `f(k, rig)` with each uploaded.
fn each(g: &Gpu, c: &Clip, n: usize, w: u32, h: u32, mut f: impl FnMut(usize, &Rig)) {
    let (rig, mut dec) = (Rig::new(g, c, w, h), Dec::new(c, n));
    let mut k = 0;
    while k < n && dec.next() {
        rig.upload(g, &dec.buf);
        f(k, &rig);
        k += 1;
    }
}

/// Feature tracks between consecutive tracking panoramas, one list per pose sequence; `f(k, i, rgba)` sees each
/// panorama.
fn tracks(
    g: &Gpu,
    c: &Clip,
    n: usize,
    poses: &[&[Pose]],
    mut f: impl FnMut(usize, usize, &[u8]),
) -> Vec<Tracks> {
    let t0 = std::time::Instant::now();
    let mut out: Vec<Tracks> = poses.iter().map(|_| vec![]).collect();
    let mut prev: Vec<Option<vis::Pano>> = poses.iter().map(|_| None).collect();
    let mut m = vec![];
    each(g, c, n, W as u32, H as u32, |k, rig| {
        for (i, p) in poses.iter().enumerate() {
            let px = rig.render(g, &p[k]);
            f(k, i, &px);
            let pano = vis::Pano::rgba(&px, W, H);
            if let Some(a) = &prev[i] {
                vis::track(a, &pano, BAND, &mut m);
                out[i].push(m.clone());
            }
            prev[i] = Some(pano);
        }
    });
    eprintln!(
        "{}: {} frames × {} poses tracked in {:.1} s",
        c.name,
        out[0].len() + 1,
        poses.len(),
        t0.elapsed().as_secs_f64()
    );
    out
}

/// Tracks at the identity pose over the whole clip (cached).
fn id_tracks(g: &Gpu, c: &Clip) -> Tracks {
    let f = fixtures().join(format!("{}_{W}.tracks", c.name));
    if let Ok(b) = std::fs::read(&f) {
        let (mut out, mut b) = (vec![], &b[..]);
        while let Some((n, r)) = b.split_first_chunk::<4>() {
            let (m, r) = r.split_at(16 * u32::from_le_bytes(*n) as usize);
            out.push(
                m.as_chunks::<16>()
                    .0
                    .iter()
                    .map(|c| {
                        let v =
                            |i: usize| f32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap());
                        [[v(0), v(1)], [v(2), v(3)]]
                    })
                    .collect(),
            );
            b = r;
        }
        return out;
    }
    let out = tracks(g, c, c.n, &[&vec![Pose::default(); c.n]], |_, _, _| {})
        .pop()
        .unwrap();
    let mut b = vec![];
    for m in &out {
        b.extend((m.len() as u32).to_le_bytes());
        for v in m.iter().flatten().flatten() {
            b.extend(v.to_le_bytes());
        }
    }
    std::fs::write(&f, b).unwrap();
    out
}

/// One frame pair: inlier rays of frames k and k + 1 (body = output world at identity pose) and the visual rotation.
struct Pair {
    k: usize,
    a: Vec<DVec3>,
    b: Vec<DVec3>,
    /// Body rotation vector from frame k to k + 1 (b ≈ exp(−th)·a).
    th: DVec3,
    rms: f64,
}

fn pairs(t: &Tracks) -> Vec<Pair> {
    t.iter()
        .enumerate()
        .filter_map(|(k, m)| {
            let r: Vec<_> = m
                .iter()
                .map(|p| {
                    (
                        vis::ray(W, H, p[0].map(f64::from)),
                        vis::ray(W, H, p[1].map(f64::from)),
                    )
                })
                .collect();
            let f = vis::fit(&r, THR)?;
            (f.n >= 40).then(|| Pair {
                k,
                a: r.iter()
                    .zip(&f.inl)
                    .filter(|x| *x.1)
                    .map(|x| x.0.0)
                    .collect(),
                b: r.iter()
                    .zip(&f.inl)
                    .filter(|x| *x.1)
                    .map(|x| x.0.1)
                    .collect(),
                th: -log(f.q),
                rms: f.rms,
            })
        })
        .collect()
}

/// Gyro-only integration in IMU axes (midpoint rule).
fn gyro(imu: &[Imu]) -> Track {
    let v = |a: [f32; 3]| DVec3::new(a[0] as f64, a[1] as f64, a[2] as f64);
    let mut q = vec![DQuat::IDENTITY];
    for p in imu.windows(2) {
        let d = exp(0.5 * (v(p[0].gyro) + v(p[1].gyro)) * (p[1].t - p[0].t) as f64 * 1e-6);
        q.push((*q.last().unwrap() * d).normalize());
    }
    Track {
        t: imu.iter().map(|s| s.t as f64 * 1e-6).collect(),
        q,
    }
}

/// IMU -> body rotation fitted to the visual rotations `th` (frame k -> k + 1) at frame times `t(k) + d`:
/// (rotation, RMS residual rad/frame).
fn axes(th: &[(usize, DVec3)], t: &dyn Fn(usize) -> f64, g: &Track, d: f64) -> (DQuat, f64) {
    let v: Vec<_> = th
        .iter()
        .map(|&(k, th)| (g.turn(t(k) + d, t(k + 1) + d), th))
        .collect();
    let m = horn(v.iter().map(|&(a, b)| (a, b, 1.0)));
    let e: f64 = v.iter().map(|&(a, b)| (b - m * a).length_squared()).sum();
    (m, (e / v.len() as f64).sqrt())
}

/// Offset with the smallest residual: 1 ms grid over ±100 ms, then 0.02 ms around the best.
fn offset(th: &[(usize, DVec3)], t: &dyn Fn(usize) -> f64, g: &Track) -> (f64, DQuat, f64) {
    let best = |r: &mut dyn Iterator<Item = f64>| {
        r.map(|d| {
            let (m, e) = axes(th, t, g, d);
            (d, m, e)
        })
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap()
    };
    let d0 = best(&mut (-100..=100).map(|i| i as f64 * 1e-3)).0;
    best(&mut (-50..=50).map(|i| d0 + i as f64 * 2e-5))
}

/// The signed permutation (det +1) closest to `m`, and the angle between them (deg).
fn perm(m: DQuat) -> ([[i8; 3]; 3], f64) {
    let mut best = ([[0i8; 3]; 3], f64::MAX);
    for p in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for s in 0..8 {
            let mut a = [[0i8; 3]; 3];
            for i in 0..3 {
                a[i][p[i]] = if s >> i & 1 == 1 { -1 } else { 1 };
            }
            let mat = glam::DMat3::from_cols_array_2d(&a.map(|r| r.map(f64::from))).transpose();
            if mat.determinant() < 0.0 {
                continue;
            }
            let e = log(m * DQuat::from_mat3(&mat).inverse())
                .length()
                .to_degrees();
            if e < best.1 {
                best = (a, e);
            }
        }
    }
    best
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        (sab, saa, sbb) = (
            sab + (x - ma) * (y - mb),
            saa + (x - ma).powi(2),
            sbb + (y - mb).powi(2),
        );
    }
    sab / (saa * sbb).sqrt()
}

/// Lens and texture position of each inlier ray of a pair (frame k, frame k + 1).
type Rows = Vec<(Vec<(usize, [f64; 2])>, Vec<(usize, [f64; 2])>)>;

/// Mean RMS (px) of rigid fits over the pairs after moving each ray to its frame's centre-row time for readout
/// `tr` (s), with body rates `w` per frame; `f(lens, uv)` is the ray's readout position 0..1.
fn rigid(
    ps: &[Pair],
    rows: &Rows,
    w: &dyn Fn(usize) -> DVec3,
    f: &dyn Fn(usize, [f64; 2]) -> f64,
    tr: f64,
) -> (f64, Vec<(usize, DVec3)>) {
    let mut th = vec![];
    let e: f64 = ps
        .iter()
        .zip(rows)
        .map(|(p, (ra, rb))| {
            let (wa, wb) = (w(p.k), w(p.k + 1));
            let a: Vec<_> =
                p.a.iter()
                    .zip(ra)
                    .map(|(&a, &(i, v))| exp(wa * (f(i, v) - 0.5) * tr) * a)
                    .collect();
            let b: Vec<_> =
                p.b.iter()
                    .zip(rb)
                    .map(|(&b, &(i, v))| exp(wb * (f(i, v) - 0.5) * tr) * b)
                    .collect();
            let q = horn(a.iter().zip(&b).map(|(&a, &b)| (a, b, 1.0)));
            th.push((p.k, -log(q)));
            (a.iter()
                .zip(&b)
                .map(|(&a, &b)| (q * a - b).length_squared())
                .sum::<f64>()
                / a.len() as f64)
                .sqrt()
        })
        .sum();
    (e / ps.len() as f64 * W as f64 / TAU, th)
}

/// IMU axes, time offset and rolling shutter from the footage; prints the measurements and checks the crate's
/// constants against them. Then the stitch's rolling-shutter term from `stab::poses`, through the shader: the
/// rigidity of the inter-frame motion at several multiples of the trailer's readout.
#[test]
fn calib() {
    let Some(d) = samples() else { return };
    let g = gpu();
    let want = perm(DQuat::from_mat3(&AXES)).0;
    for name in ["v635", "v627", "v606"] {
        let c = clip(&d, name);
        let t = id_tracks(&g, &c);
        // inliers away from the seam, with their lens rows
        let mut ps = pairs(&t);
        let rows: Rows = ps
            .iter_mut()
            .map(|p| {
                let keep: Vec<_> =
                    p.a.iter()
                        .zip(&p.b)
                        .map(|(&a, &b)| c.row(a).zip(c.row(b)))
                        .collect();
                let f = |v: &[DVec3]| {
                    v.iter()
                        .zip(&keep)
                        .filter_map(|(x, k)| k.map(|_| *x))
                        .collect::<Vec<_>>()
                };
                (p.a, p.b) = (f(&p.a), f(&p.b));
                keep.iter().flatten().copied().unzip()
            })
            .collect();
        let gy = gyro(&c.imu);
        let th: Vec<_> = ps.iter().map(|p| (p.k, p.th)).collect();
        let (dt, m, e) = offset(&th, &|k| c.t(k), &gy);
        let (p, pe) = perm(m);
        eprintln!(
            "{name}: {} pairs of {}, visual rms {:.3} px, mean rotation {:.3}°/frame, exposure {:.2}..{:.2} ms, trailer readout {:.2} ms",
            ps.len(),
            t.len(),
            ps.iter().map(|p| p.rms).sum::<f64>() / ps.len() as f64 * W as f64 / TAU,
            (ps.iter().map(|p| p.th.length()).sum::<f64>() / ps.len() as f64).to_degrees(),
            c.ex.iter().map(|e| e.time).fold(f64::MAX, f64::min) * 1e3,
            c.ex.iter().map(|e| e.time).fold(0.0, f64::max) * 1e3,
            c.m.rolling_shutter,
        );
        eprintln!(
            "  record time, no RS: offset {:.2} ms, residual {:.4}°/frame; IMU -> body {p:?} (fit {pe:.2}° from it)",
            dt * 1e3,
            e.to_degrees()
        );
        let v: Vec<_> = th
            .iter()
            .map(|&(k, th)| (m * gy.turn(c.t(k) + dt, c.t(k + 1) + dt), th))
            .collect();
        let cr = [0, 1, 2].map(|i| {
            let (a, b): (Vec<f64>, Vec<f64>) = v.iter().map(|(a, b)| (a[i], b[i])).unzip();
            corr(&a, &b)
        });
        eprintln!("  visual vs gyro per body axis: correlation {cr:.4?}");

        // rolling shutter: rigidity of the inter-frame motion against the readout time
        let w = |k: usize| m * gy.turn(c.t(k) + dt - 0.01, c.t(k) + dt + 0.01) / 0.02;
        let v = |_: usize, p: [f64; 2]| p[1];
        let sweep: Vec<(f64, f64)> = (-16..=16)
            .map(|i| {
                (
                    i as f64 * 2.5e-3,
                    rigid(&ps, &rows, &w, &v, i as f64 * 2.5e-3).0,
                )
            })
            .collect();
        eprintln!(
            "  readout ms -> rigid-fit rms px: {}",
            sweep
                .iter()
                .step_by(2)
                .map(|(t, e)| format!("{:.0}:{e:.4}", t * 1e3))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let i = (1..sweep.len() - 1)
            .min_by(|&a, &b| sweep[a].1.total_cmp(&sweep[b].1))
            .unwrap();
        let (a, b, cc) = (sweep[i - 1].1, sweep[i].1, sweep[i + 1].1);
        let tr = sweep[i].0 + 2.5e-3 * 0.5 * (a - cc) / (a - 2.0 * b + cc);
        eprintln!(
            "  best readout {:.2} ms (trailer {:.2} ms)",
            tr * 1e3,
            c.m.rolling_shutter
        );

        // readout along texture rows or columns, per lens forward (row / column 0 first) or reversed, at the
        // trailer's readout
        let ro = c.m.rolling_shutter * 1e-3;
        let dirs: Vec<(String, f64)> = [1, 0]
            .into_iter()
            .flat_map(|ax| [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]].map(|s| (ax, s)))
            .map(|(ax, s): (usize, [f64; 2])| {
                let f = |i: usize, p: [f64; 2]| 0.5 + s[i] * (p[ax] - 0.5);
                let sg = |x: f64| if x > 0.0 { '+' } else { '-' };
                (
                    format!("{}{}{}", ["cols", "rows"][ax], sg(s[0]), sg(s[1])),
                    rigid(&ps, &rows, &w, &f, ro).0,
                )
            })
            .collect();
        eprintln!(
            "  readout {:.2} ms by direction (lens 0, lens 1; + = row / column 0 first): none {:.4} px, {}",
            c.m.rolling_shutter,
            sweep[16].1,
            dirs.iter()
                .map(|(n, e)| format!("{n} {e:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // the offset again, for the centre rows at the trailer's readout, with the frame time inside the exposure
        let (_, th) = rigid(&ps, &rows, &w, &v, ro);
        let mut fit = None;
        for beta in [0.0, -0.25, -0.5, -0.75, -1.0] {
            let tb = |k: usize| c.t(k) + beta * c.ex[k].time;
            let (dt, m, e) = offset(&th, &tb, &gy);
            let (p, _) = perm(m);
            let pm = glam::DMat3::from_cols_array_2d(&p.map(|r| r.map(f64::from))).transpose();
            let mis = log(m * DQuat::from_mat3(&pm).inverse()) * 180.0 / PI;
            let v: Vec<_> = th
                .iter()
                .map(|&(k, th)| (m * gy.turn(tb(k) + dt, tb(k + 1) + dt), th))
                .collect();
            let scale = v.iter().map(|(a, b)| a.dot(*b)).sum::<f64>()
                / v.iter().map(|(a, _)| a.length_squared()).sum::<f64>();
            eprintln!(
                "  frame time = record {beta:+.2}·exposure {:+.2} ms: residual {:.4}°/frame, IMU -> body = permutation · \
                 rotation {:.2?}°, gyro scale {scale:.4}",
                dt * 1e3,
                e.to_degrees(),
                mis.to_array()
            );
            if beta == -0.5 {
                fit = Some((p, dt));
            }
        }
        let (p, dt) = fit.unwrap();
        // gyro bias left in the samples (IMU axes): the mean of gyro minus visual rotation per second. Gravity
        // corrects roll and pitch; nothing corrects yaw (no magnetometer), so this is the heading drift
        let ft = |k: usize| c.t(k) - 0.5 * c.ex[k].time + dt;
        let (s, sd) = th.iter().fold((DVec3::ZERO, 0.0), |(s, sd), &(k, th)| {
            (
                s + gy.turn(ft(k), ft(k + 1)) - AXES.transpose() * th,
                sd + ft(k + 1) - ft(k),
            )
        });
        eprintln!(
            "  residual gyro bias {:.2?}°/s (IMU axes), about body Z {:.3}°/s",
            (s / sd).to_array().map(f64::to_degrees),
            (AXES * s / sd).z.to_degrees()
        );
        // v635 barely moves (0.11°/frame): its offset and readout are poorly defined
        assert_eq!(p, want, "{name}: axes");
        if name != "v635" {
            let best = dirs.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            assert_eq!(best.0, "rows++", "{name}: readout direction");
            assert!((dt - OFFSET).abs() < 0.5e-3, "{name}: offset {dt}");
            assert!(
                (tr / (c.m.rolling_shutter * 1e-3) - 1.0).abs() < 0.1,
                "{name}: readout {tr}"
            );
        }
    }

    let c = clip(&d, "v606");
    let tr = Track::new(&c.imu, KP, KI);
    let p = stab::poses(&tr, &c.ex, &Opts::new(&c.m, Lock::Free, 0.0));
    let n = 240;
    let k = [0.0, 0.5, 1.0, 1.5, -1.0];
    let v: Vec<Vec<Pose>> = k
        .iter()
        .map(|&s| {
            p[..n]
                .iter()
                .map(|p| Pose {
                    q: Quat::IDENTITY,
                    rs: [p.rs * s; 2],
                })
                .collect()
        })
        .collect();
    let t = tracks(
        &g,
        &c,
        n,
        &v.iter().map(|v| &v[..]).collect::<Vec<_>>(),
        |_, _, _| {},
    );
    let e: Vec<f64> = t
        .iter()
        .map(|t| {
            let ps = pairs(t);
            ps.iter().map(|p| p.rms).sum::<f64>() / ps.len() as f64 * W as f64 / TAU
        })
        .collect();
    eprintln!(
        "v606 frames 0..{n}, stitch rolling-shutter term × readout {k:?}: rigid-fit rms {e:.4?} px"
    );
    assert!(e[2] < e[0] && e[0] < e[4]);
}

/// Horizon tilt of an equirect from its near-vertical edges within ±25° of the equator: world verticals are
/// meridians in a level panorama, and a small tilt ε about the horizontal axis at azimuth θa leans the edge at
/// azimuth θ and elevation e by ε·cos(θ − θa)·sec²e in equirect pixels. Returns (tilt vector (ε cos θa, ε sin θa)°, uniform lean° — a shear, e.g.
/// rolling shutter during a pan —, edge blocks used).
fn tilt(px: &[u8], w: usize, h: usize) -> Option<([f64; 2], f64, usize)> {
    let l = |x: usize, y: usize| {
        let p = &px[4 * (y * w + x)..];
        0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
    };
    let (b, row) = (16, |deg: f64| (deg / 180.0 * (h - 1) as f64) as usize);
    let mut e = vec![]; // (θ, lean)
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
                PI * (by + b / 2) as f64 / (h - 1) as f64,
            );
            e.push((th, -beta.tan() * ph.sin().powi(2)));
        }
    }
    // lean = A cos θ + B sin θ + C, least squares, then again without the 3σ (MAD) outliers
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
    Some((
        [f.x.to_degrees(), f.y.to_degrees()],
        f.z.to_degrees(),
        e.len(),
    ))
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (
        s.iter().sum::<f64>() / s.len() as f64,
        s[s.len() * 95 / 100],
        s[s.len() - 1],
    )
}

/// The stabilised output measured on the video. First the preview videos (v606, first 10 s, horizon lock) and the
/// gain of the image tilt measure on those frames (its response to added known tilts); then per mode on v606 and
/// v627 the residual inter-frame rotation (visual, as in `calib`), the horizon tilt and the rolling-shutter shear.
#[test]
fn verify() {
    let Some(d) = samples() else { return };
    let g = gpu();
    let c = clip(&d, "v606");
    let (n, ew, eh, pw, ph) = (300, 1920, 960, 960, 540);
    let tr = Track::new(&c.imu, KP, KI);
    let hp = stab::poses(&tr, &c.ex, &Opts::new(&c.m, Lock::Horizon, 0.5));
    // X5 framing: the panorama centre on lens 1 (track 0)
    let fr = Pose::euler(0.0, 0.0, if c.m.stream == 4 { 180.0 } else { 0.0 });
    let enc = |file: &str, w: usize, h: usize| {
        Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba"])
            .args(["-s", &format!("{w}x{h}"), "-r", "30", "-i", "-"])
            .args([
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            ])
            .arg(fixtures().join(file))
            .stdin(Stdio::piped())
            .spawn()
            .expect("ffmpeg")
    };
    let (mut ve, mut vp) = (
        enc("v606_equirect_raw_vs_horizon.mp4", ew, 2 * eh),
        enc("v606_perspective_raw_vs_horizon.mp4", 2 * pw, ph),
    );
    let tex = g.dev.create_texture(&wgpu::TextureDescriptor {
        label: Some("view"),
        size: wgpu::Extent3d {
            width: pw as u32,
            height: ph as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let tv = tex.create_view(&Default::default());
    let (mut sbs, mut pv, mut known) = (vec![0u8; 2 * pw * ph * 4], None, vec![]);
    each(&g, &c, n, ew as u32, eh as u32, |k, rig| {
        let pv = pv.get_or_insert_with(|| {
            stitch::Preview::new(
                &rig.cx,
                &c.cal,
                c.fr,
                c.m.blend.unwrap_or(6.0),
                None,
                wgpu::TextureFormat::Rgba8Unorm,
            )
        });
        let st = Pose {
            q: hp[k].q * fr.q,
            rs: [hp[k].rs; 2],
        };
        let io = |v: &mut Child, b: &[u8]| v.stdin.as_mut().unwrap().write_all(b).unwrap();
        for p in [fr, st] {
            io(&mut ve, &rig.render(&g, &p));
        }
        for (i, p) in [fr, st].iter().enumerate() {
            pv.pose(p);
            let mut e = g.dev.create_command_encoder(&Default::default());
            let v = stitch::View {
                fov: 100.0,
                ..Default::default()
            };
            pv.render(
                &mut e,
                &tv,
                [pw as u32, ph as u32],
                &v,
                [&rig.src[0], &rig.src[1]],
            );
            g.q.submit([e.finish()]);
            let px = stitch::read_texture(&g.dev, &g.q, &tex);
            for y in 0..ph {
                sbs[(y * 2 * pw + i * pw) * 4..][..pw * 4]
                    .copy_from_slice(&px[y * pw * 4..][..pw * 4]);
            }
        }
        io(&mut vp, &sbs);
        // known tilts about the level frame's two horizontal axes: the measured tilt vector should move by them
        if k % 30 == 0 {
            let r = [
                DVec3::ZERO,
                DVec3::X * 2.0,
                DVec3::Y * 2.0,
                DVec3::X * 5.0,
                DVec3::Y * 5.0,
            ];
            let v = r.map(|e| {
                let q = hp[k].q * Quat::from_scaled_axis((e * PI / 180.0).as_vec3()) * fr.q;
                tilt(
                    &rig.render(
                        &g,
                        &Pose {
                            q,
                            rs: [hp[k].rs; 2],
                        },
                    ),
                    ew,
                    eh,
                )
                .map(|t| t.0)
            });
            if let [Some(a), b @ ..] = v
                && b.iter().all(Option::is_some)
            {
                for (b, e) in b.iter().flatten().zip(&r[1..]) {
                    known.push((b[0] - a[0]).hypot(b[1] - a[1]) / e.length());
                }
            }
        }
    });
    for v in [ve, vp] {
        assert!(v.wait_with_output().unwrap().status.success());
    }
    let (gain, _, gmax) = stats(&known);
    let gmin = known.iter().copied().fold(f64::MAX, f64::min);
    eprintln!(
        "tilt measure on v606 frames: response to 2° and 5° tilts {gain:.3}× (min {gmin:.3}, max {gmax:.3}, {} cases)",
        known.len()
    );
    assert!((0.8..1.5).contains(&gain), "tilt measure");
    eprintln!("previews: {}", fixtures().display());

    for name in ["v606", "v627"] {
        let c = clip(&d, name);
        let tr = Track::new(&c.imu, KP, KI);
        let mode = |lock, smooth| -> Vec<Pose> {
            stab::poses(&tr, &c.ex, &Opts::new(&c.m, lock, smooth))
                .iter()
                .map(|p| Pose {
                    q: p.q,
                    rs: [p.rs; 2],
                })
                .collect()
        };
        let modes = [
            ("raw", vec![Pose::default(); c.n]),
            ("free σ 0.5 s", mode(Lock::Free, 0.5)),
            ("horizon σ 0.5 s", mode(Lock::Horizon, 0.5)),
            ("direction", mode(Lock::Direction, 0.5)),
        ];
        // body yaw rate per frame, for the shear
        let ft = |k: usize| c.t(k) - 0.5 * c.ex[k].time + OFFSET;
        let yaw: Vec<f64> = (0..c.n)
            .map(|k| tr.turn(ft(k) - 0.01, ft(k) + 0.01).z / 0.02)
            .collect();
        let mut tl: [Vec<(usize, f64, f64)>; 4] = Default::default();
        let t = tracks(
            &g,
            &c,
            c.n,
            &modes.each_ref().map(|m| &m.1[..]),
            |k, i, px| {
                if k % 5 == 0
                    && let Some((e, s, _)) = tilt(px, W, H)
                {
                    tl[i].push((k, e[0].hypot(e[1]), s));
                }
            },
        );
        let mut res = vec![];
        for (((m, _), t), tl) in modes.iter().zip(&t).zip(&tl) {
            let ps = pairs(t);
            let a: Vec<f64> = ps.iter().map(|p| p.th.length().to_degrees()).collect();
            // jitter: the inter-frame rotation minus its ±7-frame mean
            let j: Vec<f64> = (0..ps.len())
                .map(|i| {
                    let r = i.saturating_sub(7)..(i + 8).min(ps.len());
                    let n = r.len() as f64;
                    (ps[i].th - ps[r].iter().map(|p| p.th).sum::<DVec3>() / n)
                        .length()
                        .to_degrees()
                })
                .collect();
            let jr = (j.iter().map(|x| x * x).sum::<f64>() / j.len() as f64).sqrt();
            let (am, a95, amax) = stats(&a);
            let ts: Vec<f64> = tl.iter().map(|x| x.1).collect();
            let (tm, t95, tmax) = stats(&ts);
            // shear against yaw rate: rolling shutter leans verticals while panning
            let (sy, yy) = tl.iter().fold((0.0, 0.0), |(sy, yy), &(k, _, s)| {
                (sy + s * yaw[k], yy + yaw[k] * yaw[k])
            });
            // heading change over the clip: the visual turns about the output's up (world up when levelled)
            let hd = ps.iter().map(|p| p.th.z).sum::<f64>().to_degrees();
            eprintln!(
                "{name} {m}: {} pairs, inter-frame rotation mean {am:.3}° p95 {a95:.3}° max {amax:.3}°, jitter rms {jr:.4}°; \
                 tilt mean {tm:.2}° p95 {t95:.2}° max {tmax:.2}° (÷ gain: {:.2}° / {:.2}° / {:.2}°, {} frames), shear {:.3}° \
                 per rad/s of yaw; heading change {hd:.2}°",
                ps.len(),
                tm / gain,
                t95 / gain,
                tmax / gain,
                ts.len(),
                sy / yy
            );
            res.push((am, jr, tm, t95, sy / yy));
        }
        let mut csv = String::from("mode,frame,tilt_deg,lean_deg\n");
        for ((m, _), tl) in modes.iter().zip(&tl) {
            for &(k, e, s) in tl {
                csv += &format!("{m},{k},{e:.3},{s:.3}\n");
            }
        }
        std::fs::write(fixtures().join(format!("{name}_tilt.csv")), csv).unwrap();
        let [raw, free, hor, dir] = res[..] else {
            unreachable!()
        };
        assert!(
            free.1 < raw.1 / 5.0 && hor.1 < raw.1 / 5.0 && dir.1 < raw.1 / 5.0,
            "{name}: jitter"
        );
        assert!(dir.0 < raw.0 / 5.0, "{name}: direction lock");
        assert!(
            hor.2 < 1.5 && hor.3 < 3.0 && dir.2 < 1.5 && hor.2 < raw.2 / 5.0,
            "{name}: tilt"
        );
        // v627 turns too little (0.27°/frame) for the shear regression
        assert!(
            name != "v606" || hor.4.abs() < raw.4.abs() / 5.0,
            "{name}: shear"
        );
    }
}

/// The tilt measure on meridian stripes rendered with known tilts (whole band, and only beyond ±15° elevation).
#[test]
fn tilt_measure() {
    let (w, h) = (1920, 960);
    for (e, az, band) in [
        (0.0f64, 0.0f64, 0.0f64),
        (2.0, 30.0, 0.0),
        (5.0, 200.0, 0.0),
        (2.0, 120.0, 15.0),
        (5.0, 300.0, 15.0),
    ] {
        let r = DQuat::from_axis_angle(
            DVec3::new(az.to_radians().cos(), az.to_radians().sin(), 0.0),
            e.to_radians(),
        );
        let px: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let d = r * vis::ray(w, h, [(i % w) as f64, (i / w) as f64]);
                let s = (d.y.atan2(d.x) * 36.0 / PI).rem_euclid(2.0) < 1.0
                    && d.z.abs() >= band.to_radians().sin();
                let v = if s { 200 } else { 40 };
                [v, v, v, 255]
            })
            .collect();
        let (t, c, n) = tilt(&px, w, h).unwrap();
        let m = t[0].hypot(t[1]);
        eprintln!(
            "tilt {e}° at azimuth {az}°, stripes beyond ±{band}°: measured {m:.3}° (lean {c:.3}°, {n} blocks)"
        );
        assert!((m - e).abs() < 0.1 * e + 0.02 && c.abs() < 0.1);
    }
}

/// Mahony gains against the horizon tilt measured in the image: horizon-locked panoramas every 5th frame.
#[test]
fn gains() {
    let Some(d) = samples() else { return };
    let g = gpu();
    let set = [(KP, KI), (0.5, 0.05), (0.05, 0.001), (0.0, 0.0)];
    for name in ["v606", "v627"] {
        let c = clip(&d, name);
        let p: Vec<Vec<Pose>> = set
            .iter()
            .map(|&(kp, ki)| {
                stab::poses(
                    &Track::new(&c.imu, kp, ki),
                    &c.ex,
                    &Opts::new(&c.m, Lock::Horizon, 0.5),
                )
                .iter()
                .map(|p| Pose {
                    q: p.q,
                    rs: [p.rs; 2],
                })
                .collect()
            })
            .collect();
        let mut tl = vec![vec![]; set.len()];
        each(&g, &c, c.n, W as u32, H as u32, |k, rig| {
            if k % 5 == 0 {
                for (i, p) in p.iter().enumerate() {
                    if let Some((e, _, _)) = tilt(&rig.render(&g, &p[k]), W, H) {
                        tl[i].push(e[0].hypot(e[1]));
                    }
                }
            }
        });
        let r: Vec<_> = tl.iter().map(|t| stats(t)).collect();
        for (((kp, ki), (m, p95, max)), t) in set.iter().zip(&r).zip(&tl) {
            eprintln!(
                "{name} Kp {kp} Ki {ki}: horizon tilt mean {m:.2}° p95 {p95:.2}° max {max:.2}° ({} frames)",
                t.len()
            );
        }
        // the chosen gains beat gyro only
        assert!(r[0].0 < r[3].0, "{name}");
    }
}

/// The camera's thumbnail pixel (i, j) shows our equirect position (i + 0.7, j + 0.7) (`vis::ray`): the alignment
/// score peaks there in both axes on v606, v627 and v635 alike (0.84 → 0.90 on v606).
const TOFF: f64 = 0.7;

/// Equirect pyramid: 2 × 2 box means per level; high-passed (minus a radius-2 box blur applied twice) for alignment,
/// so the cameras' tone curves and the lens balance don't pull the fit.
struct Eq {
    w: usize,
    h: usize,
    lv: Vec<Vec<f32>>,
}

impl Eq {
    fn new(px: Vec<f32>, w: usize, h: usize, n: usize, hp: bool) -> Eq {
        let mut lv = vec![px];
        for l in 1..=n {
            let (pw, w2, p) = (w >> (l - 1), w >> l, &lv[l - 1]);
            let q = (0..w2 * (h >> l))
                .map(|i| {
                    let k = 2 * (i / w2) * pw + 2 * (i % w2);
                    0.25 * (p[k] + p[k + 1] + p[k + pw] + p[k + pw + 1])
                })
                .collect();
            lv.push(q);
        }
        if hp {
            for (l, p) in lv.iter_mut().enumerate() {
                let (w, h) = (w >> l, h >> l);
                let b = blur(&blur(p, w, h), w, h);
                p.iter_mut().zip(b).for_each(|(a, b)| *a -= b);
            }
        }
        Eq { w, h, lv }
    }

    /// Bilinear value of level `l` at direction `d` (level pixel k covers base pixels 2^l·k ..; wrapped in x).
    fn at(&self, l: usize, d: DVec3) -> f32 {
        let (s, w, h) = ((1 << l) as f64, self.w >> l, self.h >> l);
        let x = ((-d.y).atan2(d.x).rem_euclid(TAU) * self.w as f64 / TAU - (s - 1.0) / 2.0) / s;
        let y = (d.z.clamp(-1.0, 1.0).acos() * (self.h - 1) as f64 / PI - (s - 1.0) / 2.0) / s;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
        let xa = (x0 as isize).rem_euclid(w as isize) as usize;
        let xb = (xa + 1) % w;
        let ya = (y0 as isize).clamp(0, h as isize - 1) as usize * w;
        let yb = (y0 as isize + 1).clamp(0, h as isize - 1) as usize * w;
        let p = &self.lv[l];
        let t = p[ya + xa] + fx * (p[ya + xb] - p[ya + xa]);
        let b = p[yb + xa] + fx * (p[yb + xb] - p[yb + xa]);
        t + fy * (b - t)
    }
}

/// Box blur of radius 2, wrapped in x, clamped in y.
fn blur(p: &[f32], w: usize, h: usize) -> Vec<f32> {
    let a: Vec<f32> = (0..w * h)
        .map(|i| {
            (0..5)
                .map(|k| p[i - i % w + (i % w + w + k - 2) % w])
                .sum::<f32>()
                / 5.0
        })
        .collect();
    (0..w * h)
        .map(|i| {
            let y = i / w;
            (0..5)
                .map(|k| a[(y + k).saturating_sub(2).min(h - 1) * w + i % w])
                .sum::<f32>()
                / 5.0
        })
        .collect()
}

/// Thumbnail samples on level `l`: directions, weights (area; latitudes within ±60°, the poles hold the stick and
/// the camera's nadir patch) and the weighted zero-mean, unit-norm values times their weights.
struct Grid {
    w: usize,
    d: Vec<DVec3>,
    wt: Vec<f64>,
    wf: Vec<f64>,
}

impl Grid {
    fn new(t: &Eq, l: usize) -> Grid {
        let (s, w, h) = ((1 << l) as f64, t.w >> l, t.h >> l);
        let c = (s - 1.0) / 2.0 + TOFF;
        let d: Vec<DVec3> = (0..w * h)
            .map(|i| vis::ray(t.w, t.h, [s * (i % w) as f64 + c, s * (i / w) as f64 + c]))
            .collect();
        let cap = 60f64.to_radians().sin();
        let wt = d
            .iter()
            .map(|d| (d.z.abs() <= cap) as u8 as f64 * (1.0 - d.z * d.z).sqrt())
            .collect();
        let mut g = Grid {
            w,
            d,
            wt,
            wf: vec![],
        };
        let f = g.norm(t.lv[l].iter().map(|&x| x as f64).collect());
        g.wf = f.iter().zip(&g.wt).map(|(f, w)| f * w).collect();
        g
    }

    /// Weighted zero mean, unit weighted norm (the weights are constant along a row, so column shifts keep both).
    fn norm(&self, mut v: Vec<f64>) -> Vec<f64> {
        let (sw, sv) =
            (self.wt.iter().zip(&v)).fold((0.0, 0.0), |(a, b), (w, x)| (a + w, b + w * x));
        let m = sv / sw;
        let n = (self.wt.iter().zip(&v))
            .map(|(w, x)| w * (x - m) * (x - m))
            .sum::<f64>()
            .sqrt();
        v.iter_mut().for_each(|x| *x = (*x - m) / n);
        v
    }

    /// Correlation with normalised samples `v` shifted by `s` columns: Σ wf(i, j)·v(i + s, j).
    fn corr(&self, v: &[f64], s: usize) -> f64 {
        let w = self.w;
        self.wf
            .chunks(w)
            .zip(v.chunks(w))
            .map(|(a, b)| {
                let (a0, a1) = a.split_at(w - s);
                let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>();
                dot(a0, &b[s..]) + dot(a1, &b[..s])
            })
            .sum()
    }
}

/// exp(roll·X + pitch·Y)·Rz(yaw), degrees: a tilt about a horizontal axis of our output frame after a yaw.
fn rot(x: [f64; 3]) -> DQuat {
    exp(DVec3::new(x[0], x[1], 0.0) * PI / 180.0) * DQuat::from_rotation_z(x[2].to_radians())
}

/// The rotation R = `rot(x)` with thumbnail(d) = ours(R·d), for our equirect `a` at twice the thumbnail's `t` size:
/// yaw over the whole circle × tilts within ±30° in 2° steps by correlation at 160 × 80, tilts ±3° in 0.5° steps
/// (yaw ±3 px) at 320 × 160, then a pattern search down to 0.01° and a parabola per axis at 640 × 320. Returns the
/// correlation of the high-passed luma at 640 × 320 and x.
fn align(a: &Eq, t: &Eq) -> (f64, [f64; 3]) {
    let samp = |g: &Grid, l: usize, r: DQuat| -> Vec<f64> {
        g.norm(g.d.iter().map(|&d| a.at(l + 1, r * d) as f64).collect())
    };
    let mut best = (f64::MIN, [0.0; 3]);
    for (l, span, step, ds) in [(3, 15, 2.0, None), (2, 6, 0.5, Some(3))] {
        let (g, x0) = (Grid::new(t, l), best.1);
        let s0 = (-x0[2] / 360.0 * g.w as f64).round() as isize;
        best.0 = f64::MIN;
        for i in -span..=span {
            for j in -span..=span {
                let x = [x0[0] + step * i as f64, x0[1] + step * j as f64, 0.0];
                let v = samp(&g, l, rot(x));
                let shifts: Vec<usize> = match ds {
                    None => (0..g.w).collect(),
                    Some(d) => (-d..=d)
                        .map(|k| (s0 + k).rem_euclid(g.w as isize) as usize)
                        .collect(),
                };
                for s in shifts {
                    let c = g.corr(&v, s);
                    if c > best.0 {
                        let yaw =
                            (-360.0 * s as f64 / g.w as f64 + 180.0).rem_euclid(360.0) - 180.0;
                        best = (c, [x[0], x[1], yaw]);
                    }
                }
            }
        }
    }
    let g = Grid::new(t, 1);
    let f = |x: [f64; 3]| g.corr(&samp(&g, 1, rot(x)), 0);
    let (mut x, mut fx, mut st) = (best.1, f(best.1), 0.5);
    while st > 0.01 {
        let mut moved = false;
        for k in 0..6 {
            let mut y = x;
            y[k / 2] += if k % 2 == 0 { st } else { -st };
            let fy = f(y);
            if fy > fx {
                (x, fx, moved) = (y, fy, true);
            }
        }
        if !moved {
            st /= 2.0;
        }
    }
    for k in 0..3 {
        let (mut lo, mut hi, e) = (x, x, 0.05);
        (lo[k], hi[k]) = (x[k] - e, x[k] + e);
        let (fl, fh) = (f(lo), f(hi));
        let den = fl - 2.0 * fx + fh;
        if den < 0.0 {
            x[k] += (0.5 * e * (fl - fh) / den).clamp(-e, e);
        }
    }
    (f(x), x)
}

/// Tilt part of `q` (output -> body or a residual): x and y of the rotation vector (deg) of the shortest arc from Z
/// to q·Z, i.e. roll about our output X (the back, lens 0) and pitch about Y (the right).
fn lean(q: DQuat) -> [f64; 2] {
    let v = log(DQuat::from_rotation_arc(
        DVec3::Z,
        (q * DVec3::Z).normalize(),
    ));
    [v.x.to_degrees(), v.y.to_degrees()]
}

/// Colour against the thumbnail at 320 × 160 (low frequencies only) after alignment `r`, output -> body `q`: per
/// 15° segment along the seam, mean RGB 3°–15° on either side in ours `o` (base 2560 wide) and the thumbnail `t`.
/// Returns, in grey levels and averaged over the channels: median over the segments of the error of the step
/// across the seam |(o1 − o0) − (t1 − t0)| and of the error per side |o − t|, the MAE over the sphere within ±60°,
/// and the mean steps o1 − o0, t1 − t0 (lens 1 minus lens 0).
fn colour(o: &[Eq; 3], t: &[Eq; 3], r: DQuat, q: DQuat) -> [f64; 5] {
    let g = Grid::new(&t[0], 2);
    let mut seg = [[[0f64; 7]; 2]; 24]; // per segment and side: Σw, Σw·o rgb, Σw·t rgb
    let (mut mae, mut sw) = (0.0, 0.0);
    for (i, (&d, &w)) in g.d.iter().zip(&g.wt).enumerate() {
        if w == 0.0 {
            continue;
        }
        let rd = r * d;
        let ov = [0, 1, 2].map(|c| o[c].at(3, rd) as f64);
        let tv = [0, 1, 2].map(|c| t[c].lv[2][i] as f64);
        (mae, sw) = (
            mae + w * (0..3).map(|c| (ov[c] - tv[c]).abs()).sum::<f64>() / 3.0,
            sw + w,
        );
        let b = q * rd;
        let (e, a) = (b.x.asin().to_degrees(), b.y.atan2(b.z).to_degrees());
        if (3.0..=15.0).contains(&e.abs()) {
            let s = &mut seg[((a + 180.0) / 15.0) as usize % 24][(e > 0.0) as usize];
            s[0] += w;
            for c in 0..3 {
                s[1 + c] += w * ov[c];
                s[4 + c] += w * tv[c];
            }
        }
    }
    let (mut step, mut side, mut os, mut ts) = (vec![], vec![], 0.0, 0.0);
    for s in seg.iter().filter(|s| s[0][0] > 20.0 && s[1][0] > 20.0) {
        let m = |k: usize, i: usize| s[k][i] / s[k][0];
        let (mut e1, mut e2, mut d1, mut d2) = (0.0, 0.0, 0.0, 0.0);
        for c in 0..3 {
            let (od, td) = (m(1, 1 + c) - m(0, 1 + c), m(1, 4 + c) - m(0, 4 + c));
            e1 += (od - td).abs() / 3.0;
            e2 += ((m(0, 1 + c) - m(0, 4 + c)).abs() + (m(1, 1 + c) - m(1, 4 + c)).abs()) / 6.0;
            (d1, d2) = (d1 + od / 3.0, d2 + td / 3.0);
        }
        step.push(e1);
        side.push(e2);
        (os, ts) = (os + d1, ts + d2);
    }
    let n = step.len().max(1) as f64;
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    };
    [med(step), med(side), mae / sw, os / n, ts / n]
}

/// R, G, B and luma planes of an RGBA8 equirect.
fn planes(px: &[u8]) -> [Vec<f32>; 4] {
    let p = px.as_chunks::<4>().0;
    let ch = |c: usize| p.iter().map(|x| x[c] as f32).collect::<Vec<f32>>();
    let l = p
        .iter()
        .map(|x| 0.2126 * x[0] as f32 + 0.7152 * x[1] as f32 + 0.0722 * x[2] as f32)
        .collect();
    [ch(0), ch(1), ch(2), l]
}

fn ppm(path: &Path, rgb: &[[f32; 3]], w: usize, h: usize) {
    let mut b = format!("P6\n{w} {h}\n255\n").into_bytes();
    b.extend(
        rgb.iter()
            .flat_map(|p| p.map(|x| x.round().clamp(0.0, 255.0) as u8)),
    );
    std::fs::write(path, b).unwrap();
}

/// Levelling checked against the camera's own stitch, without a rig. The trailer thumbnail (record 2) is the
/// camera's equirect, NV12 1280 × 640, full-range BT.709 (the video's matrix fits it better than BT.601), levelled by
/// the camera and framed on lens 1 as we frame (residual yaw ≈ 0), of one frame: 2.97 s in where the camera marks
/// `trim_start` 3 s (frame 89 at 29.97 fps, 178 at 59.94), 1 s into a dash-cam segment, else frame 0; the best of
/// the frames within ±0.1 s is taken. Our horizon-levelled stitch of it (σ 0: the frame's own heading), 2560 × 1280,
/// is aligned to it (`align`): the residual tilt is what our levelling (IMU axes, Mahony filter) gets wrong against
/// the camera's. Accuracy: the alignment again after a known turn of our stitch; the horizon of both pictures from
/// their verticals (`tilt`) tells which one leans. Then the residual the other gains and the accelerometer alone
/// would leave (from their poses), and the colour either side of the seam with colour matching on and off
/// (`colour`). Files: v606, v627, v635 and any in `X5_LEVEL` (a path list); a CSV and thumbnail-over-stitch pictures
/// go to the fixtures.
#[test]
fn level() {
    let Some(d) = samples() else { return };
    let mut files: Vec<PathBuf> = ["v606", "v627", "v635"]
        .map(|n| d.join(format!("{n}.insv")))
        .into();
    if let Some(v) = std::env::var_os("X5_LEVEL") {
        files.extend(std::env::split_paths(&v).filter(|p| !p.as_os_str().is_empty()));
    }
    let out = fixtures().join("level");
    std::fs::create_dir_all(&out).unwrap();
    let g = gpu();
    let (w, h) = (2560, 1280);
    let known = [3.0, -2.0, 5.0];
    let gains = [(KP, KI), (0.5, 0.05), (0.05, 0.001), (0.0, 0.0)];
    let mut csv = String::from(
        "file,fps,trim_ms,k0,k,score,roll,pitch,yaw,tilt,known_err,img_ours,img_camera,cam_roll,cam_pitch,\
         kp05_roll,kp05_pitch,kp005_roll,kp005_pitch,gyro_roll,gyro_pitch,acc_roll,acc_pitch,rate_dps,acc_g,\
         step_off,step_on,side_off,side_on,mae_off,mae_on,ostep_off,ostep_on,tstep\n",
    );
    let mut rows = vec![];
    for p in &files {
        let name: &'static str =
            String::leak(p.file_stem().unwrap().to_string_lossy().into_owned());
        let c = clip_at(p.clone(), name);
        let mut b = vec![];
        Trailer::open(p)
            .unwrap()
            .get(insv::THUMBNAIL, &mut b)
            .unwrap();
        let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let (tw, th) = (u(16), u(20));
        assert_eq!(
            (tw, th, b.len()),
            (1280, 640, 40 + tw * th * 3 / 2),
            "{name}: thumbnail"
        );
        let rgb: Vec<[f32; 3]> = (0..tw * th)
            .map(|i| {
                let (x, y) = (i % tw, i / tw);
                let l = b[40 + i] as f32;
                let k = 40 + tw * th + y / 2 * tw + x / 2 * 2;
                let (cb, cr) = (b[k] as f32 - 128.0, b[k + 1] as f32 - 128.0);
                [
                    l + 1.5748 * cr,
                    l - 0.187324 * cb - 0.468124 * cr,
                    l + 1.8556 * cb,
                ]
                .map(|v| v.clamp(0.0, 255.0))
            })
            .collect();
        let tl = Eq::new(
            rgb.iter()
                .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
                .collect(),
            tw,
            th,
            3,
            true,
        );
        let tc = [0, 1, 2].map(|k| Eq::new(rgb.iter().map(|p| p[k]).collect(), tw, th, 2, false));

        let fr = Pose::euler(0.0, 0.0, if c.m.stream == 4 { 180.0 } else { 0.0 });
        let hp: Vec<Vec<Pose>> = gains
            .iter()
            .map(|&(kp, ki)| {
                let tr = Track::new(&c.imu, kp, ki);
                stab::poses(&tr, &c.ex, &Opts::new(&c.m, Lock::Horizon, 0.0))
                    .iter()
                    .map(|p| Pose {
                        q: p.q * fr.q,
                        rs: [p.rs; 2],
                    })
                    .collect()
            })
            .collect();
        // the thumbnail's time: trim_start, else 1 s into a dash-cam segment (VIDEO_DASH_CAM), else frame 0
        let ts = match (c.m.trim[0], c.m.group.kind) {
            (0, 23) => 1.0,
            (t, _) => t as f64 * 1e-3,
        };
        let k0 = ((ts * c.m.fps).round() as usize).min(c.n - 1);
        let ks: Vec<usize> = (((ts - 0.1) * c.m.fps).max(0.0).ceil() as usize
            ..=(((ts + 0.1) * c.m.fps) as usize).min(c.n - 1))
            .collect();
        // per candidate frame: ours, ours turned by `known`, ours with colour matching (RGBA)
        let mut px: Vec<[Vec<u8>; 3]> = vec![];
        let mut st: Option<Stitcher> = None;
        each(&g, &c, ks[ks.len() - 1] + 1, w, h, |k, rig| {
            if !ks.contains(&k) {
                return;
            }
            let p = hp[0][k];
            let st = st.get_or_insert_with(|| {
                let o = stitch::Opts {
                    blend: c.m.blend.unwrap_or(6.0),
                    ..Default::default()
                };
                Stitcher::new(&rig.cx, &c.cal, c.fr, w, h, o).unwrap()
            });
            // a fresh fit on this frame alone, as an export's first frame
            st.reset();
            st.pose(&p);
            let mut enc = g.dev.create_command_encoder(&Default::default());
            st.encode(&mut enc, [&rig.src[0], &rig.src[1]]);
            g.q.submit([enc.finish()]);
            let pk = Pose {
                q: p.q * rot(known).as_quat(),
                ..p
            };
            px.push([
                rig.render(&g, &p),
                rig.render(&g, &pk),
                stitch::read_texture(&g.dev, &g.q, &st.out),
            ]);
        });
        let (w, h) = (w as usize, h as usize);
        let lum = |px: &[u8]| Eq::new(planes(px)[3].clone(), w, h, 4, true);
        let fits: Vec<(f64, [f64; 3])> = std::thread::scope(|s| {
            let j: Vec<_> = px
                .iter()
                .map(|f| s.spawn(|| align(&lum(&f[0]), &tl)))
                .collect();
            j.into_iter().map(|j| j.join().unwrap()).collect()
        });
        let i = (0..fits.len())
            .max_by(|&a, &b| fits[a].0.total_cmp(&fits[b].0))
            .unwrap();
        let (k, (score, x)) = (ks[i], fits[i]);
        let r = rot(x);
        // turned by K, the residual becomes K⁻¹·R
        let kx = align(&lum(&px[i][1]), &tl).1;
        let kerr = log(rot(kx) * (rot(known).inverse() * r).inverse())
            .length()
            .to_degrees();

        let qa = hp[0][k].q.as_dquat();
        let res = |q: DQuat| lean(q.inverse() * qa * r);
        let gr: Vec<[f64; 2]> = hp[1..].iter().map(|p| res(p[k].q.as_dquat())).collect();
        // the accelerometer alone (±25 ms): the least turn of our pose that puts its up on the measured one
        let tf = c.t(k) - 0.5 * c.ex[k].time + OFFSET;
        let near: Vec<DVec3> = (c.imu.iter())
            .filter(|s| (s.t as f64 * 1e-6 - tf).abs() < 0.025)
            .map(|s| AXES * DVec3::new(s.acc[0] as f64, s.acc[1] as f64, s.acc[2] as f64))
            .collect();
        let am = near.iter().sum::<DVec3>() / near.len() as f64;
        let ag = near.iter().map(|a| a.length()).sum::<f64>() / near.len() as f64;
        let ra = res(DQuat::from_rotation_arc(qa * DVec3::Z, am.normalize()) * qa);
        let tr = Track::new(&c.imu, KP, KI);
        let rate = tr.turn(tf - 0.01, tf + 0.01).length().to_degrees() / 0.02;
        let cam = lean(qa.inverse());
        let rt = lean(r);
        // which of the two is level: the horizon tilt of each picture from its verticals (`tilt`; NaN: too few)
        let tb: Vec<u8> = (rgb.iter())
            .flat_map(|p| [p[0] as u8, p[1] as u8, p[2] as u8, 255])
            .collect();
        let it = [tilt(&px[i][0], w, h), tilt(&tb, tw, th)]
            .map(|t| t.map_or(f64::NAN, |t| t.0[0].hypot(t.0[1])));

        let col = |px: &[u8]| {
            let pl = planes(px);
            colour(
                &[0, 1, 2].map(|k| Eq::new(pl[k].clone(), w, h, 3, false)),
                &tc,
                r,
                qa,
            )
        };
        let (off, on) = (col(&px[i][0]), col(&px[i][2]));

        // thumbnail over our aligned stitch, 640 × 320 each
        let gv = Grid::new(&tc[0], 1);
        let oc = [0, 1, 2].map(|k| Eq::new(planes(&px[i][0])[k].clone(), w, h, 2, false));
        let mut pic: Vec<[f32; 3]> = (0..gv.d.len())
            .map(|j| [0, 1, 2].map(|k| tc[k].lv[1][j]))
            .collect();
        pic.extend(gv.d.iter().map(|&d| [0, 1, 2].map(|k| oc[k].at(2, r * d))));
        ppm(&out.join(format!("{name}.ppm")), &pic, 640, 640);

        eprintln!(
            "{name}: frame {k} (trim {} ms → {k0}), score {score:.3}; residual roll {:.2}° pitch {:.2}° (tilt {:.2}°) \
             yaw {:.2}°; known turn recovered within {kerr:.3}°; horizon in the pictures ours {:.2}° camera {:.2}°; \
             camera tilt {:.1}° / {:.1}°; gains 0.5/0.05 \
             {:.2?}, 0.05/0.001 {:.2?}, gyro only {:.2?}, accelerometer {:.2?} ({rate:.0}°/s, {ag:.2} g); colour \
             off / on: seam step error {:.2} / {:.2}, side {:.2} / {:.2}, MAE {:.2} / {:.2}, step ours {:.2} / {:.2}, \
             camera {:.2}",
            c.m.trim[0],
            rt[0],
            rt[1],
            rt[0].hypot(rt[1]),
            x[2],
            it[0],
            it[1],
            cam[0],
            cam[1],
            gr[0],
            gr[1],
            gr[2],
            ra,
            off[0],
            on[0],
            off[1],
            on[1],
            off[2],
            on[2],
            off[3],
            on[3],
            off[4],
        );
        csv += &format!(
            "{name},{},{},{k0},{k},{score:.4},{:.3},{:.3},{:.3},{:.3},{kerr:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},\
             {:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{rate:.1},{ag:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},\
             {:.3}\n",
            c.m.fps,
            c.m.trim[0],
            rt[0],
            rt[1],
            x[2],
            rt[0].hypot(rt[1]),
            it[0],
            it[1],
            cam[0],
            cam[1],
            gr[0][0],
            gr[0][1],
            gr[1][0],
            gr[1][1],
            gr[2][0],
            gr[2][1],
            ra[0],
            ra[1],
            off[0],
            on[0],
            off[1],
            on[1],
            off[2],
            on[2],
            off[3],
            on[3],
            off[4],
        );
        std::fs::write(out.join("level.csv"), &csv).unwrap();
        rows.push((name, score, k.abs_diff(k0), rt, kerr, gr, ra, off, on));
    }
    let ok: Vec<_> = rows.iter().filter(|r| r.1 >= 0.5).collect();
    let n = ok.len() as f64;
    let rms = |f: &dyn Fn(&&_) -> [f64; 2]| {
        (ok.iter()
            .map(f)
            .map(|v| v[0] * v[0] + v[1] * v[1])
            .sum::<f64>()
            / n)
            .sqrt()
    };
    let mean = [0, 1].map(|a| ok.iter().map(|r| r.3[a]).sum::<f64>() / n);
    let kmax = ok.iter().map(|r| r.4).fold(0.0, f64::max);
    let rg: Vec<f64> = (0..3).map(|i| rms(&|r| r.5[i])).collect();
    let won = ok.iter().filter(|r| r.8[0] < r.7[0]).count();
    eprintln!(
        "{} of {} files aligned (score ≥ 0.5): residual tilt rms {:.2}° (mean roll {:.2}° pitch {:.2}°), known turn \
         recovered within {kmax:.3}°; rms with gains 0.5/0.05 {:.2}°, 0.05/0.001 {:.2}°, gyro only {:.2}°, \
         accelerometer alone {:.2}°; colour matching lowers the seam step error on {won} of them (median off {:.2}, \
         on {:.2})",
        ok.len(),
        rows.len(),
        rms(&|r| r.3),
        mean[0],
        mean[1],
        rg[0],
        rg[1],
        rg[2],
        rms(&|r| r.6),
        {
            let mut v: Vec<f64> = ok.iter().map(|r| r.7[0]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        },
        {
            let mut v: Vec<f64> = ok.iter().map(|r| r.8[0]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        },
    );
    eprintln!("CSV and pictures: {}", out.display());
    // the samples align at the thumbnail's frame or its neighbour, the method resolves a known turn
    for r in &rows[..3] {
        assert!(
            r.1 > 0.8 && r.2 <= 1 && r.4 < 0.05,
            "{}: score {} at {} frames from trim, known turn {}°",
            r.0,
            r.1,
            r.2,
            r.4
        );
    }
}
