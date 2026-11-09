//! Footage tests on `X5_SAMPLES` (skip when unset); outputs (PNG frames, reframed videos, CSV tracks) go to
//! `X5_FIXTURES` (default: the system temp dir). ffmpeg is a test tool here only: it decodes the lens streams to NV12
//! and encodes the PNGs and videos. The lenses are stitched by crate stitch with crate stab's direction-locked poses,
//! so the equirect is level and its heading fixed: what moves in it is the subjects (and parallax). Run with
//! `--release`.

use glam::{Vec2, Vec3};
use media::insv::{self, Meta, Trailer};
use std::{
    collections::{BTreeSet, HashSet},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Mutex, MutexGuard, OnceLock},
    time::Instant,
};
use stitch::{Calib, Ctx, Frame, Pose, Preview, Src, Stitcher, View};
use track::{
    Ai, Blur, Cam, Emb, Framer, Gallery, Hide, LOW, Region, Sample, Scorer, Seen, Sig, State,
    Subject, Tracker, face_tiles, iou, sim, sphere, yaw_pitch,
};

/// Lens texture side the streams are scaled to, equirect for the networks, output view.
const S: u32 = 1920;
const EW: u32 = 3072;
const EH: u32 = 1536;
const VW: u32 = 1280;
const VH: u32 = 720;

struct Gpu {
    inst: wgpu::Instance,
    ad: wgpu::Adapter,
    dev: wgpu::Device,
    q: wgpu::Queue,
}

/// A device opened like `video::Gpu` (every adapter feature: burn needs them, NV12 among them; the experimental
/// ones), one per test binary; the lock keeps GPU tests (and their timings) apart.
fn gpu() -> MutexGuard<'static, Gpu> {
    static G: OnceLock<Mutex<Gpu>> = OnceLock::new();
    G.get_or_init(|| {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let ad = pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("adapter");
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
        .expect("device");
        eprintln!("GPU: {}", ad.get_info().name);
        Mutex::new(Gpu { inst, ad, dev, q })
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

/// A `<clip>_track.csv` of `subject` (a reference track checked by eye): per frame direction, score (0: the subject
/// unseen), box height (degrees).
fn reference(p: &str) -> Vec<(Vec3, f32, f32)> {
    std::fs::read_to_string(p)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<&str> = l.split(',').collect();
            let f = |i: usize| v[i].parse::<f32>().unwrap();
            (track::dir(f(3), f(4)), f(2), f(8))
        })
        .collect()
}

fn fixtures() -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES")
        .map_or_else(|| std::env::temp_dir().join("x5-track"), PathBuf::from);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Clip {
    path: PathBuf,
    m: Meta,
    cal: Calib,
    fr: Frame,
    n: usize,
    fps: f32,
    /// Per frame: direction lock (stab) and the X5's framing (panorama centre on lens 1).
    poses: Vec<Pose>,
}

fn clip(d: &Path, name: &'static str) -> Clip {
    let path = d.join(format!("{name}.insv"));
    let t = Trailer::open(&path).unwrap();
    let tr = stab::Track::new(&t.imu().unwrap(), stab::KP, stab::KI);
    let ex = t.frames(insv::EXPOSURE).unwrap();
    let fr = Pose::euler(0.0, 0.0, if t.meta.stream == 4 { 180.0 } else { 0.0 });
    let poses = stab::poses(
        &tr,
        &ex,
        &stab::Opts::new(&t.meta, stab::Lock::Direction, 0.5),
    )
    .iter()
    .map(|p| Pose {
        q: p.q * fr.q,
        rs: [p.rs; 2],
    })
    .collect::<Vec<_>>();
    let n = media::mp4::Mp4::open(&path).unwrap().tracks[0]
        .samples
        .len()
        .min(poses.len());
    Clip {
        cal: Calib::parse(&t.meta.offset).unwrap(),
        fr: Frame {
            w: S,
            h: S,
            crop: t.meta.crop,
            shift: t.meta.shift,
        },
        n,
        fps: t.meta.fps as f32,
        poses,
        m: t.meta,
        path,
    }
}

/// ffmpeg's hardware decoder: Vulkan Video, as crate video on Linux; VideoToolbox on macOS.
const HWACCEL: &str = if cfg!(target_os = "macos") {
    "videotoolbox"
} else {
    "vulkan"
};

/// Both lens streams as NV12 at S × S, frame by frame (two ffmpeg processes).
struct Dec {
    kids: Vec<Child>,
    buf: [Vec<u8>; 2],
}

impl Dec {
    fn new(c: &Clip, n: usize) -> Dec {
        Dec::sized(c, n, S)
    }

    /// At `ls` × `ls`.
    fn sized(c: &Clip, n: usize, ls: u32) -> Dec {
        // DUAL_STREAM_TRACK_REVERSE (4): track 0 holds lens 1
        let tracks = if c.m.stream == 4 { [1, 0] } else { [0, 1] };
        let kids = tracks
            .map(|t| {
                Command::new("ffmpeg")
                    .args(["-v", "error", "-hwaccel", HWACCEL, "-i"])
                    .arg(&c.path)
                    .args(["-map", &format!("0:v:{t}"), "-frames:v", &n.to_string()])
                    .args(["-vf", &format!("scale={ls}:{ls}"), "-f", "rawvideo"])
                    .args(["-pix_fmt", "nv12", "-"])
                    .stdout(Stdio::piped())
                    .spawn()
                    .expect("ffmpeg")
            })
            .into();
        let n = (ls * ls * 3 / 2) as usize;
        Dec {
            kids,
            buf: [vec![0; n], vec![0; n]],
        }
    }

    fn next(&mut self) -> bool {
        (0..2).all(|k| {
            let o = self.kids[k].stdout.as_mut().unwrap();
            o.read_exact(&mut self.buf[k]).is_ok()
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

/// Lens textures, the equirect stitcher, a preview from the lenses into a VW × VH view. A lens is one NV12 texture,
/// written and sampled through its plane aspects as crate video decodes on Linux, or an R8 and an RG8 texture as on
/// macOS (Metal has no NV12 textures).
struct Rig {
    tex: [[(wgpu::Texture, wgpu::TextureAspect); 2]; 2],
    src: [Src; 2],
    st: Stitcher,
    pv: Preview,
    view: wgpu::Texture,
}

impl Rig {
    fn new(g: &Gpu, c: &Clip) -> Rig {
        Rig::sized(g, c, S, EW)
    }

    /// Lenses `ls` × `ls` (S in `new`), the equirect `ew` × `ew` / 2 (EW × EH).
    fn sized(g: &Gpu, c: &Clip, ls: u32, ew: u32) -> Rig {
        let fr = Frame {
            w: ls,
            h: ls,
            ..c.fr
        };
        let cx = Ctx::new(&g.dev, &g.q);
        let t = |w, h, format, usage| {
            g.dev.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        use wgpu::{TextureAspect as A, TextureFormat as F, TextureUsages as U};
        let tex = [0, 1].map(|_| match cfg!(target_os = "macos") {
            true => [(F::R8Unorm, 1), (F::Rg8Unorm, 2)].map(|(f, s)| {
                (
                    t(ls / s, ls / s, f, U::TEXTURE_BINDING | U::COPY_DST),
                    A::All,
                )
            }),
            false => {
                let nv = t(ls, ls, F::NV12, U::TEXTURE_BINDING | U::COPY_DST);
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
        let blend = c.m.blend.unwrap_or(6.0);
        let o = stitch::Opts {
            blend,
            ..Default::default()
        };
        let rgba = wgpu::TextureFormat::Rgba8Unorm;
        Rig {
            src: [0, 1].map(|l| cx.rect(&planes(&tex[l]), [0, 0, ls, ls])),
            st: Stitcher::new(&cx, &c.cal, fr, ew, ew / 2, o).unwrap(),
            pv: Preview::new(&cx, &c.cal, fr, blend, Some(0.1), rgba),
            view: t(VW, VH, rgba, U::RENDER_ATTACHMENT | U::COPY_SRC),
            tex,
        }
    }

    fn upload(&self, g: &Gpu, nv: &[Vec<u8>; 2]) {
        let ls = self.tex[0][0].0.width();
        let n = (ls * ls) as usize;
        for (l, nv) in self.tex.iter().zip(nv) {
            for ((t, aspect), data, w, bpp) in
                [(&l[0], &nv[..n], ls, 1), (&l[1], &nv[n..], ls / 2, 2)]
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

    /// Stitches the uploaded lenses into the equirect for pose `p` (submitted).
    fn stitch(&self, g: &Gpu, p: &Pose) {
        self.st.pose(p);
        let mut enc = g.dev.create_command_encoder(&Default::default());
        self.st.encode(&mut enc, [&self.src[0], &self.src[1]]);
        g.q.submit([enc.finish()]);
    }

    /// The flat view `v` (16:9) for pose `p`, from the lenses, read back as RGBA8.
    fn view(&self, g: &Gpu, p: &Pose, v: &View) -> Vec<u8> {
        self.pv.pose(p);
        let mut enc = g.dev.create_command_encoder(&Default::default());
        self.pv.colour(&mut enc, [&self.src[0], &self.src[1]]);
        let tv = self.view.create_view(&Default::default());
        self.pv
            .render(&mut enc, &tv, [VW, VH], v, [&self.src[0], &self.src[1]]);
        g.q.submit([enc.finish()]);
        stitch::read_texture(&g.dev, &g.q, &self.view)
    }
}

/// An ffmpeg process that encodes raw RGBA frames of w × h to `file` (H.264).
fn encoder(file: &Path, w: u32, h: u32, fps: f32) -> Child {
    Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba"])
        .args(["-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i", "-"])
        .args([
            "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
        ])
        .arg(file)
        .stdin(Stdio::piped())
        .spawn()
        .expect("ffmpeg")
}

fn png(file: &Path, px: &[u8], w: u32, h: u32) {
    let mut e = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba"])
        .args(["-s", &format!("{w}x{h}"), "-i", "-", "-frames:v", "1"])
        .arg(file)
        .stdin(Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    e.stdin.as_mut().unwrap().write_all(px).unwrap();
    drop(e.stdin.take());
    assert!(e.wait().unwrap().success());
}

/// Draws the outline of box (centre, angular size) as seen by `pt` (direction -> pixel) into an RGBA image.
fn outline(
    px: &mut [u8],
    w: u32,
    h: u32,
    b: (Vec3, Vec2),
    col: [u8; 3],
    pt: impl Fn(Vec3) -> Option<Vec2>,
) {
    let t = (b.1 / 2.0).to_array().map(|a| a.to_radians().tan());
    let c = Cam::new(b.0, Vec3::Z, Vec2::new(t[0], t[1]));
    for i in 0..400 {
        let s = i as f32 / 100.0;
        let (e, f) = (s as usize, s.fract() * 2.0 - 1.0);
        let p = [
            Vec2::new(f, -1.0),
            Vec2::new(1.0, f),
            Vec2::new(-f, 1.0),
            Vec2::new(-1.0, -f),
        ][e];
        let Some(q) = pt(c.ray(p)) else { continue };
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let (x, y) = (q.x as i64 + dx, q.y as i64 + dy);
            if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                px[(y as usize * w as usize + x as usize) * 4..][..3].copy_from_slice(&col);
            }
        }
    }
}

/// Equirect pixel of direction `d`.
fn epx(d: Vec3) -> Option<Vec2> {
    let t = (-d.y).atan2(d.x) / std::f32::consts::TAU;
    Some(Vec2::new(
        t.rem_euclid(1.0) * EW as f32,
        d.z.clamp(-1.0, 1.0).acos() / std::f32::consts::PI * (EH - 1) as f32,
    ))
}

/// The view pixel of direction `d` for view `v`.
fn vpx(v: &View) -> impl Fn(Vec3) -> Option<Vec2> {
    let c = Cam::view(v.yaw, v.pitch, v.roll, v.fov, VW as f32 / VH as f32);
    move |d| {
        c.px(d)
            .map(|p| (p + 1.0) / 2.0 * Vec2::new(VW as f32, VH as f32))
    }
}

/// Frames `ks` of `$X5_LOOK` (v606, v635, default v627) stitched, with the detections over `sphere()` drawn, as PNGs:
/// to pick subjects by eye and to see what the detector finds.
#[test]
fn look() {
    let Some(d) = samples() else { return };
    let name = match std::env::var("X5_LOOK").as_deref() {
        Ok("v606") => "v606",
        Ok("v635") => "v635",
        _ => "v627",
    };
    let ks: Vec<usize> = std::env::var("X5_LOOK_FRAMES")
        .map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_else(|_| vec![0, 150, 300]);
    let g = gpu();
    let c = clip(&d, name);
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    let (rig, mut dec) = (Rig::new(&g, &c), Dec::new(&c, ks.iter().max().unwrap() + 1));
    let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
    for k in 0.. {
        if !dec.next() || k > *ks.iter().max().unwrap() {
            break;
        }
        if !ks.contains(&k) {
            continue;
        }
        rig.upload(&g, &dec.buf);
        rig.stitch(&g, &c.poses[k]);
        let t = Instant::now();
        let dets = ai.detect(&pano, &sphere(), 0.3);
        eprintln!(
            "{name} frame {k}: {} detections in {:.0} ms",
            dets.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
        let mut px = stitch::read_texture(&g.dev, &g.q, &rig.st.out);
        for x in &dets {
            let [y, p] = yaw_pitch(x.dir);
            eprintln!(
                "  {:>12} {:.2} yaw {y:7.1} pitch {p:6.1} size {:5.1} x {:5.1}",
                track::CLASSES[x.cls as usize],
                x.score,
                x.size.x,
                x.size.y
            );
            outline(&mut px, EW, EH, x.on(), [255, 40, 40], epx);
        }
        png(&fixtures().join(format!("{name}_{k}_det.png")), &px, EW, EH);
    }
}

/// Median, 95th percentile and maximum.
fn pct(v: &mut [f64]) -> (f64, f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    (v[n / 2], v[(n * 95 / 100).min(n - 1)], v[n - 1])
}

/// Horizontal field of view (degrees, 16:9) in which a subject `h`° tall fills 45 % of the height.
fn fit(h: f32) -> f32 {
    let v = (h / 0.45).clamp(1.0, 170.0).to_radians() / 2.0;
    (2.0 * (v.tan() * VW as f32 / VH as f32).atan())
        .to_degrees()
        .clamp(40.0, 110.0)
}

/// Tracks one subject through a clip and frames the clip automatically, then renders both as 1280 × 720 videos
/// (with the track and the auto-framing path as CSV).
/// `X5_TRACK` = `clip:yaw:pitch` picks the subject as a user taps it, with `Ai::pick` on frame 0 (default
/// `v606:-71:-43`, the woman in the purple coat, close at the start, then left behind as the camera walks through the
/// gate). Per frame, the detector also looks at the tracker's area: how often its box
/// of the subject's class overlaps the tracker's (IoU > 0.5) measures agreement. Auto framing detects over
/// `sphere()` every 10 frames. `X5_BLANK` = `start:len` hands the tracker a black panorama for those
/// frames (the subject walks on unseen), to make it lose the subject and find it again by detection. With `X5_REF` (a
/// reference track of the subject, see `reid`) the tracker must keep the subject itself on half the frames, not just a
/// box: on someone else it keeps a box too.
#[test]
fn subject() {
    let Some(d) = samples() else { return };
    let arg = std::env::var("X5_TRACK").unwrap_or_else(|_| "v606:-71:-43".into());
    let a: Vec<&str> = arg.split(':').collect();
    let name = match a[0] {
        "v627" => "v627",
        "v635" => "v635",
        _ => "v606",
    };
    let at = track::dir(a[1].parse().unwrap(), a[2].parse().unwrap());
    let n: usize = std::env::var("X5_FRAMES").map_or(usize::MAX, |v| v.parse().unwrap());
    let blank = std::env::var("X5_BLANK").map_or(0..0, |v| {
        let (a, b) = v.split_once(':').unwrap();
        let a: usize = a.parse().unwrap();
        a..a + b.parse::<usize>().unwrap()
    });
    let g = gpu();
    let c = clip(&d, name);
    let n = n.min(c.n);
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    let t = Instant::now();
    ai.warm().unwrap();
    eprintln!("warm-up {:.1} s", t.elapsed().as_secs_f64());
    let rig = Rig::new(&g, &c);
    let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
    let black = g.dev.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 64,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let black = ai.pano(&black.create_view(&Default::default()));
    let tiles = sphere();
    let mut fm = Framer::new(VW as f32 / VH as f32);
    let (mut tr, mut log) = (None::<Tracker>, vec![]);
    // per frame: sample, best IoU with a detection of the subject's class, times (ms): crop, tracker, detector in
    // the step, the agreement check
    let mut rows: Vec<(Sample, f32, [f64; 5])> = vec![];
    // detections above the ring's reach (pitch > 32.5°: the zenith tile's alone)
    let (mut cls, mut ring_ms, mut up) = (0, vec![], 0);
    // re-ID calibration: every key frame's people scored against the tracker's gallery (frame, yaw, pitch, height,
    // similarity to the subject, to the others)
    let (mut keyd, mut reid) = (vec![], String::new());
    let mut dec = Dec::new(&c, n);
    let t0 = Instant::now();
    for k in 0..n {
        assert!(dec.next(), "decode frame {k}");
        rig.upload(&g, &dec.buf);
        rig.stitch(&g, &c.poses[k]);
        // the stitch done, so that the times below are the networks' and crops' alone
        let _ = g.dev.poll(wgpu::PollType::wait_indefinitely());
        if k % 10 == 0 {
            let t = Instant::now();
            let ds = ai.detect(&pano, &tiles, 0.3);
            ring_ms.push(t.elapsed().as_secs_f64() * 1e3);
            up += ds.iter().filter(|x| yaw_pitch(x.dir)[1] > 32.5).count();
            keyd = ds.iter().filter(|x| x.cls == 0).map(|x| x.on()).collect();
            fm.push(k, ds);
        }
        let Some(t) = tr.as_mut() else {
            let s = ai
                .pick(&pano, at)
                .expect("a detection at the picked direction");
            let [y, p] = yaw_pitch(s.dir);
            eprintln!(
                "{name}: subject {} at yaw {y:.1} pitch {p:.1}, {:.1}° × {:.1}°",
                track::CLASSES[s.cls as usize],
                s.size.x,
                s.size.y
            );
            cls = s.cls;
            let mut t = Tracker::new(&mut ai, &pano, Subject::from(s)).unwrap();
            t.fps = c.fps;
            tr = Some(t);
            rows.push((
                Sample {
                    raw: s.dir,
                    dir: s.dir,
                    size: s.size,
                    score: 1.0,
                    state: State::Found,
                },
                1.0,
                [0.0; 5],
            ));
            continue;
        };
        ai.timing = true;
        ai.times = Default::default();
        let tp = if blank.contains(&k) { &black } else { &pano };
        let s = t.step(&mut ai, tp);
        let tm = ai.times;
        if k % 10 == 0 && !blank.contains(&k) {
            for (b, e) in keyd.iter().zip(ai.embed(&pano, &keyd)) {
                if let Some((me, not)) = t.appearance(&e) {
                    let [y, p] = yaw_pitch(b.0);
                    reid += &format!("{k},{y:.2},{p:.2},{:.2},{me:.4},{not:.4}\n", b.1.y);
                }
            }
        }
        // agreement: the detector on a crop around the tracker's box (at least 40° wide)
        let e = (2.0 * (s.size.element_product().sqrt() / 2.0).to_radians().tan()).max(0.364);
        let ta = Instant::now();
        let ds = ai.detect(&pano, &[Cam::new(s.raw, Vec3::Z, Vec2::splat(e))], 0.3);
        let ta = ta.elapsed().as_secs_f64() * 1e3;
        ai.timing = false;
        let best = ds
            .iter()
            .filter(|x| x.cls == cls)
            .map(|x| iou((s.raw, s.size), x.on(), false))
            .fold(0f32, f32::max);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        rows.push((
            s,
            best,
            [ms(tm.crop), ms(tm.vit), ms(tm.det), ta, ms(tm.reid)],
        ));
        if s.state != rows[rows.len() - 2].0.state {
            let [y, p] = yaw_pitch(s.raw);
            log.push(format!(
                "frame {k}: {:?} (score {:.2}, yaw {y:.1} pitch {p:.1}, {:.1}° × {:.1}°)",
                s.state, s.score, s.size.x, s.size.y
            ));
        }
    }
    drop(dec);
    eprintln!(
        "{name}: analysed {n} frames in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
    for l in &log {
        eprintln!("  {l}");
    }
    let sm: Vec<Sample> = rows.iter().map(|r| r.0).collect();
    // over the tracked frames: frame 0 is the pick
    let boxed = sm[1..].iter().filter(|s| s.score > 0.0).count();
    let agree = rows[1..]
        .iter()
        .filter(|r| r.0.score > 0.0 && r.1 > 0.5)
        .count();
    if !blank.is_empty() {
        let b = &sm[blank.start - 1];
        let back = (blank.end..n).find(|&k| sm[k].score > 0.0);
        eprintln!(
            "{name}: blank frames {blank:?}; subject back at frame {back:?} ({:?}), {:.1}° from where it was last seen",
            back.map(|k| sm[k].state),
            back.map_or(0.0, |k| sm[k].raw.angle_between(b.raw).to_degrees())
        );
    }
    // frames from the gap's end (else 1) with the box on the subject: its centre within half the reference box's height
    // (2° at least) of the reference's, a frame the reference has no subject on taking the nearest one that has
    let on = std::env::var("X5_REF").ok().map(|rp| {
        let rf = reference(&rp);
        let has: Vec<usize> = (0..rf.len()).filter(|&k| rf[k].1 > 0.0).collect();
        let ks = blank.end.max(1)..n.min(rf.len());
        let on = ks
            .clone()
            .filter(|&k| {
                let r = rf[*has.iter().min_by_key(|&&i| (i.abs_diff(k), i)).unwrap()];
                sm[k].score > 0.0
                    && sm[k].raw.angle_between(r.0).to_degrees() <= (0.5 * r.2).max(2.0)
            })
            .count();
        eprintln!(
            "{name}: box on the subject (X5_REF) on {on}/{} frames from frame {} ({:.1} %)",
            ks.len(),
            ks.start,
            100.0 * on as f64 / ks.len().max(1) as f64
        );
        (on, ks.len())
    });
    let lost = sm
        .windows(2)
        .filter(|w| w[1].state == State::Lost && w[0].state != State::Lost)
        .count();
    let found = sm
        .iter()
        .skip(1)
        .filter(|s| s.state == State::Found)
        .count();
    let col = |i: usize| -> Vec<f64> {
        rows[1..]
            .iter()
            .map(|r| r.2[i])
            .filter(|v| *v > 0.0)
            .collect()
    };
    let (c0, v0, d0, a0, r0) = (
        pct(&mut col(0)),
        pct(&mut col(1)),
        pct(&mut col(2)),
        pct(&mut col(3)),
        pct(&mut col(4)),
    );
    eprintln!(
        "{name}: box on {boxed}/{} frames; detector agrees (IoU > 0.5) on {agree}/{boxed} ({:.1} %); lost {lost}×, \
         found again {found}×",
        n - 1,
        100.0 * agree as f64 / boxed.max(1) as f64
    );
    eprintln!(
        "per frame, ms median / p95 / max: crop {:.2} / {:.2} / {:.1} ({} frames), tracker {:.2} / {:.2} / {:.1}, \
         detector in the step {:.2} / {:.2} / {:.1} ({} runs), re-ID in the step {:.2} / {:.2} / {:.1} ({} runs), \
         agreement check (1 tile) {:.2} / {:.2} / {:.1}",
        c0.0,
        c0.1,
        c0.2,
        col(0).len(),
        v0.0,
        v0.1,
        v0.2,
        d0.0,
        d0.1,
        d0.2,
        col(2).len(),
        r0.0,
        r0.1,
        r0.2,
        col(4).len(),
        a0.0,
        a0.1,
        a0.2
    );
    let csv: String = rows
        .iter()
        .enumerate()
        .map(|(k, (s, i, t))| {
            let [y, p] = yaw_pitch(s.raw);
            let [ys, ps] = yaw_pitch(s.dir);
            format!(
                "{k},{:?},{:.3},{y:.2},{p:.2},{ys:.2},{ps:.2},{:.2},{:.2},{i:.3},{:.2},{:.2},{:.2},{:.2},{:.2}\n",
                s.state, s.score, s.size.x, s.size.y, t[0], t[1], t[2], t[3], t[4]
            )
        })
        .collect();
    std::fs::write(
        fixtures().join(format!("{name}_reid.csv")),
        "frame,yaw,pitch,h,me,not\n".to_string() + &reid,
    )
    .unwrap();
    std::fs::write(
        fixtures().join(format!("{name}_track.csv")),
        "frame,state,score,yaw,pitch,yaw_s,pitch_s,w,h,iou,crop_ms,vit_ms,det_ms,agree_ms,reid_ms\n"
            .to_string() + &csv,
    )
    .unwrap();

    // render: the track smoothed for keyframes (σ 5 frames), and the auto-framing path
    let dirs = track::smooth(&sm, 5.0);
    let mut h = sm[0].size.y;
    let path = fm.path(n, c.fps);
    let subj: Vec<usize> = path
        .iter()
        .filter_map(|s| s.subject)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let r = pct(&mut ring_ms);
    let csv: String = path
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let [y, p] = yaw_pitch(s.dir);
            format!("{k},{y:.2},{p:.2},{:.2},{:?}\n", s.fov, s.subject)
        })
        .collect();
    std::fs::write(
        fixtures().join(format!("{name}_path.csv")),
        "frame,yaw,pitch,fov,subject\n".to_string() + &csv,
    )
    .unwrap();
    // the lowest pitch any view's lower edge reaches
    let edge = path
        .iter()
        .map(|s| {
            let v = ((s.fov / 2.0).to_radians().tan() * VH as f32 / VW as f32).atan();
            yaw_pitch(s.dir)[1] - v.to_degrees()
        })
        .fold(90f32, f32::min);
    eprintln!(
        "auto framing: {} key frames, detection over 9 tiles {:.1} / {:.1} / {:.1} ms (median / p95 / max), {up} \
         detections above pitch 32.5; tracklets framed: {subj:?}; lowest view edge {edge:.1}°",
        n.div_ceil(10),
        r.0,
        r.1,
        r.2
    );
    assert!(edge >= LOW - 0.01, "a view reaches into the nadir zone");
    let files = [format!("{name}_track.mp4"), format!("{name}_auto.mp4")];
    let mut enc = files
        .clone()
        .map(|f| encoder(&fixtures().join(f), VW, VH, c.fps));
    let mut dec = Dec::new(&c, n);
    let sample = [n / 8, n / 3, n / 2, 2 * n / 3, n - 1];
    for k in 0..n {
        assert!(dec.next());
        rig.upload(&g, &dec.buf);
        let s = &sm[k];
        if s.score > 0.0 {
            h += 0.1 * (s.size.y - h);
        }
        for (i, (d, fov)) in [(dirs[k], fit(h)), (path[k].dir, path[k].fov)]
            .into_iter()
            .enumerate()
        {
            let [yaw, pitch] = yaw_pitch(d);
            let v = View {
                yaw,
                pitch,
                fov,
                ..Default::default()
            };
            let mut px = rig.view(&g, &c.poses[k], &v);
            if i == 0 {
                let col = match s.state {
                    State::Tracked => [40, 255, 40],
                    State::Checked => [255, 220, 0],
                    State::Found => [60, 120, 255],
                    _ => [255, 40, 40],
                };
                outline(&mut px, VW, VH, (s.raw, s.size), col, vpx(&v));
            }
            if sample.contains(&k) {
                png(
                    &fixtures().join(format!("{name}_{}_{k}.png", ["track", "auto"][i])),
                    &px,
                    VW,
                    VH,
                );
            }
            enc[i].stdin.as_mut().unwrap().write_all(&px).unwrap();
        }
    }
    for e in enc {
        assert!(e.wait_with_output().unwrap().status.success());
    }
    eprintln!(
        "videos: {}",
        files
            .map(|f| fixtures().join(f).display().to_string())
            .join(", ")
    );
    let (kept, of) = on.unwrap_or((boxed, n));
    assert!(
        kept as f64 >= 0.5 * of as f64,
        "the tracker kept the subject on half the frames at least"
    );
    assert!(agree as f64 >= 0.5 * boxed as f64, "detector agreement");
}

/// Re-identification after a gap against a reference track of the subject (`X5_REF`, a `<clip>_track.csv` of
/// `subject` checked by eye; v627 or v606): the people of every 10th frame embedded; the subject's features of the
/// first 4 s (and those of the people within 30° of it then) make a `Gallery` as the tracker's (the newest 24 a side);
/// from 8 s on, the subject and everyone else are scored by it. Then again with the subject's features alone (the
/// tracker's gallery while nobody has been seen next to the subject) and with the first 1 to 20 features of the people
/// next to it (plain cosine below 12). Prints the similarity distributions, how many pass the tracker's rules (the
/// subject's, the others'), how often the subject is the most similar person in the frame (what finding it again in a
/// crowd needs) and whom `find()` would take; for the tracker's gallery and the subject's features alone also with the
/// speed the subject needs to be each box (`Seen::speed`) since the gallery last saw it (frame 110 of the reference,
/// heading as it did then): who is beyond its reach, the rules and `find()` on similarities less its prior
/// (`Seen::prior`), and the ROC AUC of the subject against the others without and with it.
#[test]
fn reid() {
    let (Some(d), Ok(rp)) = (samples(), std::env::var("X5_REF")) else {
        eprintln!("X5_REF unset: skipped");
        return;
    };
    let rf = reference(&rp);
    let name = if rp.ends_with("v606_track.csv") {
        "v606"
    } else {
        "v627"
    };
    let g = gpu();
    let c = clip(&d, name);
    let n = c.n.min(rf.len());
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    let rig = Rig::new(&g, &c);
    let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
    let mut dec = Dec::new(&c, n);
    // per key frame: (frame, is the subject, feature, box)
    let mut v: Vec<(usize, bool, Emb, (Vec3, Vec2))> = vec![];
    for (k, &(rd, rs, _)) in rf.iter().enumerate().take(n) {
        assert!(dec.next());
        if k % 10 != 0 || rs <= 0.0 {
            continue;
        }
        rig.upload(&g, &dec.buf);
        rig.stitch(&g, &c.poses[k]);
        let ps: Vec<_> = ai
            .detect(&pano, &sphere(), 0.3)
            .into_iter()
            .filter(|x| x.cls == 0)
            .map(|x| x.on())
            .collect();
        let near = |b: &(Vec3, Vec2), a: f32| b.0.angle_between(rd).to_degrees() < a;
        for (b, e) in ps.iter().zip(ai.embed(&pano, &ps)) {
            if near(b, 4.0) {
                v.push((k, true, e, *b));
            } else if !near(b, 8.0) && (k >= 240 || near(b, 30.0)) {
                v.push((k, false, e, *b));
            }
        }
    }
    // the subject where the gallery last saw it, heading as over the half second before
    let (d0, _, h0) = rf[110];
    let w0 = (d0 - rf[95].0) / 15.0;
    let seen = Seen {
        d: d0,
        h: 2.0 * (h0 / 2.0).to_radians().tan(),
        w: w0 - w0.dot(d0) * d0,
        k: 110,
    };
    // the tracker's gallery, then one of the subject's features alone (nobody seen next to it: plain cosine), then
    // with the first features of the people seen next to it
    for nb in [usize::MAX, 0, 1, 2, 3, 4, 6, 8, 12, 16, 20] {
        let mut gal = Gallery::default();
        let (me, not) = v
            .iter()
            .filter(|x| x.0 < 120)
            .partition::<Vec<&(usize, bool, Emb, (Vec3, Vec2))>, _>(|x| x.1);
        let g0: Vec<_> = me.into_iter().chain(not.into_iter().take(nb)).collect();
        for x in &g0 {
            gal.add(x.2, x.1);
        }
        // (frame, is the subject, similarities, speed the subject needs, its prior: None beyond its reach)
        type Sc = (usize, bool, (f32, f32), f32, Option<f32>);
        let sc: Vec<Sc> = v
            .iter()
            .filter(|x| x.0 >= 240)
            .map(|x| {
                (
                    x.0,
                    x.1,
                    gal.score(&x.2),
                    seen.speed(x.3, x.0 as u32, c.fps),
                    seen.prior(x.3, x.0 as u32, c.fps),
                )
            })
            .collect();
        // the similarity to the subject under the prior; beyond the reach -2
        let pr = |x: &Sc| (x.4.map_or(-2.0, |q| x.2.0 - q), x.2.1);
        let n = g0.iter().filter(|x| x.1).count();
        eprintln!(
            "gallery: {n} subject and {} neighbour features from frames 0..120",
            g0.len() - n
        );
        for who in [true, false] {
            let mut m: Vec<f64> = sc
                .iter()
                .filter(|x| x.1 == who)
                .map(|x| x.2.0 as f64)
                .collect();
            let n = m.len();
            let (p50, p95, mx) = pct(&mut m);
            eprintln!(
                "  {}: {n} samples, similarity to the gallery: 5 % {:.3}, median {p50:.3}, 95 % {p95:.3}, 99 % {:.3}, \
                 max {mx:.3}",
                ["others", "subject"][who as usize],
                m[n / 20],
                m[(n * 99 / 100).min(n - 1)]
            );
        }
        let pass = |who: bool, f: &dyn Fn((f32, f32)) -> bool| {
            let t: Vec<_> = sc.iter().filter(|x| x.1 == who).collect();
            format!("{}/{}", t.iter().filter(|x| f(x.2)).count(), t.len())
        };
        let ss: &[f32] = match nb {
            0 => &[0.5, 0.55, 0.6, 0.63, 0.65, 0.68, 0.7],
            usize::MAX => &[0.45, 0.5, 0.55, 0.6, 0.65],
            _ => &[],
        };
        for (s, mg) in ss.iter().flat_map(|&s| [(s, 0.02), (s, 0.05)]) {
            let f = |(m, n): (f32, f32)| m >= s && m >= n + mg;
            eprintln!(
                "  rule me >= {s}, me >= not + {mg}: subject passes {}, others {}",
                pass(true, &f),
                pass(false, &f)
            );
        }
        // per key frame: is the subject the most similar person, and whom find() takes (the most similar of those the
        // tracker's rule passes; here of everyone on the sphere, not only the 8 nearest within reach)
        let ks: BTreeSet<usize> = sc.iter().map(|x| x.0).collect();
        let (mut top, mut sub, mut me, mut other) = (0, 0, 0, 0);
        let (mut mp, mut op) = (0, 0);
        for k in ks.iter() {
            let m = sc.iter().filter(|x| x.0 == *k);
            let s = m
                .clone()
                .filter(|x| x.1)
                .map(|x| x.2.0)
                .fold(-2f32, f32::max);
            sub += (s > -2.0) as usize;
            top += (s > -2.0 && m.clone().filter(|x| !x.1).all(|x| x.2.0 < s)) as usize;
            match m
                .clone()
                .filter(|x| gal.is(x.2))
                .max_by(|a, b| a.2.0.total_cmp(&b.2.0))
            {
                Some(x) if x.1 => me += 1,
                Some(_) => other += 1,
                None => {}
            }
            match m
                .filter(|x| gal.is(pr(x)))
                .max_by(|a, b| pr(a).0.total_cmp(&pr(b).0))
            {
                Some(x) if x.1 => mp += 1,
                Some(_) => op += 1,
                None => {}
            }
        }
        eprintln!(
            "  the tracker's rule (threshold {}): subject passes {}, others {}; taken for someone seen next to it: \
             subject {}, others {}; the subject is the most similar person on {top}/{sub} key frames from 240 on; \
             find() would take the subject on {me}, someone else on {other} of {}",
            gal.same(),
            pass(true, &|s| gal.is(s)),
            pass(false, &|s| gal.is(s)),
            pass(true, &|s| gal.other(s)),
            pass(false, &|s| gal.other(s)),
            ks.len()
        );
        if nb == usize::MAX || nb == 0 {
            let cnt = |who: bool, f: &dyn Fn(&Sc) -> bool| {
                let t: Vec<_> = sc.iter().filter(|x| x.1 == who).collect();
                format!("{}/{}", t.iter().filter(|x| f(x)).count(), t.len())
            };
            // the chance that the subject scores above someone else
            let auc = |f: &dyn Fn(&Sc) -> f32| {
                let (a, b): (Vec<&Sc>, Vec<&Sc>) = sc.iter().partition(|x| x.1);
                let w: f64 = a
                    .iter()
                    .flat_map(|x| {
                        b.iter()
                            .map(move |y| f(x).total_cmp(&f(y)) as i8 as f64 / 2.0 + 0.5)
                    })
                    .sum();
                w / (a.len() * b.len()).max(1) as f64
            };
            let mut u: Vec<f64> = sc.iter().filter(|x| x.1).map(|x| x.3 as f64).collect();
            let (u50, _, umx) = pct(&mut u);
            eprintln!(
                "  with the speed the subject needs since frame 110: the subject median {u50:.3}, max {umx:.3} heights \
                 a second; beyond its reach subject {}, others {}; the rule passes subject {}, others {}; find() would \
                 take the subject on {mp}, someone else on {op}; ROC AUC subject against others {:.4} by similarity, \
                 {:.4} with the prior",
                cnt(true, &|x| x.4.is_none()),
                cnt(false, &|x| x.4.is_none()),
                cnt(true, &|x| gal.is(pr(x))),
                cnt(false, &|x| gal.is(pr(x))),
                auc(&|x| x.2.0),
                auc(&|x| pr(x).0)
            );
        }
    }
}

/// Same person against different people over everyone in a clip, without a reference (`X5_SEP`, default v627 and
/// v606): the people (score ≥ 0.3) of every frame over `sphere()` are linked into tracklets, a box taking over the
/// tracklet of the previous frame's box it overlaps most (IoU ≥ 0.5, one to one); a box overlapping another person's
/// ends its tracklet (which is which is unsure then, and its crop holds both). Every 5th frame's tracked people are
/// embedded. Same person: two features of one tracklet at least 2 s apart; different people: features of two
/// tracklets seen on one frame, whenever taken. Prints both similarity distributions, the ROC AUC, the share of
/// same-person pairs above the 1 % false-match threshold, and re-acquisition among people: how often a tracklet's
/// feature is more like its features from 2 s and more before than every other person on that frame is (rank 1).
#[test]
fn sep() {
    let Some(d) = samples() else { return };
    let names = std::env::var("X5_SEP").unwrap_or_else(|_| "v627,v606".into());
    let g = gpu();
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    let st = sphere();
    for name in names.split(',') {
        let name: &'static str = match name {
            "v635" => "v635",
            "v606" => "v606",
            _ => "v627",
        };
        let c = clip(&d, name);
        let gap = (2.0 * c.fps).round() as usize;
        let rig = Rig::new(&g, &c);
        let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
        let mut dec = Dec::new(&c, c.n);
        // open tracklets (id, box); features (frame, tracklet, feature); tracklet pairs seen on one frame
        let (mut open, mut next) = (Vec::<(usize, (Vec3, Vec2))>::new(), 0);
        let (mut fs, mut co) = (Vec::<(usize, usize, Emb)>::new(), HashSet::new());
        for k in 0..c.n {
            assert!(dec.next());
            rig.upload(&g, &dec.buf);
            rig.stitch(&g, &c.poses[k]);
            let ps: Vec<(Vec3, Vec2)> = ai
                .detect(&pano, &st, 0.3)
                .into_iter()
                .filter(|x| x.cls == 0)
                .map(|x| x.on())
                .collect();
            let mut m: Vec<(f32, usize, usize)> = open
                .iter()
                .enumerate()
                .flat_map(|(a, o)| {
                    ps.iter()
                        .enumerate()
                        .map(move |(i, b)| (iou(o.1, *b, false), a, i))
                })
                .filter(|x| x.0 >= 0.5)
                .collect();
            m.sort_by(|a, b| b.0.total_cmp(&a.0));
            let (mut ids, mut used) = (vec![None; ps.len()], vec![false; open.len()]);
            for (_, a, i) in m {
                if !used[a] && ids[i].is_none() {
                    (used[a], ids[i]) = (true, Some(open[a].0));
                }
            }
            open = (0..ps.len())
                .filter(|&i| (0..ps.len()).all(|j| j == i || iou(ps[i], ps[j], true) < 0.1))
                .map(|i| {
                    let id = ids[i].unwrap_or_else(|| {
                        next += 1;
                        next - 1
                    });
                    (id, ps[i])
                })
                .collect();
            for (i, a) in open.iter().enumerate() {
                for b in &open[i + 1..] {
                    co.insert((a.0.min(b.0), a.0.max(b.0)));
                }
            }
            if k % 5 == 0 && !open.is_empty() {
                let b: Vec<_> = open.iter().map(|o| o.1).collect();
                for (o, e) in open.iter().zip(ai.embed(&pano, &b)) {
                    fs.push((k, o.0, e));
                }
            }
        }
        drop(dec);
        let (mut same, mut diff) = (vec![], vec![]);
        for (i, a) in fs.iter().enumerate() {
            for b in &fs[i + 1..] {
                if a.1 == b.1 {
                    if a.0.abs_diff(b.0) >= gap {
                        same.push(sim(&a.2, &b.2) as f64);
                    }
                } else if co.contains(&(a.1.min(b.1), a.1.max(b.1))) {
                    diff.push(sim(&a.2, &b.2) as f64);
                }
            }
        }
        same.sort_by(f64::total_cmp);
        diff.sort_by(f64::total_cmp);
        // per mille
        let at = |v: &[f64], p: usize| v[(v.len() * p / 1000).min(v.len() - 1)];
        // the chance that a same-person pair is more similar than a different-people one
        let mut all: Vec<(f64, bool)> = same.iter().map(|&s| (s, true)).collect();
        all.extend(diff.iter().map(|&s| (s, false)));
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut below, mut sum) = (0u64, 0u64);
        for x in &all {
            if x.1 {
                sum += below;
            } else {
                below += 1;
            }
        }
        let auc = sum as f64 / (same.len() * diff.len()).max(1) as f64;
        let t1 = at(&diff, 990);
        let tpr = same.iter().filter(|&&s| s > t1).count() as f64 / same.len().max(1) as f64;
        let (mut hit, mut tot, mut crowd) = (0, 0, 0);
        for q in &fs {
            let gal: Vec<&Emb> = fs
                .iter()
                .filter(|x| x.1 == q.1 && x.0 + gap <= q.0)
                .map(|x| &x.2)
                .collect();
            let others: Vec<&Emb> = fs
                .iter()
                .filter(|x| x.0 == q.0 && x.1 != q.1)
                .map(|x| &x.2)
                .collect();
            if gal.is_empty() || others.is_empty() {
                continue;
            }
            let best = |e: &Emb| gal.iter().map(|g| sim(g, e)).fold(-1f32, f32::max);
            let me = best(&q.2);
            tot += 1;
            crowd += others.len();
            hit += others.iter().all(|o| best(o) < me) as usize;
        }
        let ids: HashSet<usize> = fs.iter().map(|x| x.1).collect();
        eprintln!(
            "{name}: {} tracklets embedded, {} features (every 5th frame); same person {gap} frames apart and more",
            ids.len(),
            fs.len()
        );
        eprintln!(
            "  same person: {} pairs, similarity 5 % {:.3}, median {:.3}, 95 % {:.3}",
            same.len(),
            at(&same, 50),
            at(&same, 500),
            at(&same, 950)
        );
        eprintln!(
            "  different people: {} pairs, median {:.3}, 99 % {:.3}, 99.9 % {:.3}, max {:.3}",
            diff.len(),
            at(&diff, 500),
            t1,
            at(&diff, 999),
            diff.last().copied().unwrap_or(0.0)
        );
        eprintln!(
            "  ROC AUC {auc:.4}; same-person pairs above the 1 % false-match threshold {t1:.3}: {:.1} %; rank 1 among \
             the people of the frame {hit}/{tot} ({:.1} %, {:.1} others each)",
            100.0 * tpr,
            100.0 * hit as f64 / tot.max(1) as f64,
            crowd as f64 / tot.max(1) as f64
        );
    }
}

/// Whether direction `w` lies in region `r`'s ellipse (the one `Blur` blurs, without its feather).
fn inside(r: &Region, w: Vec3) -> bool {
    let t = (r.size / 2.0).to_array().map(|a| a.to_radians().tan());
    Cam::new(r.dir, Vec3::Z, Vec2::new(t[0], t[1]))
        .px(w)
        .is_some_and(|p| p.length() <= 1.0)
}

/// Whether face `x` is covered by regions `r`: its centre and 8 points around the ellipse inscribed in its box each
/// lie in one.
fn covered(r: &[Region], x: &Region) -> bool {
    let t = (x.size / 2.0).to_array().map(|a| a.to_radians().tan());
    let c = Cam::new(x.dir, Vec3::Z, Vec2::new(t[0], t[1]));
    std::iter::once(x.dir)
        .chain((0..8).map(|i| c.ray(Vec2::from_angle(i as f32 * std::f32::consts::FRAC_PI_4))))
        .all(|w| r.iter().any(|r| inside(r, w)))
}

/// Privacy blur on the sample clips (`X5_PRIVACY` = comma-separated clip names, default all three; `X5_FRAMES`
/// caps the frames): faces on every frame (score ≥ 0.6, all of them in `<clip>_faces.txt`); the blur regions are
/// built from the detections of every 3rd frame alone and checked against all frames: flicker = a face the detector
/// saw on a frame that the regions do not cover (`covered`; its centre outside every region, counted too). The raw
/// detector's flicker, for comparison: frames inside a track of every-frame detections where the detector missed it.
/// Fails if more than 1 % of the confident faces (≥ 0.8) are not covered.
/// Writes blurred equirects of four frames per clip (`<clip>_blur_<k>.png`, and `_orig_`) with the regions outlined
/// in the original. macOS: faces through Core ML, `X5_YUNET=burn` for burn's YuNet.
#[test]
fn privacy() {
    let Some(d) = samples() else { return };
    let names = std::env::var("X5_PRIVACY").unwrap_or_else(|_| "v635,v606,v627".into());
    let cap: usize = std::env::var("X5_FRAMES").map_or(usize::MAX, |v| v.parse().unwrap());
    let g = gpu();
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    // macOS: Core ML's YuNet, or X5_YUNET=burn for burn's
    #[cfg(target_os = "macos")]
    {
        ai.coreml = std::env::var("X5_YUNET").map_or(true, |v| v != "burn");
    }
    let ft = face_tiles();
    let mut blur = Blur::new(&g.dev, wgpu::TextureFormat::Rgba8Unorm);
    for name in names.split(',') {
        let name: &'static str = match name {
            "v635" => "v635",
            "v606" => "v606",
            _ => "v627",
        };
        let c = clip(&d, name);
        let n = c.n.min(cap);
        let rig = Rig::new(&g, &c);
        let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
        let out = g.dev.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: rig.st.out.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let (sv, ov) = (
            rig.st.out.create_view(&Default::default()),
            out.create_view(&Default::default()),
        );
        // pass 1: detections per frame; from the second frame on, the crops' and the network's own times (each
        // waited for), the first compiling the kernels
        let mut dec = Dec::new(&c, n);
        let (mut all, mut fms, mut crops, mut nets) = (vec![], vec![], vec![], vec![]);
        for k in 0..n {
            (ai.timing, ai.times) = (k > 0, Default::default());
            assert!(dec.next());
            rig.upload(&g, &dec.buf);
            rig.stitch(&g, &c.poses[k]);
            let _ = g.dev.poll(wgpu::PollType::wait_indefinitely());
            let t = Instant::now();
            all.push(ai.faces(&pano, &ft, 0.6).unwrap());
            fms.push(t.elapsed().as_secs_f64() * 1e3);
            if k > 0 {
                let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
                crops.push(ms(ai.times.crop));
                nets.push(ms(ai.times.det) / ai.times.n[2].max(1) as f64);
            }
        }
        ai.timing = false;
        // a model that won't load falls back to burn's quietly: the run says which it measured
        #[cfg(target_os = "macos")]
        {
            let on = ai.coreml_used();
            eprintln!("  YuNet: {}", if on { "Core ML" } else { "burn" });
            assert_eq!(on, ai.coreml, "Core ML asked for, burn ran");
        }
        drop(dec);
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(fixtures().join(format!("{name}_faces.txt"))).unwrap(),
        );
        for (k, v) in all.iter().enumerate() {
            for x in v {
                let (d, s) = (x.dir, x.size);
                writeln!(f, "{k} {} {} {} {} {} {}", d.x, d.y, d.z, s.x, s.y, x.score).unwrap();
            }
        }
        drop(f);
        // regions from every 3rd frame's faces
        let mut h = Hide::new();
        for (k, f) in all.iter().enumerate().step_by(3) {
            h.push(k, f.clone());
        }
        let reg = h.regions(n, c.fps);
        let conf: Vec<(usize, Region)> = (0..n)
            .flat_map(|k| {
                all[k]
                    .iter()
                    .filter(|x| x.score >= 0.8)
                    .map(move |x| (k, *x))
            })
            .collect();
        let off = conf.iter().filter(|(k, _)| k % 3 != 0).count();
        let centre = |k: usize, x: &Region| reg[k].iter().any(|r| inside(r, x.dir));
        let miss: Vec<_> = conf.iter().filter(|(k, x)| !covered(&reg[*k], x)).collect();
        let cmiss = conf.iter().filter(|(k, x)| !centre(*k, x)).count();
        // every face the detector saw (≥ 0.6), off the frames the regions come from
        let mut w = [0; 3];
        for k in (0..n).filter(|k| k % 3 != 0) {
            for x in &all[k] {
                w[0] += 1;
                w[1] += !covered(&reg[k], x) as usize;
                w[2] += !centre(k, x) as usize;
            }
        }
        // the raw detector: tracks over every frame's faces, frames inside one without its detection
        let mut raw = Hide::new();
        for (k, f) in all.iter().enumerate() {
            raw.push(k, f.clone());
        }
        let tr = raw.tracks(c.fps);
        let long = tr.iter().filter(|t| t.len() >= 10).count();
        let gaps: usize = tr
            .iter()
            .map(|t| t[t.len() - 1].0 + 1 - t[0].0 - t.len())
            .sum();
        let nf: usize = all.iter().map(|v| v.len()).sum();
        let nr: usize = reg.iter().map(|v| v.len()).sum();
        let (f50, f95, fmx) = pct(&mut fms);
        eprintln!(
            "{name}: {n} frames; faces (≥ 0.6) {nf} ({:.2} per frame, {} frames with one), {} tracks ({long} of 10 \
             frames or more); blur regions {nr} ({:.2} per frame)",
            nf as f64 / n as f64,
            all.iter().filter(|v| !v.is_empty()).count(),
            tr.len(),
            nr as f64 / n as f64
        );
        eprintln!(
            "  flicker: {} of {} confident faces (≥ 0.8) not covered, centre outside every region {cmiss} ({off} \
             of them on frames without detection); of all {} faces (≥ 0.6) on frames without detection {} not \
             covered, centre outside {}; raw detector: {gaps} missed frames inside its tracks",
            miss.len(),
            conf.len(),
            w[0],
            w[1],
            w[2]
        );
        for (k, x) in miss.iter().take(5) {
            let [y, p] = yaw_pitch(x.dir);
            eprintln!(
                "    frame {k}: face {:.2} at yaw {y:.1} pitch {p:.1}, {:.1}° × {:.1}°",
                x.score, x.size.x, x.size.y
            );
        }
        eprintln!(
            "  ms per frame, median / p95 / max: faces ({} tiles) {f50:.1} / {f95:.1} / {fmx:.1}",
            ft.len()
        );
        let (c50, _, _) = pct(&mut crops);
        let (y50, y95, _) = pct(&mut nets);
        eprintln!(
            "  of it (median, waited for): crops {c50:.1} ms per frame; YuNet and its read {y50:.1} ms (p95 {y95:.1}) \
             per run of up to 8 tiles of 640²"
        );
        assert!(
            miss.len() * 100 <= conf.len(),
            "{name}: {} of {} confident faces not covered",
            miss.len(),
            conf.len()
        );
        // pass 2: blurred frames where most faces are
        let mut ks: Vec<usize> = (0..n).collect();
        ks.sort_by_key(|&k| std::cmp::Reverse(reg[k].len()));
        let mut pick: Vec<usize> = vec![];
        for k in ks {
            if pick.len() < 4 && pick.iter().all(|p| p.abs_diff(k) > n / 8) {
                pick.push(k);
            }
        }
        let mut dec = Dec::new(&c, pick.iter().max().unwrap() + 1);
        for (k, reg) in reg.iter().enumerate().take(pick.iter().max().unwrap() + 1) {
            assert!(dec.next());
            if !pick.contains(&k) {
                continue;
            }
            rig.upload(&g, &dec.buf);
            rig.stitch(&g, &c.poses[k]);
            let mut enc = g.dev.create_command_encoder(&Default::default());
            blur.run(&g.dev, &g.q, &mut enc, &sv, &ov, reg);
            g.q.submit([enc.finish()]);
            let mut o = stitch::read_texture(&g.dev, &g.q, &rig.st.out);
            for r in reg {
                outline(&mut o, EW, EH, r.on(), [255, 40, 40], epx);
            }
            png(&fixtures().join(format!("{name}_orig_{k}.png")), &o, EW, EH);
            let b = stitch::read_texture(&g.dev, &g.q, &out);
            png(&fixtures().join(format!("{name}_blur_{k}.png")), &b, EW, EH);
            eprintln!("  frame {k}: {} regions blurred", reg.len());
        }
    }
}

/// Labelled frames; plates with their frames.
type Labels = (Vec<usize>, Vec<(usize, Region)>);

/// Plates of a clip labelled by eye (`<clip>_plates.txt` in `X5_FIXTURES`: `f <k>` for each labelled frame, every plate
/// on it as `p <k> <x> <y> <z> <w> <h>`, its direction and angular width and height); None without the file.
fn labels(name: &str) -> Option<Labels> {
    let s = std::fs::read_to_string(fixtures().join(format!("{name}_plates.txt"))).ok()?;
    let (mut f, mut p) = (vec![], vec![]);
    for l in s.lines() {
        let v: Vec<&str> = l.split_whitespace().collect();
        let x = |i: usize| v[i].parse::<f32>().unwrap();
        match v.first() {
            Some(&"f") => f.push(v[1].parse().unwrap()),
            Some(&"p") => p.push((
                v[1].parse().unwrap(),
                Region {
                    dir: Vec3::new(x(2), x(3), x(4)).normalize(),
                    size: Vec2::new(x(5), x(6)),
                    score: 1.0,
                },
            )),
            _ => {}
        }
    }
    Some((f, p))
}

/// Licence plates against plates labelled by eye (`labels`; `X5_PLATES` = comma-separated clip names, default all
/// three; clips without labels are skipped): lenses at full resolution, the equirect at the export's 7680 (a plate is
/// a few pixels wide), plates every 3rd frame (`plates`, score ≥ 0.1 as the export, or `X5_PMIN`) linked by `Hide`.
/// Prints per width bin (pixels at 7680) the labelled plates the regions leave uncovered (`covered`), the area of the
/// regions on labelled frames that hold no labelled plate, and the time per searched frame of the plate pass against
/// the face pass on the same equirect. Fails if more than 10 % of the labelled plates 30 px wide or more (where
/// characters start to read) are left uncovered. The labels hold none that wide yet, the widest is 27 px: until they
/// do this check passes on nothing, and only the printed bins measure anything.
/// macOS: faces through Core ML, `X5_YUNET=burn` for burn's YuNet.
#[test]
fn plates() {
    let Some(d) = samples() else { return };
    let names = std::env::var("X5_PLATES").unwrap_or_else(|_| "v635,v606,v627".into());
    let all: Vec<_> = (names.split(','))
        .filter_map(|n| match n {
            "v635" => Some("v635"),
            "v606" => Some("v606"),
            "v627" => Some("v627"),
            _ => None,
        })
        .filter_map(|n| match labels(n) {
            Some(l) => Some((n, l)),
            None => {
                eprintln!("{n}: no {n}_plates.txt in X5_FIXTURES: skipped");
                None
            }
        })
        .collect();
    if all.is_empty() {
        return;
    }
    const W: u32 = 7680;
    let ppd = W as f32 / 360.0;
    let bins = [0.0, 10.0, 20.0, 40.0, f32::MAX];
    let g = gpu();
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    // macOS: Core ML's YuNet, or X5_YUNET=burn for burn's
    #[cfg(target_os = "macos")]
    {
        ai.coreml = std::env::var("X5_YUNET").map_or(true, |v| v != "burn");
    }
    let ft = face_tiles();
    let pmin = std::env::var("X5_PMIN").map_or(0.1, |v| v.parse().unwrap());
    let mut big = [0; 2];
    for (name, (lf, lp)) in all {
        let c = clip(&d, name);
        let rig = Rig::sized(&g, &c, 3840, W);
        let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
        let mut dec = Dec::sized(&c, c.n, 3840);
        let (mut h, mut tp, mut tf, mut found) = (Hide::new(), vec![], vec![], 0);
        for k in 0..c.n {
            assert!(dec.next());
            if k % 3 != 0 {
                continue;
            }
            rig.upload(&g, &dec.buf);
            rig.stitch(&g, &c.poses[k]);
            let _ = g.dev.poll(wgpu::PollType::wait_indefinitely());
            let t = Instant::now();
            let r = ai.plates(&pano, pmin).unwrap();
            tp.push(t.elapsed().as_secs_f64() * 1e3);
            found += r.len();
            h.push(k, r);
            let t = Instant::now();
            ai.faces(&pano, &ft, 0.5).unwrap();
            tf.push(t.elapsed().as_secs_f64() * 1e3);
        }
        // a model that won't load falls back to burn's quietly: the run says which it measured
        #[cfg(target_os = "macos")]
        {
            let on = ai.coreml_used();
            eprintln!("  YuNet: {}", if on { "Core ML" } else { "burn" });
            assert_eq!(on, ai.coreml, "Core ML asked for, burn ran");
        }
        drop(dec);
        let reg = h.regions(c.n, c.fps);
        // per width bin: labelled, uncovered
        let mut b = [[0usize; 2]; 4];
        for (k, x) in lp.iter().filter(|p| p.0 < c.n) {
            let i = bins.windows(2).position(|w| x.size.x * ppd < w[1]).unwrap();
            let miss = !covered(&reg[*k], x);
            b[i][0] += 1;
            b[i][1] += miss as usize;
            if x.size.x * ppd >= 30.0 {
                big[0] += 1;
                big[1] += miss as usize;
            }
            if miss {
                let [y, p] = yaw_pitch(x.dir);
                eprintln!(
                    "  {name} frame {k}: plate at yaw {y:.1} pitch {p:.1}, {:.0} px wide, uncovered",
                    x.size.x * ppd
                );
            }
        }
        let lf: Vec<usize> = lf.into_iter().filter(|&k| k < c.n).collect();
        let fp: f32 = (lf.iter())
            .flat_map(|&k| reg[k].iter().map(move |r| (k, r)))
            .filter(|(k, r)| !lp.iter().any(|(j, x)| j == k && inside(r, x.dir)))
            .map(|(_, r)| std::f32::consts::FRAC_PI_4 * r.size.x * r.size.y)
            .fold(0.0, |a, x| a + x);
        let ((p50, p95, _), (f50, f95, _)) = (pct(&mut tp), pct(&mut tf));
        eprintln!(
            "{name}: {} frames searched, {found} plates, {} tracks; {} labelled frames, {} plates; uncovered per width \
             (px at {W}) <10 / 10-20 / 20-40 / 40+: {}; regions holding no labelled plate {:.2} deg² per labelled frame; \
             ms per searched frame, median / p95: plates {p50:.1} / {p95:.1}, faces {f50:.1} / {f95:.1}",
            tp.len(),
            h.tracks(c.fps).len(),
            lf.len(),
            lp.len(),
            b.map(|[n, m]| format!("{m} of {n}")).join(" / "),
            fp / lf.len().max(1) as f32,
        );
    }
    assert!(
        big[1] * 10 <= big[0],
        "{} of {} labelled plates 30 px wide or more uncovered",
        big[1],
        big[0]
    );
}

/// Training crops for the plate detector (`X5_PLATE_DUMP` = output folder; skipped without): the clips named in
/// `X5_DUMP_CLIPS` (comma-separated, in `X5_SAMPLES`), every `X5_DUMP_EVERY`th frame (default 60) stitched as
/// `plates` does, each plate crop as the detector sees it (`<clip>_<frame>_<i>.ppm`, RGB) with its candidate boxes of
/// score ≥ 0.05 in `crops.jsonl` (crop pixels, the crop's direction and tangent extent).
#[test]
fn plate_dump() {
    let (Some(d), Some(out)) = (
        samples(),
        std::env::var_os("X5_PLATE_DUMP").map(PathBuf::from),
    ) else {
        return;
    };
    std::fs::create_dir_all(&out).unwrap();
    let every: usize = std::env::var("X5_DUMP_EVERY").map_or(60, |v| v.parse().unwrap());
    let names = std::env::var("X5_DUMP_CLIPS").unwrap();
    let g = gpu();
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    let mut log = std::fs::File::create(out.join("crops.jsonl")).unwrap();
    for name in names.split(',') {
        let c = clip(&d, name.to_string().leak());
        let rig = Rig::sized(&g, &c, 3840, 7680);
        let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
        let mut dec = Dec::sized(&c, c.n, 3840);
        let mut n = 0;
        for k in 0..c.n {
            assert!(dec.next());
            if k % every != 0 {
                continue;
            }
            rig.upload(&g, &dec.buf);
            rig.stitch(&g, &c.poses[k]);
            let _ = g.dev.poll(wgpu::PollType::wait_indefinitely());
            for (i, (cam, bgr, b)) in ai
                .plate_inputs(&pano, 0.05)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                let file = format!("{name}_{k:05}_{i:02}.ppm");
                let s = ((bgr.len() / 3) as f64).sqrt() as usize;
                let mut ppm = format!("P6\n{s} {s}\n255\n").into_bytes();
                ppm.extend(bgr.chunks(3).flat_map(|p| [p[2], p[1], p[0]]));
                std::fs::write(out.join(&file), ppm).unwrap();
                let boxes: Vec<String> = (b.iter())
                    .map(|(b, sc)| {
                        format!("[{:.1},{:.1},{:.1},{:.1},{sc:.3}]", b[0], b[1], b[2], b[3])
                    })
                    .collect();
                writeln!(
                    log,
                    "{{\"file\":\"{file}\",\"clip\":\"{name}\",\"frame\":{k},\"dir\":[{:.5},{:.5},{:.5}],\"ext\":[{:.5},{:.5}],\
                     \"boxes\":[{}]}}",
                    cam.f.x,
                    cam.f.y,
                    cam.f.z,
                    cam.ext.x,
                    cam.ext.y,
                    boxes.join(",")
                )
                .unwrap();
                n += 1;
            }
        }
        eprintln!("{name}: {n} crops");
    }
}

/// Highlights and auto edit on the sample clips (`X5_HIGH`, default all three): every 10th frame the detector over
/// `sphere()` (auto framing and people), faces and a thumbnail (visual change); camera motion from the gyro, audio
/// from the AAC track. Prints the signals of each pick, checks the timeline with crate edit (`Timeline::check`, a
/// `Project` over it) and renders each highlight's first, middle and last frame through the timeline's view
/// (`<clip>_high_<i>_<k>.png`) to look at.
#[test]
fn highlights() {
    let Some(d) = samples() else { return };
    let names = std::env::var("X5_HIGH").unwrap_or_else(|_| "v635,v606,v627".into());
    let g = gpu();
    let mut ai = Ai::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    let (ft, st) = (face_tiles(), sphere());
    for name in names.split(',') {
        let name: &'static str = match name {
            "v635" => "v635",
            "v606" => "v606",
            _ => "v627",
        };
        let c = clip(&d, name);
        let m = media::mp4::Mp4::open(&c.path).unwrap();
        let vt = &m.tracks[0];
        let fps = edit::Rate {
            num: vt.timescale,
            den: vt.samples[0].dur,
        };
        let n = c.n;
        let dur = n as f64 * fps.t(1);
        let rig = Rig::new(&g, &c);
        let pano = ai.pano(&rig.st.out.create_view(&Default::default()));
        let mut sc = Scorer::new(dur);
        let mut fm = Framer::new(VW as f32 / VH as f32);
        let mut dec = Dec::new(&c, n);
        let t0 = Instant::now();
        for k in 0..n {
            assert!(dec.next());
            if k % 10 != 0 {
                continue;
            }
            rig.upload(&g, &dec.buf);
            rig.stitch(&g, &c.poses[k]);
            let t = fps.t(k as u64);
            let ds = ai.detect(&pano, &st, 0.3);
            let fs = ai.faces(&pano, &ft, 0.6).unwrap();
            sc.people(t, &ds, &fs);
            sc.frame(t, ai.thumb(&pano).unwrap());
            fm.push(k, ds);
        }
        drop(dec);
        let tr = Trailer::open(&c.path).unwrap();
        let ex = tr.frames(insv::EXPOSURE).unwrap();
        let tk = stab::Track::new(&tr.imu().unwrap(), stab::KP, stab::KI);
        sc.gyro(&tk, ex[0].t as f64 * 1e-6 - 0.5 * ex[0].time + stab::OFFSET);
        sc.audio(&c.path).unwrap();
        let total = (0.3 * dur).max(2.0);
        let hs = sc.pick(total);
        let sig = sc.signals();
        let score = sc.score();
        eprintln!(
            "{name}: {dur:.1} s analysed in {:.1} s; {} highlights for {total:.1} s:",
            t0.elapsed().as_secs_f64(),
            hs.len()
        );
        let mean = |a: f64, b: f64, f: &dyn Fn(usize) -> f32| {
            let r = (a / track::BIN) as usize..((b / track::BIN).ceil() as usize).min(sig.len());
            r.clone().map(f).sum::<f32>() / r.len().max(1) as f32
        };
        let names = ["motion", "change", "people", "audio"];
        eprintln!(
            "  clip mean: score {:.2}, {}",
            mean(0.0, dur, &|i| score[i]),
            (0..4)
                .map(|s| format!("{} {:.2}", names[s], mean(0.0, dur, &|i| sig[i][s])))
                .collect::<Vec<_>>()
                .join(", ")
        );
        for h in &hs {
            eprintln!(
                "  {:5.1}–{:5.1} s: score {:.2}, {}",
                h.a,
                h.b,
                h.score,
                (0..4)
                    .map(|s| format!("{} {:.2}", names[s], mean(h.a, h.b, &|i| sig[i][s])))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let _ = Sig::Motion;
        let path = fm.path(n, c.fps);
        let tl = track::timeline(
            &c.path.display().to_string(),
            dur,
            fps,
            fps.num as f64 / fps.den as f64,
            &hs,
            &path,
        )
        .unwrap();
        tl.check().unwrap();
        let tl = edit::Project::with(tl).unwrap().tl().clone();
        let total_out = tl.dur();
        eprintln!(
            "  timeline: {} clips, {} frames ({total_out:.2} s), {} view keys, {} decode runs; checks pass",
            tl.clips.len(),
            tl.len(),
            tl.clips.iter().map(|c| c.view.len()).sum::<usize>(),
            tl.segs().len()
        );
        assert_eq!(tl.clips.len(), hs.len());
        assert!((total_out - hs.iter().map(|h| h.b - h.a).sum::<f64>()).abs() < 0.1);
        // the frames to look at: each clip's first, middle and last output frame, rendered from the source frame the
        // timeline shows with its view
        let mut want: Vec<(u64, usize, usize)> = vec![];
        for i in 0..tl.clips.len() {
            let r = tl.frames(i);
            for o in [r.start, (r.start + r.end) / 2, r.end - 1] {
                let e = tl.frame(o).unwrap();
                want.push((
                    o,
                    i,
                    ((e.src * fps.num as f64 / fps.den as f64).round() as usize).min(n - 1),
                ));
            }
        }
        let last = want.iter().map(|w| w.2).max().unwrap();
        let mut dec = Dec::new(&c, last + 1);
        for k in 0..=last {
            assert!(dec.next());
            for &(o, i, _) in want.iter().filter(|w| w.2 == k) {
                rig.upload(&g, &dec.buf);
                let v = tl.frame(o).unwrap().view;
                let [yaw, pitch, roll] = v.ypr();
                let view = View {
                    yaw,
                    pitch,
                    roll,
                    fov: v.fov,
                    ..Default::default()
                };
                let px = rig.view(&g, &c.poses[k], &view);
                png(
                    &fixtures().join(format!("{name}_high_{i}_{k}.png")),
                    &px,
                    VW,
                    VH,
                );
            }
        }
    }
}
