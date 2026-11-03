//! GPU tests. The sample-footage ones read `X5_SAMPLES` (directory with `v635.insv`, `v635.lrv`) and `X5_LIVE` (a
//! live-preview capture, raw H.264) and skip when unset; their fixtures (NV12 frames made with the ffmpeg CLI as a test
//! tool, reference panoramas) go to `X5_FIXTURES` (default: the system temp dir). With `X5_OLDREF` = a build of the old `insta360-stitch` reference
//! (`oldref <insv> <lens0.rgba> <lens1.rgba> <side> <w> <out.rgba> [--raft]`), `vs_old` compares against it.

use glam::{Mat3, Quat, Vec3};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
    time::Instant,
};
use stitch::{
    Calib, Ctx, Flow, Fmt, Frame, Input, Mode, Opts, Pose, Preview, Raft, Src, Stitcher, View,
};

struct Gpu {
    inst: wgpu::Instance,
    ad: wgpu::Adapter,
    dev: wgpu::Device,
    q: wgpu::Queue,
}

/// A device opened like `video::Gpu`: every adapter feature (burn needs them, NV12 among them), the experimental
/// ones, all limits. One per test binary; the lock keeps GPU tests (and their timings) from overlapping.
fn gpu() -> std::sync::MutexGuard<'static, Gpu> {
    static G: OnceLock<Mutex<Gpu>> = OnceLock::new();
    G.get_or_init(|| Mutex::new(open()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn open() -> Gpu {
    {
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
        Gpu { inst, ad, dev, q }
    }
}

/// cubecl installs its own uncaptured-error handler (panics on any error) whenever burn attaches to a device.
/// `Raft::new` on a device with our handler installed replaces it; `stitch::attach` before installing ours keeps it:
/// `Raft::new` then reuses the attachment. Provoked error: a buffer mapped at creation with an unaligned size. Also
/// times the attach (what an app pays at start-up) and `Raft::new` (the window the app's re-install left open).
#[test]
fn errors() {
    let _l = gpu();
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ours = || {
        let h = hits.clone();
        std::sync::Arc::new(move |_: wgpu::Error| {
            h.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let bad = |d: &wgpu::Device| {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 3,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        })
    };
    let n = || hits.load(std::sync::atomic::Ordering::SeqCst);
    let g = open();
    g.dev.on_uncaptured_error(ours());
    bad(&g.dev);
    assert_eq!(n(), 1, "our handler");
    let t = Instant::now();
    Raft::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    eprintln!(
        "Raft::new without a prior attach: {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bad(&g.dev)));
    eprintln!(
        "Raft::new without attach: our handler replaced (cubecl's panicked: {})",
        r.is_err()
    );
    assert!(r.is_err() && n() == 1);
    let g = open();
    let t = Instant::now();
    stitch::attach(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    eprintln!("attach alone: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    g.dev.on_uncaptured_error(ours());
    for _ in 0..2 {
        let t = Instant::now();
        Raft::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
        eprintln!(
            "Raft::new on the attached device: {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    bad(&g.dev);
    eprintln!("attach, our handler, Raft::new twice: our handler kept");
    assert_eq!(n(), 2);
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
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("x5-stitch"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The stitching inputs of an `.insv` trailer: the V3 "extra info" block's record type 1 is the `ExtraMetadata`
/// protobuf (PROTOCOL.md §9.4/§9.5). Port of insta360-stitch's reader; the app's own lives in `media`.
#[derive(Default, Debug)]
struct Meta {
    offset: String,
    /// offset_v3: the string the camera also reports as MEDIA_OFFSET_V3, the live preview's calibration
    v3: String,
    crop: f32,
    shift: [f32; 2],
    blend: Option<f64>,
    /// DUAL_STREAM_TRACK_REVERSE: track 0 holds lens 1
    reverse: bool,
}

fn trailer(p: &Path) -> Meta {
    let d = std::fs::read(p).unwrap();
    let u = |b: &[u8]| u32::from_le_bytes(b[..4].try_into().unwrap()) as usize;
    let tail = &d[d.len() - 78..];
    assert!(&tail[46..] == b"8db42d694ccc418790edff439fe026bf" && u(&tail[42..]) == 3);
    let start = d.len() - u(&tail[38..]);
    let n = u(&tail[2..]);
    let idx = &d[d.len() - 78 - n..d.len() - 78];
    let e = idx.chunks(10).find(|e| e[0] == 1).expect("metadata record");
    let md = &d[start + u(&e[6..])..][..u(&e[2..])];
    let mut m = Meta::default();
    let mut off = [None; 3];
    for (tag, v) in fields(md) {
        match (tag, v) {
            (5, F::Bytes(b)) => off[2] = Some(b),
            (54, F::Bytes(b)) => off[1] = Some(b),
            (111, F::Bytes(b)) => off[0] = Some(b),
            (128, F::Int(a)) if a as i32 > 0 => m.blend = Some(a as i32 as f64),
            (131, F::Int(t)) => m.reverse = t == 4,
            (27, F::Bytes(w)) => {
                for (t, v) in fields(w) {
                    if let F::Int(x) = v {
                        match t {
                            3 => m.crop = x as f32,
                            5 => m.shift[0] = x as i32 as f32,
                            6 => m.shift[1] = x as i32 as f32,
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    m.v3 = String::from_utf8_lossy(off[1].unwrap_or_default()).into_owned();
    let o = off
        .into_iter()
        .flatten()
        .find(|b| !b.is_empty())
        .expect("offset");
    m.offset = String::from_utf8_lossy(o).into_owned();
    m
}

enum F<'a> {
    Int(u64),
    Bytes(&'a [u8]),
    Other,
}

fn fields(mut b: &[u8]) -> impl Iterator<Item = (u64, F<'_>)> {
    fn varint(b: &mut &[u8]) -> Option<u64> {
        let mut v = 0;
        for s in (0..64).step_by(7) {
            let (&c, r) = b.split_first()?;
            *b = r;
            v |= ((c & 0x7f) as u64) << s;
            if c < 0x80 {
                return Some(v);
            }
        }
        None
    }
    std::iter::from_fn(move || {
        let k = varint(&mut b)?;
        let f = match k & 7 {
            0 => F::Int(varint(&mut b)?),
            1 | 5 => {
                b = b.get(if k & 7 == 1 { 8 } else { 4 }..)?;
                F::Other
            }
            2 => {
                let n = varint(&mut b)? as usize;
                let v = b.get(..n)?;
                b = &b[n..];
                F::Bytes(v)
            }
            _ => return None,
        };
        Some((k >> 3, f))
    })
}

const SIDE: usize = 3840;

/// Frame `n` of video track `t` as NV12, made once with the ffmpeg CLI (test tool) into the fixtures.
fn frame(file: &Path, t: usize, n: usize) -> Option<Vec<u8>> {
    let name = file.file_name()?.to_string_lossy();
    let p = fixtures().join(format!("{name}_t{t}_{n}.nv12"));
    if !p.exists() {
        let tmp = part(&p);
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(file)
            .args([
                "-map",
                &format!("0:v:{t}"),
                "-vf",
                &format!("select=eq(n\\,{n})"),
            ])
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "nv12"])
            .arg(&tmp)
            .status()
            .is_ok_and(|s| s.success());
        if !ok || std::fs::rename(&tmp, &p).is_err() {
            _ = std::fs::remove_file(&tmp);
            eprintln!("ffmpeg could not make {}: skipped", p.display());
            return None;
        }
    }
    std::fs::read(&p).ok()
}

/// Where this thread writes fixture `p` before renaming it into place: tests run in parallel, and one that finds
/// `p` must never read it half written.
fn part(p: &Path) -> PathBuf {
    let id = format!(
        "{}-{:?}.part",
        std::process::id(),
        std::thread::current().id()
    );
    p.with_extension(id)
}

/// Frame 0 of v635: trailer, calibration, both lenses as NV12 ([lens 0, lens 1]).
struct Clip {
    meta: Meta,
    cal: Calib,
    fr: Frame,
    nv: [Vec<u8>; 2],
}

fn clip() -> Option<Clip> {
    let insv = samples()?.join("v635.insv");
    let meta = trailer(&insv);
    let cal = Calib::parse(&meta.offset).unwrap();
    let mut nv = [frame(&insv, 0, 0)?, frame(&insv, 1, 0)?];
    if meta.reverse {
        nv.swap(0, 1);
    }
    let fr = Frame {
        w: SIDE as u32,
        h: SIDE as u32,
        crop: meta.crop,
        shift: meta.shift,
    };
    Some(Clip { nv, meta, cal, fr })
}

/// The X5's camera framing (lens 1 at the centre) for reversed tracks.
fn framing(m: &Meta) -> Pose {
    Pose::euler(0.0, 0.0, if m.reverse { 180.0 } else { 0.0 })
}

/// NV12 -> RGBA exactly as `stitch.wgsl` samples it at a luma texel centre: full-range BT.709, chroma sited left,
/// bilinear with clamped edges.
fn rgba(nv: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (yp, cp) = nv.split_at(w * h);
    let (cw, ch) = (w as isize / 2, h as isize / 2);
    let at = |x: isize, y: isize, k: usize| {
        cp[((y.clamp(0, ch - 1) * cw + x.clamp(0, cw - 1)) * 2) as usize + k] as f32
    };
    let mut o = vec![255u8; w * h * 4];
    for (i, p) in o.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (x, y) = (i % w, i / w);
        let (cx, cy) = (x as f32 / 2.0, y as f32 / 2.0 - 0.25);
        let (x0, y0) = (cx.floor(), cy.floor());
        let (fx, fy) = (cx - x0, cy - y0);
        let (x0, y0) = (x0 as isize, y0 as isize);
        let c = [0, 1].map(|k| {
            let t = at(x0, y0, k) + fx * (at(x0 + 1, y0, k) - at(x0, y0, k));
            let b = at(x0, y0 + 1, k) + fx * (at(x0 + 1, y0 + 1, k) - at(x0, y0 + 1, k));
            (t + fy * (b - t) - 128.0) / 255.0
        });
        let l = yp[i] as f32 / 255.0;
        let v = [
            l + 1.5748 * c[1],
            l - 0.187324 * c[0] - 0.468124 * c[1],
            l + 1.8556 * c[0],
        ];
        for k in 0..3 {
            p[k] = (v[k].clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    o
}

fn texture(g: &Gpu, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
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
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn write(
    g: &Gpu,
    t: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
    data: &[u8],
    w: u32,
    h: u32,
    bpp: u32,
) {
    g.q.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: t,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * bpp),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

/// An NV12 picture holding `nv` (Y plane, then interleaved Cb Cr), bound like a decoder output slot.
fn upload(g: &Gpu, cx: &Ctx, nv: &[u8], w: u32, h: u32) -> Src {
    cx.rect(&slot(g, nv, w, h), [0, 0, w, h])
}

/// A decoder output slot's plane views holding NV12 (`b` bytes per sample, 1) or P010 (2) samples `px`: those of
/// one NV12 / P010 texture as the Vulkan decoder writes it, or of an R8 / RG8 (R16 / RG16) texture pair as the
/// VideoToolbox one does (wgpu's Metal backend has no NV12 / P010 textures).
fn picture(g: &Gpu, px: &[u8], w: u32, h: u32, b: u32) -> [wgpu::TextureView; 2] {
    use wgpu::TextureFormat as F;
    let n = (w * h * b) as usize;
    if cfg!(target_os = "linux") {
        let t = g.dev.create_texture(&wgpu::TextureDescriptor {
            label: Some("nv12"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if b == 2 { F::P010 } else { F::NV12 },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        write(g, &t, wgpu::TextureAspect::Plane0, &px[..n], w, h, b);
        write(
            g,
            &t,
            wgpu::TextureAspect::Plane1,
            &px[n..],
            w / 2,
            h / 2,
            2 * b,
        );
        return stitch::views(&t);
    }
    let (fy, fc) = match b {
        2 => (F::R16Unorm, F::Rg16Unorm),
        _ => (F::R8Unorm, F::Rg8Unorm),
    };
    let y = texture(g, w, h, fy);
    let uv = texture(g, w / 2, h / 2, fc);
    write(g, &y, wgpu::TextureAspect::All, &px[..n], w, h, b);
    write(
        g,
        &uv,
        wgpu::TextureAspect::All,
        &px[n..],
        w / 2,
        h / 2,
        2 * b,
    );
    [y, uv].map(|t| t.create_view(&Default::default()))
}

fn slot(g: &Gpu, nv: &[u8], w: u32, h: u32) -> [wgpu::TextureView; 2] {
    picture(g, nv, w, h, 1)
}

/// The same picture as separate R8 / RG8 textures (`Ctx::planes`).
fn planes(g: &Gpu, cx: &Ctx, nv: &[u8], w: u32, h: u32) -> Src {
    let n = (w * h) as usize;
    let y = texture(g, w, h, wgpu::TextureFormat::R8Unorm);
    let uv = texture(g, w / 2, h / 2, wgpu::TextureFormat::Rg8Unorm);
    write(g, &y, wgpu::TextureAspect::All, &nv[..n], w, h, 1);
    write(g, &uv, wgpu::TextureAspect::All, &nv[n..], w / 2, h / 2, 2);
    cx.planes(
        &y.create_view(&Default::default()),
        &uv.create_view(&Default::default()),
    )
}

/// PSNR over RGB of two RGBA8 images.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let (mut s, mut n) = (0f64, 0usize);
    for (p, q) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        for k in 0..3 {
            let d = p[k] as f64 - q[k] as f64;
            s += d * d;
        }
        n += 3;
    }
    10.0 * (255.0 * 255.0 / (s / n as f64).max(1e-12)).log10()
}

/// Share of RGB values that differ by more than `t` levels.
fn over(a: &[u8], b: &[u8], t: u8) -> f64 {
    let n = a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .flat_map(|(p, q)| (0..3).map(move |k| p[k].abs_diff(q[k]) > t))
        .filter(|&d| d)
        .count();
    n as f64 / (a.len() / 4 * 3) as f64
}

/// Wall time per frame of `n` back-to-back recordings in one submit, waited for (after one warm-up).
fn time(g: &Gpu, n: u32, mut rec: impl FnMut(&mut wgpu::CommandEncoder)) -> f64 {
    let mut enc = g.dev.create_command_encoder(&Default::default());
    rec(&mut enc);
    g.q.submit([enc.finish()]);
    g.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let t = Instant::now();
    let mut enc = g.dev.create_command_encoder(&Default::default());
    for _ in 0..n {
        rec(&mut enc);
    }
    g.q.submit([enc.finish()]);
    g.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    t.elapsed().as_secs_f64() * 1e3 / n as f64
}

fn run(g: &Gpu, st: &Stitcher, src: [&Src; 2]) -> Vec<u8> {
    let mut enc = g.dev.create_command_encoder(&Default::default());
    st.encode(&mut enc, src);
    if let Some(y) = &st.yuv {
        y.encode(&mut enc);
    }
    g.q.submit([enc.finish()]);
    stitch::read_texture(&g.dev, &g.q, &st.out)
}

/// R'G'B' of a texture read back with `read_texture`: `Rgba8Unorm` or `Rgba16Float`.
fn rgb_of(px: &[u8], wide: bool) -> Vec<[f64; 3]> {
    let f16 = |b: &[u8]| {
        let v = u16::from_le_bytes([b[0], b[1]]);
        let (e, m) = ((v >> 10 & 31) as i32, (v & 1023) as f64);
        let x = if e == 0 {
            m * 2f64.powi(-24)
        } else {
            (1.0 + m / 1024.0) * 2f64.powi(e - 15)
        };
        if v >> 15 == 1 { -x } else { x }
    };
    if wide {
        px.chunks(8)
            .map(|p| [0, 1, 2].map(|k| f16(&p[2 * k..])))
            .collect()
    } else {
        px.chunks(4)
            .map(|p| [0, 1, 2].map(|k| p[k] as f64 / 255.0))
            .collect()
    }
}

/// YUV 4:2:0 samples (Y, then interleaved Cb Cr) of an R'G'B' image as `stitch::Yuv` writes them (`post.wgsl`):
/// BT.709 limited range, 8 or 10 bits, chroma filtered at its site; columns wrap (equirect) or repeat the edge.
fn yuv_of(px: &[[f64; 3]], w: usize, h: usize, f: Fmt, wrap: bool) -> Vec<u16> {
    let (s, c0) = if f.deep {
        ([64.0, 876.0, 896.0], 512.0)
    } else {
        ([16.0, 219.0, 224.0], 128.0)
    };
    let at = |x: isize, y: isize| {
        let x = if wrap {
            x.rem_euclid(w as isize)
        } else {
            x.clamp(0, w as isize - 1)
        };
        px[y.clamp(0, h as isize - 1) as usize * w + x as usize].map(|v| v.clamp(0.0, 1.0))
    };
    let l = |p: [f64; 3]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let mut o: Vec<u16> = (0..w * h)
        .map(|i| (s[0] + s[1] * l(at((i % w) as isize, (i / w) as isize))).round() as u16)
        .collect();
    let hw = if f.chroma == 1 {
        [0.0, 0.5, 0.5]
    } else {
        [0.25, 0.5, 0.25]
    };
    let vw = if f.chroma == 2 {
        [0.25, 0.5, 0.25]
    } else {
        [0.0, 0.5, 0.5]
    };
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            let mut v = [0.0; 3];
            for j in 0..9 {
                let p = at((2 * x + j % 3) as isize - 1, (2 * y + j / 3) as isize - 1);
                for k in 0..3 {
                    v[k] += hw[j % 3] * vw[j / 3] * p[k];
                }
            }
            let lv = l(v);
            o.extend(
                [(v[2] - lv) / 1.8556, (v[0] - lv) / 1.5748]
                    .map(|c| (c0 + s[2] * c).round() as u16),
            );
        }
    }
    o
}

/// The samples of a `stitch::Yuv` buffer.
fn samples_of(b: &[u8], deep: bool) -> Vec<u16> {
    if deep {
        b.chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) >> 6)
            .collect()
    } else {
        b.iter().map(|&v| v as u16).collect()
    }
}

/// Max |difference| in the Y plane and in the chroma plane.
fn yuv_diff(a: &[u16], b: &[u16], w: usize, h: usize) -> (u16, u16) {
    let d = |r: std::ops::Range<usize>| {
        a[r.clone()]
            .iter()
            .zip(&b[r])
            .map(|(p, q)| p.abs_diff(*q))
            .max()
    };
    (d(0..w * h).unwrap(), d(w * h..w * h * 3 / 2).unwrap())
}

// synthetic X5-like calibration (no real one in the repo), lens textures 64²
const CAL: &str = "2_2.000000_4270.000_4271.000_2690.000_2685.000_-0.500_0.100_90.000_0.000000_0.000000_0.000000_0.20000000_1.50000000_-0.80000000_-3.50000000_0.00000000_-0.00250000_-0.00150000_0.00500000_0.01000000_-0.00200000_-0.00250000_0.00020000_-0.02500000_10752_5376_113_2.000000_4272.000_4273.000_8070.000_2680.000_0.500_-0.100_90.500_0.000500_0.000050_-0.031500_0.24000000_1.45000000_-0.90000000_-3.20000000_0.00000000_-0.00100000_-0.00010000_0.01300000_-0.00300000_-0.00700000_0.00200000_-0.01200000_0.01700000_10752_5376_113_394240";

/// Without footage: NV12 textures, plane textures and the two halves of one padded side-by-side texture give the same
/// panorama and preview, the NV12 output matches the RGBA one, a zero rolling-shutter term changes nothing and a
/// non-zero one does, every preview mode draws, and the preview of an equirect matches its CPU reprojection.
#[test]
fn synthetic() {
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let cal = Calib::parse(CAL).unwrap();
    let fr = Frame {
        w: 64,
        h: 64,
        crop: 0.0,
        shift: [0.0; 2],
    };
    let mut s = 0x2545f491u32;
    let nv: [Vec<u8>; 2] = std::array::from_fn(|_| {
        (0..64 * 96)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((i % 64) as u32 * 2 + (s >> 26)) as u8
            })
            .collect()
    });
    let a = nv.each_ref().map(|v| upload(&g, &cx, v, 64, 64));
    let b = nv.each_ref().map(|v| planes(&g, &cx, v, 64, 64));
    let n8 = Fmt {
        deep: false,
        chroma: 1,
    };
    let o = Opts {
        yuv: Some(n8),
        ..Default::default()
    };
    let st = Stitcher::new(&cx, &cal, fr, 256, 128, o).unwrap();
    let pa = run(&g, &st, [&a[0], &a[1]]);
    let nv12 = stitch::read_buffer(&g.dev, &g.q, &st.yuv.as_ref().unwrap().buf);
    let st = Stitcher::new(&cx, &cal, fr, 256, 128, o).unwrap();
    let pb = run(&g, &st, [&b[0], &b[1]]);
    assert_eq!(pa, pb, "NV12 texture vs plane textures");
    assert!(pa.chunks(4).any(|p| p[..3] != [0, 0, 0]), "empty panorama");
    // both lenses side by side in a decoder-like slot: picture at (4, 2) of a 136 × 68 texture, noise around them
    let (tw, at) = (136, [4, 2]);
    let mut sbs: Vec<u8> = (0..tw * 68 * 3 / 2)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect();
    for (k, v) in nv.iter().enumerate() {
        for y in 0..64 {
            let x = at[0] + 64 * k;
            let d = (at[1] + y) * tw + x;
            sbs[d..d + 64].copy_from_slice(&v[64 * y..][..64]);
            if y < 32 {
                let d = tw * 68 + (at[1] / 2 + y) * tw + x; // chroma rows: tw / 2 pairs
                sbs[d..d + 64].copy_from_slice(&v[64 * 64 + 64 * y..][..64]);
            }
        }
    }
    let t = slot(&g, &sbs, tw as u32, 68);
    let r = [0, 1].map(|k| cx.rect(&t, [at[0] as u32 + 64 * k, at[1] as u32, 64, 64]));
    // a window crop that cuts the image circle, so the lenses' edge texels count
    let fc = Frame { crop: 4600.0, ..fr };
    let [ta, tr] = [&a, &r].map(|src| {
        let st = Stitcher::new(&cx, &cal, fc, 256, 128, o).unwrap();
        let px = run(&g, &st, [&src[0], &src[1]]);
        (
            px,
            stitch::read_buffer(&g.dev, &g.q, &st.yuv.as_ref().unwrap().buf),
        )
    });
    assert!(ta == tr, "rects of one texture vs two textures");
    let (dy, dc) = yuv_diff(
        &samples_of(&nv12, false),
        &yuv_of(&rgb_of(&pa, false), 256, 128, n8, true),
        256,
        128,
    );
    eprintln!("NV12 out vs RGBA out: max |ΔY| {dy}, |ΔC| {dc}");
    assert!(dy <= 1 && dc <= 1);
    // P010 from an RGBA16F panorama, and every chroma siting
    for (deep, chroma) in [(true, 0), (true, 1), (true, 2), (false, 0), (false, 2)] {
        let f = Fmt { deep, chroma };
        let o = Opts {
            yuv: Some(f),
            wide: deep,
            ..o
        };
        let st = Stitcher::new(&cx, &cal, fr, 256, 128, o).unwrap();
        let px = rgb_of(&run(&g, &st, [&a[0], &a[1]]), deep);
        let b = stitch::read_buffer(&g.dev, &g.q, &st.yuv.as_ref().unwrap().buf);
        let (dy, dc) = yuv_diff(
            &samples_of(&b, deep),
            &yuv_of(&px, 256, 128, f, true),
            256,
            128,
        );
        eprintln!("{f:?} out vs RGBA out: max |ΔY| {dy}, |ΔC| {dc}");
        assert!(dy <= 1 && dc <= 1);
    }
    // a picture of its own (a flat view) 150 wide: rows padded to 152 samples, edges repeated
    let (w, h) = (150usize, 40usize);
    let pic: Vec<u8> = (0..w * h * 4)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect();
    let t = texture(&g, w as u32, h as u32, wgpu::TextureFormat::Rgba8Unorm);
    write(
        &g,
        &t,
        wgpu::TextureAspect::All,
        &pic,
        w as u32,
        h as u32,
        4,
    );
    for deep in [false, true] {
        let f = Fmt { deep, chroma: 0 };
        let y = stitch::Yuv::new(&cx, &t.create_view(&Default::default()), f, false).unwrap();
        let mut enc = g.dev.create_command_encoder(&Default::default());
        y.encode(&mut enc);
        g.q.submit([enc.finish()]);
        let n = y.pitch as usize / (1 + deep as usize);
        assert_eq!(n, 152);
        let b = samples_of(&stitch::read_buffer(&g.dev, &g.q, &y.buf), deep);
        let b: Vec<u16> = b.chunks(n).flat_map(|r| &r[..w]).copied().collect();
        let (dy, dc) = yuv_diff(&b, &yuv_of(&rgb_of(&pic, false), w, h, f, false), w, h);
        eprintln!(
            "{f:?} of a {w}x{h} picture, pitch {}: max |ΔY| {dy}, |ΔC| {dc}",
            y.pitch
        );
        assert!(dy <= 1 && dc <= 1);
    }

    let st = Stitcher::new(&cx, &cal, fr, 256, 128, Opts::default()).unwrap();
    let mut p = Pose::euler(3.0, -2.0, 30.0);
    st.pose(&p);
    let p0 = run(&g, &st, [&a[0], &a[1]]);
    p.rs = [Vec3::ZERO; 2];
    st.pose(&p);
    assert_eq!(p0, run(&g, &st, [&a[0], &a[1]]));
    p.rs = [Vec3::new(0.0, 0.0, 0.05); 2];
    st.pose(&p);
    let p1 = run(&g, &st, [&a[0], &a[1]]);
    eprintln!(
        "rolling shutter 0.05 rad: PSNR {:.1} dB vs none",
        psnr(&p0, &p1)
    );
    assert_ne!(p0, p1);

    let pv = [&a, &r].map(|src| {
        let pv = Preview::new(
            &cx,
            &cal,
            fc,
            6.0,
            Some(0.1),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        (pv, [&src[0], &src[1]])
    });
    let t = texture(&g, 128, 64, wgpu::TextureFormat::Rgba8Unorm);
    let tv = t.create_view(&Default::default());
    // a smooth equirect, drawn by a preview without calibration
    let (ew, eh) = (256usize, 128usize);
    let eq: Vec<u8> = (0..ew * eh)
        .flat_map(|i| {
            let (x, y) = ((i % ew) as f32 / ew as f32, (i / ew) as f32 / eh as f32);
            let f = |v: f32| (127.5 + 100.0 * v).round() as u8;
            let tau = std::f32::consts::TAU;
            [
                f((tau * x).sin()),
                f((tau * y).cos()),
                f((2.0 * tau * x).cos() * (tau * y / 2.0).sin()),
                255,
            ]
        })
        .collect();
    let et = texture(&g, ew as u32, eh as u32, wgpu::TextureFormat::Rgba8Unorm);
    write(
        &g,
        &et,
        wgpu::TextureAspect::All,
        &eq,
        ew as u32,
        eh as u32,
        4,
    );
    let pano = cx.pano(&et.create_view(&Default::default()));
    let pe = Preview::pano(&cx, wgpu::TextureFormat::Rgba8Unorm);
    let q = Pose::euler(10.0, -20.0, 30.0);
    pe.pose(&q);
    let draw = |pv: &Preview, v: &View, src: Input| {
        let mut enc = g.dev.create_command_encoder(&Default::default());
        if let Input::Lenses(s) = src {
            pv.colour(&mut enc, s);
        }
        pv.render(&mut enc, &tv, [128, 64], v, src);
        g.q.submit([enc.finish()]);
        stitch::read_texture(&g.dev, &g.q, &t)
    };
    // the planet at eye distance 1 (stereographic), 0.4 and 0 (rectilinear)
    for (mode, planet) in [
        (Mode::Perspective, 1.0),
        (Mode::Equirect, 1.0),
        (Mode::Planet, 1.0),
        (Mode::Planet, 0.4),
        (Mode::Planet, 0.0),
    ] {
        let v = View {
            mode,
            fov: 120.0,
            panini: 0.5,
            planet,
            ..Default::default()
        };
        let [pa, pr] = pv.each_ref().map(|(pv, src)| draw(pv, &v, (*src).into()));
        assert!(
            pa.chunks(4).any(|p| p[..3] != [0, 0, 0]),
            "{mode:?} drew nothing"
        );
        assert_eq!(pa, pr, "{mode:?}: rects vs textures");
        let p = psnr(
            &draw(&pe, &v, (&pano).into()),
            &reproject(&eq, ew, eh, &v, q.q, 128, 64),
        );
        eprintln!(
            "preview {mode:?} (planet {planet}) of an equirect vs its CPU reprojection: PSNR {p:.1} dB"
        );
        assert!(p > 40.0);
    }
}

/// A P010 texture (Y, then interleaved Cb Cr; 10-bit codes, stored in the high bits as crate video decodes H.265
/// Main 10) holding `p`.
fn slot10(g: &Gpu, p: &[u16], w: u32, h: u32) -> [wgpu::TextureView; 2] {
    let b: Vec<u8> = p.iter().flat_map(|&v| (v << 6).to_le_bytes()).collect();
    picture(g, &b, w, h, 2)
}

/// P010 lens pictures (H.265 Main 10 as crate video decodes it): whole textures, rects of one side-by-side texture and
/// separate R16 / RG16 plane textures give the same panorama and the same preview in every mode; constant pictures
/// stitch to full-range BT.709 of their 10-bit codes (d / 1023, chroma zero 512) to within one step of the RGBA16F
/// output (the store may truncate); a picture and its 8-bit rounding (NV12) stitch and preview alike to within that
/// rounding.
#[test]
fn p010() {
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let cal = Calib::parse(CAL).unwrap();
    let fr = Frame {
        w: 64,
        h: 64,
        crop: 4600.0, // cuts the image circle: the lenses' edge texels count
        shift: [0.0; 2],
    };
    let wide = Opts {
        wide: true,
        colour: None,
        ..Default::default()
    };
    // constant pictures: every output pixel is the picture's colour (outside the image circles too, which sample the
    // edge texel), so the output checks the 10-bit scale exactly
    // one f16 step at v (10 fraction bits; normal from 2^-14), and f32 noise
    let f16 = |v: f64| 2f64.powf(v.max(2f64.powi(-14)).log2().floor() - 10.0) + 1e-6;
    for [y, cb, cr] in [
        [64, 512, 512],
        [1000, 512, 512],
        [940, 300, 700],
        [512, 100, 900],
        [300, 960, 64],
    ] {
        let px: Vec<u16> = (0..64 * 96)
            .map(|i| match i < 64 * 64 {
                true => y,
                false => [cb, cr][i % 2],
            })
            .collect();
        let t = slot10(&g, &px, 64, 64);
        let src = cx.rect(&t, [0, 0, 64, 64]);
        let st = Stitcher::new(&cx, &cal, fr, 64, 32, wide).unwrap();
        let out = rgb_of(&run(&g, &st, [&src, &src]), true);
        let (l, b, r) = (
            y as f64 / 1023.0,
            (cb as f64 - 512.0) / 1023.0,
            (cr as f64 - 512.0) / 1023.0,
        );
        let want = [
            l + 1.5748 * r,
            l - 0.187324 * b - 0.468124 * r,
            l + 1.8556 * b,
        ]
        .map(|v| v.clamp(0.0, 1.0));
        let d = out
            .iter()
            .flat_map(|p| (0..3).map(move |k| (p[k] - want[k]).abs() - f16(want[k])))
            .fold(f64::MIN, f64::max);
        eprintln!(
            "P010 constant Y'CbCr ({y}, {cb}, {cr}): R'G'B' {:.5?} (want {want:.5?}), worst |Δ| − one f16 step {d:.2e}",
            out[0]
        );
        assert!(d <= 0.0, "({y}, {cb}, {cr})");
    }

    let mut s = 0x2545f491u32;
    let p: [Vec<u16>; 2] = std::array::from_fn(|_| {
        (0..64 * 96)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((i % 64) * 8 + (i / 64 % 64) * 3 + (s >> 26) as usize) as u16
            })
            .collect()
    });
    let whole = p
        .each_ref()
        .map(|v| cx.rect(&slot10(&g, v, 64, 64), [0, 0, 64, 64]));
    // both lenses side by side at (4, 2) of a 136 × 68 texture, noise around them
    let (tw, at) = (136usize, [4usize, 2usize]);
    let mut sbs: Vec<u16> = (0..tw * 68 * 3 / 2)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 22) as u16
        })
        .collect();
    for (k, v) in p.iter().enumerate() {
        for y in 0..64 {
            let x = at[0] + 64 * k;
            let d = (at[1] + y) * tw + x;
            sbs[d..d + 64].copy_from_slice(&v[64 * y..][..64]);
            if y < 32 {
                let d = tw * 68 + (at[1] / 2 + y) * tw + x;
                sbs[d..d + 64].copy_from_slice(&v[64 * 64 + 64 * y..][..64]);
            }
        }
    }
    let t = slot10(&g, &sbs, tw as u32, 68);
    let rect = [0, 1].map(|k| cx.rect(&t, [(at[0] + 64 * k) as u32, at[1] as u32, 64, 64]));
    let planes = p.each_ref().map(|v| {
        let b: Vec<u8> = v.iter().flat_map(|&x| (x << 6).to_le_bytes()).collect();
        let y = texture(&g, 64, 64, wgpu::TextureFormat::R16Unorm);
        let uv = texture(&g, 32, 32, wgpu::TextureFormat::Rg16Unorm);
        write(
            &g,
            &y,
            wgpu::TextureAspect::All,
            &b[..64 * 64 * 2],
            64,
            64,
            2,
        );
        write(
            &g,
            &uv,
            wgpu::TextureAspect::All,
            &b[64 * 64 * 2..],
            32,
            32,
            4,
        );
        cx.planes(
            &y.create_view(&Default::default()),
            &uv.create_view(&Default::default()),
        )
    });
    // the 8-bit rounding of the same pictures (full range: chroma about its zero, 512 of 1023, 128 of 255)
    let nv = p.each_ref().map(|v| {
        let b: Vec<u8> = (v.iter().enumerate())
            .map(|(i, &x)| match i < 64 * 64 {
                true => (x as f64 * 255.0 / 1023.0).round() as u8,
                false => (128.0 + (x as f64 - 512.0) * 255.0 / 1023.0).round() as u8,
            })
            .collect();
        upload(&g, &cx, &b, 64, 64)
    });
    let o = Opts {
        colour: Some(0.1),
        ..wide
    };
    let [pw, pr, pp, pn] = [&whole, &rect, &planes, &nv].map(|src| {
        let st = Stitcher::new(&cx, &cal, fr, 256, 128, o).unwrap();
        rgb_of(&run(&g, &st, [&src[0], &src[1]]), true)
    });
    assert!(pw == pr, "P010: rects of one texture vs two textures");
    assert!(pw == pp, "P010: textures vs R16 / RG16 plane textures");
    // 8-bit rounding: at most half a code of 255 per sample, through the BT.709 matrix
    let dn = pw
        .iter()
        .zip(&pn)
        .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
        .fold(0.0, f64::max);
    eprintln!("P010 vs its NV12 rounding: panorama max |ΔR'G'B'| {dn:.4}");
    assert!(dn < 0.5 / 255.0 * (1.0 + 1.8556) + 1e-3);

    let pv = Preview::new(
        &cx,
        &cal,
        fr,
        6.0,
        Some(0.1),
        wgpu::TextureFormat::Rgba8Unorm,
    );
    let t = texture(&g, 128, 64, wgpu::TextureFormat::Rgba8Unorm);
    let tv = t.create_view(&Default::default());
    for (mode, planet) in [
        (Mode::Perspective, 1.0),
        (Mode::Equirect, 1.0),
        (Mode::Planet, 1.0),
        (Mode::Planet, 0.0),
    ] {
        let v = View {
            mode,
            fov: 120.0,
            panini: 0.5,
            planet,
            ..Default::default()
        };
        let [vw, vr, vp, vn] = [&whole, &rect, &planes, &nv].map(|src| {
            let src = [&src[0], &src[1]];
            let mut enc = g.dev.create_command_encoder(&Default::default());
            pv.colour(&mut enc, src);
            pv.render(&mut enc, &tv, [128, 64], &v, src);
            g.q.submit([enc.finish()]);
            stitch::read_texture(&g.dev, &g.q, &t)
        });
        let d = vw
            .iter()
            .zip(&vn)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        eprintln!("P010 preview {mode:?} (planet {planet}): vs NV12 rounding max |Δ| {d} levels");
        assert!(
            vw.chunks(4).any(|p| p[..3] != [0, 0, 0]),
            "{mode:?} drew nothing"
        );
        assert!(
            vw == vr && vw == vp,
            "{mode:?}: P010 textures vs rects vs planes"
        );
        assert!(d <= 2, "{mode:?}: P010 vs NV12");
    }
}

/// Writes the lens images the old stitcher reads, runs it if `X5_OLDREF` is set, returns its panorama.
fn old(c: &Clip, w: u32, raft: bool) -> Option<Vec<u8>> {
    let fx = fixtures();
    let lens = [0, 1].map(|k| fx.join(format!("v635_lens{k}.rgba")));
    for (k, p) in lens.iter().enumerate() {
        if !p.exists() {
            let tmp = part(p);
            std::fs::write(&tmp, rgba(&c.nv[k], SIDE, SIDE)).unwrap();
            std::fs::rename(tmp, p).unwrap();
        }
    }
    let out = fx.join(format!(
        "v635_old_{w}{}.rgba",
        if raft { "_raft" } else { "" }
    ));
    if !out.exists() {
        let Some(bin) = std::env::var_os("X5_OLDREF") else {
            eprintln!("X5_OLDREF unset: no reference panorama");
            return None;
        };
        let samples = std::env::var_os("X5_SAMPLES").unwrap();
        let mut cmd = Command::new(bin);
        cmd.arg(Path::new(&samples).join("v635.insv"))
            .args(&lens)
            .args([SIDE.to_string(), w.to_string()])
            .arg(&out);
        if raft {
            cmd.arg("--raft");
        }
        assert!(cmd.status().unwrap().success(), "oldref failed");
    }
    std::fs::read(out).ok()
}

/// Against insta360-stitch on the same pixels (its RGBA input converted from the NV12 frame by `rgba`, which
/// matches the shader), template and RAFT, with the timings at 7680 × 3840.
#[test]
fn vs_old() {
    let Some(c) = clip() else { return };
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let src =
        c.nv.each_ref()
            .map(|nv| upload(&g, &cx, nv, SIDE as u32, SIDE as u32));
    let src = [&src[0], &src[1]];
    let (w, h) = (7680, 3840);
    eprintln!(
        "v635: offset v{}, crop {} {:?}, blend {:?}, reverse {}",
        c.cal.version, c.meta.crop, c.meta.shift, c.meta.blend, c.meta.reverse
    );
    let o = Opts {
        blend: c.meta.blend.unwrap_or(6.0),
        colour: None,
        ..Default::default()
    };
    let st = Stitcher::new(&cx, &c.cal, c.fr, w, h, o).unwrap();
    st.pose(&framing(&c.meta));
    let new = run(&g, &st, src);
    std::fs::write(fixtures().join("v635_new_7680.rgba"), &new).unwrap();
    let ms = time(&g, 20, |e| st.encode(e, src));
    eprintln!("template 7680x3840 RGBA: {ms:.2} ms/frame");
    let tpl = old(&c, w, false);
    if let Some(r) = &tpl {
        let p = psnr(&new, r);
        eprintln!(
            "template vs insta360-stitch: PSNR {p:.2} dB, > 2 levels {:.4} %, > 8 levels {:.4} %",
            100.0 * over(&new, r, 2),
            100.0 * over(&new, r, 8)
        );
        assert!(p > 40.0, "PSNR {p:.2} dB");
    }

    let st = Stitcher::new(
        &cx,
        &c.cal,
        c.fr,
        w,
        h,
        Opts {
            colour: Some(0.1),
            ..o
        },
    )
    .unwrap();
    st.pose(&framing(&c.meta));
    let ms = time(&g, 20, |e| st.encode(e, src));
    eprintln!("template + colour matching 7680x3840: {ms:.2} ms/frame");
    for deep in [false, true] {
        let f = Fmt { deep, chroma: 0 };
        let o = Opts {
            yuv: Some(f),
            wide: deep,
            ..o
        };
        let st = Stitcher::new(&cx, &c.cal, c.fr, w, h, o).unwrap();
        st.pose(&framing(&c.meta));
        let y = st.yuv.as_ref().unwrap();
        let ms = time(&g, 20, |e| {
            st.encode(e, src);
            y.encode(e)
        });
        eprintln!("template + {f:?} 7680x3840: {ms:.2} ms/frame");
        let px = rgb_of(&stitch::read_texture(&g.dev, &g.q, &st.out), deep);
        let b = stitch::read_buffer(&g.dev, &g.q, &st.yuv.as_ref().unwrap().buf);
        let (dy, dc) = yuv_diff(
            &samples_of(&b, deep),
            &yuv_of(&px, w as usize, h as usize, f, true),
            w as usize,
            h as usize,
        );
        eprintln!("  YUV out vs RGBA out: max |ΔY| {dy}, |ΔC| {dc}");
        assert!(dy <= 1 && dc <= 1);
    }

    // RAFT seam flow
    let raft = Raft::new(&g.inst, &g.ad, &g.dev, &g.q).unwrap();
    let mut st = Stitcher::new(
        &cx,
        &c.cal,
        c.fr,
        w,
        h,
        Opts {
            flow: Some(Flow::RAFT),
            ..o
        },
    )
    .unwrap();
    st.pose(&framing(&c.meta));
    let frame = |st: &mut Stitcher| {
        let t = Instant::now();
        let mut enc = g.dev.create_command_encoder(&Default::default());
        st.belt(&mut enc, src, &raft);
        g.q.submit([enc.finish()]);
        st.flow(&raft).unwrap();
        let mut enc = g.dev.create_command_encoder(&Default::default());
        st.encode(&mut enc, src);
        g.q.submit([enc.finish()]);
        g.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    eprintln!(
        "RAFT first frame (compile + autotune): {:.1} s",
        frame(&mut st) / 1e3
    );
    // the same input every frame: the smoothed flow settles on the raw one (the old stitcher ran 4 frames too)
    let t: Vec<f64> = (0..5).map(|_| frame(&mut st)).collect();
    eprintln!("RAFT frame 7680x3840 (belt, RAFT both directions, fill, stitch), ms: {t:.1?}");
    let new = stitch::read_texture(&g.dev, &g.q, &st.out);
    std::fs::write(fixtures().join("v635_new_7680_raft.rgba"), &new).unwrap();
    if let Some(r) = old(&c, w, true) {
        eprintln!(
            "RAFT vs insta360-stitch RAFT: PSNR {:.2} dB, > 2 levels {:.4} %, > 8 levels {:.4} %",
            psnr(&new, &r),
            100.0 * over(&new, &r, 2),
            100.0 * over(&new, &r, 8)
        );
        if let Some(t) = &tpl {
            eprintln!(
                "  (insta360-stitch RAFT vs its template: PSNR {:.2} dB)",
                psnr(&r, t)
            );
        }
    }

    // a view across the seam (1280 × 720, 60°: the panorama's 21.3 px/°) drawn from the lenses through the flow
    // (`Preview::seam`), against the same view of the RAFT panorama and of the template preview
    let fmt = wgpu::TextureFormat::Rgba8Unorm;
    let (vw, vh) = (1280, 720);
    let t = texture(&g, vw, vh, fmt);
    let tv = t.create_view(&Default::default());
    let v = View {
        yaw: 90.0,
        fov: 60.0,
        ..Default::default()
    };
    let shot = |f: &dyn Fn(&mut wgpu::CommandEncoder)| {
        let mut enc = g.dev.create_command_encoder(&Default::default());
        f(&mut enc);
        g.q.submit([enc.finish()]);
        stitch::read_texture(&g.dev, &g.q, &t)
    };
    let seam = Preview::seam(&st, fmt);
    let a = shot(&|e| seam.render(e, &tv, [vw, vh], &v, src));
    let (pp, pano) = (
        Preview::pano(&cx, fmt),
        cx.pano(&st.out.create_view(&Default::default())),
    );
    let b = shot(&|e| pp.render(e, &tv, [vw, vh], &v, &pano));
    let tp = Preview::new(&cx, &c.cal, c.fr, o.blend, None, fmt);
    tp.pose(&framing(&c.meta));
    let tq = shot(&|e| tp.render(e, &tv, [vw, vh], &v, src));
    let (pb, pt) = (psnr(&a, &b), psnr(&a, &tq));
    eprintln!(
        "view across the seam through the flow: vs the RAFT panorama {pb:.2} dB, vs the template preview {pt:.2} dB"
    );
    assert!(pb > 30.0 && pb > pt);
}

/// Lens k's colour toward body direction `w`, projected on the CPU with `Calib::project` into its RGBA image of `fr`'s
/// size and sampled bilinearly; None unless fully inside its mask.
fn look(cal: &Calib, fr: Frame, k: usize, img: &[u8], w: [f64; 3]) -> Option<[f32; 3]> {
    let (n, crop) = (fr.w as usize, fr.crop as f64);
    let s = n as f64 / crop;
    let c0 = [0, 1].map(|k| (cal.side - crop) / 2.0 + fr.shift[k] as f64);
    let lim = (cal.fov / 2.0).to_radians();
    let l = &cal.lens[k];
    let p: [f64; 3] = std::array::from_fn(|r| (0..3).map(|j| l.rot[r][j] * w[j]).sum());
    if p[2] / (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() < lim.cos() - 0.01 {
        return None;
    }
    let q = cal.project(k, p);
    let t = [(q[0] - c0[0]) * s, (q[1] - c0[1]) * s];
    let r = (cal.project(k, [lim.sin(), 0.0, lim.cos()])[0] - l.cx) * s;
    let d = (t[0] - (l.cx - c0[0]) * s).hypot(t[1] - (l.cy - c0[1]) * s);
    if (r - d) * 0.2439 < 1.0 {
        return None;
    }
    let (x, y) = (t[0] - 0.5, t[1] - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
    let at = |dx: isize, dy: isize, k: usize| {
        let (xx, yy) = (
            (x0 as isize + dx).clamp(0, n as isize - 1),
            (y0 as isize + dy).clamp(0, n as isize - 1),
        );
        img[4 * (yy as usize * n + xx as usize) + k] as f32
    };
    Some(std::array::from_fn(|k| {
        let t = at(0, 0, k) + fx * (at(1, 0, k) - at(0, 0, k));
        let b = at(0, 1, k) + fx * (at(1, 1, k) - at(0, 1, k));
        t + fy * (b - t)
    }))
}

/// Both lenses at the same directions of the colour band (`stitch.wgsl` `stats`: S frame, a over ±150°, e over ±5°,
/// 1024 × 16), projected on the CPU into the RGBA lens images; the pairs where both lenses are fully inside their
/// masks and no channel is clipped or black.
fn band(c: &Clip, img: [&[u8]; 2]) -> Vec<[[f32; 3]; 2]> {
    let mut out = vec![];
    for i in 0..16384 {
        let a = (300.0 * ((i / 16) as f64 + 0.5) / 1024.0 - 150.0).to_radians();
        let e = (10.0 * ((i % 16) as f64 + 0.5) / 16.0 - 5.0).to_radians();
        let w = [-e.sin(), a.sin() * e.cos(), a.cos() * e.cos()];
        if let [Some(p), Some(q)] = [0, 1].map(|k| look(&c.cal, c.fr, k, img[k], w))
            && p.iter()
                .chain(&q)
                .all(|&v| v > 0.02 * 255.0 && v < 0.98 * 255.0)
        {
            out.push([p, q]);
        }
    }
    out
}

/// Lens 0 − lens 1 over the band after applying `fit` (None: as is): mean per channel and mean |difference|, levels.
fn mismatch(band: &[[[f32; 3]; 2]], fit: Option<[[f32; 3]; 4]>) -> ([f64; 3], f64) {
    let f = fit.unwrap_or([[1.0; 3], [0.0; 3], [1.0; 3], [0.0; 3]]);
    let (mut m, mut a) = ([0f64; 3], 0f64);
    for [p, q] in band {
        for k in 0..3 {
            let c = |v: f32, g: f32, o: f32| ((g * v / 255.0 + o).clamp(0.0, 1.0) * 255.0) as f64;
            let d = c(p[k], f[0][k], f[1][k]) - c(q[k], f[2][k], f[3][k]);
            m[k] += d;
            a += d.abs();
        }
    }
    let n = band.len() as f64;
    (m.map(|v| v / n), a / n / 3.0)
}

/// Colour matching: lens 0 vs lens 1 at the same directions of the overlap band before and after the fit, on the
/// real frame and with lens 1 darkened by 20 % (an exposure change: RGB × 0.8, i.e. Y and centred chroma × 0.8);
/// then the smoothed fit (0.1) following a switch from the real to the darkened lens 1.
#[test]
fn colour() {
    let Some(c) = clip() else { return };
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let n = SIDE * SIDE;
    let mut dark = c.nv[1].clone();
    for (i, v) in dark.iter_mut().enumerate() {
        *v = if i < n {
            (*v as f32 * 0.8).round() as u8
        } else {
            ((*v as f32 - 128.0) * 0.8 + 128.0).round() as u8
        };
    }
    let s = SIDE as u32;
    let l0 = upload(&g, &cx, &c.nv[0], s, s);
    let l1 = upload(&g, &cx, &c.nv[1], s, s);
    let d1 = upload(&g, &cx, &dark, s, s);
    let im0 = rgba(&c.nv[0], SIDE, SIDE);
    let f = |v: [f64; 3]| format!("[{:+.2} {:+.2} {:+.2}]", v[0], v[1], v[2]);
    let o = Opts {
        colour: Some(1.0),
        ..Default::default()
    };
    let (w, h) = (3840, 1920);
    let mut bands = vec![];
    for (name, src, nv) in [("real", &l1, &c.nv[1]), ("lens 1 × 0.8", &d1, &dark)] {
        let b = band(&c, [&im0, &rgba(nv, SIDE, SIDE)]);
        let st = Stitcher::new(&cx, &c.cal, c.fr, w, h, o).unwrap();
        run(&g, &st, [&l0, src]);
        let fit = st.fit().unwrap();
        let (m0, a0) = mismatch(&b, None);
        let (m1, a1) = mismatch(&b, Some(fit));
        eprintln!(
            "{name}: {} band pairs; lens 0 − lens 1 mean {} |Δ| {a0:.2} -> after the fit {} |Δ| {a1:.2}",
            b.len(),
            f(m0),
            f(m1)
        );
        eprintln!(
            "  fit: gain/offset lens 0 {:.3?} {:.4?}, lens 1 {:.3?} {:.4?}",
            fit[0], fit[1], fit[2], fit[3]
        );
        let bias = |m: [f64; 3]| m.iter().map(|v| v.abs()).sum::<f64>() / 3.0;
        assert!(
            bias(m1) < 0.25 * bias(m0).max(1.0),
            "{name}: bias {} -> {}",
            f(m0),
            f(m1)
        );
        assert!(a1 < a0, "{name}: |Δ| {a0} -> {a1}");
        bands.push(b);
    }
    let st = Stitcher::new(
        &cx,
        &c.cal,
        c.fr,
        w,
        h,
        Opts {
            colour: Some(0.1),
            ..o
        },
    )
    .unwrap();
    for _ in 0..10 {
        run(&g, &st, [&l0, &l1]);
    }
    for k in 1..=30 {
        run(&g, &st, [&l0, &d1]);
        if [1, 5, 10, 20, 30].contains(&k) {
            let (m, a) = mismatch(&bands[1], st.fit());
            eprintln!(
                "smoothing 0.1, frame {k} after the switch to the darkened lens 1: mean {} |Δ| {a:.2}",
                f(m)
            );
        }
    }
}

/// Preview vs the stitcher: each mode at 1920 × 1080 against a CPU reprojection of the 7680 × 3840 panorama, and
/// the time per frame; then the same views drawn from an equirect (the body-frame panorama, as a paused player shows
/// its RAFT stitch) against the views from the lenses.
#[test]
fn preview() {
    let Some(c) = clip() else { return };
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let src =
        c.nv.each_ref()
            .map(|nv| upload(&g, &cx, nv, SIDE as u32, SIDE as u32));
    let src = [&src[0], &src[1]];
    let blend = c.meta.blend.unwrap_or(6.0);
    let (pw, ph) = (7680usize, 3840usize);
    let o = Opts {
        blend,
        colour: None,
        ..Default::default()
    };
    let st = Stitcher::new(&cx, &c.cal, c.fr, pw as u32, ph as u32, o).unwrap();
    let pose = framing(&c.meta);
    st.pose(&pose);
    let pano = run(&g, &st, src);
    let sb = Stitcher::new(&cx, &c.cal, c.fr, pw as u32, ph as u32, o).unwrap();
    let body = run(&g, &sb, src);
    let eq = cx.pano(&sb.out.create_view(&Default::default()));
    let pv = Preview::new(
        &cx,
        &c.cal,
        c.fr,
        blend,
        None,
        wgpu::TextureFormat::Rgba8Unorm,
    );
    pv.pose(&pose);
    let (w, h) = (1920u32, 1080u32);
    let t = texture(&g, w, h, wgpu::TextureFormat::Rgba8Unorm);
    let tv = t.create_view(&Default::default());
    let views = [
        View {
            yaw: 30.0,
            pitch: 20.0,
            roll: 5.0,
            fov: 100.0,
            ..Default::default()
        },
        View {
            yaw: -40.0,
            pitch: -10.0,
            fov: 150.0,
            panini: 1.0,
            ..Default::default()
        },
        View {
            mode: Mode::Equirect,
            yaw: 90.0,
            ..Default::default()
        },
        View {
            mode: Mode::Planet,
            pitch: -90.0,
            fov: 300.0,
            ..Default::default()
        },
    ];
    for v in views {
        let ms = time(&g, 50, |e| pv.render(e, &tv, [w, h], &v, src));
        let px = stitch::read_texture(&g.dev, &g.q, &t);
        let name = format!("preview_{:?}_{}.rgba", v.mode, v.panini);
        std::fs::write(fixtures().join(name), &px).unwrap();
        let r = reproject(&pano, pw, ph, &v, Quat::IDENTITY, w as usize, h as usize);
        let p = psnr(&px, &r);
        eprintln!(
            "preview {:?} (panini {}) {w}x{h}: {ms:.3} ms/frame, vs reprojected panorama PSNR {p:.2} dB",
            v.mode, v.panini
        );
        assert!(p > 30.0);
        let me = time(&g, 50, |e| pv.render(e, &tv, [w, h], &v, &eq));
        let pe = stitch::read_texture(&g.dev, &g.q, &t);
        let r = reproject(&body, pw, ph, &v, pose.q, w as usize, h as usize);
        let (p, pr) = (psnr(&pe, &px), psnr(&pe, &r));
        eprintln!(
            "  from the equirect: {me:.3} ms/frame, vs from the lenses PSNR {p:.2} dB, vs its CPU reprojection {pr:.2} dB"
        );
        assert!(p > 30.0 && pr > 45.0);
    }
    // with colour matching, as the player runs it
    let pv = Preview::new(
        &cx,
        &c.cal,
        c.fr,
        blend,
        Some(0.1),
        wgpu::TextureFormat::Rgba8Unorm,
    );
    pv.pose(&pose);
    let ms = time(&g, 50, |e| {
        pv.colour(e, src);
        pv.render(e, &tv, [w, h], &views[0], src)
    });
    eprintln!("preview perspective + colour fit {w}x{h}: {ms:.3} ms/frame");
}

/// A `w` × `h` panorama in the stitcher's layout (column x at 2πx/w, row y at πy/(h − 1)) on the `ow` × `oh` grid of
/// that layout: per output pixel the mean of f × f bilinear samples 1 px apart (f = w/ow) centred on its direction.
/// (A box over whole pixels would sit (f − 1)/2 px off it.)
fn shrink(px: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    let f = w / ow;
    let at = |x: f64, y: f64, k: usize| {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let p = |dx: isize, dy: isize| {
            let xx = (x0 as isize + dx).rem_euclid(w as isize) as usize;
            let yy = (y0 as isize + dy).clamp(0, h as isize - 1) as usize;
            px[4 * (yy * w + xx) + k] as f64
        };
        let t = p(0, 0) + fx * (p(1, 0) - p(0, 0));
        t + fy * (p(0, 1) + fx * (p(1, 1) - p(0, 1)) - t)
    };
    let mut o = vec![255u8; ow * oh * 4];
    for (i, q) in o.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (x, y) = (
            (i % ow * f) as f64,
            (i / ow) as f64 * (h - 1) as f64 / (oh - 1) as f64,
        );
        let c = (f - 1) as f64 / 2.0;
        for (k, v) in q[..3].iter_mut().enumerate() {
            let s: f64 = (0..f * f)
                .map(|j| at(x + (j % f) as f64 - c, y + (j / f) as f64 - c, k))
                .sum();
            *v = (s / (f * f) as f64).round() as u8;
        }
    }
    o
}

/// The `.lrv` proxy (H.264 1664 × 832, both lenses side by side, its own trailer) against the `.insv`: `.lrv` frame n
/// stitched at 1664 × 832 from two rects of one texture vs `.insv` frame n + k stitched at 6656 × 3328 and shrunk,
/// PSNR per hypothesis: the frame offset k, which half holds lens 0, the window crop, a sub-pixel shift.
#[test]
fn lrv() {
    let Some(d) = samples() else { return };
    let (lrv, insv) = (d.join("v635.lrv"), d.join("v635.insv"));
    let (ml, mi) = (trailer(&lrv), trailer(&insv));
    assert!(
        ml.offset == mi.offset && ml.v3 == mi.v3 && (ml.crop, ml.shift) == (mi.crop, mi.shift),
        ".lrv vs .insv trailer"
    );
    eprintln!(
        ".lrv trailer: the .insv's calibration and window crop ({} {:?}); tracks reversed: .lrv {}, .insv {}",
        ml.crop, ml.shift, ml.reverse, mi.reverse
    );
    // a frame with motion, so neighbours differ (the .lrv lacks the .insv's last 2 frames)
    const N: usize = 151;
    let Some(nv) = frame(&lrv, 0, N) else { return };
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let cal = Calib::parse(&ml.offset).unwrap();
    let o = Opts {
        blend: ml.blend.unwrap_or(6.0),
        colour: None,
        ..Default::default()
    };
    let (w, h) = (1664u32, 832u32);
    let t = slot(&g, &nv, w, h);
    let half = [0, 1].map(|k| cx.rect(&t, [832 * k, 0, 832, 832]));
    // left: the half that holds lens 0; d: window shift in calibration px
    let at = |left: usize, crop: f32, d: f32| {
        let fr = Frame {
            w: 832,
            h: 832,
            crop,
            shift: [d; 2],
        };
        let st = Stitcher::new(&cx, &cal, fr, w, h, o).unwrap();
        run(&g, &st, [&half[left], &half[1 - left]])
    };
    let fi = Frame {
        w: SIDE as u32,
        h: SIDE as u32,
        crop: mi.crop,
        shift: mi.shift,
    };
    let st = Stitcher::new(&cx, &cal, fi, 4 * w, 4 * h, o).unwrap();
    let mut refs = vec![];
    for n in N - 1..=N + 2 {
        let (Some(a), Some(b)) = (frame(&insv, 0, n), frame(&insv, 1, n)) else {
            return;
        };
        let mut src = [a, b].map(|v| upload(&g, &cx, &v, SIDE as u32, SIDE as u32));
        if mi.reverse {
            src.swap(0, 1);
        }
        let px = run(&g, &st, [&src[0], &src[1]]);
        refs.push(shrink(
            &px,
            4 * w as usize,
            4 * h as usize,
            w as usize,
            h as usize,
        ));
    }
    let f = |v: &[f64]| {
        v.iter()
            .map(|p| format!("{p:.2}"))
            .collect::<Vec<_>>()
            .join(" / ")
    };
    let base = at(0, mi.crop, 0.0);
    let pk: Vec<f64> = refs.iter().map(|r| psnr(&base, r)).collect();
    eprintln!(
        ".lrv frame {N} vs .insv frame {N} + k, k = -1 / 0 / 1 / 2 (lens 0 left, crop {}): PSNR {} dB",
        mi.crop,
        f(&pk)
    );
    let r = &refs[1];
    let hyp = [(0, mi.crop), (1, mi.crop), (0, 0.0), (1, 0.0)];
    let ph: Vec<f64> = hyp.iter().map(|&(l, c)| psnr(&at(l, c, 0.0), r)).collect();
    eprintln!(
        "lens 0 left, crop {c} / lens 0 right, crop {c} / lens 0 left, no crop / lens 0 right, no crop: PSNR {} dB",
        f(&ph),
        c = mi.crop
    );
    // scalers that sample output pixel i at source texel i·(window/size) shift each stream by ½(window/size − 1)
    // window px against the centred mapping: the .lrv by ½(5312/832 − 5312/3840) = 2.5 px more than the .insv
    let dm = -(mi.crop as f64 / 832.0 - mi.crop as f64 / 3840.0) as f32 / 2.0;
    let ds = [-4.0, -3.5, -3.0, dm, -2.0, -1.5, -1.0, 0.0, 1.0];
    let pd: Vec<f64> = ds.iter().map(|&d| psnr(&at(0, mi.crop, d), r)).collect();
    eprintln!(
        "window shift {ds:.2?} calibration px (lens 0 left, crop {}): PSNR {} dB",
        mi.crop,
        f(&pd)
    );
    for (x, y) in [(dm, 0.0), (0.0, dm)] {
        let fr = Frame {
            w: 832,
            h: 832,
            crop: mi.crop,
            shift: [x, y],
        };
        let st = Stitcher::new(&cx, &cal, fr, w, h, o).unwrap();
        let p = psnr(&run(&g, &st, [&half[0], &half[1]]), r);
        eprintln!("window shift ({x:.2}, {y:.2}): PSNR {p:.2} dB");
    }
    let best = |v: &[f64]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    assert_eq!(best(&pk), 1, "frame offset");
    assert_eq!(best(&ph), 0, "lens order / crop");
    let top = pd[best(&pd)];
    assert!(
        pd[3] > pd[7] + 1.0 && pd[3] > top - 0.3,
        "window shift {dm}"
    );
}

/// Live preview capture (`X5_LIVE`: raw H.264 2656 × 1328, both lenses side by side, no trailer). Its calibration
/// comes from the camera: the app's capture screen takes MEDIA_OFFSET_V6, else _V3 (`getPlayerOffsetData`); the
/// sample camera answers _V3 only, the string the v635 trailer (same camera) holds as offset_v3, so that is used,
/// with the window crop of notification 8234 (5376 → 5312, as in the files). Plausibility on frame 30: a light on the
/// seam seen by both lenses, its direction in each per hypothesis (which half holds lens 0; v3 vs v6); the stitch
/// and a view of the light go to the fixtures.
#[test]
fn live() {
    let Some(file) = std::env::var_os("X5_LIVE").map(PathBuf::from) else {
        eprintln!("X5_LIVE unset: skipped");
        return;
    };
    let Some(d) = samples() else { return };
    let m = trailer(&d.join("v635.insv"));
    const N: usize = 30;
    let Some(nv) = frame(&file, 0, N) else { return };
    let (w, h) = (2656usize, 1328usize);
    let im = rgba(&nv, w, h);
    let half: [Vec<u8>; 2] = [0, 1].map(|k| {
        im.chunks(4 * w)
            .flat_map(|r| &r[4 * h * k..][..4 * h])
            .copied()
            .collect()
    });
    let fr = Frame {
        w: h as u32,
        h: h as u32,
        crop: m.crop,
        shift: m.shift,
    };
    // the light: red well above green and blue; per lens its centroid along (a) and across (e) the seam (degrees)
    // and area (deg²), over the directions both lenses see, so each sees the same part of it
    let light = |cal: &Calib, left: usize| {
        let mut c = [[0f64; 4]; 2];
        for i in 0..3600 * 240 {
            let (a, e) = (
                (i / 240) as f64 / 10.0 - 180.0,
                (i % 240) as f64 / 10.0 - 12.0,
            );
            let (ar, er) = (a.to_radians(), e.to_radians());
            let w = [-er.sin(), ar.sin() * er.cos(), ar.cos() * er.cos()];
            let [Some(p), Some(q)] = [0, 1].map(|k| look(cal, fr, k, &half[k ^ left], w)) else {
                continue;
            };
            for (c, p) in c.iter_mut().zip([p, q]) {
                let v = (p[0] - p[1].max(p[2]) - 60.0).max(0.0) as f64;
                *c = [
                    c[0] + v * a,
                    c[1] + v * e,
                    c[2] + v,
                    c[3] + (v > 0.0) as u32 as f64,
                ];
            }
        }
        c.map(|c| [c[0] / c[2], c[1] / c[2], c[3] / 100.0])
    };
    let cals = [
        ("v3", Calib::parse(&m.v3).unwrap()),
        ("v6", Calib::parse(&m.offset).unwrap()),
    ];
    let mut res = vec![];
    for (name, cal) in &cals {
        for left in [0, 1] {
            let [p, q] = light(cal, left);
            let da = (p[0] - q[0], p[1] - q[1]);
            eprintln!(
                "{name}, lens 0 {}: light at lens 0 (a {:.2}°, e {:.2}°, {:.2} deg²), lens 1 (a {:.2}°, e {:.2}°, {:.2} deg²): Δ {:.2}° along, {:.2}° across",
                ["left", "right"][left],
                p[0],
                p[1],
                p[2],
                q[0],
                q[1],
                q[2],
                da.0,
                da.1
            );
            res.push((da.0.hypot(da.1), p));
        }
    }
    assert!(
        res[0].0 < res[1].0 && res[2].0 < res[3].0,
        "lens 0 left should put the light's two images closer"
    );
    // on the GPU from rects of one texture, as the app will: the panorama, and a view of the light
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let t = slot(&g, &nv, w as u32, h as u32);
    let half = [0, 1].map(|k| cx.rect(&t, [h as u32 * k, 0, h as u32, h as u32]));
    let o = Opts {
        blend: m.blend.unwrap_or(6.0),
        ..Default::default()
    };
    let st = Stitcher::new(&cx, &cals[0].1, fr, w as u32, h as u32, o).unwrap();
    std::fs::write(
        fixtures().join("live_pano_2656.rgba"),
        run(&g, &st, [&half[0], &half[1]]),
    )
    .unwrap();
    // a view of the light per hypothesis (S frame a, e -> body w -> yaw, pitch)
    let tt = texture(&g, 640, 480, wgpu::TextureFormat::Rgba8Unorm);
    let tv = tt.create_view(&Default::default());
    for (i, (name, cal)) in cals.iter().enumerate() {
        for left in [0, 1] {
            let [a, e, _] = res[2 * i + left].1.map(|v| v.to_radians() as f32);
            let wv = [-e.sin(), a.sin() * e.cos(), a.cos() * e.cos()];
            let v = View {
                yaw: wv[1].atan2(-wv[0]).to_degrees(),
                pitch: wv[2].asin().to_degrees(),
                fov: 40.0,
                ..Default::default()
            };
            let pv = Preview::new(&cx, cal, fr, o.blend, None, wgpu::TextureFormat::Rgba8Unorm);
            let mut enc = g.dev.create_command_encoder(&Default::default());
            pv.render(
                &mut enc,
                &tv,
                [640, 480],
                &v,
                [&half[left], &half[1 - left]],
            );
            g.q.submit([enc.finish()]);
            let px = stitch::read_texture(&g.dev, &g.q, &tt);
            std::fs::write(
                fixtures().join(format!("live_light_{name}_l{left}.rgba")),
                px,
            )
            .unwrap();
        }
    }
}

/// The preview's mapping on the CPU (`stitch.wgsl` `ray`), sampling a panorama in the stitcher's layout bilinearly;
/// `q` takes the output world to the panorama's.
fn reproject(pano: &[u8], pw: usize, ph: usize, v: &View, q: Quat, w: usize, h: usize) -> Vec<u8> {
    let r = f32::to_radians;
    let m = Mat3::from_cols(Vec3::Y, Vec3::Z, Vec3::NEG_X)
        * Mat3::from_rotation_y(r(v.yaw))
        * Mat3::from_rotation_x(-r(v.pitch))
        * Mat3::from_rotation_z(r(v.roll));
    let (d, e) = (v.panini, v.planet as f64);
    let hf = r(v.fov / 2.0);
    // planet: ρ(θ) = (e + 1)·sin θ / (e + cos θ), increasing on [0, acos(−e))
    let rho = |t: f64| (e + 1.0) * t.sin() / (e + t.cos());
    let f = match v.mode {
        Mode::Perspective => (d + 1.0) / (d + hf.cos()) * hf.sin(),
        Mode::Equirect => 1.0,
        Mode::Planet => rho(hf as f64) as f32,
    };
    let px = |x: isize, y: isize, k: usize| {
        pano[4 * (y.clamp(0, ph as isize - 1) as usize * pw + x.rem_euclid(pw as isize) as usize)
            + k] as f32
    };
    let mut o = vec![255u8; w * h * 4];
    for (i, p) in o.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let u = [
            ((i % w) as f32 + 0.5) / w as f32,
            ((i / w) as f32 + 0.5) / h as f32,
        ];
        let x = [
            (2.0 * u[0] - 1.0) * f,
            (1.0 - 2.0 * u[1]) * h as f32 / w as f32 * f,
        ];
        let ray = match v.mode {
            Mode::Equirect => {
                let (lon, lat) = (
                    (2.0 * u[0] - 1.0) * std::f32::consts::PI,
                    (0.5 - u[1]) * std::f32::consts::PI,
                );
                Vec3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos())
            }
            Mode::Planet => {
                // θ by bisection, independent of the shader's closed form
                let l = x[0].hypot(x[1]);
                let (mut lo, mut hi) = (0.0, (-e).acos());
                for _ in 0..60 {
                    let m = 0.5 * (lo + hi);
                    if rho(m) < l as f64 { lo = m } else { hi = m }
                }
                let t = (0.5 * (lo + hi)) as f32;
                let n = if l > 0.0 {
                    [x[0] / l, x[1] / l]
                } else {
                    [1.0, 0.0]
                };
                Vec3::new(t.sin() * n[0], t.sin() * n[1], t.cos())
            }
            Mode::Perspective => {
                let k = x[0] * x[0] / ((d + 1.0) * (d + 1.0));
                let cl =
                    (-k * d + (k * k * d * d - (k + 1.0) * (k * d * d - 1.0)).sqrt()) / (k + 1.0);
                let s = (d + 1.0) / (d + cl);
                let (lon, lat) = (x[0].atan2(s * cl), x[1].atan2(s));
                Vec3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos())
            }
        };
        // output world -> the stitcher's pixel: world = (sinφ cosθ, −sinφ sinθ, cosφ), x = θ·W/2π, y = φ·(H−1)/π
        let d = (Mat3::from_quat(q) * m * ray).normalize();
        let th = (-d.y).atan2(d.x).rem_euclid(std::f32::consts::TAU);
        let (sx, sy) = (
            th * pw as f32 / std::f32::consts::TAU,
            d.z.clamp(-1.0, 1.0).acos() * (ph - 1) as f32 / std::f32::consts::PI,
        );
        let (x0, y0) = (sx.floor(), sy.floor());
        let (fx, fy) = (sx - x0, sy - y0);
        let (x0, y0) = (x0 as isize, y0 as isize);
        for (k, o) in p[..3].iter_mut().enumerate() {
            let t = px(x0, y0, k) + fx * (px(x0 + 1, y0, k) - px(x0, y0, k));
            let b = px(x0, y0 + 1, k) + fx * (px(x0 + 1, y0 + 1, k) - px(x0, y0 + 1, k));
            *o = (t + fy * (b - t)).round() as u8;
        }
    }
    o
}

/// The faint ring the player's tiny-planet view shows on v635 (frame 100, pitch −90°, FOV 300°; app milestone 1) is
/// in the footage, not made by the stitch: an edge across the concrete ceiling overhang above the camera. It shows as
/// an arc in the lens's own picture (FFmpeg's decode of track 1, frame 100, high-pass filtered) whose middle lies
/// about 45° off the lens axis, where only that lens is drawn, far from the seam's blend band and the image-circle
/// feather. Colour matching
/// applies one gain and offset per lens, so it cannot draw an arc; here it shifts both sides of the ring alike, and
/// the ring is there with it off.
#[test]
fn ring() {
    let Some(c) = clip() else { return };
    let insv = samples().unwrap().join("v635.insv");
    let (Some(a), Some(b)) = (frame(&insv, 0, 100), frame(&insv, 1, 100)) else {
        return;
    };
    let nv = if c.meta.reverse { [b, a] } else { [a, b] };
    let g = gpu();
    let cx = Ctx::new(&g.dev, &g.q);
    let src = nv
        .each_ref()
        .map(|nv| upload(&g, &cx, nv, SIDE as u32, SIDE as u32));
    let src = [&src[0], &src[1]];
    let n = 1024usize;
    let t = texture(&g, n as u32, n as u32, wgpu::TextureFormat::Rgba8Unorm);
    let tv = t.create_view(&Default::default());
    let v = View {
        mode: Mode::Planet,
        pitch: -90.0,
        fov: 300.0,
        ..Default::default()
    };
    // mean grey just inside and just outside the ring (centre (450, 668), radius 188 px in this view), lower half
    let band = |px: &[u8], r0: f64| {
        let (mut s, mut k) = (0.0, 0);
        for (i, p) in px.chunks(4).enumerate() {
            let (x, y) = ((i % n) as f64, (i / n) as f64);
            if y > 600.0 && ((x - 450.0).hypot(y - 668.0) - r0).abs() < 3.0 {
                s += (p[0] as f64 + p[1] as f64 + p[2] as f64) / 3.0;
                k += 1;
            }
        }
        s / k as f64
    };
    let mut steps = vec![];
    for colour in [None, Some(1.0)] {
        let pv = Preview::new(
            &cx,
            &c.cal,
            c.fr,
            6.0,
            colour,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        pv.pose(&framing(&c.meta));
        let mut enc = g.dev.create_command_encoder(&Default::default());
        pv.colour(&mut enc, src);
        pv.render(&mut enc, &tv, [n as u32; 2], &v, src);
        g.q.submit([enc.finish()]);
        let px = stitch::read_texture(&g.dev, &g.q, &t);
        let (i, o) = (band(&px, 170.0), band(&px, 206.0));
        eprintln!(
            "colour {colour:?}: grey inside the ring {i:.2}, outside {o:.2}, step {:.2}",
            o - i
        );
        steps.push(o - i);
    }
    assert!(steps[0] > 1.5, "no ring in the footage");
    assert!(
        (steps[1] - steps[0]).abs() < 0.3,
        "colour matching changes the ring"
    );
}
