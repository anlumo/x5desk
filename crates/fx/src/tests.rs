//! GPU tests against a CPU reference of the same formulas (`reference`, and `bins` / `estimate` / `coefficients` for
//! the automatic corrections), ffmpeg's lut3d / lut1d (the CLI as a test tool), synthetic fringes, overlay geometry
//! and blending, temporal smoothing, in place against two textures, synthetic underwater frames, and real footage.
//! Footage tests read `X5_SAMPLES` (`v635.insv`, `v606.insv`, `v627.insv`) and skip when it is unset; their stitched
//! equirects come from ffmpeg's v360 (uncalibrated dual fisheye, enough for effects). Outputs go to `X5_OUT`
//! (default: the workspace's `target/fx/out`).

use super::*;
use glam::Vec3;
use std::{
    path::PathBuf,
    process::Command,
    sync::{Mutex, MutexGuard, OnceLock},
    time::Instant,
};

struct Gpu {
    dev: wgpu::Device,
    q: wgpu::Queue,
    cx: Ctx,
}

/// A device opened like `video::Gpu` (every adapter feature, all limits); one per test binary, locked so GPU tests
/// and their timings don't overlap.
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
                // SAFETY: the app's device has them on (video::Gpu); nothing here uses them
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            }),
        )
        .expect("device");
        eprintln!("GPU: {}", ad.get_info().name);
        let cx = Ctx::new(&dev, &q);
        Mutex::new(Gpu { dev, q, cx })
    })
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

fn out() -> PathBuf {
    let d = std::env::var_os("X5_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/fx/out"));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn samples() -> Option<PathBuf> {
    let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        eprintln!("X5_SAMPLES unset: skipped");
    }
    d
}

/// The stitched equirect of `<clip>.insv` 2 s in, `w` × `w/2`, made once with ffmpeg's v360 (193° dual fisheye).
fn still(clip: &str, w: u32) -> Option<Image> {
    let s = samples()?;
    let p = out().join(format!("{clip}_{w}.png"));
    if !p.exists() {
        let f = format!(
            "[0:v:0][0:v:1]hstack,v360=input=dfisheye:output=equirect:ih_fov=193:iv_fov=193:w={w}:h={}",
            w / 2
        );
        // written beside it and renamed: a parallel test that finds `p` must never read it half written
        let id = format!(
            "{}-{:?}.png",
            std::process::id(),
            std::thread::current().id()
        );
        let tmp = p.with_extension(id);
        let st = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-ss", "2", "-i"])
            .arg(s.join(format!("{clip}.insv")))
            .args(["-filter_complex", &f, "-frames:v", "1"])
            .arg(&tmp)
            .status()
            .expect("ffmpeg");
        assert!(st.success(), "ffmpeg v360");
        std::fs::rename(tmp, &p).unwrap();
    }
    Some(Image::png(&std::fs::read(&p).unwrap()).unwrap())
}

fn equirect(w: u32) -> Option<Image> {
    still("v635", w)
}

fn save(name: &str, w: u32, h: u32, px: &[[f32; 4]]) {
    let b: Vec<u8> = px
        .iter()
        .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
        .collect();
    let f = std::fs::File::create(out().join(name)).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    e.set_color(png::ColorType::Rgba);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&b).unwrap();
}

fn floats(i: &Image) -> Vec<[f32; 4]> {
    i.px.as_chunks::<4>()
        .0
        .iter()
        .map(|p| [0, 1, 2, 3].map(|k| p[k] as f32 / 255.0))
        .collect()
}

fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 112;
    let m = b & 0x7f_ffff;
    if e <= 0 {
        if e < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let sh = (14 - e) as u32;
        s | (((m >> sh) + ((m >> (sh - 1)) & 1)) as u16)
    } else if e >= 31 {
        s | 0x7c00
    } else {
        s | ((((e as u32) << 10) | (m >> 13)) + ((m >> 12) & 1)) as u16
    }
}

fn f32h(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    s * match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// A frame texture in `fmt` (`Rgba8Unorm` or `Rgba16Float`) usable as fx source and destination.
fn texture(
    g: &Gpu,
    w: u32,
    h: u32,
    fmt: wgpu::TextureFormat,
    px: Option<&[[f32; 4]]>,
) -> wgpu::Texture {
    use wgpu::TextureUsages as U;
    let t = tex(
        &g.dev,
        [w, h, 1],
        wgpu::TextureDimension::D2,
        fmt,
        U::TEXTURE_BINDING | U::STORAGE_BINDING | U::RENDER_ATTACHMENT | U::COPY_DST | U::COPY_SRC,
        1,
    );
    if let Some(px) = px {
        let (b, bpp): (Vec<u8>, u32) = if fmt == wgpu::TextureFormat::Rgba8Unorm {
            (
                px.iter()
                    .flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
                    .collect(),
                4,
            )
        } else {
            (
                px.iter()
                    .flat_map(|p| p.map(f16))
                    .flat_map(u16::to_le_bytes)
                    .collect(),
                8,
            )
        };
        g.q.write_texture(
            t.as_image_copy(),
            &b,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpp * w),
                rows_per_image: Some(h),
            },
            t.size(),
        );
    }
    t
}

/// What a texture of `fmt` stores for `px` (8-bit or half-float rounding).
fn stored(fmt: wgpu::TextureFormat, px: &[[f32; 4]]) -> Vec<[f32; 4]> {
    px.iter()
        .map(|p| {
            p.map(|v| {
                if fmt == wgpu::TextureFormat::Rgba8Unorm {
                    (v.clamp(0.0, 1.0) * 255.0).round() / 255.0
                } else {
                    f32h(f16(v))
                }
            })
        })
        .collect()
}

fn read(g: &Gpu, t: &wgpu::Texture) -> Vec<[f32; 4]> {
    let (w, h) = (t.width(), t.height());
    let half = t.format() == wgpu::TextureFormat::Rgba16Float;
    let bpp = if half { 8 } else { 4 };
    let row = (bpp * w).next_multiple_of(256);
    let b = g.dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.dev.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        t.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &b,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        t.size(),
    );
    g.q.submit([enc.finish()]);
    b.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    g.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let m = b.slice(..).get_mapped_range().unwrap();
    let mut v = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        let r = &m[(y * row) as usize..][..(bpp * w) as usize];
        if half {
            v.extend(
                r.as_chunks::<8>().0.iter().map(|p| {
                    [0, 1, 2, 3].map(|k| f32h(u16::from_le_bytes([p[2 * k], p[2 * k + 1]])))
                }),
            );
        } else {
            v.extend(
                r.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| [0, 1, 2, 3].map(|k| p[k] as f32 / 255.0)),
            );
        }
    }
    v
}

fn go(g: &Gpu, fx: &mut Fx, io: &Io, ov: &[&Overlay]) {
    let mut enc = g.dev.create_command_encoder(&Default::default());
    fx.run(&mut enc, io, ov);
    g.q.submit([enc.finish()]);
    g.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

const W3: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

fn lin(c: Vec3) -> Vec3 {
    c.to_array().map(cube::lin).into()
}

fn enc(l: Vec3) -> Vec3 {
    l.to_array().map(cube::enc).into()
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Tetrahedral 3D LUT lookup, as `fx.wgsl` `cube`.
fn tetra(l: &Lut, c: Vec3) -> Vec3 {
    let n = l.n as f32;
    let (lo, hi) = (Vec3::from(l.min), Vec3::from(l.max));
    let u = ((c - lo) / (hi - lo)).clamp(Vec3::ZERO, Vec3::ONE) * (n - 1.0);
    let i = u.floor().min(Vec3::splat(n - 2.0));
    let f = u - i;
    let at = |d: [u32; 3]| {
        let [x, y, z] = [0, 1, 2].map(|k| i[k] as u32 + d[k]);
        Vec3::from(l.v[(x + l.n * (y + l.n * z)) as usize])
    };
    let (a, b, w) = if f.x > f.y {
        if f.y > f.z {
            ([1, 0, 0], [1, 1, 0], [1.0 - f.x, f.x - f.y, f.y - f.z, f.z])
        } else if f.x > f.z {
            ([1, 0, 0], [1, 0, 1], [1.0 - f.x, f.x - f.z, f.z - f.y, f.y])
        } else {
            ([0, 0, 1], [1, 0, 1], [1.0 - f.z, f.z - f.x, f.x - f.y, f.y])
        }
    } else if f.z > f.y {
        ([0, 0, 1], [0, 1, 1], [1.0 - f.z, f.z - f.y, f.y - f.x, f.x])
    } else if f.z > f.x {
        ([0, 1, 0], [0, 1, 1], [1.0 - f.y, f.y - f.z, f.z - f.x, f.x])
    } else {
        ([0, 1, 0], [1, 1, 0], [1.0 - f.y, f.y - f.x, f.x - f.z, f.z])
    };
    w[0] * at([0, 0, 0]) + w[1] * at(a) + w[2] * at(b) + w[3] * at([1, 1, 1])
}

/// 1D LUT lookup, linear between entries.
fn tab(l: &Lut, c: Vec3) -> Vec3 {
    let n = l.n as f32;
    Vec3::from_array(std::array::from_fn(|j| {
        let u = ((c[j] - l.min[j]) / (l.max[j] - l.min[j])).clamp(0.0, 1.0) * (n - 1.0);
        let i = u.floor().min(n - 2.0);
        let (f, k) = (u - i, i as usize);
        l.v[k][j] * (1.0 - f) + l.v[k + 1][j] * f
    }))
}

/// CPU reference of the documented formulas (`Params`, the crate docs) straight from the parameters and tables, so
/// the GPU's uniform packing is checked too; on what the source texture stores. Only `wb` (the white-balance matrix)
/// is shared with the GPU path; the automatic corrections' coefficients (`k` = `Fx::kb`, 16…24) come from the GPU, and
/// `estimate` / `coefficients` check those separately.
fn reference(
    pa: &Params,
    inp: Option<&Lut>,
    look: Option<&Cube>,
    k: Option<&[f32]>,
    (w, h, wrap): (u32, u32, bool),
    src: &[[f32; 4]],
) -> Vec<[f32; 4]> {
    let c1 = |v: f32| v.clamp(-1.0, 1.0);
    let u1 = |v: f32| v.clamp(0.0, 1.0);
    let bri = c1(pa.brilliance);
    let (cl, sh, pu, gr, bu) = (
        c1(pa.clarity + 0.3 * bri),
        u1(pa.sharpen),
        u1(pa.purple),
        u1(pa.green),
        u1(pa.blue),
    );
    let (hl, sd, ct, wh, bl) = (
        c1(pa.highlights - 0.5 * bri),
        c1(pa.shadows + 0.5 * bri),
        c1(pa.contrast),
        c1(pa.whites),
        c1(pa.blacks),
    );
    let (sat, vib, mix) = (c1(pa.saturation), c1(pa.vibrance), u1(pa.mix));
    let gam = (-0.6 * c1(pa.brightness)).exp2();
    let rr = (pa.reach.round() as i32).clamp(1, 12);
    let au = k.filter(|_| pa.water > 0.0 || pa.auto > 0.0);
    let n = ((w as f32 / 180.0).log2().round() as i32).clamp(1, 8) as usize;
    let (w, h) = (w as i32, h as i32);
    let ix = move |x: i32, w: i32| {
        if wrap {
            x.rem_euclid(w)
        } else {
            x.clamp(0, w - 1)
        }
    };
    let at: Vec<[f32; 4]> = src
        .iter()
        .map(|p| {
            let mut c = Vec3::new(p[0], p[1], p[2]);
            if let Some(l) = inp {
                c = tab(l, c);
            }
            [c.x, c.y, c.z, c.dot(W3)]
        })
        .collect();
    let px = |x: i32, y: i32| at[(y.clamp(0, h - 1) * w + ix(x, w)) as usize];
    let ld = |t: &[f32], d: (i32, i32), x: i32, y: i32| {
        t[(y.clamp(0, d.1 - 1) * d.0 + ix(x, d.0)) as usize]
    };
    let up2 = |t: &[f32], d: (i32, i32), x: i32, y: i32| {
        let (i, j) = (x / 2, y / 2);
        let (a, b) = (
            if x & 1 == 1 { i + 1 } else { i - 1 },
            if y & 1 == 1 { j + 1 } else { j - 1 },
        );
        0.5625 * ld(t, d, i, j)
            + 0.1875 * (ld(t, d, a, j) + ld(t, d, i, b))
            + 0.0625 * ld(t, d, a, b)
    };
    let mut s = vec![(w, h)];
    for k in 0..n {
        s.push(((s[k].0 + 1) / 2, (s[k].1 + 1) / 2));
    }
    let base = (cl != 0.0).then(|| {
        let (w1, h1) = s[1];
        let mut p: Vec<Vec<f32>> = vec![
            (0..w1 * h1)
                .map(|i| {
                    let (x, y) = (i % w1 * 2, i / w1 * 2);
                    (px(x, y)[3] + px(x + 1, y)[3] + px(x, y + 1)[3] + px(x + 1, y + 1)[3]) / 4.0
                })
                .collect(),
        ];
        let k4 = [1.0, 3.0, 3.0, 1.0];
        for k in 1..n {
            let (ow, oh) = s[k + 1];
            let v = (0..ow * oh)
                .map(|i| {
                    let (x, y) = (i % ow * 2 - 1, i / ow * 2 - 1);
                    let mut v = 0.0;
                    for (j, kj) in k4.iter().enumerate() {
                        let mut r = 0.0;
                        for (ii, ki) in k4.iter().enumerate() {
                            r += ki * ld(&p[k - 1], s[k], x + ii as i32, y + j as i32);
                        }
                        v += kj * r;
                    }
                    v / 64.0
                })
                .collect();
            p.push(v);
        }
        let mut q = p[n - 1].clone();
        for k in (0..n - 1).rev() {
            let (ow, oh) = s[k + 1];
            q = (0..ow * oh)
                .map(|i| up2(&q, s[k + 2], i % ow, i / ow))
                .collect();
        }
        q
    });
    let m = wb(c1(pa.temperature), c1(pa.tint)) * 2f64.powf(pa.exposure.clamp(-5.0, 5.0) as f64);
    let m = [0, 1, 2].map(|r| m.row(r).as_vec3());
    let band = |hu: f32, c: f32, i: f32, o: f32| {
        let d = ((hu - c) / 360.0 + 0.5).rem_euclid(1.0) - 0.5;
        1.0 - smooth(i, o, d.abs() * 360.0)
    };
    let (kct, kvb, khl) = au.map_or((0.0, 0.0, 0.0), |k| (k[23], k[24], k[25]));
    let (ct, vb, hl) = (
        (ct + kct).clamp(-1.0, 1.0),
        (vib + kvb).clamp(-1.0, 1.0),
        (hl + khl).clamp(-1.0, 1.0),
    );
    let mut o = vec![[0.0; 4]; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let s0 = px(x, y);
            let mut c = Vec3::new(s0[0], s0[1], s0[2]);
            let l = s0[3];
            if pu > 0.0 || gr > 0.0 || bu > 0.0 {
                let (mut lo, mut hi) = (l, l);
                for dy in -rr..=rr {
                    for dx in -rr..=rr {
                        let v = px(x + dx, y + dy)[3];
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                let e = smooth(0.25, 0.6, hi - lo);
                let (mx, mn) = (c.max_element(), c.min_element());
                let d = mx - mn;
                let mut hu = 0.0;
                if d > 0.0 {
                    hu = if mx == c.x {
                        (c.y - c.z) / d
                    } else if mx == c.y {
                        (c.z - c.x) / d + 2.0
                    } else {
                        (c.x - c.y) / d + 4.0
                    };
                    hu = (hu / 6.0 - (hu / 6.0).floor()) * 360.0;
                }
                let a = (e
                    * (smooth(0.6, 0.9, hi)
                        * (pu * band(hu, 285.0, 30.0, 50.0) + gr * band(hu, 120.0, 25.0, 45.0))
                        + smooth(0.85, 0.97, hi) * bu * band(hu, 220.0, 15.0, 30.0)))
                .min(1.0);
                c = l + (c - l) * (1.0 - a);
            }
            let mut dl = 0.0;
            if sh > 0.0 {
                let n = |dx: i32, dy: i32| px(x + dx, y + dy)[3];
                let b = (4.0 * l
                    + 2.0 * (n(-1, 0) + n(1, 0) + n(0, -1) + n(0, 1))
                    + n(-1, -1)
                    + n(1, -1)
                    + n(-1, 1)
                    + n(1, 1))
                    / 16.0;
                dl = 2.0 * sh * (l - b);
            }
            if let Some(b) = &base {
                let t = l.clamp(0.0, 1.0);
                dl += 0.8 * cl * (l - up2(b, s[1], x, y)) * 4.0 * t * (1.0 - t);
            }
            c = (c + dl).max(Vec3::ZERO);
            let mut v = lin(c);
            if let Some(k) = au {
                v.x += k[16] * (1.0 - v.x).max(0.0) * v.y;
                v = (v * Vec3::new(k[17], k[18], k[19]) - Vec3::new(k[20], k[21], k[22]))
                    .max(Vec3::ZERO);
            }
            v = Vec3::new(m[0].dot(v), m[1].dot(v), m[2].dot(v)).max(Vec3::ZERO);
            if hl != 0.0 || sd != 0.0 {
                let ev = (v.dot(W3).max(1e-6) / 0.18).log2();
                let s = (ev + 2.25) / 1.5;
                let t = (ev - 1.5) / 1.25;
                v *= (hl * (-0.5 * t * t).exp() + 1.25 * sd * (-0.5 * s * s).exp()).exp2();
            }
            let mut e = enc(v);
            if ct != 0.0 || wh != 0.0 || bl != 0.0 {
                let mut t = e.clamp(Vec3::ZERO, Vec3::ONE);
                e -= ct * t * (1.0 - t) * (1.0 - 2.0 * t);
                t = e.clamp(Vec3::ZERO, Vec3::ONE);
                e += 0.25 * wh * t * t * t * t;
                t = 1.0 - e.clamp(Vec3::ZERO, Vec3::ONE);
                e += 0.15 * bl * t * t * t * t;
            }
            if gam != 1.0 {
                e = e.max(Vec3::ZERO).powf(gam);
            }
            if sat != 0.0 || vb != 0.0 {
                v = lin(e);
                let yl = v.dot(W3);
                let mx = v.max_element();
                let s = (mx - v.min_element()) / mx.max(1e-6);
                e = enc(yl + (v - yl) * (1.0 + sat) * (1.0 + vb * (1.0 - s)));
            }
            let mut lk = e;
            if let Some(l) = look.and_then(|c| c.d1.as_ref()) {
                lk = tab(l, lk);
            }
            if let Some(l) = look.and_then(|c| c.d3.as_ref()) {
                lk = tetra(l, lk);
            }
            e = (e * (1.0 - mix) + lk * mix).max(Vec3::ZERO);
            o[(y * w + x) as usize] = [e.x, e.y, e.z, src[(y * w + x) as usize][3]];
        }
    }
    o
}

/// The statistics `fx.wgsl` `stat` sums (64 bins × weight, Σr, Σg, Σb, Σr·g, Σs), exactly, on the stored source.
fn bins(src: &[[f32; 4]], w: u32, h: u32, pano: bool) -> Vec<u64> {
    let q = 262143.0;
    let st = w.max(h).div_ceil(512).max(1);
    let mut z = vec![0u64; 384];
    for gy in 0..h.div_ceil(st) {
        for gx in 0..w.div_ceil(st) {
            let (x, y) = ((gx * st + st / 2).min(w - 1), (gy * st + st / 2).min(h - 1));
            let p = src[(y * w + x) as usize];
            let c = Vec3::new(p[0], p[1], p[2]);
            let v = lin(c).clamp(Vec3::ZERO, Vec3::ONE);
            let e = c.clamp(Vec3::ZERO, Vec3::ONE);
            let s = (e.max_element() - e.min_element()) / e.max_element().max(1e-6);
            let wq = if pano {
                (64.0 * (std::f32::consts::PI * (y as f32 + 0.5) / h as f32).sin()).round() as u64
            } else {
                64
            };
            let k = ((c.dot(W3).clamp(0.0, 1.0) * 64.0) as usize).min(63) * 6;
            for (j, x) in [1.0 / q, v.x, v.y, v.z, v.x * v.y, s].iter().enumerate() {
                z[k + j] += wq * (x * q).round() as u64;
            }
        }
    }
    z
}

/// CPU mirror of `fx.wgsl` `solve`'s estimate (`Fx::kb` 0…14) from the summed statistics, flags 16 underwater, 32 auto.
fn estimate(z: &[f32], fl: u32) -> [f32; 15] {
    let q = 262143.0;
    let mut bn = [0f32; 64];
    let mut bm = [Vec3::ZERO; 64];
    let mut bq = [0f32; 64];
    let mut by = [0f32; 64];
    let (mut n, mut m, mut sat) = (0f32, Vec3::ZERO, 0f32);
    for b in 0..64 {
        let k = 6 * b;
        let s = Vec3::new(z[k + 1], z[k + 2], z[k + 3]) / q;
        bn[b] = z[k];
        bm[b] = s / z[k].max(1e-9);
        bq[b] = z[k + 4] / (q * z[k].max(1e-9));
        n += z[k];
        m += s;
        sat += z[k + 5] / q;
    }
    n = n.max(1e-9);
    m /= n;
    let share = |bn: &[f32; 64], bm: &[Vec3; 64], f: f32, up: bool, d: Vec3| {
        let (mut s, mut w) = (Vec3::ZERO, 0.0);
        for i in 0..63 {
            let r = f * n - w;
            if r <= 0.0 {
                break;
            }
            let b = if up { i } else { 62 - i };
            let x = bn[b].min(r);
            s += x * bm[b];
            w += x;
        }
        if w > 0.0 { s / w } else { d }
    };
    let pct = |bn: &[f32; 64], by: &[f32; 64], f: f32| {
        let mut w = 0.0;
        for b in 0..64 {
            w += bn[b];
            if w >= f * n && bn[b] > 0.0 {
                return by[b];
            }
        }
        by[63]
    };
    let mut e = [
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    if fl & 16 != 0 {
        let k = 0.5 * (m.y - m.x).max(0.0);
        let mut mc = Vec3::ZERO;
        for b in 0..64 {
            bm[b].x += k * (bm[b].y - bq[b]);
            mc += bn[b] * bm[b];
        }
        mc /= n;
        let v = share(&bn, &bm, 0.01, true, mc);
        let a = (mc - v).max(Vec3::splat(1e-4));
        let hh = (share(&bn, &bm, 0.01, false, mc) - v).max(Vec3::splat(1e-4));
        let r = 1.0 / (a.powf(0.25) * hh.powf(0.75));
        let gw = (r / r.min_element()).min(Vec3::splat(8.0));
        for m in &mut bm {
            *m = ((*m - v) * gw).max(Vec3::ZERO);
        }
        e[..7].copy_from_slice(&[k, v.x, v.y, v.z, gw.x, gw.y, gw.z]);
    }
    if fl & 32 != 0 {
        let mut mg = Vec3::ZERO;
        for b in 0..64 {
            mg += bn[b] * bm[b];
        }
        mg /= n;
        let il = (mg.max(Vec3::splat(1e-5))
            * share(&bn, &bm, 0.02, false, mg).max(Vec3::splat(1e-5)))
        .powf(0.5);
        let yi = il.dot(W3);
        let mut ga = (yi / il)
            .powf(0.5)
            .clamp(Vec3::splat(0.8), Vec3::splat(1.25));
        ga *= yi / (ga * il).dot(W3);
        for b in 0..64 {
            by[b] = (ga * bm[b]).dot(W3);
        }
        let bk = 0.5 * pct(&bn, &by, 0.005);
        let wt = pct(&bn, &by, 0.995);
        let lk: f32 = (0..64).map(|b| bn[b] * (by[b] - bk).max(1e-4).log2()).sum();
        let cl = (wt.max(1.0) / (wt - bk).max(1e-4)).log2().max(0.0);
        let ev = (0.5 * (0.18f32.log2() - lk / n))
            .clamp(-1.0, 1.0)
            .min(cl + 0.74);
        let hl = -((ev - cl) / 0.74).clamp(0.0, 1.0);
        let (mut l1, mut l2) = (0.0, 0.0);
        for b in 0..64 {
            let y = (by[b] - bk).max(0.0) * ev.exp2();
            let t = ((y.max(1e-6) / 0.18).log2() - 1.5) / 1.25;
            let l = cube::enc(y * (hl * (-0.5 * t * t).exp()).exp2());
            l1 += bn[b] * l;
            l2 += bn[b] * l * l;
        }
        l1 /= n;
        let sd = (l2 / n - l1 * l1).max(0.0).sqrt();
        e[7..].copy_from_slice(&[
            ga.x,
            ga.y,
            ga.z,
            bk,
            ev,
            (2.0 * (0.25 - sd)).clamp(0.0, 0.3),
            (2.0 * (0.35 - sat / n)).clamp(0.0, 0.5),
            hl,
        ]);
    }
    e
}

/// CPU mirror of what `solve` makes of the smoothed estimate `e` at strengths `sw` (underwater), `sa` (auto):
/// `Fx::kb` 16…25.
fn coefficients(e: &[f32], sw: f32, sa: f32) -> [f32; 10] {
    let gw = Vec3::new(e[4], e[5], e[6]).powf(sw);
    let g2 = Vec3::new(e[7], e[8], e[9]).powf(sa) * (sa * e[11]).exp2();
    let g = gw * g2;
    let o = (sw * Vec3::new(e[1], e[2], e[3]) * gw + sa * e[10]) * g2;
    [
        sw * e[0],
        g.x,
        g.y,
        g.z,
        o.x,
        o.y,
        o.z,
        sa * e[12],
        sa * e[13],
        sa * e[14],
    ]
}

/// A buffer's f32s.
fn floats_of(g: &Gpu, b: &wgpu::Buffer) -> Vec<f32> {
    let r = g.dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: b.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.dev.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(b, 0, &r, 0, b.size());
    g.q.submit([enc.finish()]);
    r.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    g.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice(&r.slice(..).get_mapped_range().unwrap()).to_vec()
}

/// Synthetic test picture: hue × saturation sweep, grey ramp with fine detail, random blocks (edges), hue × value.
fn synth(w: u32, h: u32) -> Vec<[f32; 4]> {
    let hsv = |hu: f32, s: f32, v: f32| {
        let k = |n: f32| {
            let k = (n + hu / 60.0) % 6.0;
            v - v * s * k.min(4.0 - k).clamp(0.0, 1.0)
        };
        [k(5.0), k(3.0), k(1.0)]
    };
    let hash = |x: u32, y: u32| {
        let mut v = x.wrapping_mul(0x9e37_79b9) ^ y.wrapping_mul(0x85eb_ca6b);
        v ^= v >> 15;
        v = v.wrapping_mul(0x2c1b_3c6d);
        v ^= v >> 12;
        v
    };
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let (fx, b, fy) = (
                x as f32 / (w - 1) as f32,
                4 * y / h,
                (4 * y % h) as f32 / h as f32,
            );
            let c = match b {
                0 => hsv(360.0 * fx, 1.0 - fy, 0.95),
                1 => {
                    let v = fx + 0.08 * ((x as f32 * 0.9).sin() * (y as f32 * 0.7).cos());
                    [v.clamp(0.0, 1.0); 3]
                }
                2 => {
                    let r = hash(x / 9, y / 7);
                    [r & 255, (r >> 8) & 255, (r >> 16) & 255].map(|v| v as f32 / 255.0)
                }
                _ => hsv(360.0 * fy, 0.7, fx),
            };
            [c[0], c[1], c[2], 1.0]
        })
        .collect()
}

/// Test picture: the stitched frame on top (when footage is there), the synthetic bands below.
fn picture(w: u32) -> (u32, Vec<[f32; 4]>) {
    let mut px = match equirect(w) {
        Some(i) => floats(&i),
        None => synth(w, w / 2),
    };
    px.extend(synth(w, w / 2));
    (w, px)
}

/// Max |GPU − CPU| in output levels on the display range (8-bit: against the rounded reference; half float: values
/// clamped to 0…1 in 10-bit levels, as a 10-bit export keeps them), the share of values off by more than one level,
/// and for half float the max error in f16 steps over all values (above 1 too).
fn compare(fmt: wgpu::TextureFormat, a: &[[f32; 4]], b: &[[f32; 4]]) -> (f32, f64, f32) {
    let eight = fmt == wgpu::TextureFormat::Rgba8Unorm;
    let (mut m, mut n, mut ulp) = (0f32, 0usize, 0f32);
    for (p, q) in a.iter().zip(b) {
        for k in 0..3 {
            let d = if eight {
                (p[k] * 255.0 - (q[k].clamp(0.0, 1.0) * 255.0).round()).abs()
            } else {
                let r = f32h(f16(q[k]));
                let step = 2f32.powi(r.abs().max(2f32.powi(-14)).log2().floor() as i32 - 10);
                ulp = ulp.max((p[k] - r).abs() / step);
                (p[k].clamp(0.0, 1.0) - q[k].clamp(0.0, 1.0)).abs() * 1023.0
            };
            m = m.max(d);
            n += (d > 1.0) as usize;
        }
    }
    (m, n as f64 / (a.len() * 3) as f64, ulp)
}

#[test]
fn cube_parse() {
    let c = Cube::parse(
        "# comment\nTITLE \"test\"\nLUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 2 1\n\
         0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n",
    )
    .unwrap();
    assert_eq!(c.title, "test");
    let d = c.d3.as_ref().unwrap();
    assert_eq!((d.n, d.max, d.v[1]), (2, [1.0, 2.0, 1.0], [1.0, 0.0, 0.0]));
    assert!(c.d1.is_none());
    let c = Cube::parse("LUT_1D_SIZE 3\nLUT_1D_INPUT_RANGE -0.5 1.5\n0 0 0\n0.5 0.5 0.5\n1 1 1\n")
        .unwrap();
    assert_eq!(c.d1.as_ref().unwrap().min, [-0.5; 3]);
    // a UTF-8 BOM (Windows tools) and CRLF line ends
    let c = Cube::parse("\u{feff}LUT_1D_SIZE 2\r\n0 0 0\r\n1 1 1\r\n").unwrap();
    assert_eq!(c.d1.unwrap().v, [[0.0; 3], [1.0; 3]]);
    // shaper + cube: DOMAIN goes to the 1D table, the 3D one takes its own range
    let both = Cube::parse(&format!(
        "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\nDOMAIN_MAX 4 4 4\nLUT_3D_INPUT_RANGE 0 2\n0 0 0\n1 1 1\n{}",
        "0.5 0.5 0.5\n".repeat(8)
    ))
    .unwrap();
    assert_eq!(both.d1.as_ref().unwrap().max, [4.0; 3]);
    assert_eq!(both.d3.as_ref().unwrap().max, [2.0; 3]);
    assert_eq!(Cube::parse(&both.to_string()).unwrap(), both);
    for bad in [
        "LUT_3D_SIZE 2\n0 0 0\n",
        "LUT_3D_SIZE 1\n0 0 0\n",
        "LUT_3D_SIZE 257\n",
        "0 0 0\n",
        "LUT_1D_SIZE 2\n0 0\n1 1 1\n",
        "LUT_1D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 1 1 1\n0 0 0\n1 1 1\n",
        "LUT_1D_SIZE 2\n0 0 x\n1 1 1\n",
    ] {
        assert!(Cube::parse(bad).is_err(), "{bad:?}");
    }
    let film = Look::Film.cube();
    assert_eq!(Cube::parse(&film.to_string()).unwrap().d3.unwrap().n, 33);
    let big = Grade::default().cube("id", 65);
    let p = Cube::parse(&big.to_string()).unwrap().d3.unwrap();
    assert_eq!(p.v.len(), 65 * 65 * 65);
    assert!(
        p.v.iter()
            .zip(&big.d3.unwrap().v)
            .all(|(a, b)| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-6))
    );
}

/// PNG colour types other than RGBA8: palette with transparency, 16-bit grey + alpha, RGB.
#[test]
fn png_types() {
    let enc = |ct, bd, data: &[u8], f: &dyn Fn(&mut png::Encoder<&mut Vec<u8>>)| {
        let mut v = vec![];
        let mut e = png::Encoder::new(&mut v, 2, 1);
        e.set_color(ct);
        e.set_depth(bd);
        f(&mut e);
        e.write_header().unwrap().write_image_data(data).unwrap();
        Image::png(&v).unwrap().px
    };
    use png::{BitDepth as D, ColorType as C};
    let pal = enc(C::Indexed, D::Eight, &[1, 0], &|e| {
        e.set_palette(vec![10, 20, 30, 200, 100, 50]);
        e.set_trns(vec![128, 255]);
    });
    assert_eq!(pal, [200, 100, 50, 255, 10, 20, 30, 128]);
    let g16 = enc(
        C::GrayscaleAlpha,
        D::Sixteen,
        &[0x80, 0, 0xff, 0xff, 0, 0, 0x40, 0],
        &|_| {},
    );
    assert_eq!(g16, [128, 128, 128, 255, 0, 0, 0, 64]);
    let rgb = enc(C::Rgb, D::Eight, &[1, 2, 3, 4, 5, 6], &|_| {});
    assert_eq!(rgb, [1, 2, 3, 255, 4, 5, 6, 255]);
}

/// Parameter semantics on the CPU: neutral = identity, white balance direction and white preservation, monotone tone
/// curves at every extreme, log curves invert, the Rec.709 transform keeps mid grey.
#[test]
fn semantics() {
    assert_eq!(wb(0.0, 0.0), DMat3::IDENTITY);
    let g = |t: f32, n: f32| wb(t, n) * DVec3::ONE;
    let (warm, cool, mag) = (g(1.0, 0.0), g(-1.0, 0.0), g(0.0, 1.0));
    eprintln!("white → temperature +1 {warm:.3}, −1 {cool:.3}, tint +1 {mag:.3}");
    assert!(warm.x > warm.z && cool.z > cool.x && mag.y < mag.x.min(mag.z));
    for v in [warm, cool, mag] {
        assert!((W.dot(v) - 1.0).abs() < 1e-9);
    }
    // tone curve per channel and band gains: monotone over 0…1 at every corner of the sliders
    for k in 0..243 {
        let s = [0, 1, 2, 3, 4].map(|i| (k / 3usize.pow(i) % 3) as f32 - 1.0);
        let (c, wh, bl, hi, sh) = (s[0], 0.25 * s[1], 0.15 * s[2], s[3], 1.25 * s[4]);
        let mut prev = (f32::MIN, f32::MIN);
        for i in 0..=1000 {
            let x = i as f32 / 1000.0;
            let mut e = x;
            let mut t = e.clamp(0.0, 1.0);
            e -= c * t * (1.0 - t) * (1.0 - 2.0 * t);
            t = e.clamp(0.0, 1.0);
            e += wh * t.powi(4);
            t = 1.0 - e.clamp(0.0, 1.0);
            e += bl * t.powi(4);
            let y = cube::lin(x);
            let ev = (y.max(1e-6) / 0.18).log2();
            let z = y
                * (hi * (-0.5 * ((ev - 1.5) / 1.25).powi(2)).exp()
                    + sh * (-0.5 * ((ev + 2.25) / 1.5).powi(2)).exp())
                .exp2();
            assert!(
                e >= prev.0 - 1e-6 && z >= prev.1 - 1e-6,
                "not monotone at {x} for {s:?}"
            );
            prev = (e, z);
        }
    }
    for l in [Log::LOGC3, Log::SLOG3, Log::VLOG] {
        for x in [0.0, 0.005, 0.18, 1.0, 10.0] {
            assert!(
                (l.decode(l.encode(x)) - x).abs() < 1e-4 * (1.0 + x),
                "{l:?} {x}"
            );
        }
        let t = l.rec709(0.0);
        let v = l.encode(0.18);
        let i = (v * (t.n - 1) as f32).round() as usize;
        eprintln!(
            "mid grey {v:.4} → Rec.709 {:.4} (BT.709 OETF of 0.18: 0.409)",
            t.v[i][0]
        );
        assert!((t.v[i][0] - 0.409).abs() < 0.01);
        assert!(t.v.windows(2).all(|p| p[1][0] >= p[0][0]));
    }
    assert!((Log::SLOG3.encode(0.18) - 420.0 / 1023.0).abs() < 1e-5);
}

/// Every adjustment, the LUT stages and their combination on the GPU against the CPU reference, 8-bit and half-float
/// frames.
#[test]
fn gpu_matches_cpu() {
    let g = gpu();
    let (w, px) = picture(1024);
    let h = px.len() as u32 / w;
    let film = Look::Film.cube();
    let p = |f: &dyn Fn(&mut Params)| {
        let mut p = Params::default();
        f(&mut p);
        p
    };
    let mut l = Grade {
        sat: 0.6,
        contrast: -0.5,
        ..Grade::default()
    }
    .cube("", 17)
    .d3
    .unwrap();
    for (i, v) in l.v.iter_mut().enumerate() {
        let t = i as f32 * 0.37;
        *v = [
            v[0] + 0.1 * t.sin(),
            v[1] * v[1],
            1.0 - v[2] * 0.8 + 0.05 * t.cos(),
        ];
    }
    l.min = [-0.1, 0.05, 0.0];
    l.max = [1.2, 0.9, 1.5];
    let wild = Cube {
        title: "wild".into(),
        d1: None,
        d3: Some(l),
    };
    let one = Cube {
        title: String::new(),
        d1: Some(Lut {
            n: 256,
            min: [0.0; 3],
            max: [1.0; 3],
            v: (0..256)
                .map(|i| {
                    let x = i as f32 / 255.0;
                    [x.powf(0.8), x * x, 1.0 - x]
                })
                .collect(),
        }),
        d3: None,
    };
    let slog = Log::SLOG3.rec709(0.5);
    type Case<'a> = (&'a str, Params, Option<&'a Cube>, Option<&'a Lut>);
    let cases: Vec<Case> = vec![
        ("neutral", Params::default(), None, None),
        ("exposure +1.5", p(&|p| p.exposure = 1.5), None, None),
        ("exposure −2", p(&|p| p.exposure = -2.0), None, None),
        ("temperature +1", p(&|p| p.temperature = 1.0), None, None),
        ("temperature −1", p(&|p| p.temperature = -1.0), None, None),
        ("tint +1", p(&|p| p.tint = 1.0), None, None),
        ("tint −1", p(&|p| p.tint = -1.0), None, None),
        ("highlights +1", p(&|p| p.highlights = 1.0), None, None),
        ("highlights −1", p(&|p| p.highlights = -1.0), None, None),
        ("shadows +1", p(&|p| p.shadows = 1.0), None, None),
        ("shadows −1", p(&|p| p.shadows = -1.0), None, None),
        ("contrast +1", p(&|p| p.contrast = 1.0), None, None),
        ("contrast −1", p(&|p| p.contrast = -1.0), None, None),
        ("whites +1", p(&|p| p.whites = 1.0), None, None),
        ("whites −1", p(&|p| p.whites = -1.0), None, None),
        ("blacks +1", p(&|p| p.blacks = 1.0), None, None),
        ("blacks −1", p(&|p| p.blacks = -1.0), None, None),
        ("saturation +1", p(&|p| p.saturation = 1.0), None, None),
        ("saturation −1", p(&|p| p.saturation = -1.0), None, None),
        ("vibrance +1", p(&|p| p.vibrance = 1.0), None, None),
        ("vibrance −1", p(&|p| p.vibrance = -1.0), None, None),
        ("clarity +1", p(&|p| p.clarity = 1.0), None, None),
        ("clarity −1", p(&|p| p.clarity = -1.0), None, None),
        ("sharpen 1", p(&|p| p.sharpen = 1.0), None, None),
        (
            "defringe",
            p(&|p| {
                p.purple = 1.0;
                p.green = 1.0
            }),
            None,
            None,
        ),
        ("look Film 3D", Params::default(), Some(&film), None),
        ("look Film mix 0.4", p(&|p| p.mix = 0.4), Some(&film), None),
        ("look 17³ domain", Params::default(), Some(&wild), None),
        ("look 1D", Params::default(), Some(&one), None),
        ("input S-Log3", Params::default(), None, Some(&slog)),
        ("brightness +1", p(&|p| p.brightness = 1.0), None, None),
        ("brightness −1", p(&|p| p.brightness = -1.0), None, None),
        ("brilliance +1", p(&|p| p.brilliance = 1.0), None, None),
        ("brilliance −1", p(&|p| p.brilliance = -1.0), None, None),
        ("blue fringe", p(&|p| p.blue = 1.0), None, None),
        (
            "defringe reach 12",
            p(&|p| {
                p.purple = 1.0;
                p.blue = 0.7;
                p.reach = 12.0
            }),
            None,
            None,
        ),
        (
            "sharpen reach 1",
            p(&|p| {
                p.sharpen = 1.0;
                p.green = 0.5;
                p.reach = 1.0
            }),
            None,
            None,
        ),
        ("underwater 1", p(&|p| p.water = 1.0), None, None),
        ("underwater 0.5", p(&|p| p.water = 0.5), None, None),
        ("auto 1", p(&|p| p.auto = 1.0), None, None),
        (
            "underwater + auto",
            p(&|p| {
                p.water = 0.8;
                p.auto = 0.6
            }),
            None,
            None,
        ),
        (
            "everything",
            Params {
                exposure: 0.4,
                temperature: 0.3,
                tint: -0.2,
                highlights: -0.5,
                shadows: 0.6,
                contrast: 0.3,
                whites: 0.2,
                blacks: -0.3,
                brightness: 0.3,
                saturation: 0.2,
                vibrance: 0.4,
                clarity: 0.5,
                brilliance: 0.4,
                sharpen: 0.4,
                purple: 0.8,
                green: 0.3,
                blue: 0.5,
                reach: 6.0,
                water: 0.3,
                auto: 0.7,
                mix: 0.7,
            },
            Some(&film),
            Some(&slog),
        ),
    ];
    for fmt in FORMATS {
        let src = texture(&g, w, h, fmt, Some(&px));
        let dst = texture(&g, w, h, fmt, None);
        let sv = stored(fmt, &px);
        let mut fx = Fx::new(&g.cx, w, h, true);
        let io = fx.io(&src, &dst);
        eprintln!(
            "{fmt:?} {w}x{h}, {} pyramid levels: max |GPU − CPU| in levels, share > 1 level",
            fx.n
        );
        let mut worst = 0f32;
        for (name, p, look, inp) in &cases {
            fx.set(p);
            fx.look(*look);
            fx.input(*inp);
            go(&g, &mut fx, &io, &[]);
            let gv = read(&g, &dst);
            let k = floats_of(&g, &fx.kb);
            let rv = reference(p, *inp, *look, Some(&k), (w, h, true), &sv);
            let (m, n, ulp) = compare(fmt, &gv, &rv);
            eprintln!("  {name:20} {m:.3} {n:.1e}  f16 steps {ulp:.1}");
            worst = worst.max(m);
            if fmt == FORMATS[0] && matches!(*name, "everything" | "clarity +1" | "defringe") {
                save(
                    &format!("cpu_{}.png", name.split(' ').next().unwrap()),
                    w,
                    h,
                    &gv,
                );
            }
        }
        assert!(worst <= 1.0, "{fmt:?}: max error {worst} levels");
    }
}

/// `.cube` application against ffmpeg's lut3d (interp=tetrahedral) and lut1d (interp=linear) on the same 8-bit
/// picture.
#[test]
fn luts_match_ffmpeg() {
    let g = gpu();
    let (w, px) = picture(1024);
    let h = px.len() as u32 / w;
    let dir = out();
    let inp = dir.join("lut_in.png");
    let b: Vec<u8> = px
        .iter()
        .flat_map(|p| [0, 1, 2].map(|k| (p[k] * 255.0).round() as u8))
        .collect();
    let mut e = png::Encoder::new(
        std::io::BufWriter::new(std::fs::File::create(&inp).unwrap()),
        w,
        h,
    );
    e.set_color(png::ColorType::Rgb);
    e.write_header().unwrap().write_image_data(&b).unwrap();
    let mut wild = Grade {
        sat: 0.5,
        contrast: 0.6,
        shadow: [0.2, 0.9, 1.0],
        light: [1.0, 0.5, 0.2],
        split: 0.5,
        ..Grade::default()
    }
    .cube("wild", 17);
    // ffmpeg's lut3d scales by 1 / (max − min) but ignores DOMAIN_MIN: a 0-based domain wider than 1
    wild.d3.as_mut().unwrap().max = [1.25; 3];
    let vlog = Cube {
        title: "vlog".into(),
        d1: Some(Log::VLOG.rec709(0.0)),
        d3: None,
    };
    let cases = [
        ("film33", Look::Film.cube()),
        ("teal65", Look::TealOrange.grade().cube("teal", 65)),
        ("mono33", Look::Mono.cube()),
        ("wild17", wild),
        ("vlog1d", vlog),
    ];
    let src = texture(&g, w, h, FORMATS[0], Some(&px));
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, false);
    let io = fx.io(&src, &dst);
    for (name, c) in cases {
        let f = dir.join(format!("{name}.cube"));
        std::fs::write(&f, c.to_string()).unwrap();
        let filt = if c.d3.is_some() {
            format!("lut3d=file={}:interp=tetrahedral", f.display())
        } else {
            format!("lut1d=file={}:interp=linear", f.display())
        };
        let o = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&inp)
            .args(["-vf", &filt, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
            .output()
            .expect("ffmpeg");
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        assert_eq!(o.stdout.len(), (3 * w * h) as usize);
        // the GPU gets the table as written to the file (6 decimals)
        let c = Cube::parse(&std::fs::read_to_string(&f).unwrap()).unwrap();
        fx.set(&Params::default());
        fx.look(Some(&c));
        go(&g, &mut fx, &io, &[]);
        let gv = read(&g, &dst);
        // GPU − ffmpeg: ≤ −2, −1, 0, +1, ≥ +2
        let mut hist = [0usize; 5];
        for (p, q) in gv.iter().zip(o.stdout.as_chunks::<3>().0) {
            for k in 0..3 {
                let d = (p[k] * 255.0).round() as i32 - q[k] as i32;
                hist[(d.clamp(-2, 2) + 2) as usize] += 1;
            }
        }
        let m = if hist[0] + hist[4] > 0 {
            2
        } else {
            (hist[1] + hist[3] > 0) as u8
        };
        // ffmpeg truncates to integers where we round: the CPU reference truncated should agree but for float ties
        let rv = reference(
            &Params::default(),
            None,
            Some(&c),
            None,
            (w, h, false),
            &stored(FORMATS[0], &px),
        );
        let ties = rv
            .iter()
            .zip(o.stdout.as_chunks::<3>().0)
            .flat_map(|(p, q)| (0..3).map(move |k| (p[k] * 255.0) as i32 != q[k] as i32))
            .filter(|&d| d)
            .count();
        eprintln!(
            "{name}: max |GPU − ffmpeg| {m} levels; GPU − ffmpeg ≤−2 / −1 / 0 / +1 / ≥+2: {hist:?}; \
             CPU reference truncated ≠ ffmpeg: {ties}"
        );
        assert!(m <= 1, "{name}: {m} levels");
    }
}

/// Synthetic purple, green and blue fringes along bright edges, a fringe twice as wide (reach 4 against 10), a neutral
/// edge, and a purple object without a bright edge.
#[test]
fn defringe() {
    let g = gpu();
    let (w, h) = (560u32, 160u32);
    let mut px = vec![[0.08, 0.08, 0.08, 1.0]; (w * h) as usize];
    let mut put = |x0: u32, y0: u32, x1: u32, y1: u32, c: [f32; 4]| {
        for y in y0..y1 {
            for x in x0..x1 {
                px[(y * w + x) as usize] = c;
            }
        }
    };
    // bright patches with a fringe ring (inner part strong, outer fainter), and one without
    let purple = [[0.42, 0.14, 0.55, 1.0], [0.22, 0.09, 0.30, 1.0]];
    let green = [[0.20, 0.45, 0.12, 1.0], [0.12, 0.25, 0.09, 1.0]];
    let blue = [[0.12, 0.30, 0.62, 1.0], [0.10, 0.18, 0.34, 1.0]];
    for (x0, c, r) in [
        (20, purple, 5),
        (120, green, 5),
        (220, blue, 5),
        (330, purple, 10),
    ] {
        put(x0 - r, 30 - r, x0 + 60 + r, 90 + r, c[1]);
        put(
            x0 - r / 2 - 1,
            30 - r / 2 - 1,
            x0 + 61 + r / 2,
            91 + r / 2,
            c[0],
        );
        put(x0, 30, x0 + 60, 90, [0.97, 0.97, 0.97, 1.0]);
    }
    put(450, 30, 510, 90, [0.97, 0.97, 0.97, 1.0]);
    // purple object on mid grey, away from bright edges
    put(0, 110, w, h, [0.35, 0.35, 0.35, 1.0]);
    put(40, 120, 120, 150, purple[0]);
    let src = texture(&g, w, h, FORMATS[0], Some(&px));
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, false);
    let io = fx.io(&src, &dst);
    let sv = stored(FORMATS[0], &px);
    let chroma = |p: &[f32; 4]| {
        p[..3].iter().fold(0f32, |a, &v| a.max(v)) - p[..3].iter().fold(1f32, |a, &v| a.min(v))
    };
    let luma = |p: &[f32; 4]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    // mean chroma before / after, max luma change and max channel change (levels) over a rect minus `skip`
    let region = |o: &[[f32; 4]], r: [u32; 4], skip: Option<[u32; 4]>| {
        let (mut a, mut b, mut n, mut dl, mut dm) = (0.0, 0.0, 0, 0f32, 0f32);
        for y in r[1]..r[3] {
            for x in r[0]..r[2] {
                if skip.is_some_and(|s| x >= s[0] && x < s[2] && y >= s[1] && y < s[3]) {
                    continue;
                }
                let i = (y * w + x) as usize;
                a += chroma(&sv[i]);
                b += chroma(&o[i]);
                n += 1;
                dl = dl.max((luma(&sv[i]) - luma(&o[i])).abs());
                dm = dm.max(
                    (0..3)
                        .map(|k| (sv[i][k] - o[i][k]).abs())
                        .fold(0.0, f32::max),
                );
            }
        }
        (a / n as f32, b / n as f32, dl * 255.0, dm * 255.0)
    };
    let ring = |x0: u32, r: u32| {
        (
            [x0 - r, 30 - r, x0 + 60 + r, 90 + r],
            Some([x0, 30, x0 + 60, 90]),
        )
    };
    let zoom = |v: &[[f32; 4]]| {
        (0..w * h * 4)
            .map(|i| v[((i / (2 * w) / 2) * w + i % (2 * w) / 2) as usize])
            .collect::<Vec<_>>()
    };
    save("defringe_before.png", 2 * w, 2 * h, &zoom(&sv));
    for (reach, b) in [(4.0, 0.0), (4.0, 1.0), (10.0, 1.0)] {
        fx.set(&Params {
            purple: 1.0,
            green: 1.0,
            blue: b,
            reach,
            ..Params::default()
        });
        go(&g, &mut fx, &io, &[]);
        let o = read(&g, &dst);
        let [p, gr, bl, wide] = [(20, 5), (120, 5), (220, 5), (330, 10)].map(|(x0, r)| {
            let (a, s) = ring(x0, r);
            region(&o, a, s)
        });
        let ne = region(&o, [440, 15, 520, 105], None);
        let ob = region(&o, [40, 120, 120, 150], None);
        eprintln!("reach {reach}, blue {b}:");
        for (name, r) in [
            ("purple", p),
            ("green", gr),
            ("blue", bl),
            ("wide purple", wide),
        ] {
            eprintln!(
                "  {name} fringe: mean chroma {:.3} → {:.3} ({:.0}% less), luma moved ≤ {:.2} levels",
                r.0,
                r.1,
                100.0 * (1.0 - r.1 / r.0),
                r.2
            );
        }
        eprintln!(
            "  neutral edge: max change {:.2} levels; purple object without a bright edge: max change {:.2} levels",
            ne.3, ob.3
        );
        assert!(p.1 < 0.25 * p.0 && gr.1 < 0.25 * gr.0);
        assert!(ne.3 == 0.0 && ob.3 == 0.0 && [p, gr, bl, wide].iter().all(|r| r.2 <= 1.0));
        if b == 0.0 {
            assert_eq!(bl.3, 0.0, "blue off");
        } else {
            assert!(bl.1 < 0.25 * bl.0);
        }
        assert!(if reach > 4.0 {
            wide.1 < 0.25 * wide.0
        } else {
            wide.1 > 0.4 * wide.0
        });
        save(
            &format!("defringe_after_{reach}_{b}.png"),
            2 * w,
            2 * h,
            &zoom(&o),
        );
    }
}

/// Overlay geometry: quadrant stickers on the sphere (equirect) and flat, checked against the documented mapping.
#[test]
fn overlay_geometry() {
    let g = gpu();
    let (w, h) = (1024u32, 512u32);
    let s = 64u32;
    let q = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]];
    let img = Image {
        w: s,
        h: s,
        px: (0..s * s)
            .flat_map(|i| {
                let c = q[((i / s >= s / 2) as usize) * 2 + (i % s >= s / 2) as usize];
                [c[0], c[1], c[2], 255]
            })
            .collect(),
    };
    let black = vec![[0.0, 0.0, 0.0, 1.0]; (w * h) as usize];
    let src = texture(&g, w, h, FORMATS[0], Some(&black));
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, true);
    let io = fx.io(&src, &dst);
    let places = [
        (
            "sphere",
            Place::Sphere {
                yaw: 30.0,
                pitch: 20.0,
                fov: 40.0,
                roll: 0.0,
            },
        ),
        (
            "nadir",
            Place::Sphere {
                yaw: -120.0,
                pitch: -90.0,
                fov: 60.0,
                roll: 15.0,
            },
        ),
        (
            "wrap",
            Place::Sphere {
                yaw: 170.0,
                pitch: 70.0,
                fov: 50.0,
                roll: -30.0,
            },
        ),
        (
            "flat",
            Place::Flat {
                at: [0.3, 0.6],
                w: 0.2,
                rot: 30.0,
            },
        ),
    ];
    for (name, pl) in places {
        let ov = Overlay::new(&g.cx, &img, pl, 1.0);
        go(&g, &mut fx, &io, &[&ov]);
        let o = read(&g, &dst);
        // the sticker coordinate at each pixel centre, from the documented definitions
        let uv = |x: u32, y: u32| -> Option<(f32, f32)> {
            match pl {
                Place::Flat { at, w: sw, rot } => {
                    let (sn, c) = rot.to_radians().sin_cos();
                    let (dx, dy) = (
                        x as f32 + 0.5 - at[0] * w as f32,
                        y as f32 + 0.5 - at[1] * h as f32,
                    );
                    let hw = sw * w as f32 / 2.0;
                    Some((
                        (dx * c + dy * sn) / (2.0 * hw) + 0.5,
                        (-dx * sn + dy * c) / (2.0 * hw) + 0.5,
                    ))
                }
                Place::Sphere {
                    yaw,
                    pitch,
                    fov,
                    roll,
                } => {
                    let (lon, lat) = (
                        (360.0 * (x as f32 / w as f32 - 0.5)).to_radians(),
                        (90.0 - 180.0 * y as f32 / (h - 1) as f32).to_radians(),
                    );
                    let d = Vec3::new(lat.cos() * lon.sin(), lat.sin(), lat.cos() * lon.cos());
                    let (yw, pt) = (yaw.to_radians(), pitch.to_radians());
                    let f = Vec3::new(pt.cos() * yw.sin(), pt.sin(), pt.cos() * yw.cos());
                    let r0 = Vec3::new(yw.cos(), 0.0, -yw.sin());
                    let u0 = f.cross(r0);
                    let (sn, c) = roll.to_radians().sin_cos();
                    let (r, u) = (r0 * c - u0 * sn, r0 * sn + u0 * c);
                    let z = d.dot(f);
                    let t = (fov.to_radians() / 2.0).tan();
                    (z > 0.01).then(|| {
                        (
                            d.dot(r) / z / (2.0 * t) + 0.5,
                            0.5 - d.dot(u) / z / (2.0 * t),
                        )
                    })
                }
            }
        };
        let (mut inside, mut bad, mut outside, mut stray) = (0, 0, 0, 0);
        for y in 0..h {
            for x in 0..w {
                let p = o[(y * w + x) as usize];
                let got = [0, 1, 2].map(|k| (p[k] * 255.0).round() as i32);
                let core = |v: f32| v > 0.03 && v < 0.97 && (v - 0.5).abs() > 0.03;
                let near = |v: f32| v > -0.03 && v < 1.03;
                match uv(x, y) {
                    Some((a, b)) if core(a) && core(b) => {
                        let c = q[((b > 0.5) as usize) * 2 + (a > 0.5) as usize];
                        inside += 1;
                        bad += (0..3).any(|k| (got[k] - c[k] as i32).abs() > 2) as usize;
                    }
                    Some((a, b)) if near(a) && near(b) => {}
                    _ => {
                        outside += 1;
                        stray += (got != [0, 0, 0]) as usize;
                    }
                }
            }
        }
        eprintln!("{name}: {inside} px inside, {bad} wrong; {outside} outside, {stray} touched");
        assert!(inside > 1000 && bad == 0 && stray == 0, "{name}");
        if name == "nadir" {
            // the nadir row sees the sticker's centre, where the quadrants meet: one colour along the row
            let r = &o[((h - 1) * w) as usize..];
            assert!(r.iter().all(|p| p == &r[0]), "nadir row not uniform");
        }
        save(&format!("geom_{name}.png"), w, h, &o);
    }
}

/// A sticker at the nadir and a look on a real stitched frame, a watermark and sticker on a flat view, defringe on the
/// dome; PNGs for viewing, plus ffmpeg v360 views looking down.
#[test]
fn real_frames() {
    let Some(eq) = equirect(5760) else { return };
    let g = gpu();
    let (w, h) = (eq.w, eq.h);
    // our sticker: a dark disc with a ring and text, through PNG encode / decode
    let s = 1024u32;
    let t = Image::text("x5desk", 150.0, [255, 255, 255, 255]);
    let mut px = vec![0u8; (4 * s * s) as usize];
    for y in 0..s {
        for x in 0..s {
            let r = (x as f32 + 0.5 - 512.0).hypot(y as f32 + 0.5 - 512.0) / 512.0;
            let a = (1.0 - r).clamp(0.0, 0.004) / 0.004;
            let c = 40.0 + 180.0 * ((r - 0.9).abs() < 0.03) as u8 as f32;
            let i = (4 * (y * s + x)) as usize;
            px[i..i + 4].copy_from_slice(&[
                c as u8,
                (c * 0.9) as u8,
                (c * 0.6) as u8,
                (255.0 * a) as u8,
            ]);
        }
    }
    let (ox, oy) = ((s - t.w) / 2, (s - t.h) / 2);
    for y in 0..t.h {
        for x in 0..t.w {
            let (i, j) = (
                (4 * ((oy + y) * s + ox + x)) as usize,
                (4 * (y * t.w + x)) as usize,
            );
            let a = t.px[j + 3] as f32 / 255.0;
            for k in 0..3 {
                px[i + k] = (px[i + k] as f32 * (1.0 - a) + t.px[j + k] as f32 * a).round() as u8;
            }
        }
    }
    let mut enc = vec![];
    {
        let mut e = png::Encoder::new(&mut enc, s, s);
        e.set_color(png::ColorType::Rgba);
        e.write_header().unwrap().write_image_data(&px).unwrap();
    }
    std::fs::write(out().join("sticker.png"), &enc).unwrap();
    let sticker = Image::png(&enc).unwrap();
    assert_eq!(sticker.px, px);
    let src = texture(&g, w, h, FORMATS[0], Some(&floats(&eq)));
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, true);
    let io = fx.io(&src, &dst);
    let nadir = Overlay::new(
        &g.cx,
        &sticker,
        Place::Sphere {
            yaw: 0.0,
            pitch: -90.0,
            fov: 70.0,
            roll: 0.0,
        },
        1.0,
    );
    let mark = Overlay::new(
        &g.cx,
        &Image::text("x5desk · 2026", 64.0, [255, 255, 255, 200]),
        Place::Sphere {
            yaw: 20.0,
            pitch: -20.0,
            fov: 25.0,
            roll: 0.0,
        },
        0.9,
    );
    fx.set(&Params {
        clarity: 0.4,
        vibrance: 0.3,
        mix: 0.8,
        ..Params::default()
    });
    fx.look(Some(&Look::TealOrange.cube()));
    go(&g, &mut fx, &io, &[&nadir, &mark]);
    let o = read(&g, &dst);
    save("nadir.png", w, h, &o);
    let f = 4;
    let (sw, sh) = (w / f, h / f);
    let small: Vec<[f32; 4]> = (0..sw * sh)
        .map(|i| {
            let (x, y) = (i % sw * f, i / sw * f);
            let mut a = [0.0; 4];
            for j in 0..f {
                for k in 0..f {
                    let p = o[((y + j) * w + x + k) as usize];
                    for c in 0..4 {
                        a[c] += p[c] / (f * f) as f32;
                    }
                }
            }
            a
        })
        .collect();
    save("nadir_small.png", sw, sh, &small);
    for (name, args) in [
        (
            "nadir_down.png",
            "v360=e:flat:pitch=-90:h_fov=110:v_fov=110:w=1024:h=1024",
        ),
        (
            "nadir_front.png",
            "v360=e:flat:pitch=-30:yaw=10:h_fov=100:v_fov=70:w=1280:h=800",
        ),
    ] {
        let st = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(out().join("nadir.png"))
            .args(["-vf", args])
            .arg(out().join(name))
            .status()
            .unwrap();
        assert!(st.success());
    }
    // flat: a perspective view (half float) with a watermark (bottom right) and a rotated sticker
    let fl = out().join("flat_in.png");
    let st = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(out().join("v635_5760.png"))
        .args([
            "-vf",
            "v360=e:flat:yaw=-60:pitch=-5:h_fov=100:v_fov=62:w=1920:h=1080",
        ])
        .arg(&fl)
        .status()
        .unwrap();
    assert!(st.success());
    let fi = Image::png(&std::fs::read(&fl).unwrap()).unwrap();
    let (fw, fh) = (fi.w, fi.h);
    let src = texture(&g, fw, fh, FORMATS[1], Some(&floats(&fi)));
    let dst = texture(&g, fw, fh, FORMATS[1], None);
    let mut fx = Fx::new(&g.cx, fw, fh, false);
    let io = fx.io(&src, &dst);
    let wm = Overlay::new(
        &g.cx,
        &Image::text("x5desk", 48.0, [255, 255, 255, 255]),
        Place::Flat {
            at: [0.9, 0.93],
            w: 0.1,
            rot: 0.0,
        },
        0.7,
    );
    let st = Overlay::new(
        &g.cx,
        &sticker,
        Place::Flat {
            at: [0.15, 0.25],
            w: 0.15,
            rot: 20.0,
        },
        1.0,
    );
    fx.set(&Params {
        contrast: 0.2,
        shadows: 0.4,
        highlights: -0.3,
        temperature: 0.2,
        ..Params::default()
    });
    fx.look(Some(&Look::Film.cube()));
    go(&g, &mut fx, &io, &[&wm, &st]);
    save("flat.png", fw, fh, &read(&g, &dst));
    // our looks on the frame, 1024 wide: original, then each look, 2 × 4 tiles at half size
    let (lw, lh) = (1024u32, 512u32);
    let small = floats(&equirect(lw).unwrap());
    let src = texture(&g, lw, lh, FORMATS[0], Some(&small));
    let dst = texture(&g, lw, lh, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, lw, lh, true);
    let io = fx.io(&src, &dst);
    let mut sheet = vec![[0.0; 4]; (2 * lw * lh) as usize];
    for (k, l) in [None].into_iter().chain(Look::ALL.map(Some)).enumerate() {
        fx.look(l.map(Look::cube).as_ref());
        go(&g, &mut fx, &io, &[]);
        let o = read(&g, &dst);
        let (tx, ty) = (k as u32 % 4 * lw / 2, k as u32 / 4 * lh / 2);
        for y in 0..lh / 2 {
            for x in 0..lw / 2 {
                let mut a = [0.0; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let p = o[((2 * y + dy) * lw + 2 * x + dx) as usize];
                    for c in 0..4 {
                        a[c] += p[c] / 4.0;
                    }
                }
                sheet[((ty + y) * 2 * lw + tx + x) as usize] = a;
            }
        }
    }
    save("looks.png", 2 * lw, lh, &sheet);
    // defringe on the dome against the sky (top left of the frame): before / after crops at 2×
    let mut fx = Fx::new(&g.cx, w, h, true);
    let src = texture(&g, w, h, FORMATS[0], Some(&floats(&eq)));
    let dst = texture(&g, w, h, FORMATS[0], None);
    let io = fx.io(&src, &dst);
    fx.set(&Params {
        purple: 1.0,
        ..Params::default()
    });
    go(&g, &mut fx, &io, &[]);
    let o = read(&g, &dst);
    let iv = floats(&eq);
    let (cx0, cy0, cw, ch) = (600u32, 200u32, 480u32, 300u32);
    for (name, v) in [("dome_before.png", &iv), ("dome_after.png", &o)] {
        let c: Vec<_> = (0..4 * cw * ch)
            .map(|i| {
                let (x, y) = (i % (2 * cw) / 2, i / (2 * cw) / 2);
                v[((cy0 + y) * w + cx0 + x) as usize]
            })
            .collect();
        save(name, 2 * cw, 2 * ch, &c);
    }
    let changed = iv
        .iter()
        .zip(&o)
        .filter(|(a, b)| (0..3).any(|k| (a[k] - b[k]).abs() > 1.5 / 255.0))
        .count();
    eprintln!(
        "defringe on the real frame: {:.3}% of pixels changed by > 1 level",
        100.0 * changed as f64 / iv.len() as f64
    );
}

/// GPU time per 5760 × 2880 frame (8 recordings in one submit, waited for, median of 5; the GPU is shared with other
/// work, so expect noise).
#[test]
fn timing() {
    let Some(eq) = equirect(5760) else { return };
    let g = gpu();
    let (w, h) = (eq.w, eq.h);
    let px = floats(&eq);
    let film = Look::Film.cube();
    let slog = Log::SLOG3.rec709(0.0);
    let ov = Overlay::new(
        &g.cx,
        &Image::text("x5desk", 300.0, [255, 255, 255, 255]),
        Place::Sphere {
            yaw: 0.0,
            pitch: -90.0,
            fov: 70.0,
            roll: 0.0,
        },
        1.0,
    );
    let colour = Params {
        exposure: 0.3,
        temperature: 0.2,
        tint: 0.1,
        highlights: -0.3,
        shadows: 0.3,
        contrast: 0.2,
        whites: 0.1,
        blacks: -0.1,
        saturation: 0.1,
        vibrance: 0.2,
        ..Params::default()
    };
    let all = Params {
        clarity: 0.5,
        brilliance: 0.3,
        brightness: 0.2,
        sharpen: 0.5,
        purple: 1.0,
        green: 0.5,
        blue: 0.5,
        auto: 0.7,
        ..colour
    };
    // name, parameters, look, input LUT, overlay, in place
    type Case<'a> = (
        &'a str,
        Params,
        Option<&'a Cube>,
        Option<&'a Lut>,
        bool,
        bool,
    );
    let cases: [Case; 12] = [
        ("neutral", Params::default(), None, None, false, false),
        ("colour adjustments", colour, None, None, false, false),
        (
            "+ clarity",
            Params {
                clarity: 0.5,
                ..colour
            },
            None,
            None,
            false,
            false,
        ),
        (
            "+ defringe + sharpen",
            Params {
                purple: 1.0,
                sharpen: 0.5,
                ..colour
            },
            None,
            None,
            false,
            false,
        ),
        (
            "+ defringe reach 12",
            Params {
                purple: 1.0,
                blue: 1.0,
                reach: 12.0,
                ..colour
            },
            None,
            None,
            false,
            false,
        ),
        (
            "+ auto",
            Params {
                auto: 1.0,
                ..colour
            },
            None,
            None,
            false,
            false,
        ),
        (
            "+ underwater + auto",
            Params {
                water: 1.0,
                auto: 1.0,
                ..colour
            },
            None,
            None,
            false,
            false,
        ),
        ("+ 3D look 33³", colour, Some(&film), None, false, false),
        (
            "everything + input LUT",
            all,
            Some(&film),
            Some(&slog),
            false,
            false,
        ),
        (
            "+ nadir overlay",
            all,
            Some(&film),
            Some(&slog),
            true,
            false,
        ),
        (
            "neutral in place",
            Params::default(),
            None,
            None,
            false,
            true,
        ),
        (
            "everything in place",
            all,
            Some(&film),
            Some(&slog),
            true,
            true,
        ),
    ];
    for fmt in FORMATS {
        let src = texture(&g, w, h, fmt, Some(&px));
        let dst = texture(&g, w, h, fmt, None);
        let mut fx = Fx::new(&g.cx, w, h, true);
        // in place: the frame is graded again each run, which doesn't matter for the time
        let ipt = texture(&g, w, h, fmt, Some(&px));
        let (two, one) = (fx.io(&src, &dst), fx.io(&ipt, &ipt));
        for (name, p, look, inp, o, ip) in cases {
            fx.set(&p);
            fx.look(look);
            fx.input(inp);
            let ovs: &[&Overlay] = if o { &[&ov] } else { &[] };
            let io = if ip { &one } else { &two };
            go(&g, &mut fx, io, ovs);
            let mut ms = vec![];
            for _ in 0..5 {
                let n = 8;
                let t = Instant::now();
                let mut enc = g.dev.create_command_encoder(&Default::default());
                for _ in 0..n {
                    fx.run(&mut enc, io, ovs);
                }
                g.q.submit([enc.finish()]);
                g.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                ms.push(t.elapsed().as_secs_f64() * 1e3 / n as f64);
            }
            ms.sort_by(f64::total_cmp);
            eprintln!(
                "{fmt:?} {w}x{h} {name:24} median {:.2} ms (min {:.2}, max {:.2})",
                ms[2], ms[0], ms[4]
            );
        }
    }
}

/// Frames of the three clips (indoor zoo v635, daylight v606, night v627) at `w` × `w/2`, or the synthetic picture.
fn clips(w: u32) -> Vec<(&'static str, Vec<[f32; 4]>)> {
    match samples() {
        Some(_) => ["v635", "v606", "v627"]
            .map(|c| (c, floats(&still(c, w).unwrap())))
            .into(),
        None => vec![("synthetic", synth(w, w / 2))],
    }
}

/// `w` × `h` frame halved per axis (2×2 box).
fn half(w: u32, h: u32, px: &[[f32; 4]]) -> Vec<[f32; 4]> {
    (0..w / 2 * (h / 2))
        .map(|i| {
            let (x, y) = (2 * (i % (w / 2)), 2 * (i / (w / 2)));
            let mut a = [0.0; 4];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = px[((y + dy) * w + x + dx) as usize];
                for c in 0..4 {
                    a[c] += p[c] / 4.0;
                }
            }
            a
        })
        .collect()
}

/// The statistics against the CPU's exact sums, the estimate and coefficients against the CPU mirror of `solve` from
/// the GPU's sums: three clips (or the synthetic picture), equirect and flat, each correction alone and both.
#[test]
fn analysis() {
    let g = gpu();
    let (w, h) = (1024u32, 512u32);
    let mut worst = [0f32; 3];
    for (clip, px) in clips(w) {
        let src = texture(&g, w, h, FORMATS[0], Some(&px));
        let dst = texture(&g, w, h, FORMATS[0], None);
        let sv = stored(FORMATS[0], &px);
        for pano in [true, false] {
            let z = bins(&sv, w, h, pano);
            let mut fx = Fx::new(&g.cx, w, h, pano);
            let io = fx.io(&src, &dst);
            for (sw, sa) in [(1.0, 0.0), (0.0, 1.0), (0.7, 0.4)] {
                fx.set(&Params {
                    water: sw,
                    auto: sa,
                    ..Params::default()
                });
                go(&g, &mut fx, &io, &[]);
                let k = floats_of(&g, &fx.kb);
                let zs = &k[32..];
                let dz = zs
                    .iter()
                    .zip(&z)
                    .map(|(&a, &b)| ((a as f64 - b as f64).abs() / (b as f64).max(1.0)) as f32)
                    .fold(0.0, f32::max);
                let fl = ((sw > 0.0) as u32) << 4 | ((sa > 0.0) as u32) << 5;
                let rel = |a: &[f32], b: &[f32]| {
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
                        .fold(0.0, f32::max)
                };
                let de = rel(&estimate(zs, fl), &k[..15]);
                let dc = rel(&coefficients(&k[..15], sw, sa), &k[16..26]);
                eprintln!(
                    "{clip} {} water {sw} auto {sa}: |GPU − CPU| relative: statistics {dz:.1e}, estimate {de:.1e}, \
                     coefficients {dc:.1e}; estimate {:.3?}",
                    if pano { "equirect" } else { "flat" },
                    &k[..15]
                );
                worst = [worst[0].max(dz), worst[1].max(de), worst[2].max(dc)];
            }
        }
    }
    assert!(
        worst[0] < 1e-4 && worst[1] < 1e-3 && worst[2] < 1e-4,
        "{worst:?}"
    );
}

/// Underwater tint in linear light: c·t + b·(1 − t) per channel (transmission t, backscatter b).
fn tint(px: &[[f32; 4]], t: Vec3, b: Vec3) -> Vec<[f32; 4]> {
    px.iter()
        .map(|p| {
            let c = enc(lin(Vec3::new(p[0], p[1], p[2])) * t + b * (1.0 - t));
            [c.x, c.y, c.z, p[3]]
        })
        .collect()
}

/// Temporal smoothing: each run moves the state by 1 − e^(−Δt / τ) toward its frame's estimate, a repeated frame
/// keeps it, a seek (back, or ahead by more than 1 s) and turning a correction on take the frame's own.
#[test]
fn temporal() {
    let g = gpu();
    let (w, h) = (512u32, 256u32);
    let a = clips(w).swap_remove(0).1;
    let b = tint(&a, Vec3::new(0.3, 0.7, 0.8), Vec3::new(0.02, 0.15, 0.2));
    let (ta, tb) = (
        texture(&g, w, h, FORMATS[0], Some(&a)),
        texture(&g, w, h, FORMATS[0], Some(&b)),
    );
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, true);
    let (ia, ib) = (fx.io(&ta, &dst), fx.io(&tb, &dst));
    let p = Params {
        water: 1.0,
        auto: 1.0,
        ..Params::default()
    };
    fx.set(&p);
    let st = |fx: &mut Fx, io: &Io| {
        go(&g, fx, io, &[]);
        floats_of(&g, &fx.kb)[..15].to_vec()
    };
    // without time calls every run takes its frame's own estimate
    let ea = st(&mut fx, &ia);
    let eb = st(&mut fx, &ib);
    let step = |s: &[f32], e: &[f32], dt: f64| -> Vec<f32> {
        let a = 1.0 - (-dt / 0.5).exp() as f32;
        s.iter().zip(e).map(|(s, e)| s + (e - s) * a).collect()
    };
    let dt = 1.0 / 30.0;
    let mut want = vec![];
    let mut got = vec![];
    fx.time(0.0);
    got.push(st(&mut fx, &ia));
    want.push(ea.clone());
    fx.time(dt);
    got.push(st(&mut fx, &ib));
    want.push(step(&ea, &eb, dt));
    fx.time(dt);
    got.push(st(&mut fx, &ib));
    want.push(want[1].clone());
    fx.time(2.0 * dt);
    got.push(st(&mut fx, &ib));
    want.push(step(&want[1], &eb, dt));
    fx.time(5.0);
    got.push(st(&mut fx, &ia));
    want.push(ea.clone());
    fx.time(4.9);
    got.push(st(&mut fx, &ib));
    want.push(eb.clone());
    fx.time(4.9 + dt);
    got.push(st(&mut fx, &ia));
    want.push(step(&eb, &ea, dt));
    // auto off: the water estimate starts over from this frame, the auto part is neutral
    fx.set(&Params { auto: 0.0, ..p });
    got.push(st(&mut fx, &ia));
    let mut e = ea.clone();
    e[7..].copy_from_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    want.push(e);
    // on again, set before time in the frame: still this frame's own
    fx.set(&p);
    fx.time(5.0);
    got.push(st(&mut fx, &ib));
    want.push(eb.clone());
    fx.time(5.0 + dt);
    got.push(st(&mut fx, &ia));
    want.push(step(&eb, &ea, dt));
    // a fresh Fx timed before its first run (state never written) starts from its frame too
    let mut f2 = Fx::new(&g.cx, w, h, true);
    f2.set(&p);
    f2.time(0.0);
    f2.time(dt);
    got.push(st(&mut f2, &ib));
    want.push(eb.clone());
    let mut worst = 0f32;
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        let d = g
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
            .fold(0.0, f32::max);
        eprintln!(
            "step {k}: |state − expected| {d:.1e}; gains {:.4?}",
            &g[4..10]
        );
        worst = worst.max(d);
    }
    eprintln!(
        "own estimates: clean {:.3?}, tinted {:.3?}",
        &ea[..7],
        &eb[..7]
    );
    assert!(worst < 1e-5, "{worst}");
}

/// Overlays blend in linear light: a uniform sticker (alpha 160) at 70 % over a gradient; inside pixels against
/// enc(lin(frame)·(1 − a) + lin(colour)·a) through the layer's 8-bit sRGB storage. Also 50 % white over black.
#[test]
fn blend() {
    let g = gpu();
    let (w, h) = (512u32, 256u32);
    let bg: Vec<[f32; 4]> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            [x, 0.5 * y, 1.0 - x, 1.0]
        })
        .collect();
    let col = [200u8, 60, 20, 160];
    let sticker = Image {
        w: 16,
        h: 16,
        px: col.repeat(256),
    };
    let ov = Overlay::new(
        &g.cx,
        &sticker,
        Place::Flat {
            at: [0.5, 0.5],
            w: 0.3,
            rot: 0.0,
        },
        0.7,
    );
    let white = Overlay::new(
        &g.cx,
        &Image {
            w: 4,
            h: 4,
            px: [255u8; 64].into(),
        },
        Place::Flat {
            at: [0.5, 0.5],
            w: 0.1,
            rot: 0.0,
        },
        0.5,
    );
    let q8 = |v: f32| (v * 255.0).round() / 255.0;
    // the sticker texel (premultiplied linear, sRGB-stored), then the layer after the 70 % draw
    let at = col[3] as f32 / 255.0;
    let tx = Vec3::from_array(
        [0, 1, 2].map(|c| cube::lin(q8(cube::enc(cube::lin(col[c] as f32 / 255.0) * at)))),
    );
    let lc = enc(tx * 0.7).to_array().map(q8);
    let (lc, la) = (lin(Vec3::from(lc)), q8(at * 0.7));
    let (x0, x1, y0, y1) = (182, 330, 54, 202);
    for fmt in FORMATS {
        let src = texture(&g, w, h, fmt, Some(&bg));
        let dst = texture(&g, w, h, fmt, None);
        let mut fx = Fx::new(&g.cx, w, h, false);
        let io = fx.io(&src, &dst);
        go(&g, &mut fx, &io, &[&ov]);
        let o = read(&g, &dst);
        let sv = stored(fmt, &bg);
        let (mut m, mut old) = (0f32, 0f32);
        for y in y0..y1 {
            for x in x0..x1 {
                let i = (y * w + x) as usize;
                let b = Vec3::new(sv[i][0], sv[i][1], sv[i][2]);
                let r = enc(lin(b) * (1.0 - la) + lc);
                let e = Vec3::from_array([0, 1, 2].map(|c| col[c] as f32 / 255.0));
                let d = b * (1.0 - at * 0.7) + e * at * 0.7;
                for c in 0..3 {
                    let s = if fmt == FORMATS[0] { 255.0 } else { 1023.0 };
                    let rr = if fmt == FORMATS[0] { q8(r[c]) } else { r[c] };
                    m = m.max((o[i][c] - rr).abs() * s);
                    old = old.max((o[i][c] - d[c]).abs() * 255.0);
                }
            }
        }
        go(&g, &mut fx, &io, &[&white]);
        let c = read(&g, &dst)[(h / 2 * w + w / 2) as usize];
        let bl = texture(
            &g,
            w,
            h,
            fmt,
            Some(&vec![[0.0, 0.0, 0.0, 1.0]; (w * h) as usize]),
        );
        let io = fx.io(&bl, &dst);
        go(&g, &mut fx, &io, &[&white]);
        let k = read(&g, &dst)[(h / 2 * w + w / 2) as usize];
        eprintln!(
            "{fmt:?}: max |GPU − linear blend| {m:.2} levels ({}-bit); the display-encoded blend differs by up to {old:.0} \
             (8-bit); 50 % white over black → {:.1} (8-bit levels; display-encoded blend: 127.5); over the gradient \
             centre {:.3?}",
            if fmt == FORMATS[0] { 8 } else { 10 },
            k[0] * 255.0,
            &c[..3]
        );
        // wgpu's Metal backend leaves MTLCompileOptions at fast math, whose pow is approximate: 1.0 8-bit and 3.8
        // 10-bit levels there (M4); the float readback can also land a hair above a whole level
        let tol = if cfg!(target_os = "macos") { 4.0 } else { 1.0 };
        assert!(m <= tol + 1e-4, "{fmt:?}: {m}");
        assert!((k[0] * 255.0 - 188.0).abs() <= 1.0);
    }
}

/// In place (source = destination) gives the same pixels as two textures, with the neighbourhood effects, the
/// automatic corrections and an overlay on.
#[test]
fn in_place() {
    let g = gpu();
    let (w, px) = picture(512);
    let h = px.len() as u32 / w;
    let p = Params {
        clarity: 0.5,
        sharpen: 0.5,
        purple: 1.0,
        blue: 0.5,
        reach: 7.0,
        water: 0.4,
        auto: 0.8,
        ..Params::default()
    };
    let ov = Overlay::new(
        &g.cx,
        &Image::text("x5desk", 40.0, [255, 200, 100, 220]),
        Place::Sphere {
            yaw: 10.0,
            pitch: -10.0,
            fov: 40.0,
            roll: 5.0,
        },
        0.8,
    );
    for fmt in FORMATS {
        let (a, b, c) = (
            texture(&g, w, h, fmt, Some(&px)),
            texture(&g, w, h, fmt, None),
            texture(&g, w, h, fmt, Some(&px)),
        );
        let mut fx = Fx::new(&g.cx, w, h, true);
        fx.set(&p);
        let (two, one) = (fx.io(&a, &b), fx.io(&c, &c));
        go(&g, &mut fx, &two, &[&ov]);
        go(&g, &mut fx, &one, &[&ov]);
        let (x, y) = (read(&g, &b), read(&g, &c));
        let n = x.iter().zip(&y).filter(|(p, q)| p != q).count();
        eprintln!("{fmt:?}: {n} pixels differ between in place and two textures");
        assert_eq!(n, 0);
    }
}

/// Mean |Δ| per channel (8-bit levels) and mean CIE76 ΔE (CIELAB, D65) of `x` against `o`, weighted by solid angle.
fn err(w: u32, h: u32, o: &[[f32; 4]], x: &[[f32; 4]]) -> (f32, f32) {
    let lab = |p: &[f32; 4]| {
        let l = lin(Vec3::new(p[0], p[1], p[2]).clamp(Vec3::ZERO, Vec3::ONE));
        let v = Vec3::new(
            0.4124564 * l.x + 0.3575761 * l.y + 0.1804375 * l.z,
            0.2126729 * l.x + 0.7151522 * l.y + 0.0721750 * l.z,
            0.0193339 * l.x + 0.119192 * l.y + 0.9503041 * l.z,
        ) / Vec3::new(0.95047, 1.0, 1.08883);
        let f = v.to_array().map(|t| {
            if t > 216.0 / 24389.0 {
                t.cbrt()
            } else {
                (24389.0 / 27.0 * t + 16.0) / 116.0
            }
        });
        Vec3::new(
            116.0 * f[1] - 16.0,
            500.0 * (f[0] - f[1]),
            200.0 * (f[1] - f[2]),
        )
    };
    let (mut a, mut e, mut n) = (0.0, 0.0, 0.0);
    for (i, (p, q)) in o.iter().zip(x).enumerate() {
        let k = (std::f32::consts::PI * ((i as u32 / w) as f32 + 0.5) / h as f32).sin();
        a += k
            * (0..3)
                .map(|c| (p[c].clamp(0.0, 1.0) - q[c].clamp(0.0, 1.0)).abs())
                .sum::<f32>()
            / 3.0;
        e += k * (lab(p) - lab(q)).length();
        n += k;
    }
    (255.0 * a / n, e / n)
}

/// `x` scaled in linear light by the least-squares fit to `o` (exposure isn't recoverable from one frame).
fn fit(o: &[[f32; 4]], x: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let (mut a, mut b) = (0.0, 0.0);
    for (p, q) in o.iter().zip(x) {
        let (p, q) = (
            lin(Vec3::new(p[0], p[1], p[2])),
            lin(Vec3::new(q[0], q[1], q[2])),
        );
        a += p.dot(q) as f64;
        b += q.dot(q) as f64;
    }
    let s = (a / b) as f32;
    x.iter()
        .map(|p| {
            let c = enc(lin(Vec3::new(p[0], p[1], p[2])) * s);
            [c.x, c.y, c.z, p[3]]
        })
        .collect()
}

/// Underwater correction on synthetic underwater versions of stitched v606 and v635 frames (2048 × 1024): known
/// per-channel transmission (and backscatter) in linear light, corrected at full strength; errors against the
/// original as is and after an exposure fit, next to plain gray world (linear) as a baseline. Strips: original,
/// tinted, corrected, corrected + auto 0.5.
#[test]
fn underwater() {
    if samples().is_none() {
        return;
    }
    let g = gpu();
    let (w, h) = (2048u32, 1024u32);
    // coastal water at about 3 m (red 0.6 / m, green 0.15, blue 0.1), with its backscatter, and shallow
    let cases = [
        ("attenuation", Vec3::new(0.165, 0.638, 0.741), Vec3::ZERO),
        (
            "attenuation + backscatter",
            Vec3::new(0.165, 0.638, 0.741),
            Vec3::new(0.02, 0.18, 0.22),
        ),
        (
            "shallow + backscatter",
            Vec3::new(0.45, 0.85, 0.9),
            Vec3::new(0.01, 0.08, 0.1),
        ),
    ];
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, true);
    for clip in ["v606", "v635"] {
        let o = floats(&still(clip, w).unwrap());
        let mut strip = vec![];
        for (k, (name, t, b)) in cases.iter().enumerate() {
            let x = stored(FORMATS[0], &tint(&o, *t, *b));
            let src = texture(&g, w, h, FORMATS[0], Some(&x));
            let io = fx.io(&src, &dst);
            fx.set(&Params {
                water: 1.0,
                ..Params::default()
            });
            go(&g, &mut fx, &io, &[]);
            let c = read(&g, &dst);
            let e = floats_of(&g, &fx.kb);
            fx.set(&Params {
                water: 1.0,
                auto: 0.5,
                ..Params::default()
            });
            go(&g, &mut fx, &io, &[]);
            let ca = read(&g, &dst);
            // gray world in linear light: gains mean(means) / mean per channel
            let mut m = Vec3::ZERO;
            for p in &x {
                m += lin(Vec3::new(p[0], p[1], p[2]));
            }
            let gw = Vec3::splat(m.element_sum() / 3.0) / m;
            let gwx: Vec<[f32; 4]> = x
                .iter()
                .map(|p| {
                    let c = enc(lin(Vec3::new(p[0], p[1], p[2])) * gw);
                    [c.x, c.y, c.z, 1.0]
                })
                .collect();
            let (ti, co, cf, gf) = (
                err(w, h, &o, &x),
                err(w, h, &o, &c),
                err(w, h, &o, &fit(&o, &c)),
                err(w, h, &o, &fit(&o, &gwx)),
            );
            eprintln!(
                "{clip} {name}: k {:.3}, veil {:.4?}, gains {:.3?}\n  mean |Δ| levels / ΔE76: tinted {:.1} / {:.1}, \
                 corrected {:.1} / {:.1}, corrected exposure-fit {:.1} / {:.1}; gray world exposure-fit {:.1} / {:.1}",
                e[0],
                &e[1..4],
                &e[4..7],
                ti.0,
                ti.1,
                co.0,
                co.1,
                cf.0,
                cf.1,
                gf.0,
                gf.1
            );
            if k == 1 {
                for v in [&o, &x, &c, &ca] {
                    strip.extend(half(w, h, v));
                }
            }
        }
        save(&format!("water_{clip}.png"), w / 2, 2 * h, &strip);
    }
}

/// Auto colour on stitched frames of the three clips (2048 × 1024): the estimate, before / after side by side.
#[test]
fn auto_colour() {
    if samples().is_none() {
        return;
    }
    let g = gpu();
    let (w, h) = (2048u32, 1024u32);
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, true);
    fx.set(&Params {
        auto: 1.0,
        ..Params::default()
    });
    let mut sheet = vec![[0.0; 4]; (w * h * 3 / 2) as usize];
    for (r, (clip, o)) in clips(w).into_iter().enumerate() {
        let src = texture(&g, w, h, FORMATS[0], Some(&o));
        let io = fx.io(&src, &dst);
        go(&g, &mut fx, &io, &[]);
        let a = read(&g, &dst);
        let e = floats_of(&g, &fx.kb);
        let (n, s): (f32, f32) = (0..64).fold((0.0, 0.0), |(n, s), b| {
            (n + e[32 + 6 * b], s + e[32 + 6 * b + 5] / 262143.0)
        });
        eprintln!(
            "{clip}: white balance {:.3?}, black point {:.4}, exposure {:+.2} EV, highlights {:+.2}, contrast +{:.2}, \
             vibrance +{:.2} (mean saturation {:.3})",
            &e[7..10],
            e[10],
            e[11],
            e[14],
            e[12],
            e[13],
            s / n
        );
        for (c, v) in [o, a].iter().enumerate() {
            let s = half(w, h, v);
            for y in 0..h / 2 {
                for x in 0..w / 2 {
                    sheet[((r as u32 * h / 2 + y) * w + c as u32 * w / 2 + x) as usize] =
                        s[(y * w / 2 + x) as usize];
                }
            }
        }
    }
    save("auto.png", w, 3 * h / 2, &sheet);
}

/// Defringe on real 5760 × 2880 frames: purple + blue at reach 4 and 8 on v606 (blue sky against the sunlit gate) and
/// v635 (dome against the sky); share of pixels changed, 2× crops before / after.
#[test]
fn defringe_frames() {
    if samples().is_none() {
        return;
    }
    let g = gpu();
    let (w, h) = (5760u32, 2880u32);
    let dst = texture(&g, w, h, FORMATS[0], None);
    let mut fx = Fx::new(&g.cx, w, h, true);
    for (clip, crop) in [
        ("v606", [4980u32, 260u32, 560u32, 360u32]),
        ("v635", [600, 200, 560, 360]),
    ] {
        let o = floats(&still(clip, w).unwrap());
        let src = texture(&g, w, h, FORMATS[0], Some(&o));
        let io = fx.io(&src, &dst);
        let cut = |v: &[[f32; 4]]| -> Vec<[f32; 4]> {
            (0..4 * crop[2] * crop[3])
                .map(|i| {
                    let (x, y) = (i % (2 * crop[2]) / 2, i / (2 * crop[2]) / 2);
                    v[((crop[1] + y) * w + crop[0] + x) as usize]
                })
                .collect()
        };
        save(
            &format!("fringe_{clip}_before.png"),
            2 * crop[2],
            2 * crop[3],
            &cut(&o),
        );
        for (reach, b) in [(4.0, 0.0), (4.0, 1.0), (8.0, 1.0)] {
            fx.set(&Params {
                purple: 1.0,
                blue: b,
                reach,
                ..Params::default()
            });
            go(&g, &mut fx, &io, &[]);
            let r = read(&g, &dst);
            // pixels changed; of those blue-ish (HSV hue 190…250°), the sky's colours
            let (mut n, mut nb) = (0, 0);
            for (p, q) in o.iter().zip(&r) {
                if (0..3).any(|k| (p[k] - q[k]).abs() > 1.5 / 255.0) {
                    n += 1;
                    let (mx, mn) = (p[0].max(p[1]).max(p[2]), p[0].min(p[1]).min(p[2]));
                    nb += (mx == p[2] && mx > mn && {
                        let hu = 60.0 * ((p[0] - p[1]) / (mx - mn) + 4.0);
                        (190.0..250.0).contains(&hu)
                    }) as usize;
                }
            }
            eprintln!(
                "{clip} purple 1, blue {b}, reach {reach}: {:.3}% of pixels changed by > 1 level, {:.3}% blue-ish",
                100.0 * n as f64 / o.len() as f64,
                100.0 * nb as f64 / o.len() as f64
            );
            save(
                &format!("fringe_{clip}_{reach}_{b}.png"),
                2 * crop[2],
                2 * crop[3],
                &cut(&r),
            );
        }
    }
}
