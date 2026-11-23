//! The effects on synthetic frames (exact checks) and on the X5's footage (`X5_SAMPLES`; skipped when unset). Pictures
//! and videos go to `X5_SHOT_OUT` (default `target/effects/out` next to Cargo's test tmp directory, on disk); ffmpeg
//! serves as a test tool only (frames out of the exported videos). GPU timings are noisy: other work shares the GPU.
//! Run with `--release`.

use effects::{Ctx, Delay, FORMATS, Light, Map, Median, Stack, Trails};
use glam::{DQuat, DVec3, Quat, Vec3};
use stab::vis;
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, MutexGuard, OnceLock, atomic::AtomicBool},
    time::Instant,
};
use video::Gpu;

/// One device per test binary; the lock runs the GPU tests one at a time.
fn gpu() -> (MutexGuard<'static, ()>, Arc<Gpu>, stitch::Ctx, Ctx) {
    static G: OnceLock<(Mutex<()>, Arc<Gpu>, stitch::Ctx, Ctx)> = OnceLock::new();
    let (m, g, sx, cx) = G.get_or_init(|| {
        let g = Arc::new(Gpu::new().expect("GPU"));
        let sx = stitch::Ctx::new(&g.device, &g.queue);
        let cx = Ctx::new(&g.device, &g.queue);
        (Mutex::new(()), g, sx, cx)
    });
    (
        m.lock().unwrap_or_else(|e| e.into_inner()),
        g.clone(),
        sx.clone(),
        cx.clone(),
    )
}

fn sample(name: &str) -> Option<PathBuf> {
    let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        eprintln!("X5_SAMPLES unset: skipped");
    }
    d.map(|d| d.join(name))
}

fn out(name: &str) -> PathBuf {
    let d = std::env::var_os("X5_SHOT_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_TARGET_TMPDIR")).join("../out"));
    std::fs::create_dir_all(&d).unwrap();
    // an export never replaces a file: an earlier run's go
    let p = d.join(name);
    let _ = std::fs::remove_file(&p);
    p
}

/// A frame texture: sampled, storage, copied both ways.
fn frame(cx: &Ctx, w: u32, h: u32, f: wgpu::TextureFormat) -> wgpu::Texture {
    use wgpu::TextureUsages as U;
    cx.dev.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: f,
        usage: U::TEXTURE_BINDING | U::STORAGE_BINDING | U::COPY_SRC | U::COPY_DST,
        view_formats: &[],
    })
}

fn put(cx: &Ctx, t: &wgpu::Texture, px: &[u8]) {
    cx.q.write_texture(
        t.as_image_copy(),
        px,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * t.width()),
            rows_per_image: None,
        },
        t.size(),
    );
}

fn read(cx: &Ctx, t: &wgpu::Texture) -> Vec<u8> {
    stitch::read_texture(&cx.dev, &cx.q, t)
}

fn submit(cx: &Ctx, f: impl FnOnce(&mut wgpu::CommandEncoder)) {
    let mut e = cx.dev.create_command_encoder(&Default::default());
    f(&mut e);
    cx.q.submit([e.finish()]);
}

fn wait(cx: &Ctx) {
    cx.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

/// Wall time per submit of `f`, ms, over `n` submits after one warm-up (GPU shared: noisy).
fn time(cx: &Ctx, n: usize, mut f: impl FnMut(&mut wgpu::CommandEncoder)) -> f64 {
    submit(cx, &mut f);
    wait(cx);
    let t = Instant::now();
    for _ in 0..n {
        submit(cx, &mut f);
    }
    wait(cx);
    t.elapsed().as_secs_f64() * 1e3 / n as f64
}

/// RGBA8 -> JPEG (quality 90), scaled down by `s` (box filter) first.
fn jpg(px: &[u8], w: u32, h: u32, s: u32, name: &str) {
    let (ow, oh) = (w / s, h / s);
    let mut o = vec![0u8; (4 * ow * oh) as usize];
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..4 {
                let mut a = 0u32;
                for j in 0..s {
                    for i in 0..s {
                        a += px[(4 * ((y * s + j) * w + x * s + i) + c) as usize] as u32;
                    }
                }
                o[(4 * (y * ow + x) + c) as usize] = ((a + s * s / 2) / (s * s)) as u8;
            }
        }
    }
    std::fs::write(
        out(name),
        export::jpeg::encode(&o, ow, oh, 90, None).unwrap(),
    )
    .unwrap();
}

/// The `w` × `h` crop at (x, y) of an RGBA8 picture.
fn crop(px: &[u8], w: u32, [x, y, cw, ch]: [u32; 4]) -> Vec<u8> {
    (y..y + ch)
        .flat_map(|r| &px[(4 * (r * w + x)) as usize..(4 * (r * w + x + cw)) as usize])
        .copied()
        .collect()
}

/// Deterministic numbers in 0..1.
fn rng(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*s >> 11) as f64 / (1u64 << 53) as f64
}

/// Fraction of pixels where some channel differs by more than `t` (0..255).
fn differ(a: &[u8], b: &[u8], t: u8) -> f64 {
    let n = a
        .chunks(4)
        .zip(b.chunks(4))
        .filter(|(p, q)| (0..3).any(|c| p[c].abs_diff(q[c]) > t))
        .count();
    n as f64 / (a.len() / 4) as f64
}

/// The median against a CPU median (byte-exact on 8-bit frames, odd and even counts, ring wrap-around), and Empty
/// Scene's ground truth: a textured background with discs moving across it plus noise; the plate against the
/// background where the discs passed.
#[test]
fn median_synthetic() {
    let (_l, _g, _sx, cx) = gpu();
    let (w, h) = (512u32, 256u32);
    let mut s = 7u64;
    let bg: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let v = |f: f64, p: f64| (128.0 + 90.0 * (x * f / 37.0 + y * p / 23.0).sin()) as u8;
            [v(1.0, 0.3), v(0.7, 1.1), v(0.2, 0.8), 255]
        })
        .collect();
    // 9 frames: 2 discs each crossing the picture, noise ±6
    let n = 9;
    let disc = |k: usize, x: u32, y: u32| {
        let c = [
            (60.0 + 50.0 * k as f64, 128.0),
            (450.0 - 45.0 * k as f64, 90.0 + 12.0 * k as f64),
        ];
        c.iter()
            .any(|&(cx, cy)| (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2) < 30.0f64.powi(2))
    };
    let frames: Vec<Vec<u8>> = (0..n)
        .map(|k| {
            (0..w * h)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    let mut p = [0u8; 4];
                    for c in 0..3 {
                        let v = if disc(k, x, y) {
                            [250.0, 30.0, 200.0][c]
                        } else {
                            bg[(4 * i) as usize + c] as f64
                        };
                        p[c] = (v + 12.0 * (rng(&mut s) - 0.5)).round().clamp(0.0, 255.0) as u8;
                    }
                    p[3] = 255;
                    p
                })
                .collect()
        })
        .collect();
    let f = frame(&cx, w, h, FORMATS[0]);
    let o = frame(&cx, w, h, FORMATS[0]);
    // ring of 6: after 9 pushes it holds frames 3..9 (its last 6); after 5 an odd count
    let mut m = Median::new(&cx, w, h, 6, FORMATS[0]);
    for (k, px) in frames.iter().enumerate() {
        put(&cx, &f, px);
        submit(&cx, |e| m.push(e, &f));
        if k == 4 || k == 8 {
            submit(&cx, |e| m.run(e, &o));
            let got = read(&cx, &o);
            let held = &frames[(k + 1).saturating_sub(6)..=k];
            let mut bad = 0;
            for i in 0..(w * h) as usize {
                for c in 0..3 {
                    let mut v: Vec<u8> = held.iter().map(|x| x[4 * i + c]).collect();
                    v.sort();
                    bad += (got[4 * i + c] != v[(v.len() - 1) / 2]) as usize;
                }
            }
            eprintln!(
                "median of {} frames: {bad} values differ from the CPU median",
                held.len()
            );
            assert_eq!(bad, 0);
        }
    }
    // all 9 frames: each pixel is under a disc in at most 2 of them
    let mut m = Median::new(&cx, w, h, 9, FORMATS[0]);
    for px in &frames {
        put(&cx, &f, px);
        submit(&cx, |e| m.push(e, &f));
    }
    submit(&cx, |e| m.run(e, &o));
    let got = read(&cx, &o);
    let swept: Vec<usize> = (0..(w * h) as usize)
        .filter(|&i| (0..n).any(|k| disc(k, i as u32 % w, i as u32 / w)))
        .collect();
    let err = |p: &[u8]| {
        swept
            .iter()
            .map(|&i| {
                (0..3)
                    .map(|c| p[4 * i + c].abs_diff(bg[4 * i + c]) as f64)
                    .sum::<f64>()
                    / 3.0
            })
            .sum::<f64>()
            / swept.len() as f64
    };
    let (e1, em) = (err(&frames[4]), err(&got));
    let on: Vec<usize> = swept
        .iter()
        .copied()
        .filter(|&i| disc(4, i as u32 % w, i as u32 / w))
        .collect();
    let e4 = on
        .iter()
        .map(|&i| {
            (0..3)
                .map(|c| frames[4][4 * i + c].abs_diff(bg[4 * i + c]) as f64)
                .sum::<f64>()
                / 3.0
        })
        .sum::<f64>()
        / on.len() as f64;
    eprintln!(
        "{} swept pixels ({:.1} %): mean |frame 4 − background| {e1:.2} ({e4:.1} under its discs), |plate − background| {em:.2} (noise alone ±6 uniform: 3 / √(n) ≈ 1.2)",
        swept.len(),
        100.0 * swept.len() as f64 / (w * h) as f64
    );
    assert!(em < 2.0 && e4 > 50.0);
    jpg(&frames[4], w, h, 1, "median_synthetic_frame4.jpg");
    jpg(&got, w, h, 1, "median_synthetic_plate.jpg");
}

/// Empty Scene on v635 (6 s, held on a railing: the gyro says it turned 5° about the vertical) and on v627 frames
/// 165..600 (14.5 s at night, the camera on a stick held still, people walking past): 21 frames over each stitched at
/// 5760 × 2880 into the median, with the camera's own orientation and with the gyro's (direction lock: every frame
/// into the same world directions). Moving people measured as the share of each frame that differs
/// from the plate; the residual as the share where plates of the even and of the odd frames disagree; the median's
/// GPU time.
#[test]
fn median_footage() {
    for (clip, k0, k1) in [("v635", 0, 175), ("v627", 165, 600)] {
        let Some(p) = sample(&format!("{clip}.insv")) else {
            return;
        };
        median_clip(&p, clip, k0, k1);
    }
    // the running clean plate through an export job: v635 at 3840, every 6th frame into a ring of 15 (3 s)
    let (_l, g, sx, cx) = gpu();
    let p = sample("v635.insv").unwrap();
    let m = Median::new(&cx, 3840, 1920, 15, FORMATS[0]);
    export(
        &g,
        &sx,
        &p,
        [0.0, f64::INFINITY],
        3840,
        Some((stab::Lock::Direction, 0.0)),
        m.hook(6),
        "v635_plate",
        &[0.5, 3.0, 5.5],
        2,
    );
}

fn median_clip(p: &Path, clip: &str, k0: usize, k1: usize) {
    let (_l, g, sx, cx) = gpu();
    let c = effects::Clip::open(p).unwrap();
    let all: Vec<usize> = (k0..k1).collect();
    let turn = c.turn(&all).unwrap();
    eprintln!("{clip} frames {k0}..{k1}: the camera turned {turn:.3}° at most (gyro)");
    let (w, h, n) = (5760u32, 2880u32, 21usize);
    let ks: Vec<usize> = (0..n).map(|i| k0 + i * (all.len() - 1) / (n - 1)).collect();
    let tau = 25;
    let lock = c.poses(stab::Lock::Direction, 0.0).unwrap();
    let t = Instant::now();
    let reg = c
        .register(
            &g,
            &sx,
            &ks,
            n / 2,
            [15f64.to_radians(), 115f64.to_radians()],
            |k| c.pose(k, Some(&lock)),
        )
        .unwrap();
    let fits: Vec<String> = reg
        .iter()
        .map(|(_, f)| {
            f.as_ref().map_or("-".into(), |f| {
                format!(
                    "{:.2}°/{:.3}",
                    stab::log(f.q).length().to_degrees(),
                    f.rms.to_degrees()
                )
            })
        })
        .collect();
    eprintln!(
        "{clip}: registration in {:.2} s, rotation left after the gyro / rms of the fit per frame: {}",
        t.elapsed().as_secs_f64(),
        fits.join(" ")
    );
    let mut res = vec![];
    for (name, pv) in [("camera", 0), ("gyro", 1), ("registered", 2)] {
        let pose = |k: usize| match pv {
            0 => c.pose(k, None),
            1 => c.pose(k, Some(&lock)),
            _ => reg[ks.iter().position(|&x| x == k).unwrap()].0,
        };
        let mut m = [21u32, 11, 10].map(|k| Median::new(&cx, w, h, k, FORMATS[0]));
        let keep: Vec<wgpu::Texture> = (0..3).map(|_| frame(&cx, w, h, FORMATS[0])).collect();
        let t = Instant::now();
        c.frames(&g, &sx, w, false, &ks, pose, |f, k| {
            let i = ks.iter().position(|&x| x == k).unwrap();
            submit(&cx, |e| {
                m[0].push(e, f);
                m[1 + i % 2].push(e, f);
                if let Some(j) = [0, n / 2, n - 1].iter().position(|&x| x == i) {
                    e.copy_texture_to_texture(f.as_image_copy(), keep[j].as_image_copy(), f.size());
                }
            });
            Ok(())
        })
        .unwrap();
        wait(&cx);
        eprintln!(
            "{clip} {name}: decode and stitch of frames {}..{} ({n} kept): {:.2} s",
            ks[0],
            ks[n - 1],
            t.elapsed().as_secs_f64()
        );
        let o: Vec<wgpu::Texture> = (0..3).map(|_| frame(&cx, w, h, FORMATS[0])).collect();
        for i in 0..3 {
            submit(&cx, |e| m[i].run(e, &o[i]));
        }
        let ms = time(&cx, 5, |e| m[0].run(e, &o[0]));
        eprintln!("{clip} {name}: median of 21 frames at {w}x{h}: {ms:.1} ms (GPU shared, noisy)");
        let plate = read(&cx, &o[0]);
        let (pe, po) = (read(&cx, &o[1]), read(&cx, &o[2]));
        let fr: Vec<Vec<u8>> = keep.iter().map(|t| read(&cx, t)).collect();
        let moving: Vec<f64> = fr.iter().map(|f| differ(f, &plate, tau)).collect();
        let r = differ(&pe, &po, tau);
        let ff = differ(&fr[0], &fr[2], tau);
        eprintln!(
            "{clip} {name}: share of pixels off by > {tau}/255: frames vs plate {:.3} / {:.3} / {:.3} %, first vs last frame {:.3} %, even-frame vs odd-frame plate {:.3} %",
            100.0 * moving[0],
            100.0 * moving[1],
            100.0 * moving[2],
            100.0 * ff,
            100.0 * r
        );
        res.push((r, ff));
        jpg(&plate, w, h, 4, &format!("{clip}_{name}_plate.jpg"));
        jpg(&fr[0], w, h, 4, &format!("{clip}_{name}_frame0.jpg"));
        if pv < 2 {
            continue;
        }
        std::fs::write(
            out(&format!("{clip}_plate_5760.jpg")),
            export::jpeg::encode(&plate, w, h, 92, Some(&export::jpeg::gpano(w, h))).unwrap(),
        )
        .unwrap();
        // where the first frame differs most from the plate: a 1024 × 512 crop of both
        let (bw, bh) = (1024u32, 512u32);
        let mut best = (0usize, [0u32; 2]);
        for by in (0..h - bh).step_by(128) {
            for bx in (0..w - bw).step_by(128) {
                let mut c = 0;
                for y in (by..by + bh).step_by(4) {
                    for x in (bx..bx + bw).step_by(4) {
                        let i = (4 * (y * w + x)) as usize;
                        c += (0..3).any(|k| fr[0][i + k].abs_diff(plate[i + k]) > tau) as usize;
                    }
                }
                if c > best.0 {
                    best = (c, [bx, by]);
                }
            }
        }
        let r = [best.1[0], best.1[1], bw, bh];
        eprintln!("most change at {r:?}");
        jpg(
            &crop(&fr[0], w, r),
            bw,
            bh,
            1,
            &format!("{clip}_crop_frame0.jpg"),
        );
        jpg(
            &crop(&fr[1], w, r),
            bw,
            bh,
            1,
            &format!("{clip}_crop_mid.jpg"),
        );
        jpg(
            &crop(&fr[2], w, r),
            bw,
            bh,
            1,
            &format!("{clip}_crop_last.jpg"),
        );
        jpg(
            &crop(&plate, w, r),
            bw,
            bh,
            1,
            &format!("{clip}_crop_plate.jpg"),
        );
    }
    assert!(res[2].0 < res[1].0 && res[2].0 < res[2].1 / 4.0);
}

fn srgb(l: f64) -> f64 {
    if l <= 0.0031308 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// An export job over `t` of a clip, 360 at `w`, H.265 8-bit at 40 Mbit/s, the effect hook `fx`; frames at `at`
/// (output s) extracted by ffmpeg, scaled by `s`.
#[allow(clippy::too_many_arguments)]
fn export(
    g: &Gpu,
    sx: &stitch::Ctx,
    src: &Path,
    t: [f64; 2],
    w: u32,
    stab: Option<(stab::Lock, f64)>,
    fx: export::Hook,
    name: &str,
    at: &[f64],
    s: u32,
) {
    let path = out(&format!("{name}.mp4"));
    let job = export::Job {
        cut: export::Cut::Segs(vec![export::Seg {
            files: vec![src.to_path_buf()],
            t,
        }]),
        path: path.clone(),
        out: export::Out::Pano(w),
        enc: export::Enc::Video {
            codec: video::Codec::H265,
            deep: false,
            rate: video::Rate::Cbr(40_000_000),
            gop: 30,
        },
        raft: false,
        colour: Some(0.1),
        stab,
        fx: Some(fx),
        flat: false,
        pre: None,
    };
    let t0 = Instant::now();
    let p = export::run(g, sx, &mut None, job, &AtomicBool::new(false), &mut |_| {}).unwrap();
    eprintln!(
        "{name}.mp4: {} frames in {:.1} s, steady {:.1} fps",
        p.done,
        t0.elapsed().as_secs_f64(),
        p.fps
    );
    for &x in at {
        let o = out(&format!("{name}_{x:.1}s.jpg"));
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-ss", &format!("{x}"), "-i"])
            .arg(&path)
            .args([
                "-frames:v",
                "1",
                "-vf",
                &format!("scale=iw/{s}:-2"),
                "-q:v",
                "3",
            ])
            .arg(&o)
            .status()
            .unwrap();
        assert!(ok.success());
    }
}

/// Trails against their formula on a dot moving over black (decay, threshold, the window, add mode; 8- and 16-bit
/// frames), and the pass's GPU time at 5760 × 2880.
#[test]
fn trails_synthetic() {
    let (_l, _g, _sx, cx) = gpu();
    let (w, h) = (256u32, 64u32);
    for (fmt, bpp) in [(FORMATS[0], 4usize), (FORMATS[1], 8)] {
        for add in [false, true] {
            let p = Light {
                thr: 0.5,
                soft: 0.1,
                decay: 0.9,
                gain: 1.0,
                add,
            };
            let mut tr = Trails::new(&cx, w, h, fmt, p);
            let f = frame(&cx, w, h, fmt);
            let n = 20;
            for k in 0..n {
                // the dot at x = 10 + 8k, row 32, value 1 (dim grey 0.25 everywhere else)
                let mut px = vec![0u8; bpp * (w * h) as usize];
                for i in 0..(w * h) as usize {
                    let v = if i == (32 * w + 10 + 8 * k) as usize {
                        1.0
                    } else {
                        0.25
                    };
                    for c in 0..4 {
                        let v = if c == 3 { 1.0 } else { v };
                        if bpp == 4 {
                            px[4 * i + c] = (v * 255.0) as u8;
                        } else {
                            px[8 * i + 2 * c..8 * i + 2 * c + 2]
                                .copy_from_slice(&f16(v).to_le_bytes());
                        }
                    }
                }
                cx.q.write_texture(
                    f.as_image_copy(),
                    &px,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bpp as u32 * w),
                        rows_per_image: None,
                    },
                    f.size(),
                );
                // frames 15.. add no light
                submit(&cx, |e| tr.push(e, &f, k < 15));
            }
            let o = read(&cx, &f);
            let at = |x: u32| -> f64 {
                let i = (32 * w + x) as usize;
                if bpp == 4 {
                    o[4 * i] as f64 / 255.0
                } else {
                    h16(u16::from_le_bytes([o[8 * i], o[8 * i + 1]]))
                }
            };
            let lin = |v: f64| {
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            let mut worst = 0f64;
            for k in 0..n {
                // the frame's own light there (the last frame's dot, else grey), the trail from frame k (when on)
                // decayed over the frames after it
                let l = if k == n - 1 { 1.0 } else { lin(0.25) };
                let a = if k < 15 {
                    0.9f64.powi((n - 1 - k) as i32)
                } else {
                    0.0
                };
                let want =
                    srgb(if add { l + a } else { l.max(a) }).min(if bpp == 4 { 1.0 } else { 9.0 });
                worst = worst.max((at(10 + 8 * k) - want).abs());
            }
            // the background: grey 0.25 stays (below the threshold, nothing accumulates)
            let bg = (at(5) - 0.25).abs();
            eprintln!(
                "{fmt:?} add {add}: largest error along the trail {worst:.4}, background off by {bg:.4}"
            );
            assert!(worst < 0.006 && bg < 0.003);
        }
    }
    let (w, h) = (5760u32, 2880u32);
    let mut tr = Trails::new(&cx, w, h, FORMATS[0], Light::default());
    let f = frame(&cx, w, h, FORMATS[0]);
    eprintln!(
        "trails at {w}x{h}: {:.2} ms per frame (GPU shared, noisy)",
        time(&cx, 20, |e| tr.push(e, &f, true))
    );
}

fn f16(v: f64) -> u16 {
    let b = (v as f32).to_bits();
    let (s, e, m) = (
        (b >> 16) & 0x8000,
        ((b >> 23) & 0xff) as i32 - 127 + 15,
        b & 0x7fffff,
    );
    if v == 0.0 {
        0
    } else {
        (s | ((e as u32) << 10) | (m >> 13)) as u16
    }
}

fn h16(v: u16) -> f64 {
    let (e, m) = ((v >> 10 & 31) as i32, (v & 1023) as f64);
    if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

/// Light trails on v627 frames 165..465 (10 s at night, the camera held still, people with phones and lamps): an export
/// job (360 at 3840, the gyro's direction lock) with the trails as its effect hook.
#[test]
fn trails_v627() {
    let Some(p) = sample("v627.insv") else {
        return;
    };
    let (_l, g, sx, cx) = gpu();
    let t0 = 165.0 * 1001.0 / 30000.0;
    let tr = Trails::new(
        &cx,
        3840,
        1920,
        FORMATS[0],
        Light {
            thr: 0.45,
            soft: 0.2,
            decay: 0.5f32.powf(1.0 / (4.0 * 29.97)),
            gain: 1.0,
            add: false,
        },
    );
    export(
        &g,
        &sx,
        &p,
        [t0, t0 + 10.0],
        3840,
        Some((stab::Lock::Direction, 0.0)),
        tr.hook([0.0, f64::INFINITY]),
        "v627_trails",
        &[2.0, 5.0, 9.9],
        2,
    );
}

/// A tracker sample with a box worth cutting.
fn boxed(s: &track::Sample) -> bool {
    matches!(
        s.state,
        track::State::Tracked | track::State::Checked | track::State::Found
    )
}

/// IoU of the mask's bounding box (probability > ½, inside the paste's gate) with the subject's box, in the crop.
fn mask_iou(m: &[f32], size: glam::Vec2, margin: f32) -> f32 {
    let n = effects::Cuts::SIDE as usize;
    let b = (size / 2.0).to_array().map(|a| a.to_radians().tan());
    let e = b[0].max(b[1]) * margin;
    let g = [b[0] / e, b[1] / e];
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for (i, &v) in m.iter().enumerate() {
        let p = [(i % n) as f32, (i / n) as f32].map(|x| (x + 0.5) / n as f32 * 2.0 - 1.0);
        if v > 0.5 && p[0].abs() < 1.2 * g[0] && p[1].abs() < 1.1 * g[1] {
            for a in 0..2 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
    }
    let i: f32 = (0..2)
        .map(|a| (hi[a].min(g[a]) - lo[a].max(-g[a])).max(0.0))
        .product();
    let u = 4.0 * g[0] * g[1] + (0..2).map(|a| (hi[a] - lo[a]).max(0.0)).product::<f32>() - i;
    i / u.max(1e-9)
}

/// Clones on v627 frames 300..480 (6 s at night, the camera on a stick held still, people walking past): the man in
/// the green jacket (`walker`) on frames stitched with the gyro's direction lock. A Freeze Trail still at 2880 (a cut
/// each time he moved 1.5 of his widths, over the median of the clip), the masks against the tracker's box (IoU), then
/// export jobs at 3840 with the Freeze Trail and Echo Trail hooks, and the passes' GPU times at 5760 × 2880.
#[test]
fn clones_v627() {
    let Some(p) = sample("v627.insv") else {
        return;
    };
    let (_l, g, sx, cx) = gpu();
    let c = effects::Clip::open(&p).unwrap();
    let lock = c.poses(stab::Lock::Direction, 0.0).unwrap();
    let pose = |k: usize| c.pose(k, Some(&lock));
    let (k0, k1) = (300usize, 480usize);
    let w = 2880u32;
    let mut cuts = effects::Cuts::new(&g, &cx, 9, 512, FORMATS[0]).unwrap();
    let mut med = Median::new(&cx, w, w / 2, 18, FORMATS[0]);
    let (mut iou, mut slots, mut last) = (vec![], vec![], None::<(Vec3, f32)>);
    let t = Instant::now();
    let smp = walker(&g, &sx, &c, &pose, k0, k1, w, |f, k, s| {
        if (k - k0) % 10 == 0 {
            submit(&cx, |e| med.push(e, f));
        }
        // a copy each time the subject moved 1.5 of its widths
        if boxed(s)
            && slots.len() < 9
            && last.is_none_or(|(l, w)| l.angle_between(s.raw).to_degrees() >= 1.5 * w)
        {
            let slot = slots.len() as u32;
            submit(&cx, |e| cuts.cut(e, f, slot, s.raw, s.size));
            submit(&cx, |e| assert_eq!(cuts.seg(e).unwrap(), Some(slot)));
            iou.push(mask_iou(&cuts.mask(slot), s.size, cuts.margin));
            slots.push(slot);
            last = Some((s.raw, s.size.x));
        }
    });
    let states: Vec<char> = smp
        .iter()
        .map(|s| format!("{:?}", s.state).chars().next().unwrap())
        .collect();
    eprintln!(
        "tracked {} frames in {:.1} s: {}; mask bbox vs box IoU per cut {:?}",
        smp.len(),
        t.elapsed().as_secs_f64(),
        states.iter().collect::<String>(),
        iou.iter()
            .map(|x| (x * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    let o = frame(&cx, w, w / 2, FORMATS[0]);
    submit(&cx, |e| med.run(e, &o));
    let plate = read(&cx, &o);
    submit(&cx, |e| cuts.paste(e, &o, &slots));
    let px = read(&cx, &o);
    jpg(&px, w, w / 2, 1, "v627_freeze_still.jpg");
    // the subject's stretch of the path (across the seam), plate above clones
    let x0 = pixel(w, smp[0].raw)[0];
    let xs: Vec<f64> = smp
        .iter()
        .filter(|s| boxed(s))
        .map(|s| {
            x0 + (pixel(w, s.raw)[0] - x0 + w as f64 / 2.0).rem_euclid(w as f64) - w as f64 / 2.0
        })
        .collect();
    let (lo, hi) = xs
        .iter()
        .fold((f64::MAX, f64::MIN), |a, &x| (a.0.min(x), a.1.max(x)));
    let y = pixel(w, smp[0].raw)[1];
    let (cw, ch) = ((hi - lo + 160.0) as u32 & !1, 240u32);
    let both: Vec<u8> = [&plate, &px]
        .iter()
        .flat_map(|p| {
            (0..ch).flat_map(move |r| {
                (0..cw).flat_map(move |c| {
                    let x = ((lo - 80.0) as i64 + c as i64).rem_euclid(w as i64) as u32;
                    let i = (4 * ((y as u32 - 150 + r) * w + x)) as usize;
                    p[i..i + 4].to_vec()
                })
            })
        })
        .collect();
    jpg(&both, cw, 2 * ch, 1, "v627_freeze_still_crop.jpg");
    // he walks about 25° in these 6 s: four copies 1.5 of his widths apart
    assert!(iou.len() >= 4 && iou.iter().all(|&x| x > 0.4));

    // GPU times at 5760 × 2880: a cut with its mask (network run waited for), a paste of 8 cuts
    let big = frame(&cx, 5760, 2880, FORMATS[0]);
    submit(&cx, |e| {
        e.copy_texture_to_texture(o.as_image_copy(), big.as_image_copy(), o.size())
    });
    let s0 = smp[0];
    let tc = time(&cx, 10, |e| cuts.cut(e, &big, 0, s0.raw, s0.size));
    let t = Instant::now();
    for _ in 0..10 {
        submit(&cx, |e| cuts.cut(e, &big, 0, s0.raw, s0.size));
        submit(&cx, |e| {
            cuts.seg(e).unwrap();
        });
    }
    wait(&cx);
    let ts = t.elapsed().as_secs_f64() * 1e2;
    let tp = time(&cx, 10, |e| {
        cuts.paste(e, &big, &slots[..8.min(slots.len())])
    });
    eprintln!(
        "at 5760 × 2880: cut {tc:.2} ms, cut + mask {ts:.1} ms, paste of {} cuts {tp:.2} ms (GPU shared, noisy)",
        8.min(slots.len())
    );

    let sub = |m: Vec<track::Sample>| {
        move |k: u64| {
            m.get(k as usize)
                .filter(|s| boxed(s))
                .map(|s| (s.raw, s.size))
        }
    };
    let t = [c.times[k0], c.times[k1]];
    for (name, mode) in [
        ("v627_freeze", effects::Clones::Freeze { gap: 1.2, n: 8 }),
        ("v627_echo", effects::Clones::Echo { lag: 20, n: 4 }),
    ] {
        let cu = effects::Cuts::new(&g, &cx, mode.slots(), 512, FORMATS[0]).unwrap();
        let hook = cu.hook(mode, sub(smp.clone())).unwrap();
        export(
            &g,
            &sx,
            &p,
            t,
            3840,
            Some((stab::Lock::Direction, 0.0)),
            hook,
            name,
            &[1.0, 3.0, 5.9],
            2,
        );
    }
}

/// The man in the green jacket walking along the left of v627 from frame `k0` to `k1` (pixel (115, 1843) of frame
/// 300 as the camera framed it), picked by the detector and followed by `track::Tracker` on frames stitched at `w`
/// with `pose`; `f(frame, k, sample)` after each frame's step.
#[allow(clippy::too_many_arguments)]
fn walker(
    g: &Gpu,
    sx: &stitch::Ctx,
    c: &effects::Clip,
    pose: &dyn Fn(usize) -> stitch::Pose,
    k0: usize,
    k1: usize,
    w: u32,
    mut f: impl FnMut(&wgpu::Texture, usize, &track::Sample),
) -> Vec<track::Sample> {
    let d0 = pose(300).q.inverse()
        * c.pose(300, None).q
        * vis::ray(5760, 2880, [115.0, 1843.0]).as_vec3();
    let mut ai = track::Ai::new(&g.instance, &g.adapter, &g.device, &g.queue).unwrap();
    let ks: Vec<usize> = (k0..k1).collect();
    let (mut pano, mut tr, mut smp) = (None, None, vec![]);
    c.frames(g, sx, w, false, &ks, pose, |t, k| {
        let p = pano.get_or_insert_with(|| ai.pano(&t.create_view(&Default::default())));
        let s = match &mut tr {
            None => {
                let d = ai.pick(p, d0).ok_or("no subject at the direction")?;
                eprintln!(
                    "subject: {} {:.2}, {:.1} × {:.1}°",
                    track::CLASSES[d.cls as usize],
                    d.score,
                    d.size.x,
                    d.size.y
                );
                tr = Some(track::Tracker::new(&mut ai, p, d.into())?);
                track::Sample {
                    raw: d.dir,
                    dir: d.dir,
                    size: d.size,
                    score: 1.0,
                    state: track::State::Tracked,
                }
            }
            Some(t) => t.step(&mut ai, p),
        };
        f(t, k, &s);
        smp.push(s);
        Ok(())
    })
    .unwrap();
    smp
}

/// Pixel position of direction `d` in a `w` × `w / 2` equirect (inverse of `vis::ray`).
fn pixel(w: u32, d: Vec3) -> [f64; 2] {
    let d = d.as_dvec3();
    let th = (-d.y).atan2(d.x).rem_euclid(std::f64::consts::TAU);
    [
        th * w as f64 / std::f64::consts::TAU,
        d.z.clamp(-1.0, 1.0).acos() * (w / 2 - 1) as f64 / std::f64::consts::PI,
    ]
}

/// Gaussian numbers (Box–Muller).
fn gauss(s: &mut u64) -> f64 {
    (-2.0 * rng(s).max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * rng(s)).cos()
}

/// Frame Stack on a synthetic night: 3000 stars (brightness power law, Gaussian PSF) turning about the celestial pole
/// at 52.5° elevation by 0.25° per frame (a photo a minute), a static ground below the horizon, and noise of σ 0.03
/// on every frame (8-bit sRGB, 2048 × 1024). Each frame aligned to the one before (`effects::align`), the fits
/// chained, against the true rotation; then stacked: the mean (noise in a starless patch of sky against one frame's,
/// √40 ≈ 6.3 expected) and peaks of bright stars against a mean without alignment, and the maximum of the frames as
/// they are (star trails).
#[test]
fn stars_synthetic() {
    let (_l, _g, _sx, cx) = gpu();
    let (w, h, n) = (2048usize, 1024usize, 40usize);
    let mut s = 11u64;
    let lat = 52.5f64.to_radians();
    let pole = DVec3::new(lat.cos(), 0.0, lat.sin());
    let stars: Vec<(DVec3, f64)> = (0..3000)
        .map(|_| {
            let z = 2.0 * rng(&mut s) - 1.0;
            let a = std::f64::consts::TAU * rng(&mut s);
            let r = (1.0 - z * z).sqrt();
            (
                DVec3::new(r * a.cos(), r * a.sin(), z),
                0.05 + 0.9 * rng(&mut s).powi(6),
            )
        })
        .collect();
    // ground: blocks of grey below elevation 0 (a skyline up to 3°)
    let ground = |x: usize, y: usize| -> Option<f64> {
        let el = 90.0 - 180.0 * y as f64 / (h - 1) as f64;
        let b = (x / 37) as u64;
        let top = 3.0 * ((b.wrapping_mul(2654435761) >> 7) % 100) as f64 / 100.0;
        (el < top).then(|| 0.1 + 0.08 * (((x / 11) ^ (y / 7)) % 5) as f64 / 5.0)
    };
    let step = 0.25f64.to_radians();
    let mut render = |i: usize| {
        let q = DQuat::from_axis_angle(pole, step * i as f64);
        let mut l = vec![0.004f64; w * h];
        for &(d, b) in &stars {
            let d = q * d;
            let th = (-d.y).atan2(d.x).rem_euclid(std::f64::consts::TAU);
            let ph = d.z.clamp(-1.0, 1.0).acos();
            let (px, py) = (
                th * w as f64 / std::f64::consts::TAU,
                ph * (h - 1) as f64 / std::f64::consts::PI,
            );
            // the PSF is round on the sphere: wider in x by 1 / sin φ
            let sx = 0.8 / ph.sin().max(0.05);
            let rx = (3.0 * sx).ceil() as i64;
            for dy in -3i64..=3 {
                for dx in -rx..=rx {
                    let (x, y) = (
                        (px.round() as i64 + dx).rem_euclid(w as i64) as usize,
                        py.round() as i64 + dy,
                    );
                    if y < 0 || y >= h as i64 {
                        continue;
                    }
                    let (ex, ey) = (px.round() + dx as f64 - px, py.round() + dy as f64 - py);
                    l[y as usize * w + x] +=
                        b * (-0.5 * (ex * ex / (sx * sx) + ey * ey / 0.64)).exp();
                }
            }
        }
        let mut px = vec![255u8; 4 * w * h];
        for y in 0..h {
            for x in 0..w {
                let v = ground(x, y).unwrap_or(srgb(l[y * w + x].min(1.0)));
                let c = (v + 0.03 * gauss(&mut s)).clamp(0.0, 1.0);
                px[4 * (y * w + x)..][..3].fill((c * 255.0).round() as u8);
            }
        }
        (px, l)
    };
    let frames: Vec<Vec<u8>> = (0..n).map(|i| render(i).0).collect();
    // frame r without noise: where it is the background, the sky is starless
    let clean = render(n / 2).1;
    // registration on the middle frame: the true rotation of frame i is (i − r)·step about the pole
    let (band, r) = ([10f64.to_radians(), 85f64.to_radians()], n / 2);
    let pan: Vec<vis::Pano> = frames.iter().map(|f| vis::Pano::rgba(f, w, h)).collect();
    let t = Instant::now();
    let reg = effects::series(&pan, r, band);
    let secs = t.elapsed().as_secs_f64();
    let reg: Vec<(DQuat, f64)> = reg
        .into_iter()
        .map(|x| x.expect("every frame registered"))
        .collect();
    let err: Vec<f64> = (0..n)
        .map(|i| {
            stab::log(
                reg[i].0.inverse() * DQuat::from_axis_angle(pole, step * (i as f64 - r as f64)),
            )
            .length()
            .to_degrees()
        })
        .collect();
    let worst = err.iter().cloned().fold(0.0, f64::max);
    let rms: Vec<f64> = reg
        .iter()
        .filter(|x| x.1 > 0.0)
        .map(|x| x.1.to_degrees() * 3600.0)
        .collect();
    let axis = stab::log(reg[0].0.inverse() * reg[n - 1].0).normalize();
    eprintln!(
        "{n} frames registered on frame {r} in {secs:.2} s: rms of the fits {:.0}..{:.0}″; rotation off the truth by \
         {:.4}° at most (frames up to {:.2}° from the reference); axis of the first-to-last rotation {:.3}° from the pole",
        rms.iter().cloned().fold(f64::MAX, f64::min),
        rms.iter().cloned().fold(0.0, f64::max),
        worst,
        step.to_degrees() * r as f64,
        axis.angle_between(pole).to_degrees()
    );
    let qs: Vec<Quat> = reg.iter().map(|x| x.0.as_quat()).collect();
    assert!(worst < 0.02);
    let tex: Vec<wgpu::Texture> = frames
        .iter()
        .map(|f| effects::image(&cx, f, w as u32, h as u32))
        .collect();
    // stacks into half floats, measured in linear light
    let o = frame(&cx, w as u32, h as u32, FORMATS[1]);
    let lin = |v: f64| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut out: Vec<Vec<f64>> = vec![];
    for (name, max, al) in [
        ("mean", false, true),
        ("trails", true, false),
        ("unaligned", false, false),
    ] {
        let mut st = Stack::new(&cx, w as u32, h as u32, max);
        st.horizon = Some([-1.0, 4.0]);
        for (i, t) in tex.iter().enumerate() {
            submit(&cx, |e| {
                st.push(e, t, if al { qs[i] } else { Quat::IDENTITY })
            });
        }
        submit(&cx, |e| st.run(e, &o));
        let px = read(&cx, &o);
        let v: Vec<f64> = (0..w * h)
            .map(|i| h16(u16::from_le_bytes([px[8 * i], px[8 * i + 1]])))
            .collect();
        let b: Vec<u8> = v
            .iter()
            .flat_map(|&x| {
                let c = (x.clamp(0.0, 1.0) * 255.0).round() as u8;
                [c, c, c, 255]
            })
            .collect();
        jpg(&b, w as u32, h as u32, 1, &format!("stars_{name}.jpg"));
        out.push(v.into_iter().map(lin).collect());
    }
    jpg(&frames[r], w as u32, h as u32, 1, "stars_frame.jpg");
    let fr: Vec<f64> = (0..w * h)
        .map(|i| lin(frames[r][4 * i] as f64 / 255.0))
        .collect();
    // noise: σ on the starless sky (elevation 20..70°, no star within 4 px) in frame r and in the aligned mean (about
    // their mean: noise added to sRGB values brightens the linear mean)
    let free: Vec<usize> = (h / 9..h * 7 / 18)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            (-4i64..=4).all(|d| {
                (-4i64..=4).all(|e| {
                    clean
                        [(y as i64 + d) as usize * w + (x as i64 + e).rem_euclid(w as i64) as usize]
                        < 0.0042
                })
            })
        })
        .map(|(x, y)| y * w + x)
        .collect();
    let sd = |p: &[f64]| {
        let m = free.iter().map(|&i| p[i]).sum::<f64>() / free.len() as f64;
        (free.iter().map(|&i| (p[i] - m).powi(2)).sum::<f64>() / free.len() as f64).sqrt()
    };
    let (n0, n1) = (sd(&fr), sd(&out[0]));
    // sharpness: the 50 brightest stars' peaks (frame r's positions, high in the sky) in frame r and the stacks
    let qr = DQuat::from_axis_angle(pole, step * r as f64);
    let mut peaks: Vec<(f64, DVec3)> = stars
        .iter()
        .map(|&(d, b)| (b, qr * d))
        .filter(|x| x.1.z > 0.3)
        .collect();
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    let pk = |p: &[f64]| {
        peaks[..50]
            .iter()
            .map(|(_, d)| {
                let [x, y] = pixel(w as u32, d.as_vec3());
                let mut m = 0f64;
                for dy in -2i64..=2 {
                    for dx in -3i64..=3 {
                        let (x, y) = (
                            (x.round() as i64 + dx).rem_euclid(w as i64) as usize,
                            (y.round() as i64 + dy) as usize,
                        );
                        m = m.max(p[y * w + x]);
                    }
                }
                m
            })
            .sum::<f64>()
            / 50.0
    };
    let (p0, pa, pu) = (pk(&fr), pk(&out[0]), pk(&out[2]));
    eprintln!(
        "linear light: noise on {} starless pixels σ {n0:.5} in frame {r}, {n1:.5} in the mean \
         ({:.1}× less, √{n} = {:.1}, more where the bilinear resampling averages neighbours too); mean peak of the 50 \
         brightest stars: frame {r} {p0:.3}, aligned mean {pa:.3}, unaligned mean {pu:.3}",
        free.len(),
        n0 / n1,
        (n as f64).sqrt()
    );
    assert!(n0 / n1 > 4.5 && pa > 0.6 * p0 && pu < 0.5 * pa);
    let (w, h) = (5760u32, 2880u32);
    let mut st = Stack::new(&cx, w, h, false);
    st.horizon = Some([-1.0, 4.0]);
    let (f, o, q) = (
        frame(&cx, w, h, FORMATS[0]),
        frame(&cx, w, h, FORMATS[0]),
        qs[0],
    );
    let tp = time(&cx, 20, |e| st.push(e, &f, q));
    let tr = time(&cx, 20, |e| st.run(e, &o));
    eprintln!(
        "stack at {w}x{h}: {tp:.2} ms per frame in, {tr:.2} ms for the result (GPU shared, noisy)"
    );
}

/// Frame Delay against its formula: frame k a flat colour encoding k, so the delayed frame's value tells which
/// frames each pixel blends (columns, rows, longitude, reversed; before and after the ring fills; 8- and 16-bit),
/// and the pass's GPU time at 5760 × 2880.
#[test]
fn delay_synthetic() {
    let (_l, _g, _sx, cx) = gpu();
    let (w, h, n) = (360u32, 180u32, 9u32);
    for (fmt, bpp) in [(FORMATS[0], 4usize), (FORMATS[1], 8)] {
        for (map, rev) in [
            (Map::Cols, false),
            (Map::Rows, true),
            (Map::Lon(Vec3::NEG_X), false),
        ] {
            let mut d = Delay::new(&cx, w, h, n, fmt, map);
            d.rev = rev;
            d.span = 6.0;
            let f = frame(&cx, w, h, fmt);
            let mut worst = 0f64;
            for k in 0..14u32 {
                // value 10 + 15k (0..255 scale) in red
                let v = (10 + 15 * k) as f64 / 255.0;
                let px: Vec<u8> = (0..w * h)
                    .flat_map(|_| {
                        let c = [v, 0.5, 0.5, 1.0];
                        if bpp == 4 {
                            c.map(|x| (x * 255.0).round() as u8).to_vec()
                        } else {
                            c.iter().flat_map(|&x| f16(x).to_le_bytes()).collect()
                        }
                    })
                    .collect();
                cx.q.write_texture(
                    f.as_image_copy(),
                    &px,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bpp as u32 * w),
                        rows_per_image: None,
                    },
                    f.size(),
                );
                submit(&cx, |e| d.push(e, &f));
                let o = read(&cx, &f);
                // (longitude is undefined at the poles)
                for (x, y) in [
                    (0u32, 2u32),
                    (90, 45),
                    (180, 90),
                    (270, 135),
                    (359, 177),
                    (45, 170),
                ] {
                    let i = (y * w + x) as usize;
                    let got = if bpp == 4 {
                        o[4 * i] as f64
                    } else {
                        255.0 * h16(u16::from_le_bytes([o[8 * i], o[8 * i + 1]]))
                    };
                    let s = match map {
                        Map::Cols => x as f64 / (w - 1) as f64,
                        Map::Rows => y as f64 / (h - 1) as f64,
                        // centre column = −X, azimuth from there
                        _ => {
                            let a =
                                std::f64::consts::TAU * x as f64 / w as f64 - std::f64::consts::PI;
                            a.sin().atan2(a.cos()).abs() / std::f64::consts::PI
                        }
                    };
                    let s = if rev { 1.0 - s } else { s };
                    // frames back, at most to the oldest held
                    let t = (s * 6.0).min(k.min(n - 1) as f64);
                    let want = 10.0 + 15.0 * (k as f64 - t);
                    worst = worst.max((got - want).abs());
                }
            }
            eprintln!("{fmt:?} {map:?} rev {rev}: largest error {worst:.3} (0..255)");
            assert!(worst < 0.8);
        }
    }
    let (w, h) = (5760u32, 2880u32);
    let mut d = Delay::new(&cx, w, h, 31, FORMATS[0], Map::Lon(Vec3::NEG_X));
    let f = frame(&cx, w, h, FORMATS[0]);
    eprintln!(
        "delay at {w}x{h}, ring of 31: {:.2} ms per frame (GPU shared, noisy)",
        time(&cx, 20, |e| d.push(e, &f))
    );
}

/// Frame Delay through export jobs at 3840 with the gyro's direction lock: v627 frames 300..480 by longitude from
/// the gate (people at the sides lag up to 1.5 s), v606 frames 240..420 by rows (walking through the gate: the
/// bottom of the world 1 s behind its top).
#[test]
fn delay_footage() {
    let (_l, g, sx, cx) = gpu();
    for (clip, k0, map, n) in [
        ("v627", 300usize, Map::Lon(Vec3::NEG_X), 46u32),
        ("v606", 240, Map::Dir(Vec3::Z), 31),
    ] {
        let Some(p) = sample(&format!("{clip}.insv")) else {
            return;
        };
        let c = effects::Clip::open(&p).unwrap();
        let d = Delay::new(&cx, 3840, 1920, n, FORMATS[0], map);
        let t = [c.times[k0], c.times[k0 + 180]];
        export(
            &g,
            &sx,
            &p,
            t,
            3840,
            Some((stab::Lock::Direction, 0.0)),
            d.hook(),
            &format!("{clip}_delay"),
            &[1.0, 3.0, 5.9],
            2,
        );
    }
}

/// Planet Pair through an export job: v606 frames 0..90 (the gate in daylight) at 3840 with the horizon lock, the
/// tiny planet and its sky world side by side.
#[test]
fn planets_v606() {
    let Some(p) = sample("v606.insv") else {
        return;
    };
    let (_l, g, sx, _cx) = gpu();
    let c = effects::Clip::open(&p).unwrap();
    let pl = effects::Planets::new(&sx, FORMATS[0]);
    export(
        &g,
        &sx,
        &p,
        [c.times[0], c.times[90]],
        3840,
        Some((stab::Lock::Horizon, 0.5)),
        pl.hook(),
        "v606_planets",
        &[0.5, 2.9],
        2,
    );
}

/// An editor timeline rendered flat (1280 × 720, H.264) with the gyro's `lock`; frames at `at` (output s).
fn render(
    g: &Gpu,
    sx: &stitch::Ctx,
    src: &Path,
    tl: edit::Timeline,
    lock: stab::Lock,
    name: &str,
    at: &[f64],
) {
    let path = out(&format!("{name}.mp4"));
    let job = export::Job {
        cut: export::Cut::Edit(Box::new(export::Edit {
            tl,
            files: vec![vec![src.to_path_buf()]],
            looks: Default::default(),
            denoise: None,
            over: vec![],
            abr: 192_000,
        })),
        path: path.clone(),
        out: export::Out::Flat {
            w: 1280,
            h: 720,
            view: None,
        },
        enc: export::Enc::Video {
            codec: video::Codec::H264,
            deep: false,
            rate: video::Rate::Cbr(12_000_000),
            gop: 30,
        },
        raft: false,
        colour: Some(0.1),
        stab: Some((lock, 0.0)),
        fx: None,
        flat: false,
        pre: None,
    };
    let t0 = Instant::now();
    let p = export::run(g, sx, &mut None, job, &AtomicBool::new(false), &mut |_| {}).unwrap();
    eprintln!(
        "{name}.mp4: {} frames in {:.1} s",
        p.done,
        t0.elapsed().as_secs_f64()
    );
    let row: Vec<String> = at.iter().map(|x| format!("{x}")).collect();
    let o = out(&format!("{name}_strip.jpg"));
    // the frames at `at` side by side, 320 × 180 each
    let sel = at
        .iter()
        .map(|x| format!("eq(n\\,{})", (x * 30000.0 / 1001.0).round()))
        .collect::<Vec<_>>()
        .join("+");
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&path)
        .args([
            "-vf",
            &format!("select='{sel}',scale=320:180,tile={}x1", at.len()),
            "-frames:v",
            "1",
            "-q:v",
            "3",
        ])
        .arg(&o)
        .status()
        .unwrap();
    assert!(ok.success(), "{name}: frames at {row:?}");
}

/// The editing presets as editor timelines (crate edit's checks pass in `cut::timeline`), checked against what they
/// promise at sampled output times, then rendered: Stop Motion, Orbit Lapse and Roll Lapse on v606 (a walk through
/// the gate), Subject Framing, Subject Spin, Pull Out and Dolly Zoom on v627 with the walker's track (`walker`, its
/// directions smoothed by `track::smooth`).
#[test]
fn presets() {
    use effects::cut;
    let fps = edit::Rate {
        num: 30000,
        den: 1001,
    };
    let (Some(p6), Some(p7)) = (sample("v606.insv"), sample("v627.insv")) else {
        return;
    };
    let (_l, g, sx, _cx) = gpu();
    let c6 = effects::Clip::open(&p6).unwrap();
    let dur6 = c6.times.len() as f64 * 1001.0 / 30000.0;
    let path6 = p6.to_str().unwrap();

    let sm = cut::stop_motion(0, 2.0, 8.0, 0.5, 1.0 / 6.0).unwrap();
    let tl = cut::timeline(path6, dur6, fps, sm).unwrap();
    let srcs: Vec<f64> = (0..tl.len()).map(|n| tl.frame(n).unwrap().src).collect();
    let jumps = srcs.windows(2).filter(|w| w[1] != w[0]).count();
    eprintln!(
        "stop motion: {} clips, {} output frames ({:.2} s), the picture jumps {jumps} times",
        tl.clips.len(),
        tl.len(),
        tl.dur()
    );
    assert_eq!((tl.clips.len(), jumps), (12, 11));
    render(
        &g,
        &sx,
        &p6,
        tl,
        stab::Lock::Horizon,
        "v606_stopmotion",
        &[0.1, 0.5, 1.0, 1.5],
    );

    let v = edit::View::look(0.0, -10.0, 0.0, 100.0, edit::Proj::Persp);
    let ol = cut::orbit(0, 0.0, 30.0, 8.0, 2.0, v, 1.0, 1.0).unwrap();
    let tl = cut::timeline(path6, dur6, fps, vec![ol]).unwrap();
    let (e0, e1) = (tl.at(0.0).unwrap(), tl.at(tl.dur() / 2.0).unwrap());
    eprintln!(
        "orbit lapse: 30 s at up to 8× in {:.2} s; starts as {:?} at {:.0}°, speed {:.1}× halfway",
        tl.dur(),
        e0.view.proj,
        e0.view.fov,
        e1.speed
    );
    assert!(matches!(e0.view.proj, edit::Proj::Planet(_)) && e1.speed > 7.9 && tl.dur() < 6.0);
    render(
        &g,
        &sx,
        &p6,
        tl,
        stab::Lock::Horizon,
        "v606_orbit",
        &[0.0, 1.0, 2.0, 3.0, 4.0],
    );

    let rl = cut::roll(0, 0.0, 30.0, 8.0, 2.0, v, 1.0).unwrap();
    let tl = cut::timeline(path6, dur6, fps, vec![rl]).unwrap();
    let rolls: Vec<f32> = (0..=4)
        .map(|i| {
            tl.at(tl.dur() * i as f64 / 4.0 - 1e-3 * (i / 4) as f64)
                .unwrap()
                .view
                .ypr()[2]
        })
        .collect();
    eprintln!("roll lapse: roll at 0, ¼, ½, ¾, 1 of the clip {rolls:?}");
    render(
        &g,
        &sx,
        &p6,
        tl,
        stab::Lock::Horizon,
        "v606_roll",
        &[0.0, 1.0, 2.0, 3.0, 4.0],
    );

    // subject presets: the walker in v627 frames 300..480
    let c7 = effects::Clip::open(&p7).unwrap();
    let dur7 = c7.times.len() as f64 * 1001.0 / 30000.0;
    let path7 = p7.to_str().unwrap();
    let lock = c7.poses(stab::Lock::Direction, 0.0).unwrap();
    let pose = |k: usize| c7.pose(k, Some(&lock));
    let smp = walker(&g, &sx, &c7, &pose, 300, 480, 2880, |_, _, _| {});
    let dirs = track::smooth(&smp, 3.0);
    let subj: Vec<cut::Subj> = (0..smp.len())
        .filter(|&i| boxed(&smp[i]))
        .map(|i| cut::Subj {
            t: c7.times[300 + i],
            dir: dirs[i],
            size: smp[i].size,
        })
        .collect();
    let (a, b) = (c7.times[300], c7.times[479]);
    // where the view looks against where the subject is, and the subject's height on screen
    // the subject's height, median over ±7 samples (the tracker's box jitters)
    let hs: Vec<f32> = (0..subj.len())
        .map(|i| {
            let mut v: Vec<f32> = subj[i.saturating_sub(7)..(i + 8).min(subj.len())]
                .iter()
                .map(|s| s.size.y)
                .collect();
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        })
        .collect();
    let check = |name: &str, tl: &edit::Timeline| {
        let mut off = 0f32;
        let mut fill = vec![];
        // every 7th sample: keys sit every 15th, so most checks fall between them
        for (s, h) in subj.iter().zip(&hs).step_by(7) {
            let e = tl.at(s.t - a).unwrap();
            let [y, p, r] = e.view.ypr();
            let cam = track::Cam::view(y, p, r, e.view.fov, 16.0 / 9.0);
            off = off.max(cam.f.angle_between(s.dir).to_degrees());
            fill.push((h / 2.0).to_radians().tan() / cam.ext.y);
        }
        eprintln!(
            "{name}: view off the subject by {off:.2}° at most; subject height / frame height {:?}",
            fill.iter()
                .map(|x| (x * 100.0).round() / 100.0)
                .collect::<Vec<_>>()
        );
        (off, fill)
    };
    let fr = cut::framing(0, a, b, &subj, 0.4, 16.0 / 9.0, 0.5).unwrap();
    let tl = cut::timeline(path7, dur7, fps, vec![fr]).unwrap();
    let (off, fill) = check("subject framing", &tl);
    assert!(off < 2.0 && fill.iter().all(|f| (f - 0.4).abs() < 0.06));
    render(
        &g,
        &sx,
        &p7,
        tl,
        stab::Lock::Direction,
        "v627_framing",
        &[0.0, 1.5, 3.0, 4.5, 5.9],
    );

    let sp = cut::spin(0, a, b, &subj, 0.4, 16.0 / 9.0, 1.0).unwrap();
    let tl = cut::timeline(path7, dur7, fps, vec![sp]).unwrap();
    let (off, _) = check("subject spin", &tl);
    assert!(off < 2.0);
    render(
        &g,
        &sx,
        &p7,
        tl,
        stab::Lock::Direction,
        "v627_spin",
        &[0.0, 1.5, 3.0, 4.5, 5.9],
    );

    let po = cut::pull_out(0, a, b, &subj, 0.4, 16.0 / 9.0, 2.0).unwrap();
    let tl = cut::timeline(path7, dur7, fps, vec![po]).unwrap();
    let end = tl.at(tl.dur() - 1e-3).unwrap().view;
    eprintln!(
        "pull out: ends as {:?} at {:.0}°, pitch {:.0}°",
        end.proj,
        end.fov,
        end.ypr()[1]
    );
    render(
        &g,
        &sx,
        &p7,
        tl,
        stab::Lock::Direction,
        "v627_pullout",
        &[0.0, 2.0, 4.0, 5.0, 5.9],
    );

    let dz = cut::dolly(0, a, b, &subj, 40.0, 0.5).unwrap();
    let tl = cut::timeline(path7, dur7, fps, vec![dz]).unwrap();
    let (off, fill) = check("dolly zoom", &tl);
    let spread =
        fill.iter().cloned().fold(0f32, f32::max) / fill.iter().cloned().fold(f32::MAX, f32::min);
    assert!(off < 2.0 && spread < 1.15);
    render(
        &g,
        &sx,
        &p7,
        tl,
        stab::Lock::Direction,
        "v627_dolly",
        &[0.0, 1.5, 3.0, 4.5, 5.9],
    );
}

/// Share of the world mask's upper hemisphere (stitch's layout, `effects::MW` wide) where `f` holds, by solid angle,
/// weighted by `f`'s value.
fn upper(f: impl Fn(usize) -> f64) -> f64 {
    let (w, h) = (effects::MW as usize, effects::MW as usize / 2);
    let (mut a, mut n) = (0.0, 0.0);
    for y in 0..h / 2 {
        let s = (std::f64::consts::PI * y as f64 / (h - 1) as f64).sin();
        for x in 0..w {
            a += s * f(y * w + x);
            n += s;
        }
    }
    a / n
}

/// The short name of a clip: its number.
fn short(p: &Path) -> String {
    let s = p.file_stem().unwrap().to_string_lossy();
    s.rsplit('_').next().unwrap().to_string()
}

/// New Sky on single frames (`X5_SAMPLES`; with `X5_RIDES`, the rides' folder, rides too), each stitched at 3840
/// with the horizon lock and pushed twice (its own sky found, then replaced). The share of the upper hemisphere the
/// mask calls sky (by solid angle) against what each frame shows; with a test sky (magenta to orange, no colour match)
/// the ground below −30° untouched to the bit and the share of the upper hemisphere that changed by more than 40
/// levels within −0.05..+0.25 of the mask's (a soft night mask changes more than its share).
/// Pictures with a clear-sky gradient and the colour match (`sky_<clip>_<frame>.jpg`: the frame, the result, the mask
/// in the world), and the pass's time per frame at 2048, 3840 and 5760.
#[test]
fn sky_footage() {
    let Some(smp) = sample("v606.insv").map(|p| p.parent().unwrap().to_path_buf()) else {
        return;
    };
    let rides = std::env::var_os("X5_RIDES").map(PathBuf::from);
    let (_l, g, sx, cx) = gpu();
    // clip, frame, the mask's sky share of the upper hemisphere: at least, at most
    let mut cases = vec![
        // the gate under a blue sky (it and the buildings cover about half the hemisphere); at night; indoors under a
        // glass dome
        (smp.join("v606.insv"), 300, 0.4, 0.8),
        (smp.join("v627.insv"), 300, 0.6, 1.0),
        (smp.join("v635.insv"), 0, 0.0, 0.3),
    ];
    if let Some(r) = &rides {
        let r = |n: &str| r.join(format!("VID_{n}.insv"));
        cases.extend([
            // an overcast street; under a bridge; streets in the sun; at night (reported only)
            (r("20260413_173716_00_717"), 600, 0.35, 0.8),
            (r("20260413_173716_00_717"), 2400, 0.0, 0.08),
            (r("20260416_165209_00_738"), 600, 0.4, 0.8),
            (r("20260429_115150_00_689"), 3000, 0.45, 0.9),
            (r("20260304_210620_00_670"), 600, 0.0, 1.0),
            (r("20260311_183904_00_714"), 1000, 0.0, 1.0),
        ]);
    }
    let test = effects::gradient(1024, [255, 0, 255], [255, 128, 0]);
    let clear = effects::gradient(1024, [40, 95, 190], [190, 215, 240]);
    let (w, h) = (3840u32, 1920u32);
    let mut fails = vec![];
    for (p, k, lo, hi) in cases {
        let c = effects::Clip::open(&p).unwrap();
        let ps = c.poses(stab::Lock::Horizon, 0.5);
        let name = format!("{}_{k}", short(&p));
        c.frames(&g, &sx, w, false, &[k], |k| c.pose(k, ps.as_deref()), |f, k| {
            let q = c.world(k, ps.as_deref()).unwrap_or_default();
            let before = read(&cx, f);
            let swap = |sky: &[u8], tone| {
                put(&cx, f, &before);
                let mut s = effects::Sky::new(&g, &cx, sky, 1024, 512).unwrap();
                s.tone = tone;
                for i in 0..2 {
                    submit(&cx, |e| s.push(e, f, q, i as f64 / 30.0).unwrap());
                }
                (s.mask(), read(&cx, f), s)
            };
            let (m, after, _) = swap(&test, 0.0);
            let share = upper(|i| m[i] as f64);
            let ground = (4 * w * h * 2 / 3) as usize;
            let kept = before[ground..] == after[ground..];
            // changed by more than 40 levels in a channel, rows of the upper hemisphere by solid angle
            let (mut ch, mut n) = (0.0, 0.0);
            for y in 0..h / 2 {
                let s = (std::f64::consts::PI * y as f64 / (h - 1) as f64).sin();
                for x in 0..w {
                    let i = (4 * (y * w + x)) as usize;
                    ch += s * (0..3).any(|c| before[i + c].abs_diff(after[i + c]) > 40) as u8 as f64;
                    n += s;
                }
            }
            let ch = ch / n;
            eprintln!(
                "sky {name}: mask {share:.3} of the upper hemisphere (expected {lo}..{hi}), changed {ch:.3}, \
                 ground kept {kept}"
            );
            if !(lo..=hi).contains(&share) || !kept || ch < share - 0.05 || ch > share + 0.25 {
                fails.push(name.clone());
            }
            let (m, after, mut s) = swap(&clear, 0.6);
            let mw = effects::MW;
            let mut px = before.clone();
            px.extend(&after);
            px.extend((0..w * h).flat_map(|i| {
                let v = (255.0 * m[((i / w * mw / 2 / h) * mw + i % w * mw / w) as usize]) as u8;
                [v, v, v, 255]
            }));
            jpg(&px, w, 3 * h, 3, &format!("sky_{name}.jpg"));
            if name == "v606_300" {
                put(&cx, f, &before);
                let ms = time(&cx, 10, |e| s.push(e, f, q, 1.0).unwrap());
                let sizes = [2048u32, 5760].map(|w| {
                    let t = frame(&cx, w, w / 2, FORMATS[0]);
                    time(&cx, 10, |e| s.push(e, &t, q, 1.0).unwrap())
                });
                eprintln!(
                    "sky: {:.1} ms a frame at 2048, {ms:.1} at 3840, {:.1} at 5760 (network included)",
                    sizes[0], sizes[1]
                );
            }
            Ok(())
        })
        .unwrap();
    }
    assert!(fails.is_empty(), "{fails:?}");
}

/// New Sky's flicker on still stretches stitched at 2048 with the horizon lock: v627 frames 300..420 (night, the
/// camera on a stick held still) and, with `X5_RIDES`, ride 689 frames 2460..2700 (day, stopped 41–45 s). From the 16th
/// frame on, per frame: the world mask's mean change from the frame before over the upper hemisphere (by solid angle)
/// and at the edge (where it was 0.05..0.95), the share whose side of ½ flipped; the share of the upper half's pixels
/// that changed by more than 8 levels in a channel, the result's against the frames' own. Without smoothing (half-life
/// 0) and with the default.
#[test]
fn sky_flicker() {
    let Some(p7) = sample("v627.insv") else {
        return;
    };
    let mut cases = vec![(p7, 300, 420)];
    if let Some(r) = std::env::var_os("X5_RIDES").map(PathBuf::from) {
        cases.push((r.join("VID_20260429_115150_00_689.insv"), 2460, 2700));
    }
    let (_l, g, sx, cx) = gpu();
    let clear = effects::gradient(1024, [40, 95, 190], [190, 215, 240]);
    let w = 2048u32;
    // share of the upper half's pixels (by solid angle) where some channel differs by more than 8 levels
    let moved = |a: &[u8], b: &[u8]| {
        let (mut c, mut n) = (0.0, 0.0);
        for y in 0..w / 4 {
            let s = (std::f64::consts::PI * y as f64 / (w / 2 - 1) as f64).sin();
            for x in 0..w {
                let i = (4 * (y * w + x)) as usize;
                c += s * (0..3).any(|j| a[i + j].abs_diff(b[i + j]) > 8) as u8 as f64;
                n += s;
            }
        }
        c / n
    };
    for (p, a, b) in cases {
        let c = effects::Clip::open(&p).unwrap();
        let ps = c.poses(stab::Lock::Horizon, 0.5);
        let ks: Vec<usize> = (a..=b).collect();
        let mut res = vec![];
        for half in [0.0, 0.1] {
            let mut s = effects::Sky::new(&g, &cx, &clear, 1024, 512).unwrap();
            s.half = half;
            let mut last: Option<(Vec<f32>, Vec<u8>, Vec<u8>)> = None;
            let mut sum = [0.0; 5];
            let mut n = 0;
            c.frames(
                &g,
                &sx,
                w,
                false,
                &ks,
                |k| c.pose(k, ps.as_deref()),
                |f, k| {
                    let q = c.world(k, ps.as_deref()).unwrap_or_default();
                    let i = read(&cx, f);
                    submit(&cx, |e| s.push(e, f, q, c.times[k]).unwrap());
                    let (m, o) = (s.mask(), read(&cx, f));
                    if let Some((lm, lo, li)) = &last
                        && k >= a + 16
                    {
                        let d = upper(|j| (m[j] - lm[j]).abs() as f64);
                        let fl = upper(|j| ((m[j] > 0.5) != (lm[j] > 0.5)) as u8 as f64);
                        let edge = |j: usize| (0.05..0.95).contains(&lm[j]) as u8 as f64;
                        let de = upper(|j| edge(j) * (m[j] - lm[j]).abs() as f64) / upper(edge);
                        for (x, v) in sum
                            .iter_mut()
                            .zip([d, fl, moved(&o, lo), moved(&i, li), de])
                        {
                            *x += v;
                        }
                        n += 1;
                    }
                    last = Some((m, o, i));
                    Ok(())
                },
            )
            .unwrap();
            let r = sum.map(|x| x / n as f64);
            eprintln!(
                "sky flicker {} {a}..{b}, half-life {half} s: mask change {:.4} (at the edge {:.4}), flipped {:.4}; \
                 pixels changed: result {:.4}, frames {:.4} ({n} frames)",
                short(&p),
                r[0],
                r[4],
                r[1],
                r[2],
                r[3]
            );
            res.push(r);
        }
        assert!(res[1][0] < res[0][0] && res[1][4] < res[0][4] && res[1][1] <= res[0][1]);
    }
}

/// New Sky through export jobs at 3840 with the horizon lock, the first frame pushed once before (as the app's maker
/// does): with `X5_RIDES` ride 717 from 8 to 14 s (60 fps, trees and wires against an overcast sky) and ride 670 from
/// 8 to 14 s (night); v627 frames 300..390 (night, still). Frames at 1 and 2.5 s, and strips of 8 consecutive frames
/// of the sky ahead from 2 s (`sky_<clip>_strip.jpg`, ffmpeg's tile).
#[test]
fn sky_export() {
    let Some(p7) = sample("v627.insv") else {
        return;
    };
    let (_l, g, sx, cx) = gpu();
    let c7 = effects::Clip::open(&p7).unwrap();
    let mut cases = vec![(p7.clone(), [c7.times[300], c7.times[390]])];
    if let Some(r) = std::env::var_os("X5_RIDES").map(PathBuf::from) {
        cases.push((r.join("VID_20260413_173716_00_717.insv"), [8.0, 14.0]));
        cases.push((r.join("VID_20260304_210620_00_670.insv"), [8.0, 14.0]));
    }
    let clear = effects::gradient(1024, [40, 95, 190], [190, 215, 240]);
    let w = 3840;
    for (p, t) in cases {
        let c = effects::Clip::open(&p).unwrap();
        let st = (stab::Lock::Horizon, 0.5);
        let ps = c.poses(st.0, st.1);
        let k0 = c.times.partition_point(|&x| x < t[0] - 1e-3);
        let q: Vec<Quat> = (k0..c.times.len())
            .map(|k| c.world(k, ps.as_deref()).unwrap_or_default())
            .collect();
        let mut s = effects::Sky::new(&g, &cx, &clear, 1024, 512).unwrap();
        c.frames(
            &g,
            &sx,
            w,
            false,
            &[k0],
            |k| c.pose(k, ps.as_deref()),
            |f, _| {
                submit(&cx, |e| s.push(e, f, q[0], 0.0).unwrap());
                Ok(())
            },
        )
        .unwrap();
        let name = format!("sky_{}", short(&p));
        export(
            &g,
            &sx,
            &p,
            t,
            w,
            Some(st),
            s.hook(move |k| q.get(k as usize).copied().unwrap_or_default()),
            &name,
            &[1.0, 2.5],
            3,
        );
        let strip = out(&format!("{name}_strip.jpg"));
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-ss", "2", "-i"])
            .arg(strip.with_file_name(format!("{name}.mp4")))
            .args([
                "-frames:v",
                "1",
                "-vf",
                "crop=1280:640:1280:240,scale=640:320,tile=4x2",
                "-q:v",
                "3",
            ])
            .arg(&strip)
            .status()
            .unwrap();
        assert!(ok.success());
    }
}

/// New Sky's panorama stays put in the world: v606 frames 0 and 600 (walking through the gate, turning), stitched at
/// 2048 with the horizon lock, a black panorama with a magenta band 1.4° wide at azimuth 91° (no colour match). The
/// band's column 70° up in each result against where the frame's rotation into the world (`Clip::world`, the first
/// frame's heading taken out) puts it.
#[test]
fn sky_world() {
    let Some(p) = sample("v606.insv") else {
        return;
    };
    let (_l, g, sx, cx) = gpu();
    let (pw, w) = (1024u32, 2048u32);
    let pano: Vec<u8> = (0..pw * pw / 2)
        .flat_map(|i| match i % pw {
            256..260 => [255, 0, 255, 255],
            _ => [0, 0, 0, 255],
        })
        .collect();
    let mut s = effects::Sky::new(&g, &cx, &pano, pw, pw / 2).unwrap();
    s.tone = 0.0;
    let c = effects::Clip::open(&p).unwrap();
    let ps = c.poses(stab::Lock::Horizon, 0.5);
    let q0 = c.world(0, ps.as_deref()).unwrap();
    let y0 = Quat::from_rotation_z(track::yaw_pitch(q0 * Vec3::NEG_X)[0].to_radians());
    let y = (20.0 / 180.0 * (w / 2 - 1) as f64).round() as u32;
    let mut seen = vec![];
    c.frames(&g, &sx, w, false, &[0, 600], |k| c.pose(k, ps.as_deref()), |f, k| {
        let q = c.world(k, ps.as_deref()).unwrap();
        for i in 0..1 + (k == 0) as u32 {
            submit(&cx, |e| s.push(e, f, q, (k + i as usize) as f64 / 30.0).unwrap());
        }
        let px = read(&cx, f);
        let cols: Vec<f64> = (0..w)
            .filter(|&x| {
                let i = (4 * (y * w + x)) as usize;
                px[i] > 200 && px[i + 1] < 60 && px[i + 2] > 200
            })
            .map(|x| x as f64)
            .collect();
        // the band's middle (column 258) 70° up, in the world, then in the frame
        let (th, ph) = (std::f32::consts::TAU * 258.0 / pw as f32, 20f32.to_radians());
        let d = (y0 * q).inverse() * Vec3::new(ph.sin() * th.cos(), -ph.sin() * th.sin(), ph.cos());
        let want = (-d.y).atan2(d.x).rem_euclid(std::f32::consts::TAU) as f64 / std::f64::consts::TAU * w as f64;
        let got = cols.iter().sum::<f64>() / cols.len().max(1) as f64;
        eprintln!(
            "sky world: frame {k}, heading {:.1}°, band at column {got:.1} ({} px), expected {want:.1}",
            track::yaw_pitch(q * Vec3::NEG_X)[0],
            cols.len()
        );
        seen.push((got, want, cols.len()));
        Ok(())
    })
    .unwrap();
    assert!(seen.iter().all(|&(g, w, n)| n > 3 && (g - w).abs() < 3.0));
    assert!(
        (seen[1].1 - seen[0].1).abs() > 20.0,
        "the camera did not turn"
    );
}
