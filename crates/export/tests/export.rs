//! Export jobs on the X5's footage (`X5_SAMPLES`; skipped when unset), checked with the `ffprobe` / `ffmpeg` CLIs and
//! `exiv2` / `djpeg` (test tools only), and on the synthetic recording (`export::synth`) without them. Outputs go to
//! `X5_EXPORT_OUT` (default: Cargo's test tmp directory, on disk). GPU timings are noisy: other work shares the GPU.

use export::{Cut, Dest, Edit, Enc, Fx, Hook, Job, Out, Pre, Progress, Queue, Seg, State};
use media::part;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};
use stitch::{Ctx, Mode, View};
use video::{Codec, Gpu, Rate};

/// One device per test binary; the lock runs the GPU tests one at a time. The jobs take a host's locks, as the app's
/// do (here nobody else takes them).
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

fn sample(name: &str) -> Option<PathBuf> {
    let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        eprintln!("X5_SAMPLES unset: skipped");
    }
    d.map(|d| d.join(name))
}

fn out(name: &str) -> PathBuf {
    let d = std::env::var_os("X5_EXPORT_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    // a job never replaces a file, nor writes beside a `.part`: an earlier run's go
    for p in [&p, &part::of(&p)] {
        let _ = std::fs::remove_file(p);
    }
    p
}

/// A tool's stdout; panics with its stderr when it fails.
fn tool(cmd: &str, args: &[&str]) -> Vec<u8> {
    let o = Command::new(cmd)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("{cmd}: {e}"));
    assert!(
        o.status.success(),
        "{cmd} {args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    o.stdout
}

fn probe(p: &Path, args: &[&str]) -> String {
    let mut a = vec!["-v", "error"];
    a.extend_from_slice(args);
    a.push(p.to_str().unwrap());
    String::from_utf8(tool("ffprobe", &a)).unwrap()
}

/// `key=value` of the first line that has it.
fn get<'a>(s: &'a str, k: &str) -> &'a str {
    s.lines()
        .find_map(|l| l.strip_prefix(k)?.strip_prefix('='))
        .unwrap_or_else(|| panic!("{k} missing in\n{s}"))
}

/// Per-packet MD5s of the first audio stream.
fn audio_hashes(p: &Path) -> Vec<String> {
    probe(
        p,
        &[
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_data_hash",
            "md5",
            "-show_entries",
            "packet=data_hash",
        ],
    )
    .lines()
    .filter_map(|l| l.strip_prefix("data_hash=").map(String::from))
    .collect()
}

/// Frame `k` of `p`'s video as RGB (`rgb24` or `rgb48le`) from FFmpeg, from limited-range BT.709 (or what the file
/// says: JPEG).
fn ff_frame(p: &Path, k: usize, pix: &str) -> Vec<u8> {
    tool(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-i",
            p.to_str().unwrap(),
            "-vf",
            &format!("select=eq(n\\,{k}),scale=flags=accurate_rnd+full_chroma_int"),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix,
            "-",
        ],
    )
}

fn f16(b: &[u8]) -> f64 {
    let v = u16::from_le_bytes([b[0], b[1]]);
    let (e, m) = ((v >> 10 & 31) as i32, (v & 1023) as f64);
    let x = if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    };
    if v >> 15 == 1 { -x } else { x }
}

/// PSNR over RGB in 0..1 of an RGBA picture (`Rgba16Float` when `wide`, else `Rgba8Unorm`) against RGB from FFmpeg
/// (`rgb48le` when `deep`, else `rgb24`).
fn psnr(rgba: &[u8], wide: bool, rgb: &[u8], deep: bool) -> f64 {
    let a = |i: usize| {
        if wide {
            f16(&rgba[8 * (i / 3) + 2 * (i % 3)..]).clamp(0.0, 1.0)
        } else {
            rgba[4 * (i / 3) + i % 3] as f64 / 255.0
        }
    };
    let b = |i: usize| {
        if deep {
            u16::from_le_bytes([rgb[2 * i], rgb[2 * i + 1]]) as f64 / 65535.0
        } else {
            rgb[i] as f64 / 255.0
        }
    };
    let n = rgb.len() / (1 + deep as usize);
    let se: f64 = (0..n).map(|i| (a(i) - b(i)).powi(2)).sum();
    10.0 * (n as f64 / se.max(1e-12)).log10()
}

/// Runs `job` through a queue and waits; returns the final progress and the seconds it took.
fn export(g: &Arc<Gpu>, cx: &Ctx, job: Job) -> (Progress, f64) {
    let q = Queue::new(g.clone(), cx.clone());
    let t = Instant::now();
    let id = q.push(job);
    loop {
        match q.state(id).unwrap() {
            State::Done(p) => return (p, t.elapsed().as_secs_f64()),
            State::Failed(e) => panic!("export failed: {e}"),
            State::Cancelled => panic!("cancelled"),
            _ => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

/// An fx hook that copies the frames `ks` into textures of its own (read them back after the job).
#[allow(clippy::type_complexity)]
fn capture(g: &Arc<Gpu>, ks: &[u64]) -> (Hook, Arc<Mutex<Vec<(u64, wgpu::Texture)>>>) {
    let got = Arc::new(Mutex::new(vec![]));
    let (g, ks, o) = (g.clone(), ks.to_vec(), got.clone());
    let f = move |fx: Fx<'_>| {
        if !ks.contains(&fx.k) {
            return;
        }
        let t = g.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: fx.tex.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fx.tex.format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        fx.enc
            .copy_texture_to_texture(fx.tex.as_image_copy(), t.as_image_copy(), fx.tex.size());
        o.lock().unwrap().push((fx.k, t));
    };
    (Box::new(f), got)
}

/// v635 whole as 5760 × 2880 H.265 Main 10, RAFT on every frame, colour matching and horizon lock, CBR 100 Mbit/s:
/// stream, spherical metadata, frame count and duration per ffprobe; the audio packets are the source's; three frames
/// decoded by FFmpeg against the stitch's own RGBA16F output for them (taken by an fx hook); throughput with RAFT
/// and, in a second export, with the template seam.
#[test]
fn pano_main10() {
    let Some(src) = sample("v635.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let path = out("v635_5760_main10.mp4");
    let ks = [0, 87, 174];
    let job = |raft: bool, path: PathBuf, fx| Job {
        cut: Cut::Segs(vec![Seg {
            files: vec![src.clone()],
            t: [0.0, f64::INFINITY],
        }]),
        path,
        out: Out::Pano(5760),
        enc: Enc::Video {
            codec: Codec::H265,
            deep: true,
            rate: Rate::Cbr(100_000_000),
            gop: 30,
        },
        raft,
        colour: Some(0.1),
        stab: Some((stab::Lock::Horizon, 0.5)),
        flat: false,
        fx,
        pre: None,
    };
    let (fx, got) = capture(&g, &ks);
    let (p, secs) = export(&g, &cx, job(true, path.clone(), Some(fx)));
    eprintln!(
        "RAFT export: {} frames in {secs:.1} s (with model load and warm-up), steady {:.2} fps",
        p.done, p.fps
    );
    let s = probe(
        &path,
        &[
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_frames,nb_read_frames,duration,color_range,color_space:stream_side_data",
        ],
    );
    eprintln!("{}", s.replace('\n', " "));
    for (k, v) in [
        ("codec_name", "hevc"),
        ("profile", "Main 10"),
        ("pix_fmt", "yuv420p10le"),
        ("width", "5760"),
        ("height", "2880"),
        ("nb_frames", "175"),
        ("nb_read_frames", "175"),
        ("color_range", "tv"),
        ("color_space", "bt709"),
    ] {
        assert_eq!(get(&s, k), v, "{k}");
    }
    for x in [
        "side_data_type=Spherical Mapping\nprojection=equirectangular",
        "side_data_type=Stereo 3D\ntype=2D",
    ] {
        assert!(s.contains(x), "no {x:?}");
    }
    let d: f64 = get(&s, "duration").parse().unwrap();
    assert!((d - 175.0 * 1001.0 / 30000.0).abs() < 1e-3, "duration {d}");
    let (a, b) = (audio_hashes(&src), audio_hashes(&path));
    eprintln!(
        "audio packets: source {}, export {}, identical: {}",
        a.len(),
        b.len(),
        a == b
    );
    assert_eq!(a, b);
    let got = got.lock().unwrap();
    assert_eq!(got.len(), 3);
    for (k, t) in got.iter() {
        let px = stitch::read_texture(&g.device, &g.queue, t);
        let p = psnr(&px, true, &ff_frame(&path, *k as usize, "rgb48le"), true);
        eprintln!("frame {k}: FFmpeg's decode vs the stitch's RGBA16F: PSNR {p:.2} dB");
        assert!(p > 38.0, "frame {k}: {p:.2} dB");
    }
    let tpl = out("v635_5760_main10_template.mp4");
    let (pt, st) = export(&g, &cx, job(false, tpl.clone(), None));
    eprintln!(
        "template export: {} frames in {st:.1} s, steady {:.2} fps (RAFT {:.2} fps)",
        pt.done, pt.fps, p.fps
    );
    assert_eq!(
        get(
            &probe(&tpl, &["-show_entries", "stream=nb_frames"]),
            "nb_frames"
        ),
        "175"
    );
}

/// Planar Y'CbCr 4:2:0 of an RGBA picture as `stitch::Yuv` makes it for the encoder (BT.709, limited range, chroma
/// sited left: [¼ ½ ¼] across the even column, [½ ½] between the rows; columns wrap): 10 or 8 bits.
fn yuv_ref(rgba: &[u8], wide: bool, w: usize, h: usize, deep: bool) -> Vec<u16> {
    let rgb = |x: isize, y: isize| -> [f64; 3] {
        let i = y.clamp(0, h as isize - 1) as usize * w + x.rem_euclid(w as isize) as usize;
        std::array::from_fn(|k| {
            if wide {
                f16(&rgba[8 * i + 2 * k..])
            } else {
                rgba[4 * i + k] as f64 / 255.0
            }
            .clamp(0.0, 1.0)
        })
    };
    let (s, c0) = if deep {
        ([64.0, 876.0, 896.0], 512.0)
    } else {
        ([16.0, 219.0, 224.0], 128.0)
    };
    let l = |p: [f64; 3]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let mut o: Vec<u16> = (0..w * h)
        .map(|i| (s[0] + s[1] * l(rgb((i % w) as isize, (i / w) as isize))).round() as u16)
        .collect();
    let (mut cb, mut cr) = (vec![], vec![]);
    for y in 0..h as isize / 2 {
        for x in 0..w as isize / 2 {
            let mut v = [0.0; 3];
            for (dx, a) in [(-1, 0.25), (0, 0.5), (1, 0.25)] {
                for dy in 0..2 {
                    let p = rgb(2 * x + dx, 2 * y + dy);
                    (0..3).for_each(|k| v[k] += 0.5 * a * p[k]);
                }
            }
            cb.push((c0 + s[2] * (v[2] - l(v)) / 1.8556).round() as u16);
            cr.push((c0 + s[2] * (v[0] - l(v)) / 1.5748).round() as u16);
        }
    }
    o.extend(cb);
    o.extend(cr);
    o
}

/// PSNR of the Y, Cb and Cr planes of two planar 4:2:0 pictures (samples up to `peak`).
fn plane_psnr(a: &[u16], b: &[u16], w: usize, h: usize, peak: f64) -> [f64; 3] {
    let n = w * h;
    let r = [0..n, n..n * 5 / 4, n * 5 / 4..n * 3 / 2];
    r.map(|r| {
        let se: f64 = a[r.clone()]
            .iter()
            .zip(&b[r.clone()])
            .map(|(&p, &q)| (p as f64 - q as f64).powi(2))
            .sum();
        10.0 * (peak * peak * r.len() as f64 / se.max(1e-12)).log10()
    })
}

/// The conversion for the encoder end to end: three frames of v635 at a near-lossless constant QP, H.265 Main 10
/// (P010) and Main (NV12). FFmpeg's decoded planes against the RGBA the stitch made for those frames, converted on the
/// CPU as `stitch::Yuv` does (codec loss only); and FFmpeg's RGB against that RGBA (adds the 4:2:0 round trip, which
/// bounds the RGB PSNR of every export, `pano_main10`'s included).
#[test]
fn fidelity() {
    let Some(src) = sample("v635.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let (w, h) = (3840usize, 1920usize);
    for deep in [true, false] {
        let path = out(&format!(
            "v635_qp_{}.mp4",
            if deep { "main10" } else { "main" }
        ));
        let (fx, got) = capture(&g, &[0, 1, 2]);
        export(
            &g,
            &cx,
            Job {
                cut: Cut::Segs(vec![Seg {
                    files: vec![src.clone()],
                    t: [0.0, 0.1],
                }]),
                path: path.clone(),
                out: Out::Pano(w as u32),
                enc: Enc::Video {
                    codec: Codec::H265,
                    deep,
                    rate: Rate::Qp(4),
                    gop: 30,
                },
                raft: false,
                colour: Some(0.1),
                stab: None,
                flat: false,
                fx: Some(fx),
                pre: None,
            },
        );
        let name = if deep {
            "P010 / Main 10"
        } else {
            "NV12 / Main"
        };
        for (k, t) in got.lock().unwrap().iter() {
            let px = stitch::read_texture(&g.device, &g.queue, t);
            let raw = ff_frame(
                &path,
                *k as usize,
                if deep { "yuv420p10le" } else { "yuv420p" },
            );
            let dec: Vec<u16> = if deep {
                raw.chunks(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            } else {
                raw.iter().map(|&v| v as u16).collect()
            };
            let p = plane_psnr(
                &dec,
                &yuv_ref(&px, deep, w, h, deep),
                w,
                h,
                if deep { 1023.0 } else { 255.0 },
            );
            let rgb = psnr(
                &px,
                deep,
                &ff_frame(&path, *k as usize, if deep { "rgb48le" } else { "rgb24" }),
                deep,
            );
            eprintln!("QP 4, {name}: frame {k}: Y'CbCr PSNR {p:.2?} dB; RGB {rgb:.2} dB");
            assert!(
                p.iter().all(|&v| v > if deep { 52.0 } else { 45.0 }),
                "frame {k}: {p:.2?} dB"
            );
        }
    }
}

/// 10 s of v606 as a flat 1920 × 1080 H.264 with a slow pan (90° over the 10 s), template seam, colour matching,
/// horizon lock: H.264, size, frame count, duration, no spherical metadata, audio as the source's for the span.
#[test]
fn flat_h264() {
    let Some(src) = sample("v606.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let path = out("v606_flat_1080p_pan.mp4");
    let job = Job {
        cut: Cut::Segs(vec![Seg {
            files: vec![src.clone()],
            t: [0.0, 10.0],
        }]),
        path: path.clone(),
        out: Out::Flat {
            w: 1920,
            h: 1080,
            view: Some(Box::new(|t| View {
                mode: Mode::Perspective,
                yaw: -45.0 + 9.0 * t as f32,
                pitch: -5.0,
                fov: 100.0,
                ..Default::default()
            })),
        },
        enc: Enc::Video {
            codec: Codec::H264,
            deep: false,
            rate: Rate::Vbr(16_000_000, 24_000_000),
            gop: 30,
        },
        raft: false,
        colour: Some(0.1),
        stab: Some((stab::Lock::Horizon, 1.0)),
        flat: false,
        fx: None,
        pre: None,
    };
    let (p, secs) = export(&g, &cx, job);
    eprintln!(
        "flat export: {} frames in {secs:.1} s, steady {:.2} fps",
        p.done, p.fps
    );
    let s = probe(
        &path,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,width,height,nb_frames,duration:stream_side_data",
        ],
    );
    eprintln!("{}", s.replace('\n', " "));
    for (k, v) in [
        ("codec_name", "h264"),
        ("width", "1920"),
        ("height", "1080"),
        ("nb_frames", "300"),
    ] {
        assert_eq!(get(&s, k), v, "{k}");
    }
    assert!(
        !s.contains("Spherical"),
        "flat output with spherical metadata"
    );
    let d: f64 = get(&s, "duration").parse().unwrap();
    assert!((d - 300.0 * 1001.0 / 30000.0).abs() < 1e-3, "duration {d}");
    let (a, b) = (audio_hashes(&src), audio_hashes(&path));
    // the AAC frames over [0, 10.01 s): 1024 samples at 48 kHz each
    let n = (300.0 * 1001.0 / 30000.0 * 48000.0 / 1024.0f64).ceil() as usize;
    eprintln!("audio packets: {} (want the source's first {n})", b.len());
    assert_eq!(b, a[..n]);
    let fr = ff_frame(&path, 150, "rgb24");
    let png = out("v606_flat_1080p_pan_150.png");
    let mut c = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1920x1080",
            "-i",
            "-",
        ])
        .arg(&png)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write as _;
    c.stdin.take().unwrap().write_all(&fr).unwrap();
    assert!(c.wait().unwrap().success());
}

/// A trimmed export of v606, in and out mid-GOP (10.37 s → frame 310, IDRs at 300 and 360; out 14.6 s): the first
/// video frame is source frame 310 (PSNR against frame grabs of 309, 310, 311 through the same path) and the audio
/// starts with it (the export's decoded audio against the source's from frame 310's time: best lag in samples).
#[test]
fn trim_sync() {
    let Some(src) = sample("v606.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let path = out("v606_trim.mp4");
    let fd = 1001.0 / 30000.0;
    let job = |path: PathBuf, t: [f64; 2], enc| Job {
        cut: Cut::Segs(vec![Seg {
            files: vec![src.clone()],
            t,
        }]),
        path,
        out: Out::Pano(3840),
        enc,
        raft: false,
        colour: None,
        stab: None,
        flat: false,
        fx: None,
        pre: None,
    };
    let enc = Enc::Video {
        codec: Codec::H265,
        deep: false,
        rate: Rate::Cbr(60_000_000),
        gop: 30,
    };
    let (p, _) = export(&g, &cx, job(path.clone(), [10.37, 14.6], enc));
    assert_eq!(p.done, 438 - 310);
    let s = probe(
        &path,
        &[
            "-show_entries",
            "stream=codec_type,start_time,duration,nb_frames",
        ],
    );
    eprintln!("{}", s.replace('\n', " "));
    let pk = |sel: &str| {
        probe(
            &path,
            &[
                "-select_streams",
                sel,
                "-show_entries",
                "packet=pts_time",
                "-read_intervals",
                "%+#2",
            ],
        )
        .lines()
        .filter_map(|l| {
            l.strip_prefix("pts_time=")
                .and_then(|v| v.parse::<f64>().ok())
        })
        .collect::<Vec<_>>()
    };
    let (vp, ap) = (pk("v:0"), pk("a:0"));
    eprintln!(
        "first video packet pts {:?} s, first audio packet pts {:?} s",
        vp, ap
    );
    assert_eq!(vp[0], 0.0);
    // the source's audio frame holding frame 310's time starts before it: the edit list cuts the difference
    let t310 = 310.0 * fd;
    let a0 = (t310 * 48000.0f64 / 1024.0).floor() * 1024.0 / 48000.0;
    assert!(
        (ap[0] - (a0 - t310)).abs() < 1e-4,
        "audio starts at {} s, want {} s",
        ap[0],
        a0 - t310
    );
    // which source frame is video frame 0
    let ours = ff_frame(&path, 0, "rgb24");
    for k in [309, 310, 311] {
        let gp = out(&format!("v606_grab_{k}.jpg"));
        export(
            &g,
            &cx,
            job(gp.clone(), [k as f64 * fd, f64::INFINITY], Enc::Jpeg(95)),
        );
        let px: Vec<u8> = ff_frame(&gp, 0, "rgb24")
            .chunks(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        eprintln!(
            "export frame 0 vs grab of source frame {k}: PSNR {:.2} dB",
            psnr(&px, false, &ours, false)
        );
    }
    // audio: export from 0.1 s against the source from frame 310's time + 0.1 s, lags ±200 samples
    let pcm = |args: &[&str]| -> Vec<f32> {
        let mut a = vec!["-v", "error"];
        a.extend_from_slice(args);
        a.extend_from_slice(&["-map", "0:a:0", "-ac", "1", "-f", "f32le", "-"]);
        tool("ffmpeg", &a)
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let ex = pcm(&["-i", path.to_str().unwrap()]);
    let sr = pcm(&[
        "-ss",
        &format!("{t310:.6}"),
        "-i",
        src.to_str().unwrap(),
        "-t",
        "3",
    ]);
    let (o, n, m) = (4800usize, 48000usize, 200isize);
    let corr = |lag: isize| {
        let (mut s, mut a, mut b) = (0f64, 0f64, 0f64);
        for i in o..o + n {
            let (x, y) = (ex[i] as f64, sr[(i as isize + lag) as usize] as f64);
            (s, a, b) = (s + x * y, a + x * x, b + y * y);
        }
        s / (a * b).sqrt()
    };
    let (best, c) = (-m..=m)
        .map(|l| (l, corr(l)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    eprintln!(
        "audio: export vs source from frame 310's time: best lag {best} samples (correlation {c:.5}; at 0: {:.5})",
        corr(0)
    );
    assert!(best.abs() <= 1 && c > 0.99);
}

/// Three segments of three clips (30 frames each from v606 frame 30, v635 frame 15, v627 frame 60; in and out points
/// mid-frame) into one 3840 × 1920 H.265:
/// frame count and duration; the first frame of each segment is that clip's frame (PSNR against a frame grab of it,
/// so each clip has its own calibration and colour fit); the audio is each clip's AAC frames over its span, copied
/// unchanged and back to back.
#[test]
fn segments() {
    let Some(d) = sample("") else { return };
    let (_l, g, cx) = gpu();
    let fd = 1001.0 / 30000.0;
    let parts = [("v606.insv", 30usize), ("v635.insv", 15), ("v627.insv", 60)];
    let seg = |&(n, k): &(&str, usize), len: f64| Seg {
        files: vec![d.join(n)],
        t: [(k as f64 + 0.5) * fd, (k as f64 + 0.5 + len) * fd],
    };
    let job = |path: PathBuf, segs, enc| Job {
        cut: Cut::Segs(segs),
        path,
        out: Out::Pano(3840),
        enc,
        raft: false,
        colour: Some(0.1),
        stab: None,
        flat: false,
        fx: None,
        pre: None,
    };
    let path = out("segments.mp4");
    let enc = Enc::Video {
        codec: Codec::H265,
        deep: false,
        rate: Rate::Cbr(40_000_000),
        gop: 30,
    };
    let (p, _) = export(
        &g,
        &cx,
        job(
            path.clone(),
            parts.iter().map(|x| seg(x, 29.0)).collect(),
            enc,
        ),
    );
    assert_eq!(p.done, 90);
    let s = probe(
        &path,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration",
        ],
    );
    assert_eq!(get(&s, "nb_frames"), "90");
    let dv: f64 = get(&s, "duration").parse().unwrap();
    assert!((dv - 90.0 * fd).abs() < 1e-3, "duration {dv}");
    for (i, x) in parts.iter().enumerate() {
        let gp = out(&format!("segments_grab_{i}.jpg"));
        export(
            &g,
            &cx,
            job(gp.clone(), vec![seg(x, f64::INFINITY)], Enc::Jpeg(95)),
        );
        let px: Vec<u8> = ff_frame(&gp, 0, "rgb24")
            .chunks(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect();
        let k = 30 * i;
        let ps = psnr(&px, false, &ff_frame(&path, k, "rgb24"), false);
        let other = psnr(&px, false, &ff_frame(&path, (k + 45) % 90, "rgb24"), false);
        eprintln!(
            "export frame {k} vs a grab of {} frame {}: PSNR {ps:.2} dB (frame {}: {other:.2} dB)",
            x.0,
            x.1,
            (k + 45) % 90
        );
        assert!(ps > 35.0 && ps > other + 10.0);
    }
    // the export's AAC frames: one contiguous run of each source's, in order
    let ours = audio_hashes(&path);
    let mut at = 0;
    for (n, _) in parts {
        let src = audio_hashes(&d.join(n));
        let i = src
            .iter()
            .position(|h| *h == ours[at])
            .expect("audio frame not in the source");
        let run = src[i..]
            .iter()
            .zip(&ours[at..])
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "{n}: {run} AAC frames from source frame {i} ({:.4} s)",
            i as f64 * 1024.0 / 48000.0
        );
        at += run;
    }
    assert_eq!(at, ours.len());
    let da = ours.len() as f64 * 1024.0 / 48000.0;
    eprintln!("audio {da:.4} s, video {dv:.4} s");
    // each span starts and ends within an AAC frame of its video
    assert!(da >= dv && da < dv + 3.0 * 1024.0 / 48000.0);
}

/// One queue, jobs one after another (v635 1-1.5 s): RAFT, then the template seam and a 360 grab without it (the
/// model the queue keeps must not reach them); a flat view whose width is no multiple of 4; a job whose directory
/// is missing fails with the I/O error, not "cancelled"; an fx hook that panics fails its job without leaving output
/// or hanging the threads; the queue runs on afterwards.
#[test]
fn queue() {
    let Some(src) = sample("v635.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let q = Queue::new(g.clone(), cx.clone());
    let v = |codec, deep| Enc::Video {
        codec,
        deep,
        rate: Rate::Cbr(20_000_000),
        gop: 30,
    };
    let (h4, h5, h10) = (
        v(Codec::H264, false),
        v(Codec::H265, false),
        v(Codec::H265, true),
    );
    let flat = || Out::Flat {
        w: 854,
        h: 480,
        view: Some(Box::new(|t| View {
            yaw: 40.0 * t as f32,
            fov: 90.0,
            ..Default::default()
        })),
    };
    let pano = || Out::Pano(3840);
    let boom: Hook = Box::new(|fx: Fx<'_>| {
        if fx.k == 5 {
            panic!("fx hook at frame 5")
        }
    });
    let (enoent, bang) = (Some("No such file"), Some("panic: fx hook at frame 5"));
    // frame 7 of the flat views, to check the padded YUV rows end to end
    let ((c8, got8), (c10, got10)) = (capture(&g, &[7]), capture(&g, &[7]));
    let jobs = vec![
        ("q_raft.mp4", pano(), h10, true, None, None),
        ("q_tpl.mp4", pano(), h5, false, None, None),
        ("q_grab.jpg", pano(), Enc::Jpeg(90), false, None, None),
        ("q_flat854.mp4", flat(), h4, false, Some(c8), None),
        ("q_flat854_10.mp4", flat(), h10, true, Some(c10), None),
        ("missing/q.mp4", pano(), h5, false, None, enoent),
        ("q_panic.mp4", pano(), h5, true, Some(boom), bang),
        ("q_after.mp4", pano(), h10, true, None, None),
    ];
    let _ = std::fs::remove_dir(out("missing"));
    let mut ids = vec![];
    for (name, o, enc, raft, fx, want) in jobs {
        let path = out(name);
        let _ = std::fs::remove_file(&path);
        let id = q.push(Job {
            cut: Cut::Segs(vec![Seg {
                files: vec![src.clone()],
                t: [1.0, 1.5],
            }]),
            path: path.clone(),
            out: o,
            enc,
            raft,
            colour: Some(0.1),
            stab: None,
            flat: false,
            fx,
            pre: None,
        });
        ids.push((id, name, path, want));
    }
    let t = Instant::now();
    while (q.jobs().iter()).any(|(_, s)| matches!(s, State::Queued | State::Running(_))) {
        assert!(t.elapsed().as_secs() < 300, "{:?}", q.jobs());
        std::thread::sleep(Duration::from_millis(20));
    }
    for (id, name, path, want) in ids {
        let s = q.state(id).unwrap();
        let mut part = path.clone().into_os_string();
        part.push(".part");
        eprintln!("{name}: {s:?}");
        assert!(!Path::new(&part).exists(), "{name}: partial output left");
        match (s, want) {
            (State::Done(p), None) => {
                let s = probe(
                    &path,
                    &[
                        "-select_streams",
                        "v:0",
                        "-count_frames",
                        "-show_entries",
                        "stream=codec_name,profile,width,height,nb_read_frames",
                    ],
                );
                eprintln!("  {}", s.replace('\n', " "));
                let n = if name.ends_with(".jpg") { 1 } else { p.total };
                assert_eq!(get(&s, "nb_read_frames"), n.to_string(), "{name}");
                if name.starts_with("q_flat854") {
                    assert_eq!((get(&s, "width"), get(&s, "height")), ("854", "480"));
                    let deep = name.ends_with("_10.mp4");
                    let got = if deep { &got10 } else { &got8 }.lock().unwrap();
                    let px = stitch::read_texture(&g.device, &g.queue, &got[0].1);
                    let pix = if deep { "rgb48le" } else { "rgb24" };
                    let p = psnr(&px, deep, &ff_frame(&path, 7, pix), deep);
                    eprintln!("  frame 7: FFmpeg's decode vs the rendered frame: PSNR {p:.2} dB");
                    assert!(p > 36.0, "{name}: {p:.2} dB");
                }
            }
            (State::Failed(e), Some(w)) => {
                assert!(e.contains(w), "{name}: {e:?} lacks {w:?}");
                assert!(!path.exists(), "{name}: output left");
            }
            (s, w) => panic!("{name}: {s:?}, expected {w:?}"),
        }
    }
}

/// Cancel mid-job through the queue: the job stops, ends as Cancelled, and leaves neither the file nor the partial
/// one.
#[test]
fn cancel() {
    let Some(src) = sample("v606.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let path = out("v606_cancelled.mp4");
    let part = out("v606_cancelled.mp4.part");
    let q = Queue::new(g.clone(), cx.clone());
    let id = q.push(Job {
        cut: Cut::Segs(vec![Seg {
            files: vec![src],
            t: [0.0, f64::INFINITY],
        }]),
        path: path.clone(),
        out: Out::Pano(5760),
        enc: Enc::Video {
            codec: Codec::H265,
            deep: true,
            rate: Rate::Cbr(100_000_000),
            gop: 30,
        },
        raft: true,
        colour: Some(0.1),
        stab: None,
        flat: false,
        fx: None,
        pre: None,
    });
    let t = Instant::now();
    loop {
        match q.state(id).unwrap() {
            State::Running(p) if p.done >= 30 => {
                eprintln!(
                    "cancelling at {}/{} frames, {:.2} fps, ETA {:.0} s",
                    p.done, p.total, p.fps, p.eta
                );
                assert!(part.exists() && !path.exists());
                break;
            }
            State::Queued | State::Running(_) => std::thread::sleep(Duration::from_millis(10)),
            s => panic!("{s:?}"),
        }
        assert!(t.elapsed().as_secs() < 300);
    }
    q.cancel(id);
    let t = Instant::now();
    while q.state(id) != Some(State::Cancelled) {
        assert!(t.elapsed().as_secs() < 30, "{:?}", q.state(id));
        std::thread::sleep(Duration::from_millis(5));
    }
    eprintln!(
        "cancelled after {:.0} ms; output {}, partial {}",
        t.elapsed().as_secs_f64() * 1e3,
        path.exists(),
        part.exists()
    );
    assert!(!path.exists() && !part.exists());
}

/// Frame grabs of v635 frame 100: the equirect at 11520 × 5760 (RAFT, colour matching) with GPano XMP, and a flat
/// 1920 × 1080 view. FFmpeg and djpeg decode them without complaint, exiv2 reads the GPano tags, PSNR of FFmpeg's
/// decode against the RGBA that was encoded; the encoder's own time for the big one.
#[test]
fn grab() {
    let Some(src) = sample("v635.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let t = 100.0 * 1001.0 / 30000.0;
    for (name, o, w, h) in [
        ("v635_grab_11520.jpg", Out::Pano(11520), 11520u32, 5760u32),
        (
            "v635_grab_flat.jpg",
            Out::Flat {
                w: 1920,
                h: 1080,
                view: Some(Box::new(|_| View {
                    yaw: 30.0,
                    pitch: -10.0,
                    fov: 100.0,
                    ..Default::default()
                })),
            },
            1920,
            1080,
        ),
    ] {
        let path = out(name);
        let pano = matches!(o, Out::Pano(_));
        let (fx, got) = capture(&g, &[0]);
        let (_, secs) = export(
            &g,
            &cx,
            Job {
                cut: Cut::Segs(vec![Seg {
                    files: vec![src.clone()],
                    t: [t, t],
                }]),
                path: path.clone(),
                out: o,
                enc: Enc::Jpeg(92),
                raft: pano,
                colour: Some(0.1),
                stab: None,
                flat: false,
                fx: Some(fx),
                pre: None,
            },
        );
        let px = stitch::read_texture(&g.device, &g.queue, &got.lock().unwrap()[0].1);
        let t0 = Instant::now();
        let again = export::jpeg::encode(&px, w, h, 92, None).unwrap();
        let enc = t0.elapsed().as_secs_f64();
        let size = std::fs::metadata(&path).unwrap().len();
        let s = probe(
            &path,
            &["-show_entries", "stream=codec_name,width,height,pix_fmt"],
        );
        assert_eq!(
            (get(&s, "width"), get(&s, "height")),
            (w.to_string().as_str(), h.to_string().as_str())
        );
        let p = psnr(&px, false, &ff_frame(&path, 0, "rgb24"), false);
        let dj = tool("djpeg", &["-ppm", path.to_str().unwrap()]);
        eprintln!(
            "{name}: {w}x{h}, {:.1} MB, job {secs:.1} s, JPEG encode {:.0} ms ({} bytes without XMP); {}; PSNR vs \
             the frame {p:.2} dB; djpeg {} bytes",
            size as f64 / 1e6,
            enc * 1e3,
            again.len(),
            get(&s, "pix_fmt"),
            dj.len()
        );
        assert!(p > 36.0);
        assert_eq!(
            dj.len(),
            format!("P6\n{w} {h}\n255\n").len() + (w * h * 3) as usize
        );
        if pano {
            let x = String::from_utf8(tool("exiv2", &["-px", path.to_str().unwrap()])).unwrap();
            eprintln!("exiv2:\n{}", x.trim());
            assert!(x.contains("Xmp.GPano.ProjectionType") && x.contains("equirectangular"));
            assert!(x.contains(&format!(
                "Xmp.GPano.FullPanoWidthPixels                XmpText     5  {w}"
            )));
        }
    }
}

/// A narrow flat view (1920 × 1080, 40°, lens 1's axis, far from the seam) of v635 frame 100 with the template seam
/// and with RAFT: sharpness of each (mean squared luma gradient) and PSNR between them. RAFT views are drawn from the
/// lenses through the flow (`stitch::Preview::seam`), so away from the seam they equal the template's; an
/// intermediate 7680 panorama kept 82 % of the gradient energy here.
#[test]
fn flat_sharpness() {
    let Some(src) = sample("v635.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let t = 100.0 * 1001.0 / 30000.0;
    let [a, b] = [false, true].map(|raft| {
        let (fx, got) = capture(&g, &[0]);
        export(
            &g,
            &cx,
            Job {
                cut: Cut::Segs(vec![Seg {
                    files: vec![src.clone()],
                    t: [t, t],
                }]),
                path: out(&format!("v635_flat40_raft{}.jpg", raft as u8)),
                out: Out::Flat {
                    w: 1920,
                    h: 1080,
                    view: Some(Box::new(|_| View {
                        fov: 40.0,
                        ..Default::default()
                    })),
                },
                enc: Enc::Jpeg(92),
                raft,
                colour: Some(0.1),
                stab: None,
                flat: false,
                fx: Some(fx),
                pre: None,
            },
        );
        stitch::read_texture(&g.device, &g.queue, &got.lock().unwrap()[0].1)
    });
    let (w, h) = (1920, 1080);
    let sharp = |px: &[u8]| {
        let l = |x: usize, y: usize| {
            let p = &px[4 * (y * w + x)..];
            0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
        };
        let mut e = 0.0;
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                e += (l(x + 1, y) - l(x, y)).powi(2) + (l(x, y + 1) - l(x, y)).powi(2);
            }
        }
        e / ((w - 1) * (h - 1)) as f64
    };
    let rgb = |px: &[u8]| -> Vec<u8> { px.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect() };
    let (sa, sb, p) = (sharp(&a), sharp(&b), psnr(&a, false, &rgb(&b), false));
    eprintln!(
        "40° flat view: gradient energy with the template seam {sa:.2}, with RAFT {sb:.2} ({:.1} %); PSNR between \
         them {p:.2} dB",
        100.0 * sb / sa
    );
    assert!(sb > 0.99 * sa && p > 50.0);
}

/// The JPEG encoder alone on a synthetic picture with a size that is no multiple of the MCU (edges repeated) and
/// every quality range: FFmpeg's decode against it, and djpeg's; libjpeg's `cjpeg` at the same quality (the same
/// tables, 4:2:0, baseline) for size and PSNR.
#[test]
fn jpeg_synthetic() {
    let (w, h) = (1003usize, 701usize);
    let mut s = 1u32;
    let px: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            let n = (s >> 29) as f32;
            [
                255.0 * x,
                255.0 * y,
                128.0 + 100.0 * (12.0 * x).sin() * (9.0 * y).cos() + n,
                255.0,
            ]
            .map(|v| v.clamp(0.0, 255.0) as u8)
        })
        .collect();
    for q in [30, 75, 92, 100] {
        let b = export::jpeg::encode(
            &px,
            w as u32,
            h as u32,
            q,
            Some(&export::jpeg::gpano(w as u32, h as u32)),
        )
        .unwrap();
        let p = out(&format!("synthetic_q{q}.jpg"));
        std::fs::write(&p, &b).unwrap();
        let ps = psnr(&px, false, &ff_frame(&p, 0, "rgb24"), false);
        let dj = tool("djpeg", &["-ppm", p.to_str().unwrap()]);
        let pd = psnr(&px, false, &dj[dj.len() - w * h * 3..], false);
        let ppm = out("synthetic.ppm");
        let mut f = format!("P6\n{w} {h}\n255\n").into_bytes();
        f.extend(px.chunks(4).flat_map(|p| [p[0], p[1], p[2]]));
        std::fs::write(&ppm, f).unwrap();
        let c = tool(
            "cjpeg",
            &[
                "-quality",
                &q.to_string(),
                "-sample",
                "2x2",
                "-baseline",
                ppm.to_str().unwrap(),
            ],
        );
        let cp = out(&format!("synthetic_q{q}_cjpeg.jpg"));
        std::fs::write(&cp, &c).unwrap();
        let pc = psnr(&px, false, &ff_frame(&cp, 0, "rgb24"), false);
        eprintln!(
            "quality {q}: {} bytes, PSNR FFmpeg {ps:.2} dB, djpeg {pd:.2} dB; cjpeg {} bytes, {pc:.2} dB",
            b.len(),
            c.len()
        );
        assert!(ps > 30.0 && (ps - pd).abs() < 1.0);
        // at 100 (all-1 tables) the size follows noise below 8-bit precision (cjpeg: 258 kB with its integer DCT,
        // 235 kB with its float DCT); PSNR is compared there too
        assert!(
            (ps - pc).abs() < 0.5
                && (q == 100 || (b.len() as f64 / c.len() as f64 - 1.0).abs() < 0.1)
        );
    }
}

/// Mono PCM (f32) FFmpeg decodes from `p`'s first audio stream at 48 kHz; `pre`: input options (a seek).
fn pcm(p: &Path, pre: &[&str]) -> Vec<f32> {
    let mut a = vec!["-v", "error"];
    a.extend_from_slice(pre);
    a.extend_from_slice(&["-i", p.to_str().unwrap()]);
    a.extend_from_slice(&[
        "-map", "0:a:0", "-ac", "1", "-ar", "48000", "-f", "f32le", "-",
    ]);
    tool("ffmpeg", &a)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Normalised cross-correlation of `a[oa..oa + n]` against `b` from `ob + lag`, lags within ±m: the best lag (b
/// later: positive) and its correlation, and the correlation at lag 0.
fn xcorr(a: &[f32], oa: usize, b: &[f32], ob: usize, n: usize, m: isize) -> (isize, f64, f64) {
    let corr = |lag: isize| {
        let (mut s, mut x2, mut y2) = (0f64, 0f64, 0f64);
        for i in 0..n {
            let (x, y) = (
                a[oa + i] as f64,
                b[(ob as isize + i as isize + lag) as usize] as f64,
            );
            (s, x2, y2) = (s + x * y, x2 + x * x, y2 + y * y);
        }
        s / (x2 * y2).sqrt().max(1e-30)
    };
    let (l, c) = (-m..=m)
        .map(|l| (l, corr(l)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    (l, c, corr(0))
}

/// Presentation times (s) of the frames of the first video track, as crate edit counts source time.
fn frame_times(p: &Path) -> Vec<f64> {
    let m = media::mp4::Mp4::open(p).unwrap();
    let t = m.tracks.iter().find(|t| &t.handler == b"vide").unwrap();
    let sh = t.shift(m.timescale);
    let mut v: Vec<f64> = (t.samples.iter())
        .map(|s| (s.pts + sh) as f64 / t.timescale as f64)
        .collect();
    v.sort_by(f64::total_cmp);
    v
}

fn edit_job(tl: edit::Timeline, path: PathBuf, out: Out, enc: Enc, fx: Option<Hook>) -> Job {
    Job {
        cut: Cut::Edit(Box::new(Edit {
            tl,
            files: vec![],
            looks: HashMap::from([("film".to_string(), fx::Look::Film.cube())]),
            over: vec![],
            abr: 192_000,
            denoise: None,
        })),
        path,
        out,
        enc,
        raft: false,
        colour: None,
        stab: None,
        flat: false,
        fx,
        pre: None,
    }
}

/// An editor timeline into a flat 1920 × 1080 H.264 + AAC: v606 2–6 s at 1× ramping to 2× (speed keys at 3 s and
/// 4.5 s source), a 1 s freeze frame of v635 at 2 s, v627 1–4 s at 1× with the Film look at 0.8; a keyframed pan over
/// the output; music (a 44.1 kHz chord) at gain 0.5, ducked under the clips. Checks with FFmpeg: frame count and
/// duration exact, audio as long as the video within a sample (FFmpeg's decode, which applies the edit lists, too),
/// decodes without a message; A/V sync at the moving clips' starts (the export's audio against the source's at the
/// mapped time, best lag); during the freeze frame only the music (against the music file); the frames at known
/// output times are the timeline's source frames (PSNR against grabs of that frame and its neighbours through the
/// same view and look), the look and the view applied (against grabs without the look and with another frame's
/// view). Then a plain cut of the same clips at 1× copies the source's AAC frames, cut at the video's end.
#[test]
fn timeline() {
    let Some(d) = sample("") else { return };
    let (_l, g, cx) = gpu();
    let music = out("chord44k.wav");
    tool(
        "ffmpeg",
        &[
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.2*sin(2*PI*262*t)+0.15*sin(2*PI*330*t)|0.2*sin(2*PI*392*t)+0.1*sin(2*PI*523*t):s=44100:d=12",
            music.to_str().unwrap(),
        ],
    );
    let names = ["v606.insv", "v635.insv", "v627.insv"];
    let times: Vec<Vec<f64>> = names.iter().map(|n| frame_times(&d.join(n))).collect();
    let fps = edit::Rate {
        num: 30000,
        den: 1001,
    };
    let mut tl = edit::Timeline::new(fps);
    tl.files = (names.iter().zip(&times))
        .map(|(n, t)| edit::File {
            path: d.join(n).to_str().unwrap().into(),
            dur: t[t.len() - 1] + 1001.0 / 30000.0,
        })
        .chain([edit::File {
            path: music.to_str().unwrap().into(),
            dur: 12.0,
        }])
        .collect();
    let mut c0 = edit::Clip::new(0, 2.0, 6.0);
    c0.speed = vec![(3.0, 1.0), (4.5, 2.0)];
    let mut c2 = edit::Clip::new(2, 1.0, 4.0);
    c2.look = "film".into();
    c2.amt = 0.8;
    tl.clips = vec![c0, edit::Clip::still(1, 2.0, 1.0), c2];
    let key = |t, yaw, pitch, fov| edit::Key {
        t,
        v: edit::View::look(yaw, pitch, 0.0, fov, edit::Proj::Persp),
        e: edit::Ease::Smooth,
    };
    tl.out.view = vec![key(0.0, -30.0, -5.0, 100.0), key(6.0, 60.0, 5.0, 85.0)];
    tl.out.music = Some(edit::Music {
        file: 3,
        offset: 0.0,
        gain: 0.5,
        duck: true,
    });
    tl.fix();
    tl.check().unwrap();
    let (len, dur) = (tl.len(), tl.dur());
    let fr: Vec<_> = (0..3).map(|k| tl.frames(k)).collect();
    eprintln!("timeline: {len} frames ({dur:.6} s), clips at frames {fr:?}");
    let path = out("timeline_1080p.mp4");
    let flat = || Out::Flat {
        w: 1920,
        h: 1080,
        view: None,
    };
    let h264 = Enc::Video {
        codec: Codec::H264,
        deep: false,
        rate: Rate::Vbr(16_000_000, 24_000_000),
        gop: 30,
    };
    let (p, secs) = export(
        &g,
        &cx,
        edit_job(tl.clone(), path.clone(), flat(), h264, None),
    );
    eprintln!(
        "timeline export: {} frames in {secs:.1} s, steady {:.2} fps",
        p.done, p.fps
    );
    assert_eq!(p.done, len);
    let s = probe(
        &path,
        &[
            "-count_frames",
            "-show_entries",
            "stream=codec_name,codec_type,width,height,nb_read_frames,start_time,duration,duration_ts,sample_rate,channels",
        ],
    );
    eprintln!("{}", s.replace('\n', " "));
    let st: Vec<&str> = s.split("[/STREAM]").collect();
    for (k, v) in [
        ("codec_name", "h264"),
        ("width", "1920"),
        ("height", "1080"),
        ("nb_read_frames", &len.to_string()),
        ("duration_ts", &(len * 1001).to_string()),
    ] {
        assert_eq!(get(st[0], k), v, "{k}");
    }
    for (k, v) in [
        ("codec_name", "aac"),
        ("sample_rate", "48000"),
        ("channels", "2"),
    ] {
        assert_eq!(get(st[1], k), v, "{k}");
    }
    let f = |s: &str, k| get(s, k).parse::<f64>().unwrap();
    let (ve, ae) = (
        f(st[0], "start_time") + f(st[0], "duration"),
        f(st[1], "start_time") + f(st[1], "duration"),
    );
    let o = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "null",
            "-",
        ])
        .output()
        .unwrap();
    let ours = pcm(&path, &[]);
    eprintln!(
        "video ends {ve:.6} s ({dur:.6} s), audio {ae:.6} s: {:+.1} samples; FFmpeg decodes {} audio samples ({:+} \
         against the video); decode messages: {:?}",
        (ae - ve) * 48000.0,
        ours.len(),
        ours.len() as i64 - (dur * 48000.0).round() as i64,
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(o.status.success() && o.stderr.is_empty());
    assert!((ve - dur).abs() < 1e-6 && (ae - ve).abs() <= 1.0 / 48000.0);
    assert!((ours.len() as f64 - dur * 48000.0).abs() <= 1.0);

    // A/V sync where the moving clips start (both at 1× there): 0.5 s from 0.1 s into the clip
    let at = |n: u64| (fps.t(n) * 48000.0).round() as usize;
    for (k, file, src) in [(0, 0, 2.0), (2, 2, 1.0)] {
        let sp = pcm(
            &d.join(names[file]),
            &["-ss", &format!("{src:.6}"), "-t", "1"],
        );
        let (lag, c, c0) = xcorr(&ours, at(fr[k].start) + 4800, &sp, 4800, 24000, 200);
        eprintln!(
            "clip {k} ({} from {src} s) at output {:.4} s: best lag {lag} samples, correlation {c:.4} (lag 0: {c0:.4})",
            names[file],
            fps.t(fr[k].start)
        );
        assert!(lag.abs() <= 1 && c > 0.9, "clip {k}");
    }
    // the freeze frame: no clip audio, the music alone (from 0.6 s in: the duck has released)
    let mu = pcm(&music, &[]);
    let (a0, a1) = (at(fr[1].start) + 28800, at(fr[1].end) - 2400);
    let (lag, c, _) = xcorr(&ours, a0, &mu, a0, a1 - a0, 200);
    let rms =
        |x: &[f32]| (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    eprintln!(
        "freeze frame: against the music: best lag {lag}, correlation {c:.4}; level {:.4} (music alone {:.4} × 0.5)",
        rms(&ours[a0..a1]),
        rms(&mu[a0..a1])
    );
    assert!(lag.abs() <= 1 && c > 0.99);
    // the 2× run of clip 0 has sound (WSOLA), not silence
    let r2 = rms(&ours[at(fr[0].end) - 24000..at(fr[0].end) - 2400]);
    eprintln!("clip 0 at 2× (its last 0.5 s): level {r2:.4}");
    assert!(r2 > 0.01);
    // and stays with the pictures: a WSOLA grain plays the source at 1× around where it was taken, so 256-sample
    // windows centred on the grains (output 512·(m + 1) from the clip's start) find their lag against the source at
    // the timeline's time for that output instant
    let sp = pcm(&d.join(names[0]), &["-ss", "2.0", "-t", "5"]);
    let mut lg: Vec<(isize, f64)> = (215..250)
        .map(|m| {
            let c = 512 * (m + 1);
            let q = ((tl.content(0, c as f64 / 48000.0) - 2.0) * 48000.0).round() as usize;
            let (l, x, _) = xcorr(&ours, c - 128, &sp, q - 128, 256, 1500);
            (l, x)
        })
        .collect();
    lg.sort_by_key(|x| x.0);
    let med = lg[lg.len() / 2];
    eprintln!(
        "inside the 2× run ({:.2}–{:.2} s): grain lags {} … {} samples, median {} ({:.1} ms; correlation {:.3})",
        512.0 * 216.0 / 48000.0,
        512.0 * 251.0 / 48000.0,
        lg[0].0,
        lg[lg.len() - 1].0,
        med.0,
        med.0 as f64 / 48.0,
        med.1
    );
    assert!(med.0.abs() <= 1024 && med.1 > 0.8);

    // frames: the source frame each one shows, through its view and look
    let grab = |file: u32, src: f64, v: edit::View, look: &str| {
        let mut t = edit::Timeline::new(fps);
        t.files = tl.files.clone();
        let mut c = edit::Clip::still(file, src, 0.1);
        (c.look, c.amt) = (look.into(), 0.8);
        t.clips = vec![c];
        t.out.view = vec![edit::Key {
            t: 0.0,
            v,
            e: edit::Ease::Hold,
        }];
        t.fix();
        let (fx, got) = capture(&g, &[0]);
        export(
            &g,
            &cx,
            edit_job(t, out("timeline_grab.jpg"), flat(), Enc::Jpeg(90), Some(fx)),
        );
        stitch::read_texture(&g.device, &g.queue, &got.lock().unwrap()[0].1)
    };
    for n in [15, 45, 75, 100, 160] {
        let e = tl.frame(n).unwrap();
        let file = e.clip.file as usize;
        let k = times[file].partition_point(|&x| x <= e.src + 1e-6) - 1;
        let dec = ff_frame(&path, n as usize, "rgb24");
        let ps: Vec<f64> = [k - 1, k, k + 1]
            .iter()
            .map(|&j| {
                psnr(
                    &grab(file as u32, times[file][j], e.view, &e.clip.look),
                    false,
                    &dec,
                    false,
                )
            })
            .collect();
        eprintln!(
            "frame {n} (clip {}, source {:.4} s, speed {:.3}): {} frame {k}: PSNR {:.2} dB (frames {} / {}: {:.2} / {:.2} \
             dB)",
            e.k,
            e.src,
            e.speed,
            names[file],
            ps[1],
            k - 1,
            k + 1,
            ps[0],
            ps[2]
        );
        // H.264 at 16 Mbit/s loses most where the picture moves fastest (2×)
        assert!(ps[1] > 30.0 && ps[1] > ps[0].max(ps[2]) + 6.0, "frame {n}");
        if n == 160 {
            let (a, b) = (
                grab(file as u32, times[file][k], e.view, ""),
                grab(file as u32, times[file][k], e.view, "film"),
            );
            let plain = psnr(&a, false, &dec, false);
            let rgb: Vec<u8> = b.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
            let v = tl.frame(n - 60).unwrap().view;
            let moved = psnr(
                &grab(file as u32, times[file][k], v, "film"),
                false,
                &dec,
                false,
            );
            eprintln!(
                "  without the look: {plain:.2} dB (the look itself: {:.2} dB); with frame {}'s view: {moved:.2} dB",
                psnr(&a, false, &rgb, false),
                n - 60
            );
            assert!(ps[1] > plain + 2.0 && ps[1] > moved + 10.0);
        }
    }

    // a plain cut: the AAC frames are copied, the audio ends with the video
    let mut tp = tl.clone();
    tp.clips = vec![edit::Clip::new(0, 2.0, 3.5), edit::Clip::new(2, 1.0, 2.0)];
    tp.out.music = None;
    tp.fix();
    let pp = out("timeline_plain.mp4");
    export(
        &g,
        &cx,
        edit_job(
            tp.clone(),
            pp.clone(),
            Out::Flat {
                w: 1280,
                h: 720,
                view: None,
            },
            h264,
            None,
        ),
    );
    let s = probe(&pp, &["-show_entries", "stream=start_time,duration"]);
    let st: Vec<&str> = s.split("[/STREAM]").collect();
    let (ve, ae) = (
        f(st[0], "start_time") + f(st[0], "duration"),
        f(st[1], "start_time") + f(st[1], "duration"),
    );
    // each clip's packets are a run of its source's
    let (ours, mut i) = (audio_hashes(&pp), 0);
    for n in [0, 2] {
        let src = audio_hashes(&d.join(names[n]));
        let j = src
            .iter()
            .position(|h| *h == ours[i])
            .expect("audio frame not in the source");
        i += src[j..]
            .iter()
            .zip(&ours[i..])
            .take_while(|(a, b)| a == b)
            .count();
    }
    eprintln!(
        "plain cut: {} frames, video ends {ve:.6} s, audio {ae:.6} s; {} AAC frames, all the sources' in runs: {}",
        tp.len(),
        ours.len(),
        i == ours.len()
    );
    assert!(i == ours.len() && (ae - ve).abs() <= 1.0 / 48000.0);
}

/// A one-file timeline of `clips` at 29.97 fps, fixed.
fn one(src: &Path, clips: Vec<edit::Clip>) -> edit::Timeline {
    let t = frame_times(src);
    let mut tl = edit::Timeline::new(edit::Rate {
        num: 30000,
        den: 1001,
    });
    tl.files = vec![edit::File {
        path: src.to_str().unwrap().into(),
        dur: t[t.len() - 1] + 1001.0 / 30000.0,
    }];
    tl.clips = clips;
    tl.fix();
    tl
}

const SMALL: Enc = Enc::Video {
    codec: Codec::H264,
    deep: false,
    rate: Rate::Cbr(4_000_000),
    gop: 30,
};

fn small() -> Out {
    Out::Flat {
        w: 640,
        h: 360,
        view: None,
    }
}

/// Denoise on an editor timeline (`Edit::denoise`): a plain cut at 1× (v606 2–5 s) is mixed instead of copied, its
/// audio ends with the video, and it is the source run through `media::denoise::Stereo` from the clip's start with
/// the delay dropped: in step with it (best lag 0), closer to it than to the dry source, and quieter than that.
#[test]
fn timeline_denoise() {
    let Some(src) = sample("v606.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let tl = one(&src, vec![edit::Clip::new(0, 2.0, 5.0)]);
    let path = out("timeline_denoise.mp4");
    let mut job = edit_job(tl.clone(), path.clone(), small(), SMALL, None);
    let Cut::Edit(e) = &mut job.cut else {
        unreachable!()
    };
    e.denoise = Some(1.0);
    export(&g, &cx, job);
    let s = probe(&path, &["-show_entries", "stream=start_time,duration"]);
    let st: Vec<&str> = s.split("[/STREAM]").collect();
    let f = |s: &str, k| get(s, k).parse::<f64>().unwrap();
    let (ve, ae) = (
        f(st[0], "start_time") + f(st[0], "duration"),
        f(st[1], "start_time") + f(st[1], "duration"),
    );
    let theirs = audio_hashes(&src);
    let copied = (audio_hashes(&path).iter())
        .filter(|h| theirs.contains(h))
        .count();
    // both sides averaged, 48 kHz, from `pre` (FFmpeg input options)
    let mono = |p: &Path, pre: &[&str]| -> Vec<[f32; 2]> {
        let mut a = vec!["-v", "error"];
        a.extend_from_slice(pre);
        a.extend_from_slice(&["-i", p.to_str().unwrap()]);
        a.extend_from_slice(&["-map", "0:a:0", "-ar", "48000", "-f", "f32le", "-"]);
        let b = tool("ffmpeg", &a);
        let x: Vec<f32> = b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        x.as_chunks::<2>().0.to_vec()
    };
    let avg = |v: &[[f32; 2]]| v.iter().map(|s| 0.5 * (s[0] + s[1])).collect::<Vec<f32>>();
    let dry = mono(&src, &["-ss", "2.0", "-t", "3.2"]);
    let (mut dn, mut wet) = (media::denoise::Stereo::new(), vec![]);
    const F: usize = media::denoise::FRAME;
    for c in dry.as_chunks::<F>().0 {
        let (mut l, mut r): ([f32; F], [f32; F]) = (
            std::array::from_fn(|i| c[i][0]),
            std::array::from_fn(|i| c[i][1]),
        );
        dn.frame(&mut l, &mut r);
        wet.extend(l.iter().zip(&r).map(|(a, b)| 0.5 * (a + b)));
    }
    let (ours, dry) = (avg(&mono(&path, &[])), avg(&dry));
    let wet = &wet[media::denoise::DELAY..];
    // 2 s from 0.5 s in, lags up to ±1200 samples (the denoiser's delay is 960)
    let (o, n) = (24000, 96000);
    let (lw, cw, _) = xcorr(&ours, o, wet, o, n, 1200);
    let (ld, cd, _) = xcorr(&ours, o, &dry, o, n, 1200);
    let rms =
        |x: &[f32]| (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    let (ro, rd, rw) = (
        rms(&ours[o..o + n]),
        rms(&dry[o..o + n]),
        rms(&wet[o..o + n]),
    );
    eprintln!(
        "denoised plain cut: video ends {ve:.6} s, audio {ae:.6} s; {copied} AAC frames copied; against the source \
         denoised: lag {lw}, correlation {cw:.4}; against it dry: lag {ld}, correlation {cd:.4}; level {ro:.4} (dry \
         {rd:.4}, denoised here {rw:.4})"
    );
    assert!(copied == 0 && (ae - ve).abs() <= 1.0 / 48000.0);
    assert!(
        lw == 0 && cw > 0.95 && cw > cd + 0.05,
        "not the denoised source"
    );
    assert!(ro < 0.8 * rd, "not quieter");
}

/// Adjustments per clip (`edit::Clip::adj`): two freeze frames of one source frame (v635 at 2 s), the first at +1 EV
/// and the second plain, in one export: each clip's frame is what a one-clip export of it makes (the adjustments
/// switch at the cut and the first's do not reach the second), and the two differ.
#[test]
fn timeline_adjust() {
    let Some(src) = sample("v635.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    let lo = edit::Clip::still(0, 2.0, 0.3);
    let mut hi = lo.clone();
    hi.adj.exposure = 1.0;
    let grab = |clips: Vec<edit::Clip>, ks: &[u64]| {
        let (fx, got) = capture(&g, ks);
        let job = edit_job(
            one(&src, clips),
            out("timeline_adjust.mp4"),
            small(),
            SMALL,
            Some(fx),
        );
        export(&g, &cx, job);
        let mut v = std::mem::take(&mut *got.lock().unwrap());
        v.sort_by_key(|x| x.0);
        (v.iter())
            .map(|x| stitch::read_texture(&g.device, &g.queue, &x.1))
            .collect::<Vec<_>>()
    };
    let k = one(&src, vec![hi.clone(), lo.clone()]).frames(1).start;
    let both = grab(vec![hi.clone(), lo.clone()], &[0, k]);
    let (a, b) = (grab(vec![hi], &[0]), grab(vec![lo], &[0]));
    let diff = |x: &[u8], y: &[u8]| x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)).max().unwrap();
    let mean = |x: &[u8]| x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
    eprintln!(
        "clip 0 (+1 EV) against its own export: max difference {}; clip 1 (frame {k}) against its own: {}; means \
         {:.1} / {:.1}",
        diff(&both[0], &a[0]),
        diff(&both[1], &b[0]),
        mean(&both[0]),
        mean(&both[1])
    );
    assert!(both.len() == 2 && diff(&both[0], &a[0]) == 0 && diff(&both[1], &b[0]) == 0);
    assert!(mean(&both[0]) > mean(&both[1]) + 20.0);
}

/// The editor's tiny-planet preset (`edit::view::preset::planet`: Planet(1) looking down at 280° → a 100° level
/// perspective, eased in and out) in 30 steps through `export::view` and `stitch::Preview` from a synthetic equirect:
/// the mean absolute change between consecutive frames must rise and fall smoothly, no step more than 1.5× the larger
/// of its neighbours. The mapping without the planet distance (stereographic until the projection reaches
/// perspective) pops at the end, and Planet(0) is the perspective view exactly.
#[test]
fn planet_transition() {
    let (_l, g, cx) = gpu();
    // a smooth pattern on the sphere (no seam, no pole): sines of the direction
    let (ew, eh) = (2048usize, 1024usize);
    let eq: Vec<u8> = (0..ew * eh)
        .flat_map(|i| {
            let (th, ph) = (
                std::f64::consts::TAU * (i % ew) as f64 / ew as f64,
                std::f64::consts::PI * (i / ew) as f64 / (eh - 1) as f64,
            );
            let (x, y, z) = (ph.sin() * th.cos(), -ph.sin() * th.sin(), ph.cos());
            let c = |v: f64| (127.5 + 120.0 * v).round() as u8;
            [
                c((7.0 * x + 3.0 * z).sin() * (5.0 * y).cos()),
                c((6.0 * y - 4.0 * x).sin()),
                c((9.0 * z + 2.0 * y).cos() * (3.0 * x).sin()),
                255,
            ]
        })
        .collect();
    let mk = |w: u32, h: u32, u: wgpu::TextureUsages| {
        g.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: u,
            view_formats: &[],
        })
    };
    use wgpu::TextureUsages as U;
    let et = mk(ew as u32, eh as u32, U::TEXTURE_BINDING | U::COPY_DST);
    g.queue.write_texture(
        et.as_image_copy(),
        &eq,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * ew as u32),
            rows_per_image: None,
        },
        et.size(),
    );
    let pano = cx.pano(&et.create_view(&Default::default()));
    let pv = stitch::Preview::pano(&cx, wgpu::TextureFormat::Rgba8Unorm);
    pv.pose(&stitch::Pose::default());
    let (w, h) = (640u32, 360u32);
    let t = mk(w, h, U::RENDER_ATTACHMENT | U::COPY_SRC);
    let tv = t.create_view(&Default::default());
    let draw = |v: &View| {
        let mut e = g.device.create_command_encoder(&Default::default());
        pv.render(&mut e, &tv, [w, h], v, &pano);
        g.queue.submit([e.finish()]);
        stitch::read_texture(&g.device, &g.queue, &t)
    };
    let to = edit::View::look(20.0, 0.0, 0.0, 100.0, edit::Proj::Persp);
    let keys = edit::view::preset::planet(to, 1.0, false).unwrap();
    let mad = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).abs())
            .sum::<f64>()
            / (a.len() as f64 * 0.75)
    };
    // (name, the stitch view of a timeline view)
    type Map = fn(&edit::View) -> View;
    let maps: [(&str, Map); 2] = [
        ("planet distance", export::view),
        ("stereographic until perspective", |v| {
            let mut s = export::view(v);
            if let edit::Proj::Planet(_) = v.proj {
                s.planet = 1.0;
            }
            s
        }),
    ];
    let mut worst = vec![];
    for (name, map) in maps {
        let fr: Vec<Vec<u8>> = (0..=30)
            .map(|i| draw(&map(&edit::view::at(&keys, i as f64 / 30.0, false).unwrap())))
            .collect();
        let st: Vec<f64> = fr.windows(2).map(|w| mad(&w[0], &w[1])).collect();
        let r = (0..st.len())
            .map(|i| {
                let nb = [i.checked_sub(1).map(|j| st[j]), st.get(i + 1).copied()];
                st[i] / nb.iter().flatten().fold(0f64, |a, &b| a.max(b)).max(1e-9)
            })
            .fold(0f64, f64::max);
        eprintln!(
            "{name}: mean |Δ| per step {:?}; largest step / its larger neighbour: {r:.2}",
            st.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>()
        );
        worst.push(r);
    }
    assert!(worst[0] < 1.5, "the transition jumps");
    assert!(worst[1] > 2.0, "the reference mapping should pop");
    let p = |proj| draw(&export::view(&edit::View { proj, ..to }));
    let (a, b) = (p(edit::Proj::Persp), p(edit::Proj::Planet(0.0)));
    let md = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
    eprintln!("perspective vs Planet(0): max difference {md}");
    assert!(md <= 1);
}

/// `Job::flat`: a 2:1 `Out::Pano` picture that is not a sphere goes out without spherical metadata (no GPano in a
/// grab, no `sv3d` / `st3d` in video); the same job without the flag has both.
#[test]
fn flat_flag() {
    let Some(src) = sample("v635.insv") else {
        return;
    };
    let (_l, g, cx) = gpu();
    for flat in [false, true] {
        let job = |path: &Path, enc| Job {
            cut: Cut::Segs(vec![Seg {
                files: vec![src.clone()],
                t: [0.0, 0.1],
            }]),
            path: path.to_path_buf(),
            out: Out::Pano(2048),
            enc,
            raft: false,
            colour: None,
            stab: None,
            flat,
            fx: None,
            pre: None,
        };
        let (j, v) = (
            out(&format!("flag_{flat}.jpg")),
            out(&format!("flag_{flat}.mp4")),
        );
        export(&g, &cx, job(&j, Enc::Jpeg(90)));
        let enc = Enc::Video {
            codec: Codec::H264,
            deep: false,
            rate: Rate::Cbr(4_000_000),
            gop: 30,
        };
        export(&g, &cx, job(&v, enc));
        let b = std::fs::read(&j).unwrap();
        let gpano = b.windows(5).any(|w| w == b"GPano");
        let s = probe(
            &v,
            &[
                "-show_entries",
                "stream=width,height,nb_frames:stream_side_data",
            ],
        );
        eprintln!("flat {flat}: GPano {gpano}; ffprobe:\n{}", s.trim());
        assert_eq!(gpano, !flat);
        assert_eq!(s.contains("Spherical Mapping"), !flat);
        assert_eq!(s.contains("Stereo 3D"), !flat);
        assert!(s.contains("width=2048") && s.contains("height=1024"));
    }
}

/// The synthetic recording (`export::synth`): 300 frames of 256² per lens (10 s), made once per test binary.
fn synth() -> PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let p = out("synth300.insv");
        export::synth::insv(&p, 300, 256).unwrap();
        p
    })
    .clone()
}

/// A flat 640 × 360 H.264 job of `src` over `t` (s): template seam, the default view.
fn flat_job(src: &Path, t: [f64; 2], path: &Path) -> Job {
    Job {
        cut: Cut::Segs(vec![Seg {
            files: vec![src.to_path_buf()],
            t,
        }]),
        path: path.to_path_buf(),
        out: Out::Flat {
            w: 640,
            h: 360,
            view: None,
        },
        enc: Enc::Video {
            codec: Codec::H264,
            deep: false,
            rate: Rate::Cbr(4_000_000),
            gop: 30,
        },
        raft: false,
        colour: None,
        stab: None,
        flat: false,
        fx: None,
        pre: None,
    }
}

/// A hook that takes `ms` per frame: a job that lasts.
fn slow(ms: u64) -> Option<Hook> {
    Some(Box::new(move |_: Fx<'_>| {
        std::thread::sleep(Duration::from_millis(ms))
    }))
}

/// `job` through `run_into` on this thread.
fn run(
    g: &Gpu,
    cx: &Ctx,
    job: Job,
    d: Dest,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(Progress),
) -> Result<Progress, String> {
    export::run_into(g, cx, &mut None, job, d, stop, progress)
}

/// The samples of `p`'s one video track.
fn samples(p: &Path) -> usize {
    let m = media::mp4::Mp4::open(p).unwrap();
    let v: Vec<_> = (m.tracks.iter())
        .filter(|t| {
            matches!(
                t.codec,
                media::mp4::Codec::Avc { .. } | media::mp4::Codec::Hevc { .. }
            )
        })
        .collect();
    assert_eq!(v.len(), 1, "{}", p.display());
    v[0].samples.len()
}

/// Nothing at `p` or its `.part`, not even a link.
fn gone(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_err() && std::fs::symlink_metadata(part::of(p)).is_err()
}

/// `id`'s state once it has ended.
fn wait(q: &Queue, id: u64, secs: u64) -> State {
    let t = Instant::now();
    loop {
        match q.state(id) {
            Some(State::Queued | State::Running(_)) => {}
            Some(s) => return s,
            None => panic!("job {id} forgotten"),
        }
        assert!(t.elapsed().as_secs() < secs, "job {id}: {:?}", q.state(id));
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// `id` running with at least `n` frames done.
fn running(q: &Queue, id: u64, n: u64) {
    let t = Instant::now();
    loop {
        match q.state(id) {
            Some(State::Running(p)) if p.done >= n => return,
            Some(State::Queued | State::Running(_)) => {}
            s => panic!("job {id}: {s:?}"),
        }
        assert!(t.elapsed().as_secs() < 300, "job {id}: {:?}", q.state(id));
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `source` reads what a job reads first, without the GPU: the synthetic recording's rate, length and tracks, no IMU;
/// with stabilisation it fails as the job would; the same MP4 without its trailer is refused. v635 with its IMU.
#[test]
fn source_head() {
    let src = synth();
    let s = export::source(std::slice::from_ref(&src), false).unwrap();
    eprintln!(
        "synthetic: {:?} {} frames {:.3} s, {:?}, IMU {}, AAC {}",
        s.fps, s.frames, s.dur, s.tracks, s.imu, s.aac
    );
    assert_eq!((s.fps, s.frames), ([30000, 1001], 300));
    assert!((s.dur - 300.0 * 1001.0 / 30000.0).abs() < 1e-9);
    let t = export::Track {
        codec: Codec::H264,
        deep: false,
        size: [256, 256],
    };
    assert_eq!(s.tracks, [t; 2]);
    assert!(!s.imu && !s.aac);
    assert_eq!((s.meta.stream, s.meta.camera.as_str()), (3, "Insta360 X5"));
    let e = export::source(std::slice::from_ref(&src), true).unwrap_err();
    assert!(
        e.contains("no IMU or exposure data to stabilise with"),
        "{e}"
    );
    let plain = out("synth_plain.mp4");
    let start = media::insv::Trailer::open(&src).unwrap().start as usize;
    std::fs::write(&plain, &std::fs::read(&src).unwrap()[..start]).unwrap();
    let e = export::source(&[plain], false).unwrap_err();
    assert!(e.contains("no Insta360 trailer"), "{e}");
    let Some(v) = sample("v635.insv") else {
        return;
    };
    let s = export::source(&[v], true).unwrap();
    eprintln!(
        "v635: {:?} {} frames {:.3} s, {:?}, IMU {}, AAC {}",
        s.fps, s.frames, s.dur, s.tracks, s.imu, s.aac
    );
    assert!(s.imu && s.tracks[0].codec == Codec::H265);
}

/// A job writes into the `.part` it is handed and publishes that very file, a video and a still; the strict checks
/// (frame count, the JPEG's decode) pass. The file is known by a second handle on it: its identity at the end (on
/// macOS FAT and exFAT the writes change it).
#[test]
fn part_into() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let stop = AtomicBool::new(false);
    let strict = |p| Dest {
        part: Some(p),
        strict: true,
        frozen_ok: false,
    };
    let path = out("part_into.mp4");
    let p = part::create(&path).unwrap();
    let f = p.f.try_clone().unwrap();
    let job = flat_job(&src, [0.0, 2.0], &path);
    let r = run(&g, &cx, job, strict(p), &stop, &mut |_| {}).unwrap();
    assert_eq!((r.done, r.total, r.concealed), (60, 60, 0));
    assert!(part::is(&path, &part::id_of(&f).unwrap()) && !part::of(&path).exists());
    assert_eq!(samples(&path), 60);
    let jpg = out("part_into.jpg");
    let p = part::create(&jpg).unwrap();
    let f = p.f.try_clone().unwrap();
    let job = Job {
        out: Out::Pano(512),
        enc: Enc::Jpeg(90),
        ..flat_job(&src, [1.0, 1.0], &jpg)
    };
    run(&g, &cx, job, strict(p), &stop, &mut |_| {}).unwrap();
    assert!(part::is(&jpg, &part::id_of(&f).unwrap()) && !part::of(&jpg).exists());
    assert_eq!(
        media::mjpeg::size(&std::fs::read(&jpg).unwrap()),
        Some((512, 256))
    );
}

/// A file that appears at the job's path while it runs fails the job and stays as it was; the `.part` goes.
#[test]
fn no_replace() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let stop = AtomicBool::new(false);
    let path = out("no_replace.mp4");
    let job = flat_job(&src, [0.0, 1.0], &path);
    let e = run(&g, &cx, job, Dest::default(), &stop, &mut |p| {
        if p.done == 5 {
            std::fs::write(&path, b"theirs").unwrap();
        }
    })
    .unwrap_err();
    assert!(
        e.contains("appeared while it was rendered; nothing replaced"),
        "{e}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"theirs");
    assert!(!part::of(&path).exists());
}

/// A link at `<path>.part`, dangling or to a file, fails the job at its start without writing through it, in
/// `run_into` and on a queue's thread; the link and its target stay as they were.
#[test]
fn link_part() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let q = Queue::new(g.clone(), cx.clone());
    let stop = AtomicBool::new(false);
    let (path, target) = (out("link_part.mp4"), out("link_target"));
    let pp = part::of(&path);
    for (queued, content) in [(false, None), (false, Some(&b"keep"[..])), (true, None)] {
        let _ = std::fs::remove_file(&target);
        if let Some(c) = content {
            std::fs::write(&target, c).unwrap();
        }
        std::os::unix::fs::symlink(&target, &pp).unwrap();
        let job = flat_job(&src, [0.0, 0.5], &path);
        let e = match queued {
            false => run(&g, &cx, job, Dest::default(), &stop, &mut |_| {}).unwrap_err(),
            true => match wait(&q, q.push(job), 60) {
                State::Failed(e) => e,
                s => panic!("{s:?}"),
            },
        };
        assert!(e.contains("being written by another export"), "{e}");
        assert!(std::fs::symlink_metadata(&pp).unwrap().is_symlink());
        assert_eq!(std::fs::read(&target).ok().as_deref(), content);
        assert!(!path.exists());
        std::fs::remove_file(&pp).unwrap();
    }
}

/// A stop in the last frame's progress ends the job cancelled with nothing left: a video (its encoder still
/// draining) and a still (written by then).
#[test]
fn stop_at_end() {
    let src = synth();
    let (_l, g, cx) = gpu();
    for (name, enc) in [
        ("stop_at_end.mp4", None),
        ("stop_at_end.jpg", Some(Enc::Jpeg(90))),
    ] {
        let path = out(name);
        let mut job = flat_job(&src, [0.0, 1.0], &path);
        if let Some(e) = enc {
            job.enc = e;
        }
        let stop = AtomicBool::new(false);
        let d = Dest {
            strict: true,
            ..Default::default()
        };
        let r = run(&g, &cx, job, d, &stop, &mut |p| {
            if p.done == p.total {
                stop.store(true, Relaxed)
            }
        });
        assert_eq!(r, Err("cancelled".to_string()), "{name}");
        assert!(gone(&path), "{name}");
    }
}

/// A stop while the muxer moves the `moov` in front of 60 MB of noise frames ends the job cancelled, at the next
/// 4 MiB chunk, with nothing left.
#[test]
fn cancel_in_finish() {
    use std::os::unix::fs::FileExt;
    let src = synth();
    let (_l, g, cx) = gpu();
    let (w, h) = (1920u32, 960u32);
    let size = wgpu::Extent3d {
        width: w + 128,
        height: h + 64,
        depth_or_array_layers: 1,
    };
    // noise the hook copies over each frame from an offset of its own: nothing the encoder can shrink
    let noise = g.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut s = 1u32;
    let px: Vec<u8> = (0..size.width * size.height * 4)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 24) as u8
        })
        .collect();
    g.queue.write_texture(
        noise.as_image_copy(),
        &px,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 4),
            rows_per_image: None,
        },
        size,
    );
    let hook: Hook = Box::new(move |fx: Fx<'_>| {
        let k = fx.k as u32;
        let from = wgpu::TexelCopyTextureInfo {
            texture: &noise,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: k * 13 % 128,
                y: k * 7 % 64,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        };
        fx.enc
            .copy_texture_to_texture(from, fx.tex.as_image_copy(), fx.tex.size());
    });
    let path = out("cancel_in_finish.mp4");
    let pp = part::of(&path);
    let job = Job {
        out: Out::Flat { w, h, view: None },
        enc: Enc::Video {
            codec: Codec::H265,
            deep: false,
            rate: Rate::Cbr(60_000_000),
            gop: 30,
        },
        fx: Some(hook),
        ..flat_job(&src, [0.0, 8.0], &path)
    };
    let (stop, done) = (AtomicBool::new(false), AtomicBool::new(false));
    let (r, at, end) = std::thread::scope(|sc| {
        let w = sc.spawn(|| {
            // the mdat's size, the first thing the finish writes
            let mut b = [0u8; 4];
            let mut f = None;
            while !done.load(Relaxed) {
                f = f.or_else(|| std::fs::File::open(&pp).ok());
                if let Some(f) = &f
                    && f.read_exact_at(&mut b, 36).is_ok()
                    && b != [0; 4]
                {
                    stop.store(true, Relaxed);
                    return Some((Instant::now(), f.metadata().unwrap().len()));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            None
        });
        let r = run(&g, &cx, job, Dest::default(), &stop, &mut |_| {});
        let end = Instant::now();
        done.store(true, Relaxed);
        (r, w.join().unwrap(), end)
    });
    let (t, len) = at.expect("the finish never began");
    eprintln!(
        "stopped in the finish of {:.1} MB: cancelled {:.1} ms later",
        len as f64 / 1e6,
        (end - t).as_secs_f64() * 1e3
    );
    assert_eq!(r, Err("cancelled".to_string()));
    assert!(gone(&path));
    assert!(len > 40_000_000, "{len} bytes: too few chunks to stop in");
}

/// A queued job's cancel is final at once: `Cancelled`, its handed `.part` removed, never started; the job before it
/// runs on. `forget` drops a finished entry at once, a running one once it ends.
#[test]
fn queued_cancel() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let q = Queue::new(g.clone(), cx.clone());
    let (pa, pb) = (out("queued_a.mp4"), out("queued_b.mp4"));
    let ia = q.push(Job {
        fx: slow(10),
        ..flat_job(&src, [0.0, 5.0], &pa)
    });
    let d = Dest {
        part: Some(part::create(&pb).unwrap()),
        strict: true,
        frozen_ok: false,
    };
    let ib = q.push_into(flat_job(&src, [0.0, 1.0], &pb), d).unwrap();
    running(&q, ia, 1);
    assert!(q.busy() && q.alive());
    assert_eq!(q.state(ib), Some(State::Queued));
    q.cancel(ib);
    assert_eq!(q.state(ib), Some(State::Cancelled));
    assert!(gone(&pb));
    q.forget(ib);
    assert_eq!(q.state(ib), None);
    q.forget(ia);
    assert!(matches!(q.state(ia), Some(State::Running(_))));
    let t = Instant::now();
    while q.state(ia).is_some() {
        assert!(t.elapsed().as_secs() < 300);
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(samples(&pa), 150);
    assert!(gone(&pb) && q.jobs().is_empty());
}

/// Panics fail their job alone, leaving neither the file nor its `.part`, and the lane runs the next job: a hook that
/// panics at frame 5 and one that panics as it drops (both caught in the job). The second through `run_into` itself
/// too, as the app's export screen runs it: an error, nothing left.
#[test]
fn lane_panic() {
    struct Bomb;
    impl Drop for Bomb {
        fn drop(&mut self) {
            panic!("a hook that panics as it drops")
        }
    }
    let src = synth();
    let (_l, g, cx) = gpu();
    let late = || -> Hook {
        let bomb = Bomb;
        Box::new(move |_: Fx<'_>| {
            let _ = &bomb;
        })
    };
    let path = out("panic_direct.mp4");
    let job = Job {
        fx: Some(late()),
        ..flat_job(&src, [0.0, 0.5], &path)
    };
    let stop = AtomicBool::new(false);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(&g, &cx, job, Dest::default(), &stop, &mut |_| {})
    }));
    assert_eq!(
        r.ok(),
        Some(Err("panic: a hook that panics as it drops".into()))
    );
    assert!(gone(&path), "output left");
    let q = Queue::new(g.clone(), cx.clone());
    let boom: Hook = Box::new(|fx: Fx<'_>| {
        if fx.k == 5 {
            panic!("fx hook at frame 5")
        }
    });
    let jobs = [
        (
            "panic_hook.mp4",
            Some(boom),
            Some("panic: fx hook at frame 5"),
        ),
        (
            "panic_drop.mp4",
            Some(late()),
            Some("panic: a hook that panics as it drops"),
        ),
        ("panic_after.mp4", None, None),
    ];
    let mut ids = vec![];
    for (name, fx, want) in jobs {
        let path = out(name);
        let d = Dest {
            part: Some(part::create(&path).unwrap()),
            ..Default::default()
        };
        let job = Job {
            fx,
            ..flat_job(&src, [0.0, 0.5], &path)
        };
        ids.push((q.push_into(job, d).unwrap(), name, path, want));
    }
    for (id, name, path, want) in ids {
        let s = wait(&q, id, 120);
        eprintln!("{name}: {s:?}");
        match (s, want) {
            (State::Failed(e), Some(w)) => assert_eq!(e, w, "{name}"),
            (State::Done(p), None) => assert_eq!(samples(&path), p.total as usize),
            (s, w) => panic!("{name}: {s:?}, expected {w:?}"),
        }
        assert!(want.is_none() || gone(&path), "{name}: output left");
    }
    assert!(q.alive());
}

/// `f` as a `Job::pre`.
fn pre(
    f: impl FnOnce(&Ctx, &AtomicBool) -> Result<Option<Hook>, String> + Send + 'static,
) -> Option<Pre> {
    Some(Box::new(f))
}

/// A hook that counts its frames into `n`.
fn count(n: &Arc<AtomicU64>) -> Hook {
    let n = n.clone();
    Box::new(move |_: Fx<'_>| {
        n.fetch_add(1, Relaxed);
    })
}

/// `jobs` (file name, `pre`) on `q`, each over 0.5 s of `src` into a `.part` it is handed; how they ended, in order.
fn pre_jobs(q: &Queue, src: &Path, jobs: Vec<(&str, Option<Pre>)>) -> Vec<(State, PathBuf)> {
    let ids: Vec<_> = (jobs.into_iter())
        .map(|(name, pre)| {
            let path = out(name);
            let d = Dest {
                part: Some(part::create(&path).unwrap()),
                ..Default::default()
            };
            let job = Job {
                pre,
                ..flat_job(src, [0.0, 0.5], &path)
            };
            (q.push_into(job, d).unwrap(), path)
        })
        .collect();
    (ids.into_iter())
        .map(|(id, path)| {
            let s = wait(q, id, 120);
            eprintln!("{}: {s:?}", path.display());
            (s, path)
        })
        .collect()
}

/// `Job::pre` runs once on the lane's thread with the stop clear, before the first frame; the hook it hands back runs
/// before `Job::fx` on every frame. Also the hook alone, and `Job::fx` alone after a `pre` that hands none.
#[test]
fn pre_runs_first() {
    type Log = Arc<Mutex<Vec<String>>>;
    let src = synth();
    let (_l, g, cx) = gpu();
    let q = Queue::with(g.clone(), cx.clone(), "pre-lane").unwrap();
    let to = |log: &Log, c: char| -> Hook {
        let log = log.clone();
        Box::new(move |f: Fx<'_>| log.lock().unwrap().push(format!("{c}{}", f.k)))
    };
    let mut jobs = vec![];
    for (name, hook, fx) in [
        ("pre_both.mp4", true, true),
        ("pre_hook.mp4", true, false),
        ("pre_fx.mp4", false, true),
    ] {
        let log = Log::default();
        let (l, h) = (log.clone(), hook.then(|| to(&log, 'h')));
        let path = out(name);
        let job = Job {
            fx: fx.then(|| to(&log, 'f')),
            pre: pre(move |_, stop| {
                let th = std::thread::current();
                let s = format!("pre on {:?}, stop {}", th.name(), stop.load(Relaxed));
                l.lock().unwrap().push(s);
                Ok(h)
            }),
            ..flat_job(&src, [0.0, 0.5], &path)
        };
        jobs.push((q.push(job), name, path, log, hook, fx));
    }
    for (id, name, path, log, hook, fx) in jobs {
        let s = wait(&q, id, 120);
        let State::Done(p) = s else {
            panic!("{name}: {s:?}")
        };
        assert!(p.done == p.total && p.total >= 10, "{name}: {p:?}");
        assert_eq!(samples(&path), p.total as usize, "{name}");
        let mut want = vec![r#"pre on Some("pre-lane"), stop false"#.to_string()];
        for k in 0..p.total {
            want.extend(hook.then(|| format!("h{k}")));
            want.extend(fx.then(|| format!("f{k}")));
        }
        assert_eq!(*log.lock().unwrap(), want, "{name}");
    }
}

/// A job cancelled in its `pre`: the stop reaches it, the lane is busy meanwhile (the job `Running` at 0 frames), and
/// the job ends `Cancelled` without a frame of the hook it hands back after the stop, leaving neither the file nor
/// the `.part` it was handed. A queued job's `pre` is dropped unrun once it is cancelled.
#[test]
fn pre_cancel() {
    struct Flag(Arc<AtomicBool>);
    impl Drop for Flag {
        fn drop(&mut self) {
            self.0.store(true, Relaxed)
        }
    }
    let src = synth();
    let (_l, g, cx) = gpu();
    let q = Queue::with(g.clone(), cx.clone(), "pre-cancel").unwrap();
    let (pa, pb) = (out("pre_cancel_a.mp4"), out("pre_cancel_b.mp4"));
    let (inside, frames) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicU64::new(0)),
    );
    let (i, n) = (inside.clone(), frames.clone());
    let d = Dest {
        part: Some(part::create(&pa).unwrap()),
        ..Default::default()
    };
    let job = Job {
        pre: pre(move |_, stop| {
            i.store(true, Relaxed);
            let t = Instant::now();
            while !stop.load(Relaxed) {
                if t.elapsed().as_secs() > 120 {
                    return Err("never stopped".into());
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(Some(count(&n)))
        }),
        ..flat_job(&src, [0.0, 1.0], &pa)
    };
    let ia = q.push_into(job, d).unwrap();
    let t = Instant::now();
    while !inside.load(Relaxed) {
        assert!(t.elapsed().as_secs() < 120, "{:?}", q.state(ia));
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(q.busy());
    assert_eq!(q.state(ia), Some(State::Running(Progress::default())));
    assert!(part::of(&pa).exists());
    let (ran, dropped) = (
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    );
    let (r, flag) = (ran.clone(), Flag(dropped.clone()));
    let ib = q.push(Job {
        pre: pre(move |_, _| {
            let _ = &flag;
            r.store(true, Relaxed);
            Ok(None)
        }),
        ..flat_job(&src, [0.0, 1.0], &pb)
    });
    q.cancel(ib);
    q.cancel(ia);
    assert_eq!(wait(&q, ia, 120), State::Cancelled);
    assert_eq!(q.state(ib), Some(State::Cancelled));
    assert_eq!(frames.load(Relaxed), 0, "frames after the stop");
    let t = Instant::now();
    while q.busy() {
        assert!(t.elapsed().as_secs() < 120, "{:?}", q.jobs());
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(dropped.load(Relaxed) && !ran.load(Relaxed));
    assert!(gone(&pa) && gone(&pb));
}

/// A `pre` that fails fails its job with its error: also a wgpu error in it, at once (its own error scope), before a
/// frame of the hook it hands back. Neither the file nor its `.part` is left, and the lane runs the next job.
#[test]
fn pre_fails() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let q = Queue::with(g.clone(), cx.clone(), "pre-fails").unwrap();
    let frames = Arc::new(AtomicU64::new(0));
    let (n, gd) = (frames.clone(), g.clone());
    let wgpu = pre(move |_, _| {
        // a texture of no size: a validation error
        let _ = gd.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("nothing"),
            size: wgpu::Extent3d {
                width: 0,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Ok(Some(count(&n)))
    });
    let jobs = vec![
        ("pre_err.mp4", pre(|_, _| Err("x".into()))),
        ("pre_wgpu.mp4", wgpu),
        ("pre_next.mp4", pre(|_, _| Ok(None))),
    ];
    let s = pre_jobs(&q, &src, jobs);
    assert_eq!(s[0].0, State::Failed("x".into()));
    assert!(matches!(&s[1].0, State::Failed(e) if e.starts_with("wgpu: ")));
    assert_eq!(frames.load(Relaxed), 0, "frames after the pre's wgpu error");
    assert!(gone(&s[0].1) && gone(&s[1].1), "output left");
    let State::Done(p) = s[2].0 else {
        panic!("{:?}", s[2].0)
    };
    assert_eq!(samples(&s[2].1), p.total as usize);
    assert!(q.alive() && q.lost().is_none());
}

/// Panics in a `pre` and in the hook it hands back fail their job alone, leaving neither the file nor its `.part`;
/// the lane runs the next job.
#[test]
fn pre_panics() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let q = Queue::with(g.clone(), cx.clone(), "pre-panics").unwrap();
    let late: Hook = Box::new(|f: Fx<'_>| {
        if f.k == 5 {
            panic!("its hook at frame 5")
        }
    });
    let jobs = vec![
        ("pre_panic.mp4", pre(|_, _| panic!("a pre that panics"))),
        ("pre_hook_panic.mp4", pre(|_, _| Ok(Some(late)))),
        ("pre_after.mp4", pre(|_, _| Ok(None))),
    ];
    let s = pre_jobs(&q, &src, jobs);
    assert_eq!(s[0].0, State::Failed("panic: a pre that panics".into()));
    assert_eq!(s[1].0, State::Failed("panic: its hook at frame 5".into()));
    assert!(gone(&s[0].1) && gone(&s[1].1), "output left");
    let State::Done(p) = s[2].0 else {
        panic!("{:?}", s[2].0)
    };
    assert_eq!(samples(&s[2].1), p.total as usize);
    assert!(q.alive());
}

/// Two queues on one device run at once, as the MCP server's still and export lanes: a RAFT 360 export of the
/// synthetic recording in one; in the other, the whole recording flat, then a flat RAFT one (two models taking turns
/// at BURN). Overlapping, each file complete (strict). With `X5_SAMPLES`: v635 flat with RAFT and the horizon lock in
/// one (after a `trim`), three 360 stills in the other.
#[test]
fn two_lanes() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let a = Queue::with(g.clone(), cx.clone(), "lane-a").unwrap();
    let b = Queue::with(g.clone(), cx.clone(), "lane-b").unwrap();
    let strict = || Dest {
        strict: true,
        ..Default::default()
    };
    let (pa, pb, pc) = (out("lanes_a.mp4"), out("lanes_b.mp4"), out("lanes_c.mp4"));
    let ja = Job {
        out: Out::Pano(1024),
        enc: Enc::Video {
            codec: Codec::H265,
            deep: false,
            rate: Rate::Cbr(8_000_000),
            gop: 30,
        },
        raft: true,
        colour: Some(0.1),
        ..flat_job(&src, [0.0, 4.0], &pa)
    };
    let t = Instant::now();
    // the other lane once this one has its model and kernels
    let ia = a.push_into(ja, strict()).unwrap();
    running(&a, ia, 1);
    let jb = Job {
        fx: slow(5),
        ..flat_job(&src, [0.0, f64::INFINITY], &pb)
    };
    let jc = Job {
        raft: true,
        ..flat_job(&src, [0.0, 1.0], &pc)
    };
    let (ib, ic) = (
        b.push_into(jb, strict()).unwrap(),
        b.push_into(jc, strict()).unwrap(),
    );
    let live = |s: &Option<State>| matches!(s, Some(State::Queued | State::Running(_)));
    let on = |s: &Option<State>| matches!(s, Some(State::Running(p)) if p.done > 0);
    let mut both = [0; 2];
    loop {
        let s = [a.state(ia), b.state(ib), b.state(ic)];
        for k in 0..2 {
            both[k] += (on(&s[0]) && on(&s[k + 1])) as u32;
        }
        if !s.iter().any(live) {
            break;
        }
        assert!(t.elapsed().as_secs() < 600, "{s:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
    let s = [a.state(ia), b.state(ib), b.state(ic)];
    eprintln!(
        "synthetic in {:.1} s: {s:?}; the 360 one running with the flat ones at {both:?} looks",
        t.elapsed().as_secs_f64()
    );
    for (s, p, n) in [(&s[0], &pa, 120), (&s[1], &pb, 300), (&s[2], &pc, 30)] {
        assert!(matches!(s, Some(State::Done(x)) if x.done == n), "{s:?}");
        assert_eq!(samples(p), n as usize);
    }
    assert!(both[0] > 0 && both[1] > 0, "the lanes never ran at once");
    assert!(a.lost().is_none() && b.lost().is_none());
    let Some(v) = sample("v635.insv") else {
        return;
    };
    a.trim();
    let pv = out("lanes_v635.mp4");
    let ja = Job {
        out: Out::Flat {
            w: 1280,
            h: 720,
            view: None,
        },
        raft: true,
        colour: Some(0.1),
        stab: Some((stab::Lock::Horizon, 0.5)),
        ..flat_job(&v, [0.0, 3.0], &pv)
    };
    let ia = a.push_into(ja, strict()).unwrap();
    running(&a, ia, 10);
    let stills: Vec<_> = [0.5, 1.0, 1.5]
        .into_iter()
        .map(|s| {
            let p = out(&format!("lanes_v635_{s}.jpg"));
            let job = Job {
                out: Out::Pano(3840),
                enc: Enc::Jpeg(90),
                ..flat_job(&v, [s, s], &p)
            };
            (b.push_into(job, strict()).unwrap(), p)
        })
        .collect();
    for (id, p) in &stills {
        let s = wait(&b, *id, 300);
        eprintln!("{}: {s:?}, the export {:?}", p.display(), a.state(ia));
        assert!(matches!(s, State::Done(_)), "{s:?}");
    }
    let s = wait(&a, ia, 600);
    eprintln!("{}: {s:?}", pv.display());
    let State::Done(x) = s else { panic!("{s:?}") };
    assert_eq!(samples(&pv), x.total as usize);
}

/// `detach` returns at once while a job runs: the queued job is cancelled (its `.part` gone at once), the running one
/// finishes and publishes, and the thread ends after it.
#[test]
fn detach() {
    let src = synth();
    let (_l, g, cx) = gpu();
    let n = Arc::strong_count(&g);
    let q = Queue::with(g.clone(), cx.clone(), "detached").unwrap();
    let (pa, pb) = (out("detach_a.mp4"), out("detach_b.mp4"));
    let ia = q.push(Job {
        fx: slow(5),
        ..flat_job(&src, [0.0, f64::INFINITY], &pa)
    });
    let d = Dest {
        part: Some(part::create(&pb).unwrap()),
        ..Default::default()
    };
    q.push_into(flat_job(&src, [0.0, 1.0], &pb), d).unwrap();
    running(&q, ia, 5);
    let t = Instant::now();
    q.detach();
    let dt = t.elapsed();
    assert!(dt < Duration::from_millis(100), "{dt:?}");
    assert!(gone(&pb));
    // the thread's device handle goes when it ends
    let t = Instant::now();
    while Arc::strong_count(&g) > n {
        assert!(t.elapsed().as_secs() < 300, "the thread runs on");
        std::thread::sleep(Duration::from_millis(20));
    }
    eprintln!(
        "detached in {:.2} ms; the job ended {:.1} s later",
        dt.as_secs_f64() * 1e3,
        t.elapsed().as_secs_f64()
    );
    assert_eq!(samples(&pa), 300);
    assert!(gone(&pb));
}

/// The synthetic recording with frame 45 of both lenses broken (90 frames), made once: VideoToolbox conceals it and
/// the P pictures after it, up to the IDR at 60.
#[cfg(target_os = "macos")]
fn damaged() -> PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let p = out("synth_damaged.insv");
        export::synth::damaged(&p, 90, 256, 45).unwrap();
        p
    })
    .clone()
}

/// Strict, a concealed picture fails the job at its frame, nothing left (macOS: Linux fails the decode itself).
#[cfg(target_os = "macos")]
#[test]
fn strict_concealed() {
    let src = damaged();
    let (_l, g, cx) = gpu();
    let stop = AtomicBool::new(false);
    for (name, enc, at) in [
        ("strict_concealed.mp4", None, 45),
        ("strict_concealed.jpg", Some(Enc::Jpeg(90)), 0),
    ] {
        let path = out(name);
        let mut job = flat_job(&src, [0.0, 3.0], &path);
        if let Some(e) = enc {
            // frame 50: decoded from the IDR at 30, through the broken one
            (job.enc, job.cut) = (
                e,
                Cut::Segs(vec![Seg {
                    files: vec![src.clone()],
                    t: [50.0 * 1001.0 / 30000.0; 2],
                }]),
            );
        }
        let d = Dest {
            strict: true,
            ..Default::default()
        };
        let e = run(&g, &cx, job, d, &stop, &mut |_| {}).unwrap_err();
        eprintln!("{name}: {e}");
        assert!(
            e.starts_with(&format!("frame {at}: ")) && e.contains("could not be decoded"),
            "{e}"
        );
        assert!(gone(&path), "{name}");
    }
}

/// With `frozen_ok` the damaged recording exports whole and its concealed pictures are counted, as the app's export
/// screen (not strict) has it; a still through them too.
#[cfg(target_os = "macos")]
#[test]
fn frozen_ok() {
    let src = damaged();
    let (_l, g, cx) = gpu();
    let stop = AtomicBool::new(false);
    let frozen = || Dest {
        strict: true,
        frozen_ok: true,
        part: None,
    };
    for (name, d) in [
        ("frozen_ok.mp4", frozen()),
        ("frozen_gui.mp4", Dest::default()),
    ] {
        let path = out(name);
        let p = run(
            &g,
            &cx,
            flat_job(&src, [0.0, 3.0], &path),
            d,
            &stop,
            &mut |_| {},
        )
        .unwrap();
        eprintln!("{name}: {p:?}");
        assert_eq!(p.done, 90);
        assert!((2..=30).contains(&p.concealed), "{p:?}");
        assert_eq!(samples(&path), 90);
    }
    let path = out("frozen_ok.jpg");
    let job = Job {
        enc: Enc::Jpeg(90),
        cut: Cut::Segs(vec![Seg {
            files: vec![src.clone()],
            t: [50.0 * 1001.0 / 30000.0; 2],
        }]),
        ..flat_job(&src, [0.0, 0.0], &path)
    };
    let p = run(&g, &cx, job, frozen(), &stop, &mut |_| {}).unwrap();
    eprintln!("frozen_ok.jpg: {p:?}");
    assert!(p.concealed > 0 && path.exists());
}
