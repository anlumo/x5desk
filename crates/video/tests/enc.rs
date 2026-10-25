//! Vulkan Video encode on the X5's own footage (`X5_SAMPLES`, the live preview capture in `X5_LIVE`) and on synthetic
//! pictures, checked with the `ffprobe` / `ffmpeg` CLIs (test tools only) and with our own decoder. Streams go to
//! `X5_ENC_OUT` (default: Cargo's test tmp directory). GPU timings are noisy when other work shares the GPU.
mod mp4;

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io::Read,
    path::{Path, PathBuf},
    process::{ChildStdout, Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    time::Instant,
};
use video::{Cfg, Codec, Decoder, Encoder, Error, Gpu, Input, Packet, Rate, Ring, h264};

static GPU: Mutex<()> = Mutex::new(());

fn sample_file(name: &str) -> Option<PathBuf> {
    let Some(d) = std::env::var_os("X5_SAMPLES") else {
        println!("X5_SAMPLES unset: skipped");
        return None;
    };
    let p = PathBuf::from(d).join(name);
    if !p.exists() {
        println!("{} missing: skipped", p.display());
        return None;
    }
    Some(p)
}

fn ffmpeg() -> bool {
    let ok = ["ffmpeg", "ffprobe"].iter().all(|t| {
        Command::new(t)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    });
    if !ok {
        println!("ffmpeg / ffprobe not found: skipped");
    }
    ok
}

fn out(name: &str) -> PathBuf {
    let d = std::env::var_os("X5_ENC_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn hash(b: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    b.hash(&mut h);
    h.finish()
}

fn psnr(se: f64, n: usize, peak: f64) -> f64 {
    10.0 * (peak * peak / (se / n as f64).max(1e-12)).log10()
}

fn se8(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).pow(2) as u64)
        .sum::<u64>() as f64
}

/// Mean and minimum.
fn stats(v: &[f64]) -> (f64, f64) {
    (
        v.iter().sum::<f64>() / v.len() as f64,
        v.iter().copied().fold(f64::MAX, f64::min),
    )
}

/// `ffprobe -v error <args> <path>` as `key=value` lines.
fn probe(path: &Path, args: &[&str]) -> String {
    let o = Command::new("ffprobe")
        .args(["-v", "error", "-of", "default=nw=1"])
        .args(args)
        .arg(path)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap()
}

/// ffprobe's view of the video stream on one line; asserts each of `want` is among its `key=value` lines.
fn expect(path: &Path, want: &[&str]) {
    let s = probe(
        path,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,level,width,height,pix_fmt,color_range,color_space,color_transfer,\
             color_primaries,chroma_location",
        ],
    );
    println!(
        "{}: {}",
        path.display(),
        s.lines().collect::<Vec<_>>().join(" ")
    );
    for w in want {
        assert!(s.lines().any(|l| l == *w), "{w}");
    }
}

/// FFmpeg's decode of the given input arguments as raw frames on its stdout.
fn ff(input: &[&str], vf: Option<&str>, pix: &str) -> ChildStdout {
    let mut c = Command::new("ffmpeg");
    c.args(["-v", "error"]).args(input);
    if let Some(vf) = vf {
        c.args(["-vf", vf]);
    }
    c.args([
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix,
        "-",
    ]);
    c.stdout(Stdio::piped())
        .spawn()
        .unwrap()
        .stdout
        .take()
        .unwrap()
}

/// PSNR(Y) per frame of FFmpeg's decode of `ours` against FFmpeg's decode of `src` (input arguments) through `vf`.
fn psnr8(src: &[&str], vf: &str, ours: &Path, w: u32, h: u32) -> Vec<f64> {
    let (fs, ny) = ((w * h * 3 / 2) as usize, (w * h) as usize);
    let mut a = ff(src, Some(vf), "yuv420p");
    let mut b = ff(&["-i", ours.to_str().unwrap()], None, "yuv420p");
    let (mut x, mut y, mut v) = (vec![0; fs], vec![0; fs], vec![]);
    while a.read_exact(&mut x).is_ok() && b.read_exact(&mut y).is_ok() {
        v.push(psnr(se8(&x[..ny], &y[..ny]), ny, 255.0));
    }
    v
}

/// Errors FFmpeg reports while decoding `path` completely.
fn ff_errors(path: &Path) -> String {
    let o = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "null", "-"])
        .output()
        .unwrap();
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// An Annex B file of the packets: parameter sets in front of every key frame.
fn annexb(path: &Path, params: &[Vec<u8>], ps: &[Packet]) {
    let mut b = vec![];
    for p in ps {
        if p.key {
            for n in params {
                b.extend_from_slice(&[0, 0, 0, 1]);
                b.extend_from_slice(n);
            }
        }
        b.extend_from_slice(&p.data);
    }
    std::fs::write(path, b).unwrap();
}

/// The access units of an H.264 Annex B stream as NAL units without start codes: a new one where a parameter set,
/// SEI or AUD, or a picture's first slice follows a slice.
fn aus264(b: &[u8]) -> Vec<Vec<&[u8]>> {
    let starts: Vec<_> = (0..b.len().saturating_sub(2))
        .filter(|&i| b[i..i + 3] == [0, 0, 1])
        .map(|i| i + 3)
        .collect();
    let (mut aus, mut vcl): (Vec<Vec<&[u8]>>, _) = (vec![], false);
    for (k, &s) in starts.iter().enumerate() {
        let mut n = &b[s..starts.get(k + 1).map_or(b.len(), |&e| e - 3)];
        while let [x @ .., 0] = n {
            n = x;
        }
        if n.len() < 3 {
            continue;
        }
        let t = h264::nal_type(n);
        let v = matches!(t, 1..=5);
        if aus.is_empty() || vcl && (matches!(t, 6..=9 | 14..=18) || v && n[1] & 0x80 != 0) {
            aus.push(vec![]);
            vcl = false;
        }
        aus.last_mut().unwrap().push(n);
        vcl |= v;
    }
    aus
}

/// An MP4 of the packets through crate media's muxer; `pts` are frame numbers at `fps`.
fn mux(
    path: &Path,
    codec: Codec,
    params: &[Vec<u8>],
    ps: &[Packet],
    w: u32,
    h: u32,
    fps: [u32; 2],
) {
    use media::mp4::Codec as C;
    let c = match codec {
        Codec::H265 => C::Hevc {
            ps: params.to_vec(),
            nal: 4,
        },
        Codec::H264 => C::Avc {
            ps: params.to_vec(),
            nal: 4,
        },
    };
    let mut m = media::mux::Mux::create(path).unwrap();
    let t = m.track(c, fps[0], w, h, false).unwrap();
    let (mut b, d) = (vec![], fps[1] as i64);
    for p in ps {
        b.clear();
        p.prefixed(&mut b);
        m.write(t, &b, p.dts * d, p.pts * d, fps[1], p.key).unwrap();
    }
    m.finish(true).unwrap();
}

/// Checks an MP4 from `mux`: frame count and duration as ffprobe reads them.
fn check_mp4(path: &Path, n: usize, fps: [u32; 2]) {
    let s = probe(
        path,
        &[
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,duration",
        ],
    );
    let get = |k: &str| {
        s.lines()
            .find_map(|l| l.strip_prefix(k))
            .unwrap()
            .to_string()
    };
    let (frames, dur): (usize, f64) = (
        get("nb_read_frames=").parse().unwrap(),
        get("duration=").parse().unwrap(),
    );
    let want = n as f64 * fps[1] as f64 / fps[0] as f64;
    println!(
        "{}: {frames} frames, {dur:.4} s (want {n}, {want:.4} s)",
        path.display()
    );
    assert_eq!(frames, n);
    assert!((dur - want).abs() < 1e-3);
}

/// Frame numbers ffprobe flags as key frames.
fn keys(path: &Path) -> Vec<usize> {
    probe(path, &["-show_entries", "frame=key_frame"])
        .lines()
        .enumerate()
        .filter(|(_, l)| *l == "key_frame=1")
        .map(|(i, _)| i)
        .collect()
}

fn bitrate(ps: &[Packet], fps: [u32; 2]) -> f64 {
    let bytes: usize = ps.iter().map(|p| p.data.len()).sum();
    bytes as f64 * 8.0 * fps[0] as f64 / fps[1] as f64 / ps.len() as f64
}

/// An encoder's packets and parameter sets.
type Stream = (Vec<Packet>, Vec<Vec<u8>>);

/// Encodes `n` frames like the app: this thread produces (`fill(f, inputs)` records and submits the wgpu work that
/// writes frame `f` into every encoder's input; acquire and release are done here), each encoder runs on its own
/// thread and makes an IDR of the frames in `keys` on top of its GOP. Returns the streams and the seconds it took.
fn run(
    g: &Gpu,
    encs: Vec<Encoder>,
    n: u64,
    keys: &[u64],
    mut fill: impl FnMut(u64, &[Arc<Input>]),
) -> (Vec<Stream>, f64) {
    let inputs: Vec<_> = encs.iter().map(|e| e.input.clone()).collect();
    let t0 = Instant::now();
    let r: Vec<Stream> = std::thread::scope(|s| {
        let (txs, hs): (Vec<_>, Vec<_>) = encs
            .into_iter()
            .map(|mut enc| {
                let (tx, rx) = mpsc::channel::<(u64, bool)>();
                let h = s.spawn(move || {
                    let (mut ps, mut p) = (vec![], Packet::default());
                    for (f, key) in rx {
                        if key {
                            enc.input.key();
                        }
                        loop {
                            match enc.encode(g, f as i64) {
                                Err(Error::Full) => {
                                    assert!(enc.packet(&mut p, u64::MAX).unwrap());
                                    ps.push(p.clone());
                                }
                                r => {
                                    assert_eq!(r.unwrap(), f);
                                    break;
                                }
                            }
                        }
                        while enc.packet(&mut p, 0).unwrap() {
                            ps.push(p.clone());
                        }
                    }
                    while enc.packet(&mut p, u64::MAX).unwrap() {
                        ps.push(p.clone());
                    }
                    (ps, enc.params.clone())
                });
                (tx, h)
            })
            .unzip();
        for f in 0..n {
            for i in &inputs {
                let mut tries = 0;
                while let Err(e) = i.acquire(g, f) {
                    assert!(matches!(e, Error::Full) && tries < 400, "acquire {f}: {e}");
                    tries += 1;
                }
            }
            fill(f, &inputs);
            for i in &inputs {
                i.release(g, f);
            }
            for tx in &txs {
                tx.send((f, keys.contains(&f))).unwrap();
            }
        }
        drop(txs);
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let secs = t0.elapsed().as_secs_f64();
    for (ps, _) in &r {
        assert_eq!(ps.len() as u64, n);
        assert!(
            ps.iter()
                .enumerate()
                .all(|(i, p)| p.frame == i as u64 && p.pts == i as i64 && p.dts == p.pts)
        );
    }
    (r, secs)
}

fn decode(g: &Gpu, d: &mut Decoder, au: &[&[u8]]) -> u64 {
    loop {
        match d.decode(g, au) {
            Err(Error::Full) => continue,
            r => return r.unwrap(),
        }
    }
}

/// wgpu copies of the NV12 / P010 planes of the decoder texture `tex`, rect `[x, y, w, h]`, to `buf` at byte `off` of
/// each plane (Y at 0, CbCr at `pitch × rows`), rows `pitch` bytes apart; then back to its home state RESOURCE.
fn stage(
    e: &mut wgpu::CommandEncoder,
    tex: &wgpu::Texture,
    [x, y, w, h]: [u32; 4],
    buf: &wgpu::Buffer,
    [off, pitch, rows]: [u32; 3],
) {
    for (aspect, s, base) in [
        (wgpu::TextureAspect::Plane0, 1, 0),
        (wgpu::TextureAspect::Plane1, 2, pitch * rows),
    ] {
        e.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x / s,
                    y: y / s,
                    z: 0,
                },
                aspect,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: (base + off) as u64,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(rows / s),
                },
            },
            wgpu::Extent3d {
                width: w / s,
                height: h / s,
                depth_or_array_layers: 1,
            },
        );
    }
    let home = wgpu::TextureTransition {
        texture: tex,
        selector: None,
        state: wgpu::TextureUses::RESOURCE,
    };
    e.transition_resources(std::iter::empty(), [home].into_iter());
}

/// The window `[x, y, w, h]` of frame `f` read back through wgpu copies of the texture's plane views: NV12 as I420,
/// P010 as it is (FFmpeg's `p010le`); the caller acquires and releases `f`.
fn read(g: &Gpu, d: &Ring, [x, y, w, h]: [u32; 4], f: u64) -> Vec<u8> {
    let deep = d.texture(f).format() == wgpu::TextureFormat::P010;
    let b = 1 + deep as u32; // bytes per sample
    let pitch = (w * b).next_multiple_of(256);
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (pitch * h * 3 / 2) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = g.device.create_command_encoder(&Default::default());
    stage(&mut e, d.texture(f), [x, y, w, h], &buf, [0, pitch, h]);
    let i = g.queue.submit([e.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(i),
            timeout: None,
        })
        .unwrap();
    let m = buf.slice(..).get_mapped_range().unwrap();
    let (w, h, p, b) = (w as usize, h as usize, pitch as usize, b as usize);
    let mut out = Vec::with_capacity(w * h * 3 / 2 * b);
    for y in 0..h + if deep { h / 2 } else { 0 } {
        out.extend_from_slice(&m[y * p..y * p + w * b]);
    }
    if deep {
        return out;
    }
    let uv = &m[p * h..];
    for c in 0..2 {
        for y in 0..h / 2 {
            out.extend(uv[y * p..y * p + w].iter().skip(c).step_by(2));
        }
    }
    out
}

/// Our decoder on our stream against FFmpeg's decode of the same file, every frame by hash (Main 10: P010 against
/// FFmpeg's `p010le`, the low 6 bits included); returns the frames that differ.
fn cross_check(
    g: &Gpu,
    codec: Codec,
    params: &[Vec<u8>],
    ps: &[Packet],
    file: &Path,
    w: u32,
    h: u32,
) -> usize {
    let mut d = Decoder::new(g, codec, params, w, h).unwrap();
    let deep = d.texture(0).format() == wgpu::TextureFormat::P010;
    let fs = (w * h * 3 / 2) as usize * (1 + deep as usize);
    let pix = if deep { "p010le" } else { "yuv420p" };
    let mut o = ff(&["-i", file.to_str().unwrap()], None, pix);
    let ffh = std::thread::spawn(move || {
        let (mut b, mut v) = (vec![0; fs], vec![]);
        while o.read_exact(&mut b).is_ok() {
            v.push(hash(&b));
        }
        v
    });
    assert_eq!(d.view, [0, 0, w, h]);
    let mut ours = vec![];
    for p in ps {
        let au: Vec<&[u8]> = p.nal().collect();
        let f = decode(g, &mut d, &au);
        d.acquire(g, f);
        ours.push(hash(&read(g, &d, d.view, f)));
        d.release(g, f);
    }
    let theirs = ffh.join().unwrap();
    assert_eq!(
        theirs.len(),
        ps.len(),
        "FFmpeg decoded {} frames",
        theirs.len()
    );
    ours.iter().zip(&theirs).filter(|(a, b)| a != b).count()
}

/// 175 frames of v635 lens 0, decoded by our decoder and copied on the GPU into three encoders at once, each on its
/// own thread: H.265 Main CBR 60 Mbit/s of the 3840x1920 band through the middle of the image circle (rows
/// 960..2880, the most detail; all checks, an extra IDR requested at frame 100 and requests on the scheduled IDRs 0
/// and 60, which add none), the same of the top 3840x1920 (rows 0..1920, PSNR), and VBR 30 / 45 Mbit/s of the
/// middle band (bitrate).
#[test]
fn h265_lens0() {
    let Some(path) = sample_file("v635.insv") else {
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let t = &tracks[0];
    let (w, h, fps, n) = (3840, 1920, [30000, 1001], t.samples.len() as u64);
    let cfg = Cfg {
        codec: Codec::H265,
        deep: false,
        w,
        h,
        fps,
        gop: 60,
        rate: Rate::Cbr(60_000_000),
        vbv: 1000,
        quality: 0,
        full: true, // as the X5 records
        chroma: 0,
    };
    let vbr = Cfg {
        rate: Rate::Vbr(30_000_000, 45_000_000),
        ..cfg
    };
    let encs = [cfg, cfg, vbr].map(|c| Encoder::new(&g, &c).unwrap());
    println!("quality levels {}", encs[0].levels);
    let rects = [[0, 960, w, h], [0, 0, w, h], [0, 960, w, h]];
    let mut d = Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap();
    let mut buf = vec![];
    let (r, secs) = run(&g, encs.into(), n, &[0, 60, 100], |f, inputs| {
        mp4::sample(&mut file, t, f as usize, &mut buf);
        assert_eq!(decode(&g, &mut d, &mp4::nals(&buf, t.nal_len)), f);
        let cbs: Vec<_> = inputs
            .iter()
            .zip(rects)
            .map(|(i, r)| i.copy_texture(&g, d.texture(f), r, [0, 0], f).unwrap())
            .collect();
        d.acquire(&g, f);
        g.queue.submit(cbs);
        d.release(&g, f);
    });
    println!(
        "decode + copy + 3 encodes: {n} frames in {secs:.2} s, {:.1} fps (shared GPU)",
        n as f64 / secs
    );
    let [(ps, params), (tps, _), (vps, _)] = <[_; 3]>::try_from(r).unwrap();
    let (cbr, tbr, vbrr) = (bitrate(&ps, fps), bitrate(&tps, fps), bitrate(&vps, fps));
    println!(
        "CBR 60 Mbit/s: middle {:.2} Mbit/s ({:+.1} %), top {:.2} Mbit/s ({:+.1} %); VBR 30 / 45: {:.2} Mbit/s; \
         middle IDR {} bytes, P mean {} bytes",
        cbr / 1e6,
        (cbr / 60e6 - 1.0) * 100.0,
        tbr / 1e6,
        (tbr / 60e6 - 1.0) * 100.0,
        vbrr / 1e6,
        ps[0].data.len(),
        ps[1..].iter().map(|p| p.data.len()).sum::<usize>() / (ps.len() - 1)
    );
    let files = ["lens0.h265", "lens0_top.h265", "lens0_vbr.h265"].map(out);
    for (f, s) in files.iter().zip([&ps, &tps, &vps]) {
        annexb(f, &params, s);
    }
    expect(
        &files[0],
        &[
            "codec_name=hevc",
            "profile=Main",
            "width=3840",
            "height=1920",
            "color_range=pc",
            "color_space=bt709",
            "color_transfer=bt709",
            "color_primaries=bt709",
            "chroma_location=left",
        ],
    );
    for f in &files {
        let e = ff_errors(f);
        assert!(e.is_empty(), "{}: {e}", f.display());
    }
    let k = keys(&files[0]);
    println!("key frames {k:?}");
    assert_eq!(k, [0, 60, 100, 160]);
    assert!((cbr / 60e6 - 1.0).abs() < 0.1 && (tbr / 60e6 - 1.0).abs() < 0.1);
    assert!((vbrr / 30e6 - 1.0).abs() < 0.1);

    // PSNR(Y) against FFmpeg's decode of the source windows
    let src = ["-i", path.to_str().unwrap(), "-map", "0:v:0"];
    let [mid, top] = std::thread::scope(|s| {
        [(960, &files[0]), (0, &files[1])]
            .map(|(y, f)| s.spawn(move || psnr8(&src, &format!("crop={w}:{h}:0:{y}"), f, w, h)))
            .map(|t| t.join().unwrap())
    });
    assert!(mid.len() as u64 == n && top.len() as u64 == n);
    let ((mm, mmin), (tm, tmin)) = (stats(&mid), stats(&top));
    let steady = stats(&mid[12..]).0;
    println!(
        "PSNR(Y) vs source: middle mean {mm:.2} dB (min {mmin:.2}, from frame 12 on {steady:.2}), top mean {tm:.2} dB \
         (min {tmin:.2})"
    );
    // RADV's encoder (FFmpeg's hevc_vulkan gives the same values per frame) reaches about 39.5 dB on the middle band
    // at this rate; the top band, a quarter of it outside the image circle, clears 40 dB
    assert!(tm > 40.0 && mm > 38.5 && mmin > 33.0);

    let bad = cross_check(&g, Codec::H265, &params, &ps, &files[0], w, h);
    println!("our decoder vs FFmpeg on our H.265: {bad} of {n} frames differ");
    assert_eq!(bad, 0);
    let m = out("lens0.mp4");
    mux(&m, Codec::H265, &params, &ps, w, h, fps);
    check_mp4(&m, n as usize, fps);
}

/// Synthetic 10-bit picture `f`: a diagonal luma ramp over the limited range (0.15 codes per pixel: 8 bits would
/// band) that moves 4 pixels per frame, chroma ramps, and ±3 codes of noise on luma; P010 (Y, then interleaved
/// CbCr), samples in the top 10 bits.
fn p010(f: u64, w: usize, h: usize, o: &mut [u16]) {
    let mut s = f.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let (yp, c) = o.split_at_mut(w * h);
    for (i, v) in yp.iter_mut().enumerate() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let r = 64 + ((i % w + i / w + 4 * f as usize) % (w + h)) * 876 / (w + h);
        *v = ((r as i64 + (s % 7) as i64 - 3).clamp(64, 940) as u16) << 6;
    }
    for (i, v) in c.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let (x, y) = (i % (w / 2), i / (w / 2));
        v[0] = ((64 + x * 896 / (w / 2)) as u16) << 6;
        v[1] = ((64 + y * 896 / (h / 2)) as u16) << 6;
    }
}

/// H.265 Main 10 from synthetic P010 pictures that wgpu copies into the encoder's input textures through their plane
/// aspects: limited range, centre chroma siting; ffprobe's profile, pixel format and colour tags, a clean FFmpeg
/// decode, PSNR(Y) against the source, our decoder (P010) against FFmpeg, and the MP4. A second encoder at constant
/// QP 24 (clean decode, PSNR, our decoder).
#[test]
fn h265_main10() {
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (w, h, n, fps) = (3840u32, 1920u32, 60u64, [30, 1]);
    let cfg = Cfg {
        codec: Codec::H265,
        deep: true,
        w,
        h,
        fps,
        gop: 30,
        rate: Rate::Cbr(40_000_000),
        vbv: 1000,
        quality: 0,
        full: false,
        chroma: 1,
    };
    let qp = Cfg {
        rate: Rate::Qp(24),
        ..cfg
    };
    let encs = vec![
        Encoder::new(&g, &cfg).unwrap(),
        Encoder::new(&g, &qp).unwrap(),
    ];
    let pitch = w * 2; // both planes; a multiple of 256 as wgpu's copies want
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (pitch * h * 3 / 2) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut pic = vec![0u16; (w * h * 3 / 2) as usize];
    let (r, secs) = run(&g, encs, n, &[], |f, inputs| {
        p010(f, w as usize, h as usize, &mut pic);
        let b = unsafe { std::slice::from_raw_parts(pic.as_ptr() as *const u8, pic.len() * 2) };
        g.queue.write_buffer(&buf, 0, b);
        let mut e = g.device.create_command_encoder(&Default::default());
        for (i, (aspect, s, off)) in inputs.iter().flat_map(|i| {
            [
                (wgpu::TextureAspect::Plane0, 1, 0),
                (wgpu::TextureAspect::Plane1, 2, pitch * h),
            ]
            .map(|p| (i, p))
        }) {
            e.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: off as u64,
                        bytes_per_row: Some(pitch),
                        rows_per_image: Some(h / s),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: i.texture(f),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect,
                },
                wgpu::Extent3d {
                    width: w / s,
                    height: h / s,
                    depth_or_array_layers: 1,
                },
            );
        }
        g.queue.submit([e.finish()]);
    });
    let (ps, params) = &r[0];
    let (br, qbr) = (bitrate(ps, fps), bitrate(&r[1].0, fps));
    println!(
        "Main 10 {w}x{h}: {n} frames in {secs:.2} s (CPU-generated, uploaded, 2 encoders); CBR 40 Mbit/s: {:.2} \
         Mbit/s ({:+.1} %); QP 24: {:.2} Mbit/s",
        br / 1e6,
        (br / 40e6 - 1.0) * 100.0,
        qbr / 1e6
    );
    let f = out("main10.h265");
    annexb(&f, params, ps);
    let fq = out("main10_qp.h265");
    annexb(&fq, &r[1].1, &r[1].0);
    expect(
        &f,
        &[
            "codec_name=hevc",
            "profile=Main 10",
            "width=3840",
            "height=1920",
            "pix_fmt=yuv420p10le",
            "color_range=tv",
            "color_space=bt709",
            "color_transfer=bt709",
            "color_primaries=bt709",
            "chroma_location=center",
        ],
    );
    for f in [&f, &fq] {
        let e = ff_errors(f);
        assert!(e.is_empty(), "{e}");
    }
    assert!((br / 40e6 - 1.0).abs() < 0.1);
    let (ny, fs) = ((w * h) as usize, (w * h * 3) as usize); // 16-bit samples
    for (what, f) in [("CBR", &f), ("QP 24", &fq)] {
        let mut o = ff(&["-i", f.to_str().unwrap()], None, "yuv420p10le");
        let (mut b, mut v) = (vec![0u8; fs], vec![]);
        for i in 0..n {
            o.read_exact(&mut b).unwrap();
            p010(i, w as usize, h as usize, &mut pic);
            let se: u64 = pic[..ny]
                .iter()
                .zip(b.as_chunks::<2>().0)
                .map(|(&s, &d)| ((s >> 6) as i64 - u16::from_le_bytes(d) as i64).pow(2) as u64)
                .sum();
            v.push(psnr(se as f64, ny, 1023.0));
        }
        let (mean, min) = stats(&v);
        println!("Main 10 {what} PSNR(Y) vs source: mean {mean:.2} dB, min {min:.2} dB");
        assert!(min > 45.0);
    }
    for (what, (ps, params), f) in [("CBR", &r[0], &f), ("QP 24", &r[1], &fq)] {
        let bad = cross_check(&g, Codec::H265, params, ps, f, w, h);
        println!("our decoder vs FFmpeg on our Main 10 {what}: {bad} of {n} frames differ");
        assert_eq!(bad, 0);
    }
    let m = out("main10.mp4");
    mux(&m, Codec::H265, params, ps, w, h, fps);
    check_mp4(&m, n as usize, fps);
}

/// H.264 High 1920x1080 CBR 8 Mbit/s from a window of the X5's live preview (`X5_LIVE`, 2656x1328 at 59.94 fps, a
/// stitched and downscaled equirect), decoded by our decoder and copied on the GPU; 1080 rows are coded as 1088 and
/// cropped. An extra IDR at frame 50 and requests on the scheduled IDRs 0 and 110, which add none. ffprobe, FFmpeg,
/// PSNR, our decoder against FFmpeg, MP4. A second encoder with a 100 ms buffer (RTMP, low latency): a smaller
/// first IDR, the rate right from the first frame. In both, picture data carries the rate, not filler NAL units.
#[test]
fn h264_live() {
    let Some(path) = std::env::var_os("X5_LIVE") else {
        println!("X5_LIVE unset: skipped");
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let es = std::fs::read(&path).unwrap();
    let aus = aus264(&es);
    let sp: Vec<_> = aus[0]
        .iter()
        .copied()
        .filter(|n| matches!(h264::nal_type(n), 7 | 8))
        .collect();
    let mut d = Decoder::new(&g, Codec::H264, &sp, 0, 0).unwrap();
    let [_, _, sw, sh] = d.view;
    let (w, h, fps, n) = (1920, 1080, [60000, 1001], aus.len() as u64);
    let (x, y) = (((sw - w) / 2) & !1, ((sh - h) / 2) & !1);
    let cfg = Cfg {
        codec: Codec::H264,
        deep: false,
        w,
        h,
        fps,
        gop: 60,
        rate: Rate::Cbr(8_000_000),
        vbv: 1000,
        quality: 0,
        full: true,
        chroma: 0,
    };
    let low = Cfg { vbv: 100, ..cfg };
    let encs = [cfg, low].map(|c| Encoder::new(&g, &c).unwrap());
    let (r, secs) = run(&g, encs.into(), n, &[0, 50, 110], |f, inputs| {
        assert_eq!(decode(&g, &mut d, &aus[f as usize]), f);
        let cbs: Vec<_> = inputs
            .iter()
            .map(|i| {
                i.copy_texture(&g, d.texture(f), [x, y, w, h], [0, 0], f)
                    .unwrap()
            })
            .collect();
        d.acquire(&g, f);
        g.queue.submit(cbs);
        d.release(&g, f);
    });
    let (ps, params) = &r[0];
    // RADV starts with a QP that ignores the target (the first IDR comes out near 700 kB at 8 Mbit/s with a buffer of
    // 250 ms or more); 2.45 s are too short to pay that back, so the rate is checked from frame 1 on
    let (br, br1) = (bitrate(ps, fps), bitrate(&ps[1..], fps));
    println!(
        "H.264 {w}x{h} from {sw}x{sh} at ({x}, {y}): {n} frames in {secs:.2} s (decode + copy + 2 encodes); {:.2} Mbit/s \
         ({:+.1} %), without the first IDR ({} bytes) {:.2} Mbit/s ({:+.1} %)",
        br / 1e6,
        (br / 8e6 - 1.0) * 100.0,
        ps[0].data.len(),
        br1 / 1e6,
        (br1 / 8e6 - 1.0) * 100.0
    );
    let f = out("live.h264");
    annexb(&f, params, ps);
    expect(
        &f,
        &[
            "codec_name=h264",
            "profile=High",
            "width=1920",
            "height=1080",
            "color_range=pc",
            "color_space=bt709",
            "color_transfer=bt709",
            "color_primaries=bt709",
            "chroma_location=left",
        ],
    );
    let fl = out("live_low.h264");
    annexb(&fl, &r[1].1, &r[1].0);
    for f in [&f, &fl] {
        let e = ff_errors(f);
        assert!(e.is_empty(), "{e}");
    }
    let k = keys(&f);
    println!("key frames {k:?}");
    assert_eq!(k, [0, 50, 110]);
    assert_eq!(keys(&fl), k);
    assert!((br1 / 8e6 - 1.0).abs() < 0.1);
    let lp = &r[1].0;
    let lbr = bitrate(lp, fps);
    // filler (NAL type 12) share: CBR pads with it when pictures come out below the rate
    let fill = |ps: &[Packet]| {
        let n: usize = ps
            .iter()
            .flat_map(|p| p.nal())
            .filter(|n| n[0] & 31 == 12)
            .map(|n| n.len())
            .sum();
        n as f64 / ps.iter().map(|p| p.data.len()).sum::<usize>() as f64
    };
    let (fl0, fl1) = (fill(ps), fill(lp));
    println!(
        "100 ms buffer: {:.2} Mbit/s ({:+.1} %) with the first IDR ({} bytes); filler {:.2} % (1000 ms: {:.2} %)",
        lbr / 1e6,
        (lbr / 8e6 - 1.0) * 100.0,
        lp[0].data.len(),
        fl1 * 100.0,
        fl0 * 100.0
    );
    assert!((lbr / 8e6 - 1.0).abs() < 0.1 && lp[0].data.len() < ps[0].data.len());
    assert!(fl0 < 0.01 && fl1 < 0.01);
    let lv = psnr8(
        &["-i", path.to_str().unwrap()],
        &format!("crop={w}:{h}:{x}:{y}"),
        &fl,
        w,
        h,
    );
    let (lmean, lmin) = stats(&lv);
    println!("100 ms buffer PSNR(Y) vs source: mean {lmean:.2} dB, min {lmin:.2} dB");
    let v = psnr8(
        &["-i", path.to_str().unwrap()],
        &format!("crop={w}:{h}:{x}:{y}"),
        &f,
        w,
        h,
    );
    assert_eq!(v.len() as u64, n);
    let (mean, min) = stats(&v);
    println!("H.264 PSNR(Y) vs source: mean {mean:.2} dB, min {min:.2} dB");
    assert!(mean > 35.0);
    let bad = cross_check(&g, Codec::H264, params, ps, &f, w, h);
    println!("our decoder vs FFmpeg on our H.264: {bad} of {n} frames differ");
    assert_eq!(bad, 0);
    let m = out("live.mp4");
    mux(&m, Codec::H264, params, ps, w, h, fps);
    check_mp4(&m, n as usize, fps);
}

/// Every size the app encodes, CBR, from `M` frames of v635 (lens 0 | lens 1 side by side, the middle cut out)
/// staged in wgpu buffers and copied into the encoder's input with `Input::copy_buffer` (row pitch rounded up to
/// 256 bytes, so not the width for 1920 and 5760): frames per second of copy + encode (shared GPU), ffprobe's
/// size and profile, a clean FFmpeg decode. The frames repeat every `M`, which costs a jump in content there.
#[test]
fn sizes() {
    let Some(path) = sample_file("v635.insv") else {
        return;
    };
    if !ffmpeg() {
        return;
    }
    const M: u64 = 8;
    const N: u64 = 240;
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let fps = [30000, 1001];
    let (h5, h4) = (Codec::H265, Codec::H264);
    let sizes: [(u32, u32, &[_]); 6] = [
        (7680, 3840, &[(h5, 120_000_000)]),
        (5760, 2880, &[(h5, 100_000_000)]),
        (3840, 1920, &[(h5, 60_000_000), (h4, 60_000_000)]),
        (2560, 1440, &[(h5, 16_000_000), (h4, 16_000_000)]),
        (1920, 1080, &[(h5, 8_000_000), (h4, 8_000_000)]),
        // coded 1920x1080 / 1920x1088 and cropped: our decoder must agree with FFmpeg on the window
        (1916, 1076, &[(h5, 8_000_000), (h4, 8_000_000)]),
    ];
    let mut au = vec![];
    for (w, h, cases) in sizes {
        let pitch = w.next_multiple_of(256);
        let bufs: Vec<_> = (0..M)
            .map(|_| {
                g.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (pitch * h * 3 / 2) as u64,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mut ds: Vec<_> = tracks[..2]
            .iter()
            .map(|t| Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap())
            .collect();
        // the window [x0, x0 + w) of the 7680 wide canvas: lens 0's part, then lens 1's
        let (x0, y0) = ((7680 - w) / 2, (3840 - h) / 2);
        let a = (3840 - x0).min(w);
        for f in 0..M {
            let mut e = g.device.create_command_encoder(&Default::default());
            for (i, d) in ds.iter_mut().enumerate() {
                mp4::sample(&mut file, &tracks[i], f as usize, &mut au);
                assert_eq!(decode(&g, d, &mp4::nals(&au, tracks[i].nal_len)), f);
                let (r, off) = match i {
                    0 => ([x0, y0, a, h], 0),
                    _ => ([0, y0, w - a, h], a),
                };
                if r[2] > 0 {
                    stage(&mut e, d.texture(f), r, &bufs[f as usize], [off, pitch, h]);
                }
            }
            ds.iter().for_each(|d| d.acquire(&g, f));
            g.queue.submit([e.finish()]);
            ds.iter().for_each(|d| d.release(&g, f));
        }
        drop(ds);
        for &(codec, rate) in cases {
            let cfg = Cfg {
                codec,
                deep: false,
                w,
                h,
                fps,
                gop: 60,
                rate: Rate::Cbr(rate),
                vbv: 1000,
                quality: 0,
                full: true,
                chroma: 0,
            };
            let enc = Encoder::new(&g, &cfg).unwrap();
            let (r, secs) = run(&g, vec![enc], N, &[], |f, inputs| {
                let cb = inputs[0]
                    .copy_buffer(&g, &bufs[(f % M) as usize], 0, pitch, f)
                    .unwrap();
                g.queue.submit([cb]);
            });
            let (ps, params) = &r[0];
            let (name, ext, prof) = match codec {
                Codec::H265 => ("H.265", "h265", "Main"),
                Codec::H264 => ("H.264", "h264", "High"),
            };
            println!(
                "{name} {w}x{h} CBR {} Mbit/s: {N} frames in {secs:.2} s, {:.1} fps (copy + encode, shared GPU); \
                 {:.1} Mbit/s",
                rate / 1_000_000,
                N as f64 / secs,
                bitrate(ps, fps) / 1e6
            );
            let f = out(&format!("size_{w}x{h}.{ext}"));
            annexb(&f, params, ps);
            let (ws, hs, pr) = (
                format!("width={w}"),
                format!("height={h}"),
                format!("profile={prof}"),
            );
            expect(
                &f,
                &[&ws, &hs, &pr, "color_range=pc", "color_primaries=bt709"],
            );
            let e = ff_errors(&f);
            assert!(e.is_empty(), "{e}");
            if w % 16 != 0 {
                let bad = cross_check(&g, codec, params, ps, &f, w, h);
                println!("our decoder vs FFmpeg: {bad} of {N} frames differ");
                assert_eq!(bad, 0);
            }
        }
    }
}

/// CBR against the rate control buffer: 30 s of v635 lens 0's middle 1920x1080 window (its 175 frames looped) at
/// 1080p30 6 Mbit/s, H.264 and H.265, buffers of 100 ms (the app's RTMP setting), 250 ms and 1000 ms (files). From
/// 1000 ms the driver's rate control holds the target over the whole stream, its first IDR (RADV starts at QP 25)
/// paid back. With less, RADV (Mesa 26.2.5) runs the P-frames over their share and never pays the excess back, so the
/// stream breaks the leaky bucket it was given: +4 to +10 % at 100 ms (3 to 12 Mbit/s, this footage; each IDR capped
/// at about the buffer), up to +2.4 % at 250 ms. The initial occupancy (0 to 100 %), the GOP flags and sizes and the
/// GOP-remaining hints leave the frame sizes as they are, to the byte: RADV ignores them.
#[test]
fn cbr() {
    let Some(path) = sample_file("v635.insv") else {
        return;
    };
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let t = &tracks[0];
    let (w, h, fps, n, rate) = (1920, 1080, [30, 1], 900, 6_000_000);
    let base = Cfg {
        codec: Codec::H264,
        deep: false,
        w,
        h,
        fps,
        gop: 60,
        rate: Rate::Cbr(rate),
        vbv: 100,
        quality: 0,
        full: true,
        chroma: 0,
    };
    let cfgs: Vec<_> = [Codec::H264, Codec::H265]
        .into_iter()
        .flat_map(|codec| [100, 250, 1000].map(|vbv| Cfg { codec, vbv, ..base }))
        .collect();
    let encs = cfgs.iter().map(|c| Encoder::new(&g, c).unwrap()).collect();
    let mut d = Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap();
    let (mut buf, m) = (vec![], t.samples.len() as u64);
    let (r, _) = run(&g, encs, n, &[], |f, inputs| {
        // sample 0 is an IDR: the loop restarts the decoder's picture management
        mp4::sample(&mut file, t, (f % m) as usize, &mut buf);
        let k = decode(&g, &mut d, &mp4::nals(&buf, t.nal_len));
        let cbs: Vec<_> = inputs
            .iter()
            .map(|i| {
                i.copy_texture(&g, d.texture(k), [960, 1380, w, h], [0, 0], f)
                    .unwrap()
            })
            .collect();
        d.acquire(&g, k);
        g.queue.submit(cbs);
        d.release(&g, k);
    });
    let pc = |ps: &[Packet]| (bitrate(ps, fps) / rate as f64 - 1.0) * 100.0;
    for (c, (ps, _)) in cfgs.iter().zip(&r) {
        let mean = |key: bool| {
            let v: Vec<_> = ps
                .iter()
                .filter(|p| p.key == key)
                .map(|p| p.data.len())
                .collect();
            v.iter().sum::<usize>() / v.len()
        };
        println!(
            "{:?} CBR 6 Mbit/s, {} ms buffer: {:+.2} % over 30 s, {:+.2} % from frame 60; IDR mean {} B, P mean {} B \
             (share {} B)",
            c.codec,
            c.vbv,
            pc(ps),
            pc(&ps[60..]),
            mean(true),
            mean(false),
            rate / 8 / 30
        );
        if c.vbv == 1000 {
            assert!(pc(ps).abs() < 2.0);
        }
    }
}

/// What the hardware can't do comes back as an error from `Encoder::new`.
#[test]
fn refuse() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let ok = Cfg {
        codec: Codec::H264,
        deep: false,
        w: 1920,
        h: 1080,
        fps: [30, 1],
        gop: 30,
        rate: Rate::Cbr(8_000_000),
        vbv: 1000,
        quality: 0,
        full: false,
        chroma: 0,
    };
    assert!(Encoder::new(&g, &ok).is_ok());
    let h5 = Codec::H265;
    for (what, c) in [
        (
            "H.264 7680x3840",
            Cfg {
                w: 7680,
                h: 3840,
                ..ok
            },
        ),
        (
            "H.264 5760x2880",
            Cfg {
                w: 5760,
                h: 2880,
                ..ok
            },
        ),
        ("H.264 10-bit", Cfg { deep: true, ..ok }),
        (
            "H.265 16384x8192",
            Cfg {
                codec: h5,
                w: 16384,
                h: 8192,
                ..ok
            },
        ),
        ("odd width", Cfg { w: 1921, ..ok }),
        ("no rate control buffer", Cfg { vbv: 0, ..ok }),
        (
            "QP 99",
            Cfg {
                rate: Rate::Qp(99),
                ..ok
            },
        ),
        (
            "10 Tbit/s",
            Cfg {
                rate: Rate::Cbr(10_000_000_000_000),
                ..ok
            },
        ),
    ] {
        match Encoder::new(&g, &c) {
            Err(e) => println!("{what}: {e}"),
            Ok(_) => panic!("{what} accepted"),
        }
    }
}

/// The highest quality level asked for (clamped to `Encoder::levels`) on synthetic 1280x720 pictures (a moving ramp
/// with noise), H.265 CBR: FFmpeg decodes it clean. RADV's top level, corrupt for H.265 under rate control, is not
/// among the levels there.
#[test]
fn quality() {
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (w, h, n) = (1280u32, 720u32, 60u64);
    let cfg = Cfg {
        codec: Codec::H265,
        deep: false,
        w,
        h,
        fps: [30, 1],
        gop: 30,
        rate: Rate::Cbr(4_000_000),
        vbv: 1000,
        quality: u32::MAX,
        full: false,
        chroma: 0,
    };
    let enc = Encoder::new(&g, &cfg).unwrap();
    let lv = [
        Cfg {
            rate: Rate::Qp(28),
            ..cfg
        },
        Cfg {
            codec: Codec::H264,
            ..cfg
        },
    ]
    .map(|c| Encoder::new(&g, &c).unwrap().levels);
    let driver = g.adapter.get_info().driver;
    println!(
        "{driver}: quality levels H.265 CBR {}, H.265 QP {}, H.264 CBR {}",
        enc.levels, lv[0], lv[1]
    );
    if driver == "radv" {
        assert!(enc.levels + 1 == lv[0] && lv[0] == lv[1]);
    }
    let pitch = w.next_multiple_of(256);
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (pitch * h * 3 / 2) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let (mut pic, mut s) = (vec![128u8; (pitch * h * 3 / 2) as usize], 1u64);
    let (r, _) = run(&g, vec![enc], n, &[], |f, inputs| {
        for (i, v) in pic[..(pitch * h) as usize].iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let (x, y) = (i % pitch as usize, i / pitch as usize);
            *v = ((x + y + 3 * f as usize) % 200 + 16 + (s % 5) as usize) as u8;
        }
        g.queue.write_buffer(&buf, 0, &pic);
        g.queue
            .submit([inputs[0].copy_buffer(&g, &buf, 0, pitch, f).unwrap()]);
    });
    let f = out("quality.h265");
    annexb(&f, &r[0].1, &r[0].0);
    let e = ff_errors(&f);
    assert!(e.is_empty(), "{e}");
}
