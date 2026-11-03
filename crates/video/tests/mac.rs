//! The VideoToolbox backend: synthetic pictures through the encoder's input (`Input::copy_buffer`), VideoToolbox
//! encode, our decoder, and back, for H.264, H.265 and H.265 Main 10, with a file's settings and RTMP's; the streams
//! are also checked with `ffprobe` when it is installed (a test tool only). Key frames on request, CBR's rate, a new
//! session for an invalidated one, hardware encoders for the export presets.
#![cfg(target_os = "macos")]

use std::{
    process::Command,
    sync::Mutex,
    time::{Duration, Instant},
};
use video::{Cfg, Codec, Decoder, Encoder, Error, Gpu, Packet, Rate, Ring, h264, h265};

/// GPU tests run one at a time: each opens its own device.
static GPU: Mutex<()> = Mutex::new(());

const W: u32 = 640;
const H: u32 = 360;
const N: u64 = 24;

/// Sample `(x, y)` of plane `p` (0 Y, 1 Cb, 2 Cr) of frame `f`, 8-bit: smooth gradients that move with `f`, so the
/// encoder has motion to code and the decode has something to match.
fn px(p: usize, x: u32, y: u32, f: u64) -> u8 {
    let f = f as u32;
    match p {
        0 => (16 + (x + 2 * f) * 200 / W + (y * 20 / H)) as u8,
        1 => (64 + (y + f) * 128 / H) as u8,
        _ => (192 - x * 128 / W) as u8,
    }
}

/// Frame `f` as NV12 or P010 (the sample in the high 10 bits) with rows `pitch` bytes apart.
fn picture(f: u64, deep: bool, pitch: u32) -> Vec<u8> {
    let b = 1 + deep as usize;
    let pitch = pitch as usize;
    let mut o = vec![0u8; pitch * (H + H / 2) as usize];
    let put = |o: &mut [u8], at: usize, v: u8| match deep {
        true => o[at..at + 2].copy_from_slice(&(((v as u16) << 2) << 6).to_le_bytes()),
        false => o[at] = v,
    };
    for y in 0..H {
        for x in 0..W {
            put(&mut o, y as usize * pitch + x as usize * b, px(0, x, y, f));
        }
    }
    for y in 0..H / 2 {
        for x in 0..W / 2 {
            let at = (H + y) as usize * pitch + x as usize * 2 * b;
            put(&mut o, at, px(1, 2 * x, 2 * y, f));
            put(&mut o, at + b, px(2, 2 * x, 2 * y, f));
        }
    }
    o
}

/// Frame `f` of a busier picture for the rate test, NV12 with rows `W` apart, grey chroma: value noise of 8 px cells
/// moving by (2, 1) px a frame, and every 16th row fresh noise, so the P-frames cost bits too.
fn busy(f: u64) -> Vec<u8> {
    let hash = |x: u32, y: u32| {
        let mut h = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca77);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        (h ^ h >> 12) & 255
    };
    let noise = |x: u32, y: u32| {
        let (cx, fx, fy) = (x / 8, x % 8, y % 8);
        let row = |cy| hash(cx, cy) * (8 - fx) + hash(cx + 1, cy) * fx;
        (row(y / 8) * (8 - fy) + row(y / 8 + 1) * fy) / 64
    };
    let f = f as u32;
    let mut o = vec![128u8; (W * (H + H / 2)) as usize];
    for y in 0..H {
        for x in 0..W {
            let v = match y % 16 {
                0 => hash(x + 7919 * f, y),
                _ => noise(x + 2 * f, y + f),
            };
            o[(y * W + x) as usize] = (16 + v * 219 / 255) as u8;
        }
    }
    o
}

/// The luma plane of frame `f` of `ring`, 8-bit (10-bit rounded), through a copy of its plane view.
fn luma(g: &Gpu, ring: &Ring, [x, y, w, h]: [u32; 4], f: u64) -> Vec<u8> {
    let b = 1 + ring.deep() as u32;
    let bpr = (w * b).next_multiple_of(256);
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (bpr * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = g.device.create_command_encoder(&Default::default());
    e.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            origin: wgpu::Origin3d { x, y, z: 0 },
            ..ring.plane(f, 0)
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    ring.acquire(g, f);
    let i = g.queue.submit([e.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(i),
            timeout: None,
        })
        .unwrap();
    let m = buf.slice(..).get_mapped_range().unwrap();
    (0..h as usize)
        .flat_map(|r| {
            let row = &m[r * bpr as usize..][..(w * b) as usize];
            match b {
                2 => row
                    .chunks(2)
                    .map(|s| ((u16::from_le_bytes([s[0], s[1]]) >> 6) as f32 / 4.0).round() as u8)
                    .collect::<Vec<_>>(),
                _ => row.to_vec(),
            }
        })
        .collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

/// `W` x `H` at 30 fps, an IDR every 10 frames; `vbv` 1000 is a file's settings, 100 RTMP's (low latency).
fn cfg(codec: Codec, deep: bool, rate: Rate, vbv: u32) -> Cfg {
    Cfg {
        codec,
        deep,
        w: W,
        h: H,
        fps: [30, 1],
        gop: 10,
        rate,
        vbv,
        quality: 0,
        full: false,
        chroma: 0,
    }
}

/// Encodes frames `0..n` of `pic` (rows `pitch` bytes apart) the export's way: after each `encode` the packets that
/// are out (`Full`: one, waiting for it), then the rest, each within 1 s though it may wait a minute (a file's
/// session is made to emit what it holds back). `at(enc, f)` runs before frame `f` is encoded. Every frame comes
/// back, in order, with its pts.
fn encode(
    g: &Gpu,
    enc: &mut Encoder,
    n: u64,
    pitch: u32,
    pic: impl Fn(u64) -> Vec<u8>,
    at: impl Fn(&Encoder, u64),
) -> Vec<Packet> {
    let h = enc.input.h;
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (pitch * (h + h / 2)) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let (mut p, mut out) = (Packet::default(), vec![]);
    let one = |enc: &mut Encoder, p: &mut Packet, out: &mut Vec<Packet>| {
        if enc.packet(p, 50_000_000).unwrap() {
            out.push(p.clone());
        }
    };
    for f in 0..n {
        loop {
            match enc.input.acquire(g, f) {
                Ok(()) => break,
                Err(Error::Full) => one(enc, &mut p, &mut out),
                Err(e) => panic!("{e}"),
            }
        }
        g.queue.write_buffer(&buf, 0, &pic(f));
        let cb = enc.input.copy_buffer(g, &buf, 0, pitch, f).unwrap();
        g.queue.submit([cb]);
        enc.input.release(g, f);
        at(enc, f);
        loop {
            match enc.encode(g, f as i64 * 1000) {
                Ok(k) => break assert_eq!(k, f),
                Err(Error::Full) => one(enc, &mut p, &mut out),
                Err(e) => panic!("{e}"),
            }
        }
        while enc.packet(&mut p, 0).unwrap() {
            out.push(p.clone());
        }
    }
    loop {
        let t = Instant::now();
        if !enc.packet(&mut p, 60_000_000_000).unwrap() {
            break;
        }
        let d = t.elapsed();
        assert!(d < Duration::from_secs(1), "frame {} after {d:?}", p.frame);
        out.push(p.clone());
    }
    assert_eq!(out.len() as u64, n);
    for (f, p) in (0..).zip(&out) {
        assert_eq!(
            (p.frame, p.pts, p.dts),
            (f, f as i64 * 1000, f as i64 * 1000)
        );
    }
    out
}

/// The key frames; each packet is one exactly when it holds an IDR slice.
fn keys(codec: Codec, ps: &[Packet]) -> Vec<u64> {
    for p in ps {
        let idr = p.nal().any(|n| match codec {
            Codec::H264 => h264::nal_type(n) == 5,
            Codec::H265 => matches!(h265::nal_type(n), 19 | 20),
        });
        assert_eq!(p.key, idr, "frame {}", p.frame);
    }
    ps.iter().filter(|p| p.key).map(|p| p.frame).collect()
}

/// Decodes the packets that carry data (a dropped or lost frame has none); returns the lowest luma PSNR against
/// the frames `px` draws.
fn decode(g: &Gpu, codec: Codec, params: &[Vec<u8>], ps: &[Packet]) -> f64 {
    let mut dec = Decoder::new(g, codec, params, W, H).unwrap();
    let mut worst = f64::MAX;
    for p in ps.iter().filter(|p| !p.data.is_empty()) {
        let f = dec.decode(g, &p.nal().collect::<Vec<_>>()).unwrap();
        let want: Vec<u8> = (0..H)
            .flat_map(|y| (0..W).map(move |x| px(0, x, y, p.frame)))
            .collect();
        worst = worst.min(psnr(&luma(g, &dec, dec.view, f), &want));
        dec.release(g, f);
    }
    worst
}

/// Annex B with the parameter sets first.
fn stream(params: &[Vec<u8>], ps: &[Packet]) -> Vec<u8> {
    let mut es: Vec<u8> = params
        .iter()
        .flat_map(|p| [&[0, 0, 0, 1][..], p].concat())
        .collect();
    for p in ps {
        es.extend_from_slice(&p.data);
    }
    es
}

/// Encodes `N` frames, with an extra IDR on request at 5 (the GOP restarts there) and one at 15 (its scheduled IDR:
/// it adds none), and decodes them again; returns the stream and the lowest luma PSNR.
fn round_trip(codec: Codec, deep: bool, rate: Rate, vbv: u32) -> (Vec<u8>, f64) {
    let g = Gpu::new().unwrap();
    let mut enc = Encoder::new(&g, &cfg(codec, deep, rate, vbv)).unwrap();
    assert!(!enc.params.is_empty(), "no parameter sets");
    let pitch = (W * (1 + deep as u32)).next_multiple_of(4) + 8; // not a multiple of 256 on purpose
    let ps = encode(
        &g,
        &mut enc,
        N,
        pitch,
        |f| picture(f, deep, pitch),
        |e, f| {
            if [5, 15].contains(&f) {
                e.input.key()
            }
        },
    );
    assert!(ps.iter().all(|p| !p.data.is_empty()), "a frame dropped");
    assert_eq!(
        keys(codec, &ps),
        [0, 5, 15],
        "IDRs where the GOP and the requests put them"
    );
    let worst = decode(&g, codec, &enc.params, &ps);
    (stream(&enc.params, &ps), worst)
}

/// Checks the stream with ffprobe: no error, `n` frames.
fn probe(es: &[u8], codec: Codec, name: &str, n: u64) {
    let dir = std::env::temp_dir().join(format!("x5desk-mac-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(format!(
        "{name}.{}",
        if codec == Codec::H264 { "h264" } else { "h265" }
    ));
    std::fs::write(&p, es).unwrap();
    let o = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=codec_name,profile,width,height,pix_fmt,nb_read_frames",
        ])
        .args(["-of", "csv=p=0"])
        .arg(&p)
        .output();
    let _ = std::fs::remove_dir_all(&dir);
    let Ok(o) = o else {
        println!("no ffprobe: {name} not probed");
        return;
    };
    let s = String::from_utf8_lossy(&o.stdout);
    println!("{name}: ffprobe {}", s.trim());
    assert!(
        o.status.success() && o.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(s.trim().ends_with(&format!(",{n}")), "{s}");
}

#[test]
fn h264() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (es, db) = round_trip(Codec::H264, false, Rate::Vbr(4_000_000, 8_000_000), 1000);
    println!("H.264: luma PSNR ≥ {db:.1} dB");
    assert!(db > 35.0, "{db}");
    probe(&es, Codec::H264, "h264", N);
}

#[test]
fn h265() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (es, db) = round_trip(Codec::H265, false, Rate::Cbr(4_000_000), 1000);
    println!("H.265: luma PSNR ≥ {db:.1} dB");
    assert!(db > 35.0, "{db}");
    probe(&es, Codec::H265, "h265", N);
}

#[test]
fn main10() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (es, db) = round_trip(Codec::H265, true, Rate::Qp(22), 1000);
    println!("H.265 Main 10: luma PSNR ≥ {db:.1} dB");
    assert!(db > 35.0, "{db}");
    probe(&es, Codec::H265, "main10", N);
}

/// RTMP's settings (H.264 CBR, a 100 ms buffer: real time, no frame held back).
#[test]
fn live() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (es, db) = round_trip(Codec::H264, false, Rate::Cbr(4_000_000), 100);
    println!("H.264 low latency: luma PSNR ≥ {db:.1} dB");
    assert!(db > 35.0, "{db}");
    probe(&es, Codec::H264, "live", N);
}

/// H.264 CBR at 2 Mbit/s on busier pictures, with RTMP's 100 ms buffer and a file's 1 s: the rate from frame 1 on
/// within 10 % of the target (a DataRateLimits of one buffer came out 30 % under), no filler NAL units
/// (VideoToolbox's ConstantBitRate pads). VideoToolbox caps no IDR at the buffer (here 1.4 buffers of 100 ms, the
/// largest 100 ms 3.9 either way), so the short buffer's largest 100 ms is only held to the long one's.
#[test]
fn cbr() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    const BPS: f64 = 2e6;
    let most = [100, 1000].map(|vbv| {
        let c = Cfg {
            gop: 60,
            ..cfg(Codec::H264, false, Rate::Cbr(BPS as u64), vbv)
        };
        let mut enc = Encoder::new(&g, &c).unwrap();
        let ps = encode(&g, &mut enc, 150, W, busy, |_, _| {});
        assert!(ps.iter().all(|p| !p.data.is_empty()), "a frame dropped");
        assert_eq!(keys(Codec::H264, &ps), [0, 60, 120]);
        let rate = |ps: &[Packet]| {
            ps.iter().map(|p| p.data.len()).sum::<usize>() as f64 * 8.0 * 30.0 / ps.len() as f64
        };
        let (br, br1) = (rate(&ps), rate(&ps[1..]));
        // the largest 100 ms (3 frames), in 100 ms of the rate
        let most = (ps.windows(3))
            .map(|w| w.iter().map(|p| p.data.len()).sum::<usize>())
            .max()
            .unwrap() as f64
            / (BPS / 80.0);
        let fill: usize = (ps.iter().flat_map(|p| p.nal()))
            .filter(|n| h264::nal_type(n) == 12)
            .map(|n| n.len())
            .sum();
        println!(
            "{vbv} ms: {:.2} Mbit/s, from frame 1 {:.2} ({:+.1} %); first IDR {} bytes, largest 100 ms {most:.1} \
             times the rate's; filler {fill} bytes",
            br / 1e6,
            br1 / 1e6,
            (br1 / BPS - 1.0) * 100.0,
            ps[0].data.len(),
        );
        assert!((br1 / BPS - 1.0).abs() < 0.1, "{br1}");
        assert_eq!(fill, 0);
        most
    });
    assert!(most[0] <= most[1], "{most:?}");
}

/// A session VideoToolbox invalidates (`Encoder::invalidate`: what sleep or a media server restart can do) before
/// frame 12 is replaced there, with RTMP's settings and a file's: frame 12 is an IDR and the GOP restarts, every
/// frame comes back in order (the old session's last ones without data, if it could not emit them), and the stream
/// decodes against the parameter sets of the first session.
#[test]
fn renew() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let pitch = W;
    for vbv in [100, 1000] {
        let mut enc =
            Encoder::new(&g, &cfg(Codec::H264, false, Rate::Cbr(4_000_000), vbv)).unwrap();
        let ps = encode(
            &g,
            &mut enc,
            N,
            pitch,
            |f| picture(f, false, pitch),
            |e, f| {
                if f == 12 {
                    e.invalidate()
                }
            },
        );
        let lost: Vec<_> = (ps.iter().filter(|p| p.data.is_empty()))
            .map(|p| p.frame)
            .collect();
        assert!(lost.iter().all(|&f| f < 12), "{lost:?}");
        assert_eq!(keys(Codec::H264, &ps), [0, 10, 12, 22]);
        let db = decode(&g, Codec::H264, &enc.params, &ps);
        println!("{vbv} ms: frames {lost:?} lost, luma PSNR ≥ {db:.1} dB");
        assert!(db > 35.0, "{db}");
        let name = format!("renew{vbv}");
        probe(
            &stream(&enc.params, &ps),
            Codec::H264,
            &name,
            N - lost.len() as u64,
        );
    }
}

/// Every export preset and RTMP's largest size get the hardware encoder (no software fallback to slow them down),
/// whose first packet is an IDR that `params` describe (`packet` compares them). H.264 at 7680x3840, past the M4's
/// H.264 encoder, fails (the software encoder takes it at about 1 fps).
#[test]
fn hardware() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let c = |codec, deep, w, h, vbv| Cfg {
        w,
        h,
        gop: 30,
        ..cfg(codec, deep, Rate::Cbr(60_000_000), vbv)
    };
    for c in [
        c(Codec::H265, true, 5760, 2880, 1000),
        c(Codec::H265, false, 3840, 1920, 1000),
        c(Codec::H265, true, 7680, 3840, 1000),
        c(Codec::H264, false, 3840, 1920, 1000),
        c(Codec::H264, false, 1920, 1080, 1000),
        c(Codec::H265, false, 3840, 2160, 1000),
        c(Codec::H264, false, 1080, 1920, 1000),
        c(Codec::H264, false, 3840, 1920, 100),
    ] {
        let what = format!("{:?} {}x{} 10-bit {}", c.codec, c.w, c.h, c.deep);
        let mut e = Encoder::new(&g, &c).unwrap_or_else(|x| panic!("{what}: {x}"));
        e.input.acquire(&g, 0).unwrap();
        e.input.release(&g, 0);
        e.encode(&g, 0).unwrap();
        let mut p = Packet::default();
        assert!(e.packet(&mut p, 5_000_000_000).unwrap(), "{what}");
        assert!(p.key && !p.data.is_empty(), "{what}");
    }
    match Encoder::new(&g, &c(Codec::H264, false, 7680, 3840, 1000)) {
        Ok(_) => panic!("H.264 7680x3840 encodes"),
        Err(e) => println!("H.264 7680x3840: {e}"),
    }
}
