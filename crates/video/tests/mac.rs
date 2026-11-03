//! The VideoToolbox backend: synthetic pictures through the encoder's input (`Input::copy_buffer`), VideoToolbox
//! encode, our decoder, and back, for H.264, H.265 and H.265 Main 10; the streams are also checked with `ffprobe`
//! when it is installed (a test tool only).
#![cfg(target_os = "macos")]

use std::{process::Command, sync::Mutex};
use video::{Cfg, Codec, Decoder, Encoder, Error, Gpu, Packet, Rate, Ring};

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

/// Encodes `N` frames, decodes them again; returns the stream (Annex B with the parameter sets first) and the
/// lowest luma PSNR.
fn round_trip(codec: Codec, deep: bool, rate: Rate) -> (Vec<u8>, f64) {
    let g = Gpu::new().unwrap();
    let c = Cfg {
        codec,
        deep,
        w: W,
        h: H,
        fps: [30, 1],
        gop: 10,
        rate,
        vbv: 1000,
        quality: 0,
        full: false,
        chroma: 0,
    };
    let mut enc = Encoder::new(&g, &c).unwrap();
    assert!(!enc.params.is_empty(), "no parameter sets");
    let pitch = (W * (1 + deep as u32)).next_multiple_of(4) + 8; // not a multiple of 256 on purpose
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (pitch * (H + H / 2)) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut dec = Decoder::new(&g, codec, &enc.params, W, H).unwrap();
    let mut es: Vec<u8> = enc
        .params
        .iter()
        .flat_map(|p| [&[0, 0, 0, 1][..], p].concat())
        .collect();
    let (mut p, mut worst, mut keys) = (Packet::default(), f64::MAX, vec![]);
    let collect = |enc: &mut Encoder, p: &mut Packet, ns| {
        let mut got = vec![];
        while enc.packet(p, ns).unwrap() {
            got.push(p.clone());
        }
        got
    };
    let mut out = vec![];
    for f in 0..N {
        loop {
            match enc.input.acquire(&g, f) {
                Ok(()) => break,
                Err(Error::Full) => out.extend(collect(&mut enc, &mut p, 1_000_000)),
                Err(e) => panic!("{e}"),
            }
        }
        g.queue.write_buffer(&buf, 0, &picture(f, deep, pitch));
        let cb = enc.input.copy_buffer(&g, &buf, 0, pitch, f).unwrap();
        g.queue.submit([cb]);
        enc.input.release(&g, f);
        loop {
            match enc.encode(&g, f as i64 * 1000) {
                Ok(k) => break assert_eq!(k, f),
                Err(Error::Full) => out.extend(collect(&mut enc, &mut p, 1_000_000)),
                Err(e) => panic!("{e}"),
            }
        }
    }
    while enc.frame > out.len() as u64 {
        out.extend(collect(&mut enc, &mut p, 1_000_000_000));
    }
    for p in &out {
        assert_eq!(
            (p.pts, p.dts),
            (p.frame as i64 * 1000, p.frame as i64 * 1000)
        );
        if p.key {
            keys.push(p.frame);
        }
        es.extend_from_slice(&p.data);
        let f = dec.decode(&g, &p.nal().collect::<Vec<_>>()).unwrap();
        assert_eq!(f, p.frame);
        let want: Vec<u8> = (0..H)
            .flat_map(|y| (0..W).map(move |x| px(0, x, y, f)))
            .collect();
        worst = worst.min(psnr(&luma(&g, &dec, dec.view, f), &want));
        dec.release(&g, f);
    }
    assert_eq!(out.len() as u64, N);
    assert_eq!(keys, [0, 10, 20], "IDRs where the GOP puts them");
    (es, worst)
}

fn probe(es: &[u8], codec: Codec, name: &str) {
    let dir = std::env::temp_dir().join(format!("x5desk-mac-{}", std::process::id()));
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
    assert!(s.trim().ends_with(&format!(",{N}")), "{s}");
}

#[test]
fn h264() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (es, db) = round_trip(Codec::H264, false, Rate::Vbr(4_000_000, 8_000_000));
    println!("H.264: luma PSNR ≥ {db:.1} dB");
    assert!(db > 35.0, "{db}");
    probe(&es, Codec::H264, "h264");
}

#[test]
fn h265() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (es, db) = round_trip(Codec::H265, false, Rate::Cbr(4_000_000));
    println!("H.265: luma PSNR ≥ {db:.1} dB");
    assert!(db > 35.0, "{db}");
    probe(&es, Codec::H265, "h265");
}

#[test]
fn main10() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (es, db) = round_trip(Codec::H265, true, Rate::Qp(22));
    println!("H.265 Main 10: luma PSNR ≥ {db:.1} dB");
    assert!(db > 35.0, "{db}");
    probe(&es, Codec::H265, "main10");
}
