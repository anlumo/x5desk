//! Parsers and decoders on the X5's own footage (`X5_SAMPLES`, the live preview capture in `X5_LIVE`) and on
//! x264 / x265 streams for what the camera never sends, checked byte for byte against FFmpeg's software decode
//! (the `ffmpeg` CLI as a test tool only).
mod mp4;

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{ChildStdout, Command, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
use video::{Codec, Decoder, Error, Gpu, Ring, h264, h265};

/// GPU tests run one at a time: they measure, and each opens its own device.
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
    let ok = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        println!("ffmpeg not found: skipped");
    }
    ok
}

/// FFmpeg's software decode of `frames` of video track `t`, as yuvj420p (I420, the streams' own format).
fn reference(path: &Path, t: usize, frames: &[u64]) -> JoinHandle<Vec<u8>> {
    let mut c = select(path, t, frames);
    std::thread::spawn(move || {
        let o = c.output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        o.stdout
    })
}

/// `reference` as one hash per frame of `fs` bytes, for more frames than fit in memory.
fn reference_hashes(path: &Path, t: usize, frames: &[u64], fs: usize) -> JoinHandle<Vec<u64>> {
    let mut c = select(path, t, frames);
    std::thread::spawn(move || {
        let mut c = c.stdout(Stdio::piped()).spawn().unwrap();
        let (mut o, mut b, mut out) = (c.stdout.take().unwrap(), vec![0; fs], vec![]);
        while o.read_exact(&mut b).is_ok() {
            out.push(hash(&b));
        }
        assert!(c.wait().unwrap().success());
        out
    })
}

fn select(path: &Path, t: usize, frames: &[u64]) -> Command {
    let sel: Vec<_> = frames.iter().map(|n| format!("eq(n\\,{n})")).collect();
    let args = [
        "-v".into(),
        "error".into(),
        "-i".into(),
        path.to_str().unwrap().to_string(),
        "-map".into(),
        format!("0:v:{t}"),
        "-vf".into(),
        format!("select='{}'", sel.join("+")),
        "-fps_mode".into(),
        "passthrough".into(),
        "-f".into(),
        "rawvideo".into(),
        "-pix_fmt".into(),
        "yuvj420p".into(),
        "-".into(),
    ];
    let mut c = Command::new("ffmpeg");
    c.args(args);
    c
}

fn decode(g: &Gpu, d: &mut Decoder, au: &[&[u8]]) -> u64 {
    loop {
        match d.decode(g, au) {
            Err(Error::Full) => continue,
            r => return r.unwrap(),
        }
    }
}

/// The window `[x, y, w, h]` of frame `f` read back through wgpu copies of the NV12 texture's plane views, as I420;
/// the caller acquires and releases `f`.
fn read(g: &Gpu, d: &Ring, [x, y, w, h]: [u32; 4], f: u64) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let bpr = w.next_multiple_of(256); // both planes have w bytes per row
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (bpr * h * 3 / 2) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let tex = d.texture(f);
    let mut e = g.device.create_command_encoder(&Default::default());
    for (aspect, s, offset) in [
        (wgpu::TextureAspect::Plane0, 1, 0),
        (wgpu::TextureAspect::Plane1, 2, bpr * h),
    ] {
        let (pw, ph) = (w / s, h / s);
        e.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: tex,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: x / s as u32,
                    y: y / s as u32,
                    z: 0,
                },
                aspect,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: offset as u64,
                    bytes_per_row: Some(bpr as u32),
                    rows_per_image: Some(ph as u32),
                },
            },
            wgpu::Extent3d {
                width: pw as u32,
                height: ph as u32,
                depth_or_array_layers: 1,
            },
        );
    }
    // back to the home state before the decoder reuses the image
    let home = wgpu::TextureTransition {
        texture: tex,
        selector: None,
        state: wgpu::TextureUses::RESOURCE,
    };
    e.transition_resources(std::iter::empty(), [home].into_iter());
    let i = g.queue.submit([e.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    let i = wgpu::PollType::Wait {
        submission_index: Some(i),
        timeout: None,
    };
    g.device.poll(i).unwrap();
    let m = buf.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        out.extend_from_slice(&m[y * bpr..y * bpr + w]);
    }
    let uv = &m[bpr * h..];
    for c in 0..2 {
        for y in 0..h / 2 {
            out.extend(uv[y * bpr..y * bpr + w].iter().skip(c).step_by(2));
        }
    }
    out
}

/// Prints the mismatching bytes per plane (Y, U, V) of each frame; returns their sums.
fn compare(
    name: &str,
    frames: &[u64],
    [_, _, w, h]: [u32; 4],
    got: &[u8],
    want: &[u8],
) -> Vec<usize> {
    let (w, h) = (w as usize, h as usize);
    let (fs, ny) = (w * h * 3 / 2, w * h);
    assert_eq!(
        want.len(),
        fs * frames.len(),
        "{name}: FFmpeg returned {} bytes",
        want.len()
    );
    let mut out = vec![];
    for (k, f) in frames.iter().enumerate() {
        let (a, b) = (&got[k * fs..(k + 1) * fs], &want[k * fs..(k + 1) * fs]);
        let d = |r: std::ops::Range<usize>| {
            a[r.clone()]
                .iter()
                .zip(&b[r])
                .filter(|(x, y)| x != y)
                .count()
        };
        let (y, u, v) = (d(0..ny), d(ny..ny + ny / 4), d(ny + ny / 4..fs));
        println!("{name} frame {f}: differing bytes Y {y} U {u} V {v}");
        out.push(y + u + v);
    }
    out
}

fn exact(d: Vec<usize>) -> bool {
    d.iter().all(|&n| n == 0)
}

/// FFmpeg's software decode of an Annex B stream: raw I420 frames of the display window in display order, read
/// from its stdout.
fn ffdecode(codec: Codec, es: Vec<u8>) -> ChildStdout {
    let f = if codec == Codec::H264 { "h264" } else { "hevc" };
    let mut c = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            f,
            "-i",
            "-",
            "-fps_mode",
            "passthrough",
        ])
        .args(["-f", "rawvideo", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let (mut i, o) = (c.stdin.take().unwrap(), c.stdout.take().unwrap());
    // a failed decode also shows as missing frames on the reader's side
    std::thread::spawn(move || {
        i.write_all(&es).unwrap();
        drop(i);
        assert!(c.wait().unwrap().success());
    });
    o
}

/// The access units of an Annex B stream as NAL units without start codes: a new one where a parameter set, SEI,
/// AUD or a picture's first slice follows a slice.
fn annexb(codec: Codec, b: &[u8]) -> Vec<Vec<&[u8]>> {
    // 00 00 01 never occurs inside a NAL unit (emulation prevention)
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
        let (v, prefix, first) = match codec {
            Codec::H264 => {
                let t = h264::nal_type(n);
                (
                    matches!(t, 1..=5),
                    matches!(t, 6..=9 | 14..=18),
                    n[1] & 0x80 != 0,
                )
            }
            Codec::H265 => {
                let t = h265::nal_type(n);
                let prefix = matches!(t, 32..=35 | 39 | 41..=44 | 48..=55);
                (t < 32, prefix, n[2] & 0x80 != 0)
            }
        };
        if aus.is_empty() || vcl && (prefix || v && first) {
            aus.push(vec![]);
            vcl = false;
        }
        aus.last_mut().unwrap().push(n);
        vcl |= v;
    }
    aus
}

fn is_param(codec: Codec, n: &[u8]) -> bool {
    match codec {
        Codec::H264 => matches!(h264::nal_type(n), 7 | 8),
        Codec::H265 => matches!(h265::nal_type(n), 32..=34),
    }
}

fn hash(b: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    b.hash(&mut h);
    h.finish()
}

/// The shared device: every adapter feature, both video queues, and the encode function tables answer.
#[test]
fn gpu_setup() {
    use ash::vk;
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    assert_eq!(g.device.features(), g.adapter.features());
    let dec =
        vk::VideoCodecOperationFlagsKHR::DECODE_H264 | vk::VideoCodecOperationFlagsKHR::DECODE_H265;
    assert!(g.dec.ops.contains(dec) && *g.dec.queue.lock().unwrap() != vk::Queue::null());
    println!(
        "gfx family {}, decode {} {:?}",
        g.gfx, g.dec.family, g.dec.ops
    );
    let names: Vec<_> = g
        .ext
        .iter()
        .map(|e| e.to_string_lossy())
        .filter(|e| {
            ["video", "external", "drm", "foreign"]
                .iter()
                .any(|k| e.contains(k))
        })
        .collect();
    println!("{names:?}");
    let Some(enc) = &g.enc else {
        println!("no separate encode queue family");
        return;
    };
    println!("encode {} {:?}", enc.family, enc.ops);
    assert!(*enc.queue.lock().unwrap() != vk::Queue::null());
    unsafe {
        let mut h5 = vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(1);
        let pr = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h5);
        let (mut e5, mut ec) = (
            vk::VideoEncodeH265CapabilitiesKHR::default(),
            vk::VideoEncodeCapabilitiesKHR::default(),
        );
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut ec)
            .push_next(&mut e5);
        (g.vqi.fp().get_physical_device_video_capabilities_khr)(g.pd, &pr, &mut caps)
            .result()
            .unwrap();
        println!(
            "H.265 encode up to {:?}, {} DPB slots",
            caps.max_coded_extent, caps.max_dpb_slots
        );
        let info = vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR::default().video_profile(&pr);
        let mut q = vk::VideoEncodeQualityLevelPropertiesKHR::default();
        (g.veqi
            .fp()
            .get_physical_device_video_encode_quality_level_properties_khr)(
            g.pd, &info, &mut q
        )
        .result()
        .unwrap();
        println!(
            "quality level 0: rate control {:?}",
            q.preferred_rate_control_mode
        );
    }
}

#[test]
fn parse_x5() {
    let Some(p) = sample_file("v635.insv") else {
        return;
    };
    let (mut file, tracks) = mp4::read(&p).unwrap();
    let mut buf = vec![];
    for (ti, t) in tracks.iter().enumerate() {
        assert_eq!(t.codec, Codec::H265);
        let mut st = h265::State::default();
        for n in &t.params {
            st.put(n).unwrap();
        }
        let s = st.sps.iter().flatten().next().unwrap();
        println!(
            "track {ti}: {}x{} profile {} level_idc {} PCM {:?} max_dec_minus1 {:?} {} short-term RPS",
            s.width,
            s.height,
            s.ptl.profile_idc,
            s.ptl.level_idc,
            s.pcm,
            s.dpbm.max_dec_minus1[s.max_sub as usize],
            s.st.len()
        );
        assert_eq!(
            (s.width, s.height, s.ptl.profile_idc, s.ptl.level_idc),
            (3840, 3840, 1, 183)
        );
        assert!(s.pcm.is_some());
        let (mut idr, mut slices) = (vec![], vec![]);
        for i in 0..t.samples.len() {
            mp4::sample(&mut file, t, i, &mut buf);
            let au = mp4::nals(&buf, t.nal_len);
            let vcl: Vec<_> = au.iter().filter(|n| h265::nal_type(n) < 32).collect();
            slices.push(vcl.len());
            let sl = st.slice(vcl[0]).unwrap();
            if sl.nal_type == 19 || sl.nal_type == 20 {
                idr.push(i);
            } else {
                // P, one reference: the previous picture
                assert_eq!(
                    (
                        sl.slice_type,
                        sl.rps.s[0].n,
                        sl.rps.s[0].d[0],
                        sl.rps.s[0].used,
                        sl.rps.s[1].n
                    ),
                    (1, 1, -1, 1, 0),
                    "sample {i}"
                );
            }
        }
        println!(
            "track {ti}: IDR at {idr:?}, slices per picture {:?}",
            slices.iter().min()..=slices.iter().max()
        );
        assert!(slices.iter().all(|&n| n == 5));
        assert_eq!(idr, (0..t.samples.len()).step_by(60).collect::<Vec<_>>());
    }

    let Some(p) = sample_file("v635.lrv") else {
        return;
    };
    let (mut file, tracks) = mp4::read(&p).unwrap();
    let t = &tracks[0];
    assert_eq!(t.codec, Codec::H264);
    let mut st = h264::State::default();
    for n in &t.params {
        st.put(n).unwrap();
    }
    let s = st.sps.iter().flatten().next().unwrap();
    println!("lrv: {:?}", s);
    assert_eq!(
        (
            s.profile_idc,
            s.width_mbs * 16,
            s.height_mbs * 16,
            s.poc_type
        ),
        (100, 1664, 832, 0)
    );
    let (mut idr, mut types) = (vec![], [0; 5]);
    for i in 0..t.samples.len() {
        mp4::sample(&mut file, t, i, &mut buf);
        let au = mp4::nals(&buf, t.nal_len);
        for n in au.iter().filter(|n| matches!(h264::nal_type(n), 1 | 5)) {
            let sl = st.slice(n).unwrap();
            if sl.idr && !idr.contains(&i) {
                idr.push(i);
            }
            types[if sl.intra { 2 } else { 0 }] += 1;
        }
    }
    println!(
        "lrv: {} pictures, IDR at {idr:?}, {} P / {} I slices",
        t.samples.len(),
        types[0],
        types[2]
    );
    assert_eq!(idr, (0..t.samples.len()).step_by(30).collect::<Vec<_>>());
}

#[test]
fn h265_matches_ffmpeg() {
    let Some(path) = sample_file("v635.insv") else {
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let frames = [0, 1, 2, 59, 60, 61, 174];
    let refs: Vec<_> = (0..2).map(|t| reference(&path, t, &frames)).collect();
    let g = Gpu::new().unwrap();
    // the device has every adapter feature (plus the experimental ones) and still decodes
    assert_eq!(g.device.features(), g.adapter.features());
    println!(
        "{} adapter features enabled; decode family {}, encode {:?}",
        g.device.features().iter().count(),
        g.dec.family,
        g.enc.as_ref().map(|e| e.family)
    );
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let mut decs: Vec<_> = tracks
        .iter()
        .map(|t| Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap())
        .collect();
    let (mut got, mut buf) = (vec![vec![]; 2], vec![]);
    for i in 0..tracks[0].samples.len() {
        for (t, d) in tracks.iter().zip(&mut decs) {
            mp4::sample(&mut file, t, i, &mut buf);
            assert_eq!(decode(&g, d, &mp4::nals(&buf, t.nal_len)), i as u64);
        }
        if frames.contains(&(i as u64)) {
            for (t, d) in decs.iter().enumerate() {
                d.acquire(&g, i as u64);
                got[t].extend(read(&g, d, d.view, i as u64));
            }
        }
        // nothing else submits: the decoder sends the releases it waits for
        decs.iter().for_each(|d| d.release(&g, i as u64));
    }
    let mut ok = true;
    for (t, r) in refs.into_iter().enumerate() {
        ok &= exact(compare(
            &format!("v635 track {t}"),
            &frames,
            [0, 0, 3840, 3840],
            &got[t],
            &r.join().unwrap(),
        ));
    }
    assert!(ok);
}

#[test]
fn h264_matches_ffmpeg() {
    let Some(path) = sample_file("v635.lrv") else {
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let t = &tracks[0];
    let n = t.samples.len();
    // frames 0, 1, 2, the 2nd IDR and the frame after it, the last frame
    let mut buf = vec![];
    let idr: Vec<_> = (0..n)
        .filter(|&i| {
            mp4::sample(&mut file, t, i, &mut buf);
            mp4::nals(&buf, t.nal_len)
                .iter()
                .any(|n| h264::nal_type(n) == 5)
        })
        .collect();
    let frames = [0, 1, 2, idr[1] as u64, idr[1] as u64 + 1, n as u64 - 1];
    let want = reference(&path, 0, &frames);
    let g = Gpu::new().unwrap();
    // like the live preview: the PPS arrives in band (new session parameters), then repeats before every IDR
    assert_eq!(h264::nal_type(&t.params[0]), 7);
    let mut d = Decoder::new(&g, t.codec, &t.params[..1], t.width, t.height).unwrap();
    let (ring, size, mut got) = (d.ring.clone(), d.view, vec![]);
    // as in the app: the decoder on a media thread, sampling (here: copies) on the thread that submits to wgpu
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::scope(|s| {
        s.spawn(|| {
            let mut buf = vec![];
            for i in 0..n {
                mp4::sample(&mut file, t, i, &mut buf);
                let mut au = mp4::nals(&buf, t.nal_len);
                if idr.contains(&i) {
                    au.splice(0..0, t.params.iter().map(|p| &p[..]));
                }
                tx.send(decode(&g, &mut d, &au)).unwrap();
            }
            drop(tx);
        });
        for (i, f) in rx.iter().enumerate() {
            assert_eq!(f, i as u64);
            if frames.contains(&f) {
                ring.acquire(&g, f);
                got.extend(read(&g, &ring, size, f));
            }
            ring.release(&g, f);
        }
    });
    assert!(exact(compare(
        "v635.lrv",
        &frames,
        size,
        &got,
        &want.join().unwrap()
    )));
}

/// A consumer that stops: decode() returns Full instead of blocking, recovers on a release that nothing submits (an
/// idle UI), and dropping the decoder with frames never released returns.
#[test]
fn full_and_drop() {
    let Some(path) = sample_file("v635.lrv") else {
        return;
    };
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let t = &tracks[0];
    let mut d = Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap();
    let mut buf = vec![];
    let mut i = 0;
    loop {
        mp4::sample(&mut file, t, i, &mut buf);
        match d.decode(&g, &mp4::nals(&buf, t.nal_len)) {
            Ok(f) => assert_eq!(f, i as u64),
            Err(Error::Full) => break,
            Err(e) => panic!("{e}"),
        }
        i += 1;
    }
    assert_eq!(i, video::decode::OUT);
    // three releases staged for one submit: only the highest `free` value may be signalled. Nothing submits after
    // them, so they stay staged until decode() sends them with an empty submit.
    let ring = d.ring.clone();
    let free = || unsafe { g.dev.get_semaphore_counter_value(ring.free) }.unwrap();
    (0..3).for_each(|f| d.release(&g, f));
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        free(),
        0,
        "a staged release reached the GPU without a submit"
    );
    for f in i..i + 3 {
        mp4::sample(&mut file, t, f, &mut buf);
        let t0 = Instant::now();
        assert_eq!(d.decode(&g, &mp4::nals(&buf, t.nal_len)).unwrap(), f as u64);
        println!(
            "frame {f}: decode() {:.3} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    assert_eq!(free(), 3);
    assert!(matches!(
        d.decode(&g, &mp4::nals(&buf, t.nal_len)),
        Err(Error::Full)
    ));
    assert!(d.ready() >= 3);
    // what an explicit empty submit costs and how soon its staged signal lands
    d.release(&g, 3);
    let t0 = Instant::now();
    g.queue.submit([]);
    let t1 = t0.elapsed();
    let wait = ash::vk::SemaphoreWaitInfo::default()
        .semaphores(std::slice::from_ref(&ring.free))
        .values(&[4]);
    unsafe { g.dev.wait_semaphores(&wait, 1_000_000_000) }.unwrap();
    let t2 = t0.elapsed();
    let mut ts: Vec<_> = (0..1000)
        .map(|_| {
            let t = Instant::now();
            g.queue.submit([]);
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    ts.sort_by(f64::total_cmp);
    println!(
        "release + empty submit: submit {:.1} µs, free signalled after {:.1} µs; 1000 empty submits: median {:.1} \
         µs, p99 {:.1} µs",
        t1.as_secs_f64() * 1e6,
        t2.as_secs_f64() * 1e6,
        ts[500],
        ts[990]
    );
    drop(d);
}

/// Bit writer for re-encoding a parameter set (no emulation prevention: the caller checks none is needed).
struct W(Vec<u8>, u32);

impl W {
    fn u(&mut self, n: u32, v: u32) {
        for i in (0..n).rev() {
            if self.1.is_multiple_of(8) {
                self.0.push(0);
            }
            let l = self.0.len() - 1;
            self.0[l] |= ((v >> i & 1) as u8) << (7 - self.1 % 8);
            self.1 += 1;
        }
    }
    fn ue(&mut self, v: u32) {
        let l = 32 - (v + 1).leading_zeros();
        self.u(l - 1, 0);
        self.u(l, v + 1);
    }
    fn se(&mut self, v: i32) {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            (-2 * v) as u32
        });
    }
}

/// H.264 PPS `a` (no slice groups or scaling matrices) re-encoded with pic_init_qp_minus26 raised by `dq`.
fn pps_qp(a: &[u8], dq: i8) -> Vec<u8> {
    let p = h264::pps(a).unwrap();
    let mut w = W(vec![a[0]], 8);
    w.ue(p.id as u32);
    w.ue(p.sps_id as u32);
    w.u(1, p.cabac as u32);
    w.u(1, p.bottom_poc as u32);
    w.ue(0); // slice groups
    w.ue(p.ref_idx_default[0] as u32);
    w.ue(p.ref_idx_default[1] as u32);
    w.u(1, p.weighted_pred as u32);
    w.u(2, p.weighted_bipred as u32);
    w.se((p.qp_minus26 + dq) as i32);
    w.se(p.qs_minus26 as i32);
    w.se(p.chroma_qp as i32);
    w.u(1, p.deblock_ctl as u32);
    w.u(1, p.constrained_intra as u32);
    w.u(1, p.redundant as u32);
    w.u(1, p.t8x8 as u32);
    w.u(1, 0); // scaling matrices
    w.se(p.chroma_qp2 as i32);
    w.u(1, 1); // stop bit
    w.u((8 - w.1 % 8) % 8, 0);
    let epb = w.0.windows(3).any(|x| x[..2] == [0, 0] && x[2] <= 3);
    assert!(!epb, "needs emulation prevention");
    w.0
}

/// In-band parameter sets: the last one received under an id is the active one, also when it repeats an earlier
/// one (A → B → A), within an access unit or across them.
#[test]
fn in_band_params() {
    let Some(path) = sample_file("v635.lrv") else {
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let t = &tracks[0];
    let (sps, a) = (&t.params[0][..], &t.params[1][..]);
    assert_eq!(pps_qp(a, 0), a, "the writer reproduces A");
    let b = pps_qp(a, 6);
    let mut st = h264::State::default();
    let r = [sps, a, &b, a, a].map(|n| st.put(n).unwrap());
    assert_eq!(r, [true, true, true, true, false]);
    assert_eq!(
        st.pps[0].unwrap().qp_minus26,
        h264::pps(a).unwrap().qp_minus26
    );

    // B in band before frame 0 (a wrong picture on purpose: B is in effect), A back before the IDR at 30, B and
    // then A before the IDR at 60
    let frames = [0, 30, 60];
    let want = reference(&path, 0, &frames);
    let g = Gpu::new().unwrap();
    let mut d = Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap();
    let (mut got, mut buf) = (vec![], vec![]);
    for i in 0..=60 {
        mp4::sample(&mut file, t, i, &mut buf);
        let mut au = mp4::nals(&buf, t.nal_len);
        let pre: &[&[u8]] = match i {
            0 => &[&b[..]],
            30 => &[a],
            60 => &[&b[..], a],
            _ => &[],
        };
        au.splice(0..0, pre.iter().copied());
        let f = decode(&g, &mut d, &au);
        if frames.contains(&f) {
            d.acquire(&g, f);
            got.extend(read(&g, &d, d.view, f));
        }
        d.release(&g, f);
    }
    let diff = compare(
        "v635.lrv, in-band PPS",
        &frames,
        d.view,
        &got,
        &want.join().unwrap(),
    );
    assert!(diff[0] > 100_000 && diff[1..] == [0, 0], "{diff:?}");
}

/// The camera's live preview (H.264 High 2656x1328, I/P, SPS and PPS in band before every IDR) from an Annex B
/// capture in `X5_LIVE`: every frame byte-identical to FFmpeg.
#[test]
fn live_matches_ffmpeg() {
    let Some(path) = std::env::var_os("X5_LIVE") else {
        println!("X5_LIVE unset: skipped");
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let es = std::fs::read(&path).unwrap();
    let aus = annexb(Codec::H264, &es);
    let mut ff = ffdecode(Codec::H264, es.clone());
    let g = Gpu::new().unwrap();
    let params: Vec<_> = aus[0]
        .iter()
        .copied()
        .filter(|n| is_param(Codec::H264, n))
        .collect();
    let s = h264::sps(params[0]).unwrap();
    println!(
        "live: profile {} {}x{} POC type {} max_num_ref_frames {}",
        s.profile_idc,
        s.width_mbs * 16,
        s.height_mbs * 16,
        s.poc_type,
        s.max_num_ref_frames
    );
    let mut d = Decoder::new(&g, Codec::H264, &params, 0, 0).unwrap();
    let [_, _, w, h] = d.view;
    let mut want = vec![0; (w * h * 3 / 2) as usize];
    let (mut bad, mut idr, mut inband) = (0, 0, 0);
    for au in &aus {
        idr += au.iter().any(|n| h264::nal_type(n) == 5) as usize;
        inband += au.iter().filter(|n| is_param(Codec::H264, n)).count();
        let f = decode(&g, &mut d, au);
        d.acquire(&g, f);
        let got = read(&g, &d, d.view, f);
        d.release(&g, f);
        ff.read_exact(&mut want).unwrap();
        let n = got.iter().zip(&want).filter(|(x, y)| x != y).count();
        if n != 0 {
            println!("live frame {f}: {n} differing bytes");
            bad += 1;
        }
    }
    println!(
        "live: {} frames of {w}x{h}, {idr} IDR, {inband} parameter sets in band, {bad} differing",
        aus.len()
    );
    assert_eq!(bad, 0);
    assert_eq!(ff.read(&mut want).unwrap(), 0, "FFmpeg has more frames");
}

/// x264 / x265 streams for what the camera never sends, every frame byte-identical to FFmpeg (compared as sets:
/// FFmpeg outputs in display order, the decoder in decode order):
/// - x264 P only: POC type 2, a sliding window of 4 references, weighted prediction, cropping (404 of 416 rows)
/// - x264 B-pyramid: POC type 0, MMCO 1, temporal direct, implicit weights (POC distances)
/// - x265 B-pyramid, open GOP: sub-layer non-reference pictures, RPS in slice headers, CRAs with RASL pictures,
///   conformance window; then again from the first CRA, where only its RASL pictures may fail
#[test]
fn x26x_matches_ffmpeg() {
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (w, h) = (720, 404);
    let fs = w * h * 3 / 2;
    for (codec, name, args) in [
        (
            Codec::H264,
            "x264 P",
            "-c:v libx264 -bf 0 -refs 4 -g 25 -f h264",
        ),
        (
            Codec::H264,
            "x264 B",
            "-c:v libx264 -bf 3 -refs 3 -g 25 -x264-params b-pyramid=normal:direct=temporal:weightb=1 -f h264",
        ),
        (
            Codec::H265,
            "x265 B",
            "-c:v libx265 -x265-params keyint=25:min-keyint=25:bframes=4:ref=3:open-gop=1:log-level=error -f hevc",
        ),
    ] {
        let o = Command::new("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i"])
            .arg(format!("testsrc2=size={w}x{h}:rate=30"))
            .args(["-frames:v", "60", "-pix_fmt", "yuv420p"])
            .args(args.split(' '))
            .arg("-")
            .output()
            .unwrap();
        if !o.status.success() {
            println!("{name}: skipped, {}", String::from_utf8_lossy(&o.stderr));
            continue;
        }
        let aus = annexb(codec, &o.stdout);
        let mut ff = ffdecode(codec, o.stdout.clone());
        let mut want = vec![0; fs * aus.len()];
        ff.read_exact(&mut want).unwrap();
        assert_eq!(
            ff.read(&mut [0]).unwrap(),
            0,
            "{name}: FFmpeg has more frames"
        );
        let mut want: Vec<_> = want.chunks(fs).map(hash).collect();
        want.sort();
        let params: Vec<_> = aus[0]
            .iter()
            .copied()
            .filter(|n| is_param(codec, n))
            .collect();
        let cra = aus
            .iter()
            .position(|au| codec == Codec::H265 && au.iter().any(|n| h265::nal_type(n) == 21));
        for start in [Some(0), cra].into_iter().flatten() {
            let mut d = Decoder::new(&g, codec, &params, 0, 0).unwrap();
            let (mut got, mut skipped) = (vec![], 0);
            for au in &aus[start..] {
                match d.decode(&g, au) {
                    Ok(f) => {
                        d.acquire(&g, f);
                        got.push(hash(&read(&g, &d, d.view, f)));
                        d.release(&g, f);
                    }
                    Err(Error::Bad(e)) if start > 0 => {
                        let t = au.iter().map(|n| h265::nal_type(n)).find(|&t| t < 32);
                        assert!(
                            matches!(t, Some(8 | 9)),
                            "{name} from AU {start}: {e} at {t:?}"
                        );
                        skipped += 1;
                    }
                    Err(e) => panic!("{name}: {e}"),
                }
            }
            let miss = got
                .iter()
                .filter(|x| want.binary_search(x).is_err())
                .count();
            println!(
                "{name} from AU {start}: {} of {} pictures decoded, view {:?} in {}x{}, {miss} not among \
                 FFmpeg's frames, {skipped} RASL failed",
                got.len(),
                aus.len() - start,
                d.view,
                d.w,
                d.h
            );
            assert_eq!(miss, 0, "{name}");
            if start == 0 {
                got.sort();
                assert_eq!(got, want, "{name}: not FFmpeg's frames");
            }
        }
    }
}

const RGB: &str = r"
@group(0) @binding(0) var ty: texture_2d<f32>;
@group(0) @binding(1) var tuv: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

// full-range BT.709 to RGBA at full resolution: a stand-in for the stitcher's sampling
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
    let y = textureLoad(ty, g.xy, 0).r;
    let c = textureLoad(tuv, g.xy / 2u, 0).rg - vec2f(0.5);
    textureStore(dst, g.xy, vec4f(y + 1.5748 * c.y, y - 0.1873 * c.x - 0.4681 * c.y, y + 1.8556 * c.x, 1.0));
}
";

/// The RGB pass for each decoder: a command buffer that samples frame `f` of every ring.
fn rgb(g: &Gpu, decs: &[Decoder]) -> impl Fn(u64) -> wgpu::CommandBuffer + use<> {
    let (w, h, dv) = (decs[0].w, decs[0].h, g.device.clone());
    let sm = dv.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(RGB.into()),
    });
    let pipe = dv.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    // one bind group per (lens, ring slot); they keep the views and output textures alive
    let bgs: Vec<Vec<_>> = decs
        .iter()
        .map(|d| {
            let out = dv.create_texture(&wgpu::TextureDescriptor {
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
                usage: wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            (0..video::decode::OUT as u64)
                .map(|f| {
                    let v = |format, aspect| {
                        d.texture(f).create_view(&wgpu::TextureViewDescriptor {
                            format: Some(format),
                            aspect,
                            ..Default::default()
                        })
                    };
                    let (y, uv) = (
                        v(wgpu::TextureFormat::R8Unorm, wgpu::TextureAspect::Plane0),
                        v(wgpu::TextureFormat::Rg8Unorm, wgpu::TextureAspect::Plane1),
                    );
                    let o = out.create_view(&Default::default());
                    dv.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &pipe.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&y),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&uv),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&o),
                            },
                        ],
                    })
                })
                .collect()
        })
        .collect();
    move |f| {
        let mut e = dv.create_command_encoder(&Default::default());
        {
            let mut p = e.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipe);
            for b in &bgs {
                p.set_bind_group(0, &b[f as usize % video::decode::OUT], &[]);
                p.dispatch_workgroups(w / 16, h / 16, 1);
            }
        }
        e.finish()
    }
}

/// Frame pairs per second for both lenses of v635 (175 pairs), with an NV12 → RGBA pass per lens and frame.
#[test]
fn throughput() {
    let Some(path) = sample_file("v635.insv") else {
        return;
    };
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let mut decs: Vec<_> = tracks
        .iter()
        .map(|t| Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap())
        .collect();
    let pass = rgb(&g, &decs);
    let n = tracks[0].samples.len();
    let (t0, mut t1, mut buf, mut bytes) = (Instant::now(), Instant::now(), vec![], 0);
    for i in 0..n {
        if i == 1 {
            t1 = Instant::now();
        }
        for (t, d) in tracks.iter().zip(&mut decs) {
            mp4::sample(&mut file, t, i, &mut buf);
            bytes += buf.len();
            decode(&g, d, &mp4::nals(&buf, t.nal_len));
        }
        let i = i as u64;
        decs.iter().for_each(|d| d.acquire(&g, i));
        g.queue.submit([pass(i)]);
        decs.iter().for_each(|d| d.release(&g, i));
        g.device.poll(wgpu::PollType::Poll).unwrap();
    }
    g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let (all, rest) = (
        n as f64 / t0.elapsed().as_secs_f64(),
        (n - 1) as f64 / t1.elapsed().as_secs_f64(),
    );
    println!(
        "v635: {n} frame pairs, {all:.1} pairs/s ({rest:.1} after the first), {:.1} Mbit/s of HEVC",
        bytes as f64 * 8.0 * all / n as f64 / 1e6
    );
    assert!(all > 30.0, "slower than real time");
}

/// A paused view: frame 40 of the proxy is sampled by 10 successive submits without a release while the decoder
/// keeps trying to run ahead. It decodes the three frames after it and no more, the held frame stays byte-exact,
/// and after the release (with nothing submitting) the decoder moves on by itself.
#[test]
fn pause() {
    let Some(path) = sample_file("v635.lrv") else {
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let (t, p, out) = (&tracks[0], 40, video::decode::OUT as u64);
    let frames: Vec<_> = (p..=p + out).collect();
    let want = reference(&path, 0, &frames).join().unwrap();
    let g = Gpu::new().unwrap();
    let mut d = Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap();
    let (ring, view, n) = (d.ring.clone(), d.view, t.samples.len());
    let fs = (view[2] * view[3] * 3 / 2) as usize;
    let want = |f: u64| &want[(f - p) as usize * fs..][..fs];
    let (full, (tx, rx)) = (AtomicUsize::new(0), std::sync::mpsc::channel());
    std::thread::scope(|s| {
        s.spawn(|| {
            let mut buf = vec![];
            for i in 0..n {
                mp4::sample(&mut file, t, i, &mut buf);
                let au = mp4::nals(&buf, t.nal_len);
                let f = loop {
                    match d.decode(&g, &au) {
                        // 10 s: fail instead of hanging when the main thread failed while holding a frame
                        Err(Error::Full) => assert!(full.fetch_add(1, Relaxed) < 200, "stuck"),
                        r => break r.unwrap(),
                    };
                };
                tx.send(f).unwrap();
            }
            drop(tx);
        });
        let mut bad = 0;
        for f in rx.iter() {
            if f == p {
                let f0 = full.load(Relaxed);
                for k in 0..10 {
                    ring.acquire(&g, p);
                    let n = read(&g, &ring, view, p)
                        .iter()
                        .zip(want(p))
                        .filter(|(x, y)| x != y)
                        .count();
                    println!("held frame {p}, submit {k}: {n} bytes differ from FFmpeg");
                    bad += n;
                    std::thread::sleep(Duration::from_millis(30));
                }
                let (ready, fulls) = (ring.ready(), full.load(Relaxed) - f0);
                println!(
                    "while held: frames 0..{ready} decoded, decode() returned Full {fulls} times"
                );
                assert_eq!(ready, p + out, "the decoder ran past the held frame");
                assert!(fulls >= 3);
                ring.release(&g, p);
                std::thread::sleep(Duration::from_millis(200));
                assert_eq!(
                    ring.ready(),
                    p + out + 1,
                    "the decoder starved after the release"
                );
                continue;
            }
            ring.acquire(&g, f);
            if f > p && f <= p + out {
                let n = read(&g, &ring, view, f)
                    .iter()
                    .zip(want(f))
                    .filter(|(x, y)| x != y)
                    .count();
                println!("frame {f}: {n} bytes differ from FFmpeg");
                bad += n;
            } else {
                g.queue.submit([]);
            }
            ring.release(&g, f);
        }
        assert_eq!(bad, 0);
    });
}

/// Another thread submits unrelated wgpu work (copies and empty submits) as fast as it can, so its submits take the
/// waits and signals that `acquire` and `release` stage around the sampling submits. This thread decodes both lenses
/// of v635 and runs an RGB pass three frames behind the decoder (which then waits for the releases), and reads back
/// every third frame right after its decode: for those, the other thread's submit took the wait (it submitted
/// after the `acquire`), so the copy is ordered after the decode only by submission order. All of them byte-identical
/// to FFmpeg (compared as hashes).
#[test]
fn stress() {
    let Some(path) = sample_file("v635.insv") else {
        return;
    };
    if !ffmpeg() {
        return;
    }
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut file, tracks) = mp4::read(&path).unwrap();
    let n = tracks[0].samples.len();
    let frames: Vec<_> = (0..n as u64).step_by(3).collect();
    let refs: Vec<_> = (0..2)
        .map(|t| reference_hashes(&path, t, &frames, 3840 * 3840 * 3 / 2))
        .collect();
    let g = Gpu::new().unwrap();
    let mut decs: Vec<_> = tracks
        .iter()
        .map(|t| Decoder::new(&g, t.codec, &t.params, t.width, t.height).unwrap())
        .collect();
    let pass = rgb(&g, &decs);
    let (stop, subs) = (AtomicBool::new(false), AtomicUsize::new(0));
    let (mut got, mut early, mut buf) = (vec![vec![]; 2], 0, vec![]);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        s.spawn(|| {
            let dv = &g.device;
            let b = |usage| {
                dv.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 4 << 20,
                    usage,
                    mapped_at_creation: false,
                })
            };
            let t = |usage| {
                dv.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: 1024,
                        height: 1024,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage,
                    view_formats: &[],
                })
            };
            let (src, dst) = (
                b(wgpu::BufferUsages::COPY_SRC),
                b(wgpu::BufferUsages::COPY_DST),
            );
            let (ts, td) = (
                t(wgpu::TextureUsages::COPY_SRC),
                t(wgpu::TextureUsages::COPY_DST),
            );
            while !stop.load(Relaxed) {
                let mut e = dv.create_command_encoder(&Default::default());
                e.copy_buffer_to_buffer(&src, 0, &dst, 0, 4 << 20);
                e.copy_texture_to_texture(ts.as_image_copy(), td.as_image_copy(), ts.size());
                g.queue.submit([e.finish()]);
                subs.fetch_add(1, Relaxed);
                g.queue.submit([]);
                subs.fetch_add(1, Relaxed);
            }
        });
        const LAG: usize = 3;
        for i in 0..n + LAG {
            if i < n {
                for (t, d) in tracks.iter().zip(&mut decs) {
                    mp4::sample(&mut file, t, i, &mut buf);
                    assert_eq!(decode(&g, d, &mp4::nals(&buf, t.nal_len)), i as u64);
                }
                let i = i as u64;
                for (t, d) in decs.iter().enumerate().filter(|_| frames.contains(&i)) {
                    d.acquire(&g, i);
                    // two more submits counted: the second began after the acquire and took the wait
                    let k = subs.load(Relaxed);
                    while subs.load(Relaxed) < k + 2 {
                        std::hint::spin_loop();
                    }
                    early += (d.ready() <= i) as usize;
                    got[t].push(hash(&read(&g, d, d.view, i)));
                }
            }
            let Some(j) = i.checked_sub(LAG).map(|j| j as u64) else {
                continue;
            };
            decs.iter().for_each(|d| d.acquire(&g, j));
            g.queue.submit([pass(j)]);
            decs.iter().for_each(|d| d.release(&g, j));
        }
        stop.store(true, Relaxed);
    });
    let (s, subs) = (t0.elapsed().as_secs_f64(), subs.into_inner());
    let mut ok = true;
    for (t, r) in refs.into_iter().enumerate() {
        let r = r.join().unwrap();
        assert_eq!(r.len(), frames.len(), "FFmpeg returned {} frames", r.len());
        let bad: Vec<_> = frames
            .iter()
            .zip(got[t].iter().zip(&r))
            .filter(|(_, (a, b))| a != b)
            .map(|(f, _)| f)
            .collect();
        println!(
            "track {t}: {} frames compared, differing: {bad:?}",
            frames.len()
        );
        ok &= bad.is_empty();
    }
    println!(
        "{n} frame pairs in {s:.2} s ({:.1} pairs/s); {early} of {} copies submitted before their decode had \
         finished; the other thread made {subs} submits ({:.0}/s)",
        n as f64 / s,
        2 * frames.len(),
        subs as f64 / s,
    );
    assert!(ok);
}
