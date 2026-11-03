//! The VideoToolbox decoder on hand-made H.264 (Baseline, 64 × 64, I_PCM pictures of one luma value each, so every
//! picture decodes exactly): a picture VideoToolbox fails on shows the one before it, never what its slot held; the
//! decoder's gate; the scaling list parser on garbage. And an open-GOP H.265 stream from the ffmpeg CLI (a test tool)
//! with a broken picture.
#![cfg(target_os = "macos")]

use std::{
    sync::{Mutex, RwLock, mpsc},
    thread,
    time::{Duration, Instant},
};
use video::{Codec, Decoder, Error, Gpu, h264, h265};

/// GPU tests run one at a time: each opens its own device.
static GPU: Mutex<()> = Mutex::new(());

const S: u32 = 64;

/// RBSP bits in, a NAL unit (emulation prevention bytes in) out.
#[derive(Default)]
struct W {
    b: Vec<u8>,
    c: u64,
    n: u32,
}

impl W {
    fn u(&mut self, n: u32, v: u32) -> &mut W {
        self.c = self.c << n | (v as u64 & ((1 << n) - 1));
        self.n += n;
        while self.n >= 8 {
            self.n -= 8;
            self.b.push((self.c >> self.n) as u8);
        }
        self
    }

    fn ue(&mut self, v: u32) -> &mut W {
        let z = 31 - (v + 1).leading_zeros();
        self.u(z, 0).u(z + 1, v + 1)
    }

    fn se(&mut self, v: i32) -> &mut W {
        self.ue(if v > 0 {
            2 * v as u32 - 1
        } else {
            2 * v.unsigned_abs()
        })
    }

    fn align(&mut self) -> &mut W {
        let n = (8 - self.n) % 8;
        self.u(n, 0)
    }

    /// The bytes so far, cut where they are (no trailing bits): a truncated NAL unit.
    fn cut(&mut self) -> Vec<u8> {
        self.align();
        self.b.clone()
    }

    fn nal(&mut self) -> Vec<u8> {
        self.u(1, 1).align();
        let (mut o, mut z) = (vec![], 0);
        for &x in &self.b {
            if z >= 2 && x <= 3 {
                o.push(3);
                z = 0;
            }
            z = if x == 0 { z + 1 } else { 0 };
            o.push(x);
        }
        o
    }
}

/// SPS and PPS: Constrained Baseline, level 3, 64 × 64, POC type 2, one reference frame; CAVLC.
fn sets() -> Vec<Vec<u8>> {
    let mut s = W::default();
    s.u(8, 0x67).u(8, 66).u(8, 0xc0).u(8, 30).ue(0);
    s.ue(0).ue(2).ue(1).u(1, 0);
    s.ue(S / 16 - 1)
        .ue(S / 16 - 1)
        .u(1, 1)
        .u(1, 1)
        .u(1, 0)
        .u(1, 0);
    let mut p = W::default();
    p.u(8, 0x68).ue(0).ue(0).u(1, 0).u(1, 0).ue(0);
    p.ue(0).ue(0).u(1, 0).u(2, 0);
    p.se(0).se(0).se(0).u(1, 0).u(1, 0).u(1, 0);
    vec![s.nal(), p.nal()]
}

/// IDR slice header `id`, every macroblock I_PCM after it.
fn idr(id: u32) -> W {
    let mut w = W::default();
    w.u(8, 0x65).ue(0).ue(7).ue(0).u(4, 0).ue(id);
    w.u(1, 0).u(1, 0).se(0);
    w
}

/// An IDR whose luma is `y` all over (chroma 128).
fn flat(id: u32, y: u8) -> Vec<u8> {
    let mut w = idr(id);
    for _ in 0..(S / 16) * (S / 16) {
        w.ue(25).align();
        (0..256).for_each(|_| _ = w.u(8, y as u32));
        (0..128).for_each(|_| _ = w.u(8, 128));
    }
    w.nal()
}

/// An IDR VideoToolbox fails on (kVTVideoDecoderBadDataErr here): the header and one macroblock cut short.
fn broken(id: u32) -> Vec<u8> {
    let mut w = idr(id);
    w.ue(25).align();
    (0..100).for_each(|_| _ = w.u(8, 9));
    w.cut()
}

/// A reference P picture, frame_num `n`, every macroblock skipped: the picture before it again.
fn skip(n: u32) -> Vec<u8> {
    let mut w = W::default();
    w.u(8, 0x41).ue(0).ue(5).ue(0).u(4, n);
    w.u(1, 0).u(1, 0).u(1, 0).se(0);
    w.ue((S / 16) * (S / 16)).nal()
}

/// The mean luma of frame `f`.
fn luma(g: &Gpu, d: &Decoder, f: u64) -> f64 {
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (256 * S) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = g.device.create_command_encoder(&Default::default());
    e.copy_texture_to_buffer(
        d.plane(f, 0),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: S,
            height: S,
            depth_or_array_layers: 1,
        },
    );
    d.acquire(g, f);
    let i = g.queue.submit([e.finish()]);
    buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(i),
            timeout: None,
        })
        .unwrap();
    let m = buf.slice(..).get_mapped_range().unwrap();
    let n: u64 = (0..S as usize)
        .flat_map(|r| &m[r * 256..][..S as usize])
        .map(|&x| x as u64)
        .sum();
    n as f64 / (S * S) as f64
}

/// Decodes `au`; its frame and mean luma, then releases it.
fn take(g: &Gpu, d: &mut Decoder, au: &[u8]) -> Result<(u64, f64), Error> {
    let f = d.decode(g, &[au])?;
    let y = luma(g, d, f);
    d.release(g, f);
    Ok((f, y))
}

/// A picture VideoToolbox fails on shows the last one decoded, not the frame `slots()` earlier its slot still holds,
/// and counts in `failed`; the next random access point decodes again. After a random access point fails, the P
/// picture after it is concealed too. With nothing decoded before it (a new decoder, a restart), the picture is an
/// error: never the slot's old content or a picture from before the restart.
#[test]
fn conceal() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let mut d = Decoder::new(&g, Codec::H264, &sets(), S, S).unwrap();
    assert_eq!(d.slots(), 4);
    for k in 0..5 {
        let y = 20 + 20 * k as u8;
        assert_eq!(take(&g, &mut d, &flat(k, y)).unwrap(), (k as u64, y as f64));
    }
    // frame 5 goes to slot 1, which still holds frame 1 (40)
    assert_eq!(take(&g, &mut d, &broken(5)).unwrap(), (5, 100.0));
    assert_eq!(d.failed, 1);
    assert_eq!(take(&g, &mut d, &flat(6, 120)).unwrap(), (6, 120.0));
    assert_eq!(take(&g, &mut d, &skip(1)).unwrap(), (7, 120.0));
    assert_eq!(take(&g, &mut d, &broken(8)).unwrap(), (8, 120.0));
    assert_eq!(take(&g, &mut d, &skip(1)).unwrap(), (9, 120.0));
    assert_eq!(d.failed, 3);
    assert_eq!(take(&g, &mut d, &flat(10, 140)).unwrap(), (10, 140.0));
    d.restart();
    let e = take(&g, &mut d, &broken(11));
    println!("after a restart: {e:?}");
    assert!(matches!(e, Err(Error::Os(_, _))), "{e:?}");
    assert_eq!((d.frame, d.failed), (11, 3));
    assert_eq!(take(&g, &mut d, &flat(12, 160)).unwrap(), (11, 160.0));
    let mut d = Decoder::new(&g, Codec::H264, &sets(), S, S).unwrap();
    assert!(take(&g, &mut d, &broken(0)).is_err());
    assert_eq!(take(&g, &mut d, &flat(1, 180)).unwrap(), (0, 180.0));
}

/// A gated decoder submits its copies holding its lock shared: a free lock costs nothing, a writer holds them back
/// until it lets go (taken then, not at the timeout), and one that never does only for 500 ms (the writer may be
/// waiting for this thread).
#[test]
fn gate() {
    static L: RwLock<()> = RwLock::new(());
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let g = Gpu::new().unwrap();
    let mut d = Decoder::new(&g, Codec::H264, &sets(), S, S)
        .unwrap()
        .gated(&L);
    let t = Instant::now();
    assert_eq!(take(&g, &mut d, &flat(0, 50)).unwrap(), (0, 50.0));
    let e = t.elapsed();
    println!("lock free: decode and read back {e:?}");
    assert!(e < Duration::from_millis(250), "{e:?}");
    let (tx, rx) = mpsc::channel();
    let w = thread::spawn(move || {
        let _w = L.write().unwrap();
        tx.send(()).unwrap();
        thread::sleep(Duration::from_millis(40));
    });
    rx.recv().unwrap();
    let t = Instant::now();
    let f = d.decode(&g, &[&flat(1, 60)]).unwrap();
    let (e, took) = (t.elapsed(), d.took);
    w.join().unwrap();
    println!("behind a writer: decode {e:?}, of it the decoder's {took:?}");
    assert!(
        e >= Duration::from_millis(25) && e < Duration::from_millis(250) && took <= e,
        "{e:?} {took:?}"
    );
    assert_eq!(luma(&g, &d, f), 60.0);
    d.release(&g, f);
    let w = L.write().unwrap();
    let t = Instant::now();
    let f = d.decode(&g, &[&flat(2, 70)]).unwrap();
    let e = t.elapsed();
    drop(w);
    println!("lock held by its own thread: decode {e:?}");
    assert!(
        (Duration::from_millis(450)..Duration::from_secs(2)).contains(&e),
        "{e:?}"
    );
    assert_eq!(luma(&g, &d, f), 70.0);
}

/// High-profile SPS with all 8 scaling lists (`cut`: cut short in list 2 after `cut` coefficients; `big`: one
/// delta_scale of 300) or without any; 1920 × 1088, POC type 0.
fn high(lists: bool, cut: Option<i32>, big: bool) -> Vec<u8> {
    let mut w = W::default();
    w.u(8, 0x67).u(8, 100).u(8, 0).u(8, 40).ue(0);
    w.ue(1).ue(0).ue(0).u(1, 0).u(1, lists as u32);
    if lists {
        for i in 0..8 {
            w.u(1, 1);
            for j in 0..if i < 6 { 16 } else { 64 } {
                if i == 2 && Some(j) == cut {
                    return w.cut();
                }
                w.se(if big && i == 3 && j == 5 {
                    300
                } else {
                    j % 5 - 2
                });
            }
        }
    }
    w.ue(0).ue(0).ue(0).ue(1).u(1, 0);
    w.ue(119).ue(67).u(1, 1).u(1, 1).u(1, 0).u(1, 0);
    w.nal()
}

/// PPS with transform_8x8_mode and its 8 scaling lists (cut short in list 6 when `cut`), then
/// second_chroma_qp_index_offset −2.
fn pps(cut: bool) -> Vec<u8> {
    let mut w = W::default();
    w.u(8, 0x68).ue(0).ue(0).u(1, 1).u(1, 0).ue(0);
    w.ue(0).ue(0).u(1, 0).u(2, 0);
    w.se(0).se(0).se(0).u(1, 1).u(1, 0).u(1, 0);
    w.u(1, 1).u(1, 1);
    for i in 0..8 {
        w.u(1, 1);
        for j in 0..if i < 6 { 16 } else { 64 } {
            if cut && i == 6 && j == 10 {
                return w.cut();
            }
            w.se(2 - j % 5);
        }
    }
    w.se(-2).nal()
}

/// Scaling lists are stepped over (VideoToolbox applies them): the rest of the set reads as without them; one cut
/// short or with a delta_scale outside −128..=127 is an error, not an overflow.
#[test]
fn scaling() {
    let (with, without) = (
        h264::sps(&high(true, None, false)),
        h264::sps(&high(false, None, false)),
    );
    let s = with.as_ref().unwrap();
    assert_eq!(
        (s.width_mbs, s.height_mbs, s.max_num_ref_frames),
        (120, 68, 1)
    );
    assert_eq!(format!("{with:?}"), format!("{without:?}"));
    for c in [0, 1, 3, 15] {
        let e = h264::sps(&high(true, Some(c), false));
        assert!(matches!(e, Err(Error::Bad(_))), "cut after {c}: {e:?}");
    }
    let e = h264::sps(&high(true, None, true));
    assert!(matches!(e, Err(Error::Bad("H.264 scaling list"))), "{e:?}");
    let p = h264::pps(&pps(false)).unwrap();
    assert!(p.t8x8 && p.cabac && p.deblock_ctl);
    assert_eq!(p.chroma_qp2, -2);
    let e = h264::pps(&pps(true));
    assert!(matches!(e, Err(Error::Bad(_))), "{e:?}");
}

/// NAL units of an Annex B stream, without start codes, grouped into access units: a new one at each slice segment
/// that starts a picture (H.265 first_slice_segment_in_pic_flag); parameter sets go to `params` and to their unit.
fn units(b: &[u8], params: &mut Vec<Vec<u8>>) -> Vec<Vec<Vec<u8>>> {
    let mut nals = vec![];
    let (mut i, mut s) = (0, None);
    while i + 3 <= b.len() {
        if b[i..i + 3] == [0, 0, 1] {
            if let Some(s) = s {
                let mut e = i;
                while e > s && b[e - 1] == 0 {
                    e -= 1;
                }
                nals.push(b[s..e].to_vec());
            }
            (i, s) = (i + 3, Some(i + 3));
        } else {
            i += 1;
        }
    }
    if let Some(s) = s {
        nals.push(b[s..].to_vec());
    }
    let mut aus: Vec<Vec<Vec<u8>>> = vec![];
    let mut cur: Vec<Vec<u8>> = vec![];
    for n in nals {
        let t = h265::nal_type(&n);
        if t < 32 && n.len() > 2 && n[2] & 0x80 != 0 && cur.iter().any(|c| h265::nal_type(c) < 32) {
            aus.push(std::mem::take(&mut cur));
        }
        if (32..=34).contains(&t) {
            params.push(n.clone());
        }
        if t < 32 || (32..=34).contains(&t) {
            cur.push(n);
        }
    }
    aus.push(cur);
    aus
}

/// An open-GOP H.265 stream (libx265: a CRA every 10 frames, RASL pictures after it that reference the GOP before)
/// with a picture of the first GOP broken: the pictures after it are concealed until the next CRA, which gets a new
/// session; that session's RASL pictures, whose references it never had, are concealed without being decoded rather
/// than failing (or decoding garbage) every GOP from then on. The GOPs after decode clean. Skips without the ffmpeg
/// CLI or its libx265.
#[test]
fn open_gop() {
    let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!("x5desk-open-gop-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("open.hevc");
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=256x256:rate=30",
        ])
        .args(["-frames:v", "40", "-c:v", "libx265", "-pix_fmt", "yuv420p"])
        .args([
            "-x265-params",
            "keyint=10:min-keyint=10:open-gop=1:bframes=3:b-adapt=0:scenecut=0:log-level=error",
        ])
        .args(["-f", "hevc"])
        .arg(&p)
        .status()
        .is_ok_and(|s| s.success());
    let b = std::fs::read(&p).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    if !ok || b.is_empty() {
        eprintln!("ffmpeg with libx265 could not make the clip: skipped");
        return;
    }
    let mut params = vec![];
    let aus = units(&b, &mut params);
    let ty = |au: &[Vec<u8>]| {
        au.iter()
            .map(|n| h265::nal_type(n))
            .find(|&t| t < 32)
            .unwrap()
    };
    let cra: Vec<usize> = (0..aus.len()).filter(|&i| ty(&aus[i]) == 21).collect();
    let rasl = |a: usize| {
        (a + 1..aus.len())
            .take_while(|&i| matches!(ty(&aus[i]), 8 | 9))
            .count()
    };
    assert!(
        cra.len() >= 3 && rasl(cra[1]) > 0,
        "no open GOPs: CRAs at {cra:?}"
    );
    let g = Gpu::new().unwrap();
    let mut d = Decoder::new(&g, Codec::H265, &params, 256, 256).unwrap();
    // a trailing picture of the first GOP, cut short
    let bad = (1..cra[0].max(cra[1]))
        .find(|&i| matches!(ty(&aus[i]), 0 | 1) && i > 2)
        .unwrap();
    let mut failed = vec![];
    for (i, au) in aus.iter().enumerate() {
        let mut au: Vec<&[u8]> = au.iter().map(Vec::as_slice).collect();
        let cut;
        if i == bad {
            let s = au.iter().position(|n| h265::nal_type(n) < 32).unwrap();
            cut = au[s][..au[s].len().min(12)].to_vec();
            au[s] = &cut;
        }
        let before = d.failed;
        match d.decode(&g, &au) {
            Ok(f) => d.release(&g, f),
            Err(Error::Skip) => {}
            Err(e) => panic!("unit {i}: {e}"),
        }
        if d.failed > before {
            failed.push(i);
        }
    }
    let (c1, c2) = (
        cra.iter().find(|&&c| c > bad).copied().unwrap(),
        cra.iter().filter(|&&c| c > bad).nth(1),
    );
    println!("CRAs at {cra:?}, unit {bad} broken; concealed: {failed:?}");
    assert!(failed.contains(&bad), "{failed:?}");
    // the new session's RASL pictures are concealed, all of them
    assert!(
        (c1 + 1..=c1 + rasl(c1)).all(|i| failed.contains(&i)),
        "{failed:?}"
    );
    // and nothing from the CRA after
    if let Some(&c2) = c2 {
        assert!(failed.iter().all(|&i| i < c2), "{failed:?}");
    }
}
