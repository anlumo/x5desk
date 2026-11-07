//! What a host checks of the device before its first job (the MCP server's GPU self-test): an exact decode of a
//! hand-made H.264 High stream (`pcm`), whether the X5's H.265 decodes at its size (`h265`), what the device decodes
//! and encodes per profile (`caps`), and one picture through each encoder (`encode`). The test streams (`sets`,
//! `flat`, `skip`, `broken`) are I_PCM pictures of one luma value each, so every picture decodes exactly; crate
//! export's synthetic recording and the concealment tests use them too.
#[cfg(target_os = "linux")]
use crate::decode;
#[cfg(target_os = "macos")]
use crate::mac::sys;
use crate::{Cfg, Codec, Decoder, Encoder, Error, Gpu, Packet, Rate, bitw::W};
#[cfg(target_os = "linux")]
use ash::vk;
use std::time::Duration;
#[cfg(target_os = "linux")]
use std::{ffi::c_void, ptr};

/// How long a readback or the encode probe's packet may take.
const WAIT: Duration = Duration::from_secs(5);

/// The profiles `caps` answers for, as (codec, 10-bit): H.265 Main, H.265 Main 10, H.264 High.
pub const PROFILES: [(Codec, bool); 3] = [
    (Codec::H265, false),
    (Codec::H265, true),
    (Codec::H264, false),
];

/// What the device decodes or encodes in one profile.
#[derive(Clone, Debug, PartialEq)]
pub struct Cap {
    pub codec: Codec,
    /// H.265 Main 10
    pub deep: bool,
    /// the coded extents, w × h
    pub min: [u32; 2],
    pub max: [u32; 2],
    /// encode: the rate control modes, of "cbr", "vbr" and "cqp" (constant QP); empty for decode
    pub rate: Vec<&'static str>,
    /// encode: the highest bitrate, bit/s; None: not reported (macOS)
    pub bps: Option<u64>,
    /// encode: the QP range
    pub qp: Option<[u8; 2]>,
    /// decode, Linux: whether the driver lists NV12 / P010 among its decode output formats (for the log:
    /// `Decoder::new` doesn't ask); None: not asked
    pub fmt: Option<bool>,
}

/// SPS and PPS of a `side` × `side` stream (`side` a multiple of 16): High (`high`) or Constrained Baseline, the
/// lowest level that holds the picture, POC type 2, one reference frame, CAVLC.
pub fn sets(side: u32, high: bool) -> Vec<Vec<u8>> {
    let mbs = side / 16;
    let level = [(1620, 30), (8192, 40), (36864, 51)]
        .into_iter()
        .find(|l| mbs * mbs <= l.0)
        .map_or(62, |l| l.1);
    // Constrained Baseline (constraint_set0 and 1), or High without constraint flags
    let (idc, cs) = [(66, 0xc0), (100, 0)][high as usize];
    let mut s = W::default();
    s.u(8, 0x67).u(8, idc).u(8, cs).u(8, level).ue(0);
    if high {
        // 4:2:0, 8 bits, no transform bypass, no scaling lists
        s.ue(1).ue(0).ue(0).f(false).f(false);
    }
    // 4-bit frame_num, POC type 2, one reference frame, no gaps; frames only, direct 8x8 inference, no cropping or VUI
    s.ue(0).ue(2).ue(1).f(false);
    s.ue(mbs - 1).ue(mbs - 1).f(true).f(true).f(false).f(false);
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

/// An IDR (`idr_pic_id` `id`) whose luma is `y` all over, chroma 128.
pub fn flat(side: u32, id: u32, y: u8) -> Vec<u8> {
    let mut w = idr(id);
    for _ in 0..(side / 16) * (side / 16) {
        w.ue(25).align();
        (0..256).for_each(|_| _ = w.u(8, y as u32));
        (0..128).for_each(|_| _ = w.u(8, 128));
    }
    w.nal()
}

/// An IDR VideoToolbox fails on (kVTVideoDecoderBadDataErr): its first macroblock cut short.
pub fn broken(id: u32) -> Vec<u8> {
    let mut w = idr(id);
    w.ue(25).align();
    (0..100).for_each(|_| _ = w.u(8, 9));
    w.nal()
}

/// A reference P picture, frame_num `n`, every macroblock skipped: the picture before it again.
pub fn skip(side: u32, n: u32) -> Vec<u8> {
    let mut w = W::default();
    w.u(8, 0x41).ue(0).ue(5).ue(0).u(4, n);
    w.u(1, 0).u(1, 0).u(1, 0).se(0);
    w.ue((side / 16) * (side / 16)).nal()
}

/// The pixel check at `side` × `side`: three IDRs in High (luma 16, 128 and 235), each followed by a skipped P
/// picture, more frames than the ring has slots. Every luma sample read back must be its IDR's, and no picture
/// concealed (VideoToolbox returns a concealed one `Ok`). Ok(Some(why)): `Decoder::new` refused the stream, so the
/// check was skipped (`h265` checks the X5's own codec).
pub fn pcm(g: &Gpu, side: u32) -> Result<Option<String>, String> {
    let mut d = match Decoder::new(g, Codec::H264, &sets(side, true), side, side) {
        Ok(d) => d,
        Err(e) => return Ok(Some(format!("H.264 High: {e}"))),
    };
    for (k, y) in [16, 128, 235].into_iter().enumerate() {
        for au in [flat(side, k as u32, y), skip(side, 1)] {
            let f = d
                .decode(g, &[&au])
                .map_err(|e| format!("frame {}: {e}", d.frame))?;
            let px = luma(g, &d, side, f);
            d.release(g, f);
            let px = px.map_err(|e| format!("frame {f}: {e}"))?;
            let (lo, hi) = px.iter().fold((255, 0), |(l, h), &v| (v.min(l), v.max(h)));
            if (lo, hi) != (y, y) {
                let mean = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
                return Err(format!(
                    "frame {f}: luma {lo}..={hi}, mean {mean:.2}, not {y}"
                ));
            }
        }
    }
    match d.failed() {
        0 => Ok(None),
        n => Err(format!("{n} of 6 pictures concealed")),
    }
}

/// Frame `f`'s luma plane, `side` × `side`; the image goes back home after (Linux), as the decoder wants it.
fn luma(g: &Gpu, d: &Decoder, side: u32, f: u64) -> Result<Vec<u8>, String> {
    let bpr = side.next_multiple_of(256);
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("check"),
        size: (bpr * side) as u64,
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
                bytes_per_row: Some(bpr),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: side,
            height: side,
            depth_or_array_layers: 1,
        },
    );
    let home = d.textures(f).map(|texture| wgpu::TextureTransition {
        texture,
        selector: None,
        state: wgpu::TextureUses::RESOURCE,
    });
    e.transition_resources(std::iter::empty(), home);
    d.acquire(g, f);
    let i = g.queue.submit([e.finish()]);
    let sl = buf.slice(..);
    sl.map_async(wgpu::MapMode::Read, |_| {});
    g.device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(i),
            timeout: Some(WAIT),
        })
        .map_err(|e| format!("readback: {e}"))?;
    let m = sl
        .get_mapped_range()
        .map_err(|e| format!("readback: {e:?}"))?;
    let s = side as usize;
    Ok((0..s)
        .flat_map(|r| &m[r * bpr as usize..][..s])
        .copied()
        .collect())
}

/// Whether the X5's H.265 (Main, or Main 10 when `ten_bit`) decodes at `side` × `side`, from the driver's
/// capabilities and without a session: the decode queue's codecs, the output format, the DPB mode (`Decoder::new`
/// wants DPB and output images apart) and the coded extent.
#[cfg(target_os = "linux")]
pub fn h265(g: &Gpu, side: u32, ten_bit: bool) -> Result<(), String> {
    let p = ["Main", "Main 10"][ten_bit as usize];
    if !g
        .dec
        .ops
        .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
    {
        return Err(format!("its decode queue has no H.265 ({:?})", g.dec.ops));
    }
    let tf = texture(ten_bit);
    if !g.device.features().contains(tf.required_features()) {
        return Err(format!("wgpu lacks {tf:?} textures"));
    }
    let (c, dc) = profile(Codec::H265, ten_bit, false, |pr| unsafe {
        decode::caps(g, pr, true)
    })
    .map_err(|e| format!("H.265 {p}: {e}"))?;
    if !decode::distinct(&c, dc) {
        return Err(format!(
            "H.265 {p}: no distinct DPB and output images (decode {dc:?}, {:?}): Intel's coinciding mode isn't \
             supported",
            c.flags
        ));
    }
    let m = c.max_coded_extent;
    if side > m.width || side > m.height {
        return Err(format!(
            "H.265 {p} decodes up to {} × {}, not {side} × {side}",
            m.width, m.height
        ));
    }
    Ok(())
}

/// macOS: whether VideoToolbox has a hardware H.265 decoder, and `side` within `caps`' table; it has no query for
/// sizes or Main 10 (every Apple silicon Mac decodes both).
#[cfg(target_os = "macos")]
pub fn h265(g: &Gpu, side: u32, ten_bit: bool) -> Result<(), String> {
    if unsafe { sys::VTIsHardwareDecodeSupported(sys::kCMVideoCodecType_HEVC) } == 0 {
        return Err("VideoToolbox has no hardware H.265 decoder".into());
    }
    let c = caps(g).0;
    let m = c
        .iter()
        .find(|c| c.codec == Codec::H265 && c.deep == ten_bit)
        .map_or([0; 2], |c| c.max);
    match side > m[0] || side > m[1] {
        true => Err(format!(
            "H.265 decodes up to {} × {}, not {side} × {side}",
            m[0], m[1]
        )),
        false => Ok(()),
    }
}

/// What the device decodes and encodes, per profile of `PROFILES` it has: the driver's capabilities, without a
/// session. Decode lists what `Decoder::new` can use (DPB and output images apart); encode what the encode queue
/// has, which `encode` then tries.
#[cfg(target_os = "linux")]
pub fn caps(g: &Gpu) -> (Vec<Cap>, Vec<Cap>) {
    use vk::VideoCodecOperationFlagsKHR as Op;
    let ext = |e: vk::Extent2D| [e.width, e.height];
    let dec = PROFILES
        .into_iter()
        .filter_map(|(codec, deep)| {
            let op = [Op::DECODE_H264, Op::DECODE_H265][(codec == Codec::H265) as usize];
            let tf = texture(deep);
            if !g.dec.ops.contains(op) || !g.device.features().contains(tf.required_features()) {
                return None;
            }
            profile(codec, deep, false, |pr| unsafe {
                let (c, dc) = decode::caps(g, pr, codec == Codec::H265).ok()?;
                let want = [vk::Format::G8_B8R8_2PLANE_420_UNORM, P010][deep as usize];
                decode::distinct(&c, dc).then(|| Cap {
                    codec,
                    deep,
                    min: ext(c.min_coded_extent),
                    max: ext(c.max_coded_extent),
                    rate: vec![],
                    bps: None,
                    qp: None,
                    fmt: Some(
                        formats(g, pr, vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR).contains(&want),
                    ),
                })
            })
        })
        .collect();
    let Some(q) = &g.enc else {
        return (dec, vec![]);
    };
    let enc = PROFILES
        .into_iter()
        .filter(|&(codec, _)| {
            q.ops
                .contains([Op::ENCODE_H264, Op::ENCODE_H265][(codec == Codec::H265) as usize])
        })
        .filter_map(|(codec, deep)| {
            profile(codec, deep, true, |pr| unsafe {
                use vk::VideoEncodeRateControlModeFlagsKHR as M;
                let (mut c5, mut c4) = (
                    vk::VideoEncodeH265CapabilitiesKHR::default(),
                    vk::VideoEncodeH264CapabilitiesKHR::default(),
                );
                let mut ec = vk::VideoEncodeCapabilitiesKHR::default();
                let mut c = vk::VideoCapabilitiesKHR::default();
                ec.p_next = match codec {
                    Codec::H265 => &mut c5 as *mut _ as *mut c_void,
                    Codec::H264 => &mut c4 as *mut _ as *mut c_void,
                };
                c.p_next = &mut ec as *mut _ as *mut c_void;
                (g.vqi.fp().get_physical_device_video_capabilities_khr)(g.pd, pr, &mut c)
                    .result()
                    .ok()?;
                let qp = match codec {
                    Codec::H265 => [c5.min_qp, c5.max_qp],
                    Codec::H264 => [c4.min_qp, c4.max_qp],
                };
                let rate = [(M::CBR, "cbr"), (M::VBR, "vbr"), (M::DISABLED, "cqp")]
                    .into_iter()
                    .filter(|m| ec.rate_control_modes.contains(m.0))
                    .map(|m| m.1)
                    .collect();
                Some(Cap {
                    codec,
                    deep,
                    min: ext(c.min_coded_extent),
                    max: ext(c.max_coded_extent),
                    rate,
                    bps: Some(ec.max_bitrate),
                    qp: Some(qp.map(|q| q.clamp(0, 255) as u8)),
                    fmt: None,
                })
            })
        })
        .collect();
    (dec, enc)
}

/// macOS: VideoToolbox has no such query, so a fixed table for the hardware decoders it has: from 16 × 16 (both
/// ways on the M4) up to the app's limits (H.264 4096, H.265 8192: crate app's `H264_MAX`, `H265_MAX`); encode in
/// the rate modes `Encoder::new` sets, QP 0-51.
#[cfg(target_os = "macos")]
pub fn caps(_: &Gpu) -> (Vec<Cap>, Vec<Cap>) {
    let cap = |(codec, deep): (Codec, bool), enc: bool| Cap {
        codec,
        deep,
        min: [16; 2],
        max: [[4096, 8192][(codec == Codec::H265) as usize]; 2],
        rate: match enc {
            true => vec!["cbr", "vbr", "cqp"],
            false => vec![],
        },
        bps: None,
        qp: enc.then_some([0, 51]),
        fmt: None,
    };
    let hw = |c: Codec| {
        let t =
            [sys::kCMVideoCodecType_H264, sys::kCMVideoCodecType_HEVC][(c == Codec::H265) as usize];
        unsafe { sys::VTIsHardwareDecodeSupported(t) != 0 }
    };
    (
        PROFILES
            .into_iter()
            .filter(|p| hw(p.0))
            .map(|p| cap(p, false))
            .collect(),
        PROFILES.into_iter().map(|p| cap(p, true)).collect(),
    )
}

/// The encode probe: `Encoder::new` for `codec` (H.265 Main 10 when `deep`) at 640 × 360 with a file's CBR, and one
/// mid-grey picture through the input and the encoder, out as a packet within 5 s.
pub fn encode(g: &Gpu, codec: Codec, deep: bool) -> Result<(), String> {
    let (w, h) = (640, 360);
    let c = Cfg {
        codec,
        deep,
        w,
        h,
        fps: [30, 1],
        gop: 30,
        rate: Rate::Cbr(2_000_000),
        vbv: 1000,
        quality: 0,
        full: false,
        chroma: 0,
    };
    let s = |e: Error| e.to_string();
    let mut enc = Encoder::new(g, &c).map_err(s)?;
    let pitch = w * (1 + deep as u32);
    let n = (pitch * (h + h / 2)) as usize;
    let px = match deep {
        // 512 in the high 10 bits
        true => [0u8, 0x80].repeat(n / 2),
        false => vec![128; n],
    };
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("check"),
        size: n as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    g.queue.write_buffer(&buf, 0, &px);
    enc.input.acquire(g, 0).map_err(s)?;
    let cb = enc.input.copy_buffer(g, &buf, 0, pitch, 0).map_err(s)?;
    g.queue.submit([cb]);
    enc.input.release(g, 0);
    enc.encode(g, 0).map_err(s)?;
    let mut p = Packet::default();
    match enc.packet(&mut p, WAIT.as_nanos() as u64).map_err(s)? {
        true if !p.data.is_empty() => Ok(()),
        true => Err("an empty packet".into()),
        false => Err(format!("no packet within {WAIT:?}")),
    }
}

#[cfg(target_os = "linux")]
const P010: vk::Format = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;

/// The decoder's texture format: NV12, or P010 when `deep`.
#[cfg(target_os = "linux")]
fn texture(deep: bool) -> wgpu::TextureFormat {
    [wgpu::TextureFormat::NV12, wgpu::TextureFormat::P010][deep as usize]
}

/// Calls `f` with `codec`'s 4:2:0 profile (H.265 Main, Main 10 when `deep`; H.264 High, progressive) for decode, or
/// for encode (`enc`), as `Decoder::new` and `Encoder::new` make them.
#[cfg(target_os = "linux")]
fn profile<R>(
    codec: Codec,
    deep: bool,
    enc: bool,
    f: impl FnOnce(&vk::VideoProfileInfoKHR) -> R,
) -> R {
    use vk::VideoCodecOperationFlagsKHR as Op;
    let bits = [
        vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
        vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
    ][deep as usize];
    let idc = match codec {
        Codec::H265 => 1 + deep as u32,
        Codec::H264 => 100,
    };
    let mut d5 = vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(idc);
    let mut d4 = vk::VideoDecodeH264ProfileInfoKHR::default()
        .std_profile_idc(idc)
        .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
    let mut e5 = vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(idc);
    let mut e4 = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(idc);
    let pr = vk::VideoProfileInfoKHR::default()
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(bits)
        .chroma_bit_depth(bits);
    let pr = match (codec, enc) {
        (Codec::H265, false) => pr.video_codec_operation(Op::DECODE_H265).push_next(&mut d5),
        (Codec::H264, false) => pr.video_codec_operation(Op::DECODE_H264).push_next(&mut d4),
        (Codec::H265, true) => pr.video_codec_operation(Op::ENCODE_H265).push_next(&mut e5),
        (Codec::H264, true) => pr.video_codec_operation(Op::ENCODE_H264).push_next(&mut e4),
    };
    f(&pr)
}

/// The image formats the driver takes for `usage` in profile `pr`; none when the query fails.
#[cfg(target_os = "linux")]
unsafe fn formats(
    g: &Gpu,
    pr: &vk::VideoProfileInfoKHR,
    usage: vk::ImageUsageFlags,
) -> Vec<vk::Format> {
    let list = vk::VideoProfileListInfoKHR::default().profiles(std::slice::from_ref(pr));
    let mut fi = vk::PhysicalDeviceVideoFormatInfoKHR::default().image_usage(usage);
    fi.p_next = &list as *const _ as *const c_void;
    let fp = g.vqi.fp().get_physical_device_video_format_properties_khr;
    let (mut n, mut v) = (0, vec![]);
    if unsafe { fp(g.pd, &fi, &mut n, ptr::null_mut()) } != vk::Result::SUCCESS {
        return vec![];
    }
    v.resize(n as usize, vk::VideoFormatPropertiesKHR::default());
    if unsafe { fp(g.pd, &fi, &mut n, v.as_mut_ptr()) } != vk::Result::SUCCESS {
        return vec![];
    }
    v.iter().take(n as usize).map(|p| p.format).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264;
    use std::sync::Mutex;

    /// GPU tests run one at a time: each opens its own device.
    static GPU: Mutex<()> = Mutex::new(());

    /// The streams read back as written: High (or Baseline) at the size and level asked, the pictures' NAL types.
    #[test]
    fn streams() {
        for (side, high, idc, level) in [
            (256, true, 100, 30),
            (64, false, 66, 30),
            (1024, true, 100, 40),
        ] {
            let p = sets(side, high);
            let s = h264::sps(&p[0]).unwrap();
            assert_eq!(
                (
                    s.profile_idc,
                    s.width_mbs,
                    s.height_mbs,
                    s.max_num_ref_frames
                ),
                (idc, side / 16, side / 16, 1)
            );
            assert_eq!(p[0][3], level);
            h264::pps(&p[1]).unwrap();
        }
        assert_eq!(
            [flat(256, 0, 16)[0], skip(256, 1)[0], broken(1)[0]].map(|b| b & 31),
            [5, 1, 5]
        );
    }

    /// The pixel check passes at the self-test's size and at one whose rows aren't a multiple of 256 bytes.
    #[test]
    fn pcm() {
        let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let g = Gpu::new().unwrap();
        for side in [256, 400] {
            assert_eq!(super::pcm(&g, side), Ok(None), "{side} × {side}");
        }
    }

    /// The X5's 3840 × 3840 decodes in Main and Main 10; a picture past the device's extent is refused, naming it.
    #[test]
    fn h265() {
        let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let g = Gpu::new().unwrap();
        for deep in [false, true] {
            assert_eq!(super::h265(&g, 3840, deep), Ok(()), "10-bit {deep}");
        }
        let e = super::h265(&g, 1 << 16, false).unwrap_err();
        println!("{e}");
        assert!(e.contains("up to"), "{e}");
    }

    /// Every profile decodes, Main 10 included, at the X5's size and the pixel check's; each encode profile has rate
    /// modes and a QP range, and its probe encodes.
    #[test]
    fn caps() {
        let _l = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let g = Gpu::new().unwrap();
        let (dec, enc) = super::caps(&g);
        for c in dec.iter().chain(&enc) {
            println!("{c:?}");
        }
        let has = |c: &[Cap], p: (Codec, bool)| c.iter().any(|c| (c.codec, c.deep) == p);
        assert!(PROFILES.iter().all(|&p| has(&dec, p)), "{dec:?}");
        for c in &dec {
            let x5 = [1920, 3840][(c.codec == Codec::H265) as usize];
            assert!(
                c.min[0].max(c.min[1]) <= 256 && c.max[0].min(c.max[1]) >= x5,
                "{c:?}"
            );
        }
        #[cfg(target_os = "linux")]
        assert!(g.enc.is_none() || !enc.is_empty());
        #[cfg(target_os = "macos")]
        assert_eq!(enc.len(), 3);
        for c in &enc {
            let qp = c.qp.unwrap();
            let fits = c.min[0].max(c.min[1]) <= 360 && c.max[0].min(c.max[1]) >= 1920;
            assert!(!c.rate.is_empty() && qp[0] <= qp[1] && fits, "{c:?}");
            let t = std::time::Instant::now();
            assert_eq!(encode(&g, c.codec, c.deep), Ok(()), "{c:?}");
            println!("{:?} 10-bit {}: probe {:?}", c.codec, c.deep, t.elapsed());
        }
    }
}
