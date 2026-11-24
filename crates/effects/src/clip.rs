//! Frames of an X5 clip for the effects that look at a whole clip first: both lenses decoded (Vulkan Video) and
//! stitched to an equirect (template seam, colour matching), one frame per submit, plus the trailer's gyro.

use glam::DQuat;
use media::{
    insv::{self, Exposure, Meta, Trailer},
    mp4::{self, Mp4},
};
use std::path::Path;
use stitch::{Calib, Frame, Opts, Pose, Stitcher};
use video::{Codec, Decoder, Gpu, decode::OUT};

/// An `.insv` with both lenses as its two video tracks.
pub struct Clip {
    file: std::path::PathBuf,
    mp4: Mp4,
    /// tracks of lens 0, lens 1
    lens: [usize; 2],
    /// Presentation time of each frame, s.
    pub times: Vec<f64>,
    pub meta: Meta,
    cal: Calib,
    /// Orientation from the IMU (body -> world) and each frame's exposure; None without IMU records.
    pub imu: Option<(stab::Track, Vec<Exposure>)>,
}

impl Clip {
    pub fn open(p: &Path) -> Result<Clip, String> {
        let e = |x: std::io::Error| format!("{}: {x}", p.display());
        let tr = Trailer::open(p).map_err(e)?;
        let meta = tr.meta.clone();
        let cal = Calib::parse(&meta.calib)?;
        let mp4 = Mp4::open(p).map_err(e)?;
        let v: Vec<usize> = (0..mp4.tracks.len())
            .filter(|&t| {
                matches!(
                    mp4.tracks[t].codec,
                    mp4::Codec::Hevc { .. } | mp4::Codec::Avc { .. }
                )
            })
            .collect();
        let [a, b, ..] = v[..] else {
            return Err(format!(
                "{}: needs both lenses as two video tracks",
                p.display()
            ));
        };
        // reversed dual track (#131 = 4): the first track holds lens 1
        let lens = if meta.stream == 4 { [b, a] } else { [a, b] };
        let t0 = &mp4.tracks[lens[0]];
        let n = t0.samples.len().min(mp4.tracks[lens[1]].samples.len());
        let d = t0.shift(mp4.timescale);
        let times = t0.samples[..n]
            .iter()
            .map(|x| (x.pts + d) as f64 / t0.timescale as f64)
            .collect();
        let (imu, ex) = (tr.imu().map_err(e)?, tr.frames(insv::EXPOSURE).map_err(e)?);
        let imu = (imu.len() >= 2 && !ex.is_empty())
            .then(|| (stab::Track::new(&imu, stab::KP, stab::KI), ex));
        Ok(Clip {
            file: p.into(),
            mp4,
            lens,
            times,
            meta,
            cal,
            imu,
        })
    }

    /// The frames both lenses have as sync samples (X5: every 2 s): `frames` decodes one of them alone.
    pub fn keys(&self) -> Vec<usize> {
        let tr = self.lens.map(|t| &self.mp4.tracks[t]);
        (0..self.times.len())
            .filter(|&k| tr.iter().all(|t| t.samples[k].sync))
            .collect()
    }

    /// Camera orientation (body -> world) when frame `k`'s centre row was exposed.
    pub fn orient(&self, k: usize) -> Option<DQuat> {
        let (tr, ex) = self.imu.as_ref()?;
        let e = ex.get(k.min(ex.len() - 1))?;
        Some(tr.at(e.t as f64 * 1e-6 - 0.5 * e.time + stab::OFFSET))
    }

    /// How far the camera turned over frames `ks`: the largest angle between frame orientations, degrees (gyro with
    /// gravity correction). A locked-off shot stays within a few hundredths of a degree plus the gyro's drift.
    pub fn turn(&self, ks: &[usize]) -> Option<f64> {
        let q: Vec<DQuat> = ks.iter().map(|&k| self.orient(k)).collect::<Option<_>>()?;
        let mut m = 0f64;
        for (i, a) in q.iter().enumerate() {
            for b in &q[i + 1..] {
                m = m.max(stab::log(a.inverse() * *b).length());
            }
        }
        Some(m.to_degrees())
    }

    /// Stabilised poses (`stab::poses`): with `Lock::Direction` all frames look into the same world directions;
    /// `Lock::Forward` follows the direction of travel of the clip's analysis if it ran (`export::travel`), else
    /// it is the horizon lock.
    pub fn poses(&self, lock: stab::Lock, smooth: f64) -> Option<Vec<stab::Pose>> {
        let (tr, ex) = self.imu.as_ref()?;
        let ahead = match lock {
            stab::Lock::Forward => export::travelled(std::slice::from_ref(&self.file)),
            _ => None,
        };
        let o = stab::Opts {
            ahead: ahead.as_deref().unwrap_or_default(),
            ..stab::Opts::new(&self.meta, lock, smooth)
        };
        Some(stab::poses(tr, ex, &o))
    }

    /// The stitch pose of frame `k`: `p` (stabilisation; None: the camera's own orientation), then the exporter's
    /// framing (the X5 frames on lens 1).
    pub fn pose(&self, k: usize, p: Option<&[stab::Pose]>) -> Pose {
        let f = Pose::euler(0.0, 0.0, if self.meta.stream == 4 { 180.0 } else { 0.0 });
        match p {
            Some(p) => {
                let p = p[k.min(p.len() - 1)];
                Pose {
                    q: p.q * f.q,
                    rs: [p.rs; 2],
                }
            }
            None => f,
        }
    }

    /// Frame `k` stitched with `pose(k, p)`: the rotation from its directions to the world's (Z up; the gyro with
    /// gravity correction, its heading drifting); None without IMU records.
    pub fn world(&self, k: usize, p: Option<&[stab::Pose]>) -> Option<glam::Quat> {
        Some(self.orient(k)?.as_quat() * self.pose(k, p).q)
    }

    /// Poses that register frames `ks` onto frame `ks[r]` by what they show: each frame stitched with `pose(k)` at
    /// 2048 × 1024, the rotation left between it and the reference fitted (`crate::align`, corners between polar angles
    /// `band`, rad) and folded into its pose. The gyro's direction lock leaves its drift and a hand's sway; this takes
    /// out what a rotation can (parallax from a camera that moved stays). A frame without a fit keeps `pose(k)`.
    pub fn register(
        &self,
        g: &Gpu,
        cx: &stitch::Ctx,
        ks: &[usize],
        r: usize,
        band: [f64; 2],
        pose: impl Fn(usize) -> Pose,
    ) -> Result<Vec<(Pose, Option<stab::vis::Fit>)>, String> {
        let (w, h) = (2048u32, 1024u32);
        let mut bs = vec![];
        self.frames(g, cx, w, false, ks, &pose, |f, _| {
            let b = cx.dev.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4 * (w * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut e = cx.dev.create_command_encoder(&Default::default());
            e.copy_texture_to_buffer(
                f.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &b,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * w),
                        rows_per_image: None,
                    },
                },
                f.size(),
            );
            cx.q.submit([e.finish()]);
            bs.push(b);
            Ok(())
        })?;
        let p: Vec<stab::vis::Pano> = bs
            .iter()
            .map(|b| {
                b.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
                cx.dev
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(|e| e.to_string())?;
                let v = stab::vis::Pano::rgba(
                    &b.get_mapped_range(..).expect("mapped"),
                    w as usize,
                    h as usize,
                );
                b.unmap();
                Ok(v)
            })
            .collect::<Result<_, String>>()?;
        Ok(ks
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let (p0, f) = (
                    pose(k),
                    (i != r).then(|| crate::align(&p[r], &p[i], band)).flatten(),
                );
                // content at reference ray a shows at q·a in this frame: P' a = P q a
                let q = f.as_ref().map_or(glam::Quat::IDENTITY, |f| f.q.as_quat());
                (Pose { q: p0.q * q, ..p0 }, f)
            })
            .collect())
    }

    /// Decodes up to the last of frames `ks` (ascending) and stitches each of them to a `w` × `w / 2` equirect
    /// (`Rgba16Float` with `wide`, else `Rgba8Unorm`; sampled, copied, storage, render target) with `pose(k)`
    /// (`Clip::pose`), then calls `f(frame, k)` with the stitch submitted. The texture is the same for every frame:
    /// what `f` submits runs before the next stitch overwrites it.
    #[allow(clippy::too_many_arguments)]
    pub fn frames(
        &self,
        g: &Gpu,
        cx: &stitch::Ctx,
        w: u32,
        wide: bool,
        ks: &[usize],
        pose: impl Fn(usize) -> Pose,
        mut f: impl FnMut(&wgpu::Texture, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        let (Some(&k0), Some(&k1)) = (ks.first(), ks.last()) else {
            return Ok(());
        };
        let tr = self.lens.map(|t| &self.mp4.tracks[t]);
        if k1 >= self.times.len() {
            return Err(format!("frame {k1} of {}", self.times.len()));
        }
        let mut d = vec![];
        for t in tr {
            let (c, ps) = match &t.codec {
                mp4::Codec::Hevc { ps, .. } => (Codec::H265, ps),
                mp4::Codec::Avc { ps, .. } => (Codec::H264, ps),
                _ => unreachable!("Clip::open takes video tracks"),
            };
            d.push(Decoder::new(g, c, ps, t.width, t.height).map_err(|e| format!("decoder: {e}"))?);
        }
        let mut d: [Decoder; 2] = d.try_into().map_err(|_| "decoders")?;
        let fr = Frame {
            w: d[0].view[2],
            h: d[0].view[3],
            crop: self.meta.crop,
            shift: self.meta.shift,
        };
        let o = Opts {
            blend: self.meta.blend.unwrap_or(6.0),
            wide,
            ..Opts::default()
        };
        let st = Stitcher::new(cx, &self.cal, fr, w, w / 2, o)?;
        let src: [Vec<stitch::Src>; 2] = [0, 1].map(|l| {
            (0..OUT as u64)
                .map(|s| cx.rect(&d[l].planes(s), d[l].view))
                .collect()
        });
        let nal = tr.map(|t| match t.codec {
            mp4::Codec::Hevc { nal, .. } | mp4::Codec::Avc { nal, .. } => nal as usize,
            _ => 4,
        });
        // decoding starts at the last frame at or before the first where both lenses have a sync sample
        let key = (0..=k0)
            .rev()
            .find(|&k| tr.iter().all(|t| t.samples[k].sync))
            .unwrap_or(0);
        let (mut buf, mut i) = (vec![], 0);
        for k in key..=k1 {
            let mut fs = [0; 2];
            for l in 0..2 {
                self.mp4
                    .read(&tr[l].samples[k], &mut buf)
                    .map_err(|e| format!("frame {k}: {e}"))?;
                let mut au = [&[][..]; 64];
                let n = nals(&buf, nal[l], &mut au);
                fs[l] = loop {
                    match d[l].decode(g, &au[..n]) {
                        Ok(x) => break x,
                        Err(video::Error::Full) => {}
                        Err(e) => return Err(format!("decode frame {k}, lens {l}: {e}")),
                    }
                };
            }
            let hit = ks.get(i) == Some(&k);
            if hit {
                while ks.get(i) == Some(&k) {
                    i += 1;
                }
                st.pose(&pose(k));
                let mut enc = cx.dev.create_command_encoder(&Default::default());
                st.encode(
                    &mut enc,
                    [0, 1].map(|l| &src[l][(fs[l] % OUT as u64) as usize]),
                );
                for (x, f) in d.iter().zip(fs) {
                    x.ring.acquire(g, f);
                }
                cx.q.submit([enc.finish()]);
            }
            for (x, f) in d.iter().zip(fs) {
                x.ring.release(g, f);
            }
            if hit {
                f(&st.out, k)?;
            }
        }
        Ok(())
    }
}

/// Splits a sample's length-prefixed NAL units (`n`-byte lengths) into `out`; returns how many.
fn nals<'a>(mut b: &'a [u8], n: usize, out: &mut [&'a [u8]]) -> usize {
    let mut c = 0;
    while b.len() > n && c < out.len() {
        let l = b[..n].iter().fold(0usize, |a, &x| a << 8 | x as usize);
        let Some(x) = b.get(n..n + l) else { break };
        out[c] = x;
        c += 1;
        b = &b[n + l..];
    }
    c
}
