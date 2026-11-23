//! New Sky: the sky found by a segmentation network (EfficientViT-seg B1 on ADE20K, `seg`) on perspective tiles fixed
//! to the world, kept in an equirect of the world where it is smoothed over time, its edge fitted to each frame by a
//! guided filter (He et al., "Guided Image Filtering", colour guide: coefficients at the mask's resolution, applied to
//! the frame's own pixels), and a panorama fixed to the world in its place, its colour pulled towards the sky it
//! replaces.
//!
//! Frame k's network runs at frame k + 1 (it needs frame k submitted): the mask lags a frame, which the world frame
//! takes the camera's turning out of; what moves in the world (parallax on a ride) is a frame late before the guided
//! filter snaps the edge to the picture.

use crate::{Ctx, fi, run, seg, tv};
use glam::Quat;

/// The world mask's width (height half): 2.8 px/deg, the network's output is about 0.25–1 px/deg (48 px over 120°).
pub const MW: u32 = 1024;
/// Half-life of the colour match, s: a sky that comes into view changes the gain over a second, not a frame.
const GAIN_HALF: f64 = 1.0;

/// The pass. One `push` per submit. Memory: two world masks (4 MB each) and the coefficients (8 MB each), a frame copy,
/// the network's input and output (6 tiles: 10.6 MB and 55 kB, plus burn's own) and the panorama.
pub struct Sky {
    cx: Ctx,
    net: seg::Net,
    /// The tiles, in the world (`track::ring`).
    cams: wgpu::Buffer,
    nb: wgpu::Buffer,
    prob: wgpu::Buffer,
    ub: wgpu::Buffer,
    gain: wgpu::Buffer,
    /// World masks (probability, the frame's colour there), written in turn; the guided filter's coefficients, then
    /// their mean (with the probability).
    m: [wgpu::Texture; 2],
    ab: [wgpu::Texture; 2],
    pano: wgpu::Texture,
    /// skyin, skym, skya, skyb, skyg; skyo per frame format
    pp: [wgpu::ComputePipeline; 5],
    out: [wgpu::ComputePipeline; 2],
    /// skya, skyg per mask written; skyb
    bg: [wgpu::BindGroup; 5],
    /// The frame copy and the bind groups reading it (skyin, skym per mask written), the composite's (on the frame).
    tmp: Option<(wgpu::Texture, [wgpu::BindGroup; 3])>,
    ob: crate::Cache,
    /// Probability strides (tile, row, column) and side.
    st: [u32; 4],
    /// Half-life of the mask's memory, s (0: none).
    pub half: f32,
    /// How far the panorama's colour moves to the replaced sky's (per channel, in ratio): 0 none, 1 all the way.
    pub tone: f32,
    /// The first frame's heading, taken out of every frame's turn: the panorama's centre faces the first frame's.
    y0: Option<Quat>,
    /// Masks written; the time of the last; a network run waiting for its frame's submit.
    n: u64,
    last: Option<f64>,
    pending: bool,
}

impl Sky {
    /// The sky picture `pano` (an equirect, RGBA8 sRGB, `pw` × `ph`, halved until it fits the device) in place of the
    /// sky; the network on `g`'s device through `stitch::attach` (shared with RAFT and crate track).
    pub fn new(g: &video::Gpu, cx: &Ctx, pano: &[u8], pw: u32, ph: u32) -> Result<Self, String> {
        use wgpu::{BufferUsages as B, TextureUsages as U};
        let bd = stitch::attach(&g.instance, &g.adapter, &g.device, &g.queue)?;
        let (mut px, mut w, mut h) = (pano.to_vec(), pw, ph);
        if px.len() != (4 * w * h) as usize || w < 2 || h < 2 {
            return Err(format!("sky picture: {} bytes for {w} × {h}", px.len()));
        }
        while w.max(h) > cx.dev.limits().max_texture_dimension_2d {
            let (nw, nh) = (w / 2, h / 2);
            px = (0..nw * nh * 4)
                .map(|i| {
                    let (c, x, y) = (i % 4, i / 4 % nw, i / 4 / nw);
                    let p = |dx, dy| px[(4 * ((2 * y + dy) * w + 2 * x + dx) + c) as usize] as u32;
                    ((p(0, 0) + p(1, 0) + p(0, 1) + p(1, 1) + 2) / 4) as u8
                })
                .collect();
            (w, h) = (nw, nh);
        }
        let pt = cx.tex(
            w,
            h,
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            U::TEXTURE_BINDING | U::COPY_DST,
        );
        cx.q.write_texture(
            pt.as_image_copy(),
            &px,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: None,
            },
            crate::whole(&pt),
        );
        let cams = cx.buf(64 * seg::TILES as u64, B::STORAGE | B::COPY_DST);
        let v = |v: glam::Vec3| [v.x, v.y, v.z, 0.0];
        let c: Vec<[[f32; 4]; 4]> = track::ring(seg::TILES as usize, 120.0, 35.0)
            .iter()
            .map(|c| [v(c.f), v(c.r), v(c.u), [c.ext.x, c.ext.y, 0.0, 0.0]])
            .collect();
        cx.q.write_buffer(&cams, 0, bytemuck::cast_slice(&c));
        let (s, o) = (seg::SKY as u64, seg::SKY_OUT as u64);
        let m = [0, 1].map(|_| {
            cx.tex(
                MW,
                MW / 2,
                1,
                wgpu::TextureFormat::Rgba16Float,
                U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC,
            )
        });
        let ab = [0, 1].map(|_| {
            cx.tex(
                MW,
                MW / 2,
                1,
                wgpu::TextureFormat::Rgba32Float,
                U::STORAGE_BINDING | U::TEXTURE_BINDING,
            )
        });
        let (ub, gain) = (
            cx.buf(96, B::UNIFORM | B::COPY_DST),
            cx.buf(16, B::STORAGE | B::COPY_DST),
        );
        let pp = ["skyin", "skym", "skya", "skyb", "skyg"].map(|e| cx.pipe(e, 0));
        use wgpu::BindingResource as R;
        let mut bg: Vec<_> = (m.iter())
            .map(|t| {
                cx.bind(
                    &pp[2],
                    &[
                        (35, R::TextureView(&tv(t))),
                        (36, R::TextureView(&tv(&ab[0]))),
                    ],
                )
            })
            .collect();
        bg.extend(m.iter().map(|t| {
            cx.bind(
                &pp[4],
                &[
                    (29, ub.as_entire_binding()),
                    (39, R::TextureView(&tv(t))),
                    (40, R::TextureView(&tv(&pt))),
                    (41, R::Sampler(&cx.smp)),
                    (42, gain.as_entire_binding()),
                ],
            )
        }));
        bg.push(cx.bind(
            &pp[3],
            &[
                (37, R::TextureView(&tv(&ab[0]))),
                (38, R::TextureView(&tv(&ab[1]))),
            ],
        ));
        Ok(Self {
            net: seg::Net::new(&bd, true)?,
            nb: cx.buf(4 * seg::TILES as u64 * 3 * s * s, B::STORAGE | B::COPY_SRC),
            // the sky's slice of the output may be a view into all 150 classes
            prob: cx.buf(
                4 * seg::TILES as u64 * 150 * o * o,
                B::STORAGE | B::COPY_DST | B::COPY_SRC,
            ),
            out: [0, 1].map(|f| cx.pipe("skyo", f)),
            bg: bg.try_into().map_err(|_| "bind groups")?,
            cams,
            ub,
            gain,
            m,
            ab,
            pano: pt,
            pp,
            tmp: None,
            ob: None,
            st: [0; 4],
            half: 0.1,
            tone: 0.6,
            y0: None,
            n: 0,
            last: None,
            pending: false,
            cx: cx.clone(),
        })
    }

    /// Records the pass on frame `f` (an equirect in a `FORMATS` format; `COPY_SRC`, `STORAGE_BINDING`), its directions
    /// turned into the world's (Z up) by `q` (`Clip::world`), at `t` s: the sky found so far replaced, this frame's
    /// tiles queued for the network, which runs at the next `push` (its work submitted before `enc`). The first frame
    /// is passed through, so push it once before (stitched alike) to have its sky found when it is pushed again. A
    /// failing network run is an error.
    pub fn push(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        f: &wgpu::Texture,
        q: Quat,
        t: f64,
    ) -> Result<(), String> {
        use wgpu::{BindingResource as R, TextureUsages as U};
        let fresh = std::mem::take(&mut self.pending);
        if fresh {
            let (r, st) = self.net.run()?;
            let o = seg::SKY_OUT;
            let span = 4 * (st[0] * (seg::TILES - 1) + (st[2] + st[3]) * (o - 1) + 1) as u64;
            if span > self.prob.size() {
                return Err(format!("sky network output strides {st:?}"));
            }
            enc.copy_buffer_to_buffer(&r.buffer, r.offset, &self.prob, 0, span);
            self.st = [st[0], st[2], st[3], o];
        }
        if self
            .tmp
            .as_ref()
            .is_none_or(|x| x.0.size() != f.size() || x.0.format() != f.format())
        {
            let cx = &self.cx;
            let tt = cx.tex(
                f.width(),
                f.height(),
                1,
                f.format(),
                U::TEXTURE_BINDING | U::COPY_DST,
            );
            let v = tv(&tt);
            let src = [
                (16, R::TextureView(&v)),
                (17, R::Sampler(&cx.smp)),
                (29, self.ub.as_entire_binding()),
            ];
            let bi = cx.bind(
                &self.pp[0],
                &[
                    src[0].clone(),
                    src[1].clone(),
                    src[2].clone(),
                    (30, self.cams.as_entire_binding()),
                    (31, self.nb.as_entire_binding()),
                ],
            );
            let bm = [0, 1].map(|i| {
                cx.bind(
                    &self.pp[1],
                    &[
                        src[0].clone(),
                        src[1].clone(),
                        src[2].clone(),
                        (30, self.cams.as_entire_binding()),
                        (32, self.prob.as_entire_binding()),
                        (33, R::TextureView(&tv(&self.m[1 - i]))),
                        (34, R::TextureView(&tv(&self.m[i]))),
                    ],
                )
            });
            let [b0, b1] = bm;
            self.tmp = Some((tt, [bi, b0, b1]));
            self.ob = None;
        }
        let (tt, tb) = self.tmp.as_ref().unwrap();
        enc.copy_texture_to_texture(f.as_image_copy(), tt.as_image_copy(), crate::whole(f));
        let dt = self.last.map(|l| (t - l).max(0.0));
        let wt = |half: f64| {
            dt.map_or(1.0, |d| {
                if half > 0.0 {
                    1.0 - 0.5f64.powf(d / half)
                } else {
                    1.0
                }
            })
        };
        let y = *self.y0.get_or_insert_with(|| {
            let [yaw, _] = track::yaw_pitch(q * glam::Vec3::NEG_X);
            Quat::from_rotation_z(yaw.to_radians())
        });
        let m = glam::Mat3::from_quat(y * q);
        let k = |s: u32, d: u32| ((d as f32 / s as f32).ceil().clamp(1.0, 4.0)) as u32;
        let v = |v: glam::Vec3| [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), 0];
        let mut u = [0u32; 24];
        u[..4].copy_from_slice(&v(m.x_axis));
        u[4..8].copy_from_slice(&v(m.y_axis));
        u[8..12].copy_from_slice(&v(m.z_axis));
        u[12..16].copy_from_slice(&self.st);
        u[16] = (wt(self.half as f64) as f32).to_bits();
        u[17] = (wt(GAIN_HALF) as f32).to_bits();
        u[19] = self.tone.clamp(0.0, 1.0).to_bits();
        // taps per axis: frame pixels per tile pixel at the centre of a 120° tile (the most), per mask pixel
        u[20] = k(
            seg::SKY,
            (f.width() as f32 * 2.0 * 3f32.sqrt() / std::f32::consts::TAU) as u32,
        );
        u[21] = k(MW, f.width());
        u[22] = seg::TILES;
        u[23] = seg::SKY;
        self.cx
            .q
            .write_buffer(&self.ub, 0, bytemuck::cast_slice(&u));
        if fresh {
            let i = (self.n % 2) as usize;
            run(enc, &self.pp[1], &tb[1 + i], MW, MW / 2);
            run(enc, &self.pp[2], &self.bg[i], MW, MW / 2);
            run(enc, &self.pp[3], &self.bg[4], MW, MW / 2);
            let mut c = enc.begin_compute_pass(&Default::default());
            c.set_pipeline(&self.pp[4]);
            c.set_bind_group(0, &self.bg[2 + i], &[]);
            c.dispatch_workgroups(1, 1, 1);
            drop(c);
            (self.n, self.last) = (self.n + 1, Some(t));
        }
        if self.n > 0 {
            let p = &self.out[fi(f.format())];
            let (cx, ab, pano, gain, ub) =
                (&self.cx, &self.ab[1], &self.pano, &self.gain, &self.ub);
            let bg = crate::cached(&mut self.ob, f, || {
                cx.bind(
                    p,
                    &[
                        (29, ub.as_entire_binding()),
                        (43, R::TextureView(&tv(tt))),
                        (44, R::TextureView(&tv(ab))),
                        (45, R::TextureView(&tv(pano))),
                        (46, R::Sampler(&cx.smp)),
                        (47, gain.as_entire_binding()),
                        (48, R::TextureView(&tv(f))),
                    ],
                )
            });
            run(enc, p, bg, f.width(), f.height());
        }
        let s = seg::SKY;
        let mut c = enc.begin_compute_pass(&Default::default());
        c.set_pipeline(&self.pp[0]);
        c.set_bind_group(0, &tb[0], &[]);
        c.dispatch_workgroups(s.div_ceil(8), s.div_ceil(8), seg::TILES);
        drop(c);
        let r = self.net.io.resource();
        enc.copy_buffer_to_buffer(&self.nb, 0, &r.buffer, r.offset, self.nb.size());
        self.pending = true;
        Ok(())
    }

    /// The last mask: the sky's probability over the world, smoothed over time (before the guided filter), `MW` ×
    /// `MW / 2` row-major in stitch's layout (reads back: waits for the GPU); empty before the first.
    pub fn mask(&self) -> Vec<f32> {
        if self.n == 0 {
            return vec![];
        }
        let t = &self.m[((self.n - 1) % 2) as usize];
        let b = self.cx.buf(
            8 * (MW * MW / 2) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let mut e = self.cx.dev.create_command_encoder(&Default::default());
        e.copy_texture_to_buffer(
            t.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &b,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8 * MW),
                    rows_per_image: None,
                },
            },
            crate::whole(t),
        );
        self.cx.q.submit([e.finish()]);
        b.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        let _ = self.cx.dev.poll(wgpu::PollType::wait_indefinitely());
        let v: Vec<u16> = bytemuck::cast_slice(&b.get_mapped_range(..).expect("mapped")).to_vec();
        // half floats in 0..1: no infinities, no sign
        v.chunks(4)
            .map(|p| match (p[0] >> 10 & 31, (p[0] & 1023) as f32) {
                (0, m) => m * 2f32.powi(-24),
                (e, m) => (1.0 + m / 1024.0) * 2f32.powi(e as i32 - 15),
            })
            .collect()
    }

    /// For the exporter on a 360 output: output frame k's directions turned into the world's by `q(k)` (e.g. the
    /// stabilised pose and the gyro's orientation, the first frame's heading taken out). A failing network run panics,
    /// which fails the job.
    pub fn hook(mut self, q: impl Fn(u64) -> Quat + Send + 'static) -> export::Hook {
        Box::new(move |x: export::Fx| {
            self.push(x.enc, x.tex, q(x.k), x.t)
                .unwrap_or_else(|e| panic!("sky: {e}"))
        })
    }
}

/// A sky picture made here (no licence attached): an equirect `w` × `w / 2`, RGBA8 sRGB, `top` at the zenith fading
/// into `horizon` (in linear light, mostly near the horizon), `horizon` below it.
pub fn gradient(w: u32, top: [u8; 3], horizon: [u8; 3]) -> Vec<u8> {
    let lin = |c: u8| (c as f32 / 255.0).powf(2.2);
    let h = w / 2;
    (0..w * h)
        .flat_map(|i| {
            let e = (std::f32::consts::FRAC_PI_2
                - std::f32::consts::PI * (i / w) as f32 / (h - 1) as f32)
                .max(0.0);
            let s = e.sin().sqrt();
            let c = |j: usize| {
                ((lin(horizon[j]) * (1.0 - s) + lin(top[j]) * s).powf(1.0 / 2.2) * 255.0).round()
                    as u8
            };
            [c(0), c(1), c(2), 255]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, process::Command};

    /// The generated network against onnxruntime on the same input (the six tiles of v606 frame 300, the gate under a
    /// blue sky, stitched at 3840 with the horizon lock), when `X5_ORT` names a Python with onnxruntime and numpy.
    #[test]
    fn vs_onnxruntime() {
        let (Some(smp), Some(py)) = (std::env::var_os("X5_SAMPLES"), std::env::var("X5_ORT").ok())
        else {
            eprintln!("X5_SAMPLES or X5_ORT unset: skipped");
            return;
        };
        let d = std::env::temp_dir().join("x5-shotlab-sky-ort");
        std::fs::create_dir_all(&d).unwrap();
        let g = video::Gpu::new().unwrap();
        let (sx, cx) = (
            stitch::Ctx::new(&g.device, &g.queue),
            Ctx::new(&g.device, &g.queue),
        );
        let c = crate::Clip::open(&PathBuf::from(smp).join("v606.insv")).unwrap();
        let ps = c.poses(stab::Lock::Horizon, 0.5);
        let mut s = Sky::new(&g, &cx, &gradient(64, [0; 3], [0; 3]), 64, 32).unwrap();
        let mut inp = vec![];
        c.frames(
            &g,
            &sx,
            3840,
            false,
            &[300],
            |k| c.pose(k, ps.as_deref()),
            |f, k| {
                let q = c.world(k, ps.as_deref()).unwrap();
                for i in 0..2 {
                    let mut e = cx.dev.create_command_encoder(&Default::default());
                    s.push(&mut e, f, q, i as f64).unwrap();
                    cx.q.submit([e.finish()]);
                    if i == 0 {
                        inp = read(&cx, &s.nb);
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        let o = seg::SKY_OUT;
        let pr: Vec<f32> = bytemuck::cast_slice(&read(&cx, &s.prob)).to_vec();
        let [a, b, c, _] = s.st;
        let m: Vec<f32> = (0..seg::TILES * o * o)
            .map(|i| pr[(a * (i / (o * o)) + b * (i / o % o) + c * (i % o)) as usize])
            .collect();
        std::fs::write(d.join("in.f32"), &inp).unwrap();
        std::fs::write(d.join("burn.f32"), bytemuck::cast_slice(&m)).unwrap();
        let model = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/efficientvit_seg_b1_ade20k_6x384.onnx"
        );
        let src = format!(
            "import numpy as np, onnxruntime as o\n\
             x = np.fromfile('{0}/in.f32', np.float32).reshape({1}, 3, {2}, {2})\n\
             s = o.InferenceSession('{model}', providers=['CPUExecutionProvider'])\n\
             l = s.run(None, {{'pixel_values': x}})[0].astype(np.float64)\n\
             e = np.exp(l - l.max(1, keepdims=True))\n\
             (e[:, 2] / e.sum(1)).astype(np.float32).tofile('{0}/out.f32')",
            d.display(),
            seg::TILES,
            seg::SKY
        );
        assert!(
            Command::new(py)
                .args(["-I", "-c", &src])
                .status()
                .unwrap()
                .success()
        );
        let r: Vec<f32> = bytemuck::cast_slice(&std::fs::read(d.join("out.f32")).unwrap()).to_vec();
        let worst = m
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mean = m
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / r.len() as f64;
        let sky = r.iter().filter(|&&v| v > 0.5).count();
        let flip = m
            .iter()
            .zip(&r)
            .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
            .count();
        eprintln!(
            "EfficientViT-seg B1 burn vs onnxruntime: largest difference {worst:.2e}, mean {mean:.2e}; {sky} of {} \
             pixels sky, {flip} on the other side of ½",
            r.len()
        );
        assert!(worst < 1e-3 && sky > r.len() / 4);
    }

    fn read(cx: &Ctx, b: &wgpu::Buffer) -> Vec<u8> {
        let m = cx.buf(
            b.size(),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let mut e = cx.dev.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(b, 0, &m, 0, b.size());
        cx.q.submit([e.finish()]);
        m.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
        cx.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        m.get_mapped_range(..).unwrap().to_vec()
    }
}
