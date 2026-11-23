//! Freeze Trail, Echo Trail: the subject cut out at several times and composited onto one frame. A cut is a square
//! perspective crop around the subject's box (from crate track: `Tracker` samples or `Det`s), kept as a patch, and a
//! person mask of the crop (`seg`, PP-HumanSeg); pasting projects every equirect pixel into each cut's crop and lays
//! the patch over the frame by the mask, gated to the subject's box so the other people in the crop stay out.

use crate::{Cache, Ctx, array, cached, fi, run, seg, tv};
use glam::{Vec2, Vec3};

/// What a hook keeps of the subject.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Clones {
    /// Freeze Trail: a copy stays behind each time the subject has moved `gap` of its own width from the last copy;
    /// the latest `n` show.
    Freeze { gap: f32, n: u32 },
    /// Echo Trail: the subject as it was `lag`, 2·`lag`, …, `n`·`lag` frames ago, all moving with it.
    Echo { lag: u32, n: u32 },
}

impl Clones {
    /// Cut slots the hook needs.
    pub fn slots(&self) -> u32 {
        match *self {
            Clones::Freeze { n, .. } => n + 1,
            Clones::Echo { lag, n } => n * lag.max(1) + 2,
        }
    }
}

/// Cuts of a subject in `n` slots. One `cut` and one `paste` per submit (their parameters are written with
/// `queue.write_buffer`); `seg` masks the last cut once it is submitted. Memory per slot: a `side`² patch in the
/// frame format and 2 · 192² floats.
pub struct Cuts {
    cx: Ctx,
    net: seg::Net,
    pat: wgpu::Texture,
    pv: wgpu::TextureView,
    cams: wgpu::Buffer,
    mask: wgpu::Buffer,
    nb: wgpu::Buffer,
    /// Uniforms of the patch and network-input passes, of the paste.
    cu: [wgpu::Buffer; 2],
    pu: wgpu::Buffer,
    pp: [wgpu::ComputePipeline; 2],
    paste: [wgpu::ComputePipeline; 2],
    /// Mask strides (channel, row, column) and side, from the network's output.
    st: [u32; 4],
    /// Crop half extent over the subject box's larger one.
    pub margin: f32,
    /// Alpha = (probability − ½) · gain + ½, clamped: above 1 the edge hardens.
    pub gain: f32,
    pending: Option<u32>,
    bg: [Cache; 2],
    /// The frame copy the paste reads, its bind group.
    tmp: Option<(wgpu::Texture, wgpu::Texture, wgpu::BindGroup)>,
}

/// Floats per mask slot: the network's output, both classes.
const MSZ: u64 = 2 * (seg::SIDE * seg::SIDE) as u64;

impl Cuts {
    /// Person mask side (the network's), pixels.
    pub const SIDE: u32 = seg::SIDE;

    /// `n` slots (up to the device's array layers) of `side`² patches in `format` (`FORMATS`, that of the frames);
    /// the network on `g`'s device through `stitch::attach` (shared with RAFT and crate track).
    pub fn new(
        g: &video::Gpu,
        cx: &Ctx,
        n: u32,
        side: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        use wgpu::{BufferUsages as B, TextureUsages as U};
        let bd = stitch::attach(&g.instance, &g.adapter, &g.device, &g.queue)?;
        let n = n.max(1);
        let pat = cx.tex(
            side,
            side,
            n,
            format,
            U::STORAGE_BINDING | U::TEXTURE_BINDING,
        );
        let s = (seg::SIDE * seg::SIDE) as u64;
        Ok(Self {
            net: seg::Net::new(&bd, false)?,
            pv: array(&pat),
            pat,
            cams: cx.buf(64 * n as u64, B::STORAGE | B::COPY_DST),
            mask: cx.buf(4 * MSZ * n as u64, B::STORAGE | B::COPY_DST | B::COPY_SRC),
            nb: cx.buf(12 * s, B::STORAGE | B::COPY_SRC),
            cu: [0, 1].map(|_| cx.buf(16, B::UNIFORM | B::COPY_DST)),
            pu: cx.buf(288, B::UNIFORM | B::COPY_DST),
            pp: [cx.pipe("pat", fi(format)), cx.pipe("netin", 0)],
            paste: [0, 1].map(|f| cx.pipe("paste", f)),
            st: [0, 0, 0, seg::SIDE],
            margin: 1.3,
            gain: 2.0,
            pending: None,
            bg: [None, None],
            tmp: None,
            cx: cx.clone(),
        })
    }

    pub fn slots(&self) -> u32 {
        self.pat.depth_or_array_layers()
    }

    /// Records the cut of the subject (box centre `dir`, angular width and height `size`, degrees; directions in the
    /// frame's equirect) from frame `f` (an equirect, `TEXTURE_BINDING`) into `slot`: the patch, and the network's
    /// input, which `seg` takes after this submit.
    pub fn cut(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        f: &wgpu::Texture,
        slot: u32,
        dir: Vec3,
        size: Vec2,
    ) {
        let c = track::Cam::new(dir, Vec3::Z, Vec2::ONE);
        let b = (size.max(Vec2::splat(0.5)) / 2.0)
            .to_array()
            .map(|a| a.to_radians().tan());
        let e = b[0].max(b[1]) * self.margin;
        let v = |v: Vec3| [v.x, v.y, v.z, 0.0];
        let cam: [[f32; 4]; 4] = [v(c.f), v(c.r), v(c.u), [e, e, b[0], b[1]]];
        let q = &self.cx.q;
        q.write_buffer(&self.cams, 64 * slot as u64, bytemuck::cast_slice(&cam));
        let side = [self.pat.width(), seg::SIDE];
        for (u, s) in self.cu.iter().zip(side) {
            // frame pixels per target pixel at the centre (the most): that many taps per axis, at most 4
            let k = (2.0 * e / s as f32 * f.width() as f32 / std::f32::consts::TAU)
                .ceil()
                .clamp(1.0, 4.0);
            q.write_buffer(u, 0, bytemuck::cast_slice(&[slot, s, k as u32, 0]));
        }
        let (cx, cams, pv, nb, cu) = (&self.cx, &self.cams, &self.pv, &self.nb, &self.cu);
        for i in 0..2 {
            let p = &self.pp[i];
            let o = match i {
                0 => (20, wgpu::BindingResource::TextureView(pv)),
                _ => (21, nb.as_entire_binding()),
            };
            let bg = cached(&mut self.bg[i], f, || {
                cx.bind(
                    p,
                    &[
                        (16, wgpu::BindingResource::TextureView(&tv(f))),
                        (17, wgpu::BindingResource::Sampler(&cx.smp)),
                        (18, cu[i].as_entire_binding()),
                        (19, cams.as_entire_binding()),
                        o,
                    ],
                )
            });
            run(enc, p, bg, side[i], side[i]);
        }
        let r = self.net.io.resource();
        enc.copy_buffer_to_buffer(nb, 0, &r.buffer, r.offset, nb.size());
        self.pending = Some(slot);
    }

    /// Runs the network on the last cut, which must be submitted, and records the copy of its mask into its slot;
    /// returns that slot (None without a cut waiting). The network's work is submitted here, before `enc`.
    pub fn seg(&mut self, enc: &mut wgpu::CommandEncoder) -> Result<Option<u32>, String> {
        let Some(s) = self.pending.take() else {
            return Ok(None);
        };
        let (r, st) = self.net.run()?;
        let span = (st[1] + (st[2] + st[3]) * (seg::SIDE - 1) + 1) as u64;
        if span > MSZ {
            return Err(format!("segmentation output strides {st:?}"));
        }
        enc.copy_buffer_to_buffer(
            &r.buffer,
            r.offset,
            &self.mask,
            4 * MSZ * s as u64,
            4 * span,
        );
        self.st = [st[1], st[2], st[3], seg::SIDE];
        Ok(Some(s))
    }

    /// Records the cuts in `slots` (masked, at most 64; later ones on top) laid over frame `f` (an equirect in a
    /// `FORMATS` format; `COPY_SRC`, `STORAGE_BINDING`).
    pub fn paste(&mut self, enc: &mut wgpu::CommandEncoder, f: &wgpu::Texture, slots: &[u32]) {
        use wgpu::{BindingResource as R, TextureUsages as U};
        let n = slots.len().min(64);
        if n == 0 {
            return;
        }
        let mut u = [0u32; 72];
        u[..4].copy_from_slice(&self.st);
        u[4..8].copy_from_slice(&[n as u32, MSZ as u32, self.gain.to_bits(), 0]);
        u[8..8 + n].copy_from_slice(&slots[..n]);
        self.cx
            .q
            .write_buffer(&self.pu, 0, bytemuck::cast_slice(&u));
        let p = &self.paste[fi(f.format())];
        if self.tmp.as_ref().is_none_or(|x| x.0 != *f) {
            let t = self.cx.tex(
                f.width(),
                f.height(),
                1,
                f.format(),
                U::TEXTURE_BINDING | U::COPY_DST,
            );
            let bg = self.cx.bind(
                p,
                &[
                    (22, R::TextureView(&tv(&t))),
                    (23, R::TextureView(&self.pv)),
                    (24, R::Sampler(&self.cx.smp)),
                    (25, self.cams.as_entire_binding()),
                    (26, self.mask.as_entire_binding()),
                    (27, self.pu.as_entire_binding()),
                    (28, R::TextureView(&tv(f))),
                ],
            );
            self.tmp = Some((f.clone(), t, bg));
        }
        let (_, t, bg) = self.tmp.as_ref().unwrap();
        enc.copy_texture_to_texture(f.as_image_copy(), t.as_image_copy(), crate::whole(f));
        run(enc, p, bg, f.width(), f.height());
    }

    /// Slot `s`'s person probabilities, `SIDE`² row-major over its crop (reads back: waits for the GPU).
    pub fn mask(&self, s: u32) -> Vec<f32> {
        let b = self.cx.buf(
            4 * MSZ,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let mut e = self.cx.dev.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(&self.mask, 4 * MSZ * s as u64, &b, 0, 4 * MSZ);
        self.cx.q.submit([e.finish()]);
        b.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        let _ = self.cx.dev.poll(wgpu::PollType::wait_indefinitely());
        let v: Vec<f32> = bytemuck::cast_slice(&b.get_mapped_range(..).expect("mapped")).to_vec();
        let [c, r, x, n] = self.st;
        (0..n * n)
            .map(|i| v[(c + r * (i / n) + x * (i % n)) as usize])
            .collect()
    }

    /// For the exporter on a 360 output: the subject's box at output frame k from `subj` (e.g. `Tracker` samples
    /// while tracked), the clones of `mode` laid over each frame. A cut's mask is ready one frame later (the network
    /// runs on the cut once its frame is submitted). Needs `mode.slots()` slots. A failing network run panics, which
    /// fails the job.
    pub fn hook(
        mut self,
        mode: Clones,
        mut subj: impl FnMut(u64) -> Option<(Vec3, Vec2)> + Send + 'static,
    ) -> Result<export::Hook, String> {
        let r = mode.slots();
        if self.slots() < r {
            return Err(format!(
                "{mode:?}: {r} slots, the cuts have {}",
                self.slots()
            ));
        }
        // Freeze: the slots shown, oldest first, the next slot, the last copy's direction and width;
        // Echo: whether each slot holds a masked cut of the frame it belongs to
        let (mut shown, mut next, mut last) = (vec![], 0, None::<(Vec3, f32)>);
        let mut have = vec![false; r as usize];
        let mut list = vec![];
        Ok(Box::new(move |x: export::Fx| {
            let done = self.seg(x.enc).unwrap_or_else(|e| panic!("clones: {e}"));
            let b = subj(x.k);
            match mode {
                Clones::Freeze { gap, n } => {
                    if let Some(s) = done {
                        shown.push(s);
                        if shown.len() > n as usize {
                            shown.remove(0);
                        }
                    }
                    if let Some((d, sz)) = b
                        && last.is_none_or(|(l, w)| l.angle_between(d).to_degrees() >= gap * w)
                    {
                        self.cut(x.enc, x.tex, next, d, sz);
                        last = Some((d, sz.x));
                        next = (next + 1) % r;
                    }
                    self.paste(x.enc, x.tex, &shown);
                }
                Clones::Echo { lag, n } => {
                    if let Some(s) = done {
                        have[s as usize] = true;
                    }
                    let s = (x.k % r as u64) as u32;
                    have[s as usize] = false;
                    if let Some((d, sz)) = b {
                        self.cut(x.enc, x.tex, s, d, sz);
                    }
                    list.clear();
                    for i in (1..=n as u64).rev() {
                        if let Some(j) = x.k.checked_sub(i * lag.max(1) as u64)
                            && have[(j % r as u64) as usize]
                        {
                            list.push((j % r as u64) as u32);
                        }
                    }
                    self.paste(x.enc, x.tex, &list);
                }
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, process::Command};

    /// The generated network against onnxruntime on the same input (a cut of a person in v627 frame 300), when
    /// `X5_ORT` names a Python with onnxruntime and numpy.
    #[test]
    fn vs_onnxruntime() {
        let (Some(smp), Some(py)) = (std::env::var_os("X5_SAMPLES"), std::env::var("X5_ORT").ok())
        else {
            eprintln!("X5_SAMPLES or X5_ORT unset: skipped");
            return;
        };
        let d = std::env::temp_dir().join("x5-shotlab-ort");
        std::fs::create_dir_all(&d).unwrap();
        let g = video::Gpu::new().unwrap();
        let (sx, cx) = (
            stitch::Ctx::new(&g.device, &g.queue),
            Ctx::new(&g.device, &g.queue),
        );
        let c = crate::Clip::open(&PathBuf::from(smp).join("v627.insv")).unwrap();
        let mut cu = Cuts::new(&g, &cx, 1, 256, crate::FORMATS[0]).unwrap();
        // the woman walking right of the gate: pixel (4637, 1700) of 5760 × 2880, 22° crop
        let dir = stab::vis::ray(5760, 2880, [4637.0, 1700.0]).as_vec3();
        c.frames(
            &g,
            &sx,
            2880,
            false,
            &[300],
            |k| c.pose(k, None),
            |f, _| {
                let mut e = cx.dev.create_command_encoder(&Default::default());
                cu.cut(&mut e, f, 0, dir, Vec2::new(8.0, 16.0));
                cx.q.submit([e.finish()]);
                Ok(())
            },
        )
        .unwrap();
        let inp = {
            let b = cx.buf(
                cu.nb.size(),
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            );
            let mut e = cx.dev.create_command_encoder(&Default::default());
            e.copy_buffer_to_buffer(&cu.nb, 0, &b, 0, b.size());
            cx.q.submit([e.finish()]);
            b.map_async(wgpu::MapMode::Read, .., |r| r.unwrap());
            cx.dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            b.get_mapped_range(..).unwrap().to_vec()
        };
        let mut e = cx.dev.create_command_encoder(&Default::default());
        assert_eq!(cu.seg(&mut e).unwrap(), Some(0));
        cx.q.submit([e.finish()]);
        let m = cu.mask(0);
        std::fs::write(d.join("in.f32"), &inp).unwrap();
        let model = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/human_segmentation_pphumanseg_2023mar.onnx"
        );
        let src = format!(
            "import numpy as np, onnxruntime as o\n\
             x = np.fromfile('{0}/in.f32', np.float32).reshape(1, 3, 192, 192)\n\
             s = o.InferenceSession('{model}', providers=['CPUExecutionProvider'])\n\
             s.run(None, {{'x': x}})[0][0, 1].astype(np.float32).tofile('{0}/out.f32')",
            d.display()
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
        let person = r.iter().filter(|&&v| v > 0.5).count();
        eprintln!(
            "PP-HumanSeg burn vs onnxruntime: largest difference {worst:.2e}; {person} of {} pixels person",
            r.len()
        );
        assert!(worst < 1e-3 && person > 1000);
    }
}
