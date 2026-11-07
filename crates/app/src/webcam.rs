//! The X5's Webcam Mode as a source (PLAN.md §3.2): its MJPEG over UVC (`media::uvc`) decoded on a thread of its own
//! (`media::mjpeg`), uploaded, and drawn: the 2880 × 1440 in-camera equirect through crate stitch's preview (`Ctx::pano`, every view mode), the
//! 1920 × 1080 flat picture as it is. Offline it replays a raw MJPEG capture at the camera's 30 fps.
//!
//! Hand-over as the live preview's (`feed`): up to 3 decoded pictures queue, each due at its JPEG's arrival plus the
//! slowest decode of the last 32, and a consumer takes them in turn ([`feed::pick`]); while the live outputs run they
//! hold it and the viewer shows what they take.

use crate::{
    feed::{self, IDLE, LATE, Samples, mono},
    player::{FMT, Look, Tgt},
};
use eframe::{egui, egui_wgpu::RenderState};
use media::mjpeg::{self, Mjpeg};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    thread,
    time::{Duration, Instant},
};
use stitch::{Ctx, Mode, Preview, View};
use video::Gpu;

/// Where the pictures come from: the camera at a size (2880 × 1440 picks the equirect, 1920 × 1080 the flat view),
/// or a capture file replayed.
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    Uvc(u32, u32),
    File(PathBuf),
}

/// A decoded picture: RGBA, size, its number, when its JPEG arrived (CLOCK_MONOTONIC ns).
struct Dec {
    px: Vec<u8>,
    w: u32,
    h: u32,
    seq: u64,
    got: u64,
}

/// The decoded pictures not taken yet, their spare buffers, and what was counted.
#[derive(Default)]
struct Sh {
    q: VecDeque<Dec>,
    free: Vec<Vec<u8>>,
    /// pictures decoded, the number of the last one taken
    seq: u64,
    took: u64,
    /// arrival → decoded (ns) of the last 32 pictures
    late: [u64; 32],
    nl: usize,
    /// when the outputs last took (CLOCK_MONOTONIC ns)
    out: u64,
    /// JPEGs read, pictures decoded but never taken; output clock ticks with a picture, those that repeated it
    read: u64,
    skip: u64,
    ticks: u64,
    rep: u64,
    dec: Samples,
    err: Option<String>,
}

fn lock(sh: &Mutex<Sh>) -> std::sync::MutexGuard<'_, Sh> {
    sh.lock().unwrap_or_else(|e| e.into_inner())
}

/// Reads JPEGs and decodes each (2880 × 1440 in ~9 ms here: the camera's 4 queued buffers cover it) into the queue in
/// `sh`, which drops its oldest beyond 3.
fn run(src: Input, sh: Arc<Mutex<Sh>>, quit: Arc<AtomicBool>, wake: impl Fn()) {
    let fail = |m: String| lock(&sh).err = Some(m);
    let (mut m, mut px) = (Mjpeg::default(), vec![]);
    let mut dec = |j: &[u8]| {
        let (t, got) = (Instant::now(), mono());
        let r = m.rgba(j, &mut px);
        let mut s = lock(&sh);
        s.read += 1;
        match r {
            Ok((w, h)) => {
                if s.q.len() == 3 {
                    let o = s.q.pop_front().unwrap();
                    (s.skip, _) = (s.skip + 1, s.free.push(o.px));
                }
                let (n, b) = (s.nl, s.free.pop().unwrap_or_default());
                (s.late[n % 32], s.nl, s.seq) = (mono() - got, n + 1, s.seq + 1);
                let seq = s.seq;
                s.q.push_back(Dec {
                    px: std::mem::replace(&mut px, b),
                    w: w as u32,
                    h: h as u32,
                    seq,
                    got,
                });
                s.dec.push(t.elapsed().as_secs_f32() * 1e3);
            }
            Err(e) => s.err = Some(format!("MJPEG: {e}")),
        }
        drop(s);
        wake();
    };
    match src {
        // V4L2 only: macOS keeps a UVC camera for AVFoundation
        #[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
        Input::Uvc(..) => fail("Webcam Mode: UVC capture is Linux only".into()),
        #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
        Input::Uvc(w, h) => {
            let mut j = vec![];
            let mut u = match media::uvc::Uvc::find().and_then(|p| media::uvc::Uvc::open(p, w, h)) {
                Ok(u) => u,
                Err(e) => return fail(format!("Webcam Mode: {e}")),
            };
            while !quit.load(Relaxed) {
                match u.next(&mut j, Duration::from_millis(500)) {
                    Ok(true) => dec(&j),
                    Ok(false) => {}
                    Err(e) => return fail(format!("Webcam Mode: {e}")),
                }
            }
        }
        Input::File(p) => {
            let b = match std::fs::read(&p) {
                Ok(b) => b,
                Err(e) => return fail(format!("{}: {e}", p.display())),
            };
            let fs: Vec<&[u8]> = mjpeg::frames(&b).collect();
            if fs.is_empty() {
                return fail(format!("{}: no JPEG", p.display()));
            }
            let (t0, period) = (Instant::now(), Duration::from_nanos(1_000_000_000 / 30));
            for i in 0.. {
                if quit.load(Relaxed) {
                    break;
                }
                if let Some(d) = (t0 + period * i).checked_duration_since(Instant::now()) {
                    thread::sleep(d);
                }
                dec(fs[i as usize % fs.len()]);
            }
        }
    }
}

/// A plain scaled copy of an RGBA picture into the bound target's viewport.
const BLIT: &str = "
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
struct O { @builtin(position) p: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> O {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return O(vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0), uv);
}
@fragment fn fs(o: O) -> @location(0) vec4f { return textureSample(t, s, o.uv); }
";

/// The uploaded picture: texture, its equirect binding (2:1 pictures), its blit binding.
struct Pic {
    tex: wgpu::Texture,
    pano: stitch::Pano,
    bg: wgpu::BindGroup,
}

pub struct Webcam {
    g: Arc<Gpu>,
    cx: Ctx,
    sh: Arc<Mutex<Sh>>,
    quit: Arc<AtomicBool>,
    th: Option<thread::JoinHandle<()>>,
    pic: Option<Pic>,
    pv: Preview,
    blit: wgpu::RenderPipeline,
    smp: wgpu::Sampler,
    pub look: Look,
    pub view: View,
    tgt: Option<Tgt>,
    drawn: u64,
    /// the outputs have taken no picture since they started
    start: bool,
    /// the uploaded picture's number and JPEG arrival; the number of the one the viewer drew last
    cur: (u64, u64),
    vs: u64,
    /// pictures shown and from when; JPEG arrival → shown, ms
    pub t0: Option<Instant>,
    pub shown: u64,
    pub lat: Samples,
}

impl Webcam {
    pub fn new(g: &Arc<Gpu>, cx: &Ctx, input: Input, wake: impl Fn() + Send + 'static) -> Webcam {
        let (sh, quit) = (
            Arc::new(Mutex::new(Sh::default())),
            Arc::new(AtomicBool::new(false)),
        );
        let (s2, q2, i2) = (sh.clone(), quit.clone(), input.clone());
        let th = thread::Builder::new()
            .name("webcam".into())
            .spawn(move || run(i2, s2, q2, wake))
            .expect("webcam thread");
        let d = &g.device;
        let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT.into()),
        });
        let blit = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &m,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &m,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(FMT.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let smp = d.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Webcam {
            g: g.clone(),
            cx: cx.clone(),
            sh,
            quit,
            th: Some(th),
            pic: None,
            pv: Preview::pano(cx, FMT),
            blit,
            smp,
            look: Look::Perspective,
            view: Look::Perspective.view(),
            tgt: None,
            drawn: 0,
            start: false,
            cur: (0, 0),
            vs: 0,
            t0: None,
            shown: 0,
            lat: Samples::default(),
        }
    }

    /// The picture is the 2:1 equirect (else flat); None before the first one.
    pub fn pano(&self) -> Option<bool> {
        self.pic
            .as_ref()
            .map(|p| p.tex.width() == 2 * p.tex.height())
    }

    pub fn err(&self) -> Option<String> {
        self.sh.lock().unwrap().err.clone()
    }

    /// Pictures taken (by the viewer or a live output).
    pub fn taken(&self) -> u64 {
        self.sh.lock().unwrap().took
    }

    /// Uploads the picture due at `now` (`feed::pick`); returns its arrival time if it took one. `out`: for the live
    /// outputs, which then hold the hand-over; the viewer takes none meanwhile.
    fn take(&mut self, now: u64, out: bool) -> Option<u64> {
        let mut s = self.sh.lock().unwrap();
        let t = mono();
        let ran = t < s.out + IDLE;
        if out {
            s.out = t;
        } else if ran {
            return None;
        }
        // as the preview's outputs: the first picture once it is due a quarter of the window
        self.start |= out && !ran;
        let now = match out && self.start {
            true => now.saturating_sub(LATE / 4),
            false => now,
        };
        let l = s.late[..s.nl.min(32)]
            .iter()
            .max()
            .map_or(0, |&l| l.min(66_000_000));
        let i = feed::pick(s.q.iter().map(|p| p.got + l), now);
        if out && self.pic.is_some() {
            s.ticks += 1;
            s.rep += (ran && i.is_none()) as u64;
        }
        let i = i?;
        self.start &= !out;
        s.skip += i as u64;
        for _ in 0..i {
            let o = s.q.pop_front().unwrap();
            s.free.push(o.px);
        }
        let p = s.q.pop_front().unwrap();
        let (w, h) = (p.w, p.h);
        let g = &self.g;
        if self
            .pic
            .as_ref()
            .is_none_or(|p| (p.tex.width(), p.tex.height()) != (w, h))
        {
            let tex = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("webcam"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = tex.create_view(&Default::default());
            let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("blit"),
                layout: &self.blit.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&v),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.smp),
                    },
                ],
            });
            self.pic = Some(Pic {
                pano: self.cx.pano(&v),
                tex,
                bg,
            });
        }
        g.queue.write_texture(
            self.pic.as_ref().unwrap().tex.as_image_copy(),
            &p.px,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        (s.took, self.cur) = (p.seq, (p.seq, p.got));
        s.free.push(p.px);
        self.t0.get_or_insert_with(Instant::now);
        Some(p.got)
    }

    /// Records the picture with view `v` into `t` (viewport `vp`) and submits: an equirect through the preview, a flat
    /// picture blitted.
    fn draw_into(&self, t: &wgpu::TextureView, v: &View, vp: [f32; 4]) {
        let Some(p) = &self.pic else { return };
        let mut e = self.g.device.create_command_encoder(&Default::default());
        let flat = p.tex.width() != 2 * p.tex.height();
        if !flat {
            self.pv.set(v, vp);
        }
        {
            let mut rp = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("webcam"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: t,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_viewport(vp[0], vp[1], vp[2], vp[3], 0.0, 1.0);
            if flat {
                rp.set_pipeline(&self.blit);
                rp.set_bind_group(0, &p.bg, &[]);
                rp.draw(0..3, 0..1);
            } else {
                self.pv.draw(&mut rp, &p.pano);
            }
        }
        self.g.queue.submit([e.finish()]);
    }

    /// `[w, h]` fitted into `size` keeping the picture's aspect (a flat picture is never stretched).
    fn fit(&self, [w, h]: [f32; 2]) -> [f32; 4] {
        let Some(p) = &self.pic else {
            return [0.0, 0.0, w, h];
        };
        let a = p.tex.width() as f32 / p.tex.height() as f32;
        let (x, y) = (w.min(h * a), h.min(w / a));
        [(w - x) / 2.0, (h - y) / 2.0, x, y]
    }

    /// The viewer: renders into its texture at `size` ([`Webcam::show`]). Returns the texture.
    pub fn render(
        &mut self,
        rs: &RenderState,
        size: [u32; 2],
        dirty: bool,
    ) -> Option<egui::TextureId> {
        let resized = Tgt::fit(&mut self.tgt, &self.g, rs, size);
        let t = self.tgt.as_ref().unwrap().view.clone();
        self.show(&t, size, dirty || resized);
        self.tgt.as_ref().map(|t| t.id)
    }

    /// The viewer without egui: takes the picture due (unless the outputs run), draws it into `t` (`size`) when it is
    /// new or `dirty`.
    pub fn show(&mut self, t: &wgpu::TextureView, size: [u32; 2], dirty: bool) {
        self.take(mono(), false);
        let new = self.cur.0 != self.vs;
        if new || dirty {
            let sz = [size[0] as f32, size[1] as f32];
            let vp = match self.pano() {
                Some(false) => self.fit(sz),
                _ => [0.0, 0.0, sz[0], sz[1]],
            };
            self.draw_into(t, &self.view, vp);
        }
        if new && self.pic.is_some() {
            (self.vs, self.shown) = (self.cur.0, self.shown + 1);
            self.lat.push((mono() - self.cur.1) as f32 * 1e-6);
        }
    }

    pub fn report(&self) -> String {
        let s = self.sh.lock().unwrap();
        let secs = self.t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let [d50, d99, dm] = s.dec.pct();
        let [l50, l99, lm] = self.lat.pct();
        let (w, h) = self
            .pic
            .as_ref()
            .map_or((0, 0), |p| (p.tex.width(), p.tex.height()));
        format!(
            "{:.1} s: {w} × {h}, {} JPEGs read, {} decoded, {} shown ({:.1} fps), {} dropped (never taken); outputs: {} \
             ticks, {} repeated the last picture; MJPEG decode ms p50 {d50:.1} p99 {d99:.1} max {dm:.1}; arrival → \
             shown ms p50 {l50:.1} p99 {l99:.1} max {lm:.1}",
            secs,
            s.read,
            s.seq,
            self.shown,
            self.shown as f64 / secs.max(1e-9),
            s.skip,
            s.ticks,
            s.rep
        )
    }
}

impl Drop for Webcam {
    fn drop(&mut self) {
        self.quit.store(true, Relaxed);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

impl crate::live::Source for Webcam {
    fn name(&self) -> &str {
        "Webcam Mode"
    }

    fn tick(&mut self, at: u64) {
        self.take(at, true);
    }

    fn ready(&self) -> bool {
        self.pic.is_some()
    }

    fn draw(&mut self, t: &wgpu::TextureView, [w, h]: [u32; 2], pano: bool) {
        let s = [w as f32, h as f32];
        let (v, vp) = match (pano, self.pano()) {
            (_, Some(false)) => (self.view, self.fit(s)),
            (true, _) => (
                View {
                    mode: Mode::Equirect,
                    ..View::default()
                },
                [0.0, 0.0, s[0], s[1]],
            ),
            _ => (self.view, [0.0, 0.0, s[0], s[1]]),
        };
        self.draw_into(t, &v, vp);
        self.drawn += 1;
    }

    fn asc(&mut self) -> Option<&[u8]> {
        None
    }

    fn audio(&mut self, _: &mut dyn FnMut(u64, &[u8])) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The captures in `X5_WEBCAM` (raw MJPEG of the X5's Webcam Mode; they show a person: nothing derived from them
    /// is written anywhere) through the whole path: replayed at 30 fps, decoded by the two threads, uploaded and drawn
    /// by the preview (2880 × 1440: the equirect as a 2:1 equirect, which must give the picture back) or blitted
    /// (1920 × 1080). Compared on the CPU with the decoder's own RGBA.
    #[test]
    fn captures() {
        let Some(d) = std::env::var_os("X5_WEBCAM").map(PathBuf::from) else {
            eprintln!("X5_WEBCAM not set: skipping");
            return;
        };
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        // (the first captures, named by the size asked for, hold the stale frames of the other mode)
        for (f, w, h) in [
            ("t_2880x1440.mjpg", 2880u32, 1440u32),
            ("t_1920x1080.mjpg", 1920, 1080),
        ] {
            let p = d.join(f);
            let mut c = Webcam::new(&g, &cx, Input::File(p.clone()), || {});
            let t = g.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            // a viewer at 60 Hz for 4 s
            let tv = t.create_view(&Default::default());
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_secs(4) {
                c.show(&tv, [w, h], false);
                thread::sleep(Duration::from_millis(16));
            }
            assert!(c.err().is_none(), "{:?}", c.err());
            println!("{f}: {}", c.report());
            assert_eq!(c.pano(), Some(w == 2 * h));
            // half the rate at least: the host and the GPU are shared, the numbers above say how it went
            assert!(c.shown >= 60, "{} pictures in 4 s", c.shown);
            // a 30 fps output clock as the Live screen's, the viewer beside it: every picture taken in turn
            let k = |c: &Webcam| {
                let s = c.sh.lock().unwrap();
                [s.ticks, s.rep, s.skip, s.seq]
            };
            let (k0, s0) = (k(&c), c.shown);
            let (per, mut next, mut ui) = (1_000_000_000 / 30, mono(), Instant::now());
            let end = next + 4_000_000_000;
            while next < end {
                if mono() >= next {
                    crate::live::Source::tick(&mut c, next);
                    crate::live::Source::draw(&mut c, &tv, [w, h], true);
                    next += per;
                }
                if ui.elapsed() >= Duration::from_micros(16_667) {
                    ui = Instant::now();
                    c.show(&tv, [w, h], false);
                }
                thread::sleep(Duration::from_millis(1));
            }
            let [tk, r, d, n] = std::array::from_fn(|i| k(&c)[i] - k0[i]);
            println!(
                "{f}: output clock {tk} ticks, {r} repeated, {d} dropped, {n} decoded; viewer drew {}",
                c.shown - s0
            );
            assert!(tk >= 115 && r <= 2 && d <= 2, "{r} repeated, {d} dropped");
            // the drawn picture against the decoder's: the same pixels when the view is the picture itself; the
            // decoder held, the newest one queued is taken (its buffer goes spare, untouched) and compared
            c.quit.store(true, Relaxed);
            thread::sleep(Duration::from_millis(100));
            assert!(c.take(u64::MAX, false).is_some());
            let px = c.sh.lock().unwrap().free.last().unwrap().clone();
            let v = View {
                mode: Mode::Equirect,
                ..View::default()
            };
            c.draw_into(&tv, &v, [0.0, 0.0, w as f32, h as f32]);
            let out = stitch::read_texture(&g.device, &g.queue, &t);
            let mae = out
                .iter()
                .zip(&px)
                .enumerate()
                .filter(|(i, _)| i % 4 != 3)
                .map(|(_, (a, b))| (*a as i32 - *b as i32).abs() as f64)
                .sum::<f64>()
                / (w * h * 3) as f64;
            println!("{f}: drawn vs decoded mean abs difference {mae:.2} / 255");
            assert!(mae < 3.0);
        }
    }
}
