//! The X5's Webcam Mode as a source (PLAN.md §3.2): its MJPEG over UVC (`media::uvc`) decoded on a thread of its own
//! (`media::mjpeg`), uploaded, and drawn: the 2880 × 1440 in-camera equirect through crate stitch's preview (`Ctx::pano`, every view mode), the
//! 1920 × 1080 flat picture as it is. Offline it replays a raw MJPEG capture at the camera's 30 fps.
//!
//! Hand-over as the live preview's (`feed`): up to 3 decoded pictures queue, each due at its JPEG's arrival plus the
//! slowest decode of the last 32, and a consumer takes them in turn ([`feed::pick`]); while the live outputs run they
//! hold it and the viewer shows what they take. The outputs keep the slack they found at their first picture for as
//! long as they run, so their pictures come a steady time after their JPEGs.
//!
//! The camera's microphone (`media::uvc::mic`) runs with the stream as its sound: AAC-LC for the live outputs, each
//! packet heard as long after its sound came as the outputs show the pictures after theirs ([`Webcam::delay`]), both
//! timed by their arrival on CLOCK_MONOTONIC.

use crate::{
    feed::{self, IDLE, LATE, Samples, mono},
    player::{FMT, Look, Tgt},
};
use eframe::{egui, egui_wgpu::RenderState};
use media::{
    aacenc,
    mjpeg::{self, Mjpeg},
    uvc::mic::{self, Mic},
};
use std::{
    collections::VecDeque,
    fmt::Write,
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
    /// the microphone's AAC packets not handed out yet: per packet when its first frame came (CLOCK_MONOTONIC ns)
    /// and its end in `ab`; their AudioSpecificConfig while it runs, why it does not; packets made, stretches lost
    ai: Vec<(u64, usize)>,
    ab: Vec<u8>,
    asc: Option<[u8; 2]>,
    note: Option<String>,
    made: u64,
    lost: u64,
    /// every period it gave: when its first frame came, the samples
    #[cfg(test)]
    raw: Vec<(u64, Vec<f32>)>,
}

fn lock(sh: &Mutex<Sh>) -> std::sync::MutexGuard<'_, Sh> {
    sh.lock().unwrap_or_else(|e| e.into_inner())
}

/// The slowest decode of the last 32 (arrival → decoded, ns): a picture falls due that long after its JPEG came.
fn slack(s: &Sh) -> u64 {
    s.late[..s.nl.min(32)]
        .iter()
        .max()
        .map_or(0, |&l| l.min(66_000_000))
}

/// AAC-LC of the microphone, kbit/s (stereo): the live outputs encode it again at their own rate.
const MKBPS: u32 = 256;

/// The camera's microphone into `sh` as AAC-LC (its own thread encodes), for as long as the handle lives; None, and a
/// note in `sh`, when it is missing or fails.
fn mic(sh: &Arc<Mutex<Sh>>) -> Option<Mic> {
    let (s2, s3) = (sh.clone(), sh.clone());
    let new = || aacenc::Enc::new(mic::RATE, 2, MKBPS * 1000);
    let (mut enc, mut pi, mut pb) = (None, vec![], vec![]);
    let m = new().and_then(|e| {
        let asc = e.asc();
        let m = Mic::open(
            move |x, t0, k| {
                // a stream (the first, or the next after lost input) has its own encoder: pts count from its start
                if k == 0 {
                    enc = new().ok();
                }
                let Some(e) = enc.as_mut() else { return };
                pi.clear();
                pb.clear();
                // pts in frames from the stream's start (the encoder's priming frame at −1024)
                e.push(x, |au, pts, _| {
                    pb.extend_from_slice(au);
                    pi.push(((t0 as i64 + mic::ns(pts)) as u64, pb.len()));
                });
                let mut s = lock(&s2);
                // a second nobody took (no output listens) is of no use
                if s.ai.len() > 47 {
                    s.ai.clear();
                    s.ab.clear();
                }
                let o = s.ab.len();
                s.ab.extend_from_slice(&pb);
                s.ai.extend(pi.iter().map(|&(t, e)| (t, o + e)));
                s.lost += (k == 0 && s.made > 0) as u64;
                s.made += pi.len() as u64;
                #[cfg(test)]
                s.raw.push((t0 + mic::ns(k as i64) as u64, x.to_vec()));
            },
            move |m| lock(&s3).note = Some(format!("No sound from the X5's microphone ({m})")),
        )?;
        lock(sh).asc = Some(asc);
        Ok(m)
    });
    m.map_err(|e| lock(sh).note = Some(format!("No sound from the X5's microphone ({e})")))
        .ok()
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
            let m = mic(&sh);
            // a stop waits for the reader on the UI thread: this long at most (a mode switch sends nothing for ~1.7 s),
            // then the closes (in a mode switch: the camera's up to 36 ms, its microphone's 19 ms measured)
            let e = loop {
                if quit.load(Relaxed) {
                    break None;
                }
                match u.next(&mut j, Duration::from_millis(20)) {
                    Ok(true) => dec(&j),
                    Ok(false) => {}
                    Err(e) => break Some(e),
                }
            };
            // the microphone opens after the camera and closes after it: an X5 held the next open ~0.5 s after one
            // where it opened first or beside it (10 of 12 cycles), and the camera's close after it closed first (1 of
            // 6, opened after it)
            drop(u);
            drop(m);
            if let Some(e) = e {
                fail(format!("Webcam Mode: {e}"))
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
    /// JPEG arrival → output tick of the pictures the outputs took, ms
    pub olat: Samples,
    /// while the outputs run: their pictures fall due this long (ns) after their JPEG came, the decode's slack when
    /// they took their first one
    od: Option<u64>,
    /// the microphone's AudioSpecificConfig as handed out
    asc: Option<[u8; 2]>,
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
            olat: Samples::default(),
            od: None,
            asc: None,
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

    /// Why the stream has no sound (its microphone missing or failed): the live outputs send silence.
    pub fn note(&self) -> Option<String> {
        self.sh.lock().unwrap().note.clone()
    }

    /// Pictures taken (by the viewer or a live output).
    pub fn taken(&self) -> u64 {
        self.sh.lock().unwrap().took
    }

    /// How long after it came (ns) the outputs hear the microphone's sound: as long as they show the pictures after
    /// theirs. A picture falls due its slack after it came, and a 30 fps clock takes it half a frame (a quarter of the
    /// pick window) after that on average, its lag sweeping a frame as the camera's 28.8–29.7 fps fall behind it
    /// (17–21 ms measured at 2880 × 1440; 34 ms at 1920 × 1080, whose delivery jitter keeps a picture more queued:
    /// there the sound comes ~half a frame early). None before the outputs took a picture.
    pub fn delay(&self) -> Option<u64> {
        self.od.map(|d| d + LATE / 4)
    }

    /// Uploads the picture due at `at` (`feed::pick`); returns its arrival time if it took one. `out`: for the live
    /// outputs, which then hold the hand-over; the viewer takes none meanwhile.
    fn take(&mut self, at: u64, out: bool) -> Option<u64> {
        let mut s = self.sh.lock().unwrap();
        let t = mono();
        let ran = t < s.out + IDLE;
        if out {
            s.out = t;
        } else if ran {
            return None;
        }
        // as the preview's outputs: the first picture once it is due a quarter of the window; their slack is set again
        // with it
        if out && !ran {
            (self.start, self.od) = (true, None);
        }
        let now = match out && self.start {
            true => at.saturating_sub(LATE / 4),
            false => at,
        };
        let l = self.od.filter(|_| out).unwrap_or_else(|| slack(&s));
        let i = feed::pick(s.q.iter().map(|p| p.got + l), now);
        if out && self.pic.is_some() {
            s.ticks += 1;
            s.rep += (ran && i.is_none()) as u64;
        }
        let i = i?;
        self.start &= !out;
        if out {
            self.od.get_or_insert(l);
        }
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
        if out {
            self.olat.push(at.saturating_sub(p.got) as f32 * 1e-6);
        }
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

    /// Ends the stream now: the reader thread and with it the device, also while the live outputs still hold this
    /// source (they let go of it only after their pass; it keeps its last picture meanwhile).
    pub fn stop(&mut self) {
        self.quit.store(true, Relaxed);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }

    /// What was counted: the viewer's part while it showed pictures, the outputs' while they ran (the viewer shows
    /// none while another screen is up), the microphone's.
    pub fn report(&self) -> String {
        let s = self.sh.lock().unwrap();
        let secs = self.t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let ms = |x: &Samples| {
            let [a, b, c] = x.pct();
            format!("ms p50 {a:.1} p99 {b:.1} max {c:.1}")
        };
        let (w, h) = self
            .pic
            .as_ref()
            .map_or((0, 0), |p| (p.tex.width(), p.tex.height()));
        let mut r = format!(
            "{secs:.1} s: {w} × {h}, {} JPEGs read, {} decoded, {} dropped (never taken)",
            s.read, s.seq, s.skip
        );
        if s.dec.n > 0 {
            let _ = write!(r, "; MJPEG decode {}", ms(&s.dec));
        }
        if self.shown > 0 {
            let fps = self.shown as f64 / secs.max(1e-9);
            let _ = write!(
                r,
                "; viewer: {} shown ({fps:.1} fps), arrival → shown {}",
                self.shown,
                ms(&self.lat)
            );
        }
        if s.ticks > 0 {
            let _ = write!(
                r,
                "; outputs: {} ticks, {} repeated the last picture",
                s.ticks, s.rep
            );
            if self.olat.n > 0 {
                let _ = write!(r, ", arrival → tick {}", ms(&self.olat));
            }
        }
        match (&s.note, s.asc) {
            (Some(n), _) => r = r + "; " + n,
            (None, Some(_)) => {
                let _ = write!(
                    r,
                    "; microphone: {} AAC packets, {} stretches lost",
                    s.made, s.lost
                );
                if let Some(d) = self.delay() {
                    let _ = write!(r, ", heard {:.1} ms after it came", d as f64 / 1e6);
                }
            }
            _ => {}
        }
        r
    }

    /// Every period the microphone gave so far: when its first frame came, its samples (taken out).
    #[cfg(test)]
    pub fn raw(&self) -> Vec<(u64, Vec<f32>)> {
        std::mem::take(&mut lock(&self.sh).raw)
    }
}

impl Drop for Webcam {
    fn drop(&mut self) {
        self.stop()
    }
}

impl crate::live::Source for Webcam {
    /// Without its microphone (the Capture screen says why) the outputs' sound is silence.
    fn name(&self) -> &str {
        match lock(&self.sh).note {
            Some(_) => "Webcam Mode without sound",
            None => "Webcam Mode",
        }
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
        self.asc = lock(&self.sh).asc;
        self.asc.as_ref().map(|a| &a[..])
    }

    /// The microphone's packets, each heard [`Webcam::delay`] after its sound came, with the pictures that came with
    /// it. None before the outputs took a picture: what came meanwhile follows with their first one, from before it.
    fn audio(&mut self, f: &mut dyn FnMut(u64, &[u8])) {
        let Some(d) = self.delay() else { return };
        let mut s = lock(&self.sh);
        let Sh { ai, ab, .. } = &mut *s;
        let mut at = 0;
        for &(t, end) in ai.iter() {
            f(t + d, &ab[at..end]);
            at = end;
        }
        ai.clear();
        ab.clear();
    }
}

/// The camera is one device: the tests that stream from it take turns.
#[cfg(test)]
pub static CAM: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::live::Source;

    /// `n` frames of ffmpeg's test picture at `w` × `h` as a raw MJPEG capture in `d` (ffmpeg CLI, test tool); None
    /// when it made none.
    pub(crate) fn synth(d: &std::path::Path, w: u32, h: u32, n: u32) -> Option<PathBuf> {
        let f = d.join(format!("{w}x{h}.mjpg"));
        let made = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i"])
            .arg(format!("testsrc2=size={w}x{h}:rate=30"))
            .args(["-frames:v", &n.to_string(), "-c:v", "mjpeg", "-f", "mjpeg"])
            .arg(&f)
            .status();
        made.is_ok_and(|s| s.success()).then_some(f)
    }

    /// A replayed capture (made by [`synth`]; skipped without ffmpeg) taken by the outputs alone, as while the Live
    /// screen shows: the report gives what was counted and nothing for the viewer, which showed nothing.
    #[test]
    fn report_outputs() {
        let t = crate::clip::tests::Tmp::new("webcam-report");
        let Some(f) = synth(&t.0, 1920, 1080, 3) else {
            eprintln!("ffmpeg made no MJPEG: skipping");
            return;
        };
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let mut c = Webcam::new(&g, &cx, Input::File(f), || {});
        let t0 = Instant::now();
        while c.taken() < 10 {
            assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", c.err());
            c.tick(mono());
            thread::sleep(Duration::from_millis(33));
        }
        let r = c.report();
        println!("{r}");
        assert!(!r.contains("NaN") && r.contains("; outputs: ") && !r.contains("viewer"));
    }

    /// The X5 in Webcam Mode with its microphone held by another stream (skipped without it; nothing of what it shows
    /// or hears is kept): the pictures come, the sound does not. No AudioSpecificConfig (the outputs send silence), a
    /// note for the Capture screen, its report and the Live screen's name of the source.
    #[test]
    fn mic_busy() {
        if media::uvc::Uvc::find().is_err() {
            eprintln!("no X5 in Webcam Mode: skipping");
            return;
        }
        let _cam = CAM.lock().unwrap_or_else(|e| e.into_inner());
        let held = Mic::open(|_, _, _| {}, |e| panic!("{e}")).unwrap();
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let mut c = Webcam::new(&g, &cx, Input::Uvc(1920, 1080), || {});
        let t0 = Instant::now();
        while c.taken() < 10 {
            assert!(t0.elapsed() < Duration::from_secs(10), "{:?}", c.err());
            c.tick(mono());
            c.audio(&mut |_, _| panic!("sound"));
            thread::sleep(Duration::from_millis(33));
        }
        let (n, r) = (c.note(), c.report());
        println!("{}: {r}", c.name());
        assert!(c.err().is_none() && c.asc().is_none() && !r.contains("NaN"));
        assert!(n.is_some_and(|n| r.ends_with(&n) && n.starts_with("No sound from the X5's microphone")));
        assert_eq!(c.name(), "Webcam Mode without sound");
        drop((c, held));
    }

    /// The captures in `X5_WEBCAM` (raw MJPEG of the X5's Webcam Mode), and the X5 itself in Webcam Mode when it is
    /// there (they show whoever sits there: nothing derived from them is written anywhere), through the whole path:
    /// replayed at 30 fps or streamed, decoded by the two threads, uploaded and drawn by the preview (2880 × 1440: the
    /// equirect as a 2:1 equirect, which must give the picture back) or blitted (1920 × 1080). Compared on the CPU with
    /// the decoder's own RGBA.
    #[test]
    fn captures() {
        let d = std::env::var_os("X5_WEBCAM").map(PathBuf::from);
        let cam = media::uvc::Uvc::find().is_ok();
        if d.is_none() && !cam {
            eprintln!("X5_WEBCAM not set, no X5 in Webcam Mode: skipping");
            return;
        }
        let _cam = CAM.lock().unwrap_or_else(|e| e.into_inner());
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        // (the first captures, named by the size asked for, hold the stale frames of the other mode)
        let files = d.iter().flat_map(|d| {
            [
                ("t_2880x1440.mjpg", 2880, 1440),
                ("t_1920x1080.mjpg", 1920, 1080),
            ]
            .map(|(f, w, h)| (Input::File(d.join(f)), w, h))
        });
        let uvc = [(2880, 1440), (1920, 1080)].map(|(w, h)| (Input::Uvc(w, h), w, h));
        for (i, w, h) in files.chain(uvc.into_iter().filter(|_| cam)) {
            let f = format!("{i:?}");
            let file = matches!(i, Input::File(_));
            let mut c = Webcam::new(&g, &cx, i, || {});
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
            // a viewer at 60 Hz for 4 s from the first picture (the camera's mode switch takes ~1.9 s)
            let tv = t.create_view(&Default::default());
            let mut t0 = Instant::now();
            while c.pano().is_none() && t0.elapsed() < Duration::from_secs(10) {
                c.show(&tv, [w, h], false);
                thread::sleep(Duration::from_millis(16));
            }
            t0 = Instant::now();
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
            // the camera's 28.8–29.7 fps against the 30 fps clock and its delivery jitter repeat a picture on up to
            // ~5 % of the ticks (0–6 of 121 measured)
            let rep = if file { 2 } else { 12 };
            assert!(tk >= 115 && r <= rep && d <= 2, "{r} repeated, {d} dropped");
            // the drawn picture against the decoder's: the same pixels when the view is the picture itself; the
            // decoder stopped, the newest one queued is taken (its buffer goes spare, untouched; the viewer takes
            // again once the outputs left it alone for `IDLE`) and compared
            c.stop();
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
