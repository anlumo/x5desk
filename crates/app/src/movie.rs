//! Plain videos, without an Insta360 trailer (our own exports): a 2:1 picture is an equirect shown in every view mode
//! through crate stitch's preview, any other plays flat. As in the player the audio clock (or the silent one) is the
//! master and the media thread (`clip`) decodes ahead; each UI frame takes the newest frame due, converts it from NV12
//! or P010 (H.265 Main 10) to RGBA (`movie.wgsl`) and draws it.

use crate::{
    clip::{self, Pic, Video},
    player::{self, Aud, Cmd, FMT, Look, Tgt},
};
use eframe::{egui, egui_wgpu::RenderState};
use media::audio;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use stitch::{Ctx, Preview, View};
use video::Gpu;

pub struct Movie {
    g: Arc<Gpu>,
    pub path: PathBuf,
    v: Video,
    audio: Aud,
    clock: audio::Clock,
    a_off: f64,
    /// an audio track plays; its volume
    sound: bool,
    vol: f32,
    held: Option<Pic>,
    /// a seek target not shown yet; resume the audio once it is
    want: Option<usize>,
    resume: bool,
    playing: bool,
    /// the conversion: pipeline, a bind group per ring slot; its RGBA picture and that bound as an equirect
    pipe: wgpu::RenderPipeline,
    bg: Vec<wgpu::BindGroup>,
    rgba: wgpu::Texture,
    rv: wgpu::TextureView,
    pano: stitch::Pano,
    pv: Preview,
    look: Look,
    view: View,
    tgt: Option<Tgt>,
    flat: Option<(egui::TextureId, RenderState)>,
    size: [u32; 2],
    /// a frame not converted yet; the view changed
    fresh: bool,
    dirty: bool,
    pub err: Option<String>,
}

impl Movie {
    pub fn open(
        g: &Arc<Gpu>,
        cx: &Ctx,
        p: &Path,
        ctx: &egui::Context,
        look: Option<Look>,
    ) -> Result<Movie, String> {
        let c = ctx.clone();
        let v = Video::open(g, &[p.to_path_buf()], move || c.request_repaint())?;
        let (audio, clock, a_off, sound) = player::audio(p, v.end);
        let d = &g.device;
        let [x, y, w, h] = v.view;
        let rgba = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("movie"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FMT,
            // copied out by the tests
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let rv = rgba.create_view(&Default::default());
        let m = d.create_shader_module(wgpu::include_wgsl!("movie.wgsl"));
        let pipe = d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("movie"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &m,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &m,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(FMT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let o = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("movie window"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let deep = v.rings[0].deep();
        g.queue
            .write_buffer(&o, 0, bytemuck::cast_slice(&[x, y, deep as u32, 0]));
        let bg = (0..v.rings[0].slots() as u64)
            .map(|s| {
                // the plane's own format: R8 / RG8, or R16 / RG16 for P010
                let [y, uv] = v.rings[0].planes(s);
                d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("movie"),
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
                            resource: o.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        let look = look.unwrap_or(Look::Perspective);
        Ok(Movie {
            g: g.clone(),
            path: p.to_path_buf(),
            pano: cx.pano(&rv),
            pv: Preview::pano(cx, FMT),
            v,
            audio,
            clock,
            a_off,
            sound,
            vol: 1.0,
            held: None,
            want: None,
            resume: false,
            playing: false,
            pipe,
            bg,
            rgba,
            rv,
            look,
            view: look.view(),
            tgt: None,
            flat: None,
            size: [0; 2],
            fresh: false,
            dirty: true,
            err: None,
        })
    }

    fn sphere(&self) -> bool {
        self.rgba.width() == 2 * self.rgba.height()
    }

    /// A frame is shown and no seek is pending.
    pub fn idle(&self) -> bool {
        self.err.is_some()
            || (self.held.is_some() && self.want.is_none() && !self.fresh && !self.dirty)
    }

    pub fn vol(&mut self, v: f32) {
        self.vol = v.clamp(0.0, 1.0);
        self.audio.cmd(Cmd::Vol(self.vol));
    }

    /// Shows frame `k` next; playing goes on from there once it is shown.
    pub fn seek(&mut self, k: usize) {
        let k = k.min(self.v.times.len() - 1);
        self.v.seek(&self.g, k, self.held.take());
        if self.playing {
            self.audio.cmd(Cmd::Pause);
            self.resume = true;
        }
        self.audio.cmd(Cmd::Seek(self.v.times[k] - self.a_off));
        self.want = Some(k);
    }

    pub fn play(&mut self, on: bool) {
        self.playing = on;
        if !on {
            self.resume = false;
            self.audio.cmd(Cmd::Pause);
        } else if self
            .held
            .is_some_and(|p| p.t + 1.0 / 60.0 >= self.v.times[self.v.times.len() - 1])
        {
            self.seek(0);
        } else if self.want.is_some() {
            self.resume = true;
        } else {
            self.audio.cmd(Cmd::Play);
        }
    }

    /// Every UI frame: the clock, the frame due, the end.
    pub fn tick(&mut self, ctx: &egui::Context) {
        if let Some(e) = self.v.err() {
            self.err = Some(e);
        }
        if self.playing {
            ctx.request_repaint();
        }
        let Some(now) = self.clock.get().map(|c| c + self.a_off) else {
            ctx.request_repaint_after(Duration::from_millis(2));
            return;
        };
        if self.playing && self.want.is_none() && now >= self.v.end {
            self.play(false);
            return;
        }
        if let (Some(p), _) = self.v.take(now, false) {
            self.v.release_before(&self.g, &p);
            if self.want.take().is_some() && self.resume {
                self.resume = false;
                self.audio.cmd(Cmd::Play);
            }
            (self.held, self.fresh) = (Some(p), true);
        }
    }

    /// Records frame `p`'s conversion into the RGBA picture; the caller acquires `p` for the submit.
    fn yuv(&self, e: &mut wgpu::CommandEncoder, p: &Pic) {
        let mut rp = e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("movie"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.rv,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        rp.set_pipeline(&self.pipe);
        rp.set_bind_group(0, &self.bg[p.f[0] as usize % self.bg.len()], &[]);
        rp.draw(0..3, 0..1);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        if ui.input(|i| i.key_pressed(egui::Key::Space)) {
            self.play(!self.playing);
        }
        egui::Panel::bottom("movie controls").show(ui, |ui| {
            ui.add_space(4.0);
            let (n, end) = (self.v.times.len(), self.v.end);
            let k = self.want.or(self.held.map(|p| p.k)).unwrap_or(0);
            let mut t = self.v.times[k];
            ui.spacing_mut().slider_width = ui.available_width();
            if ui
                .add(egui::Slider::new(&mut t, 0.0..=end).show_value(false))
                .changed()
            {
                let j = clip::frame_at(&self.v.times, t);
                if j != k {
                    self.seek(j);
                }
            }
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button(if self.playing { "Pause" } else { "Play" })
                    .clicked()
                {
                    self.play(!self.playing);
                }
                let mmss = |t: f64| format!("{}:{:05.2}", (t / 60.0) as u32, t % 60.0);
                ui.monospace(format!(
                    "{k:>5} / {n}  {} / {}",
                    mmss(self.v.times[k]),
                    mmss(end)
                ));
                ui.separator();
                // the slider sits in a horizontal of its own, which never wraps (as in the player)
                if ui.available_size_before_wrap().x < 160.0 {
                    ui.end_row();
                }
                ui.label(if self.sound { "Volume" } else { "No audio" });
                let mut v = self.vol;
                if ui
                    .add_enabled(
                        self.sound,
                        egui::Slider::new(&mut v, 0.0..=1.0).show_value(false),
                    )
                    .changed()
                {
                    self.vol(v);
                }
                ui.separator();
                ui.monospace(format!("{} × {}", self.rgba.width(), self.rgba.height()));
                if self.sphere() {
                    ui.separator();
                    for (l, name) in Look::ALL {
                        if ui.selectable_label(self.look == l, name).clicked() && self.look != l {
                            self.look = l;
                            self.view = View {
                                yaw: self.view.yaw,
                                ..l.view()
                            };
                            self.dirty = true;
                        }
                    }
                }
            });
            if let Some(e) = &self.err {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
            ui.add_space(2.0);
        });
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let (rect, resp) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let sphere = self.sphere();
            let r = match sphere {
                true => self.look.fit(rect),
                false => {
                    let (w, h) = (self.rgba.width() as f32, self.rgba.height() as f32);
                    let s = (rect.width() / w).min(rect.height() / h);
                    egui::Rect::from_center_size(rect.center(), egui::vec2(w * s, h * s))
                }
            };
            if sphere {
                self.dirty |= self.look.steer(ui, &resp, r, &mut self.view);
            }
            let ppp = ui.ctx().pixels_per_point();
            let size = [r.width(), r.height()].map(|x| ((x * ppp).round() as u32).max(1));
            if let Some(p) = self.held
                && (self.fresh || self.dirty || (sphere && size != self.size))
            {
                let g = &self.g;
                let mut e = g.device.create_command_encoder(&Default::default());
                if self.fresh {
                    self.yuv(&mut e, &p);
                }
                if sphere {
                    Tgt::fit(&mut self.tgt, g, rs, size);
                    let t = self.tgt.as_ref().unwrap();
                    self.pv
                        .render(&mut e, &t.view, size, &self.view, &self.pano);
                }
                if self.fresh {
                    self.v.acquire(g, &p);
                }
                g.queue.submit([e.finish()]);
                (self.fresh, self.dirty, self.size) = (false, false, size);
            }
            let id = match sphere {
                true => self.tgt.as_ref().map(|t| t.id),
                false => Some(
                    self.flat
                        .get_or_insert_with(|| {
                            let id = rs.renderer.write().register_native_texture(
                                &self.g.device,
                                &self.rv,
                                wgpu::FilterMode::Linear,
                            );
                            (id, rs.clone())
                        })
                        .0,
                ),
            };
            let p = ui.painter_at(rect);
            p.rect_filled(rect, 0.0, egui::Color32::BLACK);
            if let Some(id) = id {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                p.image(id, r, uv, egui::Color32::WHITE);
            }
        });
    }
}

impl Drop for Movie {
    fn drop(&mut self) {
        if let Some(h) = self.held.take() {
            self.v.release(&self.g, &h);
        }
        if let Some((id, rs)) = &self.flat {
            rs.renderer.write().free_texture(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{Exp, Set, Spec};
    use std::time::Instant;

    /// FFmpeg's decode of frame `k`: luma and the interleaved chroma, as codes in 8-bit units (10-bit codes / 4).
    fn ffmpeg(p: &Path, k: usize, deep: bool) -> Vec<f64> {
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(p)
            .args(["-vf", &format!("select=eq(n\\,{k})")])
            .args([
                "-fps_mode",
                "passthrough",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
            ])
            .args(["-pix_fmt", if deep { "p010le" } else { "nv12" }, "-"])
            .output()
            .expect("ffmpeg");
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        match deep {
            // P010: the code in the high 10 bits
            true => (o.stdout.chunks(2))
                .map(|b| (u16::from_le_bytes([b[0], b[1]]) >> 6) as f64 / 4.0)
                .collect(),
            false => o.stdout.iter().map(|&b| b as f64).collect(),
        }
    }

    /// The viewer on 360 exports of v635 (`X5_SAMPLES`) through the Export screen's queue, H.265 10-bit (P010
    /// pictures) and 8-bit (NV12): a P-frame after a seek converts to FFmpeg's decode through BT.709 limited range with
    /// nearest chroma (`movie.wgsl`), every channel at most 1 level off and nearly all exact.
    #[test]
    fn deep() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES unset: skipped");
            return;
        };
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        let dir = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-movie-{}", std::process::id()));
        // the log prints each job's end
        let mut e = Exp::new(&g, &cx, &ctx, Some(dir.clone()), true);
        let mut s = Spec::of(&d.join("v635.insv")).unwrap();
        s.dur = 0.5;
        e.source(s);
        // in this order: v635_360_1920.mp4, then -2
        for deep in [true, false] {
            let x = Set {
                preset: None,
                w: 1920,
                deep,
                qp: Some(20),
                raft: false,
                stab: None,
                ..Set::default()
            };
            e.queue(x, false).unwrap();
        }
        let t0 = Instant::now();
        while !e.idle() {
            assert!(t0.elapsed().as_secs() < 120, "exports stalled");
            std::thread::sleep(Duration::from_millis(10));
        }
        e.tick();
        drop(e);
        let k = 9;
        for (f, deep) in [("v635_360_1920.mp4", true), ("v635_360_1920-2.mp4", false)] {
            let p = dir.join(f);
            let mut m = Movie::open(&g, &cx, &p, &ctx, None).unwrap();
            let fmt = m.v.rings[0].deep();
            let [w, h] = [m.rgba.width(), m.rgba.height()].map(|x| x as usize);
            assert_eq!((fmt, m.sphere(), w), (deep, true, 1920));
            m.seek(k);
            let t0 = Instant::now();
            while m.held.is_none_or(|p| p.k != k) {
                m.tick(&ctx);
                assert_eq!(
                    (m.err.as_deref(), t0.elapsed().as_secs() < 10),
                    (None, true)
                );
                std::thread::sleep(Duration::from_millis(2));
            }
            let pic = m.held.unwrap();
            let mut enc = g.device.create_command_encoder(&Default::default());
            m.yuv(&mut enc, &pic);
            m.v.acquire(&g, &pic);
            g.queue.submit([enc.finish()]);
            let px = stitch::read_texture(&g.device, &g.queue, &m.rgba);
            let v = ffmpeg(&p, k, deep);
            assert_eq!(v.len(), w * h * 3 / 2);
            let (y, uv) = v.split_at(w * h);
            let (mut off, mut max) = (0, 0.0f64);
            for (i, q) in px.chunks(4).enumerate() {
                let (r, c) = (i / w, i % w);
                let l = (y[i] - 16.0) / 219.0;
                let j = r / 2 * w + c / 2 * 2;
                let (cb, cr) = ((uv[j] - 128.0) / 224.0, (uv[j + 1] - 128.0) / 224.0);
                let rgb = [
                    l + 1.5748 * cr,
                    l - 0.1873 * cb - 0.4681 * cr,
                    l + 1.8556 * cb,
                ];
                for (a, b) in q.iter().zip(rgb) {
                    let x = b.clamp(0.0, 1.0) * 255.0;
                    off += (*a as f64 != x.round()) as usize;
                    max = max.max((*a as f64 - x).abs());
                }
            }
            eprintln!(
                "{f} ({fmt:?}), frame {k}: {off} of {} channels not the nearest level, at most {max:.3} off",
                w * h * 3
            );
            // Vulkan allows either neighbour; RADV writes 8-bit targets through f16 rounded toward zero, so a level
            // and 0.5 to ~0.63 can come out a level low. The 8-bit scale on P010 (255 instead of 65535 / 256) is up
            // to 2.6 levels off on this frame.
            assert!(max < 0.7, "{max}");
            // and plays on to the last frame, muted
            m.vol(0.0);
            m.play(true);
            while m.held.is_some_and(|p| p.k + 1 < m.v.times.len()) {
                m.tick(&ctx);
                assert_eq!(
                    (m.err.as_deref(), t0.elapsed().as_secs() < 20),
                    (None, true)
                );
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
    /// A B-frame H.264 clip (ffmpeg CLI, libx264 `-bf 1`, frame numbers burned in): a seek shows the frame asked for,
    /// with that index under the slider (frame 60 showed labelled 61, 58 for 59), and it plays on in order to the end.
    #[test]
    fn bframes() {
        // not "movie": deep's dir is x5desk-movie-<pid>, and this one's drop would remove it under deep
        let dir = crate::clip::tests::Tmp::new("movie-bf");
        let p = dir.0.join("bf1.mp4");
        if !crate::clip::tests::numbered(&p, &["-c:v", "libx264", "-bf", "1"]) {
            return;
        }
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        let mut m = Movie::open(&g, &cx, &p, &ctx, None).unwrap();
        let w = m.rgba.width() as usize;
        // the number in the picture shown (white blocks along the top)
        let shown = |m: &Movie| {
            let pic = m.held.unwrap();
            let mut enc = g.device.create_command_encoder(&Default::default());
            m.yuv(&mut enc, &pic);
            m.v.acquire(&g, &pic);
            g.queue.submit([enc.finish()]);
            let px = stitch::read_texture(&g.device, &g.queue, &m.rgba);
            (0..8)
                .map(|i| ((px[(20 * w + 40 * i + 20) * 4] > 128) as usize) << i)
                .sum::<usize>()
        };
        let t0 = Instant::now();
        for k in [60, 137, 59, 150] {
            m.seek(k);
            while m.held.is_none_or(|p| p.k != k) {
                m.tick(&ctx);
                assert_eq!(
                    (m.err.as_deref(), t0.elapsed().as_secs() < 20),
                    (None, true)
                );
                std::thread::sleep(Duration::from_millis(2));
            }
            eprintln!(
                "seek {k}: frame {} shown, number {}",
                m.held.unwrap().k,
                shown(&m)
            );
            assert_eq!(shown(&m), k);
        }
        m.play(true);
        let mut last = 150;
        while m.held.is_some_and(|p| p.k + 1 < m.v.times.len()) {
            m.tick(&ctx);
            let k = m.held.unwrap().k;
            assert!(
                k >= last && t0.elapsed().as_secs() < 30,
                "playback at {k} after {last}"
            );
            last = k;
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!((shown(&m), m.err.as_deref()), (199, None));
    }
}
