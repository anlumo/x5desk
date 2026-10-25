//! Plain videos, without an Insta360 trailer (our own exports): a 2:1 picture is an equirect shown in every view mode
//! through crate stitch's preview, any other plays flat. As in the player the audio clock (or the silent one) is the
//! master and the media thread (`clip`) decodes ahead; each UI frame takes the newest frame due, converts it from NV12
//! to RGBA (`movie.wgsl`) and draws it. 10-bit H.265 does not decode yet (crate video decodes 8-bit).

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

const OUT: u64 = video::decode::OUT as u64;

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
        g.queue
            .write_buffer(&o, 0, bytemuck::cast_slice(&[x, y, 0, 0]));
        let bg = (0..OUT)
            .map(|s| {
                let t = v.rings[0].texture(s);
                let pl = |format, aspect| {
                    t.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(format),
                        aspect,
                        ..Default::default()
                    })
                };
                d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("movie"),
                    layout: &pipe.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&pl(
                                wgpu::TextureFormat::R8Unorm,
                                wgpu::TextureAspect::Plane0,
                            )),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&pl(
                                wgpu::TextureFormat::Rg8Unorm,
                                wgpu::TextureAspect::Plane1,
                            )),
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

    fn seek(&mut self, k: usize) {
        let k = k.min(self.v.times.len() - 1);
        self.v.seek(&self.g, k, self.held.take());
        if self.playing {
            self.audio.cmd(Cmd::Pause);
            self.resume = true;
        }
        self.audio.cmd(Cmd::Seek(self.v.times[k] - self.a_off));
        self.want = Some(k);
    }

    fn play(&mut self, on: bool) {
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
                    rp.set_bind_group(0, &self.bg[(p.f[0] % OUT) as usize], &[]);
                    rp.draw(0..3, 0..1);
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
