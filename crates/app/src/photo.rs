//! Photos: with its stitching on the X5 stitches its photos itself into equirect JPEGs (`.jpg`; 11904 × 5952). A thread
//! decodes the file (crate media, zune-jpeg) to RGBA, the picture goes up as one texture, and the viewer draws it
//! through crate stitch's preview from an equirect in every view mode. Any other picture shows as it is: an `.insp`
//! taken with the stitching off (two fisheyes side by side; their stitching is not here yet), a flat frame grab.

use crate::{
    export::Spec,
    player::{FMT, Look, Tgt},
};
use eframe::{egui, egui_wgpu::RenderState};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread,
};
use stitch::{Ctx, Preview, View};
use video::Gpu;

/// A photo by its extension.
pub fn is(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        ["jpg", "jpeg", "insp"]
            .iter()
            .any(|x| e.eq_ignore_ascii_case(x))
    })
}

/// The file decoded: RGBA8 rows, width, height.
pub fn load(p: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut px = vec![];
    let (w, h) = media::mjpeg::Mjpeg::default()
        .rgba(&b, &mut px)
        .map_err(|e| format!("{}: {e}", p.display()))?;
    Ok((px, w as u32, h as u32))
}

/// Why the picture (`w` × `h`) is not an equirect, if it is not. The Insta360 trailer's file category decides (6
/// equirect; an `.insp` taken with the camera's stitching off is a dual fisheye, 2, possibly 2:1 too); without one (our
/// grabs) a 2:1 picture is an equirect.
pub fn flat(p: &Path, w: u32, h: u32) -> Option<String> {
    match media::insv::Trailer::open(p).map_or(0, |t| t.meta.category) {
        6 => None,
        0 if w == 2 * h => None,
        0 => Some(format!("{w} × {h} is not a 360 photo (2:1)")),
        2 | 5 => Some("an unstitched dual-fisheye photo (the camera's stitching was off): photo stitching is not supported yet".into()),
        c => Some(format!("not a 360 photo (Insta360 file category {c})")),
    }
}

/// `px` as an `Rgba8Unorm` texture (sampled, copy source), or why not.
pub fn upload(g: &Gpu, px: &[u8], w: u32, h: u32) -> Result<wgpu::Texture, String> {
    let max = g.device.limits().max_texture_dimension_2d;
    if w > max || h > max {
        return Err(format!(
            "{w} × {h}: larger than the GPU's {max} px textures"
        ));
    }
    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let t = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("photo"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    g.queue.write_texture(
        t.as_image_copy(),
        px,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: None,
        },
        size,
    );
    Ok(t)
}

/// The picture on the GPU: its texture, bound as an equirect, and for a flat one its egui registration.
struct Pic {
    tex: wgpu::Texture,
    pano: stitch::Pano,
    flat: Option<(egui::TextureId, RenderState)>,
}

impl Drop for Pic {
    fn drop(&mut self) {
        if let Some((id, rs)) = &self.flat {
            rs.renderer.write().free_texture(id);
        }
    }
}

/// the picture, why it is not an equirect
type Loaded = Result<((Vec<u8>, u32, u32), Option<String>), String>;

pub struct Photo {
    g: Arc<Gpu>,
    cx: Ctx,
    pub path: PathBuf,
    rx: Option<mpsc::Receiver<Loaded>>,
    pic: Option<Pic>,
    pv: Preview,
    pub look: Look,
    pub view: View,
    tgt: Option<Tgt>,
    size: [u32; 2],
    dirty: bool,
    /// why the picture shows as it is (not an equirect)
    note: Option<String>,
    pub err: Option<String>,
}

impl Photo {
    /// Starts decoding `p` on a thread of its own; `look`: the first view mode.
    pub fn open(
        g: &Arc<Gpu>,
        cx: &Ctx,
        p: &Path,
        ctx: &egui::Context,
        look: Option<Look>,
    ) -> Photo {
        let (tx, rx) = mpsc::channel();
        let (q, c) = (p.to_path_buf(), ctx.clone());
        thread::Builder::new()
            .name("photo".into())
            .spawn(move || {
                let _ = tx.send(load(&q).map(|(px, w, h)| ((px, w, h), flat(&q, w, h))));
                c.request_repaint();
            })
            .expect("photo thread");
        let look = look.unwrap_or(Look::Perspective);
        Photo {
            g: g.clone(),
            cx: cx.clone(),
            path: p.to_path_buf(),
            rx: Some(rx),
            pic: None,
            pv: Preview::pano(cx, FMT),
            look,
            view: look.view(),
            tgt: None,
            size: [0; 2],
            dirty: true,
            note: None,
            err: None,
        }
    }

    /// The decoded picture goes up when it is in.
    pub fn tick(&mut self) {
        let Some(r) = self.rx.as_ref().and_then(|r| r.try_recv().ok()) else {
            return;
        };
        self.rx = None;
        match r.and_then(|((px, w, h), n)| Ok((upload(&self.g, &px, w, h)?, n))) {
            Ok((tex, n)) => {
                self.note = n;
                let pano = self.cx.pano(&tex.create_view(&Default::default()));
                (self.pic, self.dirty) = (
                    Some(Pic {
                        tex,
                        pano,
                        flat: None,
                    }),
                    true,
                );
            }
            Err(e) => self.err = Some(e),
        }
    }

    /// Picture width, height once decoded.
    pub fn dims(&self) -> Option<[u32; 2]> {
        self.pic.as_ref().map(|p| [p.tex.width(), p.tex.height()])
    }

    fn sphere(&self) -> bool {
        self.pic.is_some() && self.note.is_none()
    }

    /// Decoded (or failed) and drawn.
    pub fn idle(&self) -> bool {
        self.err.is_some() || (self.pic.is_some() && !self.dirty)
    }

    pub fn spec(&self) -> Spec {
        Spec {
            files: vec![self.path.clone()],
            photo: true,
            dur: 0.0,
            t: 0.0,
            view: self.view,
            look: self.look,
            stab: None,
            smooth: 0.0,
            follow: None,
            faces: false,
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        egui::Panel::bottom("photo controls").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                match self.dims() {
                    Some([w, h]) => ui.monospace(format!("{w} × {h}")),
                    None if self.err.is_none() => ui.label("Decoding…"),
                    None => ui.label(""),
                };
                if let Some(n) = &self.note {
                    ui.weak(format!("shown as it is: {n}"));
                }
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
            let p = ui.painter_at(rect);
            p.rect_filled(rect, 0.0, egui::Color32::BLACK);
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            let sphere = self.sphere();
            let Some(pic) = &mut self.pic else { return };
            if !sphere {
                // as it is, fitted
                let id = pic
                    .flat
                    .get_or_insert_with(|| {
                        let v = pic.tex.create_view(&Default::default());
                        let id = rs.renderer.write().register_native_texture(
                            &self.g.device,
                            &v,
                            wgpu::FilterMode::Linear,
                        );
                        (id, rs.clone())
                    })
                    .0;
                let (w, h) = (pic.tex.width() as f32, pic.tex.height() as f32);
                let s = (rect.width() / w).min(rect.height() / h);
                p.image(
                    id,
                    egui::Rect::from_center_size(rect.center(), egui::vec2(w * s, h * s)),
                    uv,
                    egui::Color32::WHITE,
                );
                self.dirty = false;
                return;
            }
            let r = self.look.fit(rect);
            self.dirty |= self.look.steer(ui, &resp, r, &mut self.view);
            let ppp = ui.ctx().pixels_per_point();
            let size = [r.width(), r.height()].map(|x| ((x * ppp).round() as u32).max(1));
            if self.dirty || size != self.size {
                Tgt::fit(&mut self.tgt, &self.g, rs, size);
                let t = self.tgt.as_ref().unwrap();
                let mut e = self.g.device.create_command_encoder(&Default::default());
                self.pv.render(&mut e, &t.view, size, &self.view, &pic.pano);
                self.g.queue.submit([e.finish()]);
                (self.size, self.dirty) = (size, false);
            }
            if let Some(t) = &self.tgt {
                p.image(t.id, r, uv, egui::Color32::WHITE);
            }
        });
    }
}
