//! Timeline thumbnails: a worker thread decodes and stitches a source file's sync frames (effects's `Clip`, template
//! seam; one frame decoded each, so the preview's decoders keep the hardware) and draws each as a small perspective
//! view (the default framing), read back into egui textures. One file at a time, in the order asked.

use eframe::egui;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, mpsc},
    thread,
};
use video::Gpu;

/// Thumbnail size (a row of 768 bytes: copies need multiples of 256); thumbnails per file at most.
pub const W: u32 = 192;
pub const H: u32 = 108;
const MAX: usize = 60;

/// A file's thumbnails: (source time, picture), ascending.
type Shots = Vec<(f64, egui::ColorImage)>;

pub struct Thumbs {
    tx: mpsc::Sender<(String, PathBuf)>,
    rx: mpsc::Receiver<(String, Result<Shots, String>)>,
    /// per file path: (source time, texture), or the failure
    pub map: HashMap<String, Result<Vec<(f64, egui::TextureHandle)>, String>>,
    ctx: egui::Context,
}

impl Thumbs {
    pub fn new(g: &Arc<Gpu>, cx: &stitch::Ctx, ctx: &egui::Context) -> Thumbs {
        let (tx, rq) = mpsc::channel::<(String, PathBuf)>();
        let (rt, rx) = mpsc::channel();
        let (g, cx, c) = (g.clone(), cx.clone(), ctx.clone());
        thread::Builder::new()
            .name("thumbs".into())
            .spawn(move || {
                for (k, p) in rq {
                    let r = crate::export::scoped(&g.device, || shots(&g, &cx, &p));
                    if rt.send((k, r)).is_err() {
                        break;
                    }
                    c.request_repaint();
                }
            })
            .expect("thumbs thread");
        Thumbs {
            tx,
            rx,
            map: HashMap::new(),
            ctx: ctx.clone(),
        }
    }

    /// Asks for `path`'s thumbnails once.
    pub fn ask(&mut self, path: &str) {
        if !self.map.contains_key(path) {
            self.map.insert(path.into(), Ok(vec![]));
            let _ = self.tx.send((path.into(), PathBuf::from(path)));
        }
    }

    /// Takes finished files.
    pub fn tick(&mut self) {
        while let Ok((k, r)) = self.rx.try_recv() {
            let v = r.map(|v| {
                (v.into_iter().enumerate())
                    .map(|(i, (t, img))| {
                        let tex = self.ctx.load_texture(
                            format!("thumb {k} {i}"),
                            img,
                            egui::TextureOptions::LINEAR,
                        );
                        (t, tex)
                    })
                    .collect()
            });
            if let Err(e) = &v {
                eprintln!("thumbnails {k}: {e}");
            }
            self.map.insert(k, v);
        }
    }

    /// The thumbnail of `path` nearest source time `s`.
    pub fn at(&self, path: &str, s: f64) -> Option<&egui::TextureHandle> {
        let v = self.map.get(path)?.as_ref().ok()?;
        let i = v.partition_point(|x| x.0 < s);
        let j = match (i.checked_sub(1), v.get(i)) {
            (Some(a), Some(b)) if s - v[a].0 < b.0 - s => a,
            (Some(a), None) => a,
            _ => i,
        };
        v.get(j).map(|x| &x.1)
    }

    /// Files still being made.
    pub fn busy(&self) -> bool {
        self.map
            .values()
            .any(|v| v.as_ref().is_ok_and(|v| v.is_empty()))
    }
}

/// Decodes `p` through to its last thumbnail frame and renders each one; the GPU work under the app's gate.
fn shots(g: &Gpu, cx: &stitch::Ctx, p: &std::path::Path) -> Result<Shots, String> {
    let c = effects::Clip::open(p)?;
    // the frames every video track has as sync samples (X5: every 2 s)
    let m = media::mp4::Mp4::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let v: Vec<&media::mp4::Track> = (m.tracks.iter())
        .filter(|t| {
            matches!(
                t.codec,
                media::mp4::Codec::Hevc { .. } | media::mp4::Codec::Avc { .. }
            )
        })
        .collect();
    let mut ks: Vec<usize> = (0..c.times.len())
        .filter(|&k| v.iter().all(|t| t.samples.get(k).is_some_and(|s| s.sync)))
        .collect();
    let step = ks.len().div_ceil(MAX).max(1);
    ks = ks.into_iter().step_by(step).collect();
    if ks.is_empty() {
        ks.push(0);
    }
    let pv = stitch::Preview::pano(cx, crate::player::FMT);
    let tgt = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("thumb"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: crate::player::FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let tv = tgt.create_view(&Default::default());
    let v = crate::player::Look::Perspective.view();
    let mut bufs = vec![];
    for &k in &ks {
        // per thumbnail: a surface configure the UI retries waits for one, not the file's batch
        let _g = crate::GATE.read();
        c.frames(
            g,
            cx,
            512,
            false,
            &[k],
            |k| c.pose(k, None),
            |tex, k| {
                let pano = cx.pano(&tex.create_view(&Default::default()));
                let b = g.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("thumb"),
                    size: (4 * W * H) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut e = g.device.create_command_encoder(&Default::default());
                pv.render(&mut e, &tv, [W, H], &v, &pano);
                e.copy_texture_to_buffer(
                    tgt.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &b,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * W),
                            rows_per_image: None,
                        },
                    },
                    tgt.size(),
                );
                g.queue.submit([e.finish()]);
                bufs.push((c.times[k], b));
                Ok(())
            },
        )?;
    }
    bufs.iter()
        .map(|(t, b)| {
            b.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
            g.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| e.to_string())?;
            let img = egui::ColorImage::from_rgba_unmultiplied(
                [W as usize, H as usize],
                &b.get_mapped_range(..).expect("mapped"),
            );
            b.unmap();
            Ok((*t, img))
        })
        .collect()
}
