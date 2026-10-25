//! The editor's viewer: timeline frames from decks (a source file's video on its media thread, `player::Src`,
//! positioned for one timeline clip), drawn by crate stitch's preview with the frame's stabilisation pose (template
//! seam), then the clip's adjustments and look and the overlays through crate fx, into a texture egui shows. Two
//! decks: the clip playing and the next one, which seeks to its first frame before the cut.

use crate::{
    clip::Pic,
    player::{FMT, Src, Tgt},
};
use eframe::egui_wgpu::RenderState;
use media::insv::Meta;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use stitch::{Pose, View};
use video::Gpu;

/// Frames a deck may decode on to reach a later position instead of seeking.
const AHEAD: usize = 15;
pub const DECKS: usize = 2;

/// Stabilisation setting: lock and σ bits.
pub type Key = (Option<stab::Lock>, u64);

/// A source file as the viewer opens it.
pub struct Info {
    /// main files in lens order
    pub files: Vec<PathBuf>,
    pub meta: Meta,
    /// stabilisation poses per frame (or why there are none) for (lock, σ bits), and the framing (the X5 frames on
    /// lens 1)
    #[allow(clippy::type_complexity)]
    pub poses: Option<(Key, Result<Arc<[stab::Pose]>, String>)>,
    pub frame: Pose,
}

impl Info {
    pub fn open(path: &str) -> Result<Info, String> {
        let w = crate::library::pair(std::path::Path::new(path));
        let first = w.main.first().ok_or("no full-resolution file")?;
        let meta = media::insv::Trailer::open(first)
            .map_err(|e| format!("no Insta360 trailer ({e})"))?
            .meta;
        if meta.offset.is_empty() {
            return Err("no lens calibration in the trailer".into());
        }
        Ok(Info {
            frame: Pose::euler(0.0, 0.0, if meta.stream == 4 { 180.0 } else { 0.0 }),
            files: w.main,
            meta,
            poses: None,
        })
    }

    /// The stitch pose of frame `k` (shotlab's `Clip::pose`).
    pub fn pose(&self, k: usize, key: Key) -> Pose {
        match &self.poses {
            Some((pk, Ok(p))) if *pk == key && key.0.is_some() && !p.is_empty() => {
                let p = p[k.min(p.len() - 1)];
                Pose {
                    q: p.q * self.frame.q,
                    rs: [p.rs; 2],
                }
            }
            _ => self.frame,
        }
    }

    /// Whether the poses for `key` are in (or failed), or `key` does not stabilise.
    pub fn posed(&self, key: Key) -> bool {
        key.0.is_none() || self.poses.as_ref().is_some_and(|p| p.0 == key)
    }
}

pub struct Deck {
    g: Arc<Gpu>,
    pub path: String,
    pub s: Src,
    pub held: Option<Pic>,
    /// seek target not shown yet
    pub want: Option<usize>,
    /// timeline clip and edit generation it serves
    pub k: Option<(usize, u64)>,
    tinted: bool,
}

impl Deck {
    pub fn open(
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        path: &str,
        i: &Info,
        ctx: &eframe::egui::Context,
    ) -> Result<Deck, String> {
        Ok(Deck {
            g: g.clone(),
            path: path.into(),
            s: Src::open(g, cx, &i.files, &i.meta, 3840.0, ctx)?,
            held: None,
            want: None,
            k: None,
            tinted: false,
        })
    }

    /// The frame shown or about to be.
    pub fn pos(&self) -> Option<usize> {
        self.want.or(self.held.map(|p| p.k))
    }

    /// Whether frame `k` is the position or a few frames on.
    pub fn near(&self, k: usize) -> bool {
        self.pos().is_some_and(|p| p <= k && k <= p + AHEAD)
    }

    pub fn seek(&mut self, g: &Gpu, k: usize) {
        if self.pos() != Some(k) {
            self.s.v.seek(g, k, self.held.take());
            self.want = Some(k);
        }
    }

    /// Takes the frame due at source time `now` (or the seek target) once it is decoded; true if a new one came.
    /// Decoded only: at a cut the next clip's deck decodes its way to its first frame, and the playing deck's frames
    /// come late; sampling one still decoding made the UI thread wait 50–200 ms.
    pub fn take(&mut self, g: &Gpu, now: f64) -> bool {
        let now = self.want.map_or(now, |k| self.s.v.times[k]);
        let Some(p) = self.s.v.take(now, true).0 else {
            return false;
        };
        self.s.v.release_before(g, &p);
        (self.held, self.want, self.tinted) = (Some(p), None, false);
        true
    }
}

impl Drop for Deck {
    fn drop(&mut self) {
        if let Some(h) = self.held.take() {
            self.s.v.release(&self.g, &h);
        }
    }
}

/// The render targets and effect state.
pub struct Viewer {
    g: Arc<Gpu>,
    pub fc: fx::Ctx,
    pub tgt: Option<Tgt>,
    a: Option<(wgpu::Texture, wgpu::TextureView)>,
    fx: Option<(fx::Fx, fx::Io, [u32; 2], bool)>,
    /// look id in the effects (None: set it again)
    pub look: Option<String>,
    pub cubes: HashMap<String, fx::Cube>,
    /// overlays on the GPU (None: its picture failed) and what they were made from
    pub ov: Vec<Option<fx::Overlay>>,
    made: Vec<super::doc::Over>,
    pub err: Option<String>,
    /// the look's table failed
    pub bad: Option<String>,
}

impl Viewer {
    pub fn new(g: &Arc<Gpu>) -> Viewer {
        Viewer {
            g: g.clone(),
            fc: fx::Ctx::new(&g.device, &g.queue),
            tgt: None,
            a: None,
            fx: None,
            look: None,
            cubes: HashMap::new(),
            ov: vec![],
            made: vec![],
            err: None,
            bad: None,
        }
    }

    /// The table of look id `id`: our looks by name, else a `.cube` file's path.
    pub fn cube(cubes: &mut HashMap<String, fx::Cube>, id: &str) -> Result<fx::Cube, String> {
        if let Some(c) = cubes.get(id) {
            return Ok(c.clone());
        }
        let c = match fx::Look::ALL.into_iter().find(|l| l.name() == id) {
            Some(l) => l.cube(),
            None => std::fs::read_to_string(id)
                .map_err(|e| e.to_string())
                .and_then(|s| fx::Cube::parse(&s))
                .map_err(|e| format!("{id}: {e}"))?,
        };
        cubes.insert(id.into(), c.clone());
        Ok(c)
    }

    /// Brings the overlays on the GPU up to `over`. A picture is made again (a PNG read and decoded) only when it
    /// changed, not as it moves or fades.
    pub fn overs(&mut self, over: &[super::doc::Over]) {
        if self.made == over {
            return;
        }
        self.err = None;
        for (i, o) in over.iter().enumerate() {
            let same = self.made.get(i).is_some_and(|m| m.pic == o.pic);
            if let Some(v) = self.ov.get_mut(i).and_then(Option::as_mut).filter(|_| same) {
                (v.place, v.opacity) = (o.place, o.opacity);
                continue;
            }
            let v = (o.image())
                .map(|p| fx::Overlay::new(&self.fc, &p, o.place, o.opacity))
                .map_err(|e| self.err = Some(e))
                .ok();
            match self.ov.get_mut(i) {
                Some(x) => *x = v,
                None => self.ov.push(v),
            }
        }
        self.ov.truncate(over.len());
        self.made = over.to_vec();
    }

    /// Draws `d`'s frame (or its RAFT stitch `seam`) through `v` (`pano`: the 2:1 equirect a 360 export makes) with
    /// effects `p` and look `look` (id, intensity) at output time `t`; a black frame without a deck.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        rs: &RenderState,
        size: [u32; 2],
        d: Option<(&mut Deck, Pose)>,
        seam: Option<&stitch::Pano>,
        v: &View,
        pano: bool,
        p: &fx::Params,
        look: (&str, f32),
        over: &[super::doc::Over],
        t: f64,
    ) {
        let g = self.g.clone();
        self.overs(over);
        let resized = Tgt::fit(&mut self.tgt, &g, rs, size);
        let tgt = self.tgt.as_ref().unwrap();
        if resized || self.a.is_none() {
            let a = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("editor view"),
                size: tgt.tex.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let av = a.create_view(&Default::default());
            self.a = Some((a, av));
            self.fx = None;
        }
        let (a, av) = self.a.as_ref().unwrap();
        if self.fx.as_ref().is_none_or(|f| f.2 != size || f.3 != pano) {
            let f = fx::Fx::new(&self.fc, size[0], size[1], pano);
            let io = f.io(a, &tgt.tex);
            (self.fx, self.look) = (Some((f, io, size, pano)), None);
        }
        let (f, io, ..) = self.fx.as_mut().unwrap();
        if self.look.as_deref() != Some(look.0) {
            self.bad = None;
            let c = (!look.0.is_empty())
                .then(|| Viewer::cube(&mut self.cubes, look.0))
                .and_then(|r| r.map_err(|e| self.bad = Some(e)).ok());
            f.look(c.as_ref());
            self.look = Some(look.0.into());
        }
        f.set(&fx::Params {
            mix: if look.0.is_empty() { 1.0 } else { look.1 },
            ..*p
        });
        f.time(t);
        let mut e = g.device.create_command_encoder(&Default::default());
        match d {
            Some((d, pose)) if d.held.is_some() => {
                let h = d.held.unwrap();
                d.s.pv.pose(&pose);
                match seam {
                    // the RAFT stitch (body frame) takes the orientation
                    Some(x) => d.s.pv.render(&mut e, av, size, v, x),
                    None => {
                        let src = d.s.src(&h);
                        if !d.tinted {
                            d.s.pv.colour(&mut e, src);
                            d.tinted = true;
                        }
                        d.s.pv.render(&mut e, av, size, v, src);
                    }
                }
                let ov: Vec<&fx::Overlay> = self.ov.iter().flatten().collect();
                f.run(&mut e, io, &ov);
                if seam.is_none() {
                    d.s.v.acquire(&g, &h);
                }
            }
            _ => {
                e.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("editor view"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: av,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                let ov: Vec<&fx::Overlay> = self.ov.iter().flatten().collect();
                f.run(&mut e, io, &ov);
            }
        }
        g.queue.submit([e.finish()]);
    }
}
