//! Player: one open work. The audio clock (or the silent fallback) is the master; the media thread (`clip`) decodes
//! ahead; each UI frame takes the newest frame due and draws it with crate stitch's preview at window size, or,
//! paused, the RAFT stitch of that frame made on a worker thread. The UI never paces playback. Stabilisation
//! (`gyro`) turns each frame's view by its pose: the preview applies orientation and rolling shutter; the RAFT stitch
//! applies the rolling shutter and the preview turns the resulting body-frame equirect.
//!
//! Subject tracking (`ai`): with tracking armed, a box dragged around a subject in the perspective view starts a run
//! from the frame shown, or on a paused frame a click in any view: its people and animals show boxed (found on a
//! thread, `ai::pick`), and the one clicked is tracked (off the boxes, `Ai::pick` searches around the click). The run's
//! states show under the seek bar, the box on the picture, and once it is over the view follows the subject (the follow
//! view the export takes). Auto framing follows the clip's auto framing the same way, its analysis run on first use and
//! kept. Privacy preview: the faces and licence plates of the paused frame blurred as the export blurs them. Auto edit
//! hands the clip to the editor.
//!
//! The library shares the player with the live output thread (`Arc<Mutex<Player>>`), which takes the frame due and
//! draws it on its own clock while the UI may not run (a hidden window); the audio player sits on a thread of its own
//! for that ([`Aud`]).

use crate::{
    Test, ai,
    clip::{self, Pic, Video},
    gyro::{self, Stab},
    library::Work,
};
use eframe::{egui, egui_wgpu::RenderState};
use media::{
    audio,
    insv::{Meta, Trailer},
    mp4::{Codec, Mp4},
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use stitch::{Calib, Ctx, Flow, Frame, Mode, Opts, Pose, Preview, Raft, Stitcher, View};
use video::Gpu;

/// The viewer's texture: egui samples it as sRGB values, which is what the preview writes into a non-sRGB target.
pub const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    Perspective,
    Panini,
    Equirect,
    Planet,
}

impl Look {
    pub const ALL: [(Look, &str); 4] = [
        (Look::Perspective, "Perspective"),
        (Look::Panini, "Panini"),
        (Look::Equirect, "Equirect"),
        (Look::Planet, "Tiny planet"),
    ];

    pub fn parse(s: &str) -> Option<Look> {
        Some(match s {
            "perspective" => Look::Perspective,
            "panini" => Look::Panini,
            "equirect" => Look::Equirect,
            "planet" => Look::Planet,
            _ => return None,
        })
    }

    pub fn view(self) -> View {
        let v = View::default();
        match self {
            Look::Perspective => View { fov: 100.0, ..v },
            Look::Panini => View {
                fov: 170.0,
                panini: 1.0,
                ..v
            },
            Look::Equirect => View {
                mode: Mode::Equirect,
                ..v
            },
            Look::Planet => View {
                mode: Mode::Planet,
                pitch: -90.0,
                fov: 280.0,
                ..v
            },
        }
    }

    /// Field of view range for scrolling.
    pub fn fov(self) -> (f32, f32) {
        match self {
            Look::Perspective => (20.0, 160.0),
            Look::Panini => (60.0, 300.0),
            Look::Equirect => (360.0, 360.0),
            Look::Planet => (90.0, 350.0),
        }
    }

    /// The picture's rect in `rect`: the equirect keeps its 2:1.
    pub fn fit(self, rect: egui::Rect) -> egui::Rect {
        match self {
            Look::Equirect => {
                let w = rect.width().min(rect.height() * 2.0);
                egui::Rect::from_center_size(rect.center(), egui::vec2(w, w / 2.0))
            }
            _ => rect,
        }
    }

    /// A drag turns `v` with the pointer, scrolling zooms within the look's limits, a double click resets it; true
    /// if it changed. `r`: the picture's rect.
    pub fn steer(self, ui: &egui::Ui, resp: &egui::Response, r: egui::Rect, v: &mut View) -> bool {
        let mut ch = false;
        if resp.dragged() {
            let d = resp.drag_delta();
            // degrees per point: the picture follows the pointer
            let k = if self == Look::Equirect {
                360.0
            } else {
                v.fov.min(180.0)
            } / r.width().max(1.0);
            v.yaw = (v.yaw - d.x * k).rem_euclid(360.0);
            v.pitch = (v.pitch + d.y * k).clamp(-90.0, 90.0);
            ch = true;
        }
        if resp.hovered() {
            let y = ui.input(|i| i.smooth_scroll_delta().y);
            if y != 0.0 {
                let (a, b) = self.fov();
                v.fov = (v.fov * (-y * 0.002).exp()).clamp(a, b);
                ch = true;
            }
        }
        if resp.double_clicked() {
            *v = self.view();
            ch = true;
        }
        ch
    }
}

static SRC_ID: AtomicU64 = AtomicU64::new(1);

/// A playable video (the full-resolution files or the proxy): media thread, preview renderer, the lens bind groups
/// of every ring slot.
pub(crate) struct Src {
    pub(crate) id: u64,
    pub(crate) v: Video,
    pub(crate) pv: Preview,
    /// per lens: ring and rect in its pictures
    lens: [(usize, [u32; 4]); 2],
    bg: [Vec<stitch::Src>; 2],
    cal: Calib,
    fr: Frame,
    blend: f64,
    /// RAFT stitch size
    pano: [u32; 2],
    /// camera time of the first frame, µs (maps it onto the posed video)
    ff: u64,
}

impl Src {
    /// `iw`: lens picture width of the full-resolution files (the proxy's window shift depends on it).
    pub(crate) fn open(
        g: &Arc<Gpu>,
        cx: &Ctx,
        files: &[PathBuf],
        m: &Meta,
        iw: f32,
        ctx: &egui::Context,
    ) -> Result<Src, String> {
        let c = ctx.clone();
        let v = Video::open(g, files, move || c.request_repaint())?;
        let cal = Calib::parse(&m.calib)?;
        let [x, y, w, h] = v.view;
        let (lens, fr) = if v.rings.len() == 1 {
            // both lenses side by side, lens 0 left; the proxy's window sits ½(crop/w − crop/iw) calibration px off
            // the full-resolution one (crate stitch, `Frame::shift`)
            let w = w / 2;
            let d = -(m.crop / w as f32 - m.crop / iw) / 2.0;
            let r = [[x, y, w, h], [x + w, y, w, h]];
            let fr = Frame {
                w,
                h,
                crop: m.crop,
                shift: m.shift.map(|s| s + d),
            };
            ([(0, r[0]), (0, r[1])], fr)
        } else {
            // reversed dual track (#131 = 4): track 0 holds lens 1
            let r = (m.stream == 4) as usize;
            let fr = Frame {
                w,
                h,
                crop: m.crop,
                shift: m.shift,
            };
            ([(r, v.view), (1 - r, v.view)], fr)
        };
        let blend = m.blend.unwrap_or(6.0);
        let pv = Preview::new(cx, &cal, fr, blend, Some(0.1), FMT);
        let bg = lens.map(|(r, rect)| {
            (0..v.rings[r].slots() as u64)
                .map(|s| cx.rect(&v.rings[r].planes(s), rect))
                .collect()
        });
        // a lens's 200° across its picture: 360° need about 1.8 × its width
        let pw = ((fr.w as f32 * 1.8) as u32)
            .next_multiple_of(64)
            .clamp(1024, 5760);
        Ok(Src {
            id: SRC_ID.fetch_add(1, Relaxed),
            v,
            pv,
            lens,
            bg,
            cal,
            fr,
            blend,
            pano: [pw, pw / 2],
            // the `.lrv` has its own trailer
            ff: Trailer::open(&files[0])
                .ok()
                .map(|t| t.meta.first_frame)
                .filter(|&f| f > 0)
                .unwrap_or(m.first_frame),
        })
    }

    pub(crate) fn src(&self, p: &Pic) -> [&stitch::Src; 2] {
        [0, 1].map(|i| &self.bg[i][p.f[self.lens[i].0] as usize % self.bg[i].len()])
    }

    /// A RAFT job for `p`: its lens pictures copied into `snap` (made or remade to fit), stitched with the rolling-shutter
    /// term `rs`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn job(
        &self,
        g: &Gpu,
        p: &Pic,
        rs: [glam::Vec3; 2],
        key: Key,
        snap: &mut Option<Snap>,
        reply: &mpsc::Sender<Reply>,
        ctx: &egui::Context,
    ) -> Job {
        let (w, h) = (self.fr.w, self.fr.h);
        // the lens planes as they are: R8 / RG8, or R16 / RG16 for P010 (H.265 Main 10)
        let deep = self.v.rings[0].deep();
        let (fy, fc, b) = match deep {
            true => (
                wgpu::TextureFormat::R16Unorm,
                wgpu::TextureFormat::Rg16Unorm,
                2,
            ),
            false => (
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::Rg8Unorm,
                1,
            ),
        };
        if snap
            .as_ref()
            .is_none_or(|x| (x.w, x.h, x.y[0].format()) != (w, h, fy))
        {
            let tex = |w, h, format| {
                g.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("raft input"),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                })
            };
            let bpr = (w * b).next_multiple_of(256) as u64;
            *snap = Some(Snap {
                y: [0, 1].map(|_| tex(w, h, fy)),
                uv: [0, 1].map(|_| tex(w / 2, h / 2, fc)),
                buf: g.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raft input"),
                    size: bpr * h as u64 * 3,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                w,
                h,
            });
        }
        let x = snap.as_ref().unwrap();
        let bpr = (w * b).next_multiple_of(256);
        let mut e = g.device.create_command_encoder(&Default::default());
        for i in 0..2 {
            let (r, [rx, ry, _, _]) = self.lens[i];
            let ring = &self.v.rings[r];
            let o = (i as u64) * bpr as u64 * h as u64 * 3 / 2;
            for (pl, d, off, dst) in [
                (0, 1, o, &x.y[i]),
                (1, 2, o + bpr as u64 * h as u64, &x.uv[i]),
            ] {
                let layout = wgpu::TexelCopyBufferLayout {
                    offset: off,
                    bytes_per_row: Some(bpr),
                    rows_per_image: None,
                };
                let ext = wgpu::Extent3d {
                    width: w / d,
                    height: h / d,
                    depth_or_array_layers: 1,
                };
                e.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        origin: wgpu::Origin3d {
                            x: rx / d,
                            y: ry / d,
                            z: 0,
                        },
                        ..ring.plane(p.f[r], pl)
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &x.buf,
                        layout,
                    },
                    ext,
                );
                e.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &x.buf,
                        layout,
                    },
                    dst.as_image_copy(),
                    ext,
                );
            }
        }
        // decoder pictures go back to their home state before the decoder reuses them
        let home: Vec<_> = (0..self.v.rings.len())
            .flat_map(|r| self.v.rings[r].textures(p.f[r]))
            .map(|texture| wgpu::TextureTransition {
                texture,
                selector: None,
                state: wgpu::TextureUses::RESOURCE,
            })
            .collect();
        e.transition_resources(std::iter::empty(), home.into_iter());
        self.v.acquire(g, p);
        g.queue.submit([e.finish()]);
        Job {
            key,
            cal: self.cal.clone(),
            fr: self.fr,
            blend: self.blend,
            size: self.pano,
            rs,
            y: x.y.clone(),
            uv: x.uv.clone(),
            reply: reply.clone(),
            ctx: ctx.clone(),
        }
    }
}

/// Source id, frame, rolling-shutter term on.
pub(crate) type Key = (u64, usize, bool);
/// A frame of the main file under a stabilisation (lock, σ bits).
type PickKey = (usize, gyro::Mode, u64);
/// Tracking's pick: the people and animals found, or a click's subject.
type Picks = Result<Vec<track::Det>, String>;
/// The faces and plates (as asked: faces, plates) of a frame of the main file under a stabilisation (lock, σ bits).
type FaceKey = (usize, gyro::Mode, u64, [bool; 2]);
/// The regions, the faces first, and how many of them are faces.
type Faces = Result<(Vec<track::Region>, usize), String>;
pub(crate) type Reply = Result<(Key, wgpu::Texture), String>;

/// A RAFT stitch request: copies of the paused frame's lens pictures and what to stitch them with.
pub struct Job {
    key: Key,
    cal: Calib,
    fr: Frame,
    blend: f64,
    size: [u32; 2],
    /// the frame's rolling-shutter term (the preview turns the stitch, which stays in the body frame)
    rs: [glam::Vec3; 2],
    y: [wgpu::Texture; 2],
    uv: [wgpu::Texture; 2],
    reply: mpsc::Sender<Reply>,
    ctx: egui::Context,
}

/// The RAFT worker, one per app: the model and its autotuned kernels stay loaded. It stitches from copies, so the
/// UI keeps acquiring and releasing decoder frames as usual while a stitch runs.
pub fn still(g: &Arc<Gpu>, cx: &Ctx) -> mpsc::Sender<Job> {
    let (tx, rx) = mpsc::channel::<Job>();
    let (g, cx) = (g.clone(), cx.clone());
    let run = move || {
        // the stitcher per source and size
        let (mut raft, mut st): (Option<Raft>, Option<([u64; 2], Stitcher)>) = (None, None);
        while let Ok(j) = rx.recv() {
            let mut go = || {
                let raft = match &mut raft {
                    Some(r) => r,
                    None => raft.insert(Raft::new(&g.instance, &g.adapter, &g.device, &g.queue)?),
                };
                let k = [j.key.0, j.size[0] as u64];
                if st.as_ref().is_none_or(|s| s.0 != k) {
                    st = None;
                    let o = Opts {
                        blend: j.blend,
                        flow: Some(Flow::RAFT),
                        ..Default::default()
                    };
                    st = Some((
                        k,
                        Stitcher::new(&cx, &j.cal, j.fr, j.size[0], j.size[1], o)?,
                    ));
                }
                let s = &mut st.as_mut().unwrap().1;
                s.reset();
                s.pose(&Pose {
                    rs: j.rs,
                    ..Default::default()
                });
                let v = |t: &wgpu::Texture| t.create_view(&Default::default());
                let src = [0, 1].map(|i| cx.planes(&v(&j.y[i]), &v(&j.uv[i])));
                let src = [&src[0], &src[1]];
                let mut e = g.device.create_command_encoder(&Default::default());
                s.belt(&mut e, src, raft);
                g.queue.submit([e.finish()]);
                s.flow(raft)?;
                let mut e = g.device.create_command_encoder(&Default::default());
                s.encode(&mut e, src);
                g.queue.submit([e.finish()]);
                Ok((j.key, s.out.clone()))
            };
            // a surface configure the UI retries waits for the job (burn submits throughout); the outputs do not
            let r = {
                let _g = crate::BURN.lock();
                go()
            };
            let _ = j.reply.send(r);
            j.ctx.request_repaint();
        }
    };
    thread::Builder::new()
        .name("raft".into())
        .spawn(run)
        .expect("RAFT thread");
    tx
}

/// RAFT on played frames (the player's RAFT seam option): a flow stitcher of the source playing, drawn through
/// `Preview::seam` once it has a flow, and a RAFT worker of its own. A new played frame's draw records the belt (RAFT's
/// input) when the worker is idle and `n` frames passed since the last run; the flow is held until the next run's
/// lands (it scored better than carried by the gyro's turn, interpolated or extrapolated: `TODO.md`). A frame late or
/// dropped doubles `n`, 64 on time take one off; past a second's frames RAFT stops (the template seam) until the source
/// or the option changes.
struct Seam {
    /// the source's id
    id: u64,
    st: Arc<Mutex<Stitcher>>,
    pv: Preview,
    /// the worker's model once loaded, its jobs, its replies (the model, then a run's end)
    raft: Option<Arc<Raft>>,
    go: mpsc::Sender<()>,
    rx: mpsc::Receiver<Result<Option<Arc<Raft>>, String>>,
    /// a run out; a flow of this position drawn; frames since the last run, least between runs, on time since a
    /// late one; a seek since the flow (reset before the next run); before the first run's end (it compiles kernels:
    /// lateness ignored); given up
    busy: bool,
    has: bool,
    since: usize,
    n: usize,
    calm: usize,
    stale: bool,
    cold: bool,
    off: bool,
}

impl Seam {
    fn new(g: &Arc<Gpu>, cx: &Ctx, s: &Src) -> Result<Seam, String> {
        let o = Opts {
            blend: s.blend,
            flow: Some(Flow::RAFT),
            ..Default::default()
        };
        // its output stays unused: the preview draws from the lenses through its flow
        let st = Arc::new(Mutex::new(Stitcher::new(cx, &s.cal, s.fr, 8, 2, o)?));
        let pv = Preview::seam(&st.lock().unwrap(), FMT);
        let ((go, jobs), (tx, rx)) = (mpsc::channel::<()>(), mpsc::channel());
        let (g, w) = (g.clone(), st.clone());
        thread::Builder::new()
            .name("raft playing".into())
            .spawn(move || {
                let raft = match Raft::new(&g.instance, &g.adapter, &g.device, &g.queue) {
                    Ok(r) => Arc::new(r),
                    Err(e) => return drop(tx.send(Err(e))),
                };
                let _ = tx.send(Ok(Some(raft.clone())));
                while jobs.recv().is_ok() {
                    let r = {
                        let _b = crate::BURN.lock();
                        w.lock().unwrap().flow(&raft)
                    };
                    if tx.send(r.map(|_| None)).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(Seam {
            id: s.id,
            st,
            pv,
            raft: None,
            go,
            rx,
            busy: false,
            has: false,
            since: 0,
            n: 1,
            calm: 0,
            stale: false,
            cold: true,
            off: false,
        })
    }

    /// The worker's replies; Err when RAFT failed.
    fn poll(&mut self) -> Result<(), String> {
        while let Ok(r) = self.rx.try_recv() {
            match r? {
                Some(r) => self.raft = Some(r),
                None => (self.busy, self.has, self.cold) = (false, !self.stale, false),
            }
        }
        Ok(())
    }

    /// A new played frame (posed): its colour fit and, when a run is due, its belt; true then, and the caller sends
    /// `go` after submitting `e`.
    fn frame(&mut self, e: &mut wgpu::CommandEncoder, src: [&stitch::Src; 2]) -> bool {
        self.pv.colour(e, src);
        self.since += 1;
        let Some(r) = self.raft.as_ref().filter(|_| !self.busy && !self.off) else {
            return false;
        };
        let mut st = self.st.lock().unwrap();
        if self.stale {
            st.reset();
            self.stale = false;
        }
        if self.since < self.n {
            return false;
        }
        st.belt(e, src, r);
        (self.busy, self.since) = (true, 0);
        true
    }

    /// A played frame shown, `bad` if late or after dropped ones; true when RAFT gives up with it.
    fn pace(&mut self, bad: bool, fps: f64) -> bool {
        if self.off || !bad || self.cold {
            self.calm += 1;
            if self.calm >= 64 && self.n > 1 {
                (self.n, self.calm) = (self.n - 1, 0);
            }
            return false;
        }
        (self.n, self.calm) = (self.n * 2, 0);
        self.off = self.n as f64 > fps;
        self.off
    }
}

/// Lens picture copies for a RAFT job: luma and chroma per lens, through one staging buffer.
pub(crate) struct Snap {
    y: [wgpu::Texture; 2],
    uv: [wgpu::Texture; 2],
    buf: wgpu::Buffer,
    w: u32,
    h: u32,
}

/// A viewer's texture, registered with egui.
pub struct Tgt {
    pub tex: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub id: egui::TextureId,
    rs: RenderState,
}

impl Tgt {
    /// Makes `t` `size` (a new texture under the same egui id when it was another size); true if it did.
    pub fn fit(t: &mut Option<Tgt>, g: &Gpu, rs: &RenderState, size: [u32; 2]) -> bool {
        if t.as_ref()
            .is_some_and(|t| [t.tex.width(), t.tex.height()] == size)
        {
            return false;
        }
        let tex = g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FMT,
            // storage: the editor's effects write it; copy: tests read the picture back
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let mut r = rs.renderer.write();
        match t {
            Some(t) => {
                r.update_egui_texture_from_wgpu_texture(
                    &g.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    t.id,
                );
                (t.tex, t.view) = (tex, view);
            }
            None => {
                let id = r.register_native_texture(&g.device, &view, wgpu::FilterMode::Linear);
                *t = Some(Tgt {
                    tex,
                    view,
                    id,
                    rs: rs.clone(),
                });
            }
        }
        true
    }
}

impl Drop for Tgt {
    fn drop(&mut self) {
        self.rs.renderer.write().free_texture(&self.id);
    }
}

const NS: usize = 4096;

/// Pacing statistics over the last `NS` UI frames while playing.
struct Stats {
    last: Option<Instant>,
    /// UI frame gaps and A/V offsets (shown frame's time − clock), ms
    gap: Vec<f32>,
    av: Vec<f32>,
    ng: usize,
    na: usize,
    /// frames that became due, were shown, skipped, shown at least one UI frame after they were due
    due: usize,
    shown: usize,
    dropped: usize,
    late: usize,
    loops: usize,
    /// due frame at the previous UI frame
    d: Option<usize>,
    /// the last 64 due frames, 1 = late or dropped (automatic proxy), and how many are valid
    hist: u64,
    nh: u32,
}

impl Stats {
    fn new() -> Stats {
        Stats {
            last: None,
            gap: vec![0.0; NS],
            av: vec![0.0; NS],
            ng: 0,
            na: 0,
            due: 0,
            shown: 0,
            dropped: 0,
            late: 0,
            loops: 0,
            d: None,
            hist: 0,
            nh: 0,
        }
    }

    fn reset(&mut self) {
        (
            self.ng,
            self.na,
            self.due,
            self.shown,
            self.dropped,
            self.late,
            self.loops,
        ) = (0, 0, 0, 0, 0, 0, 0);
        (self.d, self.hist, self.nh) = (None, 0, 0);
    }

    fn bad(&mut self, b: bool) {
        self.hist = self.hist << 1 | b as u64;
        self.nh += 1;
    }

    fn pct(v: &[f32], n: usize) -> [f32; 4] {
        let mut v = v[..n.min(NS)].to_vec();
        if v.is_empty() {
            return [f32::NAN; 4];
        }
        v.sort_by(f32::total_cmp);
        let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        [at(0.01), at(0.5), at(0.99), v[v.len() - 1]]
    }

    fn report(&self) -> String {
        let g = Stats::pct(&self.gap, self.ng);
        let a = Stats::pct(&self.av, self.na);
        format!(
            "UI frames {}: gap ms p50 {:.2} p99 {:.2} max {:.2}; video frames due {}, shown {}, dropped {}, late {} \
             (loops {}); A/V (shown pts − clock) ms p1 {:.1} p50 {:.1} p99 {:.1} max {:.1}",
            self.ng,
            g[1],
            g[2],
            g[3],
            self.due,
            self.shown,
            self.dropped,
            self.late,
            self.loops,
            a[0],
            a[1],
            a[2],
            a[3]
        )
    }
}

/// Audio player commands.
pub(crate) enum Cmd {
    Play,
    Pause,
    Seek(f64),
    Vol(f32),
}

/// The audio player on a thread of its own: cpal's stream stays on the thread that made it, while the player (its
/// seeks and loops) runs on the UI thread and on the live output thread. A command returns once the audio player
/// has it, so after a seek `audio::Clock::get` reads None until the device takes it, on whichever thread asks.
pub(crate) struct Aud {
    tx: mpsc::Sender<Cmd>,
    ack: mpsc::Receiver<()>,
}

impl Aud {
    pub(crate) fn cmd(&self, c: Cmd) {
        if self.tx.send(c).is_ok() {
            let _ = self.ack.recv();
        }
    }
}

/// Whether `open_on` failed in the output stream (cpal's error inside: building or starting it), not in the file or
/// its decoder: media maps symphonia's errors to `ErrorKind::Other` too, and such a file fails on the default again.
fn lost(e: &std::io::Error) -> bool {
    e.get_ref().is_some_and(|x| x.is::<cpal::Error>())
}

/// The audio player of `p` on the output picked in Settings, or the silent clock when it has no AAC track or there
/// is no output device; its clock; the audio track's presentation offset (edit list) in seconds; whether it has sound.
pub(crate) fn audio(p: &Path, end: f64) -> (Aud, audio::Clock, f64, bool) {
    let off = Mp4::open(p)
        .ok()
        .and_then(|m| {
            m.tracks
                .iter()
                .find(|t| matches!(t.codec, Codec::Aac { .. }))
                .map(|t| t.shift(m.timescale) as f64 / t.timescale as f64)
        })
        .unwrap_or(0.0);
    let (tx, rx) = mpsc::channel();
    let (atx, ack) = mpsc::sync_channel(0);
    let (itx, irx) = mpsc::sync_channel(1);
    let p = p.to_path_buf();
    thread::Builder::new()
        .name("audio control".into())
        .spawn(move || {
            // here, not on the UI thread (which waits for the open): `output` waits for a lookup under way
            let r = match audio::Player::open_on(&p, crate::settings::output()) {
                // the stream failed: the device picked may be gone since, the system default then
                Err(e) if lost(&e) => {
                    eprintln!("{}: {e}: the default output", p.display());
                    audio::Player::open(&p)
                }
                r => r,
            };
            let (a, sound) = match r {
                Ok(a) => (a, true),
                Err(e) => {
                    eprintln!("{}: {e}: silent clock", p.display());
                    (audio::Player::silent(end), false)
                }
            };
            let _ = itx.send((a.clock(), sound));
            for c in rx {
                match c {
                    Cmd::Play => a.play(),
                    Cmd::Pause => a.pause(),
                    Cmd::Seek(t) => a.seek(t),
                    Cmd::Vol(v) => a.volume(v),
                }
                if atx.send(()).is_err() {
                    break;
                }
            }
        })
        .expect("audio thread");
    let (clock, sound) = irx.recv().expect("audio thread");
    (Aud { tx, ack }, clock, if sound { off } else { 0.0 }, sound)
}

pub struct Player {
    g: Arc<Gpu>,
    cx: Ctx,
    ctx: egui::Context,
    pub work: Work,
    meta: Meta,
    iw: f32,
    stab: Stab,
    audio: Aud,
    clock: audio::Clock,
    a_off: f64,
    sound: bool,
    /// full resolution, proxy
    src: [Option<Src>; 2],
    cur: usize,
    /// the frame shown (sampled by the viewer), when it came, whether it is new since the last render, whether
    /// its colour fit is recorded (by the viewer or a live output, whichever draws it first)
    held: Option<Pic>,
    held_at: Instant,
    fresh: bool,
    tinted: bool,
    /// seek target not shown yet; resume the audio once it is
    want: Option<usize>,
    resume: bool,
    playing: bool,
    looping: bool,
    vol: f32,
    look: Look,
    view: View,
    tgt: Option<Tgt>,
    size: [u32; 2],
    dirty: bool,
    // RAFT stitch of the paused frame
    still: mpsc::Sender<Job>,
    reply: (mpsc::Sender<Reply>, mpsc::Receiver<Reply>),
    busy: bool,
    raft_off: bool,
    snap: Option<Snap>,
    pano: Option<(Key, stitch::Pano)>,
    /// the RAFT seam option: RAFT on played frames too; its state for the source playing
    pub flow: bool,
    seam: Option<Seam>,
    stats: Stats,
    show_stats: bool,
    /// test hook: measure for this long once playing, then report; repaint period instead of vblank; no
    /// automatic proxy
    measure: Option<(f64, Option<Instant>, bool)>,
    tick: Option<Duration>,
    pin: bool,
    done: bool,
    note: Option<String>,
    /// the AAC track for live outputs, opened on first use
    tap: Option<Option<crate::live::Tap>>,
    /// tracking: the run on this clip; a box drawn in the view starts one (armed); the view follows the subject;
    /// the box being drawn
    pub trk: Option<Arc<ai::Track>>,
    arm: bool,
    pub follow: bool,
    from: Option<egui::Pos2>,
    /// tracking's pick (armed, paused): the frame's people and animals, boxed; a search out (with a click's
    /// direction: for its subject, which starts a run)
    dets: Option<(PickKey, Vec<track::Det>)>,
    dq: Option<(PickKey, Option<glam::Vec3>, mpsc::Receiver<Picks>)>,
    /// auto framing: the clip's run (kept while the player is open); the view follows it (tracking's or this one)
    pub frm: Option<Arc<ai::Track>>,
    pub framing: bool,
    /// privacy preview: the paused frame's faces, plates blurred; the regions found (and how many are faces), a search
    /// out, the unblurred picture and the pass
    pub blur: bool,
    pub plates: bool,
    regs: Option<(FaceKey, Vec<track::Region>, usize)>,
    fq: Option<(FaceKey, mpsc::Receiver<Faces>)>,
    bt: Option<(wgpu::Texture, wgpu::TextureView, crate::export::Blur)>,
    /// the editor is to make an auto edit of this clip
    pub auto: bool,
    /// test hook: the box to draw once the frame shows (0..1 of the view at 16:9); the stabilisation to switch to once
    /// its run is over
    tbox: Option<[f32; 4]>,
    restab: Option<(gyro::Mode, f64)>,
}

impl Player {
    pub fn open(
        g: &Arc<Gpu>,
        cx: &Ctx,
        work: Work,
        ctx: &egui::Context,
        still: &mpsc::Sender<Job>,
        gyro: &mpsc::Sender<gyro::Req>,
        test: Option<Test>,
    ) -> Result<Player, String> {
        let first = work
            .main
            .first()
            .or(work.lrv.as_ref())
            .ok_or("nothing to play")?
            .clone();
        let meta = Trailer::open(&first)
            .map_err(|e| format!("no Insta360 trailer ({e})"))?
            .meta;
        if meta.calib.is_empty() {
            return Err("no lens calibration in the trailer".into());
        }
        let cur = work.main.is_empty() as usize;
        let files = if cur == 0 {
            work.main.clone()
        } else {
            vec![first.clone()]
        };
        // the proxy alone: the X5's full-resolution lens size
        let s = Src::open(g, cx, &files, &meta, 3840.0, ctx)?;
        let iw = if cur == 0 { s.fr.w as f32 } else { 3840.0 };
        let t = test.as_ref();
        let stab = Stab::new(
            gyro,
            first.clone(),
            s.v.times.clone(),
            s.ff,
            // the X5 frames its videos on lens 1, which reversed tracks put first
            if meta.stream == 4 { 180.0 } else { 0.0 },
            t.and_then(|t| t.stab).unwrap_or(Some(stab::Lock::Horizon)),
            t.and_then(|t| t.smooth).unwrap_or(gyro::SMOOTH),
            Some((g.clone(), cx.clone())),
            ctx,
        );
        let (audio, clock, a_off, sound) = audio(&first, s.v.end);
        let mut src = [None, None];
        src[cur] = Some(s);
        let mut p = Player {
            g: g.clone(),
            cx: cx.clone(),
            ctx: ctx.clone(),
            work,
            meta,
            iw,
            stab,
            audio,
            clock,
            a_off,
            sound,
            src,
            cur,
            held: None,
            held_at: Instant::now(),
            fresh: false,
            tinted: false,
            want: None,
            resume: false,
            playing: false,
            looping: false,
            vol: 1.0,
            look: Look::Perspective,
            view: Look::Perspective.view(),
            tgt: None,
            size: [0; 2],
            dirty: true,
            still: still.clone(),
            reply: mpsc::channel(),
            busy: false,
            raft_off: false,
            snap: None,
            pano: None,
            flow: true,
            seam: None,
            stats: Stats::new(),
            show_stats: false,
            measure: None,
            tick: None,
            pin: false,
            done: false,
            note: None,
            tap: None,
            trk: None,
            arm: false,
            follow: false,
            from: None,
            dets: None,
            dq: None,
            frm: None,
            framing: false,
            blur: false,
            plates: false,
            regs: None,
            fq: None,
            bt: None,
            auto: false,
            tbox: None,
            restab: None,
        };
        if let Some(t) = test {
            (p.tbox, p.restab) = (t.tbox, t.restab);
            (p.blur, p.plates) = (t.blur, t.plates);
            p.auto = t.autoedit.is_some();
            p.vol = t.vol.unwrap_or(p.vol).clamp(0.0, 1.0);
            p.audio.cmd(Cmd::Vol(p.vol));
            if let Some(l) = t.look {
                (p.look, p.view) = (l, l.view());
            }
            if let Some([y, x, f]) = t.aim {
                (p.view.yaw, p.view.pitch) = (y, x);
                p.view.fov = if f.is_nan() { p.view.fov } else { f };
            }
            p.looping = t.looping;
            p.tick = t.novsync.then(|| Duration::from_secs_f64(1.0 / 60.0));
            p.pin = t.full;
            if t.proxy {
                p.use_src(1);
            }
            if let Some(k) = t.seek {
                p.seek(k);
            }
            if let Some(s) = t.stats {
                p.measure = Some((s, None, false));
                p.show_stats = true;
            }
            if t.play || t.stats.is_some() {
                p.play();
            }
        }
        Ok(p)
    }

    fn s(&self) -> &Src {
        self.src[self.cur].as_ref().unwrap()
    }

    /// Shows frame `k` next: from the previous sync sample, exactly `k`; the clock waits there until it is shown.
    pub fn seek(&mut self, k: usize) {
        let held = self.held.take();
        let Some(s) = self.src[self.cur].as_mut() else {
            return;
        };
        let k = k.min(s.v.times.len() - 1);
        s.v.seek(&self.g, k, held);
        let t = s.v.times[k];
        if self.playing {
            self.audio.cmd(Cmd::Pause);
            self.resume = true;
        }
        self.audio.cmd(Cmd::Seek(t - self.a_off));
        self.want = Some(k);
        if let Some(f) = &mut self.seam {
            (f.has, f.stale) = (false, true);
        }
        self.stats.d = None;
        self.stats.hist = 0;
        self.stats.nh = 0;
    }

    pub fn play(&mut self) {
        let s = self.s();
        let at = self.held.map_or(0.0, |p| p.t);
        if at + 1.0 / 60.0 >= s.v.times[s.v.times.len() - 1] && self.want.is_none() {
            self.seek(0);
        }
        self.playing = true;
        if self.want.is_some() {
            self.resume = true;
        } else {
            self.audio.cmd(Cmd::Play);
        }
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.resume = false;
        self.audio.cmd(Cmd::Pause);
    }

    fn step(&mut self, d: isize) {
        if self.playing {
            self.pause();
        }
        let k = self.want.or(self.held.map(|p| p.k)).unwrap_or(0);
        self.seek(k.saturating_add_signed(d));
    }

    /// Plays source `i` (0 full resolution, 1 proxy) from the frame shown now.
    fn use_src(&mut self, i: usize) {
        let files = match (i, &self.work.lrv) {
            _ if i == self.cur => return,
            (0, _) if !self.work.main.is_empty() => self.work.main.clone(),
            (1, Some(l)) => vec![l.clone()],
            _ => return,
        };
        let s = match Src::open(&self.g, &self.cx, &files, &self.meta, self.iw, &self.ctx) {
            Ok(s) => s,
            Err(e) => {
                self.note = Some(format!("proxy: {e}"));
                return;
            }
        };
        let t = self.held.map_or(0.0, |p| p.t);
        if let Some(h) = self.held.take() {
            self.s().v.release(&self.g, &h);
        }
        // the old source's decoders go first
        self.src[self.cur] = None;
        let k = clip::frame_at(&s.v.times, t);
        self.src[i] = Some(s);
        self.cur = i;
        self.seek(k);
    }

    /// The main file's frame at `p` (a frame of the current source).
    fn main_k(&self, p: &Pic) -> usize {
        let s = self.s();
        self.stab.frame(p.t, s.v.times[0], s.ff)
    }

    /// The stabilisation shown: lock and σ.
    pub fn stab(&self) -> (gyro::Mode, f64) {
        (self.stab.mode, self.stab.smooth)
    }

    fn pkey(&self) -> Option<PickKey> {
        let p = self.held?;
        Some((self.main_k(&p), self.stab.mode, self.stab.smooth.to_bits()))
    }

    fn fkey(&self) -> Option<FaceKey> {
        let (k, m, s) = self.pkey()?;
        Some((k, m, s, [self.blur, self.plates]))
    }

    /// Tracking's search on frame `k`: its people and animals, or with `at` the subject at that direction
    /// (`ai::pick`), on a thread of its own; it replaces one out.
    fn find(&mut self, k: PickKey, at: Option<glam::Vec3>) {
        let Some(f) = self.work.main.first() else {
            return;
        };
        let (tx, rx) = mpsc::channel();
        let (g, cx, ctx, f) = (self.g.clone(), self.cx.clone(), self.ctx.clone(), f.clone());
        thread::spawn(move || {
            let st = (k.1, f64::from_bits(k.2));
            let _ = tx.send(crate::export::scoped(&g.device, || {
                ai::pick(&g, &cx, &f, st, k.0, at)
            }));
            ctx.request_repaint();
        });
        self.dq = Some((k, at, rx));
    }

    /// A click on direction `d` of frame `k`, tracking armed: the person or animal boxed there is tracked
    /// (`track::at`); off the boxes `Ai::pick` searches around it.
    fn click(&mut self, k: PickKey, d: glam::Vec3) {
        let x = (self.dets.as_ref()).filter(|x| x.0 == k);
        match x.and_then(|x| track::at(x.1.clone(), d)) {
            Some(s) => {
                if let Err(e) = self.track(s.into()) {
                    self.note = Some(e);
                }
            }
            None => self.find(k, Some(d)),
        }
    }

    /// Tracking of subject `s` (`ai::subject` of a box in the perspective view, or a pick's), from the frame shown to
    /// the clip's end; the view follows it once the run is over.
    pub fn track(&mut self, s: track::Subject) -> Result<(), String> {
        let f = (self.work.main.first().cloned())
            .ok_or("Tracking needs the main file, not the proxy alone")?;
        let p = self.held.ok_or("no frame shown yet")?;
        if let Some(t) = &self.trk {
            t.stop.store(true, Relaxed);
        }
        let k0 = self.main_k(&p);
        let st = (self.stab.mode, self.stab.smooth);
        self.trk = Some(ai::Track::start(
            &self.g,
            &self.cx,
            &self.ctx,
            f,
            st,
            k0,
            usize::MAX,
            s,
        ));
        (self.follow, self.framing, self.arm) = (true, false, false);
        Ok(())
    }

    /// The run the view follows: tracking's or auto framing's.
    fn followed(&self) -> Option<&Arc<ai::Track>> {
        (self.trk.as_ref().filter(|_| self.follow)).or(self.frm.as_ref().filter(|_| self.framing))
    }

    /// The stitch pose of `p` (a frame of the current source) and the key of its RAFT stitch.
    fn pose(&self, p: &Pic) -> (Pose, Key) {
        let s = self.s();
        let o = self.stab.pose(p.t, s.v.times[0], s.ff);
        (o, (s.id, p.k, o.rs[0] != glam::Vec3::ZERO))
    }

    /// Every UI frame, on any screen, and before each live output frame: the clock, the frame due, loop and end, RAFT
    /// and stabilisation replies, statistics. `ui`: the UI calls, which also measures its own pacing (frame gaps,
    /// A/V offset, the `stats=` hook); else the live output thread, which runs while the UI may not (a hidden window).
    pub fn tick(&mut self, ui: bool) {
        let now_i = Instant::now();
        if let Some(e) = self.s().v.err() {
            self.note = Some(e);
        }
        if self.stab.tick() {
            self.dirty = true;
        }
        if let Some(n) = self.stab.note.take() {
            self.note = Some(n);
        }
        while let Ok(r) = self.reply.1.try_recv() {
            self.busy = false;
            match r {
                Ok((k, t)) => {
                    self.pano = Some((k, self.cx.pano(&t.create_view(&Default::default()))));
                    self.dirty = true;
                }
                Err(e) => {
                    self.note = Some(format!("RAFT stitch: {e}"));
                    self.raft_off = true;
                }
            }
        }
        // RAFT on played frames: a seam for the source playing, its worker's replies
        if self.flow && self.playing && self.seam.as_ref().is_none_or(|f| f.id != self.s().id) {
            self.seam = Seam::new(&self.g, &self.cx, self.s())
                .map_err(|e| (self.note, self.flow) = (Some(format!("RAFT seam: {e}")), false))
                .ok();
        }
        if let Some(f) = &mut self.seam
            && let Err(e) = f.poll()
        {
            (self.note, self.flow, self.seam) = (Some(format!("RAFT seam: {e}")), false, None);
        }
        let gap = ui
            .then(|| self.stats.last.replace(now_i))
            .flatten()
            .map(|l| (now_i - l).as_secs_f32() * 1e3);
        if self.playing {
            match self.tick {
                Some(d) => self.ctx.request_repaint_after(d),
                None => self.ctx.request_repaint(),
            }
        }
        // a seek the clock has not taken yet: it still reads the old position
        let Some(now) = self.clock.get().map(|c| c + self.a_off) else {
            self.ctx.request_repaint_after(Duration::from_millis(2));
            return;
        };
        let s = self.s();
        if self.playing && self.want.is_none() && now >= s.v.end {
            if self.looping {
                self.stats.loops += 1;
                self.seek(0);
            } else {
                self.pause();
            }
            return;
        }
        let (p, skip) = s.v.take(now, false);
        let d = clip::frame_at(&s.v.times, now);
        let (id, fps) = (s.id, s.v.times.len() as f64 / s.v.end.max(1e-3));
        let measuring = self.playing && self.want.is_none();
        if let Some(p) = p {
            s.v.release_before(&self.g, &p);
            let st = &mut self.stats;
            if self.want.take().is_some() {
                if self.resume {
                    self.resume = false;
                    self.audio.cmd(Cmd::Play);
                }
                // the seek target (a loop's restart): due and shown now
                if self.playing {
                    (st.due, st.shown, st.d) = (st.due + 1, st.shown + 1, Some(p.k));
                }
            } else if measuring {
                let late = st.d.is_some_and(|x| x >= p.k);
                (st.shown, st.dropped, st.late) =
                    (st.shown + 1, st.dropped + skip, st.late + late as usize);
                for _ in 0..skip {
                    st.bad(true);
                }
                st.bad(late);
                if let Some(f) = self.seam.as_mut().filter(|f| f.id == id)
                    && f.pace(late || skip > 0, fps)
                {
                    self.note = Some("RAFT seam: the GPU can't keep up while playing".into());
                }
            }
            (self.held, self.held_at, self.fresh, self.tinted) = (Some(p), now_i, true, false);
        }
        if measuring {
            let st = &mut self.stats;
            st.due += st.d.map_or(1, |x| d.saturating_sub(x));
            st.d = Some(d);
            if let Some(g) = gap {
                st.gap[st.ng % NS] = g;
                st.ng += 1;
            }
            if ui && let Some(p) = self.held {
                st.av[st.na % NS] = ((p.t - now) * 1e3) as f32;
                st.na += 1;
            }
            if ui && let Some((secs, t0, on)) = &mut self.measure {
                // from the first frame shown on time 3 s after the start of playback: the window's first presents
                // can take a second each while the compositor sends no frame callbacks
                let t = t0.get_or_insert(now_i + Duration::from_secs(3));
                if !*on && (now_i < *t || self.held.is_none_or(|p| p.k < d)) {
                    self.stats.reset();
                } else if !*on {
                    // this frame may still carry the catch-up's skips
                    self.stats.reset();
                    self.stats.d = Some(d);
                    (*t, *on) = (now_i, true);
                } else if t.elapsed().as_secs_f64() >= *secs && !self.done {
                    let t0 = *t;
                    let name = self.work.main.first().or(self.work.lrv.as_ref());
                    println!(
                        "x5desk stats: {} ({}, {}) {:.1} s from the first frame on time after 3 s: {}",
                        name.map_or("?".into(), |p| p.display().to_string()),
                        ["full resolution", "proxy"][self.cur],
                        if self.sound {
                            "audio clock"
                        } else {
                            "silent clock"
                        },
                        t0.elapsed().as_secs_f64(),
                        self.stats.report()
                    );
                    self.done = true;
                }
            }
            // full resolution falls behind: a quarter of the last 64 due frames late or dropped
            if self.cur == 0
                && !self.pin
                && self.work.lrv.is_some()
                && self.stats.nh >= 64
                && self.stats.hist.count_ones() > 16
            {
                self.note = Some(format!(
                    "Full resolution fell behind ({} of the last 64 frames late or dropped): playing the proxy",
                    self.stats.hist.count_ones()
                ));
                self.use_src(1);
            }
        }
        if let Some(m) = self.restab
            && self.trk.as_ref().is_some_and(|t| t.st().over())
        {
            (self.restab, (self.stab.mode, self.stab.smooth)) = (None, m);
            self.stab.ask();
            self.dirty = true;
        }
        // the view follows the tracked subject or the auto framing, turned from the run's frame into the frame shown
        // (the stabilisation may have changed since)
        if let (Some(t), Some(p)) = (self.followed(), self.held)
            && let k = self.main_k(&p)
            && let Some((d, r)) = t.dir(k).zip(t.turn(k, self.pose(&p).0.q))
        {
            let [yaw, pitch] = track::yaw_pitch(r * d);
            if (yaw - self.view.yaw).abs() + (pitch - self.view.pitch).abs() > 1e-3 {
                (self.view.yaw, self.view.pitch, self.dirty) = (yaw, pitch, true);
            }
        }
        if let Some((k, r)) = &self.fq
            && let Ok(x) = r.try_recv()
        {
            match x {
                Ok((v, n)) => self.regs = Some((*k, v, n)),
                Err(e) => {
                    (self.note, self.blur, self.plates) =
                        (Some(format!("Blur preview: {e}")), false, false)
                }
            }
            (self.fq, self.dirty) = (None, true);
        }
        if (self.blur || self.plates)
            && !self.playing
            && self.want.is_none()
            && self.fq.is_none()
            && self.stab.ready()
            && let (Some(k), Some(f)) = (self.fkey(), self.work.main.first())
            && self.regs.as_ref().is_none_or(|r| r.0 != k)
        {
            let (tx, rx) = mpsc::channel();
            let (g, cx, ctx, f, st) = (
                self.g.clone(),
                self.cx.clone(),
                self.ctx.clone(),
                f.clone(),
                (k.1, f64::from_bits(k.2)),
            );
            thread::spawn(move || {
                let _ = tx.send(crate::export::scoped(&g.device, || {
                    ai::privacy(&g, &cx, &f, st, k.0, k.3)
                }));
                ctx.request_repaint();
            });
            self.fq = Some((k, rx));
        }
        // tracking armed on a paused frame: its people and animals boxed, a click's subject tracked (if it is still
        // the frame shown)
        if let Some((k, at, r)) = &self.dq
            && let Ok(x) = r.try_recv()
        {
            let (k, at) = (*k, *at);
            (self.dq, self.dirty) = (None, true);
            match (x, at) {
                (Ok(v), None) => self.dets = Some((k, v)),
                (Ok(v), Some(_)) if self.arm && Some(k) == self.pkey() => {
                    let r = match v.first() {
                        Some(d) => self.track((*d).into()),
                        None => Err("Tracking: no person or animal there".into()),
                    };
                    if let Err(e) = r {
                        self.note = Some(e);
                    }
                }
                (Ok(_), _) => {}
                (Err(e), _) => (self.note, self.arm) = (Some(format!("Tracking: {e}")), false),
            }
        }
        if self.arm
            && !self.playing
            && self.want.is_none()
            && self.dq.is_none()
            && self.stab.ready()
            && let Some(k) = self.pkey()
            && self.dets.as_ref().is_none_or(|x| x.0 != k)
        {
            self.find(k, None);
        }
        // the hook's box is 0..1 of the view at 16:9 (a flat 1080p export's frame of it), whatever the window: the
        // picture changes shape while the window's layout settles and with its size, and the box with it
        if self.tbox.is_some() && self.held.is_some() && self.want.is_none() && self.stab.ready() {
            let s = ai::subject(&self.view, 16.0 / 9.0, self.tbox.take().unwrap());
            let [y, x] = track::yaw_pitch(s.dir);
            eprintln!(
                "x5desk: box: subject at yaw {y:.2}°, pitch {x:.2}°, {:.2}° × {:.2}°",
                s.size.x, s.size.y
            );
            let r = match self.look {
                Look::Perspective => self.track(s),
                _ => Err("Tracking: draw the box in the perspective view".into()),
            };
            if let Err(e) = r {
                eprintln!("x5desk: box: {e}");
                self.note = Some(e);
            }
        }
        // paused on a frame for a moment: RAFT stitch of it
        if let Some(p) = self.held
            && !self.playing
            && self.want.is_none()
            && !self.busy
            && !self.raft_off
            && self.stab.ready()
            && self.pano.as_ref().is_none_or(|x| x.0 != self.pose(&p).1)
            && now_i - self.held_at >= Duration::from_millis(150)
        {
            self.snap(p);
        } else if !self.playing && !self.busy && self.held.is_some() {
            self.ctx.request_repaint_after(Duration::from_millis(150));
        }
    }

    /// Copies the lens pictures of `p` and hands them to the RAFT worker.
    fn snap(&mut self, p: Pic) {
        let (o, key) = self.pose(&p);
        let s = self.src[self.cur].as_ref().unwrap();
        let j = s.job(
            &self.g,
            &p,
            o.rs,
            key,
            &mut self.snap,
            &self.reply.0,
            &self.ctx,
        );
        if self.still.send(j).is_ok() {
            self.busy = true;
        } else {
            self.raft_off = true;
        }
    }

    /// Nothing pending: a frame is shown and, paused, its RAFT stitch too (or RAFT is off); a requested
    /// measurement is done.
    pub fn idle(&self) -> bool {
        let raft = self.playing
            || self.raft_off
            || self
                .held
                .is_some_and(|p| self.pano.as_ref().is_some_and(|x| x.0 == self.pose(&p).1));
        let faces = !(self.blur || self.plates)
            || self.playing
            || (self.fkey()).is_some_and(|k| self.regs.as_ref().is_some_and(|r| r.0 == k));
        let pick = self.dq.is_none()
            && (!self.arm
                || self.playing
                || (self.pkey()).is_some_and(|k| self.dets.as_ref().is_some_and(|x| x.0 == k)));
        self.want.is_none()
            && self.held.is_some()
            && self.stab.ready()
            && raft
            && self.measure.is_none_or(|_| self.done)
            && self.tbox.is_none()
            && self.restab.is_none()
            && self.trk.as_ref().is_none_or(|t| t.st().over())
            && self.frm.as_ref().is_none_or(|t| t.st().over())
            && faces
            && pick
    }

    /// What the export screen takes: the main files, the frame shown, the view, the stabilisation; None while only the
    /// proxy is there.
    pub fn spec(&self) -> Option<crate::export::Spec> {
        let s = self.s();
        let k = self.want.or(self.held.map(|p| p.k)).unwrap_or(0);
        (!self.work.main.is_empty()).then(|| crate::export::Spec {
            files: self.work.main.clone(),
            photo: false,
            dur: s.v.end,
            t: s.v.times[k],
            view: self.view,
            look: self.look,
            stab: self.stab.mode,
            smooth: self.stab.smooth,
            follow: (self.followed().cloned()).filter(|t| t.st() == ai::St::Done),
            faces: self.blur,
            plates: self.plates,
        })
    }

    fn render(&mut self, rs: &RenderState, size: [u32; 2]) {
        let g = &self.g;
        Tgt::fit(&mut self.tgt, g, rs, size);
        self.size = size;
        let Some(p) = self.held else { return };
        // the privacy preview: the picture into a texture of its own, blurred from there into the viewer's
        let regs = (self.regs.as_ref())
            .filter(|r| (self.blur || self.plates) && !self.playing && Some(r.0) == self.fkey())
            .map(|r| &r.1);
        let mut bt = self.bt.take();
        if regs.is_some()
            && bt
                .as_ref()
                .is_none_or(|b| b.0.size() != self.tgt.as_ref().unwrap().tex.size())
        {
            let t = g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer before blur"),
                size: self.tgt.as_ref().unwrap().tex.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            bt = Some((t, v, crate::export::Blur::new(&g.device, FMT)));
        }
        let ((o, key), t) = (self.pose(&p), self.tgt.as_ref().unwrap());
        let s = self.src[self.cur].as_ref().unwrap();
        let dst = match (regs, &bt) {
            (Some(_), Some(b)) => &b.1,
            _ => &t.view,
        };
        // the lenses take orientation and rolling shutter, the RAFT stitch (body frame) the orientation
        s.pv.pose(&o);
        let mut seam = self.seam.as_mut().filter(|f| self.playing && f.id == s.id);
        let mut e = g.device.create_command_encoder(&Default::default());
        let (mut tint, mut run) = (self.tinted, false);
        match self.pano.as_ref().filter(|x| !self.playing && x.0 == key) {
            Some((_, pano)) => s.pv.render(&mut e, dst, size, &self.view, pano),
            None => {
                let src = s.src(&p);
                if let Some(f) = &seam {
                    f.pv.pose(&o);
                }
                if !tint {
                    s.pv.colour(&mut e, src);
                    run = seam.as_mut().is_some_and(|f| f.frame(&mut e, src));
                    tint = true;
                }
                let pv = seam.as_ref().filter(|f| f.has && !f.off);
                pv.map_or(&s.pv, |f| &f.pv)
                    .render(&mut e, dst, size, &self.view, src);
                s.v.acquire(g, &p);
            }
        }
        if let (Some(r), Some(b)) = (regs, &mut bt) {
            b.2.run(&g.device, &g.queue, &mut e, &b.1, &t.view, &self.view, r);
        }
        g.queue.submit([e.finish()]);
        if let (true, Some(f)) = (run, seam) {
            let _ = f.go.send(());
        }
        self.bt = bt;
        (self.fresh, self.dirty, self.tinted) = (false, false, tint);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        let (space, left, right) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
            )
        });
        if space {
            if self.playing {
                self.pause()
            } else {
                self.play()
            }
        }
        if left || right {
            self.step(if left { -1 } else { 1 });
        }
        egui::Panel::bottom("controls").show(ui, |ui| self.controls(ui));
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewer(ui, rs));
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let s = self.s();
        let (n, end) = (s.v.times.len(), s.v.end);
        let k = self.want.or(self.held.map(|p| p.k)).unwrap_or(0);
        let mut t = s.v.times[k];
        ui.add_space(4.0);
        ui.spacing_mut().slider_width = ui.available_width();
        let sr = ui.add(egui::Slider::new(&mut t, 0.0..=end).show_value(false));
        if sr.changed() {
            let j = clip::frame_at(&self.s().v.times, t);
            if j != k {
                self.seek(j);
            }
        }
        // the tracking run's states under the bar (the rail sits a handle's radius in from the ends)
        if let Some(tr) = &self.trk
            && let Some(ts) = tr.times.get()
        {
            let (r, _) =
                ui.allocate_exact_size(egui::vec2(sr.rect.width(), 5.0), egui::Sense::hover());
            let (x0, w) = (r.left() + r.height(), r.width() - 2.0 * r.height());
            ai::strip(ui.painter(), tr, r.y_range(), |k| {
                Some(x0 + w * (ts.get(k).copied().unwrap_or(end) / end.max(1e-6)) as f32)
            });
        }
        ui.spacing_mut().slider_width = 90.0;
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
                .clicked()
            {
                if self.playing {
                    self.pause()
                } else {
                    self.play()
                }
            }
            if ui.button("<").on_hover_text("Previous frame (←)").clicked() {
                self.step(-1);
            }
            if ui.button(">").on_hover_text("Next frame (→)").clicked() {
                self.step(1);
            }
            let mmss = |t: f64| format!("{}:{:05.2}", (t / 60.0) as u32, t % 60.0);
            ui.monospace(format!(
                "{k:>5} / {n}  {} / {}",
                mmss(self.s().v.times[k]),
                mmss(end)
            ));
            ui.separator();
            ui.checkbox(&mut self.looping, "Loop");
            // the slider sits in a horizontal of its own, which never wraps (the stabilisation pair below too)
            if ui.available_size_before_wrap().x < 160.0 {
                ui.end_row();
            }
            ui.label(if self.sound { "Volume" } else { "No audio" });
            if ui
                .add_enabled(
                    self.sound,
                    egui::Slider::new(&mut self.vol, 0.0..=1.0).show_value(false),
                )
                .changed()
            {
                self.audio.cmd(Cmd::Vol(self.vol));
            }
            ui.separator();
            let mut proxy = self.cur == 1;
            let can = self.work.lrv.is_some() && !self.work.main.is_empty();
            if ui
                .add_enabled(can, egui::Checkbox::new(&mut proxy, "Proxy"))
                .on_hover_text("Play the .lrv proxy")
                .changed()
            {
                self.use_src(proxy as usize);
            }
            if ui
                .checkbox(&mut self.flow, "RAFT seam")
                .on_hover_text(
                    "Optical flow at the seam while playing too, every few frames as the GPU allows (paused frames \
                     always have it); off: the template seam while playing",
                )
                .changed()
            {
                (self.seam, self.dirty) = (None, true);
            }
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
            ui.separator();
            let main = !self.work.main.is_empty();
            match self.trk.as_ref().map(|t| t.st()) {
                Some(st) if !st.over() => {
                    if ui.button("Cancel track").clicked() {
                        self.trk.as_ref().unwrap().stop.store(true, Relaxed);
                    }
                }
                over => {
                    ui.add_enabled_ui(main, |ui| {
                        ui.toggle_value(&mut self.arm, "Track a subject")
                            .on_hover_text(
                                "Paused, click a person or animal (those found are boxed), or drag a box around a \
                                 subject in the perspective view: it is tracked from this frame on",
                            )
                            .on_disabled_hover_text("Tracking needs the main file");
                    });
                    if over.is_some() {
                        if ui
                            .checkbox(&mut self.follow, "Follow")
                            .on_hover_text("The view follows the tracked subject (and so does a flat export)")
                            .changed()
                        {
                            (self.framing, self.dirty) = (self.framing && !self.follow, true);
                        }
                        if ui.button("Clear track").clicked() {
                            (self.trk, self.follow) = (None, false);
                        }
                    }
                }
            }
            if ui
                .add_enabled(main, egui::Checkbox::new(&mut self.framing, "Auto framing"))
                .on_hover_text(
                    "The view follows the clip's main subject, else the direction of travel (and so does a flat \
                     export); the whole clip is analysed first, once",
                )
                .on_disabled_hover_text("Auto framing needs the main file")
                .changed()
            {
                // the analysis once, again after a failure; it runs on while unticked
                if self.framing
                    && (self.frm.as_ref())
                        .is_none_or(|t| matches!(t.st(), ai::St::Failed(_) | ai::St::Cancelled))
                    && let Some(f) = self.work.main.first()
                {
                    // framed for a flat 1080p export, as the hook's box
                    let (f, st) = (f.clone(), self.stab());
                    self.frm = Some(ai::Track::framing(&self.g, &self.cx, &self.ctx, f, st, 16.0 / 9.0));
                }
                (self.follow, self.dirty) = (self.follow && !self.framing, true);
            }
            if ui
                .toggle_value(&mut self.blur, "Blur faces")
                .on_hover_text("Preview the export's face blur on the paused frame (the export starts with it on)")
                .changed()
            {
                self.dirty = true;
            }
            if ui
                .toggle_value(&mut self.plates, "Blur license plates")
                .on_hover_text(
                    "Preview the export's license plate blur on the paused frame (the export starts with it on)",
                )
                .changed()
            {
                self.dirty = true;
            }
            if ui
                .add_enabled(main, egui::Button::new("Auto edit"))
                .on_hover_text(
                    "Highlights of this clip, framed automatically, as an editor timeline (cut to the beats of the \
                     editor's auto edit music, if one is chosen)",
                )
                .clicked()
            {
                self.auto = true;
            }
            ui.separator();
            let st = &mut self.stab;
            let was = (st.mode, st.smooth);
            match &st.err {
                Some(e) => {
                    ui.label(format!("No stabilisation: {e}"));
                }
                None => {
                    // a slider sits in a horizontal of its own, which never wraps: keep the pair together
                    if ui.available_size_before_wrap().x < 300.0 {
                        ui.end_row();
                    }
                    egui::ComboBox::from_id_salt("stab")
                        .selected_text(
                            gyro::MODES
                                .iter()
                                .find(|m| m.0 == st.mode)
                                .map_or("", |m| m.1),
                        )
                        .show_ui(ui, |ui| {
                            for (m, n) in gyro::MODES {
                                ui.selectable_value(&mut st.mode, m, n);
                            }
                        })
                        .response
                        .on_hover_text("Stabilisation from the camera's gyro");
                    ui.add_enabled(
                        st.mode.is_some(),
                        egui::Slider::new(&mut st.smooth, 0.0..=gyro::SMOOTH_MAX)
                            .step_by(0.05)
                            .suffix(" s")
                            .text("Smoothing"),
                    )
                    .on_hover_text(
                        "σ of the smoothed camera path (0: follow the camera; a lock still levels)",
                    );
                }
            }
            if (st.mode, st.smooth) != was {
                st.ask();
                self.dirty = true;
            }
            ui.separator();
            ui.toggle_value(&mut self.show_stats, "Stats");
        });
        if let Some(n) = &self.note {
            ui.label(n);
        }
        ui.add_space(2.0);
    }

    fn viewer(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        let (rect, resp) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let r = self.look.fit(rect);
        let mut live = None;
        if self.arm && self.look == Look::Perspective {
            let (l, b) = ai::drag_box(ui, &resp, r, &mut self.from);
            live = l;
            if let Some(b) = b
                && let Err(e) = self.track(ai::subject(&self.view, r.aspect_ratio(), b))
            {
                self.note = Some(e);
            }
        } else if self.look.steer(ui, &resp, r, &mut self.view) {
            // a hand on the view takes over from the follow view
            self.follow &= !resp.dragged();
            self.framing &= !resp.dragged();
            self.dirty = true;
        }
        if self.arm
            && !self.playing
            && self.want.is_none()
            && resp.clicked()
            && let (Some(q), Some(k)) = (
                resp.interact_pointer_pos().filter(|q| r.contains(*q)),
                self.pkey(),
            )
        {
            let (m, mode, f, d) = crate::export::proj(&self.view);
            let (u, s) = (q - r.min, glam::Vec2::new(r.width(), r.height()));
            self.click(
                k,
                m * crate::export::ray(glam::Vec2::new(u.x, u.y), mode, f, d, s),
            );
        }
        let ppp = ui.ctx().pixels_per_point();
        let size = [r.width(), r.height()].map(|x| ((x * ppp).round() as u32).max(1));
        if self.fresh || self.dirty || size != self.size {
            self.render(rs, size);
        }
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, egui::Color32::BLACK);
        if let Some(t) = &self.tgt {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            p.image(t.id, r, uv, egui::Color32::WHITE);
        }
        if let Some(b) = live {
            ai::paint_box(&p, b);
        }
        // clipped to the picture: a box across the equirect's edge shows on both sides
        let pr = p.with_clip_rect(r);
        if self.arm
            && !self.playing
            && let Some((k, v)) = &self.dets
            && Some(*k) == self.pkey()
        {
            for d in v {
                for l in ai::outline(&self.view, r, d.dir, d.size) {
                    pr.line(l, egui::Stroke::new(2.0, egui::Color32::YELLOW));
                }
            }
        }
        if let (Some(t), Some(h)) = (&self.trk, self.held)
            && let k = self.main_k(&h)
            && let Some((s, q)) = t.at(k).zip(t.turn(k, self.pose(&h).0.q))
        {
            for l in ai::outline(&self.view, r, q * s.raw, s.size) {
                pr.line(l, egui::Stroke::new(2.0, ai::colour(s.state)));
            }
        }
        let mut lines = vec![];
        if self.held.is_none() || self.want.is_some() {
            lines.push("Decoding…".to_string());
        }
        if self.arm {
            let drag = match self.look {
                Look::Perspective => ", or drag a box around one",
                _ => "",
            };
            let n = (self.dets.as_ref())
                .filter(|x| Some(x.0) == self.pkey())
                .map(|x| x.1.len());
            lines.push(match (self.playing, &self.dq, n) {
                (true, ..) => format!("Tracking: pause to pick a subject{drag}"),
                (_, Some((_, Some(_), _)), _) => "Tracking: picking…".into(),
                (_, _, Some(n)) => {
                    format!("Tracking: click a subject ({n} people and animals found){drag}")
                }
                _ => format!("Tracking: finding people and animals…{drag}"),
            });
        }
        if let Some(t) = &self.trk {
            lines.push(match t.st() {
                ai::St::Done if self.follow => "Tracking: following the subject".into(),
                st => st.line("Tracking"),
            });
        }
        match self.frm.as_ref().map(|t| t.st()) {
            Some(ai::St::Done) if self.framing => lines.push("Auto framing: following".into()),
            Some(ai::St::Done) | None => {}
            Some(st) => lines.push(st.line("Auto framing")),
        }
        if self.blur || self.plates {
            let n = (self.regs.as_ref())
                .filter(|r| Some(r.0) == self.fkey())
                .map(|r| (r.2, r.1.len() - r.2));
            let (w, f) = match (self.blur, self.plates) {
                (true, false) => ("Face", "faces"),
                (false, true) => ("Plate", "plates"),
                _ => ("Face and plate", "faces and plates"),
            };
            lines.push(match (self.playing, n) {
                (true, _) => format!("{w} blur preview: pause on a frame"),
                (_, Some((n, p))) => {
                    let c = |n: usize, s| format!("{n} {s}{}", if n == 1 { "" } else { "s" });
                    match (self.blur, self.plates) {
                        (true, false) => format!("{w} blur preview: {}", c(n, "face")),
                        (false, true) => format!("{w} blur preview: {}", c(p, "plate")),
                        _ => format!("{w} blur preview: {}, {}", c(n, "face"), c(p, "plate")),
                    }
                }
                _ => format!("{w} blur preview: finding {f}…"),
            });
        }
        if let Some(h) = self.held
            && !self.playing
            && !self.raft_off
        {
            let k = self.pose(&h).1;
            lines.push(match &self.pano {
                Some(x) if x.0 == k => "RAFT stitch".into(),
                _ if self.busy => "RAFT stitching…".into(),
                _ => "Template seam".into(),
            });
        }
        if self.playing && self.flow {
            lines.push(
                match &self.seam {
                    Some(f) if f.off => "Template seam: RAFT can't keep up",
                    Some(f) if f.has => "RAFT seam",
                    _ => "RAFT seam starting…",
                }
                .into(),
            );
        }
        if let Some(s) = &self.stab.stage {
            lines.push(format!("Stabilising: {s}…"));
        }
        if self.show_stats {
            lines.push(self.stats.report());
        }
        let mut y = rect.top() + 6.0;
        for l in lines {
            let g = p.layout(
                l,
                egui::FontId::monospace(12.0),
                egui::Color32::WHITE,
                rect.width() - 12.0,
            );
            let at = egui::pos2(rect.left() + 6.0, y);
            p.rect_filled(
                egui::Rect::from_min_size(at, g.size()).expand(3.0),
                3.0,
                egui::Color32::from_black_alpha(160),
            );
            y += g.size().y + 8.0;
            p.galley(at, g, egui::Color32::WHITE);
        }
    }
}

/// The player as a live source: the frame shown now, drawn like the viewer draws it (the RAFT stitch when paused);
/// the AAC packets as the audio clock passes them.
impl crate::live::Source for Player {
    /// The file playing: the `.lrv` once it plays the proxy.
    fn name(&self) -> &str {
        let f = match self.cur {
            0 => self.work.main.first(),
            _ => self.work.lrv.as_ref(),
        };
        f.and_then(|p| p.file_name()?.to_str()).unwrap_or("")
    }

    fn tick(&mut self, _: u64) {
        Player::tick(self, false)
    }

    fn ready(&self) -> bool {
        self.held.is_some()
    }

    fn draw(&mut self, t: &wgpu::TextureView, [w, h]: [u32; 2], pano: bool) {
        let Some(p) = self.held else { return };
        let (g, (o, key)) = (&self.g, self.pose(&p));
        let s = self.src[self.cur].as_ref().unwrap();
        s.pv.pose(&o);
        let mut seam = self.seam.as_mut().filter(|f| self.playing && f.id == s.id);
        if let Some(f) = &seam {
            f.pv.pose(&o);
        }
        let (w, h) = (w as f32, h as f32);
        let (v, r) = match (pano, self.look) {
            (true, _) => (
                View {
                    mode: Mode::Equirect,
                    ..View::default()
                },
                [0.0, 0.0, w, h],
            ),
            // the equirect look keeps its 2:1 in a flat picture
            (false, Look::Equirect) => {
                let (a, b) = (w.min(2.0 * h), h.min(w / 2.0));
                (self.view, [(w - a) / 2.0, (h - b) / 2.0, a, b])
            }
            _ => (self.view, [0.0, 0.0, w, h]),
        };
        let raft = self.pano.as_ref().filter(|x| !self.playing && x.0 == key);
        let src = s.src(&p);
        let tint = raft.is_none() && !self.tinted;
        let mut e = g.device.create_command_encoder(&Default::default());
        let mut run = false;
        if tint {
            s.pv.colour(&mut e, src);
            run = seam.as_mut().is_some_and(|f| f.frame(&mut e, src));
        }
        let pv = seam.as_ref().filter(|f| f.has && !f.off);
        let pv = pv.map_or(&s.pv, |f| &f.pv);
        pv.set(&v, r);
        {
            let mut rp = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("live"),
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
            rp.set_viewport(r[0], r[1], r[2], r[3], 0.0, 1.0);
            match raft {
                Some((_, x)) => pv.draw(&mut rp, x),
                None => pv.draw(&mut rp, src),
            }
        }
        if raft.is_none() {
            s.v.acquire(g, &p);
        }
        g.queue.submit([e.finish()]);
        if let (true, Some(f)) = (run, seam) {
            let _ = f.go.send(());
        }
        self.tinted |= tint;
    }

    fn asc(&mut self) -> Option<&[u8]> {
        if self.tap.is_none() {
            let f = self.work.main.first().or(self.work.lrv.as_ref());
            self.tap = Some(f.and_then(|f| crate::live::Tap::open(f)));
        }
        self.tap.as_ref()?.as_ref().map(|t| &t.asc[..])
    }

    fn audio(&mut self, f: &mut dyn FnMut(u64, &[u8])) {
        if let (Some(c), Some(Some(t))) = (self.clock.get(), &mut self.tap) {
            t.take(c, crate::live::mono(), f);
        }
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        if let (Some(h), Some(s)) = (self.held.take(), &self.src[self.cur]) {
            s.v.release(&self.g, &h);
        }
        // a run on this clip would hold the models to its end
        for t in self.trk.iter().chain(&self.frm) {
            t.stop.store(true, Relaxed);
        }
    }
}

#[cfg(test)]
mod verify;

#[cfg(test)]
mod tests {
    use super::*;

    /// The controls in narrow panels (the hardware run's COSMIC-tiled window, 753 px with the side panels, cut the
    /// volume slider off at its right edge: a slider sits in a horizontal of its own, which a wrapping row does not
    /// wrap): the slider, which follows its label, ends inside the panel.
    #[test]
    fn narrow_controls() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES not set: skipping");
            return;
        };
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        let w = crate::library::pair(&d.join("v635.insv"));
        let mut p = Player::open(&g, &cx, w, &ctx, &still(&g, &cx), &gyro::worker(), None).unwrap();
        for width in (420..=760).step_by(20).map(|w| w as f32) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width, 900.0),
                )),
                ..Default::default()
            };
            // twice: egui sizes some widgets from the frame before
            let mut o = None;
            for _ in 0..2 {
                let mut x = ctx.run_ui(input.clone(), |ui| {
                    egui::CentralPanel::default().show(ui, |ui| p.controls(ui));
                });
                // no renderer takes them (epaint's debug check panics on dropped deltas)
                x.textures_delta.clear();
                o = Some(x);
            }
            // a label in a wrapping row is laid out from the row's start, its galley from there
            let r = o
                .unwrap()
                .shapes
                .iter()
                .find_map(|c| match &c.shape {
                    egui::Shape::Text(t) if t.galley.text() == "Volume" => {
                        Some(t.galley.rect.translate(t.pos.to_vec2()))
                    }
                    _ => None,
                })
                .unwrap();
            let end = r.right() + 4.0 + 90.0;
            assert!(
                end <= width - 8.0,
                "{width} px: the slider ends at {end}, past the panel"
            );
        }
    }

    /// The audio goes to the default output only when the stream on the picked device failed (media hands cpal's
    /// error over inside an `Other`): a 5.1 AAC-LC file, which symphonia's decoder refuses (`Other` as well), is no
    /// stream failure and goes to the silent clock without a second open, nor is a file without an AAC track.
    #[test]
    fn fallback_kind() {
        let f = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-six-{}.mp4", std::process::id()));
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=1"])
            .args(["-ac", "6", "-c:a", "aac"])
            .arg(&f)
            .status()
            .expect("ffmpeg");
        assert!(o.success());
        let r = audio::Player::open_on(&f, None).map(|_| ());
        let (_, _, _, sound) = audio(&f, 1.0);
        std::fs::remove_file(&f).unwrap();
        let e = r.unwrap_err();
        println!("5.1 AAC: {e} ({:?}), sound {sound}", e.kind());
        assert!(e.kind() == std::io::ErrorKind::Other && !lost(&e) && !sound);
        let dev = cpal::Error::new(cpal::ErrorKind::DeviceNotAvailable);
        assert!(lost(&std::io::Error::other(dev)));
        assert!(!lost(&std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no AAC track"
        )));
    }

    /// A file without an AAC track plays on the silent clock: held at open, wall time while playing, exactly on a
    /// seek target.
    #[test]
    fn fallback_clock() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES not set: skipping");
            return;
        };
        let f = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-noaudio-{}.mp4", std::process::id()));
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(d.join("v635.lrv"))
            .args(["-map", "0:v", "-c", "copy"])
            .arg(&f)
            .status()
            .expect("ffmpeg");
        assert!(o.success());
        let (a, c, off, sound) = audio(&f, 5.77);
        std::fs::remove_file(&f).unwrap();
        assert!(!sound);
        assert_eq!(off, 0.0);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(c.get(), Some(0.0), "held at open");
        let t0 = Instant::now();
        a.cmd(Cmd::Play);
        thread::sleep(Duration::from_millis(200));
        let (x, w) = (c.get().unwrap(), t0.elapsed().as_secs_f64());
        println!("silent clock {x:.4} s after {w:.4} s wall");
        assert!((x - w).abs() < 0.02);
        a.cmd(Cmd::Seek(2.0));
        let t1 = Instant::now();
        let x = loop {
            if let Some(x) = c.get() {
                break x;
            }
            assert!(t1.elapsed() < Duration::from_secs(1));
            thread::yield_now();
        };
        assert!((2.0..2.01).contains(&x), "after seek {x}");
    }

    /// A readback target of the viewer's format.
    fn target(g: &Gpu, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
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
        let v = t.create_view(&Default::default());
        (t, v)
    }

    /// Tracking's pick on v606's frame 0 (`X5_SAMPLES`; `X5_PICK` = `<file>:<frame>` entries, `;` between them,
    /// instead): armed on the paused frame the player finds its people and animals (on v606 a person or more); a click
    /// on the centre of the biggest person's box starts a run from that frame whose first box is that box. What
    /// `Ai::pick` finds there (a click off the boxes searches so) is printed, with its IoU against the box. The paused
    /// view with the boxes as the viewer draws them (yellow; the click's run's first box green, the search's blue)
    /// goes to `X5_FIXTURES` as `pick-<file>-<frame>-<view>.jpg`: perspective and Panini on that person (else the
    /// first found), the equirect, the equirect with them on its edge, the tiny planet, a perspective view on the box
    /// nearest a pole.
    #[test]
    fn pick() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES not set: skipping");
            return;
        };
        let out = std::env::var_os("X5_FIXTURES").map_or_else(std::env::temp_dir, PathBuf::from);
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        let (st, gw) = (still(&g, &cx), gyro::worker());
        let v606 = format!("{}:0", d.join("v606.insv").display());
        let list = std::env::var("X5_PICK").unwrap_or(v606.clone());
        // every entry's pictures first
        let mut bad = vec![];
        for e in list.split(';') {
            let (f, k) = e.rsplit_once(':').unwrap();
            let w = crate::library::pair(Path::new(f));
            let mut p = Player::open(&g, &cx, w, &ctx, &st, &gw, None).unwrap();
            p.seek(k.parse().unwrap());
            p.arm = true;
            let t0 = Instant::now();
            let run = |p: &mut Player, done: &dyn Fn(&Player) -> bool| {
                while !done(p) {
                    p.tick(false);
                    assert!(t0.elapsed().as_secs() < 600, "{e}: stuck");
                    thread::sleep(Duration::from_millis(5));
                }
            };
            run(&mut p, &|p| p.idle());
            let (k0, v) = p.dets.clone().unwrap();
            let name = |c: u32| track::CLASSES[c as usize];
            eprintln!(
                "{e}: frame {} found in {:.1} s: {:?}",
                k0.0,
                t0.elapsed().as_secs_f64(),
                (v.iter())
                    .map(|x| (name(x.cls), track::yaw_pitch(x.dir), x.size, x.score))
                    .collect::<Vec<_>>()
            );
            let b = (v.iter().filter(|x| x.cls == 0)).max_by(|a, b| {
                a.size
                    .element_product()
                    .total_cmp(&b.size.element_product())
            });
            assert!(b.is_some() || e != v606, "{e}: no person");
            let s = b.map(|b| {
                // the run's start frame and first box, and its IoU against the box
                let go = |p: &mut Player, click: bool| {
                    (p.trk, p.arm) = (None, true);
                    match click {
                        true => p.click(k0, b.dir),
                        false => p.find(k0, Some(b.dir)),
                    }
                    run(p, &|p| (p.trk.as_ref()).is_some_and(|t| t.at(k0.0).is_some()));
                    let t = p.trk.clone().unwrap();
                    t.stop.store(true, Relaxed);
                    crate::ai::tests::wait(|| t.st(), 60);
                    let s = t.at(k0.0).unwrap();
                    (t.k0, s, track::iou((b.dir, b.size), (s.raw, s.size), false))
                };
                let ((k1, s1, i1), (k2, s2, i2)) = (go(&mut p, true), go(&mut p, false));
                eprintln!(
                    "{e}: clicked at {:?} {:?}: the run's first box IoU {i1:.3}; Ai::pick's there {:?} {:?}, \
                     IoU {i2:.3}",
                    track::yaw_pitch(b.dir),
                    b.size,
                    track::yaw_pitch(s2.raw),
                    s2.size
                );
                if k1 != k0.0 || k2 != k0.0 || i1 < 0.999 {
                    bad.push(format!("{e}: from frames {k1}, {k2}, IoU {i1}"));
                }
                [s1, s2]
            });
            let [yaw, pitch] = (b.or(v.first())).map_or([0.0; 2], |x| track::yaw_pitch(x.dir));
            let [py, pp] = (v.iter().map(|x| track::yaw_pitch(x.dir)))
                .max_by(|a, b| a[1].abs().total_cmp(&b[1].abs()))
                .unwrap_or([0.0; 2]);
            let stem = Path::new(f).file_stem().unwrap().to_string_lossy();
            for (name, l, yaw, pitch) in [
                ("perspective", Look::Perspective, yaw, pitch),
                ("panini", Look::Panini, yaw, pitch),
                ("equirect", Look::Equirect, 0.0, 0.0),
                ("wrap", Look::Equirect, yaw + 180.0, 0.0),
                ("planet", Look::Planet, yaw, -90.0),
                // the box nearest a pole, the view on it
                ("pole", Look::Perspective, py, pp),
            ] {
                (p.look, p.view) = (
                    l,
                    View {
                        yaw,
                        pitch,
                        ..l.view()
                    },
                );
                let (w, h) = if l == Look::Equirect {
                    (1600, 800)
                } else {
                    (1280, 720)
                };
                let (tex, tv) = target(&g, w, h);
                crate::live::Source::draw(&mut p, &tv, [w, h], false);
                let mut px = stitch::read_texture(&g.device, &g.queue, &tex);
                let r = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32));
                let runs = (s.iter().flatten()).zip([[0, 230, 60], [40, 120, 255]]);
                let boxes = (v.iter().map(|x| (x.dir, x.size, [255, 230, 0])))
                    .chain(runs.map(|(s, c)| (s.raw, s.size, c)));
                for (dir, size, c) in boxes {
                    for l in crate::ai::outline(&p.view, r, dir, size) {
                        for ab in l.windows(2) {
                            // the side clipped to 1 px inside the picture (Liang–Barsky), then 3 px thick
                            let (a, d) = (ab[0], ab[1] - ab[0]);
                            let (mut u0, mut u1) = (0f32, 1f32);
                            let (x1, y1) = (w as f32 - 2.0, h as f32 - 2.0);
                            for (p, q) in [
                                (-d.x, a.x - 1.0),
                                (d.x, x1 - a.x),
                                (-d.y, a.y - 1.0),
                                (d.y, y1 - a.y),
                            ] {
                                if p == 0.0 {
                                    u1 = if q < 0.0 { -1.0 } else { u1 };
                                } else if p < 0.0 {
                                    u0 = u0.max(q / p);
                                } else {
                                    u1 = u1.min(q / p);
                                }
                            }
                            let n = (d.length() * (u1 - u0)).ceil().max(1.0) as usize;
                            for i in (0..=n).filter(|_| u0 <= u1) {
                                // a far end's rounding can land a hair outside
                                let q = (a + d * (u0 + (u1 - u0) * i as f32 / n as f32))
                                    .clamp(egui::pos2(1.0, 1.0), egui::pos2(x1, y1));
                                for j in 0..9 {
                                    let (x, y) = (q.x as u32 + j % 3 - 1, q.y as u32 + j / 3 - 1);
                                    px[(y * w + x) as usize * 4..][..3].copy_from_slice(&c);
                                }
                            }
                        }
                    }
                }
                let o = out.join(format!("pick-{stem}-{}-{name}.jpg", k0.0));
                std::fs::write(&o, media::jpeg::encode(&px, w, h, 90, None).unwrap()).unwrap();
            }
        }
        assert!(bad.is_empty(), "{bad:?}");
    }

    /// RAFT on played frames holds its flow between runs: on a static scene (v635's frame 0 shown again and again,
    /// a run every 3rd time) the flow drawn at every frame is that of a fresh RAFT run on the frame.
    #[test]
    fn seam_static() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES not set: skipping");
            return;
        };
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        let f = d.join("v635.insv");
        let m = Trailer::open(&f).unwrap().meta;
        let mut s = Src::open(&g, &cx, std::slice::from_ref(&f), &m, 3840.0, &ctx).unwrap();
        s.v.seek(&g, 0, None);
        let t0 = Instant::now();
        let p = loop {
            if let (Some(p), _) = s.v.take(s.v.times[0], false) {
                break p;
            }
            assert!(t0.elapsed().as_secs() < 10, "no frame");
            thread::sleep(Duration::from_millis(1));
        };
        let o = Pose::euler(0.0, 0.0, if m.stream == 4 { 180.0 } else { 0.0 });
        let src = s.src(&p);
        // fresh: a stitcher of its own, one colour fit and one run
        let raft = Raft::new(&g.instance, &g.adapter, &g.device, &g.queue).unwrap();
        let opts = Opts {
            blend: s.blend,
            flow: Some(Flow::RAFT),
            ..Default::default()
        };
        let mut st = Stitcher::new(&cx, &s.cal, s.fr, 8, 2, opts).unwrap();
        st.pose(&o);
        let mut e = g.device.create_command_encoder(&Default::default());
        st.encode(&mut e, src);
        st.belt(&mut e, src, &raft);
        s.v.acquire(&g, &p);
        g.queue.submit([e.finish()]);
        st.flow(&raft).unwrap();
        let fresh = st.flows().unwrap();
        let mut f = Seam::new(&g, &cx, &s).unwrap();
        while f.raft.is_none() {
            f.poll().unwrap();
            assert!(t0.elapsed().as_secs() < 60, "no model");
            thread::sleep(Duration::from_millis(1));
        }
        f.n = 3;
        let mut runs = 0;
        for k in 0..10 {
            f.pv.pose(&o);
            let mut e = g.device.create_command_encoder(&Default::default());
            let run = f.frame(&mut e, src);
            s.v.acquire(&g, &p);
            g.queue.submit([e.finish()]);
            if run {
                f.go.send(()).unwrap();
                runs += 1;
            }
            while f.busy {
                f.poll().unwrap();
                thread::sleep(Duration::from_millis(1));
            }
            if !f.has {
                continue;
            }
            let fl = f.st.lock().unwrap().flows().unwrap();
            let dev = (fl.iter().flatten())
                .zip(fresh.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("frame {k}: run {run}, flow drawn vs fresh: max |Δ| {dev:e} px");
            assert!(dev < 1e-3, "frame {k}: {dev} px");
        }
        assert_eq!(runs, 3);
        s.v.release(&g, &p);
    }

    /// The RAFT seam option while playing v635 (drawn as the live output draws, 640 × 360, across the seam): off, a
    /// played frame is the template preview's picture, as before the option; on, once its first flow is in, the seam
    /// preview's.
    #[test]
    fn seam_option() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES not set: skipping");
            return;
        };
        let g = Arc::new(Gpu::new().unwrap());
        let cx = Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        let w = crate::library::pair(&d.join("v635.insv"));
        let mut p = Player::open(&g, &cx, w, &ctx, &still(&g, &cx), &gyro::worker(), None).unwrap();
        // looking along the seam
        (p.looping, p.flow, p.view.yaw) = (true, false, 90.0);
        p.play();
        let (tex, tv) = target(&g, 640, 360);
        // the next new frame drawn, and the same frame through the template preview and through the seam's
        let shot = |p: &mut Player| {
            let t0 = Instant::now();
            while !p.fresh || p.want.is_some() {
                p.tick(false);
                assert!(t0.elapsed().as_secs() < 60, "no frame");
                thread::sleep(Duration::from_millis(1));
            }
            p.fresh = false;
            crate::live::Source::draw(p, &tv, [640, 360], false);
            let a = stitch::read_texture(&g.device, &g.queue, &tex);
            let h = p.held.unwrap();
            let (o, _) = p.pose(&h);
            let s = p.s();
            let r = |pv: &Preview| {
                pv.pose(&o);
                let mut e = g.device.create_command_encoder(&Default::default());
                pv.render(&mut e, &tv, [640, 360], &p.view, s.src(&h));
                s.v.acquire(&g, &h);
                g.queue.submit([e.finish()]);
                stitch::read_texture(&g.device, &g.queue, &tex)
            };
            let tpl = r(&s.pv);
            let flow = (p.seam.as_ref().filter(|f| f.has && !f.off)).map(|f| r(&f.pv));
            (a, tpl, flow)
        };
        for _ in 0..5 {
            let (a, tpl, flow) = shot(&mut p);
            assert!(
                flow.is_none() && a == tpl,
                "off: the template preview's picture"
            );
        }
        p.flow = true;
        let t0 = Instant::now();
        // once a flow is in no run starts, and a frame is compared with none out; the pacing left out (this loop,
        // reading pictures back, drops frames of its own)
        loop {
            let ready = p.seam.as_mut().is_some_and(|f| {
                // a late frame doubles `n`, and a run still out ends `cold`
                (f.n, f.off) = (if f.has { 1 << 32 } else { 1 }, false);
                f.cold |= f.has;
                f.has && !f.busy
            });
            let (a, tpl, flow) = shot(&mut p);
            if let (true, Some(flow)) = (ready, flow) {
                let n = a.iter().zip(&tpl).filter(|(x, y)| x != y).count();
                eprintln!(
                    "on, {:.1} s in: the seam preview's picture, {n} bytes from the template's",
                    t0.elapsed().as_secs_f64()
                );
                assert!(a == flow && n > 0);
                break;
            }
            assert!(t0.elapsed().as_secs() < 120, "no flow");
        }
    }
}
