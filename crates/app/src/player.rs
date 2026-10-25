//! Player: one open work. The audio clock (or the silent fallback) is the master; the media thread (`clip`) decodes
//! ahead; each UI frame takes the newest frame due and draws it with crate stitch's preview at window size, or,
//! paused, the RAFT stitch of that frame made on a worker thread. The UI never paces playback. Stabilisation
//! (`gyro`) turns each frame's view by its pose: the preview applies orientation and rolling shutter; the RAFT stitch
//! applies the rolling shutter and the preview turns the resulting body-frame equirect.
//!
//! Deep Track (`ai`): with Deep Track armed, a box dragged around a subject in the perspective view starts a run from
//! the frame shown; the run's states show under the seek bar, the box on the picture, and once it is over the view
//! follows the subject (the follow view the export takes). Privacy preview: the faces of the paused frame blurred as
//! the export blurs them. Auto edit hands the clip to the editor.
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
        Arc,
        atomic::{AtomicU64, Ordering::Relaxed},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use stitch::{Calib, Ctx, Flow, Frame, Mode, Opts, Pose, Preview, Raft, Stitcher, View};
use video::Gpu;

const OUT: u64 = video::decode::OUT as u64;
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
        let cal = Calib::parse(&m.offset)?;
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
            // DUAL_STREAM_TRACK_REVERSE: track 0 holds lens 1
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
            (0..OUT)
                .map(|s| cx.rect(v.rings[r].texture(s), rect))
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
        [0, 1].map(|i| &self.bg[i][(p.f[self.lens[i].0] % OUT) as usize])
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
        let deep = self.v.rings[0].texture(0).format() == wgpu::TextureFormat::P010;
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
            let tex = self.v.rings[r].texture(p.f[r]);
            let o = (i as u64) * bpr as u64 * h as u64 * 3 / 2;
            for (aspect, d, off, dst) in [
                (wgpu::TextureAspect::Plane0, 1, o, &x.y[i]),
                (
                    wgpu::TextureAspect::Plane1,
                    2,
                    o + bpr as u64 * h as u64,
                    &x.uv[i],
                ),
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
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: rx / d,
                            y: ry / d,
                            z: 0,
                        },
                        aspect,
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
            .map(|r| wgpu::TextureTransition {
                texture: self.v.rings[r].texture(p.f[r]),
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
/// The faces of a frame of the main file under a stabilisation (lock, σ bits).
type FaceKey = (usize, gyro::Mode, u64);
type Faces = Result<Vec<track::Region>, String>;
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
                Err(e) if e.kind() == std::io::ErrorKind::Other => {
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
    /// Deep Track: the run on this clip; a box drawn in the view starts one (armed); the view follows the subject;
    /// the box being drawn; the picture's aspect
    pub trk: Option<Arc<ai::Track>>,
    arm: bool,
    pub follow: bool,
    from: Option<egui::Pos2>,
    aspect: f32,
    /// privacy preview: the paused frame's faces blurred; the faces found, a search out, the unblurred picture and the
    /// pass
    pub blur: bool,
    regs: Option<(FaceKey, Vec<track::Region>)>,
    fq: Option<(FaceKey, mpsc::Receiver<Faces>)>,
    bt: Option<(wgpu::Texture, wgpu::TextureView, crate::export::Blur)>,
    /// the editor is to make an auto edit of this clip
    pub auto: bool,
    /// test hook: the box to draw once the frame shows; the stabilisation to switch to once its run is over
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
        if meta.offset.is_empty() {
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
            aspect: 16.0 / 9.0,
            blur: false,
            regs: None,
            fq: None,
            bt: None,
            auto: false,
            tbox: None,
            restab: None,
        };
        if let Some(t) = test {
            (p.tbox, p.restab) = (t.tbox, t.restab);
            p.blur = t.blur;
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

    fn fkey(&self) -> Option<FaceKey> {
        let p = self.held?;
        Some((self.main_k(&p), self.stab.mode, self.stab.smooth.to_bits()))
    }

    /// Deep Track of the subject in box `b` (0..1 of the picture) of the perspective view, from the frame shown to the
    /// clip's end; the view follows it once the run is over.
    pub fn track(&mut self, b: [f32; 4]) -> Result<(), String> {
        let f = (self.work.main.first().cloned())
            .ok_or("Deep Track needs the main file, not the proxy alone")?;
        if self.look != Look::Perspective {
            return Err("Deep Track: draw the box in the perspective view".into());
        }
        let p = self.held.ok_or("no frame shown yet")?;
        if let Some(t) = &self.trk {
            t.stop.store(true, Relaxed);
        }
        let k0 = self.main_k(&p);
        let s = ai::subject(&self.view, self.aspect, b);
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
        (self.follow, self.arm) = (true, false);
        Ok(())
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
        // the view follows the tracked subject, turned from the track's frame into the frame shown (the stabilisation
        // may have changed since)
        if let (true, Some(t), Some(p)) = (self.follow, &self.trk, self.held)
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
                Ok(v) => self.regs = Some((*k, v)),
                Err(e) => (self.note, self.blur) = (Some(format!("Face blur preview: {e}")), false),
            }
            (self.fq, self.dirty) = (None, true);
        }
        if self.blur
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
                    ai::faces(&g, &cx, &f, st, k.0)
                }));
                ctx.request_repaint();
            });
            self.fq = Some((k, rx));
        }
        if self.tbox.is_some() && self.held.is_some() && self.want.is_none() && self.stab.ready() {
            let b = self.tbox.take().unwrap();
            if let Err(e) = self.track(b) {
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
        let faces = !self.blur
            || self.playing
            || (self.fkey()).is_some_and(|k| self.regs.as_ref().is_some_and(|r| r.0 == k));
        self.want.is_none()
            && self.held.is_some()
            && self.stab.ready()
            && raft
            && self.measure.is_none_or(|_| self.done)
            && self.tbox.is_none()
            && self.restab.is_none()
            && self.trk.as_ref().is_none_or(|t| t.st().over())
            && faces
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
            follow: (self.trk.clone()).filter(|t| self.follow && t.st() == ai::St::Done),
            faces: self.blur,
        })
    }

    fn render(&mut self, rs: &RenderState, size: [u32; 2]) {
        let g = &self.g;
        Tgt::fit(&mut self.tgt, g, rs, size);
        self.size = size;
        let Some(p) = self.held else { return };
        // the privacy preview: the picture into a texture of its own, blurred from there into the viewer's
        let regs = (self.regs.as_ref())
            .filter(|r| self.blur && !self.playing && Some(r.0) == self.fkey())
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
        let ((o, key), s, t) = (self.pose(&p), self.s(), self.tgt.as_ref().unwrap());
        let dst = match (regs, &bt) {
            (Some(_), Some(b)) => &b.1,
            _ => &t.view,
        };
        // the lenses take orientation and rolling shutter, the RAFT stitch (body frame) the orientation
        s.pv.pose(&o);
        let mut e = g.device.create_command_encoder(&Default::default());
        let mut tint = self.tinted;
        match self.pano.as_ref().filter(|x| !self.playing && x.0 == key) {
            Some((_, pano)) => s.pv.render(&mut e, dst, size, &self.view, pano),
            None => {
                let src = s.src(&p);
                if !tint {
                    s.pv.colour(&mut e, src);
                    tint = true;
                }
                s.pv.render(&mut e, dst, size, &self.view, src);
                s.v.acquire(g, &p);
            }
        }
        if let (Some(r), Some(b)) = (regs, &mut bt) {
            b.2.run(&g.device, &g.queue, &mut e, &b.1, &t.view, &self.view, r);
        }
        g.queue.submit([e.finish()]);
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
        // the Deep Track run's states under the bar (the rail sits a handle's radius in from the ends)
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
                    ui.add_enabled_ui(main && self.look == Look::Perspective, |ui| {
                        ui.toggle_value(&mut self.arm, "Deep Track")
                            .on_hover_text("Drag a box around a subject in the view: it is tracked from this frame on")
                            .on_disabled_hover_text("Deep Track needs the main file and the perspective view");
                    });
                    if over.is_some() {
                        if ui
                            .checkbox(&mut self.follow, "Follow")
                            .on_hover_text("The view follows the tracked subject (and so does a flat export)")
                            .changed()
                        {
                            self.dirty = true;
                        }
                        if ui.button("Clear track").clicked() {
                            (self.trk, self.follow) = (None, false);
                        }
                    }
                }
            }
            if ui
                .toggle_value(&mut self.blur, "Blur faces")
                .on_hover_text("Preview the export's face blur on the paused frame (the export starts with it on)")
                .changed()
            {
                self.dirty = true;
            }
            if ui
                .add_enabled(main, egui::Button::new("Auto edit"))
                .on_hover_text("Highlights of this clip, framed automatically, as an editor timeline")
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
        self.aspect = r.aspect_ratio();
        let mut live = None;
        if self.arm && self.look == Look::Perspective {
            let (l, b) = ai::drag_box(ui, &resp, r, &mut self.from);
            live = l;
            if let Some(b) = b
                && let Err(e) = self.track(b)
            {
                self.note = Some(e);
            }
        } else if self.look.steer(ui, &resp, r, &mut self.view) {
            // a hand on the view takes over from the follow view
            self.follow &= !resp.dragged();
            self.dirty = true;
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
        if let (Some(t), Some(h), Look::Perspective) = (&self.trk, self.held, self.look)
            && let k = self.main_k(&h)
            && let Some((s, q)) = t.at(k).zip(t.turn(k, self.pose(&h).0.q))
            && let Some(o) = ai::outline(&self.view, r, q * s.raw, s.size)
        {
            p.add(egui::Shape::closed_line(
                o.to_vec(),
                egui::Stroke::new(2.0, ai::colour(s.state)),
            ));
        }
        let mut lines = vec![];
        if self.held.is_none() || self.want.is_some() {
            lines.push("Decoding…".to_string());
        }
        if self.arm {
            lines.push("Deep Track: drag a box around the subject".into());
        }
        if let Some(t) = &self.trk {
            lines.push(match t.st() {
                ai::St::Done if self.follow => "Deep Track: following the subject".into(),
                st => st.line("Deep Track"),
            });
        }
        if self.blur {
            let n = (self.regs.as_ref())
                .filter(|r| Some(r.0) == self.fkey())
                .map(|r| r.1.len());
            lines.push(match (self.playing, n) {
                (true, _) => "Face blur preview: pause on a frame".into(),
                (_, Some(n)) => format!("Face blur preview: {n} faces"),
                _ => "Face blur preview: finding faces…".into(),
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
        if let Some(s) = self.stab.stage {
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
        let (g, (o, key), s) = (&self.g, self.pose(&p), self.s());
        s.pv.pose(&o);
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
        if tint {
            s.pv.colour(&mut e, src);
        }
        s.pv.set(&v, r);
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
                Some((_, x)) => s.pv.draw(&mut rp, x),
                None => s.pv.draw(&mut rp, src),
            }
        }
        if raft.is_none() {
            s.v.acquire(g, &p);
        }
        g.queue.submit([e.finish()]);
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
        if let Some(t) = &self.trk {
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
                o = Some(ctx.run_ui(input.clone(), |ui| {
                    egui::CentralPanel::default().show(ui, |ui| p.controls(ui));
                }));
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
}
