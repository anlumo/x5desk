//! Player: one open work. The audio clock (or the silent fallback) is the master; the media thread (`clip`) decodes
//! ahead; each UI frame takes the newest frame due and draws it with crate stitch's preview at window size, or,
//! paused, the RAFT stitch of that frame made on a worker thread. The UI never paces playback. Stabilisation
//! (`gyro`) turns each frame's view by its pose: the preview applies orientation and rolling shutter; the RAFT stitch
//! applies the rolling shutter and the preview turns the resulting body-frame equirect.
//!
//! The library shares the player with the live output thread (`Arc<Mutex<Player>>`), which takes the frame due and
//! draws it on its own clock while the UI may not run (a hidden window); the audio player sits on a thread of its own
//! for that ([`Aud`]).

use crate::{
    Test,
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
const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    Perspective,
    Panini,
    Equirect,
    Planet,
}

impl Look {
    const ALL: [(Look, &str); 4] = [
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

    fn view(self) -> View {
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
    fn fov(self) -> (f32, f32) {
        match self {
            Look::Perspective => (20.0, 160.0),
            Look::Panini => (60.0, 300.0),
            Look::Equirect => (360.0, 360.0),
            Look::Planet => (90.0, 350.0),
        }
    }
}

static SRC_ID: AtomicU64 = AtomicU64::new(1);

/// A playable video (the full-resolution files or the proxy): media thread, preview renderer, the lens bind groups
/// of every ring slot.
struct Src {
    id: u64,
    v: Video,
    pv: Preview,
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
    fn open(
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

    fn src(&self, p: &Pic) -> [&stitch::Src; 2] {
        [0, 1].map(|i| &self.bg[i][(p.f[self.lens[i].0] % OUT) as usize])
    }

    /// A RAFT job for `p`: its lens pictures copied into `snap` (made or remade to fit), stitched with the rolling-shutter
    /// term `rs`.
    #[allow(clippy::too_many_arguments)]
    fn job(
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
        if snap.as_ref().is_none_or(|x| (x.w, x.h) != (w, h)) {
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
            let bpr = w.next_multiple_of(256) as u64;
            *snap = Some(Snap {
                y: [0, 1].map(|_| tex(w, h, wgpu::TextureFormat::R8Unorm)),
                uv: [0, 1].map(|_| tex(w / 2, h / 2, wgpu::TextureFormat::Rg8Unorm)),
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
        let bpr = w.next_multiple_of(256);
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
type Key = (u64, usize, bool);
type Reply = Result<(Key, wgpu::Texture), String>;

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
            // a surface configure the UI retries waits for the job (burn submits throughout)
            let r = {
                let _g = crate::GATE.read();
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
struct Snap {
    y: [wgpu::Texture; 2],
    uv: [wgpu::Texture; 2],
    buf: wgpu::Buffer,
    w: u32,
    h: u32,
}

/// The viewer's texture, registered with egui.
struct Tgt {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    id: egui::TextureId,
    rs: RenderState,
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
enum Cmd {
    Play,
    Pause,
    Seek(f64),
    Vol(f32),
}

/// The audio player on a thread of its own: cpal's stream stays on the thread that made it, while the player (its
/// seeks and loops) runs on the UI thread and on the live output thread. A command returns once the audio player
/// has it, so after a seek `audio::Clock::get` reads None until the device takes it, on whichever thread asks.
struct Aud {
    tx: mpsc::Sender<Cmd>,
    ack: mpsc::Receiver<()>,
}

impl Aud {
    fn cmd(&self, c: Cmd) {
        if self.tx.send(c).is_ok() {
            let _ = self.ack.recv();
        }
    }
}

/// The audio player of `p`, or the silent clock when it has no AAC track or there is no output device; its clock;
/// the audio track's presentation offset (edit list) in seconds; whether it has sound.
fn audio(p: &Path, end: f64) -> (Aud, audio::Clock, f64, bool) {
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
            let (a, sound) = match audio::Player::open(&p) {
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
        };
        if let Some(t) = test {
            p.vol = t.vol.unwrap_or(p.vol).clamp(0.0, 1.0);
            p.audio.cmd(Cmd::Vol(p.vol));
            if let Some(l) = t.look {
                (p.look, p.view) = (l, l.view());
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
        let (p, skip) = s.v.take(now);
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
        self.want.is_none()
            && self.held.is_some()
            && self.stab.ready()
            && raft
            && self.measure.is_none_or(|_| self.done)
    }

    fn render(&mut self, rs: &RenderState, size: [u32; 2]) {
        let g = &self.g;
        if self.tgt.is_none() || self.size != size {
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
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let mut r = rs.renderer.write();
            match &mut self.tgt {
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
                    self.tgt = Some(Tgt {
                        tex,
                        view,
                        id,
                        rs: rs.clone(),
                    });
                }
            }
            self.size = size;
        }
        let Some(p) = self.held else { return };
        let ((o, key), s, t) = (self.pose(&p), self.s(), self.tgt.as_ref().unwrap());
        // the lenses take orientation and rolling shutter, the RAFT stitch (body frame) the orientation
        s.pv.pose(&o);
        let mut e = g.device.create_command_encoder(&Default::default());
        let mut tint = self.tinted;
        match self.pano.as_ref().filter(|x| !self.playing && x.0 == key) {
            Some((_, pano)) => s.pv.render(&mut e, &t.view, size, &self.view, pano),
            None => {
                let src = s.src(&p);
                if !tint {
                    s.pv.colour(&mut e, src);
                    tint = true;
                }
                s.pv.render(&mut e, &t.view, size, &self.view, src);
                s.v.acquire(g, &p);
            }
        }
        g.queue.submit([e.finish()]);
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
        if ui
            .add(egui::Slider::new(&mut t, 0.0..=end).show_value(false))
            .changed()
        {
            let j = clip::frame_at(&self.s().v.times, t);
            if j != k {
                self.seek(j);
            }
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
        let r = if self.look == Look::Equirect {
            let w = rect.width().min(rect.height() * 2.0);
            egui::Rect::from_center_size(rect.center(), egui::vec2(w, w / 2.0))
        } else {
            rect
        };
        let v = &mut self.view;
        if resp.dragged() {
            let d = resp.drag_delta();
            // degrees per point: the picture follows the pointer
            let k = if self.look == Look::Equirect {
                360.0
            } else {
                v.fov.min(180.0)
            } / r.width().max(1.0);
            v.yaw = (v.yaw - d.x * k).rem_euclid(360.0);
            v.pitch = (v.pitch + d.y * k).clamp(-90.0, 90.0);
            self.dirty = true;
        }
        if resp.hovered() {
            let y = ui.input(|i| i.smooth_scroll_delta().y);
            if y != 0.0 {
                let (a, b) = self.look.fov();
                v.fov = (v.fov * (-y * 0.002).exp()).clamp(a, b);
                self.dirty = true;
            }
        }
        if resp.double_clicked() {
            *v = self.look.view();
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
        let mut lines = vec![];
        if self.held.is_none() || self.want.is_some() {
            lines.push("Decoding…".to_string());
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

    fn tick(&mut self) {
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
    }
}

#[cfg(test)]
mod verify;

#[cfg(test)]
mod tests {
    use super::*;

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
