//! Live screen (milestone 4): a live source's picture and audio out as RTMP, as sources of our OBS plugin and as the
//! PipeWire camera. The source is pluggable ([`Source`]): today the player (its reframed view or the 360 equirect);
//! the capture stage adds the camera's live preview and Webcam Mode.
//!
//! The outputs run on a thread of their own ([`Outs::step`]) on one frame clock at the output rate: at each tick the
//! source takes the picture due ([`Source::tick`]) and draws it into each output's target at that output's size
//! (`Rgba8Unorm`), timestamped with the tick's scheduled CLOCK_MONOTONIC time. The thread owns the outputs; it shares
//! with the UI only the settings and a copy of their state, and the source, which it locks just to draw (targets
//! made ready before, the rest after). So neither waits for the other's work: a window the compositor throttles
//! (hidden, minimised, display off) stalls no output, and an output waiting for the GPU or PipeWire stalls no UI.
//! - RTMP: target → NV12 (`stitch::Yuv`) → copy into the encoder's input (acquire / release) → the encoder thread
//!   (H.264 CBR, 100 ms rate control buffer) → `live::rtmp`; a tick without a picture (a seek) repeats the last one;
//!   an IDR when a session goes live after lost video. Audio: the source's AAC decoded, laid on the same clock as it
//!   is heard (silence where none comes: paused, seeking, no audio track) and encoded again (`media::aacenc`).
//! - OBS: `live::obs::Server` with sources `360` and `view`, each rendered only while a client watches it, into a
//!   `live::dmabuf::Ring` in the modifiers its clients import; audio decoded to f32.
//! - PipeWire: `live::pw::Cam` at the clock's rate in the mode the consumer picked (flat 1920 × 1080 or the
//!   3840 × 1920 equirect), NV12 / BGRx through a conversion pass.

use eframe::{egui, egui_wgpu::RenderState};
use live::{
    dmabuf::{Fmt, Ring},
    flv,
    obs::{self, Format, Frame, Server},
    pw,
    rtmp::{self, Rtmp, State},
};
use media::{
    aacenc,
    audio::Aac,
    mp4::{Codec, Mp4},
};
use std::{
    fmt::Write as _,
    io::Write,
    os::{
        fd::AsFd,
        unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use video::{Encoder, Gpu, Input, Packet, Rate};

/// What the outputs show.
pub trait Source {
    /// For the UI: what is playing.
    fn name(&self) -> &str;
    /// Takes the picture due now. The output thread calls it before each frame: the UI may not run.
    fn tick(&mut self);
    /// Has a picture to draw.
    fn ready(&self) -> bool;
    /// Draws the picture shown now into `t` (`Rgba8Unorm`, `size`) and submits: the reframed view (`pano` false)
    /// or the 360 equirect.
    fn draw(&mut self, t: &wgpu::TextureView, size: [u32; 2], pano: bool);
    /// The AudioSpecificConfig of its AAC audio.
    fn asc(&mut self) -> Option<&[u8]>;
    /// The AAC packets heard since the last call, each with the CLOCK_MONOTONIC ns it started playing at.
    fn audio(&mut self, f: &mut dyn FnMut(u64, &[u8]));
}

/// A source shared with the output thread.
pub type Src = Arc<Mutex<dyn Source + Send>>;

pub fn mono() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

/// A file's AAC track handed out packet by packet as the audio clock passes them.
pub struct Tap {
    m: Mp4,
    t: usize,
    pub asc: Vec<u8>,
    next: usize,
    last: f64,
    /// CLOCK_MONOTONIC ns − clock, smoothed: the clock moves in steps of the device's buffers. NaN after a jump
    /// until the clock moves past `c0`, where it jumped to.
    off: f64,
    c0: f64,
    buf: Vec<u8>,
}

impl Tap {
    pub fn open(p: &Path) -> Option<Tap> {
        let m = Mp4::open(p).ok()?;
        let t = m
            .tracks
            .iter()
            .position(|t| matches!(t.codec, Codec::Aac { .. }))?;
        let Codec::Aac { asc, .. } = &m.tracks[t].codec else {
            return None;
        };
        Some(Tap {
            asc: asc.clone(),
            m,
            t,
            next: 0,
            last: f64::NAN,
            off: f64::NAN,
            c0: 0.0,
            buf: vec![],
        })
    }

    /// Hands `f` every packet that started playing since the last call, by `clock` (seconds of the track heard, as
    /// `media::audio::Clock`) read at `now` (CLOCK_MONOTONIC ns), with the time it started. A jump (seek, loop, a
    /// stall over 0.5 s) restarts at the clock once it moves again: a seek holds it until its frame shows, for less
    /// time than the smoothing would take to follow.
    pub fn take(&mut self, clock: f64, now: u64, f: &mut dyn FnMut(u64, &[u8])) {
        let Tap {
            m,
            t,
            next,
            last,
            off,
            c0,
            buf,
            ..
        } = self;
        let tr = &m.tracks[*t];
        let ts = tr.timescale as f64;
        let o = now as f64 - clock * 1e9;
        if !(clock >= *last && clock < *last + 0.5) {
            *next = tr.samples.partition_point(|s| (s.dts as f64) < clock * ts);
            (*off, *c0) = (f64::NAN, clock);
        }
        *last = clock;
        match o - *off {
            _ if off.is_nan() && clock <= *c0 => return,
            d if d.is_nan() => *off = o,
            // a pause moves it by its length, jitter by a device buffer or so
            d if d.abs() > 50e6 => *off = o,
            d => *off += d / 16.0,
        }
        while let Some(s) = tr.samples.get(*next)
            && s.dts as f64 <= clock * ts
        {
            *next += 1;
            if m.read(s, buf).is_ok() {
                f((*off + s.dts as f64 / ts * 1e9).max(0.0) as u64, buf);
            }
        }
    }
}

/// RTMP sizes: name, width, height, 360.
const RES: [(&str, u32, u32, bool); 3] = [
    ("1920 × 1080", 1920, 1080, false),
    ("2560 × 1440", 2560, 1440, false),
    ("3840 × 1920 (360°)", 3840, 1920, true),
];
const FPS: [u32; 3] = [25, 30, 60];
/// OBS sources: id, name, size, 360.
const OBS: [(u32, &str, u32, u32, bool); 2] =
    [(1, "360", 3840, 1920, true), (2, "view", 1920, 1080, false)];
const CAM: &str = "X5 Camera (x5desk)";
const RING: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC);
/// The output thread's pass while no output runs: OBS clients come and go.
const IDLE: Duration = Duration::from_millis(20);
/// The Live screen counts as shown this long after it was drawn: the output thread renders its preview.
const SEEN: Duration = Duration::from_millis(200);
/// RTMP audio: AAC-LC stereo, kbit/s. Silence fills the timeline once the source's audio is `LATE` ns behind the
/// clock (paused, seeking, none), up to `SLACK` ns behind it: audio heard later still fits in after that.
const AKBPS: u32 = 128;
const LATE: u64 = 200_000_000;
const SLACK: u64 = 100_000_000;
/// RTMP audio: a jump of the source's audio beyond this (ns) is a gap filled with silence, or an overlap cut off;
/// less is the clock's jitter (a device buffer) or drift.
const JUMP: u64 = 40_000_000;

/// Settings, `<config>/x5desk/live` (`k=v` lines, owner-only: it holds the stream key).
#[derive(Clone, Debug, PartialEq)]
struct Set {
    url: String,
    key: String,
    res: usize,
    fps: u32,
    kbps: u32,
    obs: bool,
    cam: bool,
}

impl Default for Set {
    fn default() -> Set {
        Set {
            url: String::new(),
            key: String::new(),
            res: 0,
            fps: 30,
            kbps: 6000,
            obs: true,
            cam: false,
        }
    }
}

impl Set {
    fn file() -> Option<PathBuf> {
        crate::xdg("XDG_CONFIG_HOME", ".config").map(|d| d.join("x5desk").join("live"))
    }

    fn parse(t: &str) -> Set {
        let mut s = Set::default();
        for (k, v) in t.lines().filter_map(|l| l.split_once('=')) {
            match k {
                "url" => s.url = v.into(),
                "key" => s.key = v.into(),
                "res" => s.res = v.parse().unwrap_or(0).min(RES.len() - 1),
                "fps" => s.fps = v.parse().ok().filter(|f| FPS.contains(f)).unwrap_or(30),
                "kbps" => s.kbps = v.parse().unwrap_or(s.kbps).clamp(500, 50000),
                "obs" => s.obs = v == "1",
                "cam" => s.cam = v == "1",
                _ => {}
            }
        }
        s
    }

    fn text(&self) -> String {
        let l = |s: &str| s.replace(['\n', '\r'], "");
        format!(
            "url={}\nkey={}\nres={}\nfps={}\nkbps={}\nobs={}\ncam={}\n",
            l(&self.url),
            l(&self.key),
            self.res,
            self.fps,
            self.kbps,
            self.obs as u8,
            self.cam as u8
        )
    }

    /// Writes `f` owner-only, also when it exists already; a directory it creates is owner-only too.
    fn save(&self, f: &Path) -> std::io::Result<()> {
        if let Some(d) = f.parent() {
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(d)?;
        }
        let mut o = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(f)?;
        o.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        o.write_all(self.text().as_bytes())
    }
}

/// What the test hook asks for (`X5DESK_TEST`): stream to `rtmp`, serve OBS, publish the camera.
#[derive(Clone, Debug, Default)]
pub struct Hook {
    pub rtmp: Option<String>,
    pub res: Option<usize>,
    pub fps: Option<u32>,
    pub kbps: Option<u32>,
    pub obs: bool,
    pub cam: bool,
    /// the OBS socket instead of `live::obs::path` (tests)
    pub sock: Option<PathBuf>,
}

fn tex(g: &Gpu, w: u32, h: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
    g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("live"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn view(t: &wgpu::Texture, aspect: wgpu::TextureAspect) -> wgpu::TextureView {
    t.create_view(&wgpu::TextureViewDescriptor {
        aspect,
        usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
        ..Default::default()
    })
}

/// The Format of OBS source `w` × `h` exported with `modifier`, at `fps`, with the audio of `asc`.
fn format(modifier: u64, w: u32, h: u32, fps: u32, asc: Option<&[u8]>) -> Format {
    let (rate, ch) = asc
        .map(media::mp4::asc_info)
        .and_then(Result::ok)
        .unwrap_or((0, 0));
    Format {
        modifier,
        w,
        h,
        fourcc: Fmt::Rgba.drm(),
        fps: [fps, 1],
        color: obs::BT709,
        rate,
        ch: ch as u32,
    }
}

/// An RTMP stream: the picture, its NV12 conversion, the encoder's input and thread, the audio encoder, the
/// publisher.
struct Tx {
    rtmp: Arc<Rtmp>,
    input: Arc<Input>,
    tx: Option<mpsc::Sender<u32>>,
    th: Option<thread::JoinHandle<()>>,
    w: u32,
    h: u32,
    pano: bool,
    /// the picture (`Rgba8Unorm`, the stream's size) the converter reads
    pv: wgpu::TextureView,
    yuv: stitch::Yuv,
    /// frames handed to the encoder, encoded; frames left out (encoder behind); ticks that repeated the last frame
    f: u64,
    done: Arc<AtomicU64>,
    skip: u64,
    rep: u64,
    /// CLOCK_MONOTONIC ns of ms 0 and of audio sample 0
    t0: u64,
    /// the audio: encoder (stereo at `rate`), samples fed, the last timestamp (ms), stereo scratch; whether the
    /// timeline ends in the source's audio (else silence, or nothing yet)
    enc: aacenc::Enc,
    rate: u32,
    fed: u64,
    aud: u32,
    buf: Vec<f32>,
    cont: bool,
    state: State,
    /// bitrate: bytes at the last reading and when, kbit/s
    meter: (u64, Instant, f64),
    err: Arc<Mutex<Option<String>>>,
    st: rtmp::Stats,
}

impl Drop for Tx {
    fn drop(&mut self) {
        // the encoder thread finishes what it has, then the publisher sends the rest (rtmp::STOP)
        self.tx = None;
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

impl Tx {
    /// CLOCK_MONOTONIC ns of the next audio sample.
    fn next(&self) -> u64 {
        self.t0 + self.fed * 1_000_000_000 / self.rate as u64
    }

    /// Encodes `n` stereo frames: of `buf`, or silence.
    fn put(&mut self, n: usize, silent: bool) {
        static Z: [f32; 2048] = [0.0; 2048];
        let Tx {
            enc,
            buf,
            rtmp,
            aud,
            fed,
            rate,
            ..
        } = self;
        // pts in samples from t0, the encoder's priming frame before it
        let mut f = |au: &[u8], pts: i64, _| {
            *aud = (*aud).max((pts.max(0) as u64 * 1000 / *rate as u64) as u32);
            rtmp.audio(*aud, au);
        };
        match silent {
            true => (0..n).step_by(1024).for_each(|k| {
                enc.push(&Z[..2 * (n - k).min(1024)], &mut f);
            }),
            false => enc.push(&buf[..2 * n], &mut f),
        }
        *fed += n as u64;
    }

    /// Lays `x` (`ch` channels at the stream's rate, starting to play at `ns`) on the stream's timeline: a gap before
    /// it becomes silence, an overlap is cut off its start. Following the source's own audio, they have to exceed
    /// `JUMP`; at the start of its audio they are exact.
    fn pcm(&mut self, ns: u64, x: &[f32], ch: usize) {
        let (at, r, tol) = (self.next(), self.rate as u64, self.cont as u64 * JUMP);
        if ns > at + tol {
            self.put(((ns - at) * r / 1_000_000_000) as usize, true);
        }
        let cut = match at.saturating_sub(ns) {
            d if d > tol => (d * r / 1_000_000_000) as usize,
            _ => 0,
        };
        self.buf.clear();
        for f in x.chunks_exact(ch).skip(cut) {
            self.buf.extend([f[0], f[(ch > 1) as usize]]);
        }
        self.put(self.buf.len() / 2, false);
        self.cont = true;
    }

    /// Silence up to `SLACK` behind `now` once the source's audio is `LATE`.
    fn quiet(&mut self, now: u64) {
        let at = self.next();
        if now > at + LATE {
            self.put(
                ((now - SLACK - at) * self.rate as u64 / 1_000_000_000) as usize,
                true,
            );
            self.cont = false;
        }
    }
}

/// The encoder thread: encodes each frame as soon as it arrives (the producer keeps at most `SLOTS` in flight),
/// each packet straight to the publisher. One frame at a time would leave the encode queue idle for every
/// round trip through this thread.
fn encode(
    g: Arc<Gpu>,
    mut e: Encoder,
    rx: mpsc::Receiver<u32>,
    r: Arc<Rtmp>,
    done: Arc<AtomicU64>,
    err: Arc<Mutex<Option<String>>>,
) {
    use mpsc::TryRecvError::Disconnected;
    let mut p = Packet::default();
    // frames in flight; when the oldest one's wait began
    let (mut n, mut t) = (0, Instant::now());
    let x = loop {
        let m = match n {
            0 => rx.recv().map_err(|_| Disconnected),
            _ => rx.try_recv(),
        };
        match m {
            Ok(ms) => match e.encode(&g, ms as i64) {
                Ok(_) => {
                    t = if n == 0 { Instant::now() } else { t };
                    n += 1;
                }
                Err(x) => break format!("encoder: {x}"),
            },
            Err(Disconnected) if n == 0 => return,
            // nothing new: the oldest packet, looking at the channel every 2 ms
            Err(_) => match e.packet(&mut p, 2_000_000) {
                Ok(true) => {
                    r.video(p.pts as u32, 0, p.key, p.nal());
                    done.store(p.frame + 1, Relaxed);
                    (n, t) = (n - 1, Instant::now());
                }
                Ok(false) if t.elapsed() > Duration::from_secs(1) => {
                    break "encoder: no packet within 1 s".into();
                }
                Ok(false) => {}
                Err(x) => break format!("encoder: {x}"),
            },
        }
    };
    *err.lock().unwrap() = Some(x);
}

/// An OBS source: its ring (and views, the modifiers asked for), its state.
#[derive(Default)]
struct Out {
    ring: Option<(Ring, Vec<wgpu::TextureView>, Vec<u64>)>,
    s: Os,
}

/// RGBA → NV12 planes (BT.709 limited, chroma the 2 × 2 mean) and RGBA → BGRA, for the PipeWire camera.
struct Conv {
    pipe: [wgpu::RenderPipeline; 3],
    bgl: wgpu::BindGroupLayout,
}

const CONV: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
const K = vec3f(0.2126, 0.7152, 0.0722);
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn rgb(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return textureLoad(src, vec2i(p.xy), 0);
}
@fragment fn y(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return vec4f((16.0 + 219.0 * dot(textureLoad(src, vec2i(p.xy), 0).rgb, K)) / 255.0, 0.0, 0.0, 1.0);
}
@fragment fn uv(@builtin(position) p: vec4f) -> @location(0) vec4f {
    let q = vec2i(p.xy) * 2;
    let c = 0.25 * (textureLoad(src, q, 0) + textureLoad(src, q + vec2i(1, 0), 0) + textureLoad(src, q + vec2i(0, 1), 0)
        + textureLoad(src, q + vec2i(1, 1), 0)).rgb;
    let y = dot(c, K);
    return vec4f((128.0 + 224.0 * vec2f((c.b - y) / 1.8556, (c.r - y) / 1.5748)) / 255.0, 0.0, 1.0);
}
"#;

impl Conv {
    fn new(g: &Gpu) -> Conv {
        let d = &g.device;
        let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("live conv"),
            source: wgpu::ShaderSource::Wgsl(CONV.into()),
        });
        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |fs, f: wgpu::TextureFormat| {
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fs),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &m,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &m,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(f.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        use wgpu::TextureFormat as T;
        Conv {
            pipe: [
                pipe("rgb", T::Bgra8Unorm),
                pipe("y", T::R8Unorm),
                pipe("uv", T::Rg8Unorm),
            ],
            bgl,
        }
    }

    /// Records `src` converted into `dst`: BGRA (one target) or NV12 (the luma and chroma plane views).
    fn encode(
        &self,
        g: &Gpu,
        e: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &[wgpu::TextureView],
    ) {
        let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            }],
        });
        let k = if dst.len() == 2 { 1 } else { 0 };
        for (i, v) in dst.iter().enumerate() {
            let mut p = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: v,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            p.set_pipeline(&self.pipe[k + i]);
            p.set_bind_group(0, &bg, &[]);
            p.draw(0..3, 0..1);
        }
    }
}

/// The PipeWire camera: the node (announcing `fps`), its mode's scratch picture and the views of the buffers' images.
struct Cam {
    cam: pw::Cam,
    fps: u32,
    mode: Option<pw::Mode>,
    /// the picture the conversion reads, at the mode's size
    pic: Option<wgpu::TextureView>,
    views: Vec<(Arc<live::dmabuf::Image>, Vec<wgpu::TextureView>)>,
    sent: u64,
}

/// The egui preview of what goes out.
struct Pv {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    id: egui::TextureId,
    rs: RenderState,
}

impl Drop for Pv {
    fn drop(&mut self) {
        self.rs.renderer.write().free_texture(&self.id);
    }
}

/// The outputs' state for the Live screen (and the tests), as of the output thread's last pass.
#[derive(Clone, Default)]
struct Stat {
    name: Option<Arc<str>>,
    /// the source's audio rate
    rate: Option<u32>,
    /// the frame clock: frames made, made more than a frame period late, the latest (ns); ticks without a picture
    /// (the thread stalled, the source had none: a seek)
    made: (u64, u64, u64),
    missed: u64,
    rtmp: Option<Rs>,
    obs: Option<[Os; 2]>,
    cam: Option<Cs>,
    /// why RTMP, OBS or the camera failed to start (the camera: or a frame)
    err: [Option<String>; 3],
}

/// An RTMP stream: the publisher's state, kbit/s, frames handed to the encoder, left out (encoder behind), ticks
/// that repeated the last frame, when it started (CLOCK_MONOTONIC ns), the audio rate, the encoder's error.
#[derive(Clone)]
struct Rs {
    st: rtmp::Stats,
    kbps: f64,
    f: u64,
    skip: u64,
    rep: u64,
    t0: u64,
    rate: u32,
    err: Option<String>,
}

/// An OBS source: watched, its ring's modifier, frames sent, skipped (all buffers held), missed by its clients (each
/// holds `obs::HOLD`), the ring's error.
#[derive(Clone, Default)]
struct Os {
    on: bool,
    m: Option<u64>,
    sent: u64,
    skip: u64,
    held: u64,
    err: Option<String>,
}

/// The PipeWire camera: its node, the consumer's mode, the rate it announces, frames sent, the stream's error.
#[derive(Clone)]
struct Cs {
    node: Option<u32>,
    mode: Option<pw::Mode>,
    fps: u32,
    sent: u64,
    err: Option<String>,
}

/// What the Live screen and the output thread share. Each side holds the lock only to copy in or out, never while
/// it works: the thread waits for the GPU and PipeWire, the screen for the compositor.
#[derive(Default)]
struct Sh {
    /// the settings, which the thread follows; a stream start or stop asked for
    set: Set,
    go: bool,
    src: Option<Weak<Mutex<dyn Source + Send>>>,
    grab: Option<PathBuf>,
    /// UI frames so far; when the Live screen was drawn last, and its preview (view, size, 360) that renders while
    /// it shows
    frames: u64,
    seen: Option<Instant>,
    pv: Option<(wgpu::TextureView, [u32; 2], bool)>,
    st: Stat,
}

/// The Live screen's handle on the outputs. Their thread owns them ([`Outs`]); the screen edits its own copy of the
/// settings and hands it over, and shows their state.
pub struct Live {
    sh: Arc<Mutex<Sh>>,
    quit: Arc<AtomicBool>,
    th: Option<thread::JoinHandle<()>>,
    g: Arc<Gpu>,
    set: Set,
    /// where settings go; None under the test hook
    file: Option<PathBuf>,
    /// the OBS socket
    sock: Option<PathBuf>,
    show_key: bool,
    pv: Option<Pv>,
    /// test hook: save the next frame's `view` and `360` pictures as `<stem>-view.png`, `<stem>-360.png`
    pub grab: Option<PathBuf>,
}

impl Live {
    pub fn new(g: &Arc<Gpu>, cx: &stitch::Ctx, ctx: &egui::Context, hook: Option<Hook>) -> Live {
        // the test hook neither reads nor writes the user's settings (their stream key)
        let file = hook.is_none().then(Set::file).flatten();
        let set = match &hook {
            Some(h) => Set {
                url: h.rtmp.clone().unwrap_or_default(),
                res: h.res.unwrap_or(0).min(RES.len() - 1),
                fps: h.fps.filter(|f| FPS.contains(f)).unwrap_or(30),
                kbps: h.kbps.unwrap_or(6000),
                obs: h.obs,
                cam: h.cam,
                ..Set::default()
            },
            None => file
                .as_ref()
                .and_then(|f| std::fs::read_to_string(f).ok())
                .map_or_else(Set::default, |t| Set::parse(&t)),
        };
        let sock = hook
            .as_ref()
            .and_then(|h| h.sock.clone())
            .or_else(|| obs::path().ok());
        let sh = Arc::new(Mutex::new(Sh {
            set: set.clone(),
            ..Default::default()
        }));
        let mut o = Outs::new(g, cx, ctx, &sh, set.clone(), sock.clone(), hook);
        o.publish();
        let quit = Arc::new(AtomicBool::new(false));
        let q = quit.clone();
        let th = thread::Builder::new()
            .name("live".into())
            .spawn(move || {
                while !q.load(Relaxed) {
                    // a surface configure the UI retries waits for this pass, the next pass for it
                    let d = {
                        let _g = crate::GATE.read();
                        o.step()
                    };
                    thread::park_timeout(d);
                }
            })
            .expect("live thread");
        Live {
            sh,
            quit,
            th: Some(th),
            g: g.clone(),
            set,
            file,
            sock,
            show_key: false,
            pv: None,
            grab: None,
        }
    }

    /// Every UI frame: the source the outputs show (the open player), the test hook's grab.
    pub fn tick(&mut self, src: Option<Src>) {
        let mut s = self.sh.lock().unwrap();
        s.frames += 1;
        s.src = src.as_ref().map(Arc::downgrade);
        if let Some(f) = self.grab.take() {
            s.grab = Some(f);
        }
    }

    /// The Live screen: the settings, the preview, the outputs' state.
    pub fn ui(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        let st = {
            let mut s = self.sh.lock().unwrap();
            s.seen = Some(Instant::now());
            s.st.clone()
        };
        let (was, mut go) = (self.set.clone(), false);
        ui.heading("Live");
        match &st.name {
            Some(n) => ui.label(format!(
                "Source: the player, {n} (its view, or the 360° picture)"
            )),
            None => ui.label("Source: nothing playing. Open a video in the Library."),
        };
        ui.add_space(6.0);
        // the picture the RTMP resolution asks for, smaller
        let (_, pw, ph, pano) = RES[self.set.res];
        let size = [960, 960 * ph / pw];
        let new = self
            .pv
            .as_ref()
            .is_none_or(|p| [p.tex.width(), p.tex.height()] != size);
        if new {
            let t = tex(&self.g, size[0], size[1], wgpu::TextureFormat::Rgba8Unorm);
            let v = t.create_view(&Default::default());
            let id = rs.renderer.write().register_native_texture(
                &self.g.device,
                &v,
                wgpu::FilterMode::Linear,
            );
            self.pv = Some(Pv {
                tex: t,
                view: v,
                id,
                rs: rs.clone(),
            });
        }
        if let Some(p) = &self.pv {
            let w = ui.available_width().min(640.0);
            ui.image(egui::load::SizedTexture::new(
                p.id,
                egui::vec2(w, w * p.tex.height() as f32 / p.tex.width() as f32),
            ));
            ui.label(if pano {
                "360° equirect"
            } else {
                "Reframed view"
            });
        }
        if st.made.0 > 0 {
            ui.label(format!(
                "Output clock: {} frames made, {} of them over a frame late (at most {:.0} ms); {} ticks missed (the \
                 output thread stalled, or the source had no picture: a seek)",
                st.made.0,
                st.made.1,
                st.made.2 as f64 / 1e6,
                st.missed
            ));
        }
        ui.add_space(8.0);
        ui.separator();
        ui.heading("RTMP");
        let streaming = st.rtmp.is_some();
        ui.add_enabled_ui(!streaming, |ui| {
            egui::Grid::new("rtmp").num_columns(2).show(ui, |ui| {
                ui.label("Server URL");
                ui.add(
                    egui::TextEdit::singleline(&mut self.set.url)
                        .hint_text("rtmp://host/app")
                        .desired_width(360.0),
                );
                ui.end_row();
                ui.label("Stream key");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.set.key)
                            .password(!self.show_key)
                            .hint_text("empty: the URL's last segment")
                            .desired_width(300.0),
                    );
                    ui.toggle_value(&mut self.show_key, "Show");
                });
                ui.end_row();
                ui.label("Resolution");
                egui::ComboBox::from_id_salt("res")
                    .selected_text(RES[self.set.res].0)
                    .show_ui(ui, |ui| {
                        for (i, r) in RES.iter().enumerate() {
                            ui.selectable_value(&mut self.set.res, i, r.0);
                        }
                    });
                ui.end_row();
                ui.label("Frame rate");
                egui::ComboBox::from_id_salt("fps")
                    .selected_text(format!("{} fps", self.set.fps))
                    .show_ui(ui, |ui| {
                        for f in FPS {
                            ui.selectable_value(&mut self.set.fps, f, format!("{f} fps"));
                        }
                    })
                    .response
                    .on_hover_text("Every output's rate: RTMP, OBS and the PipeWire camera");
                ui.end_row();
                ui.label("Bitrate");
                ui.add(
                    egui::DragValue::new(&mut self.set.kbps)
                        .range(500..=50000)
                        .speed(50)
                        .suffix(" kbit/s"),
                );
                ui.end_row();
            });
        });
        go |= ui
            .button(if streaming {
                "Stop streaming"
            } else {
                "Start streaming"
            })
            .clicked();
        if let Some(r) = &st.rtmp {
            let s = &r.st;
            let up = r.t0.abs_diff(mono()) / 1_000_000_000;
            ui.monospace(format!(
                "{:?} {}:{:02}:{:02} · {:.0} kbit/s · {} frames to the encoder · sent {} video, {} audio · dropped {} \
                 frames ({} network, {} encoder) · repeated {} (no picture: a seek) · reconnects {}",
                s.state,
                up / 3600,
                up / 60 % 60,
                up % 60,
                r.kbps,
                r.f,
                s.sent[0],
                s.sent[1],
                s.drop[0] + r.skip,
                s.drop[0],
                r.skip,
                r.rep,
                s.reconnects
            ));
            for e in [&s.err, &r.err].into_iter().flatten() {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
            if let Some(a) = st.rate.filter(|&a| a != r.rate) {
                ui.label(format!(
                    "The source's audio is {a} Hz, the stream's {} Hz: silence until the stream restarts.",
                    r.rate
                ));
            }
        }
        if let Some(e) = &st.err[0] {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        ui.add_space(8.0);
        ui.separator();
        ui.heading("OBS");
        ui.checkbox(&mut self.set.obs, "Serve the x5desk OBS source")
            .on_hover_text("OBS: add an \"X5 Camera (x5desk)\" source (plugin x5desk-obs.so)");
        if let Some(os) = &st.obs {
            if let Some(p) = &self.sock {
                ui.label(format!("At {}", p.display()));
            }
            for ((_, name, w, h, _), o) in OBS.iter().zip(os) {
                let m = o.m.map_or(String::new(), |m| format!(", modifier {m:#x}"));
                ui.monospace(format!(
                    "{name:>4} ({w} × {h}): {}{m} · sent {} · skipped {} (all buffers held) · missed {} (a client \
                     holds {} frames)",
                    if o.on { "watched" } else { "not watched" },
                    o.sent,
                    o.skip,
                    o.held,
                    obs::HOLD
                ));
                if let Some(e) = &o.err {
                    ui.colored_label(ui.visuals().error_fg_color, e);
                }
            }
        }
        if let Some(e) = &st.err[1] {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        ui.add_space(8.0);
        ui.separator();
        ui.heading("PipeWire camera");
        ui.checkbox(&mut self.set.cam, format!("Publish \"{CAM}\""))
            .on_hover_text(
                "A PipeWire video source: 1920 × 1080 reframed view or 3840 × 1920 equirect",
            );
        if let Some(c) = &st.cam {
            let node = c.node.map_or("registering".into(), |n| format!("node {n}"));
            let m = match c.mode {
                Some(m) => format!(
                    "streaming {:?} {} × {}{} · sent {}",
                    m.fmt,
                    m.w,
                    m.h,
                    m.modifier
                        .map_or(" (MemFd)".into(), |x| format!(", modifier {x:#x}")),
                    c.sent
                ),
                None => "no consumer".into(),
            };
            ui.monospace(format!("{node} · {} fps · {m}", c.fps));
            if let Some(e) = &c.err {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
        }
        if let Some(e) = &st.err[2] {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        if self.set != was
            && let Some(f) = &self.file
            && let Err(e) = self.set.save(f)
        {
            eprintln!("{}: {e}", f.display());
        }
        if self.set != was || go || new {
            let mut s = self.sh.lock().unwrap();
            s.set.clone_from(&self.set);
            s.go ^= go;
            s.pv = self.pv.as_ref().map(|p| (p.view.clone(), size, pano));
            drop(s);
            if let Some(t) = &self.th {
                t.thread().unpark();
            }
        }
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        self.quit.store(true, Relaxed);
        if let Some(t) = self.th.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

/// The outputs and their frame clock, owned by the output thread ([`Outs::step`]).
struct Outs {
    g: Arc<Gpu>,
    cx: stitch::Ctx,
    ctx: egui::Context,
    sh: Arc<Mutex<Sh>>,
    /// the settings applied
    set: Set,
    sock: Option<PathBuf>,
    src: Option<Weak<Mutex<dyn Source + Send>>>,
    asc: Option<Vec<u8>>,
    /// the frame clock: the next tick (CLOCK_MONOTONIC ns), 0 = stopped
    next: u64,
    tx: Option<Tx>,
    obs: Option<(Server, [Out; 2])>,
    dec: Option<Aac>,
    cam: Option<Cam>,
    conv: Option<Conv>,
    /// the Live screen shows (drawn within `SEEN`): its preview renders
    vis: bool,
    pv: Option<(wgpu::TextureView, [u32; 2], bool)>,
    /// the test hook's stream starts once the source shows a picture (its audio config is known then); it had one
    /// at the last pass
    pending: bool,
    ready: bool,
    grab: Option<PathBuf>,
    /// the audio heard since the last pass: per packet its start (ns) and end in `ab`; their bytes
    ai: Vec<(u64, usize)>,
    ab: Vec<u8>,
    st: Stat,
    /// test hook: the state on stderr every 5 s (the window may not draw); UI frames so far
    hook: bool,
    told: Instant,
    frames: u64,
}

impl Outs {
    fn new(
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        ctx: &egui::Context,
        sh: &Arc<Mutex<Sh>>,
        set: Set,
        sock: Option<PathBuf>,
        hook: Option<Hook>,
    ) -> Outs {
        let mut l = Outs {
            g: g.clone(),
            cx: cx.clone(),
            ctx: ctx.clone(),
            sh: sh.clone(),
            set,
            sock,
            src: None,
            asc: None,
            next: 0,
            tx: None,
            obs: None,
            dec: None,
            cam: None,
            conv: None,
            vis: false,
            pv: None,
            pending: hook.as_ref().is_some_and(|h| h.rtmp.is_some()),
            ready: false,
            grab: None,
            ai: vec![],
            ab: vec![],
            st: Stat::default(),
            hook: hook.is_some(),
            told: Instant::now() + Duration::from_secs(5),
            frames: 0,
        };
        l.obs(l.set.obs);
        l.cam(l.set.cam);
        l
    }

    fn obs(&mut self, on: bool) {
        self.st.err[1] = None;
        if !on {
            self.obs = None;
            return;
        }
        let sock = self
            .sock
            .as_deref()
            .ok_or(std::io::ErrorKind::NotFound.into());
        match sock.and_then(Server::bind) {
            Ok(mut s) => {
                for (id, name, ..) in OBS {
                    s.add(id, name);
                }
                self.obs = Some((s, Default::default()));
            }
            Err(e) => self.st.err[1] = Some(format!("OBS socket: {e}")),
        }
    }

    fn cam(&mut self, on: bool) {
        self.st.err[2] = None;
        // the old node goes first: the new one takes its name
        self.cam = None;
        if !on {
            return;
        }
        match pw::Cam::new(self.g.clone(), CAM, &pw::SIZES, self.set.fps) {
            Ok(cam) => {
                self.cam = Some(Cam {
                    cam,
                    fps: self.set.fps,
                    mode: None,
                    pic: None,
                    views: vec![],
                    sent: 0,
                })
            }
            Err(e) => self.st.err[2] = Some(format!("PipeWire: {e}")),
        }
    }

    /// Tells OBS clients each exported source's format again (its audio or rate changed).
    fn formats(&mut self) {
        let Some((s, outs)) = &mut self.obs else {
            return;
        };
        for ((id, _, w, h, _), o) in OBS.iter().zip(outs.iter()) {
            if let Some(r) = &o.ring {
                let m = r.0.imgs[0].modifier;
                s.format(*id, format(m, *w, *h, self.set.fps, self.asc.as_deref()));
            }
        }
    }

    fn start(&mut self) -> Result<Tx, String> {
        let s = &self.set;
        let (_, w, h, pano) = RES[s.res];
        if s.url.trim().is_empty() {
            return Err("no RTMP URL".into());
        }
        let c = video::Cfg {
            codec: video::Codec::H264,
            deep: false,
            w,
            h,
            fps: [s.fps, 1],
            gop: 2 * s.fps,
            rate: Rate::Cbr(s.kbps as u64 * 1000),
            vbv: 100,
            quality: 0,
            full: false,
            chroma: 0,
        };
        let e = Encoder::new(&self.g, &c).map_err(|e| format!("H.264 encoder: {e}"))?;
        let [sps, pps] = &e.params[..] else {
            return Err("H.264 encoder: not one SPS and one PPS".into());
        };
        // the source's rate if the encoder has it, else 48 kHz; always stereo
        let rate = self
            .asc
            .as_deref()
            .and_then(|a| media::mp4::asc_info(a).ok())
            .map_or(48000, |a| if a.0 == 44100 { a.0 } else { 48000 });
        let enc =
            aacenc::Enc::new(rate, 2, AKBPS * 1000).map_err(|e| format!("AAC encoder: {e}"))?;
        let meta = flv::Meta {
            w,
            h,
            fps: s.fps as f64,
            vkbps: s.kbps as f64,
            rate,
            ch: 2,
            akbps: AKBPS as f64,
        };
        let cfg = rtmp::Cfg {
            meta,
            avcc: flv::avcc(sps, pps),
            asc: enc.asc().to_vec(),
            lag: 1000,
        };
        let r = Arc::new(
            Rtmp::start(s.url.trim(), s.key.trim(), cfg).map_err(|e| format!("RTMP: {e}"))?,
        );
        let pic = tex(&self.g, w, h, wgpu::TextureFormat::Rgba8Unorm);
        let pv = pic.create_view(&Default::default());
        let yuv = stitch::Yuv::new(
            &self.cx,
            &pv,
            stitch::Fmt {
                deep: false,
                chroma: 0,
            },
            pano,
        )?;
        let (tx, rx) = mpsc::channel();
        let (done, err) = (Arc::new(AtomicU64::new(0)), Arc::new(Mutex::new(None)));
        let input = e.input.clone();
        let (g, r2, d2, e2) = (self.g.clone(), r.clone(), done.clone(), err.clone());
        let th = thread::Builder::new()
            .name("rtmp encode".into())
            .spawn(move || encode(g, e, rx, r2, d2, e2))
            .map_err(|e| e.to_string())?;
        Ok(Tx {
            rtmp: r,
            input,
            tx: Some(tx),
            th: Some(th),
            w,
            h,
            pano,
            pv,
            yuv,
            f: 0,
            done,
            skip: 0,
            rep: 0,
            // the tick of its first frame, so the session starts with picture and sound (the publisher drops what
            // comes before its first key frame)
            t0: match self.next {
                0 => mono(),
                n => n,
            },
            enc,
            rate,
            fed: 0,
            aud: 0,
            buf: vec![],
            cont: false,
            state: State::Connecting,
            meter: (0, Instant::now(), 0.0),
            err,
            st: Default::default(),
        })
    }

    /// Takes the screen's changes: the source, the preview, the test hook's grab, the settings (applied), a stream
    /// start or stop.
    fn sync(&mut self) {
        let (set, go) = {
            let mut s = self.sh.lock().unwrap();
            self.src.clone_from(&s.src);
            self.pv.clone_from(&s.pv);
            self.frames = s.frames;
            self.vis = s.seen.is_some_and(|t| t.elapsed() < SEEN);
            if let Some(f) = s.grab.take() {
                self.grab = Some(f);
            }
            let set = (s.set != self.set).then(|| s.set.clone());
            (set, std::mem::take(&mut s.go))
        };
        if let Some(n) = set {
            let o = std::mem::replace(&mut self.set, n);
            if o.obs != self.set.obs {
                self.obs(self.set.obs);
            }
            // the camera announces its rate when it registers, OBS gets it in each source's format
            if o.cam != self.set.cam || (o.fps != self.set.fps && self.cam.is_some()) {
                self.cam(self.set.cam);
            }
            if o.fps != self.set.fps {
                self.formats();
            }
        }
        if go {
            self.toggle_rtmp();
        }
    }

    /// One pass of the output thread: the screen's changes, OBS clients, RTMP state, the source, at a tick of the
    /// frame clock a frame for every output, the audio heard since the last pass, the state for the screen. Returns
    /// how long to wait for the next one.
    ///
    /// The source stays locked only while it is asked (the UI locks it every frame): for its name and audio and, at
    /// a tick, to draw its next picture into the outputs' targets, all made ready before (DMA-BUFs acquired, a
    /// PipeWire buffer dequeued). What follows the draws runs after: conversion, encoding, sending.
    fn step(&mut self) -> Duration {
        self.sync();
        if self.pending && self.ready {
            self.pending = false;
            self.toggle_rtmp();
        }
        let g = &*self.g;
        let mut watched = false;
        if let Some((s, outs)) = &mut self.obs {
            let changed = s.poll();
            for ((id, _, w, h, _), o) in OBS.iter().zip(outs.iter_mut()) {
                o.s.on = s.watched(*id);
                // clients that still read its buffers hold their DMA-BUFs
                if !o.s.on {
                    (o.ring, o.s.m) = (None, None);
                    continue;
                }
                watched = true;
                let m = s.modifiers(*id, Fmt::Rgba.drm());
                if o.ring.as_ref().is_none_or(|r| changed && r.2 != m) {
                    o.ring = None;
                    // a client holding obs::HOLD leaves one to render into
                    match Ring::new(g, Fmt::Rgba, *w, *h, RING, &m, obs::HOLD + 1) {
                        Ok(r) => {
                            let v = r
                                .imgs
                                .iter()
                                .map(|i| i.tex.create_view(&Default::default()))
                                .collect();
                            let x = r.imgs[0].modifier;
                            s.format(*id, format(x, *w, *h, self.set.fps, self.asc.as_deref()));
                            (o.ring, o.s.m, o.s.err) = (Some((r, v, m)), Some(x), None);
                        }
                        Err(e) => o.s.err = Some(e.to_string()),
                    }
                }
            }
        }
        if let Some(t) = &mut self.tx {
            t.st = t.rtmp.stats();
            // a new session starts at a key frame: make one now instead of waiting for the GOP's. The first one has
            // frame 0 unless the publisher dropped video while connecting
            let lost = t.st.reconnects > 0 || t.st.drop[0] > 0;
            if t.st.state == State::Live && t.state != State::Live && lost {
                t.input.key();
            }
            t.state = t.st.state;
            let dt = t.meter.1.elapsed().as_secs_f64();
            if dt >= 1.0 {
                t.meter = (
                    t.st.bytes,
                    Instant::now(),
                    (t.st.bytes - t.meter.0) as f64 * 8e-3 / dt,
                );
            }
        }
        let on = self.tx.is_some() || watched || self.cam.is_some() || self.vis;
        let (now, period) = (mono(), 1_000_000_000 / self.set.fps as u64);
        if !on {
            self.next = 0;
        } else if now > self.next + 4 * period {
            // a stall of more than a few frames skips ahead on the grid instead of catching up
            let k = (now - self.next) / period;
            self.st.missed += (self.next > 0) as u64 * k;
            self.next = match self.next {
                0 => now,
                n => n + k * period,
            };
        }
        let due = (on && now >= self.next).then(|| {
            self.next += period;
            self.next - period
        });
        let aud = self.tx.is_some() || watched;
        let src = self.src.as_ref().and_then(Weak::upgrade);
        let Outs {
            tx,
            obs,
            cam,
            conv,
            vis,
            pv,
            grab,
            ready: was,
            asc,
            ai,
            ab,
            st,
            ..
        } = self;
        // the targets (view, size, 360): RTMP, OBS 360 and view, the camera, the preview, the test hook's grabs
        let mut tg: [Option<(wgpu::TextureView, [u32; 2], bool)>; 7] = Default::default();
        let (mut rt, mut ob, mut cf) = (None, [None; 2], None);
        if due.is_some() {
            if let Some(t) = tx.as_mut() {
                match t.f - t.done.load(Relaxed) < video::encode::SLOTS as u64 {
                    true => (tg[0], rt) = (Some((t.pv.clone(), [t.w, t.h], t.pano)), Some(t)),
                    false => t.skip += 1,
                }
            }
            // DMA-BUFs and PipeWire buffers only while the source has a picture (it had one at the last pass): without
            // one they go back unused, and their acquire submits would only keep the queue busy
            if let Some((s, outs)) = obs.as_mut().filter(|_| *was) {
                for (k, ((id, _, w, h, pano), o)) in OBS.iter().zip(outs.iter_mut()).enumerate() {
                    let Some((r, v, _)) = &mut o.ring else {
                        continue;
                    };
                    let Some(i) = r.next(|i| s.busy(*id, i as u32)) else {
                        o.s.skip += 1;
                        continue;
                    };
                    match r.imgs[i].acquire() {
                        Ok(()) => {
                            (ob[k], tg[1 + k]) = (Some(i), Some((v[i].clone(), [*w, *h], *pano)))
                        }
                        Err(e) => o.s.err = Some(e.to_string()),
                    }
                }
            }
            if let Some(Cam {
                cam: c,
                mode,
                pic,
                views,
                sent,
                ..
            }) = cam.as_mut().filter(|_| *was)
            {
                match c.dequeue() {
                    Ok(Some(fr)) => {
                        let m = fr.mode;
                        if *mode != Some(m) {
                            (*mode, *pic) = (Some(m), None);
                        }
                        // the images of buffers PipeWire dropped (a new format, a new consumer)
                        views.retain(|v| Arc::strong_count(&v.0) > 1);
                        let k = match views.iter().position(|v| Arc::ptr_eq(&v.0, &fr.img)) {
                            Some(k) => k,
                            None => {
                                use wgpu::TextureAspect as A;
                                let t = &fr.img.tex;
                                let v = match m.fmt {
                                    Fmt::Nv12 => vec![view(t, A::Plane0), view(t, A::Plane1)],
                                    _ => vec![view(t, A::All)],
                                };
                                views.push((fr.img.clone(), v));
                                views.len() - 1
                            }
                        };
                        // RGBA straight into the buffer's image, the others through the conversion
                        let t = match m.fmt {
                            Fmt::Rgba => views[k].1[0].clone(),
                            _ => pic
                                .get_or_insert_with(|| {
                                    tex(g, m.w, m.h, wgpu::TextureFormat::Rgba8Unorm)
                                        .create_view(&Default::default())
                                })
                                .clone(),
                        };
                        tg[3] = Some((t, [m.w, m.h], m.w == 2 * m.h));
                        cf = Some((fr, k, views, pic, sent));
                    }
                    Ok(None) => {}
                    Err(e) => st.err[2] = Some(format!("PipeWire frame: {e}")),
                }
            }
            if *vis {
                tg[4] = pv.clone();
            }
        }
        let gt = (grab.is_some() && *was)
            .then(|| OBS.map(|(.., w, h, _)| tex(g, w, h, wgpu::TextureFormat::Rgba8Unorm)));
        for (k, t) in gt.iter().flatten().enumerate() {
            let v = t.create_view(&Default::default());
            tg[5 + k] = Some((v, [t.width(), t.height()], OBS[k].4));
        }
        let (mut ready, mut ac) = (false, false);
        match &src {
            None => st.name = None,
            Some(m) => {
                let mut s = m.lock().unwrap();
                let n = s.name();
                if st.name.as_deref() != Some(n) {
                    st.name = Some(n.into());
                }
                let a = s.asc();
                if a != asc.as_deref() {
                    *asc = a.map(<[u8]>::to_vec);
                    ac = true;
                }
                if due.is_some() {
                    s.tick();
                }
                ready = s.ready();
                for (v, size, pano) in tg.iter().flatten().filter(|_| ready) {
                    s.draw(v, *size, *pano);
                }
                if aud {
                    s.audio(&mut |ns, p| {
                        ab.extend_from_slice(p);
                        ai.push((ns, ab.len()));
                    });
                }
            }
        }
        // the last reference to a player the UI closed may be this one
        drop(src);
        if let Some(d) = due {
            match ready {
                true => {
                    let (l, m) = (now - d, &mut st.made);
                    *m = (m.0 + 1, m.1 + (l > period) as u64, m.2.max(l));
                }
                false => st.missed += 1,
            }
            // without a picture (a seek) RTMP repeats its last frame, the others keep showing theirs
            if let Some(t) = rt.filter(|t| ready || t.f > 0) {
                let mut e = g.device.create_command_encoder(&Default::default());
                // the converter's buffer still holds the last frame
                if ready {
                    t.yuv.encode(&mut e);
                }
                let sent = t.input.acquire(g, t.f).and_then(|_| {
                    let cb = t.input.copy_buffer(g, &t.yuv.buf, 0, t.yuv.pitch, t.f)?;
                    g.queue.submit([e.finish(), cb]);
                    t.input.release(g, t.f);
                    Ok(())
                });
                match sent {
                    Ok(()) => {
                        let ms = (d.saturating_sub(t.t0) / 1_000_000) as u32;
                        if t.tx.as_ref().is_some_and(|x| x.send(ms).is_ok()) {
                            t.f += 1;
                            t.rep += !ready as u64;
                        }
                    }
                    Err(video::Error::Full) => t.skip += 1,
                    Err(e) => *t.err.lock().unwrap() = Some(format!("encoder input: {e}")),
                }
            }
            if let Some((s, outs)) = obs.as_mut().filter(|_| ready) {
                for (k, ((id, _, w, h, _), o)) in OBS.iter().zip(outs.iter_mut()).enumerate() {
                    let (Some(i), Some((r, ..))) = (ob[k], &o.ring) else {
                        continue;
                    };
                    let img = &r.imgs[i];
                    let sf = match img.release() {
                        Ok(f) => f,
                        Err(e) => {
                            o.s.err = Some(e.to_string());
                            continue;
                        }
                    };
                    let mut f = Frame {
                        ts: d,
                        modifier: img.modifier,
                        w: *w,
                        h: *h,
                        fourcc: Fmt::Rgba.drm(),
                        buf: i as u32,
                        ..Default::default()
                    };
                    for (k, p) in img.planes.iter().enumerate() {
                        (f.off[k], f.stride[k]) = (p.off, p.stride);
                    }
                    let fd = [img.fd.as_fd(); 4];
                    let c = s.clients(*id);
                    let n = s.frame(*id, &f, &fd[..img.planes.len()], Some(sf.as_fd()));
                    o.s.sent += (n > 0) as u64;
                    o.s.held += c.saturating_sub(n) as u64;
                }
            }
        }
        // unless drawn, the frame goes back to PipeWire unused
        if let (Some((fr, k, views, pic, sent)), Some(d)) = (cf.filter(|_| ready), due) {
            if let Some(p) = pic.as_ref().filter(|_| fr.mode.fmt != Fmt::Rgba) {
                let cv = conv.get_or_insert_with(|| Conv::new(g));
                let mut e = g.device.create_command_encoder(&Default::default());
                cv.encode(g, &mut e, p, &views[k].1);
                g.queue.submit([e.finish()]);
            }
            match fr.queue(d) {
                Ok(()) => *sent += 1,
                Err(e) => st.err[2] = Some(format!("PipeWire frame: {e}")),
            }
        }
        if let Some(ts) = gt.filter(|_| ready)
            && let Some(f) = grab.take()
        {
            let stem = f.file_stem().unwrap_or_default().to_string_lossy();
            for (t, (_, name, w, h, _)) in ts.iter().zip(OBS) {
                let px = stitch::read_texture(&g.device, &g.queue, t);
                let p = f.with_file_name(format!("{stem}-{name}.png"));
                match crate::png::write(&p, w, h, &px) {
                    Ok(()) => eprintln!("x5desk: {name} picture → {}", p.display()),
                    Err(e) => eprintln!("x5desk: {}: {e}", p.display()),
                }
            }
        }
        if ac {
            let a = self.asc.as_deref();
            self.st.rate = a.and_then(|a| media::mp4::asc_info(a).ok()).map(|a| a.0);
            self.dec = None;
            self.formats();
        }
        self.ready = ready;
        if aud {
            self.audio(now);
        }
        if due.is_some() && self.vis {
            self.ctx.request_repaint();
        }
        self.publish();
        self.tell();
        match on {
            true => Duration::from_nanos(self.next.saturating_sub(mono())),
            false => IDLE,
        }
    }

    /// The audio heard since the last pass (`ai`, `ab`), decoded once: to the RTMP stream on its timeline (silence
    /// where none came, at `now`), f32 to OBS.
    fn audio(&mut self, now: u64) {
        let Outs {
            tx,
            obs,
            dec,
            asc,
            ai,
            ab,
            ..
        } = self;
        let mut ob = obs
            .as_mut()
            .filter(|(s, _)| OBS.iter().any(|o| s.watched(o.0)));
        if dec.is_none() {
            *dec = asc.as_deref().and_then(|a| Aac::new(a).ok());
        }
        let mut rt = tx.as_mut();
        if let Some(d) = dec.as_mut() {
            let mut at = 0;
            for &(ns, end) in ai.iter() {
                let (pkt, rate, ch) = (&ab[at..end], d.rate, d.ch as u32);
                at = end;
                let Ok(x) = d.decode(pkt) else { continue };
                if let Some(t) = rt.as_deref_mut().filter(|t| t.rate == rate) {
                    t.pcm(ns, x, ch as usize);
                }
                if let Some((s, _)) = ob.as_deref_mut() {
                    for (id, ..) in OBS {
                        if s.watched(id) {
                            s.audio(id, ns, rate, ch, x);
                        }
                    }
                }
            }
        }
        ai.clear();
        ab.clear();
        if let Some(t) = rt {
            t.quiet(now);
        }
    }

    fn toggle_rtmp(&mut self) {
        self.st.err[0] = None;
        match self.tx.take() {
            // the publisher may take rtmp::STOP to send what's queued: not on this thread
            Some(t) => drop(thread::spawn(move || drop(t))),
            None => match self.start() {
                Ok(t) => self.tx = Some(t),
                Err(e) => self.st.err[0] = Some(e),
            },
        }
    }

    /// Hands the screen the outputs' state.
    fn publish(&mut self) {
        let st = &mut self.st;
        st.rtmp = self.tx.as_ref().map(|t| Rs {
            st: t.st.clone(),
            kbps: t.meter.2,
            f: t.f,
            skip: t.skip,
            rep: t.rep,
            t0: t.t0,
            rate: t.rate,
            err: t.err.lock().unwrap().clone(),
        });
        st.obs = self
            .obs
            .as_ref()
            .map(|(_, o)| [o[0].s.clone(), o[1].s.clone()]);
        st.cam = self.cam.as_ref().map(|c| Cs {
            node: c.cam.node(),
            mode: c.cam.mode(),
            fps: c.fps,
            sent: c.sent,
            err: c.cam.error(),
        });
        let st = st.clone();
        self.sh.lock().unwrap().st = st;
    }

    /// Test hook: the outputs' state on stderr every 5 s.
    fn tell(&mut self) {
        if !self.hook || Instant::now() < self.told {
            return;
        }
        self.told += Duration::from_secs(5);
        let m = &self.st.made;
        let mut s = format!(
            "x5desk live: {} UI frames; clock {} made, {} late (max {:.0} ms), {} missed",
            self.frames,
            m.0,
            m.1,
            m.2 as f64 / 1e6,
            self.st.missed
        );
        if let Some(t) = &self.tx {
            let st = &t.st;
            let _ = write!(
                s,
                "; RTMP {:?} sent {} video {} audio, {} repeated, dropped {} network {} encoder, reconnects {}",
                st.state, st.sent[0], st.sent[1], t.rep, st.drop[0], t.skip, st.reconnects
            );
        }
        if let Some((_, outs)) = &self.obs {
            for ((_, name, ..), o) in OBS.iter().zip(outs) {
                let _ = write!(
                    s,
                    "; OBS {name} sent {} skipped {} missed {}",
                    o.s.sent, o.s.skip, o.s.held
                );
            }
        }
        if let Some(c) = &self.cam {
            let _ = write!(s, "; camera sent {}", c.sent);
        }
        eprintln!("{s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    #[test]
    fn settings() {
        let s = Set {
            url: "rtmp://a.example/live".into(),
            key: "abc\ndef".into(),
            res: 2,
            fps: 60,
            kbps: 9000,
            obs: false,
            cam: true,
        };
        let t = Set::parse(&s.text());
        assert_eq!(
            t,
            Set {
                key: "abcdef".into(),
                ..s.clone()
            }
        );
        // out of range values fall back
        let t = Set::parse("res=9\nfps=31\nkbps=1\nobs=0\n");
        assert_eq!(
            (t.res, t.fps, t.kbps, t.obs, t.cam),
            (2, 30, 500, false, false)
        );
        let f = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-live-{}", std::process::id()));
        // an existing file with looser permissions ends up owner-only too
        std::fs::write(&f, "").unwrap();
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
        s.save(&f).unwrap();
        assert_eq!(
            std::fs::metadata(&f).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            Set::parse(&std::fs::read_to_string(&f).unwrap()).key,
            "abcdef"
        );
        std::fs::remove_file(&f).unwrap();
    }

    /// Every size and rate the screen offers gets an H.264 encoder (level, size and bitrate limits).
    #[test]
    fn encoders() {
        let g = Gpu::new().unwrap();
        let mut bad = vec![];
        for (name, w, h, _) in RES {
            for fps in FPS {
                let c = video::Cfg {
                    codec: video::Codec::H264,
                    deep: false,
                    w,
                    h,
                    fps: [fps, 1],
                    gop: 2 * fps,
                    rate: Rate::Cbr(50_000_000),
                    vbv: 100,
                    quality: 0,
                    full: false,
                    chroma: 0,
                };
                if let Err(e) = Encoder::new(&g, &c) {
                    bad.push(format!("{name} at {fps} fps: {e}"));
                }
            }
        }
        assert!(bad.is_empty(), "{bad:?}");
    }

    fn samples() -> Option<PathBuf> {
        let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
        if d.is_none() {
            eprintln!("X5_SAMPLES not set: skipping");
        }
        d
    }

    fn out(name: &str) -> PathBuf {
        std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-{}-{name}", std::process::id()))
    }

    /// Every packet once, in order, each when the clock passes its start and timed by it; a seek or a jump
    /// restarts at the clock.
    #[test]
    fn tap() {
        let Some(d) = samples() else { return };
        let mut t = Tap::open(&d.join("v635.insv")).unwrap();
        let tr = t.m.tracks[t.t].clone();
        let ts = tr.timescale as f64;
        let start = |i: usize| tr.samples[i].dts as f64 / ts;
        let base = 1_000_000_000_000u64;
        let mut got = vec![];
        let run = |t: &mut Tap, a: f64, b: f64, got: &mut Vec<(u64, f64, usize)>| {
            let mut c = a;
            while c < b {
                let now = base + (c * 1e9) as u64;
                t.take(c, now, &mut |ns, p| got.push((ns, c, p.len())));
                c += 0.016;
            }
        };
        run(&mut t, 0.0, 2.0, &mut got);
        let n = tr
            .samples
            .iter()
            .filter(|s| s.dts as f64 / ts <= 1.999)
            .count();
        assert!(n > 80 && got.len().abs_diff(n) <= 1, "{} of {n}", got.len());
        for (i, &(ns, c, len)) in got.iter().enumerate() {
            assert_eq!(len, tr.samples[i].size as usize);
            // handed out within one step of its start, timed at its start
            assert!(c >= start(i) && c < start(i) + 0.0161, "packet {i} at {c}");
            assert!(
                (ns as f64 - base as f64 - start(i) * 1e9).abs() < 2e3,
                "packet {i}"
            );
        }
        // back to 0.5 s: from the first packet starting there
        got.clear();
        run(&mut t, 0.5, 0.6, &mut got);
        let k = tr.samples.partition_point(|s| (s.dts as f64) < 0.5 * ts);
        assert_eq!(got[0].2, tr.samples[k].size as usize);
        assert!((got[0].0 as f64 - base as f64 - start(k) * 1e9).abs() < 2e3);
        // a jump forward over 0.5 s skips what lies between
        got.clear();
        run(&mut t, 3.0, 3.05, &mut got);
        let k = tr.samples.partition_point(|s| (s.dts as f64) < 3.0 * ts);
        assert!((got[0].0 as f64 - base as f64 - start(k) * 1e9).abs() < 2e3);
        // a seek to 1 s holds the clock there (40 ms here) until its frame shows: nothing is heard meanwhile, and the
        // packets after are timed from when it moves again
        got.clear();
        for i in 0..3 {
            t.take(1.0, base + 3_100_000_000 + i * 16_000_000, &mut |ns, p| {
                got.push((ns, 1.0, p.len()))
            });
        }
        assert!(got.is_empty());
        let resume = base as f64 + 3.14e9;
        let mut c = 1.016;
        while c < 1.1 {
            t.take(c, (resume + (c - 1.0) * 1e9) as u64, &mut |ns, p| {
                got.push((ns, c, p.len()))
            });
            c += 0.016;
        }
        let k = tr.samples.partition_point(|s| (s.dts as f64) < 1.0 * ts);
        assert_eq!(got[0].2, tr.samples[k].size as usize);
        assert!((got[0].0 as f64 - resume - (start(k) - 1.0) * 1e9).abs() < 2e3);
        // a clock that moves in 20 ms device buffers, read every 16 ms: the times stay evenly spaced
        let mut t = Tap::open(&d.join("v635.insv")).unwrap();
        got.clear();
        let mut c = 0.0;
        while c < 2.0 {
            let k = (c / 0.02f64).floor() * 0.02;
            t.take(k, base + (c * 1e9) as u64, &mut |ns, _| {
                got.push((ns, c, 0))
            });
            c += 0.016;
        }
        let dur = tr.samples[0].dur as f64 / ts * 1e9;
        let dev = got[8..]
            .windows(2)
            .map(|w| (w[1].0 as f64 - w[0].0 as f64 - dur).abs())
            .fold(0.0, f64::max);
        println!(
            "tap: {n} packets in 2 s at {} Hz, each handed out within 16 ms of its start; on a 20 ms stepped clock \
             spaced {:.2} ms ± {:.2} ms",
            tr.timescale,
            dur / 1e6,
            dev / 1e6
        );
        assert!(dev < 2e6);
    }

    /// Frame n is the flat colour `col(n)`; audio is a file's AAC on a wall clock from the first call (`t0`, and
    /// its CLOCK_MONOTONIC ns).
    struct Pat {
        g: Arc<Gpu>,
        n: u32,
        tap: Option<Tap>,
        t0: Option<(Instant, u64)>,
    }

    fn pat(g: &Arc<Gpu>, d: &Path) -> Arc<Mutex<Pat>> {
        Arc::new(Mutex::new(Pat {
            g: g.clone(),
            n: 0,
            tap: Tap::open(&d.join("v635.insv")),
            t0: None,
        }))
    }

    fn col(n: u32) -> [u8; 3] {
        let a = (n * 37 % 200 + 28) as u8;
        [a, 128, 255 - a]
    }

    impl Source for Pat {
        fn name(&self) -> &str {
            "pattern"
        }

        fn tick(&mut self) {}

        fn ready(&self) -> bool {
            true
        }

        fn draw(&mut self, t: &wgpu::TextureView, _: [u32; 2], _: bool) {
            let [r, g, b] = col(self.n).map(|x| x as f64 / 255.0);
            self.n += 1;
            let mut e = self.g.device.create_command_encoder(&Default::default());
            e.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: t,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.g.queue.submit([e.finish()]);
        }

        fn asc(&mut self) -> Option<&[u8]> {
            self.tap.as_ref().map(|t| &t.asc[..])
        }

        fn audio(&mut self, f: &mut dyn FnMut(u64, &[u8])) {
            let t0 = self.t0.get_or_insert_with(|| (Instant::now(), mono())).0;
            if let Some(t) = &mut self.tap {
                t.take(t0.elapsed().as_secs_f64(), mono(), f);
            }
        }
    }

    fn tool(cmd: &str, args: &[&str]) -> Option<String> {
        let o = std::process::Command::new(cmd).args(args).output().ok()?;
        assert!(
            o.status.success(),
            "{cmd}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        Some(String::from_utf8_lossy(&o.stdout).into_owned())
    }

    /// Hands `l` the source `p` as one UI frame would, then lets its output thread run for `secs` without a UI (as
    /// with the window hidden), calling `each` every 2 ms.
    fn run(l: &mut Live, p: &Arc<Mutex<Pat>>, secs: f64, mut each: impl FnMut(&Live)) {
        l.tick(Some(p.clone() as Src));
        let t = Instant::now();
        while t.elapsed().as_secs_f64() < secs {
            each(l);
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// The audio of a file decoded to mono f32 at 8 kHz (ffmpeg, test tool).
    fn pcm8k(f: &Path) -> Vec<f32> {
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(f)
            .args(["-map", "0:a", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"])
            .output()
            .unwrap();
        o.stdout
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    /// 4 s of the pattern with a file's AAC through the whole RTMP path into `ffmpeg -listen` (test tool), with no
    /// UI running: H.264 1920 × 1080, every frame in order on the 30 fps grid, timestamps rising; the audio
    /// re-encoded, continuous, and the file's own audio at the time it was heard.
    #[test]
    fn rtmp() {
        stream(30);
    }

    /// The same at 60 fps: the output thread keeps a clock faster than the display's.
    #[test]
    fn rtmp60() {
        stream(60);
    }

    fn stream(fps: u32) {
        let Some(d) = samples() else { return };
        if tool("ffprobe", &["-version"]).is_none() {
            eprintln!("no ffprobe: skipping");
            return;
        }
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let file = out(&format!("live-rtmp{fps}.flv"));
        let f = file.to_str().unwrap();
        let (mut ff, url) = listen(port, &file);
        let g = Arc::new(Gpu::new().unwrap());
        let cx = stitch::Ctx::new(&g.device, &g.queue);
        let hook = Hook {
            rtmp: Some(url),
            fps: Some(fps),
            kbps: Some(4000),
            ..Default::default()
        };
        let mut l = Live::new(&g, &cx, &egui::Context::default(), Some(hook));
        let p = pat(&g, &d);
        run(&mut l, &p, 4.0, |_| {});
        // the thread's last pass hands over the final state
        let sh = l.sh.clone();
        drop(l);
        let s = sh.lock().unwrap().st.clone();
        let t = s.rtmp.as_ref().unwrap();
        // frames the encoder had no room for, ticks the clock missed (a stalled thread), frames made late
        let (st, kbps, missed, skip, made, t0) =
            (t.st.clone(), t.kbps, s.missed, t.skip, s.made, t.t0);
        let (n, a0) = {
            let p = p.lock().unwrap();
            (p.n, p.t0.unwrap().1)
        };
        println!(
            "RTMP {fps} fps: {n} frames drawn, {st:?}, {skip} not encoded, {kbps:.0} kbit/s; clock: {} made, {} \
             late (max {:.1} ms), {missed} missed",
            made.0,
            made.1,
            made.2 as f64 / 1e6
        );
        assert_eq!(st.state, State::Live);
        reap(&mut ff);
        let st = tool(
            "ffprobe",
            &[
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,profile,width,height,sample_rate,channels",
                "-of",
                "csv=p=0",
                f,
            ],
        )
        .unwrap();
        println!("streams: {}", st.trim().replace('\n', " | "));
        assert!(
            st.lines().any(|l| l.starts_with("h264,High,1920,1080")),
            "{st}"
        );
        assert!(st.lines().any(|l| l.starts_with("aac,LC,48000,2")), "{st}");
        let (v, a) = packets(f);
        // gaps in frame periods (FLV has whole ms)
        let per = 1e3 / fps as f64;
        let gaps: Vec<f64> = v.windows(2).map(|w| (w[1].0 - w[0].0) * 1e3).collect();
        let off = gaps.iter().map(|g| (g - (g / per).round() * per).abs());
        let miss: u64 = gaps.iter().map(|g| (g / per).round() as u64 - 1).sum();
        println!(
            "video: {} packets, {} key, {miss} frame periods left out (encoder behind: {skip}), gaps off the \
             {fps} fps grid by up to {:.2} ms; audio: {} packets over {:.2} s",
            v.len(),
            v.iter().filter(|x| x.1).count(),
            off.clone().fold(0.0, f64::max),
            a.len(),
            a.last().unwrap_or(&0.0) - a.first().unwrap_or(&0.0)
        );
        assert!(v[0].1, "starts with a key frame");
        // every frame drawn went out but those the encoder had no room for (the first ones may predate the
        // connection's first key frame)
        assert!(
            v.len() as u64 + 2 + skip >= n as u64 && v.len() as u32 <= n,
            "{} of {n}",
            v.len()
        );
        // the clock's frames, each on the grid; only the ones the UI reports as dropped are missing
        assert!(off.fold(0.0, f64::max) <= 1.0 && gaps.iter().all(|&g| g > per - 1.0));
        assert!(
            miss <= skip + missed && v.len() as u64 >= skip,
            "{miss} missing, {skip} dropped, {missed} missed"
        );
        // continuous: 1024 samples at 48 kHz each, from the start (FLV has whole ms)
        let ag: Vec<f64> = a.windows(2).map(|w| (w[1] - w[0]) * 1e3).collect();
        assert!(a[0] < 0.03 && ag.iter().all(|&g| g < 22.5), "{a:?}");
        let secs = v[v.len() - 1].0 - v[0].0 + 1.0 / fps as f64;
        assert!(
            (a.len() as f64 - secs * 46.875).abs() < 10.0,
            "{} audio packets",
            a.len()
        );
        // the file's audio as it was heard: the received audio at stream time τ is the file's at τ − (a0 − t0); the
        // best match of 1.5 s from 1 s in, at 8 kHz, within ±0.3 s of that
        let (rx, fx) = (pcm8k(&file), pcm8k(&d.join("v635.insv")));
        let (r0, w, j0) = (
            8000,
            12000,
            8000 - ((a0 as f64 - t0 as f64) / 1e9 * 8000.0).round() as i64,
        );
        let x = &rx[r0..r0 + w];
        let ex = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let (k, c) = (-2400..=2400i64)
            .filter_map(|k| {
                let y = fx.get((j0 + k).try_into().ok()?..)?.get(..w)?;
                let xy: f32 = x.iter().zip(y).map(|(a, b)| a * b).sum();
                Some((
                    k,
                    xy / (ex * y.iter().map(|v| v * v).sum::<f32>().sqrt()).max(1e-9),
                ))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        // decoded, the stream starts with the encoder's priming frame (1024 samples before sample 0)
        let off = k as f64 / 8.0 + 1024.0 / 48.0;
        println!(
            "audio: {} packets, gaps {:.0}..{:.0} ms, level {:.3} rms; the file's audio heard {off:.1} ms off \
             (correlation {c:.3})",
            a.len(),
            ag.iter().copied().fold(f64::MAX, f64::min),
            ag.iter().copied().fold(0.0, f64::max),
            ex / (w as f32).sqrt()
        );
        assert!(c > 0.8 && off.abs() < 5.0, "{off} ms, {c}");
        // the colours, in order
        let raw = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i", f, "-map", "0:v"])
            // one picture per packet, none repeated into the gaps
            .args(["-fps_mode", "passthrough"])
            .args([
                "-vf",
                "scale=16:9:in_color_matrix=bt709:in_range=tv,format=rgb24",
            ])
            .args(["-f", "rawvideo", "-"])
            .output()
            .unwrap()
            .stdout;
        let frames: Vec<[u8; 3]> = raw.chunks(16 * 9 * 3).map(|c| [c[0], c[1], c[2]]).collect();
        assert_eq!(frames.len(), v.len());
        let first = (0..8).min_by_key(|&n| err(frames[0], col(n))).unwrap();
        // a frame drawn while the encoder's input was full never went out: the next may skip patterns
        let (mut k, mut worst) = (first, err(frames[0], col(first)));
        for &c in &frames[1..] {
            k = (k + 1..=k + 1 + skip as u32)
                .min_by_key(|&j| err(c, col(j)))
                .unwrap();
            worst = worst.max(err(c, col(k)));
        }
        println!(
            "decoded: frame 0 is pattern {first}, max colour error {worst} over {}",
            frames.len()
        );
        // 60 fps has half the bits per frame, and frames the encoder left out make the next one's change larger;
        // patterns lie 37 or more apart
        assert!(worst <= if fps > 30 { 10 } else { 4 });
        std::fs::remove_file(&file).unwrap();
    }

    /// `ffmpeg -listen` (test tool) on `port` into `file`, listening; and its URL.
    fn listen(port: u16, file: &Path) -> (std::process::Child, String) {
        let url = format!("rtmp://127.0.0.1:{port}/live/test");
        let ff = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-listen",
                "1",
                "-i",
                &url,
            ])
            .args(["-c", "copy"])
            .arg(file)
            .stdin(std::process::Stdio::null())
            .spawn()
            .unwrap();
        // until it listens: a probe connection would be taken as the publisher
        let hex = format!(":{port:04X} ");
        let t = Instant::now();
        while !std::fs::read_to_string("/proc/net/tcp")
            .unwrap()
            .lines()
            .any(|l| l.contains(&hex) && l.split_whitespace().nth(3) == Some("0A"))
        {
            assert!(
                t.elapsed() < Duration::from_secs(10),
                "ffmpeg isn't listening"
            );
            thread::sleep(Duration::from_millis(20));
        }
        (ff, url)
    }

    fn reap(c: &mut std::process::Child) {
        let t = Instant::now();
        while c.try_wait().unwrap().is_none() {
            assert!(t.elapsed() < Duration::from_secs(10), "ffmpeg kept running");
            thread::sleep(Duration::from_millis(20));
        }
    }

    /// Video (pts s, key) and audio (pts s) packets of a file (ffprobe, test tool).
    fn packets(f: &str) -> (Vec<(f64, bool)>, Vec<f64>) {
        let pk = tool(
            "ffprobe",
            &[
                "-v",
                "error",
                "-show_entries",
                "packet=codec_type,pts_time,flags",
                "-of",
                "csv=p=0",
                f,
            ],
        )
        .unwrap();
        let (mut v, mut a) = (vec![], vec![]);
        for l in pk.lines() {
            let x: Vec<&str> = l.split(',').collect();
            let t: f64 = x[1].parse().unwrap();
            match x[0] {
                "video" => v.push((t, x[2].starts_with('K'))),
                _ => a.push(t),
            }
        }
        (v, a)
    }

    /// The server goes away mid-stream and another takes its port: the publisher reconnects, and the new session
    /// starts with an IDR made for it, not at the next GOP's.
    #[test]
    fn rtmp_reconnect() {
        let Some(d) = samples() else { return };
        if tool("ffprobe", &["-version"]).is_none() {
            eprintln!("no ffprobe: skipping");
            return;
        }
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let (fa, fb) = (out("live-a.flv"), out("live-b.flv"));
        let (mut a, url) = listen(port, &fa);
        let g = Arc::new(Gpu::new().unwrap());
        let cx = stitch::Ctx::new(&g.device, &g.queue);
        let hook = Hook {
            rtmp: Some(url),
            kbps: Some(4000),
            ..Default::default()
        };
        let mut l = Live::new(&g, &cx, &egui::Context::default(), Some(hook));
        let p = pat(&g, &d);
        // between IDRs (0 and 60): what's queued at the reconnect holds none
        run(&mut l, &p, 2.3, |_| {});
        a.kill().unwrap();
        a.wait().unwrap();
        let (mut b, _) = listen(port, &fb);
        // the frame handed to the encoder when the new session went live
        let mut at = None;
        run(&mut l, &p, 5.0, |l| {
            let s = l.sh.lock().unwrap();
            let t = s.st.rtmp.as_ref().unwrap();
            if at.is_none() && t.st.state == State::Live && t.st.reconnects == 1 {
                at = Some(t.f);
            }
        });
        let sh = l.sh.clone();
        drop(l);
        let s = sh.lock().unwrap().st.clone();
        let t = s.rtmp.as_ref().unwrap();
        let (f, st) = (t.f, t.st.clone());
        reap(&mut b);
        let at = at.expect("never live again");
        let (v, _) = packets(fb.to_str().unwrap());
        let keys: Vec<usize> = (0..v.len()).filter(|&i| v[i].1).collect();
        println!(
            "reconnect: {st:?}; live again at frame {at}, {} frames since; the new session has {} video packets, \
             key frames at {keys:?}",
            f - at,
            v.len()
        );
        assert_eq!(st.reconnects, 1);
        assert!(v[0].1);
        // from about the moment it went live, not from the next GOP (up to 60 frames later)
        assert!(v.len() as u64 + 3 >= f - at, "{} of {}", v.len(), f - at);
        std::fs::remove_file(&fb).unwrap();
        let _ = std::fs::remove_file(&fa);
    }

    fn err(a: [u8; 3], b: [u8; 3]) -> u8 {
        a.iter().zip(b).map(|(x, y)| x.abs_diff(y)).max().unwrap()
    }

    /// An OBS client on our socket: the source list, the `view` source's frames (DMA-BUF, LINEAR, read through a
    /// mapping after their sync_file) in order and paced at 30 fps, its audio as f32; nothing is rendered once it
    /// selects no source.
    #[test]
    fn obs() {
        let Some(d) = samples() else { return };
        let g = Arc::new(Gpu::new().unwrap());
        let cx = stitch::Ctx::new(&g.device, &g.queue);
        let run_dir =
            std::env::var_os("XDG_RUNTIME_DIR").map_or_else(std::env::temp_dir, PathBuf::from);
        let dir = run_dir.join(format!("x5desk-live-obs-{}", std::process::id()));
        let sock = dir.join("obs.sock");
        let hook = Hook {
            obs: true,
            sock: Some(sock.clone()),
            ..Default::default()
        };
        let mut l = Live::new(&g, &cx, &egui::Context::default(), Some(hook));
        assert_eq!(l.sh.lock().unwrap().st.err[1], None);
        let p = pat(&g, &d);
        let mut c = obs::Client::connect(&sock, 0, &[]).unwrap();
        let (mut list, mut fmt, mut frames, mut pcm) = (vec![], None, vec![], (0usize, 0u32, 0u32));
        let mut sel = false;
        run(&mut l, &p, 2.5, |_| {
            while let Ok(m) = c.recv(Some(Duration::ZERO)) {
                match m {
                    obs::Msg::Hello(v) => list = v,
                    obs::Msg::Format(2, f) => fmt = Some(f),
                    obs::Msg::Frame {
                        src: 2,
                        f,
                        planes,
                        sync,
                    } => {
                        let fd = planes[0].as_ref().unwrap().as_fd();
                        let mut pf = libc::pollfd {
                            fd: sync.as_ref().unwrap().as_fd().as_raw_fd(),
                            events: libc::POLLIN,
                            revents: 0,
                        };
                        assert_eq!(unsafe { libc::poll(&mut pf, 1, 1000) }, 1, "sync_file");
                        let n = (f.stride[0] * f.h) as usize;
                        let px = unsafe {
                            let m = libc::mmap(
                                std::ptr::null_mut(),
                                n,
                                libc::PROT_READ,
                                libc::MAP_SHARED,
                                fd.as_raw_fd(),
                                0,
                            );
                            assert_ne!(m, libc::MAP_FAILED);
                            let b = std::slice::from_raw_parts(m.cast::<u8>(), n);
                            let o = (f.off[0] + f.stride[0] * (f.h / 2) + 4 * (f.w / 2)) as usize;
                            let px = [b[0], b[1], b[2], b[o], b[o + 1], b[o + 2]];
                            libc::munmap(m, n);
                            px
                        };
                        frames.push((f.ts, px));
                        c.done(2, f.buf).unwrap();
                    }
                    obs::Msg::Audio {
                        src: 2,
                        rate,
                        ch,
                        pcm: x,
                        ..
                    } => pcm = (pcm.0 + x.len(), rate, ch),
                    _ => {}
                }
            }
            if !list.is_empty() && !sel {
                c.select(2).unwrap();
                sel = true;
            }
        });
        let names: Vec<&str> = list.iter().map(|x| x.1.as_str()).collect();
        assert_eq!(names, ["360", "view"]);
        let f = fmt.unwrap();
        assert_eq!(
            (f.w, f.h, f.fourcc, f.modifier, f.rate, f.ch),
            (1920, 1080, Fmt::Rgba.drm(), obs::MOD_LINEAR, 48000, 2)
        );
        let gaps: Vec<u64> = frames.windows(2).map(|w| w[1].0 - w[0].0).collect();
        // frames drawn while the client held obs::HOLD never reached it (the output thread doesn't wait for it)
        let held = l.sh.lock().unwrap().st.obs.as_ref().unwrap()[1].held as u32;
        let px = |i: usize| -> [u8; 3] { frames[i].1[..3].try_into().unwrap() };
        let first = (0..30).min_by_key(|&n| err(px(0), col(n))).unwrap();
        // in order: the next frame may skip what was held back
        let (mut k, mut bad) = (first, 0);
        for (i, (_, p)) in frames.iter().enumerate() {
            if i > 0 {
                k = (k + 1..=k + 1 + held)
                    .min_by_key(|&j| err(px(i), col(j)))
                    .unwrap();
            }
            bad += (p[..3] != col(k) || p[3..] != col(k)) as usize;
        }
        println!(
            "OBS: {} frames from pattern {first}, {bad} not matching ({held} held back); ts gaps {:?} ns; audio {} \
             samples at {} Hz × {}",
            frames.len(),
            gaps.iter().min().zip(gaps.iter().max()),
            pcm.0,
            pcm.1,
            pcm.2
        );
        assert!(frames.len() >= 40 && bad == 0);
        // on the clock's grid; a longer gap only where the clock missed ticks or every buffer was held
        let per = 1_000_000_000 / 30;
        let extra: u64 = gaps.iter().map(|g| g / per - 1).sum();
        assert!(gaps.iter().all(|&g| g >= per && g % per == 0), "{gaps:?}");
        let s = l.sh.lock().unwrap().st.clone();
        let v = &s.obs.as_ref().unwrap()[1];
        assert!(extra <= s.missed + v.skip + v.held);
        assert_eq!((pcm.1, pcm.2), (48000, 2));
        assert!(pcm.0 > 48000 * 2 * 3 / 2, "{} samples", pcm.0);
        // no source selected: nothing renders
        c.select(0).unwrap();
        run(&mut l, &p, 0.2, |_| {
            while c.recv(Some(Duration::ZERO)).is_ok() {}
        });
        let n = p.lock().unwrap().n;
        run(&mut l, &p, 0.3, |_| {
            while c.recv(Some(Duration::ZERO)).is_ok() {}
        });
        assert_eq!(p.lock().unwrap().n, n, "rendered without a watcher");
        drop((c, l));
        std::fs::remove_dir(&dir).unwrap();
    }
}
