//! The plugin in a real, isolated OBS. Runs only with `X5_OBS=<fresh scratch dir>` (OBS's config home; it opens an
//! OBS window on the desktop for about half a minute).
//!
//! A sender (crate live's server, DMA-BUF export on `video::Gpu`) publishes the sources in [`SRCS`] with a moving
//! pattern and a 440 Hz tone: RGBA, BGRA and NV12 in the modifiers the plugin negotiates, RGBA and NV12 forced
//! LINEAR. The plugin is installed into the scratch config the way a user installs it, OBS starts with
//! obs-websocket on port 4466, and over the websocket the test adds the source to a scene, compares
//! `GetSourceScreenshot`s with the pattern for each source, checks the source list in the properties, restarts the
//! app, and reads the audio meters. Then it publishes crate live's PipeWire camera node and reads the camera
//! portal's `IsCameraPresent`; with `X5_OBS_PW` set it also asks OBS's own PipeWire camera source for its devices.
#![cfg(target_os = "linux")]

use live::{
    dmabuf::{Fmt, Ring},
    obs::{BT709, Format, Frame, Server},
    pw,
};
use serde_json::{Value, json};
use std::{
    io::{self, Read, Write},
    net::TcpStream,
    os::fd::AsFd,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
    },
    thread,
    time::{Duration, Instant},
};
use video::Gpu;

const PORT: u16 = 4466;
/// The sender's sources: id, name, format, size, whether the ring follows the plugin's modifiers (else LINEAR).
const SRCS: [(u32, &str, Fmt, u32, u32, bool); 5] = [
    (1, "360", Fmt::Rgba, 1920, 960, true),
    (2, "flat", Fmt::Nv12, 1280, 720, true),
    (3, "lens", Fmt::Rgba, 960, 960, false),
    (4, "lens-nv12", Fmt::Nv12, 960, 960, false),
    (5, "bgra", Fmt::Bgra, 1280, 640, true),
];
const U: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC);

const WGSL: &str = r#"
struct P { f: u32, a: u32, b: u32, c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn rgba(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let x = u32(q.x); let y = u32(q.y);
    return vec4f(f32((x + 4u * p.f) & 255u), f32((y + 2u * p.f) & 255u), f32(p.f & 255u), 255.0) / 255.0;
}
@fragment fn luma(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let x = u32(q.x); let y = u32(q.y);
    return vec4f(f32(16u + (x + y + 3u * p.f) % 220u) / 255.0, 0.0, 0.0, 1.0);
}
// chroma texel coordinates: 8-texel blocks are 16 picture pixels
@fragment fn chroma(@builtin(position) q: vec4f) -> @location(0) vec4f {
    let x = u32(q.x) / 8u; let y = u32(q.y) / 8u;
    return vec4f(f32(64u + (x * 8u + p.f) % 128u), f32(64u + (y * 8u + 2u * p.f) % 128u), 0.0, 255.0) / 255.0;
}
"#;

/// Pixel (x, y) of frame `f` as OBS should show it.
fn rgba(x: u32, y: u32, f: u32) -> [u8; 3] {
    [(x + 4 * f) as u8, (y + 2 * f) as u8, f as u8]
}

/// The NV12 pattern through BT.709 limited range, with chroma constant over 16 × 16 blocks.
fn nv12(x: u32, y: u32, f: u32) -> [u8; 3] {
    let l = (16 + (x + y + 3 * f) % 220) as f64;
    let cb = (64 + (x / 16 * 8 + f) % 128) as f64;
    let cr = (64 + (y / 16 * 8 + 2 * f) % 128) as f64;
    let (l, u, v) = (
        (l - 16.) * 255. / 219.,
        (cb - 128.) * 255. / 224.,
        (cr - 128.) * 255. / 224.,
    );
    [
        l + 1.5748 * v,
        l - 0.187324 * u - 0.468124 * v,
        l + 1.8556 * u,
    ]
    .map(|c| c.round().clamp(0., 255.) as u8)
}

struct Pat {
    pipe: [wgpu::RenderPipeline; 4],
    bg: wgpu::BindGroup,
    u: wgpu::Buffer,
}

impl Pat {
    fn new(g: &Gpu) -> Pat {
        let d = &g.device;
        let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let u = d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: u.as_entire_binding(),
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
        Pat {
            pipe: [
                pipe("rgba", T::Rgba8Unorm),
                pipe("rgba", T::Bgra8Unorm),
                pipe("luma", T::R8Unorm),
                pipe("chroma", T::Rg8Unorm),
            ],
            bg,
            u,
        }
    }

    /// Renders frame `f` into `t` and submits.
    fn draw(&self, g: &Gpu, t: &wgpu::Texture, f: u32) {
        g.queue
            .write_buffer(&self.u, 0, &[f, 0, 0, 0].map(u32::to_ne_bytes).concat());
        let mut e = g.device.create_command_encoder(&Default::default());
        use wgpu::TextureAspect as A;
        let passes: &[(usize, A)] = match t.format() {
            wgpu::TextureFormat::NV12 => &[(2, A::Plane0), (3, A::Plane1)],
            wgpu::TextureFormat::Bgra8Unorm => &[(1, A::All)],
            _ => &[(0, A::All)],
        };
        for &(k, aspect) in passes {
            let v = t.create_view(&wgpu::TextureViewDescriptor {
                aspect,
                usage: Some(wgpu::TextureUsages::RENDER_ATTACHMENT),
                ..Default::default()
            });
            let mut p = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &v,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            p.set_pipeline(&self.pipe[k]);
            p.set_bind_group(0, &self.bg, &[]);
            p.draw(0..3, 0..1);
        }
        g.queue.submit([e.finish()]);
    }
}

fn now() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

/// Per source: frames sent, frames skipped (every image held), the ring's modifier.
#[derive(Clone, Debug, Default)]
struct Stats {
    sent: u32,
    skipped: u32,
    modifier: Option<u64>,
}

/// The app side: the sources in [`SRCS`] at 30 fps (rendered while watched) plus a tone, until `stop`.
fn sender(
    g: Arc<Gpu>,
    p: PathBuf,
    frame: Arc<AtomicU32>,
    stats: Arc<Mutex<[Stats; 5]>>,
    stop: Arc<AtomicBool>,
) {
    let pat = Pat::new(&g);
    let mut s = Server::bind(&p).unwrap();
    for (id, name, ..) in SRCS {
        s.add(id, name);
    }
    let mut rings: [Option<(Ring, Vec<u64>)>; 5] = Default::default();
    let (t0, mut f, mut samples) = (now(), 0u32, 0u64);
    let mut pcm = vec![0f32; 960];
    while !stop.load(Relaxed) {
        let changed = s.poll();
        // video at 30 fps
        if now() - t0 >= f as u64 * 1_000_000_000 / 30 {
            for (k, &(id, _, fmt, w, h, neg)) in SRCS.iter().enumerate() {
                if !s.watched(id) {
                    continue;
                }
                let m = if neg {
                    s.modifiers(id, fmt.drm())
                } else {
                    vec![]
                };
                if rings[k].as_ref().is_none_or(|r| changed && r.1 != m) {
                    let r = Ring::new(&g, fmt, w, h, U, &m, 4).unwrap();
                    let md = r.imgs[0].modifier;
                    s.format(
                        id,
                        Format {
                            modifier: md,
                            w,
                            h,
                            fourcc: fmt.drm(),
                            fps: [30, 1],
                            color: BT709,
                            rate: 48000,
                            ch: 2,
                        },
                    );
                    stats.lock().unwrap()[k].modifier = Some(md);
                    rings[k] = Some((r, m));
                }
                let r = &mut rings[k].as_mut().unwrap().0;
                let Some(i) = r.next(|i| s.busy(id, i as u32)) else {
                    stats.lock().unwrap()[k].skipped += 1;
                    continue;
                };
                let img = &r.imgs[i];
                img.acquire().unwrap();
                pat.draw(&g, &img.tex, f);
                let sf = img.release().unwrap();
                let mut fr = Frame {
                    ts: now(),
                    modifier: img.modifier,
                    w,
                    h,
                    fourcc: fmt.drm(),
                    buf: i as u32,
                    ..Default::default()
                };
                for (k, pl) in img.planes.iter().enumerate() {
                    (fr.off[k], fr.stride[k]) = (pl.off, pl.stride);
                }
                let fds = [img.fd.as_fd(); 4];
                if s.frame(id, &fr, &fds[..img.planes.len()], Some(sf.as_fd())) > 0 {
                    stats.lock().unwrap()[k].sent += 1;
                }
            }
            frame.store(f, Relaxed);
            f += 1;
        }
        // 10 ms of a 440 Hz tone at half scale, timestamps continuous from t0
        while now() - t0 >= (samples + 480) * 1_000_000_000 / 48000 {
            for (i, c) in pcm.chunks_mut(2).enumerate() {
                let t = (samples + i as u64) as f64 / 48000.;
                c.fill((0.5 * (std::f64::consts::TAU * 440. * t).sin()) as f32);
            }
            for (id, ..) in SRCS {
                s.audio(id, t0 + samples * 1_000_000_000 / 48000, 48000, 2, &pcm);
            }
            samples += 480;
        }
        thread::sleep(Duration::from_millis(2));
    }
}

/// A minimal obs-websocket v5 client (RFC 6455 text frames, JSON).
struct Ws {
    s: TcpStream,
    id: u64,
    /// events received while waiting for responses
    ev: Vec<Value>,
}

impl Ws {
    fn connect(port: u16) -> io::Result<Ws> {
        let mut s = TcpStream::connect(("127.0.0.1", port))?;
        s.write_all(
            b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
              Sec-WebSocket-Key: eDVkZXNrLW9icy10ZXN0IQ==\r\nSec-WebSocket-Version: 13\r\n\
              Sec-WebSocket-Protocol: obswebsocket.json\r\n\r\n",
        )?;
        let mut h = vec![];
        while !h.ends_with(b"\r\n\r\n") {
            let mut b = [0];
            s.read_exact(&mut b)?;
            h.push(b[0]);
        }
        if !h.starts_with(b"HTTP/1.1 101") {
            return Err(io::Error::other(String::from_utf8_lossy(&h).into_owned()));
        }
        let mut w = Ws {
            s,
            id: 0,
            ev: vec![],
        };
        assert_eq!(w.read()?["op"], 0, "Hello");
        // InputVolumeMeters only
        w.send(&json!({"op": 1, "d": {"rpcVersion": 1, "eventSubscriptions": 1 << 16}}))?;
        assert_eq!(w.read()?["op"], 2, "Identified");
        Ok(w)
    }

    fn read(&mut self) -> io::Result<Value> {
        let mut msg = vec![];
        loop {
            let mut h = [0u8; 2];
            self.s.read_exact(&mut h)?;
            let mut n = (h[1] & 0x7f) as u64;
            if n == 126 {
                let mut b = [0; 2];
                self.s.read_exact(&mut b)?;
                n = u16::from_be_bytes(b) as u64;
            } else if n == 127 {
                let mut b = [0; 8];
                self.s.read_exact(&mut b)?;
                n = u64::from_be_bytes(b);
            }
            let mut key = [0; 4];
            if h[1] & 0x80 != 0 {
                self.s.read_exact(&mut key)?;
            }
            let mut p = vec![0; n as usize];
            self.s.read_exact(&mut p)?;
            p.iter_mut().enumerate().for_each(|(i, b)| *b ^= key[i % 4]);
            match h[0] & 0x0f {
                8 => return Err(io::ErrorKind::ConnectionAborted.into()),
                9 => {
                    self.frame(10, &p)?;
                    continue;
                }
                10 => continue,
                _ => msg.extend_from_slice(&p),
            }
            if h[0] & 0x80 != 0 {
                return serde_json::from_slice(&msg).map_err(io::Error::other);
            }
        }
    }

    /// A masked frame; the key is 0, which leaves the payload as it is.
    fn frame(&mut self, op: u8, p: &[u8]) -> io::Result<()> {
        let mut f = vec![0x80 | op];
        match p.len() {
            n @ 0..126 => f.push(0x80 | n as u8),
            n @ 126..65536 => {
                f.push(0x80 | 126);
                f.extend_from_slice(&(n as u16).to_be_bytes());
            }
            n => {
                f.push(0x80 | 127);
                f.extend_from_slice(&(n as u64).to_be_bytes());
            }
        }
        f.extend_from_slice(&[0; 4]);
        f.extend_from_slice(p);
        self.s.write_all(&f)
    }

    fn send(&mut self, v: &Value) -> io::Result<()> {
        self.frame(1, v.to_string().as_bytes())
    }

    /// A request; its responseData, or Err with the status comment.
    fn try_req(&mut self, ty: &str, data: Value) -> Result<Value, String> {
        self.id += 1;
        let id = self.id.to_string();
        self.send(
            &json!({"op": 6, "d": {"requestType": ty, "requestId": id, "requestData": data}}),
        )
        .unwrap();
        loop {
            let mut m = self.read().unwrap();
            match m["op"].as_u64() {
                Some(7) if m["d"]["requestId"] == id.as_str() => {
                    let d = m["d"].take();
                    return match d["requestStatus"]["result"].as_bool() {
                        Some(true) => Ok(d["responseData"].clone()),
                        _ => Err(format!("{ty}: {}", d["requestStatus"])),
                    };
                }
                Some(5) => self.ev.push(m["d"].take()),
                _ => {}
            }
        }
    }

    fn req(&mut self, ty: &str, data: Value) -> Value {
        self.try_req(ty, data).unwrap()
    }

    /// The screenshot of `src` at its own size, as RGB rows; Err while the source has no size.
    fn shot(&mut self, src: &str) -> Result<(usize, usize, Vec<[u8; 3]>), String> {
        let r = self.try_req(
            "GetSourceScreenshot",
            json!({"sourceName": src, "imageFormat": "bmp"}),
        )?;
        let d = r["imageData"].as_str().unwrap();
        Ok(bmp(&b64(&d[d.find(',').unwrap() + 1..])))
    }
}

fn b64(s: &str) -> Vec<u8> {
    let val = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        _ => 63,
    };
    let s: Vec<u8> = s
        .bytes()
        .filter(|&c| c != b'=' && !c.is_ascii_whitespace())
        .map(val)
        .collect();
    let mut o = Vec::with_capacity(s.len() * 3 / 4);
    for c in s.chunks(4) {
        let n = c
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, &v)| a | (v as u32) << (18 - 6 * i));
        o.extend_from_slice(&n.to_be_bytes()[1..c.len()]);
    }
    o
}

/// 24- or 32-bit BMP (BI_RGB or BI_BITFIELDS) → width, height, RGB top-down.
fn bmp(b: &[u8]) -> (usize, usize, Vec<[u8; 3]>) {
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    assert_eq!(&b[..2], b"BM");
    let (off, w, h, bpp) = (
        u(10) as usize,
        u(18) as usize,
        u(22) as i32,
        u16::from_le_bytes([b[28], b[29]]),
    );
    let masks = match u(30) {
        3 => [u(54), u(58), u(62)],
        _ => [0xff0000, 0xff00, 0xff],
    };
    let (hh, bp) = (h.unsigned_abs() as usize, bpp as usize / 8);
    let stride = (w * bp).next_multiple_of(4);
    let mut px = Vec::with_capacity(w * hh);
    for y in 0..hh {
        let r = off + if h > 0 { hh - 1 - y } else { y } * stride;
        for x in 0..w {
            let p = &b[r + x * bp..];
            let v = u32::from_le_bytes([p[0], p[1], p[2], if bp == 4 { p[3] } else { 0 }]);
            px.push(masks.map(|m| ((v & m) >> m.trailing_zeros()) as u8));
        }
    }
    (w, hh, px)
}

/// Finds the frame among the last ones sent that the screenshot shows, then compares every pixel (NV12: those away
/// from the 16-pixel block edges, where chroma is interpolated): (frame, max error, pixels off by more than 2,
/// pixels checked).
fn compare(
    (w, h, px): &(usize, usize, Vec<[u8; 3]>),
    last: u32,
    fmt: Fmt,
) -> (u32, u8, usize, usize) {
    let nv = fmt == Fmt::Nv12;
    let err = |f: u32, step: usize| {
        let (mut m, mut bad, mut n) = (0u8, 0, 0);
        for i in (0..w * h).step_by(step) {
            let (x, y) = ((i % w) as u32, (i / w) as u32);
            if nv && !((2..14).contains(&(x % 16)) && (2..14).contains(&(y % 16))) {
                continue;
            }
            let e = if nv { nv12(x, y, f) } else { rgba(x, y, f) }
                .iter()
                .zip(&px[i])
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            (m, bad, n) = (m.max(e), bad + (e > 2) as usize, n + 1);
        }
        (m, bad, n)
    };
    let f = (last.saturating_sub(150)..=last)
        .min_by_key(|&f| err(f, 997).1)
        .unwrap();
    let (m, bad, n) = err(f, 1);
    (f, m, bad, n)
}

fn obs_log(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_default()
}

/// OBS, terminated on drop.
struct Obs(Child);

impl Drop for Obs {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0.id() as i32, libc::SIGTERM) };
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(10) {
            if let Ok(Some(s)) = self.0.try_wait() {
                println!("OBS exited: {s}");
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
        println!("OBS ignored SIGTERM for 10 s, killing it");
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// OBS on a fresh config in `cfg` (no first-run wizard, the plugin where users install it), obs-websocket on `port`
/// without auth; up and with our source kind registered. Returns OBS, the websocket and OBS's log file.
fn launch(cfg: &Path, port: u16) -> (Obs, Ws, PathBuf) {
    let _ = std::fs::remove_dir_all(cfg);
    let o = cfg.join("obs-studio");
    let bin = o.join("plugins/x5desk-obs/bin/64bit");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::create_dir_all(o.join("plugin_config/obs-websocket")).unwrap();
    std::fs::write(o.join("global.ini"), "[General]\nLastVersion=536936450\n").unwrap();
    std::fs::write(
        o.join("user.ini"),
        "[General]\nFirstRun=true\n[BasicWindow]\nSysTrayEnabled=true\nSysTrayWhenStarted=true\n",
    )
    .unwrap();
    std::fs::write(
        o.join("plugin_config/obs-websocket/config.json"),
        json!({"alerts_enabled": false, "auth_required": false, "first_load": false,
               "server_enabled": true, "server_password": "", "server_port": port})
        .to_string(),
    )
    .unwrap();
    // cargo test rebuilds the library next to this binary (target/<profile>/deps)
    let exe = std::env::current_exe().unwrap();
    let so = exe.parent().unwrap().join("libx5desk_obs.so");
    std::fs::copy(&so, bin.join("x5desk-obs.so")).expect("the built plugin");

    let log = cfg.join("obs-stdout.log");
    let lf = std::fs::File::create(&log).unwrap();
    // OBS's PipeWire camera source asks the camera portal for access. xdg-desktop-portal takes a host app's id from
    // its systemd scope name, so with X5_OBS_PW OBS runs in a scope named with OBS's own app id: the portal then
    // applies the camera permission the user gave OBS (it asks on the desktop if there is none)
    let pw = std::env::var_os("X5_OBS_PW").is_some();
    let mut cmd = Command::new(if pw { "systemd-run" } else { "obs" });
    if pw {
        let unit = format!(
            "--unit=app-com.obsproject.Studio-x5e2e{}.scope",
            std::process::id()
        );
        cmd.args(["--user", "--scope", "--quiet", &unit, "--", "obs"]);
    }
    let obs = Obs(cmd
        .args(["--multi", "--minimize-to-tray", "--disable-shutdown-check"])
        .env("XDG_CONFIG_HOME", cfg)
        .env("QT_QPA_PLATFORM", "xcb")
        .stdin(Stdio::null())
        .stdout(lf.try_clone().unwrap())
        .stderr(lf)
        .spawn()
        .expect("obs"));
    let t = Instant::now();
    let mut ws = loop {
        match Ws::connect(port) {
            Ok(w) => break w,
            Err(e) if t.elapsed() < Duration::from_secs(30) => {
                let _ = e;
                thread::sleep(Duration::from_millis(250));
            }
            Err(e) => panic!("obs-websocket: {e}\n{}", obs_log(&log)),
        }
    };
    // code 207 until OBS finished loading
    let v = loop {
        match ws.try_req("GetVersion", json!({})) {
            Ok(v) => break v,
            Err(e) if t.elapsed() < Duration::from_secs(30) => {
                let _ = e;
                thread::sleep(Duration::from_millis(250));
            }
            Err(e) => panic!("{e}"),
        }
    };
    println!(
        "OBS up after {:.1} s: {}",
        t.elapsed().as_secs_f64(),
        v["obsVersion"]
    );
    let kinds = ws.req("GetInputKindList", json!({}));
    assert!(
        kinds["inputKinds"]
            .as_array()
            .unwrap()
            .iter()
            .any(|k| k == "x5desk_source"),
        "plugin not loaded: {kinds}\n{}",
        obs_log(&log)
    );
    (obs, ws, log)
}

#[test]
fn obs() {
    let Some(cfg) = std::env::var_os("X5_OBS").map(PathBuf::from) else {
        println!("X5_OBS (a scratch config dir for an isolated OBS) is unset; skipped");
        return;
    };
    let (obs, mut ws, log) = launch(&cfg, PORT);
    let pw = std::env::var_os("X5_OBS_PW").is_some();
    let g = Arc::new(Gpu::new().expect("GPU"));
    let run = std::env::var_os("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR");
    let sock = Path::new(&run).join(format!("x5desk-obs-e2e-{}/obs.sock", std::process::id()));
    let (frame, stats) = (
        Arc::new(AtomicU32::new(0)),
        Arc::new(Mutex::new(Default::default())),
    );
    let start = || {
        let stop = Arc::new(AtomicBool::new(false));
        let (g, p, f, s, x) = (
            g.clone(),
            sock.clone(),
            frame.clone(),
            stats.clone(),
            stop.clone(),
        );
        (thread::spawn(move || sender(g, p, f, s, x)), stop)
    };
    let mut tx = start();

    ws.req("CreateScene", json!({"sceneName": "x5"}));
    ws.req("SetCurrentProgramScene", json!({"sceneName": "x5"}));
    let s = sock.to_str().unwrap();
    ws.req(
        "CreateInput",
        json!({"sceneName": "x5", "inputName": "x5cam", "inputKind": "x5desk_source",
               "inputSettings": {"socket": s, "source": "360"}}),
    );

    for (k, &(_, name, fmt, w, h, _)) in SRCS.iter().enumerate() {
        let size = (w as usize, h as usize);
        ws.req(
            "SetInputSettings",
            json!({"inputName": "x5cam", "inputSettings": {"source": name}}),
        );
        // until the source shows the new size
        let t = Instant::now();
        let mut sh = ws.shot("x5cam");
        while sh.as_ref().map_or(true, |s| (s.0, s.1) != size)
            && t.elapsed() < Duration::from_secs(5)
        {
            thread::sleep(Duration::from_millis(100));
            sh = ws.shot("x5cam");
        }
        let sh = sh.map(|s| (s.0, s.1));
        assert_eq!(sh, Ok(size), "{name}: never showed\n{}", obs_log(&log));
        println!(
            "{name}: shown after {:.0} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        thread::sleep(Duration::from_millis(300));
        let mut fs = vec![];
        for _ in 0..3 {
            let t = Instant::now();
            let sh = ws.shot("x5cam").unwrap();
            let (f, m, bad, n) = compare(&sh, frame.load(Relaxed), fmt);
            println!(
                "{name}: frame {f}: max error {m}, {bad} of {n} pixels off by > 2 (screenshot {:.0} ms)",
                t.elapsed().as_secs_f64() * 1e3
            );
            assert!(m <= 2 && bad == 0, "{name} frame {f}: max error {m}");
            fs.push(f);
            thread::sleep(Duration::from_millis(300));
        }
        assert!(
            fs.windows(2).all(|w| w[1] > w[0]),
            "{name}: not moving: {fs:?}"
        );
        let st = stats.lock().unwrap()[k].clone();
        println!(
            "{name}: sender sent {}, skipped {}, modifier {:#x}",
            st.sent,
            st.skipped,
            st.modifier.unwrap()
        );
    }

    // the source list in the properties, from the server's hello
    let items = ws.req(
        "GetInputPropertiesListPropertyItems",
        json!({"inputName": "x5cam", "propertyName": "source"}),
    );
    let names: Vec<&str> = items["propertyItems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["itemName"].as_str().unwrap())
        .collect();
    println!("properties list: {names:?}");
    assert_eq!(names[0], "First available");
    assert!(names[1..].iter().eq(SRCS.iter().map(|s| &s.1)));

    // the app quits and comes back: the source goes blank, then reconnects
    tx.1.store(true, Relaxed);
    tx.0.join().unwrap();
    thread::sleep(Duration::from_millis(500));
    let (_, _, px) = ws.shot("x5cam").unwrap();
    assert!(px.iter().all(|p| *p == [0; 3]), "not blank without the app");
    tx = start();
    let t = Instant::now();
    loop {
        let sh = ws.shot("x5cam").unwrap();
        if sh.2.iter().any(|p| *p != [0; 3]) {
            let (f, m, bad, _) = compare(&sh, frame.load(Relaxed), Fmt::Rgba);
            println!(
                "app restarted: frames again after {:.0} ms (frame {f}, max error {m})",
                t.elapsed().as_secs_f64() * 1e3
            );
            assert!(m <= 2 && bad == 0);
            break;
        }
        assert!(t.elapsed() < Duration::from_secs(5), "no reconnect");
        thread::sleep(Duration::from_millis(100));
    }

    // audio meters for our input: a half-scale sine peaks at 0.5 in both channels
    ws.req(
        "SetInputSettings",
        json!({"inputName": "x5cam", "inputSettings": {"source": "360"}}),
    );
    ws.ev.clear();
    thread::sleep(Duration::from_secs(2));
    ws.req("GetVersion", json!({}));
    let peaks: Vec<Vec<f64>> = ws
        .ev
        .iter()
        .filter(|e| e["eventType"] == "InputVolumeMeters")
        .flat_map(|e| {
            e["eventData"]["inputs"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter(|i| i["inputName"] == "x5cam")
        .map(|i| {
            i["inputLevelsMul"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c[1].as_f64().unwrap())
                .collect()
        })
        .collect();
    println!(
        "audio: {} meter events, last peaks {:?}",
        peaks.len(),
        peaks.last()
    );
    let tail = &peaks[peaks.len().saturating_sub(10)..];
    assert!(
        !tail.is_empty()
            && tail
                .iter()
                .all(|p| p.len() == 2 && p.iter().all(|v| (v - 0.5).abs() < 0.03)),
        "audio peaks {tail:?}"
    );

    // OBS's own stats, and the plugin's (logged every 5 s)
    thread::sleep(Duration::from_secs(5));
    let st = ws.req("GetStats", json!({}));
    println!(
        "OBS: {:.1} fps, render {:.2} ms avg, {} of {} render frames skipped",
        st["activeFps"].as_f64().unwrap_or(0.),
        st["averageFrameRenderTime"].as_f64().unwrap_or(0.),
        st["renderSkippedFrames"],
        st["renderTotalFrames"]
    );

    // crate live's camera node: does the camera portal count it, and does OBS's PipeWire camera source list it
    let present = || {
        let o = Command::new("busctl")
            .args([
                "--user",
                "get-property",
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
            ])
            .args(["org.freedesktop.portal.Camera", "IsCameraPresent"])
            .output();
        o.map_or_else(
            |e| e.to_string(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
    };
    let before = present();
    let cam = pw::Cam::new(g.clone(), "X5 Camera e2e", &pw::SIZES, 30).expect("PipeWire node");
    let t = Instant::now();
    while cam.node().is_none() && t.elapsed() < Duration::from_secs(5) {
        thread::sleep(Duration::from_millis(50));
    }
    thread::sleep(Duration::from_millis(500));
    println!(
        "PipeWire node id {:?}; portal IsCameraPresent {before} before, {} with it",
        cam.node(),
        present()
    );
    if !pw {
        println!("X5_OBS_PW unset: OBS's PipeWire camera source not tried");
    } else {
        // feed the node while OBS streams it
        let (feed, cf) = (AtomicBool::new(true), AtomicU32::new(0));
        thread::scope(|sc| {
            sc.spawn(|| {
                let pat = Pat::new(&g);
                while feed.load(Relaxed) {
                    if let Ok(Some(fr)) = cam.dequeue() {
                        let f = cf.load(Relaxed) + 1;
                        pat.draw(&g, &fr.img.tex, f);
                        fr.queue(now()).unwrap();
                        cf.store(f, Relaxed);
                    }
                    thread::sleep(Duration::from_millis(33));
                }
            });
            let r = ws.try_req(
                "CreateInput",
                json!({"sceneName": "x5", "inputName": "pwcam", "inputKind": "pipewire-camera-source",
                       "inputSettings": {}, "sceneItemEnabled": false}),
            );
            let t = Instant::now();
            let mut items = Value::Null;
            while r.is_ok() && t.elapsed() < Duration::from_secs(8) {
                items = ws
                    .try_req(
                        "GetInputPropertiesListPropertyItems",
                        json!({"inputName": "pwcam", "propertyName": "device_id"}),
                    )
                    .unwrap_or_else(Value::String);
                if items.to_string().contains("X5 Camera e2e") {
                    break;
                }
                thread::sleep(Duration::from_millis(500));
            }
            println!(
                "OBS's PipeWire camera source ({r:?}) lists after {:.1} s: {items}",
                t.elapsed().as_secs_f64()
            );
            let dev = items["propertyItems"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|i| i["itemName"] == "X5 Camera e2e");
            if let Some(d) = dev {
                ws.req(
                    "SetInputSettings",
                    json!({"inputName": "pwcam", "inputSettings": {"device_id": d["itemValue"]}}),
                );
                let t = Instant::now();
                let mut sh = ws.shot("pwcam");
                while sh
                    .as_ref()
                    .map_or(true, |s| s.2.iter().all(|p| *p == [0; 3]))
                    && t.elapsed() < Duration::from_secs(5)
                {
                    thread::sleep(Duration::from_millis(200));
                    sh = ws.shot("pwcam");
                }
                let mode = cam.mode();
                match (sh, mode) {
                    (Ok(sh), Some(m)) => {
                        let (f, e, bad, n) = compare(&sh, cf.load(Relaxed), m.fmt);
                        println!(
                            "PipeWire camera in OBS: {}x{} shown, stream {m:?}; frame {f}: max error {e}, {bad} of {n} off by > 2",
                            sh.0, sh.1
                        );
                    }
                    (sh, m) => println!(
                        "PipeWire camera in OBS: no picture ({:?}), stream {m:?}",
                        sh.err()
                    ),
                }
            }
            feed.store(false, Relaxed);
        });
    }
    drop(cam);

    let lines: Vec<String> = obs_log(&log)
        .lines()
        .filter(|l| l.contains("[x5desk]") || l.contains("x5cam") || l.contains("timestamp"))
        .map(Into::into)
        .collect();
    println!("plugin log:\n{}", lines.join("\n"));
    tx.1.store(true, Relaxed);
    tx.0.join().unwrap();
    drop(ws);
    drop(obs);
    let _ = std::fs::remove_dir(sock.parent().unwrap());
}

/// PNG → width, height, RGB rows, through ffmpeg (test tool).
fn png(p: &Path) -> (usize, usize, Vec<[u8; 3]>) {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(p)
        .output()
        .expect("ffprobe");
    let s = String::from_utf8_lossy(&probe.stdout);
    let wh: Vec<usize> = s.trim().split(',').map(|x| x.parse().unwrap()).collect();
    let o = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(p)
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        .output()
        .expect("ffmpeg");
    let px = o.stdout.as_chunks().0.to_vec();
    assert_eq!(px.len(), wh[0] * wh[1]);
    (wh[0], wh[1], px)
}

/// RGB rows → PNG, through ffmpeg (test tool).
fn save(p: &Path, (w, h, px): &(usize, usize, Vec<[u8; 3]>)) {
    let mut c = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{w}x{h}"),
        ])
        .args(["-i", "-"])
        .arg(p)
        .stdin(Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    c.stdin
        .take()
        .unwrap()
        .write_all(px.as_flattened())
        .unwrap();
    assert!(c.wait().unwrap().success());
}

/// Max channel error, pixels off by more than 2, between two pictures of the same size.
fn diff(a: &(usize, usize, Vec<[u8; 3]>), b: &(usize, usize, Vec<[u8; 3]>)) -> (u8, usize) {
    assert_eq!((a.0, a.1), (b.0, b.1));
    a.2.iter().zip(&b.2).fold((0, 0), |(m, n), (x, y)| {
        let e = x.iter().zip(y).map(|(p, q)| p.abs_diff(*q)).max().unwrap();
        (m.max(e), n + (e > 2) as usize)
    })
}

/// The app itself with an isolated OBS: `X5_OBS_APP` (a scratch dir), `X5_APP` (the x5desk binary), `X5_SAMPLES`.
/// First the app opens v635.insv paused on frame 60 (a still picture, RAFT-stitched) on its Live screen and serves
/// OBS on its default socket: OBS shows the `view` and then the `360` source, each screenshot must equal the app's
/// own picture of that source (`grab=`), and once the source is hidden the app renders it no more (its reports count
/// no more frames sent; its screenshot: "not watched"). Then the app plays the file in a loop: the picture moves, and
/// the plugin's statistics give the pacing over 10 s, at least 27 fps shown, in both runs. The app runs on Wayland:
/// it needs the display on.
#[test]
fn app() {
    let (Some(cfg), Some(bin), Some(dl)) = (
        std::env::var_os("X5_OBS_APP").map(PathBuf::from),
        std::env::var_os("X5_APP").map(PathBuf::from),
        std::env::var_os("X5_SAMPLES").map(PathBuf::from),
    ) else {
        println!("X5_OBS_APP, X5_APP or X5_SAMPLES unset; skipped");
        return;
    };
    let (obs, mut ws, log) = launch(&cfg, PORT + 1);
    let out = cfg.join("out");
    std::fs::create_dir_all(&out).unwrap();
    let start = |name: &str, test: &str, after: u32| {
        let lf = std::fs::File::create(cfg.join(format!("app-{name}.log"))).unwrap();
        Command::new(&bin)
            .arg(dl.join("v635.insv"))
            .env("XDG_CONFIG_HOME", cfg.join("app-config"))
            .env("X5DESK_TEST", test)
            .env("X5DESK_SHOT", out.join(format!("app-{name}.png")))
            .env("X5DESK_SHOT_AFTER", after.to_string())
            .stdin(Stdio::null())
            .stdout(lf.try_clone().unwrap())
            .stderr(lf)
            .spawn()
            .expect("x5desk")
    };
    let end = |app: &mut Child, name: &str| {
        let t = Instant::now();
        let status = loop {
            if let Some(s) = app.try_wait().unwrap() {
                break s;
            }
            assert!(
                t.elapsed() < Duration::from_secs(90),
                "the app kept running"
            );
            thread::sleep(Duration::from_millis(200));
        };
        let l = std::fs::read_to_string(cfg.join(format!("app-{name}.log"))).unwrap_or_default();
        println!("app ({name}) exited: {status}\n{l}");
    };
    let shown = |ws: &mut Ws, size: (usize, usize)| {
        let t = Instant::now();
        while ws.shot("x5cam").map_or(true, |s| {
            (s.0, s.1) != size || s.2.iter().all(|p| *p == [0; 3])
        }) {
            assert!(
                t.elapsed() < Duration::from_secs(30),
                "{size:?} never showed"
            );
            thread::sleep(Duration::from_millis(200));
        }
        println!("{size:?} shown after {:.1} s", t.elapsed().as_secs_f64());
    };
    // the plugin's statistics (logged every 5 s) over the next 10 s: each period shows at least 27 of the app's 30
    // fps (the GPU is shared with other work)
    let pacing = |ws: &mut Ws, what: &str| {
        let t0 = obs_log(&log).lines().count();
        thread::sleep(Duration::from_secs(11));
        let st = ws.req("GetStats", json!({}));
        let lines: Vec<String> = obs_log(&log)
            .lines()
            .skip(t0)
            .filter(|l| l.contains("fps shown"))
            .map(Into::into)
            .collect();
        println!(
            "pacing over 10 s, {what}:\n{}\nOBS: {:.1} fps, render {:.2} ms avg, {} of {} render frames skipped",
            lines.join("\n"),
            st["activeFps"].as_f64().unwrap_or(0.),
            st["averageFrameRenderTime"].as_f64().unwrap_or(0.),
            st["renderSkippedFrames"],
            st["renderTotalFrames"]
        );
        assert!(lines.len() >= 2);
        for l in &lines {
            let fps: f64 = l
                .split(" fps shown")
                .next()
                .and_then(|x| x.rsplit(' ').next())
                .and_then(|x| x.parse().ok())
                .unwrap();
            assert!(fps >= 27.0, "{l}");
        }
    };
    // a still picture once two screenshots 2 s apart agree (the RAFT stitch replaced the template seam); saved
    let still = |ws: &mut Ws, size: (usize, usize), name: &str| {
        let t = Instant::now();
        let mut last: Option<(usize, usize, Vec<[u8; 3]>)> = None;
        loop {
            assert!(
                t.elapsed() < Duration::from_secs(40),
                "no still {size:?} picture\n{}",
                obs_log(&log)
            );
            let Ok(s) = ws.shot("x5cam") else { continue };
            if (s.0, s.1) != size || s.2.iter().all(|p| *p == [0; 3]) {
                thread::sleep(Duration::from_millis(200));
                continue;
            }
            if last.as_ref().is_some_and(|l| diff(l, &s).0 == 0) {
                println!("{size:?} still after {:.1} s", t.elapsed().as_secs_f64());
                save(&out.join(format!("obs-{name}.png")), &s);
                return s;
            }
            last = Some(s);
            thread::sleep(Duration::from_millis(2000));
        }
    };

    let mut app = start(
        "paused",
        &format!("live,obs,seek=60,grab={}", out.join("grab.png").display()),
        // after the source is hidden, about 45 s in
        60,
    );
    ws.req("CreateScene", json!({"sceneName": "x5"}));
    ws.req("SetCurrentProgramScene", json!({"sceneName": "x5"}));
    ws.req(
        "CreateInput",
        json!({"sceneName": "x5", "inputName": "x5cam", "inputKind": "x5desk_source",
               "inputSettings": {"source": "view"}}),
    );
    shown(&mut ws, (1920, 1080));
    // the app saved its pictures once the paused frame's RAFT stitch was done: the picture OBS shows from then on
    let t = Instant::now();
    while !std::fs::read_to_string(cfg.join("app-paused.log"))
        .unwrap_or_default()
        .contains("view picture")
    {
        assert!(
            t.elapsed() < Duration::from_secs(60),
            "the app never grabbed its pictures"
        );
        thread::sleep(Duration::from_millis(200));
    }
    println!(
        "the app's pictures after {:.1} s",
        t.elapsed().as_secs_f64()
    );
    let view = still(&mut ws, (1920, 1080), "view");
    pacing(&mut ws, "paused (the same picture each frame)");
    ws.req(
        "SetInputSettings",
        json!({"inputName": "x5cam", "inputSettings": {"source": "360"}}),
    );
    let pano = still(&mut ws, (3840, 1920), "360");
    // hidden: the plugin selects no source, so the app stops rendering it
    let id = ws.req(
        "GetSceneItemId",
        json!({"sceneName": "x5", "sourceName": "x5cam"}),
    )["sceneItemId"]
        .clone();
    let enable = |ws: &mut Ws, on: bool| {
        ws.req(
            "SetSceneItemEnabled",
            json!({"sceneName": "x5", "sceneItemId": id, "sceneItemEnabled": on}),
        );
    };
    enable(&mut ws, false);
    end(&mut app, "paused");
    // hidden, the plugin selected no source: the app's last two reports (5 s apart) count the same frames sent
    let rep: Vec<String> = std::fs::read_to_string(cfg.join("app-paused.log"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("x5desk live:"))
        .map(|l| l.split("; OBS ").skip(1).collect::<Vec<_>>().join("; "))
        .collect();
    assert!(
        rep.len() >= 2 && rep[rep.len() - 1] == rep[rep.len() - 2],
        "{rep:?}"
    );
    println!(
        "hidden: the app sent no more frames ({})",
        rep[rep.len() - 1]
    );
    for (name, shot) in [("view", &view), ("360", &pano)] {
        let ours = png(&out.join(format!("grab-{name}.png")));
        let (m, bad) = diff(shot, &ours);
        println!(
            "{name}: OBS screenshot vs the app's picture ({} × {}): max error {m}, {bad} pixels off by > 2",
            ours.0, ours.1
        );
        assert!(m <= 2 && bad == 0, "{name}");
    }

    // playing: shown again, moving
    ws.req(
        "SetInputSettings",
        json!({"inputName": "x5cam", "inputSettings": {"source": "view"}}),
    );
    enable(&mut ws, true);
    let mut app = start("playing", "play,loop,live,obs", 30);
    shown(&mut ws, (1920, 1080));
    thread::sleep(Duration::from_secs(3));
    ws.ev.clear();
    pacing(&mut ws, "playing");
    // the file's sound, decoded by the app, in OBS's meters
    let peaks: Vec<f64> = ws
        .ev
        .iter()
        .filter(|e| e["eventType"] == "InputVolumeMeters")
        .flat_map(|e| {
            e["eventData"]["inputs"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter(|i| i["inputName"] == "x5cam")
        .flat_map(|i| {
            let v = i["inputLevelsMul"].as_array().cloned().unwrap_or_default();
            v.into_iter().map(|c| c[1].as_f64().unwrap_or(0.))
        })
        .collect();
    let max = peaks.iter().copied().fold(0., f64::max);
    let on = peaks.iter().filter(|&&p| p > 0.001).count();
    println!(
        "playing: audio meters {} channel readings, {on} above -60 dBFS, peak {max:.3}",
        peaks.len()
    );
    assert!(on > peaks.len() / 2);
    let a = ws.shot("x5cam").unwrap();
    thread::sleep(Duration::from_millis(500));
    let b = ws.shot("x5cam").unwrap();
    save(&out.join("obs-playing.png"), &b);
    let (m, n) = diff(&a, &b);
    println!("playing: two screenshots 0.5 s apart differ by up to {m} in {n} pixels");
    assert!(n > 10000, "not moving");
    end(&mut app, "playing");
    let lines: Vec<String> = obs_log(&log)
        .lines()
        .filter(|l| l.contains("[x5desk]"))
        .map(Into::into)
        .collect();
    println!("plugin log:\n{}", lines.join("\n"));
    drop(ws);
    drop(obs);
}
