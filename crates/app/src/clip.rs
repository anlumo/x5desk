//! A clip's video: the media thread demuxes (crate `media`) and decodes (crate `video`) ahead of the master clock
//! into the decoders' rings and queues the frames; the UI takes the newest frame due and never paces anything.
//! Frame hand-over (ARCHITECTURE.md): the UI acquires the frame it samples and releases it when it takes a newer one
//! (`release(next − 1)` also frees the frames it skipped) or when it seeks; the media thread releases what the UI
//! never got: frames decoded only to reach a seek target and the queue a seek discards.

use media::mp4::{Codec, Mp4};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    thread,
    time::Duration,
};
use video::{Decoder, Gpu, Ring};

/// A frame counts as due this much early: a seek puts the clock on the target's time rounded to audio samples.
const EPS: f64 = 1e-3;

/// A decoded frame: decoder frame index per ring, sample index, presentation time (s), seek epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pic {
    pub f: [u64; 2],
    pub k: usize,
    pub t: f64,
    e: u32,
}

#[derive(Default)]
struct Q {
    pics: VecDeque<Pic>,
    /// the UI's latest seek: epoch and target sample
    epoch: u32,
    target: usize,
    err: Option<String>,
}

struct Sh {
    q: Mutex<Q>,
    quit: AtomicBool,
}

impl Sh {
    fn q(&self) -> std::sync::MutexGuard<'_, Q> {
        self.q.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The last frame with `times[k] <= t` (0 before the first).
pub fn frame_at(times: &[f64], t: f64) -> usize {
    times.partition_point(|&x| x <= t + EPS).saturating_sub(1)
}

/// The last sync sample at or before `k` (0 if none).
fn idr(sync: &[usize], k: usize) -> usize {
    sync.partition_point(|&s| s <= k)
        .checked_sub(1)
        .map_or(0, |i| sync[i])
}

#[derive(Debug, PartialEq)]
enum Plan {
    /// the queue holds the target: drop this many frames before it
    Keep(usize),
    /// decode from this sample, dropping everything before the target
    Go(usize),
}

/// Seek to sample `t`, with `pos` = the next sample to decode and `q0` = the first queued one: keep the queue from `t`
/// on if it holds `t`; else decode on from `pos` when no sync sample lies in (pos, t] (`idr` = the last one ≤ t);
/// else restart at `idr`.
fn plan(t: usize, pos: usize, q0: Option<usize>, idr: usize) -> Plan {
    match q0 {
        Some(q0) if q0 <= t && t < pos => Plan::Keep(t - q0),
        _ if idr <= pos && pos <= t => Plan::Go(pos),
        _ => Plan::Go(idr),
    }
}

/// The video streams of a clip, one per lens (`.insv`: two tracks or two files) or one with both lenses side by side
/// (`.lrv`), on a media thread.
pub struct Video {
    sh: Arc<Sh>,
    th: Option<thread::JoinHandle<()>>,
    pub rings: Vec<Arc<Ring>>,
    /// display window (x, y, w, h) of the first stream's pictures
    pub view: [u32; 4],
    /// presentation time of each frame, and the end of the last one (s)
    pub times: Arc<[f64]>,
    pub end: f64,
    epoch: u32,
}

impl Video {
    /// The video tracks of `files` in order (at most two) on a new media thread; `wake` runs after each queued frame.
    pub fn open(
        g: &Arc<Gpu>,
        files: &[PathBuf],
        wake: impl Fn() + Send + 'static,
    ) -> Result<Video, String> {
        let (mut mp4s, mut st) = (vec![], vec![]);
        for (i, p) in files.iter().enumerate() {
            let m = Mp4::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            for (t, tr) in m.tracks.iter().enumerate() {
                if matches!(tr.codec, Codec::Hevc { .. } | Codec::Avc { .. }) && st.len() < 2 {
                    st.push((i, t));
                }
            }
            mp4s.push(m);
        }
        if st.is_empty() {
            return Err("no H.264 / H.265 video track".into());
        }
        let mut decs = vec![];
        for &(i, t) in &st {
            let tr = &mp4s[i].tracks[t];
            let (c, ps) = match &tr.codec {
                Codec::Hevc { ps, .. } => (video::Codec::H265, ps),
                Codec::Avc { ps, .. } => (video::Codec::H264, ps),
                _ => unreachable!(),
            };
            decs.push(Decoder::new(g, c, ps, tr.width, tr.height).map_err(|e| e.to_string())?);
        }
        let (m, tr) = (&mp4s[st[0].0], &mp4s[st[0].0].tracks[st[0].1]);
        let n = st
            .iter()
            .map(|&(i, t)| mp4s[i].tracks[t].samples.len())
            .min()
            .unwrap();
        if n == 0 {
            return Err("empty video track".into());
        }
        let (d, ts) = (tr.shift(m.timescale), tr.timescale as f64);
        let times: Arc<[f64]> = tr.samples[..n]
            .iter()
            .map(|s| (s.pts + d) as f64 / ts)
            .collect();
        let s = tr.samples[n - 1];
        let end = (s.pts + d + s.dur as i64) as f64 / ts;
        let sync: Vec<usize> = (0..n).filter(|&k| tr.samples[k].sync).collect();
        let sh = Arc::new(Sh {
            q: Mutex::new(Q::default()),
            quit: AtomicBool::new(false),
        });
        let rings = decs.iter().map(|d| d.ring.clone()).collect();
        let view = decs[0].view;
        let (g, s, t) = (g.clone(), sh.clone(), times.clone());
        let th = thread::Builder::new()
            .name("media".into())
            .spawn(move || run(&g, &s, &mp4s, &st, decs, &t, &sync, wake))
            .map_err(|e| e.to_string())?;
        Ok(Video {
            sh,
            th: Some(th),
            rings,
            view,
            times,
            end,
            epoch: 0,
        })
    }

    /// Stages the waits for `p` (before each submit that samples it).
    pub fn acquire(&self, g: &Gpu, p: &Pic) {
        for (r, ring) in self.rings.iter().enumerate() {
            ring.acquire(g, p.f[r]);
        }
    }

    /// Releases `p` and everything before it (after the last submit that samples it).
    pub fn release(&self, g: &Gpu, p: &Pic) {
        for (r, ring) in self.rings.iter().enumerate() {
            ring.release(g, p.f[r]);
        }
    }

    /// Releases everything before `p`.
    pub fn release_before(&self, g: &Gpu, p: &Pic) {
        for (r, ring) in self.rings.iter().enumerate() {
            if p.f[r] > 0 {
                ring.release(g, p.f[r] - 1);
            }
        }
    }

    /// Takes the newest queued frame due at `now` and how many due ones it skipped. The caller samples it and
    /// releases the frame it held with `release_before`. `done`: only frames fully decoded, so sampling one never
    /// makes the GPU queue (and a UI thread presenting behind it) wait for the decoder; a due frame still decoding
    /// waits for a later call.
    pub fn take(&self, now: f64, done: bool) -> (Option<Pic>, usize) {
        let mut q = self.sh.q();
        let (mut p, mut n) = (None, 0);
        // frames of an older epoch wait for the media thread to drop or keep them
        while let Some(&x) = q.pics.front()
            && x.e == self.epoch
            && x.t <= now + EPS
            && (!done || self.rings.iter().zip(&x.f).all(|(r, &f)| r.ready() > f))
        {
            n += p.is_some() as usize;
            p = Some(x);
            q.pics.pop_front();
        }
        (p, n)
    }

    /// Whether a frame of the current position waits in the queue (decoded or not).
    pub fn queued(&self) -> bool {
        self.sh.q().pics.front().is_some_and(|x| x.e == self.epoch)
    }

    /// Shows sample `k` next: the first frame `take` returns from now on is `k`. `held` (the frame the caller
    /// sampled last) is released here, so the media thread may reuse every slot.
    pub fn seek(&mut self, g: &Gpu, k: usize, held: Option<Pic>) {
        if let Some(h) = held {
            self.release(g, &h);
        }
        self.epoch += 1;
        let mut q = self.sh.q();
        (q.epoch, q.target) = (self.epoch, k.min(self.times.len() - 1));
    }

    /// The media thread's first error since the last call.
    pub fn err(&self) -> Option<String> {
        self.sh.q().err.take()
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        self.sh.quit.store(true, Relaxed);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

/// Splits a sample's length-prefixed NAL units (`n`-byte lengths) into `out`; returns how many.
fn nals<'a>(mut b: &'a [u8], n: usize, out: &mut [&'a [u8]]) -> usize {
    let mut c = 0;
    while b.len() > n && c < out.len() {
        let l = b[..n].iter().fold(0usize, |a, &x| a << 8 | x as usize);
        let Some(x) = b.get(n..n + l) else { break };
        out[c] = x;
        c += 1;
        b = &b[n + l..];
    }
    c
}

/// The media thread: decodes every stream's sample `pos` into its decoder, queues the frame (or releases it while
/// running up to a seek target), and follows the UI's seeks. Backpressure is the rings: `decode` waits while the
/// UI holds the slot it needs.
#[allow(clippy::too_many_arguments)]
fn run(
    g: &Gpu,
    sh: &Sh,
    mp4s: &[Mp4],
    st: &[(usize, usize)],
    mut decs: Vec<Decoder>,
    times: &[f64],
    sync: &[usize],
    wake: impl Fn(),
) {
    let n = times.len();
    let nal: Vec<usize> = st
        .iter()
        .map(|&(i, t)| match mp4s[i].tracks[t].codec {
            Codec::Hevc { nal, .. } | Codec::Avc { nal, .. } => nal as usize,
            _ => 4,
        })
        .collect();
    let (mut pos, mut skip, mut e, mut buf) = (0, 0, 0, vec![]);
    let mut f = [0u64; 2];
    let fail = |m: String| {
        let mut q = sh.q();
        if q.err.is_none() {
            eprintln!("media: {m}");
            q.err = Some(m);
        }
    };
    let release = |decs: &[Decoder], f: [u64; 2]| {
        for (d, &f) in decs.iter().zip(&f) {
            d.release(g, f);
        }
    };
    'main: while !sh.quit.load(Relaxed) {
        {
            let mut q = sh.q();
            if q.epoch != e {
                e = q.epoch;
                let t = q.target.min(n - 1);
                match plan(t, pos, q.pics.front().map(|p| p.k), idr(sync, t)) {
                    Plan::Keep(d) => {
                        if d > 0 {
                            release(&decs, q.pics[d - 1].f);
                            q.pics.drain(..d);
                        }
                        q.pics.iter_mut().for_each(|p| p.e = e);
                    }
                    Plan::Go(p) => {
                        q.pics.clear();
                        for d in &decs {
                            if d.frame > 0 {
                                d.release(g, d.frame - 1);
                            }
                        }
                        pos = p;
                    }
                }
                skip = t;
            }
        }
        if pos >= n {
            thread::sleep(Duration::from_millis(5));
            continue;
        }
        let mut ok = true;
        for (i, &(fi, ti)) in st.iter().enumerate() {
            let s = mp4s[fi].tracks[ti].samples[pos];
            if let Err(x) = mp4s[fi].read(&s, &mut buf) {
                fail(format!("sample {pos}: {x}"));
                ok = false;
                continue;
            }
            let mut au = [&[][..]; 64];
            let c = nals(&buf, nal[i], &mut au);
            loop {
                match decs[i].decode(g, &au[..c]) {
                    Ok(x) => {
                        f[i] = x;
                        break;
                    }
                    Err(video::Error::Full) => {
                        if sh.quit.load(Relaxed) {
                            return;
                        }
                        // nothing of this sample decoded yet: free to follow a seek instead
                        if i == 0 && sh.q().epoch != e {
                            continue 'main;
                        }
                    }
                    Err(x) => {
                        fail(format!("frame {pos}, stream {i}: {x}"));
                        ok = false;
                        break;
                    }
                }
            }
        }
        if ok && pos >= skip {
            sh.q().pics.push_back(Pic {
                f,
                k: pos,
                t: times[pos],
                e,
            });
            wake();
        } else if ok {
            release(&decs, f);
        }
        pos += 1;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::{path::Path, time::Instant};

    #[test]
    fn seek_maths() {
        let times: Vec<f64> = (0..175).map(|k| k as f64 * 1001.0 / 30000.0).collect();
        assert_eq!(frame_at(&times, -1.0), 0);
        assert_eq!(frame_at(&times, 0.0), 0);
        // due EPS early
        assert_eq!(frame_at(&times, times[1] - 0.002), 0);
        assert_eq!(frame_at(&times, times[1] - 0.0005), 1);
        // a clock on the target rounded to 48 kHz samples, a hair early, still shows the target
        assert_eq!(
            frame_at(&times, (times[100] * 48000.0).floor() / 48000.0),
            100
        );
        assert_eq!(frame_at(&times, 1e9), 174);
        let sync = [0, 60, 120];
        let i = |k| idr(&sync, k);
        assert_eq!([i(0), i(59), i(60), i(100), i(174)], [0, 0, 60, 60, 120]);
        assert_eq!(idr(&[], 5), 0);
        // queued 10..14, next to decode 14
        assert_eq!(plan(12, 14, Some(10), 0), Plan::Keep(2));
        assert_eq!(plan(10, 14, Some(10), 0), Plan::Keep(0));
        // just past the queue, no sync sample in between: decode on
        assert_eq!(plan(14, 14, Some(10), 0), Plan::Go(14));
        assert_eq!(plan(59, 14, Some(10), 0), Plan::Go(14));
        // a sync sample after pos: restart there
        assert_eq!(plan(100, 14, Some(10), 60), Plan::Go(60));
        // backwards: restart
        assert_eq!(plan(5, 14, Some(10), 0), Plan::Go(0));
        assert_eq!(plan(9, 70, Some(66), 0), Plan::Go(0));
        assert_eq!(plan(0, 0, None, 0), Plan::Go(0));
    }

    #[test]
    fn nal_split() {
        let b = [0, 0, 0, 2, 0xa, 0xb, 0, 0, 0, 1, 0xc, 0, 0, 0, 9];
        let mut o = [&[][..]; 4];
        assert_eq!(nals(&b, 4, &mut o), 2);
        assert_eq!((o[0], o[1]), (&[0xa, 0xb][..], &[0xc][..]));
    }

    /// FFmpeg's NV12 (P010 when `deep`) of frame `k` (presentation order) of video stream `t` (test tool).
    fn ffmpeg(p: &Path, t: usize, k: usize, deep: bool) -> Vec<u8> {
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(p)
            .args([
                "-map",
                &format!("0:v:{t}"),
                "-vf",
                &format!("select=eq(n\\,{k})"),
            ])
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
        o.stdout
    }

    /// The window of frame `f` of `ring` read back as NV12, or P010 (2 bytes per sample, little-endian) from a 10-bit
    /// ring, through copies of its plane views; the caller acquires and releases it.
    fn read(g: &Gpu, ring: &Ring, [x, y, w, h]: [u32; 4], f: u64) -> Vec<u8> {
        let tex = ring.texture(f);
        // bytes per sample: a P010 row of either plane is 2 · w bytes
        let b = 1 + (tex.format() == wgpu::TextureFormat::P010) as usize;
        let (w, h) = (w as usize, h as usize);
        let bpr = (b * w).next_multiple_of(256);
        let buf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bpr * h * 3 / 2) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e = g.device.create_command_encoder(&Default::default());
        for (aspect, s, offset) in [
            (wgpu::TextureAspect::Plane0, 1, 0),
            (wgpu::TextureAspect::Plane1, 2, bpr * h),
        ] {
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: x / s,
                        y: y / s,
                        z: 0,
                    },
                    aspect,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: offset as u64,
                        bytes_per_row: Some(bpr as u32),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: (w / s as usize) as u32,
                    height: (h / s as usize) as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        // back home before the decoder reuses the image
        e.transition_resources(
            std::iter::empty(),
            [wgpu::TextureTransition {
                texture: tex,
                selector: None,
                state: wgpu::TextureUses::RESOURCE,
            }]
            .into_iter(),
        );
        let i = g.queue.submit([e.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        g.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(i),
                timeout: None,
            })
            .unwrap();
        let m = buf.slice(..).get_mapped_range().unwrap();
        (0..h * 3 / 2)
            .flat_map(|r| &m[r * bpr..r * bpr + b * w])
            .copied()
            .collect()
    }

    /// A directory of the test run's own under `X5_FIXTURES` (default: the temp dir), removed when dropped, also
    /// when the test fails.
    pub(crate) struct Tmp(pub PathBuf);

    impl Tmp {
        pub fn new(name: &str) -> Tmp {
            let d = std::env::var_os("X5_FIXTURES")
                .map_or_else(std::env::temp_dir, PathBuf::from)
                .join(format!("x5desk-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&d).unwrap();
            Tmp(d)
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Makes `p` with the ffmpeg CLI (test tool; false if it is missing): 200 frames of 640 × 360 at 25 fps, frame
    /// n's number burned in as digits and in binary along the top (`number`), encoded with `enc`.
    pub(crate) fn numbered(p: &Path, enc: &[&str]) -> bool {
        let bits = "if(lt(Y\\,40)\\,16+219*bitand(floor(N/pow(2\\,floor(X/40)))\\,1)";
        let vf = format!(
            "geq=lum='{bits}\\,lum(X\\,Y))':cb='if(lt(Y\\,40)\\,128\\,cb(X\\,Y))':\
             cr='if(lt(Y\\,40)\\,128\\,cr(X\\,Y))',drawtext=text='%{{frame_num}}':fontsize=160:\
             fontcolor=white:box=1:boxcolor=black:x=(w-tw)/2:y=(h-th)/2+20"
        );
        let Ok(o) = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=640x360:r=25:d=8",
            ])
            .args(["-vf", &vf])
            .args(enc)
            .arg(p)
            .output()
        else {
            return false;
        };
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        true
    }

    /// The number `numbered` burned into a picture whose luma rows are `w` samples of `b` bytes (the high byte read).
    pub(crate) fn number(y: &[u8], w: usize, b: usize) -> usize {
        (0..8)
            .map(|i| ((y[(20 * w + 40 * i + 20) * b + b - 1] > 128) as usize) << i)
            .sum()
    }

    /// Frame `k` the way the player shows it after a seek, read back.
    fn show(g: &Gpu, v: &mut Video, k: usize, held: &mut Option<Pic>) -> Vec<u8> {
        v.seek(g, k, held.take());
        let t0 = Instant::now();
        let x = loop {
            if let (Some(x), _) = v.take(v.times[k], true) {
                break x;
            }
            assert!(t0.elapsed().as_secs() < 10, "seek {k} stalled");
            thread::sleep(Duration::from_millis(1));
        };
        assert_eq!((x.k, x.t), (k, v.times[k]));
        v.acquire(g, &x);
        let px = read(g, &v.rings[0], v.view, x.f[0]);
        v.release_before(g, &x);
        *held = Some(x);
        px
    }

    /// A 10-bit H.265 clip (ffmpeg CLI, libx265 Main 10) decodes into P010: frame 30 read back after a seek is
    /// FFmpeg's P010 to the byte, and carries its number.
    #[test]
    fn deep() {
        let dir = Tmp::new("deep");
        let p = dir.0.join("main10.mp4");
        let enc = [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "log-level=error:bframes=0",
        ];
        if !numbered(&p, &enc) {
            eprintln!("no ffmpeg: skipped");
            return;
        }
        let g = Arc::new(Gpu::new().unwrap());
        let mut v = Video::open(&g, std::slice::from_ref(&p), || {}).unwrap();
        assert_eq!(v.rings[0].texture(0).format(), wgpu::TextureFormat::P010);
        let (mut held, k) = (None, 30);
        let px = show(&g, &mut v, k, &mut held);
        let bad = px
            .iter()
            .zip(ffmpeg(&p, 0, k, true))
            .filter(|(a, b)| *a != b)
            .count();
        eprintln!(
            "main10: frame {k}: {} bytes, {bad} differ from FFmpeg, number {}",
            px.len(),
            number(&px, 640, 2)
        );
        assert_eq!((px.len(), bad, number(&px, 640, 2)), (640 * 360 * 3, 0, k));
        v.release(&g, &held.unwrap());
        assert_eq!(v.err(), None);
    }

    /// Seeks through the clip the way the player does (seek, then `take` at the target's time) and checks the
    /// frame shown; frame 100 is read back from both lenses' textures and compared with FFmpeg.
    #[test]
    fn seek_exact() {
        let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
            eprintln!("X5_SAMPLES not set: skipping");
            return;
        };
        let p = d.join("v635.insv");
        let g = Arc::new(Gpu::new().unwrap());
        let mut v = Video::open(&g, std::slice::from_ref(&p), || {}).unwrap();
        assert_eq!((v.rings.len(), v.times.len()), (2, 175));
        let (mut held, mut checked): (Option<Pic>, bool) = (None, false);
        // repeated and backward targets, inside and past the decoded queue
        for k in [0, 59, 60, 100, 174, 100, 101, 102, 60, 59, 0] {
            let t0 = Instant::now();
            v.seek(&g, k, held.take());
            // seeks to odd frames take decoded frames only
            let done = k % 2 == 1;
            let x = loop {
                if let (Some(x), _) = v.take(v.times[k], done) {
                    break x;
                }
                assert!(t0.elapsed().as_secs() < 10, "seek {k} stalled");
                thread::sleep(Duration::from_millis(1));
            };
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "seek {k}: shown frame {} (decoder frames {:?}) after {ms:.1} ms",
                x.k, x.f
            );
            assert_eq!(x.k, k);
            assert_eq!(x.t, v.times[k]);
            assert!(!done || v.rings.iter().zip(&x.f).all(|(r, &f)| r.ready() > f));
            // nothing later is due at the target's time
            assert_eq!(v.take(v.times[k], false).0, None);
            if k == 100 && !checked {
                checked = true;
                for (r, ring) in v.rings.iter().enumerate() {
                    ring.acquire(&g, x.f[r]);
                    let ours = read(&g, ring, v.view, x.f[r]);
                    let theirs = ffmpeg(&p, r, k, false);
                    let bad = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
                    println!(
                        "frame {k}, track {r}: {} bytes, {bad} differ from FFmpeg",
                        ours.len()
                    );
                    assert_eq!(ours.len(), theirs.len());
                    assert_eq!(bad, 0);
                }
            }
            v.release_before(&g, &x);
            held = Some(x);
        }
        assert!(checked);
        assert_eq!(v.err(), None);
    }
}
