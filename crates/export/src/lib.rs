//! Render jobs (PLAN.md milestones 5 and 7, groups 7, 8, 10, 13, 15) on the shared device. Per frame: decode both
//! lenses (`video::Decoder` on a media thread) -> stitch (template or RAFT seam, colour matching) -> stabilisation
//! (`stab` poses as the stitch pose) -> 360 equirect or a reframed flat view -> effects on the RGBA frame (an editor
//! timeline's looks, adjustments and overlays through crate fx, then `Job::fx`) -> NV12 / P010 -> `video::Encoder` on
//! its own thread -> MP4 (`media::mux`, `sv3d` / `st3d` for 360). Audio: spans played at 1× copy their source's AAC;
//! an editor timeline that is more than a plain cut mixes its clips' audio, time-stretched to their speed, with the
//! music and encodes it (`Edit`). Either way the audio ends with the last video frame. Frame grabs (`Enc::Jpeg`) take
//! the same path into a JPEG (`jpeg`, GPano for 360). `Queue` runs jobs one after another on a background thread with
//! progress and cancel.
//!
//! Output goes to `<path>.part` and is renamed to `path` once complete; an error or a cancel removes it. wgpu errors and
//! panics on the job's threads (render, decode, encode) are caught (error scopes, `catch_unwind`) and fail the job
//! instead of reaching the device's handler or hanging the other threads. A host that reconfigures a surface on the
//! device hands the job's threads its locks around their GPU work (`locks`).

pub mod jpeg;

use media::{
    aacenc,
    audio::Aac,
    insv::{self, Trailer},
    mix::{self, Kind, Mixer, Track},
    mp4::{self, Mp4},
    music::Music,
    mux::Mux,
    stretch::{self, Stretch},
};
use std::{
    collections::HashMap,
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock, RwLockReadGuard,
        atomic::{AtomicBool, AtomicU64, Ordering::Relaxed},
        mpsc,
    },
    thread,
    time::Instant,
};
use stitch::{Calib, Ctx, Flow, Fmt, Frame, Mode, Opts, Pose, Preview, Raft, Stitcher, View, Yuv};
use video::{Cfg, Codec, Decoder, Encoder, Gpu, Packet, Rate, decode::OUT};

/// A frame counts as shown at t this much early (a time rounded to audio samples still shows its frame).
const EPS: f64 = 1e-3;
const CANCELLED: &str = "cancelled";
/// The soundtrack's rate (the mixer's; the X5 records at it) and mixing block, frames.
const R: f64 = mix::RATE;
const BLOCK: u64 = 1024;

/// The host's locks (`locks`): gate, burn.
static LOCKS: OnceLock<(&'static RwLock<()>, &'static Mutex<()>)> = OnceLock::new();

/// Makes every job take a host's locks around its GPU work; call once, before the first job. wgpu fails a surface
/// configure while another thread submits, so a host that retries it holds `gate` exclusively, after `burn` (the app,
/// `GATE` and `BURN` in its main.rs). The job's threads hold `gate` shared around each submit of their own and `burn`
/// around the RAFT model's loading and each flow (burn submits throughout). They never wait for another thread while
/// holding `gate` (a decoder's wait for a slot, ≤ 50 ms, is for a release the renderer makes without it) and take
/// nothing while holding it.
pub fn locks(gate: &'static RwLock<()>, burn: &'static Mutex<()>) {
    let _ = LOCKS.set((gate, burn));
}

fn gate() -> Option<RwLockReadGuard<'static, ()>> {
    LOCKS
        .get()
        .map(|l| l.0.read().unwrap_or_else(PoisonError::into_inner))
}

fn burn() -> Option<MutexGuard<'static, ()>> {
    LOCKS
        .get()
        .map(|l| l.1.lock().unwrap_or_else(PoisonError::into_inner))
}

/// A span of one clip.
#[derive(Clone, Debug)]
pub struct Seg {
    /// The clip: an `.insv` with both lenses as two tracks, or a file per lens; lens order and calibration come
    /// from the first file's trailer.
    pub files: Vec<PathBuf>,
    /// In and out point, s on the clip's timeline: from the frame shown at `t[0]` to the last frame starting before
    /// `t[1]` (`f64::INFINITY`: the end), at least one frame.
    pub t: [f64; 2],
}

/// What a job plays.
pub enum Cut {
    /// Spans of clips one after another at 1×, each source frame once (the clips share a frame rate); their AAC is
    /// copied.
    Segs(Vec<Seg>),
    Edit(Box<Edit>),
}

/// An editor timeline (crate edit). Output frame n, at `tl.out.fps`, shows `tl.frame(n)`: its clip's source frame
/// at that source time (cuts, speed ramps, freeze frames), its view on a flat output (unless `Out::Flat` has a view of
/// its own) and its clip's look.
///
/// Audio: when every clip plays forwards at 1×, unmuted, at gain 1, and there is no music, the clips' AAC is copied.
/// Otherwise each clip's audio follows its speed curve (`media::stretch`: pitch kept, 0.5×–2×, silent at other
/// speeds) under its gain envelope, with 5 ms fades at its ends; freeze frames and muted clips are silent. The music
/// plays at its offset and gain, ducked by the mixer (`media::mix`, sidechain on the clip audio, `edit::DUCK` deep)
/// when `edit::Music::duck`; the mix, through the mixer's limiter, is AAC-encoded (`media::aacenc`).
///
/// Reversed clips are not supported yet: such a job fails.
pub struct Edit {
    /// Fixed and checked, as `edit::Project::tl` keeps it.
    pub tl: edit::Timeline,
    /// The files of `tl.files` entry i (as `Seg::files`, e.g. a recording in two files); missing or empty: the
    /// entry's path. The music is the first file of its entry.
    pub files: Vec<Vec<PathBuf>>,
    /// Look ids (`edit::Clip::look`) to their tables (`fx::Look::cube`, a parsed `.cube`); an unknown id fails the
    /// job. A clip's `amt` is its look's intensity (`fx::Params::mix`).
    pub looks: HashMap<String, fx::Cube>,
    /// Colour adjustments on every frame, under the clip's look.
    pub adjust: fx::Params,
    /// Overlays on every frame, after the colour: image, placement, opacity (watermarks, stickers, text).
    pub over: Vec<(fx::Image, fx::Place, f32)>,
    /// AAC bit rate of a mixed soundtrack, bit/s (stereo: 128000–256000).
    pub abr: u32,
}

impl Edit {
    fn paths(&self, f: usize) -> Vec<PathBuf> {
        match self.files.get(f) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => vec![PathBuf::from(&self.tl.files[f].path)],
        }
    }

    /// Nothing but cuts at 1×: the source audio can be copied.
    fn plain(&self) -> bool {
        self.tl.out.music.is_none()
            && self.tl.clips.iter().all(|c| {
                c.hold == 0.0
                    && !c.rev
                    && !c.mute
                    && c.speed.iter().all(|s| s.1 == 1.0)
                    && c.vol.iter().all(|v| v.1 == 1.0)
            })
    }
}

/// The stitch view of an editor view: yaw, pitch and roll of its orientation; its projection's Panini or planet
/// distance (`Proj::Planet(d)` is `Mode::Planet` at eye distance d, so a tiny planet unfolds continuously). The
/// exporter's mapping, for the editor's preview too.
pub fn view(v: &edit::View) -> View {
    let [yaw, pitch, roll] = v.ypr();
    let (mode, panini, planet) = match v.proj {
        edit::Proj::Persp => (Mode::Perspective, 0.0, 1.0),
        edit::Proj::Panini(d) => (Mode::Perspective, d, 1.0),
        edit::Proj::Planet(d) => (Mode::Planet, 0.0, d),
        edit::Proj::Equirect => (Mode::Equirect, 0.0, 1.0),
    };
    View {
        mode,
        yaw,
        pitch,
        roll,
        fov: v.fov,
        panini,
        planet,
    }
}

/// What the frames show.
pub enum Out {
    /// The whole sphere, equirect `w` × `w / 2` (3840, 5760, 7680; a multiple of 8); `sv3d` / `st3d` in video.
    Pano(u32),
    /// A reframed view, `w` × `h` (even), drawn by `stitch::Preview`: `view(t)` at output time t (s); None: an
    /// editor timeline's views (the default view for spans).
    Flat {
        w: u32,
        h: u32,
        view: Option<Box<dyn FnMut(f64) -> View + Send>>,
    },
}

/// How the frames are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enc {
    /// MP4: H.265 (`deep`: Main 10 from P010) or H.264 (8-bit, up to 4096²), an IDR every `gop` frames.
    Video {
        codec: Codec,
        deep: bool,
        rate: Rate,
        gop: u32,
    },
    /// The first frame as a JPEG at this quality (1..=100), with GPano XMP for `Out::Pano`.
    Jpeg(u8),
}

/// One frame handed to `Job::fx`.
pub struct Fx<'a> {
    /// Record passes here; they run after the stitch (and an editor timeline's effects) and before the conversion for
    /// the encoder.
    pub enc: &'a mut wgpu::CommandEncoder,
    /// The frame, R'G'B': `Rgba16Float` for 10-bit video, else `Rgba8Unorm`; usages TEXTURE_BINDING,
    /// STORAGE_BINDING, RENDER_ATTACHMENT, COPY_SRC, COPY_DST. Change it in place.
    pub tex: &'a wgpu::Texture,
    /// Output frame index and time (s).
    pub k: u64,
    pub t: f64,
}

pub struct Job {
    pub cut: Cut,
    pub path: PathBuf,
    pub out: Out,
    pub enc: Enc,
    /// RAFT seam flow on every frame, else the template seam.
    pub raft: bool,
    /// Colour matching between the lenses: weight of each new fit (`stitch::Opts::colour`); None: off.
    pub colour: Option<f32>,
    /// Stabilisation from the trailer's IMU: lock and smoothing σ (s); None: the camera's own orientation.
    pub stab: Option<(stab::Lock, f64)>,
    /// Effects on each frame before it is encoded (crate fx).
    pub fx: Option<Hook>,
    /// A `Out::Pano` picture that is not a sphere (a flat 2:1 picture a 360 hook made): no `sv3d` / `st3d`, no GPano,
    /// no wrapping across longitude in the effects. False for 360 output.
    pub flat: bool,
}

/// `Job::fx`: called once per frame on the render thread.
pub type Hook = Box<dyn FnMut(Fx<'_>) + Send>;

/// Frames done of all, steady throughput (frames per second after the first frame, which includes RAFT's warm-up),
/// estimated seconds left.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
    pub fps: f64,
    pub eta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    Queued,
    Running(Progress),
    Done(Progress),
    Failed(String),
    Cancelled,
}

/// A source clip, opened.
struct Clip {
    mp4: Vec<Mp4>,
    /// (file, track) of lens 0, lens 1
    lens: [(usize, usize); 2],
    /// presentation time (s) of each frame
    times: Vec<f64>,
    /// timescale, frame duration
    fps: [u32; 2],
    meta: insv::Meta,
    cal: Calib,
    /// (file, track) of the AAC track
    aac: Option<(usize, usize)>,
    poses: Option<Vec<stab::Pose>>,
}

impl Clip {
    fn open(files: &[PathBuf], lock: Option<(stab::Lock, f64)>) -> Result<Clip, String> {
        let f0 = files.first().ok_or("segment without files")?;
        let tr = Trailer::open(f0)
            .map_err(|e| format!("{}: no Insta360 trailer ({e})", f0.display()))?;
        let meta = tr.meta.clone();
        let cal = Calib::parse(&meta.offset)?;
        let (mut mp4, mut v, mut aac) = (vec![], vec![], None);
        for (i, p) in files.iter().enumerate() {
            let m = Mp4::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
            for (t, tr) in m.tracks.iter().enumerate() {
                match tr.codec {
                    mp4::Codec::Hevc { .. } | mp4::Codec::Avc { .. } if v.len() < 2 => {
                        v.push((i, t))
                    }
                    mp4::Codec::Aac { .. } if aac.is_none() => aac = Some((i, t)),
                    _ => {}
                }
            }
            mp4.push(m);
        }
        let [a, b] = v[..] else {
            return Err(format!(
                "{}: needs both lenses (two video tracks, or a file per lens)",
                f0.display()
            ));
        };
        // DUAL_STREAM_TRACK_REVERSE: the first track holds lens 1
        let lens = if meta.stream == 4 { [b, a] } else { [a, b] };
        let tk = |(i, t): (usize, usize)| &mp4[i].tracks[t];
        let t0 = tk(lens[0]);
        let n = t0.samples.len().min(tk(lens[1]).samples.len());
        let d = t0.shift(mp4[lens[0].0].timescale);
        let times: Vec<f64> = t0.samples[..n]
            .iter()
            .map(|x| (x.pts + d) as f64 / t0.timescale as f64)
            .collect();
        let fps = [t0.timescale, t0.samples.first().map_or(0, |x| x.dur)];
        if n == 0 || fps[1] == 0 {
            return Err(format!("{}: empty video track", f0.display()));
        }
        let poses = match lock {
            Some((lock, smooth)) => {
                let err = |e: std::io::Error| format!("{}: trailer: {e}", f0.display());
                let (imu, ex) = (
                    tr.imu().map_err(err)?,
                    tr.frames(insv::EXPOSURE).map_err(err)?,
                );
                if imu.len() < 2 || ex.is_empty() {
                    return Err(format!(
                        "{}: no IMU or exposure data to stabilise with",
                        f0.display()
                    ));
                }
                let o = stab::Opts::new(&meta, lock, smooth);
                Some(stab::poses(
                    &stab::Track::new(&imu, stab::KP, stab::KI),
                    &ex,
                    &o,
                ))
            }
            None => None,
        };
        Ok(Clip {
            mp4,
            lens,
            times,
            fps,
            meta,
            cal,
            aac,
            poses,
        })
    }

    /// The frame shown at source time `s`: the last one starting by `s + e`.
    fn at(&self, s: f64, e: f64) -> usize {
        self.times
            .partition_point(|&x| x <= s + e)
            .saturating_sub(1)
    }

    /// The last frame at or before `k` where both lenses have a sync sample: decoding up to `k` starts there.
    fn key(&self, k: usize) -> usize {
        let tr = self.lens.map(|(i, t)| &self.mp4[i].tracks[t]);
        (0..=k)
            .rev()
            .find(|&k| tr.iter().all(|t| t.samples[k].sync))
            .unwrap_or(0)
    }

    /// The stitch pose of frame `k`: stabilisation, then the camera's framing (the X5 frames on lens 1, which
    /// reversed tracks put first).
    fn pose(&self, k: usize) -> Pose {
        let f = Pose::euler(0.0, 0.0, if self.meta.stream == 4 { 180.0 } else { 0.0 });
        match &self.poses {
            Some(p) => {
                let p = p[k.min(p.len() - 1)];
                Pose {
                    q: p.q * f.q,
                    rs: [p.rs; 2],
                }
            }
            None => f,
        }
    }
}

/// One decode pass over a clip: the source frame each output frame shows, non-decreasing.
struct Run {
    c: usize,
    k: Vec<usize>,
}

/// What a job renders: its clips, opened; the decode passes in output order; the spans whose AAC a copy takes
/// (clip, source time of the first output frame, output frames: one per segment or timeline clip); the output frame
/// rate (timescale, frame duration); for an editor timeline the clip of each file entry.
struct Plan {
    clips: Vec<Clip>,
    runs: Vec<Run>,
    spans: Vec<(usize, f64, u64)>,
    fps: [u32; 2],
    of: Vec<usize>,
}

fn plan(cut: &Cut, lock: Option<(stab::Lock, f64)>) -> Result<Plan, String> {
    let e = match cut {
        Cut::Segs(segs) => {
            let clips = (segs.iter())
                .map(|s| Clip::open(&s.files, lock))
                .collect::<Result<Vec<_>, _>>()?;
            let fps = clips.first().ok_or("job without segments")?.fps;
            if let Some(c) = clips.iter().find(|c| c.fps != fps) {
                return Err(format!("frame rates differ: {:?} vs {:?}", c.fps, fps));
            }
            let (mut runs, mut spans) = (vec![], vec![]);
            for (c, (s, cl)) in segs.iter().zip(&clips).enumerate() {
                let k0 = cl.at(s.t[0], EPS);
                let k1 = cl.times.partition_point(|&x| x < s.t[1] - EPS).max(k0 + 1);
                runs.push(Run {
                    c,
                    k: (k0..k1).collect(),
                });
                spans.push((c, cl.times[k0], (k1 - k0) as u64));
            }
            return Ok(Plan {
                clips,
                runs,
                spans,
                fps,
                of: vec![],
            });
        }
        Cut::Edit(e) => e,
    };
    let tl = &e.tl;
    tl.check().map_err(|x| format!("timeline: {x}"))?;
    if tl.find(0).is_none() {
        return Err("empty timeline (or not fixed: `Timeline::fix`)".into());
    }
    for (k, c) in tl.clips.iter().enumerate() {
        if c.rev && c.hold == 0.0 {
            return Err(format!(
                "clip {k}: reversed clips are not supported by export yet"
            ));
        }
        if !c.look.is_empty() && !e.looks.contains_key(&c.look) {
            return Err(format!("clip {k}: no look {:?}", c.look));
        }
    }
    let (mut clips, mut of) = (vec![], vec![usize::MAX; tl.files.len()]);
    for c in &tl.clips {
        let f = c.file as usize;
        if of[f] == usize::MAX {
            of[f] = clips.len();
            clips.push(Clip::open(&e.paths(f), lock)?);
        }
    }
    // a frame shows the source frame with the latest pts ≤ its source time + 1 µs (crate edit)
    let at = |n: u64| tl.frame(n).map(|x| x.src).unwrap_or_default();
    let mut runs: Vec<Run> = vec![];
    for s in tl.segs() {
        let (c, cl) = (of[s.file as usize], &clips[of[s.file as usize]]);
        let k: Vec<usize> = s.frames.map(|n| cl.at(at(n), 1e-6)).collect();
        match runs.last_mut() {
            // decoding on is no longer than from the sync frame a new pass would start at
            Some(r) if r.c == c && r.k.last().is_some_and(|&l| k[0] >= l && cl.key(k[0]) <= l) => {
                r.k.extend(k)
            }
            _ => runs.push(Run { c, k }),
        }
    }
    let spans = (0..tl.clips.len())
        .map(|k| {
            let r = tl.frames(k);
            (of[tl.clips[k].file as usize], at(r.start), r.end - r.start)
        })
        .collect();
    Ok(Plan {
        clips,
        runs,
        spans,
        fps: [tl.out.fps.num, tl.out.fps.den],
        of,
    })
}

/// AAC frames to copy, in output order, placed back to back from `start` (track ticks on the output timeline, ≤ 0:
/// the muxer's edit list cuts the part before the first video frame). Each span contributes the frames overlapping
/// its exported time; at a span boundary the next one starts with the frame nearest the running end, so audio stays
/// within half a frame of the video.
struct Audio {
    codec: mp4::Codec,
    ts: u32,
    start: i64,
    /// (clip, sample)
    frames: Vec<(usize, usize)>,
}

/// `fd`: the output frame duration, s.
fn audio(p: &Plan, fd: f64) -> Result<Option<Audio>, String> {
    let Some((i, t)) = p.clips[p.spans[0].0].aac else {
        return Ok(None);
    };
    let first = &p.clips[p.spans[0].0].mp4[i].tracks[t];
    let (codec, ts) = (first.codec.clone(), first.timescale);
    let (mut start, mut out, mut run) = (None, 0.0, 0i64);
    let mut frames = vec![];
    for (j, &(c, v0, n)) in p.spans.iter().enumerate() {
        let cl = &p.clips[c];
        let tr = cl.aac.map(|(i, t)| (&cl.mp4[i], &cl.mp4[i].tracks[t]));
        let Some((m, tr)) = tr.filter(|(_, tr)| tr.codec == codec && tr.timescale == ts) else {
            return Err(format!(
                "segment {j}: no AAC track like the first segment's"
            ));
        };
        let sh = tr.shift(m.timescale);
        let ss = &tr.samples;
        let at = |s: &mp4::Sample| (s.pts + sh) as f64 / ts as f64;
        let v1 = v0 + n as f64 * fd;
        let mut s0 = ss.partition_point(|s| at(s) <= v0).saturating_sub(1);
        let mut s1 = ss.partition_point(|s| at(s) < v1);
        // output ticks of sample s
        let o = |s: usize| ((at(&ss[s]) - v0 + out) * ts as f64).round() as i64;
        match start {
            None => {
                run = o(s0);
                start = Some(run);
            }
            Some(_) => {
                while s0 + 1 < s1 && o(s0) + (ss[s0].dur as i64) / 2 < run {
                    s0 += 1;
                }
            }
        }
        run += ss[s0..s1].iter().map(|s| s.dur as i64).sum::<i64>();
        out += n as f64 * fd;
        // the last span's frames reach the video's end (back to back they may fall short; the muxer cuts the rest)
        while j + 1 == p.spans.len() && s1 < ss.len() && run < (out * ts as f64).round() as i64 {
            run += ss[s1].dur as i64;
            s1 += 1;
        }
        frames.extend((s0..s1).map(|s| (c, s)));
    }
    Ok(Some(Audio {
        codec,
        ts,
        start: start.unwrap_or(0),
        frames,
    }))
}

/// A clip's audio while it plays: its source AAC decoded in order from the clip's start, fed to the time-stretch
/// one output frame at a time with the speed the timeline gives that frame (source s over output s), so the sound
/// follows the pictures through ramps.
struct Voice {
    /// timeline clip, its output frames, the next one to feed
    k: usize,
    n: Range<u64>,
    j: u64,
    /// clip, (file, track) of its AAC, the track's shift to presentation time
    c: usize,
    tr: (usize, usize),
    sh: i64,
    dec: Aac,
    /// next packet; decoded stereo from source frame `s0` (presentation time × 48 kHz)
    i: usize,
    pkt: Vec<u8>,
    src: Vec<f32>,
    s0: i64,
    st: Stretch,
    chunk: Vec<f32>,
    /// stretched output from output frame `y0`; the stretch finished
    y: Vec<f32>,
    y0: u64,
    done: bool,
}

impl Voice {
    fn new(tl: &edit::Timeline, p: &Plan, k: usize) -> Result<Voice, String> {
        let c = p.of[tl.clips[k].file as usize];
        let cl = &p.clips[c];
        let (fi, ti) = cl.aac.ok_or("no audio")?;
        let (m, tr) = (&cl.mp4[fi], &cl.mp4[fi].tracks[ti]);
        let mp4::Codec::Aac { asc, .. } = &tr.codec else {
            unreachable!("Clip::open takes AAC only")
        };
        let dec = Aac::new(asc).map_err(|e| format!("clip {k}: AAC: {e}"))?;
        if dec.rate as f64 != R {
            return Err(format!(
                "clip {k}: audio at {} Hz, the mix runs at 48 kHz",
                dec.rate
            ));
        }
        let sh = tr.shift(m.timescale);
        let ts = tr.timescale as i64;
        let s = (tl.content(k, 0.0) * R).round() as i64;
        // the packet holding s, and one before it for the decoder's overlap
        let i = (tr.samples)
            .partition_point(|x| (x.pts + sh) * 48000 / ts <= s)
            .saturating_sub(2);
        let n = tl.frames(k);
        Ok(Voice {
            k,
            j: n.start,
            y0: (tl.out.fps.t(n.start) * R).round() as u64,
            n,
            c,
            tr: (fi, ti),
            sh,
            dec,
            i,
            pkt: vec![],
            src: vec![],
            s0: tr.samples.get(i).map_or(s, |x| (x.pts + sh) * 48000 / ts),
            st: Stretch::new(2),
            chunk: vec![],
            y: vec![],
            done: false,
        })
    }

    /// Stretches on until the output reaches output frame `to` or the clip's end.
    fn fill(&mut self, tl: &edit::Timeline, p: &Plan, to: u64) -> Result<(), String> {
        let fps = tl.out.fps;
        let (m, tr) = (
            &p.clips[self.c].mp4[self.tr.0],
            &p.clips[self.c].mp4[self.tr.0].tracks[self.tr.1],
        );
        let ts = tr.timescale as i64;
        while !self.done && self.y0 + (self.y.len() / 2) as u64 <= to {
            if self.j == self.n.end {
                self.st.finish(&mut self.y);
                self.done = true;
                break;
            }
            let (ta, tb) = (
                fps.t(self.j - self.n.start),
                fps.t(self.j + 1 - self.n.start),
            );
            let (sa, sb) = (tl.content(self.k, ta), tl.content(self.k, tb));
            let (a, b) = ((sa * R).round() as i64, (sb * R).round() as i64);
            while self.s0 + (self.src.len() / 2) as i64 <= b && self.i < tr.samples.len() {
                let x = tr.samples[self.i];
                self.i += 1;
                // a gap in the packets: silence
                let gap = (x.pts + self.sh) * 48000 / ts - self.s0 - (self.src.len() / 2) as i64;
                self.src
                    .resize(self.src.len() + 2 * gap.max(0) as usize, 0.0);
                m.read(&x, &mut self.pkt)
                    .map_err(|e| format!("clip {}: audio: {e}", self.k))?;
                let ch = self.dec.ch.max(1) as usize;
                match self.dec.decode(&self.pkt) {
                    // mono on both sides, else the first two channels
                    Ok(pcm) => (pcm.chunks_exact(ch))
                        .for_each(|f| self.src.extend_from_slice(&[f[0], f[ch.min(2) - 1]])),
                    // a damaged packet: silence for it
                    Err(_) => self.src.resize(self.src.len() + 2 * x.dur as usize, 0.0),
                }
            }
            self.chunk.clear();
            for f in a..b {
                let i = (f - self.s0) as usize;
                self.chunk
                    .extend_from_slice(self.src.get(2 * i..2 * i + 2).unwrap_or(&[0.0; 2]));
            }
            let d = ((b - self.s0).max(0) as usize).min(self.src.len() / 2);
            self.src.drain(..2 * d);
            self.s0 += d as i64;
            // a speed at a bound by rounding stays inside
            let mut v = (sb - sa) / (tb - ta);
            for x in [stretch::MIN, stretch::MAX] {
                if (v - x).abs() < 1e-9 * x {
                    v = x;
                }
            }
            self.st.run(&self.chunk, v, &mut self.y);
            self.j += 1;
        }
        Ok(())
    }
}

/// An editor timeline's soundtrack, mixed and AAC-encoded block by block as the video is muxed: a mixer track per clip
/// with sound (`Voice`, made while the clip plays) and one for the music.
struct Mixdown<'a> {
    tl: &'a edit::Timeline,
    p: &'a Plan,
    mx: Mixer,
    /// per clip track of `mx`, in order (the music's comes last): timeline clip, output frames, voice while it plays
    vs: Vec<(usize, [u64; 2], Option<Voice>)>,
    /// the music, the output frame its next frame plays at
    music: Option<(Music, u64)>,
    enc: aacenc::Enc,
    /// mixer input frames done; the soundtrack's frames (the video's length)
    at: u64,
    total: u64,
    o: Vec<f32>,
}

impl<'a> Mixdown<'a> {
    /// None without any sound.
    fn new(e: &'a Edit, p: &'a Plan) -> Result<Option<Mixdown<'a>>, String> {
        let (tl, fps) = (&e.tl, e.tl.out.fps);
        let (mut mx, mut vs) = (Mixer::new(), vec![]);
        for (k, c) in tl.clips.iter().enumerate() {
            if c.hold > 0.0 || c.mute || p.clips[p.of[c.file as usize]].aac.is_none() {
                continue;
            }
            let n = tl.frames(k);
            let span = [fps.t(n.start), fps.t(n.end)];
            let mut t = Track::new(Kind::Clip, span);
            t.fade = [0.005; 2];
            // the envelope on content time, sampled per output frame
            if !c.vol.is_empty() {
                t.env = (n.clone())
                    .filter_map(|f| Some((fps.t(f), tl.frame(f)?.gain)))
                    .collect();
            }
            mx.tracks.push(t);
            vs.push((k, span.map(|s| (s * R).round() as u64), None));
        }
        let mut music = None;
        if let Some(m) = &tl.out.music {
            let path = e.paths(m.file as usize).swap_remove(0);
            let err = |x: std::io::Error| format!("music {}: {x}", path.display());
            let mut mu = Music::open(&path).map_err(err)?;
            let len = if mu.duration > 0.0 {
                mu.duration
            } else {
                tl.files[m.file as usize].dur
            };
            // music time = output time + offset
            let span = [(-m.offset).max(0.0), tl.dur().min(len - m.offset)];
            if span[1] > span[0] {
                mu.seek(span[0] + m.offset).map_err(err)?;
                let mut t = Track::new(Kind::Music, span);
                t.gain = m.gain;
                mx.tracks.push(t);
                // the mixer ducks: edit's own factor (`Eval::music`) would duck twice
                mx.duck.depth = if m.duck { edit::DUCK } else { 1.0 };
                music = Some((mu, (span[0] * R).round() as u64));
            }
        }
        if vs.is_empty() && music.is_none() {
            return Ok(None);
        }
        Ok(Some(Mixdown {
            tl,
            p,
            mx,
            vs,
            music,
            enc: aacenc::Enc::new(R as u32, 2, e.abr).map_err(|x| format!("AAC encoder: {x}"))?,
            at: 0,
            total: (tl.dur() * R).round() as u64,
            o: vec![0.0; 2 * BLOCK as usize],
        }))
    }

    /// Mixes and encodes up to output frame `to` (the end at most), writing the AAC frames to track `t`; at the end
    /// (`to` = u64::MAX) also the encoder's last frames.
    fn run(&mut self, to: u64, m: &mut Mux, t: usize) -> Result<(), String> {
        let (lat, mut r) = (mix::LATENCY as u64, Ok(()));
        // output frame i of a block is its input frame i − LATENCY
        let end = (self.total + lat).min(to.saturating_add(lat));
        while self.at < end {
            let (p, n) = (self.at, (end - self.at).min(BLOCK));
            let q = p + n;
            for (j, (k, span, v)) in self.vs.iter_mut().enumerate() {
                let b = &mut self.mx.tracks[j].buf;
                b.clear();
                if q <= span[0] || p >= span[1] {
                    *v = None;
                    continue;
                }
                let v = match v {
                    Some(v) => v,
                    None => v.insert(Voice::new(self.tl, self.p, *k)?),
                };
                v.fill(self.tl, self.p, q)?;
                b.resize(2 * n as usize, 0.0);
                let s = p.max(v.y0);
                let (i, len) = ((s - v.y0) as usize, (q - s) as usize);
                let got = (v.y.len() / 2).saturating_sub(i).min(len);
                b[2 * (s - p) as usize..][..2 * got].copy_from_slice(&v.y[2 * i..2 * (i + got)]);
                let d = (q.saturating_sub(v.y0) as usize).min(v.y.len() / 2);
                v.y.drain(..2 * d);
                v.y0 += d as u64;
            }
            if let Some((mu, mt)) = &mut self.music {
                let b = &mut self.mx.tracks[self.vs.len()].buf;
                b.clear();
                b.resize(2 * n as usize, 0.0);
                if q > *mt {
                    let s = mt.saturating_sub(p) as usize;
                    let got = mu
                        .read(&mut b[2 * s..])
                        .map_err(|x| format!("music: {x}"))?;
                    b.truncate(2 * (s + got));
                    *mt = q;
                }
            }
            self.mx.run(p as f64 / R, &mut self.o[..2 * n as usize]);
            let s = 2 * lat.saturating_sub(p).min(n) as usize;
            self.enc.push(&self.o[s..2 * n as usize], |au, pts, d| {
                if r.is_ok() {
                    r = m.write(t, au, pts, pts, d, true);
                }
            });
            self.at = q;
        }
        if to == u64::MAX {
            self.enc.finish(|au, pts, d| {
                if r.is_ok() {
                    r = m.write(t, au, pts, pts, d, true);
                }
            });
            m.end(t, self.total as i64);
        }
        r.map_err(|e| format!("mux: {e}"))
    }
}

/// The soundtrack: AAC frames copied, or an editor timeline's mix.
enum Sound<'a> {
    Copy(Audio),
    Mix(Box<Mixdown<'a>>),
}

fn panicked(x: Box<dyn std::any::Any + Send>) -> String {
    let m = (x.downcast_ref::<String>().cloned())
        .or_else(|| x.downcast_ref::<&str>().map(|s| s.to_string()));
    format!("panic: {}", m.unwrap_or_default())
}

/// Runs `f` with this thread's wgpu errors and panics captured: they come back as `Err` instead of reaching the
/// device's uncaptured-error handler (the app's panics) or unwinding past the job's cleanup. A panic must not end a
/// job's thread silently: the others would wait for it (a decoder spinning on a full ring, the renderer on the
/// encoder's input).
fn scoped<T>(dev: &wgpu::Device, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    use wgpu::ErrorFilter as E;
    let s = [E::Validation, E::OutOfMemory, E::Internal].map(|f| dev.push_error_scope(f));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|x| Err(panicked(x)));
    let mut e = None;
    for g in s.into_iter().rev() {
        if let Some(x) = pollster::block_on(g.pop()) {
            e.get_or_insert(x);
        }
    }
    match e {
        Some(e) => Err(format!("wgpu: {e}")),
        None => r,
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

/// The media thread of one decode pass: decodes both lenses from the last sync sample before the first frame
/// through the last and hands over the decoder indices of each frame from the first on, also of those the pass skips
/// (faster than 1×): a release frees every earlier frame too, so only the renderer, which holds them in order, may
/// release once it has one. Frames before the first are released here, unseen.
fn decode(
    g: &Gpu,
    c: &Clip,
    ks: &[usize],
    mut d: [Decoder; 2],
    tx: &mpsc::SyncSender<Result<(usize, [u64; 2]), String>>,
    quit: &AtomicBool,
) -> Result<(), String> {
    let tr = c.lens.map(|(i, t)| &c.mp4[i].tracks[t]);
    let nal = tr.map(|t| match t.codec {
        mp4::Codec::Hevc { nal, .. } | mp4::Codec::Avc { nal, .. } => nal as usize,
        _ => 4,
    });
    let mut buf = vec![];
    for k in c.key(ks[0])..=ks[ks.len() - 1] {
        let mut f = [0; 2];
        for l in 0..2 {
            let (m, x) = (&c.mp4[c.lens[l].0], tr[l].samples[k]);
            m.read(&x, &mut buf)
                .map_err(|e| format!("frame {k}: {e}"))?;
            let mut au = [&[][..]; 64];
            let n = nals(&buf, nal[l], &mut au);
            f[l] = loop {
                // its flush submits on wgpu's queue; the slot it waits for (≤ 50 ms) the renderer releases without
                // the gate
                let r = {
                    let _g = gate();
                    d[l].decode(g, &au[..n])
                };
                match r {
                    Ok(x) => break x,
                    Err(video::Error::Full) if !quit.load(Relaxed) => {}
                    Err(video::Error::Full) => return Ok(()),
                    Err(e) => return Err(format!("decode frame {k}, lens {l}: {e}")),
                }
            };
        }
        if k < ks[0] {
            for (x, f) in d.iter().zip(f) {
                x.release(g, f);
            }
        } else if tx.send(Ok((k, f))).is_err() {
            break;
        }
    }
    Ok(())
}

/// The encoder thread: encodes the frames the renderer releases, muxes the packets and the soundtrack up to each
/// packet's end, ends the soundtrack with the last frame, finishes the file (`moov` in front).
#[allow(clippy::too_many_arguments)]
fn mux(
    g: &Gpu,
    mut e: Encoder,
    rx: mpsc::Receiver<u64>,
    part: &Path,
    c: &Cfg,
    sph: bool,
    clips: &[Clip],
    mut au: Option<Sound>,
    quit: &AtomicBool,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{}: {e}", part.display());
    let mut m = Mux::create(part).map_err(io)?;
    let ps = e.params.clone();
    let codec = match c.codec {
        Codec::H265 => mp4::Codec::Hevc { ps, nal: 4 },
        Codec::H264 => mp4::Codec::Avc { ps, nal: 4 },
    };
    let v = m.track(codec, c.fps[0], c.w, c.h, sph).map_err(io)?;
    let a = match &au {
        Some(Sound::Copy(a)) => Some(m.track(a.codec.clone(), a.ts, 0, 0, false)),
        Some(Sound::Mix(x)) => Some(m.track(
            mp4::Codec::Aac {
                asc: x.enc.asc().to_vec(),
                rate: R as u32,
                ch: 2,
            },
            R as u32,
            0,
            0,
            false,
        )),
        None => None,
    }
    .transpose()
    .map_err(io)?;
    let d = c.fps[1] as i64;
    let (mut p, mut b, mut s) = (Packet::default(), vec![], vec![]);
    let (mut ai, mut vend) = (0, 0i64);
    let mut at = match &au {
        Some(Sound::Copy(a)) => a.start,
        _ => 0,
    };
    // a video packet, then the soundtrack up to its end (None: the rest)
    let mut put = |m: &mut Mux, p: Option<&Packet>| -> Result<(), String> {
        if let Some(p) = p {
            b.clear();
            p.prefixed(&mut b);
            m.write(v, &b, p.dts * d, p.pts * d, d as u32, p.key)
                .map_err(io)?;
            vend = vend.max((p.pts + 1) * d);
        }
        // track ticks of video ticks t
        let tk = |t: i64, ts: u32| {
            ((t as i128 * ts as i128 + c.fps[0] as i128 / 2) / c.fps[0] as i128) as i64
        };
        match (a, &mut au) {
            (Some(t), Some(Sound::Copy(a))) => {
                let end = p.map_or(i64::MAX, |p| tk((p.pts + 1) * d, a.ts));
                while ai < a.frames.len() && at < end {
                    let (ci, si) = a.frames[ai];
                    let (fi, ti) = clips[ci].aac.unwrap();
                    let x = clips[ci].mp4[fi].tracks[ti].samples[si];
                    clips[ci].mp4[fi].read(&x, &mut s).map_err(io)?;
                    m.write(t, &s, at, at, x.dur, true).map_err(io)?;
                    (ai, at) = (ai + 1, at + x.dur as i64);
                }
                if p.is_none() {
                    m.end(t, tk(vend, a.ts));
                }
                Ok(())
            }
            (Some(t), Some(Sound::Mix(x))) => match p {
                Some(p) => x.run(tk((p.pts + 1) * d, R as u32) as u64, m, t),
                None => x.run(u64::MAX, m, t),
            },
            _ => Ok(()),
        }
    };
    let err = |x: video::Error| format!("encoder: {x}");
    for n in rx {
        loop {
            // its flush submits on wgpu's queue
            let r = {
                let _g = gate();
                e.encode(g, n as i64)
            };
            match r {
                Ok(_) => break,
                Err(video::Error::Full) => {
                    if e.packet(&mut p, 50_000_000).map_err(err)? {
                        put(&mut m, Some(&p))?;
                    } else if quit.load(Relaxed) {
                        return Err(CANCELLED.into());
                    }
                }
                Err(x) => return Err(format!("encode frame {n}: {x}")),
            }
        }
        while e.packet(&mut p, 0).map_err(err)? {
            put(&mut m, Some(&p))?;
        }
    }
    if quit.load(Relaxed) {
        return Err(CANCELLED.into());
    }
    while e.packet(&mut p, u64::MAX).map_err(err)? {
        put(&mut m, Some(&p))?;
    }
    put(&mut m, None)?;
    m.finish(true).map_err(io)
}

/// What draws a clip's frames: the stitcher (the 360 output, or only the seam flow of a flat view) and the flat
/// view's preview.
struct Draw {
    st: Option<Stitcher>,
    pv: Option<Preview>,
    /// the frame (`st.out` for 360), a view of it, its YUV converter
    tex: wgpu::Texture,
    tv: wgpu::TextureView,
    yuv: Option<Yuv>,
}

impl Draw {
    /// `j`: the job's output, encoding, RAFT, colour matching.
    fn new(
        cx: &Ctx,
        c: &Clip,
        fr: Frame,
        j: (&Out, Enc, bool, Option<f32>),
        flat: Option<&wgpu::Texture>,
    ) -> Result<Draw, String> {
        let deep = matches!(j.1, Enc::Video { deep: true, .. });
        let fmt = matches!(j.1, Enc::Video { .. }).then_some(Fmt { deep, chroma: 0 });
        let o = Opts {
            blend: c.meta.blend.unwrap_or(6.0),
            flow: j.2.then_some(Flow::RAFT),
            colour: j.3,
            wide: deep,
            yuv: None,
        };
        let (st, pv, tex) = match (j.0, flat) {
            (Out::Pano(w), _) => {
                let st = Stitcher::new(cx, &c.cal, fr, *w, w / 2, o)?;
                let t = st.out.clone();
                (Some(st), None, t)
            }
            (Out::Flat { .. }, Some(t)) if j.2 => {
                // the view is drawn from the lenses through the stitcher's flow: its own output stays unused
                let st = Stitcher::new(cx, &c.cal, fr, 8, 2, o)?;
                let pv = Preview::seam(&st, t.format());
                (Some(st), Some(pv), t.clone())
            }
            (Out::Flat { .. }, Some(t)) => {
                let pv = Preview::new(cx, &c.cal, fr, o.blend, o.colour, t.format());
                (None, Some(pv), t.clone())
            }
            (Out::Flat { .. }, None) => return Err("flat output without a target".into()),
        };
        let tv = tex.create_view(&Default::default());
        let yuv = match fmt {
            Some(f) => Some(Yuv::new(cx, &tv, f, pv.is_none())?),
            None => None,
        };
        Ok(Draw {
            st,
            pv,
            tex,
            tv,
            yuv,
        })
    }

    fn pose(&self, p: &Pose) {
        // a seam preview shares its stitcher's pose
        match (&self.pv, &self.st) {
            (Some(pv), _) => pv.pose(p),
            (None, Some(st)) => st.pose(p),
            (None, None) => {}
        }
    }

    /// `fresh`: a source frame not drawn before (a repeat, e.g. a freeze frame, keeps the colour fit).
    fn draw(
        &self,
        enc: &mut wgpu::CommandEncoder,
        src: [&stitch::Src; 2],
        v: Option<&View>,
        fresh: bool,
    ) {
        match (&self.pv, &self.st, v) {
            (Some(pv), _, Some(v)) => {
                if fresh {
                    pv.colour(enc, src);
                }
                pv.render(enc, &self.tv, [self.tex.width(), self.tex.height()], v, src);
            }
            (None, Some(st), _) => st.encode(enc, src),
            _ => {}
        }
    }
}

/// Runs `job` on this thread, the render thread; a media thread per decode pass and an encoder thread run beside
/// it. `raft` is created on the first RAFT job and kept: attach burn to the device (`stitch::attach`) before
/// installing the device's error handler, or `Raft::new` replaces it. `stop` cancels (checked every frame);
/// `progress` sees every frame. Returns the final progress. Errors (a cancel: "cancelled"; a panic, e.g. in
/// `Job::fx`: "panic: …") remove the partial output; it never panics.
pub fn run(
    g: &Gpu,
    cx: &Ctx,
    raft: &mut Option<Raft>,
    mut job: Job,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(Progress),
) -> Result<Progress, String> {
    let mut part = job.path.clone().into_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    // also a panic while setting up (a damaged clip's parser, the model) ends as an error
    let r = scoped(&g.device, || {
        let mut p = plan(&job.cut, job.stab)?;
        if matches!(job.enc, Enc::Jpeg(_)) {
            p.runs.truncate(1);
            p.runs[0].k.truncate(1);
        }
        let (w, h) = match &job.out {
            Out::Pano(w) => (*w, w / 2),
            Out::Flat { w, h, .. } => (*w, *h),
        };
        let max = g.device.limits().max_texture_dimension_2d;
        if w == 0 || h == 0 || w > max || h > max {
            return Err(format!("output {w}x{h} outside 1..={max}"));
        }
        if job.raft && raft.is_none() {
            let _b = burn();
            *raft = Some(Raft::new(&g.instance, &g.adapter, &g.device, &g.queue)?);
        }
        render(
            g,
            cx,
            // the model stays loaded between jobs; a template stitcher has no belt for it
            raft.as_ref().filter(|_| job.raft),
            &mut job,
            &p,
            (w, h),
            &part,
            stop,
            progress,
        )
    });
    match &r {
        Ok(_) => {
            std::fs::rename(&part, &job.path).map_err(|e| format!("{}: {e}", job.path.display()))?
        }
        Err(_) => _ = std::fs::remove_file(&part),
    }
    r
}

#[allow(clippy::too_many_arguments)]
fn render(
    g: &Gpu,
    cx: &Ctx,
    raft: Option<&Raft>,
    job: &mut Job,
    pl: &Plan,
    (w, h): (u32, u32),
    part: &Path,
    stop: &AtomicBool,
    progress: &mut dyn FnMut(Progress),
) -> Result<Progress, String> {
    let quit = AtomicBool::new(false);
    let fps = pl.fps;
    // output frame n starts at n·fd (one rounding, as `edit::Rate::t`)
    let at = |n: u64| (n as f64 * fps[1] as f64) / fps[0] as f64;
    let cfg = match job.enc {
        Enc::Video {
            codec,
            deep,
            rate,
            gop,
        } => Some(Cfg {
            codec,
            deep,
            w,
            h,
            fps,
            gop: gop.max(1),
            rate,
            vbv: 1000,
            quality: 0,
            full: false,
            chroma: 0,
        }),
        Enc::Jpeg(_) => None,
    };
    let format = match cfg {
        Some(Cfg { deep: true, .. }) => wgpu::TextureFormat::Rgba16Float,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    };
    let tex = |label, usage| {
        g.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    };
    use wgpu::TextureUsages as U;
    let flat = matches!(job.out, Out::Flat { .. }).then(|| {
        tex(
            "export frame",
            U::TEXTURE_BINDING
                | U::STORAGE_BINDING
                | U::RENDER_ATTACHMENT
                | U::COPY_SRC
                | U::COPY_DST,
        )
    });
    let sph = matches!(job.out, Out::Pano(_)) && !job.flat;
    let ed = match &job.cut {
        Cut::Edit(e) => Some(&**e),
        Cut::Segs(_) => None,
    };
    let (au, enc) = match &cfg {
        Some(c) => (
            match ed {
                None => audio(pl, at(1))?.map(Sound::Copy),
                Some(e) => match e.plain().then(|| audio(pl, at(1))) {
                    Some(Ok(Some(a))) => Some(Sound::Copy(a)),
                    _ => Mixdown::new(e, pl)?.map(|x| Sound::Mix(Box::new(x))),
                },
            },
            Some(Encoder::new(g, c).map_err(|e| format!("encoder: {e}"))?),
        ),
        None => (None, None),
    };
    // crate fx on an editor timeline's frames: its state and the copy of the frame it reads, the overlays
    let fxc = ed
        .filter(|e| {
            e.adjust != fx::Params::default()
                || !e.over.is_empty()
                || e.tl.clips.iter().any(|c| !c.look.is_empty())
        })
        .map(|_| fx::Ctx::new(&g.device, &g.queue));
    let mut gfx = (fxc.as_ref()).map(|x| {
        (
            fx::Fx::new(x, w, h, sph),
            tex("fx input", U::TEXTURE_BINDING | U::COPY_DST),
        )
    });
    let over: Vec<fx::Overlay> = match (&fxc, ed) {
        (Some(x), Some(e)) => (e.over.iter())
            .map(|(i, p, o)| fx::Overlay::new(x, i, *p, *o))
            .collect(),
        _ => vec![],
    };
    let ovr: Vec<&fx::Overlay> = over.iter().collect();
    // the timeline clip whose look is set
    let mut look: Option<usize> = None;
    let input = enc.as_ref().map(|e| e.input.clone());
    let total = pl.runs.iter().map(|r| r.k.len() as u64).sum::<u64>();
    let (mut p, mut t1) = (
        Progress {
            total,
            ..Default::default()
        },
        None,
    );
    let quit = &quit;
    thread::scope(|sc| {
        let (etx, erx) = mpsc::channel::<u64>();
        let et = enc.map(|e| {
            let c = cfg.as_ref().unwrap();
            let clips = &pl.clips[..];
            sc.spawn(move || {
                let r = scoped(&g.device, || mux(g, e, erx, part, c, sph, clips, au, quit));
                if r.is_err() {
                    quit.store(true, Relaxed);
                }
                r
            })
        });
        let r = scoped(&g.device, || -> Result<(), String> {
            let mut n = 0u64;
            for run in &pl.runs {
                let c = &pl.clips[run.c];
                let mut d = vec![];
                for (i, t) in c.lens {
                    let tr = &c.mp4[i].tracks[t];
                    let (codec, ps) = match &tr.codec {
                        mp4::Codec::Hevc { ps, .. } => (Codec::H265, ps),
                        mp4::Codec::Avc { ps, .. } => (Codec::H264, ps),
                        _ => return Err("not a video track".into()),
                    };
                    d.push(
                        Decoder::new(g, codec, ps, tr.width, tr.height)
                            .map_err(|e| format!("decoder: {e}"))?,
                    );
                }
                let [d0, d1]: [Decoder; 2] = d.try_into().map_err(|_| "decoders")?;
                let fr = Frame {
                    w: d0.view[2],
                    h: d0.view[3],
                    crop: c.meta.crop,
                    shift: c.meta.shift,
                };
                let rings = [d0.ring.clone(), d1.ring.clone()];
                let bg: [Vec<stitch::Src>; 2] = [&d0, &d1].map(|d| {
                    (0..OUT as u64)
                        .map(|s| cx.rect(d.texture(s), d.view))
                        .collect()
                });
                let j = (&job.out, job.enc, job.raft, job.colour);
                let mut dr = scoped(&g.device, || Draw::new(cx, c, fr, j, flat.as_ref()))?;
                let io = gfx.as_ref().map(|(f, t)| f.io(t, &dr.tex));
                let (tx, rx) = mpsc::sync_channel(OUT);
                let ks = &run.k[..];
                sc.spawn(move || {
                    if let Err(e) = scoped(&g.device, || decode(g, c, ks, [d0, d1], &tx, quit)) {
                        let _ = tx.send(Err(e));
                    }
                });
                let mut i = 0;
                for m in rx {
                    let (k, f) = m?;
                    let release = || {
                        for (r, f) in rings.iter().zip(f) {
                            r.release(g, f);
                        }
                    };
                    // a frame the pass skips
                    if run.k.get(i) != Some(&k) {
                        release();
                    }
                    // every output frame that shows source frame k; the frame is released after the last
                    let mut fresh = true;
                    while run.k.get(i) == Some(&k) {
                        i += 1;
                        let last = run.k.get(i) != Some(&k);
                        if stop.load(Relaxed) || quit.load(Relaxed) {
                            return Err(CANCELLED.into());
                        }
                        let t = at(n);
                        let ev = ed.and_then(|e| e.tl.frame(n));
                        let v = match &mut job.out {
                            Out::Flat { view: Some(f), .. } => Some(f(t)),
                            Out::Flat { view: None, .. } => {
                                Some(ev.map_or_else(View::default, |e| view(&e.view)))
                            }
                            Out::Pano(_) => None,
                        };
                        let acquire = || {
                            for (r, f) in rings.iter().zip(f) {
                                r.acquire(g, f);
                            }
                        };
                        scoped(&g.device, || {
                            let src = [0, 1].map(|l| &bg[l][(f[l] % OUT as u64) as usize]);
                            dr.pose(&c.pose(k));
                            if let (Some(st), Some(r), true) = (&mut dr.st, raft, fresh) {
                                let _b = burn();
                                let mut e = g.device.create_command_encoder(&Default::default());
                                st.belt(&mut e, src, r);
                                acquire();
                                g.queue.submit([e.finish()]);
                                st.flow(r)?;
                            }
                            let mut e = g.device.create_command_encoder(&Default::default());
                            dr.draw(&mut e, src, v.as_ref(), fresh);
                            if let (Some((fx, tmp)), Some(io), Some(ev), Some(ed)) =
                                (&mut gfx, &io, ev, ed)
                            {
                                if look != Some(ev.k) {
                                    let c = ev.clip;
                                    if look.is_none_or(|l| ed.tl.clips[l].look != c.look) {
                                        fx.look(
                                            ed.looks.get(&c.look).filter(|_| !c.look.is_empty()),
                                        );
                                    }
                                    fx.set(&fx::Params {
                                        mix: c.amt,
                                        ..ed.adjust
                                    });
                                    look = Some(ev.k);
                                }
                                e.copy_texture_to_texture(
                                    dr.tex.as_image_copy(),
                                    tmp.as_image_copy(),
                                    dr.tex.size(),
                                );
                                fx.run(&mut e, io, &ovr);
                            }
                            if let Some(fx) = &mut job.fx {
                                fx(Fx {
                                    enc: &mut e,
                                    tex: &dr.tex,
                                    k: n,
                                    t,
                                });
                            }
                            let (Some(i), Some(y)) = (&input, &dr.yuv) else {
                                // a frame grab
                                let px = {
                                    let _g = gate();
                                    acquire();
                                    g.queue.submit([e.finish()]);
                                    stitch::read_texture(&g.device, &g.queue, &dr.tex)
                                };
                                release();
                                let Enc::Jpeg(q) = job.enc else { return Ok(()) };
                                let xmp = sph.then(|| jpeg::gpano(w, h));
                                let b = jpeg::encode(&px, w, h, q, xmp.as_deref())?;
                                return std::fs::write(part, b)
                                    .map_err(|e| format!("{}: {e}", part.display()));
                            };
                            y.encode(&mut e);
                            // outside the gate: the encoder thread takes it to free a slot
                            while let Err(x) = i.acquire(g, n) {
                                if quit.load(Relaxed) || stop.load(Relaxed) {
                                    return Err(CANCELLED.into());
                                }
                                if !matches!(x, video::Error::Full) {
                                    return Err(format!("encoder input: {x}"));
                                }
                            }
                            let cb = i
                                .copy_buffer(g, &y.buf, 0, y.pitch, n)
                                .map_err(|x| x.to_string())?;
                            {
                                let _g = gate();
                                acquire();
                                g.queue.submit([e.finish(), cb]);
                            }
                            if last {
                                release();
                            }
                            i.release(g, n);
                            etx.send(n).map_err(|_| "encoder thread gone".to_string())
                        })?;
                        fresh = false;
                        n += 1;
                        let now = Instant::now();
                        let s = (now - *t1.get_or_insert(now)).as_secs_f64();
                        p.done = n;
                        p.fps = if n > 1 && s > 0.0 {
                            (n - 1) as f64 / s
                        } else {
                            0.0
                        };
                        p.eta = if p.fps > 0.0 {
                            (total - n.min(total)) as f64 / p.fps
                        } else {
                            0.0
                        };
                        progress(p);
                    }
                }
            }
            Ok(())
        });
        if r.is_err() {
            quit.store(true, Relaxed);
        }
        drop(etx);
        let er = et.map(|t| t.join().unwrap_or_else(|x| Err(panicked(x))));
        match er {
            // the encoder thread's own failure (I/O, the encoder) stops the frames, so it is the cause; its
            // "cancelled" follows the frames' failure
            Some(Err(e)) if e != CANCELLED || r.is_ok() => Err(e),
            _ => r.map(|_| p),
        }
    })
}

struct Entry {
    id: u64,
    state: State,
    stop: Arc<AtomicBool>,
}

#[derive(Default)]
struct Sh {
    jobs: Mutex<Vec<Entry>>,
    next: AtomicU64,
}

impl Sh {
    fn jobs(&self) -> std::sync::MutexGuard<'_, Vec<Entry>> {
        self.jobs.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set(&self, id: u64, s: State) {
        if let Some(e) = self.jobs().iter_mut().find(|e| e.id == id) {
            e.state = s;
        }
    }
}

/// Export jobs, one after another on a background thread ("export"), which keeps the RAFT model between jobs.
pub struct Queue {
    tx: Option<mpsc::Sender<(u64, Job)>>,
    sh: Arc<Sh>,
    th: Option<thread::JoinHandle<()>>,
}

impl Queue {
    pub fn new(g: Arc<Gpu>, cx: Ctx) -> Queue {
        let (tx, rx) = mpsc::channel::<(u64, Job)>();
        let sh = Arc::new(Sh::default());
        let s = sh.clone();
        let th = thread::Builder::new()
            .name("export".into())
            .spawn(move || {
                let mut raft = None;
                for (id, job) in rx {
                    let Some(stop) = s.jobs().iter().find(|e| e.id == id).map(|e| e.stop.clone())
                    else {
                        continue;
                    };
                    if stop.load(Relaxed) {
                        s.set(id, State::Cancelled);
                        continue;
                    }
                    s.set(id, State::Running(Progress::default()));
                    let r = run(&g, &cx, &mut raft, job, &stop, &mut |p| {
                        s.set(id, State::Running(p))
                    });
                    let st = match r {
                        Ok(p) => State::Done(p),
                        Err(_) if stop.load(Relaxed) => State::Cancelled,
                        Err(e) => State::Failed(e),
                    };
                    s.set(id, st);
                }
            })
            .expect("export thread");
        Queue {
            tx: Some(tx),
            sh,
            th: Some(th),
        }
    }

    /// Queues `job`; returns its id.
    pub fn push(&self, job: Job) -> u64 {
        let id = self.sh.next.fetch_add(1, Relaxed);
        self.sh.jobs().push(Entry {
            id,
            state: State::Queued,
            stop: Arc::new(AtomicBool::new(false)),
        });
        if let Some(tx) = &self.tx {
            let _ = tx.send((id, job));
        }
        id
    }

    /// Cancels a queued or running job; a running one stops within a frame and leaves no output.
    pub fn cancel(&self, id: u64) {
        if let Some(e) = self.sh.jobs().iter().find(|e| e.id == id) {
            e.stop.store(true, Relaxed);
        }
    }

    pub fn state(&self, id: u64) -> Option<State> {
        self.sh
            .jobs()
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.state.clone())
    }

    /// Every job not cleared, oldest first.
    pub fn jobs(&self) -> Vec<(u64, State)> {
        self.sh
            .jobs()
            .iter()
            .map(|e| (e.id, e.state.clone()))
            .collect()
    }

    /// Forgets the finished jobs (done, failed, cancelled).
    pub fn clear(&self) {
        self.sh
            .jobs()
            .retain(|e| matches!(e.state, State::Queued | State::Running(_)));
    }
}

impl Drop for Queue {
    /// Cancels everything and waits for the running job to stop.
    fn drop(&mut self) {
        for e in self.sh.jobs().iter() {
            e.stop.store(true, Relaxed);
        }
        self.tx = None;
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}
