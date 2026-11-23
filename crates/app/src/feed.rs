//! The camera's live preview (milestone 3): its side-by-side H.264 (both fisheyes, lens 0 left, PROTOCOL.md §8.11)
//! decoded by crate video on a thread of its own, stitched by crate stitch's preview straight from the two halves of
//! each decoder picture with the camera's live calibration, levelled by the stream's IMU. A [`Feed`] is a live
//! source (`live::Source`) too: the Capture viewer and the live outputs show the same picture.
//!
//! Hand-over: the decode thread queues pictures; a consumer takes them in turn as they fall due ([`pick`]), which
//! releases the older ones. While the live outputs run they hold the hand-over and the viewer shows what they take:
//! one cursor, so a 30 fps output clock is never sampling a queue a 60 Hz viewer moves. While a consumer takes, a
//! full ring makes the decode thread wait for it; when none does (no viewer, no output), it advances itself once
//! `video::decode::OUT − 1` wait. Both happen under the lock the draws record and submit under ([`Hold`]), so a release
//! is never staged before the submit of a draw that samples the frame.
//!
//! Pacing: the X5 sends its 30 fps over USB in pairs (two packets within a few ms, then ~67 ms of nothing: half the
//! packet intervals under 17 ms, measured 2026-10-25). A picture falls due at its camera time plus the slowest
//! delivery (arrival and decode) of the last 32 pictures, at most two frame periods behind the fastest: every picture
//! is there when it is due, evenly spaced. The outputs always go at this pace; the viewer can show the newest picture instead.

use crate::player::{Look, Tgt};
use eframe::{egui, egui_wgpu::RenderState};
use glam::{DQuat, DVec3};
use link::stream;
use std::{
    collections::VecDeque,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, Ordering::Relaxed},
        mpsc::{Receiver, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant},
};
use stitch::{Calib, Ctx, Frame, Mode, Pose, Preview, View};
use video::{Decoder, Gpu, Ring, decode::OUT};

pub use crate::live::mono;

/// A decoded picture: decoder frame, camera ms (0 for the first ones after the camera booted), when its packet
/// arrived and when its decode finished (CLOCK_MONOTONIC ns).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pic {
    pub f: u64,
    pub ts: u64,
    pub got: u64,
    pub dec: u64,
}

/// What was counted: IMU samples, video packets, packets before the first IDR (a stream that starts mid-GOP),
/// pictures decoded, decoded but never drawn, camera-side gaps in the timestamps (frames the camera did not send),
/// packets the decoder refused, pictures ready over 10 ms after they were due (stalls); output clock ticks with a
/// picture, those that repeated the last one.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub imu: u64,
    pub got: u64,
    pub wait: u64,
    pub dec: u64,
    pub skip: u64,
    pub gaps: u64,
    pub bad: u64,
    pub late: u64,
    pub ticks: u64,
    pub rep: u64,
}

/// The newest `N` samples of something, ms.
#[derive(Clone)]
pub struct Samples {
    v: Vec<f32>,
    pub n: usize,
}

const N: usize = 4096;

impl Default for Samples {
    fn default() -> Samples {
        Samples {
            v: vec![0.0; N],
            n: 0,
        }
    }
}

impl Samples {
    pub fn push(&mut self, x: f32) {
        self.v[self.n % N] = x;
        self.n += 1;
    }

    /// p50, p99, max (NaN without samples).
    pub fn pct(&self) -> [f32; 3] {
        let mut v = self.v[..self.n.min(N)].to_vec();
        if v.is_empty() {
            return [f32::NAN; 3];
        }
        v.sort_by(f32::total_cmp);
        let at = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        [at(0.5), at(0.99), v[v.len() - 1]]
    }

    fn text(&self) -> String {
        let [a, b, c] = self.pct();
        format!("p50 {a:.2} p99 {b:.2} max {c:.2}")
    }
}

/// The IMU attitude kept for the frames' times: 3 s at the X5's ~500 samples per second.
const ATT: usize = 1500;

/// Two frame periods of the preview's 30 fps, ns.
pub const LATE: u64 = 66_666_667;

/// A consumer that has not taken for this long (ns) is gone: a hidden window, the outputs stopped.
pub const IDLE: u64 = 100_000_000;

/// Which of the queued pictures, due at `due` (oldest first), a consumer takes at `now`: the oldest one due less than
/// [`LATE`] ago, so a clock at the camera's rate takes each one in turn; else the newest one due (after a stall;
/// unpaced, all due at 0: always the newest). The window is two frames wide: a picture due just after a tick waits
/// for the next tick, and from then on the clock takes each picture about a frame after it fell due, a frame from
/// either end of the window. So jitter below a frame neither drops nor repeats one; a stall or the clocks drifting a
/// frame apart do.
pub fn pick<I>(due: I, now: u64) -> Option<usize>
where
    I: DoubleEndedIterator<Item = u64> + ExactSizeIterator + Clone,
{
    let n = due.clone().rposition(|d| d <= now)?;
    Some(
        due.take(n)
            .position(|d| now.saturating_sub(d) < LATE)
            .unwrap_or(n),
    )
}

/// Shared by the decode thread and the consumers.
#[derive(Default)]
struct Hold {
    q: VecDeque<Pic>,
    held: Option<Pic>,
    /// the held picture was drawn (replacing it before drops it)
    drawn: bool,
    /// the decoder's output and its display window, once the first picture is decoded
    ring: Option<(Arc<Ring>, [u32; 4])>,
    /// body → world per IMU sample, camera ms
    att: VecDeque<(u64, DQuat)>,
    c: Counts,
    /// decode time per picture, submit → done (macOS: `Decoder::took`); time between two video packets' arrivals
    dec: Samples,
    iv: Samples,
    /// pacing the viewer: on; arrival + decode − camera time (ns) of the last 32 pictures
    pace: bool,
    late: [i64; 32],
    nl: usize,
    /// when a consumer that can draw last took, when the outputs did (CLOCK_MONOTONIC ns)
    took: u64,
    out: u64,
    err: Option<String>,
}

impl Hold {
    /// Camera ns → CLOCK_MONOTONIC: the fastest of the last 32 deliveries (packet arrival and decode), and the slowest
    /// at most two frame periods behind it (what the ring holds beside the held picture and the one being decoded),
    /// which pictures are due at. A window, not the fastest ever: the two clocks drift apart.
    fn lag(&self) -> Option<[i64; 2]> {
        let l = &self.late[..self.nl.min(32)];
        let (a, b) = (*l.iter().min()?, *l.iter().max()?);
        Some([a, b.min(a + 66_000_000)])
    }

    /// The pose of the picture at camera ms `ts` with the framing `frame`: levelled by the IMU attitude then (heading
    /// kept) if `level`.
    fn pose(&self, ts: u64, level: bool, frame: Pose) -> Pose {
        let a = &self.att;
        let i = a.partition_point(|x| x.0 <= ts);
        let r = match (level && ts > 0, i) {
            (false, _) | (_, 0) => return frame,
            (_, i) => a[i - 1].1,
        };
        // crate stab's horizon lock with no smoothing: the virtual camera keeps the body's heading (azimuth of its X)
        let x = r * DVec3::X;
        let s = DQuat::from_rotation_z(x.y.atan2(x.x));
        Pose {
            q: (r.inverse() * s).as_quat() * frame.q,
            rs: Default::default(),
        }
    }

    /// The queued picture [`pick`] gives at `now`, paced or not.
    fn next(&self, now: u64, pace: bool) -> Option<usize> {
        let l = self.lag().filter(|_| pace).map_or(i64::MIN, |l| l[1]);
        let due = |p: &Pic| match p.ts {
            0 => 0,
            t => (t as i64 * 1_000_000).saturating_add(l).max(0) as u64,
        };
        pick(self.q.iter().map(due), now)
    }

    /// The picture [`Hold::next`] gives becomes the held one; it and every older one are released up to it.
    fn take(&mut self, g: &Gpu, now: u64, pace: bool) -> bool {
        let Some(i) = self.next(now, pace) else {
            return false;
        };
        let p = self.q[i];
        self.c.skip += i as u64 + (self.held.is_some() && !self.drawn) as u64;
        self.q.drain(..=i);
        if let Some((r, _)) = &self.ring
            && p.f > 0
        {
            r.release(g, p.f - 1);
        }
        (self.held, self.drawn) = (Some(p), false);
        true
    }
}

/// The NAL units of an Annex-B access unit (start codes and trailing zeros cut off), as ranges into `b`.
fn nals(b: &[u8], out: &mut Vec<Range<usize>>) {
    out.clear();
    let (mut i, mut s) = (0, None);
    while i + 2 < b.len() {
        if b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1 {
            if let Some(s) = s {
                let e = s + b[s..i].iter().rposition(|&x| x != 0).map_or(0, |e| e + 1);
                out.push(s..e);
            }
            i += 3;
            s = Some(i);
        } else {
            i += 1;
        }
    }
    if let Some(s) = s.filter(|&s| s < b.len()) {
        out.push(s..b.len());
    }
}

/// Gravity-corrected gyro integration (crate stab's Mahony filter and gains, forward only: the stream has no end to
/// run back from), starting from the first window's mean. IMU axes as in the files (`stab::AXES`). The
/// correction's accelerometer is the mean of the last `AVG` samples, as the files' filter's (`stab::Track::new`): raw
/// samples of a vibrating mount seldom pass the 1 g gate, and those that do lean.
#[derive(Default)]
struct Att {
    q: Option<DQuat>,
    bi: DVec3,
    t: i64,
    acc: VecDeque<DVec3>,
    sum: DVec3,
}

/// 100 ms of the stream's 500 samples a second.
const AVG: usize = 50;

impl Att {
    fn step(&mut self, s: &link::Imu) -> DQuat {
        let v = |a: [f64; 3]| stab::AXES * DVec3::from(a);
        let (w, a) = (v(s.gyro), v(s.acc));
        if self.acc.len() == AVG {
            self.sum -= self.acc.pop_front().unwrap();
        }
        self.acc.push_back(a);
        self.sum += a;
        let a = self.sum / self.acc.len() as f64;
        let q = match self.q {
            Some(q) if self.acc.len() == AVG => {
                let dt = (s.ts - self.t).clamp(0, 100) as f64 * 1e-3;
                let n = a.length();
                let e = if (n - 1.0).abs() < 0.1 {
                    (a / n).cross(q.inverse() * DVec3::Z)
                } else {
                    DVec3::ZERO
                };
                self.bi += stab::KI * dt * e;
                (q * stab::exp((w + stab::KP * e + self.bi) * dt)).normalize()
            }
            // the first 100 ms: no window yet, the heading still free
            _ => DQuat::from_rotation_arc(a.normalize_or(DVec3::Z), DVec3::Z),
        };
        (self.q, self.t) = (Some(q), s.ts);
        q
    }
}

/// Camera ms more than this apart: frames the camera did not send (it streams at 30 fps).
const GAP: u64 = 50;

/// The decode thread: packets in, pictures queued, the IMU attitude kept.
fn run(
    g: Arc<Gpu>,
    h: Arc<Mutex<Hold>>,
    rx: Receiver<link::Frame>,
    quit: Arc<AtomicBool>,
    wake: impl Fn(),
) {
    let (mut dec, mut ns, mut att, mut last, mut prev) =
        (None::<Decoder>, vec![], Att::default(), 0u64, 0u64);
    let lock = || h.lock().unwrap_or_else(|e| e.into_inner());
    let fail = |m: String| {
        let mut h = lock();
        h.c.bad += 1;
        if h.err.is_none() {
            eprintln!("x5desk: live preview: {m}");
            h.err = Some(m);
        }
    };
    while !quit.load(Relaxed) {
        let fr = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(f) => f,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match fr.kind {
            stream::GYRO => {
                let mut h = lock();
                for s in fr.imu() {
                    h.c.imu += 1;
                    let q = att.step(&s);
                    if h.att.len() == ATT {
                        h.att.pop_front();
                    }
                    h.att.push_back((s.ts.max(0) as u64, q));
                }
            }
            stream::VIDEO => {
                // when the link framed it: the time between two packets is the camera's and USB's, not this thread's
                let got = mono() - fr.at.elapsed().as_nanos() as u64;
                {
                    let mut h = lock();
                    h.c.got += 1;
                    if prev > 0 {
                        h.iv.push((got - prev) as f32 * 1e-6);
                    }
                }
                prev = got;
                nals(&fr.data, &mut ns);
                let mut au = [&[][..]; 32];
                let k = ns.len().min(au.len());
                for (a, r) in au.iter_mut().zip(&ns) {
                    *a = &fr.data[r.clone()];
                }
                let au = &au[..k];
                if dec.is_none() {
                    // the stream starts at an IDR with its parameter sets in band
                    let ps: Vec<&[u8]> = au
                        .iter()
                        .copied()
                        .filter(|n| !n.is_empty() && matches!(video::h264::nal_type(n), 7 | 8))
                        .collect();
                    if ps.len() < 2 {
                        lock().c.wait += 1;
                        continue;
                    }
                    let d = Decoder::new(&g, video::Codec::H264, &ps, 2, 2);
                    // its copies go on wgpu's queue: under GATE, as the app's other submits
                    #[cfg(target_os = "macos")]
                    let d = d.map(|d| d.gated(&crate::GATE));
                    match d {
                        Ok(d) => dec = Some(d),
                        Err(e) => {
                            fail(format!("decoder: {e}"));
                            return;
                        }
                    }
                }
                let d = dec.as_mut().unwrap();
                if fr.ts > 0 && last > 0 && fr.ts > last + GAP {
                    lock().c.gaps += ((fr.ts - last) as f64 / 33.37).round() as u64 - 1;
                }
                last = if fr.ts > 0 { fr.ts } else { last };
                #[cfg(target_os = "macos")]
                let failed = d.failed;
                let f = loop {
                    match d.decode(&g, au) {
                        Ok(f) => break Some(f),
                        // the consumer kept the slot past the decoder's wait (a stall): take its pictures over
                        Err(video::Error::Full) => {
                            if quit.load(Relaxed) {
                                return;
                            }
                            lock().take(&g, u64::MAX, false);
                        }
                        Err(e) => {
                            fail(format!("frame at {} ms: {e}", fr.ts));
                            break None;
                        }
                    }
                };
                let Some(f) = f else { continue };
                let r = d.ring.clone();
                // the decode queue's own time, from the submit (after any wait for a slot): the done timeline passes f
                #[cfg(not(target_os = "macos"))]
                let (t, dt) = {
                    let t0 = mono();
                    while r.ready() <= f && mono() - t0 < 200_000_000 {
                        thread::sleep(Duration::from_micros(200));
                    }
                    let t = mono();
                    (t, t - t0)
                };
                // VideoToolbox decodes synchronously, timed by the decoder from its slot on; a picture it failed on
                // shows the one before (`Decoder::failed`): refused, not decoded
                #[cfg(target_os = "macos")]
                let (t, dt, bad) = (mono(), d.took.as_nanos() as u64, d.failed > failed);
                #[cfg(not(target_os = "macos"))]
                let bad = false;
                let mut h = lock();
                if h.ring.is_none() {
                    h.ring = Some((r, d.view));
                }
                h.c.dec += !bad as u64;
                h.c.bad += bad as u64;
                h.dec.push(dt as f32 * 1e-6);
                // when it could be shown: arrival and decode, not a wait for a slot the consumer holds (which would
                // pace it later and hold the slots longer)
                if fr.ts > 0 {
                    let n = h.nl;
                    let c = fr.ts as i64 * 1_000_000;
                    let x = (got + dt) as i64 - c;
                    // a stall: ready (behind any packets this thread had not got to) well after it was due
                    if h.lag().is_some_and(|l| t as i64 - c > l[1] + 10_000_000) {
                        h.c.late += 1;
                    }
                    (h.late[n % 32], h.nl) = (x, n + 1);
                }
                h.q.push_back(Pic {
                    f,
                    ts: fr.ts,
                    got,
                    dec: t,
                });
                // the ring is full: the next decode waits for a consumer, if there is one
                if h.q.len() >= OUT - 1 && t > h.took + IDLE {
                    h.take(&g, u64::MAX, false);
                }
                drop(h);
                wake();
            }
            _ => {}
        }
    }
}

/// GPU timestamps around the viewer's stitch pass: query set, resolve buffer, read-back buffer, 0 idle / 1 mapping /
/// 2 mapped, ns per tick.
struct Stamp {
    qs: wgpu::QuerySet,
    res: wgpu::Buffer,
    rd: wgpu::Buffer,
    st: Arc<AtomicU8>,
    tick: f32,
}

/// What the viewer measures: when the first picture was taken (by it or a live output), pictures it drew, stitch
/// time (GPU), latency from the packet's arrival and from the capture (camera clock mapped onto CLOCK_MONOTONIC), all
/// ms.
#[derive(Clone, Default)]
pub struct Stats {
    pub t0: Option<Instant>,
    pub shown: u64,
    pub stitch: Samples,
    pub arrive: Samples,
    pub g2g: Samples,
}

pub struct Feed {
    g: Arc<Gpu>,
    cx: Ctx,
    h: Arc<Mutex<Hold>>,
    quit: Arc<AtomicBool>,
    th: Option<thread::JoinHandle<()>>,
    /// lens bind groups per ring slot
    bg: Vec<[stitch::Src; 2]>,
    cal: Option<Calib>,
    /// where the calibration came from, for the UI
    pub from: String,
    fr: Frame,
    pv: Option<Preview>,
    /// the held picture's colour fit is recorded; the picture the viewer showed last
    tinted: bool,
    shown: Option<Pic>,
    pub level: bool,
    /// the camera frames its videos on lens 1
    frame: Pose,
    pub look: Look,
    pub view: View,
    tgt: Option<Tgt>,
    stamp: Option<Stamp>,
    pub stats: Stats,
    /// CLOCK_MONOTONIC ns − camera ns: from GET_OPTIONS `media_clock` round trips
    pub off: Option<i64>,
    /// packets the link dropped because the decode thread fell behind
    pub lost: u64,
    name: String,
    /// the outputs have taken no picture since they started
    start: bool,
}

impl Feed {
    /// Starts the decode thread on the camera's packets `rx`; `wake` runs after each decoded picture. It draws once
    /// it has the calibration ([`Feed::cal`]).
    pub fn new(
        g: &Arc<Gpu>,
        cx: &Ctx,
        rx: Receiver<link::Frame>,
        name: &str,
        wake: impl Fn() + Send + 'static,
    ) -> Feed {
        let h = Hold {
            pace: true,
            ..Default::default()
        };
        let (h, quit) = (Arc::new(Mutex::new(h)), Arc::new(AtomicBool::new(false)));
        let (g2, h2, q2) = (g.clone(), h.clone(), quit.clone());
        let th = thread::Builder::new()
            .name("live decode".into())
            .spawn(move || run(g2, h2, rx, q2, wake))
            .expect("live decode thread");
        let stamp = g
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                let b = |usage| {
                    g.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("stitch time"),
                        size: 16,
                        usage,
                        mapped_at_creation: false,
                    })
                };
                Stamp {
                    qs: g.device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("stitch time"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 2,
                    }),
                    res: b(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                    rd: b(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
                    st: Arc::new(AtomicU8::new(0)),
                    tick: g.queue.get_timestamp_period(),
                }
            });
        Feed {
            g: g.clone(),
            cx: cx.clone(),
            h,
            quit,
            th: Some(th),
            bg: vec![],
            cal: None,
            from: String::new(),
            // X5 live preview: 1328² per lens of the 5376² calibration frame cut to 5312² (notification 8234)
            fr: Frame {
                w: 1328,
                h: 1328,
                crop: 5312.0,
                shift: [0.0; 2],
            },
            pv: None,
            tinted: false,
            shown: None,
            level: true,
            frame: Pose::euler(0.0, 0.0, 180.0),
            look: Look::Perspective,
            view: Look::Perspective.view(),
            tgt: None,
            stamp,
            stats: Stats::default(),
            off: None,
            lost: 0,
            name: name.into(),
            start: false,
        }
    }

    /// The calibration string (v6 / v3 / v1, `Calib::parse`) and which one it is.
    pub fn cal(&mut self, s: &str, from: &str) -> Result<(), String> {
        self.cal = Some(Calib::parse(s)?);
        self.from = from.into();
        self.pv = None;
        Ok(())
    }

    /// The stream's parameters (notification 8234): lens picture size and the window crop.
    pub fn params(&mut self, p: &link::pb::LiveNote) {
        let Some(c) = &p.crop else { return };
        let fr = Frame {
            w: p.width / 2,
            h: p.height,
            crop: c.dst_w as f32,
            shift: [c.x as f32, c.y as f32],
        };
        if (fr.w, fr.h, fr.crop, fr.shift) != (self.fr.w, self.fr.h, self.fr.crop, self.fr.shift)
            && fr.w > 0
        {
            (self.fr, self.pv) = (fr, None);
        }
    }

    pub fn counts(&self) -> Counts {
        self.h.lock().unwrap().c
    }

    /// The viewer paced by the camera clock (else the newest picture: lowest latency, uneven).
    pub fn paced(&self) -> bool {
        self.h.lock().unwrap().pace
    }

    pub fn pace(&self, on: bool) {
        self.h.lock().unwrap().pace = on;
    }

    pub fn err(&self) -> Option<String> {
        self.h.lock().unwrap().err.clone()
    }

    /// Takes the picture due at `now` ([`pick`]); true if it took one. `out`: for the live outputs, which then hold
    /// the hand-over and always go at the camera's pace; the viewer takes none meanwhile.
    fn take(&mut self, now: u64, out: bool) -> bool {
        let mut h = self.h.lock().unwrap();
        let t = mono();
        let ran = t < h.out + IDLE;
        if out {
            h.out = t;
        } else if ran {
            return false;
        }
        // a consumer that cannot draw yet must not hold the decoder back
        if self.cal.is_some() {
            h.took = t;
        }
        // outputs starting take their first picture only once it is due a quarter of the window: the pictures after
        // it then fall due at least that far from either end of the window at their ticks
        self.start |= out && !ran;
        let now = match out && self.start {
            true => now.saturating_sub(LATE / 4),
            false => now,
        };
        let pace = out || h.pace;
        let n = h.take(&self.g, now, pace);
        self.start &= !(out && n);
        if out && h.held.is_some() {
            h.c.ticks += 1;
            h.c.rep += (ran && !n) as u64;
        }
        drop(h);
        self.tinted &= !n;
        if n {
            self.stats.t0.get_or_insert_with(Instant::now);
        }
        n
    }

    pub fn ready(&self) -> bool {
        self.cal.is_some() && self.h.lock().unwrap().held.is_some()
    }

    /// Makes the preview and the lens bind groups once the calibration and the decoder's ring are there.
    fn prep(&mut self, h: &Hold) -> bool {
        let (Some(cal), Some((r, [x, y, w, hh]))) = (&self.cal, &h.ring) else {
            return false;
        };
        if self.pv.is_none() {
            // the camera writes blend angle 0 into its files; 6° is the app's default
            self.pv = Some(Preview::new(
                &self.cx,
                cal,
                self.fr,
                6.0,
                Some(0.1),
                crate::player::FMT,
            ));
        }
        if self.bg.is_empty() {
            let w2 = w / 2;
            self.bg = (0..OUT as u64)
                .map(|s| {
                    let t = r.planes(s);
                    [
                        self.cx.rect(&t, [*x, *y, w2, *hh]),
                        self.cx.rect(&t, [x + w2, *y, w2, *hh]),
                    ]
                })
                .collect();
        }
        true
    }

    /// Records the held picture with view `v` into `t` (viewport `vp`) and submits; `stamp`: time the pass on the GPU.
    /// Returns the picture drawn.
    fn draw_into(
        &mut self,
        t: &wgpu::TextureView,
        v: &View,
        vp: [f32; 4],
        stamp: bool,
    ) -> Option<Pic> {
        let hh = self.h.clone();
        let mut h = hh.lock().unwrap();
        let p = h.held?;
        if !self.prep(&h) {
            return None;
        }
        let (pv, src, ring) = (
            self.pv.as_ref().unwrap(),
            &self.bg[(p.f % OUT as u64) as usize],
            &h.ring.as_ref().unwrap().0,
        );
        let src = [&src[0], &src[1]];
        pv.pose(&h.pose(p.ts, self.level, self.frame));
        pv.set(v, vp);
        let mut e = self.g.device.create_command_encoder(&Default::default());
        if !self.tinted {
            pv.colour(&mut e, src);
        }
        let st = self
            .stamp
            .as_ref()
            .filter(|s| stamp && s.st.load(Relaxed) == 0);
        {
            let mut rp = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("live preview"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: t,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: st.map(|s| wgpu::RenderPassTimestampWrites {
                    query_set: &s.qs,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
                ..Default::default()
            });
            rp.set_viewport(vp[0], vp[1], vp[2], vp[3], 0.0, 1.0);
            pv.draw(&mut rp, src);
        }
        if let Some(s) = st {
            e.resolve_query_set(&s.qs, 0..2, &s.res, 0);
            e.copy_buffer_to_buffer(&s.res, 0, &s.rd, 0, 16);
        }
        ring.acquire(&self.g, p.f);
        self.g.queue.submit([e.finish()]);
        h.drawn = true;
        drop(h);
        if let Some(s) = st {
            s.st.store(1, Relaxed);
            let x = s.st.clone();
            s.rd.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                x.store(if r.is_ok() { 2 } else { 0 }, Relaxed)
            });
        }
        self.tinted = true;
        Some(p)
    }

    /// The viewer: renders into its texture at `size` ([`Feed::show`]). Returns the texture.
    pub fn render(
        &mut self,
        rs: &RenderState,
        size: [u32; 2],
        dirty: bool,
    ) -> Option<egui::TextureId> {
        let resized = Tgt::fit(&mut self.tgt, &self.g, rs, size);
        let t = self.tgt.as_ref().unwrap().view.clone();
        self.show(&t, size, dirty || resized);
        self.tgt.as_ref().map(|t| t.id)
    }

    /// The viewer without egui: takes the picture due (unless the outputs run), draws the held one into `t` (`size`)
    /// when it is new or `dirty`, and measures.
    pub fn show(&mut self, t: &wgpu::TextureView, size: [u32; 2], dirty: bool) {
        let new = self.take(mono(), false);
        let held = self.h.lock().unwrap().held;
        if new || dirty || held != self.shown {
            let v = self.view;
            let vp = [0.0, 0.0, size[0] as f32, size[1] as f32];
            if let Some(p) = self.draw_into(t, &v, vp, true)
                && self.shown != Some(p)
            {
                let now = mono();
                let s = &mut self.stats;
                s.shown += 1;
                s.arrive.push((now - p.got) as f32 * 1e-6);
                if let Some(o) = self.off.filter(|_| p.ts > 0) {
                    s.g2g
                        .push((now as i64 - (p.ts as i64 * 1_000_000 + o)) as f32 * 1e-6);
                }
                self.shown = Some(p);
            }
        }
        if let Some(s) = &self.stamp
            && s.st.load(Relaxed) == 2
        {
            if let Ok(m) = s.rd.slice(..).get_mapped_range() {
                let t = [0, 8].map(|o| u64::from_le_bytes(m[o..o + 8].try_into().unwrap()));
                if t[1] > t[0] {
                    self.stats.stitch.push((t[1] - t[0]) as f32 * s.tick * 1e-6);
                }
            }
            s.rd.unmap();
            s.st.store(0, Relaxed);
        }
    }

    /// Statistics since the viewer showed its first picture.
    pub fn report(&self) -> String {
        let (d, iv, tilt, pace) = {
            let h = self.h.lock().unwrap();
            // the body's up against the world's: how far the camera stands off level
            let t = h
                .att
                .back()
                .map(|a| (a.1 * DVec3::Z).angle_between(DVec3::Z).to_degrees());
            let l = h.lag().map_or(0, |l| l[1] - l[0]) as f64 * 1e-6;
            let p = match h.pace {
                true => format!("paced, {l:.1} ms behind the fastest delivery"),
                false => format!("newest picture; outputs paced {l:.1} ms behind"),
            };
            (h.dec.clone(), h.iv.clone(), t, p)
        };
        let c = self.counts();
        let s = &self.stats;
        let secs = s.t0.map_or(0.0, |t| t.elapsed().as_secs_f64());
        format!(
            "{:.1} s ({pace}): {} pictures shown ({:.1} fps); packets {} received ({} before the first IDR), {} decoded, {} lost \
             on the link, {} refused, {} stalled (ready over 10 ms after due); dropped {} (decoded, never drawn); \
             outputs: {} ticks, {} repeated the last picture; camera gaps {}; packet interval ms {} ({} under 17 ms); decode ms {}; stitch ms (GPU) {}; \
             packet → shown ms {}; capture → shown ms {}; IMU samples {}{}",
            secs,
            s.shown,
            s.shown as f64 / secs.max(1e-9),
            c.got,
            c.wait,
            c.dec,
            self.lost,
            c.bad,
            c.late,
            c.skip,
            c.ticks,
            c.rep,
            c.gaps,
            iv.text(),
            iv.v[..iv.n.min(N)].iter().filter(|&&x| x < 17.0).count(),
            d.text(),
            s.stitch.text(),
            s.arrive.text(),
            if s.g2g.n > 0 {
                s.g2g.text()
            } else {
                "(no camera clock)".into()
            },
            c.imu,
            tilt.map_or(String::new(), |t| format!(", camera {t:.1}° off level"))
        )
    }
}

impl Drop for Feed {
    fn drop(&mut self) {
        self.quit.store(true, Relaxed);
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
        let h = self.h.lock().unwrap();
        if let (Some(p), Some((r, _))) = (h.held, &h.ring) {
            r.release(&self.g, p.f);
        }
    }
}

/// The preview as a live source: at each output tick the picture due then, the view or the 360 equirect; no audio
/// (the preview carries none).
impl crate::live::Source for Feed {
    fn name(&self) -> &str {
        &self.name
    }

    fn tick(&mut self, at: u64) {
        self.take(at, true);
    }

    fn ready(&self) -> bool {
        Feed::ready(self)
    }

    fn draw(&mut self, t: &wgpu::TextureView, [w, h]: [u32; 2], pano: bool) {
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
        self.draw_into(t, &v, r, false);
    }

    fn asc(&mut self) -> Option<&[u8]> {
        None
    }

    fn audio(&mut self, _: &mut dyn FnMut(u64, &[u8])) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb() {
        let b = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 0, 1, 0x65, 4, 5, 0, 0,
        ];
        let mut v = vec![];
        nals(&b, &mut v);
        let n: Vec<&[u8]> = v.iter().map(|r| &b[r.clone()]).collect();
        assert_eq!(n, [&[0x67, 1, 2][..], &[0x68, 3], &[0x65, 4, 5, 0, 0]]);
    }

    /// One run of [`pacing`]: the X5's delivery of `n` pictures to a consumer ticking every `per` ns from `t0`, through a
    /// ring of `OUT` slots (a full one makes the decoder wait for the consumer, up to the decoder's 50 ms). `pick`:
    /// the policy (the hold and the time) → the queued picture taken. Returns the pictures taken and the ticks that
    /// took none (after the first).
    fn sim(
        arr: &[u64],
        ts: &[u64],
        t0: u64,
        per: u64,
        pick: impl Fn(&Hold, u64) -> Option<usize>,
    ) -> (Vec<u64>, u64) {
        let mut h = Hold::default();
        let (mut taken, mut rep, mut i, mut t) = (vec![], 0, 0, t0);
        let decode = |h: &mut Hold, i: usize| {
            let n = h.nl;
            (h.late[n % 32], h.nl) = (arr[i] as i64 - ts[i] as i64 * 1_000_000, n + 1);
            h.q.push_back(Pic {
                f: i as u64,
                ts: ts[i],
                got: arr[i],
                dec: arr[i],
            });
        };
        while i < arr.len() || !h.q.is_empty() {
            // pictures decoded before this tick; the one that finds the ring full waits for the tick
            while i < arr.len() && arr[i] <= t {
                if h.q.len() + h.held.is_some() as usize >= OUT {
                    if t - arr[i] > 50_000_000 {
                        h.q.clear();
                    }
                    break;
                }
                decode(&mut h, i);
                i += 1;
            }
            match pick(&h, t) {
                Some(k) => {
                    taken.push(h.q[k].f);
                    h.held = Some(h.q[k]);
                    h.q.drain(..=k);
                }
                None => rep += !taken.is_empty() as u64,
            }
            t += per;
        }
        (taken, rep)
    }

    /// The hand-over against the X5's delivery: 30 s of pictures sent in pairs (the first one held back for the second,
    /// up to 3 ms of jitter; camera times in ms, steps of 33 and 34), taken by a 30 fps output clock at ten phases,
    /// with and without stalls (a 100 ms hold-up every 5 s), and by a 60 Hz viewer. Paced, a 30 fps clock takes every
    /// picture once and in order, but around a stall: it repeats while it waits and drops one to catch up (at most
    /// two of each per stall); the newest picture drops half.
    #[test]
    fn pacing() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % 3_000_000
        };
        let ms = 1_000_000u64;
        let n = 900usize;
        let ts: Vec<u64> = (0..n as u64)
            .map(|i| 278_164 + (i * 1000 + 15) / 30)
            .collect();
        for (fps, stall) in [(30, false), (30, true), (60, false)] {
            let per = 1_000_000_000 / fps;
            let mut arr: Vec<u64> = (0..n)
                .map(|i| ts[(i | 1).min(n - 1)] * ms + 7_000 * ms + rnd() + (i as u64 & 1) * 2 * ms)
                .collect();
            for i in 1..n {
                if stall && i % 150 == 75 {
                    arr[i] += 100 * ms;
                }
                arr[i] = arr[i].max(arr[i - 1]);
            }
            let (mut worst, mut old) = ((0, 0), (0, 0));
            for ph in 0..10 {
                let t0 = arr[0] + ph * per / 10;
                let (tk, rep) = sim(&arr, &ts, t0, per, |h, t| h.next(t, true));
                assert!(tk.windows(2).all(|w| w[0] < w[1]), "out of order");
                let drop = (tk[tk.len() - 1] - tk[0] + 1) as usize - tk.len();
                let rep = match fps {
                    // a 60 Hz clock shows each picture twice
                    60 => rep - (tk.len() as u64 - 1),
                    _ => rep,
                };
                worst = (worst.0.max(drop), worst.1.max(rep));
                let (tk, r) = sim(&arr, &ts, t0, per, |h, t| h.next(t, false));
                let d = (tk[tk.len() - 1] - tk[0] + 1) as usize - tk.len();
                old = (old.0.max(d), old.1.max(r));
                println!(
                    "{fps} Hz, stalls {stall}, phase {ph}/10: paced dropped {drop}, repeated {rep}; newest dropped {d}"
                );
            }
            println!("{fps} Hz, stalls {stall}: worst paced {worst:?}, newest {old:?}");
            match (fps, stall) {
                (30, false) => assert!(worst.0 == 0 && worst.1 <= 1, "{worst:?}"),
                (30, true) => assert!(worst.0 <= 12 && worst.1 <= 12, "{worst:?}"),
                _ => assert_eq!(worst.0, 0),
            }
        }
    }

    /// Level and still: the attitude stays level; tilted 20° about the lens axis (body X), the levelling pose undoes
    /// the roll and keeps the heading.
    #[test]
    fn levelling() {
        // body = AXES · imu: imu (−z, x, −y) of body (x, y, z); at rest +1 g along body +Z → imu x = −1
        let imu = |b: DVec3| [-b.z, b.x, -b.y];
        let mut a = Att::default();
        for k in 0..500 {
            let q = a.step(&link::Imu {
                ts: k * 2,
                acc: imu(DVec3::Z),
                gyro: [0.0; 3],
            });
            assert!((q * DVec3::Z - DVec3::Z).length() < 1e-9);
        }
        let tilt = DQuat::from_rotation_x(20f64.to_radians());
        let mut a = Att::default();
        let mut q = DQuat::IDENTITY;
        for k in 0..2000 {
            // gravity seen in the tilted body frame
            q = a.step(&link::Imu {
                ts: k * 2,
                acc: imu(tilt.inverse() * DVec3::Z),
                gyro: [0.0; 3],
            });
        }
        let up = q * DVec3::Z;
        println!("tilted body: up in the world {up:?}");
        assert!((up - tilt * DVec3::Z).length() < 1e-3);
        let mut h = Hold::default();
        h.att.push_back((100, q));
        let p = h.pose(120, true, Pose::default());
        // the output's up (world Z in the virtual camera) lands on the body's measured up
        let o = p.q.as_dquat() * DVec3::Z;
        assert!((o - q.inverse() * DVec3::Z).length() < 1e-5, "{o:?}");
        assert_eq!(h.pose(120, false, Pose::default()).q, glam::Quat::IDENTITY);
    }

    /// Recordings' IMU at the stream's rate (every second sample of the files' 1 kHz) through the live filter: its
    /// gravity against the files' two-way filter's (`stab::Track::new`), rms and worst after 10 s. Files: `X5_LEVEL`
    /// (a path list; unset: skipped). Rides on a scooter's stem stay within 6° rms (0.7–4.9°; gating raw samples from
    /// the first one on: 2.5–16.7°).
    #[test]
    fn ride() {
        let Some(v) = std::env::var_os("X5_LEVEL") else {
            println!("X5_LEVEL unset: skipped");
            return;
        };
        let mut all = 0f64;
        for f in std::env::split_paths(&v).filter(|p| !p.as_os_str().is_empty()) {
            let imu = media::insv::Trailer::open(&f).unwrap().imu().unwrap();
            let tr = stab::Track::new(&imu, stab::KP, stab::KI);
            let (mut a, mut sum, mut n, mut worst) = (Att::default(), 0.0, 0, 0f64);
            for s in imu.iter().step_by(2) {
                let q = a.step(&link::Imu {
                    ts: (s.t / 1000) as i64,
                    acc: s.acc.map(f64::from),
                    gyro: s.gyro.map(f64::from),
                });
                let t = s.t as f64 * 1e-6;
                if t - tr.t[0] < 10.0 {
                    continue;
                }
                let e = (q.inverse() * DVec3::Z)
                    .angle_between(tr.at(t).inverse() * DVec3::Z)
                    .to_degrees();
                (sum, n, worst) = (sum + e * e, n + 1, worst.max(e));
            }
            let rms = (sum / n.max(1) as f64).sqrt();
            println!(
                "{}: {n} samples, gravity off the file filter's by {rms:.2}° rms, {worst:.2}° at worst",
                f.file_name().unwrap().to_string_lossy()
            );
            all = all.max(rms);
        }
        assert!(all < 6.0, "{all}");
    }
}
