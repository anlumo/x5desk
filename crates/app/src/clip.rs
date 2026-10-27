//! A clip's video: the media thread demuxes (crate `media`) and decodes (crate `video`) ahead of the master clock
//! into the decoders' rings and queues the frames; the UI takes the newest frame due and never paces anything.
//! Frames are numbered and queued in presentation order: with B-frames the media thread keeps a decoded frame back
//! until every frame shown before it is queued.
//! Frame hand-over (ARCHITECTURE.md): the UI acquires the frame it samples and releases it when it takes a newer one
//! (`release_before` also frees the frames it skipped) or when it seeks; the media thread releases what the UI
//! never got: frames decoded only to reach a seek target and the queue a seek discards. A ring's release frees every
//! earlier decoder frame, so a release stops below the lowest decoder frame still to show (`Pic::r`).

use media::mp4::{Codec, Mp4, Sample, Track};
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
use video::{Decoder, Gpu, Ring, decode::OUT};

/// A frame counts as due this much early: a seek puts the clock on the target's time rounded to audio samples.
const EPS: f64 = 1e-3;

/// A decoded frame: decoder frame index per ring; per ring the lowest decoder frame of the frames after it (with
/// B-frames a later frame may decode first), which its release stops below; frame index (presentation order),
/// presentation time (s), seek epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pic {
    pub f: [u64; 2],
    r: [u64; 2],
    pub k: usize,
    pub t: f64,
    e: u32,
}

impl Pic {
    /// Per ring: the lowest decoder frame of this frame and every later one.
    fn lo(&self) -> [u64; 2] {
        [0, 1].map(|i| self.f[i].min(self.r[i]))
    }
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

/// Presentation order of samples in decode order: the sample of each frame (by pts, equal ones in decode order), and
/// how many decoded frames a ring holds at once for a consumer that keeps one while the next decodes, from the
/// lowest decoder frame still to show to the next frame's (2 without B-frames, 7 with libx264's defaults).
fn order(s: &[Sample]) -> (Vec<usize>, usize) {
    let mut ord: Vec<usize> = (0..s.len()).collect();
    ord.sort_by_key(|&i| s[i].pts);
    let (mut w, mut m) = (1, usize::MAX);
    for k in (0..ord.len()).rev() {
        m = m.min(ord[k]);
        if let Some(&x) = ord.get(k + 1) {
            w = w.max(x + 1 - m);
        }
    }
    (ord, w)
}

/// The samples a decoder starting at the first one skips (`video::Error::Skip`) among those shown before it, its
/// leading pictures: an H.265 IRAP's RASL pictures, all of an H.264 non-IDR I picture's.
fn skipped(m: &Mp4, tr: &Track, n: usize) -> Result<Vec<usize>, String> {
    let lead: Vec<usize> = (1..n)
        .filter(|&s| tr.samples[s].pts < tr.samples[0].pts)
        .collect();
    let (Codec::Hevc { nal, ps } | Codec::Avc { nal, ps }) = &tr.codec else {
        unreachable!()
    };
    let (h265, mut buf) = (matches!(tr.codec, Codec::Hevc { .. }), vec![]);
    // H.265: the NAL unit type of the sample's first slice; H.264: whether it is a non-IDR I picture
    let mut ty = |s: usize| -> Result<u8, String> {
        m.read(&tr.samples[s], &mut buf)
            .map_err(|e| e.to_string())?;
        let mut au = [&[][..]; 64];
        let c = nals(&buf, *nal as usize, &mut au);
        let au = au[..c].iter().copied().filter(|n| n.len() > 2);
        if h265 {
            return Ok(au
                .map(video::h265::nal_type)
                .find(|&t| t < 32)
                .unwrap_or(63));
        }
        let mut h4 = video::h264::State::default();
        (ps.iter().map(Vec::as_slice))
            .chain(au.clone())
            .for_each(|n| _ = h4.put(n));
        let mut sl = au.filter(|n| matches!(video::h264::nal_type(n), 1 | 5));
        Ok(sl.all(|n| h4.slice(n).is_ok_and(|s| s.intra && !s.idr)) as u8)
    };
    if lead.is_empty() {
        return Ok(lead);
    }
    Ok(match (h265, ty(0)?) {
        (true, 16..=21) => {
            let mut r = vec![];
            for s in lead {
                if matches!(ty(s)?, 8 | 9) {
                    r.push(s);
                }
            }
            r
        }
        (false, 1) => lead,
        _ => vec![],
    })
}

/// The sample decoding to frame `k` starts at: the last sync sample at or before `k`'s (decode order) that is not
/// shown after `k` (an open GOP's leading frames need the GOP before), else 0. `pk`: the frame of each sample.
fn start(sync: &[usize], ord: &[usize], pk: &[usize], k: usize) -> usize {
    let s = &sync[..sync.partition_point(|&s| s <= ord[k])];
    s.iter().rev().find(|&&s| pk[s] <= k).map_or(0, |&s| s)
}

#[derive(Debug, PartialEq)]
enum Plan {
    /// the queue holds the target: drop the frames before it
    Keep,
    /// decode on, dropping everything before the target
    On,
    /// restart at this sample, dropping everything before the target
    Go(usize),
}

/// Seek to frame `t`, with `nk` = the next frame to queue, `pos` = the next sample to decode, `q0` = the first queued
/// frame and `s` = the sample decoding to `t` starts at: keep the queue from `t` on if it holds `t`; else decode on
/// when `t` lies ahead and `s` is not past `pos`; else restart at `s`.
fn plan(t: usize, nk: usize, pos: usize, q0: Option<usize>, s: usize) -> Plan {
    match q0 {
        Some(q0) if q0 <= t && t < nk => Plan::Keep,
        _ if nk <= t && s <= pos => Plan::On,
        _ => Plan::Go(s),
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
    /// presentation time of each frame (ascending), and the end of the last one (s)
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
        let (mut ord, w) = order(&tr.samples[..n]);
        // never decoded, so no frames (a stream copy cut at an open GOP's I picture)
        let skip = skipped(m, tr, n)?;
        ord.retain(|s| !skip.contains(s));
        let mut decs = vec![];
        for &(i, t) in &st {
            let tr = &mp4s[i].tracks[t];
            let (c, ps) = match &tr.codec {
                Codec::Hevc { ps, .. } => (video::Codec::H265, ps),
                Codec::Avc { ps, .. } => (video::Codec::H264, ps),
                _ => unreachable!(),
            };
            // decoding runs as far ahead of a consumer holding one frame as without B-frames (2 frames, OUT slots)
            let d = Decoder::with_out(g, c, ps, tr.width, tr.height, w + OUT - 2);
            decs.push(d.map_err(|e| e.to_string())?);
        }
        let times: Arc<[f64]> = (ord.iter())
            .map(|&s| (tr.samples[s].pts + d) as f64 / ts)
            .collect();
        let s = tr.samples[ord[ord.len() - 1]];
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
            .spawn(move || run(&g, &s, &mp4s, &st, decs, n, &ord, &t, &sync, wake))
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
        free(g, &self.rings, p.r);
    }

    /// Releases everything before `p`.
    pub fn release_before(&self, g: &Gpu, p: &Pic) {
        free(g, &self.rings, p.lo());
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

/// Releases every decoder frame below `r[i]` of ring `i`.
fn free<R: std::ops::Deref<Target = Ring>>(g: &Gpu, rings: &[R], r: [u64; 2]) {
    for (ring, &r) in rings.iter().zip(&r) {
        if r > 0 {
            ring.release(g, r - 1);
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

/// The media thread: decodes every stream's sample `pos` (of `ns`) into its decoder, queues the frames in
/// presentation order (or releases them while running up to a seek target), and follows the UI's seeks. Backpressure
/// is the rings: `decode` waits while the UI holds the slot it needs.
#[allow(clippy::too_many_arguments)]
fn run(
    g: &Gpu,
    sh: &Sh,
    mp4s: &[Mp4],
    st: &[(usize, usize)],
    mut decs: Vec<Decoder>,
    ns: usize,
    ord: &[usize],
    times: &[f64],
    sync: &[usize],
    wake: impl Fn(),
) {
    let n = times.len();
    // usize::MAX: no frame (a skipped leading picture)
    let mut pk = vec![usize::MAX; ns];
    for (k, &s) in ord.iter().enumerate() {
        pk[s] = k;
    }
    let nal: Vec<usize> = st
        .iter()
        .map(|&(i, t)| match mp4s[i].tracks[t].codec {
            Codec::Hevc { nal, .. } | Codec::Avc { nal, .. } => nal as usize,
            _ => 4,
        })
        .collect();
    // pos: the next sample to decode, nk: the next frame to queue; own: nothing of this epoch is queued, so the UI
    // holds no frame and what the run-up to the target drops is ours to release
    let (mut pos, mut nk, mut e, mut own, mut buf) = (0, 0, 0, true, vec![]);
    let mut f = [0u64; 2];
    // decoded frames waiting for an earlier one: frame index, decoder frames (None: one failed)
    let mut pend: Vec<(usize, Option<[u64; 2]>)> = vec![];
    // per ring the lowest decoder frame of those waiting and those not decoded yet
    let low = |decs: &[Decoder], pend: &[(usize, Option<[u64; 2]>)]| {
        let mut r = [0; 2];
        for (i, d) in decs.iter().enumerate() {
            r[i] = (pend.iter().filter_map(|p| p.1)).fold(d.frame, |a, f| a.min(f[i]));
        }
        r
    };
    let fail = |m: String| {
        let mut q = sh.q();
        if q.err.is_none() {
            eprintln!("media: {m}");
            q.err = Some(m);
        }
    };
    'main: while !sh.quit.load(Relaxed) {
        {
            let mut q = sh.q();
            if q.epoch != e {
                e = q.epoch;
                let t = q.target.min(n - 1);
                let s = start(sync, ord, &pk, t);
                match plan(t, nk, pos, q.pics.front().map(|p| p.k), s) {
                    Plan::Keep => {
                        // a frame that failed to decode left a gap
                        let d = q.pics.partition_point(|p| p.k < t);
                        free(g, &decs, q.pics.get(d).map_or(low(&decs, &pend), Pic::lo));
                        q.pics.drain(..d);
                        q.pics.iter_mut().for_each(|p| p.e = e);
                        own = false;
                    }
                    Plan::On => {
                        q.pics.clear();
                        pend.retain(|p| p.0 >= t);
                        (nk, own) = (t, true);
                        free(g, &decs, low(&decs, &pend));
                    }
                    Plan::Go(s) => {
                        q.pics.clear();
                        pend.clear();
                        free(g, &decs, low(&decs, &[]));
                        // a CRA or an H.264 non-IDR I picture there skips its leading frames, shown before it:
                        // `start` picks a sync sample only for frames from it on
                        decs.iter_mut().for_each(Decoder::restart);
                        (pos, nk, own) = (s, t, true);
                    }
                }
            }
        }
        if pos >= ns {
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
                    Err(video::Error::Skip) => {
                        ok = false;
                        break;
                    }
                    Err(x) => {
                        fail(format!("frame {pos}, stream {i}: {x}"));
                        ok = false;
                        break;
                    }
                }
            }
        }
        let k = pk[pos];
        pos += 1;
        if (nk..usize::MAX).contains(&k) {
            pend.push((k, ok.then_some(f)));
        } else if own {
            free(g, &decs, low(&decs, &pend));
        }
        let mut q = None;
        while let Some(i) = pend.iter().position(|p| p.0 == nk) {
            let (k, f) = pend.swap_remove(i);
            nk += 1;
            if let Some(f) = f {
                let r = low(&decs, &pend);
                let t = times[k];
                q.get_or_insert_with(|| sh.q())
                    .pics
                    .push_back(Pic { f, r, k, t, e });
                own = false;
            }
        }
        if q.take().is_some() {
            wake();
        }
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
        // no B-frames: frames are samples
        let id: Vec<usize> = (0..175).collect();
        let i = |k| start(&[0, 60, 120], &id, &id, k);
        assert_eq!([i(0), i(59), i(60), i(100), i(174)], [0, 0, 60, 60, 120]);
        assert_eq!(start(&[], &id, &id, 5), 0);
        // queued 10..14, next to decode 14
        assert_eq!(plan(12, 14, 14, Some(10), 0), Plan::Keep);
        assert_eq!(plan(10, 14, 14, Some(10), 0), Plan::Keep);
        // just past the queue, no sync sample in between: decode on
        assert_eq!(plan(14, 14, 14, Some(10), 0), Plan::On);
        assert_eq!(plan(59, 14, 14, Some(10), 0), Plan::On);
        // a sync sample after pos: restart there
        assert_eq!(plan(100, 14, 14, Some(10), 60), Plan::Go(60));
        // backwards: restart
        assert_eq!(plan(5, 14, 14, Some(10), 0), Plan::Go(0));
        assert_eq!(plan(9, 70, 70, Some(66), 0), Plan::Go(0));
        assert_eq!(plan(0, 0, 0, None, 0), Plan::On);
        // B-frames: frame 4 decoded (waiting for 3), frames 1, 2 queued
        assert_eq!(plan(4, 3, 5, Some(1), 0), Plan::On);
        assert_eq!(plan(2, 3, 5, Some(1), 0), Plan::Keep);
    }

    /// Samples in decode order with these presentation times (sync where `key`).
    fn samples(pts: &[i64], key: &[usize]) -> Vec<Sample> {
        (pts.iter().enumerate())
            .map(|(i, &pts)| Sample {
                off: 0,
                dts: i as i64,
                pts,
                size: 0,
                dur: 1,
                sync: key.contains(&i),
            })
            .collect()
    }

    /// Presentation order and the ring a consumer that holds a frame needs: without B-frames, libx264's `-bf 1` and its
    /// defaults (three B-frames, the middle one a reference: P4 B2 b1 b3, and frame 4 waits for 1 to 3 while the
    /// consumer holds frame 0), an open GOP whose leading frames a seek into them starts before.
    #[test]
    fn order_maths() {
        let gop = |n: i64, m: &[i64]| -> Vec<i64> {
            let w = m.len() as i64;
            std::iter::once(0)
                .chain((0..n).flat_map(|g| m.iter().map(move |x| g * w + x)))
                .collect()
        };
        for (what, pts, w) in [
            ("no B-frames", gop(20, &[1]), 2),
            ("-bf 1", gop(20, &[2, 1]), 4),
            ("libx264 defaults", gop(20, &[4, 2, 1, 3]), 7),
        ] {
            let (ord, ww) = order(&samples(&pts, &[0]));
            let shown: Vec<i64> = ord.iter().map(|&s| pts[s]).collect();
            assert!(shown.windows(2).all(|x| x[0] < x[1]), "{what}: {shown:?}");
            assert_eq!(ww, w, "{what}");
        }
        // I0 P3 B1 B2 | I6 (open GOP) B4 B5 P9 B7 B8: B4 and B5 lead I6
        let pts = [0, 3, 1, 2, 6, 4, 5, 9, 7, 8];
        let (ord, _) = order(&samples(&pts, &[0, 4]));
        let mut pk = vec![0; ord.len()];
        for (k, &s) in ord.iter().enumerate() {
            pk[s] = k;
        }
        let s = |k| start(&[0, 4], &ord, &pk, k);
        assert_eq!([s(0), s(3), s(4), s(5), s(6), s(9)], [0, 0, 0, 0, 4, 4]);
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

    /// Makes `p` with the ffmpeg CLI (test tool): 200 frames of 640 × 360 at 25 fps, frame n's number burned in as
    /// digits and in binary along the top (`number`), encoded with `enc`. False, with a note, when ffmpeg is missing
    /// or cannot make it (an encoder or filter it lacks, drawtext without fontconfig).
    pub(crate) fn numbered(p: &Path, enc: &[&str]) -> bool {
        let bits = "if(lt(Y\\,40)\\,16+219*bitand(floor(N/pow(2\\,floor(X/40)))\\,1)";
        let vf = format!(
            "geq=lum='{bits}\\,lum(X\\,Y))':cb='if(lt(Y\\,40)\\,128\\,cb(X\\,Y))':\
             cr='if(lt(Y\\,40)\\,128\\,cr(X\\,Y))',drawtext=text='%{{frame_num}}':fontsize=160:\
             fontcolor=white:box=1:boxcolor=black:x=(w-tw)/2:y=(h-th)/2+20"
        );
        let why = match std::process::Command::new("ffmpeg")
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
        {
            Ok(o) if o.status.success() => return true,
            Ok(o) => String::from_utf8_lossy(&o.stderr).trim().to_string(),
            Err(e) => format!("no ffmpeg ({e})"),
        };
        eprintln!("ffmpeg {enc:?} cannot make the clip, skipped: {why}");
        false
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

    /// Plays frames `ks`, after the one shown, holding each while the next decodes the way the player does: each
    /// comes once, in order; `f` checks its picture.
    fn play(
        g: &Gpu,
        v: &Video,
        ks: std::ops::Range<usize>,
        held: &mut Option<Pic>,
        what: &str,
        mut f: impl FnMut(usize, &[u8]),
    ) {
        let t0 = Instant::now();
        for k in ks {
            let x = loop {
                if let (Some(x), n) = v.take(v.times[k], true) {
                    assert_eq!((x.k, n), (k, 0), "{what}: playback");
                    break x;
                }
                assert!(
                    t0.elapsed().as_secs() < 20,
                    "{what}: playback stalled at {k}"
                );
                thread::sleep(Duration::from_millis(1));
            };
            v.acquire(g, &x);
            f(k, &read(g, &v.rings[0], v.view, x.f[0]));
            v.release_before(g, &x);
            *held = Some(x);
        }
    }

    /// B-frames (H.264 from the ffmpeg CLI, frame numbers burned in, `-bf 1` and libx264's defaults): frames are
    /// numbered in presentation order (frame 59 showed for 60 and 136 for 137); seeks show the frame asked for, as
    /// FFmpeg decodes it, and playback holding each frame while the next decodes shows every frame once, in order (the
    /// ring holds what the stream's reordering needs, plus 2 to decode ahead).
    #[test]
    fn bframes() {
        let dir = Tmp::new("bframes");
        let g = Arc::new(Gpu::new().unwrap());
        for (name, enc) in [
            ("bf1", &["-c:v", "libx264", "-bf", "1"][..]),
            ("x264", &["-c:v", "libx264"][..]),
        ] {
            let p = dir.0.join(format!("{name}.mp4"));
            if !numbered(&p, enc) {
                return;
            }
            let m = Mp4::open(&p).unwrap();
            let (ord, w) = order(&m.tracks[0].samples);
            let late = (0..ord.len()).filter(|&k| ord[k] != k).count();
            eprintln!(
                "{name}: {late} of {} frames decode out of order, ring {w}",
                ord.len()
            );
            assert!(late > 0, "{name}: no B-frames");
            let mut v = Video::open(&g, std::slice::from_ref(&p), || {}).unwrap();
            assert_eq!(v.rings[0].slots(), w + OUT - 2);
            let [_, _, vw, _] = v.view.map(|x| x as usize);
            let mut held = None;
            for k in [0, 60, 137, 59, 199, 100, 101, 61, 3, 0] {
                let px = show(&g, &mut v, k, &mut held);
                assert_eq!(number(&px, vw, 1), k, "{name}: seek {k}");
                if [60, 137].contains(&k) {
                    let bad = px
                        .iter()
                        .zip(ffmpeg(&p, 0, k, false))
                        .filter(|(a, b)| *a != b)
                        .count();
                    eprintln!(
                        "{name}: frame {k}: {bad} of {} bytes differ from FFmpeg",
                        px.len()
                    );
                    assert_eq!(bad, 0);
                }
            }
            // play from 0 holding each frame: every one comes, in order
            show(&g, &mut v, 0, &mut held);
            let t0 = Instant::now();
            play(&g, &v, 1..v.times.len(), &mut held, name, |k, px| {
                assert_eq!(number(px, vw, 1), k, "{name}: playback")
            });
            eprintln!(
                "{name}: played {} frames in order in {:.2} s",
                v.times.len(),
                t0.elapsed().as_secs_f64()
            );
            v.release(&g, &held.take().unwrap());
            assert_eq!(v.err(), None);
        }
    }

    /// FFmpeg's frames of `p`, all of them, as NV12 (test tool).
    fn ffmpeg_all(p: &Path) -> Vec<u8> {
        let o = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(p)
            .args(["-fps_mode", "passthrough", "-f", "rawvideo"])
            .args(["-pix_fmt", "nv12", "-"])
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        o.stdout
    }

    /// Open GOPs (ffmpeg CLI, frame numbers burned in): H.265 (libx265, a CRA every 50 frames with RASL frames before
    /// it) and H.264 (libx264, a non-IDR I picture every 32 frames with B-frames before it, sync samples in the MP4).
    /// A seek to such a random access point or just after it restarts there, its leading frames skipped (H.265: they
    /// reported "missing reference"; H.264: later frames showed other frames' pictures, frame 64 for 66); one to a
    /// leading frame decodes the GOP before. Each shows the frame asked for, and so does playback from before one
    /// across it and from a restart across the later ones, every frame FFmpeg's to the byte, with no error. A stream
    /// copy cut that starts at one keeps its leading frames as samples, which no decoder shows: they are no frames
    /// (a seek to one waited for ever), and the cut plays FFmpeg's frames.
    #[test]
    fn open_gop() {
        let dir = Tmp::new("opengop");
        let x265 =
            "log-level=error:keyint=50:min-keyint=50:scenecut=0:open-gop=1:bframes=4:b-adapt=0";
        let x264 = "open-gop=1:scenecut=0:b-adapt=0";
        let g = Arc::new(Gpu::new().unwrap());
        for (name, enc) in [
            (
                "h265",
                &["-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", x265][..],
            ),
            (
                "h264",
                &[
                    "-c:v",
                    "libx264",
                    "-bf",
                    "3",
                    "-g",
                    "32",
                    "-x264-params",
                    x264,
                ][..],
            ),
        ] {
            let p = dir.0.join(format!("{name}.mp4"));
            if !numbered(&p, enc) {
                continue;
            }
            let m = Mp4::open(&p).unwrap();
            let tr = &m.tracks[0];
            let (ord, _) = order(&tr.samples);
            let mut pk = vec![0; ord.len()];
            for (k, &s) in ord.iter().enumerate() {
                pk[s] = k;
            }
            // random access points after the first, and the leading frames: shown before the sync sample they
            // follow in decode order
            let sync: Vec<usize> = (0..ord.len()).filter(|&s| tr.samples[s].sync).collect();
            let ra: Vec<usize> = sync[1..].iter().map(|&s| pk[s]).collect();
            let mut lead: Vec<usize> = (0..ord.len())
                .filter(|&s| {
                    let c = sync[sync.partition_point(|&x| x <= s) - 1];
                    tr.samples[s].pts < tr.samples[c].pts
                })
                .map(|s| pk[s])
                .collect();
            lead.sort();
            eprintln!(
                "{name}: random access frames {ra:?}, {} leading frames, sync samples {sync:?}",
                lead.len()
            );
            assert!(ra.len() >= 2 && lead.contains(&(ra[0] - 1)) && lead.contains(&(ra[1] - 1)));
            let (a, b) = (ra[0], ra[1]);
            let fs = 640 * 360 * 3 / 2;
            let ff = ffmpeg_all(&p);
            assert_eq!(ff.len(), 200 * fs);
            // frame k of `ff` is the one numbered k + k0
            let check = |ff: &[u8], k0: usize, k: usize, px: &[u8], what: &str| {
                let f = &ff[k * fs..(k + 1) * fs];
                let bad = px.iter().zip(f).filter(|(x, y)| x != y).count();
                assert_eq!(
                    (number(px, 640, 1), px.len(), bad),
                    (k + k0, fs, 0),
                    "{name}: {what} {k}"
                );
            };
            let mut v = Video::open(&g, std::slice::from_ref(&p), || {}).unwrap();
            let mut held = None;
            // the target, the frame decoding restarts at: a random access point for one on it or just after it,
            // the GOP before for a leading frame
            for (k, r) in [
                (a + 1, a),
                (a, a),
                (a - 1, 0),
                (b + 2, b),
                (b - 1, a),
                (a - 2, 0),
                (b, b),
                (0, 0),
                (a + 3, a),
            ] {
                let s = start(&sync, &ord, &pk, k);
                eprintln!("{name}: seek {k}: decoding restarts at frame {}", pk[s]);
                assert_eq!(pk[s], r, "{name}: seek {k}");
                check(&ff, 0, k, &show(&g, &mut v, k, &mut held), "seek");
                assert_eq!(v.err(), None, "{name}: seek {k}");
            }
            // across a random access point from before it, and from a restart at one across the later ones
            for (k0, k1) in [(a - 4, a + 4), (b + 1, 200)] {
                check(&ff, 0, k0, &show(&g, &mut v, k0, &mut held), "seek");
                play(&g, &v, k0 + 1..k1, &mut held, name, |k, px| {
                    check(&ff, 0, k, px, "playback")
                });
                eprintln!("{name}: played {k0}..{k1}: FFmpeg's frames");
            }
            v.release(&g, &held.take().unwrap());
            assert_eq!(v.err(), None);
            drop(v);

            // the cut: its first sample is frame a, its leading frames samples after it shown before it
            let c = dir.0.join(format!("{name}-cut.mp4"));
            let o = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-ss"])
                .arg(format!("{}", a as f64 / 25.0))
                .arg("-i")
                .arg(&p)
                .args(["-c", "copy"])
                .arg(&c)
                .output()
                .unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
            let s = Mp4::open(&c).unwrap().tracks[0].samples.clone();
            let kept = (1..s.len()).filter(|&i| s[i].pts < s[0].pts).count();
            let ff = ffmpeg_all(&c);
            let n = ff.len() / fs;
            eprintln!(
                "{name} cut at {a}: {} samples, {kept} leading, FFmpeg shows {n} frames",
                s.len()
            );
            assert!(kept > 0 && n + kept == s.len(), "{name}: cut");
            let mut v = Video::open(&g, std::slice::from_ref(&c), || {}).unwrap();
            assert_eq!(v.times.len(), n, "{name}: cut frames");
            for k in [1, n - 1, 3, 0] {
                check(&ff, a, k, &show(&g, &mut v, k, &mut held), "cut seek");
            }
            play(&g, &v, 1..n, &mut held, name, |k, px| {
                check(&ff, a, k, px, "cut playback")
            });
            v.release(&g, &held.take().unwrap());
            assert_eq!(v.err(), None, "{name}: cut");
        }
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
