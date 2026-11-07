//! The X5's microphone in Webcam Mode (PLAN.md §3.2: UAC 1 beside the UVC interfaces, stereo, 48 kHz, S16) through
//! cpal: ALSA's direct hardware device of the card "Insta360 X5", each period timed on CLOCK_MONOTONIC.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

/// The X5's only rate.
pub const RATE: u32 = 48000;
/// Frames a period: 42.7 ms, of which ALSA buffers two.
const PERIOD: u32 = 2048;

/// ns of `k` frames at [`RATE`].
pub fn ns(k: i64) -> i64 {
    k * 125_000 / 6
}

/// The stream: dropping it closes the device.
pub struct Mic {
    _s: cpal::Stream,
    /// ALSA's name of the device (`hw:CARD=<n>,DEV=0`)
    pub id: String,
}

impl Mic {
    /// Opens the X5's capture device and calls `f(pcm, t0, k)` on cpal's thread with each period: interleaved stereo
    /// ±1.0 from frame `k` of a stream whose frame j came at CLOCK_MONOTONIC `t0` + [`ns`]`(j)`. After lost input (an
    /// overrun) a new stream starts at frame 0 with its own `t0`. `e` gets the stream's other errors (the device gone).
    pub fn open(
        mut f: impl FnMut(&[f32], u64, u64) + Send + 'static,
        mut e: impl FnMut(String) + Send + 'static,
    ) -> io::Result<Mic> {
        let err = io::Error::other;
        let x5 = |d: &cpal::Device| {
            d.description().ok().and_then(|x| {
                let hw = x.driver().filter(|p| p.starts_with("hw:"))?;
                (x.name().split(',').next() == Some("Insta360 X5")).then(|| hw.to_owned())
            })
        };
        let (d, id) = cpal::default_host()
            .input_devices()
            .map_err(err)?
            .find_map(|d| x5(&d).map(|id| (d, id)))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no Insta360 X5 microphone"))?;
        let cfg = cpal::StreamConfig {
            channels: 2,
            sample_rate: RATE,
            buffer_size: cpal::BufferSize::Fixed(PERIOD),
        };
        // overruns so far: the callback restarts after each
        let lost = Arc::new(AtomicU64::new(0));
        let l2 = lost.clone();
        let (mut x, mut k, mut t0, mut seen) = (vec![], 0, 0, 0);
        let cb = move |d: &[i16], _: &cpal::InputCallbackInfo| {
            let n = (d.len() / 2) as u64;
            let now = (super::now() * 1e9) as u64;
            if lost.load(Relaxed) != seen {
                (seen, k) = (lost.load(Relaxed), 0);
            }
            // the period's last frame came by `now`, at best just then: the earliest start this gives, rising by
            // 1 ms/s to follow the camera's clock against ours
            let s = now.saturating_sub(ns((k + n) as i64) as u64);
            t0 = match k {
                0 => s,
                _ => s.min(t0 + ns(n as i64) as u64 / 1000),
            };
            x.clear();
            x.extend(d.iter().map(|&v| v as f32 / 32768.0));
            f(&x, t0, k);
            k += n;
        };
        let s = d
            .build_input_stream(
                cfg,
                cb,
                move |x: cpal::Error| match x.kind() {
                    cpal::ErrorKind::Xrun => drop(l2.fetch_add(1, Relaxed)),
                    _ => e(x.to_string()),
                },
                None,
            )
            .map_err(err)?;
        s.play().map_err(err)?;
        Ok(Mic { _s: s, id })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::Mutex,
        thread,
        time::{Duration, Instant},
    };

    /// The X5's microphone (skipped without the X5 in Webcam Mode; it hears the room: nothing of it is kept): found,
    /// 48 kHz stereo in whole periods, not silent, each period handed out soon after its last frame came, its start
    /// against our clock steady (the camera's clock drifts from ours by less than the estimate follows).
    /// `X5_MIC_SECS` (default 5).
    #[test]
    fn real() {
        if !super::super::tests::x5() {
            eprintln!("no X5 in Webcam Mode (2e1a:0005): skipping");
            return;
        }
        let secs = std::env::var("X5_MIC_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5.0);
        // per period: when it was handed out, t0, k, frames, Σx²
        let log = Arc::new(Mutex::new(vec![]));
        let l2 = log.clone();
        let t = Instant::now();
        let m = Mic::open(
            move |x, t0, k| {
                let now = (super::super::now() * 1e9) as u64;
                let e = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>();
                l2.lock().unwrap().push((now, t0, k, x.len() as u64 / 2, e));
            },
            |e| panic!("{e}"),
        )
        .unwrap();
        let open = t.elapsed().as_secs_f64() * 1e3;
        thread::sleep(Duration::from_secs_f64(secs));
        let (id, t) = (m.id.clone(), Instant::now());
        drop(m);
        let close = t.elapsed().as_secs_f64() * 1e3;
        let v = log.lock().unwrap().clone();
        // a stream restarts at frame 0 after an overrun
        let lost = v.iter().filter(|p| p.2 == 0).count() - 1;
        let n = v.iter().map(|p| p.3).sum::<u64>();
        // how late each period came after its last frame (by the estimate then); the earliest start each period gives,
        // its least in the first and the last second: the camera's clock against ours (fast: positive)
        let mut late: Vec<f64> = (v.iter())
            .map(|p| (p.0 as f64 - p.1 as f64 - ns((p.2 + p.3) as i64) as f64) / 1e6)
            .collect();
        let s = |p: &(u64, u64, u64, u64, f64)| p.0 as f64 - ns((p.2 + p.3) as i64) as f64;
        let (a, b) = (v[0].0, v[v.len() - 1].0);
        let min = |r: std::ops::Range<u64>| {
            (v.iter().filter(|p| r.contains(&p.0)))
                .map(s)
                .fold(f64::MAX, f64::min)
        };
        let ppm =
            (min(a..a + 1_000_000_000) - min(b - 1_000_000_000..b + 1)) / (b - a) as f64 * 1e6;
        let db = 10.0 * (v.iter().map(|p| p.4).sum::<f64>() / (2 * n) as f64).log10();
        late.sort_by(f64::total_cmp);
        println!(
            "{id}: opened in {open:.0} ms, closed in {close:.0} ms; {} periods of {:?} frames, {n} frames in {:.2} s \
             ({:.1} Hz by our clock); handed out ms after the last frame: min {:.1} p50 {:.1} max {:.1}; the camera's clock {ppm:+.0} ppm \
             against ours; level {db:.1} dB RMS; {lost} overruns",
            v.len(),
            (v.iter().map(|p| p.3).min(), v.iter().map(|p| p.3).max()),
            (b - a) as f64 / 1e9,
            (n - v[0].3) as f64 / ((b - a) as f64 / 1e9),
            late[0],
            late[late.len() / 2],
            late[late.len() - 1],
        );
        assert!(id.starts_with("hw:") && v.iter().all(|p| p.3 == PERIOD as u64));
        // the estimate follows up to 1000 ppm
        assert!(ppm.abs() < 500.0 && db > -80.0, "{ppm} ppm, {db} dB");
        assert!(late[late.len() / 2] < 20.0, "handed out late");
    }
}
