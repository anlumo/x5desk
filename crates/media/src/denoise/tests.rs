//! The reference is RNNoise's C code, built from source as a test tool only: `X5_RNNOISE=<dir>` holds `ref_c` (the
//! portable C path) and `ref_sse` (the default x86-64 build), each run as `<bin> in.f32 out.f32 vad.f32` on f32 mono
//! at 16-bit scale (a ten-line driver around `rnnoise_process_frame`). Both are linked from rnnoise 70f1d25's
//! denoise.c rnn.c pitch.c kiss_fft.c celt_lpc.c nnet.c nnet_default.c parse_lpcnet_weights.c rnnoise_tables.c and
//! the model's rnnoise_data.c with `gcc -O2 -DDISABLE_DEBUG_FLOAT`; `ref_c` adds `-U__SSE2__ -U__SSE__ -U__SSE2_MATH__
//! -U__SSE_MATH__`. `X5_SPEECH=<file>` is a clean speech recording (anything crate music reads); without it
//! `reduces_noise` skips and the others use a synthetic voice.

use super::*;
use crate::{music, tests::out};
use std::{f64::consts::TAU, process::Command, time::Instant};

/// Clean speech, mono 48 kHz: `secs` from 20 s into `X5_SPEECH` high-passed at 100 Hz (4th-order Butterworth:
/// LibriVox recordings often carry mains hum, which the denoiser rightly removes), or a synthetic voice (gliding
/// harmonics in syllables with pauses).
fn speech(secs: f64) -> (Vec<f32>, &'static str) {
    let n = (secs * 48000.0) as usize;
    if let Some(p) = std::env::var_os("X5_SPEECH") {
        let s = music::load(&p).unwrap();
        let mut x: Vec<f64> = s
            .chunks(2)
            .skip(20 * 48000)
            .take(n)
            .map(|f| f[0] as f64)
            .collect();
        for q in [0.5412, 1.3066] {
            // RBJ high-pass biquad, direct form I
            let w = TAU * 100.0 / 48000.0;
            let a = w.sin() / (2.0 * q);
            let (b0, b1, a0, a1, a2) = (
                (1.0 + w.cos()) / 2.0,
                -(1.0 + w.cos()),
                1.0 + a,
                -2.0 * w.cos(),
                1.0 - a,
            );
            let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
            for v in &mut x {
                let y = (b0 * *v + b1 * x1 + b0 * x2 - a1 * y1 - a2 * y2) / a0;
                (x2, x1, y2, y1) = (x1, *v, y1, y);
                *v = y;
            }
        }
        return (x.iter().map(|&v| v as f32).collect(), "recorded speech");
    }
    let v = (0..n)
        .map(|i| {
            let t = i as f64 / 48000.0;
            let f = 110.0 + 40.0 * (TAU * 0.4 * t).sin();
            let gate =
                ((TAU * 2.5 * t).sin().max(0.0) * if (t % 3.0) < 2.2 { 1.0 } else { 0.0 }).powi(2);
            let v: f64 = (1..30)
                .map(|h| (TAU * f * h as f64 * t).sin() / (h as f64).powf(1.2))
                .sum();
            (v * gate * 0.08) as f32
        })
        .collect();
    (v, "synthetic voice")
}

/// Wind on a microphone: brown noise in gusts (slow random level, bursts), high-passed at ~20 Hz.
fn wind(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    let mut rnd = move || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 8) as f64 / (1u32 << 24) as f64 - 0.5
    };
    let (mut b, mut hp, mut lvl, mut gust) = (0f64, 0f64, 0.5f64, 0f64);
    (0..n)
        .map(|_| {
            b = 0.998 * b + rnd() * 0.1;
            hp += (b - hp) * 0.0026;
            gust = 0.9999 * gust + rnd() * 0.002;
            lvl = (0.3 + 4.0 * gust.abs()).min(1.5);
            ((b - hp) * lvl) as f32
        })
        .collect()
}

fn hiss(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f64 / (1u32 << 24) as f64 - 0.5) as f32
        })
        .collect()
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

/// `s` plus `noise` scaled to `snr` dB under it.
fn mix(s: &[f32], noise: &[f32], snr: f64) -> Vec<f32> {
    let k = rms(s) / rms(noise) / 10f64.powf(snr / 20.0);
    s.iter()
        .zip(noise)
        .map(|(a, b)| a + (k * *b as f64) as f32)
        .collect()
}

fn ours(x: &[f32], strength: f32) -> (Vec<f32>, Vec<f32>) {
    let mut d = Denoise::new();
    d.strength = strength;
    let (mut y, mut v) = (x[..x.len() / FRAME * FRAME].to_vec(), vec![]);
    for f in y.chunks_exact_mut(FRAME) {
        v.push(d.frame(f));
    }
    (y, v)
}

/// SNR (dB) of `y` delayed by DELAY against `c`, and segmental SNR over 20 ms frames where `c` is above −50 dBFS
/// (each clamped to −10..35 dB).
fn snr(c: &[f32], y: &[f32]) -> (f64, f64) {
    let n = c.len().min(y.len() - DELAY);
    let (mut s, mut e) = (0f64, 0f64);
    let mut seg = vec![];
    for w in (0..n - 960).step_by(960) {
        let (mut ws, mut we) = (0f64, 0f64);
        for i in w..w + 960 {
            let d = y[i + DELAY] as f64 - c[i] as f64;
            (ws, we) = (ws + (c[i] as f64).powi(2), we + d * d);
        }
        (s, e) = (s + ws, e + we);
        if ws / 960.0 > 1e-5 {
            seg.push((10.0 * (ws / we.max(1e-20)).log10()).clamp(-10.0, 35.0));
        }
    }
    (
        10.0 * (s / e).log10(),
        seg.iter().sum::<f64>() / seg.len() as f64,
    )
}

/// Bit-exact against the portable C path. The default x86 build (approximate reciprocal in tanh / sigmoid, unsigned
/// int8 input with its own bias) is not bit-exact with that path itself; the network's state drifts apart a little,
/// by an amount that depends on the input (its SNR against the portable build measured 25-68 dB, peaks 12-79 LSB, on
/// noise, speech, tones). Against what the denoiser changes it stays put (51-70 dB below on those and the inputs
/// here), so that is the bound: a broken build or another model would differ on the order of that change.
#[test]
fn matches_reference() {
    let Some(dir) = std::env::var_os("X5_RNNOISE").map(std::path::PathBuf::from) else {
        eprintln!("X5_RNNOISE not set: skipping");
        return;
    };
    // debug builds run the network at ~70 ms a frame
    let (s, what) = speech(if cfg!(debug_assertions) { 2.0 } else { 20.0 });
    let x = mix(&s, &wind(s.len(), 3), 5.0);
    let d = out().join("denoise");
    std::fs::create_dir_all(&d).unwrap();
    let inp = d.join("ref_in.f32");
    std::fs::write(
        &inp,
        x.iter()
            .flat_map(|v| (v * 32768.0).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let t = Instant::now();
    let (y, vad) = ours(&x, 1.0);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "{what} + wind at 5 dB, {} frames: ours {ms:.0} ms ({:.2} ms per 10 ms frame, {} build, shared machine)",
        vad.len(),
        ms / vad.len() as f64,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    let f32s = |p: &std::path::Path| -> Vec<f32> {
        std::fs::read(p)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    // what the denoiser changes: the yardstick for the x86 build's drift
    let dn: f64 = x
        .iter()
        .zip(&y[DELAY..])
        .map(|(a, b)| (*b as f64 - *a as f64).powi(2))
        .sum();
    for bin in ["ref_c", "ref_sse"] {
        let (o, v) = (
            d.join(format!("{bin}_out.f32")),
            d.join(format!("{bin}_vad.f32")),
        );
        let st = Command::new(dir.join(bin))
            .args([&inp, &o, &v])
            .status()
            .unwrap();
        assert!(st.success(), "{bin}");
        let (r, rv) = (f32s(&o), f32s(&v));
        assert_eq!((r.len(), rv.len()), (y.len(), vad.len()));
        let (mut mx, mut e, mut sig, mut same) = (0f64, 0f64, 0f64, 0);
        for (a, b) in y.iter().zip(&r) {
            let b = *b as f64 / 32768.0;
            let df = (*a as f64 - b).abs();
            (mx, e, sig, same) = (
                mx.max(df),
                e + df * df,
                sig + b * b,
                same + (*a as f64 == b) as usize,
            );
        }
        let vd = vad
            .iter()
            .zip(&rv)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let db = |a: f64| {
            if e == 0.0 {
                "inf".into()
            } else {
                format!("{:.1}", 10.0 * (a / e).log10())
            }
        };
        println!(
            "vs {bin}: {same} of {} samples identical, max difference {:.3} LSB (16-bit), SNR {} dB, {} dB below the \
             denoiser's change, VAD max difference {vd:.1e}",
            y.len(),
            mx * 32768.0,
            db(sig),
            db(dn),
        );
        if bin == "ref_c" {
            assert!(
                mx == 0.0 && vd == 0.0,
                "not bit-exact against the portable C path"
            );
        } else {
            assert!(dn / e > 1e4, "the x86 build is not the same denoiser");
        }
    }
}

/// Noise reduction on speech + noise mixes: SNR and segmental SNR before and after, the noise left in the speech
/// pauses, and what the denoiser does to clean speech. Needs real speech (the network takes the synthetic voice for
/// noise).
#[test]
fn reduces_noise() {
    if std::env::var_os("X5_SPEECH").is_none() || cfg!(debug_assertions) {
        eprintln!(
            "needs X5_SPEECH and a release build (debug runs the network at ~70 ms a frame): skipping"
        );
        return;
    }
    let (s, what) = speech(40.0);
    let n = s.len();
    // speech pauses: 20 ms windows where the clean speech is below −55 dBFS
    let pauses: Vec<usize> = (0..n / 960 - 1)
        .filter(|&w| rms(&s[w * 960..w * 960 + 960]) < 10f64.powf(-55.0 / 20.0))
        .collect();
    let (clean, _) = ours(&s, 1.0);
    let (cs, cseg) = snr(&s, &clean);
    println!(
        "{what}, {:.0} s, {} pauses of 20 ms; clean speech through the denoiser: SNR {cs:.1} dB, segmental {cseg:.1} dB",
        n as f64 / 48000.0,
        pauses.len()
    );
    for (name, noise) in [("wind", wind(n, 11)), ("hiss", hiss(n, 5))] {
        for db in [0.0, 5.0, 10.0] {
            let x = mix(&s, &noise, db);
            let (y, vad) = ours(&x, 1.0);
            let nz: Vec<f32> = x.iter().zip(&s).map(|(a, b)| a - b).collect();
            // what is left in the pauses, against the noise that was there
            let (mut ein, mut eout) = (0f64, 0f64);
            for &w in &pauses {
                let r = w * 960..w * 960 + 960;
                ein += nz[r.clone()]
                    .iter()
                    .map(|v| (*v as f64).powi(2))
                    .sum::<f64>();
                eout += y[r.start + DELAY..r.end + DELAY]
                    .iter()
                    .map(|v| (*v as f64).powi(2))
                    .sum::<f64>();
            }
            let (i0, iseg) = snr(&s, &[&[0f32; DELAY][..], &x].concat());
            let (o0, oseg) = snr(&s, &y);
            let nr = 10.0 * (ein / eout).log10();
            let voiced = vad.iter().filter(|&&v| v > 0.5).count() as f64 / vad.len() as f64;
            println!(
                "{name} at {db:>2} dB: SNR {i0:5.1} -> {o0:5.1} dB, segmental {iseg:5.1} -> {oseg:5.1} dB, noise in pauses -{nr:.1} dB, VAD > 0.5 in {:.0} % of frames",
                voiced * 100.0
            );
            assert!(
                o0 > i0 + 3.0 && oseg > iseg + 3.0 && nr > 12.0,
                "{name} {db}"
            );
        }
    }
}

/// Strength 0 gives the input back DELAY samples late, bit for bit; in between it mixes linearly.
#[test]
fn strength() {
    let (s, _) = speech(1.0);
    let x = mix(&s, &hiss(s.len(), 9), 10.0);
    let (dry, _) = ours(&x, 0.0);
    assert_eq!(&dry[DELAY..], &x[..dry.len() - DELAY]);
    let (wet, _) = ours(&x, 1.0);
    let (half, _) = ours(&x, 0.5);
    let e = half
        .iter()
        .zip(&wet)
        .zip(&dry)
        .map(|((h, w), d)| (h - (d + 0.5 * (w - d))).abs())
        .fold(0f32, f32::max);
    assert!(e < 1e-6, "{e}");
}

/// Stereo: two sides alike give exactly the mono result on both; with the voice louder on the left and independent
/// wind on each side, the level difference between the sides (per band and 20 ms) changes far less through the
/// linked gains than through two independent denoisers.
#[test]
fn stereo() {
    let secs = if cfg!(debug_assertions) { 1.0 } else { 10.0 };
    let (s, what) = speech(secs);
    let n = s.len() / FRAME * FRAME;
    let x = mix(&s, &wind(n, 3), 5.0);
    let (mono, _) = ours(&x, 1.0);
    let mut st = Stereo::new();
    let (mut l, mut r) = (x[..n].to_vec(), x[..n].to_vec());
    for (a, b) in l.chunks_exact_mut(FRAME).zip(r.chunks_exact_mut(FRAME)) {
        st.frame(a, b);
    }
    assert!(l == mono && r == mono);
    // left: voice + wind; right: the voice 4.4 dB down + other wind
    let xs = [
        mix(&s, &wind(n, 7), 5.0),
        mix(
            &s.iter().map(|v| v * 0.6).collect::<Vec<_>>(),
            &wind(n, 8),
            0.6,
        ),
    ];
    let mut lk = xs.clone();
    let mut st = Stereo::new();
    let [a, b] = &mut lk;
    for (a, b) in a.chunks_exact_mut(FRAME).zip(b.chunks_exact_mut(FRAME)) {
        st.frame(a, b);
    }
    let ind = [ours(&xs[0], 1.0).0, ours(&xs[1], 1.0).0];
    // band levels of 1024-sample Hann windows every 960 samples, input against output (DELAY later)
    let fft = crate::fft::Fft::new(1024);
    let (mut za, mut zb) = (
        vec![crate::fft::C::default(); 1024],
        vec![crate::fft::C::default(); 1024],
    );
    let edges = [10, 21, 43, 85, 171, 341]; // 470 Hz … 16 kHz
    let mut lv = |x: &[f32], at: usize| -> [f64; 5] {
        for (i, z) in za.iter_mut().enumerate() {
            let w = 0.5 - 0.5 * (TAU * (i as f64 + 0.5) / 1024.0).cos();
            *z = crate::fft::C::new(x[at + i] * w as f32, 0.0);
        }
        fft.fwd(&za, &mut zb);
        std::array::from_fn(|b| {
            (edges[b]..edges[b + 1])
                .map(|k| (zb[k].r as f64).powi(2) + (zb[k].i as f64).powi(2))
                .sum::<f64>()
        })
    };
    let mut dev = |y: &[Vec<f32>; 2]| {
        let (mut sq, mut m) = (0f64, 0usize);
        for at in (0..n - DELAY - 1024).step_by(960) {
            let (il, ir, ol, or) = (
                lv(&xs[0], at),
                lv(&xs[1], at),
                lv(&y[0], at + DELAY),
                lv(&y[1], at + DELAY),
            );
            for b in 0..5 {
                // bands the denoiser left audible on both sides (above −70 dBFS; the FFT scales by 1/n, so a
                // full-scale sine peaks at 1/16 under the Hann window)
                if ol[b].min(or[b]) > 1e-7 / 16.0 {
                    let d = 10.0 * (ol[b] / or[b]).log10() - 10.0 * (il[b] / ir[b]).log10();
                    (sq, m) = (sq + d * d, m + 1);
                }
            }
        }
        (sq / m.max(1) as f64).sqrt()
    };
    let (dl, di) = (dev(&lk), dev(&ind));
    println!(
        "{what}, {secs} s: level difference between the sides changes by {dl:.2} dB RMS linked, {di:.2} dB independent"
    );
    assert!(dl < 0.5 * di, "{dl} {di}");
}
