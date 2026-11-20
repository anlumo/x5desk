use super::*;
use crate::{
    audio::Aac,
    mp4::{Codec, Mp4},
    mux::Mux,
    tests::{out, samples, tool},
};
use std::{path::Path, process::Command, time::Instant};

#[test]
fn codebooks_are_complete_prefix_codes() {
    for (k, t) in huff::CB
        .iter()
        .enumerate()
        .skip(1)
        .chain([(12, &&huff::SCF[..])])
    {
        let mut a: Vec<(u64, u32)> = t
            .iter()
            .map(|e| (((e & 0xff_ffff) as u64) << (32 - (e >> 24)), e >> 24))
            .collect();
        let kraft: f64 = a.iter().map(|&(_, l)| 0.5f64.powi(l as i32)).sum();
        a.sort();
        for w in a.windows(2) {
            assert_ne!(
                w[0].0 >> (32 - w[0].1),
                w[1].0 >> (32 - w[0].1),
                "book {k}: prefix"
            );
        }
        assert_eq!(kraft, 1.0, "book {k}");
    }
}

#[test]
fn mdct_matches_definition() {
    let mut e = Enc::new(48000, 2, 192000).unwrap();
    for m in [1024, 128] {
        let z: Vec<f32> = (0..2 * m)
            .map(|n| (n * 7919 % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let mut o = vec![0f32; m];
        let (f, t) = if m == 1024 {
            (&e.fl, &e.twl)
        } else {
            (&e.fs, &e.tws)
        };
        mdct(f, t, &z, &mut o, &mut e.za, &mut e.zb);
        let mut err = 0f64;
        for k in 0..m {
            let x: f64 = (0..2 * m)
                .map(|n| {
                    2.0 * z[n] as f64
                        * (std::f64::consts::PI / m as f64
                            * (n as f64 + 0.5 + m as f64 / 2.0)
                            * (k as f64 + 0.5))
                            .cos()
                })
                .sum();
            err = err.max((x - o[k] as f64).abs());
        }
        println!("MDCT {m}: max error {err:.2e}");
        assert!(err < 1e-4 * m as f64, "{m}");
    }
}

/// The AAC track of a sample file, decoded by our demuxer and symphonia: interleaved stereo.
fn source(name: &str) -> Vec<f32> {
    let m = Mp4::open(samples().unwrap().join(name)).unwrap();
    let t = m
        .tracks
        .iter()
        .find(|t| matches!(t.codec, Codec::Aac { .. }))
        .unwrap();
    let Codec::Aac { asc, .. } = &t.codec else {
        unreachable!()
    };
    let (mut a, mut b, mut v) = (Aac::new(asc).unwrap(), vec![], vec![]);
    for s in &t.samples {
        m.read(s, &mut b).unwrap();
        v.extend_from_slice(a.decode(&b).unwrap());
    }
    v
}

/// Access units with pts and duration, and the encoder (for its counters).
fn encode(pcm: &[f32], ch: u16, br: u32, long_only: bool) -> (Vec<(Vec<u8>, i64, u32)>, Enc) {
    encode_at(pcm, 48000, ch, br, long_only)
}

fn encode_at(
    pcm: &[f32],
    rate: u32,
    ch: u16,
    br: u32,
    long_only: bool,
) -> (Vec<(Vec<u8>, i64, u32)>, Enc) {
    let mut e = Enc::new(rate, ch, br).unwrap();
    e.long_only = long_only;
    let mut v = vec![];
    // uneven pushes: the frame buffering must not care
    for c in pcm.chunks(ch as usize * 777) {
        e.push(c, |a, p, d| v.push((a.to_vec(), p, d)));
    }
    e.finish(|a, p, d| v.push((a.to_vec(), p, d)));
    (v, e)
}

fn f32s(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn write_f32(p: &Path, v: &[f32]) {
    std::fs::write(
        p,
        v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
}

/// ffmpeg decode to f32; fails on any decoder message.
fn ffdecode(src: &Path, dst: &Path) -> Vec<f32> {
    let o = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-i",
            src.to_str().unwrap(),
            "-f",
            "f32le",
            dst.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        o.status.success() && o.stderr.is_empty(),
        "ffmpeg: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    f32s(dst)
}

/// Offset of `d` against `r` (frames, 0..max) by cross-correlation of the first channel.
fn lag(r: &[f32], d: &[f32], ch: usize, max: usize) -> usize {
    let (a, n) = (48000.min(r.len() / ch / 2), 8192);
    (0..max)
        .max_by(|&x, &y| {
            let c = |l: usize| {
                (0..n)
                    .map(|i| r[(a + i) * ch] as f64 * d[(a + i + l) * ch] as f64)
                    .sum::<f64>()
            };
            c(x).total_cmp(&c(y))
        })
        .unwrap()
}

/// Segmental SNR (dB, mean over 1024-frame segments above −70 dBFS) and a noise-to-mask ratio from a model of its
/// own (half-Bark bands, Schroeder spreading, offset 10 + z/2 dB, absolute threshold at 96 dB SPL full scale): mean
/// NMR (dB) and the share of frames with a band above its mask.
fn quality(r: &[f32], d: &[f32], ch: usize) -> (f64, f64, f64) {
    let n = r.len().min(d.len()) / ch;
    let mut seg = vec![];
    for c in 0..ch {
        for s in (0..n - 1024).step_by(1024) {
            let (mut e, mut z) = (0f64, 0f64);
            for i in s..s + 1024 {
                let (a, b) = (r[i * ch + c] as f64, d[i * ch + c] as f64);
                (e, z) = (e + a * a, z + (a - b) * (a - b));
            }
            if e / 1024.0 > 1e-7 {
                seg.push(10.0 * (e / z.max(1e-20)).log10());
            }
        }
    }
    let fft = Fft::new(2048);
    let hann: Vec<f32> = (0..2048)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * (i as f32 + 0.5) / 2048.0).cos())
        .collect();
    let z: Vec<usize> = (0..1024)
        .map(|k| (2.0 * bark(k as f32 * 48000.0 / 2048.0)) as usize)
        .collect();
    let nz = z[1023] + 1;
    // Terhardt's curve without the encoder's hold above 15 kHz
    let terhardt = |f: f32| {
        let k = f.max(20.0) / 1000.0;
        3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4)
    };
    let ath_e: Vec<f64> = (0..nz)
        .map(|b| {
            let a = (0..1024)
                .filter(|&k| z[k] == b)
                .map(|k| terhardt(k as f32 * 48000.0 / 2048.0))
                .fold(f32::MAX, f32::min);
            512.0f64.powi(2) * 1.5 * 10f64.powf((a as f64 - 96.0) / 10.0)
        })
        .collect();
    let (mut nmr, mut bad, mut frames) = (0f64, 0, 0);
    let (mut a, mut b) = (vec![C::default(); 2048], vec![C::default(); 2048]);
    let mut band = |x: &dyn Fn(usize) -> f32, e: &mut Vec<f64>| {
        for i in 0..2048 {
            a[i] = C::new(x(i) * hann[i], 0.0);
        }
        fft.fwd(&a, &mut b);
        e.iter_mut().for_each(|v| *v = 0.0);
        for k in 0..1024 {
            e[z[k]] += ((b[k].r * b[k].r + b[k].i * b[k].i) as f64) * 2048.0 * 2048.0;
        }
    };
    let (mut es, mut en) = (vec![0f64; nz], vec![0f64; nz]);
    for c in 0..ch {
        for s in (0..n - 2048).step_by(1024) {
            band(&|i| r[(s + i) * ch + c], &mut es);
            band(&|i| r[(s + i) * ch + c] - d[(s + i) * ch + c], &mut en);
            if es.iter().sum::<f64>() < 512.0f64.powi(2) * 1.5 * 1e-7 {
                continue;
            }
            let (mut sum, mut worst) = (0f64, 0f64);
            for t in 0..nz {
                let sf = |dz: f64| {
                    15.81 + 7.5 * (dz + 0.474) - 17.5 * (1.0 + (dz + 0.474).powi(2)).sqrt()
                };
                let m: f64 = (0..nz)
                    .map(|j| es[j] * 10f64.powf(sf((t as f64 - j as f64) / 2.0) / 10.0))
                    .sum::<f64>()
                    * 10f64.powf(-(10.0 + 0.25 * t as f64) / 10.0);
                let q = en[t] / m.max(ath_e[t]);
                (sum, worst) = (sum + q, worst.max(q));
            }
            nmr += 10.0 * (sum / nz as f64).log10();
            bad += (worst > 1.0) as usize;
            frames += 1;
        }
    }
    (
        seg.iter().sum::<f64>() / seg.len() as f64,
        nmr / frames as f64,
        bad as f64 / frames as f64,
    )
}

/// v635's audio at 192 kbit/s: FFmpeg decodes it cleanly, the bit rate holds, the quality holds the bar it was judged
/// by against FFmpeg's own encoder at the same rate (printed beside it), and it survives our mux, demux and decoder.
#[test]
fn encodes_v635_like_ffmpeg() {
    if samples().is_none() {
        return;
    }
    let pcm = source("v635.insv");
    let secs = pcm.len() as f64 / 2.0 / 48000.0;
    let d = out().join("aacenc");
    std::fs::create_dir_all(&d).unwrap();
    let t0 = Instant::now();
    let (aus, e) = encode(&pcm, 2, 192000, false);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let bytes: usize = aus.iter().map(|a| a.0.len()).sum();
    let kbps = bytes as f64 * 8.0 / secs / 1e3;
    println!(
        "ours: {} AUs, {} short frames, {kbps:.1} kbit/s, {ms:.0} ms for {secs:.2} s ({} build, shared machine)",
        aus.len(),
        e.shorts,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    assert!((kbps / 192.0 - 1.0).abs() < 0.05, "bit rate {kbps}");
    assert_eq!(aus[0].1, -1024);
    assert_eq!(
        aus.iter().map(|a| a.2 as usize).sum::<usize>() - 1024,
        pcm.len() / 2
    );
    let adts = d.join("v635_ours.aac");
    let mut f = vec![];
    for (a, ..) in &aus {
        f.extend_from_slice(&e.adts(a.len()));
        f.extend_from_slice(a);
    }
    std::fs::write(&adts, f).unwrap();
    let ours = ffdecode(&adts, &d.join("v635_ours.f32"));
    let l = lag(&pcm, &ours, 2, 2200);
    assert_eq!(l, 1024, "ADTS has the priming frame in front");
    let ours = &ours[l * 2..];
    // FFmpeg's native encoder, same input and rate
    let inp = d.join("v635_in.f32");
    write_f32(&inp, &pcm);
    let ffm = d.join("v635_ff.m4a");
    let (a, b) = (inp.to_str().unwrap(), ffm.to_str().unwrap());
    tool(
        "ffmpeg",
        &[
            "-v", "error", "-y", "-f", "f32le", "-ar", "48000", "-ac", "2", "-i", a, "-c:a", "aac",
            "-b:a", "192k", b,
        ],
    );
    let m = Mp4::open(&ffm).unwrap();
    let ffkbps = m.tracks[0]
        .samples
        .iter()
        .map(|s| s.size as f64)
        .sum::<f64>()
        * 8.0
        / secs
        / 1e3;
    let ff = ffdecode(&ffm, &d.join("v635_ff.f32"));
    let lf = lag(&pcm, &ff, 2, 2200);
    let ff = &ff[lf * 2..];
    let (qs, qn, qb) = quality(&pcm, ours, 2);
    let (fs, fn_, fb) = quality(&pcm, ff, 2);
    println!(
        "ours:   {kbps:.1} kbit/s, segmental SNR {qs:.2} dB, NMR {qn:.2} dB, {:.1} % frames above mask",
        qb * 100.0
    );
    println!(
        "ffmpeg: {ffkbps:.1} kbit/s, segmental SNR {fs:.2} dB, NMR {fn_:.2} dB, {:.1} % frames above mask (lag {lf})",
        fb * 100.0
    );
    // the bar: our figures when they were judged in the range of FFmpeg's encoder (9.0.2, 2026-10: 43.88 dB, NMR
    // -18.75 dB, 3 dB allowed), less 1 dB; FFmpeg's own move with the installed version (9.1: 50.40 dB, -29.21 dB)
    assert!(
        qs > 42.17 - 1.0 && qn < -18.53 + 1.0,
        "quality under the bar"
    );
    // MP4 round trip: our mux, our demux, symphonia's decoder; FFmpeg reads the edit list
    let mp4 = d.join("v635_ours.m4a");
    let mut x = Mux::create(&mp4).unwrap();
    let t = x
        .track(
            Codec::Aac {
                asc: e.asc().to_vec(),
                rate: 48000,
                ch: 2,
            },
            48000,
            0,
            0,
            false,
        )
        .unwrap();
    for (a, p, du) in &aus {
        x.write(t, a, *p, *p, *du, true).unwrap();
    }
    x.finish(true).unwrap();
    let m = Mp4::open(&mp4).unwrap();
    let Codec::Aac { asc, rate, ch } = &m.tracks[0].codec else {
        panic!()
    };
    assert_eq!((*rate, *ch, asc.as_slice()), (48000, 2, &e.asc()[..]));
    let (mut dec, mut b, mut rt) = (Aac::new(asc).unwrap(), vec![], vec![]);
    for s in &m.tracks[0].samples {
        m.read(s, &mut b).unwrap();
        rt.extend_from_slice(dec.decode(&b).unwrap());
    }
    let rt = &rt[2048..];
    let same = ours
        .iter()
        .zip(rt)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let ffmp4 = ffdecode(&mp4, &d.join("v635_ours_mp4.f32"));
    println!(
        "mp4: {} samples demuxed, symphonia vs FFmpeg decode max diff {same:.2e}; FFmpeg decodes {} frames (input {})",
        m.tracks[0].samples.len(),
        ffmp4.len() / 2,
        pcm.len() / 2
    );
    assert!(same < 1e-4);
    assert_eq!(
        ffmp4.len(),
        pcm.len(),
        "edit list trims priming and padding"
    );
    assert_eq!(lag(&pcm, &ffmp4, 2, 64), 0);
}

/// Clicks over a quiet tone: short blocks come in at the attacks and cut the pre-echo against long blocks only.
#[test]
fn short_blocks_cut_pre_echo() {
    let n = 48000 * 3;
    let mut pcm = vec![0f32; n * 2];
    let mut clicks = vec![];
    let mut seed = 1u32;
    for i in 0..n {
        let t = (std::f32::consts::TAU * 440.0 * i as f32 / 48000.0).sin() * 0.03;
        (pcm[2 * i], pcm[2 * i + 1]) = (t, t * 0.8);
    }
    for k in 0..11 {
        let at = 12000 + k * 11000 + k * k * 37;
        clicks.push(at);
        for i in 0..480 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((seed >> 9) as f32 / (1 << 23) as f32 - 0.5) * (-(i as f32) / 80.0).exp();
            pcm[2 * (at + i)] += v;
            pcm[2 * (at + i) + 1] += v * 0.7;
        }
    }
    let mut pre = [0f64; 2];
    for (j, long) in [false, true].into_iter().enumerate() {
        let (aus, e) = encode(&pcm, 2, 128000, long);
        let mut dec = Aac::new(&e.asc()).unwrap();
        let mut y = vec![];
        for (a, ..) in &aus {
            y.extend_from_slice(dec.decode(a).unwrap());
        }
        let y = &y[2048..];
        // error energy from 20 to 5 ms before each click against the tone's energy there (the last 5 ms are the
        // attack's own short window and backward masking covers them)
        for &c in &clicks {
            let (mut s, mut z) = (0f64, 0f64);
            for i in c - 960..c - 240 {
                (s, z) = (
                    s + (pcm[2 * i] as f64).powi(2),
                    z + (pcm[2 * i] as f64 - y[2 * i] as f64).powi(2),
                );
            }
            pre[j] += 10.0 * (s / z).log10() / clicks.len() as f64;
        }
        let (q, ..) = quality(&pcm, y, 2);
        println!(
            "{}: {} short frames, SNR 20-5 ms before clicks {:.1} dB, segmental SNR {q:.1} dB",
            if long { "long only" } else { "switching" },
            e.shorts,
            pre[j]
        );
        assert!(long || e.shorts >= clicks.len());
    }
    assert!(
        pre[0] > pre[1] + 6.0,
        "short blocks: {} dB vs long: {} dB",
        pre[0],
        pre[1]
    );
}

/// The other samples at 128 and 256 kbit/s, held to the bar they were judged by against FFmpeg's encoder at the same
/// rates (printed beside them).
#[test]
fn rates_across_samples() {
    if samples().is_none() {
        return;
    }
    let d = out().join("aacenc");
    std::fs::create_dir_all(&d).unwrap();
    // the bar at 128 and 256 kbit/s: our segmental SNR, NMR and share of frames above the mask when they were judged
    // in the range of FFmpeg's encoder (9.0.2, 2026-10: SNR within 6 dB, NMR within 3 dB or under -20 dB, no more
    // frames above the mask; at 128 kbit/s it bought SNR with noise above the mask in most frames), less 1 dB and 2
    // points; FFmpeg's own move with the installed version (v606 at 128 kbit/s: SNR 32.5 dB in 9.0.2, 40.9 dB in 9.1)
    let bars = [
        ("v606.insv", [(30.7, -8.4, 0.137), (51.5, -22.7, 0.0)]),
        ("v627.insv", [(35.2, -10.4, 0.069), (60.6, -27.5, 0.0)]),
    ];
    for (name, bar) in bars {
        let pcm = source(name);
        let secs = pcm.len() as f64 / 96000.0;
        let inp = d.join("rates_in.f32");
        write_f32(&inp, &pcm);
        for (br, (snr, nmr, above)) in [128000, 256000].into_iter().zip(bar) {
            let (aus, e) = encode(&pcm, 2, br, false);
            let kbps = aus.iter().map(|a| a.0.len()).sum::<usize>() as f64 * 8.0 / secs / 1e3;
            let mut dec = Aac::new(&e.asc()).unwrap();
            let mut y = vec![];
            for (a, ..) in &aus {
                y.extend_from_slice(dec.decode(a).unwrap());
            }
            let (qs, qn, qb) = quality(&pcm, &y[2048..], 2);
            write_f32(&d.join("rates_ours.f32"), &y[2048..]);
            let ffm = d.join("rates_ff.m4a");
            let (a, b, r) = (
                inp.to_str().unwrap(),
                ffm.to_str().unwrap(),
                format!("{}k", br / 1000),
            );
            tool(
                "ffmpeg",
                &[
                    "-v", "error", "-y", "-f", "f32le", "-ar", "48000", "-ac", "2", "-i", a,
                    "-c:a", "aac", "-b:a", &r, b,
                ],
            );
            let m = Mp4::open(&ffm).unwrap();
            let fk = m.tracks[0]
                .samples
                .iter()
                .map(|s| s.size as f64)
                .sum::<f64>()
                * 8.0
                / secs
                / 1e3;
            let ff = ffdecode(&ffm, &d.join("rates_ff.f32"));
            let (fs, fn_, fb) = quality(&pcm, &ff, 2);
            println!(
                "{name} {r}: ours {kbps:.1} kbit/s SNR {qs:.1} dB NMR {qn:.1} dB {:.1} % | ffmpeg {fk:.1} kbit/s SNR {fs:.1} dB NMR {fn_:.1} dB {:.1} % ({} short frames)",
                qb * 100.0,
                fb * 100.0,
                e.shorts
            );
            assert!(
                (kbps * 1e3 / br as f64 - 1.0).abs() < 0.05,
                "{name} {r}: {kbps}"
            );
            assert!(
                qs > snr - 1.0 && qn < nmr + 1.0 && qb <= above + 0.02,
                "{name} {r}: under the bar"
            );
        }
    }
}

/// Mono (SCE) at 64 kbit/s through symphonia; stereo at 44.1 kHz.
#[test]
fn mono_sweep() {
    let n = 48000 * 2;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * (200.0 * t + 2000.0 * t * t)).sin() * 0.5
        })
        .collect();
    let (aus, e) = encode(&pcm, 1, 64000, false);
    assert_eq!(e.asc(), [0x11, 0x88]);
    let mut dec = Aac::new(&e.asc()).unwrap();
    let mut y = vec![];
    for (a, ..) in &aus {
        y.extend_from_slice(dec.decode(a).unwrap());
    }
    let (q, nmr, _) = quality(&pcm, &y[1024..], 1);
    let kbps = aus.iter().map(|a| a.0.len()).sum::<usize>() as f64 * 8.0 / 2.0 / 1e3;
    println!("mono sweep: {kbps:.1} kbit/s, segmental SNR {q:.1} dB, NMR {nmr:.1} dB");
    assert!((kbps / 64.0 - 1.0).abs() < 0.05 && q > 20.0);
    let st: Vec<f32> = pcm[..88200].iter().flat_map(|&x| [x, -0.5 * x]).collect();
    let (aus, e) = encode_at(&st, 44100, 2, 128000, false);
    assert_eq!(e.asc(), [0x12, 0x10]);
    let mut dec = Aac::new(&e.asc()).unwrap();
    let mut y = vec![];
    for (a, ..) in &aus {
        y.extend_from_slice(dec.decode(a).unwrap());
    }
    let (q, ..) = quality(&st, &y[2048..], 2);
    println!("44.1 kHz stereo sweep: segmental SNR {q:.1} dB");
    assert!(q > 20.0);
}

/// At the AAC-LC limit (6144 bits per channel and frame) no AU exceeds it, whether the input wants more bits (white
/// noise) or fewer (silence, a tone: fill elements); above the limit the encoder refuses. FFmpeg decodes without a
/// message.
#[test]
fn au_limit() {
    let d = out().join("aacenc");
    std::fs::create_dir_all(&d).unwrap();
    let mut r = 1u32;
    for (rate, ch, br) in [(48000, 2, 576000), (48000, 1, 288000), (44100, 2, 529200)] {
        assert!(Enc::new(rate, ch, br + 1).is_err());
        for name in ["noise", "silence", "tone"] {
            let pcm: Vec<f32> = (0..rate as usize * 3 * ch as usize)
                .map(|i| match name {
                    "noise" => {
                        r = r.wrapping_mul(1664525).wrapping_add(1013904223);
                        r as i32 as f32 / 2147483648.0
                    }
                    "tone" => ((i / ch as usize) as f32 * 0.13).sin() * 0.5,
                    _ => 0.0,
                })
                .collect();
            let (aus, e) = encode_at(&pcm, rate, ch, br, false);
            let max = aus.iter().map(|a| a.0.len()).max().unwrap();
            // over the frames sent (priming and padding included)
            let secs = (aus.len() * N) as f64 / rate as f64;
            let kbps = aus.iter().map(|a| a.0.len()).sum::<usize>() as f64 * 8.0 / secs / 1e3;
            let p = d.join(format!("limit_{name}_{ch}_{rate}.aac"));
            let mut adts = vec![];
            for (a, ..) in &aus {
                adts.extend_from_slice(&e.adts(a.len()));
                adts.extend_from_slice(a);
            }
            std::fs::write(&p, adts).unwrap();
            ffdecode(&p, &d.join("limit.f32"));
            println!(
                "{rate} Hz {ch} ch {br} bit/s {name}: largest AU {} bits (limit {}), {kbps:.1} kbit/s",
                max * 8,
                6144 * ch as usize
            );
            assert!(max * 8 <= 6144 * ch as usize, "{name}");
            assert!(kbps <= br as f64 / 1e3 * 1.001, "{name}");
        }
    }
}
