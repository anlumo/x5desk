use super::*;
use crate::tests::fixtures;
use std::{f64::consts::TAU, time::Instant};

/// xorshift32: noise without a dependency.
struct Rng(u32);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 as f32 / 2147483648.0 - 1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Click,
    Drums,
    Swing,
}

/// Adds `f(s since the hit)` over `len` s from `at` s to mono `y`.
fn hit(y: &mut [f32], at: f64, len: f64, mut f: impl FnMut(f64) -> f32) {
    let i0 = (at * 48000.0).round() as usize;
    for i in i0..(i0 + (len * 48000.0) as usize).min(y.len()) {
        y[i] += f((i - i0) as f64 / 48000.0);
    }
}

/// A 4/4 track at `bpm`, `secs` long with `lead` s of silence first, as interleaved stereo at 48 kHz, and its beat
/// times (the first 0.37 s after the lead). Click: a 5 ms blip per beat, higher on the bar's first, in white noise at
/// −40 dB. Drums: kick on 1 and 3, snare on 2 and 4, closed hi-hat on the eighths (`Swing`: the off-beat ones 2/3 of
/// the way to the next beat), a bass on 1, the off-beat eighth of 2 and on 3, a melody of harmonic tones on random
/// eighths, a chord pad changing every bar, white noise at −30 dB.
fn synth(k: Kind, bpm: f64, secs: f64, lead: f64, seed: u32) -> (Vec<f32>, Vec<f64>) {
    let n = (secs * 48000.0) as usize;
    let (p, mut y, mut r) = (60.0 / bpm, vec![0f32; n], Rng(seed | 1));
    let beats: Vec<f64> = (0..)
        .map(|i| lead + 0.37 + i as f64 * p)
        .take_while(|&t| t < secs - 0.05)
        .collect();
    let sw = if k == Kind::Swing { 2.0 / 3.0 } else { 0.5 };
    let roots = [55.0, 73.42, 61.74, 82.41];
    for (i, &t) in beats.iter().enumerate() {
        let one = i % 4 == 0;
        if k == Kind::Click {
            let f = if one { 1500.0 } else { 1000.0 };
            hit(&mut y, t, 0.03, |u| {
                ((TAU * f * u).sin() * (-u / 0.005).exp() * 0.5) as f32
            });
            continue;
        }
        if i % 2 == 0 {
            hit(&mut y, t, 0.4, |u| {
                let ph = TAU * (50.0 * u + 2.1 * (1.0 - (-u / 0.03).exp()));
                (0.6 * ph.sin() * (-u / 0.15).exp()) as f32
            });
        } else {
            hit(&mut y, t, 0.25, |u| {
                0.25 * r.f() * (-u / 0.05).exp() as f32
                    + (0.15 * (TAU * 190.0 * u).sin() * (-u / 0.08).exp()) as f32
            });
        }
        for e in [0.0, sw] {
            let mut last = 0f32;
            hit(&mut y, t + e * p, 0.06, |u| {
                let v = r.f();
                let h = v - last;
                last = v;
                0.06 * h * (-u / 0.015).exp() as f32
            });
        }
        let root = roots[i / 4 % 4];
        let at = match i % 4 {
            0 | 2 => Some(t),
            1 => Some(t + sw * p),
            _ => None,
        };
        if let Some(at) = at {
            hit(&mut y, at, 0.45, |u| {
                (0.25 * (TAU * root * u).sin() * (-u / 0.25).exp()) as f32
            });
        }
        for e in [0.0, sw] {
            if r.f() > 0.2 {
                continue;
            }
            let f =
                330.0 * 2f64.powf([0, 2, 4, 7, 9, 12][(r.f().abs() * 5.99) as usize] as f64 / 12.0);
            hit(&mut y, t + e * p, 0.4, |u| {
                let s = (1..=3)
                    .map(|h| (TAU * f * h as f64 * u).sin() / h as f64)
                    .sum::<f64>();
                (0.07 * s * (u / 0.01).min(1.0) * (-u / 0.3).exp()) as f32
            });
        }
        if one {
            let l = 4.0 * p;
            hit(&mut y, t, l, |u| {
                let s = [4.0, 5.0, 6.0]
                    .iter()
                    .map(|m| (TAU * root * m * u).sin())
                    .sum::<f64>();
                (0.04 * s * (u / 0.05).min(1.0) * ((l - u) / 0.05).min(1.0)) as f32
            });
        }
    }
    let nl = if k == Kind::Click { 0.01 } else { 0.03 };
    let lead = (lead * 48000.0) as usize;
    for v in &mut y[lead..] {
        *v += nl * r.f();
    }
    (y.iter().flat_map(|&v| [v, v]).collect(), beats)
}

/// The envelope (or beats) of interleaved stereo `s`, pushed in blocks of 8192 frames as `of` reads a file.
fn onset(s: &[f32]) -> Onset {
    let mut o = Onset::new();
    for b in s.chunks(2 * 8192) {
        o.push(b);
    }
    o
}

/// MIREX's beat F-measure: detections matched one to one to the true beats within ±70 ms.
fn fmeasure(det: &[f64], truth: &[f64]) -> f64 {
    let mut used = vec![false; det.len()];
    let mut hit = 0;
    for &b in truth {
        if let Some(i) = (0..det.len())
            .filter(|&i| !used[i] && (det[i] - b).abs() <= 0.07)
            .min_by(|&i, &j| (det[i] - b).abs().total_cmp(&(det[j] - b).abs()))
        {
            used[i] = true;
            hit += 1;
        }
    }
    let (p, r) = (
        hit as f64 / det.len().max(1) as f64,
        hit as f64 / truth.len().max(1) as f64,
    );
    if hit == 0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// Clicks at irregular times (0.3 to 0.9 s apart, 20 s, in noise): every click is an envelope peak above 2 deviations
/// within 8 ms of its time (`LAT` is their mean offset, frames at their window's start), and no peak sits away from a
/// click.
#[test]
fn onsets() {
    let mut r = Rng(7);
    let mut t = vec![0.5];
    while *t.last().unwrap() < 19.0 {
        t.push(t.last().unwrap() + 0.6 + 0.3 * r.f() as f64);
    }
    let mut y = vec![0f32; 20 * 48000];
    for &c in &t {
        hit(&mut y, c, 0.03, |u| {
            ((TAU * 1000.0 * u).sin() * (-u / 0.005).exp() * 0.5) as f32
        });
    }
    for v in &mut y {
        *v += 0.01 * r.f();
    }
    let s: Vec<f32> = y.iter().flat_map(|&v| [v, v]).collect();
    let e = onset(&s).env(0..BANDS);
    let at = |f: usize| (f * HOP) as f64 / RATE as f64 + LAT;
    let peaks: Vec<f64> = (1..e.len() - 1)
        .filter(|&f| e[f] > 2.0 && e[f] >= e[f - 1] && e[f] > e[f + 1])
        .map(at)
        .collect();
    let off: Vec<f64> = t
        .iter()
        .map(|&c| {
            peaks
                .iter()
                .map(|&q| q - c)
                .min_by(|a, b| a.abs().total_cmp(&b.abs()))
                .unwrap()
        })
        .collect();
    let mean = off.iter().sum::<f64>() / off.len() as f64;
    let worst = off.iter().fold(0f64, |a, &b| a.max(b.abs()));
    let stray = peaks
        .iter()
        .filter(|&&q| t.iter().all(|&c| (q - c).abs() > 0.05))
        .count();
    println!(
        "{} clicks: {} peaks, offset mean {:.1} ms, worst {:.1} ms (LAT {:.1} ms), {stray} away from a click",
        t.len(),
        peaks.len(),
        mean * 1e3,
        worst * 1e3,
        LAT * 1e3
    );
    assert!(mean.abs() < 0.002 && worst < 0.008 && stray == 0);
}

/// The synthetic tracks, 30 s each, at 70 to 180 BPM: click tracks, drums with bass and a melody straight and swung,
/// and drums after 4 s of silence. Beat F-measure (±70 ms, MIREX) and the tempo against the truth: right within 4 %,
/// an octave off (twice or half), or off by 3:2.
#[test]
fn tracks() {
    let tempi = [
        70.0, 76.0, 84.0, 92.0, 100.0, 110.0, 120.0, 128.0, 140.0, 150.0, 165.0, 180.0,
    ];
    let mut rows = vec![];
    let t0 = Instant::now();
    for (j, k) in [Kind::Click, Kind::Drums, Kind::Swing]
        .into_iter()
        .enumerate()
    {
        for (i, &bpm) in tempi.iter().enumerate() {
            rows.push((k, bpm, 0.0, (j * 100 + i) as u32 + 11));
        }
    }
    for (i, bpm) in [96.0, 128.0, 172.0].into_iter().enumerate() {
        rows.push((Kind::Drums, bpm, 4.0, 900 + i as u32));
    }
    let mut res = vec![];
    for &(k, bpm, lead, seed) in &rows {
        let (s, truth) = synth(k, bpm, 30.0, lead, seed);
        let b = onset(&s).finish();
        let det: Vec<f64> = b.t.iter().map(|x| x.0).collect();
        let f = fmeasure(&det, &truth);
        let off: Vec<f64> = (det.iter())
            .filter_map(|&d| truth.iter().map(|&t| d - t).find(|x| x.abs() <= 0.07))
            .collect();
        let q = b.bpm / bpm;
        let cls = match q {
            q if (q - 1.0).abs() < 0.04 => "ok",
            q if (q - 2.0).abs() < 0.08 || (q - 0.5).abs() < 0.02 => "octave",
            q if (q - 1.5).abs() < 0.06 || (q - 2.0 / 3.0).abs() < 0.03 => "3:2",
            _ => "wrong",
        };
        let m = off.iter().sum::<f64>() / off.len().max(1) as f64;
        println!(
            "{k:?} {bpm} BPM{}: tempo {:.1} ({cls}), {} beats of {}, F {f:.3}, mean offset {:.1} ms",
            if lead > 0.0 { " after 4 s silence" } else { "" },
            b.bpm,
            det.len(),
            truth.len(),
            m * 1e3
        );
        res.push((k, f, cls, m));
    }
    for k in [Kind::Click, Kind::Drums, Kind::Swing] {
        let v: Vec<_> = res.iter().filter(|r| r.0 == k).collect();
        let n = |c: &str| v.iter().filter(|r| r.2 == c).count();
        println!(
            "{k:?}: mean F {:.3}, min {:.3}, mean offset {:.1} ms; tempo right {}, octave {}, 3:2 {}, wrong {} of {}",
            v.iter().map(|r| r.1).sum::<f64>() / v.len() as f64,
            v.iter().map(|r| r.1).fold(1.0, f64::min),
            v.iter().map(|r| r.3).sum::<f64>() / v.len() as f64 * 1e3,
            n("ok"),
            n("octave"),
            n("3:2"),
            n("wrong"),
            v.len()
        );
    }
    println!(
        "{} tracks of 30 s in {:.1} s",
        rows.len(),
        t0.elapsed().as_secs_f64()
    );
    assert!(res.iter().all(|r| r.1 >= 0.6 && r.2 != "wrong"));
}

/// Which beats score strong, on the drum tracks (straight and swung, 70 to 180 BPM): per bar, how often the first
/// (kick, bass, a new chord) and the third (kick, bass) outscore the second and fourth (snare), by the bass bands'
/// envelope (`Beats::t`) and by all bands' (the tracker's local score).
#[test]
fn accents() {
    let (mut won, mut bars, mut ratio) = ([0; 2], 0, [vec![], vec![]]);
    for (i, bpm) in [72.0, 96.0, 118.0, 135.0, 160.0, 178.0]
        .into_iter()
        .enumerate()
    {
        for k in [Kind::Drums, Kind::Swing] {
            let (s, truth) = synth(k, bpm, 30.0, 0.0, 300 + i as u32);
            let o = onset(&s);
            let b = o.finish();
            let all = smooth(&o.env(0..BANDS), 60.0 * FPS / b.bpm / 32.0);
            // each true beat's detection, if any: its strength by the bass bands and by all
            let st: Vec<Option<[f32; 2]>> = (truth.iter())
                .map(|&t| {
                    let x = b.t.iter().find(|x| (x.0 - t).abs() <= 0.07)?;
                    Some([x.1, all[((x.0 - LAT) * FPS).round() as usize]])
                })
                .collect();
            for bar in st.as_chunks::<4>().0 {
                if let [Some(a), Some(x), Some(y), Some(z)] = *bar {
                    bars += 1;
                    for j in 0..2 {
                        won[j] += (a[j].min(y[j]) > x[j].max(z[j])) as usize;
                        ratio[j].push((a[j] + y[j]) / (x[j] + z[j]));
                    }
                }
            }
        }
    }
    for (j, name) in ["bass bands", "all bands"].iter().enumerate() {
        ratio[j].sort_by(f32::total_cmp);
        println!(
            "{name}: beats 1 and 3 both over 2 and 4 in {} of {bars} bars; their strength over 2 and 4's: median {:.2}",
            won[j],
            ratio[j][ratio[j].len() / 2]
        );
    }
    assert!(won[0] * 2 > bars && ratio[0][ratio[0].len() / 2] > 1.5);
}

/// Through a file: a drum track at 120 BPM written as a 16-bit WAV gives the same beats from `of` as from its
/// samples (within the 16-bit rounding), and its length.
#[test]
fn file() {
    let (s, truth) = synth(Kind::Drums, 120.0, 20.0, 0.0, 5);
    let p = fixtures().join("beat120.wav");
    let n = (s.len() / 2) as u32;
    let mut w = Vec::with_capacity(44 + 4 * n as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + 4 * n).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    for v in [16u32, 0x0002_0001, 48000, 4 * 48000, 0x0010_0004] {
        w.extend_from_slice(&v.to_le_bytes());
    }
    w.extend_from_slice(b"data");
    w.extend_from_slice(&(4 * n).to_le_bytes());
    for v in &s {
        w.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    std::fs::write(&p, w).unwrap();
    let a = of(&p).unwrap();
    let b = onset(&s).finish();
    let worst =
        a.t.iter()
            .zip(&b.t)
            .fold(0f64, |m, (x, y)| m.max((x.0 - y.0).abs()));
    println!(
        "{}: {} beats ({} from the samples, {} true), {:.1} BPM, {:.3} s, worst difference {:.1} ms",
        p.display(),
        a.t.len(),
        b.t.len(),
        truth.len(),
        a.bpm,
        a.dur,
        worst * 1e3
    );
    assert!(a.t.len() == b.t.len() && worst < 0.005 && (a.dur - 20.0).abs() < 1e-3);
}
