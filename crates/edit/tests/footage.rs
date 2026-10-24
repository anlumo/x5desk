//! Frame accuracy against real sample times: the video track of `v606.insv` (29.97, 935 samples) from `X5_SAMPLES`
//! (skipped when unset). Every output frame of timelines cut from it must show the expected sample, at 29.97 and
//! 25 fps output, forwards, backwards, slowed, sped up, ramped and frozen.

use edit::{Clip, Project, Rate};
use media::mp4::Mp4;
use std::path::PathBuf;

#[test]
fn real_sample_times() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        eprintln!("X5_SAMPLES unset: skipped");
        return;
    };
    let m = Mp4::open(d.join("v606.insv")).unwrap();
    let tr = m.tracks.iter().find(|t| &t.handler == b"vide").unwrap();
    let sh = tr.shift(m.timescale);
    let mut pts: Vec<f64> = tr
        .samples
        .iter()
        .map(|s| (s.pts + sh) as f64 / tr.timescale as f64)
        .collect();
    pts.sort_by(f64::total_cmp);
    eprintln!(
        "{} samples, timescale {}, first pts {}, last {}",
        pts.len(),
        tr.timescale,
        pts[0],
        pts[pts.len() - 1]
    );
    // the sample a frame shows: latest pts ≤ src + 1 µs
    let pick = |s: f64| pts.partition_point(|&p| p <= s + 1e-6) as i64 - 1;

    for fps in [
        Rate {
            num: 30000,
            den: 1001,
        },
        Rate { num: 25, den: 1 },
    ] {
        let mut p = Project::new(fps).unwrap();
        let f = p.file("v606.insv", tr.secs()).unwrap();
        // (speed p / q, first sample, samples)
        let cases: [(i64, i64, i64, i64); 6] = [
            (1, 1, 30, 300),
            (2, 1, 100, 200),
            (1, 2, 400, 60),
            (1, 10, 500, 12),
            (16, 1, 0, 928),
            (3, 1, 600, 250),
        ];
        let mut want = vec![];
        let ratio = 30000.0 * fps.den as f64 / (1001.0 * fps.num as f64); // source frames per output frame at 1×
        for rev in [false, true] {
            for &(sp, sq, a, n) in &cases {
                let mut c = Clip::new(f, pts[a as usize], pts[(a + n) as usize]);
                c.speed = vec![(0.0, sp as f64 / sq as f64)];
                c.rev = rev;
                let k = p.tl().clips.len();
                p.insert(k, c).unwrap();
                let v = sp as f64 / sq as f64 * ratio;
                let len = ((n as f64 / v).round() as i64).max(1);
                // forwards frame j shows a + ⌊j·v⌋; backwards b − ⌈(j + 1)·v⌉, never before a
                let e = 1e-9;
                want.extend((0..len).map(|j| {
                    let x = j as f64 * v;
                    if rev {
                        (a + n - ((x + v - e).ceil() as i64)).max(a)
                    } else {
                        a + (x + e).floor() as i64
                    }
                }));
            }
        }
        let tl = p.tl();
        let got: Vec<i64> = (0..tl.len())
            .map(|n| pick(tl.frame(n).unwrap().src))
            .collect();
        assert_eq!(got.len(), want.len(), "{fps:?}");
        for (n, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g, w, "{fps:?} frame {n}");
        }
        eprintln!(
            "{fps:?}: {} output frames, all on the expected sample",
            got.len()
        );

        // a ramp and a freeze: monotonic samples, each frame's source time matches a numeric integration of 1 / speed
        let mut c = Clip::new(f, pts[50], pts[900]);
        c.speed = vec![
            (pts[100], 0.25),
            (pts[300], 6.0),
            (pts[500], 6.0),
            (pts[700], 0.5),
        ];
        let k = tl.clips.len();
        p.insert(k, c.clone()).unwrap();
        let fr = p.freeze(fps.t(p.tl().frames(k).start + 40), 1.5).unwrap();
        let tl = p.tl();
        let r = tl.frames(fr);
        // the freeze holds the frame under the playhead, which then plays on
        let held = pick(tl.frame(r.end).unwrap().src);
        assert!(
            r.clone().all(|n| pick(tl.frame(n).unwrap().src) == held)
                && held >= pick(tl.frame(r.start - 1).unwrap().src)
        );
        let (mut prev, mut worst) = (-1, 0f64);
        let mut n = tl.frames(k).start;
        let (mut s, mut u, h) = (c.a, 0.0, 1e-5);
        for (i, n2) in (tl.frames(k).start..tl.len()).enumerate() {
            let e = tl.frame(n2).unwrap();
            if e.clip.hold > 0.0 {
                continue;
            }
            let x = pick(e.src);
            assert!(x >= prev, "{fps:?} ramp frame {i}");
            prev = x;
            // integrate 1 / speed by RK4 in source time up to this frame's output time
            let tau = fps.t(n - tl.frames(k).start);
            while u + h / c.rate(s) < tau {
                let k1 = 1.0 / c.rate(s);
                let k2 = 1.0 / c.rate(s + h / 2.0);
                let k4 = 1.0 / c.rate(s + h);
                u += h / 6.0 * (k1 + 4.0 * k2 + k4);
                s += h;
            }
            let s_num = s + (tau - u) * c.rate(s);
            worst = worst.max((s_num - e.src).abs());
            n += 1;
        }
        eprintln!(
            "{fps:?}: ramp {} frames, closed form vs numeric source time ≤ {worst:.2e} s",
            n - tl.frames(k).start
        );
        assert!(worst < 1e-6);
    }
}
