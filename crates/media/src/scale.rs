//! Shrinking RGBA images (an agent's preview of a still): an area filter, each output pixel the mean of the source
//! area it covers.

/// `w` × `h` fitted within a long edge of `long`, the aspect kept (rounded, at least 1); as it is when it fits.
pub fn fit(w: u32, h: u32, long: u32) -> (u32, u32) {
    let m = w.max(h);
    if m <= long {
        return (w, h);
    }
    let s = |x: u32| ((x as u64 * long as u64 + m as u64 / 2) / m as u64).max(1) as u32;
    (s(w), s(h))
}

/// `px` (`w` × `h` RGBA, rows packed) resampled to `nw` × `nh`: each output pixel is the mean of the source area it
/// covers, a pixel it covers in part weighted by its share. Made for shrinking; enlarging repeats pixels.
pub fn area(px: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    assert!(w > 0 && h > 0 && nw > 0 && nh > 0, "empty image");
    assert_eq!(px.len(), w as usize * h as usize * 4, "not {w} × {h} RGBA");
    let (xs, ys) = (taps(w, nw), taps(h, nh));
    let row = w as usize * 4;
    let mut out = Vec::with_capacity(nw as usize * nh as usize * 4);
    let mut acc = vec![0f32; nw as usize * 4];
    for ty in &ys {
        acc.fill(0.);
        for &(y, wy) in ty {
            let r = px[y * row..][..row].as_chunks::<4>().0;
            for (a, tx) in acc.as_chunks_mut::<4>().0.iter_mut().zip(&xs) {
                for &(x, wx) in tx {
                    let k = wx * wy;
                    for (a, &p) in a.iter_mut().zip(&r[x]) {
                        *a += p as f32 * k;
                    }
                }
            }
        }
        out.extend(acc.iter().map(|&v| v.round().clamp(0., 255.) as u8));
    }
    out
}

/// Per output index: the source indices its span covers, with weights summing to 1.
fn taps(n: u32, m: u32) -> Vec<Vec<(usize, f32)>> {
    let s = n as f64 / m as f64;
    (0..m)
        .map(|i| {
            let (a, b) = (i as f64 * s, ((i + 1) as f64 * s).min(n as f64));
            let v: Vec<_> = (a.floor() as usize..b.ceil() as usize)
                .map(|j| (j, b.min(j as f64 + 1.) - a.max(j as f64)))
                .filter(|&(_, c)| c > 1e-9)
                .collect();
            let t: f64 = v.iter().map(|&(_, c)| c).sum();
            v.into_iter().map(|(j, c)| (j, (c / t) as f32)).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pattern with edges and gradients: `w` × `h` RGBA, alpha opaque.
    fn pattern(w: u32, h: u32) -> Vec<u8> {
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                let check = if (x / 7 + y / 5) % 2 == 0 { 200 } else { 40 };
                let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
                [
                    check,
                    (255. * fx) as u8,
                    (128. + 120. * (fx * 9. + fy * 4.).sin()) as u8,
                    255,
                ]
            })
            .collect()
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let e = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        10. * (255f64.powi(2) / e.max(1e-12)).log10()
    }

    /// The preview sizes: a 2:1 sphere and a 16:9 frame to a long edge of 1568; smaller ones stay.
    #[test]
    fn fits() {
        assert_eq!(fit(7680, 3840, 1568), (1568, 784));
        assert_eq!(fit(1920, 1080, 1568), (1568, 882));
        assert_eq!(fit(1080, 1920, 1568), (882, 1568));
        assert_eq!(fit(8192, 4096, 1568), (1568, 784));
        assert_eq!(fit(1280, 720, 1568), (1280, 720));
        assert_eq!(fit(8192, 1, 1568), (1568, 1));
    }

    /// One colour stays exactly that colour at any ratio.
    #[test]
    fn flat_exact() {
        let (w, h) = (1920, 1080);
        let px = [17u8, 128, 254, 255].repeat((w * h) as usize);
        let (nw, nh) = fit(w, h, 1568);
        let o = area(&px, w, h, nw, nh);
        assert_eq!(o.len(), (nw * nh * 4) as usize);
        assert!(
            o.as_chunks::<4>()
                .0
                .iter()
                .all(|p| p == &[17, 128, 254, 255])
        );
    }

    /// A whole factor is the box filter exactly; another ratio is close to the mean of 16 × 16 point samples per
    /// output pixel (a box filter of the nearest source pixels).
    #[test]
    fn against_box() {
        let (w, h) = (240, 120);
        let px = pattern(w, h);
        let o = area(&px, w, h, 60, 30);
        let mut b = vec![];
        for y in 0..30 {
            for x in 0..60 {
                for c in 0..4 {
                    let s: u32 = (0..16)
                        .map(|i| {
                            px[(((y * 4 + i / 4) * w + x * 4 + i % 4) * 4 + c) as usize] as u32
                        })
                        .sum();
                    b.push(((s as f32) / 16.).round() as u8);
                }
            }
        }
        assert_eq!(o, b);
        let (nw, nh) = (98, 49);
        let o = area(&px, w, h, nw, nh);
        let mut r = vec![];
        for y in 0..nh {
            for x in 0..nw {
                let mut s = [0f64; 4];
                for i in 0..16 {
                    for j in 0..16 {
                        let sx =
                            ((x as f64 + (j as f64 + 0.5) / 16.) * w as f64 / nw as f64) as u32;
                        let sy =
                            ((y as f64 + (i as f64 + 0.5) / 16.) * h as f64 / nh as f64) as u32;
                        let k = ((sy.min(h - 1) * w + sx.min(w - 1)) * 4) as usize;
                        (0..4).for_each(|c| s[c] += px[k + c] as f64);
                    }
                }
                r.extend(s.map(|v| (v / 256.).round() as u8));
            }
        }
        let q = psnr(&o, &r);
        println!("{nw} × {nh}: {q:.1} dB");
        assert!(q > 40., "{q:.1} dB");
    }
}
