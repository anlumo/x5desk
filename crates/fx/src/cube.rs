//! LUTs: `.cube` files (Adobe Cube LUT 1.0, plus Resolve's input-range keywords), our own looks generated from
//! parameters, and a parametric log → Rec.709 transform.

use std::fmt;

/// One table: 1D (`n` entries per channel) or 3D (`n`³ entries, red fastest, then green, then blue), over the input
/// domain `min`…`max` per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct Lut {
    pub n: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub v: Vec<[f32; 3]>,
}

/// A `.cube` file: a 1D table, a 3D table, or both (a 1D shaper applied first).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cube {
    pub title: String,
    pub d1: Option<Lut>,
    pub d3: Option<Lut>,
}

impl Cube {
    /// Parses `.cube` text. Sizes: 1D 2…65536, 3D 2…256. `DOMAIN_MIN` / `DOMAIN_MAX` set the domain of the table(s);
    /// with both tables they apply to the 1D one and the 3D one takes `LUT_3D_INPUT_RANGE` or 0…1.
    /// `LUT_1D_INPUT_RANGE` / `LUT_3D_INPUT_RANGE` (Resolve) set one table's domain on all channels. Other keywords are
    /// ignored.
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut c = Self::default();
        let (mut dom, mut r1, mut r3) = (None, None, None);
        let (mut lo, mut hi) = ([0.0; 3], [1.0; 3]);
        let (mut n1, mut n3) = (0usize, 0usize);
        let mut v = vec![];
        for (no, l) in s.trim_start_matches('\u{feff}').lines().enumerate() {
            let e = |m: &str| format!(".cube line {}: {m}", no + 1);
            let l = l.trim();
            let mut t = l.split_whitespace();
            let Some(k) = t.next() else { continue };
            let num = |t: &mut std::str::SplitWhitespace, n: usize| -> Result<Vec<f32>, String> {
                let x: Vec<f32> = t
                    .map(|x| x.parse::<f32>().map_err(|_| e("bad number")))
                    .collect::<Result<_, _>>()?;
                if x.len() == n && x.iter().all(|v| v.is_finite()) {
                    Ok(x)
                } else {
                    Err(e(&format!("{n} numbers expected")))
                }
            };
            let size = |t: &mut std::str::SplitWhitespace, max: usize| {
                t.next()
                    .and_then(|x| x.parse::<usize>().ok())
                    .filter(|n| (2..=max).contains(n))
                    .ok_or_else(|| e(&format!("size 2…{max} expected")))
            };
            match k {
                _ if k.starts_with('#') => {}
                "TITLE" => c.title = l[5..].trim().trim_matches('"').into(),
                "DOMAIN_MIN" => {
                    lo.copy_from_slice(&num(&mut t, 3)?);
                    dom = Some(());
                }
                "DOMAIN_MAX" => {
                    hi.copy_from_slice(&num(&mut t, 3)?);
                    dom = Some(());
                }
                "LUT_1D_SIZE" => n1 = size(&mut t, 65536)?,
                "LUT_3D_SIZE" => n3 = size(&mut t, 256)?,
                "LUT_1D_INPUT_RANGE" => r1 = Some(num(&mut t, 2)?),
                "LUT_3D_INPUT_RANGE" => r3 = Some(num(&mut t, 2)?),
                _ if k.starts_with(|c: char| c.is_ascii_digit() || "+-.".contains(c)) => {
                    let mut t = l.split_whitespace();
                    let x = num(&mut t, 3)?;
                    v.push([x[0], x[1], x[2]]);
                }
                _ => {}
            }
        }
        if n1 == 0 && n3 == 0 {
            return Err(".cube: no LUT_1D_SIZE or LUT_3D_SIZE".into());
        }
        if v.len() != n1 + n3.pow(3) {
            return Err(format!(
                ".cube: {} entries, {} expected",
                v.len(),
                n1 + n3.pow(3)
            ));
        }
        let r = |r: Option<Vec<f32>>, d: bool| match r {
            Some(r) => ([r[0]; 3], [r[1]; 3]),
            None if d => (lo, hi),
            None => ([0.0; 3], [1.0; 3]),
        };
        let ok = |(lo, hi): ([f32; 3], [f32; 3])| {
            (0..3)
                .all(|i| lo[i] < hi[i])
                .then_some((lo, hi))
                .ok_or(".cube: domain min must be below max")
        };
        let d = dom.is_some();
        if n1 > 0 {
            let (min, max) = ok(r(r1, true))?;
            c.d1 = Some(Lut {
                n: n1 as u32,
                min,
                max,
                v: v[..n1].to_vec(),
            });
        }
        if n3 > 0 {
            let (min, max) = ok(r(r3, d && n1 == 0))?;
            c.d3 = Some(Lut {
                n: n3 as u32,
                min,
                max,
                v: v[n1..].to_vec(),
            });
        }
        Ok(c)
    }
}

/// Writes `.cube` text (both tables: Resolve's input-range keywords; else `DOMAIN_MIN` / `DOMAIN_MAX`).
impl fmt::Display for Cube {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.title.is_empty() {
            writeln!(f, "TITLE \"{}\"", self.title)?;
        }
        let both = self.d1.is_some() && self.d3.is_some();
        for (k, l) in [("1D", &self.d1), ("3D", &self.d3)] {
            let Some(l) = l else { continue };
            writeln!(f, "LUT_{k}_SIZE {}", l.n)?;
            if both {
                writeln!(f, "LUT_{k}_INPUT_RANGE {} {}", l.min[0], l.max[0])?;
            } else {
                writeln!(f, "DOMAIN_MIN {} {} {}", l.min[0], l.min[1], l.min[2])?;
                writeln!(f, "DOMAIN_MAX {} {} {}", l.max[0], l.max[1], l.max[2])?;
            }
        }
        for l in [&self.d1, &self.d3].into_iter().flatten() {
            for c in &l.v {
                writeln!(f, "{:.6} {:.6} {:.6}", c[0], c[1], c[2])?;
            }
        }
        Ok(())
    }
}

const W: [f32; 3] = [0.2126, 0.7152, 0.0722];

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn lin(c: f32) -> f32 {
    let x = c.max(0.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn enc(l: f32) -> f32 {
    let x = l.max(0.0);
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Parameters of a look, evaluated per colour into a 3D LUT. On encoded input c, in this order: to linear; mono mix;
/// gains (scaled to keep luminance); saturation around the luminance Y; split toning (shadow and highlight tints,
/// scaled to luminance 1, weighted (1 − L)² and L² by the encoded luminance L, times `split`); to encoded; contrast
/// x −= contrast·t(1 − t)(1 − 2t); fade x = fade + (1 − fade)·x; clamped to 0…1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grade {
    /// Black-and-white channel weights (summing to 1); None = colour.
    pub mono: Option<[f32; 3]>,
    pub gain: [f32; 3],
    /// −1…1.
    pub sat: f32,
    pub shadow: [f32; 3],
    pub light: [f32; 3],
    /// 0…1.
    pub split: f32,
    /// −1…1.
    pub contrast: f32,
    /// Lifted black, 0…0.2.
    pub fade: f32,
}

impl Default for Grade {
    fn default() -> Self {
        Self {
            mono: None,
            gain: [1.0; 3],
            sat: 0.0,
            shadow: [1.0; 3],
            light: [1.0; 3],
            split: 0.0,
            contrast: 0.0,
            fade: 0.0,
        }
    }
}

impl Grade {
    pub fn eval(&self, c: [f32; 3]) -> [f32; 3] {
        let mut l = c.map(lin);
        if let Some(m) = self.mono {
            l = [dot(l, m); 3];
        }
        let g = dot(self.gain, W);
        l = [0, 1, 2].map(|i| l[i] * self.gain[i] / g);
        let y = dot(l, W);
        l = l.map(|v| (y + (v - y) * (1.0 + self.sat.clamp(-1.0, 1.0))).max(0.0));
        let y = enc(dot(l, W)).clamp(0.0, 1.0);
        let (ws, wh) = ((1.0 - y).powi(2) * self.split, y * y * self.split);
        let (ts, tl) = (dot(self.shadow, W), dot(self.light, W));
        l = [0, 1, 2].map(|i| {
            l[i] * (1.0 + ws * (self.shadow[i] / ts - 1.0))
                * (1.0 + wh * (self.light[i] / tl - 1.0))
        });
        l.map(|v| {
            let t = enc(v).clamp(0.0, 1.0);
            let x = t - self.contrast.clamp(-1.0, 1.0) * t * (1.0 - t) * (1.0 - 2.0 * t);
            (self.fade + (1.0 - self.fade) * x).clamp(0.0, 1.0)
        })
    }

    /// The look as an `n`³ table over 0…1.
    pub fn cube(&self, title: &str, n: u32) -> Cube {
        let s = (n - 1) as f32;
        let v = (0..n.pow(3))
            .map(|i| self.eval([i % n, i / n % n, i / (n * n)].map(|v| v as f32 / s)))
            .collect();
        Cube {
            title: title.into(),
            d1: None,
            d3: Some(Lut {
                n,
                min: [0.0; 3],
                max: [1.0; 3],
                v,
            }),
        }
    }
}

/// Our own looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    Vivid,
    Warm,
    Cool,
    Film,
    Mono,
    TealOrange,
}

impl Look {
    pub const ALL: [Look; 6] = [
        Look::Vivid,
        Look::Warm,
        Look::Cool,
        Look::Film,
        Look::Mono,
        Look::TealOrange,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Look::Vivid => "Vivid",
            Look::Warm => "Warm",
            Look::Cool => "Cool",
            Look::Film => "Film",
            Look::Mono => "Black & White",
            Look::TealOrange => "Teal & Orange",
        }
    }

    pub fn grade(self) -> Grade {
        let g = Grade::default();
        match self {
            Look::Vivid => Grade {
                sat: 0.35,
                contrast: 0.2,
                ..g
            },
            Look::Warm => Grade {
                gain: [1.10, 1.0, 0.86],
                sat: 0.05,
                ..g
            },
            Look::Cool => Grade {
                gain: [0.90, 1.0, 1.12],
                ..g
            },
            Look::Film => Grade {
                gain: [1.03, 1.0, 0.96],
                sat: -0.12,
                shadow: [0.6, 1.0, 1.0],
                light: [1.0, 0.92, 0.75],
                split: 0.12,
                contrast: 0.3,
                fade: 0.04,
                ..g
            },
            Look::Mono => Grade {
                mono: Some([0.30, 0.59, 0.11]),
                contrast: 0.2,
                ..g
            },
            Look::TealOrange => Grade {
                sat: 0.1,
                shadow: [0.15, 0.85, 1.0],
                light: [1.0, 0.65, 0.35],
                split: 0.35,
                contrast: 0.15,
                ..g
            },
        }
    }

    /// The look as a 33³ table.
    pub fn cube(self) -> Cube {
        self.grade().cube(self.name(), 33)
    }
}

/// A camera log curve between encoded v and scene-linear x (0.18 = mid grey), in the shape ARRI LogC, Sony S-Log3 and
/// Panasonic V-Log share: v = c·log10(a·x + b) + d for x ≥ cut, else v = e·x + f. Gamut is taken as Rec.709 (no
/// matrix). Insta360's I-Log curve is unpublished (the app ships it only as an Insta360 LUT, which we don't use), so
/// there is no preset for it: footage shot in I-Log needs its parameters fitted first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Log {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
    pub cut: f32,
}

impl Log {
    /// ARRI LogC3 at EI 800 (ARRI, "ALEXA Log C Curve: Usage in VFX").
    pub const LOGC3: Log = Log {
        a: 5.555556,
        b: 0.052272,
        c: 0.247190,
        d: 0.385537,
        e: 5.367655,
        f: 0.092809,
        cut: 0.010591,
    };
    /// Sony S-Log3 (Sony, "Technical Summary for S-Gamut3.Cine/S-Log3"): log10((x + 0.01) / 0.19)·261.5 / 1023 +
    /// 420 / 1023.
    pub const SLOG3: Log = Log {
        a: 1.0 / 0.19,
        b: 0.01 / 0.19,
        c: 261.5 / 1023.0,
        d: 420.0 / 1023.0,
        e: (171.2103 - 95.0) / (0.01125 * 1023.0),
        f: 95.0 / 1023.0,
        cut: 0.01125,
    };
    /// Panasonic V-Log (Panasonic, "V-Log/V-Gamut Reference Manual").
    pub const VLOG: Log = Log {
        a: 1.0,
        b: 0.00873,
        c: 0.241514,
        d: 0.598206,
        e: 5.6,
        f: 0.125,
        cut: 0.01,
    };

    pub fn encode(&self, x: f32) -> f32 {
        if x >= self.cut {
            self.c * (self.a * x + self.b).log10() + self.d
        } else {
            self.e * x + self.f
        }
    }

    pub fn decode(&self, v: f32) -> f32 {
        if v >= self.e * self.cut + self.f {
            (10f32.powf((v - self.d) / self.c) - self.b) / self.a
        } else {
            (v - self.f) / self.e
        }
    }

    /// Log → Rec.709 as a 4096-entry 1D table over 0…1 for `Fx::input`: x = decode(v)·2^exposure, a highlight
    /// shoulder that keeps everything below 0.6 (mid grey included) and approaches 1 (y = 0.6 + 0.4·(1 − e^(−(x −
    /// 0.6) / 0.4)) above, slope 1 at the joint), then the BT.709 OETF (4.5y below 0.018, 1.099y^0.45 − 0.099).
    pub fn rec709(&self, exposure: f32) -> Lut {
        let n = 4096;
        let k = 2f32.powf(exposure);
        let v = (0..n)
            .map(|i| {
                let x = (self.decode(i as f32 / (n - 1) as f32) * k).max(0.0);
                let y = if x <= 0.6 {
                    x
                } else {
                    0.6 + 0.4 * (1.0 - (-(x - 0.6) / 0.4).exp())
                };
                [if y < 0.018 {
                    4.5 * y
                } else {
                    1.099 * y.powf(0.45) - 0.099
                }; 3]
            })
            .collect();
        Lut {
            n,
            min: [0.0; 3],
            max: [1.0; 3],
            v,
        }
    }
}
