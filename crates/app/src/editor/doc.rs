//! What a project holds beyond crate edit's timeline: the overlays, denoise and stabilisation the export applies to the
//! whole timeline. They ride in the project file as a top-level `"x5desk"` object, which edit's reader ignores (unknown
//! fields), so one file holds the whole project. Colour adjustments are per clip, in edit's clips; a file from before
//! held the timeline's in `"x5desk"`, and its clips take them on load.

use crate::gyro;

/// An overlay's picture.
#[derive(Clone, Debug, PartialEq)]
pub enum Pic {
    /// Text in egui's font at `px` pixels per em.
    Text { s: String, px: f32, rgba: [u8; 4] },
    /// A sticker or watermark PNG.
    Png(String),
    /// A filled disc with a soft edge (nadir patch over the tripod).
    Disc([u8; 4]),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Over {
    pub pic: Pic,
    pub place: fx::Place,
    pub opacity: f32,
}

impl Over {
    pub fn image(&self) -> Result<fx::Image, String> {
        match &self.pic {
            Pic::Text { s, px, rgba } => Ok(fx::Image::text(s, *px, *rgba)),
            Pic::Png(p) => fx::Image::png(&std::fs::read(p).map_err(|e| format!("{p}: {e}"))?),
            Pic::Disc(c) => {
                let n = 256u32;
                let px = (0..n * n)
                    .flat_map(|i| {
                        let (x, y) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
                        let r = ((x - 128.0).hypot(y - 128.0) / 128.0 - 0.96) / 0.04;
                        let a = (1.0 - r.clamp(0.0, 1.0)) * c[3] as f32;
                        [c[0], c[1], c[2], a.round() as u8]
                    })
                    .collect();
                Ok(fx::Image { w: n, h: n, px })
            }
        }
    }
}

/// The timeline-wide effects and settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Extra {
    pub over: Vec<Over>,
    /// RNNoise strength on the clips' audio; None: off.
    pub denoise: Option<f32>,
    pub stab: gyro::Mode,
    pub smooth: f64,
}

impl Default for Extra {
    fn default() -> Extra {
        Extra {
            over: vec![],
            denoise: None,
            stab: Some(stab::Lock::Horizon),
            smooth: gyro::SMOOTH,
        }
    }
}

/// A clip's adjustments: key (in files from before: `"x5desk"."adjust"`), label, range; `adj` lists the fields in this
/// order.
#[rustfmt::skip]
pub const ADJ: [(&str, &str, f32, f32); 20] = [
    ("exposure", "Exposure (EV)", -3.0, 3.0), ("brightness", "Brightness", -1.0, 1.0),
    ("brilliance", "Brilliance", -1.0, 1.0), ("contrast", "Contrast", -1.0, 1.0),
    ("highlights", "Highlights", -1.0, 1.0), ("shadows", "Shadows", -1.0, 1.0),
    ("whites", "Whites", -1.0, 1.0), ("blacks", "Blacks", -1.0, 1.0),
    ("saturation", "Saturation", -1.0, 1.0), ("vibrance", "Vibrance", -1.0, 1.0),
    ("temperature", "Temperature", -1.0, 1.0), ("tint", "Tint", -1.0, 1.0),
    ("clarity", "Clarity", -1.0, 1.0), ("sharpen", "Sharpen", 0.0, 1.0),
    ("auto", "Auto colour", 0.0, 1.0), ("water", "Underwater", 0.0, 1.0),
    ("purple", "Purple fringe", 0.0, 1.0), ("green", "Green fringe", 0.0, 1.0),
    ("blue", "Blue fringe", 0.0, 1.0), ("reach", "Fringe reach (px)", 1.0, 12.0),
];

pub fn adj(p: &mut fx::Params) -> [&mut f32; 20] {
    let fx::Params {
        exposure,
        temperature,
        tint,
        highlights,
        shadows,
        contrast,
        whites,
        blacks,
        brightness,
        saturation,
        vibrance,
        clarity,
        brilliance,
        sharpen,
        purple,
        green,
        blue,
        reach,
        water,
        auto,
        mix: _,
    } = p;
    [
        exposure,
        brightness,
        brilliance,
        contrast,
        highlights,
        shadows,
        whites,
        blacks,
        saturation,
        vibrance,
        temperature,
        tint,
        clarity,
        sharpen,
        auto,
        water,
        purple,
        green,
        blue,
        reach,
    ]
}

/// A JSON value (the reader of `"x5desk"`; numbers as f64).
#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    B(bool),
    N(f64),
    S(String),
    A(Vec<J>),
    O(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::O(m) => m.iter().find(|e| e.0 == k).map(|e| &e.1),
            _ => None,
        }
    }

    fn n(&self) -> Option<f64> {
        match self {
            J::N(x) => Some(*x),
            _ => None,
        }
    }

    fn nums<const N: usize>(&self) -> Option<[f64; N]> {
        match self {
            J::A(v) if v.len() == N => {
                let mut o = [0.0; N];
                for (o, x) in o.iter_mut().zip(v) {
                    *o = x.n()?;
                }
                Some(o)
            }
            _ => None,
        }
    }
}

/// Parses a JSON document (nesting up to 32).
pub fn parse(s: &str) -> Result<J, String> {
    fn ws(b: &[u8], i: &mut usize) {
        while b.get(*i).is_some_and(u8::is_ascii_whitespace) {
            *i += 1;
        }
    }
    fn st(s: &str, i: &mut usize) -> Result<String, String> {
        let b = s.as_bytes();
        *i += 1;
        let mut o = String::new();
        loop {
            match b.get(*i) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    *i += 1;
                    return Ok(o);
                }
                Some(b'\\') => {
                    let c = *b.get(*i + 1).ok_or("bad escape")?;
                    *i += 2;
                    o.push(match c {
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'u' => {
                            let h = s.get(*i..*i + 4).ok_or("bad \\u")?;
                            *i += 4;
                            let mut c = u32::from_str_radix(h, 16).map_err(|_| "bad \\u")?;
                            // a surrogate pair
                            if (0xd800..0xdc00).contains(&c) && s.get(*i..*i + 2) == Some("\\u") {
                                let l = s.get(*i + 2..*i + 6).ok_or("bad \\u")?;
                                let l = u32::from_str_radix(l, 16).map_err(|_| "bad \\u")?;
                                *i += 6;
                                c = 0x10000
                                    + ((c - 0xd800) << 10)
                                    + (l.wrapping_sub(0xdc00) & 0x3ff);
                            }
                            char::from_u32(c).unwrap_or('\u{fffd}')
                        }
                        c => c as char,
                    })
                }
                Some(_) => {
                    let r = &s[*i..];
                    let n = r.find(['"', '\\']).unwrap_or(r.len());
                    o.push_str(&r[..n]);
                    *i += n;
                }
            }
        }
    }
    fn val(s: &str, i: &mut usize, d: u32) -> Result<J, String> {
        let b = s.as_bytes();
        if d > 32 {
            return Err("nested too deep".into());
        }
        ws(b, i);
        let lit = |i: &mut usize, w: &str, v: J| {
            s[*i..]
                .starts_with(w)
                .then(|| {
                    *i += w.len();
                    v
                })
                .ok_or_else(|| format!("bad value at {}", *i))
        };
        match b.get(*i) {
            Some(b'{') => {
                *i += 1;
                let mut m = vec![];
                loop {
                    ws(b, i);
                    match b.get(*i) {
                        Some(b'}') if m.is_empty() => {
                            *i += 1;
                            return Ok(J::O(m));
                        }
                        Some(b'"') => {}
                        _ => return Err(format!("expected a key at {}", *i)),
                    }
                    let k = st(s, i)?;
                    ws(b, i);
                    if b.get(*i) != Some(&b':') {
                        return Err(format!("expected : at {}", *i));
                    }
                    *i += 1;
                    m.push((k, val(s, i, d + 1)?));
                    ws(b, i);
                    match b.get(*i) {
                        Some(b',') => *i += 1,
                        Some(b'}') => {
                            *i += 1;
                            return Ok(J::O(m));
                        }
                        _ => return Err(format!("expected , or }} at {}", *i)),
                    }
                }
            }
            Some(b'[') => {
                *i += 1;
                let mut v = vec![];
                ws(b, i);
                if b.get(*i) == Some(&b']') {
                    *i += 1;
                    return Ok(J::A(v));
                }
                loop {
                    v.push(val(s, i, d + 1)?);
                    ws(b, i);
                    match b.get(*i) {
                        Some(b',') => *i += 1,
                        Some(b']') => {
                            *i += 1;
                            return Ok(J::A(v));
                        }
                        _ => return Err(format!("expected , or ] at {}", *i)),
                    }
                }
            }
            Some(b'"') => st(s, i).map(J::S),
            Some(b't') => lit(i, "true", J::B(true)),
            Some(b'f') => lit(i, "false", J::B(false)),
            Some(b'n') => lit(i, "null", J::Null),
            Some(_) => {
                let n = s[*i..]
                    .find(|c: char| !matches!(c, '0'..='9' | '-' | '+' | '.' | 'e' | 'E'))
                    .unwrap_or(s.len() - *i);
                let x: f64 = s[*i..*i + n]
                    .parse()
                    .map_err(|_| format!("bad number at {}", *i))?;
                *i += n;
                x.is_finite()
                    .then_some(J::N(x))
                    .ok_or_else(|| "number out of range".into())
            }
            None => Err("unexpected end".into()),
        }
    }
    let mut i = 0;
    let v = val(s, &mut i, 0)?;
    ws(s.as_bytes(), &mut i);
    if i < s.len() {
        return Err(format!("data after the document at {i}"));
    }
    Ok(v)
}

fn esc(s: &str) -> String {
    let mut o = String::from('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

const LOCKS: [(gyro::Mode, &str); 5] = [
    (None, "off"),
    (Some(stab::Lock::Free), "free"),
    (Some(stab::Lock::Horizon), "horizon"),
    (Some(stab::Lock::Direction), "direction"),
    (Some(stab::Lock::Forward), "forward"),
];

impl Extra {
    /// The `"x5desk"` object.
    pub fn json(&self) -> String {
        let over: Vec<String> = (self.over.iter())
            .map(|o| {
                let pic = match &o.pic {
                    Pic::Text { s, px, rgba } => {
                        format!("\"text\": {}, \"px\": {px:?}, \"rgba\": {rgba:?}", esc(s))
                    }
                    Pic::Png(p) => format!("\"png\": {}", esc(p)),
                    Pic::Disc(c) => format!("\"disc\": {c:?}"),
                };
                let place = match o.place {
                    fx::Place::Flat { at, w, rot } => {
                        format!("\"flat\": [{:?}, {:?}, {w:?}, {rot:?}]", at[0], at[1])
                    }
                    fx::Place::Sphere {
                        yaw,
                        pitch,
                        fov,
                        roll,
                    } => format!("\"sphere\": [{yaw:?}, {pitch:?}, {fov:?}, {roll:?}]"),
                };
                format!("\n   {{{pic}, {place}, \"opacity\": {:?}}}", o.opacity)
            })
            .collect();
        let lock = LOCKS
            .iter()
            .find(|l| l.0 == self.stab)
            .map_or("off", |l| l.1);
        format!(
            "{{\"over\": [{}],\n  \"denoise\": {},\n  \"stab\": \"{lock}\", \"smooth\": {:?}}}",
            over.join(","),
            self.denoise.map_or("null".into(), |d| format!("{d:?}")),
            self.smooth
        )
    }

    /// From a `"x5desk"` object; missing fields keep their defaults, malformed ones are errors.
    pub fn read(j: &J) -> Result<Extra, String> {
        let mut x = Extra::default();
        if let Some(J::A(v)) = j.get("over") {
            for (i, o) in v.iter().enumerate() {
                let bad = || format!("overlay {i}");
                let pic = if let Some(J::S(s)) = o.get("text") {
                    let c = o.get("rgba").and_then(J::nums::<4>).ok_or_else(bad)?;
                    Pic::Text {
                        s: s.clone(),
                        px: o.get("px").and_then(J::n).ok_or_else(bad)? as f32,
                        rgba: c.map(|v| v.clamp(0.0, 255.0) as u8),
                    }
                } else if let Some(J::S(p)) = o.get("png") {
                    Pic::Png(p.clone())
                } else {
                    let c = o.get("disc").and_then(J::nums::<4>).ok_or_else(bad)?;
                    Pic::Disc(c.map(|v| v.clamp(0.0, 255.0) as u8))
                };
                let place = match (
                    o.get("flat").and_then(J::nums::<4>),
                    o.get("sphere").and_then(J::nums::<4>),
                ) {
                    (Some(f), _) => fx::Place::Flat {
                        at: [f[0] as f32, f[1] as f32],
                        w: f[2] as f32,
                        rot: f[3] as f32,
                    },
                    (_, Some(s)) => fx::Place::Sphere {
                        yaw: s[0] as f32,
                        pitch: s[1] as f32,
                        fov: s[2] as f32,
                        roll: s[3] as f32,
                    },
                    _ => return Err(bad()),
                };
                let opacity = o.get("opacity").and_then(J::n).unwrap_or(1.0) as f32;
                x.over.push(Over {
                    pic,
                    place,
                    opacity,
                });
            }
        }
        x.denoise = match j.get("denoise") {
            None | Some(J::Null) => None,
            Some(v) => Some(v.n().ok_or("denoise")? as f32),
        };
        if let Some(J::S(s)) = j.get("stab") {
            x.stab = LOCKS.iter().find(|l| l.1 == s).ok_or("stab")?.0;
        }
        if let Some(v) = j.get("smooth") {
            x.smooth = v.n().ok_or("smooth")?.clamp(0.0, gyro::SMOOTH_MAX);
        }
        Ok(x)
    }
}

/// A project file: edit's document with the extras added as `"x5desk"`.
pub fn save(pj: &edit::Project, x: &Extra) -> String {
    let s = pj.save();
    let s = s.trim_end().strip_suffix('}').unwrap_or(&s);
    format!("{s},\n \"x5desk\": {}}}\n", x.json())
}

pub fn load(s: &str) -> Result<(edit::Project, Extra), String> {
    let (mut tl, doc) = (edit::Timeline::load(s).map_err(|e| e.0)?, parse(s)?);
    let j = doc.get("x5desk");
    // a file from before: the timeline's adjustments go on the clips without their own
    if let Some(a) = j.and_then(|j| j.get("adjust")) {
        let mut p = fx::Params::default();
        for (d, v) in ADJ.iter().zip(adj(&mut p)) {
            if let Some(n) = a.get(d.0) {
                *v = n.n().ok_or(format!("adjust.{}", d.0))? as f32;
            }
        }
        let def = fx::Params::default();
        (tl.clips.iter_mut().filter(|c| c.adj == def)).for_each(|c| c.adj = p);
    }
    let x = j.map_or(Ok(Extra::default()), Extra::read)?;
    Ok((edit::Project::with(tl).map_err(|e| e.0)?, x))
}
