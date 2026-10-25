//! Project files: a JSON document, written and read by hand (no serde: one flat schema). `"x5edit"` holds the
//! format version; unknown fields are ignored, so later versions may add some. Reading never panics: malformed text,
//! wrong types, out-of-range values and broken invariants are errors ([`Timeline::check`]).

use crate::{
    Clip, Error, File, Music, Rate, Result, Timeline, bad,
    view::{Ease, Key, Proj, View},
};
use glam::Quat;
use std::{fmt::Write, str::FromStr};

pub const VERSION: u32 = 1;
/// Nesting limit while parsing (the schema needs 6).
const DEPTH: u32 = 16;

const EASES: [(Ease, &str); 6] = [
    (Ease::Linear, "linear"),
    (Ease::Smooth, "smooth"),
    (Ease::In, "in"),
    (Ease::Out, "out"),
    (Ease::InOut, "inout"),
    (Ease::Hold, "hold"),
];

/// A clip's adjustments by name: every `fx::Params` field but `mix` (the look's intensity is `Clip::amt`). A file
/// holds the ones that differ from fx's defaults; missing ones are the defaults (files from before had none).
macro_rules! adj {
    ($($f:ident)*) => {
        pub(crate) fn adj(p: &mut fx::Params) -> [(&'static str, &mut f32); 20] {
            let fx::Params { $($f,)* mix: _ } = p;
            [$((stringify!($f), $f)),*]
        }
    };
}
adj!(exposure brightness brilliance contrast highlights shadows whites blacks saturation vibrance temperature tint
    clarity sharpen auto water purple green blue reach);

fn esc(o: &mut String, s: &str) {
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => _ = write!(o, "\\u{:04x}", c as u32),
            c => o.push(c),
        }
    }
    o.push('"');
}

fn keys(o: &mut String, ks: &[Key], ind: &str) {
    o.push('[');
    for (i, k) in ks.iter().enumerate() {
        let (q, v) = (k.v.q, &k.v);
        let (p, d) = match v.proj {
            Proj::Persp => ("persp", None),
            Proj::Panini(d) => ("panini", Some(d)),
            Proj::Planet(d) => ("planet", Some(d)),
            Proj::Equirect => ("equirect", None),
        };
        let sep = if i > 0 { "," } else { "" };
        _ = write!(
            o,
            "{sep}\n{ind}{{\"t\": {:?}, \"q\": [{:?}, {:?}, {:?}, {:?}], ",
            k.t, q.x, q.y, q.z, q.w
        );
        _ = write!(o, "\"fov\": {:?}, \"proj\": \"{p}\"", v.fov);
        if let Some(d) = d {
            _ = write!(o, ", \"d\": {d:?}");
        }
        let e = EASES.iter().find(|x| x.0 == k.e).unwrap().1;
        _ = write!(o, ", \"ease\": \"{e}\"}}");
    }
    o.push(']');
}

fn pairs<T: std::fmt::Debug>(o: &mut String, v: &[(f64, T)]) {
    o.push('[');
    for (i, p) in v.iter().enumerate() {
        _ = write!(o, "{}[{:?}, {:?}]", if i > 0 { ", " } else { "" }, p.0, p.1);
    }
    o.push(']');
}

impl Timeline {
    pub fn save(&self) -> String {
        let mut o = String::new();
        let Rate { num, den } = self.out.fps;
        _ = write!(
            o,
            "{{\"x5edit\": {VERSION},\n \"fps\": [{num}, {den}],\n \"files\": ["
        );
        for (i, f) in self.files.iter().enumerate() {
            o.push_str(if i > 0 {
                ",\n  {\"path\": "
            } else {
                "\n  {\"path\": "
            });
            esc(&mut o, &f.path);
            _ = write!(o, ", \"dur\": {:?}}}", f.dur);
        }
        o.push_str("],\n \"clips\": [");
        for (i, c) in self.clips.iter().enumerate() {
            let sep = if i > 0 { "," } else { "" };
            _ = write!(
                o,
                "{sep}\n  {{\"file\": {}, \"a\": {:?}, \"b\": {:?}, \"hold\": {:?}, ",
                c.file, c.a, c.b, c.hold
            );
            _ = write!(
                o,
                "\"rev\": {}, \"mute\": {}, \"amt\": {:?}, \"look\": ",
                c.rev, c.mute, c.amt
            );
            esc(&mut o, &c.look);
            o.push_str(", \"adjust\": {");
            let (mut a, mut d) = (c.adj, fx::Params::default());
            let set = adj(&mut a).into_iter().zip(adj(&mut d));
            for (i, ((k, v), _)) in set.filter(|(x, y)| *x.1 != *y.1).enumerate() {
                _ = write!(o, "{}\"{k}\": {v:?}", if i > 0 { ", " } else { "" });
            }
            o.push_str("}, \"speed\": ");
            pairs(&mut o, &c.speed);
            o.push_str(", \"vol\": ");
            pairs(&mut o, &c.vol);
            o.push_str(", \"view\": ");
            keys(&mut o, &c.view, "   ");
            o.push('}');
        }
        o.push_str("],\n \"view\": ");
        keys(&mut o, &self.out.view, "  ");
        match &self.out.music {
            Some(m) => {
                _ = write!(
                    o,
                    ",\n \"music\": {{\"file\": {}, \"offset\": {:?}, \"gain\": {:?}, \"duck\": {}}}}}\n",
                    m.file, m.offset, m.gain, m.duck
                )
            }
            None => o.push_str(",\n \"music\": null}\n"),
        }
        o
    }

    pub fn load(s: &str) -> Result<Timeline> {
        let mut p = P {
            s: s.as_bytes(),
            src: s,
            i: 0,
            depth: 0,
        };
        let v = p.val()?;
        p.ws();
        if p.i < p.s.len() {
            return p.err("data after the project");
        }
        let ver: u32 = num(get(&v, "x5edit").map_err(|_| Error("not an x5desk project".into()))?)?;
        if ver != VERSION {
            return bad(format!("project format {ver}; this build reads {VERSION}"));
        }
        let fps = arr(get(&v, "fps")?)?;
        let [n, d] = fps else {
            return bad("fps: expected [num, den]");
        };
        let mut tl = Timeline::new(Rate {
            num: num(n)?,
            den: num(d)?,
        });
        for (i, f) in arr(get(&v, "files")?)?.iter().enumerate() {
            let e = |e: Error| Error(format!("file {i}: {}", e.0));
            let path = str(get(f, "path").map_err(e)?).map_err(e)?.to_owned();
            tl.files.push(File {
                path,
                dur: num(get(f, "dur").map_err(e)?).map_err(e)?,
            });
        }
        for (i, c) in arr(get(&v, "clips")?)?.iter().enumerate() {
            tl.clips
                .push(clip(c).map_err(|e| Error(format!("clip {i}: {}", e.0)))?);
        }
        tl.out.view = view(get(&v, "view")?).map_err(|e| Error(format!("view: {}", e.0)))?;
        tl.out.music = match v.get("music") {
            None | Some(Val::Null) => None,
            Some(m) => Some(music(m).map_err(|e| Error(format!("music: {}", e.0)))?),
        };
        tl.check()?;
        tl.fix();
        Ok(tl)
    }
}

fn clip(c: &Val) -> Result<Clip> {
    let opt = |k: &str| c.get(k).filter(|v| !matches!(v, Val::Null));
    let mut r = Clip::new(
        num(get(c, "file")?)?,
        num(get(c, "a")?)?,
        num(get(c, "b")?)?,
    );
    if let Some(v) = opt("hold") {
        r.hold = num(v)?;
    }
    if let Some(v) = opt("rev") {
        r.rev = bool(v)?;
    }
    if let Some(v) = opt("mute") {
        r.mute = bool(v)?;
    }
    if let Some(v) = opt("amt") {
        r.amt = num(v)?;
    }
    if let Some(v) = opt("look") {
        r.look = str(v)?.to_owned();
    }
    if let Some(v) = opt("adjust") {
        let Val::Obj(_) = v else {
            return bad("adjust: expected an object");
        };
        for (k, x) in adj(&mut r.adj) {
            if let Some(n) = v.get(k) {
                *x = num(n).map_err(|e| Error(format!("adjust.{k}: {}", e.0)))?;
            }
        }
    }
    if let Some(v) = opt("speed") {
        r.speed = pair(v)?;
    }
    if let Some(v) = opt("vol") {
        r.vol = pair(v)?;
    }
    if let Some(v) = opt("view") {
        r.view = view(v)?;
    }
    Ok(r)
}

fn pair<A: FromStr, B: FromStr>(v: &Val) -> Result<Vec<(A, B)>> {
    (arr(v)?.iter())
        .map(|p| match arr(p)? {
            [a, b] => Ok((num(a)?, num(b)?)),
            _ => bad("expected [time, value]"),
        })
        .collect()
}

fn view(v: &Val) -> Result<Vec<Key>> {
    (arr(v)?.iter())
        .map(|k| {
            let q = arr(get(k, "q")?)?;
            let [x, y, z, w] = q else {
                return bad("q: expected [x, y, z, w]");
            };
            let d = || num::<f32>(get(k, "d")?);
            let proj = match str(get(k, "proj")?)? {
                "persp" => Proj::Persp,
                "panini" => Proj::Panini(d()?),
                "planet" => Proj::Planet(d()?),
                "equirect" => Proj::Equirect,
                p => return bad(format!("projection {p:?}")),
            };
            let e = str(get(k, "ease")?)?;
            let Some(&(e, _)) = EASES.iter().find(|x| x.1 == e) else {
                return bad(format!("ease {e:?}"));
            };
            Ok(Key {
                t: num(get(k, "t")?)?,
                v: View {
                    q: Quat::from_xyzw(num(x)?, num(y)?, num(z)?, num(w)?),
                    fov: num(get(k, "fov")?)?,
                    proj,
                },
                e,
            })
        })
        .collect()
}

fn music(m: &Val) -> Result<Music> {
    Ok(Music {
        file: num(get(m, "file")?)?,
        offset: num(get(m, "offset")?)?,
        gain: num(get(m, "gain")?)?,
        duck: bool(get(m, "duck")?)?,
    })
}

/// A parsed JSON value; numbers stay text until the schema says which type they are.
#[derive(Debug)]
enum Val<'a> {
    Null,
    Bool(bool),
    Num(&'a str),
    Str(String),
    Arr(Vec<Val<'a>>),
    Obj(Vec<(String, Val<'a>)>),
}

impl<'a> Val<'a> {
    fn get(&self, k: &str) -> Option<&Val<'a>> {
        match self {
            Val::Obj(m) => m.iter().find(|e| e.0 == k).map(|e| &e.1),
            _ => None,
        }
    }
}

fn get<'b, 'a>(v: &'b Val<'a>, k: &str) -> Result<&'b Val<'a>> {
    v.get(k).ok_or_else(|| Error(format!("missing {k:?}")))
}

/// A number of type T; floats must be finite (JSON has no NaN or infinity, `1e999` parses to one).
fn num<T: FromStr>(v: &Val) -> Result<T> {
    match v {
        Val::Num(s) => match s.parse::<T>() {
            Ok(x) if s.parse::<f64>().is_ok_and(f64::is_finite) => Ok(x),
            _ => bad(format!("number {s}")),
        },
        _ => bad("expected a number"),
    }
}

fn bool(v: &Val) -> Result<bool> {
    match v {
        Val::Bool(b) => Ok(*b),
        _ => bad("expected true or false"),
    }
}

fn str<'b>(v: &'b Val) -> Result<&'b str> {
    match v {
        Val::Str(s) => Ok(s),
        _ => bad("expected a string"),
    }
}

fn arr<'b, 'a>(v: &'b Val<'a>) -> Result<&'b [Val<'a>]> {
    match v {
        Val::Arr(a) => Ok(a),
        _ => bad("expected a list"),
    }
}

struct P<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
    depth: u32,
}

impl<'a> P<'a> {
    fn err<T>(&self, m: &str) -> Result<T> {
        bad(format!("byte {}: {m}", self.i))
    }

    fn ws(&mut self) {
        while matches!(self.s.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        let hit = self.s.get(self.i) == Some(&c);
        self.i += hit as usize;
        hit
    }

    fn val(&mut self) -> Result<Val<'a>> {
        self.ws();
        let Some(&c) = self.s.get(self.i) else {
            return self.err("unexpected end");
        };
        match c {
            b'{' | b'[' => {
                if self.depth == DEPTH {
                    return self.err("nested too deep");
                }
                self.depth += 1;
                self.i += 1;
                let close = if c == b'{' { b'}' } else { b']' };
                let (mut o, mut a) = (vec![], vec![]);
                if !self.eat(close) {
                    loop {
                        if c == b'{' {
                            self.ws();
                            if self.s.get(self.i) != Some(&b'"') {
                                return self.err("expected a field name");
                            }
                            let k = self.str()?;
                            if !self.eat(b':') {
                                return self.err("expected ':'");
                            }
                            o.push((k, self.val()?));
                        } else {
                            a.push(self.val()?);
                        }
                        if self.eat(close) {
                            break;
                        }
                        if !self.eat(b',') {
                            return self.err("expected ',' or the end of the list");
                        }
                    }
                }
                self.depth -= 1;
                Ok(if c == b'{' { Val::Obj(o) } else { Val::Arr(a) })
            }
            b'"' => Ok(Val::Str(self.str()?)),
            b'-' | b'0'..=b'9' => {
                let j = self.i;
                while matches!(
                    self.s.get(self.i),
                    Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                ) {
                    self.i += 1;
                }
                Ok(Val::Num(&self.src[j..self.i]))
            }
            _ => {
                for (w, v) in [
                    ("true", Val::Bool(true)),
                    ("false", Val::Bool(false)),
                    ("null", Val::Null),
                ] {
                    if self.s[self.i..].starts_with(w.as_bytes()) {
                        self.i += w.len();
                        return Ok(v);
                    }
                }
                self.err("unexpected character")
            }
        }
    }

    /// A string at `"`; the input is UTF-8 and runs are cut at ASCII bytes only, so the result is too.
    fn str(&mut self) -> Result<String> {
        self.i += 1;
        let mut o = String::new();
        loop {
            let j = self.i;
            while !matches!(self.s.get(self.i), None | Some(b'"' | b'\\')) {
                self.i += 1;
            }
            o.push_str(&self.src[j..self.i]);
            match self.s.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(o);
                }
                _ => {
                    let Some(&e) = self.s.get(self.i + 1) else {
                        return self.err("unterminated string");
                    };
                    self.i += 2;
                    o.push(match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut u = self.hex()?;
                            if (0xd800..0xdc00).contains(&u) {
                                if !self.s[self.i..].starts_with(b"\\u") {
                                    return self.err("lone surrogate");
                                }
                                self.i += 2;
                                let l = self.hex()?;
                                if !(0xdc00..0xe000).contains(&l) {
                                    return self.err("bad surrogate pair");
                                }
                                u = 0x10000 + ((u - 0xd800) << 10) + (l - 0xdc00);
                            }
                            match char::from_u32(u) {
                                Some(c) => c,
                                None => return self.err("bad \\u escape"),
                            }
                        }
                        _ => return self.err("bad escape"),
                    });
                }
            }
        }
    }

    fn hex(&mut self) -> Result<u32> {
        let h = self
            .s
            .get(self.i..self.i + 4)
            .and_then(|h| std::str::from_utf8(h).ok());
        match h.and_then(|h| {
            u32::from_str_radix(h, 16)
                .ok()
                .filter(|_| h.bytes().all(|b| b.is_ascii_hexdigit()))
        }) {
            Some(u) => {
                self.i += 4;
                Ok(u)
            }
            None => self.err("bad \\u escape"),
        }
    }
}
