//! AMF0, the subset that RTMP commands and `onMetaData` use.

pub fn num(o: &mut Vec<u8>, v: f64) {
    o.push(0);
    o.extend_from_slice(&v.to_be_bytes());
}

pub fn bool(o: &mut Vec<u8>, v: bool) {
    o.extend_from_slice(&[1, v as u8]);
}

pub fn str(o: &mut Vec<u8>, s: &str) {
    o.push(2);
    key(o, s);
}

pub fn null(o: &mut Vec<u8>) {
    o.push(5);
}

/// A property name inside an object or ECMA array: a string without the type marker.
pub fn key(o: &mut Vec<u8>, s: &str) {
    o.extend_from_slice(&(s.len() as u16).to_be_bytes());
    o.extend_from_slice(s.as_bytes());
}

pub fn obj(o: &mut Vec<u8>) {
    o.push(3);
}

pub fn ecma(o: &mut Vec<u8>, n: u32) {
    o.push(8);
    o.extend_from_slice(&n.to_be_bytes());
}

/// Ends an object or ECMA array.
pub fn end(o: &mut Vec<u8>) {
    o.extend_from_slice(&[0, 0, 9]);
}

#[derive(Clone, Debug, PartialEq)]
pub enum V {
    Num(f64),
    Bool(bool),
    Str(String),
    /// Object or ECMA array.
    Obj(Vec<(String, V)>),
    /// Null, undefined, and the types RTMP replies never need (dates, strict arrays).
    Null,
}

impl V {
    pub fn get(&self, k: &str) -> Option<&V> {
        match self {
            V::Obj(p) => p.iter().find(|p| p.0 == k).map(|p| &p.1),
            _ => None,
        }
    }

    pub fn str(&self) -> &str {
        match self {
            V::Str(s) => s,
            _ => "",
        }
    }

    pub fn num(&self) -> Option<f64> {
        match self {
            V::Num(n) => Some(*n),
            _ => None,
        }
    }
}

/// Every value in `b`, up to the first one that doesn't parse.
pub fn parse(mut b: &[u8]) -> Vec<V> {
    let mut v = vec![];
    while let Some(x) = val(&mut b) {
        v.push(x);
    }
    v
}

fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    let (h, t) = b.split_at_checked(n)?;
    *b = t;
    Some(h)
}

fn u32(b: &mut &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(take(b, 4)?.try_into().ok()?))
}

fn string(b: &mut &[u8], n: usize) -> Option<String> {
    Some(String::from_utf8_lossy(take(b, n)?).into_owned())
}

fn s16(b: &mut &[u8]) -> Option<String> {
    let n = u16::from_be_bytes(take(b, 2)?.try_into().ok()?);
    string(b, n as usize)
}

fn val(b: &mut &[u8]) -> Option<V> {
    let t = take(b, 1)?[0];
    Some(match t {
        0 => V::Num(f64::from_be_bytes(take(b, 8)?.try_into().ok()?)),
        1 => V::Bool(take(b, 1)?[0] != 0),
        2 => V::Str(s16(b)?),
        12 => {
            let n = u32(b)?;
            V::Str(string(b, n as usize)?)
        }
        3 | 8 => {
            if t == 8 {
                u32(b)?;
            }
            let mut p = vec![];
            loop {
                let k = s16(b)?;
                if k.is_empty() && b.first() == Some(&9) {
                    *b = &b[1..];
                    break V::Obj(p);
                }
                p.push((k, val(b)?));
            }
        }
        10 => {
            for _ in 0..u32(b)? {
                val(b)?;
            }
            V::Null
        }
        11 => {
            take(b, 10)?;
            V::Null
        }
        5 | 6 => V::Null,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut o = vec![];
        str(&mut o, "_result");
        num(&mut o, 4.0);
        null(&mut o);
        obj(&mut o);
        key(&mut o, "code");
        str(&mut o, "NetStream.Publish.Start");
        key(&mut o, "ok");
        bool(&mut o, true);
        end(&mut o);
        ecma(&mut o, 1);
        key(&mut o, "w");
        num(&mut o, 1664.0);
        end(&mut o);
        let v = parse(&o);
        assert_eq!(v.len(), 5);
        assert_eq!(v[0].str(), "_result");
        assert_eq!(v[1].num(), Some(4.0));
        assert_eq!(v[2], V::Null);
        assert_eq!(v[3].get("code").unwrap().str(), "NetStream.Publish.Start");
        assert_eq!(v[3].get("ok"), Some(&V::Bool(true)));
        assert_eq!(v[4].get("w").and_then(V::num), Some(1664.0));
    }
}
