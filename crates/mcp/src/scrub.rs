//! Secrets never leave the process. Whatever secret a host learns (the camera's access-point password, the Wi-Fi
//! client id) goes into its [`Secrets`]; every line written to a client passes through a [`Writer`] around the
//! transport, and the log through [`Secrets::scrub`], which replace each secret, raw and JSON-escaped (once and twice:
//! a tool result's text is JSON inside a JSON string), with `[redacted]`. A client's line stays JSON
//! ([`Secrets::scrub_line`]): only strings and keys are masked, numbers stay as they are (a digit-only secret such as
//! the factory password `88888888` would otherwise break `"size_bytes":1888888880`, and the client would drop the
//! answer). Secrets shorter than [`MIN`] bytes are not scrubbed: they would mangle ordinary output. The DTOs never
//! carry them in the first place; this is the second line.

use serde_json::Value;
use std::{
    borrow::Cow,
    fmt, io,
    pin::Pin,
    sync::{Arc, RwLock},
    task::{Context, Poll, ready},
};
use tokio::io::AsyncWrite;

/// What a secret becomes.
pub const MASK: &str = "[redacted]";

/// Secrets shorter than this are left alone.
pub const MIN: usize = 4;

/// How deep JSON text inside strings is scrubbed as JSON (a tool result's text is one level).
const DEPTH: usize = 4;

/// The secrets a host knows, shared by its connections and its log.
#[derive(Clone, Default)]
pub struct Secrets(Arc<RwLock<Vec<Vec<u8>>>>);

impl fmt::Debug for Secrets {
    /// How many byte patterns it masks, never the secrets.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = self.0.read().map_or(0, |v| v.len());
        write!(f, "Secrets({n} patterns)")
    }
}

/// `s` as it reads inside a JSON string.
fn esc(s: &str) -> String {
    // a string always serialises, quoted
    let j = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
    j[1..j.len() - 1].to_string()
}

impl Secrets {
    /// Adds `s` (and its escaped forms); one under [`MIN`] bytes is ignored.
    pub fn add(&self, s: &str) {
        if s.len() < MIN {
            return;
        }
        let one = esc(s);
        let two = esc(&one);
        let mut v = self.0.write().unwrap();
        for p in [s.to_string(), one, two] {
            if !v.iter().any(|x| *x == p.as_bytes()) {
                v.push(p.into_bytes())
            }
        }
        // longest first: a secret inside another is masked as part of the longer one
        v.sort_by_key(|p| std::cmp::Reverse(p.len()));
    }

    /// No secret known yet.
    pub fn is_empty(&self) -> bool {
        self.0.read().unwrap().is_empty()
    }

    /// `b` with every secret replaced by [`MASK`].
    pub fn scrub_bytes<'a>(&self, b: &'a [u8]) -> Cow<'a, [u8]> {
        let v = self.0.read().unwrap();
        let mut out: Cow<[u8]> = Cow::Borrowed(b);
        for p in v.iter() {
            if let Some(r) = replace(&out, p) {
                out = Cow::Owned(r)
            }
        }
        out
    }

    /// `s` with every secret replaced by [`MASK`].
    pub fn scrub<'a>(&self, s: &'a str) -> Cow<'a, str> {
        match self.scrub_bytes(s.as_bytes()) {
            Cow::Borrowed(_) => Cow::Borrowed(s),
            // a secret is valid UTF-8 and so is the mask: whole characters are swapped
            Cow::Owned(b) => Cow::Owned(String::from_utf8_lossy(&b).into_owned()),
        }
    }

    /// One line written to a client (its `\n` or `\r\n` kept), still JSON when it was: the secrets in its strings
    /// and keys masked (JSON text inside a string scrubbed as JSON again, [`DEPTH`] levels deep), numbers left as
    /// they are, written compact. A line without a secret stays as it is; one that does not parse is masked as
    /// bytes ([`Secrets::scrub_bytes`]).
    pub fn scrub_line<'a>(&self, l: &'a [u8]) -> Cow<'a, [u8]> {
        let b = self.scrub_bytes(l);
        if matches!(b, Cow::Borrowed(_)) {
            return b;
        }
        let end = l.len()
            - l.iter()
                .rev()
                .take_while(|&&c| c == b'\n' || c == b'\r')
                .count();
        let Ok(mut v) = serde_json::from_slice::<Value>(&l[..end]) else {
            return b;
        };
        if !self.walk(&mut v, DEPTH) {
            // the secret is only in numbers (or across tokens): the line as it was
            return Cow::Borrowed(l);
        }
        match serde_json::to_vec(&v) {
            Ok(mut o) => {
                o.extend_from_slice(&l[end..]);
                Cow::Owned(o)
            }
            Err(_) => b,
        }
    }

    /// Masks the secrets in `v`'s strings and keys; whether it changed anything.
    fn walk(&self, v: &mut Value, depth: usize) -> bool {
        match v {
            Value::String(s) => match self.text(s, depth) {
                Some(t) => {
                    *s = t;
                    true
                }
                None => false,
            },
            Value::Array(a) => a.iter_mut().fold(false, |c, x| self.walk(x, depth) | c),
            Value::Object(m) => {
                let c = m.values_mut().fold(false, |c, x| self.walk(x, depth) | c);
                if !m.keys().any(|k| matches!(self.scrub(k), Cow::Owned(_))) {
                    return c;
                }
                *m = (std::mem::take(m).into_iter())
                    .map(|(k, x)| (self.scrub(&k).into_owned(), x))
                    .collect();
                true
            }
            _ => false,
        }
    }

    /// A string's text scrubbed: JSON text (an object or an array, a tool result's `text`) as JSON, so it stays
    /// that; None when it holds no secret.
    fn text(&self, s: &str, depth: usize) -> Option<String> {
        let Cow::Owned(t) = self.scrub(s) else {
            return None;
        };
        if depth > 0
            && s.trim_start().starts_with(['{', '['])
            && let Ok(mut v) = serde_json::from_str::<Value>(s)
        {
            // its numbers stay too
            return match self.walk(&mut v, depth - 1) {
                true => Some(serde_json::to_string(&v).unwrap_or(t)),
                false => None,
            };
        }
        Some(t)
    }
}

/// `b` with every `p` replaced by [`MASK`], None when it holds none.
fn replace(b: &[u8], p: &[u8]) -> Option<Vec<u8>> {
    let mut i = b.windows(p.len()).position(|w| w == p)?;
    let mut o = Vec::with_capacity(b.len());
    let mut from = 0;
    loop {
        o.extend_from_slice(&b[from..i]);
        o.extend_from_slice(MASK.as_bytes());
        from = i + p.len();
        match b[from..].windows(p.len()).position(|w| w == p) {
            Some(k) => i = from + k,
            None => break,
        }
    }
    o.extend_from_slice(&b[from..]);
    Some(o)
}

/// A scrubbing layer around any writer: bytes are held until their line ends (`\n`), then written with the secrets
/// masked. A line without its end is written at shutdown. Wrap a transport's write half in it (the host does for
/// every connection it serves).
pub struct Writer<W> {
    w: W,
    s: Secrets,
    /// the line still open
    line: Vec<u8>,
    /// scrubbed bytes not written yet, from `at`
    out: Vec<u8>,
    at: usize,
}

/// Scrubbed bytes held before a write waits for the writer below.
const HOLD: usize = 1 << 20;

impl<W: AsyncWrite + Unpin> Writer<W> {
    /// `w` scrubbed of `s` (secrets added later count from then on).
    pub fn new(w: W, s: Secrets) -> Writer<W> {
        Writer {
            w,
            s,
            line: vec![],
            out: vec![],
            at: 0,
        }
    }

    /// The writer below; bytes still held are dropped.
    pub fn into_inner(self) -> W {
        self.w
    }

    /// Writes what is held; Ready once it is all out.
    fn drain(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        while self.at < self.out.len() {
            match ready!(Pin::new(&mut self.w).poll_write(cx, &self.out[self.at..]))? {
                0 => return Poll::Ready(Err(io::ErrorKind::WriteZero.into())),
                n => self.at += n,
            }
        }
        (self.out, self.at) = (vec![], 0);
        Poll::Ready(Ok(()))
    }

    /// Moves the whole lines of `line` (all of it when `all`) to `out`, each scrubbed as JSON
    /// ([`Secrets::scrub_line`]).
    fn cut(&mut self, all: bool) {
        let n = match all {
            true => self.line.len(),
            false => self
                .line
                .iter()
                .rposition(|&b| b == b'\n')
                .map_or(0, |i| i + 1),
        };
        if n == 0 {
            return;
        }
        let rest = self.line.split_off(n);
        let l = std::mem::replace(&mut self.line, rest);
        for x in l.split_inclusive(|&b| b == b'\n') {
            self.out.extend_from_slice(&self.s.scrub_line(x))
        }
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for Writer<W> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let me = self.get_mut();
        if let Poll::Ready(Err(e)) = me.drain(cx) {
            return Poll::Ready(Err(e));
        }
        if me.out.len() - me.at >= HOLD {
            // the writer below is behind: wait for it (drain registered the waker)
            return Poll::Pending;
        }
        me.line.extend_from_slice(buf);
        me.cut(false);
        Poll::Ready(Ok(buf.len()))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        ready!(me.drain(cx))?;
        Pin::new(&mut me.w).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let me = self.get_mut();
        me.cut(true);
        ready!(me.drain(cx))?;
        Pin::new(&mut me.w).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn scrub() {
        let s = Secrets::default();
        s.add("abc");
        assert!(s.is_empty(), "under MIN: not scrubbed");
        s.add("pa\"ss/w\\rd");
        s.add("88888888");
        assert_eq!(s.scrub("no secret here"), "no secret here");
        assert!(matches!(s.scrub("plain"), Cow::Borrowed(_)));
        assert_eq!(
            s.scrub("pw 88888888, again 88888888."),
            "pw [redacted], again [redacted]."
        );
        // text (the log) and a client's line that is not JSON are masked as bytes; a JSON number stays
        assert_eq!(s.scrub("888888888"), "[redacted]8");
        assert_eq!(&*s.scrub_line(b"pw 888888888\n"), b"pw [redacted]8\n");
        assert!(matches!(s.scrub_line(b"888888888\n"), Cow::Borrowed(_)));
        // its debug form counts the patterns (three forms of the first, the digits' one), never shows them
        assert_eq!(format!("{s:?}"), "Secrets(4 patterns)");
        // raw, JSON-escaped, and escaped twice (JSON text inside a JSON string)
        let raw = "pa\"ss/w\\rd";
        let one = serde_json::to_string(&serde_json::json!({ "pw": raw })).unwrap();
        let two = serde_json::to_string(&serde_json::json!({ "text": one })).unwrap();
        for t in [raw.to_string(), one.clone(), two.clone()] {
            let o = s.scrub(&t);
            assert!(!o.contains("ss/w"), "{t} → {o}");
            assert!(o.contains(MASK), "{o}");
        }
        assert_eq!(s.scrub(&one), r#"{"pw":"[redacted]"}"#);
        // a secret inside a longer one goes with it
        s.add("8888888899");
        assert_eq!(s.scrub("x8888888899x"), "x[redacted]x");
        // non-ASCII stays whole
        s.add("pässwörd");
        assert_eq!(s.scrub("ä pässwörd ö"), "ä [redacted] ö");
    }

    #[tokio::test]
    async fn writer() {
        let s = Secrets::default();
        s.add("s3cr3t-pw");
        let (a, mut b) = tokio::io::duplex(64);
        let mut w = Writer::new(a, s.clone());
        let t = tokio::spawn(async move {
            let mut v = vec![];
            b.read_to_end(&mut v).await.unwrap();
            v
        });
        // a secret split across writes, lines bigger than the pipe, a line left open at the end
        w.write_all(b"{\"a\":\"s3cr").await.unwrap();
        w.write_all(b"3t-pw\"}\n").await.unwrap();
        let big = format!("{}s3cr3t-pw\n", "x".repeat(1000));
        w.write_all(big.as_bytes()).await.unwrap();
        w.flush().await.unwrap();
        w.write_all(b"tail s3cr3t-pw").await.unwrap();
        w.shutdown().await.unwrap();
        drop(w);
        let v = String::from_utf8(t.await.unwrap()).unwrap();
        assert!(!v.contains("s3cr3t"), "{v}");
        let want = format!(
            "{{\"a\":\"[redacted]\"}}\n{}[redacted]\ntail [redacted]",
            "x".repeat(1000)
        );
        assert_eq!(v, want);
    }

    /// A digit-only secret (the factory access-point password) in a JSON-RPC line: the line stays JSON, its numbers
    /// as they were, the secret masked in every string and key, the result's text (JSON in a string) JSON too.
    #[tokio::test]
    async fn json() {
        let s = Secrets::default();
        s.add("88888888");
        let sc = json!({
            "size_bytes": 1888888880u64,
            "progress": 88888888.5,
            "pw": "88888888",
            "88888888": "a key",
            "list": ["x88888888y", 1888888880u64],
        });
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 88888888,
            "result": { "content": [{ "type": "text", "text": sc.to_string() }], "structuredContent": sc },
        });
        let mut w = Writer::new(vec![], s.clone());
        w.write_all(format!("{msg}\n").as_bytes()).await.unwrap();
        // no secret, or one in numbers only: the line as it was (its keys' order too)
        w.write_all(b"{\"z\":1,\"a\":\"x\"}\r\n{\"z\":1888888880,\"a\":\"x\"}\n")
            .await
            .unwrap();
        // not JSON: as bytes
        w.write_all(b"pw 88888888 in 1888888880\n").await.unwrap();
        w.shutdown().await.unwrap();
        let out = String::from_utf8(w.into_inner()).unwrap();
        let l: Vec<&str> = out.split_inclusive('\n').collect();
        assert_eq!(l.len(), 4, "{out}");
        let v: Value = serde_json::from_str(l[0]).unwrap_or_else(|e| panic!("{e}: {}", l[0]));
        assert!(
            l[0].ends_with("}\n") && !l[0].contains("\"88888888"),
            "{}",
            l[0]
        );
        let want = json!({
            "size_bytes": 1888888880u64,
            "progress": 88888888.5,
            "pw": MASK,
            MASK: "a key",
            "list": [format!("x{MASK}y"), 1888888880u64],
        });
        assert_eq!(v["result"]["structuredContent"], want);
        assert_eq!((&v["id"], &v["jsonrpc"]), (&json!(88888888), &json!("2.0")));
        let text = v["result"]["content"][0]["text"].as_str().unwrap();
        assert_eq!(serde_json::from_str::<Value>(text).unwrap(), want, "{text}");
        assert_eq!(l[1], "{\"z\":1,\"a\":\"x\"}\r\n");
        assert_eq!(l[2], "{\"z\":1888888880,\"a\":\"x\"}\n");
        assert_eq!(l[3], "pw [redacted] in 1[redacted]0\n");
    }
}
