//! RTMP publisher: simple handshake, chunk stream, AMF0 commands, reconnects. A thread per
//! publisher connects and writes; while live, a second one reads the server's control messages.
//! Frames wait in a bounded queue that drops non-key video when the network falls behind.
//! Timestamps are ms on one clock for audio and video, and never decrease within a stream.

use crate::{amf, flv};
use std::{
    collections::{HashMap, VecDeque},
    io::{self, Read, Write},
    net::{Shutdown, TcpStream, ToSocketAddrs},
    sync::{Arc, Condvar, Mutex},
    thread,
    time::{Duration, SystemTime},
};

const CHUNK: usize = 4096;
const T: Duration = Duration::from_secs(10);
/// How long dropping a live publisher may take to send what's queued and unpublish.
pub const STOP: Duration = Duration::from_secs(3);

pub struct Cfg {
    pub meta: flv::Meta,
    /// AVCDecoderConfigurationRecord ([`flv::avcc`]); empty for no video.
    pub avcc: Vec<u8>,
    /// AudioSpecificConfig; empty for no audio.
    pub asc: Vec<u8>,
    /// Queued media in ms beyond which non-key video is dropped; audio goes at twice that.
    pub lag: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum State {
    #[default]
    Connecting,
    Live,
    Retrying,
    Stopped,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub state: State,
    /// Frames written to the socket, [video, audio].
    pub sent: [u64; 2],
    /// Frames dropped, [video, audio].
    pub drop: [u64; 2],
    pub bytes: u64,
    pub reconnects: u32,
    pub err: Option<String>,
}

struct Msg {
    ty: u8,
    ts: u32,
    key: bool,
    b: Vec<u8>,
}

/// Queue state; protocol control replies (types below [`flv::AUDIO`]) sit in front of media.
#[derive(Default)]
struct S {
    q: VecDeque<Msg>,
    pool: Vec<Vec<u8>>,
    lag: i64,
    /// Dropping video until the next key frame.
    skip: bool,
    stop: bool,
    /// The reader saw the connection end.
    dead: bool,
    /// Lets `drop` abort a session; `None` while resolving and connecting.
    sock: Option<TcpStream>,
    st: Stats,
}

impl S {
    fn recycle(&mut self, mut b: Vec<u8>) {
        if self.pool.len() < 16 {
            b.clear();
            self.pool.push(b);
        }
    }

    fn lose(&mut self, m: Msg) {
        self.st.drop[(m.ty == flv::AUDIO) as usize] += 1;
        self.recycle(m.b);
    }

    fn push(&mut self, m: Msg) {
        let d = self
            .q
            .iter()
            .find(|x| x.ty >= flv::AUDIO)
            .map_or(0, |x| m.ts.wrapping_sub(x.ts) as i32 as i64);
        let v = m.ty == flv::VIDEO;
        let max = if v { self.lag } else { 2 * self.lag };
        if v && m.key {
            self.skip = false;
            if d > max {
                // the stream jumps to this key frame: what's queued from before it goes
                for x in std::mem::take(&mut self.q) {
                    if x.ty == flv::VIDEO
                        || x.ty == flv::AUDIO && (x.ts.wrapping_sub(m.ts) as i32) < 0
                    {
                        self.lose(x)
                    } else {
                        self.q.push_back(x)
                    }
                }
            }
        } else if v && self.skip || d > max {
            self.skip |= v;
            return self.lose(m);
        }
        self.q.push_back(m);
    }

    /// A new session: drops old control replies, and video before the first key frame. Audio
    /// stays; [`pump`] drops what's older than the session's start.
    fn trim(&mut self, video: bool) {
        let mut k = !video;
        for m in std::mem::take(&mut self.q) {
            k |= m.key;
            match m.ty {
                flv::AUDIO => self.q.push_back(m),
                flv::VIDEO if k => self.q.push_back(m),
                flv::VIDEO => self.lose(m),
                _ => self.recycle(m.b),
            }
        }
        self.skip = !k;
    }
}

struct Q {
    m: Mutex<S>,
    cv: Condvar,
}

struct Url {
    host: String,
    port: u16,
    app: String,
    tc: String,
    key: String,
}

pub struct Rtmp {
    q: Arc<Q>,
    th: Option<thread::JoinHandle<()>>,
}

impl Rtmp {
    /// Publishes to `url` (`rtmp://host[:port]/app`) as stream `key`; an empty key takes the last
    /// path segment of `url`. Connects, and reconnects with backoff, on its own thread.
    pub fn start(url: &str, key: &str, cfg: Cfg) -> io::Result<Self> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidInput, format!("{m}: {url}"));
        let (tc, key) = if key.is_empty() {
            url.rsplit_once('/').ok_or_else(|| bad("no key"))?
        } else {
            (url.trim_end_matches('/'), key)
        };
        let r = tc
            .strip_prefix("rtmp://")
            .ok_or_else(|| bad("only rtmp:// is supported"))?;
        let (h, app) = r.split_once('/').ok_or_else(|| bad("no app"))?;
        let (host, port) = match h.rsplit_once(':') {
            Some((h, p)) if !p.ends_with(']') => (h, p.parse().map_err(|_| bad("bad port"))?),
            _ => (h, 1935),
        };
        let u = Url {
            host: host.trim_start_matches('[').trim_end_matches(']').into(),
            port,
            app: app.into(),
            tc: tc.into(),
            key: key.into(),
        };
        let q = Arc::new(Q {
            m: Mutex::new(S {
                lag: cfg.lag as i64,
                ..Default::default()
            }),
            cv: Condvar::new(),
        });
        let q2 = q.clone();
        let th = thread::Builder::new()
            .name("rtmp".into())
            .spawn(move || run(&q2, &u, &cfg))?;
        Ok(Self { q, th: Some(th) })
    }

    /// One access unit; `cts` = pts − dts in ms. `nals` without start codes ([`flv::nals`]).
    pub fn video<'a>(
        &self,
        dts: u32,
        cts: i32,
        key: bool,
        nals: impl IntoIterator<Item = &'a [u8]>,
    ) {
        let mut b = self.buf();
        flv::avc(&mut b, key, cts, nals);
        self.push(Msg {
            ty: flv::VIDEO,
            ts: dts,
            key,
            b,
        });
    }

    /// One raw AAC frame.
    pub fn audio(&self, ts: u32, raw: &[u8]) {
        let mut b = self.buf();
        flv::aac(&mut b, raw);
        self.push(Msg {
            ty: flv::AUDIO,
            ts,
            key: false,
            b,
        });
    }

    pub fn stats(&self) -> Stats {
        self.q.m.lock().unwrap().st.clone()
    }

    fn buf(&self) -> Vec<u8> {
        self.q.m.lock().unwrap().pool.pop().unwrap_or_default()
    }

    fn push(&self, m: Msg) {
        self.q.m.lock().unwrap().push(m);
        self.q.cv.notify_one();
    }
}

impl Drop for Rtmp {
    /// A live session sends what's queued, then unpublishes, within [`STOP`]; any other state
    /// gives up at once. A pending DNS lookup or connect is left to finish on its own: the thread
    /// then sees the stop and ends without publishing.
    fn drop(&mut self) {
        let mut s = self.q.m.lock().unwrap();
        s.stop = true;
        self.q.cv.notify_all();
        if s.st.state == State::Live {
            s = self
                .q
                .cv
                .wait_timeout_while(s, STOP, |s| s.st.state != State::Stopped)
                .unwrap()
                .0;
        }
        if let Some(t) = &s.sock {
            let _ = t.shutdown(Shutdown::Both);
        }
        let join = s.sock.is_some() || s.st.state != State::Connecting;
        drop(s);
        if let Some(t) = self.th.take().filter(|_| join) {
            let _ = t.join();
        }
    }
}

fn run(q: &Arc<Q>, u: &Url, cfg: &Cfg) {
    let mut wait = 1;
    let mut s = loop {
        let r = session(q, u, cfg);
        let mut s = q.m.lock().unwrap();
        s.sock = None;
        if r.is_ok() || s.stop {
            break s;
        }
        if s.st.state == State::Live {
            wait = 1;
        }
        s.st.err = r.err().map(|e| e.to_string());
        s.st.state = State::Retrying;
        s.st.reconnects += 1;
        let mut s =
            q.cv.wait_timeout_while(s, Duration::from_secs(wait), |s| !s.stop)
                .unwrap()
                .0;
        if s.stop {
            break s;
        }
        s.st.state = State::Connecting;
        wait = (wait * 2).min(16);
    };
    s.st.state = State::Stopped;
    drop(s);
    q.cv.notify_all();
}

fn session(q: &Arc<Q>, u: &Url, cfg: &Cfg) -> io::Result<()> {
    let mut r = Err(io::ErrorKind::NotFound.into());
    for a in (u.host.as_str(), u.port).to_socket_addrs()? {
        r = TcpStream::connect_timeout(&a, T);
        if r.is_ok() {
            break;
        }
    }
    let mut s = r?;
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(T))?;
    s.set_write_timeout(Some(T))?;
    {
        let mut g = q.m.lock().unwrap();
        if g.stop {
            return Ok(());
        }
        g.sock = Some(s.try_clone()?);
    }

    // Simple handshake: C1 = time 0, zero, random; C2 echoes S1.
    let mut h = [0u8; 1537];
    h[0] = 3;
    let mut x = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64 | 1);
    for c in h[9..].chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        c.copy_from_slice(&x.to_le_bytes()[..c.len()]);
    }
    s.write_all(&h)?;
    s.read_exact(&mut h)?;
    if h[0] != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("RTMP version {}", h[0]),
        ));
    }
    s.write_all(&h[1..])?;
    s.read_exact(&mut h[1..])?;

    let mut rd = Rd {
        s: s.try_clone()?,
        chunk: 128,
        cs: HashMap::new(),
        n: 0,
        win: 0,
        acked: 0,
        out: 0,
    };
    let (mut o, mut b) = (vec![], vec![]);
    put(&mut o, 2, 1, 0, 0, &(CHUNK as u32).to_be_bytes());
    amf::str(&mut b, "connect");
    amf::num(&mut b, 1.0);
    amf::obj(&mut b);
    for (k, v) in [
        ("app", &*u.app),
        ("type", "nonprivate"),
        ("flashVer", "FMLE/3.0 (compatible; x5desk)"),
        ("tcUrl", &*u.tc),
    ] {
        amf::key(&mut b, k);
        amf::str(&mut b, v);
    }
    amf::end(&mut b);
    put(&mut o, 3, 20, 0, 0, &b);
    s.write_all(&o)?;
    o.clear();
    reply(&mut rd, &mut s, &mut o, 1.0)?;
    let video = !cfg.avcc.is_empty();

    for (tx, c) in [
        (2.0, "releaseStream"),
        (3.0, "FCPublish"),
        (4.0, "createStream"),
    ] {
        b.clear();
        amf::str(&mut b, c);
        amf::num(&mut b, tx);
        amf::null(&mut b);
        if tx < 4.0 {
            amf::str(&mut b, &u.key);
        }
        put(&mut o, 3, 20, 0, 0, &b);
    }
    s.write_all(&o)?;
    o.clear();
    let sid = reply(&mut rd, &mut s, &mut o, 4.0)?.num().unwrap_or(1.0) as u32;
    b.clear();
    amf::str(&mut b, "publish");
    amf::num(&mut b, 5.0);
    amf::null(&mut b);
    amf::str(&mut b, &u.key);
    amf::str(&mut b, "live");
    put(&mut o, 3, 20, sid, 0, &b);
    s.write_all(&o)?;
    o.clear();
    reply(&mut rd, &mut s, &mut o, 0.0)?;

    b.clear();
    amf::str(&mut b, "@setDataFrame");
    flv::meta(&mut b, &cfg.meta);
    put(&mut o, 5, flv::DATA, sid, 0, &b);
    if video {
        b.clear();
        flv::avc_seq(&mut b, &cfg.avcc);
        put(&mut o, 6, flv::VIDEO, sid, 0, &b);
    }
    if !cfg.asc.is_empty() {
        b.clear();
        flv::aac_seq(&mut b, &cfg.asc);
        put(&mut o, 4, flv::AUDIO, sid, 0, &b);
    }
    s.write_all(&o)?;
    o.clear();
    s.set_read_timeout(None)?;
    {
        let mut g = q.m.lock().unwrap();
        g.dead = false;
        g.st.state = State::Live;
        g.st.err = None;
        g.trim(video);
    }

    let q2 = q.clone();
    let rt = thread::Builder::new()
        .name("rtmp-rd".into())
        .spawn(move || {
            let mut r = vec![];
            while let Ok((ty, b)) = rd.msg() {
                rd.ctl(ty, &b, |ty, b| {
                    r.push(Msg {
                        ty,
                        ts: 0,
                        key: false,
                        b: b.to_vec(),
                    })
                });
                if !r.is_empty() {
                    let mut g = q2.m.lock().unwrap();
                    r.drain(..).for_each(|m| g.q.push_front(m));
                    drop(g);
                    // `drop` may wait on the same condvar
                    q2.cv.notify_all();
                }
            }
            q2.m.lock().unwrap().dead = true;
            q2.cv.notify_all();
        })?;
    let r = pump(q, &mut s, u, sid, video, &mut o, &mut b);
    if r.is_ok() {
        // FIN after the queued data, then let the reader drain until the server closes, so close()
        // never finds unread data and resets the connection with our tail unsent.
        let _ = s.set_read_timeout(Some(T));
        let _ = s.shutdown(Shutdown::Write);
    } else {
        let _ = s.shutdown(Shutdown::Both);
    }
    let _ = rt.join();
    r
}

/// Writes queued messages until stopped and drained, then unpublishes. A session's timestamps
/// start at its first frame; with `video`, its first key frame: until one is queued, media waits.
fn pump(
    q: &Q,
    s: &mut impl Write,
    u: &Url,
    sid: u32,
    video: bool,
    o: &mut Vec<u8>,
    b: &mut Vec<u8>,
) -> io::Result<()> {
    let mut base = None;
    let mut done: Option<(Msg, usize)> = None;
    loop {
        let m = {
            let mut g = q.m.lock().unwrap();
            if let Some((m, n)) = done.take() {
                if m.ty >= flv::AUDIO {
                    g.st.sent[(m.ty == flv::AUDIO) as usize] += 1;
                }
                g.st.bytes += n as u64;
                g.recycle(m.b);
            }
            loop {
                if video && base.is_none() {
                    base = g.q.iter().find(|m| m.key).map(|m| m.ts);
                }
                // control replies are always in front of media
                if g.q
                    .front()
                    .is_some_and(|m| m.ty < flv::AUDIO || base.is_some() || !video)
                {
                    break g.q.pop_front();
                }
                if g.dead {
                    return Err(io::Error::new(
                        io::ErrorKind::ConnectionAborted,
                        "server closed the connection",
                    ));
                }
                if g.stop {
                    // no key frame came, so none of it can go out
                    for m in std::mem::take(&mut g.q) {
                        g.lose(m)
                    }
                    break None;
                }
                g = q.cv.wait(g).unwrap();
            }
        };
        let Some(m) = m else { break };
        // what came after the session's start but belongs before it goes
        let ts = (m.ty >= flv::AUDIO).then(|| m.ts.wrapping_sub(*base.get_or_insert(m.ts)) as i32);
        if ts.is_some_and(|t| t < 0) {
            q.m.lock().unwrap().lose(m);
            continue;
        }
        let (cs, sid) = match m.ty {
            flv::AUDIO => (4, sid),
            flv::VIDEO => (6, sid),
            _ => (2, 0),
        };
        put(o, cs, m.ty, sid, ts.unwrap_or(0) as u32, &m.b);
        let (r, n) = (s.write_all(o), o.len());
        o.clear();
        if r.is_err() && m.ty >= flv::AUDIO {
            q.m.lock().unwrap().lose(m);
            return r;
        }
        r?;
        done = Some((m, n));
    }
    for (tx, c) in [(6.0, "FCUnpublish"), (7.0, "deleteStream")] {
        b.clear();
        amf::str(b, c);
        amf::num(b, tx);
        amf::null(b);
        if tx < 7.0 {
            amf::str(b, &u.key)
        } else {
            amf::num(b, sid as f64)
        }
        put(o, 3, 20, 0, 0, b);
    }
    s.write_all(o)
}

/// Reads until the `_result` of transaction `tx`, or with `tx` 0 the `onStatus` that starts
/// publishing, answering control messages meanwhile.
fn reply(rd: &mut Rd, s: &mut TcpStream, o: &mut Vec<u8>, tx: f64) -> io::Result<amf::V> {
    loop {
        let (ty, b) = rd.msg()?;
        rd.ctl(ty, &b, |ty, b| put(o, 2, ty, 0, 0, b));
        if !o.is_empty() {
            s.write_all(o)?;
            o.clear();
        }
        let v = match ty {
            20 => amf::parse(&b),
            // AMF3 command: a format byte, then AMF0
            17 => amf::parse(b.get(1..).unwrap_or_default()),
            _ => continue,
        };
        let id = v.get(1).and_then(amf::V::num);
        let i = v.get(3).unwrap_or(&amf::V::Null);
        let (code, lvl) = (
            i.get("code").map_or("", amf::V::str),
            i.get("level").map_or("", amf::V::str),
        );
        let e = || {
            io::Error::other(format!(
                "{code}: {}",
                i.get("description").map_or("", amf::V::str)
            ))
        };
        match v.first().map_or("", amf::V::str) {
            "_result" if id == Some(tx) => return Ok(i.clone()),
            "_error" if id == Some(tx) => return Err(e()),
            "onStatus" if tx == 0.0 && code == "NetStream.Publish.Start" => {
                return Ok(amf::V::Null);
            }
            "onStatus" if tx == 0.0 && lvl == "error" => return Err(e()),
            _ => {}
        }
    }
}

/// One message as chunks: a type-0 header with the absolute timestamp, then type-3 continuations,
/// which repeat an extended timestamp as FFmpeg and librtmp expect.
fn put(o: &mut Vec<u8>, cs: u8, ty: u8, sid: u32, ts: u32, b: &[u8]) {
    let ext = ts >= 0xFF_FFFF;
    o.push(cs);
    o.extend_from_slice(&ts.min(0xFF_FFFF).to_be_bytes()[1..]);
    o.extend_from_slice(&(b.len() as u32).to_be_bytes()[1..]);
    o.push(ty);
    o.extend_from_slice(&sid.to_le_bytes());
    for (i, c) in b.chunks(CHUNK).enumerate() {
        if i > 0 {
            o.push(0xC0 | cs);
        }
        if ext {
            o.extend_from_slice(&ts.to_be_bytes());
        }
        o.extend_from_slice(c);
    }
}

/// Incoming chunk stream.
struct Rd {
    s: TcpStream,
    chunk: usize,
    cs: HashMap<u32, Cs>,
    /// Bytes read since the handshake.
    n: u64,
    /// The server's window acknowledgement size.
    win: u64,
    acked: u64,
    /// Our window acknowledgement size, as last sent.
    out: u32,
}

#[derive(Default)]
struct Cs {
    len: usize,
    ty: u8,
    ext: bool,
    b: Vec<u8>,
}

impl Rd {
    fn get(&mut self, b: &mut [u8]) -> io::Result<()> {
        self.s.read_exact(b)?;
        self.n += b.len() as u64;
        Ok(())
    }

    /// The next complete message, (type, payload). Applies Set Chunk Size and Abort itself.
    fn msg(&mut self) -> io::Result<(u8, Vec<u8>)> {
        loop {
            let mut h = [0; 11];
            self.get(&mut h[..1])?;
            let fmt = h[0] >> 6;
            let id = match h[0] & 63 {
                0 => {
                    self.get(&mut h[..1])?;
                    64 + h[0] as u32
                }
                1 => {
                    self.get(&mut h[..2])?;
                    64 + h[0] as u32 + 256 * h[1] as u32
                }
                x => x as u32,
            };
            self.get(&mut h[..[11, 7, 3, 0][fmt as usize]])?;
            let mut c = self.cs.remove(&id).unwrap_or_default();
            if fmt < 3 {
                c.ext = h[..3] == [0xFF; 3];
            }
            if fmt < 2 {
                c.len = u32::from_be_bytes([0, h[3], h[4], h[5]]) as usize;
                c.ty = h[6];
            }
            if c.ext {
                self.get(&mut [0; 4])?;
            }
            let at = c.b.len();
            c.b.resize(at + c.len.saturating_sub(at).min(self.chunk), 0);
            self.get(&mut c.b[at..])?;
            if at < c.b.len() && c.b.len() < c.len {
                self.cs.insert(id, c);
                continue;
            }
            let b = std::mem::take(&mut c.b);
            let ty = c.ty;
            self.cs.insert(id, c);
            let n = b
                .get(..4)
                .map_or(0, |x| u32::from_be_bytes(x.try_into().unwrap()));
            match ty {
                1 => self.chunk = (n & 0x7FFF_FFFF).max(1) as usize,
                // Abort discards the partial message; the chunk stream's header state stays
                2 => {
                    if let Some(c) = self.cs.get_mut(&n) {
                        c.b.clear()
                    }
                }
                _ => {}
            }
            return Ok((ty, b));
        }
    }

    /// Answers protocol and user control messages through `f(type, payload)`, and acknowledges
    /// each window of received bytes.
    fn ctl(&mut self, ty: u8, b: &[u8], mut f: impl FnMut(u8, &[u8])) {
        let n = b
            .get(..4)
            .map_or(0, |x| u32::from_be_bytes(x.try_into().unwrap()));
        match ty {
            // ping request → ping response with the same timestamp
            4 if b.len() >= 6 && b[..2] == [0, 6] => f(4, &[0, 7, b[2], b[3], b[4], b[5]]),
            5 => self.win = n as u64,
            6 if n != self.out => {
                self.out = n;
                f(5, &n.to_be_bytes());
            }
            _ => {}
        }
        if self.win > 0 && self.n - self.acked >= self.win {
            self.acked = self.n;
            f(3, &(self.n as u32).to_be_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn m(ty: u8, ts: u32, key: bool) -> Msg {
        Msg {
            ty,
            ts,
            key,
            b: vec![if key { 0x17 } else { 0x27 }; 10],
        }
    }

    /// Pumps `v` (type, ts, key) through a session that went live with nothing queued, then
    /// stops; what goes out as (type, ts, key), and the drop counts.
    fn session(v: &[(u8, u32, bool)]) -> (Vec<(u8, u32, bool)>, [u64; 2]) {
        let q = Q {
            m: Mutex::new(S {
                lag: 1000,
                ..Default::default()
            }),
            cv: Condvar::new(),
        };
        let mut g = q.m.lock().unwrap();
        g.trim(true);
        v.iter().for_each(|&(ty, ts, k)| g.push(m(ty, ts, k)));
        g.stop = true;
        drop(g);
        let u = Url {
            host: String::new(),
            port: 0,
            app: String::new(),
            tc: String::new(),
            key: "k".into(),
        };
        let mut w = vec![];
        pump(&q, &mut w, &u, 1, true, &mut vec![], &mut vec![]).unwrap();
        // one chunk per message: none is longer than CHUNK
        let (mut r, mut o) = (vec![], &w[..]);
        while !o.is_empty() {
            let be = |x: &[u8]| u32::from_be_bytes([0, x[0], x[1], x[2]]);
            if matches!(o[7], flv::AUDIO | flv::VIDEO) {
                r.push((o[7], be(&o[1..4]), o[12] == 0x17));
            }
            o = &o[12 + be(&o[4..7]) as usize..];
        }
        (r, q.m.lock().unwrap().st.drop)
    }

    #[test]
    fn first_key() {
        // audio arrives first, then the key frame captured before it, as when video takes
        // longer to encode: the session starts at the key frame
        let mut v = vec![(flv::AUDIO, 100, false)];
        for i in 0..10 {
            v.push((flv::VIDEO, 90 + 33 * i, i == 0));
            v.push((flv::AUDIO, 121 + 21 * i, false));
        }
        let (r, d) = session(&v);
        let vid: Vec<_> = r
            .iter()
            .filter(|x| x.0 == flv::VIDEO)
            .map(|x| (x.1, x.2))
            .collect();
        let aud: Vec<_> = r
            .iter()
            .filter(|x| x.0 == flv::AUDIO)
            .map(|x| x.1)
            .collect();
        assert_eq!(vid, (0..10).map(|i| (33 * i, i == 0)).collect::<Vec<_>>());
        assert_eq!(
            aud,
            [10].into_iter()
                .chain((0..10).map(|i| 31 + 21 * i))
                .collect::<Vec<_>>()
        );
        assert_eq!(d, [0, 0]);
        // audio older than the key frame is before the session's start
        let (r, d) = session(&[
            (flv::AUDIO, 70, false),
            (flv::AUDIO, 100, false),
            (flv::VIDEO, 90, true),
        ]);
        assert_eq!(
            (r, d),
            (vec![(flv::AUDIO, 10, false), (flv::VIDEO, 0, true)], [0, 1])
        );
        // no key frame before the stop: nothing goes out
        let (r, d) = session(&[
            (flv::AUDIO, 100, false),
            (flv::VIDEO, 123, false),
            (flv::AUDIO, 121, false),
        ]);
        assert_eq!((r, d), (vec![], [1, 2]));
    }

    #[test]
    fn drops() {
        let mut s = S {
            lag: 100,
            ..Default::default()
        };
        s.push(m(flv::VIDEO, 0, true));
        s.push(m(flv::AUDIO, 10, false));
        s.push(m(flv::VIDEO, 100, false));
        // 101 ms behind the queue head: this P frame and the rest of its GOP go
        s.push(m(flv::VIDEO, 101, false));
        s.push(m(flv::AUDIO, 150, false));
        s.push(m(flv::VIDEO, 133, false));
        assert_eq!((s.q.len(), s.st.drop), (4, [2, 0]));
        s.push(m(flv::AUDIO, 201, false));
        assert_eq!(s.st.drop, [2, 1]);
        // a late key frame replaces everything queued
        s.q.push_front(m(3, 0, false));
        s.push(m(flv::VIDEO, 300, true));
        assert_eq!(
            s.q.iter().map(|m| (m.ty, m.ts)).collect::<Vec<_>>(),
            [(3, 0), (flv::VIDEO, 300)]
        );
        assert_eq!(s.st.drop, [4, 3]);
        assert!(!s.skip);
        // a new session drops video before the first key frame
        let mut s = S {
            lag: 1000,
            ..Default::default()
        };
        for (ty, ts, k) in [
            (flv::VIDEO, 0, false),
            (flv::AUDIO, 5, false),
            (flv::VIDEO, 33, true),
            (flv::AUDIO, 40, false),
        ] {
            s.push(m(ty, ts, k));
        }
        s.trim(true);
        assert_eq!(s.q.iter().map(|m| m.ts).collect::<Vec<_>>(), [5, 33, 40]);
        assert_eq!((s.st.drop, s.skip), ([1, 0], false));
        s.trim(true);
        assert_eq!(s.q.len(), 3);
        // a late key frame keeps the audio from after it
        s.q.push_front(m(3, 0, false));
        s.push(m(flv::AUDIO, 2000, false));
        s.push(m(flv::VIDEO, 1100, true));
        assert_eq!(
            s.q.iter().map(|m| (m.ty, m.ts)).collect::<Vec<_>>(),
            [(3, 0), (flv::AUDIO, 2000), (flv::VIDEO, 1100)]
        );
        assert_eq!(s.st.drop, [2, 2]);
    }

    #[test]
    fn chunks() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut w = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let mut rd = Rd {
            s: l.accept().unwrap().0,
            chunk: 128,
            cs: HashMap::new(),
            n: 0,
            win: 0,
            acked: 0,
            out: 0,
        };
        let body: Vec<u8> = (0..10000u32).map(|i| i as u8).collect();
        let mut o = vec![];
        put(&mut o, 2, 1, 0, 0, &(CHUNK as u32).to_be_bytes());
        put(&mut o, 2, 5, 0, 0, &5000u32.to_be_bytes());
        put(&mut o, 6, flv::VIDEO, 1, 0x1234_5678, &body);
        put(&mut o, 4, flv::AUDIO, 1, 7, &body[..3]);
        put(&mut o, 6, flv::VIDEO, 1, 0xFF_FFFF, &body[..5000]);
        w.write_all(&o).unwrap();
        assert_eq!(rd.msg().unwrap().0, 1);
        assert_eq!(rd.chunk, CHUNK);
        let (ty, b) = rd.msg().unwrap();
        let mut acks = vec![];
        rd.ctl(ty, &b, |ty, b| acks.push((ty, b.to_vec())));
        assert_eq!(rd.win, 5000);
        assert_eq!(rd.msg().unwrap(), (flv::VIDEO, body.clone()));
        rd.ctl(flv::VIDEO, &[], |ty, b| acks.push((ty, b.to_vec())));
        assert_eq!(acks, [(3, (rd.n as u32).to_be_bytes().to_vec())]);
        assert_eq!(rd.msg().unwrap(), (flv::AUDIO, body[..3].to_vec()));
        assert_eq!(rd.msg().unwrap(), (flv::VIDEO, body[..5000].to_vec()));
        rd.win = 0;
        let mut r = vec![];
        rd.ctl(4, &[0, 6, 1, 2, 3, 4], |ty, b| r.push((ty, b.to_vec())));
        rd.ctl(6, &[0, 0, 0x10, 0, 2], |ty, b| r.push((ty, b.to_vec())));
        rd.ctl(6, &[0, 0, 0x10, 0, 2], |ty, b| r.push((ty, b.to_vec())));
        assert_eq!(r, [(4, vec![0, 7, 1, 2, 3, 4]), (5, vec![0, 0, 0x10, 0])]);
        // an aborted message; the next one on that chunk stream reuses its header (fmt 3)
        o.clear();
        put(&mut o, 6, flv::VIDEO, 1, 0, &body[..5000]);
        o.truncate(12 + CHUNK);
        put(&mut o, 2, 2, 0, 0, &6u32.to_be_bytes());
        for c in body[1..5001].chunks(CHUNK) {
            o.push(0xC6);
            o.extend_from_slice(c);
        }
        w.write_all(&o).unwrap();
        assert_eq!(rd.msg().unwrap(), (2, 6u32.to_be_bytes().to_vec()));
        assert_eq!(rd.msg().unwrap(), (flv::VIDEO, body[1..5001].to_vec()));
    }

    #[test]
    fn urls() {
        let cfg = || Cfg {
            meta: Default::default(),
            avcc: vec![],
            asc: vec![],
            lag: 0,
        };
        assert!(Rtmp::start("rtmps://a.example/app", "k", cfg()).is_err());
        assert!(Rtmp::start("rtmp://a.example", "k", cfg()).is_err());
        assert!(Rtmp::start("rtmp://a.example:x/app", "k", cfg()).is_err());
        // the port is closed, so it keeps retrying until dropped
        let r = Rtmp::start("rtmp://127.0.0.1:9/live/test", "", cfg()).unwrap();
        r.video(0, 0, true, [&[0x65u8][..]]);
        drop(r);
    }

    /// Dropping never waits for a pending connect or a silent server.
    #[cfg(target_os = "linux")]
    #[test]
    fn stop() {
        use std::{os::fd::AsRawFd, time::Instant};
        let cfg = || Cfg {
            meta: Default::default(),
            avcc: vec![],
            asc: vec![],
            lag: 0,
        };
        // a full accept queue: the kernel drops the next SYN, so that connect hangs
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        assert_eq!(unsafe { libc::listen(l.as_raw_fd(), 0) }, 0);
        let a = l.local_addr().unwrap();
        let _q = TcpStream::connect(a).unwrap();
        let r = Rtmp::start(&format!("rtmp://{a}/live/test"), "", cfg()).unwrap();
        thread::sleep(Duration::from_millis(300));
        assert_eq!(r.stats().state, State::Connecting);
        let t = Instant::now();
        drop(r);
        let e1 = t.elapsed();
        // a server that takes the connection and never answers the handshake
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let u = format!("rtmp://{}/live/test", l.local_addr().unwrap());
        let r = Rtmp::start(&u, "", cfg()).unwrap();
        let _c = l.accept().unwrap();
        thread::sleep(Duration::from_millis(50));
        let t = Instant::now();
        drop(r);
        let e2 = t.elapsed();
        println!("drop while connecting: {e1:?}, during the handshake: {e2:?}");
        assert!(e1.max(e2) < Duration::from_millis(100));
    }
}
