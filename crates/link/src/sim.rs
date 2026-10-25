//! A simulated camera for tests of the code above the link (feature `sim`): the real I/O thread runs over a
//! transport that syncs like the X5, hands every request to a closure and serves HTTP through the socket tunnel the
//! way the X5 does (range start honoured, body to EOF in paced chunks, connections never closed by the camera).

use crate::{Camera, Error, Event, Link, Msg, Result, start, ucd};
use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::Duration,
};

/// Request handler: method and body in, status and body out.
pub type Req = Box<dyn FnMut(u16, &[u8]) -> (u16, Vec<u8>) + Send>;
/// The camera's files by URI.
pub type Files = Arc<dyn Fn(&str) -> Option<Arc<Vec<u8>>> + Send + Sync>;

/// Control of a running simulation.
#[derive(Clone)]
pub struct Sim(Sender<Msg>);

impl Sim {
    /// Sends notification `code`.
    pub fn note(&self, code: u16, body: &[u8]) {
        let mut v = vec![];
        let mut h = [0; 9];
        h[..2].copy_from_slice(&code.to_le_bytes());
        h[2] = 2;
        // end | camera→phone
        h[3..7].copy_from_slice(&(3u32 << 30).to_le_bytes());
        ucd::put(&mut v, 4, &h, body);
        let _ = self.0.send(Msg::Rx(v));
    }

    /// Sends stream packet `kind` (a live-preview packet, [`crate::stream`]) stamped `ts` (camera ms).
    pub fn frame(&self, kind: u8, ts: u64, data: &[u8]) {
        let mut v = Vec::with_capacity(16 + data.len());
        let mut h = [0; 9];
        h[0] = kind;
        h[1..].copy_from_slice(&ts.to_le_bytes());
        ucd::put(&mut v, 1, &h, data);
        let _ = self.0.send(Msg::Rx(v));
    }

    /// The camera is unplugged: the session ends with [`Error::NoDevice`].
    pub fn unplug(&self) {
        let _ = self.0.send(Msg::Gone(Error::NoDevice));
    }
}

struct Cam {
    tx: Sender<Msg>,
    req: Req,
    files: Files,
    /// bytes per second and connection
    rate: u64,
    open: Vec<(u32, Arc<AtomicBool>)>,
}

impl Link for Cam {
    fn send(&mut self, p: &[u8]) -> io::Result<()> {
        let mut v = vec![];
        if p == ucd::SYNC {
            v.extend(ucd::HB);
            v.extend(ucd::SYNC);
        } else if let Ok(Some((ucd::Pkt::Msg { code, id, body, .. }, _))) = ucd::next(p) {
            let (s, b) = (self.req)(code, body);
            ucd::msg(&mut v, s, id, true, &b);
        }
        if !v.is_empty() {
            let _ = self.tx.send(Msg::Rx(v));
        }
        Ok(())
    }

    fn http_open(&mut self, c: u32, q: &[u8]) -> io::Result<()> {
        let q = String::from_utf8_lossy(q);
        let uri = q.split(' ').nth(1).unwrap_or("").to_owned();
        let a: usize = q
            .split("bytes=")
            .nth(1)
            .and_then(|r| r.split('-').next()?.parse().ok())
            .unwrap_or(0);
        let stop = Arc::new(AtomicBool::new(false));
        self.open.push((c, stop.clone()));
        let (tx, f, rate) = (self.tx.clone(), (self.files)(&uri), self.rate.max(1));
        thread::spawn(move || {
            let send = |d: &[u8]| {
                let mut v = vec![];
                ucd::tunnel(&mut v, c, 0x04, 0, d);
                tx.send(Msg::Rx(v)).is_ok()
            };
            let Some(f) = f.filter(|f| a < f.len()) else {
                send(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                return;
            };
            let n = f.len();
            let h = format!(
                "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {a}-{}/{n}\r\n\
                 Content-Length: {}\r\nEtag: \"5f00.{n}\"\r\nConnection: keep-alive\r\nConnection: close\r\n\r\n",
                n - 1,
                n - a
            );
            if !send(h.as_bytes()) {
                return;
            }
            const CHUNK: usize = 128 << 10;
            let gap = Duration::from_secs_f64(CHUNK as f64 / rate as f64);
            for d in f[a..].chunks(CHUNK) {
                thread::sleep(gap);
                if stop.load(Relaxed) || !send(d) {
                    return;
                }
            }
        });
        Ok(())
    }

    fn http_ack(&mut self, _: u32, _: u64) -> io::Result<()> {
        Ok(())
    }

    fn http_close(&mut self, c: u32) -> io::Result<()> {
        if let Some(i) = self.open.iter().position(|o| o.0 == c) {
            self.open.swap_remove(i).1.store(true, Relaxed);
        }
        Ok(())
    }

    fn busy(&self) -> bool {
        false
    }

    fn host(&self) -> &str {
        "127.0.0.1"
    }

    fn stop(&mut self) {
        for (_, s) in self.open.drain(..) {
            s.store(true, Relaxed)
        }
    }
}

/// A camera that never answers: enumerated in app mode but asleep.
struct Mute;

impl Link for Mute {
    fn send(&mut self, _: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn http_open(&mut self, _: u32, _: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn http_ack(&mut self, _: u32, _: u64) -> io::Result<()> {
        Ok(())
    }

    fn http_close(&mut self, _: u32) -> io::Result<()> {
        Ok(())
    }

    fn busy(&self) -> bool {
        false
    }

    fn host(&self) -> &str {
        "127.0.0.1"
    }

    fn stop(&mut self) {}
}

impl Camera {
    /// A connection to a camera that never answers: it syncs until [`Error::Silent`] after 30 s, or until it is
    /// dropped.
    pub fn mute() -> crate::Opening {
        start(|_, _| Ok(Mute), None)
    }

    /// A session with a simulated camera: `req` answers requests, `files` serves HTTP at `rate` bytes per second
    /// and connection.
    pub fn sim(
        req: impl FnMut(u16, &[u8]) -> (u16, Vec<u8>) + Send + 'static,
        files: impl Fn(&str) -> Option<Arc<Vec<u8>>> + Send + Sync + 'static,
        rate: u64,
    ) -> Result<(Camera, Receiver<Event>, Sim)> {
        let (stx, srx) = mpsc::sync_channel(1);
        let files: Files = Arc::new(files);
        let req: Req = Box::new(req);
        let (c, ev) = start(
            move |tx: Sender<Msg>, _| {
                let _ = stx.send(tx.clone());
                Ok(Cam {
                    tx,
                    req,
                    files,
                    rate,
                    open: vec![],
                })
            },
            None,
        )
        .wait()?;
        let s = srx.recv().map_err(|_| Error::Closed)?;
        Ok((c, ev, Sim(s)))
    }
}
