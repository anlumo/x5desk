//! Wi-Fi (PROTOCOL.md §3): UCD on TCP 6666 of the camera's access point, HTTP on port 80.

use crate::{Error, Link, Msg, Pool, Result};
use std::{
    io::{self, Read, Write},
    net::{Shutdown, SocketAddr, TcpStream},
    sync::mpsc::Sender,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const CONNECT: Duration = Duration::from_secs(5);
/// Minimum gap between two control packets (§3.6).
const GAP: Duration = Duration::from_millis(4);

pub(crate) struct Tcp {
    ctl: TcpStream,
    web: SocketAddr,
    host: String,
    http: Vec<(u32, TcpStream)>,
    tx: Sender<Msg>,
    pool: Pool,
    last: Instant,
    rd: Vec<JoinHandle<()>>,
}

/// Forwards everything `s` receives: control bytes as `Rx`, HTTP bytes as `Http(c)`, an HTTP end as an empty `Http`.
fn reader(
    mut s: TcpStream,
    c: Option<u32>,
    tx: Sender<Msg>,
    pool: Pool,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("x5-tcp-rx".into())
        .spawn(move || {
            let mut b = vec![0; 256 << 10];
            loop {
                let n = match s.read(&mut b) {
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Ok(0) | Err(_) if c.is_some() => 0,
                    Ok(0) => {
                        return drop(
                            tx.send(Msg::Gone(Error::Io(io::ErrorKind::UnexpectedEof.into()))),
                        );
                    }
                    Err(e) => return drop(tx.send(Msg::Gone(Error::Io(e)))),
                    Ok(n) => n,
                };
                let mut v = pool.lock().unwrap().pop().unwrap_or_default();
                v.clear();
                v.extend_from_slice(&b[..n]);
                if tx
                    .send(match c {
                        Some(c) => Msg::Http(c, v),
                        None => Msg::Rx(v),
                    })
                    .is_err()
                    || n == 0
                {
                    return;
                }
            }
        })
}

impl Tcp {
    /// `ctl`: the UCD port (6666), `web`: the HTTP server (80).
    pub fn open(ctl: SocketAddr, web: SocketAddr, tx: Sender<Msg>, pool: Pool) -> Result<Tcp> {
        let host = web.ip().to_string();
        let ctl = TcpStream::connect_timeout(&ctl, CONNECT)?;
        ctl.set_nodelay(true)?;
        // A camera that stops reading would otherwise block the I/O thread for good.
        ctl.set_write_timeout(Some(Duration::from_secs(5)))?;
        let rd = vec![reader(ctl.try_clone()?, None, tx.clone(), pool.clone())?];
        Ok(Tcp {
            ctl,
            web,
            host,
            http: vec![],
            tx,
            pool,
            last: Instant::now() - GAP,
            rd,
        })
    }
}

impl Link for Tcp {
    fn send(&mut self, p: &[u8]) -> io::Result<()> {
        if let Some(w) = (self.last + GAP).checked_duration_since(Instant::now()) {
            thread::sleep(w)
        }
        self.last = Instant::now();
        self.ctl.write_all(p)
    }

    fn http_open(&mut self, c: u32, q: &[u8]) -> io::Result<()> {
        let mut s = TcpStream::connect_timeout(&self.web, CONNECT)?;
        s.write_all(q)?;
        self.rd.retain(|h| !h.is_finished());
        self.rd.push(reader(
            s.try_clone()?,
            Some(c),
            self.tx.clone(),
            self.pool.clone(),
        )?);
        self.http.push((c, s));
        Ok(())
    }

    fn http_ack(&mut self, _: u32, _: u64) -> io::Result<()> {
        Ok(())
    }

    fn http_close(&mut self, c: u32) -> io::Result<()> {
        if let Some(i) = self.http.iter().position(|h| h.0 == c) {
            let _ = self.http.swap_remove(i).1.shutdown(Shutdown::Both);
        }
        Ok(())
    }

    fn busy(&self) -> bool {
        false
    }

    fn host(&self) -> &str {
        &self.host
    }

    fn stop(&mut self) {
        let _ = self.ctl.shutdown(Shutdown::Both);
        for (_, s) in self.http.drain(..) {
            let _ = s.shutdown(Shutdown::Both);
        }
        for h in self.rd.drain(..) {
            let _ = h.join();
        }
    }
}
