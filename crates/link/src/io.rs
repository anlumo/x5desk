//! The camera's I/O thread: sync, heartbeats, requests and responses, fragments, notifications, stream packets
//! and HTTP transfers, over any [`Link`].

use crate::{
    Done, Error, Event, Frame, Link, Msg, Note, Pool, Result,
    http::{Prog, Xfer},
    pb,
    ucd::{self, Pkt},
};
use prost::Message;
use std::{
    borrow::Cow,
    hash::{BuildHasher, RandomState},
    path::PathBuf,
    sync::{
        Arc,
        atomic::Ordering::Relaxed,
        mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, sync_channel},
    },
    time::{Duration, Instant},
};

const SEC: Duration = Duration::from_secs(1);
/// No inbound byte for this long: the link is dead (the camera heartbeats about once a second, §8.10).
const IDLE: Duration = Duration::from_secs(30);

pub(crate) struct Io<L> {
    l: L,
    ch: Receiver<Msg>,
    ev: Sender<Event>,
    pool: Pool,
    rx: Vec<u8>,
    out: Vec<u8>,
    id: u32,
    reqs: Vec<(u32, Instant, Done)>,
    frags: Vec<(u32, Vec<u8>)>,
    xs: Vec<Xfer>,
    nc: u32,
    hb: Instant,
    seen: Instant,
    live: bool,
    auth: Option<i32>,
    closing: Vec<SyncSender<Result<()>>>,
}

pub(crate) fn run<L: Link>(
    open: impl FnOnce(Sender<Msg>, Pool) -> Result<L>,
    tx: Sender<Msg>,
    ch: Receiver<Msg>,
    ev: Sender<Event>,
    uuid: Option<String>,
    ready: SyncSender<Result<()>>,
) {
    let pool = Pool::default();
    let l = match open(tx, pool.clone()) {
        Ok(l) => l,
        Err(e) => return drop(ready.send(Err(e))),
    };
    let now = Instant::now();
    let mut s = Io {
        l,
        ch,
        ev,
        pool,
        rx: vec![],
        out: vec![],
        id: 0,
        reqs: vec![],
        frags: vec![],
        xs: vec![],
        // Tunnel ids unlike the last session's: what the camera still sends for those (after a crash, up to the
        // granted windows) must not land in this session's connections.
        nc: RandomState::new().hash_one(()) as u32,
        hb: now,
        seen: now,
        live: false,
        auth: None,
        closing: vec![],
    };
    match s
        .sync()
        .and_then(|_| uuid.map_or(Ok(()), |u| s.authorize(&u)))
    {
        Ok(()) => {
            let _ = ready.send(Ok(()));
            let e = loop {
                if let Err(e) = s.step(SEC) {
                    break e;
                }
            };
            s.shut(Some(e))
        }
        Err(e) => {
            // Released before the caller hears of it, so it can retry right away.
            s.shut(None);
            let _ = ready.send(Err(e));
        }
    }
}

impl<L: Link> Io<L> {
    fn recycle(&self, v: Vec<u8>) {
        let mut p = self.pool.lock().unwrap();
        if p.len() < 64 {
            p.push(v)
        }
    }

    /// Sync handshake (§3.3): drop everything before the marker, frame from the byte after it.
    fn sync(&mut self) -> Result<()> {
        self.l.send(ucd::SYNC)?;
        self.heartbeat()?;
        let end = Instant::now() + Duration::from_secs(30);
        let mut got = 0;
        loop {
            let now = Instant::now();
            if now >= end {
                let m = format!("no sync marker in 30 s, {got} bytes received");
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    m,
                )));
            }
            if now >= self.hb + SEC {
                self.heartbeat()?
            }
            match self
                .ch
                .recv_timeout((self.hb + SEC).min(end).saturating_duration_since(now))
            {
                Ok(Msg::Rx(v)) => {
                    got += v.len();
                    self.rx.extend_from_slice(&v);
                    self.recycle(v);
                    if self.rx.windows(8).any(|w| w == ucd::BUSY) {
                        return Err(Error::Busy);
                    }
                    if let Some(i) = ucd::mark(&self.rx) {
                        self.rx.drain(..i);
                        self.seen = Instant::now();
                        return self.parse();
                    }
                    let k = self.rx.len().saturating_sub(ucd::MARK.len() - 1);
                    self.rx.drain(..k);
                }
                Ok(Msg::Close(tx)) => return Err(self.close(tx)),
                Ok(Msg::Gone(e)) => return Err(e),
                Ok(_) | Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Err(Error::Closed),
            }
        }
    }

    fn close(&mut self, tx: Option<SyncSender<Result<()>>>) -> Error {
        self.closing.extend(tx);
        Error::Closed
    }

    fn heartbeat(&mut self) -> Result<()> {
        self.hb = Instant::now();
        Ok(self.l.send(ucd::HB)?)
    }

    /// Wi-Fi only, in the app's order (§5.1): CHECK_TYPE (GET_OPTIONS), then CHECK_AUTHORIZATION; on UNAUTHORIZED
    /// wait for the user to confirm on the camera (8209; PENDING keeps waiting).
    fn authorize(&mut self, uuid: &str) -> Result<()> {
        let call = |s: &mut Self, code, q: Vec<u8>| {
            let (tx, rx) = sync_channel(1);
            s.request(code, &q, 10 * SEC, Box::new(move |r| drop(tx.send(r))))?;
            s.pump(11 * SEC, |_| rx.try_recv().ok())?
        };
        let q = pb::GetOptions {
            option_types: vec![48, 15, 11, 198],
        };
        call(self, crate::code::GET_OPTIONS, q.encode_to_vec())?;
        // An 8209 may arrive right behind the response.
        self.auth = None;
        let q = pb::CheckAuthorization {
            id: uuid.into(),
            initiator_type: 2,
            ..Default::default()
        };
        let b = call(self, crate::code::CHECK_AUTH, q.encode_to_vec())?;
        match pb::CheckAuthorizationResp::decode(&b[..])?.authorization_status {
            0 => Ok(()),
            1 => match self.pump(60 * SEC, |s| s.auth.take().filter(|&r| r != 4))? {
                0 => Ok(()),
                r => Err(Error::Unauthorized(r)),
            },
            s => Err(Error::Unauthorized(s)),
        }
    }

    /// Runs the loop until `f` yields, at most `t`.
    fn pump<T>(&mut self, t: Duration, mut f: impl FnMut(&mut Self) -> Option<T>) -> Result<T> {
        let end = Instant::now() + t;
        loop {
            if let Some(x) = f(self) {
                return Ok(x);
            }
            let now = Instant::now();
            if now >= end {
                return Err(Error::Timeout);
            }
            self.step(end - now)?
        }
    }

    /// One turn of the loop: wait up to `t` for one message, then heartbeat, timeouts and transfer upkeep.
    fn step(&mut self, t: Duration) -> Result<()> {
        let now = Instant::now();
        let wake = self
            .reqs
            .iter()
            .map(|r| r.1)
            .fold((self.hb + SEC).min(now + t), Instant::min);
        match self.ch.recv_timeout(wake.saturating_duration_since(now)) {
            Ok(Msg::Rx(mut v)) => {
                if self.rx.is_empty() {
                    std::mem::swap(&mut self.rx, &mut v);
                } else {
                    self.rx.extend_from_slice(&v)
                }
                self.recycle(v);
                self.seen = Instant::now();
                self.parse()?
            }
            Ok(Msg::Http(c, v)) => {
                self.http(c, &v, v.is_empty())?;
                self.recycle(v)
            }
            Ok(Msg::Req(code, body, t, done)) if self.closing.is_empty() => {
                self.request(code, &body, t, done)?
            }
            Ok(Msg::Req(.., done)) => done(Err(Error::Closed)),
            Ok(Msg::Get(uri, path, n, prog, done)) if self.closing.is_empty() => {
                self.get(uri, path, n, prog, done)?
            }
            Ok(Msg::Get(.., done)) => done(Err(Error::Closed)),
            Ok(Msg::Close(tx)) => return Err(self.close(tx)),
            Ok(Msg::Gone(e)) => return Err(e),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Err(Error::Closed),
        }
        let now = Instant::now();
        if now >= self.hb + SEC {
            self.heartbeat()?
        }
        if now - self.seen >= IDLE {
            return Err(Error::Timeout);
        }
        for i in (0..self.reqs.len()).rev() {
            if self.reqs[i].1 <= now {
                (self.reqs.swap_remove(i).2)(Err(Error::Timeout))
            }
        }
        // Back to front: `fail` and `fill` may remove the transfer.
        for i in (0..self.xs.len()).rev() {
            let x = &self.xs[i];
            if x.prog.cancel.load(Relaxed) {
                self.fail(i, Error::Cancelled)
            } else if now - x.last >= IDLE {
                self.fail(i, Error::Timeout)
            } else if x.cs.is_empty() {
                // Held back while the link was busy.
                self.fill(i)?
            }
        }
        Ok(())
    }

    /// Frames every whole packet in `rx`.
    fn parse(&mut self) -> Result<()> {
        let rx = std::mem::take(&mut self.rx);
        let mut o = 0;
        let r = loop {
            match ucd::next(&rx[o..]) {
                Ok(Some((p, n))) => {
                    o += n;
                    if let Err(e) = self.on(p) {
                        break Err(e);
                    }
                }
                Ok(None) => break Ok(()),
                Err(()) => break Err(Error::Desync),
            }
        };
        self.rx = rx;
        self.rx.drain(..o);
        r
    }

    fn on(&mut self, p: Pkt) -> Result<()> {
        match p {
            Pkt::Msg {
                code,
                note,
                id,
                end,
                body,
            } => {
                // Reassembly by (direction, id); method and status come from the last fragment (§4.8).
                let key = id | (note as u32) << 30;
                let i = self.frags.iter().position(|f| f.0 == key);
                if !end {
                    match i {
                        Some(i) => self.frags[i].1.extend_from_slice(body),
                        None => {
                            if self.frags.len() == 256 {
                                self.frags.remove(0);
                            }
                            self.frags.push((key, body.to_vec()))
                        }
                    }
                    return Ok(());
                }
                let b = match i {
                    Some(i) => {
                        let mut v = self.frags.remove(i).1;
                        v.extend_from_slice(body);
                        Cow::Owned(v)
                    }
                    None => Cow::Borrowed(body),
                };
                if note {
                    // An empty body is {SUCCESS, BLE_CONNECT} (§5.2); other operation types are not about this
                    // connection.
                    if code == 8209
                        && let Ok(n) = pb::NotificationAuthorizationResult::decode(&*b)
                        && n.authorization_operation_type == 0
                    {
                        self.auth = Some(n.authorization_result)
                    }
                    let _ = self.ev.send(Event::Note(Note::new(code, &b)));
                } else if let Some(i) = self.reqs.iter().position(|r| r.0 == id) {
                    let done = self.reqs.swap_remove(i).2;
                    done(if code == 200 {
                        Ok(b.into_owned())
                    } else {
                        Err(Error::Status(code, pb::Error::decode(&*b).ok()))
                    })
                }
            }
            Pkt::Stream { kind, ts, data } => {
                let _ = self.ev.send(Event::Frame(Frame {
                    kind,
                    ts,
                    data: data.to_vec(),
                }));
            }
            // Camera→host: 0x04 data, 0x02 / 0x10 close (§8.6).
            Pkt::Tunnel { c, flags, data } => self.http(c, data, flags & 0x12 != 0)?,
            Pkt::Other => {}
        }
        Ok(())
    }

    fn request(&mut self, code: u16, body: &[u8], t: Duration, done: Done) -> Result<()> {
        self.id = if self.id >= 0x3FFF_FFFF {
            1
        } else {
            self.id + 1
        };
        match code {
            crate::code::START_LIVE => self.live = true,
            crate::code::STOP_LIVE => self.live = false,
            _ => {}
        }
        self.reqs.push((self.id, Instant::now() + t, done));
        let n = body.len().div_ceil(ucd::FRAG).max(1);
        for i in 0..n {
            self.out.clear();
            ucd::msg(
                &mut self.out,
                code,
                self.id,
                i + 1 == n,
                &body[i * ucd::FRAG..((i + 1) * ucd::FRAG).min(body.len())],
            );
            self.l.send(&self.out)?
        }
        Ok(())
    }

    fn get(
        &mut self,
        uri: String,
        path: Option<PathBuf>,
        n: usize,
        prog: Arc<Prog>,
        done: Done,
    ) -> Result<()> {
        match Xfer::new(uri, path.as_deref(), n, prog) {
            Ok(mut x) => {
                x.done = Some(done);
                self.xs.push(x);
                self.fill(self.xs.len() - 1)
            }
            Err(e) => Ok(done(Err(e.into()))),
        }
    }

    /// Bytes (or the end) of HTTP connection `c`.
    fn http(&mut self, c: u32, d: &[u8], eof: bool) -> Result<()> {
        let Io { l, xs, .. } = self;
        let Some((xi, i)) = xs
            .iter()
            .enumerate()
            .find_map(|(xi, x)| Some((xi, x.cs.iter().position(|k| k.c == c)?)))
        else {
            return Ok(());
        };
        let x = &mut xs[xi];
        let mut gone = false;
        x.cs[i].got += d.len() as u64;
        l.http_ack(c, x.cs[i].got)?;
        if !d.is_empty() {
            match x.data(i, d) {
                Ok(true) => {
                    x.cs.remove(i);
                    gone = true;
                }
                Ok(false) => {}
                Err(e) => return Ok(self.fail(xi, e)),
            }
        }
        if eof && !gone {
            x.drop_conn(i);
            gone = true;
        }
        if gone {
            l.http_close(c)?;
        }
        if x.lost > 3 {
            return Ok(self.fail(xi, Error::Http("connection closed early".into())));
        }
        self.fill(xi)
    }

    /// Opens connections for transfer `xi` as far as it and the link allow; completes it when it is done.
    fn fill(&mut self, xi: usize) -> Result<()> {
        let Io { l, xs, nc, .. } = self;
        let x = &mut xs[xi];
        if !l.busy() {
            let host = l.host().to_owned();
            if let Err(e) = x.fill(nc, &host, &mut |c, q| l.http_open(c, q).map(|_| l.busy())) {
                return Ok(self.fail(xi, e.into()));
            }
        }
        if x.complete() {
            let mut x = self.xs.swap_remove(xi);
            let done = x.done.take().unwrap();
            done(x.finish())
        }
        Ok(())
    }

    fn fail(&mut self, xi: usize, e: Error) {
        let mut x = self.xs.swap_remove(xi);
        for k in &x.cs {
            let _ = self.l.http_close(k.c);
        }
        if matches!(e, Error::Changed) {
            x.discard()
        } else {
            x.save()
        }
        (x.done.take().unwrap())(Err(e))
    }

    /// Fails the transfers (closing their connections), and on a clean close reads up to the answer of one last
    /// request: STOP_LIVE_STREAM if the preview runs, else a capture status. Over USB the camera hands whatever it
    /// already queued for the IN pipe to the next client: up to 2.2 MB of tunnel data after a 4-connection
    /// download, and a client that sends only SYNC then missed the marker for 30 s. The camera answers after all
    /// of it. Then fails what is pending and stops the transport.
    fn shut(mut self, e: Option<Error>) {
        while !self.xs.is_empty() {
            self.fail(0, Error::Closed)
        }
        if matches!(e, Some(Error::Closed)) {
            let (tx, rx) = sync_channel(1);
            let code = if self.live {
                crate::code::STOP_LIVE
            } else {
                crate::code::CAPTURE_STATUS
            };
            if self
                .request(code, &[], 2 * SEC, Box::new(move |r| drop(tx.send(r))))
                .is_ok()
            {
                let end = Instant::now() + 2 * SEC;
                while rx.try_recv().is_err() && Instant::now() < end {
                    if let Err(Error::Closed) | Ok(()) = self.step(end - Instant::now()) {
                        continue;
                    }
                    break;
                }
            }
        }
        for r in self.reqs.drain(..) {
            (r.2)(Err(Error::Closed))
        }
        self.l.stop();
        if let Some(e) = e.filter(|e| !matches!(e, Error::Closed)) {
            let _ = self.ev.send(Event::Closed(e));
        }
        // Release the interface before the caller hears that the session is closed.
        let closing = std::mem::take(&mut self.closing);
        drop(self);
        for tx in closing {
            let _ = tx.send(Ok(()));
        }
    }
}
