//! USB app mode (PROTOCOL.md §8): vendor interface 0, bulk OUT 0x01 / IN 0x81; HTTP through the SocketTunnel.

use crate::{Error, Link, Msg, Pool, Result, ucd};
use futures_core::Stream;
use nusb::{
    DeviceInfo, Endpoint, Interface, MaybeFuture,
    transfer::{Buffer, Bulk, In, Out, TransferError},
};
use std::{
    io,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc::Sender,
    },
    task::{Context, Poll, Wake, Waker},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Each camera packet ends in a short USB packet (§8.3), so one read returns at most one whole packet; tunnel
/// packets carry up to 512 KiB.
const RX: usize = 1 << 20;
/// Window the host grants each tunnel connection beyond what it consumed, and the step at which it re-announces
/// it (the app's values, §8.6).
const WIN: u64 = 0xA0_0000;
const STEP: u64 = 0x20_0000;
/// Minimum gap between two packets (both app stacks throttle sends, §3.6).
const GAP: Duration = Duration::from_millis(4);

pub(crate) struct Usb {
    out: Endpoint<Bulk, Out>,
    buf: Option<Buffer>,
    v: Vec<u8>,
    last: Instant,
    /// Tunnel connections and the window last announced for each.
    tun: Vec<(u32, u64)>,
    first: bool,
    /// The connection that carried RESET_ALL, until it answers.
    gate: Option<u32>,
    stop: Arc<AtomicBool>,
    rd: Option<JoinHandle<()>>,
    _if: Interface,
}

/// An Insta360 device on the bus: none, a camera as a USB disk (the X5's U-disk mode, also the first seconds after
/// plug-in in app mode, without media then), one in another mode (Webcam Mode), or one in app mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Plug {
    None,
    Disk,
    Other,
    App,
}

/// The PID is not fixed (the app checks the VID only); Webcam Mode shares the VID but has no vendor interface. As a
/// disk the X5 is `070a:4027` (§8.1), the vendor id of its USB stack, not Insta360's: the product string tells, as in
/// the app's mass-storage test.
fn plug(d: &DeviceInfo) -> Plug {
    match d.vendor_id() {
        0x2e1a | 0x4255
            if d.interfaces()
                .any(|i| i.interface_number() == 0 && i.class() == 0xff) =>
        {
            Plug::App
        }
        0x2e1a | 0x4255 => Plug::Other,
        0x070a
            if d.product_string()
                .is_some_and(|s| s.starts_with("Insta360")) =>
        {
            Plug::Disk
        }
        _ => Plug::None,
    }
}

/// The best [`Plug`] state on the bus now.
pub fn scan() -> Plug {
    nusb::list_devices()
        .wait()
        .ok()
        .and_then(|ds| ds.map(|d| plug(&d)).max())
        .unwrap_or(Plug::None)
}

struct Unpark(thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark()
    }
}

/// Watches the bus on its own thread: calls `f` with the [`scan`] state now, then after every change, until `f`
/// returns false.
pub fn watch(mut f: impl FnMut(Plug) -> bool + Send + 'static) -> Result<()> {
    let mut w = nusb::watch_devices()?;
    thread::Builder::new()
        .name("x5-hotplug".into())
        .spawn(move || {
            let mut p = scan();
            if !f(p) {
                return;
            }
            let wk = Waker::from(Arc::new(Unpark(thread::current())));
            let mut cx = Context::from_waker(&wk);
            loop {
                match Pin::new(&mut w).poll_next(&mut cx) {
                    Poll::Ready(Some(_)) => {}
                    Poll::Ready(None) => return,
                    Poll::Pending => {
                        thread::park();
                        continue;
                    }
                }
                // A device's interfaces can show up after its event, and the X5 passes through other ids while it
                // switches modes: look again for a while after any event.
                let end = Instant::now() + Duration::from_secs(2);
                loop {
                    let q = scan();
                    if q != p {
                        p = q;
                        if !f(p) {
                            return;
                        }
                    }
                    if Instant::now() >= end {
                        break;
                    }
                    thread::sleep(Duration::from_millis(250));
                }
            }
        })?;
    Ok(())
}

impl Usb {
    pub fn open(tx: Sender<Msg>, pool: Pool) -> Result<Usb> {
        let di = nusb::list_devices()
            .wait()?
            .find(|d| plug(d) == Plug::App)
            .ok_or(Error::NoDevice)?;
        let iface = di.open().wait()?.detach_and_claim_interface(0).wait()?;
        let out = iface.endpoint::<Bulk, Out>(0x01)?;
        let mut inp = iface.endpoint::<Bulk, In>(0x81)?;
        // Reads pending before anything is sent (§8.9).
        for _ in 0..4 {
            let mut b = inp.allocate(RX);
            b.set_requested_len(RX);
            inp.submit(b);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let rd = thread::Builder::new()
            .name("x5-usb-rx".into())
            .spawn(move || {
                while !s.load(Relaxed) {
                    let Some(c) = inp.wait_next_complete(Duration::from_millis(100)) else {
                        continue;
                    };
                    if let Err(e) = c.status {
                        let e = if e == TransferError::Disconnected {
                            Error::NoDevice
                        } else {
                            Error::Io(e.into())
                        };
                        let _ = tx.send(Msg::Gone(e));
                        return;
                    }
                    if c.actual_len > 0 {
                        let mut v = pool.lock().unwrap().pop().unwrap_or_default();
                        v.clear();
                        v.extend_from_slice(&c.buffer[..c.actual_len]);
                        if tx.send(Msg::Rx(v)).is_err() {
                            return;
                        }
                    }
                    let mut b = c.buffer;
                    b.set_requested_len(RX);
                    inp.submit(b);
                }
            })?;
        let now = Instant::now();
        Ok(Usb {
            out,
            buf: None,
            v: vec![],
            last: now - GAP,
            tun: vec![],
            first: true,
            gate: None,
            stop,
            rd: Some(rd),
            _if: iface,
        })
    }

    fn tunnel(&mut self, c: u32, flags: u16, win: u64, data: &[u8]) -> io::Result<()> {
        let mut v = std::mem::take(&mut self.v);
        v.clear();
        ucd::tunnel(&mut v, c, flags, win, data);
        let r = self.send(&v);
        self.v = v;
        r
    }
}

impl Link for Usb {
    /// One UCD packet per bulk OUT submission, never coalesced, no ZLP (§8.3).
    fn send(&mut self, p: &[u8]) -> io::Result<()> {
        if let Some(w) = (self.last + GAP).checked_duration_since(Instant::now()) {
            thread::sleep(w)
        }
        let mut b = match self.buf.take() {
            Some(b) if b.capacity() >= p.len() => b,
            _ => self.out.allocate(p.len().max(1 << 16)),
        };
        b.clear();
        b.extend_from_slice(p);
        self.out.submit(b);
        let c = self
            .out
            .wait_next_complete(Duration::from_secs(5))
            .ok_or(io::ErrorKind::TimedOut)?;
        self.last = Instant::now();
        c.status?;
        self.buf = Some(c.buffer);
        Ok(())
    }

    fn http_open(&mut self, c: u32, q: &[u8]) -> io::Result<()> {
        // NEW_CON | DATA | WINDOW; the session's first connection also resets the camera's stale ones (0x20).
        let f = if self.first { 0x2d } else { 0x0d };
        if self.first {
            self.gate = Some(c);
        }
        self.first = false;
        self.tun.push((c, WIN));
        self.tunnel(c, f, WIN, q)
    }

    fn http_ack(&mut self, c: u32, n: u64) -> io::Result<()> {
        if self.gate == Some(c) {
            self.gate = None;
        }
        let Some(i) = self.tun.iter().position(|t| t.0 == c) else {
            return Ok(());
        };
        if n + WIN - self.tun[i].1 < STEP {
            return Ok(());
        }
        self.tun[i].1 = n + WIN;
        self.tunnel(c, 0x08, n + WIN, &[])
    }

    fn http_close(&mut self, c: u32) -> io::Result<()> {
        if self.gate == Some(c) {
            self.gate = None;
        }
        let Some(i) = self.tun.iter().position(|t| t.0 == c) else {
            return Ok(());
        };
        let w = self.tun.swap_remove(i).1;
        self.tunnel(c, 0x02, w, &[])
    }

    /// The camera applies RESET_ALL asynchronously. After a session that left connections streaming, it answered
    /// the resetting connection only after ~270 ms, and connections opened in that time never got an answer
    /// (6 of 6 resumed downloads stalled; 0 of 4 once later connections waited for the first answer).
    fn busy(&self) -> bool {
        self.gate.is_some()
    }

    fn host(&self) -> &str {
        "127.0.0.1"
    }

    fn stop(&mut self) {
        self.stop.store(true, Relaxed);
        if let Some(h) = self.rd.take() {
            let _ = h.join();
        }
    }
}
