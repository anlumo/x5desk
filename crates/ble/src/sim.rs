//! A simulated adapter for tests of the code above the crate (feature `sim`): [`Ble::sim`] scans (reporting the
//! advertisements it was given, folded as a real scan folds them), finds and connects like the real one, and a
//! connection runs the real session ([`Sess`]: framing, reassembly, waits, timeouts) against a closure that answers
//! requests; notifications come from [`Sim::note`]. Nothing reaches the radio, except that [`Ble::wake_up`] still
//! starts the real wake advertisement before it scans.

use crate::{
    Adv, Ble, Camera, Dev, Error, Event, Link, Result, Uuid,
    io::{self, Cmd},
    sess::{Op, Sess},
    wire,
};
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
    time::{Duration, Instant},
};
use tokio::{
    sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
    time,
};

/// Request handler: method and body in, status and body out; it may send notifications through the [`Sim`].
pub type Req = Box<dyn FnMut(u16, &[u8], &Sim) -> (u16, Vec<u8>) + Send>;

enum S {
    Note(u16, Vec<u8>),
    Lose,
    Mute(bool),
}

/// Control of a simulated adapter's camera link.
#[derive(Clone)]
pub struct Sim(UnboundedSender<S>);

impl Sim {
    /// The connected camera sends notification `code` (after the response the handler is building, if any).
    pub fn note(&self, code: u16, body: &[u8]) {
        let _ = self.0.send(S::Note(code, body.into()));
    }

    /// The link drops: [`Event::Closed`], pending calls end with [`Error::Closed`].
    pub fn lose(&self) {
        let _ = self.0.send(S::Lose);
    }

    /// From the next connect on, the camera behaves as an X5 attached by USB in app mode: it takes every write and
    /// sends nothing, no first notification, no answer.
    pub fn mute(&self, on: bool) {
        let _ = self.0.send(S::Mute(on));
    }
}

/// How often a scan reports each advertisement.
const EVERY: Duration = Duration::from_millis(200);

impl Ble {
    /// A simulated adapter that sees `advs`; connecting to one of them opens a session answered by `req` (one link
    /// at a time: a new connect drops the old one).
    pub fn sim(
        advs: Vec<Adv>,
        req: impl FnMut(u16, &[u8], &Sim) -> (u16, Vec<u8>) + Send + 'static,
    ) -> Result<(Ble, Receiver<Event>, Sim)> {
        let (tx, cmds) = unbounded_channel();
        let (etx, erx) = mpsc::channel();
        let (stx, srx) = unbounded_channel();
        let sim = Sim(stx);
        let s = sim.clone();
        let req: Req = Box::new(req);
        thread::Builder::new()
            .name("x5-ble-sim".into())
            .spawn(move || run(advs, req, s, cmds, srx, etx))?;
        Ok((Ble(tx), erx, sim))
    }
}

fn run(
    advs: Vec<Adv>,
    mut req: Req,
    sim: Sim,
    mut cmds: UnboundedReceiver<Cmd>,
    mut srx: UnboundedReceiver<S>,
    ev: mpsc::Sender<Event>,
) {
    let Ok(rt) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return;
    };
    // a camera handle dropped here (a connect nobody waited for) must not wait for this thread
    io::BLE.set(true);
    rt.block_on(async move {
        let mut ids = io::Ids::new();
        let (mut scan, mut mute) = (false, false);
        let mut finds: Vec<(String, Instant, SyncSender<Result<Adv>>)> = vec![];
        let mut link: Option<(Sess, UnboundedReceiver<Op>, Dev)> = None;
        let mut tick = time::interval(EVERY);
        loop {
            let next = link
                .as_mut()
                .and_then(|l| l.0.expire(Instant::now()))
                .map(time::Instant::from_std);
            // in the order things were sent: a control message before a later command, a note after its response
            tokio::select! {
                biased;
                m = srx.recv() => match (m, &mut link) {
                    (Some(S::Note(c, b)), Some((s, ..))) => {
                        let mut h = [0; 9];
                        h[..2].copy_from_slice(&c.to_le_bytes());
                        h[2] = 2;
                        // end | camera→phone
                        h[3..7].copy_from_slice(&(3u32 << 30).to_le_bytes());
                        let mut v = vec![];
                        wire::put(&mut v, 4, &h, &b);
                        let _ = s.feed(&v);
                    }
                    (Some(S::Mute(on)), _) => mute = on,
                    (Some(S::Lose), Some(_)) => {
                        let (mut s, _, d) = link.take().unwrap();
                        s.fail();
                        let _ = ev.send(Event::Closed(d, Error::Closed));
                    }
                    _ => {}
                },
                c = cmds.recv() => match c {
                    None => break,
                    Some(Cmd::Scan(on, r)) => {
                        scan = on;
                        let _ = r.send(Ok(()));
                    }
                    Some(Cmd::Find(s, end, r)) => finds.push((s, end, r)),
                    Some(Cmd::Connect(d, r)) => {
                        if !advs.iter().any(|a| a.dev == d) {
                            let _ = r.send(Err(Error::NoDevice));
                            continue;
                        }
                        if let Some((mut s, ..)) = link.take() {
                            s.fail()
                        }
                        let (otx, orx) = unbounded_channel();
                        let mut s = Sess::new(d.clone(), ev.clone());
                        // the X5's first notification, a heartbeat: plain packets
                        let hb = [7, 0, 0, 0, 5, 0, 0];
                        if !mute {
                            let _ = s.feed(&hb);
                        }
                        let u = |x: u128| Uuid::from_u128(x << 96 | 0x0000_1000_8000_0080_5f9b_34fb);
                        let l = Link {
                            service: u(0xbe80),
                            tx: u(0xbe81),
                            rx: u(0xbe82),
                            mtu: 517,
                            wrap: false,
                            first: (!mute).then(|| (Duration::from_millis(30), hb.into())),
                        };
                        link = Some((s, orx, d.clone()));
                        let _ = r.send(Ok(Camera::new(otx, d, l)));
                    }
                    #[allow(unreachable_patterns)]
                    Some(_) => {}
                },
                o = async {
                    match &mut link {
                        Some(l) => l.1.recv().await,
                        None => std::future::pending().await,
                    }
                } => {
                    let now = Instant::now();
                    match o {
                        Some(Op::Req(c, b, t, d)) => {
                            let s = &mut link.as_mut().unwrap().0;
                            if mute {
                                s.op(Op::Req(c, b, t, d), now);
                                s.out.clear();
                                continue;
                            }
                            let (st, body) = req(c, &b, &sim);
                            let s = &mut link.as_mut().unwrap().0;
                            s.op(Op::Req(c, b, t, d), now);
                            s.out.clear();
                            let mut v = vec![];
                            wire::msg(&mut v, st, s.id, true, &body);
                            let _ = s.feed(&v);
                        }
                        Some(o @ Op::Note(..)) => link.as_mut().unwrap().0.op(o, now),
                        Some(Op::Close(r)) => {
                            if let Some((mut s, ..)) = link.take() {
                                s.fail()
                            }
                            if let Some(r) = r {
                                let _ = r.send(Ok(()));
                            }
                        }
                        None => {
                            if let Some((mut s, ..)) = link.take() {
                                s.fail()
                            }
                        }
                    }
                }
                _ = tick.tick(), if scan || !finds.is_empty() => {
                    let now = Instant::now();
                    for a in &advs {
                        let mut a = a.clone();
                        io::fold(&mut ids, &mut a);
                        finds.retain(|f| {
                            let hit = a.is(&f.0);
                            if hit {
                                let _ = f.2.send(Ok(a.clone()));
                            }
                            !hit
                        });
                        if scan {
                            let _ = ev.send(Event::Seen(a));
                        }
                    }
                    finds.retain(|f| {
                        if f.1 <= now {
                            let _ = f.2.send(Err(Error::Timeout));
                        }
                        f.1 > now
                    });
                }
                _ = async {
                    match next {
                        Some(t) => time::sleep_until(t).await,
                        None => std::future::pending().await,
                    }
                } => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{code, msg, pb};
    use prost::Message;

    /// Scan, find, connect, a request, authorization through the camera's answer (8209), station mode (8232), a
    /// timeout, a lost link: the real session code on the simulated link.
    #[test]
    fn session() {
        let mut a = Adv::test("X5 SIM001");
        a.dev = crate::Dev("sim/X5".into());
        let (b, ev, sim) = Ble::sim(vec![a], |c, q, s| match c {
            code::READ_SETTINGS => {
                let o = pb::Settings {
                    wifi: Some(pb::settings::Wifi {
                        ssid: "X5 SIM001.OSC".into(),
                        password: "88888888".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                };
                let r = pb::ReadSettingsReply {
                    keys: vec![],
                    settings: Some(o),
                };
                (200, r.encode_to_vec())
            }
            code::AUTHORIZE => {
                // not authorized (1), then the user allows it on the camera
                s.note(code::AUTH_NOTE, &[8, 0, 16, 0]);
                let r = pb::AuthorizeReply { status: 1 };
                (200, r.encode_to_vec())
            }
            code::JOIN_STATION => {
                let i = msg::JoinStation::decode(q).unwrap().station;
                let r = msg::StationNote {
                    result: 0,
                    station: i.map(|i| msg::Station {
                        ip: "192.0.2.7".into(),
                        password: String::new(),
                        ..i
                    }),
                };
                s.note(code::STATION_NOTE, &r.encode_to_vec());
                (200, vec![])
            }
            code::READ_WIFI_MODE => (500, vec![]),
            _ => (200, vec![]),
        })
        .unwrap();
        b.scan(true).wait().unwrap();
        let seen = loop {
            if let Event::Seen(a) = ev.recv_timeout(Duration::from_secs(2)).unwrap() {
                break a;
            }
        };
        assert!(seen.x5 && seen.token() == Some("SIM001"));
        let f = b
            .find("IAH0000SIM001", Duration::from_secs(2))
            .wait()
            .unwrap();
        assert_eq!(f.dev, seen.dev);
        assert!(matches!(
            b.connect(&crate::Dev("other".into())).wait(),
            Err(Error::NoDevice)
        ));
        let c = b.connect(&seen.dev).wait().unwrap();
        assert!(c.link().first.is_some() && !c.link().wrap);
        assert_eq!(c.wifi().wait().unwrap().password, "88888888");
        c.authorize("uuid", Duration::from_secs(2)).wait().unwrap();
        let r = c
            .station("Home", "secret", Duration::from_secs(2))
            .wait()
            .unwrap();
        let i = r.station.unwrap();
        assert_eq!((r.result, &*i.ssid, &*i.ip), (0, "Home", "192.0.2.7"));
        assert!(matches!(
            c.wifi_mode().wait(),
            Err(Error::Status(500, None))
        ));
        assert!(matches!(
            c.note(code::WIFI_UP, Duration::from_millis(50), true)
                .wait(),
            Err(Error::Timeout)
        ));
        let w = c.note(code::WIFI_UP, Duration::from_secs(5), true);
        sim.lose();
        assert!(matches!(w.wait(), Err(Error::Closed)));
        let closed = loop {
            if let Event::Closed(d, _) = ev.recv_timeout(Duration::from_secs(2)).unwrap() {
                break d;
            }
        };
        assert_eq!(closed, seen.dev);
        let t = Instant::now();
        drop(c);
        assert!(t.elapsed() < Duration::from_secs(1));
        // attached by USB: connected, but nothing comes back
        sim.mute(true);
        let c = b.connect(&seen.dev).wait().unwrap();
        assert!(c.link().first.is_none());
        let r = c.raw(code::READ_SETTINGS, vec![], Duration::from_millis(100));
        assert!(matches!(r.wait(), Err(Error::Timeout)));
    }
}
