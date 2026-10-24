//! The UCD session on a GATT pipe, without I/O: frames requests, picks the framing from the camera's first
//! notification, reassembles responses and notifications, and times out what the camera never answers.

use crate::{Dev, Done, Error, Event, Note, Result, pb, wire};
use prost::Message;
use std::{
    sync::mpsc::{Sender, SyncSender},
    time::{Duration, Instant},
};

pub(crate) enum Op {
    Req(u16, Vec<u8>, Duration, Done),
    /// The next notification with this code within the time. `fresh`: one that arrived earlier and was not taken
    /// does not count.
    Note(u16, Duration, bool, Done),
    Close(Option<SyncSender<Result<()>>>),
}

pub(crate) struct Sess {
    /// UCP framing; `None` until the first notification (or its timeout) decides.
    pub ucp: Option<bool>,
    /// Frames to write, in order.
    pub out: Vec<Vec<u8>>,
    pub id: u32,
    dev: Dev,
    ev: Sender<Event>,
    /// UCP bytes not yet unframed.
    raw: Vec<u8>,
    rx: Vec<u8>,
    /// `(id, deadline, done)`.
    reqs: Vec<(u32, Instant, Done)>,
    /// `(code, deadline, done)`.
    waits: Vec<(u32, Instant, Done)>,
    /// Newest body per notification code that no waiter took.
    last: Vec<(u16, Vec<u8>)>,
    /// Incomplete messages by `(notification, id)` (§4.6), oldest first.
    frags: Vec<((bool, u32), Vec<u8>)>,
}

impl Sess {
    pub fn new(dev: Dev, ev: Sender<Event>) -> Sess {
        Sess {
            ucp: None,
            out: vec![],
            id: 0,
            dev,
            ev,
            raw: vec![],
            rx: vec![],
            reqs: vec![],
            waits: vec![],
            last: vec![],
            frags: vec![],
        }
    }

    pub fn op(&mut self, o: Op, now: Instant) {
        match o {
            Op::Req(c, b, t, d) => {
                // First id 1; after 2^30 − 1 it wraps to 0 (§4.6).
                self.id = if self.id >= 0x3FFF_FFFF {
                    0
                } else {
                    self.id + 1
                };
                wire::frames(&mut self.out, c, self.id, &b, self.ucp == Some(true));
                self.reqs.push((self.id, now + t, d))
            }
            Op::Note(c, t, fresh, d) => {
                let i = self.last.iter().position(|x| x.0 == c);
                match i.map(|i| self.last.swap_remove(i)) {
                    Some((_, b)) if !fresh => d(Ok(b)),
                    _ => self.waits.push((c as u32, now + t, d)),
                }
            }
            Op::Close(_) => {}
        }
    }

    /// Bytes of one notification of the camera's characteristic.
    pub fn feed(&mut self, b: &[u8]) -> Result<()> {
        if !*self
            .ucp
            .get_or_insert_with(|| b.starts_with(&[0xFF, 6, 0x41]))
        {
            self.rx.extend_from_slice(b);
            return self.parse();
        }
        self.raw.extend_from_slice(b);
        let mut i = 0;
        loop {
            match wire::unframe(&self.raw[i..]) {
                Ok(None) => break,
                // Resync on the next frame start.
                Err(()) => i += 1,
                Ok(Some((d, t, p))) => {
                    let n = p.len() + 7;
                    match (d, t) {
                        (6, 0x41) => {
                            let mut v = vec![];
                            wire::sync(&mut v);
                            self.out.push(v)
                        }
                        // Wake-auth, busy, classic-BT and AI-voice replies: not used with an X5.
                        (0x0D, _) => {}
                        _ => self.rx.extend_from_slice(p),
                    }
                    i += n
                }
            }
        }
        self.raw.drain(..i);
        self.parse()
    }

    fn parse(&mut self) -> Result<()> {
        let mut i = 0;
        while let Some((p, n)) = wire::next(&self.rx[i..]).map_err(|_| Error::Desync)? {
            i += n;
            let wire::Pkt::Msg {
                code,
                note,
                id,
                end,
                body,
            } = p
            else {
                continue;
            };
            // Method and id come from the last fragment (§4.8).
            let j = self.frags.iter().position(|x| x.0 == (note, id));
            let b = match (j, end) {
                (None, true) => body.to_vec(),
                (Some(j), true) => {
                    let mut v = self.frags.remove(j).1;
                    v.extend_from_slice(body);
                    v
                }
                (Some(j), false) => {
                    self.frags[j].1.extend_from_slice(body);
                    continue;
                }
                (None, false) => {
                    if self.frags.len() >= 256 {
                        self.frags.remove(0);
                    }
                    self.frags.push(((note, id), body.to_vec()));
                    continue;
                }
            };
            if note {
                self.note(code, b)
            } else if let Some(j) = self.reqs.iter().position(|r| r.0 == id) {
                let d = self.reqs.swap_remove(j).2;
                d(match code {
                    200 => Ok(b),
                    s => Err(Error::Status(
                        s,
                        (!b.is_empty())
                            .then(|| pb::Error::decode(&b[..]).ok())
                            .flatten(),
                    )),
                })
            }
        }
        self.rx.drain(..i);
        Ok(())
    }

    fn note(&mut self, c: u16, b: Vec<u8>) {
        let mut hit = false;
        let mut j = 0;
        while j < self.waits.len() {
            if self.waits[j].0 == c as u32 {
                (self.waits.swap_remove(j).2)(Ok(b.clone()));
                hit = true
            } else {
                j += 1
            }
        }
        if !hit {
            match self.last.iter_mut().find(|x| x.0 == c) {
                Some(x) => x.1.clone_from(&b),
                None => self.last.push((c, b.clone())),
            }
        }
        let _ = self
            .ev
            .send(Event::Note(self.dev.clone(), Note::new(c, &b)));
    }

    /// Fails what timed out; the next deadline.
    pub fn expire(&mut self, now: Instant) -> Option<Instant> {
        let mut next: Option<Instant> = None;
        for v in [&mut self.reqs, &mut self.waits] {
            let mut j = 0;
            while j < v.len() {
                if v[j].1 <= now {
                    (v.swap_remove(j).2)(Err(Error::Timeout))
                } else {
                    next = Some(next.map_or(v[j].1, |n| n.min(v[j].1)));
                    j += 1
                }
            }
        }
        next
    }

    /// Ends everything still waiting with [`Error::Closed`].
    pub fn fail(&mut self) {
        for (_, _, d) in self.reqs.drain(..).chain(self.waits.drain(..)) {
            d(Err(Error::Closed))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Camera, Link, Pending, Step, code, msg, wire::hex};
    use std::{
        sync::mpsc::{self, Receiver},
        thread,
    };
    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    const UUID: &str = "01234567-89ab-cdef-0123-456789abcdef";

    /// `ops` first: it drops before `c`, whose drop then finds no link task to wait for.
    struct Sim {
        ops: UnboundedReceiver<Op>,
        c: Camera,
        s: Sess,
        ev: Receiver<Event>,
    }

    fn sim(ucp: Option<bool>) -> Sim {
        let (tx, ops) = unbounded_channel();
        let (etx, ev) = mpsc::channel();
        let dev = Dev::test();
        let c = Camera::new(tx, dev.clone(), Link::default());
        let mut s = Sess::new(dev, etx);
        s.ucp = ucp;
        Sim { ops, c, s, ev }
    }

    impl Sim {
        /// Hands every queued op to the session.
        fn pump(&mut self) {
            while let Ok(o) = self.ops.try_recv() {
                self.s.op(o, Instant::now())
            }
        }

        /// Waits for the next op (a flow on its helper thread) and hands it over.
        fn step(&mut self) {
            let o = self.ops.blocking_recv().unwrap();
            self.s.op(o, Instant::now())
        }

        fn out(&mut self) -> Vec<Vec<u8>> {
            std::mem::take(&mut self.s.out)
        }

        /// A response to request `id`: status 200 and a protobuf body.
        fn ok(&mut self, id: u32, m: &impl Message) {
            let mut v = vec![];
            wire::put(&mut v, 4, &resp(200, id), &m.encode_to_vec());
            self.s.feed(&v).unwrap()
        }

        fn notify(&mut self, code: u16, body: &[u8]) {
            let mut h = resp(code, 0x77);
            h[6] |= 0x40;
            let mut v = vec![];
            wire::put(&mut v, 4, &h, body);
            self.s.feed(&v).unwrap()
        }
    }

    fn resp(code: u16, id: u32) -> [u8; 9] {
        let mut h = [0; 9];
        h[..2].copy_from_slice(&code.to_le_bytes());
        h[2] = 2;
        h[3..7].copy_from_slice(&(1 << 31 | id).to_le_bytes());
        h
    }

    fn wifi() -> pb::options::WifiInfo {
        pb::options::WifiInfo {
            ssid: "X5 ABCDEF.OSC".into(),
            password: "12345678".into(),
            channel: 149,
            wifi_state: 2,
            ..Default::default()
        }
    }

    fn opts(w: pb::options::WifiInfo) -> pb::GetOptionsResp {
        pb::GetOptionsResp {
            option_types: vec![36, 37, 43],
            value: Some(pb::Options {
                wifi_info: Some(w),
                ..Default::default()
            }),
        }
    }

    /// Every request of ble.md §10 and PROTOCOL.md §1.4 / §2.6 / §2.8 / §5.2 through the typed API, byte for byte.
    #[test]
    fn requests() {
        let mut m = sim(Some(false));
        m.s.id = 4;
        let w = m.c.wifi();
        let a = m.c.check_auth(UUID);
        let g = m.c.wifi_mode();
        let s = m.c.set_wifi_mode(msg::AP, "");
        let i = m.c.wifi_conn();
        let o = m.c.open_wifi(0);
        m.pump();
        let auth = "38 00 00 00 04 00 00 27 00 02 06 00 00 80 00 00 0a 24 30 31 32 33 34 35 36 37 2d 38 39 61 62 \
                    2d 63 64 65 66 2d 30 31 32 33 2d 34 35 36 37 38 39 61 62 63 64 65 66 10 02";
        assert_eq!(
            m.out(),
            [
                hex("16 00 00 00 04 00 00 08 00 02 05 00 00 80 00 00 08 24 08 25 08 2b"),
                hex(auth),
                hex("10 00 00 00 04 00 00 96 00 02 07 00 00 80 00 00"),
                hex("12 00 00 00 04 00 00 93 00 02 08 00 00 80 00 00 08 01"),
                hex("10 00 00 00 04 00 00 71 00 02 09 00 00 80 00 00"),
                hex("10 00 00 00 04 00 00 21 00 02 0a 00 00 80 00 00"),
            ]
        );
        // Answers, out of order.
        m.ok(10, &());
        m.ok(9, &msg::GetWifiConnectionInfoResp::default());
        m.ok(8, &());
        m.ok(
            7,
            &msg::GetWifiModeResp {
                wifi_mode: msg::STA,
            },
        );
        m.ok(6, &pb::CheckAuthorizationResp::default());
        // Response header from ble.md §10 (status 200, id 5) with a GetOptionsResp body.
        let mut v = hex("00 00 00 00 04 00 00 c8 00 02 05 00 00 80 00 00");
        v.extend(opts(wifi()).encode_to_vec());
        v[0] = v.len() as u8;
        // Delivered one byte per notification: chunk boundaries carry no meaning (ble.md §4).
        for b in v.chunks(1) {
            m.s.feed(b).unwrap()
        }
        assert_eq!(w.wait().unwrap(), wifi());
        // AUTHORIZED is the proto3 default: an empty body (PROTOCOL.md §5.2).
        assert_eq!(a.wait().unwrap().authorization_status, 0);
        assert_eq!(g.wait().unwrap(), msg::STA);
        s.wait().unwrap();
        i.wait().unwrap();
        o.wait().unwrap();

        // Ids are taken when the session handles the request.
        let mut m = sim(Some(false));
        let on = pb::Options {
            bt_wakeup_sw: 1,
            ..Default::default()
        };
        for (id, f) in [
            (
                6,
                Box::new(|c: &Camera| drop(c.set_wifi_mode(msg::AP, "AT"))) as Box<dyn Fn(&Camera)>,
            ),
            (0, Box::new(|c| drop(c.options(&[48, 15, 11, 198])))),
            (5, Box::new(move |c| drop(c.set_options(&[97], on.clone())))),
            (3, Box::new(|c| drop(c.check_auth(UUID)))),
            (1, Box::new(|c| drop(c.wifi()))),
        ] {
            m.s.id = id;
            f(&m.c);
            m.pump();
        }
        let mut auth4 = hex(auth);
        auth4[10] = 4;
        assert_eq!(
            m.out(),
            [
                hex("16 00 00 00 04 00 00 93 00 02 07 00 00 80 00 00 08 01 1a 02 41 54"),
                hex("19 00 00 00 04 00 00 08 00 02 01 00 00 80 00 00 08 30 08 0f 08 0b 08 c6 01"),
                hex("17 00 00 00 04 00 00 07 00 02 06 00 00 80 00 00 08 61 12 03 88 06 01"),
                auth4,
                hex("16 00 00 00 04 00 00 08 00 02 02 00 00 80 00 00 08 24 08 25 08 2b"),
            ]
        );
    }

    #[test]
    fn errors_and_notes() {
        let mut m = sim(None);
        let p = m.c.raw(code::GET_OPTIONS, vec![], Duration::from_secs(5));
        m.pump();
        // The first notification is plain UCD, so the session stays in UCD framing. Error response with
        // `Error{EXECUTE_ERROR}` (PROTOCOL.md §4.10), then 8209 REJECT and 8195 BATTERY_UPDATE (§4.10).
        let b = hex("12 00 00 00 04 00 00 f4 01 02 01 00 00 80 00 00 08 03 \
             12 00 00 00 04 00 00 11 20 02 05 00 00 c0 00 00 08 01 \
             16 00 00 00 04 00 00 03 20 02 05 00 00 c0 00 00 0a 04 10 50 18 64");
        m.s.feed(&b).unwrap();
        assert_eq!(m.s.ucp, Some(false));
        match p.wait() {
            Err(Error::Status(500, Some(e))) => assert_eq!(e.code, 3),
            r => panic!("{r:?}"),
        }
        let Event::Note(_, Note::Auth(a)) = m.ev.try_recv().unwrap() else {
            panic!()
        };
        assert_eq!(
            (a.authorization_result, a.authorization_operation_type),
            (1, 0)
        );
        let Event::Note(_, Note::Other(8195, body)) = m.ev.try_recv().unwrap() else {
            panic!()
        };
        assert_eq!(body, hex("0a 04 10 50 18 64"));
        // A fragmented notification (`end` 0, then 1) and a desync.
        m.s.feed(&hex(
            "12 00 00 00 04 00 00 11 20 02 09 00 00 40 00 00 08 01",
        ))
        .unwrap();
        assert!(m.ev.try_recv().is_err());
        m.s.feed(&hex(
            "12 00 00 00 04 00 00 11 20 02 09 00 00 c0 00 00 10 02",
        ))
        .unwrap();
        let Event::Note(_, Note::Auth(a)) = m.ev.try_recv().unwrap() else {
            panic!()
        };
        assert_eq!(
            (a.authorization_result, a.authorization_operation_type),
            (1, 2)
        );
        assert!(matches!(
            m.s.feed(&hex("03 00 00 00 04 00 00")),
            Err(Error::Desync)
        ));

        // Latched notifications: one that arrived earlier counts unless `fresh`.
        let mut m = sim(Some(false));
        m.notify(code::WIFI_START, &[]);
        assert!(matches!(
            m.ev.try_recv(),
            Ok(Event::Note(_, Note::WifiStarted))
        ));
        let f = m.c.note(code::WIFI_START, Duration::from_secs(5), true);
        m.pump();
        assert!(f.poll().is_none());
        let l = m.c.note(code::WIFI_START, Duration::from_secs(5), false);
        m.pump();
        assert!(l.poll().is_none());
        m.notify(code::WIFI_START, &[]);
        assert_eq!(f.wait().unwrap(), b"");
        assert_eq!(l.wait().unwrap(), b"");
        // Timeouts and close.
        let t = m.c.note(code::AUTH_RESULT, Duration::ZERO, false);
        let r = m.c.raw(code::GET_WIFI_MODE, vec![], Duration::from_secs(5));
        m.pump();
        assert!(m.s.expire(Instant::now()).is_some());
        assert!(matches!(t.wait(), Err(Error::Timeout)));
        m.s.fail();
        assert!(matches!(r.wait(), Err(Error::Closed)));
    }

    #[test]
    fn ucp() {
        let mut m = sim(None);
        // Camera sync (`ff 06 41`, payload made up) → UCP mode and the legacy sync reply.
        let mut f = vec![];
        wire::frame(&mut f, 6, 0x41, &hex("ab ba ab ba"));
        m.s.feed(&f[..3]).unwrap();
        m.s.feed(&f[3..]).unwrap();
        assert_eq!(m.s.ucp, Some(true));
        assert_eq!(
            m.out(),
            [hex("ff 07 41 0a 00 ab ba ab ba ab ba ab ba ab ba 4e 23")]
        );
        // Requests go out wrapped.
        let p = m.c.wifi_mode();
        m.pump();
        let o = m.out();
        assert_eq!(
            o,
            [hex(
                "ff 07 40 10 00 10 00 00 00 04 00 00 96 00 02 01 00 00 80 00 00 fc fa"
            )]
        );
        // The camera's frames carry UCD packets; a corrupt byte in front is skipped.
        let mut v = vec![];
        wire::put(
            &mut v,
            4,
            &resp(200, 1),
            &msg::GetWifiModeResp { wifi_mode: msg::AP }.encode_to_vec(),
        );
        let mut f = vec![0x55];
        wire::frame(&mut f, 6, 0x40, &v);
        m.s.feed(&f).unwrap();
        assert_eq!(p.wait().unwrap(), msg::AP);
    }

    #[test]
    fn authorize() {
        let mut m = sim(Some(false));
        let a = m.c.authorize(UUID, Duration::from_secs(5));
        m.step(); // waiter for 8209, registered before the request goes out
        m.step();
        let o = m.out();
        assert_eq!(o.len(), 1);
        assert_eq!(&o[0][7..9], &code::CHECK_AUTH.to_le_bytes());
        m.ok(
            1,
            &pb::CheckAuthorizationResp {
                authorization_status: 1,
                ..Default::default()
            },
        );
        // PENDING keeps waiting; the success the camera sends may be an empty body (PROTOCOL.md §5.2).
        m.notify(code::AUTH_RESULT, &hex("08 04"));
        m.notify(code::AUTH_RESULT, &[]);
        m.step();
        a.wait().unwrap();

        // Rejected on the camera.
        let mut m = sim(Some(false));
        let a = m.c.authorize(UUID, Duration::from_secs(5));
        m.step();
        m.step();
        m.ok(
            1,
            &pb::CheckAuthorizationResp {
                authorization_status: 1,
                ..Default::default()
            },
        );
        m.notify(code::AUTH_RESULT, &hex("08 01"));
        assert!(matches!(a.wait(), Err(Error::Refused(1))));

        // Another phone holds the camera.
        let mut m = sim(Some(false));
        let a = m.c.authorize(UUID, Duration::from_secs(5));
        m.step();
        m.step();
        m.ok(
            1,
            &pb::CheckAuthorizationResp {
                authorization_status: 3,
                ..Default::default()
            },
        );
        assert!(matches!(a.wait(), Err(Error::Unauthorized(3))));
    }

    #[test]
    fn ap() {
        // Camera in STA mode: wait for 8215, GET_WIFI_MODE, SET_WIFI_MODE(AP), wait for a new 8215, read the AP.
        let mut m = sim(Some(false));
        m.notify(code::WIFI_START, &[]);
        let a = m.c.ap(Duration::from_secs(5));
        m.step(); // takes the 8215 that came first
        m.step();
        assert_eq!(
            m.out(),
            [hex("10 00 00 00 04 00 00 96 00 02 01 00 00 80 00 00")]
        );
        m.ok(
            1,
            &msg::GetWifiModeResp {
                wifi_mode: msg::STA,
            },
        );
        m.step(); // fresh waiter for 8215
        m.step();
        assert_eq!(
            m.out(),
            [hex("12 00 00 00 04 00 00 93 00 02 02 00 00 80 00 00 08 01")]
        );
        m.ok(2, &());
        m.notify(code::WIFI_START, &[]);
        m.step();
        assert_eq!(
            m.out(),
            [hex(
                "16 00 00 00 04 00 00 08 00 02 03 00 00 80 00 00 08 24 08 25 08 2b"
            )]
        );
        m.ok(3, &opts(wifi()));
        assert_eq!(a.wait().unwrap(), wifi());
    }

    #[test]
    fn station() {
        let mut m = sim(Some(false));
        let r = m.c.station("home", "secret", Duration::from_secs(5));
        m.step(); // waiter for 8232
        m.step();
        // SET_WIFI_MODE{1, STA}, PHONE_INFO, SET_WIFI_CONNECTION_INFO (PROTOCOL.md §2.9).
        assert_eq!(
            m.out(),
            [hex(
                "14 00 00 00 04 00 00 93 00 02 01 00 00 80 00 00 08 01 10 01"
            )]
        );
        m.ok(1, &());
        m.step();
        let o = m.out();
        assert_eq!(&o[0][7..9], &code::PHONE_INFO.to_le_bytes());
        m.ok(2, &());
        m.step();
        assert_eq!(
            m.out(),
            [hex(
                "20 00 00 00 04 00 00 70 00 02 03 00 00 80 00 00 0a 0e 0a 04 68 6f 6d 65 1a 06 73 65 63 72 65 74"
            )]
        );
        m.ok(3, &());
        let res = msg::CameraWifiConnectionResult {
            wifi_connection_result: 5,
            wifi_connection_info: Some(msg::WifiConnectionInfo {
                ip_addr: "10.0.0.7".into(),
                ..Default::default()
            }),
        };
        m.notify(code::WIFI_STATUS, &res.encode_to_vec());
        assert_eq!(r.wait().unwrap(), res);
    }

    /// The wake-up's session (`crate::session`) on its helper thread, its steps on the receiver.
    fn wake(m: &Sim, t: Duration) -> (Pending<()>, Receiver<Step>) {
        let (tx, rx) = mpsc::channel();
        let p =
            m.c.flow(move |c| crate::session(c, UUID, t, &mut |s| drop(tx.send(s))));
        (p, rx)
    }

    fn body(f: &[u8]) -> &[u8] {
        &f[16..]
    }

    #[test]
    fn wake_up() {
        // An asleep camera (CHECK_AUTHORIZATION 7) that wakes on the second try; its 8215 came early.
        let mut m = sim(Some(false));
        m.notify(code::WIFI_START, &[]);
        let (p, steps) = wake(&m, Duration::from_secs(5));
        // CHECK_TYPE, byte for byte as PROTOCOL.md §1.4.
        m.step();
        assert_eq!(
            m.out(),
            [hex(
                "19 00 00 00 04 00 00 08 00 02 01 00 00 80 00 00 08 30 08 0f 08 0b 08 c6 01"
            )]
        );
        let id = pb::Options {
            camera_type: "Insta360 X5".into(),
            ..Default::default()
        };
        m.ok(
            1,
            &pb::GetOptionsResp {
                option_types: vec![48],
                value: Some(id.clone()),
            },
        );
        // SYNCING_OPTIONS: the BLE option list alone; the app's GET_CURRENT_CAPTURE_STATUS stays local for an X5.
        m.step();
        let o = m.out();
        assert_eq!(o.len(), 1);
        assert_eq!(&o[0][7..9], &code::GET_OPTIONS.to_le_bytes());
        let q = pb::GetOptions::decode(body(&o[0])).unwrap();
        assert_eq!(q.option_types, crate::SYNC);
        let on = pb::Options {
            bt_wakeup_sw: 1,
            ..Default::default()
        };
        m.ok(
            2,
            &pb::GetOptionsResp {
                option_types: vec![97],
                value: Some(on.clone()),
            },
        );
        // CHECK_AUTHORIZATION, id 3 straight after the sync: asleep, asked again after 2 s, then authorized (empty
        // body).
        let mut auth = hex(
            "38 00 00 00 04 00 00 27 00 02 03 00 00 80 00 00 0a 24 30 31 32 33 34 35 36 37 2d 38 39 61 62 \
             2d 63 64 65 66 2d 30 31 32 33 2d 34 35 36 37 38 39 61 62 63 64 65 66 10 02",
        );
        m.step(); // waiter for 8209
        m.step();
        assert_eq!(m.out(), [auth.clone()]);
        m.ok(
            3,
            &pb::CheckAuthorizationResp {
                authorization_status: 7,
                ..Default::default()
            },
        );
        let t0 = Instant::now();
        m.step();
        m.step();
        assert!(t0.elapsed() >= Duration::from_millis(1900));
        auth[10] = 4;
        assert_eq!(m.out(), [auth]);
        m.ok(4, &());
        // The 8215 from before counts.
        m.step();
        p.wait().unwrap();
        let s: Vec<_> = steps.try_iter().collect();
        assert!(
            matches!(
                &s[..],
                [
                    Step::Options(a),
                    Step::Synced(Ok(b)),
                    Step::Auth(7),
                    Step::Auth(0),
                    Step::Ready
                ] if *a == id && *b == on
            ),
            "{s:?}"
        );

        // No 8215 within the time: OPEN_CAMERA_WIFI, then a fresh 8215.
        let mut m = sim(Some(false));
        m.s.id = 10;
        let (p, steps) = wake(&m, Duration::from_millis(50));
        m.step();
        m.ok(11, &pb::GetOptionsResp::default());
        m.step();
        m.ok(12, &pb::GetOptionsResp::default());
        m.step();
        m.step();
        m.ok(13, &());
        m.step(); // waiter for 8215
        m.out();
        thread::sleep(Duration::from_millis(60));
        m.s.expire(Instant::now());
        m.step(); // fresh waiter for 8215
        m.step();
        assert_eq!(
            m.out(),
            [hex("10 00 00 00 04 00 00 21 00 02 0e 00 00 80 00 00")]
        );
        m.ok(14, &());
        m.notify(code::WIFI_START, &[]);
        p.wait().unwrap();
        let s: Vec<_> = steps.try_iter().collect();
        assert!(
            matches!(&s[2..], [Step::Auth(0), Step::OpenWifi, Step::Ready]),
            "{s:?}"
        );

        // Held by another phone: no retry.
        let mut m = sim(Some(false));
        let (p, _steps) = wake(&m, Duration::from_secs(5));
        m.step();
        m.ok(1, &pb::GetOptionsResp::default());
        m.step();
        m.ok(2, &pb::GetOptionsResp::default());
        m.step();
        m.step();
        m.ok(
            3,
            &pb::CheckAuthorizationResp {
                authorization_status: 3,
                ..Default::default()
            },
        );
        assert!(matches!(p.wait(), Err(Error::Unauthorized(3))));
    }
}
