//! The Wireless pane's worker against ble's simulated adapter: the real session code under the flows, every message
//! the camera gets checked. Nothing reaches the radio (the wake-up's own advertisement would: its steps are checked
//! as the worker reports them).

use super::*;
use std::sync::{Arc, Mutex};

type Log = Arc<Mutex<Vec<(u16, Vec<u8>)>>>;

/// A worker on the simulated X5 that logs what the camera gets; the simulation's control.
fn spawn(log: &Log) -> (Bt, Arc<Mutex<Option<ble::Sim>>>) {
    let (l, sim) = (log.clone(), Arc::new(Mutex::new(None)));
    let s2 = sim.clone();
    let open: Open = Box::new(move || {
        let l = l.clone();
        let (b, ev, s) = Ble::sim(vec![adv()], move |c, q, s| {
            l.lock().unwrap().push((c, q.to_vec()));
            answer(c, q, s)
        })?;
        *s2.lock().unwrap() = Some(s);
        Ok((b, ev))
    });
    (
        Bt::new(&egui::Context::default(), open, "client-id".into(), vec![]),
        sim,
    )
}

fn until(b: &mut Bt, secs: f64, f: impl Fn(&St) -> bool) {
    let end = Instant::now() + Duration::from_secs_f64(secs);
    loop {
        b.tick();
        if f(&b.st) {
            return;
        }
        assert!(Instant::now() < end, "stuck at {:#?}", b.st);
        thread::sleep(Duration::from_millis(5));
    }
}

fn codes(l: &Log) -> Vec<u16> {
    l.lock().unwrap().iter().map(|x| x.0).collect()
}

/// Scan, connect, pair (the camera's user allows it), the access point, station mode, disconnect; commands sent
/// while a flow runs wait for it.
#[test]
fn flows() {
    let log = Log::default();
    let (mut b, _) = spawn(&log);
    b.tx.send(Cmd::Scan(true)).unwrap();
    until(&mut b, 3.0, |s| !s.seen.is_empty());
    let x = b.st.seen[0].clone();
    assert_eq!(
        (&*x.token, x.rssi, x.proto.map(|p| p.wifi)),
        ("SIM001", Some(-58), Some(120))
    );
    b.tx.send(Cmd::Scan(false)).unwrap();
    b.tx.send(Cmd::Connect(x.dev.clone())).unwrap();
    // queued behind the connect
    b.tx.send(Cmd::Pair).unwrap();
    until(&mut b, 3.0, |s| s.conn == Conn::On);
    assert!(b.st.link.as_ref().unwrap().first.is_some());
    until(&mut b, 3.0, |s| {
        s.busy.is_some_and(|b| b.starts_with("Pairing"))
    });
    let t = Instant::now();
    until(&mut b, 5.0, |s| s.paired && s.busy.is_none());
    println!(
        "paired after {:.2} s (the simulated user allows after 1 s)",
        t.elapsed().as_secs_f64()
    );
    let (c, q) = log.lock().unwrap().last().cloned().unwrap();
    let a = pb::CheckAuthorization::decode(&q[..]).unwrap();
    assert_eq!(
        (c, &*a.id, a.initiator_type),
        (code::CHECK_AUTH, "client-id", 2)
    );
    b.tx.send(Cmd::Ap).unwrap();
    until(&mut b, 3.0, |s| s.ap.is_some());
    assert_eq!(b.st.ap, Some(("X5 SIM001.OSC".into(), "88888888".into())));
    let (c, q) = log.lock().unwrap().last().cloned().unwrap();
    assert_eq!(c, code::GET_OPTIONS);
    assert_eq!(
        pb::GetOptions::decode(&q[..]).unwrap().option_types,
        [36, 37, 43]
    );
    let n = log.lock().unwrap().len();
    b.tx.send(Cmd::Join("HomeNet".into(), "secret pw".into()))
        .unwrap();
    until(&mut b, 5.0, |s| s.sta.is_some());
    assert_eq!(b.st.sta, Some((0, "HomeNet".into(), "192.168.1.57".into())));
    assert_eq!(
        b.st.note,
        Some(("the camera joined HomeNet at 192.168.1.57".into(), false))
    );
    // PROTOCOL.md §2.9: STA mode, phone info, the network
    let l = log.lock().unwrap()[n..].to_vec();
    assert_eq!(
        l.iter().map(|x| x.0).collect::<Vec<_>>(),
        [code::SET_WIFI_MODE, code::PHONE_INFO, code::SET_WIFI_CONN]
    );
    assert_eq!(
        msg::SetWifiMode::decode(&l[0].1[..]).unwrap().wifi_mode,
        msg::STA
    );
    let i = msg::SetWifiConnectionInfo::decode(&l[2].1[..])
        .unwrap()
        .wifi_connection_info
        .unwrap();
    assert_eq!((&*i.ssid, &*i.password), ("HomeNet", "secret pw"));
    b.tx.send(Cmd::Close).unwrap();
    until(&mut b, 3.0, |s| {
        s.conn == Conn::Off && !s.paired && s.ap.is_none()
    });
    assert!(b.idle());
    println!("camera got: {:?}", codes(&log));
}

/// Pairing cancelled on this side withdraws the camera's prompt; a lost link resets the state; commands that need a
/// camera say so.
#[test]
fn cancel_and_loss() {
    let log = Log::default();
    let (mut b, sim) = spawn(&log);
    b.tx.send(Cmd::Ap).unwrap();
    until(&mut b, 3.0, |s| s.note.as_ref().is_some_and(|n| n.1));
    assert_eq!(b.st.note.as_ref().unwrap().0, "connect to a camera first");
    b.tx.send(Cmd::Connect(adv().dev)).unwrap();
    until(&mut b, 3.0, |s| s.conn == Conn::On);
    b.tx.send(Cmd::Pair).unwrap();
    until(&mut b, 3.0, |s| s.busy.is_some());
    b.tx.send(Cmd::Cancel).unwrap();
    until(&mut b, 3.0, |s| s.busy.is_none());
    assert_eq!(b.st.note, Some(("pairing cancelled".into(), false)));
    until(&mut b, 3.0, |_| codes(&log).contains(&code::CANCEL_AUTH));
    assert!(!b.st.paired);
    sim.lock().unwrap().as_ref().unwrap().lose();
    until(&mut b, 3.0, |s| s.conn == Conn::Off);
    assert_eq!(
        b.st.note,
        Some(("Bluetooth link lost: link closed".into(), true))
    );
}

/// A camera attached by USB takes the connection and sends nothing: the pane says why.
#[test]
fn usb_attached() {
    let log = Log::default();
    let (mut b, sim) = spawn(&log);
    b.tx.send(Cmd::Scan(true)).unwrap();
    until(&mut b, 3.0, |s| !s.seen.is_empty());
    sim.lock().unwrap().as_ref().unwrap().mute(true);
    b.tx.send(Cmd::Connect(adv().dev)).unwrap();
    until(&mut b, 3.0, |s| s.conn == Conn::On);
    assert!(b.st.link.as_ref().unwrap().first.is_none());
    let n = b.st.note.clone().unwrap();
    assert!(
        n.1 && n
            .0
            .contains("ignores Bluetooth while it is connected by USB"),
        "{n:?}"
    );
}

/// The wake-up's steps as the flow reports them, in words.
#[test]
fn wake_steps() {
    let (mut b, _) = spawn(&Log::default());
    let mut a = adv();
    a.rssi = Some(-61);
    for s in [
        Step::Beacon(None),
        Step::Seen(a),
        Step::Retry(Error::Timeout),
        Step::Auth(7),
        Step::Auth(1),
        Step::Auth(0),
        Step::OpenWifi,
        Step::Ready,
    ] {
        b.tx.send(Cmd::Step(Box::new(s))).unwrap();
    }
    until(&mut b, 3.0, |s| s.steps.len() == 8);
    assert_eq!(
        b.st.steps,
        [
            "wake advertisement on",
            "camera seen at -61 dBm",
            "connect failed (timeout), trying again",
            "camera asleep, asking again",
            "allow this computer on the camera's screen",
            "authorized",
            "switching the camera's Wi-Fi on",
            "camera ready, its Wi-Fi is up",
        ]
    );
    assert_eq!(joined(3), "wrong password");
    assert_eq!(why(&Error::Refused(1)), "refused on the camera");
    assert_eq!(
        why(&Error::Unauthorized(4)),
        "another phone, watch or remote holds the camera's connection"
    );
}
