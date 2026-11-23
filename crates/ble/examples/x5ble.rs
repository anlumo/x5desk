//! BLE checks on this machine's adapter and a real X5.
//!
//! ```text
//! x5ble scan [secs]                  every device advertising for secs (10), X5s marked
//! x5ble wake <suffix> [uuid|-] [hold] [pause] [standby]
//!                                    the app's wake-up (`Ble::wake_up`): wake advertisement, find "X5 <suffix>",
//!                                    hold secs (0) with only the advertisement up, connect, the app's session up to
//!                                    "camera ready" (asleep: CHECK_AUTHORIZATION 7, asked again), pause secs (0)
//!                                    after each step; `standby` then sends SET_STANDBY_MODE{WAKE_UP} (56), which the
//!                                    app never sends. Prints each step and each change in the camera's
//!                                    advertisement with its time, then keeps the link 10 s for notifications
//! x5ble info <suffix|address>        find "X5 <suffix>" (30 s; an address `AA:BB:…` skips the scan), connect, print
//!                                    the link and identity, read the Wi-Fi info (password masked), disconnect
//! x5ble connect <suffix|address> [uuid] [wakeup]
//!                                    `info`, then authorize (confirm on the camera), with `wakeup` switch on
//!                                    `ble_wake` (for `wake`), wait for the AP and print its credentials
//! ```
//!
//! `uuid` (or `X5_UUID`, also for `-`) is the persistent client id; without one a fresh id is made and printed.
//!
//! Every way out of `connect` (success, error, Ctrl-C or SIGTERM, panic) drops the camera handle, which disconnects
//! before the process ends.

use ble::{Adv, Ble, Dev, Event, Note, Pending, Step};
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{BuildHasher, RandomState},
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc::Receiver,
    },
    thread,
    time::{Duration, Instant},
};

type E = Box<dyn std::error::Error>;
type R = Result<(), E>;

/// Set by SIGINT / SIGTERM: the waits below turn it into an error instead of the process dying with the link up.
static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
fn trap() {
    unsafe extern "C" {
        fn signal(sig: i32, f: extern "C" fn(i32)) -> usize;
    }
    extern "C" fn on(_: i32) {
        STOP.store(true, Relaxed)
    }
    // SIGINT, SIGTERM.
    unsafe {
        signal(2, on);
        signal(15, on);
    }
}

#[cfg(not(unix))]
fn trap() {}

/// [`Pending::wait`] that gives up on a signal.
fn w<T>(p: Pending<T>) -> Result<T, E> {
    loop {
        if let Some(r) = p.poll() {
            return Ok(r?);
        }
        if STOP.load(Relaxed) {
            return Err("interrupted".into());
        }
        thread::sleep(Duration::from_millis(20))
    }
}

fn main() -> R {
    trap();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let secs =
        |i: usize, d| Duration::from_secs(a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d));
    match (a.first().map(|s| s.as_str()), a.get(1)) {
        (Some("scan"), _) => scan(secs(1, 10)),
        (Some("wake"), Some(s)) => {
            let u = a.get(2).filter(|u| *u != "-").cloned();
            let u = u.or_else(|| std::env::var("X5_UUID").ok());
            wake(
                s,
                u.unwrap_or_else(uuid),
                secs(3, 0),
                secs(4, 0),
                a.get(5).is_some_and(|x| x == "standby"),
            )
        }
        (Some("info"), Some(s)) => connect(s, None),
        (Some("connect"), Some(s)) => {
            let u = a.get(2).cloned().or_else(|| std::env::var("X5_UUID").ok());
            connect(
                s,
                Some((
                    u.unwrap_or_else(uuid),
                    a.get(3).is_some_and(|w| w == "wakeup"),
                )),
            )
        }
        _ => Err(
            "usage: x5ble scan [secs] | wake <suffix> [uuid|-] [hold] [pause] [standby] \
             | info <suffix|address> | connect <suffix|address> [uuid] [wakeup]"
                .into(),
        ),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter()
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn line(a: &Adv) -> String {
    let mut s = format!(
        "{} rssi {:>4} {:?} services {}{}",
        a.addr,
        a.rssi.map_or("?".into(), |r| r.to_string()),
        a.name.as_deref().unwrap_or(""),
        a.services.len(),
        if a.insta() { " (be80)" } else { "" },
    );
    for (c, d) in &a.mfr {
        s += &format!(" mfr {c:04x}:[{}]", hex(d));
    }
    if let Some(p) = a.proto() {
        let w = match p.wifi {
            0 => "off".into(),
            255 => "not started".into(),
            t => format!("up {t} s"),
        };
        s += &format!(" (wifi {w}, password v{})", p.pwd);
    }
    if let Some(b) = a.beacon() {
        s += &format!(" ibeacon {}/{}", b.major, b.minor);
    }
    if a.x5 {
        s += &format!("  <-- X5 {}", a.token().unwrap_or("(no X5 name yet)"));
    }
    s
}

fn scan(t: Duration) -> R {
    let (b, ev) = Ble::open()?;
    let t0 = Instant::now();
    b.scan(true).wait()?;
    let mut seen = BTreeMap::new();
    let mut n = 0;
    while let Some(left) = (t0 + t).checked_duration_since(Instant::now())
        && !STOP.load(Relaxed)
    {
        if let Ok(Event::Seen(a)) = ev.recv_timeout(left.min(Duration::from_millis(100))) {
            n += 1;
            seen.insert(a.dev.0.clone(), a);
        }
    }
    b.scan(false).wait()?;
    for a in seen.values() {
        println!("{}", line(a));
    }
    // The last advertisement per address carries what the scan learnt from all of them.
    let x5 = seen.values().filter(|a| a.x5).count();
    let insta = seen.values().filter(|a| a.insta()).count();
    println!(
        "{n} advertisements from {} devices in {:.1} s; named {}, X5 {x5}, be80 service {insta}",
        seen.len(),
        t0.elapsed().as_secs_f64(),
        seen.values().filter(|a| a.name.is_some()).count(),
    );
    Ok(())
}

/// Waits for the camera with this suffix while scanning.
fn find(b: &Ble, ev: &Receiver<Event>, s: &str, t: Duration) -> Result<Adv, E> {
    let t0 = Instant::now();
    b.scan(true).wait()?;
    let r = loop {
        let Some(left) = (t0 + t).checked_duration_since(Instant::now()) else {
            break Err(format!("no \"X5 {s}\" within {t:?}").into());
        };
        if STOP.load(Relaxed) {
            break Err("interrupted".into());
        }
        if let Ok(Event::Seen(a)) = ev.recv_timeout(left.min(Duration::from_millis(100)))
            && a.is(s)
        {
            println!("{:.2} s: {}", t0.elapsed().as_secs_f64(), line(&a));
            break Ok(a);
        }
    };
    b.scan(false).wait()?;
    r
}

fn wake(s: &str, uuid: String, hold: Duration, pause: Duration, standby: bool) -> R {
    let (b, ev) = Ble::open()?;
    let t0 = Instant::now();
    let at = move || t0.elapsed().as_secs_f64();
    // A scan of our own beside the flow's: every change in the camera's advertisement (the Wi-Fi seconds count
    // down, so about one line a second) and every notification.
    b.scan(true).wait()?;
    let k = s.to_string();
    thread::spawn(move || {
        let (mut mfr, mut names) = (vec![], BTreeSet::new());
        for e in ev {
            match e {
                Event::Seen(a) if a.is(&k) => {
                    if a.mfr != mfr || names.insert(a.name.clone()) {
                        println!("{:.2} s: {}", at(), line(&a));
                        mfr = a.mfr;
                    }
                }
                Event::Note(_, Note::Other(c, d)) => {
                    println!("{:.2} s: note {c} [{}]", at(), hex(&d))
                }
                Event::Note(_, n) => println!("{:.2} s: note {n:?}", at()),
                Event::Closed(_, e) => println!("{:.2} s: link lost: {e}", at()),
                Event::Adapter(e) => println!("{:.2} s: adapter: {e:?}", at()),
                Event::Seen(_) => {}
            }
        }
    });
    let f = b.wake_up(s, &uuid, hold, Duration::from_secs(30), move |x| {
        println!("{:.2} s: {}", at(), step(&x));
        thread::sleep(pause)
    });
    // Not interruptible: the flow owns the link until it ends.
    let c = f.wait();
    if let Err(e) = &c {
        println!("{:.2} s: wake-up failed: {e}", at());
    }
    let c = c?;
    if standby {
        // SetStandbyMode{STANDBY_MODE_WAKE_UP}.
        let r = w(c.raw(56, vec![0x08, 0x02], Duration::from_secs(10)));
        println!("{:.2} s: SET_STANDBY_MODE(WAKE_UP): {r:?}", at());
    }
    let end = Instant::now() + Duration::from_secs(10);
    while Instant::now() < end && !STOP.load(Relaxed) {
        thread::sleep(Duration::from_millis(50))
    }
    drop(c);
    b.scan(false).wait()?;
    println!("{:.2} s: disconnected", at());
    Ok(())
}

fn step(s: &Step) -> String {
    match s {
        Step::Beacon(None) => "wake advertisement up".into(),
        Step::Beacon(Some(e)) => format!("no wake advertisement ({e}), going on"),
        Step::Seen(a) => format!("found {}", line(a)),
        Step::Retry(e) => format!("connect failed: {e}; looking again"),
        Step::Connected(l) => match &l.first {
            Some((t, d)) => format!(
                "connected, first notification after {:.0} ms: [{}]",
                t.as_secs_f64() * 1e3,
                hex(d)
            ),
            None => "connected, no notification within 5 s".into(),
        },
        Step::Options(o) => format!(
            "identity {:?}, battery {:?}",
            o.model,
            o.battery.as_ref().map(|b| b.level)
        ),
        Step::Synced(Ok(o)) => format!(
            "option sync: ble_wake {}, firmware {:?} / {:?}",
            o.ble_wake, o.firmware, o.ota_version
        ),
        Step::Synced(Err(e)) => format!("option sync failed: {e}"),
        Step::Auth(s) => format!(
            "CHECK_AUTHORIZATION {s}{}",
            match s {
                0 => " (authorized)",
                1 => " (confirm on the camera)",
                7 => " (asleep: again in 2 s)",
                _ => "",
            }
        ),
        Step::OpenWifi => "no 8215 within 30 s: OPEN_CAMERA_WIFI".into(),
        Step::Ready => "8215: camera ready, wake advertisement off".into(),
    }
}

fn uuid() -> String {
    let (a, b) = (
        RandomState::new().hash_one(1),
        RandomState::new().hash_one(2),
    );
    let u = format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        a >> 32,
        a >> 16 & 0xFFFF,
        a & 0xFFF,
        b >> 48 & 0x3FFF | 0x8000,
        b & 0xFFFF_FFFF_FFFF
    );
    println!("new client id {u} (pass it again to stay authorized)");
    u
}

/// `auth`: client id and `wakeup`; without it the session stops before authorizing.
fn connect(s: &str, auth: Option<(String, bool)>) -> R {
    let (b, ev) = Ble::open()?;
    let d = if s.contains(':') {
        Dev(format!("hci0/dev_{}", s.replace(':', "_").to_uppercase()))
    } else {
        find(&b, &ev, s, Duration::from_secs(30))?.dev
    };
    thread::spawn(move || {
        for e in ev {
            match e {
                Event::Note(_, Note::Other(c, d)) => println!("note {c} [{}]", hex(&d)),
                Event::Note(_, n) => println!("note {n:?}"),
                Event::Closed(_, e) => println!("link lost: {e}"),
                Event::Adapter(e) => println!("adapter: {e:?}"),
                Event::Seen(_) => {}
            }
        }
    });
    let t0 = Instant::now();
    // Not interruptible: a connect abandoned midway could bring the link up after the process is gone.
    let c = b.connect(&d).wait();
    println!(
        "connect {} after {:.2} s",
        if c.is_ok() { "done" } else { "failed" },
        t0.elapsed().as_secs_f64()
    );
    let c = c?;
    let r = session(&c, auth);
    let t0 = Instant::now();
    drop(c);
    println!("disconnected in {:.2} s", t0.elapsed().as_secs_f64());
    r
}

fn session(c: &ble::Camera, auth: Option<(String, bool)>) -> R {
    let l = c.link();
    println!(
        "service {} tx {} rx {} mtu {} ucp {}",
        l.service, l.tx, l.rx, l.mtu, l.ucp
    );
    match &l.first {
        Some((t, d)) => println!(
            "first notification after {:.0} ms: [{}]",
            t.as_secs_f64() * 1e3,
            hex(d)
        ),
        // The app's error 4001 for this names the battery, U-Disk mode and another phone.
        None => println!("no notification within 5 s (camera on USB or held by another phone?)"),
    }
    let o = w(c.options(&[48, 15, 11, 198]))?;
    let sn = &o.serial;
    println!(
        "model {:?}, serial …{}, firmware {:?} / {:?}, battery {:?}, connections {:?}",
        o.model,
        &sn[sn.len().saturating_sub(6)..],
        o.firmware,
        o.ota_version,
        o.battery,
        o.links
    );
    // PROTOCOL.md §11 X14: does the camera hand out the password before authorization?
    match w(c.wifi()) {
        Ok(w) => println!(
            "before authorization: ssid {:?}, password {}, channel {}, mode {}, state {}",
            w.ssid,
            if w.password.is_empty() {
                "empty"
            } else {
                "present"
            },
            w.channel,
            w.mode,
            w.state
        ),
        Err(e) => println!("before authorization: {e}"),
    }
    let Some((uuid, wakeup)) = auth else {
        return Ok(());
    };
    println!("authorizing as {uuid}: confirm on the camera if it asks (60 s)");
    let t0 = Instant::now();
    w(c.authorize(&uuid, Duration::from_secs(60)))?;
    println!("authorized after {:.1} s", t0.elapsed().as_secs_f64());
    if wakeup {
        let on = ble::pb::Settings {
            ble_wake: 1,
            ..Default::default()
        };
        w(c.set_options(&[97], on))?;
    }
    println!("ble_wake {}", w(c.options(&[97]))?.ble_wake);
    let t0 = Instant::now();
    let a = w(c.ap(Duration::from_secs(30)))?;
    println!(
        "AP after {:.1} s: ssid {:?}, password {:?}, channel {}, mode {}, state {}",
        t0.elapsed().as_secs_f64(),
        a.ssid,
        a.password,
        a.channel,
        a.mode,
        a.state
    );
    Ok(())
}
