//! Needs this machine's Bluetooth adapter; run with `--ignored`. Scans and advertises: nothing is connected or
//! written.

use ble::{Ble, Dev, Error, Event};
use std::time::{Duration, Instant};

#[test]
#[ignore]
fn scan() {
    let (b, ev) = Ble::open().unwrap();
    b.scan(true).wait().unwrap();
    let end = Instant::now() + Duration::from_secs(5);
    let mut n = 0;
    while let Some(t) = end.checked_duration_since(Instant::now()) {
        if let Ok(Event::Seen(a)) = ev.recv_timeout(t) {
            // BlueZ's address-derived alias never shows up as a name.
            assert!(
                a.name
                    .as_deref()
                    .is_none_or(|s| !s.contains(&a.addr.replace(':', "-")))
            );
            n += 1
        }
    }
    b.scan(false).wait().unwrap();
    println!("{n} advertisements in 5 s");
    assert!(n > 0, "nothing advertises near this adapter");
    // A device the adapter does not know: refused before any radio work.
    let d = Dev("hci0/dev_00_00_00_00_00_00".into());
    assert!(matches!(b.connect(&d).wait(), Err(Error::NoDevice)));
}

/// The wake advertisement (flags and the Apple record, the whole 31 bytes) for a made-up token: BlueZ takes it and
/// the adapter advertises for a second. Nothing is scanned or connected; no camera has this token.
#[test]
#[ignore]
#[cfg(target_os = "linux")]
fn advertise() {
    let w = ble::wake("ZZZZZZ").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    drop(w);
}
