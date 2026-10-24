//! BlueZ's choice of bearer. btleplug discovers with `Transport::Auto`, so BlueZ also runs BR/EDR inquiry and learns the
//! X5's classic side (class 0x000424); `Device1.Connect` then prefers BR/EDR and fails with "No more profiles to
//! connect to". Scans are therefore BlueZ discoveries of our own, LE only ([`Scan`]), and a connect first makes sure
//! BlueZ knows no classic side ([`le`]). `Bearer.LE1.Connect` (BlueZ 5.84) would pick the bearer directly, but only
//! with bluetoothd's `--experimental`.

use crate::{Error, Result, os};
use dbus::{
    Path,
    arg::{PropMap, RefArg, Variant, prop_cast},
    blocking::{Connection, stdintf::org_freedesktop_dbus::Properties},
};
use std::{
    thread,
    time::{Duration, Instant},
};

const T: Duration = Duration::from_secs(5);
/// How long [`le`] looks for the device; the X5 advertises several times a second.
const FIND: Duration = Duration::from_secs(10);
const ADP: &str = "org.bluez.Adapter1";
const DEV: &str = "org.bluez.Device1";

/// An LE-only discovery. BlueZ ends a discovery when the D-Bus connection that started it closes: dropping stops it.
pub(crate) struct Scan {
    _c: Connection,
}

impl Scan {
    /// On the adapter at `adapter` (`/org/bluez/hci0`).
    pub fn start(adapter: &str) -> Result<Scan> {
        let c = Connection::new_system().map_err(os)?;
        let a = c.with_proxy("org.bluez", adapter, T);
        let () = a
            .method_call(ADP, "SetDiscoveryFilter", (filter(),))
            .map_err(os)?;
        let () = a.method_call(ADP, "StartDiscovery", ()).map_err(os)?;
        Ok(Scan { _c: c })
    }
}

/// LE only; every advertisement updates the device (btleplug's scan did the same), so the X5's alternating
/// advertisements all arrive.
fn filter() -> PropMap {
    let v = |x: Box<dyn RefArg>| Variant(x);
    PropMap::from([
        ("Transport".into(), v(Box::new("le".to_string()))),
        ("DuplicateData".into(), v(Box::new(true))),
    ])
}

/// Whether BlueZ's record of a device (its `Device1` properties; `bredr`: it has `Bearer.BREDR1`) makes `Connect`
/// pick BR/EDR while being only a cache: a classic side (`Class` comes from inquiry; `Bearer.BREDR1` also from
/// advertising flags) and not connected, paired or bonded.
fn stale(m: &PropMap, bredr: bool) -> bool {
    let on = |k| prop_cast::<bool>(m, k).is_some_and(|b| *b);
    (bredr || m.contains_key("Class")) && !["Connected", "Paired", "Bonded"].into_iter().any(on)
}

/// Makes BlueZ's next `Device1.Connect` to `dev` (`hci0/dev_…`) go over LE: a [`stale`] record (any earlier dual-mode
/// scan, ours or another program's) is removed and the device found again by an LE-only scan, which knows no classic
/// side. Blocks up to [`FIND`]; an unknown device (or a name that is no D-Bus path) is left to the caller.
pub(crate) fn le(dev: &str) -> Result<()> {
    let Ok(p) = Path::new(format!("/org/bluez/{dev}")) else {
        return Ok(());
    };
    let a = p.rsplit_once('/').map_or("", |x| x.0);
    let c = Connection::new_system().map_err(os)?;
    let d = c.with_proxy("org.bluez", p.clone(), T);
    match d.get_all(DEV) {
        Ok(m) if stale(&m, d.get_all("org.bluez.Bearer.BREDR1").is_ok()) => {}
        _ => return Ok(()),
    }
    let () = c
        .with_proxy("org.bluez", a, T)
        .method_call(ADP, "RemoveDevice", (p.clone(),))
        .map_err(os)?;
    let _s = Scan::start(a)?;
    let end = Instant::now() + FIND;
    while d.get::<String>(DEV, "Address").is_err() {
        if Instant::now() > end {
            return Err(Error::NoDevice);
        }
        thread::sleep(Duration::from_millis(100))
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer() {
        let v = |x: Box<dyn RefArg>| Variant(x);
        // The X5 after a dual-mode scan (`Class` 0x000424), with `on` true.
        let dual = |on: &str| -> PropMap {
            ["Connected", "Paired", "Bonded"]
                .map(|k| (k.to_string(), v(Box::new(k == on))))
                .into_iter()
                .chain([("Class".to_string(), v(Box::new(0x424u32)))])
                .collect()
        };
        assert!(stale(&dual(""), false) && stale(&dual(""), true));
        // Never removed while connected, paired or bonded.
        for k in ["Connected", "Paired", "Bonded"] {
            assert!(!stale(&dual(k), true), "{k}");
        }
        // After an LE-only scan: no `Class`, only `Bearer.LE1`.
        let le = PropMap::from([("Connected".into(), v(Box::new(false)))]);
        assert!(!stale(&le, false));
        // Classic support from advertising flags, which only `Bearer.BREDR1` shows.
        assert!(stale(&le, true));
        let f = filter();
        assert_eq!(prop_cast::<String>(&f, "Transport").unwrap(), "le");
        assert_eq!(prop_cast::<bool>(&f, "DuplicateData"), Some(&true));
    }
}
