//! The wake advertisement through BlueZ's `LEAdvertisingManager1` (PROTOCOL.md §2.10): we export one
//! `org.bluez.LEAdvertisement1` object, register it, and answer BlueZ's `GetAll` and `Release` on a raw D-Bus
//! channel. A blocking `RegisterAdvertisement` would deadlock: BlueZ reads our properties before it replies.

use crate::{Error, Result, Wake, os, wire};
use dbus::{
    Message, Path,
    arg::{PropMap, RefArg, Variant},
    channel::{BusType, Channel},
    message::MessageType,
    strings::ErrorName,
};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
        mpsc::{SyncSender, sync_channel},
    },
    thread,
    time::Duration,
};

const MGR: &str = "org.bluez.LEAdvertisingManager1";
const T: Duration = Duration::from_secs(5);

pub(crate) fn start(suffix: &str) -> Result<Wake> {
    let d = wire::wake(suffix).ok_or(Error::Unsupported)?;
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = sync_channel(1);
    let s = stop.clone();
    let t = thread::Builder::new()
        .name("x5-wake".into())
        .spawn(move || {
            if let Err(e) = run(d, &s, &tx) {
                let _ = tx.send(Err(e));
            }
        })?;
    let w = Wake { stop, t: Some(t) };
    rx.recv().unwrap_or(Err(Error::Closed))?;
    Ok(w)
}

fn call(path: &str, iface: &str, m: &str) -> Message {
    Message::new_method_call("org.bluez", path, iface, m).unwrap()
}

fn run(d: [u8; 24], stop: &AtomicBool, ready: &SyncSender<Result<()>>) -> Result<()> {
    static N: AtomicU32 = AtomicU32::new(0);
    let c = Channel::get_private(BusType::System).map_err(os)?;
    // The first adapter that can advertise.
    let r = c
        .send_with_reply_and_block(
            call(
                "/",
                "org.freedesktop.DBus.ObjectManager",
                "GetManagedObjects",
            ),
            T,
        )
        .map_err(os)?;
    let objs: HashMap<Path<'static>, HashMap<String, PropMap>> = r.read1().map_err(os)?;
    let a = objs
        .into_iter()
        .filter(|o| o.1.contains_key(MGR))
        .map(|o| o.0)
        .min()
        .ok_or(Error::NoAdapter)?;
    let me = Path::from(format!("/x5desk/wake{}", N.fetch_add(1, Relaxed)));
    let mut props = PropMap::new();
    let v = |x: Box<dyn RefArg>| Variant(x);
    props.insert("Type".into(), v(Box::new("peripheral".to_string())));
    let md = HashMap::from([(wire::APPLE, v(Box::new(d.to_vec())))]);
    props.insert("ManufacturerData".into(), v(Box::new(md)));
    // The app's settings: low-latency mode (100 ms) and TX_POWER_HIGH (1 dBm on Android; BlueZ applies it when
    // the adapter lists CanSetTxPower).
    props.insert("MinInterval".into(), v(Box::new(100u32)));
    props.insert("MaxInterval".into(), v(Box::new(100u32)));
    props.insert("TxPower".into(), v(Box::new(1i16)));
    let reg = c
        .send(call(&a, MGR, "RegisterAdvertisement").append2(&me, PropMap::new()))
        .map_err(|_| Error::Closed)?;
    let mut up = false;
    let mut unreg = None;
    // Polls left after a stop: BlueZ also drops the advertisement when our connection closes.
    let mut left = 30;
    let bad = ErrorName::from("org.freedesktop.DBus.Error.UnknownMethod");
    loop {
        if stop.load(Relaxed) {
            if up && unreg.is_none() {
                let m = call(&a, MGR, "UnregisterAdvertisement").append1(&me);
                unreg = Some(c.send(m).map_err(|_| Error::Closed)?);
            }
            if left == 0 {
                return Ok(());
            }
            left -= 1
        }
        let Some(m) = c
            .blocking_pop_message(Duration::from_millis(100))
            .map_err(os)?
        else {
            continue;
        };
        let rs = m.get_reply_serial();
        match m.msg_type() {
            MessageType::MethodCall if m.path().as_deref() == Some(&*me) => {
                let r = match m.member().as_deref() {
                    Some("GetAll") => m.method_return().append1(&props),
                    // BlueZ dropped the advertisement (adapter off, bluetoothd restart).
                    Some("Release") => {
                        let _ = c.send(m.method_return());
                        c.flush();
                        return Ok(());
                    }
                    _ => m.error(&bad, c"no such method"),
                };
                let _ = c.send(r);
            }
            MessageType::MethodCall => drop(c.send(m.error(&bad, c"no such object"))),
            MessageType::MethodReturn | MessageType::Error if rs == Some(reg) => {
                let r = match m.msg_type() {
                    MessageType::MethodReturn => Ok(()),
                    _ => Err(os(format!("RegisterAdvertisement: {:?}", m.get1::<&str>()))),
                };
                up = r.is_ok();
                let _ = ready.send(r);
                if !up {
                    return Ok(());
                }
            }
            MessageType::MethodReturn | MessageType::Error if rs.is_some() && rs == unreg => {
                return Ok(());
            }
            _ => {}
        }
    }
}
