//! The BLE thread: a current-thread tokio runtime driving btleplug (scan, connect) and one task per connection that
//! moves bytes between GATT and its [`Sess`].

use crate::{
    Adv, Camera, Dev, Error, Event, Link, Result, os,
    sess::{Op, Sess},
    wire,
};
#[cfg(not(target_os = "linux"))]
use btleplug::api::ScanFilter;
use btleplug::{
    api::{
        Central, CentralEvent, CharPropFlags, Characteristic, Manager as _, Peripheral as _,
        Service, WriteType,
    },
    platform::{Adapter, Manager},
};
use futures_util::StreamExt;
use std::{
    cell::Cell,
    collections::{BTreeSet, HashMap},
    sync::mpsc::{Sender, SyncSender},
    time::Duration,
};
use tokio::{
    sync::mpsc::{UnboundedReceiver, unbounded_channel},
    task::JoinSet,
    time::{self, Instant},
};

pub(crate) enum Cmd {
    Scan(bool, SyncSender<Result<()>>),
    Connect(Dev, SyncSender<Result<Camera>>),
}

/// Connect timeout (the app: 20 s).
const CONNECT: Duration = Duration::from_secs(20);
/// How long the camera gets to send its first notification (the legacy app stack's limit, ble.md §5.1).
const FIRST: Duration = Duration::from_secs(5);
/// Spacing between frames (the app's send throttle, PROTOCOL.md §2.3).
const GAP: Duration = Duration::from_millis(4);

thread_local! {
    /// Set on the BLE thread, where dropping a [`Camera`] must not wait for its own link task.
    pub(crate) static BLE: Cell<bool> = const { Cell::new(false) };
}

/// Per address during one scan: marked as an X5, the last `X5 <token>` name.
pub(crate) type Ids = HashMap<Dev, (bool, Option<String>)>;

/// Folds one advertisement into what the scan knows about its address, and that back into the advertisement.
pub(crate) fn fold(ids: &mut Ids, a: &mut Adv) {
    let k = ids.entry(a.dev.clone()).or_default();
    k.0 |= a.mark();
    if a.name.as_deref().is_some_and(crate::x5name) {
        k.1.clone_from(&a.name)
    }
    a.x5 = k.0;
    a.x5name.clone_from(&k.1);
}

pub(crate) fn run(
    mut cmds: UnboundedReceiver<Cmd>,
    ev: Sender<Event>,
    ready: SyncSender<Result<()>>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => return drop(ready.send(Err(e.into()))),
    };
    BLE.set(true);
    rt.block_on(async move {
        let r = async {
            let a = Manager::new().await.map_err(os)?.adapters().await.map_err(os)?;
            let a = a.into_iter().next().ok_or(Error::NoAdapter)?;
            let es = a.events().await.map_err(os)?;
            Ok((a, es))
        };
        let (a, mut es) = match r.await {
            Ok(x) => x,
            Err(e) => return drop(ready.send(Err(e))),
        };
        let _ = ready.send(Ok(()));
        let mut js = JoinSet::new();
        let mut on = false;
        // BlueZ: our own LE-only discovery instead of btleplug's (bluez.rs).
        #[cfg(target_os = "linux")]
        let mut le = None;
        let mut ids = Ids::new();
        loop {
            tokio::select! {
                c = cmds.recv() => match c {
                    Some(Cmd::Scan(s, tx)) => {
                        if s && !on {
                            ids.clear()
                        }
                        #[cfg(target_os = "linux")]
                        let r = if s {
                            scan(&a).await.map(|x| drop(le.replace(x)))
                        } else {
                            le.take();
                            Ok(())
                        };
                        #[cfg(not(target_os = "linux"))]
                        let r = if s { a.start_scan(ScanFilter::default()).await } else { a.stop_scan().await }
                            .map_err(os);
                        on = s && r.is_ok();
                        let _ = tx.send(r);
                    }
                    Some(Cmd::Connect(d, tx)) => drop(js.spawn(conn(a.clone(), d, ev.clone(), tx))),
                    None => break,
                },
                Some(e) = es.next() => if on {
                    seen(&a, e, &mut ids, &ev).await
                },
                Some(_) = js.join_next(), if !js.is_empty() => {}
            }
        }
        #[cfg(target_os = "linux")]
        drop(le);
        #[cfg(not(target_os = "linux"))]
        if on {
            let _ = a.stop_scan().await;
        }
        // Cameras outlive the `Ble` handle; the thread ends with the last one.
        while js.join_next().await.is_some() {}
    })
}

/// Starts an LE-only discovery on btleplug's adapter.
#[cfg(target_os = "linux")]
async fn scan(a: &Adapter) -> Result<crate::bluez::Scan> {
    // `hci0 (usb:…)`: btleplug's adapter id, which names the D-Bus object.
    let i = a.adapter_info().await.map_err(os)?;
    let p = format!("/org/bluez/{}", i.split(' ').next().unwrap_or_default());
    tokio::task::spawn_blocking(move || crate::bluez::Scan::start(&p))
        .await
        .map_err(os)?
}

async fn seen(a: &Adapter, e: CentralEvent, ids: &mut Ids, ev: &Sender<Event>) {
    use CentralEvent::*;
    let id = match e {
        DeviceDiscovered(id)
        | DeviceUpdated(id)
        | ManufacturerDataAdvertisement { id, .. }
        | ServiceDataAdvertisement { id, .. }
        | ServicesAdvertisement { id, .. }
        | RssiUpdate { id, .. } => id,
        _ => return,
    };
    let Ok(p) = a.peripheral(&id).await else {
        return;
    };
    let Ok(Some(q)) = p.properties().await else {
        return;
    };
    let mut mfr: Vec<_> = q.manufacturer_data.into_iter().collect();
    mfr.sort_unstable_by_key(|m| m.0);
    let mut v = Adv {
        dev: Dev(id.to_string()),
        addr: q.address.to_string(),
        // BlueZ's local name is the alias, which falls back to the address.
        name: q
            .advertisement_name
            .or(q.local_name.filter(|_| !cfg!(target_os = "linux"))),
        rssi: q.rssi,
        services: q.services,
        mfr,
        x5: false,
        x5name: None,
    };
    fold(ids, &mut v);
    let _ = ev.send(Event::Seen(v));
}

/// The camera service and its write / notify characteristics (ble.md §4.6): walking the characteristics, the
/// notify one is the first with NOTIFY, the write one the last with READ and WRITE up to it. Android walks in handle
/// order; btleplug sorts by UUID, so a read / write characteristic after the notify one is taken too.
fn pick(ss: &BTreeSet<Service>) -> Option<(&Service, &Characteristic, &Characteristic)> {
    let s = ss.iter().find(|s| s.uuid.to_string().contains("be80"))?;
    let rw = CharPropFlags::READ | CharPropFlags::WRITE;
    let (mut tx, mut rx) = (None, None);
    for c in &s.characteristics {
        if c.properties.contains(rw) {
            tx = Some(c)
        }
        if c.properties.contains(CharPropFlags::NOTIFY) {
            rx = Some(c);
            break;
        }
    }
    let tx = tx.or_else(|| s.characteristics.iter().find(|c| c.properties.contains(rw)))?;
    Some((s, tx, rx?))
}

async fn conn(a: Adapter, d: Dev, ev: Sender<Event>, done: SyncSender<Result<Camera>>) {
    #[cfg(target_os = "linux")]
    {
        let d = d.0.clone();
        if let Err(e) = tokio::task::spawn_blocking(move || crate::bluez::le(&d))
            .await
            .map_err(os)
            .and_then(|r| r)
        {
            return drop(done.send(Err(e)));
        }
    }
    let ps = match a.peripherals().await {
        Ok(ps) => ps,
        Err(e) => return drop(done.send(Err(os(e)))),
    };
    let Some(p) = ps.into_iter().find(|p| p.id().to_string() == d.0) else {
        return drop(done.send(Err(Error::NoDevice)));
    };
    let r = async {
        let es = a.events().await.map_err(os)?;
        time::timeout(CONNECT, p.connect())
            .await
            .map_err(|_| Error::Timeout)?
            .map_err(os)?;
        p.discover_services().await.map_err(os)?;
        let ss = p.services();
        let (s, tx, rx) = pick(&ss).ok_or(Error::NoService)?;
        let mut ns = p.notifications().await.map_err(os)?;
        // The app waits 200 ms after the MTU exchange before it subscribes (ble.md §4.5).
        time::sleep(Duration::from_millis(200)).await;
        p.subscribe(rx).await.map_err(os)?;
        let t0 = Instant::now();
        let first = time::timeout(FIRST, async {
            while let Some(n) = ns.next().await {
                if n.uuid == rx.uuid {
                    return Some((t0.elapsed(), n.value));
                }
            }
            None
        })
        .await
        .ok()
        .flatten();
        let l = Link {
            service: s.uuid,
            tx: tx.uuid,
            rx: rx.uuid,
            mtu: p.mtu(),
            // The legacy app stack switches to UCP when the first notification starts `ff 06 41` (ble.md §5.1).
            ucp: first
                .as_ref()
                .is_some_and(|f| f.1.starts_with(&[0xFF, 6, 0x41])),
            first,
        };
        Ok((es, ns, tx.clone(), l))
    };
    let (mut es, mut ns, tx, l) = match r.await {
        Ok(x) => x,
        Err(e) => {
            let _ = p.disconnect().await;
            return drop(done.send(Err(e)));
        }
    };
    let n = (l.mtu as usize).saturating_sub(3).clamp(20, wire::CHUNK);
    let (pid, rx) = (p.id(), l.rx);
    let mut s = Sess::new(d.clone(), ev.clone());
    s.ucp = Some(l.ucp);
    let mut e = l.first.as_ref().and_then(|f| s.feed(&f.1).err());
    let (otx, mut ops) = unbounded_channel();
    // A caller that stopped waiting drops the camera here, which closes it.
    let _ = done.send(Ok(Camera::new(otx, d.clone(), l)));
    let mut last = Instant::now();
    let mut asked = None;
    let e = loop {
        if e.is_some() {
            break e;
        }
        for f in s.out.drain(..) {
            time::sleep_until(last + GAP).await;
            for c in f.chunks(n) {
                // The app drops the rest of a frame whose write fails (PROTOCOL.md §2.3); its request times out.
                if p.write(&tx, c, WriteType::WithResponse).await.is_err() {
                    break;
                }
            }
            last = Instant::now();
        }
        let dl = s.expire(std::time::Instant::now());
        tokio::select! {
            o = ops.recv() => match o {
                Some(Op::Close(c)) => {
                    asked = c;
                    break None;
                }
                Some(o) => s.op(o, std::time::Instant::now()),
                None => break None,
            },
            m = ns.next() => match m {
                Some(m) if m.uuid == rx => e = s.feed(&m.value).err(),
                Some(_) => {}
                None => break Some(Error::Closed),
            },
            Some(c) = es.next() => if matches!(c, CentralEvent::DeviceDisconnected(i) if i == pid) {
                break Some(Error::Closed);
            },
            _ = async {
                match dl {
                    Some(t) => time::sleep_until(t.into()).await,
                    None => std::future::pending().await,
                }
            } => {}
        }
    };
    s.fail();
    let _ = p.disconnect().await;
    match (e, asked) {
        (Some(e), _) => drop(ev.send(Event::Closed(d, e))),
        (None, Some(c)) => drop(c.send(Ok(()))),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Uuid, wire};

    /// The X5's two advertisements on one address, interleaved with another device (the order the hardware check
    /// saw: whichever came last named the device).
    #[test]
    fn ids() {
        let bsa = || {
            let mut a = Adv::test("My BSA Bluetooth Device");
            a.services = vec![Uuid::from_u128(0x0000be80_0000_1000_8000_00805f9b34fb)];
            a.mfr = vec![(0x000F, vec![0, 0, 0])];
            a
        };
        let x5 = |n: &str, b| {
            let mut a = Adv::test(n);
            a.mfr = vec![(wire::INSTA, vec![0, b, 1])];
            a
        };
        let mut other = Adv::test("X4 Air GHIJKL");
        other.dev = Dev("hci0/dev_11_11_11_11_11_11".into());
        let mut ids = Ids::new();
        let mut f = |mut a: Adv| {
            fold(&mut ids, &mut a);
            (a.x5, a.token().map(String::from), a.is("IAHxx00ABCDEF"))
        };
        let tok = || Some("ABCDEF".to_string());
        // `be80` alone marks the address before the X5 name ever came.
        assert_eq!(f(bsa()), (true, None, false));
        assert_eq!(f(other.clone()), (false, None, false));
        assert_eq!(f(x5("X5 ABCDEF", 0xFF)), (true, tok(), true));
        // The BSA advertisement renames the device; the address stays the camera with its token.
        assert_eq!(f(bsa()), (true, tok(), true));
        assert_eq!(f(other.clone()), (false, None, false));
        assert_eq!(f(x5("X5 ABCDEF", 0x78)), (true, tok(), true));
        assert_eq!(f(bsa()), (true, tok(), true));
        // A nameless advertisement with only the 1a2b record marks a new address; the last X5 name wins.
        let mut n = x5("", 0xFF);
        n.name = None;
        n.dev = Dev("hci0/dev_22_22_22_22_22_22".into());
        assert_eq!(f(n.clone()), (true, None, false));
        n.name = Some("X5 ZYXWVU".into());
        assert_eq!(f(n.clone()), (true, Some("ZYXWVU".into()), false));
        n.name = Some("X5 ABCDEF-2".into());
        assert_eq!(f(n), (true, Some("ABCDEF-2".into()), true));
        assert_eq!(ids.len(), 3);
    }
}
