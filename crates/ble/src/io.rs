//! The BLE thread: a current-thread tokio runtime driving btleplug (scan, connect) and one task per connection that
//! moves bytes between GATT and its [`Sess`].

use crate::{
    Adv, Camera, Dev, Error, Event, Link, Result, os,
    sess::{Op, Sess},
    wire,
};
#[cfg(not(target_os = "linux"))]
use btleplug::api::{CentralState, ScanFilter};
#[cfg(target_os = "macos")]
use btleplug::{api::RetrievePeripheralsOptions, platform::PeripheralId};
use btleplug::{
    api::{
        Central, CentralEvent, CharPropFlags, Characteristic, Manager as _, Peripheral as _,
        Service, ValueNotification, WriteType,
    },
    platform::{Adapter, Manager, Peripheral},
};
use futures_util::{Stream, StreamExt};
use std::{
    cell::Cell,
    collections::{BTreeSet, HashMap},
    pin::Pin,
    sync::mpsc::{Sender, SyncSender},
    time::Duration,
};
use tokio::{
    sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel},
    task::JoinSet,
    time::{self, Instant},
};

pub(crate) enum Cmd {
    Scan(bool, SyncSender<Result<()>>),
    Connect(Dev, SyncSender<Result<Camera>>),
    /// The first advertisement of the X5 with this serial (or suffix) before the deadline, scanning meanwhile.
    Find(String, std::time::Instant, SyncSender<Result<Adv>>),
}

/// Connect timeout (the app: 20 s).
const CONNECT: Duration = Duration::from_secs(20);
/// How long the camera gets to send its first notification (the legacy app stack's limit, ble.md §5.1).
const FIRST: Duration = Duration::from_secs(5);
/// Spacing between frames (the app's send throttle, PROTOCOL.md §2.3).
const GAP: Duration = Duration::from_millis(4);
/// How often a link asks whether it is still up where a lost event could hide its end (btleplug's own backends).
const LIVE: Duration = Duration::from_secs(2);

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
            let a = Manager::new()
                .await
                .map_err(bt)?
                .adapters()
                .await
                .map_err(bt)?;
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
        // The radio scans while the caller asked for it (`on`) or a find waits; only the caller gets `Seen`.
        let mut on = false;
        let mut finds: Vec<(String, std::time::Instant, SyncSender<Result<Adv>>)> = vec![];
        let mut sc: Radio = None;
        let mut ids = Ids::new();
        loop {
            let dl = finds.iter().map(|f| f.1).min();
            tokio::select! {
                c = cmds.recv() => match c {
                    Some(Cmd::Scan(s, tx)) => {
                        let r = radio(&a, &mut sc, s || !finds.is_empty(), &mut ids).await;
                        on = s && r.is_ok();
                        let _ = tx.send(r);
                    }
                    Some(Cmd::Find(k, t, tx)) => match radio(&a, &mut sc, true, &mut ids).await {
                        Ok(()) => finds.push((k, t, tx)),
                        Err(e) => drop(tx.send(Err(e))),
                    },
                    Some(Cmd::Connect(d, tx)) => drop(js.spawn(conn(a.clone(), d, ev.clone(), tx))),
                    None => break,
                },
                Some(e) = es.next() => match e {
                    // CoreBluetooth ends a scan when Bluetooth goes off and never takes it up again. Any update
                    // forgets the scan: btleplug drops events that overflow its channel, an "off" among them.
                    #[cfg(not(target_os = "linux"))]
                    CentralEvent::StateUpdate(s) => {
                        sc = None;
                        let r = match up(s) {
                            Ok(()) => radio(&a, &mut sc, on || !finds.is_empty(), &mut ids).await,
                            e => e,
                        };
                        let _ = ev.send(Event::Adapter(r.err()));
                    }
                    e => if sc.is_some() && let Some(v) = seen(&a, e, &mut ids).await {
                        for f in finds.extract_if(.., |f| v.is(&f.0)) {
                            let _ = f.2.send(Ok(v.clone()));
                        }
                        if on {
                            let _ = ev.send(Event::Seen(v));
                        }
                    },
                },
                _ = async {
                    match dl {
                        Some(t) => time::sleep_until(t.into()).await,
                        None => std::future::pending().await,
                    }
                } => {
                    let now = std::time::Instant::now();
                    for f in finds.extract_if(.., |f| f.1 <= now) {
                        let _ = f.2.send(Err(Error::Timeout));
                    }
                }
                Some(_) = js.join_next(), if !js.is_empty() => {}
            }
            if !on && finds.is_empty() {
                let _ = radio(&a, &mut sc, false, &mut ids).await;
            }
        }
        let _ = radio(&a, &mut sc, false, &mut ids).await;
        // Cameras outlive the `Ble` handle; the thread ends with the last one.
        while js.join_next().await.is_some() {}
    })
}

/// A running scan. BlueZ: our own LE-only discovery instead of btleplug's (bluez.rs), which ends when dropped.
#[cfg(target_os = "linux")]
type Radio = Option<crate::bluez::Scan>;
#[cfg(not(target_os = "linux"))]
type Radio = Option<()>;

/// Starts or stops scanning; a start forgets what the last scan learnt about each address.
async fn radio(a: &Adapter, r: &mut Radio, s: bool, ids: &mut Ids) -> Result<()> {
    if s == r.is_some() {
        return Ok(());
    }
    if !s {
        #[cfg(not(target_os = "linux"))]
        a.stop_scan().await.map_err(os)?;
        *r = None;
        return Ok(());
    }
    ids.clear();
    #[cfg(target_os = "linux")]
    {
        *r = Some(scan(a).await?);
    }
    #[cfg(not(target_os = "linux"))]
    {
        // CoreBluetooth takes a scan in any state and scans only while Bluetooth is on and allowed
        a.adapter_state().await.map_err(os).and_then(up)?;
        a.start_scan(ScanFilter::default()).await.map_err(os)?;
        *r = Some(());
    }
    Ok(())
}

/// btleplug's error; a refused Bluetooth permission (macOS) as [`Error::Denied`].
fn bt(e: btleplug::Error) -> Error {
    match e {
        btleplug::Error::PermissionDenied => Error::Denied,
        e => os(e),
    }
}

/// The adapter's state as a result. btleplug reports CoreBluetooth's unauthorized, unsupported and resetting alike;
/// once the first state came (`Ble::open` waits for it), that is nearly always Bluetooth not allowed for the app.
#[cfg(not(target_os = "linux"))]
fn up(s: CentralState) -> Result<()> {
    match s {
        CentralState::PoweredOn => Ok(()),
        CentralState::PoweredOff => Err(Error::Off),
        CentralState::Unknown => Err(Error::Denied),
    }
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

/// The advertisement behind an adapter event, folded into what the scan knows.
async fn seen(a: &Adapter, e: CentralEvent, ids: &mut Ids) -> Option<Adv> {
    use CentralEvent::*;
    let id = match e {
        DeviceDiscovered(id)
        | DeviceUpdated(id)
        | ManufacturerDataAdvertisement { id, .. }
        | ServiceDataAdvertisement { id, .. }
        | ServicesAdvertisement { id, .. }
        | RssiUpdate { id, .. } => id,
        _ => return None,
    };
    let q = a.peripheral(&id).await.ok()?.properties().await.ok()??;
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
    Some(v)
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

/// The peripheral CoreBluetooth knows by this UUID (`retrievePeripheralsWithIdentifiers`), back in btleplug's list.
#[cfg(target_os = "macos")]
async fn known(a: &Adapter, d: &Dev) -> Option<Peripheral> {
    let o = RetrievePeripheralsOptions {
        identifiers: Some(vec![PeripheralId::from(crate::Uuid::parse_str(&d.0).ok()?)]),
        services: None,
    };
    let ps = a.retrieve_peripherals(o).await.ok()?;
    ps.into_iter().find(|p| p.id().to_string() == d.0)
}

/// Hands the link's notifications on as they come, also while its task waits for a write: btleplug's own backends
/// (macOS) pass them through a broadcast channel of 16 that drops what overflows. Ends with the stream; there also
/// when a check every [`LIVE`] finds the link down, as the adapter's events take such a channel too and a lost
/// `DeviceDisconnected` would leave the link up with every request timing out.
async fn pump(
    p: Peripheral,
    mut ns: Pin<Box<dyn Stream<Item = ValueNotification> + Send>>,
    tx: UnboundedSender<ValueNotification>,
) {
    let mut t =
        cfg!(not(target_os = "linux")).then(|| time::interval_at(Instant::now() + LIVE, LIVE));
    loop {
        tokio::select! {
            n = ns.next() => if n.is_none_or(|n| tx.send(n).is_err()) {
                return;
            },
            _ = async {
                match &mut t {
                    Some(t) => {
                        t.tick().await;
                    }
                    None => std::future::pending().await,
                }
            } => if !p.is_connected().await.unwrap_or(false) {
                return;
            },
        }
    }
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
    // CoreBluetooth takes a connect in any state too; with Bluetooth off btleplug has forgotten every device
    #[cfg(not(target_os = "linux"))]
    if let Err(e) = a.adapter_state().await.map_err(os).and_then(up) {
        return drop(done.send(Err(e)));
    }
    let ps = match a.peripherals().await {
        Ok(ps) => ps,
        Err(e) => return drop(done.send(Err(os(e)))),
    };
    let p = ps.into_iter().find(|p| p.id().to_string() == d.0);
    // btleplug's CoreBluetooth backend forgets a peripheral at every disconnect; CoreBluetooth still knows it
    #[cfg(target_os = "macos")]
    let p = match p {
        None => known(&a, &d).await,
        p => p,
    };
    let Some(p) = p else {
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
        let (ntx, mut ns) = unbounded_channel();
        // dropped with the link, which ends the pump
        let mut aux = JoinSet::new();
        aux.spawn(pump(p.clone(), p.notifications().await.map_err(os)?, ntx));
        // The app waits 200 ms after the MTU exchange before it subscribes (ble.md §4.5).
        time::sleep(Duration::from_millis(200)).await;
        p.subscribe(rx).await.map_err(os)?;
        let t0 = Instant::now();
        let first = time::timeout(FIRST, async {
            while let Some(n) = ns.recv().await {
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
        Ok((es, ns, aux, tx.clone(), l))
    };
    let (mut es, mut ns, _aux, tx, l) = match r.await {
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
            m = ns.recv() => match m {
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
