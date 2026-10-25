//! The Insta360 X5's Bluetooth LE side (`../insta360/re/analysis/ble.md`, PROTOCOL.md §2): scan and identify
//! cameras, wake one from standby, connect over GATT, and the UCD session the app runs on it: authorization, Wi-Fi
//! mode and access-point credentials, station mode, any other request, notifications.
//!
//! [`Ble`] owns one thread per adapter with a private tokio runtime (btleplug is async); nothing async leaves the
//! crate. Calls return [`Pending`], which the caller blocks on or polls. Advertisements, notifications and lost links
//! arrive on the [`Event`] receiver returned by [`Ble::open`]. A [`Camera`] is a cheap, cloneable handle; the link
//! closes on [`Camera::close`] or when the last handle is dropped, and that drop waits (up to 5 s) until BlueZ has
//! the link down: a process that exits right after an error must not leave the LE link up.
//!
//! Waking a sleeping camera: [`Ble::wake_up`], the app's way. Its first part is the phone advertising, not a GATT
//! operation ([`wake`], BlueZ only); the rest is the app's BLE session up to "camera ready".
//!
//! Status on real hardware (2026-10-24): scans find the X5 (it alternates two advertisements on one address, see
//! [`Adv`]). Connects go over LE, also after a dual-mode scan (`bluez`), and find the pipe the app picks (`be80`:
//! `be81` read / write, `be82` notify, `be83` read), MTU 517. While the camera is attached over USB in app mode it
//! acknowledges every write and sends nothing (the app's error 4001). Unplugged, its first notification (a heartbeat,
//! `07 00 00 00 05 00 00`) comes at once, `GET_OPTIONS` and the Wi-Fi query answer, authorization succeeds and
//! `bt_wakeup_sw` can be set. In standby (short press, screen dark) it keeps advertising `X5 <token>` and answers
//! `GET_OPTIONS`. [`Ble::wake_up`] (2026-10-25, camera asleep with Wi-Fi off): found at once, the connect takes about
//! 35 s (the camera turns its Wi-Fi on just before it completes, against under 1 s while its Wi-Fi still counts down),
//! then authorized on the first ask (never 7) and 8215; `SET_STANDBY_MODE{WAKE_UP}` (56) answers 400. The screen
//! stays dark throughout: on an X5 the wake-up brings up its radios and the session, never the display.

#[cfg(target_os = "linux")]
mod bluez;
mod io;
mod sess;
#[cfg(feature = "sim")]
mod sim;
#[cfg(target_os = "linux")]
mod wake;
mod wire;

pub use proto as pb;
#[cfg(feature = "sim")]
pub use sim::Sim;
pub use uuid::Uuid;
pub use wire::{Beacon, Proto};

use prost::Message;
use sess::Op;
use std::{
    fmt,
    sync::{
        Arc,
        atomic::AtomicBool,
        mpsc::{self, Receiver, SyncSender, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};

/// Message codes used on BLE (`MessageCode`): requests below 8192, notifications from 8192.
pub mod code {
    pub const SET_OPTIONS: u16 = 7;
    pub const GET_OPTIONS: u16 = 8;
    pub const OPEN_WIFI: u16 = 33;
    pub const CLOSE_WIFI: u16 = 34;
    pub const CHECK_AUTH: u16 = 39;
    pub const CANCEL_AUTH: u16 = 40;
    pub const SET_WIFI_CONN: u16 = 112;
    pub const GET_WIFI_CONN: u16 = 113;
    pub const SET_WIFI_MODE: u16 = 147;
    pub const WIFI_SCAN: u16 = 148;
    pub const GET_WIFI_MODE: u16 = 150;
    pub const PHONE_INFO: u16 = 220;

    pub const AUTH_RESULT: u16 = 8209;
    /// The camera's access point is up (empty body).
    pub const WIFI_START: u16 = 8215;
    /// Station-mode join result.
    pub const WIFI_STATUS: u16 = 8232;
    pub const WIFI_MODE_CHANGE: u16 = 8247;
    pub const WIFI_SCAN_LIST: u16 = 8249;
    pub const WIFI_CLOSED: u16 = 8300;
}

/// Camera messages the BLE session uses that `proto` does not compile yet; tags from `re/proto/` (the files named on
/// each type).
pub mod msg {
    use prost::Message;

    /// `WifiMode` / `Options.WifiInfo.Mode`.
    pub const AP: i32 = 0;
    pub const STA: i32 = 1;

    /// `commands/get_wifi_mode.proto`
    #[derive(Clone, PartialEq, Message)]
    pub struct GetWifiModeResp {
        #[prost(int32, tag = "1")]
        pub wifi_mode: i32,
    }

    /// `commands/set_wifi_mode.proto`; the app always sends `place_holder` 1.
    #[derive(Clone, PartialEq, Message)]
    pub struct SetWifiMode {
        #[prost(uint32, tag = "1")]
        pub place_holder: u32,
        #[prost(int32, tag = "2")]
        pub wifi_mode: i32,
        #[prost(string, tag = "3")]
        pub country_code: String,
    }

    /// `wifi_mode.proto`, notification 8247.
    #[derive(Clone, PartialEq, Message)]
    pub struct WifiModeResult {
        #[prost(int32, tag = "1")]
        pub wifi_mode: i32,
        #[prost(int32, tag = "2")]
        pub err_code: i32,
    }

    /// `wifi_restart.proto`
    #[derive(Clone, PartialEq, Message)]
    pub struct RestartWifi {
        #[prost(uint32, tag = "1")]
        pub restart_channel: u32,
    }

    /// `wifi_connection_info.proto`: the camera's station-mode (home network) connection. `result`:
    /// `WifiConnectionResult` (0 success, 1 timeout, 2 connect failed, 3 wrong password, 4 no IP, 5 connected, 6 not
    /// found, … 11 channel forbidden).
    #[derive(Clone, PartialEq, Message)]
    pub struct WifiConnectionInfo {
        #[prost(string, tag = "1")]
        pub ssid: String,
        #[prost(string, tag = "2")]
        pub bssid: String,
        #[prost(string, tag = "3")]
        pub password: String,
        #[prost(string, tag = "4")]
        pub ip_addr: String,
        #[prost(int32, tag = "5")]
        pub result: i32,
        #[prost(int32, tag = "6")]
        pub is_auth_wifi: i32,
    }

    /// `commands/set_wifi_connection_info.proto`
    #[derive(Clone, PartialEq, Message)]
    pub struct SetWifiConnectionInfo {
        #[prost(message, optional, tag = "1")]
        pub wifi_connection_info: Option<WifiConnectionInfo>,
    }

    /// `commands/get_wifi_connection_info.proto`: the same layout.
    pub type GetWifiConnectionInfoResp = SetWifiConnectionInfo;

    /// `notifications/camera_wifi_connection_result.proto`, notification 8232.
    #[derive(Clone, PartialEq, Message)]
    pub struct CameraWifiConnectionResult {
        #[prost(int32, tag = "1")]
        pub wifi_connection_result: i32,
        #[prost(message, optional, tag = "2")]
        pub wifi_connection_info: Option<WifiConnectionInfo>,
    }

    /// The same file, notification 8300; `close_type` 0 normal, 1 shutdown, 2 firmware update.
    #[derive(Clone, PartialEq, Message)]
    pub struct WifiCloseInfo {
        #[prost(int32, tag = "1")]
        pub close_type: i32,
    }

    /// `commands/get_wifi_scan_info.proto`
    #[derive(Clone, PartialEq, Message)]
    pub struct GetWifiScanInfo {
        #[prost(int32, tag = "1")]
        pub interval: i32,
        #[prost(int32, tag = "2")]
        pub count: i32,
    }

    /// `wifi_scan_info.proto`
    #[derive(Clone, PartialEq, Message)]
    pub struct WifiScanInfo {
        #[prost(string, tag = "1")]
        pub bssid: String,
        #[prost(uint32, tag = "2")]
        pub frequency: u32,
        #[prost(int32, tag = "3")]
        pub signal_level: i32,
        #[prost(string, tag = "4")]
        pub flags: String,
        #[prost(string, tag = "5")]
        pub ssid: String,
    }

    /// `wifi_scan_info.proto`, notification 8249.
    #[derive(Clone, PartialEq, Message)]
    pub struct WifiScanInfoList {
        #[prost(message, repeated, tag = "1")]
        pub wifi_info: Vec<WifiScanInfo>,
        #[prost(int32, tag = "2")]
        pub list_count: i32,
    }
}

#[derive(Debug)]
pub enum Error {
    /// The Bluetooth stack (btleplug, BlueZ over D-Bus), as text.
    Os(String),
    Io(std::io::Error),
    NoAdapter,
    /// The adapter does not know the device (not seen by a scan or remembered by the OS).
    NoDevice,
    /// No Insta360 service (`be80`) with a notify and a read / write characteristic.
    NoService,
    Timeout,
    Closed,
    /// The byte stream lost its framing.
    Desync,
    /// A response status other than 200, with the camera's error body if it sent one. 505: busy switching Wi-Fi.
    Status(u16, Option<pb::Error>),
    Decode(prost::DecodeError),
    /// `CHECK_AUTHORIZATION` answered neither AUTHORIZED nor UNAUTHORIZED (`AuthorizationStatus`: 2 busy,
    /// 3–6 held by another phone / watch / bike computer / remote, 7 asleep).
    Unauthorized(i32),
    /// The answer on the camera's screen (`AuthorizationResult`: 1 rejected, 2 timed out, 3 busy).
    Refused(i32),
    /// Wake advertising needs BlueZ, and a serial suffix of at most 6 bytes.
    Unsupported,
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Os(e) => f.write_str(e),
            Error::Io(e) => write!(f, "{e}"),
            Error::NoAdapter => f.write_str("no Bluetooth adapter"),
            Error::NoDevice => f.write_str("device not known to the adapter"),
            Error::NoService => f.write_str("no Insta360 GATT service"),
            Error::Timeout => f.write_str("timeout"),
            Error::Closed => f.write_str("link closed"),
            Error::Desync => f.write_str("stream desync"),
            Error::Status(s, Some(e)) => write!(f, "status {s}: {:?} {}", e.code(), e.message),
            Error::Status(s, None) => write!(f, "status {s}"),
            Error::Decode(e) => write!(f, "{e}"),
            Error::Unauthorized(s) => write!(f, "not authorized ({s})"),
            Error::Refused(r) => write!(f, "refused on the camera ({r})"),
            Error::Unsupported => f.write_str("not supported here"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<prost::DecodeError> for Error {
    fn from(e: prost::DecodeError) -> Self {
        Error::Decode(e)
    }
}

pub(crate) fn os(e: impl fmt::Display) -> Error {
    Error::Os(e.to_string())
}

/// The result of a call, delivered by the BLE thread or a helper thread.
pub struct Pending<T>(Receiver<Result<T>>);

impl<T> Pending<T> {
    pub fn wait(self) -> Result<T> {
        self.0.recv().unwrap_or(Err(Error::Closed))
    }

    /// Non-blocking: `None` while the call runs.
    pub fn poll(&self) -> Option<Result<T>> {
        match self.0.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(Error::Closed)),
        }
    }
}

pub(crate) fn pend<T>() -> (SyncSender<Result<T>>, Pending<T>) {
    let (tx, rx) = mpsc::sync_channel(1);
    (tx, Pending(rx))
}

pub(crate) type Done = Box<dyn FnOnce(Result<Vec<u8>>) + Send>;

/// A peripheral as the adapter names it (BlueZ: `hci0/dev_AA_BB_CC_DD_EE_FF`; macOS: a UUID). It stays valid while
/// the OS remembers the device, so it can be stored to reconnect without a scan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dev(pub String);

impl fmt::Display for Dev {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One advertisement (merged with the scan response by the OS), with what the scan knows about its address.
///
/// An X5 alternates two advertisements on one address: `My BSA Bluetooth Device` with the `be80` service and
/// manufacturer record `000f: 00 00 00` (Broadcom's company id and its stack's default name), and `X5 <token>` with
/// record `1a2b: 00 tt vv`, its Wi-Fi state ([`Adv::proto`]). BlueZ's name follows whichever came last, so a single
/// advertisement does not identify the camera: use [`Adv::x5`], [`Adv::token`] and [`Adv::is`], which cover the whole
/// scan. Nothing in either says whether the camera sleeps.
#[derive(Debug, Clone)]
pub struct Adv {
    pub dev: Dev,
    /// Bluetooth address; zero on macOS.
    pub addr: String,
    /// The name as of this advertisement.
    pub name: Option<String>,
    pub rssi: Option<i16>,
    pub services: Vec<Uuid>,
    /// Manufacturer records by company id, lowest first (the app parses the lowest, PROTOCOL.md §1.3).
    pub mfr: Vec<(u16, Vec<u8>)>,
    /// Some advertisement from this address during this scan marked it as an X5 ([`Adv::mark`]).
    pub x5: bool,
    /// The last `X5 <token>` name this address advertised during this scan.
    pub x5name: Option<String>,
}

fn x5name(n: &str) -> bool {
    wire::model(n).is_some_and(|m| m.0 == "X5")
}

impl Adv {
    /// Advertises the Insta360 camera service: a UUID whose text contains `be80` (ble.md §2.2).
    pub fn insta(&self) -> bool {
        self.services.iter().any(|u| u.to_string().contains("be80"))
    }

    /// This advertisement alone marks its address as an X5: the name `X5 <token>`, the `be80` service or a
    /// manufacturer record with company id `1a2b`. The last two are not X5-specific (every Insta360 camera has
    /// `be80`).
    pub fn mark(&self) -> bool {
        self.name.as_deref().is_some_and(x5name)
            || self.insta()
            || self.mfr.iter().any(|m| m.0 == wire::INSTA)
    }

    /// The serial token (the serial's last six characters) of the last `X5 <token>` name; `None` until the scan saw
    /// one from this address.
    pub fn token(&self) -> Option<&str> {
        wire::model(self.x5name.as_deref()?).map(|m| m.1)
    }

    /// Whether this is the X5 with this serial (or serial suffix), by the last `X5 <token>` name of the scan.
    pub fn is(&self, serial: &str) -> bool {
        self.x5name.as_deref().is_some_and(|n| wire::is(n, serial))
    }

    /// The Wi-Fi state in the `X5 <token>` advertisement's `1a2b` record (BlueZ keeps the last record per company, so
    /// it survives the BSA advertisements in between).
    pub fn proto(&self) -> Option<Proto> {
        self.mfr
            .iter()
            .find(|m| m.0 == wire::INSTA)
            .and_then(|m| Proto::parse(&m.1))
    }

    /// The iBeacon in Apple's manufacturer record: an X5 asking for a transfer ([`Beacon::X5`]), or a phone waking a
    /// camera (major 0, UUID starting `\tORBIT`).
    pub fn beacon(&self) -> Option<Beacon> {
        self.mfr
            .iter()
            .find(|m| m.0 == wire::APPLE)
            .and_then(|m| Beacon::parse(&m.1))
    }
}

/// Notifications the BLE flows use, decoded; anything else arrives as `Other` (`link::Note::new` decodes most of
/// those: the schemas are the same on every link).
#[derive(Debug)]
pub enum Note {
    Auth(pb::NotificationAuthorizationResult),
    WifiStarted,
    WifiMode(msg::WifiModeResult),
    WifiStatus(msg::CameraWifiConnectionResult),
    WifiScan(msg::WifiScanInfoList),
    WifiClosed(msg::WifiCloseInfo),
    Other(u16, Vec<u8>),
}

impl Note {
    pub fn new(code: u16, b: &[u8]) -> Note {
        fn d<M: Message + Default>(b: &[u8], f: fn(M) -> Note) -> Option<Note> {
            M::decode(b).ok().map(f)
        }
        match code {
            code::AUTH_RESULT => d(b, Note::Auth),
            code::WIFI_START => Some(Note::WifiStarted),
            code::WIFI_MODE_CHANGE => d(b, Note::WifiMode),
            code::WIFI_STATUS => d(b, Note::WifiStatus),
            code::WIFI_SCAN_LIST => d(b, Note::WifiScan),
            code::WIFI_CLOSED => d(b, Note::WifiClosed),
            _ => None,
        }
        .unwrap_or_else(|| Note::Other(code, b.to_vec()))
    }
}

pub enum Event {
    /// An advertisement while scanning, from any device: [`Adv::x5`] picks the cameras, [`Adv::is`] one camera.
    Seen(Adv),
    Note(Dev, Note),
    /// The link to this camera is gone (not sent after [`Camera::close`] or dropping the last handle).
    Closed(Dev, Error),
}

/// Progress of [`Ble::wake_up`], in order.
#[derive(Debug)]
pub enum Step {
    /// The wake advertisement is up, or could not start (the flow goes on without it).
    Beacon(Option<Error>),
    /// The camera advertises ([`Adv::proto`]: its Wi-Fi state).
    Seen(Adv),
    /// The connect failed; the flow looks for the camera again.
    Retry(Error),
    Connected(Link),
    /// The identity `GET_OPTIONS[48, 15, 11, 198]`: the camera's software answers.
    Options(pb::Options),
    /// The app's option sync on BLE, `bt_wakeup_sw` among it.
    Synced(Result<pb::Options>),
    /// A `CHECK_AUTHORIZATION` answer: 0 authorized, 1 waits for the user, 7 asleep (asked again every 2 s).
    Auth(i32),
    /// No 8215 within the time: `OPEN_CAMERA_WIFI` sent.
    OpenWifi,
    /// 8215, the camera's Wi-Fi is up: the app's "camera ready". The wake advertisement stops next.
    Ready,
}

/// The GATT pipe a [`Camera`] runs on.
#[derive(Debug, Clone, Default)]
pub struct Link {
    pub service: Uuid,
    /// Read / write characteristic, written with Write Requests.
    pub tx: Uuid,
    /// Notify characteristic.
    pub rx: Uuid,
    /// ATT MTU as the OS reports it (23 when unknown); writes are chunked at `min(206, mtu − 3)`.
    pub mtu: u16,
    /// UCP "Go2BLE" framing instead of plain UCD (expected: plain).
    pub ucp: bool,
    /// The camera's first notification after subscribing and how long it took (PROTOCOL.md §11 X13); `None` if
    /// nothing came within 5 s.
    pub first: Option<(Duration, Vec<u8>)>,
}

/// Handle to the adapter's thread. Clones share it; the thread ends when the last clone is dropped and every
/// camera is closed.
#[derive(Clone)]
pub struct Ble(UnboundedSender<io::Cmd>);

impl Ble {
    /// The first Bluetooth adapter.
    pub fn open() -> Result<(Ble, Receiver<Event>)> {
        let (tx, rx) = unbounded_channel();
        let (etx, erx) = mpsc::channel();
        let (rtx, rrx) = pend();
        thread::Builder::new()
            .name("x5-ble".into())
            .spawn(move || io::run(rx, etx, rtx))?;
        rrx.wait()?;
        Ok((Ble(tx), erx))
    }

    /// Starts or stops scanning (no filter, like the app; BlueZ: LE only). Every advertisement arrives as
    /// [`Event::Seen`].
    pub fn scan(&self, on: bool) -> Pending<()> {
        let (tx, p) = pend();
        let _ = self.0.send(io::Cmd::Scan(on, tx));
        p
    }

    /// Connects (20 s), finds the camera service and characteristics, subscribes, and picks the framing from the
    /// first notification (up to 5 s). No pairing: the app never bonds an X5. BlueZ: a cached record of the camera's
    /// classic side, which would make BlueZ connect over BR/EDR, is dropped first and the camera found again (≤ 10 s).
    pub fn connect(&self, d: &Dev) -> Pending<Camera> {
        let (tx, p) = pend();
        let _ = self.0.send(io::Cmd::Connect(d.clone(), tx));
        p
    }

    /// Scans until the X5 with this serial (or its token, the last six characters) advertises, at most `t`. The scan
    /// stops afterwards unless [`Ble::scan`] asked for one.
    pub fn find(&self, serial: &str, t: Duration) -> Pending<Adv> {
        let (tx, p) = pend();
        let _ = self
            .0
            .send(io::Cmd::Find(serial.into(), Instant::now() + t, tx));
        p
    }

    /// Wakes the X5 with this serial (or token) and connects the way the app does (PROTOCOL.md §2.6, §2.7), calling
    /// `step` with each [`Step`] on the flow's thread (the flow waits while it runs):
    ///
    /// 1. The wake advertisement for the token ([`wake`]); without it (not BlueZ) the flow goes on.
    /// 2. Find the camera, wait `hold` with only the advertisement up (the app: 0), connect. A failed connect looks
    ///    for the camera again until `t` is over.
    /// 3. The app's BLE session up to "camera ready", with the advertisement still up (the app stops it only when it
    ///    starts joining the camera's Wi-Fi): identity `GET_OPTIONS`, the option sync (the app's
    ///    `GET_CURRENT_CAPTURE_STATUS` never goes out to an X5 on BLE), `CHECK_AUTHORIZATION` with the client's `uuid`
    ///    (asked again every 2 s while the camera answers 7, asleep; UNAUTHORIZED waits up to `t` for the user, as
    ///    [`Camera::authorize`]), then 8215 within `t`: one seen earlier counts, else `OPEN_CAMERA_WIFI` once and
    ///    another `t`.
    /// 4. 8215 is the app's criterion for a camera that is up. The advertisement stops and the camera is returned,
    ///    ready for [`Camera::ap`].
    ///
    /// The app advertises only for a camera missing from its scan; the X5 keeps advertising in standby, so this flow
    /// always advertises, which costs an awake camera nothing. Dropping the result does not cut the flow short: it
    /// ends after at most about `4 t`, a few response timeouts (10 s), the connects and the user's answer.
    pub fn wake_up(
        &self,
        serial: &str,
        uuid: &str,
        hold: Duration,
        t: Duration,
        mut step: impl FnMut(Step) + Send + 'static,
    ) -> Pending<Camera> {
        let (tx, p) = pend();
        let (b, s, u) = (self.clone(), serial.to_string(), uuid.to_string());
        let t2 = tx.clone();
        if let Err(e) = thread::Builder::new()
            .name("x5-ble-wake".into())
            .spawn(move || drop(t2.send(woken(&b, &s, &u, hold, t, &mut step))))
        {
            let _ = tx.send(Err(e.into()));
        }
        p
    }
}

/// [`Ble::wake_up`] on this thread.
fn woken(
    b: &Ble,
    serial: &str,
    uuid: &str,
    mut hold: Duration,
    t: Duration,
    step: &mut dyn FnMut(Step),
) -> Result<Camera> {
    let end = Instant::now() + t;
    let token = serial
        .get(serial.len().saturating_sub(6)..)
        .unwrap_or(serial);
    let w = match wake(token) {
        Ok(w) => {
            step(Step::Beacon(None));
            Some(w)
        }
        Err(e) => {
            step(Step::Beacon(Some(e)));
            None
        }
    };
    let c = loop {
        let a = b
            .find(serial, end.saturating_duration_since(Instant::now()))
            .wait()?;
        let d = a.dev.clone();
        step(Step::Seen(a));
        thread::sleep(std::mem::take(&mut hold));
        match b.connect(&d).wait() {
            Ok(c) => break c,
            Err(e) if Instant::now() < end => step(Step::Retry(e)),
            Err(e) => return Err(e),
        }
    };
    step(Step::Connected(c.link().clone()));
    session(&c, uuid, t, step)?;
    drop(w);
    Ok(c)
}

/// The app's option sync on BLE (`CameraSupportConfig.getSupportOptionsListV2`): firmware, activation time, channel
/// list, media offset, sensor info, `bt_wakeup_sw`, BLE and box versions, OTA versions, charge box, cloud busy,
/// splicing, offsets, heat shell, battery, quick reader, P2P, watermark, dash cam, screen battery, storage, idle
/// analysis, live broadcast.
const SYNC: [i32; 26] = [
    30, 19, 37, 22, 67, 97, 89, 116, 115, 114, 110, 85, 156, 135, 103, 150, 149, 11, 147, 167, 157,
    158, 180, 20, 181, 165,
];
/// How often the wake-up asks again while the camera answers 7 (asleep); the app has no handling for 7.
const ASLEEP: Duration = Duration::from_secs(2);

/// The app's BLE session up to "camera ready" (PROTOCOL.md §2.7 steps 2–6), on this thread.
fn session(c: &Camera, uuid: &str, t: Duration, step: &mut dyn FnMut(Step)) -> Result<()> {
    step(Step::Options(c.options(&[48, 15, 11, 198]).wait()?));
    // The app fails the connect on errors here; for waking only a lost link matters. Its GET_CURRENT_CAPTURE_STATUS
    // goes out on BLE only for the GO Ultra and GO 3S (`CameraSupportConfig.supportGetCaptureStatus`): for an X5 the
    // controller answers −431 itself.
    let o = c.options(&SYNC).wait();
    if let Err(Error::Closed) = o {
        return Err(Error::Closed);
    }
    step(Step::Synced(o));
    let end = Instant::now() + t;
    loop {
        match auth(c, uuid, t, step) {
            Err(Error::Unauthorized(7)) if Instant::now() < end => thread::sleep(ASLEEP),
            r => break r?,
        }
    }
    match c.note(code::WIFI_START, t, false).wait() {
        Err(Error::Timeout) => {
            step(Step::OpenWifi);
            let w = c.note(code::WIFI_START, T + t, true);
            c.open_wifi(0).wait()?;
            w.wait()?;
        }
        r => drop(r?),
    }
    step(Step::Ready);
    Ok(())
}

/// [`Camera::authorize`] on this thread, reporting the camera's status.
fn auth(c: &Camera, uuid: &str, t: Duration, step: &mut dyn FnMut(Step)) -> Result<()> {
    let end = Instant::now() + t;
    let mut w = c.note(code::AUTH_RESULT, t, true);
    let s = c.check_auth(uuid).wait()?.authorization_status;
    step(Step::Auth(s));
    match s {
        0 => return Ok(()),
        1 => {}
        s => return Err(Error::Unauthorized(s)),
    }
    loop {
        let n = pb::NotificationAuthorizationResult::decode(&w.wait()?[..])?;
        match (n.authorization_operation_type, n.authorization_result) {
            (0, 0) => return Ok(()),
            (0, r) if r != 4 => return Err(Error::Refused(r)),
            _ => {}
        }
        let left = end.saturating_duration_since(Instant::now());
        w = c.note(code::AUTH_RESULT, left, false)
    }
}

struct Inner {
    tx: UnboundedSender<Op>,
    dev: Dev,
    link: Link,
    /// The caller's handles own the link; a flow's helper thread gets one that does not, so that neither an
    /// abandoned flow nor a helper finishing after its caller keeps the link up.
    own: bool,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let (tx, rx) = mpsc::sync_channel(1);
        // On the BLE thread (a connect whose caller stopped waiting) the link task closes after this returns.
        if self.own && self.tx.send(Op::Close(Some(tx))).is_ok() && !io::BLE.get() {
            let _ = rx.recv_timeout(BYE);
        }
    }
}

/// A connected camera. Dropping the last handle disconnects and waits for it ([`Camera::close`]); a flow still
/// running ([`Camera::authorize`], [`Camera::ap`], …) does not keep the link and ends with [`Error::Closed`].
#[derive(Clone)]
pub struct Camera(Arc<Inner>);

/// Default response timeout (the app: 8 s, two retries).
const T: Duration = Duration::from_secs(10);
/// How long dropping the last [`Camera`] waits for the disconnect. BlueZ answers `Disconnect` once the link is down;
/// a link task stuck in a write to a hung camera only gets there after the ATT timeout (30 s).
const BYE: Duration = Duration::from_secs(5);

fn dec<M: Message + Default>(b: Vec<u8>) -> Result<M> {
    Ok(M::decode(&b[..])?)
}

fn unit(_: Vec<u8>) -> Result<()> {
    Ok(())
}

impl Camera {
    pub(crate) fn new(tx: UnboundedSender<Op>, dev: Dev, link: Link) -> Camera {
        Camera(Arc::new(Inner {
            tx,
            dev,
            link,
            own: true,
        }))
    }

    pub fn dev(&self) -> &Dev {
        &self.0.dev
    }

    pub fn link(&self) -> &Link {
        &self.0.link
    }

    fn send<R: Send + 'static>(
        &self,
        code: u16,
        body: Vec<u8>,
        t: Duration,
        f: fn(Vec<u8>) -> Result<R>,
    ) -> Pending<R> {
        let (tx, p) = pend();
        let _ = self.0.tx.send(Op::Req(
            code,
            body,
            t,
            Box::new(move |r| drop(tx.send(r.and_then(f)))),
        ));
        p
    }

    /// Any request; the raw response body.
    pub fn raw(&self, code: u16, body: Vec<u8>, t: Duration) -> Pending<Vec<u8>> {
        self.send(code, body, t, Ok)
    }

    /// Any request with a protobuf response.
    pub fn call<R: Message + Default + Send + 'static>(
        &self,
        code: u16,
        q: &impl Message,
    ) -> Pending<R> {
        self.send(code, q.encode_to_vec(), T, dec::<R>)
    }

    /// The body of the next notification `code` within `t`. Without `fresh`, one that arrived earlier and nobody
    /// took counts (the app treats an 8215 seen at any time as "AP up").
    pub fn note(&self, code: u16, t: Duration, fresh: bool) -> Pending<Vec<u8>> {
        let (tx, p) = pend();
        let _ = self.0.tx.send(Op::Note(
            code,
            t,
            fresh,
            Box::new(move |r| drop(tx.send(r))),
        ));
        p
    }

    /// Runs `f` on a helper thread.
    fn flow<R: Send + 'static>(
        &self,
        f: impl FnOnce(&Camera) -> Result<R> + Send + 'static,
    ) -> Pending<R> {
        let (tx, p) = pend();
        let i = &self.0;
        let c = Camera(Arc::new(Inner {
            tx: i.tx.clone(),
            dev: i.dev.clone(),
            link: i.link.clone(),
            own: false,
        }));
        let t = tx.clone();
        if let Err(e) = thread::Builder::new()
            .name("x5-ble-flow".into())
            .spawn(move || drop(t.send(f(&c))))
        {
            let _ = tx.send(Err(e.into()));
        }
        p
    }

    /// `GET_OPTIONS`: the camera's [`pb::Options`] with the given `OptionType`s set (identity: `[48, 15, 11, 198]`).
    pub fn options(&self, types: &[i32]) -> Pending<pb::Options> {
        let q = pb::GetOptions {
            option_types: types.into(),
        };
        self.send(code::GET_OPTIONS, q.encode_to_vec(), T, |b| {
            Ok(dec::<pb::GetOptionsResp>(b)?.value.unwrap_or_default())
        })
    }

    /// `SET_OPTIONS`, e.g. `bt_wakeup_sw` (97) = 1 so that [`wake`] works once the camera sleeps.
    pub fn set_options(&self, types: &[i32], o: pb::Options) -> Pending<()> {
        let q = pb::SetOptions {
            option_types: types.into(),
            value: Some(o),
        };
        self.send(code::SET_OPTIONS, q.encode_to_vec(), T, unit)
    }

    /// `CHECK_AUTHORIZATION` with the client's persistent id (the same one the Wi-Fi session uses), as an Android
    /// phone like `link`. UNAUTHORIZED makes the camera ask its user.
    pub fn check_auth(&self, uuid: &str) -> Pending<pb::CheckAuthorizationResp> {
        let q = pb::CheckAuthorization {
            id: uuid.into(),
            initiator_type: 2,
            ..Default::default()
        };
        self.call(code::CHECK_AUTH, &q)
    }

    /// The app's authorization step (PROTOCOL.md §5.2): `CHECK_AUTHORIZATION`; on UNAUTHORIZED wait up to `t` for the
    /// user's answer on the camera (8209 for BLE_CONNECT; PENDING keeps waiting; an empty body is SUCCESS).
    pub fn authorize(&self, uuid: &str, t: Duration) -> Pending<()> {
        let uuid = uuid.to_string();
        self.flow(move |c| auth(c, &uuid, t, &mut |_| {}))
    }

    /// `CANCEL_AUTHORIZATION`: withdraws the prompt on the camera.
    pub fn cancel_auth(&self) -> Pending<()> {
        self.send(code::CANCEL_AUTH, vec![], T, unit)
    }

    /// The access point's SSID, password (plaintext on the X5), channel, mode and state: `GET_OPTIONS[36, 37, 43]`.
    pub fn wifi(&self) -> Pending<pb::options::WifiInfo> {
        let q = pb::GetOptions {
            option_types: vec![36, 37, 43],
        };
        self.send(code::GET_OPTIONS, q.encode_to_vec(), T, |b| {
            let o = dec::<pb::GetOptionsResp>(b)?.value.unwrap_or_default();
            Ok(o.wifi_info.unwrap_or_default())
        })
    }

    /// `GET_WIFI_MODE`: [`msg::AP`], [`msg::STA`], …
    pub fn wifi_mode(&self) -> Pending<i32> {
        self.send(code::GET_WIFI_MODE, vec![], T, |b| {
            Ok(dec::<msg::GetWifiModeResp>(b)?.wifi_mode)
        })
    }

    /// `SET_WIFI_MODE`; `country` (ISO code) may be empty. The camera then sends 8247 and, for AP, 8215.
    pub fn set_wifi_mode(&self, mode: i32, country: &str) -> Pending<()> {
        let q = msg::SetWifiMode {
            place_holder: 1,
            wifi_mode: mode,
            country_code: country.into(),
        };
        self.send(code::SET_WIFI_MODE, q.encode_to_vec(), T, unit)
    }

    /// `OPEN_CAMERA_WIFI`, on `channel` if not 0. The app only uses it to reset the channel or from its settings.
    pub fn open_wifi(&self, channel: u32) -> Pending<()> {
        let q = msg::RestartWifi {
            restart_channel: channel,
        };
        let b = if channel > 0 {
            q.encode_to_vec()
        } else {
            vec![]
        };
        self.send(code::OPEN_WIFI, b, T, unit)
    }

    /// `GET_WIFI_CONNECTION_INFO`: the camera's station-mode connection (SSID, address), not its own AP.
    pub fn wifi_conn(&self) -> Pending<msg::WifiConnectionInfo> {
        self.send(code::GET_WIFI_CONN, vec![], T, |b| {
            Ok(dec::<msg::GetWifiConnectionInfoResp>(b)?
                .wifi_connection_info
                .unwrap_or_default())
        })
    }

    /// The app's hand-over to Wi-Fi (PROTOCOL.md §2.7, steps 6–8), after [`Camera::authorize`]: wait up to `t` for
    /// 8215 (AP up; one seen earlier counts, else `OPEN_CAMERA_WIFI` once and wait again), make sure the mode is AP
    /// (`SET_WIFI_MODE` and a new 8215, at least 50 s when the camera is busy switching), then read the
    /// credentials. Join the SSID and use `link::Camera::wifi` with the same UUID.
    pub fn ap(&self, t: Duration) -> Pending<pb::options::WifiInfo> {
        self.flow(move |c| {
            match c.note(code::WIFI_START, t, false).wait() {
                Err(Error::Timeout) => {
                    let w = c.note(code::WIFI_START, T + t, true);
                    c.open_wifi(0).wait()?;
                    w.wait()?;
                }
                r => drop(r?),
            }
            if c.wifi_mode().wait()? != msg::AP {
                let w = c.note(code::WIFI_START, T + t.max(Duration::from_secs(50)), true);
                match c.set_wifi_mode(msg::AP, "").wait() {
                    Err(Error::Status(505, _)) | Ok(()) => drop(w.wait()?),
                    Err(e) => return Err(e),
                }
            }
            c.wifi().wait()
        })
    }

    /// Station mode, the app's net-prov pipeline (PROTOCOL.md §2.9): `SET_WIFI_MODE{STA}`, `PHONE_INFO`,
    /// `SET_WIFI_CONNECTION_INFO{ssid, password}`, then the camera's join result (8232) within `t` (the app: 40 s).
    /// `PHONE_INFO` says device UNKNOWN, manufacturer "x5desk", the OS as version (the app sends its Android build
    /// values; whether the camera cares is unknown).
    pub fn station(
        &self,
        ssid: &str,
        password: &str,
        t: Duration,
    ) -> Pending<msg::CameraWifiConnectionResult> {
        let q = msg::SetWifiConnectionInfo {
            wifi_connection_info: Some(msg::WifiConnectionInfo {
                ssid: ssid.into(),
                password: password.into(),
                ..Default::default()
            }),
        };
        self.flow(move |c| {
            let w = c.note(code::WIFI_STATUS, 3 * T + t, true);
            c.set_wifi_mode(msg::STA, "").wait()?;
            let p = pb::PhoneInfo {
                manufacturer: "x5desk".into(),
                os_version: std::env::consts::OS.into(),
                ..Default::default()
            };
            c.send(code::PHONE_INFO, p.encode_to_vec(), T, unit)
                .wait()?;
            c.send(code::SET_WIFI_CONN, q.encode_to_vec(), T, unit)
                .wait()?;
            dec(w.wait()?)
        })
    }

    /// `GET_WIFI_SCAN_LIST` (interval 0, count 1, as the app): the networks the camera sees, from 8249 within `t`.
    pub fn wifi_scan(&self, t: Duration) -> Pending<msg::WifiScanInfoList> {
        self.flow(move |c| {
            let w = c.note(code::WIFI_SCAN_LIST, T + t, true);
            let q = msg::GetWifiScanInfo {
                interval: 0,
                count: 1,
            };
            c.send(code::WIFI_SCAN, q.encode_to_vec(), T, unit).wait()?;
            dec(w.wait()?)
        })
    }

    /// Disconnects, whatever other handles exist; pending calls end with [`Error::Closed`]. Answers once BlueZ has the
    /// link down.
    pub fn close(&self) -> Pending<()> {
        let (tx, p) = pend();
        let _ = self.0.tx.send(Op::Close(Some(tx)));
        p
    }
}

/// The phone-side wake advertisement for a sleeping camera (PROTOCOL.md §2.6), running until dropped. The camera
/// needs `bt_wakeup_sw` on. The app scans for `X5 <suffix>` meanwhile (it gives up after 30 s) and keeps advertising
/// through the BLE session until the AP join starts; [`Ble::wake_up`] does all of it.
pub struct Wake {
    stop: Arc<AtomicBool>,
    t: Option<JoinHandle<()>>,
}

impl Drop for Wake {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(t) = self.t.take() {
            let _ = t.join();
        }
    }
}

/// Starts the wake advertisement for the camera whose serial ends in `suffix` (its BLE name's token); returns once
/// the adapter advertises.
#[cfg(target_os = "linux")]
pub fn wake(suffix: &str) -> Result<Wake> {
    wake::start(suffix)
}

/// Starts the wake advertisement. Not available here: btleplug only scans and connects, CoreBluetooth cannot
/// advertise manufacturer data, and the Windows publisher is not wired up yet.
#[cfg(not(target_os = "linux"))]
pub fn wake(_: &str) -> Result<Wake> {
    Err(Error::Unsupported)
}

#[cfg(test)]
impl Dev {
    pub(crate) fn test() -> Dev {
        Dev("hci0/dev_00_00_00_00_00_00".into())
    }
}

#[cfg(test)]
impl Adv {
    pub(crate) fn test(name: &str) -> Adv {
        Adv {
            dev: Dev::test(),
            addr: "00:00:00:00:00:00".into(),
            name: Some(name.into()),
            rssi: Some(-60),
            services: vec![],
            mfr: vec![],
            x5: false,
            x5name: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;

    #[test]
    fn adv() {
        let mut a = Adv::test("X5 ABCDEF");
        a.services = vec![Uuid::from_u128(0x0000be80_0000_1000_8000_00805f9b34fb)];
        assert!(a.insta() && a.mark());
        // Token and serial match come from the scan's record, not from this advertisement's name.
        assert_eq!((a.token(), a.is("ABCDEF")), (None, false));
        a.x5name = Some("X5 ABCDEF".into());
        assert_eq!(a.token(), Some("ABCDEF"));
        assert!(a.is("IAHxx00ABCDEF") && !a.is("IAHxx00ABCDEG"));
        assert!(a.beacon().is_none());
        a.name = Some("X4 Air ABCDEF".into());
        assert!(a.mark());
        a.services.clear();
        assert!(!a.insta() && !a.mark());
        a.name = None;
        a.mfr = vec![(wire::INSTA, vec![0, 0x78, 1])];
        assert!(a.mark());
        assert_eq!(a.proto(), Some(Proto { wifi: 120, pwd: 1 }));
        // BlueZ's merged view of both advertisements: the `1a2b` record, not the lowest company's.
        a.mfr = vec![(0x000F, vec![0, 0, 0]), (wire::INSTA, vec![0, 0x52, 1])];
        assert_eq!(a.proto(), Some(Proto { wifi: 82, pwd: 1 }));
        a.mfr = vec![(wire::APPLE, wire::wake("ABCDEF").unwrap().to_vec())];
        assert!(!a.mark());
        assert_eq!(a.proto(), None);
        assert_eq!(a.beacon().unwrap().major, 0);
    }

    /// A link task that answers `Close` after `t`, or never (holding the reply) when `t` is `None`.
    fn fake(t: Option<Duration>) -> (Camera, Arc<AtomicBool>, thread::JoinHandle<()>) {
        let (tx, mut ops) = unbounded_channel();
        let down = Arc::new(AtomicBool::new(false));
        let d = down.clone();
        let j = thread::spawn(move || {
            // Requests and waits stay unanswered (their callbacks alive) until the task ends.
            let mut held = vec![];
            while let Some(o) = ops.blocking_recv() {
                let Op::Close(Some(c)) = o else {
                    held.push(o);
                    continue;
                };
                match t {
                    Some(t) => {
                        thread::sleep(t);
                        d.store(true, Relaxed);
                        let _ = c.send(Ok(()));
                        return;
                    }
                    None => held.push(Op::Close(Some(c))),
                }
            }
        });
        (Camera::new(tx, Dev::test(), Link::default()), down, j)
    }

    #[test]
    fn close_on_drop() {
        // The last handle's drop returns only after the disconnect; other handles' drops do not wait.
        let (c, down, j) = fake(Some(Duration::from_millis(300)));
        let t0 = Instant::now();
        drop(c.clone());
        assert!(t0.elapsed() < Duration::from_millis(100) && !down.load(Relaxed));
        drop(c);
        assert!(down.load(Relaxed));
        assert!(t0.elapsed() >= Duration::from_millis(300));
        j.join().unwrap();

        // An explicit close answers once the link is down; the later drop finds the task gone and returns at once.
        let (c, down, j) = fake(Some(Duration::from_millis(100)));
        c.close().wait().unwrap();
        assert!(down.load(Relaxed));
        j.join().unwrap();
        let t0 = Instant::now();
        drop(c);
        assert!(t0.elapsed() < Duration::from_millis(100));

        // A flow running on its helper thread does not hold the link: the caller's drop closes it, and the flow
        // ends with `Closed`.
        let (c, down, j) = fake(Some(Duration::from_millis(300)));
        let a = c.ap(Duration::from_secs(30));
        thread::sleep(Duration::from_millis(50));
        let t0 = Instant::now();
        drop(c);
        assert!(down.load(Relaxed) && t0.elapsed() >= Duration::from_millis(300));
        j.join().unwrap();
        assert!(matches!(a.wait(), Err(Error::Closed)));

        // A link task that never answers: the drop gives up after `BYE`.
        let (c, down, _j) = fake(None);
        let t0 = Instant::now();
        drop(c);
        let e = t0.elapsed();
        assert!(e >= BYE && e < BYE + Duration::from_secs(1), "{e:?}");
        assert!(!down.load(Relaxed));

        // On the BLE thread (a connect nobody waited for) the drop only queues the close.
        let (c, down, j) = fake(Some(Duration::from_millis(300)));
        let t0 = Instant::now();
        thread::spawn(move || {
            io::BLE.set(true);
            drop(c)
        })
        .join()
        .unwrap();
        assert!(t0.elapsed() < Duration::from_millis(100) && !down.load(Relaxed));
        j.join().unwrap();
        assert!(down.load(Relaxed));
    }

    #[test]
    fn notes() {
        assert!(matches!(
            Note::new(code::WIFI_START, &[]),
            Note::WifiStarted
        ));
        // Empty 8209: the proto3 encoding of {SUCCESS, BLE_CONNECT} (PROTOCOL.md §5.2).
        let Note::Auth(a) = Note::new(code::AUTH_RESULT, &[]) else {
            panic!()
        };
        assert_eq!(
            (a.authorization_result, a.authorization_operation_type),
            (0, 0)
        );
        let Note::WifiMode(m) = Note::new(code::WIFI_MODE_CHANGE, &[8, 1]) else {
            panic!()
        };
        assert_eq!(m.wifi_mode, msg::STA);
        assert!(matches!(
            Note::new(code::AUTH_RESULT, &[0xFF]),
            Note::Other(8209, _)
        ));
        assert!(matches!(Note::new(8195, &[]), Note::Other(8195, _)));
    }
}
