//! Camera link for the Insta360 X5 (`../insta360/re/PROTOCOL.md`): UCD v1 framing and session over USB (nusb) or
//! Wi-Fi (TCP), HTTP through the USB socket tunnel or straight to the camera, downloads over several connections
//! with resume, and a typed API.
//!
//! All I/O of one camera runs on its own thread. A connection starts as an [`Opening`] (dropped before it is ready,
//! it cancels the sync); [`Camera`] is a cheap, cloneable handle; every call returns a [`Pending`] that the caller
//! can block on or poll. Notifications, live-preview packets and the loss of the link arrive on the [`Event`]
//! receiver returned with the handle. The session closes when the last handle is dropped or on [`Camera::close`].

mod http;
mod io;
#[cfg(feature = "sim")]
mod sim;
mod tcp;
mod ucd;
mod usb;

pub use http::Prog;
pub use proto as pb;
#[cfg(feature = "sim")]
pub use sim::Sim;
pub use usb::{Plug, scan, watch};

use prost::Message;
use std::{
    fmt,
    net::IpAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender, SyncSender, TryRecvError},
    },
    thread,
    time::Duration,
};

/// Request codes (PROTOCOL.md §6.1).
pub mod code {
    pub const START_LIVE: u16 = 1;
    pub const STOP_LIVE: u16 = 2;
    pub const TAKE_PICTURE: u16 = 3;
    pub const START_CAPTURE: u16 = 4;
    pub const STOP_CAPTURE: u16 = 5;
    pub const SET_OPTIONS: u16 = 7;
    pub const GET_OPTIONS: u16 = 8;
    pub const SET_PHOTO_OPTIONS: u16 = 9;
    pub const GET_PHOTO_OPTIONS: u16 = 10;
    pub const GET_FILE_EXTRA: u16 = 11;
    pub const DELETE_FILES: u16 = 12;
    pub const GET_FILE_LIST: u16 = 13;
    pub const CAPTURE_STATUS: u16 = 15;
    pub const ERASE_SD: u16 = 24;
    pub const GET_FILEINFO_LIST: u16 = 38;
    pub const CHECK_AUTH: u16 = 39;
    pub const START_TIMELAPSE: u16 = 22;
    pub const STOP_TIMELAPSE: u16 = 23;
    pub const KEY_TIME_POINT: u16 = 60;
    pub const GET_FILE: u16 = 83;
    pub const SET_FAVORITE: u16 = 176;
    pub const GET_EDITINFO_LIST: u16 = 201;
    pub const FORMAT_STORAGE: u16 = 232;
}

/// Live-preview packet kinds (§4.2, §8.11).
pub mod stream {
    /// H.264 Annex-B, one access unit per packet, both fisheyes side by side.
    pub const VIDEO: u8 = 0x20;
    /// Raw IMU batches.
    pub const GYRO: u8 = 0x30;
    /// Per-frame exposure record.
    pub const EXPOSURE: u8 = 0x40;
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// No camera in USB app mode.
    NoDevice,
    /// Another client holds the camera's session.
    Busy,
    /// The byte stream lost its framing.
    Desync,
    Timeout,
    /// No answer to the sync in 30 s (bytes received meanwhile): the camera is enumerated but asleep or hung.
    Silent(usize),
    Closed,
    /// A response status other than 200, with the camera's error body if it sent one.
    Status(u16, Option<pb::Error>),
    Decode(prost::DecodeError),
    Http(String),
    /// Wi-Fi authorization refused or not given (`AuthorizeReply::status` / `AuthNote::outcome`).
    Unauthorized(i32),
    Cancelled,
    /// The camera's file changed since the partial download; the partial file was removed.
    Changed,
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::NoDevice => f.write_str("no camera in USB app mode"),
            Error::Busy => f.write_str("camera busy: another client holds the session"),
            Error::Desync => f.write_str("stream desync"),
            Error::Timeout => f.write_str("timeout"),
            Error::Silent(n) => write!(f, "no answer to the sync in 30 s ({n} bytes received)"),
            Error::Closed => f.write_str("session closed"),
            Error::Status(s, Some(e)) => write!(f, "status {s}: {:?} {}", e.code(), e.text),
            Error::Status(s, None) => write!(f, "status {s}"),
            Error::Decode(e) => write!(f, "{e}"),
            Error::Http(e) => write!(f, "HTTP: {e}"),
            Error::Unauthorized(s) => write!(f, "not authorized ({s})"),
            Error::Cancelled => f.write_str("cancelled"),
            Error::Changed => f.write_str("file changed on the camera"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<nusb::Error> for Error {
    fn from(e: nusb::Error) -> Self {
        Error::Io(e.into())
    }
}

impl From<prost::DecodeError> for Error {
    fn from(e: prost::DecodeError) -> Self {
        Error::Decode(e)
    }
}

pub enum Event {
    Note(Note),
    Frame(Frame),
    /// The link is gone (unplugged, desync, silence); the session has ended.
    Closed(Error),
}

/// A stream packet (type 1): [`stream`] kind, camera clock in ms, payload, when the I/O thread framed it. The first
/// video packets after a start may carry `ts` 0 (0 or 5 of them seen on the X5, fw v1.13.21); exposure records carry
/// their frame's `ts`.
pub struct Frame {
    pub kind: u8,
    pub ts: u64,
    pub data: Vec<u8>,
    pub at: std::time::Instant,
}

/// One sample of a [`stream::GYRO`] packet: camera clock in ms, accelerometer (g; |acc| ≈ 1 at rest), gyroscope.
/// The X5 sends 40 samples per packet, about 500 per second.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Imu {
    pub ts: i64,
    pub acc: [f64; 3],
    pub gyro: [f64; 3],
}

impl Frame {
    /// The samples of a gyro packet: 56-byte items `i64 ts, f64 ax, ay, az, gx, gy, gz`.
    pub fn imu(&self) -> impl Iterator<Item = Imu> + '_ {
        self.data.as_chunks::<56>().0.iter().map(|c| {
            let f = |i: usize| f64::from_le_bytes(c[8 + 8 * i..16 + 8 * i].try_into().unwrap());
            Imu {
                ts: i64::from_le_bytes(c[..8].try_into().unwrap()),
                acc: [f(0), f(1), f(2)],
                gyro: [f(3), f(4), f(5)],
            }
        })
    }
}

/// Notifications (§6.2), decoded where the schema is known; anything else, or a body that does not decode,
/// arrives as `Other`.
#[derive(Clone, Debug)]
pub enum Note {
    AutoSplit(pb::SplitNote),
    Battery(pb::Battery),
    BatteryLow(pb::Battery),
    Shutdown(pb::ShutdownNote),
    Storage(pb::CardNote),
    StorageFull,
    CaptureStopped(pb::StoppedNote),
    PictureState(pb::PhotoNote),
    CaptureStatus(pb::CaptureNote),
    Auth(pb::AuthNote),
    /// Temperature level (8214).
    Temp(pb::TempNote),
    WifiStarted,
    Posture(pb::PostureNote),
    RecordStatus(pb::RecordNote),
    LiveParams(pb::LiveNote),
    SubMode(pb::ModeNote),
    DeleteResult(pb::DeleteNote),
    StorageLocation(pb::VolumeNote),
    Photo(pb::TakePhotoReply),
    PhotoOptions(pb::ParamsNote),
    Other(u16, Vec<u8>),
}

impl Note {
    pub fn new(code: u16, b: &[u8]) -> Note {
        fn d<M: Message + Default>(b: &[u8], f: fn(M) -> Note) -> Option<Note> {
            M::decode(b).ok().map(f)
        }
        let n = match code {
            8194 => d(b, Note::AutoSplit),
            8195 => d(b, |m: pb::BatteryNote| {
                Note::Battery(m.battery.unwrap_or_default())
            }),
            8196 => d(b, |m: pb::BatteryLowNote| {
                Note::BatteryLow(m.battery.unwrap_or_default())
            }),
            8197 => d(b, Note::Shutdown),
            8198 => d(b, Note::Storage),
            8199 => Some(Note::StorageFull),
            8201 => d(b, Note::CaptureStopped),
            8202 => d(b, Note::PictureState),
            8208 => d(b, Note::CaptureStatus),
            8209 => d(b, Note::Auth),
            8214 => d(b, Note::Temp),
            8215 => Some(Note::WifiStarted),
            8219 => d(b, Note::Posture),
            8230 => d(b, Note::RecordStatus),
            8234 => d(b, Note::LiveParams),
            8275 => d(b, Note::SubMode),
            8279 => d(b, Note::DeleteResult),
            8305 => d(b, Note::StorageLocation),
            8311 => d(b, Note::Photo),
            8334 => d(b, Note::PhotoOptions),
            _ => None,
        };
        n.unwrap_or_else(|| Note::Other(code, b.to_vec()))
    }
}

/// The result of a call, delivered by the I/O thread or a helper thread.
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

/// A result another thread of the caller's delivers, e.g. a local copy that runs among the downloads.
impl<T> From<Receiver<Result<T>>> for Pending<T> {
    fn from(r: Receiver<Result<T>>) -> Pending<T> {
        Pending(r)
    }
}

/// A running download: shared progress (also used to cancel) and the final size.
pub struct Download {
    pub prog: Arc<Prog>,
    pub done: Pending<u64>,
}

/// Identity from `GET_OPTIONS` (§1.4).
#[derive(Debug, Clone)]
pub struct Ident {
    pub model: String,
    pub serial: String,
    pub firmware: String,
    pub battery: Option<pb::Battery>,
}

/// Width, height and NV12 picture of a trailer thumbnail as the X5 sends it (fw v1.13.21; 1280×640 for videos,
/// 2560×1280 for photos): a 40-byte header with `u32` 1 (YUV, the thumbnail type), the total size, and width / height
/// at offsets 16 / 20. `None` for any other layout, also for a size that is zero or odd (no whole 2 × 2 chroma
/// blocks).
pub fn nv12(b: &[u8]) -> Option<(u32, u32, &[u8])> {
    let u = |o: usize| {
        b.get(o..o + 4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
    };
    let (w, h) = (u(16)?, u(20)?);
    let ok = |x: u32| x > 0 && x % 2 == 0;
    // u64: the header is the camera's, its product may not fit
    (u(0)? == 1
        && u(4)? as usize == b.len()
        && ok(w)
        && ok(h)
        && b.len() as u64 == 40 + w as u64 * h as u64 * 3 / 2)
        .then(|| (w, h, &b[40..]))
}

pub enum Mode {
    Photo(pb::PhotoMode),
    Video(pb::VideoMode),
}

pub(crate) type Done = Box<dyn FnOnce(Result<Vec<u8>>) + Send>;
/// Receive buffers handed back by the I/O thread to the reader threads.
pub(crate) type Pool = Arc<Mutex<Vec<Vec<u8>>>>;

pub(crate) enum Msg {
    Req(u16, Vec<u8>, Duration, Done),
    Get(String, Option<PathBuf>, usize, Arc<Prog>, Done),
    Close(Option<SyncSender<Result<()>>>),
    /// Control-stream bytes from a reader thread.
    Rx(Vec<u8>),
    /// Bytes of a direct HTTP connection (Wi-Fi); empty = the server closed it.
    Http(u32, Vec<u8>),
    Gone(Error),
}

/// A byte transport carrying the UCD stream plus HTTP connections to the camera's server. Readers feed `Msg::Rx`
/// / `Msg::Http` / `Msg::Gone` into the session's channel.
pub(crate) trait Link: Send + 'static {
    /// One whole UCD packet.
    fn send(&mut self, p: &[u8]) -> std::io::Result<()>;
    fn http_open(&mut self, c: u32, req: &[u8]) -> std::io::Result<()>;
    /// Something arrived on connection `c`; `n` bytes consumed so far.
    fn http_ack(&mut self, c: u32, n: u64) -> std::io::Result<()>;
    fn http_close(&mut self, c: u32) -> std::io::Result<()>;
    /// No new HTTP connection may open now.
    fn busy(&self) -> bool;
    /// `Host` header value for HTTP requests.
    fn host(&self) -> &str;
    /// Stops the reader threads.
    fn stop(&mut self);
}

struct Inner(Sender<Msg>);

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self.0.send(Msg::Close(None));
    }
}

#[derive(Clone)]
pub struct Camera(Arc<Inner>);

const T: Duration = Duration::from_secs(10);
const LONG: Duration = Duration::from_secs(30);

fn dec<M: Message + Default>(b: Vec<u8>) -> Result<M> {
    Ok(M::decode(&b[..])?)
}

fn unit(_: Vec<u8>) -> Result<()> {
    Ok(())
}

/// Dotted version numbers compared numerically ("v1.13.21").
fn ver(s: &str) -> Vec<u32> {
    s.split(|c: char| !c.is_ascii_digit())
        .filter_map(|x| x.parse().ok())
        .collect()
}

fn ident(b: Vec<u8>) -> Result<Ident> {
    let o = dec::<pb::ReadSettingsReply>(b)?
        .settings
        .unwrap_or_default();
    // fw = the lower of 110 / 114 if both are set, else the set one, else 30 (§1.4).
    let (a, b) = (&o.ota_version, &o.box_ota_version);
    let firmware = match (a.is_empty(), b.is_empty()) {
        (false, false) => std::cmp::min_by_key(a, b, |s| ver(s)),
        (false, true) => a,
        (true, false) => b,
        _ => &o.firmware,
    }
    .clone();
    Ok(Ident {
        model: o.model,
        serial: o.serial,
        firmware,
        battery: o.battery,
    })
}

/// A session being opened: the I/O thread syncs (up to 30 s with a silent camera, [`Error::Silent`]) and, on Wi-Fi,
/// authorizes. Dropping it before it is ready cancels that at once: its handle's drop closes the I/O thread, which
/// releases the camera before it ends.
pub struct Opening {
    c: Option<(Camera, Receiver<Event>)>,
    ready: Receiver<Result<()>>,
}

impl Opening {
    /// Non-blocking: `None` while the camera syncs; the session once it is ready.
    pub fn poll(&mut self) -> Option<Result<(Camera, Receiver<Event>)>> {
        let r = match self.ready.try_recv() {
            Ok(r) => r,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Err(Error::Closed),
        };
        Some(r.and_then(|_| self.c.take().ok_or(Error::Closed)))
    }

    pub fn wait(mut self) -> Result<(Camera, Receiver<Event>)> {
        self.ready.recv().unwrap_or(Err(Error::Closed))?;
        self.c.take().ok_or(Error::Closed)
    }
}

/// One that is already done, e.g. a simulated camera's.
impl From<Result<(Camera, Receiver<Event>)>> for Opening {
    fn from(r: Result<(Camera, Receiver<Event>)>) -> Opening {
        let (tx, ready) = mpsc::sync_channel(1);
        let (c, r) = match r {
            Ok(c) => (Some(c), Ok(())),
            Err(e) => (None, Err(e)),
        };
        let _ = tx.send(r);
        Opening { c, ready }
    }
}

fn start<L: Link>(
    open: impl FnOnce(Sender<Msg>, Pool) -> Result<L> + Send + 'static,
    uuid: Option<String>,
) -> Opening {
    let (tx, ch) = mpsc::channel();
    let (etx, erx) = mpsc::channel();
    let (rtx, ready) = mpsc::sync_channel(1);
    let t = tx.clone();
    match thread::Builder::new()
        .name("x5-link".into())
        .spawn(move || io::run(open, t, ch, etx, uuid, rtx))
    {
        Ok(_) => Opening {
            c: Some((Camera(Arc::new(Inner(tx))), erx)),
            ready,
        },
        Err(e) => Err(e.into()).into(),
    }
}

impl Camera {
    /// Opens the first camera in USB app mode and syncs (no authorization on USB).
    pub fn usb() -> Opening {
        start(usb::Usb::open, None)
    }

    /// Connects to the camera's access point (`192.168.42.1`), syncs and runs CHECK_AUTHORIZATION with the
    /// client's persistent UUID; if the camera asks, waits up to 60 s for the user to confirm on it.
    pub fn wifi(ip: IpAddr, uuid: &str) -> Opening {
        start(
            move |tx, pool| tcp::Tcp::open((ip, 6666).into(), (ip, 80).into(), tx, pool),
            Some(uuid.into()),
        )
    }

    fn send<R: Send + 'static>(
        &self,
        code: u16,
        body: Vec<u8>,
        t: Duration,
        f: fn(Vec<u8>) -> Result<R>,
    ) -> Pending<R> {
        let (tx, rx) = mpsc::sync_channel(1);
        let _ = self.0.0.send(Msg::Req(
            code,
            body,
            t,
            Box::new(move |r| drop(tx.send(r.and_then(f)))),
        ));
        Pending(rx)
    }

    /// Any request; the raw response body.
    pub fn raw(&self, code: u16, body: Vec<u8>, t: Duration) -> Pending<Vec<u8>> {
        self.send(code, body, t, Ok)
    }

    /// Any request with a protobuf response.
    pub fn call<R: Message + Default + Send + 'static>(
        &self,
        code: u16,
        req: &impl Message,
        t: Duration,
    ) -> Pending<R> {
        self.send(code, req.encode_to_vec(), t, dec)
    }

    /// Runs a multi-step call on a helper thread.
    fn job<R: Send + 'static>(
        &self,
        f: impl FnOnce(&Camera) -> Result<R> + Send + 'static,
    ) -> Pending<R> {
        let (tx, rx) = mpsc::sync_channel(1);
        let c = self.clone();
        let _ = thread::Builder::new()
            .name("x5-job".into())
            .spawn(move || drop(tx.send(f(&c))));
        Pending(rx)
    }

    /// HTTP GET of `uri` on the camera's server into memory.
    pub fn get(&self, uri: &str) -> Pending<Vec<u8>> {
        let (tx, rx) = mpsc::sync_channel(1);
        let done: Done = Box::new(move |r| drop(tx.send(r)));
        let _ = self
            .0
            .0
            .send(Msg::Get(uri.into(), None, 1, Default::default(), done));
        Pending(rx)
    }

    /// Downloads `uri` to `dst` over `conns` connections (one, in order, where `<dst>.part` can't hold holes: FAT,
    /// exFAT); resumes from `<dst>.part` / `<dst>.part.todo` if they are there. Data goes to `<dst>.part`, renamed to
    /// `dst` when complete.
    pub fn download(&self, uri: &str, dst: &Path, conns: usize) -> Download {
        let (tx, rx) = mpsc::sync_channel(1);
        let prog = Arc::new(Prog::default());
        let p = prog.clone();
        let done: Done = Box::new(move |r| {
            drop(tx.send(r.map(|_| p.total.load(std::sync::atomic::Ordering::Relaxed))))
        });
        let _ = self.0.0.send(Msg::Get(
            uri.into(),
            Some(dst.into()),
            conns,
            prog.clone(),
            done,
        ));
        Download {
            prog,
            done: Pending(rx),
        }
    }

    /// Ends the session: stops a running preview, closes HTTP connections, fails pending calls.
    pub fn close(&self) -> Pending<()> {
        let (tx, rx) = mpsc::sync_channel(1);
        let _ = self.0.0.send(Msg::Close(Some(tx)));
        Pending(rx)
    }

    /// Model, serial, firmware and battery.
    pub fn identify(&self) -> Pending<Ident> {
        let q = pb::ReadSettings {
            keys: vec![48, 15, 30, 110, 114, 11],
        };
        self.send(code::GET_OPTIONS, q.encode_to_vec(), T, ident)
    }

    /// The camera fills only the settings it knows; it echoes their keys in `ReadSettingsReply`.
    pub fn options(&self, ts: &[pb::Setting]) -> Pending<pb::Settings> {
        let q = pb::ReadSettings {
            keys: ts.iter().map(|&t| t as i32).collect(),
        };
        self.send(code::GET_OPTIONS, q.encode_to_vec(), T, |b| {
            Ok(dec::<pb::ReadSettingsReply>(b)?
                .settings
                .unwrap_or_default())
        })
    }

    pub fn set_options(
        &self,
        ts: &[pb::Setting],
        v: pb::Settings,
    ) -> Pending<pb::WriteSettingsReply> {
        let q = pb::WriteSettings {
            keys: ts.iter().map(|&t| t as i32).collect(),
            settings: Some(v),
        };
        self.call(code::SET_OPTIONS, &q, T)
    }

    pub fn photo_options(&self, ts: &[pb::Param], m: pb::Profile) -> Pending<pb::ReadParamsReply> {
        let q = pb::ReadParams {
            keys: ts.iter().map(|&t| t as i32).collect(),
            profile: m as i32,
        };
        self.call(code::GET_PHOTO_OPTIONS, &q, T)
    }

    pub fn set_photo_options(
        &self,
        ts: &[pb::Param],
        v: pb::Params,
        m: pb::Profile,
    ) -> Pending<pb::WriteParamsReply> {
        let q = pb::WriteParams {
            keys: ts.iter().map(|&t| t as i32).collect(),
            params: Some(v),
            profile: m as i32,
        };
        self.call(code::SET_PHOTO_OPTIONS, &q, T)
    }

    /// Idle is an empty status (`state` IDLE).
    pub fn capture_status(&self) -> Pending<pb::CaptureStatus> {
        self.send(code::CAPTURE_STATUS, vec![], T, |b| {
            Ok(dec::<pb::CaptureStatusReply>(b)?.status.unwrap_or_default())
        })
    }

    /// All URIs of one media type on one card location, paged 100 at a time (§7.3).
    pub fn files(&self, mt: pb::Media, loc: pb::Volume) -> Pending<Vec<String>> {
        self.job(move |c| {
            let mut v = vec![];
            loop {
                let q = pb::ListFiles {
                    media: mt as i32,
                    offset: v.len() as u32,
                    limit: 100,
                    volume: loc as i32,
                };
                let r: pb::ListFilesReply = c.call(code::GET_FILE_LIST, &q, T).wait()?;
                if r.uris.is_empty() {
                    break Ok(v);
                }
                v.extend(r.uris);
                if v.len() >= r.total as usize {
                    break Ok(v);
                }
            }
        })
    }

    /// Per-file `Metadata` of photos and main videos (no LRVs); `None` where the camera sent none (§7.5).
    pub fn file_infos(&self, loc: pb::Volume) -> Pending<Vec<(String, Option<pb::Metadata>)>> {
        let q = pb::ListEntries { volume: loc as i32 };
        self.send(code::GET_FILEINFO_LIST, q.encode_to_vec(), LONG, |b| {
            let l = dec::<pb::Entries>(b)?;
            let m = |b: &[u8]| {
                (!b.is_empty())
                    .then(|| pb::Metadata::decode(b).ok())
                    .flatten()
            };
            Ok(l.entries
                .into_iter()
                .map(|f| (f.path, m(&f.meta)))
                .collect())
        })
    }

    /// One trailer record of a camera file (`ExtraType`, §7.6), raw.
    pub fn file_extra(&self, uri: &str, kind: u32) -> Pending<Vec<u8>> {
        use pb::read_extra::{Gyro, gyro::*};
        // The app always sends an empty time range.
        let g = Gyro {
            range: Some(Range::Time(Time::default())),
        };
        let q = pb::ReadExtra {
            uri: uri.into(),
            kind,
            gyro: Some(g),
        };
        self.send(code::GET_FILE_EXTRA, q.encode_to_vec(), T, Ok)
    }

    /// The file's trailer thumbnail (see [`nv12`]).
    pub fn thumbnail(&self, uri: &str) -> Pending<Vec<u8>> {
        self.file_extra(uri, 2)
    }

    /// The camera's capture-parameter JSON (`/CONF`, §8.10) and its version (the app caches it per version).
    /// No GET_FILE_PACKAGE_FINISH follows: the app sends that only to delete log files, and the X5 answers 400.
    pub fn conf(&self) -> Pending<(i32, Vec<u8>)> {
        self.job(|c| {
            let r: pb::LocateFileReply = c
                .call(code::GET_FILE, &pb::LocateFile { kind: 3 }, T)
                .wait()?;
            Ok((r.version, c.get(&r.uri).wait()?))
        })
    }

    pub fn delete(&self, uris: Vec<String>) -> Pending<pb::DeleteFilesReply> {
        self.call(code::DELETE_FILES, &pb::DeleteFiles { uris }, LONG)
    }

    /// Per-file marks; `favorite` holds the favourite flags.
    pub fn favorites(&self) -> Pending<pb::FileMarks> {
        self.send(code::GET_EDITINFO_LIST, vec![], T, dec)
    }

    /// Sets favourite flags: `path` = file path, `on` 0/1, `modified` = local time in
    /// s (newer wins); the app sends at most 100 per request.
    pub fn set_favorites(&self, v: Vec<pb::FavoriteFlag>) -> Pending<()> {
        self.send(
            code::SET_FAVORITE,
            pb::SetFavorites { flags: v }.encode_to_vec(),
            T,
            unit,
        )
    }

    /// Camera storage (location 0) is erased with ERASE_SD_CARD, other locations with FORMAT_STORAGE.
    pub fn format(&self, loc: pb::Volume) -> Pending<()> {
        let t = Duration::from_secs(120);
        match loc {
            pb::Volume::Camera => self.send(code::ERASE_SD, vec![], t, unit),
            l => self.send(
                code::FORMAT_STORAGE,
                pb::FormatVolume { volume: l as i32 }.encode_to_vec(),
                t,
                unit,
            ),
        }
    }

    /// Camera clock: unix seconds, offset from UTC in s, IANA zone; then the zone alone. Both go out now, ahead of
    /// anything sent after, and neither waits for an answer: the app's two `setOptionAsync` (§5.1 step 5, §5.3).
    pub fn set_time(&self, unix: u64, tz_s: i32, tz: &str) -> [Pending<pb::WriteSettingsReply>; 2] {
        use pb::Setting::*;
        let o = |clock, utc_offset| pb::Settings {
            clock,
            utc_offset,
            zone: tz.into(),
            ..Default::default()
        };
        [
            self.set_options(&[Clock, UtcOffset, Zone], o(unix, tz_s)),
            self.set_options(&[Zone], o(0, 0)),
        ]
    }

    /// Switches the capture mode by setting the photo or video mode (§8.10).
    pub fn set_mode(&self, m: Mode) -> Pending<pb::WriteSettingsReply> {
        let mut o = pb::Settings::default();
        let t = match m {
            Mode::Photo(p) => {
                o.photo_mode = p as i32;
                pb::Setting::PhotoMode
            }
            Mode::Video(v) => {
                o.video_mode = v as i32;
                pb::Setting::VideoMode
            }
        };
        self.set_options(&[t], o)
    }

    /// A photo in the current photo mode (`TakePhoto{KIND_SINGLE | KIND_BURST, format}` as the app sends it).
    pub fn take_photo(&self, q: pb::TakePhoto) -> Pending<pb::TakePhotoReply> {
        self.call(code::TAKE_PICTURE, &q, LONG)
    }

    /// Starts recording in the current video mode, whose record mode is `m` (RECORD_MODE_NORMAL: the app's
    /// `startNormalRecord`).
    pub fn start_capture(&self, m: pb::RecordMode) -> Pending<()> {
        let q = pb::StartRecording { mode: m as i32 };
        self.send(code::START_CAPTURE, q.encode_to_vec(), LONG, unit)
    }

    /// The new file's URI.
    pub fn stop_capture(&self, m: pb::RecordMode) -> Pending<pb::VideoFile> {
        let q = pb::StopRecording { mode: m as i32 };
        self.send(code::STOP_CAPTURE, q.encode_to_vec(), LONG, |b| {
            Ok(dec::<pb::StopRecordingReply>(b)?.file.unwrap_or_default())
        })
    }

    /// Interval photos, Starlapse and timelapse video (msg 22 / 23).
    pub fn start_timelapse(&self, m: pb::LapseMode) -> Pending<()> {
        let q = pb::StartLapse { mode: m as i32 };
        self.send(code::START_TIMELAPSE, q.encode_to_vec(), LONG, unit)
    }

    pub fn stop_timelapse(&self, m: pb::LapseMode) -> Pending<pb::VideoFile> {
        let q = pb::StopLapse { mode: m as i32 };
        self.send(code::STOP_TIMELAPSE, q.encode_to_vec(), LONG, |b| {
            Ok(dec::<pb::StopLapseReply>(b)?.file.unwrap_or_default())
        })
    }

    /// Highlight mark in the running recording at `ms` since its start (the app's capture time).
    pub fn highlight(&self, ms: u64) -> Pending<()> {
        let q = pb::Mark {
            ms,
            kind: 0,
            tag: "com.insta360.keyTime.highlight".into(),
        };
        self.send(code::KEY_TIME_POINT, q.encode_to_vec(), T, unit)
    }

    /// Starts the preview with the app's X5 parameters (§8.11). The camera ignores the requested resolution and
    /// announces the real one in [`Note::LiveParams`]; packets arrive as [`Event::Frame`].
    pub fn live_start(&self) -> Pending<()> {
        let q = pb::StartPreview {
            video: true,
            bitrate: 40,
            resolution: 44,
            gyro: true,
            bitrate2: 40,
            resolution2: 34,
        };
        self.send(code::START_LIVE, q.encode_to_vec(), T, unit)
    }

    pub fn live_stop(&self) -> Pending<()> {
        self.send(code::STOP_LIVE, vec![], T, unit)
    }
}

#[cfg(test)]
mod tests;
