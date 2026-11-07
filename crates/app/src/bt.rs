//! Bluetooth and Wi-Fi in the Camera screen's Wireless pane (crate ble): scan for X5s, connect, pair (the camera asks
//! its user to allow this computer), wake one from standby, read its access point's name and password, and station
//! mode (the camera joins the user's network; the camera worker then connects over it, `link::Camera::wifi`, at the
//! address the join or a read of the camera's existing station connection reports, or one typed in). The X5 ignores
//! Bluetooth while it is connected by USB in app mode (2026-10-24): the pane says so and offers USB first.
//!
//! A worker thread owns the adapter (opened on first use) and the camera handle and runs one flow at a time; commands
//! that come meanwhile wait for it. The UI sends commands and draws the state the worker reports.

use ble::{Ble, Camera, Dev, Error, Event, Pending, Step, code, msg, pb};
use eframe::egui;
use prost::Message;
use std::{
    collections::VecDeque,
    net::IpAddr,
    sync::mpsc::{self, Receiver, RecvTimeoutError, Sender},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub type Open = Box<dyn FnMut() -> ble::Result<(Ble, Receiver<Event>)> + Send>;

pub enum Cmd {
    Scan(bool),
    Connect(Dev),
    Close,
    /// Authorize this computer: the camera asks its user.
    Pair,
    Cancel,
    /// Wake the camera with this serial or token from standby, connect and authorize, until its Wi-Fi is up.
    Wake(String),
    /// Read the access point's name and password.
    Ap,
    /// Station mode: SSID, password.
    Join(String, String),
    /// Read the camera's Wi-Fi mode and station-mode connection (network, address).
    Sta,
    /// A step of the wake-up, from its thread (boxed: the identity step carries the camera's options).
    Step(Box<Step>),
    Quit,
}

/// An X5 the scan saw.
#[derive(Clone, Debug)]
pub struct Seen {
    pub dev: Dev,
    /// the serial's last six characters, from its `X5 <token>` name
    pub token: String,
    pub rssi: Option<i16>,
    pub proto: Option<ble::Proto>,
    pub at: Instant,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Conn {
    #[default]
    Off,
    Connecting,
    On,
}

/// What the worker reports.
#[derive(Clone, Debug, Default)]
pub struct St {
    /// the adapter could not be opened, is off or may not be used: the scan row's words
    pub err: Option<String>,
    pub scan: bool,
    pub seen: Vec<Seen>,
    pub conn: Conn,
    /// the camera connected or being connected
    pub dev: Option<Dev>,
    pub link: Option<ble::Link>,
    /// what the running flow does
    pub busy: Option<&'static str>,
    /// commands waiting for it
    pub queued: usize,
    pub paired: bool,
    pub ap: Option<(String, String)>,
    /// station mode: the camera's result (`WifiConnectionResult`), the network, the camera's address on it
    pub sta: Option<(i32, String, String)>,
    /// the wake-up's steps so far
    pub steps: Vec<String>,
    pub note: Option<(String, bool)>,
}

enum Flow {
    Pair(Pending<()>),
    Wake(Pending<Camera>),
    Ap(Pending<pb::options::WifiInfo>),
    Join(String, Pending<msg::CameraWifiConnectionResult>),
    /// the station read: the Wi-Fi mode first (the camera kept for the next request), then the connection
    Mode(Camera, Pending<i32>),
    Sta(Option<i32>, Pending<msg::WifiConnectionInfo>),
}

/// How long the camera's user gets to allow the connection, how long a wake-up and a station-mode join may take (the
/// app: 40 s for the join).
const ASK: Duration = Duration::from_secs(60);
const WAKE: Duration = Duration::from_secs(30);
const JOIN: Duration = Duration::from_secs(40);

struct W {
    up: Sender<St>,
    wake: Box<dyn Fn() + Send>,
    tx: Sender<Cmd>,
    open: Open,
    /// the client id the camera authorizes (shared with the Wi-Fi link)
    id: String,
    ble: Option<(Ble, Receiver<Event>)>,
    cam: Option<Camera>,
    conn: Option<Pending<Camera>>,
    flow: Option<Flow>,
    todo: VecDeque<Cmd>,
    st: St,
    dirty: bool,
    /// a scan asked for and refused while Bluetooth was off, resetting or not allowed: it starts once Bluetooth is back
    again: bool,
}

/// `AuthorizationStatus` and `AuthorizationResult` as the user reads them.
fn why(e: &Error) -> String {
    match e {
        Error::Unauthorized(2) | Error::Refused(3) => "the camera is busy".into(),
        Error::Unauthorized(3..=6) => {
            "another phone, watch or remote holds the camera's connection".into()
        }
        Error::Unauthorized(7) => "the camera is asleep".into(),
        Error::Refused(1) => "refused on the camera".into(),
        Error::Refused(2) => "the camera's prompt timed out".into(),
        Error::Timeout => "no answer in time".into(),
        e => e.to_string(),
    }
}

/// The scan row's words for an adapter that is off, may not be used, or could not be opened.
fn down(e: &Error) -> String {
    match e {
        Error::Off => "Bluetooth is off.".into(),
        // btleplug reports an adapter that is resetting or has no LE as one that may not be used
        Error::Denied => {
            "Bluetooth is not available: allow x5desk (or the terminal that started it) in System \
                          Settings > Privacy & Security > Bluetooth."
                .into()
        }
        e => format!("Bluetooth: {e}"),
    }
}

/// `WifiConnectionResult` of station mode.
pub fn joined(r: i32) -> &'static str {
    match r {
        0 | 5 => "joined",
        1 => "timed out",
        2 => "could not connect",
        3 => "wrong password",
        4 => "got no IP address",
        6 => "network not found",
        11 => "channel not allowed in this country",
        _ => "failed",
    }
}

impl W {
    fn note(&mut self, s: impl Into<String>, err: bool) {
        let s = s.into();
        if err {
            eprintln!("x5desk: bluetooth: {s}");
        }
        (self.st.note, self.dirty) = (Some((s, err)), true);
    }

    fn adapter(&mut self) -> Option<Ble> {
        if self.ble.is_none() {
            match (self.open)() {
                Ok(b) => (self.ble, self.st.err) = (Some(b), None),
                Err(e) => {
                    self.st.err = Some(down(&e));
                    return None;
                }
            }
        }
        self.ble.as_ref().map(|b| b.0.clone())
    }

    /// A command refused because Bluetooth is off or may not be used: the scan row says so until it is back.
    fn refused(&mut self, e: &Error) {
        if matches!(e, Error::Off | Error::Denied) {
            (self.st.err, self.dirty) = (Some(down(e)), true)
        }
    }

    fn run(mut self, rx: Receiver<Cmd>) {
        loop {
            let busy = self.flow.is_some() || self.conn.is_some() || self.st.scan;
            match rx.recv_timeout(Duration::from_millis(if busy { 30 } else { 200 })) {
                Ok(Cmd::Quit) | Err(RecvTimeoutError::Disconnected) => break,
                Ok(c) => self.todo.push_back(c),
                Err(RecvTimeoutError::Timeout) => {}
            }
            // a step, a cancel or a disconnect goes at once; the rest waits for the running flow
            while let Some(c) = self.todo.front() {
                let now = matches!(c, Cmd::Step(_) | Cmd::Cancel | Cmd::Close | Cmd::Scan(_));
                if !now && (self.flow.is_some() || self.conn.is_some()) {
                    break;
                }
                let c = self.todo.pop_front().unwrap();
                self.cmd(c);
                self.dirty = true
            }
            self.poll();
            self.events();
            if self.dirty || self.st.queued != self.todo.len() {
                self.st.queued = self.todo.len();
                self.dirty = false;
                let _ = self.up.send(self.st.clone());
                (self.wake)()
            }
        }
        // the last handle's drop waits until the link is down
        drop(self.cam.take());
    }

    fn need(&mut self) -> Option<Camera> {
        let c = self.cam.clone();
        if c.is_none() {
            self.note("connect to a camera first", true)
        }
        c
    }

    fn cmd(&mut self, c: Cmd) {
        match c {
            Cmd::Scan(on) => {
                let Some(b) = self.adapter() else { return };
                match b.scan(on).wait() {
                    Ok(()) => {
                        if on {
                            (self.st.seen, self.st.err) = Default::default()
                        }
                        (self.st.scan, self.again) = (on, false)
                    }
                    Err(e) => {
                        self.again = on && matches!(e, Error::Off | Error::Denied);
                        self.refused(&e);
                        self.note(format!("scan: {e}"), true)
                    }
                }
            }
            Cmd::Connect(d) => {
                let Some(b) = self.adapter() else { return };
                self.close();
                self.conn = Some(b.connect(&d));
                (self.st.conn, self.st.dev) = (Conn::Connecting, Some(d));
            }
            Cmd::Close => {
                self.close();
                self.note("disconnected", false)
            }
            Cmd::Pair => {
                if let Some(c) = self.need() {
                    self.flow = Some(Flow::Pair(c.authorize(&self.id, ASK)));
                    self.st.busy = Some("Pairing: allow this computer on the camera's screen");
                }
            }
            Cmd::Cancel => {
                if matches!(self.flow, Some(Flow::Pair(_))) {
                    // withdraws the prompt; the flow's own wait ends on its time
                    if let Some(c) = &self.cam {
                        drop(c.cancel_auth())
                    }
                    (self.flow, self.st.busy) = (None, None);
                    self.note("pairing cancelled", false)
                }
            }
            Cmd::Wake(s) => {
                let Some(b) = self.adapter() else { return };
                self.close();
                let tx = self.tx.clone();
                let p = b.wake_up(&s, &self.id, Duration::ZERO, WAKE, move |x| {
                    let _ = tx.send(Cmd::Step(Box::new(x)));
                });
                self.flow = Some(Flow::Wake(p));
                self.st.steps.clear();
                self.st.busy = Some("Waking the camera");
            }
            Cmd::Step(x) => {
                let s = match *x {
                    Step::Beacon(None) => "wake advertisement on".into(),
                    Step::Beacon(Some(e)) => {
                        format!("no wake advertisement ({e}): looking for the camera")
                    }
                    Step::Seen(a) => format!(
                        "camera seen{}",
                        a.rssi.map_or("".into(), |r| format!(" at {r} dBm"))
                    ),
                    Step::Retry(e) => format!("connect failed ({e}), trying again"),
                    Step::Connected(_) => "connected".into(),
                    Step::Options(o) => format!("camera answers: {}", o.camera_type),
                    Step::Synced(_) => "options read".into(),
                    Step::Auth(0) => "authorized".into(),
                    Step::Auth(1) => "allow this computer on the camera's screen".into(),
                    Step::Auth(7) => "camera asleep, asking again".into(),
                    Step::Auth(s) => format!("authorization status {s}"),
                    Step::OpenWifi => "switching the camera's Wi-Fi on".into(),
                    Step::Ready => "camera ready, its Wi-Fi is up".into(),
                };
                self.st.steps.push(s)
            }
            Cmd::Ap => {
                if let Some(c) = self.need() {
                    self.flow = Some(Flow::Ap(c.wifi()));
                    self.st.busy = Some("Reading the access point");
                }
            }
            Cmd::Join(s, p) => {
                if let Some(c) = self.need() {
                    self.flow = Some(Flow::Join(s.clone(), c.station(&s, &p, JOIN)));
                    self.st.busy = Some("The camera joins the network");
                }
            }
            Cmd::Sta => {
                if let Some(c) = self.need() {
                    self.flow = Some(Flow::Mode(c.clone(), c.wifi_mode()));
                    self.st.busy = Some("Reading the station connection");
                }
            }
            Cmd::Quit => {}
        }
    }

    fn close(&mut self) {
        (self.conn, self.flow) = (None, None);
        if let Some(c) = self.cam.take() {
            let _ = c.close().wait();
        }
        let s = &mut self.st;
        (s.conn, s.dev, s.link, s.busy, s.paired, s.ap, s.sta) = Default::default();
        self.dirty = true
    }

    fn connected(&mut self, c: Camera) {
        let l = c.link().clone();
        if l.first.is_none() {
            self.note(
                "connected, but the camera sent nothing: it ignores Bluetooth while it is connected by USB",
                true,
            )
        }
        (self.st.conn, self.st.link, self.st.err) = (Conn::On, Some(l), None);
        self.st.dev = Some(c.dev().clone());
        self.cam = Some(c);
        self.dirty = true
    }

    fn poll(&mut self) {
        if let Some(r) = self.conn.as_ref().and_then(Pending::poll) {
            self.conn = None;
            match r {
                Ok(c) => {
                    self.connected(c);
                    if self.st.note.as_ref().is_none_or(|n| !n.1) {
                        self.note("connected", false)
                    }
                }
                Err(e) => {
                    (self.st.conn, self.st.dev) = (Conn::Off, None);
                    self.refused(&e);
                    self.note(format!("connect: {e}"), true)
                }
            }
        }
        let Some(f) = self.flow.take() else { return };
        let done = match &f {
            Flow::Pair(p) => p.poll().map(|r| match r {
                Ok(()) => {
                    self.st.paired = true;
                    ("this computer is allowed on the camera".into(), false)
                }
                Err(e) => (format!("pairing: {}", why(&e)), true),
            }),
            Flow::Wake(p) => p.poll().map(|r| match r {
                Ok(c) => {
                    self.connected(c);
                    self.st.paired = true;
                    ("camera awake, its Wi-Fi is up".into(), false)
                }
                Err(e) => {
                    self.refused(&e);
                    (format!("wake-up: {}", why(&e)), true)
                }
            }),
            Flow::Ap(p) => p.poll().map(|r| match r {
                Ok(w) => {
                    self.st.ap = Some((w.ssid, w.password));
                    ("access point read".into(), false)
                }
                Err(e) => (format!("access point: {}", why(&e)), true),
            }),
            Flow::Join(s, p) => p.poll().map(|r| match r {
                Ok(r) => {
                    let ip = r
                        .wifi_connection_info
                        .map(|i| i.ip_addr)
                        .unwrap_or_default();
                    let k = r.wifi_connection_result;
                    self.st.sta = Some((k, s.clone(), ip.clone()));
                    let ok = matches!(k, 0 | 5);
                    (
                        format!(
                            "the camera {} {s}{}",
                            joined(k),
                            if ok { format!(" at {ip}") } else { "".into() }
                        ),
                        !ok,
                    )
                }
                Err(e) => (format!("station mode: {}", why(&e)), true),
            }),
            Flow::Mode(c, p) => match p.poll() {
                // a mode that cannot be read leaves the connection to tell
                Some(r) => {
                    self.flow = Some(Flow::Sta(r.ok(), c.wifi_conn()));
                    return;
                }
                None => None,
            },
            Flow::Sta(m, p) => p.poll().map(|r| match r {
                Ok(i) => {
                    let ap = *m == Some(msg::AP);
                    self.st.sta =
                        (!ap && !i.ip_addr.is_empty()).then_some((i.result, i.ssid, i.ip_addr));
                    let n = match (ap, &self.st.sta) {
                        (true, _) => {
                            "the camera's Wi-Fi is in access-point mode: no station connection"
                                .into()
                        }
                        (_, None) => "the camera reports no station connection".into(),
                        (_, Some((k, s, ip))) => format!("the camera {} {s} at {ip}", joined(*k)),
                    };
                    (
                        n,
                        (self.st.sta.as_ref()).is_some_and(|x| !matches!(x.0, 0 | 5)),
                    )
                }
                Err(e) => (format!("station connection: {}", why(&e)), true),
            }),
        };
        match done {
            Some((s, e)) => {
                self.st.busy = None;
                self.note(s, e)
            }
            None => self.flow = Some(f),
        }
    }

    fn events(&mut self) {
        let Some((_, ev)) = &self.ble else { return };
        let mut lost = None;
        while let Ok(e) = ev.try_recv() {
            match e {
                Event::Seen(a) if a.x5 => {
                    let s = Seen {
                        token: a.token().unwrap_or("?").into(),
                        rssi: a.rssi,
                        proto: a.proto(),
                        at: Instant::now(),
                        dev: a.dev,
                    };
                    match self.st.seen.iter_mut().find(|x| x.dev == s.dev) {
                        Some(x) => *x = s,
                        None => self.st.seen.push(s),
                    }
                    self.dirty = true
                }
                Event::Closed(d, e) if self.st.dev.as_ref() == Some(&d) => lost = Some(e),
                // macOS: Bluetooth switched off or on, or allowed or not
                Event::Adapter(e) => {
                    if e.is_none() && std::mem::take(&mut self.again) {
                        self.todo.push_back(Cmd::Scan(true));
                    }
                    (self.st.err, self.dirty) = (e.as_ref().map(down), true)
                }
                _ => {}
            }
        }
        if let Some(e) = lost {
            self.cam = None;
            self.close();
            self.note(format!("Bluetooth link lost: {e}"), true)
        }
    }
}

/// What the pane asks of the Camera screen.
pub enum Act {
    /// Show the camera over USB.
    Usb,
    /// Connect the camera worker to the camera at this address (station mode).
    Wifi(IpAddr),
}

pub struct Bt {
    tx: Sender<Cmd>,
    rx: Receiver<St>,
    th: Option<JoinHandle<()>>,
    pub st: St,
    ssid: String,
    pw: String,
    /// the camera's address on the user's network, as joined, read or typed
    addr: String,
    show: bool,
    ask: bool,
}

impl Drop for Bt {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
        if let Some(h) = self.th.take() {
            let _ = h.join();
        }
    }
}

impl Bt {
    /// `id`: the client id the camera authorizes; `first`: commands to run at once (test hook).
    pub fn new(ctx: &egui::Context, open: Open, id: String, first: Vec<Cmd>) -> Bt {
        let (tx, crx) = mpsc::channel();
        let (utx, rx) = mpsc::channel();
        let c = ctx.clone();
        let w = W {
            up: utx,
            wake: Box::new(move || c.request_repaint()),
            tx: tx.clone(),
            open,
            id,
            ble: None,
            cam: None,
            conn: None,
            flow: None,
            todo: first.into(),
            st: St::default(),
            dirty: false,
            again: false,
        };
        let th = thread::Builder::new()
            .name("x5-bt".into())
            .spawn(move || w.run(crx))
            .expect("bluetooth thread");
        Bt {
            tx,
            rx,
            th: Some(th),
            st: St::default(),
            ssid: String::new(),
            pw: String::new(),
            addr: String::new(),
            show: false,
            ask: false,
        }
    }

    pub fn tick(&mut self) {
        while let Ok(s) = self.rx.try_recv() {
            // a new join or read fills the address in
            if let Some((0 | 5, _, ip)) = &s.sta
                && s.sta != self.st.sta
            {
                self.addr.clone_from(ip)
            }
            self.st = s
        }
    }

    /// Nothing running or waiting (test hook).
    pub fn idle(&self) -> bool {
        let s = &self.st;
        s.busy.is_none() && s.queued == 0 && s.conn != Conn::Connecting
    }

    /// `usb`: an X5 is plugged in in app mode.
    pub fn ui(&mut self, ui: &mut egui::Ui, usb: bool) -> Option<Act> {
        let mut act = None;
        let (s, tx) = (&self.st, self.tx.clone());
        let send = |c| drop(tx.send(c));
        let (warn, weak) = (ui.visuals().warn_fg_color, ui.visuals().weak_text_color());
        if usb {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.colored_label(warn, "⚠ The X5 is connected by USB.");
                    ui.label(
                        "While the cable is in it ignores Bluetooth: it accepts what is sent and never answers. Use USB: \
                         it is faster and needs no pairing. To use Bluetooth or Wi-Fi, unplug the cable.",
                    );
                    if ui.button("Use USB").clicked() {
                        act = Some(Act::Usb);
                    }
                });
            });
            ui.add_space(8.0);
        }
        let free = s.busy.is_none();
        let on = s.conn == Conn::On;
        ui.heading("Bluetooth");
        ui.horizontal_wrapped(|ui| {
            let mut sc = s.scan;
            if ui.toggle_value(&mut sc, "Scan").changed() {
                send(Cmd::Scan(sc));
            }
            match (&s.err, s.scan, s.seen.len()) {
                (Some(e), ..) => {
                    ui.colored_label(ui.visuals().error_fg_color, e);
                }
                (_, true, 0) => {
                    ui.label("Looking for cameras…");
                    ui.spinner();
                }
                (_, true, n) => {
                    ui.label(format!("{n} camera{} found", if n == 1 { "" } else { "s" }));
                }
                _ => {
                    ui.colored_label(weak, "Press Scan to look for cameras nearby.");
                }
            }
        });
        if !s.seen.is_empty() {
            egui::Grid::new("bt seen")
                .num_columns(4)
                .striped(true)
                .spacing([16.0, 6.0])
                .show(ui, |ui| {
                    for x in &s.seen {
                        ui.strong(format!("X5 {}", x.token));
                        let age = x.at.elapsed().as_secs();
                        ui.label(match (age, x.rssi) {
                            (5.., _) => format!("seen {age} s ago"),
                            (_, Some(r)) => format!("{r} dBm"),
                            _ => "".into(),
                        })
                        .on_hover_text("Signal strength");
                        ui.label(match x.proto.map(|p| p.wifi) {
                            None | Some(0) => "Wi-Fi off".into(),
                            Some(255) => "Wi-Fi not started".into(),
                            Some(t) => format!("Wi-Fi on for {t} s more"),
                        });
                        ui.horizontal(|ui| {
                            let me = s.dev.as_ref() == Some(&x.dev);
                            if ui
                                .add_enabled(free && !(me && on), egui::Button::new("Connect"))
                                .clicked()
                            {
                                send(Cmd::Connect(x.dev.clone()));
                            }
                            if ui
                                .add_enabled(free && x.token != "?", egui::Button::new("Wake"))
                                .on_hover_text(
                                    "Connect to a camera in standby (screen dark after a short press of its power \
                                     button), pair and switch its Wi-Fi on. Needs the camera's Bluetooth wake-up \
                                     setting on. The X5's screen stays dark.",
                                )
                                .clicked()
                            {
                                send(Cmd::Wake(x.token.clone()));
                            }
                        });
                        ui.end_row();
                    }
                });
        }
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            match (s.conn, &s.dev) {
                (Conn::On, Some(d)) => {
                    let t = s.seen.iter().find(|x| x.dev == *d).map_or("", |x| &x.token);
                    ui.strong(format!("Connected to X5 {t}"));
                    if let Some(l) = &s.link {
                        ui.colored_label(weak, format!("MTU {}", l.mtu));
                    }
                    if ui.button("Disconnect").clicked() {
                        send(Cmd::Close);
                    }
                }
                (Conn::Connecting, _) => {
                    ui.label("Connecting…");
                    ui.spinner();
                }
                _ => {
                    ui.colored_label(weak, "Not connected over Bluetooth.");
                }
            }
            if let Some(b) = s.busy {
                ui.separator();
                ui.label(b);
                ui.spinner();
                if b.starts_with("Pairing") && ui.button("Cancel").clicked() {
                    send(Cmd::Cancel);
                }
            }
        });
        if let Some((n, e)) = &s.note {
            match e {
                true => ui.colored_label(ui.visuals().error_fg_color, n),
                false => ui.label(n),
            };
        }
        if !s.steps.is_empty() {
            ui.collapsing("Wake-up steps", |ui| {
                for x in &s.steps {
                    ui.label(format!("· {x}"));
                }
            });
        }
        ui.add_space(8.0);
        egui::Grid::new("bt acts")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                if ui
                    .add_enabled(on && free, egui::Button::new(if s.paired { "Paired ✔" } else { "Pair…" }))
                    .clicked()
                {
                    send(Cmd::Pair);
                }
                ui.label(
                    "Allows this computer on the camera: the camera asks on its screen, confirm there. It remembers \
                     the computer for Bluetooth and Wi-Fi.",
                );
                ui.end_row();
                if ui
                    .add_enabled(on && free, egui::Button::new("Read access point"))
                    .clicked()
                {
                    send(Cmd::Ap);
                }
                ui.vertical(|ui| {
                    match &s.ap {
                        Some((ssid, pw)) => {
                            ui.horizontal(|ui| {
                                ui.label("Name");
                                ui.monospace(ssid);
                                ui.label("Password");
                                ui.monospace(if self.show {
                                    pw.clone()
                                } else {
                                    "•".repeat(pw.chars().count())
                                });
                                ui.toggle_value(&mut self.show, "Show");
                            });
                        }
                        None => {
                            ui.label("The name and password of the camera's own Wi-Fi network.");
                        }
                    }
                    ui.colored_label(
                        weak,
                        "Joining the camera's network takes this computer off its current one; x5desk does not \
                         switch networks for you. Station mode below keeps both on your network.",
                    );
                });
                ui.end_row();
            });
        ui.add_space(12.0);
        ui.heading("Wi-Fi: station mode");
        ui.label(
            "The camera joins your Wi-Fi network, and x5desk connects to it there. The network's name and password go \
             to the camera over Bluetooth: connect and pair first. This changes the camera's Wi-Fi setting.",
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Network");
            ui.add(egui::TextEdit::singleline(&mut self.ssid).desired_width(160.0));
            ui.label("Password");
            ui.add(
                egui::TextEdit::singleline(&mut self.pw)
                    .password(true)
                    .desired_width(160.0),
            );
            if ui
                .add_enabled(
                    on && free && !self.ssid.is_empty(),
                    egui::Button::new("Join…"),
                )
                .clicked()
            {
                self.ask = true;
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(on && free, egui::Button::new("Read station connection ⚠"))
                .on_hover_text(
                    "Asks the camera over Bluetooth for its Wi-Fi mode and the network and address it has in station \
                     mode (from an earlier join); changes nothing.\n⚠ Not run on an X5 yet: the app's messages, \
                     tested on a simulated camera",
                )
                .clicked()
            {
                send(Cmd::Sta);
            }
            if let Some((k, ssid, ip)) = &s.sta {
                let t = format!("The camera {} {ssid}", joined(*k));
                match k {
                    0 | 5 => ui.label(format!("{t}: its address there is {ip}.")),
                    _ => ui.colored_label(ui.visuals().error_fg_color, format!("{t} (result {k}).")),
                };
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Camera address");
            ui.add(
                egui::TextEdit::singleline(&mut self.addr)
                    .hint_text("e.g. 192.168.1.57")
                    .desired_width(160.0),
            )
            .on_hover_text(
                "Filled in by a join or a read; or type the address your router gave the camera",
            );
            let a = self.addr.trim().parse::<IpAddr>();
            if ui
                .add_enabled(a.is_ok(), egui::Button::new("Connect over Wi-Fi"))
                .on_hover_text("The Camera screen connects to the camera at this address")
                .clicked()
                && let Ok(a) = a
            {
                act = Some(Act::Wifi(a));
            }
        });
        if self.ask {
            let mut go = None;
            let r = egui::Modal::new(egui::Id::unique("bt join")).show(ui.ctx(), |ui| {
                ui.set_max_width(420.0);
                ui.heading("Station mode");
                ui.label(format!(
                    "Make the camera join \"{}\"? It changes the camera's Wi-Fi setting; the camera's own network may \
                     go off while it is on yours.",
                    self.ssid
                ));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Join").clicked() {
                        go = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        go = Some(false);
                    }
                });
            });
            if r.should_close() && go.is_none() {
                go = Some(false);
            }
            if let Some(g) = go {
                self.ask = false;
                if g {
                    send(Cmd::Join(self.ssid.clone(), self.pw.clone()));
                }
            }
        }
        act
    }
}

/// A simulated adapter with one X5 that answers like the camera unplugged from USB ([`answer`]; the `ble=sim` hook);
/// `sta`: the camera joined [`home`] in an earlier session (`ble=sta`).
pub fn sim(sta: bool) -> Open {
    Box::new(move || {
        let mut w = if sta { home() } else { Net::default() };
        let (b, ev, _) = Ble::sim(vec![adv()], move |c, q, s| answer(c, q, s, &mut w))?;
        Ok((b, ev))
    })
}

/// The simulated X5's Wi-Fi: its mode ([`msg::AP`] at first), its station-mode connection.
type Net = (i32, msg::WifiConnectionInfo);

/// On HomeNet from an earlier session.
fn home() -> Net {
    let i = msg::WifiConnectionInfo {
        ssid: "HomeNet".into(),
        ip_addr: "192.168.1.57".into(),
        result: 5,
        ..Default::default()
    };
    (msg::STA, i)
}

/// The simulated X5's advertisement, both of its kinds merged as BlueZ shows them.
fn adv() -> ble::Adv {
    ble::Adv {
        dev: Dev("sim/X5 SIM001".into()),
        addr: "00:00:00:00:00:00".into(),
        name: Some("X5 SIM001".into()),
        rssi: Some(-58),
        services: vec![ble::Uuid::from_u128(
            0xbe80 << 96 | 0x0000_1000_8000_0080_5f9b_34fb,
        )],
        mfr: vec![(0x1A2B, vec![0, 120, 1])],
        x5: false,
        x5name: None,
    }
}

/// The simulated X5: identity and access point, an authorization its user allows after a second, a station-mode
/// join that succeeds after a second and stays in `w` (Wi-Fi mode and connection, read back as asked).
fn answer(c: u16, q: &[u8], s: &ble::Sim, w: &mut Net) -> (u16, Vec<u8>) {
    let later = |c: u16, b: Vec<u8>| {
        let s = s.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(1));
            s.note(c, &b)
        });
    };
    match c {
        code::GET_OPTIONS => {
            let o = pb::Options {
                camera_type: "Insta360 X5".into(),
                wifi_info: Some(pb::options::WifiInfo {
                    ssid: "X5 SIM001.OSC".into(),
                    password: "88888888".into(),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let r = pb::GetOptionsResp {
                option_types: vec![],
                value: Some(o),
            };
            (200, r.encode_to_vec())
        }
        code::CHECK_AUTH => {
            // SUCCESS for BLE_CONNECT with the zero spelled out: the X5 sends a non-empty 8209 (PROTOCOL.md §5.2)
            later(code::AUTH_RESULT, vec![8, 0]);
            let r = pb::CheckAuthorizationResp {
                authorization_status: 1,
                ..Default::default()
            };
            (200, r.encode_to_vec())
        }
        code::SET_WIFI_CONN => {
            let i = msg::SetWifiConnectionInfo::decode(q)
                .ok()
                .and_then(|m| m.wifi_connection_info)
                .unwrap_or_default();
            w.1 = msg::WifiConnectionInfo {
                ssid: i.ssid,
                ip_addr: "192.168.1.57".into(),
                ..Default::default()
            };
            let r = msg::CameraWifiConnectionResult {
                wifi_connection_result: 0,
                wifi_connection_info: Some(w.1.clone()),
            };
            later(code::WIFI_STATUS, r.encode_to_vec());
            (200, vec![])
        }
        code::SET_WIFI_MODE => {
            w.0 = msg::SetWifiMode::decode(q).map_or(w.0, |m| m.wifi_mode);
            (200, vec![])
        }
        code::GET_WIFI_MODE => (200, msg::GetWifiModeResp { wifi_mode: w.0 }.encode_to_vec()),
        code::GET_WIFI_CONN => {
            let r = msg::GetWifiConnectionInfoResp {
                wifi_connection_info: Some(w.1.clone()),
            };
            (200, r.encode_to_vec())
        }
        _ => (200, vec![]),
    }
}

#[cfg(test)]
mod tests;
