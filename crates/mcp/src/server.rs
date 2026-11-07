//! The MCP side of a host: one [`Conn`] per client, an rmcp `ServerHandler` over the shared host. The tools are a
//! table ([`TOOLS`]: name, tier, description, input schema, annotations); the capture tools and the camera's file
//! writes are in [`capture`], the GPU tools and the render jobs in [`render`]. `tools/list` lists those of the enabled
//! tiers (a connection's own `--tiers` narrowing them: the hosting process's `stdio`); `tools/call` checks the tier
//! again (a client's list can be stale), runs the tool and logs the call. A tool's
//! failure is a tool result with `isError` (the model reads it), an unknown tool a protocol error. Results are JSON
//! built field by field from the worker's state: the access point's password and the Wi-Fi client id are never in
//! them, and the id goes only to a local address. Thumbnails are JPEG.

use crate::{
    config::{Tier, Tiers},
    host::{Host, Shared, State, capturing},
    job::{self, Job, Prog, Which},
    log::utc,
    paths,
    render::{GpuSt, NO_GPU, panicked},
};
use cam::{
    album::{self, Item, Kind, View},
    camera::{self as cc, Cmd, Conn as Link, Info},
    dl,
};
use link::Plug;
use rmcp::{
    ErrorData, RoleServer, ServerHandler,
    handler::server::{common::schema_for_empty_input, tool::schema_for_input},
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
        JsonObject, ListToolsResult, PaginatedRequestParams, ProgressNotificationParam,
        ServerCapabilities, ServerConfig, SubscriptionFilter, Tool, ToolAnnotations,
    },
    schemars::{self, JsonSchema},
    service::{RequestContext, SubscriptionContext, SubscriptionSink},
};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    net::IpAddr,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

mod capture;
pub(crate) mod render;

use capture::{DeleteArgs, FavArgs, ModeArgs, PhotoArgs, RecordArgs, SetArgs};

/// One client's service over the host; its `subscriptions/listen` streams (2026-07-28) get the tier changes.
pub(crate) struct Conn {
    host: Arc<Host>,
    client: String,
    /// this client's own `--tiers` on top of the host's (the hosting process's `stdio`)
    narrow: Option<Tiers>,
    pub sinks: Arc<Mutex<Vec<SubscriptionSink>>>,
}

impl Conn {
    pub fn new(host: Arc<Host>, client: &str, narrow: Option<Tiers>) -> Conn {
        Conn {
            host,
            client: client.into(),
            narrow,
            sinks: Default::default(),
        }
    }

    fn sh(&self) -> &Shared {
        &self.host.sh
    }

    /// The tiers this client has: the host's, narrowed by its `--tiers` (the hosting process's own), and in a call by
    /// the `--tiers` its front sent along ([`crate::front::NARROW`]).
    fn tiers(&self) -> Tiers {
        let t = self.host.tiers();
        let t = self.narrow.map_or(t, |n| t.and(n));
        CALL.try_with(|c| t.and(*c)).unwrap_or(t)
    }
}

tokio::task_local! {
    /// The narrowing a tool call carries from a proxying front's `--tiers`.
    static CALL: Tiers;
}

/// Why a client's `--tiers` refuses tool `name` of `tier` (`narrow` leaves it out): enabling it would not help.
pub(crate) fn left_out(name: &str, tier: Tier, narrow: Tiers) -> String {
    format!(
        "{name} is in the {} tier, which this client's --tiers leaves out (it allows {narrow})",
        tier.name()
    )
}

/// What the server says before any tool is enabled.
const OFF: &str = "x5desk (Insta360 X5 camera control) has no tools enabled. The user enables them once, in a \
                   terminal: `x5desk mcp enable --tiers read,local` (add `camera` to let agents connect to the \
                   camera and capture; `delete` and `untested` only by name). Enabled tiers show up here without a \
                   restart.";

/// What the instructions add with a GPU backend.
const RENDERS: &str = " Stills and exports of X5 recordings: catalog, then render_still or export, then job_wait; \
                       photos: later.";

/// A tool: name, tier, description, input schema, annotations.
pub struct Def {
    /// the tool's name (API: kept)
    pub name: &'static str,
    /// the tier that enables it
    pub tier: Tier,
    /// what `tools/list` says of it
    pub desc: &'static str,
    schema: fn() -> Arc<JsonObject>,
    /// read only; else destructive, idempotent
    ann: Option<(bool, bool)>,
    /// needs the GPU backend: listed only with one ([`crate::render::Backend`])
    pub gpu: bool,
}

fn input<T: JsonSchema + 'static>() -> Arc<JsonObject> {
    schema_for_input::<T>().expect("tool arguments are an object")
}

impl Def {
    /// The tool as `tools/list` lists it.
    pub fn tool(&self) -> Tool {
        let a = match self.ann {
            None => ToolAnnotations::new().read_only(true).destructive(false),
            Some((d, i)) => ToolAnnotations::new()
                .read_only(false)
                .destructive(d)
                .idempotent(i),
        };
        Tool::new(self.name, self.desc, (self.schema)()).with_annotations(a.open_world(false))
    }
}

/// `album_list` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ListArgs {
    /// first item to return (default 0)
    offset: Option<usize>,
    /// how many (default 100, at most 500)
    limit: Option<usize>,
    /// only videos or only photos
    kind: Option<KindArg>,
}

#[derive(Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum KindArg {
    Video,
    Photo,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct KeyArgs {
    /// an item key from album_list
    key: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct PathArgs {
    /// absolute path of the file or card
    path: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct JobArgs {
    /// a job handle (`<host>-<n>`) as album_download, render_still or export returned it
    job: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WaitArgs {
    /// a job handle (`<host>-<n>`) as album_download, render_still or export returned it
    job: String,
    /// seconds to wait at most (default 30, at most 300)
    wait_s: Option<f64>,
}

/// `job_list` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct JobListArgs {
    /// only downloads, exports or stills
    kind: Option<JobKind>,
    /// how many, newest first (default 50, at most 500)
    limit: Option<usize>,
}

#[derive(Deserialize, JsonSchema, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum JobKind {
    Download,
    Export,
    Still,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct DownloadArgs {
    /// item keys from album_list
    keys: Vec<String>,
    /// a folder inside the agent folder to put them in (relative, no `..`; made if missing)
    subdir: Option<String>,
    /// only each item's proxy (.lrv: small, low resolution) instead of all its files
    proxy: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ConnectArgs {
    /// the camera's address on this local network (private or link-local); none: USB
    wifi_ip: Option<String>,
}

/// Every tool, in the order of the tiers.
pub const TOOLS: &[Def] = &[
    Def {
        name: "status",
        tier: Tier::Read,
        desc: "Start here: the x5desk host, the enabled tiers, a camera summary, the jobs, the agent folder (where \
               downloads go) and the read roots (what file_info may read).",
        schema: schema_for_empty_input,
        ann: None,
        gpu: false,
    },
    Def {
        name: "camera_status",
        tier: Tier::Read,
        desc: "The X5's state: USB mode, connection, model, serial, firmware, battery, storage, whether it records, \
               the album's source, mounted cards, recent notes.",
        schema: schema_for_empty_input,
        ann: None,
        gpu: false,
    },
    Def {
        name: "album_list",
        tier: Tier::Read,
        desc: "Items of the album (the connected camera's files, or an opened card's), newest first: key, files, \
               kind, size, date, length, dimensions, favourite, and the local copy in the agent folder if there is \
               one. Needs camera_connect or card_open first.",
        schema: input::<ListArgs>,
        ann: None,
        gpu: false,
    },
    Def {
        name: "album_thumbnail",
        tier: Tier::Read,
        desc: "An album item's thumbnail (JPEG, 320 x 160 equirectangular) by its key from album_list.",
        schema: input::<KeyArgs>,
        ann: None,
        gpu: false,
    },
    Def {
        name: "capture_status",
        tier: Tier::Read,
        desc: "The camera's capture state: its mode (and the modes it has: name, label, photo/record/timelapse/live, \
               whether x5desk ran it on an X5), the mode's settings with their values and the values allowed now, \
               whether it captures and since when, whether this host started it, battery, storage, remaining photos \
               and video time, temperature, recent capture notes. Needs the camera connected (camera_connect).",
        schema: schema_for_empty_input,
        ann: None,
        gpu: false,
    },
    Def {
        name: "card_list",
        tier: Tier::Read,
        desc: "Mounted X5 cards (the camera in U-disk mode, or its card in a reader) and which one the album shows.",
        schema: schema_for_empty_input,
        ann: None,
        gpu: false,
    },
    Def {
        name: "card_open",
        tier: Tier::Read,
        desc: "Shows a mounted card (its root from card_list) in the album instead of the camera.",
        schema: input::<PathArgs>,
        ann: None,
        gpu: false,
    },
    Def {
        name: "card_close",
        tier: Tier::Read,
        desc: "Shows the camera's files in the album again instead of a card.",
        schema: schema_for_empty_input,
        ann: None,
        gpu: false,
    },
    Def {
        name: "file_info",
        tier: Tier::Read,
        desc: "Size, time and metadata (camera, serial, firmware, dimensions, length, GPS, tracks) of a local X5 \
               file (insv, insp, lrv, mp4, jpg, dng) under the read roots (status lists them).",
        schema: input::<PathArgs>,
        ann: None,
        gpu: false,
    },
    Def {
        name: "job_status",
        tier: Tier::Read,
        desc: "A job's state and progress: per file for downloads (a file a later album_download started again, as \
               it had stopped, failed, or finished and gone, shows as it was then: the new transfer is that job's), \
               in frames for renders (render_still, export), with their settings.",
        schema: input::<JobArgs>,
        ann: None,
        gpu: false,
    },
    Def {
        name: "job_list",
        tier: Tier::Read,
        desc: "The jobs and their states, newest first: this host's and those kept from earlier hosts (every \
               unfinished one and the newest 100 downloads and 100 renders), under their handles; kind and limit \
               narrow it. Render rows are short: job_status has the rest.",
        schema: input::<JobListArgs>,
        ann: None,
        gpu: false,
    },
    Def {
        name: "job_wait",
        tier: Tier::Read,
        desc: "Waits until a job is finished (or wait_s passed), with progress notifications; returns its state. A \
               done still comes with its image.",
        schema: input::<WaitArgs>,
        ann: None,
        gpu: false,
    },
    Def {
        name: "catalog",
        tier: Tier::Read,
        desc: "What render_still and export accept: presets, sizes, codecs, frame rates, looks, adjustments, \
               stabilisation, projections, defaults, what agents can't do yet, and the GPU's state and limits (it \
               starts with the first render_still or export).",
        schema: schema_for_empty_input,
        ann: None,
        gpu: true,
    },
    Def {
        name: "album_download",
        tier: Tier::Local,
        desc: "Downloads album items (keys from album_list) into the agent folder or a subfolder of it, never over \
               a file (names in use are numbered: name-2.insv; an item whose download stopped resumes under its \
               names). Returns a job; follow it with job_wait. Needs the camera connected (camera_connect) or a card \
               open (card_open). A download that waits or runs when the host ends goes on in the next host: from \
               the camera once it is connected (camera_connect), from a card at once while it is mounted.",
        schema: input::<DownloadArgs>,
        ann: Some((false, false)),
        gpu: false,
    },
    Def {
        name: "job_cancel",
        tier: Tier::Local,
        desc: "Stops a job. A download's files, but not those another uncancelled job of the same items shares nor \
               those a later album_download started again (that job's now): their partial files stay, so \
               downloading the items again resumes them, also in a later host. A render's (render_still, export) \
               partial file is removed: cancelled means no file.",
        schema: input::<JobArgs>,
        ann: Some((false, true)),
        gpu: false,
    },
    Def {
        name: "render_still",
        tier: Tier::Local,
        desc: "Renders one frame of an X5 recording (a path under the read roots) to a JPEG in the agent folder or a \
               subfolder, never over a file (names in use are numbered): the 360 sphere (with GPano metadata) or a \
               flat view. Waits up to wait_s and answers with the file and a preview image; a longer render \
               answers a job that job_wait finishes with the image. job_cancel stops it and leaves no file. Photos \
               (.insp): later.",
        schema: input::<render::StillArgs>,
        ann: Some((false, false)),
        gpu: true,
    },
    Def {
        name: "export",
        tier: Tier::Local,
        desc: "Exports an X5 recording (a path under the read roots) to an MP4 in the agent folder or a subfolder \
               of it, never over a file (names in use are numbered): a catalog preset, or its settings changed, \
               over the whole clip or start_s-end_s. Answers a job at once: follow it with job_wait, stop it with \
               job_cancel (no file is left). Exports end with this MCP server; they don't carry over. A picture \
               that can't be decoded fails the export unless allow_frozen.",
        schema: input::<render::ExportArgs>,
        ann: Some((false, false)),
        gpu: true,
    },
    Def {
        name: "camera_connect",
        tier: Tier::Camera,
        desc: "Connects to the X5 over USB (in its Android App USB mode), or over Wi-Fi at wifi_ip (the camera's \
               address on this local network: private or link-local, nothing else; it gets the client id the camera \
               authorized for x5desk, so give only the camera's). Never sets the camera's clock. The album then \
               shows the camera's files. Fails while another app (the x5desk window, the phone app) holds the camera.",
        schema: input::<ConnectArgs>,
        ann: Some((false, true)),
        gpu: false,
    },
    Def {
        name: "camera_release",
        tier: Tier::Camera,
        desc: "Ends the camera session so other apps can connect. Running downloads stop and resume on the next \
               connect.",
        schema: schema_for_empty_input,
        ann: Some((false, true)),
        gpu: false,
    },
    Def {
        name: "capture_mode",
        tier: Tier::Camera,
        desc: "Switches the camera to a photo or video mode (a name or label from capture_status), not while it \
               captures. Modes x5desk has not run on an X5 need the user's `untested` tier; live modes are not for \
               agents.",
        schema: input::<ModeArgs>,
        ann: Some((false, true)),
        gpu: false,
    },
    Def {
        name: "capture_set",
        tier: Tier::Camera,
        desc: "Sets one setting of the current mode to one of the values capture_status lists as allowed now, not \
               while the camera captures; answers once the camera reports it. Settings x5desk has not written on an \
               X5 need the user's `untested` tier.",
        schema: input::<SetArgs>,
        ann: Some((false, true)),
        gpu: false,
    },
    Def {
        name: "capture_photo",
        tier: Tier::Camera,
        desc: "Takes a photo in the current photo mode (counting its self-timer down first unless self_timer is \
               false; a cancel before the shutter takes none). Answers the file, its album key (album_download \
               downloads it) and whether the album lists it. Not idempotent: each call takes one.",
        schema: input::<PhotoArgs>,
        ann: Some((false, false)),
        gpu: false,
    },
    Def {
        name: "capture_record",
        tier: Tier::Camera,
        desc: "Starts or stops a recording or timelapse of the current video mode, or waits for the camera to stop \
               capturing. start with seconds (1-600): the host stops it then, and up to 45 s the call waits and \
               answers the clip; refused while this host renders (render_still, export). stop answers the clip's \
               album key. A recording this host started stops when the host ends (status names the host); a crash \
               of this host leaves an untimed one running: capture_record stop ends it.",
        schema: input::<RecordArgs>,
        ann: Some((false, false)),
        gpu: false,
    },
    Def {
        name: "capture_mark",
        tier: Tier::Camera,
        desc: "Marks a highlight in the running recording (modes with highlight marks), at the time it has run.",
        schema: schema_for_empty_input,
        ann: Some((false, false)),
        gpu: false,
    },
    Def {
        name: "camera_favourite",
        tier: Tier::Camera,
        desc: "Sets or clears the favourite flag of album items on the camera. Needs the user's `untested` tier: the \
               X5 (firmware v1.13.21) refused it in every form tried.",
        schema: input::<FavArgs>,
        ann: Some((false, true)),
        gpu: false,
    },
    Def {
        name: "camera_delete",
        tier: Tier::Delete,
        desc: "Deletes album items (keys from album_list) from the camera, not while it captures nor while one is \
               being downloaded; cannot be undone. Answers what is gone (per item: deleted, partly, not), and which \
               deleted items were never downloaded here (local: verified, by_name, none); downloaded_only keeps \
               those not verified. Needs the camera connected (camera tier).",
        schema: input::<DeleteArgs>,
        ann: Some((true, false)),
        gpu: false,
    },
];

/// The tools `tools/list` lists for tiers `t`; the GPU's only with a backend (`gpu`).
pub(crate) fn listed(t: Tiers, gpu: bool) -> impl Iterator<Item = &'static Def> {
    TOOLS
        .iter()
        .filter(move |d| t.has(d.tier) && (gpu || !d.gpu))
}

fn args<T: DeserializeOwned>(a: JsonObject) -> Result<T, String> {
    serde_json::from_value(Value::Object(a)).map_err(|e| format!("arguments: {e}"))
}

fn ok(v: Value) -> Result<CallToolResult, String> {
    Ok(CallToolResult::structured(v))
}

/// Standard base64 with padding.
pub fn b64(b: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for (i, k) in [18, 12, 6, 0].into_iter().enumerate() {
            s.push(if i <= c.len() {
                A[(n >> k & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    s
}

fn kind(k: Kind) -> &'static str {
    match k {
        Kind::Video => "video",
        Kind::Photo => "photo",
        Kind::Other => "other",
    }
}

fn plug(p: Option<Plug>) -> Value {
    match p {
        None => Value::Null,
        Some(Plug::None) => "none".into(),
        Some(Plug::Disk) => "U-disk mode".into(),
        Some(Plug::Other) => "another USB mode (not Android App)".into(),
        Some(Plug::App) => "app mode".into(),
    }
}

fn link(c: Link) -> &'static str {
    match c {
        Link::Off => "off",
        Link::Connecting => "connecting",
        Link::On => "on",
    }
}

/// The album's source: `camera`, or the card's mount point.
fn source(i: &Info) -> Value {
    i.src
        .as_ref()
        .map_or("camera".into(), |r| r.display().to_string().into())
}

/// The camera fields of `status`; never the access point's password.
fn summary(i: &Info) -> Value {
    let id = i.id.as_ref();
    let b = i.battery.as_ref();
    json!({
        "usb": plug(i.plug),
        "connection": link(i.conn),
        "model": id.map(|x| x.model.clone()),
        "serial": id.map(|x| x.serial.clone()),
        "firmware": id.map(|x| x.firmware.clone()),
        "battery_percent": b.map(|b| cc::level(b).0),
        "charging": b.map(|b| cc::level(b).1),
        "low_battery": b.map(cc::low),
        "storage_free_bytes": i.storage.as_ref().map(|s| s.free_space),
        "storage_total_bytes": i.storage.as_ref().map(|s| s.total_space),
        "recording": i.busy,
        "album_source": source(i),
    })
}

fn cards(i: &Info) -> Value {
    let v: Vec<Value> = (i.disk.iter())
        .map(|v| {
            json!({
                "root": v.root.display().to_string(),
                "name": v.name(),
                "device": v.dev,
                "camera": v.cam,
                "open": i.src.as_ref() == Some(&v.root),
            })
        })
        .collect();
    v.into()
}

/// The camera's status without secrets: the access point's SSID only.
fn camera(s: &State) -> Value {
    let i = &s.info;
    let mut v = summary(i);
    let o = v.as_object_mut().unwrap();
    o.insert("recording".into(), capturing(s).into());
    o.insert(
        "access_point_ssid".into(),
        json!(i.wifi.as_ref().map(|w| w.0.clone())),
    );
    o.insert("clock_offset_s".into(), json!(i.skew));
    o.insert("connect_attempts".into(), json!(i.tries));
    o.insert(
        "connecting_for_s".into(),
        json!(i.since.map(|t| t.elapsed().as_secs())),
    );
    o.insert("cards".into(), cards(i));
    let n: Vec<Value> = (s.notes.iter().rev().take(10).rev())
        .map(|(_, t, e)| json!({ "text": t, "error": e }))
        .collect();
    o.insert("notes".into(), n.into());
    v
}

/// A friendlier word for what a failed connect noted: a camera another process holds, another client's session, no
/// camera.
fn why(m: &str) -> String {
    let l = m.to_ascii_lowercase();
    if l.contains("another client holds the session") {
        format!(
            "another client holds the camera's session (the Insta360 app, or x5desk over Wi-Fi?): close \
             it there, then try again ({m})"
        )
    } else if [
        "exclusive access",
        "busy",
        "already claimed",
        "in use",
        // nusb on Windows: WinUSB opens a device for one process (ERROR_ACCESS_DENIED)
        "failed to open device (error 5)",
    ]
    .iter()
    .any(|s| l.contains(s))
    {
        format!(
            "the camera is in use by another app (x5desk?): disconnect it there, then try again ({m})"
        )
    } else if l.contains("no camera in usb app mode") {
        format!("no X5 in USB app mode: plug it in and choose Android App on its USB menu ({m})")
    } else {
        m.to_string()
    }
}

/// Whether `ip` can be a camera's on this computer's network: IPv4 private or link-local, IPv6 unique-local
/// (fc00::/7) or link-local (fe80::/10), an IPv4-mapped one as its IPv4. Not loopback, unspecified, multicast or
/// public.
fn lan(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => a.is_private() || a.is_link_local(),
        IpAddr::V6(a) => match a.to_ipv4_mapped() {
            Some(a) => lan(a.into()),
            None => a.is_unique_local() || a.is_unicast_link_local(),
        },
    }
}

/// How a file of an item fits at a number: its stopped download resumes, its finished one is kept, or the name is
/// free.
#[derive(Clone, Copy, PartialEq)]
enum Fit {
    Resume,
    Done,
    Free,
}

/// Where an item's files (album URIs `u`, queue URIs `qu`) go in `dir`: numbered alike (their recording key stays
/// one), past names in use, being written, queued, or `taken` in this call. A stopped download of the same file
/// resumes under its name, and with one that does, the files the queue finished at that number are kept (a job
/// cancelled between them); a finished item again goes past them. Names that are no safe file names are refused.
/// Renders need no rule here: one makes its `<dst>.part` (`O_EXCL`) under `jobs` before it lets go of it, so
/// [`paths::free`], called under `jobs`, numbers past every unfinished render; and render names never take a camera
/// file's. Keep it so: a render reserving its `.part` after `jobs` is released would need one.
fn place(
    q: &dl::Queue,
    dir: &Path,
    taken: &HashSet<PathBuf>,
    u: &[&String],
    qu: &[String],
) -> Result<Vec<PathBuf>, String> {
    if let Some(n) = u.iter().map(|u| album::name(u)).find(|n| !album::safe(n)) {
        return Err(format!("{n:?} is no safe file name"));
    }
    let fit = |d: &PathBuf, u: &String| {
        let same = |x: &&dl::Dl| x.dst == *d && x.uri == *u;
        if taken.contains(d) {
            None
        } else if q.v.iter().filter(same).any(|x| x.st != dl::St::Done) {
            Some(Fit::Resume)
        } else if q.v.iter().filter(same).any(|x| x.st == dl::St::Done) && d.is_file() {
            Some(Fit::Done)
        } else if paths::free(d) && !q.v.iter().any(|x| x.dst == *d && x.st != dl::St::Done) {
            Some(Fit::Free)
        } else {
            None
        }
    };
    let dsts = (1..)
        .map(|i| -> Vec<PathBuf> {
            u.iter()
                .map(|u| dir.join(paths::numbered(album::name(u), i)))
                .collect()
        })
        .find(|d| {
            let f: Option<Vec<Fit>> = d.iter().zip(qu).map(|(d, u)| fit(d, u)).collect();
            f.is_some_and(|f| !f.contains(&Fit::Done) || f.contains(&Fit::Resume))
        })
        .unwrap();
    match dsts.iter().find(|d| d.parent() != Some(dir)) {
        Some(d) => Err(format!("{}: not in {}", d.display(), dir.display())),
        None => Ok(dsts),
    }
}

/// Whether a download for `d` waits or runs for a job: not one stopped, failed or finished, nor one whose cancel the
/// link has yet to see.
fn wanted(q: &dl::Queue, d: &Path) -> bool {
    q.v.iter().any(|x| {
        x.dst == d
            && match x.st {
                dl::St::Queued => true,
                dl::St::Running => !x.prog.cancel.load(std::sync::atomic::Ordering::Relaxed),
                _ => false,
            }
    })
}

/// The queue's URI of an album file: a card's is an import (`file://`).
fn qury(src: &Option<PathBuf>, u: &str) -> String {
    match src {
        Some(r) => format!("file://{}", r.join(u.trim_start_matches('/')).display()),
        None => u.to_string(),
    }
}

fn job_dto(handle: String, j: &Job, p: &Prog, cam: bool, rec: bool) -> Value {
    let files: Vec<Value> = (p.parts.iter())
        .map(|x| {
            json!({
                "path": x.dst.display().to_string(),
                "state": x.state,
                "done_bytes": x.done,
                "total_bytes": x.total,
                "error": (!x.err.is_empty()).then(|| x.err.clone()),
            })
        })
        .collect();
    let waits = p.state == "queued" && !cam && j.files.iter().any(|f| dl::local(&f.uri).is_none());
    json!({
        "job": handle,
        "kind": "download",
        "state": p.state,
        "finished": p.finished(),
        "done_bytes": p.done,
        "total_bytes": p.total,
        "keys": j.keys,
        "created": utc(j.at),
        "files": files,
        "note": match (waits, rec && p.state == "queued") {
            (true, _) => Some("waits for the camera: call camera_connect"),
            (_, true) => Some("held while the camera records"),
            _ => None,
        },
    })
}

/// The GPU's state as `status` shows it; never starts it.
fn gpu(sh: &Shared) -> Value {
    match &sh.render {
        None => json!({ "state": "none", "note": NO_GPU }),
        Some(r) => gpu_value(render::gpu_st(&**r), sh.gpu_at.get().copied()),
    }
}

/// A GPU state as `status` and `catalog` show it; `started`: when this host's start found it up.
fn gpu_value(st: GpuSt, started: Option<std::time::SystemTime>) -> Value {
    match st {
        GpuSt::Off => {
            json!({ "state": "off", "note": "it starts with the first render_still or export" })
        }
        GpuSt::Starting => json!({ "state": "starting" }),
        GpuSt::Ready(i) => json!({
            "state": "ready",
            "adapter": i.adapter,
            "backend": i.backend,
            "driver": i.driver,
            "decode": i.decode,
            "encode": i.encode,
            "max_texture": i.max_texture,
            "raft": match i.raft {
                None => json!({ "available": true }),
                Some(w) => json!({ "available": false, "why": w }),
            },
            "errors": i.errors,
            "suspect": i.suspect,
            "wedged": i.wedged,
            "checked_ms": i.ms,
            "started": started.map(utc),
        }),
        GpuSt::Failed {
            why,
            step,
            retry_in,
        } => json!({
            "state": "failed",
            "why": why,
            "step": step,
            "retry_in_s": retry_in.map(|d| d.as_secs()),
        }),
        GpuSt::Dead(why) => json!({ "state": "dead", "why": why }),
    }
}

/// The local copies of an item's files, when all are there: finished downloads of the queue, else by name in the
/// agent folder (not for a name that is no safe file name).
fn local(it: &Item, q: &dl::Queue, agent: &Path) -> Option<Vec<String>> {
    (it.uris.iter())
        .map(|u| {
            let n = Some(album::name(u)).filter(|n| album::safe(n))?;
            let done = (q.v.iter())
                .filter(|d| d.st == dl::St::Done && album::name(&d.uri) == n && d.dst.is_file())
                .map(|d| d.dst.clone())
                .next_back();
            done.or_else(|| Some(agent.join(n)).filter(|p| p.is_file()))
                .map(|p| p.display().to_string())
        })
        .collect()
}

/// An album item; `card`: the root of the card it is on (its first file's absolute path then in `path`, for the tools
/// that read local files).
fn item(it: &Item, have: Option<Vec<String>>, card: Option<&Path>) -> Value {
    let files: Vec<&str> = it.uris.iter().map(|u| album::name(u)).collect();
    let path = card.zip(it.uris.first());
    json!({
        "key": it.key,
        "kind": kind(it.kind),
        "files": files,
        "uris": it.uris,
        "path": path.map(|(r, u)| paths::shown(&r.join(u.trim_start_matches('/')))),
        "size_bytes": it.size,
        "date": (it.date > 0).then(|| album::date(it.date)),
        "duration_ms": it.ms,
        "width": it.dim[0],
        "height": it.dim[1],
        "favourite": it.fav,
        "local": have,
    })
}

/// What `file_info` reads: the file system's size and time, the trailer's metadata, the MP4's tracks.
fn file_info(p: &Path) -> Result<Value, String> {
    let m = std::fs::metadata(p).map_err(|e| format!("{}: {e}", paths::shown(p)))?;
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut v = json!({
        "path": paths::shown(p),
        "kind": ext,
        "size_bytes": m.len(),
        "modified": m.modified().ok().map(utc),
    });
    let o = v.as_object_mut().unwrap();
    if let Ok(t) = media::insv::Trailer::open(p) {
        let x = &t.meta;
        let gps = t
            .recs
            .iter()
            .any(|r| r.kind == media::insv::GPS && r.size > 0);
        o.insert(
            "trailer".into(),
            json!({
                "camera": x.camera,
                "serial": x.serial,
                "firmware": x.firmware,
                "created": (x.created > 0).then(|| album::date(x.created)),
                "utc_offset_s": x.tz,
                "duration_ms": x.duration_ms,
                "dimension": x.dimension,
                "resolution": x.resolution,
                "fps": x.fps,
                "frames": x.frames,
                "gps": gps,
                "flowstate": x.flowstate,
            }),
        );
    }
    if matches!(ext.as_str(), "insv" | "lrv" | "mp4")
        && let Ok(f) = media::mp4::Mp4::open(p)
    {
        let s = |b: [u8; 4]| String::from_utf8_lossy(&b).into_owned();
        let tracks: Vec<Value> = (f.tracks.iter())
            .map(|t| {
                json!({
                    "handler": s(t.handler),
                    "codec": s(t.fourcc),
                    "width": t.width,
                    "height": t.height,
                    "duration_s": t.secs(),
                    "samples": t.samples.len(),
                })
            })
            .collect();
        o.insert(
            "mp4".into(),
            json!({
                "brand": s(f.brand),
                "duration_s": f.duration as f64 / f.timescale.max(1) as f64,
                "tracks": tracks,
            }),
        );
    }
    Ok(v)
}

impl Conn {
    /// The album of the source, listed: waits up to 10 s for a listing under way. Fails without a camera or card.
    async fn album(&self) -> Result<(Vec<Item>, Option<PathBuf>), String> {
        let sh = self.sh();
        {
            let s = sh.st.lock().unwrap();
            if s.info.src.is_none() && s.info.conn == Link::Off {
                return Err("no camera connected and no card open: call camera_connect (camera tier) or card_open \
                            (card_list lists the cards) first"
                    .into());
            }
        }
        let r = sh
            .until(10.0, |s| Some((s.album.clone()?, s.info.src.clone())))
            .await;
        r.ok_or_else(|| match sh.st.lock().unwrap().info.busy {
            true => "the camera is recording: its files are listed once it stops".into(),
            false => "the album was not listed within 10 s; see camera_status".into(),
        })
    }

    fn status(&self, ctx: &RequestContext<RoleServer>) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let roots: Vec<String> = sh.read_roots().iter().map(|r| paths::shown(r)).collect();
        let (run, wait, fin) = {
            let (j, q) = (sh.jobs.lock().unwrap(), sh.q.lock().unwrap());
            let dl = j.v.iter().map(|j| j.prog(&q).state);
            dl.chain((0..j.r.v.len()).map(|i| j.r.prog(i).state))
                .fold((0, 0, 0), |(r, w, f), p| match p {
                    "running" => (r + 1, w, f),
                    "queued" => (r, w + 1, f),
                    _ => (r, w, f + 1),
                })
        };
        // the backend asked with no lock held
        let g = gpu(sh);
        let s = sh.st.lock().unwrap();
        let t = self.tiers();
        let next = match (s.info.conn, &s.info.src) {
            (Link::On, _) | (_, Some(_)) => "album_list lists the album",
            _ if t.has(Tier::Camera) => {
                "camera_connect connects the camera (or card_open opens a card)"
            }
            _ => "card_open opens a card; connecting the camera needs the camera tier",
        };
        let tiers: Vec<&str> = t.iter().map(Tier::name).collect();
        let mut cam = summary(&s.info);
        cam["recording"] = capturing(&s).into();
        ok(json!({
            "host": {
                "kind": "headless",
                "id": sh.id,
                "version": env!("CARGO_PKG_VERSION"),
                "simulated_camera": sh.sim,
                "clients": sh.conns.load(std::sync::atomic::Ordering::Relaxed),
            },
            "protocol": ctx.protocol_version().map(|v| v.as_str().to_string()),
            "client": self.client,
            "tiers": tiers,
            "camera": cam,
            "jobs": { "running": run, "queued": wait, "finished": fin },
            "agent_folder": sh.agent.display().to_string(),
            "read_roots": roots,
            "gpu": g,
            "next": next,
        }))
    }

    async fn album_list(&self, a: ListArgs) -> Result<CallToolResult, String> {
        let (items, src) = self.album().await?;
        let (off, n) = (a.offset.unwrap_or(0), a.limit.unwrap_or(100).clamp(1, 500));
        let w = View {
            kind: a.kind.map(|k| match k {
                KindArg::Video => Kind::Video,
                KindArg::Photo => Kind::Photo,
            }),
            ..Default::default()
        };
        let mut idx = vec![];
        w.apply(&items, &mut idx);
        let sh = self.sh();
        let q = sh.q.lock().unwrap();
        let v: Vec<Value> = (idx.iter().skip(off).take(n))
            .map(|&i| item(&items[i], local(&items[i], &q, &sh.agent), src.as_deref()))
            .collect();
        drop(q);
        let busy = sh.st.lock().unwrap().info.busy;
        ok(json!({
            "source": src.map_or("camera".into(), |r| r.display().to_string()),
            "total": idx.len(),
            "offset": off,
            "more": off + v.len() < idx.len(),
            "recording": busy,
            "items": v,
        }))
    }

    async fn album_thumbnail(&self, a: KeyArgs) -> Result<CallToolResult, String> {
        let (items, _) = self.album().await?;
        if !items.iter().any(|i| i.key == a.key) {
            return Err(format!(
                "no item {:?} in the album: album_list lists the keys",
                a.key
            ));
        }
        let sh = self.sh();
        let have = sh.st.lock().unwrap().thumb(&a.key).cloned();
        let t = match have {
            Some(t) => Some(t),
            None => {
                sh.cmd(Cmd::Want(vec![a.key.clone()]));
                sh.until(10.0, |s| s.thumb(&a.key).cloned()).await
            }
        };
        let (w, h, px) = match t {
            None => return Err(format!("{}: no thumbnail within 10 s", a.key)),
            Some(None) => return Err(format!("{}: the item has no thumbnail", a.key)),
            Some(Some(t)) => t,
        };
        let j = tokio::task::spawn_blocking(move || media::jpeg::encode(&px, w, h, 85, None))
            .await
            .map_err(|e| e.to_string())??;
        Ok(CallToolResult::success(vec![
            ContentBlock::image(b64(&j), "image/jpeg"),
            ContentBlock::text(format!("{}: {w} x {h} JPEG, {} bytes", a.key, j.len())),
        ]))
    }

    fn card_list(&self) -> Result<CallToolResult, String> {
        let s = self.sh().st.lock().unwrap();
        ok(json!({ "cards": cards(&s.info), "album_source": source(&s.info) }))
    }

    async fn card_open(&self, a: PathArgs) -> Result<CallToolResult, String> {
        let sh = self.sh();
        if sh.st.lock().unwrap().deleting.is_some() {
            return Err("a delete runs on the camera: the album shows its outcome first".into());
        }
        let want = std::fs::canonicalize(&a.path).unwrap_or_else(|_| a.path.clone().into());
        let root = {
            let s = sh.st.lock().unwrap();
            let v = s
                .info
                .disk
                .iter()
                .find(|v| std::fs::canonicalize(&v.root).is_ok_and(|r| r == want));
            v.map(|v| v.root.clone())
        };
        let root =
            root.ok_or_else(|| format!("{}: not a mounted X5 card; card_list lists them", a.path))?;
        sh.cmd(Cmd::Source(Some(root.clone())));
        let n = sh.until(10.0, |s| {
            (s.info.src.as_ref() == Some(&root)).then(|| s.album.as_ref().map(Vec::len))?
        });
        match n.await {
            Some(n) => ok(json!({ "album_source": root.display().to_string(), "items": n })),
            None => Err(format!(
                "{}: the card was not listed within 10 s",
                root.display()
            )),
        }
    }

    async fn card_close(&self) -> Result<CallToolResult, String> {
        let sh = self.sh();
        if sh.st.lock().unwrap().deleting.is_some() {
            return Err("a delete runs on the camera: the album shows its outcome first".into());
        }
        if sh.st.lock().unwrap().info.src.is_none() {
            return ok(
                json!({ "album_source": "camera", "note": "the album showed the camera already" }),
            );
        }
        sh.cmd(Cmd::Source(None));
        match sh.until(10.0, |s| s.info.src.is_none().then_some(())).await {
            Some(()) => ok(json!({ "album_source": "camera" })),
            None => Err("the album did not switch back within 10 s".into()),
        }
    }

    async fn file_info(&self, a: PathArgs) -> Result<CallToolResult, String> {
        let p = paths::readable(&a.path, &self.sh().read_roots())?;
        let v = tokio::task::spawn_blocking(move || file_info(&p))
            .await
            .map_err(|e| e.to_string())??;
        ok(v)
    }

    /// A job's state, of either kind, by its handle.
    fn job(&self, h: &str) -> Result<Value, String> {
        let sh = self.sh();
        let (cam, rec) = {
            let s = sh.st.lock().unwrap();
            (s.info.conn == Link::On, s.info.busy)
        };
        let jobs = sh.jobs.lock().unwrap();
        Ok(match jobs.which(h)? {
            Which::Dl(i) => {
                let (j, q) = (&jobs.v[i], sh.q.lock().unwrap());
                job_dto(j.h.clone(), j, &j.prog(&q), cam, rec)
            }
            Which::Render(i) => render::render_dto(&jobs.r.v[i], &jobs.r.prog(i)),
        })
    }

    /// The jobs of `kind` (all: none), newest first, at most `limit`: downloads in full, renders in short rows.
    fn job_list(&self, a: JobListArgs) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let (cam, rec) = {
            let s = sh.st.lock().unwrap();
            (s.info.conn == Link::On, s.info.busy)
        };
        let (jobs, q) = (sh.jobs.lock().unwrap(), sh.q.lock().unwrap());
        let want = |k: JobKind| a.kind.is_none_or(|w| w == k);
        let mut all: Vec<(std::time::SystemTime, Which)> = vec![];
        if want(JobKind::Download) {
            all.extend(jobs.v.iter().enumerate().map(|(i, j)| (j.at, Which::Dl(i))))
        }
        for (i, r) in jobs.r.v.iter().enumerate() {
            let k = match r.kind {
                crate::render::Kind::Export => JobKind::Export,
                crate::render::Kind::Still => JobKind::Still,
            };
            if want(k) {
                all.push((r.at, Which::Render(i)))
            }
        }
        // oldest first, those of one time as made; then the other way round
        all.sort_by_key(|x| x.0);
        let n = all.len();
        let v: Vec<Value> = (all.iter().rev().take(a.limit.unwrap_or(50).clamp(1, 500)))
            .map(|(_, w)| match *w {
                Which::Dl(i) => {
                    let j = &jobs.v[i];
                    job_dto(j.h.clone(), j, &j.prog(&q), cam, rec)
                }
                Which::Render(i) => render::row(&jobs.r.v[i], &jobs.r.prog(i)),
            })
            .collect();
        ok(json!({ "jobs": v, "total": n }))
    }

    async fn job_wait(
        &self,
        a: WaitArgs,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let wait = Duration::from_secs_f64(a.wait_s.unwrap_or(30.0).clamp(0.0, 300.0));
        let w = sh.jobs.lock().unwrap().which(&a.job)?;
        if let Which::Render(_) = w {
            return self.render_wait(&a.job, wait, ctx).await;
        }
        let (t0, tok) = (Instant::now(), ctx.meta.get_progress_token());
        let (mut sent, mut last) = (t0, -1.0);
        loop {
            let p = {
                let (jobs, q) = (sh.jobs.lock().unwrap(), sh.q.lock().unwrap());
                jobs.find(&a.job)?.prog(&q)
            };
            if p.finished() || t0.elapsed() >= wait {
                let mut v = self.job(&a.job)?;
                v["waited_s"] = json!((t0.elapsed().as_secs_f64() * 10.0).round() / 10.0);
                return ok(v);
            }
            if let Some(t) = &tok
                && sent.elapsed() >= Duration::from_secs(1)
                && p.done as f64 > last
            {
                (sent, last) = (Instant::now(), p.done as f64);
                // a total only once every file's size is known: a sum of those started would drop as the next begins
                let all = p.total > 0 && p.parts.iter().all(|x| x.total > 0);
                let mut n = ProgressNotificationParam::new(t.clone(), p.done as f64).with_message(
                    match all {
                        true => format!("{}: {} of {} bytes", p.state, p.done, p.total),
                        false => format!("{}: {} bytes", p.state, p.done),
                    },
                );
                if all {
                    n = n.with_total(p.total as f64)
                }
                let _ = ctx.peer.notify_progress(n).await;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(200)) => {}
                _ = ctx.ct.cancelled() => return Err("cancelled".into()),
                _ = sh.ct.cancelled() => return Err("the host is shutting down".into()),
            }
        }
    }

    async fn album_download(&self, a: DownloadArgs) -> Result<CallToolResult, String> {
        if a.keys.is_empty() {
            return Err("give the keys of the items to download (album_list lists them)".into());
        }
        let (items, src) = self.album().await?;
        let mut picks = vec![];
        let mut bad = vec![];
        for k in &a.keys {
            let Some(it) = items.iter().find(|i| i.key == *k) else {
                bad.push(k.as_str());
                continue;
            };
            let u: Vec<&String> = match a.proxy.unwrap_or(false) {
                true => it
                    .uris
                    .iter()
                    .filter(|u| u.to_ascii_lowercase().ends_with(".lrv"))
                    .collect(),
                false => it.uris.iter().collect(),
            };
            if u.is_empty() {
                return Err(format!("{k} has no proxy (.lrv)"));
            }
            picks.push((k, u));
        }
        if !bad.is_empty() {
            return Err(format!(
                "no items {bad:?} in the album: album_list lists the keys"
            ));
        }
        let sh = self.sh();
        std::fs::create_dir_all(&sh.agent).map_err(|e| format!("{}: {e}", sh.agent.display()))?;
        let dir = paths::sub(&sh.agent, a.subdir.as_deref().unwrap_or(""))?;
        let dir = paths::make(&sh.agent, &dir)?;
        let mut files = vec![];
        let h = {
            // the jobs before the queue, both held until the job is in: a cancel meanwhile sees it with its files
            let mut jobs = sh.jobs.lock().unwrap();
            let mut q = sh.q.lock().unwrap();
            let mut taken = HashSet::new();
            // every item placed before any is queued: a refused one queues nothing
            let mut all = vec![];
            for (k, u) in picks {
                let qu: Vec<String> = u.iter().map(|u| qury(&src, u)).collect();
                let d = place(&q, &dir, &taken, &u, &qu).map_err(|e| format!("{k}: {e}"))?;
                taken.extend(d.iter().cloned());
                all.push((d, qu));
            }
            for (d, qu) in all {
                for (d, u) in d.into_iter().zip(qu) {
                    // a file no download waits or runs for is this job's from here: the older jobs keep its state
                    if !d.exists() && !wanted(&q, &d) {
                        jobs.take(&d, &q);
                        // a cancel the link has yet to see: queued again, it resumes once it stopped (as a hold)
                        if let Some(x) =
                            q.v.iter_mut()
                                .find(|x| x.dst == d && x.st == dl::St::Running)
                        {
                            x.st = dl::St::Queued
                        }
                    }
                    q.add(&u, d.clone());
                    files.push(job::File::new(u, d));
                }
            }
            let h = jobs.add(a.keys.clone(), files);
            sh.save(&mut jobs, &mut q);
            h
        };
        sh.log.line(&format!("{} queued job {h}", self.client));
        ok(self.job(&h)?)
    }

    async fn job_cancel(&self, a: JobArgs) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let w = sh.jobs.lock().unwrap().which(&a.job)?;
        let Which::Dl(i) = w else {
            return self.render_cancel(&a.job).await;
        };
        // its own downloads stopped, but none another job still wants (two calls for one item share them) nor one a
        // later job started again (`i` holds: a running host only adds jobs)
        let cut: Vec<PathBuf> = {
            let mut jobs = sh.jobs.lock().unwrap();
            let cut = jobs.cut(i);
            let mut q = sh.q.lock().unwrap();
            for d in &cut {
                if let Some(i) = q.v.iter().position(|x| x.dst == *d) {
                    q.cancel(i)
                }
            }
            jobs.v[i].cancelled = true;
            sh.save(&mut jobs, &mut q);
            cut
        };
        // a running download ends once the link sees the flag
        let end = Instant::now() + Duration::from_secs(3);
        while Instant::now() < end {
            let run = (sh.q.lock().unwrap().v.iter())
                .any(|x| x.st == dl::St::Running && cut.contains(&x.dst));
            if !run {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await
        }
        ok(self.job(&a.job)?)
    }

    async fn camera_connect(&self, a: ConnectArgs) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let _a = sh.act.lock().await;
        let ip = (a.wifi_ip.as_deref().map(str::trim))
            .map(|s| {
                s.parse::<IpAddr>()
                    .map_err(|e| format!("wifi_ip {s:?}: {e}"))
            })
            .transpose()?;
        // the client id goes to that address: only to one a camera can have here
        if let Some(ip) = ip
            && !lan(ip)
        {
            return Err(format!(
                "wifi_ip {ip}: the camera's address is on the local network, a private (10.x, 172.16-31.x, \
                 192.168.x, IPv6 fc00::/7) or link-local one (169.254.x, fe80::/10)"
            ));
        }
        let (conn, seq) = {
            let s = sh.st.lock().unwrap();
            (s.info.conn, s.seq)
        };
        if ip.is_some() && sh.sim {
            return Err("the simulated camera has no Wi-Fi: connect without wifi_ip".into());
        }
        if conn == Link::Off {
            match ip {
                Some(ip) => sh.cmd(Cmd::Wifi {
                    ip,
                    id: sh.wifi(),
                    clock: false,
                }),
                None => sh.cmd(Cmd::Connect { clock: false }),
            }
        }
        // the worker's notes say how it went: "connected to …", or why not
        let fail = |t: &str| {
            let w = [
                "connect:",
                "the X5 is in app mode",
                "disconnect the camera first",
                "connection cancelled",
            ];
            w.iter().any(|w| t.starts_with(w))
        };
        let r = sh.until(20.0, |s| {
            let mut n = s.since(seq);
            if s.info.conn == Link::On
                && (conn == Link::On || n.any(|n| n.1.starts_with("connected")))
            {
                return Some(Ok(()));
            }
            s.since(seq).find(|n| fail(&n.1)).map(|n| Err(n.1.clone()))
        });
        match r.await {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(why(&e)),
            None => {
                return Err("still connecting after 20 s; camera_status shows how it goes".into());
            }
        }
        // the camera's files in the album, not a card's
        if sh.st.lock().unwrap().info.src.is_some() {
            sh.cmd(Cmd::Source(None));
            sh.until(5.0, |s| s.info.src.is_none().then_some(())).await;
        }
        let mut v = camera(&sh.st.lock().unwrap());
        if conn == Link::On {
            v["note"] = "connected already".into()
        }
        ok(v)
    }

    async fn camera_release(&self) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let _a = sh.act.lock().await;
        self.free()?;
        // also a start not shown recording yet: the camera may begin without a session to stop it
        if sh.st.lock().unwrap().rec.is_some() {
            return Err(
                "a recording this host started (or its start) is on the camera: capture_record stop stops it first \
                 (or the host's end does)"
                    .into(),
            );
        }
        if sh.st.lock().unwrap().info.conn != Link::Off {
            sh.cmd(Cmd::Disconnect);
            if sh
                .until(10.0, |s| (s.info.conn == Link::Off).then_some(()))
                .await
                .is_none()
            {
                return Err("the camera did not disconnect within 10 s".into());
            }
        }
        let v = camera(&sh.st.lock().unwrap());
        ok(v)
    }

    async fn run(
        &self,
        name: &str,
        a: JsonObject,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<CallToolResult, String> {
        match name {
            "status" => self.status(ctx),
            "camera_status" => ok(camera(&self.sh().st.lock().unwrap())),
            "album_list" => self.album_list(args(a)?).await,
            "album_thumbnail" => self.album_thumbnail(args(a)?).await,
            "card_list" => self.card_list(),
            "card_open" => self.card_open(args(a)?).await,
            "card_close" => self.card_close().await,
            "file_info" => self.file_info(args(a)?).await,
            "job_status" => ok(self.job(&args::<JobArgs>(a)?.job)?),
            "job_list" => self.job_list(args(a)?),
            "job_wait" => self.job_wait(args(a)?, ctx).await,
            "album_download" => self.album_download(args(a)?).await,
            "job_cancel" => self.job_cancel(args(a)?).await,
            "camera_connect" => self.camera_connect(args(a)?).await,
            "camera_release" => self.camera_release().await,
            "capture_status" => self.capture_status().await,
            "capture_mode" => self.capture_mode(args(a)?).await,
            "capture_set" => self.capture_set(args(a)?).await,
            "capture_photo" => self.capture_photo(args(a)?, ctx).await,
            "capture_record" => self.capture_record(args(a)?, ctx).await,
            "capture_mark" => self.capture_mark().await,
            "camera_favourite" => self.camera_favourite(args(a)?).await,
            "camera_delete" => self.camera_delete(args(a)?).await,
            "catalog" => self.catalog(),
            "render_still" => self.render_still(a, ctx).await,
            "export" => self.export(a).await,
            _ => Err(format!("{name}: not implemented")),
        }
    }
}

impl ServerHandler for Conn {
    fn get_info(&self) -> ServerConfig {
        let (all, t) = (self.host.tiers(), self.tiers());
        let ins = match (t.is_empty(), self.narrow) {
            (true, Some(n)) if !all.is_empty() => format!(
                "x5desk (Insta360 X5 camera control): this client's --tiers ({n}) leaves out every enabled tier \
                 ({all}), so it has no tools."
            ),
            (true, _) => OFF.to_string(),
            (false, _) => format!(
                "x5desk controls an Insta360 X5 360° camera and its files. Enabled tiers: {t}. Start with status; \
                 connect the camera (camera_connect) or open a card (card_open), list the album (album_list), and \
                 download into the agent folder (album_download, then job_wait). With the camera tier: \
                 capture_status, then capture_mode, capture_set, capture_photo, capture_record. The camera's clock \
                 is never set.{}",
                if self.sh().render.is_some() {
                    RENDERS
                } else {
                    ""
                }
            ),
        };
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_tool_list_changed()
                .build(),
        )
        .with_server_info(Implementation::new("x5desk", env!("CARGO_PKG_VERSION")))
        .with_instructions(ins)
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let gpu = self.sh().render.is_some();
        Ok(ListToolsResult::with_all_items(
            listed(self.tiers(), gpu).map(Def::tool).collect(),
        ))
    }

    async fn call_tool(
        &self,
        req: CallToolRequestParams,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let name = req.name.to_string();
        let Some(d) = TOOLS.iter().find(|d| d.name == name) else {
            return Err(ErrorData::invalid_params(
                format!("unknown tool {name:?}"),
                None,
            ));
        };
        let gpu = self.sh().render.is_some();
        let a = req.arguments.unwrap_or_default();
        let sum = serde_json::to_string(&a).unwrap_or_default();
        let t0 = Instant::now();
        // a proxying front's --tiers (it refuses the tools those leave out itself; the rest it sends along)
        let call = (ctx.meta.get(crate::front::NARROW))
            .and_then(Value::as_str)
            .map_or(Ok(Tiers::EVERY), Tiers::parse)
            .unwrap_or(Tiers::NONE);
        let r = CALL.scope(call, async {
            let _c = self.sh().call();
            let t = self.tiers();
            let narrow = self.narrow.map_or(call, |n| n.and(call));
            match (t.has(d.tier), Some(narrow).filter(|n| !n.has(d.tier))) {
                _ if d.gpu && !gpu => Err(format!("{name}: {NO_GPU}")),
                (true, _) => self.run(&name, a, &ctx).await,
                (false, Some(n)) => Err(left_out(&name, d.tier, n)),
                (false, _) => Err(format!(
                    "{name} is in the {0} tier, which is not enabled (enabled: {t}). The user enables it with \
                     `x5desk mcp enable --tiers <tiers>,{0}`",
                    d.tier.name()
                )),
            }
        });
        let r = r.await;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        self.sh().log.call(
            &self.client,
            &name,
            &sum,
            r.as_ref().err().map(String::as_str),
            ms,
        );
        Ok(match r {
            Ok(r) => r,
            Err(e) => CallToolResult::error(vec![ContentBlock::text(e)]),
        }
        .into())
    }

    fn accepted_subscription_filter(&self, _: &SubscriptionFilter) -> Option<SubscriptionFilter> {
        Some(SubscriptionFilter::builder().tools_list_changed().build())
    }

    async fn listen(&self, ctx: SubscriptionContext) -> Result<(), ErrorData> {
        let s = ctx.sink().clone();
        let id = s.id().clone();
        self.sinks.lock().unwrap().push(s);
        ctx.cancelled().await;
        self.sinks.lock().unwrap().retain(|x| *x.id() != id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64() {
        for (i, o) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
        ] {
            assert_eq!(b64(i.as_bytes()), o);
        }
        assert_eq!(b64(&[0xff, 0xfe, 0xfd]), "//79");
    }

    #[test]
    fn table() {
        let mut n = HashSet::new();
        for d in TOOLS {
            assert!(n.insert(d.name), "{} twice", d.name);
            let t = d.tool();
            assert_eq!(
                t.input_schema.get("type"),
                Some(&json!("object")),
                "{}",
                d.name
            );
            let a = t.annotations.unwrap();
            assert_eq!(a.read_only_hint, Some(d.tier == Tier::Read), "{}", d.name);
        }
        let del: Vec<&str> = (TOOLS.iter())
            .filter(|d| d.tier == Tier::Delete)
            .map(|d| d.name)
            .collect();
        assert_eq!(del, ["camera_delete"]);
        assert!(
            TOOLS.iter().all(|d| d.tier != Tier::Untested),
            "untested has no tools"
        );
        // the GPU tools: catalog read-only, the renders local, both taking a recording's path
        let gpu: Vec<(&str, Tier)> = (TOOLS.iter().filter(|d| d.gpu))
            .map(|d| (d.name, d.tier))
            .collect();
        assert_eq!(
            gpu,
            [
                ("catalog", Tier::Read),
                ("render_still", Tier::Local),
                ("export", Tier::Local)
            ]
        );
        for n in ["render_still", "export"] {
            let t = TOOLS.iter().find(|d| d.name == n).unwrap().tool();
            assert_eq!(t.input_schema["required"], json!(["path"]), "{n}");
            assert_eq!(t.annotations.unwrap().destructive_hint, Some(false), "{n}");
        }
        let d = TOOLS
            .iter()
            .find(|d| d.name == "camera_delete")
            .unwrap()
            .tool();
        assert_eq!(d.annotations.unwrap().destructive_hint, Some(true));
        let f = |s: &str| why(s);
        assert!(
            f("connect: could not open interface for exclusive access (0xe00002c5)")
                .contains("in use by another app")
        );
        assert!(f("connect: interface is busy (errno 16)").contains("in use by another app"));
        assert!(
            f("connect: camera busy: another client holds the session")
                .contains("another client holds")
        );
        assert!(f("connect: no camera in USB app mode").contains("Android App"));
        assert!(f("connect: failed to open device (error 5)").contains("in use by another app"));
        assert_eq!(
            f("connect: failed to open device (error 2)"),
            "connect: failed to open device (error 2)"
        );
        assert_eq!(f("connect: timeout"), "connect: timeout");
    }

    /// Without a backend the GPU tools are neither listed nor run (Windows, mcp's tests); with one, catalog is in the
    /// read tier and render_still and export in the local tier, the instructions name them, and status shows the GPU
    /// without starting it (the fake's start panics).
    #[tokio::test(flavor = "multi_thread")]
    async fn gpu_tools() {
        use crate::{
            Opts,
            config::Source,
            render::{Backend, Render, tests::Fake},
        };
        use rmcp::ServiceExt;
        const GPU: [&str; 3] = ["catalog", "render_still", "export"];
        let names = |t: &str, gpu| -> Vec<&str> {
            listed(Tiers::parse(t).unwrap(), gpu)
                .map(|d| d.name)
                .collect()
        };
        assert!(
            names("read,local,camera", false)
                .iter()
                .all(|n| !GPU.contains(n))
        );
        assert!(TOOLS.iter().filter(|d| d.gpu).map(|d| d.name).eq(GPU));
        let r = names("read", true);
        assert!(
            r.contains(&"catalog") && !r.contains(&"render_still"),
            "{r:?}"
        );
        let l = names("read,local", true);
        assert_eq!(l.len(), names("read,local", false).len() + 3, "{l:?}");
        for gpu in [false, true] {
            let render = gpu.then(|| Arc::new(Fake::Ok) as Arc<dyn Render>);
            let h = Host::new(Opts {
                sim: Some(Default::default()),
                tiers: Some(Source::Fixed(Tiers::parse("read,local").unwrap())),
                backend: Backend { render },
                ..Default::default()
            })
            .await
            .unwrap();
            let (up_c, up_s) = tokio::io::duplex(1 << 16);
            let (down_s, down_c) = tokio::io::duplex(1 << 16);
            let hh = h.clone();
            tokio::spawn(async move { hh.serve(up_s, down_s, "test", None).await });
            let c = ().serve((down_c, up_c)).await.unwrap();
            let mut got: Vec<String> = (c.list_all_tools().await.unwrap().into_iter())
                .map(|t| t.name.to_string())
                .collect();
            got.sort();
            let mut want = names("read,local", gpu);
            want.sort();
            assert_eq!(got, want, "gpu {gpu}");
            let ins = c.peer_info().unwrap().instructions.clone().unwrap();
            assert_eq!(ins.contains("render_still"), gpu, "{ins}");
            let call = |n: &str| {
                let p = CallToolRequestParams::new(n.to_string());
                c.call_tool(p)
            };
            let st = call("status").await.unwrap().structured_content.unwrap();
            let e = call("render_still").await.unwrap();
            let t = e.content[0].as_text().unwrap().text.clone();
            assert_eq!(e.is_error, Some(true));
            match gpu {
                false => {
                    assert_eq!(st["gpu"]["state"], "none", "{st}");
                    assert_eq!(t, format!("render_still: {NO_GPU}"));
                }
                true => {
                    assert_eq!(st["gpu"]["state"], "off", "{st}");
                    assert!(t.contains("missing field `path`"), "{t}");
                }
            }
            c.cancel().await.unwrap();
            h.shutdown().await;
        }
    }

    #[test]
    fn local_addresses() {
        for a in [
            "192.168.42.1",
            "10.0.0.5",
            "172.16.0.1",
            "172.31.255.254",
            "169.254.10.20",
            "fd12:3456::1",
            "fc00::1",
            "fe80::1",
            "::ffff:192.168.1.2",
        ] {
            assert!(lan(a.parse().unwrap()), "{a}");
        }
        for a in [
            "8.8.8.8",
            "203.0.113.9",
            "172.32.0.1",
            "127.0.0.1",
            "0.0.0.0",
            "255.255.255.255",
            "224.0.0.251",
            "::1",
            "::",
            "ff02::1",
            "2001:db8::1",
            "2606:4700::1111",
            "::ffff:8.8.8.8",
            "::ffff:127.0.0.1",
        ] {
            assert!(!lan(a.parse().unwrap()), "{a}");
        }
    }

    /// A download waits or runs for a job until it stopped, failed or finished, or a cancel reached it.
    #[test]
    fn wants() {
        let mut q = dl::Queue::default();
        let d = std::env::temp_dir().join(format!("x5desk-mcp-wanted-{}", std::process::id()));
        assert!(!wanted(&q, &d));
        q.add("/DCIM/Camera01/a", d.clone());
        assert!(wanted(&q, &d), "queued");
        q.v[0].st = dl::St::Running;
        assert!(wanted(&q, &d), "running");
        q.cancel(0);
        assert!(!wanted(&q, &d), "cancelled, the link not stopped yet");
        for st in [dl::St::Paused, dl::St::Done, dl::St::Failed] {
            q.v[0].st = st;
            assert!(!wanted(&q, &d), "{st:?}");
        }
    }

    /// An item's files at one number: free names, a stopped download resumed with its finished sibling kept, a
    /// finished item numbered past, names in use or no safe file names.
    #[test]
    fn places() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-place-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let us: Vec<String> = [
            "/DCIM/Camera01/VID_1_00_603.insv",
            "/DCIM/Camera01/LRV_1_01_603.lrv",
        ]
        .map(String::from)
        .into();
        let u: Vec<&String> = us.iter().collect();
        let n = |p: &[PathBuf]| -> Vec<String> {
            p.iter()
                .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
                .collect()
        };
        let none = HashSet::new();
        let mut q = dl::Queue::default();
        let p = place(&q, &d, &none, &u, &us).unwrap();
        assert_eq!(n(&p), ["VID_1_00_603.insv", "LRV_1_01_603.lrv"]);
        assert!(p.iter().all(|p| p.parent() == Some(d.as_path())));
        // a job cancelled between them: the video finished, the proxy stopped
        for (u, p) in us.iter().zip(&p) {
            q.add(u, p.clone())
        }
        std::fs::write(&p[0], b"v").unwrap();
        q.v[0].st = dl::St::Done;
        q.cancel(1);
        assert_eq!(
            place(&q, &d, &none, &u, &us).unwrap(),
            p,
            "resumed in place"
        );
        // taken in this call: past it
        let taken: HashSet<PathBuf> = p.iter().cloned().collect();
        assert_eq!(
            n(&place(&q, &d, &taken, &u, &us).unwrap())[0],
            "VID_1_00_603-2.insv"
        );
        // both finished: the item again goes past them
        std::fs::write(&p[1], b"l").unwrap();
        q.v[1].st = dl::St::Done;
        assert_eq!(
            n(&place(&q, &d, &none, &u, &us).unwrap()),
            ["VID_1_00_603-2.insv", "LRV_1_01_603-2.lrv"]
        );
        // only the proxy finished (a proxy download): the whole item goes past it too
        std::fs::remove_file(&p[0]).unwrap();
        q.v.remove(0);
        assert_eq!(
            n(&place(&q, &d, &none, &u, &us).unwrap())[0],
            "VID_1_00_603-2.insv"
        );
        for b in [
            r"..\..\AppData\Roaming\x.bat",
            r"C:\Users\Public\x.exe",
            "CON.insv",
            "a:b",
        ] {
            let bu = format!("/DCIM/Camera01/{b}");
            let e = place(&q, &d, &none, &[&bu], std::slice::from_ref(&bu)).unwrap_err();
            assert!(e.contains("no safe file name"), "{b}: {e}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }
}
