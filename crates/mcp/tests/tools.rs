//! The tools over an in-memory pipe: an rmcp client against a `--sim` host (cam's simulated X5 with its fake card, a
//! fake mounted card, temp dirs). Every byte the server writes passes a tap on its way to the client, so the scrub
//! checks see what a client would.

use mcp::{
    Host, Opts, Tiers,
    config::Source,
    render::{Backend, Catalog, GpuInfo, GpuSt, Kind, Lane, Plan, Render, Req, Run, Start},
    renders::{End, GONE, RJob, Renders},
    sim,
};
use rmcp::{
    ClientHandler, RoleClient, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, ProgressNotificationParam, ProtocolVersion,
        ServerNotification, SubscriptionFilter,
    },
    service::{
        ClientLifecycleMode, NotificationContext, RunningService, serve_client_with_lifecycle,
    },
};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering::Relaxed},
    },
    time::{Duration, Instant, SystemTime},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Progress and total of each progress notification.
type Progs = Arc<Mutex<Vec<(f64, Option<f64>)>>>;

/// What the client was told besides answers.
#[derive(Clone, Default)]
struct H {
    prog: Progs,
    changed: Arc<AtomicUsize>,
}

impl ClientHandler for H {
    async fn on_progress(&self, p: ProgressNotificationParam, _: NotificationContext<RoleClient>) {
        self.prog.lock().unwrap().push((p.progress, p.total))
    }

    async fn on_tool_list_changed(&self, _: NotificationContext<RoleClient>) {
        self.changed.fetch_add(1, Relaxed);
    }
}

struct Cli {
    s: RunningService<RoleClient, H>,
    h: H,
    /// every byte the server wrote to this client
    got: Arc<Mutex<Vec<u8>>>,
}

async fn host(tiers: Tiers, rate: u64, size: usize) -> Arc<Host> {
    Host::new(Opts {
        sim: Some(sim::Opts {
            rate,
            size,
            dir: None,
        }),
        tiers: Some(Source::Fixed(tiers)),
        ..Default::default()
    })
    .await
    .unwrap()
}

/// A client on `h`: the 2025-11-25 handshake (`initialize`) or 2026-07-28's discovery.
async fn connect(h: &Arc<Host>, modern: bool) -> Cli {
    connect_as(h, modern, None).await
}

/// A client on `h` whose connection is narrowed by `narrow` (a hosting process's own `--tiers`).
async fn connect_as(h: &Arc<Host>, modern: bool, narrow: Option<Tiers>) -> Cli {
    let (up_c, up_s) = tokio::io::duplex(1 << 16);
    let (down_s, mut down_t) = tokio::io::duplex(1 << 16);
    let (mut cli_t, cli_c) = tokio::io::duplex(1 << 16);
    let got = Arc::new(Mutex::new(vec![]));
    let g = got.clone();
    tokio::spawn(async move {
        let mut b = vec![0; 1 << 16];
        loop {
            match down_t.read(&mut b).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    g.lock().unwrap().extend_from_slice(&b[..n]);
                    if cli_t.write_all(&b[..n]).await.is_err() {
                        break;
                    }
                }
            }
        }
    });
    let hh = h.clone();
    tokio::spawn(async move { hh.serve(up_s, down_s, "test", narrow).await });
    let c = H::default();
    let s = match modern {
        false => c.clone().serve((cli_c, up_c)).await.unwrap(),
        true => {
            let m = ClientLifecycleMode::Discover {
                preferred_versions: vec![ProtocolVersion::V_2026_07_28],
            };
            serve_client_with_lifecycle(c.clone(), (cli_c, up_c), m)
                .await
                .unwrap()
        }
    };
    Cli { s, h: c, got }
}

async fn call(c: &Cli, name: &str, a: Value) -> CallToolResult {
    let p =
        CallToolRequestParams::new(name.to_string()).with_arguments(a.as_object().unwrap().clone());
    c.s.call_tool(p)
        .await
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn text(r: &CallToolResult) -> String {
    r.content
        .iter()
        .filter_map(|c| c.as_text())
        .map(|t| t.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The JSON of a successful call.
async fn ok(c: &Cli, name: &str, a: Value) -> Value {
    let r = call(c, name, a).await;
    assert_eq!(r.is_error, Some(false), "{name}: {}", text(&r));
    r.structured_content.unwrap_or(Value::Null)
}

/// The message of a failed call.
async fn err(c: &Cli, name: &str, a: Value) -> String {
    let r = call(c, name, a).await;
    assert_eq!(r.is_error, Some(true), "{name} should fail: {}", text(&r));
    text(&r)
}

async fn names(c: &Cli) -> Vec<String> {
    let mut v: Vec<String> =
        c.s.list_all_tools()
            .await
            .unwrap()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
    v.sort();
    v
}

fn unb64(s: &str) -> Vec<u8> {
    let v = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        _ => 63,
    };
    let b: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut o = vec![];
    for ch in b.chunks(4) {
        let n = ch
            .iter()
            .enumerate()
            .fold(0u32, |n, (i, &c)| n | (v(c) as u32) << (18 - 6 * i));
        o.extend_from_slice(&n.to_be_bytes()[1..ch.len()]);
    }
    o
}

const READ: [&str; 12] = [
    "album_list",
    "album_thumbnail",
    "camera_status",
    "capture_status",
    "card_close",
    "card_list",
    "card_open",
    "file_info",
    "job_list",
    "job_status",
    "job_wait",
    "status",
];

/// The camera tier's tools.
const CAMERA: [&str; 8] = [
    "camera_connect",
    "camera_release",
    "capture_mode",
    "capture_set",
    "capture_photo",
    "capture_record",
    "capture_mark",
    "camera_favourite",
];

/// The GPU tools, listed only with a backend: catalog in the read tier, the others in the local tier.
const GPU_READ: [&str; 1] = ["catalog"];
const GPU_LOCAL: [&str; 2] = ["export", "render_still"];

/// A backend with no GPU that refuses everything (the lists need one; `tests/render.rs` has the working fake).
struct NoGpu;

impl Render for NoGpu {
    fn catalog(&self) -> Catalog {
        Catalog::default()
    }
    fn check(&self, _: &Req) -> Result<Plan, String> {
        Err("no GPU".into())
    }
    fn start(&self, _: &Start) -> Result<GpuInfo, String> {
        Err("no GPU".into())
    }
    fn gpu(&self) -> GpuSt {
        GpuSt::Off
    }
    fn selftest(&self) -> Result<GpuInfo, (u8, String)> {
        Err((1, "no GPU".into()))
    }
    fn probe(&self) -> Result<(), String> {
        Err("no GPU".into())
    }
    fn submit(
        &self,
        _: Req,
        _: &Plan,
        _: &Path,
        _: media::part::Part,
        _: Lane,
    ) -> Result<Arc<dyn Run>, String> {
        Err("no GPU".into())
    }
    fn wedge(&self, _: Lane, _: &str) {}
    fn trim(&self) {}
}

async fn until(secs: f64, mut f: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs_f64(secs);
    while Instant::now() < end {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await
    }
    f()
}

/// `tools/list` per tier set (the GPU tools only with a backend), calls outside them refused naming the tier, nothing
/// enabled says how to enable, and a changed tiers file reaches clients of both protocol generations.
#[tokio::test(flavor = "multi_thread")]
async fn tiers() {
    let h = host(Tiers::parse("read").unwrap(), 8_000_000, 1000).await;
    let c = connect(&h, false).await;
    assert_eq!(names(&c).await, READ);
    assert!(
        err(&c, "album_download", json!({ "keys": ["x"] }))
            .await
            .contains("the local tier")
    );
    assert!(
        err(&c, "camera_connect", json!({}))
            .await
            .contains("the camera tier")
    );
    let p = CallToolRequestParams::new("format_card");
    assert!(
        c.s.call_tool(p).await.is_err(),
        "an unknown tool is a protocol error"
    );

    let h0 = host(Tiers::NONE, 8_000_000, 1000).await;
    let c0 = connect(&h0, false).await;
    assert!(names(&c0).await.is_empty());
    let i = c0.s.peer_info().unwrap().instructions.clone().unwrap();
    assert!(i.contains("x5desk mcp enable --tiers read,local"), "{i}");
    assert!(
        err(&c0, "status", json!({}))
            .await
            .contains("the read tier")
    );

    let mut all = READ.to_vec();
    all.extend(["album_download", "job_cancel"]);
    all.sort();
    let h1 = host(Tiers::parse("read,local").unwrap(), 8_000_000, 1000).await;
    assert_eq!(names(&connect(&h1, false).await).await, all);

    // the tiers file changes: read → read, local, camera
    let d = std::env::temp_dir().join(format!("x5desk-mcp-tiers-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join("mcp");
    mcp::config::enable_at(&f, Tiers::parse("read").unwrap()).unwrap();
    let h2 = Host::new(Opts {
        sim: Some(Default::default()),
        tiers: Some(Source::File(f.clone())),
        narrow: Some(Tiers::parse("read,local,camera").unwrap()),
        ..Default::default()
    })
    .await
    .unwrap();
    let (old, new) = (connect(&h2, false).await, connect(&h2, true).await);
    let mut sub = new
        .s
        .listen(SubscriptionFilter::builder().tools_list_changed().build())
        .await
        .unwrap();
    assert_eq!(names(&old).await, READ);
    assert_eq!(names(&new).await, READ);
    let st = ok(&new, "status", json!({})).await;
    assert_eq!(st["protocol"], "2026-07-28");
    assert_eq!(
        ok(&old, "status", json!({})).await["protocol"],
        "2025-11-25"
    );
    // delete is named but narrowed away by the command line
    mcp::config::enable_at(&f, Tiers::parse("read,local,camera,delete").unwrap()).unwrap();
    let n = tokio::time::timeout(Duration::from_secs(6), sub.next())
        .await
        .unwrap()
        .unwrap();
    assert!(
        matches!(n, Some(ServerNotification::ToolListChangedNotification(_))),
        "{n:?}"
    );
    assert!(until(3.0, || old.h.changed.load(Relaxed) == 1).await);
    let mut want = all.clone();
    want.extend(CAMERA);
    want.sort();
    assert_eq!(names(&old).await, want);
    assert_eq!(names(&new).await, want);
    assert_eq!(h2.tiers().to_string(), "read,local,camera");
    drop(sub);
    let _ = std::fs::remove_dir_all(&d);
    h2.shutdown().await;

    // with a backend the GPU tools join their tiers' lists
    for (t, mut want) in [("read", READ.to_vec()), ("read,local", all.clone())] {
        let h = Host::new(Opts {
            sim: Some(Default::default()),
            tiers: Some(Source::Fixed(Tiers::parse(t).unwrap())),
            backend: Backend {
                render: Some(Arc::new(NoGpu)),
            },
            ..Default::default()
        })
        .await
        .unwrap();
        want.extend(GPU_READ);
        if t != "read" {
            want.extend(GPU_LOCAL)
        }
        want.sort();
        assert_eq!(names(&connect(&h, false).await).await, want, "{t}");
        h.shutdown().await;
    }
}

/// A connection's own `--tiers` narrow only it: its tools, `status`, instructions and errors name its set (a call
/// left out is not sent to `enable`), the host's other connections keep the host's tiers.
#[tokio::test(flavor = "multi_thread")]
async fn narrowed_connection() {
    let h = host(Tiers::parse("read,local").unwrap(), 8_000_000, 1000).await;
    let mine = connect_as(&h, false, Tiers::parse("read").ok()).await;
    let other = connect(&h, false).await;
    assert_eq!(names(&mine).await, READ);
    assert_eq!(
        ok(&mine, "status", json!({})).await["tiers"],
        json!(["read"])
    );
    let i = mine.s.peer_info().unwrap().instructions.clone().unwrap();
    assert!(i.contains("Enabled tiers: read."), "{i}");
    let e = err(&mine, "album_download", json!({ "keys": ["x"] })).await;
    assert!(
        e.contains("this client's --tiers leaves out") && !e.contains("enable"),
        "{e}"
    );
    // a tier neither enabled nor in its --tiers: --tiers is what stops it
    let e = err(&mine, "camera_connect", json!({})).await;
    assert!(e.contains("--tiers leaves out"), "{e}");
    let mut all = READ.to_vec();
    all.extend(["album_download", "job_cancel"]);
    all.sort();
    assert_eq!(names(&other).await, all);
    assert_eq!(
        ok(&other, "status", json!({})).await["tiers"],
        json!(["read", "local"])
    );
    // --tiers allows camera, the host does not: enabling it is the way
    let wide = connect_as(&h, false, Tiers::parse("read,camera").ok()).await;
    let e = err(&wide, "camera_connect", json!({})).await;
    assert!(
        e.contains("x5desk mcp enable --tiers <tiers>,camera"),
        "{e}"
    );
    // nothing left: said so, not sent to `enable`
    let none = connect_as(&h, false, Tiers::parse("camera").ok()).await;
    assert!(names(&none).await.is_empty());
    let i = none.s.peer_info().unwrap().instructions.clone().unwrap();
    assert!(
        i.contains("leaves out every enabled tier (read,local)"),
        "{i}"
    );
    h.shutdown().await;
}

/// The files of an item in the agent folder equal the camera's.
fn same(sim: &sim::Sim, files: &Value) {
    let c = sim.fake.lock().unwrap();
    for f in files.as_array().unwrap() {
        let p = PathBuf::from(f["path"].as_str().unwrap());
        let n = p.file_name().unwrap().to_str().unwrap();
        // name-2.insv is a copy of name.insv
        let orig = n.replace("-2.", ".").replace("-3.", ".");
        let u = c
            .data
            .keys()
            .find(|u| u.ends_with(&format!("/{orig}")))
            .unwrap();
        assert_eq!(
            std::fs::read(&p).unwrap(),
            **c.data.get(u).unwrap(),
            "{}",
            p.display()
        );
    }
}

fn key(items: &Value, file: &str) -> String {
    let i = items
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["files"][0] == file);
    i.unwrap_or_else(|| panic!("no {file} in {items}"))["key"]
        .as_str()
        .unwrap()
        .into()
}

/// The tools on the simulated camera and card: status, cards, connect, album, thumbnail, downloads with progress,
/// numbering past taken names and resuming, cancel, file_info in and out of the roots, job handles, release.
#[tokio::test(flavor = "multi_thread")]
async fn camera() {
    // 3 MB proxies and videos at 1.6 MB/s: a download takes seconds, long enough for progress and a cancel
    let h = host(Tiers::SIM, 400_000, 3_000_000).await;
    let c = connect(&h, false).await;
    let sim = h.sim().unwrap();
    let st = ok(&c, "status", json!({})).await;
    assert_eq!(st["host"]["kind"], "headless");
    assert_eq!(st["host"]["id"], h.id());
    assert_eq!(st["tiers"], json!(["read", "local", "camera"]));
    assert_eq!(st["camera"]["connection"], "off");
    assert_eq!(st["agent_folder"], sim.agent.display().to_string());

    // the fake card, mounted: opened, listed, closed again
    let card = sim.card.display().to_string();
    let mut cards = Value::Null;
    for _ in 0..100 {
        cards = ok(&c, "card_list", json!({})).await;
        if cards["cards"].as_array().is_some_and(|v| !v.is_empty()) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await
    }
    assert_eq!(cards["cards"][0]["root"], card, "{cards}");
    let o = ok(&c, "card_open", json!({ "path": card })).await;
    assert_eq!(
        (o["album_source"].as_str(), o["items"].as_u64()),
        (Some(card.as_str()), Some(2))
    );
    let l = ok(&c, "album_list", json!({})).await;
    assert_eq!(
        (l["source"].as_str(), l["total"].as_u64()),
        (Some(card.as_str()), Some(2))
    );
    let k = key(&l["items"], sim::CARD[0]);
    let it = (l["items"].as_array().unwrap().iter())
        .find(|i| i["key"] == k)
        .unwrap();
    let at = sim.card.join("DCIM").join("Camera01").join(sim::CARD[0]);
    assert_eq!(
        it["path"],
        at.display().to_string(),
        "a card's item: its file's path"
    );
    // a card's files are imported like the camera's
    let j = ok(
        &c,
        "album_download",
        json!({ "keys": [k], "subdir": "card" }),
    )
    .await;
    let w = ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 20 })).await;
    assert_eq!(w["state"], "done", "{w}");
    for f in w["files"].as_array().unwrap() {
        let p = PathBuf::from(f["path"].as_str().unwrap());
        let n = p.file_name().unwrap();
        assert!(p.starts_with(sim.agent.join("card")));
        let src = sim.card.join("DCIM").join("Camera01").join(n);
        assert_eq!(std::fs::read(&p).unwrap(), std::fs::read(src).unwrap());
    }
    assert_eq!(
        ok(&c, "card_close", json!({})).await["album_source"],
        "camera"
    );
    let e = err(&c, "album_list", json!({})).await;
    assert!(e.contains("no camera connected and no card open"), "{e}");
    assert!(
        err(&c, "card_open", json!({ "path": "/" }))
            .await
            .contains("not a mounted X5 card")
    );

    // the camera, held by another process first
    sim.held.store(true, Relaxed);
    let e = err(&c, "camera_connect", json!({})).await;
    assert!(e.contains("in use by another app (x5desk?)"), "{e}");
    sim.held.store(false, Relaxed);
    let s = ok(&c, "camera_connect", json!({})).await;
    assert_eq!(
        (s["connection"].as_str(), s["model"].as_str()),
        (Some("on"), Some("Insta360 X5"))
    );
    assert_eq!(
        (s["serial"].as_str(), s["access_point_ssid"].as_str()),
        (Some("SIM0001"), Some("X5 SIM001.OSC"))
    );
    assert_eq!(s["album_source"], "camera");
    // the password is known to the host, but no answer carries it
    let pw = h.with(|s| s.info.wifi.clone()).unwrap().1;
    assert_eq!(pw, sim::AP);
    let s2 = ok(&c, "camera_connect", json!({})).await;
    assert_eq!(s2["connection"], "on", "connected already: the same answer");
    assert!(
        err(&c, "camera_connect", json!({ "wifi_ip": "192.168.1.2" }))
            .await
            .contains("no Wi-Fi")
    );
    assert!(
        err(&c, "camera_connect", json!({ "wifi_ip": "nope" }))
            .await
            .contains("wifi_ip")
    );
    // the client id goes only to an address a camera can have here
    for a in [
        "203.0.113.9",
        "8.8.8.8",
        "127.0.0.1",
        "0.0.0.0",
        "224.0.0.251",
        "::1",
        "2001:db8::1",
    ] {
        let e = err(&c, "camera_connect", json!({ "wifi_ip": a })).await;
        assert!(e.contains("on the local network"), "{a}: {e}");
    }
    for a in ["169.254.1.2", "fd00::5", "fe80::1"] {
        let e = err(&c, "camera_connect", json!({ "wifi_ip": a })).await;
        assert!(e.contains("no Wi-Fi"), "{a}: {e}");
    }
    let cs = ok(&c, "camera_status", json!({})).await;
    assert_eq!(cs["battery_percent"], 80);
    assert_eq!(cs["storage_free_bytes"], 100_000_000_000u64);

    let l = ok(&c, "album_list", json!({})).await;
    assert_eq!(l["total"], 5, "{l}");
    let items = &l["items"];
    assert_eq!(items[0]["date"], "2026-05-03 19:06:39", "newest first");
    assert!(
        items[0]["path"].is_null(),
        "a camera's item has no local path"
    );
    assert_eq!(
        ok(&c, "album_list", json!({ "kind": "video" })).await["total"],
        3
    );
    let p = ok(
        &c,
        "album_list",
        json!({ "kind": "photo", "limit": 1, "offset": 1 }),
    )
    .await;
    assert_eq!(
        (
            p["total"].as_u64(),
            p["items"].as_array().unwrap().len(),
            p["more"].as_bool()
        ),
        (Some(2), 1, Some(false))
    );
    assert!(
        err(&c, "album_list", json!({ "limit": "x" }))
            .await
            .contains("arguments")
    );
    assert!(
        err(&c, "album_list", json!({ "colour": 1 }))
            .await
            .contains("unknown field")
    );
    let vid = key(items, "VID_20260116_152935_00_603.insv");
    let it = items
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["key"] == vid)
        .unwrap();
    assert_eq!(
        it["files"],
        json!([
            "VID_20260116_152935_00_603.insv",
            "LRV_20260116_152935_01_603.lrv"
        ])
    );
    assert_eq!(
        (it["kind"].as_str(), it["duration_ms"].as_u64()),
        (Some("video"), Some(61_000))
    );
    assert_eq!(
        (it["width"].as_u64(), it["local"].is_null()),
        (Some(7680), true)
    );

    // a thumbnail: JPEG of the 320 × 160 picture
    let r = call(&c, "album_thumbnail", json!({ "key": vid })).await;
    assert_eq!(r.is_error, Some(false), "{}", text(&r));
    let img = r.content.iter().find_map(|c| c.as_image()).unwrap();
    assert_eq!(img.mime_type, "image/jpeg");
    let j = unb64(&img.data);
    assert_eq!(
        (&j[..2], &j[j.len() - 2..]),
        (&[0xff, 0xd8][..], &[0xff, 0xd9][..])
    );
    let mut px = vec![];
    assert_eq!(
        media::mjpeg::Mjpeg::default().rgba(&j, &mut px).unwrap(),
        (320, 160)
    );
    assert!(text(&r).contains("320 x 160"));
    assert!(
        err(&c, "album_thumbnail", json!({ "key": "/DCIM/nope" }))
            .await
            .contains("album_list")
    );

    // a download with progress, its bytes the camera's
    let j = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    let job = j["job"].as_str().unwrap().to_string();
    assert!(job.starts_with(&format!("{}-", h.id())), "{job}");
    assert_eq!(j["files"].as_array().unwrap().len(), 2);
    let w = ok(&c, "job_wait", json!({ "job": job, "wait_s": 60 })).await;
    assert_eq!(
        (w["state"].as_str(), w["finished"].as_bool()),
        (Some("done"), Some(true)),
        "{w}"
    );
    assert_eq!(w["done_bytes"], 6_000_000);
    same(sim, &w["files"]);
    let pr = c.h.prog.lock().unwrap().clone();
    assert!(pr.len() >= 2, "progress about every second: {pr:?}");
    assert!(
        pr.windows(2).all(|w| w[1].0 > w[0].0),
        "progress grows: {pr:?}"
    );
    assert!(
        pr.iter().all(|p| p.1.is_none_or(|t| t == 6e6)),
        "a total only once both files' sizes are known: {pr:?}"
    );
    assert!(
        pr.iter().any(|p| p.1.is_none()),
        "the proxy's size unknown at first: {pr:?}"
    );
    let l = ok(&c, "album_list", json!({})).await;
    let it = l["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["key"] == vid)
        .unwrap()
        .clone();
    assert_eq!(it["local"].as_array().map(Vec::len), Some(2), "{it}");
    assert_eq!(
        ok(&c, "job_status", json!({ "job": job })).await["state"],
        "done"
    );
    assert_eq!(
        ok(&c, "job_list", json!({})).await["jobs"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    // again: numbered past the files there; cancelled, its partial data kept; again: resumed under the same names
    let j2 = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    let names: Vec<&str> = j2["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["path"].as_str().unwrap())
        .collect();
    assert!(
        names[0].ends_with("VID_20260116_152935_00_603-2.insv"),
        "{names:?}"
    );
    assert!(
        names[1].ends_with("LRV_20260116_152935_01_603-2.lrv"),
        "{names:?}"
    );
    let part = PathBuf::from(format!("{}.part", names[0]));
    assert!(
        until(10.0, || part.metadata().is_ok_and(|m| m.len() > 0)).await,
        "the download started"
    );
    let x = ok(&c, "job_cancel", json!({ "job": j2["job"] })).await;
    assert_eq!(x["state"], "cancelled", "{x}");
    assert!(
        part.exists() && !Path::new(names[0]).exists(),
        "the partial file stays"
    );
    let j3 = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    assert_eq!(
        j3["files"][0]["path"], names[0],
        "a stopped download resumes, not numbered again"
    );
    let w = ok(&c, "job_wait", json!({ "job": j3["job"], "wait_s": 60 })).await;
    assert_eq!(w["state"], "done", "{w}");
    same(sim, &w["files"]);

    // a name another process is writing (.part) is numbered past; a subfolder; `..` refused
    let photo = key(items, "IMG_20260116_174103_00_620.jpg");
    std::fs::write(sim.agent.join("IMG_20260116_174103_00_620.jpg.part"), b"x").unwrap();
    let j = ok(&c, "album_download", json!({ "keys": [photo] })).await;
    assert!(
        j["files"][0]["path"]
            .as_str()
            .unwrap()
            .ends_with("IMG_20260116_174103_00_620-2.jpg"),
        "{j}"
    );
    let j = ok(
        &c,
        "album_download",
        json!({ "keys": [photo], "subdir": "trip/day1" }),
    )
    .await;
    let p = j["files"][0]["path"].as_str().unwrap();
    assert!(
        p.starts_with(sim.agent.join("trip").join("day1").to_str().unwrap()),
        "{p}"
    );
    let w = ok(&c, "job_wait", json!({ "job": j["job"] })).await;
    same(sim, &w["files"]);
    for s in ["../out", "a/../../out", "/tmp"] {
        assert!(
            err(
                &c,
                "album_download",
                json!({ "keys": [photo], "subdir": s })
            )
            .await
            .contains(s)
        );
    }
    let proxy = ok(
        &c,
        "album_download",
        json!({ "keys": [vid], "proxy": true, "subdir": "lrv" }),
    )
    .await;
    assert_eq!(proxy["files"].as_array().unwrap().len(), 1);
    assert!(
        err(
            &c,
            "album_download",
            json!({ "keys": [photo], "proxy": true })
        )
        .await
        .contains("no proxy")
    );
    assert!(
        err(&c, "album_download", json!({ "keys": ["/DCIM/x"] }))
            .await
            .contains("album_list")
    );
    assert!(
        err(&c, "album_download", json!({ "keys": [] }))
            .await
            .contains("keys")
    );

    // file_info: in the roots, out of them, not an X5 file
    let f = ok(&c, "file_info", json!({ "path": names[0] })).await;
    assert_eq!(f["size_bytes"], 3_000_000);
    assert_eq!(f["kind"], "insv");
    let out = std::env::temp_dir().join(format!("x5desk-mcp-out-{}.insv", std::process::id()));
    std::fs::write(&out, b"x").unwrap();
    let there = err(&c, "file_info", json!({ "path": out })).await;
    assert!(there.contains("not under a read root"), "{there}");
    let _ = std::fs::remove_file(&out);
    assert_eq!(
        err(&c, "file_info", json!({ "path": out })).await,
        there,
        "outside the roots, a missing file reads as one that is there"
    );
    std::fs::write(sim.agent.join("notes.txt"), b"x").unwrap();
    assert!(
        err(
            &c,
            "file_info",
            json!({ "path": sim.agent.join("notes.txt") })
        )
        .await
        .contains("not an X5 file")
    );
    let up = sim
        .agent
        .join("..")
        .join("agent")
        .join(Path::new(names[0]).file_name().unwrap());
    assert!(
        ok(&c, "file_info", json!({ "path": up })).await["path"]
            .as_str()
            .is_some()
    );

    // job handles: another host's, a bad one, an unknown one of ours
    assert!(
        err(&c, "job_status", json!({ "job": "zz.0000-1" }))
            .await
            .contains("its host has ended")
    );
    assert!(
        err(&c, "job_status", json!({ "job": "nonsense" }))
            .await
            .contains("not a job handle")
    );
    assert!(
        err(&c, "job_wait", json!({ "job": format!("{}-99", h.id()) }))
            .await
            .contains("unknown job")
    );

    assert_eq!(
        ok(&c, "camera_release", json!({})).await["connection"],
        "off"
    );
    assert_eq!(
        ok(&c, "camera_release", json!({})).await["connection"],
        "off"
    );
    // nothing secret went out, nor into the log
    let got = c.got.lock().unwrap().clone();
    for m in [sim::MARKER, sim::AP] {
        assert!(!got.windows(m.len()).any(|w| w == m.as_bytes()), "{m}");
    }
    let log = std::fs::read_to_string(&sim.log).unwrap();
    assert!(
        log.contains("test album_download") && !log.contains(sim::MARKER) && !log.contains(sim::AP),
        "{log}"
    );
    h.shutdown().await;
}

/// Jobs sharing downloads: a second call for an item that downloads joins its files, and cancelling it stops
/// nothing; a job cancelled between an item's files resumes under the same names with the finished file kept, and
/// cancelling that job again leaves the job that resumed them, which it shows as they were.
#[tokio::test(flavor = "multi_thread")]
async fn shared() {
    let h = host(Tiers::SIM, 400_000, 3_000_000).await;
    let c = connect(&h, false).await;
    let sim = h.sim().unwrap();
    ok(&c, "camera_connect", json!({})).await;
    let l = ok(&c, "album_list", json!({})).await;
    let vid = key(&l["items"], "VID_20260116_152935_00_603.insv");
    let paths = |j: &Value| -> Vec<String> {
        (j["files"].as_array().unwrap().iter())
            .map(|f| f["path"].as_str().unwrap().to_string())
            .collect()
    };
    let state = |j: &Value, i: usize| j["files"][i]["state"].as_str().unwrap().to_string();

    let a = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    let b = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    assert_eq!(paths(&b), paths(&a), "the same files");
    let x = ok(&c, "job_cancel", json!({ "job": b["job"] })).await;
    assert_eq!(x["state"], "cancelled", "{x}");
    let s = ok(&c, "job_status", json!({ "job": a["job"] })).await;
    assert!(
        matches!(s["state"].as_str(), Some("queued" | "running")),
        "{s}"
    );

    // the video done, the proxy under way: cancelled
    let mut s = Value::Null;
    for _ in 0..400 {
        s = ok(&c, "job_status", json!({ "job": a["job"] })).await;
        if state(&s, 0) == "done" && s["files"][1]["done_bytes"].as_u64() > Some(0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await
    }
    assert_eq!(
        (state(&s, 0), state(&s, 1)),
        ("done".into(), "running".into()),
        "{s}"
    );
    let x = ok(&c, "job_cancel", json!({ "job": a["job"] })).await;
    assert_eq!(
        (x["state"].as_str(), state(&x, 1)),
        (Some("cancelled"), "cancelled".into()),
        "{x}"
    );
    let (v, lrv) = (paths(&a)[0].clone(), paths(&a)[1].clone());
    assert!(Path::new(&v).is_file() && Path::new(&format!("{lrv}.part")).is_file());

    // again: the same names, the video kept (not fetched again), the proxy resumed
    let u = "/DCIM/Camera01/VID_20260116_152935_00_603.insv";
    let opens = sim.fake.lock().unwrap().opens.get(u).copied();
    let j = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    assert_eq!(paths(&j), paths(&a), "{j}");
    assert_eq!(state(&j, 0), "done", "{j}");
    // the cancelled job stays so, and cancelling it again leaves the new job's download
    let x = ok(&c, "job_cancel", json!({ "job": a["job"] })).await;
    assert_eq!(x["state"], "cancelled", "{x}");
    let w = ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 60 })).await;
    assert_eq!(w["state"], "done", "{w}");
    same(sim, &w["files"]);
    assert_eq!(sim.fake.lock().unwrap().opens.get(u).copied(), opens);
    assert!(!Path::new(&format!("{lrv}.part")).exists());
    // the proxy, which the new job resumed, as it was then
    let s = ok(&c, "job_status", json!({ "job": a["job"] })).await;
    assert_eq!(
        (s["state"].as_str(), state(&s, 0), state(&s, 1)),
        (Some("cancelled"), "done".into(), "cancelled".into()),
        "{s}"
    );
    h.shutdown().await;
}

/// A job's status once its bytes flow (10 s at most).
async fn flowing(c: &Cli, job: &Value) -> Value {
    let mut s = Value::Null;
    for _ in 0..400 {
        s = ok(c, "job_status", json!({ "job": job })).await;
        if s["done_bytes"].as_u64() > Some(0) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await
    }
    s
}

/// Its download stopped: the job's bytes do not grow (the simulated camera sends 128 kB per connection every 0.33 s
/// here), none of its files is there.
async fn stopped(c: &Cli, job: &Value) -> bool {
    let a = ok(c, "job_status", json!({ "job": job })).await;
    tokio::time::sleep(Duration::from_millis(800)).await;
    let b = ok(c, "job_status", json!({ "job": job })).await;
    let none = (b["files"].as_array().unwrap().iter())
        .all(|f| !Path::new(f["path"].as_str().unwrap()).exists());
    a["done_bytes"] == b["done_bytes"] && none
}

/// A file a newer call starts again is its job's alone: one that failed (its folder not writable) and the call that
/// retries it, one done whose file went and the call that fetches it again. Cancelling the newer job stops that
/// download; the older job shows the file as it was.
#[tokio::test(flavor = "multi_thread")]
async fn taken_over() {
    // 2 MB files at 1.6 MB/s
    let h = host(Tiers::SIM, 400_000, 2_000_000).await;
    let c = connect(&h, false).await;
    ok(&c, "camera_connect", json!({})).await;
    let l = ok(&c, "album_list", json!({})).await;
    let vid = key(&l["items"], "VID_20260116_152935_00_603.insv");
    let paths = |j: &Value| -> Vec<String> {
        (j["files"].as_array().unwrap().iter())
            .map(|f| f["path"].as_str().unwrap().to_string())
            .collect()
    };

    // root writes into a read-only folder all the same
    #[cfg(unix)]
    if unsafe { libc::geteuid() } != 0 {
        use std::os::unix::fs::PermissionsExt;
        let states = |j: &Value| -> Vec<String> {
            (j["files"].as_array().unwrap().iter())
                .map(|f| f["state"].as_str().unwrap().to_string())
                .collect()
        };
        let ro = h.agent().join("ro");
        std::fs::create_dir_all(&ro).unwrap();
        let mode = |m| std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(m)).unwrap();
        mode(0o500);
        let args = json!({ "keys": [vid], "subdir": "ro" });
        let a = ok(&c, "album_download", args.clone()).await;
        let w = ok(&c, "job_wait", json!({ "job": a["job"], "wait_s": 20 })).await;
        assert_eq!(w["state"], "failed", "{w}");
        mode(0o700);
        let j = ok(&c, "album_download", args).await;
        assert_eq!(paths(&j), paths(&a), "retried under the same names: {j}");
        let s = flowing(&c, &j["job"]).await;
        assert_eq!(s["state"], "running", "{s}");
        let x = ok(&c, "job_cancel", json!({ "job": j["job"] })).await;
        assert_eq!(x["state"], "cancelled", "{x}");
        assert!(stopped(&c, &j["job"]).await, "the retry stopped");
        let s = ok(&c, "job_status", json!({ "job": a["job"] })).await;
        assert_eq!(
            (s["state"].as_str(), states(&s)),
            (Some("failed"), vec!["failed".to_string(); 2]),
            "as it was when retried: {s}"
        );
        assert!(s["files"][0]["error"].is_string(), "{s}");
    }

    let args = json!({ "keys": [vid], "proxy": true, "subdir": "again" });
    let a = ok(&c, "album_download", args.clone()).await;
    let w = ok(&c, "job_wait", json!({ "job": a["job"], "wait_s": 20 })).await;
    assert_eq!(w["state"], "done", "{w}");
    let f = paths(&w)[0].clone();
    std::fs::remove_file(&f).unwrap();
    let b = ok(&c, "album_download", args).await;
    assert_eq!(paths(&b), std::slice::from_ref(&f), "the name free again");
    let s = flowing(&c, &b["job"]).await;
    assert_eq!(s["state"], "running", "{s}");
    let s = ok(&c, "job_status", json!({ "job": a["job"] })).await;
    assert_eq!(
        (s["state"].as_str(), s["done_bytes"].as_u64()),
        (Some("done"), Some(2_000_000)),
        "not the new transfer: {s}"
    );
    let x = ok(&c, "job_cancel", json!({ "job": b["job"] })).await;
    assert_eq!(x["state"], "cancelled", "{x}");
    assert!(stopped(&c, &b["job"]).await, "the download again stopped");
    assert!(Path::new(&format!("{f}.part")).exists());
    assert_eq!(
        ok(&c, "job_status", json!({ "job": a["job"] })).await["state"],
        "done"
    );
    h.shutdown().await;
}

/// The queue and the jobs outlive their host. One ends mid-download while the next starts, which waits for it to let
/// go of them: the next shows the job under its handle, queued with its `.part`, and once the camera is connected
/// resumes it where it stopped; an item whose job was cancelled resumes under its names when downloaded again, the
/// old job showing it as it was. A third host still has them all.
#[tokio::test(flavor = "multi_thread")]
async fn across_hosts() {
    let d = std::env::temp_dir().join(format!("x5desk-mcp-across-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    // 3 MB files at 1.6 MB/s
    let o = || Opts {
        sim: Some(sim::Opts {
            rate: 400_000,
            size: 3_000_000,
            dir: Some(d.clone()),
        }),
        tiers: Some(Source::Fixed(Tiers::SIM)),
        ..Default::default()
    };
    let paths = |j: &Value| -> Vec<String> {
        (j["files"].as_array().unwrap().iter())
            .map(|f| f["path"].as_str().unwrap().to_string())
            .collect()
    };
    let a = Host::new(o()).await.unwrap();
    let c = connect(&a, false).await;
    ok(&c, "camera_connect", json!({})).await;
    let l = ok(&c, "album_list", json!({})).await;
    let vid = key(&l["items"], "VID_20260116_152935_00_603.insv");
    let proxy =
        json!({ "keys": [key(&l["items"], "VID_20260117_141148_00_635.insv")], "proxy": true });
    let p = ok(&c, "album_download", proxy.clone()).await;
    assert_eq!(flowing(&c, &p["job"]).await["state"], "running");
    ok(&c, "job_cancel", json!({ "job": p["job"] })).await;
    let j = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    let mut s = Value::Null;
    for _ in 0..400 {
        s = ok(&c, "job_status", json!({ "job": j["job"] })).await;
        if s["done_bytes"].as_u64() >= Some(2_000_000) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await
    }
    assert_eq!(s["state"], "running", "{s}");
    let had = s["done_bytes"].as_u64().unwrap();

    // the host ends; the next one starts meanwhile and waits for the queue
    let end = tokio::spawn({
        let a = a.clone();
        async move { a.shutdown().await }
    });
    let b = Host::new(o()).await.unwrap();
    end.await.unwrap();
    assert!(b.id() != a.id());
    let c = connect(&b, false).await;
    let s = ok(&c, "job_status", json!({ "job": j["job"] })).await;
    assert_eq!(
        (s["state"].as_str(), s["note"].as_str()),
        (
            Some("queued"),
            Some("waits for the camera: call camera_connect")
        ),
        "{s}"
    );
    let f = paths(&s);
    assert!(
        Path::new(&format!("{}.part", f[0])).exists() && !Path::new(&f[0]).exists(),
        "{f:?}"
    );
    let listed: Vec<Value> = (ok(&c, "job_list", json!({})).await["jobs"].as_array())
        .unwrap()
        .iter()
        .map(|j| j["job"].clone())
        .collect();
    assert_eq!(listed, [j["job"].clone(), p["job"].clone()], "newest first");
    assert_eq!(
        ok(&c, "status", json!({})).await["jobs"],
        json!({ "running": 0, "queued": 1, "finished": 1 })
    );
    // connected, it resumes: its first bytes are those it had, not a new start's
    ok(&c, "camera_connect", json!({})).await;
    let s = flowing(&c, &j["job"]).await;
    assert!(
        s["done_bytes"].as_u64().unwrap() >= had * 3 / 4,
        "resumed from {had}: {s}"
    );
    let w = ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 60 })).await;
    assert_eq!(w["state"], "done", "{w}");
    same(b.sim().unwrap(), &w["files"]);
    // the cancelled item again: under its names, the old job as it was
    let q = ok(&c, "album_download", proxy).await;
    assert_eq!(paths(&q), paths(&p), "resumed, not numbered past");
    assert!(
        q["job"]
            .as_str()
            .unwrap()
            .starts_with(&format!("{}-", b.id()))
    );
    let w = ok(&c, "job_wait", json!({ "job": q["job"], "wait_s": 60 })).await;
    assert_eq!(w["state"], "done", "{w}");
    same(b.sim().unwrap(), &w["files"]);
    assert_eq!(
        ok(&c, "job_status", json!({ "job": p["job"] })).await["state"],
        "cancelled"
    );
    b.shutdown().await;

    // a third host has the three jobs as they ended
    let h = Host::new(o()).await.unwrap();
    let c = connect(&h, false).await;
    let st: Vec<(Value, Value)> = (ok(&c, "job_list", json!({})).await["jobs"].as_array())
        .unwrap()
        .iter()
        .map(|j| (j["job"].clone(), j["state"].clone()))
        .collect();
    assert_eq!(
        st,
        [
            (q["job"].clone(), json!("done")),
            (j["job"].clone(), json!("done")),
            (p["job"].clone(), json!("cancelled"))
        ]
    );
    let log = std::fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(log.contains("kept from earlier hosts: 3 jobs"), "{log}");
    h.shutdown().await;
    std::fs::remove_dir_all(&d).unwrap();
}

/// Render jobs an earlier host left: the next host settles an unfinished one by its `.part`'s identity (cancelled,
/// the `.part` gone) and shows them in the job tools beside the downloads, newest first, by kind and limit; job_wait
/// and job_cancel answer a finished one at once, as it ended.
#[tokio::test(flavor = "multi_thread")]
async fn renders_kept() {
    let d = std::env::temp_dir().join(format!("x5desk-mcp-rkept-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let agent = d.join("agent");
    std::fs::create_dir_all(&agent).unwrap();
    let (mut r, _) = Renders::load(d.join("mcp-renders")).unwrap();
    let t0 = SystemTime::now() - Duration::from_secs(60);
    let job = |kind: Kind, dst: PathBuf, at: SystemTime| RJob {
        h: String::new(),
        kind,
        at,
        src: vec!["/v/VID_20260117_141148_00_635.insv".into()],
        dst,
        part: None,
        lane: match kind {
            Kind::Export => Lane::Export,
            Kind::Still => Lane::Still,
        },
        settings: json!({ "preset": "pano57" }),
        frames: if kind == Kind::Still { 1 } else { 175 },
        bytes: 92_000_000,
        cancelled: false,
        end: None,
        live: None,
    };
    // an export its host left running, its `.part` there; a still that finished
    let x = agent.join("VID_20260117_141148_00_635_pano57.mp4");
    let mut e = job(Kind::Export, x.clone(), t0);
    e.part = Some(media::part::create(&x).unwrap().id);
    r.add("zz.0001-1".into(), e);
    let mut s = job(
        Kind::Still,
        agent.join("VID_20260117_141148_00_635_3.00s_360_7680.jpg"),
        t0 + Duration::from_secs(1),
    );
    s.end = Some(End::done(1, 3.4, 5_000, 0));
    r.add("zz.0001-2".into(), s);
    r.save().unwrap();
    let h = Host::new(Opts {
        sim: Some(sim::Opts {
            dir: Some(d.clone()),
            ..Default::default()
        }),
        tiers: Some(Source::Fixed(Tiers::SIM)),
        ..Default::default()
    })
    .await
    .unwrap();
    let c = connect(&h, false).await;
    let v = ok(&c, "job_status", json!({ "job": "zz.0001-1" })).await;
    assert_eq!(
        (
            &v["kind"],
            &v["state"],
            &v["finished"],
            &v["note"],
            &v["frames_total"]
        ),
        (
            &json!("export"),
            &json!("cancelled"),
            &json!(true),
            &json!(GONE),
            &json!(175)
        ),
        "{v}"
    );
    assert_eq!(v["path"], x.display().to_string());
    assert!(!media::part::of(&x).exists() && !x.exists());
    let v = ok(&c, "job_status", json!({ "job": " zz.0001-2 " })).await;
    assert_eq!(
        (
            &v["kind"],
            &v["state"],
            &v["bytes"],
            &v["concealed"],
            &v["settings"]["preset"]
        ),
        (
            &json!("still"),
            &json!("done"),
            &json!(5_000),
            &json!(false),
            &json!("pano57")
        ),
        "{v}"
    );
    let list = |a: Value| {
        let c = &c;
        async move {
            let v = ok(c, "job_list", a).await;
            let h: Vec<String> = (v["jobs"].as_array().unwrap().iter())
                .map(|j| j["job"].as_str().unwrap().to_string())
                .collect();
            (h, v["total"].as_u64().unwrap(), v)
        }
    };
    let (l, n, v) = list(json!({})).await;
    assert_eq!(
        (l, n),
        (vec!["zz.0001-2".to_string(), "zz.0001-1".into()], 2)
    );
    assert!(v["jobs"][0].get("settings").is_none(), "a short row: {v}");
    assert_eq!(v["jobs"][1]["state"], "cancelled");
    assert_eq!(list(json!({ "kind": "still" })).await.0, ["zz.0001-2"]);
    assert_eq!(list(json!({ "kind": "export" })).await.0, ["zz.0001-1"]);
    assert_eq!(list(json!({ "kind": "download" })).await.1, 0);
    assert_eq!(list(json!({ "limit": 1 })).await.0, ["zz.0001-2"]);
    assert!(
        err(&c, "job_list", json!({ "kind": "photo" }))
            .await
            .contains("arguments")
    );
    let t = Instant::now();
    let w = ok(&c, "job_wait", json!({ "job": "zz.0001-1", "wait_s": 30 })).await;
    assert!(
        w["state"] == "cancelled" && t.elapsed() < Duration::from_secs(2),
        "{w}"
    );
    let x = ok(&c, "job_cancel", json!({ "job": "zz.0001-2" })).await;
    assert_eq!(
        (&x["state"], &x["cancel_requested"]),
        (&json!("done"), &json!(false)),
        "{x}"
    );
    assert_eq!(
        ok(&c, "status", json!({})).await["jobs"],
        json!({ "running": 0, "queued": 0, "finished": 2 })
    );
    assert!(
        err(&c, "job_status", json!({ "job": "zz.0001-3" }))
            .await
            .contains("its host has ended")
    );
    h.shutdown().await;
    // the next host has them as they ended
    let (k, _) = Renders::load(d.join("mcp-renders")).unwrap();
    assert_eq!(
        (k.v.iter())
            .map(|r| r.end.as_ref().unwrap().state.as_str())
            .collect::<Vec<_>>(),
        ["cancelled", "done"]
    );
    std::fs::remove_dir_all(&d).unwrap();
}

/// A host whose worker a camera call holds up (the camera leaves a listing unanswered while a download runs) still
/// hands the queue and the jobs on: after `ENDING` it stops the download and saves them without the worker, and the
/// next host, started meanwhile, takes them well within `HANDOVER`. The old worker, back from its call, writes and
/// starts nothing; the next host resumes the download once connected.
#[tokio::test(flavor = "multi_thread")]
async fn held_up() {
    let d = std::env::temp_dir().join(format!("x5desk-mcp-held-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let o = |rate| Opts {
        sim: Some(sim::Opts {
            rate,
            size: 4_000_000,
            dir: Some(d.clone()),
        }),
        tiers: Some(Source::Fixed(Tiers::SIM)),
        ..Default::default()
    };
    // 4 MB files at 0.4 MB/s
    let a = Host::new(o(100_000)).await.unwrap();
    let c = Arc::new(connect(&a, false).await);
    ok(&c, "camera_connect", json!({})).await;
    let l = ok(&c, "album_list", json!({})).await;
    let vid = key(&l["items"], "VID_20260116_152935_00_603.insv");
    let j = ok(&c, "album_download", json!({ "keys": [vid] })).await;
    assert_eq!(flowing(&c, &j["job"]).await["state"], "running");
    let card = a.sim().unwrap().card.display().to_string();
    ok(&c, "card_open", json!({ "path": card })).await;
    // back to the camera's album: its listing goes unanswered, the worker waits in it (10 s)
    let fake = a.sim().unwrap().fake.clone();
    fake.lock().unwrap().mute = vec![link::code::GET_FILE_LIST];
    let lists = || {
        (fake.lock().unwrap().log.iter())
            .filter(|x| x.1 == link::code::GET_FILE_LIST)
            .count()
    };
    let n = lists();
    tokio::spawn({
        let c = c.clone();
        async move {
            let p = CallToolRequestParams::new("card_close")
                .with_arguments(json!({}).as_object().unwrap().clone());
            let _ = c.s.call_tool(p).await;
        }
    });
    assert!(
        until(10.0, || lists() > n).await,
        "the listing was asked for"
    );
    let asked = Instant::now();
    let end = tokio::spawn({
        let a = a.clone();
        async move { a.shutdown().await }
    });
    let b = Host::new(o(8_000_000)).await.unwrap();
    let took = asked.elapsed();
    end.await.unwrap();
    assert!(
        took >= mcp::host::ENDING && took < mcp::host::HANDOVER,
        "{took:?}"
    );
    let log = std::fs::read_to_string(&a.sim().unwrap().log).unwrap();
    assert!(log.contains("did not end within 5 s"), "{log}");
    let c = connect(&b, false).await;
    let s = ok(&c, "job_status", json!({ "job": j["job"] })).await;
    assert_eq!(s["state"], "queued", "{s}");
    let f = PathBuf::from(s["files"][0]["path"].as_str().unwrap());
    let part = PathBuf::from(format!("{}.part", f.display()));
    let size = || std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let opens = || fake.lock().unwrap().opens.values().sum::<usize>();
    let (had, o1) = (size(), opens());
    assert!(had > 0, "a part to resume");
    // the old worker is back from its call: it wrote and started nothing since
    tokio::time::sleep((asked + Duration::from_secs(12)).saturating_duration_since(Instant::now()))
        .await;
    assert_eq!((size(), opens()), (had, o1), "the old worker kept off");
    ok(&c, "camera_connect", json!({})).await;
    let w = ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 60 })).await;
    assert_eq!(w["state"], "done", "{w}");
    same(b.sim().unwrap(), &w["files"]);
    b.shutdown().await;
    drop(a);
    std::fs::remove_dir_all(&d).unwrap();
}

/// Every tool, ok and failing: neither the marker the host knows from its start nor the camera's access-point
/// password, which it learns only from the camera (both passed in arguments, the password once connected), is in a
/// byte a client receives (on either protocol generation) or in the log.
#[tokio::test(flavor = "multi_thread")]
async fn scrub() {
    let h = host(Tiers::SIM, 8_000_000, 20_000).await;
    let sim = h.sim().unwrap();
    let (c, m) = (connect(&h, false).await, connect(&h, true).await);
    let card = sim.card.display().to_string();
    assert_eq!(
        h.secrets().scrub(sim::AP),
        sim::AP,
        "not known before a connect"
    );
    for c in [&c, &m] {
        for (t, a) in [
            ("status", json!({})),
            ("camera_status", json!({})),
            ("album_list", json!({})),
            ("camera_connect", json!({ "wifi_ip": sim::MARKER })),
            ("camera_connect", json!({})),
            ("camera_status", json!({})),
            ("status", json!({})),
            ("album_list", json!({ "kind": sim::AP })),
            ("card_open", json!({ "path": sim::AP })),
            ("file_info", json!({ "path": sim::AP })),
            (
                "album_download",
                json!({ "keys": [sim::AP], "subdir": sim::AP }),
            ),
            ("job_status", json!({ "job": sim::AP })),
            ("album_list", json!({ "kind": "photo" })),
            ("album_list", json!({ "kind": sim::MARKER })),
            (
                "album_thumbnail",
                json!({ "key": "/DCIM/Camera01/IMG_20260116_174103_620_" }),
            ),
            ("album_thumbnail", json!({ "key": sim::MARKER })),
            ("card_list", json!({})),
            ("card_open", json!({ "path": sim::MARKER })),
            ("card_open", json!({ "path": card })),
            ("card_close", json!({})),
            ("file_info", json!({ "path": sim::MARKER })),
            (
                "file_info",
                json!({ "path": sim.agent.join(format!("{}.insv", sim::MARKER)) }),
            ),
            (
                "album_download",
                json!({ "keys": ["/DCIM/Camera01/IMG_20260116_174103_620_"] }),
            ),
            (
                "album_download",
                json!({ "keys": [sim::MARKER], "subdir": sim::MARKER }),
            ),
            ("job_list", json!({})),
            ("job_status", json!({ "job": sim::MARKER })),
            (
                "job_wait",
                json!({ "job": format!("{}-1", h.id()), "wait_s": 10 }),
            ),
            ("job_cancel", json!({ "job": format!("{}-1", h.id()) })),
            ("job_cancel", json!({ "job": sim::MARKER })),
            ("camera_release", json!({})),
        ] {
            let _ = call(c, t, a).await;
        }
        let p = CallToolRequestParams::new(sim::MARKER);
        assert!(c.s.call_tool(p).await.is_err());
    }
    // connected, the host knows the password from the camera, and says what it may
    ok(&c, "camera_connect", json!({})).await;
    assert_eq!(
        h.with(|s| s.info.wifi.clone()).unwrap().1,
        sim::AP,
        "the host learnt it"
    );
    assert_eq!(
        h.secrets().scrub(sim::AP),
        "[redacted]",
        "a secret from then on"
    );
    for t in ["status", "camera_status", "album_list", "card_list"] {
        let v = ok(&c, t, json!({})).await.to_string();
        assert!(!v.contains(sim::AP) && !v.contains(sim::MARKER), "{t}: {v}");
    }
    let e = err(&c, "card_open", json!({ "path": sim::AP })).await;
    assert!(e.contains("[redacted]") && !e.contains(sim::AP), "{e}");
    for c in [&c, &m] {
        let got = c.got.lock().unwrap().clone();
        assert!(got.len() > 10_000, "{} bytes", got.len());
        for s in [sim::MARKER, sim::AP] {
            assert!(!got.windows(s.len()).any(|w| w == s.as_bytes()), "{s}");
        }
        assert!(
            String::from_utf8_lossy(&got).contains("[redacted]"),
            "the arguments' secrets are masked"
        );
    }
    let log = std::fs::read_to_string(&sim.log).unwrap();
    assert!(
        log.contains("[redacted]") && !log.contains(sim::MARKER) && !log.contains(sim::AP),
        "{log}"
    );
    h.shutdown().await;
}

/// The camera goes after the idle time without a call, and a shut-down host ends its connections.
#[tokio::test(flavor = "multi_thread")]
async fn idle() {
    let h = Host::new(Opts {
        sim: Some(Default::default()),
        tiers: Some(Source::Fixed(Tiers::SIM)),
        idle: Duration::from_millis(1500),
        ..Default::default()
    })
    .await
    .unwrap();
    let c = connect(&h, false).await;
    assert_eq!(
        ok(&c, "camera_connect", json!({})).await["connection"],
        "on"
    );
    let on = || h.with(|s| s.info.conn == cam::camera::Conn::On);
    assert!(on());
    tokio::time::sleep(Duration::from_millis(700)).await;
    assert!(on(), "not idle yet");
    assert!(until(5.0, || !on()).await, "released when idle");
    let log = std::fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(
        log.contains("released the camera after 1 s without a call"),
        "{log}"
    );
    h.shutdown().await;
    assert!(
        c.s.list_all_tools().await.is_err(),
        "the host's connections end with it"
    );
    assert!(
        h.serve(tokio::io::empty(), tokio::io::sink(), "late", None)
            .await
            .is_err()
    );
}
