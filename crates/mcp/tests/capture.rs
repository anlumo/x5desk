//! The capture tools and the camera's file writes over an in-memory pipe, on cam's simulated X5 with capture
//! (`/CONF`, modes, settings, photos and clips, their notifications): what each checks before anything goes to the
//! camera, what goes out and what comes back; the tested-on-an-X5 limit and the `untested` tier; deletes and what they
//! report; a recording this host started stopped at its end, also after the link was lost, and keeping the camera
//! from the idle release; the camera's clock and card never touched.

use cam::sim::Fake;
use link::{code, pb};
use mcp::{Host, Opts, Tiers, config::Source, sim};
use prost::Message;
use rmcp::{RoleClient, ServiceExt, model::CallToolRequestParams, service::RunningService};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

type Cli = RunningService<RoleClient, ()>;

async fn host_with(tiers: &str, idle: Duration, rate: u64) -> Arc<Host> {
    Host::new(Opts {
        sim: Some(sim::Opts {
            rate,
            size: 200_000,
            dir: None,
        }),
        tiers: Some(Source::Fixed(Tiers::parse(tiers).unwrap())),
        idle,
        ..Default::default()
    })
    .await
    .unwrap()
}

async fn host(tiers: &str) -> Arc<Host> {
    host_with(tiers, Duration::from_secs(300), 50_000_000).await
}

async fn connect(h: &Arc<Host>) -> Cli {
    let (a, b) = tokio::io::duplex(1 << 20);
    let (r, w) = tokio::io::split(b);
    let hh = h.clone();
    tokio::spawn(async move { hh.serve(r, w, "test", None).await });
    ().serve(a).await.unwrap()
}

/// A client with the camera connected and its capture state read.
async fn on(h: &Arc<Host>) -> Cli {
    let c = connect(h).await;
    assert_eq!(
        ok(&c, "camera_connect", json!({})).await["connection"],
        "on"
    );
    ok(&c, "capture_status", json!({})).await;
    c
}

/// A call's JSON, or its error text.
async fn call(c: &Cli, name: &str, a: Value) -> Result<Value, String> {
    let p =
        CallToolRequestParams::new(name.to_string()).with_arguments(a.as_object().unwrap().clone());
    let r = c
        .call_tool(p)
        .await
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let t: Vec<String> = (r.content.iter())
        .filter_map(|c| c.as_text())
        .map(|t| t.text.clone())
        .collect();
    match r.is_error {
        Some(true) => Err(t.join("\n")),
        _ => Ok(r.structured_content.unwrap_or_default()),
    }
}

async fn ok(c: &Cli, name: &str, a: Value) -> Value {
    call(c, name, a)
        .await
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

async fn err(c: &Cli, name: &str, a: Value) -> String {
    match call(c, name, a).await {
        Ok(v) => panic!("{name} should fail: {v}"),
        Err(e) => e,
    }
}

fn fake(h: &Host) -> &Fake {
    &h.sim().unwrap().fake
}

fn count(h: &Host, c: u16) -> usize {
    cam::sim::count(fake(h), c)
}

async fn until(secs: f64, mut f: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs_f64(secs);
    while Instant::now() < end {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await
    }
    f()
}

/// A setting of capture_status's answer.
fn setting(st: &Value, k: &str) -> Value {
    (st["settings"].as_array().unwrap().iter())
        .find(|x| x["key"] == k)
        .cloned()
        .unwrap_or_else(|| panic!("no setting {k}: {st}"))
}

/// The key of the album item with file `f`.
async fn key(c: &Cli, f: &str) -> String {
    let l = ok(c, "album_list", json!({ "limit": 500 })).await;
    (l["items"].as_array().unwrap().iter())
        .find(|i| i["files"].as_array().unwrap().iter().any(|x| x == f))
        .unwrap_or_else(|| panic!("no item with {f}: {l}"))["key"]
        .as_str()
        .unwrap()
        .to_string()
}

async fn keys(c: &Cli) -> Vec<String> {
    let l = ok(c, "album_list", json!({ "limit": 500 })).await;
    (l["items"].as_array().unwrap().iter())
        .map(|i| i["key"].as_str().unwrap().to_string())
        .collect()
}

/// Nothing ever set the camera's clock or erased its card.
fn clockless(h: &Host) {
    use pb::Setting::*;
    for (_, c, b) in fake(h).lock().unwrap().log.iter() {
        assert!(![code::ERASE_SD, code::FORMAT_STORAGE].contains(c), "{c}");
        if *c == code::SET_OPTIONS {
            let q = pb::WriteSettings::decode(&b[..]).unwrap();
            for t in [Clock, UtcOffset, Zone] {
                assert!(!q.keys.contains(&(t as i32)), "{t:?}: {q:?}");
            }
        }
    }
}

/// The capture state, the modes and settings with what x5desk ran on an X5; a tested setting written and read back,
/// untested, unknown and disallowed ones refused before anything goes out; a photo answered with its key and
/// downloadable; modes by name or label, live and untested modes refused; a setting fixed by another, a read-only
/// one; a recording that keeps the camera's mode and settings and the camera itself, a highlight, its stop with the
/// clip; a timed one; a camera in a mode the protocol does not know counts as untested; no clock, no format.
#[tokio::test(flavor = "multi_thread")]
async fn capture() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    let st = ok(&c, "capture_status", json!({})).await;
    println!("{st:#}");
    assert_eq!(st["mode"]["name"], "PHOTO_SINGLE");
    assert_eq!(st["mode"]["tested"], true);
    assert_eq!(
        (&st["lens"], &st["capturing"]),
        (&json!("360"), &json!(false))
    );
    assert_eq!(st["untested_allowed"], false);
    let mode = |n: &str| {
        (st["modes"].as_array().unwrap().iter())
            .find(|x| x["name"] == n)
            .cloned()
            .unwrap()
    };
    assert_eq!(mode("VIDEO_NORMAL")["tested"], true);
    assert_eq!(mode("VIDEO_PURE")["tested"], false);
    assert_eq!(mode("VIDEO_LIVING_MODE_PANO")["kind"], "live");
    let wb = setting(&st, "white_balance");
    assert_eq!((&wb["value"], &wb["shown"]), (&json!("0"), &json!("Auto")));
    assert_eq!(
        (&wb["tested"], &wb["settable"]),
        (&json!(true), &json!(true))
    );
    assert_eq!(setting(&st, "exposure_bias")["tested"], false);

    // a tested setting, its value given as a number
    let n = count(&h, code::SET_PHOTO_OPTIONS);
    let r = ok(
        &c,
        "capture_set",
        json!({ "key": "white_balance", "value": 4000 }),
    )
    .await;
    assert_eq!(
        (&r["value"], &r["shown"]),
        (&json!("4000"), &json!("4000 K"))
    );
    assert_eq!(count(&h, code::SET_PHOTO_OPTIONS), n + 1);
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(setting(&st, "white_balance")["value"], "4000");
    let r = ok(
        &c,
        "capture_set",
        json!({ "key": "white_balance", "value": "4000" }),
    )
    .await;
    assert_eq!(r["note"], "set already");
    for (k, v, w) in [
        (
            "exposure_bias",
            json!("0.3"),
            "`x5desk mcp enable --tiers read,local,camera,untested`",
        ),
        ("white_balance", json!(1234), "not allowed"),
        ("record_resolution", json!("5760_2880_30"), "has no setting"),
    ] {
        let e = err(&c, "capture_set", json!({ "key": k, "value": v })).await;
        assert!(e.contains(w), "{k}: {e}");
    }
    assert_eq!(
        count(&h, code::SET_PHOTO_OPTIONS),
        n + 1,
        "nothing more went out"
    );

    // a photo: its file, key, in the album, downloadable
    let p = ok(&c, "capture_photo", json!({ "self_timer": false })).await;
    println!("{p}");
    let k = p["key"].as_str().unwrap().to_string();
    assert!(p["file"].as_str().unwrap().ends_with(".insp"), "{p}");
    assert_eq!(p["in_album"], true);
    let j = ok(&c, "album_download", json!({ "keys": [k] })).await;
    let w = ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 20 })).await;
    assert_eq!(w["state"], "done", "{w}");

    // modes: untested, live and unknown ones refused, a label switches
    let e = err(&c, "capture_mode", json!({ "mode": "VIDEO_PURE" })).await;
    assert!(e.contains("untested"), "{e}");
    let e = err(
        &c,
        "capture_mode",
        json!({ "mode": "VIDEO_LIVING_MODE_PANO" }),
    )
    .await;
    assert!(e.contains("live mode"), "{e}");
    let e = err(&c, "capture_mode", json!({ "mode": "nosuch" })).await;
    assert!(e.contains("VIDEO_NORMAL (Video)"), "{e}");
    let n = count(&h, code::SET_OPTIONS);
    assert_eq!(
        ok(&c, "capture_mode", json!({ "mode": "video" })).await["mode"],
        "VIDEO_NORMAL"
    );
    assert_eq!(count(&h, code::SET_OPTIONS), n + 1);
    assert!(
        err(&c, "capture_photo", json!({}))
            .await
            .contains("photo mode")
    );

    // fixed by another setting; read only; tested ones in VIDEO_NORMAL
    let r = ok(
        &c,
        "capture_set",
        json!({ "key": "exposure_iso", "value": 0 }),
    )
    .await;
    assert_eq!(r["note"], "in effect already");
    let e = err(
        &c,
        "capture_set",
        json!({ "key": "exposure_iso", "value": 400 }),
    )
    .await;
    assert!(
        e.contains("fixed at 0") && e.contains("exposure_program"),
        "{e}"
    );
    let e = err(
        &c,
        "capture_set",
        json!({ "key": "record_duration", "value": 600 }),
    )
    .await;
    assert!(e.contains("read only"), "{e}");
    let r = ok(
        &c,
        "capture_set",
        json!({ "key": "record_resolution", "value": "5760_2880_30" }),
    )
    .await;
    assert_eq!(r["shown"], "5760 × 2880, 30 fps");
    ok(
        &c,
        "capture_set",
        json!({ "key": "exposure_program", "value": "MANUAL" }),
    )
    .await;
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(
        setting(&st, "exposure_iso")["allowed"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(
        ok(
            &c,
            "capture_set",
            json!({ "key": "exposure_iso", "value": "400" })
        )
        .await["value"],
        "400"
    );

    // a recording: the camera's mode, settings and session kept meanwhile
    let r = ok(&c, "capture_record", json!({ "action": "start" })).await;
    assert_eq!(r["capturing"], true, "{r}");
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(
        (&st["capturing"], &st["started_by_this_host"]),
        (&json!(true), &json!(true))
    );
    assert_eq!(st["capture_state"], "RECORDING");
    assert_eq!(
        ok(&c, "status", json!({})).await["camera"]["recording"],
        true
    );
    let (sets, opts) = (
        count(&h, code::SET_PHOTO_OPTIONS),
        count(&h, code::SET_OPTIONS),
    );
    for (t, a, w) in [
        (
            "capture_mode",
            json!({ "mode": "PHOTO_SINGLE" }),
            "captures",
        ),
        (
            "capture_set",
            json!({ "key": "white_balance", "value": "4000" }),
            "captures",
        ),
        ("camera_release", json!({}), "capture_record stop"),
        (
            "capture_record",
            json!({ "action": "start" }),
            "this host started",
        ),
    ] {
        let e = err(&c, t, a).await;
        assert!(e.contains(w), "{t}: {e}");
    }
    assert_eq!(
        (
            count(&h, code::SET_PHOTO_OPTIONS),
            count(&h, code::SET_OPTIONS)
        ),
        (sets, opts)
    );
    let m = ok(&c, "capture_mark", json!({})).await;
    assert!(m["marked_at_s"].is_number(), "{m}");
    assert_eq!(count(&h, code::KEY_TIME_POINT), 1);
    let s = ok(&c, "capture_record", json!({ "action": "stop" })).await;
    println!("{s}");
    assert!(s["file"].as_str().unwrap().ends_with(".insv"), "{s}");
    assert_eq!(
        (&s["in_album"], &s["capturing"]),
        (&json!(true), &json!(false))
    );
    assert_eq!(
        ok(&c, "capture_status", json!({})).await["started_by_this_host"],
        false
    );
    let r = ok(&c, "capture_record", json!({ "action": "stop" })).await;
    assert_eq!(r["note"], "the camera did not capture");
    assert!(
        err(&c, "capture_mark", json!({}))
            .await
            .contains("does not record")
    );

    // a timed one: the call waits for its clip; a long one answers at once
    let t0 = Instant::now();
    let r = ok(
        &c,
        "capture_record",
        json!({ "action": "start", "seconds": 2 }),
    )
    .await;
    println!("{r} after {:?}", t0.elapsed());
    assert!(t0.elapsed() >= Duration::from_secs(2) && t0.elapsed() < Duration::from_secs(15));
    assert!(r["file"].as_str().unwrap().ends_with(".insv"), "{r}");
    let r = ok(
        &c,
        "capture_record",
        json!({ "action": "start", "seconds": 120 }),
    )
    .await;
    assert_eq!(
        (&r["capturing"], &r["stops_in_s"]),
        (&json!(true), &json!(120.0))
    );
    assert!(ok(&c, "capture_record", json!({ "action": "stop" })).await["file"].is_string());
    let w = ok(
        &c,
        "capture_record",
        json!({ "action": "wait", "wait_s": 1 }),
    )
    .await;
    assert_eq!(w["capturing"], false);
    assert!(
        err(
            &c,
            "capture_record",
            json!({ "action": "start", "seconds": 0.5 })
        )
        .await
        .contains("1 to 600")
    );

    // a sub-mode the protocol does not know reads as VIDEO_NORMAL: untested all the same
    let note = |h: &Host| {
        let l = fake(h).lock().unwrap().link.clone().unwrap();
        l.note(8275, &pb::ModeNote::default().encode_to_vec())
    };
    fake(&h).lock().unwrap().capture.as_mut().unwrap().video = 99;
    note(&h);
    let gone = || h.with(|s| s.cap.vals.is_some() && !cam::conf::known(&s.cap.o));
    assert!(until(5.0, gone).await);
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(st["mode"]["name"], "VIDEO_NORMAL");
    assert_eq!(
        (&st["mode"]["tested"], &st["mode"]["known_to_the_protocol"]),
        (&json!(false), &json!(false))
    );
    let n = count(&h, code::START_CAPTURE);
    let e = err(&c, "capture_record", json!({ "action": "start" })).await;
    assert!(e.contains("untested"), "{e}");
    assert_eq!(count(&h, code::START_CAPTURE), n);
    fake(&h).lock().unwrap().capture.as_mut().unwrap().video = 0;
    note(&h);

    // not this client's: favourites (untested), deletes (the delete tier)
    let e = err(&c, "camera_favourite", json!({ "keys": [k], "on": true })).await;
    assert!(e.contains("untested"), "{e}");
    let e = err(&c, "camera_delete", json!({ "keys": [k] })).await;
    assert!(e.contains("the delete tier"), "{e}");
    clockless(&h);
    h.shutdown().await;
}

/// With the `untested` tier: an untested mode switched to, its setting written, a recording in it; a timelapse whose
/// stop names no file; favourites set and cleared.
#[tokio::test(flavor = "multi_thread")]
async fn untested() {
    let h = host("read,local,camera,untested").await;
    let c = on(&h).await;
    assert_eq!(
        ok(&c, "capture_status", json!({})).await["untested_allowed"],
        true
    );
    assert_eq!(
        ok(&c, "capture_mode", json!({ "mode": "PureVideo" })).await["mode"],
        "VIDEO_PURE"
    );
    ok(
        &c,
        "capture_set",
        json!({ "key": "white_balance", "value": "4000" }),
    )
    .await;
    ok(&c, "capture_record", json!({ "action": "start" })).await;
    assert!(ok(&c, "capture_record", json!({ "action": "stop" })).await["file"].is_string());

    ok(&c, "capture_mode", json!({ "mode": "VIDEO_TIMELAPSE" })).await;
    fake(&h).lock().unwrap().capture.as_mut().unwrap().nameless = true;
    let n = count(&h, code::START_TIMELAPSE);
    ok(&c, "capture_record", json!({ "action": "start" })).await;
    assert_eq!(count(&h, code::START_TIMELAPSE), n + 1);
    let s = ok(&c, "capture_record", json!({ "action": "stop" })).await;
    assert!(
        s["uri"].is_null() && s["note"].as_str().unwrap().contains("named no file"),
        "{s}"
    );
    fake(&h).lock().unwrap().capture.as_mut().unwrap().nameless = false;

    ok(&c, "capture_mode", json!({ "mode": "PHOTO_SINGLE" })).await;
    let k = ok(&c, "capture_photo", json!({})).await["key"]
        .as_str()
        .unwrap()
        .to_string();
    for on in [true, false] {
        let r = ok(&c, "camera_favourite", json!({ "keys": [k], "on": on })).await;
        assert_eq!(r["items"][0]["favourite"], on, "{r}");
    }
    clockless(&h);
    h.shutdown().await;
}

/// The self-timer: counted down by the host (as the camera's app does), skipped with `self_timer: false`.
#[tokio::test(flavor = "multi_thread")]
async fn self_timer() {
    let h = host("read,local,camera,untested").await;
    let c = on(&h).await;
    ok(
        &c,
        "capture_set",
        json!({ "key": "photography_self_timer", "value": 2 }),
    )
    .await;
    let t0 = Instant::now();
    let p = ok(&c, "capture_photo", json!({})).await;
    assert_eq!(p["self_timer_s"], 2);
    assert!(t0.elapsed() >= Duration::from_secs(2), "{:?}", t0.elapsed());
    let t0 = Instant::now();
    let p = ok(&c, "capture_photo", json!({ "self_timer": false })).await;
    assert_eq!(p["self_timer_s"], 0);
    assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
    h.shutdown().await;
}

/// A command whose camera call hangs: its tool fails on its own outcome, and the next tool (waiting its turn) gets
/// its own, not the earlier one's.
#[tokio::test(flavor = "multi_thread")]
async fn own_outcome() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    fake(&h).lock().unwrap().mute = vec![code::SET_PHOTO_OPTIONS];
    let (s, p) = tokio::join!(
        call(
            &c,
            "capture_set",
            json!({ "key": "white_balance", "value": "4000" })
        ),
        async {
            tokio::time::sleep(Duration::from_millis(300)).await;
            call(&c, "capture_photo", json!({})).await
        }
    );
    println!("{s:?}\n{p:?}");
    let s = s.unwrap_err();
    assert!(s.contains("the camera:"), "{s}");
    assert!(p.unwrap()["file"].as_str().unwrap().ends_with(".insp"));
    fake(&h).lock().unwrap().mute.clear();
    h.shutdown().await;
}

/// Deletes: what is gone per item and how it was kept here (a finished download of it, files by name, none);
/// downloaded_only keeping those not downloaded; a key deleted already; unknown keys deleting nothing; a delete the
/// camera fails half way named partly done; refused while the camera records, an item downloads, or the album shows
/// a card; a client without the camera tier told so.
#[tokio::test(flavor = "multi_thread")]
async fn deletes() {
    let h = host("read,local,camera,delete").await;
    let c = on(&h).await;
    let a = key(&c, "IMG_20260116_174103_00_620.jpg").await;
    let b = key(&c, "IMG_20260116_175112_00_621.jpg").await;
    let j = ok(&c, "album_download", json!({ "keys": [a] })).await;
    assert_eq!(
        ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 20 })).await["state"],
        "done"
    );
    let r = ok(
        &c,
        "camera_delete",
        json!({ "keys": [a, b], "downloaded_only": true }),
    )
    .await;
    println!("{r:#}");
    assert_eq!(r["deleted"], json!([{ "key": a, "local": "verified" }]));
    assert_eq!(
        r["kept_not_downloaded"],
        json!([{ "key": b, "local": "none" }])
    );
    assert!(!keys(&c).await.contains(&a));
    let r = ok(&c, "camera_delete", json!({ "keys": [a] })).await;
    assert_eq!(
        (&r["deleted"], &r["already_deleted"]),
        (&json!([]), &json!([a]))
    );
    let e = err(
        &c,
        "camera_delete",
        json!({ "keys": [b, "/DCIM/Camera01/nosuch_"] }),
    )
    .await;
    assert!(e.contains("nothing deleted"), "{e}");
    assert!(keys(&c).await.contains(&b));
    // its files there by name: counted so
    let ag = h.agent().to_path_buf();
    std::fs::create_dir_all(&ag).unwrap();
    for f in [
        "IMG_20260116_175112_00_621.jpg",
        "IMG_20260116_175112_00_621.dng",
    ] {
        std::fs::write(ag.join(f), b"x").unwrap();
    }
    let r = ok(&c, "camera_delete", json!({ "keys": [b] })).await;
    assert_eq!(r["deleted"], json!([{ "key": b, "local": "by_name" }]));
    assert_eq!(r["never_downloaded"], json!([]));

    // the camera fails on the proxy: the clip gone, its proxy left
    let v = key(&c, "VID_20260116_152935_00_603.insv").await;
    let lrv = "/DCIM/Camera01/LRV_20260116_152935_01_603.lrv";
    fake(&h).lock().unwrap().stuck.insert(lrv.into(), 3);
    let e = err(&c, "camera_delete", json!({ "keys": [v] })).await;
    println!("{e}");
    assert!(
        e.contains("partly_deleted") && e.contains("LRV_20260116_152935_01_603.lrv"),
        "{e}"
    );
    fake(&h).lock().unwrap().stuck.clear();
    let r = ok(&c, "camera_delete", json!({ "keys": [v] })).await;
    assert_eq!(r["never_downloaded"], json!([v]));

    // not while recording
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    ok(&c, "capture_record", json!({ "action": "start" })).await;
    let w = key(&c, "VID_20260117_141148_00_635.insv").await;
    let n = count(&h, code::DELETE_FILES);
    assert!(
        err(&c, "camera_delete", json!({ "keys": [w] }))
            .await
            .contains("captures")
    );
    ok(&c, "capture_record", json!({ "action": "stop" })).await;
    // nor on a card's album
    let card = h.sim().unwrap().card.display().to_string();
    ok(&c, "card_open", json!({ "path": card })).await;
    assert!(
        err(&c, "camera_delete", json!({ "keys": [w] }))
            .await
            .contains("card_close")
    );
    ok(&c, "card_close", json!({})).await;
    assert_eq!(count(&h, code::DELETE_FILES), n);
    clockless(&h);
    h.shutdown().await;

    // an item downloading
    let h = host_with("read,local,camera,delete", Duration::from_secs(300), 20_000).await;
    let c = on(&h).await;
    let w = key(&c, "VID_20260117_141148_00_635.insv").await;
    let j = ok(&c, "album_download", json!({ "keys": [w] })).await;
    let e = err(&c, "camera_delete", json!({ "keys": [w] })).await;
    assert!(
        e.contains("being downloaded") && e.contains(j["job"].as_str().unwrap()),
        "{e}"
    );
    ok(&c, "job_cancel", json!({ "job": j["job"] })).await;
    h.shutdown().await;

    // the delete tier without the camera tier
    let h = host("read,delete").await;
    let c = connect(&h).await;
    let e = err(&c, "camera_delete", json!({ "keys": [w] })).await;
    assert!(e.contains("camera tier"), "{e}");
    h.shutdown().await;
}

/// A recording this host started is stopped at its end, also when the link was lost while it recorded and came back;
/// meanwhile it keeps the camera from the idle release.
#[tokio::test(flavor = "multi_thread")]
async fn end_stops_recording() {
    let h = host_with("read,local,camera", Duration::from_millis(1500), 50_000_000).await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    ok(&c, "capture_record", json!({ "action": "start" })).await;
    let up = || h.with(|s| s.info.conn == cam::camera::Conn::On);
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(up(), "not released while it records");
    // the link lost and back: still this host's
    let l = fake(&h).lock().unwrap().link.clone().unwrap();
    l.unplug();
    assert!(until(5.0, || !up()).await);
    ok(&c, "camera_connect", json!({})).await;
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(
        (&st["capturing"], &st["started_by_this_host"]),
        (&json!(true), &json!(true)),
        "{st}"
    );
    let n = count(&h, code::STOP_CAPTURE);
    let t0 = Instant::now();
    h.shutdown().await;
    println!("ended in {:?}", t0.elapsed());
    assert_eq!(count(&h, code::STOP_CAPTURE), n + 1);
    assert_eq!(fake(&h).lock().unwrap().cap, 0);
    let log = std::fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(
        log.contains("stopped the recording this host started"),
        "{log}"
    );
    clockless(&h);

    // the user's own recording is not this host's to stop
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    fake(&h).lock().unwrap().cap = pb::CaptureState::Recording as i32;
    let l = fake(&h).lock().unwrap().link.clone().unwrap();
    let s = pb::CaptureNote {
        state: pb::CaptureState::Recording as i32,
        ..Default::default()
    };
    l.note(8208, &s.encode_to_vec());
    assert!(until(5.0, || h.with(mcp::host::capturing)).await);
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(
        (&st["capturing"], &st["started_by_this_host"]),
        (&json!(true), &json!(false))
    );
    let n = count(&h, code::STOP_CAPTURE);
    h.shutdown().await;
    assert_eq!(count(&h, code::STOP_CAPTURE), n);
}

/// A client whose connection is narrowed by its own `--tiers`.
async fn connect_as(h: &Arc<Host>, narrow: &str) -> Cli {
    let (a, b) = tokio::io::duplex(1 << 20);
    let (r, w) = tokio::io::split(b);
    let (hh, n) = (h.clone(), Tiers::parse(narrow).ok());
    tokio::spawn(async move { hh.serve(r, w, "test", n).await });
    ().serve(a).await.unwrap()
}

/// A start the camera refuses, or one sent with the capture state unreadable, began nothing: not this host's, the
/// camera released as usual; the untested tier enabled but left out by a client's own --tiers says so, without an
/// `enable` that would replace the user's tiers.
#[tokio::test(flavor = "multi_thread")]
async fn refused_start() {
    let h = host("read,local,camera,untested").await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    for (code, w) in [
        (code::START_CAPTURE, "refused the start"),
        (code::CAPTURE_STATUS, "nothing sent"),
    ] {
        fake(&h).lock().unwrap().refuse = vec![code];
        let e = err(&c, "capture_record", json!({ "action": "start" })).await;
        assert!(e.contains(w), "{e}");
        fake(&h).lock().unwrap().refuse.clear();
        let st = ok(&c, "capture_status", json!({})).await;
        assert_eq!(st["started_by_this_host"], false, "{st}");
    }
    let n = connect_as(&h, "read,local,camera").await;
    let e = err(&n, "capture_mode", json!({ "mode": "VIDEO_PURE" })).await;
    assert!(
        e.contains("--tiers leaves out") && !e.contains("enable"),
        "{e}"
    );
    assert_eq!(
        ok(&c, "camera_release", json!({})).await["connection"],
        "off"
    );
    h.shutdown().await;
}

/// A start the camera answers at once but begins capturing only 7 s later: a second start and the release are
/// refused meanwhile, and the host's end, coming before the camera captures, waits for it and stops it.
#[tokio::test(flavor = "multi_thread")]
async fn slow_start() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    fake(&h).lock().unwrap().capture.as_mut().unwrap().lag = Some(Duration::from_secs(7));
    let r = ok(&c, "capture_record", json!({ "action": "start" })).await;
    assert_eq!(r["capturing"], false, "{r}");
    let e = err(&c, "capture_record", json!({ "action": "start" })).await;
    assert!(e.contains("this host started"), "{e}");
    let e = err(&c, "camera_release", json!({})).await;
    assert!(e.contains("this host started"), "{e}");
    let n = count(&h, code::STOP_CAPTURE);
    h.shutdown().await;
    let log = std::fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert_eq!(count(&h, code::STOP_CAPTURE), n + 1, "{log}");
    assert!(
        log.contains("stopped the recording this host started"),
        "{log}"
    );
}

/// A timed recording whose link is lost before its stop is due: the host's timer tries again and stops it once the
/// camera is connected again; the call waiting for the clip gets it.
#[tokio::test(flavor = "multi_thread")]
async fn timed_retry() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    let n = count(&h, code::STOP_CAPTURE);
    let h2 = h.clone();
    let other = async move {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let l = fake(&h2).lock().unwrap().link.clone().unwrap();
        l.unplug();
        tokio::time::sleep(Duration::from_secs(3)).await;
        let c2 = connect(&h2).await;
        ok(&c2, "camera_connect", json!({})).await;
    };
    let (r, ()) = tokio::join!(
        ok(
            &c,
            "capture_record",
            json!({ "action": "start", "seconds": 3 })
        ),
        other
    );
    println!("{r}");
    assert!(r["file"].as_str().unwrap().ends_with(".insv"), "{r}");
    assert_eq!(count(&h, code::STOP_CAPTURE), n + 1);
    let log = std::fs::read_to_string(&h.sim().unwrap().log).unwrap();
    assert!(log.contains("again every 5 s"), "{log}");
    h.shutdown().await;
}

/// A recording this host started, the link lost, and the new session's first capture state not readable: still this
/// host's (a default state is no word from the camera), and the host's end stops it once the camera is read.
#[tokio::test(flavor = "multi_thread")]
async fn reconnect_unread() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    ok(&c, "capture_record", json!({ "action": "start" })).await;
    let l = fake(&h).lock().unwrap().link.clone().unwrap();
    l.unplug();
    let up = || h.with(|s| s.info.conn == cam::camera::Conn::On);
    assert!(until(5.0, || !up()).await);
    fake(&h).lock().unwrap().refuse = vec![code::CAPTURE_STATUS];
    ok(&c, "camera_connect", json!({})).await;
    let read = || h.with(|s| s.csid == s.tsid && s.cap.on && s.cap.mode.is_some());
    assert!(until(10.0, read).await);
    assert!(h.with(|s| s.rec.is_some()), "kept");
    fake(&h).lock().unwrap().refuse.clear();
    let l = fake(&h).lock().unwrap().link.clone().unwrap();
    l.note(8275, &pb::ModeNote::default().encode_to_vec());
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(
        (&st["capturing"], &st["started_by_this_host"]),
        (&json!(true), &json!(true)),
        "{st}"
    );
    let n = count(&h, code::STOP_CAPTURE);
    h.shutdown().await;
    assert_eq!(count(&h, code::STOP_CAPTURE), n + 1);
}

/// downloaded_only in a later host: a download whose file still has the size the camera sent is verified (the jobs
/// keep it), one whose file changed meanwhile is not, and stays on the camera.
#[tokio::test(flavor = "multi_thread")]
async fn verified_later() {
    let d = std::env::temp_dir().join(format!("x5desk-mcp-verified-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    let mk = async || {
        Host::new(Opts {
            sim: Some(sim::Opts {
                rate: 50_000_000,
                size: 200_000,
                dir: Some(d.clone()),
            }),
            tiers: Some(Source::Fixed(
                Tiers::parse("read,local,camera,delete").unwrap(),
            )),
            ..Default::default()
        })
        .await
        .unwrap()
    };
    let h = mk().await;
    let c = on(&h).await;
    let a = key(&c, "IMG_20260116_174103_00_620.jpg").await;
    let b = key(&c, "VID_20260117_141148_00_635.insv").await;
    let j = ok(&c, "album_download", json!({ "keys": [a, b] })).await;
    let w = ok(&c, "job_wait", json!({ "job": j["job"], "wait_s": 30 })).await;
    assert_eq!(w["state"], "done", "{w}");
    let lrv = (w["files"].as_array().unwrap().iter())
        .map(|f| f["path"].as_str().unwrap().to_string())
        .find(|p| p.ends_with(".lrv"))
        .unwrap();
    h.shutdown().await;
    // the clip's proxy changed after its download
    std::fs::write(&lrv, b"changed").unwrap();
    let h = mk().await;
    let c = on(&h).await;
    let r = ok(
        &c,
        "camera_delete",
        json!({ "keys": [a, b], "downloaded_only": true }),
    )
    .await;
    println!("{r:#}");
    assert_eq!(r["deleted"], json!([{ "key": a, "local": "verified" }]));
    assert_eq!(
        r["kept_not_downloaded"],
        json!([{ "key": b, "local": "by_name" }])
    );
    h.shutdown().await;
    let _ = std::fs::remove_dir_all(&d);
}

/// A call narrowed by a proxying front's --tiers (`_meta`): tools outside them refused as this client's own
/// --tiers would, and the untested tier's modes too, though the host has it.
#[tokio::test(flavor = "multi_thread")]
async fn proxied_narrowing() {
    let h = host("read,local,camera,untested").await;
    let c = on(&h).await;
    let narrowed = async |name: &str, a: Value, t: &str| {
        let mut p = CallToolRequestParams::new(name.to_string())
            .with_arguments(a.as_object().unwrap().clone());
        let mut m = rmcp::model::RequestMetaObject::default();
        m.insert(mcp::front::NARROW.into(), json!(t));
        p.meta = Some(m);
        let r = c.call_tool(p).await.unwrap();
        let t: Vec<String> = (r.content.iter())
            .filter_map(|c| c.as_text())
            .map(|t| t.text.clone())
            .collect();
        (r.is_error == Some(true), t.join("\n"))
    };
    let (e, t) = narrowed(
        "capture_mode",
        json!({ "mode": "VIDEO_PURE" }),
        "read,local,camera",
    )
    .await;
    assert!(e && t.contains("--tiers leaves out"), "{t}");
    let (e, t) = narrowed("camera_release", json!({}), "read").await;
    assert!(e && t.contains("--tiers leaves out"), "{t}");
    let (e, t) = narrowed(
        "capture_mode",
        json!({ "mode": "VIDEO_PURE" }),
        "read,camera,untested",
    )
    .await;
    assert!(!e, "{t}");
    h.shutdown().await;
}

/// A timed recording whose camera begins capturing only 7 s after the start: the timer's stop finds nothing at first,
/// keeps the recording this host's, and stops it once the camera captures; a mode change waits meanwhile.
#[tokio::test(flavor = "multi_thread")]
async fn slow_timed() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    fake(&h).lock().unwrap().capture.as_mut().unwrap().lag = Some(Duration::from_secs(7));
    let n = count(&h, code::STOP_CAPTURE);
    let c2 = connect(&h).await;
    let (r, e) = tokio::join!(
        ok(
            &c,
            "capture_record",
            json!({ "action": "start", "seconds": 3 })
        ),
        async {
            tokio::time::sleep(Duration::from_secs(6)).await;
            err(&c2, "capture_mode", json!({ "mode": "PHOTO_SINGLE" })).await
        }
    );
    println!("{r}\n{e}");
    assert!(e.contains("this host started"), "{e}");
    assert!(r["file"].as_str().unwrap().ends_with(".insv"), "{r}");
    assert_eq!(count(&h, code::STOP_CAPTURE), n + 1);
    h.shutdown().await;
}

/// A start the camera answers but never begins: after `host::LATE` (10 s) it is no longer this host's, so the camera
/// can be released again.
#[tokio::test(flavor = "multi_thread")]
async fn never_begun() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    fake(&h).lock().unwrap().capture.as_mut().unwrap().lag = Some(Duration::from_secs(3600));
    ok(&c, "capture_record", json!({ "action": "start" })).await;
    assert!(
        err(&c, "camera_release", json!({}))
            .await
            .contains("this host started")
    );
    assert!(until(30.0, || h.with(|s| s.rec.is_none())).await, "cleared");
    let e = h.with(|s| s.ended.clone());
    assert!(format!("{e:?}").contains("did not begin"), "{e:?}");
    assert_eq!(
        ok(&c, "camera_release", json!({})).await["connection"],
        "off"
    );
    h.shutdown().await;
}

/// A recording this host started, the link lost for 20 s: still this host's when the camera is back (the camera's
/// capture time says nothing here), and stopped at the host's end.
#[tokio::test(flavor = "multi_thread")]
async fn late_reconnect() {
    let h = host("read,local,camera").await;
    let c = on(&h).await;
    ok(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await;
    ok(
        &c,
        "capture_record",
        json!({ "action": "start", "seconds": 120 }),
    )
    .await;
    let l = fake(&h).lock().unwrap().link.clone().unwrap();
    l.unplug();
    tokio::time::sleep(Duration::from_secs(20)).await;
    ok(&c, "camera_connect", json!({})).await;
    let st = ok(&c, "capture_status", json!({})).await;
    assert_eq!(
        (&st["capturing"], &st["started_by_this_host"]),
        (&json!(true), &json!(true)),
        "{st}"
    );
    let n = count(&h, code::STOP_CAPTURE);
    h.shutdown().await;
    assert_eq!(count(&h, code::STOP_CAPTURE), n + 1);
}
