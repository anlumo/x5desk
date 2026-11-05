//! Opt-in, on a real X5 over USB in its Android App mode, a host with its agent folder and log in a scratch dir
//! (`Opts::dir`: nothing of the user's read or written), never setting the camera's clock:
//! - `X5_CAMERA=<scratch dir>` (the variable of the app's camera tests): connects, lists the album and downloads its
//!   smallest item; a second host of this process is refused the camera the first one holds; the first releases it.
//!   The camera keeps its files and settings.
//! - `X5_CAMERA_AGENT=<scratch dir>`: an agent's writes, in the modes and settings x5desk ran on an X5 (the camera
//!   must start in one of them, 360): white balance changed and set back, a photo, a timed clip, a clip left
//!   recording that the host's end stops (a second host finds it stopped and listed), a highlight mark; then the
//!   test's own items deleted. Mode, white balance and file list end as they began, also when a check fails.

use mcp::{Host, Opts, Tiers, config::Source};
use rmcp::{RoleClient, ServiceExt, model::CallToolRequestParams, service::RunningService};
use serde_json::{Value, json};
use std::{collections::HashSet, path::PathBuf, sync::Arc, time::Duration};

async fn connect(h: &Arc<Host>) -> RunningService<RoleClient, ()> {
    let (a, b) = tokio::io::duplex(1 << 20);
    let (r, w) = tokio::io::split(b);
    let hh = h.clone();
    tokio::spawn(async move { hh.serve(r, w, "test", None).await });
    ().serve(a).await.unwrap()
}

/// A call's JSON (Ok) or its error text (Err).
async fn call(c: &RunningService<RoleClient, ()>, name: &str, a: Value) -> Result<Value, String> {
    let p =
        CallToolRequestParams::new(name.to_string()).with_arguments(a.as_object().unwrap().clone());
    let r = c.call_tool(p).await.unwrap();
    let t: Vec<String> = r
        .content
        .iter()
        .filter_map(|c| c.as_text())
        .map(|t| t.text.clone())
        .collect();
    match r.is_error {
        Some(true) => Err(t.join("\n")),
        _ => Ok(r.structured_content.unwrap_or_default()),
    }
}

/// Waits up to 10 s for the bus watch to see the X5 in app mode.
async fn plugged(c: &RunningService<RoleClient, ()>) {
    for _ in 0..100 {
        if call(c, "status", json!({})).await.unwrap()["camera"]["usb"] == "app mode" {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await
    }
    panic!("no X5 in USB app mode")
}

#[tokio::test(flavor = "multi_thread")]
async fn real_camera() {
    let Some(d) = std::env::var_os("X5_CAMERA").map(PathBuf::from) else {
        eprintln!("X5_CAMERA not set: skipping");
        return;
    };
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let host = async |s: &str| {
        Host::new(Opts {
            dir: Some(d.join(s)),
            tiers: Some(Source::Fixed(Tiers::SIM)),
            ..Default::default()
        })
        .await
        .unwrap()
    };
    let h = host("a").await;
    let c = connect(&h).await;
    plugged(&c).await;
    let s = call(&c, "camera_connect", json!({})).await.unwrap();
    println!(
        "connected: {} {} {}, battery {}",
        s["model"], s["serial"], s["firmware"], s["battery_percent"]
    );
    assert_eq!(s["connection"], "on");
    let l = call(&c, "album_list", json!({ "limit": 500 }))
        .await
        .unwrap();
    let items = l["items"].as_array().unwrap();
    println!("album: {} items", l["total"]);
    let it = (items.iter())
        .filter(|i| i["size_bytes"].as_u64().unwrap_or(0) > 0)
        .min_by_key(|i| i["size_bytes"].as_u64())
        .expect("an item with files");
    println!("downloading {} ({} bytes)", it["key"], it["size_bytes"]);
    let j = call(&c, "album_download", json!({ "keys": [it["key"]] }))
        .await
        .unwrap();
    let w = call(&c, "job_wait", json!({ "job": j["job"], "wait_s": 300 }))
        .await
        .unwrap();
    assert_eq!(w["state"], "done", "{w}");
    for f in w["files"].as_array().unwrap() {
        let p = PathBuf::from(f["path"].as_str().unwrap());
        assert!(p.starts_with(d.join("a").join("agent")), "{}", p.display());
        assert_eq!(
            std::fs::metadata(&p).unwrap().len(),
            f["total_bytes"].as_u64().unwrap(),
            "{}",
            p.display()
        );
        let i = call(&c, "file_info", json!({ "path": p })).await.unwrap();
        println!("{}: {}", p.display(), i["trailer"]);
    }
    // the camera is this host's: another one is told so
    let h2 = host("b").await;
    let c2 = connect(&h2).await;
    plugged(&c2).await;
    let e = call(&c2, "camera_connect", json!({})).await.unwrap_err();
    println!("second host: {e}");
    assert!(e.contains("in use by another app"), "{e}");
    h2.shutdown().await;
    let r = call(&c, "camera_release", json!({})).await.unwrap();
    assert_eq!(r["connection"], "off");
    h.shutdown().await;
}

/// The album's keys and the files of each.
async fn files(c: &RunningService<RoleClient, ()>) -> Result<Vec<(String, Vec<String>)>, String> {
    let mut v = vec![];
    loop {
        let l = call(c, "album_list", json!({ "offset": v.len(), "limit": 500 })).await?;
        for i in l["items"].as_array().cloned().unwrap_or_default() {
            let f = (i["files"].as_array().cloned().unwrap_or_default().iter())
                .filter_map(|f| f.as_str().map(String::from))
                .collect();
            v.push((i["key"].as_str().unwrap_or_default().to_string(), f));
        }
        if l["more"] != true {
            return Ok(v);
        }
    }
}

/// A setting's value in a capture_status answer.
fn value(st: &Value, k: &str) -> Option<String> {
    (st["settings"].as_array()?.iter()).find(|x| x["key"] == k)?["value"]
        .as_str()
        .map(String::from)
}

fn check(ok: bool, what: impl FnOnce() -> String) -> Result<(), String> {
    if ok { Ok(()) } else { Err(what()) }
}

#[tokio::test(flavor = "multi_thread")]
async fn real_agent_writes() {
    let Some(d) = std::env::var_os("X5_CAMERA_AGENT").map(PathBuf::from) else {
        eprintln!("X5_CAMERA_AGENT not set: skipping");
        return;
    };
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let host = async |s: &str| {
        Host::new(Opts {
            dir: Some(d.join(s)),
            tiers: Some(Source::Fixed(
                Tiers::parse("read,local,camera,delete").unwrap(),
            )),
            ..Default::default()
        })
        .await
        .unwrap()
    };
    let h = host("a").await;
    let c = connect(&h).await;
    plugged(&c).await;
    let s = call(&c, "camera_connect", json!({})).await.unwrap();
    println!(
        "connected: {} {}, battery {}",
        s["model"], s["firmware"], s["battery_percent"]
    );
    let st = call(&c, "capture_status", json!({})).await.unwrap();
    let mode = st["mode"]["name"].as_str().unwrap_or_default().to_string();
    if st["mode"]["tested"] != true || st["capturing"] != false {
        eprintln!(
            "X5_CAMERA_AGENT: the camera must start idle in Photo or Video (360), not {mode}: skipping"
        );
        call(&c, "camera_release", json!({})).await.unwrap();
        h.shutdown().await;
        return;
    }
    let before = files(&c).await.unwrap();
    let old: HashSet<String> = before.iter().map(|x| x.0.clone()).collect();
    // the items the test made, by key
    let mut ours: Vec<String> = vec![];
    let mut wb: Option<String> = None;
    let r: Result<(), String> = async {
        call(&c, "capture_mode", json!({ "mode": "PHOTO_SINGLE" })).await?;
        let st = call(&c, "capture_status", json!({})).await?;
        let cur = value(&st, "white_balance").ok_or("no white balance in PHOTO_SINGLE")?;
        let set = (st["settings"].as_array().unwrap().iter())
            .find(|x| x["key"] == "white_balance")
            .unwrap();
        let other = (set["allowed"].as_array().unwrap().iter())
            .filter_map(|a| a["value"].as_str())
            .find(|a| *a != cur)
            .ok_or("one white balance only")?
            .to_string();
        wb = Some(cur.clone());
        let r = call(
            &c,
            "capture_set",
            json!({ "key": "white_balance", "value": other }),
        )
        .await?;
        println!("white balance {cur} → {}", r["value"]);
        let p = call(&c, "capture_photo", json!({ "self_timer": false })).await?;
        println!("photo {p}");
        ours.push(p["key"].as_str().ok_or("no photo key")?.into());
        call(
            &c,
            "capture_set",
            json!({ "key": "white_balance", "value": cur }),
        )
        .await?;
        call(&c, "capture_mode", json!({ "mode": "VIDEO_NORMAL" })).await?;
        let v = call(
            &c,
            "capture_record",
            json!({ "action": "start", "seconds": 3 }),
        )
        .await?;
        println!("timed clip {v}");
        ours.push(v["key"].as_str().ok_or("no clip key")?.into());
        let v = call(&c, "capture_record", json!({ "action": "start" })).await?;
        check(v["capturing"] == true, || format!("not recording: {v}"))?;
        tokio::time::sleep(Duration::from_millis(1500)).await;
        let m = call(&c, "capture_mark", json!({})).await?;
        println!("mark {m}");
        Ok(())
    }
    .await;
    // the host's end stops the clip it left recording
    h.shutdown().await;
    let log = std::fs::read_to_string(d.join("a").join("mcp.log")).unwrap_or_default();
    println!(
        "{}",
        log.lines()
            .filter(|l| l.contains("recording"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    // a second host: the clip stopped, then the test's items deleted and the camera set back, whatever failed before
    let h = host("b").await;
    let c = connect(&h).await;
    plugged(&c).await;
    let mut on = Err(String::new());
    for _ in 0..15 {
        on = call(&c, "camera_connect", json!({})).await;
        if on.is_ok() {
            break;
        }
        println!("connect: {on:?}");
        tokio::time::sleep(Duration::from_secs(2)).await
    }
    on.unwrap();
    let st = call(&c, "capture_status", json!({})).await;
    let stopped = st.as_ref().is_ok_and(|st| st["capturing"] == false);
    if !stopped {
        let s = call(&c, "capture_record", json!({ "action": "stop" })).await;
        println!("the end did not stop it ({st:?}); stopped now: {s:?}");
    }
    // the test's items: those it was told of, and new ones (the clip the end stopped), of files that were not there
    let had: HashSet<&String> = before.iter().flat_map(|x| &x.1).collect();
    let after = files(&c).await;
    for (k, f) in after.as_deref().unwrap_or_default() {
        if !old.contains(k) && !ours.contains(k) {
            println!("new item {k}: {f:?}");
            ours.push(k.clone())
        }
    }
    let mine = |k: &String| {
        !old.contains(k)
            && (after.as_deref().unwrap_or_default().iter())
                .find(|x| x.0 == *k)
                .is_some_and(|x| x.1.iter().all(|f| !had.contains(f)))
    };
    let del: Vec<&String> = ours.iter().filter(|k| mine(k)).collect();
    let mut gone = Ok(Value::Null);
    if !del.is_empty() {
        gone = call(&c, "camera_delete", json!({ "keys": del })).await;
        println!("deleted {gone:?}");
    }
    if let Some(w) = &wb {
        let _ = call(&c, "capture_mode", json!({ "mode": "PHOTO_SINGLE" })).await;
        let _ = call(
            &c,
            "capture_set",
            json!({ "key": "white_balance", "value": w }),
        )
        .await;
    }
    let back = call(&c, "capture_mode", json!({ "mode": mode })).await;
    let st = call(&c, "capture_status", json!({})).await;
    let now = files(&c).await;
    let _ = call(&c, "camera_release", json!({})).await;
    h.shutdown().await;
    let (st, now) = (st.unwrap(), now.unwrap());
    r.unwrap();
    assert!(stopped, "the host's end left the clip recording");
    assert_eq!(
        ours.len(),
        3,
        "a photo, a timed clip, the clip the end stopped: {ours:?}"
    );
    let g = gone.unwrap();
    assert_eq!(g["deleted"].as_array().map(Vec::len), Some(3), "{g}");
    back.unwrap();
    assert_eq!(st["mode"]["name"], mode.as_str());
    assert_eq!(now, before, "the file list as it began");
}
