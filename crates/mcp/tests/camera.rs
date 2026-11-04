//! Opt-in, on a real X5 over USB in its Android App mode: `X5_CAMERA=<scratch dir>` (the variable of the app's camera
//! tests). A host with its agent folder and log in that dir (`Opts::dir`: nothing of the user's read or written)
//! connects without setting the clock, lists the album and downloads its smallest item; a second host of this
//! process is refused the camera the first one holds; the first releases it. The camera keeps its files and settings.

use mcp::{Host, Opts, Tiers, config::Source};
use rmcp::{RoleClient, ServiceExt, model::CallToolRequestParams, service::RunningService};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc, time::Duration};

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
