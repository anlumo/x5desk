//! `x5desk mcp` as clients start it: the real binary, `--sim`, on an endpoint in a temp dir (never the user's; in
//! `/tmp` when its path would not fit `sun_path`; home, config, cache and temp dirs point into the temp dir too),
//! driven over its stdio with the JSON-RPC lines of a 2025-11-25 client. Every child is killed and reaped when its
//! guard drops, also when a test fails.

use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

/// A temp dir of the test's and its endpoint, removed whole when it drops.
struct Tmp {
    d: PathBuf,
    /// the endpoint the processes of this test share: a pipe named after the test, or a socket in `d`
    ep: PathBuf,
    /// a short dir of the socket's own when `d`'s path would not fit `sun_path`
    short: Option<PathBuf>,
}

impl Tmp {
    fn new(name: &str) -> Tmp {
        let n = format!("x5mcp-{name}-{}", std::process::id());
        let d = std::env::temp_dir().join(&n);
        let _ = std::fs::remove_dir_all(&d);
        for s in ["home", "config", "cache", "tmp"] {
            std::fs::create_dir_all(d.join(s)).unwrap();
        }
        #[cfg(unix)]
        let short = (d.join("mcp.sock").as_os_str().len() >= mcp::endpoint::SUN).then(|| {
            let s = Path::new("/tmp").join(&n);
            let _ = std::fs::remove_dir_all(&s);
            std::fs::create_dir_all(&s).unwrap();
            s
        });
        #[cfg(not(unix))]
        let short = None;
        let ep = match cfg!(windows) {
            true => format!(r"\\.\pipe\x5desk-test-{name}-{}", std::process::id()).into(),
            false => short.as_ref().unwrap_or(&d).join("mcp.sock"),
        };
        Tmp { d, ep, short }
    }

    /// The endpoint the processes of this test share.
    fn endpoint(&self) -> PathBuf {
        self.ep.clone()
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.d);
        if let Some(s) = &self.short {
            let _ = std::fs::remove_dir_all(s);
        }
    }
}

/// `x5desk <args>` with the test's dirs and endpoint.
fn x5desk(t: &Tmp, args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_x5desk"));
    c.args(args)
        .env("X5DESK_MCP_ENDPOINT", t.endpoint())
        .env("HOME", t.d.join("home"))
        .env("XDG_CONFIG_HOME", t.d.join("config"))
        .env("XDG_CACHE_HOME", t.d.join("cache"))
        .env("TMPDIR", t.d.join("tmp"))
        .env("USERPROFILE", t.d.join("home"))
        .env("APPDATA", t.d.join("config"))
        .env("LOCALAPPDATA", t.d.join("cache"))
        .env_remove("XDG_RUNTIME_DIR")
        .env_remove("X5DESK_TEST")
        .env_remove("X5DESK_MCP_TIERS");
    c
}

/// A running `x5desk mcp --sim`.
struct Kid {
    c: Child,
    stdin: Option<ChildStdin>,
    /// stdout's lines as they come
    out: mpsc::Receiver<String>,
    /// all of stdout and stderr so far
    all: Arc<Mutex<Vec<String>>>,
    err: Arc<Mutex<String>>,
    next: u64,
    /// a 2026-07-28 client: no handshake, its version and capabilities in every request's `_meta`
    modern: bool,
}

impl Drop for Kid {
    fn drop(&mut self) {
        let _ = self.c.kill();
        let _ = self.c.wait();
    }
}

fn spawn(t: &Tmp, env: &[(&str, &str)]) -> Kid {
    spawn_args(t, &["mcp", "--sim"], env)
}

fn spawn_args(t: &Tmp, args: &[&str], env: &[(&str, &str)]) -> Kid {
    let mut c = x5desk(t, args);
    c.envs(env.iter().copied());
    spawn_cmd(c)
}

/// Runs `c` with its stdio piped.
fn spawn_cmd(mut c: Command) -> Kid {
    c.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut c = c.spawn().unwrap();
    let (tx, out) = mpsc::channel();
    let (all, err) = (
        Arc::new(Mutex::new(vec![])),
        Arc::new(Mutex::new(String::new())),
    );
    let (o, a) = (c.stdout.take().unwrap(), all.clone());
    std::thread::spawn(move || {
        for l in BufReader::new(o).lines() {
            let Ok(l) = l else { break };
            a.lock().unwrap().push(l.clone());
            let _ = tx.send(l);
        }
    });
    let (mut e, ee) = (c.stderr.take().unwrap(), err.clone());
    std::thread::spawn(move || {
        let mut b = [0; 4096];
        while let Ok(n @ 1..) = e.read(&mut b) {
            ee.lock()
                .unwrap()
                .push_str(&String::from_utf8_lossy(&b[..n]));
        }
    });
    Kid {
        stdin: c.stdin.take(),
        c,
        out,
        all,
        err,
        next: 0,
        modern: false,
    }
}

impl Kid {
    fn send(&mut self, v: Value) {
        let i = self.stdin.as_mut().unwrap();
        writeln!(i, "{v}").unwrap();
        i.flush().unwrap();
    }

    fn request(&mut self, method: &str, mut params: Value) -> u64 {
        self.next += 1;
        let id = self.next;
        if self.modern {
            params["_meta"] = json!({
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": { "name": "x5desk-test", "version": "1" },
            });
        }
        self.send(json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        id
    }

    /// The answer to request `id`, notifications on the way skipped.
    fn answer(&mut self, id: u64) -> Value {
        self.answer_in(id, 20).0
    }

    /// The answer to request `id` within `secs`, and the notifications before it.
    fn answer_in(&mut self, id: u64, secs: u64) -> (Value, Vec<Value>) {
        let (end, mut notes) = (Instant::now() + Duration::from_secs(secs), vec![]);
        loop {
            let left = end.saturating_duration_since(Instant::now());
            let l = (self.out.recv_timeout(left)).unwrap_or_else(|e| {
                panic!(
                    "no answer to {id}: {e}; stderr:\n{}",
                    self.err.lock().unwrap()
                )
            });
            let v: Value = serde_json::from_str(&l).unwrap();
            if v["id"] == id {
                return (v, notes);
            }
            if v.get("id").is_none() {
                notes.push(v)
            }
        }
    }

    /// The 2025-11-25 handshake.
    fn init(&mut self) -> Value {
        let p = json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "x5desk-test", "version": "1" },
        });
        let id = self.request("initialize", p);
        let r = self.answer(id);
        self.send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        r
    }

    /// A tool call's answer: a result or an error.
    fn call(&mut self, name: &str, args: Value) -> Value {
        let id = self.request("tools/call", json!({ "name": name, "arguments": args }));
        self.answer(id)
    }

    /// A tool call's JSON, which must succeed.
    fn ok(&mut self, name: &str, args: Value) -> Value {
        let r = self.call(name, args);
        assert_eq!(r["result"]["isError"], false, "{name}: {r}");
        r["result"]["structuredContent"].clone()
    }

    /// Ends stdin and waits for the exit.
    fn close(&mut self) -> ExitStatus {
        drop(self.stdin.take());
        self.wait()
    }

    fn wait(&mut self) -> ExitStatus {
        let end = Instant::now() + Duration::from_secs(20);
        while Instant::now() < end {
            if let Some(s) = self.c.try_wait().unwrap() {
                return s;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("still running; stderr:\n{}", self.err.lock().unwrap())
    }

    fn stderr(&self) -> String {
        self.err.lock().unwrap().clone()
    }
}

/// Every line on stdout is JSON-RPC, a forced `println!` lands on stderr, the marker and the camera's access-point
/// password never on stdout, and stdin's end stops the host cleanly: exit 0, its socket and temp dir gone.
#[test]
fn stdout_is_protocol() {
    let t = Tmp::new("stdout");
    let mut k = spawn(&t, &[("X5DESK_TEST", "mcp-print")]);
    let r = k.init();
    assert_eq!(r["result"]["protocolVersion"], "2025-11-25", "{r}");
    assert_eq!(r["result"]["serverInfo"]["name"], "x5desk");
    let id = k.request("tools/list", json!({}));
    let tools = k.answer(id);
    assert!(
        tools["result"]["tools"].as_array().unwrap().len() > 10,
        "{tools}"
    );
    let s = k.ok("status", json!({}));
    assert_eq!(
        (
            &s["host"]["kind"],
            &s["host"]["simulated_camera"],
            &s["client"],
            &s["protocol"]
        ),
        (
            &json!("headless"),
            &json!(true),
            &json!("stdio"),
            &json!("2025-11-25")
        )
    );
    assert_eq!(k.ok("camera_connect", json!({}))["connection"], "on");
    k.ok("camera_status", json!({}));
    assert!(
        !k.ok("album_list", json!({}))["items"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    if !cfg!(windows) {
        assert!(t.endpoint().exists(), "the host's socket");
    }
    assert!(k.close().success(), "stderr:\n{}", k.stderr());
    // the reader thread has seen stdout's end once the process is gone and its pipe drained
    std::thread::sleep(Duration::from_millis(200));
    let all = k.all.lock().unwrap().clone();
    assert_eq!(all.len(), 6, "an answer to each request");
    for l in &all {
        let v: Value =
            serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON on stdout: {l:?}: {e}"));
        assert_eq!(v["jsonrpc"], "2.0", "{l}");
        assert!(
            !l.contains(mcp::sim::MARKER) && !l.contains(mcp::sim::AP),
            "{l}"
        );
    }
    let e = k.stderr();
    assert!(
        e.contains("X5DESK_TEST=mcp-print: a line printed on stdout"),
        "{e}"
    );
    assert!(!e.contains(mcp::sim::MARKER) && !e.contains(mcp::sim::AP));
    assert!(!t.endpoint().exists(), "the socket goes with its host");
    let left: Vec<_> = std::fs::read_dir(t.d.join("tmp")).unwrap().collect();
    assert!(
        left.is_empty(),
        "the simulated camera's temp dir goes with its host: {left:?}"
    );
}

/// Status's host id once it is not `not`, calls that meet the host's end (-32603) tried again.
fn host_id(k: &mut Kid, not: &str) -> String {
    let end = Instant::now() + Duration::from_secs(20);
    while Instant::now() < end {
        let r = k.call("status", json!({}));
        if let Some(e) = r.get("error") {
            assert_eq!(e["code"], -32603, "{r}");
            assert_eq!(e["message"], "host ended; try again");
            continue;
        }
        let s = &r["result"]["structuredContent"];
        let id = s["host"]["id"].as_str().unwrap().to_string();
        if id != not {
            // the new host saw the replayed 2025-11-25 handshake; a 2026-07-28 client needs none
            let v = if k.modern { "2026-07-28" } else { "2025-11-25" };
            assert_eq!(s["protocol"], v, "{s}");
            return id;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("no new host; stderr:\n{}", k.stderr())
}

/// Three clients share the first one's host (one of them on 2026-07-28); with it killed, the others elect one of
/// them, the 2025-11-25 client's handshake replayed, and agree on it.
#[test]
fn shared_host() {
    let t = Tmp::new("shared");
    let mut a = spawn(&t, &[]);
    a.init();
    let s = a.ok("status", json!({}));
    let first = s["host"]["id"].as_str().unwrap().to_string();
    let mut b = spawn(&t, &[]);
    let mut c = spawn(&t, &[]);
    c.modern = true;
    assert_eq!(b.init()["result"]["protocolVersion"], "2025-11-25");
    for k in [&mut b, &mut c] {
        let s = k.ok("status", json!({}));
        assert_eq!(s["host"]["id"], first, "proxied: {s}");
        assert!(
            s["client"].as_str().unwrap().starts_with("endpoint "),
            "{s}"
        );
    }
    assert!(
        b.stderr().contains("proxy to the host at"),
        "{}",
        b.stderr()
    );
    a.c.kill().unwrap();
    a.c.wait().unwrap();
    let nb = host_id(&mut b, &first);
    let nc = host_id(&mut c, &first);
    assert_eq!(nb, nc, "one host for both");
    for k in [&b, &c] {
        assert!(k.stderr().contains("the host went away"), "{}", k.stderr());
    }
    // the new host's camera, through either client
    assert_eq!(b.ok("camera_connect", json!({}))["connection"], "on");
    assert_eq!(c.ok("camera_status", json!({}))["connection"], "on");
    assert!(b.close().success(), "{}", b.stderr());
    assert!(c.close().success(), "{}", c.stderr());
}

/// Signal `sig` ends a host cleanly: exit 0, the socket and the simulated camera's temp dir gone.
#[cfg(unix)]
fn ends_at(sig: i32, name: &str) {
    let t = Tmp::new(name);
    let mut k = spawn(&t, &[]);
    k.init();
    assert_eq!(k.ok("camera_connect", json!({}))["connection"], "on");
    // SAFETY: a signal to our own child, which has not been reaped (it is still running).
    assert_eq!(unsafe { libc::kill(k.c.id() as i32, sig) }, 0);
    assert!(k.wait().success(), "{}", k.stderr());
    assert!(k.stderr().contains("stopping"), "{}", k.stderr());
    assert!(!t.endpoint().exists());
    let left: Vec<_> = std::fs::read_dir(t.d.join("tmp")).unwrap().collect();
    assert!(left.is_empty(), "{left:?}");
}

/// SIGTERM ends a host cleanly.
#[cfg(unix)]
#[test]
fn sigterm() {
    ends_at(libc::SIGTERM, "term")
}

/// SIGHUP (the client's terminal closed) ends a host cleanly.
#[cfg(unix)]
#[test]
fn sighup() {
    ends_at(libc::SIGHUP, "hup")
}

/// Started with SIGHUP and SIGINT ignored (`nohup`, a shell's background job), a host keeps them so: it lives through
/// both, still answers, and ends cleanly at stdin's end.
#[cfg(unix)]
#[test]
fn nohup() {
    use std::os::unix::process::CommandExt;
    let t = Tmp::new("nohup");
    let mut c = x5desk(&t, &["mcp", "--sim"]);
    // SAFETY: signal(2) is async-signal-safe and touches nothing of the parent's; SIG_IGN stays across exec.
    unsafe {
        c.pre_exec(|| {
            for s in [libc::SIGHUP, libc::SIGINT] {
                if libc::signal(s, libc::SIG_IGN) == libc::SIG_ERR {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let mut k = spawn_cmd(c);
    k.init();
    assert_eq!(k.ok("camera_connect", json!({}))["connection"], "on");
    for s in [libc::SIGHUP, libc::SIGINT] {
        // SAFETY: a signal to our own child, which has not been reaped (it is still running).
        assert_eq!(unsafe { libc::kill(k.c.id() as i32, s) }, 0);
    }
    std::thread::sleep(Duration::from_millis(300));
    assert!(k.c.try_wait().unwrap().is_none(), "{}", k.stderr());
    assert_eq!(k.ok("status", json!({}))["camera"]["connection"], "on");
    assert!(k.close().success(), "{}", k.stderr());
    assert!(!k.stderr().contains("stopping"), "{}", k.stderr());
    assert!(!t.endpoint().exists());
    let left: Vec<_> = std::fs::read_dir(t.d.join("tmp")).unwrap().collect();
    assert!(left.is_empty(), "{left:?}");
}

/// A host started with `--tiers read` narrows its own client only: a proxy without `--tiers` gets the host's tiers
/// (here `X5DESK_MCP_TIERS`), and the narrowed client is not told to enable what is enabled.
#[test]
fn tiers_per_client() {
    let t = Tmp::new("narrow");
    let env = [("X5DESK_MCP_TIERS", "read,local")];
    let mut a = spawn_args(&t, &["mcp", "--sim", "--tiers", "read"], &env);
    a.init();
    let s = a.ok("status", json!({}));
    assert_eq!(
        (&s["client"], &s["tiers"]),
        (&json!("stdio"), &json!(["read"])),
        "{s}"
    );
    let first = s["host"]["id"].clone();
    let tools = |k: &mut Kid| -> Vec<String> {
        let id = k.request("tools/list", json!({}));
        let r = k.answer(id);
        let v = r["result"]["tools"].as_array().unwrap().iter();
        v.map(|t| t["name"].as_str().unwrap().to_string()).collect()
    };
    assert!(!tools(&mut a).contains(&"album_download".to_string()));
    let r = a.call("album_download", json!({ "keys": ["x"] }));
    let e = r["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(r["result"]["isError"], true, "{r}");
    assert!(
        e.contains("--tiers leaves out") && !e.contains("enable"),
        "{e}"
    );
    let mut b = spawn(&t, &env);
    b.init();
    let s = b.ok("status", json!({}));
    assert_eq!(s["host"]["id"], first, "proxied: {s}");
    assert_eq!(s["tiers"], json!(["read", "local"]), "{s}");
    let n = tools(&mut b);
    assert!(
        n.contains(&"album_download".to_string()) && n.contains(&"job_cancel".to_string()),
        "{n:?}"
    );
    let r = b.call("album_download", json!({ "keys": ["x"] }));
    assert!(!r.to_string().contains("tier"), "a tier of the host's: {r}");
    assert!(b.close().success(), "{}", b.stderr());
    assert!(a.close().success(), "{}", a.stderr());
}

/// A client that writes its requests and closes stdin at once still gets every answer, from a host of its own
/// process and through a proxy, and the process exits 0.
#[test]
fn stdin_end_answers() {
    let t = Tmp::new("drain");
    let oneshot = |k: &mut Kid| -> Vec<Value> {
        let p = json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "x5desk-test", "version": "1" },
        });
        k.request("initialize", p);
        k.send(json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        k.request("tools/list", json!({}));
        k.request("tools/call", json!({ "name": "status", "arguments": {} }));
        k.request(
            "tools/call",
            json!({ "name": "camera_connect", "arguments": {} }),
        );
        assert!(k.close().success(), "{}", k.stderr());
        std::thread::sleep(Duration::from_millis(200));
        let all = k.all.lock().unwrap().clone();
        let mut v: Vec<Value> = all
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        // answered as each is done
        v.retain(|x| x.get("id").is_some());
        v.sort_by_key(|x| x["id"].as_u64());
        assert_eq!(
            v.iter().map(|x| x["id"].clone()).collect::<Vec<_>>(),
            [json!(1), json!(2), json!(3), json!(4)],
            "stderr:\n{}",
            k.stderr()
        );
        assert_eq!(
            v[3]["result"]["structuredContent"]["connection"], "on",
            "{}",
            v[3]
        );
        v
    };
    let mut h = spawn(&t, &[]);
    h.init();
    let v = oneshot(&mut spawn(&t, &[]));
    let c = v[2]["result"]["structuredContent"]["client"]
        .as_str()
        .unwrap();
    assert!(c.starts_with("endpoint "), "proxied: {}", v[2]);
    assert!(h.close().success(), "{}", h.stderr());
    let v = oneshot(&mut spawn(&t, &[]));
    assert_eq!(
        v[2]["result"]["structuredContent"]["client"], "stdio",
        "{}",
        v[2]
    );
}

/// Usage errors exit 2, `enable` writes the tiers into the (test's) config dir, `--print-config` names this binary.
#[test]
fn command_line() {
    let t = Tmp::new("cli");
    let o = x5desk(&t, &["mcp", "--bogus"]).output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(
        e.contains("unknown argument") && e.contains("usage: x5desk mcp"),
        "{e}"
    );
    assert!(o.stdout.is_empty());
    let o = x5desk(&t, &["mcp", "enable", "--tiers", "read,local"])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(String::from_utf8_lossy(&o.stdout).contains("enabled: read,local"));
    let f = t.d.join("config").join("x5desk").join("mcp");
    assert_eq!(std::fs::read_to_string(&f).unwrap(), "tiers = read,local\n");
    let o = x5desk(&t, &["mcp", "disable"]).output().unwrap();
    assert!(o.status.success());
    assert_eq!(std::fs::read_to_string(&f).unwrap(), "tiers = \n");
    let o = x5desk(&t, &["mcp", "--print-config"]).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let s = String::from_utf8(o.stdout).unwrap();
    assert!(
        s.contains("claude mcp add --transport stdio x5desk -- "),
        "{s}"
    );
    let j: Value = serde_json::from_str(&s[s.find('{').unwrap()..]).unwrap();
    let exe = Path::new(env!("CARGO_BIN_EXE_x5desk"));
    let got = PathBuf::from(j["mcpServers"]["x5desk"]["command"].as_str().unwrap());
    assert_eq!(got.canonicalize().unwrap(), exe.canonicalize().unwrap());
    assert_eq!(j["mcpServers"]["x5desk"]["args"], json!(["mcp"]));
}

// The GPU tools on this machine's GPU: the binary's own engine (one per server process) on synth's recording and on
// `X5_SAMPLES`. The servers run `--sim` with `X5DESK_TEST=mcp-sim-dir`, so their agent folder, log and records stay in
// the test's temp dir after they end (`kept`).

/// The GPU tests one at a time: each server opens a device of its own, and some time what they do.
fn gpu() -> std::sync::MutexGuard<'static, ()> {
    static GPU: Mutex<()> = Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// The servers' `XDG_RUNTIME_DIR`: burn's cubecl cache in it (`x5desk-cubecl`) stays warm across the GPU tests and
/// runs; only `gpu_sigterm_compile` starts cold.
fn warm() -> PathBuf {
    let d = std::env::temp_dir().join("x5mcp-gpu-run");
    private(&d);
    d
}

/// Dir `d`, made owner-only (cubecl's base must be the user's alone).
fn private(d: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(d).unwrap();
    std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700)).unwrap();
}

/// synth's recording, 300 frames of 256² a lens (no IMU), in `<t>/src`.
fn synth(t: &Tmp) -> PathBuf {
    let p = t.d.join("src").join("synth300.insv");
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    export::synth::insv(&p, 300, 256).unwrap();
    p
}

/// `x5desk mcp --sim` with GPU tools, initialised: `allow` a read root, its dirs kept (`mcp-sim-dir`, then the tokens
/// in `more`), burn's cache under `run`.
fn gpu_kid(t: &Tmp, allow: &Path, more: &str, run: &Path) -> Kid {
    let test = format!("mcp-sim-dir{more}");
    let args = ["mcp", "--sim", "--allow-dir", allow.to_str().unwrap()];
    let env = [
        ("X5DESK_TEST", test.as_str()),
        ("XDG_RUNTIME_DIR", run.to_str().unwrap()),
    ];
    let mut k = spawn_args(t, &args, &env);
    k.init();
    k
}

/// `k`'s kept dir (`mcp-sim-dir`): the agent folder `agent`, `mcp.log`, `mcp-renders`.
fn kept(t: &Tmp, k: &Kid) -> PathBuf {
    t.d.join("tmp").join(format!("x5desk-mcp-{}", k.c.id()))
}

/// A tool result's JSON: its structured content, else (a done still) its first text block.
fn data(r: &Value) -> Value {
    match &r["result"]["structuredContent"] {
        Value::Null => {
            serde_json::from_str(r["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
        }
        s => s.clone(),
    }
}

impl Kid {
    /// A GPU tool's call answered within `secs`, which must succeed: its JSON and the whole answer.
    fn render(&mut self, name: &str, args: Value, secs: u64) -> (Value, Value) {
        let id = self.request("tools/call", json!({ "name": name, "arguments": args }));
        let r = self.answer_in(id, secs).0;
        assert_eq!(r["result"]["isError"], false, "{name}: {r}");
        (data(&r), r)
    }

    /// `job_wait` on `job` up to `secs` with a progress token, called again while it runs (a call waits 50 s at
    /// most): its JSON, the job's progress notifications.
    fn wait_job(&mut self, job: &str, secs: u64) -> (Value, Vec<Value>) {
        let end = Instant::now() + Duration::from_secs(secs);
        let mut all = vec![];
        loop {
            let w = end
                .saturating_duration_since(Instant::now())
                .as_secs()
                .clamp(1, 50);
            let a = json!({ "job": job, "wait_s": w });
            let p =
                json!({ "name": "job_wait", "arguments": a, "_meta": { "progressToken": job } });
            let id = self.request("tools/call", p);
            let (r, n) = self.answer_in(id, w + 20);
            assert_eq!(r["result"]["isError"], false, "{r}");
            all.extend(n.into_iter().filter(|v| {
                v["method"] == "notifications/progress" && v["params"]["progressToken"] == job
            }));
            let v = data(&r);
            if v["finished"] == true || Instant::now() >= end {
                return (v, all);
            }
        }
    }

    /// Job `job`'s state once `f` holds (looked at every 50 ms, up to `secs`); it must not finish before.
    fn until(&mut self, job: &str, secs: u64, f: impl Fn(&Value) -> bool) -> Value {
        let end = Instant::now() + Duration::from_secs(secs);
        loop {
            let v = self.ok("job_status", json!({ "job": job }));
            if f(&v) {
                return v;
            }
            assert!(v["finished"] != true, "finished first: {v}");
            assert!(Instant::now() < end, "{v}; stderr:\n{}", self.stderr());
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// At least `n` frames done.
fn frames_at(n: u64) -> impl Fn(&Value) -> bool {
    move |v| v["frames_done"].as_u64().is_some_and(|d| d >= n)
}

/// A JPEG's width and height, from its frame header.
fn jpeg_size(b: &[u8]) -> [u32; 2] {
    let mut i = 2;
    while i + 9 < b.len() {
        assert_eq!(b[i], 0xff, "a marker at {i}");
        let (m, n) = (b[i + 1], u16::from_be_bytes([b[i + 2], b[i + 3]]) as usize);
        if (0xc0..=0xcf).contains(&m) && ![0xc4, 0xc8, 0xcc].contains(&m) {
            let x = |j: usize| u16::from_be_bytes([b[i + j], b[i + j + 1]]) as u32;
            return [x(7), x(5)];
        }
        i += 2 + n;
    }
    panic!("no frame header")
}

/// Standard base64, decoded.
fn unb64(s: &str) -> Vec<u8> {
    let v = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64: {c}"),
    } as u32;
    let mut o = vec![];
    for c in s.trim_end_matches('=').as_bytes().chunks(4) {
        let n = c.iter().fold(0, |a, &x| a << 6 | v(x)) << (6 * (4 - c.len()));
        o.extend_from_slice(&n.to_be_bytes()[1..c.len()]);
    }
    o
}

/// The video samples of MP4 `p`.
fn samples(p: &Path) -> usize {
    let m = media::mp4::Mp4::open(p).unwrap();
    (m.tracks.iter())
        .filter(|t| &t.handler == b"vide")
        .map(|t| t.samples.len())
        .sum()
}

/// Job `job`'s line in `<d>/mcp-renders`.
fn record(d: &Path, job: &str) -> Value {
    let s = std::fs::read_to_string(d.join("mcp-renders")).unwrap();
    (s.lines().filter(|l| !l.starts_with('#')))
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|v| v["job"] == job)
        .unwrap_or_else(|| panic!("{job} not in\n{s}"))
}

/// The names in dir `d` (none when it is missing), sorted.
fn names(d: &Path) -> Vec<String> {
    let mut v: Vec<String> = (std::fs::read_dir(d).into_iter().flatten())
        .map(|e| e.unwrap().file_name().to_string_lossy().into())
        .collect();
    v.sort();
    v
}

/// The job of DTO `v` left nothing: neither its file nor its `.part`.
fn nothing(v: &Value) {
    let p = PathBuf::from(v["path"].as_str().unwrap());
    for f in [&p, &media::part::of(&p)] {
        assert!(f.symlink_metadata().is_err(), "{} is there", f.display());
    }
}

/// Neither an `.mp4` nor a `.part` in `d`'s agent folder.
fn no_video(d: &Path) {
    let v = names(&d.join("agent"));
    let left: Vec<_> = (v.iter())
        .filter(|n| n.ends_with(".mp4") || n.ends_with(".part"))
        .collect();
    assert!(left.is_empty(), "{left:?}");
}

/// A long export of `src`: 360 at 5760 × 2880 in H.265 Main 10 (`pano57`) without RAFT, the whole clip.
fn long(src: &Path) -> Value {
    json!({ "path": src.to_str().unwrap(), "preset": "pano57", "raft": false, "stabilisation": "off" })
}

/// A string as it is, anything else as JSON (for what the tests print).
fn bare(v: &Value) -> String {
    v.as_str().map_or_else(|| v.to_string(), String::from)
}

/// The job handle in DTO `v`.
fn job(v: &Value) -> String {
    v["job"].as_str().unwrap().to_string()
}

/// `x5desk mcp gpu-check`, the self-test child, on this machine's GPU (its stdin held open: it ends at its EOF): one
/// `"v":1,"ok":true` line with the device's caps, nothing else on stdout, exit 0.
#[test]
fn gpu_selftest() {
    let _g = gpu();
    let t = Tmp::new("selftest");
    let mut k = spawn_args(&t, &["mcp", "gpu-check"], &[]);
    let s = k.wait();
    assert!(s.success(), "{s}; stderr:\n{}", k.stderr());
    std::thread::sleep(Duration::from_millis(200));
    let all = k.all.lock().unwrap().clone();
    assert_eq!(all.len(), 1, "{all:?}");
    let v: Value = serde_json::from_str(&all[0]).unwrap();
    assert_eq!((&v["v"], &v["ok"]), (&json!(1), &json!(true)), "{v}");
    let i = &v["info"];
    assert!(!i["adapter"].as_str().unwrap().is_empty(), "{v}");
    let has = |k: &str, c: &str| (i[k].as_array().unwrap().iter()).any(|x| x["codec"] == c);
    assert!(has("decode", "h265") && has("encode", "h264"), "{v}");
}

/// burn kept off the device (`mcp-noattach`, as a failed attach): the GPU starts without RAFT and a still (its
/// default seam the template then) renders; the catalog says why, and `raft: true` is refused with the reason.
#[test]
fn gpu_attach_fail() {
    let _g = gpu();
    let t = Tmp::new("noattach");
    let src = synth(&t);
    let mut k = gpu_kid(&t, src.parent().unwrap(), ",mcp-noattach", &warm());
    let mut a =
        json!({ "path": src.to_str().unwrap(), "output": "360", "width": 1024, "time_s": 1 });
    let (v, _) = k.render("render_still", a.clone(), 60);
    assert_eq!(
        (&v["state"], &v["settings"]["raft"]),
        (&json!("done"), &json!(false)),
        "{v}"
    );
    let c = k.ok("catalog", json!({}));
    let why = "attach turned off (X5DESK_TEST=mcp-noattach)";
    assert_eq!(c["gpu"]["state"], "ready", "{}", c["gpu"]);
    assert_eq!(c["gpu"]["raft"], json!({ "available": false, "why": why }));
    assert_eq!(
        (
            &c["raft"]["available"],
            &c["raft"]["default"],
            &c["raft"]["why"]
        ),
        (&json!(false), &json!(false), &json!(why)),
        "{}",
        c["raft"]
    );
    a["raft"] = true.into();
    let r = k.call("render_still", a);
    let e = r["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(r["result"]["isError"], true, "{r}");
    assert!(e.contains(why) && e.contains("raft: false"), "{e}");
    assert!(k.close().success(), "{}", k.stderr());
}

/// synth's recording through the engine: a template 360 still at 3840 (the file 3840 × 1920 with GPano, a 1568 × 784
/// preview in the answer's second block); a flat H.264 export at QP 24, then a longer one (pano57 without RAFT), each
/// followed by job_wait to done with rising progress out of the frames + 1: 300 samples.
#[test]
fn gpu_synthetic() {
    let _g = gpu();
    let t = Tmp::new("synth");
    let src = synth(&t);
    let mut k = gpu_kid(&t, src.parent().unwrap(), "", &warm());
    let p = src.to_str().unwrap();
    let a = json!({ "path": p, "output": 360, "width": 3840, "raft": false, "time_s": 2 });
    let (v, r) = k.render("render_still", a, 60);
    assert_eq!(
        (&v["state"], &v["width"], &v["height"]),
        (&json!("done"), &json!(3840), &json!(1920)),
        "{v}"
    );
    let c = r["result"]["content"].as_array().unwrap();
    assert_eq!(
        (c.len(), &c[1]["type"], &c[1]["mimeType"]),
        (2, &json!("image"), &json!("image/jpeg"))
    );
    assert!(r["result"]["structuredContent"].is_null(), "{r}");
    let pv = unb64(c[1]["data"].as_str().unwrap());
    assert_eq!((jpeg_size(&pv), pv.len() < 1_000_000), ([1568, 784], true));
    let f = std::fs::read(v["path"].as_str().unwrap()).unwrap();
    assert_eq!(jpeg_size(&f), [3840, 1920]);
    assert!(f.windows(5).any(|w| w == b"GPano"), "no GPano");
    let a = json!({
        "path": p, "output": "flat", "width": 1280, "height": 720, "codec": "h264", "qp": 24, "raft": false,
        "stabilisation": "off",
    });
    // the flat export, then a long one (pano57 without RAFT) that job_wait follows for some seconds
    for (a, qp) in [(a, json!(24)), (long(&src), Value::Null)] {
        let (v, _) = k.render("export", a, 60);
        assert_eq!(v["frames_total"], 300, "{v}");
        let t0 = Instant::now();
        let (v, n) = k.wait_job(&job(&v), 120);
        let s = &v["settings"];
        println!(
            "synth {} {} × {} {}: {:.1} s, {} frames/s, {} notifications",
            bare(&s["output"]),
            s["width"],
            s["height"],
            bare(&s["codec"]),
            t0.elapsed().as_secs_f64(),
            v["fps"],
            n.len()
        );
        assert_eq!(
            (&v["state"], &v["frames_done"], &s["qp"]),
            (&json!("done"), &json!(300), &qp),
            "{v}"
        );
        assert_eq!(samples(Path::new(v["path"].as_str().unwrap())), 300);
        let x: Vec<f64> = (n.iter())
            .map(|n| n["params"]["progress"].as_f64().unwrap())
            .collect();
        assert!(n.iter().all(|n| n["params"]["total"] == 301.0), "{n:?}");
        assert!(x.windows(2).all(|w| w[0] < w[1]), "{x:?}");
        if qp.is_null() {
            assert!(x.len() >= 2, "{n:?}");
        }
    }
    assert!(k.close().success(), "{}", k.stderr());
}

/// A running export cancelled at ≥ 10 frames: `cancelled` within 3 s, neither its file nor its `.part`, the agent
/// folder as before, the record ended so.
#[test]
fn gpu_cancel_leaves_nothing() {
    let _g = gpu();
    let t = Tmp::new("gcancel");
    let src = synth(&t);
    let mut k = gpu_kid(&t, src.parent().unwrap(), "", &warm());
    let d = kept(&t, &k);
    let before = names(&d.join("agent"));
    let (v, _) = k.render("export", long(&src), 60);
    let h = job(&v);
    k.until(&h, 60, frames_at(10));
    let t0 = Instant::now();
    let v = k.ok("job_cancel", json!({ "job": h }));
    assert_eq!(v["state"], "cancelled", "{v}");
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    nothing(&v);
    assert_eq!(names(&d.join("agent")), before);
    assert!(k.close().success(), "{}", k.stderr());
    assert_eq!(record(&d, &h)["end"]["state"], "cancelled");
}

/// A still started at ≥ 10 frames of a RAFT export ends first: stills have a lane of their own.
#[test]
fn gpu_lanes() {
    let _g = gpu();
    let t = Tmp::new("lanes");
    let src = synth(&t);
    let mut k = gpu_kid(&t, src.parent().unwrap(), "", &warm());
    let p = src.to_str().unwrap();
    let (v, _) = k.render("export", json!({ "path": p, "preset": "pano4k" }), 60);
    assert_eq!(v["settings"]["raft"], true, "{v}");
    let h = job(&v);
    k.until(&h, 120, frames_at(10));
    let a = json!({ "path": p, "output": "flat", "width": 640, "height": 360, "raft": false });
    let (s, _) = k.render("render_still", a, 60);
    assert_eq!(s["state"], "done", "{s}");
    let v = k.ok("job_status", json!({ "job": h }));
    assert_eq!(v["finished"], false, "the export ended first: {v}");
    assert_eq!(
        k.ok("job_cancel", json!({ "job": h }))["state"],
        "cancelled"
    );
    assert!(k.close().success(), "{}", k.stderr());
}

/// stdin's end at ≥ 10 frames of an export: exit 0 within 15 s, neither the file nor its `.part`, and the job's
/// record (what the next host on that dir reads) ended cancelled with its host.
#[test]
fn gpu_end_mid_export() {
    let _g = gpu();
    let t = Tmp::new("gend");
    let src = synth(&t);
    let mut k = gpu_kid(&t, src.parent().unwrap(), "", &warm());
    let (v, _) = k.render("export", long(&src), 60);
    let h = job(&v);
    k.until(&h, 60, frames_at(10));
    let t0 = Instant::now();
    assert!(k.close().success(), "{}", k.stderr());
    assert!(t0.elapsed() < Duration::from_secs(15), "{:?}", t0.elapsed());
    nothing(&v);
    let r = record(&kept(&t, &k), &h);
    assert_eq!(r["end"]["state"], "cancelled", "{r}");
    assert!(
        r["end"]["err"]
            .as_str()
            .unwrap()
            .starts_with("its host ended"),
        "{r}"
    );
}

/// SIGTERM to `k` while its export `h` runs: exit 0 within 15 s, every stdout line JSON-RPC, neither an `.mp4` nor a
/// `.part` left, the job's record ended cancelled, the self-test child's pass in the log. Returns how long it took to
/// exit.
fn term_export(t: &Tmp, k: &mut Kid, h: &str) -> Duration {
    // SAFETY: a signal to our own child, which has not been reaped (it is still running).
    assert_eq!(unsafe { libc::kill(k.c.id() as i32, libc::SIGTERM) }, 0);
    let t0 = Instant::now();
    assert!(k.wait().success(), "{}", k.stderr());
    let took = t0.elapsed();
    assert!(took < Duration::from_secs(15), "{took:?}");
    std::thread::sleep(Duration::from_millis(200));
    for l in k.all.lock().unwrap().iter() {
        let v: Value =
            serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON on stdout: {l:?}: {e}"));
        assert_eq!(v["jsonrpc"], "2.0", "{l}");
    }
    let d = kept(t, k);
    no_video(&d);
    assert_eq!(record(&d, h)["end"]["state"], "cancelled");
    let log = std::fs::read_to_string(d.join("mcp.log")).unwrap();
    assert!(log.contains("gpu-check: passed in"), "{log}");
    took
}

/// SIGTERM at ≥ 10 frames of an export.
#[test]
fn gpu_sigterm() {
    let _g = gpu();
    let t = Tmp::new("gterm");
    let src = synth(&t);
    let mut k = gpu_kid(&t, src.parent().unwrap(), "", &warm());
    let (v, _) = k.render("export", long(&src), 60);
    let h = job(&v);
    k.until(&h, 60, frames_at(10));
    term_export(&t, &mut k, &h);
}

/// SIGTERM in the first RAFT compile (burn's cache cold), work the lane can't cancel (design §7.2): sent once cubecl's
/// database is there (its first kernel built; the compile then runs ≈ 12 s more here). The end waits `RENDERS` for
/// the run, settles it away (its log line: still compiling) and `_exit`s, the lane still in the driver; otherwise as
/// at a frame.
#[test]
fn gpu_sigterm_compile() {
    let _g = gpu();
    let t = Tmp::new("gtermc");
    let src = synth(&t);
    let run = t.d.join("run");
    private(&run);
    let mut k = gpu_kid(&t, src.parent().unwrap(), "", &run);
    let a = json!({ "path": src.to_str().unwrap(), "preset": "pano4k" });
    let (v, _) = k.render("export", a, 60);
    let h = job(&v);
    let cc = run.join("x5desk-cubecl");
    let db = || (names(&cc).iter()).any(|n| cc.join(n).join("default.db").exists());
    let v = k.until(&h, 60, |v| v["phase"] == "compiling" && db());
    assert_eq!(v["frames_done"], 0, "{v}");
    let took = term_export(&t, &mut k, &h);
    let log = std::fs::read_to_string(kept(&t, &k).join("mcp.log")).unwrap();
    let still = format!("export {h} still in compiling at the host's end");
    assert!(
        took >= mcp::host::RENDERS && log.contains(&still),
        "ended in {took:?}: not in the compile (over, or not begun)\n{log}"
    );
    assert!(
        k.stderr()
            .contains("ending with a GPU thread in the driver"),
        "{}",
        k.stderr()
    );
}

/// A proxied client's export on the hosting client's host: the hosting client's stdin ends at ≥ 10 frames, and the
/// export ends with its host, cancelled (no `.part`; the record in the host's dir). The proxied client then hosts
/// itself, with dirs of its own (`mcp-sim-dir`): its handle names a job of a host that has ended.
#[test]
fn gpu_proxied_end() {
    let _g = gpu();
    let t = Tmp::new("gproxy");
    let src = synth(&t);
    let run = warm();
    let mut a = gpu_kid(&t, src.parent().unwrap(), "", &run);
    let mut b = gpu_kid(&t, src.parent().unwrap(), "", &run);
    let s = b.ok("status", json!({}));
    assert!(
        s["client"].as_str().unwrap().starts_with("endpoint "),
        "proxied: {s}"
    );
    let (v, _) = b.render("export", long(&src), 60);
    let h = job(&v);
    b.until(&h, 60, frames_at(10));
    let t0 = Instant::now();
    assert!(a.close().success(), "{}", a.stderr());
    assert!(t0.elapsed() < Duration::from_secs(15), "{:?}", t0.elapsed());
    nothing(&v);
    let d = kept(&t, &a);
    no_video(&d);
    let r = record(&d, &h);
    assert_eq!(r["end"]["state"], "cancelled", "{r}");
    assert!(
        r["end"]["err"]
            .as_str()
            .unwrap()
            .starts_with("its host ended"),
        "{r}"
    );
    let end = Instant::now() + Duration::from_secs(20);
    let r = loop {
        let r = b.call("job_status", json!({ "job": h }));
        match r.get("error") {
            Some(e) if Instant::now() < end => assert_eq!(e["code"], -32603, "{r}"),
            _ => break r,
        }
    };
    let e = r["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(r["result"]["isError"], true, "{r}");
    assert!(e.contains("its host has ended"), "{e}");
    assert!(b.close().success(), "{}", b.stderr());
}

/// The milestone on real footage, `X5_SAMPLES`: its v635.insv, else its first .insv, read in place (a link to it
/// works; its real folder is the read root). The catalog without a GPU, the file's info; a RAFT 360 still at 3840 and
/// a flat one at 1920 × 1080, each the size asked; 2-5 s exported flat 1280 × 720 H.264 without RAFT and 360 as
/// pano4k with it, followed by job_wait to done, their samples the job's frames; a whole-clip pano57 export (RAFT,
/// Main 10) cancelled at ≥ 30 frames: cancelled within 3 s, no file, no `.part`; stdin's end with an export running.
/// Timings on stdout (`--nocapture`); the files copied to `X5_EXPORT_OUT` when set.
#[test]
fn x5_samples() {
    let Some(d) = std::env::var_os("X5_SAMPLES").map(PathBuf::from) else {
        println!("X5_SAMPLES unset: skipped");
        return;
    };
    let v635 = d.join("v635.insv");
    let src = match v635.exists() {
        true => v635,
        false => (names(&d).iter())
            .find(|n| n.ends_with(".insv"))
            .map(|n| d.join(n))
            .expect("an .insv in X5_SAMPLES"),
    };
    let real = std::fs::canonicalize(&src).unwrap();
    let _g = gpu();
    let t = Tmp::new("x5");
    let mut k = gpu_kid(&t, real.parent().unwrap(), "", &warm());
    let p = src.to_str().unwrap();
    let out = std::env::var_os("X5_EXPORT_OUT").map(PathBuf::from);
    let keep = |v: &Value| {
        let (Some(o), Some(f)) = (&out, v["path"].as_str()) else {
            return;
        };
        std::fs::create_dir_all(o).unwrap();
        let f = Path::new(f);
        std::fs::copy(f, o.join(f.file_name().unwrap())).unwrap();
        println!("  copied to {}", o.join(f.file_name().unwrap()).display());
    };
    let c = k.ok("catalog", json!({}));
    assert_eq!(c["gpu"]["state"], "off", "the catalog starts no GPU");
    assert!(
        (c["presets"].as_array().unwrap().iter()).any(|x| x["id"] == "pano57"),
        "{c}"
    );
    let f = k.ok("file_info", json!({ "path": p }));
    assert_eq!(f["trailer"]["camera"], "Insta360 X5", "{f}");
    let vt = &f["mp4"]["tracks"][0];
    let rate = vt["samples"].as_f64().unwrap() / vt["duration_s"].as_f64().unwrap();
    println!(
        "{}: {:.2} s at {rate:.3} frames/s",
        real.display(),
        vt["duration_s"]
    );
    for (a, size) in [
        (
            json!({ "path": p, "output": 360, "width": 3840, "time_s": 2 }),
            [3840, 1920],
        ),
        (
            json!({ "path": p, "output": "flat", "width": 1920, "height": 1080, "time_s": 2 }),
            [1920, 1080],
        ),
    ] {
        let t0 = Instant::now();
        let (v, r) = k.render("render_still", a, 120);
        println!(
            "still {}: {:.2} s (the job {} s) → {}",
            bare(&v["settings"]["output"]),
            t0.elapsed().as_secs_f64(),
            v["elapsed_s"],
            bare(&v["path"])
        );
        assert_eq!(
            (&v["state"], &v["settings"]["raft"]),
            (&json!("done"), &json!(true)),
            "{v}"
        );
        assert_eq!(v["source"], json!([real.to_str().unwrap()]), "{v}");
        let b = std::fs::read(v["path"].as_str().unwrap()).unwrap();
        assert_eq!(jpeg_size(&b), size);
        assert_eq!(b.windows(5).any(|w| w == b"GPano"), size[0] == 2 * size[1]);
        let pv = unb64(r["result"]["content"][1]["data"].as_str().unwrap());
        let [w, h] = jpeg_size(&pv);
        assert!(
            w.max(h) == 1568 && pv.len() < 1_000_000,
            "{w} × {h}, {} bytes",
            pv.len()
        );
        keep(&v);
    }
    for a in [
        json!({
            "path": p, "output": "flat", "width": 1280, "height": 720, "codec": "h264", "raft": false,
            "start_s": 2, "end_s": 5,
        }),
        json!({ "path": p, "preset": "pano4k", "start_s": 2, "end_s": 5 }),
    ] {
        let (v, _) = k.render("export", a, 60);
        let n = v["frames_total"].as_u64().unwrap();
        assert!((n as f64 - 3.0 * rate).abs() <= 1.0, "{v}");
        let t0 = Instant::now();
        let (v, notes) = k.wait_job(&job(&v), 300);
        let s = &v["settings"];
        println!(
            "export {} {} × {} {} RAFT {}: {n} frames in {:.1} s ({} frames/s, {} bytes, {} notifications) → {}",
            bare(&s["output"]),
            s["width"],
            s["height"],
            bare(&s["codec"]),
            s["raft"],
            t0.elapsed().as_secs_f64(),
            v["fps"],
            v["bytes"],
            notes.len(),
            bare(&v["path"])
        );
        assert_eq!(
            (&v["state"], &v["frames_done"]),
            (&json!("done"), &json!(n)),
            "{v}"
        );
        assert_eq!(samples(Path::new(v["path"].as_str().unwrap())), n as usize);
        assert!(!notes.is_empty());
        keep(&v);
    }
    let (v, _) = k.render("export", json!({ "path": p, "preset": "pano57" }), 60);
    assert_eq!(
        (&v["settings"]["raft"], &v["settings"]["ten_bit"]),
        (&json!(true), &json!(true)),
        "{v}"
    );
    let h = job(&v);
    let r = k.until(&h, 120, frames_at(30));
    let t0 = Instant::now();
    let c = k.ok("job_cancel", json!({ "job": h }));
    println!(
        "pano57 cancelled at {} frames ({} frames/s) in {:.2} s",
        r["frames_done"],
        r["fps"],
        t0.elapsed().as_secs_f64()
    );
    assert_eq!(c["state"], "cancelled", "{c}");
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    nothing(&c);
    let a = json!({ "path": p, "preset": "flat1080", "start_s": 10, "end_s": 40 });
    let (v, _) = k.render("export", a, 60);
    let h = job(&v);
    k.until(&h, 120, frames_at(10));
    let t0 = Instant::now();
    assert!(k.close().success(), "{}", k.stderr());
    println!(
        "the host's end mid-export: {:.2} s",
        t0.elapsed().as_secs_f64()
    );
    assert!(t0.elapsed() < Duration::from_secs(15), "{:?}", t0.elapsed());
    nothing(&v);
    let d = kept(&t, &k);
    assert_eq!(record(&d, &h)["end"]["state"], "cancelled");
    let log = std::fs::read_to_string(d.join("mcp.log")).unwrap();
    println!(
        "{}",
        log.lines()
            .filter(|l| !l.contains(" stdio "))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
