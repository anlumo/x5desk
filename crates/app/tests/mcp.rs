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
        let end = Instant::now() + Duration::from_secs(20);
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
                return v;
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
