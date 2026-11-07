//! `x5desk mcp …`, which the app hands its arguments after `mcp` before anything of its own runs or prints:
//! - `x5desk mcp [--sim] [--tiers a,b] [--allow-dir DIR]…` serves MCP on stdio: the stdout guard first (`out`), then
//!   the front (`front`) electing a host on the endpoint (`endpoint`). `--tiers` only narrows, and only this
//!   process's client (`front`); `--allow-dir` adds read roots of the host (each must exist). `--sim`: the simulated
//!   camera in temp dirs, no endpoint unless `X5DESK_MCP_ENDPOINT` names one, the tiers from `X5DESK_MCP_TIERS` (else
//!   all but `delete` and `untested`). `X5DESK_TEST` likewise uses no endpoint, and without `--sim` keeps the agent
//!   folder and log in a temp dir (`Opts::dir`: a real camera, so `X5DESK_MCP_TIERS` gives at most `read,local`); its
//!   token `mcp-print` prints a line through `println!` before serving, which the guard sends to stderr; with
//!   `--sim`, its token `mcp-sim-dir` keeps the simulated camera's dirs in `$TMPDIR/x5desk-mcp-<pid>` (process tests
//!   read the agent folder after the end).
//! - `x5desk mcp enable --tiers read,local[,camera][,delete][,untested]` writes `<config>/x5desk/mcp` (the set
//!   replaced) and prints what is enabled; `x5desk mcp disable` turns every tier off.
//! - `x5desk mcp --print-config` prints the Claude Code command and Claude Desktop's `mcpServers` entry for this
//!   executable; refused from a disk image (`/Volumes/`) or a translocated copy (`/AppTranslocation/`), gone later.
//! - `x5desk mcp gpu-check` (not in the usage): the GPU self-test child a host's backend runs before it opens the
//!   device itself ([`gpu_check`]'s one line on stdout). It ends at stdin's EOF (its parent gone) and after 25 s.
//!
//! Wrong arguments: the usage on stderr, exit 2. A server that could not start exits 1, one whose client went (stdin's
//! end, once the open requests are answered; SIGTERM, SIGINT and SIGHUP unless started with them ignored (`nohup`);
//! Windows: Ctrl-C, Ctrl-Break, its console closed, logoff, shutdown) 0.

use crate::{
    Opts, Tiers, config,
    front::{Elect, Front, Lines},
    out,
    render::{Backend, GpuInfo, NO_GPU, panicked},
    sim,
};
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    fs::File,
    io::{Read, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const USAGE: &str = "\
usage: x5desk mcp [--sim] [--tiers <tiers>] [--allow-dir <dir>]...  serve MCP on stdin and stdout
       x5desk mcp enable --tiers <tiers>  enable tiers for agents (replaces the enabled set)
       x5desk mcp disable                 turn every tier off
       x5desk mcp --print-config          how to add x5desk to Claude Code and Claude Desktop
<tiers>: a comma list of read, local, camera, delete, untested (delete and untested only by their names); when
         serving, --tiers only narrows. untested: modes and settings x5desk has not run on an X5
";

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Cmd {
    Serve {
        sim: bool,
        tiers: Option<Tiers>,
        allow: Vec<PathBuf>,
    },
    Enable(Tiers),
    Disable,
    Print,
    Help,
    GpuCheck,
}

/// The value of flag `f` at `a[i]` (`f=v`, or `f v`) and how many arguments it took; none: not that flag.
fn value(a: &[OsString], i: usize, f: &str) -> Option<Result<(OsString, usize), String>> {
    let x = a[i].to_str()?;
    if x == f {
        return Some(
            a.get(i + 1)
                .map(|v| (v.clone(), 2))
                .ok_or(format!("{f} needs a value")),
        );
    }
    let v = x.strip_prefix(f)?.strip_prefix('=')?;
    Some(Ok((v.into(), 1)))
}

fn tiers(v: &OsString) -> Result<Tiers, String> {
    Tiers::parse(v.to_str().ok_or("--tiers: not text")?)
}

fn parse(a: &[OsString]) -> Result<Cmd, String> {
    let word = |i: usize| a.get(i).and_then(|x| x.to_str());
    match (word(0), a.len()) {
        (Some("enable"), _) => {
            return match a.get(1).and_then(|_| value(a, 1, "--tiers")) {
                Some(Ok((v, n))) if 1 + n == a.len() => Ok(Cmd::Enable(tiers(&v)?)),
                Some(Err(e)) => Err(e),
                _ => Err("enable takes --tiers <tiers>".into()),
            };
        }
        (Some("disable"), 1) => return Ok(Cmd::Disable),
        (Some("--print-config"), 1) => return Ok(Cmd::Print),
        (Some("-h" | "--help" | "help"), 1) => return Ok(Cmd::Help),
        (Some("gpu-check"), 1) => return Ok(Cmd::GpuCheck),
        _ => {}
    }
    let (mut sim, mut t, mut allow, mut i) = (false, None::<Tiers>, vec![], 0);
    while i < a.len() {
        if word(i) == Some("--sim") {
            (sim, i) = (true, i + 1);
            continue;
        }
        if let Some(r) = value(a, i, "--tiers") {
            let (v, n) = r?;
            let x = tiers(&v)?;
            (t, i) = (Some(t.map_or(x, |o| o.and(x))), i + n);
            continue;
        }
        if let Some(r) = value(a, i, "--allow-dir") {
            let (v, n) = r?;
            allow.push(PathBuf::from(v));
            i += n;
            continue;
        }
        return Err(format!("unknown argument {:?}", a[i]));
    }
    Ok(Cmd::Serve {
        sim,
        tiers: t,
        allow,
    })
}

/// Runs `x5desk mcp` with the arguments after `mcp`; returns the exit code. `b`: the GPU tools' backend (the app's
/// engine; none: no GPU tools).
pub fn main(args: Vec<OsString>, b: Backend) -> i32 {
    match parse(&args) {
        Ok(Cmd::Help) => {
            print!("{USAGE}");
            0
        }
        Ok(Cmd::Enable(t)) => match config::enable(t) {
            Ok((p, t)) => {
                println!(
                    "x5desk mcp: enabled: {t} ({}); running servers follow within 2 s",
                    p.display()
                );
                0
            }
            Err(e) => {
                eprintln!("x5desk mcp: {e}");
                1
            }
        },
        Ok(Cmd::Disable) => match config::disable() {
            Ok(p) => {
                println!("x5desk mcp: every tier off ({})", p.display());
                0
            }
            Err(e) => {
                eprintln!("x5desk mcp: {e}");
                1
            }
        },
        Ok(Cmd::Print) => print_config(),
        Ok(Cmd::GpuCheck) => gpu_cmd(&b),
        Ok(Cmd::Serve { sim, tiers, allow }) => serve(sim, tiers, allow, b),
        Err(e) => {
            eprint!("x5desk mcp: {e}\n{USAGE}");
            2
        }
    }
}

/// Why a client config naming `exe` would break once the disk image is ejected or the app moved.
fn refused(exe: &Path) -> Option<String> {
    let s = exe.to_string_lossy();
    if s.contains("/AppTranslocation/") {
        Some(format!(
            "{s}: macOS runs this x5desk from a temporary copy (App Translocation, a downloaded app not moved yet): \
             move x5desk.app to /Applications in Finder, open it from there once, then run \
             /Applications/x5desk.app/Contents/MacOS/x5desk mcp --print-config"
        ))
    } else if s.starts_with("/Volumes/") {
        Some(format!(
            "{s}: x5desk runs from a disk image or another volume: copy x5desk.app to /Applications, then run \
             /Applications/x5desk.app/Contents/MacOS/x5desk mcp --print-config"
        ))
    } else {
        None
    }
}

/// `s` as one shell word: POSIX quoting, on Windows double quotes.
fn quote(s: &str) -> String {
    let plain = !s.is_empty()
        && s.bytes().all(|b| {
            b.is_ascii_alphanumeric() || b"/._-+:@%,=".contains(&b) || (cfg!(windows) && b == b'\\')
        });
    match (plain, cfg!(windows)) {
        (true, _) => s.into(),
        (false, true) => format!("\"{s}\""),
        (false, false) => format!("'{}'", s.replace('\'', r"'\''")),
    }
}

/// The client configs for the executable at `exe`.
fn configs(exe: &str) -> String {
    let j = json!({ "mcpServers": { "x5desk": { "command": exe, "args": ["mcp"] } } });
    format!(
        "# Claude Code\nclaude mcp add --transport stdio x5desk -- {} mcp\n\n\
         # Claude Desktop: in claude_desktop_config.json\n{}\n",
        quote(exe),
        serde_json::to_string_pretty(&j).unwrap_or_default()
    )
}

fn print_config() -> i32 {
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("x5desk mcp: where is this executable? {e}");
            return 1;
        }
    };
    if let Some(why) = refused(&exe) {
        eprintln!("x5desk mcp: {why}");
        return 1;
    }
    let Some(s) = exe.to_str() else {
        eprintln!("x5desk mcp: {}: not a UTF-8 path", exe.display());
        return 1;
    };
    print!("{}", configs(s));
    0
}

/// The self-test's outcome: the device's info, or the step that failed (None: before any) and why.
fn selftest(b: &Backend) -> Result<GpuInfo, (Option<u8>, String)> {
    let r = b.render.as_ref().ok_or((None, NO_GPU.to_string()))?;
    match catch_unwind(AssertUnwindSafe(|| r.selftest())) {
        Ok(r) => r.map_err(|(s, w)| (Some(s), w)),
        Err(p) => Err((None, format!("the self-test panicked: {}", panicked(&*p)))),
    }
}

fn line(r: &Result<GpuInfo, (Option<u8>, String)>) -> String {
    match r {
        Ok(i) => json!({ "v": 1, "ok": true, "info": i }),
        Err((step, why)) => json!({ "v": 1, "ok": false, "step": step, "why": why }),
    }
    .to_string()
}

/// What `x5desk mcp gpu-check` prints: `{"v":1,"ok":true,"info":GpuInfo}` or `{"v":1,"ok":false,"step":…,"why":…}`
/// on one line. A panic in the self-test is a failure without a step.
pub fn gpu_check(b: &Backend) -> String {
    line(&selftest(b))
}

/// [`gpu_check`]'s line read back, as the parent gets it from the child's stdout.
pub fn gpu_checked(l: &str) -> Result<GpuInfo, (Option<u8>, String)> {
    let bad = |e: String| (None, format!("the GPU check's answer: {e}"));
    let v: Value = serde_json::from_str(l.trim()).map_err(|e| bad(e.to_string()))?;
    if v["v"] != 1 {
        return Err(bad(format!("version {}", v["v"])));
    }
    match v["ok"].as_bool() {
        Some(true) => serde_json::from_value(v["info"].clone()).map_err(|e| bad(e.to_string())),
        Some(false) => Err((
            v["step"].as_u64().map(|s| s as u8),
            v["why"].as_str().unwrap_or("no reason given").to_string(),
        )),
        None => Err(bad("no ok".into())),
    }
}

/// Ends the process now: no atexit, no driver teardown (an ICD's destructors may hang or crash).
fn quit(code: i32) -> ! {
    #[cfg(unix)]
    // SAFETY: _exit(2) only ends the process; what had to be written was flushed, stderr is unbuffered.
    unsafe {
        libc::_exit(code)
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

/// `x5desk mcp gpu-check`: stdout guarded (a driver's prints go to stderr), the self-test, its line, gone. It also
/// ends at stdin's EOF (the parent gone: a SIGKILLed host takes it along) and after 25 s (a driver that hangs).
fn gpu_cmd(b: &Backend) -> i32 {
    let mut out = match out::guard() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("x5desk mcp gpu-check: {e}");
            return 1;
        }
    };
    fn gone(why: &str) -> ! {
        eprintln!("x5desk mcp gpu-check: {why}");
        quit(1)
    }
    let _ = std::thread::Builder::new()
        .name("x5-gpu-stdin".into())
        .spawn(move || {
            let (mut i, mut b) = (std::io::stdin(), [0; 256]);
            loop {
                match i.read(&mut b) {
                    Ok(0) => gone("stdin closed: the parent is gone"),
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => gone(&format!("stdin: {e}")),
                }
            }
        });
    let _ = std::thread::Builder::new()
        .name("x5-gpu-limit".into())
        .spawn(move || {
            std::thread::sleep(Duration::from_secs(25));
            gone("no answer within 25 s")
        });
    let r = selftest(b);
    if let Err(e) = writeln!(out, "{}", line(&r)).and_then(|_| out.flush()) {
        gone(&format!("stdout: {e}"))
    }
    quit(if r.is_ok() { 0 } else { 1 })
}

/// The `gpu-check` child's executable: this one (Linux by /proc/self/exe, which still names it once the file is
/// replaced or gone).
fn exe() -> Option<PathBuf> {
    match cfg!(target_os = "linux") {
        true => Some("/proc/self/exe".into()),
        false => std::env::current_exe().ok(),
    }
}

/// Whether the comma list `X5DESK_TEST` has token `t`.
fn token(test: Option<&str>, t: &str) -> bool {
    test.is_some_and(|x| x.split(',').any(|w| w.trim() == t))
}

/// stdin's lines, read on a thread of their own (a blocking read nobody can cancel; it goes with the process).
fn reader() -> mpsc::UnboundedReceiver<Vec<u8>> {
    let (tx, rx) = mpsc::unbounded_channel();
    let _ = std::thread::Builder::new()
        .name("x5-mcp-stdin".into())
        .spawn(move || {
            let (mut ls, mut b, mut i) = (Lines::default(), vec![0; 1 << 16], std::io::stdin());
            loop {
                match i.read(&mut b) {
                    Ok(0) => return,
                    Ok(n) => {
                        for l in ls.push(&b[..n]) {
                            if tx.send(l).is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(_) => return,
                }
            }
        });
    rx
}

/// The protocol's writer on a thread of its own: each line written whole and flushed. The second channel says it
/// ended (all written, or the client's end of stdout gone).
fn writer(
    mut f: File,
) -> std::io::Result<(
    std::sync::mpsc::Sender<Vec<u8>>,
    std::sync::mpsc::Receiver<()>,
)> {
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let (dtx, drx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("x5-mcp-stdout".into())
        .spawn(move || {
            for l in rx {
                if f.write_all(&l).and_then(|_| f.flush()).is_err() {
                    break;
                }
            }
            let _ = dtx.send(());
        })?;
    Ok((tx, drx))
}

/// Whether signal `s` is ignored now (`SIG_IGN` inherited, as under `nohup`).
#[cfg(unix)]
fn ignored(s: libc::c_int) -> bool {
    // SAFETY: a zeroed sigaction is a valid value; with a null new action sigaction(2) only reads the current one into
    // `old`, which is ours.
    unsafe {
        let mut old: libc::sigaction = std::mem::zeroed();
        libc::sigaction(s, std::ptr::null(), &mut old) == 0 && old.sa_sigaction == libc::SIG_IGN
    }
}

/// SIGCHLD back to its default when started with it ignored: Linux then reaps our children itself, and waiting for
/// the `gpu-check` child fails (ECHILD). Before any thread of ours runs.
#[cfg(unix)]
fn reap_own() {
    if ignored(libc::SIGCHLD) {
        // SAFETY: SIG_DFL is a valid disposition; no handler of ours is replaced
        unsafe { libc::signal(libc::SIGCHLD, libc::SIG_DFL) };
    }
}

/// Cancels `stop` at SIGTERM, SIGINT or SIGHUP (its terminal closed); SIGINT and SIGHUP not when started with them
/// ignored (`nohup`, a background job of a shell without job control), which tokio's handler would undo. Windows:
/// Ctrl-C, Ctrl-Break, its console closed, the user logging off, the system shutting down (tokio holds the console
/// handler meanwhile, so the host gets the few seconds Windows allows).
fn signals(stop: CancellationToken) {
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{Signal, SignalKind, signal};
            let on = |k: SignalKind, n| (!ignored(n)).then(|| signal(k).ok()).flatten();
            let (mut t, mut i, mut h) = (
                signal(SignalKind::terminate()).ok(),
                on(SignalKind::interrupt(), libc::SIGINT),
                on(SignalKind::hangup(), libc::SIGHUP),
            );
            async fn recv(s: &mut Option<Signal>) {
                match s {
                    Some(s) => drop(s.recv().await),
                    None => std::future::pending().await,
                }
            }
            tokio::select! {
                _ = recv(&mut t) => {}
                _ = recv(&mut i) => {}
                _ = recv(&mut h) => {}
            }
        }
        #[cfg(windows)]
        {
            use tokio::signal::windows::{ctrl_break, ctrl_close, ctrl_logoff, ctrl_shutdown};
            let (Ok(mut b), Ok(mut c), Ok(mut l), Ok(mut s)) =
                (ctrl_break(), ctrl_close(), ctrl_logoff(), ctrl_shutdown())
            else {
                return;
            };
            tokio::select! {
                r = tokio::signal::ctrl_c() => if r.is_err() { return },
                _ = b.recv() => {}
                _ = c.recv() => {}
                _ = l.recv() => {}
                _ = s.recv() => {}
            }
        }
        // first: a closed terminal fails the write
        stop.cancel();
        let _ = writeln!(std::io::stderr(), "x5desk mcp: stopping");
    });
}

fn serve(sim: bool, narrow: Option<Tiers>, allow: Vec<PathBuf>, backend: Backend) -> i32 {
    #[cfg(unix)]
    reap_own();
    // stdout is the protocol's from here on: whatever else prints goes to stderr
    let out = match out::guard() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("x5desk mcp: {e}");
            return 1;
        }
    };
    let test = std::env::var("X5DESK_TEST").ok();
    if token(test.as_deref(), "mcp-print") {
        println!("x5desk mcp: X5DESK_TEST=mcp-print: a line printed on stdout");
    }
    // an --allow-dir is checked now, also by a process that only proxies (it may host later)
    for d in &allow {
        if let Err(e) = std::fs::canonicalize(d) {
            eprintln!("x5desk mcp: --allow-dir {}: {e}", d.display());
            return 1;
        }
    }
    let mine = std::env::temp_dir().join(format!("x5desk-mcp-{}", std::process::id()));
    let dir = (test.is_some() && !sim).then(|| mine.clone());
    if let Some(d) = &dir {
        eprintln!(
            "x5desk mcp: X5DESK_TEST: the agent folder and the log in {}",
            d.display()
        );
    }
    // the simulated camera's dirs kept, so a process test reads them after the end
    let keep = (sim && token(test.as_deref(), "mcp-sim-dir")).then_some(mine);
    if let Some(d) = &keep {
        eprintln!(
            "x5desk mcp: X5DESK_TEST=mcp-sim-dir: the simulated camera's dirs in {}",
            d.display()
        );
    }
    let ep = endpoint(sim || test.is_some());
    if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        eprintln!(
            "x5desk mcp: serving MCP on stdin and stdout, for MCP clients (`x5desk mcp --print-config`)"
        );
    }
    let rt = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("x5-mcp")
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("x5desk mcp: {e}");
            return 1;
        }
    };
    let o = Opts {
        sim: sim.then(|| sim::Opts {
            dir: keep,
            ..Default::default()
        }),
        narrow,
        allow,
        dir,
        backend,
        gpu_child: exe(),
        ..Default::default()
    };
    let code = rt.block_on(run(o, ep, out));
    // the host's tasks dropped (a `--sim` host's temp dir goes with them); stdin's thread goes with the process
    rt.shutdown_timeout(Duration::from_secs(2));
    code
}

/// The endpoint, or none (tests, or an error said on stderr: this process then hosts alone).
fn endpoint(test: bool) -> Option<PathBuf> {
    crate::endpoint::path(test).unwrap_or_else(|e| {
        eprintln!("x5desk mcp: endpoint: {e}; hosting without it");
        None
    })
}

async fn run(o: Opts, ep: Option<PathBuf>, out: File) -> i32 {
    let narrow = o.narrow;
    let (otx, done) = match writer(out) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("x5desk mcp: {e}");
            return 1;
        }
    };
    let mut stdin = reader();
    let stop = CancellationToken::new();
    signals(stop.clone());
    let mut f = Front::new(Elect::new(o, ep), narrow);
    let r = f.run(&mut stdin, &otx, &stop).await;
    // the listener closed first, then the host: the camera released, downloads' `.part` kept
    f.end().await;
    drop(otx);
    let _ = done.recv_timeout(Duration::from_secs(2));
    match r {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("x5desk mcp: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Cmd, String> {
        parse(&s.split_whitespace().map(OsString::from).collect::<Vec<_>>())
    }

    #[test]
    fn args() {
        let t = |s: &str| Tiers::parse(s).unwrap();
        assert_eq!(
            p(""),
            Ok(Cmd::Serve {
                sim: false,
                tiers: None,
                allow: vec![]
            })
        );
        assert_eq!(
            p("--sim --tiers read,local --allow-dir /a --allow-dir=/b --tiers=read,camera"),
            Ok(Cmd::Serve {
                sim: true,
                tiers: Some(t("read")),
                allow: vec!["/a".into(), "/b".into()]
            }),
            "--tiers twice: both narrow"
        );
        assert_eq!(
            p("enable --tiers read,local"),
            Ok(Cmd::Enable(t("read,local")))
        );
        assert_eq!(p("enable --tiers=delete"), Ok(Cmd::Enable(t("delete"))));
        assert_eq!(p("disable"), Ok(Cmd::Disable));
        assert_eq!(p("--print-config"), Ok(Cmd::Print));
        assert_eq!(p("--help"), Ok(Cmd::Help));
        assert_eq!(p("gpu-check"), Ok(Cmd::GpuCheck));
        for bad in [
            "--bogus",
            "file.insv",
            "--tiers",
            "--tiers all",
            "--allow-dir",
            "enable",
            "enable read",
            "enable --tiers read extra",
            "disable now",
            "--print-config --sim",
            "--simx",
            "gpu-check --sim",
            "--sim gpu-check",
        ] {
            assert!(p(bad).is_err(), "{bad}");
        }
        assert!(p("--tiers all").unwrap_err().contains("unknown tier"));
    }

    /// The self-test child's line: one JSON line of version 1, read back as it was, whether it passed, failed at a
    /// step, panicked or had no backend.
    #[test]
    fn gpu_line() {
        use crate::render::{Backend, tests::Fake};
        use std::sync::Arc;
        let b = |f| Backend {
            render: Some(Arc::new(f)),
        };
        let l = gpu_check(&b(Fake::Ok));
        assert!(!l.contains('\n'), "{l}");
        let v: Value = serde_json::from_str(&l).unwrap();
        assert_eq!((&v["v"], &v["ok"]), (&json!(1), &json!(true)), "{l}");
        assert_eq!(v["info"]["adapter"], "Fake GPU");
        assert_eq!(
            gpu_checked(&format!("{l}\n")),
            Ok(crate::render::tests::info())
        );
        let l = gpu_check(&b(Fake::Fails));
        let v: Value = serde_json::from_str(&l).unwrap();
        assert_eq!(
            (&v["v"], &v["ok"], &v["step"]),
            (&json!(1), &json!(false), &json!(3)),
            "{l}"
        );
        let e = gpu_checked(&l).unwrap_err();
        assert_eq!(e.0, Some(3));
        assert!(e.1.contains("cannot decode"), "{e:?}");
        let e = gpu_checked(&gpu_check(&b(Fake::Panics))).unwrap_err();
        assert_eq!(e.0, None);
        assert!(e.1.contains("panicked: the driver went away"), "{e:?}");
        assert_eq!(
            gpu_checked(&gpu_check(&Backend::default())),
            Err((None, NO_GPU.to_string()))
        );
        for bad in [
            "",
            "x5desk: oops",
            r#"{"v":2,"ok":true}"#,
            r#"{"v":1}"#,
            r#"{"v":1,"ok":true}"#,
        ] {
            let e = gpu_checked(bad).unwrap_err();
            assert!(
                e.0.is_none() && e.1.starts_with("the GPU check's answer"),
                "{bad}: {e:?}"
            );
        }
    }

    /// A server started with SIGCHLD ignored (Linux then reaps its children itself) waits for its `gpu-check` child: in
    /// a child process of this test, whose disposition it changes.
    #[cfg(unix)]
    #[test]
    fn reaps() {
        if std::env::var_os("X5_MCP_REAP").is_some() {
            // SAFETY: SIG_IGN is a valid disposition, in a process that runs this test alone
            unsafe { libc::signal(libc::SIGCHLD, libc::SIG_IGN) };
            reap_own();
            let mut c = std::process::Command::new("/bin/sh")
                .args(["-c", "exit 3"])
                .spawn()
                .unwrap();
            assert_eq!(c.wait().unwrap().code(), Some(3));
            return;
        }
        let o = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "cli::tests::reaps",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("X5_MCP_REAP", "1")
            .output()
            .unwrap();
        let said = String::from_utf8_lossy(&o.stdout) + String::from_utf8_lossy(&o.stderr);
        assert!(o.status.success() && said.contains("1 passed"), "{said}");
    }

    #[test]
    fn tokens() {
        assert!(token(Some("mcp-print, mcp-sim-dir"), "mcp-sim-dir"));
        assert!(!token(Some("mcp-sim-dirs"), "mcp-sim-dir"));
        assert!(!token(None, "mcp-print"));
    }

    #[test]
    fn print_config() {
        assert!(
            refused(Path::new(
                "/Volumes/x5desk/x5desk.app/Contents/MacOS/x5desk"
            ))
            .is_some()
        );
        let tr = "/private/var/folders/x/T/AppTranslocation/ABC/d/x5desk.app/Contents/MacOS/x5desk";
        assert!(refused(Path::new(tr)).unwrap().contains("Translocation"));
        assert_eq!(
            refused(Path::new("/Applications/x5desk.app/Contents/MacOS/x5desk")),
            None
        );
        let exe = "/Users/a b/it's/x5desk";
        let c = configs(exe);
        if !cfg!(windows) {
            assert!(
                c.contains(r"x5desk -- '/Users/a b/it'\''s/x5desk' mcp"),
                "{c}"
            );
            assert!(configs("/usr/bin/x5desk").contains("x5desk -- /usr/bin/x5desk mcp"));
        }
        let j: serde_json::Value = serde_json::from_str(&c[c.find('{').unwrap()..]).unwrap();
        assert_eq!(
            j["mcpServers"]["x5desk"],
            json!({ "command": exe, "args": ["mcp"] })
        );
    }
}
