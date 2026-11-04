//! `x5desk mcp …`, which the app hands its arguments after `mcp` before anything of its own runs or prints:
//! - `x5desk mcp [--sim] [--tiers a,b] [--allow-dir DIR]…` serves MCP on stdio: the stdout guard first (`out`), then
//!   the front (`front`) electing a host on the endpoint (`endpoint`). `--tiers` only narrows, and only this
//!   process's client (`front`); `--allow-dir` adds read roots of the host (each must exist). `--sim`: the simulated
//!   camera in temp dirs, no endpoint unless `X5DESK_MCP_ENDPOINT` names one, the tiers from `X5DESK_MCP_TIERS` (else
//!   all but `delete`). `X5DESK_TEST` likewise uses no endpoint, and without `--sim` keeps the agent folder and log in
//!   a temp dir (`Opts::dir`); its token `mcp-print` prints a line through `println!` before serving, which the guard
//!   sends to stderr.
//! - `x5desk mcp enable --tiers read,local[,camera][,delete]` writes `<config>/x5desk/mcp` (the set replaced) and
//!   prints what is enabled; `x5desk mcp disable` turns every tier off.
//! - `x5desk mcp --print-config` prints the Claude Code command and Claude Desktop's `mcpServers` entry for this
//!   executable; refused from a disk image (`/Volumes/`) or a translocated copy (`/AppTranslocation/`), gone later.
//!
//! Wrong arguments: the usage on stderr, exit 2. A server that could not start exits 1, one whose client went (stdin's
//! end, once the open requests are answered; SIGTERM, SIGINT and SIGHUP unless started with them ignored (`nohup`);
//! Windows: Ctrl-C, Ctrl-Break, its console closed, logoff, shutdown) 0.

use crate::{
    Opts, Tiers, config,
    front::{Elect, Front, Lines},
    out,
};
use serde_json::json;
use std::{
    ffi::OsString,
    fs::File,
    io::{Read, Write},
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
<tiers>: a comma list of read, local, camera, delete (delete only by its name); when serving, --tiers only narrows
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

/// Runs `x5desk mcp` with the arguments after `mcp`; returns the exit code.
pub fn main(args: Vec<OsString>) -> i32 {
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
        Ok(Cmd::Serve { sim, tiers, allow }) => serve(sim, tiers, allow),
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

fn serve(sim: bool, narrow: Option<Tiers>, allow: Vec<PathBuf>) -> i32 {
    // stdout is the protocol's from here on: whatever else prints goes to stderr
    let out = match out::guard() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("x5desk mcp: {e}");
            return 1;
        }
    };
    let test = std::env::var("X5DESK_TEST").ok();
    if test
        .as_deref()
        .is_some_and(|t| t.split(',').any(|w| w.trim() == "mcp-print"))
    {
        println!("x5desk mcp: X5DESK_TEST=mcp-print: a line printed on stdout");
    }
    // an --allow-dir is checked now, also by a process that only proxies (it may host later)
    for d in &allow {
        if let Err(e) = std::fs::canonicalize(d) {
            eprintln!("x5desk mcp: --allow-dir {}: {e}", d.display());
            return 1;
        }
    }
    let dir = (test.is_some() && !sim)
        .then(|| std::env::temp_dir().join(format!("x5desk-mcp-{}", std::process::id())));
    if let Some(d) = &dir {
        eprintln!(
            "x5desk mcp: X5DESK_TEST: the agent folder and the log in {}",
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
        sim: sim.then(Default::default),
        narrow,
        allow,
        dir,
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
        ] {
            assert!(p(bad).is_err(), "{bad}");
        }
        assert!(p("--tiers all").unwrap_err().contains("unknown tier"));
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
