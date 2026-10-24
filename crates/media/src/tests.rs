//! Test helpers. Footage comes from `X5_SAMPLES`; generated fixtures go to `X5_FIXTURES` (default: `out()/fixtures`).

use std::{path::PathBuf, process::Command};

/// The sample directory, or None (with a note) when `X5_SAMPLES` is unset.
pub fn samples() -> Option<PathBuf> {
    let d = std::env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        eprintln!("X5_SAMPLES not set: skipping");
    }
    d
}

/// Large test outputs: `out/` next to the build (`<target dir>/out`), on disk rather than in the temp dir.
pub fn out() -> PathBuf {
    let e = std::env::current_exe().unwrap();
    let d = e.ancestors().nth(3).unwrap().join("out");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Generated fixtures: `X5_FIXTURES`, else `out()/fixtures` (on disk: some are tens of MB).
pub fn fixtures() -> PathBuf {
    let d = std::env::var_os("X5_FIXTURES").map_or_else(|| out().join("fixtures"), PathBuf::from);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Runs a test tool (ffprobe / ffmpeg) and returns stdout; panics with stderr on failure.
pub fn tool(cmd: &str, args: &[&str]) -> String {
    let o = Command::new(cmd)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("{cmd}: {e}"));
    assert!(
        o.status.success(),
        "{cmd} {args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).unwrap()
}

pub type Pkt = (i64, i64, u32, u32, u64, bool);

/// ffprobe's packets per stream: (pts, dts, duration, size, pos, key).
pub fn probe(path: &str) -> Vec<Vec<Pkt>> {
    let out = tool(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "packet=stream_index,pts,dts,duration,size,pos,flags",
            "-of",
            "csv=p=0",
            path,
        ],
    );
    let mut v: Vec<Vec<_>> = vec![];
    for l in out.lines() {
        let f: Vec<_> = l.split(',').collect();
        let i: usize = f[0].parse().unwrap();
        v.resize_with(v.len().max(i + 1), Vec::new);
        let n = |k: usize| f[k].parse::<i64>().unwrap();
        v[i].push((
            n(1),
            n(2),
            n(3) as u32,
            n(4) as u32,
            n(5) as u64,
            f[6].contains('K'),
        ));
    }
    v
}
