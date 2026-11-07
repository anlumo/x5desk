//! The call log, `<cache>/x5desk/mcp.log` (0600 on unix): one scrubbed line per tool call (UTC time, client, tool,
//! an argument summary, ok or the error, ms), and the host's own events. At 1 MB it moves to `mcp.log.1` and starts
//! over. A failing write is said once on stderr and otherwise ignored.

use crate::scrub::Secrets;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

/// Where a log moves to `.1`.
pub const MAX: u64 = 1 << 20;

/// `<cache>/x5desk/mcp.log`.
pub fn file() -> Option<PathBuf> {
    cam::dirs::xdg("XDG_CACHE_HOME", ".cache").map(|d| d.join("x5desk").join("mcp.log"))
}

/// A host's log.
pub struct Log {
    path: Option<PathBuf>,
    s: Secrets,
    /// the open file and its length; whether a failure was said
    f: Mutex<(Option<(File, u64)>, bool)>,
}

/// `2026-11-04 12:34:56Z` for a time since the epoch.
pub fn utc(t: SystemTime) -> String {
    let s = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (d, r) = ((s / 86400) as i64, s % 86400);
    // civil from days (H. Hinnant)
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + (mon <= 2) as i64;
    format!(
        "{year:04}-{mon:02}-{day:02} {:02}:{:02}:{:02}Z",
        r / 3600,
        r % 3600 / 60,
        r % 60
    )
}

fn open(p: &Path) -> std::io::Result<(File, u64)> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?
    }
    let mut o = OpenOptions::new();
    o.create(true).append(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    let f = o.open(p)?;
    // one made before by someone else's umask
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    let n = f.metadata()?.len();
    Ok((f, n))
}

impl Log {
    /// A log at `path` (none: nothing is written) scrubbed of `s`.
    pub fn new(path: Option<PathBuf>, s: Secrets) -> Log {
        Log {
            path,
            s,
            f: Mutex::new((None, false)),
        }
    }

    /// Where it writes; none: nowhere.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Appends `line` (scrubbed, newlines flattened) after the time.
    pub fn line(&self, line: &str) {
        let Some(p) = &self.path else { return };
        let l = format!(
            "{} {}\n",
            utc(SystemTime::now()),
            self.s.scrub(&line.replace(['\n', '\r'], " "))
        );
        let mut g = self.f.lock().unwrap();
        let r = (|| {
            if g.0.as_ref().is_some_and(|f| f.1 + l.len() as u64 > MAX) {
                g.0 = None;
                let mut old = p.as_os_str().to_owned();
                old.push(".1");
                std::fs::rename(p, old)?;
            }
            if g.0.is_none() {
                g.0 = Some(open(p)?);
            }
            let f = g.0.as_mut().unwrap();
            f.0.write_all(l.as_bytes())?;
            f.1 += l.len() as u64;
            Ok::<_, std::io::Error>(())
        })();
        if let Err(e) = r {
            g.0 = None;
            if !std::mem::replace(&mut g.1, true) {
                eprintln!("x5desk mcp: {}: {e}", p.display());
            }
        }
    }

    /// One tool call: `client tool args ok|error: … ms`; `args` cut to 300 characters.
    pub fn call(&self, client: &str, tool: &str, args: &str, err: Option<&str>, ms: f64) {
        let a: String = match args.chars().count() > 300 {
            true => args.chars().take(300).chain("…".chars()).collect(),
            false => args.into(),
        };
        let r = err.map_or("ok".into(), |e| format!("error: {e}"));
        self.line(&format!("{client} {tool} {a} {r} {ms:.0} ms"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn times() {
        let at = |s: u64| utc(UNIX_EPOCH + Duration::from_secs(s));
        assert_eq!(at(0), "1970-01-01 00:00:00Z");
        assert_eq!(at(951_782_400), "2000-02-29 00:00:00Z");
        assert_eq!(at(1_793_806_496), "2026-11-04 15:34:56Z");
    }

    #[test]
    fn rotates() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let s = Secrets::default();
        s.add("hunter22");
        let l = Log::new(Some(d.join("mcp.log")), s);
        l.call(
            "stdio",
            "status",
            "{\"pw\":\"hunter22\"}",
            Some("bad\nhunter22"),
            1.4,
        );
        let t = std::fs::read_to_string(d.join("mcp.log")).unwrap();
        assert!(
            t.ends_with(" stdio status {\"pw\":\"[redacted]\"} error: bad [redacted] 1 ms\n"),
            "{t}"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = std::fs::metadata(d.join("mcp.log"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(m & 0o777, 0o600);
        }
        let long = "x".repeat(1000);
        for _ in 0..1100 {
            l.line(&long);
        }
        let n = |f: &str| std::fs::metadata(d.join(f)).map_or(0, |m| m.len());
        assert!(
            n("mcp.log.1") <= MAX && n("mcp.log.1") > MAX - 2000,
            "{}",
            n("mcp.log.1")
        );
        assert!(n("mcp.log") < MAX / 2);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
