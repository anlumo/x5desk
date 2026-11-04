//! What agents may do: the tiers the user enabled, in `<config>/x5desk/mcp` (0600 on unix, its dir owner-only through
//! `cam::dirs::private`). One line `tiers = read,local,camera`; other keys and lines are kept and ignored, a missing
//! or unreadable file enables nothing. Command-line flags only narrow the set ([`Tiers::and`]), and `delete` is only
//! ever enabled by naming it (there is no "all"). A host re-reads the file at most every 2 s ([`Watch`]). Writers
//! ([`set`]: `enable`, `disable`; [`get_or_set`]: a host's Wi-Fi id, looked up or made) take turns under a lock on
//! `<config>/x5desk/mcp.lock`.

use std::{
    fmt,
    fs::{File, OpenOptions},
    io,
    path::{Path, PathBuf},
    time::SystemTime,
};

/// A group of tools enabled as one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// state, listings, thumbnails, file metadata, jobs
    Read,
    /// writes into the agent folder: downloads (later stills and exports)
    Local,
    /// the camera session: connect and release (later capture and settings)
    Camera,
    /// deleting files on the camera (M2)
    Delete,
}

impl Tier {
    /// Every tier, in the order they build on each other.
    pub const ALL: [Tier; 4] = [Tier::Read, Tier::Local, Tier::Camera, Tier::Delete];

    /// The tier's name in the config file and on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Read => "read",
            Tier::Local => "local",
            Tier::Camera => "camera",
            Tier::Delete => "delete",
        }
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of tiers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tiers(u8);

impl Tiers {
    /// Nothing enabled.
    pub const NONE: Tiers = Tiers(0);

    /// What `--sim` enables without `X5DESK_MCP_TIERS`: all but `delete`.
    pub const SIM: Tiers = Tiers(0b0111);

    /// Whether `t` is in the set.
    pub fn has(self, t: Tier) -> bool {
        self.0 & t.bit() != 0
    }

    /// The set and `t`.
    pub fn with(self, t: Tier) -> Tiers {
        Tiers(self.0 | t.bit())
    }

    /// Both sets: a narrowing never adds.
    pub fn and(self, o: Tiers) -> Tiers {
        Tiers(self.0 & o.0)
    }

    /// Nothing enabled.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The tiers in the set, in [`Tier::ALL`]'s order.
    pub fn iter(self) -> impl Iterator<Item = Tier> {
        Tier::ALL.into_iter().filter(move |t| self.has(*t))
    }

    /// A comma list of tier names (spaces around them allowed, empty parts ignored). There is no "all": `delete` is
    /// only enabled by its name.
    pub fn parse(s: &str) -> Result<Tiers, String> {
        let mut t = Tiers::NONE;
        for p in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match Tier::ALL
                .into_iter()
                .find(|x| x.name().eq_ignore_ascii_case(p))
            {
                Some(x) => t = t.with(x),
                None => {
                    return Err(format!(
                        "unknown tier {p:?}: name them, from read, local, camera, delete"
                    ));
                }
            }
        }
        Ok(t)
    }
}

impl fmt::Display for Tiers {
    /// `read,local`, or `none`.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        let v: Vec<&str> = self.iter().map(Tier::name).collect();
        f.write_str(&v.join(","))
    }
}

/// `<config>/x5desk/mcp`; none without a home.
pub fn file() -> Option<PathBuf> {
    cam::dirs::xdg("XDG_CONFIG_HOME", ".config").map(|d| d.join("x5desk").join("mcp"))
}

/// The value of `key` in a config text: the last `key = value` line.
pub fn get<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines()
        .filter_map(|l| l.split_once('='))
        .filter(|(k, _)| k.trim() == key)
        .map(|(_, v)| v.trim())
        .next_back()
}

/// The tiers a config text enables: none unless its `tiers` line parses.
pub fn parse(text: &str) -> Tiers {
    get(text, "tiers")
        .and_then(|v| Tiers::parse(v).ok())
        .unwrap_or_default()
}

/// The tiers the file at `p` enables; none when it is missing or unreadable.
pub fn load(p: &Path) -> Tiers {
    std::fs::read_to_string(p).map_or(Tiers::NONE, |s| parse(&s))
}

/// The lock file `l` opened, made if missing (0600 and not through a symlink on unix); not locked yet.
pub fn lock_file(l: &Path) -> io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut o, libc::O_NOFOLLOW);
    o.open(l)
}

/// `<p>.lock` taken: held until the file drops ([`lock_file`]).
fn lock(p: &Path) -> io::Result<File> {
    let mut l = p.as_os_str().to_owned();
    l.push(".lock");
    let f = lock_file(Path::new(&l))?;
    loop {
        match f.lock() {
            Ok(()) => return Ok(f),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

/// Sets `key` in the file at `p` (keeping its other lines), written whole: owner-only dir, the file 0600 on unix.
/// Read, rewritten and renamed into place under `<p>.lock`, through a temp name of this call's own, so concurrent
/// writers (in any process) lose nothing of each other's.
pub fn set(p: &Path, key: &str, value: &str) -> io::Result<()> {
    let _l = locked(p)?;
    let old = std::fs::read_to_string(p).unwrap_or_default();
    put(p, &old, key, value)
}

/// The value of `key` in the file at `p` if it has a non-empty one, else `make()`'s, stored as [`set`] stores it:
/// looked up and written under the one lock, so of callers at the same moment (in any process) one makes the value
/// and the others get it.
pub fn get_or_set(p: &Path, key: &str, make: impl FnOnce() -> String) -> io::Result<String> {
    let _l = locked(p)?;
    let old = std::fs::read_to_string(p).unwrap_or_default();
    if let Some(v) = get(&old, key).filter(|v| !v.is_empty()) {
        return Ok(v.into());
    }
    let v = make();
    put(p, &old, key, &v)?;
    Ok(v)
}

/// The file's dir made owner-only, then its lock taken.
fn locked(p: &Path) -> io::Result<File> {
    if let Some(d) = p.parent() {
        cam::dirs::private(d)?
    }
    lock(p)
}

/// Writes the file at `p` as its text `old` with `key` set to `value` (under its lock).
fn put(p: &Path, old: &str, key: &str, value: &str) -> io::Result<()> {
    let mut s = format!("{key} = {value}\n");
    for l in old.lines() {
        if l.split_once('=').is_none_or(|(k, _)| k.trim() != key) {
            s += l;
            s += "\n";
        }
    }
    let mut t = p.as_os_str().to_owned();
    t.push(format!(
        ".{}-{:016x}.new",
        std::process::id(),
        crate::job::random()
    ));
    let t = PathBuf::from(t);
    let mut o = OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    let r = (o.open(&t))
        .and_then(|mut f| io::Write::write_all(&mut f, s.as_bytes()))
        .and_then(|_| std::fs::rename(&t, p));
    if r.is_err() {
        let _ = std::fs::remove_file(&t);
    }
    r
}

/// Enables exactly `t` in the file at `p` (`x5desk mcp enable --tiers …` replaces the set); returns what it now
/// enables.
pub fn enable_at(p: &Path, t: Tiers) -> io::Result<Tiers> {
    let v: Vec<&str> = t.iter().map(Tier::name).collect();
    set(p, "tiers", &v.join(","))?;
    Ok(load(p))
}

/// [`enable_at`] the user's file; returns it and what it now enables.
pub fn enable(t: Tiers) -> io::Result<(PathBuf, Tiers)> {
    let p = file().ok_or_else(|| io::Error::other("no config dir (HOME not set)"))?;
    let t = enable_at(&p, t)?;
    Ok((p, t))
}

/// Turns every tier off in the user's file.
pub fn disable() -> io::Result<PathBuf> {
    enable(Tiers::NONE).map(|r| r.0)
}

/// Where a host's tiers come from.
#[derive(Clone, Debug)]
pub enum Source {
    /// a config file, re-read when it changes
    File(PathBuf),
    /// a set that never changes (`--sim`, tests)
    Fixed(Tiers),
}

/// The file's modification time and length, `None` when it is missing.
type Stamp = Option<(Option<SystemTime>, u64)>;

/// The enabled set of a [`Source`] narrowed by the command line, re-read from a file when it changed.
pub struct Watch {
    src: Source,
    narrow: Option<Tiers>,
    /// the file's stamp when last read
    seen: Option<Stamp>,
    now: Tiers,
}

impl Watch {
    /// Reads the source once; `narrow`: the command line's `--tiers`.
    pub fn new(src: Source, narrow: Option<Tiers>) -> Watch {
        let mut w = Watch {
            src,
            narrow,
            seen: None,
            now: Tiers::NONE,
        };
        w.poll();
        w
    }

    /// The enabled set as last read.
    pub fn tiers(&self) -> Tiers {
        self.now
    }

    /// Reads the file again if its time or length changed; returns the enabled set when that changed.
    pub fn poll(&mut self) -> Option<Tiers> {
        let base = match &self.src {
            Source::Fixed(t) => *t,
            Source::File(p) => {
                let m = std::fs::metadata(p)
                    .ok()
                    .map(|m| (m.modified().ok(), m.len()));
                if self.seen == Some(m) {
                    return None;
                }
                self.seen = Some(m);
                load(p)
            }
        };
        let t = self.narrow.map_or(base, |n| base.and(n));
        (std::mem::replace(&mut self.now, t) != t).then_some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers() {
        let t = Tiers::parse(" read, Local ,,camera").unwrap();
        assert!(
            t.has(Tier::Read) && t.has(Tier::Local) && t.has(Tier::Camera) && !t.has(Tier::Delete)
        );
        assert_eq!(t.to_string(), "read,local,camera");
        assert_eq!(Tiers::NONE.to_string(), "none");
        assert_eq!(Tiers::parse("").unwrap(), Tiers::NONE);
        // no wildcard: delete is named or off
        for s in ["all", "*", "read,everything"] {
            assert!(Tiers::parse(s).unwrap_err().contains("unknown tier"), "{s}");
        }
        assert_eq!(Tiers::SIM.to_string(), "read,local,camera");
        // narrowing never adds
        let n = Tiers::parse("read,delete").unwrap();
        assert_eq!(t.and(n).to_string(), "read");
        assert_eq!(parse("x = 1\ntiers = read\nid = abc\n").to_string(), "read");
        assert_eq!(
            parse("tiers = read\ntiers=camera,local\n").to_string(),
            "local,camera"
        );
        // a line that does not parse enables nothing
        assert_eq!(parse("tiers = read,all\n"), Tiers::NONE);
        assert_eq!(parse("garbage\n"), Tiers::NONE);
    }

    #[test]
    fn file() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let p = d.join("x5desk").join("mcp");
        assert_eq!(load(&p), Tiers::NONE);
        let mut w = Watch::new(
            Source::File(p.clone()),
            Some(Tiers::parse("read,camera").unwrap()),
        );
        assert_eq!(w.tiers(), Tiers::NONE);
        assert_eq!(w.poll(), None);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, "id = keep me\n").unwrap();
        assert_eq!(w.poll(), None, "no tiers line: still none");
        let t = enable_at(&p, Tiers::parse("read,local").unwrap()).unwrap();
        assert_eq!(t.to_string(), "read,local");
        let s = std::fs::read_to_string(&p).unwrap();
        assert_eq!(s, "tiers = read,local\nid = keep me\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!((m(&p), m(p.parent().unwrap())), (0o600, 0o700));
        }
        // the file intersected with the command line
        assert_eq!(w.poll().map(|t| t.to_string()).as_deref(), Some("read"));
        assert_eq!(w.poll(), None);
        enable_at(&p, Tiers::NONE).unwrap();
        assert_eq!(
            std::fs::read_to_string(&p).unwrap(),
            "tiers = \nid = keep me\n"
        );
        assert_eq!(w.poll(), Some(Tiers::NONE));
        std::fs::remove_file(&p).unwrap();
        assert_eq!(w.poll(), None, "gone: none, as before");
        let mut f = Watch::new(Source::Fixed(Tiers::SIM), None);
        assert_eq!((f.tiers(), f.poll()), (Tiers::SIM, None));
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// `enable` and a host keeping its Wi-Fi id at the same moment, many times over: both land, neither fails, no
    /// temp file stays.
    #[test]
    fn concurrent() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-config-race-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let p = d.join("x5desk").join("mcp");
        set(&p, "other", "kept").unwrap();
        for i in 0..200 {
            let t = Tiers::parse(["read,local", "camera"][i % 2]).unwrap();
            let id = format!("id-{i}");
            let b = std::sync::Barrier::new(2);
            let (e, k) = std::thread::scope(|s| {
                let e = s.spawn(|| {
                    b.wait();
                    enable_at(&p, t)
                });
                let k = s.spawn(|| {
                    b.wait();
                    set(&p, "id", &id)
                });
                (e.join().unwrap(), k.join().unwrap())
            });
            assert_eq!(e.unwrap(), t, "round {i}");
            k.unwrap_or_else(|e| panic!("round {i}: {e}"));
            let s = std::fs::read_to_string(&p).unwrap();
            assert_eq!(
                (parse(&s), get(&s, "id"), get(&s, "other")),
                (t, Some(id.as_str()), Some("kept")),
                "round {i}: {s}"
            );
        }
        let mut left: Vec<_> = (std::fs::read_dir(p.parent().unwrap()).unwrap())
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        assert_eq!(left, ["mcp", "mcp.lock"]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = std::fs::metadata(d.join("x5desk").join("mcp.lock")).unwrap();
            assert_eq!(m.permissions().mode() & 0o777, 0o600);
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Hosts looking up their Wi-Fi id at the same moment, the first time: one makes it, all get that one, the
    /// tiers stay; later it is only read.
    #[test]
    fn one_id() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-config-id-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let p = d.join("x5desk").join("mcp");
        enable_at(&p, Tiers::parse("read").unwrap()).unwrap();
        for i in 0..50 {
            set(&p, "id", "").unwrap();
            let b = std::sync::Barrier::new(4);
            let got: Vec<String> = std::thread::scope(|s| {
                let v: Vec<_> = (0..4)
                    .map(|k| {
                        let (b, p) = (&b, &p);
                        s.spawn(move || {
                            b.wait();
                            get_or_set(p, "id", || format!("id-{i}-{k}")).unwrap()
                        })
                    })
                    .collect();
                v.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let s = std::fs::read_to_string(&p).unwrap();
            assert!(
                got.iter().all(|g| Some(g.as_str()) == get(&s, "id")),
                "round {i}: {got:?} {s}"
            );
            assert_eq!(parse(&s).to_string(), "read", "round {i}");
        }
        let id = get(&std::fs::read_to_string(&p).unwrap(), "id")
            .unwrap()
            .to_string();
        assert_eq!(get_or_set(&p, "id", || unreachable!()).unwrap(), id);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
