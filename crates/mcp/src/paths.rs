//! Where agents read and write. Everything an agent writes goes into the agent folder, `<videos>/x5desk/agent`, or a
//! subfolder of it ([`sub`], [`make`]); names already taken there, or being written (`<name>.part`), are numbered as
//! the app's exports are ([`numbered`]). Reads ([`readable`]) are of X5 files ([`EXT`]) under a read root: the agent
//! folder, the app's download folders ([`downloads`]), mounted cards, and the `--allow-dir`s, all compared
//! canonicalised (a link out of a root leads nowhere); what is outside them is not told apart, there or missing.
//! Windows: UNC and `\\?\` input is refused, std's canonical forms (`\\?\C:\…`) are compared with each other only,
//! and shown plain ([`shown`]).

use std::path::{Component, Path, PathBuf};

/// The extensions an agent may read, compared without case.
pub const EXT: [&str; 6] = ["insv", "insp", "lrv", "mp4", "jpg", "dng"];

/// `<videos>/x5desk/agent`.
pub fn agent() -> Option<PathBuf> {
    cam::dirs::videos().map(|v| v.join("x5desk").join("agent"))
}

/// The app's download folders: `dest=` of `<config>/x5desk/camera` when set, and `<videos>/x5desk` (its default,
/// where earlier downloads may be). Read at every call: the app may have moved it.
pub fn downloads() -> Vec<PathBuf> {
    let f = cam::dirs::xdg("XDG_CONFIG_HOME", ".config").map(|d| d.join("x5desk").join("camera"));
    downloads_in(f.as_deref(), cam::dirs::videos())
}

/// [`downloads`] with the app's camera settings in `f` and the video folder `videos`.
fn downloads_in(f: Option<&Path>, videos: Option<PathBuf>) -> Vec<PathBuf> {
    let set = f.and_then(|f| std::fs::read_to_string(f).ok());
    let dest = set.as_deref().and_then(|s| {
        s.lines()
            .find_map(|l| l.strip_prefix("dest="))
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
    });
    dest.into_iter()
        .chain(videos.map(|v| v.join("x5desk")))
        .collect()
}

/// `p` as an agent can give it back: on Windows std's canonical verbatim forms plain (`\\?\C:\x` → `C:\x`,
/// `\\?\UNC\server\share` → `\\server\share`), which [`readable`] would refuse; elsewhere as it is.
pub fn shown(p: &Path) -> String {
    let s = p.display().to_string();
    if cfg!(windows) { plain(&s) } else { s }
}

/// [`shown`]'s mapping of a Windows path; other verbatim forms (`\\?\Volume{…}`) stay.
fn plain(s: &str) -> String {
    if let Some(r) = s.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{r}");
    }
    match s.strip_prefix(r"\\?\") {
        Some(r)
            if r.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
                && r.get(1..3) == Some(r":\") =>
        {
            r.to_string()
        }
        _ => s.to_string(),
    }
}

/// Input a Windows path API would take for a share or a device: `\\server\…`, `\\?\…`, `\\.\…` (and their `/`
/// spellings).
fn foreign(s: &str) -> bool {
    cfg!(windows)
        && (s.starts_with(r"\\")
            || s.starts_with("//")
            || s.starts_with(r"/\")
            || s.starts_with(r"\/"))
}

/// Whether `p` has one of the [`EXT`]s.
pub fn x5(p: &Path) -> bool {
    let e = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    EXT.iter().any(|x| x.eq_ignore_ascii_case(e))
}

/// The roots that exist, canonicalised, without repeats.
pub fn canon(v: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut o: Vec<PathBuf> = vec![];
    for p in v.into_iter().filter_map(|p| std::fs::canonicalize(p).ok()) {
        if !o.contains(&p) {
            o.push(p)
        }
    }
    o
}

/// The file an agent names in `s` if it may read it: an absolute path to an existing file with an X5 extension that,
/// canonicalised, lies under one of `roots` (canonical, [`canon`]). A path outside them gets the one answer whether
/// anything is there or not.
pub fn readable(s: &str, roots: &[PathBuf]) -> Result<PathBuf, String> {
    if foreign(s) {
        return Err(format!("{s}: network and device paths are refused"));
    }
    let p = Path::new(s);
    if !p.is_absolute() {
        return Err(format!("{s}: give an absolute path"));
    }
    let under = |c: &Path| roots.iter().any(|r| c.starts_with(r));
    let out = || {
        format!(
            "{s}: not under a read root (the agent folder, the download folders, a mounted card, an --allow-dir; \
             see status)"
        )
    };
    let c = match std::fs::canonicalize(p) {
        Ok(c) => c,
        // the system's reason only inside a root (its nearest ancestor there is): outside, it would tell what exists
        Err(e) => {
            let near = p
                .ancestors()
                .skip(1)
                .find_map(|a| std::fs::canonicalize(a).ok());
            return Err(match near.is_some_and(|a| under(&a)) {
                true => format!("{s}: {e}"),
                false => out(),
            });
        }
    };
    if !under(&c) {
        return Err(out());
    }
    if !x5(&c) {
        return Err(format!("{s}: not an X5 file ({})", EXT.join(", ")));
    }
    if !c.is_file() {
        return Err(format!("{s}: not a file"));
    }
    Ok(c)
}

/// The subfolder `s` of the agent folder: relative, without `..`, a root, a drive or a share; `:` is refused too
/// (a Windows drive-relative path, an alternate data stream), and control characters (a line break or tab would
/// split the kept queue's line). Empty: the agent folder itself.
pub fn sub(agent: &Path, s: &str) -> Result<PathBuf, String> {
    if foreign(s) || s.contains(':') || s.chars().any(char::is_control) {
        return Err(format!("{s}: not a subfolder name"));
    }
    let mut d = agent.to_path_buf();
    for c in Path::new(s).components() {
        match c {
            Component::Normal(n) => d.push(n),
            Component::CurDir => {}
            Component::ParentDir => return Err(format!("{s}: `..` is refused")),
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("{s}: give a folder relative to the agent folder"));
            }
        }
    }
    Ok(d)
}

/// Makes `d` (from [`sub`]) and checks that, canonicalised, it is still inside the agent folder: no link leads out.
/// Returns `d` as given.
pub fn make(agent: &Path, d: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    let (a, c) = (std::fs::canonicalize(agent), std::fs::canonicalize(d));
    match (a, c) {
        (Ok(a), Ok(c)) if c.starts_with(&a) => Ok(d.to_path_buf()),
        (Ok(_), Ok(c)) => Err(format!(
            "{}: leads out of the agent folder (to {})",
            d.display(),
            c.display()
        )),
        (Err(e), _) | (_, Err(e)) => Err(format!("{}: {e}", d.display())),
    }
}

/// `name` with `-i` before its extension when `i` > 1, as the app numbers its exports.
pub fn numbered(name: &str, i: u32) -> String {
    if i <= 1 {
        return name.to_string();
    }
    match name.rsplit_once('.') {
        Some((s, e)) if !s.is_empty() => format!("{s}-{i}.{e}"),
        _ => format!("{name}-{i}"),
    }
}

/// Nothing is at `p`, and nothing is being written there (`<p>.part`).
pub fn free(p: &Path) -> bool {
    let mut part = p.as_os_str().to_owned();
    part.push(".part");
    !p.exists() && !Path::new(&part).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rm(PathBuf);

    impl Drop for Rm {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn reads() {
        let t = Rm(std::env::temp_dir().join(format!("x5desk-mcp-paths-{}", std::process::id())));
        let (root, out) = (t.0.join("root"), t.0.join("out"));
        for d in [&root, &out, &root.join("sub")] {
            std::fs::create_dir_all(d).unwrap();
        }
        for f in ["a.insv", "B.LRV", "c.txt", "sub/d.mp4"] {
            std::fs::write(root.join(f), b"x").unwrap();
        }
        std::fs::write(out.join("e.insv"), b"x").unwrap();
        let roots = canon([root.clone(), t.0.join("missing"), root.clone()]);
        assert_eq!(roots.len(), 1, "missing and repeated roots dropped");
        let s = |p: &Path| p.to_str().unwrap().to_string();
        assert!(readable(&s(&root.join("a.insv")), &roots).is_ok());
        assert!(
            readable(&s(&root.join("B.LRV")), &roots).is_ok(),
            "extensions without case"
        );
        assert!(readable(&s(&root.join("sub/d.mp4")), &roots).is_ok());
        let e = |p: &str| readable(p, &roots).unwrap_err();
        assert!(e(&s(&root.join("c.txt"))).contains("not an X5 file"));
        assert!(e(&s(&out.join("e.insv"))).contains("not under a read root"));
        assert!(e(&s(&root.join("sub/../../out/e.insv"))).contains("not under a read root"));
        // inside a root the system says what is wrong; outside, a missing file reads as one that is there
        let missing = e(&s(&root.join("nope.insv")));
        assert!(!missing.contains("not under a read root"), "{missing}");
        assert!(!e(&s(&root.join("gone/nope.insv"))).contains("not under a read root"));
        assert_eq!(
            e(&s(&out.join("nope.insv"))).replace("nope", "e"),
            e(&s(&out.join("e.insv")))
        );
        for p in [
            out.join("gone/nope.insv"),
            t.0.join("gone").join("x.insv"),
            root.join("../out/nope.insv"),
            root.join("sub/../../gone/x.insv"),
        ] {
            assert!(
                e(&s(&p)).contains("not under a read root"),
                "{}",
                p.display()
            );
        }
        assert!(e("a.insv").contains("absolute"));
        let dir = root.join("x.insv");
        std::fs::create_dir(&dir).unwrap();
        assert!(e(&s(&dir)).contains("not a file"));
        #[cfg(unix)]
        {
            // a link out of the root, and one inside it to a file of another kind
            std::os::unix::fs::symlink(out.join("e.insv"), root.join("l.insv")).unwrap();
            assert!(e(&s(&root.join("l.insv"))).contains("not under a read root"));
            std::os::unix::fs::symlink(root.join("c.txt"), root.join("m.insv")).unwrap();
            assert!(e(&s(&root.join("m.insv"))).contains("not an X5 file"));
            // a link to a folder outside: what is there or missing behind it is not told apart
            std::os::unix::fs::symlink(&out, root.join("o")).unwrap();
            assert_eq!(
                e(&s(&root.join("o/nope.insv"))).replace("nope", "e"),
                e(&s(&root.join("o/e.insv")))
            );
        }
        #[cfg(windows)]
        for p in [
            r"\\server\share\a.insv",
            r"\\?\C:\a.insv",
            r"\\.\pipe\x",
            "//server/share/a.insv",
        ] {
            assert!(e(p).contains("network and device"), "{p}");
        }
    }

    #[test]
    fn writes() {
        let t = Rm(std::env::temp_dir().join(format!("x5desk-mcp-sub-{}", std::process::id())));
        let a = t.0.join("agent");
        assert_eq!(sub(&a, "").unwrap(), a);
        assert_eq!(sub(&a, "trip/day1").unwrap(), a.join("trip").join("day1"));
        assert_eq!(sub(&a, "./trip").unwrap(), a.join("trip"));
        for s in [
            "..",
            "a/../../b",
            "a/..",
            "/etc",
            "C:x",
            "a:b",
            "a\nqueued\tb",
            "a\tb",
            "a\rb",
            "a\0b",
            "a\u{7f}b",
        ] {
            assert!(sub(&a, s).is_err(), "{s}");
        }
        #[cfg(windows)]
        for s in [r"\\server\share", r"\\?\C:\x", r"C:\x", r"\x"] {
            assert!(sub(&a, s).is_err(), "{s}");
        }
        let d = make(&a, &sub(&a, "trip").unwrap()).unwrap();
        assert!(d.is_dir() && d.ends_with("trip"));
        #[cfg(unix)]
        {
            std::fs::create_dir_all(t.0.join("elsewhere")).unwrap();
            std::os::unix::fs::symlink(t.0.join("elsewhere"), a.join("out")).unwrap();
            assert!(
                make(&a, &sub(&a, "out/x").unwrap())
                    .unwrap_err()
                    .contains("leads out")
            );
        }
        assert_eq!(numbered("VID_1_00_603.insv", 1), "VID_1_00_603.insv");
        assert_eq!(numbered("VID_1_00_603.insv", 2), "VID_1_00_603-2.insv");
        assert_eq!(numbered("noext", 3), "noext-3");
        assert_eq!(numbered(".hidden", 2), ".hidden-2");
        let f = a.join("f.insv");
        assert!(free(&f));
        std::fs::write(a.join("f.insv.part"), b"").unwrap();
        assert!(!free(&f), "being written");
        std::fs::remove_file(a.join("f.insv.part")).unwrap();
        std::fs::write(&f, b"").unwrap();
        assert!(!free(&f));
    }

    /// The app's download folders: its `dest=` and the default beside it, read again each time.
    #[test]
    fn download_roots() {
        let t = Rm(std::env::temp_dir().join(format!("x5desk-mcp-dl-{}", std::process::id())));
        std::fs::create_dir_all(&t.0).unwrap();
        let (f, v) = (t.0.join("camera"), PathBuf::from("/home/me/Videos"));
        let d = || downloads_in(Some(&f), Some(v.clone()));
        assert_eq!(d(), [v.join("x5desk")], "no settings yet");
        std::fs::write(&f, "dest=/Volumes/SSD/x5\nauto=1\n").unwrap();
        assert_eq!(d(), [PathBuf::from("/Volumes/SSD/x5"), v.join("x5desk")]);
        std::fs::write(&f, "auto=0\ndest=/mnt/clips\n").unwrap();
        assert_eq!(d()[0], PathBuf::from("/mnt/clips"), "moved: read again");
        std::fs::write(&f, "dest=\n").unwrap();
        assert_eq!(d(), [v.join("x5desk")]);
        assert!(downloads_in(None, None).is_empty());
    }

    #[test]
    fn plain_paths() {
        for (i, o) in [
            (
                r"\\?\C:\Users\me\Videos\x5desk\agent",
                r"C:\Users\me\Videos\x5desk\agent",
            ),
            (r"\\?\d:\x.insv", r"d:\x.insv"),
            (r"\\?\UNC\server\share\x.insv", r"\\server\share\x.insv"),
            (r"\\?\Volume{1234}\x.insv", r"\\?\Volume{1234}\x.insv"),
            (r"\\?\C:", r"\\?\C:"),
            (r"\\?\1:\x", r"\\?\1:\x"),
            (r"C:\x.insv", r"C:\x.insv"),
            (r"\\server\share", r"\\server\share"),
            ("/home/me/x.insv", "/home/me/x.insv"),
        ] {
            assert_eq!(plain(i), o, "{i}");
        }
        let p = Path::new("/home/me/Videos/x5desk/agent/VID.insv");
        if !cfg!(windows) {
            assert_eq!(shown(p), p.display().to_string());
        }
    }
}
