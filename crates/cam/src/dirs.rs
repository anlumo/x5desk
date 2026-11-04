//! Where the app keeps its files: the config and cache dirs (XDG base directories; on macOS the Library's, on Windows
//! the profile's), the user's video folder, and owner-only dirs for what others must not read or plant.

use std::path::{Path, PathBuf};

/// `$<var>` if absolute, else `~/<home>` (XDG base directories). On macOS `~/.config` and `~/.cache` are
/// `~/Library/Application Support` and `~/Library/Caches`, on Windows `%APPDATA%` and `%LOCALAPPDATA%`; the XDG
/// variables still override them.
pub fn xdg(var: &str, home: &str) -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    let home = match home {
        ".config" => "Library/Application Support",
        ".cache" => "Library/Caches",
        h => h,
    };
    #[cfg(windows)]
    let base = || match home {
        ".config" => std::env::var_os("APPDATA").map(PathBuf::from),
        ".cache" => std::env::var_os("LOCALAPPDATA").map(PathBuf::from),
        h => std::env::var_os("USERPROFILE").map(|p| PathBuf::from(p).join(h)),
    };
    #[cfg(not(windows))]
    let base = || std::env::var_os("HOME").map(|h| PathBuf::from(h).join(home));
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|d| d.is_absolute())
        .or_else(base)
}

/// The user's video folder: `~/Videos`, on macOS `~/Movies`, on Windows `%USERPROFILE%\Videos`.
pub fn videos() -> Option<PathBuf> {
    let d = if cfg!(target_os = "macos") {
        "Movies"
    } else {
        "Videos"
    };
    let home = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(home).map(|h| PathBuf::from(h).join(d))
}

/// `d`, made owner-only if missing, if it is a dir of this user's that no other can write to: in a shared tmp another
/// user may have made it first, or put a symlink there.
#[cfg(unix)]
pub fn mine(d: PathBuf) -> Option<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    let _ = std::fs::DirBuilder::new().mode(0o700).create(&d);
    let m = d.symlink_metadata().ok()?;
    // SAFETY: geteuid has no preconditions.
    let me = unsafe { libc::geteuid() };
    (m.is_dir() && m.uid() == me && m.mode() & 0o022 == 0).then_some(d)
}

/// `d`, made if missing, if it is a dir (not a link): the profile's temp and app dirs it goes in are the user's alone
/// by their ACL.
#[cfg(not(unix))]
pub fn mine(d: PathBuf) -> Option<PathBuf> {
    let _ = std::fs::create_dir(&d);
    d.symlink_metadata().ok()?.is_dir().then_some(d)
}

/// Makes the app's config dir `d` before a save there: created if missing, owner-only also when it exists already
/// (it holds the stream key, the client id, the recent files). Its parents keep their modes.
#[cfg(unix)]
pub fn private(d: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    if let Some(p) = d.parent() {
        std::fs::create_dir_all(p)?
    }
    match std::fs::DirBuilder::new().mode(0o700).create(d) {
        Err(e) if e.kind() != std::io::ErrorKind::AlreadyExists => return Err(e),
        _ => {}
    }
    std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o700))
}

/// Makes the app's config dir `d` before a save there: in the profile's `%APPDATA%`, whose ACL keeps it the user's.
#[cfg(not(unix))]
pub fn private(d: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(d)
}

#[cfg(test)]
mod tests {
    /// The hook's cubecl dir: made owner-only; refused when another user owns it (here root), when it is a symlink (to
    /// a dir of ours too) or when others can write to it.
    #[cfg(unix)]
    #[test]
    fn cubecl_dir() {
        use std::os::unix::fs::{PermissionsExt, symlink};
        struct Rm(std::path::PathBuf);
        impl Drop for Rm {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let run = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(std::env::temp_dir, Into::into);
        let b = Rm(run.join(format!("x5desk-mine-{}", std::process::id())));
        std::fs::create_dir(&b.0).unwrap();
        let mode = |d: &std::path::Path| std::fs::metadata(d).unwrap().permissions().mode() & 0o777;
        let d = b.0.join("x5desk-cubecl");
        assert_eq!(super::mine(d.clone()), Some(d.clone()));
        assert_eq!(mode(&d), 0o700);
        assert_eq!(super::mine("/".into()), None);
        let l = b.0.join("link");
        symlink(&d, &l).unwrap();
        assert_eq!(super::mine(l), None);
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert_eq!(super::mine(d.clone()), None);
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(super::mine(d.clone()), Some(d));
    }
}
