//! A recording's files by name: its main files in lens order (`_00_` lens 0 or both, `_10_` lens 1) and its proxy
//! (`_01_` / `_11_`), paired by the recording key the album and the Library group by.

use std::path::{Path, PathBuf};

/// The files of one recording: main files in lens order (one with both lenses on the X5), the proxy.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Work {
    pub main: Vec<PathBuf>,
    pub lrv: Option<PathBuf>,
}

/// A path's file name (empty if it has none or it is not UTF-8).
pub fn name(p: &Path) -> &str {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("")
}

/// A path's extension, lower case.
pub fn ext(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

/// Recording key: the name without its extension and its stream parts (`00`, `01`, `10`, `11`), `LRV` read as
/// `VID` (the official app's fileKey, re/analysis/media.md §3.4).
pub fn key(name: &str) -> String {
    let stem = name.rsplit_once('.').map_or(name, |s| s.0);
    let mut k = String::new();
    for p in stem
        .split('_')
        .filter(|p| !matches!(*p, "00" | "01" | "10" | "11"))
    {
        k += if p.eq_ignore_ascii_case("lrv") {
            "VID"
        } else {
            p
        };
        k += "_";
    }
    k
}

/// The stream part (`00` lens 0 or both, `10` lens 1, `01` / `11` their proxies); files without one sort first.
fn part(p: &Path) -> u8 {
    let n = name(p);
    let n = n.rsplit_once('.').map_or(n, |s| s.0);
    n.split('_')
        .find_map(|x| ["00", "01", "10", "11"].iter().position(|s| *s == x))
        .map_or(0, |i| i as u8 + 1)
}

/// The work `p` belongs to among `all` (its directory).
pub fn group(p: &Path, all: &[PathBuf]) -> Work {
    let k = key(name(p));
    let (mut main, mut lrv) = (vec![], vec![]);
    for q in all.iter().map(PathBuf::as_path).chain([p]) {
        let v = match ext(q).as_str() {
            "insv" | "mp4" => &mut main,
            "lrv" => &mut lrv,
            _ => continue,
        };
        if key(name(q)) == k && !v.contains(&q) {
            v.push(q);
        }
    }
    main.sort_by_key(|q| (part(q), name(q).to_string()));
    lrv.sort_by_key(|q| (part(q), name(q).to_string()));
    Work {
        main: main.into_iter().map(Path::to_path_buf).collect(),
        lrv: lrv.first().map(|q| q.to_path_buf()),
    }
}

/// The work `p` belongs to among the files of its directory.
pub fn pair(p: &Path) -> Work {
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty());
    let all: Vec<PathBuf> = std::fs::read_dir(dir.unwrap_or(Path::new(".")))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .collect();
    group(p, &all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys() {
        let k = key("VID_20250101_120000_00_001.insv");
        assert_eq!(k, key("VID_20250101_120000_10_001.insv"));
        assert_eq!(k, key("LRV_20250101_120000_01_001.lrv"));
        assert_eq!(k, key("LRV_20250101_120000_11_001.lrv"));
        assert_ne!(k, key("VID_20250101_120000_00_002.insv"));
        assert_ne!(k, key("PRO_VID_20250101_120000_00_001.insv"));
        assert_eq!(
            key("PRO_VID_20250101_120000_00_001.insv"),
            key("PRO_LRV_20250101_120000_01_001.lrv")
        );
        assert_eq!(key("v635.insv"), key("v635.lrv"));
        assert_ne!(key("v635.insv"), key("v606.insv"));
    }

    #[test]
    fn pairs() {
        let all: Vec<PathBuf> = [
            "VID_20250101_120000_10_001.insv",
            "LRV_20250101_120000_11_001.lrv",
            "LRV_20250101_120000_01_001.lrv",
            "VID_20250101_120000_00_001.insv",
            "VID_20250101_120500_00_002.insv",
            "LRV_20250101_120500_01_002.lrv",
            "IMG_20250101_120000_00_003.insp",
            "v635.insv",
            "v635.lrv",
            "v606.insv",
        ]
        .iter()
        .map(|n| Path::new("/d").join(n))
        .collect();
        let p = |n: &str| Path::new("/d").join(n);
        let dual = Work {
            main: vec![
                p("VID_20250101_120000_00_001.insv"),
                p("VID_20250101_120000_10_001.insv"),
            ],
            lrv: Some(p("LRV_20250101_120000_01_001.lrv")),
        };
        // any file of the recording opens the same work
        for n in [
            "VID_20250101_120000_10_001.insv",
            "LRV_20250101_120000_11_001.lrv",
            "VID_20250101_120000_00_001.insv",
        ] {
            assert_eq!(group(&p(n), &all), dual, "{n}");
        }
        assert_eq!(
            group(&p("LRV_20250101_120500_01_002.lrv"), &all),
            Work {
                main: vec![p("VID_20250101_120500_00_002.insv")],
                lrv: Some(p("LRV_20250101_120500_01_002.lrv")),
            }
        );
        assert_eq!(
            group(&p("v635.lrv"), &all),
            Work {
                main: vec![p("v635.insv")],
                lrv: Some(p("v635.lrv")),
            }
        );
        assert_eq!(
            group(&p("v606.insv"), &all),
            Work {
                main: vec![p("v606.insv")],
                lrv: None,
            }
        );
        // a proxy without its main file, a file missing from the listing
        assert_eq!(
            group(&p("x.lrv"), &[]),
            Work {
                main: vec![],
                lrv: Some(p("x.lrv")),
            }
        );
    }
}
