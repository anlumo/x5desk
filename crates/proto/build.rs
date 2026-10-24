use std::{fs, path::Path, path::PathBuf};

fn walk(d: &Path, v: &mut Vec<PathBuf>) {
    for e in fs::read_dir(d).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, v)
        } else if p.extension().is_some_and(|x| x == "proto") {
            v.push(p)
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=proto");
    let mut v = vec![];
    walk(Path::new("proto"), &mut v);
    v.sort();
    let fds = protox::compile(&v, ["proto"]).unwrap();
    // The schema comments are only Java class references from the app.
    prost_build::Config::new()
        .disable_comments(["."])
        .compile_fds(fds)
        .unwrap();
}
