//! Versions for the About page: the toolchain and the main crates as the lock file pins them.

fn main() {
    println!("cargo::rerun-if-changed=../../Cargo.lock");
    let l = std::fs::read_to_string("../../Cargo.lock").unwrap_or_default();
    let v: Vec<String> = [
        "eframe",
        "egui",
        "wgpu",
        "ash",
        "burn",
        "btleplug",
        "nusb",
        "cpal",
        "symphonia",
        "zune-jpeg",
        "prost",
    ]
    .iter()
    .filter_map(|n| {
        let name = format!("name = \"{n}\"");
        let p = l
            .split("[[package]]")
            .find(|p| p.lines().any(|x| x == name))?;
        let v = p.lines().find_map(|x| x.strip_prefix("version = \""))?;
        Some(format!("{n} {}", v.trim_end_matches('"')))
    })
    .collect();
    println!("cargo::rustc-env=X5_DEPS={}", v.join(", "));
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let r = std::process::Command::new(rustc)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    println!("cargo::rustc-env=X5_RUSTC={r}");
}
