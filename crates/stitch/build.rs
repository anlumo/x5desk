//! RAFT-small model code with embedded weights, generated from the ONNX export in `models/`.

fn main() {
    println!("cargo::rerun-if-changed=models/raft_small_128x2176.onnx");
    burn_onnx::ModelGen::new()
        .input("models/raft_small_128x2176.onnx")
        .out_dir("model/")
        .load_strategy(burn_onnx::LoadStrategy::Embedded)
        .run_from_script();
    // route every conv call through `hc` (src/raft.rs), which runs the f16-weight convs in f16
    let p = std::path::Path::new(&std::env::var("OUT_DIR").unwrap())
        .join("model/raft_small_128x2176.rs");
    let src = std::fs::read_to_string(&p).unwrap();
    let mut n = 0;
    let out: String = src
        .lines()
        .map(|l| {
            let i = l.find("self.conv2d");
            match i.and_then(|i| l[i..].find(".forward(").map(|j| (i, i + j))) {
                // `self.conv2dN.forward(x);` -> `hc(&self.conv2dN, x);`
                Some((i, j)) => {
                    n += 1;
                    format!("{}hc(&{}, {}\n", &l[..i], &l[i..j], &l[j + 9..])
                }
                None => format!("{l}\n"),
            }
        })
        .collect();
    assert!(n > 0, "no conv calls in the generated model");
    std::fs::write(p, out).unwrap();
}
