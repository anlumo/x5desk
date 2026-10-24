//! Network code with embedded weights, generated from the ONNX file in `models/` (source, licence, checksum:
//! `models/README`).

fn main() {
    let m = "human_segmentation_pphumanseg_2023mar";
    println!("cargo::rerun-if-changed=models/{m}.onnx");
    burn_onnx::ModelGen::new()
        .input(&format!("models/{m}.onnx"))
        .out_dir("model/")
        .load_strategy(burn_onnx::LoadStrategy::Embedded)
        .run_from_script();
    // Every stride-2 3 × 3 conv of the network is SAME_UPPER on an even input: pad 0 before, 1 after. burn-onnx 0.22
    // resolves six of them as 1 on both sides (one input pixel off; against onnxruntime the mask moved and blurred
    // by up to 0.89): set them all.
    let p = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join(format!("model/{m}.rs"));
    let src = std::fs::read_to_string(&p).unwrap();
    let mut out = String::with_capacity(src.len());
    let (mut n, mut fixed) = (0, 0);
    for b in src.split_inclusive(".init(device);") {
        if b.contains("Conv2dConfig::new(")
            && b.contains(", [3, 3])")
            && b.contains(".with_stride([2, 2])")
        {
            n += 1;
            if b.contains("PaddingConfig2d::Explicit(1, 1, 1, 1)") {
                fixed += 1;
            }
            out.push_str(&b.replace(
                "PaddingConfig2d::Explicit(1, 1, 1, 1)",
                "PaddingConfig2d::Explicit(0, 0, 1, 1)",
            ));
        } else {
            out.push_str(b);
        }
    }
    assert_eq!((n, fixed), (20, 6), "{m}: stride-2 convs, symmetric ones");
    std::fs::write(&p, out).unwrap();
}
