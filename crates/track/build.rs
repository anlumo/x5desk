//! Network code with embedded weights, generated from the ONNX files in `models/` (sources, licences, checksums:
//! `models/README`).

fn main() {
    for m in [
        "object_tracking_vittrack_2023sep",
        "yolox_nano",
        "osnet_ain_x1_0_msmt17",
        "face_detection_yunet_2023mar",
    ] {
        println!("cargo::rerun-if-changed=models/{m}.onnx");
        burn_onnx::ModelGen::new()
            .input(&format!("models/{m}.onnx"))
            .out_dir("model/")
            .load_strategy(burn_onnx::LoadStrategy::Embedded)
            .run_from_script();
        // the exports fix the batch at 1 in their head reshapes; 0 keeps the input's, so tiles go in one batch
        let fix: &[(&str, usize)] = match m {
            "yolox_nano" => &[("1, 85, -1]", 3)],
            "face_detection_yunet_2023mar" => {
                &[("1, -1, 1]", 6), ("1, -1, 4]", 3), ("1, -1, 10]", 3)]
            }
            _ => continue,
        };
        let p =
            std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join(format!("model/{m}.rs"));
        let mut src = std::fs::read_to_string(&p).unwrap();
        for (r, n) in fix {
            let r = format!(".reshape([{r}");
            assert_eq!(src.matches(&r).count(), *n, "{m}: head reshapes {r}");
            src = src.replace(&r, &r.replacen("[1", "[0", 1));
        }
        std::fs::write(&p, src).unwrap();
    }
}
