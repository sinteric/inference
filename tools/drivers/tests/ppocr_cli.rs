//! `ppocr-onednn` end to end on one page of the imported PP-OCR assets:
//! warmup lines are not written, a rerun resumes, a failing page is an error
//! line, values are rounded, and `--workspace-limit-bytes` reaches the
//! model. Skipped unless `SINTERIC_OCR_ASSETS` names the asset root with
//! its parity fixtures (`round3/kd0092/reference.json`).

use std::path::PathBuf;
use std::process::Command;

use serde_json::{json, Value};

#[test]
fn jsonl_warmup_resume_errors_and_rounding() {
    let Some(root) = std::env::var_os("SINTERIC_OCR_ASSETS").map(PathBuf::from) else {
        eprintln!("skipping CLI protocol: SINTERIC_OCR_ASSETS unset");
        return;
    };
    if !root.join("round3/kd0092/reference.json").is_file() {
        eprintln!(
            "skipping CLI protocol: no parity fixtures under {}",
            root.display()
        );
        return;
    }
    let dir = root.join(format!("round3/cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("round3/kd0092/reference.json")).unwrap())
            .unwrap();
    let w = reference["page_width"].as_u64().unwrap() as u32;
    let h = reference["page_height"].as_u64().unwrap() as u32;
    let mut rgb = std::fs::read(root.join("det/round2/kd0092/page.bgr")).unwrap();
    for p in rgb.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
    let png_path = dir.join("page.png");
    let mut encoder = png::Encoder::new(std::fs::File::create(&png_path).unwrap(), w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&rgb)
        .unwrap();
    let input = dir.join("in.jsonl");
    let out = dir.join("out.jsonl");
    let rows = [
        json!({"img_id":"warmup","path":png_path}),
        json!({"img_id":42,"path":png_path}),
        json!({"img_id":"failure","path":dir.join("missing.png")}),
        json!({"img_id":42,"path":dir.join("also-missing.png")}),
    ];
    std::fs::write(
        &input,
        rows.iter()
            .map(|r| r.to_string() + "\n")
            .collect::<String>(),
    )
    .unwrap();
    let run = || {
        let output = Command::new(env!("CARGO_BIN_EXE_ppocr-onednn"))
            .args(["--model"])
            .arg(&root)
            .args(["--threads", "1", "--warmup", "1", "--images"])
            .arg(&input)
            .arg("--out")
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    run();
    let before = std::fs::read(&out).unwrap();
    let text = std::str::from_utf8(&before).unwrap();
    let lines = text
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["img_id"], 42);
    assert!(lines[0]["ms"].as_f64().unwrap() > 0.);
    assert!(lines[0]["rss_mb"].as_f64().unwrap() > 0.);
    let r = |x: f64, n: f64| (x * n).round() / n;
    let expected = reference["lines"].as_array().unwrap();
    let got = lines[0]["lines"].as_array().unwrap();
    assert_eq!(got.len(), expected.len());
    for (a, b) in got.iter().zip(expected) {
        assert_eq!(a["text"], b["text"]);
        assert_eq!(
            a["score"].as_f64().unwrap(),
            r(a["score"].as_f64().unwrap(), 10000.)
        );
        let p = a["poly"].as_array().unwrap();
        assert_eq!(p.len(), 4);
        for (p, q) in p.iter().zip(b["quad"].as_array().unwrap()) {
            for j in 0..2 {
                assert_eq!(p[j].as_f64().unwrap(), r(q[j].as_f64().unwrap(), 10.));
            }
        }
    }
    assert_eq!(lines[1]["img_id"], "failure");
    assert!(lines[1]["error"].is_string());
    assert_eq!(lines[1].as_object().unwrap().len(), 2);
    run();
    assert_eq!(
        std::fs::read(&out).unwrap(),
        before,
        "resume must append nothing"
    );
    // Output without a final newline is repaired before the next append.
    let no_newline = dir.join("no-newline.jsonl");
    std::fs::write(
        &no_newline,
        "{\"img_id\":\"existing\",\"error\":\"previous\"}",
    )
    .unwrap();
    let bad_input = dir.join("errors.jsonl");
    std::fs::write(
        &bad_input,
        format!(
            "{}\n",
            json!({"img_id":"next","path":dir.join("missing.png")})
        ),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_ppocr-onednn"))
        .arg("--model")
        .arg(&root)
        .args(["--threads", "1", "--images"])
        .arg(&bad_input)
        .arg("--out")
        .arg(&no_newline)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        std::fs::read_to_string(no_newline).unwrap().lines().count(),
        2
    );

    // An explicit CLI limit reaches the model session and produces a page error.
    let limited_out = dir.join("limited.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_ppocr-onednn"))
        .arg("--model")
        .arg(&root)
        .args([
            "--threads",
            "1",
            "--workspace-limit-bytes",
            "1048576",
            "--images",
        ])
        .arg(&input)
        .arg("--out")
        .arg(&limited_out)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let limited = std::fs::read_to_string(limited_out).unwrap();
    let first: Value = serde_json::from_str(limited.lines().next().unwrap()).unwrap();
    assert!(first["error"].as_str().unwrap().contains("1048576"));
}
