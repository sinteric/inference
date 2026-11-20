use crate::cpu::runtime::{BackendOverrides, IsaCap, Options, Runtime};
use crate::image::{ImageView, PixelFormat, Quad};
use crate::ppocr::cpu::{DetConfig, Detector, RecConfig, Recognizer};
use serde_json::Value;
use std::path::{Path, PathBuf};
fn root() -> Option<PathBuf> {
    let p = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets?;
    p.join("round3/reference-index.json").is_file().then_some(p)
}
fn json(p: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn npy(p: impl AsRef<Path>) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    let start = 10 + u16::from_le_bytes(b[8..10].try_into().unwrap()) as usize;
    b[start..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}
fn quad(v: &Value) -> Quad {
    std::array::from_fn(|i| std::array::from_fn(|j| v[i][j].as_f64().unwrap() as f32))
}
#[test]
fn crop_pixel_parity() {
    crate::test_support::own_process!(crop_pixel_parity);
    let Some(root) = root() else {
        eprintln!("skipping crop parity: explicit OCR fixtures absent");
        return;
    };
    let mut samples = 0;
    let mut differences = 0;
    let mut worst = 0f32;
    let mut check = |image: &ImageView<'_>, q: Quad, expected: Vec<f32>, label: String| {
        let crop = crate::image::recognizer_crop(image, q, 16, 1600, 64 * 1024 * 1024).unwrap();
        assert_eq!(crop.values.len(), expected.len(), "{label}");
        let max = crop
            .values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let bad = crop
            .values
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a != b)
            .count();
        if bad > 0 {
            println!("CROP {label} max={max} different={bad}/{}", expected.len());
        }
        worst = worst.max(max);
        differences += bad;
        samples += expected.len();
    };
    let bytes = std::fs::read(root.join("round3/crop-oracle.bgr")).unwrap();
    let image = ImageView::try_new(&bytes, 63, 47, 63 * 3, PixelFormat::Bgr8).unwrap();
    for (i, c) in json(root.join("round3/crop-oracle.json"))
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let q = quad(&c["quad"]);
        let mut warp = crate::image::warp_quad_bgr(&image, q, 64 * 1024 * 1024).unwrap();
        if warp.height as f64 >= 1.5 * warp.width as f64 {
            warp = crate::image::rotate_ccw_bgr(&warp.view().unwrap()).unwrap();
        }
        let expected = std::fs::read(root.join(format!("round3/oracle{i:03}.bgr"))).unwrap();
        let bad = warp
            .data
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a != b)
            .count();
        if bad > 0 {
            println!(
                "WARP oracle {i}: different={bad}/{} dims={}x{}",
                expected.len(),
                warp.width,
                warp.height
            );
        }
        assert_eq!(warp.data.len(), expected.len(), "oracle {i}");
        check(
            &image,
            q,
            npy(root.join(format!("round3/oracle{i:03}.npy"))),
            format!("oracle {i}"),
        );
    }
    for name in ["kd0092", "kd0330", "kd0444", "kd0822"] {
        let dir = root.join("round3").join(name);
        let reference = json(dir.join("reference.json"));
        let bytes = std::fs::read(root.join("det/round2").join(name).join("page.bgr")).unwrap();
        let w = reference["page_width"].as_u64().unwrap() as u32;
        let h = reference["page_height"].as_u64().unwrap() as u32;
        let image = ImageView::try_new(&bytes, w, h, w as usize * 3, PixelFormat::Bgr8).unwrap();
        for (i, c) in reference["crops"].as_array().unwrap().iter().enumerate() {
            check(
                &image,
                quad(&c["quad"]),
                npy(dir.join(c["file"].as_str().unwrap())),
                format!("{name}/{i}"),
            );
        }
    }
    println!("CROP TOTAL samples={samples} differences={differences} max_abs={worst}");
    assert_eq!(differences, 0);
}
fn edit_distance(a: &str, b: &str) -> usize {
    let b = b.chars().collect::<Vec<_>>();
    let mut row = (0..=b.len()).collect::<Vec<_>>();
    for (i, ca) in a.chars().enumerate() {
        let mut old = row[0];
        row[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let prev = row[j + 1];
            row[j + 1] = (old + usize::from(ca != cb)).min(row[j] + 1).min(prev + 1);
            old = prev;
        }
    }
    row[b.len()]
}
/// The reference's recognition score threshold.
const DROP_SCORE: f32 = 0.5;
struct Line {
    quad: Quad,
    text: String,
    score: f32,
}
/// Detector then recognizer on one shared backend, lines below the
/// reference's score threshold dropped: the reference's Flat page OCR.
#[test]
fn det_then_rec_parity() {
    crate::test_support::own_process!(det_then_rec_parity);
    let Some(root) = root() else {
        eprintln!("skipping det+rec parity: explicit OCR fixtures absent");
        return;
    };
    let rt = Runtime::try_new(
        Options::new()
            .with_threads(1)
            .with_isa_cap(IsaCap::Avx512CoreBf16),
    )
    .unwrap();
    let shared = rt
        .shared_backend(
            BackendOverrides::default()
                .with_workspace_limit_bytes(Some(crate::ppocr::cpu::DEFAULT_WORKSPACE_LIMIT_BYTES)),
        )
        .unwrap();
    let det = Detector::load(
        &shared,
        DetConfig::new(root.join("det")).with_verify_weights(true),
    )
    .unwrap();
    let rec = Recognizer::load(
        &shared,
        RecConfig::new(root.join("rec")).with_verify_weights(true),
    )
    .unwrap();
    let mut comparisons = Vec::new();
    let mut all_exact = 0;
    let mut all_lines = 0;
    for name in ["kd0092", "kd0330", "kd0444", "kd0822"] {
        let reference = json(root.join("round3").join(name).join("reference.json"));
        let bytes = std::fs::read(root.join("det/round2").join(name).join("page.bgr")).unwrap();
        let w = reference["page_width"].as_u64().unwrap() as u32;
        let h = reference["page_height"].as_u64().unwrap() as u32;
        let image = ImageView::try_new(&bytes, w, h, w as usize * 3, PixelFormat::Bgr8).unwrap();
        let quads = det.detect(&image).unwrap();
        let lines = rec
            .recognize(&image, &quads)
            .unwrap()
            .into_iter()
            .zip(&quads)
            .filter(|(l, _)| l.score >= DROP_SCORE)
            .map(|(l, &quad)| Line {
                quad,
                text: l.text,
                score: l.score,
            })
            .collect::<Vec<_>>();
        let expected = reference["lines"].as_array().unwrap();
        let mut exact = 0;
        let mut chars = 0;
        let mut edits = 0;
        let mut matched = 0;
        let mut worst = 0f32;
        let mut used = vec![false; expected.len()];
        for line in &lines {
            let best = expected
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .min_by(|(_, a), (_, b)| {
                    let distance = |v: &Value| {
                        line.quad
                            .iter()
                            .flatten()
                            .zip(quad(&v["quad"]).iter().flatten())
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f32>()
                    };
                    distance(a).total_cmp(&distance(b))
                });
            if let Some((j, e)) = best {
                if line.quad == quad(&e["quad"]) {
                    matched += 1;
                    used[j] = true;
                    let text = e["text"].as_str().unwrap();
                    if text == line.text {
                        exact += 1;
                    } else {
                        println!("TEXT {name}: {:?} vs {:?}", line.text, text);
                    }
                    chars += text.chars().count();
                    edits += edit_distance(&line.text, text);
                    worst = worst.max((line.score - e["score"].as_f64().unwrap() as f32).abs());
                }
            }
        }
        let item = serde_json::json!({"img_id":name,"ours_lines":lines.len(),"ort_lines":expected.len(),"matched_lines":matched,"exact_text_lines":exact,"exact_text_match_rate":exact as f64/matched.max(1) as f64,"reference_characters":chars,"character_edits":edits,"cer":edits as f64/chars.max(1) as f64,"minimum_box_iou":if matched>0 {1.}else{0.},"max_score_abs":worst});
        println!("E2E {item}");
        comparisons.push(item);
        all_exact += exact;
        all_lines += expected.len();
    }
    let output = serde_json::json!({"onnxruntime":"1.28.0","pages":comparisons,"total_reference_lines":all_lines,"total_exact_text_lines":all_exact,"reference":"Python ORT nets + independent cv2/pyclipper pre/post; Exact identical-width batching"});
    let path = root.parent().unwrap().join("parity/phase-a-e2e.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_string_pretty(&output).unwrap() + "\n").unwrap();
    assert_eq!(all_exact, all_lines);
    for c in &comparisons {
        assert_eq!(c["ours_lines"], c["ort_lines"]);
        assert_eq!(c["matched_lines"], c["ort_lines"]);
        assert!(c["max_score_abs"].as_f64().unwrap() <= 1e-3);
    }
}
