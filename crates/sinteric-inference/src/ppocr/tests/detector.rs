use crate::cpu::onednn::{OnednnCpu, SessionOptions};
use crate::image::{ImageSize, ImageView, PixelFormat, Quad};
use crate::ppocr::det::{self, DetOptions, Detector};
use crate::ppocr::net::OcrNet;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
fn root() -> Option<PathBuf> {
    let root = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets?;
    root.join("det/round2/geometry.json")
        .is_file()
        .then(|| root.join("det/round2"))
}
fn json(p: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn quad(v: &Value) -> Quad {
    std::array::from_fn(|i| {
        let p = floats(&v[i]);
        [p[0], p[1]]
    })
}
fn points(v: &Value) -> Vec<[f32; 2]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let p = floats(v);
            [p[0], p[1]]
        })
        .collect()
}
fn npy(p: impl AsRef<Path>) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    assert_eq!(b[6], 1);
    let start = 10 + u16::from_le_bytes(b[8..10].try_into().unwrap()) as usize;
    assert!(std::str::from_utf8(&b[10..start])
        .unwrap()
        .contains("'<f4'"));
    b[start..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}
fn canonical(p: &[[f32; 2]]) -> Vec<[i32; 2]> {
    let mut p = p
        .iter()
        .map(|p| [p[0] as i32, p[1] as i32])
        .collect::<Vec<_>>();
    p.sort();
    p
}
#[test]
fn image_and_geometry_reference() {
    crate::test_support::own_process!(image_and_geometry_reference);
    let Some(root) = root() else {
        eprintln!("skipping round2 geometry: explicit OCR fixtures absent");
        return;
    };
    for c in json(root.join("resize.json")).as_array().unwrap() {
        let w = c["width"].as_u64().unwrap() as u32;
        let h = c["height"].as_u64().unwrap() as u32;
        let input = c["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        let output = c["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        let image = ImageView::new(&input, w, h, w as usize * 3, PixelFormat::Bgr8).unwrap();
        assert_eq!(
            crate::image::resize_triangle_bgr(
                &image,
                c["output_width"].as_u64().unwrap() as u32,
                c["output_height"].as_u64().unwrap() as u32
            )
            .unwrap(),
            output
        );
    }
    let cases = json(root.join("geometry.json"));
    for (i, c) in cases["contours"].as_array().unwrap().iter().enumerate() {
        let mask = c["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect::<Vec<_>>();
        let got = crate::image::contours(
            &mask,
            c["width"].as_u64().unwrap() as usize,
            c["height"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let expected = c["contours"]
            .as_array()
            .unwrap()
            .iter()
            .map(points)
            .collect::<Vec<_>>();
        assert_eq!(got.len(), expected.len(), "contour case {i}");
        for (j, (a, b)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(canonical(a), canonical(b), "contour case {i}/{j}");
        }
    }
    let mut worst_score = 0f32;
    let mut worst_rect = 1f64;
    for (i, c) in cases["rectangles"].as_array().unwrap().iter().enumerate() {
        let q = quad(&c["quad"]);
        let got = crate::image::unclip(q, 1.4).unwrap();
        assert_eq!(
            canonical(&crate::image::convex_hull(&got).unwrap()),
            canonical(&crate::image::convex_hull(&points(&c["expanded"])).unwrap()),
            "unclip {i}"
        );
        let got = crate::image::min_area_rect(&got).unwrap().unwrap().0;
        let overlap = iou(got, quad(&c["rect"]));
        worst_rect = worst_rect.min(overlap);
        assert!(overlap >= 0.9999, "rectangle {i} overlap {overlap} {got:?}");
        let score = crate::image::polygon_mean_score(&floats(&c["prob"]), 160, 160, q).unwrap();
        let delta = (score - c["score"].as_f64().unwrap() as f32).abs();
        worst_score = worst_score.max(delta);
        assert!(delta <= 1e-6, "score {i} {delta}");
    }
    println!("GEOMETRY: 128 random masks, 128 offsets/rectangles/scores, 6 resizes; score max_abs={worst_score:e}, rect min_iou={worst_rect}");
}
fn area(p: &[[f64; 2]]) -> f64 {
    (0..p.len())
        .map(|i| p[i][0] * p[(i + 1) % p.len()][1] - p[i][1] * p[(i + 1) % p.len()][0])
        .sum::<f64>()
        .abs()
        * 0.5
}
fn iou(a: Quad, b: Quad) -> f64 {
    let mut p = a.map(|p| p.map(f64::from)).to_vec();
    let b = b.map(|p| p.map(f64::from));
    let aa = area(&p);
    let ab = area(&b);
    for i in 0..4 {
        let u = b[i];
        let v = b[(i + 1) % 4];
        let side = |p: [f64; 2]| (v[0] - u[0]) * (p[1] - u[1]) - (v[1] - u[1]) * (p[0] - u[0]);
        let input = std::mem::take(&mut p);
        if input.is_empty() {
            return 0.;
        }
        for j in 0..input.len() {
            let s = input[j];
            let e = input[(j + 1) % input.len()];
            let ds = side(s);
            let de = side(e);
            if (ds >= 0.) != (de >= 0.) {
                let t = ds / (ds - de);
                p.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
            }
            if de >= 0. {
                p.push(e);
            }
        }
    }
    let intersection = area(&p);
    intersection / (aa + ab - intersection)
}
#[test]
fn detector_parity() {
    crate::test_support::own_process!(detector_parity);
    let Some(root) = root() else {
        eprintln!("skipping round2 detector: explicit OCR fixtures absent");
        return;
    };
    let threads = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_threads
        .unwrap_or(1);
    let cpu = OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::new(threads).unwrap(),
        SessionOptions {
            max_cpu_isa: crate::test_support::env::conv_isa(),
            workspace_limit: 1024 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let det = Detector::load_on(cpu, root.parent().unwrap(), DetOptions::default(), true).unwrap();
    assert!(det.forward(&[1, 3, 1, 1], &[0.; 3]).is_err());
    assert!(det.forward(&[usize::MAX, 3, 32, 32], &[]).is_err());
    assert!(det
        .forward(&[1, 3, 32, 32], &vec![f32::NAN; 3 * 32 * 32])
        .is_err());
    let mut dirs = std::fs::read_dir(&root)
        .unwrap()
        .map(|d| d.unwrap().path())
        .filter(|p| p.join("index.json").is_file())
        .collect::<Vec<_>>();
    dirs.sort_by_key(|p| {
        (
            std::fs::metadata(p.join("input.npy")).unwrap().len(),
            p.clone(),
        )
    });
    assert_eq!(dirs.len(), 8);
    let mut hasher = Sha256::new();
    for directory in dirs {
        let name = directory.file_name().unwrap().to_string_lossy();
        let meta = json(directory.join("index.json"));
        assert_eq!(meta["onnxruntime"], "1.28.0");
        let shape = meta["input"]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let input = npy(directory.join("input.npy"));
        let expected = npy(directory.join("output000.npy"));
        let info = det.warmup(shape[2], shape[3]).unwrap();
        if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx512f") {
            for imp in &info.implementations {
                assert!(!imp.contains("ref:"), "{imp}");
            }
        }
        let output = det.forward(&shape, &input).unwrap();
        assert_eq!(output.values.len(), expected.len());
        let max = output
            .values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mismatches = output
            .values
            .iter()
            .zip(&expected)
            .filter(|(a, b)| (**a > 0.2) != (**b > 0.2))
            .count();
        let fraction = mismatches as f64 / expected.len() as f64;
        println!("DET {name} threads={threads}: max_abs={max:e}, mismatch={mismatches}/{} ({fraction:e}), arena={}, scratch={}, primitives={}",expected.len(),info.arena_bytes,info.scratch_bytes,info.implementations.len());
        assert!(max <= 1e-4, "{name} tensor drift {max}");
        assert!(fraction <= 1e-5, "{name} threshold drift");
        for &v in &output.values {
            hasher.update(v.to_le_bytes());
        }
        if directory.join("page.bgr").is_file() {
            let reference = json(directory.join("reference.json"));
            let w = reference["page_width"].as_u64().unwrap() as u32;
            let h = reference["page_height"].as_u64().unwrap() as u32;
            let bytes = std::fs::read(directory.join("page.bgr")).unwrap();
            let image = ImageView::new(&bytes, w, h, w as usize * 3, PixelFormat::Bgr8).unwrap();
            let prepared = det.preprocess(&image).unwrap();
            assert_eq!(prepared.shape, shape);
            assert_eq!(prepared.values, input, "{name} preprocessing");
            let boxes = det::postprocess(
                &output.values,
                shape[3],
                shape[2],
                ImageSize {
                    width: w,
                    height: h,
                },
                det.options(),
            )
            .unwrap();
            let oracle_boxes = det::postprocess(
                &expected,
                shape[3],
                shape[2],
                ImageSize {
                    width: w,
                    height: h,
                },
                det.options(),
            )
            .unwrap();
            let expected = reference["boxes"]
                .as_array()
                .unwrap()
                .iter()
                .map(quad)
                .collect::<Vec<_>>();
            let compare = |label: &str, boxes: &[Quad]| {
                assert_eq!(boxes.len(), expected.len(), "{name} {label} box count");
                let mut used = vec![false; expected.len()];
                let mut worst = 1f64;
                for (i, &a) in boxes.iter().enumerate() {
                    let (j, overlap) = expected
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| !used[*j])
                        .map(|(j, &b)| (j, iou(a, b)))
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap();
                    used[j] = true;
                    worst = worst.min(overlap);
                    assert!(
                        overlap >= 0.99,
                        "{name} {label} box {i} best {j} IoU {overlap}: {a:?} vs {:?}",
                        expected[j]
                    );
                }
                println!(
                    "BOX {name} {label}: {}/{} matched, min_iou={worst}",
                    boxes.len(),
                    expected.len()
                );
            };
            compare("Rust/ORT map", &oracle_boxes);
            compare("Rust/oneDNN map", &boxes);
            // Exercise the decoded-image public path on one page; the remaining
            // cases independently compare each stage without duplicate inference.
            if name == "kd0092" {
                assert_eq!(det.detect(&image).unwrap(), boxes);
            }
        }
    }
    println!("IDENTITY {}", hex::encode(hasher.finalize()));
}
#[test]
fn detector_thread_identity() {
    crate::test_support::own_process!(detector_thread_identity);
    if root().is_none() {
        return;
    }
    let mut expected = None;
    for threads in [1, 2, 4] {
        let out = crate::test_support::run_child(
            concat!(module_path!(), "::detector_parity"),
            &[("PPOCR_TEST_THREADS", threads.to_string())],
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let hash = stdout
            .lines()
            .find(|l| l.starts_with("IDENTITY "))
            .unwrap()
            .to_owned();
        if let Some(e) = &expected {
            assert_eq!(e, &hash, "threads {threads}");
        } else {
            expected = Some(hash);
        }
        println!("threads {threads} {}", expected.as_ref().unwrap());
    }
}

#[cfg(feature = "debug-dumps")]
#[test]
#[ignore]
fn dump_means() {
    crate::test_support::own_process!(dump_means);
    let Some(root) = root() else {
        eprintln!("skipping prefix dumps: OCR fixtures absent");
        return;
    };
    let cpu = OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::new(1).unwrap(),
        SessionOptions {
            max_cpu_isa: crate::test_support::env::conv_isa(),
            workspace_limit: 1024 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let det = Detector::load_on(cpu, root.parent().unwrap(), DetOptions::default(), true).unwrap();
    let dir = root.join("kd0092");
    let meta = json(dir.join("index.json"));
    let shape: [usize; 4] =
        std::array::from_fn(|i| meta["input"]["shape"][i].as_u64().unwrap() as usize);
    let input = npy(dir.join("input.npy"));
    for count in [8, 9, 22, 35, 45, 58, 68, 73, 78, 83, 95, 101, 107, 113] {
        let t = det.debug_prefix(count, shape, &input).unwrap();
        let bytes = t
            .values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(dir.join(format!("prefix{count}.f32")), bytes).unwrap();
        println!("PREFIX {count} {:?}", t.shape);
    }
}
