//! UVDoc against ONNX Runtime 1.28 fixtures (`tools/ppocr-import/uvdoc_reference.py`)
//! under `$SINTERIC_OCR_ASSETS/uvdoc/reference`. Skips when they are absent.
use ppocr::net::OcrNet;
use ppocr::uvdoc::{Unwarper, UvdocOptions};
use rt_image::{ImageView, PixelFormat};
use rt_onednn::{CpuIsaCap, OnednnCpu, SessionOptions};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// The grid is the network's output, normalized to [-1, 1]: FP32
/// convolution reordering against MLAS stays far below this.
const GRID_ATOL: f32 = 1e-4;
/// The output samples the page at the grid's positions, so a grid error of
/// `e` moves a sample by `e * (side - 1) / 2` pixels and its value by up to
/// that times the page's steepest edge. Bounded on the mean, not the max.
const OUTPUT_MEAN_ATOL: f64 = 1e-4;

fn root() -> Option<PathBuf> {
    let p = rt_core::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets?;
    p.join("uvdoc/reference/cases.json").is_file().then_some(p)
}
fn json(p: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}
fn npy(p: impl AsRef<Path>) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    assert_eq!(b[6], 1);
    let start = 10 + u16::from_le_bytes(b[8..10].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&b[10..start]).unwrap();
    assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"));
    let shape = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let values = b[start..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (shape, values)
}
fn load(root: &Path, threads: usize) -> Unwarper {
    let cpu = {
        // Finish the process-global ISA setup before another test enters oneDNN.
        static STARTUP: Mutex<()> = Mutex::new(());
        let _startup = STARTUP.lock().unwrap();
        OnednnCpu::load_with(
            &rt_threads::ThreadBudget::new(threads).unwrap(),
            SessionOptions {
                max_cpu_isa: CpuIsaCap::for_tests(),
                workspace_limit: 1024 * 1024 * 1024,
                ..Default::default()
            },
        )
        .unwrap()
    };
    Unwarper::load(cpu, root.join("uvdoc"), UvdocOptions::default(), true).unwrap()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}

#[test]
fn uvdoc_matches_onnx_runtime() {
    let Some(root) = root() else {
        eprintln!("skipping UVDoc parity: explicit OCR fixtures absent");
        return;
    };
    let reference = root.join("uvdoc/reference");
    let threads = rt_core::env::TestOverrides::from_env()
        .unwrap()
        .ocr_threads
        .unwrap_or(1);
    let model = load(&root, threads);
    for plan in model.warmup().unwrap() {
        println!(
            "plan arena={} scratch={} primitives={}",
            plan.arena_bytes,
            plan.scratch_bytes,
            plan.implementations.len()
        );
        for imp in plan.implementations {
            println!("  {imp}");
            assert!(!imp.contains("ref:"), "{imp}");
        }
    }
    let mut hasher = Sha256::new();
    let mut count = 0usize;
    for case in json(reference.join("cases.json"))["cases"]
        .as_array()
        .unwrap()
    {
        let name = case["name"].as_str().unwrap();
        let dir = reference.join(name);
        let (shape, input) = npy(dir.join("input.npy"));
        let (grid_shape, want_grid) = npy(dir.join("grid.npy"));
        let (_, want) = npy(dir.join("output.npy"));
        let grid = model.grid(&shape, &input).unwrap();
        assert_eq!(grid.shape, grid_shape);
        let grid_err = max_abs(&grid.values, &want_grid);
        let got = model.forward(&shape, &input).unwrap();
        assert_eq!(got.shape, shape);
        let out_max = max_abs(&got.values, &want);
        let out_mean = got
            .values
            .iter()
            .zip(&want)
            .map(|(a, b)| f64::from((a - b).abs()))
            .sum::<f64>()
            / want.len() as f64;
        // The 8-bit page the pipeline sees: pack ORT's output the same way.
        let (h, w) = (shape[2], shape[3]);
        let packed = |planes: &[f32]| {
            rt_image::pack_unit_nchw(planes, w as u32, h as u32, PixelFormat::Rgb8, 1).unwrap()
        };
        let (ours, theirs) = (packed(&got.values), packed(&want));
        let differing = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        let worst_level = ours
            .iter()
            .zip(&theirs)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        for v in grid.values.iter().chain(&got.values) {
            hasher.update(v.to_le_bytes());
            count += 1;
        }
        println!(
            "{name} {w}x{h} threads={threads}: grid max_abs {grid_err:.3e}, output max_abs {out_max:.3e} \
             mean_abs {out_mean:.3e}, 8-bit {differing}/{} values differ (worst {worst_level})",
            ours.len()
        );
        assert!(grid_err <= GRID_ATOL, "{name}: grid {grid_err}");
        assert!(
            out_mean <= OUTPUT_MEAN_ATOL,
            "{name}: output mean {out_mean}"
        );
        // `unwarp` is `forward` plus 8-bit packing, from decoded pixels.
        if name != "random" {
            let rgb = packed(&input);
            let view = ImageView::new(&rgb, w as u32, h as u32, w * 3, PixelFormat::Rgb8).unwrap();
            let flat = model.unwarp(&view).unwrap();
            assert_eq!((flat.width, flat.height), (w as u32, h as u32));
            assert_eq!(flat.data, ours, "{name}: unwarp differs from forward");
        }
    }
    println!("IDENTITY {} floats={count}", hex::encode(hasher.finalize()));
}

/// The repo invariant: values depend on shape and ISA, never on threads. The
/// pool is process-global, so each count runs the parity test in its own
/// process.
#[test]
fn uvdoc_thread_identity() {
    if root().is_none() {
        return;
    }
    let mut expected = None;
    for threads in [1, 2, 4] {
        let out = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "uvdoc_matches_onnx_runtime", "--nocapture"])
            .env("PPOCR_TEST_THREADS", threads.to_string())
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
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
