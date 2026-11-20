use crate::cpu::onednn::{OnednnCpu, SessionOptions};
use crate::ppocr::net::OcrNet;
use crate::ppocr::rec::{RecOptions, Recognizer, CTC_CLASSES};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
fn backend(threads: usize) -> Arc<OnednnCpu> {
    // Finish the process-global ISA setup before another test enters oneDNN.
    static STARTUP: Mutex<()> = Mutex::new(());
    let _startup = STARTUP.lock().unwrap();
    OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::new(threads).unwrap(),
        SessionOptions {
            max_cpu_isa: crate::test_support::env::conv_isa(),
            workspace_limit: 1024 * 1024 * 1024,
            ..Default::default()
        },
    )
    .unwrap()
}
fn root() -> Option<PathBuf> {
    let p = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_assets?;
    p.join("rec/ort/1x3x48x16/index.json")
        .is_file()
        .then_some(p)
}
fn npy(p: impl AsRef<Path>) -> Vec<f32> {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    let start = 10 + u16::from_le_bytes(b[8..10].try_into().unwrap()) as usize;
    b[start..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}
#[test]
fn recognizer_parity() {
    crate::test_support::own_process!(recognizer_parity);
    let Some(root) = root() else {
        eprintln!("skipping rec parity: explicit OCR fixtures absent");
        return;
    };
    let threads = crate::test_support::env::TestOverrides::from_env()
        .unwrap()
        .ocr_threads
        .unwrap_or(1);
    let cpu = backend(threads);
    let rec =
        Recognizer::load_on(cpu.clone(), root.join("rec"), RecOptions::default(), true).unwrap();
    assert!(rec.forward(&[usize::MAX, 3, 48, 16], &[]).is_err());
    assert!(rec
        .forward(&[1, 3, 48, 16], &vec![f32::NAN; 3 * 48 * 16])
        .is_err());
    let mut hasher = Sha256::new();
    let mut count = 0;
    for (batch, width) in [(1, 16), (1, 48), (1, 320), (1, 1600), (2, 48)] {
        let dir = root.join(format!("rec/ort/{batch}x3x48x{width}"));
        let meta: Value =
            serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();
        assert_eq!(meta["onnxruntime"], "1.28.0");
        let input = npy(dir.join("input.npy"));
        let expected = npy(dir.join("output000.npy"));
        for plan in rec.warmup(batch, width).unwrap() {
            for imp in plan.implementations {
                assert!(!imp.contains("ref:"), "{imp}");
            }
        }
        let got = rec.forward(&[batch, 3, 48, width], &input).unwrap();
        assert_eq!(expected.len(), got.values.len());
        let max = got
            .values
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let argmax = |r: &[f32]| {
            let mut i = 0;
            for j in 1..r.len() {
                if r[j] > r[i] {
                    i = j;
                }
            }
            i
        };
        let mismatch = got
            .values
            .as_chunks::<CTC_CLASSES>()
            .0
            .iter()
            .zip(expected.as_chunks::<CTC_CLASSES>().0.iter())
            .filter(|(a, b)| argmax(*a) != argmax(*b))
            .count();
        println!("REC batch={batch} width={width} threads={threads} max_abs={max:e} argmax_mismatch={mismatch} floats={}",expected.len());
        if max > 1e-4 {
            let mut bad = got
                .values
                .iter()
                .zip(&expected)
                .enumerate()
                .map(|(i, (a, b))| ((a - b).abs(), i, *a, *b))
                .collect::<Vec<_>>();
            bad.sort_by(|a, b| b.0.total_cmp(&a.0));
            println!("WORST {:?}", &bad[..5]);
        }
        assert!(max <= 1e-4, "width {width}: {max}");
        assert_eq!(mismatch, 0);
        for v in got.values {
            hasher.update(v.to_le_bytes());
            count += 1;
        }
    }
    let implementations = cpu.stats().implementations;
    for imp in &implementations {
        assert!(!imp.contains("ref:"), "{imp}");
    }
    println!(
        "SVTR: {} inspected non-ref primitives",
        implementations.len()
    );
    println!("IDENTITY {} floats={count}", hex::encode(hasher.finalize()));
}
#[test]
fn recognizer_thread_identity() {
    crate::test_support::own_process!(recognizer_thread_identity);
    if root().is_none() {
        return;
    }
    let mut expected = None;
    for threads in [1, 2, 4] {
        let out = crate::test_support::run_child(
            concat!(module_path!(), "::recognizer_parity"),
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

#[test]
fn padded_attention_and_ctc_span() {
    crate::test_support::own_process!(padded_attention_and_ctc_span);
    let Some(root) = root() else {
        return;
    };
    let directory = root.join("round3/masked");
    if !directory.join("index.json").is_file() {
        return;
    }
    let cpu = backend(1);
    let options = RecOptions {
        batching: crate::ppocr::rec::Batching::Bucketed,
        ..RecOptions::default()
    };
    let rec = Recognizer::load_on(cpu, root.join("rec"), options, true).unwrap();
    let input = npy(directory.join("input.npy"));
    let expected = npy(directory.join("output.npy"));
    let output = rec
        .forward_padded(&[2, 3, 48, 48], &input, &[16, 32])
        .unwrap();
    let max = output
        .values
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("MASKED max_abs={max:e}");
    assert!(max <= 1e-4);
    let decoded = rec.decode(&output, Some(&[2, 4])).unwrap();
    assert_eq!(decoded.len(), 2);
    assert!(rec.forward_padded(&[2, 3, 48, 48], &input, &[16]).is_err());
    assert!(rec
        .forward_padded(&[2, 3, 48, 48], &input, &[16, 49])
        .is_err());
}
