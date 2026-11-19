//! Explicit AVX512_CORE_BF16 opt-in. Isolated process so it cannot steal the
//! native-ISA lib tests (AMX on EMR). Used for Zen5-emulation benches/tests.

#[path = "../../../crates/rt-assets/tests/support/mod.rs"]
mod asset_test_support;

use embeddinggemma2::{Frontend, Model, PromptKind};
use rt_backend::Hardware;
use rt_onednn::{CpuIsaCap, SessionOptions};

fn skip_assets() -> Option<std::path::PathBuf> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/embeddinggemma2.toml"
    );
    let m = rt_assets::Manifest::load(path).ok()?;
    let root = asset_test_support::asset_root(&m)?;
    let st = rt_assets::locate(&m, &root, "model.safetensors").ok()?;
    let tok = rt_assets::locate(&m, &root, "tokenizer.json").ok()?;
    if rt_assets::skip_if_missing(&st) || rt_assets::skip_if_missing(&tok) {
        return None;
    }
    let rt_assets::AssetStatus::Ready(p) = st else {
        return None;
    };
    Some(p.parent().unwrap().to_path_buf())
}

#[test]
fn parity_pinned_avx512_core_bf16() {
    let Some(dir) = skip_assets() else {
        return;
    };
    let budget = rt_threads::ThreadBudget::new(1).unwrap();
    let session = SessionOptions {
        max_cpu_isa: CpuIsaCap::Avx512CoreBf16,
        ..SessionOptions::default()
    };
    let backend = rt_onednn::OnednnCpu::load_with(&budget, session).expect("libdnnl");
    if backend.cpu_isa() != 0xe7 {
        eprintln!(
            "skipping: effective ISA 0x{:x} is not AVX512_CORE_BF16",
            backend.cpu_isa()
        );
        return;
    }
    let model = Model::load(&dir, backend, 64).expect("load");
    let fe = Frontend::load(&dir).expect("tokenizer");
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/parity.json");
    let text = std::fs::read_to_string(fixture).expect("fixtures/parity.json");
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let cases = value["cases"].as_array().expect("cases");
    let mut min_cos = 1.0f32;
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let kind = match case["prompt"].as_str().unwrap() {
            "query" => PromptKind::Query,
            _ => PromptKind::Document,
        };
        let batch = fe.encode(&[text.to_string()], kind).unwrap();
        let got = model.forward(&batch).unwrap();
        let fp32: Vec<f32> = case["embedding_fp32"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let cos = rt_qual::cosine(&fp32, &got[0]).unwrap();
        min_cos = min_cos.min(cos);
        assert!(cos >= 0.9999, "cosine {cos} < 0.9999 for {}", case["id"]);
    }
    eprintln!("pinned AVX512_CORE_BF16 min cosine vs FP32={min_cos}");
}
