//! Explicit AVX512_CORE_BF16 opt-in. Isolated process so it cannot steal the
//! native-ISA lib tests (AMX on EMR). Used for Zen5-emulation benches/tests.

use crate::cpu::onednn::{CpuIsaCap, SessionOptions};
use crate::embeddinggemma2::{Frontend, Model, PromptKind};

fn skip_assets() -> Option<std::path::PathBuf> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/embeddinggemma2.toml"
    );
    let m = crate::assets::Manifest::load(path).ok()?;
    let root = crate::test_support::assets::asset_root(&m)?;
    let st = crate::test_support::assets::locate(&m, &root, "model.safetensors").ok()?;
    let tok = crate::test_support::assets::locate(&m, &root, "tokenizer.json").ok()?;
    if crate::test_support::assets::skip_if_missing(&st)
        || crate::test_support::assets::skip_if_missing(&tok)
    {
        return None;
    }
    let crate::test_support::assets::AssetStatus::Ready(p) = st else {
        return None;
    };
    Some(p.parent().unwrap().to_path_buf())
}

#[test]
fn parity_pinned_avx512_core_bf16() {
    crate::test_support::own_process!(parity_pinned_avx512_core_bf16);
    let Some(dir) = skip_assets() else {
        return;
    };
    let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
    let session = SessionOptions {
        max_cpu_isa: CpuIsaCap::Avx512CoreBf16,
        ..SessionOptions::default()
    };
    let backend = crate::cpu::onednn::OnednnCpu::load_with(&budget, session).expect("libdnnl");
    if backend.cpu_isa() != 0xe7 {
        eprintln!(
            "skipping: effective ISA 0x{:x} is not AVX512_CORE_BF16",
            backend.cpu_isa()
        );
        return;
    }
    let model = Model::load(&dir, backend, 64).expect("load");
    let fe = Frontend::load(&dir).expect("tokenizer");
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/embeddinggemma2/parity.json"
    );
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
        let cos = crate::test_support::qual::cosine(&fp32, &got[0]).unwrap();
        min_cos = min_cos.min(cos);
        assert!(cos >= 0.9999, "cosine {cos} < 0.9999 for {}", case["id"]);
    }
    eprintln!("pinned AVX512_CORE_BF16 min cosine vs FP32={min_cos}");
}
