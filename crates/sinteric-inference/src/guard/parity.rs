//! The guard against its PyTorch FP32 reference (`fixtures/guard/parity.json`,
//! written by `generate_parity_fixtures.py` beside it): token ids exactly,
//! the logit, the score and checksums of the final hidden states. Each test
//! runs in its own process, because a [`Runtime`] sets the process-global
//! ISA cap.
//!
//! ⚠ The AVX2 test caps oneDNN on an AVX-512 host: it runs the AVX2 code
//! paths, not an AVX2 CPU. The fixtures cover eight texts in two languages;
//! they cannot show a drift that only some other text would trigger.

use std::path::PathBuf;

use serde_json::Value;

use super::cpu::{Config, Guard, Precision};
use crate::cpu::{Isa, IsaCap, Options, Runtime};

/// The asset directory, or `None` (the test is skipped) when the pinned
/// files are not in the cache.
fn model_dir() -> Option<PathBuf> {
    let m = crate::assets::Manifest::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/guard.toml"
    ))
    .expect("committed asset manifest");
    let root = crate::test_support::assets::asset_root(&m)?;
    let mut dir = None;
    for file in ["model.safetensors", "tokenizer.json", "config.json"] {
        let status = crate::test_support::assets::locate(&m, &root, file).ok()?;
        if crate::test_support::assets::skip_if_missing(&status) {
            return None;
        }
        let crate::test_support::assets::AssetStatus::Ready(p) = status else {
            return None;
        };
        dir = p.parent().map(PathBuf::from);
    }
    dir
}

fn fixtures() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/guard/parity.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).unwrap()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// Largest gaps from the reference over every case.
#[derive(Debug, Default)]
struct Gaps {
    logit: f32,
    score: f32,
    hidden_l2_rel: f64,
    hidden_head: f32,
}

/// Run every fixture case through `guard`, assert token ids and decisions,
/// and return the largest gaps.
fn compare(guard: &Guard) -> Gaps {
    let value = fixtures();
    assert_eq!(
        value["max_len"].as_u64().unwrap() as usize,
        guard.max_len(),
        "fixture max_len"
    );
    let cases = value["cases"].as_array().unwrap();
    let texts: Vec<&str> = cases.iter().map(|c| c["text"].as_str().unwrap()).collect();
    let got = guard.classify(&texts).unwrap();
    let mut gaps = Gaps::default();
    for ((case, injection), text) in cases.iter().zip(&got).zip(&texts) {
        let id = case["id"].as_str().unwrap();
        let want_ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(injection.tokens, want_ids.len(), "{id}: token count");
        assert_eq!(
            injection.truncated,
            case["truncated"].as_bool().unwrap(),
            "{id}: truncated"
        );
        let want_logit = case["logit"].as_f64().unwrap() as f32;
        let want_score = case["score"].as_f64().unwrap() as f32;
        assert_eq!(
            injection.logit > 0.0,
            want_logit > 0.0,
            "{id}: decision flipped (logit {} vs {want_logit})",
            injection.logit
        );
        gaps.logit = gaps.logit.max((injection.logit - want_logit).abs());
        gaps.score = gaps.score.max((injection.score - want_score).abs());

        // The ids classify encoded: the full tokenization when it fits.
        let tokens = guard.tokenize(text).unwrap();
        if !injection.truncated {
            assert_eq!(tokens.ids, want_ids, "{id}: token ids");
        } else {
            let n = want_ids.len();
            assert_eq!(tokens.ids[..n - 1], want_ids[..n - 1], "{id}: head ids");
            assert_eq!(tokens.ids.last(), want_ids.last(), "{id}: last id");
        }
        let hidden = guard.encode(&[&want_ids]).unwrap();
        let row = hidden.row(0);
        let l2 = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>();
        let want_l2 = case["hidden_l2"].as_f64().unwrap();
        gaps.hidden_l2_rel = gaps
            .hidden_l2_rel
            .max((l2.sqrt() - want_l2).abs() / want_l2);
        let head = floats(&case["hidden_head"]);
        let diff = crate::test_support::qual::max_abs_diff(&head, &row[..head.len()]).unwrap();
        gaps.hidden_head = gaps.hidden_head.max(diff);
        // The library's head over the same hidden states gives classify's logit.
        assert_eq!(
            guard.injection_logit(row).unwrap(),
            injection.logit,
            "{id}: injection_logit vs classify"
        );
    }
    gaps
}

fn load(cap: IsaCap, precision: Precision) -> Option<(Runtime, Guard)> {
    let dir = model_dir()?;
    let rt = Runtime::new(Options::new().with_threads(2).with_isa_cap(cap)).unwrap();
    if !Guard::precisions(&rt).contains(&precision) {
        eprintln!("skipping: {precision} does not run on {}", rt.isa());
        return None;
    }
    let guard = Guard::load(&rt, Config::new(dir).with_precision(precision)).unwrap();
    Some((rt, guard))
}

/// FP32 at the best ISA the host has: the reference within float
/// reassociation.
#[test]
fn fp32_matches_pytorch() {
    crate::test_support::own_process!(fp32_matches_pytorch);
    let Some((rt, guard)) = load(IsaCap::Best, Precision::Fp32) else {
        return;
    };
    let gaps = compare(&guard);
    eprintln!("guard fp32 {}: {gaps:?}", rt.isa());
    assert_fp32(&gaps);
}

/// FP32 under an AVX2 cap: what a host without AVX-512 runs.
#[test]
fn fp32_matches_pytorch_pinned_avx2() {
    crate::test_support::own_process!(fp32_matches_pytorch_pinned_avx2);
    let Some((rt, guard)) = load(IsaCap::Avx2, Precision::Fp32) else {
        return;
    };
    if rt.isa() != Isa::Avx2 {
        eprintln!("skipping: effective ISA {} is not AVX2", rt.isa());
        return;
    }
    let gaps = compare(&guard);
    eprintln!("guard fp32 AVX2: {gaps:?}");
    assert_fp32(&gaps);
}

/// BF16 matmuls: looser, but every decision and token count still matches.
#[test]
fn bf16_keeps_every_decision() {
    crate::test_support::own_process!(bf16_keeps_every_decision);
    let Some((rt, guard)) = load(IsaCap::Best, Precision::Bf16) else {
        return;
    };
    let gaps = compare(&guard);
    eprintln!("guard bf16 {}: {gaps:?}", rt.isa());
    assert!(gaps.logit <= BF16_LOGIT, "{gaps:?}");
    assert!(gaps.score <= BF16_SCORE, "{gaps:?}");
}

// Measured on an AMX host (2026-11-20): FP32 logit 5.7e-6, score 1.5e-7,
// hidden L2 4.5e-8 relative, first values 4.8e-7 (AVX2 cap: 6.7e-6, 1.6e-7,
// 6.9e-8, 3.0e-7); BF16 logit 0.035, score 9.9e-5. The gates leave about 20x
// for another CPU's float reassociation.
const FP32_LOGIT: f32 = 1e-4;
const FP32_SCORE: f32 = 1e-5;
const FP32_HIDDEN_L2_REL: f64 = 1e-6;
const FP32_HIDDEN_HEAD: f32 = 1e-5;
const BF16_LOGIT: f32 = 0.25;
const BF16_SCORE: f32 = 0.01;

fn assert_fp32(gaps: &Gaps) {
    assert!(gaps.logit <= FP32_LOGIT, "{gaps:?}");
    assert!(gaps.score <= FP32_SCORE, "{gaps:?}");
    assert!(gaps.hidden_l2_rel <= FP32_HIDDEN_L2_REL, "{gaps:?}");
    assert!(gaps.hidden_head <= FP32_HIDDEN_HEAD, "{gaps:?}");
}
