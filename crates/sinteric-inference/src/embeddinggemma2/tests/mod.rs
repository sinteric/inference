//! Tests of EmbeddingGemma 2: parity with the PyTorch FP32 fixtures, the
//! media towers and the AMX tile-state guard. Each former integration test
//! is a submodule.

mod amx_tile_state;
mod multimodal;
mod parity_pinned_avx2;
mod parity_pinned_avx512;

use super::config::{is_global, DOC_PREFIX, QUERY_PREFIX};
use super::cpu::Precision;
use super::*;
use crate::cpu::onednn::{isa_has_amx, CacheOptions};
use crate::cpu::WeightSource;

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

fn run_parity(model: &Model, fe: &Tokenizer) {
    run_parity_at(model, fe, 0.9999);
}

fn run_parity_at(model: &Model, fe: &Tokenizer, gate: f32) {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/embeddinggemma2/parity.json"
    );
    let text = std::fs::read_to_string(fixture).expect("fixtures/parity.json");
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let cases = value["cases"].as_array().expect("cases");
    let mut min_cos = 1.0f32;
    let mut max_abs_bf16 = 0.0f32;
    let mut below_9999 = 0usize;
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let kind = match case["prompt"].as_str().unwrap() {
            "query" => PromptKind::Query,
            _ => PromptKind::Document,
        };
        let batch = fe.encode(&[text.to_string()], kind).unwrap();
        if let Some(want) = case["input_ids"].as_array() {
            let got = &batch.input_ids[0];
            let want: Vec<u32> = want.iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(got, &want, "tokenizer mismatch for {}", case["id"]);
        }
        let got = model.forward(&batch).unwrap();
        let fp32: Vec<f32> = case["embedding_fp32"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let cos = crate::test_support::qual::cosine(&fp32, &got[0]).unwrap();
        min_cos = min_cos.min(cos);
        if cos < 0.9999 {
            below_9999 += 1;
            eprintln!(
                "cosine {cos} < 0.9999 for {} tokens={}",
                case["id"],
                batch.input_ids[0].len()
            );
        }
        assert!(cos >= gate, "cosine {cos} < {gate} for {}", case["id"]);
        if let Some(bf) = case.get("embedding_bf16").and_then(|v| v.as_array()) {
            let bf: Vec<f32> = bf.iter().map(|v| v.as_f64().unwrap() as f32).collect();
            max_abs_bf16 =
                max_abs_bf16.max(crate::test_support::qual::max_abs_diff(&bf, &got[0]).unwrap());
        }
    }
    eprintln!(
        "eg2 parity precision={} isa={} min cosine vs FP32={min_cos} (<0.9999: {below_9999}) max|Δ| vs BF16={max_abs_bf16}",
        model.precision(),
        model.backend_stats().isa
    );
}

#[test]
fn named() {
    assert_eq!(QUERY_PREFIX, "task: search result | query: ");
    assert_eq!(DOC_PREFIX, "title: none | text: ");
    assert_eq!(MRL_DIMS, [768, 512, 256, 128]);
    assert_eq!(MAX_SEQ, 8192);
    assert!(is_global(5) && !is_global(4));
}

#[test]
fn mrl_renormalizes() {
    let v: Vec<f32> = (0..768).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let t = mrl_truncate(&v, 256);
    assert_eq!(t.len(), 256);
    let n: f32 = t.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-5, "{n}");
}

#[test]
fn load_skips_when_weights_or_onednn_absent() {
    let Some(dir) = skip_assets() else {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/embeddinggemma2.toml"
        );
        let m = crate::assets::Manifest::load(path).expect("committed asset manifest");
        assert_eq!(m.repo, "google/embeddinggemma-2");
        return;
    };
    let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
    let backend = crate::cpu::onednn::OnednnCpu::load(&budget).expect("libdnnl");
    Model::load(&dir, backend, 64).expect("weights present; oneDNN load");
}

/// The default: whatever ISA oneDNN selects (AMX on EMR, AVX512_BF16 on Turin).
#[test]
fn parity_vs_pytorch_fixtures() {
    let Some(dir) = skip_assets() else {
        return;
    };
    let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
    let backend = crate::cpu::onednn::OnednnCpu::load(&budget).expect("libdnnl");
    let model = Model::load(&dir, backend, 64).expect("load");
    let fe = Tokenizer::load(&dir).expect("tokenizer");
    run_parity(&model, &fe);
}

/// FP32 graph against the PyTorch FP32 reference: no BF16 rounding of
/// activations, so it must track the reference far tighter than BF16.
#[test]
fn parity_fp32_vs_pytorch_fixtures() {
    let Some(dir) = skip_assets() else {
        return;
    };
    let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
    let backend = crate::cpu::onednn::OnednnCpu::load(&budget).expect("libdnnl");
    let model =
        Model::load_with_precision(&dir, backend, 64, &CacheOptions::default(), Precision::Fp32)
            .expect("load");
    let fe = Tokenizer::load(&dir).expect("tokenizer");
    run_parity_at(&model, &fe, 0.99999);
}

/// Same fixtures on the live AMX kernel when the host advertises AMX.
#[test]
fn parity_amx_path_when_supported() {
    let Some(dir) = skip_assets() else {
        return;
    };
    let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
    let backend = crate::cpu::onednn::OnednnCpu::load(&budget).expect("libdnnl");
    if !isa_has_amx(backend.cpu_isa()) {
        eprintln!(
            "skipping: effective ISA 0x{:x} has no AMX",
            backend.cpu_isa()
        );
        return;
    }
    eprintln!("AMX path isa=0x{:x}", backend.cpu_isa());
    let model = Model::load(&dir, backend, 64).expect("load");
    let fe = Tokenizer::load(&dir).expect("tokenizer");
    run_parity(&model, &fe);
}

#[test]
fn packed_cache_foreign_isa_miss() {
    let Some(dir) = skip_assets() else {
        return;
    };
    let tmp = std::env::temp_dir().join(format!(
        "eg2-isa-cache-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&tmp);
    let cache = CacheOptions {
        enabled: true,
        file: Some(tmp.clone()),
        ..CacheOptions::default()
    };
    {
        let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
        let backend = crate::cpu::onednn::OnednnCpu::load(&budget).expect("libdnnl");
        let live = backend.cpu_isa();
        let model = Model::load_with(&dir, backend, 64, &cache).expect("empty-cache load");
        let fe = Tokenizer::load(&dir).expect("tokenizer");
        eprintln!(
            "empty-cache load_source={:?} cpu_isa=0x{live:x}",
            model.load_report().packed[0].source
        );
        run_parity(&model, &fe);
    }
    assert!(tmp.is_file(), "empty-cache load should write a v5 cache");
    let mut bytes = std::fs::read(&tmp).unwrap();
    // cpu_isa is at offset 56 (magic8 + ver4 + sha32 + dnnl12).
    let isa_off = 8 + 4 + 32 + 12;
    let live = u64::from_le_bytes(bytes[isa_off..isa_off + 8].try_into().unwrap());
    let foreign = if live == 0xfef { 0xe7u64 } else { 0xfef };
    bytes[isa_off..isa_off + 8].copy_from_slice(&foreign.to_le_bytes());
    std::fs::write(&tmp, &bytes).unwrap();
    {
        let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
        let backend = crate::cpu::onednn::OnednnCpu::load(&budget).expect("libdnnl");
        let isa = backend.cpu_isa();
        let model = Model::load_with(&dir, backend, 64, &cache).expect("foreign-ISA miss rebuilds");
        let fe = Tokenizer::load(&dir).expect("tokenizer");
        assert!(
            !matches!(
                model.load_report().packed[0].source,
                WeightSource::Cache { .. }
            ),
            "foreign-ISA cache must not restore under live ISA 0x{isa:x}"
        );
        run_parity(&model, &fe);
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn mixed_and_ragged_match_per_sequence() {
    mixed_and_ragged(Precision::Bf16);
}

#[test]
fn fp32_mixed_and_ragged_match_per_sequence() {
    mixed_and_ragged(Precision::Fp32);
}

fn mixed_and_ragged(precision: Precision) {
    let Some(dir) = skip_assets() else {
        return;
    };
    let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
    let backend = crate::cpu::onednn::OnednnCpu::load(&budget).expect("libdnnl");
    let model = Model::load_with_precision(&dir, backend, 64, &CacheOptions::default(), precision)
        .expect("load");
    let short_ids: Vec<u32> = (2..9).collect();
    let long_ids: Vec<u32> = (2..21).collect();
    let ones = |n| vec![1u32; n];
    let short = Batch {
        input_ids: vec![short_ids.clone()],
        attention_mask: vec![ones(short_ids.len())],
    };
    let long = Batch {
        input_ids: vec![long_ids.clone()],
        attention_mask: vec![ones(long_ids.len())],
    };
    let mut padded_short = short_ids.clone();
    padded_short.resize(long_ids.len(), 0);
    let mut short_mask = ones(short_ids.len());
    short_mask.resize(long_ids.len(), 0);
    let mixed = Batch {
        input_ids: vec![padded_short, long_ids.clone()],
        attention_mask: vec![short_mask, ones(long_ids.len())],
    };
    let ragged = Batch {
        input_ids: vec![short_ids, long_ids],
        attention_mask: vec![ones(7), ones(19)],
    };
    let a = model.forward(&short).unwrap();
    let b = model.forward(&long).unwrap();
    let padded_only = Batch {
        input_ids: vec![mixed.input_ids[0].clone()],
        attention_mask: vec![mixed.attention_mask[0].clone()],
    };
    let p = model.forward(&padded_only).unwrap();
    let m = model.forward(&mixed).unwrap();
    let r = model.forward(&ragged).unwrap();
    for (got, want, name) in [
        (&p[0], &a[0], "padded-short"),
        (&m[0], &a[0], "mixed-short"),
        (&m[1], &b[0], "mixed-long"),
        (&r[0], &a[0], "ragged-short"),
        (&r[1], &b[0], "ragged-long"),
    ] {
        let cos = crate::test_support::qual::cosine(got, want).unwrap();
        assert!(
            cos >= 0.9999,
            "{precision:?} {name} cosine {cos} (grouped vs dedicated B1)"
        );
    }
}

fn ident_row(seq: usize, seed: u32) -> (Vec<u32>, Vec<u32>) {
    let ids: Vec<u32> = (seed..(seed + seq as u32)).collect();
    let mask = vec![1u32; ids.len()];
    (ids, mask)
}

fn ident_embed_seqs(
    dir: &std::path::Path,
    threads: usize,
    pinned: bool,
    precision: Precision,
) -> Vec<(String, Vec<f32>)> {
    let budget = crate::cpu::threads::ThreadBudget::new(threads).unwrap();
    let session = crate::cpu::onednn::SessionOptions {
        max_cpu_isa: if pinned {
            crate::cpu::onednn::CpuIsaCap::Avx512CoreBf16
        } else {
            crate::cpu::onednn::CpuIsaCap::BestAvailable
        },
        ..crate::cpu::onednn::SessionOptions::default()
    };
    let backend = crate::cpu::onednn::OnednnCpu::load_with(&budget, session).expect("libdnnl");
    if pinned {
        assert_eq!(backend.strategy(), crate::cpu::onednn::IsaStrategy::Avx512);
    }
    // Packed-weight layout is probed at fixed M=64 (`select_weight_md`);
    // cache is ISA+weight, not activation M or thread count.
    let model = Model::load_with_precision(dir, backend, 256, &CacheOptions::default(), precision)
        .expect("load");
    let mut out = Vec::new();
    for batch in [1usize, 3, 5] {
        for seq in (1usize..=33).chain(if batch <= 3 { vec![129, 300] } else { vec![] }) {
            let input_ids: Vec<Vec<u32>> = (0..batch)
                .map(|bi| ident_row(seq, 2 + bi as u32 * 17).0)
                .collect();
            let attention_mask = vec![vec![1u32; seq]; batch];
            let got = model
                .forward(&Batch {
                    input_ids,
                    attention_mask,
                })
                .unwrap();
            for (i, emb) in got.into_iter().enumerate() {
                out.push((format!("b{batch}s{seq}i{i}"), emb));
            }
        }
    }
    for (tag, lens) in [
        ("mixA", vec![1usize, 15, 17, 32]),
        ("mixB", vec![1, 8, 16, 33]),
        ("mixC", vec![7, 129, 300]),
    ] {
        let mixed = Batch {
            input_ids: lens
                .iter()
                .enumerate()
                .map(|(i, &seq)| ident_row(seq, 2 + i as u32 * 17).0)
                .collect(),
            attention_mask: lens.iter().map(|&seq| vec![1u32; seq]).collect(),
        };
        let got = model.forward(&mixed).unwrap();
        for (i, emb) in got.into_iter().enumerate() {
            out.push((format!("{tag}{i}"), emb));
        }
    }
    out
}

fn ident_encode(rows: &[(String, Vec<f32>)]) -> String {
    let mut s = String::new();
    for (key, emb) in rows {
        s.push_str(key);
        s.push(':');
        for x in emb {
            s.push_str(&format!("{:08x}", x.to_bits()));
        }
        s.push(';');
    }
    s
}

fn ident_decode(line: &str) -> Vec<(String, Vec<f32>)> {
    let mut rows = Vec::new();
    for part in line.split(';').filter(|p| !p.is_empty()) {
        let (head, hex) = part.split_once(':').expect("key:hex");
        assert_eq!(hex.len() % 8, 0);
        let emb: Vec<f32> = (0..hex.len() / 8)
            .map(|i| f32::from_bits(u32::from_str_radix(&hex[i * 8..i * 8 + 8], 16).unwrap()))
            .collect();
        rows.push((head.to_string(), emb));
    }
    rows
}

/// Ranges/padding are shape+ISA only (`pad_and_ranges_are_shape_only`).
/// Embeddings across thread widths 1/2/4/8 must match at cosine >= 0.9995
/// (FP32: 0.99999) for s=1..=33, 129/300, and mixed batches (oneDNN
/// reduction is not bit-identical). Children load the pool once.
#[test]
fn embeddings_match_across_thread_counts() {
    let Some(dir) = skip_assets() else {
        return;
    };
    let env = crate::test_support::env::TestOverrides::from_env().unwrap();
    if let Some(t) = env.ident_threads {
        let precision = if env.ident_fp32.unwrap_or(false) {
            Precision::Fp32
        } else {
            Precision::Bf16
        };
        let rows = ident_embed_seqs(&dir, t, env.ident_avx512.expect("child ISA"), precision);
        println!("EG2_IDENT {}", ident_encode(&rows));
        return;
    }
    // FP32 never uses AMX, so the pinned AVX-512 run covers it.
    for (isa, precision) in [("native", "bf16"), ("avx512", "bf16"), ("avx512", "fp32")] {
        let gate = if precision == "fp32" { 0.99999 } else { 0.9995 };
        if isa == "native" {
            let cpus = crate::cpu::threads::physical_cpu_ids().unwrap_or_default();
            if !crate::test_support::amx::amx_tile_state_is_preserved(&cpus) {
                continue;
            }
        }
        let mut references: Vec<(usize, Vec<(String, Vec<f32>)>)> = Vec::new();
        let mut min_cross = 1.0f32;
        for t in [1usize, 2, 4, 8] {
            if crate::cpu::threads::ThreadBudget::new(t).is_err() {
                eprintln!("skip t={t}: not enough physical cores");
                continue;
            }
            let out = crate::test_support::run_child(
                concat!(module_path!(), "::embeddings_match_across_thread_counts"),
                &[
                    ("EG2_IDENT_THREADS", t.to_string()),
                    ("EG2_IDENT_ISA", isa.into()),
                    ("EG2_IDENT_PRECISION", precision.into()),
                ],
            );
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let line = stdout
                .lines()
                .find_map(|l| l.split_once("EG2_IDENT ").map(|(_, data)| data))
                .unwrap_or_else(|| {
                    panic!("no EG2_IDENT in child t={t} stdout={stdout} stderr={stderr}")
                });
            let rows = ident_decode(line.trim());
            for (previous_threads, expected) in &references {
                assert_eq!(expected.len(), rows.len());
                for ((k0, a), (k1, b)) in expected.iter().zip(&rows) {
                    assert_eq!(k0, k1);
                    let cos = crate::test_support::qual::cosine(a, b).unwrap();
                    min_cross = min_cross.min(cos);
                    assert!(
                        cos >= gate,
                        "{k0} {isa} {precision} cosine {cos} between {previous_threads} and {t} threads"
                    );
                }
            }
            references.push((t, rows));
        }
        eprintln!("EG2 isa={isa} precision={precision} min cross-thread cosine={min_cross}");
    }
}
