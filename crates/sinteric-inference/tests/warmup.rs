//! Exercise opt-in warmup through the library API with pinned local assets.
#[path = "../../rt-assets/tests/support/mod.rs"]
mod asset_test_support;

use sinteric_inference::embeddinggemma2::{Batch, Config, EmbeddingGemma2, PromptKind, Tokenizer};
use sinteric_inference::{Error, Runtime, RuntimeOptions};

struct TensorOnly;
impl Tokenizer for TensorOnly {
    fn encode(&self, _: &[String], _: PromptKind) -> sinteric_inference::Result<Batch> {
        Err(Error::Config(
            "tensor-only tokenizer must not be used by warmup".into(),
        ))
    }
}

fn asset_dir() -> Option<std::path::PathBuf> {
    let manifest = rt_assets::Manifest::parse(embeddinggemma2::ASSET_MANIFEST).unwrap();
    let root = asset_test_support::asset_root(&manifest)?;
    let status = rt_assets::locate(&manifest, &root, "model.safetensors").unwrap();
    if rt_assets::skip_if_missing(&status) {
        return None;
    }
    let rt_assets::AssetStatus::Ready(weights) = status else {
        return None;
    };
    Some(weights.parent().unwrap().to_path_buf())
}

#[test]
fn load_warmup_preserves_output_bits_and_request_limits() {
    let Some(dir) = asset_dir() else {
        return;
    };
    let rt = Runtime::new(RuntimeOptions::new().with_threads(1)).unwrap();
    // Request caps apply to caller inputs. The internal 16-token dummy is
    // independent of tokenizer, max_seq_len, max_batch and MRL output width.
    let config = Config::new(dir)
        .with_max_seq_len(1)
        .with_max_batch(1)
        .with_mrl_dim(128);
    let baseline = {
        let model = EmbeddingGemma2::load_with_tokenizer(&rt, config.clone(), TensorOnly).unwrap();
        model.embed_ids(vec![vec![2]], None).unwrap()
    };
    let model =
        EmbeddingGemma2::load_with_tokenizer(&rt, config.with_warmup(true), TensorOnly).unwrap();
    let warmed = model.embed_ids(vec![vec![2]], None).unwrap();
    assert_eq!(baseline.dim, warmed.dim);
    assert_eq!(
        baseline
            .data
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        warmed.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert!(matches!(
        model.embed_ids(vec![vec![2; 2]], None),
        Err(Error::Shape(_))
    ));
    assert!(matches!(
        model.embed_ids(vec![vec![2], vec![2]], None),
        Err(Error::Shape(_))
    ));
}

#[test]
fn small_workspace_limit_caps_retention_not_requests() {
    let Some(dir) = asset_dir() else {
        return;
    };
    // `workspace_limit_bytes` caps the retained activation pool only. A budget
    // far below the warmup's activation plan must still load, warm up and
    // embed (same behavior as before typed errors).
    let rt = Runtime::new(
        RuntimeOptions::new()
            .with_threads(1)
            .with_workspace_limit_bytes(2 * 1024 * 1024),
    )
    .unwrap();
    let model =
        EmbeddingGemma2::load_with_tokenizer(&rt, Config::new(&dir).with_warmup(true), TensorOnly)
            .expect("warmup must not be rejected by the retention budget");
    let out = model.embed_ids(vec![vec![2; 300]], None).unwrap();
    assert_eq!(out.data.len(), out.dim);
}
