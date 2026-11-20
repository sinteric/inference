//! The encoder's hyper-parameters, read from the Hugging Face `config.json`.

use serde_json::Value;

use crate::guard::encoder::Precision;
use crate::{check_files, Result, ResultContext as Context};

/// The attention tile (query rows per chunk) the guard runs at: BF16 runs
/// EmbeddingGemma 2's measured tile, FP32 the one value measured on every
/// ISA. Neither is derived from cache sizes; `Config::with_tile` overrides
/// both.
pub(crate) fn attn_tile(precision: Precision) -> usize {
    if precision == Precision::Bf16 {
        256
    } else {
        64
    }
}

/// ModernBERT encoder hyper-parameters, from `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EncoderConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub eps: f32,
    /// Sliding-window radius: a local layer attends to `|i - j| <= radius`.
    pub local_radius: usize,
    /// Per layer: full attention (`true`) or sliding window (`false`).
    pub global: Vec<bool>,
    pub global_theta: f64,
    pub local_theta: f64,
    pub pad_id: u32,
    /// `max_position_embeddings`: the longest input the encoder was built for.
    pub positions: usize,
}

fn uint(v: &Value, key: &str) -> Result<usize> {
    v[key]
        .as_u64()
        .map(|n| n as usize)
        .with_context(|| format!("config.json {key}"))
}

impl EncoderConfig {
    pub(crate) fn from_json(cfg: &Value) -> Result<Self> {
        check_files!(
            cfg["model_type"] == "modernbert",
            "config.json model_type {} is not modernbert",
            cfg["model_type"]
        );
        for key in ["attention_bias", "mlp_bias", "norm_bias", "classifier_bias"] {
            check_files!(
                cfg[key].as_bool() == Some(false),
                "config.json {key}: only bias-free ModernBERT is supported"
            );
        }
        check_files!(
            cfg["classifier_pooling"] == "mean" && cfg["classifier_activation"] == "gelu",
            "config.json: mean pooling with a GELU classifier head is required"
        );
        check_files!(
            cfg["hidden_activation"] == "gelu",
            "config.json hidden_activation must be gelu"
        );
        let hidden = uint(cfg, "hidden_size")?;
        let heads = uint(cfg, "num_attention_heads")?;
        let layers = uint(cfg, "num_hidden_layers")?;
        let intermediate = uint(cfg, "intermediate_size")?;
        let vocab = uint(cfg, "vocab_size")?;
        check_files!(
            heads > 0 && hidden % heads == 0 && layers > 0 && intermediate > 0 && vocab > 0,
            "config.json dimensions"
        );
        let head_dim = hidden / heads;
        check_files!(head_dim >= 2 && head_dim % 2 == 0, "RoPE width {head_dim}");
        let eps = cfg["norm_eps"]
            .as_f64()
            .or_else(|| cfg["layer_norm_eps"].as_f64())
            .context("config.json norm_eps")? as f32;
        let window = uint(cfg, "local_attention")?;
        check_files!(window >= 2 && window % 2 == 0, "local_attention {window}");
        let global = match cfg["layer_types"].as_array() {
            Some(types) => types
                .iter()
                .map(|t| match t.as_str() {
                    Some("full_attention") => Ok(true),
                    Some("sliding_attention") => Ok(false),
                    _ => Err(crate::Error::config(format!("layer type {t}"))),
                })
                .collect::<Result<Vec<_>>>()?,
            None => {
                let every = uint(cfg, "global_attn_every_n_layers")?;
                check_files!(every > 0, "global_attn_every_n_layers");
                (0..layers).map(|i| i % every == 0).collect()
            }
        };
        check_files!(
            global.len() == layers,
            "layer_types has {} entries for {layers} layers",
            global.len()
        );
        let theta = |kind: &str, legacy: &str| -> Result<f64> {
            let params = &cfg["rope_parameters"][kind];
            if params.is_object() {
                check_files!(
                    params["rope_type"] == "default",
                    "rope_parameters.{kind}.rope_type must be default"
                );
                return params["rope_theta"]
                    .as_f64()
                    .with_context(|| format!("rope_parameters.{kind}.rope_theta"));
            }
            cfg[legacy]
                .as_f64()
                .with_context(|| format!("rope_parameters.{kind}.rope_theta or {legacy}"))
        };
        let global_theta = theta("full_attention", "global_rope_theta")?;
        let local_theta = theta("sliding_attention", "local_rope_theta")?;
        check_files!(
            global_theta.is_finite()
                && global_theta > 1.0
                && local_theta.is_finite()
                && local_theta > 1.0,
            "rope_theta"
        );
        Ok(Self {
            hidden,
            heads,
            head_dim,
            layers,
            intermediate,
            vocab,
            eps,
            local_radius: window / 2,
            global,
            global_theta,
            local_theta,
            pad_id: cfg["pad_token_id"].as_u64().unwrap_or(0) as u32,
            positions: uint(cfg, "max_position_embeddings")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hf() -> Value {
        json!({
            "model_type": "modernbert", "attention_bias": false, "mlp_bias": false,
            "norm_bias": false, "classifier_bias": false, "classifier_pooling": "mean",
            "classifier_activation": "gelu", "hidden_activation": "gelu",
            "hidden_size": 768, "num_attention_heads": 12, "num_hidden_layers": 4,
            "intermediate_size": 1152, "vocab_size": 256000, "norm_eps": 1e-5,
            "local_attention": 128, "global_attn_every_n_layers": 3, "pad_token_id": 0,
            "max_position_embeddings": 8192,
            "rope_parameters": {
                "full_attention": {"rope_type": "default", "rope_theta": 160000},
                "sliding_attention": {"rope_type": "default", "rope_theta": 80000.0}
            }
        })
    }

    #[test]
    fn parses_the_shape_and_layer_cadence() {
        let c = EncoderConfig::from_json(&hf()).unwrap();
        assert_eq!((c.head_dim, c.local_radius), (64, 64));
        assert_eq!(c.global, [true, false, false, true]);
        assert_eq!((c.global_theta, c.local_theta), (160000.0, 80000.0));
        assert_eq!(c.positions, 8192);
        let mut typed = hf();
        typed["layer_types"] = json!(["sliding_attention", "full_attention"]);
        assert!(
            EncoderConfig::from_json(&typed).is_err(),
            "4 layers, 2 types"
        );
        typed["num_hidden_layers"] = json!(2);
        assert_eq!(
            EncoderConfig::from_json(&typed).unwrap().global,
            [false, true]
        );
    }

    #[test]
    fn rejects_unsupported_variants() {
        for (key, value) in [
            ("model_type", json!("bert")),
            ("attention_bias", json!(true)),
            ("classifier_pooling", json!("cls")),
            ("classifier_activation", json!("relu")),
            ("local_attention", json!(127)),
        ] {
            let mut cfg = hf();
            cfg[key] = value;
            assert!(EncoderConfig::from_json(&cfg).is_err(), "{key}");
        }
        let mut yarn = hf();
        yarn["rope_parameters"]["full_attention"]["rope_type"] = json!("yarn");
        assert!(EncoderConfig::from_json(&yarn).is_err());
    }
}
