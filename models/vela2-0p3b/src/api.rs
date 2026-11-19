//! Public Vela 2.0 0.3B API: load from a [`rt_runtime::Runtime`].
//!
//! ```ignore
//! use vela2_0p3b::api::{Config, Vela2};
//! let vela: Vela2 = rt.load(Config::new("/models/vela2-0p3b"))?;
//! let reply = vela.system_one(&body)?;
//! ```

use std::path::PathBuf;

use rt_core::env::{EnvOverrides, ModelKind};
use rt_onednn::{CacheOptions, OnednnCpu};
use rt_runtime::{BackendOverrides, CacheConfig, Precision, Runtime};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::onednn::model::Model;
use crate::{Error, Frontend, Result};

pub use crate::{Inputs, Logits, SystemOneError};

/// Hidden Vela knobs (attention tile, …). Prefer omitting this.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct Advanced {
    /// Attention tile. `None` uses the measured per-ISA table (64 for Vela).
    #[serde(default)]
    pub tile: Option<usize>,
}

/// Vela 2.0 0.3B load config. Runs [`Precision::Fp32`] (strict) only.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct Config {
    /// Directory with `model.safetensors`, `tokenizer.json`, `config.json`.
    pub model_dir: PathBuf,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    #[serde(default)]
    pub max_batch: Option<usize>,
    /// Explicit pin manifest; `None` uses the embedded model manifest.
    #[serde(default)]
    pub manifest: Option<PathBuf>,
    #[serde(default)]
    pub verify_weights: bool,
    /// What this model may keep between calls, in bytes; overrides
    /// [`RuntimeOptions::memory_budget_bytes`](rt_runtime::RuntimeOptions::memory_budget_bytes).
    /// `Some(0)` is unbounded.
    #[serde(default)]
    pub memory_budget_bytes: Option<usize>,
    /// Matmul precision. Vela runs [`Precision::Fp32`] only (`None` is the
    /// same); anything else is [`Error::Unsupported`].
    #[serde(default)]
    pub precision: Option<Precision>,
    #[serde(default)]
    #[doc(hidden)]
    pub advanced: Advanced,
}

impl Config {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            cache: CacheConfig::default(),
            max_seq_len: None,
            max_batch: None,
            manifest: None,
            verify_weights: false,
            memory_budget_bytes: None,
            precision: None,
            advanced: Advanced::default(),
        }
    }

    /// Apply cache, manifest, verification and exact model-directory overrides.
    /// Only this model's verification/manifest variables are used.
    pub fn with_overrides(mut self, env: &EnvOverrides) -> Self {
        if let Some(root) = &env.assets {
            self.model_dir = root.clone();
        }
        if let Some(enabled) = env.packed_cache {
            self.cache.enabled = enabled;
        }
        if let Some(file) = &env.packed_cache_file {
            self.cache.file = Some(file.clone());
        }
        let model = env.model(ModelKind::Vela);
        if let Some(verify) = model.verify_weights {
            self.verify_weights = verify;
        }
        if let Some(manifest) = &model.manifest {
            self.manifest = Some(manifest.clone());
        }
        self
    }

    pub fn with_max_seq_len(mut self, n: usize) -> Self {
        self.max_seq_len = Some(n);
        self
    }
    pub fn with_max_batch(mut self, n: usize) -> Self {
        self.max_batch = Some(n);
        self
    }
    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = Some(precision);
        self
    }
    pub fn with_cache(mut self, cache: CacheConfig) -> Self {
        self.cache = cache;
        self
    }
    #[doc(hidden)]
    pub fn with_tile(mut self, n: usize) -> Self {
        self.advanced.tile = Some(n);
        self
    }

    fn tile(&self, auto: usize) -> Result<usize> {
        let tile = self.advanced.tile.unwrap_or(auto);
        if tile == 0 {
            return Err(Error::Config("tile must be > 0".into()));
        }
        Ok(tile)
    }

    fn cache_options(&self) -> CacheOptions {
        self.cache
            .options(self.verify_weights, self.manifest.clone())
    }
}

/// Loaded Vela 2.0 0.3B. `Send + Sync`; the tokenizer cache is behind a mutex.
///
/// # Concurrency
///
/// `system_one` / `predict` / `forward` are **blocking** (run them on
/// `tokio::task::spawn_blocking`). Concurrent forwards on one model acquire a
/// slot from the runtime's bounded execute pool
/// ([`rt_runtime::RuntimeOptions::exec_slots`], default 1). Parallel GEMM/glue
/// regions share the process-global thread pool.
pub struct Vela2 {
    model: Model<OnednnCpu>,
    frontend: Frontend,
    max_batch: Option<usize>,
    max_seq_len: Option<usize>,
}

/// Precisions Vela implements: strict FP32 only.
const PRECISIONS: &[Precision] = &[Precision::Fp32];

impl rt_runtime::Model for Vela2 {
    type Config = Config;
    const PRECISIONS: &'static [Precision] = PRECISIONS;
    fn load(rt: &Runtime, config: Config) -> Result<Self> {
        Self::load(rt, config)
    }
    fn precision(&self) -> Precision {
        Precision::Fp32
    }
}

impl Vela2 {
    pub fn load(rt: &Runtime, config: Config) -> Result<Self> {
        Precision::resolve(config.precision, "Vela 2.0", PRECISIONS)?;
        if config.max_batch == Some(0) || config.max_seq_len == Some(0) {
            return Err(Error::Config(
                "max_batch and max_seq_len must be positive".into(),
            ));
        }
        let backend = rt.backend_with(
            BackendOverrides::default().with_memory_budget_bytes(config.memory_budget_bytes),
        )?;
        let tile = config.tile(rt.plan().vela_attn_tile)?;
        let cache = config.cache_options();
        let model = Model::load_with(&config.model_dir, backend, tile, &cache, false)?;
        let mut frontend = Frontend::load(&config.model_dir)?;
        if let Some(max) = config.max_batch {
            if max == 0 {
                return Err(Error::Config("max_batch must be >= 1".into()));
            }
            frontend.batch_size = max;
        }
        if let Some(max) = config.max_seq_len {
            if max == 0 {
                return Err(Error::Config("max_seq_len must be >= 1".into()));
            }
            frontend.token_budget = max;
        }
        Ok(Self {
            model,
            frontend,
            max_batch: config.max_batch,
            max_seq_len: config.max_seq_len,
        })
    }

    /// SystemOne request body → response (`Frontend::system_one_request`).
    pub fn system_one(&self, body: &Value) -> Result<Value> {
        self.frontend
            .system_one_request(body, |inputs| self.forward(inputs))
    }

    /// Backend counters (packed weights, cached plans, retained buffers and
    /// scratch) for drivers and diagnostics.
    #[doc(hidden)]
    pub fn backend_stats(&self) -> Value {
        self.model.stats()
    }

    /// Tensor path: pre-tokenized [`Inputs`] → [`Logits`].
    pub fn forward(&self, inputs: &Inputs) -> Result<Logits> {
        if let Some(max) = self.max_batch {
            let n = inputs.input_ids.len();
            if n > max {
                return Err(Error::Shape(format!("batch {n} > max_batch {max}")));
            }
        }
        if let Some(max) = self.max_seq_len {
            if inputs.input_ids.iter().any(|row| row.len() > max) {
                return Err(Error::Shape(format!(
                    "Vela sequence exceeds max_seq_len {max}"
                )));
            }
        }
        self.model.forward(inputs)
    }

    /// Generic prediction request → response (`Frontend::predict`).
    pub fn predict(&self, request: &Value) -> Result<Value> {
        self.frontend
            .predict(request, |inputs| self.forward(inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_snapshot_applies_only_this_model() {
        let values = [
            ("VELA_PACKED_CACHE", "false"),
            ("SINTERIC_ASSETS", "assets/exact"),
            ("VELA_VERIFY_WEIGHTS", "false"),
            ("VELA_MANIFEST", "vela.toml"),
        ];
        let env = EnvOverrides::from_lookup(|name| {
            values
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| (*v).into())
        })
        .unwrap();
        let mut vela = Config::new("caller/vela");
        vela.verify_weights = true;
        let vela = vela.with_overrides(&env);
        assert_eq!(vela.model_dir, PathBuf::from("assets/exact"));
        assert!(!vela.cache.enabled);
        assert!(!vela.verify_weights);
        assert_eq!(vela.manifest, Some("vela.toml".into()));
        assert_eq!(vela.clone().with_overrides(&EnvOverrides::default()), vela);
        let only_eg =
            EnvOverrides::from_lookup(|n| (n == "EG2_MANIFEST").then(|| "only-eg.toml".into()))
                .unwrap();
        assert!(Config::new("vela")
            .with_overrides(&only_eg)
            .manifest
            .is_none());
    }

    #[test]
    fn tile_none_uses_plan_some_overrides() {
        assert_eq!(Config::new("/m").tile(48).unwrap(), 48);
        assert_eq!(Config::new("/m").with_tile(32).tile(48).unwrap(), 32);
        assert!(Config::new("/m").with_tile(0).tile(48).is_err());
    }
}
