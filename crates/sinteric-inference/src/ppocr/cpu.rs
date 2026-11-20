//! The PP-OCR models tuned for CPUs: the [`Detector`], the [`Recognizer`] and
//! the UVDoc [`Unwarper`], each loaded from a [`Runtime`]. They run FP32
//! only, the precision they are qualified at (text identical to ONNX
//! Runtime's), so no config has a precision. Composing them into page OCR
//! (stage order, score threshold, when to keep an unwarped page, reading
//! order) is the caller's job.
//!
//! ```ignore
//! use sinteric_inference::ppocr::cpu::{DetConfig, Detector, RecConfig, Recognizer};
//! let det = Detector::load(&rt, DetConfig::new("/models/ppocr/det"))?;
//! let rec = Recognizer::load(&rt, RecConfig::new("/models/ppocr/rec"))?;
//! let quads = det.detect(&image)?;
//! let lines = rec.recognize(&image, &quads)?; // one per quad, same order
//!
//! let uvdoc = Unwarper::load(&rt, UvdocConfig::new("/models/ppocr/uvdoc"))?;
//! let flat = uvdoc.unwarp(&photo)?; // same size and pixel format
//!
//! // Both on one backend (one convolution arena sized to the larger model,
//! // ~150 MiB less RSS than one backend each):
//! let ocr_rt = rt.shared_backend(
//!     cpu::BackendOverrides::default().with_workspace_limit_bytes(Some(1 << 30)),
//! )?;
//! let det = Detector::load(&ocr_rt, DetConfig::new("/models/ppocr/det"))?;
//! let rec = Recognizer::load(&ocr_rt, RecConfig::new("/models/ppocr/rec"))?;
//! ```
//!
//! A [`Detector`] or [`Recognizer`] loaded on a plain [`Runtime`] owns its
//! backend (engine, convolution arena, primitive cache), so it can be used,
//! shared or unloaded independently.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::cpu::onednn::OnednnCpu;
use crate::cpu::runtime::{release_free_memory, BackendOverrides, Runtime};
use crate::ppocr::weights::{AssetConfig, Assets, ModelKind};
use crate::{Error, Result};

pub use crate::ppocr::det::{DetOptions, DetTimings, Detector};
pub use crate::ppocr::rec::{Batching, RecOptions, RecTimings, Recognizer};
pub use crate::ppocr::uvdoc::{Unwarper, UvdocOptions, UvdocTimings};

/// Backend workspace cap a model gets on its own backend when its config
/// leaves `workspace_limit_bytes` unset.
pub const DEFAULT_WORKSPACE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;

macro_rules! model_config {
    ($(#[$doc:meta])* $name:ident, $options:ty, $kind:expr, $what:literal) => {
        $(#[$doc])*
        #[derive(Clone, Debug, Serialize, Deserialize)]
        #[non_exhaustive]
        pub struct $name {
            /// Directory with this model's pinned weights.
            pub model_dir: PathBuf,
            /// Backend workspace cap: each planned convolution arena plus
            /// native scratch. `None` is [`DEFAULT_WORKSPACE_LIMIT_BYTES`]
            /// (independently of
            /// [`Options::workspace_limit_bytes`](crate::cpu::Options::workspace_limit_bytes)),
            /// or the handle's cap on a [`Runtime::shared_backend`] handle,
            /// where setting it is an error.
            #[serde(default)]
            pub workspace_limit_bytes: Option<usize>,
            /// What the backend may keep between calls, in bytes; overrides
            /// [`Options::memory_budget_bytes`](crate::cpu::Options::memory_budget_bytes).
            /// `Some(0)` is unbounded. Set it on the handle instead when
            /// loading through [`Runtime::shared_backend`].
            #[serde(default)]
            pub memory_budget_bytes: Option<usize>,
            #[serde(default)]
            pub options: $options,
            #[serde(default)]
            pub verify_weights: bool,
            /// Explicit pin manifest; `None` uses the embedded PP-OCR manifest.
            #[serde(default)]
            pub manifest: Option<PathBuf>,
        }

        impl $name {
            pub fn new(model_dir: impl Into<PathBuf>) -> Self {
                Self {
                    model_dir: model_dir.into(),
                    workspace_limit_bytes: None,
                    memory_budget_bytes: None,
                    options: <$options>::default(),
                    verify_weights: false,
                    manifest: None,
                }
            }
            pub fn with_workspace_limit_bytes(mut self, bytes: usize) -> Self {
                self.workspace_limit_bytes = Some(bytes);
                self
            }
            pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
                self.memory_budget_bytes = Some(bytes);
                self
            }
            fn backend(&self, rt: &Runtime) -> Result<Arc<OnednnCpu>> {
                let workspace = match self.workspace_limit_bytes {
                    None if !rt.shares_backend() => Some(DEFAULT_WORKSPACE_LIMIT_BYTES),
                    w => w,
                };
                rt.backend_with(
                    BackendOverrides::default()
                        .with_workspace_limit_bytes(workspace)
                        .with_memory_budget_bytes(self.memory_budget_bytes),
                )
            }
            pub fn with_options(mut self, options: $options) -> Self {
                self.options = options;
                self
            }
            pub fn with_verify_weights(mut self, verify: bool) -> Self {
                self.verify_weights = verify;
                self
            }
            pub fn with_manifest(mut self, manifest: impl Into<PathBuf>) -> Self {
                self.manifest = Some(manifest.into());
                self
            }
            pub fn validate(&self) -> Result<()> {
                if self.workspace_limit_bytes == Some(0) {
                    return Err(Error::config(
                        concat!($what, " workspace_limit_bytes must be >= 1"),
                    ));
                }
                self.options.validate()
            }
            fn assets(&self) -> Result<Assets> {
                let mut a = AssetConfig::new(&self.model_dir, $kind);
                a.verify_weights = self.verify_weights;
                a.manifest = self.manifest.clone();
                Assets::load(&a)
            }
        }
    };
}

model_config!(
    /// PP-OCRv6 tiny detector load config.
    DetConfig,
    DetOptions,
    ModelKind::Det,
    "detector"
);
model_config!(
    /// Korean PP-OCRv5 mobile recognizer load config.
    RecConfig,
    RecOptions,
    ModelKind::Rec,
    "recognizer"
);
model_config!(
    /// UVDoc unwarper load config.
    UvdocConfig,
    UvdocOptions,
    ModelKind::Uvdoc,
    "unwarper"
);

/// `load` and `unload` of a PP-OCR model and its config.
macro_rules! model_load {
    ($model:ty, $config:ty) => {
        impl $model {
            /// Load the pinned weights from the config's directory.
            pub fn load(rt: &Runtime, config: $config) -> Result<Self> {
                let model = (|| {
                    config.validate()?;
                    let assets = config.assets()?;
                    let cpu = config.backend(rt)?;
                    <$model>::from_assets(cpu, assets, config.options)
                })();
                release_free_memory();
                model
            }

            /// Free this model's weights, oneDNN objects and buffers (its
            /// backend too, unless a [`Runtime::shared_backend`] handle or
            /// another model still holds it), and return the freed heap pages
            /// to the OS.
            pub fn unload(self) {
                drop(self);
                release_free_memory();
            }
        }
    };
}

model_load!(Detector, DetConfig);
model_load!(Recognizer, RecConfig);
model_load!(Unwarper, UvdocConfig);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_config_defaults_and_serde() {
        let default = DetConfig::new("explicit");
        assert_eq!(default.workspace_limit_bytes, None);
        let json: DetConfig = serde_json::from_str(r#"{"model_dir":"explicit"}"#).unwrap();
        let toml: RecConfig = toml::from_str("model_dir = 'explicit'").unwrap();
        assert_eq!(json.workspace_limit_bytes, None);
        assert_eq!(toml.workspace_limit_bytes, None);
        // Configs written before the field became optional still parse.
        let old: DetConfig =
            serde_json::from_str(r#"{"model_dir":"x","workspace_limit_bytes":1024}"#).unwrap();
        assert_eq!(old.workspace_limit_bytes, Some(1024));
        let small = default.with_workspace_limit_bytes(256 * 1024 * 1024);
        let json: DetConfig =
            serde_json::from_str(&serde_json::to_string(&small).unwrap()).unwrap();
        let toml: DetConfig = toml::from_str(&toml::to_string(&small).unwrap()).unwrap();
        assert_eq!(json.workspace_limit_bytes, small.workspace_limit_bytes);
        assert_eq!(toml.workspace_limit_bytes, small.workspace_limit_bytes);
        assert!(small.with_workspace_limit_bytes(0).validate().is_err());
        assert!(RecConfig::new("x")
            .with_workspace_limit_bytes(0)
            .validate()
            .is_err());
    }

    #[test]
    fn options_validate_through_config() {
        let mut c = RecConfig::new("explicit");
        assert!(c.validate().is_ok());
        c.options.max_batch = 17;
        assert!(c.validate().is_err());
        let mut d = DetConfig::new("explicit");
        d.options.max_candidates = 0;
        assert!(d.validate().is_err());
        let mut u = UvdocConfig::new("explicit");
        assert!(u.validate().is_ok());
        u.options.plan_capacity = 0;
        assert!(u.validate().is_err());
    }
}
