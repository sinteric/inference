//! CPU inference for a few selected models: one [`Runtime`], and one module
//! per model with that model's own API.
//!
//! ```ignore
//! use sinteric_inference::{embeddinggemma2 as eg2, ppocr, vela2, Runtime, RuntimeOptions};
//!
//! let rt = Runtime::new(RuntimeOptions::new().with_threads(4))?;
//!
//! let emb: eg2::EmbeddingGemma2 = rt.load(eg2::Config::new("/models/embeddinggemma2"))?;
//! let vectors = emb.embed(["what is sinteric?"], eg2::PromptKind::Query)?;
//!
//! let vela: vela2::Vela2 = rt.load(vela2::Config::new("/models/vela2-0p3b"))?;
//! let reply = vela.system_one(&body)?;
//!
//! let det: ppocr::Detector = rt.load(ppocr::DetConfig::new("/models/ppocr/det"))?;
//! let rec: ppocr::Recognizer = rt.load(ppocr::RecConfig::new("/models/ppocr/rec"))?;
//! let quads = det.detect(&image)?;
//! let lines = rec.recognize(&image, &quads)?;
//! ```
//!
//! This crate ships models and the runtime only. Pipelines built from them
//! (page OCR's stage order, score thresholds, reading order) belong to the
//! caller. Models that run one after another can share one backend through
//! [`Runtime::shared_backend`].
//!
//! Create [`Runtime`] once per process, then load models from it
//! (`rt.load::<M>(config)` or `M::load(&rt, config)`). Models are
//! `Send + Sync`; share them with `Arc`. The library path never reads process
//! environment. See [`rt_runtime`] for process-global state and concurrency.
//!
//! Unload with [`Model::unload`] (`model.unload()`; for a shared model the
//! last owner does `if let Some(m) = Arc::into_inner(model) { m.unload() }`). It frees
//! the model's weights, oneDNN objects and buffers and returns freed heap
//! pages to the OS. The runtime's thread pool and ISA cap stay.
//!
//! To run a model in its own process (its own cores, heap and crash domain),
//! load it as [`worker::Isolated`]; the host calls
//! [`worker::main_if_worker`] first in `main`.
//!
//! Each model is behind a cargo feature of the same name, all on by default.
//! A model outside this repository builds on the `rt-*` crates at the same
//! revision and loads from the same [`Runtime`] (`docs/library.md`, "A model
//! of your own").
//!
//! # Precision
//!
//! Every model config has `precision: Option<Precision>`; `None` is the
//! model's validated default, and a precision the model does not implement
//! is [`Error::Unsupported`] rather than a silent fallback. A loaded model
//! reports what it runs ([`Model::precision`]); [`Runtime::cpu_isa_name`]
//! reports the oneDNN ISA running it.
//!
//! | Model | [`Model::PRECISIONS`] (first is the default) |
//! |---|---|
//! | EmbeddingGemma 2 | `Bf16` (BF16 matmul, FP32 stream), `Fp32` |
//! | Vela 2.0 | `Fp32` (strict) |
//! | PP-OCR detector / recognizer | `Fp32` (exact ONNX Runtime text parity) |
//!
//! ```ignore
//! let eg: EmbeddingGemma2 = rt.load(Config::new(dir).with_precision(Precision::Fp32))?;
//! assert_eq!(eg.precision(), Precision::Fp32);
//! ```

pub use rt_core::env::{EnvOverrides, ModelEnvOverrides};
pub use rt_core::{env, Error, Result};
pub use rt_runtime::{
    release_free_memory, BackendOverrides, CacheConfig, CpuIsa, Model, Precision, Runtime,
    RuntimeOptions,
};

/// EmbeddingGemma 2 (text, image, audio): [`embeddinggemma2::EmbeddingGemma2`].
#[cfg(feature = "embeddinggemma2")]
pub mod embeddinggemma2 {
    pub use ::embeddinggemma2::api::*;
}

/// Vela 2.0 0.3B: [`vela2::Vela2`].
#[cfg(feature = "vela2")]
pub mod vela2 {
    pub use ::vela2_0p3b::api::*;
}

/// PP-OCR models: the [`ppocr::Detector`] (PP-OCRv6 tiny) and the
/// [`ppocr::Recognizer`] (Korean PP-OCRv5 mobile).
#[cfg(feature = "ppocr")]
pub mod ppocr {
    pub use ::ppocr::api::*;
}

/// Models in their own worker process: [`worker::Isolated`].
#[cfg(unix)]
pub mod worker;
