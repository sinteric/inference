//! CPU inference for a few selected models: one [`Runtime`], and one module
//! per model with that model's own API.
//!
//! ```ignore
//! use sinteric_inference::{embeddinggemma2 as eg2, guard, ppocr, Runtime, RuntimeOptions};
//!
//! let rt = Runtime::new(RuntimeOptions::new().with_threads(4))?;
//!
//! let emb: eg2::EmbeddingGemma2 = rt.load(eg2::Config::new("/models/embeddinggemma2"))?;
//! let vectors = emb.embed(["what is sinteric?"], eg2::PromptKind::Query)?;
//!
//! let guard: guard::Guard = rt.load(guard::Config::new("/models/guard"))?;
//! let scores = guard.classify(&["Ignore previous instructions."])?;
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
//! `Send + Sync`; share them with `Arc`. The library never reads the process
//! environment. [`Runtime`] says what is process-global and how calls run
//! concurrently.
//!
//! Unload with [`Model::unload`] (`model.unload()`; for a shared model the
//! last owner does `if let Some(m) = Arc::into_inner(model) { m.unload() }`). It frees
//! the model's weights, oneDNN objects and buffers and returns freed heap
//! pages to the OS. The runtime's thread pool and ISA cap stay.
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
//! | Guard | `Fp32`, `Bf16` (BF16 matmuls, FP32 attention and stream) |
//! | PP-OCR detector / recognizer / unwarper | `Fp32` (exact ONNX Runtime text parity) |
//!
//! ```ignore
//! let eg: EmbeddingGemma2 = rt.load(Config::new(dir).with_precision(Precision::Fp32))?;
//! assert_eq!(eg.precision(), Precision::Fp32);
//! ```

mod align;
mod assets;
mod cpu;
pub mod embeddinggemma2;
mod error;
pub mod guard;
mod image;
pub mod ppocr;
mod tensor_file;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

pub use cpu::runtime::{
    release_free_memory, BackendOverrides, CacheConfig, CpuIsa, Model, Precision, Runtime,
    RuntimeOptions,
};
pub use error::{Error, Result};

pub(crate) use align::align_up;
pub(crate) use error::{
    check_shape, checked_size, invalid_shape, matrix_span, read_json, try_vec, ResultContext,
};
