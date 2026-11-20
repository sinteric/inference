//! EmbeddingGemma 2 (`google/embeddinggemma-2`): text, image and audio in,
//! one 768-wide unit vector per input out.
//!
//! This module holds what is true of the model on any hardware: its prompt
//! kinds, its tokenizer, what it returns. [`cpu`] holds the model tuned for
//! CPUs. Text is always loaded; the image and audio towers load when asked
//! for, and their soft tokens replace the placeholder rows of the text
//! sequence.

mod config;
pub mod cpu;
mod embeddings;
mod frontend;
mod media;
mod model;
#[cfg(test)]
mod tests;

pub use crate::image::{ImageView, PixelFormat};
pub use config::EMBED_DIM as EMBEDDING_DIM;
pub use embeddings::{Embeddings, Part};
pub use frontend::{Batch, PromptKind, Tokenizer};
pub use media::{ImageTokens, MAX_SAMPLES, SAMPLE_RATE};

/// Pinned asset identity embedded at build time; no source-tree path is
/// needed at runtime.
pub(crate) const ASSET_MANIFEST: &str = include_str!("../../../../assets/embeddinggemma2.toml");

pub(crate) use config::{EMBED_DIM, MAX_SEQ, MRL_DIMS};
pub(crate) use model::{mrl_truncate, Model};
