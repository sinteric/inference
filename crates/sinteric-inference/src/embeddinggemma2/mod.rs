//! EmbeddingGemma 2 (text, image, audio): [`EmbeddingGemma2`].
//!
//! oneDNN BF16 matmuls with an FP32 residual stream by default, or FP32
//! matmuls with [`Precision::Fp32`]. Text is always loaded. The vision and
//! audio towers load only when [`Config::image`] / [`Config::audio`] ask for
//! them; their soft tokens replace the placeholder rows of the text residual.

use crate::{Error, Result};

mod api;
mod config;
mod frontend;
mod media;
mod model;
#[cfg(test)]
mod tests;

pub use api::*;

/// Pinned asset identity embedded at build time; no source-tree path is
/// needed at runtime.
pub(crate) const ASSET_MANIFEST: &str = include_str!("../../../../assets/embeddinggemma2.toml");

pub(crate) use config::{EMBED_DIM, MAX_SEQ, MRL_DIMS};
pub(crate) use model::{mrl_truncate, Model};
