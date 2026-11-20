//! The guard: the Horizon-Labs prompt-injection guard
//! (`Horizon-Labs/prompt-injection-guard-base`, a ModernBERT encoder with a
//! two-label head), FP32 or BF16 on oneDNN. [`Guard::classify`] scores texts;
//! [`Guard::encode`] returns the hidden states a fine-tune's own heads read.

mod api;
mod config;
mod encoder;
mod head;
mod load;
#[cfg(test)]
mod parity;

pub use api::*;

/// Pinned asset identity embedded at build time.
pub(crate) const ASSET_MANIFEST: &str = include_str!("../../../../assets/guard.toml");
