//! The guard: Horizon-Labs' prompt-injection guard
//! (`Horizon-Labs/prompt-injection-guard-base`), a ModernBERT encoder with a
//! two-label head.
//!
//! This module holds what is true of the guard on any hardware: its tokens,
//! its hidden states, its verdict. [`cpu`] holds the guard tuned for CPUs.
//! [`cpu::Guard::classify`] scores texts; [`cpu::Guard::encode`] returns the
//! hidden states a fine-tune's own heads read.

use serde::Serialize;

mod config;
pub mod cpu;
mod encoder;
mod head;
mod load;
#[cfg(test)]
mod parity;

pub use encoder::Hidden;

/// Pinned asset identity embedded at build time.
pub(crate) const ASSET_MANIFEST: &str = include_str!("../../../../assets/guard.toml");

/// A tokenized text: the model's special tokens included, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Tokens {
    /// Token ids, special tokens included.
    pub ids: Vec<u32>,
    /// Per token, `[start, end)` in Unicode scalar values of the text
    /// (`(0, 0)` for a special token the tokenizer added).
    pub offsets: Vec<(usize, usize)>,
    /// Per token, whether it is a special token.
    pub special: Vec<bool>,
}

/// What the guard says about one text
/// ([`cpu::Guard::classify`](cpu::Guard::classify)).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct Injection {
    /// Injection logit `l[INJECTION] - l[SAFE]`.
    pub logit: f32,
    /// `sigmoid(logit)`: the probability that the text is an injection.
    pub score: f32,
    /// Tokens encoded, special tokens included.
    pub tokens: usize,
    /// The text did not fit in the model's `max_len`: only its head was
    /// read.
    pub truncated: bool,
}
