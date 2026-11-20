//! PP-OCR models: the [`Detector`] (PP-OCRv6 tiny), the [`Recognizer`]
//! (Korean PP-OCRv5 mobile, SVTR with CTC) and the UVDoc [`Unwarper`], FP32
//! on oneDNN. Decoded pixels go in; page OCR built from them is the caller's.

use crate::{Error, Result};

mod api;
mod det;
mod det_graph;
mod net;
mod rec;
mod rec_graph;
#[cfg(feature = "reference")]
#[doc(hidden)]
pub mod reference;
#[cfg(test)]
mod tests;
mod uvdoc;
mod uvdoc_graph;
mod weights;

pub use api::*;

/// Pinned asset identity embedded at build time.
pub(crate) const ASSET_MANIFEST: &str = include_str!("../../../../assets/ppocr.toml");
