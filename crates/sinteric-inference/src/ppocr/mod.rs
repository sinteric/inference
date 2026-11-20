//! PP-OCR: the detector (PP-OCRv6 tiny), the recognizer (Korean PP-OCRv5
//! mobile, SVTR with CTC) and the UVDoc unwarper. Decoded pixels go in; page
//! OCR built from them is the caller's.
//!
//! This module holds what is true of the models on any hardware: the image
//! they read and what they return. [`cpu`] holds them tuned for CPUs.

use crate::{Error, Result};

pub mod cpu;
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

pub use crate::image::{ImageView, PixelFormat, Quad};
pub use rec::DecodedLine;
pub use uvdoc::UnwarpedImage;

/// Pinned asset identity embedded at build time.
pub(crate) const ASSET_MANIFEST: &str = include_str!("../../../../assets/ppocr.toml");
