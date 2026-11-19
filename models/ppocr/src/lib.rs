//! PP-OCR models: FP32 oneDNN PP-OCRv6 tiny detector, Korean PP-OCRv5
//! mobile SVTR/CTC recognizer and UVDoc unwarper. Decoded pixels enter
//! through rt-image. Page OCR built from them is the caller's.
pub use rt_core::{Error, Result};
pub mod api;
#[cfg(feature = "cli")]
pub mod decode;
pub mod det;
mod det_graph;
pub mod net;
pub mod rec;
mod rec_graph;
pub mod uvdoc;
mod uvdoc_graph;
pub mod weights;

pub const ASSET_MANIFEST: &str = include_str!("../../../assets/ppocr.toml");
