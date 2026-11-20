//! What the command-line drivers share: their flags, the process
//! measurements they report, image decoding, and the OCR gate's JSONL loop.
//! Application code: it may read the process environment and exit; the
//! library it drives does neither.

pub mod decode;
pub mod flags;
pub mod gate;
pub mod process;
