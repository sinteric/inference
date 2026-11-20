//! Decoded-pixel integration: the caller owns the codec; this example reads
//! BGR bytes.
//!
//! ```text
//! cargo run -p sinteric-inference --example ocr -- \
//!   MODEL_DIR PAGE.bgr WIDTH HEIGHT THREADS
//! ```
use anyhow::{bail, Context, Result};
use sinteric_inference::cpu::{BackendOverrides, Options, Runtime};
use sinteric_inference::ppocr::cpu::{
    DetConfig, Detector, RecConfig, Recognizer, DEFAULT_WORKSPACE_LIMIT_BYTES,
};
use sinteric_inference::ppocr::{ImageView, PixelFormat};

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        bail!("usage: ocr MODEL_DIR PAGE.bgr WIDTH HEIGHT THREADS");
    }
    let bytes = std::fs::read(&args[1])?;
    let w = args[2].parse::<u32>()?;
    let h = args[3].parse::<u32>()?;
    let threads = args[4].parse::<usize>()?;
    let rt = Runtime::new(Options::new().with_threads(threads))?;
    // Detector and recognizer on one backend: one convolution arena sized to
    // the larger model.
    let ocr_rt = rt.shared_backend(
        BackendOverrides::default().with_workspace_limit_bytes(Some(DEFAULT_WORKSPACE_LIMIT_BYTES)),
    )?;
    let root = std::path::Path::new(&args[0]);
    let det = Detector::load(&ocr_rt, DetConfig::new(root.join("det")))?;
    let rec = Recognizer::load(&ocr_rt, RecConfig::new(root.join("rec")))?;
    let stride = (w as usize)
        .checked_mul(3)
        .context("image stride overflows")?;
    let image = ImageView::new(&bytes, w, h, stride, PixelFormat::Bgr8)?;
    let quads = det.detect(&image)?;
    let lines = rec.recognize(&image, &quads)?;
    let page = quads
        .iter()
        .zip(&lines)
        .map(|(quad, line)| serde_json::json!({"quad": quad, "text": line.text, "score": line.score}))
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string(&page)?);
    det.unload();
    rec.unload();
    Ok(())
}
