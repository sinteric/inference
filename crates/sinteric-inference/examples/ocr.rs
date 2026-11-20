//! Decoded-pixel integration: caller owns the codec; this example reads BGR bytes.
//! Usage: ocr MODEL_DIR PAGE.bgr WIDTH HEIGHT THREADS
use sinteric_inference::ppocr::{
    DetConfig, Detector, ImageView, PixelFormat, RecConfig, Recognizer,
    DEFAULT_WORKSPACE_LIMIT_BYTES,
};
use sinteric_inference::{BackendOverrides, CpuIsa, Model, Runtime, RuntimeOptions};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err("usage: ocr MODEL_DIR PAGE.bgr WIDTH HEIGHT THREADS".into());
    }
    let bytes = std::fs::read(&args[1])?;
    let w = args[2].parse::<u32>()?;
    let h = args[3].parse::<u32>()?;
    let threads = args[4].parse::<usize>()?;
    let rt = Runtime::new(
        RuntimeOptions::new()
            .with_threads(threads)
            .with_cpu_isa(CpuIsa::Avx512CoreBf16),
    )?;
    // Detector and recognizer on one backend.
    let ocr_rt = rt.shared_backend(
        BackendOverrides::default().with_workspace_limit_bytes(Some(DEFAULT_WORKSPACE_LIMIT_BYTES)),
    )?;
    let root = std::path::Path::new(&args[0]);
    let det: Detector = ocr_rt.load(DetConfig::new(root.join("det")))?;
    let rec: Recognizer = ocr_rt.load(RecConfig::new(root.join("rec")))?;
    let image = ImageView::new(
        &bytes,
        w,
        h,
        (w as usize)
            .checked_mul(3)
            .ok_or("image stride overflows")?,
        PixelFormat::Bgr8,
    )?;
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
