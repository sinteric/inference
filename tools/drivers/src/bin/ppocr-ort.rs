//! The OCR gate's ONNX Runtime reference: the same pages, flags and JSONL
//! as `ppocr-onednn`, and the library's own steps around the networks
//! (`ppocr::reference::Steps`: preprocessing, DB boxes, crops, CTC decode),
//! with ONNX Runtime running the detector and recognizer. Crops are batched
//! by equal width, widest first, as the library's `Batching::Exact` does.
//!
//! `--model DIR` holds the imported assets (`rec/` for the dictionary) and
//! the pinned exports `DIR/onnx/det.onnx` and `DIR/onnx/rec.onnx`.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use drivers::flags::Flags;
use drivers::gate::{self, Line, Pages};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use sinteric_inference::ppocr::reference::Steps;
use sinteric_inference::ppocr::ImageView;

const VALUED: &[&str] = &["--model", "--threads", "--warmup", "--images", "--out"];

/// An ONNX Runtime error as `anyhow`; a builder's error carries the
/// builder, which is neither `Send` nor `Sync`.
fn ort_error(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("onnxruntime: {e}")
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    ensure!(path.is_file(), "missing ONNX export {}", path.display());
    Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_intra_threads(threads)
        .map_err(ort_error)?
        .with_inter_threads(1)
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)
}

/// Run `session` on one `[shape]` input; its first output's shape and values.
fn run(session: &mut Session, shape: &[usize], values: Vec<f32>) -> Result<(Vec<usize>, Vec<f32>)> {
    let input = Tensor::from_array((shape.to_vec(), values))?;
    let outputs = session.run(ort::inputs![input])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let shape = shape
        .iter()
        .map(|&d| usize::try_from(d).context("negative ONNX Runtime dimension"))
        .collect::<Result<_>>()?;
    Ok((shape, data.to_vec()))
}

struct OrtOcr {
    steps: Steps,
    det: Session,
    rec: Session,
}

impl OrtOcr {
    fn lines(&mut self, image: &ImageView<'_>) -> Result<Vec<Line>> {
        let (shape, values) = self.steps.detector_input(image)?;
        let (shape, prob) = run(&mut self.det, &shape, values)?;
        let boxes = self.steps.boxes(&shape, &prob, image)?;
        let crops = boxes
            .iter()
            .map(|&q| self.steps.crop(image, q))
            .collect::<sinteric_inference::Result<Vec<_>>>()?;
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(crops[i].0));
        let mut decoded: Vec<Option<(String, f32)>> = vec![None; boxes.len()];
        for group in order.chunk_by(|&a, &b| crops[a].0 == crops[b].0) {
            for batch in group.chunks(self.steps.max_batch()) {
                let width = crops[batch[0]].0;
                let mut values = Vec::with_capacity(batch.len() * 3 * 48 * width);
                for &i in batch {
                    values.extend_from_slice(&crops[i].1);
                }
                let (shape, prob) = run(&mut self.rec, &[batch.len(), 3, 48, width], values)?;
                let lengths = vec![self.steps.ctc_steps(width); batch.len()];
                let lines = self.steps.decode(shape, prob, &lengths)?;
                if lines.len() != batch.len() {
                    bail!("{} lines for {} crops", lines.len(), batch.len());
                }
                for (&i, line) in batch.iter().zip(lines) {
                    decoded[i] = Some(line);
                }
            }
        }
        Ok(boxes
            .into_iter()
            .zip(decoded)
            .filter_map(|(quad, line)| line.map(|(text, score)| Line { quad, text, score }))
            .collect())
    }
}

fn main() -> Result<()> {
    let flags = Flags::parse(std::env::args().skip(1), VALUED, &[])?;
    let model: PathBuf = flags.required("--model")?;
    let threads: usize = flags.required("--threads")?;
    ensure!(threads >= 1, "--threads must be >= 1");
    let mut ocr = OrtOcr {
        steps: Steps::load(&model)?,
        det: session(&model.join("onnx/det.onnx"), threads)?,
        rec: session(&model.join("onnx/rec.onnx"), threads)?,
    };
    let pages = Pages {
        images: flags.required("--images")?,
        out: flags.required("--out")?,
        warmup: flags.get("--warmup")?.unwrap_or(0),
        max_pixels: ocr.steps.max_pixels(),
    };
    gate::run(&pages, |_, view| Ok((ocr.lines(view)?, None)))
}
