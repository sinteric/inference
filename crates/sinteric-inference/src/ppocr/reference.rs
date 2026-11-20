//! The PP-OCR steps around the two networks, for a reference that runs the
//! networks somewhere else: the OCR gate's ONNX Runtime runner (`ppocr-ort`
//! in `tools/drivers`). Behind the `reference` feature and hidden from the
//! docs, because it is not a supported API: it changes with the gate.
//!
//! Each step is the one [`super::Detector`] and [`super::Recognizer`] run, at
//! one thread, so the reference and the library differ only in the networks.

use std::path::Path;

use crate::image::{recognizer_crop, ImageSize, ImageView, Quad};
use crate::ppocr::det::{self, DetOptions};
use crate::ppocr::net::Tensor;
use crate::ppocr::rec::{self, RecOptions};
use crate::ppocr::weights::{AssetConfig, Assets, ModelKind};
use crate::{check_shape, Result};

/// The default detector and recognizer options and the recognizer's
/// dictionary.
pub struct Steps {
    det: DetOptions,
    rec: RecOptions,
    dictionary: Vec<String>,
}

impl std::fmt::Debug for Steps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Steps")
            .field("det", &self.det)
            .field("rec", &self.rec)
            .finish_non_exhaustive()
    }
}

impl Steps {
    /// `model_dir` holds the imported assets; the dictionary comes from
    /// `rec/`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let rec_assets = Assets::load(&AssetConfig::new(model_dir.join("rec"), ModelKind::Rec))?;
        Ok(Self {
            det: DetOptions::default(),
            rec: RecOptions::default(),
            dictionary: rec_assets.dictionary().to_vec(),
        })
    }

    /// The largest page the detector accepts, in pixels.
    pub fn max_pixels(&self) -> usize {
        self.det.max_pixels
    }

    /// The most crops one recognizer call takes.
    pub fn max_batch(&self) -> usize {
        self.rec.max_batch
    }

    /// The detector's input for `image`: `(shape, values)`.
    pub fn detector_input(&self, image: &ImageView<'_>) -> Result<(Vec<usize>, Vec<f32>)> {
        let t = det::preprocess_threads(image, &self.det, 1)?;
        Ok((t.shape, t.values))
    }

    /// Text boxes from the detector's `[1, 1, H, W]` probability map.
    pub fn boxes(&self, shape: &[usize], prob: &[f32], image: &ImageView<'_>) -> Result<Vec<Quad>> {
        check_shape!(
            shape.len() == 4 && shape[0] == 1 && shape[1] == 1,
            "detector output {shape:?} is not [1, 1, H, W]"
        );
        let page = ImageSize {
            width: image.width(),
            height: image.height(),
        };
        det::postprocess(prob, shape[3], shape[2], page, &self.det)
    }

    /// The recognizer's input for one box: `(width, values)`, the values a
    /// `[3, 48, width]` crop.
    pub fn crop(&self, image: &ImageView<'_>, quad: Quad) -> Result<(usize, Vec<f32>)> {
        let r = &self.rec;
        let crop = recognizer_crop(image, quad, r.min_width, r.max_width, r.max_pixels)?;
        Ok((crop.width, crop.values))
    }

    /// CTC steps the recognizer emits for a crop `width` pixels wide.
    pub fn ctc_steps(&self, width: usize) -> usize {
        rec::ctc_steps(width)
    }

    /// Greedy CTC over the recognizer's `[batch, steps, classes]` output, row
    /// `i` read to `lengths[i]` steps: `(text, score)` per row.
    pub fn decode(
        &self,
        shape: Vec<usize>,
        prob: Vec<f32>,
        lengths: &[usize],
    ) -> Result<Vec<(String, f32)>> {
        let lines = rec::decode(
            &Tensor::new(shape, prob)?,
            &self.dictionary,
            Some(lengths),
            1,
        )?;
        Ok(lines.into_iter().map(|l| (l.text, l.score)).collect())
    }
}
