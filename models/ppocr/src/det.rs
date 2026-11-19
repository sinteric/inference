//! PP-OCRv6 tiny detector and its default DB post-processing settings.
use crate::net::{self, OcrNet, Tensor};
use crate::weights::{AssetConfig, Assets, ModelKind};
use crate::{det_graph, Error, Result};
use rt_image::{ImageSize, PixelFormat};
pub use rt_image::{ImageView, Quad};
use rt_onednn::conv::{ConvNet, PlanInfo};
use rt_onednn::{Hardware, OnednnCpu};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct DetOptions {
    pub thresh: f32,
    pub box_thresh: f32,
    pub unclip_ratio: f32,
    pub max_candidates: usize,
    pub min_side_pre: f32,
    pub min_side_post: f32,
    pub max_side: Option<u32>,
    pub plan_capacity: usize,
    /// Bound both source and resized pixels before preprocessing allocations.
    pub max_pixels: usize,
}
impl Default for DetOptions {
    fn default() -> Self {
        Self {
            thresh: 0.2,
            box_thresh: 0.45,
            unclip_ratio: 1.4,
            max_candidates: 1000,
            min_side_pre: 3.,
            min_side_post: 5.,
            max_side: None,
            plan_capacity: 16,
            max_pixels: 64 * 1024 * 1024,
        }
    }
}
impl DetOptions {
    pub fn validate(&self) -> Result<()> {
        if !self.thresh.is_finite()
            || !(0. ..=1.).contains(&self.thresh)
            || !self.box_thresh.is_finite()
            || !(0. ..=1.).contains(&self.box_thresh)
            || !self.unclip_ratio.is_finite()
            || self.unclip_ratio <= 0.
            || !self.min_side_pre.is_finite()
            || self.min_side_pre <= 0.
            || !self.min_side_post.is_finite()
            || self.min_side_post <= 0.
            || self.max_candidates == 0
            || self.plan_capacity == 0
            || self.max_pixels == 0
            || self.max_side == Some(0)
        {
            return Err(Error::Config("invalid detector options".into()));
        }
        Ok(())
    }
    fn pixels(&self, w: usize, h: usize) -> Result<()> {
        let n = rt_core::checked_size("detector pixels", &[w, h])?;
        if n > self.max_pixels {
            return Err(Error::Shape(format!(
                "detector {w}x{h} exceeds max_pixels {}",
                self.max_pixels
            )));
        }
        Ok(())
    }
}

pub struct Detector {
    assets: Assets,
    net: ConvNet,
    options: DetOptions,
    threads: usize,
    cpu: Arc<OnednnCpu>,
}
impl Detector {
    /// Load and pack fixed-layout weights on an explicitly configured backend.
    pub fn load(
        cpu: Arc<OnednnCpu>,
        model_dir: impl AsRef<Path>,
        options: DetOptions,
        verify_weights: bool,
    ) -> Result<Self> {
        options.validate()?;
        let mut config = AssetConfig::new(model_dir.as_ref(), ModelKind::Det);
        config.verify_weights = verify_weights;
        let assets = Assets::load(&config)?;
        Self::from_assets(cpu, assets, options)
    }
    pub(crate) fn from_assets(
        cpu: Arc<OnednnCpu>,
        assets: Assets,
        options: DetOptions,
    ) -> Result<Self> {
        options.validate()?;
        let threads = cpu.threads();
        let net = det_graph::build(cpu.clone(), &assets, options.plan_capacity)?;
        Ok(Self {
            assets,
            net,
            options,
            threads,
            cpu,
        })
    }
    #[doc(hidden)]
    #[cfg(feature = "debug-dumps")]
    pub fn debug_prefix(&self, count: usize, shape: [usize; 4], input: &[f32]) -> Result<Tensor> {
        self.validate_shape(&shape)?;
        net::validate(&shape, input.len())?;
        let (s, v) = self.net.prefix(count)?.run(shape, input)?;
        Tensor::new(s.to_vec(), v)
    }
    pub fn assets(&self) -> &Assets {
        &self.assets
    }
    pub fn options(&self) -> &DetOptions {
        &self.options
    }
    /// Backend counters (arena, scratch, packed weights, cached plans) for
    /// drivers and diagnostics.
    #[doc(hidden)]
    pub fn backend_stats(&self) -> serde_json::Value {
        self.cpu.stats_json()
    }
    pub fn warmup(&self, height: usize, width: usize) -> Result<PlanInfo> {
        self.validate_shape(&[1, 3, height, width])?;
        self.net.warmup([1, 3, height, width])
    }
    fn validate_shape(&self, shape: &[usize]) -> Result<()> {
        rt_core::check_shape!(
            shape.len() == 4
                && shape[0] > 0
                && shape[1] == 3
                && shape[2] > 0
                && shape[3] > 0
                && shape[2] % 32 == 0
                && shape[3] % 32 == 0,
            "detector requires NCHW BGR with H/W positive multiples of 32"
        );
        self.options.pixels(shape[3], shape[2])?;
        self.options.pixels(
            rt_core::checked_size("detector batch", &[shape[0], shape[3]])?,
            shape[2],
        )
    }
    /// Prepare BGR ImageNet NCHW input. No side cap is applied by default.
    pub fn preprocess(&self, image: &ImageView<'_>) -> Result<Tensor> {
        preprocess_threads(image, &self.options, self.threads)
    }
    /// Text-line quads on `image`, in reading order.
    pub fn detect(&self, image: &ImageView<'_>) -> Result<Vec<Quad>> {
        self.detect_timed(image).map(|(boxes, _)| boxes)
    }
    /// [`Detector::detect`] plus where the time went.
    pub fn detect_timed(&self, image: &ImageView<'_>) -> Result<(Vec<Quad>, DetTimings)> {
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1000.;
        let t = std::time::Instant::now();
        let input = self.preprocess(image)?;
        let pre = ms(t);
        let t = std::time::Instant::now();
        let prob = self.forward(&input.shape, &input.values)?;
        let fwd = ms(t);
        let t = std::time::Instant::now();
        let boxes = postprocess(
            &prob.values,
            prob.shape[3],
            prob.shape[2],
            ImageSize {
                width: image.width(),
                height: image.height(),
            },
            &self.options,
        )?;
        Ok((
            boxes,
            DetTimings {
                preprocess_ms: pre,
                forward_ms: fwd,
                postprocess_ms: ms(t),
            },
        ))
    }
}
/// Where [`Detector::detect_timed`] spent its time, in milliseconds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DetTimings {
    /// Resize + normalize.
    pub preprocess_ms: f64,
    /// Network + sigmoid.
    pub forward_ms: f64,
    /// DB boxes.
    pub postprocess_ms: f64,
}
impl OcrNet for Detector {
    fn forward(&self, shape: &[usize], input: &[f32]) -> Result<Tensor> {
        net::validate(shape, input.len())?;
        self.validate_shape(shape)?;
        rt_core::check_shape!(
            crate::rec::all_par(self.threads, input, f32::is_finite),
            "nonfinite detector input"
        );
        let (shape, mut values) = self
            .net
            .run([shape[0], shape[1], shape[2], shape[3]], input)?;
        rt_core::check_shape!(
            crate::rec::all_par(self.threads, &values, f32::is_finite),
            "nonfinite detector output"
        );
        // DB head Sigmoid, bit-identical to ONNX Runtime's x86-64 kernel.
        rt_kernels::logistic_ort_inplace(&mut values, self.threads);
        Tensor::new(shape.to_vec(), values)
    }
}

pub fn preprocess(image: &ImageView<'_>, options: &DetOptions) -> Result<Tensor> {
    preprocess_threads(image, options, 1)
}

/// [`preprocess`] split by rows across `threads`; same values.
pub fn preprocess_threads(
    image: &ImageView<'_>,
    options: &DetOptions,
    threads: usize,
) -> Result<Tensor> {
    options.validate()?;
    options.pixels(image.width() as usize, image.height() as usize)?;
    let size = rt_image::detector_size(
        ImageSize {
            width: image.width(),
            height: image.height(),
        },
        options.max_side,
    )?;
    options.pixels(size.width as usize, size.height as usize)?;
    let bytes = rt_image::resize_triangle_bgr_threads(image, size.width, size.height, threads)?;
    let view = ImageView::new(
        &bytes,
        size.width,
        size.height,
        size.width as usize * 3,
        PixelFormat::Bgr8,
    )?;
    Tensor::new(
        vec![1, 3, size.height as usize, size.width as usize],
        rt_image::normalize_nchw_threads(
            &view,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
            threads,
        )?,
    )
}

/// Shared DB post-processing for oneDNN and the later ORT benchmark adapter.
pub fn postprocess(
    prob: &[f32],
    width: usize,
    height: usize,
    page: ImageSize,
    options: &DetOptions,
) -> Result<Vec<Quad>> {
    options.validate()?;
    options.pixels(width, height)?;
    options.pixels(page.width as usize, page.height as usize)?;
    rt_core::check_shape!(
        width > 0
            && height > 0
            && page.width > 0
            && page.height > 0
            && prob.len() == rt_core::checked_size("DB probability map", &[width, height])?,
        "invalid DB map/page dimensions"
    );
    rt_core::check_shape!(
        prob.iter().all(|v| v.is_finite() && (0. ..=1.).contains(v)),
        "invalid DB probabilities"
    );
    let mut mask = rt_core::try_vec::<u8>(prob.len())?;
    for (m, &p) in mask.iter_mut().zip(prob) {
        *m = u8::from(p > options.thresh);
    }
    let contours = rt_image::contours(&mask, width, height)?;
    let mut boxes = Vec::new();
    let capacity = contours.len().min(options.max_candidates);
    boxes
        .try_reserve_exact(capacity)
        .map_err(|_| Error::Allocation {
            bytes: capacity.saturating_mul(32),
        })?;
    for contour in contours.iter().take(options.max_candidates) {
        let Some((rect, side)) = rt_image::min_area_rect(contour)? else {
            continue;
        };
        if side < options.min_side_pre
            || rt_image::polygon_mean_score(prob, width, height, rect)? < options.box_thresh
        {
            continue;
        }
        let expanded = rt_image::unclip(rect, options.unclip_ratio)?;
        let Some((mut rect, side)) = rt_image::min_area_rect(&expanded)? else {
            continue;
        };
        if side < options.min_side_post {
            continue;
        }
        for p in &mut rect {
            p[0] = (p[0] / width as f32 * page.width as f32)
                .round_ties_even()
                .clamp(0., (page.width - 1) as f32);
            p[1] = (p[1] / height as f32 * page.height as f32)
                .round_ties_even()
                .clamp(0., (page.height - 1) as f32);
        }
        boxes.push(rect);
    }
    sort_boxes(&mut boxes);
    Ok(boxes)
}
/// Paddle reading order: sort by y/x, then adjacent insertion with 10px y tolerance.
pub fn sort_boxes(boxes: &mut [Quad]) {
    boxes.sort_by(|a, b| {
        a[0][1]
            .total_cmp(&b[0][1])
            .then(a[0][0].total_cmp(&b[0][0]))
    });
    for i in 1..boxes.len() {
        let mut j = i;
        while j > 0
            && (boxes[j][0][1] - boxes[j - 1][0][1]).abs() < 10.
            && boxes[j][0][0] < boxes[j - 1][0][0]
        {
            boxes.swap(j, j - 1);
            j -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_and_invalid_maps() {
        for options in [
            DetOptions {
                thresh: f32::NAN,
                ..Default::default()
            },
            DetOptions {
                max_side: Some(0),
                ..Default::default()
            },
            DetOptions {
                max_candidates: 0,
                ..Default::default()
            },
            DetOptions {
                unclip_ratio: -1.,
                ..Default::default()
            },
            DetOptions {
                plan_capacity: 0,
                ..Default::default()
            },
        ] {
            assert!(options.validate().is_err());
        }
        let size = ImageSize {
            width: 2,
            height: 2,
        };
        for values in [vec![], vec![0.; 3], vec![f32::NAN; 4], vec![2.; 4]] {
            assert!(postprocess(&values, 2, 2, size, &DetOptions::default()).is_err());
        }
        assert!(postprocess(&[0.; 4], 2, 2, size, &DetOptions::default())
            .unwrap()
            .is_empty());
        let image = ImageView::new(&[0; 12], 2, 2, 6, PixelFormat::Bgr8).unwrap();
        assert!(preprocess(
            &image,
            &DetOptions {
                max_pixels: 3,
                ..Default::default()
            }
        )
        .is_err());
        assert!(preprocess(
            &image,
            &DetOptions {
                max_pixels: 100,
                ..Default::default()
            }
        )
        .is_err()); // resized 32x32
    }
    #[test]
    fn default_db_settings_and_reading_order() {
        let w = 64;
        let h = 64;
        let mut prob = vec![0.; w * h];
        for y in 20..30 {
            for x in 10..40 {
                prob[y * w + x] = 0.9;
            }
        }
        let boxes = postprocess(
            &prob,
            w,
            h,
            ImageSize {
                width: 128,
                height: 128,
            },
            &DetOptions::default(),
        )
        .unwrap();
        assert_eq!(boxes.len(), 1);
        assert!(boxes[0][0][0] < 20. && boxes[0][0][1] < 40. && boxes[0][2][0] > 78.);
        let q = |x: f32, y: f32| [[x, y], [x + 1., y], [x + 1., y + 1.], [x, y + 1.]];
        let mut boxes = [q(20., 21.), q(30., 0.), q(0., 5.), q(0., 25.)];
        sort_boxes(&mut boxes);
        assert_eq!(boxes, [q(0., 5.), q(30., 0.), q(0., 25.), q(20., 21.)]);
    }
}
