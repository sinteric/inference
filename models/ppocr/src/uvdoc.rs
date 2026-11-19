//! UVDoc document unwarping: flatten a photographed page.
//!
//! The pinned export takes a page as RGB in `[0, 1]` at any size and returns
//! the rectified page at the same size. Inside, it resizes the page to
//! 712 x 488 (bilinear, align-corners), predicts a 45 x 31 grid of normalized
//! source positions, brings that grid back to the page's size (bilinear,
//! align-corners) and samples the page through it (bilinear, zeros padding,
//! align-corners). The network runs on oneDNN; the two resizes and the
//! sample run on the host, the grid resize and the sample fused
//! (`rt_image::grid_sample_coarse_align_corners`).
//!
//! The rectified page is a different raster from the one handed in: a box
//! found on it does not address the caller's pixels. When to keep it is the
//! caller's decision.
use crate::net::{self, OcrNet, Tensor};
use crate::uvdoc_graph::{
    self, Nets, BRIDGE_CHANNELS, GRID_HEIGHT, GRID_WIDTH, HEAD_CHANNELS, HEAD_PAD, INPUT_HEIGHT,
    INPUT_WIDTH,
};
use crate::weights::{AssetConfig, Assets, ModelKind};
use crate::{Error, Result};
use rt_image::{ImageView, PixelFormat};
use rt_onednn::conv::PlanInfo;
use rt_onednn::{Hardware, OnednnCpu};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Unwarper settings. The network's shapes are fixed by the export, so there
/// is nothing to tune about the model itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct UvdocOptions {
    /// Cached shape plans per network. Every network runs one fixed shape.
    pub plan_capacity: usize,
    /// Bound the page's pixels before preprocessing allocates.
    pub max_pixels: usize,
}
impl Default for UvdocOptions {
    fn default() -> Self {
        Self {
            plan_capacity: 2,
            max_pixels: 64 * 1024 * 1024,
        }
    }
}
impl UvdocOptions {
    /// Refuse a zero plan capacity or pixel bound.
    pub fn validate(&self) -> Result<()> {
        if self.plan_capacity == 0 || self.max_pixels == 0 {
            return Err(Error::Config("invalid unwarper options".into()));
        }
        Ok(())
    }
    fn pixels(&self, w: usize, h: usize) -> Result<()> {
        let n = rt_core::checked_size("unwarper pixels", &[w, h])?;
        if n > self.max_pixels {
            return Err(Error::Shape(format!(
                "unwarper {w}x{h} exceeds max_pixels {}",
                self.max_pixels
            )));
        }
        Ok(())
    }
}

/// A rectified page: packed 8-bit pixels in the input's format and size.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct UnwarpedImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
}
impl UnwarpedImage {
    /// Borrow the pixels as an [`ImageView`], ready for the detector.
    pub fn view(&self) -> Result<ImageView<'_>> {
        ImageView::new(
            &self.data,
            self.width,
            self.height,
            self.width as usize * self.format.channels(),
            self.format,
        )
    }
}

/// Where [`Unwarper::unwarp_timed`] spent its time, in milliseconds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct UvdocTimings {
    /// Unit-range planes and the 712 x 488 resize.
    pub preprocess_ms: f64,
    /// The three networks and the head's reflect pads.
    pub forward_ms: f64,
    /// Grid resize, sample and 8-bit packing.
    pub sample_ms: f64,
}

/// The UVDoc unwarper: packed weights on one oneDNN backend.
///
/// `Send + Sync`; calls block. Values depend on the input and the ISA, never
/// on the thread count.
pub struct Unwarper {
    assets: Assets,
    nets: Nets,
    options: UvdocOptions,
    threads: usize,
    cpu: Arc<OnednnCpu>,
}
impl Unwarper {
    /// Load and pack fixed-layout weights on an explicitly configured backend.
    pub fn load(
        cpu: Arc<OnednnCpu>,
        model_dir: impl AsRef<Path>,
        options: UvdocOptions,
        verify_weights: bool,
    ) -> Result<Self> {
        options.validate()?;
        let mut config = AssetConfig::new(model_dir.as_ref(), ModelKind::Uvdoc);
        config.verify_weights = verify_weights;
        let assets = Assets::load(&config)?;
        Self::from_assets(cpu, assets, options)
    }
    pub(crate) fn from_assets(
        cpu: Arc<OnednnCpu>,
        assets: Assets,
        options: UvdocOptions,
    ) -> Result<Self> {
        options.validate()?;
        let threads = cpu.threads();
        let nets = uvdoc_graph::build(cpu.clone(), &assets, options.plan_capacity)?;
        Ok(Self {
            assets,
            nets,
            options,
            threads,
            cpu,
        })
    }
    /// The verified asset set the weights were packed from.
    pub fn assets(&self) -> &Assets {
        &self.assets
    }
    /// The options the model was loaded with.
    pub fn options(&self) -> &UvdocOptions {
        &self.options
    }
    /// Backend counters (arena, scratch, packed weights, cached plans) for
    /// drivers and diagnostics.
    #[doc(hidden)]
    pub fn backend_stats(&self) -> serde_json::Value {
        self.cpu.stats_json()
    }
    /// Build every network's plan; the shapes are fixed, so this is every
    /// plan the model will use.
    pub fn warmup(&self) -> Result<Vec<PlanInfo>> {
        let (ph, pw) = (GRID_HEIGHT + 2 * HEAD_PAD, GRID_WIDTH + 2 * HEAD_PAD);
        Ok(vec![
            self.nets
                .backbone
                .warmup([1, 3, INPUT_HEIGHT, INPUT_WIDTH])?,
            self.nets.head.warmup([1, BRIDGE_CHANNELS, ph, pw])?,
            self.nets.points.warmup([1, HEAD_CHANNELS, ph, pw])?,
        ])
    }
    fn validate_shape(&self, shape: &[usize], len: usize) -> Result<()> {
        net::validate(shape, len)?;
        rt_core::check_shape!(
            shape.len() == 4 && shape[0] == 1 && shape[1] == 3,
            "unwarper requires NCHW RGB [1, 3, H, W]"
        );
        self.options.pixels(shape[3], shape[2])
    }
    /// The page as unit-range RGB planes, `[1, 3, H, W]`.
    pub fn preprocess(&self, image: &ImageView<'_>) -> Result<Tensor> {
        self.options
            .pixels(image.width() as usize, image.height() as usize)?;
        Tensor::new(
            vec![1, 3, image.height() as usize, image.width() as usize],
            rt_image::rgb_unit_nchw_threads(image, self.threads)?,
        )
    }
    /// The model's grid for a preprocessed page: `[1, 2, 45, 31]`, plane 0
    /// the normalized source x, plane 1 the source y, both in about
    /// `[-1, 1]`.
    pub fn grid(&self, shape: &[usize], input: &[f32]) -> Result<Tensor> {
        self.validate_shape(shape, input.len())?;
        rt_core::check_shape!(
            crate::rec::all_par(self.threads, input, f32::is_finite),
            "nonfinite unwarper input"
        );
        let resized = rt_image::resize_linear_align_corners_nchw(
            input,
            3,
            shape[2],
            shape[3],
            INPUT_HEIGHT,
            INPUT_WIDTH,
            self.threads,
        )?;
        self.network(&resized)
    }
    fn network(&self, resized: &[f32]) -> Result<Tensor> {
        let (gh, gw) = (GRID_HEIGHT, GRID_WIDTH);
        let c = BRIDGE_CHANNELS;
        let (_, fused) = self
            .nets
            .backbone
            .run([1, 3, INPUT_HEIGHT, INPUT_WIDTH], resized)?;
        let padded = rt_image::reflect_pad_nchw(&fused, c, gh, gw, HEAD_PAD)?;
        let (ph, pw) = (gh + 2 * HEAD_PAD, gw + 2 * HEAD_PAD);
        let (_, hidden) = self.nets.head.run([1, c, ph, pw], &padded)?;
        let padded = rt_image::reflect_pad_nchw(&hidden, HEAD_CHANNELS, gh, gw, HEAD_PAD)?;
        let (shape, grid) = self.nets.points.run([1, HEAD_CHANNELS, ph, pw], &padded)?;
        rt_core::check_shape!(
            shape == [1, 2, gh, gw] && grid.iter().all(|v| v.is_finite()),
            "unwarper grid shape or values"
        );
        Tensor::new(shape.to_vec(), grid)
    }
    /// Sample unit-range planes `[1, 3, H, W]` through `grid`.
    pub fn sample(&self, shape: &[usize], input: &[f32], grid: &Tensor) -> Result<Tensor> {
        self.validate_shape(shape, input.len())?;
        rt_core::check_shape!(
            grid.shape == [1, 2, GRID_HEIGHT, GRID_WIDTH],
            "unwarper grid shape"
        );
        let values = rt_image::grid_sample_coarse_align_corners(
            input,
            3,
            shape[2],
            shape[3],
            &grid.values,
            GRID_HEIGHT,
            GRID_WIDTH,
            self.threads,
        )?;
        Tensor::new(shape.to_vec(), values)
    }
    /// Rectify `image`: the same size and pixel format, each value the
    /// model's sample clamped to `[0, 1]` and rounded to 8 bits.
    pub fn unwarp(&self, image: &ImageView<'_>) -> Result<UnwarpedImage> {
        self.unwarp_timed(image).map(|(image, _)| image)
    }
    /// [`Unwarper::unwarp`] plus where the time went.
    pub fn unwarp_timed(&self, image: &ImageView<'_>) -> Result<(UnwarpedImage, UvdocTimings)> {
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1000.;
        let t = Instant::now();
        let input = self.preprocess(image)?;
        let resized = rt_image::resize_linear_align_corners_nchw(
            &input.values,
            3,
            input.shape[2],
            input.shape[3],
            INPUT_HEIGHT,
            INPUT_WIDTH,
            self.threads,
        )?;
        let preprocess_ms = ms(t);
        let t = Instant::now();
        let grid = self.network(&resized)?;
        drop(resized);
        let forward_ms = ms(t);
        let t = Instant::now();
        let planes = self.sample(&input.shape, &input.values, &grid)?;
        drop(input);
        let data = rt_image::pack_unit_nchw(
            &planes.values,
            image.width(),
            image.height(),
            image.format(),
            self.threads,
        )?;
        Ok((
            UnwarpedImage {
                data,
                width: image.width(),
                height: image.height(),
                format: image.format(),
            },
            UvdocTimings {
                preprocess_ms,
                forward_ms,
                sample_ms: ms(t),
            },
        ))
    }
}
/// The whole export: unit-range RGB `[1, 3, H, W]` in, the rectified page
/// `[1, 3, H, W]` out.
impl OcrNet for Unwarper {
    fn forward(&self, shape: &[usize], input: &[f32]) -> Result<Tensor> {
        let grid = self.grid(shape, input)?;
        self.sample(shape, input, &grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_validate() {
        assert!(UvdocOptions::default().validate().is_ok());
        for options in [
            UvdocOptions {
                plan_capacity: 0,
                ..Default::default()
            },
            UvdocOptions {
                max_pixels: 0,
                ..Default::default()
            },
        ] {
            assert!(options.validate().is_err());
        }
        let small = UvdocOptions {
            max_pixels: 100,
            ..Default::default()
        };
        assert!(small.pixels(10, 10).is_ok());
        assert!(matches!(small.pixels(11, 10), Err(Error::Shape(_))));
    }

    /// The grid's size is the backbone's four stride-2 stages applied to the
    /// export's fixed input; the reflect-padded head restores it exactly.
    #[test]
    fn grid_size_follows_the_fixed_input() {
        let stage = |n: usize| (n + 2 * 2 - 5) / 2 + 1;
        let down = |n: usize| stage(stage(stage(stage(n))));
        assert_eq!(down(INPUT_HEIGHT), GRID_HEIGHT);
        assert_eq!(down(INPUT_WIDTH), GRID_WIDTH);
    }
}
