//! Decoded pixel views, deterministic resampling, and OCR geometry.
//! No codecs, environment access, or filesystem lookups are used here.
pub use rt_core::{Error, Result};
mod bicubic;
mod crop;
mod geometry;
mod warp;
pub use bicubic::{resize_bicubic_aa_rgb, to_rgb};
pub use crop::*;
pub use geometry::*;
pub use warp::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Bgr8,
    Rgb8,
    Gray8,
}
impl PixelFormat {
    pub fn channels(self) -> usize {
        if self == Self::Gray8 {
            1
        } else {
            3
        }
    }
}

/// Borrowed decoded pixels. Construction validates the entire strided extent.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: usize,
    format: PixelFormat,
}
impl<'a> ImageView<'a> {
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
    ) -> Result<Self> {
        rt_core::check_shape!(width > 0 && height > 0, "empty image");
        let row = rt_core::checked_size("image row", &[width as usize, format.channels()])?;
        rt_core::check_shape!(stride >= row, "image stride {stride} < row bytes {row}");
        let extent = (height as usize - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(row))
            .ok_or(Error::SizeOverflow("image extent"))?;
        rt_core::check_shape!(
            extent <= isize::MAX as usize && data.len() >= extent,
            "image buffer is shorter than strided extent"
        );
        Ok(Self {
            data,
            width,
            height,
            stride,
            format,
        })
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub fn format(&self) -> PixelFormat {
        self.format
    }
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
    fn bgr(&self, x: usize, y: usize, c: usize) -> u8 {
        let c = match self.format {
            PixelFormat::Gray8 => 0,
            PixelFormat::Bgr8 => c,
            PixelFormat::Rgb8 => 2 - c,
        };
        self.data[y * self.stride + x * self.format.channels() + c]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

/// Nearest multiple of 32 (ties to even), with an optional downscale-only cap.
pub fn detector_size(src: ImageSize, max_side: Option<u32>) -> Result<ImageSize> {
    rt_core::check_shape!(
        src.width > 0 && src.height > 0 && max_side != Some(0),
        "invalid detector resize dimensions"
    );
    let scale = max_side.map_or(1., |m| {
        (m as f64 / src.width.max(src.height) as f64).min(1.)
    });
    let side = |s: u32| -> Result<u32> {
        let n = ((s as f64 * scale / 32.).round_ties_even().max(1.) * 32.) as u64;
        u32::try_from(n).map_err(|_| Error::SizeOverflow("detector resize side"))
    };
    Ok(ImageSize {
        width: side(src.width)?,
        height: side(src.height)?,
    })
}

// The sample positions, f32 weights, vertical f32 intermediate, final u8
// rounding and antialias support match image 0.25's FilterType::Triangle.
fn taps(src: usize, dst: usize, i: usize, weights: &mut [f32]) -> (usize, usize) {
    let ratio = src as f32 / dst as f32;
    let support = ratio.max(1.);
    let center = (i as f32 + 0.5) * ratio;
    let left = ((center - support).floor() as i64).clamp(0, src as i64 - 1) as usize;
    let right = ((center + support).ceil() as i64).clamp(left as i64 + 1, src as i64) as usize;
    let center = center - 0.5;
    let mut sum = 0.;
    for (j, w) in weights[..right - left].iter_mut().enumerate() {
        *w = (1. - ((left + j) as f32 - center).abs() / support).max(0.);
        sum += *w;
    }
    for w in &mut weights[..right - left] {
        *w /= sum;
    }
    (left, right)
}

/// Resize decoded pixels to interleaved BGR8. Gray is replicated, RGB swapped.
/// Triangle is separable, vertical then horizontal, with antialiasing.
pub fn resize_triangle_bgr(src: &ImageView<'_>, width: u32, height: u32) -> Result<Vec<u8>> {
    resize_triangle_bgr_threads(src, width, height, 1)
}

/// [`resize_triangle_bgr`] split by output rows across `threads`. Every
/// output pixel uses the same taps in the same order at any thread count.
pub fn resize_triangle_bgr_threads(
    src: &ImageView<'_>,
    width: u32,
    height: u32,
    threads: usize,
) -> Result<Vec<u8>> {
    rt_core::check_shape!(width > 0 && height > 0, "empty resize destination");
    let (sw, sh, w, h) = (
        src.width as usize,
        src.height as usize,
        width as usize,
        height as usize,
    );
    let mut out = rt_core::try_vec(rt_core::checked_size("resized image", &[w, h, 3])?)?;
    if w == sw && h == sh {
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    out[(y * w + x) * 3 + c] = src.bgr(x, y, c);
                }
            }
        }
        return Ok(out);
    }
    let vertical = tap_table(sh, h)?;
    let horizontal = tap_table(sw, w)?;
    let mut tmp =
        rt_core::try_vec::<f32>(rt_core::checked_size("triangle intermediate", &[sw, h, 3])?)?;
    let tmp_ptr = tmp.as_mut_ptr() as usize;
    // Vertical pass: each intermediate value starts at 0 and adds the source
    // taps in ascending order, as the per-pixel loop did.
    rt_threads::par_for(threads, h, 8, |lo, hi| {
        for y in lo..hi {
            // SAFETY: rows [lo, hi) are disjoint across workers; `tmp` holds h rows of sw * 3.
            let row = unsafe {
                std::slice::from_raw_parts_mut((tmp_ptr as *mut f32).add(y * sw * 3), sw * 3)
            };
            let (left, weights) = vertical.get(y);
            for (i, &weight) in weights.iter().enumerate() {
                let sy = left + i;
                let line = &src.data[sy * src.stride..];
                match src.format {
                    PixelFormat::Bgr8 => {
                        for (v, &p) in row.iter_mut().zip(&line[..sw * 3]) {
                            *v += p as f32 * weight;
                        }
                    }
                    PixelFormat::Rgb8 => {
                        for (v, p) in row
                            .as_chunks_mut::<3>()
                            .0
                            .iter_mut()
                            .zip(line[..sw * 3].as_chunks::<3>().0)
                        {
                            v[0] += p[2] as f32 * weight;
                            v[1] += p[1] as f32 * weight;
                            v[2] += p[0] as f32 * weight;
                        }
                    }
                    PixelFormat::Gray8 => {
                        for (v, &p) in row.as_chunks_mut::<3>().0.iter_mut().zip(&line[..sw]) {
                            let p = p as f32 * weight;
                            v[0] += p;
                            v[1] += p;
                            v[2] += p;
                        }
                    }
                }
            }
        }
    });
    let out_ptr = out.as_mut_ptr() as usize;
    rt_threads::par_for(threads, h, 8, |lo, hi| {
        for y in lo..hi {
            // SAFETY: rows [lo, hi) are disjoint across workers; `out` holds h rows of w * 3.
            let dst = unsafe {
                std::slice::from_raw_parts_mut((out_ptr as *mut u8).add(y * w * 3), w * 3)
            };
            let row = &tmp[y * sw * 3..(y + 1) * sw * 3];
            for (x, px) in dst.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let (left, weights) = horizontal.get(x);
                for (c, v) in px.iter_mut().enumerate() {
                    let mut value = 0.;
                    for (i, weight) in weights.iter().enumerate() {
                        value += row[(left + i) * 3 + c] * weight;
                    }
                    *v = value.clamp(0., 255.).round() as u8;
                }
            }
        }
    });
    Ok(out)
}

/// Triangle taps for every destination index: first source index and weights.
struct TapTable {
    first: Vec<(usize, usize, usize)>,
    weights: Vec<f32>,
}
impl TapTable {
    fn get(&self, i: usize) -> (usize, &[f32]) {
        let (left, start, len) = self.first[i];
        (left, &self.weights[start..start + len])
    }
}
fn tap_table(src: usize, dst: usize) -> Result<TapTable> {
    let mut scratch = rt_core::try_vec::<f32>(src)?;
    let mut first = Vec::with_capacity(dst);
    let mut weights = Vec::new();
    for i in 0..dst {
        let (left, right) = taps(src, dst, i, &mut scratch);
        first.push((left, weights.len(), right - left));
        weights.extend_from_slice(&scratch[..right - left]);
    }
    Ok(TapTable { first, weights })
}

/// Convert pixels to BGR NCHW f32 using `(v / 255 - mean) / std`.
pub fn normalize_nchw(src: &ImageView<'_>, mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>> {
    normalize_nchw_threads(src, mean, std, 1)
}

/// [`normalize_nchw`] split by rows across `threads`; same values.
pub fn normalize_nchw_threads(
    src: &ImageView<'_>,
    mean: [f32; 3],
    std: [f32; 3],
    threads: usize,
) -> Result<Vec<f32>> {
    rt_core::check_shape!(
        mean.iter().all(|v| v.is_finite()) && std.iter().all(|v| v.is_finite() && *v > 0.),
        "invalid normalization constants"
    );
    let (w, h) = (src.width as usize, src.height as usize);
    let plane = rt_core::checked_size("image plane", &[w, h])?;
    let mut out = rt_core::try_vec(rt_core::checked_size("NCHW image", &[3, plane])?)?;
    let out_ptr = out.as_mut_ptr() as usize;
    rt_threads::par_for(threads, h, 16, |lo, hi| {
        for c in 0..3 {
            // SAFETY: rows [lo, hi) of each plane are disjoint across workers.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_ptr as *mut f32).add(c * plane + lo * w),
                    (hi - lo) * w,
                )
            };
            for (y, row) in (lo..hi).zip(dst.chunks_exact_mut(w)) {
                for (x, v) in row.iter_mut().enumerate() {
                    *v = (src.bgr(x, y, c) as f32 / 255. - mean[c]) / std[c];
                }
            }
        }
    });
    Ok(out)
}

/// Legacy keep-ratio helper, rounding upwards to the supplied stride.
pub fn keep_ratio_hw(src: ImageSize, max_side: u32, stride: u32) -> Result<ImageSize> {
    rt_core::check_shape!(
        src.width > 0 && src.height > 0 && max_side > 0 && stride > 0,
        "invalid resize dimensions"
    );
    let scale = max_side as f32 / src.width.max(src.height) as f32;
    let side = |s: u32| -> Result<u32> {
        ((s as f32 * scale) as u32)
            .max(stride)
            .div_ceil(stride)
            .checked_mul(stride)
            .ok_or(Error::SizeOverflow("resize side"))
    };
    Ok(ImageSize {
        width: side(src.width)?,
        height: side(src.height)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The per-pixel resize these row-parallel loops replaced.
    fn resize_reference(src: &ImageView<'_>, w: usize, h: usize) -> Vec<u8> {
        let (sw, sh) = (src.width as usize, src.height as usize);
        let mut out = vec![0u8; w * h * 3];
        let mut tmp = vec![0f32; sw * h * 3];
        let mut weights = vec![0f32; sw.max(sh)];
        for y in 0..h {
            let (left, right) = taps(sh, h, y, &mut weights);
            for x in 0..sw {
                for c in 0..3 {
                    let mut value = 0.;
                    for (i, weight) in weights[..right - left].iter().enumerate() {
                        value += src.bgr(x, left + i, c) as f32 * weight;
                    }
                    tmp[(y * sw + x) * 3 + c] = value;
                }
            }
        }
        for x in 0..w {
            let (left, right) = taps(sw, w, x, &mut weights);
            for y in 0..h {
                for c in 0..3 {
                    let mut value = 0.;
                    for (i, weight) in weights[..right - left].iter().enumerate() {
                        value += tmp[(y * sw + left + i) * 3 + c] * weight;
                    }
                    out[(y * w + x) * 3 + c] = value.clamp(0., 255.).round() as u8;
                }
            }
        }
        out
    }
    #[test]
    fn threaded_resize_and_normalize_match_per_pixel_loops() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        };
        for (format, sw, sh, w, h) in [
            (PixelFormat::Bgr8, 97, 61, 64, 32),
            (PixelFormat::Rgb8, 50, 77, 96, 128),
            (PixelFormat::Gray8, 131, 45, 32, 64),
            (PixelFormat::Bgr8, 40, 40, 160, 96),
        ] {
            let stride = sw * format.channels() + 5;
            let data: Vec<u8> = (0..stride * sh).map(|_| next()).collect();
            let view = ImageView::new(&data, sw as u32, sh as u32, stride, format).unwrap();
            let expected = resize_reference(&view, w, h);
            for threads in [1, 3, 4] {
                let got = resize_triangle_bgr_threads(&view, w as u32, h as u32, threads).unwrap();
                assert_eq!(got, expected, "{format:?} {sw}x{sh}->{w}x{h} t{threads}");
            }
            let norm = normalize_nchw(&view, [0.1, 0.2, 0.3], [0.5, 0.6, 0.7]).unwrap();
            for threads in [2, 4] {
                let got = normalize_nchw_threads(&view, [0.1, 0.2, 0.3], [0.5, 0.6, 0.7], threads)
                    .unwrap();
                assert!(got
                    .iter()
                    .zip(&norm)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }
    #[test]
    fn validated_strides_colors_and_normalization() {
        let bgr = [10, 20, 30, 40, 50, 60, 99, 99, 70, 80, 90, 100, 110, 120];
        let view = ImageView::new(&bgr, 2, 2, 8, PixelFormat::Bgr8).unwrap();
        assert_eq!(
            resize_triangle_bgr(&view, 2, 2).unwrap(),
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        );
        let rgb = [30, 20, 10, 60, 50, 40];
        let rgb = ImageView::new(&rgb, 2, 1, 6, PixelFormat::Rgb8).unwrap();
        assert_eq!(
            resize_triangle_bgr(&rgb, 2, 1).unwrap(),
            [10, 20, 30, 40, 50, 60]
        );
        let gray = [0, 255];
        let gray = ImageView::new(&gray, 2, 1, 2, PixelFormat::Gray8).unwrap();
        assert_eq!(
            normalize_nchw(&gray, [0.5; 3], [0.5; 3]).unwrap(),
            [-1., 1., -1., 1., -1., 1.]
        );
        assert!(normalize_nchw(&gray, [0.; 3], [0.; 3]).is_err());
        let one = [121u8];
        let one = ImageView::new(&one, 1, 1, 1, PixelFormat::Gray8).unwrap();
        assert!(resize_triangle_bgr(&one, 17, 3)
            .unwrap()
            .iter()
            .all(|v| *v == 121));
    }
    #[test]
    fn malformed_and_huge_images_return_errors() {
        for (w, h, stride) in [
            (0, 1, 0),
            (1, 0, 3),
            (1, 1, 2),
            (2, 2, 6),
            (1, 2, usize::MAX),
            (u32::MAX, u32::MAX, usize::MAX),
        ] {
            assert!(ImageView::new(&[0; 3], w, h, stride, PixelFormat::Bgr8).is_err());
        }
        let one = ImageView::new(&[0], 1, 1, 1, PixelFormat::Gray8).unwrap();
        assert!(resize_triangle_bgr(&one, 0, 1).is_err());
        assert!(resize_triangle_bgr(&one, u32::MAX, u32::MAX).is_err());
        assert!(detector_size(
            ImageSize {
                width: u32::MAX,
                height: 1
            },
            None
        )
        .is_err());
        assert!(detector_size(
            ImageSize {
                width: 1,
                height: 1
            },
            Some(0)
        )
        .is_err());
        assert!(contours(&[], usize::MAX, 1).is_err());
        assert!(contours(&[1], 2, 2).is_err());
        assert!(min_area_rect(&[[f32::NAN, 0.]]).is_err());
        assert!(unclip([[0., 0.]; 4], f32::INFINITY).is_err());
        assert!(polygon_mean_score(&[], 1, 1, [[0., 0.]; 4]).is_err());
    }
    #[test]
    fn detector_rounding_cap_and_antialias() {
        assert_eq!(
            detector_size(
                ImageSize {
                    width: 80,
                    height: 112
                },
                None
            )
            .unwrap(),
            ImageSize {
                width: 64,
                height: 128
            }
        );
        assert_eq!(
            detector_size(
                ImageSize {
                    width: 2400,
                    height: 1696
                },
                None
            )
            .unwrap(),
            ImageSize {
                width: 2400,
                height: 1696
            }
        );
        assert_eq!(
            detector_size(
                ImageSize {
                    width: 2400,
                    height: 1600
                },
                Some(960)
            )
            .unwrap(),
            ImageSize {
                width: 960,
                height: 640
            }
        );
        assert_eq!(
            detector_size(
                ImageSize {
                    width: 32,
                    height: 32
                },
                Some(960)
            )
            .unwrap(),
            ImageSize {
                width: 32,
                height: 32
            }
        );
        let bytes = [0, 0, 255, 255];
        let image = ImageView::new(&bytes, 4, 1, 4, PixelFormat::Gray8).unwrap();
        assert_eq!(
            resize_triangle_bgr(&image, 2, 1).unwrap(),
            [36, 36, 36, 219, 219, 219]
        );
    }
    #[test]
    fn borders_holes_diagonals_and_rectangles() {
        assert!(contours(&[0; 9], 3, 3).unwrap().is_empty());
        assert_eq!(contours(&[1], 1, 1).unwrap(), vec![vec![[0., 0.]]]);
        assert_eq!(
            contours(&[1, 0, 0, 1], 2, 2).unwrap(),
            vec![vec![[0., 0.], [1., 1.]]]
        );
        let mut mask = vec![1; 49];
        for y in 2..5 {
            for x in 2..5 {
                mask[y * 7 + x] = 0;
            }
        }
        let c = contours(&mask, 7, 7).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[1].len(), 4);
        let q = min_area_rect(&c[1]).unwrap().unwrap().0;
        for (actual, expected) in q
            .iter()
            .flatten()
            .zip([[0., 0.], [6., 0.], [6., 6.], [0., 6.]].iter().flatten())
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(polygon_mean_score(&vec![0.7; 49], 7, 7, q).unwrap(), 0.7);
        assert_eq!(
            convex_hull(&[[0., 0.], [1., 0.], [2., 0.]]).unwrap(),
            vec![[0., 0.], [2., 0.]]
        );
        assert!(min_area_rect(&[[0., 0.], [1., 0.], [2., 0.]])
            .unwrap()
            .is_none());
    }
}
