//! Unit-range planes, align-corners bilinear resize and the bilinear grid
//! sample a dense-grid unwarper (UVDoc) applies. Coordinates and weights
//! follow ONNX Runtime 1.28's CPU `Resize` (linear, `align_corners`) and
//! `GridSample` (bilinear, zeros padding, `align_corners = 1`) kernels,
//! including their FP32 operation order.
use crate::{ImageView, PixelFormat, Result};
use rt_core::{checked_size, try_vec};

/// Planar RGB in `[0, 1]`: `v / 255`, no mean or deviation. Gray is
/// replicated and BGR swapped, so plane 0 is always red.
pub fn rgb_unit_nchw_threads(src: &ImageView<'_>, threads: usize) -> Result<Vec<f32>> {
    let (w, h) = (src.width as usize, src.height as usize);
    let plane = checked_size("image plane", &[w, h])?;
    let mut out = try_vec(checked_size("NCHW image", &[3, plane])?)?;
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
                    // `bgr` indexes B, G, R; plane `c` is R, G, B.
                    *v = src.bgr(x, y, 2 - c) as f32 / 255.;
                }
            }
        }
    });
    Ok(out)
}

/// One axis of an align-corners linear resize: for each output index the
/// two source indices and their weights, as ORT's `SetupUpsampleBilinear`
/// computes them.
struct Axis {
    lo: Vec<usize>,
    hi: Vec<usize>,
    d_lo: Vec<f32>,
    d_hi: Vec<f32>,
}
impl Axis {
    fn new(input: usize, output: usize) -> Result<Self> {
        rt_core::check_shape!(
            input > 0 && output > 0 && input <= 1 << 24 && output <= 1 << 24,
            "invalid align-corners resize axis {input} -> {output}"
        );
        let (mut lo, mut hi) = (try_vec(output)?, try_vec(output)?);
        let (mut d_lo, mut d_hi) = (try_vec(output)?, try_vec(output)?);
        let (li, lo_len) = (input as f32, output as f32);
        for i in 0..output {
            // ORT special-cases a scale of exactly 1 to the identity; the
            // align-corners formula gives the same coordinate there.
            let x = if output == input {
                i as f32
            } else if output == 1 {
                0.
            } else {
                i as f32 * (li - 1.) / (lo_len - 1.)
            };
            let x = x.clamp(0., li - 1.);
            let a = (x as usize).min(input - 1);
            let b = (a + 1).min(input - 1);
            let (da, db) = if a == b {
                (0.5, 0.5)
            } else {
                ((x - a as f32).abs(), (x - b as f32).abs())
            };
            lo[i] = a;
            hi[i] = b;
            d_lo[i] = da;
            d_hi[i] = db;
        }
        Ok(Self { lo, hi, d_lo, d_hi })
    }
}

/// Align-corners bilinear resize of `channels` planes of `height x width`
/// to `out_height x out_width` (ONNX `Resize`, mode `linear`,
/// `coordinate_transformation_mode = align_corners`, no antialias).
pub fn resize_linear_align_corners_nchw(
    src: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
    threads: usize,
) -> Result<Vec<f32>> {
    let plane = checked_size("resize input plane", &[height, width])?;
    rt_core::check_shape!(
        channels > 0 && src.len() == checked_size("resize input", &[channels, plane])?,
        "align-corners resize input extent"
    );
    let (ys, xs) = (Axis::new(height, out_height)?, Axis::new(width, out_width)?);
    let out_plane = checked_size("resize output plane", &[out_height, out_width])?;
    let mut out = try_vec(checked_size("resize output", &[channels, out_plane])?)?;
    let out_ptr = out.as_mut_ptr() as usize;
    let rows = checked_size("resize rows", &[channels, out_height])?;
    rt_threads::par_for(threads, rows, 8, |lo, hi| {
        for r in lo..hi {
            let (c, y) = (r / out_height, r % out_height);
            let x_plane = &src[c * plane..(c + 1) * plane];
            // SAFETY: each output row is written by exactly one worker.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_ptr as *mut f32).add(c * out_plane + y * out_width),
                    out_width,
                )
            };
            let (y1, y2) = (ys.lo[y] * width, ys.hi[y] * width);
            let (dy1, dy2) = (ys.d_lo[y], ys.d_hi[y]);
            for (x, v) in dst.iter_mut().enumerate() {
                let (x1, x2) = (xs.lo[x], xs.hi[x]);
                let (dx1, dx2) = (xs.d_lo[x], xs.d_hi[x]);
                *v = dx2 * dy2 * x_plane[y1 + x1]
                    + dx1 * dy2 * x_plane[y1 + x2]
                    + dx2 * dy1 * x_plane[y2 + x1]
                    + dx1 * dy1 * x_plane[y2 + x2];
            }
        }
    });
    Ok(out)
}

/// Sample `image` (`channels` planes of `height x width`) through a coarse
/// normalized grid (`[2, grid_height, grid_width]`, plane 0 x, plane 1 y,
/// both in `[-1, 1]`) brought to the image's own size.
///
/// Equal to ONNX `Resize` (linear, `align_corners`) of the grid to
/// `height x width`, `Transpose` to NHWC, then `GridSample` (bilinear, zeros
/// padding, `align_corners = 1`) of `image`, fused so the full-size grid is
/// never stored. Each output value is computed by the same FP32 operations.
pub fn grid_sample_coarse_align_corners(
    image: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    grid: &[f32],
    grid_height: usize,
    grid_width: usize,
    threads: usize,
) -> Result<Vec<f32>> {
    let plane = checked_size("grid sample plane", &[height, width])?;
    let grid_plane = checked_size("grid plane", &[grid_height, grid_width])?;
    rt_core::check_shape!(
        channels > 0
            && image.len() == checked_size("grid sample input", &[channels, plane])?
            && grid.len() == checked_size("grid", &[2, grid_plane])?,
        "grid sample extent"
    );
    rt_core::check_shape!(
        i64::try_from(height).is_ok() && i64::try_from(width).is_ok(),
        "grid sample dimensions"
    );
    let (ys, xs) = (
        Axis::new(grid_height, height)?,
        Axis::new(grid_width, width)?,
    );
    let mut out = try_vec(checked_size("grid sample output", &[channels, plane])?)?;
    let out_ptr = out.as_mut_ptr() as usize;
    // ORT replaces a coordinate too large for an int64 floor with the low
    // border, 0 under align_corners.
    let safe = |v: f32| {
        let bound = (1i64 << 62) as f32;
        if v.is_finite() && (-bound..=bound).contains(&v) {
            v
        } else {
            0.
        }
    };
    let (h, w) = (height as i64, width as i64);
    let (wm1, hm1) = ((width - 1) as f32, (height - 1) as f32);
    rt_threads::par_for(threads, height, 4, |lo, hi| {
        for oy in lo..hi {
            let (gy1, gy2) = (ys.lo[oy] * grid_width, ys.hi[oy] * grid_width);
            let (dy1, dy2) = (ys.d_lo[oy], ys.d_hi[oy]);
            for ox in 0..width {
                let (gx1, gx2) = (xs.lo[ox], xs.hi[ox]);
                let (dx1, dx2) = (xs.d_lo[ox], xs.d_hi[ox]);
                let at = |p: &[f32]| {
                    dx2 * dy2 * p[gy1 + gx1]
                        + dx1 * dy2 * p[gy1 + gx2]
                        + dx2 * dy1 * p[gy2 + gx1]
                        + dx1 * dy1 * p[gy2 + gx2]
                };
                let nx = at(&grid[..grid_plane]);
                let ny = at(&grid[grid_plane..]);
                let x = safe((nx + 1.) / 2. * wm1);
                let y = safe((ny + 1.) / 2. * hm1);
                let (x1, y1) = (x.floor() as i64, y.floor() as i64);
                let (x2, y2) = (x1 + 1, y1 + 1);
                let dx2 = x2 as f32 - x;
                let dx1 = x - x1 as f32;
                let dy2 = y2 as f32 - y;
                let dy1 = y - y1 as f32;
                let index = |r: i64, c: i64| {
                    (r >= 0 && r < h && c >= 0 && c < w).then(|| r as usize * width + c as usize)
                };
                let (i11, i12, i21, i22) =
                    (index(y1, x1), index(y1, x2), index(y2, x1), index(y2, x2));
                for c in 0..channels {
                    let p = &image[c * plane..(c + 1) * plane];
                    let px = |i: Option<usize>| i.map_or(0., |i| p[i]);
                    let v = dy2 * (dx2 * px(i11) + dx1 * px(i12))
                        + dy1 * (dx2 * px(i21) + dx1 * px(i22));
                    // SAFETY: row `oy` of every plane is written by one worker.
                    unsafe { *(out_ptr as *mut f32).add(c * plane + oy * width + ox) = v };
                }
            }
        }
    });
    Ok(out)
}

/// Pack unit-range RGB planes into interleaved 8-bit pixels of `format`,
/// each value clamped to `[0, 1]` and rounded half away from zero. Gray8
/// takes the red plane: a gray input was replicated into all three.
pub fn pack_unit_nchw(
    planes: &[f32],
    width: u32,
    height: u32,
    format: PixelFormat,
    threads: usize,
) -> Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    let plane = checked_size("packed plane", &[w, h])?;
    rt_core::check_shape!(
        planes.len() == checked_size("packed planes", &[3, plane])?,
        "packed planes extent"
    );
    let channels = format.channels();
    let mut out = try_vec(checked_size("packed image", &[plane, channels])?)?;
    let out_ptr = out.as_mut_ptr() as usize;
    let to_u8 = |v: f32| (v.clamp(0., 1.) * 255.).round() as u8;
    rt_threads::par_for(threads, h, 16, |lo, hi| {
        // SAFETY: rows [lo, hi) are disjoint across workers.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                (out_ptr as *mut u8).add(lo * w * channels),
                (hi - lo) * w * channels,
            )
        };
        for (y, row) in (lo..hi).zip(dst.chunks_exact_mut(w * channels)) {
            for (x, px) in row.chunks_exact_mut(channels).enumerate() {
                let rgb = [0, 1, 2].map(|c| to_u8(planes[c * plane + y * w + x]));
                match format {
                    PixelFormat::Rgb8 => px.copy_from_slice(&rgb),
                    PixelFormat::Bgr8 => px.copy_from_slice(&[rgb[2], rgb[1], rgb[0]]),
                    PixelFormat::Gray8 => px[0] = rgb[0],
                }
            }
        }
    });
    Ok(out)
}

/// Reflect-pad `channels` planes of `height x width` by `pad` on every side
/// (ONNX `Pad`, mode `reflect`: the edge pixel itself is not repeated).
pub fn reflect_pad_nchw(
    src: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    pad: usize,
) -> Result<Vec<f32>> {
    let plane = checked_size("pad input plane", &[height, width])?;
    rt_core::check_shape!(
        channels > 0
            && src.len() == checked_size("pad input", &[channels, plane])?
            && pad < height
            && pad < width,
        "reflect pad of {pad} on {height}x{width}"
    );
    let (ph, pw) = (height + 2 * pad, width + 2 * pad);
    let mut out = try_vec(checked_size("pad output", &[channels, ph, pw])?)?;
    let reflect = |i: usize, n: usize| -> usize {
        // i indexes the padded axis; map it back into [0, n).
        let i = i as isize - pad as isize;
        let n = n as isize;
        let i = if i < 0 { -i } else { i };
        (if i >= n { 2 * (n - 1) - i } else { i }) as usize
    };
    for c in 0..channels {
        for y in 0..ph {
            let sy = reflect(y, height);
            for x in 0..pw {
                out[(c * ph + y) * pw + x] = src[c * plane + sy * width + reflect(x, width)];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    /// The align-corners resize keeps corners exactly and interpolates
    /// linearly between them; equal sizes are the identity.
    #[test]
    fn align_corners_resize_keeps_the_corners() {
        let src = [0., 1., 2., 3.]; // 2x2
        let out = resize_linear_align_corners_nchw(&src, 1, 2, 2, 3, 3, 1).unwrap();
        assert_eq!(out, [0., 0.5, 1., 1., 1.5, 2., 2., 2.5, 3.]);
        let same = resize_linear_align_corners_nchw(&src, 1, 2, 2, 2, 2, 1).unwrap();
        assert_eq!(same, src);
        // A one-pixel axis reads index 0 at weight 0.5 + 0.5.
        let one = resize_linear_align_corners_nchw(&[4., 6.], 1, 1, 2, 2, 1, 1).unwrap();
        assert_eq!(one, [4., 4.]);
        assert!(resize_linear_align_corners_nchw(&src, 1, 2, 2, 0, 3, 1).is_err());
        assert!(resize_linear_align_corners_nchw(&src[..3], 1, 2, 2, 3, 3, 1).is_err());
    }

    /// The identity grid returns the image; a grid shifted past an edge
    /// reads zeros there, as zeros padding does.
    #[test]
    fn identity_grid_returns_the_image_and_outside_reads_zero() {
        let (h, w) = (4usize, 5usize);
        let image = (0..h * w).map(|v| v as f32).collect::<Vec<_>>();
        // A 2x2 corner grid upsampled with align-corners is the identity.
        let grid = [-1., 1., -1., 1., -1., -1., 1., 1.];
        let out = grid_sample_coarse_align_corners(&image, 1, h, w, &grid, 2, 2, 2).unwrap();
        for (a, b) in out.iter().zip(&image) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Every x pushed one full width to the right: all outside.
        let grid = [3., 3., 3., 3., -1., -1., 1., 1.];
        let out = grid_sample_coarse_align_corners(&image, 1, h, w, &grid, 2, 2, 1).unwrap();
        assert!(out.iter().all(|v| *v == 0.));
        assert!(grid_sample_coarse_align_corners(&image, 1, h, w, &grid[..7], 2, 2, 1).is_err());
    }

    /// Reflect padding mirrors about the edge pixel without repeating it.
    #[test]
    fn reflect_pad_mirrors_about_the_edge() {
        let src = [1., 2., 3., 4., 5., 6.]; // 2x3
        let out = reflect_pad_nchw(&src, 1, 2, 3, 1).unwrap();
        assert_eq!(
            out,
            [
                5., 4., 5., 6., 5., //
                2., 1., 2., 3., 2., //
                5., 4., 5., 6., 5., //
                2., 1., 2., 3., 2.,
            ]
        );
        assert!(reflect_pad_nchw(&src, 1, 2, 3, 2).is_err());
    }

    /// Planes round-trip through 8-bit packing in each format.
    #[test]
    fn unit_planes_round_trip_through_packing() {
        let bgr = [10u8, 20, 30, 40, 50, 60];
        let view = ImageView::new(&bgr, 2, 1, 6, PixelFormat::Bgr8).unwrap();
        let planes = rgb_unit_nchw_threads(&view, 1).unwrap();
        assert_eq!(planes[0], 30. / 255.);
        assert_eq!(
            pack_unit_nchw(&planes, 2, 1, PixelFormat::Bgr8, 1).unwrap(),
            bgr
        );
        assert_eq!(
            pack_unit_nchw(&planes, 2, 1, PixelFormat::Rgb8, 1).unwrap(),
            [30, 20, 10, 60, 50, 40]
        );
        let gray = [7u8, 9];
        let view = ImageView::new(&gray, 2, 1, 2, PixelFormat::Gray8).unwrap();
        let planes = rgb_unit_nchw_threads(&view, 1).unwrap();
        assert_eq!(
            pack_unit_nchw(&planes, 2, 1, PixelFormat::Gray8, 1).unwrap(),
            gray
        );
        assert!(matches!(
            pack_unit_nchw(&planes[..5], 2, 1, PixelFormat::Gray8, 1),
            Err(Error::Shape(_))
        ));
    }
}
