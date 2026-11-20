//! OpenCV-compatible INTER_CUBIC perspective sampling and INTER_LINEAR resize.
//! Continuous cubic (OpenCV 5.0) and fixed-point linear rules; see NOTICE.
use crate::image::{ImageView, PixelFormat, Quad, Result};
use crate::{checked_size, try_vec};

#[derive(Debug)]
pub struct BgrImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}
impl BgrImage {
    pub fn view(&self) -> Result<ImageView<'_>> {
        ImageView::new(
            &self.data,
            self.width,
            self.height,
            checked_size("BGR row", &[self.width as usize, 3])?,
            PixelFormat::Bgr8,
        )
    }
}
fn bounded_pixels(width: u32, height: u32, max_pixels: usize) -> Result<()> {
    crate::check_shape!(
        width > 0 && height > 0 && max_pixels > 0,
        "empty crop or zero pixel limit"
    );
    crate::check_shape!(
        checked_size("crop pixels", &[width as usize, height as usize])? <= max_pixels,
        "crop exceeds max_pixels {max_pixels}"
    );
    Ok(())
}
fn cubic(x: f32) -> [f32; 4] {
    let a2 = x * x;
    let b = 1. - x;
    let b2 = b * b;
    let c0 = -0.75 * (x * b2);
    let c3 = -0.75 * (a2 * b);
    let c1 = a2.mul_add(1.25f32.mul_add(x, -2.25), 1.);
    [c0, c1, 1. - c0 - c1 - c3, c3]
}
fn inverse(m: [f64; 9]) -> Result<[f32; 9]> {
    let d = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    crate::check_shape!(d.is_finite() && d != 0., "singular crop homography");
    let d = 1. / d;
    Ok([
        (m[4] * m[8] - m[5] * m[7]) * d,
        (m[2] * m[7] - m[1] * m[8]) * d,
        (m[1] * m[5] - m[2] * m[4]) * d,
        (m[5] * m[6] - m[3] * m[8]) * d,
        (m[0] * m[8] - m[2] * m[6]) * d,
        (m[2] * m[3] - m[0] * m[5]) * d,
        (m[3] * m[7] - m[4] * m[6]) * d,
        (m[1] * m[6] - m[0] * m[7]) * d,
        (m[0] * m[4] - m[1] * m[3]) * d,
    ]
    .map(|v| v as f32))
}
// Solve the 8x8 homography system with partial-pivot LU, with m[8]=1.
fn homography(src: Quad, dst: Quad) -> Result<[f64; 9]> {
    let mut a = [[0.; 9]; 8];
    for i in 0..4 {
        let (x, y, u, v) = (
            src[i][0] as f64,
            src[i][1] as f64,
            dst[i][0] as f64,
            dst[i][1] as f64,
        );
        a[i] = [
            x,
            y,
            1.,
            0.,
            0.,
            0.,
            (-src[i][0] * dst[i][0]) as f64,
            (-src[i][1] * dst[i][0]) as f64,
            u,
        ];
        a[i + 4] = [
            0.,
            0.,
            0.,
            x,
            y,
            1.,
            (-src[i][0] * dst[i][1]) as f64,
            (-src[i][1] * dst[i][1]) as f64,
            v,
        ];
    }
    for k in 0..8 {
        let mut pivot = k;
        for i in k + 1..8 {
            if a[i][k].abs() > a[pivot][k].abs() {
                pivot = i;
            }
        }
        crate::check_shape!(
            a[pivot][k].abs() > f64::EPSILON,
            "degenerate perspective quad"
        );
        a.swap(k, pivot);
        for i in k + 1..8 {
            let ratio = a[i][k] * (-1. / a[k][k]);
            for j in k + 1..9 {
                a[i][j] += ratio * a[k][j];
            }
            a[i][k] = 0.;
        }
    }
    let mut m = [0.; 9];
    m[8] = 1.;
    for i in (0..8).rev() {
        let mut sum = a[i][8];
        for j in i + 1..8 {
            sum -= a[i][j] * m[j];
        }
        m[i] = sum / a[i][i];
    }
    crate::check_shape!(
        m.iter().all(|x| x.is_finite()),
        "nonfinite perspective transform"
    );
    Ok(m)
}
/// Warp a TL/TR/BR/BL quad at full resolution with cubic interpolation and
/// BORDER_REPLICATE. Output extents are floor(max opposing edge lengths).
pub fn warp_quad_bgr(src: &ImageView<'_>, quad: Quad, max_pixels: usize) -> Result<BgrImage> {
    crate::check_shape!(
        quad.iter().flatten().all(|v| v.is_finite()),
        "nonfinite crop quad"
    );
    let edge = |i: usize, j: usize| {
        ((quad[i][0] - quad[j][0]).powi(2) + (quad[i][1] - quad[j][1]).powi(2)).sqrt()
    };
    let wf = edge(0, 1).max(edge(2, 3));
    let hf = edge(0, 3).max(edge(1, 2));
    crate::check_shape!(
        wf.is_finite()
            && hf.is_finite()
            && wf >= 1.
            && hf >= 1.
            && wf < (u32::MAX as f32)
            && hf < (u32::MAX as f32),
        "invalid crop edges"
    );
    let (width, height) = (wf as u32, hf as u32);
    bounded_pixels(width, height, max_pixels)?;
    // Reject folded/self-intersecting/collinear public quads at the boundary.
    let mut sign = 0.;
    for i in 0..4 {
        let p = quad[i].map(f64::from);
        let q = quad[(i + 1) % 4].map(f64::from);
        let r = quad[(i + 2) % 4].map(f64::from);
        let cross = (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
        crate::check_shape!(
            cross != 0. && (sign == 0. || cross * sign > 0.),
            "crop quad must be convex and ordered"
        );
        sign = cross;
    }
    let target = [
        [0., 0.],
        [width as f32, 0.],
        [width as f32, height as f32],
        [0., height as f32],
    ];
    let m = inverse(homography(quad, target)?)?;
    crate::check_shape!(m.iter().all(|v| v.is_finite()), "invalid crop transform");
    let den = |x: f64, y: f64| m[6] as f64 * x + m[7] as f64 * y + m[8] as f64;
    let ds = [
        den(0., 0.),
        den(width as f64, 0.),
        den(0., height as f64),
        den(width as f64, height as f64),
    ];
    crate::check_shape!(
        ds.iter()
            .all(|d| d.is_finite() && d.abs() > f32::EPSILON as f64 && *d * ds[0] > 0.),
        "crop crosses homography horizon"
    );
    let (w, h) = (width as usize, height as usize);
    let mut data = try_vec(checked_size("warp BGR", &[w, h, 3])?)?;
    for y in 0..h {
        let mx = y as f32 * m[1] + m[2];
        let my = y as f32 * m[4] + m[5];
        let mz = y as f32 * m[7] + m[8];
        for x in 0..w {
            let den = mz as f64 + m[6] as f64 * x as f64;
            let invz = 1. / den;
            let sx = ((mx as f64 + m[0] as f64 * x as f64) * invz) as f32;
            let sy = ((my as f64 + m[3] as f64 * x as f64) * invz) as f32;
            let sx = sx.clamp(
                -(src.width().max(16) as f32),
                src.width().max(16) as f32 * 2.,
            );
            let sy = sy.clamp(
                -(src.height().max(16) as f32),
                src.height().max(16) as f32 * 2.,
            );
            let ix = sx.floor() as i64;
            let iy = sy.floor() as i64;
            let wx = cubic(sx - ix as f32);
            let wy = cubic(sy - iy as f32);
            for c in 0..3 {
                let mut acc = 0f32;
                for ky in 0..4 {
                    let yy = (iy + ky as i64 - 1).clamp(0, src.height() as i64 - 1) as usize;
                    let mut row = 0.;
                    for (kx, &weight) in wx.iter().enumerate() {
                        let xx = (ix + kx as i64 - 1).clamp(0, src.width() as i64 - 1) as usize;
                        if kx == 0 {
                            row = src.bgr(xx, yy, c) as f32 * weight;
                        } else {
                            row = (src.bgr(xx, yy, c) as f32).mul_add(weight, row);
                        }
                    }
                    acc = row.mul_add(wy[ky], acc);
                }
                data[(y * w + x) * 3 + c] = acc.round_ties_even().clamp(0., 255.) as u8;
            }
        }
    }
    Ok(BgrImage {
        data,
        width,
        height,
    })
}
/// Rotate 90 degrees counterclockwise, including the exact h/w >= 1.5 rule's
/// boundary when used by `recognizer_crop`.
pub fn rotate_ccw_bgr(src: &ImageView<'_>) -> Result<BgrImage> {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let mut data = try_vec(checked_size("rotated BGR", &[w, h, 3])?)?;
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                data[((w - 1 - x) * h + y) * 3 + c] = src.bgr(x, y, c);
            }
        }
    }
    Ok(BgrImage {
        data,
        width: src.height(),
        height: src.width(),
    })
}
/// OpenCV's 11-bit separable INTER_LINEAR rounding for uint8 pixels.
pub fn resize_bilinear_bgr(src: &ImageView<'_>, width: u32, height: u32) -> Result<Vec<u8>> {
    crate::check_shape!(width > 0 && height > 0, "empty bilinear destination");
    let (w, h) = (width as usize, height as usize);
    let mut out = try_vec(checked_size("bilinear BGR", &[w, h, 3])?)?;
    let coeff = |i: usize, s: u32, d: u32, clamp: bool| {
        let position = ((i as f64 + 0.5) * (s as f64 / d as f64) - 0.5) as f32;
        let mut base = position.floor() as i64;
        let mut f = position - base as f32;
        if clamp && (base < 0 || base >= s as i64 - 1) {
            base = base.clamp(0, s as i64 - 1);
            f = 0.;
        }
        (
            base,
            ((1. - f) * 2048.).round_ties_even() as i32,
            (f * 2048.).round_ties_even() as i32,
        )
    };
    for y in 0..h {
        let (sy, b0, b1) = coeff(y, src.height(), height, false);
        let y0 = sy.clamp(0, src.height() as i64 - 1) as usize;
        let y1 = (sy + 1).clamp(0, src.height() as i64 - 1) as usize;
        for x in 0..w {
            let (sx, a0, a1) = coeff(x, src.width(), width, true);
            let x0 = sx as usize;
            let x1 = (sx + 1).min(src.width() as i64 - 1) as usize;
            for c in 0..3 {
                let h0 = src.bgr(x0, y0, c) as i32 * a0 + src.bgr(x1, y0, c) as i32 * a1;
                let h1 = src.bgr(x0, y1, c) as i32 * a0 + src.bgr(x1, y1, c) as i32 * a1;
                out[(y * w + x) * 3 + c] =
                    ((((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2) as u8;
            }
        }
    }
    Ok(out)
}
#[derive(Debug)]
pub struct RecCrop {
    pub width: usize,
    pub values: Vec<f32>,
}
/// Full-resolution perspective crop, optional CCW turn, resize to height 48,
/// ceil width clamped to caller limits, then BGR NCHW (v/255-0.5)/0.5.
pub fn recognizer_crop(
    src: &ImageView<'_>,
    quad: Quad,
    min_width: usize,
    max_width: usize,
    max_pixels: usize,
) -> Result<RecCrop> {
    crate::check_shape!(
        min_width > 0 && min_width <= max_width && max_width <= u32::MAX as usize,
        "invalid crop width bounds"
    );
    bounded_pixels(src.width(), src.height(), max_pixels)?;
    let mut crop = warp_quad_bgr(src, quad, max_pixels)?;
    if crop.height as f64 >= 1.5 * crop.width as f64 {
        crop = rotate_ccw_bgr(&crop.view()?)?;
    }
    let numerator = checked_size("crop resize width", &[48, crop.width as usize])?;
    let width = numerator
        .div_ceil(crop.height as usize)
        .clamp(min_width, max_width);
    bounded_pixels(width as u32, 48, max_pixels)?;
    let bytes = resize_bilinear_bgr(&crop.view()?, width as u32, 48)?;
    let resized = ImageView::new(&bytes, width as u32, 48, width * 3, PixelFormat::Bgr8)?;
    Ok(RecCrop {
        width,
        values: crate::image::normalize_nchw(&resized, [0.5; 3], [0.5; 3])?,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crop_boundaries_and_rotation() {
        let bytes = vec![123; 20 * 30 * 3];
        let image = ImageView::new(&bytes, 20, 30, 60, PixelFormat::Bgr8).unwrap();
        let q = [[0., 0.], [20., 0.], [20., 30.], [0., 30.]];
        let crop = recognizer_crop(&image, q, 16, 1600, 10000).unwrap();
        assert_eq!(crop.width, 72);
        assert!(crop.values.iter().all(|&v| v == (123. / 255. - 0.5) / 0.5));
        for q in [
            [[0., 0.]; 4],
            [[f32::NAN, 0.]; 4],
            [[0., 0.], [10., 10.], [0., 10.], [10., 0.]],
        ] {
            assert!(warp_quad_bgr(&image, q, 600).is_err());
        }
        assert!(warp_quad_bgr(
            &image,
            [[0., 0.], [1e30, 0.], [1e30, 1e30], [0., 1e30]],
            600
        )
        .is_err());
        assert!(recognizer_crop(&image, q, 0, 1600, 600).is_err());
        assert!(recognizer_crop(&image, q, 16, 1600, 599).is_err());
        assert!(resize_bilinear_bgr(&image, u32::MAX, u32::MAX).is_err());
    }
}
