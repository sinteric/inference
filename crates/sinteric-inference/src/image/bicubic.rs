//! Antialiased bicubic resize with torch's uint8 CPU kernel
//! (`interpolate(mode="bicubic", antialias=True)` on uint8, which
//! torchvision `resize(antialias=True)` runs): separable horizontal then
//! vertical passes with int16 fixed-point taps, so output bytes match
//! torch exactly. A pass whose size is unchanged is skipped.

use crate::image::{ImageView, PixelFormat};
use crate::{check_shape, Result};

/// Keys cubic with `a = -0.5` (torch `aa_filter` for bicubic).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// Fixed-point taps for one axis: output `i` reads `len[i]` inputs from
/// `start[i]` with weights `w[i * stride..]`, scaled by `2^precision`.
struct Taps {
    start: Vec<usize>,
    len: Vec<usize>,
    w: Vec<i16>,
    stride: usize,
    precision: u32,
}

impl Taps {
    /// torch `_compute_index_ranges_int16_weights` (align_corners=false).
    fn new(input: usize, output: usize) -> Self {
        let scale = input as f64 / output as f64;
        let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let stride = support.ceil() as usize * 2 + 1;
        let mut start = Vec::with_capacity(output);
        let mut len = Vec::with_capacity(output);
        let mut wf = vec![0f64; output * stride];
        let mut max_w = 0f64;
        for i in 0..output {
            let center = scale * (i as f64 + 0.5);
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let n = hi.saturating_sub(lo).min(stride);
            let row = &mut wf[i * stride..i * stride + n];
            let mut total = 0f64;
            for (j, w) in row.iter_mut().enumerate() {
                *w = cubic((j as f64 + lo as f64 - center + 0.5) * invscale);
                total += *w;
            }
            if total != 0.0 {
                for w in row.iter_mut() {
                    *w /= total;
                }
            }
            for &w in row.iter() {
                max_w = max_w.max(w);
            }
            start.push(lo);
            len.push(n);
        }
        let mut precision = 0u32;
        while precision < 22 {
            let next = (0.5 + max_w * (1u64 << (precision + 1)) as f64) as i64;
            if next >= 1 << 15 {
                break;
            }
            precision += 1;
        }
        let one = (1u64 << precision) as f64;
        let w = wf
            .iter()
            .map(|&v| {
                let v = v * one;
                (if v < 0.0 {
                    (-0.5 + v) as i32
                } else {
                    (0.5 + v) as i32
                }) as i16
            })
            .collect();
        Self {
            start,
            len,
            w,
            stride,
            precision,
        }
    }

    #[inline]
    fn apply(&self, i: usize, read: impl Fn(usize) -> i32) -> u8 {
        let w = &self.w[i * self.stride..i * self.stride + self.len[i]];
        let mut acc: i32 = 1 << (self.precision.max(1) - 1);
        for (j, &wj) in w.iter().enumerate() {
            acc += read(self.start[i] + j) * wj as i32;
        }
        (acc >> self.precision).clamp(0, 255) as u8
    }
}

/// Packed RGB copy of any [`PixelFormat`] (gray replicated, BGR swapped),
/// as PIL `convert("RGB")` produces.
pub fn to_rgb(src: &ImageView<'_>) -> Result<Vec<u8>> {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let mut out = crate::try_vec::<u8>(crate::checked_size("RGB image", &[w, h, 3])?)?;
    let c = src.format().channels();
    for y in 0..h {
        let row = &src.data()[y * src.stride()..y * src.stride() + w * c];
        let dst = &mut out[y * w * 3..(y + 1) * w * 3];
        match src.format() {
            PixelFormat::Rgb8 => dst.copy_from_slice(row),
            PixelFormat::Bgr8 => {
                for (d, s) in dst
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(row.as_chunks::<3>().0)
                {
                    *d = [s[2], s[1], s[0]];
                }
            }
            PixelFormat::Gray8 => {
                for (d, &g) in dst.as_chunks_mut::<3>().0.iter_mut().zip(row) {
                    *d = [g, g, g];
                }
            }
        }
    }
    Ok(out)
}

/// Resize packed RGB `width x height` to `out_w x out_h`, byte-exact with
/// torch's antialiased bicubic uint8 kernel.
pub fn resize_bicubic_aa_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
    out_w: usize,
    out_h: usize,
    nthreads: usize,
) -> Result<Vec<u8>> {
    check_shape!(
        width > 0 && height > 0 && out_w > 0 && out_h > 0,
        "empty resize {width}x{height} -> {out_w}x{out_h}"
    );
    check_shape!(
        rgb.len() == crate::checked_size("RGB image", &[width, height, 3])?,
        "RGB buffer length {} != {width}x{height}x3",
        rgb.len()
    );
    let horizontal = if out_w != width {
        let taps = Taps::new(width, out_w);
        let mut mid =
            crate::try_vec::<u8>(crate::checked_size("resize rows", &[height, out_w, 3])?)?;
        let mid_p = mid.as_mut_ptr() as usize;
        crate::cpu::threads::par_for(nthreads, height, 8, |lo, hi| {
            // SAFETY: rows [lo, hi) are disjoint across workers; `mid` holds height rows.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (mid_p as *mut u8).add(lo * out_w * 3),
                    (hi - lo) * out_w * 3,
                )
            };
            for (y, row) in (lo..hi).zip(dst.chunks_exact_mut(out_w * 3)) {
                let src = &rgb[y * width * 3..(y + 1) * width * 3];
                for x in 0..out_w {
                    for c in 0..3 {
                        row[x * 3 + c] = taps.apply(x, |j| src[j * 3 + c] as i32);
                    }
                }
            }
        });
        mid
    } else {
        rgb.to_vec()
    };
    if out_h == height {
        return Ok(horizontal);
    }
    let taps = Taps::new(height, out_h);
    let row_len = out_w * 3;
    let mut out = crate::try_vec::<u8>(crate::checked_size("resize output", &[out_h, row_len])?)?;
    let out_p = out.as_mut_ptr() as usize;
    crate::cpu::threads::par_for(nthreads, out_h, 8, |lo, hi| {
        // SAFETY: rows [lo, hi) are disjoint across workers; `out` holds out_h rows.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                (out_p as *mut u8).add(lo * row_len),
                (hi - lo) * row_len,
            )
        };
        for (y, row) in (lo..hi).zip(dst.chunks_exact_mut(row_len)) {
            for (i, v) in row.iter_mut().enumerate() {
                *v = taps.apply(y, |j| horizontal[j * row_len + i] as i32);
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_constant_images() {
        let rgb: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7) as u8).collect();
        assert_eq!(resize_bicubic_aa_rgb(&rgb, 4, 3, 4, 3, 1).unwrap(), rgb);
        // Normalized taps keep a flat image flat in both directions.
        let flat = vec![200u8; 37 * 23 * 3];
        for (w, h) in [(11, 7), (80, 50), (37, 5), (3, 23)] {
            let out = resize_bicubic_aa_rgb(&flat, 37, 23, w, h, 2).unwrap();
            assert_eq!(out.len(), w * h * 3);
            assert!(out.iter().all(|&v| v == 200), "{w}x{h}");
        }
    }

    #[test]
    fn taps_follow_torch_precision_rule() {
        // Downscale by 2: support 4, 9 taps; the largest tap sets precision.
        let t = Taps::new(8, 4);
        assert_eq!(t.stride, 9);
        assert_eq!(t.start[0], 0);
        let max = *t.w.iter().max().unwrap() as i64;
        assert!(
            max < 1 << 15 && max * 2 >= 1 << 15,
            "{max} p={}",
            t.precision
        );
    }

    #[test]
    fn formats_convert_like_pil() {
        let gray = [10u8, 20];
        let v = ImageView::new(&gray, 2, 1, 2, PixelFormat::Gray8).unwrap();
        assert_eq!(to_rgb(&v).unwrap(), [10, 10, 10, 20, 20, 20]);
        let bgr = [1u8, 2, 3];
        let v = ImageView::new(&bgr, 1, 1, 3, PixelFormat::Bgr8).unwrap();
        assert_eq!(to_rgb(&v).unwrap(), [3, 2, 1]);
    }
}
