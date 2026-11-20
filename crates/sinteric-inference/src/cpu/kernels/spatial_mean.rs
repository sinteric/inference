//! OCR spatial means with fixed FP32 reduction order. The reference ORT build
//! (the `ort` crate's bundled ONNX Runtime) compiles Eigen for AVX, so its
//! ReduceMean is Eigen's eight-lane redux with two accumulators; its NCHWc
//! GlobalAveragePool is sequential. Keeping these orders matters on full pages,
//! where the FP32 rounding bias propagates through SE gates. Partition only
//! independent N*C rows.
use crate::{Error, Result};
#[derive(Clone, Copy, Debug)]
pub enum SpatialMean {
    Sequential,
    /// Eigen `sum()` over an AVX packet map whose tensor starts 32-byte aligned
    /// (ORT arena buffers do), so row `c` begins at element offset `c * plane`.
    EigenAvx,
}
pub fn spatial_mean(
    input: &[f32],
    output: &mut [f32],
    plane: usize,
    order: SpatialMean,
    threads: usize,
) -> Result<()> {
    crate::check_shape!(
        plane > 0
            && !output.is_empty()
            && threads > 0
            && input.len()
                == output
                    .len()
                    .checked_mul(plane)
                    .ok_or(Error::size_overflow("spatial mean"))?,
        "invalid spatial mean extent"
    );
    let ptr = output.as_mut_ptr() as usize;
    crate::cpu::kernels::for_ranges(threads, output.len(), 1, |lo, hi| {
        for c in lo..hi {
            let row = &input[c * plane..(c + 1) * plane];
            let mut value = match order {
                SpatialMean::Sequential => {
                    let mut sum = 0.;
                    for &v in row {
                        sum += v;
                    }
                    sum
                }
                SpatialMean::EigenAvx => eigen_avx_sum(row, (8 - c * plane % 8) % 8),
            };
            value /= plane as f32;
            // Boundary validation proves disjoint, in-bounds output rows.
            unsafe {
                *(ptr as *mut f32).add(c) = value;
            }
        }
    });
    Ok(())
}
/// Eigen 3.4 `redux_impl<LinearVectorizedTraversal>` for `Packet8f`: `head`
/// scalars precede the first aligned packet; packets alternate between two
/// accumulators, one odd packet joins the first, then `predux` and the head and
/// tail scalars in order.
fn eigen_avx_sum(row: &[f32], head: usize) -> f32 {
    const P: usize = 8;
    let start = head.min(row.len());
    let end = start + (row.len() - start) / P * P;
    if end == start {
        return row[1..].iter().fold(row[0], |s, &v| s + v);
    }
    let load = |i: usize| -> [f32; P] { row[i..i + P].try_into().unwrap() };
    let add = |a: &mut [f32; P], b: [f32; P]| a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
    let mut r0 = load(start);
    if end - start > P {
        let mut r1 = load(start + P);
        let end2 = start + (end - start) / (2 * P) * (2 * P);
        for i in (start + 2 * P..end2).step_by(2 * P) {
            add(&mut r0, load(i));
            add(&mut r1, load(i + P));
        }
        add(&mut r0, r1);
        if end > end2 {
            add(&mut r0, load(end2));
        }
    }
    // predux: add high/low halves, then pairs 0/2, 1/3.
    let mut sum = ((r0[0] + r0[4]) + (r0[2] + r0[6])) + ((r0[1] + r0[5]) + (r0[3] + r0[7]));
    for &v in row[..start].iter().chain(&row[end..]) {
        sum += v;
    }
    sum
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduction_order_and_boundaries() {
        let x = [
            1e8, 1., 1., 1., 1., 1., 1., 1., -1e8, 1., 1., 1., 1., 1., 1., 1.,
        ];
        let mut y = [0.];
        spatial_mean(&x, &mut y, 16, SpatialMean::Sequential, 1).unwrap();
        assert_eq!(y[0], 7. / 16.);
        spatial_mean(&x, &mut y, 16, SpatialMean::EigenAvx, 1).unwrap();
        assert_eq!(y[0], 14. / 16.);
        assert!(spatial_mean(&[], &mut y, 0, SpatialMean::EigenAvx, 1).is_err());
        assert!(spatial_mean(&x, &mut y, usize::MAX, SpatialMean::Sequential, 1).is_err());
        assert!(spatial_mean(&x, &mut y, 16, SpatialMean::Sequential, 0).is_err());
    }

    #[test]
    fn eigen_head_follows_row_alignment() {
        // Row 1 starts at element 12, so its first four values are scalar head
        // added after the packet: (-1e8 + 7 ones absorbed to -1e8) + 1e8 + 3.
        let mut x = [1f32; 24];
        x[12] = 1e8;
        x[16] = -1e8;
        let mut y = [0.; 2];
        spatial_mean(&x, &mut y, 12, SpatialMean::EigenAvx, 2).unwrap();
        assert_eq!(y, [1., 3. / 12.]);
        // Short rows (no full packet) are a plain left fold.
        let mut y = [0.; 8];
        let x: Vec<f32> = (0..56).map(|i| (i % 7) as f32 * 0.1).collect();
        spatial_mean(&x, &mut y, 7, SpatialMean::EigenAvx, 3).unwrap();
        let mut z = [0.; 8];
        spatial_mean(&x, &mut z, 7, SpatialMean::Sequential, 1).unwrap();
        assert_eq!(y, z);
    }
}
