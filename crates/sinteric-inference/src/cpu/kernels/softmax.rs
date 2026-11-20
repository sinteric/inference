use crate::cpu::kernels::parallel::{for_ranges, rows_mut};

pub fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                softmax_inplace_avx512(row);
            }
            return;
        }
    }
    softmax_inplace_scalar(row);
}

fn softmax_inplace_scalar(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        row.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 {
        row.fill(0.0);
        return;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_inplace_avx512(row: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = row.len();
    let ptr = row.as_mut_ptr();
    let mut i = 0;
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    while i + 16 <= n {
        vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(ptr.add(i)));
        i += 16;
    }
    let mut max = _mm512_reduce_max_ps(vmax);
    while i < n {
        max = max.max(*ptr.add(i));
        i += 1;
    }
    if !max.is_finite() {
        row.fill(0.0);
        return;
    }
    let maxv = _mm512_set1_ps(max);
    i = 0;
    let mut vacc = _mm512_setzero_ps();
    while i + 16 <= n {
        let e = exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(ptr.add(i)), maxv));
        _mm512_storeu_ps(ptr.add(i), e);
        vacc = _mm512_add_ps(vacc, e);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(vacc);
    while i < n {
        let e = (*ptr.add(i) - max).exp();
        *ptr.add(i) = e;
        sum += e;
        i += 1;
    }
    if sum == 0.0 {
        row.fill(0.0);
        return;
    }
    let invv = _mm512_set1_ps(1.0 / sum);
    i = 0;
    while i + 16 <= n {
        _mm512_storeu_ps(ptr.add(i), _mm512_mul_ps(_mm512_loadu_ps(ptr.add(i)), invv));
        i += 16;
    }
    let inv = 1.0 / sum;
    while i < n {
        *ptr.add(i) *= inv;
        i += 1;
    }
}

/// `exp(x)` on 16 lanes. Relative error is ~1e-6 on the softmax domain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn exp_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    // Clamp so n = round(x / ln2) fits a reasonable exponent.
    let x = _mm512_min_ps(
        _mm512_max_ps(x, _mm512_set1_ps(-87.336)),
        _mm512_set1_ps(88.376),
    );
    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let n = _mm512_roundscale_ps(
        _mm512_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
    let f = _mm512_fnmadd_ps(n, ln2, x);
    // exp(f) ≈ 1 + f*(1 + f*(1/2 + f*(1/6 + f*(1/24 + f*(1/120 + f/720)))))
    let c6 = _mm512_set1_ps(1.0 / 720.0);
    let c5 = _mm512_set1_ps(1.0 / 120.0);
    let c4 = _mm512_set1_ps(1.0 / 24.0);
    let c3 = _mm512_set1_ps(1.0 / 6.0);
    let c2 = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let mut p = _mm512_fmadd_ps(c6, f, c5);
    p = _mm512_fmadd_ps(p, f, c4);
    p = _mm512_fmadd_ps(p, f, c3);
    p = _mm512_fmadd_ps(p, f, c2);
    p = _mm512_fmadd_ps(p, f, one);
    p = _mm512_fmadd_ps(p, f, one);
    _mm512_mul_ps(p, _mm512_scalef_ps(one, n))
}

pub(crate) fn softmax_rows_nthreads(logits: &mut [f32], cols: usize, nthreads: usize) {
    debug_assert!(cols > 0 && logits.len().is_multiple_of(cols));
    let rows = logits.len() / cols;
    let p = logits.as_mut_ptr() as usize;
    for_ranges(nthreads, rows, 4, |lo, hi| {
        let logits = unsafe { rows_mut(p as *mut f32, lo, hi, cols) };
        for i in 0..(hi - lo) {
            softmax_inplace(&mut logits[i * cols..(i + 1) * cols]);
        }
    });
}

/// Softmax over `[lo, hi)` of one row; everything else is written to 0.
/// `blocked` positions inside the window are treated as −∞ (padding).
pub fn softmax_window(row: &mut [f32], lo: usize, hi: usize, blocked: &[usize]) {
    let n = row.len();
    let lo = lo.min(n);
    let hi = hi.min(n).max(lo);
    row[..lo].fill(0.0);
    if hi < n {
        row[hi..].fill(0.0);
    }
    if lo == hi {
        return;
    }
    for &p in blocked {
        if p >= lo && p < hi {
            row[p] = f32::NEG_INFINITY;
        }
    }
    softmax_inplace(&mut row[lo..hi]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let s: f32 = x.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn window_zeros_outside_and_skips_blocked() {
        let mut x = [9.0f32, 1.0, 2.0, 3.0, 9.0];
        softmax_window(&mut x, 1, 4, &[2]);
        assert_eq!(x[0], 0.0);
        assert_eq!(x[4], 0.0);
        assert_eq!(x[2], 0.0);
        let s = x[1] + x[3];
        assert!((s - 1.0).abs() < 1e-5);
        assert!(x[3] > x[1]);
    }

    #[test]
    fn all_masked_is_zero() {
        let mut x = [1.0f32, 2.0];
        softmax_window(&mut x, 0, 2, &[0, 1]);
        assert_eq!(x, [0.0, 0.0]);
    }

    #[test]
    fn matches_scalar_on_random_row() {
        let mut a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.07 - 1.3).collect();
        let mut b = a.clone();
        softmax_inplace_scalar(&mut a);
        softmax_inplace(&mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 2e-5, "{x} vs {y}");
        }
    }
}
