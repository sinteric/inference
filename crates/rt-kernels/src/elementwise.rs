use crate::parallel::{for_ranges, rows_mut, rows_ref};
/// `dst[i] += src[i]` over the whole slice.
pub(crate) fn add_inplace(dst: &mut [f32], src: &[f32], nthreads: usize) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let d_p = dst.as_mut_ptr() as usize;
    let s_p = src.as_ptr() as usize;
    for_ranges(nthreads, n, 4096, |lo, hi| {
        let dst = unsafe { rows_mut(d_p as *mut f32, lo, hi, 1) };
        let src = unsafe { rows_ref(s_p as *const f32, lo, hi, 1) };
        add_range(dst, src);
    });
}

/// Add a bias vector to every row of `dst` (`dst` is row-major `[rows, bias.len()]`).
pub(crate) fn add_row_bias(dst: &mut [f32], bias: &[f32], nthreads: usize) {
    let n = bias.len();
    debug_assert!(n > 0 && dst.len().is_multiple_of(n));
    let rows = dst.len() / n;
    let d_p = dst.as_mut_ptr() as usize;
    for_ranges(nthreads, rows, 8, |lo, hi| {
        let dst = unsafe { rows_mut(d_p as *mut f32, lo, hi, n) };
        for r in 0..(hi - lo) {
            add_range(&mut dst[r * n..(r + 1) * n], bias);
        }
    });
}

/// `dst[i] *= scale` over the whole slice.
pub fn scale_inplace(dst: &mut [f32], scale: f32, nthreads: usize) {
    let n = dst.len();
    let d_p = dst.as_mut_ptr() as usize;
    for_ranges(nthreads, n, 4096, |lo, hi| {
        let dst = unsafe { rows_mut(d_p as *mut f32, lo, hi, 1) };
        scale_range(dst, scale);
    });
}

/// `dst[i] = (dst[i] + src[i]) * scale` in one pass (PLE residual + layer scalar).
pub(crate) fn add_scale_inplace(dst: &mut [f32], src: &[f32], scale: f32, nthreads: usize) {
    debug_assert_eq!(dst.len(), src.len());
    let n = dst.len();
    let d_p = dst.as_mut_ptr() as usize;
    let s_p = src.as_ptr() as usize;
    for_ranges(nthreads, n, 4096, |lo, hi| {
        let dst = unsafe { rows_mut(d_p as *mut f32, lo, hi, 1) };
        let src = unsafe { rows_ref(s_p as *const f32, lo, hi, 1) };
        add_scale_range(dst, src, scale);
    });
}

/// `out[i] = a[i] * b[i]`.
pub(crate) fn mul_into(a: &[f32], b: &[f32], out: &mut [f32], nthreads: usize) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let a_p = a.as_ptr() as usize;
    let b_p = b.as_ptr() as usize;
    let o_p = out.as_mut_ptr() as usize;
    for_ranges(nthreads, n, 4096, |lo, hi| {
        let a = unsafe { rows_ref(a_p as *const f32, lo, hi, 1) };
        let b = unsafe { rows_ref(b_p as *const f32, lo, hi, 1) };
        let out = unsafe { rows_mut(o_p as *mut f32, lo, hi, 1) };
        mul_range(a, b, out);
    });
}

fn add_range(dst: &mut [f32], src: &[f32]) {
    let mut i = 0;
    let n = dst.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                use std::arch::x86_64::*;
                while i + 16 <= n {
                    let a = _mm512_loadu_ps(dst.as_ptr().add(i));
                    let b = _mm512_loadu_ps(src.as_ptr().add(i));
                    _mm512_storeu_ps(dst.as_mut_ptr().add(i), _mm512_add_ps(a, b));
                    i += 16;
                }
            }
        }
    }
    while i < n {
        dst[i] += src[i];
        i += 1;
    }
}

fn add_scale_range(dst: &mut [f32], src: &[f32], scale: f32) {
    let mut i = 0;
    let n = dst.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                use std::arch::x86_64::*;
                let s = _mm512_set1_ps(scale);
                while i + 16 <= n {
                    let a = _mm512_loadu_ps(dst.as_ptr().add(i));
                    let b = _mm512_loadu_ps(src.as_ptr().add(i));
                    _mm512_storeu_ps(
                        dst.as_mut_ptr().add(i),
                        _mm512_mul_ps(_mm512_add_ps(a, b), s),
                    );
                    i += 16;
                }
            }
        }
    }
    while i < n {
        dst[i] = (dst[i] + src[i]) * scale;
        i += 1;
    }
}

fn scale_range(dst: &mut [f32], scale: f32) {
    let mut i = 0;
    let n = dst.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                use std::arch::x86_64::*;
                let s = _mm512_set1_ps(scale);
                while i + 16 <= n {
                    let a = _mm512_loadu_ps(dst.as_ptr().add(i));
                    _mm512_storeu_ps(dst.as_mut_ptr().add(i), _mm512_mul_ps(a, s));
                    i += 16;
                }
            }
        }
    }
    while i < n {
        dst[i] *= scale;
        i += 1;
    }
}

fn mul_range(a: &[f32], b: &[f32], out: &mut [f32]) {
    let mut i = 0;
    let n = a.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                use std::arch::x86_64::*;
                while i + 16 <= n {
                    let x = _mm512_loadu_ps(a.as_ptr().add(i));
                    let y = _mm512_loadu_ps(b.as_ptr().add(i));
                    _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_mul_ps(x, y));
                    i += 16;
                }
            }
        }
    }
    while i < n {
        out[i] = a[i] * b[i];
        i += 1;
    }
}
