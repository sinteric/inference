use crate::cpu::kernels::parallel::for_rows;

/// `dst[i] += src[i]` over the whole slice.
///
/// # Panics
/// If the slices differ in length.
pub(crate) fn add_inplace(dst: &mut [f32], src: &[f32], nthreads: usize) {
    assert_eq!(dst.len(), src.len(), "add lengths");
    for_rows(nthreads, 4096, dst, 1, |lo, hi, dst| {
        add_range(dst, &src[lo..hi])
    });
}

/// `dst[i] *= scale` over the whole slice.
pub fn scale_inplace(dst: &mut [f32], scale: f32, nthreads: usize) {
    for_rows(nthreads, 4096, dst, 1, |_, _, dst| scale_range(dst, scale));
}

fn add_range(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len(), "add lengths");
    let mut i = 0;
    let n = dst.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU has AVX-512F (checked just above), and every
            // load and store covers `[i, i + 16)` with `i + 16 <= n`, inside
            // both slices.
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

fn scale_range(dst: &mut [f32], scale: f32) {
    let mut i = 0;
    let n = dst.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU has AVX-512F (checked just above), and every
            // load and store covers `[i, i + 16)` with `i + 16 <= n`.
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
