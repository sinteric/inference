use crate::cpu::kernels::parallel::{for_ranges, rows_mut, rows_ref};
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

/// `dst[i] *= scale` over the whole slice.
pub fn scale_inplace(dst: &mut [f32], scale: f32, nthreads: usize) {
    let n = dst.len();
    let d_p = dst.as_mut_ptr() as usize;
    for_ranges(nthreads, n, 4096, |lo, hi| {
        let dst = unsafe { rows_mut(d_p as *mut f32, lo, hi, 1) };
        scale_range(dst, scale);
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
