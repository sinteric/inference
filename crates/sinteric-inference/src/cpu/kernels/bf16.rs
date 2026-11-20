//! Round-to-nearest-even f32↔bf16 conversions (IEEE-like, matching PyTorch / VCVTNEPS2BF16).

use crate::cpu::kernels::parallel::for_rows;

#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let abs = bits & 0x7fff_ffff;
    if abs > 0x7f80_0000 {
        // quiet NaN: keep top half and force a mantissa bit
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    ((bits as u64 + rounding_bias as u64) >> 16) as u16
}

#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn f32_slice_to_bf16(src: &[f32], dst: &mut [u16]) {
    f32_slice_to_bf16_nthreads(src, dst, 1);
}

/// # Panics
/// If `src` and `dst` differ in length.
pub(crate) fn f32_slice_to_bf16_nthreads(src: &[f32], dst: &mut [u16], nthreads: usize) {
    assert_eq!(src.len(), dst.len(), "bf16 conversion lengths");
    if src.is_empty() {
        return;
    }
    for_rows(nthreads, 1024, dst, 1, |lo, hi, dst| {
        convert_range(&src[lo..hi], dst)
    });
}

fn convert_range(src: &[f32], dst: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
            // SAFETY: the CPU has both features (checked just above).
            unsafe {
                convert_range_avx512(src, dst);
            }
            return;
        }
    }
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = f32_to_bf16_bits(x);
    }
}

/// # Safety
/// The CPU must support AVX-512F and AVX512_BF16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn convert_range_avx512(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    // Every load and store below stays inside `[0, n)` of both slices.
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 32 <= n {
        let lo = _mm512_loadu_ps(src.as_ptr().add(i));
        let hi = _mm512_loadu_ps(src.as_ptr().add(i + 16));
        // src1 (hi) → upper 16 bf16s, src2 (lo) → lower 16.
        let packed = _mm512_cvtne2ps_pbh(hi, lo);
        _mm512_storeu_si512(
            dst.as_mut_ptr().add(i).cast(),
            std::mem::transmute::<__m512bh, __m512i>(packed),
        );
        i += 32;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = f32_to_bf16_bits(*src.get_unchecked(i));
        i += 1;
    }
}

/// sqrt(512) rounded the way bfloat16 rounds it (22.6274… → 22.625).
pub const SQRT_512_BF16: f32 = 22.625;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_512_matches_hf_note() {
        let bits = f32_to_bf16_bits(512f32.sqrt());
        assert_eq!(bf16_bits_to_f32(bits), SQRT_512_BF16);
    }

    #[test]
    fn roundtrip_ones() {
        let b = f32_to_bf16_bits(1.0);
        assert_eq!(bf16_bits_to_f32(b), 1.0);
    }

    #[test]
    fn rne_halfway() {
        let even = f32::from_bits(0x3f80_8000);
        let bits = f32_to_bf16_bits(even);
        assert_eq!(bits, 0x3f80);
    }

    #[test]
    fn vector_path_matches_scalar() {
        let src: Vec<f32> = (0..64).map(|i| (i as f32) * 0.07 - 1.3).collect();
        let mut a = vec![0u16; src.len()];
        let mut b = vec![0u16; src.len()];
        for (d, &x) in a.iter_mut().zip(&src) {
            *d = f32_to_bf16_bits(x);
        }
        f32_slice_to_bf16_nthreads(&src, &mut b, 2);
        assert_eq!(a, b);
    }
}
