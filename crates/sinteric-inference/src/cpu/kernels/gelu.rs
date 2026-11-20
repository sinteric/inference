//! GELU (erf), matching `torch.nn.functional.gelu` / ModernBERT `hidden_activation: gelu`.

#[inline]
pub fn gelu_erf_scalar(x: f32) -> f32 {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    0.5 * x * (1.0 + libm_erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[inline]
#[allow(clippy::excessive_precision)]
fn libm_erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26, sufficient for FP32 GEMM glue.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    #[allow(clippy::excessive_precision)]
    let y = {
        let t = 1.0 / (1.0 + 0.3275911 * x);
        1.0 - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp()
    };
    sign * y
}

/// `gelu_pytorch_tanh` / oneDNN `dnnl_eltwise_gelu_tanh`.
#[inline]
pub fn gelu_tanh_scalar(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_583_473_205_6; // sqrt(2/π)
    const COEFF: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

/// `y[i] = gelu_pytorch_tanh(x[i]) * f[i]`.
pub(crate) fn gelu_tanh_mul(x: &[f32], f: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), f.len());
    debug_assert_eq!(x.len(), y.len());
    gelu_tanh_mul_range(x, f, y);
}

fn gelu_tanh_mul_range(x: &[f32], f: &[f32], y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                gelu_tanh_mul_avx512(x, f, y);
            }
            return;
        }
    }
    for i in 0..x.len() {
        y[i] = gelu_tanh_scalar(x[i]) * f[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gelu_tanh_mul_avx512(x: &[f32], f: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    const SQRT_2_OVER_PI: f32 = 0.797_884_583_473_205_6;
    const COEFF: f32 = 0.044_715;
    let n = x.len();
    let k = _mm512_set1_ps(SQRT_2_OVER_PI);
    let c = _mm512_set1_ps(COEFF);
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        let fac = _mm512_loadu_ps(f.as_ptr().add(i));
        let v3 = _mm512_mul_ps(v, _mm512_mul_ps(v, v));
        let inner = _mm512_mul_ps(k, _mm512_fmadd_ps(c, v3, v));
        let t = tanh_avx512(inner);
        let gelu = _mm512_mul_ps(_mm512_mul_ps(half, v), _mm512_add_ps(one, t));
        _mm512_storeu_ps(y.as_mut_ptr().add(i), _mm512_mul_ps(gelu, fac));
        i += 16;
    }
    while i < n {
        *y.get_unchecked_mut(i) = gelu_tanh_scalar(*x.get_unchecked(i)) * *f.get_unchecked(i);
        i += 1;
    }
}

/// Rational tanh that matches libm closely enough for PLE mixing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn tanh_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    // tanh(x) = 2/(1+e^{-2x}) - 1, with a clamp to avoid overflow.
    let two = _mm512_set1_ps(2.0);
    let one = _mm512_set1_ps(1.0);
    let xmax = _mm512_set1_ps(8.0);
    let x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-8.0)), xmax);
    let e = crate_exp_neg_two(x);
    let denom = _mm512_add_ps(one, e);
    _mm512_sub_ps(_mm512_div_ps(two, denom), one)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn crate_exp_neg_two(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let n2 = _mm512_set1_ps(-2.0);
    exp_ps(_mm512_mul_ps(n2, x))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn exp_ps(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_zero() {
        assert!(gelu_erf_scalar(0.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_one_positive() {
        // gelu(1) ≈ 0.841344746
        let g = gelu_erf_scalar(1.0);
        assert!((g - 0.8413).abs() < 1e-3, "{g}");
    }

    #[test]
    fn gelu_tanh_zero() {
        assert!(gelu_tanh_scalar(0.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_tanh_one_near_erf() {
        let t = gelu_tanh_scalar(1.0);
        let e = gelu_erf_scalar(1.0);
        assert!((t - e).abs() < 0.02, "tanh {t} erf {e}");
        assert!(t > 0.8 && t < 0.86, "{t}");
    }

    #[test]
    fn gelu_tanh_mul_tracks_separate() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.07 - 1.5).collect();
        let f: Vec<f32> = (0..64).map(|i| 0.25 + i as f32 * 0.01).collect();
        let mut a = vec![0f32; x.len()];
        let mut b = vec![0f32; x.len()];
        for i in 0..x.len() {
            a[i] = gelu_tanh_scalar(x[i]) * f[i];
        }
        gelu_tanh_mul(&x, &f, &mut b);
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 2e-4, "{u} vs {v}");
        }
    }
}
