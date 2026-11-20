//! ONNX Runtime's x86-64 `Sigmoid`: MLAS `MlasComputeLogisticF32KernelFma3`
//! (onnxruntime 1.28.0, `mlas/lib/x86_64/LogisticKernelFma3.S`), reproduced
//! operation for operation. It is Eigen's rational approximation, not
//! 1/(1+e^-x), so matching ORT outputs bit for bit near a threshold needs
//! this exact sequence: clamp to [-18, 18], two FMA Horner chains in x^2,
//! one divide, +0.5, clamp below at 0 (no upper clamp).

#![expect(
    clippy::excessive_precision,
    reason = "the constants are MLAS's digits, so they check against its source; each rounds to the f32 MLAS loads"
)]

use crate::cpu::kernels::parallel::for_rows;

const LOWER: f32 = -18.0;
const UPPER: f32 = 18.0;
const ALPHA_9: f32 = 4.37031012579801e-11;
const ALPHA_7: f32 = 1.15627324459942e-07;
const ALPHA_5: f32 = 6.08574864600143e-05;
const ALPHA_3: f32 = 8.51377133304701e-03;
const ALPHA_1: f32 = 2.48287947061529e-01;
const BETA_10: f32 = 6.10247389755681e-13;
const BETA_8: f32 = 5.76102136993427e-09;
const BETA_6: f32 = 6.29106785017040e-06;
const BETA_4: f32 = 1.70198817374094e-03;
const BETA_2: f32 = 1.16817656904453e-01;
const BETA_0: f32 = 9.93151921023180e-01;

/// One element. `mul_add` is a single-rounding fused multiply-add on every
/// target (hardware FMA or a correctly rounded libm `fmaf`), so all paths
/// produce identical bits. The comparisons mirror `vmaxps`/`vminps` operand
/// order (the second operand is returned on NaN).
#[inline(always)]
fn logistic(x: f32) -> f32 {
    let x = if LOWER > x { LOWER } else { x };
    let x = if UPPER < x { UPPER } else { x };
    let x2 = x * x;
    let mut p = x2.mul_add(ALPHA_9, ALPHA_7);
    p = x2.mul_add(p, ALPHA_5);
    p = x2.mul_add(p, ALPHA_3);
    p = x2.mul_add(p, ALPHA_1);
    let mut q = x2.mul_add(BETA_10, BETA_8);
    q = x2.mul_add(q, BETA_6);
    q = x2.mul_add(q, BETA_4);
    q = x2.mul_add(q, BETA_2);
    q = x2.mul_add(q, BETA_0);
    let r = x * p / q + 0.5;
    if 0.0 > r {
        0.0
    } else {
        r
    }
}

fn range_portable(x: &mut [f32]) {
    for v in x {
        *v = logistic(*v);
    }
}

/// # Safety
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn range_fma(x: &mut [f32]) {
    // Same scalar code; the target feature lets LLVM vectorize it with
    // hardware FMA. Every operation is per element and IEEE, so the bits
    // equal `range_portable`.
    for v in x {
        *v = logistic(*v);
    }
}

fn range(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: the CPU has both features (checked just above).
            return unsafe { range_fma(x) };
        }
    }
    range_portable(x)
}

/// In-place ORT/MLAS-exact logistic over `x`, split across `nthreads`.
pub fn logistic_ort_inplace(x: &mut [f32], nthreads: usize) {
    for_rows(nthreads, 1 << 16, x, 1, |_, _, chunk| range(chunk));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_shape_and_all_paths_agree() {
        assert_eq!(logistic(0.0), 0.5);
        assert_eq!(logistic(-100.0), logistic(-18.0));
        assert_eq!(logistic(100.0), logistic(18.0));
        let xs: Vec<f32> = (-40_000..=40_000).map(|i| i as f32 * 5e-4).collect();
        // The rational form is not monotone in the clamped tails, so only
        // accuracy is checked here.
        for &x in &xs {
            let y = logistic(x);
            let exact = 1.0 / (1.0 + (-(x as f64)).exp());
            assert!((y as f64 - exact).abs() < 2e-6, "{x}: {y} vs {exact}");
        }
        let mut fast = xs.clone();
        logistic_ort_inplace(&mut fast, 3);
        let mut slow = xs.clone();
        range_portable(&mut slow);
        assert!(fast
            .iter()
            .zip(&slow)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}
