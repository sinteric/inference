//! Portable reference kernels.
//!
//! The oneDNN backend lives in `rt-onednn` (C API). This crate keeps a scalar
//! reference GEMM plus the portable LN/softmax/GELU/YaRN helpers used by
//! asset-free tests.

#![allow(clippy::too_many_arguments)]

mod bf16;
mod checked;
mod elementwise;
mod gelu;
mod layernorm;
mod logistic;
mod parallel;
mod ref_gemm;
mod rmsnorm;
mod rope;
mod seq_matmul;
mod softmax;
mod spatial_mean;
pub use spatial_mean::{spatial_mean, SpatialMean};
mod yarn;

pub use bf16::{bf16_bits_to_f32, f32_to_bf16_bits, SQRT_512_BF16};
pub use gelu::{gelu_erf_scalar, gelu_tanh_scalar};
pub use logistic::logistic_ort_inplace;
pub use parallel::{for_ranges, for_row_tiles, row_mut, row_ref, rows_mut, rows_ref};
pub use ref_gemm::sgemm_ref;
pub use rope::DefaultRope;
pub use seq_matmul::{seq_matmul_nn, seq_matmul_nt};
pub use softmax::{softmax_inplace, softmax_window};

pub use yarn::{apply_rope, yarn_inv_freq, RopeCache};

pub use rt_core::{Error, Result};
use rt_threads::ThreadBudget;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Ref,
    OneDnn,
    Mlas,
}

pub trait F32Gemm: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn nthreads(&self) -> usize;

    /// C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C[M,N], row-major.
    fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) -> Result<()>;
}

/// Naive / portable backend. Always compiled. Used by asset-free tests.
pub struct RefBackend {
    nthreads: usize,
}

impl RefBackend {
    pub fn new(budget: &ThreadBudget) -> Self {
        Self {
            nthreads: budget.nthreads,
        }
    }
}

impl F32Gemm for RefBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Ref
    }
    fn nthreads(&self) -> usize {
        self.nthreads
    }
    fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &[f32],
        lda: usize,
        b: &[f32],
        ldb: usize,
        beta: f32,
        c: &mut [f32],
        ldc: usize,
    ) -> Result<()> {
        sgemm_ref(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
}

/// Named backend constructor. Native kinds return [`rt_core::Error::NativeUnavailable`]
/// here; the oneDNN path is `rt-onednn::OnednnCpu`.
pub fn backend(kind: BackendKind, budget: &ThreadBudget) -> Result<Box<dyn F32Gemm>> {
    match kind {
        BackendKind::Ref => Ok(Box::new(RefBackend::new(budget))),
        BackendKind::OneDnn => {
            let _ = budget;
            Err(rt_core::Error::NativeUnavailable("onednn"))
        }
        BackendKind::Mlas => {
            let _ = budget;
            Err(rt_core::Error::NativeUnavailable("mlas"))
        }
    }
}

pub use checked::{
    add_inplace, add_rmsnorm, add_rmsnorm_then_norm, add_row_bias, add_scale_inplace,
    apply_rope_heads, apply_rope_heads_pair, apply_rope_vec, bf16_slice_to_f32,
    f32_slice_to_bf16_nthreads, gelu_tanh_mul, gelu_tanh_nthreads, layer_norm, layout_qkv_yarn,
    mul_into, qkv_split_norm_rope, rmsnorm, rmsnorm_inplace, rmsnorm_nthreads,
    rmsnorm_rope_to_bf16, rmsnorm_to_bf16, rotate_half_into, scaled_rmsnorm_chunks, softmax_rows,
    softmax_rows_nthreads,
};

/// GELU gate for callers that validated the enclosing tensor once.
///
/// # Safety
/// `x`, `gate`, and `out` must have identical lengths.
pub unsafe fn gelu_tanh_mul_unchecked(x: &[f32], gate: &[f32], out: &mut [f32]) {
    gelu::gelu_tanh_mul(x, gate, out);
}

pub use checked::{f32_slice_to_bf16, gelu_erf, gelu_tanh};

pub use elementwise::scale_inplace;

#[cfg(test)]
mod tests {
    use super::*;
    use rt_threads::ThreadBudget;

    #[test]
    fn ref_gemm_identity() {
        let b = ThreadBudget::new(1).unwrap();
        let k = RefBackend::new(&b);
        let a = [1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let mut c = [0.0f32; 4];
        k.sgemm(2, 2, 2, 1.0, &a, 2, &a, 2, 0.0, &mut c, 2).unwrap();
        // [[1,2],[3,4]]^2 = [[7,10],[15,22]]
        assert!((c[0] - 7.0).abs() < 1e-5);
        assert!((c[1] - 10.0).abs() < 1e-5);
        assert!((c[2] - 15.0).abs() < 1e-5);
        assert!((c[3] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn native_unavailable_is_clean() {
        let b = ThreadBudget::new(1).unwrap();
        match backend(BackendKind::OneDnn, &b) {
            Err(rt_core::Error::NativeUnavailable("onednn")) => {}
            Ok(_) => panic!("trait wrapper must not link the original ABI"),
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn add_scale_matches_add_then_scale() {
        let src: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mut a: Vec<f32> = (0..64).map(|i| 1.0 + i as f32 * 0.05).collect();
        let mut b = a.clone();
        add_inplace(&mut a, &src, 2).unwrap();
        scale_inplace(&mut a, 0.75, 2);
        add_scale_inplace(&mut b, &src, 0.75, 2).unwrap();
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-6, "{u} vs {v}");
        }
    }

    #[test]
    fn add_row_bias_matches_nested() {
        let bias = [0.5f32, -1.0, 2.0, 0.25];
        let mut a: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut b = a.clone();
        for r in 0..3 {
            for c in 0..4 {
                a[r * 4 + c] += bias[c];
            }
        }
        add_row_bias(&mut b, &bias, 2).unwrap();
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-6, "{u} vs {v}");
        }
    }
}
