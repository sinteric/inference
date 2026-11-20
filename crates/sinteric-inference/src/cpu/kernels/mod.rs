//! Portable kernels the graphs run between oneDNN primitives: elementwise
//! glue, norms, RoPE, softmax windows, BF16 conversion and the helpers that
//! split work across the pool. `checked` wraps them with shape checks.

#![allow(clippy::too_many_arguments)]

mod bf16;
mod checked;
mod elementwise;
mod gelu;
mod logistic;
mod parallel;
#[cfg(test)]
mod ref_gemm;
mod rmsnorm;
mod rope;
mod seq_matmul;
mod softmax;
mod spatial_mean;

#[cfg(test)]
pub use bf16::f32_to_bf16_bits;
pub use bf16::{bf16_bits_to_f32, SQRT_512_BF16};
pub use gelu::gelu_erf_scalar;
#[cfg(test)]
pub use gelu::gelu_tanh_scalar;
pub use logistic::logistic_ort_inplace;
pub use parallel::{for_ranges, for_row_tiles, row_mut, row_ref, rows_mut, rows_ref};
#[cfg(test)]
pub use ref_gemm::sgemm_ref;
pub use rope::DefaultRope;
pub use seq_matmul::{seq_matmul_nn, seq_matmul_nt};
pub use softmax::{softmax_inplace, softmax_window};
pub use spatial_mean::{spatial_mean, SpatialMean};

pub use checked::{
    add_inplace, add_rmsnorm, add_rmsnorm_then_norm, f32_slice_to_bf16_nthreads, layout_qkv_rope,
    qkv_split_norm_rope, rmsnorm_inplace, rmsnorm_nthreads, rmsnorm_rope_to_bf16, rmsnorm_to_bf16,
    scaled_rmsnorm_chunks, softmax_rows_nthreads,
};

/// GELU gate for callers that validated the enclosing tensor once.
///
/// # Safety
/// `x`, `gate`, and `out` must have identical lengths.
pub unsafe fn gelu_tanh_mul_unchecked(x: &[f32], gate: &[f32], out: &mut [f32]) {
    gelu::gelu_tanh_mul(x, gate, out);
}

pub use checked::f32_slice_to_bf16;

pub use elementwise::scale_inplace;
