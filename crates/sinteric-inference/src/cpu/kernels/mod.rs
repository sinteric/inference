//! Portable kernels the graphs run between oneDNN primitives: elementwise
//! glue, norms, RoPE, softmax windows, BF16 conversion and the helpers that
//! split work across the pool. `checked` wraps them with shape checks.

#![expect(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its shapes and strides, each one required (STYLE.md)"
)]

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
pub use parallel::{for_ranges, for_row_tiles, for_rows, for_rows2, RowsMut};
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

pub(crate) use gelu::gelu_tanh_mul;

pub use checked::f32_slice_to_bf16;

pub use elementwise::scale_inplace;
