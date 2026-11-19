//! Checked entry points. Validation runs before parallel or SIMD kernels.
use rt_core::{check_shape, checked_size, Result};
fn same(a: usize, b: usize, what: &str) -> Result<()> {
    check_shape!(a == b, "{what}: length {a} != {b}");
    Ok(())
}
fn norm(x: usize, y: usize, width: usize, scale: Option<&[f32]>) -> Result<()> {
    same(x, y, "normalization output")?;
    check_shape!(
        width > 0 && x.is_multiple_of(width),
        "normalization length {x}, width {width}"
    );
    if let Some(s) = scale {
        same(s.len(), width, "normalization scale")?;
    }
    Ok(())
}
fn heads(len: usize, b: usize, s: usize, nh: usize, d: usize) -> Result<()> {
    check_shape!(
        s > 0 && nh > 0 && d > 0,
        "invalid heads [{b}, {nh}, {s}, {d}]"
    );
    same(
        len,
        checked_size("head tensor", &[b, s, nh, d])?,
        "head tensor",
    )
}
fn rope_shape(c: &[f32], si: &[f32], s: usize, d: usize) -> Result<()> {
    let need = checked_size("RoPE tables", &[s, d])?;
    check_shape!(
        d >= 2 && d.is_multiple_of(2) && c.len() >= need && si.len() >= need,
        "RoPE dimension {d}, positions {s}, tables {}/{}",
        c.len(),
        si.len()
    );
    Ok(())
}

/// Convert BF16 bits to FP32. Source and destination lengths match.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn bf16_slice_to_f32(src: &[u16], dst: &mut [f32]) -> Result<()> {
    same(src.len(), dst.len(), "BF16 conversion")?;
    super::bf16::bf16_slice_to_f32(src, dst);
    Ok(())
}

/// Threaded FP32-to-BF16 conversion using round-to-nearest-even. Source and destination lengths match.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn f32_slice_to_bf16_nthreads(src: &[f32], dst: &mut [u16], nthreads: usize) -> Result<()> {
    same(src.len(), dst.len(), "BF16 conversion")?;
    super::bf16::f32_slice_to_bf16_nthreads(src, dst, nthreads);
    Ok(())
}

/// Write GELU-tanh times a gate. All three slice lengths match.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn gelu_tanh_mul(x: &[f32], f: &[f32], y: &mut [f32]) -> Result<()> {
    same(x.len(), y.len(), "GELU output")?;
    same(x.len(), f.len(), "GELU gate")?;
    super::gelu::gelu_tanh_mul(x, f, y);
    Ok(())
}

/// Threaded GELU-tanh over equal-length input and output slices.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn gelu_tanh_nthreads(x: &[f32], y: &mut [f32], nthreads: usize) -> Result<()> {
    same(x.len(), y.len(), "GELU")?;
    super::gelu::gelu_tanh_nthreads(x, y, nthreads);
    Ok(())
}

/// LayerNorm over rows of `hidden` values. Output matches input; gamma and optional beta have `hidden` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn layer_norm(
    x: &[f32],
    hidden: usize,
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
) -> Result<()> {
    norm(x.len(), y.len(), hidden, Some(gamma))?;
    if let Some(b) = beta {
        same(b.len(), hidden, "LayerNorm bias")?;
    }
    super::layernorm::layer_norm(x, hidden, gamma, beta, eps, y);
    Ok(())
}

/// RMSNorm over rows of `hidden` values, writing BF16. Optional scale has `hidden` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn rmsnorm_to_bf16(
    x: &[f32],
    hidden: usize,
    scale: Option<&[f32]>,
    eps: f32,
    y: &mut [u16],
    nthreads: usize,
) -> Result<()> {
    norm(x.len(), y.len(), hidden, scale)?;
    super::rmsnorm::rmsnorm_to_bf16(x, hidden, scale, eps, y, nthreads);
    Ok(())
}

/// RMSNorm and optional RoPE on `[b, nh, s, d]`, writing BF16. `d` is at most 512; RoPE tables contain at least `s * d` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn rmsnorm_rope_to_bf16(
    x: &[f32],
    out: &mut [u16],
    b: usize,
    s: usize,
    nh: usize,
    d: usize,
    rms_scale: Option<&[f32]>,
    eps: f32,
    rope: Option<(&[f32], &[f32])>,
    nthreads: usize,
) -> Result<()> {
    heads(x.len(), b, s, nh, d)?;
    norm(x.len(), out.len(), d, rms_scale)?;
    check_shape!(d <= 512, "RMSNorm RoPE head dimension {d} exceeds 512");
    if let Some((c, si)) = rope {
        rope_shape(c, si, s, d)?;
    }
    super::rmsnorm::rmsnorm_rope_to_bf16(x, out, b, s, nh, d, rms_scale, eps, rope, nthreads);
    Ok(())
}

/// Split a fused `[b * s, (nq + 2 * nkv) * d]` QKV projection into
/// head-major f32 Q/K/V `[b, nh, s, d]`: RMSNorm + RoPE on Q/K, unscaled
/// RMSNorm on V. `d` is even; RoPE tables contain at least `s * d` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn qkv_split_norm_rope(
    fused: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    q_scale: Option<&[f32]>,
    k_scale: Option<&[f32]>,
    eps: f32,
    rope: (&[f32], &[f32]),
    nthreads: usize,
) -> Result<()> {
    heads(q.len(), b, s, nq, d)?;
    heads(k.len(), b, s, nkv, d)?;
    same(v.len(), k.len(), "V heads")?;
    let need = checked_size("fused QKV", &[b, s, nq + 2 * nkv, d])?;
    check_shape!(
        fused.len() >= need,
        "fused QKV length {} < {need}",
        fused.len()
    );
    for g in [q_scale, k_scale].into_iter().flatten() {
        same(g.len(), d, "head norm scale")?;
    }
    rope_shape(rope.0, rope.1, s, d)?;
    super::rmsnorm::qkv_split_norm_rope(
        fused, q, k, v, b, s, nq, nkv, d, q_scale, k_scale, eps, rope, nthreads,
    );
    Ok(())
}

/// RMSNorm of each `width` chunk of `[rows, chunks * width]` after
/// multiplying by `pre_scale`, written chunk-major `[chunks, rows, width]`.
/// `width` is at most 512.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn scaled_rmsnorm_chunks(
    x: &[f32],
    rows: usize,
    chunks: usize,
    width: usize,
    pre_scale: f32,
    scale: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
    nthreads: usize,
) -> Result<()> {
    let n = checked_size("chunked rows", &[rows, chunks, width])?;
    check_shape!(
        width > 0 && width <= 512 && x.len() >= n,
        "chunked RMSNorm width {width}, input {} < {n}",
        x.len()
    );
    same(out.len(), n, "chunked RMSNorm output")?;
    if let Some(g) = scale {
        same(g.len(), width, "normalization scale")?;
    }
    super::rmsnorm::scaled_rmsnorm_chunks(
        x, rows, chunks, width, pre_scale, scale, eps, out, nthreads,
    );
    Ok(())
}

/// [`add_rmsnorm`], then `y = rmsnorm(residual) [* y_scale]` in the same pass.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn add_rmsnorm_then_norm(
    residual: &mut [f32],
    x: &[f32],
    hidden: usize,
    rms_scale: Option<&[f32]>,
    add_scale: f32,
    eps: f32,
    y_scale: Option<&[f32]>,
    y: &mut [f32],
    nthreads: usize,
) -> Result<()> {
    norm(x.len(), residual.len(), hidden, rms_scale)?;
    norm(x.len(), y.len(), hidden, y_scale)?;
    super::rmsnorm::add_rmsnorm_then_norm(
        residual, x, hidden, rms_scale, add_scale, eps, y_scale, y, nthreads,
    );
    Ok(())
}

/// Normalize each row, add it to the residual, then apply `add_scale`. Input and residual lengths match; optional scale has `hidden` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn add_rmsnorm(
    residual: &mut [f32],
    x: &[f32],
    hidden: usize,
    rms_scale: Option<&[f32]>,
    add_scale: f32,
    eps: f32,
    nthreads: usize,
) -> Result<()> {
    norm(x.len(), residual.len(), hidden, rms_scale)?;
    super::rmsnorm::add_rmsnorm(residual, x, hidden, rms_scale, add_scale, eps, nthreads);
    Ok(())
}

/// In-place RMSNorm over rows of `hidden` values. Optional scale has `hidden` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn rmsnorm_inplace(
    x: &mut [f32],
    hidden: usize,
    scale: Option<&[f32]>,
    eps: f32,
    nthreads: usize,
) -> Result<()> {
    norm(x.len(), x.len(), hidden, scale)?;
    super::rmsnorm::rmsnorm_inplace(x, hidden, scale, eps, nthreads);
    Ok(())
}

/// Threaded RMSNorm over rows of `hidden` values. Input/output lengths match; optional scale has `hidden` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn rmsnorm_nthreads(
    x: &[f32],
    hidden: usize,
    scale: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
    nthreads: usize,
) -> Result<()> {
    norm(x.len(), y.len(), hidden, scale)?;
    super::rmsnorm::rmsnorm_nthreads(x, hidden, scale, eps, y, nthreads);
    Ok(())
}

/// RMSNorm over rows of `hidden` values. Input/output lengths match; optional scale has `hidden` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn rmsnorm(
    x: &[f32],
    hidden: usize,
    scale: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
) -> Result<()> {
    norm(x.len(), y.len(), hidden, scale)?;
    super::rmsnorm::rmsnorm(x, hidden, scale, eps, y);
    Ok(())
}

/// Apply RoPE to Q `[b, nq, s, d]` and K `[b, nkv, s, d]`. Cos/sin contain at least `s * d` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn apply_rope_heads_pair(
    q: &mut [f32],
    k: &mut [f32],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    cos: &[f32],
    sin: &[f32],
    nthreads: usize,
) -> Result<()> {
    heads(q.len(), b, s, nq, d)?;
    heads(k.len(), b, s, nkv, d)?;
    rope_shape(cos, sin, s, d)?;
    super::rope::apply_rope_heads_pair(q, k, b, s, nq, nkv, d, cos, sin, nthreads);
    Ok(())
}

/// Apply RoPE to `[b, nh, s, d]`. Cos/sin contain at least `s * d` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn apply_rope_heads(
    x: &mut [f32],
    b: usize,
    s: usize,
    nh: usize,
    d: usize,
    cos: &[f32],
    sin: &[f32],
    nthreads: usize,
) -> Result<()> {
    heads(x.len(), b, s, nh, d)?;
    rope_shape(cos, sin, s, d)?;
    super::rope::apply_rope_heads(x, b, s, nh, d, cos, sin, nthreads);
    Ok(())
}

/// Apply RoPE to one even-width head. Cos/sin contain at least one head row.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn apply_rope_vec(x: &mut [f32], cos: &[f32], sin: &[f32]) -> Result<()> {
    rope_shape(cos, sin, 1, x.len())?;
    super::rope::apply_rope_vec(x, cos, sin);
    Ok(())
}

/// Write `cat(-x2, x1)` to output. Input/output lengths match and are even.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn rotate_half_into(x: &[f32], out: &mut [f32]) -> Result<()> {
    same(x.len(), out.len(), "rotate_half")?;
    check_shape!(x.len().is_multiple_of(2), "odd rotate_half dimension");
    super::rope::rotate_half_into(x, out);
    Ok(())
}

/// Threaded in-place softmax over positive-width rows. The slice length is a multiple of `cols`.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn softmax_rows_nthreads(logits: &mut [f32], cols: usize, nthreads: usize) -> Result<()> {
    check_shape!(
        cols > 0 && logits.len().is_multiple_of(cols),
        "softmax shape: {} values, {cols} columns",
        logits.len()
    );
    super::softmax::softmax_rows_nthreads(logits, cols, nthreads);
    Ok(())
}

/// In-place softmax over positive-width rows. The slice length is a multiple of `cols`.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn softmax_rows(logits: &mut [f32], cols: usize) -> Result<()> {
    check_shape!(
        cols > 0 && logits.len().is_multiple_of(cols),
        "softmax shape: {} values, {cols} columns",
        logits.len()
    );
    super::softmax::softmax_rows(logits, cols);
    Ok(())
}

/// Multiply equal-length slices into an equally sized destination.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn mul_into(a: &[f32], b: &[f32], out: &mut [f32], nthreads: usize) -> Result<()> {
    same(a.len(), b.len(), "multiply")?;
    same(a.len(), out.len(), "multiply output")?;
    super::elementwise::mul_into(a, b, out, nthreads);
    Ok(())
}

/// Compute `(dst + src) * scale` for equal-length slices.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn add_scale_inplace(dst: &mut [f32], src: &[f32], scale: f32, nthreads: usize) -> Result<()> {
    same(dst.len(), src.len(), "scaled residual addition")?;
    super::elementwise::add_scale_inplace(dst, src, scale, nthreads);
    Ok(())
}

/// Add a nonempty bias to each matrix row. Destination length is a multiple of bias length.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn add_row_bias(dst: &mut [f32], bias: &[f32], nthreads: usize) -> Result<()> {
    check_shape!(
        !bias.is_empty() && dst.len().is_multiple_of(bias.len()),
        "bias width {} for matrix length {}",
        bias.len(),
        dst.len()
    );
    super::elementwise::add_row_bias(dst, bias, nthreads);
    Ok(())
}

/// Add equal-length source values to the destination.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn add_inplace(dst: &mut [f32], src: &[f32], nthreads: usize) -> Result<()> {
    same(dst.len(), src.len(), "residual addition")?;
    super::elementwise::add_inplace(dst, src, nthreads);
    Ok(())
}

/// Split `[b, s, 3 * nh * d]` into `[b, nh, s, d]` Q/K/V, apply YaRN and scale Q. Cos/sin contain at least `s * d / 2` values.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn layout_qkv_yarn(
    fused: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    b: usize,
    s: usize,
    nh: usize,
    d: usize,
    cos: &[f32],
    sin: &[f32],
    q_scale: f32,
    nthreads: usize,
) -> Result<()> {
    heads(q.len(), b, s, nh, d)?;
    same(q.len(), k.len(), "QK shape")?;
    same(q.len(), v.len(), "QV shape")?;
    same(
        fused.len(),
        checked_size("fused QKV", &[q.len(), 3])?,
        "fused QKV",
    )?;
    let need = checked_size("YaRN tables", &[s, d / 2])?;
    check_shape!(cos.len() >= need && sin.len() >= need, "short YaRN tables");
    check_shape!(d >= 2 && d.is_multiple_of(2), "YaRN head dimension {d}");
    super::yarn::layout_qkv_yarn(fused, q, k, v, b, s, nh, d, cos, sin, q_scale, nthreads);
    Ok(())
}

/// Convert FP32 to BF16 using round-to-nearest-even. Source and destination lengths match.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn f32_slice_to_bf16(src: &[f32], dst: &mut [u16]) -> Result<()> {
    same(src.len(), dst.len(), "BF16 conversion")?;
    super::bf16::f32_slice_to_bf16(src, dst);
    Ok(())
}
/// Apply GELU-erf to equal-length input and output slices.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn gelu_erf(x: &[f32], y: &mut [f32]) -> Result<()> {
    same(x.len(), y.len(), "GELU")?;
    super::gelu::gelu_erf(x, y);
    Ok(())
}
/// Apply GELU-tanh to equal-length input and output slices.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn gelu_tanh(x: &[f32], y: &mut [f32]) -> Result<()> {
    same(x.len(), y.len(), "GELU")?;
    super::gelu::gelu_tanh(x, y);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_shapes_return_errors_before_simd() {
        assert!(f32_slice_to_bf16(&[1.], &mut []).is_err());
        assert!(bf16_slice_to_f32(&[1], &mut []).is_err());
        assert!(gelu_erf(&[1.], &mut []).is_err());
        assert!(gelu_tanh_mul(&[1.], &[], &mut [0.]).is_err());
        assert!(layer_norm(&[1.], 0, &[], None, 1e-6, &mut [0.]).is_err());
        assert!(rmsnorm(&[1.], 1, Some(&[]), 1e-6, &mut [0.]).is_err());
        assert!(
            rmsnorm_rope_to_bf16(&[0.], &mut [0], usize::MAX, 2, 2, 2, None, 1e-6, None, 1)
                .is_err()
        );
        assert!(apply_rope_vec(&mut [1., 2.], &[], &[]).is_err());
        assert!(rotate_half_into(&[1.], &mut [0.]).is_err());
        assert!(softmax_rows(&mut [1.], 0).is_err());
        assert!(add_row_bias(&mut [0.], &[], 1).is_err());
        assert!(mul_into(&[], &[1.], &mut [], 1).is_err());
        assert!(crate::DefaultRope::new(0, 1, 10000.).is_err());
        assert!(crate::DefaultRope::new(2, usize::MAX, 10000.).is_err());
        assert!(crate::DefaultRope::new(2, 1, 10000.)
            .unwrap()
            .row(usize::MAX)
            .is_err());
        assert!(crate::for_row_tiles(1, 1, 0, 1, |_, _| {}).is_err());
        assert!(crate::yarn_inv_freq(3, 10000., 4., 8192., 32., 1., true).is_err());
    }
}
