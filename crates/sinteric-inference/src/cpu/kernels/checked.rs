//! Checked entry points. Validation runs before parallel or SIMD kernels.
use crate::{check_shape, checked_size, Result};
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

/// Threaded FP32-to-BF16 conversion using round-to-nearest-even. Source and destination lengths match.
///
/// Returns a shape or size error before modifying output when dimensions are invalid.
pub fn f32_slice_to_bf16_nthreads(src: &[f32], dst: &mut [u16], nthreads: usize) -> Result<()> {
    same(src.len(), dst.len(), "BF16 conversion")?;
    super::bf16::f32_slice_to_bf16_nthreads(src, dst, nthreads);
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
pub fn layout_qkv_rope(
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
    let need = checked_size("RoPE tables", &[s, d / 2])?;
    check_shape!(cos.len() >= need && sin.len() >= need, "short RoPE tables");
    check_shape!(d >= 2 && d.is_multiple_of(2), "RoPE head dimension {d}");
    super::rope::layout_qkv_rope(fused, q, k, v, b, s, nh, d, cos, sin, q_scale, nthreads);
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_shapes_return_errors_before_simd() {
        assert!(f32_slice_to_bf16(&[1.], &mut []).is_err());
        assert!(
            rmsnorm_rope_to_bf16(&[0.], &mut [0], usize::MAX, 2, 2, 2, None, 1e-6, None, 1)
                .is_err()
        );
        assert!(crate::cpu::kernels::DefaultRope::new(0, 1, 10000.).is_err());
        assert!(crate::cpu::kernels::DefaultRope::new(2, usize::MAX, 10000.).is_err());
        assert!(crate::cpu::kernels::for_row_tiles(1, 1, 0, 1, |_, _| {}).is_err());
    }
}
