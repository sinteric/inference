//! RMSNorm matching EmbeddingGemma 2 / Gemma 4:
//! `y = x * (mean(x^2) + eps)^(-0.5) [* scale]`, computed in f32.
//! Scale is optional (`v_norm` has `with_scale=False`).

use crate::cpu::kernels::bf16::f32_to_bf16_bits;
use crate::cpu::kernels::parallel::{for_ranges, rows_mut, rows_ref};
use crate::cpu::kernels::rope::apply_rope_vec;

/// Single-threaded [`rmsnorm_nthreads`]: the reference the tests compare against.
#[cfg(test)]
pub(crate) fn rmsnorm(x: &[f32], hidden: usize, scale: Option<&[f32]>, eps: f32, y: &mut [f32]) {
    rmsnorm_nthreads(x, hidden, scale, eps, y, 1);
}

pub(crate) fn rmsnorm_nthreads(
    x: &[f32],
    hidden: usize,
    scale: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
    nthreads: usize,
) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(hidden > 0 && x.len().is_multiple_of(hidden));
    if let Some(s) = scale {
        debug_assert_eq!(s.len(), hidden);
    }
    let rows = x.len() / hidden;
    let inv_n = 1.0 / hidden as f32;
    let x_p = x.as_ptr() as usize;
    let y_p = y.as_mut_ptr() as usize;
    let scale_p = scale.map(|s| s.as_ptr() as usize);
    for_ranges(nthreads, rows, 8, |lo, hi| {
        let x = unsafe { rows_ref(x_p as *const f32, lo, hi, hidden) };
        let y = unsafe { rows_mut(y_p as *mut f32, lo, hi, hidden) };
        let scale = scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, hidden) });
        for i in 0..(hi - lo) {
            let row = &x[i * hidden..(i + 1) * hidden];
            let out = &mut y[i * hidden..(i + 1) * hidden];
            let inv = inv_rms(row, inv_n, eps);
            apply_scale_f32(row, scale, inv, out);
        }
    });
}

/// In-place RMSNorm. `inv_rms` is computed from the whole row before any store.
pub(crate) fn rmsnorm_inplace(
    x: &mut [f32],
    hidden: usize,
    scale: Option<&[f32]>,
    eps: f32,
    nthreads: usize,
) {
    debug_assert!(hidden > 0 && x.len().is_multiple_of(hidden));
    if let Some(s) = scale {
        debug_assert_eq!(s.len(), hidden);
    }
    let rows = x.len() / hidden;
    let inv_n = 1.0 / hidden as f32;
    let x_p = x.as_mut_ptr() as usize;
    let scale_p = scale.map(|s| s.as_ptr() as usize);
    for_ranges(nthreads, rows, 8, |lo, hi| {
        let x = unsafe { rows_mut(x_p as *mut f32, lo, hi, hidden) };
        let scale = scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, hidden) });
        for i in 0..(hi - lo) {
            let row = &mut x[i * hidden..(i + 1) * hidden];
            let inv = inv_rms(row, inv_n, eps);
            apply_scale_inplace(row, scale, inv);
        }
    });
}

fn apply_scale_inplace(row: &mut [f32], scale: Option<&[f32]>, inv: f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        // Keep main's SIMD arithmetic without aliasing a shared input slice
        // with the mutable output slice of an in-place operation.
        unsafe { apply_scale_inplace_avx512(row, scale, inv) };
        return;
    }
    if let Some(s) = scale {
        for (x, g) in row.iter_mut().zip(s.iter()) {
            *x = *x * inv * *g;
        }
    } else {
        for x in row.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_scale_inplace_avx512(row: &mut [f32], scale: Option<&[f32]>, inv: f32) {
    use std::arch::x86_64::*;
    let invv = _mm512_set1_ps(inv);
    let mut i = 0;
    if let Some(s) = scale {
        while i + 16 <= row.len() {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            let g = _mm512_loadu_ps(s.as_ptr().add(i));
            _mm512_storeu_ps(
                row.as_mut_ptr().add(i),
                _mm512_mul_ps(_mm512_mul_ps(v, invv), g),
            );
            i += 16;
        }
        while i < row.len() {
            *row.get_unchecked_mut(i) = *row.get_unchecked(i) * inv * *s.get_unchecked(i);
            i += 1;
        }
    } else {
        while i + 16 <= row.len() {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            _mm512_storeu_ps(row.as_mut_ptr().add(i), _mm512_mul_ps(v, invv));
            i += 16;
        }
        while i < row.len() {
            *row.get_unchecked_mut(i) *= inv;
            i += 1;
        }
    }
}

/// `residual[row] += rms(x[row]) [* rms_scale]`, then optionally
/// `residual[row] *= add_scale`. `add_scale == 1` is a plain residual add
/// (attention / FFN). PLE uses `add_scale = layer_scalar` so the sum is
/// multiplied once, matching `add_scale_inplace`.
pub(crate) fn add_rmsnorm(
    residual: &mut [f32],
    x: &[f32],
    hidden: usize,
    rms_scale: Option<&[f32]>,
    add_scale: f32,
    eps: f32,
    nthreads: usize,
) {
    debug_assert_eq!(residual.len(), x.len());
    debug_assert!(hidden > 0 && x.len().is_multiple_of(hidden));
    if let Some(s) = rms_scale {
        debug_assert_eq!(s.len(), hidden);
    }
    let rows = x.len() / hidden;
    let inv_n = 1.0 / hidden as f32;
    let r_p = residual.as_mut_ptr() as usize;
    let x_p = x.as_ptr() as usize;
    let scale_p = rms_scale.map(|s| s.as_ptr() as usize);
    for_ranges(nthreads, rows, 8, |lo, hi| {
        let residual = unsafe { rows_mut(r_p as *mut f32, lo, hi, hidden) };
        let x = unsafe { rows_ref(x_p as *const f32, lo, hi, hidden) };
        let scale = scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, hidden) });
        for i in 0..(hi - lo) {
            let row = &x[i * hidden..(i + 1) * hidden];
            let dst = &mut residual[i * hidden..(i + 1) * hidden];
            let inv = inv_rms(row, inv_n, eps);
            apply_scale_add(row, scale, inv, add_scale, dst);
        }
    });
}

fn apply_scale_add(row: &[f32], scale: Option<&[f32]>, inv: f32, add_scale: f32, dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                apply_scale_add_avx512(row, scale, inv, add_scale, dst);
            }
            return;
        }
    }
    if let Some(s) = scale {
        for c in 0..row.len() {
            let y = row[c] * inv * s[c];
            dst[c] = (dst[c] + y) * add_scale;
        }
    } else {
        for c in 0..row.len() {
            let y = row[c] * inv;
            dst[c] = (dst[c] + y) * add_scale;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_scale_add_avx512(
    row: &[f32],
    scale: Option<&[f32]>,
    inv: f32,
    add_scale: f32,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    let n = row.len();
    let invv = _mm512_set1_ps(inv);
    let asv = _mm512_set1_ps(add_scale);
    let mut i = 0;
    if let Some(s) = scale {
        while i + 16 <= n {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            let g = _mm512_loadu_ps(s.as_ptr().add(i));
            let y = _mm512_mul_ps(_mm512_mul_ps(v, invv), g);
            let acc = _mm512_add_ps(_mm512_loadu_ps(dst.as_ptr().add(i)), y);
            _mm512_storeu_ps(dst.as_mut_ptr().add(i), _mm512_mul_ps(acc, asv));
            i += 16;
        }
        while i < n {
            let y = *row.get_unchecked(i) * inv * *s.get_unchecked(i);
            *dst.get_unchecked_mut(i) = (*dst.get_unchecked(i) + y) * add_scale;
            i += 1;
        }
    } else {
        while i + 16 <= n {
            let y = _mm512_mul_ps(_mm512_loadu_ps(row.as_ptr().add(i)), invv);
            let acc = _mm512_add_ps(_mm512_loadu_ps(dst.as_ptr().add(i)), y);
            _mm512_storeu_ps(dst.as_mut_ptr().add(i), _mm512_mul_ps(acc, asv));
            i += 16;
        }
        while i < n {
            let y = *row.get_unchecked(i) * inv;
            *dst.get_unchecked_mut(i) = (*dst.get_unchecked(i) + y) * add_scale;
            i += 1;
        }
    }
}

/// RMSNorm (+ optional RoPE) writing BF16. `x`/`out` are `[b, nh, s, d]`.
/// `cos`/`sin` are `max * d`; pass `None` to skip RoPE (V heads).
pub(crate) fn rmsnorm_rope_to_bf16(
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
) {
    debug_assert_eq!(x.len(), b * nh * s * d);
    debug_assert_eq!(out.len(), x.len());
    if let Some(g) = rms_scale {
        debug_assert_eq!(g.len(), d);
    }
    let rows = b * nh * s;
    let inv_n = 1.0 / d as f32;
    let x_p = x.as_ptr() as usize;
    let y_p = out.as_mut_ptr() as usize;
    let scale_p = rms_scale.map(|s| s.as_ptr() as usize);
    let (c_p, s_p, rope_len) = match rope {
        Some((c, si)) => (c.as_ptr() as usize, si.as_ptr() as usize, c.len()),
        None => (0, 0, 0),
    };
    for_ranges(nthreads, rows, 8, |lo, hi| {
        let x = unsafe { rows_ref(x_p as *const f32, lo, hi, d) };
        let out = unsafe { rows_mut(y_p as *mut u16, lo, hi, d) };
        let scale = scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, d) });
        let rope = if rope_len > 0 {
            let c = unsafe { std::slice::from_raw_parts(c_p as *const f32, rope_len) };
            let si = unsafe { std::slice::from_raw_parts(s_p as *const f32, rope_len) };
            Some((c, si))
        } else {
            None
        };
        let mut tmp = [0f32; 512];
        debug_assert!(d <= tmp.len());
        for i in 0..(hi - lo) {
            let r = lo + i;
            let p = r % s;
            let row = &x[i * d..(i + 1) * d];
            let dst = &mut tmp[..d];
            let inv = inv_rms(row, inv_n, eps);
            apply_scale_f32(row, scale, inv, dst);
            if let Some((c, si)) = rope {
                apply_rope_vec(dst, &c[p * d..(p + 1) * d], &si[p * d..(p + 1) * d]);
            }
            apply_scale_bf16_from_f32(dst, &mut out[i * d..(i + 1) * d]);
        }
    });
}

/// Split a token-major fused `[b * s, (nq + 2 * nkv) * d]` QKV projection
/// into head-major `[b, nh, s, d]` f32 Q/K/V in one pass: per-head RMSNorm
/// with RoPE on Q and K, unscaled RMSNorm on V. Rows past `b * s` are unread.
pub(crate) fn qkv_split_norm_rope(
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
) {
    let stride = (nq + 2 * nkv) * d;
    debug_assert!(fused.len() >= b * s * stride);
    debug_assert_eq!(q.len(), b * nq * s * d);
    debug_assert_eq!(k.len(), b * nkv * s * d);
    debug_assert_eq!(v.len(), k.len());
    let inv_n = 1.0 / d as f32;
    let f_p = fused.as_ptr() as usize;
    let q_p = q.as_mut_ptr() as usize;
    let k_p = k.as_mut_ptr() as usize;
    let v_p = v.as_mut_ptr() as usize;
    let qs_p = q_scale.map(|g| g.as_ptr() as usize);
    let ks_p = k_scale.map(|g| g.as_ptr() as usize);
    let (c_p, s_p, rope_len) = (
        rope.0.as_ptr() as usize,
        rope.1.as_ptr() as usize,
        rope.0.len(),
    );
    for_ranges(nthreads, b * s, 4, |lo, hi| {
        let qs = qs_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, d) });
        let ks = ks_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, d) });
        let cos = unsafe { std::slice::from_raw_parts(c_p as *const f32, rope_len) };
        let sin = unsafe { std::slice::from_raw_parts(s_p as *const f32, rope_len) };
        for idx in lo..hi {
            let bi = idx / s;
            let p = idx % s;
            let src = unsafe { rows_ref(f_p as *const f32, idx, idx + 1, stride) };
            let c = &cos[p * d..(p + 1) * d];
            let si = &sin[p * d..(p + 1) * d];
            for h in 0..nq {
                let row = &src[h * d..(h + 1) * d];
                let r = (bi * nq + h) * s + p;
                let dst = unsafe { rows_mut(q_p as *mut f32, r, r + 1, d) };
                apply_scale_f32(row, qs, inv_rms(row, inv_n, eps), dst);
                apply_rope_vec(dst, c, si);
            }
            for h in 0..nkv {
                let r = (bi * nkv + h) * s + p;
                let row = &src[(nq + h) * d..(nq + h + 1) * d];
                let dst = unsafe { rows_mut(k_p as *mut f32, r, r + 1, d) };
                apply_scale_f32(row, ks, inv_rms(row, inv_n, eps), dst);
                apply_rope_vec(dst, c, si);
                let row = &src[(nq + nkv + h) * d..(nq + nkv + h + 1) * d];
                let dst = unsafe { rows_mut(v_p as *mut f32, r, r + 1, d) };
                apply_scale_f32(row, None, inv_rms(row, inv_n, eps), dst);
            }
        }
    });
}

/// `out[c][r] = rmsnorm(x[r][c] * pre_scale) * scale` for every `width`-wide
/// chunk `c` of each row `r`: row-major `[rows, chunks * width]` in,
/// chunk-major `[chunks, rows, width]` out. `width` is at most 512.
pub(crate) fn scaled_rmsnorm_chunks(
    x: &[f32],
    rows: usize,
    chunks: usize,
    width: usize,
    pre_scale: f32,
    scale: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
    nthreads: usize,
) {
    debug_assert!(x.len() >= rows * chunks * width && width <= 512);
    debug_assert_eq!(out.len(), rows * chunks * width);
    let inv_n = 1.0 / width as f32;
    let x_p = x.as_ptr() as usize;
    let o_p = out.as_mut_ptr() as usize;
    let scale_p = scale.map(|g| g.as_ptr() as usize);
    for_ranges(nthreads, rows, 4, |lo, hi| {
        let scale = scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, width) });
        let mut tmp = [0f32; 512];
        let tmp = &mut tmp[..width];
        for r in lo..hi {
            let src = unsafe { rows_ref(x_p as *const f32, r, r + 1, chunks * width) };
            for c in 0..chunks {
                for (t, &v) in tmp.iter_mut().zip(&src[c * width..(c + 1) * width]) {
                    *t = v * pre_scale;
                }
                let o = c * rows + r;
                let dst = unsafe { rows_mut(o_p as *mut f32, o, o + 1, width) };
                apply_scale_f32(tmp, scale, inv_rms(tmp, inv_n, eps), dst);
            }
        }
    });
}

/// [`add_rmsnorm`] followed by `y[row] = rmsnorm(residual[row]) [* y_scale]`
/// in the same pass, so the next projection's input needs no second sweep.
pub(crate) fn add_rmsnorm_then_norm(
    residual: &mut [f32],
    x: &[f32],
    hidden: usize,
    rms_scale: Option<&[f32]>,
    add_scale: f32,
    eps: f32,
    y_scale: Option<&[f32]>,
    y: &mut [f32],
    nthreads: usize,
) {
    debug_assert_eq!(residual.len(), x.len());
    debug_assert_eq!(y.len(), x.len());
    debug_assert!(hidden > 0 && x.len().is_multiple_of(hidden));
    let rows = x.len() / hidden;
    let inv_n = 1.0 / hidden as f32;
    let r_p = residual.as_mut_ptr() as usize;
    let x_p = x.as_ptr() as usize;
    let y_p = y.as_mut_ptr() as usize;
    let scale_p = rms_scale.map(|s| s.as_ptr() as usize);
    let y_scale_p = y_scale.map(|s| s.as_ptr() as usize);
    for_ranges(nthreads, rows, 8, |lo, hi| {
        let residual = unsafe { rows_mut(r_p as *mut f32, lo, hi, hidden) };
        let x = unsafe { rows_ref(x_p as *const f32, lo, hi, hidden) };
        let y = unsafe { rows_mut(y_p as *mut f32, lo, hi, hidden) };
        let scale = scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, hidden) });
        let y_scale =
            y_scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, hidden) });
        for i in 0..(hi - lo) {
            let row = &x[i * hidden..(i + 1) * hidden];
            let dst = &mut residual[i * hidden..(i + 1) * hidden];
            apply_scale_add(row, scale, inv_rms(row, inv_n, eps), add_scale, dst);
            let inv = inv_rms(dst, inv_n, eps);
            apply_scale_f32(dst, y_scale, inv, &mut y[i * hidden..(i + 1) * hidden]);
        }
    });
}

fn apply_scale_bf16_from_f32(row: &[f32], out: &mut [u16]) {
    apply_scale_bf16(row, None, 1.0, out);
}

/// RMSNorm writing BF16 (RNE) so a following BF16 matmul does not keep an f32 copy.
pub(crate) fn rmsnorm_to_bf16(
    x: &[f32],
    hidden: usize,
    scale: Option<&[f32]>,
    eps: f32,
    y: &mut [u16],
    nthreads: usize,
) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(hidden > 0 && x.len().is_multiple_of(hidden));
    if let Some(s) = scale {
        debug_assert_eq!(s.len(), hidden);
    }
    let rows = x.len() / hidden;
    let inv_n = 1.0 / hidden as f32;
    let x_p = x.as_ptr() as usize;
    let y_p = y.as_mut_ptr() as usize;
    let scale_p = scale.map(|s| s.as_ptr() as usize);
    for_ranges(nthreads, rows, 8, |lo, hi| {
        let x = unsafe { rows_ref(x_p as *const f32, lo, hi, hidden) };
        let y = unsafe { rows_mut(y_p as *mut u16, lo, hi, hidden) };
        let scale = scale_p.map(|p| unsafe { std::slice::from_raw_parts(p as *const f32, hidden) });
        for i in 0..(hi - lo) {
            let row = &x[i * hidden..(i + 1) * hidden];
            let out = &mut y[i * hidden..(i + 1) * hidden];
            let inv = inv_rms(row, inv_n, eps);
            apply_scale_bf16(row, scale, inv, out);
        }
    });
}

fn inv_rms(row: &[f32], inv_n: f32, eps: f32) -> f32 {
    let mean_sq = mean_sq(row);
    (mean_sq * inv_n + eps).powf(-0.5)
}

fn mean_sq(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { mean_sq_avx512(row) };
        }
    }
    let mut mean_sq = 0.0f32;
    for &v in row {
        mean_sq += v * v;
    }
    mean_sq
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn mean_sq_avx512(row: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = row.len();
    let mut i = 0;
    let mut acc = _mm512_setzero_ps();
    while i + 16 <= n {
        let v = _mm512_loadu_ps(row.as_ptr().add(i));
        acc = _mm512_fmadd_ps(v, v, acc);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    while i < n {
        let v = *row.get_unchecked(i);
        sum += v * v;
        i += 1;
    }
    sum
}

fn apply_scale_f32(row: &[f32], scale: Option<&[f32]>, inv: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                apply_scale_f32_avx512(row, scale, inv, out);
            }
            return;
        }
    }
    if let Some(s) = scale {
        for c in 0..row.len() {
            out[c] = row[c] * inv * s[c];
        }
    } else {
        for c in 0..row.len() {
            out[c] = row[c] * inv;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_scale_f32_avx512(row: &[f32], scale: Option<&[f32]>, inv: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = row.len();
    let invv = _mm512_set1_ps(inv);
    let mut i = 0;
    if let Some(s) = scale {
        while i + 16 <= n {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            let g = _mm512_loadu_ps(s.as_ptr().add(i));
            _mm512_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm512_mul_ps(_mm512_mul_ps(v, invv), g),
            );
            i += 16;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *row.get_unchecked(i) * inv * *s.get_unchecked(i);
            i += 1;
        }
    } else {
        while i + 16 <= n {
            let v = _mm512_loadu_ps(row.as_ptr().add(i));
            _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_mul_ps(v, invv));
            i += 16;
        }
        while i < n {
            *out.get_unchecked_mut(i) = *row.get_unchecked(i) * inv;
            i += 1;
        }
    }
}

fn apply_scale_bf16(row: &[f32], scale: Option<&[f32]>, inv: f32, out: &mut [u16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
            unsafe {
                apply_scale_bf16_avx512(row, scale, inv, out);
            }
            return;
        }
    }
    if let Some(s) = scale {
        for c in 0..row.len() {
            out[c] = f32_to_bf16_bits(row[c] * inv * s[c]);
        }
    } else {
        for c in 0..row.len() {
            out[c] = f32_to_bf16_bits(row[c] * inv);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn apply_scale_bf16_avx512(row: &[f32], scale: Option<&[f32]>, inv: f32, out: &mut [u16]) {
    use std::arch::x86_64::*;
    let n = row.len();
    let invv = _mm512_set1_ps(inv);
    let mut i = 0;
    while i + 32 <= n {
        let mut lo = _mm512_mul_ps(_mm512_loadu_ps(row.as_ptr().add(i)), invv);
        let mut hi = _mm512_mul_ps(_mm512_loadu_ps(row.as_ptr().add(i + 16)), invv);
        if let Some(s) = scale {
            lo = _mm512_mul_ps(lo, _mm512_loadu_ps(s.as_ptr().add(i)));
            hi = _mm512_mul_ps(hi, _mm512_loadu_ps(s.as_ptr().add(i + 16)));
        }
        let packed = _mm512_cvtne2ps_pbh(hi, lo);
        _mm512_storeu_si512(
            out.as_mut_ptr().add(i).cast(),
            std::mem::transmute::<__m512bh, __m512i>(packed),
        );
        i += 32;
    }
    if let Some(s) = scale {
        while i < n {
            *out.get_unchecked_mut(i) =
                f32_to_bf16_bits(*row.get_unchecked(i) * inv * *s.get_unchecked(i));
            i += 1;
        }
    } else {
        while i < n {
            *out.get_unchecked_mut(i) = f32_to_bf16_bits(*row.get_unchecked(i) * inv);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::kernels::bf16::f32_to_bf16_bits;

    #[test]
    fn unit_scale_unit_vector() {
        let x = [3.0f32, 4.0];
        let g = [1.0f32, 1.0];
        let mut y = [0.0f32; 2];
        rmsnorm(&x, 2, Some(&g), 0.0, &mut y);
        // mean(x^2)=12.5, inv=1/sqrt(12.5)=0.2828427, y ≈ [0.8485, 1.1314]
        let n = (y[0] * y[0] + y[1] * y[1]).sqrt();
        assert!((n - 2.0f32.sqrt()).abs() < 1e-5, "{n}");
    }

    #[test]
    fn unscaled_matches_scaled_ones() {
        let x = [1.0f32, -2.0, 3.0];
        let ones = [1.0f32; 3];
        let mut a = [0.0f32; 3];
        let mut b = [0.0f32; 3];
        rmsnorm(&x, 3, None, 1e-6, &mut a);
        rmsnorm(&x, 3, Some(&ones), 1e-6, &mut b);
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() < 1e-6);
        }
    }

    #[test]
    fn bf16_matches_f32_then_convert() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.11 - 2.0).collect();
        let g: Vec<f32> = (0..32).map(|i| 0.9 + i as f32 * 0.01).collect();
        let mut f = vec![0f32; x.len()];
        let mut a = vec![0u16; x.len()];
        let mut b = vec![0u16; x.len()];
        rmsnorm_nthreads(&x, 32, Some(&g), 1e-6, &mut f, 2);
        for (d, &v) in a.iter_mut().zip(&f) {
            *d = f32_to_bf16_bits(v);
        }
        rmsnorm_to_bf16(&x, 32, Some(&g), 1e-6, &mut b, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn qkv_split_matches_norm_then_rope() {
        let (b, s, nq, nkv, d) = (2usize, 5usize, 4usize, 2usize, 32usize);
        let stride = (nq + 2 * nkv) * d;
        let fused: Vec<f32> = (0..b * s * stride + stride)
            .map(|i| ((i * 37 % 101) as f32) * 0.02 - 1.0)
            .collect();
        let qg: Vec<f32> = (0..d).map(|i| 0.9 + i as f32 * 0.01).collect();
        let kg: Vec<f32> = (0..d).map(|i| 1.1 - i as f32 * 0.01).collect();
        let rope = crate::cpu::kernels::DefaultRope::new(d, s, 10_000.0).unwrap();
        let mut q = vec![0f32; b * nq * s * d];
        let mut k = vec![0f32; b * nkv * s * d];
        let mut v = vec![0f32; k.len()];
        qkv_split_norm_rope(
            &fused,
            &mut q,
            &mut k,
            &mut v,
            b,
            s,
            nq,
            nkv,
            d,
            Some(&qg),
            Some(&kg),
            1e-6,
            (&rope.cos, &rope.sin),
            3,
        );
        let mut want = vec![0f32; d];
        for bi in 0..b {
            for p in 0..s {
                let base = (bi * s + p) * stride;
                let (c, si) = rope.row(p, d);
                let heads = [
                    (nq, 0, Some(&qg), &q),
                    (nkv, nq, Some(&kg), &k),
                    (nkv, nq + nkv, None, &v),
                ];
                for (nh, off, g, got) in heads {
                    for h in 0..nh {
                        let row = &fused[base + (off + h) * d..base + (off + h + 1) * d];
                        rmsnorm(row, d, g.map(|g| g.as_slice()), 1e-6, &mut want);
                        if g.is_some() {
                            apply_rope_vec(&mut want, c, si);
                        }
                        let r = (bi * nh + h) * s + p;
                        assert_eq!(&got[r * d..(r + 1) * d], &want[..]);
                    }
                }
            }
        }
    }

    #[test]
    fn scaled_chunks_match_scale_then_rmsnorm() {
        let (rows, chunks, width) = (3usize, 4usize, 48usize);
        let x: Vec<f32> = (0..rows * chunks * width)
            .map(|i| ((i * 13 % 59) as f32) * 0.05 - 1.4)
            .collect();
        let g: Vec<f32> = (0..width).map(|i| 0.7 + i as f32 * 0.02).collect();
        let mut got = vec![0f32; x.len()];
        scaled_rmsnorm_chunks(&x, rows, chunks, width, 0.125, Some(&g), 1e-6, &mut got, 2);
        let scaled: Vec<f32> = x.iter().map(|v| v * 0.125).collect();
        let mut want = vec![0f32; width];
        for r in 0..rows {
            for c in 0..chunks {
                let src = &scaled[(r * chunks + c) * width..(r * chunks + c + 1) * width];
                rmsnorm(src, width, Some(&g), 1e-6, &mut want);
                let o = (c * rows + r) * width;
                assert_eq!(&got[o..o + width], &want[..]);
            }
        }
    }

    #[test]
    fn add_then_norm_matches_two_passes() {
        let hidden = 64;
        let x: Vec<f32> = (0..5 * hidden)
            .map(|i| ((i * 7 % 31) as f32) * 0.1 - 1.5)
            .collect();
        let r0: Vec<f32> = (0..5 * hidden)
            .map(|i| ((i * 11 % 23) as f32) * 0.2 - 2.0)
            .collect();
        let g1: Vec<f32> = (0..hidden).map(|i| 0.5 + i as f32 * 0.01).collect();
        let g2: Vec<f32> = (0..hidden).map(|i| 1.5 - i as f32 * 0.01).collect();
        let mut want_r = r0.clone();
        add_rmsnorm(&mut want_r, &x, hidden, Some(&g1), 0.75, 1e-6, 2);
        let mut want_y = vec![0f32; x.len()];
        rmsnorm_nthreads(&want_r, hidden, Some(&g2), 1e-6, &mut want_y, 2);
        let mut r = r0;
        let mut y = vec![0f32; x.len()];
        add_rmsnorm_then_norm(
            &mut r,
            &x,
            hidden,
            Some(&g1),
            0.75,
            1e-6,
            Some(&g2),
            &mut y,
            3,
        );
        assert_eq!(r, want_r);
        assert_eq!(y, want_y);
    }

    #[test]
    fn inplace_matches_out_of_place() {
        for hidden in [1, 15, 16, 17, 32, 65, 512] {
            let x: Vec<f32> = (0..3 * hidden).map(|i| i as f32 * 0.03 - 1.0).collect();
            let g: Vec<f32> = (0..hidden).map(|i| 0.8 + i as f32 * 0.01).collect();
            for scale in [None, Some(g.as_slice())] {
                let mut a = vec![0f32; x.len()];
                let mut b = x.clone();
                rmsnorm_nthreads(&x, hidden, scale, 1e-6, &mut a, 2);
                rmsnorm_inplace(&mut b, hidden, scale, 1e-6, 2);
                assert_eq!(a, b, "hidden={hidden}, scaled={}", scale.is_some());
            }
        }
    }

    #[test]
    fn add_rmsnorm_matches_separate() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.0).collect();
        let g: Vec<f32> = (0..32).map(|i| 0.85 + i as f32 * 0.01).collect();
        let residual: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        let mut a = x.clone();
        rmsnorm_inplace(&mut a, 32, Some(&g), 1e-6, 2);
        let mut ref_h = residual.clone();
        for (d, s) in ref_h.iter_mut().zip(&a) {
            *d = (*d + *s) * 0.5;
        }
        let mut b = residual.clone();
        add_rmsnorm(&mut b, &x, 32, Some(&g), 0.5, 1e-6, 2);
        for (u, v) in ref_h.iter().zip(&b) {
            assert!((u - v).abs() < 1e-5, "{u} vs {v}");
        }
    }

    #[test]
    fn rmsnorm_rope_to_bf16_matches_separate() {
        let b = 1usize;
        let s = 2usize;
        let nh = 2usize;
        let d = 32usize;
        let x: Vec<f32> = (0..b * nh * s * d)
            .map(|i| (i as f32) * 0.07 - 1.2)
            .collect();
        let g: Vec<f32> = (0..d).map(|i| 0.9 + i as f32 * 0.01).collect();
        let cos: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.01).cos()).collect();
        let sin: Vec<f32> = (0..s * d).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut f = vec![0f32; x.len()];
        rmsnorm_nthreads(&x, d, Some(&g), 1e-6, &mut f, 2);
        let mut want = vec![0u16; x.len()];
        for h in 0..nh {
            for p in 0..s {
                let off = (h * s + p) * d;
                let row = &mut f[off..off + d];
                apply_rope_vec(row, &cos[p * d..(p + 1) * d], &sin[p * d..(p + 1) * d]);
                for i in 0..d {
                    want[off + i] = f32_to_bf16_bits(row[i]);
                }
            }
        }
        let mut got = vec![0u16; x.len()];
        rmsnorm_rope_to_bf16(
            &x,
            &mut got,
            b,
            s,
            nh,
            d,
            Some(&g),
            1e-6,
            Some((&cos, &sin)),
            2,
        );
        assert_eq!(want, got);
    }

    #[test]
    fn rmsnorm_matches_across_thread_counts() {
        let hidden = 768usize;
        let rows = 25usize;
        let x: Vec<f32> = (0..rows * hidden)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.07)
            .collect();
        let scale: Vec<f32> = (0..hidden).map(|i| 0.9 + (i as f32) * 0.0001).collect();
        let mut y1 = vec![0f32; x.len()];
        let mut y4 = vec![0f32; x.len()];
        rmsnorm_nthreads(&x, hidden, Some(&scale), 1e-6, &mut y1, 1);
        rmsnorm_nthreads(&x, hidden, Some(&scale), 1e-6, &mut y4, 4);
        assert_eq!(y1, y4);
        let mut a1 = x.clone();
        let mut a4 = x.clone();
        add_rmsnorm(&mut a1, &x, hidden, Some(&scale), 1.0, 1e-6, 1);
        add_rmsnorm(&mut a4, &x, hidden, Some(&scale), 1.0, 1e-6, 4);
        assert_eq!(a1, a4);
    }
}
