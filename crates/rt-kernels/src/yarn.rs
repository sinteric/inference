//! YaRN RoPE matching Vela / ModernBERT:
//! factor=4, original_max=8192, beta_fast=32, beta_slow=1, truncate=true,
//! dim=64, base=160000 (local and global theta are equal in Vela 2.0 0.3B).

#[derive(Clone, Debug)]
pub struct RopeCache {
    pub inv_freq: Vec<f32>,
    pub attention_scaling: f32,
    pub dim: usize,
}

pub fn yarn_inv_freq(
    dim: usize,
    base: f32,
    factor: f32,
    original_max: f32,
    beta_fast: f32,
    beta_slow: f32,
    truncate: bool,
) -> rt_core::Result<RopeCache> {
    rt_core::check_shape!(
        dim >= 2
            && dim.is_multiple_of(2)
            && [base, factor, original_max, beta_fast, beta_slow]
                .iter()
                .all(|v| v.is_finite() && *v > 0.)
            && base != 1.,
        "invalid YaRN parameters"
    );
    rt_core::checked_size("YaRN vectors", &[dim, 4])?;
    let mut pos_freqs = rt_core::try_vec::<f32>(dim / 2)?;
    for (j, p) in pos_freqs.iter_mut().enumerate() {
        let i = j * 2;
        *p = base.powf(i as f32 / dim as f32);
    }
    let inv_extrap: Vec<f32> = pos_freqs.iter().map(|p| 1.0 / p).collect();
    let inv_interp: Vec<f32> = pos_freqs.iter().map(|p| 1.0 / (factor * p)).collect();

    let find_correction_dim = |num_rotations: f32| -> f32 {
        (dim as f32 * (original_max / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * base.ln())
    };
    let mut low = find_correction_dim(beta_fast);
    let mut high = find_correction_dim(beta_slow);
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    let n = dim / 2;
    let mut inv_freq = rt_core::try_vec::<f32>(n)?;
    for i in 0..n {
        let ramp = if high == low {
            1.0
        } else {
            ((i as f32 - low) / (high - low)).clamp(0.0, 1.0)
        };
        // interpolation * (1 - (1-ramp)) + extrapolation * (1-ramp)
        // HF: inv_freq_mask = 1 - linear_ramp(low, high); then
        // interp * (1-mask) + extrap * mask
        let mask = 1.0 - ramp;
        inv_freq[i] = inv_interp[i] * (1.0 - mask) + inv_extrap[i] * mask;
    }
    let attention_scaling = if factor <= 1.0 {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    };
    Ok(RopeCache {
        inv_freq,
        attention_scaling,
        dim,
    })
}

/// Rotate half: cat(-x2, x1) on the last dim.
fn rotate_half(x: &[f32], dim: usize, out: &mut [f32]) {
    let half = dim / 2;
    for i in 0..half {
        out[i] = -x[half + i];
        out[half + i] = x[i];
    }
}

/// Split fused `[b, s, 3*h]` Wqkv, apply YaRN RoPE to Q/K, scale Q, copy V.
/// Cos/sin tables are `[max, d/2]` (not the duplicated last-dim EG2 layout).
pub(crate) fn layout_qkv_yarn(
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
) {
    let h = nh * d;
    let half = d / 2;
    debug_assert!(d >= 2 && d.is_multiple_of(2));
    debug_assert_eq!(fused.len(), b * s * 3 * h);
    debug_assert_eq!(q.len(), b * s * h);
    debug_assert_eq!(k.len(), q.len());
    debug_assert_eq!(v.len(), q.len());
    let ntok = b * s;
    let src_p = fused.as_ptr() as usize;
    let q_p = q.as_mut_ptr() as usize;
    let k_p = k.as_mut_ptr() as usize;
    let v_p = v.as_mut_ptr() as usize;
    let c_p = cos.as_ptr() as usize;
    let si_p = sin.as_ptr() as usize;
    let src_len = fused.len();
    let q_len = q.len();
    let k_len = k.len();
    let v_len = v.len();
    let cos_len = cos.len();
    let sin_len = sin.len();
    crate::parallel::for_ranges(nthreads, ntok, 4, |lo, hi| {
        let src = unsafe { std::slice::from_raw_parts(src_p as *const f32, src_len) };
        let q = unsafe { std::slice::from_raw_parts_mut(q_p as *mut f32, q_len) };
        let k = unsafe { std::slice::from_raw_parts_mut(k_p as *mut f32, k_len) };
        let v = unsafe { std::slice::from_raw_parts_mut(v_p as *mut f32, v_len) };
        let cos = unsafe { std::slice::from_raw_parts(c_p as *const f32, cos_len) };
        let sin = unsafe { std::slice::from_raw_parts(si_p as *const f32, sin_len) };
        for idx in lo..hi {
            let bi = idx / s;
            let p = idx % s;
            let c = &cos[p * half..(p + 1) * half];
            let si = &sin[p * half..(p + 1) * half];
            for head in 0..nh {
                let src_off = (bi * s + p) * 3 * h + head * d;
                let dst = ((bi * nh + head) * s + p) * d;
                yarn_rope_scale(
                    &src[src_off..src_off + d],
                    &mut q[dst..dst + d],
                    c,
                    si,
                    q_scale,
                );
                yarn_rope_scale(
                    &src[src_off + h..src_off + h + d],
                    &mut k[dst..dst + d],
                    c,
                    si,
                    1.0,
                );
                v[dst..dst + d].copy_from_slice(&src[src_off + 2 * h..src_off + 2 * h + d]);
            }
        }
    });
}

fn yarn_rope_scale(x: &[f32], out: &mut [f32], cos: &[f32], sin: &[f32], scale: f32) {
    let half = x.len() / 2;
    debug_assert_eq!(out.len(), x.len());
    debug_assert_eq!(cos.len(), half);
    debug_assert_eq!(sin.len(), half);
    for i in 0..half {
        let a = x[i];
        let b = x[half + i];
        out[i] = (a * cos[i] - b * sin[i]) * scale;
        out[half + i] = (b * cos[i] + a * sin[i]) * scale;
    }
}

/// Apply RoPE to q and k in-place. Layout `[seq, heads, dim]`.
pub fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    seq: usize,
    heads: usize,
    cache: &RopeCache,
) -> rt_core::Result<()> {
    rt_core::check_shape!(
        cache.dim >= 2 && cache.dim.is_multiple_of(2) && cache.inv_freq.len() == cache.dim / 2,
        "invalid YaRN cache"
    );
    let need = rt_core::checked_size("YaRN tensor", &[seq, heads, cache.dim])?;
    rt_core::check_shape!(
        q.len() == need && k.len() == need,
        "YaRN Q/K lengths {}/{}, expected {need}",
        q.len(),
        k.len()
    );
    let dim = cache.dim;
    let mut rh = rt_core::try_vec::<f32>(dim)?;
    let mut cos = rt_core::try_vec::<f32>(dim)?;
    let mut sin = rt_core::try_vec::<f32>(dim)?;
    for t in 0..seq {
        for (i, &freq) in cache.inv_freq.iter().enumerate() {
            let ang = t as f32 * freq;
            let c = ang.cos() * cache.attention_scaling;
            let s = ang.sin() * cache.attention_scaling;
            cos[i] = c;
            cos[i + dim / 2] = c;
            sin[i] = s;
            sin[i + dim / 2] = s;
        }
        for h in 0..heads {
            let off = (t * heads + h) * dim;
            rotate_half(&q[off..off + dim], dim, &mut rh);
            for d in 0..dim {
                q[off + d] = q[off + d] * cos[d] + rh[d] * sin[d];
            }
            rotate_half(&k[off..off + dim], dim, &mut rh);
            for d in 0..dim {
                k[off + d] = k[off + d] * cos[d] + rh[d] * sin[d];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yarn_len() {
        let c = yarn_inv_freq(64, 160_000.0, 4.0, 8192.0, 32.0, 1.0, true).unwrap();
        assert_eq!(c.inv_freq.len(), 32);
        assert!(c.attention_scaling > 1.0);
        assert!(c.inv_freq.iter().all(|x| x.is_finite() && *x > 0.0));
    }

    #[test]
    fn layout_qkv_yarn_matches_scalar() {
        let b = 2usize;
        let s = 4usize;
        let nh = 3usize;
        let d = 8usize;
        let h = nh * d;
        let half = d / 2;
        let fused: Vec<f32> = (0..b * s * 3 * h)
            .map(|i| (i as f32) * 0.01 - 0.2)
            .collect();
        let mut cos = vec![0f32; s * half];
        let mut sin = vec![0f32; s * half];
        for p in 0..s {
            for i in 0..half {
                let ang = (p * half + i) as f32 * 0.05;
                cos[p * half + i] = ang.cos();
                sin[p * half + i] = ang.sin();
            }
        }
        let scale = (d as f32).sqrt().recip();
        let mut q0 = vec![0f32; b * s * h];
        let mut k0 = q0.clone();
        let mut v0 = q0.clone();
        for bi in 0..b {
            for p in 0..s {
                let c = &cos[p * half..(p + 1) * half];
                let si = &sin[p * half..(p + 1) * half];
                for head in 0..nh {
                    let src = (bi * s + p) * 3 * h + head * d;
                    let dst = ((bi * nh + head) * s + p) * d;
                    for i in 0..half {
                        let qa = fused[src + i];
                        let qb = fused[src + half + i];
                        let ka = fused[src + h + i];
                        let kb = fused[src + h + half + i];
                        q0[dst + i] = (qa * c[i] - qb * si[i]) * scale;
                        q0[dst + half + i] = (qb * c[i] + qa * si[i]) * scale;
                        k0[dst + i] = ka * c[i] - kb * si[i];
                        k0[dst + half + i] = kb * c[i] + ka * si[i];
                    }
                    v0[dst..dst + d].copy_from_slice(&fused[src + 2 * h..src + 2 * h + d]);
                }
            }
        }
        let mut q = vec![0f32; q0.len()];
        let mut k = vec![0f32; k0.len()];
        let mut v = vec![0f32; v0.len()];
        layout_qkv_yarn(
            &fused, &mut q, &mut k, &mut v, b, s, nh, d, &cos, &sin, scale, 2,
        );
        for (u, w) in q.iter().zip(&q0) {
            assert!((u - w).abs() < 2e-6, "{u} vs {w}");
        }
        for (u, w) in k.iter().zip(&k0) {
            assert!((u - w).abs() < 2e-6, "{u} vs {w}");
        }
        assert_eq!(v, v0);
    }

    #[test]
    fn rope_preserves_norm() {
        let cache = yarn_inv_freq(4, 10_000.0, 1.0, 2048.0, 32.0, 1.0, true).unwrap();
        let mut q = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut k = q.clone();
        let nq: f32 = q.iter().map(|x| x * x).sum();
        apply_rope(&mut q, &mut k, 1, 1, &cache).unwrap();
        let nq2: f32 = q.iter().map(|x| x * x).sum();
        assert!((nq - nq2).abs() < 1e-4, "{nq} vs {nq2}");
    }
}
