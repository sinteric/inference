//! Default (non-YaRN) RoPE matching Hugging Face `apply_rotary_pos_emb` + `rotate_half`.
//! Cos/sin last-dim is duplicated: `emb = cat(freqs, freqs)`.

#[derive(Clone, Debug)]
pub struct DefaultRope {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl DefaultRope {
    pub fn new(dim: usize, max: usize, theta: f32) -> crate::Result<Self> {
        crate::check_shape!(
            dim >= 2 && dim.is_multiple_of(2) && max > 0 && theta.is_finite() && theta > 0.0,
            "RoPE dim {dim}, max {max}, theta {theta}"
        );
        let half = dim / 2;
        let mut inv = crate::try_vec::<f32>(half)?;
        for (i, v) in inv.iter_mut().enumerate() {
            *v = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        }
        let mut cos = crate::try_vec::<f32>(crate::checked_size("RoPE table", &[max, dim])?)?;
        let mut sin = crate::try_vec::<f32>(crate::checked_size("RoPE table", &[max, dim])?)?;
        for p in 0..max {
            for i in 0..half {
                let ang = p as f32 * inv[i];
                let c = ang.cos();
                let s = ang.sin();
                let base = p * dim;
                cos[base + i] = c;
                cos[base + half + i] = c;
                sin[base + i] = s;
                sin[base + half + i] = s;
            }
        }
        Ok(Self { cos, sin })
    }

    /// Cos and sin of position `pos` in a table of width `dim`.
    #[cfg(test)]
    pub(crate) fn row(&self, pos: usize, dim: usize) -> (&[f32], &[f32]) {
        let r = pos * dim..(pos + 1) * dim;
        (&self.cos[r.clone()], &self.sin[r])
    }
}

/// Apply RoPE in-place to one head vector of length `dim`.
pub(crate) fn apply_rope_vec(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let dim = x.len();
    debug_assert_eq!(cos.len(), dim);
    debug_assert_eq!(sin.len(), dim);
    debug_assert!(dim.is_multiple_of(2));
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                apply_rope_vec_avx512(x, cos, sin);
            }
            return;
        }
    }
    let half = dim / 2;
    for i in 0..half {
        let x1 = x[i];
        let x2 = x[half + i];
        x[i] = x1 * cos[i] - x2 * sin[i];
        x[half + i] = x2 * cos[half + i] + x1 * sin[half + i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_rope_vec_avx512(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    use std::arch::x86_64::*;
    let half = x.len() / 2;
    let mut i = 0;
    while i + 16 <= half {
        let x1 = _mm512_loadu_ps(x.as_ptr().add(i));
        let x2 = _mm512_loadu_ps(x.as_ptr().add(half + i));
        let c1 = _mm512_loadu_ps(cos.as_ptr().add(i));
        let s1 = _mm512_loadu_ps(sin.as_ptr().add(i));
        let c2 = _mm512_loadu_ps(cos.as_ptr().add(half + i));
        let s2 = _mm512_loadu_ps(sin.as_ptr().add(half + i));
        let y1 = _mm512_fmsub_ps(x1, c1, _mm512_mul_ps(x2, s1));
        let y2 = _mm512_fmadd_ps(x2, c2, _mm512_mul_ps(x1, s2));
        _mm512_storeu_ps(x.as_mut_ptr().add(i), y1);
        _mm512_storeu_ps(x.as_mut_ptr().add(half + i), y2);
        i += 16;
    }
    while i < half {
        let x1 = *x.get_unchecked(i);
        let x2 = *x.get_unchecked(half + i);
        *x.get_unchecked_mut(i) = x1 * *cos.get_unchecked(i) - x2 * *sin.get_unchecked(i);
        *x.get_unchecked_mut(half + i) =
            x2 * *cos.get_unchecked(half + i) + x1 * *sin.get_unchecked(half + i);
        i += 1;
    }
}

/// Split fused `[b, s, 3*h]` Wqkv, apply RoPE to Q/K, scale Q, copy V.
/// Cos/sin tables are `[max, d/2]` (not the duplicated last-dim EG2 layout).
pub(crate) fn layout_qkv_rope(
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
    crate::cpu::kernels::parallel::for_ranges(nthreads, ntok, 4, |lo, hi| {
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
                rope_scale(
                    &src[src_off..src_off + d],
                    &mut q[dst..dst + d],
                    c,
                    si,
                    q_scale,
                );
                rope_scale(
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

fn rope_scale(x: &[f32], out: &mut [f32], cos: &[f32], sin: &[f32], scale: f32) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_qkv_rope_matches_scalar() {
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
        layout_qkv_rope(
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
    fn preserves_l2() {
        let rope = DefaultRope::new(4, 8, 10_000.0).unwrap();
        let mut x = [1.0f32, 0.0, 0.0, 1.0];
        let n0: f32 = x.iter().map(|v| v * v).sum();
        let (c, s) = rope.row(3, 4);
        apply_rope_vec(&mut x, c, s);
        let n1: f32 = x.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-5);
    }

    #[test]
    fn tables_match_head_dim() {
        let local = DefaultRope::new(256, 16, 10_000.0).unwrap();
        let global = DefaultRope::new(512, 16, 1_000_000.0).unwrap();
        assert_eq!(local.cos.len(), 16 * 256);
        assert_eq!(global.sin.len(), 16 * 512);
    }
}
