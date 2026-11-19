//! Default (non-YaRN) RoPE matching Hugging Face `apply_rotary_pos_emb` + `rotate_half`.
//! Cos/sin last-dim is duplicated: `emb = cat(freqs, freqs)`.

#[derive(Clone, Debug)]
pub struct DefaultRope {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub dim: usize,
    pub max: usize,
}

impl DefaultRope {
    pub fn new(dim: usize, max: usize, theta: f32) -> rt_core::Result<Self> {
        rt_core::check_shape!(
            dim >= 2 && dim.is_multiple_of(2) && max > 0 && theta.is_finite() && theta > 0.0,
            "RoPE dim {dim}, max {max}, theta {theta}"
        );
        let half = dim / 2;
        let mut inv = rt_core::try_vec::<f32>(half)?;
        for (i, v) in inv.iter_mut().enumerate() {
            *v = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
        }
        let mut cos = rt_core::try_vec::<f32>(rt_core::checked_size("RoPE table", &[max, dim])?)?;
        let mut sin = rt_core::try_vec::<f32>(rt_core::checked_size("RoPE table", &[max, dim])?)?;
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
        Ok(Self { cos, sin, dim, max })
    }

    pub fn row(&self, pos: usize) -> rt_core::Result<(&[f32], &[f32])> {
        let d = self.dim;
        let off = pos
            .checked_mul(d)
            .ok_or(rt_core::Error::SizeOverflow("RoPE row"))?;
        let end = off
            .checked_add(d)
            .ok_or(rt_core::Error::SizeOverflow("RoPE row"))?;
        rt_core::check_shape!(
            pos < self.max && end <= self.cos.len() && end <= self.sin.len(),
            "RoPE position {pos}, max {}",
            self.max
        );
        Ok((&self.cos[off..end], &self.sin[off..end]))
    }
}

/// `rotate_half`: cat(-x2, x1) on the last dim.
#[inline]
pub(crate) fn rotate_half_into(x: &[f32], out: &mut [f32]) {
    let half = x.len() / 2;
    for i in 0..half {
        out[i] = -x[half + i];
        out[half + i] = x[i];
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

/// Apply RoPE over `[b, nh, s, d]` (row-major) using tables of length `max * d`.
pub(crate) fn apply_rope_heads(
    x: &mut [f32],
    b: usize,
    s: usize,
    nh: usize,
    d: usize,
    cos: &[f32],
    sin: &[f32],
    nthreads: usize,
) {
    debug_assert_eq!(x.len(), b * nh * s * d);
    let rows = b * nh * s;
    let x_p = x.as_mut_ptr() as usize;
    let c_p = cos.as_ptr() as usize;
    let s_p = sin.as_ptr() as usize;
    crate::parallel::for_ranges(nthreads, rows, 16, |lo, hi| {
        let x = unsafe { std::slice::from_raw_parts_mut(x_p as *mut f32, rows * d) };
        let cos = unsafe { std::slice::from_raw_parts(c_p as *const f32, cos.len()) };
        let sin = unsafe { std::slice::from_raw_parts(s_p as *const f32, sin.len()) };
        for r in lo..hi {
            let p = r % s;
            let off = r * d;
            apply_rope_vec(
                &mut x[off..off + d],
                &cos[p * d..(p + 1) * d],
                &sin[p * d..(p + 1) * d],
            );
        }
    });
}

/// Apply the same cos/sin tables to Q and K in one parallel region.
pub(crate) fn apply_rope_heads_pair(
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
) {
    debug_assert_eq!(q.len(), b * nq * s * d);
    debug_assert_eq!(k.len(), b * nkv * s * d);
    let ntok = b * s;
    let q_p = q.as_mut_ptr() as usize;
    let k_p = k.as_mut_ptr() as usize;
    let c_p = cos.as_ptr() as usize;
    let s_p = sin.as_ptr() as usize;
    let q_len = q.len();
    let k_len = k.len();
    let cos_len = cos.len();
    let sin_len = sin.len();
    crate::parallel::for_ranges(nthreads, ntok, 4, |lo, hi| {
        let q = unsafe { std::slice::from_raw_parts_mut(q_p as *mut f32, q_len) };
        let k = unsafe { std::slice::from_raw_parts_mut(k_p as *mut f32, k_len) };
        let cos = unsafe { std::slice::from_raw_parts(c_p as *const f32, cos_len) };
        let sin = unsafe { std::slice::from_raw_parts(s_p as *const f32, sin_len) };
        for idx in lo..hi {
            let bi = idx / s;
            let p = idx % s;
            let c = &cos[p * d..(p + 1) * d];
            let si = &sin[p * d..(p + 1) * d];
            for h in 0..nq {
                let off = ((bi * nq + h) * s + p) * d;
                apply_rope_vec(&mut q[off..off + d], c, si);
            }
            for h in 0..nkv {
                let off = ((bi * nkv + h) * s + p) * d;
                apply_rope_vec(&mut k[off..off + d], c, si);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_l2() {
        let rope = DefaultRope::new(4, 8, 10_000.0).unwrap();
        let mut x = [1.0f32, 0.0, 0.0, 1.0];
        let n0: f32 = x.iter().map(|v| v * v).sum();
        let (c, s) = rope.row(3).unwrap();
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

    #[test]
    fn pair_matches_separate() {
        let rope = DefaultRope::new(8, 4, 10_000.0).unwrap();
        let mut q1: Vec<f32> = (0..192).map(|i| i as f32 * 0.01).collect();
        let mut k1: Vec<f32> = (0..64).map(|i| i as f32 * 0.02).collect();
        let mut q2 = q1.clone();
        let mut k2 = k1.clone();
        apply_rope_heads(&mut q1, 2, 4, 3, 8, &rope.cos, &rope.sin, 2);
        apply_rope_heads(&mut k1, 2, 4, 1, 8, &rope.cos, &rope.sin, 2);
        apply_rope_heads_pair(&mut q2, &mut k2, 2, 4, 3, 1, 8, &rope.cos, &rope.sin, 2);
        for (u, v) in q1.iter().zip(&q2) {
            assert!((u - v).abs() < 1e-6);
        }
        for (u, v) in k1.iter().zip(&k2) {
            assert!((u - v).abs() < 1e-6);
        }
    }
}
