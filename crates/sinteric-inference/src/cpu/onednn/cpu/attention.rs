//! Attention on oneDNN: bidirectional, optionally local (a radius around each
//! query), with excluded key positions per sequence. QK and PV are batched
//! GEMMs over query tiles; the mask and the output scatter run on the pool.

use super::*;

/// Validate attention tensors once before any row-parallel work or pointer use.
pub(super) fn validate_attention<T>(
    q: &[T],
    k: &[T],
    v: &[T],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    radius: usize,
    padding: &[Vec<usize>],
) -> Result<()> {
    crate::check_shape!(
        b > 0 && s > 0 && nq > 0 && nkv > 0 && nq.is_multiple_of(nkv) && d > 0 && tile > 0,
        "attention shape [{b}, {nq}/{nkv}, {s}, {d}], tile {tile}"
    );
    let qn = crate::checked_size("attention queries", &[b, nq, s, d])?;
    let kn = crate::checked_size("attention keys", &[b, nkv, s, d])?;
    crate::check_shape!(
        q.len() == qn && k.len() == kn && v.len() == kn,
        "attention tensor lengths {}/{}/{}, expected {qn}/{kn}/{kn}",
        q.len(),
        k.len(),
        v.len()
    );
    crate::check_shape!(
        padding.len() == b && padding.iter().all(|r| r.iter().all(|&p| p < s)),
        "attention padding indices outside batch {b}, sequence {s}"
    );
    s.checked_add(radius)
        .and_then(|n| n.checked_add(1))
        .ok_or(crate::Error::size_overflow("attention radius"))?;
    crate::checked_size("attention score bytes", &[b, nq, s, s.min(tile), 4])?;
    Ok(())
}

/// FP32 attention: tiled QK and PV GEMMs and oneDNN softmax, with a
/// threaded mask (`+= f32::MIN`) and output scatter.
pub(super) fn attention_fp32(
    hw: &OnednnCpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nh: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    validate_attention(q, k, v, b, s, nh, nh, d, tile, radius, padding)?;
    let g = b * nh;
    let threads = hw.threads.max(1) as usize;
    let mut out = hw.buffer(crate::checked_size("activation elements", &[b, s, nh, d])?)?;
    let tile = tile.min(s).max(1);
    for start in (0..s).step_by(tile) {
        let n = tile.min(s - start);
        let ks = if local {
            start.saturating_sub(radius)
        } else {
            0
        };
        let ke = if local {
            (start + n + radius).min(s)
        } else {
            s
        };
        let nk = ke - ks;
        let mut scores = hw.mm(
            g,
            n,
            nk,
            d,
            s * d,
            d,
            s * d,
            d,
            true,
            &q[start * d..],
            &k[ks * d..],
        )?;
        let scores_p = scores.as_mut_ptr() as usize;
        let rows = g * n;
        for_ranges(threads, rows, 4, |rlo, rhi| {
            for row_i in rlo..rhi {
                let gi = row_i / n;
                let qi = row_i % n;
                let excluded = &padding[gi / nh];
                if !local && excluded.is_empty() {
                    continue;
                }
                let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                let lo = if local {
                    (start + qi).saturating_sub(radius).saturating_sub(ks)
                } else {
                    0
                };
                let hi = if local {
                    (start + qi + radius + 1).min(ke) - ks
                } else {
                    nk
                };
                for score in &mut row[..lo] {
                    *score += f32::MIN;
                }
                for score in &mut row[hi..] {
                    *score += f32::MIN;
                }
                for &pos in excluded {
                    if pos >= ks + lo && pos < ks + hi {
                        row[pos - ks] += f32::MIN;
                    }
                }
            }
        });
        // oneDNN's softmax: a small slice of the attention time, and the
        // rounding the models' references were checked against.
        let probs = hw.softmax(g * n, nk, &scores)?;
        hw.recycle(scores);
        let y = hw.mm(
            g,
            n,
            d,
            nk,
            n * nk,
            nk,
            s * d,
            d,
            false,
            &probs,
            &v[ks * d..],
        )?;
        hw.recycle(probs);
        {
            let y_p = y.as_ptr() as usize;
            let out_p = out.as_mut_ptr() as usize;
            let rows = b * n;
            for_ranges(threads, rows, 8, |lo, hi| {
                for idx in lo..hi {
                    let bi = idx / n;
                    let qi = idx % n;
                    for head in 0..nh {
                        let src = ((bi * nh + head) * n + qi) * d;
                        let dst = (bi * s + start + qi) * (nh * d) + head * d;
                        let src = unsafe { row_ref(y_p as *const f32, src / d, d) };
                        let dst = unsafe { row_mut(out_p as *mut f32, dst / d, d) };
                        dst.copy_from_slice(src);
                    }
                }
            });
        }
        hw.recycle(y);
    }
    Ok(out)
}

pub(super) fn attention_gqa_bf16(
    hw: &OnednnCpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    scale: f32,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    validate_attention(q, k, v, b, s, nq, nkv, d, tile, radius, padding)?;
    if nq == 0 || nkv == 0 || !nq.is_multiple_of(nkv) {
        return Err(crate::Error::internal("GQA head grouping"));
    }
    let threads = hw.threads.max(1) as usize;
    let q_scale = if (scale - 1.0).abs() > 0.0 {
        let mut qs = hw.buffer(q.len())?;
        let q_p = q.as_ptr() as usize;
        let qs_p = qs.as_mut_ptr() as usize;
        let n = q.len();
        for_ranges(threads, n, 4096, |lo, hi| {
            let q = unsafe { rows_ref(q_p as *const f32, lo, hi, 1) };
            let qs = unsafe { rows_mut(qs_p as *mut f32, lo, hi, 1) };
            for (dst, &src) in qs.iter_mut().zip(q.iter()) {
                *dst = src * scale;
            }
        });
        qs
    } else {
        Vec::new()
    };
    let q_ref: &[f32] = if q_scale.is_empty() { q } else { &q_scale };
    let mut q_bf = hw.buffer_u16(q_ref.len())?;
    let mut k_bf = hw.buffer_u16(k.len())?;
    let mut v_bf = hw.buffer_u16(v.len())?;
    let t_conv = Instant::now();
    f32_slice_to_bf16_nthreads(q_ref, &mut q_bf, threads)?;
    f32_slice_to_bf16_nthreads(k, &mut k_bf, threads)?;
    f32_slice_to_bf16_nthreads(v, &mut v_bf, threads)?;
    hw.add_convert_s(t_conv.elapsed().as_secs_f64());
    if !q_scale.is_empty() {
        hw.recycle(q_scale);
    }
    let out = attention_gqa_from_bf16(
        hw, &q_bf, &k_bf, &v_bf, b, s, nq, nkv, d, tile, local, radius, padding,
    );
    hw.recycle_u16(q_bf);
    hw.recycle_u16(k_bf);
    hw.recycle_u16(v_bf);
    out
}

/// Main's tiled QK → softmax → PV (oneshot oneDNN, f32 P, AMX short-M PV pad).
/// Preconverted BF16 Q/K/V skip the f32→bf16 convert.
pub(super) fn attention_gqa_from_bf16(
    hw: &OnednnCpu,
    q_bf: &[u16],
    k_bf: &[u16],
    v_bf: &[u16],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    validate_attention(q_bf, k_bf, v_bf, b, s, nq, nkv, d, tile, radius, padding)?;
    if nq == 0 || nkv == 0 || !nq.is_multiple_of(nkv) {
        return Err(crate::Error::internal("GQA head grouping"));
    }
    let groups = nq / nkv;
    let threads = hw.threads.max(1) as usize;
    let g_kv = b * nkv;
    let live_rows = b * s;
    let mut out = hw.buffer(crate::checked_size(
        "activation elements",
        &[hw.strategy.residual_rows(live_rows), nq, d],
    )?)?;
    // Only live queries/keys enter attention. o_proj consumes this zero tail
    // directly, avoiding a per-layer allocation and live-to-padded copy.
    out[live_rows * nq * d..].fill(0.0);
    // Clamp to the live sequence (and sliding window). Do not round the *query*
    // tile up to a constant 256 — that made B1×128 pay for a 256-row shape.
    // AMX folds a starving last window from shape only (777 → 256+256+265).
    // B1/B4×300 keep step_by (rem=44 pad waste 8/88). AVX-512 is always
    // step_by(tile). Thread count does not change the split.
    let tile = crate::cpu::onednn::clamp_attn_tile(tile, s, local, radius, b, nq);
    let q_ranges = hw.strategy.attn_query_ranges(s, tile, b, nq, nkv, local);
    let q_max = q_ranges.iter().map(|&(_, n)| n).max().unwrap_or(1);
    let mut q_tile = if q_max < s {
        hw.buffer_u16(crate::checked_size(
            "activation elements",
            &[g_kv, groups, q_max, d],
        )?)?
    } else {
        Vec::new()
    };
    for (start, n) in q_ranges {
        let ks = if local {
            start.saturating_sub(radius)
        } else {
            0
        };
        let ke = if local {
            (start + n + radius).min(s)
        } else {
            s
        };
        let nk = ke - ks;
        let m = groups * n;
        let (q_ptr, q_ab): (&[u16], usize) = if start == 0 && n == s {
            (q_bf, groups * s * d)
        } else {
            gather_q_tile(q_bf, &mut q_tile, b, nkv, groups, s, d, start, n, threads);
            (&q_tile[..g_kv * m * d], m * d)
        };
        let t_qk = Instant::now();
        let mut scores = hw.mm_bf16_src(
            g_kv,
            m,
            nk,
            d,
            q_ab,
            d,
            s * d,
            d,
            true,
            q_ptr,
            &k_bf[ks * d..],
        )?;
        hw.add_category("attn_qk", t_qk.elapsed().as_secs_f64());
        let blocked_by_batch: Vec<Vec<usize>> = padding
            .iter()
            .map(|pad| {
                pad.iter()
                    .copied()
                    // Filter before subtracting: then_some eagerly evaluates
                    // p - ks and underflows in debug builds for earlier keys.
                    .filter(|&p| p >= ks && p < ke)
                    .map(|p| p - ks)
                    .collect()
            })
            .collect();
        let t_sm = Instant::now();
        let probs = if local {
            let scores_p = scores.as_mut_ptr() as usize;
            let rows = g_kv * m;
            for_ranges(threads, rows, 4, |rlo, rhi| {
                for row_i in rlo..rhi {
                    let gi = row_i / m;
                    let rest = row_i % m;
                    let qi = rest % n;
                    let bi = gi / nkv;
                    let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                    let qpos = start + qi;
                    let lo = qpos.saturating_sub(radius).saturating_sub(ks);
                    let hi = (qpos + radius + 1).min(ke) - ks;
                    softmax_window(row, lo, hi, &blocked_by_batch[bi]);
                }
            });
            scores
        } else {
            if blocked_by_batch.iter().any(|p| !p.is_empty()) {
                let scores_p = scores.as_mut_ptr() as usize;
                let rows = g_kv * m;
                for_ranges(threads, rows, 4, |rlo, rhi| {
                    for row_i in rlo..rhi {
                        let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                        let bi = (row_i / m) / nkv;
                        for &pos in &blocked_by_batch[bi] {
                            row[pos] = f32::NEG_INFINITY;
                        }
                    }
                });
            }
            if g_kv * m >= 512 && nk >= 256 {
                let probs = hw.softmax(g_kv * m, nk, &scores)?;
                hw.recycle(scores);
                probs
            } else {
                softmax_rows_nthreads(&mut scores, nk, threads)?;
                scores
            }
        };
        hw.add_category("attn_softmax", t_sm.elapsed().as_secs_f64());
        let t_pv = Instant::now();
        let y = if hw.strategy.attn_pad_m(m) == m {
            let mut y = hw.buffer(crate::checked_size("activation elements", &[g_kv, m, d])?)?;
            hw.run_ex(
                Key {
                    kind: 0,
                    g: g_kv,
                    m,
                    n: d,
                    k: nk,
                    ab: m * nk,
                    ar: nk,
                    bb: s * d,
                    br: d,
                    trans: false,
                    param: 0,
                    sum: false,
                    weight: 0,
                    dt: MatDt::Bf16,
                },
                &probs,
                None,
                None,
                Some(&v_bf[ks * d..]),
                None,
                &mut y,
                None,
            )?;
            y
        } else {
            hw.attn_pv_inner(g_kv, m, d, nk, &probs, &v_bf[ks * d..], s * d, d)?
        };
        hw.add_category("attn_pv", t_pv.elapsed().as_secs_f64());
        hw.recycle(probs);
        {
            let y_p = y.as_ptr() as usize;
            let out_p = out.as_mut_ptr() as usize;
            let rows = b * n;
            for_ranges(threads, rows, 8, |lo, hi| {
                for idx in lo..hi {
                    let bi = idx / n;
                    let qi = idx % n;
                    for kvh in 0..nkv {
                        for grp in 0..groups {
                            let qh = kvh * groups + grp;
                            let src = (((bi * nkv + kvh) * groups + grp) * n + qi) * d;
                            let dst = (bi * s + start + qi) * (nq * d) + qh * d;
                            let src = unsafe { row_ref(y_p as *const f32, src / d, d) };
                            let dst = unsafe { row_mut(out_p as *mut f32, dst / d, d) };
                            dst.copy_from_slice(src);
                        }
                    }
                }
            });
        }
        hw.recycle(y);
    }
    hw.recycle_u16(q_tile);
    Ok(out)
}

/// FP32 GQA attention: per KV head, the `groups` query heads that share it
/// form one `groups * n`-row f32 matmul against K and V (no `repeat_kv`).
/// Output is exactly `[b * s, nq * d]`, token-major, with no residual pad.
pub(super) fn attention_gqa_f32(
    hw: &OnednnCpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    scale: f32,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    validate_attention(q, k, v, b, s, nq, nkv, d, tile, radius, padding)?;
    let groups = nq / nkv;
    let threads = hw.threads.max(1) as usize;
    let g_kv = b * nkv;
    let q_scaled = if (scale - 1.0).abs() > 0.0 {
        let mut qs = hw.buffer(q.len())?;
        for (dst, &src) in qs.iter_mut().zip(q) {
            *dst = src * scale;
        }
        qs
    } else {
        Vec::new()
    };
    let q: &[f32] = if q_scaled.is_empty() { q } else { &q_scaled };
    let mut out = hw.buffer(crate::checked_size("activation elements", &[b * s, nq, d])?)?;
    let tile = crate::cpu::onednn::clamp_attn_tile(tile, s, local, radius, b, nq);
    // f32 runs on AVX-512 FMA on every host, so AMX tail folding never applies.
    let q_ranges =
        crate::cpu::onednn::IsaStrategy::Avx512.attn_query_ranges(s, tile, b, nq, nkv, local);
    let q_max = q_ranges.iter().map(|&(_, n)| n).max().unwrap_or(1);
    let mut q_tile = if q_max < s {
        hw.buffer(crate::checked_size(
            "activation elements",
            &[g_kv, groups, q_max, d],
        )?)?
    } else {
        Vec::new()
    };
    for (start, n) in q_ranges {
        let ks = if local {
            start.saturating_sub(radius)
        } else {
            0
        };
        let ke = if local {
            (start + n + radius).min(s)
        } else {
            s
        };
        let nk = ke - ks;
        let m = groups * n;
        let (q_ptr, q_ab): (&[f32], usize) = if start == 0 && n == s {
            (q, groups * s * d)
        } else {
            gather_q_tile(q, &mut q_tile, b, nkv, groups, s, d, start, n, threads);
            (&q_tile[..g_kv * m * d], m * d)
        };
        let t_qk = Instant::now();
        let mut scores = hw.mm(g_kv, m, nk, d, q_ab, d, s * d, d, true, q_ptr, &k[ks * d..])?;
        hw.add_category("attn_qk", t_qk.elapsed().as_secs_f64());
        let blocked_by_batch: Vec<Vec<usize>> = padding
            .iter()
            .map(|pad| {
                pad.iter()
                    .copied()
                    .filter(|&p| p >= ks && p < ke)
                    .map(|p| p - ks)
                    .collect()
            })
            .collect();
        let t_sm = Instant::now();
        let scores_p = scores.as_mut_ptr() as usize;
        let rows = g_kv * m;
        if local {
            for_ranges(threads, rows, 4, |rlo, rhi| {
                for row_i in rlo..rhi {
                    let gi = row_i / m;
                    let qi = (row_i % m) % n;
                    let bi = gi / nkv;
                    let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                    let qpos = start + qi;
                    let lo = qpos.saturating_sub(radius).saturating_sub(ks);
                    let hi = (qpos + radius + 1).min(ke) - ks;
                    softmax_window(row, lo, hi, &blocked_by_batch[bi]);
                }
            });
        } else {
            let blocked = blocked_by_batch.iter().any(|p| !p.is_empty());
            for_ranges(threads, rows, 4, |rlo, rhi| {
                for row_i in rlo..rhi {
                    let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                    if blocked {
                        let bi = (row_i / m) / nkv;
                        softmax_window(row, 0, nk, &blocked_by_batch[bi]);
                    } else {
                        crate::cpu::kernels::softmax_inplace(row);
                    }
                }
            });
        }
        hw.add_category("attn_softmax", t_sm.elapsed().as_secs_f64());
        let t_pv = Instant::now();
        let y = hw.mm(
            g_kv,
            m,
            d,
            nk,
            m * nk,
            nk,
            s * d,
            d,
            false,
            &scores,
            &v[ks * d..],
        )?;
        hw.add_category("attn_pv", t_pv.elapsed().as_secs_f64());
        hw.recycle(scores);
        {
            let y_p = y.as_ptr() as usize;
            let out_p = out.as_mut_ptr() as usize;
            for_ranges(threads, b * n, 8, |lo, hi| {
                for idx in lo..hi {
                    let bi = idx / n;
                    let qi = idx % n;
                    for kvh in 0..nkv {
                        for grp in 0..groups {
                            let qh = kvh * groups + grp;
                            let src = (((bi * nkv + kvh) * groups + grp) * n + qi) * d;
                            let dst = (bi * s + start + qi) * (nq * d) + qh * d;
                            let src = unsafe { row_ref(y_p as *const f32, src / d, d) };
                            let dst = unsafe { row_mut(out_p as *mut f32, dst / d, d) };
                            dst.copy_from_slice(src);
                        }
                    }
                }
            });
        }
        hw.recycle(y);
    }
    hw.recycle(q_tile);
    if !q_scaled.is_empty() {
        hw.recycle(q_scaled);
    }
    Ok(out)
}

fn gather_q_tile<T: Copy>(
    q: &[T],
    dst: &mut [T],
    b: usize,
    nkv: usize,
    groups: usize,
    s: usize,
    d: usize,
    start: usize,
    n: usize,
    nthreads: usize,
) {
    let nq = nkv * groups;
    let rows = b * nkv * groups * n;
    let q_p = q.as_ptr() as usize;
    let d_p = dst.as_mut_ptr() as usize;
    for_ranges(nthreads, rows, 16, |lo, hi| {
        for idx in lo..hi {
            let qi = idx % n;
            let rest = idx / n;
            let grp = rest % groups;
            let rest = rest / groups;
            let kvh = rest % nkv;
            let bi = rest / nkv;
            let qh = kvh * groups + grp;
            let src = ((bi * nq + qh) * s + start + qi) * d;
            let off = (((bi * nkv + kvh) * groups + grp) * n + qi) * d;
            let src = unsafe { row_ref(q_p as *const T, src / d, d) };
            let dst = unsafe { row_mut(d_p as *mut T, off / d, d) };
            dst.copy_from_slice(src);
        }
    });
}

impl OnednnCpu {
    /// Attention PV: `probs[g,M,K] @ V[g,K,D]`. Query `M` uses the same
    /// shape+ISA pad as packed linear (see [`crate::cpu::onednn::IsaStrategy`]).
    /// `nk` is the live KV span.
    fn attn_pv_inner(
        &self,
        g: usize,
        m: usize,
        d: usize,
        nk: usize,
        probs: &[f32],
        v: &[u16],
        v_bb: usize,
        v_br: usize,
    ) -> AResult<Vec<f32>> {
        check_shape!(probs.len() >= g * m * nk, "pv probs shape");
        let m_pad = self.strategy.attn_pad_m(m);
        let key = |rows: usize| Key {
            kind: 0,
            g,
            m: rows,
            n: d,
            k: nk,
            ab: rows * nk,
            ar: nk,
            bb: v_bb,
            br: v_br,
            trans: false,
            param: 0,
            sum: false,
            weight: 0,
            dt: MatDt::Bf16,
        };
        if m_pad == m {
            let mut y = self.buffer(crate::checked_size("activation elements", &[g, m, d])?)?;
            self.run_ex(key(m), probs, None, None, Some(v), None, &mut y, None)?;
            return Ok(y);
        }
        let mut a_pad =
            self.buffer(crate::checked_size("activation elements", &[g, m_pad, nk])?)?;
        for gi in 0..g {
            let src = gi * m * nk;
            let dst = gi * m_pad * nk;
            a_pad[dst..dst + m * nk].copy_from_slice(&probs[src..src + m * nk]);
            a_pad[dst + m * nk..dst + m_pad * nk].fill(0.0);
        }
        let mut dst = self.buffer(crate::checked_size("activation elements", &[g, m_pad, d])?)?;
        let rc = self.run_ex(
            key(m_pad),
            &a_pad,
            None,
            None,
            Some(v),
            None,
            &mut dst,
            None,
        );
        self.recycle(a_pad);
        rc?;
        let mut y = self.buffer(crate::checked_size("activation elements", &[g, m, d])?)?;
        for gi in 0..g {
            let src = gi * m_pad * d;
            let dst_o = gi * m * d;
            y[dst_o..dst_o + m * d].copy_from_slice(&dst[src..src + m * d]);
        }
        self.recycle(dst);
        Ok(y)
    }

    pub(crate) fn attention_gqa_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        attention_gqa_f32(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, scale, padding,
        )
    }

    pub(crate) fn attention_gqa(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        attention_gqa_bf16(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, scale, padding,
        )
    }

    pub(crate) fn attention_gqa_src(
        &self,
        q: &[u16],
        k: &[u16],
        v: &[u16],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        if (scale - 1.0).abs() > 0.0 {
            let up = |x: &[u16]| -> Vec<f32> {
                x.iter()
                    .map(|&b| f32::from_bits((b as u32) << 16))
                    .collect()
            };
            return attention_gqa_bf16(
                self,
                &up(q),
                &up(k),
                &up(v),
                b,
                s,
                nq,
                nkv,
                d,
                tile,
                local,
                radius,
                scale,
                padding,
            );
        }
        attention_gqa_from_bf16(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, padding,
        )
    }

    pub(crate) fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nh: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        attention_fp32(self, q, k, v, b, s, nh, d, tile, local, radius, padding)
    }
}
