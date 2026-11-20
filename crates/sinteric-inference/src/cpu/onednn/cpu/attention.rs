//! Attention on oneDNN: bidirectional, optionally local (a radius around each
//! query), with excluded key positions per sequence. QK and PV are batched
//! GEMMs over query tiles; the mask and the output scatter run on the pool.

use super::*;
use crate::cpu::kernels::{for_rows, RowsMut};

/// One attention call: the shape of Q, K and V (`[batch, heads, seq,
/// head_dim]`, head-major), which keys each query sees, and the scale on Q.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Attention<'a> {
    /// Sequences in the batch.
    pub(crate) batch: usize,
    /// Positions per sequence, the same for queries and keys.
    pub(crate) seq: usize,
    pub(crate) q_heads: usize,
    /// Key and value heads; each serves `q_heads / kv_heads` query heads.
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    /// Query rows per tile, before the backend clamps it to the shape.
    pub(crate) tile: usize,
    /// `Some(r)`: each query sees the keys within `r` positions of it.
    /// `None`: every key.
    pub(crate) window: Option<usize>,
    /// Multiplies Q before QK; 1.0 leaves Q as given.
    pub(crate) scale: f32,
    /// Per sequence, the key positions no query sees (padding).
    pub(crate) padding: &'a [Vec<usize>],
}

/// One query tile: queries `[start, start + n)` against keys `[ks, ke)`.
#[derive(Clone, Copy, Debug)]
struct Tile {
    start: usize,
    n: usize,
    ks: usize,
    ke: usize,
}

impl Tile {
    fn keys(&self) -> usize {
        self.ke - self.ks
    }
}

impl Attention<'_> {
    fn local(&self) -> bool {
        self.window.is_some()
    }

    fn radius(&self) -> usize {
        self.window.unwrap_or(0)
    }

    fn groups(&self) -> usize {
        self.q_heads / self.kv_heads
    }

    /// Validate the tensors once, before any row-parallel work.
    fn validate<T>(&self, q: &[T], k: &[T], v: &[T]) -> Result<()> {
        let (b, s, nq, nkv, d, tile) = (
            self.batch,
            self.seq,
            self.q_heads,
            self.kv_heads,
            self.head_dim,
            self.tile,
        );
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
            self.padding.len() == b && self.padding.iter().all(|r| r.iter().all(|&p| p < s)),
            "attention padding indices outside batch {b}, sequence {s}"
        );
        s.checked_add(self.radius())
            .and_then(|n| n.checked_add(1))
            .ok_or(crate::Error::size_overflow("attention radius"))?;
        crate::checked_size("attention score bytes", &[b, nq, s, s.min(tile), 4])?;
        Ok(())
    }

    /// The tile of queries `[start, start + n)` and the keys they can see.
    fn tile_at(&self, start: usize, n: usize) -> Tile {
        let (ks, ke) = match self.window {
            Some(radius) => (
                start.saturating_sub(radius),
                (start + n + radius).min(self.seq),
            ),
            None => (0, self.seq),
        };
        Tile { start, n, ks, ke }
    }

    /// Per sequence, the excluded keys inside the tile's key span, counted
    /// from its first key.
    fn blocked_in(&self, t: &Tile) -> Vec<Vec<usize>> {
        self.padding
            .iter()
            .map(|pad| {
                pad.iter()
                    .copied()
                    // Filter before subtracting: then_some eagerly evaluates
                    // p - ks and underflows in debug builds for earlier keys.
                    .filter(|&p| p >= t.ks && p < t.ke)
                    .map(|p| p - t.ks)
                    .collect()
            })
            .collect()
    }

    /// The query tiles' bounds for the grouped (GQA) kernels.
    fn query_ranges(&self, strategy: crate::cpu::onednn::IsaStrategy) -> Vec<(usize, usize)> {
        let tile = crate::cpu::onednn::clamp_attn_tile(
            self.tile,
            self.seq,
            self.local(),
            self.radius(),
            self.batch,
            self.q_heads,
        );
        strategy.attn_query_ranges(
            self.seq,
            tile,
            self.batch,
            self.q_heads,
            self.kv_heads,
            self.local(),
        )
    }
}

/// FP32 multi-head attention (`q_heads == kv_heads`, no scale): tiled QK and
/// PV GEMMs and oneDNN softmax, with a threaded mask (`+= f32::MIN`) and
/// output scatter.
pub(super) fn attention_fp32(
    hw: &OnednnCpu,
    a: &Attention,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<Vec<f32>> {
    a.validate(q, k, v)?;
    crate::check_shape!(
        a.q_heads == a.kv_heads && a.scale == 1.0,
        "multi-head attention takes one K/V head per query head and no scale"
    );
    let (b, s, nh, d) = (a.batch, a.seq, a.q_heads, a.head_dim);
    let g = b * nh;
    let threads = hw.threads.max(1) as usize;
    let mut out = hw.buffer(crate::checked_size("activation elements", &[b, s, nh, d])?)?;
    let tile = a.tile.min(s).max(1);
    for start in (0..s).step_by(tile) {
        let t = a.tile_at(start, tile.min(s - start));
        let (n, nk) = (t.n, t.keys());
        let qk = Gemm::nt(g, n, nk, d, s * d, s * d);
        let mut scores = hw.mm(qk, &q[start * d..], &k[t.ks * d..])?;
        mask_fp32(threads, a, &t, &mut scores);
        // oneDNN's softmax: a small slice of the attention time, and the
        // rounding the models' references were checked against.
        let probs = hw.softmax(g * n, nk, &scores)?;
        hw.recycle(scores);
        let pv = Gemm::nn(g, n, d, nk, n * nk, s * d);
        let y = hw.mm(pv, &probs, &v[t.ks * d..])?;
        hw.recycle(probs);
        scatter_heads(threads, a, &t, &y, &mut out);
        hw.recycle(y);
    }
    Ok(out)
}

/// Adds `f32::MIN` to every score a query may not see: keys outside its
/// window and excluded keys.
fn mask_fp32(threads: usize, a: &Attention, t: &Tile, scores: &mut [f32]) {
    let (n, nk, local, radius) = (t.n, t.keys(), a.local(), a.radius());
    for_rows(threads, 4, scores, nk, |rlo, rhi, rows| {
        for (row_i, row) in (rlo..rhi).zip(rows.chunks_exact_mut(nk)) {
            let gi = row_i / n;
            let qi = row_i % n;
            let excluded = &a.padding[gi / a.q_heads];
            if !local && excluded.is_empty() {
                continue;
            }
            let (lo, hi) = if local {
                (
                    (t.start + qi).saturating_sub(radius).saturating_sub(t.ks),
                    (t.start + qi + radius + 1).min(t.ke) - t.ks,
                )
            } else {
                (0, nk)
            };
            for score in &mut row[..lo] {
                *score += f32::MIN;
            }
            for score in &mut row[hi..] {
                *score += f32::MIN;
            }
            for &pos in excluded {
                if pos >= t.ks + lo && pos < t.ks + hi {
                    row[pos - t.ks] += f32::MIN;
                }
            }
        }
    });
}

/// BF16 GQA attention from f32 Q, K and V: scales Q, converts all three to
/// BF16, then runs [`attention_gqa_from_bf16`].
pub(super) fn attention_gqa_bf16(
    hw: &OnednnCpu,
    a: &Attention,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<Vec<f32>> {
    a.validate(q, k, v)?;
    let threads = hw.threads.max(1) as usize;
    let q_scale = if (a.scale - 1.0).abs() > 0.0 {
        let mut qs = hw.buffer(q.len())?;
        for_rows(threads, 4096, &mut qs, 1, |lo, hi, qs| {
            for (dst, &src) in qs.iter_mut().zip(&q[lo..hi]) {
                *dst = src * a.scale;
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
    let out = attention_gqa_from_bf16(hw, a, &q_bf, &k_bf, &v_bf);
    hw.recycle_u16(q_bf);
    hw.recycle_u16(k_bf);
    hw.recycle_u16(v_bf);
    out
}

/// Main's tiled QK → softmax → PV (oneshot oneDNN, f32 P, AMX short-M PV
/// pad) on BF16 Q, K and V, which it takes as already scaled.
pub(super) fn attention_gqa_from_bf16(
    hw: &OnednnCpu,
    a: &Attention,
    q_bf: &[u16],
    k_bf: &[u16],
    v_bf: &[u16],
) -> Result<Vec<f32>> {
    a.validate(q_bf, k_bf, v_bf)?;
    let (s, nq, d) = (a.seq, a.q_heads, a.head_dim);
    let threads = hw.threads.max(1) as usize;
    let g_kv = a.batch * a.kv_heads;
    let live_rows = a.batch * s;
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
    let q_ranges = a.query_ranges(hw.strategy);
    let mut q_tile = query_tile_buffer(&q_ranges, a, |n| hw.buffer_u16(n))?;
    for (start, n) in q_ranges {
        let t = a.tile_at(start, n);
        let (nk, m) = (t.keys(), a.groups() * n);
        let (q_ptr, q_ab): (&[u16], usize) = if start == 0 && n == s {
            (q_bf, a.groups() * s * d)
        } else {
            gather_q_tile(q_bf, &mut q_tile, a, &t, threads);
            (&q_tile[..g_kv * m * d], m * d)
        };
        let t_qk = Instant::now();
        let qk = Gemm::nt(g_kv, m, nk, d, q_ab, s * d);
        let scores = hw.mm_bf16(qk, Operand::Bf16(q_ptr), &k_bf[t.ks * d..])?;
        hw.add_category("attn_qk", t_qk.elapsed().as_secs_f64());
        let t_sm = Instant::now();
        let probs = softmax_bf16_tile(hw, threads, a, &t, scores)?;
        hw.add_category("attn_softmax", t_sm.elapsed().as_secs_f64());
        let t_pv = Instant::now();
        let y = pv_bf16(hw, a, &t, &probs, v_bf)?;
        hw.add_category("attn_pv", t_pv.elapsed().as_secs_f64());
        hw.recycle(probs);
        scatter_heads(threads, a, &t, &y, &mut out);
        hw.recycle(y);
    }
    hw.recycle_u16(q_tile);
    Ok(out)
}

/// A buffer for the largest gathered query tile, or none when one tile
/// covers the whole sequence.
fn query_tile_buffer<T>(
    q_ranges: &[(usize, usize)],
    a: &Attention,
    buffer: impl Fn(usize) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    let q_max = q_ranges.iter().map(|&(_, n)| n).max().unwrap_or(1);
    if q_max < a.seq {
        buffer(crate::checked_size(
            "activation elements",
            &[a.batch * a.kv_heads, a.groups(), q_max, a.head_dim],
        )?)
    } else {
        Ok(Vec::new())
    }
}

/// Softmax of one tile's BF16-path scores: per row inside the window when
/// local; otherwise excluded keys at −∞, then oneDNN's softmax for large
/// tiles and the pool's for small ones.
fn softmax_bf16_tile(
    hw: &OnednnCpu,
    threads: usize,
    a: &Attention,
    t: &Tile,
    mut scores: Vec<f32>,
) -> Result<Vec<f32>> {
    let blocked = a.blocked_in(t);
    let (nk, m) = (t.keys(), a.groups() * t.n);
    if a.local() {
        softmax_local(threads, a, t, &blocked, &mut scores);
        return Ok(scores);
    }
    if blocked.iter().any(|p| !p.is_empty()) {
        for_rows(threads, 4, &mut scores, nk, |rlo, rhi, rows| {
            for (row_i, row) in (rlo..rhi).zip(rows.chunks_exact_mut(nk)) {
                let bi = (row_i / m) / a.kv_heads;
                for &pos in &blocked[bi] {
                    row[pos] = f32::NEG_INFINITY;
                }
            }
        });
    }
    let rows = a.batch * a.kv_heads * m;
    if rows >= 512 && nk >= 256 {
        let probs = hw.softmax(rows, nk, &scores)?;
        hw.recycle(scores);
        Ok(probs)
    } else {
        softmax_rows_nthreads(&mut scores, nk, threads)?;
        Ok(scores)
    }
}

/// Softmax of each score row over the keys inside its query's window, with
/// the excluded keys at −∞ and everything outside the window 0.
fn softmax_local(
    threads: usize,
    a: &Attention,
    t: &Tile,
    blocked: &[Vec<usize>],
    scores: &mut [f32],
) {
    let (nk, m, radius) = (t.keys(), a.groups() * t.n, a.radius());
    for_rows(threads, 4, scores, nk, |rlo, rhi, rows| {
        for (row_i, row) in (rlo..rhi).zip(rows.chunks_exact_mut(nk)) {
            let gi = row_i / m;
            let qi = (row_i % m) % t.n;
            let bi = gi / a.kv_heads;
            let qpos = t.start + qi;
            let lo = qpos.saturating_sub(radius).saturating_sub(t.ks);
            let hi = (qpos + radius + 1).min(t.ke) - t.ks;
            softmax_window(row, lo, hi, &blocked[bi]);
        }
    });
}

/// PV for one tile of the BF16 path: probabilities in f32 (converted on
/// the way in) against BF16 V.
fn pv_bf16(
    hw: &OnednnCpu,
    a: &Attention,
    t: &Tile,
    probs: &[f32],
    v_bf: &[u16],
) -> Result<Vec<f32>> {
    let (s, d) = (a.seq, a.head_dim);
    let (nk, m, g_kv) = (t.keys(), a.groups() * t.n, a.batch * a.kv_heads);
    let pv = Gemm::nn(g_kv, m, d, nk, m * nk, s * d);
    hw.mm_bf16(pv, Operand::F32(probs), &v_bf[t.ks * d..])
}

/// FP32 GQA attention: per KV head, the `groups` query heads that share it
/// form one `groups * n`-row f32 matmul against K and V (no `repeat_kv`).
/// Output is exactly `[b * s, nq * d]`, token-major, with no residual pad.
pub(super) fn attention_gqa_f32(
    hw: &OnednnCpu,
    a: &Attention,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> Result<Vec<f32>> {
    a.validate(q, k, v)?;
    let (s, nq, d) = (a.seq, a.q_heads, a.head_dim);
    let threads = hw.threads.max(1) as usize;
    let g_kv = a.batch * a.kv_heads;
    let q_scaled = if (a.scale - 1.0).abs() > 0.0 {
        let mut qs = hw.buffer(q.len())?;
        for (dst, &src) in qs.iter_mut().zip(q) {
            *dst = src * a.scale;
        }
        qs
    } else {
        Vec::new()
    };
    let q: &[f32] = if q_scaled.is_empty() { q } else { &q_scaled };
    let mut out = hw.buffer(crate::checked_size(
        "activation elements",
        &[a.batch * s, nq, d],
    )?)?;
    // f32 runs on AVX-512 FMA on every host, so AMX tail folding never applies.
    let q_ranges = a.query_ranges(crate::cpu::onednn::IsaStrategy::Avx512);
    let mut q_tile = query_tile_buffer(&q_ranges, a, |n| hw.buffer(n))?;
    for (start, n) in q_ranges {
        let t = a.tile_at(start, n);
        let (nk, m) = (t.keys(), a.groups() * n);
        let (q_ptr, q_ab): (&[f32], usize) = if start == 0 && n == s {
            (q, a.groups() * s * d)
        } else {
            gather_q_tile(q, &mut q_tile, a, &t, threads);
            (&q_tile[..g_kv * m * d], m * d)
        };
        let t_qk = Instant::now();
        let mut scores = hw.mm(Gemm::nt(g_kv, m, nk, d, q_ab, s * d), q_ptr, &k[t.ks * d..])?;
        hw.add_category("attn_qk", t_qk.elapsed().as_secs_f64());
        let t_sm = Instant::now();
        softmax_f32_tile(threads, a, &t, &mut scores);
        hw.add_category("attn_softmax", t_sm.elapsed().as_secs_f64());
        let t_pv = Instant::now();
        let y = hw.mm(
            Gemm::nn(g_kv, m, d, nk, m * nk, s * d),
            &scores,
            &v[t.ks * d..],
        )?;
        hw.add_category("attn_pv", t_pv.elapsed().as_secs_f64());
        hw.recycle(scores);
        scatter_heads(threads, a, &t, &y, &mut out);
        hw.recycle(y);
    }
    hw.recycle(q_tile);
    if !q_scaled.is_empty() {
        hw.recycle(q_scaled);
    }
    Ok(out)
}

/// Softmax of one tile's FP32-path scores, on the pool: inside each query's
/// window when local, else over every key with the excluded ones at −∞.
fn softmax_f32_tile(threads: usize, a: &Attention, t: &Tile, scores: &mut [f32]) {
    let blocked = a.blocked_in(t);
    if a.local() {
        softmax_local(threads, a, t, &blocked, scores);
        return;
    }
    let (nk, m) = (t.keys(), a.groups() * t.n);
    let any_blocked = blocked.iter().any(|p| !p.is_empty());
    for_rows(threads, 4, scores, nk, |rlo, rhi, rows| {
        for (row_i, row) in (rlo..rhi).zip(rows.chunks_exact_mut(nk)) {
            if any_blocked {
                let bi = (row_i / m) / a.kv_heads;
                softmax_window(row, 0, nk, &blocked[bi]);
            } else {
                crate::cpu::kernels::softmax_inplace(row);
            }
        }
    });
}

/// Copies the tile's per-head results `y` (`[b, kv_heads, groups, n, d]`)
/// into `out`'s token-major rows (`[b, s, q_heads, d]`).
fn scatter_heads(threads: usize, a: &Attention, t: &Tile, y: &[f32], out: &mut [f32]) {
    let (s, nq, nkv, d, n, groups) = (a.seq, a.q_heads, a.kv_heads, a.head_dim, t.n, a.groups());
    let out = RowsMut::new(out, d);
    for_ranges(threads, a.batch * n, 8, |lo, hi| {
        for idx in lo..hi {
            let bi = idx / n;
            let qi = idx % n;
            for kvh in 0..nkv {
                for grp in 0..groups {
                    let qh = kvh * groups + grp;
                    let src = (((bi * nkv + kvh) * groups + grp) * n + qi) * d;
                    // SAFETY: query `(bi, qi)` alone writes row
                    // `(bi * s + start + qi) * nq + qh`, one per head `qh`,
                    // and `for_ranges` hands each query to one call.
                    let dst = unsafe { out.row((bi * s + t.start + qi) * nq + qh) };
                    dst.copy_from_slice(&y[src..src + d]);
                }
            }
        }
    });
}

/// Gathers the tile's queries into `dst` as `[b, kv_heads, groups, n, d]`,
/// so the query heads that share a K/V head form one matmul.
fn gather_q_tile<T: Copy + Send + Sync>(
    q: &[T],
    dst: &mut [T],
    a: &Attention,
    t: &Tile,
    nthreads: usize,
) {
    let (s, nq, nkv, d, n, groups) = (a.seq, a.q_heads, a.kv_heads, a.head_dim, t.n, a.groups());
    let rows = a.batch * nkv * groups * n;
    // Row `idx` of `dst` is `((bi * nkv + kvh) * groups + grp) * n + qi`.
    for_rows(nthreads, 16, &mut dst[..rows * d], d, |lo, hi, dst| {
        for (idx, dst) in (lo..hi).zip(dst.chunks_exact_mut(d)) {
            let qi = idx % n;
            let rest = idx / n;
            let grp = rest % groups;
            let rest = rest / groups;
            let kvh = rest % nkv;
            let bi = rest / nkv;
            let qh = kvh * groups + grp;
            let src = ((bi * nq + qh) * s + t.start + qi) * d;
            dst.copy_from_slice(&q[src..src + d]);
        }
    });
}

impl OnednnCpu {
    /// FP32 grouped-query attention ([`attention_gqa_f32`]).
    pub(crate) fn attention_gqa_f32(
        &self,
        a: &Attention,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<Vec<f32>> {
        attention_gqa_f32(self, a, q, k, v)
    }

    /// BF16 grouped-query attention from f32 Q, K and V
    /// ([`attention_gqa_bf16`]).
    pub(crate) fn attention_gqa(
        &self,
        a: &Attention,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<Vec<f32>> {
        attention_gqa_bf16(self, a, q, k, v)
    }

    /// BF16 grouped-query attention from BF16 Q, K and V. A scale other than
    /// 1.0 widens them to f32 and takes the [`attention_gqa_bf16`] path, which
    /// scales Q before rounding it back.
    pub(crate) fn attention_gqa_src(
        &self,
        a: &Attention,
        q: &[u16],
        k: &[u16],
        v: &[u16],
    ) -> Result<Vec<f32>> {
        if (a.scale - 1.0).abs() > 0.0 {
            let up = |x: &[u16]| -> Vec<f32> {
                x.iter()
                    .map(|&b| f32::from_bits((b as u32) << 16))
                    .collect()
            };
            return attention_gqa_bf16(self, a, &up(q), &up(k), &up(v));
        }
        attention_gqa_from_bf16(self, a, q, k, v)
    }

    /// FP32 multi-head attention ([`attention_fp32`]).
    pub(crate) fn attention(
        &self,
        a: &Attention,
        q: &[f32],
        k: &[f32],
        v: &[f32],
    ) -> Result<Vec<f32>> {
        attention_fp32(self, a, q, k, v)
    }
}
