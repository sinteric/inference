//! BF16 graph ([`Precision::Bf16`], the default where the CPU has BF16
//! matmul kernels): matmuls are BF16×BF16 with FP32 accumulation; RMSNorm,
//! softmax, RoPE, the residual stream, pooling and L2 stay in FP32. Each
//! normalization writes the BF16 rows the next projection reads, and on AMX
//! the residual rows are padded to the backend's row tile.

use crate::cpu::kernels::{
    add_rmsnorm, f32_slice_to_bf16_nthreads, for_ranges, for_row_tiles, gelu_tanh_mul,
    rmsnorm_rope_to_bf16, rmsnorm_to_bf16, scale_inplace, RowsMut,
};
use crate::cpu::onednn::Attention;
use crate::embeddinggemma2::config::{
    head_dim, is_global, n_kv, ATTN_SCALE, EMBED_DIM, HIDDEN, N_LAYERS, N_QUERY_HEADS, PLE_WIDTH,
    RMS_EPS, SLIDING_WINDOW,
};
use crate::embeddinggemma2::frontend::Batch;
use crate::{check_shape, Result, ResultContext as Context};
use std::time::Instant;

use super::{l2_rows, mean_pool, Dense, Model, Precision, Splice};

impl Model {
    fn linear_src(&self, x: &[u16], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len().is_multiple_of(l.weight.k()), "linear matrix {name}");
        let mut y = self.backend.buffer(crate::checked_size(
            "activation elements",
            &[x.len() / l.weight.k(), l.weight.n()],
        )?)?;
        self.backend.linear_bf16_src(x, &l.weight, &mut y, false)?;
        Ok(y)
    }

    fn to_bf16(&self, x: &[f32]) -> Result<Vec<u16>> {
        let mut y = self.backend.buffer_u16(x.len())?;
        f32_slice_to_bf16_nthreads(x, &mut y, self.backend.threads())?;
        Ok(y)
    }

    fn rms(&self, x: &[f32], name: &str, width: usize) -> Result<Vec<f32>> {
        let scale = self
            .norms
            .get(name)
            .with_context(|| format!("norm {name}"))?;
        check_shape!(
            x.len().is_multiple_of(width) && scale.len() == width,
            "rms {name}"
        );
        self.backend
            .rmsnorm(x.len() / width, width, x, Some(scale), RMS_EPS)
    }

    fn rms_bf16(&self, x: &[f32], name: &str, width: usize) -> Result<Vec<u16>> {
        let scale = self
            .norms
            .get(name)
            .with_context(|| format!("norm {name}"))?;
        check_shape!(
            x.len().is_multiple_of(width) && scale.len() == width,
            "rms {name}"
        );
        let mut y = self.backend.buffer_u16(x.len())?;
        let t = Instant::now();
        rmsnorm_to_bf16(
            x,
            width,
            Some(scale),
            RMS_EPS,
            &mut y,
            self.backend.threads(),
        )?;
        self.backend.add_category("glue", t.elapsed().as_secs_f64());
        Ok(y)
    }

    /// Residual rows rounded to the AMX tile / 4-tile block (shape+ISA
    /// only, including `M<16→16`). Pad rows are zeroed so glue and GEMM
    /// never see stale pool data.
    fn work_rows(&self, live: usize) -> usize {
        self.backend.residual_rows(live)
    }

    /// `[b, s, nh * d]` rows split into `[b, nh, s, d]` heads.
    fn to_heads(&self, src: &[f32], b: usize, s: usize, nh: usize, d: usize) -> Result<Vec<f32>> {
        let stride = nh * d;
        let mut out = self
            .backend
            .buffer(crate::checked_size("activation elements", &[b, nh, s, d])?)?;
        let ntok = b * s;
        let rows = RowsMut::new(&mut out, d);
        for_ranges(self.backend.threads(), ntok, 8, |lo, hi| {
            for idx in lo..hi {
                let bi = idx / s;
                let p = idx % s;
                for h in 0..nh {
                    let so = (bi * s + p) * stride + h * d;
                    // SAFETY: token `(bi, p)` alone writes row `(bi * nh + h)
                    // * s + p` for head `h`, and `for_ranges` hands each token
                    // to one call.
                    let dst = unsafe { rows.row((bi * nh + h) * s + p) };
                    dst.copy_from_slice(&src[so..so + d]);
                }
            }
        });
        Ok(out)
    }

    fn split_qkv_heads(
        &self,
        fused: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let qn = nq * d;
        let kn = nkv * d;
        let stride = qn + kn + kn;
        check_shape!(
            fused.len() >= b * s * stride && fused.len().is_multiple_of(stride),
            "fused qkv width"
        );
        // Projection rows include the residual pad; keys and queries must
        // contain only live tokens so pad rows cannot enter attention.
        let fused = &fused[..b * s * stride];
        let mut q = self
            .backend
            .buffer(crate::checked_size("activation elements", &[b, nq, s, d])?)?;
        let mut k = self
            .backend
            .buffer(crate::checked_size("activation elements", &[b, nkv, s, d])?)?;
        let mut v = self
            .backend
            .buffer(crate::checked_size("activation elements", &[b, nkv, s, d])?)?;
        let ntok = b * s;
        let (qr, kr, vr) = (
            RowsMut::new(&mut q, d),
            RowsMut::new(&mut k, d),
            RowsMut::new(&mut v, d),
        );
        for_ranges(self.backend.threads(), ntok, 8, |lo, hi| {
            for idx in lo..hi {
                let bi = idx / s;
                let p = idx % s;
                let src = &fused[(bi * s + p) * stride..][..stride];
                // Token `(bi, p)` alone writes row `(bi * heads + h) * s + p`
                // of each output for head `h`, and `for_ranges` hands each
                // token to one call.
                for h in 0..nq {
                    // SAFETY: this token's own row of Q for head `h` (above).
                    let dst = unsafe { qr.row((bi * nq + h) * s + p) };
                    dst.copy_from_slice(&src[h * d..][..d]);
                }
                for h in 0..nkv {
                    let r = (bi * nkv + h) * s + p;
                    // SAFETY: this token's own row of K for head `h` (above).
                    unsafe { kr.row(r) }.copy_from_slice(&src[qn + h * d..][..d]);
                    // SAFETY: this token's own row of V for head `h` (above).
                    unsafe { vr.row(r) }.copy_from_slice(&src[qn + kn + h * d..][..d]);
                }
            }
        });
        Ok((q, k, v))
    }

    pub(super) fn forward_dense(&self, batch: &Batch, splices: &[Splice]) -> Result<Vec<Vec<f32>>> {
        let b = batch.input_ids.len();
        let s = batch.input_ids[0].len();
        check_shape!(s > 0 && s <= self.max, "sequence length");
        check_shape!(
            batch.attention_mask.len() == b
                && batch
                    .input_ids
                    .iter()
                    .zip(&batch.attention_mask)
                    .all(|(ids, m)| ids.len() == s && m.len() == s),
            "dense batch"
        );
        if self.precision == Precision::Fp32 {
            return self.forward_dense_fp32(batch, b, s, splices);
        }
        self.backend.begin_shape(b, s);
        let (tokens, m_pad) = self.reserve_dense(b, s)?;

        let mut x = self.embed(batch, b, s, m_pad, splices)?;
        let mut ple_raw = self.linear(&x, "language_model.ple.per_layer_model_projection")?;
        let scale = (HIDDEN as f32).powf(-0.5);
        scale_inplace(&mut ple_raw, scale, self.backend.threads());
        let ple = self.rms(
            &ple_raw,
            "language_model.ple.per_layer_projection_norm.weight",
            PLE_WIDTH,
        )?;
        self.backend.recycle(ple_raw);

        let padding: Vec<Vec<usize>> = batch
            .attention_mask
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (v == 0).then_some(i))
                    .collect()
            })
            .collect();
        let dense = Dense {
            b,
            s,
            tokens,
            m_pad,
            padding: &padding,
        };
        for layer in 0..N_LAYERS {
            let base = format!("language_model.layers.{layer}");
            let o = self.self_attention(layer, &base, &x, &dense)?;
            check_shape!(x.len() == o.len(), "attn residual");
            self.add_norm(
                &mut x,
                &o,
                &format!("{base}.post_attention_layernorm.weight"),
                1.0,
            )?;
            self.backend.recycle(o);
            let down = self.mlp(&base, &x)?;
            self.add_norm(
                &mut x,
                &down,
                &format!("{base}.post_feedforward_layernorm.weight"),
                1.0,
            )?;
            self.backend.recycle(down);
            let proj = self.per_layer_input(layer, &base, &x, &ple, tokens)?;
            let norm = format!("{base}.ple_block.post_per_layer_input_norm.weight");
            self.add_norm(&mut x, &proj, &norm, self.scalars[layer])?;
            self.backend.recycle(proj);
        }
        self.backend.recycle(ple);

        let x_bf = self.rms_bf16(&x, "language_model.norm.weight", HIDDEN)?;
        self.backend.recycle(x);
        let proj = self.linear_src(&x_bf, "language_model.embedding_projection")?;
        self.backend.recycle_u16(x_bf);
        // Pool only the live prefix, using the original attention mask.
        let pooled = mean_pool(&proj[..tokens * EMBED_DIM], batch, b, s, EMBED_DIM);
        self.backend.recycle(proj);
        Ok(l2_rows(pooled, EMBED_DIM))
    }

    /// Check that a `b x s` forward fits the workspace limit, then reserve
    /// every buffer it takes before allocating any: the token count and the
    /// padded row count.
    fn reserve_dense(&self, b: usize, s: usize) -> Result<(usize, usize)> {
        let tokens = crate::checked_size("EG2 tokens", &[b, s])?;
        crate::checked_size(
            "EG2 workspace",
            &[
                tokens
                    .checked_add(64)
                    .ok_or(crate::Error::size_overflow("padded rows"))?,
                N_LAYERS,
                PLE_WIDTH,
                4,
            ],
        )?;
        let m_pad = self.work_rows(tokens);
        // These are the reservation counts below; sum before allocating any.
        let row_bytes = HIDDEN * 4 * 4
            + 2048 * 3 * 4
            + 3072 * 2 * 4
            + PLE_WIDTH * 3 * 4
            + N_LAYERS * PLE_WIDTH * 2 * 4
            + HIDDEN * 3 * 2
            + HIDDEN * 2 * 4;
        let planned = crate::checked_size("EG2 reserved activations", &[m_pad, row_bytes])?
            .checked_add(crate::checked_size(
                "EG2 reserved BF16 heads",
                &[tokens, N_QUERY_HEADS, 512, 5, 2],
            )?)
            .ok_or(crate::Error::size_overflow("EG2 workspace"))?;
        self.backend.validate_workspace(planned)?;
        let elements = |dims: &[usize]| crate::checked_size("activation elements", dims);
        self.backend.reserve(elements(&[m_pad, HIDDEN])?, 4)?;
        self.backend.reserve(elements(&[m_pad, 2048])?, 3)?;
        self.backend.reserve(elements(&[m_pad, 3072])?, 2)?;
        self.backend.reserve(elements(&[m_pad, PLE_WIDTH])?, 3)?;
        self.backend
            .reserve(elements(&[m_pad, N_LAYERS, PLE_WIDTH])?, 2)?;
        self.backend.reserve_u16(elements(&[m_pad, HIDDEN])?, 3)?;
        self.backend
            .reserve_u16(elements(&[tokens, N_QUERY_HEADS, 512])?, 3)?;
        self.backend
            .reserve_u16(elements(&[tokens, N_QUERY_HEADS, 512])?, 2)?;
        self.backend.reserve(elements(&[m_pad, HIDDEN])?, 2)?;
        Ok((tokens, m_pad))
    }

    /// `x += rmsnorm(delta) * scale` row by row, `norm` naming the norm's
    /// weight; timed as glue.
    fn add_norm(&self, x: &mut [f32], delta: &[f32], norm: &str, scale: f32) -> Result<()> {
        let weight = self.norms.get(norm).with_context(|| norm.to_string())?;
        let t_glue = Instant::now();
        add_rmsnorm(
            x,
            delta,
            HIDDEN,
            Some(weight),
            scale,
            RMS_EPS,
            self.backend.threads(),
        )?;
        self.backend
            .add_category("glue", t_glue.elapsed().as_secs_f64());
        Ok(())
    }

    /// Layer `layer`'s attention over `x`, through its output projection.
    fn self_attention(
        &self,
        layer: usize,
        base: &str,
        x: &[f32],
        dense: &Dense<'_>,
    ) -> Result<Vec<f32>> {
        let Dense {
            b,
            s,
            m_pad,
            padding,
            ..
        } = *dense;
        let global = is_global(layer);
        let d = head_dim(layer);
        let nkv = n_kv(layer);
        let nq = N_QUERY_HEADS;

        let [q, k, v] = self.qkv_heads(layer, base, x, dense)?;
        let q_scale = self
            .norms
            .get(&format!("{base}.self_attn.q_norm.weight"))
            .with_context(|| format!("{base}.self_attn.q_norm.weight"))?;
        let k_scale = self
            .norms
            .get(&format!("{base}.self_attn.k_norm.weight"))
            .with_context(|| format!("{base}.self_attn.k_norm.weight"))?;
        let rope = if global {
            &self.global_rope
        } else {
            &self.local_rope
        };
        let mut q_bf = self.backend.buffer_u16(q.len())?;
        let mut k_bf = self.backend.buffer_u16(k.len())?;
        let mut v_bf = self.backend.buffer_u16(v.len())?;
        let threads = self.backend.threads();
        // Row-major [b,nh,s,d] so each pass streams contiguous head rows.
        let t_glue = Instant::now();
        let rotation = Some((&rope.cos[..], &rope.sin[..]));
        rmsnorm_rope_to_bf16(
            &q,
            &mut q_bf,
            b,
            s,
            nq,
            d,
            Some(q_scale),
            RMS_EPS,
            rotation,
            threads,
        )?;
        rmsnorm_rope_to_bf16(
            &k,
            &mut k_bf,
            b,
            s,
            nkv,
            d,
            Some(k_scale),
            RMS_EPS,
            rotation,
            threads,
        )?;
        rmsnorm_to_bf16(&v, d, None, RMS_EPS, &mut v_bf, threads)?;
        self.backend
            .add_category("glue", t_glue.elapsed().as_secs_f64());
        self.backend.recycle(q);
        self.backend.recycle(k);
        self.backend.recycle(v);

        let shape = Attention {
            batch: b,
            seq: s,
            q_heads: nq,
            kv_heads: nkv,
            head_dim: d,
            tile: self.tile,
            window: (!global).then_some(SLIDING_WINDOW),
            scale: ATTN_SCALE,
            padding,
        };
        let attn = self
            .backend
            .attention_gqa_src(&shape, &q_bf, &k_bf, &v_bf)?;
        self.backend.recycle_u16(q_bf);
        self.backend.recycle_u16(k_bf);
        self.backend.recycle_u16(v_bf);
        check_shape!(attn.len() == m_pad * nq * d, "attn residual rows");

        let o = self.linear(&attn, &format!("{base}.self_attn.o_proj"))?;
        self.backend.recycle(attn);
        Ok(o)
    }

    /// Layer `layer`'s Q, K and V heads (`[b, heads, s, d]`) from the rows
    /// `x`, through the fused projection when the weights have one.
    fn qkv_heads(
        &self,
        layer: usize,
        base: &str,
        x: &[f32],
        dense: &Dense<'_>,
    ) -> Result<[Vec<f32>; 3]> {
        let Dense { b, s, .. } = *dense;
        let (d, nkv, nq) = (head_dim(layer), n_kv(layer), N_QUERY_HEADS);
        let h_bf = self.rms_bf16(x, &format!("{base}.input_layernorm.weight"), HIDDEN)?;
        let qkv_name = format!("{base}.self_attn.qkv_proj");
        let qkv = if self.linears.contains_key(&qkv_name) {
            let fused = self.linear_src(&h_bf, &qkv_name)?;
            let (q, k, v) = self.split_qkv_heads(&fused, b, s, nq, nkv, d)?;
            self.backend.recycle(fused);
            [q, k, v]
        } else {
            let q_flat = self.linear_src(&h_bf, &format!("{base}.self_attn.q_proj"))?;
            let k_flat = self.linear_src(&h_bf, &format!("{base}.self_attn.k_proj"))?;
            let v_flat = self.linear_src(&h_bf, &format!("{base}.self_attn.v_proj"))?;
            let q = self.to_heads(&q_flat, b, s, nq, d)?;
            let k = self.to_heads(&k_flat, b, s, nkv, d)?;
            let v = self.to_heads(&v_flat, b, s, nkv, d)?;
            self.backend.recycle(q_flat);
            self.backend.recycle(k_flat);
            self.backend.recycle(v_flat);
            [q, k, v]
        };
        self.backend.recycle_u16(h_bf);
        Ok(qkv)
    }

    /// The gated feed-forward over `x`, through its down projection.
    fn mlp(&self, base: &str, x: &[f32]) -> Result<Vec<f32>> {
        let pre_bf = self.rms_bf16(
            x,
            &format!("{base}.pre_feedforward_layernorm.weight"),
            HIDDEN,
        )?;
        let up = self.linear_src(&pre_bf, &format!("{base}.mlp.up_proj"))?;
        let gate_w = &self
            .linears
            .get(&format!("{base}.mlp.gate_proj"))
            .with_context(|| format!("{base}.mlp.gate_proj"))?
            .weight;
        let gated = self.backend.gated_linear_tanh_src(&pre_bf, gate_w, &up)?;
        self.backend.recycle_u16(pre_bf);
        self.backend.recycle(up);
        let down = self.linear(&gated, &format!("{base}.mlp.down_proj"))?;
        self.backend.recycle(gated);
        Ok(down)
    }

    /// Layer `layer`'s per-layer input: `x`'s gate times that layer's slice
    /// of `ple`, through its projection.
    fn per_layer_input(
        &self,
        layer: usize,
        base: &str,
        x: &[f32],
        ple: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>> {
        let res_bf = self.to_bf16(x)?;
        let gate = self.linear_src(&res_bf, &format!("{base}.ple_block.per_layer_input_gate"))?;
        self.backend.recycle_u16(res_bf);
        let mut mixed = self.backend.buffer(gate.len())?;
        let ple_off = layer * PLE_WIDTH;
        let threads = self.backend.threads();
        let mix_rows = gate.len() / PLE_WIDTH;
        let rows = RowsMut::new(&mut mixed, PLE_WIDTH);
        for_row_tiles(
            threads,
            mix_rows,
            self.backend.residual_row_tile(tokens),
            8,
            |lo, hi| {
                // SAFETY: `for_row_tiles` hands out disjoint ranges, each once.
                let out = unsafe { rows.range(lo, hi) };
                for (ti, row) in (lo..hi).zip(out.as_chunks_mut::<PLE_WIDTH>().0) {
                    let src = &gate[ti * PLE_WIDTH..(ti + 1) * PLE_WIDTH];
                    let p = &ple[ti * (N_LAYERS * PLE_WIDTH) + ple_off..][..PLE_WIDTH];
                    gelu_tanh_mul(src, p, row);
                }
            },
        )?;
        self.backend.recycle(gate);
        let proj = self.linear(&mixed, &format!("{base}.ple_block.per_layer_projection"))?;
        self.backend.recycle(mixed);
        Ok(proj)
    }
}
