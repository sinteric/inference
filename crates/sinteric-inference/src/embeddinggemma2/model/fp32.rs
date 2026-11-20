//! FP32 graph: f32 activations into every matmul, against the BF16
//! checkpoint weights widened exactly (`Hardware::pack_bf16_f32`, BF16
//! storage on oneDNN). Same operators and order as the BF16 graph, but
//! nothing is converted to BF16, so the fusions differ:
//!
//! - one pass splits the fused QKV projection into normalized, rotated
//!   head-major Q/K/V (the BF16 graph copies, then normalizes into BF16);
//! - residual updates also write the next projection's normalized input;
//! - PLE inputs are normalized once, layer-major, so each layer's
//!   `gelu_tanh(gate) * per_layer_input` is a matmul post-op;
//! - rows are exactly the live tokens (no AMX residual padding);
//! - pooling runs before `embedding_projection` (linear, no bias), so the
//!   projection sees `b` rows instead of `b * s`.

use crate::cpu::kernels::{
    add_rmsnorm, add_rmsnorm_then_norm, qkv_split_norm_rope, scaled_rmsnorm_chunks,
};
use crate::{check_shape, Result, ResultContext as Context};
use std::time::Instant;

use super::{l2_rows, mean_pool, Model};
use crate::embeddinggemma2::config::{
    head_dim, is_global, n_kv, ATTN_SCALE, EMBED_DIM, HIDDEN, INTERMEDIATE, N_LAYERS,
    N_QUERY_HEADS, PLE_WIDTH, RMS_EPS, SLIDING_WINDOW,
};
use crate::embeddinggemma2::frontend::Batch;

impl Model {
    fn norm(&self, name: &str) -> Result<&[f32]> {
        self.norms
            .get(name)
            .map(Vec::as_slice)
            .with_context(|| format!("norm {name}"))
    }

    fn linear_f32(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len() % l.weight.k() == 0, "linear matrix {name}");
        let mut y = self.backend.buffer(crate::checked_size(
            "activation elements",
            &[x.len() / l.weight.k(), l.weight.n()],
        )?)?;
        self.backend.linear(x, &l.weight, &mut y, false)?;
        Ok(y)
    }

    /// `gelu_tanh(x @ W) * gate` as one matmul with post-ops.
    fn gated_f32(&self, x: &[f32], name: &str, gate: &[f32]) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        self.backend.gated_linear_tanh(x, &l.weight, gate)
    }

    fn glue<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let t = Instant::now();
        let out = f();
        self.backend.add_category("glue", t.elapsed().as_secs_f64());
        out
    }

    pub(super) fn forward_dense_fp32(
        &self,
        batch: &Batch,
        b: usize,
        s: usize,
        splices: &[super::Splice],
    ) -> Result<Vec<Vec<f32>>> {
        self.backend.begin_shape(b, s);
        let tokens = crate::checked_size("EG2 tokens", &[b, s])?;
        let threads = self.backend.threads();
        // Live activation rows per token, summed before allocating any:
        // residual/normalized/projection rows, the widest fused QKV and
        // query heads, MLP rows, and the PLE input before and after norm.
        let row_floats = HIDDEN * 6
            + (N_QUERY_HEADS + 2) * 512
            + N_QUERY_HEADS * 512 * 2
            + INTERMEDIATE * 3
            + N_LAYERS * PLE_WIDTH * 2;
        self.backend.validate_workspace(crate::checked_size(
            "EG2 FP32 activations",
            &[tokens, row_floats, 4],
        )?)?;
        for (width, n) in [
            (HIDDEN, 6),
            (N_QUERY_HEADS * head_dim(0), 1),
            (INTERMEDIATE, 3),
            ((N_QUERY_HEADS + 2) * 512, 1),
            (N_LAYERS * PLE_WIDTH, 2),
        ] {
            self.backend.reserve(
                crate::checked_size("activation elements", &[tokens, width])?,
                n,
            )?;
        }

        let mut x = self.embed(batch, b, s, tokens, splices)?;
        let ple_raw = self.linear_f32(&x, "language_model.ple.per_layer_model_projection")?;
        let mut ple = self.backend.buffer(ple_raw.len())?;
        let ple_norm = self.norm("language_model.ple.per_layer_projection_norm.weight")?;
        self.glue(|| {
            scaled_rmsnorm_chunks(
                &ple_raw,
                tokens,
                N_LAYERS,
                PLE_WIDTH,
                (HIDDEN as f32).powf(-0.5),
                Some(ple_norm),
                RMS_EPS,
                &mut ple,
                threads,
            )
        })?;
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

        let mut h = self.backend.buffer(x.len())?;
        let input_norm = self.norm("language_model.layers.0.input_layernorm.weight")?;
        self.glue(|| {
            crate::cpu::kernels::rmsnorm_nthreads(
                &x,
                HIDDEN,
                Some(input_norm),
                RMS_EPS,
                &mut h,
                threads,
            )
        })?;

        for layer in 0..N_LAYERS {
            let base = format!("language_model.layers.{layer}");
            let global = is_global(layer);
            let d = head_dim(layer);
            let nkv = n_kv(layer);
            let nq = N_QUERY_HEADS;

            let fused = self.linear_f32(&h, &format!("{base}.self_attn.qkv_proj"))?;
            self.backend.recycle(h);
            let mut q = self.backend.buffer(tokens * nq * d)?;
            let mut k = self.backend.buffer(tokens * nkv * d)?;
            let mut v = self.backend.buffer(tokens * nkv * d)?;
            let rope = if global {
                &self.global_rope
            } else {
                &self.local_rope
            };
            let q_scale = self.norm(&format!("{base}.self_attn.q_norm.weight"))?;
            let k_scale = self.norm(&format!("{base}.self_attn.k_norm.weight"))?;
            self.glue(|| {
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
                    Some(q_scale),
                    Some(k_scale),
                    RMS_EPS,
                    (&rope.cos, &rope.sin),
                    threads,
                )
            })?;
            self.backend.recycle(fused);
            let attn = self.backend.attention_gqa_f32(
                &q,
                &k,
                &v,
                b,
                s,
                nq,
                nkv,
                d,
                self.tile,
                !global,
                SLIDING_WINDOW,
                ATTN_SCALE,
                &padding,
            )?;
            self.backend.recycle(q);
            self.backend.recycle(k);
            self.backend.recycle(v);
            check_shape!(attn.len() == tokens * nq * d, "attention rows");

            let o = self.linear_f32(&attn, &format!("{base}.self_attn.o_proj"))?;
            self.backend.recycle(attn);
            let post_attn = self.norm(&format!("{base}.post_attention_layernorm.weight"))?;
            let pre_ff = self.norm(&format!("{base}.pre_feedforward_layernorm.weight"))?;
            let mut pre = self.backend.buffer(x.len())?;
            self.glue(|| {
                add_rmsnorm_then_norm(
                    &mut x,
                    &o,
                    HIDDEN,
                    Some(post_attn),
                    1.0,
                    RMS_EPS,
                    Some(pre_ff),
                    &mut pre,
                    threads,
                )
            })?;
            self.backend.recycle(o);

            let up = self.linear_f32(&pre, &format!("{base}.mlp.up_proj"))?;
            let gated = self.gated_f32(&pre, &format!("{base}.mlp.gate_proj"), &up)?;
            self.backend.recycle(pre);
            self.backend.recycle(up);
            let down = self.linear_f32(&gated, &format!("{base}.mlp.down_proj"))?;
            self.backend.recycle(gated);
            let post_ff = self.norm(&format!("{base}.post_feedforward_layernorm.weight"))?;
            self.glue(|| add_rmsnorm(&mut x, &down, HIDDEN, Some(post_ff), 1.0, RMS_EPS, threads))?;
            self.backend.recycle(down);

            let ple_layer = &ple[layer * tokens * PLE_WIDTH..(layer + 1) * tokens * PLE_WIDTH];
            let mixed = self.gated_f32(
                &x,
                &format!("{base}.ple_block.per_layer_input_gate"),
                ple_layer,
            )?;
            let proj =
                self.linear_f32(&mixed, &format!("{base}.ple_block.per_layer_projection"))?;
            self.backend.recycle(mixed);
            let ple_out = self.norm(&format!(
                "{base}.ple_block.post_per_layer_input_norm.weight"
            ))?;
            let next = if layer + 1 < N_LAYERS {
                format!("language_model.layers.{}.input_layernorm.weight", layer + 1)
            } else {
                "language_model.norm.weight".to_string()
            };
            let next = self.norm(&next)?;
            h = self.backend.buffer(x.len())?;
            self.glue(|| {
                add_rmsnorm_then_norm(
                    &mut x,
                    &proj,
                    HIDDEN,
                    Some(ple_out),
                    self.scalars[layer],
                    RMS_EPS,
                    Some(next),
                    &mut h,
                    threads,
                )
            })?;
            self.backend.recycle(proj);
        }
        self.backend.recycle(ple);
        self.backend.recycle(x);

        // `h` holds the final-normed rows. Mean-pool live tokens, then project.
        let pooled = mean_pool(&h, batch, b, s, HIDDEN);
        self.backend.recycle(h);
        let proj = self.linear_f32(&pooled, "language_model.embedding_projection")?;
        check_shape!(proj.len() == b * EMBED_DIM, "embedding rows");
        Ok(l2_rows(proj, EMBED_DIM))
    }
}
