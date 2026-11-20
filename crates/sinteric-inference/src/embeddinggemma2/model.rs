//! EmbeddingGemma 2 text tower on [`OnednnCpu`].
//!
//! [`Precision::Bf16`] (default): matmuls are BF16×BF16 with FP32
//! accumulation; RMSNorm, softmax, RoPE, the residual stream, pooling and L2
//! stay in FP32. [`Precision::Fp32`] runs its own graph ([`fp32`]) with f32
//! activations into every matmul. Token embeddings are a zero-copy mmap of
//! the safetensors BF16 table.

use crate::cpu::kernels::{
    add_rmsnorm, bf16_bits_to_f32, f32_slice_to_bf16_nthreads, for_ranges, for_row_tiles,
    rmsnorm_rope_to_bf16, rmsnorm_to_bf16, row_mut, row_ref, scale_inplace, DefaultRope,
    SQRT_512_BF16,
};
use crate::cpu::onednn::{cache_path_with, CacheOptions, OnednnCpu, PackedWeight};
use crate::tensor_file::Mapped;
use crate::{check_shape, Result, ResultContext as Context};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::embeddinggemma2::config::{
    head_dim, is_global, n_kv, ATTN_SCALE, EMBED_DIM, HIDDEN, N_LAYERS, N_QUERY_HEADS, PLE_WIDTH,
    RMS_EPS, SLIDING_WINDOW, VOCAB,
};
use crate::embeddinggemma2::frontend::Batch;

mod audio;
mod fp32;
mod load;
mod tower;
mod vision;

use load::expected_weight_pin;
use tower::TowerSource;

pub use crate::cpu::runtime::Precision;

/// Precisions EG2 implements; the first is the default. Weights are the BF16
/// checkpoint either way: [`Precision::Fp32`] widens them exactly (BF16
/// storage), so only activations differ.
pub const PRECISIONS: &[Precision] = &[Precision::Bf16, Precision::Fp32];

/// Stats label of a precision EG2 runs.
fn precision_label(p: Precision) -> &'static str {
    if p == Precision::Fp32 {
        "FP32"
    } else {
        "BF16-matmul/FP32-stream"
    }
}

struct Linear {
    weight: PackedWeight,
}

/// Soft tokens replacing placeholder rows of one sequence: `(first
/// position, [n, HIDDEN] rows)`, in sequence order.
pub(crate) type Splice = Vec<(usize, Vec<f32>)>;

pub struct Model {
    embeddings: Mapped<u16>,
    vision: Option<vision::VisionTower>,
    audio: Option<audio::AudioTower>,
    backend: Arc<OnednnCpu>,
    linears: HashMap<String, Linear>,
    norms: HashMap<String, Vec<f32>>,
    scalars: Vec<f32>,
    local_rope: DefaultRope,
    global_rope: DefaultRope,
    tile: usize,
    max: usize,
    precision: Precision,
    load_source: &'static str,
    load_profile: BTreeMap<String, f64>,
}

impl Model {
    fn linear(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len() % l.weight.k() == 0, "linear matrix {name}");
        let mut y = self.backend.buffer(crate::checked_size(
            "activation elements",
            &[x.len() / l.weight.k(), l.weight.n()],
        )?)?;
        self.backend.linear_bf16(x, &l.weight, &mut y, false)?;
        Ok(y)
    }

    fn linear_src(&self, x: &[u16], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len() % l.weight.k() == 0, "linear matrix {name}");
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
        check_shape!(x.len() % width == 0 && scale.len() == width, "rms {name}");
        self.backend
            .rmsnorm(x.len() / width, width, x, Some(scale), RMS_EPS)
    }

    fn rms_bf16(&self, x: &[f32], name: &str, width: usize) -> Result<Vec<u16>> {
        let scale = self
            .norms
            .get(name)
            .with_context(|| format!("norm {name}"))?;
        check_shape!(x.len() % width == 0 && scale.len() == width, "rms {name}");
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

    /// Scaled token embeddings in `m_pad >= b * s` rows; pad rows are zero.
    /// `splices` (empty, or one per sequence) overwrite placeholder rows
    /// with unscaled soft tokens.
    fn embed(
        &self,
        batch: &Batch,
        b: usize,
        s: usize,
        m_pad: usize,
        splices: &[Splice],
    ) -> Result<Vec<f32>> {
        check_shape!(
            self.embeddings.as_slice().len() == VOCAB * HIDDEN,
            "embedding shape"
        );
        let ntok = b * s;
        check_shape!(m_pad >= ntok, "embedding rows");
        let mut x = self.backend.buffer(crate::checked_size(
            "activation elements",
            &[m_pad, HIDDEN],
        )?)?;
        let x_p = x.as_mut_ptr() as usize;
        let ids: Vec<u32> = batch.input_ids.iter().flatten().copied().collect();
        let table = self.embeddings.as_slice().as_ptr() as usize;
        for_row_tiles(
            self.backend.threads(),
            ntok,
            self.backend.residual_row_tile(ntok),
            8,
            |lo, hi| {
                for t in lo..hi {
                    let id = ids[t] as usize;
                    let src = unsafe { row_ref(table as *const u16, id, HIDDEN) };
                    let dst = unsafe { row_mut(x_p as *mut f32, t, HIDDEN) };
                    embed_row(src, dst, SQRT_512_BF16);
                }
            },
        )?;
        if m_pad > ntok {
            x[ntok * HIDDEN..].fill(0.0);
        }
        check_shape!(splices.is_empty() || splices.len() == b, "splice rows");
        for (bi, splice) in splices.iter().enumerate() {
            for (pos, rows) in splice {
                let n = rows.len() / HIDDEN;
                check_shape!(
                    rows.len() % HIDDEN == 0 && pos + n <= s,
                    "soft tokens {pos}+{n} outside sequence of {s}"
                );
                x[(bi * s + pos) * HIDDEN..][..rows.len()].copy_from_slice(rows);
            }
        }
        Ok(x)
    }

    fn to_heads(&self, src: &[f32], b: usize, s: usize, nh: usize, d: usize) -> Result<Vec<f32>> {
        self.to_heads_strided(src, b, s, nh, d, nh * d, 0)
    }

    fn to_heads_strided(
        &self,
        src: &[f32],
        b: usize,
        s: usize,
        nh: usize,
        d: usize,
        stride: usize,
        off: usize,
    ) -> Result<Vec<f32>> {
        let mut out = self
            .backend
            .buffer(crate::checked_size("activation elements", &[b, nh, s, d])?)?;
        let ntok = b * s;
        let src_p = src.as_ptr() as usize;
        let dst_p = out.as_mut_ptr() as usize;
        for_ranges(self.backend.threads(), ntok, 8, |lo, hi| {
            for idx in lo..hi {
                let bi = idx / s;
                let p = idx % s;
                for h in 0..nh {
                    let so = (bi * s + p) * stride + off + h * d;
                    let dst = ((bi * nh + h) * s + p) * d;
                    let src = unsafe { row_ref(src_p as *const f32, so / d, d) };
                    let dst = unsafe { row_mut(dst_p as *mut f32, dst / d, d) };
                    dst.copy_from_slice(src);
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
        let src_p = fused.as_ptr() as usize;
        let q_p = q.as_mut_ptr() as usize;
        let k_p = k.as_mut_ptr() as usize;
        let v_p = v.as_mut_ptr() as usize;
        for_ranges(self.backend.threads(), ntok, 8, |lo, hi| {
            for idx in lo..hi {
                let bi = idx / s;
                let p = idx % s;
                let base = (bi * s + p) * stride;
                for h in 0..nq {
                    let so = base + h * d;
                    let dst = ((bi * nq + h) * s + p) * d;
                    let src = unsafe { row_ref(src_p as *const f32, so / d, d) };
                    let dst = unsafe { row_mut(q_p as *mut f32, dst / d, d) };
                    dst.copy_from_slice(src);
                }
                for h in 0..nkv {
                    let ks = base + qn + h * d;
                    let vs = base + qn + kn + h * d;
                    let dst = ((bi * nkv + h) * s + p) * d;
                    let ksrc = unsafe { row_ref(src_p as *const f32, ks / d, d) };
                    let vsrc = unsafe { row_ref(src_p as *const f32, vs / d, d) };
                    let kdst = unsafe { row_mut(k_p as *mut f32, dst / d, d) };
                    let vdst = unsafe { row_mut(v_p as *mut f32, dst / d, d) };
                    kdst.copy_from_slice(ksrc);
                    vdst.copy_from_slice(vsrc);
                }
            }
        });
        Ok((q, k, v))
    }

    pub fn forward(&self, batch: &Batch) -> Result<Vec<Vec<f32>>> {
        let b = batch.input_ids.len();
        batch.validate(self.max)?;
        check_shape!(
            batch.attention_mask.len() == b
                && batch
                    .input_ids
                    .iter()
                    .zip(&batch.attention_mask)
                    .all(|(ids, m)| !ids.is_empty() && ids.len() == m.len()),
            "attention_mask length"
        );
        if dense_uniform(batch) {
            return self.forward_dense(batch, &[]);
        }
        let rows = compact_rows(batch)?;
        check_shape!(
            rows.iter().all(|(_, ids)| ids.len() <= self.max),
            "sequence length"
        );
        let rectangular = batch
            .input_ids
            .iter()
            .all(|ids| ids.len() == batch.input_ids[0].len());
        if !worth_split(&rows) && rectangular {
            // Short unique lengths: N serial B1 forwards lose to one padded batch.
            return self.forward_dense(batch, &[]);
        }
        self.forward_grouped_rows(rows)
    }

    /// Sort by live length and run equal-length groups with no pad tokens.
    fn forward_grouped_rows(&self, mut rows: Vec<(usize, Vec<u32>)>) -> Result<Vec<Vec<f32>>> {
        let b = rows.len();
        rows.sort_by_key(|(_, ids)| std::cmp::Reverse(ids.len()));
        let mut out = vec![Vec::new(); b];
        let mut i = 0;
        while i < rows.len() {
            let len = rows[i].1.len();
            let mut j = i + 1;
            while j < rows.len() && rows[j].1.len() == len {
                j += 1;
            }
            let sub = Batch {
                input_ids: rows[i..j].iter().map(|(_, ids)| ids.clone()).collect(),
                attention_mask: vec![vec![1u32; len]; j - i],
            };
            let got = self.forward_dense(&sub, &[])?;
            for (k, (orig, _)) in rows[i..j].iter().enumerate() {
                out[*orig] = got[k].clone();
            }
            i = j;
        }
        Ok(out)
    }

    fn forward_dense(&self, batch: &Batch, splices: &[Splice]) -> Result<Vec<Vec<f32>>> {
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
        let tokens = crate::checked_size("EG2 tokens", &[b, s])?;
        crate::checked_size(
            "EG2 workspace",
            &[
                tokens
                    .checked_add(64)
                    .ok_or(crate::Error::SizeOverflow("padded rows"))?,
                N_LAYERS,
                PLE_WIDTH,
                4,
            ],
        )?;
        let m_pad = self.work_rows(tokens);
        // These are the existing reservation counts; sum before allocating any.
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
            .ok_or(crate::Error::SizeOverflow("EG2 workspace"))?;
        self.backend.validate_workspace(planned)?;
        self.backend.reserve(
            crate::checked_size("activation elements", &[m_pad, HIDDEN])?,
            4,
        )?;
        self.backend.reserve(
            crate::checked_size("activation elements", &[m_pad, 2048])?,
            3,
        )?;
        self.backend.reserve(
            crate::checked_size("activation elements", &[m_pad, 3072])?,
            2,
        )?;
        self.backend.reserve(
            crate::checked_size("activation elements", &[m_pad, PLE_WIDTH])?,
            3,
        )?;
        self.backend.reserve(
            crate::checked_size("activation elements", &[m_pad, N_LAYERS, PLE_WIDTH])?,
            2,
        )?;
        self.backend.reserve_u16(
            crate::checked_size("activation elements", &[m_pad, HIDDEN])?,
            3,
        )?;
        self.backend.reserve_u16(
            crate::checked_size("activation elements", &[tokens, N_QUERY_HEADS, 512])?,
            3,
        )?;
        self.backend.reserve_u16(
            crate::checked_size("activation elements", &[tokens, N_QUERY_HEADS, 512])?,
            2,
        )?;
        self.backend.reserve(
            crate::checked_size("activation elements", &[m_pad, HIDDEN])?,
            2,
        )?;

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

        for layer in 0..N_LAYERS {
            let base = format!("language_model.layers.{layer}");
            let global = is_global(layer);
            let d = head_dim(layer);
            let nkv = n_kv(layer);
            let nq = N_QUERY_HEADS;

            let residual = x;
            let h_bf =
                self.rms_bf16(&residual, &format!("{base}.input_layernorm.weight"), HIDDEN)?;
            let qkv_name = format!("{base}.self_attn.qkv_proj");
            let (q, k, v) = if self.linears.contains_key(&qkv_name) {
                let fused = self.linear_src(&h_bf, &qkv_name)?;
                let qkv = self.split_qkv_heads(&fused, b, s, nq, nkv, d)?;
                self.backend.recycle(fused);
                qkv
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
                (q, k, v)
            };
            self.backend.recycle_u16(h_bf);
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
            rmsnorm_rope_to_bf16(
                &q,
                &mut q_bf,
                b,
                s,
                nq,
                d,
                Some(q_scale),
                RMS_EPS,
                Some((&rope.cos, &rope.sin)),
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
                Some((&rope.cos, &rope.sin)),
                threads,
            )?;
            rmsnorm_to_bf16(&v, d, None, RMS_EPS, &mut v_bf, threads)?;
            self.backend
                .add_category("glue", t_glue.elapsed().as_secs_f64());
            self.backend.recycle(q);
            self.backend.recycle(k);
            self.backend.recycle(v);

            let attn = self.backend.attention_gqa_src(
                &q_bf,
                &k_bf,
                &v_bf,
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
            self.backend.recycle_u16(q_bf);
            self.backend.recycle_u16(k_bf);
            self.backend.recycle_u16(v_bf);
            check_shape!(attn.len() == m_pad * nq * d, "attn residual rows");

            let o = self.linear(&attn, &format!("{base}.self_attn.o_proj"))?;
            self.backend.recycle(attn);
            let post_attn = self
                .norms
                .get(&format!("{base}.post_attention_layernorm.weight"))
                .with_context(|| format!("{base}.post_attention_layernorm.weight"))?;
            let mut h = residual;
            check_shape!(h.len() == o.len(), "attn residual");
            let t_glue = Instant::now();
            add_rmsnorm(
                &mut h,
                &o,
                HIDDEN,
                Some(post_attn),
                1.0,
                RMS_EPS,
                self.backend.threads(),
            )?;
            self.backend
                .add_category("glue", t_glue.elapsed().as_secs_f64());
            self.backend.recycle(o);

            let residual = h;
            let pre_bf = self.rms_bf16(
                &residual,
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
            let post_ff = self
                .norms
                .get(&format!("{base}.post_feedforward_layernorm.weight"))
                .with_context(|| format!("{base}.post_feedforward_layernorm.weight"))?;
            let mut h = residual;
            let t_glue = Instant::now();
            add_rmsnorm(
                &mut h,
                &down,
                HIDDEN,
                Some(post_ff),
                1.0,
                RMS_EPS,
                self.backend.threads(),
            )?;
            self.backend
                .add_category("glue", t_glue.elapsed().as_secs_f64());
            self.backend.recycle(down);

            let residual = h;
            let res_bf = self.to_bf16(&residual)?;
            let gate =
                self.linear_src(&res_bf, &format!("{base}.ple_block.per_layer_input_gate"))?;
            self.backend.recycle_u16(res_bf);
            let mut mixed = self.backend.buffer(gate.len())?;
            let ple_off = layer * PLE_WIDTH;
            let threads = self.backend.threads();
            let gate_p = gate.as_ptr() as usize;
            let ple_p = ple.as_ptr() as usize;
            let mix_p = mixed.as_mut_ptr() as usize;
            let mix_rows = gate.len() / PLE_WIDTH;
            for_row_tiles(
                threads,
                mix_rows,
                self.backend.residual_row_tile(tokens),
                8,
                |lo, hi| {
                    for ti in lo..hi {
                        let src = unsafe { row_ref(gate_p as *const f32, ti, PLE_WIDTH) };
                        let p = unsafe {
                            row_ref(
                                (ple_p as *const f32).add(ti * (N_LAYERS * PLE_WIDTH) + ple_off),
                                0,
                                PLE_WIDTH,
                            )
                        };
                        let row = unsafe { row_mut(mix_p as *mut f32, ti, PLE_WIDTH) };
                        unsafe { crate::cpu::kernels::gelu_tanh_mul_unchecked(src, p, row) };
                    }
                },
            )?;
            self.backend.recycle(gate);
            let proj = self.linear(&mixed, &format!("{base}.ple_block.per_layer_projection"))?;
            self.backend.recycle(mixed);
            let ple_norm = self
                .norms
                .get(&format!(
                    "{base}.ple_block.post_per_layer_input_norm.weight"
                ))
                .with_context(|| format!("{base}.ple_block.post_per_layer_input_norm.weight"))?;
            x = residual;
            let t_glue = Instant::now();
            add_rmsnorm(
                &mut x,
                &proj,
                HIDDEN,
                Some(ple_norm),
                self.scalars[layer],
                RMS_EPS,
                self.backend.threads(),
            )?;
            self.backend
                .add_category("glue", t_glue.elapsed().as_secs_f64());
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

    /// Load the image and/or audio towers next to the text tower. Each keeps
    /// its own packed cache beside the text one, so text-only loads never
    /// read or pack them.
    pub fn load_media(
        &mut self,
        path: &Path,
        cache: &CacheOptions,
        image: bool,
        audio: bool,
    ) -> Result<()> {
        if !image && !audio {
            return Ok(());
        }
        let st_path = path.join("model.safetensors");
        let (sha, _) = expected_weight_pin(cache.manifest.as_deref())?;
        let text_cache = cache_path_with(&st_path, self.backend.cpu_isa(), cache.file.as_deref());
        let src = TowerSource {
            st_path: &st_path,
            text_cache: &text_cache,
            sha: &sha,
            cache,
        };
        if image && self.vision.is_none() {
            let t = Instant::now();
            let tower = vision::VisionTower::load(&src, &self.backend, self.precision)
                .map_err(|e| e.at(path))?;
            self.load_profile
                .insert("vision_load_s".into(), t.elapsed().as_secs_f64());
            self.vision = Some(tower);
        }
        if audio && self.audio.is_none() {
            let t = Instant::now();
            let tower = audio::AudioTower::load(&src, &self.backend, self.precision)
                .map_err(|e| e.at(path))?;
            self.load_profile
                .insert("audio_load_s".into(), t.elapsed().as_secs_f64());
            self.audio = Some(tower);
        }
        Ok(())
    }

    pub fn threads(&self) -> usize {
        self.backend.threads()
    }

    pub fn has_image(&self) -> bool {
        self.vision.is_some()
    }

    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Soft tokens `[img.soft_tokens(), HIDDEN]` of one image.
    pub(crate) fn image_soft_tokens(
        &self,
        img: &crate::embeddinggemma2::media::ImagePatches,
    ) -> Result<Vec<f32>> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| crate::Error::Unsupported {
                what: "EG2 image input",
                reason: "load with Config::image = true".into(),
            })?;
        tower.forward(&self.backend, self.precision, self.tile, img)
    }

    /// Soft tokens `[f.soft_tokens(), HIDDEN]` of one audio clip.
    pub(crate) fn audio_soft_tokens(
        &self,
        f: &crate::embeddinggemma2::media::AudioFeatures,
    ) -> Result<Vec<f32>> {
        let tower = self
            .audio
            .as_ref()
            .ok_or_else(|| crate::Error::Unsupported {
                what: "EG2 audio input",
                reason: "load with Config::audio = true".into(),
            })?;
        tower.forward(&self.backend, self.precision, f)
    }

    /// Embed sequences whose placeholder rows are replaced by soft tokens.
    /// Equal lengths share one dense forward; outputs keep input order.
    pub(crate) fn forward_spliced(&self, rows: Vec<(Vec<u32>, Splice)>) -> Result<Vec<Vec<f32>>> {
        check_shape!(
            rows.iter()
                .all(|(ids, _)| !ids.is_empty() && ids.len() <= self.max),
            "sequence length"
        );
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(rows[i].0.len()));
        let mut rows: Vec<Option<(Vec<u32>, Splice)>> = rows.into_iter().map(Some).collect();
        let mut out = vec![Vec::new(); rows.len()];
        let mut i = 0;
        while i < order.len() {
            let len = rows[order[i]].as_ref().map_or(0, |r| r.0.len());
            let mut j = i + 1;
            while j < order.len() && rows[order[j]].as_ref().map_or(0, |r| r.0.len()) == len {
                j += 1;
            }
            let mut ids = Vec::with_capacity(j - i);
            let mut splices = Vec::with_capacity(j - i);
            for &k in &order[i..j] {
                let (row, splice) = rows[k].take().context("spliced row")?;
                ids.push(row);
                splices.push(splice);
            }
            let sub = Batch {
                attention_mask: vec![vec![1u32; len]; ids.len()],
                input_ids: ids,
            };
            sub.validate(self.max)?;
            let got = self.forward_dense(&sub, &splices)?;
            for (&k, emb) in order[i..j].iter().zip(got) {
                out[k] = emb;
            }
            i = j;
        }
        Ok(out)
    }

    /// Run one discarded 16-token forward through the normal execution path.
    ///
    /// Populates execution caches and faults working pages without changing
    /// weights or subsequent results. Uses the normal execution path and
    /// returns its typed errors. No warmup runs implicitly when loading a Model.
    pub fn warmup(&self) -> Result<()> {
        let batch = Batch {
            input_ids: vec![vec![2u32; 16]],
            attention_mask: vec![vec![1u32; 16]],
        };
        self.forward(&batch)?;
        Ok(())
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn stats(&self) -> Value {
        let mut stats = self.backend.stats();
        stats["load_source"] = serde_json::json!(self.load_source);
        stats["load_profile_s"] = serde_json::json!(self.load_profile);
        stats["precision"] = serde_json::json!(precision_label(self.precision));
        stats["modalities"] = serde_json::json!({
            "image": self.vision.as_ref().map(|t| t.load_source()),
            "audio": self.audio.as_ref().map(|t| t.load_source()),
        });
        stats
    }
}

fn embed_row(src: &[u16], dst: &mut [f32], scale: f32) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
            unsafe {
                embed_row_avx512(src, dst, scale);
            }
            return;
        }
    }
    for (d, &bits) in dst.iter_mut().zip(src) {
        *d = bf16_bits_to_f32(bits) * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn embed_row_avx512(src: &[u16], dst: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let n = src.len();
    let s = _mm512_set1_ps(scale);
    let mut i = 0;
    while i + 16 <= n {
        let packed = _mm256_loadu_si256(src.as_ptr().add(i).cast());
        let f = _mm512_cvtpbh_ps(std::mem::transmute::<__m256i, __m256bh>(packed));
        _mm512_storeu_ps(dst.as_mut_ptr().add(i), _mm512_mul_ps(f, s));
        i += 16;
    }
    while i < n {
        *dst.get_unchecked_mut(i) = bf16_bits_to_f32(*src.get_unchecked(i)) * scale;
        i += 1;
    }
}

fn dense_uniform(batch: &Batch) -> bool {
    let s = batch.input_ids[0].len();
    batch.input_ids.iter().all(|ids| ids.len() == s)
        && batch
            .attention_mask
            .iter()
            .all(|m| m.len() == s && m.iter().all(|&v| v != 0))
}

fn compact_rows(batch: &Batch) -> Result<Vec<(usize, Vec<u32>)>> {
    let mut rows = Vec::with_capacity(batch.input_ids.len());
    for (i, (ids, mask)) in batch
        .input_ids
        .iter()
        .zip(&batch.attention_mask)
        .enumerate()
    {
        let live: Vec<u32> = ids
            .iter()
            .zip(mask)
            .filter_map(|(&id, &m)| (m != 0).then_some(id))
            .collect();
        check_shape!(!live.is_empty(), "empty sequence");
        rows.push((i, live));
    }
    Ok(rows)
}

/// Split mixed lengths only when pad waste is large and groups are not all tiny B1s.
fn worth_split(rows: &[(usize, Vec<u32>)]) -> bool {
    if rows.len() <= 1 {
        return false;
    }
    let max = rows.iter().map(|(_, r)| r.len()).max().unwrap_or(0);
    let live: usize = rows.iter().map(|(_, r)| r.len()).sum();
    let padded = rows.len().saturating_mul(max);
    if live.saturating_mul(5) / 4 >= padded {
        return false;
    }
    let mut counts = HashMap::new();
    for (_, r) in rows {
        *counts.entry(r.len()).or_insert(0usize) += 1;
    }
    if counts.len() == 1 {
        return true;
    }
    let all_singletons = counts.values().all(|&n| n == 1);
    !(all_singletons && max <= 256)
}

fn mean_pool(x: &[f32], batch: &Batch, b: usize, s: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; b * dim];
    for bi in 0..b {
        let mut count = 0.0f32;
        for p in 0..s {
            if batch.attention_mask[bi][p] == 0 {
                continue;
            }
            count += 1.0;
            let src = (bi * s + p) * dim;
            let dst = bi * dim;
            for j in 0..dim {
                out[dst + j] += x[src + j];
            }
        }
        let inv = 1.0 / count.max(1.0);
        for j in 0..dim {
            out[bi * dim + j] *= inv;
        }
    }
    out
}

fn l2_rows(mut x: Vec<f32>, dim: usize) -> Vec<Vec<f32>> {
    let mut rows = Vec::with_capacity(x.len() / dim);
    for row in x.chunks_exact_mut(dim) {
        let sum = row.iter().map(|v| v * v).sum::<f32>();
        let inv = 1.0 / sum.sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v *= inv;
        }
        rows.push(row.to_vec());
    }
    rows
}

pub fn mrl_truncate(emb: &[f32], dim: usize) -> Vec<f32> {
    let d = dim.min(emb.len());
    let mut v = emb[..d].to_vec();
    let sum = v.iter().map(|x| x * x).sum::<f32>();
    let inv = 1.0 / sum.sqrt().max(1e-12);
    for x in &mut v {
        *x *= inv;
    }
    v
}
