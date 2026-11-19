//! CPU-only Vela 2 inference. The JSON tensor contract is the released ONNX contract.
//! Rust schema/tokenization/calibration/decoding live in the sibling `native-text` crate.
use anyhow::{bail, ensure, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{linear, linear_no_bias, Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::time::Instant;
mod blas;
pub mod native_text;
mod workspace;

#[derive(Deserialize)]
struct Config {
    encoder_config: EncoderConfig,
    proj_dim: usize,
    readout: String,
    max_length: usize,
}
#[derive(Deserialize)]
struct EncoderConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    global_attn_every_n_layers: usize,
    global_rope_theta: f64,
    local_rope_theta: f64,
    local_attention: usize,
    rope_scaling: RopeConfig,
}
#[derive(Deserialize)]
struct RopeConfig {
    factor: f64,
    original_max_position_embeddings: usize,
    beta_fast: f64,
    beta_slow: f64,
    truncate: bool,
    rope_type: String,
}
/// Released six-tensor contract. Indices include official dummy rows when a head is absent.
#[derive(Debug, Clone, Deserialize)]
pub struct Inputs {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub q_index: Vec<[usize; 4]>,
    pub opt_index: Vec<[usize; 3]>,
    pub unit_index: Vec<[usize; 2]>,
    pub ent_index: Vec<[usize; 2]>,
}
/// Raw head tensors; apply the pinned task-specific calibration and decoding.
pub struct Output {
    pub opt_logits: Tensor,
    pub span_logits: Tensor,
}
/// Initialize the optional process-wide FP32 library before loading optimized models.
pub use blas::initialize as initialize_blas;
#[doc(hidden)]
pub use blas::initialize_with_options as initialize_blas_with_options;
pub use blas::{projection_allocation_profile, ProjectionAllocation};
pub use workspace::WorkspaceStats;
#[derive(Clone)]
pub struct ModelOptions {
    /// Validated at load: dense, blocked or adaptive attention.
    pub mode: String,
    /// Validated at load: eager or immutable file-backed mmap embedding.
    pub embedding: String,
    /// Positive attention query tile size.
    pub chunk: usize,
    pub fused_norm: bool,
    /// Use the explicitly initialized standalone numerical backend.
    pub blas: bool,
    pub blas_attention: bool,
    /// Own M-independent packed projection weights instead of retaining originals.
    pub packed: bool,
    pub fused_geglu: bool,
    pub fused_qkv: bool,
    pub fused_softmax: bool,
    pub native_norm: bool,
    /// Experimental inference-only bounded per-forward scratch reuse.
    pub projection_workspaces: bool,
    /// Experimental update of forward-private initialized residual storage.
    pub residual_inplace: bool,
    /// Experimental QKV/RoPE storage under the same workspace lease/budget.
    pub qkv_rope_workspace: bool,
    /// Experimental reuse of fresh private attention-score storage.
    pub inplace_softmax: bool,
}
struct Normalization {
    inner: LayerNorm,
    native: bool,
}
impl Module for Normalization {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if self.native {
            blas::layer_norm(
                x,
                self.inner.weight(),
                self.inner.bias().unwrap(),
                self.inner.eps() as f32,
            )
        } else {
            self.inner.forward(x)
        }
    }
}
struct Projection {
    inner: Option<Linear>,
    blas: bool,
    fixed: Option<blas::Packed>,
    fixed_bias: Option<Tensor>,
}
impl Projection {
    fn update_private_residual(&self, input: &Tensor, residual: &Tensor) -> Result<()> {
        let packed = self
            .fixed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("residual update requires fixed packing"))?;
        ensure!(
            self.fixed_bias.is_none() && residual.is_contiguous(),
            "residual update requires unbiased projection and contiguous activation"
        );
        let dims = input.dims();
        let k = *dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("residual update rank"))?;
        let rows = input.elem_count() / k;
        let input = input.contiguous()?.reshape((rows, k))?;
        let output = residual.reshape((rows, packed.n()))?;
        // The Model owns this fresh residual; it never escapes a forward.
        unsafe { packed.forward_residual_into(&input, &output)? };
        Ok(())
    }
    fn forward_workspace(
        &self,
        x: &Tensor,
        workspace: &mut workspace::Workspace,
        slot: usize,
    ) -> Result<Tensor> {
        let fixed = self
            .fixed
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("workspace requires fixed packing"))?;
        ensure!(
            self.fixed_bias.is_none(),
            "workspace projection must be unbiased"
        );
        let shape = x.dims();
        let k = *shape
            .last()
            .ok_or_else(|| anyhow::anyhow!("workspace projection rank"))?;
        let flat = x.contiguous()?.reshape((x.elem_count() / k, k))?;
        let y = workspace.project(slot, fixed, &flat)?;
        let mut dims = shape.to_vec();
        *dims.last_mut().unwrap() = fixed.n();
        Ok(y.reshape(dims)?)
    }
    fn forward_with_residual(
        &self,
        x: &Tensor,
        residual: &Tensor,
        fuse: bool,
    ) -> candle_core::Result<Tensor> {
        // The process-wide experimental flag cannot assume every model uses
        // fixed packing. Default, unpacked and M-dependent projections retain
        // their ordinary forward path, including any bias.
        if fuse && self.fixed.is_some() && self.fixed_bias.is_none() {
            self.residual(x, residual)
        } else {
            self.forward(x)? + residual
        }
    }
    fn residual(&self, x: &Tensor, residual: &Tensor) -> candle_core::Result<Tensor> {
        let fixed = self.fixed.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("residual GEMM requires fixed packed weights".into())
        })?;
        if self.fixed_bias.is_some() {
            return Err(candle_core::Error::Msg(
                "residual GEMM supports unbiased projections".into(),
            ));
        }
        let shape = x.dims();
        let k = *shape
            .last()
            .ok_or_else(|| candle_core::Error::Msg("residual GEMM rank".into()))?;
        let m = x.elem_count() / k;
        let flat = x.contiguous()?.reshape((m, k))?;
        let r = residual.contiguous()?.reshape((m, fixed.n()))?;
        let value = fixed.forward_residual(&flat, &r)?;
        let mut dims = shape.to_vec();
        *dims.last_mut().unwrap() = fixed.n();
        value.reshape(dims)
    }
}
impl Module for Projection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if !self.blas {
            return self.inner.as_ref().unwrap().forward(x);
        }
        let shape = x.dims();
        let k = *shape.last().unwrap();
        let n = if let Some(fixed) = &self.fixed {
            fixed.n()
        } else {
            self.inner.as_ref().unwrap().weight().dim(0)?
        };
        let flat = x.contiguous()?.reshape((x.elem_count() / k, k))?;
        let y = if let Some(fixed) = &self.fixed {
            fixed.forward(&flat)?
        } else {
            blas::matmul(&flat, &self.inner.as_ref().unwrap().weight().t()?)?
        };
        let mut out_shape = shape.to_vec();
        *out_shape.last_mut().unwrap() = n;
        let y = y.reshape(out_shape)?;
        let bias = if self.fixed.is_some() {
            self.fixed_bias.as_ref()
        } else {
            self.inner.as_ref().unwrap().bias()
        };
        match bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}
fn projection(
    i: usize,
    o: usize,
    vb: VarBuilder,
    bias: bool,
    options: &ModelOptions,
) -> candle_core::Result<Projection> {
    let inner = if bias {
        linear(i, o, vb)?
    } else {
        linear_no_bias(i, o, vb)?
    };
    if options.packed && blas::m_independent() {
        // Pack once at load and release the original FP32 tensor; all M shapes share B.
        let fixed = blas::Packed::new(inner.weight())?;
        let fixed_bias = inner.bias().cloned();
        Ok(Projection {
            inner: None,
            blas: true,
            fixed: Some(fixed),
            fixed_bias,
        })
    } else {
        Ok(Projection {
            inner: Some(inner),
            blas: options.blas,
            fixed: None,
            fixed_bias: None,
        })
    }
}
struct Rope {
    cos: Tensor,
    sin: Tensor,
}
impl Rope {
    fn new(c: &EncoderConfig, theta: f64, len: usize) -> Result<Self> {
        let dim = c.hidden_size / c.num_attention_heads;
        let rc = &c.rope_scaling;
        ensure!(
            rc.rope_type == "yarn" && rc.factor >= 1.0 && dim % 2 == 0,
            "unsupported RoPE config"
        );
        let correction = |rot: f64| {
            dim as f64
                * (rc.original_max_position_embeddings as f64 / (rot * 2.0 * std::f64::consts::PI))
                    .ln()
                / (2.0 * theta.ln())
        };
        let (mut lo, mut hi) = (correction(rc.beta_fast), correction(rc.beta_slow));
        if rc.truncate {
            lo = lo.floor();
            hi = hi.ceil();
        }
        lo = lo.max(0.);
        hi = hi.min((dim - 1) as f64);
        if lo == hi {
            hi += 0.001;
        }
        let amp = (1.0 + 0.1 * rc.factor.ln()) as f32;
        let inv: Vec<f32> = (0..dim / 2)
            .map(|i| {
                let freq = (theta as f32).powf((2 * i) as f32 / dim as f32);
                let extrap = 1.0 - ((i as f32 - lo as f32) / (hi - lo) as f32).clamp(0., 1.);
                (1.0 / (rc.factor as f32 * freq)) * (1.0 - extrap) + (1.0 / freq) * extrap
            })
            .collect();
        let mut cos = Vec::with_capacity(len * dim / 2);
        let mut sin = Vec::with_capacity(len * dim / 2);
        for p in 0..len {
            for f in &inv {
                let x = p as f32 * f;
                cos.push(x.cos() * amp);
                sin.push(x.sin() * amp);
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (len, dim / 2), &Device::Cpu)?,
            sin: Tensor::from_vec(sin, (len, dim / 2), &Device::Cpu)?,
        })
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_nn::rotary_emb::rope(
            &x.contiguous()?,
            &self.cos,
            &self.sin,
        )?)
    }
}
struct Layer {
    qkv: Projection,
    attn_out: Projection,
    wi: Projection,
    wo: Projection,
    attn_norm: Option<Normalization>,
    mlp_norm: Normalization,
    local: bool,
}
pub struct Model {
    workspaces: workspace::Pool,
    config: Config,
    options: ModelOptions,
    embedding: Option<Embedding>,
    mapped: Option<MmapedSafetensors>,
    embed_norm: Normalization,
    final_norm: Normalization,
    layers: Vec<Layer>,
    global_rope: Rope,
    local_rope: Rope,
    norm: Normalization,
    w_o: Projection,
    w_q: Projection,
    w_p: Projection,
    w_t: Projection,
    w_e: Projection,
    mlp1: Projection,
    mlp2: Projection,
    tau: f64,
    tau_span: f64,
}
fn norm(
    vb: VarBuilder,
    h: usize,
    eps: f64,
    fused: bool,
    bias: bool,
    native: bool,
) -> Result<Normalization> {
    let w = vb.get(h, "weight")?;
    let inner = if bias {
        LayerNorm::new(w, vb.get(h, "bias")?, eps)
    } else if fused || native {
        LayerNorm::new(w, Tensor::zeros(h, DType::F32, &Device::Cpu)?, eps)
    } else {
        LayerNorm::new_no_bias(w, eps)
    };
    Ok(Normalization { inner, native })
}
impl ModelOptions {
    pub fn optimized() -> Self {
        Self {
            mode: "blocked".into(),
            embedding: "mmap".into(),
            chunk: 64,
            fused_norm: true,
            blas: true,
            blas_attention: true,
            packed: true,
            fused_geglu: true,
            fused_qkv: true,
            fused_softmax: true,
            native_norm: true,
            projection_workspaces: false,
            residual_inplace: false,
            qkv_rope_workspace: false,
            inplace_softmax: false,
        }
    }
}
impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            mode: "dense".into(),
            embedding: "eager".into(),
            chunk: 128,
            fused_norm: false,
            blas: false,
            blas_attention: false,
            packed: false,
            fused_geglu: false,
            fused_qkv: false,
            fused_softmax: false,
            native_norm: false,
            projection_workspaces: false,
            residual_inplace: false,
            qkv_rope_workspace: false,
            inplace_softmax: false,
        }
    }
}
impl Model {
    /// Diagnostic counters for this model's bounded, opt-in scratch pool.
    pub fn workspace_stats(&self) -> WorkspaceStats {
        self.workspaces.stats()
    }
    /// Run one prepared batch without benchmark repetitions or task decoding.
    pub fn forward(&self, input: &Inputs) -> Result<Output> {
        let (opt_logits, span_logits, _) = self.forward_profiled(input, false)?;
        Ok(Output {
            opt_logits,
            span_logits,
        })
    }

    pub fn load(path: impl AsRef<std::path::Path>, options: ModelOptions) -> Result<Self> {
        let path = path.as_ref();
        ensure!(
            ["dense", "blocked", "adaptive"].contains(&options.mode.as_str())
                && ["eager", "mmap"].contains(&options.embedding.as_str())
                && options.chunk > 0,
            "invalid model options"
        );
        ensure!(
            (!options.blas_attention
                && !options.packed
                && !options.fused_geglu
                && !options.fused_qkv
                && !options.fused_softmax
                && !options.native_norm)
                || options.blas,
            "custom options require BLAS"
        );
        ensure!(
            !options.blas || blas::is_initialized(),
            "initialize_blas must precede optimized model loading"
        );
        ensure!(
            !options.projection_workspaces || (options.packed && blas::m_independent()),
            "projection workspaces require fixed M-independent packing"
        );
        ensure!(
            !options.inplace_softmax || (options.blas_attention && options.fused_softmax),
            "private softmax requires native attention and fused softmax"
        );
        ensure!(
            !options.qkv_rope_workspace || (options.projection_workspaces && options.fused_qkv),
            "QKV/RoPE workspace requires projection workspaces and fused QKV"
        );
        ensure!(!options.residual_inplace || (options.packed && blas::m_independent() && options.native_norm && options.fused_qkv && options.fused_geglu && blas::gemm_residual_enabled()),"private residual updates require fixed packing, native normalization, fused QKV/GeGLU and enabled residual fusion");
        let config: Config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        ensure!(
            config.readout == "both",
            "this checkpoint needs the both readout"
        );
        let c = &config.encoder_config;
        let h = c.hidden_size;
        let file = path.join("model.safetensors");
        // The caller must keep this immutable model file unchanged for the process lifetime.
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&file], DType::F32, &Device::Cpu)? };
        let embedding = if options.embedding == "eager" {
            Some(candle_nn::embedding(
                c.vocab_size,
                h,
                vb.pp("encoder.embeddings.tok_embeddings"),
            )?)
        } else {
            None
        };
        let mapped = if embedding.is_none() {
            Some(unsafe { MmapedSafetensors::new(&file)? })
        } else {
            None
        };
        let embed_norm = norm(
            vb.pp("encoder.embeddings.norm"),
            h,
            c.layer_norm_eps,
            options.fused_norm,
            false,
            options.native_norm,
        )?;
        let final_norm = norm(
            vb.pp("encoder.final_norm"),
            h,
            c.layer_norm_eps,
            options.fused_norm,
            false,
            options.native_norm,
        )?;
        let mut layers = vec![];
        for i in 0..c.num_hidden_layers {
            let b = vb.pp(format!("encoder.layers.{i}"));
            layers.push(Layer {
                qkv: projection(h, 3 * h, b.pp("attn.Wqkv"), false, &options)?,
                attn_out: projection(h, h, b.pp("attn.Wo"), false, &options)?,
                wi: projection(h, 2 * c.intermediate_size, b.pp("mlp.Wi"), false, &options)?,
                wo: projection(c.intermediate_size, h, b.pp("mlp.Wo"), false, &options)?,
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(norm(
                        b.pp("attn_norm"),
                        h,
                        c.layer_norm_eps,
                        options.fused_norm,
                        false,
                        options.native_norm,
                    )?)
                },
                mlp_norm: norm(
                    b.pp("mlp_norm"),
                    h,
                    c.layer_norm_eps,
                    options.fused_norm,
                    false,
                    options.native_norm,
                )?,
                local: i % c.global_attn_every_n_layers != 0,
            });
        }
        let global_rope = Rope::new(c, c.global_rope_theta, config.max_length)?;
        let local_rope = Rope::new(c, c.local_rope_theta, config.max_length)?;
        let d = config.proj_dim;
        let head_norm = norm(vb.pp("norm"), h, 1e-5, true, true, options.native_norm)?;
        let w_o = projection(h, d, vb.pp("w_o"), true, &options)?;
        let w_q = projection(h, d, vb.pp("w_q"), true, &options)?;
        let w_p = projection(h, d, vb.pp("w_p"), true, &options)?;
        let w_t = projection(h, d, vb.pp("w_t"), true, &options)?;
        let w_e = projection(h, d, vb.pp("w_e"), true, &options)?;
        let mlp1 = projection(h, 2 * h, vb.pp("cls_mlp.0"), true, &options)?;
        let mlp2 = projection(2 * h, 1, vb.pp("cls_mlp.3"), true, &options)?;
        let tau = vb.get((), "log_tau")?.exp()?.to_scalar::<f32>()? as f64;
        let tau_span = vb.get((), "log_tau_span")?.exp()?.to_scalar::<f32>()? as f64;
        let workspaces = workspace::Pool::new(options.projection_workspaces);
        Ok(Self {
            workspaces,
            config,
            options,
            embedding,
            mapped,
            embed_norm,
            final_norm,
            layers,
            global_rope,
            local_rope,
            norm: head_norm,
            w_o,
            w_q,
            w_p,
            w_t,
            w_e,
            mlp1,
            mlp2,
            tau,
            tau_span,
        })
    }
    fn embed(&self, ids: &[Vec<u32>]) -> Result<Tensor> {
        let (b, s, h) = (
            ids.len(),
            ids[0].len(),
            self.config.encoder_config.hidden_size,
        );
        if let Some(e) = &self.embedding {
            let x = Tensor::from_vec(
                ids.iter().flatten().copied().collect::<Vec<_>>(),
                (b, s),
                &Device::Cpu,
            )?;
            return Ok(e.forward(&x)?);
        }
        let view = self
            .mapped
            .as_ref()
            .unwrap()
            .get("encoder.embeddings.tok_embeddings.weight")?;
        ensure!(
            view.shape() == [self.config.encoder_config.vocab_size, h]
                && DType::try_from(view.dtype())? == DType::F32,
            "embedding must be FP32 with expected shape"
        );
        let data: &[f32] = bytemuck::try_cast_slice(view.data())
            .map_err(|e| anyhow::anyhow!("embedding FP32 alignment: {e}"))?;
        let mut out = Vec::with_capacity(b * s * h);
        for &id in ids.iter().flatten() {
            let p = id as usize * h;
            out.extend_from_slice(&data[p..p + h]);
        }
        Ok(Tensor::from_vec(out, (b, s, h), &Device::Cpu)?)
    }
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &[Vec<u32>],
        native_mask: &Tensor,
        local: bool,
    ) -> Result<Tensor> {
        let (b, nh, s, d) = q.dims4()?;
        let radius = self.config.encoder_config.local_attention / 2;
        // Small sequences fit a full attention tile. Longer inputs retain the
        // bounded-memory chunk size; this mode is an explicit measured experiment.
        let chunk = if self.options.mode == "dense" || (self.options.mode == "adaptive" && s <= 128)
        {
            s
        } else {
            self.options.chunk.min(s)
        };
        let mut outputs = vec![];
        for start in (0..s).step_by(chunk) {
            let n = chunk.min(s - start);
            let (ks, ke) = if local && self.options.mode != "dense" {
                (start.saturating_sub(radius), (start + n + radius).min(s))
            } else {
                (0, s)
            };
            let nk = ke - ks;
            let qn = q.narrow(2, start, n)?;
            let qc = if self.options.fused_qkv {
                if self.options.blas_attention {
                    qn
                } else {
                    qn.contiguous()?
                }
            } else {
                (qn.contiguous()? * (d as f64).powf(-0.5))?
            };
            let kn = k.narrow(2, ks, nk)?;
            let kc = if self.options.blas_attention {
                kn
            } else {
                kn.contiguous()?
            };
            let kt = kc.transpose(2, 3)?;
            let scores = if self.options.blas_attention {
                blas::matmul(&qc, &kt)?
            } else {
                qc.matmul(&kt)?
            };
            let probs = if self.options.fused_softmax {
                if self.options.inplace_softmax {
                    // Native QK GEMM creates fresh no-backward storage; its scores
                    // never escape this attention tile. The following V GEMM owns
                    // separate storage before any tile/layer is reused.
                    blas::private_masked_softmax(
                        &scores,
                        native_mask,
                        start,
                        ks,
                        if local { Some(radius) } else { None },
                    )?;
                    scores.clone()
                } else {
                    blas::masked_softmax(
                        &scores,
                        native_mask,
                        start,
                        ks,
                        if local { Some(radius) } else { None },
                    )?
                }
            } else {
                let mut mv = Vec::with_capacity(b * n * nk);
                for row in mask {
                    for qi in start..start + n {
                        for ki in ks..ke {
                            mv.push(if row[ki] == 0 || (local && qi.abs_diff(ki) > radius) {
                                f32::MIN
                            } else {
                                0.
                            });
                        }
                    }
                }
                let m = Tensor::from_vec(mv, (b, 1, n, nk), &Device::Cpu)?;
                candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&m)?)?
            };
            let vn = v.narrow(2, ks, nk)?;
            let vc = if self.options.blas_attention {
                vn
            } else {
                vn.contiguous()?
            };
            let out = if self.options.blas_attention {
                blas::matmul(&probs, &vc)?
            } else {
                probs.matmul(&vc)?
            };
            ensure!(out.dims() == [b, nh, n, d], "attention output shape");
            outputs.push(out);
        }
        Ok(Tensor::cat(&outputs, 2)?)
    }
    pub fn forward_profiled(
        &self,
        r: &Inputs,
        profile_enabled: bool,
    ) -> Result<(Tensor, Tensor, BTreeMap<String, f64>)> {
        let _allocation_profile = blas::ProfileScope::enter(profile_enabled);
        let b = r.input_ids.len();
        ensure!(b > 0, "empty batch");
        let s = r.input_ids[0].len();
        ensure!(
            s > 0 && s <= self.config.max_length,
            "sequence outside released max_length"
        );
        ensure!(
            r.attention_mask.len() == b
                && r.input_ids.iter().all(|v| v.len() == s)
                && r.attention_mask.iter().all(|v| v.len() == s),
            "ragged tensors"
        );
        ensure!(
            r.attention_mask.iter().flatten().all(|&v| v <= 1),
            "attention mask must be binary"
        );
        ensure!(
            r.input_ids
                .iter()
                .flatten()
                .all(|&x| (x as usize) < self.config.encoder_config.vocab_size),
            "invalid token id"
        );
        ensure!(
            !r.q_index.is_empty()
                && !r.opt_index.is_empty()
                && !r.unit_index.is_empty()
                && !r.ent_index.is_empty(),
            "indices require official dummy rows"
        );
        ensure!(
            r.q_index
                .iter()
                .all(|v| v[0] < b && v[1] < s && v[2] <= v[3] && v[3] <= s),
            "invalid question index"
        );
        ensure!(
            r.opt_index
                .iter()
                .all(|v| v[0] < b && v[1] < s && v[2] < r.q_index.len()),
            "invalid option index"
        );
        ensure!(
            r.unit_index
                .iter()
                .chain(&r.ent_index)
                .all(|v| v[0] < b && v[1] < s),
            "invalid span index"
        );
        let c = &self.config.encoder_config;
        let (nh, h, d) = (
            c.num_attention_heads,
            c.hidden_size,
            c.hidden_size / c.num_attention_heads,
        );
        let native_mask = Tensor::from_vec(
            r.attention_mask
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (b, s),
            &Device::Cpu,
        )?;
        let mut profile = BTreeMap::new();
        let mut mark = Instant::now();
        // This lease is declared before all cached tensor views. Those views
        // leave scope at each layer boundary, and drop before the lease returns.
        let mut workspace = if self.options.projection_workspaces {
            Some(self.workspaces.checkout()?)
        } else {
            None
        };
        let mut x = self.embed(&r.input_ids)?.apply(&self.embed_norm)?;
        record(&mut profile, "embedding_norm", &mut mark, profile_enabled);
        let gemm_residual = blas::gemm_residual_enabled();
        for (i, l) in self.layers.iter().enumerate() {
            let residual = if self.options.residual_inplace {
                None
            } else {
                Some(x.clone())
            };
            let z = if let Some(norm) = &l.attn_norm {
                x.apply(norm)?
            } else {
                x.clone()
            };
            let qkv = if let Some(workspace) = workspace.as_mut() {
                l.qkv.forward_workspace(&z, workspace, 0)?
            } else {
                z.apply(&l.qkv)?
            };
            // Layer zero can read x directly. Its read-only alias is no longer
            // needed once eager QKV computation has completed.
            drop(z);
            let rope = if l.local {
                &self.local_rope
            } else {
                &self.global_rope
            };
            let (q, k, v) = if self.options.fused_qkv {
                let qkv = if self.options.qkv_rope_workspace {
                    workspace
                        .as_mut()
                        .unwrap()
                        .qkv_rope(&qkv, &rope.cos, &rope.sin, nh)?
                } else {
                    blas::qkv_rope(&qkv, &rope.cos, &rope.sin, nh)?
                };
                (qkv.i(0)?, qkv.i(1)?, qkv.i(2)?)
            } else {
                let qkv = qkv.reshape((b, s, 3, nh, d))?.permute((2, 0, 3, 1, 4))?;
                (
                    rope.apply(&qkv.i(0)?)?,
                    rope.apply(&qkv.i(1)?)?,
                    qkv.i(2)?.contiguous()?,
                )
            };
            record(&mut profile, "norm_qkv_rope", &mut mark, profile_enabled);
            let a = self.attention(&q, &k, &v, &r.attention_mask, &native_mask, l.local)?;
            record(
                &mut profile,
                if l.local {
                    "local_attention"
                } else {
                    "global_attention"
                },
                &mut mark,
                profile_enabled,
            );
            let a = if self.options.fused_qkv {
                blas::unpack_heads(&a)?
            } else {
                a.transpose(1, 2)?.contiguous()?.reshape((b, s, h))?
            };
            if self.options.residual_inplace {
                l.attn_out.update_private_residual(&a, &x)?;
            } else {
                x = l.attn_out.forward_with_residual(
                    &a,
                    residual.as_ref().unwrap(),
                    gemm_residual,
                )?;
            }
            drop(a);
            record(&mut profile, "attention_output", &mut mark, profile_enabled);
            let normed = x.apply(&l.mlp_norm)?;
            let z = if let Some(workspace) = workspace.as_mut() {
                l.wi.forward_workspace(&normed, workspace, 1)?
            } else {
                normed.apply(&l.wi)?
            };
            drop(normed);
            record(
                &mut profile,
                "mlp_norm_input_projection",
                &mut mark,
                profile_enabled,
            );
            let z = if self.options.fused_geglu {
                blas::geglu(&z)?
            } else {
                let z = z.chunk(2, D::Minus1)?;
                (&z[0].gelu_erf()? * &z[1])?
            };
            record(&mut profile, "geglu", &mut mark, profile_enabled);
            if self.options.residual_inplace {
                l.wo.update_private_residual(&z, &x)?;
            } else {
                x = l.wo.forward_with_residual(&z, &x, gemm_residual)?;
            }
            record(&mut profile, "mlp_output", &mut mark, profile_enabled);
            if !x.is_contiguous() {
                bail!("noncontiguous layer {i}");
            }
        }
        let x = x.apply(&self.final_norm)?;
        let flat = x.reshape((b * s, h))?;
        let gather = |indices: Vec<u32>| -> Result<Tensor> {
            let ids = Tensor::from_vec(indices.clone(), indices.len(), &Device::Cpu)?;
            Ok(flat.index_select(&ids, 0)?)
        };
        let ho = gather(
            r.opt_index
                .iter()
                .map(|v| (v[0] * s + v[1]) as u32)
                .collect(),
        )?;
        let hq = gather(r.q_index.iter().map(|v| (v[0] * s + v[1]) as u32).collect())?;
        // Linear-time prefix sums replace Candle cumsum's quadratic triangular matmul.
        // Accumulate in f64 then round prefixes to f32, matching torch CPU cumsum.
        let hv = x.flatten_all()?.to_vec1::<f32>()?;
        let pooled = pool_questions(&hv, b, s, h, &r.q_index)?;
        let pooled = Tensor::from_vec(pooled, (r.q_index.len(), h), &Device::Cpu)?;
        let v = normalize(pooled.apply(&self.norm)?.apply(&self.w_p)?)?;
        let uq = hq.apply(&self.norm)?.apply(&self.w_q)?;
        let own = Tensor::from_vec(
            r.opt_index.iter().map(|v| v[2] as u32).collect::<Vec<_>>(),
            r.opt_index.len(),
            &Device::Cpu,
        )?;
        let u = normalize((ho.apply(&self.norm)?.apply(&self.w_o)? + uq.index_select(&own, 0)?)?)?;
        let lg = ((u * v.index_select(&own, 0)?)?.sum(D::Minus1)? / self.tau)?;
        let lg = (lg
            + ho.apply(&self.mlp1)?
                .relu()?
                .apply(&self.mlp2)?
                .squeeze(1)?)?;
        let ht = gather(
            r.unit_index
                .iter()
                .map(|v| (v[0] * s + v[1]) as u32)
                .collect(),
        )?;
        let he = gather(
            r.ent_index
                .iter()
                .map(|v| (v[0] * s + v[1]) as u32)
                .collect(),
        )?;
        let t = normalize(ht.apply(&self.norm)?.apply(&self.w_t)?)?;
        let e = normalize(he.apply(&self.norm)?.apply(&self.w_e)?)?;
        let span = (t.matmul(&e.t()?)? / self.tau_span)?;
        record(&mut profile, "final_norm_heads", &mut mark, profile_enabled);
        Ok((lg, span, profile))
    }
}
fn normalize(x: Tensor) -> Result<Tensor> {
    let n = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(1e-12, f32::MAX as f64)?;
    Ok(x.broadcast_div(&n)?)
}
fn record(p: &mut BTreeMap<String, f64>, name: &str, t: &mut Instant, on: bool) {
    if on {
        *p.entry(name.into()).or_default() += t.elapsed().as_secs_f64();
    }
    *t = Instant::now();
}

fn pool_questions(
    hidden: &[f32],
    batch: usize,
    seq: usize,
    width: usize,
    questions: &[[usize; 4]],
) -> Result<Vec<f32>> {
    let size = batch.checked_mul(seq).and_then(|v| v.checked_mul(width));
    ensure!(size == Some(hidden.len()), "pooled hidden shape");
    let mut rows = vec![Vec::new(); batch];
    for (i, &[row, _, start, end]) in questions.iter().enumerate() {
        ensure!(
            row < batch && start <= end && end <= seq,
            "pooled question index"
        );
        rows[row].push((i, start, end));
    }
    let mut output = vec![0f32; questions.len() * width];
    for (row, items) in rows
        .iter()
        .enumerate()
        .filter(|(_, items)| !items.is_empty())
    {
        let mut endpoints = BTreeMap::<usize, Vec<(usize, bool)>>::new();
        for &(i, start, end) in items {
            // Start precedes end for an empty interval, preserving exact zero.
            endpoints.entry(start).or_default().push((i, false));
            endpoints.entry(end).or_default().push((i, true));
        }
        let last = *endpoints.keys().next_back().unwrap();
        let mut prefix = vec![0f64; width];
        for position in 0..=last {
            if let Some(events) = endpoints.get(&position) {
                for &(i, is_end) in events {
                    let dest = &mut output[i * width..(i + 1) * width];
                    if is_end {
                        let count = (questions[i][3] - questions[i][2]).max(1) as f32;
                        for (out, &sum) in dest.iter_mut().zip(&prefix) {
                            *out = ((sum as f32) - *out) / count;
                        }
                    } else {
                        for (out, &sum) in dest.iter_mut().zip(&prefix) {
                            *out = sum as f32;
                        }
                    }
                }
            }
            if position < last {
                let values =
                    &hidden[(row * seq + position) * width..(row * seq + position + 1) * width];
                // Contiguous independent lanes preserve each lane's original
                // F64 addition order and round only requested F32 prefixes.
                for (sum, &value) in prefix.iter_mut().zip(values) {
                    *sum += value as f64;
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    #[test]
    fn private_residual_updates_reject_default_unpacked_options_before_loading() {
        let options = ModelOptions {
            residual_inplace: true,
            ..ModelOptions::default()
        };
        let error = match Model::load("/nonexistent-residual-contract-model", options) {
            Ok(_) => panic!("incompatible options loaded"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("private residual updates require"));
        let options = ModelOptions {
            inplace_softmax: true,
            ..ModelOptions::default()
        };
        let error = Model::load("/nonexistent/vela-private-softmax-test", options)
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("private softmax requires native attention"));
    }

    #[test]
    fn residual_flag_preserves_unpacked_projection_and_bias() {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![1f32, 2., -3., 1., 0.5, -2.], (3, 2), &device).unwrap();
        let x = Tensor::from_vec(vec![1f32, 2., -1., 3.], (2, 2), &device).unwrap();
        let residual =
            Tensor::from_vec(vec![0.25f32, 1., 2., -2., 0., 3.], (2, 3), &device).unwrap();
        for bias in [
            None,
            Some(Tensor::from_vec(vec![0.5f32, 1., -2.], 3, &device).unwrap()),
        ] {
            let has_bias = bias.is_some();
            let projection = Projection {
                inner: Some(Linear::new(weight.clone(), bias)),
                blas: false,
                fixed: None,
                fixed_bias: None,
            };
            // This is the representation produced by ModelOptions::default(),
            // even when another packed model enabled fusion process-wide.
            let result = projection
                .forward_with_residual(&x, &residual, true)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let expected = if has_bias {
                vec![vec![5.75, 1., -3.5], vec![3.5, 7., -5.5]]
            } else {
                vec![vec![5.25, 0., -1.5], vec![3., 6., -3.5]]
            };
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn pooled_prefixes_preserve_rounding_empty_intervals_and_shared_endpoints() {
        let hidden = vec![
            16777216f32,
            1e-7,
            -0.1,
            1.,
            0.2,
            0.3,
            -16777216.,
            -0.5,
            0.7,
            1.5,
            0.2,
            -0.9,
            -3.,
            4.,
            5.,
            2.,
            -2.,
            1.,
            1.,
            8.,
            -2.,
            5.,
            -1.,
            0.,
        ];
        let questions = vec![
            [0, 0, 1, 2],
            [0, 0, 0, 3],
            [0, 0, 2, 2],
            [1, 0, 0, 4],
            [1, 0, 1, 3],
            [0, 0, 1, 4],
            [1, 0, 1, 3],
        ];
        let actual = pool_questions(&hidden, 2, 4, 3, &questions).unwrap();
        let mut released = vec![0f32; questions.len() * 3];
        for (i, &[row, _, start, end]) in questions.iter().enumerate() {
            for lane in 0..3 {
                let mut sum = 0f64;
                let mut lower = 0f32;
                for position in 0..end {
                    if position == start {
                        lower = sum as f32;
                    }
                    sum += hidden[(row * 4 + position) * 3 + lane] as f64;
                }
                if start == end {
                    lower = sum as f32;
                }
                released[i * 3 + lane] = ((sum as f32) - lower) / (end - start).max(1) as f32;
            }
        }
        assert!(actual
            .iter()
            .zip(&released)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        // Rounding prefixes before subtraction differs from directly averaging
        // this single-token interval, whose original hidden value is 1.
        assert_eq!(actual[0], 0.);
        assert_eq!(actual[3], 1f32 / 3.);
        assert_eq!(&actual[6..9], &[0f32, 0., 0.]);
    }
}
