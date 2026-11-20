//! The ModernBERT encoder on [`OnednnCpu`], and its injection head.
//!
//! [`Precision::Fp32`]: every matmul is FP32 against FP32 packed weights, with
//! a fused GeGLU. [`Precision::Bf16`]: the dense projections (QKV, attention
//! output, both GeGLU halves, MLP output) run as BF16 matmuls with FP32
//! accumulation; LayerNorm, attention (QK, softmax, PV), RoPE, GeGLU and the
//! residual stream stay FP32. The checkpoint is FP32 either way (BF16 weights
//! are rounded to nearest even at pack time). The token table is a zero-copy
//! mmap of the safetensors file.
//!
//! BF16 follows EmbeddingGemma 2 on non-AMX ISAs (AVX512_CORE_BF16):
//! activations are rounded to BF16 once per projection and handed to the
//! matmuls pre-converted, and attention is the backend's BF16 attention. On AMX
//! activations go in as FP32 and attention runs in FP32, which was faster than
//! the native path there (91 vs 109 ms at 512 tokens). On a host that loses AMX
//! tile state when a thread is preempted the runtime does not use AMX at all.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::cpu::kernels::{add_inplace, f32_slice_to_bf16_nthreads, for_ranges, layout_qkv_rope};
use crate::cpu::onednn::{
    file_stamp, CacheOptions, CacheWriter, OnednnCpu, PackedWeight, RawTensor,
};
use crate::cpu::runtime::{BackendStats, CacheWrite, Isa, LoadReport, PackedWeights, WeightSource};
use crate::guard::config::EncoderConfig;
use crate::guard::head::InjectionHead;
use crate::guard::load::{
    cache_file_for, cache_meta, pack_from_file, read_cache, take_layers, take_norm,
};
use crate::tensor_file::TensorFile;
use crate::{check_shape, Result, ResultContext as Context};

/// What the guard's matmuls compute in on a CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Precision {
    /// Every matmul FP32 against FP32 weights, with a fused GeGLU: matches
    /// PyTorch FP32. The default on every CPU.
    Fp32,
    /// The dense projections (QKV, attention output, both MLP matmuls) take
    /// BF16 inputs with FP32 accumulation; attention, norms and the head
    /// stay FP32. Faster, and on the public guard's gate it keeps every
    /// decision. Only where the CPU has BF16 matmul kernels.
    Bf16,
}

crate::cpu::runtime::precision_names!(Precision);

impl Precision {
    /// The precisions the guard runs on `isa`, its default first.
    pub(crate) fn on(isa: Isa) -> &'static [Self] {
        if isa.has_bf16_matmul() {
            &[Self::Fp32, Self::Bf16]
        } else {
            &[Self::Fp32]
        }
    }
}

const EMB_NAME: &str = "model.embeddings.tok_embeddings.weight";

/// Zero-copy token table: a view of the mapped safetensors file, which the
/// encoder keeps for [`Encoder::tensor`]. Only gathered rows become resident.
struct Embeddings {
    file: TensorFile,
    vocab: usize,
    hidden: usize,
}

impl Embeddings {
    fn new(file: TensorFile, vocab: usize, hidden: usize) -> Result<Self> {
        file.view::<f32>(EMB_NAME, &[vocab, hidden])?;
        Ok(Self {
            file,
            vocab,
            hidden,
        })
    }

    fn table(&self) -> &[f32] {
        // Checked in `new`.
        self.file
            .view(EMB_NAME, &[self.vocab, self.hidden])
            .unwrap_or(&[])
    }
}

struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    /// Default (non-scaled) RoPE tables `[len, d/2]`, computed in f32 like
    /// Hugging Face's `ModernBertRotaryEmbedding`.
    fn new(len: usize, d: usize, theta: f64) -> Result<Self> {
        let half = d / 2;
        let size = crate::checked_size("RoPE table", &[len, half])?;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / d as f32))
            .collect();
        let mut cos = crate::try_vec::<f32>(size)?;
        let mut sin = crate::try_vec::<f32>(size)?;
        for p in 0..len {
            for (i, f) in inv.iter().enumerate() {
                let phase = p as f32 * f;
                cos[p * half + i] = phase.cos();
                sin[p * half + i] = phase.sin();
            }
        }
        Ok(Self { cos, sin })
    }
}

pub(crate) struct Layer {
    /// `None` on layer 0 (identity, as in Hugging Face).
    pub(crate) attn_norm: Option<Vec<f32>>,
    pub(crate) wqkv: PackedWeight,
    pub(crate) wo: PackedWeight,
    pub(crate) mlp_norm: Vec<f32>,
    /// First (activated) and second (gate) halves of `mlp.Wi`.
    pub(crate) wi_left: PackedWeight,
    pub(crate) wi_right: PackedWeight,
    pub(crate) mlp_wo: PackedWeight,
}

/// The padded batch one forward runs over: `b` rows of `s` positions,
/// `tokens = b * s`.
#[derive(Clone, Copy)]
struct Batch {
    b: usize,
    s: usize,
    tokens: usize,
}

/// Final hidden states of a padded batch.
#[derive(Clone, Debug)]
pub struct Hidden {
    hidden: usize,
    seq: usize,
    lens: Vec<usize>,
    data: Vec<f32>,
}

impl Hidden {
    /// Number of sequences.
    pub fn rows(&self) -> usize {
        self.lens.len()
    }

    /// Hidden states of sequence `i`: `len × hidden` values, padding excluded.
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.seq * self.hidden;
        &self.data[start..start + self.lens[i] * self.hidden]
    }
}

/// How encoder batches are right-padded.
///
/// The oneDNN backend builds execution plans per shape, and the token count is
/// part of the shape, so every distinct padded width costs ~8-14 new plans
/// (~5 ms each). With exact widths, traffic of many different lengths keeps
/// the bounded plan cache thrashing (measured on realistic single queries:
/// 75% of calls rebuilt plans, p50 ~90 ms vs ~40 ms warm). Padding to a
/// bounded set of widths keeps every plan cached. Pad keys are masked, so
/// outputs change only by float reassociation (~1e-6 in FP32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    /// The default: eight widths per power of two, at least 8 apart: lengths
    /// in `(2^k, 2^(k+1)]` round up to a multiple of `max(2^k / 8, 8)` (so up
    /// to 128 tokens: multiples of 8; then 16 to 256, 32 to 512, 64 to
    /// 1024). That is under 8 pad tokens up to 64 tokens, then under 12.5% of
    /// the row (a fixed multiple of 32 padded 20 tokens to 32 and 134 to 160;
    /// buckets pad them to 24 and 144). 40 widths for a 1024-token encoder,
    /// 32 for 512.
    Buckets,
    /// Round up to a multiple of this many tokens; `1` pads only to the
    /// longest row (every distinct length is a new shape).
    Multiple(usize),
}

/// Padded widths per power of two of [`Padding::Buckets`].
pub(crate) const BUCKETS_PER_OCTAVE: usize = 8;
/// Smallest gap between two [`Padding::Buckets`] widths.
pub(crate) const MIN_BUCKET_STEP: usize = 8;

/// `len` rounded up to a multiple of `multiple` (`0` or `1`: unchanged), at
/// most `max_len`. `len` itself must be `<= max_len`.
pub(crate) fn pad_width(len: usize, multiple: usize, max_len: usize) -> usize {
    if multiple <= 1 {
        return len;
    }
    len.div_ceil(multiple)
        .saturating_mul(multiple)
        .min(max_len)
        .max(len)
}

/// `len` rounded up to the next [`Padding::Buckets`] width, at most
/// `max_len`: lengths in `(2^k, 2^(k+1)]` round up to a multiple of
/// `max(2^k / BUCKETS_PER_OCTAVE, MIN_BUCKET_STEP)` (so up to 128 tokens:
/// multiples of 8; then 16 to 256, 32 to 512, 64 to 1024).
pub(crate) fn bucket_width(len: usize, max_len: usize) -> usize {
    if len <= MIN_BUCKET_STEP {
        return pad_width(len, MIN_BUCKET_STEP, max_len);
    }
    let octave = 1usize << (usize::BITS - 1 - (len - 1).leading_zeros());
    let step = (octave / BUCKETS_PER_OCTAVE).max(MIN_BUCKET_STEP);
    pad_width(len, step, max_len)
}

impl Padding {
    /// Width a batch whose longest row has `len` tokens is padded to.
    pub fn width(self, len: usize, max_len: usize) -> usize {
        match self {
            Self::Buckets => bucket_width(len, max_len),
            Self::Multiple(n) => pad_width(len, n, max_len),
        }
    }

    /// Whether every length is its own width.
    pub fn is_exact(self) -> bool {
        matches!(self, Self::Multiple(0 | 1))
    }
}

/// The encoder: weights, RoPE tables and the injection head.
pub(crate) struct Encoder {
    backend: Arc<OnednnCpu>,
    cfg: EncoderConfig,
    precision: Precision,
    /// BF16 on a non-AMX ISA: pre-converted BF16 activations and BF16
    /// attention. (AMX takes FP32 activations and FP32 attention instead.)
    native_bf16: bool,
    tile: usize,
    max_len: usize,
    /// How batches are padded (see [`Padding`]).
    padding: Padding,
    embeddings: Embeddings,
    emb_norm: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    head: InjectionHead,
    /// Bias-free LayerNorm runs as a LayerNorm with a zero shift.
    zero_shift: Vec<f32>,
    global_rope: Rope,
    local_rope: Rope,
    load_source: WeightSource,
    load_profile: BTreeMap<String, f64>,
}

impl Encoder {
    /// Load the encoder and its injection head from a model directory.
    ///
    /// `tile` is the attention query tile; `max_len` bounds sequence length
    /// (and sizes the RoPE tables); `sha256` is what `model.safetensors`
    /// must hash to, which keys the packed-weight cache.
    pub(crate) fn load(
        dir: &Path,
        backend: Arc<OnednnCpu>,
        tile: usize,
        cache: &CacheOptions,
        precision: Precision,
        max_len: usize,
        sha256: &str,
    ) -> Result<Self> {
        Self::load_impl(dir, backend, tile, cache, precision, max_len, sha256)
            .map_err(|e| e.at(dir))
    }

    fn load_impl(
        dir: &Path,
        backend: Arc<OnednnCpu>,
        tile: usize,
        cache: &CacheOptions,
        precision: Precision,
        max_len: usize,
        sha256: &str,
    ) -> Result<Self> {
        check_shape!(tile > 0, "tile must be > 0");
        let cfg = EncoderConfig::from_json(&crate::read_json(dir.join("config.json"))?)?;
        let positions = cfg.positions;
        if !(2..=positions).contains(&max_len) {
            return Err(crate::Error::config(format!(
                "max_len {max_len} outside 2..={positions}"
            )));
        }
        let (h, inter) = (cfg.hidden, cfg.intermediate);
        crate::checked_size("guard largest projection", &[h, inter, 2, 4])?;

        let st_path = dir.join("model.safetensors");
        let cache_file = cache_file_for(&st_path, backend.cpu_isa(), cache, precision);
        let mut load_profile = BTreeMap::new();
        let file = TensorFile::open(&st_path)?;
        if cache.verify_weights {
            let t = Instant::now();
            crate::assets::verify_sha256_mmap(&st_path, sha256)?;
            load_profile.insert("verify_s".into(), t.elapsed().as_secs_f64());
        }

        let mut linears: HashMap<String, PackedWeight> = HashMap::new();
        let mut raw: Option<Vec<RawTensor>> = None;
        let mut load_source = None;
        let stamp = file_stamp(&st_path).with_context(|| st_path.display().to_string())?;
        if cache.enabled {
            let t = Instant::now();
            raw = read_cache(
                &backend,
                precision,
                &cache_file,
                sha256,
                stamp,
                &mut linears,
            );
            load_profile.insert("cache_read_s".into(), t.elapsed().as_secs_f64());
            if raw.is_some() {
                load_source = Some(WeightSource::Cache {
                    path: cache_file.clone(),
                });
            }
        }
        let raw = match raw {
            Some(raw) => raw,
            None => {
                let t = Instant::now();
                let writer = cache.enabled.then(|| {
                    let meta = cache_meta(sha256.to_string(), stamp, backend.cpu_isa());
                    CacheWriter::start(&cache_file, &meta)
                });
                let (raw, writer) =
                    pack_from_file(&backend, precision, &cfg, &file, writer, &mut linears)?;
                load_profile.insert("pack_s".into(), t.elapsed().as_secs_f64());
                let cache = match writer {
                    Some(w) => {
                        let t = Instant::now();
                        let written = w.close(&raw);
                        load_profile.insert("cache_write_s".into(), t.elapsed().as_secs_f64());
                        written
                    }
                    None => CacheWrite::Off,
                };
                load_source = Some(WeightSource::Safetensors { cache });
                raw
            }
        };
        let mut small: HashMap<String, Vec<f32>> =
            raw.into_iter().map(|t| (t.name, t.data)).collect();
        let emb_norm = take_norm(&mut small, "model.embeddings.norm.weight", h)?;
        let final_norm = take_norm(&mut small, "model.final_norm.weight", h)?;
        let layers = take_layers(&cfg, &mut linears, &mut small)?;
        let head = InjectionHead::from_tensors(&mut small, h, cfg.eps)?;
        let embeddings = Embeddings::new(file, cfg.vocab, h)?;
        let global_rope = Rope::new(max_len, cfg.head_dim, cfg.global_theta)?;
        let local_rope = Rope::new(max_len, cfg.head_dim, cfg.local_theta)?;
        let backend_isa = backend.cpu_isa();
        Ok(Self {
            backend,
            cfg,
            precision,
            native_bf16: precision == Precision::Bf16
                && (!crate::cpu::onednn::isa_has_amx(backend_isa)),
            tile,
            max_len,
            padding: Padding::Buckets,
            embeddings,
            emb_norm,
            layers,
            final_norm,
            head,
            zero_shift: vec![0.0; h],
            global_rope,
            local_rope,
            load_source: load_source
                .ok_or_else(|| crate::Error::internal("guard load set no weight source"))?,
            load_profile,
        })
    }

    pub(crate) fn precision(&self) -> Precision {
        self.precision
    }

    pub(crate) fn config(&self) -> &EncoderConfig {
        &self.cfg
    }

    pub(crate) fn max_len(&self) -> usize {
        self.max_len
    }

    /// Pool threads the backend's operators run on.
    pub(crate) fn threads(&self) -> usize {
        self.backend.threads()
    }

    /// How batches are right-padded (default [`Padding::Buckets`]).
    pub(crate) fn padding(&self) -> Padding {
        self.padding
    }

    /// Set [`Encoder::padding`] (a multiple must be `>= 1`).
    pub(crate) fn set_padding(&mut self, padding: Padding) -> Result<()> {
        check_shape!(padding != Padding::Multiple(0), "pad multiple must be >= 1");
        self.padding = padding;
        Ok(())
    }

    /// Width a batch whose longest row has `len` tokens is padded to.
    pub(crate) fn padded_len(&self, len: usize) -> usize {
        self.padding.width(len, self.max_len)
    }

    /// Every padded width a batch can have, ascending.
    pub(crate) fn padded_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = (1..=self.max_len).map(|l| self.padded_len(l)).collect();
        widths.dedup();
        widths
    }

    /// Injection logit `l[INJECTION] - l[SAFE]` of one sequence's hidden
    /// states (`len × hidden`, padding excluded).
    pub(crate) fn injection_logit(&self, hidden: &[f32]) -> f32 {
        self.head.logit(hidden)
    }

    /// An FP32 tensor of exactly `shape` from the mapped checkpoint.
    pub(crate) fn tensor(&self, name: &str, shape: &[usize]) -> Result<&[f32]> {
        self.embeddings.file.view(name, shape)
    }

    pub(crate) fn backend_stats(&self) -> BackendStats {
        self.backend.stats()
    }

    /// Where the packed weights came from, and the load's phases.
    pub(crate) fn load_report(&self) -> LoadReport {
        LoadReport {
            packed: vec![PackedWeights {
                part: "encoder",
                source: self.load_source.clone(),
            }],
            phases_s: self.load_profile.clone(),
        }
    }

    /// Whether the mapped checkpoint has a tensor named `name`.
    pub(crate) fn has_tensor(&self, name: &str) -> bool {
        self.embeddings.file.index().shape(name).is_ok()
    }

    fn layer_norm(&self, x: &[f32], scale: &[f32]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        check_shape!(x.len().is_multiple_of(h) && scale.len() == h, "norm matrix");
        self.backend
            .layernorm(x.len() / h, h, x, scale, &self.zero_shift, self.cfg.eps)
    }

    /// `x @ W^T` for FP32 `x`, in the model's precision.
    fn linear(&self, x: &[f32], w: &PackedWeight) -> Result<Vec<f32>> {
        check_shape!(x.len().is_multiple_of(w.k()), "linear matrix");
        let mut y = self.backend.buffer(crate::checked_size(
            "activation elements",
            &[x.len() / w.k(), w.n()],
        )?)?;
        if self.precision == Precision::Bf16 {
            self.backend.linear_bf16(x, w, &mut y, false)?;
        } else {
            self.backend.linear(x, w, &mut y, false)?;
        }
        Ok(y)
    }

    fn to_bf16(&self, x: &[f32]) -> Result<Vec<u16>> {
        let mut y = self.backend.buffer_u16(x.len())?;
        f32_slice_to_bf16_nthreads(x, &mut y, self.backend.threads())?;
        Ok(y)
    }

    /// `x @ W^T` for an activation already in BF16 (`native_bf16` only).
    fn linear_src(&self, x: &[u16], w: &PackedWeight) -> Result<Vec<f32>> {
        check_shape!(x.len().is_multiple_of(w.k()), "linear matrix");
        let mut y = self.backend.buffer(crate::checked_size(
            "activation elements",
            &[x.len() / w.k(), w.n()],
        )?)?;
        self.backend.linear_bf16_src(x, w, &mut y, false)?;
        Ok(y)
    }

    /// `y += x @ W^T`.
    fn linear_add(&self, x: &[f32], w: &PackedWeight, y: &mut [f32]) -> Result<()> {
        if self.precision == Precision::Bf16 {
            let o = self.linear(x, w)?;
            add_inplace(y, &o, self.backend.threads())?;
            self.backend.recycle(o);
            Ok(())
        } else {
            self.backend.linear(x, w, y, true)
        }
    }

    fn embed(&self, ids: &[u32], tokens: usize) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        check_shape!(ids.len() == tokens, "token count");
        check_shape!(
            ids.iter().all(|&id| (id as usize) < self.cfg.vocab),
            "token id outside the vocabulary"
        );
        let mut x = self
            .backend
            .buffer(crate::checked_size("activation elements", &[tokens, h])?)?;
        let table = self.embeddings.table();
        let (x_p, t_p, t_len) = (
            x.as_mut_ptr() as usize,
            table.as_ptr() as usize,
            table.len(),
        );
        for_ranges(self.backend.threads(), tokens, 8, |lo, hi| {
            // SAFETY: ranges are disjoint rows of `x`; ids were bounds-checked
            // against the table above; both buffers outlive this call.
            let x = unsafe { std::slice::from_raw_parts_mut(x_p as *mut f32, tokens * h) };
            // SAFETY: a shared view of the mapped table, which outlives this
            // call and is never written.
            let table = unsafe { std::slice::from_raw_parts(t_p as *const f32, t_len) };
            for t in lo..hi {
                let id = ids[t] as usize;
                x[t * h..(t + 1) * h].copy_from_slice(&table[id * h..(id + 1) * h]);
            }
        });
        Ok(x)
    }

    /// One layer's attention, added into the residual stream `x`.
    fn attention_block(
        &self,
        layer: &Layer,
        global: bool,
        shape: Batch,
        padding: &[Vec<usize>],
        x: &mut [f32],
    ) -> Result<()> {
        let Batch { b, s, tokens } = shape;
        let (h, nh, d) = (self.cfg.hidden, self.cfg.heads, self.cfg.head_dim);
        let normed = match &layer.attn_norm {
            Some(w) => Some(self.layer_norm(x, w)?),
            None => None,
        };
        let input = normed.as_deref().unwrap_or(x);
        let qkv = if self.native_bf16 {
            let bits = self.to_bf16(input)?;
            let y = self.linear_src(&bits, &layer.wqkv)?;
            self.backend.recycle_u16(bits);
            y
        } else {
            self.linear(input, &layer.wqkv)?
        };
        if let Some(n) = normed {
            self.backend.recycle(n);
        }
        let rope = if global {
            &self.global_rope
        } else {
            &self.local_rope
        };
        let mut q = self.backend.buffer(tokens * h)?;
        let mut k = self.backend.buffer(tokens * h)?;
        let mut v = self.backend.buffer(tokens * h)?;
        layout_qkv_rope(
            &qkv[..tokens * 3 * h],
            &mut q,
            &mut k,
            &mut v,
            b,
            s,
            nh,
            d,
            &rope.cos,
            &rope.sin,
            (d as f32).sqrt().recip(),
            self.backend.threads(),
        )?;
        self.backend.recycle(qkv);
        let radius = self.cfg.local_radius;
        let attn = if self.native_bf16 {
            // Q is already scaled; `attention_gqa` rounds Q/K/V to BF16.
            // Its output carries zero rows of backend padding.
            let mut a = self.backend.attention_gqa(
                &q, &k, &v, b, s, nh, nh, d, self.tile, !global, radius, 1.0, padding,
            )?;
            a.truncate(tokens * h);
            a
        } else {
            self.backend
                .attention(&q, &k, &v, b, s, nh, d, self.tile, !global, radius, padding)?
        };
        self.backend.recycle(q);
        self.backend.recycle(k);
        self.backend.recycle(v);
        self.linear_add(&attn, &layer.wo, x)?;
        self.backend.recycle(attn);
        Ok(())
    }

    /// One layer's GeGLU MLP, `gelu(x @ W_left) * (x @ W_right)`, added into
    /// the residual stream `x`.
    fn mlp_block(&self, layer: &Layer, tokens: usize, x: &mut [f32]) -> Result<()> {
        let inter = self.cfg.intermediate;
        let normed = self.layer_norm(x, &layer.mlp_norm)?;
        let gated = if self.precision == Precision::Bf16 {
            let (left, right) = if self.native_bf16 {
                let bits = self.to_bf16(&normed)?;
                let l = self.linear_src(&bits, &layer.wi_left)?;
                let r = self.linear_src(&bits, &layer.wi_right)?;
                self.backend.recycle_u16(bits);
                (l, r)
            } else {
                (
                    self.linear(&normed, &layer.wi_left)?,
                    self.linear(&normed, &layer.wi_right)?,
                )
            };
            let act = self.backend.gelu_erf(tokens, inter, &left)?;
            self.backend.recycle(left);
            let gated = self.backend.mul(tokens, inter, &act, &right)?;
            self.backend.recycle(act);
            self.backend.recycle(right);
            gated
        } else {
            let right = self.linear(&normed, &layer.wi_right)?;
            let gated = self.backend.gated_linear(&normed, &layer.wi_left, &right)?;
            self.backend.recycle(right);
            gated
        };
        self.backend.recycle(normed);
        self.linear_add(&gated, &layer.mlp_wo, x)?;
        self.backend.recycle(gated);
        Ok(())
    }

    /// Run the encoder over `rows`, right-padded to [`Encoder::padded_len`] of
    /// the longest. Every row is attended in full: pad keys are masked, so a
    /// row's output does not depend on the other rows or the padded width.
    pub(crate) fn forward(&self, rows: &[&[u32]]) -> Result<Hidden> {
        check_shape!(!rows.is_empty(), "empty batch");
        let lens: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        let longest = lens.iter().copied().max().unwrap_or(0);
        check_shape!(
            lens.iter().all(|&l| l > 0) && longest <= self.max_len,
            "sequence lengths must be in 1..={}",
            self.max_len
        );
        let s = self.padded_len(longest);
        let b = rows.len();
        let (h, inter) = (self.cfg.hidden, self.cfg.intermediate);
        let tokens = crate::checked_size("guard tokens", &[b, s])?;
        self.backend.begin_shape(b, s);
        let per_token = crate::checked_size("guard row bytes", &[h * 10 + inter * 3, 4])?;
        self.backend.validate_workspace(crate::checked_size(
            "guard activations",
            &[tokens, per_token],
        )?)?;
        let mut ids = crate::try_vec::<u32>(tokens)?;
        ids.fill(self.cfg.pad_id);
        let mut padding = Vec::with_capacity(b);
        for (i, row) in rows.iter().enumerate() {
            ids[i * s..i * s + row.len()].copy_from_slice(row);
            padding.push((row.len()..s).collect::<Vec<_>>());
        }
        let emb = self.embed(&ids, tokens)?;
        let mut x = self.layer_norm(&emb, &self.emb_norm)?;
        self.backend.recycle(emb);
        for (i, layer) in self.layers.iter().enumerate() {
            let shape = Batch { b, s, tokens };
            self.attention_block(layer, self.cfg.global[i], shape, &padding, &mut x)?;
            self.mlp_block(layer, tokens, &mut x)?;
        }
        let out = self.layer_norm(&x, &self.final_norm)?;
        self.backend.recycle(x);
        check_shape!(out.iter().all(|v| v.is_finite()), "nonfinite hidden states");
        Ok(Hidden {
            hidden: h,
            seq: s,
            lens,
            data: out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guard::load::small_tensor_names;

    #[test]
    fn pad_width_rounds_up_and_caps_at_max_len() {
        assert_eq!(pad_width(1, 32, 512), 32);
        assert_eq!(pad_width(32, 32, 512), 32);
        assert_eq!(pad_width(33, 32, 512), 64);
        assert_eq!(pad_width(500, 32, 512), 512);
        assert_eq!(pad_width(500, 32, 510), 510);
        assert_eq!(pad_width(510, 32, 510), 510);
        assert_eq!(pad_width(77, 1, 512), 77);
        assert_eq!(pad_width(77, 0, 512), 77);
        let widths: Vec<usize> = {
            let mut w: Vec<usize> = (1..=512).map(|l| pad_width(l, 32, 512)).collect();
            w.dedup();
            w
        };
        assert_eq!(widths.len(), 16);
        assert_eq!((widths[0], widths[15]), (32, 512));
    }

    fn widths(padding: Padding, max_len: usize) -> Vec<usize> {
        let mut w: Vec<usize> = (1..=max_len).map(|l| padding.width(l, max_len)).collect();
        w.dedup();
        w
    }

    #[test]
    fn bucket_width_bounds_padding_and_width_count() {
        let w = |l| bucket_width(l, 1024);
        assert_eq!(
            (w(1), w(8), w(9), w(20), w(60), w(64)),
            (8, 8, 16, 24, 64, 64)
        );
        assert_eq!(
            (w(65), w(128), w(129), w(134), w(250)),
            (72, 128, 144, 144, 256)
        );
        assert_eq!(
            (w(257), w(500), w(513), w(1000), w(1024)),
            (288, 512, 576, 1024, 1024)
        );
        assert_eq!(bucket_width(500, 510), 510);
        assert_eq!(bucket_width(510, 510), 510);
        for max_len in [512, 1024, 510] {
            let mut last = 0;
            for len in 1..=max_len {
                let width = bucket_width(len, max_len);
                assert!(width >= len && width >= last && width <= max_len);
                // Under 8 pad tokens up to 64 tokens, then under 12.5%.
                if width < max_len {
                    assert!(
                        width - len < MIN_BUCKET_STEP.max(len / 8),
                        "{len} -> {width}"
                    );
                }
                last = width;
            }
        }
        let b = widths(Padding::Buckets, 1024);
        assert_eq!(b.len(), 40);
        assert_eq!((b[0], b[39]), (8, 1024));
        assert_eq!(widths(Padding::Buckets, 512).len(), 32);
        assert_eq!(widths(Padding::Multiple(32), 1024).len(), 32);
        assert_eq!(widths(Padding::Multiple(1), 100).len(), 100);
        assert!(Padding::Multiple(1).is_exact() && !Padding::Buckets.is_exact());
    }

    #[test]
    fn rope_rotates_by_position_over_theta_powers() {
        let r = Rope::new(4, 4, 100.0).unwrap();
        // Position 0 is the identity; frequencies are 1 and 100^-0.5.
        assert_eq!((r.cos[0], r.sin[0]), (1.0, 0.0));
        assert!((r.sin[2] - 1f32.sin()).abs() < 1e-6, "pos 1, freq 0");
        assert!((r.sin[3] - 0.1f32.sin()).abs() < 1e-6, "pos 1, freq 1");
        assert_eq!(r.cos.len(), 4 * 2);
    }

    #[test]
    fn small_tensor_list_skips_layer_zero_attn_norm() {
        let names = small_tensor_names(3);
        assert!(!names.contains(&"model.layers.0.attn_norm.weight".to_string()));
        assert!(names.contains(&"model.layers.2.attn_norm.weight".to_string()));
        assert!(names.contains(&"model.layers.0.mlp_norm.weight".to_string()));
        assert_eq!(names.len(), 6 + 5);
    }
}
