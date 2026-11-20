//! Public EmbeddingGemma 2 API: load from a [`crate::cpu::runtime::Runtime`], share
//! across threads.
//!
//! ```ignore
//! use crate::embeddinggemma2::api::{Config, EmbeddingGemma2, PromptKind};
//! let model: EmbeddingGemma2 = rt.load(Config::new("/models/embeddinggemma2"))?;
//! let vectors = model.embed(["what is sinteric?"], PromptKind::Query)?;
//! ```
//!
//! Images and audio are opt-in at load ([`Config::with_image`],
//! [`Config::with_audio`]) and embed through [`EmbeddingGemma2::embed_parts`]:
//!
//! ```ignore
//! use crate::embeddinggemma2::api::{Config, EmbeddingGemma2, ImageView, Part, PixelFormat};
//! let model: EmbeddingGemma2 = rt.load(Config::new(dir).with_image(true).with_audio(true))?;
//! let photo = ImageView::new(&rgb, width, height, width as usize * 3, PixelFormat::Rgb8)?;
//! let vectors = model.embed_parts(
//!     &[&[Part::Text("a photo of"), Part::Image(photo)], &[Part::Audio(&samples_16k)]],
//!     None,
//! )?;
//! ```

use std::path::PathBuf;

use crate::cpu::onednn::CacheOptions;
use crate::cpu::runtime::{BackendOverrides, CacheConfig, Runtime};
use serde::{Deserialize, Serialize};

use crate::embeddinggemma2::{mrl_truncate, Error, Model, Result, EMBED_DIM, MAX_SEQ, MRL_DIMS};

pub use crate::embeddinggemma2::model::Precision;

pub use crate::embeddinggemma2::config::EMBED_DIM as EMBEDDING_DIM;
pub use crate::embeddinggemma2::frontend::{Batch, Frontend, PromptKind};
pub use crate::embeddinggemma2::media::{ImageTokens, MAX_SAMPLES, SAMPLE_RATE};
pub use crate::image::{ImageView, PixelFormat};

use crate::embeddinggemma2::config::{
    AUDIO_CLOSE_ID, AUDIO_OPEN_ID, AUDIO_TOKEN_ID, BOS_ID, EOS_ID, IMAGE_CLOSE_ID, IMAGE_OPEN_ID,
    IMAGE_TOKEN_ID,
};
use crate::embeddinggemma2::media;

/// Hidden EG2 knobs (attention tile and load warmup). Prefer omitting this.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct Advanced {
    /// Attention tile. `None` uses the measured per-ISA table (256 on AMX
    /// and AVX512_CORE_BF16). Hidden experimental override for new CPUs.
    #[serde(default)]
    pub tile: Option<usize>,
    /// Run one discarded 16-token forward at load. Default: false.
    #[serde(default)]
    pub warmup: bool,
}

/// EmbeddingGemma 2 load config. Matmul precision defaults to BF16 (FP32
/// stream); [`Precision::Fp32`] runs FP32 matmuls on the same weights.
/// [`EmbeddingGemma2::precision`] reports what a loaded model runs. The
/// image and audio towers load only when enabled, since each adds resident
/// weights (about 0.3 GB for images, 0.6 GB for audio).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct Config {
    /// Directory with `model.safetensors`, `tokenizer.json`, `config.json`.
    pub model_dir: PathBuf,
    #[serde(default)]
    pub cache: CacheConfig,
    /// Encode/forward sequence cap. `None` is the model default (8192).
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    /// Reject embed batches larger than this.
    #[serde(default)]
    pub max_batch: Option<usize>,
    /// Post-hoc MRL output dim (768/512/256/128). `None` is full 768.
    #[serde(default)]
    pub mrl_dim: Option<usize>,
    /// Explicit pin manifest; `None` uses the embedded model manifest.
    #[serde(default)]
    pub manifest: Option<PathBuf>,
    #[serde(default)]
    pub verify_weights: bool,
    /// What this model may keep between calls, in bytes; overrides
    /// [`RuntimeOptions::memory_budget_bytes`](crate::cpu::runtime::RuntimeOptions::memory_budget_bytes).
    /// `Some(0)` is unbounded.
    #[serde(default)]
    pub memory_budget_bytes: Option<usize>,
    /// Matmul precision: [`Precision::Bf16`] or [`Precision::Fp32`]. `None`
    /// is the model default, BF16.
    #[serde(default)]
    pub precision: Option<Precision>,
    /// Load the image tower so [`Part::Image`] can be embedded.
    #[serde(default)]
    pub image: bool,
    /// Load the audio tower so [`Part::Audio`] can be embedded.
    #[serde(default)]
    pub audio: bool,
    /// Soft tokens per image; `None` is the processor default,
    /// [`ImageTokens::T280`]. Images are resized to fit it.
    #[serde(default)]
    pub image_tokens: Option<ImageTokens>,
    #[serde(default)]
    #[doc(hidden)]
    pub advanced: Advanced,
}

impl Config {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            cache: CacheConfig::default(),
            max_seq_len: None,
            max_batch: None,
            mrl_dim: None,
            manifest: None,
            verify_weights: false,
            memory_budget_bytes: None,
            precision: None,
            image: false,
            audio: false,
            image_tokens: None,
            advanced: Advanced::default(),
        }
    }

    pub fn with_mrl_dim(mut self, dim: usize) -> Self {
        self.mrl_dim = Some(dim);
        self
    }
    pub fn with_max_seq_len(mut self, n: usize) -> Self {
        self.max_seq_len = Some(n);
        self
    }
    pub fn with_max_batch(mut self, n: usize) -> Self {
        self.max_batch = Some(n);
        self
    }
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = Some(precision);
        self
    }
    pub fn with_image(mut self, yes: bool) -> Self {
        self.image = yes;
        self
    }
    pub fn with_audio(mut self, yes: bool) -> Self {
        self.audio = yes;
        self
    }
    pub fn with_image_tokens(mut self, tokens: ImageTokens) -> Self {
        self.image_tokens = Some(tokens);
        self
    }
    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }
    pub fn with_cache(mut self, cache: CacheConfig) -> Self {
        self.cache = cache;
        self
    }
    #[doc(hidden)]
    pub fn with_tile(mut self, n: usize) -> Self {
        self.advanced.tile = Some(n);
        self
    }

    /// Run one discarded 16-token forward before returning the loaded model.
    /// Warmup uses the normal execution path; errors
    /// propagate from load. Default: false.
    #[doc(hidden)]
    pub fn with_warmup(mut self, yes: bool) -> Self {
        self.advanced.warmup = yes;
        self
    }

    fn tile(&self, auto: usize) -> Result<usize> {
        let tile = self.advanced.tile.unwrap_or(auto);
        if tile == 0 {
            return Err(Error::Config("tile must be > 0".into()));
        }
        Ok(tile)
    }

    fn cache_options(&self) -> CacheOptions {
        self.cache
            .options(self.verify_weights, self.manifest.clone())
    }
}

/// Flat embedding matrix: `data.len() == n * dim`, row-major.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Embeddings {
    pub dim: usize,
    pub data: Vec<f32>,
}

impl Embeddings {
    /// Rows of `dim` floats; `data.len()` must be a multiple of `dim`.
    pub fn new(dim: usize, data: Vec<f32>) -> Result<Self> {
        if dim == 0 || data.len() % dim != 0 {
            return Err(Error::Shape(format!(
                "{} floats do not form rows of {dim}",
                data.len()
            )));
        }
        Ok(Self { dim, data })
    }

    pub fn n(&self) -> usize {
        self.data.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn row(&self, i: usize) -> Result<&[f32]> {
        let n = self.n();
        if i >= n {
            return Err(Error::Shape(format!("row {i} >= n {n}")));
        }
        let start = i * self.dim;
        Ok(&self.data[start..start + self.dim])
    }

    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        let dim = self.dim;
        self.data
            .chunks_exact(dim.max(1))
            .take(if dim == 0 { 0 } else { self.n() })
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }
}

/// One piece of a multimodal input. Parts concatenate in order, as
/// Sentence Transformers joins them.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Part<'a> {
    Text(&'a str),
    /// Any [`PixelFormat`]; resized to [`Config::image_tokens`].
    Image(ImageView<'a>),
    /// 16 kHz mono samples; past 30 s ([`MAX_SAMPLES`]) is truncated.
    Audio(&'a [f32]),
}

/// Pluggable tokenizer. The bundled Gemma tokenizer implements this.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, texts: &[String], kind: PromptKind) -> Result<Batch>;
    fn set_max_seq(&mut self, _max: usize) {}
    /// Ids of `text` alone (no BOS/EOS, no prefix), for
    /// [`EmbeddingGemma2::embed_parts`]. Must reject media placeholder tokens.
    fn encode_text(&self, _text: &str) -> Result<Vec<u32>> {
        Err(Error::Unsupported {
            what: "EG2 multimodal input",
            reason: "this tokenizer does not implement encode_text".into(),
        })
    }
}

impl Tokenizer for Frontend {
    fn encode(&self, texts: &[String], kind: PromptKind) -> Result<Batch> {
        Frontend::encode(self, texts, kind)
    }
    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        Frontend::encode_text(self, text)
    }
    fn set_max_seq(&mut self, max: usize) {
        self.max_seq = max;
    }
}

/// Loaded EmbeddingGemma 2 text tower. `Send + Sync`; share with `Arc`.
///
/// # Concurrency
///
/// `embed` / `embed_ids` / `embed_batch` / `embed_into` are **blocking** (run
/// them on `tokio::task::spawn_blocking`). Concurrent calls on one model
/// acquire a slot from the runtime's bounded execute pool
/// ([`crate::cpu::runtime::RuntimeOptions::exec_slots`], default 1). Parallel GEMM/glue
/// regions share the process-global thread pool.
pub struct EmbeddingGemma2<T = Frontend> {
    model: Model,
    tokenizer: T,
    mrl_dim: usize,
    max_batch: Option<usize>,
    max_seq_len: usize,
    workspace_limit: usize,
    image_tokens: ImageTokens,
}

impl EmbeddingGemma2<Frontend> {
    /// Load weights and the bundled tokenizer from the configured directory.
    /// If enabled with [`Config::with_warmup`], a discarded 16-token forward
    /// completes before this returns. Warmup errors retain the model path.
    pub fn load(rt: &Runtime, config: Config) -> Result<Self> {
        let tokenizer = Frontend::load(&config.model_dir)?;
        Self::load_with_tokenizer(rt, config, tokenizer)
    }
}

impl crate::cpu::runtime::Model for EmbeddingGemma2<Frontend> {
    type Config = Config;
    const PRECISIONS: &'static [Precision] = crate::embeddinggemma2::model::PRECISIONS;
    fn load(rt: &Runtime, config: Config) -> Result<Self> {
        Self::load(rt, config)
    }
    fn precision(&self) -> Precision {
        EmbeddingGemma2::precision(self)
    }
}

impl<T: Tokenizer> EmbeddingGemma2<T> {
    /// Load with a caller-provided tokenizer. Optional load warmup uses
    /// synthetic token IDs and does not call the tokenizer.
    pub fn load_with_tokenizer(rt: &Runtime, mut config: Config, mut tokenizer: T) -> Result<Self> {
        let mrl_dim = resolve_mrl(config.mrl_dim)?;
        if config.max_batch == Some(0) {
            return Err(Error::Config("max_batch must be >= 1".into()));
        }
        if let Some(max) = config.max_seq_len {
            if max == 0 || max > MAX_SEQ {
                return Err(Error::Config(format!(
                    "max_seq_len {max} out of range 1..={MAX_SEQ}"
                )));
            }
            tokenizer.set_max_seq(max);
        }
        let image_tokens = config.image_tokens.unwrap_or_default();
        let inner = load_model(rt, &mut config)?;
        Ok(Self {
            image_tokens,
            model: inner,
            tokenizer,
            mrl_dim,
            max_batch: config.max_batch,
            max_seq_len: config.max_seq_len.unwrap_or(MAX_SEQ),
            workspace_limit: rt.workspace_limit(),
        })
    }

    pub fn embed(
        &self,
        texts: impl IntoIterator<Item = impl AsRef<str>>,
        kind: PromptKind,
    ) -> Result<Embeddings> {
        let texts = collect_texts(texts, self.max_batch, self.workspace_limit)?;
        self.check_batch(texts.len())?;
        let batch = self.tokenizer.encode(&texts, kind)?;
        self.embed_batch(&batch)
    }

    pub fn embed_ids(
        &self,
        input_ids: Vec<Vec<u32>>,
        attention_mask: Option<Vec<Vec<u32>>>,
    ) -> Result<Embeddings> {
        self.check_batch(input_ids.len())?;
        if input_ids.is_empty() {
            return Err(Error::Shape("empty embed batch".into()));
        }
        for row in &input_ids {
            if row.is_empty() || row.len() > self.max_seq_len {
                return Err(Error::Shape(format!(
                    "EG2 sequence length {} outside 1..={}",
                    row.len(),
                    self.max_seq_len
                )));
            }
        }
        let attention_mask = match attention_mask {
            Some(m) => m,
            None => input_ids.iter().map(|row| vec![1u32; row.len()]).collect(),
        };
        self.embed_batch(&Batch {
            input_ids,
            attention_mask,
        })
    }

    pub fn embed_batch(&self, batch: &Batch) -> Result<Embeddings> {
        let n = batch.input_ids.len();
        self.check_batch(n)?;
        batch.validate(self.max_seq_len)?;
        let need = crate::checked_size("embedding output", &[n, self.mrl_dim])?;
        let bytes = crate::checked_size("embedding output bytes", &[need, 4])?;
        if bytes > self.workspace_limit {
            return Err(Error::WorkspaceLimit {
                operation: "embedding output",
                requested: bytes,
                limit: self.workspace_limit,
            });
        }
        let mut data = crate::try_vec::<f32>(need)?;
        self.embed_batch_into(batch, &mut data)?;
        Ok(Embeddings {
            dim: self.mrl_dim,
            data,
        })
    }

    /// Write `n * dim` floats into `out` (row-major). `out.len()` must be exact.
    pub fn embed_into(
        &self,
        texts: impl IntoIterator<Item = impl AsRef<str>>,
        kind: PromptKind,
        out: &mut [f32],
    ) -> Result<()> {
        let texts = collect_texts(texts, self.max_batch, self.workspace_limit)?;
        self.check_batch(texts.len())?;
        let batch = self.tokenizer.encode(&texts, kind)?;
        self.embed_batch_into(&batch, out)
    }

    fn embed_batch_into(&self, batch: &Batch, out: &mut [f32]) -> Result<()> {
        self.check_batch(batch.input_ids.len())?;
        batch.validate(self.max_seq_len)?;
        let n = batch.input_ids.len();
        let need = n
            .checked_mul(self.mrl_dim)
            .ok_or_else(|| Error::Shape("embed output size overflow".into()))?;
        if out.len() != need {
            return Err(Error::Shape(format!(
                "embed_into buffer {} != n {n} * dim {}",
                out.len(),
                self.mrl_dim
            )));
        }
        let rows = self.model.forward(batch)?;
        self.write_rows(&rows, out)
    }

    /// Copy (MRL-truncated) rows into `out`, `rows.len() * dim` floats.
    fn write_rows(&self, rows: &[Vec<f32>], out: &mut [f32]) -> Result<()> {
        let n = out.len() / self.mrl_dim;
        if rows.len() != n {
            return Err(Error::Shape(format!(
                "forward returned {} rows, expected {n}",
                rows.len()
            )));
        }
        for (i, row) in rows.iter().enumerate() {
            let dst = &mut out[i * self.mrl_dim..(i + 1) * self.mrl_dim];
            if self.mrl_dim < EMBED_DIM {
                let truncated = mrl_truncate(row, self.mrl_dim);
                if truncated.len() != self.mrl_dim {
                    return Err(Error::Shape("mrl truncate width".into()));
                }
                dst.copy_from_slice(&truncated);
            } else {
                if row.len() != self.mrl_dim {
                    return Err(Error::Shape(format!(
                        "row width {} != dim {}",
                        row.len(),
                        self.mrl_dim
                    )));
                }
                dst.copy_from_slice(row);
            }
        }
        Ok(())
    }

    /// Embed inputs made of text, images and audio. `prompt` prepends the
    /// query/document prefix; `None` adds none (the Sentence Transformers
    /// default for multimodal input). Each image and clip runs its tower
    /// first; its soft tokens then replace the placeholder tokens.
    pub fn embed_parts(
        &self,
        inputs: &[&[Part<'_>]],
        prompt: Option<PromptKind>,
    ) -> Result<Embeddings> {
        let n = inputs.len();
        self.check_batch(n)?;
        let need = crate::checked_size("embedding output", &[n, self.mrl_dim])?;
        let bytes = crate::checked_size("embedding output bytes", &[need, 4])?;
        if bytes > self.workspace_limit {
            return Err(Error::WorkspaceLimit {
                operation: "embedding output",
                requested: bytes,
                limit: self.workspace_limit,
            });
        }
        let mut rows = Vec::with_capacity(n);
        for parts in inputs {
            rows.push(self.prepare(parts, prompt)?);
        }
        let mut spliced = Vec::with_capacity(n);
        for (ids, media) in rows {
            let mut splice = Vec::with_capacity(media.len());
            for (pos, m) in media {
                let soft = match &m {
                    Media::Image(p) => self.model.image_soft_tokens(p)?,
                    Media::Audio(f) => self.model.audio_soft_tokens(f)?,
                };
                splice.push((pos, soft));
            }
            spliced.push((ids, splice));
        }
        let got = self.model.forward_spliced(spliced)?;
        let mut data = crate::try_vec::<f32>(need)?;
        self.write_rows(&got, &mut data)?;
        Ok(Embeddings {
            dim: self.mrl_dim,
            data,
        })
    }

    /// Token ids [`EmbeddingGemma2::embed_parts`] runs for one input, with
    /// each placeholder expanded to its soft-token count.
    #[doc(hidden)]
    pub fn parts_input_ids(
        &self,
        parts: &[Part<'_>],
        prompt: Option<PromptKind>,
    ) -> Result<Vec<u32>> {
        Ok(self.prepare(parts, prompt)?.0)
    }

    /// Soft tokens (`[n, HIDDEN]` row-major) of each media part of one input,
    /// in order. Stage check against the reference towers.
    #[doc(hidden)]
    pub fn parts_soft_tokens(&self, parts: &[Part<'_>]) -> Result<Vec<Vec<f32>>> {
        let (_, media) = self.prepare(parts, None)?;
        media
            .iter()
            .map(|(_, m)| match m {
                Media::Image(p) => self.model.image_soft_tokens(p),
                Media::Audio(f) => self.model.audio_soft_tokens(f),
            })
            .collect()
    }

    /// Preprocess one input. Placeholder counts fix the sequence length,
    /// which is checked before any tower runs.
    fn prepare(
        &self,
        parts: &[Part<'_>],
        prompt: Option<PromptKind>,
    ) -> Result<(Vec<u32>, Vec<(usize, Media)>)> {
        if parts.is_empty() {
            return Err(Error::Shape("EG2 input has no parts".into()));
        }
        let threads = self.model.threads();
        let mut media = Vec::new();
        let mut ids = vec![BOS_ID];
        let mut text = prompt.map(|k| k.prefix().to_string()).unwrap_or_default();
        let flush = |text: &mut String, ids: &mut Vec<u32>| -> Result<()> {
            if !text.is_empty() {
                ids.extend(self.tokenizer.encode_text(text)?);
                text.clear();
            }
            Ok(())
        };
        for part in parts {
            match part {
                Part::Text(t) => text.push_str(t),
                Part::Image(view) => {
                    if !self.model.has_image() {
                        return Err(not_loaded("image"));
                    }
                    flush(&mut text, &mut ids)?;
                    let patches = media::image_patches(view, self.image_tokens.count(), threads)?;
                    ids.push(IMAGE_OPEN_ID);
                    let pos = ids.len();
                    ids.resize(pos + patches.soft_tokens(), IMAGE_TOKEN_ID);
                    ids.push(IMAGE_CLOSE_ID);
                    media.push((pos, Media::Image(patches)));
                }
                Part::Audio(samples) => {
                    if !self.model.has_audio() {
                        return Err(not_loaded("audio"));
                    }
                    flush(&mut text, &mut ids)?;
                    let features = media::audio_features(samples, threads)?;
                    ids.push(AUDIO_OPEN_ID);
                    let pos = ids.len();
                    ids.resize(pos + features.soft_tokens(), AUDIO_TOKEN_ID);
                    ids.push(AUDIO_CLOSE_ID);
                    media.push((pos, Media::Audio(features)));
                }
            }
        }
        flush(&mut text, &mut ids)?;
        ids.push(EOS_ID);
        if ids.len() > self.max_seq_len {
            return Err(Error::Shape(format!(
                "{} tokens exceed max_seq_len {}",
                ids.len(),
                self.max_seq_len
            )));
        }
        Ok((ids, media))
    }

    /// Whether [`Part::Image`] / [`Part::Audio`] inputs are loaded.
    pub fn supports_image(&self) -> bool {
        self.model.has_image()
    }

    pub fn supports_audio(&self) -> bool {
        self.model.has_audio()
    }

    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }

    pub fn embedding_dim(&self) -> usize {
        self.mrl_dim
    }

    /// The matmul precision this model runs.
    pub fn precision(&self) -> Precision {
        self.model.precision()
    }

    /// Backend counters (packed weights, cached plans, retained buffers and
    /// scratch) for drivers and diagnostics.
    #[doc(hidden)]
    pub fn backend_stats(&self) -> serde_json::Value {
        self.model.stats()
    }

    fn check_batch(&self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(Error::Shape("empty embed batch".into()));
        }
        if let Some(max) = self.max_batch {
            if n > max {
                return Err(Error::Shape(format!("batch {n} > max_batch {max}")));
            }
        }
        Ok(())
    }
}

enum Media {
    Image(media::ImagePatches),
    Audio(media::AudioFeatures),
}

fn not_loaded(what: &'static str) -> Error {
    Error::Unsupported {
        what: if what == "image" {
            "EG2 image input"
        } else {
            "EG2 audio input"
        },
        reason: format!("load with Config::with_{what}(true)"),
    }
}

fn resolve_mrl(dim: Option<usize>) -> Result<usize> {
    let d = dim.unwrap_or(EMBED_DIM);
    if MRL_DIMS.contains(&d) {
        Ok(d)
    } else {
        Err(Error::Config(format!("mrl_dim {d} not in {MRL_DIMS:?}")))
    }
}

fn load_model(rt: &Runtime, config: &mut Config) -> Result<Model> {
    let backend = rt.backend_with(
        BackendOverrides::default().with_memory_budget_bytes(config.memory_budget_bytes),
    )?;
    let tile = config.tile(crate::embeddinggemma2::config::attn_tile(rt.cpu_isa()))?;
    let cache = config.cache_options();
    let precision = Precision::resolve(
        config.precision,
        "EmbeddingGemma 2",
        crate::embeddinggemma2::model::PRECISIONS,
    )?;
    let mut model =
        Model::load_with_precision(&config.model_dir, backend, tile, &cache, precision)?;
    model.load_media(&config.model_dir, &cache, config.image, config.audio)?;
    if config.advanced.warmup {
        model.warmup().map_err(|e| e.at(&config.model_dir))?;
    }
    Ok(model)
}

/// Collect caller iterators incrementally so a configured batch/byte cap is
/// checked before cloning strings or growing the batch.
fn collect_texts(
    texts: impl IntoIterator<Item = impl AsRef<str>>,
    max_batch: Option<usize>,
    limit: usize,
) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut bytes = 0usize;
    for text in texts {
        let text = text.as_ref();
        let n = out.len() + 1;
        if max_batch.is_some_and(|max| n > max) {
            return Err(Error::Shape(format!(
                "batch {n} > max_batch {}",
                max_batch.unwrap_or(0)
            )));
        }
        bytes = bytes
            .checked_add(text.len())
            .and_then(|n| n.checked_add(std::mem::size_of::<String>()))
            .ok_or(Error::SizeOverflow("text batch"))?;
        if bytes > limit {
            return Err(Error::WorkspaceLimit {
                operation: "text batch",
                requested: bytes,
                limit,
            });
        }
        out.try_reserve(1)
            .map_err(|_| Error::Allocation { bytes })?;
        let mut owned = String::new();
        owned
            .try_reserve_exact(text.len())
            .map_err(|_| Error::Allocation { bytes: text.len() })?;
        owned.push_str(text);
        out.push(owned);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_and_byte_caps_apply_during_collection() {
        assert!(collect_texts(std::iter::repeat("x"), Some(1), 1024).is_err());
        assert!(collect_texts(["hello"], None, 1).is_err());
        assert_eq!(
            collect_texts(["a", "b"], Some(2), 1024).unwrap(),
            ["a", "b"]
        );
        let e = Embeddings {
            dim: 0,
            data: vec![1.],
        };
        assert!(e.row(0).is_err());
        assert_eq!(e.rows().count(), 0);
    }
    #[test]
    fn embeddings_rows() {
        let e = Embeddings {
            dim: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(e.n(), 2);
        assert_eq!(e.row(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(e.row(1).unwrap(), &[3.0, 4.0]);
        assert!(e.row(2).is_err());
        let rows: Vec<&[f32]> = e.rows().collect();
        assert_eq!(rows, vec![&[1.0, 2.0][..], &[3.0, 4.0][..]]);
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn serde_roundtrip_and_defaults() {
        let c = Config::new("/models/eg2").with_mrl_dim(256);
        let json = serde_json::to_string(&c).unwrap();
        assert!(!json.contains("library") && json.contains(r#""precision":null"#));
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.precision, None);
        let fp32 = c.clone().with_precision(Precision::Fp32);
        let json32 = serde_json::to_string(&fp32).unwrap();
        assert!(json32.contains(r#""precision":"fp32""#), "{json32}");
        assert_eq!(serde_json::from_str::<Config>(&json32).unwrap(), fp32);
        assert_eq!(back.model_dir, PathBuf::from("/models/eg2"));
        assert_eq!(back.mrl_dim, Some(256));
        assert!(back.cache.enabled);
        assert!(back.advanced.tile.is_none());
        assert!(!back.advanced.warmup);

        let c: Config = serde_json::from_str(r#"{"model_dir":"/m","mrl_dim":256}"#).unwrap();
        assert_eq!(c.mrl_dim, Some(256));
        assert!(c.cache.enabled);
    }

    #[test]
    fn warmup_serde_roundtrip_and_legacy_defaults() {
        let c = Config::new("/models/eg2").with_warmup(true);
        let back: Config = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
        assert!(!c.with_warmup(false).advanced.warmup);
        for json in [
            r#"{"model_dir":"/m"}"#,
            r#"{"model_dir":"/m","advanced":{"tile":64}}"#,
        ] {
            let c: Config = serde_json::from_str(json).unwrap();
            assert!(!c.advanced.warmup);
        }
    }

    #[test]
    fn tile_none_uses_plan_some_overrides() {
        assert_eq!(Config::new("/m").tile(112).unwrap(), 112);
        assert_eq!(Config::new("/m").with_tile(64).tile(112).unwrap(), 64);
        assert!(Config::new("/m").with_tile(0).tile(112).is_err());
    }

    #[test]
    fn mrl_dims_validate() {
        assert_eq!(resolve_mrl(None).unwrap(), EMBED_DIM);
        assert!(resolve_mrl(Some(300)).is_err());
    }
}
