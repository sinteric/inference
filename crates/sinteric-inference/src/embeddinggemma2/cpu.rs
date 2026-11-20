//! EmbeddingGemma 2 tuned for CPUs: [`EmbeddingGemma2`], loaded from a
//! [`Runtime`] and shared across threads.
//!
//! ```ignore
//! use sinteric_inference::{cpu::{Options, Runtime}, embeddinggemma2 as eg2};
//! let rt = Runtime::new(Options::new())?;
//! let model = eg2::cpu::EmbeddingGemma2::load(&rt, eg2::cpu::Config::new(dir).with_truncate(true))?;
//! let vectors = model.embed(["what is sinteric?"], eg2::PromptKind::Query)?;
//! ```
//!
//! Images and audio are opt-in at load ([`Config::with_image`],
//! [`Config::with_audio`]) and embed through [`EmbeddingGemma2::embed_parts`]:
//!
//! ```ignore
//! use sinteric_inference::embeddinggemma2::{cpu, ImageView, Part, PixelFormat};
//! let model = cpu::EmbeddingGemma2::load(&rt, cpu::Config::new(dir).with_image(true).with_audio(true))?;
//! let photo = ImageView::new(&rgb, width, height, width as usize * 3, PixelFormat::Rgb8)?;
//! let vectors = model.embed_parts(
//!     &[&[Part::Text("a photo of"), Part::Image(photo)], &[Part::Audio(&samples_16k)]],
//!     None,
//! )?;
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::cpu::onednn::CacheOptions;
use crate::cpu::runtime::{
    release_free_memory, resolve_precision, BackendOverrides, BackendStats, CacheConfig,
    LoadReport, Runtime,
};
use crate::embeddinggemma2::config::{
    AUDIO_CLOSE_ID, AUDIO_OPEN_ID, AUDIO_TOKEN_ID, BOS_ID, EOS_ID, IMAGE_CLOSE_ID, IMAGE_OPEN_ID,
    IMAGE_TOKEN_ID,
};
use crate::embeddinggemma2::{
    media, mrl_truncate, Batch, Embeddings, ImageTokens, Model, Part, PromptKind, Tokenizer,
    EMBED_DIM, MAX_SEQ, MRL_DIMS,
};
use crate::{Error, Result};

pub use crate::embeddinggemma2::model::Precision;

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

/// What [`EmbeddingGemma2::load`] loads. Every field but the directory has
/// a default. The image and audio towers load only when enabled, since each
/// adds resident weights (about 0.3 GB for images, 0.6 GB for audio).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct Config {
    /// Directory with `model.safetensors`, `tokenizer.json`, `config.json`.
    pub model_dir: PathBuf,
    #[serde(default)]
    pub cache: CacheConfig,
    /// The window: the most tokens one input may have, prompt and special
    /// tokens included. `None` is the model's 8,192.
    #[serde(default)]
    pub max_seq_len: Option<usize>,
    /// Cut a text past the window to its first `window - 1` tokens and close
    /// it with `<eos>`, as Sentence Transformers truncates, instead of
    /// refusing it ([`crate::ErrorKind::TooLarge`]). Multimodal inputs are
    /// never cut. Default: false.
    #[serde(default)]
    pub truncate: bool,
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
    /// [`Options::memory_budget_bytes`](crate::cpu::Options::memory_budget_bytes).
    /// `Some(0)` is unbounded.
    #[serde(default)]
    pub memory_budget_bytes: Option<usize>,
    /// Matmul precision. `None` lets the library pick for the runtime's CPU:
    /// [`Precision::Bf16`] where it has BF16 matmul kernels, else
    /// [`Precision::Fp32`] ([`EmbeddingGemma2::precisions`]).
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
            truncate: false,
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
    pub fn with_truncate(mut self, yes: bool) -> Self {
        self.truncate = yes;
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
            return Err(Error::config("tile must be > 0"));
        }
        Ok(tile)
    }

    fn cache_options(&self) -> CacheOptions {
        self.cache
            .options(self.verify_weights, self.manifest.clone())
    }
}

/// EmbeddingGemma 2 loaded on a CPU. `Send + Sync`; share it with `Arc`.
///
/// # Concurrency
///
/// Every call blocks (from async code, run it on
/// `tokio::task::spawn_blocking`). Concurrent calls each take one of the
/// model's execute slots ([`Options::exec_slots`](crate::cpu::Options::exec_slots),
/// default 1); their parallel regions share the runtime's one pool.
pub struct EmbeddingGemma2 {
    model: Model,
    tokenizer: Tokenizer,
    mrl_dim: usize,
    max_batch: Option<usize>,
    max_seq_len: usize,
    workspace_limit: usize,
    image_tokens: ImageTokens,
}

impl std::fmt::Debug for EmbeddingGemma2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingGemma2")
            .field("precision", &self.model.precision())
            .field("mrl_dim", &self.mrl_dim)
            .field("max_seq_len", &self.max_seq_len)
            .field("truncates", &self.tokenizer.truncates())
            .field("image", &self.model.has_image())
            .field("audio", &self.model.has_audio())
            .finish_non_exhaustive()
    }
}

impl EmbeddingGemma2 {
    /// The precisions this model runs on `rt`'s CPU, the one it picks when
    /// a config names none first.
    pub fn precisions(rt: &Runtime) -> &'static [Precision] {
        Precision::on(rt.isa())
    }

    /// Load the weights and the tokenizer from [`Config::model_dir`]. With
    /// [`Config::with_warmup`], one discarded 16-token forward runs before
    /// this returns.
    pub fn load(rt: &Runtime, mut config: Config) -> Result<Self> {
        let mrl_dim = resolve_mrl(config.mrl_dim)?;
        if config.max_batch == Some(0) {
            return Err(Error::config("max_batch must be >= 1"));
        }
        let max_seq_len = config.max_seq_len.unwrap_or(MAX_SEQ);
        let min = if config.truncate { 2 } else { 1 };
        if !(min..=MAX_SEQ).contains(&max_seq_len) {
            return Err(Error::config(format!(
                "max_seq_len {max_seq_len} out of range {min}..={MAX_SEQ}"
            )));
        }
        let mut tokenizer = Tokenizer::load(&config.model_dir)?;
        tokenizer.set_window(max_seq_len, config.truncate);
        let image_tokens = config.image_tokens.unwrap_or_default();
        let model = load_model(rt, &mut config);
        release_free_memory();
        Ok(Self {
            image_tokens,
            model: model?,
            tokenizer,
            mrl_dim,
            max_batch: config.max_batch,
            max_seq_len,
            workspace_limit: rt.workspace_limit(),
        })
    }

    /// Free this model's weights, oneDNN objects and buffers, and return the
    /// freed heap pages to the OS. The runtime stays. For a model shared by
    /// `Arc`, the last owner unloads:
    /// `if let Some(m) = Arc::into_inner(model) { m.unload() }`. Dropping it
    /// frees the same memory but may leave it in the allocator.
    pub fn unload(self) {
        drop(self);
        release_free_memory();
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
            return Err(Error::workspace_limit(
                "embedding output",
                bytes,
                self.workspace_limit,
            ));
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
            .ok_or_else(|| Error::internal("embed output size overflow"))?;
        if out.len() != need {
            return Err(Error::invalid_input(format!(
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
            return Err(Error::internal(format!(
                "forward returned {} rows, expected {n}",
                rows.len()
            )));
        }
        for (i, row) in rows.iter().enumerate() {
            let dst = &mut out[i * self.mrl_dim..(i + 1) * self.mrl_dim];
            if self.mrl_dim < EMBED_DIM {
                let truncated = mrl_truncate(row, self.mrl_dim);
                if truncated.len() != self.mrl_dim {
                    return Err(Error::internal("mrl truncate width"));
                }
                dst.copy_from_slice(&truncated);
            } else {
                if row.len() != self.mrl_dim {
                    return Err(Error::internal(format!(
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
            return Err(Error::workspace_limit(
                "embedding output",
                bytes,
                self.workspace_limit,
            ));
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
            return Err(Error::invalid_input("an input has no parts"));
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
            return Err(Error::too_large(format!(
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

    /// The tokenizer, with this model's window and truncation.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn embedding_dim(&self) -> usize {
        self.mrl_dim
    }

    /// The matmul precision this model runs.
    pub fn precision(&self) -> Precision {
        self.model.precision()
    }

    /// Where each part's packed weights came from, and how long the load's
    /// phases took.
    pub fn load_report(&self) -> LoadReport {
        self.model.load_report()
    }

    /// The backend's memory and counters now.
    pub fn backend_stats(&self) -> BackendStats {
        self.model.backend_stats()
    }

    fn check_batch(&self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(Error::invalid_input("empty batch"));
        }
        if let Some(max) = self.max_batch {
            if n > max {
                return Err(Error::too_large(format!("batch {n} > max_batch {max}")));
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
    Error::unsupported(
        if what == "image" {
            "EG2 image input"
        } else {
            "EG2 audio input"
        },
        format!("load with Config::with_{what}(true)"),
    )
}

fn resolve_mrl(dim: Option<usize>) -> Result<usize> {
    let d = dim.unwrap_or(EMBED_DIM);
    if MRL_DIMS.contains(&d) {
        Ok(d)
    } else {
        Err(Error::config(format!("mrl_dim {d} not in {MRL_DIMS:?}")))
    }
}

fn load_model(rt: &Runtime, config: &mut Config) -> Result<Model> {
    let backend = rt.backend_with(
        BackendOverrides::default().with_memory_budget_bytes(config.memory_budget_bytes),
    )?;
    let tile = config.tile(crate::embeddinggemma2::config::attn_tile(rt.cpu_isa()))?;
    let cache = config.cache_options();
    let precision = resolve_precision(
        config.precision,
        Precision::on(rt.isa()),
        "EmbeddingGemma 2",
        rt.isa(),
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
            return Err(Error::too_large(format!(
                "batch {n} > max_batch {}",
                max_batch.unwrap_or(0)
            )));
        }
        bytes = bytes
            .checked_add(text.len())
            .and_then(|n| n.checked_add(std::mem::size_of::<String>()))
            .ok_or(Error::size_overflow("text batch"))?;
        if bytes > limit {
            return Err(Error::workspace_limit("text batch", bytes, limit));
        }
        out.try_reserve(1)
            .map_err(|_| Error::out_of_memory(bytes))?;
        let mut owned = String::new();
        owned
            .try_reserve_exact(text.len())
            .map_err(|_| Error::out_of_memory(text.len()))?;
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
        let e = collect_texts(std::iter::repeat("x"), Some(1), 1024).unwrap_err();
        assert_eq!(e.kind(), crate::ErrorKind::TooLarge);
        let e = collect_texts(["hello"], None, 1).unwrap_err();
        assert_eq!(e.kind(), crate::ErrorKind::TooLarge);
        assert_eq!(
            collect_texts(["a", "b"], Some(2), 1024).unwrap(),
            ["a", "b"]
        );
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
