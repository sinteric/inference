//! EmbeddingGemma 2 text tower on [`OnednnCpu`]: one graph per
//! [`Precision`] ([`bf16`], [`fp32`]) over the same weights. Token embeddings
//! are a zero-copy mmap of the safetensors BF16 table.

use crate::cpu::kernels::{bf16_bits_to_f32, for_row_tiles, DefaultRope, RowsMut, SQRT_512_BF16};
use crate::cpu::onednn::{cache_path_with, CacheOptions, OnednnCpu, PackedWeight};
use crate::cpu::runtime::{BackendStats, Isa, LoadReport, PackedWeights, WeightSource};
use crate::tensor_file::Mapped;
use crate::{check_shape, Result, ResultContext as Context};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::embeddinggemma2::config::{HIDDEN, VOCAB};
use crate::embeddinggemma2::frontend::Batch;

mod audio;
mod bf16;
mod fp32;
mod load;
mod tower;
mod vision;

use load::expected_weight_pin;
use tower::TowerSource;

/// What EmbeddingGemma 2's matmuls compute in on a CPU. RMSNorm, softmax,
/// RoPE, the residual stream, pooling and the output stay FP32 either way,
/// and the weights are the BF16 checkpoint either way: FP32 widens them
/// exactly, so only the matmul inputs differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Precision {
    /// BF16 matmul inputs, FP32 accumulation. The default where the CPU has
    /// BF16 matmul kernels ([`Isa::has_bf16_matmul`]): faster than FP32 at
    /// every length but the shortest texts on a Zen 5.
    Bf16,
    /// FP32 matmuls, matching the PyTorch FP32 reference. The default, and
    /// the only precision, on AVX2.
    Fp32,
}

crate::cpu::runtime::precision_names!(Precision);

impl Precision {
    /// The precisions EmbeddingGemma 2 runs on `isa`, its default first.
    pub(crate) fn on(isa: Isa) -> &'static [Self] {
        if isa.has_bf16_matmul() {
            &[Self::Bf16, Self::Fp32]
        } else {
            &[Self::Fp32]
        }
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
    load_source: WeightSource,
    load_profile: BTreeMap<String, f64>,
}

impl Model {
    fn linear(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let l = self
            .linears
            .get(name)
            .with_context(|| format!("linear {name}"))?;
        check_shape!(x.len().is_multiple_of(l.weight.k()), "linear matrix {name}");
        let mut y = self.backend.buffer(crate::checked_size(
            "activation elements",
            &[x.len() / l.weight.k(), l.weight.n()],
        )?)?;
        self.backend.linear_bf16(x, &l.weight, &mut y, false)?;
        Ok(y)
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
        let ids: Vec<u32> = batch.input_ids.iter().flatten().copied().collect();
        let table = self.embeddings.as_slice();
        let rows = RowsMut::new(&mut x, HIDDEN);
        for_row_tiles(
            self.backend.threads(),
            ntok,
            self.backend.residual_row_tile(ntok),
            8,
            |lo, hi| {
                // SAFETY: `for_row_tiles` hands out disjoint ranges, each once.
                let dst = unsafe { rows.range(lo, hi) };
                for (&id, dst) in ids[lo..hi].iter().zip(dst.as_chunks_mut::<HIDDEN>().0) {
                    let id = id as usize;
                    embed_row(&table[id * HIDDEN..(id + 1) * HIDDEN], dst, SQRT_512_BF16);
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
        let tower = self.vision.as_ref().ok_or_else(|| {
            crate::Error::unsupported("EG2 image input", "load with Config::image = true")
        })?;
        tower.forward(&self.backend, self.precision, self.tile, img)
    }

    /// Soft tokens `[f.soft_tokens(), HIDDEN]` of one audio clip.
    pub(crate) fn audio_soft_tokens(
        &self,
        f: &crate::embeddinggemma2::media::AudioFeatures,
    ) -> Result<Vec<f32>> {
        let tower = self.audio.as_ref().ok_or_else(|| {
            crate::Error::unsupported("EG2 audio input", "load with Config::audio = true")
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

    pub fn backend_stats(&self) -> BackendStats {
        self.backend.stats()
    }

    /// Where each part's packed weights came from, and the load's phases.
    pub fn load_report(&self) -> LoadReport {
        let mut packed = vec![PackedWeights {
            part: "text",
            source: self.load_source.clone(),
        }];
        if let Some(t) = &self.vision {
            packed.push(PackedWeights {
                part: "image",
                source: t.load_source().clone(),
            });
        }
        if let Some(t) = &self.audio {
            packed.push(PackedWeights {
                part: "audio",
                source: t.load_source().clone(),
            });
        }
        LoadReport {
            packed,
            phases_s: self.load_profile.clone(),
        }
    }
}

fn embed_row(src: &[u16], dst: &mut [f32], scale: f32) {
    assert_eq!(src.len(), dst.len(), "embedding row length");
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bf16") {
            // SAFETY: the CPU has both features (checked just above) and the
            // lengths are checked above.
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

/// # Safety
/// The CPU must support AVX-512F and AVX512_BF16, and `dst` must be as long
/// as `src`.
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

/// The shape of one dense forward.
struct Dense<'a> {
    /// Rows in the batch.
    b: usize,
    /// Tokens per row.
    s: usize,
    /// `b * s`.
    tokens: usize,
    /// Rows the workspace holds, `tokens` padded up (`Model::work_rows`).
    m_pad: usize,
    /// Each row's padded positions, which attention skips.
    padding: &'a [Vec<usize>],
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
