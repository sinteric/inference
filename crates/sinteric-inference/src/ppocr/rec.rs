//! Korean PP-OCRv5 mobile: PPLCNetV3, two SVTR blocks and greedy CTC.
use crate::cpu::onednn::{OnednnCpu, PackedWeight};
use crate::image::{ImageView, Quad, RecCrop};
use crate::ppocr::net::{self, OcrNet, Tensor};
use crate::ppocr::rec_graph::{self, Segments};
use crate::ppocr::weights::Assets;
use crate::{checked_size, try_vec, Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

pub const DICTIONARY_ENTRIES: usize = 11945;
pub const CTC_CLASSES: usize = DICTIONARY_ENTRIES + 2;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Batching {
    #[default]
    Exact,
    Bucketed,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct RecOptions {
    pub height: usize,
    pub min_width: usize,
    pub max_width: usize,
    pub batching: Batching,
    pub plan_capacity: usize,
    pub max_batch: usize,
    /// Bound the page and each warped line crop in
    /// [`Recognizer::recognize`] before allocating.
    pub max_pixels: usize,
}
impl Default for RecOptions {
    fn default() -> Self {
        Self {
            height: 48,
            min_width: 16,
            max_width: 1600,
            batching: Batching::Exact,
            plan_capacity: 512,
            max_batch: 16,
            max_pixels: 64 * 1024 * 1024,
        }
    }
}
impl RecOptions {
    pub fn validate(&self) -> Result<()> {
        if self.height != 48
            || self.min_width < 16
            || self.max_width > 1600
            || self.min_width > self.max_width
            || self.max_batch == 0
            || self.max_batch > 16
            || self.plan_capacity == 0
            || self.max_pixels == 0
        {
            return Err(Error::config(
                "invalid recognizer options (height=48, widths 16..1600, batch 1..16)",
            ));
        }
        Ok(())
    }
}
// Elements per task for the row-parallel host passes over head outputs.
// Splitting rows across threads leaves every element's arithmetic unchanged.
const ROW_GRAIN: usize = 1 << 15;
pub(crate) fn all_par(threads: usize, values: &[f32], valid: impl Fn(f32) -> bool + Sync) -> bool {
    let ok = AtomicBool::new(true);
    crate::cpu::kernels::for_ranges(threads, values.len(), ROW_GRAIN, |lo, hi| {
        if !values[lo..hi].iter().all(|&v| valid(v)) {
            ok.store(false, Ordering::Relaxed);
        }
    });
    ok.into_inner()
}
// A oneDNN matmul primitive holds ~0.6 MB of kernel descriptors, so one per
// row count made projection plans the largest resident cost. Projections run
// in chunks of at most CHUNK_ROWS rows with the tail padded to ROW_TILE rows:
// each row's result is bitwise independent of the primitive's M (see
// `linear_rows_are_independent_of_m`), so outputs are unchanged and a projection
// needs at most CHUNK_ROWS / ROW_TILE primitives.
const CHUNK_ROWS: usize = 256;
const ROW_TILE: usize = 16;
struct Linear {
    weights: PackedWeight,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}
impl Linear {
    fn load(
        cpu: &OnednnCpu,
        assets: &Assets,
        id: usize,
        bias_id: usize,
        k: usize,
        n: usize,
    ) -> Result<Self> {
        let name = format!("rec.matmul{id:03}.w");
        if assets.graph().tensors.get(&name) != Some(&vec![k, n]) {
            return Err(Error::model_files(format!("compiled projection {name}")));
        }
        let raw = assets.tensor(&name)?;
        let mut transposed = try_vec(raw.len())?;
        for i in 0..k {
            for j in 0..n {
                transposed[j * k + i] = raw[i * n + j];
            }
        }
        let bias = assets.tensor(&format!("rec.const{bias_id:03}"))?;
        if bias.len() != n {
            return Err(Error::model_files("recognizer linear bias"));
        }
        Ok(Self {
            weights: cpu.pack(n, k, &transposed)?,
            bias,
            input: k,
            output: n,
        })
    }
    fn run(&self, cpu: &OnednnCpu, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let (k, n) = (self.input, self.output);
        crate::check_shape!(x.len() == rows * k, "recognizer projection input");
        let mut y = try_vec(checked_size("recognizer projection", &[rows, n])?)?;
        for start in (0..rows).step_by(CHUNK_ROWS) {
            let m = (rows - start).min(CHUNK_ROWS);
            let (src, dst) = (
                &x[start * k..(start + m) * k],
                &mut y[start * n..(start + m) * n],
            );
            let padded = m.next_multiple_of(ROW_TILE);
            if padded == m {
                cpu.linear(src, &self.weights, dst, false)?;
            } else {
                let mut src_padded = try_vec(padded * k)?;
                src_padded[..m * k].copy_from_slice(src);
                let mut dst_padded = try_vec(padded * n)?;
                cpu.linear(&src_padded, &self.weights, &mut dst_padded, false)?;
                dst.copy_from_slice(&dst_padded[..m * n]);
            }
        }
        let p = y.as_mut_ptr() as usize;
        crate::cpu::kernels::for_ranges(cpu.threads(), rows, ROW_GRAIN.div_ceil(n), |lo, hi| {
            // SAFETY: workers receive disjoint row ranges of `y`.
            let y = unsafe { crate::cpu::kernels::rows_mut(p as *mut f32, lo, hi, n) };
            for row in y.chunks_exact_mut(n) {
                for (v, b) in row.iter_mut().zip(&self.bias) {
                    *v += b;
                }
            }
        });
        Ok(y)
    }
}
struct Norm {
    scale: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}
impl Norm {
    fn load(assets: &Assets, id: usize) -> Result<Self> {
        Ok(Self {
            scale: assets.tensor(&format!("rec.layernorm{id:03}.w"))?,
            bias: assets.tensor(&format!("rec.layernorm{id:03}.b"))?,
            eps: if id == 4 { 1e-6 } else { 1e-5 },
        })
    }
    fn run(&self, cpu: &OnednnCpu, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        cpu.layernorm(rows, 120, x, &self.scale, &self.bias, self.eps)
    }
}
struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    proj: Linear,
    up: Linear,
    down: Linear,
    scale: f32,
}
impl Block {
    fn load(cpu: &OnednnCpu, assets: &Assets, id: usize) -> Result<Self> {
        Ok(Self {
            norm1: Norm::load(assets, id * 2)?,
            norm2: Norm::load(assets, id * 2 + 1)?,
            qkv: Linear::load(cpu, assets, id * 6, 215 + id * 4, 120, 360)?,
            proj: Linear::load(cpu, assets, id * 6 + 3, 216 + id * 4, 120, 120)?,
            up: Linear::load(cpu, assets, id * 6 + 4, 217 + id * 4, 120, 240)?,
            down: Linear::load(cpu, assets, id * 6 + 5, 218 + id * 4, 240, 120)?,
            scale: assets.tensor(&format!("rec.const{}", 224 + id))?[0],
        })
    }
    fn run(
        &self,
        cpu: &OnednnCpu,
        x: &[f32],
        batch: usize,
        t: usize,
        lengths: &[usize],
    ) -> Result<Vec<f32>> {
        let rows = batch * t;
        let norm = self.norm1.run(cpu, x, rows)?;
        let qkv = self.qkv.run(cpu, &norm, rows)?;
        let mut q = try_vec(rows * 120)?;
        let mut k = try_vec(rows * 120)?;
        let mut v = try_vec(rows * 120)?;
        for b in 0..batch {
            for h in 0..8 {
                for s in 0..t {
                    for d in 0..15 {
                        let dst = ((b * 8 + h) * t + s) * 15 + d;
                        let src = (b * t + s) * 360 + h * 15 + d;
                        q[dst] = qkv[src] * self.scale;
                        k[dst] = qkv[src + 120];
                        v[dst] = qkv[src + 240];
                    }
                }
            }
        }
        // Attention matmuls run as exact sequential-FMA kernels: oneDNN sums
        // these shapes in the same order
        // (`onednn_attention_matmuls_are_sequential_fma_chains`) but
        // would keep a ~0.6 MB primitive per (batch, t). A single token keeps
        // oneDNN, whose 1x1 case sums differently.
        let (g, threads) = (batch * 8, cpu.threads());
        let mut logits = if t > 1 {
            crate::cpu::kernels::seq_matmul_nt(g, t, 15, t, &q, &k, threads)?
        } else {
            cpu.mm(g, t, t, 15, t * 15, 15, t * 15, 15, true, &q, &k)?
        };
        for b in 0..batch {
            if lengths[b] < t {
                for h in 0..8 {
                    for s in 0..t {
                        logits
                            [((b * 8 + h) * t + s) * t + lengths[b]..((b * 8 + h) * t + s + 1) * t]
                            .fill(f32::NEG_INFINITY);
                    }
                }
            }
        }
        let prob = cpu.softmax(batch * 8 * t, t, &logits)?;
        let context = if t > 1 {
            crate::cpu::kernels::seq_matmul_nn(g, t, t, 15, &prob, &v, threads)?
        } else {
            cpu.mm(g, t, 15, t, t * t, t, t * 15, 15, false, &prob, &v)?
        };
        let mut joined = try_vec(rows * 120)?;
        for b in 0..batch {
            for h in 0..8 {
                for s in 0..t {
                    for d in 0..15 {
                        joined[(b * t + s) * 120 + h * 15 + d] =
                            context[((b * 8 + h) * t + s) * 15 + d];
                    }
                }
            }
        }
        let mut residual = self.proj.run(cpu, &joined, rows)?;
        for (a, b) in residual.iter_mut().zip(x) {
            *a += b;
        }
        let norm = self.norm2.run(cpu, &residual, rows)?;
        let hidden = self.up.run(cpu, &norm, rows)?;
        // The pinned export uses SiLU (swish), not GELU, in both SVTR MLPs.
        let hidden = cpu.swish(rows, 240, &hidden)?;
        let out = self.down.run(cpu, &hidden, rows)?;
        for (a, b) in residual.iter_mut().zip(out) {
            *a += b;
        }
        Ok(residual)
    }
}

/// The PP-OCR text recognizer: packed weights on one oneDNN backend and the
/// recognition dictionary. It reads the text in each quadrilateral.
///
/// `Send + Sync`; calls block, one [`Recognizer::recognize`] at a time.
/// Values depend on the input and the ISA, never on the thread count.
pub struct Recognizer {
    assets: Assets,
    segments: Segments,
    cpu: Arc<OnednnCpu>,
    blocks: [Block; 2],
    norm: Norm,
    head: Linear,
    options: RecOptions,
    /// One [`Recognizer::recognize`] at a time bounds the crops and batch
    /// activations held at once.
    execute: Mutex<()>,
}
impl std::fmt::Debug for Recognizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recognizer")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl Recognizer {
    #[cfg(test)]
    pub(crate) fn load_on(
        cpu: Arc<OnednnCpu>,
        model_dir: impl AsRef<std::path::Path>,
        options: RecOptions,
        verify_weights: bool,
    ) -> Result<Self> {
        options.validate()?;
        let mut config = crate::ppocr::weights::AssetConfig::new(
            model_dir.as_ref(),
            crate::ppocr::weights::ModelKind::Rec,
        );
        config.verify_weights = verify_weights;
        let assets = Assets::load(&config)?;
        Self::from_assets(cpu, assets, options)
    }
    pub(crate) fn from_assets(
        cpu: Arc<OnednnCpu>,
        assets: Assets,
        options: RecOptions,
    ) -> Result<Self> {
        options.validate()?;
        // Up to 200 output widths, their batch sizes, and the SVTR/head ops.
        cpu.set_primitive_capacity(options.plan_capacity.saturating_mul(8).max(256))?;
        let segments = rec_graph::build(cpu.clone(), &assets, options.plan_capacity)?;
        let blocks = [
            Block::load(&cpu, &assets, 0)?,
            Block::load(&cpu, &assets, 1)?,
        ];
        let norm = Norm::load(&assets, 4)?;
        let head = Linear::load(&cpu, &assets, 12, 223, 120, CTC_CLASSES)?;
        Ok(Self {
            assets,
            segments,
            cpu,
            blocks,
            norm,
            head,
            options,
            execute: Mutex::new(()),
        })
    }
    pub fn options(&self) -> &RecOptions {
        &self.options
    }
    /// The backend's memory and counters now.
    pub fn backend_stats(&self) -> crate::cpu::runtime::BackendStats {
        self.cpu.stats()
    }
    fn validate_shape(&self, shape: &[usize]) -> Result<()> {
        crate::check_shape!(
            shape.len() == 4
                && shape[0] > 0
                && shape[0] <= self.options.max_batch
                && shape[1] == 3
                && shape[2] == 48
                && (self.options.min_width..=self.options.max_width).contains(&shape[3]),
            "recognizer requires [N,3,48,W], configured batch/width limits"
        );
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn warmup(
        &self,
        batch: usize,
        width: usize,
    ) -> Result<Vec<crate::cpu::onednn::conv::PlanInfo>> {
        self.validate_shape(&[batch, 3, 48, width])?;
        let a = self.segments.backbone.warmup([batch, 3, 48, width])?;
        let t = a.output_shape[3];
        Ok(vec![
            a,
            self.segments.neck_in.warmup([batch, 480, 1, t])?,
            self.segments.neck_out.warmup([batch, 120, 1, t])?,
            self.segments.fusion.warmup([batch, 960, 1, t])?,
        ])
    }
    /// Network up to the CTC head: `[batch * t, 120]` features and `t`.
    fn features(
        &self,
        shape: &[usize],
        input: &[f32],
        widths: &[usize],
    ) -> Result<(Vec<f32>, usize)> {
        net::validate(shape, input.len())?;
        self.validate_shape(shape)?;
        let threads = self.cpu.threads();
        crate::check_shape!(
            all_par(threads, input, f32::is_finite),
            "nonfinite recognizer input"
        );
        crate::check_shape!(
            widths.len() == shape[0]
                && widths
                    .iter()
                    .all(|&w| w >= self.options.min_width && w <= shape[3]),
            "invalid recognizer valid widths"
        );
        if self.options.batching == Batching::Exact && widths.iter().any(|&w| w != shape[3]) {
            return Err(Error::config("padding requires Bucketed batching"));
        }
        let batch = shape[0];
        let (s, original) = self
            .segments
            .backbone
            .run([batch, 3, 48, shape[3]], input)?;
        let t = s[3];
        let lengths = widths.iter().map(|&w| ctc_steps(w)).collect::<Vec<_>>();
        let (_, neck) = self.segments.neck_in.run(s, &original)?;
        let mut tokens = transpose(&neck, batch, 120, t, false)?;
        for block in &self.blocks {
            tokens = block.run(&self.cpu, &tokens, batch, t, &lengths)?;
        }
        let tokens = self.norm.run(&self.cpu, &tokens, batch * t)?;
        let neck = transpose(&tokens, batch, 120, t, true)?;
        let (_, neck) = self.segments.neck_out.run([batch, 120, 1, t], &neck)?;
        let mut combined = try_vec(checked_size("recognizer concat", &[batch, 960, t])?)?;
        for b in 0..batch {
            combined[b * 960 * t..b * 960 * t + 480 * t]
                .copy_from_slice(&original[b * 480 * t..(b + 1) * 480 * t]);
            combined[b * 960 * t + 480 * t..(b + 1) * 960 * t]
                .copy_from_slice(&neck[b * 480 * t..(b + 1) * 480 * t]);
        }
        let (_, features) = self.segments.fusion.run([batch, 960, 1, t], &combined)?;
        Ok((transpose(&features, batch, 120, t, false)?, t))
    }
    /// Valid widths select attention keys and the CTC span for opt-in padding.
    pub(crate) fn forward_padded(
        &self,
        shape: &[usize],
        input: &[f32],
        widths: &[usize],
    ) -> Result<Tensor> {
        let (features, t) = self.features(shape, input, widths)?;
        let batch = shape[0];
        let logits = self.head.run(&self.cpu, &features, batch * t)?;
        let probabilities = self.cpu.softmax(batch * t, CTC_CLASSES, &logits)?;
        crate::check_shape!(
            all_par(self.cpu.threads(), &probabilities, f32::is_finite),
            "nonfinite recognizer output"
        );
        Tensor::new(vec![batch, t, CTC_CLASSES], probabilities)
    }
    /// [`Recognizer::forward_padded`] then [`Recognizer::decode`] with each
    /// line's valid length, without keeping the whole probability tensor:
    /// the CTC head, softmax and argmax run in blocks of up to `CHUNK_ROWS`
    /// time steps (rows padded to `ROW_TILE`), keeping one class id and its
    /// probability per step.
    pub(crate) fn recognize_padded(
        &self,
        shape: &[usize],
        input: &[f32],
        widths: &[usize],
    ) -> Result<Vec<DecodedLine>> {
        let (features, t) = self.features(shape, input, widths)?;
        let (batch, threads) = (shape[0], self.cpu.threads());
        let lengths = widths.iter().map(|&w| ctc_steps(w)).collect::<Vec<_>>();
        let rows = batch * t;
        let mut ids = try_vec::<usize>(rows)?;
        let mut probs = try_vec::<f32>(rows)?;
        let mut padded = Vec::new();
        // Batches running side by side each hold a block of logits and
        // probabilities (12 MiB each at 256 rows); smaller blocks keep their
        // sum down, and one thread gains nothing from larger ones.
        let block = if crate::cpu::threads::in_parallel() {
            4 * ROW_TILE
        } else {
            CHUNK_ROWS
        };
        for start in (0..rows).step_by(block) {
            let m = (rows - start).min(block);
            let tile = m.next_multiple_of(ROW_TILE);
            let src = &features[start * 120..(start + m) * 120];
            let src = if tile == m {
                src
            } else {
                padded.clear();
                padded.extend_from_slice(src);
                padded.resize(tile * 120, 0.);
                &padded[..]
            };
            let logits = self.head.run(&self.cpu, src, tile)?;
            let chunk = self.cpu.softmax(tile, CTC_CLASSES, &logits)?;
            drop(logits);
            let ok = AtomicBool::new(true);
            let (pi, pp) = (
                ids[start..].as_mut_ptr() as usize,
                probs[start..].as_mut_ptr() as usize,
            );
            crate::cpu::kernels::for_ranges(
                threads,
                m,
                ROW_GRAIN.div_ceil(CTC_CLASSES),
                |lo, hi| {
                    // SAFETY: workers receive disjoint row ranges of `ids` and `probs`.
                    let (ids, probs) = unsafe {
                        (
                            crate::cpu::kernels::rows_mut(pi as *mut usize, lo, hi, 1),
                            crate::cpu::kernels::rows_mut(pp as *mut f32, lo, hi, 1),
                        )
                    };
                    for ((i, id), p) in (lo..hi).zip(ids).zip(probs) {
                        let row = &chunk[i * CTC_CLASSES..(i + 1) * CTC_CLASSES];
                        if !row.iter().all(|v| v.is_finite() && (0. ..=1.).contains(v)) {
                            ok.store(false, Ordering::Relaxed);
                            return;
                        }
                        let r = start + i;
                        if r % t < lengths[r / t] {
                            *id = first_argmax(row);
                            *p = row[*id];
                        }
                    }
                },
            );
            crate::check_shape!(ok.into_inner(), "invalid CTC probability");
        }
        Ok(assemble(
            &ids,
            &probs,
            batch,
            t,
            CTC_CLASSES,
            self.assets.dictionary(),
            |b| lengths[b],
        ))
    }
    #[cfg(test)]
    pub(crate) fn decode(
        &self,
        probabilities: &Tensor,
        lengths: Option<&[usize]>,
    ) -> Result<Vec<DecodedLine>> {
        decode(
            probabilities,
            self.assets.dictionary(),
            lengths,
            self.cpu.threads(),
        )
    }
    #[cfg(feature = "debug-dumps")]
    #[doc(hidden)]
    pub(crate) fn debug_backbone(
        &self,
        count: usize,
        shape: [usize; 4],
        input: &[f32],
    ) -> Result<Tensor> {
        self.validate_shape(&shape)?;
        net::validate(&shape, input.len())?;
        let (s, v) = self.segments.backbone.prefix(count)?.run(shape, input)?;
        Tensor::new(s.to_vec(), v)
    }
}
impl Recognizer {
    /// Read the text in each of `quads` on `image`: warp and resize every
    /// quad to a 48-pixel line, run them in width groups, then greedy CTC.
    /// Returns one line per quad, in `quads` order; callers choose their own
    /// score threshold.
    pub fn recognize(&self, image: &ImageView<'_>, quads: &[Quad]) -> Result<Vec<DecodedLine>> {
        self.recognize_timed(image, quads).map(|(lines, _)| lines)
    }
    /// [`Recognizer::recognize`] plus where the time went.
    pub fn recognize_timed(
        &self,
        image: &ImageView<'_>,
        quads: &[Quad],
    ) -> Result<(Vec<DecodedLine>, RecTimings)> {
        let _execute = self
            .execute
            .lock()
            .map_err(|_| Error::poisoned("recognizer execution"))?;
        let rec = &self.options;
        let threads = self.cpu.threads();
        let crop_start = Instant::now();
        let mut slots = Vec::new();
        slots.try_reserve_exact(quads.len()).map_err(|_| {
            Error::out_of_memory(
                quads
                    .len()
                    .saturating_mul(std::mem::size_of::<OnceLock<Result<RecCrop>>>()),
            )
        })?;
        slots.resize_with(quads.len(), OnceLock::new);
        // Each crop is independent. Fixed single-crop tasks preserve all pixel
        // operation orders and reading order at every thread count.
        crate::cpu::threads::par_for(threads, quads.len(), 1, |lo, hi| {
            for i in lo..hi {
                let crop = crate::image::recognizer_crop(
                    image,
                    quads[i],
                    rec.min_width,
                    rec.max_width,
                    rec.max_pixels,
                );
                let _ = slots[i].set(crop);
            }
        });
        let mut crops = Vec::new();
        crops.try_reserve_exact(quads.len()).map_err(|_| {
            Error::out_of_memory(quads.len().saturating_mul(std::mem::size_of::<RecCrop>()))
        })?;
        for slot in slots {
            crops.push(
                slot.into_inner()
                    .ok_or_else(|| Error::internal("missing OCR crop task"))??,
            );
        }
        let crop_ms = ms_since(crop_start);
        let rec_start = Instant::now();
        let mut order = (0..crops.len()).collect::<Vec<_>>();
        order.sort_by_key(|&i| {
            std::cmp::Reverse(batch_width(crops[i].width, rec.batching, rec.max_width))
        });
        // Width groups, widest first: [start, end) of `order` and the width.
        let mut groups = Vec::new();
        let mut start = 0;
        while start < order.len() {
            let width = batch_width(crops[order[start]].width, rec.batching, rec.max_width);
            let mut end = start + 1;
            let max_batch = rec.max_batch.min(GROUP_COLUMNS / width).max(1);
            while end < order.len()
                && end - start < max_batch
                && batch_width(crops[order[end]].width, rec.batching, rec.max_width) == width
            {
                end += 1;
            }
            groups.push((start, end, width));
            start = end;
        }
        let widths = crops.iter().map(|c| c.width).collect::<Vec<_>>();
        let values = crops
            .into_iter()
            .map(|c| Mutex::new(c.values))
            .collect::<Vec<_>>();
        let run = |&(start, end, width): &(usize, usize, usize)| -> Result<Batch> {
            let batch = end - start;
            let pack_start = Instant::now();
            let plane = 48 * width;
            let mut input = try_vec(checked_size("OCR crop batch", &[batch, 3, plane])?)?;
            let mut valid = Vec::with_capacity(batch);
            for (b, &i) in order[start..end].iter().enumerate() {
                let w = widths[i];
                valid.push(w);
                let crop = std::mem::take(
                    &mut *values[i].lock().map_err(|_| Error::poisoned("OCR crop"))?,
                );
                for c in 0..3 {
                    for y in 0..48 {
                        input[(b * 3 + c) * plane + y * width..(b * 3 + c) * plane + y * width + w]
                            .copy_from_slice(&crop[(c * 48 + y) * w..(c * 48 + y + 1) * w]);
                    }
                }
            }
            let forward_start = Instant::now();
            let pack_ms = ms_since(pack_start);
            let lines = self.recognize_padded(&[batch, 3, 48, width], &input, &valid)?;
            Ok(Batch {
                lines,
                pack_ms,
                forward_ms: ms_since(forward_start),
            })
        };
        // Groups run one per worker, each worker taking the next group
        // when it finishes: small batches barely scale across threads, so
        // running several side by side uses the cores they leave idle. Each
        // batch's arithmetic is unchanged; only its threading differs. The
        // workers hold the compute lock throughout, so once another model in
        // the process waits for it they stop taking groups, and the rest run
        // one at a time across every thread, interleaving with that model's
        // calls as before, until it has gone quiet.
        let results = groups
            .iter()
            .map(|_| OnceLock::new())
            .collect::<Vec<OnceLock<Result<Batch>>>>();
        let contended = || crate::cpu::threads::compute_contended(CONTENTION_WINDOW);
        let mut next_group = 0;
        while next_group < groups.len() {
            if threads < 2 || contended() {
                let _ = results[next_group].set(run(&groups[next_group]));
                next_group += 1;
                continue;
            }
            let next = AtomicUsize::new(next_group);
            crate::cpu::threads::par_for(threads, threads, 1, |_, _| {
                while !contended() {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(group) = groups.get(k) else { break };
                    let _ = results[k].set(run(group));
                }
            });
            // Every group taken below the end has run.
            next_group = next.into_inner().min(groups.len());
        }
        let mut decoded = try_vec::<Option<DecodedLine>>(quads.len())?;
        let mut timings = RecTimings {
            crop_ms,
            batches: groups.len(),
            ..RecTimings::default()
        };
        for (&(start, end, _), slot) in groups.iter().zip(results) {
            let batch = slot
                .into_inner()
                .ok_or_else(|| Error::internal("missing OCR batch task"))??;
            timings.pack_ms += batch.pack_ms;
            timings.forward_ms += batch.forward_ms;
            for (&i, line) in order[start..end].iter().zip(batch.lines) {
                decoded[i] = Some(line);
            }
        }
        timings.recognize_ms = ms_since(rec_start);
        let lines = decoded
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::internal("missing recognized line"))?;
        Ok((lines, timings))
    }
}
/// Where [`Recognizer::recognize_timed`] spent its time, in milliseconds.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RecTimings {
    /// Warping and resizing every quad to a line crop.
    pub crop_ms: f64,
    /// Width groups, from the first group starting to the last finishing.
    pub recognize_ms: f64,
    /// Crop packing, then network + CTC head + greedy decode, summed over
    /// width groups. Groups that run side by side each count their own
    /// time, so the sums can exceed `recognize_ms`.
    pub pack_ms: f64,
    pub forward_ms: f64,
    pub batches: usize,
}
struct Batch {
    lines: Vec<DecodedLine>,
    pack_ms: f64,
    forward_ms: f64,
}
/// Largest recognizer group in crop columns (batch x width). Groups run
/// side by side, so this bounds the activations each worker holds at once;
/// wide lines go in smaller batches.
const GROUP_COLUMNS: usize = 2560;
/// How recently another caller must have waited for the compute lock for
/// recognizer groups to stop running side by side.
const CONTENTION_WINDOW: std::time::Duration = std::time::Duration::from_millis(100);
fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}
fn batch_width(width: usize, mode: Batching, max: usize) -> usize {
    if mode == Batching::Exact {
        return width;
    }
    if width <= 320 {
        return width.div_ceil(32).saturating_mul(32).min(max);
    }
    let mut bucket = 320;
    while bucket < width {
        bucket = (bucket * 5).div_ceil(4);
    }
    bucket.min(max)
}
impl OcrNet for Recognizer {
    fn forward(&self, shape: &[usize], input: &[f32]) -> Result<Tensor> {
        net::validate(shape, input.len())?;
        self.validate_shape(shape)?;
        self.forward_padded(shape, input, &vec![shape[3]; shape[0]])
    }
}
fn transpose(x: &[f32], batch: usize, c: usize, t: usize, to_nchw: bool) -> Result<Vec<f32>> {
    let mut y = try_vec(x.len())?;
    for b in 0..batch {
        for s in 0..t {
            for ch in 0..c {
                let n = (b * c + ch) * t + s;
                let m = (b * t + s) * c + ch;
                if to_nchw {
                    y[n] = x[m];
                } else {
                    y[m] = x[n];
                }
            }
        }
    }
    Ok(y)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DecodedLine {
    pub text: String,
    pub score: f32,
}
// Independent lanes break the scalar max/index dependency over 11,947
// classes. Strict comparisons and the final smallest-index tie rule preserve
// greedy CTC exactly. The caller validates nonempty finite probability rows.
fn first_argmax(row: &[f32]) -> usize {
    debug_assert!(!row.is_empty());
    let mut maxima = [row[0]; 8];
    let mut indices = [0usize; 8];
    let (chunks, tail) = row.as_chunks::<8>();
    for (block, values) in chunks.iter().enumerate() {
        for lane in 0..8 {
            if values[lane] > maxima[lane] {
                maxima[lane] = values[lane];
                indices[lane] = block * 8 + lane;
            }
        }
    }
    let mut id = 0;
    for lane in 0..8 {
        if maxima[lane] > row[id] || (maxima[lane] == row[id] && indices[lane] < id) {
            id = indices[lane];
        }
    }
    for (offset, &value) in tail.iter().enumerate() {
        if value > row[id] {
            id = chunks.len() * 8 + offset;
        }
    }
    id
}

/// CTC steps the recognizer emits for a crop `width` pixels wide: the
/// network downsamples width by 8.
pub(crate) fn ctc_steps(width: usize) -> usize {
    width.div_ceil(4) / 2
}

/// Greedy CTC decode: first argmax on ties; repeats collapse before the
/// blank is removed. Row validation and argmax run on `threads` workers;
/// text and scores are assembled sequentially in the original order.
#[cfg(any(test, feature = "reference"))]
pub fn decode(
    prob: &Tensor,
    dictionary: &[String],
    lengths: Option<&[usize]>,
    threads: usize,
) -> Result<Vec<DecodedLine>> {
    net::validate(&prob.shape, prob.values.len())?;
    crate::check_shape!(
        prob.shape.len() == 3
            && prob.shape[2] == dictionary.len() + 2
            && !dictionary.iter().any(|s| s.is_empty()),
        "CTC classes/dictionary mismatch"
    );
    let (batch, t, classes) = (prob.shape[0], prob.shape[1], prob.shape[2]);
    crate::check_shape!(
        lengths.is_none_or(|a| a.len() == batch && a.iter().all(|&n| n > 0 && n <= t)),
        "invalid CTC lengths"
    );
    let length = |b: usize| lengths.map_or(t, |a| a[b]);
    let mut ids = try_vec::<usize>(batch * t)?;
    let mut probs = try_vec::<f32>(batch * t)?;
    let ok = AtomicBool::new(true);
    let (pi, pp) = (ids.as_mut_ptr() as usize, probs.as_mut_ptr() as usize);
    crate::cpu::kernels::for_ranges(threads, batch * t, ROW_GRAIN.div_ceil(classes), |lo, hi| {
        // SAFETY: workers receive disjoint row ranges of `ids` and `probs`.
        let (ids, probs) = unsafe {
            (
                crate::cpu::kernels::rows_mut(pi as *mut usize, lo, hi, 1),
                crate::cpu::kernels::rows_mut(pp as *mut f32, lo, hi, 1),
            )
        };
        for ((r, id), p) in (lo..hi).zip(ids).zip(probs) {
            let row = &prob.values[r * classes..(r + 1) * classes];
            if !row.iter().all(|v| v.is_finite() && (0. ..=1.).contains(v)) {
                ok.store(false, Ordering::Relaxed);
                return;
            }
            if r % t < length(r / t) {
                *id = first_argmax(row);
                *p = row[*id];
            }
        }
    });
    crate::check_shape!(ok.into_inner(), "invalid CTC probability");
    Ok(assemble(
        &ids, &probs, batch, t, classes, dictionary, length,
    ))
}
/// Collapse per-step class ids (`probs[r]` is the probability of `ids[r]`)
/// into text and mean score.
fn assemble(
    ids: &[usize],
    probs: &[f32],
    batch: usize,
    t: usize,
    classes: usize,
    dictionary: &[String],
    length: impl Fn(usize) -> usize,
) -> Vec<DecodedLine> {
    let mut result = Vec::new();
    for b in 0..batch {
        let mut text = String::new();
        let mut total = 0f32;
        let mut count = 0;
        let mut previous = 0;
        for r in b * t..b * t + length(b) {
            let id = ids[r];
            if id != 0 && id != previous {
                if id == classes - 1 {
                    text.push(' ');
                } else {
                    text.push_str(&dictionary[id - 1]);
                }
                total += probs[r];
                count += 1;
            }
            previous = id;
        }
        result.push(DecodedLine {
            text,
            score: if count == 0 { 0. } else { total / count as f32 },
        });
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn width_buckets() {
        for w in 16..=1600 {
            assert_eq!(batch_width(w, Batching::Exact, 1600), w);
            assert!((w..=1600).contains(&batch_width(w, Batching::Bucketed, 1600)));
        }
        fn send_sync<T: Send + Sync>() {}
        send_sync::<Recognizer>();
    }
    #[test]
    fn ctc_argmax_first_tie_across_lanes_and_tail() {
        let mut row = vec![0.; CTC_CLASSES];
        row[19] = 1.;
        row[7] = 1.;
        row[CTC_CLASSES - 1] = 1.;
        assert_eq!(first_argmax(&row), 7);
        row[7] = 0.;
        assert_eq!(first_argmax(&row), 19);
        row[19] = 0.;
        assert_eq!(first_argmax(&row), CTC_CLASSES - 1);
        row[0] = 1.;
        assert_eq!(first_argmax(&row), 0);
        assert_eq!(first_argmax(&[0.]), 0);
    }
    #[test]
    fn ctc_decode_is_thread_count_invariant() {
        let (batch, t) = (3, 40);
        let mut seed = 0x9e37_79b9u32;
        let values = (0..batch * t * CTC_CLASSES)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1 << 24) as f32
            })
            .collect();
        let mut prob = Tensor::new(vec![batch, t, CTC_CLASSES], values).unwrap();
        let d = (0..DICTIONARY_ENTRIES)
            .map(|i| char::from_u32(0x4e00 + i as u32).unwrap().to_string())
            .collect::<Vec<_>>();
        let lengths = [40, 17, 1];
        let one = decode(&prob, &d, Some(&lengths), 1).unwrap();
        let four = decode(&prob, &d, Some(&lengths), 4).unwrap();
        for (a, b) in one.iter().zip(&four) {
            assert_eq!(a.text, b.text);
            assert_eq!(a.score.to_bits(), b.score.to_bits());
        }
        // Rows past a line's length are still validated.
        prob.values[(2 * t + 5) * CTC_CLASSES] = f32::NAN;
        assert!(decode(&prob, &d, Some(&lengths), 4).is_err());
    }
    #[test]
    fn ctc_blank_repeats_ties_and_scores() {
        let mut v = vec![0.; 7 * 4];
        for (i, (id, score)) in [
            (1, 0.9),
            (1, 0.8),
            (0, 1.),
            (1, 0.6),
            (2, 0.7),
            (3, 0.8),
            (0, 1.),
        ]
        .into_iter()
        .enumerate()
        {
            v[i * 4 + id] = score;
        }
        let t = Tensor::new(vec![1, 7, 4], v).unwrap();
        let d = vec!["가".into(), "나".into()];
        let got = decode(&t, &d, None, 1).unwrap();
        assert_eq!(got[0].text, "가가나 ");
        assert!((got[0].score - 0.75).abs() < 1e-6);
        assert_eq!(decode(&t, &d, Some(&[3]), 2).unwrap()[0].text, "가");
        assert!(decode(&t, &d, Some(&[8]), 1).is_err());
        assert!(decode(&t, &[], None, 1).is_err());
        let t = Tensor::new(vec![1, 1, 4], vec![0.5, 0.5, 0., 0.]).unwrap();
        assert_eq!(decode(&t, &d, None, 1).unwrap()[0].text, "");
    }
}
