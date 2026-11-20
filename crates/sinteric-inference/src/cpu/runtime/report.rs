//! What a model's load did and what its backend holds, as values a host can
//! log or export. The library prints nothing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Serialize;

use super::Isa;

/// How a model's weights were loaded.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[non_exhaustive]
pub struct LoadReport {
    /// Where each set of packed weights came from, in load order
    /// (EmbeddingGemma 2: its text tower, then image and audio when loaded).
    pub packed: Vec<PackedWeights>,
    /// Seconds spent in each load phase, by name (`mmap_s`, `pack_s`,
    /// `cache_write_s`, ...).
    pub phases_s: BTreeMap<String, f64>,
}

/// One set of packed weights.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct PackedWeights {
    /// Which part of the model (`text`, `image`, `audio`, `encoder`).
    pub part: &'static str,
    pub source: WeightSource,
}

/// Where packed weights came from.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "from")]
#[non_exhaustive]
pub enum WeightSource {
    /// Mapped from the packed-weight cache file.
    Cache { path: PathBuf },
    /// Packed from the safetensors file; `cache` says what became of the
    /// cache file.
    Safetensors { cache: CacheWrite },
}

/// What became of the packed-weight cache file on a load that packed.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "result")]
#[non_exhaustive]
pub enum CacheWrite {
    /// The config turned the cache off.
    Off,
    /// Written, for the next load to map.
    Written { path: PathBuf },
    /// Not written, and why. The load went on from safetensors.
    Failed { path: PathBuf, reason: String },
}

/// A backend's memory and counters at one moment: what it holds between
/// calls, and what its calls have cost so far.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct BackendStats {
    /// The ISA its primitives run on.
    pub isa: Isa,
    pub exec_slots: usize,
    /// Packed matmul weights.
    pub packed_weight_bytes: usize,
    /// Packed convolution weights.
    pub conv_packed_weight_bytes: usize,
    /// What it keeps between calls, against [`BackendStats::memory_budget_bytes`].
    pub retained_bytes: usize,
    pub memory_budget_bytes: usize,
    pub plan_bytes: usize,
    pub conv_plan_bytes: usize,
    pub conv_arena_bytes: usize,
    pub conv_scratch_bytes: usize,
    pub execute_scratch_bytes: usize,
    pub retained_output_buffer_bytes: usize,
    /// Rows and columns of the pooled activation buffers' current shape.
    pub active_pool_shape: Option<(usize, usize)>,
    pub cached_plans: usize,
    pub plan_capacity: usize,
    pub plan_creations: u64,
    pub capacity_evictions: u64,
    pub budget_evictions: u64,
    pub budget_evicted_bytes: u64,
    pub budget_trims: usize,
    pub cached_weight_layouts: usize,
    /// oneDNN's implementation name of each cached plan.
    pub implementations: Vec<String>,
    pub primitive_setup_s: f64,
    /// Seconds and executions by oneDNN primitive kind.
    pub primitive_exec_s: BTreeMap<i32, f64>,
    pub executions: BTreeMap<i32, u64>,
    pub bf16_convert_s: f64,
    /// Seconds by the model's own operator category.
    pub category_exec_s: BTreeMap<String, f64>,
}
