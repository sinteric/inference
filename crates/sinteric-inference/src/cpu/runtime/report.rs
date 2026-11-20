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
    /// Where they came from.
    pub source: WeightSource,
}

/// Where packed weights came from.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "from")]
#[non_exhaustive]
pub enum WeightSource {
    /// Mapped from the packed-weight cache file.
    Cache {
        /// The cache file.
        path: PathBuf,
    },
    /// Packed from the safetensors file; `cache` says what became of the
    /// cache file.
    Safetensors {
        /// What became of the cache file.
        cache: CacheWrite,
    },
}

/// What became of the packed-weight cache file on a load that packed.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "result")]
#[non_exhaustive]
pub enum CacheWrite {
    /// The config turned the cache off.
    Off,
    /// Written, for the next load to map.
    Written {
        /// The cache file.
        path: PathBuf,
    },
    /// Not written, and why. The load went on from safetensors.
    Failed {
        /// The cache file it tried to write.
        path: PathBuf,
        /// The write's error.
        reason: String,
    },
}

/// A backend's memory and counters at one moment: what it holds between
/// calls, and what its calls have cost so far.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct BackendStats {
    /// The ISA its primitives run on.
    pub isa: Isa,
    /// Calls it runs at once ([`super::Options::exec_slots`]).
    pub exec_slots: usize,
    /// Packed matmul weights.
    pub packed_weight_bytes: usize,
    /// Packed convolution weights.
    pub conv_packed_weight_bytes: usize,
    /// What it keeps between calls, against [`BackendStats::memory_budget_bytes`].
    pub retained_bytes: usize,
    /// The most it may keep between calls; `0` is unbounded.
    pub memory_budget_bytes: usize,
    /// Cached matmul and attention plans.
    pub plan_bytes: usize,
    /// Cached convolution plans.
    pub conv_plan_bytes: usize,
    /// Idle convolution activation arenas.
    pub conv_arena_bytes: usize,
    /// Idle convolution primitive scratch.
    pub conv_scratch_bytes: usize,
    /// Idle per-call oneDNN scratch.
    pub execute_scratch_bytes: usize,
    /// Pooled activation buffers.
    pub retained_output_buffer_bytes: usize,
    /// Rows and columns of the pooled activation buffers' current shape.
    pub active_pool_shape: Option<(usize, usize)>,
    /// Matmul and attention plans in the cache.
    pub cached_plans: usize,
    /// The most plans the cache holds before it drops the least recently
    /// run.
    pub plan_capacity: usize,
    /// Plans built so far.
    pub plan_creations: u64,
    /// Plans dropped because the cache was full.
    pub capacity_evictions: u64,
    /// Plans dropped to stay within the memory budget.
    pub budget_evictions: u64,
    /// Bytes those budget evictions freed.
    pub budget_evicted_bytes: u64,
    /// Times freed pages were returned to the OS.
    pub budget_trims: usize,
    /// Weight layouts the matmul plans share.
    pub cached_weight_layouts: usize,
    /// oneDNN's implementation name of each cached plan.
    pub implementations: Vec<String>,
    /// Seconds spent building oneDNN primitives.
    pub primitive_setup_s: f64,
    /// Seconds and executions by oneDNN primitive kind.
    pub primitive_exec_s: BTreeMap<i32, f64>,
    /// Executions by oneDNN primitive kind.
    pub executions: BTreeMap<i32, u64>,
    /// Seconds converting attention inputs to BF16.
    pub bf16_convert_s: f64,
    /// Seconds by the model's own operator category.
    pub category_exec_s: BTreeMap<String, f64>,
}
