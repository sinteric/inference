//! Process-wide runtime options and the packed-weight cache config shared by
//! every model. The library path never reads process environment.

use std::path::PathBuf;

use crate::cpu::onednn::{CacheOptions, CpuIsaCap, SessionOptions, DEFAULT_WORKSPACE_LIMIT};
use crate::cpu::threads::CpuTopology;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

fn default_true() -> bool {
    true
}
fn default_threads() -> usize {
    CpuTopology::detect().physical_cores.max(1)
}
fn default_workspace() -> usize {
    DEFAULT_WORKSPACE_LIMIT
}
fn default_exec_slots() -> usize {
    1
}

/// Packed-weight cache.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct CacheConfig {
    /// On by default.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Directory for the sidecar file (joined with the auto-generated name).
    /// Ignored when [`Self::file`] is set.
    #[serde(default)]
    pub dir: Option<PathBuf>,
    /// Explicit cache file path.
    #[serde(default)]
    pub file: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: None,
            file: None,
        }
    }
}

impl CacheConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Backend cache options for one model load. `dir` is joined with the
    /// versioned sidecar name; an explicit `file` wins.
    pub fn options(&self, verify_weights: bool, manifest: Option<PathBuf>) -> CacheOptions {
        let file = if self.file.is_some() {
            self.file.clone()
        } else {
            self.dir.as_ref().map(|d| {
                d.join(format!(
                    "packed-cache.onednn-3.13.4.v{}",
                    crate::cpu::onednn::CACHE_VERSION
                ))
            })
        };
        CacheOptions {
            enabled: self.enabled,
            file,
            verify_weights,
            manifest,
        }
    }
}

/// oneDNN dispatch cap. Default is the best ISA the CPU has.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum CpuIsa {
    #[default]
    BestAvailable,
    Avx512CoreBf16,
    Avx2,
}

impl From<CpuIsa> for CpuIsaCap {
    fn from(v: CpuIsa) -> Self {
        match v {
            CpuIsa::BestAvailable => CpuIsaCap::BestAvailable,
            CpuIsa::Avx512CoreBf16 => CpuIsaCap::Avx512CoreBf16,
            CpuIsa::Avx2 => CpuIsaCap::Avx2,
        }
    }
}

impl CpuIsa {
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match CpuIsaCap::from_name(name)? {
            CpuIsaCap::BestAvailable => Self::BestAvailable,
            CpuIsaCap::Avx512CoreBf16 => Self::Avx512CoreBf16,
            CpuIsaCap::Avx2 => Self::Avx2,
        })
    }
}

/// Process-global runtime options. Construct once via [`crate::cpu::runtime::Runtime::new`].
///
/// Owns oneDNN ISA cap, pinned thread-pool width, cpuset/pin, and the
/// per-model execute-slot default. There is no `library` path: oneDNN is
/// statically linked at build time. Does not mutate process environment.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct RuntimeOptions {
    /// Pinned pool width. Default is detected physical cores (not a constant 4).
    #[serde(default = "default_threads")]
    pub threads: usize,
    /// Explicit cpuset (`"0,2,4,6"`). `None` picks physical cores.
    #[serde(default)]
    pub cpuset: Option<String>,
    /// Process-wide `sched_setaffinity`. CLI/dedicated inference process only;
    /// default off for in-process embedding. Does not set `OMP_*` / `GOMP_*`.
    #[serde(default)]
    pub pin_process: bool,
    #[serde(default)]
    pub cpu_isa: CpuIsa,
    /// `dnnl_set_verbose` level.
    #[serde(default)]
    pub verbose: i32,
    #[serde(default = "default_workspace")]
    pub workspace_limit_bytes: usize,
    /// Concurrent oneDNN workspaces per loaded model. Default `1` serializes
    /// executes (same wall time as a single stream). Increase so concurrent
    /// blocking model calls each get a stream+scratch. Parallel
    /// regions still share the process-global pool (one `parallel_for` at a
    /// time) so AMX tile config stays on one primitive.
    #[serde(default = "default_exec_slots")]
    pub exec_slots: usize,
    /// Default memory budget of each loaded model, in bytes: what it may keep
    /// between calls (cached plans, pooled activations, convolution arena).
    /// Over it the model drops caches and returns pages to the OS instead of
    /// failing. `0` (default) is unbounded. Model configs can override it.
    #[serde(default)]
    pub memory_budget_bytes: usize,
    /// Cap glibc malloc arenas (`mallopt(M_ARENA_MAX)`) in [`crate::cpu::runtime::Runtime::new`].
    /// Freed heap kept per calling thread otherwise grows with concurrency.
    /// Process-wide, so off (`None`) by default.
    #[serde(default)]
    pub malloc_arena_max: Option<usize>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeOptions {
    pub fn new() -> Self {
        Self {
            threads: default_threads(),
            cpuset: None,
            pin_process: false,
            cpu_isa: CpuIsa::BestAvailable,
            verbose: 0,
            workspace_limit_bytes: default_workspace(),
            exec_slots: default_exec_slots(),
            memory_budget_bytes: 0,
            malloc_arena_max: None,
        }
    }

    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = n;
        self
    }
    pub fn with_cpuset(mut self, cpuset: impl Into<String>) -> Self {
        self.cpuset = Some(cpuset.into());
        self
    }
    pub fn with_cpu_isa(mut self, isa: CpuIsa) -> Self {
        self.cpu_isa = isa;
        self
    }
    pub fn with_pin_process(mut self, pin: bool) -> Self {
        self.pin_process = pin;
        self
    }
    pub fn with_workspace_limit_bytes(mut self, n: usize) -> Self {
        self.workspace_limit_bytes = n;
        self
    }
    pub fn with_exec_slots(mut self, n: usize) -> Self {
        self.exec_slots = n;
        self
    }
    pub fn with_memory_budget_bytes(mut self, n: usize) -> Self {
        self.memory_budget_bytes = n;
        self
    }
    pub fn with_malloc_arena_max(mut self, n: usize) -> Self {
        self.malloc_arena_max = Some(n);
        self
    }
    pub fn with_verbose(mut self, verbose: i32) -> Self {
        self.verbose = verbose;
        self
    }

    pub(crate) fn thread_budget(&self) -> Result<crate::cpu::threads::ThreadBudget> {
        if !(0..=2).contains(&self.verbose) {
            return Err(Error::Config(format!(
                "verbose {} outside 0..=2",
                self.verbose
            )));
        }
        let bytes = crate::checked_size("execute slots", &[self.exec_slots, 192])?;
        let limit = if self.workspace_limit_bytes == 0 {
            DEFAULT_WORKSPACE_LIMIT
        } else {
            self.workspace_limit_bytes
        };
        if bytes > limit {
            return Err(Error::WorkspaceLimit {
                operation: "execute slots",
                requested: bytes,
                limit,
            });
        }
        if self.exec_slots == 0 {
            return Err(Error::Config("exec_slots must be >= 1".into()));
        }
        if self.threads == 0 {
            return Err(Error::Config("threads must be >= 1".into()));
        }
        let mut b = crate::cpu::threads::ThreadBudget::new(self.threads)?;
        if let Some(cs) = &self.cpuset {
            b = b.with_cpuset(cs)?;
        }
        Ok(b)
    }

    pub(crate) fn session(&self) -> SessionOptions {
        SessionOptions {
            max_cpu_isa: self.cpu_isa.into(),
            verbose: self.verbose,
            workspace_limit: if self.workspace_limit_bytes == 0 {
                DEFAULT_WORKSPACE_LIMIT
            } else {
                self.workspace_limit_bytes
            },
            exec_slots: self.exec_slots,
            memory_budget: self.memory_budget_bytes,
            threads: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip_and_defaults() {
        let rt = RuntimeOptions::new().with_threads(4).with_exec_slots(2);
        let json = serde_json::to_string(&rt).unwrap();
        let back: RuntimeOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.threads, 4);
        assert_eq!(back.exec_slots, 2);
        assert_eq!(back.cpu_isa, CpuIsa::BestAvailable);
        assert_eq!(back.workspace_limit_bytes, 512 * 1024 * 1024);
        assert!(!back.pin_process);
        assert!(!json.contains("library"));
    }

    #[test]
    fn deserialize_subset() {
        let rt: RuntimeOptions =
            serde_json::from_str(r#"{"threads":8,"cpu_isa":"avx512_core_bf16"}"#).unwrap();
        assert_eq!(rt.threads, 8);
        assert_eq!(rt.exec_slots, 1);
        assert_eq!(rt.cpu_isa, CpuIsa::Avx512CoreBf16);
        let rt: RuntimeOptions = toml::from_str(
            r#"
threads = 8
cpu_isa = "best_available"
exec_slots = 1
"#,
        )
        .unwrap();
        assert_eq!(rt.threads, 8);
    }

    #[test]
    fn default_threads_follows_physical_cores() {
        let n = CpuTopology::detect().physical_cores.max(1);
        assert_eq!(RuntimeOptions::new().threads, n);
        assert!(n >= 1);
    }

    #[test]
    fn options_reject_zero_threads() {
        let err = RuntimeOptions::new()
            .with_threads(0)
            .thread_budget()
            .unwrap_err();
        match err {
            Error::Config(msg) => assert!(msg.contains("threads")),
            other => panic!("expected Config, got {other}"),
        }
    }

    #[test]
    fn cache_options_join_dir_and_prefer_file() {
        let mut c = CacheConfig::new();
        assert!(c.options(false, None).file.is_none());
        c.dir = Some("cache".into());
        let o = c.options(true, Some("m.toml".into()));
        assert!(o.file.unwrap().starts_with("cache"));
        assert!(o.verify_weights);
        assert_eq!(o.manifest, Some("m.toml".into()));
        c.file = Some("exact".into());
        assert_eq!(c.options(false, None).file, Some("exact".into()));
    }
}
