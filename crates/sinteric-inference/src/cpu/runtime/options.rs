//! The runtime's options and the packed-weight cache config every model
//! with packed weights takes. Nothing here reads the process environment.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::IsaCap;
use crate::cpu::onednn::{CacheOptions, SessionOptions, DEFAULT_WORKSPACE_LIMIT};
use crate::cpu::threads::CpuTopology;
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

/// The packed-weight cache: a file of a model's weights already in oneDNN's
/// blocked layout for this CPU, so a later load maps it instead of
/// repacking. It is keyed by the weights' hash, oneDNN's version and the
/// ISA, and rebuilt when any of them changes.
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
    pub(crate) fn options(&self, verify_weights: bool, manifest: Option<PathBuf>) -> CacheOptions {
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

/// What a [`super::Runtime`] is built with. Every field has a default
/// ([`Options::new`]); none reads the process environment.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct Options {
    /// Width of the pinned thread pool every model computes on. Default: the
    /// physical cores of the CPUs this process may run on.
    #[serde(default = "default_threads")]
    pub threads: usize,
    /// The CPUs to pin the pool to (`"0,2,4,6"`, `"0-3"`). `None` picks one
    /// CPU per physical core.
    #[serde(default)]
    pub cpuset: Option<String>,
    /// The most oneDNN may dispatch to. Default: [`IsaCap::Best`].
    #[serde(default)]
    pub isa_cap: IsaCap,
    /// The activation memory one call may hold, in bytes. A call that needs
    /// more fails with [`crate::ErrorKind::TooLarge`]. `0` is the default,
    /// 512 MiB.
    #[serde(default = "default_workspace")]
    pub workspace_limit_bytes: usize,
    /// Calls one model runs at once, each with its own oneDNN stream and
    /// scratch. Default `1`: calls on one model take turns. Their parallel
    /// regions share the one pool either way.
    #[serde(default = "default_exec_slots")]
    pub exec_slots: usize,
    /// What each loaded model may keep between calls (cached plans, pooled
    /// activations, the convolution arena), in bytes. Past it the model drops
    /// caches and returns pages to the OS rather than failing. `0` (default)
    /// is unbounded; a model's config can override it.
    #[serde(default)]
    pub memory_budget_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self::new()
    }
}

impl Options {
    pub fn new() -> Self {
        Self {
            threads: default_threads(),
            cpuset: None,
            isa_cap: IsaCap::Best,
            workspace_limit_bytes: default_workspace(),
            exec_slots: default_exec_slots(),
            memory_budget_bytes: 0,
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
    pub fn with_isa_cap(mut self, cap: IsaCap) -> Self {
        self.isa_cap = cap;
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

    pub(crate) fn thread_budget(&self) -> Result<crate::cpu::threads::ThreadBudget> {
        let bytes = crate::checked_size("execute slots", &[self.exec_slots, 192])?;
        let limit = if self.workspace_limit_bytes == 0 {
            DEFAULT_WORKSPACE_LIMIT
        } else {
            self.workspace_limit_bytes
        };
        if bytes > limit {
            return Err(Error::config(format!(
                "{} execute slots need {bytes} bytes, past the workspace limit of {limit}",
                self.exec_slots
            )));
        }
        if self.exec_slots == 0 {
            return Err(Error::config("exec_slots must be >= 1"));
        }
        if self.threads == 0 {
            return Err(Error::config("threads must be >= 1"));
        }
        let mut b = crate::cpu::threads::ThreadBudget::new(self.threads)?;
        if let Some(cs) = &self.cpuset {
            b = b.with_cpuset(cs)?;
        }
        Ok(b)
    }

    pub(crate) fn session(&self) -> SessionOptions {
        SessionOptions {
            max_cpu_isa: self.isa_cap.backend(),
            verbose: 0,
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
        let rt = Options::new().with_threads(4).with_exec_slots(2);
        let json = serde_json::to_string(&rt).unwrap();
        let back: Options = serde_json::from_str(&json).unwrap();
        assert_eq!(back.threads, 4);
        assert_eq!(back.exec_slots, 2);
        assert_eq!(back.isa_cap, IsaCap::Best);
        assert_eq!(back.workspace_limit_bytes, 512 * 1024 * 1024);
        assert!(!json.contains("library"));
    }

    #[test]
    fn deserialize_subset() {
        let rt: Options =
            serde_json::from_str(r#"{"threads":8,"isa_cap":"avx512_core_bf16"}"#).unwrap();
        assert_eq!(rt.threads, 8);
        assert_eq!(rt.exec_slots, 1);
        assert_eq!(rt.isa_cap, IsaCap::Avx512CoreBf16);
    }

    #[test]
    fn default_threads_follows_physical_cores() {
        let n = CpuTopology::detect().physical_cores.max(1);
        assert_eq!(Options::new().threads, n);
        assert!(n >= 1);
    }

    #[test]
    fn options_reject_zero_threads() {
        let err = Options::new().with_threads(0).thread_budget().unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::Config);
        assert!(err.to_string().contains("threads"), "{err}");
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
