//! Typed load options. Apply an explicit `rt_core::env::EnvOverrides` snapshot;
//! option constructors and loaders never read process environment.

use onednn_sys::{
    dnnl_cpu_isa_avx2, dnnl_cpu_isa_avx512_core_bf16, dnnl_cpu_isa_default, dnnl_cpu_isa_t,
};
use rt_core::env::{CpuIsaOverride, EnvOverrides, ModelKind};
use std::path::PathBuf;

/// Default activation-pool cap (512 MiB). Matches the Turin EG2 workspace.
pub const DEFAULT_WORKSPACE_LIMIT: usize = 512 * 1024 * 1024;

/// oneDNN dispatch cap. Process-global: applied by [`crate::OnednnCpu::load_with`].
/// A later mismatch is an error. Models loaded from one runtime share the cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CpuIsaCap {
    /// `dnnl_set_max_cpu_isa(dnnl_cpu_isa_default)` — best ISA the CPU has.
    #[default]
    BestAvailable,
    /// Cap at AVX512_CORE_BF16 (Zen 5 / Turin emulation on AMX hosts).
    Avx512CoreBf16,
    /// Cap at AVX2: what a host without AVX-512 runs, reproducible on one
    /// that has it.
    Avx2,
}

impl CpuIsaCap {
    /// The cap a convolution or OCR test runs under: `SINTERIC_TEST_CPU_ISA`
    /// when set, AVX512_CORE_BF16 otherwise.
    #[doc(hidden)]
    pub fn for_tests() -> Self {
        rt_core::env::TestOverrides::from_env()
            .expect("valid test overrides")
            .conv_isa
            .map_or(Self::Avx512CoreBf16, Self::from)
    }

    pub fn from_name(name: &str) -> rt_core::Result<Self> {
        Ok(CpuIsaOverride::from_name(name)?.into())
    }

    pub fn as_dnnl(self) -> dnnl_cpu_isa_t {
        match self {
            Self::BestAvailable => dnnl_cpu_isa_default,
            Self::Avx512CoreBf16 => dnnl_cpu_isa_avx512_core_bf16,
            Self::Avx2 => dnnl_cpu_isa_avx2,
        }
    }
}

/// Packed-weight cache and weight-pin options for one model load.
#[derive(Clone, Debug)]
pub struct CacheOptions {
    /// On by default. Independent of `VELA_PACKED_CACHE`.
    pub enabled: bool,
    /// Explicit cache file. `None` uses the sidecar next to safetensors.
    pub file: Option<PathBuf>,
    /// Full sha256 of the 1.18 GiB safetensors (download already verified).
    pub verify_weights: bool,
    /// Override path to the asset manifest (sha/size pin).
    pub manifest: Option<PathBuf>,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            file: None,
            verify_weights: false,
            manifest: None,
        }
    }
}

impl CacheOptions {
    /// Apply only present overrides. Model-specific pins never cross models.
    pub fn with_overrides(mut self, env: &EnvOverrides, model: ModelKind) -> Self {
        if let Some(enabled) = env.packed_cache {
            self.enabled = enabled;
        }
        if let Some(file) = &env.packed_cache_file {
            self.file = Some(file.clone());
        }
        let model = env.model(model);
        if let Some(verify) = model.verify_weights {
            self.verify_weights = verify;
        }
        if let Some(manifest) = &model.manifest {
            self.manifest = Some(manifest.clone());
        }
        self
    }
}

impl From<CpuIsaOverride> for CpuIsaCap {
    fn from(isa: CpuIsaOverride) -> Self {
        match isa {
            CpuIsaOverride::BestAvailable => Self::BestAvailable,
            CpuIsaOverride::Avx512CoreBf16 => Self::Avx512CoreBf16,
            CpuIsaOverride::Avx2 => Self::Avx2,
        }
    }
}

fn default_exec_slots() -> usize {
    1
}

/// oneDNN session options (ISA cap, verbose, activation-pool bytes, execute slots).
#[derive(Clone, Debug)]
pub struct SessionOptions {
    pub max_cpu_isa: CpuIsaCap,
    /// `dnnl_set_verbose` level. 0 is quiet.
    pub verbose: i32,
    /// Activation buffer pool cap in bytes.
    pub workspace_limit: usize,
    /// Bounded oneDNN execute workspaces (stream + scratch) per backend.
    /// `1` serializes executes (same as a single stream). Increase so
    /// concurrent `embed` / `forward` each get a stream. GEMMs and glue still
    /// share the process-global thread pool (one `parallel_for` at a time).
    pub exec_slots: usize,
    /// Bytes the backend may keep between calls: cached plans, pooled
    /// activation buffers, the convolution arena and execute scratch. Over
    /// it, pooled buffers and then the least recently used plans are dropped
    /// and freed pages go back to the OS; calls never fail because of it.
    /// `0` (default) keeps today's count-bounded caches.
    pub memory_budget: usize,
    /// Pool threads this backend's operators run on: oneDNN primitives are
    /// built for and execute on at most this many workers, and so does the
    /// backend's glue. `0` (default) is the whole pool; wider than the pool
    /// is the pool. The pool itself and the compute lock stay
    /// process-global: this caps the backend's parallelism, it does not
    /// give it workers of its own.
    pub threads: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            max_cpu_isa: CpuIsaCap::BestAvailable,
            verbose: 0,
            workspace_limit: DEFAULT_WORKSPACE_LIMIT,
            exec_slots: default_exec_slots(),
            memory_budget: 0,
            threads: 0,
        }
    }
}

impl SessionOptions {
    /// Apply only present overrides; this does not read process environment.
    pub fn with_overrides(mut self, env: &EnvOverrides) -> Self {
        if let Some(isa) = env.max_cpu_isa {
            self.max_cpu_isa = isa.into();
        }
        if let Some(verbose) = env.verbose {
            self.verbose = verbose;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_snapshot_reuses_values_without_resetting_absent_options() {
        let env = EnvOverrides::from_lookup(|name| match name {
            "VELA_PACKED_CACHE" => Some("off".into()),
            "EG2_MANIFEST" => Some("eg2.toml".into()),
            "EG2_VERIFY_WEIGHTS" => Some("true".into()),
            "DNNL_VERBOSE" => Some("1".into()),
            _ => None,
        })
        .unwrap();
        let eg = CacheOptions::default().with_overrides(&env, ModelKind::EmbeddingGemma2);
        let vela = CacheOptions::default().with_overrides(&env, ModelKind::Vela);
        assert!(!eg.enabled && !vela.enabled);
        assert!(eg.verify_weights && !vela.verify_weights);
        assert_eq!(
            eg.manifest.as_deref(),
            Some(std::path::Path::new("eg2.toml"))
        );
        assert!(vela.manifest.is_none());
        let session = SessionOptions {
            exec_slots: 3,
            workspace_limit: 4096,
            max_cpu_isa: CpuIsaCap::Avx512CoreBf16,
            ..Default::default()
        }
        .with_overrides(&env);
        assert_eq!(session.verbose, 1);
        assert_eq!(session.max_cpu_isa, CpuIsaCap::Avx512CoreBf16);
        assert_eq!(session.exec_slots, 3);
        assert_eq!(session.workspace_limit, 4096);
    }

    #[test]
    fn cache_default_is_on_without_env() {
        let c = CacheOptions::default();
        assert!(c.enabled);
        assert!(c.file.is_none());
        assert!(!c.verify_weights);
        assert!(c.manifest.is_none());
    }

    #[test]
    fn isa_names() {
        assert_eq!(
            CpuIsaCap::from_name("AVX512_CORE_BF16").unwrap(),
            CpuIsaCap::Avx512CoreBf16
        );
        assert_eq!(
            CpuIsaCap::from_name("best_available").unwrap(),
            CpuIsaCap::BestAvailable
        );
        assert_eq!(CpuIsaCap::from_name("avx2").unwrap(), CpuIsaCap::Avx2);
        assert!(CpuIsaCap::from_name("AVX512_FP16").is_err());
    }

    #[test]
    fn session_default_is_best_isa_and_512mib() {
        let s = SessionOptions::default();
        assert_eq!(s.max_cpu_isa, CpuIsaCap::BestAvailable);
        assert_eq!(s.verbose, 0);
        assert_eq!(s.workspace_limit, 512 * 1024 * 1024);
        assert_eq!(s.exec_slots, 1);
        assert_eq!(s.memory_budget, 0);
    }
}
