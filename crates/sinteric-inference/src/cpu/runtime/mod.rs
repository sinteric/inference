//! The CPU runtime: [`Runtime`], built once per process from [`Options`],
//! that every model on a CPU loads from.
//!
//! ```no_run
//! use sinteric_inference::cpu;
//! # fn main() -> sinteric_inference::Result<()> {
//! let rt = cpu::Runtime::try_new(cpu::Options::new().with_threads(4))?;
//! println!("{} on {} threads", rt.isa(), rt.threads());
//! # Ok(())
//! # }
//! ```
//!
//! # What is process-wide
//!
//! | Resource | Scope | Set by |
//! |---|---|---|
//! | oneDNN's ISA cap (`dnnl_set_max_cpu_isa`) | process | [`Runtime::try_new`]; a second runtime with another cap fails |
//! | the pinned thread pool every model computes on | process | [`Runtime::try_new`]; a second runtime with another width or cpuset fails |
//! | oneDNN engine, execute slots, activation pool | per model | its load, or one shared by the models of a [`Runtime::shared_backend`] handle |
//! | packed weights, weight mmaps | per model | its load |
//!
//! The library never touches the process's own CPU affinity or its
//! allocator: a host that wants either sets it itself ([`Runtime::cpus`]
//! names the CPUs the pool is pinned to). oneDNN is linked statically
//! (pinned 3.13.4); nothing is loaded at run time.
//!
//! # Concurrency
//!
//! Model calls block; from async code run them with
//! `tokio::task::spawn_blocking`. Concurrent calls on one model each take one
//! of its execute slots ([`Options::exec_slots`], default 1). Their GEMMs
//! and glue share the one pool and run one parallel region at a time, which
//! keeps AMX tile state on one primitive.

use std::sync::Arc;

use crate::cpu::onednn::{OnednnCpu, TuningPlan};
use crate::cpu::threads::ThreadBudget;
use crate::{Error, ErrorKind, Result};

mod isa;
mod options;
mod report;

pub use isa::{AmxHealth, Isa, IsaCap};
pub use options::{CacheConfig, Options};
pub use report::{BackendStats, CacheWrite, LoadReport, PackedWeights, WeightSource};

/// Return free heap pages to the OS (glibc `malloc_trim(0)`), after a load
/// or an unload has freed large temporaries. Returns whether memory was
/// released; `false` on non-glibc targets.
pub(crate) fn release_free_memory() -> bool {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: malloc_trim only touches allocator-internal free lists.
        unsafe { libc::malloc_trim(0) != 0 }
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        false
    }
}

/// The process's CPU runtime: the ISA, the pinned thread pool and the
/// defaults every model on it loads with. Cheap to clone; clones share it.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
    /// Set on a [`Runtime::shared_backend`] handle: every model loaded
    /// through it runs on this backend.
    shared: Option<Arc<OnednnCpu>>,
}

struct RuntimeInner {
    budget: ThreadBudget,
    options: Options,
    plan: TuningPlan,
    isa: Isa,
    amx: AmxHealth,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("isa", &self.isa())
            .field("threads", &self.threads())
            .field("shares_backend", &self.shares_backend())
            .finish_non_exhaustive()
    }
}

impl Runtime {
    /// Install oneDNN's ISA cap and the pinned thread pool, then return the
    /// runtime models load from.
    ///
    /// Fails with [`ErrorKind::Unsupported`] on a CPU below AVX2, the
    /// library's floor, and with [`ErrorKind::Config`] when another runtime
    /// in this process installed a different cap, width or cpuset.
    pub fn try_new(options: Options) -> Result<Self> {
        let budget = options.thread_budget()?;
        budget.install_library_workers()?;
        // Creating a backend applies the ISA cap (after the AMX probe) first.
        let warmup = OnednnCpu::load_with(&budget, options.session())?;
        let isa = Isa::from_bits(warmup.cpu_isa).ok_or_else(|| {
            Error::new(
                ErrorKind::Unsupported,
                format!(
                    "oneDNN dispatches to ISA 0x{:x} here, below AVX2, the library's floor",
                    warmup.cpu_isa
                ),
            )
        })?;
        let amx = if options.isa_cap == IsaCap::Best && crate::cpu::onednn::amx_probe_enabled() {
            crate::cpu::onednn::amx_health().into()
        } else {
            AmxHealth::NotProbed
        };
        let plan = TuningPlan::new(
            crate::cpu::threads::CpuTopology::detect(),
            warmup.cpu_isa,
            budget.nthreads,
        );
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                budget,
                options,
                plan,
                isa,
                amx,
            }),
            shared: None,
        })
    }

    /// The options this runtime was built with.
    #[must_use]
    pub fn options(&self) -> &Options {
        &self.inner.options
    }

    /// The ISA every model on this runtime runs on.
    #[must_use]
    pub fn isa(&self) -> Isa {
        self.inner.isa
    }

    /// What the AMX probe found; [`AmxHealth::TileStateLoss`] means the
    /// runtime runs AVX512_CORE_BF16 although the CPU has AMX.
    #[must_use]
    pub fn amx_health(&self) -> AmxHealth {
        self.inner.amx
    }

    /// Width of the thread pool.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.inner.budget.nthreads
    }

    /// The CPUs the pool's threads are pinned to, one per thread. A host
    /// that dedicates its process to inference can pin the process to them.
    #[must_use]
    pub fn cpus(&self) -> &[usize] {
        &self.inner.budget.cpus
    }

    /// oneDNN's `dnnl_cpu_isa_t` value of [`Runtime::isa`], for model
    /// loaders.
    pub(crate) fn cpu_isa(&self) -> u64 {
        self.inner.plan.cpu_isa
    }

    /// The activation-memory cap of one call.
    pub(crate) fn workspace_limit(&self) -> usize {
        self.inner.options.session().workspace_limit
    }

    /// A handle on this runtime whose models all load onto one backend:
    /// one engine, execute-slot pool, activation pool, convolution arena,
    /// primitive cache and memory budget, sized by `overrides`. Models that
    /// run one after another (an OCR detector and recognizer) then keep one
    /// working set instead of one each.
    ///
    /// ```no_run
    /// # use sinteric_inference::{cpu, ppocr};
    /// # fn main() -> sinteric_inference::Result<()> {
    /// # let rt = cpu::Runtime::try_new(cpu::Options::new())?;
    /// # let (det_dir, rec_dir) = ("/models/ppocr/det", "/models/ppocr/rec");
    /// let ocr = rt.shared_backend(
    ///     cpu::BackendOverrides::default()
    ///         .with_workspace_limit_bytes(Some(1 << 30))
    ///         .with_memory_budget_bytes(Some(512 << 20)),
    /// )?;
    /// let det = ppocr::cpu::Detector::load(&ocr, ppocr::cpu::DetConfig::new(det_dir))?;
    /// let rec = ppocr::cpu::Recognizer::load(&ocr, ppocr::cpu::RecConfig::new(rec_dir))?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Workspace and budget belong to the handle: a model config loaded
    /// through it that sets either is [`ErrorKind::Config`]. The backend is
    /// freed once the handle and every model loaded through it are dropped.
    pub fn shared_backend(&self, overrides: BackendOverrides) -> Result<Runtime> {
        if self.shared.is_some() {
            return Err(Error::config(
                "this runtime handle already shares one backend",
            ));
        }
        Ok(Self {
            inner: Arc::clone(&self.inner),
            shared: Some(self.backend_with(overrides)?),
        })
    }

    /// Whether this handle came from [`Runtime::shared_backend`].
    #[must_use]
    pub fn shares_backend(&self) -> bool {
        self.shared.is_some()
    }

    /// The backend with the runtime's defaults, for tests that drive it
    /// directly.
    #[cfg(test)]
    pub(crate) fn backend(&self) -> Result<Arc<OnednnCpu>> {
        self.backend_with(BackendOverrides::default())
    }

    /// A fresh per-model backend (engine, execute slots, activation pool) on
    /// the runtime's ISA and thread pool, with `overrides` of the runtime
    /// defaults; or the shared one on a [`Runtime::shared_backend`] handle,
    /// where any override is [`ErrorKind::Config`] (the handle's settings
    /// apply). For model loaders.
    pub(crate) fn backend_with(&self, overrides: BackendOverrides) -> Result<Arc<OnednnCpu>> {
        if let Some(shared) = &self.shared {
            if overrides != BackendOverrides::default() {
                return Err(Error::config(
                    "set workspace_limit_bytes, memory_budget_bytes and threads on \
                     Runtime::shared_backend, not on a model loaded through it",
                ));
            }
            return Ok(Arc::clone(shared));
        }
        let mut session = self.inner.options.session();
        if let Some(bytes) = overrides.workspace_limit_bytes {
            if bytes == 0 {
                return Err(Error::config("workspace limit must be >= 1"));
            }
            session.workspace_limit = bytes;
        }
        if let Some(bytes) = overrides.memory_budget_bytes {
            session.memory_budget = bytes;
        }
        if let Some(threads) = overrides.threads {
            if threads == 0 {
                return Err(Error::config("threads must be >= 1"));
            }
            session.threads = threads;
        }
        OnednnCpu::load_with(&self.inner.budget, session)
    }
}

/// What a [`Runtime::shared_backend`] handle sizes its one backend with.
/// `None` keeps the [`Options`] value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct BackendOverrides {
    /// See [`Options::workspace_limit_bytes`].
    pub workspace_limit_bytes: Option<usize>,
    /// See [`Options::memory_budget_bytes`]; `Some(0)` is unbounded.
    pub memory_budget_bytes: Option<usize>,
    /// Run this backend's operators on at most this many of the runtime's
    /// pool threads (`None`: all of them; wider than the pool is the pool).
    /// The pool and its one-operator-at-a-time compute lock stay shared
    /// with every other model on the runtime.
    pub threads: Option<usize>,
}

impl BackendOverrides {
    /// Sets [`Self::memory_budget_bytes`].
    #[must_use]
    pub fn with_memory_budget_bytes(mut self, bytes: Option<usize>) -> Self {
        self.memory_budget_bytes = bytes;
        self
    }
    /// Sets [`Self::workspace_limit_bytes`].
    #[must_use]
    pub fn with_workspace_limit_bytes(mut self, bytes: Option<usize>) -> Self {
        self.workspace_limit_bytes = bytes;
        self
    }
    /// Sets [`Self::threads`].
    #[must_use]
    pub fn with_threads(mut self, threads: Option<usize>) -> Self {
        self.threads = threads;
        self
    }
}

#[cfg(test)]
mod tests;

/// `requested`, or the first of `supported` (the model's default on this
/// ISA) when `None`. A precision outside `supported` is
/// [`ErrorKind::Unsupported`]: no model falls back silently.
pub(crate) fn resolve_precision<P: Copy + PartialEq + std::fmt::Display>(
    requested: Option<P>,
    supported: &[P],
    model: &str,
    isa: Isa,
) -> Result<P> {
    let default = *supported
        .first()
        .ok_or_else(|| Error::internal(format!("{model} runs no precision on {isa}")))?;
    let p = requested.unwrap_or(default);
    if supported.contains(&p) {
        return Ok(p);
    }
    let names: Vec<_> = supported.iter().map(ToString::to_string).collect();
    Err(Error::unsupported(
        "precision",
        format!("{model} runs {} on {isa}, not {p}", names.join(", ")),
    ))
}

/// `as_str`, `Display` and `FromStr` (`fp32`, `bf16`) for a model's
/// precision enum with `Fp32` and `Bf16` variants.
macro_rules! precision_names {
    ($ty:ty) => {
        impl $ty {
            /// The lowercase name: `fp32` or `bf16`.
            pub const fn as_str(self) -> &'static str {
                match self {
                    Self::Fp32 => "fp32",
                    Self::Bf16 => "bf16",
                }
            }
        }

        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl std::str::FromStr for $ty {
            type Err = $crate::Error;
            fn from_str(s: &str) -> $crate::Result<Self> {
                match s {
                    "fp32" => Ok(Self::Fp32),
                    "bf16" => Ok(Self::Bf16),
                    _ => Err($crate::Error::config(format!(
                        "unknown precision {s:?} (expected fp32 or bf16)"
                    ))),
                }
            }
        }
    };
}
pub(crate) use precision_names;
