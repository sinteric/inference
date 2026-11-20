//! Model-agnostic process runtime.
//!
//! [`Runtime`] owns every process-global resource (oneDNN ISA cap, pinned
//! thread pool, cpuset, execute-slot default). It knows no model. Each model
//! crate implements [`Model`] and loads from a `&Runtime`, so two models in
//! one process cannot disagree on process globals.
//!
//! ```ignore
//! let rt = Runtime::new(RuntimeOptions::new().with_threads(4))?;
//! let m: SomeModel = rt.load(SomeModelConfig::new("/models/x"))?;
//! ```
//!
//! # Process-global state
//!
//! [`Runtime::new`] owns these. A second [`Runtime::new`] with different ISA /
//! threads is [`Error::Config`].
//!
//! | Resource | Scope | Behaviour |
//! |---|---|---|
//! | `dnnl_set_max_cpu_isa` | process | set by [`Runtime::new`] |
//! | pinned thread pool (GEMM + glue) | process | set by [`Runtime::new`] |
//! | `dnnl_set_verbose` | process | last requested level |
//! | process CPU affinity | process | only if [`RuntimeOptions::pin_process`] |
//! | glibc `M_ARENA_MAX` | process | only if [`RuntimeOptions::malloc_arena_max`] |
//! | [`crate::cpu::onednn::OnednnCpu`] engine, execute slots, activation pool | per model | isolated, unless loaded through [`Runtime::shared_backend`] |
//! | packed weights, embeddings mmap | per model | isolated |
//!
//! oneDNN is **statically linked** (pinned 3.13.4). There is no `library`
//! path and no `dlopen`.
//!
//! # Concurrency
//!
//! Model calls are **blocking**. In async code use
//! `tokio::task::spawn_blocking`. Concurrent calls on one model take a slot
//! from a bounded execute pool ([`RuntimeOptions::exec_slots`], default 1).
//! GEMMs and glue share one process-global thread pool and run one
//! `parallel_for` at a time (safe for AMX; models in one process share that
//! pool).

use crate::{Error, Result};

mod options;
mod precision;

pub use options::{CacheConfig, CpuIsa, RuntimeOptions};
pub use precision::Precision;

use std::sync::Arc;

use crate::cpu::onednn::{OnednnCpu, TuningPlan};
use crate::cpu::threads::ThreadBudget;

/// A model that loads from a [`Runtime`].
///
/// Only loading and unloading are uniform. Inference APIs are model-specific
/// on purpose (`embed`, `forward`, `detect`, ...): there is no shared
/// tensor-in / tensor-out call.
pub trait Model: Sized + Send + Sync {
    type Config;
    /// Precisions this model implements; the first is its default (what a
    /// config with `precision: None` loads). See [`Precision::resolve`].
    const PRECISIONS: &'static [Precision];

    fn load(rt: &Runtime, config: Self::Config) -> Result<Self>;

    /// The precision this loaded model computes in.
    fn precision(&self) -> Precision;

    /// Release this model: packed weights, weight mmaps, oneDNN engine,
    /// primitives, streams, scratchpads and activation pool are freed, then
    /// freed heap pages are handed back to the OS ([`release_free_memory`]).
    /// Process-global state (ISA cap, thread pool) belongs to the
    /// [`Runtime`] and stays.
    ///
    /// Taking `self` means no call can be in flight. For a shared model, the
    /// last owner unloads: `if let Some(m) = Arc::into_inner(model) { m.unload() }`.
    /// A plain `drop` frees the same memory but may leave it in the
    /// allocator's free lists instead of returning it to the OS.
    fn unload(self) {
        drop(self);
        release_free_memory();
    }
}

/// Return free heap pages to the OS (glibc `malloc_trim(0)`). Large buffers
/// are already unmapped on free; this releases the rest of the freed arena
/// memory so RSS drops after an unload. Returns whether memory was released.
/// No-op (`false`) on non-glibc targets. Cost is a walk of the allocator's
/// arenas: call it after unloading, not per request.
pub fn release_free_memory() -> bool {
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

/// Process-global handle: ISA, threads, cpuset/pin, execute-slot default.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
    /// Set on a [`Runtime::shared_backend`] handle: every model loaded
    /// through it runs on this backend.
    shared: Option<Arc<OnednnCpu>>,
}

struct RuntimeInner {
    budget: ThreadBudget,
    options: RuntimeOptions,
    plan: TuningPlan,
}

impl Runtime {
    /// Install process-global oneDNN ISA / pinned thread pool, then return a
    /// handle used to load models.
    pub fn new(options: RuntimeOptions) -> Result<Self> {
        let budget = options.thread_budget()?;
        if let Some(n) = options.malloc_arena_max {
            set_malloc_arena_max(n)?;
        }
        if options.pin_process {
            budget.apply_process_policy()?;
        } else {
            budget.install_library_workers()?;
        }
        // Creating a backend configures ISA and the shared threadpool first.
        let warmup = OnednnCpu::load_with(&budget, options.session())?;
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
            }),
            shared: None,
        })
    }

    /// Load any [`Model`]: `rt.load::<M>(config)` is `M::load(&rt, config)`,
    /// then [`release_free_memory`]. Loading frees large temporaries (raw
    /// weights before packing, conversion buffers) that the allocator would
    /// otherwise keep resident.
    pub fn load<M: Model>(&self, config: M::Config) -> Result<M> {
        let model = M::load(self, config);
        release_free_memory();
        model
    }

    pub fn options(&self) -> &RuntimeOptions {
        &self.inner.options
    }

    /// Per-ISA tiles / grains for model loaders.
    pub fn plan(&self) -> TuningPlan {
        self.inner.plan
    }

    /// Effective oneDNN ISA (`dnnl_get_effective_cpu_isa`) every model on
    /// this runtime dispatches to, after [`RuntimeOptions::cpu_isa`].
    pub fn cpu_isa(&self) -> u64 {
        self.inner.plan.cpu_isa
    }

    /// [`Runtime::cpu_isa`] as oneDNN's name (`AVX512_CORE_BF16`,
    /// `AVX10_1_512_AMX`, ...), or `"UNKNOWN"`.
    pub fn cpu_isa_name(&self) -> &'static str {
        cpu_isa_name(self.cpu_isa())
    }

    /// Effective activation-retention cap (`0` in options means the default).
    pub fn workspace_limit(&self) -> usize {
        self.inner.options.session().workspace_limit
    }

    /// A handle on this runtime whose models all load onto one backend:
    /// one engine, execute-slot pool, activation pool, convolution arena,
    /// primitive cache and memory budget, sized by `overrides`. Models that
    /// run one after another (an OCR detector and recognizer) then keep one
    /// working set instead of one each.
    ///
    /// ```ignore
    /// let ocr = rt.shared_backend(
    ///     BackendOverrides::default()
    ///         .with_workspace_limit_bytes(Some(1 << 30))
    ///         .with_memory_budget_bytes(Some(512 << 20)),
    /// )?;
    /// let det: Detector = ocr.load(DetConfig::new("/models/ppocr/det"))?;
    /// let rec: Recognizer = ocr.load(RecConfig::new("/models/ppocr/rec"))?;
    /// ```
    ///
    /// Workspace and budget belong to the handle: a model config loaded
    /// through it that sets either is [`Error::Config`]. The backend is
    /// freed once the handle and every model loaded through it are dropped.
    pub fn shared_backend(&self, overrides: BackendOverrides) -> Result<Runtime> {
        if self.shared.is_some() {
            return Err(Error::Config(
                "this runtime handle already shares one backend".into(),
            ));
        }
        Ok(Self {
            inner: Arc::clone(&self.inner),
            shared: Some(self.backend_with(overrides)?),
        })
    }

    /// Whether this handle came from [`Runtime::shared_backend`].
    pub fn shares_backend(&self) -> bool {
        self.shared.is_some()
    }

    /// A fresh per-model backend (engine, execute slots, activation pool) on
    /// the runtime's ISA and thread pool, or the shared one on a
    /// [`Runtime::shared_backend`] handle. For model loaders.
    pub fn backend(&self) -> Result<Arc<OnednnCpu>> {
        self.backend_with(BackendOverrides::default())
    }

    /// [`Runtime::backend`] with a model-specific workspace cap, for models
    /// whose working set differs from the text models' (OCR convolution
    /// arenas).
    pub fn backend_with_workspace_limit(&self, bytes: usize) -> Result<Arc<OnednnCpu>> {
        self.backend_with(BackendOverrides {
            workspace_limit_bytes: Some(bytes),
            ..BackendOverrides::default()
        })
    }

    /// [`Runtime::backend`] with per-model overrides of the runtime defaults.
    /// On a [`Runtime::shared_backend`] handle any override is
    /// [`Error::Config`]: the handle's settings apply.
    pub fn backend_with(&self, overrides: BackendOverrides) -> Result<Arc<OnednnCpu>> {
        if let Some(shared) = &self.shared {
            if overrides != BackendOverrides::default() {
                return Err(Error::Config(
                    "set workspace_limit_bytes, memory_budget_bytes and threads on \
                     Runtime::shared_backend, not on a model loaded through it"
                        .into(),
                ));
            }
            return Ok(Arc::clone(shared));
        }
        let mut session = self.inner.options.session();
        if let Some(bytes) = overrides.workspace_limit_bytes {
            if bytes == 0 {
                return Err(Error::Config("workspace limit must be >= 1".into()));
            }
            session.workspace_limit = bytes;
        }
        if let Some(bytes) = overrides.memory_budget_bytes {
            session.memory_budget = bytes;
        }
        if let Some(threads) = overrides.threads {
            if threads == 0 {
                return Err(Error::Config("threads must be >= 1".into()));
            }
            session.threads = threads;
        }
        OnednnCpu::load_with(&self.inner.budget, session)
    }
}

/// oneDNN `dnnl_cpu_isa_t` value to its name.
pub fn cpu_isa_name(isa: u64) -> &'static str {
    match isa {
        0x1 => "SSE41",
        0x3 => "AVX",
        0x7 => "AVX2",
        0xf => "AVX2_VNNI",
        0x1f => "AVX2_VNNI_2",
        0x27 => "AVX512_CORE",
        0x67 => "AVX512_CORE_VNNI",
        0xe7 => "AVX512_CORE_BF16",
        0x1ef => "AVX10_1_512",
        0xfef => "AVX10_1_512_AMX",
        0x1fef => "AVX10_1_512_AMX_FP16",
        0x201ff => "AVX10_2_512",
        0x22fff => "AVX10_2_512_AMX_2",
        _ => "UNKNOWN",
    }
}

/// Per-model overrides for [`Runtime::backend_with`]. `None` keeps the
/// [`RuntimeOptions`] value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct BackendOverrides {
    pub workspace_limit_bytes: Option<usize>,
    /// See [`RuntimeOptions::memory_budget_bytes`]; `Some(0)` is unbounded.
    pub memory_budget_bytes: Option<usize>,
    /// Run this backend's operators on at most this many of the runtime's
    /// pool threads (`None`: all of them; wider than the pool is the pool).
    /// The pool and its one-operator-at-a-time compute lock stay shared
    /// with every other model on the runtime; see
    /// [`SessionOptions::threads`](crate::cpu::onednn::SessionOptions::threads).
    pub threads: Option<usize>,
}

impl BackendOverrides {
    pub fn with_memory_budget_bytes(mut self, bytes: Option<usize>) -> Self {
        self.memory_budget_bytes = bytes;
        self
    }
    pub fn with_workspace_limit_bytes(mut self, bytes: Option<usize>) -> Self {
        self.workspace_limit_bytes = bytes;
        self
    }
    pub fn with_threads(mut self, threads: Option<usize>) -> Self {
        self.threads = threads;
        self
    }
}

/// `mallopt(M_ARENA_MAX, n)`: bound the number of glibc malloc arenas so
/// freed heap is not kept per calling thread. Arenas that already exist stay.
fn set_malloc_arena_max(n: usize) -> Result<()> {
    if n == 0 || n > i32::MAX as usize {
        return Err(Error::Config(format!("malloc_arena_max {n} outside 1..")));
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: mallopt only changes allocator tuning parameters.
        if unsafe { libc::mallopt(libc::M_ARENA_MAX, n as i32) } != 1 {
            return Err(Error::Config("mallopt(M_ARENA_MAX) failed".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
