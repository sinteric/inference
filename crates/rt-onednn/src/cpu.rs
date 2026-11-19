//! oneDNN-CPU [`rt_backend::Hardware`]: same numerics as the retired C++ bridge.
//!
//! Strict fpmath/accumulation, user scratchpads, primitive cache capacity 0,
//! packed weight reorder, residual sum post-op, kind-7 GeGLU
//! (`MatMul + GELU_ERF + BINARY_MUL`).

#![allow(clippy::too_many_arguments)]

use crate::api::{memory_none, Api};
use crate::cache::{empty_layout_tag, LayoutTag, LAYOUT_TAG_LEN};
use crate::options::{CpuIsaCap, SessionOptions, DEFAULT_WORKSPACE_LIMIT};
use onednn_sys::*;
use rt_backend::{Hardware, Packed, PACKED_LAYOUT_TAG_LEN};
use rt_core::{check_shape, Result, Result as AResult, ResultContext as Context};
use rt_kernels::{
    f32_slice_to_bf16_nthreads, for_ranges, rmsnorm_nthreads, row_mut, row_ref, rows_mut, rows_ref,
    softmax_rows_nthreads, softmax_window,
};
use rt_threads::{compute_lock, ThreadBudget};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// Matmul operand types. The destination is always f32.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum MatDt {
    F32,
    /// BF16 activations and weights; f32 activations are converted first.
    Bf16,
    /// f32 activations against BF16-stored weights that the kernel widens
    /// exactly to f32: FP32 math at BF16 weight memory.
    F32Bf16,
}

impl MatDt {
    fn src(self) -> dnnl_data_type_t {
        if self == Self::Bf16 {
            dnnl_bf16
        } else {
            dnnl_f32
        }
    }
    fn weights(self) -> dnnl_data_type_t {
        if self == Self::F32 {
            dnnl_f32
        } else {
            dnnl_bf16
        }
    }
}

pub struct Aligned {
    ptr: NonNull<u8>,
    layout: std::alloc::Layout,
    pub bytes: usize,
}

impl Aligned {
    pub fn new(bytes: usize) -> AResult<Self> {
        let n = bytes.max(64);
        let layout = std::alloc::Layout::from_size_align(n, 64)?;
        let ptr = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .context("aligned allocation")?;
        Ok(Self {
            ptr,
            layout,
            bytes: n,
        })
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.bytes) }
    }
    /// Copy `src` into a 64-byte-aligned buffer without zero-filling first.
    pub fn copy_from(src: &[u8]) -> AResult<Self> {
        let n = src.len().max(64);
        let layout = std::alloc::Layout::from_size_align(n, 64)?;
        let ptr =
            NonNull::new(unsafe { std::alloc::alloc(layout) }).context("aligned allocation")?;
        unsafe {
            ptr.as_ptr()
                .copy_from_nonoverlapping(src.as_ptr(), src.len());
            if n > src.len() {
                ptr.as_ptr().add(src.len()).write_bytes(0, n - src.len());
            }
        }
        Ok(Self {
            ptr,
            layout,
            bytes: n,
        })
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

unsafe impl Send for Aligned {}
unsafe impl Sync for Aligned {}

pub(crate) struct WeightLayout {
    api: Arc<Api>,
    packed_md: dnnl_memory_desc_t,
    raw_md: dnnl_memory_desc_t,
    scratch_md: dnnl_memory_desc_t,
    reorder: dnnl_primitive_t,
    bytes: usize,
    scratch: usize,
}

impl Drop for WeightLayout {
    fn drop(&mut self) {
        unsafe {
            if !self.reorder.is_null() {
                (self.api.prim_destroy)(self.reorder);
            }
            if !self.packed_md.is_null() {
                (self.api.md_destroy)(self.packed_md);
            }
            if !self.raw_md.is_null() {
                (self.api.md_destroy)(self.raw_md);
            }
            if !self.scratch_md.is_null() {
                (self.api.md_destroy)(self.scratch_md);
            }
        }
    }
}

unsafe impl Send for WeightLayout {}
unsafe impl Sync for WeightLayout {}

pub struct PackedWeight {
    layout: Arc<WeightLayout>,
    data: Aligned,
    n: usize,
    k: usize,
    dt: MatDt,
    tag: LayoutTag,
}

impl Packed for PackedWeight {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn bytes(&self) -> usize {
        self.layout.bytes
    }
    fn layout_tag(&self) -> [u8; PACKED_LAYOUT_TAG_LEN] {
        const {
            assert!(PACKED_LAYOUT_TAG_LEN == LAYOUT_TAG_LEN);
        }
        self.tag
    }
}

impl PackedWeight {
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }
}

struct MemSlot {
    arg: i32,
    mem: dnnl_memory_t,
}

struct Plan {
    api: Arc<Api>,
    prim: dnnl_primitive_t,
    slots: Vec<MemSlot>,
    slot_specs: Vec<(i32, dnnl_memory_desc_t)>,
    mds: Vec<dnnl_memory_desc_t>,
    scratch: usize,
    name: String,
    /// `State::tick` at the last execute; the smallest is evicted first.
    used: AtomicU64,
    /// Heap growth while the plan was built (oneDNN primitive state), an
    /// estimate charged against the memory budget.
    bytes: usize,
}

impl Drop for Plan {
    fn drop(&mut self) {
        unsafe {
            if !self.prim.is_null() {
                (self.api.prim_destroy)(self.prim);
            }
            for s in &self.slots {
                if !s.mem.is_null() {
                    (self.api.mem_destroy)(s.mem);
                }
            }
            for md in &self.mds {
                if !md.is_null() {
                    (self.api.md_destroy)(*md);
                }
            }
        }
    }
}

unsafe impl Send for Plan {}
// Shared only to keep `prim` alive through an execute that runs outside the
// state lock; plan-owned memories are bound under that lock.
unsafe impl Sync for Plan {}
unsafe impl Send for ExecWorkspace {}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Key {
    kind: i32,
    g: usize,
    m: usize,
    n: usize,
    k: usize,
    ab: usize,
    ar: usize,
    bb: usize,
    br: usize,
    trans: bool,
    param: u32,
    sum: bool,
    weight: usize,
    dt: MatDt,
}

struct State {
    plans: HashMap<Key, Arc<Plan>>,
    /// Plan cache bound; full caches evict the least recently executed plan.
    capacity: usize,
    tick: u64,
    creations: u64,
    layouts: HashMap<(usize, usize, MatDt), Arc<WeightLayout>>,
    executions: BTreeMap<i32, u64>,
    setup_s: f64,
    exec_s: BTreeMap<i32, f64>,
    convert_s: f64,
    weight_bytes: usize,
    category_s: BTreeMap<String, f64>,
    evictions: u64,
    evicted_bytes: u64,
    /// Plans dropped because the cache was full (LRU), as opposed to
    /// `evictions`, which counts memory-budget evictions.
    capacity_evictions: u64,
}

/// What a backend keeps between calls, mirrored in atomics so the pool and
/// convolution paths can check the budget without taking the plan lock.
#[derive(Default)]
pub(crate) struct Retained {
    plans: AtomicUsize,
    pool: AtomicUsize,
    pub(crate) conv_workspace: AtomicUsize,
    exec_scratch: AtomicUsize,
    /// Convolution plans (OCR). Each is charged the heap it added when
    /// built; primitives shared with later plans stay charged to the first.
    pub(crate) conv_plans: AtomicUsize,
    pub(crate) conv_evictions: AtomicUsize,
    pub(crate) conv_evicted: AtomicUsize,
    /// Bytes released since the last `malloc_trim`.
    released: AtomicUsize,
    trims: AtomicUsize,
}

impl Retained {
    fn total(&self) -> usize {
        self.without_conv_workspace() + self.conv_workspace.load(Ordering::Relaxed)
    }
    fn without_conv_workspace(&self) -> usize {
        self.plans.load(Ordering::Relaxed)
            + self.pool.load(Ordering::Relaxed)
            + self.exec_scratch.load(Ordering::Relaxed)
            + self.conv_plans.load(Ordering::Relaxed)
    }
}

/// Return freed pages to the OS once this much has been released.
const TRIM_AFTER_BYTES: usize = 64 << 20;

/// Live heap bytes (glibc `mallinfo2`); `None` where it is not available.
pub(crate) fn heap_in_use() -> Option<usize> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: mallinfo2 only reads allocator statistics.
        let m = unsafe { libc::mallinfo2() };
        Some(m.uordblks + m.hblkhd)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        None
    }
}

/// Activation buffer pool, separate from execute workspaces so `buffer`/`recycle`
/// do not wait on a GEMM.
// Poison recovery for pool/statistics bookkeeping is safe: every buffer is owned,
// and no guard exposes partial numerical state to callers.
struct Pool {
    buffers: HashMap<usize, Vec<Vec<f32>>>,
    buffers_u16: HashMap<usize, Vec<Vec<u16>>>,
    buffer_bytes: usize,
    active_shape: Option<(usize, usize)>,
}

/// Per-execute oneDNN stream + scratch. Concurrent `embed` calls take a slot
/// from [`ExecPool`] instead of sharing one stream under a hidden lock.
struct ExecWorkspace {
    stream: dnnl_stream_t,
    scratch: Aligned,
    bf16_a: Aligned,
    bf16_b: Aligned,
}

impl ExecWorkspace {
    fn new(api: &Api, engine: dnnl_engine_t, threadpool: *mut c_void) -> AResult<Self> {
        let mut stream = ptr::null_mut();
        Api::check(
            unsafe { (api.stream_create)(&mut stream, engine, threadpool) },
            "stream",
        )?;
        Ok(Self {
            stream,
            scratch: Aligned::new(64)?,
            bf16_a: Aligned::new(64)?,
            bf16_b: Aligned::new(64)?,
        })
    }
}

struct ExecPool {
    idle: Mutex<Vec<ExecWorkspace>>,
    cv: Condvar,
}

struct ExecLease<'a> {
    pool: &'a ExecPool,
    ws: Option<ExecWorkspace>,
}

impl ExecPool {
    /// An idle workspace. Inside a pool job (independent batches running one
    /// per worker) a missing workspace is created instead of waited for: the
    /// job's other chunks may hold every slot until the job ends.
    fn acquire(&self, make: impl FnOnce() -> AResult<ExecWorkspace>) -> AResult<ExecLease<'_>> {
        let mut idle = self.idle.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(ws) = idle.pop() {
                return Ok(ExecLease {
                    pool: self,
                    ws: Some(ws),
                });
            }
            if rt_threads::in_parallel() {
                drop(idle);
                return Ok(ExecLease {
                    pool: self,
                    ws: Some(make()?),
                });
            }
            idle = self.cv.wait(idle).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl<'a> Deref for ExecLease<'a> {
    type Target = ExecWorkspace;
    fn deref(&self) -> &ExecWorkspace {
        // reason: acquire constructs Some; only Drop takes it, after the last dereference.
        self.ws.as_ref().expect("exec lease is present until Drop")
    }
}

impl<'a> DerefMut for ExecLease<'a> {
    fn deref_mut(&mut self) -> &mut ExecWorkspace {
        // reason: acquire constructs Some; only Drop takes it, after the last dereference.
        self.ws.as_mut().expect("exec lease is present until Drop")
    }
}

impl Drop for ExecLease<'_> {
    fn drop(&mut self) {
        if let Some(ws) = self.ws.take() {
            self.pool
                .idle
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(ws);
            self.pool.cv.notify_one();
        }
    }
}

/// oneDNN CPU backend. Packed weights and the primitive cache are shared.
/// Execute uses a bounded pool of streams/scratchpads (`SessionOptions.exec_slots`,
/// default 1 = serialized, same as a single stream). Concurrent slots share
/// the process-global `rt-threads` pool (one `parallel_for` at a time).
pub struct OnednnCpu {
    api: Arc<Api>,
    engine: dnnl_engine_t,
    exec: ExecPool,
    exec_slots: usize,
    state: Mutex<State>,
    pool: Mutex<Pool>,
    pub(crate) conv_workspace: crate::conv::Workspaces,
    pub(crate) conv_weight_bytes: AtomicUsize,
    threads: i32,
    /// This backend's `threadpool_iface` when [`SessionOptions::threads`]
    /// caps it below the pool; `None` borrows the process-wide one. Dropped
    /// after `Drop` has destroyed every stream.
    capped_threadpool: Option<CappedThreadpool>,
    workspace_limit: usize,
    /// Bytes this backend may keep between calls (`0` = unbounded).
    memory_budget: usize,
    pub(crate) retained: Retained,
    pub cpu_isa: u64,
    strategy: crate::IsaStrategy,
    pub info: String,
}

unsafe impl Send for OnednnCpu {}
unsafe impl Sync for OnednnCpu {}

/// An owned `threadpool_iface` reporting at most `n` threads to oneDNN.
struct CappedThreadpool(*mut c_void);

impl CappedThreadpool {
    fn new(n: i32) -> AResult<Self> {
        let p = unsafe { onednn_sys::sinteric_dnnl_threadpool_create_capped(n) };
        if p.is_null() {
            return Err(rt_core::Error::Allocation { bytes: 16 });
        }
        Ok(Self(p))
    }
}

impl Drop for CappedThreadpool {
    fn drop(&mut self) {
        unsafe { onednn_sys::sinteric_dnnl_threadpool_destroy(self.0) };
    }
}

impl Drop for OnednnCpu {
    fn drop(&mut self) {
        // Primitives and memories must die before the engine/stream.
        if let Ok(mut workspaces) = self.conv_workspace.state.lock() {
            workspaces.idle.clear();
        }
        if let Ok(mut s) = self.state.lock() {
            s.plans.clear();
            s.layouts.clear();
        }
        if let Ok(mut idle) = self.exec.idle.lock() {
            for ws in idle.drain(..) {
                unsafe {
                    if !ws.stream.is_null() {
                        (self.api.stream_destroy)(ws.stream);
                    }
                }
            }
        }
        unsafe {
            if !self.engine.is_null() {
                (self.api.engine_destroy)(self.engine);
            }
        }
    }
}

fn touch_pages<T>(buf: &mut [T]) {
    if buf.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(buf);
    let ptr = buf.as_mut_ptr().cast::<u8>();
    let mut off = 0;
    while off < bytes {
        unsafe {
            ptr.add(off).write_volatile(ptr.add(off).read());
        }
        off += 4096;
    }
    unsafe {
        ptr.add(bytes - 1).write_volatile(ptr.add(bytes - 1).read());
    }
}

fn fill_bf16(buf: &mut Aligned, src: &[f32], nthreads: i32) -> AResult<*mut c_void> {
    let need = rt_core::checked_size("BF16 conversion", &[src.len(), 2])?.max(64);
    if buf.bytes < need {
        *buf = Aligned::new(need)?;
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(buf.ptr.as_ptr().cast::<u16>(), src.len()) };
    f32_slice_to_bf16_nthreads(src, dst, nthreads.max(1) as usize)?;
    Ok(buf.ptr())
}

fn fill(dims: &[i64]) -> [i64; 12] {
    let mut out = [0i64; 12];
    out[..dims.len()].copy_from_slice(dims);
    out
}

/// `BestAvailable` becomes `Avx512CoreBf16` on a host whose AMX tile state is
/// lost across preemption ([`crate::amx_health`]); explicit caps are kept.
fn resolve_isa_cap(cap: CpuIsaCap) -> CpuIsaCap {
    if cap != CpuIsaCap::BestAvailable || !crate::amx_probe_enabled() {
        return cap;
    }
    match crate::amx_health() {
        crate::AmxHealth::TileStateLoss {
            corrupt,
            iterations,
            preemptions,
        } => {
            static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| {
                eprintln!(
                    "warning: AMX tile state is lost when a thread is preempted on this host \
                     ({corrupt} wrong results in {iterations} tile GEMMs, {preemptions} \
                     preemptions); using AVX512_CORE_BF16 instead of AMX"
                );
            });
            CpuIsaCap::Avx512CoreBf16
        }
        _ => cap,
    }
}

impl OnednnCpu {
    /// Set the plan cache bound (default 256) and drop the cached plans. A
    /// full cache evicts its least recently executed plan.
    #[doc(hidden)]
    pub fn set_primitive_capacity(&self, capacity: usize) -> Result<()> {
        check_shape!(capacity > 0, "zero primitive cache capacity");
        let _compute = compute_lock();
        let mut s = self
            .state
            .lock()
            .map_err(|_| rt_core::Error::Poisoned("backend"))?;
        s.plans.clear();
        s.capacity = capacity;
        self.retained.plans.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// The plan cache bound (see [`OnednnCpu::set_primitive_capacity`]).
    #[doc(hidden)]
    pub fn primitive_capacity(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .capacity
    }

    pub(crate) fn conv_api(&self) -> &Api {
        &self.api
    }
    /// The `threadpool_iface` this backend's streams run on.
    pub(crate) fn threadpool(&self) -> *mut c_void {
        self.capped_threadpool
            .as_ref()
            .map_or(self.api.threadpool, |tp| tp.0)
    }
    pub(crate) fn conv_engine(&self) -> dnnl_engine_t {
        self.engine
    }
    /// Channels per block of the convolution layer's blocked layouts: 16 where
    /// oneDNN dispatches AVX-512 (its `nChw16c` kernels), 8 below that (the
    /// AVX2 and SSE4.1 kernels take `nChw8c`). Measured 2026-11-01 on oneDNN
    /// 3.13.4: capped to AVX2, a 32-channel 5x5 convolution over `nChw16c`
    /// fell to `ref:any` at 19 MMAC/s against `jit:avx512_core`'s 74 GMAC/s.
    pub fn conv_block(&self) -> usize {
        let avx512 = u64::from(dnnl_cpu_isa_avx512_core as u32);
        if self.cpu_isa & avx512 == avx512 {
            16
        } else {
            8
        }
    }
    pub(crate) fn conv_limit(&self) -> usize {
        self.workspace_limit
    }

    /// Whether the backend keeps more than its memory budget.
    pub(crate) fn over_budget(&self) -> bool {
        self.exceeds_budget(0)
    }

    /// Whether keeping `extra` more bytes would exceed the memory budget.
    pub(crate) fn exceeds_budget(&self, extra: usize) -> bool {
        self.memory_budget > 0 && self.retained.total() + extra > self.memory_budget
    }

    /// Whether the backend would be over its memory budget with the
    /// convolution arena and scratch at `workspace` bytes.
    pub(crate) fn over_budget_with_workspace(&self, workspace: usize) -> bool {
        self.memory_budget > 0
            && self.retained.without_conv_workspace() + workspace > self.memory_budget
    }

    /// Count released bytes and return freed pages to the OS once enough
    /// have accumulated (glibc keeps them resident otherwise).
    pub(crate) fn note_released(&self, bytes: usize) {
        let released = self.retained.released.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if released >= TRIM_AFTER_BYTES {
            self.retained.released.store(0, Ordering::Relaxed);
            self.retained.trims.fetch_add(1, Ordering::Relaxed);
            // SAFETY: malloc_trim only touches allocator-internal free lists.
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            unsafe {
                libc::malloc_trim(0);
            }
        }
    }

    /// Keep within the memory budget after `keep` was added: drop pooled
    /// activation buffers first (cheap to reallocate), then the least
    /// recently executed plans. Never fails; an evicted plan is rebuilt the
    /// next time its shape runs.
    fn enforce_budget(&self, s: &mut State, keep: &Key) {
        if !self.over_budget() {
            return;
        }
        let freed = {
            let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.buffers.clear();
            pool.buffers_u16.clear();
            std::mem::take(&mut pool.buffer_bytes)
        };
        self.retained.pool.store(0, Ordering::Relaxed);
        let mut released = freed;
        while self.over_budget() && s.plans.len() > 1 {
            let old = s
                .plans
                .iter()
                .filter(|(key, _)| *key != keep)
                .min_by_key(|(_, plan)| plan.used.load(Ordering::Relaxed))
                .map(|(key, _)| key.clone());
            let Some(plan) = old.and_then(|old| s.plans.remove(&old)) else {
                break;
            };
            self.retained.plans.fetch_sub(plan.bytes, Ordering::Relaxed);
            s.evictions += 1;
            s.evicted_bytes += plan.bytes as u64;
            released += plan.bytes;
        }
        self.note_released(released);
    }

    /// Checked size of a request. `workspace_limit` caps only the *retained*
    /// activation pool (as before this crate returned typed errors); it is not a
    /// per-request bound, so shapes allowed by the model's `max_seq_len` /
    /// `max_batch` keep working. Oversized requests fail with `SizeOverflow` or
    /// `Allocation` instead of aborting.
    fn check_workspace(&self, operation: &'static str, dims: &[usize]) -> Result<usize> {
        rt_core::checked_size(operation, dims)
    }

    pub fn load(budget: &ThreadBudget) -> AResult<Arc<Self>> {
        Self::load_with(budget, SessionOptions::default())
    }

    pub fn load_with(budget: &ThreadBudget, opts: SessionOptions) -> AResult<Arc<Self>> {
        if budget.nthreads == 0
            || budget.nthreads > libc::CPU_SETSIZE as usize
            || opts.exec_slots == 0
            || !(0..=2).contains(&opts.verbose)
        {
            return Err(rt_core::Error::Config(format!(
                "invalid session: threads {}, exec_slots {}, verbose {}",
                budget.nthreads, opts.exec_slots, opts.verbose
            )));
        }
        let limit = if opts.workspace_limit == 0 {
            DEFAULT_WORKSPACE_LIMIT
        } else {
            opts.workspace_limit
        };
        let slot_bytes = rt_core::checked_size("execute slots", &[opts.exec_slots, 192])?;
        if slot_bytes > limit {
            return Err(rt_core::Error::WorkspaceLimit {
                operation: "execute slots",
                requested: slot_bytes,
                limit,
            });
        }
        budget.install_library_workers()?;
        let api = Arc::new(Api::init(
            resolve_isa_cap(opts.max_cpu_isa),
            opts.verbose,
            budget.nthreads_i32(),
        )?);
        let mut engine = ptr::null_mut();
        Api::check(
            unsafe { (api.engine_create)(&mut engine, dnnl_cpu, 0) },
            "engine",
        )?;
        let threads = if opts.threads == 0 {
            budget.nthreads
        } else {
            opts.threads.min(budget.nthreads)
        };
        let threads_i32 = i32::try_from(threads)
            .map_err(|_| rt_core::Error::Config(format!("invalid threads {threads}")))?;
        let capped_threadpool = if threads < budget.nthreads {
            Some(CappedThreadpool::new(threads_i32)?)
        } else {
            None
        };
        let threadpool = capped_threadpool.as_ref().map_or(api.threadpool, |tp| tp.0);
        let exec_slots = opts.exec_slots.max(1);
        let mut idle = Vec::new();
        idle.try_reserve_exact(exec_slots)
            .map_err(|_| rt_core::Error::Allocation { bytes: slot_bytes })?;
        for _ in 0..exec_slots {
            idle.push(ExecWorkspace::new(&api, engine, threadpool)?);
        }
        let cpu_isa = unsafe { (api.get_effective_cpu_isa)() as u32 as u64 };
        let info = format!("{}; isa=0x{cpu_isa:x}", api.info);
        let workspace_limit = if opts.workspace_limit == 0 {
            DEFAULT_WORKSPACE_LIMIT
        } else {
            opts.workspace_limit
        };
        Ok(Arc::new(Self {
            api,
            engine,
            exec: ExecPool {
                idle: Mutex::new(idle),
                cv: Condvar::new(),
            },
            exec_slots,
            conv_workspace: Default::default(),
            conv_weight_bytes: AtomicUsize::new(0),
            cpu_isa,
            strategy: crate::IsaStrategy::from_cpu_isa(cpu_isa),
            state: Mutex::new(State {
                plans: HashMap::new(),
                capacity: 256,
                tick: 0,
                creations: 0,
                layouts: HashMap::new(),
                executions: BTreeMap::new(),
                setup_s: 0.,
                exec_s: BTreeMap::new(),
                convert_s: 0.,
                weight_bytes: 0,
                category_s: BTreeMap::new(),
                evictions: 0,
                evicted_bytes: 0,
                capacity_evictions: 0,
            }),
            pool: Mutex::new(Pool {
                buffers: HashMap::new(),
                buffers_u16: HashMap::new(),
                buffer_bytes: 0,
                active_shape: None,
            }),
            threads: threads_i32,
            capped_threadpool,
            workspace_limit,
            memory_budget: opts.memory_budget,
            retained: Retained::default(),
            info,
        }))
    }

    fn strict_attr(
        &self,
        sum: bool,
        gate: Option<dnnl_memory_desc_t>,
    ) -> AResult<(dnnl_primitive_attr_t, Option<dnnl_post_ops_t>)> {
        let mut attr = ptr::null_mut();
        Api::check(unsafe { (self.api.attr_create)(&mut attr) }, "attr")?;
        Api::check(
            unsafe { (self.api.attr_fpmath)(attr, dnnl_fpmath_mode_strict) },
            "fpmath",
        )?;
        Api::check(
            unsafe { (self.api.attr_accum)(attr, dnnl_accumulation_mode_strict) },
            "accumulation",
        )?;
        Api::check(
            unsafe { (self.api.attr_scratch)(attr, dnnl_scratchpad_mode_user) },
            "scratchpad",
        )?;
        let mut post = ptr::null_mut();
        if sum || gate.is_some() {
            Api::check(unsafe { (self.api.post_ops_create)(&mut post) }, "post_ops")?;
            if sum {
                Api::check(
                    unsafe { (self.api.post_ops_sum)(post, 1.0, 0, dnnl_data_type_undef) },
                    "sum post-op",
                )?;
            }
            if let Some(md) = gate {
                Api::check(
                    unsafe { (self.api.post_ops_eltwise)(post, dnnl_eltwise_gelu_erf, 0.0, 0.0) },
                    "gelu post-op",
                )?;
                Api::check(
                    unsafe { (self.api.post_ops_binary)(post, dnnl_binary_mul, md) },
                    "binary-mul post-op",
                )?;
            }
            Api::check(
                unsafe { (self.api.attr_post_ops)(attr, post) },
                "set post_ops",
            )?;
        }
        Ok((attr, if post.is_null() { None } else { Some(post) }))
    }

    fn md_strides(&self, dims: &[i64], strides: &[i64]) -> AResult<dnnl_memory_desc_t> {
        self.md_strides_dt(dims, strides, dnnl_f32)
    }

    fn md_strides_dt(
        &self,
        dims: &[i64],
        strides: &[i64],
        dt: dnnl_data_type_t,
    ) -> AResult<dnnl_memory_desc_t> {
        let d = fill(dims);
        let s = fill(strides);
        let mut md = ptr::null_mut();
        Api::check(
            unsafe {
                (self.api.md_strides)(&mut md, dims.len() as i32, d.as_ptr(), dt, s.as_ptr())
            },
            "memory desc strides",
        )?;
        Ok(md)
    }

    fn md_any_dt(&self, dims: &[i64], dt: dnnl_data_type_t) -> AResult<dnnl_memory_desc_t> {
        let d = fill(dims);
        let mut md = ptr::null_mut();
        Api::check(
            unsafe {
                (self.api.md_tag)(
                    &mut md,
                    dims.len() as i32,
                    d.as_ptr(),
                    dt,
                    dnnl_format_tag_any,
                )
            },
            "memory desc any",
        )?;
        Ok(md)
    }

    fn bf16_attr(
        &self,
        sum: bool,
        gate: Option<(dnnl_memory_desc_t, dnnl_alg_kind_t)>,
    ) -> AResult<(dnnl_primitive_attr_t, Option<dnnl_post_ops_t>)> {
        let mut attr = ptr::null_mut();
        Api::check(unsafe { (self.api.attr_create)(&mut attr) }, "attr")?;
        Api::check(
            unsafe { (self.api.attr_fpmath)(attr, dnnl_fpmath_mode_strict) },
            "fpmath",
        )?;
        // Strict accumulation is f32 for non-quantized (incl. BF16) matmuls.
        // Explicit `accumulation_mode_f32` is a non-default attr that brgemm
        // / gemm_bf16 reject as unimplemented.
        Api::check(
            unsafe { (self.api.attr_accum)(attr, dnnl_accumulation_mode_strict) },
            "accumulation",
        )?;
        Api::check(
            unsafe { (self.api.attr_scratch)(attr, dnnl_scratchpad_mode_user) },
            "scratchpad",
        )?;
        let mut post = ptr::null_mut();
        if sum || gate.is_some() {
            Api::check(unsafe { (self.api.post_ops_create)(&mut post) }, "post_ops")?;
            if sum {
                Api::check(
                    unsafe { (self.api.post_ops_sum)(post, 1.0, 0, dnnl_data_type_undef) },
                    "sum post-op",
                )?;
            }
            if let Some((md, alg)) = gate {
                Api::check(
                    unsafe { (self.api.post_ops_eltwise)(post, alg, 0.0, 0.0) },
                    "gelu post-op",
                )?;
                Api::check(
                    unsafe { (self.api.post_ops_binary)(post, dnnl_binary_mul, md) },
                    "binary-mul post-op",
                )?;
            }
            Api::check(
                unsafe { (self.api.attr_post_ops)(attr, post) },
                "set post_ops",
            )?;
        }
        Ok((attr, if post.is_null() { None } else { Some(post) }))
    }

    fn clone_md(&self, src: const_dnnl_memory_desc_t) -> AResult<dnnl_memory_desc_t> {
        check_shape!(!src.is_null(), "null memory desc");
        let mut md = ptr::null_mut();
        Api::check(unsafe { (self.api.md_clone)(&mut md, src) }, "clone md")?;
        Ok(md)
    }

    fn md_query_i32(&self, md: dnnl_memory_desc_t, what: dnnl_query_t) -> i32 {
        let mut v = 0i32;
        let rc = unsafe { (self.api.md_query)(md, what, (&mut v as *mut i32).cast()) };
        if rc == dnnl_success {
            v
        } else {
            0
        }
    }

    fn md_query_dims(&self, md: dnnl_memory_desc_t, what: dnnl_query_t) -> [i64; 12] {
        let mut p: *const dnnl_dim_t = ptr::null();
        let rc =
            unsafe { (self.api.md_query)(md, what, (&mut p as *mut *const dnnl_dim_t).cast()) };
        let mut out = [0i64; 12];
        if rc == dnnl_success && !p.is_null() {
            unsafe {
                out.copy_from_slice(std::slice::from_raw_parts(p, 12));
            }
        }
        out
    }

    fn layout_tag_of(&self, md: dnnl_memory_desc_t, packed_bytes: usize) -> LayoutTag {
        let mut tag = empty_layout_tag();
        tag[0..4].copy_from_slice(b"MD01");
        let dt = self.md_query_i32(md, dnnl_query_data_type) as u32;
        let fk = self.md_query_i32(md, dnnl_query_format_kind) as u32;
        let ndims = self.md_query_i32(md, dnnl_query_ndims_s32) as u32;
        let nblks = self.md_query_i32(md, dnnl_query_inner_nblks_s32) as u32;
        tag[4..8].copy_from_slice(&dt.to_le_bytes());
        tag[8..12].copy_from_slice(&fk.to_le_bytes());
        tag[12..16].copy_from_slice(&ndims.to_le_bytes());
        tag[16..20].copy_from_slice(&nblks.to_le_bytes());
        tag[24..32].copy_from_slice(&(packed_bytes as u64).to_le_bytes());
        let dims = self.md_query_dims(md, dnnl_query_dims);
        let padded = self.md_query_dims(md, dnnl_query_padded_dims);
        let strides = self.md_query_dims(md, dnnl_query_strides);
        let inner_blks = self.md_query_dims(md, dnnl_query_inner_blks);
        let inner_idxs = self.md_query_dims(md, dnnl_query_inner_idxs);
        for i in 0..4 {
            let o = 32 + i * 8;
            tag[o..o + 8].copy_from_slice(&dims[i].to_le_bytes());
            tag[64 + i * 8..72 + i * 8].copy_from_slice(&padded[i].to_le_bytes());
            tag[96 + i * 8..104 + i * 8].copy_from_slice(&strides[i].to_le_bytes());
            tag[128 + i * 8..136 + i * 8].copy_from_slice(&inner_blks[i].to_le_bytes());
            tag[160 + i * 8..168 + i * 8].copy_from_slice(&inner_idxs[i].to_le_bytes());
        }
        tag
    }

    fn impl_name(&self, pd: dnnl_primitive_desc_t) -> String {
        let mut p: *const std::ffi::c_char = ptr::null();
        let rc = unsafe {
            (self.api.pd_query)(
                pd,
                dnnl_query_impl_info_str,
                0,
                (&mut p as *mut *const std::ffi::c_char).cast(),
            )
        };
        if rc != dnnl_success || p.is_null() {
            return "unknown".into();
        }
        crate::api::cstr(p).to_string()
    }

    fn scratch_size(&self, pd: dnnl_primitive_desc_t) -> usize {
        let md = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        if md.is_null() {
            0
        } else {
            unsafe { (self.api.md_size)(md) }
        }
    }

    fn create_prim(&self, pd: dnnl_primitive_desc_t) -> AResult<dnnl_primitive_t> {
        let mut prim = ptr::null_mut();
        Api::check(
            unsafe { (self.api.prim_create)(&mut prim, pd) },
            "primitive",
        )?;
        Ok(prim)
    }

    fn mem(&self, md: dnnl_memory_desc_t) -> AResult<dnnl_memory_t> {
        let mut mem = ptr::null_mut();
        Api::check(
            unsafe { (self.api.mem_create)(&mut mem, md, self.engine, memory_none()) },
            "memory",
        )?;
        Ok(mem)
    }

    fn select_weight_md(&self, n: i64, k: i64, dt: MatDt) -> AResult<(dnnl_memory_desc_t, usize)> {
        // Layout probe uses a fixed activation M=64. Packed-weight cache is
        // ISA + weight shape, not live token count or thread width.
        let a = self.md_strides_dt(&[64, k], &[k, 1], dt.src())?;
        let b = self.md_any_dt(&[k, n], dt.weights())?;
        let c = self.md_strides(&[64, n], &[n, 1])?;
        let (attr, post) = if dt != MatDt::F32 {
            self.bf16_attr(false, None)?
        } else {
            self.strict_attr(false, None)?
        };
        let mut pd = ptr::null_mut();
        let st = unsafe { (self.api.matmul_pd)(&mut pd, self.engine, a, b, ptr::null(), c, attr) };
        unsafe {
            (self.api.attr_destroy)(attr);
            if let Some(p) = post {
                (self.api.post_ops_destroy)(p);
            }
            (self.api.md_destroy)(a);
            (self.api.md_destroy)(b);
            (self.api.md_destroy)(c);
        }
        Api::check(st, "weight-format matmul pd")?;
        let queried = unsafe { (self.api.pd_query_md)(pd, dnnl_query_weights_md, 0) };
        let packed = self.clone_md(queried)?;
        let bytes = unsafe { (self.api.md_size)(packed) };
        unsafe { (self.api.pd_destroy)(pd) };
        Ok((packed, bytes))
    }

    fn make_layout(&self, n: usize, k: usize, dt: MatDt) -> AResult<Arc<WeightLayout>> {
        check_shape!(
            n > 0 && k > 0 && n <= i64::MAX as usize && k <= i64::MAX as usize,
            "packed matrix [{n}, {k}]"
        );
        let raw_bytes = rt_core::checked_size(
            "packed matrix bytes",
            &[n, k, if dt == MatDt::F32 { 4 } else { 2 }],
        )?;
        check_shape!(
            raw_bytes <= 1024 * 1024 * 1024,
            "packed matrix [{n}, {k}] exceeds 1 GiB weight limit"
        );
        let _ = self.api.apply_threads(self.threads);
        let (packed_md, bytes) = self.select_weight_md(n as i64, k as i64, dt)?;
        let raw_md = self.md_strides_dt(&[k as i64, n as i64], &[1, k as i64], dt.weights())?;
        let (attr, post) = if dt != MatDt::F32 {
            self.bf16_attr(false, None)?
        } else {
            self.strict_attr(false, None)?
        };
        let mut pd = ptr::null_mut();
        let st = unsafe {
            (self.api.reorder_pd)(&mut pd, raw_md, self.engine, packed_md, self.engine, attr)
        };
        unsafe {
            (self.api.attr_destroy)(attr);
            if let Some(p) = post {
                (self.api.post_ops_destroy)(p);
            }
        }
        Api::check(st, "reorder pd")?;
        let scratch = self.scratch_size(pd);
        self.check_workspace("operator scratch", &[scratch])?;
        let q = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        let scratch_md = if q.is_null() {
            self.md_strides(&[16], &[1])?
        } else {
            self.clone_md(q)?
        };
        let reorder = self.create_prim(pd)?;
        unsafe { (self.api.pd_destroy)(pd) };
        check_shape!(
            bytes > 0 && bytes <= 1024 * 1024 * 1024 && scratch <= 512 * 1024 * 1024,
            "packing allocation size"
        );
        Ok(Arc::new(WeightLayout {
            api: self.api.clone(),
            packed_md,
            raw_md,
            scratch_md,
            reorder,
            bytes,
            scratch,
        }))
    }

    fn layout_for(&self, n: usize, k: usize, dt: MatDt) -> AResult<Arc<WeightLayout>> {
        let s = self
            .state
            .lock()
            .map_err(|_| rt_core::Error::Poisoned("backend"))?;
        if let Some(l) = s.layouts.get(&(n, k, dt)) {
            return Ok(l.clone());
        }
        drop(s);
        let l = self.make_layout(n, k, dt)?;
        let mut s = self
            .state
            .lock()
            .map_err(|_| rt_core::Error::Poisoned("backend"))?;
        s.layouts.insert((n, k, dt), l.clone());
        Ok(l)
    }

    /// Shape policy selected once from the effective (possibly pinned) ISA.
    pub fn strategy(&self) -> crate::IsaStrategy {
        self.strategy
    }

    pub fn pack(&self, n: usize, k: usize, src: &[f32]) -> AResult<PackedWeight> {
        check_shape!(
            src.len() == n.checked_mul(k).context("weight size")?,
            "weight shape"
        );
        self.reorder_pack(n, k, MatDt::F32, src.as_ptr() as *const u8)
    }

    pub fn pack_bf16(&self, n: usize, k: usize, src: &[u16]) -> AResult<PackedWeight> {
        check_shape!(
            src.len() == n.checked_mul(k).context("weight size")?,
            "weight shape"
        );
        self.reorder_pack(n, k, MatDt::Bf16, src.as_ptr() as *const u8)
    }

    /// Pack BF16 weights for f32 activations: BF16 storage, FP32 math.
    pub fn pack_bf16_f32(&self, n: usize, k: usize, src: &[u16]) -> AResult<PackedWeight> {
        check_shape!(
            src.len() == n.checked_mul(k).context("weight size")?,
            "weight shape"
        );
        self.reorder_pack(n, k, MatDt::F32Bf16, src.as_ptr() as *const u8)
    }

    fn reorder_pack(&self, n: usize, k: usize, dt: MatDt, src: *const u8) -> AResult<PackedWeight> {
        let layout = self.layout_for(n, k, dt)?;
        let mut ws = self
            .exec
            .acquire(|| ExecWorkspace::new(&self.api, self.engine, self.threadpool()))?;
        let mut s = self
            .state
            .lock()
            .map_err(|_| rt_core::Error::Poisoned("backend"))?;
        let data = Aligned::new(layout.bytes)?;
        if ws.scratch.bytes < layout.scratch {
            ws.scratch = Aligned::new(layout.scratch)?;
        }
        let _ = self.api.apply_threads(self.threads);
        let src_mem = self.mem(layout.raw_md)?;
        let dst_mem = self.mem(layout.packed_md)?;
        let scratch_mem = self.mem(layout.scratch_md)?;
        unsafe {
            Api::check(
                (self.api.mem_set)(src_mem, src as *mut c_void),
                "reorder src",
            )?;
            Api::check((self.api.mem_set)(dst_mem, data.ptr()), "reorder dst")?;
            Api::check(
                (self.api.mem_set)(scratch_mem, ws.scratch.ptr()),
                "reorder scratch",
            )?;
            let args = [
                dnnl_exec_arg_t {
                    arg: DNNL_ARG_FROM,
                    memory: src_mem,
                },
                dnnl_exec_arg_t {
                    arg: DNNL_ARG_TO,
                    memory: dst_mem,
                },
                dnnl_exec_arg_t {
                    arg: DNNL_ARG_SCRATCHPAD,
                    memory: scratch_mem,
                },
            ];
            Api::check(
                (self.api.prim_execute)(layout.reorder, ws.stream, 3, args.as_ptr()),
                "reorder execute",
            )?;
            Api::check((self.api.stream_wait)(ws.stream), "reorder wait")?;
            (self.api.mem_destroy)(src_mem);
            (self.api.mem_destroy)(dst_mem);
            (self.api.mem_destroy)(scratch_mem);
        }
        s.weight_bytes += layout.bytes;
        let tag = self.layout_tag_of(layout.packed_md, layout.bytes);
        Ok(PackedWeight {
            layout,
            data,
            n,
            k,
            dt,
            tag,
        })
    }

    pub fn pack_from_cache(&self, n: usize, k: usize, bytes: &[u8]) -> AResult<PackedWeight> {
        self.pack_from_cache_dt(n, k, bytes, MatDt::F32, None)
    }

    pub fn pack_from_cache_bf16(&self, n: usize, k: usize, bytes: &[u8]) -> AResult<PackedWeight> {
        self.pack_from_cache_dt(n, k, bytes, MatDt::Bf16, None)
    }

    fn pack_from_cache_dt(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        dt: MatDt,
        expected: Option<&LayoutTag>,
    ) -> AResult<PackedWeight> {
        let layout = self.layout_for(n, k, dt)?;
        let tag = self.layout_tag_of(layout.packed_md, layout.bytes);
        if let Some(exp) = expected {
            check_shape!(
                *exp == tag,
                "packed layout ISA/format mismatch (cache vs live oneDNN MD)"
            );
        }
        check_shape!(bytes.len() == layout.bytes, "cached packed size");
        let data = Aligned::copy_from(bytes)?;
        let mut s = self
            .state
            .lock()
            .map_err(|_| rt_core::Error::Poisoned("backend"))?;
        s.weight_bytes += layout.bytes;
        Ok(PackedWeight {
            layout,
            data,
            n,
            k,
            dt,
            tag,
        })
    }

    fn build_plan(&self, key: &Key, w: Option<&PackedWeight>) -> AResult<Plan> {
        let _ = self.api.apply_threads(self.threads);
        let kind = key.kind;
        let g = key.g as i64;
        let m = key.m as i64;
        let n = key.n as i64;
        let k = key.k as i64;
        let mut owned_mds: Vec<dnnl_memory_desc_t> = Vec::new();
        let src_md;
        let mut weights_md = ptr::null_mut();
        let dst_md;
        let mut scale_md = ptr::null_mut();
        let mut gate_md = ptr::null_mut();
        let mut pd = ptr::null_mut();
        let (attr, post);

        if kind == 0 || kind == 7 || kind == 8 {
            check_shape!(k > 0, "matmul K");
            let src_dt = key.dt.src();
            src_md = if g == 1 {
                self.md_strides_dt(&[m, k], &[key.ar as i64, 1], src_dt)?
            } else {
                self.md_strides_dt(&[g, m, k], &[key.ab as i64, key.ar as i64, 1], src_dt)?
            };
            owned_mds.push(src_md);
            if let Some(weight) = w {
                check_shape!(g == 1, "batched packed weight");
                check_shape!(
                    weight.dt == key.dt,
                    "packed weight dtype does not match plan"
                );
                weights_md = weight.layout.packed_md;
            } else if g == 1 {
                weights_md = if key.trans {
                    self.md_strides_dt(&[k, n], &[1, key.br as i64], key.dt.weights())?
                } else {
                    self.md_strides_dt(&[k, n], &[key.br as i64, 1], key.dt.weights())?
                };
                owned_mds.push(weights_md);
            } else {
                weights_md = if key.trans {
                    self.md_strides_dt(
                        &[g, k, n],
                        &[key.bb as i64, 1, key.br as i64],
                        key.dt.weights(),
                    )?
                } else {
                    self.md_strides_dt(
                        &[g, k, n],
                        &[key.bb as i64, key.br as i64, 1],
                        key.dt.weights(),
                    )?
                };
                owned_mds.push(weights_md);
            }
            dst_md = if g == 1 {
                self.md_strides(&[m, n], &[n, 1])?
            } else {
                self.md_strides(&[g, m, n], &[m * n, n, 1])?
            };
            owned_mds.push(dst_md);
            if kind == 7 || kind == 8 {
                check_shape!(!key.sum, "gated projection sum");
                gate_md = dst_md;
            }
            let pair = if key.dt != MatDt::F32 || kind == 8 {
                let gate = if kind == 7 {
                    Some((gate_md, dnnl_eltwise_gelu_erf))
                } else if kind == 8 {
                    Some((gate_md, dnnl_eltwise_gelu_tanh))
                } else {
                    None
                };
                self.bf16_attr(key.sum, gate)?
            } else {
                self.strict_attr(key.sum, if kind == 7 { Some(gate_md) } else { None })?
            };
            attr = pair.0;
            post = pair.1;
            Api::check(
                unsafe {
                    (self.api.matmul_pd)(
                        &mut pd,
                        self.engine,
                        src_md,
                        weights_md,
                        ptr::null(),
                        dst_md,
                        attr,
                    )
                },
                "matmul pd",
            )?;
        } else {
            src_md = self.md_strides(&[m, n], &[n, 1])?;
            dst_md = src_md;
            owned_mds.push(src_md);
            let pair = self.strict_attr(false, None)?;
            attr = pair.0;
            post = pair.1;
            if kind == 1 {
                scale_md = self.md_strides(&[n], &[1])?;
                owned_mds.push(scale_md);
                let eps = f32::from_bits(key.param);
                Api::check(
                    unsafe {
                        (self.api.layernorm_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_forward_inference,
                            src_md,
                            dst_md,
                            ptr::null(),
                            eps,
                            dnnl_use_scale | dnnl_use_shift,
                            attr,
                        )
                    },
                    "layernorm pd",
                )?;
            } else if kind == 2 {
                Api::check(
                    unsafe {
                        (self.api.softmax_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_forward_inference,
                            dnnl_softmax_accurate,
                            src_md,
                            dst_md,
                            1,
                            attr,
                        )
                    },
                    "softmax pd",
                )?;
            } else if kind == 3 || kind == 5 || kind == 6 || kind == 9 {
                let alg = match kind {
                    3 => dnnl_eltwise_gelu_erf,
                    6 => dnnl_eltwise_gelu_tanh,
                    9 => dnnl_eltwise_swish,
                    _ => dnnl_eltwise_relu,
                };
                Api::check(
                    unsafe {
                        (self.api.eltwise_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_forward_inference,
                            alg,
                            src_md,
                            dst_md,
                            if kind == 9 { 1.0 } else { 0.0 },
                            0.0,
                            attr,
                        )
                    },
                    "eltwise pd",
                )?;
            } else if kind == 4 {
                weights_md = src_md;
                Api::check(
                    unsafe {
                        (self.api.binary_pd)(
                            &mut pd,
                            self.engine,
                            dnnl_binary_mul,
                            src_md,
                            weights_md,
                            dst_md,
                            attr,
                        )
                    },
                    "binary pd",
                )?;
            } else {
                rt_core::invalid_shape!("operator kind {kind}");
            }
        }

        unsafe {
            (self.api.attr_destroy)(attr);
            if let Some(p) = post {
                (self.api.post_ops_destroy)(p);
            }
        }
        let scratch = self.scratch_size(pd);
        self.check_workspace("operator scratch", &[scratch])?;
        check_shape!(scratch <= 512 * 1024 * 1024, "operator scratch budget");
        let name = self.impl_name(pd);
        let prim = self.create_prim(pd)?;
        let q = unsafe { (self.api.pd_query_md)(pd, dnnl_query_scratchpad_md, 0) };
        let scratch_md = if q.is_null() {
            self.md_strides(&[16], &[1])?
        } else {
            self.clone_md(q)?
        };
        owned_mds.push(scratch_md);
        unsafe { (self.api.pd_destroy)(pd) };

        let mut slots = Vec::new();
        let mut slot_specs = Vec::new();
        let src_arg = if kind == 4 {
            DNNL_ARG_SRC_0
        } else {
            DNNL_ARG_SRC
        };
        let mut push_slot = |arg: i32, md: dnnl_memory_desc_t| -> AResult<()> {
            slots.push(MemSlot {
                arg,
                mem: self.mem(md)?,
            });
            slot_specs.push((arg, md));
            Ok(())
        };
        push_slot(src_arg, src_md)?;
        push_slot(DNNL_ARG_DST, dst_md)?;
        push_slot(DNNL_ARG_SCRATCHPAD, scratch_md)?;
        if kind == 0 || kind == 7 || kind == 8 {
            push_slot(DNNL_ARG_WEIGHTS, weights_md)?;
        } else if kind == 1 {
            push_slot(DNNL_ARG_SCALE, scale_md)?;
            push_slot(DNNL_ARG_SHIFT, scale_md)?;
        } else if kind == 4 {
            push_slot(DNNL_ARG_SRC_1, src_md)?;
        }
        if kind == 7 || kind == 8 {
            push_slot(
                (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * 2) | DNNL_ARG_SRC_1,
                gate_md,
            )?;
        }

        Ok(Plan {
            api: self.api.clone(),
            prim,
            slot_specs,
            slots,
            mds: owned_mds,
            scratch,
            name,
            used: AtomicU64::new(0),
            bytes: 0,
        })
    }

    fn run(
        &self,
        key: Key,
        a: &[f32],
        b: Option<&[f32]>,
        shift: Option<&[f32]>,
        out: &mut [f32],
        w: Option<&PackedWeight>,
    ) -> AResult<()> {
        self.run_ex(key, a, None, b, None, shift, out, w)
    }

    fn run_ex(
        &self,
        key: Key,
        a: &[f32],
        a_bf: Option<&[u16]>,
        b: Option<&[f32]>,
        b_bf: Option<&[u16]>,
        shift: Option<&[f32]>,
        out: &mut [f32],
        w: Option<&PackedWeight>,
    ) -> AResult<()> {
        let a_ok = a_bf.map(|v| !v.is_empty()).unwrap_or(!a.is_empty());
        check_shape!(a_ok && !out.is_empty(), "empty operator input");
        let mut ws = self
            .exec
            .acquire(|| ExecWorkspace::new(&self.api, self.engine, self.threadpool()))?;
        // Create a missing plan before taking the compute lock: primitive
        // creation (JIT, descriptors) touches no other caller's state, so a
        // cold shape on one model no longer stalls every model in the process.
        let missing = !self
            .state
            .lock()
            .map_err(|_| rt_core::Error::Poisoned("backend"))?
            .plans
            .contains_key(&key);
        let mut built = None;
        if missing {
            let t = Instant::now();
            let heap = heap_in_use();
            let mut plan = self.build_plan(&key, w)?;
            // Other threads allocate meanwhile, so this is an estimate; a
            // plan holds well under 64 MiB.
            plan.bytes = heap
                .zip(heap_in_use())
                .map_or(0, |(a, b)| b.saturating_sub(a).min(64 << 20));
            built = Some((plan, t));
        }
        // After the exec slot: AMX is programmed on this caller thread inside
        // `prim_execute`, so two in-flight executes must not overlap.
        let _compute = compute_lock();
        let mut s = self
            .state
            .lock()
            .map_err(|_| rt_core::Error::Poisoned("backend"))?;
        if !s.plans.contains_key(&key) {
            // Evicted since the check above (budget), or built here first time.
            let (plan, t) = match built {
                Some(built) => built,
                None => {
                    let t = Instant::now();
                    drop(s);
                    let heap = heap_in_use();
                    let mut plan = self.build_plan(&key, w)?;
                    plan.bytes = heap
                        .zip(heap_in_use())
                        .map_or(0, |(a, b)| b.saturating_sub(a).min(64 << 20));
                    s = self
                        .state
                        .lock()
                        .map_err(|_| rt_core::Error::Poisoned("backend"))?;
                    (plan, t)
                }
            };
            // Evicting an arbitrary entry (HashMap order) kept evicting plans
            // the next layer needed: Vela spent ~150 ms per request re-creating
            // plans once 256 distinct shapes had been seen.
            if s.plans.len() >= s.capacity {
                let old = s
                    .plans
                    .iter()
                    .min_by_key(|(_, plan)| plan.used.load(Ordering::Relaxed))
                    .map(|(key, _)| key.clone());
                if let Some(plan) = old.and_then(|old| s.plans.remove(&old)) {
                    self.retained.plans.fetch_sub(plan.bytes, Ordering::Relaxed);
                    s.capacity_evictions += 1;
                }
            }
            self.retained.plans.fetch_add(plan.bytes, Ordering::Relaxed);
            s.plans.insert(key.clone(), Arc::new(plan));
            s.creations += 1;
            s.setup_s += t.elapsed().as_secs_f64();
            self.enforce_budget(&mut s, &key);
        }
        s.tick += 1;
        let tick = s.tick;
        // Held through the execute: inside a pool job another batch may
        // evict this plan from the cache meanwhile.
        let plan = Arc::clone(s.plans.get(&key).context("execution plan")?);
        plan.used.store(tick, Ordering::Relaxed);
        let scratch_need = plan.scratch;
        self.check_workspace("operator scratch", &[scratch_need])?;
        if ws.scratch.bytes < scratch_need.max(64) {
            let old = ws.scratch.bytes;
            ws.scratch = Aligned::new(scratch_need.max(64))?;
            self.retained
                .exec_scratch
                .fetch_add(ws.scratch.bytes - old, Ordering::Relaxed);
        }
        let prim = plan.prim;
        // Plan-owned memories are shared, so they are only bound under the
        // state lock. With several slots, or inside a pool job, each execute
        // binds its own and lets the others run.
        let concurrent = self.exec_slots > 1 || rt_threads::in_parallel();
        let slot_specs = if concurrent {
            plan.slot_specs.clone()
        } else {
            Vec::new()
        };
        let slots: Vec<(i32, dnnl_memory_t)> = if concurrent {
            Vec::new()
        } else {
            plan.slots.iter().map(|slot| (slot.arg, slot.mem)).collect()
        };
        let mut state_guard = if concurrent {
            drop(s);
            None
        } else {
            Some(s)
        };
        let _ = self.api.apply_threads(self.threads);
        let t_conv = Instant::now();
        let mut converted = false;
        let src_p = if let Some(bf) = a_bf {
            bf.as_ptr() as *mut c_void
        } else if key.dt == MatDt::Bf16 {
            converted = true;
            fill_bf16(&mut ws.bf16_a, a, self.threads)?
        } else {
            a.as_ptr() as *mut c_void
        };
        let bp = if let Some(v) = w {
            v.data.ptr.as_ptr() as *mut c_void
        } else if let Some(bf) = b_bf {
            bf.as_ptr() as *mut c_void
        } else if key.dt == MatDt::Bf16 {
            if let Some(bv) = b {
                converted = true;
                fill_bf16(&mut ws.bf16_b, bv, self.threads)?
            } else {
                ptr::null_mut()
            }
        } else {
            b.map(|v| v.as_ptr() as *mut c_void)
                .unwrap_or(ptr::null_mut())
        };
        let convert_dt = if converted {
            t_conv.elapsed().as_secs_f64()
        } else {
            0.0
        };
        let shift_p = shift
            .map(|v| v.as_ptr() as *mut c_void)
            .unwrap_or(ptr::null_mut());
        let t = Instant::now();
        let stream = ws.stream;
        let scratch_ptr = ws.scratch.ptr();
        let mut ephemeral = Vec::new();
        let slot_pairs: Vec<(i32, dnnl_memory_t)> = if concurrent {
            for (arg, md) in slot_specs {
                ephemeral.push((arg, self.mem(md)?));
            }
            ephemeral.clone()
        } else {
            slots
        };
        unsafe {
            let mut args: Vec<dnnl_exec_arg_t> = Vec::with_capacity(slot_pairs.len());
            for (arg, mem) in &slot_pairs {
                let handle = if *arg == DNNL_ARG_SRC || *arg == DNNL_ARG_SRC_0 {
                    src_p
                } else if *arg == DNNL_ARG_DST {
                    out.as_mut_ptr() as *mut c_void
                } else if *arg == DNNL_ARG_SCRATCHPAD {
                    scratch_ptr
                } else if *arg == DNNL_ARG_WEIGHTS
                    || *arg == DNNL_ARG_SRC_1
                    || *arg == DNNL_ARG_SCALE
                {
                    bp
                } else if *arg == DNNL_ARG_SHIFT
                    || *arg == (DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE * 2) | DNNL_ARG_SRC_1
                {
                    shift_p
                } else {
                    ptr::null_mut()
                };
                Api::check((self.api.mem_set)(*mem, handle), "set handle")?;
                args.push(dnnl_exec_arg_t {
                    arg: *arg,
                    memory: *mem,
                });
            }
            Api::check(
                (self.api.prim_execute)(prim, stream, args.len() as i32, args.as_ptr()),
                "execute",
            )?;
            Api::check((self.api.stream_wait)(stream), "wait")?;
            if concurrent {
                for (_, mem) in ephemeral {
                    (self.api.mem_destroy)(mem);
                }
            }
        }
        let elapsed = t.elapsed().as_secs_f64();
        let mut s = match state_guard.take() {
            Some(g) => g,
            None => self
                .state
                .lock()
                .map_err(|_| rt_core::Error::Poisoned("backend"))?,
        };
        if convert_dt > 0.0 {
            s.convert_s += convert_dt;
        }
        *s.executions.entry(key.kind).or_default() += 1;
        *s.exec_s.entry(key.kind).or_default() += elapsed;
        let category = if key.kind == 0 || key.kind == 7 || key.kind == 8 {
            if key.weight == 0 {
                "attention_or_span"
            } else {
                "packed_linear"
            }
        } else {
            "other_primitives"
        };
        *s.category_s.entry(category.into()).or_default() += elapsed;
        Ok(())
    }

    pub fn linear_inner(
        &self,
        x: &[f32],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> AResult<()> {
        check_shape!(x.len() % w.k == 0, "linear input width");
        let m = x.len() / w.k;
        check_shape!(
            out.len() == rt_core::checked_size("linear output", &[m, w.n])?,
            "linear destination"
        );
        self.run(
            Key {
                kind: 0,
                g: 1,
                m,
                n: w.n,
                k: w.k,
                ab: 0,
                ar: w.k,
                bb: 0,
                br: w.k,
                trans: true,
                param: 0,
                sum,
                weight: w.layout.packed_md as usize,
                // BF16-stored FP32 weights keep FP32 math; BF16 weights
                // belong to `linear_bf16`.
                dt: if w.dt == MatDt::F32Bf16 {
                    MatDt::F32Bf16
                } else {
                    MatDt::F32
                },
            },
            x,
            None,
            None,
            out,
            Some(w),
        )
    }

    pub fn linear_bf16_inner(
        &self,
        x: &[f32],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> AResult<()> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "linear_bf16 requires packed BF16 weights"
        );
        check_shape!(x.len() % w.k == 0, "linear input width");
        let m = x.len() / w.k;
        check_shape!(
            out.len() == rt_core::checked_size("linear output", &[m, w.n])?,
            "linear destination"
        );
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            return self.run(
                self.packed_bf16_key(0, m, w, sum),
                x,
                None,
                None,
                out,
                Some(w),
            );
        }
        let mut src =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0.0);
        let mut dst =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.n])?)?;
        if sum {
            // The sum post-op accumulates into `dst`, so it starts as `out`.
            dst[..out.len()].copy_from_slice(out);
            dst[out.len()..].fill(0.0);
        }
        let rc = self.run(
            self.packed_bf16_key(0, m_pad, w, sum),
            &src,
            None,
            None,
            &mut dst,
            Some(w),
        );
        if rc.is_ok() {
            out.copy_from_slice(&dst[..out.len()]);
        }
        self.hw_recycle(dst);
        self.hw_recycle(src);
        rc
    }

    pub fn gated_inner(&self, x: &[f32], w: &PackedWeight, gate: &[f32]) -> AResult<Vec<f32>> {
        check_shape!(x.len() % w.k == 0, "gated input");
        let m = x.len() / w.k;
        check_shape!(
            gate.len() == rt_core::checked_size("gate output", &[m, w.n])?,
            "gate shape"
        );
        let mut out = self.hw_buffer(gate.len())?;
        self.run(
            Key {
                kind: 7,
                g: 1,
                m,
                n: w.n,
                k: w.k,
                ab: 0,
                ar: w.k,
                bb: 0,
                br: w.k,
                trans: true,
                param: 0,
                sum: false,
                weight: w.layout.packed_md as usize,
                dt: MatDt::F32,
            },
            x,
            None,
            Some(gate),
            &mut out,
            Some(w),
        )?;
        Ok(out)
    }

    pub fn gated_tanh_inner(&self, x: &[f32], w: &PackedWeight, gate: &[f32]) -> AResult<Vec<f32>> {
        check_shape!(x.len() % w.k == 0, "gated input");
        let m = x.len() / w.k;
        check_shape!(
            gate.len() == rt_core::checked_size("gate output", &[m, w.n])?,
            "gate shape"
        );
        if w.dt != MatDt::Bf16 {
            // FP32 math: no BF16 conversion and no AMX row padding.
            let mut out = self.hw_buffer(gate.len())?;
            self.run(
                Key {
                    dt: w.dt,
                    ..self.packed_bf16_key(8, m, w, false)
                },
                x,
                None,
                Some(gate),
                &mut out,
                Some(w),
            )?;
            return Ok(out);
        }
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            let mut out = self.hw_buffer(gate.len())?;
            self.run(
                self.packed_bf16_key(8, m, w, false),
                x,
                None,
                Some(gate),
                &mut out,
                Some(w),
            )?;
            return Ok(out);
        }
        let mut src =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0.0);
        let mut gate_pad =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.n])?)?;
        gate_pad[..gate.len()].copy_from_slice(gate);
        gate_pad[gate.len()..].fill(0.0);
        let mut dst =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.n])?)?;
        let rc = self.run(
            self.packed_bf16_key(8, m_pad, w, false),
            &src,
            None,
            Some(&gate_pad),
            &mut dst,
            Some(w),
        );
        self.hw_recycle(src);
        self.hw_recycle(gate_pad);
        rc?;
        let mut out = self.hw_buffer(gate.len())?;
        out.copy_from_slice(&dst[..gate.len()]);
        self.hw_recycle(dst);
        Ok(out)
    }

    pub fn linear_bf16_src_inner(
        &self,
        x: &[u16],
        w: &PackedWeight,
        out: &mut [f32],
        sum: bool,
    ) -> AResult<()> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "linear_bf16_src requires packed BF16 weights"
        );
        check_shape!(x.len() % w.k == 0, "linear input width");
        let m = x.len() / w.k;
        check_shape!(
            out.len() == rt_core::checked_size("linear output", &[m, w.n])?,
            "linear destination"
        );
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            return self.run_ex(
                self.packed_bf16_key(0, m, w, sum),
                &[],
                Some(x),
                None,
                None,
                None,
                out,
                Some(w),
            );
        }
        let mut src =
            self.hw_buffer_u16(rt_core::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0);
        let mut dst =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.n])?)?;
        if sum {
            dst[..out.len()].copy_from_slice(out);
            dst[out.len()..].fill(0.0);
        }
        let rc = self.run_ex(
            self.packed_bf16_key(0, m_pad, w, sum),
            &[],
            Some(&src),
            None,
            None,
            None,
            &mut dst,
            Some(w),
        );
        if rc.is_ok() {
            out.copy_from_slice(&dst[..out.len()]);
        }
        self.hw_recycle(dst);
        self.hw_recycle_u16(src);
        rc
    }

    pub fn gated_tanh_src_inner(
        &self,
        x: &[u16],
        w: &PackedWeight,
        gate: &[f32],
    ) -> AResult<Vec<f32>> {
        check_shape!(
            w.dt == MatDt::Bf16,
            "gated_linear_tanh_src requires packed BF16 weights"
        );
        check_shape!(x.len() % w.k == 0, "gated input");
        let m = x.len() / w.k;
        check_shape!(
            gate.len() == rt_core::checked_size("gate output", &[m, w.n])?,
            "gate shape"
        );
        let m_pad = self.strategy.gemm_pad_m(m);
        if m_pad == m {
            let mut out = self.hw_buffer(gate.len())?;
            self.run_ex(
                self.packed_bf16_key(8, m, w, false),
                &[],
                Some(x),
                None,
                None,
                Some(gate),
                &mut out,
                Some(w),
            )?;
            return Ok(out);
        }
        let mut src =
            self.hw_buffer_u16(rt_core::checked_size("activation elements", &[m_pad, w.k])?)?;
        src[..x.len()].copy_from_slice(x);
        src[x.len()..].fill(0);
        let mut gate_pad =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.n])?)?;
        gate_pad[..gate.len()].copy_from_slice(gate);
        gate_pad[gate.len()..].fill(0.0);
        let mut dst =
            self.hw_buffer(rt_core::checked_size("activation elements", &[m_pad, w.n])?)?;
        let rc = self.run_ex(
            self.packed_bf16_key(8, m_pad, w, false),
            &[],
            Some(&src),
            None,
            None,
            Some(&gate_pad),
            &mut dst,
            Some(w),
        );
        self.hw_recycle_u16(src);
        self.hw_recycle(gate_pad);
        rc?;
        let mut out = self.hw_buffer(gate.len())?;
        out.copy_from_slice(&dst[..gate.len()]);
        self.hw_recycle(dst);
        Ok(out)
    }

    fn packed_bf16_key(&self, kind: i32, m: usize, w: &PackedWeight, sum: bool) -> Key {
        Key {
            kind,
            g: 1,
            m,
            n: w.n,
            k: w.k,
            ab: 0,
            ar: w.k,
            bb: 0,
            br: w.k,
            trans: true,
            param: 0,
            sum,
            weight: w.layout.packed_md as usize,
            dt: MatDt::Bf16,
        }
    }

    pub fn mm_inner(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[f32],
        b: &[f32],
    ) -> AResult<Vec<f32>> {
        let na = rt_core::matrix_span(g, m, k, ab, ar)?;
        let nb = if trans {
            rt_core::matrix_span(g, n, k, bb, br)?
        } else {
            rt_core::matrix_span(g, k, n, bb, br)?
        };
        self.check_workspace("matmul output", &[g, m, n, 4])?;
        check_shape!(a.len() >= na && b.len() >= nb, "matmul slice bounds");
        let mut out = self.hw_buffer(rt_core::checked_size("activation elements", &[g, m, n])?)?;
        self.run(
            Key {
                kind: 0,
                g,
                m,
                n,
                k,
                ab,
                ar,
                bb,
                br,
                trans,
                param: 0,
                sum: false,
                weight: 0,
                dt: MatDt::F32,
            },
            a,
            Some(b),
            None,
            &mut out,
            None,
        )?;
        Ok(out)
    }

    pub fn mm_bf16_inner(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[f32],
        b: &[f32],
    ) -> AResult<Vec<f32>> {
        let na = rt_core::matrix_span(g, m, k, ab, ar)?;
        let nb = if trans {
            rt_core::matrix_span(g, n, k, bb, br)?
        } else {
            rt_core::matrix_span(g, k, n, bb, br)?
        };
        self.check_workspace("matmul output", &[g, m, n, 4])?;
        check_shape!(a.len() >= na && b.len() >= nb, "matmul slice bounds");
        let mut out = self.hw_buffer(rt_core::checked_size("activation elements", &[g, m, n])?)?;
        self.run(
            Key {
                kind: 0,
                g,
                m,
                n,
                k,
                ab,
                ar,
                bb,
                br,
                trans,
                param: 0,
                sum: false,
                weight: 0,
                dt: MatDt::Bf16,
            },
            a,
            Some(b),
            None,
            &mut out,
            None,
        )?;
        Ok(out)
    }

    pub fn mm_bf16_src_inner(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[u16],
        b: &[u16],
    ) -> AResult<Vec<f32>> {
        let na = rt_core::matrix_span(g, m, k, ab, ar)?;
        let nb = if trans {
            rt_core::matrix_span(g, n, k, bb, br)?
        } else {
            rt_core::matrix_span(g, k, n, bb, br)?
        };
        self.check_workspace("matmul output", &[g, m, n, 4])?;
        check_shape!(a.len() >= na && b.len() >= nb, "matmul slice bounds");
        let m_pad = self.strategy.attn_pad_m(m);
        let tight = ar == k && (g == 1 || ab == m * ar);
        if m_pad == m || !tight {
            let mut out =
                self.hw_buffer(rt_core::checked_size("activation elements", &[g, m, n])?)?;
            self.run_ex(
                Key {
                    kind: 0,
                    g,
                    m,
                    n,
                    k,
                    ab,
                    ar,
                    bb,
                    br,
                    trans,
                    param: 0,
                    sum: false,
                    weight: 0,
                    dt: MatDt::Bf16,
                },
                &[],
                Some(a),
                None,
                Some(b),
                None,
                &mut out,
                None,
            )?;
            return Ok(out);
        }
        let mut a_pad = self.hw_buffer_u16(rt_core::checked_size(
            "activation elements",
            &[g, m_pad, k],
        )?)?;
        for gi in 0..g {
            let src = gi * ab;
            let dst = gi * m_pad * k;
            a_pad[dst..dst + m * k].copy_from_slice(&a[src..src + m * k]);
            a_pad[dst + m * k..dst + m_pad * k].fill(0);
        }
        let mut dst = self.hw_buffer(rt_core::checked_size(
            "activation elements",
            &[g, m_pad, n],
        )?)?;
        let rc = self.run_ex(
            Key {
                kind: 0,
                g,
                m: m_pad,
                n,
                k,
                ab: m_pad * k,
                ar: k,
                bb,
                br,
                trans,
                param: 0,
                sum: false,
                weight: 0,
                dt: MatDt::Bf16,
            },
            &[],
            Some(&a_pad),
            None,
            Some(b),
            None,
            &mut dst,
            None,
        );
        self.hw_recycle_u16(a_pad);
        rc?;
        let mut out = self.hw_buffer(rt_core::checked_size("activation elements", &[g, m, n])?)?;
        for gi in 0..g {
            let src = gi * m_pad * n;
            let dst_o = gi * m * n;
            out[dst_o..dst_o + m * n].copy_from_slice(&dst[src..src + m * n]);
        }
        self.hw_recycle(dst);
        Ok(out)
    }

    /// Attention PV: `probs[g,M,K] @ V[g,K,D]`. Query `M` uses the same
    /// shape+ISA pad as packed linear (see [`crate::IsaStrategy`]).
    /// `nk` is the live KV span.
    fn attn_pv_inner(
        &self,
        g: usize,
        m: usize,
        d: usize,
        nk: usize,
        probs: &[f32],
        v: &[u16],
        v_bb: usize,
        v_br: usize,
    ) -> AResult<Vec<f32>> {
        check_shape!(probs.len() >= g * m * nk, "pv probs shape");
        let m_pad = self.strategy.attn_pad_m(m);
        let key = |rows: usize| Key {
            kind: 0,
            g,
            m: rows,
            n: d,
            k: nk,
            ab: rows * nk,
            ar: nk,
            bb: v_bb,
            br: v_br,
            trans: false,
            param: 0,
            sum: false,
            weight: 0,
            dt: MatDt::Bf16,
        };
        if m_pad == m {
            let mut y =
                self.hw_buffer(rt_core::checked_size("activation elements", &[g, m, d])?)?;
            self.run_ex(key(m), probs, None, None, Some(v), None, &mut y, None)?;
            return Ok(y);
        }
        let mut a_pad = self.hw_buffer(rt_core::checked_size(
            "activation elements",
            &[g, m_pad, nk],
        )?)?;
        for gi in 0..g {
            let src = gi * m * nk;
            let dst = gi * m_pad * nk;
            a_pad[dst..dst + m * nk].copy_from_slice(&probs[src..src + m * nk]);
            a_pad[dst + m * nk..dst + m_pad * nk].fill(0.0);
        }
        let mut dst = self.hw_buffer(rt_core::checked_size(
            "activation elements",
            &[g, m_pad, d],
        )?)?;
        let rc = self.run_ex(
            key(m_pad),
            &a_pad,
            None,
            None,
            Some(v),
            None,
            &mut dst,
            None,
        );
        self.hw_recycle(a_pad);
        rc?;
        let mut y = self.hw_buffer(rt_core::checked_size("activation elements", &[g, m, d])?)?;
        for gi in 0..g {
            let src = gi * m_pad * d;
            let dst_o = gi * m * d;
            y[dst_o..dst_o + m * d].copy_from_slice(&dst[src..src + m * d]);
        }
        self.hw_recycle(dst);
        Ok(y)
    }

    /// FP32 SiLU for OCR's SVTR MLP. Existing model op kinds are unchanged.
    pub fn swish(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(9, m, n, x, None, None, 0.)
    }

    pub fn op_inner(
        &self,
        kind: i32,
        m: usize,
        n: usize,
        x: &[f32],
        other: Option<&[f32]>,
        shift: Option<&[f32]>,
        eps: f32,
    ) -> AResult<Vec<f32>> {
        check_shape!(
            m > 0 && n > 0 && x.len() == rt_core::checked_size("operator shape", &[m, n])?,
            "operator [{m}, {n}] input length {}",
            x.len()
        );
        if kind == 1 {
            check_shape!(
                other.map(|v| v.len()) == Some(n) && shift.map(|v| v.len()) == Some(n),
                "layernorm parameters"
            );
        }
        if kind == 4 {
            check_shape!(other.map(|v| v.len()) == Some(x.len()), "binary shape");
        }
        let mut out = self.hw_buffer(x.len())?;
        self.run(
            Key {
                kind,
                g: 1,
                m,
                n,
                k: 0,
                ab: 0,
                ar: n,
                bb: 0,
                br: n,
                trans: false,
                param: eps.to_bits(),
                sum: false,
                weight: 0,
                dt: MatDt::F32,
            },
            x,
            other,
            shift,
            &mut out,
            None,
        )?;
        Ok(out)
    }

    fn hw_buffer(&self, len: usize) -> Result<Vec<f32>> {
        self.check_workspace("activation buffer", &[len, 4])?;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(buffer) = pool.buffers.get_mut(&len).and_then(Vec::pop) {
            pool.buffer_bytes -= buffer.capacity() * 4;
            self.retained
                .pool
                .store(pool.buffer_bytes, Ordering::Relaxed);
            return Ok(buffer);
        }
        drop(pool);
        rt_core::try_vec(len)
    }

    fn hw_recycle(&self, buffer: Vec<f32>) {
        const PER_LEN: usize = 4;
        let limit = self.workspace_limit;
        let bytes = buffer.capacity() * 4;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.is_empty()
            || bytes > limit
            || pool.buffer_bytes + bytes > limit
            || pool.buffers.get(&buffer.len()).map_or(0, Vec::len) >= PER_LEN
            || self.exceeds_budget(bytes)
        {
            return;
        }
        pool.buffer_bytes += bytes;
        self.retained
            .pool
            .store(pool.buffer_bytes, Ordering::Relaxed);
        pool.buffers.entry(buffer.len()).or_default().push(buffer);
    }

    fn hw_buffer_u16(&self, len: usize) -> Result<Vec<u16>> {
        self.check_workspace("BF16 activation buffer", &[len, 2])?;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(buffer) = pool.buffers_u16.get_mut(&len).and_then(Vec::pop) {
            pool.buffer_bytes -= buffer.capacity() * 2;
            self.retained
                .pool
                .store(pool.buffer_bytes, Ordering::Relaxed);
            return Ok(buffer);
        }
        drop(pool);
        rt_core::try_vec(len)
    }

    fn hw_recycle_u16(&self, buffer: Vec<u16>) {
        const PER_LEN: usize = 4;
        let limit = self.workspace_limit;
        let bytes = buffer.capacity() * 2;
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if buffer.is_empty()
            || bytes > limit
            || pool.buffer_bytes + bytes > limit
            || pool.buffers_u16.get(&buffer.len()).map_or(0, Vec::len) >= PER_LEN
            || self.exceeds_budget(bytes)
        {
            return;
        }
        pool.buffer_bytes += bytes;
        self.retained
            .pool
            .store(pool.buffer_bytes, Ordering::Relaxed);
        pool.buffers_u16
            .entry(buffer.len())
            .or_default()
            .push(buffer);
    }

    fn hw_reserve(&self, len: usize, n: usize) -> Result<()> {
        self.check_workspace("reserve activation buffers", &[len, n, 4])?;
        self.check_workspace("reserve activation", &[len, 4])?;
        if len == 0 || n == 0 {
            return Ok(());
        }
        // Top up to `n` pooled buffers of this length. Allocating all `n`
        // on every forward (and dropping the extras on recycle) cost a fresh
        // zero-fill and page faults per forward.
        let have = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .buffers
            .get(&len)
            .map_or(0, Vec::len);
        for _ in have..n {
            let mut buf = rt_core::try_vec::<f32>(len)?;
            touch_pages(&mut buf);
            self.hw_recycle(buf);
        }
        Ok(())
    }

    fn hw_reserve_u16(&self, len: usize, n: usize) -> Result<()> {
        self.check_workspace("reserve BF16 activation buffers", &[len, n, 2])?;
        self.check_workspace("reserve BF16 activation", &[len, 2])?;
        if len == 0 || n == 0 {
            return Ok(());
        }
        // Top up to `n` pooled buffers of this length. Allocating all `n`
        // on every forward (and dropping the extras on recycle) cost a fresh
        // zero-fill and page faults per forward.
        let have = self
            .pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .buffers_u16
            .get(&len)
            .map_or(0, Vec::len);
        for _ in have..n {
            let mut buf = rt_core::try_vec::<u16>(len)?;
            touch_pages(&mut buf);
            self.hw_recycle_u16(buf);
        }
        Ok(())
    }

    fn add_convert_s(&self, dt: f64) {
        if let Ok(mut s) = self.state.lock() {
            s.convert_s += dt;
        }
    }

    fn add_category(&self, key: &str, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        if let Ok(mut s) = self.state.lock() {
            *s.category_s.entry(key.into()).or_default() += dt;
        }
    }

    pub fn stats_json(&self) -> Value {
        let (conv_arena, conv_scratch) = self
            .conv_workspace
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .idle
            .iter()
            .map(|w| w.bytes())
            .fold((0, 0), |(a, s), (wa, ws)| (a + wa, s + ws));
        let execute_scratch_bytes = self
            .exec
            .idle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|w| w.scratch.bytes)
            .sum::<usize>();
        let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        serde_json::json!({
            "conv_packed_weight_bytes": self.conv_weight_bytes.load(Ordering::Relaxed),
            "conv_arena_bytes": conv_arena,
            "conv_scratch_bytes": conv_scratch,
            "execute_scratch_bytes": execute_scratch_bytes,
            "cached_plans": s.plans.len(),
            "plan_capacity": s.capacity,
            "plan_creations": s.creations,
            "capacity_evictions": s.capacity_evictions,
            "cached_weight_layouts": s.layouts.len(),
            "primitive_setup_s": s.setup_s,
            "primitive_exec_s": s.exec_s,
            "bf16_convert_s": s.convert_s,
            "category_exec_s": s.category_s,
            "executions": s.executions,
            "packed_weight_bytes": s.weight_bytes,
            "retained_output_buffer_bytes": pool.buffer_bytes,
            "memory_budget_bytes": self.memory_budget,
            "retained_bytes": self.retained.total(),
            "plan_bytes": self.retained.plans.load(Ordering::Relaxed),
            "conv_plan_bytes": self.retained.conv_plans.load(Ordering::Relaxed),
            "budget_evictions": s.evictions
                + self.retained.conv_evictions.load(Ordering::Relaxed) as u64,
            "budget_evicted_bytes": s.evicted_bytes
                + self.retained.conv_evicted.load(Ordering::Relaxed) as u64,
            "budget_trims": self.retained.trims.load(Ordering::Relaxed),
            "active_pool_shape": pool.active_shape,
            "implementations": s.plans.values().map(|p| p.name.clone()).collect::<Vec<_>>(),
            "cpu_isa": self.cpu_isa,
            "exec_slots": self.exec_slots,
            "session": "bounded execute pool (stream+scratch per slot); default 1 is serialized; parallel_for is process-global",
        })
    }
}

impl Hardware for OnednnCpu {
    type Packed = PackedWeight;

    fn info(&self) -> &str {
        &self.info
    }
    fn threads(&self) -> usize {
        self.threads as usize
    }
    fn cpu_isa(&self) -> u64 {
        self.cpu_isa
    }
    fn residual_rows(&self, tokens: usize) -> usize {
        self.strategy.residual_rows(tokens)
    }
    fn residual_row_tile(&self, live_rows: usize) -> usize {
        self.strategy.residual_row_tile(live_rows)
    }
    fn validate_workspace(&self, bytes: usize) -> Result<()> {
        self.check_workspace("model activations", &[bytes])
            .map(|_| ())
    }
    fn begin_shape(&self, b: usize, s: usize) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if pool.active_shape != Some((b, s)) {
            pool.buffers.clear();
            pool.buffers_u16.clear();
            pool.buffer_bytes = 0;
            pool.active_shape = Some((b, s));
        }
    }
    fn buffer(&self, len: usize) -> Result<Vec<f32>> {
        self.hw_buffer(len)
    }
    fn recycle(&self, buf: Vec<f32>) {
        self.hw_recycle(buf);
    }
    fn reserve(&self, len: usize, n: usize) -> Result<()> {
        self.hw_reserve(len, n)
    }
    fn buffer_u16(&self, len: usize) -> Result<Vec<u16>> {
        self.hw_buffer_u16(len)
    }
    fn recycle_u16(&self, buf: Vec<u16>) {
        self.hw_recycle_u16(buf);
    }
    fn reserve_u16(&self, len: usize, n: usize) -> Result<()> {
        self.hw_reserve_u16(len, n)
    }
    fn pack(&self, n: usize, k: usize, src: &[f32]) -> Result<Self::Packed> {
        OnednnCpu::pack(self, n, k, src)
    }
    fn restore_packed(&self, n: usize, k: usize, bytes: &[u8]) -> Result<Self::Packed> {
        self.pack_from_cache(n, k, bytes)
    }
    fn restore_packed_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<Self::Packed> {
        let mut tag = empty_layout_tag();
        if layout.len() == LAYOUT_TAG_LEN {
            tag.copy_from_slice(layout);
        }
        self.pack_from_cache_dt(n, k, bytes, MatDt::F32, Some(&tag))
    }
    fn unpack_into(&self, packed: &Self::Packed, dst: &mut [u8]) -> Result<()> {
        let src = packed.as_bytes();
        if dst.len() < src.len() {
            return Err(rt_core::Error::Shape("packed unpack dest".into()));
        }
        dst[..src.len()].copy_from_slice(src);
        Ok(())
    }
    fn linear(&self, x: &[f32], w: &Self::Packed, out: &mut [f32], sum: bool) -> Result<()> {
        self.linear_inner(x, w, out, sum)
    }
    fn gated_linear(&self, x: &[f32], w: &Self::Packed, gate: &[f32]) -> Result<Vec<f32>> {
        self.gated_inner(x, w, gate)
    }
    fn mm(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        self.mm_inner(g, m, n, k, ab, ar, bb, br, trans, a, b)
    }
    fn layernorm(
        &self,
        m: usize,
        n: usize,
        x: &[f32],
        scale: &[f32],
        shift: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.op_inner(1, m, n, x, Some(scale), Some(shift), eps)
    }
    fn softmax(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(2, m, n, x, None, None, 0.)
    }
    fn gelu_erf(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(3, m, n, x, None, None, 0.)
    }
    fn relu(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(5, m, n, x, None, None, 0.)
    }
    fn mul(&self, m: usize, n: usize, x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(4, m, n, x, Some(y), None, 0.)
    }
    fn rmsnorm(
        &self,
        m: usize,
        n: usize,
        x: &[f32],
        scale: Option<&[f32]>,
        eps: f32,
    ) -> Result<Vec<f32>> {
        if x.len() != m.saturating_mul(n) {
            return Err(rt_core::Error::Shape("rmsnorm shape".into()));
        }
        let mut y = self.hw_buffer(x.len())?;
        rmsnorm_nthreads(x, n, scale, eps, &mut y, self.threads.max(1) as usize)?;
        Ok(y)
    }
    fn gelu_tanh(&self, m: usize, n: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.op_inner(6, m, n, x, None, None, 0.)
    }
    fn pack_bf16(&self, n: usize, k: usize, src: &[u16]) -> Result<Self::Packed> {
        OnednnCpu::pack_bf16(self, n, k, src)
    }
    fn restore_packed_bf16(&self, n: usize, k: usize, bytes: &[u8]) -> Result<Self::Packed> {
        self.pack_from_cache_bf16(n, k, bytes)
    }
    fn restore_packed_bf16_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<Self::Packed> {
        let mut tag = empty_layout_tag();
        if layout.len() == LAYOUT_TAG_LEN {
            tag.copy_from_slice(layout);
        }
        self.pack_from_cache_dt(n, k, bytes, MatDt::Bf16, Some(&tag))
    }
    fn pack_bf16_f32(&self, n: usize, k: usize, src: &[u16]) -> Result<Self::Packed> {
        OnednnCpu::pack_bf16_f32(self, n, k, src)
    }
    fn restore_packed_bf16_f32_checked(
        &self,
        n: usize,
        k: usize,
        bytes: &[u8],
        layout: &[u8],
    ) -> Result<Self::Packed> {
        let mut tag = empty_layout_tag();
        if layout.len() == LAYOUT_TAG_LEN {
            tag.copy_from_slice(layout);
        }
        self.pack_from_cache_dt(n, k, bytes, MatDt::F32Bf16, Some(&tag))
    }
    fn attention_gqa_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        attention_gqa_f32(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, scale, padding,
        )
    }
    fn linear_bf16(&self, x: &[f32], w: &Self::Packed, out: &mut [f32], sum: bool) -> Result<()> {
        self.linear_bf16_inner(x, w, out, sum)
    }
    fn linear_bf16_src(
        &self,
        x: &[u16],
        w: &Self::Packed,
        out: &mut [f32],
        sum: bool,
    ) -> Result<()> {
        self.linear_bf16_src_inner(x, w, out, sum)
    }
    fn mm_bf16(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>> {
        self.mm_bf16_inner(g, m, n, k, ab, ar, bb, br, trans, a, b)
    }
    fn mm_bf16_src(
        &self,
        g: usize,
        m: usize,
        n: usize,
        k: usize,
        ab: usize,
        ar: usize,
        bb: usize,
        br: usize,
        trans: bool,
        a: &[u16],
        b: &[u16],
    ) -> Result<Vec<f32>> {
        self.mm_bf16_src_inner(g, m, n, k, ab, ar, bb, br, trans, a, b)
    }
    fn gated_linear_tanh(&self, x: &[f32], w: &Self::Packed, gate: &[f32]) -> Result<Vec<f32>> {
        self.gated_tanh_inner(x, w, gate)
    }
    fn gated_linear_tanh_src(&self, x: &[u16], w: &Self::Packed, gate: &[f32]) -> Result<Vec<f32>> {
        self.gated_tanh_src_inner(x, w, gate)
    }
    fn attention_gqa(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        attention_gqa_bf16(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, scale, padding,
        )
    }
    fn attention_gqa_src(
        &self,
        q: &[u16],
        k: &[u16],
        v: &[u16],
        b: usize,
        s: usize,
        nq: usize,
        nkv: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        scale: f32,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        if (scale - 1.0).abs() > 0.0 {
            let up = |x: &[u16]| -> Vec<f32> {
                x.iter()
                    .map(|&b| f32::from_bits((b as u32) << 16))
                    .collect()
            };
            return attention_gqa_bf16(
                self,
                &up(q),
                &up(k),
                &up(v),
                b,
                s,
                nq,
                nkv,
                d,
                tile,
                local,
                radius,
                scale,
                padding,
            );
        }
        attention_gqa_from_bf16(
            self, q, k, v, b, s, nq, nkv, d, tile, local, radius, padding,
        )
    }
    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        s: usize,
        nh: usize,
        d: usize,
        tile: usize,
        local: bool,
        radius: usize,
        padding: &[Vec<usize>],
    ) -> Result<Vec<f32>> {
        attention_fp32(self, q, k, v, b, s, nh, d, tile, local, radius, padding)
    }
    fn add_category_s(&self, key: &str, dt: f64) {
        self.add_category(key, dt);
    }
    fn stats(&self) -> Value {
        self.stats_json()
    }
}

/// Vela FP32 attention: same QK/PV GEMMs and oneDNN softmax as the trait
/// default, with threaded mask (`+= f32::MIN`) and output scatter.
fn attention_fp32(
    hw: &OnednnCpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nh: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    rt_backend::validate_attention(q, k, v, b, s, nh, nh, d, tile, radius, padding)?;
    let g = b * nh;
    let threads = hw.threads.max(1) as usize;
    let mut out = hw.buffer(rt_core::checked_size(
        "activation elements",
        &[b, s, nh, d],
    )?)?;
    let tile = tile.min(s).max(1);
    for start in (0..s).step_by(tile) {
        let n = tile.min(s - start);
        let ks = if local {
            start.saturating_sub(radius)
        } else {
            0
        };
        let ke = if local {
            (start + n + radius).min(s)
        } else {
            s
        };
        let nk = ke - ks;
        let mut scores = hw.mm_inner(
            g,
            n,
            nk,
            d,
            s * d,
            d,
            s * d,
            d,
            true,
            &q[start * d..],
            &k[ks * d..],
        )?;
        let scores_p = scores.as_mut_ptr() as usize;
        let rows = g * n;
        for_ranges(threads, rows, 4, |rlo, rhi| {
            for row_i in rlo..rhi {
                let gi = row_i / n;
                let qi = row_i % n;
                let excluded = &padding[gi / nh];
                if !local && excluded.is_empty() {
                    continue;
                }
                let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                let lo = if local {
                    (start + qi).saturating_sub(radius).saturating_sub(ks)
                } else {
                    0
                };
                let hi = if local {
                    (start + qi + radius + 1).min(ke) - ks
                } else {
                    nk
                };
                for score in &mut row[..lo] {
                    *score += f32::MIN;
                }
                for score in &mut row[hi..] {
                    *score += f32::MIN;
                }
                for &pos in excluded {
                    if pos >= ks + lo && pos < ks + hi {
                        row[pos - ks] += f32::MIN;
                    }
                }
            }
        });
        // Keep oneDNN softmax: it is already a small slice of Vela time and
        // stays bit-identical to the previous default_attention path.
        let probs = hw.softmax(g * n, nk, &scores)?;
        hw.recycle(scores);
        let y = hw.mm_inner(
            g,
            n,
            d,
            nk,
            n * nk,
            nk,
            s * d,
            d,
            false,
            &probs,
            &v[ks * d..],
        )?;
        hw.recycle(probs);
        {
            let y_p = y.as_ptr() as usize;
            let out_p = out.as_mut_ptr() as usize;
            let rows = b * n;
            for_ranges(threads, rows, 8, |lo, hi| {
                for idx in lo..hi {
                    let bi = idx / n;
                    let qi = idx % n;
                    for head in 0..nh {
                        let src = ((bi * nh + head) * n + qi) * d;
                        let dst = (bi * s + start + qi) * (nh * d) + head * d;
                        let src = unsafe { row_ref(y_p as *const f32, src / d, d) };
                        let dst = unsafe { row_mut(out_p as *mut f32, dst / d, d) };
                        dst.copy_from_slice(src);
                    }
                }
            });
        }
        hw.recycle(y);
    }
    Ok(out)
}

fn attention_gqa_bf16(
    hw: &OnednnCpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    scale: f32,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    rt_backend::validate_attention(q, k, v, b, s, nq, nkv, d, tile, radius, padding)?;
    if nq == 0 || nkv == 0 || !nq.is_multiple_of(nkv) {
        return Err(rt_core::Error::Shape("GQA head grouping".into()));
    }
    let threads = hw.threads.max(1) as usize;
    let q_scale = if (scale - 1.0).abs() > 0.0 {
        let mut qs = hw.buffer(q.len())?;
        let q_p = q.as_ptr() as usize;
        let qs_p = qs.as_mut_ptr() as usize;
        let n = q.len();
        for_ranges(threads, n, 4096, |lo, hi| {
            let q = unsafe { rows_ref(q_p as *const f32, lo, hi, 1) };
            let qs = unsafe { rows_mut(qs_p as *mut f32, lo, hi, 1) };
            for (dst, &src) in qs.iter_mut().zip(q.iter()) {
                *dst = src * scale;
            }
        });
        qs
    } else {
        Vec::new()
    };
    let q_ref: &[f32] = if q_scale.is_empty() { q } else { &q_scale };
    let mut q_bf = hw.buffer_u16(q_ref.len())?;
    let mut k_bf = hw.buffer_u16(k.len())?;
    let mut v_bf = hw.buffer_u16(v.len())?;
    let t_conv = Instant::now();
    f32_slice_to_bf16_nthreads(q_ref, &mut q_bf, threads)?;
    f32_slice_to_bf16_nthreads(k, &mut k_bf, threads)?;
    f32_slice_to_bf16_nthreads(v, &mut v_bf, threads)?;
    hw.add_convert_s(t_conv.elapsed().as_secs_f64());
    if !q_scale.is_empty() {
        hw.recycle(q_scale);
    }
    let out = attention_gqa_from_bf16(
        hw, &q_bf, &k_bf, &v_bf, b, s, nq, nkv, d, tile, local, radius, padding,
    );
    hw.recycle_u16(q_bf);
    hw.recycle_u16(k_bf);
    hw.recycle_u16(v_bf);
    out
}

/// Main's tiled QK → softmax → PV (oneshot oneDNN, f32 P, AMX short-M PV pad).
/// Preconverted BF16 Q/K/V skip the f32→bf16 convert.
fn attention_gqa_from_bf16(
    hw: &OnednnCpu,
    q_bf: &[u16],
    k_bf: &[u16],
    v_bf: &[u16],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    rt_backend::validate_attention(q_bf, k_bf, v_bf, b, s, nq, nkv, d, tile, radius, padding)?;
    if nq == 0 || nkv == 0 || !nq.is_multiple_of(nkv) {
        return Err(rt_core::Error::Shape("GQA head grouping".into()));
    }
    let groups = nq / nkv;
    let threads = hw.threads.max(1) as usize;
    let g_kv = b * nkv;
    let live_rows = b * s;
    let mut out = hw.buffer(rt_core::checked_size(
        "activation elements",
        &[hw.strategy.residual_rows(live_rows), nq, d],
    )?)?;
    // Only live queries/keys enter attention. o_proj consumes this zero tail
    // directly, avoiding a per-layer allocation and live-to-padded copy.
    out[live_rows * nq * d..].fill(0.0);
    // Clamp to the live sequence (and sliding window). Do not round the *query*
    // tile up to a constant 256 — that made B1×128 pay for a 256-row shape.
    // AMX folds a starving last window from shape only (777 → 256+256+265).
    // B1/B4×300 keep step_by (rem=44 pad waste 8/88). AVX-512 is always
    // step_by(tile). Thread count does not change the split.
    let tile = crate::clamp_attn_tile(tile, s, local, radius, b, nq);
    let q_ranges = hw.strategy.attn_query_ranges(s, tile, b, nq, nkv, local);
    let q_max = q_ranges.iter().map(|&(_, n)| n).max().unwrap_or(1);
    let mut q_tile = if q_max < s {
        hw.buffer_u16(rt_core::checked_size(
            "activation elements",
            &[g_kv, groups, q_max, d],
        )?)?
    } else {
        Vec::new()
    };
    for (start, n) in q_ranges {
        let ks = if local {
            start.saturating_sub(radius)
        } else {
            0
        };
        let ke = if local {
            (start + n + radius).min(s)
        } else {
            s
        };
        let nk = ke - ks;
        let m = groups * n;
        let (q_ptr, q_ab): (&[u16], usize) = if start == 0 && n == s {
            (q_bf, groups * s * d)
        } else {
            gather_q_tile(q_bf, &mut q_tile, b, nkv, groups, s, d, start, n, threads);
            (&q_tile[..g_kv * m * d], m * d)
        };
        let t_qk = Instant::now();
        let mut scores = hw.mm_bf16_src_inner(
            g_kv,
            m,
            nk,
            d,
            q_ab,
            d,
            s * d,
            d,
            true,
            q_ptr,
            &k_bf[ks * d..],
        )?;
        hw.add_category("attn_qk", t_qk.elapsed().as_secs_f64());
        let blocked_by_batch: Vec<Vec<usize>> = padding
            .iter()
            .map(|pad| {
                pad.iter()
                    .copied()
                    // Filter before subtracting: then_some eagerly evaluates
                    // p - ks and underflows in debug builds for earlier keys.
                    .filter(|&p| p >= ks && p < ke)
                    .map(|p| p - ks)
                    .collect()
            })
            .collect();
        let t_sm = Instant::now();
        let probs = if local {
            let scores_p = scores.as_mut_ptr() as usize;
            let rows = g_kv * m;
            for_ranges(threads, rows, 4, |rlo, rhi| {
                for row_i in rlo..rhi {
                    let gi = row_i / m;
                    let rest = row_i % m;
                    let qi = rest % n;
                    let bi = gi / nkv;
                    let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                    let qpos = start + qi;
                    let lo = qpos.saturating_sub(radius).saturating_sub(ks);
                    let hi = (qpos + radius + 1).min(ke) - ks;
                    softmax_window(row, lo, hi, &blocked_by_batch[bi]);
                }
            });
            scores
        } else {
            if blocked_by_batch.iter().any(|p| !p.is_empty()) {
                let scores_p = scores.as_mut_ptr() as usize;
                let rows = g_kv * m;
                for_ranges(threads, rows, 4, |rlo, rhi| {
                    for row_i in rlo..rhi {
                        let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                        let bi = (row_i / m) / nkv;
                        for &pos in &blocked_by_batch[bi] {
                            row[pos] = f32::NEG_INFINITY;
                        }
                    }
                });
            }
            if g_kv * m >= 512 && nk >= 256 {
                let probs = hw.softmax(g_kv * m, nk, &scores)?;
                hw.recycle(scores);
                probs
            } else {
                softmax_rows_nthreads(&mut scores, nk, threads)?;
                scores
            }
        };
        hw.add_category("attn_softmax", t_sm.elapsed().as_secs_f64());
        let t_pv = Instant::now();
        let y = if hw.strategy.attn_pad_m(m) == m {
            let mut y =
                hw.hw_buffer(rt_core::checked_size("activation elements", &[g_kv, m, d])?)?;
            hw.run_ex(
                Key {
                    kind: 0,
                    g: g_kv,
                    m,
                    n: d,
                    k: nk,
                    ab: m * nk,
                    ar: nk,
                    bb: s * d,
                    br: d,
                    trans: false,
                    param: 0,
                    sum: false,
                    weight: 0,
                    dt: MatDt::Bf16,
                },
                &probs,
                None,
                None,
                Some(&v_bf[ks * d..]),
                None,
                &mut y,
                None,
            )?;
            y
        } else {
            hw.attn_pv_inner(g_kv, m, d, nk, &probs, &v_bf[ks * d..], s * d, d)?
        };
        hw.add_category("attn_pv", t_pv.elapsed().as_secs_f64());
        hw.recycle(probs);
        {
            let y_p = y.as_ptr() as usize;
            let out_p = out.as_mut_ptr() as usize;
            let rows = b * n;
            for_ranges(threads, rows, 8, |lo, hi| {
                for idx in lo..hi {
                    let bi = idx / n;
                    let qi = idx % n;
                    for kvh in 0..nkv {
                        for grp in 0..groups {
                            let qh = kvh * groups + grp;
                            let src = (((bi * nkv + kvh) * groups + grp) * n + qi) * d;
                            let dst = (bi * s + start + qi) * (nq * d) + qh * d;
                            let src = unsafe { row_ref(y_p as *const f32, src / d, d) };
                            let dst = unsafe { row_mut(out_p as *mut f32, dst / d, d) };
                            dst.copy_from_slice(src);
                        }
                    }
                }
            });
        }
        hw.recycle(y);
    }
    hw.recycle_u16(q_tile);
    Ok(out)
}

/// FP32 GQA attention: per KV head, the `groups` query heads that share it
/// form one `groups * n`-row f32 matmul against K and V (no `repeat_kv`).
/// Output is exactly `[b * s, nq * d]`, token-major, with no residual pad.
fn attention_gqa_f32(
    hw: &OnednnCpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    s: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    tile: usize,
    local: bool,
    radius: usize,
    scale: f32,
    padding: &[Vec<usize>],
) -> Result<Vec<f32>> {
    rt_backend::validate_attention(q, k, v, b, s, nq, nkv, d, tile, radius, padding)?;
    let groups = nq / nkv;
    let threads = hw.threads.max(1) as usize;
    let g_kv = b * nkv;
    let q_scaled = if (scale - 1.0).abs() > 0.0 {
        let mut qs = hw.buffer(q.len())?;
        for (dst, &src) in qs.iter_mut().zip(q) {
            *dst = src * scale;
        }
        qs
    } else {
        Vec::new()
    };
    let q: &[f32] = if q_scaled.is_empty() { q } else { &q_scaled };
    let mut out = hw.buffer(rt_core::checked_size(
        "activation elements",
        &[b * s, nq, d],
    )?)?;
    let tile = crate::clamp_attn_tile(tile, s, local, radius, b, nq);
    // f32 runs on AVX-512 FMA on every host, so AMX tail folding never applies.
    let q_ranges = crate::IsaStrategy::Avx512.attn_query_ranges(s, tile, b, nq, nkv, local);
    let q_max = q_ranges.iter().map(|&(_, n)| n).max().unwrap_or(1);
    let mut q_tile = if q_max < s {
        hw.buffer(rt_core::checked_size(
            "activation elements",
            &[g_kv, groups, q_max, d],
        )?)?
    } else {
        Vec::new()
    };
    for (start, n) in q_ranges {
        let ks = if local {
            start.saturating_sub(radius)
        } else {
            0
        };
        let ke = if local {
            (start + n + radius).min(s)
        } else {
            s
        };
        let nk = ke - ks;
        let m = groups * n;
        let (q_ptr, q_ab): (&[f32], usize) = if start == 0 && n == s {
            (q, groups * s * d)
        } else {
            gather_q_tile(q, &mut q_tile, b, nkv, groups, s, d, start, n, threads);
            (&q_tile[..g_kv * m * d], m * d)
        };
        let t_qk = Instant::now();
        let mut scores =
            hw.mm_inner(g_kv, m, nk, d, q_ab, d, s * d, d, true, q_ptr, &k[ks * d..])?;
        hw.add_category("attn_qk", t_qk.elapsed().as_secs_f64());
        let blocked_by_batch: Vec<Vec<usize>> = padding
            .iter()
            .map(|pad| {
                pad.iter()
                    .copied()
                    .filter(|&p| p >= ks && p < ke)
                    .map(|p| p - ks)
                    .collect()
            })
            .collect();
        let t_sm = Instant::now();
        let scores_p = scores.as_mut_ptr() as usize;
        let rows = g_kv * m;
        if local {
            for_ranges(threads, rows, 4, |rlo, rhi| {
                for row_i in rlo..rhi {
                    let gi = row_i / m;
                    let qi = (row_i % m) % n;
                    let bi = gi / nkv;
                    let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                    let qpos = start + qi;
                    let lo = qpos.saturating_sub(radius).saturating_sub(ks);
                    let hi = (qpos + radius + 1).min(ke) - ks;
                    softmax_window(row, lo, hi, &blocked_by_batch[bi]);
                }
            });
        } else {
            let blocked = blocked_by_batch.iter().any(|p| !p.is_empty());
            for_ranges(threads, rows, 4, |rlo, rhi| {
                for row_i in rlo..rhi {
                    let row = unsafe { row_mut(scores_p as *mut f32, row_i, nk) };
                    if blocked {
                        let bi = (row_i / m) / nkv;
                        softmax_window(row, 0, nk, &blocked_by_batch[bi]);
                    } else {
                        rt_kernels::softmax_inplace(row);
                    }
                }
            });
        }
        hw.add_category("attn_softmax", t_sm.elapsed().as_secs_f64());
        let t_pv = Instant::now();
        let y = hw.mm_inner(
            g_kv,
            m,
            d,
            nk,
            m * nk,
            nk,
            s * d,
            d,
            false,
            &scores,
            &v[ks * d..],
        )?;
        hw.add_category("attn_pv", t_pv.elapsed().as_secs_f64());
        hw.recycle(scores);
        {
            let y_p = y.as_ptr() as usize;
            let out_p = out.as_mut_ptr() as usize;
            for_ranges(threads, b * n, 8, |lo, hi| {
                for idx in lo..hi {
                    let bi = idx / n;
                    let qi = idx % n;
                    for kvh in 0..nkv {
                        for grp in 0..groups {
                            let qh = kvh * groups + grp;
                            let src = (((bi * nkv + kvh) * groups + grp) * n + qi) * d;
                            let dst = (bi * s + start + qi) * (nq * d) + qh * d;
                            let src = unsafe { row_ref(y_p as *const f32, src / d, d) };
                            let dst = unsafe { row_mut(out_p as *mut f32, dst / d, d) };
                            dst.copy_from_slice(src);
                        }
                    }
                }
            });
        }
        hw.recycle(y);
    }
    hw.recycle(q_tile);
    if !q_scaled.is_empty() {
        hw.recycle(q_scaled);
    }
    Ok(out)
}

fn gather_q_tile<T: Copy>(
    q: &[T],
    dst: &mut [T],
    b: usize,
    nkv: usize,
    groups: usize,
    s: usize,
    d: usize,
    start: usize,
    n: usize,
    nthreads: usize,
) {
    let nq = nkv * groups;
    let rows = b * nkv * groups * n;
    let q_p = q.as_ptr() as usize;
    let d_p = dst.as_mut_ptr() as usize;
    for_ranges(nthreads, rows, 16, |lo, hi| {
        for idx in lo..hi {
            let qi = idx % n;
            let rest = idx / n;
            let grp = rest % groups;
            let rest = rest / groups;
            let kvh = rest % nkv;
            let bi = rest / nkv;
            let qh = kvh * groups + grp;
            let src = ((bi * nq + qh) * s + start + qi) * d;
            let off = (((bi * nkv + kvh) * groups + grp) * n + qi) * d;
            let src = unsafe { row_ref(q_p as *const T, src / d, d) };
            let dst = unsafe { row_mut(d_p as *mut T, off / d, d) };
            dst.copy_from_slice(src);
        }
    });
}
