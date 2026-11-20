//! The oneDNN CPU backend the model graphs run on: same numerics as the
//! retired C++ bridge.
//!
//! Strict fpmath/accumulation, user scratchpads, primitive cache capacity 0,
//! packed weight reorder, residual sum post-op, kind-7 GeGLU
//! (`MatMul + GELU_ERF + BINARY_MUL`).

use crate::cpu::kernels::{
    f32_slice_to_bf16_nthreads, for_ranges, softmax_rows_nthreads, softmax_window,
};
use crate::cpu::onednn::api::Api;
use crate::cpu::onednn::cache::LayoutTag;
use crate::cpu::onednn::options::{CpuIsaCap, SessionOptions, DEFAULT_WORKSPACE_LIMIT};
use crate::cpu::threads::{compute_lock, ThreadBudget};
use crate::{check_shape, Result, Result as AResult, ResultContext as Context};
use onednn_sys::*;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

mod attention;
mod buffers;
mod ops;
mod pack;
mod plan;

pub(crate) use attention::Attention;
pub(crate) use ops::Gemm;
pub(crate) use plan::{Operand, Operands};

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

/// A 64-byte-aligned heap buffer of at least 64 bytes, every byte of it
/// initialized, that oneDNN reads and writes through [`Aligned::ptr`].
pub struct Aligned {
    ptr: NonNull<u8>,
    layout: std::alloc::Layout,
    bytes: usize,
}

impl Aligned {
    /// A zeroed buffer of `bytes` (at least 64).
    pub fn new(bytes: usize) -> AResult<Self> {
        let n = bytes.max(64);
        let layout = std::alloc::Layout::from_size_align(n, 64)?;
        // SAFETY: `layout` has a nonzero size (at least 64).
        let ptr = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .context("aligned allocation")?;
        Ok(Self {
            ptr,
            layout,
            bytes: n,
        })
    }
    /// The buffer's length in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// The start of the buffer, for oneDNN to read or write. A write through
    /// it must not overlap a live [`Aligned::as_bytes`].
    pub fn ptr(&self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the allocation holds `self.bytes` bytes, all initialized
        // when it was made, and lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.bytes) }
    }
    /// The buffer as f32s, its length rounded down to whole ones.
    pub(crate) fn as_f32_mut(&mut self) -> &mut [f32] {
        // SAFETY: the allocation holds `self.bytes` initialized bytes, 64-byte
        // aligned, every bit pattern is an f32, and `self` is borrowed
        // mutably for as long as the slice.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.bytes / 4) }
    }
    /// The buffer as BF16 bit patterns, its length rounded down to whole
    /// ones.
    pub(crate) fn as_u16_mut(&mut self) -> &mut [u16] {
        // SAFETY: as for `as_f32_mut`, with two-byte elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast(), self.bytes / 2) }
    }
    /// Copy `src` into a 64-byte-aligned buffer without zero-filling first.
    pub fn copy_from(src: &[u8]) -> AResult<Self> {
        let n = src.len().max(64);
        let layout = std::alloc::Layout::from_size_align(n, 64)?;
        // SAFETY: `layout` has a nonzero size (at least 64).
        let ptr =
            NonNull::new(unsafe { std::alloc::alloc(layout) }).context("aligned allocation")?;
        // SAFETY: the new allocation holds `n >= src.len()` bytes and cannot
        // overlap `src`; the copy and the zero tail initialize all of them.
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
        // SAFETY: allocated with `self.layout` and freed only here.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// SAFETY: an `Aligned` owns its allocation as a `Box<[u8]>` would. Sharing
// it shares `as_bytes` reads; a write through `ptr` is a caller's unsafe
// call, which must keep it from overlapping them.
unsafe impl Send for Aligned {}
// SAFETY: as for `Send`.
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
        // SAFETY: each handle, where not null, was created for this layout
        // alone and is destroyed only here; the weights packed with it keep
        // only their bytes.
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

// SAFETY: oneDNN descriptors and primitives are not tied to the thread that
// made them, and are immutable once made: a primitive may run on several
// threads at once given separate memories, which each reorder binds.
unsafe impl Send for WeightLayout {}
// SAFETY: as for `Send`.
unsafe impl Sync for WeightLayout {}

/// A weight in the blocked layout this CPU's matmuls read, with the layout
/// it was packed for.
pub struct PackedWeight {
    layout: Arc<WeightLayout>,
    data: Aligned,
    n: usize,
    k: usize,
    dt: MatDt,
    tag: LayoutTag,
}

impl std::fmt::Debug for PackedWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedWeight")
            .field("n", &self.n)
            .field("k", &self.k)
            .field("dt", &self.dt)
            .field("bytes", &self.layout.bytes)
            .finish_non_exhaustive()
    }
}

impl PackedWeight {
    pub(crate) fn n(&self) -> usize {
        self.n
    }
    pub(crate) fn k(&self) -> usize {
        self.k
    }
    pub(crate) fn bytes(&self) -> usize {
        self.layout.bytes
    }
    /// ISA-dependent blocked-layout fingerprint, stored in the weight cache.
    pub(crate) fn layout_tag(&self) -> LayoutTag {
        self.tag
    }
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
        // SAFETY: each handle, where not null, was created for this plan
        // alone and is destroyed only here, once nothing holds the plan.
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

// SAFETY: oneDNN handles are not tied to the thread that made them.
unsafe impl Send for Plan {}
// SAFETY: shared only to keep `prim` alive through an execute that runs
// outside the state lock; a primitive may run on several threads at once,
// and plan-owned memories are bound only under that lock.
unsafe impl Sync for Plan {}
// SAFETY: a stream is not tied to the thread that made it, and the exec pool
// hands each workspace to one caller at a time.
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
            // SAFETY: `engine` and `threadpool` are live (the backend's own),
            // and oneDNN writes the new handle to a live local.
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
            if crate::cpu::threads::in_parallel() {
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
/// the process-global pool in `crate::cpu::threads` (one `parallel_for` at a
/// time).
pub struct OnednnCpu {
    api: Arc<Api>,
    engine: dnnl_engine_t,
    exec: ExecPool,
    exec_slots: usize,
    state: Mutex<State>,
    pool: Mutex<Pool>,
    pub(crate) conv_workspace: crate::cpu::onednn::conv::Workspaces,
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
    strategy: crate::cpu::onednn::IsaStrategy,
}

// SAFETY: the engine may be used from any thread; every other handle lives
// in a plan, layout or workspace whose sharing is argued at its type, or
// behind the state, pool and exec locks.
unsafe impl Send for OnednnCpu {}
// SAFETY: as for `Send`.
unsafe impl Sync for OnednnCpu {}

/// An owned `threadpool_iface` reporting at most `n` threads to oneDNN.
struct CappedThreadpool(*mut c_void);

impl CappedThreadpool {
    fn new(n: i32) -> AResult<Self> {
        // SAFETY: no preconditions; the result is owned here, or null.
        let p = unsafe { onednn_sys::sinteric_dnnl_threadpool_create_capped(n) };
        if p.is_null() {
            return Err(crate::Error::out_of_memory(16));
        }
        Ok(Self(p))
    }
}

impl Drop for CappedThreadpool {
    fn drop(&mut self) {
        // SAFETY: created by `new` (not null) and destroyed only here, after
        // `OnednnCpu::drop` has destroyed every stream that used it.
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
                // SAFETY: every workspace is idle (nothing borrows `self`),
                // and its stream is destroyed only here.
                unsafe {
                    if !ws.stream.is_null() {
                        (self.api.stream_destroy)(ws.stream);
                    }
                }
            }
        }
        // SAFETY: the plans, layouts and streams that used the engine are
        // gone (above), and it is destroyed only here.
        unsafe {
            if !self.engine.is_null() {
                (self.api.engine_destroy)(self.engine);
            }
        }
    }
}

/// Rewrites one element in every 4 KiB page of `buf`, and its last, so that
/// the pages are faulted in now rather than on the first forward.
fn touch_pages<T: Copy>(buf: &mut [T]) {
    let step = (4096 / std::mem::size_of::<T>().max(1)).max(1);
    let last = buf.len().checked_sub(1);
    for i in (0..buf.len()).step_by(step).chain(last) {
        let p: *mut T = &mut buf[i];
        // SAFETY: `p` comes from a live `&mut` to an initialized element.
        unsafe { p.write_volatile(p.read_volatile()) };
    }
}

fn fill_bf16(buf: &mut Aligned, src: &[f32], nthreads: i32) -> AResult<*mut c_void> {
    let need = crate::checked_size("BF16 conversion", &[src.len(), 2])?.max(64);
    if buf.bytes < need {
        *buf = Aligned::new(need)?;
    }
    f32_slice_to_bf16_nthreads(
        src,
        &mut buf.as_u16_mut()[..src.len()],
        nthreads.max(1) as usize,
    )?;
    Ok(buf.ptr())
}

fn fill(dims: &[i64]) -> [i64; 12] {
    let mut out = [0i64; 12];
    out[..dims.len()].copy_from_slice(dims);
    out
}

/// `BestAvailable` becomes `Avx512CoreBf16` on a host whose AMX tile state is
/// lost across preemption ([`crate::cpu::onednn::amx_health`]); explicit caps
/// are kept. Nothing is printed: [`crate::cpu::Runtime::amx_health`] reports
/// what the probe found.
fn resolve_isa_cap(cap: CpuIsaCap) -> CpuIsaCap {
    if cap != CpuIsaCap::BestAvailable || !crate::cpu::onednn::amx_probe_enabled() {
        return cap;
    }
    match crate::cpu::onednn::amx_health() {
        crate::cpu::onednn::AmxHealth::TileStateLoss { .. } => CpuIsaCap::Avx512CoreBf16,
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
            .map_err(|_| crate::Error::poisoned("backend"))?;
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
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            // SAFETY: malloc_trim only touches allocator-internal free lists.
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
        crate::checked_size(operation, dims)
    }

    #[cfg(test)]
    pub fn load(budget: &ThreadBudget) -> AResult<Arc<Self>> {
        Self::load_with(budget, SessionOptions::default())
    }

    pub fn load_with(budget: &ThreadBudget, opts: SessionOptions) -> AResult<Arc<Self>> {
        if budget.nthreads == 0
            || budget.nthreads > libc::CPU_SETSIZE as usize
            || opts.exec_slots == 0
            || !(0..=2).contains(&opts.verbose)
        {
            return Err(crate::Error::config(format!(
                "invalid session: threads {}, exec_slots {}, verbose {}",
                budget.nthreads, opts.exec_slots, opts.verbose
            )));
        }
        let limit = if opts.workspace_limit == 0 {
            DEFAULT_WORKSPACE_LIMIT
        } else {
            opts.workspace_limit
        };
        let slot_bytes = crate::checked_size("execute slots", &[opts.exec_slots, 192])?;
        if slot_bytes > limit {
            return Err(crate::Error::workspace_limit(
                "execute slots",
                slot_bytes,
                limit,
            ));
        }
        budget.install_library_workers()?;
        let api = Arc::new(Api::init(
            resolve_isa_cap(opts.max_cpu_isa),
            opts.verbose,
            budget.nthreads_i32(),
        )?);
        let mut engine = ptr::null_mut();
        Api::check(
            // SAFETY: oneDNN writes the new handle to a live local.
            unsafe { (api.engine_create)(&mut engine, dnnl_cpu, 0) },
            "engine",
        )?;
        let threads = if opts.threads == 0 {
            budget.nthreads
        } else {
            opts.threads.min(budget.nthreads)
        };
        let threads_i32 = i32::try_from(threads)
            .map_err(|_| crate::Error::config(format!("invalid threads {threads}")))?;
        let capped_threadpool = if threads < budget.nthreads {
            Some(CappedThreadpool::new(threads_i32)?)
        } else {
            None
        };
        let threadpool = capped_threadpool.as_ref().map_or(api.threadpool, |tp| tp.0);
        let exec_slots = opts.exec_slots.max(1);
        let mut idle = Vec::new();
        idle.try_reserve_exact(exec_slots)
            .map_err(|_| crate::Error::out_of_memory(slot_bytes))?;
        for _ in 0..exec_slots {
            idle.push(ExecWorkspace::new(&api, engine, threadpool)?);
        }
        // SAFETY: no preconditions once the library is loaded (`Api::init`).
        let cpu_isa = unsafe { (api.get_effective_cpu_isa)() as u32 as u64 };
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
            strategy: crate::cpu::onednn::IsaStrategy::from_cpu_isa(cpu_isa),
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
        }))
    }

    /// Shape policy selected once from the effective (possibly pinned) ISA.
    #[cfg(test)]
    pub fn strategy(&self) -> crate::cpu::onednn::IsaStrategy {
        self.strategy
    }

    fn add_convert_s(&self, dt: f64) {
        if let Ok(mut s) = self.state.lock() {
            s.convert_s += dt;
        }
    }

    pub(crate) fn add_category(&self, key: &str, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        if let Ok(mut s) = self.state.lock() {
            *s.category_s.entry(key.into()).or_default() += dt;
        }
    }

    pub(crate) fn stats(&self) -> crate::cpu::runtime::BackendStats {
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
        crate::cpu::runtime::BackendStats {
            isa: crate::cpu::runtime::Isa::from_bits(self.cpu_isa)
                .unwrap_or(crate::cpu::runtime::Isa::Avx2),
            exec_slots: self.exec_slots,
            packed_weight_bytes: s.weight_bytes,
            conv_packed_weight_bytes: self.conv_weight_bytes.load(Ordering::Relaxed),
            retained_bytes: self.retained.total(),
            memory_budget_bytes: self.memory_budget,
            plan_bytes: self.retained.plans.load(Ordering::Relaxed),
            conv_plan_bytes: self.retained.conv_plans.load(Ordering::Relaxed),
            conv_arena_bytes: conv_arena,
            conv_scratch_bytes: conv_scratch,
            execute_scratch_bytes,
            retained_output_buffer_bytes: pool.buffer_bytes,
            active_pool_shape: pool.active_shape,
            cached_plans: s.plans.len(),
            plan_capacity: s.capacity,
            plan_creations: s.creations,
            capacity_evictions: s.capacity_evictions,
            budget_evictions: s.evictions
                + self.retained.conv_evictions.load(Ordering::Relaxed) as u64,
            budget_evicted_bytes: s.evicted_bytes
                + self.retained.conv_evicted.load(Ordering::Relaxed) as u64,
            budget_trims: self.retained.trims.load(Ordering::Relaxed),
            cached_weight_layouts: s.layouts.len(),
            implementations: s.plans.values().map(|p| p.name.clone()).collect(),
            primitive_setup_s: s.setup_s,
            primitive_exec_s: s.exec_s.clone(),
            executions: s.executions.clone(),
            bf16_convert_s: s.convert_s,
            category_exec_s: s.category_s.clone(),
        }
    }
}

/// What the model graphs read of the backend's setup, and its counters.
impl OnednnCpu {
    pub(crate) fn threads(&self) -> usize {
        self.threads as usize
    }
    pub(crate) fn cpu_isa(&self) -> u64 {
        self.cpu_isa
    }
    pub(crate) fn residual_rows(&self, tokens: usize) -> usize {
        self.strategy.residual_rows(tokens)
    }
    pub(crate) fn residual_row_tile(&self, live_rows: usize) -> usize {
        self.strategy.residual_row_tile(live_rows)
    }
    pub(crate) fn validate_workspace(&self, bytes: usize) -> Result<()> {
        self.check_workspace("model activations", &[bytes])
            .map(|_| ())
    }
    pub(crate) fn begin_shape(&self, b: usize, s: usize) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        if pool.active_shape != Some((b, s)) {
            pool.buffers.clear();
            pool.buffers_u16.clear();
            pool.buffer_bytes = 0;
            pool.active_shape = Some((b, s));
        }
    }
}
