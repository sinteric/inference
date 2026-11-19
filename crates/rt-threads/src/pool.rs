//! Persistent workers pinned to the same CPUs as the process budget.
//!
//! oneDNN GEMMs and row-parallel glue share this pool through
//! `dnnl::threadpool_interop::threadpool_iface`. The caller waits; workers
//! run the closures (Eigen-style). Nested `parallel_for` is serial.
//!
//! Concurrent callers (`exec_slots > 1` or two models in one process) queue
//! on the compute lock: one `parallel_for` occupies the workers at a time.
//! That keeps AMX / AVX-512 tile config on one primitive and avoids the
//! previous single-slot overwrite (a second caller could replace `job`/`f`
//! while the first was still draining).

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

thread_local! {
    static IN_PARALLEL: Cell<u32> = const { Cell::new(0) };
    static IS_WORKER: Cell<bool> = const { Cell::new(false) };
    static HOLDING_COMPUTE: Cell<bool> = const { Cell::new(false) };
}

/// Process-global compute lock. oneDNN `execute` programs AMX on the
/// *caller* thread before `parallel_for`; two in-flight executes therefore
/// SIGSEGV even if jobs queue. Re-entrant on the owning thread so
/// `parallel_for` nested under `execute` does not deadlock.
///
/// Callers are served in arrival order ([`ComputeOrder::Fifo`], the default):
/// each operator goes to the caller that has waited longest, so concurrent
/// requests share the pool operator by operator and a short request is not
/// starved by a long one. [`ComputeOrder::Any`] is a plain mutex, where the
/// thread that just released usually takes the lock straight back.
// Poison recovery is intentional: this lock protects scheduling only, not model data.
static COMPUTE: Mutex<()> = Mutex::new(());
static FIFO: FifoLock = FifoLock::new();
static ORDER_FIFO: AtomicBool = AtomicBool::new(true);

/// Which waiting caller gets the compute lock next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeOrder {
    /// The longest waiter (hand-off on release).
    Fifo,
    /// Whoever the OS wakes first; the releasing thread often wins.
    Any,
}

/// Choose the compute lock order for the process. Set it before any model
/// runs: a caller that already holds or waits on the lock keeps the order
/// it started with.
pub fn set_compute_order(order: ComputeOrder) {
    ORDER_FIFO.store(order == ComputeOrder::Fifo, Ordering::Relaxed);
}

pub fn compute_order() -> ComputeOrder {
    if ORDER_FIFO.load(Ordering::Relaxed) {
        ComputeOrder::Fifo
    } else {
        ComputeOrder::Any
    }
}

struct Waiter {
    thread: Thread,
    granted: AtomicBool,
}

/// Hand-off lock: release passes ownership to the front of the queue.
struct FifoLock {
    state: Mutex<FifoState>,
}

struct FifoState {
    held: bool,
    queue: VecDeque<Arc<Waiter>>,
}

impl FifoLock {
    const fn new() -> Self {
        Self {
            state: Mutex::new(FifoState {
                held: false,
                queue: VecDeque::new(),
            }),
        }
    }

    fn lock(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Release hands off while anyone waits, so a free lock has no queue.
        if !s.held {
            s.held = true;
            return;
        }
        let me = Arc::new(Waiter {
            thread: thread::current(),
            granted: AtomicBool::new(false),
        });
        s.queue.push_back(Arc::clone(&me));
        drop(s);
        note_wait();
        while !me.granted.load(Ordering::Acquire) {
            thread::park();
        }
    }

    fn unlock(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match s.queue.pop_front() {
            Some(next) => {
                drop(s);
                next.granted.store(true, Ordering::Release);
                next.thread.unpark();
            }
            None => s.held = false,
        }
    }
}

enum Held {
    Nested,
    Any(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>),
    Fifo,
}

pub struct ComputeGuard {
    held: Held,
}

impl Drop for ComputeGuard {
    fn drop(&mut self) {
        if matches!(self.held, Held::Nested) {
            return;
        }
        HOLDING_COMPUTE.with(|c| c.set(false));
        if matches!(self.held, Held::Fifo) {
            FIFO.unlock();
        }
    }
}

/// Milliseconds since first use, plus one, when a caller last had to wait
/// for the compute lock; 0 = never.
static LAST_WAIT: AtomicU64 = AtomicU64::new(0);

fn clock_ms() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

fn note_wait() {
    LAST_WAIT.store(clock_ms(), Ordering::Relaxed);
}

/// Whether a caller waited for the compute lock within `window`, i.e.
/// another caller in the process (another model or request) is computing.
/// Long regions such as a pool job running independent inputs one per
/// worker make such callers wait for the whole region, so they back off
/// while this holds.
pub fn compute_contended(window: Duration) -> bool {
    let last = LAST_WAIT.load(Ordering::Relaxed);
    last != 0 && clock_ms().saturating_sub(last) <= window.as_millis() as u64
}

/// Exclusive compute region (oneDNN execute / glue `par_for`). Nested
/// calls on the same thread are a no-op, and so are calls from inside a pool
/// job: the job's caller already holds the region for its workers.
pub fn compute_lock() -> ComputeGuard {
    if HOLDING_COMPUTE.with(|c| c.get()) || in_parallel() {
        return ComputeGuard { held: Held::Nested };
    }
    let held = if ORDER_FIFO.load(Ordering::Relaxed) {
        FIFO.lock();
        Held::Fifo
    } else {
        Held::Any(match COMPUTE.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                note_wait();
                COMPUTE.lock().unwrap_or_else(|e| e.into_inner())
            }
        })
    };
    HOLDING_COMPUTE.with(|c| c.set(true));
    ComputeGuard { held }
}

struct Job {
    next: AtomicUsize,
    /// Workers `0..workers` take chunks; the others skip the job, so a job
    /// capped below the pool width runs on that many workers, not on
    /// whichever wake first.
    workers: usize,
    nchunks: usize,
    chunk: usize,
    n: usize,
    /// Chunks that have finished `f`. Completion is chunk-based so extra
    /// workers that find no work do not have to check in.
    completed: AtomicUsize,
}

struct Slot {
    shutdown: bool,
    gen: usize,
    job: Option<Arc<Job>>,
    /// SAFETY: valid for the duration of the matching `run()` call.
    f: Option<&'static (dyn Fn(usize, usize) + Sync)>,
}

unsafe impl Send for Slot {}

struct Inner {
    lock: Mutex<Slot>,
    work_cv: Condvar,
    done_cv: Condvar,
    nworkers: usize,
    gen: AtomicUsize,
    spin_count: u32,
}

pub struct Pool {
    inner: Arc<Inner>,
    _joins: Vec<JoinHandle<()>>,
}

static POOL: OnceLock<Pool> = OnceLock::new();
static NTHREADS: OnceLock<usize> = OnceLock::new();

fn pin_to(cpu: usize) {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set);
    }
    let _ = cpu;
}

fn enter_parallel() {
    IN_PARALLEL.with(|c| c.set(c.get() + 1));
}

fn leave_parallel() {
    IN_PARALLEL.with(|c| c.set(c.get().saturating_sub(1)));
}

pub fn in_parallel() -> bool {
    IS_WORKER.with(|c| c.get()) || IN_PARALLEL.with(|c| c.get() > 0)
}

pub fn installed_nthreads() -> usize {
    NTHREADS.get().copied().unwrap_or(1)
}

fn drain(job: &Job, f: &(dyn Fn(usize, usize) + Sync)) {
    loop {
        let i = job.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.nchunks {
            break;
        }
        let lo = i * job.chunk;
        let hi = lo.saturating_add(job.chunk).min(job.n);
        f(lo, hi);
        job.completed.fetch_add(1, Ordering::AcqRel);
    }
}

fn worker(inner: Arc<Inner>, index: usize, cpu: Option<usize>) {
    if let Some(c) = cpu {
        pin_to(c);
    }
    IS_WORKER.with(|c| c.set(true));
    let mut my_gen = 0usize;
    loop {
        for _ in 0..inner.spin_count {
            if inner.gen.load(Ordering::Acquire) != my_gen {
                break;
            }
            std::hint::spin_loop();
        }
        let mut guard = inner.lock.lock().unwrap_or_else(|e| e.into_inner());
        while guard.gen == my_gen && !guard.shutdown {
            guard = inner.work_cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        if guard.shutdown {
            break;
        }
        my_gen = guard.gen;
        // Caller clears `job` once every chunk has finished. A late worker
        // can observe the new generation after that; skip rather than panic.
        let Some(job) = guard.job.clone() else {
            continue;
        };
        let Some(f) = guard.f else {
            continue;
        };
        drop(guard);
        if index >= job.workers {
            continue;
        }
        enter_parallel();
        drain(&job, f);
        leave_parallel();
        let _done = inner.lock.lock().unwrap_or_else(|e| e.into_inner());
        if job.completed.load(Ordering::Acquire) >= job.nchunks {
            inner.done_cv.notify_all();
        }
    }
}

static INSTALL: Mutex<()> = Mutex::new(());

/// Install a process-wide pool. A later different size is ignored; use
/// [`install_or_check`] to reject conflicting process configuration.
pub fn install(nthreads: usize, cpus: &[usize]) -> rt_core::Result<()> {
    install_with(nthreads, cpus, 500)
}
pub fn install_with(nthreads: usize, cpus: &[usize], spin_count: u32) -> rt_core::Result<()> {
    validate_pool(nthreads, cpus)?;
    let _lock = INSTALL
        .lock()
        .map_err(|_| rt_core::Error::Poisoned("pool installation"))?;
    if NTHREADS.get().is_some() {
        return Ok(());
    }
    create_pool(nthreads, cpus, spin_count)
}
fn validate_pool(nthreads: usize, cpus: &[usize]) -> rt_core::Result<()> {
    if nthreads == 0
        || nthreads > libc::CPU_SETSIZE as usize
        || cpus.iter().any(|&c| c >= libc::CPU_SETSIZE as usize)
    {
        return Err(rt_core::Error::Config(format!(
            "invalid pool threads {nthreads} or CPU set"
        )));
    }
    Ok(())
}
fn create_pool(nthreads: usize, cpus: &[usize], spin_count: u32) -> rt_core::Result<()> {
    validate_pool(nthreads, cpus)?;
    if nthreads > 1 {
        let inner = Arc::new(Inner {
            lock: Mutex::new(Slot {
                shutdown: false,
                gen: 0,
                job: None,
                f: None,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            nworkers: nthreads,
            gen: AtomicUsize::new(0),
            spin_count,
        });
        let mut joins = Vec::with_capacity(nthreads);
        for i in 0..nthreads {
            let cpu = cpus.get(i).copied();
            let worker_inner = Arc::clone(&inner);
            match thread::Builder::new()
                .name(format!("snt-par-{i}"))
                .spawn(move || worker(worker_inner, i, cpu))
            {
                Ok(join) => joins.push(join),
                Err(e) => {
                    inner
                        .lock
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .shutdown = true;
                    inner.work_cv.notify_all();
                    for join in joins {
                        let _ = join.join();
                    }
                    return Err(e.into());
                }
            }
        }
        // INSTALL serializes initialization, so no competing setter exists.
        let _ = POOL.set(Pool {
            inner,
            _joins: joins,
        });
    }
    let _ = NTHREADS.set(nthreads);
    Ok(())
}

/// Install or validate the process-global worker count.
pub fn install_or_check(nthreads: usize, cpus: &[usize]) -> rt_core::Result<()> {
    install_or_check_with(nthreads, cpus, 500)
}
pub fn install_or_check_with(
    nthreads: usize,
    cpus: &[usize],
    spin_count: u32,
) -> rt_core::Result<()> {
    validate_pool(nthreads, cpus)?;
    let _lock = INSTALL
        .lock()
        .map_err(|_| rt_core::Error::Poisoned("pool installation"))?;
    if let Some(&have) = NTHREADS.get() {
        if have != nthreads {
            return Err(rt_core::Error::Config(format!("thread pool is process-global ({have} threads already installed; requested {nthreads})")));
        }
        return Ok(());
    }
    create_pool(nthreads, cpus, spin_count)
}

/// Run `f` on `[0, n)` split across up to `nthreads` workers. Returns whether
/// the persistent pool handled the call.
pub fn try_run(nthreads: usize, n: usize, grain: usize, f: &(dyn Fn(usize, usize) + Sync)) -> bool {
    if in_parallel() {
        f(0, n);
        return true;
    }
    let Some(pool) = POOL.get() else {
        return false;
    };
    let nthreads = nthreads.min(pool.inner.nworkers).max(1);
    if nthreads <= 1 {
        f(0, n);
        return true;
    }
    let chunk = n.div_ceil(nthreads).max(grain);
    let nchunks = n.div_ceil(chunk);
    if nchunks <= 1 {
        f(0, n);
        return true;
    }
    // One in-flight job (and exclusive of oneDNN `execute` on another
    // caller — see [`compute_lock`]).
    let _compute = compute_lock();
    let job = Arc::new(Job {
        next: AtomicUsize::new(0),
        workers: nthreads,
        nchunks,
        chunk,
        n,
        completed: AtomicUsize::new(0),
    });
    // SAFETY: `f` outlives this call; workers observe the new generation and
    // finish before we return (we wait on `completed`).
    let f_static: &'static (dyn Fn(usize, usize) + Sync) = unsafe { std::mem::transmute(f) };
    let mut guard = pool.inner.lock.lock().unwrap_or_else(|e| e.into_inner());
    guard.job = Some(Arc::clone(&job));
    guard.f = Some(f_static);
    guard.gen = guard.gen.wrapping_add(1);
    pool.inner.gen.store(guard.gen, Ordering::Release);
    pool.inner.work_cv.notify_all();
    while job.completed.load(Ordering::Acquire) < nchunks {
        guard = pool
            .inner
            .done_cv
            .wait(guard)
            .unwrap_or_else(|e| e.into_inner());
    }
    guard.job = None;
    guard.f = None;
    true
}

/// Parallel-for over `[0, n)`. Uses the pinned pool when installed, otherwise
/// `std::thread::scope`. Nested calls (oneDNN GEMM or glue already running)
/// execute serially on the current thread.
pub fn par_for(nthreads: usize, n: usize, grain: usize, f: impl Fn(usize, usize) + Sync) {
    if n == 0 {
        return;
    }
    let nthreads = nthreads.max(1);
    if in_parallel() || nthreads == 1 || n < grain.saturating_mul(2) {
        f(0, n);
        return;
    }
    if try_run(nthreads, n, grain, &f) {
        return;
    }
    let chunk = n.div_ceil(nthreads).max(grain);
    std::thread::scope(|scope| {
        let f = &f;
        let mut start = 0;
        let mut spawned = 0;
        while start < n {
            let end = start.saturating_add(chunk).min(n);
            let s = start;
            if spawned + 1 >= nthreads || end == n {
                f(s, end);
                start = end;
                continue;
            }
            if thread::Builder::new()
                .spawn_scoped(scope, move || f(s, end))
                .is_err()
            {
                f(s, end);
            }
            spawned += 1;
            start = end;
        }
    });
}

#[no_mangle]
pub extern "C" fn sinteric_tp_num_threads() -> i32 {
    i32::try_from(installed_nthreads()).unwrap_or(i32::MAX)
}

#[no_mangle]
pub extern "C" fn sinteric_tp_in_parallel() -> i32 {
    i32::from(in_parallel())
}

#[no_mangle]
pub unsafe extern "C" fn sinteric_tp_parallel_for(
    n: i32,
    cb: Option<unsafe extern "C" fn(i32, i32, *mut c_void)>,
    ctx: *mut c_void,
) {
    sinteric_tp_parallel_for_capped(n, cb, ctx, 0)
}

/// [`sinteric_tp_parallel_for`] on at most `max_threads` pool workers
/// (`<= 0`: the whole pool). The `threadpool_iface` of a backend with a
/// thread cap calls this.
#[no_mangle]
pub unsafe extern "C" fn sinteric_tp_parallel_for_capped(
    n: i32,
    cb: Option<unsafe extern "C" fn(i32, i32, *mut c_void)>,
    ctx: *mut c_void,
    max_threads: i32,
) {
    let Some(cb) = cb else {
        return;
    };
    if n <= 0 {
        return;
    }
    if n == 1 || in_parallel() {
        // oneDNN may precompute partitions when building the primitive.
        // Preserve every (i, n) even when nested execution is serial.
        for i in 0..n {
            cb(i, n, ctx);
        }
        return;
    }
    let n_us = n as usize;
    let all = installed_nthreads().max(1);
    let nthreads = match usize::try_from(max_threads) {
        Ok(cap) if cap > 0 => cap.min(all),
        _ => all,
    };
    let ctx_us = ctx as usize;
    par_for(nthreads, n_us, 1, |lo, hi| {
        for i in lo..hi {
            cb(i as i32, n, ctx_us as *mut c_void);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn onednn_callbacks_preserve_all_partitions() {
        unsafe extern "C" fn callback(i: i32, n: i32, ctx: *mut c_void) {
            let hits = &*(ctx as *const Vec<AtomicUsize>);
            assert_eq!(n as usize, hits.len());
            hits[i as usize].fetch_add(1, Ordering::Relaxed);
        }
        for nested in [false, true] {
            let hits: Vec<_> = (0..7).map(|_| AtomicUsize::new(0)).collect();
            if nested {
                enter_parallel();
            }
            unsafe {
                sinteric_tp_parallel_for(7, Some(callback), &hits as *const _ as *mut c_void);
            }
            if nested {
                leave_parallel();
            }
            assert!(hits.iter().all(|n| n.load(Ordering::Relaxed) == 1));
        }
    }

    #[test]
    fn fifo_lock_serves_waiters_in_arrival_order() {
        let lock = FifoLock::new();
        let order = Mutex::new(Vec::new());
        lock.lock();
        std::thread::scope(|scope| {
            for i in 0..4 {
                let (lock, order) = (&lock, &order);
                scope.spawn(move || {
                    lock.lock();
                    order.lock().unwrap().push(i);
                    lock.unlock();
                });
                // Enqueue one at a time so arrival order is known.
                while lock.state.lock().unwrap().queue.len() < i + 1 {
                    std::thread::yield_now();
                }
            }
            // The holder releasing and re-locking queues behind the waiters.
            lock.unlock();
            lock.lock();
            order.lock().unwrap().push(99);
            lock.unlock();
        });
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 99]);
        assert!(!lock.state.lock().unwrap().held);
    }

    #[test]
    fn par_for_covers_without_pool() {
        let mut hits = vec![0u32; 20];
        let hits = std::sync::Mutex::new(&mut hits[..]);
        par_for(4, 20, 1, |lo, hi| {
            let mut g = hits.lock().unwrap();
            for i in lo..hi {
                g[i] += 1;
            }
        });
        assert!(hits.lock().unwrap().iter().all(|&c| c == 1));
    }

    #[test]
    fn nested_par_for_is_serial() {
        enter_parallel();
        let mut hits = vec![0u32; 8];
        let hits = std::sync::Mutex::new(&mut hits[..]);
        par_for(4, 8, 1, |lo, hi| {
            let mut g = hits.lock().unwrap();
            for i in lo..hi {
                g[i] += 1;
            }
        });
        leave_parallel();
        assert!(hits.lock().unwrap().iter().all(|&c| c == 1));
    }

    #[test]
    fn concurrent_par_for_covers_exactly_once_per_caller() {
        install(2, &[]).unwrap();
        const N: usize = 1 << 12;
        const CALLERS: usize = 8;
        let hits: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
        let hits = std::sync::Arc::new(hits);
        std::thread::scope(|scope| {
            for _ in 0..CALLERS {
                let hits = std::sync::Arc::clone(&hits);
                scope.spawn(move || {
                    par_for(2, N, 8, |lo, hi| {
                        for i in lo..hi {
                            hits[i].fetch_add(1, Ordering::Relaxed);
                        }
                    });
                });
            }
        });
        assert!(hits.iter().all(|c| c.load(Ordering::Relaxed) == CALLERS));
    }
}
