//! Linked oneDNN C API. Process-global ISA / threadpool concurrency are applied here.

use crate::cpu::onednn::options::CpuIsaCap;
use crate::{check_shape, Result};
use onednn_sys::*;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::{Mutex, OnceLock};

/// Process-global oneDNN ISA cap. [`Api::init`] is the owner; a later
/// mismatch is an error. Models loaded from a [`crate`] session never
/// re-apply this.
/// Process-global ISA cap. The lock is held across `dnnl_set_max_cpu_isa`, so
/// no backend can reach oneDNN (and freeze the dispatcher) before the first
/// cap is applied; later callers block until it is recorded.
static MAX_CPU_ISA: Mutex<Option<CpuIsaCap>> = Mutex::new(None);
/// Process-global threadpool width. First `apply_threads` wins.
static TP_THREADS: OnceLock<i32> = OnceLock::new();
/// Process-lifetime C++ `threadpool_iface`. Streams borrow this pointer.
struct ThreadpoolPtr(*mut c_void);
unsafe impl Send for ThreadpoolPtr {}
unsafe impl Sync for ThreadpoolPtr {}
static THREADPOOL: OnceLock<ThreadpoolPtr> = OnceLock::new();

/// Function table bound to the statically linked `libdnnl.a`.
pub struct Api {
    pub engine_create:
        unsafe extern "C" fn(*mut dnnl_engine_t, dnnl_engine_kind_t, usize) -> dnnl_status_t,
    pub engine_destroy: unsafe extern "C" fn(dnnl_engine_t) -> dnnl_status_t,
    pub stream_create:
        unsafe extern "C" fn(*mut dnnl_stream_t, dnnl_engine_t, *mut c_void) -> dnnl_status_t,
    pub threadpool: *mut c_void,
    pub stream_wait: unsafe extern "C" fn(dnnl_stream_t) -> dnnl_status_t,
    pub stream_destroy: unsafe extern "C" fn(dnnl_stream_t) -> dnnl_status_t,
    #[allow(dead_code)]
    pub set_cache: unsafe extern "C" fn(i32) -> dnnl_status_t,
    pub attr_create: unsafe extern "C" fn(*mut dnnl_primitive_attr_t) -> dnnl_status_t,
    pub attr_destroy: unsafe extern "C" fn(dnnl_primitive_attr_t) -> dnnl_status_t,
    pub attr_fpmath:
        unsafe extern "C" fn(dnnl_primitive_attr_t, dnnl_fpmath_mode_t) -> dnnl_status_t,
    pub attr_accum:
        unsafe extern "C" fn(dnnl_primitive_attr_t, dnnl_accumulation_mode_t) -> dnnl_status_t,
    pub attr_scratch:
        unsafe extern "C" fn(dnnl_primitive_attr_t, dnnl_scratchpad_mode_t) -> dnnl_status_t,
    pub attr_post_ops:
        unsafe extern "C" fn(dnnl_primitive_attr_t, const_dnnl_post_ops_t) -> dnnl_status_t,
    pub post_ops_create: unsafe extern "C" fn(*mut dnnl_post_ops_t) -> dnnl_status_t,
    pub post_ops_destroy: unsafe extern "C" fn(dnnl_post_ops_t) -> dnnl_status_t,
    pub post_ops_sum:
        unsafe extern "C" fn(dnnl_post_ops_t, f32, i32, dnnl_data_type_t) -> dnnl_status_t,
    pub post_ops_eltwise:
        unsafe extern "C" fn(dnnl_post_ops_t, dnnl_alg_kind_t, f32, f32) -> dnnl_status_t,
    pub post_ops_binary: unsafe extern "C" fn(
        dnnl_post_ops_t,
        dnnl_alg_kind_t,
        const_dnnl_memory_desc_t,
    ) -> dnnl_status_t,
    pub md_strides: unsafe extern "C" fn(
        *mut dnnl_memory_desc_t,
        i32,
        *const dnnl_dim_t,
        dnnl_data_type_t,
        *const dnnl_dim_t,
    ) -> dnnl_status_t,
    pub md_tag: unsafe extern "C" fn(
        *mut dnnl_memory_desc_t,
        i32,
        *const dnnl_dim_t,
        dnnl_data_type_t,
        dnnl_format_tag_t,
    ) -> dnnl_status_t,
    pub md_clone:
        unsafe extern "C" fn(*mut dnnl_memory_desc_t, const_dnnl_memory_desc_t) -> dnnl_status_t,
    pub md_destroy: unsafe extern "C" fn(dnnl_memory_desc_t) -> dnnl_status_t,
    pub md_size: unsafe extern "C" fn(const_dnnl_memory_desc_t) -> usize,
    pub md_query:
        unsafe extern "C" fn(const_dnnl_memory_desc_t, dnnl_query_t, *mut c_void) -> dnnl_status_t,
    pub get_effective_cpu_isa: unsafe extern "C" fn() -> dnnl_cpu_isa_t,
    pub mem_create: unsafe extern "C" fn(
        *mut dnnl_memory_t,
        const_dnnl_memory_desc_t,
        dnnl_engine_t,
        *mut c_void,
    ) -> dnnl_status_t,
    pub mem_set: unsafe extern "C" fn(dnnl_memory_t, *mut c_void) -> dnnl_status_t,
    pub mem_destroy: unsafe extern "C" fn(dnnl_memory_t) -> dnnl_status_t,
    pub pd_query: unsafe extern "C" fn(
        const_dnnl_primitive_desc_t,
        dnnl_query_t,
        i32,
        *mut c_void,
    ) -> dnnl_status_t,
    pub pd_query_md: unsafe extern "C" fn(
        const_dnnl_primitive_desc_t,
        dnnl_query_t,
        i32,
    ) -> const_dnnl_memory_desc_t,
    pub pd_destroy: unsafe extern "C" fn(dnnl_primitive_desc_t) -> dnnl_status_t,
    pub prim_create:
        unsafe extern "C" fn(*mut dnnl_primitive_t, const_dnnl_primitive_desc_t) -> dnnl_status_t,
    pub prim_destroy: unsafe extern "C" fn(dnnl_primitive_t) -> dnnl_status_t,
    pub prim_execute: unsafe extern "C" fn(
        const_dnnl_primitive_t,
        dnnl_stream_t,
        i32,
        *const dnnl_exec_arg_t,
    ) -> dnnl_status_t,
    pub reorder_pd: unsafe extern "C" fn(
        *mut dnnl_primitive_desc_t,
        const_dnnl_memory_desc_t,
        dnnl_engine_t,
        const_dnnl_memory_desc_t,
        dnnl_engine_t,
        const_dnnl_primitive_attr_t,
    ) -> dnnl_status_t,
    pub matmul_pd: unsafe extern "C" fn(
        *mut dnnl_primitive_desc_t,
        dnnl_engine_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        const_dnnl_primitive_attr_t,
    ) -> dnnl_status_t,
    pub binary_pd: unsafe extern "C" fn(
        *mut dnnl_primitive_desc_t,
        dnnl_engine_t,
        dnnl_alg_kind_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        const_dnnl_primitive_attr_t,
    ) -> dnnl_status_t,
    pub eltwise_pd: unsafe extern "C" fn(
        *mut dnnl_primitive_desc_t,
        dnnl_engine_t,
        dnnl_prop_kind_t,
        dnnl_alg_kind_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        f32,
        f32,
        const_dnnl_primitive_attr_t,
    ) -> dnnl_status_t,
    pub softmax_pd: unsafe extern "C" fn(
        *mut dnnl_primitive_desc_t,
        dnnl_engine_t,
        dnnl_prop_kind_t,
        dnnl_alg_kind_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        i32,
        const_dnnl_primitive_attr_t,
    ) -> dnnl_status_t,
    pub layernorm_pd: unsafe extern "C" fn(
        *mut dnnl_primitive_desc_t,
        dnnl_engine_t,
        dnnl_prop_kind_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        const_dnnl_memory_desc_t,
        f32,
        u32,
        const_dnnl_primitive_attr_t,
    ) -> dnnl_status_t,
    pub info: String,
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    /// Bind the statically linked oneDNN. Threadpool max concurrency is
    /// applied **before** primitive creation so hashing sees the budget
    /// (TLS; the pool is not active until execute). Then [`CpuIsaCap`] via
    /// `dnnl_set_max_cpu_isa` (including before `dnnl_version`). Idempotent
    /// when the cap matches; mismatch is an error.
    pub fn init(isa: CpuIsaCap, verbose: i32, nthreads: i32) -> Result<Self> {
        apply_tp_threads(nthreads)?;
        let threadpool = threadpool_ptr()?;
        unsafe {
            apply_max_cpu_isa(dnnl_set_max_cpu_isa, isa)?;
            let _ = dnnl_set_verbose(verbose);

            let v = dnnl_version();
            check_shape!(!v.is_null(), "dnnl_version returned null");
            let ver = &*v;
            check_shape!(
                ver.major == onednn_sys::PINNED_MAJOR
                    && ver.minor == onednn_sys::PINNED_MINOR
                    && ver.patch == onednn_sys::PINNED_PATCH,
                "oneDNN ABI/release mismatch: {}.{}.{} (want 3.13.4)",
                ver.major,
                ver.minor,
                ver.patch
            );
            let hash = if ver.hash.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ver.hash).to_string_lossy().into_owned()
            };
            check_shape!(
                dnnl_set_primitive_cache_capacity(0) == dnnl_success,
                "primitive cache configuration"
            );

            let info = format!(
                "oneDNN{}.{}.{} {hash}; Rust C API; statically linked; threadpool; standalone Rust-owned data; public primitives; FP32 strict",
                ver.major, ver.minor, ver.patch
            );

            Ok(Self {
                engine_create: dnnl_engine_create,
                engine_destroy: dnnl_engine_destroy,
                stream_create: dnnl_threadpool_interop_stream_create,
                threadpool,
                stream_wait: dnnl_stream_wait,
                stream_destroy: dnnl_stream_destroy,
                set_cache: dnnl_set_primitive_cache_capacity,
                attr_create: dnnl_primitive_attr_create,
                attr_destroy: dnnl_primitive_attr_destroy,
                attr_fpmath: dnnl_primitive_attr_set_fpmath_mode,
                attr_accum: dnnl_primitive_attr_set_accumulation_mode,
                attr_scratch: dnnl_primitive_attr_set_scratchpad_mode,
                attr_post_ops: dnnl_primitive_attr_set_post_ops,
                post_ops_create: dnnl_post_ops_create,
                post_ops_destroy: dnnl_post_ops_destroy,
                post_ops_sum: dnnl_post_ops_append_sum,
                post_ops_eltwise: dnnl_post_ops_append_eltwise,
                post_ops_binary: dnnl_post_ops_append_binary,
                md_strides: dnnl_memory_desc_create_with_strides,
                md_tag: dnnl_memory_desc_create_with_tag,
                md_clone: dnnl_memory_desc_clone,
                md_destroy: dnnl_memory_desc_destroy,
                md_size: dnnl_memory_desc_get_size,
                md_query: dnnl_memory_desc_query,
                get_effective_cpu_isa: dnnl_get_effective_cpu_isa,
                mem_create: dnnl_memory_create,
                mem_set: dnnl_memory_set_data_handle,
                mem_destroy: dnnl_memory_destroy,
                pd_query: dnnl_primitive_desc_query,
                pd_query_md: dnnl_primitive_desc_query_md,
                pd_destroy: dnnl_primitive_desc_destroy,
                prim_create: dnnl_primitive_create,
                prim_destroy: dnnl_primitive_destroy,
                prim_execute: dnnl_primitive_execute,
                reorder_pd: dnnl_reorder_primitive_desc_create,
                matmul_pd: dnnl_matmul_primitive_desc_create,
                binary_pd: dnnl_binary_primitive_desc_create,
                eltwise_pd: dnnl_eltwise_forward_primitive_desc_create,
                softmax_pd: dnnl_softmax_forward_primitive_desc_create,
                layernorm_pd: dnnl_layer_normalization_forward_primitive_desc_create,
                info,
            })
        }
    }

    pub fn check(st: dnnl_status_t, what: &'static str) -> Result<()> {
        if st != dnnl_success {
            return Err(crate::Error::BackendStatus {
                operation: what,
                status: st,
            });
        }
        Ok(())
    }

    /// Set oneDNN's (thread-local) max concurrency before creating
    /// primitives. `n` is the process pool width, or a smaller per-backend
    /// thread cap (`SessionOptions::threads`); wider than the pool is an
    /// error.
    pub fn apply_threads(&self, n: i32) -> Result<()> {
        let capped = || matches!(TP_THREADS.get(), Some(&have) if (1..have).contains(&n));
        if capped() {
            return set_max_concurrency(n);
        }
        // Another thread may have set the width between the check and here.
        apply_tp_threads(n).or_else(|e| {
            if capped() {
                set_max_concurrency(n)
            } else {
                Err(e)
            }
        })
    }
}

pub fn threadpool_ptr() -> Result<*mut c_void> {
    let p = THREADPOOL
        .get_or_init(|| {
            let p = unsafe { sinteric_dnnl_threadpool_create() };
            ThreadpoolPtr(p)
        })
        .0;
    if p.is_null() {
        return Err(crate::Error::Allocation {
            bytes: std::mem::size_of::<ThreadpoolPtr>(),
        });
    }
    Ok(p)
}

fn apply_tp_threads(n: i32) -> Result<()> {
    match TP_THREADS.set(n) {
        Ok(()) => {}
        Err(_)
            if TP_THREADS
                .get()
                .copied()
                .ok_or_else(|| crate::Error::Config("threadpool initialization".into()))?
                == n => {}
        Err(_) => {
            let have = TP_THREADS
                .get()
                .copied()
                .ok_or_else(|| crate::Error::Config("threadpool initialization".into()))?;
            return Err(crate::Error::Config(format!(
                "threadpool width is process-global ({have} already set; requested {n})"
            )));
        }
    }
    set_max_concurrency(n)
}

fn set_max_concurrency(n: i32) -> Result<()> {
    let st = unsafe { dnnl_threadpool_interop_set_max_concurrency(n) };
    check_shape!(
        st == dnnl_success,
        "dnnl_threadpool_interop_set_max_concurrency({n}) status {st}"
    );
    Ok(())
}

fn apply_max_cpu_isa(
    set_max_cpu_isa: unsafe extern "C" fn(dnnl_cpu_isa_t) -> dnnl_status_t,
    isa: CpuIsaCap,
) -> Result<()> {
    apply_max_cpu_isa_in(&MAX_CPU_ISA, set_max_cpu_isa, isa)
}

fn apply_max_cpu_isa_in(
    cell: &Mutex<Option<CpuIsaCap>>,
    set_max_cpu_isa: unsafe extern "C" fn(dnnl_cpu_isa_t) -> dnnl_status_t,
    isa: CpuIsaCap,
) -> Result<()> {
    let mut slot = cell
        .lock()
        .map_err(|_| crate::Error::Poisoned("oneDNN ISA cap"))?;
    if let Some(have) = *slot {
        if have != isa {
            return Err(crate::Error::Config(format!(
                "oneDNN ISA cap is process-global (already {have:?}; requested {isa:?})"
            )));
        }
        return Ok(());
    }
    let st = unsafe { set_max_cpu_isa(isa.as_dnnl()) };
    // `unimplemented` means the build disabled dispatcher control;
    // `invalid_arguments` on default can mean the ISA is already
    // frozen by a prior env/init. First-wins still records `isa`.
    if st != dnnl_success && isa != CpuIsaCap::BestAvailable {
        crate::invalid_shape!("dnnl_set_max_cpu_isa({:?}) status {st}", isa);
    }
    *slot = Some(isa);
    Ok(())
}

/// `DNNL_MEMORY_NONE` is NULL.
pub fn memory_none() -> *mut c_void {
    ptr::null_mut()
}

pub fn cstr<'a>(p: *const c_char) -> &'a str {
    if p.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
    }
}

#[cfg(test)]
mod isa_cap_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static CALLS: AtomicUsize = AtomicUsize::new(0);
    static APPLIED: AtomicBool = AtomicBool::new(false);
    unsafe extern "C" fn slow_setter(_: dnnl_cpu_isa_t) -> dnnl_status_t {
        CALLS.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(50));
        APPLIED.store(true, Ordering::SeqCst);
        dnnl_success
    }

    #[test]
    fn concurrent_callers_wait_for_the_native_setter() {
        let cell = Mutex::new(None);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(|| {
                        let r = apply_max_cpu_isa_in(&cell, slow_setter, CpuIsaCap::Avx512CoreBf16);
                        // No caller may return before the cap reached oneDNN.
                        (r.is_ok(), APPLIED.load(Ordering::SeqCst))
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), (true, true));
            }
        });
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        assert!(apply_max_cpu_isa_in(&cell, slow_setter, CpuIsaCap::BestAvailable).is_err());
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
    }
}
