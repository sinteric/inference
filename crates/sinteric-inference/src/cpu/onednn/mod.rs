//! oneDNN 3.13.4 behind a safe wrapper: [`OnednnCpu`], the backend every
//! model graph runs on.
//!
//! The C++ `bridge.cpp` is no longer linked: every primitive
//! (matmul, reorder, layernorm, softmax, eltwise, binary, kind-7 GeGLU
//! post-ops) is built with `dnnl_*` from Rust.
//!
//! # Process-global state
//!
//! Multiple [`OnednnCpu`] instances can coexist (each has its own engine,
//! execute-slot pool, primitive cache, and activation pool). These are
//! process-global and owned by the first successful [`OnednnCpu::load_with`]
//! (or the library `Runtime`) in the process:
//!
//! - **oneDNN ISA cap** (`dnnl_set_max_cpu_isa`)
//! - **Pinned thread pool** (`crate::cpu::threads::par_for`) used for GEMMs and glue
//! - **Threadpool max concurrency** (`dnnl_threadpool_interop_set_max_concurrency`)
//!
//! A later load that disagrees is an error. `dnnl_set_verbose` is last-write.
//! On a host that loses AMX tile state when a thread is preempted
//! ([`amx_health`]), a `BestAvailable` cap resolves to `AVX512_CORE_BF16`
//! before it is applied.
//! Process CPU affinity: only
//! [`crate::cpu::threads::ThreadBudget::apply_process_policy`] (CLI/bench). That path
//! does not mutate process environment.
//!
//! oneDNN is **statically linked** with `DNNL_CPU_RUNTIME=THREADPOOL`. Streams
//! are created with `dnnl_threadpool_interop_stream_create`. `ONEDNN_MAX_CPU_ISA`
//! is not read; the cap is applied through `dnnl_set_max_cpu_isa` before any
//! other oneDNN API.

mod amx_probe;
mod api;
mod cache;
pub mod conv;
mod cpu;
mod options;
mod strategy;
mod tuning;

#[cfg(test)]
pub use amx_probe::set_amx_probe;
pub use amx_probe::{amx_health, amx_probe_enabled, AmxHealth};
pub use cache::{
    apply_cache, cache_path_with, cache_write_failed, file_stamp, pinned_dnnl, CacheMeta,
    CacheWriter, FileStamp, PackedEntry, RawTensor, CACHE_VERSION,
};
pub use cpu::{OnednnCpu, PackedWeight};
pub use options::{CacheOptions, CpuIsaCap, SessionOptions, DEFAULT_WORKSPACE_LIMIT};
pub use tuning::TuningPlan;

/// `dnnl_cpu_isa_avx512_core_amx` / `avx10_1_512_amx` bit pattern (`0xfef`).
/// Also matches `avx10_1_512_amx_fp16` (`0x1fef`).
#[inline]
pub fn isa_has_amx(isa: u64) -> bool {
    let amx = onednn_sys::dnnl_cpu_isa_avx512_core_amx as u64;
    (isa & amx) == amx
}

pub use strategy::{clamp_attn_tile, IsaStrategy};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod bad_input_tests {
    use super::*;
    #[test]
    fn malformed_native_requests_return_errors() {
        let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
        let hw = OnednnCpu::load(&budget).unwrap();
        assert!(hw.pack(0, 1, &[]).is_err());
        assert!(hw.pack(usize::MAX, 2, &[]).is_err());
        assert!(hw.pack_from_cache(usize::MAX, 2, &[]).is_err());
        assert!(hw.mm(0, 1, 1, 1, 0, 1, 0, 1, false, &[], &[]).is_err());
        assert!(hw.mm(1, 1, 1, 1, 0, 1, 0, 1, false, &[], &[1.]).is_err());
        assert!(hw
            .mm(2, 2, 2, 2, usize::MAX, 2, 0, 2, false, &[], &[])
            .is_err());
        assert!(hw.buffer(usize::MAX).is_err());
        // The workspace limit caps pool retention only; larger buffers still allocate.
        assert_eq!(
            hw.buffer(DEFAULT_WORKSPACE_LIMIT / 4 + 1).unwrap().len(),
            DEFAULT_WORKSPACE_LIMIT / 4 + 1
        );
        assert!(hw.reserve(2, usize::MAX).is_err());
        assert!(hw.softmax(1, 0, &[]).is_err());
        assert!(hw
            .layernorm(1, 2, &[1., 2.], &[1.], &[0., 0.], 1e-6)
            .is_err());
        assert!(hw
            .attention(&[], &[], &[], 1, 1, 1, 1, 1, false, 0, &[vec![]])
            .is_err());
        assert!(hw
            .attention(
                &[0.],
                &[0.],
                &[0.],
                1,
                1,
                1,
                1,
                1,
                false,
                usize::MAX,
                &[vec![]]
            )
            .is_err());
    }
    #[test]
    fn invalid_session_options_fail_before_global_installation() {
        let budget = crate::cpu::threads::ThreadBudget::new(1).unwrap();
        assert!(OnednnCpu::load_with(
            &budget,
            SessionOptions {
                exec_slots: 0,
                ..SessionOptions::default()
            }
        )
        .is_err());
        assert!(OnednnCpu::load_with(
            &budget,
            SessionOptions {
                exec_slots: usize::MAX,
                ..SessionOptions::default()
            }
        )
        .is_err());
    }
}
