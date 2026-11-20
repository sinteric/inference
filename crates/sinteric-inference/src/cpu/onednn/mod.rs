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
//! Nothing here sets the process's CPU affinity or environment.
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
    apply_cache, cache_path_with, file_stamp, pinned_dnnl, CacheMeta, CacheWriter, FileStamp,
    RawTensor, CACHE_VERSION,
};
pub(crate) use cpu::{Attention, Gemm};
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
        let budget = crate::cpu::threads::ThreadBudget::try_new(1).unwrap();
        let hw = OnednnCpu::load(&budget).unwrap();
        assert!(hw.pack(0, 1, &[]).is_err());
        assert!(hw.pack(usize::MAX, 2, &[]).is_err());
        assert!(hw.restore_packed_checked(usize::MAX, 2, &[], &[]).is_err());
        assert!(hw.mm(Gemm::nn(0, 1, 1, 1, 0, 0), &[], &[]).is_err());
        assert!(hw.mm(Gemm::nn(1, 1, 1, 1, 0, 0), &[], &[1.]).is_err());
        assert!(hw
            .mm(Gemm::nn(2, 2, 2, 2, usize::MAX, 0), &[], &[])
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
        let one = Attention {
            batch: 1,
            seq: 1,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 1,
            tile: 1,
            window: None,
            scale: 1.0,
            padding: &[vec![]],
        };
        assert!(hw.attention(&one, &[], &[], &[]).is_err());
        let wide = Attention {
            window: Some(usize::MAX),
            ..one
        };
        assert!(hw.attention(&wide, &[0.], &[0.], &[0.]).is_err());
    }
    #[test]
    fn invalid_session_options_fail_before_global_installation() {
        let budget = crate::cpu::threads::ThreadBudget::try_new(1).unwrap();
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
