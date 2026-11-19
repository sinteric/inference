//! Bindgen FFI for the oneDNN 3.13.4 C API.
//!
//! `build.rs` statically links pinned `libdnnl.a` (CPU THREADPOOL runtime).
//! [`rt-onednn`] calls the `dnnl_*` symbols directly.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused)]
#![allow(clippy::all)]

pub use rt_core::{Error, Result};
use std::ffi::c_void;

include!("bindings.rs");

unsafe extern "C" {
    pub fn sinteric_dnnl_threadpool_create() -> *mut c_void;
    /// A threadpool that reports at most `max_threads` threads to oneDNN and
    /// runs its `parallel_for` on at most that many pool workers (`<= 0`: all).
    pub fn sinteric_dnnl_threadpool_create_capped(max_threads: i32) -> *mut c_void;
    pub fn sinteric_dnnl_threadpool_destroy(threadpool: *mut c_void);
    /// Run the AMX tile-state probe for about `budget_ms` milliseconds.
    pub fn sinteric_amx_tile_state_probe(budget_ms: i32) -> SntAmxProbeResult;
}

/// Result of [`sinteric_amx_tile_state_probe`] (`snt_amx_probe_result`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SntAmxProbeResult {
    /// `0` when the probe ran, `-1` when AMX is unavailable to this process.
    pub status: i32,
    /// Tile GEMMs checked.
    pub iterations: u64,
    /// Results that differed from the first (lost tile state).
    pub corrupt: u64,
    /// Involuntary context switches of the checked thread.
    pub preemptions: u64,
}

/// Pinned oneDNN release this crate was generated against.
pub const PINNED_MAJOR: i32 = 3;
pub const PINNED_MINOR: i32 = 13;
pub const PINNED_PATCH: i32 = 4;
pub const PINNED_COMMIT: &str = "c531fe1a5cd631b7f60277cdf6675c588cadbc28";

/// `DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)` from `dnnl_types.h`.
#[inline]
pub fn dnnl_arg_attr_multiple_post_op(idx: i32) -> Result<i32> {
    if idx < 0 {
        return Err(Error::Config(format!("negative post-op index {idx}")));
    }
    idx.checked_add(1)
        .and_then(|n| DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE.checked_mul(n))
        .ok_or(Error::SizeOverflow("oneDNN post-op index"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_release_constants() {
        assert_eq!(PINNED_MAJOR, 3);
        assert_eq!(PINNED_MINOR, 13);
        assert_eq!(PINNED_PATCH, 4);
        assert_eq!(dnnl_success, 0);
        assert_eq!(dnnl_bf16, 2);
        assert_eq!(dnnl_f32, 3);
        assert_eq!(dnnl_eltwise_gelu_tanh, 0x2b);
        assert_eq!(dnnl_accumulation_mode_f32, 4);
        assert_eq!(dnnl_query_ndims_s32, 256);
        assert_eq!(dnnl_query_inner_nblks_s32, 263);
        assert_eq!(dnnl_query_strides, 14);
        assert_eq!(dnnl_blocked, 2);
        assert_eq!(dnnl_cpu_isa_avx512_core_bf16, 0xe7);
        assert_eq!(dnnl_cpu, 1);
        assert_eq!(DNNL_ARG_SRC, 1);
        assert_eq!(DNNL_ARG_WEIGHTS, 33);
        assert_eq!(DNNL_ARG_DST, 17);
        assert_eq!(DNNL_ARG_SCRATCHPAD, 80);
        assert_eq!(dnnl_arg_attr_multiple_post_op(1).unwrap(), 65536);
        assert_eq!(DNNL_MAX_NDIMS, 12);
        assert_eq!(DNNL_RUNTIME_THREADPOOL, 8);
    }

    #[test]
    fn threadpool_iface_create_destroy() {
        let tp = unsafe { sinteric_dnnl_threadpool_create() };
        assert!(!tp.is_null());
        unsafe { sinteric_dnnl_threadpool_destroy(tp) };
    }
}
