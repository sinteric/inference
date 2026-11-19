//! Shared, model-independent runtime primitives.
//!
//! `rt-*` crates never depend on a model crate. Models may depend on `rt-*`.

mod align;
pub mod env;
mod error;
mod workspace;

pub use align::{align_up, AlignedBuf, CACHELINE};
pub use error::{
    checked_size, matrix_span, read_json, try_vec, Error, Result, ResultContext, SystemOneError,
};
pub use workspace::{ScratchLease, Workspace};

/// Bytes reserved for packed (non-embedding) Vela 2.0 0.3B weights in FP32.
///
/// 112_461_312 parameters × 4 bytes. Embeddings are counted separately because
/// they stay in a lookup table rather than the packed GEMM buffers.
pub const VELA2_PACKED_WEIGHT_BYTES: usize = 449_845_248;

/// Default scratch-cap used by the Candle+MLAS variant (64 MiB).
pub const DEFAULT_SCRATCH_CAP_BYTES: usize = 64 * 1024 * 1024;

#[cfg(test)]
mod bad_input_tests {
    use super::*;
    #[test]
    fn checked_sizes_and_alignment_reject_overflow() {
        assert!(matches!(
            checked_size("test", &[usize::MAX, 2]),
            Err(Error::SizeOverflow(_))
        ));
        assert!(AlignedBuf::new(usize::MAX).is_err());
        assert!(align_up(usize::MAX, 64).is_err());
        assert!(align_up(1, 0).is_err());
        assert!(align_up(1, 3).is_err());
        assert!(try_vec::<f32>(usize::MAX).is_err());
        assert!(matrix_span(0, 1, 1, 1, 1).is_err());
        assert!(matrix_span(usize::MAX, 2, 2, usize::MAX, 2).is_err());
    }
    #[test]
    fn workspace_and_copy_reject_invalid_requests() {
        let mut ws = Workspace::with_cap_bytes(64).unwrap();
        ws.lease(1).unwrap();
        assert!(ws.lease(usize::MAX).is_err());
        let mut buf = AlignedBuf::new(2).unwrap();
        assert!(buf.copy_from_slice(&[1.]).is_err());
        assert_eq!(buf.try_clone().unwrap().as_slice(), &[0., 0.]);
    }
}
