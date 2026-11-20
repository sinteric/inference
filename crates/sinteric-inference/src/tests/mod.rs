//! Tests of the shared size checks, and of the public API as a caller uses
//! it (the submodules).

mod concurrent;
mod ocr;
mod warmup;

use super::*;
#[test]
fn checked_sizes_and_alignment_reject_overflow() {
    assert!(checked_size("test", &[usize::MAX, 2])
        .is_err_and(|e| e.kind() == crate::ErrorKind::TooLarge));
    assert!(align_up(usize::MAX, 64).is_err());
    assert!(align_up(1, 0).is_err());
    assert!(align_up(1, 3).is_err());
    assert!(try_vec::<f32>(usize::MAX).is_err());
    assert!(matrix_span(0, 1, 1, 1, 1).is_err());
    assert!(matrix_span(usize::MAX, 2, 2, usize::MAX, 2).is_err());
}
