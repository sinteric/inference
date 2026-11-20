//! Tests of the runtime. The submodules were integration tests, each in its
//! own process.

mod amx_bf16_race_repro;
mod amx_bf16_race_sweep;

use super::*;

#[test]
fn runtime_reports_its_isa_threads_and_cpus() {
    crate::test_support::own_process!(runtime_reports_its_isa_threads_and_cpus);
    let rt = Runtime::try_new(Options::new().with_threads(2)).unwrap();
    assert_eq!(rt.threads(), 2);
    assert_eq!(rt.cpus().len(), 2);
    // The library's floor: `Runtime::try_new` refuses anything below it.
    assert!(rt.isa().bits() & Isa::Avx2.bits() == Isa::Avx2.bits());
    assert_eq!(rt.options().threads, 2);
}

#[test]
fn a_capped_runtime_runs_at_most_its_cap() {
    crate::test_support::own_process!(a_capped_runtime_runs_at_most_its_cap);
    let rt = Runtime::try_new(Options::new().with_threads(1).with_isa_cap(IsaCap::Avx2)).unwrap();
    assert_eq!(rt.isa(), Isa::Avx2);
    assert!(!rt.isa().has_bf16_matmul());
    assert_eq!(rt.amx_health(), AmxHealth::NotProbed);
}

#[test]
fn a_shared_backend_handle_owns_its_settings() {
    crate::test_support::own_process!(a_shared_backend_handle_owns_its_settings);
    let rt = Runtime::try_new(Options::new().with_threads(1)).unwrap();
    assert!(!rt.shares_backend());
    let shared = rt.shared_backend(BackendOverrides::default()).unwrap();
    assert!(shared.shares_backend());
    let again = shared
        .shared_backend(BackendOverrides::default())
        .unwrap_err();
    assert_eq!(again.kind(), ErrorKind::Config);
    assert!(shared
        .backend_with(BackendOverrides::default().with_threads(Some(1)))
        .is_err_and(|e| e.kind() == ErrorKind::Config));
    assert!(Arc::ptr_eq(
        &shared.backend().unwrap(),
        &shared.backend().unwrap()
    ));
}
