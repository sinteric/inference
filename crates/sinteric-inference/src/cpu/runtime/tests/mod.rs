//! Tests of the runtime. The submodules were integration tests, each in its
//! own process.

mod amx_bf16_race_repro;
mod amx_bf16_race_sweep;

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

static DROPPED: AtomicBool = AtomicBool::new(false);

struct Dummy(#[allow(dead_code)] Vec<u8>);
impl Drop for Dummy {
    fn drop(&mut self) {
        DROPPED.store(true, Ordering::SeqCst);
    }
}
impl Model for Dummy {
    type Config = usize;
    const PRECISIONS: &'static [Precision] = &[Precision::Fp32];
    fn load(_: &Runtime, bytes: usize) -> Result<Self> {
        Ok(Self(vec![1; bytes]))
    }
    fn precision(&self) -> Precision {
        Precision::Fp32
    }
}

#[test]
fn isa_names() {
    assert_eq!(cpu_isa_name(0xe7), "AVX512_CORE_BF16");
    assert_eq!(cpu_isa_name(0xfef), "AVX10_1_512_AMX");
    assert_eq!(cpu_isa_name(0x12345), "UNKNOWN");
}

#[test]
fn unload_drops_model_and_shared_model_unloads_on_last_owner() {
    let shared = Arc::new(Dummy(vec![1; 64 << 20]));
    let other = Arc::clone(&shared);
    assert!(Arc::into_inner(shared).is_none());
    assert!(!DROPPED.load(Ordering::SeqCst));
    if let Some(m) = Arc::into_inner(other) {
        m.unload();
    }
    assert!(DROPPED.load(Ordering::SeqCst));
    release_free_memory();
}
