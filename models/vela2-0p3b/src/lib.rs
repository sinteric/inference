//! Vela 2.0 0.3B CPU runtime.
//!
//! Production path: Rust graph + oneDNN 3.13.4 C API ([`onednn`]).
//! Candle/MLAS comparisons live in the separate `bench/vela-mlas` crate.
//! [`api`] is the public, runtime-loaded surface ([`api::Vela2`]).

/// Pinned asset identity embedded at build time; no source-tree path is needed at runtime.
#[cfg(test)]
#[path = "../../../crates/rt-assets/tests/support/mod.rs"]
mod asset_test_support;

pub const ASSET_MANIFEST: &str = include_str!("../../../assets/vela2-0p3b.toml");

pub mod api;
pub mod onednn;

pub use rt_core::{Error, Result, DEFAULT_SCRATCH_CAP_BYTES, VELA2_PACKED_WEIGHT_BYTES};
pub use rt_text::{system_preparation, Frontend, Inputs, Logits, SystemOneError};

/// Bytes a caller's model cache reserves for one session: the packed weights
/// plus the default scratch cap.
pub fn reservation_bytes() -> usize {
    VELA2_PACKED_WEIGHT_BYTES + DEFAULT_SCRATCH_CAP_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_weight_constant() {
        assert_eq!(VELA2_PACKED_WEIGHT_BYTES, 112_461_312 * 4);
        assert_eq!(
            reservation_bytes(),
            VELA2_PACKED_WEIGHT_BYTES + 64 * 1024 * 1024
        );
    }

    #[test]
    fn load_skips_when_weights_or_onednn_absent() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/vela2-0p3b.toml");
        let m = rt_assets::Manifest::load(path).expect("committed asset manifest");
        let Some(root) = crate::asset_test_support::asset_root(&m) else {
            return;
        };
        let st = rt_assets::locate(&m, &root, "model.safetensors").unwrap();
        if rt_assets::skip_if_missing(&st) {
            return;
        }
        let rt_assets::AssetStatus::Ready(p) = st else {
            panic!("expected ready");
        };
        let dir = p.parent().unwrap();
        let budget = rt_threads::ThreadBudget::new(1).unwrap();
        let backend = rt_onednn::OnednnCpu::load(&budget).expect("libdnnl");
        onednn::model::Model::load(dir, backend, 64).expect("weights present; oneDNN load");
    }
}
