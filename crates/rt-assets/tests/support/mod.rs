//! Asset-cache defaults for repository tests only; never part of the library.
use rt_assets::{AssetRoot, Manifest};

pub fn asset_root(manifest: &Manifest) -> Option<AssetRoot> {
    let env = rt_core::env::EnvOverrides::from_env().expect("valid test environment overrides");
    if let Some(root) = env.assets {
        return Some(AssetRoot::new(root));
    }
    let Some(home) = std::env::var_os("HOME") else {
        eprintln!("skipping asset-backed test: set SINTERIC_ASSETS or HOME");
        return None;
    };
    Some(
        AssetRoot::in_cache(
            std::path::PathBuf::from(home).join(".cache/sinteric-assets"),
            manifest,
        )
        .expect("valid pinned manifest"),
    )
}
