//! Where tests find pinned assets (an explicit root, else the user's cache)
//! and whether each file is there at its pinned size. Tests only; never part
//! of the library.

use std::fs;
use std::path::{Path, PathBuf};

use crate::assets::Manifest;
use crate::{Error, Result};

pub(crate) fn asset_root(manifest: &Manifest) -> Option<AssetRoot> {
    let env = crate::test_support::env::TestOverrides::from_env().expect("valid test environment");
    if let Some(root) = env.assets {
        return Some(AssetRoot::new(root));
    }
    let Some(home) = std::env::var_os("HOME") else {
        eprintln!("skipping asset-backed test: set SINTERIC_ASSETS or HOME");
        return None;
    };
    Some(
        AssetRoot::in_cache(PathBuf::from(home).join(".cache/sinteric-assets"), manifest)
            .expect("valid pinned manifest"),
    )
}

/// The directory that holds one manifest's files.
#[derive(Debug, Clone)]
pub(crate) struct AssetRoot {
    pub root: PathBuf,
}

impl AssetRoot {
    /// Exactly this directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `<model>@<first 12 revision bytes>` under `cache_root`.
    pub fn in_cache(cache_root: impl AsRef<Path>, manifest: &Manifest) -> Result<Self> {
        for (name, value) in [("model", &manifest.model), ("revision", &manifest.revision)] {
            if value.is_empty()
                || !value.is_ascii()
                || value == "."
                || value == ".."
                || value.contains(['/', '\\', '\0'])
            {
                return Err(Error::model_files(format!(
                    "invalid manifest {name} for asset directory"
                )));
            }
        }
        Ok(Self::new(cache_root.as_ref().join(format!(
            "{}@{}",
            manifest.model,
            &manifest.revision[..12.min(manifest.revision.len())]
        ))))
    }

    pub fn path_to(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
}

/// Outcome of a required-asset check. Tests use [`skip_if_missing`].
#[derive(Debug)]
pub(crate) enum AssetStatus {
    Ready(PathBuf),
    Missing { path: PathBuf, reason: String },
}

pub(crate) fn locate(manifest: &Manifest, root: &AssetRoot, rel: &str) -> Result<AssetStatus> {
    let file = manifest
        .file(rel)
        .ok_or_else(|| Error::model_files(format!("{rel} not in manifest")))?;
    let path = root.path_to(rel);
    if !path.is_file() {
        return Ok(AssetStatus::Missing {
            reason: format!("absent (looked at {})", path.display()),
            path,
        });
    }
    let meta = fs::metadata(&path)?;
    if meta.len() != file.size {
        return Ok(AssetStatus::Missing {
            reason: format!("size {} != pinned {}", meta.len(), file.size),
            path,
        });
    }
    Ok(AssetStatus::Ready(path))
}

/// For tests: return `true` if the caller should `return` (skip). Prints why.
pub(crate) fn skip_if_missing(status: &AssetStatus) -> bool {
    match status {
        AssetStatus::Ready(_) => false,
        AssetStatus::Missing { path, reason } => {
            eprintln!("skipping asset-backed test: {} ({reason})", path.display());
            true
        }
    }
}
