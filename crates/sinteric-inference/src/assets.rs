//! Pinned-asset manifests and the sha256 check of a model file.
//!
//! A pin names a hub repo, revision, and per file its path, sha256 and
//! licence. Bytes live outside git; where tests find them is
//! `test_support::assets`.

use crate::{Error, Result};

use std::fs;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// One file inside a [`Manifest`].
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct PinnedFile {
    pub path: String,
    pub sha256: String,
    pub size: u64,
    #[serde(default)]
    pub lfs: bool,
    #[serde(default)]
    pub license: Option<String>,
}

/// Pinned identity of a model (or fixture set).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub model: String,
    pub source: String,
    pub repo: String,
    pub revision: String,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub notes: Option<String>,
    pub files: Vec<PinnedFile>,
}

impl Manifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|e| Error::from(e).at(path))?;
        Self::parse(&text).map_err(|e| e.at(path))
    }

    /// Parse an embedded or caller-provided manifest without filesystem access.
    pub fn parse(text: &str) -> Result<Self> {
        let manifest: Self =
            toml::from_str(text).map_err(|e| Error::model_files(format!("manifest parse: {e}")))?;
        if !manifest.revision.is_ascii() {
            return Err(Error::model_files("revision must be ASCII"));
        }
        Ok(manifest)
    }

    pub fn file(&self, path: &str) -> Option<&PinnedFile> {
        self.files.iter().find(|f| f.path == path)
    }
}

/// Mmap the file and hash the mapped bytes (no extra copy of the tensor file).
pub fn verify_sha256_mmap(path: &Path, expected: &str) -> Result<()> {
    let f = fs::File::open(path)?;
    // SAFETY: model files must remain immutable while mapped
    // (docs/library.md), and the map is only hashed, then dropped.
    let map = unsafe { memmap2::MmapOptions::new().map(&f) }?;
    let mut hasher = Sha256::new();
    hasher.update(&map[..]);
    let got = hex::encode(hasher.finalize());
    if !got.eq_ignore_ascii_case(expected) {
        return Err(Error::model_files(format!(
            "sha256 mismatch for {}: got {got} want {expected}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::test_support::assets::{locate, skip_if_missing, AssetRoot, AssetStatus};

    const SAMPLE: &str = r#"
model = "guard"
source = "huggingface"
repo = "Horizon-Labs/prompt-injection-guard-base"
revision = "fc071c5754f8cf2edce9dc1d7fd3513ef9287007"
license = "Apache-2.0"

[[files]]
path = "model.safetensors"
sha256 = "d0c0f899f118481855cfe34ba718cf5e44613d266270721c380d54741626d405"
size = 1230141424
lfs = true
"#;

    #[test]
    fn explicit_roots_and_invalid_cache_components() {
        let mut m = Manifest::parse(SAMPLE).unwrap();
        let root = AssetRoot::in_cache("caller/cache", &m).unwrap();
        assert_eq!(root.root, PathBuf::from("caller/cache/guard@fc071c5754f8"));
        let root = AssetRoot::new("caller/exact");
        assert_eq!(
            root.path_to("tokenizer.json"),
            PathBuf::from("caller/exact/tokenizer.json")
        );
        assert!(
            matches!(locate(&m, &root, "model.safetensors").unwrap(), AssetStatus::Missing { path, .. } if path == Path::new("caller/exact/model.safetensors"))
        );
        assert!(locate(&m, &root, "not-in-manifest").is_err());
        for bad in [
            "",
            "..",
            "../escape",
            "/absolute",
            "a/b",
            "a\\b",
            "a\0b",
            "한",
        ] {
            m.model = bad.into();
            assert!(AssetRoot::in_cache("caller/cache", &m).is_err(), "{bad:?}");
            m.model = "valid".into();
            m.revision = bad.into();
            assert!(AssetRoot::in_cache("caller/cache", &m).is_err(), "{bad:?}");
            m.revision = "abc".into();
        }
        assert!(
            Manifest::parse(&SAMPLE.replace("fc071c5754f8cf2edce9dc1d7fd3513ef9287007", "한"))
                .is_err()
        );
    }

    #[test]
    fn parse_manifest() {
        let m: Manifest = toml::from_str(SAMPLE).unwrap();
        assert_eq!(m.repo, "Horizon-Labs/prompt-injection-guard-base");
        assert_eq!(m.revision.len(), 40);
        assert_eq!(m.files[0].size, 1_230_141_424);
    }

    #[test]
    fn missing_asset_skips() {
        let m: Manifest = toml::from_str(SAMPLE).unwrap();
        let Some(root) = crate::test_support::assets::asset_root(&m) else {
            return;
        };
        let st = locate(&m, &root, "model.safetensors").unwrap();
        // On a clean CI machine the 1.2 GiB weights are not present.
        match st {
            AssetStatus::Missing { .. } => assert!(skip_if_missing(&st)),
            AssetStatus::Ready(ref p) => {
                assert!(p.is_file());
                assert!(!skip_if_missing(&st));
            }
        }
    }

    #[test]
    fn repo_manifest_skips_when_weights_absent() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/guard.toml");
        let m = Manifest::load(path).expect("committed asset manifest");
        assert_eq!(m.revision, "fc071c5754f8cf2edce9dc1d7fd3513ef9287007");
        let Some(root) = crate::test_support::assets::asset_root(&m) else {
            return;
        };
        let st = locate(&m, &root, "model.safetensors").unwrap();
        match &st {
            AssetStatus::Missing { .. } => assert!(skip_if_missing(&st)),
            AssetStatus::Ready(_) => assert!(!skip_if_missing(&st)),
        }
    }

    #[test]
    fn verify_sha256_on_tiny_file() {
        let dir = std::env::temp_dir().join("sinteric-assets-test");
        let _ = fs::create_dir_all(&dir);
        let p = dir.join("hello.bin");
        fs::write(&p, b"hello").unwrap();
        verify_sha256_mmap(
            &p,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        )
        .unwrap();
        assert!(verify_sha256_mmap(&p, "00").is_err());
    }
}
