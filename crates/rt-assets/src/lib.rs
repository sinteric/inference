//! Pinned-asset manifest + verified loader.
//!
//! A pin names a hub repo, revision, and per file its path, sha256 and
//! licence. Bytes live outside git. Tests that need assets skip
//! cleanly when the files are absent.

#[cfg(test)]
extern crate self as rt_assets;

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

pub use rt_core::{Error, Result};

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
            toml::from_str(text).map_err(|e| Error::Asset(format!("manifest parse: {e}")))?;
        if !manifest.revision.is_ascii() {
            return Err(Error::Asset("revision must be ASCII".into()));
        }
        Ok(manifest)
    }

    pub fn file(&self, path: &str) -> Option<&PinnedFile> {
        self.files.iter().find(|f| f.path == path)
    }
}

/// Where on disk the verified bytes live.
#[derive(Debug, Clone)]
pub struct AssetRoot {
    pub root: PathBuf,
}

impl AssetRoot {
    /// Exact directory containing the model files. Never reads environment.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve `<model>@<first 12 revision bytes>` under a caller-owned cache root.
    /// The caller chooses the cache root; there is no HOME or temporary fallback.
    pub fn in_cache(cache_root: impl AsRef<Path>, manifest: &Manifest) -> Result<Self> {
        for (name, value) in [("model", &manifest.model), ("revision", &manifest.revision)] {
            if value.is_empty()
                || !value.is_ascii()
                || value == "."
                || value == ".."
                || value.contains(['/', '\\', '\0'])
            {
                return Err(Error::Asset(format!(
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

    pub fn present(&self, file: &PinnedFile) -> bool {
        let p = self.path_to(&file.path);
        p.is_file() && p.metadata().map(|m| m.len() == file.size).unwrap_or(false)
    }
}

/// Outcome of a required-asset check. Tests use [`skip_if_missing`].
#[derive(Debug)]
pub enum AssetStatus {
    Ready(PathBuf),
    Missing { path: PathBuf, reason: String },
}

pub fn locate(manifest: &Manifest, root: &AssetRoot, rel: &str) -> Result<AssetStatus> {
    let file = manifest
        .file(rel)
        .ok_or_else(|| Error::Asset(format!("{rel} not in manifest")))?;
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

/// Verify sha256 of a present file. Does not read more than `size` bytes.
pub fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    verify_sha256_mmap(path, expected)
}

/// Mmap the file and hash the mapped bytes (no extra copy of the tensor file).
pub fn verify_sha256_mmap(path: &Path, expected: &str) -> Result<()> {
    let f = fs::File::open(path)?;
    let map = unsafe { memmap2::MmapOptions::new().map(&f) }?;
    let mut hasher = Sha256::new();
    hasher.update(&map[..]);
    let got = hex::encode(hasher.finalize());
    if !got.eq_ignore_ascii_case(expected) {
        return Err(Error::Asset(format!(
            "sha256 mismatch for {}: got {got} want {expected}",
            path.display()
        )));
    }
    Ok(())
}

/// Expected sha256 of `rel` in a loaded manifest.
pub fn expected_sha256(manifest: &Manifest, rel: &str) -> Result<String> {
    manifest
        .file(rel)
        .map(|f| f.sha256.clone())
        .ok_or_else(|| Error::Asset(format!("{rel} not in manifest")))
}

/// For tests: return `true` if the caller should `return` (skip). Prints why.
pub fn skip_if_missing(status: &AssetStatus) -> bool {
    match status {
        AssetStatus::Ready(_) => false,
        AssetStatus::Missing { path, reason } => {
            eprintln!("skipping asset-backed test: {} ({reason})", path.display());
            true
        }
    }
}

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_support;

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
model = "vela2-0p3b"
source = "huggingface"
repo = "vllm-sr/Vela-2.0-0.3B"
revision = "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409"
license = "Apache-2.0"

[[files]]
path = "model.safetensors"
sha256 = "bd37dd0db177d766bc554eded78ae8493bc248172d0480f6a543b30932a0cc8e"
size = 1236473252
lfs = true
"#;

    #[test]
    fn explicit_roots_and_invalid_cache_components() {
        let mut m = Manifest::parse(SAMPLE).unwrap();
        let root = AssetRoot::in_cache("caller/cache", &m).unwrap();
        assert_eq!(
            root.root,
            PathBuf::from("caller/cache/vela2-0p3b@d6f03aa9baca")
        );
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
            Manifest::parse(&SAMPLE.replace("d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409", "한"))
                .is_err()
        );
    }

    #[test]
    fn parse_manifest() {
        let m: Manifest = toml::from_str(SAMPLE).unwrap();
        assert_eq!(m.repo, "vllm-sr/Vela-2.0-0.3B");
        assert_eq!(m.revision.len(), 40);
        assert_eq!(m.files[0].size, 1_236_473_252);
    }

    #[test]
    fn missing_asset_skips() {
        let m: Manifest = toml::from_str(SAMPLE).unwrap();
        let Some(root) = crate::test_support::asset_root(&m) else {
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
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/vela2-0p3b.toml");
        let m = Manifest::load(path).expect("committed asset manifest");
        assert_eq!(m.revision, "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409");
        let Some(root) = crate::test_support::asset_root(&m) else {
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
        let dir = std::env::temp_dir().join("sinteric-rt-assets-test");
        let _ = fs::create_dir_all(&dir);
        let p = dir.join("hello.bin");
        fs::write(&p, b"hello").unwrap();
        verify_sha256(
            &p,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        )
        .unwrap();
        assert!(verify_sha256(&p, "00").is_err());
    }
}
