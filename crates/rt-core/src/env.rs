//! Explicit, validated process-environment snapshots.
//!
//! Only [`EnvOverrides::from_env`] reads production runtime overrides. Take one
//! snapshot at application startup and pass it to configuration `with_overrides`
//! methods. Constructors and model loaders never call it implicitly. Unset
//! values leave the caller's options unchanged; present values override them.
//!
//! | Variable | Type / accepted values | Meaning | Default when unset |
//! |---|---|---|---|
//! | `ONEDNN_MAX_CPU_ISA`, `DNNL_MAX_CPU_ISA` | `best_available`, `default`, `best`, `AVX512_CORE_BF16`, `AVX2` (case insensitive) | Process ISA cap; first name wins | Keep config (best available) |
//! | `DNNL_VERBOSE`, `ONEDNN_VERBOSE` | integer `0..=2` | oneDNN verbosity; first name wins | Keep config (0) |
//! | `VELA_PACKED_CACHE` | boolean | Packed cache for either model | Keep config (true) |
//! | `VELA_PACKED_CACHE_FILE` | nonempty OS path | Explicit packed cache file for either model | Keep config (sidecar) |
//! | `EG2_VERIFY_WEIGHTS`, `VELA_VERIFY_WEIGHTS` | boolean, also `full` for true | Verify the respective model's full weight SHA256 | Keep config (false) |
//! | `EG2_MANIFEST`, `VELA_MANIFEST` | nonempty OS path | Respective model's manifest | Keep config (embedded pin) |
//! | `SINTERIC_ASSETS` | nonempty OS path | Exact model directory / asset root, not a cache parent | Keep caller's model directory |
//! | `VELA_STANDALONE_PROFILE` | boolean | Vela CLI profiling only | Off |
//!
//! Booleans accept `1/0`, `true/false`, `on/off`, case insensitive. Empty,
//! malformed, out-of-range and non-Unicode scalar values return
//! [`crate::Error::Environment`]. Paths preserve non-Unicode OS strings.
//! Both aliases are validated even if the first takes precedence. No filesystem
//! access or HOME lookup occurs here. Build-time variables are unrelated.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::{Error, Result};

/// ISA names accepted by the environment parser, independent of a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuIsaOverride {
    BestAvailable,
    Avx512CoreBf16,
    Avx2,
}

impl CpuIsaOverride {
    pub fn from_name(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" | "best" | "best_available" => Ok(Self::BestAvailable),
            "avx512_core_bf16" => Ok(Self::Avx512CoreBf16),
            "avx2" => Ok(Self::Avx2),
            _ => Err(Error::Config(format!(
                "unknown CPU ISA cap {value:?}; supported: best_available, AVX512_CORE_BF16, AVX2"
            ))),
        }
    }
}

/// Model-specific overrides; one model's variables never affect the other.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelEnvOverrides {
    pub verify_weights: Option<bool>,
    pub manifest: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    EmbeddingGemma2,
    Vela,
}

/// A reusable snapshot. See the module table for names, types and defaults.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct EnvOverrides {
    pub max_cpu_isa: Option<CpuIsaOverride>,
    pub verbose: Option<i32>,
    pub packed_cache: Option<bool>,
    pub packed_cache_file: Option<PathBuf>,
    pub assets: Option<PathBuf>,
    pub eg2: ModelEnvOverrides,
    pub vela: ModelEnvOverrides,
    /// Consumed only by the standalone Vela CLI; model loading stays explicit.
    pub vela_standalone_profile: Option<bool>,
}

impl EnvOverrides {
    /// Read every supported production override exactly once into this snapshot.
    /// Calling again deliberately captures a new snapshot; no global cache exists.
    pub fn from_env() -> Result<Self> {
        Self::from_lookup(|name| std::env::var_os(name))
    }

    /// Parse an application-provided environment without accessing process state.
    /// Unknown names are never queried. Useful for deterministic configuration
    /// and tests without mutating the process environment.
    pub fn from_lookup(mut lookup: impl FnMut(&str) -> Option<OsString>) -> Result<Self> {
        let isa = scalar(&mut lookup, "ONEDNN_MAX_CPU_ISA", CpuIsaOverride::from_name)?;
        let isa_alias = scalar(&mut lookup, "DNNL_MAX_CPU_ISA", CpuIsaOverride::from_name)?;
        let verbose = scalar(&mut lookup, "DNNL_VERBOSE", parse_verbose)?;
        let verbose_alias = scalar(&mut lookup, "ONEDNN_VERBOSE", parse_verbose)?;
        Ok(Self {
            max_cpu_isa: isa.or(isa_alias),
            verbose: verbose.or(verbose_alias),
            packed_cache: scalar(&mut lookup, "VELA_PACKED_CACHE", parse_bool)?,
            packed_cache_file: path(&mut lookup, "VELA_PACKED_CACHE_FILE")?,
            assets: path(&mut lookup, "SINTERIC_ASSETS")?,
            eg2: ModelEnvOverrides {
                verify_weights: scalar(&mut lookup, "EG2_VERIFY_WEIGHTS", parse_verify)?,
                manifest: path(&mut lookup, "EG2_MANIFEST")?,
            },
            vela: ModelEnvOverrides {
                verify_weights: scalar(&mut lookup, "VELA_VERIFY_WEIGHTS", parse_verify)?,
                manifest: path(&mut lookup, "VELA_MANIFEST")?,
            },
            vela_standalone_profile: scalar(&mut lookup, "VELA_STANDALONE_PROFILE", parse_bool)?,
        })
    }

    pub fn model(&self, model: ModelKind) -> &ModelEnvOverrides {
        match model {
            ModelKind::EmbeddingGemma2 => &self.eg2,
            ModelKind::Vela => &self.vela,
        }
    }
}

fn invalid(name: &'static str, reason: impl Into<String>) -> Error {
    Error::Environment {
        name,
        reason: reason.into(),
    }
}

fn scalar<T>(
    lookup: &mut impl FnMut(&str) -> Option<OsString>,
    name: &'static str,
    parse: impl FnOnce(&str) -> Result<T>,
) -> Result<Option<T>> {
    lookup(name)
        .map(|value| {
            let value = value
                .into_string()
                .map_err(|_| invalid(name, "expected Unicode scalar value"))?;
            parse(&value).map_err(|e| invalid(name, e.to_string()))
        })
        .transpose()
}

fn path(
    lookup: &mut impl FnMut(&str) -> Option<OsString>,
    name: &'static str,
) -> Result<Option<PathBuf>> {
    lookup(name)
        .map(|value| {
            if value.is_empty() || value.as_encoded_bytes().contains(&0) {
                return Err(invalid(name, "expected nonempty path without NUL"));
            }
            Ok(PathBuf::from(value))
        })
        .transpose()
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(Error::Config(
            "expected boolean: 1/0, true/false, on/off".into(),
        )),
    }
}
fn parse_verify(value: &str) -> Result<bool> {
    if value.eq_ignore_ascii_case("full") {
        Ok(true)
    } else {
        parse_bool(value)
    }
}
fn parse_verbose(value: &str) -> Result<i32> {
    value
        .parse::<i32>()
        .ok()
        .filter(|v| (0..=2).contains(v))
        .ok_or_else(|| Error::Config("expected verbosity integer 0..=2".into()))
}

/// Internal subprocess-test controls. These are not production overrides.
/// `EG2_{IDENT,GEMM}_THREADS` accepts 1..=1024 and the corresponding `_ISA`
/// accepts `native` or `avx512`. MLAS tests use explicit qualified library paths
/// and `CANDLE_TEST_PARTITION=rows|auto`. Read only by test harnesses.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct TestOverrides {
    /// Isolated OCR primitive identity workers: PPOCR_TEST_THREADS.
    pub ocr_threads: Option<usize>,
    /// ISA cap for the convolution and OCR tests: SINTERIC_TEST_CPU_ISA, any
    /// name `ONEDNN_MAX_CPU_ISA` accepts. Unset keeps AVX512_CORE_BF16.
    pub conv_isa: Option<CpuIsaOverride>,
    /// Explicit imported det/rec asset root: SINTERIC_OCR_ASSETS.
    pub ocr_assets: Option<PathBuf>,
    pub ident_threads: Option<usize>,
    pub ident_avx512: Option<bool>,
    /// EG2 identity child precision: EG2_IDENT_PRECISION=bf16|fp32 (true = fp32).
    pub ident_fp32: Option<bool>,
    pub gemm_threads: Option<usize>,
    pub gemm_avx512: Option<bool>,
    pub vela_mlas_library: Option<PathBuf>,
    pub candle_mlas_library: Option<PathBuf>,
    pub candle_partition_auto: Option<bool>,
}
impl TestOverrides {
    pub fn from_env() -> Result<Self> {
        let mut lookup = |name: &str| std::env::var_os(name);
        let threads = |v: &str| {
            v.parse::<usize>()
                .ok()
                .filter(|v| (1..=1024).contains(v))
                .ok_or_else(|| Error::Config("expected test threads 1..=1024".into()))
        };
        let isa = |v: &str| match v {
            "native" => Ok(false),
            "avx512" => Ok(true),
            _ => Err(Error::Config("expected test ISA native|avx512".into())),
        };
        Ok(Self {
            ocr_threads: scalar(&mut lookup, "PPOCR_TEST_THREADS", threads)?,
            conv_isa: scalar(
                &mut lookup,
                "SINTERIC_TEST_CPU_ISA",
                CpuIsaOverride::from_name,
            )?,
            ocr_assets: path(&mut lookup, "SINTERIC_OCR_ASSETS")?,
            ident_threads: scalar(&mut lookup, "EG2_IDENT_THREADS", threads)?,
            ident_avx512: scalar(&mut lookup, "EG2_IDENT_ISA", isa)?,
            ident_fp32: scalar(&mut lookup, "EG2_IDENT_PRECISION", |v| match v {
                "fp32" => Ok(true),
                "bf16" => Ok(false),
                _ => Err(Error::Config("expected test precision bf16|fp32".into())),
            })?,
            gemm_threads: scalar(&mut lookup, "EG2_GEMM_THREADS", threads)?,
            gemm_avx512: scalar(&mut lookup, "EG2_GEMM_ISA", isa)?,
            vela_mlas_library: path(&mut lookup, "VELA_TEST_MLAS_LIBRARY")?,
            candle_mlas_library: path(&mut lookup, "CANDLE_TEST_MLAS_LIBRARY")?,
            candle_partition_auto: scalar(&mut lookup, "CANDLE_TEST_PARTITION", |v| match v {
                "auto" => Ok(true),
                "rows" => Ok(false),
                _ => Err(Error::Config("expected test partition rows|auto".into())),
            })?,
        })
    }
}

/// Snapshot used only to assert that thread installation does not mutate env.
#[doc(hidden)]
pub fn thread_environment_snapshot() -> [Option<OsString>; 3] {
    ["GOMP_SPINCOUNT", "OMP_WAIT_POLICY", "OMP_NUM_THREADS"].map(std::env::var_os)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn parse(values: &[(&str, &str)]) -> Result<EnvOverrides> {
        EnvOverrides::from_lookup(|name| {
            values
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| OsString::from(v))
        })
    }
    #[test]
    fn absent_overrides_are_empty_and_each_name_is_read_once() {
        let mut names = std::collections::HashSet::new();
        let env = EnvOverrides::from_lookup(|name| {
            assert!(names.insert(name.to_owned()));
            None
        })
        .unwrap();
        assert_eq!(env, EnvOverrides::default());
        assert_eq!(names.len(), 12);
    }
    #[test]
    fn typed_values_alias_precedence_and_model_isolation() {
        let env = parse(&[
            ("ONEDNN_MAX_CPU_ISA", "avx512_core_bf16"),
            ("DNNL_MAX_CPU_ISA", "default"),
            ("DNNL_VERBOSE", "2"),
            ("ONEDNN_VERBOSE", "1"),
            ("VELA_PACKED_CACHE", "OFF"),
            ("EG2_VERIFY_WEIGHTS", "full"),
            ("VELA_VERIFY_WEIGHTS", "false"),
            ("EG2_MANIFEST", "eg2.toml"),
            ("VELA_MANIFEST", "vela.toml"),
            ("VELA_PACKED_CACHE_FILE", "cache/packed"),
            ("SINTERIC_ASSETS", "models/eg2"),
        ])
        .unwrap();
        assert_eq!(env.max_cpu_isa, Some(CpuIsaOverride::Avx512CoreBf16));
        assert_eq!(env.verbose, Some(2));
        assert_eq!(env.packed_cache, Some(false));
        assert_eq!(env.eg2.verify_weights, Some(true));
        assert_eq!(env.vela.verify_weights, Some(false));
        assert_eq!(env.eg2.manifest, Some("eg2.toml".into()));
        assert_eq!(env.vela.manifest, Some("vela.toml".into()));
        assert_eq!(env.assets, Some("models/eg2".into()));
        assert_eq!(
            parse(&[("DNNL_MAX_CPU_ISA", "AVX2")]).unwrap().max_cpu_isa,
            Some(CpuIsaOverride::Avx2)
        );
        assert_eq!(
            parse(&[("DNNL_MAX_CPU_ISA", "best"), ("ONEDNN_VERBOSE", "1")])
                .unwrap()
                .verbose,
            Some(1)
        );
    }
    #[test]
    fn invalid_values_are_named_errors_even_for_shadowed_aliases() {
        for (name, value) in [
            ("ONEDNN_MAX_CPU_ISA", ""),
            ("DNNL_MAX_CPU_ISA", "avx512_fp16"),
            ("DNNL_VERBOSE", "loud"),
            ("ONEDNN_VERBOSE", "3"),
            ("DNNL_VERBOSE", "-1"),
            ("VELA_PACKED_CACHE", "maybe"),
            ("VELA_VERIFY_WEIGHTS", ""),
            ("EG2_VERIFY_WEIGHTS", "yes"),
            ("VELA_STANDALONE_PROFILE", "-1"),
            ("SINTERIC_ASSETS", ""),
            ("EG2_MANIFEST", ""),
            ("VELA_MANIFEST", "\0"),
            ("VELA_PACKED_CACHE_FILE", ""),
        ] {
            assert!(
                matches!(parse(&[(name, value)]), Err(Error::Environment { name: n, .. }) if n == name)
            );
        }
        assert!(parse(&[("DNNL_VERBOSE", "0"), ("ONEDNN_VERBOSE", "bad")]).is_err());
        assert!(parse(&[
            ("ONEDNN_MAX_CPU_ISA", "default"),
            ("DNNL_MAX_CPU_ISA", "bad")
        ])
        .is_err());
    }
    #[cfg(unix)]
    #[test]
    fn os_paths_preserved_and_non_unicode_scalars_rejected() {
        use std::os::unix::ffi::OsStringExt;
        let raw = OsString::from_vec(vec![b'/', 0xff]);
        let env =
            EnvOverrides::from_lookup(|name| (name == "SINTERIC_ASSETS").then(|| raw.clone()))
                .unwrap();
        assert_eq!(env.assets, Some(PathBuf::from(&raw)));
        assert!(matches!(
            EnvOverrides::from_lookup(|name| (name == "DNNL_VERBOSE").then(|| raw.clone())),
            Err(Error::Environment {
                name: "DNNL_VERBOSE",
                ..
            })
        ));
    }
}
