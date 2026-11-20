//! The environment variables the crate's tests read. The library itself
//! reads none: a host passes every setting through the API.
//!
//! | Variable | Meaning |
//! |---|---|
//! | `SINTERIC_ASSETS` | Exact model directory for asset-backed tests |
//! | `SINTERIC_OCR_ASSETS` | Imported PP-OCR asset root (`det/`, `rec/`, `uvdoc/`) |
//! | `SINTERIC_TEST_CPU_ISA` | ISA cap for the convolution and OCR tests (`best`, `avx512_core_bf16`, `avx2`); AVX512_CORE_BF16 when unset |
//! | `PPOCR_TEST_THREADS` | Workers for the isolated OCR primitive identity tests |
//! | `EG2_IDENT_THREADS`, `EG2_IDENT_ISA`, `EG2_IDENT_PRECISION` | The EmbeddingGemma 2 identity child's threads, `native` or `avx512`, `bf16` or `fp32` |
//! | `EG2_GEMM_THREADS`, `EG2_GEMM_ISA` | The GEMM identity child's threads and ISA |

use std::ffi::OsString;
use std::path::PathBuf;

use crate::cpu::onednn::CpuIsaCap;

/// One read of every test variable. A malformed value fails the test.
#[derive(Debug, Default)]
pub(crate) struct TestOverrides {
    pub(crate) assets: Option<PathBuf>,
    pub(crate) ocr_assets: Option<PathBuf>,
    pub(crate) conv_isa: Option<CpuIsaCap>,
    pub(crate) ocr_threads: Option<usize>,
    pub(crate) ident_threads: Option<usize>,
    pub(crate) ident_avx512: Option<bool>,
    /// `EG2_IDENT_PRECISION=fp32` (`bf16` is false).
    pub(crate) ident_fp32: Option<bool>,
    pub(crate) gemm_threads: Option<usize>,
    pub(crate) gemm_avx512: Option<bool>,
}

impl TestOverrides {
    pub(crate) fn from_env() -> Result<Self, String> {
        let threads = |v: &str| {
            v.parse::<usize>()
                .ok()
                .filter(|v| (1..=1024).contains(v))
                .ok_or_else(|| "expected 1..=1024".to_string())
        };
        let isa = |v: &str| match v {
            "native" => Ok(false),
            "avx512" => Ok(true),
            _ => Err("expected native|avx512".to_string()),
        };
        Ok(Self {
            assets: path("SINTERIC_ASSETS")?,
            ocr_assets: path("SINTERIC_OCR_ASSETS")?,
            conv_isa: scalar("SINTERIC_TEST_CPU_ISA", |v| {
                CpuIsaCap::from_name(v).map_err(|e| e.to_string())
            })?,
            ocr_threads: scalar("PPOCR_TEST_THREADS", threads)?,
            ident_threads: scalar("EG2_IDENT_THREADS", threads)?,
            ident_avx512: scalar("EG2_IDENT_ISA", isa)?,
            ident_fp32: scalar("EG2_IDENT_PRECISION", |v| match v {
                "fp32" => Ok(true),
                "bf16" => Ok(false),
                _ => Err("expected bf16|fp32".to_string()),
            })?,
            gemm_threads: scalar("EG2_GEMM_THREADS", threads)?,
            gemm_avx512: scalar("EG2_GEMM_ISA", isa)?,
        })
    }
}

/// The cap a convolution or OCR test runs under: `SINTERIC_TEST_CPU_ISA`
/// when set, AVX512_CORE_BF16 otherwise.
pub(crate) fn conv_isa() -> CpuIsaCap {
    TestOverrides::from_env()
        .expect("valid test environment")
        .conv_isa
        .unwrap_or(CpuIsaCap::Avx512CoreBf16)
}

/// The OpenMP variables the thread pool must leave as it found them.
pub(crate) fn thread_environment_snapshot() -> [Option<OsString>; 3] {
    ["GOMP_SPINCOUNT", "OMP_WAIT_POLICY", "OMP_NUM_THREADS"].map(std::env::var_os)
}

fn scalar<T>(
    name: &str,
    parse: impl FnOnce(&str) -> Result<T, String>,
) -> Result<Option<T>, String> {
    std::env::var_os(name)
        .map(|value| {
            let value = value
                .into_string()
                .map_err(|_| format!("{name}: expected Unicode"))?;
            parse(&value).map_err(|e| format!("{name}: {e}"))
        })
        .transpose()
}

fn path(name: &str) -> Result<Option<PathBuf>, String> {
    std::env::var_os(name)
        .map(|value| {
            if value.is_empty() {
                return Err(format!("{name}: expected a nonempty path"));
            }
            Ok(PathBuf::from(value))
        })
        .transpose()
}
