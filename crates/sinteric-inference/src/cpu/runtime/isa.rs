//! The instruction sets oneDNN dispatches to, the cap a runtime may put on
//! them, and what the AMX tile-state probe found.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize, Serializer};

use crate::cpu::onednn::{AmxHealth as ProbeResult, CpuIsaCap};
use crate::{Error, Result};

/// The instruction set every model on a [`super::Runtime`] runs on: what
/// oneDNN dispatches to on this CPU, after the runtime's [`IsaCap`].
///
/// A later ISA holds everything an earlier one in the same family does; ask
/// what a model needs with [`Isa::has_bf16_matmul`] and [`Isa::has_amx`]
/// rather than comparing variants. Below AVX2 the library does not run
/// ([`super::Runtime::new`] refuses it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Isa {
    Avx2,
    Avx2Vnni,
    Avx2Vnni2,
    Avx512Core,
    Avx512CoreVnni,
    Avx512CoreBf16,
    Avx10_1_512,
    Avx10_1_512Amx,
    Avx10_1_512AmxFp16,
    Avx10_2_512,
    Avx10_2_512Amx2,
}

/// oneDNN's `dnnl_cpu_isa_t` value and name of each [`Isa`].
const ISAS: [(Isa, u64, &str); 11] = [
    (Isa::Avx2, 0x7, "AVX2"),
    (Isa::Avx2Vnni, 0xf, "AVX2_VNNI"),
    (Isa::Avx2Vnni2, 0x1f, "AVX2_VNNI_2"),
    (Isa::Avx512Core, 0x27, "AVX512_CORE"),
    (Isa::Avx512CoreVnni, 0x67, "AVX512_CORE_VNNI"),
    (Isa::Avx512CoreBf16, 0xe7, "AVX512_CORE_BF16"),
    (Isa::Avx10_1_512, 0x1ef, "AVX10_1_512"),
    (Isa::Avx10_1_512Amx, 0xfef, "AVX10_1_512_AMX"),
    (Isa::Avx10_1_512AmxFp16, 0x1fef, "AVX10_1_512_AMX_FP16"),
    (Isa::Avx10_2_512, 0x201ff, "AVX10_2_512"),
    (Isa::Avx10_2_512Amx2, 0x22fff, "AVX10_2_512_AMX_2"),
];

/// `dnnl_cpu_isa_avx512_core_bf16`: BF16 matmul kernels.
const BF16_BITS: u64 = 0xe7;
/// `dnnl_cpu_isa_avx512_core_amx`: AMX tile kernels.
const AMX_BITS: u64 = 0xfef;

impl Isa {
    /// The ISA oneDNN reports as `bits`, or `None` below AVX2 or for a value
    /// this library does not know.
    pub(crate) fn from_bits(bits: u64) -> Option<Self> {
        ISAS.iter()
            .find(|(_, b, _)| *b == bits)
            .map(|(isa, ..)| *isa)
    }

    /// oneDNN's `dnnl_cpu_isa_t` value.
    pub(crate) fn bits(self) -> u64 {
        ISAS.iter()
            .find(|(isa, ..)| *isa == self)
            .map_or(0, |(_, bits, _)| *bits)
    }

    /// oneDNN's name for it (`AVX512_CORE_BF16`, `AVX10_1_512_AMX`, ...).
    pub fn name(self) -> &'static str {
        ISAS.iter()
            .find(|(isa, ..)| *isa == self)
            .map_or("UNKNOWN", |(.., name)| name)
    }

    /// Whether oneDNN has BF16 matmul kernels here (AVX512_CORE_BF16 and
    /// every later ISA). Without them a BF16 matmul runs on oneDNN's
    /// reference path, orders of magnitude slower, so no model offers BF16.
    pub fn has_bf16_matmul(self) -> bool {
        self.bits() & BF16_BITS == BF16_BITS
    }

    /// Whether oneDNN runs AMX tile kernels here.
    pub fn has_amx(self) -> bool {
        self.bits() & AMX_BITS == AMX_BITS
    }
}

impl fmt::Display for Isa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Serialize for Isa {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// The most a [`super::Runtime`] lets oneDNN dispatch to. Process-wide:
/// every runtime in a process must ask for the same cap.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum IsaCap {
    /// The best ISA the CPU has, except that a host whose AMX tile state is
    /// lost across preemption runs AVX512_CORE_BF16 ([`AmxHealth`]).
    #[default]
    Best,
    /// AVX512_CORE_BF16 at most: what an AMD Zen 4 or Zen 5 runs, on an
    /// Intel host with AMX.
    Avx512CoreBf16,
    /// AVX2 at most: the library's floor, on a host with more.
    Avx2,
}

impl IsaCap {
    pub(crate) fn backend(self) -> CpuIsaCap {
        match self {
            Self::Best => CpuIsaCap::BestAvailable,
            Self::Avx512CoreBf16 => CpuIsaCap::Avx512CoreBf16,
            Self::Avx2 => CpuIsaCap::Avx2,
        }
    }
}

/// Parses `best` (or `best_available`, `default`), `avx512_core_bf16` or
/// `avx2`, in any case.
impl FromStr for IsaCap {
    type Err = Error;
    fn from_str(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "best" | "best_available" | "default" => Ok(Self::Best),
            "avx512_core_bf16" => Ok(Self::Avx512CoreBf16),
            "avx2" => Ok(Self::Avx2),
            _ => Err(Error::config(format!(
                "unknown ISA cap {name:?}: expected best, avx512_core_bf16 or avx2"
            ))),
        }
    }
}

/// What the runtime found about AMX before choosing its ISA.
///
/// Some virtual machines lose a thread's AMX tile registers when the
/// scheduler preempts it mid-GEMM, and the GEMM then returns a wrong block
/// without failing. Under [`IsaCap::Best`] the runtime runs a 40 ms probe for
/// it and, on such a host, runs AVX512_CORE_BF16 instead of AMX.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "result")]
#[non_exhaustive]
pub enum AmxHealth {
    /// Not probed: the cap is below AMX.
    NotProbed,
    /// The process cannot use AMX (no CPU support or no OS permission).
    Unavailable,
    /// No wrong result. `conclusive` is false when the probe thread was
    /// preempted too rarely to tell.
    Healthy { conclusive: bool },
    /// Tile state was lost: `corrupt` wrong results in `iterations` tile
    /// GEMMs over `preemptions` preemptions. The runtime runs
    /// AVX512_CORE_BF16.
    TileStateLoss {
        corrupt: u64,
        iterations: u64,
        preemptions: u64,
    },
}

impl From<ProbeResult> for AmxHealth {
    fn from(r: ProbeResult) -> Self {
        match r {
            ProbeResult::Unavailable => Self::Unavailable,
            ProbeResult::Healthy { conclusive } => Self::Healthy { conclusive },
            ProbeResult::TileStateLoss {
                corrupt,
                iterations,
                preemptions,
            } => Self::TileStateLoss {
                corrupt,
                iterations,
                preemptions,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isa_bits_names_and_capabilities_agree() {
        for (isa, bits, name) in ISAS {
            assert_eq!(Isa::from_bits(bits), Some(isa));
            assert_eq!(isa.bits(), bits);
            assert_eq!(isa.name(), name);
            assert_eq!(isa.to_string(), name);
        }
        assert_eq!(Isa::from_bits(0x1), None, "SSE41 is below the floor");
        assert_eq!(Isa::from_bits(0x3), None, "AVX is below the floor");
        assert!(!Isa::Avx2.has_bf16_matmul());
        assert!(!Isa::Avx512CoreVnni.has_bf16_matmul());
        assert!(!Isa::Avx2Vnni2.has_bf16_matmul());
        assert!(Isa::Avx512CoreBf16.has_bf16_matmul() && !Isa::Avx512CoreBf16.has_amx());
        assert!(Isa::Avx10_1_512.has_bf16_matmul() && !Isa::Avx10_1_512.has_amx());
        for amx in [
            Isa::Avx10_1_512Amx,
            Isa::Avx10_1_512AmxFp16,
            Isa::Avx10_2_512Amx2,
        ] {
            assert!(amx.has_amx() && amx.has_bf16_matmul(), "{amx}");
        }
        assert_eq!(
            serde_json::to_string(&Isa::Avx512CoreBf16).unwrap(),
            "\"AVX512_CORE_BF16\""
        );
    }

    #[test]
    fn caps_parse_in_any_case() {
        for (name, cap) in [
            ("best", IsaCap::Best),
            ("best_available", IsaCap::Best),
            ("AVX512_CORE_BF16", IsaCap::Avx512CoreBf16),
            ("avx2", IsaCap::Avx2),
        ] {
            assert_eq!(name.parse::<IsaCap>().unwrap(), cap);
        }
        let err = "avx512".parse::<IsaCap>().unwrap_err();
        assert_eq!(err.kind(), crate::ErrorKind::Config);
        assert_eq!(
            serde_json::from_str::<IsaCap>("\"avx512_core_bf16\"").unwrap(),
            IsaCap::Avx512CoreBf16
        );
    }
}
