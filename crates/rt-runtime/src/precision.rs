//! Numeric precision a model computes in. Every model config takes one
//! (`None` is the model's validated default) and every loaded model reports
//! the one it runs ([`crate::Model::precision`]).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Matmul/convolution precision. Norms, softmax, residual stream, pooling
/// and outputs stay FP32 in every mode.
///
/// A model implements a subset ([`crate::Model::PRECISIONS`]); asking for
/// another is [`Error::Unsupported`], never a silent fallback. The ISA that
/// runs it is separate: [`crate::RuntimeOptions::cpu_isa`] caps it and
/// [`crate::Runtime::cpu_isa_name`] reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Precision {
    /// FP32 inputs, weights and accumulation (oneDNN strict fpmath). Weights
    /// may be stored as BF16 and widened exactly when the checkpoint is BF16.
    Fp32,
    /// BF16 matmul inputs, FP32 accumulation and FP32 stream (what oneDNN
    /// calls BF32 when it down-converts f32 tensors inside a kernel).
    Bf16,
}

impl Precision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Bf16 => "bf16",
        }
    }

    /// `requested`, or the first of `supported` (the model's default) when
    /// `None`. A precision outside `supported` is [`Error::Unsupported`].
    pub fn resolve(requested: Option<Self>, model: &str, supported: &[Self]) -> Result<Self> {
        let default = *supported
            .first()
            .ok_or_else(|| Error::Config(format!("{model} declares no precision")))?;
        let p = requested.unwrap_or(default);
        if supported.contains(&p) {
            return Ok(p);
        }
        let names: Vec<_> = supported.iter().map(|p| p.as_str()).collect();
        Err(Error::Unsupported {
            what: "precision",
            reason: format!("{model} runs {} only, not {p}", names.join(", ")),
        })
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Precision {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "fp32" => Ok(Self::Fp32),
            "bf16" => Ok(Self::Bf16),
            _ => Err(Error::Config(format!(
                "unknown precision {s:?} (expected fp32 or bf16)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_defaults_to_first_and_rejects_unsupported() {
        let both = [Precision::Bf16, Precision::Fp32];
        assert_eq!(
            Precision::resolve(None, "m", &both).unwrap(),
            Precision::Bf16
        );
        assert_eq!(
            Precision::resolve(Some(Precision::Fp32), "m", &both).unwrap(),
            Precision::Fp32
        );
        let err =
            Precision::resolve(Some(Precision::Bf16), "vela", &[Precision::Fp32]).unwrap_err();
        assert!(
            matches!(
                err,
                Error::Unsupported {
                    what: "precision",
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(
            err.to_string().contains("vela runs fp32 only, not bf16"),
            "{err}"
        );
        assert!(Precision::resolve(None, "m", &[]).is_err());
    }

    #[test]
    fn names_and_serde_round_trip() {
        for p in [Precision::Fp32, Precision::Bf16] {
            assert_eq!(p.as_str().parse::<Precision>().unwrap(), p);
            let json = serde_json::to_string(&p).unwrap();
            assert_eq!(json, format!("\"{p}\""));
            assert_eq!(serde_json::from_str::<Precision>(&json).unwrap(), p);
        }
        assert!("bf32".parse::<Precision>().is_err());
    }
}
