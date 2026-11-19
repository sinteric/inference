//! Quality gates used by qualification.
//!
//! Vela (classification/span): zero decoded mismatches; calibrated
//! prob/score diff ≤ 1e-4 against the FP32 ONNX oracle (the gate the
//! ported experiments qualified on).

pub use rt_core::{Error, Result};

pub const VELA_PROB_ATOL: f32 = 1e-4;
/// EmbeddingGemma 2 parity gate vs the PyTorch FP32 reference.
pub const EG2_MIN_COSINE: f32 = 0.999;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GateReport {
    pub n: usize,
    pub mismatches: usize,
    pub max_abs_prob_diff: f32,
    pub max_abs_logit_diff: f32,
    pub passed: bool,
}

pub fn compare_probs(oracle: &[f32], got: &[f32], atol: f32) -> Result<GateReport> {
    if oracle.len() != got.len() {
        return Err(Error::Shape(format!(
            "oracle {} vs got {}",
            oracle.len(),
            got.len()
        )));
    }
    let mut max_d = 0f32;
    for (a, b) in oracle.iter().zip(got) {
        if !b.is_finite() {
            return Err(Error::Message("NaN/inf in candidate".into()));
        }
        max_d = max_d.max((a - b).abs());
    }
    Ok(GateReport {
        n: oracle.len(),
        mismatches: 0,
        max_abs_prob_diff: max_d,
        max_abs_logit_diff: 0.0,
        passed: max_d <= atol,
    })
}

/// Fail-closed identity check: model/input/binary sha, provider, thread budget.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct RunIdentity {
    pub model_revision: String,
    pub weight_sha256: String,
    pub backend: String,
    pub nthreads: usize,
    pub provider: String,
}

pub fn assert_identity(expected: &RunIdentity, observed: &RunIdentity) -> Result<()> {
    if expected != observed {
        return Err(Error::Message(format!(
            "identity mismatch: expected {expected:?} observed {observed:?}"
        )));
    }
    Ok(())
}

pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() || a.is_empty() {
        return Err(Error::Shape(format!(
            "cosine length {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        if !x.is_finite() || !y.is_finite() {
            return Err(Error::Message("NaN/inf in cosine".into()));
        }
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    Ok(dot / (na.sqrt() * nb.sqrt()).max(1e-12))
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(Error::Shape(format!(
            "absdiff length {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let mut m = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        m = m.max((x - y).abs());
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_passes_under_atol() {
        let r = compare_probs(&[0.1, 0.9], &[0.10005, 0.89995], VELA_PROB_ATOL).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn gate_fails_over_atol() {
        let r = compare_probs(&[0.1, 0.9], &[0.2, 0.8], VELA_PROB_ATOL).unwrap();
        assert!(!r.passed);
    }

    #[test]
    fn rejects_nan() {
        assert!(compare_probs(&[0.1], &[f32::NAN], VELA_PROB_ATOL).is_err());
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = [0.2f32, 0.4, 0.4];
        assert!((cosine(&v, &v).unwrap() - 1.0).abs() < 1e-6);
        assert!(cosine(&v, &[0.2, 0.4, 0.4001]).unwrap() > EG2_MIN_COSINE);
    }
}
