//! Comparisons the parity tests make against their upstream references.

use crate::{Error, Result};

/// Cosine similarity of two equal-length, finite vectors.
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

/// Largest absolute difference of two equal-length vectors.
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
    fn cosine_identical_is_one() {
        let v = [0.2f32, 0.4, 0.4];
        assert!((cosine(&v, &v).unwrap() - 1.0).abs() < 1e-6);
        assert!(cosine(&v, &[0.2, 0.4, 0.4001]).unwrap() > 0.999);
        assert!(cosine(&v, &[0.2, f32::NAN, 0.4]).is_err());
        assert!(cosine(&v, &[0.2]).is_err());
    }

    #[test]
    fn max_abs_diff_is_the_largest_gap() {
        assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.5, 1.0]).unwrap(), 1.0);
        assert!(max_abs_diff(&[1.0], &[]).is_err());
    }
}
