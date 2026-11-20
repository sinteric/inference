//! The injection head of `ModernBertForSequenceClassification`, in FP32:
//! mean over **all** of a sequence's tokens → `head.dense` → GELU →
//! `head.norm` → `classifier`; the logit is `l[INJECTION] - l[SAFE]`.

use std::collections::HashMap;

use crate::cpu::kernels::gelu_erf_scalar;
use crate::{check_shape, Result, ResultContext as Context};

/// Dot product with eight independent lanes so the loop vectorizes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let mut lanes = [0f32; 8];
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            lanes[i] += x[i] * y[i];
        }
    }
    lanes.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

/// `y[o] = w[o] . x (+ b[o])` for a row-major `[out, x.len()]` matrix.
fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32]) {
    for (o, out) in y.iter_mut().enumerate() {
        *out = dot(&w[o * x.len()..(o + 1) * x.len()], x) + b.map_or(0.0, |b| b[o]);
    }
}

/// `nn.LayerNorm` over one row (biased variance).
fn layer_norm(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32, y: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = (var + eps).sqrt().recip();
    for i in 0..x.len() {
        y[i] = (x[i] - mean) * inv * w[i] + b.map_or(0.0, |b| b[i]);
    }
}

/// Mean of the rows of width `h`, accumulated in f64; `max(rows, 1)` divisor.
fn mean_rows(x: &[f32], h: usize) -> Vec<f32> {
    let rows = x.len() / h;
    let mut acc = vec![0f64; h];
    for row in x.chunks_exact(h) {
        for (a, v) in acc.iter_mut().zip(row) {
            *a += f64::from(*v);
        }
    }
    let div = rows.max(1) as f64;
    acc.into_iter().map(|a| (a / div) as f32).collect()
}

/// A head tensor with a checked shape, taken out of `map`.
fn take(map: &mut HashMap<String, Vec<f32>>, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = map
        .remove(name)
        .with_context(|| format!("head tensor {name}"))?;
    check_shape!(
        t.len() == len,
        "head tensor {name}: {} values, expected {len}",
        t.len()
    );
    Ok(t)
}

/// The injection head's weights.
pub(crate) struct InjectionHead {
    hidden: usize,
    eps: f32,
    dense: Vec<f32>,
    norm: Vec<f32>,
    classifier_w: Vec<f32>,
    classifier_b: Vec<f32>,
}

impl InjectionHead {
    /// Take the head tensors out of `map`. `eps` is the encoder's norm
    /// epsilon, which `head.norm` shares.
    pub(crate) fn from_tensors(
        map: &mut HashMap<String, Vec<f32>>,
        hidden: usize,
        eps: f32,
    ) -> Result<Self> {
        let h = hidden;
        Ok(Self {
            hidden,
            eps,
            dense: take(map, "head.dense.weight", h * h)?,
            norm: take(map, "head.norm.weight", h)?,
            classifier_w: take(map, "classifier.weight", 2 * h)?,
            classifier_b: take(map, "classifier.bias", 2)?,
        })
    }

    /// Injection logit `l[1] - l[0]` of one sequence's hidden states
    /// (`len × hidden`, padding excluded).
    pub(crate) fn logit(&self, hidden: &[f32]) -> f32 {
        let h = self.hidden;
        let pooled = mean_rows(hidden, h);
        let mut x = vec![0f32; h];
        linear(&pooled, &self.dense, None, &mut x);
        for v in &mut x {
            *v = gelu_erf_scalar(*v);
        }
        let mut normed = vec![0f32; h];
        layer_norm(&x, &self.norm, None, self.eps, &mut normed);
        let mut logits = [0f32; 2];
        linear(
            &normed,
            &self.classifier_w,
            Some(&self.classifier_b),
            &mut logits,
        );
        logits[1] - logits[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_a_plain_sum_past_the_lanes() {
        let a: Vec<f32> = (0..19).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..19).map(|i| 1.0 - i as f32 * 0.1).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - want).abs() < 1e-4);
    }

    #[test]
    fn mean_pools_every_row() {
        assert_eq!(mean_rows(&[1.0, 2.0, 3.0, 6.0], 2), [2.0, 4.0]);
        assert_eq!(mean_rows(&[], 2), [0.0, 0.0]);
    }

    #[test]
    fn head_tensors_are_shape_checked() {
        let mut map: HashMap<String, Vec<f32>> = [
            ("head.dense.weight", vec![0.0; 4]),
            ("head.norm.weight", vec![1.0; 2]),
            ("classifier.weight", vec![0.0; 4]),
            ("classifier.bias", vec![0.25, 1.0]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let mut short = map.clone();
        short.insert("classifier.bias".into(), vec![0.0]);
        assert!(InjectionHead::from_tensors(&mut short, 2, 1e-5).is_err());
        let head = InjectionHead::from_tensors(&mut map, 2, 1e-5).unwrap();
        assert!(map.is_empty());
        // Zero dense and classifier weights: the logit is the bias difference.
        assert_eq!(head.logit(&[3.0, -1.0, 0.5, 2.0]), 0.75);
    }
}
