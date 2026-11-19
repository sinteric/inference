//! LayerNorm. Encoder norms are weight-only (`norm_bias: false`). The
//! readout `norm` is a standard `nn.LayerNorm` (weight + bias).

pub(crate) fn layer_norm(
    x: &[f32],
    hidden: usize,
    gamma: &[f32],
    beta: Option<&[f32]>,
    eps: f32,
    y: &mut [f32],
) {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(gamma.len(), hidden);
    debug_assert_eq!(x.len() % hidden, 0);
    if let Some(b) = beta {
        debug_assert_eq!(b.len(), hidden);
    }
    let rows = x.len() / hidden;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let out = &mut y[r * hidden..(r + 1) * hidden];
        let mut mean = 0.0f32;
        for &v in row {
            mean += v;
        }
        mean /= hidden as f32;
        let mut var = 0.0f32;
        for &v in row {
            let d = v - mean;
            var += d * d;
        }
        var /= hidden as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..hidden {
            let n = (row[c] - mean) * inv * gamma[c];
            out[c] = match beta {
                Some(b) => n + b[c],
                None => n,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_unit_gamma() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let g = [1.0f32, 1.0];
        let mut y = [0.0f32; 4];
        layer_norm(&x, 2, &g, None, 1e-5, &mut y);
        // each pair is mean-centered
        assert!((y[0] + y[1]).abs() < 1e-4);
        assert!((y[2] + y[3]).abs() < 1e-4);
    }
}
