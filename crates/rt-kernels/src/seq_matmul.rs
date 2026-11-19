//! Small batched f32 matmuls with a fixed summation order: every output is
//! one fused multiply-add chain over ascending `k`, starting from 0.0. For
//! the shapes the OCR recognizer's attention uses, oneDNN's f32 matmul
//! produces the same bits (checked by `rt-onednn`'s attention_order test), so
//! these replace per-shape oneDNN primitives without changing results.

use crate::parallel::{for_ranges, rows_mut};
use rt_core::{check_shape, try_vec, Result};

/// `out[i][..n] = Σ_k a[i][k] · b[k][..n]`, one FMA chain per output.
#[inline(always)]
fn row(a: &[f32], b: &[f32], n: usize, out: &mut [f32]) {
    out.fill(0.);
    for (&s, b) in a.iter().zip(b.chunks_exact(n)) {
        for (o, &v) in out.iter_mut().zip(b) {
            *o = s.mul_add(v, *o);
        }
    }
}

fn rows_portable(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32]) {
    for (a, out) in a.chunks_exact(k).zip(out.chunks_exact_mut(n)) {
        row(a, b, n, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_fma(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32]) {
    // Same scalar code; `mul_add` is single-rounding on every path.
    for (a, out) in a.chunks_exact(k).zip(out.chunks_exact_mut(n)) {
        row(a, b, n, out);
    }
}

fn rows(a: &[f32], b: &[f32], k: usize, n: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { rows_fma(a, b, k, n, out) };
        }
    }
    rows_portable(a, b, k, n, out)
}

/// Rows of all groups split across `nthreads`; `b_of(group)` is that
/// group's `[k, n]` right operand.
fn run<'a>(
    g: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b_of: impl Fn(usize) -> &'a [f32] + Sync,
    nthreads: usize,
) -> Result<Vec<f32>> {
    let mut out = try_vec(g * m * n)?;
    let p = out.as_mut_ptr() as usize;
    let grain = (1usize << 14).div_ceil(k * n).max(1);
    for_ranges(nthreads, g * m, grain, |lo, hi| {
        // SAFETY: workers receive disjoint row ranges of `out`.
        let out = unsafe { rows_mut(p as *mut f32, lo, hi, n) };
        let mut r = lo;
        while r < hi {
            let group = r / m;
            let end = hi.min((group + 1) * m);
            rows(
                &a[r * k..end * k],
                b_of(group),
                k,
                n,
                &mut out[(r - lo) * n..(end - lo) * n],
            );
            r = end;
        }
    });
    Ok(out)
}

/// `a [g, m, k] · b [g, k, n] -> [g, m, n]`.
pub fn seq_matmul_nn(
    g: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    nthreads: usize,
) -> Result<Vec<f32>> {
    check_shape!(
        a.len() == g * m * k && b.len() == g * k * n,
        "sequential matmul operands"
    );
    run(g, m, k, n, a, |i| &b[i * k * n..(i + 1) * k * n], nthreads)
}

/// `a [g, m, k] · b [g, n, k]ᵀ -> [g, m, n]`.
pub fn seq_matmul_nt(
    g: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    nthreads: usize,
) -> Result<Vec<f32>> {
    check_shape!(
        a.len() == g * m * k && b.len() == g * n * k,
        "sequential matmul operands"
    );
    let mut bt = try_vec(g * k * n)?;
    for (b, bt) in b.chunks_exact(n * k).zip(bt.chunks_exact_mut(k * n)) {
        for j in 0..n {
            for c in 0..k {
                bt[c * n + j] = b[j * k + c];
            }
        }
    }
    run(g, m, k, n, a, |i| &bt[i * k * n..(i + 1) * k * n], nthreads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scalar_fma_chains_on_every_path() {
        let v = |n: usize, s: u32| {
            (0..n)
                .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ s) % 2001) as f32 / 1000. - 1.)
                .collect::<Vec<_>>()
        };
        let (g, m, k, n) = (3, 37, 15, 29);
        let a = v(g * m * k, 1);
        let b = v(g * k * n, 2);
        let nn = seq_matmul_nn(g, m, k, n, &a, &b, 4).unwrap();
        let mut bt = vec![0f32; b.len()];
        for x in 0..g {
            for c in 0..k {
                for j in 0..n {
                    bt[(x * n + j) * k + c] = b[(x * k + c) * n + j];
                }
            }
        }
        let nt = seq_matmul_nt(g, m, k, n, &a, &bt, 1).unwrap();
        let mut portable = vec![0f32; g * m * n];
        for x in 0..g {
            rows_portable(
                &a[x * m * k..(x + 1) * m * k],
                &b[x * k * n..(x + 1) * k * n],
                k,
                n,
                &mut portable[x * m * n..(x + 1) * m * n],
            );
        }
        for x in 0..g {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f32;
                    for c in 0..k {
                        acc = a[(x * m + i) * k + c].mul_add(b[(x * k + c) * n + j], acc);
                    }
                    let at = (x * m + i) * n + j;
                    assert_eq!(acc.to_bits(), nn[at].to_bits());
                    assert_eq!(acc.to_bits(), nt[at].to_bits());
                    assert_eq!(acc.to_bits(), portable[at].to_bits());
                }
            }
        }
        assert!(seq_matmul_nn(g, m, k, n, &a[1..], &b, 1).is_err());
    }
}
