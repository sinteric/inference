use rt_core::{Error, Result};

pub fn sgemm_ref(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) -> Result<()> {
    if a.len() < (m.saturating_sub(1)).saturating_mul(lda).saturating_add(k)
        || b.len() < (k.saturating_sub(1)).saturating_mul(ldb).saturating_add(n)
        || c.len() < (m.saturating_sub(1)).saturating_mul(ldc).saturating_add(n)
    {
        return Err(Error::Shape("sgemm: buffer too small".into()));
    }
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * lda + p] * b[p * ldb + j];
            }
            let idx = i * ldc + j;
            c[idx] = alpha * acc + beta * c[idx];
        }
    }
    Ok(())
}
