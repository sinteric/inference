//! The OCR recognizer computes its attention matmuls with
//! `crate::cpu::kernels::seq_matmul_*` instead of per-shape oneDNN primitives. That
//! keeps results only if oneDNN's f32 matmul also sums each output as one
//! FMA chain in ascending k for these shapes (8 heads of width 15, t >= 2).
use crate::cpu::onednn::{CpuIsaCap, Gemm, OnednnCpu, SessionOptions};

fn values(n: usize, mut seed: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 23) as f32 - 1.
        })
        .collect()
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

#[test]
fn onednn_attention_matmuls_are_sequential_fma_chains() {
    crate::test_support::own_process!(onednn_attention_matmuls_are_sequential_fma_chains);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
    let cpu = OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::new(threads).unwrap(),
        SessionOptions {
            max_cpu_isa: CpuIsaCap::Avx512CoreBf16,
            workspace_limit: 1 << 28,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    cpu.set_primitive_capacity(16).unwrap();
    let ts = (2..=40).chain([63, 64, 65, 127, 128, 129, 200]);
    for batch in [1, 3, 16] {
        for t in ts.clone() {
            let g = batch * 8;
            let (q, k, v) = (
                values(g * t * 15, 1),
                values(g * t * 15, 2),
                values(g * t * 15, 3),
            );
            let p = values(g * t * t, 4)
                .iter()
                .map(|x| x.abs())
                .collect::<Vec<_>>();
            let want = cpu
                .mm(Gemm::nt(g, t, t, 15, t * 15, t * 15), &q, &k)
                .unwrap();
            let got = crate::cpu::kernels::seq_matmul_nt(g, t, 15, t, &q, &k, threads).unwrap();
            assert!(same(&want, &got), "scores differ at batch={batch} t={t}");
            let want = cpu
                .mm(Gemm::nn(g, t, 15, t, t * t, t * 15), &p, &v)
                .unwrap();
            let got = crate::cpu::kernels::seq_matmul_nn(g, t, t, 15, &p, &v, threads).unwrap();
            assert!(same(&want, &got), "context differs at batch={batch} t={t}");
        }
    }
}
