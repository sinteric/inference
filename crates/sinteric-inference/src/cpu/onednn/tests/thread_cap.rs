//! `SessionOptions::threads` caps one backend below the process pool while
//! another backend in the process keeps the whole pool, and the capped
//! backend computes the same results.
use crate::cpu::onednn::{CpuIsaCap, OnednnCpu, SessionOptions};

fn values(n: usize, mut seed: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 23) as f32 - 1.
        })
        .collect()
}

fn backend(pool: usize, threads: usize) -> std::sync::Arc<OnednnCpu> {
    OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::try_new(pool).unwrap(),
        SessionOptions {
            max_cpu_isa: CpuIsaCap::Avx512CoreBf16,
            workspace_limit: 1 << 28,
            threads,
            ..SessionOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn capped_backend_shares_the_pool_and_matches_the_full_one() {
    crate::test_support::own_process!(capped_backend_shares_the_pool_and_matches_the_full_one);
    let full = backend(4, 0);
    let capped = backend(4, 2);
    let single = backend(4, 1);
    let wide = backend(4, 64);
    assert_eq!(full.threads(), 4);
    assert_eq!(capped.threads(), 2);
    assert_eq!(single.threads(), 1);
    assert_eq!(wide.threads(), 4, "wider than the pool is the pool");
    for (k, n, m) in [(768, 768, 134), (768, 1152, 160), (1152, 768, 37)] {
        let wv = values(n * k, 7);
        let x = values(m * k, 11);
        let run = |cpu: &OnednnCpu| {
            let w = cpu.pack(n, k, &wv).unwrap();
            let mut y = vec![0f32; m * n];
            cpu.linear(&x, &w, &mut y, false).unwrap();
            y
        };
        let want = run(&full);
        // Interleave backends: a capped plan must not leak its width into the
        // full backend's next plan on this thread, nor the other way round.
        for cpu in [&capped, &full, &single, &wide, &capped] {
            let got = run(cpu);
            let max = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max <= 1e-5,
                "k={k} n={n} m={m} threads={}: max diff {max}",
                cpu.threads()
            );
        }
    }
}
