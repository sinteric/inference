//! Callers (the OCR recognizer) pad and chunk projection rows to bound the
//! number of matmul primitives. That is exact only because each output row
//! is bitwise independent of the primitive's M and of the row's position.
use crate::cpu::onednn::{CpuIsaCap, OnednnCpu, SessionOptions};

fn values(n: usize, mut seed: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 23) as f32 - 1.
        })
        .collect()
}

#[test]
fn linear_rows_are_independent_of_m() {
    crate::test_support::own_process!(linear_rows_are_independent_of_m);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(4));
    let cpu = OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::try_new(threads).unwrap(),
        SessionOptions {
            max_cpu_isa: CpuIsaCap::Avx512CoreBf16,
            workspace_limit: 1 << 28,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    for (k, n, max_m) in [(120, 360, 700), (240, 120, 700), (120, 11947, 300)] {
        let w = cpu.pack(n, k, &values(n * k, 7)).unwrap();
        let x = values(max_m * k, 11);
        let linear = |rows: &[f32]| {
            let mut y = vec![0f32; rows.len() / k * n];
            cpu.linear(rows, &w, &mut y, false).unwrap();
            y
        };
        let whole = linear(&x);
        let ms = (1..=40).chain([63, 64, 65, 255, 256, 257, max_m - 1]);
        for m in ms {
            let mut padded = x[..m * k].to_vec();
            padded.resize(m.next_multiple_of(16) * k, 0.);
            for (label, got) in [
                ("exact", linear(&x[..m * k])),
                ("padded", linear(&padded)),
                ("offset", linear(&x[(max_m - m) * k..])),
            ] {
                let want = if label == "offset" {
                    &whole[(max_m - m) * n..]
                } else {
                    &whole[..m * n]
                };
                assert!(
                    got[..m * n]
                        .iter()
                        .zip(want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "k={k} n={n} m={m} {label} rows differ from M={max_m}"
                );
            }
        }
    }
}
