//! The plan cache evicts its least recently executed plan.
use crate::cpu::onednn::{OnednnCpu, SessionOptions};

fn creations(cpu: &OnednnCpu) -> u64 {
    cpu.stats().plan_creations
}

#[test]
fn full_cache_evicts_least_recently_used_plan() {
    crate::test_support::own_process!(full_cache_evicts_least_recently_used_plan);
    let cpu = OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::new(1).unwrap(),
        SessionOptions::default(),
    )
    .unwrap();
    cpu.set_primitive_capacity(2).unwrap();
    // Kind 3 is GELU (erf), an elementwise plan keyed by its row count.
    let gelu = |m: usize| {
        cpu.op_inner(3, m, 8, &vec![-1.; m * 8], None, None, 0.)
            .unwrap()
    };
    gelu(1);
    gelu(2);
    gelu(1);
    assert_eq!(creations(&cpu), 2);
    // m = 2 is the least recently used plan, so m = 3 replaces it.
    gelu(3);
    gelu(1);
    assert_eq!(creations(&cpu), 3);
    gelu(2);
    assert_eq!(creations(&cpu), 4);
    let stats = cpu.stats();
    assert_eq!(stats.cached_plans, 2);
    assert_eq!(stats.plan_capacity, 2);
    assert_eq!(cpu.primitive_capacity(), 2);
    // m = 3 and m = 2 each replaced a plan; the budget evicted nothing.
    assert_eq!(stats.capacity_evictions, 2);
    assert_eq!(stats.budget_evictions, 0);
}

fn values(n: usize, mut seed: u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 23) as f32 - 1.
        })
        .collect()
}

#[test]
fn memory_budget_evicts_plans_without_changing_results() {
    crate::test_support::own_process!(memory_budget_evicts_plans_without_changing_results);
    let load = |memory_budget| {
        OnednnCpu::load_with(
            &crate::cpu::threads::ThreadBudget::new(1).unwrap(),
            SessionOptions {
                memory_budget,
                ..SessionOptions::default()
            },
        )
        .unwrap()
    };
    let (free, tight) = (load(0), load(1));
    let (k, n) = (256, 384);
    let weights = values(n * k, 7);
    let (wf, wt) = (
        free.pack(n, k, &weights).unwrap(),
        tight.pack(n, k, &weights).unwrap(),
    );
    let x = values(64 * k, 11);
    let linear = |cpu: &OnednnCpu, w, m: usize| {
        let mut y = vec![0f32; m * n];
        cpu.linear(&x[..m * k], w, &mut y, false).unwrap();
        y
    };
    for m in [1, 2, 3, 5, 8, 13, 21, 34, 2, 13, 64, 1] {
        let (a, b) = (linear(&free, &wf, m), linear(&tight, &wt, m));
        assert!(
            a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "m={m}: results differ under a memory budget"
        );
    }
    let (fs, ts) = (free.stats(), tight.stats());
    assert_eq!(fs.budget_evictions, 0);
    assert!(fs.cached_plans > 5);
    if ts.plan_bytes > 1 {
        // Over budget the backend keeps only the plan that just ran.
        assert_eq!(ts.cached_plans, 1, "{ts:?}");
        assert!(ts.budget_evictions >= 10, "{ts:?}");
    }
}

#[test]
fn memory_budget_bounds_pooled_buffers() {
    crate::test_support::own_process!(memory_budget_bounds_pooled_buffers);
    let cpu = OnednnCpu::load_with(
        &crate::cpu::threads::ThreadBudget::new(1).unwrap(),
        SessionOptions {
            memory_budget: 1 << 20,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    // Four 256 KiB buffers fill a 1 MiB budget; a 2 MiB one does not fit
    // and is freed instead of pooled.
    let small: Vec<_> = (0..4).map(|_| cpu.buffer(64 << 10).unwrap()).collect();
    for buffer in small {
        cpu.recycle(buffer);
    }
    cpu.recycle(cpu.buffer(512 << 10).unwrap());
    let stats = cpu.stats();
    assert_eq!(stats.retained_output_buffer_bytes, 1 << 20, "{stats:?}");
    assert!(stats.retained_bytes <= 1 << 20, "{stats:?}");
}
