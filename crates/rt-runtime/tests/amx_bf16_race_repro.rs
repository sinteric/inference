//! Repro for the BF16 matmul nondeterminism seen on AMX in a VM that loses AMX
//! tile state when a thread is preempted (`tools/amx-tile-state/README.md`). Needs
//! no model: random BF16 weights of the QKV shape, the same pre-converted input
//! and a dirty output buffer, repeated, with busy threads keeping the machine
//! oversubscribed; any run whose output differs from the first is a wrong
//! block. Statistical, so ignored:
//!
//! ```text
//! cargo test --release -p rt-runtime --test amx_bf16_race_repro -- --ignored --nocapture
//! AMX_REPRO_NO_PROBE=1 cargo test ...             # AMX unguarded: fails on an affected VM
//! AMX_REPRO_ISA=AVX512_CORE_BF16 cargo test ...   # pinned: expected 0 differing runs
//! ```
//!
//! With the default (guarded) runtime an affected host runs AVX512_CORE_BF16
//! and the test passes; unguarded on the dev VM (4 threads, 8 busy threads)
//! it reports a few differing runs per 400 calls, each a 1024-value block off
//! by up to ~1.4.

use rt_backend::Hardware;
use rt_kernels::{f32_slice_to_bf16, f32_slice_to_bf16_nthreads};
use rt_runtime::{CpuIsa, Runtime, RuntimeOptions};

fn random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
        })
        .collect()
}

#[test]
#[ignore = "statistical repro, minutes"]
fn linear_bf16_src_runs_are_repeatable() {
    if std::env::var_os("AMX_REPRO_NO_PROBE").is_some() {
        rt_runtime::set_amx_probe(false);
    }
    let mut options = RuntimeOptions::new().with_threads(4);
    if let Ok(name) = std::env::var("AMX_REPRO_ISA") {
        options = options.with_cpu_isa(CpuIsa::from_name(&name).unwrap());
    }
    let rt = Runtime::new(options).unwrap();
    let hw = rt.backend().unwrap();
    let (n, k) = (2304usize, 768usize);
    let mut weights = vec![0u16; n * k];
    f32_slice_to_bf16(&random(n * k, 1, 0.1), &mut weights).unwrap();
    let w = hw.pack_bf16(n, k, &weights).unwrap();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut differing = 0;
    std::thread::scope(|scope| {
        for _ in 0..16 {
            scope.spawn(|| {
                let mut x = 1u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    for _ in 0..10_000 {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                    }
                    std::hint::black_box(x);
                }
            });
        }
        for m in [300usize, 257, 129, 512] {
            let x = random(m * k, 2, 2.0);
            let mut bits = vec![0u16; x.len()];
            f32_slice_to_bf16_nthreads(&x, &mut bits, 4).unwrap();
            let mut first = vec![0f32; m * n];
            hw.linear_bf16_src(&bits, &w, &mut first, false).unwrap();
            for rep in 0..400 {
                let mut y = random(m * n, 100 + rep, 1000.0);
                hw.linear_bf16_src(&bits, &w, &mut y, false).unwrap();
                differing += usize::from(y != first);
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    eprintln!("{}: {differing}/1600 runs differ", rt.cpu_isa_name());
    if rt_onednn::amx_probe_enabled() || !rt_onednn::isa_has_amx(rt.cpu_isa()) {
        assert_eq!(differing, 0, "BF16 matmul is not repeatable");
    }
}
