//! Model-free check of EG2's BF16 matmul and attention calls with the real
//! layer shapes (hidden 512, MLP 2048, 4 query heads, sliding layers with
//! 2 KV heads of 256, global layers with 1 KV head of 512) while busy threads
//! keep the machine oversubscribed. EG2 BF16 calls `linear_bf16`,
//! `linear_bf16_src` and `attention_gqa_src`, which on a host that loses AMX
//! tile state across preemption return an occasional wrong block through
//! `brg_matmul:avx10_1_512_amx`. Each op repeats on identical inputs into a
//! dirty buffer; any output that differs from the first is a wrong block.
//!
//! Env: EG2_REPRO_THREADS (4), EG2_REPRO_CALLS (300 per op and shape),
//! EG2_REPRO_NO_PROBE (AMX unguarded). Statistical, so ignored.
use crate::cpu::onednn::Attention;
use crate::cpu::runtime::{Options, Runtime};
use std::sync::atomic::{AtomicBool, Ordering};

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

fn bf16(x: &[f32]) -> Vec<u16> {
    let mut y = vec![0u16; x.len()];
    crate::cpu::kernels::f32_slice_to_bf16(x, &mut y).unwrap();
    y
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "statistical repro"]
fn eg2_bf16_ops_are_repeatable_under_load() {
    crate::test_support::own_process!(eg2_bf16_ops_are_repeatable_under_load);
    if std::env::var_os("EG2_REPRO_NO_PROBE").is_some() {
        crate::cpu::onednn::set_amx_probe(false);
    }
    let threads = env("EG2_REPRO_THREADS", 4);
    let calls = env("EG2_REPRO_CALLS", 300);
    let rt = Runtime::new(Options::new().with_threads(threads)).unwrap();
    let hw = rt.backend().unwrap();
    let tile = crate::embeddinggemma2::config::attn_tile(rt.cpu_isa());
    let stop = AtomicBool::new(false);
    let cpus = std::thread::available_parallelism().map_or(2, |n| n.get());
    let mut total_bad = 0;
    let mut total = 0;
    std::thread::scope(|scope| {
        for _ in 0..cpus * 2 {
            scope.spawn(|| {
                let mut x = 1u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..10_000 {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                    }
                    std::hint::black_box(x);
                }
            });
        }
        // (name, k, n): the matmuls of one sliding and one global layer.
        let linears = [
            ("qkv sliding src", 512usize, 2048usize, true),
            ("qkv global src", 512, 3072, true),
            ("up/gate src", 512, 2048, true),
            ("o_proj sliding f32", 1024, 512, false),
            ("o_proj global f32", 2048, 512, false),
            ("down f32", 2048, 512, false),
        ];
        for (name, k, n, src) in linears {
            let weights = bf16(&random(n * k, 1, 0.1));
            let w = hw.pack_bf16(n, k, &weights).unwrap();
            for m in [128usize, 300, 512] {
                let x = random(m * k, 2, 2.0);
                let xb = bf16(&x);
                let run = |dirty: u64| {
                    let mut y = random(m * n, dirty, 1000.0);
                    if src {
                        hw.linear_bf16_src(&xb, &w, &mut y, false).unwrap();
                    } else {
                        hw.linear_bf16(&x, &w, &mut y, false).unwrap();
                    }
                    y
                };
                let first = run(1);
                let bad = (0..calls).filter(|&r| run(100 + r as u64) != first).count();
                eprintln!("RESULT {name:<20} m={m:<4} k={k:<5} n={n:<5}: {bad}/{calls}");
                total_bad += bad;
                total += calls;
            }
        }
        // (name, nq, nkv, d, local)
        let attns = [
            ("attn sliding", 4usize, 2usize, 256usize, true),
            ("attn global", 4, 1, 512, false),
        ];
        for (name, nq, nkv, d, local) in attns {
            for s in [128usize, 300, 512] {
                let q = bf16(&random(nq * s * d, 3, 2.0));
                let k = bf16(&random(nkv * s * d, 4, 2.0));
                let v = bf16(&random(nkv * s * d, 5, 2.0));
                let pad = vec![vec![]];
                let shape = Attention {
                    batch: 1,
                    seq: s,
                    q_heads: nq,
                    kv_heads: nkv,
                    head_dim: d,
                    tile,
                    window: local.then_some(512),
                    scale: 1.0,
                    padding: &pad,
                };
                let run = || hw.attention_gqa_src(&shape, &q, &k, &v).unwrap();
                let first = run();
                let bad = (0..calls).filter(|_| run() != first).count();
                eprintln!("RESULT {name:<20} s={s:<4} nq={nq} nkv={nkv} d={d:<4}: {bad}/{calls}");
                total_bad += bad;
                total += calls;
            }
        }
        stop.store(true, Ordering::Relaxed);
    });
    eprintln!(
        "TOTAL isa={} threads={threads}: {total_bad}/{total} calls differ",
        rt.isa()
    );
    if crate::cpu::onednn::amx_probe_enabled() || !rt.isa().has_amx() {
        assert_eq!(total_bad, 0, "EG2 BF16 ops are not repeatable");
    }
}
