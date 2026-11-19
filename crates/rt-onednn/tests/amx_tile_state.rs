//! BF16 matmul stays repeatable on the default ISA while the machine is
//! oversubscribed. On a host that loses AMX tile state when a thread is
//! preempted, an unguarded AMX GEMM returns an occasional wrong 32x32 block;
//! the default cap must then fall back to AVX512_CORE_BF16 (see
//! `rt_onednn::amx_health`). Skips when the process cannot use AMX.
use rt_backend::Hardware;
use rt_onednn::{amx_health, isa_has_amx, AmxHealth, OnednnCpu, SessionOptions};
use std::sync::atomic::{AtomicBool, Ordering};

fn values(n: usize, mut seed: u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / (1 << 23) as f32 - 1.) * scale
        })
        .collect()
}

#[test]
fn default_isa_bf16_linear_is_repeatable_under_load() {
    if amx_health() == AmxHealth::Unavailable {
        eprintln!("skip: AMX is not available to this process");
        return;
    }
    // AMX_TILE_STATE_NO_PROBE=1 runs unguarded: fails on an affected host.
    let guarded = std::env::var_os("AMX_TILE_STATE_NO_PROBE").is_none();
    rt_onednn::set_amx_probe(guarded);
    let cpus = std::thread::available_parallelism().map_or(2, |n| n.get());
    let threads = cpus.clamp(2, 8);
    let cpu = OnednnCpu::load_with(
        &rt_threads::ThreadBudget::new(threads).unwrap(),
        SessionOptions {
            workspace_limit: 1 << 28,
            ..SessionOptions::default()
        },
    )
    .unwrap();
    let healthy = amx_health().amx_usable();
    if guarded {
        assert_eq!(
            isa_has_amx(cpu.cpu_isa),
            healthy,
            "effective ISA {:#x} disagrees with {:?}",
            cpu.cpu_isa,
            amx_health()
        );
    }
    let (m, k, n) = (300, 768, 2304);
    let mut bits = vec![0u16; n * k];
    rt_kernels::f32_slice_to_bf16(&values(n * k, 1, 0.1), &mut bits).unwrap();
    let w = cpu.pack_bf16(n, k, &bits).unwrap();
    let x = values(m * k, 2, 2.0);
    let mut first = vec![0f32; m * n];
    cpu.linear_bf16(&x, &w, &mut first, false).unwrap();

    // Twice as many busy threads as CPUs keep the scheduler preempting the
    // pool workers for the whole run.
    let stop = AtomicBool::new(false);
    let calls: usize = std::env::var("AMX_TILE_STATE_CALLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let differing = std::thread::scope(|scope| {
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
        let mut differing = 0;
        for rep in 0..calls {
            let mut y = values(m * n, 100 + rep as u32, 1000.0);
            cpu.linear_bf16(&x, &w, &mut y, false).unwrap();
            differing += usize::from(y != first);
        }
        stop.store(true, Ordering::Relaxed);
        differing
    });
    assert_eq!(
        differing,
        0,
        "{differing}/{calls} BF16 linear calls differ (isa {:#x}, {:?})",
        cpu.cpu_isa,
        amx_health()
    );
}
