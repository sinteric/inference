//! Parametrized sweep for the AMX BF16 wrong-block bug. Ignored (statistical).
//!
//! Env: AMX_REPRO_THREADS (default 4), AMX_REPRO_REPS (400), AMX_REPRO_MODE
//! (src|f32|attn, default src), AMX_REPRO_SHAPES ("m:k:n,..."; attn uses m as
//! sequence length), AMX_REPRO_ISA, AMX_REPRO_VERBOSE, AMX_REPRO_NO_PROBE (AMX
//! unguarded; the default runtime falls back from AMX on an affected host).

use crate::cpu::kernels::{f32_slice_to_bf16, f32_slice_to_bf16_nthreads, layout_qkv_rope};
use crate::cpu::onednn::Attention;
use crate::cpu::runtime::{IsaCap, Options, Runtime};

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

/// Involuntary context switches summed over every thread of this process.
fn involuntary_switches() -> u64 {
    let mut total = 0;
    if let Ok(dir) = std::fs::read_dir("/proc/self/task") {
        for t in dir.flatten() {
            if let Ok(s) = std::fs::read_to_string(t.path().join("status")) {
                for l in s.lines() {
                    if let Some(v) = l.strip_prefix("nonvoluntary_ctxt_switches:") {
                        total += v.trim().parse::<u64>().unwrap_or(0);
                    }
                }
            }
        }
    }
    total
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

#[test]
#[ignore = "statistical repro"]
fn sweep() {
    crate::test_support::own_process!(sweep);
    let threads: usize = env("AMX_REPRO_THREADS", 4);
    let reps: usize = env("AMX_REPRO_REPS", 400);
    let mode: String = env("AMX_REPRO_MODE", "src".to_string());
    let shapes: String = env(
        "AMX_REPRO_SHAPES",
        "300:768:2304,257:768:2304,129:768:2304,512:768:2304".into(),
    );
    if std::env::var_os("AMX_REPRO_NO_PROBE").is_some() {
        crate::cpu::onednn::set_amx_probe(false);
    }
    let mut options = Options::new().with_threads(threads);
    if let Some(name) = std::env::var("AMX_REPRO_ISA")
        .ok()
        .filter(|n| !n.is_empty())
    {
        options = options.with_isa_cap(name.parse::<IsaCap>().unwrap());
    }
    let rt = Runtime::new(options).unwrap();
    let hw = rt.backend().unwrap();
    if let Some(v) = std::env::var("AMX_REPRO_VERBOSE")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        // SAFETY: sets oneDNN's process-wide verbose level; takes no pointer.
        unsafe { onednn_sys::dnnl_set_verbose(v) };
    }
    let mut total_bad = 0;
    let mut total = 0;
    for sh in shapes.split(',') {
        let v: Vec<usize> = sh.split(':').map(|x| x.parse().unwrap()).collect();
        let (m, k, n) = (v[0], v[1], v[2]);
        let mut weights = vec![0u16; n * k];
        f32_slice_to_bf16(&random(n * k, 1, 0.1), &mut weights).unwrap();
        let w = hw.pack_bf16(n, k, &weights).unwrap();
        let x = random(m * k, 2, 2.0);
        let mut bits = vec![0u16; x.len()];
        f32_slice_to_bf16_nthreads(&x, &mut bits, threads).unwrap();
        let (nh, d) = (12usize, 64usize);
        let run = |dirty: u64| -> Vec<f32> {
            match mode.as_str() {
                "src" => {
                    let mut y = random(m * n, dirty, 1000.0);
                    hw.linear_bf16_src(&bits, &w, &mut y, false).unwrap();
                    y
                }
                "f32" => {
                    let mut y = random(m * n, dirty, 1000.0);
                    hw.linear_bf16(&x, &w, &mut y, false).unwrap();
                    y
                }
                _ => {
                    let s = m;
                    let fused = random(s * 3 * nh * d, 7, 2.0);
                    let half = d / 2;
                    let (cos, sin) = (random(s * half, 8, 1.0), random(s * half, 9, 1.0));
                    let mut q = vec![0f32; s * nh * d];
                    let (mut kk, mut vv) = (q.clone(), q.clone());
                    layout_qkv_rope(
                        &fused, &mut q, &mut kk, &mut vv, 1, s, nh, d, &cos, &sin, 0.125, threads,
                    )
                    .unwrap();
                    let pad = vec![vec![]];
                    let shape = Attention {
                        batch: 1,
                        seq: s,
                        q_heads: nh,
                        kv_heads: nh,
                        head_dim: d,
                        tile: 256,
                        window: (mode == "attn_local").then_some(64),
                        scale: 1.0,
                        padding: &pad,
                    };
                    let y = hw.attention_gqa(&shape, &q, &kk, &vv).unwrap();
                    let _ = dirty;
                    y[..s * nh * d].to_vec()
                }
            }
        };
        let first = run(1);
        let mut bad = 0;
        // [bad, good] x [with switches, without]
        let mut corr = [[0usize; 2]; 2];
        for r in 0..reps {
            let sw0 = involuntary_switches();
            let y = run(100 + r as u64);
            let switched = involuntary_switches() > sw0;
            corr[usize::from(y == first)][usize::from(!switched)] += 1;
            if y != first {
                bad += 1;
                if bad <= 2 {
                    let idx: Vec<usize> = (0..y.len()).filter(|&i| y[i] != first[i]).collect();
                    let maxerr = idx
                        .iter()
                        .map(|&i| (y[i] - first[i]).abs())
                        .fold(0f32, f32::max);
                    eprintln!(
                        "  bad rep {r}: {} diff values, idx {}..{}, max err {maxerr}",
                        idx.len(),
                        idx[0],
                        idx[idx.len() - 1]
                    );
                }
            }
        }
        eprintln!(
            "RESULT isa={} thr={threads} mode={mode} m={m} k={k} n={n}: {bad}/{reps}",
            rt.isa()
        );
        eprintln!(
            "  CTX bad&switched={} bad&quiet={} good&switched={} good&quiet={}",
            corr[0][0], corr[0][1], corr[1][0], corr[1][1]
        );
        total_bad += bad;
        total += reps;
    }
    eprintln!(
        "TOTAL isa={} thr={threads} mode={mode}: {total_bad}/{total}",
        rt.isa()
    );
}
