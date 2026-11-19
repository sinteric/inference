//! Safe oneDNN 3.13.4 C API wrapper and CPU [`rt_backend::Hardware`].
//!
//! The C++ `bridge.cpp` is no longer on the production path: every primitive
//! (matmul, reorder, layernorm, softmax, eltwise, binary, kind-7 GeGLU
//! post-ops) is built with `dnnl_*` from Rust.
//!
//! # Process-global state
//!
//! Multiple [`OnednnCpu`] instances can coexist (each has its own engine,
//! execute-slot pool, primitive cache, and activation pool). These are
//! process-global and owned by the first successful [`OnednnCpu::load_with`]
//! (or the library `Runtime`) in the process:
//!
//! - **oneDNN ISA cap** (`dnnl_set_max_cpu_isa`)
//! - **Pinned thread pool** (`rt_threads::par_for`) used for GEMMs and glue
//! - **Threadpool max concurrency** (`dnnl_threadpool_interop_set_max_concurrency`)
//!
//! A later load that disagrees is an error. `dnnl_set_verbose` is last-write.
//! On a host that loses AMX tile state when a thread is preempted
//! ([`amx_health`]), a `BestAvailable` cap resolves to `AVX512_CORE_BF16`
//! before it is applied; [`set_amx_probe`] turns that check off.
//! Process CPU affinity: only
//! [`rt_threads::ThreadBudget::apply_process_policy`] (CLI/bench). That path
//! does not mutate process environment.
//!
//! oneDNN is **statically linked** with `DNNL_CPU_RUNTIME=THREADPOOL`. Streams
//! are created with `dnnl_threadpool_interop_stream_create`. `ONEDNN_MAX_CPU_ISA`
//! is not read; the cap is applied through `dnnl_set_max_cpu_isa` before any
//! other oneDNN API.

pub use rt_core::{Error, Result};
mod amx_probe;
mod api;
mod cache;
pub mod conv;
mod cpu;
mod options;
mod strategy;
mod tuning;

pub use amx_probe::{amx_health, amx_probe_enabled, set_amx_probe, AmxHealth};
pub use cache::{
    apply_cache, cache_path_for, cache_path_with, cache_write_failed, empty_layout_tag, file_stamp,
    pinned_dnnl, read_cache, sha256_bytes, write_cache, CacheMeta, CacheWriter, FileStamp,
    LayoutTag, PackedEntry, RawTensor, CACHE_VERSION, LAYOUT_TAG_LEN,
};
pub use cpu::{OnednnCpu, PackedWeight};
pub use options::{CacheOptions, CpuIsaCap, SessionOptions, DEFAULT_WORKSPACE_LIMIT};
pub use rt_backend::{Hardware, Packed};
pub use tuning::{eg2_attn_tile, vela_attn_tile, TuningPlan};

/// `dnnl_cpu_isa_avx512_core_amx` / `avx10_1_512_amx` bit pattern (`0xfef`).
/// Also matches `avx10_1_512_amx_fp16` (`0x1fef`).
#[inline]
pub fn isa_has_amx(isa: u64) -> bool {
    let amx = onednn_sys::dnnl_cpu_isa_avx512_core_amx as u64;
    (isa & amx) == amx
}

pub use strategy::{
    attn_amx_fold_remainder, attn_score_max_query_rows, clamp_attn_tile, IsaStrategy,
    AMX_BF16_TILE_BLOCK, AMX_BF16_TILE_ROWS, ATTN_SCORE_BYTES_CAP,
};

/// Round to an AMX tile (legacy helper for callers outside the residual layout).
pub fn amx_pad_rows(m: usize, isa: u64) -> usize {
    if isa_has_amx(isa) {
        m.div_ceil(AMX_BF16_TILE_ROWS)
            .saturating_mul(AMX_BF16_TILE_ROWS)
    } else {
        m
    }
}

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_support;

#[cfg(test)]
mod tests {
    use super::*;
    use rt_backend::Hardware;
    use rt_kernels::{gelu_erf_scalar, sgemm_ref};
    use rt_threads::ThreadBudget;

    fn gemm_pad_m(m: usize, isa: u64) -> usize {
        IsaStrategy::from_cpu_isa(isa).gemm_pad_m(m)
    }
    fn attn_pad_m(m: usize, isa: u64) -> usize {
        IsaStrategy::from_cpu_isa(isa).attn_pad_m(m)
    }
    fn attn_query_ranges(
        s: usize,
        tile: usize,
        isa: u64,
        batch: usize,
        nq: usize,
        nkv: usize,
        local: bool,
    ) -> Vec<(usize, usize)> {
        IsaStrategy::from_cpu_isa(isa).attn_query_ranges(s, tile, batch, nq, nkv, local)
    }

    #[test]
    fn amx_isa_bits_and_row_padding() {
        assert!(!isa_has_amx(0xe7));
        assert!(isa_has_amx(0xfef));
        assert!(isa_has_amx(0x1fef));
        assert_eq!(amx_pad_rows(19, 0xe7), 19);
        assert_eq!(amx_pad_rows(19, 0xfef), 32);
        assert_eq!(amx_pad_rows(16, 0xfef), 16);
        assert_eq!(amx_pad_rows(128, 0xfef), 128);
        assert_eq!(amx_pad_rows(0, 0xfef), 0);
        assert_eq!(gemm_pad_m(100, 0xfef), 128);
        assert_eq!(gemm_pad_m(129, 0xfef), 192);
        assert_eq!(gemm_pad_m(300, 0xfef), 320);
        assert_eq!(gemm_pad_m(777, 0xfef), 832);
        assert_eq!(gemm_pad_m(1024, 0xfef), 1024);
        assert_eq!(gemm_pad_m(1200, 0xfef), 1200);
        assert_eq!(gemm_pad_m(1248, 0xfef), 1248);
        assert_eq!(gemm_pad_m(7, 0xfef), 16);
        assert_eq!(gemm_pad_m(15, 0xfef), 16);
        assert_eq!(gemm_pad_m(17, 0xfef), 32);
        assert_eq!(gemm_pad_m(19, 0xfef), 32);
        assert_eq!(gemm_pad_m(50, 0xfef), 64);
        assert_eq!(gemm_pad_m(57, 0xfef), 64);
        assert_eq!(gemm_pad_m(64, 0xfef), 64);
        assert_eq!(gemm_pad_m(100, 0xe7), 100);
        assert_eq!(gemm_pad_m(129, 0xe7), 129);
        assert_eq!(gemm_pad_m(777, 0xe7), 777);
        assert_eq!(attn_pad_m(4, 0xfef), 16);
        assert_eq!(attn_pad_m(15, 0xfef), 16);
        assert_eq!(attn_pad_m(16, 0xfef), 16);
        assert_eq!(attn_pad_m(19, 0xfef), 32);
        assert_eq!(attn_pad_m(44, 0xfef), 48);
        assert_eq!(attn_pad_m(50, 0xfef), 64);
        assert_eq!(attn_pad_m(72, 0xfef), 128);
        assert_eq!(attn_pad_m(114, 0xfef), 128);
        assert_eq!(attn_pad_m(200, 0xfef), 256);
        assert_eq!(attn_pad_m(128, 0xfef), 128);
        assert_eq!(attn_pad_m(530, 0xfef), 576);
        assert_eq!(attn_pad_m(19, 0x1), 19);
    }

    fn assert_covers(s: usize, tile: usize, ranges: &[(usize, usize)]) {
        let mut cover = 0usize;
        for &(start, n) in ranges {
            assert_eq!(start, cover);
            cover += n;
        }
        assert_eq!(cover, s, "s={s} tile={tile} {ranges:?}");
    }

    fn ranges(s: usize, tile: usize, isa: u64, batch: usize) -> Vec<(usize, usize)> {
        attn_query_ranges(s, tile, isa, batch, 4, 2, true)
    }

    fn ranges_global(s: usize, tile: usize, isa: u64, batch: usize) -> Vec<(usize, usize)> {
        attn_query_ranges(s, tile, isa, batch, 4, 1, false)
    }

    #[test]
    fn attn_query_ranges_folds_short_tail_on_amx_only() {
        const AMX: u64 = 0xfef;
        const AVX: u64 = 0xe7;
        assert_eq!(
            ranges(777, 256, AMX, 1),
            vec![(0, 256), (256, 256), (512, 265)]
        );
        assert_eq!(
            ranges(777, 256, AVX, 16),
            vec![(0, 256), (256, 256), (512, 256), (768, 9)]
        );
        // Large unaligned folds would copy the whole preceding score tile.
        assert_eq!(ranges(777, 256, AMX, 16), ranges(777, 256, AVX, 16));
        // Aligned grouped M still folds at large batch without pad copies.
        assert_eq!(ranges(776, 256, AMX, 16).last(), Some(&(512, 264)));
        // rem=44 pad waste 8/88; same step_by at every batch and thread count.
        assert_eq!(ranges(300, 256, AMX, 1), vec![(0, 256), (256, 44)]);
        assert_eq!(ranges(300, 256, AMX, 4), vec![(0, 256), (256, 44)]);
        assert_eq!(ranges(300, 256, AVX, 4), vec![(0, 256), (256, 44)]);
        assert_eq!(ranges(330, 256, AMX, 4), vec![(0, 256), (256, 74)]);
        assert_eq!(ranges(330, 256, AMX, 1), vec![(0, 256), (256, 74)]);
        assert_eq!(ranges(270, 256, AMX, 4), vec![(0, 270)]);
        // rem=17, sliding M=34, pad=14 > 34/4 → fold. rem=20 pad=8/40 does not.
        assert_eq!(ranges(273, 256, AMX, 4), vec![(0, 273)]);
        assert_eq!(ranges(276, 256, AMX, 4), vec![(0, 256), (256, 20)]);
        assert_eq!(ranges(100, 256, AMX, 4), vec![(0, 100)]);
        assert_eq!(ranges(100, 256, AVX, 4), vec![(0, 100)]);
        assert_eq!(ranges(512, 256, AMX, 4), vec![(0, 256), (256, 256)]);
        assert_eq!(ranges(512, 256, AVX, 4), vec![(0, 256), (256, 256)]);
        // Global cap: live-row clamp is 248; AMX conservatively uses 240
        // so unaligned grouped tails also fit the 256 MiB score budget.
        let tile_4224 = clamp_attn_tile(256, 4224, false, 512, 16, 4);
        assert_eq!(tile_4224, 248);
        let stepped_4224: Vec<_> = (0..4224)
            .step_by(240)
            .map(|start| (start, 240.min(4224 - start)))
            .collect();
        assert_eq!(ranges_global(4224, tile_4224, AMX, 16), stepped_4224);
        // Large scores limit unaligned folds; aligned folds still avoid copies.
        assert_eq!(
            ranges(4224, 256, AMX, 16).last().copied(),
            Some((4096, 128))
        );
        assert_eq!(
            ranges(4104, 256, AMX, 16).last().copied(),
            Some((3840, 264))
        );
        assert!(attn_amx_fold_remainder(9, 4, 2));
        assert!(attn_amx_fold_remainder(17, 4, 2));
        assert!(!attn_amx_fold_remainder(20, 4, 2));
        assert!(!attn_amx_fold_remainder(44, 4, 2));
        assert!(!attn_amx_fold_remainder(74, 4, 2));
        for s in 1..900 {
            for tile in [16, 64, 128, 256, 512] {
                for batch in [1usize, 4, 16] {
                    let folded = ranges(s, tile, AMX, batch);
                    let stepped = ranges(s, tile, AVX, batch);
                    assert_covers(s, tile, &folded);
                    assert_covers(s, tile, &stepped);
                    let max_n = folded.iter().map(|&(_, n)| n).max().unwrap();
                    assert!(
                        max_n < tile.saturating_mul(2) || s < tile,
                        "s={s} tile={tile} b={batch} {folded:?}"
                    );
                    if folded.len() > 1 && folded != stepped {
                        let min_n = folded.iter().map(|&(_, n)| n).min().unwrap();
                        assert!(min_n >= tile, "s={s} tile={tile} b={batch} {folded:?}");
                    }
                    if s > tile && s % tile != 0 {
                        let rem = s % tile;
                        let folded_m = 2 * (tile + rem);
                        let fits_copy_budget = attn_pad_m(folded_m, AMX) == folded_m
                            || batch * 4 * (tile + rem) * s * 4 <= 16 * 1024 * 1024;
                        if attn_amx_fold_remainder(rem, 4, 2) && fits_copy_budget {
                            assert_ne!(folded, stepped, "s={s} tile={tile} b={batch}");
                        } else {
                            assert_eq!(folded, stepped, "s={s} tile={tile} b={batch}");
                        }
                        let last = *stepped.last().unwrap();
                        assert!(last.1 < tile, "avx tail s={s} tile={tile} {stepped:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn pad_and_ranges_are_shape_only() {
        const AMX: u64 = 0xfef;
        const AVX: u64 = 0xe7;
        // gemm_pad_m / attn_query_ranges take no nthreads. Same shape+ISA
        // is the same pad and the same query split at every pool width.
        for m in [1usize, 15, 17, 50, 57, 64, 129, 300, 777] {
            let p = gemm_pad_m(m, AMX);
            let want = if m == 0 {
                m
            } else {
                let p16 = m
                    .div_ceil(AMX_BF16_TILE_ROWS)
                    .saturating_mul(AMX_BF16_TILE_ROWS);
                if p16 >= AMX_BF16_TILE_BLOCK {
                    m.div_ceil(AMX_BF16_TILE_BLOCK) * AMX_BF16_TILE_BLOCK
                } else {
                    p16
                }
            };
            assert_eq!(p, want);
            assert_eq!(gemm_pad_m(m, AVX), m);
        }
        for s in [1usize, 15, 17, 129, 300, 777, 4224] {
            for tile in [16, 64, 128, 248, 256, 512] {
                for batch in [1usize, 3, 5, 16] {
                    for isa in [AMX, AVX] {
                        for local in [true, false] {
                            let nkv = if local { 2 } else { 1 };
                            let a = attn_query_ranges(s, tile, isa, batch, 4, nkv, local);
                            let b = attn_query_ranges(s, tile, isa, batch, 4, nkv, local);
                            assert_eq!(a, b, "s={s} tile={tile} b={batch} isa={isa:#x}");
                            assert_covers(s, tile.max(1), &a);
                            if !local {
                                let t = clamp_attn_tile(tile, s, false, 512, batch, 4);
                                let c = attn_query_ranges(s, t, isa, batch, 4, nkv, false);
                                let max_n = c.iter().map(|&(_, n)| n).max().unwrap_or(0);
                                let cap = attn_score_max_query_rows(batch, 4, s);
                                assert!(
                                    max_n <= cap,
                                    "global fold exceeded cap s={s} tile={tile}->{t} b={batch} max_n={max_n} cap={cap} {c:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn primitives_match_reference_within_1e6() {
        let budget = ThreadBudget::new(1).unwrap();
        let hw = match OnednnCpu::load(&budget) {
            Ok(hw) => hw,
            Err(e) if e.to_string().contains("process-global") => return,
            Err(e) => panic!("{e}"),
        };
        // Matmul 2x3 @ 3x2, same fixture as bench/tests PrimitiveTests.
        let a = [1f32, 2., 3., -1., 0., 1.];
        let b = [1f32, -1., 2., 1., 0., 2.]; // K=3 N=2 as [K,N] trans=false with br=2?
                                             // C++: g=1,m=2,n=2,k=3, ar=k=3, trans=0, b strides br=n? execute(0,2,2,3,a,b)
                                             // op_create kind 0: a {m,k} strides {ar,1} ar=k=3; b {k,n} trans=0 strides {br,1} br=n=2
                                             // so B is row-major [K,N] = [[1,-1],[2,1],[0,2]]
                                             // A is [M,K] = [[1,2,3],[-1,0,1]]
                                             // C = A @ B = [[5,7],[-1,3]]
        let out = hw.mm(1, 2, 2, 3, 0, 3, 0, 2, false, &a, &b).unwrap();
        let expected = [5f32, 7., -1., 3.];
        for (x, y) in out.iter().zip(&expected) {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
        let mut cref = [0f32; 4];
        sgemm_ref(2, 2, 3, 1.0, &a, 3, &b, 2, 0.0, &mut cref, 2).unwrap();
        for (x, y) in out.iter().zip(&cref) {
            assert!((x - y).abs() < 1e-6, "ref {x} vs {y}");
        }

        let gate = [0.5f32, 2., -1., 3.];
        // Unpacked kind-7 needs packed weights. Pack B as n=2 k=3 from row-major [N,K]:
        // weight storage in pack() is [n,k] = [N,K] = transpose of [K,N] row-major.
        // [K,N] row-major b = n rows of... wait pack src is n*k with n=output, k=input.
        // Linear: y = x @ W^T with W stored [n,k].
        // Our mm used A[M,K] @ B[K,N] with B row-major.
        // Packed linear: x[M,K] @ packed(W[N,K]).
        // W[N,K] row-major = [[1,2,0],[-1,1,2]] which is transpose of B[K,N].
        let w_nk = [1f32, 2., 0., -1., 1., 2.];
        let packed = Hardware::pack(&*hw, 2, 3, &w_nk).unwrap();
        let gated = hw.gated_linear(&a, &packed, &gate).unwrap();
        for i in 0..4 {
            let g2 = gelu_erf_scalar(expected[i]) * gate[i];
            assert!((gated[i] - g2).abs() < 1e-5, "{} vs {g2}", gated[i]);
        }

        let source = [1f32, 2., 4., 2., 4., 8.];
        let scale = [1f32, 2., 0.5];
        let shift = [0.1f32, 0.2, 0.3];
        let ln = hw.layernorm(2, 3, &source, &scale, &shift, 1e-5).unwrap();
        for start in [0, 3] {
            let row = &source[start..start + 3];
            let mean = row.iter().sum::<f32>() / 3.0;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 3.0;
            for j in 0..3 {
                let e = (row[j] - mean) / (var + 1e-5).sqrt() * scale[j] + shift[j];
                assert!((ln[start + j] - e).abs() < 1e-5, "{} vs {e}", ln[start + j]);
            }
        }

        let sm_in = [0f32, 1., 2., -1000., 0., 1000.];
        let sm = hw.softmax(2, 3, &sm_in).unwrap();
        for start in [0, 3] {
            let row = &sm_in[start..start + 3];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let s: f32 = exps.iter().sum();
            let sum: f32 = sm[start..start + 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
            for j in 0..3 {
                assert!((sm[start + j] - exps[j] / s).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn bf16_linear_tracks_fp32() {
        let budget = ThreadBudget::new(1).unwrap();
        let hw = match OnednnCpu::load(&budget) {
            Ok(hw) => hw,
            Err(e) if e.to_string().contains("process-global") => return,
            Err(e) => panic!("{e}"),
        };
        // EG2 linears are 512-wide; tiny 2×3 GEMMs have no AVX512_BF16 kernel.
        let n = 16usize;
        let k = 32usize;
        let m = 8usize;
        let w_nk: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
        let packed_f = Hardware::pack(&*hw, n, k, &w_nk).unwrap();
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        let mut out_f = vec![0f32; m * n];
        hw.linear(&a, &packed_f, &mut out_f, false).unwrap();

        let g = hw.gelu_tanh(1, 1, &[1.0]).unwrap();
        assert!((g[0] - rt_kernels::gelu_tanh_scalar(1.0)).abs() < 1e-5);
        let rms = hw.rmsnorm(1, 2, &[3.0, 4.0], None, 0.0).unwrap();
        let nrm = (rms[0] * rms[0] + rms[1] * rms[1]).sqrt();
        assert!((nrm - 2.0f32.sqrt()).abs() < 1e-5);

        let w_bf: Vec<u16> = w_nk
            .iter()
            .map(|&x| rt_kernels::f32_to_bf16_bits(x))
            .collect();
        // Hosted ubuntu-latest mixes Ice Lake (AVX512_BF16) with Cascade Lake
        // / AMD (no BF16 kernel). Production EG2 packing uses 512-wide GEMMs
        // on Turin; skip the ISA-sensitive check when pd create is unimplemented.
        let packed_b = match hw.pack_bf16(n, k, &w_bf) {
            Ok(p) => p,
            Err(e) if e.to_string().contains("status 3") => {
                eprintln!("skipping BF16 matmul: no oneDNN BF16 kernel ({e})");
                return;
            }
            Err(e) => panic!("pack_bf16: {e}"),
        };
        let mut out_b = vec![0f32; m * n];
        hw.linear_bf16(&a, &packed_b, &mut out_b, false)
            .expect("linear_bf16");
        for (f, b) in out_f.iter().zip(&out_b) {
            assert!((f - b).abs() < 0.15, "fp32 {f} bf16 {b}");
        }

        // Short M (not a multiple of the AMX 16-row tile) must still track FP32.
        let m_short = 19usize;
        let a_short: Vec<f32> = (0..m_short * k)
            .map(|i| ((i % 5) as f32) * 0.2 - 0.4)
            .collect();
        let mut out_f_s = vec![0f32; m_short * n];
        hw.linear(&a_short, &packed_f, &mut out_f_s, false).unwrap();
        let mut out_b_s = vec![0f32; m_short * n];
        hw.linear_bf16(&a_short, &packed_b, &mut out_b_s, false)
            .expect("linear_bf16 short M");
        for (f, b) in out_f_s.iter().zip(&out_b_s) {
            assert!((f - b).abs() < 0.15, "short-M fp32 {f} bf16 {b}");
        }
        // Accumulating into `out` survives the AMX row pad: `sum` adds to
        // what was there instead of to the pad buffer's stale contents.
        let base: Vec<f32> = (0..m_short * n).map(|i| (i % 3) as f32).collect();
        let mut summed = base.clone();
        hw.linear_bf16(&a_short, &packed_b, &mut summed, true)
            .expect("linear_bf16 short M sum");
        for ((s, b), o) in summed.iter().zip(&base).zip(&out_b_s) {
            assert!((s - (b + o)).abs() < 1e-4, "short-M sum {s} vs {b} + {o}");
        }

        // Attention-like PV: g=2, M=19 (not a multiple of 16).
        let g = 2usize;
        let m_a = 19usize;
        let n_a = 32usize;
        let k_a = 32usize;
        let a_g: Vec<f32> = (0..g * m_a * k_a)
            .map(|i| ((i % 5) as f32) * 0.2 - 0.4)
            .collect();
        let b_g: Vec<f32> = (0..g * k_a * n_a)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let out_mm_f = hw
            .mm(
                g,
                m_a,
                n_a,
                k_a,
                m_a * k_a,
                k_a,
                k_a * n_a,
                n_a,
                false,
                &a_g,
                &b_g,
            )
            .unwrap();
        let a_bf: Vec<u16> = a_g
            .iter()
            .map(|&x| rt_kernels::f32_to_bf16_bits(x))
            .collect();
        let b_bf: Vec<u16> = b_g
            .iter()
            .map(|&x| rt_kernels::f32_to_bf16_bits(x))
            .collect();
        let out_mm_b = match hw.mm_bf16_src(
            g,
            m_a,
            n_a,
            k_a,
            m_a * k_a,
            k_a,
            k_a * n_a,
            n_a,
            false,
            &a_bf,
            &b_bf,
        ) {
            Ok(v) => v,
            Err(e) if e.to_string().contains("status 3") => return,
            Err(e) => panic!("mm_bf16_src short M: {e}"),
        };
        for (f, b) in out_mm_f.iter().zip(&out_mm_b) {
            assert!((f - b).abs() < 0.25, "attn short-M fp32 {f} bf16 {b}");
        }
    }

    /// BF16-stored weights for FP32 math must give what packing the same
    /// (exactly widened) weights as f32 gives, for plain and gated matmuls.
    #[test]
    fn bf16_stored_fp32_matmuls_match_f32_weights() {
        let budget = ThreadBudget::new(1).unwrap();
        let hw = match OnednnCpu::load(&budget) {
            Ok(hw) => hw,
            Err(e) if e.to_string().contains("process-global") => return,
            Err(e) => panic!("{e}"),
        };
        for (n, k) in [(512usize, 512usize), (2048, 512), (768, 512), (512, 2048)] {
            let w_bf: Vec<u16> = (0..n * k)
                .map(|i| rt_kernels::f32_to_bf16_bits(((i * 37 % 113) as f32) * 0.01 - 0.55))
                .collect();
            let w_f: Vec<f32> = w_bf
                .iter()
                .map(|&b| rt_kernels::bf16_bits_to_f32(b))
                .collect();
            let packed_f = Hardware::pack(&*hw, n, k, &w_f).unwrap();
            let packed_b = Hardware::pack_bf16_f32(&*hw, n, k, &w_bf).unwrap();
            assert!(packed_b.bytes() < packed_f.bytes(), "BF16 storage");
            for m in [1usize, 7, 19, 64, 300] {
                let a: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 13 % 29) as f32) * 0.07 - 1.0)
                    .collect();
                let mut want = vec![0f32; m * n];
                let mut got = vec![0f32; m * n];
                hw.linear(&a, &packed_f, &mut want, false).unwrap();
                hw.linear(&a, &packed_b, &mut got, false).unwrap();
                for (w, g) in want.iter().zip(&got) {
                    assert!(
                        (w - g).abs() <= 1e-4 * w.abs().max(1.0),
                        "[{m},{n},{k}] {w} vs {g}"
                    );
                }
                let gate: Vec<f32> = (0..m * n).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
                let want = hw.gated_linear_tanh(&a, &packed_f, &gate).unwrap();
                let got = hw.gated_linear_tanh(&a, &packed_b, &gate).unwrap();
                let mut lin = vec![0f32; m * n];
                hw.linear(&a, &packed_f, &mut lin, false).unwrap();
                for ((w, g), (l, gt)) in want.iter().zip(&got).zip(lin.iter().zip(&gate)) {
                    let e = rt_kernels::gelu_tanh_scalar(*l) * gt;
                    assert!((w - g).abs() <= 1e-4 * w.abs().max(1.0), "gated {w} vs {g}");
                    assert!(
                        (g - e).abs() <= 1e-3 * e.abs().max(1.0),
                        "gated {g} vs ref {e}"
                    );
                }
            }
        }
    }

    /// FP32 GQA attention against a scalar softmax(QKᵀ)V reference with
    /// sliding windows, padding and multi-tile query ranges.
    #[test]
    fn attention_gqa_f32_matches_reference() {
        let budget = ThreadBudget::new(2).unwrap_or_else(|_| ThreadBudget::new(1).unwrap());
        let hw = match OnednnCpu::load(&budget) {
            Ok(hw) => hw,
            Err(e) if e.to_string().contains("process-global") => return,
            Err(e) => panic!("{e}"),
        };
        let (nq, nkv, d) = (4usize, 2usize, 32usize);
        for (b, s, tile, local, radius) in [
            (1usize, 7usize, 256usize, false, 0usize),
            (2, 19, 8, true, 5),
            (2, 40, 16, false, 0),
            (1, 33, 256, true, 3),
        ] {
            let q: Vec<f32> = (0..b * nq * s * d)
                .map(|i| ((i * 7 % 23) as f32) * 0.05 - 0.5)
                .collect();
            let k: Vec<f32> = (0..b * nkv * s * d)
                .map(|i| ((i * 5 % 19) as f32) * 0.06 - 0.5)
                .collect();
            let v: Vec<f32> = (0..b * nkv * s * d)
                .map(|i| ((i * 3 % 17) as f32) * 0.1 - 0.8)
                .collect();
            let mut padding = vec![Vec::new(); b];
            if b > 1 {
                padding[1] = vec![s - 2, s - 1];
            }
            let got = hw
                .attention_gqa_f32(
                    &q, &k, &v, b, s, nq, nkv, d, tile, local, radius, 1.0, &padding,
                )
                .unwrap();
            assert_eq!(got.len(), b * s * nq * d);
            let groups = nq / nkv;
            for bi in 0..b {
                for h in 0..nq {
                    let kvh = h / groups;
                    for i in 0..s {
                        let qr = &q[((bi * nq + h) * s + i) * d..][..d];
                        let mut w: Vec<f64> = (0..s)
                            .map(|j| {
                                let blocked =
                                    (local && i.abs_diff(j) > radius) || padding[bi].contains(&j);
                                if blocked {
                                    return f64::NEG_INFINITY;
                                }
                                let kr = &k[((bi * nkv + kvh) * s + j) * d..][..d];
                                qr.iter()
                                    .zip(kr)
                                    .map(|(a, c)| (*a as f64) * (*c as f64))
                                    .sum()
                            })
                            .collect();
                        let mx = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let mut sum = 0.0;
                        for x in &mut w {
                            *x = (*x - mx).exp();
                            sum += *x;
                        }
                        for c in 0..d {
                            let e: f64 = (0..s)
                                .map(|j| w[j] / sum * v[((bi * nkv + kvh) * s + j) * d + c] as f64)
                                .sum();
                            let g = got[(bi * s + i) * nq * d + h * d + c] as f64;
                            assert!(
                                (g - e).abs() < 1e-4,
                                "b{b} s{s} local={local} [{bi},{h},{i},{c}] {g} vs {e}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn gemm_cosine(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b) {
            let x = x as f64;
            let y = y as f64;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    /// The packed BF16 GEMM rows for each M in `ms`, or `None` when the
    /// effective ISA lacks AVX512_CORE_BF16: oneDNN has no BF16 matmul for a
    /// packed weight format there (`pack_bf16` returns unimplemented). On a
    /// CPU that has it, an unimplemented pack fails the test.
    fn packed_bf16_rows(
        threads: usize,
        ms: &[usize],
        pinned: bool,
    ) -> Option<Vec<(usize, Vec<f32>)>> {
        let budget = ThreadBudget::new(threads).unwrap();
        let opts = SessionOptions {
            max_cpu_isa: if pinned {
                CpuIsaCap::Avx512CoreBf16
            } else {
                CpuIsaCap::BestAvailable
            },
            ..SessionOptions::default()
        };
        let hw = OnednnCpu::load_with(&budget, opts).expect("libdnnl");
        assert_eq!(hw.strategy(), IsaStrategy::from_cpu_isa(hw.cpu_isa()));
        let bf16 = onednn_sys::dnnl_cpu_isa_avx512_core_bf16 as u64;
        if hw.cpu_isa() & bf16 != bf16 {
            eprintln!("no AVX512_CORE_BF16: effective isa 0x{:x}", hw.cpu_isa());
            return None;
        }
        if pinned {
            assert_eq!(hw.strategy(), IsaStrategy::Avx512);
        }
        let n = 512usize;
        let k = 2048usize;
        // Nonperiodic, exactly representable BF16 inputs isolate partitioning
        // errors from BF16 quantization. Reference accumulation is FP32.
        let value = |i: usize, seed: u64| {
            let mut x = (i as u64).wrapping_add(seed);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            (((x ^ (x >> 31)) % 127) as f32 - 63.0) / 64.0
        };
        let w: Vec<u16> = (0..n * k)
            .map(|i| rt_kernels::f32_to_bf16_bits(value(i, 17)))
            .collect();
        let packed = hw.pack_bf16(n, k, &w).expect("pack");
        let max_m = *ms.iter().max().unwrap();
        let a: Vec<u16> = (0..max_m * k)
            .map(|i| rt_kernels::f32_to_bf16_bits(value(i, 91)))
            .collect();
        let mut reference = vec![0.0f32; max_m * n];
        for row in 0..max_m {
            for col in 0..n {
                reference[row * n + col] = (0..k)
                    .map(|j| {
                        rt_kernels::bf16_bits_to_f32(a[row * k + j])
                            * rt_kernels::bf16_bits_to_f32(w[col * k + j])
                    })
                    .sum();
            }
        }
        let mut out = Vec::new();
        let mut min_fp32 = 1.0f32;
        for &m in ms {
            // Exercise both the internal pad and the model's padded-once path.
            let pad = hw.strategy().gemm_pad_m(m);
            let mut padded = vec![0u16; pad * k];
            padded[..m * k].copy_from_slice(&a[..m * k]);
            let mut y = vec![f32::NAN; pad * n];
            hw.linear_bf16_src(&padded, &packed, &mut y, false)
                .expect("padded linear");
            assert!(y.iter().all(|v| v.is_finite()), "unwritten output M={m}");
            assert!(y[m * n..].iter().all(|&v| v == 0.0), "nonzero pad M={m}");
            y.truncate(m * n);
            let mut live = vec![f32::NAN; m * n];
            hw.linear_bf16_src(&a[..m * k], &packed, &mut live, false)
                .expect("live linear");
            let explicit_cos = gemm_cosine(&live, &y);
            let max_diff = live
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(explicit_cos >= 0.9999, "internal vs explicit pad M={m} t={threads} cosine={explicit_cos} max_diff={max_diff}");
            let cos = gemm_cosine(&reference[..m * n], &y);
            min_fp32 = min_fp32.min(cos);
            assert!(cos >= 0.9999, "FP32 M={m} t={threads} cosine={cos}");
            for row in 0..m {
                assert!(
                    gemm_cosine(
                        &reference[row * n..(row + 1) * n],
                        &y[row * n..(row + 1) * n]
                    ) >= 0.9999,
                    "FP32 row {row} M={m} t={threads}"
                );
            }
            out.push((m, y));
        }
        eprintln!(
            "GEMM isa={:?} t={threads} min FP32 cosine={min_fp32}",
            hw.strategy()
        );
        Some(out)
    }

    /// Full-width oneDNN partitioning of small and large padded GEMMs.
    #[test]
    fn packed_bf16_matches_across_thread_counts() {
        let ms: Vec<usize> = (1..=33)
            .chain([48, 64, 100, 128, 129, 192, 256, 300, 320])
            .collect();
        let env = rt_core::env::TestOverrides::from_env().unwrap();
        if let Some(t) = env.gemm_threads {
            let Some(rows) = packed_bf16_rows(t, &ms, env.gemm_avx512.expect("child ISA")) else {
                println!("EG2_GEMM_SKIP");
                return;
            };
            print!("EG2_GEMM");
            for (m, y) in rows {
                print!(" {m}:");
                for x in y {
                    print!("{:08x}", x.to_bits());
                }
            }
            println!();
            return;
        }
        let exe = std::env::current_exe().unwrap();
        'isa: for isa in ["native", "avx512"] {
            if isa == "native" {
                let cpus = rt_threads::physical_cpu_ids().unwrap_or_default();
                if !crate::test_support::amx_tile_state_is_preserved(&cpus) {
                    continue;
                }
            }
            let mut references: Vec<(usize, Vec<(usize, Vec<f32>)>)> = Vec::new();
            let mut min_cross = 1.0f32;
            for t in [1usize, 2, 4, 8] {
                if ThreadBudget::new(t).is_err() {
                    eprintln!("skip t={t}: not enough physical cores");
                    continue;
                }
                let out = std::process::Command::new(&exe)
                    .env("EG2_GEMM_THREADS", t.to_string())
                    .env("EG2_GEMM_ISA", isa)
                    .args([
                        "tests::packed_bf16_matches_across_thread_counts",
                        "--exact",
                        "--test-threads=1",
                        "--nocapture",
                    ])
                    .output()
                    .expect("spawn gemm child");
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                assert!(
                    out.status.success(),
                    "child t={t} status={} stdout={stdout} stderr={stderr}",
                    out.status
                );
                eprint!("{stderr}");
                // ⚠ A CPU without AVX512_CORE_BF16 skips both passes, so a
                // green run there, such as on a hosted CI runner without it,
                // proves nothing about packed BF16. libtest prints the
                // child's line after its own "test … " prefix.
                if stdout.contains("EG2_GEMM_SKIP") {
                    eprintln!("skip isa={isa}: no AVX512_CORE_BF16");
                    continue 'isa;
                }
                let line = stdout
                    .lines()
                    .find_map(|l| l.split_once("EG2_GEMM ").map(|(_, data)| data))
                    .unwrap_or_else(|| {
                        panic!("no EG2_GEMM in child t={t} stdout={stdout} stderr={stderr}")
                    });
                let mut rows = Vec::new();
                for part in line.split_whitespace() {
                    let (mh, hex) = part.split_once(':').expect("m:hex");
                    let m: usize = mh.parse().unwrap();
                    assert_eq!(hex.len() % 8, 0);
                    let y: Vec<f32> = (0..hex.len() / 8)
                        .map(|i| {
                            f32::from_bits(u32::from_str_radix(&hex[i * 8..i * 8 + 8], 16).unwrap())
                        })
                        .collect();
                    rows.push((m, y));
                }
                for (previous_threads, expected) in &references {
                    assert_eq!(expected.len(), rows.len());
                    for ((k0, a), (k1, b)) in expected.iter().zip(&rows) {
                        assert_eq!(k0, k1);
                        let cos = gemm_cosine(a, b);
                        min_cross = min_cross.min(cos);
                        assert!(
                            cos >= 0.9995,
                            "M={k0} cosine {cos} between {previous_threads} and {t} threads"
                        );
                    }
                }
                references.push((t, rows));
            }
            eprintln!("GEMM isa={isa} min cross-thread cosine={min_cross}");
        }
    }
}

#[cfg(test)]
mod bad_input_tests {
    use super::*;
    #[test]
    fn malformed_native_requests_return_errors() {
        let budget = rt_threads::ThreadBudget::new(1).unwrap();
        let hw = OnednnCpu::load(&budget).unwrap();
        assert!(hw.pack(0, 1, &[]).is_err());
        assert!(hw.pack(usize::MAX, 2, &[]).is_err());
        assert!(hw.pack_from_cache(usize::MAX, 2, &[]).is_err());
        assert!(hw.mm(0, 1, 1, 1, 0, 1, 0, 1, false, &[], &[]).is_err());
        assert!(hw.mm(1, 1, 1, 1, 0, 1, 0, 1, false, &[], &[1.]).is_err());
        assert!(hw
            .mm(2, 2, 2, 2, usize::MAX, 2, 0, 2, false, &[], &[])
            .is_err());
        assert!(hw.buffer(usize::MAX).is_err());
        // The workspace limit caps pool retention only; larger buffers still allocate.
        assert_eq!(
            hw.buffer(DEFAULT_WORKSPACE_LIMIT / 4 + 1).unwrap().len(),
            DEFAULT_WORKSPACE_LIMIT / 4 + 1
        );
        assert!(hw.reserve(2, usize::MAX).is_err());
        assert!(hw.softmax(1, 0, &[]).is_err());
        assert!(hw
            .layernorm(1, 2, &[1., 2.], &[1.], &[0., 0.], 1e-6)
            .is_err());
        assert!(hw
            .attention(&[], &[], &[], 1, 1, 1, 1, 1, false, 0, &[vec![]])
            .is_err());
        assert!(hw
            .attention(
                &[0.],
                &[0.],
                &[0.],
                1,
                1,
                1,
                1,
                1,
                false,
                usize::MAX,
                &[vec![]]
            )
            .is_err());
    }
    #[test]
    fn invalid_session_options_fail_before_global_installation() {
        let budget = rt_threads::ThreadBudget::new(1).unwrap();
        assert!(OnednnCpu::load_with(
            &budget,
            SessionOptions {
                exec_slots: 0,
                ..SessionOptions::default()
            }
        )
        .is_err());
        assert!(OnednnCpu::load_with(
            &budget,
            SessionOptions {
                exec_slots: usize::MAX,
                ..SessionOptions::default()
            }
        )
        .is_err());
    }
}
