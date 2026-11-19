# Porting notes: from the Vela experiments to the C API path

The Vela runtime was ported from earlier standalone oneDNN and Candle + MLAS
experiments (see [history](history.md)), then rewritten so Rust binds
oneDNN's C API directly. This page records what changed on purpose and what
must stay identical.

## Deviations (intentional)

1. **C API, no C++ bridge in production.** Every primitive the bridge built
   (`dnnl_matmul_*`, reorder, layernorm, softmax, eltwise, binary, kind-7
   GeGLU post-ops, user scratchpads, strict fpmath/accumulation, primitive
   cache capacity 0) is expressed with the public C API. `bridge.cpp` is kept
   as a numeric reference and is not linked by `build_sdk.sh`.
2. **Candle removed from production crates.** `candle-core` / `candle-nn` /
   `candle-mlas` are used only by the separate `bench/vela-mlas` research crate
   with `--features bench-mlas`. Both research crates are non-default members.
3. **Any thread count.** Drivers no longer reject `N ∉ {1,2,4}`. Default
   affinity is one logical CPU per physical core (sysfs `thread_siblings_list`,
   **lowest** sibling id, matching `bench/harness/support.py:physical_cpus`).
   Process-owned drivers also start a persistent pinned pool on those
   CPUs; oneDNN THREADPOOL GEMMs and glue share it.
4. **Safetensors mmap + packed cache.** On by default (`VELA_PACKED_CACHE=0`
   disables). Default load does not sha256 the 1.18 GiB file (download
   already did). Cache v5 (`SNTPKW05`) stores packed GEMM weights plus small
   raw tensors (norms/biases/log_tau), not the embedding table, keyed by
   effective CPU ISA and per-tensor packed memory-descriptor fingerprints
   (AMX vs AVX512_CORE_BF16 blocking). Embeddings stay a zero-copy mmap view;
   only gathered rows are RSS. Auto-invalidates on sha / size / mtime /
   oneDNN version / ISA / layout tag. Unwritable cache path falls back
   to packing and logs one stderr warning. `VELA_VERIFY_WEIGHTS=1` opts into
   a full hash. Tokenizer load overlaps model load.
5. **Hardware trait.** `rt_backend::Hardware` (matmul+post-ops, attention,
   norm, eltwise, buffer pool) sits between the Vela graph and oneDNN-CPU.
   EG2 adds RMSNorm, GQA attention, `gelu_tanh`, and BF16 pack/linear/mm
   with defaults so Vela still uses the original FP32 methods only.
6. **No `-march=native` in default CI.** Hosted runners are not Turin.
7. **`std::env::set_var` is `unsafe` on rustc 1.99.** Process-owned
   `ThreadBudget::apply_process_policy` pins affinity (CLI only) and does
   not set process environment variables.
   The library path never reads process env; ISA/verbose go through
   `dnnl_set_max_cpu_isa` / `dnnl_set_verbose` before any other oneDNN
   API. EG2 does not force an ISA cap; `--pin-isa AVX512_CORE_BF16` is
   opt-in (`SessionOptions.max_cpu_isa`). Production embedding uses
   `crates/sinteric-inference` (`Runtime`, `embeddinggemma2::Config`, `vela2::Config`).
8. **AMX short-M padding.** Packed BF16 GEMM M is rounded up to 16 rows
   only when `dnnl_get_effective_cpu_isa` includes AMX. Turin (no AMX) and
   bench shapes 128/512/2048/8192 are unchanged.

## What must stay identical

- oneDNN primitive kinds 0/1/2/3/4/5/7, packed-B reorder, GeGLU post-ops
  (MatMul + GELU-erf + binary-mul), strict fpmath/accumulation, user scratchpad,
  primitive cache capacity 0, YaRN tables, prefix-pool f64→f32 rounding.
- Frontend regexes, calibration, span decode, SystemOne.
- Frozen gate: 227 cases, zero mismatches, atol 1e-4 vs FP32 ONNX.
- Asset-free primitive parity vs the previous oneDNN path: within 1e-6.
