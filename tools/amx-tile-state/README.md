# Raw AMX tile-state stress (no oneDNN)

`amx_stress.c`: N threads, each loads a tile config, then repeats a fixed
16x16x256 BF16 `tdpbf16ps` GEMM and compares the stored C tile with the first
result. A thread that is preempted while its tiles are live must resume with
the same tile data. On the affected VM (Emerald Rapids under KVM, guest kernel
6.12.94) it resumes with zeroed tiles: every differing result is all zeros.

`./build.sh` builds and runs four cases. Measured on that VM:

| case | bad / iterations |
|---|---|
| 1 thread alone | 0 / 20M |
| 1 thread + busy loop on the same CPU | 444-563 / 30M (two runs counting switches: 453 bad of 622 involuntary context switches, 1,522 of 2,274) |
| 8 threads | 779 / 160M |
| 16 threads | 4,109 / 320M |

Not an oneDNN or thread-pool bug: no oneDNN code is involved.

## What it does to a model, and the runtime's fallback

Through oneDNN the same fault shows up as a nondeterministic BF16 AMX matmul:
with two or more busy threads, `linear_bf16_src`, `linear_bf16` and
`attention_gqa` occasionally returned one wrong block of 1,024 values (a 32×32
C tile group) off by up to ~1.4 where outputs are O(10). In a 22-layer
ModernBERT encoder that became run-to-run differences of up to 0.69 in a
classifier logit. The AVX-512 kernels never did this.

`crates/rt-runtime/tests/amx_bf16_race_repro.rs` reproduces it with no model
(random BF16 weights of a QKV shape, busy threads keeping the machine
oversubscribed), and `amx_bf16_race_sweep.rs` sweeps shapes, modes and thread
counts. Both are ignored, because they are statistical and take minutes:

```bash
cargo test --release -p rt-runtime --test amx_bf16_race_repro -- --ignored --nocapture
AMX_REPRO_NO_PROBE=1 cargo test ...            # AMX unguarded: fails on an affected VM
AMX_REPRO_ISA=AVX512_CORE_BF16 cargo test ...  # pinned: expected 0 differing runs
```

The runtime guards against it: the first load in a process runs a ~40 ms tile
probe (`rt_runtime::amx_health()`) and, on a host that fails it, resolves
`best_available` to `AVX512_CORE_BF16` with a warning on stderr
([`docs/library.md`](../../docs/library.md), "AMX tile-state probe"). The
guarded default is bit-repeatable. `rt_runtime::set_amx_probe(false)` keeps
AMX regardless, for a host known to be fixed. What the probe cannot see: a host
that loses tile state only under a load pattern the 40 ms probe does not
create; it reduces the risk, it does not prove the host sound.
