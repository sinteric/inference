# EmbeddingGemma 2 text-tower bench protocol

Driver: `eg2-onednn` (Rust + oneDNN 3.13.4 C API). Matmuls are BF16×BF16
with FP32 accumulation. RMSNorm, softmax, RoPE, residual, pooling and L2
stay in FP32. By default the runtime calls `dnnl_set_max_cpu_isa(default)` so oneDNN
selects the best ISA (AMX on Emerald Rapids, AVX512_CORE_BF16 on OCI E6
Turin). Pin only as an explicit opt-in for Zen 5 emulation:

```bash
# Intel host, match Turin ISA (`dnnl_set_max_cpu_isa`, not the env var)
./target/release/eg2-onednn --pin-isa AVX512_CORE_BF16 --model ... --library ... --threads 4
```

Omit `--tile` to use the measured per-ISA table (256 on AMX and
AVX512_CORE_BF16). `--tile N` is a hidden experimental override, not a
per-shape constant. Do not derive tiles from reported L2.

## Shapes

The merge gate is the **broad matrix**, not the five historical protocol
shapes. `python3 bench/matrix.py` generates prepared-tensor requests:

- sequence lengths **7, 19, 64, 100, 128, 300, 512, 777, 1024, 2048, 4096, 8192**
  (odds / non-multiples of 16 and 64 included on purpose)
- uniform batches **B1 / B4 / B16** at every length
- mixed-length batches **mix-b4-*** and **mix-b16-*** (padded to the max
  length in the batch; `attention_mask` is 0 on pad)
- threads **1 / 2 / 4 / 8 when the machine has that many physical cores**
- first-call latency of each shape and process **cold load**

Token ids cycle a small alphabet. Huge cells use fewer warmup/repeats so
the matrix stays a gate. `--quick` is for iteration only.

A cell **fails the gate** if warmed p50 is **>2% and >0.3 ms** slower than
the baseline JSON (noise floor). Busy cores (`sum(cpu_s)/sum(wall_s)`) and
RSS are reported, not gated.

## Command

```bash
python3 scripts/download_assets.py assets/embeddinggemma2.toml
cargo build -p embeddinggemma2 --bin eg2-onednn --release

# Full gate (AMX / best ISA). Skip thread counts the host does not have.
python3 bench/matrix.py \
  --model "$EG2_MODEL" \
  --binary target/release/eg2-onednn \
  --output results/eg2-matrix

# Turin-emulation ISA
python3 bench/matrix.py --pin-isa AVX512_CORE_BF16 \
  --model "$EG2_MODEL" --binary target/release/eg2-onednn \
  --output results/eg2-matrix-avx512

# Compare a candidate (exit 1 on >2% p50 regressions)
python3 bench/matrix.py --compare results/eg2-matrix/matrix.json \
  --model "$EG2_MODEL" --binary target/release/eg2-onednn \
  --output results/eg2-pr
```

The older five-shape JSONL harness remains for spot checks:

```bash
python3 bench/harness/benchmark.py \
  --requests bench/protocols/embeddinggemma2/requests.jsonl \
  --commands bench/protocols/embeddinggemma2/commands.json \
  --scope model --budgets 4 --rounds 1 --warmup 3 --repeat 10 \
  --timeout 900 \
  --output results/eg2-onednn
```

Replace `{model}` / `{library}` with the local snapshot (`--library` is
ignored: oneDNN is statically linked). `{threads}` is replaced by the
harness. Report p50/p95 from `matrix.json` (or `samples.jsonl`), worker
`VmRSS` / `VmHWM`, `first_call_ms`, and `cold_load`. B16×8192 is dominated
by the four global layers (head_dim 512); the matrix already drops repeats
on that cell.
