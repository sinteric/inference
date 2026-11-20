# inference

A Rust library that runs a few selected models, each tuned for specific
hardware. It is not a general-purpose runtime: every model here has its own
graph, fusions and kernels, written and measured for the hardware it targets,
and a model is added only when it is worth that work.

It is a library and nothing else: an application links it and runs the
models in its own process. It has no notion of environments, so there is no
production or development setting and no default for one. The application
chooses the model, the threads, the ISA cap and the caches through
`cpu::Runtime` and each model's `Config`; the library picks each model's
precision for the CPU it runs on and says which ([library guide](docs/library.md)).

## Models

| Model | What it does | Precision | Module |
|---|---|---|---|
| [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) | Embeddings: the text tower, with opt-in image and audio towers (`EmbeddingGemma2::embed_parts`) | BF16 matmuls with an FP32 residual stream, or FP32; FP32 only below AVX-512 BF16 | `embeddinggemma2::cpu` |
| [`Horizon-Labs/prompt-injection-guard-base`](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base) | Prompt-injection guard: a ModernBERT encoder with a two-label head (`Guard::classify`), and its hidden states for heads of a caller's own (`Guard::encode`) | FP32, or BF16 matmuls where the CPU has BF16 kernels | `guard::cpu` |
| PP-OCR | A detector, a Korean CTC recognizer and the UVDoc page unwarper, as three models (`Detector`, `Recognizer`, `Unwarper`) | FP32 | `ppocr::cpu` |

EmbeddingGemma 2's 262k×512 embedding table stays a memory map, so only the
rows a text reaches become resident. The PP-OCR unwarper's grid matches ONNX
Runtime 1.28.0 to 5e-7. Page OCR built from the PP-OCR models, orientation
included, is the caller's.

## Hardware

Each model is tuned per hardware target, and the targets today are x86-64
CPUs through oneDNN 3.13.4's C API, linked statically:

- AMD EPYC Turin (Zen 5, AVX-512 BF16) and Intel Emerald Rapids (AMX) are the
  targets the models are tuned and measured on. The benchmarks there showed
  oneDNN fastest or tied everywhere, at about a third to a fifth of ONNX
  Runtime's RSS ([`docs/results/`](docs/results)).
- PP-OCR, EmbeddingGemma 2 (FP32) and the guard (FP32) are also checked at AVX2, with oneDNN
  capped at AVX2 on an AVX-512 host rather than on an AVX2-only CPU. The
  [library guide](docs/library.md) says what was measured there and what it
  costs.

CPUs are the current focus, not a limit. GPUs (AMD, NVIDIA, Intel) are not
supported yet; when one is needed it gets the same treatment: a few selected
models, each tuned for that hardware. Every model's graph calls the oneDNN CPU
backend directly; there is no layer that pretends another device would fit
its shape.

## Drivers

The gates and benchmarks run each model through a thin command-line driver in
`tools/drivers`, built only on the library's public API and configured by
flags alone:

| Driver | Model | Notes |
|---|---|---|
| `eg2-onednn` | EmbeddingGemma 2, text tower | The bench harness's persistent JSON worker |
| `ppocr-onednn` | PP-OCR | PNG/JPEG pages in, JSONL out, for the OCR gate |
| `ppocr-ort` | PP-OCR on ONNX Runtime 1.28.0 | The OCR gate's reference, behind the drivers' `ort` feature |

## Layout

```
crates/onednn-sys          bindgen FFI for dnnl_* (oneDNN 3.13.4 C API, linked statically)
crates/sinteric-inference  the library: cpu::Runtime + one module per model, with a cpu submodule each; internals private
tools/drivers              command-line drivers for the gates and benchmarks
tools/golden               the refactor gate: every model's outputs bit for bit
tools/ppocr-import         pinned ONNX importer and ORT fixture/qualification tools
tools/amx-tile-state       the AMX tile-state fault the runtime's probe guards against
assets/*.toml              Hub repo / revision / sha256 (bytes not in git)
bench/                     the EG2 broad matrix and the harness (any N threads)
```

## Toolchain

**Rust 1.99.1** (`rust-toolchain.toml`). cmake, g++. `CC=gcc CXX=g++`.

```bash
# asset-free (hosted CI path); tests that need model files skip
# cmake + g++ required. Cache: ONEDNN_STATIC_ROOT=$HOME/.cache/sinteric-sdk/onednn-static-tp
cargo test --workspace --lib --bins --tests

# EmbeddingGemma 2 text tower (mmap embeddings; image/audio: eg2_media_bench example)
python3 scripts/download_assets.py assets/embeddinggemma2.toml
cargo build -p drivers --bin eg2-onednn --release
./target/release/eg2-onednn --model "$EG2_MODEL" --threads 4
# Best ISA by default (AMX on EMR). Match Turin with --pin-isa AVX512_CORE_BF16.

# The guard
python3 scripts/download_assets.py assets/guard.toml
cargo run -p sinteric-inference --release --example guard -- "$GUARD_MODEL" 4 "some text"

# GNU_STACK must be RW
scripts/check-noexecstack.sh target/release/eg2-onednn
```

Thread budgets are any `N >= 1`. Default placement is **one logical CPU per
physical core**, using the **lowest** SMT sibling (matching the harness;
`1,3,5,7` on a 4-core SMT guest is rejected as over-budget). Pass
`--cpuset 0,2,4,6` to override.
The library does not read `available_parallelism` or the process
environment, and it never sets the process's affinity: the drivers pin
themselves to `Runtime::cpus`. Configure threads, ISA and caches through
`cpu::Runtime` and each model's `Config` (see [`docs/library.md`](docs/library.md)).

Weights are mmapped and packed into oneDNN blocked layouts. The packed-weight
cache is **on by default** (`CacheConfig::enabled`). The file
(`model.safetensors.onednn-3.13.4.v5.isa{hex}`) stores the *manifest* sha256,
source size/mtime, oneDNN version, effective CPU ISA, per-tensor packed
memory-descriptor fingerprints, packed GEMM weights, norms and biases — not
the embedding table, which stays a zero-copy mmap view of safetensors
(only gathered rows become RSS). A hit restores packed weights from an mmap
of the cache and drops those pages after each copy. The cache
auto-invalidates if size/mtime, the manifest sha, the oneDNN version, the
effective ISA, or a packed layout tag change. If the cache path is not writable, load packs from safetensors,
says why in the model's `load_report()`, and continues. Full sha256 is opt-in
(`verify_weights`) and is already done at download. Safetensors remains the
source of truth.

## Assets

```bash
python3 scripts/download_assets.py assets/embeddinggemma2.toml
python3 scripts/download_assets.py assets/guard.toml
```

Tests that need those files skip when they are absent. The PP-OCR assets are
converted from the pinned ONNX exports by `tools/ppocr-import`.

## Bench / qualification

```bash
# EG2 broad matrix (merge gate: seqs 7..8192, B1/B4/B16, mixed lengths, 1/2/4/8)
python3 bench/matrix.py --model "$EG2_MODEL" --binary target/release/eg2-onednn \
  --output results/eg2-matrix

python3 bench/harness/benchmark.py --requests evaluation/model-requests.jsonl \
  --commands evaluation/backend-commands.json --scope model \
  --budgets 1,2,4,8 --rounds 3 --warmup 5 --repeat 15 --output results/new-model
```

Hosted CI runs the harness's own unit tests (`bench/tests`).

## Turin

Hosted CI is portable. Build Turin binaries with
`RUSTFLAGS='-C target-cpu=znver5'` on a Zen 5 host; such a binary SIGILLs
elsewhere.

## Contributing

Rust follows the Google Rust Style Guide, with the changes and exclusions in
[`STYLE.md`](STYLE.md). Format with `python3 scripts/fmt.py` (a pinned nightly
rustfmt; CI runs it with `--check`); the compiler is the stable toolchain in
`rust-toolchain.toml`.

A change is judged by the gates first, then by speed and RSS. Bit-identical
output is not required. The gates:

- **EmbeddingGemma 2:** its cosine gates against the PyTorch FP32 fixtures.
- **The guard:** token ids exactly, and the logit, score and hidden states
  against the PyTorch FP32 reference on eight English and Korean texts
  (`crates/sinteric-inference/fixtures/guard/parity.json`), at the best ISA
  and under an AVX2 cap; BF16 keeps every decision.
- **PP-OCR:** the detector's and recognizer's text and polygons exact against
  ONNX Runtime 1.28.0 on the 100-page Korean document gate (60 real pages, 40
  synthetic; not distributed), which they pass today.

A change that only restructures code is held to more than the gates: every
model's output stays bit for bit what it was. `tools/golden` records each
model's outputs on fixed inputs, with call times and peak RSS, and compares
two records; record the base and the change on one machine, at the same ISA
cap and thread count (its crate docs say how and what it cannot see).

What is not done yet, and what was tried and ruled out, is in
[`docs/open-work.md`](docs/open-work.md).

## License

Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Third-party code
and model notices are in
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md). No model weights
or tokenizer files are in this repository: they are downloaded from their
upstream repositories under their own licences, and whoever redistributes them
carries those licences and notices with them.
