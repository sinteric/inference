# inference

An inference runtime for a small set of selected models. Each model gets its
own graph, fusions, and kernels, highly optimized for specific target
hardware. This is not a general-purpose runtime.

Current models: [`vllm-sr/Vela-2.0-0.3B`](https://huggingface.co/vllm-sr/Vela-2.0-0.3B)
(FP32 ModernBERT encoder guard) and
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
(BF16 oneDNN matmuls with an FP32 residual stream, or FP32 matmuls; mmap'd
262k×512 embedding table; opt-in image and audio towers through
`api::EmbeddingGemma2::embed_parts`; `eg2-onednn` drives the text tower).
PP-OCR in `models/ppocr` provides the FP32 detector, Korean CTC recognizer and
UVDoc page unwarper as separate models (`ppocr::{Detector, Recognizer,
Unwarper}`); page OCR built from them, orientation included, is the caller's.
The detector and recognizer pass exact text parity with ORT 1.28.0 on the
100-page Korean document gate; the unwarper's grid matches ORT 1.28.0 to 5e-7.
Shipped CPU targets today include AMD EPYC Turin (AVX-512 BF16) and Intel
Emerald Rapids (AMX); the runtime is not limited to one ISA. PP-OCR's
convolutions also run on AVX2, checked with oneDNN capped to AVX2 rather than
on an AVX2-only CPU ([library guide](docs/library.md)).

**Production path: Rust + oneDNN 3.13.4 C API.** Production crates have no
Candle/MLAS dependency. Comparisons live in the non-default `bench/vela-mlas`
crate behind its `bench-mlas` feature.

| Driver | Origin | Role |
|---|---|---|
| `vela2-onednn` | oneDNN C API (`onednn-sys` + `rt-onednn`) | **prod default** — strict fpmath, kind-7 GeGLU, packed reorder |
| `eg2-onednn` | same Hardware / oneDNN C API | EmbeddingGemma 2 text tower — BF16 matmul, FP32 stream |
| `ppocr-onednn` | FP32 oneDNN convolution + SVTR | Flat OCR; decoded pixels in library, PNG + JSONL gate driver |
| `vela2-mlas` / `vela2-native-text` | `bench/vela-mlas` (`--features bench-mlas`) | comparison only |

Frozen gate those experiments published: **227 cases, zero mismatches, atol 1e-4**
vs FP32 ONNX. Asset-free primitive tests vs the previous oneDNN path are within
**1e-6** (matmul / GeGLU / LN / softmax).

Benchmarks on OCI E6 Turin and Intel Emerald Rapids showed oneDNN fastest or
tied everywhere, at about 1/3–1/5 of ORT's RSS. See `docs/results/`.

## Layout

```
crates/rt-*          model-independent (core, threads, assets, text, qual, backend trait)
crates/onednn-sys    bindgen FFI for dnnl_* (oneDNN 3.13.4 C API)
crates/rt-onednn     safe wrapper + OnednnCpu Hardware impl + packed-weight cache
crates/rt-runtime    model-agnostic Runtime (process globals) + Model load trait
crates/sinteric-inference  production library: Runtime + one module per model
crates/candle-mlas   research numerical backend (not a default-member)
bench/vela-mlas      Candle/MLAS comparison library + drivers (not a default-member)
models/vela2-0p3b    Vela graph over the Hardware trait; `api::Vela2`; vela2-onednn driver
models/embeddinggemma2  EmbeddingGemma 2 text + image + audio; `api::EmbeddingGemma2`; eg2-onednn driver
models/ppocr        PP-OCR detector, Korean recognizer, UVDoc unwarper; `api::{Detector, Recognizer, Unwarper}`; ppocr-onednn driver
tools/ppocr-import   pinned ONNX importer and ORT fixture/qualification tools
tools/amx-tile-state the AMX tile-state fault the runtime's probe guards against
native/onednn        shared-SDK recipe (python tests). Rust statically links libdnnl.a.
assets/*.toml        Hub repo / revision / sha256 (bytes not in git)
bench/harness        validate.py / benchmark.py / support.py (any N threads)
```

`models/*` may depend on `rt-*`. `rt-*` never depends on a model.

## Toolchain

**Rust 1.99.1** (`rust-toolchain.toml`). cmake, g++. `CC=gcc CXX=g++`.

```bash
# asset-free (hosted CI path; no Candle, no MLAS)
cargo test --lib --bins --tests

# oneDNN backend (prod default; statically linked 3.13.4)
# cmake + g++ required. Cache: ONEDNN_STATIC_ROOT=$HOME/.cache/sinteric-sdk/onednn-static-tp
cargo build -p vela2-0p3b --bin vela2-onednn --release
./target/release/vela2-onednn --model "$SINTERIC_ASSETS" \
  --threads 2 --frontend

# EmbeddingGemma 2 text tower (mmap embeddings; image/audio: eg2_media_bench example)
python3 scripts/download_assets.py assets/embeddinggemma2.toml
cargo build -p embeddinggemma2 --bin eg2-onednn --release
./target/release/eg2-onednn --model "$SINTERIC_ASSETS" \
  --threads 4 --frontend
# Best ISA by default (AMX on EMR). Match Turin with --pin-isa AVX512_CORE_BF16.

# MLAS comparison (optional; pulls Candle)
native/mlas/build_mlas.sh
cargo build -p vela-mlas --features bench-mlas --bin vela2-mlas --release

# GNU_STACK must be RW
scripts/check-noexecstack.sh
```

Thread budgets are any `N >= 1`. Default placement is **one logical CPU per
physical core**, using the **lowest** SMT sibling (matching the harness;
`1,3,5,7` on a 4-core SMT guest is rejected as over-budget). Pass
`--cpuset 0,2,4,6` to override.
Libraries do not read `available_parallelism` or process env. Configure
threads, ISA, cache, and affinity through [`Runtime`](crates/rt-runtime)
(see [`docs/library.md`](docs/library.md)). Process-owned drivers may call
`ThreadBudget::apply_process_policy`.

Weights are mmapped and packed into oneDNN blocked layouts. The packed-weight
cache is **on by default** (`CacheOptions.enabled`; CLI `VELA_PACKED_CACHE=0` disables it). The file
(`model.safetensors.onednn-3.13.4.v5.isa{hex}`) stores the *manifest* sha256,
source size/mtime, oneDNN version, effective CPU ISA, per-tensor packed
memory-descriptor fingerprints, packed GEMM weights, norms and biases — not
the embedding table, which stays a zero-copy mmap view of safetensors
(only gathered rows become RSS). A hit restores packed weights from an mmap
of the cache and drops those pages after each copy. The cache
auto-invalidates if size/mtime, the manifest sha, the oneDNN version, the
effective ISA, or a packed layout tag change. If the cache path is not writable, load packs from safetensors,
logs one stderr warning, and continues. Full sha256 is opt-in
(`VELA_VERIFY_WEIGHTS=1`) and is already done at download. With `--frontend`,
tokenizer load overlaps model load. Safetensors remains the source of truth.

## Assets

```bash
python3 scripts/download_assets.py assets/vela2-0p3b.toml
python3 scripts/download_assets.py assets/embeddinggemma2.toml
```

Tests that need those files skip when they are absent.

## Bench / qualification

```bash
python3 bench/harness/validate.py --requests evaluation/api-requests.jsonl \
  --reference evaluation/api-reference.jsonl --atol 0.0001 \
  --command '["target/release/vela2-onednn","--model","...","--threads","2","--frontend"]' \
  --output results/api-summary.json

# EG2 broad matrix (merge gate: seqs 7..8192, B1/B4/B16, mixed lengths, 1/2/4/8)
python3 bench/matrix.py --model "$EG2_MODEL" --binary target/release/eg2-onednn \
  --output results/eg2-matrix

python3 bench/harness/benchmark.py --requests evaluation/model-requests.jsonl \
  --commands evaluation/backend-commands.json --scope model \
  --budgets 1,2,4,8 --rounds 3 --warmup 5 --repeat 15 --output results/new-model
```

The 227-case corpus is **not** distributed. Hosted CI runs the portable
`bench/tests/test_validation.py` unit tests and skips the oracle.

## Turin

Hosted CI is portable. Build Turin binaries with
`RUSTFLAGS='-C target-cpu=znver5'` on a Zen 5 host; such a binary SIGILLs
elsewhere. See `docs/porting.md` for what the port changed on purpose.

## Contributing

Rust follows the Google Rust Style Guide, with the changes and exclusions in
[`STYLE.md`](STYLE.md). Format with `python3 scripts/fmt.py` (a pinned nightly
rustfmt; CI runs it with `--check`); the compiler is the stable toolchain in
`rust-toolchain.toml`.

A change is judged by the gates first, then by speed and RSS: Vela within
atol 1e-4 on the 227 cases, EmbeddingGemma 2 within its cosine gates against
the PyTorch FP32 fixtures, and PP-OCR text and polygons exact against ONNX
Runtime 1.28.0 on the 100-page gate. Bit-identical output is not required.
What is not done yet, and what was tried and ruled out, is in
[`docs/open-work.md`](docs/open-work.md).

## License

Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Third-party code
and model notices are in
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md). No model weights
or tokenizer files are in this repository: they are downloaded from their
upstream repositories under their own licences, and whoever redistributes them
carries those licences and notices with them.
