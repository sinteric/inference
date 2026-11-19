# Using sinteric-inference as a library

`sinteric-inference` is the crate an application links to run these models in
its own process. It exposes one model-agnostic `Runtime` and one module per
model, each with that model's own API: `embeddinggemma2` (embeddings), `vela2`
(Vela 2.0 0.3B) and `ppocr` (PP-OCR detector and Korean recognizer). Each
model module is behind a cargo feature of the same name (all on by default).
The crate ships models and the runtime only: pipelines built from them, such
as page OCR's stage order, score threshold and reading order, belong to the
caller.
CLI/bench binaries (`eg2-onednn`, `vela2-onednn`, `ppocr-onednn`) are thin
wrappers over the same oneDNN
backend.

## Add the dependency

Add a git dependency pinned to a commit, plus a small adapter in your
application:

```toml
# Cargo.toml
sinteric-inference = { git = "https://github.com/sinteric/inference", rev = "<sha>" }
```

```rust
use sinteric_inference::{embeddinggemma2 as eg2, vela2, Runtime, RuntimeOptions};

let opts: RuntimeOptions = toml::from_str(&std::fs::read_to_string("runtime.toml")?)?;
let rt = Runtime::new(opts)?;

let embedder: eg2::EmbeddingGemma2 = rt.load(eg2::Config::new("/models/embeddinggemma2"))?;
let embeddings = embedder.embed(["query text"], eg2::PromptKind::Query)?;
let (dim, data) = (embeddings.dim, embeddings.data);
// or: embedder.embed_into(["query text"], eg2::PromptKind::Query, &mut buf)?;

let vela: vela2::Vela2 = rt.load(vela2::Config::new("/models/vela2-0p3b"))?;
let answers = vela.system_one(&system_one_body)?;
```

`rt.load::<M>(config)` and `M::load(&rt, config)` are the same call; every
model implements `sinteric_inference::Model`. Only loading is uniform: each
model keeps its own inference API.

`RuntimeOptions`, `embeddinggemma2::Config`, and `vela2::Config` are
`Serialize + Deserialize` and `#[non_exhaustive]`. Use `RuntimeOptions::new()` / `Default` and the
per-model `new(model_dir)` constructors plus builders or public field
assignments. Map your application's fields onto them (or nest under `[inference]`).

Required:

- construct **one** [`Runtime`](../crates/sinteric-inference/src/lib.rs) per
  process (`Runtime::new`)
- `model_dir` in each model's `Config` — weights + tokenizer

There is **no** `library` path. oneDNN 3.13.4 is statically linked at build
time. EG2 defaults to BF16 matmul / FP32 stream (`Precision::Fp32` opt-in);
Vela is FP32 strict. There is no separate BF32 precision: oneDNN's BF32 math
mode (FP32 tensors, BF16 matmul) is what `Bf16` already runs.

## A model of your own

A model that lives outside this repository (its weights, heads or
post-processing are its owner's) is a crate of its own built on the `rt-*`
crates: `rt-core` (errors, shapes), `rt-backend` (the `Hardware` trait),
`rt-onednn` (the oneDNN backend and packed-weight cache), `rt-kernels`,
`rt-threads`, `rt-assets` and `rt-runtime` (the `Runtime` and the `Model`
trait it loads). Implement `rt_runtime::Model` for its handle and it loads from
the same `Runtime` as the models here, sharing the pool, the ISA cap and the
AMX probe.

Take every crate from one revision of this repository, the one the
`sinteric-inference` dependency names:

```toml
sinteric-inference = { git = "https://github.com/sinteric/inference", rev = "<sha>" }
rt-core = { git = "https://github.com/sinteric/inference", rev = "<sha>" }
rt-runtime = { git = "https://github.com/sinteric/inference", rev = "<sha>" }
# ... and each other rt-* crate the model uses, at the same <sha>
```

Two revisions do not build: `onednn-sys` links the static oneDNN, which Cargo
allows once per build, and two `rt-runtime`s are two `Model` traits, so the
model would not load from the other's `Runtime`. The `rt-*` crates are this
repository's internals, not a stable API: a revision bump can change them, and
the caller's model moves with it.

## Build

`onednn-sys` builds pinned oneDNN from source (`DNNL_LIBRARY_TYPE=STATIC`,
CPU-only, `ONEDNN_CPU_RUNTIME=THREADPOOL`) unless `ONEDNN_STATIC_ROOT` points
at a prefix that already contains a THREADPOOL `libdnnl.a` for 3.13.4
(`c531fe1a`). An OpenMP archive at that path is not overwritten.

Host requirements: **cmake** and a C/C++ compiler (`gcc`/`g++` or `CC`/`CXX`).
Ninja is used when present. Pinned oneDNN is **3.13.4** (`c531fe1a`). The
resulting binary does not need `libdnnl.so` or `libgomp` at runtime.

GEMMs and glue share one pinned pool (`rt-threads`) via
`dnnl::threadpool_interop::threadpool_iface`. Streams are created with
`dnnl_threadpool_interop_stream_create`. Width comes from
`RuntimeOptions.threads` (`dnnl_threadpool_interop_set_max_concurrency`),
not `OMP_NUM_THREADS`. The library does not mutate process environment or
install constructors that `setenv`.

```bash
# CI cache / local reuse
export ONEDNN_STATIC_ROOT="$HOME/.cache/sinteric-sdk/onednn-static-tp"
cargo build -p sinteric-inference
```

## RuntimeOptions

| Field | Default | Notes |
|---|---|---|
| `threads` | detected physical cores | Pinned pool: GEMMs + glue |
| `cpuset` | physical cores, lowest SMT sibling | `"0,2,4,6"` |
| `pin_process` | `false` | Process affinity only; dedicated process only |
| `cpu_isa` | `best_available` | `avx512_core_bf16` to cap (Turin emulation); `avx2` to try a CPU without AVX-512 |
| `verbose` | `0` | `dnnl_set_verbose` |
| `workspace_limit_bytes` | 512 MiB | Text model retained activation-pool cap (not a per-request limit); also bounds collected embedding text/output buffers. OCR uses its own config limit |
| `exec_slots` | `1` | Bounded oneDNN stream/scratch pool per model |
| `memory_budget_bytes` | `0` (unbounded) | Default per-model budget for what a model keeps between calls; see [Memory budget](#memory-budget). Model configs override it |
| `malloc_arena_max` | `None` | `mallopt(M_ARENA_MAX)` in `Runtime::new`; process-wide, so opt-in |

## Per-model configs

| Field | EG2 default | Vela default | Notes |
|---|---|---|---|
| `model_dir` | required | required | Weights + tokenizer |
| `cache.enabled` | `true` | `true` | Packed-weight cache |
| `cache.dir` / `cache.file` | sidecar next to safetensors | sidecar next to safetensors | Optional explicit directory/file; default filename includes ISA |
| `max_seq_len` | 8192 | model default | Encode/forward cap |
| `max_batch` | unlimited | unlimited | Reject larger batches |
| `memory_budget_bytes` | runtime default | runtime default | Overrides `RuntimeOptions::memory_budget_bytes`; `Some(0)` is unbounded |
| `mrl_dim` | `768` | n/a | EG2 post-hoc 768/512/256/128 |
| `image` / `audio` | `false` | n/a | Load the EG2 image / audio tower (`embed_parts`) |
| `image_tokens` | `ImageTokens::T280` | n/a | EG2 soft tokens per image: `T70`/`T140`/`T280`/`T560`/`T1120` (`70`…`1120` in JSON) |
| `manifest` | embedded EG2 manifest | embedded Vela manifest | Optional explicit TOML path for the sha/size pin |
| `verify_weights` | `false` | `false` | Full safetensors sha256 |

`advanced.tile` is `#[doc(hidden)]`. Omit it: `Runtime::new` picks the
attention tile from a measured per-ISA table (256 on AMX and on
AVX512_CORE_BF16). Sysfs L2 is not trusted — a 512 KiB VM report picked
64 and lost 3–22% on E6. `with_tile` / `--tile` is an experimental
override for new CPUs, not a per-sequence-length constant.

Env vars are **not** read by `Runtime::new` / model `load`. Applications and CLIs
can explicitly capture `EnvOverrides::from_env()` and apply that snapshot to
runtime and model configs with `with_overrides(&env)`.

ISA is applied with `dnnl_set_max_cpu_isa` before any other oneDNN API
(`dnnl_cpu_isa_default` for `best_available`).

**AMX tile-state probe.** Some virtual machines lose the AMX tile registers of
a thread that is preempted mid-GEMM; the thread resumes with zeroed tiles and
an AMX matmul returns an occasional wrong 32x32 block (nondeterministic, only
with more runnable threads than free vCPUs, never with the AVX-512 kernels).
Before the ISA is applied, the first load in the process runs a ~40 ms check
(`rt_runtime::amx_health()`, cached): a tile GEMM that a second thread keeps
preempting on one CPU. If any result is wrong, `best_available` resolves to
`AVX512_CORE_BF16` and a warning is printed to stderr. An explicit
`avx512_core_bf16` skips the check. `rt_runtime::set_amx_probe(false)`, called
before the first load, turns it off and keeps AMX.

## Public API

- `Runtime::new(RuntimeOptions)` → process-global handle (crate
  `rt-runtime`, re-exported). It knows no model. `Error` is re-exported at
  the crate root.
- `embeddinggemma2::EmbeddingGemma2`: `rt.load(embeddinggemma2::Config)` →
  `Send + Sync`. Share with `Arc`.
  - `embed(texts, PromptKind)` → `Embeddings { dim, data }` with `row` /
    `rows` / `n`. `embed_into(&mut [f32])` writes `n * dim` floats.
  - Bundled Gemma tokenizer; swap with
    `EmbeddingGemma2::load_with_tokenizer(&rt, config, tokenizer)` (any
    `embeddinggemma2::Tokenizer`).
  - `embed_parts(&[&[Part]], Option<PromptKind>)` embeds text, images and
    audio in one input (see below).
- `vela2::Vela2`: `rt.load(vela2::Config)` → `Send + Sync`.
  `system_one(body)` is SystemOne, `predict(request)` the generic request
  path, `forward(Inputs)` the tensor path.
- Errors are `sinteric_inference::Error` (`Config`, `Shape`, `Unsupported`,
  `Backend`, …). Bad input does not panic.

### EG2 image and audio

The image and audio towers are opt-in so a text-only process does not pay
for them: `Config::with_image(true)` / `with_audio(true)`. Each input is a
list of parts, embedded the way Sentence Transformers embeds a multimodal
input (text parts concatenated, each image or clip replaced by its soft
tokens, no prompt prefix unless one is passed):

```rust
use sinteric_inference::embeddinggemma2::{self as eg2, ImageView, Part, PixelFormat};

let model: eg2::EmbeddingGemma2 = rt.load(eg2::Config::new(dir).with_image(true).with_audio(true))?;
let image = ImageView::new(&rgb, width, height, width * 3, PixelFormat::Rgb8)?;
let clip: &[f32] = &mono_16khz; // mono, 16 kHz, at most 30 s
let out = model.embed_parts(&[&[Part::Text("a photo of "), Part::Image(image)], &[Part::Audio(clip)]], None)?;
```

- Images arrive decoded (`Rgb8`, `Bgr8` or `Gray8`); the crate has no image
  or audio codec. They are resized (antialiased bicubic, matching the
  reference processor) to fit `Config::with_image_tokens(ImageTokens::T280)`
  soft tokens (the default); more tokens keep more detail and cost more.
- Audio is mono 16 kHz `f32` samples; resample before calling. One soft
  token per 40 ms.
- Media on a model loaded without its tower returns `Error::Unsupported`;
  `supports_image()` / `supports_audio()` report what was loaded.
- Gated against Sentence Transformers FP32 (`tests/multimodal.rs`):
  cosine >= 0.999 for BF16, >= 0.99999 for FP32.


## Concurrency

`embed` / `system_one` / `forward` **block the calling thread**. In Tokio:

```rust
let model = std::sync::Arc::clone(&model);
let texts = vec!["query".to_string()];
let rows = tokio::task::spawn_blocking(move || {
    model.embed(texts, eg2::PromptKind::Query)
}).await??;
```

Concurrent calls on **one** model use a bounded pool of oneDNN
stream+scratch workspaces (`RuntimeOptions.exec_slots`). Default `1`
serializes executes (same performance as a single stream). Raise it so
each in-flight call has its own stream. GEMMs and glue still share the
process-global thread pool and run one `parallel_for` at a time (AMX
tile config stays on one primitive). Two models loaded from the same
`Runtime` share that pool.

### Inside a web server

Measured with `bench/serve-sim` (an axum server embedding all three
models; results in its README):

- All model compute in one process runs one operator at a time, so
  concurrent requests add latency, not throughput: a process serves at
  most `1 / solo latency` requests per second however many arrive.
- Waiting calls get the compute lock in arrival order
  (`rt_threads::ComputeOrder::Fifo`, the default), so concurrent requests
  share the pool operator by operator and a short request is not starved
  by a long one. With EG2, Vela and OCR traffic mixed this cut a 24 ms EG2
  query's p99 from 9.8 s to 4.2 s. A per-model semaphore of 1 in front of
  `spawn_blocking` on top of that gives each model an even share and the
  lowest tails. `exec_slots > 1` adds no throughput.
- For more throughput, run one process per model, each with its own
  `threads` and `cpuset`: the models then run in parallel instead of
  taking turns (on 4 cores, 3x Vela and 1.7x OCR throughput at about
  500 MiB more RSS). [Worker processes](#worker-processes) do this from
  the library. Give each process its own cores; pools that share cores
  do worse.
- Always `spawn_blocking`. Calling a model on a tokio worker stalls every
  other handler on that worker (`/health` p99 1.5 s).
- Memory has three parts. Weights. What each model keeps between calls,
  reached on the first calls of each input shape and flat after that: the
  activation pool (capped by `workspace_limit_bytes`), OCR's convolution
  arena (its last page's size) and cached plans, bounded by count unless
  a [memory budget](#memory-budget) bounds the total in bytes. OCR plans
  per line width (`RecConfig.plan_capacity`, 512) and its plans hold about
  435 MiB on the 100-page gate set. And freed heap that glibc keeps in
  per-thread arenas, which grows with the number of threads calling in;
  `MALLOC_ARENA_MAX=2` in the process environment bounds it at no
  measured latency cost (`RuntimeOptions::malloc_arena_max` sets it from
  the library). `Runtime::load` and `unload` return freed heap
  to the OS; `release_free_memory()` does the same on demand.
- The pinned pool uses only CPUs in the process's affinity mask
  (`taskset`, cpuset cgroups). A CFS CPU quota is not detected: set
  `threads` to the quota.
- `pin_process` sets the calling thread's affinity, which only threads
  created afterwards inherit: create the `Runtime` before starting tokio
  or other thread pools.

## Memory budget

A loaded model keeps memory between calls so the next call is fast: cached
oneDNN plans (one per operator shape), pooled activation buffers, OCR's
convolution arena and execute scratch. Without a budget these are bounded
only by count (plans) or by `workspace_limit_bytes` (the activation pool),
and grow with the number of distinct input shapes seen.

`memory_budget_bytes` bounds the sum per model (per backend: models loaded
through one `Runtime::shared_backend` handle share one budget). Weights are not included. When
a call leaves the model over budget:

1. Pooled activation buffers are dropped (they are cheap to reallocate) and
   new ones are pooled only while they fit.
2. The least recently executed plans are dropped until the model fits again,
   OCR's convolution plans included (about 400 MiB of the recognizer's
   memory after a few hundred line widths). The plan the call is running is
   kept.
3. OCR's convolution arena is cut back to the current page's size instead of
   staying at the largest page seen.
4. Once 64 MiB has been released, freed pages go back to the OS
   (`malloc_trim`).

Calls never fail because of the budget, and results are bit-identical with
and without it: an evicted plan is rebuilt the next time its shape runs. The
cost is latency, the re-plan time when an evicted shape returns. A single
call that needs more than the whole budget still runs; what it allocates is
freed afterwards. The per-request ceilings that do fail are separate:
the OCR models' `workspace_limit_bytes` (1 GiB arena per page) and the embed
output / text batch checks against `workspace_limit_bytes`.

A plan's size is the heap growth while it was built (`mallinfo2`), an
estimate: other threads allocating at the same time skew it, least in a
process that runs one model. `backend_stats()` reports `memory_budget_bytes`,
`retained_bytes`, `plan_bytes`, `conv_plan_bytes`, `budget_evictions`,
`budget_evicted_bytes` and `budget_trims`, plus `plan_capacity` and
`capacity_evictions` (plans the full LRU cache dropped; when this keeps
growing, shapes are thrashing the cache and calls pay plan creation again).

## Worker processes

`worker::Isolated<M>` runs a model in a child process with its own thread
pool, cpuset, heap and memory budget, and has the same call methods as `M`
(`embed`, `embed_ids`, `embedding_dim`; `system_one`, `predict`;
`detect`; `recognize`). The child is the host binary itself, so call
`main_if_worker()` first in `main`:

```rust
use sinteric_inference::worker::{self, Isolated, WorkerOptions};

fn main() -> anyhow::Result<()> {
    worker::main_if_worker(); // in a worker: serves the model, never returns
    let rt = Runtime::new(RuntimeOptions::new())?;
    let eg2 = Isolated::<EmbeddingGemma2>::load(&rt, eg2::Config::new(dir),
        WorkerOptions::new(1).with_cpuset("0"))?;
    let vela = Isolated::<Vela2>::load(&rt, vela2::Config::new(dir),
        WorkerOptions::new(2).with_cpuset("1-2"))?;
    let det = Isolated::<Detector>::load(&rt, DetConfig::new(det_dir),
        WorkerOptions::new(1).with_cpuset("3").with_memory_budget_bytes(512 << 20))?;
    let rec = Isolated::<Recognizer>::load(&rt, RecConfig::new(rec_dir),
        WorkerOptions::new(1).with_cpuset("3").with_memory_budget_bytes(512 << 20))?;
    // The models now compute in parallel, each in its own process.
}
```

- The worker's `RuntimeOptions` are the host runtime's, with `threads`,
  `cpuset` and, when set, `memory_budget_bytes` from `WorkerOptions`, so
  each model can get its own budget. With a cpuset the whole worker is
  pinned (`pin_process`). The budget and `malloc_arena_max` apply inside
  the worker.
- Calls go over a Unix socket pair: JSON for requests and results, raw
  bytes for embeddings and images. Errors keep their kind (`Request`,
  `Config`, `Shape`); other remote errors arrive as `Message`.
- If the worker dies (crash, OOM kill), calls in flight fail with
  `Error::Worker` and the next call starts and loads a new worker. A
  model that fails to load fails each such call the same way.
- Dropping the `Isolated` (or `unload()`) closes the socket; the worker
  exits and all its memory returns to the OS. The worker also exits when
  the host dies. It never writes to the host's stdout.
- Unix only.

## Unloading

`model.unload()` ([`Model::unload`]) frees everything the model owns:
packed weights, weight/embedding mmaps, the oneDNN engine, primitives,
streams, scratchpads and the activation pool. It then calls
`release_free_memory()` (glibc `malloc_trim(0)`) so freed heap pages go back
to the OS rather than staying in the allocator's free lists. It takes the
model by value, so no call can be in flight. For a model shared via `Arc`,
the last owner unloads: `if let Some(m) = Arc::into_inner(model) { m.unload() }`.
A plain `drop` frees the same objects without the trim. The `Runtime`'s
thread pool and ISA cap are process-global and stay; a later load reuses them.

## Process-global vs per-model

`Runtime::new` installs ISA and the pinned thread pool (GEMMs + glue),
reads core topology (sysfs) for the default thread width, and picks
attention tiles from the per-ISA table. Two models loaded from the same
`Runtime` always agree. Each model
has its own engine, execute-slot pool, and weights. `pin_process` is
process-wide affinity — leave it off when the host already pins the process.

A second `Runtime::new` with a different `threads` or `cpu_isa` returns
`Error::Config`. Matching options are idempotent.

A model can run on fewer pool threads than `RuntimeOptions.threads`:
`BackendOverrides::threads` builds and
executes that backend's oneDNN primitives and glue on at most `n` workers
(the first `n`), while other models keep the whole pool. The pool and the
compute lock stay process-global: models on one `Runtime` still take turns
operator by operator, so a narrower model leaves cores to work outside the
pool (another library's threads), not to another model on the pool. For
truly parallel models, use [worker processes](#worker-processes).

## Examples

```bash
cargo run -p sinteric-inference --example embeddinggemma2 -- \
  "$SINTERIC_ASSETS" 4

cargo run -p sinteric-inference --example vela2 -- \
  "$VELA_ASSETS" 2 < body.json
```

## Errors

Every library crate's boundary re-exports the same non-exhaustive,
`thiserror`-based `rt_core::Error` and `Result<T>`. This includes `rt-text`,
`rt-onednn`, `rt-runtime`, both model crates, and `sinteric-inference`. Library APIs no
longer return `anyhow::Error`; application binaries may still use anyhow.

| Error | Meaning / context |
|---|---|
| `Environment { name, reason }` | Invalid explicitly requested environment override; includes its variable name |
| `Config`, `Unsupported` | Invalid options or conflicting process-global settings |
| `Shape` | Invalid batch, sequence, mask, tensor dimensions or indices; the message identifies the failing shape/tensor |
| `Missing` | Required config field or model tensor is absent |
| `File { path, source }`, `Io`, `Json`, `Asset` | Model, tokenizer, manifest or cache file failure; file errors preserve the source path |
| `Tokenizer` | Tokenizer deserialization or encoding failed; original error is retained |
| `SizeOverflow`, `Layout`, `Allocation` | Checked size arithmetic, allocation layout or allocation failure |
| `WorkspaceLimit { operation, requested, limit }` | Collected embedding texts/outputs or execute-slot bookkeeping exceed the configured budget; OCR convolution arena plus scratch exceeds the model's `DetConfig`/`RecConfig.workspace_limit_bytes` |
| `BackendStatus { operation, status }`, `Backend` | Native backend failure |
| `NativeUnavailable` | A requested native SDK backend is not linked |
| `Slice` | A fixed-width byte conversion failed |
| `Message` | General diagnostic from lower-level helpers |
| `Poisoned` | A numerical-state or frontend mutex was poisoned |
| `Worker` | A model's worker process could not start, exited or broke the protocol; the next call starts a new worker |
| `Request(SystemOneError)` | Structured SystemOne validation failure; `detail()` retains location, message and kind |
| `Context { context, source }` | Diagnostic context around a retained original error |

Match variants, with a wildcard for future additions. `File` and `Context`
retain their underlying error; use `std::error::Error::source()` to inspect
nested causes. Error display text is diagnostic, not a stable protocol.

Empty batches, empty/all-padding sequences, invalid token IDs and masks,
ragged Vela tensors, out-of-range head indices, and sequences/batches over
configured limits return errors. EG2 permits ragged batches with matching
per-row masks. The bundled EG2 tokenizer rejects overlong encodings instead
of silently truncating them. Configured limits apply to tensor APIs and to
custom tokenizers as well as bundled tokenizers.

Safetensors readers validate header lengths, tensor byte extents, dtype and
required projection/normalization dimensions. JSON/header inputs are bounded
at 128 MiB before allocation. Structurally invalid packed caches are rejected
or rebuilt from safetensors; cache writing remains best effort. Immutable model
files must remain intact while their memory mappings are in use.

`RuntimeOptions.workspace_limit_bytes` keeps its previous meaning for text model execution: it
caps the **retained** activation pool, not the size of a single request. Any
shape allowed by `max_seq_len` / `max_batch` runs as before (e.g. EG2 at
batch 1 × 8192 tokens or batch 16 × 1024 tokens still needs >1.5 GB of
transient activations with the 512 MiB default). Oversized requests whose
size arithmetic overflows or whose allocation fails return `SizeOverflow` /
`Allocation` instead of aborting; use `max_seq_len` / `max_batch` to bound
request size. `sinteric-inference` additionally rejects collected embedding
text/output buffers (caller-side `Vec`s) larger than `workspace_limit_bytes`
with `WorkspaceLimit`. This budget excludes packed weights and is not a
process RSS limit. Concurrent requests retain the existing execute-pool and
activation-pool behavior.

Numerical-state and frontend lock poisoning returns an error. Scheduling,
statistics and owned-buffer bookkeeping locks recover their guards, since they
do not publish partially modified model weights. User-supplied parallel
callbacks must not panic; unsafe pointer helpers retain their documented safety
contracts.

## Environment overrides

`rt_core::env` is the single runtime environment reader, also exported as
`sinteric_inference::env`. `EnvOverrides` is re-exported at the `sinteric-inference`
crate root. Capture it **once at application startup**, then reuse it. No
runtime/model constructor, asset lookup, cache lookup, or kernel captures it
implicitly. No process-global snapshot is cached, and applying a snapshot
never rereads or changes the environment.

```rust
use sinteric_inference::{embeddinggemma2, vela2, EnvOverrides, Runtime, RuntimeOptions};

let env = EnvOverrides::from_env()?;
let options = RuntimeOptions::new().with_overrides(&env).with_threads(4);
let rt = Runtime::new(options)?;
let eg2 = embeddinggemma2::Config::new("/models/eg2").with_overrides(&env);
let vela = vela2::Config::new("/models/vela").with_overrides(&env);
// Load only the models the application needs.
let embedder: embeddinggemma2::EmbeddingGemma2 = rt.load(eg2)?;
```

A present override wins over the supplied config; an absent override preserves
it. Apply later builders/field assignments when explicit application settings
should win. `RuntimeOptions::with_env_overrides()?` is a convenience for runtime
options alone; use a shared snapshot when configuring models too.
`EnvOverrides::from_lookup` accepts an application-owned lookup for deterministic
configuration without accessing process environment.

| Variable | Type / accepted values | Meaning | Unset default |
|---|---|---|---|
| `ONEDNN_MAX_CPU_ISA` / `DNNL_MAX_CPU_ISA` | `best_available`, `default`, `best`, `AVX512_CORE_BF16`, `AVX2` (case insensitive) | Process ISA cap; first name wins | Preserve config (`best_available`) |
| `DNNL_VERBOSE` / `ONEDNN_VERBOSE` | integer `0..=2` | oneDNN verbosity; first name wins | Preserve config (`0`) |
| `VELA_PACKED_CACHE` | boolean | Packed cache for **either** model (legacy name retained) | Preserve config (`true`) |
| `VELA_PACKED_CACHE_FILE` | nonempty OS path | Explicit packed cache file for either model | Preserve config (sidecar next to weights) |
| `EG2_VERIFY_WEIGHTS` | boolean, or `full` for true | EG2 full weight SHA256 verification | Preserve EG2 config (`false`) |
| `VELA_VERIFY_WEIGHTS` | boolean, or `full` for true | Vela full weight SHA256 verification | Preserve Vela config (`false`) |
| `EG2_MANIFEST` | nonempty OS path | Explicit EG2 pin manifest | Preserve EG2 config (embedded pin) |
| `VELA_MANIFEST` | nonempty OS path | Explicit Vela pin manifest | Preserve Vela config (embedded pin) |
| `SINTERIC_ASSETS` | nonempty OS path | Exact model directory / asset root; overrides `model_dir` when applied to a model config | Preserve required caller path |
| `VELA_STANDALONE_PROFILE` | boolean | Standalone Vela CLI profiling; not applied to library model configs | Off |

Booleans accept `1/0`, `true/false`, `on/off` (case insensitive). Empty or
malformed values, out-of-range numbers, and non-Unicode scalars return
`Error::Environment { name, reason }`. Paths preserve non-Unicode OS strings
and reject empty values/NUL. Both aliases are validated even when one is
shadowed. EG2/Vela manifest and verification overrides are independent; the
old cross-model fallback precedence is removed. `SINTERIC_ASSETS` names one
exact model directory: set the two configs' `model_dir` separately when
loading both models from different directories. Likewise, use separate cache
files for different models when overriding the shared legacy cache-file knob.

The two oneDNN CLIs capture one snapshot for their session/cache/profile
options; `--pin-isa` takes precedence. Their model path still comes from
`--model` (default `model`). Repository asset-backed tests explicitly choose
`SINTERIC_ASSETS` or `$HOME/.cache/sinteric-assets/<model>@<rev12>` in test
support, and skip when assets are absent. HOME is not a library override.

## Paths and caches

`embeddinggemma2::Config::new(model_dir)` and `vela2::Config::new(model_dir)` require the
caller's exact model directory. Model/tokenizer paths never come from HOME,
`/tmp`, the current source checkout, or implicit environment reads. Relative
paths are resolved against the application's working directory. An explicitly
applied `SINTERIC_ASSETS` override can replace the supplied model directory.

Asset lookup uses `AssetRoot::new(exact_root)` or
`AssetRoot::in_cache(caller_cache_root, &manifest)?`; neither chooses a root.
The latter appends `<model>@<rev12>` and validates its path components.
`rt_assets::locate(&manifest, &root, relative_file)?` reports ready/missing
assets without downloading them. Applications own asset provisioning.

Weight identity defaults to each model's embedded `ASSET_MANIFEST`. Set
`manifest: Some(path)` for an explicit pin; no adjacent manifest is discovered
implicitly. Missing or malformed explicit manifests return path-bearing errors.
`verify_weights: true` opts into a full weight SHA256 check.

Packed caches are enabled by default. Without an override they are sidecars
next to `model.safetensors`, keyed by oneDNN version, effective ISA and packed
layout. `cache.file` selects an exact file; otherwise `cache.dir` selects a
caller-owned directory. The file wins when both are supplied. Use different
files for different models. Set `config.cache.enabled = false` to disable
caches when the model directory is read-only. Cache write failures are best effort:
inference can continue from the packed weights in memory. Asset roots and packed caches are
separate concepts; neither library API invents a home or temporary cache root.

## Warmup

EG2 load warmup is opt-in and defaults to **false**. Set
`embeddinggemma2::Config::new(model_dir).with_warmup(true)` or serialize
`advanced.warmup = true`. The advanced field and builder are hidden from the
normal API listing; they are available to callers that want explicit warmup.
Existing JSON/TOML configs without this field continue to default to false.

Warmup runs one dummy forward with a batch of one, 16 token IDs of `2`, and
an all-one attention mask, then discards the output. It uses the normal
shape/ISA-selected execution path to fault working pages and populate execution
caches. Weights and subsequent embedding result bits are unchanged. Warmup
requires no tokenizer call and is independent of caller request caps
(`max_seq_len`, `max_batch`, `mrl_dim`); those caps still apply to requests.
Any failure propagates as a typed load
error retaining the model directory, and no model handle is returned. Warmup adds work before load returns;
no warmup runs during default loading.

The `eg2-onednn` driver accepts `--warmup` (once per loaded instance)
and reports the flag in its ready JSON. Lower-level callers may explicitly
call `embeddinggemma2::Model::warmup()?`. Vela has no load-warmup setting.

## OCR

`ppocr` replaces the three OCR placeholder crates. It provides
`weights::Assets::load(&AssetConfig::new(explicit_dir, ModelKind::Det|Rec))`,
validated graph descriptions, mmap-backed FP32 tensor loading, the Korean
dictionary and detector/recognizer configurations. Graph and dictionary files are
always checked against hashes compiled from `assets/ppocr.toml`; optional
`verify_weights` hashes tensor payloads too. Safetensors names, byte extents,
dtypes and shapes are checked regardless of that option. Source files must
remain immutable while mapped. Paths are explicit; libraries read no env.
See [the importer instructions](../tools/ppocr-import/README.md) for provisioning.

`rt_onednn::conv::{ConvWeights, ConvNet, Node, Op, PostOp}` provides FP32
convolution/depthwise/deconvolution, pooling, resampling, reduction, concat,
PReLU, reorders and activation/binary post-ops. Create it from an existing
`Arc<OnednnCpu>`; the existing pinned pool, ISA and strict FP32 policy apply.
Weights use explicit tags fixed by channels/ISA, independent of H/W. Graph
inputs/outputs are NCHW; internal activations use channel blocks of the ISA's
vector width, nChw16c with AVX-512 and nChw8c below it, with explicit plain
reorders where required by optimized small-channel conv, nonzero-preserving
eltwise, reduction and PReLU implementations. The block follows the ISA
because oneDNN's AVX2 convolutions take only 8-channel blocks: given 16, every
convolution fell to the reference kernel (measured 2026-11-01 on a 32-channel
5x5: 19 MMAC/s against 74 GMAC/s for the AVX-512 kernel). oneDNN names no
8-channel deconvolution weight layout, so below AVX-512 a deconvolution is
`Unsupported`; the detector's upsampling is a convolution and depth-to-space
and is unaffected. A convolution whose padding is wider than the jit kernel's
register block also falls to the reference kernel (oneDNN 3.13.4 requires
left padding no wider than that block, which shrinks with the ISA's vector
width). When the plan sees that, it zero-pads the input itself and runs the
convolution unpadded, the same sums on a jit kernel: UVDoc's widest bridge
convolutions need this on every ISA, its dilated 5x5s only on AVX2. The probe
asks oneDNN without the node's post-ops, so a convolution that its post-ops
alone send to the reference kernel is padded for nothing and stays there.

Under the `avx2` cap (measured 2026-11-01 on a shared 4-vCPU AVX-512 container, one
960×960 synthetic page for the detector and a random 2×48×320 batch for the
recognizer, against ORT 1.28.0) no OCR primitive uses a reference kernel, the
detector's probability map stays within 4.8e-4 with no pixel crossing the 0.2
threshold, and the recognizer's argmax is identical. The detector took about
1.3 times and the recognizer 1.6 times as long as under `avx512_core_bf16`.
EmbeddingGemma 2's FP32 graph under the `avx2` cap matches the PyTorch FP32
fixtures at cosine 1.0 to float precision (2026-11-19,
`tests/parity_pinned_avx2.rs`). oneDNN has no packed BF16 matmul below
`avx512_core_bf16`, so a caller on an AVX2 CPU picks FP32. Aligned channel concatenation
places producers directly into joined storage at N=1 when their lifetimes permit;
other complete-block concats use blocked copies, and general concats use oneDNN. Spatial means use fixed FP32 accumulation orders to match ORT. General grouped conv
beyond groups=1/depthwise is unsupported. Integral nearest upsampling matches
ONNX asymmetric/floor; linear resampling uses half-pixel centers, not
align-corners.

`ConvNet::new(cpu, nodes, capacity)` takes a configurable positive LRU capacity.
`warmup([N,C,H,W])` builds a shape plan and reports implementation names and
arena/scratch sizes. `run(shape, input)` executes it. Plans share packed
weights and immutable memory descriptors. Steps are cached by actual operand shapes
and graph identity, so SE gates are reused across widths. Identical native descriptors
share primitive code. A largest-first liveness arena borrows the network input.
For OCR, arena **plus largest scratchpad** is rejected with `WorkspaceLimit`
before activation allocation when it exceeds `SessionOptions.workspace_limit`.
This bounds each convolution segment; SVTR and host image/output buffers are
additional. Existing EG2/Vela retained-pool semantics are unchanged. No OCR disk cache is written. Concurrency uses the common compute
lock. Convolution segments on one backend reuse one aligned arena, scratchpad and
stream, growing to the maximum required size. Retained arena plus scratch remains
within the configured convolution bound; host/SVTR buffers remain additional.

The statically linked oneDNN build must enable convolution, deconvolution,
pooling, resampling, concat and PReLU in addition to the text primitives.
`onednn-sys` builds these by default. If a supplied pinned THREADPOOL prefix
omits them, the build fails and names each missing `BUILD_*` define. There is
no silent OUT_DIR fallback. An absent archive is built from source; an existing
narrow 3.13.4 THREADPOOL prefix can be refreshed explicitly:

```bash
CC=gcc CXX=g++ ONEDNN_STATIC_ROOT="$HOME/.cache/sinteric-sdk/onednn-static-tp" \
  ONEDNN_REBUILD_NARROW_PREFIX=1 cargo build -j6 -p onednn-sys --release
```

This opt-in modifies only the named matching prefix; unset it for normal builds.

`ppocr::Detector` executes the hand-written PP-OCRv6 tiny graph. Load it from
the runtime and pass decoded pixels:

```rust
use sinteric_inference::ppocr::{DetConfig, Detector, ImageView, PixelFormat};

let detector: Detector = rt.load(DetConfig::new("/models/ppocr/det").with_verify_weights(true))?;
let image = ImageView::new(&decoded_bgr, width, height, stride, PixelFormat::Bgr8)?;
let boxes: Vec<[[f32; 2]; 4]> = detector.detect(&image)?;
```

`ImageView` also accepts RGB8 and Gray8, validates nonzero dimensions, row
stride and complete byte extent, and borrows the supplied pixels. The library
contains no image codec. `detect` returns clockwise TL/TR/BR/BL page quads in
Paddle reading order (10px y tolerance). `preprocess`, `OcrNet::forward` and
`det::postprocess` expose the stages for parity and reference adapters.
`warmup(height, width)` builds a cached plan and reports its workspace.

Defaults are BGR ImageNet normalization, Triangle resize to nearest multiple
of 32 (ties to even), **no max-side cap**, DB thresh 0.2, fast polygon score
threshold 0.45, unclip 1.4, 1000 candidates, minimum sides 3/5 pixels. Optional
`max_side` is downscale-only. `max_pixels` defaults to 64 Mi pixels and bounds
source/resized allocations without silently resizing oversized input. All
options and tensor boundaries return typed errors for invalid input.

The Triangle implementation follows `image` 0.25's vertical f32 intermediate,
antialiasing, f32 tap weights and final byte rounding. It is byte-identical on
the four pages and 1000 generated resizes against image 0.25.10; no deviation
is known. Suzuki-Abe contours include holes (RETR_LIST/CHAIN_APPROX_SIMPLE).
Caliper reconstruction, LINE_8 polygon scoring and integer round-join offsets
preserve cv2/pyclipper behavior, including the rounding before integer unclip.

The optimized 2400x1696 detector plan uses **227,942,464 bytes arena +
320 bytes scratch = 227,942,784 bytes** at one thread. The DB head rearranges
weights once into 1x1 convolutions followed by depth-to-space; fused post-ops
are retained. This removes the large deconvolution scratchpad and padded C=1
output. These are planned workspace sizes, excluding image/host output buffers.

`ppocr::Detector` and `ppocr::Recognizer` are each a `Model` with its own
config. They accept decoded, borrowed pixels and return typed errors. Page OCR
is the caller's composition of the two:

```rust
use sinteric_inference::ppocr::{
    DetConfig, Detector, ImageView, PixelFormat, RecConfig, Recognizer,
    DEFAULT_WORKSPACE_LIMIT_BYTES,
};
use sinteric_inference::{BackendOverrides, CpuIsa, Model, Runtime, RuntimeOptions};

let rt = Runtime::new(
    RuntimeOptions::new()
        .with_threads(4)
        .with_cpu_isa(CpuIsa::Avx512CoreBf16),
)?;
// Optional: one backend for both models (about 150 MiB less peak RSS).
let ocr_rt = rt.shared_backend(
    BackendOverrides::default().with_workspace_limit_bytes(Some(DEFAULT_WORKSPACE_LIMIT_BYTES)),
)?;
let det: Detector = ocr_rt.load(DetConfig::new("/models/ppocr/det"))?;
let rec: Recognizer = ocr_rt.load(RecConfig::new("/models/ppocr/rec"))?;
let image = ImageView::new(&decoded_bgr, width, height, stride, PixelFormat::Bgr8)?;
let quads = det.detect(&image)?; // clockwise TL/TR/BR/BL page coordinates, reading order
let lines = rec.recognize(&image, &quads)?; // one per quad: decoded UTF-8, mean confidence
for (quad, line) in quads.iter().zip(&lines) {
    if line.score >= 0.5 {
        // PP-OCR's default threshold; the recognizer itself drops nothing.
        println!("{} {} {:?}", line.text, line.score, quad);
    }
}
det.unload();
rec.unload();
```

The `ppocr` module re-exports `DetConfig`, `RecConfig`, `UvdocConfig`,
`Detector`, `Recognizer`, `Unwarper`, `DetOptions`, `RecOptions`,
`UvdocOptions`, `Batching`, `DecodedLine`, `DetTimings`, `RecTimings`,
`UvdocTimings`, `UnwarpedImage`, `DEFAULT_WORKSPACE_LIMIT_BYTES`, `ImageView`,
`PixelFormat` and `Quad`. Config and result types serialize with serde and are
non-exhaustive. Use constructors/defaults, `with_*` builders and field
assignments. `detect_timed` / `recognize_timed` also return stage timings.

| `DetConfig` / `RecConfig` / `UvdocConfig` field | Default | Notes |
|---|---|---|
| `model_dir` | required | That model's directory (`det/`, `rec/` or `uvdoc/` of an imported root) |
| `workspace_limit_bytes` | `None`: 1 GiB (1,073,741,824 bytes) | Independent of `RuntimeOptions.workspace_limit_bytes`; the backend cap on each planned convolution arena plus native scratch. On a `shared_backend` handle the handle's cap applies and setting this is `Error::Config` |
| `options` | `DetOptions` / `RecOptions` / `UvdocOptions` defaults | The settings described below; Exact recognition batching |
| `memory_budget_bytes` | runtime default | See [Memory budget](#memory-budget); on a `shared_backend` handle set it on the handle |
| `manifest` | embedded PP-OCR manifest | Optional explicit TOML path |
| `verify_weights` | `false` | Check the weight payload hash |

A model loaded on a plain `Runtime` owns its backend (engine, convolution
arena, primitive cache), so it can be shared or unloaded independently.
`Runtime::shared_backend(overrides)` returns a handle whose models all load
onto one backend, keeping one arena sized to the larger plan: on the
100-page gate, two separate backends raised peak RSS by about 150 MiB. The
backend is freed once the handle and its models are dropped. All use the
runtime's ISA cap and pinned pool. The default fits a 2400×2400 detector
plan: **322,560,064 arena bytes plus 320 scratch bytes = 322,560,384 bytes** at
one thread, including alignment. EG2/Vela retain their 512 MiB runtime default.
Set `with_workspace_limit_bytes(bytes)` or the serde field for an explicit
positive cap. Plans exceeding it return `Error::WorkspaceLimit` before
activation allocation. This cap excludes weights, preprocessing, outputs, SVTR
host tensors and cross-segment storage; it is not a process RSS limit.

`manifest` optionally selects an explicit TOML; its graph/dictionary
identity must match the compiled export. No disk packed cache is used in phase A.
`verify_weights` checks that model's payloads. Library loading does not read env or decode
image files. The `ocr` example composes both models on one backend and reads
caller-provided raw BGR bytes:

```bash
cargo run -p sinteric-inference --release --example ocr -- \
  /models/ppocr /pages/page.bgr 1654 2339 4  # root containing det/ and rec/
```

`Detector::detect` and `Recognizer::recognize` block and are `Send + Sync`;
share the models through `Arc` and use `spawn_blocking` in async applications.
A recognizer mutex serializes `recognize` calls to bound concurrent crop and
batch storage, independent of the text model execute-slot setting. The
detector and recognizer share the runtime's pinned pool. No process
environment is mutated. Orientation is not a model: a caller that needs it
can read the page at each quarter turn and keep the best.

Recognizer defaults: full-resolution perspective warp with replicated borders
and cubic interpolation, 90-degree CCW rotation when **h/w >= 1.5**, height 48,
ceil resized width clamped to **16..1600**, bilinear resize and BGR normalization
`(v/255 - 0.5)/0.5`. The independent oracle is **OpenCV 5.0.0.93**: its cubic
warp uses continuous coordinates and FP32 coefficients (4.x quantized cubic
sampling differs). Crop dimensions/pixel limits and quads are checked before
allocation; zero, nonfinite, nonconvex or degenerate crops return errors.

`Recognizer` runs the
hand-written PPLCNetV3 and two SVTR blocks (dim 120, eight heads, SiLU MLP) using
FP32 oneDNN conv/matmul/LN/softmax/SiLU primitives. `OcrNet::forward` returns
`[N,T,11947]` probabilities; `decode` collapses repeats then removes blank 0,
appends dictionary space, and averages kept-step max probabilities.
`recognize(&image, &quads)` crops every quad (bounded by `RecOptions.max_pixels`),
runs the width groups and returns one line per quad, in order; it drops none,
so the score threshold (PP-OCR's default is 0.5) is the caller's. `warmup(batch, width)` reports the convolution segment
plans; matmul/norm/softmax descriptors are populated on first execution.

`Exact` is the default: process widths longest-first, batch only identical
resized widths (up to `RecOptions.max_batch`, default 16), then restore reading
order. `Bucketed` is explicit experimental configuration for round-4 measurement:
zero-pad normalized pixels, apply key masks in both attention blocks, and truncate
CTC to each crop's true token span. Padding still changes convolution edges;
Round-6 four-page qualification found **426/437 exact texts**, so Exact remains
the default. Detector padding to coarser grids also changes boxes and is disabled.
Four convolution segments have bounded shape/step LRUs (default **512** each);
the matmul/norm/softmax plan cache is a true LRU (256 plans by default; the
OCR recognizer raises it to **4096**). Pointwise convolutions
reuse code through independent spatial tiles (powers of two, 16..512 positions),
with each scalar/channel fusion kept on the primitive. Spatial convolutions,
global means and attention retain exact shapes. This preserves tensor/text parity
and avoids padded convolution-edge semantics. The recognizer's segments share its
convolution workspace; host tensors remain separate. CTC argmax uses independent lanes
with the first-index tie rule preserved, shared with the ORT yardstick.
Crops run independently on the pinned
pool and are released after copying into their recognition batch.
The `debug-dumps` feature exposes detector and recognizer backbone prefixes.
With an explicit `EnvOverrides` verbosity of 2 (`ONEDNN_VERBOSE=2` in the CLI),
the driver emits oneDNN create/exec records plus `ppocr_profile` stage timings
and retained workspace/weight bytes. The JSONL result file keeps its existing schema.

The `ppocr` crate's default `cli` feature builds `ppocr-onednn`;
`sinteric-inference` disables that feature, so its dependency path has no codec.
The only added external dependency is the optional pure-Rust `png` codec and its
pure-Rust compression dependencies, used solely by the driver and its tests:

```bash
cargo build -j6 -p ppocr --release --bin ppocr-onednn
ppocr-onednn --model /models/ppocr --threads 4 --warmup 3 \
  --images pages.jsonl --out results.jsonl
```

Input rows are `{"img_id":"page1","path":"/pages/page1.png"}`. The driver appends
`{"img_id":...,"lines":[{"text":...,"score":...,"poly":[[x,y],...]}],"ms":...,"rss_mb":...}`.
Scores round to four decimals, quad coordinates and metrics to one. Wall time
includes preprocessing, inference and postprocessing and excludes PNG decode;
RSS is Linux VmHWM in MiB. Per-image failures append `{"img_id":...,"error":"..."}`.
The first K input images warm up and produce no rows; IDs already in the output
(including error rows) are skipped. Output flushes per image and can be resumed.
`--model` names the root containing `det/` and `rec/`. The driver loads both
on one `shared_backend` handle with the 1 GiB default (`--workspace-limit-bytes BYTES`
overrides it) and drops lines scoring below 0.5, as the ORT reference does.

Round-3 qualification against **ORT 1.28.0**: widths 16/48/320/1600 (N=1) and
N=2 at width 48, maximum probability error **2.2053719e-5**, zero argmax
mismatches. All **3,106,220 probabilities** are bitwise identical at 1/2/4
threads; all inspected conv and 60 SVTR/head primitives use non-`ref:` kernels.
A separate two-crop masked ONNX oracle passes with max error **1.5735626e-5**.
All 446 page crops plus 69 generated quads are exact against cv2
(**14,544,432 normalized values**). Four-page Flat parity is **437/437** exact
line texts (99/70/99/169), box IoU 1.0, CER 0, and maximum score error
**1.0073185e-5**.

Round-6 tiling/arena validation retains **437/437 exact texts, IoU 1.0**;
max score error is **1.04904175e-5**, recognizer max probability error
**2.18153e-5**, and masked-oracle max error **1.5974045e-5**.
The 100-page gate (60 scanned Korean documents and 40 synthetic pages, on a
4-vCPU Ryzen 7 9700X) against pyke ORT 1.28.0 passes: 100/100 pages and
11288/11288 lines have identical text and polygons at 1, 2 and 4 threads.

| threads | p50 ms (ours / ORT) | p95 ms (ours / ORT) | peak RSS MiB (ours / ORT) |
|---|---|---|---|
| 4 | 419 / 844 | 936 / 2891 | 1705 / 2130 |
| 2 | 648 / 934 | 1546 / 3193 | 1665 / 2144 |
| 1 | 1082 / 1119 | 2829 / 3983 | 1598 / 2128 |

### UVDoc unwarper

`ppocr::Unwarper` flattens a photographed page with PaddlePaddle's UVDoc
export (pinned in `assets/ppocr.toml`). Load it like the other two, on the same
backend if you like, and hand it decoded pixels:

```rust
use sinteric_inference::ppocr::{UvdocConfig, Unwarper};

let unwarper: Unwarper = ocr_rt.load(UvdocConfig::new("/models/ppocr/uvdoc"))?;
let flat = unwarper.unwarp(&image)?; // same size and pixel format as `image`
let quads = det.detect(&flat.view()?)?;
```

The page is resized to 712×488 (bilinear, align-corners), the network
predicts a 45×31 grid of normalized source positions, and the page is sampled
through that grid brought back to its own size (bilinear, zeros outside,
align-corners); every value is then clamped to `[0, 1]` and rounded to
8 bits. The convolutions run on oneDNN in FP32, as three networks: the
backbone with the bridge, and the two head convolutions that read
reflect-padded inputs. The resizes, the reflect pads and the sample run on the
host (`rt_image::{resize_linear_align_corners_nchw, reflect_pad_nchw,
grid_sample_coarse_align_corners}`).
`grid` and `sample` expose the two halves; `OcrNet::forward` takes and returns
unit-range RGB planes `[1, 3, H, W]`, as the ONNX export does. `warmup()`
builds every plan the model uses, since its network shapes never change.

The rectified page is a different raster: a box found on it does not address
the caller's pixels, and whether it reads better than the original is the
caller's question to ask. The model cannot tell a flat page from a curved one;
it resamples every page it is given.

`UvdocOptions`: `plan_capacity` (default 2) and `max_pixels` (default 64 Mi,
checked before any allocation). Measured 2026-11-01 against ORT 1.28.0 on a
seeded random 96×64 input and a 960×960 invoice, flat and curled
(`tools/ppocr-import/uvdoc_reference.py`), on a shared 4-vCPU AVX-512 container:

| ISA cap | grid max abs | output mean abs | 8-bit values that differ (of 2,764,800) | 960×960 network, 1 / 4 threads |
|---|---|---|---|---|
| `avx512_core_bf16` | 4.8e-7 | ≤ 1.0e-6 | ≤ 183, each by one level | ~250 / ~95 ms |
| `avx2` | 1.8e-6 | ≤ 2.6e-6 | ≤ 531, each by one level | ~460 / ~150 ms |

ORT took about 380 ms for the whole 960×960 page on one core; resize and
sample add about 70 ms to ours. Smoke evidence only. Outputs are bitwise
identical at 1, 2 and 4 threads under each cap. What these tests cannot see:
photographs of real curved pages (the fixtures' curl is a synthetic
sinusoid), and a real AVX2-only CPU: the `avx2` row is oneDNN capped to AVX2
on an AVX-512 machine.

Asset-backed tests skip when `SINTERIC_OCR_ASSETS` is absent. Tests alone capture
that path using `TestOverrides`; it names the root containing `det/`, `rec/` and
`uvdoc/`.
`PPOCR_TEST_THREADS` controls isolated primitive identity test subprocesses.
`SINTERIC_TEST_CPU_ISA` (`avx2` or `avx512_core_bf16`, the default) caps the ISA
for the convolution and OCR tests, so one AVX-512 machine runs both.
