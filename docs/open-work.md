# Open work

Checked against the tree on 2026-11-19; the items about Vela 2.0 and worker
processes left with them on 2026-11-20. Each item says what is missing and why
it matters; none of it is scheduled. When an item lands, delete it here in the
same change.

## Serving robustness

A host that runs these models behind a server needs these more than speed.

- **Input limits.** EmbeddingGemma 2 bounds a text only by the workspace limit,
  so a large text is tokenized before it is refused or cut to the window.
- **Deadlines and cancellation.** Every call blocks, and the compute lock waits
  without a timeout; a dropped async future still runs to completion.
- **Observability.** `backend_stats()` lists every primitive implementation
  it has run, with no bound; there are no compute-lock wait stats and no
  `tracing` feature. Only OCR reports per-stage timings.
- **Warmup and readiness.** Every model can warm up at load, and
  `load_report()` says where its weights came from, but no model reports its
  revision or its limits.
- **Config consistency.** Each model pins its weights its own way: EG2 and the
  OCR models through a TOML manifest, the guard through a SHA-256 field.
- **Hot reload.** Swapping a model under load and unloading the old one after it
  drains is left to each host.
- **Docs and versioning.** Doctests are `ignore`, so drift does not fail the
  build; switch them to `no_run`. The crate is 0.1.0 with no tags.

## Numerics

- **EmbeddingGemma 2 BF16 on AMX changes with the thread count.** On a KVM Xeon
  guest with AMX (2026-11-19), whose tile-state probe passed,
  `embeddings_match_across_thread_counts` fails its native pass: one input
  (batch 3, 22 tokens) embeds at cosine 0.9983 between 1 and 2 threads, below
  the test's 0.9995. The result repeats exactly, so it is not the
  nondeterministic tile-state fault the probe guards against. The pinned
  AVX-512 BF16 and FP32 passes are exact across thread counts on the same
  host. Unexplained; the suspect is a different oneDNN kernel choice per
  thread count. The same cosine, to every digit, before and after the fold
  into one crate (a5804b4 and ca10c37, 2026-11-20).

## Speed and memory

Measured on a 4-core Zen 5 (Ryzen 7 9700X) box unless noted; *est.* marks an
estimate that was never measured.

- **Request coalescing (EmbeddingGemma 2, the guard).** Compute is serialized
  per process, so requests queued behind the compute lock could run as one
  batch with no added wait. One padded batch of 4 ran 2.7× faster than four
  single forwards. Exactness needs a BF16 test that a row's result does not
  depend on how many rows share the matmul (proven today for FP32 only).
- **Shape-free matmuls (oneDNN ukernel brgemm).** Every new sequence length
  costs a plan (~150 ms for Vela 2.0, the encoder the guard replaced) and plan
  memory is what the memory budget
  evicts. oneDNN's experimental ukernel API runs on AVX-512 without AMX and
  would let one kernel cover every length. It needs
  `ONEDNN_EXPERIMENTAL_UKERNEL=ON`, new bindings, and a re-gate of the guard
  and EmbeddingGemma 2 against their PyTorch references. Also the base for a fused attention kernel.
- **OCR recognizer plans independent of width.** Per-width depthwise, stem, pool
  and SE convolutions hold ~399 MiB of plans and cost 150–370 ms on a cold
  page. Padding to width buckets failed before (426/437 lines exact) because
  padding leaked through bias, affine and means.
- **OCR batch-size invariance.** Batching recognizer crops is assumed, not
  proven, to give the same bits as one crop at a time. One test would make the
  batch policy a free tuning knob.
- **EmbeddingGemma 2 glue.** Attention scatter, the gated output and the GELU mix
  write FP32 that is converted to BF16 right after; norms and RoPE split work
  into chunks smaller than a pool round trip. *est.* a few percent.
- **EmbeddingGemma 2 BF16 images scale poorly.** BF16 image embedding is only 2×
  faster from 1 to 4 threads (FP32: 2.9×) and peaks ~400 MB above FP32. The
  suspect is BF16 attention at 2,520 tokens. Unexplained.
- **Long EmbeddingGemma 2 inputs (2k+ tokens).** Recomputing per-layer
  embeddings from a saved input would save ~370 MB at 8,192 tokens (*est.*);
  the attention buffers could stay BF16.
- **Packed weights shared across processes.** The packed-weight cache bodies
  are unaligned and copied to the heap. Page-aligned, read-only mapped bodies
  would share weights between processes and across restarts.
- **Allocator and huge pages.** Untried: a 2 MiB-aligned weight arena with
  `MADV_HUGEPAGE`, and jemalloc or mimalloc as the host allocator (which would
  also address the glibc arena growth `MALLOC_ARENA_MAX=2` works around).
  Public evidence is mixed (0–5%).
- **Contention on shared cores.** Pool workers are pinned and spin before they
  park, so one preempted worker stalls a whole region on a shared cloud core.
  No benchmark with busy neighbour cores exists yet; build one, then try
  mask-only pinning, passive waits and more chunks.

## Models

- **OCR layout and tables.** PP-DocLayoutV3 (layout) and SLANet_plus (tables) are
  not ported. Their backbones reuse what exists; deformable-attention sampling
  with top-k, and an autoregressive decoder with single-row matmuls, are new.
- **Korean recognizer spacing.** On synthetic Korean pages, line accuracy
  counting spaces was 0.54 while character error was low: most misses are
  word spacing. The cheapest accuracy gain found so far. Not started.
- **EmbeddingGemma 2 video.** The upstream model takes video; this port takes
  text, images and audio.

## Ruled out

Tried or checked, with the reason; do not retry without new evidence.

- **Runtime-M matmul (`DNNL_RUNTIME_DIM_VAL`).** On `avx512_core(_bf16)` oneDNN
  falls to its gemm matmuls, which reject the threadpool runtime this repo
  builds. Runtime M works only for AMX int8/bf16.
- **oneDNN Graph SDPA.** No fused FP32/BF16 kernel on x64 CPUs; it decomposes
  into the same per-shape primitives.
- **Persistent JIT cache (`cache_blob`).** GPU only in oneDNN.
- **oneDNN 3.14.** Its release candidate had nothing for AVX-512, Zen 5, brgemm,
  ukernels or memory (checked 2026-10-19).
- **int8 for EmbeddingGemma 2.** ~2 ms saved on short queries; Gemma's activation
  outliers put the cosine gate at risk.
- **Winograd, K-split or BF16 for OCR.** Each changes accumulation order and
  breaks exact text parity.
- **Coarser detector input grid.** 291/446 boxes exact at a 64-pixel grid.
- **Padded recognizer batches.** 426/437 lines exact.
- **Pool spin and caller-drain knobs, forked convolution tile copies.** No gain,
  or slower (tile copies on the pool cost 10%).
- **FP32-packed EmbeddingGemma 2 weights.** +260 MiB RSS and 1.2–1.5× slower on
  short inputs, for 0–4% on inputs of 512 tokens or more.
