# History

This repo started as a port of earlier Vela 2.0 CPU experiments (a
standalone oneDNN runtime and a Candle + MLAS runtime). A reconstructed
skeleton came first, then faithful copies of the qualified sources, and then
**Rust + oneDNN C API** became the only backend of the library crates and
Candle left the default dependency graph. Vela's frozen gate was 227 cases,
zero mismatches, atol 1e-4 against FP32 ONNX; that corpus was never part of
this repo.

On 2026-11-20 the repo was cut down to what it is now: one library crate
over `onednn-sys`. Vela 2.0 left, since nothing linked it any more and the
guard (a ModernBERT encoder like Vela) replaced it. So did the Candle + MLAS
comparison crates, the `Hardware` trait every graph called through, the
worker processes, the environment overrides and an axum test server. The
porting notes of the Vela experiments (`docs/porting.md`) and those crates
are in git history from that date back. The paragraphs below describe the
EmbeddingGemma 2 port as it was made and still hold, except where a name
they use has since gone.

EmbeddingGemma 2 (`eg2-onednn`) is a separate per-model graph on the same
`rt_backend::Hardware` / oneDNN C API path. Vela numerics are unchanged
(FP32 strict). EG2 uses BF16 packed matmuls with FP32 accumulation; RMSNorm,
softmax, RoPE, residual, pooling and L2 stay in FP32. Parity gate: cosine
≥ 0.999 vs sentence-transformers FP32. Packed-weight cache v5 keys on
`dnnl_get_effective_cpu_isa` and packed MD fingerprints so an AMX-built
cache is not restored under `AVX512_CORE_BF16`. The library does not force
`ONEDNN_MAX_CPU_ISA`; `--pin-isa AVX512_CORE_BF16` is an explicit Zen 5
emulation opt-in. On AMX, packed BF16 GEMMs and tightly packed attention QK/PV pad M to the
16-wide `tdpbf16ps` tile so a short last tile does not mix AMX with an
AVX-512 remainder. PV pads only when M is not already a multiple of 16.
Default thread placement keeps the
lowest SMT sibling per core (matching the bench harness). BF16 activation
conversion is timed as `bf16_convert_s`, not `primitive_exec_s`.
Vela 2.0 stays FP32 strict. Packed linears dominate; remaining glue is
QKV YaRN/layout, attention mask/scatter, and first-forward pool fills.
QKV layout+YaRN+Q-scale run in one token-parallel region (same f32 ops);
attention mask/scatter is threaded; softmax stays on oneDNN for
bit-identity. The plan cache (256 entries) used to evict an arbitrary
HashMap entry, which kept evicting plans the next layer needed: with varied
sequence lengths Vela spent ~150 ms per request re-creating plans. A true LRU
for every model cut a 100-distinct-length cold run by ~30% with identical logits.
Padding rows to bound plan count, as OCR does, gave nothing on top and cost
1-5% warm, so Vela keeps exact-shape plans.
EG2 residual adds stay scalar/AVX-512: they follow RMSNorm, so a matmul
sum post-op would be the wrong tensor. Q/K/V pack as one fused matmul and
split in a single pass; Q/K RoPE share one parallel region; PLE residual
is fused add+scale; Q/K/V and post-projection RMSNorms run in-place.
Attention tiles clamp to the live sequence (and sliding window); they do
not pad to 256. Sliding softmax is AVX-512 + the pinned row pool; global
layers still use oneDNN softmax when the score matrix is large. Q/K
RMSNorm+RoPE and V RMSNorm write BF16 so attention skips f32→bf16;
post-attn / post-FFN / PLE residuals fuse RMSNorm with the add (PLE also
applies the layer scalar). Activation workspaces are reserved per
`(batch, seq)` and reused. PLE GELU is fused
with the per-layer mix multiply. Concurrent `embed` uses a bounded pool of oneDNN
stream+scratch workspaces (`exec_slots`, default 1 = serialized).

`crates/sinteric-inference` is the in-process library API
(`Runtime::try_new(RuntimeOptions)`, one module per model with its own `Config`).
oneDNN 3.13.4 is statically linked (`onednn-sys` cmake). The library never
reads process env.
