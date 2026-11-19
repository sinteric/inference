# Dependency and model notices

This experiment distributes authored Rust/C++ inference and benchmark source.
No model weights, tokenizer assets, private evaluation text/predictions, original
Python runtime, or private MLAS checkout are included.

The Vela driver uses vllm-sr/Vela-2.0-0.3B revision
`d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409`. The asset notices inspected before
inference identify Decision contributions under Apache-2.0, retained upstream
Vela/mmBERT notices under MIT, Transformers under Apache-2.0, and Gemma tokenizer
terms. Retain those notices with separately obtained model/tokenizer assets.

The independent native frontend is adapted from the existing authored,
qualified Rust Vela experiment. Its Candle model adapter is removed.

oneDNN 3.13.4 is pinned to `c531fe1a5cd631b7f60277cdf6675c588cadbc28`
and uses Apache-2.0. The public oneDNN library is built into this experiment's
isolated prefix, with CPU THREADPOOL and GPU disabled; original installs are unchanged.
Rust dependencies retain their upstream notices: tokenizers (Apache-2.0),
fancy-regex, serde, serde_json, anyhow, libc, libloading, memmap2 and their
transitive dependencies. Consult the locked dependency versions and upstream
package notices for redistribution.

No precision, upstream library code, ISA reporting, global resource limits,
or security settings are changed by this implementation.
