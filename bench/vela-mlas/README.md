# Vela Candle/MLAS comparisons

Research-only Rust library and JSON drivers, moved from `vela2-0p3b`.
This package and `crates/candle-mlas` are excluded from workspace default
members. The model and runtime crates have no Candle/MLAS dependencies.

Build both drivers explicitly:

```bash
cargo build -j4 -p vela-mlas --features bench-mlas --release
# Optional standalone MLAS shared library (not needed for cargo check):
native/mlas/build_mlas.sh
```

Binaries keep the names `vela2-mlas` and `vela2-native-text` under
`target/release/`. The Python comparison and quality harnesses that drove them are not part of
this repo; model fixtures, reference runtimes and qualified MLAS libraries are
caller-provided.
CI runs none of these comparisons.

Rust callers now import `vela_mlas::mlas` instead of `vela2_0p3b::mlas`.
The feature gates the comparison library, dependencies and both binaries.
These experimental APIs retain their existing research-only behavior and
are not part of the library API.
