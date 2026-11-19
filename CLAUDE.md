# inference — start here

- **What this is** (the models, the drivers, the gates) → [`README.md`](README.md);
  the library API a caller builds against → [`docs/library.md`](docs/library.md).
- **Rust style** — the Google Rust Style Guide is the base; every rule in it is
  kept, changed or excluded in one table, each change and exclusion with its
  reason → [`STYLE.md`](STYLE.md). Every crate under `crates/` and `models/` is
  library code; the drivers and harnesses under `bench/` and `tools/` are
  application code.
- **Formatting** runs a pinned nightly rustfmt: `python3 scripts/fmt.py`, and
  CI checks it with `--check` ([`STYLE.md`](STYLE.md), "Use the formatting
  tool"). The compiler stays on the stable toolchain in `rust-toolchain.toml`.
