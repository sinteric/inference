# Rust style

This repository's Rust follows the [Google Rust Style Guide][g], and through it
the public [Rust Style Guide](https://doc.rust-lang.org/nightly/style-guide/)
and [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/). **Every
rule in the guide applies unless the table below marks it excluded or changed,
and each of those rows carries the reason.** A rule the guide adds later
applies until someone records it here. Why one external base: a written base
ends the "which way is right" review argument before it starts, and the guide
gives a reason for each rule.

Why some rules do not fit: the guide serves google3, Chromium, Android and
Fuchsia, where Rust ships in operating systems and browsers built with GN or
Bazel, with `panic = abort`. This is a set of crates built with Cargo and
linked into other people's programs, with its own pinned thread pool, so the
rules that follow from that setting are excluded.

[g]: https://google.github.io/styleguide/rustguide.html

## Library code and application code

The guide's API rules bind library code and are a strong recommendation for
application code ([§api-design-context][adc]). **Every crate under `crates/`
and `models/` is library code**: other programs build against them, through
`sinteric-inference` or, for a model of their own, the `rt-*` crates
([`docs/library.md`](docs/library.md), "A model of your own"). **The drivers
and harnesses are application code**: the `[[bin]]` targets (`eg2-onednn`,
`vela2-onednn`, `ppocr-onednn`), everything under `bench/` and `tools/`, and
examples and tests.

[adc]: https://google.github.io/styleguide/rustguide.html#api-design-context

## The rules

"Held by" names what catches a break: rustfmt (`python3 scripts/fmt.py
--check`, a CI step), a clippy lint (CI runs clippy with `-D warnings` over the
packages `.github/workflows/ci.yml` names), or review. ⚠ A rule held by review is held only as well as
the reviewer remembers it.

| Guide rule | Here | Held by | Why, where it is not "adopted" |
| --- | --- | --- | --- |
| [Follow public Rust style rules](https://google.github.io/styleguide/rustguide.html#public-rules) | adopted | rustfmt, clippy, review | |
| [Apply API design rules in context](https://google.github.io/styleguide/rustguide.html#api-design-context) | adopted | review | Library code is defined above. |
| [Prefer small crates](https://google.github.io/styleguide/rustguide.html#small-crates) [chrome] | **excluded** | | It exists for Chromium's GN build graph and caching. Our crates follow layers instead: `rt-*` know no model, `onednn-sys` is the FFI, each model is one crate under `models/`, and `sinteric-inference` is the one API a caller links. `models/*` may depend on `rt-*`; `rt-*` never depends on a model. |
| [Flat file layout](https://google.github.io/styleguide/rustguide.html#structure) [chrome] | **excluded** | | The guide itself keeps Cargo's layout for code built with Cargo, which all of ours is. |
| [Name files after the primary item](https://google.github.io/styleguide/rustguide.html#userdefined-item) | adopted | review | Older files move only when they are already being restructured: a sweep would collide with every open branch. |
| [No APIs in non-leaf modules](https://google.github.io/styleguide/rustguide.html#non-leaf-modules) | adopted | review | Same as the row above. |
| [No glob imports](https://google.github.io/styleguide/rustguide.html#glob-imports) | **changed** | review | The guide's own exceptions apply (`use super::*` in tests, `use Enum::*` in a function about that enum). Changed for one more: `sinteric-inference`'s model modules re-export their model crate's `api` whole (`pub use ::embeddinggemma2::api::*`), because each module is that API under a stable path, and naming every item again would make adding one a change in two crates. |
| [One module per `use`](https://google.github.io/styleguide/rustguide.html#use-statement) | adopted | rustfmt `imports_granularity = "Module"` | |
| [Locally scoped trait imports](https://google.github.io/styleguide/rustguide.html#prefer-local-trait-imports) | adopted | review | |
| [Module-qualified calls to top-level functions](https://google.github.io/styleguide/rustguide.html#toplevel-functions) | adopted | review | |
| [Type names imported and unqualified](https://google.github.io/styleguide/rustguide.html#type-names) | adopted | review | |
| [`chromium::import!`](https://google.github.io/styleguide/rustguide.html#import-macro) [chrome] | **excluded** | | It is a Chromium build macro; Cargo resolves crates for us. |
| [Use the formatting tool](https://google.github.io/styleguide/rustguide.html#formatting-other) | **changed** | rustfmt (`scripts/fmt.py --check`, a CI step) | The guide's three non-default options exist only in nightly rustfmt, so `python3 scripts/fmt.py` runs a pinned nightly rustfmt (the script names it). The compiler stays on stable (`rust-toolchain.toml`) and nothing else uses nightly. `crates/onednn-sys/vendor/` is oneDNN's own headers and is not ours to format. |
| [Opting out of formatting](https://google.github.io/styleguide/rustguide.html#formatting-optout) | adopted | review | A test that reads source text is not a reason to skip formatting: such a test should parse, not count lines. |
| [Avoid `Rc` and `Arc`](https://google.github.io/styleguide/rustguide.html#avoid-rc-arc) | adopted | review | The guide's exception covers what a shared model needs: a loaded model is `Send + Sync` and callers share it by `Arc`. |
| [Borrows over deep copies](https://google.github.io/styleguide/rustguide.html#prefer-borrows) | adopted | review | |
| [Flexible references for immutable data](https://google.github.io/styleguide/rustguide.html#flexible-references) | adopted | clippy `ptr_arg`, `borrowed_box` | |
| [Iterators vs owned collections in returns](https://google.github.io/styleguide/rustguide.html#return-iterators) | adopted | review | |
| [Iterators vs slices in parameters](https://google.github.io/styleguide/rustguide.html#accept-iterators) | adopted | review | Kernels take slices: a kernel reads a contiguous buffer by index, often from several threads, which an iterator cannot give it. |
| [Accepting `Into`](https://google.github.io/styleguide/rustguide.html#accepting-into) | adopted | review | The guide's own note binds here: application code almost never takes `impl Into<T>`. |
| [`enum` over `bool` in APIs](https://google.github.io/styleguide/rustguide.html#enum-vs-bool) | adopted | review | |
| [Reporting errors](https://google.github.io/styleguide/rustguide.html#reporting-errors) | **changed** (two additions) | review | Added: a model directory, its weights, a tokenizer, an image and a text are outside input, never a precondition, so a wrong shape, a missing tensor or a bad pixel buffer fails with `Result`, never a panic. Why: a panic in a library is a crash in the caller's process, and a model file is the caller's to supply. Added: every library boundary returns the typed `rt_core::Error`, which a caller can match on; `anyhow` is for the application binaries only ([`docs/library.md`](docs/library.md), "Errors"). We do not take the guide's option of avoiding panics entirely: a panic still marks a bug in our own code. |
| [Newtypes vs type aliases](https://google.github.io/styleguide/rustguide.html#newtypes-vs-type-aliases) | adopted | review | |
| [Named fields over tuples](https://google.github.io/styleguide/rustguide.html#prefer-named-fields) | adopted | review | |
| [Add trait impls when needed](https://google.github.io/styleguide/rustguide.html#traits-yagni) | adopted | review | |
| [Implement standard traits](https://google.github.io/styleguide/rustguide.html#known-traits) | adopted | clippy `should_implement_trait` | |
| [`#[derive]` when possible](https://google.github.io/styleguide/rustguide.html#derive) | adopted | clippy `derivable_impls` | |
| [Generics vs `dyn Trait`](https://google.github.io/styleguide/rustguide.html#generics-vs-dyn-trait) | adopted | review | The model graphs are generic over `rt_backend::Hardware`, so the backend's calls in a hot loop are static. |
| [Trait vs inherent impl](https://google.github.io/styleguide/rustguide.html#trait-vs-inherent-impl) | adopted | review | |
| [Extension traits](https://google.github.io/styleguide/rustguide.html#extension-traits) | adopted | review | |
| [Guidance for important traits](https://google.github.io/styleguide/rustguide.html#important-traits) | adopted | clippy `new_without_default`, review | |
| [Pair `From` with a constructor](https://google.github.io/styleguide/rustguide.html#from-impls) | adopted | review | |
| [Options structs implement `Default`](https://google.github.io/styleguide/rustguide.html#options-should-implement-default) | **changed** | review | A public configuration is an options struct with `Default` (`RuntimeOptions`, each model's `Config`). Changed: clippy's `too_many_arguments` is allowed workspace-wide (root `Cargo.toml`), because a kernel's arguments are its shapes and strides (`m`, `n`, `k`, row strides, a thread budget), each one required, and a struct that bundles them only moves the list one line up. Outside kernels, a function with that many inputs takes an options struct. |
| [Handling poison](https://google.github.io/styleguide/rustguide.html#poison) | **changed** | review | A lock that only serializes, or that guards a step that cannot stop halfway (a cache insert), may ignore poison with `PoisonError::into_inner`. A lock over data with invariants either restores them on poison or panics. Why: the guide's concern is reading state a panic left broken, and a lock that panics forever on a healthy value turns one failed call into a dead model. |
| [No `async`/`.await`](https://google.github.io/styleguide/rustguide.html#async-await-chrome) [chrome] | **excluded** | | Chromium-only. The library has no `async`: inference is CPU-bound and blocks, and an async caller runs it on `spawn_blocking`. `bench/serve-sim` is an async host and may use it. |
| [No `std::thread`](https://google.github.io/styleguide/rustguide.html#std-thread) [chrome] | **excluded** | | Chromium-only: it has its own sequence scheduler. Our threads are the runtime's own pinned pool (`rt-threads`), which oneDNN runs on. |
| [`fuchsia_sync::Mutex`](https://google.github.io/styleguide/rustguide.html#fuchsia-mutex) [fuchsia] | **excluded** | | Fuchsia-only. Which mutex to use follows the poison row above. |
| [Macros](https://google.github.io/styleguide/rustguide.html#macros) | adopted | review | |
| [Naming methods](https://google.github.io/styleguide/rustguide.html#naming-methods) | adopted | clippy `wrong_self_convention` | |
| [Naming lifetimes](https://google.github.io/styleguide/rustguide.html#lifetimes) | adopted | review | |
| [Naming generic parameters](https://google.github.io/styleguide/rustguide.html#generic-parameters) | adopted | review | |
| [Naming extension traits](https://google.github.io/styleguide/rustguide.html#naming-extension-traits) | adopted | review | |
| [Naming initialization functions](https://google.github.io/styleguide/rustguide.html#naming-initialization-functions) | adopted | review | A constructor that returns `Result` is `try_new`. Older ones are renamed when their file is next touched. |
| [Avoid abbreviations](https://google.github.io/styleguide/rustguide.html#avoid-abbreviations) | **changed** (one addition) | review | Numerics vocabulary counts as known: the ISA names oneDNN uses (`avx2`, `avx512_core_bf16`, `amx`), `fp32`/`bf16`, `gemm`, `qkv`, `rope`, `geglu`, `ln`/`rms`, `md` (a oneDNN memory descriptor), `isa`, `ort`, and the model short names `eg2` and `ppocr`. |
| [When and how to write documentation](https://google.github.io/styleguide/rustguide.html#writing-doc) | adopted | review; `missing_docs` once the tree meets it | |
| [Unsafe code](https://google.github.io/styleguide/rustguide.html#unsafe) | adopted | review | Most `unsafe` here is the oneDNN FFI and SIMD intrinsics. Each block says what it relies on. ⚠ clippy `undocumented_unsafe_blocks` is not on yet: the tree does not meet it. |
| [No unstable features](https://google.github.io/styleguide/rustguide.html#unstable-features) | adopted | the stable compiler | Nightly rustfmt is a formatter, not a language feature. |
| [Getters and setters](https://google.github.io/styleguide/rustguide.html#getters-and-setters) | adopted | review | |
| [Accessors that accept closures](https://google.github.io/styleguide/rustguide.html#closure-accessors) | adopted | review | |
| [Implement expected API sets together](https://google.github.io/styleguide/rustguide.html#expected-api-sets) | adopted | clippy `len_without_is_empty` | |
| [Don't use others' `#[doc(hidden)]`](https://google.github.io/styleguide/rustguide.html#doc-hidden) | adopted | review | |
| [`const fn` responsibly](https://google.github.io/styleguide/rustguide.html#const-functions) | adopted | review | |
| [Operator overloads](https://google.github.io/styleguide/rustguide.html#operator-overloads) | adopted | review | |
| [`#[non_exhaustive]` on public enums](https://google.github.io/styleguide/rustguide.html#non-exhaustive-enums) | adopted | review | Library code. `rt_core::Error`, `CpuIsa` and `Precision` are marked. |
| [`#[non_exhaustive]` on public structs](https://google.github.io/styleguide/rustguide.html#non-exhaustive-structs) | adopted | review | Library code. `RuntimeOptions` and the model configurations are marked, which is why they are built with `new` and builders. |
| [`#[must_use]` on functions](https://google.github.io/styleguide/rustguide.html#must-use-on-functions) | adopted | review | Library code, in full. |
| [`#[must_use]` on types](https://google.github.io/styleguide/rustguide.html#must-use-on-types) | adopted | review | |
| [No code before `main`](https://google.github.io/styleguide/rustguide.html#global-initialization) | adopted | review | Process-global state (the oneDNN ISA cap, the thread pool, the AMX probe's result) is set on the first `Runtime::new`, not before `main`. |
| [Don't overuse type inference](https://google.github.io/styleguide/rustguide.html#type-inference) | adopted | review | |
| [Pattern match over check-then-unwrap](https://google.github.io/styleguide/rustguide.html#dont-check-then-unwrap) | adopted | clippy `unnecessary_unwrap` | |
| [`expect` over `unwrap`](https://google.github.io/styleguide/rustguide.html#expect-over-unwrap) | adopted | review (clippy `unwrap_used` not yet on) | The rule is about panics that stand for an invariant of our code. A test's `unwrap` on its own setup is not one: the test name already says what failed, so tests may `unwrap`. |
| [Don't omit patterns on `non_exhaustive` enums](https://google.github.io/styleguide/rustguide.html#non-exhaustive-enum-matching) | adopted | review | |
| [Context strings on assertions](https://google.github.io/styleguide/rustguide.html#assertion-context) | adopted | review | |
| [Combinator chains vs direct iteration](https://google.github.io/styleguide/rustguide.html#combinator-chains) | adopted | clippy `needless_for_each`, review | |
| [Don't call `catch_unwind`](https://google.github.io/styleguide/rustguide.html#catch-unwind) | **changed** | review | Allowed in tests that assert a panic, and at one boundary: a worker process (`sinteric_inference::worker`) catches a panic in a call and answers it as an error, because a call that never answers leaves its caller waiting forever. Each says why at the call. Why: the guide's ban rests on `panic = abort` in Chrome, Android and Fuchsia, and we cannot choose that for the programs that link this library. |
| [`format!`/`write!` for concatenation](https://google.github.io/styleguide/rustguide.html#prefer-format-or-write) | adopted | review | |
| [Per-codebase test framework](https://google.github.io/styleguide/rustguide.html#test-framework) [chrome, fuchsia] | **excluded** | | Those frameworks plug Rust tests into C++ and Fuchsia harnesses. We use `#[test]` under `cargo test`; tests that need weights skip when the files are absent. |

### The workspace's clippy allows

The root `Cargo.toml` allows a set of clippy lints workspace-wide
(`[workspace.lints.clippy]`). `too_many_arguments` has its reason in the
options-struct row above. The others (`needless_range_loop`,
`excessive_precision`, `manual_range_contains` and the rest of that list) were
allowed when the qualified numerics code was ported as-is, so that the port
changed no arithmetic: an index loop over several buffers and a constant
written to more digits than `f32` holds are how a kernel matches its reference
bit for bit. They are allowances, not rules: a kernel rewritten later meets
clippy's default, and a lint leaves the list once the tree meets it.

## What the guide leaves open

The guide says how a line of Rust should read. These say what the guide does
not, and review holds all of them.

1. **One thing has one home.** A piece of logic is written once and called from
   wherever it is needed. Why: a fix reaches only the copy its author knew
   about. The exception is a reference a test compares against: the scalar
   kernel beside its SIMD twin is the test's oracle, not a duplicate.
2. **A unit fits in one read.** A function does one job in at most 100 lines; a
   file holds one subject in at most 1,000 lines, not counting its tests. Why:
   a reader has lost the top of a long function before reaching its end.
   ⚠ Some files are past this today (the oneDNN backend's `cpu.rs` and
   `conv.rs`, and the model graphs); they are split when next restructured.
3. **Nothing is left over.** Code no caller reaches is deleted. Code that
   exists only for a test lives in the test. No code is commented out. Why:
   leftover code is still read, maintained and believed.
4. **A comment says why, never when.** It gives the constraint, the reason, or
   what the code cannot see, and tells no story of how the code got here; a
   dated measurement stays as the reason a choice was made. Why: history told
   in the present tense is believed as the present.
5. **Types carry the rules.** A choice is an `enum`, and so is a knob with a
   fixed set of values (`ImageTokens`, not one method per value); an error is
   a typed value the caller can match on; and a number changes type through
   `From` or `TryFrom` where a wrong cast would lose data. Why: what a type says, the
   compiler checks at every call.
6. **Numbers are measured, dated and placed.** A claim about speed, memory or
   accuracy names the CPU, the ISA, the thread count, the date, and what it was
   compared against, and says what it cannot see (a capped ISA is not that
   CPU). Why: a runtime tuned for specific hardware is judged by its numbers,
   and an undated number reads as current forever.
7. **Tests read as the specification.** A test's name states the behaviour. A
   change that only moves or renames code edits no assertion. Why: the tests
   are how a refactor proves it changed nothing.

## What the tools cannot see

Most of the guide is judgement, and only review holds it: module-qualified
calls, `enum` over `bool`, options structs, `expect` with a reason, explicit
types on `.into()` and `.parse()`, combinator chains, `Rc`/`Arc` against
borrowing, assertion messages, poison handling. An `#[allow]` silences any
lint, and nothing yet requires it to give a reason. A lint is added to
`[workspace.lints]` only once the tree meets it, because a lint with open sites
is a red build. Clippy sees only the packages CI names and the workspace crates
they depend on: the research crates (`crates/candle-mlas`, `bench/vela-mlas`)
and the benches are not linted.
