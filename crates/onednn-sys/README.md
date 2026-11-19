# onednn-sys

Bindgen FFI and **static link** of **oneDNN 3.13.4** (`c531fe1a`) C API (`dnnl_*`).

`src/bindings.rs` is the committed bindgen output of `wrapper.h` against
`vendor/oneapi/dnnl/*.h`. Regenerate with:

```bash
ONEDNN_BINDGEN=1 cargo build -p onednn-sys --features bindgen
```

## Build

`build.rs` statically links a CPU-only **THREADPOOL** oneDNN (`DNNL_LIBRARY_TYPE=STATIC`,
`ONEDNN_CPU_RUNTIME=THREADPOOL`) matching this crate's pinned release
(**3.13.4** / `c531fe1a`). Requirements: **cmake** and a C/C++ compiler
(`cc`/`c++` or `CC`/`CXX`). Ninja is used when present. There is no OpenMP /
libgomp dependency.

`snt_threadpool.cpp` implements `dnnl::threadpool_interop::threadpool_iface`
and forwards to `sinteric_tp_*` in `rt-threads`. Streams are created with
`dnnl_threadpool_interop_stream_create`.

Cache / CI:

- `ONEDNN_STATIC_ROOT` — install prefix that already contains a THREADPOOL
  `libdnnl.a` (and headers) for 3.13.4. An existing archive with the wrong
  version/runtime or missing primitives fails with a diagnostic; it is never
  silently replaced or bypassed with an OUT_DIR build.
- Otherwise the pinned commit's source is fetched into
  `$HOME/.cache/sinteric-sdk/.sdk-src/onednn-<commit>` and built into
  `$HOME/.cache/sinteric-sdk/onednn-static-tp`. The fetch takes GitHub's
  archive of the commit (`curl`), and when that is refused, a shallow
  `git fetch` of the same commit (`git`), checked to be that commit. A source
  tree already there (it has `CMakeLists.txt`) is used as it is.

There is no runtime `dlopen` of `libdnnl.so`. `rt-onednn` calls `dnnl_*`
directly. The library does not set process environment variables.

The enabled set is `MATMUL;REORDER;LAYER_NORMALIZATION;SOFTMAX;ELTWISE;BINARY;
REDUCTION;CONVOLUTION;DECONVOLUTION;POOLING;RESAMPLING;CONCAT;PRELU` (one line).
Concat is required by the detector FPN/stem; the existing PReLU wrapper remains
for phase B. An absent archive is built from pinned source automatically. A
narrow existing archive reports every missing `BUILD_*` define. Refresh it only
with this explicit build-time opt-in (the command modifies the named prefix):

```bash
CC=gcc CXX=g++ ONEDNN_STATIC_ROOT="$HOME/.cache/sinteric-sdk/onednn-static-tp" \
  ONEDNN_REBUILD_NARROW_PREFIX=1 cargo build -j6 -p onednn-sys --release
```

Unset `ONEDNN_REBUILD_NARROW_PREFIX` for normal builds. The opt-in applies only
to a matching 3.13.4 THREADPOOL prefix; version/runtime mismatches still fail.
No SDK directory needs to be deleted and `cargo clean` is unnecessary.
