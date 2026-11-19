#!/usr/bin/env bash
# Historical C++ bridge rebuild. Production no longer uses it: Rust binds the
# oneDNN C API directly. Kept so the retired ABI can be rebuilt for diffs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${ONEDNN_ROOT:-${VELA_ONEDNN_PREFIX:-${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}/onednn}}"
if [[ ! -f "$PREFIX/include/oneapi/dnnl/dnnl.hpp" ]]; then
  echo "oneDNN SDK missing at $PREFIX; run native/onednn/build_sdk.sh first" >&2
  exit 1
fi
cmake -S "$ROOT" -B "$ROOT/build" -DONEDNN_ROOT="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
  -DVELA_MARCH_NATIVE="${VELA_MARCH_NATIVE:-OFF}"
cmake --build "$ROOT/build" -j"${VELA_BUILD_JOBS:-$(nproc)}"
cmake --install "$ROOT/build" --prefix "$PREFIX"
echo "retired C++ bridge installed at $PREFIX/libvela_standalone_dnnl.so (not the prod path)"
