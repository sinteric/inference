#!/usr/bin/env bash
# Fetch and install oneDNN 3.13.4 (C API). The C++ bridge is no longer built:
# Rust calls dnnl_* directly via crates/onednn-sys + crates/rt-onednn.
set -euo pipefail

REVISION=c531fe1a5cd631b7f60277cdf6675c588cadbc28
ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${ONEDNN_ROOT:-${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}/onednn}"
SRC="${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}/.sdk-src/onednn-${REVISION}"
JOBS="${VELA_BUILD_JOBS:-${JOBS:-$(nproc)}}"
SRC_URL="https://github.com/uxlfoundation/oneDNN/archive/${REVISION}.tar.gz"

mkdir -p "$(dirname "$SRC")" "$PREFIX"

if [[ -n ${VELA_ONEDNN_PREFIX:-} ]]; then
  sdk_prefix=$VELA_ONEDNN_PREFIX
else
  if [[ -n ${VELA_ONEDNN_SOURCE:-} ]]; then
    source_dir=$VELA_ONEDNN_SOURCE
    [[ $(git -C "$source_dir" rev-parse HEAD) == "$REVISION" ]]
    [[ -z $(git -C "$source_dir" status --porcelain --untracked-files=no) ]]
  else
    if [[ ! -f "$SRC/CMakeLists.txt" ]]; then
      rm -rf "$SRC"
      mkdir -p "$SRC"
      echo "fetching oneDNN 3.13.4 @ ${REVISION}"
      curl -fsSL "$SRC_URL" | tar -xz --strip-components=1 -C "$SRC"
    fi
    source_dir=$SRC
  fi
  sdk_prefix="$PREFIX"
  cmake -S "$source_dir" -B "${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}/.sdk-src/onednn-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$sdk_prefix" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,noexecstack" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-z,noexecstack" \
    -DONEDNN_CPU_RUNTIME=OMP -DONEDNN_GPU_RUNTIME=NONE \
    -DONEDNN_BUILD_GRAPH=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_EXAMPLES=OFF \
    -DONEDNN_ENABLE_WORKLOAD=INFERENCE \
    '-DONEDNN_ENABLE_PRIMITIVE=MATMUL;REORDER;LAYER_NORMALIZATION;SOFTMAX;ELTWISE;BINARY;REDUCTION' \
    -DONEDNN_ENABLE_CONCURRENT_EXEC=ON -DONEDNN_ENABLE_PRIMITIVE_CACHE=OFF
  cmake --build "${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}/.sdk-src/onednn-build" --parallel "$JOBS"
  cmake --install "${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}/.sdk-src/onednn-build"
fi

version_header="$sdk_prefix/include/oneapi/dnnl/dnnl_version.h"
for spec in 'MAJOR 3' 'MINOR 13' 'PATCH 4'; do
  read -r name value <<< "$spec"
  grep -qE "^#define DNNL_VERSION_${name}[[:space:]]+${value}$" "$version_header"
done

echo "$REVISION" > "$sdk_prefix/PINNED_COMMIT"
lib=
for cand in "$sdk_prefix/lib/libdnnl.so" "$sdk_prefix/lib64/libdnnl.so"; do
  if [[ -f $cand ]]; then lib=$cand; break; fi
done
[[ -n $lib ]]
echo "oneDNN C API installed at $lib"
ldd "$lib"
# Keep a convenience symlink next to the old bridge path so existing scripts work.
ln -sfn "$lib" "$sdk_prefix/libdnnl.so"
echo "$lib"
