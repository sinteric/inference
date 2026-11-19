#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Reuse the generic recipe; include the Vela extension only in this build.
MLAS_WORK_DIR="$root" MLAS_TARGET_NATIVE=ON \
VELA_KERNEL_SOURCE="$root/native/mlas/vela_kernels.cpp" \
exec sh "$root/candle-mlas/scripts/build_mlas.sh"
