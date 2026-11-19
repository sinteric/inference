#!/usr/bin/env bash
# Builds libcandle_mlas.so + libvela_mlas.so, the MLAS kernels of the optional
# MLAS comparison, with the original candle-mlas recipe
# (onnxruntime_mlas.cmake, OpenMP, -z noexecstack).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PREFIX="${MLAS_ROOT:-${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}/mlas}"
export MLAS_WORK_DIR="${MLAS_WORK_DIR:-$PREFIX}"
export VELA_KERNEL_SOURCE="$ROOT/native/mlas/vela_kernels.cpp"
# Hosted CI is portable. Turin/self-hosted can set MLAS_TARGET_NATIVE=ON.
export MLAS_TARGET_NATIVE="${MLAS_TARGET_NATIVE:-OFF}"
mkdir -p "$MLAS_WORK_DIR"
bash "$ROOT/crates/candle-mlas/scripts/build_mlas.sh"
mkdir -p "$PREFIX"
cp -a "$MLAS_WORK_DIR/vendor/mlas-build/libcandle_mlas.so" "$PREFIX/"
if [[ -f "$MLAS_WORK_DIR/vendor/mlas-build/libvela_mlas.so" ]]; then
  cp -a "$MLAS_WORK_DIR/vendor/mlas-build/libvela_mlas.so" "$PREFIX/"
fi
echo "f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7" > "$PREFIX/PINNED_COMMIT"
echo "MLAS bridges installed at $PREFIX"
ls -l "$PREFIX"/lib*.so
