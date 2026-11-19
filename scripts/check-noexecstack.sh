#!/usr/bin/env bash
# GNU_STACK must be RW, not RWE. Carried from the original Vela experiment.
set -euo pipefail
fail=0
check() {
  local f="$1"
  if [[ ! -e "$f" ]]; then
    echo "missing $f"
    return
  fi
  local line
  line=$(readelf -lW "$f" | awk '/GNU_STACK/ {print}' | tail -n1)
  echo "$f: $line"
  if echo "$line" | grep -qE 'RWE|E '; then
    echo "FAIL: GNU_STACK is executable on $f (link with -Wl,-z,noexecstack)" >&2
    fail=1
  fi
  if ! echo "$line" | grep -q 'GNU_STACK'; then
    echo "FAIL: no GNU_STACK on $f" >&2
    fail=1
  fi
  if echo "$line" | grep -q 'GNU_STACK'; then
    python3 - "$f" "$line" <<'PY' || fail=1
import sys
line=sys.argv[2]
# Original CI asserted the penultimate token is RW.
parts=line.split()
assert parts[-2]=='RW', (sys.argv[1], line)
PY
  fi
}
PREFIX="${SINTERIC_SDK:-$HOME/.cache/sinteric-sdk}"
shopt -s nullglob
for f in \
  "$PREFIX"/onednn/lib/libdnnl.so \
  "$PREFIX"/onednn/lib64/libdnnl.so \
  "$PREFIX"/onednn/libdnnl.so \
  "$PREFIX"/onednn/libvela_standalone_dnnl.so \
  "$PREFIX"/mlas/libcandle_mlas.so \
  "$PREFIX"/mlas/libvela_mlas.so \
  "$@"; do
  check "$f"
done
if [[ $fail -ne 0 ]]; then
  exit 1
fi
echo "noexecstack ok"
