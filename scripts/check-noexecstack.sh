#!/usr/bin/env bash
# Every ELF named on the command line must have a GNU_STACK header that is
# RW, not RWE: an executable stack would come from an object built without
# `-z noexecstack`, such as hand-written assembly in a statically linked
# dependency. CI runs it on the drivers, which link all of oneDNN.
set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "usage: $0 ELF..." >&2
  exit 2
fi
fail=0
for f in "$@"; do
  if [[ ! -e "$f" ]]; then
    echo "FAIL: missing $f" >&2
    fail=1
    continue
  fi
  line=$(readelf -lW "$f" | awk '/GNU_STACK/ {print}' | tail -n1)
  echo "$f: $line"
  if [[ -z $line ]]; then
    echo "FAIL: no GNU_STACK on $f" >&2
    fail=1
  elif [[ $(awk '{print $(NF-1)}' <<<"$line") != RW ]]; then
    echo "FAIL: GNU_STACK is not RW on $f (link with -Wl,-z,noexecstack)" >&2
    fail=1
  fi
done
if [[ $fail -ne 0 ]]; then
  exit 1
fi
echo "noexecstack ok"
