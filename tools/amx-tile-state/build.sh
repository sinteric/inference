#!/bin/sh
# Build and run the raw AMX tile-state stress (no oneDNN). Needs gcc >= 11.
set -e
cd "$(dirname "$0")"
gcc -O2 -mamx-tile -mamx-bf16 -pthread amx_stress.c -o amx_stress
echo "== 1 thread alone (expect bad=0)"
./amx_stress 1 20000000
echo "== 1 thread + busy loop on the same CPU (bad>0 on an affected VM)"
taskset -c 0 sh -c 'while :; do :; done' &
P=$!
taskset -c 0 ./amx_stress 1 30000000 || true
kill $P
echo "== 8 threads on 8 CPUs, then 16 threads"
./amx_stress 8 20000000 || true
./amx_stress 16 20000000 || true
