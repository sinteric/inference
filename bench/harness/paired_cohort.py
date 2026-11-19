#!/usr/bin/env python3
"""Paired fresh-process cohorts (A/B/B/A), warm N / timed M.

Does not print private text. Emits a sanitized JSON summary.

  python3 bench/harness/paired_cohort.py \\
      --cmd-a './target/release/vela2-infer --backend onednn --model-dir $SINTERIC_ASSETS' \\
      --cmd-b 'python3 bench/baselines/ort/onnx_driver.py --model-dir $SINTERIC_ASSETS' \\
      --warm 5 --timed 15 --cohorts 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def run_once(cmd: str, env: dict) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, shell=True, check=True, env=env, stdout=subprocess.DEVNULL)
    return (time.perf_counter() - t0) * 1e3


def summarise(xs: list[float]) -> dict:
    s = sorted(xs)
    def pct(p):
        if not s:
            return None
        k = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
        return s[k]
    return {
        "n": len(s),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "mean": statistics.fmean(s) if s else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd-a", required=True)
    ap.add_argument("--cmd-b", required=True)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--timed", type=int, default=15)
    ap.add_argument("--cohorts", type=int, default=3)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()
    env = os.environ.copy()
    order = []
    for i in range(args.cohorts):
        order.extend(["A", "B", "B", "A"] if i % 2 == 0 else ["B", "A", "A", "B"])
    samples = {"A": [], "B": []}
    for who in order:
        cmd = args.cmd_a if who == "A" else args.cmd_b
        for _ in range(args.warm):
            run_once(cmd, env)
        for _ in range(args.timed):
            samples[who].append(run_once(cmd, env))
    out = {
        "protocol": "warm{}/timed{}/cohorts{}".format(args.warm, args.timed, args.cohorts),
        "A": summarise(samples["A"]),
        "B": summarise(samples["B"]),
        "host": {
            "note": "record lscpu / governor / OMP in the RESULTS.md; this JSON is sanitized",
        },
    }
    text = json.dumps(out, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
