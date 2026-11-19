#!/usr/bin/env python3
"""One-command EG2 broad matrix: many seqs, mixed batches, thread scaling, cold load.

Generates prepared-tensor requests (no giant JSONL). Reports p50/p95, busy cores,
RSS, first-call latency, and process cold load. Optional --compare against a
previous JSON with a 2% p50 gate (and a 0.3 ms floor so tiny cells do not flake).

  python3 bench/matrix.py --model $EG2_MODEL --output results/eg2-matrix
  python3 bench/matrix.py --compare results/eg2-matrix/matrix.json --output results/eg2-pr
  python3 bench/matrix.py --sweep --output results/eg2-sweep   # dense length smoothness gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))

from matrix import (  # noqa: E402
    GATE_ABS_MS,
    GATE_REL,
    SWEEP_CASES_PER_PROCESS,
    all_cases,
    cache_sizes,
    cell_key,
    compare_cells,
    format_compare,
    format_smoothness,
    format_table,
    n_physical_cores,
    smoothness_violations,
    summarize_times,
    sweep_cases,
    sweep_svg,
    thread_budgets,
)
from support import (  # noqa: E402
    Worker,
    check_affinity_budget,
    memory,
    physical_cpus,
    worker_task_affinities,
)


def default_model():
    env = os.environ.get("EG2_MODEL") or os.environ.get("SINTERIC_EG2_MODEL")
    if env:
        return env
    return str(Path.home() / ".cache/sinteric-assets/embeddinggemma2@914f7f89142e")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=default_model(), help="EG2 snapshot directory")
    p.add_argument(
        "--binary",
        default="target/release/eg2-onednn",
        help="eg2-onednn path (built with cargo -p embeddinggemma2 --bin eg2-onednn --release)",
    )
    p.add_argument("--output", required=True, help="directory for matrix.json / table.txt")
    p.add_argument(
        "--threads",
        default="auto",
        help="comma-separated thread counts, or auto (1,2,4,8-if-present)",
    )
    p.add_argument("--pin-isa", default=None, help="e.g. AVX512_CORE_BF16")
    p.add_argument(
        "--precision",
        default=None,
        choices=["bf16", "fp32"],
        help="EG2 matmul precision; omit for the driver default (bf16)",
    )
    p.add_argument("--tile", type=int, default=None, help="attention tile; omit to use driver default")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--quick", action="store_true", help="subset for iteration, not the merge gate")
    p.add_argument(
        "--sweep",
        action="store_true",
        help="dense length sweep (1..600 step 7, plus ~50 longs); smoothness gate",
    )
    p.add_argument(
        "--sweep-batch",
        type=int,
        default=1,
        help="batch size for --sweep (default 1)",
    )
    p.add_argument("--compare", default=None, help="previous matrix.json; fail if any cell >2% slower")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--limit", default=None, help="comma-separated case ids to run")
    return p.parse_args()


def worker_cmd(args, threads):
    cmd = [
        args.binary,
        "--model",
        args.model,
        "--threads",
        str(threads),
    ]
    if args.tile is not None:
        cmd += ["--tile", str(args.tile)]
    if args.pin_isa:
        cmd += ["--pin-isa", args.pin_isa]
    if args.precision:
        cmd += ["--precision", args.precision]
    return cmd


def run_case(worker, case, cpus, threads):
    first_req = dict(case["request"], warmup=0, repeat=1)
    first = worker.call(first_req)
    if isinstance(first, dict) and "error" in first:
        return {"id": cell_key(threads, case["id"]), "error": first["error"]}
    first_ms = (first.get("wall_s") or [float("nan")])[0] * 1000.0
    warm_req = dict(case["request"], warmup=case["warmup"], repeat=case["repeat"])
    resp = worker.call(warm_req)
    if isinstance(resp, dict) and "error" in resp:
        return {"id": cell_key(threads, case["id"]), "error": resp["error"]}
    walls, cpus_t = resp["wall_s"], resp["cpu_s"]
    if len(walls) != case["repeat"] or len(cpus_t) != case["repeat"]:
        return {
            "id": cell_key(threads, case["id"]),
            "error": f"sample count {len(walls)} != {case['repeat']}",
        }
    affinities = worker_task_affinities(worker.child.pid)
    check_affinity_budget(cpus, affinities)
    mem = memory(worker.child.pid)
    stats = summarize_times(walls, cpus_t)
    prim = resp.get("primitive_stats") or {}
    return {
        "id": cell_key(threads, case["id"]),
        "case_id": case["id"],
        "kind": case["kind"],
        "threads": threads,
        "batch": case["batch"],
        "seq": case["seq"],
        "lengths": case["lengths"],
        "padded_seq": case["padded_seq"],
        "live_tokens": case["live_tokens"],
        "warmup": case["warmup"],
        "repeat": case["repeat"],
        "first_call_ms": first_ms,
        "rss_bytes": mem.get("VmRSS"),
        "hwm_bytes": mem.get("VmHWM"),
        "isa": prim.get("cpu_isa"),
        **stats,
    }


def main():
    args = parse_args()
    binary = Path(args.binary)
    if not binary.is_file():
        raise SystemExit(f"missing binary {binary} (build eg2-onednn --release)")
    if not Path(args.model).is_dir():
        raise SystemExit(f"missing model dir {args.model}")
    requested = None if args.threads == "auto" else [int(v) for v in args.threads.split(",")]
    budgets, skipped, nphys = thread_budgets(requested)
    if args.sweep:
        cases = sweep_cases(quick=args.quick, batch=args.sweep_batch)
    else:
        cases = all_cases(quick=args.quick)
    if args.limit:
        keep = {x.strip() for x in args.limit.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in keep]
        if not cases:
            raise SystemExit(f"no cases match --limit {keep}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    protocol = {
        "quick": args.quick,
        "sweep": args.sweep,
        "sweep_batch": args.sweep_batch if args.sweep else None,
        "seqs": list({c["seq"] for c in cases if c["kind"] == "uniform"}),
        "batches": list({c["batch"] for c in cases if c["kind"] == "uniform"}),
        "threads": budgets,
        "threads_skipped": skipped,
        "n_physical_cores": nphys,
        "cache": cache_sizes(),
        "pin_isa": args.pin_isa,
        "precision": args.precision or "bf16",
        "tile": args.tile,
        "gate": {"rel": GATE_REL, "abs_ms": GATE_ABS_MS},
        "cold_definition": "fresh process; ready.load_wall_s is model+backend load; first_call_ms is the first forward of that shape",
        "busy_cores": "sum(cpu_s)/sum(wall_s) over warmed repeats",
        "mixed": "B4/B16 padded to max length in the batch; attention_mask 0 on pad",
        "sweep_cases_per_process": SWEEP_CASES_PER_PROCESS if args.sweep else None,
    }
    cells = []
    cold = []
    errors = 0
    for n in budgets:
        cpus = physical_cpus(n)
        os.sched_setaffinity(0, cpus)
        os.environ.update(
            TOKENIZERS_PARALLELISM="false",
            RAYON_NUM_THREADS="1",
        )
        cmd = worker_cmd(args, n)
        chunks = (
            [cases[i : i + SWEEP_CASES_PER_PROCESS] for i in range(0, len(cases), SWEEP_CASES_PER_PROCESS)]
            if args.sweep
            else [cases]
        )
        print(
            f"== threads={n} cpus={cpus} cases={len(cases)} chunks={len(chunks)} ==",
            flush=True,
        )
        log = (out / f"t{n}.stderr.log").open("w")
        try:
            for ci, chunk in enumerate(chunks):
                with Worker(cmd, args.timeout, stderr=log) as worker:
                    ready = worker.ready
                    mem = memory(worker.child.pid)
                    if ci == 0:
                        cold.append(
                            {
                                "threads": n,
                                "startup_wall_s": worker.startup_wall_s,
                                "load_wall_s": ready.get("load_wall_s"),
                                "rss_bytes": mem.get("VmRSS"),
                                "hwm_bytes": mem.get("VmHWM"),
                                "isa": ready.get("cpu_isa"),
                                "tile": ready.get("tile"),
                                "affinity": ready.get("affinity"),
                            }
                        )
                    print(
                        f"  chunk {ci + 1}/{len(chunks)} seq {chunk[0]['seq']}..{chunk[-1]['seq']} "
                        f"load_wall_s={ready.get('load_wall_s')} "
                        f"rss_mb={(mem.get('VmRSS') or 0)/1024/1024:.1f} "
                        f"isa={ready.get('cpu_isa')}",
                        flush=True,
                    )
                    for case in chunk:
                        try:
                            row = run_case(worker, case, cpus, n)
                        except Exception as exc:  # noqa: BLE001 — keep the rest of the matrix
                            row = {"id": cell_key(n, case["id"]), "error": str(exc)}
                        cells.append(row)
                        if "error" in row:
                            errors += 1
                            print(f"  {row['id']}: ERROR {row['error']}", flush=True)
                            if args.fail_fast:
                                raise SystemExit(1)
                        else:
                            print(
                                f"  {row['id']}: p50={row['p50_ms']:.2f}ms p95={row['p95_ms']:.2f}ms "
                                f"busy={row['busy_cores']:.2f} first={row['first_call_ms']:.2f}ms "
                                f"live={row['live_tokens']}",
                                flush=True,
                            )
        finally:
            log.close()

    report = {"protocol": protocol, "cold_load": cold, "cells": cells}
    (out / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    table = format_table(cells)
    (out / "table.txt").write_text(table + "\n")
    print(table, flush=True)

    rc = 0
    if errors:
        print(f"{errors} cells errored", flush=True)
        rc = 1
    if args.compare:
        prev = json.loads(Path(args.compare).read_text())
        result = compare_cells(prev.get("cells") or [], cells)
        (out / "compare.json").write_text(json.dumps(result, indent=2) + "\n")
        text = format_compare(result)
        print(text, flush=True)
        if result["regressions"] or result["missing"]:
            rc = 1
    if args.sweep:
        sweep_svg(cells, out / "sweep.svg")
        smooth = smoothness_violations(cells)
        (out / "smoothness.json").write_text(json.dumps(smooth, indent=2) + "\n")
        text = format_smoothness(smooth)
        print(text, flush=True)
        if smooth["violations"]:
            rc = 1
    print(json.dumps({"complete": True, "cells": len(cells), "errors": errors, "output": str(out / "matrix.json")}))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
