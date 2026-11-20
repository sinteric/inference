#!/usr/bin/env python3
"""EG2 thread scaling + in-process concurrency (exec_slots / model instances).

Measures 1/2/4/8-if-present thread efficiency on a broad seq set, then
concurrent waves: two callers on one model (exec_slots=2) and two model
instances in one process. Sequential cells stay concurrency=1 so they
compare with the matrix gate.

  python3 bench/scaling.py --model $EG2_MODEL --output results/eg2-scale
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))

from scaling import (  # noqa: E402
    GATE_ABS_MS,
    GATE_REL,
    cell_key,
    concurrent_cases,
    concurrent_rows,
    format_concurrent,
    format_efficiency,
    efficiency_rows,
    n_physical_cores,
    summarize_times,
    thread_budgets,
    uniform_scale_cases,
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
    p.add_argument("--model", default=default_model())
    p.add_argument("--binary", default="target/release/eg2-onednn")
    p.add_argument("--output", required=True)
    p.add_argument("--threads", default="auto", help="comma-separated, or auto (1,2,4,8-if-present)")
    p.add_argument("--pin-isa", default=None)
    p.add_argument("--tile", type=int, default=None)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-concurrent", action="store_true")
    return p.parse_args()


def worker_cmd(args, threads, exec_slots=1, instances=1):
    cmd = [
        args.binary,
        "--model",
        args.model,
        "--threads",
        str(threads),
        "--exec-slots",
        str(exec_slots),
        "--instances",
        str(instances),
    ]
    if args.tile is not None:
        cmd += ["--tile", str(args.tile)]
    if args.pin_isa:
        cmd += ["--pin-isa", args.pin_isa]
    return cmd


def run_case(worker, case, cpus, threads, exec_slots=1, instances=1, concurrency=1):
    first_req = dict(case["request"], warmup=0, repeat=1)
    first = worker.call(first_req)
    if isinstance(first, dict) and "error" in first:
        return {"id": cell_key(threads, case["id"], exec_slots, instances, concurrency), "error": first["error"]}
    first_ms = (first.get("wall_s") or [float("nan")])[0] * 1000.0
    warm_req = dict(case["request"], warmup=case["warmup"], repeat=case["repeat"])
    resp = worker.call(warm_req)
    if isinstance(resp, dict) and "error" in resp:
        return {"id": cell_key(threads, case["id"], exec_slots, instances, concurrency), "error": resp["error"]}
    walls, cpus_t = resp["wall_s"], resp["cpu_s"]
    affinities = worker_task_affinities(worker.child.pid)
    check_affinity_budget(cpus, affinities)
    mem = memory(worker.child.pid)
    stats = summarize_times(walls, cpus_t)
    return {
        "id": cell_key(threads, case["id"], exec_slots, instances, concurrency),
        "case_id": case["id"],
        "kind": case["kind"],
        "threads": threads,
        "exec_slots": exec_slots,
        "instances": instances,
        "concurrency": concurrency,
        "batch": case["batch"],
        "seq": case["seq"],
        "live_tokens": case["live_tokens"],
        "warmup": case["warmup"],
        "repeat": case["repeat"],
        "first_call_ms": first_ms,
        "rss_bytes": mem.get("VmRSS"),
        "hwm_bytes": mem.get("VmHWM"),
        **stats,
    }


def run_group(args, out, threads, cases, exec_slots, instances, cells, errors):
    if not cases:
        return errors
    cpus = physical_cpus(threads)
    os.sched_setaffinity(0, cpus)
    os.environ.update(TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1")
    cmd = worker_cmd(args, threads, exec_slots=exec_slots, instances=instances)
    label = f"threads={threads} slots={exec_slots} inst={instances} cases={len(cases)}"
    print(f"== {label} ==", flush=True)
    log = (out / f"t{threads}-s{exec_slots}-i{instances}.stderr.log").open("w")
    try:
        with Worker(cmd, args.timeout, stderr=log) as worker:
            ready = worker.ready
            mem = memory(worker.child.pid)
            print(
                f"  load_wall_s={ready.get('load_wall_s')} "
                f"rss_mb={(mem.get('VmRSS') or 0)/1024/1024:.1f} "
                f"isa={ready.get('isa')} exec_slots={ready.get('exec_slots')} "
                f"instances={ready.get('instances')}",
                flush=True,
            )
            for case in cases:
                conc = case.get("concurrency") or case["request"].get("concurrency") or 1
                try:
                    row = run_case(
                        worker,
                        case,
                        cpus,
                        threads,
                        exec_slots=exec_slots,
                        instances=instances,
                        concurrency=conc,
                    )
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "id": cell_key(threads, case["id"], exec_slots, instances, conc),
                        "error": str(exc),
                    }
                cells.append(row)
                if "error" in row:
                    errors += 1
                    print(f"  {row['id']}: ERROR {row['error']}", flush=True)
                else:
                    print(
                        f"  {row['id']}: p50={row['p50_ms']:.2f}ms busy={row['busy_cores']:.2f} "
                        f"first={row['first_call_ms']:.2f}ms",
                        flush=True,
                    )
    finally:
        log.close()
    return errors


def main():
    args = parse_args()
    binary = Path(args.binary)
    if not binary.is_file():
        raise SystemExit(f"missing binary {binary}")
    if not Path(args.model).is_dir():
        raise SystemExit(f"missing model dir {args.model}")
    requested = None if args.threads == "auto" else [int(v) for v in args.threads.split(",")]
    budgets, skipped, nphys = thread_budgets(requested)
    scale_cases = uniform_scale_cases(quick=args.quick)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cells = []
    errors = 0
    for n in budgets:
        errors = run_group(args, out, n, scale_cases, 1, 1, cells, errors)

    if not args.skip_concurrent:
        groups = {}
        for case in concurrent_cases(nphys, quick=args.quick):
            key = (case["threads"], case["exec_slots"], case["instances"])
            groups.setdefault(key, []).append(case)
        for (th, slots, inst), group in groups.items():
            errors = run_group(args, out, th, group, slots, inst, cells, errors)

    eff = efficiency_rows(cells)
    conc = concurrent_rows(cells)
    report = {
        "protocol": {
            "threads": budgets,
            "threads_skipped": skipped,
            "n_physical_cores": nphys,
            "gate": {"rel": GATE_REL, "abs_ms": GATE_ABS_MS},
            "busy_cores": "sum(cpu_s)/sum(wall_s); concurrent waves use process CPU",
            "concurrent": "wave_p50 is C in-flight forwards; wave/seq ~ C means no overlap",
        },
        "cells": cells,
        "efficiency": eff,
        "concurrent": conc,
    }
    (out / "scaling.json").write_text(json.dumps(report, indent=2) + "\n")
    tables = ["# thread efficiency (vs 1-thread sequential)", format_efficiency(eff)]
    if conc:
        tables += ["", "# concurrent waves", format_concurrent(conc)]
    text = "\n".join(tables) + "\n"
    (out / "table.txt").write_text(text)
    print(text, flush=True)
    rc = 1 if errors else 0
    print(json.dumps({"complete": True, "cells": len(cells), "errors": errors, "output": str(out / "scaling.json")}))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
