"""EG2 thread-scaling and in-process concurrency cases. Standard library only."""
from __future__ import annotations

from pathlib import Path

# Broad enough to catch shape-specific thread splits; not the full matrix gate.
SCALE_SEQS = (7, 19, 64, 128, 512, 777, 2048, 8192)
SCALE_BATCHES = (1, 4)
CONCURRENT_SEQS = (7, 64, 128, 512, 2048)
ALPHABET = tuple(range(2, 1002))
GATE_REL = 0.02
GATE_ABS_MS = 0.3


def n_physical_cores():
    base = Path("/sys/devices/system/cpu")
    groups = set()
    if base.is_dir():
        for cpu in base.glob("cpu[0-9]*"):
            if not cpu.name[3:].isdigit():
                continue
            sib = cpu / "topology" / "thread_siblings_list"
            groups.add(sib.read_text().strip() if sib.is_file() else cpu.name)
    return max(len(groups), 1)


def thread_budgets(requested=None, nphys=None):
    """1/2/4 and 8 when the machine has that many physical cores."""
    nphys = n_physical_cores() if nphys is None else nphys
    if requested:
        want = [int(v) for v in requested]
        if any(n < 1 for n in want):
            raise ValueError("thread counts must be >= 1")
        skipped = [n for n in want if n > nphys]
        keep = [n for n in want if n <= nphys]
        if not keep:
            raise ValueError(f"no requested thread count fits {nphys} physical cores")
        return keep, skipped, nphys
    keep = [n for n in (1, 2, 4, 8) if n <= nphys]
    skipped = [n for n in (1, 2, 4, 8) if n > nphys]
    return keep, skipped, nphys


def token_row(length, seed):
    return [ALPHABET[(seed + i) % len(ALPHABET)] for i in range(length)]


def pad_batch(rows):
    if not rows:
        raise ValueError("empty batch")
    padded_seq = max(len(r) for r in rows)
    if padded_seq < 1:
        raise ValueError("empty sequence")
    ids, mask = [], []
    for row in rows:
        live = len(row)
        ids.append(list(row) + [0] * (padded_seq - live))
        mask.append([1] * live + [0] * (padded_seq - live))
    live_tokens = sum(sum(m) for m in mask)
    return ids, mask, padded_seq, live_tokens


def warmup_repeat(live_tokens, quick=False):
    if quick:
        if live_tokens <= 512:
            return 2, 6
        if live_tokens <= 4096:
            return 1, 3
        return 1, 2
    if live_tokens <= 512:
        return 3, 12
    if live_tokens <= 4096:
        return 2, 6
    if live_tokens <= 16384:
        return 2, 4
    return 1, 2


def _tensor_case(cid, kind, batch, seq, extra=None):
    rows = [token_row(seq, seed=bi * 17 + seq) for bi in range(batch)]
    ids, mask, padded, live = pad_batch(rows)
    warmup, repeat = warmup_repeat(live)
    req = {
        "operation": "tensor",
        "input_ids": ids,
        "attention_mask": mask,
        "return_embeddings": False,
    }
    if extra:
        req.update(extra)
    return {
        "id": cid,
        "kind": kind,
        "batch": batch,
        "seq": seq,
        "live_tokens": live,
        "padded_seq": padded,
        "warmup": warmup,
        "repeat": repeat,
        "request": req,
    }


def uniform_scale_cases(quick=False):
    seqs = SCALE_SEQS[:4] if quick else SCALE_SEQS
    batches = (1,) if quick else SCALE_BATCHES
    cases = []
    for batch in batches:
        for seq in seqs:
            if quick and batch == 4 and seq not in (128, 512):
                continue
            cases.append(_tensor_case(f"b{batch}-s{seq}", "scale", batch, seq))
    return cases


def concurrent_cases(nphys, quick=False):
    """In-process waves: exec_slots and separate model instances."""
    seqs = CONCURRENT_SEQS[:3] if quick else CONCURRENT_SEQS
    out = []
    for seq in seqs:
        out.append(
            _tensor_case(
                f"slots2-b1-s{seq}",
                "slots",
                1,
                seq,
                extra={"concurrency": 2},
            )
        )
        out[-1]["exec_slots"] = 2
        out[-1]["instances"] = 1
        out[-1]["concurrency"] = 2
        out[-1]["threads"] = nphys
    if not quick:
        out.append(
            _tensor_case(
                "inst2-b1-s128",
                "instances",
                1,
                128,
                extra={"concurrency": 2},
            )
        )
        out[-1]["exec_slots"] = 1
        out[-1]["instances"] = 2
        out[-1]["concurrency"] = 2
        out[-1]["threads"] = nphys
    return out


def cell_key(threads, case_id, exec_slots=1, instances=1, concurrency=1):
    suffix = ""
    if exec_slots != 1 or instances != 1 or concurrency != 1:
        suffix = f"/slots{exec_slots}/inst{instances}/c{concurrency}"
    return f"t{threads}/{case_id}{suffix}"


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize_times(wall, cpu):
    if not wall:
        raise ValueError("empty samples")
    busy = (sum(cpu) / sum(wall)) if sum(wall) > 0 else 0.0
    return {
        "n": len(wall),
        "p50_ms": percentile(wall, 0.5) * 1000.0,
        "p95_ms": percentile(wall, 0.95) * 1000.0,
        "busy_cores": busy,
    }


def efficiency_rows(cells):
    """Speedup vs 1-thread on the same case_id (sequential cells only)."""
    by = {}
    for c in cells:
        if c.get("kind") != "scale" or "error" in c or "p50_ms" not in c:
            continue
        by.setdefault(c["case_id"], {})[c["threads"]] = c
    rows = []
    for cid, by_t in sorted(by.items()):
        if 1 not in by_t:
            continue
        base = by_t[1]["p50_ms"]
        for n, cell in sorted(by_t.items()):
            speedup = (base / cell["p50_ms"]) if cell["p50_ms"] > 0 else float("nan")
            rows.append(
                {
                    "id": cell["id"],
                    "case_id": cid,
                    "threads": n,
                    "p50_ms": cell["p50_ms"],
                    "busy_cores": cell.get("busy_cores"),
                    "speedup_vs_t1": speedup,
                    "efficiency": speedup / n if n else float("nan"),
                }
            )
    return rows


def concurrent_rows(cells):
    """Wave p50 vs sequential p50 at the same thread count and shape."""
    seq = {
        (c["threads"], c["case_id"]): c
        for c in cells
        if c.get("kind") == "scale" and "error" not in c
    }
    rows = []
    for c in cells:
        if c.get("kind") not in ("slots", "instances") or "error" in c:
            continue
        conc = c.get("concurrency") or 1
        look = (c["threads"], f"b{c['batch']}-s{c['seq']}")
        base = seq.get(look)
        wave = c["p50_ms"]
        row = {
            "id": c["id"],
            "kind": c["kind"],
            "threads": c["threads"],
            "exec_slots": c.get("exec_slots"),
            "instances": c.get("instances"),
            "concurrency": conc,
            "wave_p50_ms": wave,
            "busy_cores": c.get("busy_cores"),
            "seq_p50_ms": None,
            "wave_over_seq": None,
            "qps": (conc / (wave / 1000.0)) if wave > 0 else float("nan"),
        }
        if base and base["p50_ms"] > 0:
            row["seq_p50_ms"] = base["p50_ms"]
            row["wave_over_seq"] = wave / base["p50_ms"]
            row["seq_qps"] = 1000.0 / base["p50_ms"]
        rows.append(row)
    return rows


def format_efficiency(rows):
    lines = [
        f"{'id':<28} {'p50_ms':>10} {'busy':>6} {'speedup':>8} {'eff':>6}"
    ]
    for r in rows:
        lines.append(
            f"{r['id']:<28} {r['p50_ms']:10.2f} {r['busy_cores']:6.2f} "
            f"{r['speedup_vs_t1']:8.2f} {r['efficiency']:6.2f}"
        )
    return "\n".join(lines)


def format_concurrent(rows):
    lines = [
        f"{'id':<42} {'wave':>10} {'seq':>10} {'wave/seq':>9} {'qps':>8} {'busy':>6}"
    ]
    for r in rows:
        seq = r.get("seq_p50_ms")
        ratio = r.get("wave_over_seq")
        seq_s = f"{seq:10.2f}" if seq is not None else f"{'n/a':>10}"
        ratio_s = f"{ratio:9.2f}" if ratio is not None else f"{'n/a':>9}"
        lines.append(
            f"{r['id']:<42} {r['wave_p50_ms']:10.2f} {seq_s} {ratio_s} "
            f"{r['qps']:8.2f} {r['busy_cores']:6.2f}"
        )
    return "\n".join(lines)
