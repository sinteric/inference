"""EG2 broad-matrix cases, compare gate, and host topology. Standard library only."""
from __future__ import annotations

import math
import random
from pathlib import Path

from support import physical_cpus

SEQS = (7, 19, 64, 100, 128, 300, 512, 777, 1024, 2048, 4096, 8192)
BATCHES = (1, 4, 16)
ALPHABET = tuple(range(2, 1002))
GATE_REL = 0.02
GATE_ABS_MS = 0.3
ALIGNED_GRAIN = 64
SWEEP_DENSE_STOP = 600
SWEEP_DENSE_STEP = 7
SWEEP_LONG_COUNT = 50
SWEEP_LONG_START = 600
SWEEP_LONG_STOP = 8192
SWEEP_SEED = 1
SWEEP_LANDMARKS = (
    64,
    100,
    128,
    192,
    256,
    300,
    320,
    384,
    448,
    512,
    640,
    768,
    777,
    896,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
)
# Small aligned shapes to create before unaligned cells, without running 8k
# first (that reserved a huge workspace and dropped busy cores to ~1.5).
SWEEP_WARM_ALIGNED = (64, 128, 256, 512, 1024)
# Fresh worker every N sweep cases. One process for all 155 lengths still
# collapsed occupancy after ~s=300 (busy ~1.5/4) from primitive/workspace churn.
SWEEP_CASES_PER_PROCESS = 8

MIXED = (
    ("mix-b4-short", (7, 19, 64, 128)),
    ("mix-b4-mid", (100, 300, 512, 777)),
    ("mix-b4-long", (1024, 2048, 4096, 8192)),
    ("mix-b16-short", (7, 19, 64, 100, 128, 300) * 2 + (7, 19, 64, 100)),
    ("mix-b16-mid", (100, 300, 512, 777, 1024) * 3 + (100,)),
    ("mix-b16-long", (512, 1024, 2048, 4096, 8192) * 3 + (512,)),
)

QUICK_UNIFORM = {
    (1, s) for s in SEQS
} | {
    (4, 19),
    (4, 128),
    (4, 512),
    (4, 2048),
    (16, 64),
    (16, 128),
}
QUICK_MIXED = {"mix-b4-short", "mix-b16-short"}


def n_physical_cores():
    """Count physical cores (one logical id per thread_siblings group)."""
    base = Path("/sys/devices/system/cpu")
    groups = set()
    if base.is_dir():
        for cpu in base.glob("cpu[0-9]*"):
            if not cpu.name[3:].isdigit():
                continue
            sib = cpu / "topology" / "thread_siblings_list"
            groups.add(sib.read_text().strip() if sib.is_file() else cpu.name)
    return max(len(groups), 1)


def cache_sizes():
    """cpu0 cache sizes from sysfs, e.g. {'l1_data': '48K', 'l2_unified': '2048K'}."""
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    out = {}
    if not base.is_dir():
        return out
    indexes = []
    for idx in base.glob("index*"):
        tail = idx.name[5:]
        if tail.isdigit():
            indexes.append((int(tail), idx))
    for _, idx in sorted(indexes):
        try:
            level = (idx / "level").read_text().strip()
            typ = (idx / "type").read_text().strip().lower().replace(" ", "_")
            size = (idx / "size").read_text().strip()
        except OSError:
            continue
        out[f"l{level}_{typ}"] = size
    return out


def thread_budgets(requested=None, nphys=None):
    """1/2/4 and 8 when the machine has 8 physical cores. Never invent SMT twins."""
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
    """Pad token rows to a common length; attention_mask is 1 on live tokens."""
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


def warmup_repeat(live_tokens, quick=False, sweep=False):
    """Fewer repeats on huge cells so the matrix stays a gate, not a soak."""
    if sweep:
        # Short cells need enough samples that AMX remainder noise does not
        # trip the 2% smoothness gate (8 samples flaked s=57 vs s=64).
        if live_tokens <= 256:
            return 2, 15
        if live_tokens <= 1024:
            return 1, 8
        if live_tokens <= 4096:
            return 1, 4
        return 1, 2
    if quick:
        if live_tokens <= 512:
            return 2, 8
        if live_tokens <= 4096:
            return 1, 4
        return 1, 2
    if live_tokens <= 512:
        return 3, 15
    if live_tokens <= 4096:
        return 2, 8
    if live_tokens <= 16384:
        return 2, 5
    if live_tokens <= 65536:
        return 1, 3
    return 1, 2


def _case(cid, kind, batch, lengths, seq, padded_seq, ids, mask, live_tokens, quick, sweep=False):
    warmup, repeat = warmup_repeat(live_tokens, quick=quick, sweep=sweep)
    return {
        "id": cid,
        "kind": kind,
        "batch": batch,
        "seq": seq,
        "lengths": list(lengths),
        "padded_seq": padded_seq,
        "live_tokens": live_tokens,
        "warmup": warmup,
        "repeat": repeat,
        "request": {
            "operation": "tensor",
            "input_ids": ids,
            "attention_mask": mask,
            "return_embeddings": False,
        },
    }


def uniform_cases(quick=False):
    cases = []
    for batch in BATCHES:
        for seq in SEQS:
            if quick and (batch, seq) not in QUICK_UNIFORM:
                continue
            rows = [token_row(seq, seed=bi * 17 + seq) for bi in range(batch)]
            ids, mask, padded, live = pad_batch(rows)
            cases.append(
                _case(
                    f"b{batch}-s{seq}",
                    "uniform",
                    batch,
                    (seq,) * batch,
                    seq,
                    padded,
                    ids,
                    mask,
                    live,
                    quick,
                )
            )
    return cases


def mixed_cases(quick=False):
    cases = []
    for cid, lengths in MIXED:
        if quick and cid not in QUICK_MIXED:
            continue
        rows = [token_row(n, seed=i * 31 + n) for i, n in enumerate(lengths)]
        ids, mask, padded, live = pad_batch(rows)
        cases.append(
            _case(
                cid,
                "mixed",
                len(lengths),
                lengths,
                padded,
                padded,
                ids,
                mask,
                live,
                quick,
            )
        )
    return cases


def all_cases(quick=False):
    return uniform_cases(quick=quick) + mixed_cases(quick=quick)


def next_aligned(seq, grain=ALIGNED_GRAIN):
    """Smallest multiple of `grain` that is >= seq."""
    if seq < 1:
        return grain
    if seq % grain == 0:
        return seq
    return (seq // grain + 1) * grain


def sweep_lengths(quick=False, seed=SWEEP_SEED):
    """Dense 1..600 step 7, plus ~50 random longs and aligned landmarks."""
    if quick:
        return tuple(list(range(1, 64, SWEEP_DENSE_STEP)) + [64, 100, 128])
    dense = list(range(1, SWEEP_DENSE_STOP + 1, SWEEP_DENSE_STEP))
    rng = random.Random(seed)
    population = list(range(SWEEP_LONG_START, SWEEP_LONG_STOP + 1))
    longs = rng.sample(population, min(SWEEP_LONG_COUNT, len(population)))
    return tuple(sorted(set(dense) | set(longs) | set(SWEEP_LANDMARKS)))


def sweep_cases(quick=False, seed=SWEEP_SEED, batch=1):
    """Length sweep used as the unaligned-smoothness gate. Default batch is 1."""
    if batch < 1:
        raise ValueError("sweep batch must be >= 1")
    cases = []
    for seq in sweep_lengths(quick=quick, seed=seed):
        rows = [token_row(seq, seed=seq + bi * 17) for bi in range(batch)]
        ids, mask, padded, live = pad_batch(rows)
        cases.append(
            _case(
                f"sweep-s{seq}" if batch == 1 else f"sweep-b{batch}-s{seq}",
                "sweep",
                batch,
                (seq,) * batch,
                seq,
                padded,
                ids,
                mask,
                live,
                quick,
                sweep=True,
            )
        )
    # Warm 64/128/256/512/1024 first so unaligned cells reuse those GEMM
    # primitives (100 pads to 128). Then measure remaining lengths in
    # increasing order so RSS grows monotonically — putting 8192 first
    # left later cells at ~1.5 busy cores and made s=100 5× slower.
    cases.sort(
        key=lambda c: (0 if c["seq"] in SWEEP_WARM_ALIGNED else 1, c["seq"])
    )
    return cases


def smoothness_violations(cells, grain=ALIGNED_GRAIN, rel=GATE_REL, abs_ms=GATE_ABS_MS):
    """Flag lengths slower than the next larger aligned length (same threads)."""
    groups = {}
    for c in cells:
        if "error" in c or c.get("p50_ms") is None:
            continue
        key = (c.get("threads"), c.get("batch") or 1)
        groups.setdefault(key, []).append(c)
    violations = []
    for key, rows in sorted(groups.items()):
        by_seq = {}
        for c in rows:
            seq = c.get("seq")
            if seq is None:
                continue
            prev = by_seq.get(seq)
            if prev is None or c["p50_ms"] < prev["p50_ms"]:
                by_seq[seq] = c
        seqs = sorted(by_seq)
        for seq in seqs:
            target = next_aligned(seq, grain)
            if target == seq:
                continue
            nxt = next((x for x in seqs if x >= target), None)
            if nxt is None:
                continue
            cur, aligned = by_seq[seq], by_seq[nxt]
            bp, cp = float(aligned["p50_ms"]), float(cur["p50_ms"])
            delta_pct = ((cp - bp) / bp * 100.0) if bp > 0 else 0.0
            abs_delta = cp - bp
            if bp > 0 and (cp - bp) / bp > rel and abs_delta > abs_ms:
                violations.append(
                    {
                        "id": cur.get("id"),
                        "seq": seq,
                        "aligned_seq": nxt,
                        "p50_ms": cp,
                        "aligned_p50_ms": bp,
                        "delta_pct": delta_pct,
                        "delta_ms": abs_delta,
                        "busy_cores": cur.get("busy_cores"),
                        "threads": key[0],
                    }
                )
    return {
        "rel": rel,
        "abs_ms": abs_ms,
        "grain": grain,
        "violations": violations,
    }


def format_smoothness(result):
    lines = [
        f"smoothness: p50(s) > p50(next {result['grain']}-aligned) by "
        f">{result['rel']*100:.0f}% and >{result['abs_ms']} ms",
        f"violations={len(result['violations'])}",
    ]
    for row in result["violations"]:
        lines.append(
            f"  FAIL {row['id']}: s={row['seq']} {row['p50_ms']:.2f}ms > "
            f"s={row['aligned_seq']} {row['aligned_p50_ms']:.2f}ms "
            f"({row['delta_pct']:+.1f}%) busy={row.get('busy_cores')}"
        )
    return "\n".join(lines)


def sweep_svg(cells, path):
    """Write a latency-vs-length SVG (log-x). No extra Python deps."""
    pts = [
        (c["seq"], c["p50_ms"], c.get("busy_cores"))
        for c in cells
        if "error" not in c and c.get("seq") and c.get("p50_ms") is not None
    ]
    pts.sort()
    if not pts:
        Path(path).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="40">'
            '<text x="12" y="24">no sweep points</text></svg>\n'
        )
        return
    w, h, pad = 960, 420, 56
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = max(min(xs), 1), max(xs)
    ymin, ymax = 0.0, max(ys) * 1.05 or 1.0

    def xpix(seq):
        lo, hi = math.log(xmin), math.log(xmax)
        if hi <= lo:
            return pad
        return pad + (math.log(seq) - lo) / (hi - lo) * (w - 2 * pad)

    def ypix(ms):
        return h - pad - (ms - ymin) / (ymax - ymin) * (h - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-size="14">'
        "EG2 B1 p50 vs sequence length</text>",
        f'<text x="{w/2}" y="{h-12}" text-anchor="middle" font-size="12">'
        "seq (log)</text>",
        f'<text x="16" y="{h/2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 16 {h/2})">p50 ms</text>',
    ]
    ticks = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for t in ticks:
        if t < xmin or t > xmax:
            continue
        x = xpix(t)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{h-pad}" '
            'stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{h-pad+16}" text-anchor="middle" '
            f'font-size="10">{t}</text>'
        )
    d = " ".join(
        f"{'M' if i == 0 else 'L'}{xpix(s):.1f},{ypix(ms):.1f}"
        for i, (s, ms, _) in enumerate(pts)
    )
    parts.append(f'<path d="{d}" fill="none" stroke="#1d4ed8" stroke-width="1.5"/>')
    for s, ms, busy in pts:
        color = "#dc2626" if (busy or 0) < 3.0 else "#1d4ed8"
        parts.append(
            f'<circle cx="{xpix(s):.1f}" cy="{ypix(ms):.1f}" r="2.2" fill="{color}"/>'
        )
    parts.append("</svg>\n")
    Path(path).write_text("".join(parts))


def cell_key(threads, case_id):
    return f"t{threads}/{case_id}"


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


def compare_cells(baseline, current, rel=GATE_REL, abs_ms=GATE_ABS_MS):
    """Return regressions where current p50 is worse than baseline by >rel and >abs_ms."""
    base_map = {c["id"]: c for c in baseline}
    cur_map = {c["id"]: c for c in current}
    missing = sorted(set(base_map) - set(cur_map))
    extra = sorted(set(cur_map) - set(base_map))
    regressions = []
    deltas = []
    for cid in sorted(set(base_map) & set(cur_map)):
        b, c = base_map[cid], cur_map[cid]
        if "error" in b or "error" in c:
            if "error" in c and "error" not in b:
                regressions.append(
                    {
                        "id": cid,
                        "baseline_p50_ms": b.get("p50_ms"),
                        "current_p50_ms": None,
                        "delta_pct": None,
                        "reason": c["error"],
                    }
                )
            continue
        bp, cp = float(b["p50_ms"]), float(c["p50_ms"])
        delta_pct = ((cp - bp) / bp * 100.0) if bp > 0 else 0.0
        abs_delta = cp - bp
        row = {
            "id": cid,
            "baseline_p50_ms": bp,
            "current_p50_ms": cp,
            "delta_pct": delta_pct,
            "delta_ms": abs_delta,
        }
        deltas.append(row)
        if bp > 0 and (cp - bp) / bp > rel and abs_delta > abs_ms:
            regressions.append(row)
    return {
        "rel": rel,
        "abs_ms": abs_ms,
        "missing": missing,
        "extra": extra,
        "deltas": deltas,
        "regressions": regressions,
    }


def format_table(cells):
    lines = [
        f"{'id':<28} {'p50_ms':>10} {'p95_ms':>10} {'busy':>6} {'rss_mb':>8} {'first_ms':>10}"
    ]
    for c in cells:
        if "error" in c:
            lines.append(f"{c['id']:<28} ERROR {c['error']}")
            continue
        rss = c.get("rss_bytes") or 0
        lines.append(
            f"{c['id']:<28} {c['p50_ms']:10.2f} {c['p95_ms']:10.2f} "
            f"{c['busy_cores']:6.2f} {rss/1024/1024:8.1f} {c.get('first_call_ms', float('nan')):10.2f}"
        )
    return "\n".join(lines)


def format_compare(result):
    lines = [
        f"gate: p50 regression > {result['rel']*100:.0f}% and > {result['abs_ms']} ms",
        f"missing={len(result['missing'])} extra={len(result['extra'])} "
        f"regressions={len(result['regressions'])}",
    ]
    for row in result["regressions"]:
        if row.get("delta_pct") is None:
            lines.append(f"  FAIL {row['id']}: {row.get('reason')}")
        else:
            lines.append(
                f"  FAIL {row['id']}: {row['baseline_p50_ms']:.2f} -> "
                f"{row['current_p50_ms']:.2f} ms ({row['delta_pct']:+.1f}%)"
            )
    if result["missing"]:
        lines.append("missing cells: " + ", ".join(result["missing"]))
    return "\n".join(lines)
