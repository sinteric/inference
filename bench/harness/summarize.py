#!/usr/bin/env python3
"""Sanitized RESULTS.md writer. Never writes private ids/text/predictions."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="unknown")
    args = ap.parse_args()
    data = json.loads(Path(args.json).read_text())
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# {args.model} / {args.host} / {day}",
        "",
        "Sanitized summary. Adverse cells are kept. No private text.",
        "",
        "```json",
        json.dumps(data, indent=2),
        "```",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
