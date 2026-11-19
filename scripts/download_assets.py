#!/usr/bin/env python3
"""Fetch Hub files listed in assets/<model>.toml and verify sha256.

Usage:
  python3 scripts/download_assets.py assets/vela2-0p3b.toml
Env:
  SINTERIC_ASSETS  destination root (default ~/.cache/sinteric-assets/<model>@<rev12>)
  HF_TOKEN         optional Hub token
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--only", action="append", default=[], help="subset of paths")
    args = ap.parse_args()
    man = tomllib.loads(Path(args.manifest).read_text())
    rev = man["revision"]
    dest = os.environ.get("SINTERIC_ASSETS")
    if not dest:
        dest = str(
            Path.home()
            / ".cache/sinteric-assets"
            / f"{man['model']}@{rev[:12]}"
        )
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    want = set(args.only)
    for f in man["files"]:
        rel = f["path"]
        if want and rel not in want:
            continue
        sha = str(f.get("sha256", "skip"))
        if sha in ("skip", "none") or len(sha) != 64:
            print(f"note: {rel} has no 64-hex sha256 in the manifest; download only")
        url = f"https://huggingface.co/{man['repo']}/resolve/{rev}/{rel}"
        out = dest_p / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_file() and out.stat().st_size == f["size"]:
            if len(sha) != 64 or sha256_file(out) == sha:
                print(f"ok  {rel}")
                continue
        print(f"get {rel}")
        req = urllib.request.Request(url, headers={"User-Agent": "sinteric-inference"})
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as r, out.open("wb") as w:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                w.write(buf)
        if len(sha) == 64 and sha != "0" * 64:
            got = sha256_file(out)
            if got != sha:
                print(f"sha256 mismatch {rel}: {got} != {sha}", file=sys.stderr)
                return 2
        print(f"ok  {rel}")
    print(f"assets in {dest_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
