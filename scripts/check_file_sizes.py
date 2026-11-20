#!/usr/bin/env python3
"""Fail when a Rust file holds more than 1,000 lines outside its tests.

STYLE.md, "What the guide leaves open", rule 2: a file holds one subject in at
most 1,000 lines, not counting its tests. clippy's `too_many_lines` holds the
function half of that rule; nothing in clippy measures a file, so this does.

What it counts: every line of a `.rs` file under `crates/`, `bench/` and
`tools/` up to an inline `#[cfg(test)] mod tests {`. A file under a `tests/`
directory, or named `tests.rs`, is a test and is skipped, as are oneDNN's
vendored headers and the bindgen output.

What it cannot see: whether a short file holds one subject, and test code
that is not in a `mod tests` block (a `#[cfg(test)]` helper beside the code
counts toward the limit).
"""

import pathlib
import sys

LIMIT = 1000
ROOTS = ("crates", "bench", "tools")
SKIP_PARTS = {"tests", "vendor", "target"}
SKIP_NAMES = {"tests.rs", "bindings.rs"}


def code_lines(path: pathlib.Path) -> int:
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines[:-1]):
        if line.strip() == "#[cfg(test)]" and lines[i + 1].strip() == "mod tests {":
            return i
    return len(lines)


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    over = []
    for top in ROOTS:
        for path in sorted((root / top).rglob("*.rs")):
            rel = path.relative_to(root)
            if SKIP_PARTS & set(rel.parts[:-1]) or path.name in SKIP_NAMES:
                continue
            n = code_lines(path)
            if n > LIMIT:
                over.append(f"{rel}: {n} lines outside tests (limit {LIMIT})")
    for line in over:
        print(line)
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
