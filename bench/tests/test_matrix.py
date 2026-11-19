"""Asset-free tests for the EG2 broad matrix (no model, no binary)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from matrix import (  # noqa: E402
    BATCHES,
    GATE_ABS_MS,
    SEQS,
    SWEEP_CASES_PER_PROCESS,
    all_cases,
    compare_cells,
    mixed_cases,
    n_physical_cores,
    next_aligned,
    pad_batch,
    smoothness_violations,
    sweep_cases,
    sweep_lengths,
    thread_budgets,
    token_row,
    uniform_cases,
    warmup_repeat,
)


class MatrixCaseTests(unittest.TestCase):
    def test_seqs_cover_short_odd_and_long(self):
        self.assertEqual(SEQS[0], 7)
        self.assertIn(19, SEQS)
        self.assertIn(100, SEQS)
        self.assertIn(777, SEQS)
        self.assertEqual(SEQS[-1], 8192)
        for s in (7, 19, 100, 777):
            self.assertNotEqual(s % 16, 0, s)
            self.assertNotEqual(s % 64, 0, s)

    def test_uniform_cartesian_and_padding(self):
        cases = uniform_cases()
        ids = {c["id"] for c in cases}
        for b in BATCHES:
            for s in SEQS:
                self.assertIn(f"b{b}-s{s}", ids)
        b1 = next(c for c in cases if c["id"] == "b1-s19")
        self.assertEqual(b1["batch"], 1)
        self.assertEqual(b1["padded_seq"], 19)
        self.assertEqual(b1["live_tokens"], 19)
        self.assertEqual(len(b1["request"]["input_ids"][0]), 19)
        self.assertTrue(all(v == 1 for v in b1["request"]["attention_mask"][0]))
        b4 = next(c for c in cases if c["id"] == "b4-s777")
        self.assertEqual(b4["batch"], 4)
        self.assertEqual(b4["live_tokens"], 4 * 777)

    def test_mixed_batches_mask_padding(self):
        mixed = {c["id"]: c for c in mixed_cases()}
        short = mixed["mix-b4-short"]
        self.assertEqual(short["batch"], 4)
        self.assertEqual(short["lengths"], [7, 19, 64, 128])
        self.assertEqual(short["padded_seq"], 128)
        self.assertEqual(short["live_tokens"], 7 + 19 + 64 + 128)
        masks = short["request"]["attention_mask"]
        self.assertEqual(sum(masks[0]), 7)
        self.assertEqual(masks[0][7:], [0] * (128 - 7))
        self.assertEqual(short["request"]["input_ids"][0][7:], [0] * (128 - 7))
        long = mixed["mix-b16-long"]
        self.assertEqual(long["batch"], 16)
        self.assertEqual(long["padded_seq"], 8192)
        self.assertLess(long["live_tokens"], 16 * 8192)

    def test_odd_lengths_are_not_snapped_to_16(self):
        rows = [token_row(7, 0), token_row(19, 1)]
        ids, mask, padded, live = pad_batch(rows)
        self.assertEqual(padded, 19)
        self.assertEqual(live, 26)
        self.assertEqual(len(ids[0]), 19)
        self.assertEqual(sum(mask[0]), 7)

    def test_quick_is_a_subset(self):
        full = {c["id"] for c in all_cases()}
        quick = {c["id"] for c in all_cases(quick=True)}
        self.assertTrue(quick < full)
        self.assertIn("b1-s7", quick)
        self.assertIn("b1-s8192", quick)
        self.assertIn("mix-b4-short", quick)
        self.assertNotIn("b16-s8192", quick)

    def test_warmup_shrinks_with_tokens(self):
        self.assertEqual(warmup_repeat(128), (3, 15))
        self.assertEqual(warmup_repeat(8192), (2, 5))
        self.assertEqual(warmup_repeat(16 * 8192), (1, 2))
        w_q, r_q = warmup_repeat(128, quick=True)
        self.assertLessEqual(r_q, 8)
        w_s, r_s = warmup_repeat(128, sweep=True)
        self.assertGreaterEqual(r_s, 15)

    def test_compare_gate_uses_rel_and_abs_floor(self):
        base = [{"id": "t4/b1-s128", "p50_ms": 100.0}]
        ok = [{"id": "t4/b1-s128", "p50_ms": 101.5}]
        result = compare_cells(base, ok)
        self.assertEqual(result["regressions"], [])
        bad = [{"id": "t4/b1-s128", "p50_ms": 110.0}]
        result = compare_cells(base, bad)
        self.assertEqual(len(result["regressions"]), 1)
        self.assertGreater(result["regressions"][0]["delta_pct"], 2.0)
        tiny = [{"id": "t4/b1-s7", "p50_ms": 1.0}]
        tiny_now = [{"id": "t4/b1-s7", "p50_ms": 1.2}]
        result = compare_cells(tiny, tiny_now)
        self.assertEqual(result["regressions"], [])
        self.assertGreater(GATE_ABS_MS, 0)

    def test_compare_errors_and_missing(self):
        base = [{"id": "t4/b1-s128", "p50_ms": 10.0}]
        cur = [{"id": "t4/b1-s128", "error": "boom"}]
        result = compare_cells(base, cur)
        self.assertEqual(len(result["regressions"]), 1)
        cur2 = [{"id": "t4/b1-s64", "p50_ms": 8.0}]
        result = compare_cells(base, cur2)
        self.assertEqual(result["missing"], ["t4/b1-s128"])

    def test_thread_budgets_skip_above_physical(self):
        keep, skipped, nphys = thread_budgets(nphys=4)
        self.assertEqual(keep, [1, 2, 4])
        self.assertEqual(skipped, [8])
        self.assertEqual(nphys, 4)
        keep, skipped, _ = thread_budgets([1, 2, 4, 8], nphys=8)
        self.assertEqual(keep, [1, 2, 4, 8])
        self.assertEqual(skipped, [])
        with self.assertRaises(ValueError):
            thread_budgets([16], nphys=4)

    def test_n_physical_cores_positive(self):
        n = n_physical_cores()
        self.assertGreaterEqual(n, 1)
        self.assertIsInstance(n, int)

    def test_sweep_covers_dense_and_unaligned(self):
        lens = sweep_lengths()
        self.assertEqual(lens[0], 1)
        for n in range(1, 601, 7):
            self.assertIn(n, lens)
        self.assertIn(100, lens)
        self.assertIn(300, lens)
        self.assertIn(777, lens)
        self.assertIn(8192, lens)
        longs = [n for n in lens if n >= 600]
        self.assertGreaterEqual(len(longs), 50)
        quick = sweep_lengths(quick=True)
        self.assertLess(len(quick), 20)
        self.assertIn(1, quick)
        self.assertIn(100, quick)

    def test_sweep_cases_are_b1(self):
        cases = sweep_cases(quick=True)
        self.assertTrue(all(c["batch"] == 1 and c["kind"] == "sweep" for c in cases))
        self.assertTrue(all(c["id"].startswith("sweep-s") for c in cases))
        self.assertEqual(cases[0]["repeat"], 15)
        seqs = [c["seq"] for c in cases]
        # Quick set only contains 64 and 128 of the warm-aligned prefix.
        self.assertEqual(seqs[:2], [64, 128])
        self.assertEqual(seqs[2:], sorted(seqs[2:]))

    def test_sweep_cases_honor_batch(self):
        cases = sweep_cases(quick=True, batch=4)
        self.assertTrue(all(c["batch"] == 4 and c["kind"] == "sweep" for c in cases))
        self.assertTrue(all(c["id"].startswith("sweep-b4-s") for c in cases))
        self.assertEqual(len(cases[0]["request"]["input_ids"]), 4)
        with self.assertRaises(ValueError):
            sweep_cases(quick=True, batch=0)

    def test_full_sweep_does_not_run_8k_before_short_unaligned(self):
        seqs = [c["seq"] for c in sweep_cases()]
        self.assertEqual(seqs[:5], [64, 128, 256, 512, 1024])
        self.assertEqual(seqs[-1], 8192)
        self.assertLess(seqs.index(100), seqs.index(8192))
        self.assertLess(seqs.index(777), seqs.index(8192))
        rest = seqs[5:]
        self.assertEqual(rest, sorted(rest))

    def test_sweep_restarts_worker_before_occupancy_cliffs(self):
        # One process for 155 lengths dropped busy cores after ~s=300.
        self.assertGreaterEqual(SWEEP_CASES_PER_PROCESS, 1)
        self.assertLessEqual(SWEEP_CASES_PER_PROCESS, 8)

    def test_smoothness_flags_unaligned_slower_than_next_aligned(self):
        cells = [
            {"id": "t4/sweep-s100", "threads": 4, "batch": 1, "seq": 100, "p50_ms": 36.0, "busy_cores": 2.8},
            {"id": "t4/sweep-s128", "threads": 4, "batch": 1, "seq": 128, "p50_ms": 24.0, "busy_cores": 3.6},
            {"id": "t4/sweep-s777", "threads": 4, "batch": 1, "seq": 777, "p50_ms": 311.0, "busy_cores": 2.2},
            {"id": "t4/sweep-s1024", "threads": 4, "batch": 1, "seq": 1024, "p50_ms": 234.0, "busy_cores": 3.0},
        ]
        result = smoothness_violations(cells)
        seqs = {v["seq"] for v in result["violations"]}
        self.assertIn(100, seqs)
        self.assertIn(777, seqs)
        ok = [
            {"id": "t4/sweep-s100", "threads": 4, "batch": 1, "seq": 100, "p50_ms": 22.0},
            {"id": "t4/sweep-s128", "threads": 4, "batch": 1, "seq": 128, "p50_ms": 24.0},
        ]
        self.assertEqual(smoothness_violations(ok)["violations"], [])
        self.assertEqual(next_aligned(100), 128)
        self.assertEqual(next_aligned(777), 832)
        self.assertEqual(next_aligned(512), 512)


if __name__ == "__main__":
    unittest.main()
