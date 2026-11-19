"""Asset-free tests for EG2 thread-scaling cases."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from scaling import (  # noqa: E402
    SCALE_SEQS,
    concurrent_cases,
    concurrent_rows,
    efficiency_rows,
    thread_budgets,
    uniform_scale_cases,
)


class ScalingCaseTests(unittest.TestCase):
    def test_seqs_include_odd_and_long(self):
        self.assertIn(7, SCALE_SEQS)
        self.assertIn(19, SCALE_SEQS)
        self.assertIn(777, SCALE_SEQS)
        self.assertIn(8192, SCALE_SEQS)
        for s in (7, 19, 777):
            self.assertNotEqual(s % 16, 0, s)

    def test_uniform_covers_short_and_batch(self):
        ids = {c["id"] for c in uniform_scale_cases()}
        self.assertIn("b1-s7", ids)
        self.assertIn("b1-s8192", ids)
        self.assertIn("b4-s128", ids)
        self.assertIn("b4-s512", ids)

    def test_thread_budgets_skip_above_physical(self):
        keep, skipped, nphys = thread_budgets(nphys=4)
        self.assertEqual(keep, [1, 2, 4])
        self.assertEqual(skipped, [8])
        self.assertEqual(nphys, 4)
        keep8, skipped8, _ = thread_budgets(nphys=8)
        self.assertEqual(keep8, [1, 2, 4, 8])
        self.assertEqual(skipped8, [])

    def test_concurrent_cases_set_slots_and_instances(self):
        cases = {c["id"]: c for c in concurrent_cases(nphys=4)}
        slots = cases["slots2-b1-s128"]
        self.assertEqual(slots["exec_slots"], 2)
        self.assertEqual(slots["instances"], 1)
        self.assertEqual(slots["concurrency"], 2)
        self.assertEqual(slots["request"]["concurrency"], 2)
        inst = cases["inst2-b1-s128"]
        self.assertEqual(inst["exec_slots"], 1)
        self.assertEqual(inst["instances"], 2)
        self.assertEqual(inst["concurrency"], 2)

    def test_efficiency_vs_one_thread(self):
        cells = [
            {"id": "t1/b1-s128", "case_id": "b1-s128", "kind": "scale", "threads": 1, "p50_ms": 40.0, "busy_cores": 1.0},
            {"id": "t2/b1-s128", "case_id": "b1-s128", "kind": "scale", "threads": 2, "p50_ms": 22.0, "busy_cores": 1.8},
            {"id": "t4/b1-s128", "case_id": "b1-s128", "kind": "scale", "threads": 4, "p50_ms": 12.0, "busy_cores": 3.5},
        ]
        rows = {r["threads"]: r for r in efficiency_rows(cells)}
        self.assertAlmostEqual(rows[1]["speedup_vs_t1"], 1.0)
        self.assertAlmostEqual(rows[2]["speedup_vs_t1"], 40.0 / 22.0)
        self.assertAlmostEqual(rows[4]["efficiency"], (40.0 / 12.0) / 4.0)

    def test_concurrent_wave_over_seq(self):
        cells = [
            {"id": "t4/b1-s128", "case_id": "b1-s128", "kind": "scale", "threads": 4, "batch": 1, "seq": 128, "p50_ms": 10.0},
            {
                "id": "t4/slots2-b1-s128/slots2/inst1/c2",
                "kind": "slots",
                "threads": 4,
                "batch": 1,
                "seq": 128,
                "concurrency": 2,
                "exec_slots": 2,
                "instances": 1,
                "p50_ms": 19.0,
                "busy_cores": 3.6,
            },
        ]
        rows = concurrent_rows(cells)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["wave_over_seq"], 1.9)
        self.assertGreater(rows[0]["qps"], rows[0]["seq_qps"])


if __name__ == "__main__":
    unittest.main()
