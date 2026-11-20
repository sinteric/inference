"""Portable contract tests and optional public oneDNN primitive integration tests; no model assets."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness'))
from support import Worker, canonical_response, check_affinity_budget, command_json, compare, load_records, physical_cpus, worker_task_affinities
from benchmark import percentile, preverify_reference, stats


class ValidationTests(unittest.TestCase):
    def test_decoded_structure_and_integer_offsets_must_match(self):
        for changed in [{'label':'B','start':1},{'label':'A','start':2},{'label':'A','start':1.0}]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                compare({'label':'A','start':1},changed)

    def test_probabilities_gate_but_raw_logits_are_reported(self):
        result=compare({'opt_logits':[1.0],'probability':.5},{'opt_logits':[1.0002],'probability':.50001})
        self.assertAlmostEqual(result['max_abs_raw_logit_diff'],.0002)
        self.assertAlmostEqual(result['max_abs_float_diff'],.00001)
        with self.assertRaises(ValueError):
            compare({'opt_logits':[1.0]}, {'opt_logits':[1.0002]},raw_atol=1e-4)
        with self.assertRaises(ValueError):
            compare({'probability':.5},{'probability':.5002})

    def test_preparation_requires_exact_values_and_shapes(self):
        for changed in [[[1,3]],[[1]],[[1,2],[3,4]]]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                compare({'input_ids':[[1,2]]},{'input_ids':changed},exact=True)

    def test_nonfinite_numbers_and_boolean_substitution_fail(self):
        for value in [float('nan'),float('inf'),True]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                compare(.5,value)

    def test_request_sets_cannot_be_empty_or_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'records.jsonl'
            for data in ['', '{"id":"synthetic"}\n{"id":"synthetic"}\n']:
                path.write_text(data)
                with self.assertRaises(ValueError):
                    load_records(path)

    def test_command_uses_argument_array_without_shell(self):
        self.assertEqual(command_json('["program","literal argument"]'),['program','literal argument'])
        for value in ['[]','"program"','[1]']:
            with self.assertRaises(ValueError):
                command_json(value)

    def test_timing_metadata_is_not_a_quality_output(self):
        self.assertEqual(canonical_response({}, {'opt_logits':[1.0],'span_logits':[[2.0]],'wall_s':[99]}),
                         {'opt_logits':[1.0],'span_logits':[[2.0]]})

    def test_affinity_budget_uses_cpu_ids_not_cpu_times(self):
        ids = physical_cpus(2)
        affinities = [ids[:1], ids]
        check_affinity_budget(ids, affinities)
        cpu_times = [0.12, 0.15]
        with self.assertRaises(ValueError) as ctx:
            check_affinity_budget(cpu_times, affinities)
        self.assertIn('CPU affinity budget', str(ctx.exception))
        me = os.getpid()
        mine = sorted(os.sched_getaffinity(me))
        check_affinity_budget(mine, worker_task_affinities(me))

    def test_benchmark_model_scope_does_not_shadow_cpu_ids(self):
        src = (Path(__file__).resolve().parents[1] / 'harness' / 'benchmark.py').read_text()
        self.assertIn("cpus=physical_cpus(n)", src)
        self.assertIn("walls,cpu_times=response['wall_s'],response['cpu_s']", src)
        self.assertNotIn("walls,cpus=response['wall_s'],response['cpu_s']", src)
        self.assertIn("check_affinity_budget(cpus, affinities)", src)

    def test_physical_cpus_allow_any_positive_count(self):
        for n in (1,2,3,5,8):
            cpus=physical_cpus(n)
            self.assertEqual(len(cpus),n)
            self.assertTrue(all(isinstance(c,int) and c>=0 for c in cpus))
        two=physical_cpus(2)
        if len(set(physical_cpus(8)))>=2:
            # Default 2-thread budget should not be forced to [0,1] when those
            # are SMT siblings; physical_cpus picks one id per sibling group.
            self.assertEqual(len(set(two)),2)

    def test_cpu_work_and_percentiles_have_correct_denominators(self):
        self.assertEqual(percentile([1,2,3,4],.5),2.5)
        self.assertAlmostEqual(percentile([1,2,3,4],.95),3.85)
        result=stats([1,3],[2,6],4)
        self.assertEqual(result['cpu_utilization_cores'],2)
        self.assertEqual(result['throughput_rows_s'],2)

    def test_reference_hash_is_checked_before_timed_startup(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            graph=Path(directory)/'synthetic.onnx'
            graph.write_bytes(b'synthetic test bytes; not a model')
            expected=hashlib.sha256(graph.read_bytes()).hexdigest()
            command=['python3','scripts/onnx_driver.py','--onnx',str(graph),'--expected-sha256',expected]
            self.assertIn('--preverified',preverify_reference(command))
            with self.assertRaises(ValueError):
                preverify_reference(command[:-1]+['0'*64])

    def test_persistent_worker_and_validator_cli(self):
        code='import json,sys\nprint(json.dumps({"ready":True}),flush=True)\nfor line in sys.stdin:\n print(json.dumps(json.loads(line)),flush=True)\n'
        command=[sys.executable,'-u','-c',code]
        with Worker(command,timeout=5) as worker:
            self.assertEqual(worker.call({'operation':'prepare','values':[1,2]}),{'operation':'prepare','values':[1,2]})
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            request={'operation':'prepare','values':[1,2]}
            (directory/'requests.jsonl').write_text(json.dumps({'id':'synthetic','request':request})+'\n')
            (directory/'reference.jsonl').write_text(json.dumps({'id':'synthetic','response':request})+'\n')
            result=subprocess.run([sys.executable,str(ROOT/'harness/validate.py'),'--requests',str(directory/'requests.jsonl'),
                '--reference',str(directory/'reference.jsonl'),'--command',json.dumps(command),'--exact',
                '--output',str(directory/'summary.json')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(json.loads((directory/'summary.json').read_text())['pass'])
            (directory/'reference.jsonl').write_text(json.dumps({'id':'synthetic','response':dict(request,values=[1,3])})+'\n')
            result=subprocess.run([sys.executable,str(ROOT/'harness/validate.py'),'--requests',str(directory/'requests.jsonl'),
                '--reference',str(directory/'reference.jsonl'),'--command',json.dumps(command),'--exact',
                '--output',str(directory/'failed-summary.json')],capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            self.assertFalse(json.loads((directory/'failed-summary.json').read_text())['pass'])


if __name__=='__main__':
    unittest.main()
