#!/usr/bin/env python3
"""Public ORT FP32 prepared-tensor reference worker; no text frontend or private helper imports."""
import argparse
import json
import os
import time
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "harness"))
from support import KEYS, file_sha, memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', required=True)
    p.add_argument('--expected-sha256', required=True, help='Hash of the previously verified FP32 reference graph')
    p.add_argument('--preverified', action='store_true', help='Caller already verified the graph hash before timed worker startup')
    p.add_argument('--threads', type=int, choices=[1, 2, 4], default=2)
    a = p.parse_args()
    verify_start, verify_cpu = time.perf_counter_ns(), time.process_time_ns()
    if not a.preverified and file_sha(a.onnx) != a.expected_sha256:
        p.error('Reference graph hash differs')
    verification_wall_s=(time.perf_counter_ns()-verify_start)/1e9
    verification_cpu_s=(time.process_time_ns()-verify_cpu)/1e9
    os.sched_setaffinity(0, range(a.threads))
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[name] = str(a.threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import numpy as np
    import onnxruntime as ort
    start, cpu = time.perf_counter_ns(), time.process_time_ns()
    options = ort.SessionOptions()
    options.intra_op_num_threads = a.threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(a.onnx, sess_options=options, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if {v.name for v in inputs} != set(KEYS) or any(v.type != 'tensor(int64)' for v in inputs):
        raise ValueError('Expected official six-tensor INT64 graph contract')
    if len(outputs) != 2 or any(v.type != 'tensor(float)' for v in outputs):
        raise ValueError('Expected two FP32 official graph outputs')
    print(json.dumps({'ready':True,'backend':'original-fp32-onnx','precision':'FP32',
                      'threads':a.threads,'optimization':str(options.graph_optimization_level),
                      'execution_mode':str(options.execution_mode),'providers':session.get_providers(),
                      'reference_sha256':a.expected_sha256,'load_wall_s':(time.perf_counter_ns()-start)/1e9,
                      'load_cpu_s':(time.process_time_ns()-cpu)/1e9,'memory':memory(os.getpid()),
                      'reference_hash_preverified':a.preverified,'asset_verification_wall_s':verification_wall_s,
                      'asset_verification_cpu_s':verification_cpu_s}),flush=True)
    import sys
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('operation','tensor') != 'tensor':
                raise ValueError('This worker implements prepared-tensor inference only')
            arrays = {key:np.asarray(request[key],dtype=np.int64) for key in KEYS}
            repeats, warmups = request.get('repeat',1), request.get('warmup',0)
            if not 1 <= repeats <= 10000 or not 0 <= warmups <= 10000:
                raise ValueError('Invalid repetition count')
            for _ in range(warmups):
                session.run(None, arrays)
            walls, cpus = [], []
            for _ in range(repeats):
                start, cpu = time.perf_counter_ns(), time.process_time_ns()
                option, span = session.run(None, arrays)
                walls.append((time.perf_counter_ns()-start)/1e9)
                cpus.append((time.process_time_ns()-cpu)/1e9)
            response = {'wall_s':walls,'cpu_s':cpus,'memory':memory(os.getpid())}
            if request.get('return_logits',True):
                response.update(opt_logits=option.tolist(),span_logits=span.tolist())
        except Exception as error:
            response = {'error':{'type':type(error).__name__,'message':str(error)}}
        print(json.dumps(response,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
