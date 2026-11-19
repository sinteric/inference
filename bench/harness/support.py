"""Portable JSON worker transport and independent output comparison. Standard library only."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time

KEYS = ['input_ids', 'attention_mask', 'q_index', 'opt_index', 'unit_index', 'ent_index']
RAW_LOGITS = {'opt_logits', 'decision_logits', 'span_logits'}


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [record['id'] for record in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError('Records must be nonempty and have unique IDs')
    return records


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_cpus(n):
    """First `n` physical cores (lowest sibling id of each thread_siblings group)."""
    base = Path('/sys/devices/system/cpu')
    groups = {}
    extras = []
    if base.is_dir():
        for cpu in sorted(base.glob('cpu[0-9]*'), key=lambda p: int(p.name[3:])):
            ident = int(cpu.name[3:])
            sib = cpu / 'topology' / 'thread_siblings_list'
            key = sib.read_text().strip() if sib.is_file() else str(ident)
            groups.setdefault(key, ident)
            extras.append(ident)
    phys = sorted(groups.values()) if groups else extras
    if len(phys) < n:
        for c in extras:
            if c not in phys:
                phys.append(c)
    if not phys:
        phys = list(range(max(n, 1)))
    return [phys[i % len(phys)] for i in range(n)]


def command_json(value):
    command = json.loads(value)
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        raise ValueError('Command must be a nonempty JSON array of strings')
    return command


def cpu_clock(pid):
    lib = ctypes.CDLL(None)
    lib.clock_getcpuclockid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.clock_getcpuclockid.restype = ctypes.c_int
    clock = ctypes.c_int()
    error = lib.clock_getcpuclockid(pid, ctypes.byref(clock))
    if error:
        raise OSError(error, os.strerror(error))
    return lambda: time.clock_gettime_ns(clock.value)


def memory(pid):
    result = {}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            result[line.split(':')[0]] = int(line.split()[1]) * 1024
    return result


def worker_task_affinities(pid):
    """Per-task CPU sets of a live worker (logical CPU ids, not CPU-time samples)."""
    return [sorted(os.sched_getaffinity(int(t.name))) for t in Path(f'/proc/{pid}/task').iterdir()]


def check_affinity_budget(allowed_cpus, affinities):
    """Raise if any worker task ran outside the CPU-id budget.

    `allowed_cpus` must be logical CPU ids (ints), never `response['cpu_s']`.
    """
    allowed = set(allowed_cpus)
    if any(not set(cpuset) <= allowed for cpuset in affinities):
        raise ValueError('Worker exceeded CPU affinity budget')


class Worker:
    def __init__(self, command, timeout=60, stderr=None):
        self.timeout = timeout
        start, cpu = time.perf_counter_ns(), time.process_time_ns()
        self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=stderr, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.child.stdout, selectors.EVENT_READ)
        try:
            self.ready = self.receive()
            if not isinstance(self.ready, dict) or not self.ready.get('ready'):
                raise ValueError('Worker did not announce ready')
            self.clock = cpu_clock(self.child.pid)
            self.startup_wall_s = (time.perf_counter_ns()-start)/1e9
            self.startup_cpu_s = (time.process_time_ns()-cpu+self.clock())/1e9
        except BaseException:
            self.close()
            raise

    def receive(self):
        if not self.selector.select(self.timeout):
            raise TimeoutError('Worker response timeout')
        line = self.child.stdout.readline()
        if not line:
            raise RuntimeError(f'Worker EOF, exit={self.child.poll()}')
        return json.loads(line)

    def call(self, request):
        self.child.stdin.write(json.dumps(request, ensure_ascii=False, separators=(',', ':'))+'\n')
        self.child.stdin.flush()
        return self.receive()

    def close(self):
        self.selector.close()
        if self.child.stdin:
            try:
                self.child.stdin.close()
            except BrokenPipeError:
                pass
        try:
            status = self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            try:
                status = self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.child.kill()
                status = self.child.wait()
        if self.child.stdout:
            self.child.stdout.close()
        return status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        status = self.close()
        if args[0] is None and status:
            raise RuntimeError(f'Worker failed with exit={status}')


def compare(reference, candidate, *, atol=1e-4, raw_atol=None, exact=False, path=(), stats=None):
    """Exact structure/labels/integers; gated floats; separately reported raw logits."""
    if stats is None:
        stats = {'float_count': 0, 'raw_logit_count': 0, 'max_abs_float_diff': 0.0,
                 'max_abs_raw_logit_diff': 0.0}
    location = '.'.join(map(str, path)) or '<root>'
    if isinstance(reference, dict):
        if not isinstance(candidate, dict) or reference.keys() != candidate.keys():
            raise ValueError(f'Object keys differ at {location}')
        for key in reference:
            compare(reference[key], candidate[key], atol=atol, raw_atol=raw_atol,
                    exact=exact, path=path+(key,), stats=stats)
    elif isinstance(reference, list):
        if not isinstance(candidate, list) or len(reference) != len(candidate):
            raise ValueError(f'Array shape differs at {location}')
        for index, (r, c) in enumerate(zip(reference, candidate)):
            compare(r, c, atol=atol, raw_atol=raw_atol, exact=exact,
                    path=path+(index,), stats=stats)
    elif isinstance(reference, float):
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            raise ValueError(f'Numeric type differs at {location}')
        if not math.isfinite(reference) or not math.isfinite(candidate):
            raise ValueError(f'Nonfinite value at {location}')
        diff = abs(reference-candidate)
        raw = any(part in RAW_LOGITS for part in path if isinstance(part, str))
        count, maximum = ('raw_logit_count','max_abs_raw_logit_diff') if raw else ('float_count','max_abs_float_diff')
        stats[count] += 1
        stats[maximum] = max(stats[maximum], diff)
        tolerance = 0.0 if exact else raw_atol if raw else atol
        if tolerance is not None and diff > tolerance:
            raise ValueError(f'Numeric tolerance exceeded at {location}: {diff}')
    elif type(reference) is not type(candidate) or reference != candidate:
        raise ValueError(f'Exact value differs at {location}')
    return stats


def canonical_response(request, response):
    if request.get('operation', 'tensor') == 'tensor' and 'error' not in response:
        return {key: response[key] for key in ['opt_logits', 'span_logits']}
    return response
