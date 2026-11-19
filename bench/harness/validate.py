#!/usr/bin/env python3
"""Compare caller-supplied frozen JSON requests/oracles without any private runtime imports."""
import argparse
import json
import math
from pathlib import Path
from support import Worker, canonical_response, command_json, compare, file_sha, load_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--command', required=True, type=command_json)
    p.add_argument('--output', required=True)
    p.add_argument('--exact', action='store_true', help='Require exact preparation tensors/structure')
    p.add_argument('--atol', type=float, default=1e-4)
    p.add_argument('--raw-atol', type=float, help='Optional raw-logit gate; default reports raw differences')
    p.add_argument('--timeout', type=float, default=60)
    a = p.parse_args()
    if not math.isfinite(a.atol) or a.atol < 0 or (a.raw_atol is not None and (not math.isfinite(a.raw_atol) or a.raw_atol < 0)):
        p.error('Tolerances must be finite and nonnegative')
    requests = load_records(a.requests)
    reference = {r['id']: r['response'] for r in load_records(a.reference)}
    if set(reference) != {r['id'] for r in requests}:
        p.error('Frozen request/reference IDs must match exactly')
    summary = {'cases': 0, 'failures': [], 'atol': a.atol, 'raw_atol': a.raw_atol,
               'exact': a.exact, 'request_sha256': file_sha(a.requests),
               'reference_sha256': file_sha(a.reference), 'numeric': None}
    with Path(a.output).open('x') as target, Worker(a.command, a.timeout) as worker:
        for case in requests:
            request = case['request']
            response = canonical_response(request, worker.call(request))
            try:
                summary['numeric'] = compare(reference[case['id']], response, atol=a.atol,
                                             raw_atol=a.raw_atol, exact=a.exact,
                                             stats=summary['numeric'])
            except ValueError as error:
                summary['failures'].append({'id': case['id'], 'reason': str(error)})
            summary['cases'] += 1
        summary['pass'] = not summary['failures']
        json.dump(summary, target, indent=2)
        target.write('\n')
    print(json.dumps({'cases': summary['cases'], 'pass': summary['pass'], 'numeric': summary['numeric']}))
    if not summary['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
