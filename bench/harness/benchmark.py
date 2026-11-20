#!/usr/bin/env python3
"""Caller-supplied frozen fixtures and JSON workers; rotated CPU controls without private helpers."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from support import Worker, command_json, file_sha, load_records, memory, physical_cpus, check_affinity_budget, worker_task_affinities


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lower = int(position)
    upper = min(lower+1,len(values)-1)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def stats(wall, cpu, batch):
    return {'p50_ms':percentile(wall,.5)*1000,'p95_ms':percentile(wall,.95)*1000,
            'p99_ms':percentile(wall,.99)*1000,'cpu_mean_ms':statistics.mean(cpu)*1000,
            'cpu_utilization_cores':sum(cpu)/sum(wall),'throughput_rows_s':batch*len(wall)/sum(wall)}


def preverify_reference(command):
    """Keep this worker's full-file identity check outside measured cold startup."""
    if not any(Path(part).name=='onnx_driver.py' for part in command):
        return command
    try:
        graph=command[command.index('--onnx')+1]
        expected=command[command.index('--expected-sha256')+1]
    except (ValueError,IndexError) as error:
        raise ValueError('ONNX worker requires explicit graph and expected hash') from error
    if file_sha(graph)!=expected:
        raise ValueError('Reference graph hash differs before worker startup')
    return command if '--preverified' in command else command+['--preverified']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',required=True,help='JSONL records: id, request, batch')
    p.add_argument('--commands',required=True,help='JSON file mapping backend names to command arrays; {threads} is replaced')
    p.add_argument('--scope',choices=['model','api'],required=True)
    p.add_argument('--budgets',default='1,2,4',help='comma-separated positive thread counts; any N>=1')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeat',type=int,default=15)
    p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    budgets=[int(v) for v in a.budgets.split(',')]
    if not budgets or any(n < 1 for n in budgets) or a.rounds<1 or a.repeat<1 or a.warmup<0:
        p.error('Invalid CPU budget or repetition count')
    cases=load_records(a.requests)
    for case in cases:
        if not isinstance(case.get('batch'),int) or case['batch']<1:
            p.error('Each fixture requires a positive batch size')
        if a.scope=='model' and case['request'].get('operation','tensor')!='tensor':
            p.error('Model scope requires prepared-tensor requests')
    commands=json.loads(Path(a.commands).read_text())
    if not isinstance(commands,dict) or not commands:
        p.error('Backend commands must be a nonempty object')
    commands={name:preverify_reference(command_json(json.dumps(command))) for name,command in commands.items()}
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    protocol={'scope':a.scope,'budgets':budgets,'rounds':a.rounds,'repeat':a.repeat,'warmup':a.warmup,
              'fixture_sha256':file_sha(a.requests),'commands_sha256':file_sha(a.commands),
              'cold_definition':'fresh process with ordinary OS page cache; later cases are first-shape, not fresh-process',
              'memory_definition':'sampled current worker RSS plus lifetime VmHWM; sum of peaks is only an upper bound',
              'timing_boundary':'internal forward' if a.scope=='model' else 'complete persistent JSON request and controller/worker CPU'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    names=list(commands)
    for n in budgets:
        cpus=physical_cpus(n)
        os.sched_setaffinity(0,cpus)
        os.environ.update(OMP_NUM_THREADS=str(n),OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='ACTIVE',
                          GOMP_SPINCOUNT='500',OMP_PROC_BIND='close',
                          OMP_PLACES=','.join('{'+str(c)+'}' for c in cpus),
                          RAYON_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                          OPENBLAS_NUM_THREADS=str(n),MKL_NUM_THREADS=str(n))
        for cohort in range(a.rounds):
            for name in names[cohort%len(names):]+names[:cohort%len(names)]:
                command=[v.replace('{threads}',str(n)) for v in commands[name]]
                run=out/f'{n}cpu-{cohort+1}-{name}'
                run.mkdir()
                with (run/'stderr.log').open('x') as errors, Worker(command,a.timeout,stderr=errors) as worker:
                    if worker.ready.get('threads')!=n:
                        raise ValueError('Worker thread budget differs')
                    (run/'startup.json').write_text(json.dumps({'command':command,'ready':worker.ready,
                        'startup_wall_s':worker.startup_wall_s,'startup_combined_cpu_s':worker.startup_cpu_s,
                        'controller':memory(os.getpid()),'worker':memory(worker.child.pid)},indent=2)+'\n')
                    with (run/'samples.jsonl').open('x') as target:
                        for case in cases:
                            request=dict(case['request'])
                            if a.scope=='model':
                                request.update(repeat=1,warmup=0,return_logits=False)
                            clock=lambda:time.process_time_ns()+worker.clock()
                            start,cpu=time.perf_counter_ns(),clock()
                            first=worker.call(request)
                            cold={'wall_s':(time.perf_counter_ns()-start)/1e9,
                                  'combined_cpu_s':(clock()-cpu)/1e9,'worker_memory':memory(worker.child.pid)}
                            if isinstance(first,dict) and 'error' in first:
                                raise ValueError(f'Worker returned error for fixture {case["id"]}: {first["error"]}')
                            if a.scope=='model':
                                cold.update(internal_wall_s=first['wall_s'][0],internal_cpu_s=first['cpu_s'][0])
                                response=worker.call(dict(request,repeat=a.repeat,warmup=a.warmup))
                                if 'error' in response:
                                    raise ValueError(response['error'])
                                walls,cpu_times=response['wall_s'],response['cpu_s']
                            else:
                                for _ in range(a.warmup):
                                    response=worker.call(request)
                                    if isinstance(response,dict) and 'error' in response:
                                        raise ValueError(response['error'])
                                walls,cpu_times=[],[]
                                for _ in range(a.repeat):
                                    start,cpu=time.perf_counter_ns(),clock()
                                    response=worker.call(request)
                                    walls.append((time.perf_counter_ns()-start)/1e9)
                                    cpu_times.append((clock()-cpu)/1e9)
                                    if isinstance(response,dict) and 'error' in response:
                                        raise ValueError(response['error'])
                            if len(walls)!=a.repeat or len(cpu_times)!=a.repeat:
                                raise ValueError('Worker sample count differs')
                            affinities=worker_task_affinities(worker.child.pid)
                            check_affinity_budget(cpus, affinities)
                            controller,process=memory(os.getpid()),memory(worker.child.pid)
                            record={'id':case['id'],'backend':name,'threads':n,'cohort':cohort+1,
                                    'batch':case['batch'],'cold':cold,'wall_s':walls,'cpu_s':cpu_times,
                                    'controller_memory':controller,'worker_memory':process,
                                    'combined_rss_bytes':controller['VmRSS']+process['VmRSS'],
                                    'individual_peak_sum_upper_bound_bytes':controller['VmHWM']+process['VmHWM'],
                                    **stats(walls,cpu_times,case['batch'])}
                            target.write(json.dumps(record)+'\n')
                            target.flush()
    print(json.dumps({'complete':True,'scope':a.scope,'budgets':budgets,'rounds':a.rounds}))


if __name__=='__main__':
    main()
