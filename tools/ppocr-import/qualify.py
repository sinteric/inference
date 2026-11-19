#!/usr/bin/env python3
"""Qualify deterministic import and persist ORT reference fixtures (round 1)."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper,numpy_helper


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def qualify(models,assets,pages):
    if ort.__version__!='1.28.0': raise ValueError(f'expected onnxruntime 1.28.0, got {ort.__version__}')
    importer=module('ppocr_import','import.py');dumper=module('ppocr_dump','ort_dump.py');report={}
    for kind,(repo,_,_) in importer.PINS.items():
        source=models/repo/'inference.onnx'
        with tempfile.TemporaryDirectory(dir=assets,prefix='reimport-') as temporary:
            importer.run(source,Path(temporary),kind)
            for path in Path(temporary).iterdir():
                if path.read_bytes()!=(assets/kind/path.name).read_bytes(): raise ValueError(f'nondeterministic {path.name}')
        model=onnx.load(source)
        nodes,constants,outputs,bn=importer.fold(model)
        nodes=importer.match_patterns(nodes,constants,outputs)
        nodes,constants=importer.expand_patterns(nodes,constants)
        used={x for n in nodes for x in n['inputs'] if x in constants}
        tensors=[numpy_helper.from_array(constants[k],k) for k in sorted(used)]
        inputs=list(model.graph.input)
        if model.opset_import[0].version<=8: inputs += [helper.make_tensor_value_info(i.name,i.data_type,i.dims) for i in tensors]
        out=[]
        for original,name in zip(model.graph.output,outputs):
            o=onnx.ValueInfoProto();o.CopyFrom(original);o.name=name;out.append(o)
        ns=[helper.make_node(n['op'],n['inputs'],[n['output']],name=n['source_name'],**n['attrs']) for n in nodes]
        folded=helper.make_model(helper.make_graph(ns,'normalized',inputs,out,tensors),opset_imports=list(model.opset_import));folded.ir_version=model.ir_version
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;opts.log_severity_level=3
        reference=ort.InferenceSession(model.SerializeToString(),opts,providers=['CPUExecutionProvider'])
        converted=ort.InferenceSession(folded.SerializeToString(),opts,providers=['CPUExecutionProvider'])
        cases=[]
        if kind=='det':
            shapes=[(1,3,64,96),(1,3,96,64),(2,3,64,64)]
        else:
            shapes=[(1,3,48,w) for w in [16,48,320,1600]]+[(2,3,48,48)]
        def case(name,x):
            a=reference.run(None,{'x':x})[0];b=converted.run(None,{'x':x})[0]
            delta=float(np.max(np.abs(a-b))); entry={'name':name,'input_shape':list(x.shape),'max_abs':delta}
            if delta>1e-4: raise ValueError(f'{kind}/{name}: max abs {delta}')
            if kind=='rec':
                entry['argmax_mismatches']=int(np.count_nonzero(a.argmax(-1)!=b.argmax(-1)))
                if entry['argmax_mismatches']: raise ValueError('rec argmax mismatch')
            else:
                entry['threshold_mismatch_fraction']=float(np.mean((a>0.2)!=(b>0.2)))
                if entry['threshold_mismatch_fraction']>1e-5: raise ValueError('det threshold mismatch')
            directory=assets/kind/'ort'/name;directory.mkdir(parents=True,exist_ok=True)
            np.save(directory/'input.npy',x,allow_pickle=False);np.save(directory/'output000.npy',a,allow_pickle=False)
            index={'source_sha256':importer.PINS[kind][2],'onnxruntime':ort.__version__,'input':{'file':'input.npy','shape':list(x.shape)},'outputs':[{'name':model.graph.output[0].name,'file':'output000.npy','shape':list(a.shape)}]}
            (directory/'index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
            cases.append(entry);print(json.dumps({'model':kind,**entry}),flush=True)
        for shape in shapes:
            case('x'.join(map(str,shape)),np.random.default_rng(0).uniform(-1,1,shape).astype(np.float32))
        if kind=='det' and pages:
            for name in ['kd0092','kd0330','kd0444','kd0822']:
                case(name,dumper.page_tensor(pages/f'{name}_200.png'))
        report[kind]={'onnx':onnx.__version__,'onnxruntime':ort.__version__,'bn_folded':bn,'cases':cases,'deterministic':True,'max_abs':max(c['max_abs'] for c in cases)}
    (assets/'round1-parity.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models-root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--pages',type=Path)
    a=p.parse_args();qualify(a.models_root,a.assets,a.pages)
