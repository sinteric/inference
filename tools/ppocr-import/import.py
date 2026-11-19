#!/usr/bin/env python3
"""Deterministic importer for the three pinned Paddle ONNX exports (no runtime ONNX)."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import helper, numpy_helper

VERSION = 1
REC_YAML_SHA256 = "f757fa1c40e99edcf27e9cce879b93eb2a51fa46f5ef39095689b8c37dd75998"
PINS = {
    'det': ('PP-OCRv6_tiny_det_onnx', '2ba1506c0380b8f0b03dd142459aac66d4421f6c', '193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8'),
    'rec': ('korean_PP-OCRv5_mobile_rec_onnx', '5c6f574b8e2230adf4287b33e736d71b9fabd28e', '92f0b7785e64fc9090106a241cf4c1eb97472824558272751b88a2a4476d3a08'),
    'uvdoc': ('UVDoc_onnx', '3bcf535371727d11e783101f79a504c68848aae3', '54cab30dc2cf347b4f3d6971c833b1c3d84f5ef17280f1f6d15796bff20e63d6'),
}
# These shape/view ops describe the compiled SVTR attention graph, not a generic
# runtime interpreter. Anything outside this list or the matched patterns fails.
OPS = {'Conv', 'ConvTranspose', 'Relu', 'HardSigmoid', 'HardSwish', 'Swish', 'GeluErf',
       'Sigmoid', 'GlobalAveragePool', 'ReduceMean', 'Resize', 'MaxPool', 'AveragePool',
       'Concat', 'Add', 'Mul', 'MatMul', 'Softmax', 'LayerNorm', 'Shape', 'Slice',
       'Squeeze', 'Unsqueeze', 'Reshape', 'Transpose'}
# UVDoc: a dilated-convolution backbone, a reflect-padded head, then the
# align-corners resizes and the grid sample that apply its 2D grid.
UVDOC_OPS = {'Conv', 'Relu', 'Add', 'Concat', 'PRelu', 'Pad', 'Unsqueeze', 'Squeeze',
             'Resize', 'Transpose', 'GridSample', 'Shape', 'Slice', 'Reshape'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def fold(model):
    constants = {i.name: numpy_helper.to_array(i).copy() for i in model.graph.initializer}
    aliases, nodes = {}, []
    def resolve(x):
        while x in aliases:
            x = aliases[x]
        return x
    for node in model.graph.node:
        ins = [resolve(x) for x in node.input]
        a = attrs(node)
        if node.op_type == 'Constant':
            constants[node.output[0]] = numpy_helper.to_array(a['value']).copy()
            continue
        if node.op_type == 'Identity':
            aliases[node.output[0]] = ins[0]
            continue
        v = [constants.get(x) for x in ins]
        result = None
        if all(x is not None for x in v):
            op = node.op_type
            if op == 'Reshape': result = np.reshape(v[0], v[1].astype(int).tolist())
            elif op == 'Shape': result = np.array(v[0].shape, dtype=np.int64)
            elif op == 'Squeeze': result = np.squeeze(v[0], axis=tuple(a.get('axes', [])))
            elif op == 'Unsqueeze':
                result = v[0]
                for axis in sorted(a['axes']): result = np.expand_dims(result, axis)
            elif op == 'Concat': result = np.concatenate(v, axis=a['axis'])
            elif op in ('Add', 'Mul', 'Sub', 'Div', 'Pow'):
                result = {'Add':np.add,'Mul':np.multiply,'Sub':np.subtract,'Div':np.divide,'Pow':np.power}[op](*v)
            elif op == 'Transpose': result = np.transpose(v[0], a.get('perm'))
            elif op == 'Slice':
                slices = [slice(None)] * v[0].ndim
                for axis,start,end in zip(a.get('axes', range(len(a['starts']))), a['starts'], a['ends']): slices[axis] = slice(start,end)
                result = v[0][tuple(slices)]
            else: raise ValueError(f'unmatched constant op {op}: {node.name}')
        if result is not None:
            constants[node.output[0]] = np.asarray(result).copy()
        else:
            nodes.append({'op':node.op_type,'inputs':ins,'output':node.output[0],'attrs':a,'source_name':node.name})
    outputs = [resolve(o.name) for o in model.graph.output]
    # Fuse affine transforms and explicit BN into a singly consumed Conv.
    folded_bn = 0
    changed = True
    while changed:
        changed = False
        producers = {n['output']:n for n in nodes}
        uses = collections.Counter(x for n in nodes for x in n['inputs'])
        uses.update(outputs)
        for n in nodes:
            if n['op'] not in ('BatchNormalization','Add','Mul'): continue
            c = producers.get(n['inputs'][0])
            if c is None or c['op'] not in ('Conv','ConvTranspose') or uses[c['output']] != 1: continue
            if any(x not in constants for x in n['inputs'][1:]): continue
            w = constants[c['inputs'][1]]
            if c['op'] == 'ConvTranspose' and n['op'] == 'BatchNormalization': raise ValueError('unexpected deconv BN')
            channels = w.shape[0] if c['op'] == 'Conv' else w.shape[1]
            b = constants[c['inputs'][2]].reshape(-1) if len(c['inputs']) > 2 else np.zeros(channels, dtype=np.float32)
            if n['op'] == 'BatchNormalization':
                gamma,beta,mean,var = [constants[x] for x in n['inputs'][1:]]
                scale = gamma / np.sqrt(var + np.float32(n['attrs'].get('epsilon', 1e-5)))
                bias = beta - mean * scale
                folded_bn += 1
            else:
                val = constants[n['inputs'][1]]
                if val.size != channels or (val.ndim > 1 and val.shape != (1, channels, 1, 1)): continue
                val = val.reshape(-1)
                scale,bias = (val,np.zeros(channels,np.float32)) if n['op']=='Mul' else (np.ones(channels,np.float32),val)
            if c['op'] == 'Conv': new_w = w * scale[:,None,None,None]
            else: new_w = w * scale[None,:,None,None]
            wn,bn = c['output']+'.folded_w',c['output']+'.folded_b'
            constants[wn] = new_w.astype(np.float32)
            constants[bn] = (b*scale+bias).astype(np.float32)
            c['inputs'] = c['inputs'][:1] + [wn,bn]
            c['output'] = n['output']
            nodes.remove(n)
            changed = True
            break
    return nodes,constants,outputs,folded_bn


def match_patterns(nodes, constants, outputs, kind='det'):
    def scalar(x, value):
        v = constants.get(x)
        return v is not None and v.size == 1 and abs(float(v.reshape(-1)[0])-value) < 1e-6
    # Replace closed expressions only: an intermediate with another consumer must
    # never disappear. All parameters are checked rather than guessed from names.
    while True:
        producers = {n['output']:n for n in nodes}
        uses = collections.Counter(x for n in nodes for x in n['inputs']); uses.update(outputs)
        replacement = None
        for n in nodes:
            if n['op'] != 'Mul': continue
            for x,y in [n['inputs'], n['inputs'][::-1]]:
                gate = producers.get(y)
                if gate and gate['op'] in ('HardSigmoid','Sigmoid') and gate['inputs']==[x] and uses[y]==1:
                    replacement = (n,[gate], 'HardSwish' if gate['op']=='HardSigmoid' else 'Swish',[x],gate['attrs'])
                    break
            if replacement: break
            # ((erf(x/sqrt(2)) + 1) * x) * 0.5
            if not scalar(n['inputs'][1],0.5): continue
            mul = producers.get(n['inputs'][0])
            if not mul or mul['op']!='Mul': continue
            add = producers.get(mul['inputs'][0]); x=mul['inputs'][1]
            if not add or add['op']!='Add':
                add=producers.get(mul['inputs'][1]); x=mul['inputs'][0]
            if not add or add['op']!='Add' or not scalar(add['inputs'][1],1): continue
            erf=producers.get(add['inputs'][0])
            div=producers.get(erf['inputs'][0]) if erf and erf['op']=='Erf' else None
            if div and div['op']=='Div' and div['inputs'][0]==x and scalar(div['inputs'][1],np.sqrt(2)):
                chain=[mul,add,erf,div]
                if all(uses[v['output']]==1 for v in chain): replacement=(n,chain,'GeluErf',[x],{})
            if replacement: break
        if replacement is None: break
        n,chain,op,ins,a = replacement
        # Place the replacement at the final expression; dead intermediates vanish.
        n.update(op=op,inputs=ins,attrs=a)
        nodes = [v for v in nodes if all(v is not c for c in chain)]
    # LayerNorm: centered variance, epsilon, sqrt/div, learned scale and shift.
    for final in list(nodes):
        if final['op']!='Add': continue
        p={n['output']:n for n in nodes}
        mul=p.get(final['inputs'][0])
        if not mul or mul['op']!='Mul' or final['inputs'][1] not in constants or mul['inputs'][1] not in constants: continue
        div=p.get(mul['inputs'][0])
        if not div or div['op']!='Div': continue
        sub=p.get(div['inputs'][0]); sqrt=p.get(div['inputs'][1])
        if not sub or sub['op']!='Sub' or not sqrt or sqrt['op']!='Sqrt': continue
        mean=p.get(sub['inputs'][1]); eps=p.get(sqrt['inputs'][0])
        if not mean or mean['op']!='ReduceMean' or mean['inputs']!=[sub['inputs'][0]] or mean['attrs'].get('axes')!=[-1]: continue
        if not eps or eps['op']!='Add' or eps['inputs'][1] not in constants: continue
        var=p.get(eps['inputs'][0]); power=p.get(var['inputs'][0]) if var else None
        if not var or var['op']!='ReduceMean' or var['attrs'].get('axes')!=[-1] or not power or power['op']!='Pow' or power['inputs'][0]!=sub['output'] or not scalar(power['inputs'][1],2): continue
        chain=[mul,div,sub,sqrt,mean,eps,var,power]
        allowed={v['output'] for v in chain}
        if any(x in allowed for n in nodes if n not in chain and n is not final for x in n['inputs']) or any(x in allowed for x in outputs): continue
        e=constants[eps['inputs'][1]]
        if e.size!=1 or not np.isfinite(e).all() or float(e.reshape(-1)[0])<=0: raise ValueError('invalid LN epsilon')
        final.update(op='LayerNorm',inputs=[sub['inputs'][0],mul['inputs'][1],final['inputs'][1]],attrs={'axis':-1,'epsilon':float(e.reshape(-1)[0])})
        nodes=[n for n in nodes if all(n is not c for c in chain)]
    # Remove shape calculations that were only consumed by eliminated expressions.
    needed=set(outputs); live=[]
    for n in reversed(nodes):
        if n['output'] in needed:
            live.append(n); needed.update(n['inputs'])
    nodes=list(reversed(live))
    for n in nodes:
        if n['op'] not in (UVDOC_OPS if kind=='uvdoc' else OPS): raise ValueError(f"unmatched node {n['source_name']}: {n['op']}")
        a=n['attrs']
        if kind=='uvdoc':
            if n['op']=='Resize' and (a.get('mode')!=b'linear' or a.get('coordinate_transformation_mode')!=b'align_corners'):
                raise ValueError('unmatched Resize semantics')
            if n['op']=='Pad' and a.get('mode')!=b'reflect': raise ValueError('unmatched Pad semantics')
            if n['op']=='GridSample' and (a.get('mode')!=b'bilinear' or a.get('padding_mode')!=b'zeros' or a.get('align_corners')!=1):
                raise ValueError('unmatched GridSample semantics')
        elif n['op']=='Resize' and (a.get('mode')!=b'nearest' or a.get('coordinate_transformation_mode')!=b'asymmetric' or a.get('nearest_mode')!=b'floor'):
            raise ValueError('unmatched Resize semantics')
    return nodes


def expand_patterns(nodes, constants):
    expanded=[]; constants=dict(constants)
    def add(op,ins,out,a=None): expanded.append({'op':op,'inputs':ins,'output':out,'attrs':a or {},'source_name':out+'.expanded'})
    for n in nodes:
        op=n['op'];x=n['inputs'][0];out=n['output'];prefix=out+'.pattern.'
        def const(name,v):
            k=prefix+'constant.'+name;constants[k]=np.asarray(v,np.float32);return k
        if op in ('HardSwish','Swish'):
            add('HardSigmoid' if op=='HardSwish' else 'Sigmoid',[x],prefix+'gate',n['attrs'])
            add('Mul',[x,prefix+'gate'],out)
        elif op=='GeluErf':
            add('Div',[x,const('sqrt2',np.sqrt(2))],prefix+'div')
            add('Erf',[prefix+'div'],prefix+'erf')
            add('Add',[prefix+'erf',const('one',1)],prefix+'add')
            add('Mul',[prefix+'add',x],prefix+'mul')
            add('Mul',[prefix+'mul',const('half',0.5)],out)
        elif op=='LayerNorm':
            a={'axes':[-1],'keepdims':1}
            add('ReduceMean',[x],prefix+'mean',a)
            add('Sub',[x,prefix+'mean'],prefix+'sub')
            add('Pow',[prefix+'sub',const('two',2)],prefix+'pow')
            add('ReduceMean',[prefix+'pow'],prefix+'var',a)
            add('Add',[prefix+'var',const('eps',n['attrs']['epsilon'])],prefix+'eps')
            add('Sqrt',[prefix+'eps'],prefix+'sqrt')
            add('Div',[prefix+'sub',prefix+'sqrt'],prefix+'div')
            add('Mul',[prefix+'div',n['inputs'][1]],prefix+'mul')
            add('Add',[prefix+'mul',n['inputs'][2]],out)
        else: expanded.append(n)
    return expanded,constants


def json_value(v):
    if isinstance(v,bytes): return v.decode('utf-8')
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,(list,tuple)): return [json_value(x) for x in v]
    if isinstance(v,dict): return {k:json_value(x) for k,x in v.items()}
    return v


def write_tensors(path, tensors):
    header,parts,offset={},[],0
    for name in sorted(tensors):
        a=np.ascontiguousarray(tensors[name],dtype='<f4'); data=a.tobytes()
        header[name]={'dtype':'F32','shape':list(tensors[name].shape),'data_offsets':[offset,offset+len(data)]}
        parts.append(data); offset+=len(data)
    h=json.dumps(header,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()
    h+=b' '*((-len(h))%8)
    path.write_bytes(struct.pack('<Q',len(h))+h+b''.join(parts))


def dictionary(path):
    # The pinned YAML has one scalar per line. This deliberately accepts only
    # that representation; a changed upstream YAML must fail instead of guessing.
    lines=path.read_text(encoding='utf-8').splitlines(); start=lines.index('  character_dict:')+1
    result=[]
    for line in lines[start:]:
        if not line.startswith('  - '): raise ValueError('unmatched dictionary YAML layout')
        s=line[4:]
        if s.startswith("'"):
            if not s.endswith("'"): raise ValueError('multiline YAML scalar')
            s=s[1:-1].replace("''", "'")
        elif s.startswith('"'): s=json.loads(s)
        result.append(s)
    if len(result)!=11945 or any(not x or '\n' in x or '\r' in x for x in result): raise ValueError('invalid dictionary')
    return '\n'.join(result)+'\n'


def verify_folding(model,nodes,constants,outputs,kind):
    import onnxruntime as ort
    if ort.__version__!='1.28.0': raise ValueError(f'expected onnxruntime 1.28.0, got {ort.__version__}')
    ns=[helper.make_node(n['op'],n['inputs'],[n['output']],name=n['source_name'],**n['attrs']) for n in nodes]
    # Names of graph outputs can change when Identity was folded.
    outs=[]
    for original,name in zip(model.graph.output,outputs):
        o=onnx.ValueInfoProto(); o.CopyFrom(original); o.name=name; outs.append(o)
    used={x for n in nodes for x in n['inputs'] if x in constants}
    initializers=[numpy_helper.from_array(constants[k],k) for k in sorted(used)]
    inputs=list(model.graph.input)
    if model.opset_import[0].version <= 8:
        inputs += [helper.make_tensor_value_info(i.name,i.data_type,i.dims) for i in initializers]
    graph=helper.make_graph(ns,'folded',inputs,outs,initializers)
    folded=helper.make_model(graph,opset_imports=list(model.opset_import)); folded.ir_version=model.ir_version
    opts=ort.SessionOptions(); opts.intra_op_num_threads=1; opts.log_severity_level=3
    source=ort.InferenceSession(model.SerializeToString(),opts,providers=['CPUExecutionProvider'])
    target=ort.InferenceSession(folded.SerializeToString(),opts,providers=['CPUExecutionProvider'])
    shape={'det':(1,3,64,96),'rec':(1,3,48,48),'uvdoc':(1,3,96,64)}[kind]
    # UVDoc reads RGB in [0, 1]; the other two read normalized values.
    low=0 if kind=='uvdoc' else -1
    x=np.random.default_rng(0).uniform(low,1,shape).astype(np.float32)
    name=model.graph.input[0].name
    a=source.run(None,{name:x})[0]; b=target.run(None,{name:x})[0]
    delta=float(np.max(np.abs(a-b))); print('folding ORT',ort.__version__,'max_abs',delta)
    if delta>1e-4: raise ValueError('folding parity failed')
    return delta


def run(source, output, kind, verify=False):
    if onnx.__version__!='1.23.2': raise ValueError(f'expected onnx 1.23.2, got {onnx.__version__}')
    data=source.read_bytes(); pin=PINS[kind]
    if sha(data)!=pin[2]: raise ValueError('source SHA256 does not match pinned ONNX')
    model=onnx.load_model_from_string(data)
    nodes,constants,outputs,bn=fold(model)
    parity=verify_folding(model,nodes,constants,outputs,kind) if verify else None
    nodes=match_patterns(nodes,constants,outputs,kind)
    if verify:
        expanded,expanded_constants=expand_patterns(nodes,constants)
        parity=max(parity,verify_folding(model,expanded,expanded_constants,outputs,kind))
    used={x for n in nodes for x in n['inputs'] if x in constants}
    rename={}; weights={}; ints={}
    # Prefer stable per-producing-op names for convolution and linear weights.
    counts=collections.Counter()
    for n in nodes:
        op=n['op']; index=counts[op]; counts[op]+=1; n['name']=f'{kind}.{op.lower()}{index:03d}'
        if op in ('Conv','ConvTranspose','MatMul','LayerNorm'):
            for j,x in enumerate(n['inputs'][1:]):
                if x in used and x not in rename and constants[x].dtype.kind=='f': rename[x]=n['name']+('.w' if j==0 else '.b')
    for x in sorted(used):
        a=constants[x]
        if a.dtype.kind=='f':
            name=rename.setdefault(x,f'{kind}.const{len(rename):03d}'); weights[name]=a.astype(np.float32)
        elif a.dtype.kind in 'iu':
            name=f'{kind}.shape{len(ints):03d}'; rename[x]=name; ints[name]={'shape':list(a.shape),'values':a.reshape(-1).tolist()}
        else: raise ValueError(f'unmatched constant dtype {a.dtype}')
    for n in nodes: n['inputs']=[rename.get(x,x) for x in n['inputs']]
    graph={'schema_version':VERSION,'importer_version':VERSION,'model':kind,'source_sha256':pin[2],
           'inputs':[{'name':i.name,'shape':[d.dim_value if d.HasField('dim_value') else None for d in i.type.tensor_type.shape.dim]} for i in model.graph.input],
           'outputs':outputs,'nodes':nodes,'integer_constants':ints,
           'tensors':{n:list(a.shape) for n,a in sorted(weights.items())}}
    output.mkdir(parents=True,exist_ok=True)
    write_tensors(output/'model.safetensors',weights)
    (output/'graph.json').write_text(json.dumps(json_value(graph),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    if kind=='rec':
        yaml_path=source.with_name('inference.yml')
        if sha(yaml_path.read_bytes())!=REC_YAML_SHA256: raise ValueError('dictionary YAML SHA256 mismatch')
        (output/'dict.txt').write_text(dictionary(yaml_path),encoding='utf-8')
    files={p.name:{'sha256':sha(p.read_bytes()),'size':p.stat().st_size} for p in sorted(output.iterdir()) if p.name in ('model.safetensors','graph.json','dict.txt')}
    print(json.dumps({'model':kind,'bn_folded':bn,'ops':dict(counts),'files':files,'folding_max_abs':parity},sort_keys=True,indent=2))
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=PINS,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();run(a.source,a.output,a.model,a.verify)
