#!/usr/bin/env python3
"""Write reproducible ONNX Runtime input/intermediate/output .npy fixtures."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort


def triangle_resize(image, height, width):
    """Separable triangle filter with antialiasing for downsampling, BGR f32."""
    out=image.astype(np.float32)
    if out.shape[:2] == (height,width): return image.copy()
    for axis,size in [(0,height),(1,width)]:
        original=out.shape[axis]; ratio=np.float32(original)/np.float32(size); support=max(ratio,np.float32(1))
        shape=list(out.shape); shape[axis]=size; target=np.empty(shape,np.float32)
        for i in range(size):
            center=np.float32(np.float32(i)+np.float32(0.5))*ratio
            left=max(0,min(original-1,int(np.floor(center-support)))); right=max(left+1,min(original,int(np.ceil(center+support))))
            pos=np.float32(center-np.float32(0.5))
            weights=np.maximum(np.float32(0),np.float32(1)-np.abs((np.arange(left,right,dtype=np.float32)-pos)/support))
            total=np.float32(0)
            for weight in weights: total=np.float32(total+weight)
            weights=weights/total
            row=np.zeros(np.take(out,0,axis=axis).shape,np.float32)
            for j,weight in enumerate(weights): row += np.take(out,left+j,axis=axis)*weight
            index=[slice(None)]*out.ndim; index[axis]=i; target[tuple(index)]=row
        out=target
    return np.floor(out+np.float32(0.5)).clip(0,255).astype(np.uint8)


def page_tensor(path):
    page=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if page is None: raise ValueError(f'cannot decode {path}')
    h,w=page.shape[:2]
    # Positive dimensions; ties use Python/Paddle round-to-even.
    h=max(32,round(h/32)*32);w=max(32,round(w/32)*32)
    image=triangle_resize(page,h,w).astype(np.float32)/255
    image=(image-np.array([0.485,0.456,0.406],np.float32))/np.array([0.229,0.224,0.225],np.float32)
    return image.transpose(2,0,1)[None].copy()


def dump(model,output,input_path=None,shape=None,page=None,tensors=()):
    if ort.__version__!='1.28.0': raise ValueError(f'expected onnxruntime 1.28.0, got {ort.__version__}')
    data=model.read_bytes();m=onnx.load_model_from_string(data)
    if input_path is not None: x=np.load(input_path,allow_pickle=False)
    elif page is not None: x=page_tensor(page)
    else:
        if not shape or len(shape)!=4 or any(d<=0 for d in shape): raise ValueError('expected positive NCHW shape')
        x=np.random.default_rng(0).uniform(-1,1,shape).astype(np.float32)
    if x.dtype!=np.float32 or x.ndim!=4: raise ValueError('input must be NCHW float32')
    if tensors:
        m=onnx.shape_inference.infer_shapes(m)
        values={v.name:v for v in list(m.graph.value_info)+list(m.graph.output)+list(m.graph.input)}
        for name in tensors:
            if name not in values: raise ValueError(f'unknown tensor {name}')
            if all(o.name!=name for o in m.graph.output): m.graph.output.append(values[name])
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.log_severity_level=3
    session=ort.InferenceSession(m.SerializeToString(),options,providers=['CPUExecutionProvider'])
    results=session.run(None,{session.get_inputs()[0].name:x})
    output.mkdir(parents=True,exist_ok=True);np.save(output/'input.npy',x,allow_pickle=False)
    index={'source_sha256':hashlib.sha256(data).hexdigest(),'onnxruntime':ort.__version__,
           'preprocess':'BGR Triangle uint8 ImageNet; no side cap' if page else 'supplied NCHW or seed=0 uniform(-1,1)',
           'input':{'file':'input.npy','shape':list(x.shape)},'outputs':[]}
    for i,(meta,y) in enumerate(zip(session.get_outputs(),results)):
        name=f'output{i:03d}.npy';np.save(output/name,y,allow_pickle=False)
        index['outputs'].append({'name':meta.name,'file':name,'shape':list(y.shape)})
    (output/'index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
    print(json.dumps(index,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--input',type=Path);group.add_argument('--shape',type=int,nargs=4);group.add_argument('--page',type=Path)
    p.add_argument('--tensor',action='append',default=[]);a=p.parse_args();dump(a.model,a.output,a.input,a.shape,a.page,a.tensor)
