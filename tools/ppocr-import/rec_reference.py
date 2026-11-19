#!/usr/bin/env python3
"""Independent cv2 crop + ORT 1.28.0 Flat OCR reference and crop oracle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


def crop(page,quad,min_width=16,max_width=1600):
    q=np.asarray(quad,np.float32)
    width=int(max(np.linalg.norm(q[0]-q[1]),np.linalg.norm(q[2]-q[3])))
    height=int(max(np.linalg.norm(q[0]-q[3]),np.linalg.norm(q[1]-q[2])))
    target=np.array([[0,0],[width,0],[width,height],[0,height]],np.float32)
    matrix=cv2.getPerspectiveTransform(q,target)
    warped=cv2.warpPerspective(page,matrix,(width,height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    if height/width>=1.5: warped=np.rot90(warped)
    h,w=warped.shape[:2]
    width=max(min_width,min(max_width,int(np.ceil(48*w/h))))
    resized=cv2.resize(warped,(width,48),interpolation=cv2.INTER_LINEAR)
    x=(resized.astype(np.float32)/np.float32(255)-np.float32(0.5))/np.float32(0.5)
    return warped,resized,x.transpose(2,0,1)[None].copy()


def decode(prob,dictionary):
    ids=prob.argmax(-1);scores=prob.max(-1);keep=(ids!=0)&np.r_[True,ids[1:]!=ids[:-1]]
    return ''.join(dictionary[i] for i in ids[keep]),float(scores[keep].mean()) if keep.any() else 0.


def main(a):
    if cv2.__version__!='5.0.0': raise ValueError(f'expected cv2 5.0.0, got {cv2.__version__}')
    if ort.__version__!='1.28.0': raise ValueError(f'expected ORT 1.28.0, got {ort.__version__}')
    spec=importlib.util.spec_from_file_location('det_reference',Path(__file__).with_name('det_reference.py'))
    det=importlib.util.module_from_spec(spec);spec.loader.exec_module(det)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;opts.log_severity_level=3
    data=a.model.read_bytes();assert hashlib.sha256(data).hexdigest()=='92f0b7785e64fc9090106a241cf4c1eb97472824558272751b88a2a4476d3a08'
    rec=ort.InferenceSession(data,opts,providers=['CPUExecutionProvider'])
    dictionary=['']+(a.assets/'rec/dict.txt').read_text().splitlines()+[' ']
    root=a.assets/'round3';root.mkdir(exist_ok=True)
    result={'onnxruntime':ort.__version__,'opencv':cv2.__version__,'rec_source_sha256':hashlib.sha256(data).hexdigest(),'pages':[]}
    for name in ['kd0092','kd0330','kd0444','kd0822']:
        directory=root/name;directory.mkdir(exist_ok=True)
        # These maps were produced by ORT 1.28.0 from the pinned detector.
        dd=a.assets/'det/round2'/name
        meta=json.loads((dd/'index.json').read_text());assert meta['onnxruntime']=='1.28.0';assert meta['source_sha256']=='193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8'
        page=cv2.imread(str(a.pages/f'{name}_200.png'));h,w=page.shape[:2]
        quads=det.boxes(np.load(dd/'output000.npy')[0,0],h,w)
        all_crops=[];lines=[]
        for i,q in enumerate(quads):
            warped,resized,x=crop(page,q)
            np.save(directory/f'crop{i:03d}.npy',x,allow_pickle=False)
            if i<4: (directory/f'warp{i:03d}.bgr').write_bytes(warped.tobytes())
            p=rec.run(None,{rec.get_inputs()[0].name:x})[0][0]
            text,score=decode(p,dictionary)
            info={'quad':q,'width':x.shape[3],'text':text,'score':score,'file':f'crop{i:03d}.npy','warped_width':warped.shape[1],'warped_height':warped.shape[0]}
            all_crops.append(info)
            if score>=0.5: lines.append({'quad':q,'text':text,'score':score})
        info={'img_id':name,'page_width':w,'page_height':h,'crops':all_crops,'lines':lines}
        (directory/'reference.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
        result['pages'].append({'img_id':name,'boxes':len(quads),'lines':len(lines)})
        print(name,'boxes',len(quads),'lines',len(lines),flush=True)
    # Exercise the optional padded path independently: insert an additive key
    # mask into each attention Softmax, preserving the original ONNX otherwise.
    import onnx
    model=onnx.load_model_from_string(data)
    model.graph.input.append(onnx.helper.make_tensor_value_info('ocr_key_mask',onnx.TensorProto.FLOAT,[None,1,1,None]))
    nodes=[]
    for node in model.graph.node:
        if node.op_type=='Softmax' and any(attr.name=='axis' and attr.i==3 for attr in node.attribute):
            masked=node.input[0]+'_ocr_masked'
            nodes.append(onnx.helper.make_node('Add',[node.input[0],'ocr_key_mask'],[masked],name=masked))
            node.input[0]=masked
        nodes.append(node)
    del model.graph.node[:];model.graph.node.extend(nodes)
    masked=ort.InferenceSession(model.SerializeToString(),opts,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(7);x=np.zeros((2,3,48,48),np.float32)
    for i,width in enumerate([16,32]):x[i,:,:,:width]=rng.uniform(-1,1,(3,48,width))
    mask=np.full((2,1,1,6),-np.inf,np.float32);mask[0,:,:,:2]=0;mask[1,:,:,:4]=0
    prob=masked.run(None,{'x':x,'ocr_key_mask':mask})[0]
    directory=root/'masked';directory.mkdir(exist_ok=True)
    np.save(directory/'input.npy',x,allow_pickle=False);np.save(directory/'output.npy',prob,allow_pickle=False)
    (directory/'index.json').write_text(json.dumps({'onnxruntime':ort.__version__,'widths':[16,32],'shape':list(x.shape),'description':'pinned graph plus additive key masks at both attention logits'})+'\n')
    # Slanted, out-of-image, narrow, vertical and exact 1.5-boundary quads.
    rng=np.random.default_rng(9);page=rng.integers(0,256,(47,63,3),dtype=np.uint8);(root/'crop-oracle.bgr').write_bytes(page.tobytes());cases=[]
    quads=[[[0,0],[20,0],[20,30],[0,30]],[[2,3],[60,5],[57,44],[1,46]], [[-5,-3],[33,1],[35,32],[-2,34]],[[20,0],[24,0],[24,46],[20,46]], [[1,1],[60,1],[60,3],[1,3]]]
    for _ in range(64):
        q=cv2.boxPoints(((float(rng.uniform(10,50)),float(rng.uniform(10,35))),(float(rng.uniform(3,40)),float(rng.uniform(3,30))),float(rng.uniform(-60,60))))
        q,_=det.rect(q);quads.append(q.tolist())
    for i,q in enumerate(quads):
        warped,resized,x=crop(page,q)
        np.save(root/f'oracle{i:03d}.npy',x,allow_pickle=False)
        (root/f'oracle{i:03d}.bgr').write_bytes(warped.tobytes())
        cases.append({'quad':q,'width':x.shape[3],'warped_width':warped.shape[1],'warped_height':warped.shape[0]})
    (root/'crop-oracle.json').write_text(json.dumps(cases)+'\n')
    (root/'reference-index.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--pages',type=Path,required=True);p.add_argument('--model',type=Path,required=True);main(p.parse_args())
