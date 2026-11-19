#!/usr/bin/env python3
"""Round-2 independent cv2/pyclipper reference; ORT strictly 1.28.0."""
import argparse
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

spec=importlib.util.spec_from_file_location('ort_dump',Path(__file__).with_name('ort_dump.py'))
dump=importlib.util.module_from_spec(spec);spec.loader.exec_module(dump)

def rect(points):
    r=cv2.minAreaRect(np.array(points,dtype=np.float32).reshape(-1,2))
    p=sorted(cv2.boxPoints(r).tolist(),key=lambda a:(a[0],a[1]))
    left=sorted(p[:2],key=lambda a:a[1]);right=sorted(p[2:],key=lambda a:a[1])
    return np.array([left[0],right[0],right[1],left[1]],np.float32),min(r[1])

def score(prob,box):
    h,w=prob.shape
    xmin=int(np.clip(np.floor(box[:,0].min()),0,w-1));xmax=int(np.clip(np.ceil(box[:,0].max()),0,w-1))
    ymin=int(np.clip(np.floor(box[:,1].min()),0,h-1));ymax=int(np.clip(np.ceil(box[:,1].max()),0,h-1))
    p=box.copy();p[:,0]-=xmin;p[:,1]-=ymin
    mask=np.zeros((ymax-ymin+1,xmax-xmin+1),np.uint8);cv2.fillPoly(mask,[p.astype(np.int32)],1)
    return cv2.mean(prob[ymin:ymax+1,xmin:xmax+1],mask)[0]

def unclip(box):
    poly=Polygon(box)
    offset=pyclipper.PyclipperOffset();offset.AddPath(box,pyclipper.JT_ROUND,pyclipper.ET_CLOSEDPOLYGON)
    return offset.Execute(poly.area*1.4/poly.length)

def boxes(prob,page_h,page_w):
    h,w=prob.shape
    contours,_=cv2.findContours((prob>0.2).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for c in contours[:1000]:
        box,side=rect(c)
        if side<3 or score(prob,box)<0.45: continue
        expanded=unclip(box)
        if len(expanded)!=1: continue
        box,side=rect(expanded[0])
        if side<5: continue
        box[:,0]=np.clip(np.round(box[:,0]/w*page_w),0,page_w-1)
        box[:,1]=np.clip(np.round(box[:,1]/h*page_h),0,page_h-1)
        result.append(box.tolist())
    result.sort(key=lambda a:(a[0][1],a[0][0]))
    for i in range(1,len(result)):
        j=i
        while j>0 and abs(result[j][0][1]-result[j-1][0][1])<10 and result[j][0][0]<result[j-1][0][0]:
            result[j],result[j-1]=result[j-1],result[j];j-=1
    return result

def main(a):
    target=a.assets/'det'/'round2';target.mkdir(parents=True,exist_ok=True)
    for name in ['kd0092','kd0330','kd0444','kd0822']:
        page=a.pages/f'{name}_200.png';pixels=cv2.imread(str(page));h,w=pixels.shape[:2]
        directory=target/name
        dump.dump(a.model,directory,page=page)
        prob=np.load(directory/'output000.npy')[0,0]
        (directory/'page.bgr').write_bytes(pixels.tobytes())
        b=boxes(prob,h,w)
        (directory/'reference.json').write_text(json.dumps({'page_width':w,'page_height':h,'boxes':b},indent=2)+'\n')
        print(name,'boxes',len(b),flush=True)
    for shape in [(1,3,64,96),(1,3,96,64),(2,3,64,64),(1,3,2400,1696)]:
        dump.dump(a.model,target/'x'.join(map(str,shape)),shape=shape)
    rng=np.random.default_rng(42);cases=[]
    for _ in range(128):
        h,w=map(int,rng.integers(1,32,2));mask=(rng.random((h,w))>0.65).astype(np.uint8)
        cs,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cases.append({'width':w,'height':h,'mask':mask.ravel().tolist(),'contours':[c.reshape(-1,2).tolist() for c in cs]})
    rectangles=[]
    for _ in range(128):
        q=cv2.boxPoints(((float(rng.uniform(20,100)),float(rng.uniform(20,100))),(float(rng.uniform(5,80)),float(rng.uniform(5,20))),float(rng.uniform(-90,90))))
        q,_=rect(q);prob=rng.uniform(0,1,(160,160)).astype(np.float32)
        expanded=unclip(q)
        rectangles.append({'quad':q.tolist(),'expanded':expanded[0], 'prob':prob.ravel().tolist(),'score':score(prob,q),'rect':rect(expanded[0])[0].tolist()})
    (target/'geometry.json').write_text(json.dumps({'contours':cases,'rectangles':rectangles}))
    # Triangle reference patterns, including antialiasing and strided/color tests in Rust.
    resize=[]
    for sh,sw,h,w in [(1,1,7,3),(2,3,5,7),(7,13,3,5),(23,17,32,32),(33,65,32,64),(16,16,16,16)]:
        x=rng.integers(0,256,(sh,sw,3),dtype=np.uint8)
        resize.append({'width':sw,'height':sh,'output_width':w,'output_height':h,'input':x.ravel().tolist(),'output':dump.triangle_resize(x,h,w).ravel().tolist()})
    (target/'resize.json').write_text(json.dumps(resize))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True);p.add_argument('--pages',type=Path,required=True);p.add_argument('--model',type=Path,required=True);main(p.parse_args())
