#!/usr/bin/env python3
"""ORT 1.28 fixtures for the UVDoc unwarper: per case the unit-range RGB
input, the model's 45x31 grid and the rectified output.

Cases: one seeded random 96x64 input, then each supplied page as it is and,
with --warp, the same page curled by a fixed sinusoid (a stand-in for a
photographed book page, not a measurement of one).

    python uvdoc_reference.py --model /models/UVDoc_onnx/inference.onnx \
        --pages page.png --warp --output /assets/ppocr/uvdoc/reference
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

SOURCE_SHA256 = '54cab30dc2cf347b4f3d6971c833b1c3d84f5ef17280f1f6d15796bff20e63d6'
# The grid the point head answers, before the export resizes it.
GRID = 'p2o.pd_op.add.41.0'


def session(model_path):
    data = Path(model_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('UVDoc source SHA256 does not match the pin')
    if ort.__version__ != '1.28.0':
        raise ValueError(f'expected onnxruntime 1.28.0, got {ort.__version__}')
    model = onnx.load_model_from_string(data)
    grid = onnx.ValueInfoProto()
    grid.name = GRID
    model.graph.output.append(grid)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), opts, providers=['CPUExecutionProvider'])


def curl(rgb):
    import cv2
    h, w = rgb.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # A page bowed by its binding: rows sag toward the middle, columns lean.
    map_y = ys + 0.04 * h * np.sin(np.pi * xs / w)
    map_x = xs + 0.02 * w * np.sin(2 * np.pi * ys / h)
    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(255, 255, 255))


def write(out, name, sess, x):
    case = out / name
    case.mkdir(parents=True, exist_ok=True)
    y, grid = sess.run(None, {'image': x})
    np.save(case / 'input.npy', x)
    np.save(case / 'grid.npy', grid.astype(np.float32))
    np.save(case / 'output.npy', y.astype(np.float32))
    return {'name': name, 'shape': list(x.shape), 'grid_shape': list(grid.shape)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--pages', nargs='*', default=[])
    p.add_argument('--warp', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sess = session(a.model)
    cases = [write(a.output, 'random', sess,
                   np.random.default_rng(0).uniform(0, 1, (1, 3, 96, 64)).astype(np.float32))]
    import cv2
    for page in a.pages:
        rgb = cv2.cvtColor(cv2.imread(page, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        variants = [('flat', rgb)] + ([('curled', curl(rgb))] if a.warp else [])
        for kind, img in variants:
            x = (img.astype(np.float32) / 255.).transpose(2, 0, 1)[None].copy()
            cases.append(write(a.output, f'{Path(page).stem}-{kind}', sess, x))
    (a.output / 'cases.json').write_text(json.dumps(
        {'onnxruntime': ort.__version__, 'source_sha256': SOURCE_SHA256, 'cases': cases},
        indent=2, sort_keys=True) + '\n')
    print(json.dumps(cases, indent=2))


if __name__ == '__main__':
    main()
