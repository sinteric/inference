# PP-OCR import and ORT fixtures

Offline tooling for the three **exact** ONNX pins in `assets/ppocr.toml`.
Use Python with **onnx 1.23.2**, numpy, **onnxruntime 1.28.0** and **opencv-python-headless 5.0.0.93**
(verify the installed versions before qualification). No Python
safetensors or YAML dependency is needed. This is build/test tooling, not a
library runtime dependency.

```bash
python import.py --model det --source /models/PP-OCRv6_tiny_det_onnx/inference.onnx \
  --output /assets/ppocr/det --verify
python import.py --model rec --source /models/korean_PP-OCRv5_mobile_rec_onnx/inference.onnx \
  --output /assets/ppocr/rec --verify
python import.py --model uvdoc --source /models/UVDoc_onnx/inference.onnx \
  --output /assets/ppocr/uvdoc --verify
python -m unittest discover -s . -p 'test_*.py'
```

`--source` must match the embedded source SHA256. The importer folds constant
expressions and Identity aliases, then Conv+BN and constant channel affine
operations. It matches closed Erf-GELU, sigmoid/hard-sigmoid multiplication,
and centered-variance LayerNorm expressions. External consumers prevent an
expression from being fused. Dynamic shape/view operations describe the
compiled SVTR graph. An unmatched operator or Resize convention fails. Each
model has its own operator list: PReLU, reflect Pad and GridSample are accepted
only for UVDoc, whose Resize must be linear align-corners, Pad reflect and
GridSample bilinear with zeros padding and align-corners.

Outputs have stable per-op names, sorted tensors/JSON keys and a canonical
little-endian FP32 safetensors encoding. Integer shape constants stay in JSON;
empty ROI constants remain valid zero-length tensors. `dict.txt` contains the
11,945 YAML entries in order, without blank or appended space. Only the
single-line YAML scalar syntax used by the pinned file is accepted.

`--verify` reconstructs folded and normalized expressions as ONNX and compares
with the original export using ORT CPU (one thread, max absolute error ≤1e-4).
`qualify.py` repeats import into a temporary directory under the supplied asset
root, requires identical output bytes, compares more shapes and writes ORT
fixtures plus `round1-parity.json`:

```bash
python qualify.py --models-root /models --assets /assets/ppocr --pages /pages
```

Cases: det N=1/2 with rectangular inputs and four full pages (when supplied);
rec widths 16/48/320/1600 and N=2. Gates also require identical rec argmax and
≤1e-5 detector threshold-map mismatch. These qualify **import transformations**;
they do not claim Rust detector/recognizer or pipeline parity (rounds 2/3).

Dump arbitrary ORT tensors separately:

```bash
python ort_dump.py --model /models/det/inference.onnx --shape 1 3 64 96 --output /fixtures/small
python ort_dump.py --model /models/det/inference.onnx --input /fixtures/input.npy \
  --tensor p2o.pd_op.relu.0.0 --output /fixtures/layer
python ort_dump.py --model /models/det/inference.onnx --page /pages/kd0092_200.png \
  --output /fixtures/page
```

The dumper writes `input.npy`, numbered outputs and a tensor-name index. Page
preprocessing is BGR, antialiased Triangle resize to nearest multiple-of-32
sides, ImageNet normalization, no side cap. This Python preprocessor is a
reference fixture producer; parity against Rust image operations is round 2.
The linear oneDNN resampler uses half-pixel centers, so UVDoc's align-corners
resizes run on the host in `rt-image`.

Rust asset tests read `SINTERIC_OCR_ASSETS` explicitly through
`rt_core::env::TestOverrides`. Provision `det/`, `rec/` and `uvdoc/` under that root;
no HOME or temporary-path fallback is used. Graph/dictionary hashes and all
weight names/shapes/dtypes are checked on load; payload hashing is opt-in
(`AssetConfig.verify_weights`). `graph.json` is a pinned contract, not an
executable graph interpreter. No packed disk cache is written.

Round 2 corrected the reference pin to **onnxruntime==1.28.0**, matching the
native library distributed by `ort` 2.0.0-rc.13. All round-1 fixtures and import
qualification have been regenerated with that version; the earlier 1.31 run is
historical only. Install the explicit pins in `requirements.txt`.

Round-2 detector qualification (assets stay outside git):

```bash
python det_reference.py --assets /assets/ppocr --pages /pages \
  --model /models/PP-OCRv6_tiny_det_onnx/inference.onnx
SINTERIC_OCR_ASSETS=/assets/ppocr cargo test -j6 -p ppocr --test detector --release -- --nocapture
```

This uses `ort_dump.py` for all eight detector cases (four full pages, small
rectangular shapes, N=2 and 2400x1696). It saves decoded BGR bytes, the Python
cv2 RETR_LIST/CHAIN_APPROX_SIMPLE + pyclipper DB reference, and random geometry
oracles under `det/round2`. The Rust tests check the preprocessing bytes/tensors,
probability map gates, one-to-one polygon IoU, and bitwise 1/2/4-thread identity.
The reference uses thresh=0.2, box_thresh=0.45, unclip=1.4, candidates=1000,
minimum sides 3/5, clipped page coordinates and Paddle reading order.
`--features debug-dumps` enables an ignored `dump_means` integration test and
`Detector::debug_prefix` for intermediate diagnostics against `ort_dump.py`.

Round-3 crop and Flat reference (first generate the round-2 detector fixtures):

```bash
python rec_reference.py --assets /assets/ppocr --pages /pages \
  --model /models/korean_PP-OCRv5_mobile_rec_onnx/inference.onnx
SINTERIC_OCR_ASSETS=/assets/ppocr cargo test -j6 -p ppocr --release -- --test-threads=2
```

This reuses the pinned detector's ORT 1.28 maps, independently extracts DB boxes
and full-resolution cv2 crops, and executes the original rec ONNX with ORT 1.28.
It saves each normalized crop and reference line under `round3/`, plus 69
synthetic crop oracles and a two-crop attention-mask ONNX reference. OpenCV 5.0
uses continuous cubic warp sampling; the reference rejects other cv2 versions.
Rust checks all crop pixels, rec probabilities/argmax/thread identity, masked
attention, CTC, four-page Flat text/score/box parity and CLI resume/warmup/error
semantics. The per-page comparison is written next to the asset root at
`parity/phase-a-e2e.json`. PNG decoding belongs only to the optional CLI feature;
none of these files or Python/codec dependencies are required by the library.

UVDoc reference (after importing `uvdoc/`):

```bash
python uvdoc_reference.py --model /models/UVDoc_onnx/inference.onnx \
  --pages /pages/page.png --warp --output /assets/ppocr/uvdoc/reference
SINTERIC_OCR_ASSETS=/assets/ppocr cargo test -p ppocr --release --test uvdoc -- --nocapture
```

The script runs the original export on ORT 1.28 with its grid exposed as an
extra output and saves, per case, the unit-range RGB input, the 45x31 grid and
the rectified page: one seeded random 96x64 input, each page as given and,
with `--warp`, the page curled by a fixed sinusoid. The curl is a stand-in for
a photographed book page, not a measurement of one. The Rust test checks the
grid (max abs ≤1e-4), the output (mean abs ≤1e-4), that `unwarp` from 8-bit
pixels equals the packed forward output, and bitwise identity at 1/2/4 threads.
