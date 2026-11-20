# Third-party notices

| Component | Version / pin | Licence | Role |
|---|---|---|---|
| oneDNN | 3.13.4 `c531fe1a5cd631b7f60277cdf6675c588cadbc28` | Apache-2.0 | default GEMM / LN / softmax primitives |
| HuggingFace tokenizers | 0.23 | Apache-2.0 | frontend |
| Prompt-injection guard | `Horizon-Labs/prompt-injection-guard-base` (see `assets/guard.toml`) | Apache-2.0 | weights + tokenizer for `guard` |
| EmbeddingGemma 2 | `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b` | Apache-2.0 | text-tower weights + Gemma tokenizer |
| PP-OCRv6_tiny_det, korean_PP-OCRv5_mobile_rec, UVDoc | `PaddlePaddle/*_onnx` (see `assets/ppocr.toml`) | Apache-2.0 | weights converted offline by `tools/ppocr-import`; not distributed here |
| PaddleOCR | upstream | Apache-2.0 | `ppocr` follows its DB post-processing (`box_score_fast`, unclip), CTC decode and reading order |
| OpenCV, Clipper 6.4.2 | upstream | see `crates/sinteric-inference/src/image/NOTICE` (BSD-style, BSL-1.0) | adapted geometry and interpolation in the library's image module |
| ONNX Runtime | 1.28.0 (`ort` crate, downloaded at build) | MIT | the OCR gate's reference only: `ppocr-ort` in `tools/drivers`, behind its non-default `ort` feature |

No model weights or tokenizer files are committed to this repository. Each is
fetched from its upstream repo and stays under that repo's licence. Licences
were read from each Hub repo's card on 2026-11-19. Whoever redistributes
weights, including the files `tools/ppocr-import` converts, ships the
upstream licence with them, any upstream NOTICE file, and a note that the
files were changed (Apache-2.0 section 4).
