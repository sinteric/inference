# Third-party notices

| Component | Version / pin | Licence | Role |
|---|---|---|---|
| oneDNN | 3.13.4 `c531fe1a5cd631b7f60277cdf6675c588cadbc28` | Apache-2.0 | default GEMM / LN / softmax primitives |
| ONNX Runtime MLAS | 1.30.0 `f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7` | MIT | comparison GEMM (`MlasSgemmThreaded`) |
| Candle | 0.11.0 | Apache-2.0 / MIT | research-only `bench/vela-mlas --features bench-mlas`; absent from the library crates |
| HuggingFace tokenizers | 0.23 | Apache-2.0 | frontend |
| Vela 2.0 0.3B | `vllm-sr/Vela-2.0-0.3B@d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409` | Apache-2.0 (tokenizer: Gemma Terms) | weights, `vela2_inference.py` frontend |
| EmbeddingGemma 2 | `google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b` | Apache-2.0 | text-tower weights + Gemma tokenizer |
| PP-OCRv6_tiny_det, korean_PP-OCRv5_mobile_rec, UVDoc | `PaddlePaddle/*_onnx` (see `assets/ppocr.toml`) | Apache-2.0 | weights converted offline by `tools/ppocr-import`; not distributed here |
| PaddleOCR | upstream | Apache-2.0 | `models/ppocr` follows its DB post-processing (`box_score_fast`, unclip), CTC decode and reading order |
| OpenCV, Clipper 6.4.2 | upstream | see `crates/rt-image/NOTICE` (BSD-style, BSL-1.0) | adapted geometry and interpolation in `rt-image` |
| ONNX Runtime | 1.28.0 (`ort` crate, downloaded at build) | MIT | reference only, in the non-default `bench/ocr-ort` |

Vela 2.0's tokenizer carries the Gemma Terms of Use; see
`DISTRIBUTION_TERMS.md` on its Hub repo. Do not commit tokenizer.json.

No model weights or tokenizer files are committed to this repository. Each is
fetched from its upstream repo and stays under that repo's licence. Licences
were read from each Hub repo's card on 2026-11-19. Whoever redistributes
weights, including the files `tools/ppocr-import` converts, ships the
upstream licence with them, any upstream NOTICE file, and a note that the
files were changed (Apache-2.0 section 4).
