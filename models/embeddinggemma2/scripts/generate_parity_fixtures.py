#!/usr/bin/env python3
"""Generate tiny EmbeddingGemma 2 parity fixtures vs the PyTorch text tower.

Writes models/embeddinggemma2/fixtures/parity.json with:
  - token ids (tokenizer check vs the Rust frontend)
  - FP32 embeddings (cosine >= 0.999 gate)
  - BF16 embeddings (max abs diff, reported)

Requires a local copy of google/embeddinggemma-2 (see assets/embeddinggemma2.toml)
and:

  pip install torch
  pip install 'transformers>=5.18'  # or git+https://github.com/huggingface/transformers

Text-only: vision_config and audio_config are cleared so the 170M+300M towers
are not constructed. Mean pooling includes prompt tokens; padding is excluded.
MRL is not a model module (post-hoc slice + re-L2) and is not in these fixtures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = [
    {"id": "q-en-short", "prompt": "query", "text": "oldest palace in Seoul"},
    {"id": "d-en-short", "prompt": "document", "text": "Gyeongbokgung is a palace in Seoul."},
    {
        "id": "q-ko-short",
        "prompt": "query",
        "text": "서울에서 가장 오래된 궁궐은?",
    },
    {
        "id": "d-ko-short",
        "prompt": "document",
        "text": "경복궁은 서울에 있는 조선의 법궁이다.",
    },
    {"id": "q-code", "prompt": "query", "text": "how to mmap safetensors in rust"},
]

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "


def prefixes(kind: str, text: str) -> str:
    return (QUERY_PREFIX if kind == "query" else DOC_PREFIX) + text


def mean_pool_l2(last_hidden, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, p=2, dim=-1)


def embed_dtype(model_dir: Path, dtype, device: str, pairs: list[tuple[str, str]], tokenizer):
    import torch
    from transformers import AutoConfig, AutoModel

    if hasattr(torch, "accelerator"):
        torch.accelerator.current_accelerator = lambda: None  # type: ignore[method-assign]

    torch.set_num_threads(4)
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg.vision_config = None
    cfg.audio_config = None
    inner = AutoModel.from_pretrained(
        model_dir,
        config=cfg,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    inner.eval()
    inner.to(device)
    out = []
    with torch.inference_mode():
        for kind, text in pairs:
            encoded = tokenizer(
                prefixes(kind, text),
                add_special_tokens=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = inner(**encoded).last_hidden_state
            vec = mean_pool_l2(hidden, encoded["attention_mask"])[0]
            out.append(vec.detach().to(torch.float64).cpu().tolist())
    del inner
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local snapshot of google/embeddinggemma-2")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "fixtures" / "parity.json"),
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    model_dir = Path(args.model)

    import torch

    # CPU-only torch 2.6 raises in transformers 5.19's flex-attn probe.
    if hasattr(torch, "accelerator"):
        torch.accelerator.current_accelerator = lambda: None  # type: ignore[method-assign]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def encode_ids(kind: str, text: str) -> list[int]:
        return tokenizer(
            prefixes(kind, text),
            add_special_tokens=True,
            truncation=True,
            max_length=8192,
        )["input_ids"]

    pairs = [(c["prompt"], c["text"]) for c in CASES]
    print("encoding FP32…", flush=True)
    fp32 = embed_dtype(model_dir, torch.float32, args.device, pairs, tokenizer)
    print("encoding BF16…", flush=True)
    bf16 = embed_dtype(model_dir, torch.bfloat16, args.device, pairs, tokenizer)

    cases = []
    for c, e32, e16 in zip(CASES, fp32, bf16):
        cases.append(
            {
                **c,
                "prefixed": prefixes(c["prompt"], c["text"]),
                "input_ids": encode_ids(c["prompt"], c["text"]),
                "embedding_fp32": e32,
                "embedding_bf16": e16,
            }
        )
    payload = {
        "model": "google/embeddinggemma-2",
        "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "pooling": "mean_include_prompt",
        "normalize": "l2",
        "notes": "cosine gate vs embedding_fp32 >= 0.999; report max abs vs embedding_bf16",
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out, "cases", len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
