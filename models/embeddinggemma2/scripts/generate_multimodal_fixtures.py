#!/usr/bin/env python3
"""Generate EmbeddingGemma 2 image/audio parity fixtures vs Sentence Transformers.

Writes models/embeddinggemma2/fixtures/multimodal.json. Media are synthetic and
integer-defined (see `image_pixels` / `audio_samples`), so the Rust tests build
byte-identical inputs without committing media files. Per case:

  - input_ids (placeholder expansion check vs the Rust frontend)
  - preprocessing: resized size, soft-token count, FNV-1a 64 of the resized
    RGB bytes; audio frame counts and log-mel checksums
  - FP32 embedding through SentenceTransformer.encode (cosine gate)

With --debug-dir, also dumps pixel_values / input_features / soft tokens as
.npy for stage-by-stage debugging (not committed).

  pip install 'sentence-transformers>=6.1' 'transformers>=5.18' torch pillow
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Kept in sync with models/embeddinggemma2/tests/multimodal_media.rs.
IMAGES = {
    # (height, width, channels)
    "tex-300x200": (300, 200, 3),
    "tex-64x512": (64, 512, 3),
    "tex-4000x10": (4000, 10, 3),
    "tex-768x1024": (768, 1024, 3),
    "gray-150x150": (150, 150, 1),
}
AUDIO = {
    # samples at 16 kHz
    "chirp-1s": 16000,
    "chirp-odd": 37123,
    "chirp-35s": 560000,
}

CASES = [
    {"id": "img", "parts": [["image", "tex-300x200"]], "prompt": None},
    {"id": "img-query", "parts": [["image", "tex-300x200"]], "prompt": "query"},
    {"id": "img-wide-doc", "parts": [["image", "tex-64x512"]], "prompt": "document"},
    {"id": "img-thin", "parts": [["image", "tex-4000x10"]], "prompt": None},
    {"id": "img-large", "parts": [["image", "tex-768x1024"]], "prompt": None},
    {"id": "img-gray", "parts": [["image", "gray-150x150"]], "prompt": None},
    {"id": "text-img", "parts": [["text", "a photo of"], ["image", "tex-300x200"]], "prompt": None},
    {"id": "aud", "parts": [["audio", "chirp-1s"]], "prompt": None},
    {"id": "aud-odd-query", "parts": [["audio", "chirp-odd"]], "prompt": "query"},
    {"id": "aud-long", "parts": [["audio", "chirp-35s"]], "prompt": None},
    {"id": "text-aud", "parts": [["text", "recording of"], ["audio", "chirp-1s"]], "prompt": None},
    {
        "id": "text-img-aud",
        "parts": [["text", "slide and narration"], ["image", "tex-64x512"], ["audio", "chirp-odd"]],
        "prompt": "document",
    },
]

# get_aspect_ratio_preserving_size probes: (height, width, max_soft_tokens).
SIZE_PROBES = [
    (300, 200, 280), (64, 512, 280), (4000, 10, 280), (10, 4000, 280), (768, 1024, 280),
    (1, 1, 280), (48, 48, 70), (2000, 3000, 70), (2000, 3000, 140), (2000, 3000, 560),
    (2000, 3000, 1120), (100, 100, 1120), (5000, 7, 70), (17, 9000, 140), (1080, 1920, 280),
]

PROMPTS = {"query": "task: search result | query: ", "document": "title: none | text: "}


def image_pixels(h: int, w: int, c: int) -> np.ndarray:
    y = np.arange(h, dtype=np.int64)[:, None, None]
    x = np.arange(w, dtype=np.int64)[None, :, None]
    ch = np.arange(c, dtype=np.int64)[None, None, :]
    v = (x * 3 + y * 5 + ch * 85) ^ ((x * y) >> 3)
    v = v + ((x // 16 + y // 16) % 2) * 64
    return (v & 255).astype(np.uint8)


def audio_samples(n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float32)
    state = 12345
    for i in range(n):
        state = (state * 1103515245 + 12345) % (1 << 31)
        noise = ((state >> 16) & 0x7FFF) - 16384
        freq = 200 + (i // 1600) % 8 * 110  # stepped tones, 100 ms each
        saw = (i * freq * 65536 // 16000) % 65536 - 32768
        env = (i // 800) % 20  # 50 ms bursts ramping over 1 s
        s = (saw // 4 + noise // 16) * env // 20
        out[i] = np.float32(s) / np.float32(32768.0)
    return out


def fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def sig(x: float, digits: int = 7) -> float:
    """Round to `digits` significant digits; 7 is far below the cosine gates."""
    return float(f"%.{digits}g" % x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "fixtures" / "multimodal.json"))
    ap.add_argument("--debug-dir")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

    torch.manual_seed(0)
    m = SentenceTransformer(args.model_dir, device="cpu", model_kwargs={"dtype": torch.float32})
    model = m[0].model
    seen: dict = {}

    forward = model.forward

    # ST calls `model.forward` directly (modality_config), so module hooks never fire.
    def spy(*a, **kwargs):
        out = forward(*a, **kwargs)
        seen["inputs"] = {k: v.detach().clone() for k, v in kwargs.items() if torch.is_tensor(v)}
        seen["image"] = None if out.image_hidden_states is None else out.image_hidden_states.detach().clone()
        seen["audio"] = None if out.audio_hidden_states is None else out.audio_hidden_states.detach().clone()
        return out

    model.forward = spy

    def pil(name: str):
        h, w, c = IMAGES[name]
        px = image_pixels(h, w, c)
        return Image.fromarray(px[:, :, 0], "L") if c == 1 else Image.fromarray(px, "RGB")

    dbg = Path(args.debug_dir) if args.debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)

    out_cases = []
    for case in CASES:
        parts = case["parts"]
        if len(parts) == 1:
            kind, name = parts[0]
            inp = pil(name) if kind == "image" else {"array": audio_samples(AUDIO[name]), "sampling_rate": 16000}
            if kind == "audio":
                inp = {"audio": inp}
        else:
            inp = {}
            for kind, val in parts:
                if kind == "text":
                    inp["text"] = val
                elif kind == "image":
                    inp["image"] = pil(val)
                else:
                    inp["audio"] = {"array": audio_samples(AUDIO[val]), "sampling_rate": 16000}
        kw = {"prompt_name": case["prompt"]} if case["prompt"] else {}
        seen.clear()
        emb = m.encode([inp], convert_to_numpy=True, normalize_embeddings=False, **kw)[0]
        ins = seen["inputs"]
        row = dict(case)
        row["prefix"] = PROMPTS.get(case["prompt"], "")
        row["input_ids"] = ins["input_ids"][0].tolist()
        if "pixel_values" in ins:
            pv = ins["pixel_values"][0]
            pos = ins["image_position_ids"][0]
            real = (pos != -1).all(dim=-1)
            n = int(real.sum())
            pw = int(pos[real][:, 0].max()) + 1
            ph = n // pw
            # Patches are (py, px, c) within a 16x16 patch; rebuild the resized RGB image.
            u8 = torch.round(pv[:n] * 255.0).to(torch.uint8).reshape(ph, pw, 16, 16, 3)
            img = u8.permute(0, 2, 1, 3, 4).reshape(ph * 16, pw * 16, 3).contiguous().numpy()
            row["image"] = {
                "resized_hw": [ph * 16, pw * 16],
                "patches": n,
                "soft_tokens": int(seen["image"].shape[0]),
                "resized_fnv1a64": fnv1a64(img.tobytes()),
            }
            if dbg:
                np.save(dbg / f"{case['id']}.pixel_values.npy", pv[:n].numpy())
                np.save(dbg / f"{case['id']}.positions.npy", pos[:n].numpy())
                np.save(dbg / f"{case['id']}.image_soft.npy", seen["image"].numpy())
        if "input_features" in ins:
            feats = ins["input_features"][0].double()
            mask = ins["input_features_mask"][0].bool()
            valid = feats[mask]
            row["audio"] = {
                "frames": int(feats.shape[0]),
                "valid_frames": int(mask.sum()),
                "soft_tokens": int(seen["audio"].shape[0]),
                "feature_sum": sig(float(valid.sum()), 10),
                "feature_abs_sum": sig(float(valid.abs().sum()), 10),
                "first_frame": [sig(float(v)) for v in valid[0]],
                "last_frame": [sig(float(v)) for v in valid[-1]],
            }
            if dbg:
                np.save(dbg / f"{case['id']}.input_features.npy", valid.float().numpy())
                np.save(dbg / f"{case['id']}.audio_soft.npy", seen["audio"].numpy())
        row["embedding_fp32"] = [sig(float(v)) for v in emb]
        out_cases.append(row)
        print(case["id"], len(row["input_ids"]), row.get("image", {}).get("soft_tokens"), row.get("audio", {}).get("soft_tokens"), flush=True)

    probes = []
    for h, w, s in SIZE_PROBES:
        th, tw = get_aspect_ratio_preserving_size(height=h, width=w, patch_size=16, max_patches=s * 9, pooling_kernel_size=3)
        probes.append({"h": h, "w": w, "max_soft_tokens": s, "target_hw": [int(th), int(tw)]})

    doc = {
        "model": "google/embeddinggemma-2",
        "reference": "SentenceTransformer.encode, dtype float32, CPU",
        "notes": "media from image_pixels/audio_samples; cosine gate vs embedding_fp32",
        "images": {k: list(v) for k, v in IMAGES.items()},
        "audio": AUDIO,
        "size_probes": probes,
        "cases": out_cases,
    }
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
