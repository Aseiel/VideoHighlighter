"""Save the CLIP prefilter model as fp16 torch weights for bundling.

The mac build's counterpart of ``tools/export_clip_ov.py``. On a Mac, CLIP runs
through torch, which reaches the Apple GPU through Metal; OpenVINO cannot reach
it. So the mac build ships these weights instead of the IR, and the processor
route is torch's too (llm/clip_prefilter.py, BUNDLED_TORCH_DIRNAME).

    python -m tools.export_clip_torch

fp16 halves the file (~300 MB rather than ~600 MB). The GPU runs fp16 anyway;
the processor widens the weights to fp32 when it loads them, and CLIP's
embeddings agree with fp32 to 0.99999 (the index stores fp16 either way).

Idempotent: skips if the weights are already there (use --force to redo).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.clip_prefilter import BUNDLED_TORCH_DIRNAME, MODEL_ID  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Save CLIP as fp16 torch weights for bundling")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out", default=os.path.join("models", BUNDLED_TORCH_DIRNAME))
    ap.add_argument("--force", action="store_true", help="re-save even if present")
    args = ap.parse_args()

    weights = os.path.join(args.out, "model.safetensors")
    if os.path.isfile(weights) and not args.force:
        print(f"[export_clip_torch] weights already present, skipping: {weights}")
        return 0

    import torch
    from transformers import CLIPModel, CLIPProcessor

    print(f"[export_clip_torch] {args.model_id} -> {args.out} (fp16 safetensors)")
    os.makedirs(args.out, exist_ok=True)
    model = CLIPModel.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model.save_pretrained(args.out, safe_serialization=True)
    CLIPProcessor.from_pretrained(args.model_id).save_pretrained(args.out)

    # Prove the saved copy is the model: it must load and answer like the
    # original does, on the processor, in fp32.
    ref = CLIPModel.from_pretrained(args.model_id, torch_dtype=torch.float32).eval()
    got = CLIPModel.from_pretrained(args.out, torch_dtype=torch.float32).eval()
    pixels = torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    def embed(m):
        # image_embeds as the app's forward pass makes them, spelled out:
        # get_image_features changed its return type across transformers versions.
        return m.visual_projection(m.vision_model(pixel_values=pixels).pooler_output)[0]

    with torch.inference_mode():
        a, b = embed(ref), embed(got)
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print(f"[export_clip_torch] saved copy vs original: cosine {cos:.6f}")
    if cos < 0.999:
        print("[export_clip_torch] the saved weights do not reproduce the model")
        return 1

    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"[export_clip_torch] done, {size / 2**20:.0f} MB in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
