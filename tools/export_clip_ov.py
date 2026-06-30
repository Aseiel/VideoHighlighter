"""Pre-convert the CLIP prefilter model to OpenVINO IR for bundling.

Run this once at build time (CI does it before PyInstaller). It exports both the
model and its processor into ``models/<BUNDLED_OV_DIRNAME>/`` so the packaged app
can load with ``export=False`` — avoiding the runtime torch source introspection
that fails in a frozen exe with "could not get source code".

    python -m tools.export_clip_ov

Idempotent: skips the conversion if a valid IR already exists (use --force to
re-export).
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running both as `python -m tools.export_clip_ov` and `python tools/export_clip_ov.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.clip_prefilter import MODEL_ID, BUNDLED_OV_DIRNAME  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Export CLIP to OpenVINO IR for bundling")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out", default=os.path.join("models", BUNDLED_OV_DIRNAME))
    ap.add_argument("--force", action="store_true", help="re-export even if IR exists")
    args = ap.parse_args()

    xml = os.path.join(args.out, "openvino_model.xml")
    if os.path.isfile(xml) and not args.force:
        print(f"✅ IR already present, skipping: {xml}")
        return 0

    from optimum.intel import OVModelForZeroShotImageClassification
    from transformers import CLIPProcessor

    print(f"⏳ Exporting {args.model_id} → {args.out} (OpenVINO IR)...")
    os.makedirs(args.out, exist_ok=True)
    model = OVModelForZeroShotImageClassification.from_pretrained(
        args.model_id, export=True,
    )
    model.save_pretrained(args.out)
    # Save the processor next to the model so the app never reaches the network.
    CLIPProcessor.from_pretrained(args.model_id).save_pretrained(args.out)

    assert os.path.isfile(xml), f"export produced no {xml}"
    print(f"✅ Done: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
