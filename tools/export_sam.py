"""Export the SAM outliner (SlimSAM-77, Apache-2.0) for OpenVINO.

    python -m tools.export_sam [--out models/slimsam-uniform-77] [--force]

Writes the files ``modules/vision/outlines.py`` (``OpenVinoSamOutliner``)
loads, and nothing else:

  encoder.onnx   pixel_values [1, 3, 1024, 1024] -> image_embeddings
                 [1, 256, 64, 64]. Once per frame.
  decoder.onnx   image_embeddings + boxes [N, 4] (pixels of the resized
                 frame) -> masks [N, 256, 256] (logits), iou_scores [N].
                 Prompt encoder and mask decoder, every box in one call.
  outliner.json  the export id, source revision, and the outline PyTorch
                 draws for ``outlines.sam_probe()``, which every route must
                 reproduce before the app uses it.

Both files keep their large weights as fp16 with a Cast back to fp32
(``modules/vision/onnx_weights.py``), so they are half the size and compute
as fp32. Then checks, in a fresh process with neither torch nor transformers
imported, that OpenVINO reproduces the probe on the CPU and on an Intel GPU if
there is one, and fails if not.

The point is load time: importing ``transformers`` took ~22 s, which every
first outline in a session waited for; this loads in a second or two.

Needs torch, transformers and onnx (build time only; the app needs none of
them for this). Idempotent: skips when the files exist, unless --force.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vision import outlines as ol  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "models", ol.SAM_DIRNAME)
FILES = (ol.SAM_ENCODER_FILE, ol.SAM_DECODER_FILE, ol.SAM_META_FILE)


def reference_outline(model, processor) -> list:
    """How PyTorch, through transformers' own pre- and post-processing,
    outlines the probe: the outline every exported route must reproduce."""
    import cv2
    import torch
    from PIL import Image

    frame, box = ol.sam_probe()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, input_boxes=[[list(box)]], return_tensors="pt")
    inputs = {k: (v.float() if k not in ("original_sizes", "reshaped_input_sizes")
                  and v.is_floating_point() else v) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs, multimask_output=False)
    mask = processor.image_processor.post_process_masks(
        out.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"])[0][0, 0]
    h, w = frame.shape[:2]
    return ol.mask_to_contour(mask.numpy().astype("uint8"), frame_size=(w, h))


def export(out_dir: str) -> None:
    import onnx
    import torch
    import transformers
    from transformers import SamModel, SamProcessor

    from modules.vision.onnx_weights import store_weights_fp16

    model = SamModel.from_pretrained(ol.DEFAULT_SAM_MODEL, revision=ol.SAM_SOURCE_REVISION).eval()
    processor = SamProcessor.from_pretrained(ol.DEFAULT_SAM_MODEL, revision=ol.SAM_SOURCE_REVISION)

    class Encoder(torch.nn.Module):
        def __init__(self, sam):
            super().__init__()
            self.sam = sam

        def forward(self, pixel_values):
            return self.sam.get_image_embeddings(pixel_values)

    class Decoder(torch.nn.Module):
        def __init__(self, sam):
            super().__init__()
            self.sam = sam

        def forward(self, image_embeddings, boxes):
            out = self.sam(image_embeddings=image_embeddings, input_boxes=boxes[None],
                           multimask_output=False)
            return out.pred_masks[0, :, 0], out.iou_scores[0, :, 0]

    os.makedirs(out_dir, exist_ok=True)
    pixels = torch.zeros(1, 3, ol.SAM_INPUT, ol.SAM_INPUT)
    with torch.inference_mode():
        embeddings = Encoder(model)(pixels)
    for name, module, args, inputs, outputs, dynamic in (
            (ol.SAM_ENCODER_FILE, Encoder(model), (pixels,), ["pixel_values"],
             ["image_embeddings"], None),
            (ol.SAM_DECODER_FILE, Decoder(model), (embeddings, torch.tensor([[10.0, 10.0, 200.0, 200.0]])),
             ["image_embeddings", "boxes"], ["masks", "iou_scores"],
             {"boxes": {0: "n"}, "masks": {0: "n"}, "iou_scores": {0: "n"}})):
        fp32 = os.path.join(out_dir, name + ".fp32")
        torch.onnx.export(module.eval(), args, fp32, input_names=inputs, output_names=outputs,
                          dynamic_axes=dynamic, opset_version=17, dynamo=False)
        converted, n = store_weights_fp16(onnx.load(fp32))
        onnx.save(converted, os.path.join(out_dir, name))
        os.remove(fp32)
        print(f"[export_sam] {name}: {n} weight tensors stored as fp16, "
              f"{os.path.getsize(os.path.join(out_dir, name)) / 1e6:.1f} MB")

    probe = reference_outline(model, processor)
    if not probe:
        raise SystemExit("[export_sam] PyTorch found nothing in the probe; refusing to export")
    meta = {
        "id": ol.SAM_EXPORT_ID,
        "format": ol.SAM_FORMAT,
        "source": ol.DEFAULT_SAM_MODEL,
        "revision": ol.SAM_SOURCE_REVISION,
        "licence": "Apache-2.0",
        "input": ol.SAM_INPUT,
        "mask": ol.SAM_MASK,
        "probe_box": list(ol.sam_probe()[1]),
        "probe_outline": probe,
        "exported_with": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    with open(os.path.join(out_dir, ol.SAM_META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)


CHECK = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from modules.vision import outlines as ol
from modules.vision import frame_encoder as fe
devices = ["CPU"] + ([fe._openvino_gpu()[0]] if fe._openvino_gpu()[0] else [])
for device in devices:
    t = time.time()
    o = ol.OpenVinoSamOutliner(folder=sys.argv[2], device=device)
    o._load()                       # raises unless the probe is reproduced
    print(f"[export_sam] {o.device}: probe reproduced, loaded in {time.time() - t:.1f}s")
    if o.device != f"OpenVINO {device}":
        raise SystemExit(f"[export_sam] {device} fell back to {o.device}")
leaked = [m for m in ("torch", "transformers") if m in sys.modules]
if leaked:
    raise SystemExit(f"[export_sam] the OpenVINO route imported {leaked}")
"""


def check(out_dir: str) -> None:
    """Every OpenVINO device here reproduces the probe, in a process that
    imports neither torch nor transformers (the app's situation)."""
    done = subprocess.run([sys.executable, "-c", CHECK, ROOT, out_dir])
    if done.returncode:
        raise SystemExit("[export_sam] the export is not faithful on this machine")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.force and all(os.path.isfile(os.path.join(args.out, f)) for f in FILES):
        print(f"[export_sam] {args.out} already exported (use --force to redo)")
        return 0
    t = time.time()
    export(args.out)
    check(args.out)
    print(f"[export_sam] done in {time.time() - t:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
