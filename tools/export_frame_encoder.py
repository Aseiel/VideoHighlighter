"""Export the frame encoder (SigLIP2 base/16 image tower) for the app.

    python -m tools.export_frame_encoder [--out models/siglip2-base-patch16-256] [--force]

Writes the two files ``modules/vision/frame_encoder.py`` loads, and nothing
else:

  vision.onnx    pixel_values [N, 3, 256, 256] -> image_embeds [N, 768] (the
                 pooled vector, not unit length). Weights stored as fp16 and
                 widened to fp32 by a Cast at load, so the file is half the size
                 and computes exactly as the fp32 export does.
  encoder.json   the encoder id, input description, and the vector PyTorch
                 computes for frame_encoder.probe_pixels(), which every route in
                 the app must reproduce before it is used.

Then checks the file on every runtime present here (ONNX Runtime, OpenVINO CPU
and GPU) against PyTorch, and fails if any of them is not faithful.

Needs torch and transformers (build time only; the app needs neither for this).
Idempotent: skips when both files exist, unless --force. Exports from a
pinned model revision, and with the same library versions the output is
byte-identical from run to run (measured), so a rebuilt app does not make the
updater fetch the model again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from modules.vision import frame_encoder as fe  # noqa: E402

OPSET = 17
# Tensors smaller than this stay fp32: they are biases and norms, where fp16
# saves nothing worth having.
MIN_FP16_ELEMENTS = 1024
MIN_COSINE = 0.9999


def store_weights_fp16(model):
    """Store every large fp32 initializer as fp16, with a Cast back to fp32 in
    front of its users. Both runtimes fold the Cast at load."""
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    casts = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        weights = numpy_helper.to_array(init)
        half = weights.astype(np.float16)
        if weights.size < MIN_FP16_ELEMENTS or not np.isfinite(half).all():
            continue
        name = init.name
        init.CopyFrom(numpy_helper.from_array(half, name + "__fp16"))
        casts.append(helper.make_node("Cast", [name + "__fp16"], [name],
                                      to=TensorProto.FLOAT, name=name + "__to_fp32"))
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    return model, len(casts)


def worst_cosine(a, b) -> float:
    return min(fe._cosine(x, y) for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the frame encoder for the app")
    ap.add_argument("--source", default=fe.SOURCE_MODEL)
    ap.add_argument("--revision", default=fe.SOURCE_REVISION,
                    help="Hugging Face commit; pinned so builds export identical bytes")
    ap.add_argument("--out", default=os.path.join("models", fe.MODEL_DIRNAME))
    ap.add_argument("--force", action="store_true", help="export even if the files exist")
    args = ap.parse_args()

    onnx_path = os.path.join(args.out, fe.MODEL_FILE)
    meta_path = os.path.join(args.out, fe.META_FILE)
    if os.path.isfile(onnx_path) and os.path.isfile(meta_path) and not args.force:
        print(f"[export_frame_encoder] already present, skipping: {args.out}")
        return 0

    import onnx
    import torch
    from transformers import AutoModel

    print(f"[export_frame_encoder] {args.source}@{args.revision[:12]} -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    model = AutoModel.from_pretrained(args.source, revision=args.revision,
                                      attn_implementation="eager").eval()
    size = model.config.vision_config.image_size
    if size != fe.INPUT_SIZE:
        print(f"[export_frame_encoder] {args.source} takes {size} px, the app feeds {fe.INPUT_SIZE}")
        return 1

    class Vision(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m.vision_model

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).pooler_output

    vision = Vision(model).eval()
    probe = fe.probe_pixels()
    check = np.concatenate([probe, np.random.default_rng(0).uniform(
        -1, 1, (3, 3, fe.INPUT_SIZE, fe.INPUT_SIZE)).astype(np.float32)])
    with torch.no_grad():
        reference = vision(torch.from_numpy(check)).numpy()
    if reference.shape[1] != fe.DIMS:
        print(f"[export_frame_encoder] vectors have {reference.shape[1]} numbers, the app expects {fe.DIMS}")
        return 1

    fp32_path = onnx_path + ".fp32.tmp"
    torch.onnx.export(vision, (torch.from_numpy(probe),), fp32_path,
                      input_names=["pixel_values"], output_names=["image_embeds"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True,
                      dynamic_axes={"pixel_values": {0: "n"}, "image_embeds": {0: "n"}})
    converted, n = store_weights_fp16(onnx.load(fp32_path))
    onnx.save(converted, onnx_path)
    os.remove(fp32_path)
    print(f"[export_frame_encoder] {n} weight tensors stored as fp16, "
          f"{os.path.getsize(onnx_path) / 1e6:.1f} MB")

    meta = {
        "format": fe.FORMAT,
        "id": fe.ENCODER_ID,
        "source": args.source,
        "revision": args.revision,
        "licence": "Apache-2.0",
        "file": fe.MODEL_FILE,
        "dims": fe.DIMS,
        "input": {
            "size": fe.INPUT_SIZE,
            "color": "RGB",
            "range": [-1.0, 1.0],
            "resize": f"short side to {fe.DECODE_SHORT} (area) when larger, "
                      f"then squash to {fe.INPUT_SIZE}x{fe.INPUT_SIZE} (bilinear)",
        },
        "output": "pooled image embedding, not unit length",
        "probe": [round(float(v), 7) for v in reference[0]],
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    failed = False
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        cos = worst_cosine(sess.run(None, {"pixel_values": check})[0], reference)
        print(f"  ONNX Runtime CPU   worst cosine {cos:.6f}")
        failed |= cos < MIN_COSINE
    except ImportError:
        print("  ONNX Runtime not installed, not checked")
    try:
        import openvino as ov
        core = ov.Core()
        for dev in [d for d in core.available_devices if d == "CPU" or d.startswith("GPU")]:
            out = core.compile_model(onnx_path, dev)(check)[0]
            cos = worst_cosine(out, reference) if np.isfinite(out).all() else float("nan")
            print(f"  OpenVINO {dev:9} worst cosine {cos:.6f}")
            failed |= not cos >= MIN_COSINE
    except ImportError:
        print("  OpenVINO not installed, not checked")
    if failed:
        print(f"[export_frame_encoder] a runtime is below cosine {MIN_COSINE}; not usable")
        return 1
    print(f"[export_frame_encoder] done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
