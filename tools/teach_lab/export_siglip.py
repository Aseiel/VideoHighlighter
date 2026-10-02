"""Export SigLIP2's two towers to ONNX and OpenVINO IR, and check them against PyTorch.

    python tools/teach_lab/export_siglip.py <out dir> [--name google/siglip2-base-patch16-256]

Writes into <out dir>:
  vision.onnx      pixel_values [N, 3, S, S] float32 -> image_embeds [N, D]  (pooled, not unit)
  text.onnx        input_ids [N, 64] int64 -> text_embeds [N, D]
  vision.xml/.bin  OpenVINO IR, weights stored FP16
  text.xml/.bin
  siglip.json      image size, mean/std, text length, logit_scale, logit_bias, model id

Input is RGB, resized to S x S, scaled to [-1, 1] ((x / 255 - 0.5) / 0.5), as
the model card's processor does. Text is lower-cased by the caller and padded
to 64 tokens ("max_length"), which is how SigLIP was trained.

Checks: ONNX Runtime (CPU) and OpenVINO (CPU, GPU) against PyTorch fp32 on the
same random images and prompts -- worst cosine per output.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

TEXT_LEN = 64


class Vision(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m.vision_model

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).pooler_output


class Text(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m.text_model

    def forward(self, input_ids):
        return self.m(input_ids=input_ids).pooler_output


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--name", default="google/siglip2-base-patch16-256")
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    os.makedirs(args.out, exist_ok=True)
    model = AutoModel.from_pretrained(args.name, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(args.name)
    size = model.config.vision_config.image_size
    vis, txt = Vision(model).eval(), Text(model).eval()

    rng = np.random.default_rng(0)
    px = torch.from_numpy(rng.uniform(-1, 1, (4, 3, size, size)).astype(np.float32))
    ids = tok(["a person walking", "two people", "an empty room", "a close-up of a hand"],
              padding="max_length", max_length=TEXT_LEN, truncation=True, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ref_v, ref_t = vis(px).numpy(), txt(ids).numpy()

    paths = {k: os.path.join(args.out, f"{k}.onnx") for k in ("vision", "text")}
    torch.onnx.export(vis, (px[:1],), paths["vision"], input_names=["pixel_values"],
                      output_names=["image_embeds"], opset_version=17, dynamo=False,
                      dynamic_axes={"pixel_values": {0: "n"}, "image_embeds": {0: "n"}})
    # With no attention mask (SigLIP pads to 64 and attends to all of it) the
    # eager mask is None anyway; transformers 5's mask builder cannot be traced.
    import transformers.models.siglip.modeling_siglip as ms
    ms.create_bidirectional_mask = lambda *a, **k: None
    torch.onnx.export(txt, (ids[:1],), paths["text"], input_names=["input_ids"],
                      output_names=["text_embeds"], opset_version=17, dynamo=False,
                      dynamic_axes={"input_ids": {0: "n"}, "text_embeds": {0: "n"}})

    import onnxruntime as ort
    import openvino as ov
    core = ov.Core()
    for k, (inp, ref) in {"vision": (px.numpy(), ref_v), "text": (ids.numpy(), ref_t)}.items():
        sess = ort.InferenceSession(paths[k], providers=["CPUExecutionProvider"])
        print(f"{k}: onnxruntime CPU vs torch, worst cosine {cos(sess.run(None, {sess.get_inputs()[0].name: inp})[0], ref):.6f}")
        ir = ov.convert_model(paths[k])
        ov.save_model(ir, os.path.join(args.out, f"{k}.xml"), compress_to_fp16=True)
        for dev in ("CPU", "GPU"):
            if dev not in core.available_devices:
                continue
            net = core.compile_model(os.path.join(args.out, f"{k}.xml"), dev)
            out = net(inp)[0]
            print(f"{k}: OpenVINO {dev} (IR fp16 weights, default precision) vs torch, "
                  f"worst cosine {cos(out, ref):.6f}, finite {bool(np.isfinite(out).all())}")

    meta = {"model": args.name, "image_size": size, "mean": [0.5] * 3, "std": [0.5] * 3,
            "text_length": TEXT_LEN, "lowercase": True,
            "logit_scale": float(model.logit_scale.exp()), "logit_bias": float(model.logit_bias),
            "dims": int(ref_v.shape[1])}
    with open(os.path.join(args.out, "siglip.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    for f in sorted(os.listdir(args.out)):
        print(f"  {f:16} {os.path.getsize(os.path.join(args.out, f)) / 1e6:8.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
