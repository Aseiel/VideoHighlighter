"""Export the frame encoder (SigLIP2 base/16, both towers) for the app.

    python -m tools.export_frame_encoder [--out models/siglip2-base-patch16-256] [--force]

Writes the files ``modules/vision/frame_encoder.py`` loads, and nothing else:

  vision.onnx    pixel_values [N, 3, 256, 256] -> image_embeds [N, 768] (the
                 pooled vector, not unit length). Weights stored as fp16 and
                 widened to fp32 by a Cast at load, so the file is half the size
                 and computes exactly as the fp32 export does.
  text.onnx      input_ids [N, 64] -> text_embeds [N, 768], for actions people
                 type. The 256k-token embedding table is most of the tower, so
                 it is stored as int8 with one scale per token; the rest fp16.
  tokenizer.json the tower's tokenizer (Gemma's), read by the `tokenizers`
                 library the app already ships.
  actions.npz    the Kinetics-700 action names (kinetics_700_labels.json) and
                 their vectors, so an action from that list costs no text-tower
                 call at run time.
  encoder.json   the encoder id, input description, the vectors PyTorch
                 computes for frame_encoder.probe_pixels() and PROBE_TEXT, which
                 every route in the app must reproduce before it is used, and
                 the model's own score scale and bias.

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
# Shared with installing a fine-tuned action model, which the app does itself.
from modules.vision.onnx_weights import store_weights_fp16  # noqa: E402,F401

OPSET = 17
MIN_COSINE = 0.9999
# The int8 embedding table moves text vectors a little more than fp16 does.
MIN_TEXT_COSINE = 0.999
KINETICS_700 = "kinetics_700_labels.json"


def store_embedding_int8(model):
    """Store the token embedding table as int8 rows with one fp32 scale each.

    The table is [256000, 768], about 70 % of the text tower. Its Gather is
    rewritten as Gather(int8 rows) -> Cast -> Mul(Gather(scales)); a token's
    vector then differs from fp32 by at most half a step of its own row's
    scale. Returns (model, rows), or (model, 0) when there is no such table.
    """
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    for node in graph.node:
        if node.op_type != "Gather" or node.input[0] not in inits:
            continue
        table = inits[node.input[0]]
        if table.data_type != TensorProto.FLOAT or len(table.dims) != 2 or table.dims[0] < 100000:
            continue
        weights = numpy_helper.to_array(table)
        scale = np.abs(weights).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(weights / scale[:, None]), -127, 127).astype(np.int8)
        name = node.input[0]
        graph.initializer.remove(table)
        graph.initializer.extend([numpy_helper.from_array(q, name + "__int8"),
                                  numpy_helper.from_array(scale.astype(np.float32)[:, None],
                                                          name + "__scale")])
        ids, out = node.input[1], node.output[0]
        rows, cast, sc = out + "__q", out + "__f", out + "__s"
        new = [helper.make_node("Gather", [name + "__int8", ids], [rows], axis=0),
               helper.make_node("Cast", [rows], [cast], to=TensorProto.FLOAT),
               helper.make_node("Gather", [name + "__scale", ids], [sc], axis=0),
               helper.make_node("Mul", [cast, sc], [out])]
        at = list(graph.node).index(node)
        graph.node.remove(node)
        for i, n in enumerate(new):
            graph.node.insert(at + i, n)
        return model, int(weights.shape[0])
    return model, 0


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
    have_all = all(os.path.isfile(os.path.join(args.out, f)) for f in
                   (fe.MODEL_FILE, fe.META_FILE, fe.TEXT_FILE, fe.TEXT_TOKENIZER, fe.ACTIONS_FILE))
    if have_all and not args.force:
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

    # ── the text tower ──
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.source, revision=args.revision)
    tokenizer_path = os.path.join(args.out, fe.TEXT_TOKENIZER)
    tokenizer.backend_tokenizer.save(tokenizer_path)

    def ids_for(texts):
        # The app's own tokenizer, so the export checks what the app will feed.
        return fe.tokenize(texts, tokenizer_path)

    for sample in (fe.PROBE_TEXT, "riding a bike", "Explosion!", "x " * 300):
        expected = tokenizer([sample.lower()], padding="max_length", max_length=fe.TEXT_LENGTH,
                             truncation=True, return_tensors="np")["input_ids"]
        if not np.array_equal(ids_for([sample]), expected.astype(np.int64)):
            print(f"[export_frame_encoder] the app's tokenizer differs from transformers' on {sample[:30]!r}")
            return 1

    class Text(torch.nn.Module):
        """SiglipTextTransformer.forward without its mask helper, which fails
        under tracing. The text tower attends to every position (it is trained
        on padded input with no mask), so None is the mask it builds anyway."""

        def __init__(self, m):
            super().__init__()
            self.m = m.text_model

        def forward(self, input_ids):
            hidden = self.m.embeddings(input_ids=input_ids)
            hidden = self.m.encoder(inputs_embeds=hidden, attention_mask=None).last_hidden_state
            hidden = self.m.final_layer_norm(hidden)
            return self.m.head(hidden[:, -1, :])

    text = Text(model).eval()
    text_check = ids_for([fe.PROBE_TEXT, "riding a bike", "Explosion!", "x " * 300])
    with torch.no_grad():
        text_reference = text(torch.from_numpy(text_check)).numpy()
    text_path = os.path.join(args.out, fe.TEXT_FILE)
    tmp = text_path + ".fp32.tmp"
    torch.onnx.export(text, (torch.from_numpy(text_check[:1]),), tmp,
                      input_names=["input_ids"], output_names=["text_embeds"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True,
                      dynamic_axes={"input_ids": {0: "n"}, "text_embeds": {0: "n"}})
    converted, rows = store_embedding_int8(onnx.load(tmp))
    converted, n_text = store_weights_fp16(converted)
    onnx.save(converted, text_path)
    os.remove(tmp)
    print(f"[export_frame_encoder] text tower: {rows} embedding rows as int8, "
          f"{n_text} tensors as fp16, {os.path.getsize(text_path) / 1e6:.1f} MB")

    # The Kinetics-700 names, encoded once here the way the app encodes typed ones.
    labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               KINETICS_700)
    with open(labels_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    labels = [raw[k] for k in sorted(raw, key=int)] if isinstance(raw, dict) else list(raw)
    with torch.no_grad():
        per_template = [text(torch.from_numpy(ids_for([t.format(a) for a in labels]))).numpy()
                        for t in fe.TEXT_TEMPLATES]
    np.savez(os.path.join(args.out, fe.ACTIONS_FILE),
             labels=np.array(labels), vectors=fe.ensemble(per_template).astype(np.float16))

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
        "text": {
            "file": fe.TEXT_FILE,
            "tokenizer": fe.TEXT_TOKENIZER,
            "length": fe.TEXT_LENGTH,
            "lowercase": True,
            "templates": list(fe.TEXT_TEMPLATES),
            "probe_text": fe.PROBE_TEXT,
            "probe": [round(float(v), 7) for v in text_reference[0]],
        },
        # sigmoid(logit_scale * cosine + logit_bias) is the model's own
        # image-text match probability; the scale also sets how sharply the
        # action names compete in action_siglip.
        "logit_scale": round(float(model.logit_scale.exp()), 6),
        "logit_bias": round(float(model.logit_bias), 6),
        "actions": {
            "file": fe.ACTIONS_FILE,
            "count": len(labels),
            "source": "Kinetics-700-2020 class names, DeepMind, CC BY 4.0",
        },
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
            # As the app compiles it: a check of other settings checks nothing.
            out = core.compile_model(onnx_path, dev, fe.openvino_config(dev))(check)[0]
            cos = worst_cosine(out, reference) if np.isfinite(out).all() else float("nan")
            print(f"  OpenVINO {dev:9} worst cosine {cos:.6f}")
            failed |= not cos >= MIN_COSINE
    except ImportError:
        print("  OpenVINO not installed, not checked")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(text_path, providers=["CPUExecutionProvider"])
        cos = worst_cosine(sess.run(None, {"input_ids": text_check})[0], text_reference)
        print(f"  text, ONNX Runtime CPU worst cosine {cos:.6f}")
        failed |= cos < MIN_TEXT_COSINE
    except ImportError:
        pass
    try:
        import openvino as ov
        out = ov.Core().compile_model(text_path, "CPU", fe.openvino_config("CPU"))(text_check)[0]
        cos = worst_cosine(out, text_reference)
        print(f"  text, OpenVINO CPU     worst cosine {cos:.6f}")
        failed |= not cos >= MIN_TEXT_COSINE
    except ImportError:
        pass
    if failed:
        print(f"[export_frame_encoder] a runtime is below cosine {MIN_COSINE}; not usable")
        return 1
    print(f"[export_frame_encoder] done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
