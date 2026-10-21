"""Export the multilingual CLIP text model (llm/clip_multilingual.py) to ONNX.

    python -m tools.export_clip_multilingual [--out models/clip-vit-b32-multilingual]

Downloads sentence-transformers/clip-ViT-B-32-multilingual-v1 (Apache-2.0) at a
pinned revision and writes, into one folder:

  text.onnx      DistilBERT-multilingual + mean pooling + the 768->512 dense
                 layer, as one graph: (input_ids, attention_mask) -> vector.
                 The token embedding table (~70 % of the model) is stored as
                 int8 rows with one scale each, the rest as fp16.
  tokenizer.json the source model's own.
  encoder.json   source, revision, and the probe vector the app checks at load.

Before writing anything it proves three things: the app's tokenizer makes the
same ids as the source model's; the ONNX graph reproduces the torch model; and
its vectors sit in CLIP ViT-B/32's space, i.e. the same sentence in several
languages is closest to CLIP's own encoding of the English one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from llm import clip_multilingual as cm  # noqa: E402

OPSET = 17

# The same two prompts in English and other scripts: each must be closest to
# CLIP's own English encoding of its meaning, not of the other prompt.
ALIGNMENT = (
    ("a photo of a dog", ("一只狗的照片", "zdjęcie psa", "ein Foto von einem Hund",
                          "犬の写真")),
    ("a photo of a car", ("一辆汽车的照片", "zdjęcie samochodu", "ein Foto von einem Auto",
                          "車の写真")),
)
CHECK_TEXTS = ["a photo of a dog", "一只狗在草地上奔跑", "Ludzie tańczą na weselu",
               "夕焼けの海辺", "x"]


def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join("models", cm.MODEL_DIRNAME))
    ap.add_argument("--revision", default=cm.SOURCE_REVISION)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if cm.find_model_dir() == os.path.abspath(args.out) and not args.force:
        print(f"[export_clip_multilingual] already in {args.out}, skipping")
        return 0

    import onnx
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    from modules.vision.onnx_weights import store_weights_fp16
    from tools.export_frame_encoder import store_embedding_int8

    src = snapshot_download(cm.SOURCE_MODEL, revision=args.revision, allow_patterns=[
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.txt", "modules.json",
        "sentence_bert_config.json", "1_Pooling/*", "2_Dense/*"])

    # The source's own pipeline, read from its files rather than assumed.
    with open(os.path.join(src, "modules.json"), encoding="utf-8") as fh:
        kinds = [m["type"].rsplit(".", 1)[-1] for m in json.load(fh)]
    if kinds != ["Transformer", "Pooling", "Dense"]:
        print(f"[export_clip_multilingual] unexpected modules {kinds}")
        return 1
    with open(os.path.join(src, "1_Pooling", "config.json"), encoding="utf-8") as fh:
        pooling = json.load(fh)
    if not pooling.get("pooling_mode_mean_tokens") or any(
            v for k, v in pooling.items() if k.startswith("pooling_mode_") and
            k != "pooling_mode_mean_tokens"):
        print(f"[export_clip_multilingual] pooling is not plain mean: {pooling}")
        return 1
    with open(os.path.join(src, "2_Dense", "config.json"), encoding="utf-8") as fh:
        dense_cfg = json.load(fh)
    if not dense_cfg.get("activation_function", "").endswith("Identity"):
        print(f"[export_clip_multilingual] dense activation {dense_cfg}")
        return 1
    with open(os.path.join(src, "sentence_bert_config.json"), encoding="utf-8") as fh:
        if json.load(fh).get("max_seq_length") != cm.MAX_LENGTH:
            print("[export_clip_multilingual] max_seq_length changed")
            return 1

    # Eager attention, and the mask handed over already 4D: transformers'
    # builder for it (masking_utils) does not trace to ONNX, and a prepared
    # mask is passed through untouched. Checked against transformers' own
    # 2D-mask path below.
    bert = AutoModel.from_pretrained(src, torch_dtype=torch.float32,
                                     attn_implementation="eager").eval()
    dense = torch.nn.Linear(dense_cfg["in_features"], dense_cfg["out_features"],
                            bias=dense_cfg.get("bias", True))
    weights = load_file(os.path.join(src, "2_Dense", "model.safetensors"))
    dense.load_state_dict({k.split(".", 1)[1]: v for k, v in weights.items()})
    dense.eval()

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert, self.dense = bert, dense

        def forward(self, input_ids, attention_mask):
            keep = attention_mask[:, None, None, :].to(torch.float32)
            additive = (1.0 - keep) * torch.finfo(torch.float32).min
            hidden = self.bert(input_ids=input_ids, attention_mask=additive).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return self.dense(pooled)

    text = Text().eval()

    # 1. The app's tokenizer is the source's.
    hf_tok = AutoTokenizer.from_pretrained(src)
    want = hf_tok(CHECK_TEXTS, padding=True, truncation=True, max_length=cm.MAX_LENGTH,
                  return_tensors="np")
    tok = cm.make_tokenizer(os.path.join(src, "tokenizer.json"))
    ids, mask = cm.tokenize(tok, CHECK_TEXTS)
    if not (np.array_equal(ids, want["input_ids"]) and
            np.array_equal(mask, want["attention_mask"])):
        print("[export_clip_multilingual] the tokenizers disagree")
        return 1

    with torch.no_grad():
        reference = text(torch.from_numpy(ids), torch.from_numpy(mask)).numpy()
        hidden = bert(input_ids=torch.from_numpy(ids),
                      attention_mask=torch.from_numpy(mask)).last_hidden_state
        m = torch.from_numpy(mask).unsqueeze(-1).float()
        stock = dense((hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy()
    same = float(min((_unit(reference) * _unit(stock)).sum(1)))
    if same < 0.99999:
        print(f"[export_clip_multilingual] the 4D mask changes the answer (cosine {same:.6f})")
        return 1

    # 2. The ONNX graph.
    os.makedirs(args.out, exist_ok=True)
    onnx_path = os.path.join(args.out, cm.MODEL_FILE)
    tmp = onnx_path + ".fp32.tmp"
    torch.onnx.export(text, (torch.from_numpy(ids[:1]), torch.from_numpy(mask[:1])), tmp,
                      input_names=["input_ids", "attention_mask"], output_names=["text_embeds"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True,
                      dynamic_axes={"input_ids": {0: "n", 1: "len"},
                                    "attention_mask": {0: "n", 1: "len"},
                                    "text_embeds": {0: "n"}})
    converted, rows = store_embedding_int8(onnx.load(tmp))
    converted, n = store_weights_fp16(converted)
    onnx.save(converted, onnx_path)
    os.remove(tmp)
    with open(os.path.join(src, "tokenizer.json"), "rb") as fh, \
            open(os.path.join(args.out, cm.TOKENIZER_FILE), "wb") as out:
        out.write(fh.read())
    probe = _unit(reference[:1])[0]
    with open(os.path.join(args.out, cm.META_FILE), "w", encoding="utf-8") as fh:
        json.dump({"format": cm.FORMAT, "source": cm.SOURCE_MODEL, "revision": args.revision,
                   "license": "apache-2.0", "clip_model": "openai/clip-vit-base-patch32",
                   "dims": cm.DIMS, "probe_text": cm.PROBE_TEXT,
                   "probe": [round(float(x), 7) for x in probe]}, fh, indent=1)
    print(f"[export_clip_multilingual] {rows} embedding rows as int8, {n} tensors as fp16, "
          f"{os.path.getsize(onnx_path) / 2**20:.1f} MB")

    model = cm.MultilingualText(args.out)
    got = model.encode(CHECK_TEXTS)
    worst = float(min((_unit(reference) * got).sum(1)))
    print(f"[export_clip_multilingual] ONNX vs torch: worst cosine {worst:.5f}")
    if worst < cm.PROBE_MIN_COSINE:
        print("[export_clip_multilingual] the ONNX graph does not reproduce the model")
        return 1

    # 3. In CLIP's space: each translation is nearest CLIP's own English prompt.
    from llm.clip_prefilter import MODEL_ID
    from transformers import CLIPModel, CLIPTokenizer
    clip = CLIPModel.from_pretrained(MODEL_ID).eval()
    clip_tok = CLIPTokenizer.from_pretrained(MODEL_ID)
    english = [e for e, _ in ALIGNMENT]
    with torch.no_grad():
        enc = clip_tok(english, padding=True, return_tensors="pt")
        out = clip.text_model(**enc).pooler_output
        anchors = _unit(clip.text_projection(out).numpy())
    failures = 0
    for i, (eng, others) in enumerate(ALIGNMENT):
        for phrase in (eng,) + others:
            sims = anchors @ model.encode([phrase])[0]
            ok = int(np.argmax(sims)) == i
            failures += not ok
            print(f"   {'✓' if ok else '✗'} {phrase!r}: cos to {eng!r} {sims[i]:.3f}, "
                  f"to the other {sims[1 - i]:.3f}")
    if failures:
        print(f"[export_clip_multilingual] {failures} phrase(s) landed nearer the wrong prompt")
        return 1
    print(f"[export_clip_multilingual] done: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
