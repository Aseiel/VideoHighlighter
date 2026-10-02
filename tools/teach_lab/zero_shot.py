"""Open vocabulary without training: does typing a class name find its clips?

    python tools/teach_lab/zero_shot.py <ds_features.npz> <frames.npz> --model <hf id> [--min 20]

frames.npz holds SigLIP2 image vectors per frame under "siglip" (big_features.py
for so400m, siglip_base_features.py for base). Every clip's 8 frame vectors are
averaged; each class name becomes text through the same model's text tower
(several phrasings, averaged); the nearest text wins. Nothing is trained, so
every clip counts -- there is nothing to hold out.
"""
import argparse
import sys
from collections import Counter

import numpy as np
import torch

TEMPLATES = ["{}", "a photo of {}", "a video frame showing {}", "people {}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("frames")
    ap.add_argument("--model", required=True)
    ap.add_argument("--min", type=int, default=20)
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    z = np.load(args.ds, allow_pickle=True)
    f = np.load(args.frames, allow_pickle=True)
    wf = {n: i for i, n in enumerate(f["names"])}
    ok = [i for i, (p, l, s) in enumerate(zip(z["paths"], z["labels"], z["split"]))
          if "|" not in l and p in wf and s in ("train", "val")]
    y = z["labels"][ok].astype(str)
    counts = Counter(y)
    keep = [i for i, c in zip(ok, y) if counts[c] >= args.min]
    y = z["labels"][keep].astype(str)
    img = f["siglip"][[wf[p] for p in z["paths"][keep]]].astype(np.float32).mean(1)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    classes = sorted(set(y))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    with torch.no_grad():
        txt = []
        for c in classes:
            t = tok([tpl.format(c) for tpl in TEMPLATES], padding="max_length", max_length=64,
                    truncation=True, return_tensors="pt")
            e = model.get_text_features(**t)
            e = getattr(e, "pooler_output", e)          # transformers 5 returns an output object
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            txt.append(torch.nn.functional.normalize(e, dim=0).numpy())
    txt = np.stack(txt)
    sim = img @ txt.T
    order = np.argsort(-sim, axis=1)
    yi = np.array([classes.index(c) for c in y])
    acc = np.mean(order[:, 0] == yi)
    top3 = np.mean([t in r for t, r in zip(yi, order[:, :3])])
    bal = np.mean([np.mean(order[yi == k, 0] == k) for k in range(len(classes))])
    print(f"{args.model}: {len(y)} clips, {len(classes)} classes, no training")
    print(f"  zero-shot accuracy {acc:.3f}  balanced {bal:.3f}  top-3 {top3:.3f}  "
          f"(always guessing the biggest class: {max(counts[c] for c in classes) / len(y):.3f})")


if __name__ == "__main__":
    sys.exit(main())
