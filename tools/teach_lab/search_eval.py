"""Visual search on two encoders, same clips, same frames: does SigLIP2 base
search better than the app's CLIP ViT-B/32?

    python tools/teach_lab/search_eval.py <ds_features.npz> --enc clip-b32=<clip_frames.npz>:clip \
        --enc siglip2-base=<siglip_base.npz>:siglip [--split source_split.json] [--out res.json]

Each --enc is name=<npz>:<key>, the npz holding 8 per-frame vectors per clip.
Clips: single-label, train/val, classes with 20+ clips, present in every file.
A clip's vector is the mean of its unit frame vectors, made unit again. Every
score that trains or picks examples holds out whole source videos (5 folds).

  text      each class name typed as a query (4 phrasings, averaged), nothing
            trained: closed-set accuracy, and ranking -- mean AP over classes
            and precision of the top 20 when every clip is ranked by the query
  example   the app's search by example: k example clips from other videos,
            their vectors averaged into one query, the held-out videos' clips
            ranked by cosine. mAP and precision@20, k = 1, 5, 20, 3 draws
  centroid  nearest class mean (teach's prototypes, one centre per class)
  head      eval_big's small per-frame head (class weights ^0.5), 5 folds, and
            on --split's 29 held-out videos when given
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import eval_big as B  # noqa: E402
import eval_grouping as E  # noqa: E402

TEMPLATES = ["{}", "a photo of {}", "a video frame showing {}", "people {}"]
SIGLIP_ID = "google/siglip2-base-patch16-256"
SO400M_ID = "google/siglip2-so400m-patch14-384"
B32_ID = "google/siglip2-base-patch32-256"


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def text_vectors(name, classes):
    """One unit vector per class: each phrasing embedded, unit, averaged, unit."""
    prompts = [t.format(c) for c in classes for t in TEMPLATES]
    if name.startswith("clip"):
        sys.modules.setdefault("sentence_transformers", None)
        from llm.clip_index import ClipEmbedder
        clip = ClipEmbedder(device="GPU")
        clip.load()
        e = clip.embed_texts(prompts)
    else:
        from transformers import AutoModel, AutoTokenizer
        mid = SO400M_ID if "so400m" in name else B32_ID if "b32" in name else SIGLIP_ID
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid).eval()
        with torch.no_grad():
            t = tok(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            e = model.get_text_features(**t)
            e = getattr(e, "pooler_output", e).numpy()
    e = unit(e).reshape(len(classes), len(TEMPLATES), -1).mean(1)
    return unit(e)


def ap(scores, rel):
    order = np.argsort(-scores)
    r = rel[order]
    if not r.any():
        return np.nan
    hits = np.cumsum(r)
    return float((hits[r] / (np.flatnonzero(r) + 1)).mean())


def p_at(scores, rel, k=20):
    return float(rel[np.argsort(-scores)[:k]].mean())


def text_search(x, y, classes, txt):
    sim = x @ txt.T
    yi = np.searchsorted(classes, y)
    order = np.argsort(-sim, 1)
    acc = np.mean(order[:, 0] == yi)
    bal = np.mean([np.mean(order[yi == k, 0] == k) for k in range(len(classes))])
    top3 = np.mean([t in r for t, r in zip(yi, order[:, :3])])
    aps = [ap(sim[:, k], yi == k) for k in range(len(classes))]
    pk = [p_at(sim[:, k], yi == k) for k in range(len(classes))]
    prev = np.mean([np.mean(yi == k) for k in range(len(classes))])
    return {"acc": acc, "bal": bal, "top3": top3, "mAP": float(np.mean(aps)), "P@20": float(np.mean(pk)),
            "mAP_chance": float(prev)}


def example_search(x, y, splits, ks=(1, 5, 20), draws=3):
    out = {}
    classes = sorted(set(y))
    for k in ks:
        aps, pks = [], []
        for d in range(draws):
            rng = np.random.default_rng(d)
            for tr, te in splits:
                for c in classes:
                    pool = tr[y[tr] == c]
                    rel = y[te] == c
                    if len(pool) < k or not rel.any():
                        continue
                    q = unit(x[rng.choice(pool, k, replace=False)].mean(0))
                    s = x[te] @ q
                    aps.append(ap(s, rel))
                    pks.append(p_at(s, rel))
        out[k] = {"mAP": float(np.nanmean(aps)), "P@20": float(np.mean(pks))}
    return out


def centroid(x, y, splits):
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        cl = np.array(sorted(set(y[tr])))
        cen = unit(np.stack([x[tr][y[tr] == c].mean(0) for c in cl]))
        pred[te] = cl[np.argmax(x[te] @ cen.T, 1)]
    return E.scores(y, pred)


def head_folds(seq, y, video, splits):
    classes = np.array(sorted(set(y)))
    yi = np.searchsorted(classes, y)
    pred = np.empty(len(y), dtype=object)
    top3 = [None] * len(y)
    for tr, te in splits:
        xs = B.standardise([seq], tr)
        m = B.train_head([xs[0][tr]], yi[tr], video[tr], len(classes), con=0, balance=0.5)
        with torch.no_grad():
            logits = m([torch.as_tensor(xs[0][te])])[1]
        order = logits.argsort(1, descending=True).numpy()
        pred[te] = classes[order[:, 0]]
        for i, row in zip(te, order[:, :3]):
            top3[i] = set(classes[row])
    return E.scores(y, pred, top3)


def head_split(seq, y, video, split, min_train=5):
    sp = json.load(open(split, encoding="utf-8"))
    is_tr, is_va = np.isin(video, sp["train_videos"]), np.isin(video, sp["val_videos"])
    counts = Counter(y[is_tr])
    keep = np.array([counts[c] >= min_train for c in y])
    tr, va = np.where(is_tr & keep)[0], np.where(is_va & keep)[0]
    classes = np.array(sorted(set(y[tr])))
    va = va[np.isin(y[va], classes)]
    yi = np.searchsorted(classes, y)
    xs = B.standardise([seq], tr)
    m = B.train_head([xs[0][tr]], yi[tr], video[tr], len(classes), con=0, balance=0.5)
    with torch.no_grad():
        pred = m([torch.as_tensor(xs[0][va])])[1].argmax(1).numpy()
    return {"acc": float(np.mean(pred == yi[va])), "clips": int(len(va))}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("ds")
    ap_.add_argument("--enc", action="append", required=True)
    ap_.add_argument("--split")
    ap_.add_argument("--min", type=int, default=20)
    ap_.add_argument("--out")
    args = ap_.parse_args()
    z = np.load(args.ds, allow_pickle=True)
    encs = {}
    for spec in args.enc:
        name, rest = spec.split("=", 1)
        path, key = rest.rsplit(":", 1)
        f = np.load(path, allow_pickle=True)
        encs[name] = ({n: i for i, n in enumerate(f["names"])}, f[key])
    ok = [i for i, (p, l, s) in enumerate(zip(z["paths"], z["labels"], z["split"]))
          if "|" not in l and s in ("train", "val") and all(p in w for w, _ in encs.values())]
    counts = Counter(z["labels"][ok])
    idx = np.array([i for i in ok if counts[z["labels"][i]] >= args.min])
    y = z["labels"][idx].astype(str)
    video = z["video"][idx].astype(str)
    classes = np.array(sorted(set(y)))
    splits = E.folds(y, video, 5)
    print(f"{len(y)} clips, {len(classes)} classes, {len(set(video))} videos\n", flush=True)
    res = {}
    for name, (where, arr) in encs.items():
        seq = unit(arr[[where[p] for p in z["paths"][idx]]].astype(np.float32))      # N, 8, D
        x = unit(seq.mean(1))
        r = {"dims": int(x.shape[1])}
        r["text"] = text_search(x, y, classes, text_vectors(name, list(classes)))
        r["example"] = example_search(x, y, splits)
        r["centroid"] = centroid(x, y, splits)
        r["head"] = head_folds(seq, y, video, splits)
        if args.split:
            r["head_split"] = head_split(seq, y, video, args.split)
        res[name] = r
        t, h = r["text"], r["head"]
        print(f"{name} ({r['dims']}-d)")
        print(f"  text     acc {t['acc']:.3f} bal {t['bal']:.3f} top3 {t['top3']:.3f} | "
              f"mAP {t['mAP']:.3f} (chance {t['mAP_chance']:.3f}) P@20 {t['P@20']:.3f}")
        for k, e in r["example"].items():
            print(f"  example  k={k:<3} mAP {e['mAP']:.3f}  P@20 {e['P@20']:.3f}")
        print(f"  centroid acc {r['centroid']['acc']:.3f} bal {r['centroid']['bal']:.3f}")
        print(f"  head     acc {h['acc']:.3f} bal {h['bal']:.3f} top3 {h['top3']:.3f}"
              + (f" | 29-video split {r['head_split']['acc']:.3f} ({r['head_split']['clips']} clips)"
                 if args.split else ""), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
