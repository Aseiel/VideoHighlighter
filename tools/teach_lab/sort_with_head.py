"""Sort clips into folders with a trained action head, the way the app would.

    python tools/teach_lab/sort_with_head.py <clips folder> <head folder> <out folder>
        [--cache <file.npz>] [--compare <earlier sorted.csv>] [--backend auto]

Every clip goes through the app's own pieces: 4 frames across it
(``model_training.action_head.features``), the frame encoder
(``modules.vision.frame_encoder``) and the exported head (``head.onnx`` +
``head.json``, from ``model_training.action_head.train``). Each action has its
own score and its own trust threshold, and each taught pair one on the lower
of its two scores, so a clip lands in:

  <out>/<action>/          one action trusted
  <out>/<a>_<b>/           two or more trusted (by score, highest first),
                           the dataset's own folder convention for a clip
                           showing both
  <out>/_unsure/           nothing trusted: the guesses are in sorted.csv

Files are hard links (no copies; the clips folder is never changed).
``sorted.csv`` has every clip's top five actions with scores and what was
trusted; ``summary.json`` the counts. ``--compare`` reports how often an
earlier sort's folder agrees.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

VIDEO = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


def place(src, folder, out):
    os.makedirs(os.path.join(out, folder), exist_ok=True)
    dst = os.path.join(out, folder, os.path.basename(src))
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("head")
    ap.add_argument("out")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--compare", default=None, help="an earlier sort's sorted.csv")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    from model_training.action_head import features as Fx
    from model_training.action_head import head as H
    from model_training.action_head import trust
    from modules.vision import frame_encoder

    with open(os.path.join(args.head, "head.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    classes = meta["classes"]
    thresholds = meta["trust_thresholds"]
    pairs = [(classes.index(a), classes.index(b))
             for a, b in (p["actions"] for p in meta.get("pair_thresholds", []))]
    pair_th = [p["threshold"] for p in meta.get("pair_thresholds", [])]
    encoder = frame_encoder.load(args.backend)
    if encoder is None:
        return 1
    if meta["encoder"] != encoder.encoder_id:
        print(f"head was trained on {meta['encoder']}, this encoder is {encoder.encoder_id}")
        return 1
    names = sorted(n for n in os.listdir(args.clips) if n.lower().endswith(VIDEO))
    paths = [os.path.join(args.clips, n) for n in names]
    cache = Fx.FeatureCache(args.cache or os.path.join(args.out, "features.npz"),
                            encoder.encoder_id, int(meta["frames"]), encoder.dims)
    x, ok = Fx.encode_clips(paths, args.clips, encoder, cache)
    session = H.load_onnx_session(os.path.join(args.head, "head.onnx"))
    scores = H.onnx_proba(session, x)
    found = trust.detected(scores, thresholds, pairs, pair_th)

    os.makedirs(args.out, exist_ok=True)
    rows, folders = [], Counter()
    for i, (name, path) in enumerate(zip(names, paths)):
        if not ok[i]:
            folder = "_unreadable"
        else:
            hits = [k for k in np.argsort(-scores[i]) if found[i, k]]
            folder = "_".join(classes[k] for k in hits) if hits else "_unsure"
        folders[folder] += 1
        place(path, folder, args.out)
        top = np.argsort(-scores[i])[:5]
        row = {"clip": name, "folder": folder}
        for r, k in enumerate(top, 1):
            row[f"guess{r}"] = classes[k]
            row[f"score{r}"] = f"{scores[i, k]:.3f}"
        rows.append(row)
    with open(os.path.join(args.out, "sorted.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    summary = {"head": os.path.abspath(args.head), "encoder": encoder.encoder_id,
               "clips": len(rows), "folders": dict(folders.most_common()),
               "trusted_share": round(1 - (folders["_unsure"] + folders["_unreadable"]) / len(rows), 4),
               "two_or_more": sum(v for k, v in folders.items() if not k.startswith("_") and "_" in k)}
    if args.compare and os.path.isfile(args.compare):
        with open(args.compare, encoding="utf-8") as fh:
            before = {r.get("crop") or r.get("clip"): r["folder"] for r in csv.DictReader(fh)}
        both = [(r["folder"], before[r["clip"]]) for r in rows
                if r["clip"] in before and not r["folder"].startswith("_")
                and not before[r["clip"]].startswith("_")]
        summary["compare"] = {
            "earlier": os.path.abspath(args.compare),
            "sorted_in_both": len(both),
            "same_folder": round(float(np.mean([a == b for a, b in both])), 4) if both else None,
            "shares_an_action": (round(float(np.mean([bool(set(a.split("_")) & set(b.split("_")))
                                                       for a, b in both])), 4) if both else None),
        }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "folders"}, indent=1))
    print(f"{len(folders)} folders; largest: {folders.most_common(8)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
