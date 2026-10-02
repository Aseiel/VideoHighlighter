"""Group new footage in the similarity learned from the hand-sorted dataset.

    python tools/teach_lab/regroup.py <crops folder> <crops big.npz> <out folder>
        --ds <ds_features.npz> --big <ds big.npz> [--groups 20] [--min 8]
        [--smooth 0.5] [--raw 0]

1. Train the eval_big head on every single-label dataset clip whose class has
   --min clips or more (cross-entropy + cross-video contrastive), all videos.
2. Embed the crops (big_features.py output for the crops folder).
   --smooth mixes in the same crop position 2.5 s before and after (0 = off).
   --raw blends in the backbones' own view, which keeps content the dataset
   has no class for apart (the learned space does not carry to unseen classes).
3. KMeans into --groups groups; each group is described by the dataset
   classes its crops' nearest dataset clips carry (cosine, 15 neighbours).

Writes <out>/gNN/ (hard links, most typical first, _sheet.jpg), groups.csv,
groups.json and groups.html.
"""
import argparse
import csv
import glob
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discover as D  # noqa: E402
import eval_big as B  # noqa: E402

NAME = re.compile(r"^(.*?)__(\d+)(.*)\.mp4$")


def dataset(ds, big, min_clips):
    z = np.load(ds, allow_pickle=True)
    b = np.load(big, allow_pickle=True)
    wb = {n: i for i, n in enumerate(b["names"])}
    ok = [i for i, (p, l) in enumerate(zip(z["paths"], z["labels"])) if "|" not in l and p in wb]
    y = z["labels"][ok].astype(str)
    counts = Counter(y)
    keep = [i for i, c in zip(ok, y) if counts[c] >= min_clips]
    rows = [wb[p] for p in z["paths"][keep]]
    seqs = [b[k][rows].astype(np.float32) for k in B.BIG]
    return seqs, z["labels"][keep].astype(str), z["video"][keep].astype(str)


def smooth(names, emb, w):
    if w <= 0:
        return emb
    where = {}
    for i, n in enumerate(names):
        m = NAME.match(n)
        if m:
            where[(m.group(1), m.group(3), int(m.group(2)))] = i
    out = emb.copy()
    for (vid, pos, ms), i in where.items():
        nb = [where[k] for k in ((vid, pos, ms - 2500), (vid, pos, ms + 2500)) if k in where]
        if nb:
            out[i] = (1 - w) * emb[i] + w * emb[nb].mean(0)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    from sklearn.cluster import KMeans
    ap = argparse.ArgumentParser()
    ap.add_argument("crops"); ap.add_argument("crops_big"); ap.add_argument("out")
    ap.add_argument("--ds", required=True, help="ds_features.py output for the dataset")
    ap.add_argument("--big", required=True, help="big_features.py output for the dataset")
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--min", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--con", type=float, default=0.0, help="cross-video contrastive weight (0 measured best)")
    ap.add_argument("--raw", type=float, default=0.0, help="share of the raw backbone view in the grouping (0-1)")
    args = ap.parse_args()

    seqs, y, video = dataset(args.ds, args.big, args.min)
    classes = np.array(sorted(set(y)))
    print(f"regroup: training on {len(y)} clips, {len(classes)} classes, {len(set(video))} videos", flush=True)
    all_idx = np.arange(len(y))
    c = np.load(args.crops_big, allow_pickle=True)
    names = [str(n) for n in c["names"]]
    cseqs = [c[k].astype(np.float32) for k in B.BIG]
    # standardise with the dataset's statistics, the same transform for both
    stats = [(s.reshape(-1, s.shape[-1]).mean(0), s.reshape(-1, s.shape[-1]).std(0) + 1e-5) for s in seqs]
    seqs = [((s - mu) / sd).astype(np.float32) for s, (mu, sd) in zip(seqs, stats)]
    cseqs = [((s - mu) / sd).astype(np.float32) for s, (mu, sd) in zip(cseqs, stats)]
    model = B.train_head(seqs, np.searchsorted(classes, y), video, len(classes), con=args.con)
    with torch.no_grad():
        demb = model.embed([torch.as_tensor(s) for s in seqs]).numpy()
        emb = np.concatenate([model.embed([torch.as_tensor(s[i:i + 512]) for s in cseqs]).numpy()
                              for i in range(0, len(names), 512)])
    emb = smooth(names, emb, args.smooth)
    space = emb
    if args.raw > 0:
        # the backbones' own view (PCA of the mean-over-frames features): the
        # learned space only knows the dataset's classes, this keeps the rest apart
        from sklearn.decomposition import PCA
        raw = np.concatenate([s.mean(1) / np.sqrt(s.shape[-1]) for s in cseqs], 1)
        raw = PCA(32, random_state=0).fit_transform(raw)
        raw = smooth(names, raw / np.linalg.norm(raw, axis=1, keepdims=True), args.smooth)
        space = np.concatenate([emb * np.sqrt(1 - args.raw), raw * np.sqrt(args.raw)], 1) \
            if args.raw < 1 else raw

    km = KMeans(args.groups, n_init=10, random_state=0).fit(space)
    lab = km.labels_
    dist = np.linalg.norm(space - km.cluster_centers_[lab], axis=1)
    sim = emb @ demb.T
    nn = np.argsort(-sim, axis=1)[:, :15]
    near = [Counter(y[r]) for r in nn]

    for old in glob.glob(os.path.join(args.out, "g*")):
        if os.path.isdir(old):
            for f in os.listdir(old):
                os.remove(os.path.join(old, f))
            os.rmdir(old)
    os.makedirs(args.out, exist_ok=True)

    spread = {g: float(np.median(dist[lab == g])) for g in range(args.groups)}
    order = sorted(spread, key=spread.get)
    rename = {g: f"g{i + 1:02d}" for i, g in enumerate(order)}
    rows, summary = [], {}
    for g in order:
        gname = rename[g]
        idx = np.where(lab == g)[0]
        idx = idx[np.argsort(dist[idx])]
        folder = os.path.join(args.out, gname)
        os.makedirs(folder, exist_ok=True)
        votes = Counter()
        for i in idx:
            votes.update(near[i])
            try:
                os.link(os.path.join(args.crops, names[i]), os.path.join(folder, names[i]))
            except OSError:
                pass
            m = NAME.match(names[i])
            rows.append({"crop": names[i], "at_s": int(m.group(2)) / 1000 if m else "",
                         "group": gname, "nearest_class": near[i].most_common(1)[0][0],
                         "nearest_share": round(near[i].most_common(1)[0][1] / 15, 2)})
        total = sum(votes.values())
        D.contact_sheet([names[i] for i in idx[:36]], args.crops, os.path.join(folder, "_sheet.jpg"))
        summary[gname] = {"crops": int(len(idx)), "spread": round(spread[g], 3),
                          "minutes": sorted({int(r["at_s"] // 60) for r in rows if r["group"] == gname}),
                          "looks_like": {k: round(v / total, 2) for k, v in votes.most_common(4)}}
    with open(os.path.join(args.out, "groups.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.out, "groups.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    page(args.out, rows, summary)
    np.savez(os.path.join(args.out, "embedding.npz"), names=np.array(names), emb=emb, group=lab)
    for g, s in summary.items():
        print(f"{g} {s['crops']:4} spread {s['spread']:.3f}  " +
              ", ".join(f"{k} {v:.0%}" for k, v in s["looks_like"].items()), flush=True)


def page(out_dir, rows, summary):
    import colorsys
    names = list(summary)
    colour = {g: "#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, .55, .9))
              for i, g in enumerate(names)}
    end = max(r["at_s"] for r in rows) + 5
    by = defaultdict(Counter)
    for r in rows:
        by[int(r["at_s"] // 2.5)][r["group"]] += 1
    cells = "".join(f'<i style="background:{colour.get(by[s].most_common(1)[0][0], "#111") if by[s] else "#111"}" '
                    f'title="{int(s * 2.5 // 60):02d}:{s * 2.5 % 60:04.1f} {by[s].most_common(1)[0][0] if by[s] else ""}"></i>'
                    for s in range(int(end // 2.5) + 1))
    parts = "".join(
        f"<section><h2 style='border-color:{colour[g]}'>{g} <small>{s['crops']} crops · spread {s['spread']} · "
        f"minutes {', '.join(map(str, s['minutes'][:20]))}</small></h2>"
        f"<p>nearest in the sorted dataset: {html.escape(', '.join(f'{k} {v:.0%}' for k, v in s['looks_like'].items()))}</p>"
        f"<img src='{g}/_sheet.jpg' loading='lazy'></section>" for g, s in summary.items())
    with open(os.path.join(out_dir, "groups.html"), "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset=utf-8><title>Learned groups</title><style>"
                 "body{background:#111;color:#ddd;font:14px system-ui;margin:16px}"
                 "#strip{display:flex;height:28px;margin:8px 0 24px}#strip i{flex:1}"
                 "h2{border-left:8px solid;padding-left:8px;font-size:16px}small{color:#999;font-weight:normal}"
                 "img{max-width:100%}p{color:#aaa;margin:4px 0 8px}</style>"
                 f"<h1>{len(names)} learned groups</h1><div id=strip>{cells}</div>{parts}")


if __name__ == "__main__":
    sys.exit(main())
