"""Sort a video's crops with a linear layer trained on the hand-sorted dataset.

    python tools/teach_lab/apply_classifier.py <ds_features.npz> <crop features.npz> <out folder>
        --crops <crops folder> --ds-extra <ds video_features.npz> --crop-extra <crop video_features.npz>
        [--groups groups.csv] [--mix clip8=1,r21d=1,intel=1] [--learned-groups 20]
        [--min 20] [--precision 0.7]

1. Train logistic regression (balanced) on every single-class train/val clip
   of classes with at least --min clips.
2. Pick the confidence threshold on held-out videos (GroupKFold): the lowest
   one at which --precision of what is sorted is right.
3. Score every crop; a 5 s window takes, per class, its best crop; windows are
   smoothed with their neighbours 2.5 s either side.
4. Write <out>/<class>/ (hard links to the window's best crop), <out>/_unsure/,
   timeline.csv, and -- with --groups -- the class mix of every discovery group.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_grouping as E  # noqa: E402


def parse_mix(text):
    w = {"clip8": 1.0, "r21d": 1.0, "intel": 1.0} if not text else {}
    for part in filter(None, text.split(",")):
        k, v = part.split("=")
        w[k.strip()] = float(v)
    return w


def fit_scalers(blocks, weights):
    from sklearn.preprocessing import StandardScaler
    return {k: StandardScaler().fit(blocks[k]) for k, w in weights.items() if w > 0}


def transform(blocks, weights, scalers):
    parts = []
    for k, w in weights.items():
        if w <= 0:
            continue
        x = scalers[k].transform(blocks[k]) / np.sqrt(blocks[k].shape[1])
        parts.append(x * np.sqrt(w))
    return np.concatenate(parts, axis=1)


def threshold_for(blocks, weights, y, video, precision, C):
    """Lowest confidence at which held-out predictions reach ``precision``."""
    splits = E.folds(y, video, 5)
    _, pred, conf = E.logreg(blocks, weights, y, splits, C=C, return_proba=True)
    right = pred == y
    order = np.argsort(-conf)
    precision_at = np.cumsum(right[order]) / np.arange(1, len(order) + 1)
    ok = np.where(precision_at >= precision)[0]
    if not len(ok):
        return 1.0, 0.0, float(right.mean())
    last = ok.max()
    return float(conf[order][last]), float((last + 1) / len(y)), float(right.mean())


def learned_groups(args, model, x, names, classes, proba):
    """Group the crops in the space the classifier learned from the sorting,
    with discover.py's folders, contact sheets and overview page."""
    import shutil

    import discover as DI
    from sklearn.cluster import KMeans

    z = model.decision_function(x)
    km = KMeans(args.learned_groups, n_init=10, random_state=0).fit(z)
    dist = np.linalg.norm(z - km.cluster_centers_[km.labels_], axis=1)
    out = os.path.join(args.out, "groups")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    members = defaultdict(list)
    for i, g in enumerate(km.labels_):
        members[int(g)].append((dist[i], i))
    spread = {g: float(np.median([d for d, _ in m])) for g, m in members.items()}
    rows, summary = [], {}
    for rank, g in enumerate(sorted(members, key=lambda g: spread[g]), 1):
        gname = f"g{rank:02d}"
        folder = os.path.join(out, gname)
        os.makedirs(folder)
        ids = [i for _, i in sorted(members[g])]
        for i in ids:
            try:
                os.link(os.path.join(args.crops, names[i]), os.path.join(folder, names[i]))
            except OSError:
                pass
            t = int(re.match(r"v\d+__(\d+)", names[i]).group(1)) / 1000.0
            rows.append({"crop": names[i], "at_s": t, "group": gname})
        DI.contact_sheet([names[i] for i in ids[:36]], args.crops, os.path.join(folder, "_sheet.jpg"))
        votes = Counter(classes[proba[ids].argmax(1)])
        p = proba[ids].mean(0)
        times = [int(re.match(r"v\d+__(\d+)", names[i]).group(1)) / 1000.0 for i in ids]
        summary[gname] = {
            "crops": len(ids), "samples": len({re.match(r"(v\d+__\d+)", names[i]).group(1) for i in ids}),
            "spread": round(spread[g], 2), "minutes": sorted({int(t // 60) for t in times}),
            "people_avg": 0, "main_torso_upright": 0, "two_people_share": 0, "motion": 0, "rhythm_hz": 0,
            # the page's "earlier guesses" line shows what the classifier says instead
            "earlier_guesses": {k: v for k, v in votes.most_common(4)},
            "classifier_mean": {classes[j]: round(float(p[j]), 2) for j in np.argsort(-p)[:3]},
        }
    with open(os.path.join(out, "groups.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out, "groups.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    DI.overview(out, rows, summary)
    print(f"learned groups: {len(summary)} in {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_features")
    ap.add_argument("crop_features")
    ap.add_argument("out")
    ap.add_argument("--crops", required=True, help="the folder the crops are in")
    ap.add_argument("--groups")
    ap.add_argument("--mix", default="")
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--precision", type=float, default=0.7)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--learned-groups", type=int, default=0, help="also group the crops in the learned space")
    ap.add_argument("--ds-extra", required=True, help="video_features.py output for the dataset")
    ap.add_argument("--crop-extra", required=True, help="video_features.py output for the crops")
    args = ap.parse_args()
    weights = parse_mix(args.mix)

    from sklearn.linear_model import LogisticRegression

    blocks, y, video, small = E.load(args.dataset_features, args.min, args.ds_extra)
    thr, coverage, acc = threshold_for(blocks, weights, y, video, args.precision, args.C)
    print(f"held-out videos: accuracy {acc:.2f}; at {args.precision:.0%} precision the "
          f"threshold is {thr:.2f} and {coverage:.0%} of clips get sorted", flush=True)

    scalers = fit_scalers(blocks, weights)
    model = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced")
    model.fit(transform(blocks, weights, scalers), y)
    classes = model.classes_

    z = np.load(args.crop_features, allow_pickle=True)
    ex = np.load(args.crop_extra, allow_pickle=True)
    at_z = {n: i for i, n in enumerate(z["names"])}
    at_x = {n: i for i, n in enumerate(ex["names"])}
    names = [n for n in z["names"] if n in at_x]
    cb = {}
    for k in weights:
        if k in ("clip", "pose", "motion"):
            cb[k] = z[k][[at_z[n] for n in names]].astype(np.float32)
        else:
            cb[k] = ex[k][[at_x[n] for n in names]].astype(np.float32)
    print(f"crops with every feature: {len(names)}", flush=True)
    proba = model.predict_proba(transform(cb, weights, scalers))

    # per window: each class's best crop
    windows = defaultdict(list)
    for i, n in enumerate(names):
        ms = int(re.match(r"v\d+__(\d+)", n).group(1))
        windows[ms].append(i)
    times = sorted(windows)
    # the crop the classifier is surest about stands for the window, so the
    # window's scores stay a distribution the threshold was calibrated on
    pick = {t: windows[t][int(proba[windows[t]].max(1).argmax())] for t in times}
    wp = np.stack([proba[pick[t]] for t in times])
    best_crop = {t: {c: names[pick[t]] for c in classes} for t in times}
    # neighbours 2.5 s either side (the two cuts interleave at 2.5 s)
    sm = wp.copy()
    for k, t in enumerate(times):
        acc_p, wsum = wp[k] * 0.5, 0.5
        for dk in (-1, 1):
            j = k + dk
            if 0 <= j < len(times) and abs(times[j] - t) <= 2600:
                acc_p, wsum = acc_p + wp[j] * 0.25, wsum + 0.25
        sm[k] = acc_p / wsum
    sm = sm / sm.sum(1, keepdims=True)

    os.makedirs(args.out, exist_ok=True)
    rows, per_class = [], Counter()
    for k, t in enumerate(times):
        order = np.argsort(-sm[k])
        c1, c2 = classes[order[0]], classes[order[1]]
        conf = float(sm[k, order[0]])
        cls = c1 if conf >= thr else "_unsure"
        crop = best_crop[t][c1]
        folder = os.path.join(args.out, cls)
        os.makedirs(folder, exist_ok=True)
        dst = os.path.join(folder, crop)
        if not os.path.exists(dst):
            try:
                os.link(os.path.join(args.crops, crop), dst)
            except OSError:
                pass
        per_class[cls] += 1
        rows.append({"at": f"{int(t // 60000):02d}:{(t % 60000) / 1000:04.1f}", "ms": t,
                     "class": cls, "best": c1, "conf": round(conf, 3), "second": c2,
                     "second_conf": round(float(sm[k, order[1]]), 3), "crop": crop})
    with open(os.path.join(args.out, "timeline.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    changes = sum(1 for a, b in zip(rows, rows[1:]) if a["class"] != b["class"])

    group_mix = {}
    if args.groups and os.path.exists(args.groups):
        idx = {n: i for i, n in enumerate(names)}
        members = defaultdict(list)
        with open(args.groups, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["crop"] in idx:
                    members[r["group"]].append(idx[r["crop"]])
        for g, ids in sorted(members.items()):
            p = proba[ids].mean(0)
            top = np.argsort(-p)[:3]
            votes = Counter(classes[proba[ids].argmax(1)])
            group_mix[g] = {"crops": len(ids),
                            "mean_proba": {classes[i]: round(float(p[i]), 3) for i in top},
                            "votes": dict(votes.most_common(3)),
                            "agreement": round(votes.most_common(1)[0][1] / len(ids), 2)}
        with open(os.path.join(args.out, "group_classes.json"), "w", encoding="utf-8") as fh:
            json.dump(group_mix, fh, indent=1)

    if args.learned_groups:
        learned_groups(args, model, transform(cb, weights, scalers), names, classes, proba)

    summary = {"threshold": thr, "precision_target": args.precision, "coverage_on_dataset": coverage,
               "heldout_accuracy": acc, "windows": len(times), "label_changes": changes,
               "by_class": dict(per_class.most_common()), "classes_left_out": small, "mix": weights}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    if group_mix:
        for g, v in group_mix.items():
            print(g, v["crops"], "agreement", v["agreement"], v["votes"])


if __name__ == "__main__":
    sys.exit(main())
