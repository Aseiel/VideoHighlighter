"""Train a taught-action head on a dataset of sorted clips.

    python -m model_training.action_head.train --data-path <dataset>
        [--out <folder>] [--name taught-actions] [--frames 4] [--folds 5]
        [--steps 750,1500,3000] [--min-clips 5] [--precision 0.7]
        [--backend auto|intel|directml|cpu] [--cache <file.npz>] [--aliases <file.json>]

The dataset is the app's layout: ``train/``, ``val/`` (and ``test/``) holding
one folder per class, clips directly inside. A folder named ``a_b`` shows two
classes. The head names one action per clip, so it trains on single-class
clips; clips of two classes in ``test/`` are the confusion test (step 5).

What it does:

1. **Encodes every clip once** (4 frames across it, ``features.py``), cached.
2. **Scores on unseen source videos.** ``train/`` and ``val/`` are pooled and
   split by source video (the clip name before ``_temp``/``_highlight``), five
   ways: neighbouring clips of one video share scene, people and light, so a
   split by clip rewards remembering the scene. Each fold trains a head on the
   other videos and predicts its own. The out-of-fold predictions choose the
   training length (``--steps``) and give every number this prints.
3. **Sets each class's trust threshold** from those predictions
   (``trust.trust_thresholds``): the confidence above which that class is
   right ``--precision`` of the time, at 80 % confidence. A class that never
   gets there is only ever a suggestion.
4. **Trains the saved head on every clip** with the chosen length.
5. **Scores ``test/``** with the fold heads, each clip by a head that never saw
   its source video. A clip of two classes counts as found when the head's top
   two guesses are exactly its two classes.
6. **Scores ``val/`` as the dataset defines it** (trained on ``train/`` only),
   next to how many of its clips share a source video with ``train/``: that
   share is how much of the score is remembering the scene.

Writes ``head.onnx`` and ``head.json`` (encoder id, frames, classes,
thresholds, held-out scores) into the output folder. Nothing in them names a
file or holds a frame.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

HEAD_FORMAT = 1
HEAD_FILE = "head.onnx"
META_FILE = "head.json"
DEFAULT_NAME = "taught-actions"


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - a stream without reconfigure is fine as is
            pass


def default_cache(data_path: str, encoder_id: str, frames: int) -> str:
    from modules.system import app_paths
    digest = hashlib.sha1(os.path.abspath(data_path).lower().encode("utf-8")).hexdigest()[:12]
    return os.path.join(app_paths.user_data_dir(), "cache", "action_head",
                        f"{digest}-{encoder_id}-{frames}f.npz")


def default_out(name: str) -> str:
    from modules.system import app_paths
    return os.path.join(app_paths.action_models_dir(), name)


def select_clips(clips, min_clips: int):
    """Single-label clips of classes with ``min_clips`` or more. Returns
    ``(kept, notes)``, notes being sentences for the log."""
    notes = []
    pool = [c for c in clips if c.split in ("train", "val")]
    multi = [c for c in pool if len(c.labels) != 1]
    if multi:
        notes.append(f"{len(multi)} clips in folders with two or more classes are left out "
                     f"(the head names one action per clip)")
    single = [c for c in pool if len(c.labels) == 1]
    counts = Counter(c.labels[0] for c in single)
    small = sorted(k for k, v in counts.items() if v < min_clips)
    if small:
        notes.append(f"{len(small)} classes have fewer than {min_clips} clips and are left out: "
                     + ", ".join(f"{k} ({counts[k]})" for k in small))
    return [c for c in single if counts[c.labels[0]] >= min_clips], notes


def group_folds(y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> list:
    """``folds`` (train, held-out) index pairs; no source video on both sides."""
    from sklearn.model_selection import StratifiedGroupKFold
    folds = max(2, min(folds, len(set(groups.tolist()))))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def out_of_fold(x, y, groups, n_classes, splits, steps, seed, log,
                x_extra=None, groups_extra=None):
    """Held-out probabilities for every clip, and for ``x_extra`` (test clips)
    the mean over the fold heads that never saw the clip's source video."""
    from model_training.action_head import head as H
    proba = np.zeros((len(y), n_classes), np.float32)
    n_extra = 0 if x_extra is None else len(x_extra)
    extra_sum = np.zeros((n_extra, n_classes), np.float32)
    extra_n = np.zeros(n_extra)
    for i, (tr, te) in enumerate(splits, 1):
        model = H.train_head(x[tr], y[tr], n_classes, steps=steps, seed=seed)
        proba[te] = H.predict_proba(model, x[te])
        if n_extra:
            unseen = ~np.isin(groups_extra, groups[tr])
            if unseen.any():
                extra_sum[unseen] += H.predict_proba(model, x_extra[unseen])
                extra_n[unseen] += 1
        log(f"    fold {i}/{len(splits)}: {np.mean(proba[te].argmax(1) == y[te]):.3f} "
            f"on {len(te)} clips")
    extra = extra_sum / np.maximum(extra_n, 1)[:, None]
    extra[extra_n == 0] = np.nan
    return proba, extra


def score_test(proba: np.ndarray, test_clips, classes, thresholds) -> dict:
    """Test clips, scored by heads that never saw their source video.

    Single-class clips: accuracy. Two-class clips (the confusion test): how
    often the top guess is one of the two, how often the top two are exactly
    the two, and what trust would do with them.
    """
    from model_training.action_head import trust
    index = {c: k for k, c in enumerate(classes)}
    out: dict = {}
    scored = ~np.isnan(proba).any(1)
    known = np.array([all(lb in index for lb in c.labels) for c in test_clips], bool)
    n_labels = np.array([len(c.labels) for c in test_clips])
    m = scored & known & (n_labels == 1)
    if m.any():
        y1 = np.array([index[c.labels[0]] for c, keep in zip(test_clips, m) if keep])
        out["single"] = {"clips": int(m.sum()),
                         "accuracy": round(float(np.mean(proba[m].argmax(1) == y1)), 4)}
    m = scored & known & (n_labels == 2)
    unknown = int((~known & (n_labels == 2)).sum())
    if m.any():
        pairs = [{index[lb] for lb in c.labels} for c, keep in zip(test_clips, m) if keep]
        p = proba[m]
        top2 = np.argsort(-p, 1)[:, :2]
        first_right = np.array([t[0] in pr for t, pr in zip(top2, pairs)], bool)
        both = np.array([set(t) == pr for t, pr in zip(top2, pairs)], bool)
        trusted = trust.trusted_mask(p, thresholds)
        out["two_actions"] = {
            "clips": int(m.sum()),
            "left_out_unknown_class": unknown,
            "top1_is_one_of_them": round(float(first_right.mean()), 4),
            "top2_are_both": round(float(both.mean()), 4),
            "trusted_share": round(float(trusted.mean()), 4),
            "trusted_into_one_of_them": (round(float(first_right[trusted].mean()), 4)
                                         if trusted.any() else None),
        }
    elif unknown:
        out["two_actions"] = {"clips": 0, "left_out_unknown_class": unknown}
    return out


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="Train a taught-action head on the frame encoder")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out", default=None, help="output folder (default: models/actions/<name>)")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", default="750,1500,3000",
                    help="training lengths to compare on held-out videos")
    ap.add_argument("--min-clips", type=int, default=5)
    ap.add_argument("--precision", type=float, default=0.7,
                    help="held-out precision a class must reach to be trusted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=None, help="encoder route (compute.backend value)")
    ap.add_argument("--cache", default=None, help="feature cache file (.npz)")
    ap.add_argument("--aliases", default=None,
                    help="JSON of folder or class name -> class name (\"\" leaves it out), as teach uses")
    args = ap.parse_args(argv)
    log = print

    from model_training.action_head import features as Fx
    from model_training.action_head import head as H
    from model_training.action_head import trust
    from modules.teach.benchmark import load_aliases, read_dataset
    from modules.vision import frame_encoder

    started = time.time()
    data = read_dataset(args.data_path, aliases=load_aliases(args.aliases))
    clips, notes = select_clips(data["clips"], args.min_clips)
    test_clips = [c for c in data["clips"] if c.split == "test"]
    for note in notes:
        log(f"ℹ️ {note}")
    if not clips:
        log("❌ No class has enough single-label clips to train on")
        return 1

    encoder = frame_encoder.load(args.backend, log=log)
    if encoder is None:
        log("❌ The frame encoder is not available; see the lines above")
        return 1
    cache = Fx.FeatureCache(args.cache or default_cache(args.data_path, encoder.encoder_id, args.frames),
                            encoder.encoder_id, args.frames, encoder.dims)
    x, ok = Fx.encode_clips([c.path for c in clips + test_clips], args.data_path, encoder,
                            cache, log=log)
    x_test, ok_test = x[len(clips):], ok[len(clips):]
    x, ok = x[:len(clips)], ok[:len(clips)]
    clips = [c for c, good in zip(clips, ok) if good]
    x = x[ok]
    test_clips = [c for c, good in zip(test_clips, ok_test) if good]
    x_test = x_test[ok_test]
    g_test = np.array([c.group for c in test_clips])
    classes = sorted({c.labels[0] for c in clips})
    y = np.array([classes.index(c.labels[0]) for c in clips])
    groups = np.array([c.group for c in clips])
    n_videos = len(set(groups.tolist()))
    log(f"\n{len(clips)} clips, {len(classes)} classes, {n_videos} source videos")

    splits = group_folds(y, groups, args.folds, args.seed)
    log(f"Scoring on unseen source videos ({len(splits)} folds)")
    best = None
    for steps in [int(s) for s in str(args.steps).split(",") if s.strip()]:
        log(f"  {steps} steps")
        proba, proba_test = out_of_fold(x, y, groups, len(classes), splits, steps, args.seed,
                                        log, x_extra=x_test, groups_extra=g_test)
        acc = float(np.mean(proba.argmax(1) == y))
        log(f"  {steps} steps: held-out accuracy {acc:.3f}")
        if best is None or acc > best[1] + 1e-9:
            best = (steps, acc, proba, proba_test)
    steps, acc, proba, proba_test = best
    pred = proba.argmax(1)
    thresholds = trust.trust_thresholds(proba, y, target=args.precision)
    trusted = trust.trusted_mask(proba, thresholds)
    sorted_precision = float(np.mean(pred[trusted] == y[trusted])) if trusted.any() else 0.0
    test_scores = score_test(proba_test, test_clips, classes, thresholds) if test_clips else {}

    in_train = np.array([c.split == "train" for c in clips], bool)
    val_scores = None
    if in_train.any() and (~in_train).any():
        log(f"\nScoring val/ as the dataset defines it (training on train/ only)")
        model = H.train_head(x[in_train], y[in_train], len(classes), steps=steps, seed=args.seed)
        val_pred = H.predict_proba(model, x[~in_train]).argmax(1)
        shared = np.isin(groups[~in_train], groups[in_train])
        val_scores = {"clips": int((~in_train).sum()),
                      "accuracy": round(float(np.mean(val_pred == y[~in_train])), 4),
                      "clips_sharing_a_video_with_train": int(shared.sum())}

    log(f"\nTraining the saved head on all {len(clips)} clips ({steps} steps)")
    model = H.train_head(x, y, len(classes), steps=steps, seed=args.seed)
    out = args.out or default_out(args.name)
    os.makedirs(out, exist_ok=True)
    head_path = os.path.join(out, HEAD_FILE)
    H.export_onnx(model, head_path, args.frames)
    check = H.onnx_proba(H.load_onnx_session(head_path), x[:64])
    drift = float(np.abs(check - H.predict_proba(model, x[:64])).max())
    if drift > 1e-4:
        log(f"❌ The exported head disagrees with the trained one (max {drift:.2e})")
        return 1

    per_class = {}
    for k, name in enumerate(classes):
        mine, said = y == k, pred == k
        per_class[name] = {
            "clips": int(mine.sum()),
            "videos": int(len(set(groups[mine].tolist()))),
            "heldout_recall": round(float(np.mean(pred[mine] == k)), 3),
            "heldout_precision": (round(float(np.mean(y[said] == k)), 3) if said.any() else None),
            "trust_threshold": (None if thresholds[k] is None else round(thresholds[k], 4)),
        }
    meta = {
        "format": HEAD_FORMAT,
        "kind": "action-head",
        "encoder": encoder.encoder_id,
        "frames": args.frames,
        "input": f"features [N, frames, {encoder.dims}]: frame encoder vectors, "
                 f"frames evenly across the clip",
        "output": "logits [N, classes]; softmax gives probabilities",
        "classes": classes,
        "trust_thresholds": [None if t is None else round(t, 4) for t in thresholds],
        "trust_precision": args.precision,
        "steps": steps,
        "heldout": {
            "how": f"{len(splits)} folds by source video",
            "clips": len(clips), "videos": n_videos,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(trust.balanced_accuracy(pred, y), 4),
            "top3": round(trust.top_k_hit(proba, y, 3), 4),
            "trusted_share": round(float(trusted.mean()), 4),
            "trusted_precision": round(sorted_precision, 4),
        },
        "test": test_scores,
        "val_folder": val_scores,
        "per_class": per_class,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)

    h = meta["heldout"]
    log(f"\nHeld out (whole source videos never seen): accuracy {h['accuracy']:.3f}, "
        f"balanced {h['balanced_accuracy']:.3f}, top-3 {h['top3']:.3f}")
    n_trusted = sum(t is not None for t in thresholds)
    log(f"Trusted classes: {n_trusted} of {len(classes)}. Above their thresholds "
        f"{h['trusted_share']:.0%} of held-out clips are sorted, "
        f"{h['trusted_precision']:.0%} of them correctly")
    if val_scores:
        log(f"val/ as the dataset defines it (trained on train/ only): accuracy "
            f"{val_scores['accuracy']:.3f} on {val_scores['clips']} clips, "
            f"{val_scores['clips_sharing_a_video_with_train']} of which share a source video "
            f"with train/")
    if "single" in test_scores:
        t = test_scores["single"]
        log(f"test/, single action: accuracy {t['accuracy']:.3f} on {t['clips']} clips")
    t = test_scores.get("two_actions")
    if t and t["clips"]:
        log(f"test/, two actions ({t['clips']} clips, each scored by heads that never saw "
            f"its source video): top guess is one of the two {t['top1_is_one_of_them']:.0%}, "
            f"top two are exactly the two {t['top2_are_both']:.0%}")
        if t["trusted_into_one_of_them"] is not None:
            log(f"  trusted: {t['trusted_share']:.0%} would be sorted, "
                f"{t['trusted_into_one_of_them']:.0%} of those into one of their two actions")
    if t and t["left_out_unknown_class"]:
        log(f"  {t['left_out_unknown_class']} two-action clips name a class the head does "
            f"not have and are not scored")
    width = max(len(c) for c in classes)
    log(f"\n{'class':{width}}  clips videos recall precision threshold")
    for name, row in sorted(per_class.items(), key=lambda kv: -kv[1]["clips"]):
        prec = "-" if row["heldout_precision"] is None else f"{row['heldout_precision']:.2f}"
        th = "not trusted" if row["trust_threshold"] is None else f"{row['trust_threshold']:.2f}"
        log(f"{name:{width}}  {row['clips']:5} {row['videos']:6} {row['heldout_recall']:6.2f} "
            f"{prec:>9} {th:>9}")
    log(f"\n✅ Saved {head_path} and {META_FILE} ({time.time() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
