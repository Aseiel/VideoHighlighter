"""Did adding a video to the dataset make the model better on the other videos?

The trainer's own held-out score cannot answer this. An added video becomes a
held-out fold of its own, scored by heads that never saw it, and it drags the
average down even when nothing got worse
(``docs/training/negative/judging-added-video-by-overall-score.md``). So the
question is asked the way ``docs/plans/2026-10-04-adding-reviewed-footage.md``
asked it:

* the folds are drawn over the dataset's **own** videos only, the same folds
  for both sides;
* per fold two heads are trained on the other videos: one without the added
  clips, one with them;
* both are scored on the same held-out clips. The added video is never
  scored, so nothing it teaches about itself counts.

Several seeds, because one seed moves a class by more than most additions
do. A change is only called one when every seed agrees on its direction, it
is large enough to matter (``OVERALL_MIN``, ``CLASS_MIN``) and it stands clear
of the seeds' own spread; anything else is reported as no clear change, which
is the honest name for it. The addition is worth keeping (``keep``) unless it
is worse overall, or more classes clearly lost than clearly gained.

    python -m model_training.action_head.compare --data-path <dataset> [--added <video> ...]

With no ``--added``, the videos recorded by ``modules.teach.intake`` are used.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Optional, Sequence

import numpy as np

BETTER, WORSE, SAME = "better", "worse", "no clear change"
WITHOUT, WITH = "without", "with"
KNOWN, FRESH = "classes the dataset has", "classes only this video has"
# Top-1 on held-out videos moves about a point between seeds on a few thousand
# clips; less than that is not a change.
OVERALL_MIN = 0.01
# A class's recall, on the few dozen held-out clips a class has, moves several
# points by chance.
CLASS_MIN = 0.05
# Held-out clips a class needs before its recall is reported at all.
CLASS_MIN_CLIPS = 5


def _direction(deltas: Sequence[float], minimum: float) -> str:
    """Better or worse only when every seed says so, the mean change is at
    least ``minimum``, and it is at least twice its own standard error: one
    seed that moved a lot is not a finding."""
    deltas = np.asarray(deltas, float)
    if not len(deltas):
        return SAME
    mean = deltas.mean()
    error = deltas.std(ddof=1) / np.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    if abs(mean) >= max(minimum, 2 * error):
        if (deltas > 0).all():
            return BETTER
        if (deltas < 0).all():
            return WORSE
    return SAME


def compare_features(x: np.ndarray, targets: np.ndarray, groups: np.ndarray,
                     added: np.ndarray, classes: Sequence[str], *,
                     seeds: int = 3, folds: int = 5, steps: int = 1500,
                     parts: Optional[dict] = None,
                     train_fn: Optional[Callable] = None,
                     predict_fn: Optional[Callable] = None,
                     folds_fn: Optional[Callable] = None,
                     log: Callable[[str], None] = print,
                     should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """The comparison on vectors already encoded.

    ``x`` [N, frames, dims], ``targets`` multi-hot [N, classes], ``groups`` the
    source video of each clip, ``added`` a bool mask of the clips in question.
    ``parts`` names bool masks over the added clips; each is also scored on
    its own (the dataset plus only that part), and ``drop`` lists the parts
    that measured worse, which is what says which of the added clips to
    take out again.
    ``train_fn(x, targets, n_classes, steps=, seed=)``, ``predict_fn(model, x)``
    and ``folds_fn(strat, groups, folds, seed)`` default to the trainer's own.
    """
    if train_fn is None or predict_fn is None:
        from model_training.action_head import head as H
        train_fn, predict_fn = train_fn or H.train_head, predict_fn or H.predict_proba
    if folds_fn is None:
        from model_training.action_head.train import group_folds as folds_fn
    added = np.asarray(added, bool)
    base = np.flatnonzero(~added)
    extra = np.flatnonzero(added)
    if not len(extra):
        raise ValueError("none of the dataset's clips belong to the added videos")
    if len(set(groups[base].tolist())) < 2:
        raise ValueError("the dataset needs clips from 2 or more other videos to compare on")
    n_classes = targets.shape[1]
    single = targets.sum(1) == 1
    strat = targets.argmax(1)
    y_all = targets.argmax(1)
    # Every side is trained on the fold's other videos plus its own clips:
    # none, all of the added ones, or one part of them.
    sides = {WITHOUT: np.array([], int), WITH: extra}
    for name, mask in (parts or {}).items():
        idx = np.flatnonzero(np.asarray(mask, bool) & added)
        if len(idx):
            sides[f"part:{name}"] = idx

    top1 = {side: [] for side in sides}
    top3 = {side: [] for side in sides}
    hits = {side: np.zeros((seeds, n_classes)) for side in sides}
    held = np.zeros(n_classes)
    for seed in range(seeds):
        splits = folds_fn(strat[base], groups[base], folds, seed)
        got = {side: [0, 0] for side in sides}
        n = 0
        for tr, te in splits:
            te_idx = base[te][single[base[te]]]
            y = y_all[te_idx]
            n += len(y)
            if seed == 0:
                held += np.bincount(y, minlength=n_classes)
            for side, own in sides.items():
                if should_stop and should_stop():
                    raise InterruptedError("stopped")
                tr_idx = np.concatenate([base[tr], own])
                model = train_fn(x[tr_idx], targets[tr_idx], n_classes, steps=steps, seed=seed)
                p = predict_fn(model, x[te_idx])
                right = p.argmax(1) == y
                got[side][0] += int(right.sum())
                got[side][1] += int((np.argsort(-p, 1)[:, :3] == y[:, None]).any(1).sum())
                hits[side][seed] += np.bincount(y[right], minlength=n_classes)
        for side in got:
            top1[side].append(got[side][0] / max(n, 1))
            top3[side].append(got[side][1] / max(n, 1))
        log(f"  seed {seed + 1}/{seeds}: top-1 without {top1[WITHOUT][-1]:.3f}, "
            f"with {top1[WITH][-1]:.3f} ({n} held-out clips)")

    def judged(side):
        deltas = np.array(top1[side]) - np.array(top1[WITHOUT])
        rows = []
        for k, name in enumerate(classes):
            if held[k] < CLASS_MIN_CLIPS:
                continue
            r0, r1 = hits[WITHOUT][:, k] / held[k], hits[side][:, k] / held[k]
            rows.append({
                "class": name, "held_out_clips": int(held[k]),
                "dataset_clips": int(targets[base, k].sum()),
                "dataset_videos": len(set(groups[base][targets[base, k] > 0].tolist())),
                "added_clips": int(targets[sides[side], k].sum()),
                "recall_without": round(float(r0.mean()), 4),
                "recall_with": round(float(r1.mean()), 4),
                "change": round(float((r1 - r0).mean()), 4),
                "verdict": _direction(r1 - r0, CLASS_MIN)})
        rows.sort(key=lambda row: row["change"])
        verdict = _direction(deltas, OVERALL_MIN)
        worse = [r["class"] for r in rows if r["verdict"] == WORSE]
        better = [r["class"] for r in rows if r["verdict"] == BETTER]
        return {"verdict": verdict, "keep": verdict != WORSE and len(worse) <= len(better),
                "summary": _sentence(verdict, deltas, better, worse),
                "top1_change": round(float(deltas.mean()), 4),
                "top1_change_per_seed": [round(float(d), 4) for d in deltas],
                "classes_better": better, "classes_worse": worse, "per_class": rows}

    whole = judged(WITH)
    unscored = [name for k, name in enumerate(classes)
                if targets[extra, k].sum() and not targets[base, k].sum()]
    result = {
        **whole,
        "top1_without": _stat(top1[WITHOUT]), "top1_with": _stat(top1[WITH]),
        "top3_without": _stat(top3[WITHOUT]), "top3_with": _stat(top3[WITH]),
        "only_in_added": unscored,
        "clips": {"dataset": int(len(base)), "added": int(len(extra)),
                  "added_share": round(len(extra) / len(x), 4)},
        "how": {"seeds": seeds, "folds": folds, "steps": steps,
                "scored_on": "held-out clips of the dataset's other videos; "
                             "the added videos are never scored"}}
    if len(sides) > 2:
        result["parts"] = []
        for side in sides:
            if side.startswith("part:"):
                one = judged(side)
                one.pop("per_class")
                result["parts"].append({"part": side[5:], "clips": int(len(sides[side])), **one})
        result["parts"].sort(key=lambda row: row["top1_change"])
        result["drop"] = [row["part"] for row in result["parts"] if not row["keep"]]
        if result["drop"]:
            result["summary"] += (" Parts that measured worse on their own: "
                                  + ", ".join(result["drop"]) + ".")
    return result


def _stat(values) -> dict:
    values = np.asarray(values, float)
    return {"mean": round(float(values.mean()), 4), "std": round(float(values.std()), 4)}


def _sentence(verdict, deltas, better, worse) -> str:
    change = f"{deltas.mean() * 100:+.1f} points top-1 on the other videos"
    if verdict == WORSE:
        text = f"Worse: {change}, in every seed. Do not train on this as it is."
    elif verdict == BETTER:
        text = f"Better: {change}, in every seed."
    else:
        every = ("lower in every seed, but " if (deltas < 0).all() else
                 "higher in every seed, but " if (deltas > 0).all() else "")
        text = f"No clear change overall ({change}: {every}not clear of the seeds' spread)."
    if better:
        text += f" Clearly gained ({len(better)}): " + ", ".join(better) + "."
    if worse:
        text += f" Clearly lost ({len(worse)}): " + ", ".join(worse) + "."
    return text


def load(data_path: str, added: Optional[Sequence[str]] = None, *,
         aliases: Optional[str] = None, min_clips: int = 5, frames: int = 4,
         backend: Optional[str] = None, cache: Optional[str] = None,
         log: Callable[[str], None] = print,
         should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """The dataset as the trainer sees it (same clips, classes, encoder and
    feature cache), with the mask of the clips that belong to ``added``."""
    from model_training.action_head import features as Fx
    from model_training.action_head.train import default_cache, select_clips
    from modules.teach import intake
    from modules.teach.benchmark import load_aliases, read_dataset
    from modules.vision import frame_encoder

    added = list(added or intake.added_videos(data_path))
    if not added:
        raise ValueError("say which videos were added (--added <video>): this dataset has "
                         "no record of one")
    data = read_dataset(data_path, aliases=load_aliases(aliases))
    clips, classes, notes = select_clips(data["clips"], min_clips)
    for note in notes:
        log(f"ℹ️ {note}")
    if not classes:
        raise ValueError("no class has enough single-action clips to train on")
    encoder = frame_encoder.load(backend, log=log)
    if encoder is None:
        raise RuntimeError("the frame encoder is not available; see the lines above")
    store = Fx.FeatureCache(cache or default_cache(data_path, encoder.encoder_id, frames),
                            encoder.encoder_id, frames, encoder.dims)
    x, ok = Fx.encode_clips([c.path for c in clips], data_path, encoder, store,
                            log=log, should_stop=should_stop)
    clips = [c for c, good in zip(clips, ok) if good]
    index = {name: k for k, name in enumerate(classes)}
    targets = np.zeros((len(clips), len(classes)), np.float32)
    for i, clip in enumerate(clips):
        targets[i, [index[lb] for lb in clip.labels]] = 1
    groups = np.array([c.group for c in clips])
    return {"x": x[ok], "targets": targets, "groups": groups, "classes": classes,
            "added": np.isin(groups, added), "added_videos": added,
            "folders": np.array([c.folder for c in clips]), "clips": clips}


def split_parts(d: dict, how: Optional[str]) -> dict:
    """Masks over the added clips, by ``how``:

    * ``"kinds"``: clips of classes the dataset already has, against clips
      that show a class only the added videos have. The second kind cannot be
      scored at all, so it is the first suspect when an addition costs.
    * ``"folders"``: one part per class folder. Thorough, and slow: every
      folder is another head per fold and seed.
    """
    if not how:
        return {}
    added, targets = d["added"], d["targets"]
    if how == "folders":
        return {name: added & (d["folders"] == name)
                for name in sorted(set(d["folders"][added].tolist()))}
    if how == "kinds":
        known = targets[~added].sum(0) > 0
        fresh = added & (targets[:, ~known].sum(1) > 0)
        return {KNOWN: added & ~fresh, FRESH: fresh}
    raise ValueError(f"parts is 'kinds' or 'folders', not {how!r}")


def compare(data_path: str, added: Optional[Sequence[str]] = None, *, seeds: int = 3,
            folds: int = 5, steps: int = 1500, parts: Optional[str] = None,
            log: Callable[[str], None] = print,
            should_stop: Optional[Callable[[], bool]] = None, **load_options) -> dict:
    """Read, encode (cached) and compare. See the module's text."""
    d = load(data_path, added, log=log, should_stop=should_stop, **load_options)
    log(f"Comparing on {int((~d['added']).sum())} clips of the dataset's other videos, "
        f"with and without {int(d['added'].sum())} clips of: {', '.join(d['added_videos'])}")
    result = compare_features(d["x"], d["targets"], d["groups"], d["added"], d["classes"],
                              seeds=seeds, folds=folds, steps=steps, log=log,
                              parts=split_parts(d, parts), should_stop=should_stop)
    if parts == "kinds" and FRESH in result.get("drop", []):
        fresh = d["added"] & split_parts(d, "kinds")[FRESH]
        result["drop_folders"] = sorted(set(d["folders"][fresh].tolist()))
    elif parts == "folders":
        result["drop_folders"] = list(result.get("drop", []))
    result["dataset"] = os.path.abspath(data_path)
    result["added_videos"] = d["added_videos"]
    return result


def main(argv=None) -> int:
    """0 when the addition is worth keeping, 1 when it measured worse, 2 on a
    problem it explained."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--added", nargs="*", default=None,
                    help="names of the added source videos (default: those intake recorded)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--min-clips", type=int, default=5)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--cache", default=None, help="feature cache file (.npz)")
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--parts", choices=("kinds", "folders"), default=None,
                    help="also score parts of the added clips on their own")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args(argv)
    from model_training.action_head.train import _utf8_stdout
    _utf8_stdout()
    try:
        result = compare(args.data_path, args.added, seeds=args.seeds, folds=args.folds,
                         steps=args.steps, parts=args.parts, min_clips=args.min_clips, frames=args.frames,
                         backend=args.backend, cache=args.cache, aliases=args.aliases)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"❌ {exc}")
        return 2
    print()
    print(result["summary"])
    print(f"top-1 without {result['top1_without']['mean']:.3f} ± {result['top1_without']['std']:.3f}"
          f" | with {result['top1_with']['mean']:.3f} ± {result['top1_with']['std']:.3f}")
    print("class: recall without -> with (held-out clips; dataset clips/videos, added)")
    for row in result["per_class"]:
        if row["added_clips"] or row["verdict"] != SAME:
            mark = {BETTER: "+", WORSE: "-", SAME: " "}[row["verdict"]]
            print(f" {mark} {row['class']}: {row['recall_without']:.2f} -> "
                  f"{row['recall_with']:.2f} ({row['held_out_clips']}; "
                  f"{row['dataset_clips']}/{row['dataset_videos']}, +{row['added_clips']})")
    for row in result.get("parts", []):
        print(f" part '{row['part']}' ({row['clips']} clips) alone: "
              f"{row['top1_change'] * 100:+.1f} points, {row['verdict']}"
              + ("" if row["keep"] else "  <- drop"))
    if result["only_in_added"]:
        print("Only in the added videos, so not scored: " + ", ".join(result["only_in_added"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
    return 0 if result["keep"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
