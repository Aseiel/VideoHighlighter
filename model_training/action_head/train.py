"""Train a taught-action head on a dataset of sorted clips.

    python -m model_training.action_head.train --data-path <dataset>
        [--out <folder>] [--name taught-actions] [--frames 4] [--folds 5]
        [--steps 750,1500,3000] [--min-clips 5] [--precision 0.7]
        [--backend auto|intel|directml|cpu] [--cache <file.npz>]
        [--aliases <file.json>] [--teach-test] [--min-videos 3]
        [--finetune-blocks 4 [--ft-epochs 10] [--ft-lr 1e-4] [--ft-decay 0.8]
         [--ft-head-lr 2e-4] [--ft-batch 16] [--device auto|xpu|cuda|cpu]
         [--frame-cache <folder>]]

The dataset is the app's layout: ``train/``, ``val/`` (and ``test/``) holding
one folder per class, clips directly inside. A folder named ``a_b`` shows both
``a`` and ``b``: the head scores every action on its own, so such clips teach
it that two actions can be present together. It only learns the
combinations it is shown; measured, taught combinations are found far more
often than ones it never saw.

What it does:

1. **Encodes every clip once** (4 frames across it, ``features.py``), cached.
2. **Scores on unseen source videos.** The clips are split by source video
   (the clip name before ``_temp``/``_highlight``), five ways: neighbouring
   clips of one video share scene, people and light, so a split by clip
   rewards remembering the scene. Each fold trains a head on the other videos
   and scores its own. The out-of-fold scores choose the training length
   (``--steps``) and give every number this prints.
3. **Sets each action's trust threshold** from those scores
   (``trust.trust_thresholds``): the score above which "this action is
   present" is right ``--precision`` of the time, at 80 % confidence, with
   hits from ``--min-videos`` source videos or more. An action that never
   gets there is only ever a suggestion. Each pair of actions taught in 5 or
   more clips gets a threshold of its own, on the lower of its two scores.
4. **Trains the saved head on every clip** with the chosen length.
5. **Scores ``test/``.** Without ``--teach-test`` it is not trained on: each
   clip is scored by the fold heads that never saw its source video. With
   ``--teach-test`` its clips join the folds (scored the same honest way) and
   the saved head learns from them too.
6. **Scores ``val/`` as the dataset defines it** (trained on ``train/`` only),
   next to how many of its clips share a source video with ``train/``: that
   share is how much of the score is remembering the scene.

Writes ``head.onnx`` and ``head.json`` (encoder id, frames, classes,
thresholds, held-out scores) into the output folder. Nothing in them names a
file or holds a frame.

**Fine-tuning the image model** (``--finetune-blocks N``, a graphics card;
``finetune.py``). The steps above run first, unchanged, and become the
baseline. Every clip is also decoded once into a frame cache (``frames.py``,
kept for the next run). Then each fold's frozen head is the start of an LP-FT
run that trains the top N blocks of the image tower with it, scored on the
fold's unseen videos after every epoch. The epoch with the best held-out
accuracy over all folds is the length of the final run, and the trust
thresholds and every printed number come from the fine-tuned held-out scores
at that epoch. The fine-tuned model (its own ``vision.onnx``, ~186 MB) is
saved only when it beats the frozen head on the same folds; otherwise the
frozen head is, and the log says why (``tower.py`` writes and checks it).
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
from types import SimpleNamespace
from typing import Callable, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

HEAD_FORMAT = 1
HEAD_FILE = "head.onnx"
META_FILE = "head.json"
DEFAULT_NAME = "taught-actions"
POOL_SPLITS = ("train", "val")
DEFAULT_STEPS = "750,1500,3000"   # training lengths compared on held-out videos
STOPPED = 130          # main()'s return code when ``should_stop`` ended the run
SAVED_FINETUNED = "Saved model: fine-tuned image model and head"
SAVED_FROZEN = "Saved model: head on the shared image model"


class Stopped(Exception):
    """``should_stop`` said so: the run ends between two steps, saving nothing."""


def _check(should_stop) -> None:
    if should_stop is not None and should_stop():
        raise Stopped()


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


def default_frame_cache(data_path: str) -> str:
    from model_training.action_head import frames as FR
    from modules.system import app_paths
    digest = hashlib.sha1(os.path.abspath(data_path).lower().encode("utf-8")).hexdigest()[:12]
    return os.path.join(app_paths.user_data_dir(), "cache", "action_head",
                        f"{digest}-frames{FR.SLOTS}")


def default_out(name: str) -> str:
    from modules.system import app_paths
    return os.path.join(app_paths.action_models_dir(), name)


def select_clips(clips, min_clips: int, splits=POOL_SPLITS):
    """The clips to train on, from ``splits``: every clip whose actions are
    all classes with ``min_clips`` or more single-action clips. Returns
    ``(kept, classes, notes)``, notes being sentences for the log."""
    notes = []
    pool = [c for c in clips if c.split in splits]
    counts = Counter(c.labels[0] for c in pool if len(c.labels) == 1)
    classes = sorted(k for k, v in counts.items() if v >= min_clips)
    known = set(classes)
    small = sorted(k for k, v in counts.items() if v < min_clips)
    if small:
        notes.append(f"{len(small)} classes have fewer than {min_clips} single-action clips "
                     f"and are left out: " + ", ".join(f"{k} ({counts[k]})" for k in small))
    kept = [c for c in pool if c.labels and set(c.labels) <= known]
    multi = sum(len(c.labels) > 1 for c in kept)
    if multi:
        notes.append(f"{multi} clips show two or more actions and teach them together")
    unknown = [c for c in pool if not set(c.labels) <= known]
    if unknown:
        notes.append(f"{len(unknown)} clips name an action that is not a class here "
                     f"and are left out")
    return kept, classes, notes


def group_folds(strat: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> list:
    """``folds`` (train, held-out) index pairs; no source video on both sides."""
    from sklearn.model_selection import StratifiedGroupKFold
    folds = max(2, min(folds, len(set(groups.tolist()))))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(strat)), strat, groups))


def out_of_fold(x, targets, groups, splits, steps, seed, log,
                x_extra=None, groups_extra=None, should_stop=None, models=None):
    """Held-out scores for every clip, and for ``x_extra`` (test clips) the
    mean over the fold heads that never saw the clip's source video (NaN when
    none qualifies). Each fold's head is appended to ``models`` when given."""
    from model_training.action_head import head as H
    n_classes = targets.shape[1]
    scores = np.zeros((len(targets), n_classes), np.float32)
    n_extra = 0 if x_extra is None else len(x_extra)
    extra_sum = np.zeros((n_extra, n_classes), np.float32)
    extra_n = np.zeros(n_extra)
    for i, (tr, te) in enumerate(splits, 1):
        _check(should_stop)
        model = H.train_head(x[tr], targets[tr], n_classes, steps=steps, seed=seed)
        if models is not None:
            models.append(model)
        scores[te] = H.predict_proba(model, x[te])
        if n_extra:
            unseen = ~np.isin(groups_extra, groups[tr])
            if unseen.any():
                extra_sum[unseen] += H.predict_proba(model, x_extra[unseen])
                extra_n[unseen] += 1
        one = targets[te].sum(1) == 1
        acc = np.mean(scores[te][one].argmax(1) == targets[te][one].argmax(1)) if one.any() else 0.0
        log(f"    fold {i}/{len(splits)}: {acc:.3f} on {int(one.sum())} single-action clips")
    extra = extra_sum / np.maximum(extra_n, 1)[:, None]
    extra[extra_n == 0] = np.nan
    return scores, extra


def _log_scores(log, title: str, s: dict) -> None:
    if "single" in s:
        t = s["single"]
        log(f"{title}, single action ({t['clips']} clips): accuracy {t['accuracy']:.3f}, "
            f"balanced {t['balanced_accuracy']:.3f}, top-3 {t['top3']:.3f}")
        if t["trusted_precision"] is not None:
            log(f"  trusted: {t['trusted_share']:.0%} sorted, {t['trusted_precision']:.0%} "
                f"of those correctly")
    if "two_actions" in s:
        t = s["two_actions"]
        log(f"{title}, two actions ({t['clips']} clips): both in the top 2 "
            f"{t['both_in_top2']:.0%}, both in the top 5 {t['both_in_top5']:.0%}, "
            f"top one of them {t['top1_is_one_of_them']:.0%}")
        log(f"  trusted: both detected {t['both_detected']:.0%}, one {t['one_detected']:.0%}, "
            f"something else detected too {t['wrong_detected']:.0%}")


def main(argv=None, *, log: Callable[[str], None] = print,
         should_stop: Optional[Callable[[], bool]] = None) -> int:
    """The command line, callable in-process too: the app's Train > Actions
    runs it on a worker thread (``log`` gets every line, ``should_stop`` is
    asked between steps). Returns 0 when saved, 1 on a problem it explained,
    ``STOPPED`` when stopped."""
    ap = argparse.ArgumentParser(description="Train a taught-action head on the frame encoder")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out", default=None, help="output folder (default: models/actions/<name>)")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", default=DEFAULT_STEPS,
                    help="training lengths to compare on held-out videos")
    ap.add_argument("--min-clips", type=int, default=5)
    ap.add_argument("--precision", type=float, default=0.7,
                    help="held-out precision an action must reach to be trusted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=None, help="encoder route (compute.backend value)")
    ap.add_argument("--cache", default=None, help="feature cache file (.npz)")
    ap.add_argument("--aliases", default=None,
                    help='JSON of folder or class name -> class name ("" leaves it out), as teach uses')
    ap.add_argument("--min-videos", type=int, default=3,
                    help="source videos a trusted action's held-out hits must come from")
    ap.add_argument("--teach-test", action="store_true",
                    help="also learn from test/ (still scored only by heads that never saw each video)")
    ft = ap.add_argument_group("fine-tuning the image model (a graphics card)")
    ft.add_argument("--finetune-blocks", type=int, default=0,
                    help="also train the image tower's top N blocks (0: the frozen head only)")
    ft.add_argument("--ft-epochs", type=int, default=10,
                    help="epochs compared on held-out videos; the best is the final length")
    ft.add_argument("--ft-lr", type=float, default=1e-4, help="the top block's learning rate")
    ft.add_argument("--ft-decay", type=float, default=0.8,
                    help="each block below the top trains at this times the one above")
    ft.add_argument("--ft-head-lr", type=float, default=2e-4)
    ft.add_argument("--ft-batch", type=int, default=16, help="clips per training step")
    ft.add_argument("--device", default="auto", help="auto (Intel, then NVIDIA), xpu, cuda or cpu")
    ft.add_argument("--frame-cache", default=None,
                    help="folder for the decoded frames (default: in the app's data folder)")
    args = ap.parse_args(argv)
    try:
        return _train(args, log, should_stop)
    except Stopped:
        log("⏹️ Stopped; nothing was saved")
        return STOPPED


def _evaluate(scores, scores_extra, targets, label_sets, groups, is_test, extra, index, args):
    """Trust thresholds from held-out ``scores``, and what they score: the
    held-out clips, and test/ (taught, or scored by heads that never saw it)."""
    from model_training.action_head import trust
    thresholds = trust.trust_thresholds(scores, targets, target=args.precision,
                                        groups=groups, min_videos=args.min_videos)
    pairs = trust.taught_pairs(targets)
    pair_th = trust.pair_thresholds(scores, targets, pairs, target=args.precision,
                                    groups=groups, min_videos=args.min_videos)
    heldout = trust.score_clips(scores[~is_test],
                                [s for s, t in zip(label_sets, is_test) if not t], thresholds,
                                pairs, pair_th)
    if args.teach_test:
        test_scores = (trust.score_clips(scores[is_test],
                                         [s for s, t in zip(label_sets, is_test) if t], thresholds,
                                         pairs, pair_th)
                       if is_test.any() else {})
    elif extra:
        scored = ~np.isnan(scores_extra).any(1)
        test_scores = trust.score_clips(scores_extra[scored],
                                        [{index[lb] for lb in c.labels}
                                         for c, s in zip(extra, scored) if s], thresholds,
                                        pairs, pair_th)
    else:
        test_scores = {}
    return {"thresholds": thresholds, "pairs": pairs, "pair_th": pair_th,
            "heldout": heldout, "test": test_scores}


def _epochs(n: int) -> str:
    return f"{n} epoch" + ("" if n == 1 else "s")


def _single_accuracy(scores, targets, single) -> float:
    return float(np.mean(scores[single].argmax(1) == targets[single].argmax(1))) if single.any() else 0.0


def _encode_from_frames(frame_cache, keys, encoder, cache, log, should_stop) -> None:
    """Frozen vectors from the frame cache's scoring frames, for clips the
    feature cache does not have yet: the frames are already decoded, and the
    vectors are the ones the frozen trainer makes (``preprocess`` is
    ``scale_pixels(prepare_frame)``), so a later frozen run reuses them."""
    from model_training.action_head import frames as FR
    todo = [k for k in keys if k not in cache and frame_cache.ok[frame_cache.row(k)]]
    if not todo:
        return
    log(f"Encoding {len(todo)} clips ({len(keys) - len(todo)} cached) "
        f"with {encoder.encoder_id} on {encoder.label}")
    started, chunk = time.time(), 64
    for n in range(0, len(todo), chunk):
        _check(should_stop)
        part = todo[n:n + chunk]
        rgb = frame_cache.read([frame_cache.row(k) for k in part], slice(0, FR.SCORING))
        px = FR.pixels(rgb)                               # [clips, 4, 3, S, S]
        vec = encoder.encode_pixels(px.reshape((-1,) + px.shape[2:]))
        vec = vec.reshape(len(part), FR.SCORING, -1)
        for k, v in zip(part, vec):
            cache.put(k, v)
        done = n + len(part)
        if done % 256 < chunk or done == len(todo):
            cache.save()
            rate = done / max(time.time() - started, 1e-6)
            log(f"  {done}/{len(todo)} clips, {rate:.1f}/s, about "
                f"{(len(todo) - done) / rate / 60:.1f} min left")
    cache.save()


def _finetune_folds(FT, settings, device, base_vm, fold_heads, folds, frame_cache, rows,
                    rows_extra, targets, groups, g_extra, single, log, should_stop) -> dict:
    """LP-FT on every fold from its frozen head, scored on the fold's unseen
    videos after every epoch. Returns the best epoch (by held-out single-action
    accuracy over all folds) with its held-out scores, and test/ clips' scores
    from the folds that never saw their video."""
    epochs, n_extra = settings.epochs, len(rows_extra)
    by_epoch = np.zeros((epochs,) + targets.shape, np.float32)
    extra_sum = np.zeros((epochs, n_extra, targets.shape[1]), np.float32)
    extra_n = np.zeros(n_extra)
    log(f"\nFine-tuning the top {settings.blocks} image-model blocks on {device}, scored on "
        f"unseen source videos ({len(folds)} folds x {epochs} epochs)")
    for i, ((tr, te), head0) in enumerate(zip(folds, fold_heads), 1):
        _check(should_stop)
        started = time.time()
        log(f"  fine-tune fold {i}/{len(folds)}: {len(tr)} clips to learn from, {len(te)} held out")
        one = single[te]
        unseen = (~np.isin(g_extra, groups[tr])) if n_extra else np.zeros(0, bool)

        def eval_fn(tower, head, te=te, one=one, unseen=unseen):
            p = FT.predict(tower, head, frame_cache, rows[te], device)
            pe = (FT.predict(tower, head, frame_cache, rows_extra[unseen], device)
                  if unseen.any() else None)
            acc = (float(np.mean(p[one].argmax(1) == targets[te][one].argmax(1)))
                   if one.any() else 0.0)
            return (p, pe), acc

        _, _, results = FT.finetune(base_vm, head0, frame_cache, rows[tr], targets[tr], settings,
                                    device, log=log, check=lambda: _check(should_stop),
                                    eval_fn=eval_fn)
        for e, ((p, pe), _) in enumerate(results):
            by_epoch[e, te] = p
            if pe is not None:
                extra_sum[e, unseen] += pe
        extra_n[unseen] += 1
        accs = [a for _, a in results]
        log(f"  fine-tune fold {i}/{len(folds)}: {accs[-1]:.3f} after {epochs} epochs "
            f"(best {max(accs):.3f} at epoch {int(np.argmax(accs)) + 1}), "
            f"{time.time() - started:.0f} s")
        FT.empty_cache(device)
    accs = [_single_accuracy(by_epoch[e], targets, single) for e in range(epochs)]
    log("  held-out single-action accuracy by epoch: "
        + ", ".join(f"{e + 1}: {a:.3f}" for e, a in enumerate(accs)))
    best = int(np.argmax(accs))
    extra = extra_sum[best] / np.maximum(extra_n, 1)[:, None]
    extra[extra_n == 0] = np.nan
    return {"epochs": best + 1, "accuracy": accs[best], "scores": by_epoch[best],
            "scores_extra": extra, "by_epoch": [round(a, 4) for a in accs]}


def _train(args, log, should_stop) -> int:
    from model_training.action_head import features as Fx
    from model_training.action_head import head as H
    from model_training.action_head import trust
    from modules.teach.benchmark import load_aliases, read_dataset
    from modules.vision import frame_encoder

    started = time.time()
    finetuning = args.finetune_blocks > 0
    FT = device = None
    if finetuning:
        from model_training.action_head import finetune as FT
        if args.frames != 4:
            log("❌ Fine-tuning scores 4 frames per clip, as the app does; leave --frames at 4")
            return 1
        try:
            device = FT.resolve_device(args.device)
            import transformers  # noqa: F401 - the starting weights come through it
        except FT.DeviceError as e:
            log(f"❌ {e}")
            return 1
        except ImportError:
            log("❌ Fine-tuning needs the transformers package, which is not installed here")
            return 1
    data = read_dataset(args.data_path, aliases=load_aliases(args.aliases))
    splits_used = POOL_SPLITS + (("test",) if args.teach_test else ())
    clips, classes, notes = select_clips(data["clips"], args.min_clips, splits_used)
    for note in notes:
        log(f"ℹ️ {note}")
    if not classes:
        log("❌ No class has enough single-action clips to train on")
        return 1
    known = set(classes)
    extra = ([] if args.teach_test else
             [c for c in data["clips"] if c.split == "test" and c.labels and set(c.labels) <= known])

    encoder = frame_encoder.load(args.backend, log=log)
    if encoder is None:
        log("❌ The frame encoder is not available; see the lines above")
        return 1
    cache = Fx.FeatureCache(args.cache or default_cache(args.data_path, encoder.encoder_id, args.frames),
                            encoder.encoder_id, args.frames, encoder.dims)
    n = len(clips)
    paths = [c.path for c in clips + extra]
    frame_cache = shared_probe = None
    if finetuning:
        from model_training.action_head import frames as FR
        keys = [Fx.clip_key(p, args.data_path) for p in paths]
        frame_cache = FR.build(args.frame_cache or default_frame_cache(args.data_path), paths, keys,
                               log=log, check=lambda: _check(should_stop))
        if frame_cache is None:
            return 1
        _encode_from_frames(frame_cache, keys, encoder, cache, log, should_stop)
        shared_probe = encoder.encode_pixels(frame_encoder.probe_pixels())[0]
    x_all, ok = Fx.encode_clips(paths, args.data_path, encoder,
                                cache, log=log, should_stop=should_stop)
    _check(should_stop)
    if finetuning:
        # Training needs the card's memory; the shared encoder's work is done.
        encoder.close()
        ok &= np.array([bool(frame_cache.ok[frame_cache.row(k)]) for k in keys])
        rows_all = np.array([frame_cache.row(k) for k in keys], int)
    x, x_extra = x_all[:n][ok[:n]], x_all[n:][ok[n:]]
    clips = [c for c, good in zip(clips, ok[:n]) if good]
    extra = [c for c, good in zip(extra, ok[n:]) if good]
    if finetuning:
        rows, rows_extra = rows_all[:n][ok[:n]], rows_all[n:][ok[n:]]

    index = {c: k for k, c in enumerate(classes)}
    label_sets = [{index[lb] for lb in c.labels} for c in clips]
    targets = np.zeros((len(clips), len(classes)), np.float32)
    for i, s in enumerate(label_sets):
        targets[i, list(s)] = 1
    strat = np.array([min(s) for s in label_sets])
    groups = np.array([c.group for c in clips])
    g_extra = np.array([c.group for c in extra])
    n_videos = len(set(groups.tolist()))
    log(f"\n{len(clips)} clips ({int((targets.sum(1) > 1).sum())} with two or more actions), "
        f"{len(classes)} classes, {n_videos} source videos")

    folds = group_folds(strat, groups, args.folds, args.seed)
    log(f"Scoring on unseen source videos ({len(folds)} folds)")
    single = targets.sum(1) == 1
    best = None
    for steps in [int(s) for s in str(args.steps).split(",") if s.strip()]:
        log(f"  {steps} steps")
        fold_heads = []
        scores, scores_extra = out_of_fold(x, targets, groups, folds, steps, args.seed, log,
                                           x_extra=x_extra if extra else None,
                                           groups_extra=g_extra, should_stop=should_stop,
                                           models=fold_heads)
        acc = _single_accuracy(scores, targets, single)
        log(f"  {steps} steps: held-out single-action accuracy {acc:.3f}")
        if best is None or acc > best[1] + 1e-9:
            best = (steps, acc, scores, scores_extra, fold_heads)
    steps, acc, scores, scores_extra, fold_heads = best
    is_test = np.array([c.split == "test" for c in clips], bool)
    frozen = _evaluate(scores, scores_extra, targets, label_sets, groups, is_test, extra, index, args)

    in_train = np.array([c.split == "train" for c in clips], bool)
    in_val = np.array([c.split == "val" for c in clips], bool) & single
    val_scores = None
    if in_train.any() and in_val.any():
        log("\nScoring val/ as the dataset defines it (training on train/ only)")
        model = H.train_head(x[in_train], targets[in_train], len(classes), steps=steps,
                             seed=args.seed)
        val_pred = H.predict_proba(model, x[in_val]).argmax(1)
        shared = np.isin(groups[in_val], groups[in_train])
        val_scores = {"clips": int(in_val.sum()),
                      "accuracy": round(float(np.mean(val_pred == targets[in_val].argmax(1))), 4),
                      "clips_sharing_a_video_with_train": int(shared.sum())}

    out = args.out or default_out(args.name)
    ft = None
    if finetuning:
        settings = FT.Settings(blocks=args.finetune_blocks, epochs=args.ft_epochs, lr=args.ft_lr,
                               decay=args.ft_decay, head_lr=args.ft_head_lr, batch=args.ft_batch,
                               seed=args.seed)
        with FT.keep_awake():
            try:
                run = SimpleNamespace(
                    args=args, log=log, should_stop=should_stop, settings=settings,
                    device=device, frame_cache=frame_cache, shared_probe=shared_probe,
                    fold_heads=fold_heads, folds=folds, rows=rows, rows_extra=rows_extra, x=x,
                    targets=targets, groups=groups, g_extra=g_extra, single=single, steps=steps,
                    frozen_acc=acc, frozen=frozen, label_sets=label_sets, is_test=is_test,
                    extra=extra, index=index, classes=classes, n_videos=n_videos,
                    val_scores=val_scores, out=out, started=started)
                code, ft = _finetune_and_save(FT, run)
            except FT.OutOfMemory as e:
                log(f"❌ {e}")
                return 1
            finally:
                frame_cache.close()
        if code is not None:
            return code
        log(f"\nThe fine-tuned model did not beat the frozen one on unseen videos "
            f"({ft['accuracy']:.3f} against {acc:.3f}), so the frozen model is saved "
            f"(2 MB, on the shared image model).")

    _check(should_stop)
    log(f"\nTraining the saved head on all {len(clips)} clips ({steps} steps)")
    model = H.train_head(x, targets, len(classes), steps=steps, seed=args.seed)
    os.makedirs(out, exist_ok=True)
    head_path = os.path.join(out, HEAD_FILE)
    H.export_onnx(model, head_path, args.frames)
    check = H.onnx_proba(H.load_onnx_session(head_path), x[:64])
    drift = float(np.abs(check - H.predict_proba(model, x[:64])).max())
    if drift > 1e-4:
        log(f"❌ The exported head disagrees with the trained one (max {drift:.2e})")
        return 1
    stale = os.path.join(out, "vision.onnx")      # an earlier fine-tuned model's tower
    if os.path.isfile(stale):
        os.remove(stale)

    meta = _meta(args, encoder.encoder_id, encoder.dims, classes, frozen, steps, folds,
                 len(clips), n_videos, val_scores, scores, targets, groups, single)
    if ft is not None:
        meta["finetune_not_saved"] = {
            "why": "its held-out single-action accuracy did not beat the frozen head's",
            "heldout_single_accuracy": round(ft["accuracy"], 4),
            "frozen_heldout_single_accuracy": round(acc, 4),
            "by_epoch": ft["by_epoch"], **FT.settings_dict(settings)}
    with open(os.path.join(out, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)

    _report(log, frozen, val_scores, args, classes, meta["per_class"])
    log(SAVED_FROZEN)
    log(f"\n✅ Saved {head_path} and {META_FILE} ({time.time() - started:.0f} s)")
    return 0


def _finetune_and_save(FT, run):
    """The fine-tune after the frozen folds (``run`` holds what those
    produced): per-fold LP-FT, then, when it beats the frozen head on the same
    folds, the final run on every clip and the saved model. Returns
    ``(0 or 1, ft)`` when it decided the run, or ``(None, ft)`` when the frozen
    head should be saved instead."""
    from model_training.action_head import head as H
    from model_training.action_head import tower as TW
    from modules.vision import frame_encoder

    r, log, settings, device = run, run.log, run.settings, run.device
    args, targets, classes = run.args, run.targets, run.classes

    log(f"\nLoading the starting weights ({frame_encoder.SOURCE_MODEL}; the first run downloads "
        f"about 1.5 GB)")
    try:
        base_vm = FT.load_base_tower(frame_encoder.SOURCE_MODEL, frame_encoder.SOURCE_REVISION)
    except Exception as e:  # noqa: BLE001 - offline, or a broken download
        log(f"❌ Could not load the starting weights: {type(e).__name__}: {e}")
        return 1, None
    try:
        cos = FT.start_check(base_vm, r.shared_probe, device)
    except ValueError as e:
        log(f"❌ {e}")
        return 1, None
    base_vm = base_vm.cpu()
    log(f"  the starting tower matches the app's encoder (cosine {cos:.5f})")

    ft = _finetune_folds(FT, settings, device, base_vm, r.fold_heads, r.folds, r.frame_cache,
                         r.rows, r.rows_extra, targets, r.groups, r.g_extra, r.single, log,
                         r.should_stop)
    tuned = _evaluate(ft["scores"], ft["scores_extra"], targets, r.label_sets, r.groups,
                      r.is_test, r.extra, r.index, args)
    log("")
    _log_scores(log, "Frozen image model, on unseen videos", r.frozen["heldout"])
    _log_scores(log, f"Fine-tuned image model ({_epochs(ft['epochs'])}), on unseen videos",
                tuned["heldout"])
    if r.frozen["test"] or tuned["test"]:
        _log_scores(log, "Frozen image model, test/", r.frozen["test"])
        _log_scores(log, "Fine-tuned image model, test/", tuned["test"])
    if not ft["accuracy"] > r.frozen_acc + 1e-9:
        return None, ft

    _check(r.should_stop)
    log(f"\nFinal model: training the head on all {len(targets)} clips ({r.steps} steps), then "
        f"fine-tuning with it ({_epochs(ft['epochs'])})")
    head0 = H.train_head(r.x, targets, len(classes), steps=r.steps, seed=args.seed)
    final = FT.Settings(**{**FT.settings_dict(settings), "epochs": ft["epochs"]})
    tower, head, _ = FT.finetune(base_vm, head0, r.frame_cache, r.rows, targets, final, device,
                                 log=log, check=lambda: _check(r.should_stop), label="final ")
    check_vectors = FT.vectors(tower, r.frame_cache, r.rows[:8], device)
    tower_vm = tower.vm.float().cpu().eval()
    del tower
    FT.empty_cache(device)

    dims = int(check_vectors.shape[-1])
    meta = _meta(args, TW.finetuned_id(frame_encoder.ENCODER_ID), dims, classes, tuned, r.steps,
                 r.folds, len(targets), r.n_videos, None, ft["scores"], targets, r.groups, r.single)
    meta["input"] = (f"features [N, frames, {dims}]: vectors from this folder's own image "
                     f"model, frames evenly across the clip")
    meta["finetune"] = {
        **FT.settings_dict(final), "epochs_compared": settings.epochs, "by_epoch": ft["by_epoch"],
        "base": frame_encoder.SOURCE_MODEL, "revision": frame_encoder.SOURCE_REVISION,
        "starts_from": frame_encoder.ENCODER_ID,
        "frozen_heldout": r.frozen["heldout"], "frozen_test": r.frozen["test"],
        "frozen_val_folder": r.val_scores}
    log(f"\nWriting the model to {r.out}")
    try:
        TW.save(tower_vm, head, r.out, meta, check_vectors, log=log)
    except ValueError as e:
        log(f"❌ The fine-tuned model was not saved: {e}")
        return 1, ft

    _report(log, tuned, None, args, classes, meta["per_class"])
    log(SAVED_FINETUNED)
    log(f"\n✅ Saved {r.out} ({time.time() - r.started:.0f} s)")
    return 0, ft


def _meta(args, encoder_id, dims, classes, ev, steps, folds, n_clips, n_videos, val_scores,
          scores, targets, groups, single) -> dict:
    """head.json for a head whose held-out ``scores`` gave ``ev``."""
    from model_training.action_head import trust
    thresholds, pairs, pair_th = ev["thresholds"], ev["pairs"], ev["pair_th"]
    found = trust.detected(scores, thresholds, pairs, pair_th)
    pred = scores.argmax(1)
    per_class = {}
    for k, name in enumerate(classes):
        shows, said = targets[:, k] > 0, found[:, k]
        mine = shows & single
        per_class[name] = {
            "clips": int(mine.sum()),
            "clips_with_another_action": int((shows & ~single).sum()),
            "videos": int(len(set(groups[shows].tolist()))),
            "heldout_recall": round(float(np.mean(pred[mine] == k)), 3) if mine.any() else None,
            "heldout_detected_precision": (round(float(np.mean(shows[said])), 3)
                                           if said.any() else None),
            "trust_threshold": (None if thresholds[k] is None else round(thresholds[k], 4)),
        }
    return {
        "format": HEAD_FORMAT,
        "kind": "action-head",
        "encoder": encoder_id,
        "frames": args.frames,
        "input": f"features [N, frames, {dims}]: frame encoder vectors, "
                 f"frames evenly across the clip",
        "output": "logits [N, classes]; a sigmoid gives each action's own score, and an "
                  "action is detected when its score reaches its trust threshold",
        "activation": "sigmoid",
        "classes": classes,
        "trust_thresholds": [None if t is None else round(t, 4) for t in thresholds],
        "pair_thresholds": [{"actions": [classes[a], classes[b]],
                             "threshold": None if t is None else round(t, 4)}
                            for (a, b), t in zip(pairs, pair_th)],
        "trust_precision": args.precision,
        "trust_min_videos": args.min_videos,
        "steps": steps,
        "heldout": {"how": f"{len(folds)} folds by source video", "clips": n_clips,
                    "videos": n_videos, **ev["heldout"]},
        "test": {"taught": bool(args.teach_test), **ev["test"]},
        "val_folder": val_scores,
        "per_class": per_class,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _report(log, ev, val_scores, args, classes, per_class) -> None:
    thresholds, pair_th = ev["thresholds"], ev["pair_th"]
    log("")
    _log_scores(log, "Held out (whole source videos never seen)", ev["heldout"])
    if val_scores:
        log(f"val/ as the dataset defines it (trained on train/ only): accuracy "
            f"{val_scores['accuracy']:.3f} on {val_scores['clips']} clips, "
            f"{val_scores['clips_sharing_a_video_with_train']} of which share a source video "
            f"with train/")
    if ev["test"]:
        _log_scores(log, "test/" + (" (taught, scored by heads that never saw each video)"
                                    if args.teach_test else " (not taught)"), ev["test"])
    log(f"Trusted actions: {sum(t is not None for t in thresholds)} of {len(classes)}; "
        f"trusted pairs: {sum(t is not None for t in pair_th)} of {len(ev['pairs'])} taught")
    width = max(len(c) for c in classes)
    log(f"\n{'class':{width}}  clips +pair videos recall precision   threshold")
    for name, row in sorted(per_class.items(), key=lambda kv: -kv[1]["clips"]):
        rec = "-" if row["heldout_recall"] is None else f"{row['heldout_recall']:.2f}"
        prec = ("-" if row["heldout_detected_precision"] is None
                else f"{row['heldout_detected_precision']:.2f}")
        th = "not trusted" if row["trust_threshold"] is None else f"{row['trust_threshold']:.2f}"
        log(f"{name:{width}}  {row['clips']:5} {row['clips_with_another_action']:5} "
            f"{row['videos']:6} {rec:>6} {prec:>9} {th:>11}")

if __name__ == "__main__":
    _utf8_stdout()
    sys.exit(main())
