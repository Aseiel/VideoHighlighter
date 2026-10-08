"""Sort every sample into a class, the way sorter.py did for a trained model.

``sorter.py`` needs a model that already knows the classes. A project starts
without one, so round 1 sorts by CLIP (``scoring``): by the words, and by the
examples as soon as there are any. From round 2 the project's own model sorts
too, through ``sorter.py``'s classifier, and the review queue asks about the
samples where the two disagree.

The result is recorded on each sample (scores, proposal, lead), and optionally
laid out as folders, ``sorted/<class>/``, ``sorted/_unsure/``,
``sorted/_none/`` — the layout the manual process used, so sorting by hand
still works: move what is wrong, then ``review.from_folders``.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Callable, Optional

import numpy as np

from modules.teach import embed as embed_mod
from modules.teach import scoring
from modules.teach.naming import PROMPTS
from modules.teach.project import (
    ACCEPTED, NEGATIVE, NONE, PENDING, REJECTED, UNSURE, Project,
)

SORTED_DIR = "sorted"
REJECT_DIR = "_reject"
PLACED_FILE = ".placed.json"


def class_prompts(project: Project, spec) -> list:
    prompts = [PROMPTS[project.task].format(spec.name)]
    if spec.description:
        prompts.append(spec.description)
    return prompts


def build_prototypes(project: Project, vectors: dict, embedder) -> list:
    """One prototype per class, plus ``NONE`` once negatives exist."""
    prototypes = []
    for spec in project.classes:
        # Only what a person decided: an auto-accepted sample in the prototype
        # would teach the sorter to agree with itself.
        ids = list(dict.fromkeys(list(spec.examples)
                                 + [s.id for s in project.accepted(spec.name)
                                    if s.is_human]))
        examples = [vectors[i] for i in ids if i in vectors]
        texts = embedder.texts(class_prompts(project, spec)) if not examples else []
        proto = scoring.build_prototype(spec.name, examples, texts,
                                        centers=project.settings.prototypes_per_class)
        if proto is not None:
            prototypes.append(proto)
    negatives = [vectors[s.id] for s in project.samples
                 if s.verdict == NEGATIVE and s.is_human and s.id in vectors]
    if negatives:
        prototypes.append(scoring.build_prototype(NONE, negatives))
    return prototypes


def _video_of(sample) -> str:
    """The video a sample came from, for holding videos out: its source, or
    for a dataset clip (one source for them all) the name it was cut under."""
    from modules.teach import benchmark
    from modules.teach.dataset_sort import DATASET_SOURCE

    if sample.source == DATASET_SOURCE:
        return "dataset:" + benchmark.group_of(os.path.basename(sample.path))
    return sample.source


def linear_training_set(project: Project, vectors: dict) -> tuple:
    """``(rows, labels, groups, left_out)`` for the linear layer.

    What a person decided only, as for prototypes: each class's examples and
    human-accepted samples, and human negatives as ``NONE``. A class with
    fewer than ``linear_min_examples`` is left out (``left_out`` says so): a
    layer cannot learn a class from a clip or two, and one that tries labels
    everything faintly like that clip with it.
    """
    need = project.settings.linear_min_examples
    rows, labels, groups, left_out = [], [], [], {}
    by_id = {s.id: s for s in project.samples}
    for spec in project.classes:
        ids = list(dict.fromkeys(list(spec.examples)
                                 + [s.id for s in project.accepted(spec.name) if s.is_human]))
        ids = [i for i in ids if i in vectors and i in by_id]
        if len(ids) < need:
            left_out[spec.name] = len(ids)
            continue
        for i in ids:
            rows.append(vectors[i])
            labels.append(spec.name)
            groups.append(_video_of(by_id[i]))
    negatives = [s for s in project.samples
                 if s.verdict == NEGATIVE and s.is_human and s.id in vectors]
    if len(negatives) >= need:
        for s in negatives:
            rows.append(vectors[s.id])
            labels.append(NONE)
            groups.append(_video_of(s))
    return rows, labels, groups, left_out


def score_linear(project: Project, vectors: dict, ids: list) -> Optional[tuple]:
    """``(results, info)`` as ``scoring.score_samples`` gives, from a linear
    layer -- or ``None`` when fewer than two classes have enough examples."""
    from modules.teach import linear

    rows, labels, groups, left_out = linear_training_set(project, vectors)
    if len({l for l in labels if l != NONE}) < 2:
        return None
    settings = project.settings
    x = np.stack(rows)
    model = linear.fit(x, labels)
    # Both gates from one held-out run: proposing at linear_precision, and
    # auto-accepting -- deciding with nobody looking -- at the stricter
    # linear_auto_precision. Without held-out videos, the settings' own gates.
    measured = linear.held_out(x, labels, groups)
    calibration = None
    gate, auto_gate = settings.gate, settings.auto_gate
    if measured is not None:
        conf, right = measured
        gate, share = linear.threshold_at(conf, right, settings.linear_precision)
        auto_gate, auto_share = linear.threshold_at(conf, right, settings.linear_auto_precision)
        calibration = {"threshold": round(gate, 4), "coverage": round(share, 3),
                       "auto_threshold": round(auto_gate, 4), "auto_coverage": round(auto_share, 3),
                       "heldout_accuracy": round(float(right.mean()), 3),
                       "groups": len(set(groups))}
    proba = model.proba(np.stack([vectors[i] for i in ids])) if ids else np.zeros((0, 0))
    out = {}
    for i, sid in enumerate(ids):
        row = proba[i]
        order = np.argsort(-row)
        best = model.classes[int(order[0])]
        lead = float(row[order[0]] - row[order[1]]) if len(order) > 1 else float(row[order[0]])
        if best == NONE:
            proposal = NONE
        elif row[order[0]] >= gate and lead >= settings.margin:
            proposal = best
        else:
            proposal = UNSURE
        scores = {c: round(float(row[j]), 4) for j, c in enumerate(model.classes) if c != NONE}
        out[sid] = (scores, proposal, round(lead, 4))
    info = {"gate": round(float(gate), 4), "auto_gate": round(float(auto_gate), 4),
            "calibration": calibration,
            "trained_on": len(labels), "classes": [c for c in model.classes if c != NONE],
            "left_out": left_out}
    return out, info


def sort_project(project: Project, embedder, *,
                 frame_reader: Optional[Callable] = None,
                 model_classifier: Optional[Callable] = None,
                 progress: Optional[Callable] = None) -> dict:
    """Score every sample; propose a class for every undecided one.

    ``model_classifier(path) -> (label, confidence)`` is the last round's model
    (``head_classifier``), when there is one.
    """
    if not project.classes:
        raise ValueError("add a class first")
    cache = embed_mod.VectorCache(project.root, getattr(embedder, "model_id", ""))
    vectors = embed_mod.sample_vectors(
        project.samples, embedder, cache, project.settings.frames_per_sample,
        frame_reader=frame_reader or embed_mod.read_frames, progress=progress)
    ids = [s.id for s in project.samples if s.id in vectors]
    settings = project.settings
    prototypes, linear_info, scorer = [], None, "prototypes"
    if settings.scorer == "linear":
        scored = score_linear(project, vectors, ids)
        if scored is not None:
            results, linear_info = scored
            scorer = "linear"
    if scorer == "prototypes":
        prototypes = build_prototypes(project, vectors, embedder)
        matrix = np.stack([vectors[i] for i in ids]) if ids else np.zeros((0, 1))
        results = scoring.score_samples(ids, matrix, prototypes, gate=settings.gate,
                                        margin=settings.margin, floor=settings.floor)

    tally = {}
    for sample in project.samples:
        # A clip nothing could be decoded from is set aside rather than left
        # looking unsorted, which would keep `sort` the next step for ever.
        sample.unreadable = sample.id not in vectors
        if sample.id not in results:
            continue
        sample.scores, sample.proposed, sample.margin = results[sample.id]
        if model_classifier is not None and sample.verdict == PENDING:
            try:
                label, confidence = model_classifier(sample.focus_paths[0]
                                                     if sample.focus_paths else sample.path)
            except Exception as exc:        # a broken clip must not stop the sort
                print(f"teach.sort: model could not read {sample.id}: {exc}")
                label, confidence = "", 0.0
            sample.model_proposed = label or ""
            sample.model_confidence = float(confidence or 0.0)
        if sample.verdict == PENDING:
            tally[sample.proposed] = tally.get(sample.proposed, 0) + 1
    project.save()

    from modules.teach import autolabel
    # A layer's scores are probabilities, not prototype likeness: auto-accept
    # goes by the probability that was right often enough on held-out videos.
    auto = autolabel.apply(project, gate=linear_info["auto_gate"] if linear_info else None)
    return {
        "auto": auto,
        "scored": len(results),
        "unreadable": len(project.samples) - len(results),
        "proposed": tally,
        "scorer": scorer,
        "prototypes": {p.name: {"from": p.kind, "examples": p.n_examples}
                       for p in prototypes},
        **({"linear": linear_info} if linear_info else {}),
    }


def _place(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def lay_out_folders(project: Project) -> dict:
    """``sorted/<class>/<sample>.mp4`` for sorting by hand.

    Undecided samples go where they are proposed; decided ones where the
    verdict put them, so the folders always show the current state. Hard links
    where the disk allows, so this costs no space. Rebuilt from scratch each
    time: the folders are a view, the verdicts are the record.
    """
    root = project.path(SORTED_DIR)
    if os.path.isdir(root):
        shutil.rmtree(root)
    placed = {}
    for sample in project.samples:
        if sample.verdict == ACCEPTED:
            folder = sample.label
        elif sample.verdict == NEGATIVE:
            folder = NONE
        elif sample.verdict == REJECTED:
            folder = REJECT_DIR
        else:
            folder = sample.proposed or UNSURE
        target_dir = os.path.join(root, folder)
        os.makedirs(target_dir, exist_ok=True)
        name = sample.id + os.path.splitext(sample.path)[1]
        if os.path.exists(sample.path):
            _place(sample.path, os.path.join(target_dir, name))
            placed[name] = folder
    for spec in project.classes:
        os.makedirs(os.path.join(root, spec.name), exist_ok=True)
    for extra in (UNSURE, NONE, REJECT_DIR):
        os.makedirs(os.path.join(root, extra), exist_ok=True)
    with open(os.path.join(root, PLACED_FILE), "w", encoding="utf-8") as handle:
        json.dump(placed, handle, indent=1)
    return {"folder": root, "files": len(placed)}


def _head_model(folder: str, *, encoder=None, head=None,
                frame_reader: Optional[Callable] = None) -> tuple:
    """``(path -> each action's score or None, classes)`` for a trained action
    head (``head.onnx`` + ``head.json``).

    The encoder the head reads (its own, for a fine-tuned head) on the head's
    own number of frames, spread over the clip as the trainer reads them, so
    what it says here is what the model will say in use.
    """
    from modules.vision import action_siglip

    head = head or action_siglip.ActionHead(folder)
    enc = encoder or action_siglip.encoder_for(head, log=print)
    if enc is None:
        raise RuntimeError("the frame encoder is not available")
    if head.encoder_id and head.encoder_id != enc.encoder_id:
        raise RuntimeError(f"head trained on {head.encoder_id}, not {enc.encoder_id}")
    read = frame_reader or embed_mod.read_frames

    def scores(path: str):
        frames = read(path, head.frames)
        if len(frames) != head.frames:
            return None
        return head.scores(enc.encode_bgr(frames)[None])[0]

    return scores, head.classes


def head_scorer(folder: str, *, encoder=None, head=None,
                frame_reader: Optional[Callable] = None) -> Callable:
    """A trained action head as ``path -> {label: score}`` ({} if unreadable)."""
    scores, classes = _head_model(folder, encoder=encoder, head=head,
                                  frame_reader=frame_reader)

    def score(path: str) -> dict:
        s = scores(path)
        return {} if s is None else {c: float(v) for c, v in zip(classes, s)}

    return score


def head_classifier(folder: str, *, encoder=None, head=None,
                    frame_reader: Optional[Callable] = None) -> Callable:
    """A trained round's action head as a proposer: ``path -> (label, score)``."""
    scores, classes = _head_model(folder, encoder=encoder, head=head,
                                  frame_reader=frame_reader)

    def classify(path: str):
        s = scores(path)
        if s is None:
            return "", 0.0
        best = int(np.argmax(s))
        return classes[best], float(s[best])

    return classify


def round_classifier(project) -> Optional[Callable]:
    """The installed round's model as a proposer, or None before there is one."""
    import os as _os

    from modules.teach.project import ACTIONS

    if project.task != ACTIONS:
        return None
    for record in reversed(project.rounds):
        metrics = record.get("metrics") or {}
        if record.get("installed") and metrics.get("head"):
            if _os.path.isdir(metrics["head"]):
                try:
                    return head_classifier(metrics["head"])
                except Exception as exc:        # a broken model must not stop the sort
                    print(f"teach.sort: round {record.get('round')} model unusable: {exc}")
                    return None
    return None
