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
        ids = list(dict.fromkeys(list(spec.examples)
                                 + [s.id for s in project.accepted(spec.name)]))
        examples = [vectors[i] for i in ids if i in vectors]
        texts = embedder.texts(class_prompts(project, spec)) if not examples else []
        proto = scoring.build_prototype(spec.name, examples, texts)
        if proto is not None:
            prototypes.append(proto)
    negatives = [vectors[s.id] for s in project.samples
                 if s.verdict == NEGATIVE and s.id in vectors]
    if negatives:
        prototypes.append(scoring.build_prototype(NONE, negatives))
    return prototypes


def sort_project(project: Project, embedder, *,
                 frame_reader: Optional[Callable] = None,
                 model_classifier: Optional[Callable] = None,
                 progress: Optional[Callable] = None) -> dict:
    """Score every sample; propose a class for every undecided one.

    ``model_classifier(path) -> (label, confidence)`` is the last round's model
    (``sorter_classifier``), when there is one.
    """
    if not project.classes:
        raise ValueError("add a class first")
    cache = embed_mod.VectorCache(project.root, getattr(embedder, "model_id", ""))
    vectors = embed_mod.sample_vectors(
        project.samples, embedder, cache, project.settings.frames_per_sample,
        frame_reader=frame_reader or embed_mod.read_frames, progress=progress)
    prototypes = build_prototypes(project, vectors, embedder)

    ids = [s.id for s in project.samples if s.id in vectors]
    matrix = np.stack([vectors[i] for i in ids]) if ids else np.zeros((0, 1))
    settings = project.settings
    results = scoring.score_samples(ids, matrix, prototypes, gate=settings.gate,
                                    margin=settings.margin, floor=settings.floor)

    tally = {}
    for sample in project.samples:
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

    return {
        "scored": len(results),
        "unreadable": len(project.samples) - len(results),
        "proposed": tally,
        "prototypes": {p.name: {"from": p.kind, "examples": p.n_examples}
                       for p in prototypes},
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


def sorter_classifier(xml: str, bin_path: str, mapping: str) -> Callable:
    """The last round's model, through sorter.py's own classify_clip.

    Takes an Intel-encoder decoder as OpenVINO IR plus its mapping — the
    kind sorter.py loads for the app's installed model (``teach sort
    --model-xml``). Projects train R3D by default, which this does not read.
    """
    import json as _json

    import sorter
    from openvino.runtime import Core

    with open(mapping, "r", encoding="utf-8") as handle:
        idx_to_label = {int(k): v for k, v in _json.load(handle)["idx_to_label"].items()}
    core = Core()
    enc = core.compile_model(core.read_model(str(sorter.ENCODER_XML),
                                             str(sorter.ENCODER_BIN)), "CPU")
    dec = core.compile_model(core.read_model(xml, bin_path), "CPU")

    def classify(path: str):
        label, confidence, _, _ = sorter.classify_clip(
            path, enc, enc.input(0), enc.output(0), dec, dec.input(0),
            dec.output(0), idx_to_label)
        return label or "", confidence

    return classify
