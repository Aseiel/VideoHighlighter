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
        proto = scoring.build_prototype(spec.name, examples, texts)
        if proto is not None:
            prototypes.append(proto)
    negatives = [vectors[s.id] for s in project.samples
                 if s.verdict == NEGATIVE and s.is_human and s.id in vectors]
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
    auto = autolabel.apply(project)
    return {
        "auto": auto,
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


def r3d_classifier(weights: str, mapping: str, *, wrapper_factory=None,
                   frame_reader: Optional[Callable] = None) -> Callable:
    """A trained round's R3D model as a proposer: ``path -> (label, confidence)``.

    The same ``R3DModelWrapper`` action recognition uses in the app, loaded
    with the round's weights, so what it proposes here is what the model will
    say in use. Sixteen frames spread over the clip, as it was trained on.
    """
    import json as _json

    with open(mapping, "r", encoding="utf-8") as handle:
        data = _json.load(handle)
    idx_to_label = {int(k): v for k, v in (data.get("idx_to_label") or {}).items()}
    # train.py writes the variant at the top level; older mappings nest it.
    variant = (data.get("model_variant")
               or (data.get("metadata") or {}).get("model_variant") or "r3d_18")
    # A filtered (production) mapping lists fewer labels than the head has
    # outputs; the weights need the head's own size.
    num_classes = int(data.get("num_classes_total")
                      or (max(idx_to_label) + 1 if idx_to_label else 0))
    if wrapper_factory is None:
        from action_recognition import R3DModelWrapper as wrapper_factory
    # "cuda" falls back to the processor by itself when no usable card is there.
    model = wrapper_factory(model_name=variant, device_str="cuda", half_precision=False,
                            custom_weights=weights, custom_num_classes=num_classes)
    read = frame_reader or embed_mod.read_frames

    def classify(path: str):
        frames = read(path, 16)
        if len(frames) != 16:
            return "", 0.0
        logits = np.asarray(model.predict_from_frames(frames), dtype=np.float64).ravel()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(np.argmax(probs))
        return idx_to_label.get(best, ""), float(probs[best])

    return classify


def round_classifier(project) -> Optional[Callable]:
    """The installed round's model as a proposer, or None before there is one."""
    import os as _os

    from modules.teach.project import ACTIONS

    if project.task != ACTIONS:
        return None
    for record in reversed(project.rounds):
        metrics = record.get("metrics") or {}
        if record.get("installed") and metrics.get("weights") and metrics.get("mapping"):
            if _os.path.exists(metrics["weights"]) and _os.path.exists(metrics["mapping"]):
                try:
                    return r3d_classifier(metrics["weights"], metrics["mapping"])
                except Exception as exc:        # a broken model must not stop the sort
                    print(f"teach.sort: round {record.get('round')} model unusable: {exc}")
                    return None
    return None


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
