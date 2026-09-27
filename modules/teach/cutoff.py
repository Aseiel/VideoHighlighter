"""Ask about a box only when it is in doubt, and learn where doubt starts.

A fixed threshold ("ask below 50%") means something different for every
class and every kind of footage, and a CLIP likeness is not a probability
anyway. So the cutoff is learnt, per class, from the answers a person gave:
the lowest score above which checked boxes were right often enough
(``1 - MAX_ERROR``).
Boxes scoring above it are accepted without a question; everything below is
asked.

It stays honest the same way sample auto-accept does:

* **It starts late.** No cutoff until ``MIN_DECIDED`` boxes of the class
  were checked, with at least ``MIN_ABOVE`` of them above the cutoff.
* **It keeps being checked.** One box in ``SPOT_CHECK_EVERY`` that would be
  accepted is asked anyway (chosen by the box, so the choice is stable).
  Those answers go into the next cutoff.
* **It takes back what it gave.** When answers push the cutoff up, boxes it
  accepted that now fall below are asked after all.

Only boxes that a person, or an agent judging for one, decided count.
Which boxes were decided here is kept in ``auto_boxes.json``, beside the
labels, so ``label_store`` stays the format every edition shares.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from modules.teach.project import Project
from modules.vision.label_store import ACCEPTED, PENDING, REJECTED, LabelStore

AUTO_FILE = "auto_boxes.json"
# Sources whose confidence is on the calibrated scale this learns on. Stock
# detector boxes keep their own rule (``boxes.AUTO_BOX_CONFIDENCE``).
LEARNT_SOURCES = ("found",)
MIN_DECIDED = 10
MIN_ABOVE = 5
SPOT_CHECK_EVERY = 8
# Wrong boxes tolerated above the cutoff. Stricter than sample auto-accept:
# a wrong box is a wrong training label, not just a wrong folder.
MAX_ERROR = 0.1


def box_key(box) -> str:
    return "|".join([box.video, f"{box.time:.3f}", box.class_name,
                     ",".join(f"{v:.5f}" for v in box.box)])


def _load(project: Project) -> set:
    try:
        with open(project.path(AUTO_FILE), "r", encoding="utf-8") as handle:
            return set(json.load(handle))
    except (OSError, ValueError):
        return set()


def _save(project: Project, keys: set) -> None:
    path = project.path(AUTO_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(sorted(keys), handle, indent=0)
    os.replace(tmp, path)


def auto_decided(project: Project) -> set:
    """Keys of boxes accepted here without a question (and not re-checked)."""
    return _load(project)


def is_spot_check(box) -> bool:
    digest = hashlib.sha1(box_key(box).encode("utf-8")).digest()
    return digest[0] % SPOT_CHECK_EVERY == 0


def checked(project: Project, labels: LabelStore, class_name: str,
            auto: Optional[set] = None) -> list:
    """``(confidence, right)`` for every box of the class a person decided."""
    auto = auto_decided(project) if auto is None else auto
    return [(b.confidence, b.verdict == ACCEPTED) for b in labels.boxes
            if b.class_name == class_name and b.source in LEARNT_SOURCES
            and b.verdict in (ACCEPTED, REJECTED) and box_key(b) not in auto]


def learn(decided: list, max_error: float) -> Optional[float]:
    """The lowest score whose boxes above it were right often enough, or None.

    ``decided`` is ``(confidence, right)`` pairs. Walks down from the highest
    score, keeping the lowest point where the share of wrong answers at or
    above it is still within ``max_error``.
    """
    if len(decided) < MIN_DECIDED:
        return None
    ranked = sorted(decided, key=lambda pair: -pair[0])
    best, wrong = None, 0
    for n, (confidence, right) in enumerate(ranked, 1):
        wrong += not right
        tie = n < len(ranked) and ranked[n][0] == confidence
        if tie:
            continue             # a cutoff cannot fall between equal scores
        if n >= MIN_ABOVE and wrong / n <= max_error:
            best = confidence
    return best


def cutoffs(project: Project, labels: LabelStore) -> dict:
    """``{class: cutoff or None}`` from the answers so far."""
    auto = auto_decided(project)
    return {name: learn(checked(project, labels, name, auto),
                        min(MAX_ERROR, project.settings.auto_max_error))
            for name in project.class_names()}


def apply(project: Project, labels: LabelStore) -> dict:
    """Accept pending boxes above their class's cutoff; take back those below.

    Saves ``labels`` and returns what changed. The caller turns accepted
    boxes into sample verdicts (``find.settle``).
    """
    auto = auto_decided(project)
    # A box a person has since decided is theirs, not ours.
    still = {box_key(b) for b in labels.boxes if b.verdict == ACCEPTED}
    auto &= still
    limits = cutoffs(project, labels) if project.settings.auto_accept else {}
    accepted = taken_back = 0
    for box in labels.boxes:
        if box.source not in LEARNT_SOURCES:
            continue
        limit = limits.get(box.class_name)
        key = box_key(box)
        if key in auto and (limit is None or box.confidence < limit):
            labels.set_verdict(box, PENDING)
            auto.discard(key)
            taken_back += 1
        elif (box.verdict == PENDING and limit is not None
              and box.confidence >= limit and not is_spot_check(box)):
            labels.set_verdict(box, ACCEPTED)
            auto.add(key)
            accepted += 1
    labels.save()
    _save(project, auto)
    return {"cutoffs": limits, "auto_accepted": accepted, "taken_back": taken_back}


def mark_checked(project: Project, keys) -> None:
    """A person just decided these boxes: they are no longer ours."""
    auto = auto_decided(project)
    if auto & set(keys):
        _save(project, auto - set(keys))
