"""Share a taught detector: the project's installed round, ready for model_hub.

model_hub already has the whole sharing flow: the publish wizard, its
checklist, and the rule that a package carries the model and never the
material (CLAUDE.md). A project only has to hand it the right file and a
draft with every technical field filled in. Name, description and category
are the person's to write, and the checklist is theirs to tick.

Detectors only: model_hub can put object detection on a timeline today
(``model_hub.manifest.USABLE_TASKS``), and an action model can't be shared
there yet.
"""
from __future__ import annotations

import os

from modules.teach.project import OBJECTS, Project


class NotShareable(Exception):
    """The message says why and what to do."""


def installed_detector(project: Project) -> str:
    """The ONNX of the project's installed round."""
    from modules.teach.train import installed_detector_files

    if project.task != OBJECTS:
        raise NotShareable("Only object detectors can be shared on the model hub so far.")
    found = installed_detector_files(project)
    if found is None:
        raise NotShareable("Nothing installed yet: train a round first.")
    onnx = os.path.splitext(found["xml"])[0] + ".onnx"
    if not os.path.exists(onnx):
        raise NotShareable(f"The installed model's ONNX file is gone ({onnx}); "
                           "train again to share it.")
    return onnx


def measured(project: Project) -> dict:
    """Frames with accepted boxes, and the videos they came from."""
    from modules.teach.boxes import store

    frames = {(b.video, round(b.time, 3)) for b in store(project).accepted()}
    videos = {s.source for s in project.samples if s.verdict != "pending"
              and not s.id.endswith("__example")}
    return {"trained_frames": len(frames), "videos": len(videos)}


def share_draft(project: Project):
    """``(onnx path, model_hub Manifest draft)`` for the installed round.

    Metrics are counts the project measured for *that* round, recorded when
    it trained: its round number, frames with accepted boxes, and videos.
    They are the only numbers the hub allows, and none of them can carry
    footage.
    """
    from model_hub.package import draft_for_trained_detector

    from modules.teach.train import installed_round

    onnx = installed_detector(project)
    record = installed_round(project) or {}
    counts = measured(project) if "trained_frames" not in record else record
    metrics = {"rounds": int(record.get("round") or len(project.rounds)),
               "train_frames": counts["trained_frames"],
               "videos": counts.get("videos") or None}
    return onnx, draft_for_trained_detector(onnx, metrics=metrics)
