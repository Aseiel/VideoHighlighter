"""Person-focused versions of action samples, from the existing cropper.

``modules/crop/actions.py`` finds the people in a clip and writes one cropped
video per person (two or three side by side), or copies the clip when there is
one. An action model trains far better on a crop that holds the action than on
a wide shot where it is a tenth of the frame, which is why the manual process
ran every sample through it. This runs the same cropper, unchanged, over the
project's samples.

Only for ``actions``. The cropper's output names start with the input's stem,
so each focused file is traced back to its sample by name.

**A crop is a sample.** A five-second shot often holds two people doing two
different things, and a sample has one label. So an undecided sample the
cropper found people in is *replaced* by one sample per crop: grouped, sorted,
reviewed and trained on by itself. The whole sample is kept in
``replaced.json`` (never sorted or trained on). A sample someone already
decided keeps its crops attached instead (``focus_paths``), trained under its
label as before, so no decision changes meaning under it.
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import Callable, Optional

from modules.teach.project import ACTIONS, OPEN, PENDING, Project, Sample


def _crop_id(sample_id: str, path: str) -> str:
    """The crop's file name; a shot the cropper kept whole is written under the
    sample's own name, and a crop never shares an id with what it replaced."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem if stem != sample_id else f"{sample_id}_whole"


def focus_project(project: Project, *, cropper: Optional[Callable] = None) -> dict:
    """Run the cropper over every sample it has not seen yet.

    The samples are linked into a scratch folder under their sample ids, so the
    cropper sees example clips (which live wherever the user keeps them) as
    well as cut samples, and nothing twice. ``cropper(input_folder,
    output_folder)`` defaults to the real one.
    """
    if project.task != ACTIONS:
        return {"skipped": "focus is for action projects"}
    out_dir = project.path("focus")
    inbox = project.path("focus", ".inbox")
    shutil.rmtree(inbox, ignore_errors=True)
    os.makedirs(inbox, exist_ok=True)

    todo = [s for s in project.samples if not s.focus_tried and os.path.exists(s.path)]
    for sample in todo:
        link = os.path.join(inbox, sample.id + os.path.splitext(sample.path)[1])
        try:
            os.link(sample.path, link)
        except OSError:
            shutil.copy2(sample.path, link)

    if todo:
        if cropper is None:
            from modules.crop.actions import main as run_cropper

            def cropper(input_folder, output_folder):
                run_cropper(input_folder=input_folder, output_folder=output_folder,
                            debug=False)

        cropper(inbox, out_dir)

    made, crops, replaced = 0, {}, set()
    for sample in todo:
        found = sorted(p for p in glob.glob(os.path.join(out_dir, sample.id + "*"))
                       if os.path.isfile(p))
        sample.focus_tried = True
        made += bool(found)
        if found and sample.verdict == PENDING:
            crops[sample.id] = [
                Sample(id=_crop_id(sample.id, path), source=sample.source,
                       path=path, start=sample.start, duration=sample.duration,
                       focus_tried=True, crop_of=sample.id)
                for path in found]
            replaced.add(sample.id)
        else:
            sample.focus_paths = found

    if replaced:
        kept = []
        for sample in project.samples:
            if sample.id in replaced:
                project.replaced.append(sample)
                kept.extend(crops[sample.id])      # in the sample's place: time order
            else:
                kept.append(sample)
        project.samples = kept
        # Groups nobody has said anything about were groups of whole samples;
        # the crops are grouped afresh. Decided groups keep what was decided.
        project.groups = [g for g in project.groups
                          if g.state != OPEN or not replaced.intersection(g.members)]
    shutil.rmtree(inbox, ignore_errors=True)
    project.save()
    return {"cropped": made, "nothing_found": len(todo) - made,
            "split": len(replaced), "crops": sum(len(c) for c in crops.values()),
            "samples_total": len(project.samples)}
