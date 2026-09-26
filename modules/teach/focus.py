"""Person-focused versions of action samples, from the existing cropper.

``modules/crop/actions.py`` finds the people in a clip and writes one cropped
video per person (two or three side by side), or copies the clip when there is
one. An action model trains far better on a crop that holds the action than on
a wide shot where it is a tenth of the frame, which is why the manual process
ran every sample through it. This runs the same cropper, unchanged, over the
project's samples.

Only for ``actions``. The cropper's output names start with the input's stem,
so each focused file is traced back to its sample by name.
"""
from __future__ import annotations

import glob
import os
import shutil
from typing import Callable, Optional

from modules.teach.project import ACTIONS, Project


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

    made = 0
    for sample in todo:
        found = sorted(p for p in glob.glob(os.path.join(out_dir, sample.id + "*"))
                       if os.path.isfile(p))
        sample.focus_paths = found
        sample.focus_tried = True
        made += bool(found)
    shutil.rmtree(inbox, ignore_errors=True)
    project.save()
    return {"cropped": made, "nothing_found": len(todo) - made,
            "samples_total": len(project.samples)}
