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
from typing import Callable, Optional

from modules.teach.project import ACTIONS, Project


def focus_project(project: Project, *, cropper: Optional[Callable] = None) -> dict:
    """Run the cropper over samples without a focused version yet.

    ``cropper(input_folder, output_folder)`` defaults to the real one; the
    cropper already skips inputs it has processed, so re-running is cheap.
    """
    if project.task != ACTIONS:
        return {"skipped": "focus is for action projects"}
    samples_dir = project.path("samples")
    out_dir = project.path("focus")
    os.makedirs(out_dir, exist_ok=True)

    if cropper is None:
        from modules.crop.actions import main as run_cropper

        def cropper(input_folder, output_folder):
            run_cropper(input_folder=input_folder, output_folder=output_folder,
                        debug=False)

    cropper(samples_dir, out_dir)

    matched = 0
    for sample in project.samples:
        found = sorted(glob.glob(os.path.join(out_dir, sample.id + "*")))
        found = [p for p in found if os.path.isfile(p)]
        if found:
            sample.focus_paths = found
            matched += 1
    project.save()
    return {"samples_with_focus": matched, "samples_total": len(project.samples)}
